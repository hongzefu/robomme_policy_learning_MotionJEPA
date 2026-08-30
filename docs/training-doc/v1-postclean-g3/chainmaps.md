# v4 重构前后链路图（AGENTS 18；随 v1-postclean-g3 留档）

## 图 A-before：重构前训练链（25 跳，锚 732fae3b）

| # | 跳 | 形状/dtype/字节量 | 这一跳改数吗 |
|---|---|---|---|
| A1 | `_resolve_backend`（MMEVLA_DATA_BACKEND 三态分派） | — | 否（纯分派） |
| A2 | legacy 旁路：`RoboMMEDataset`（散 npy + MemoryBuffer） | — | 是（legacy 侧独立装配） |
| A3 | `_create_framesamp_dataset` 闸（require_no_pack_lock/StoreMeta/require_verified/subset 闸） | — | 否 |
| A4 | 清单查表 (g, step)（O(1) 数组，含 exec_start_idx 换算） | int 标量 | 否 |
| A5 | `pickle.load` 源库 data/{idx}.pkl + 身份互校 | dict（image u8 256², state f32(8), actions f32(20,8), prompt str…） | 否 |
| A6 | `actions[:action_horizon]` 截断 | (20,8) f32 | 否（切片） |
| A7 | `even_sampling_indices(step, 32)` 选帧 | list[int] ≤32 | 否 |
| A8 | `FrameSampStore` gather（fd 游程合并 pread） | (n,16,2048) bf16 + (n,16,768) f32 + (n,8) f32 | 否（字节搬运） |
| A9 | `_pad` 预分配 + 填充区清零 | (32,16,2048) bf16 / (32,16,768) f32 / (32,8) f32 / mask (32,) bool | 否（补零） |
| A10 | reshape+repeat 展平到 token 维 | (512,2048) bf16 / (512,768) f32 / (512,8) f32 / (512,) bool | 否（视图/复制） |
| A11 | `_normalize_state`（q01/q99 f64） | state (8,)→f64、static_state_emb (512,8)→f64 | 是（归一化） |
| A12 | `_NONE_KEYS` 补 None（11 键） | None | 否（None 非叶子，collate 剪掉，n_keys=12 实证） |
| A13 | RepackTransform（15 键白名单改名） | — | 否 |
| A14 | `RoboMMEInputs`（image/state/actions/prompt 构造 + 10 个 data.get） | image u8 | 否 |
| A15 | `DeltaActions(make_bool_mask(7,-1))` | (20,8) | 是（差分） |
| A16 | `InjectDefaultPrompt` / `ResizeImages(224,224)` | image (224,224,3) u8 | 是（resize） |
| A17 | `TokenizePromptWithSymbolicMemory`（symbolic 死分支 + 两个无默认 pop） | tokens (64,) i32 + mask | 是（tokenize） |
| A18 | `PadStatesAndActions(32)` | state/actions padding 到 32 维 | 否（补零） |
| A19 | `Normalize`（norm_stats、quantile） | actions | 是 |
| A20 | `_collate_fn`（np.stack 逐键） | batch 维 b8；per-batch ≈30.9 MiB | 否 |
| A21 | `TorchDataLoader`（spawn worker、seed、sharding） | — | 否 |
| A22 | `HistAugObservation.from_dict`（含 recur_*/symbolic_* 字段位） | — | 否 |
| A23 | `preprocess_observation`（train=True 增广路径，确定性档下无 rng 增广） | images f32 [-1,1] | 是 |
| A24 | 模型前向（embed_memory 含 stats 链 / embed_prefix 5 元组 / recurrent-symbolic 死分支 / compute_loss 双返回） | loss (b,ah) f32 | 是 |
| A25 | `train_step`（fold_in/DiffState/value_and_grad has_aux=True/tx.update/EMA/info 五标量 + stats 三返回） | 标量 f32 | 是 |

## 图 A-after：重构后训练链（V4.6 tip）

与 A-before 同构，差异仅为旁路删除与形态收敛（数值路径零变更，G3 位级实证）：

- A1/A2 删除：`create_data_loader` 无条件 `_create_framesamp_dataset`（V4.1）。
- A12 `_NONE_KEYS` 11→5 键（V4.2；None 非叶子零贡献，n_keys=12 前后实证相同）。
- A13 Repack 15→9 键、A14 data.get 10→4（V4.2；被删键恒 None/被 pop）。
- A17 收敛为 `TokenizePromptWithState`（V4.2；保留支路逐字不变，subgoal 死分支删）。
- A22 `HistAugObservation` 删 recur_*4+symbolic_*2 字段（V4.3；叶子集不变）。
- A24 模型删 recurrent/symbolic 分支与 stats 返回链（V4.3；`stats≡None` 非叶子，
  expert/modulation 构图与 lazy_init 一字不动）。
- A25 `train_step` 三返回改二返回、has_aux 删除（V4.3；N3 GRAD_FIXTURE 逐位证伪）。
- A7 `even_sampling_indices` 迁 `shared.sampling`（V4.4；函数体一字不动）。
- 其余 A3–A11、A15–A16、A18–A21、A23 逐字保留。

## 图 B：在线评估链（15 跳）

- B0 pos 表：`PosEmb3D(768)(arange(4096), grid)`——旧三档 8x8/4x4/2x2 共 ≈1.01 GiB
  → 新仅 4x4 一档 ≈192 MiB；4x4 档逐位不变（N4 POS_TABLE=PASS rows=4096）。
- B1 client `pack_buffer`（images u8 (t,1,h,w,3) + states f32 (t,8)）——不变。
- B2 归一化 `/255*2-1` → B3 `resize_with_pad(224,224)` → B4 注入的模型
  `vision_enc_fn`（jit SigLIP）——逐字保留。
- B5 device_get（旧逐步重复 get → 新循环外一次；字节相同）。
- B6 池化 `pool_tokens_to_size`：旧 ×3 档 → 新 ×1 档（4x4 档逐位不变，
  N4 ENC_LAYER=PASS steps=13 keys=3）。
- B7 存 `_history_feats[step]`——旧 8 键（含 image_pixels u8 ≈780KB/步）→ 新 3 键
  （≈112KB/步）；装配只读三键（不可观测性论证 + N4 实证）。
- B8 重复/越界 step：旧 assert/静默空切片 → 新显式 raise（OOB_PROBE=PASS）。
- B9 token_drop 打分堆整删（唯一消费 image_pixels 的活路径）。
- B10 `even_sampling_indices` → B11 gather → B12 `_prepare_frame_sampling`
  （right_padding_token_emb 复用，四元组 (512,2048) bf16 / (512,768) f32 /
  (512,8) f32 / (512,) bool）——逐字同式（N4 ASSEMBLY=PASS steps=13）。
- B13 `_normalize_state`（policy 侧，f64 stats）→ B14 client subgoal 注入删除
  （V4.5，无对拍，盲区第 2 条）→ B15 `infer`→`sample_actions`（本轮不改，
  盲区第 3 条）。

## 图 C：建库链（7 跳）

- C1 h5 读取（build_robomme_dataset/build_shard）——import 边改指
  `dataset_builder.mem_buffer`（V4.0），计算零改动。
- C2 `DatasetProcessor.__init__`——加 `--force` 闸（rmtree 前显式 raise；
  rmtree 之后计算路径一行不动）。
- C3 MemoryBuffer.add_buffer（SigLIP 编码 + 三档池化 + pos 表切片）——冻结副本，
  COPY_DIFF 3 行 import 差 + 4 条 sha256 哨兵。
- C4 逐步 `token_emb_{step}.npy` 落盘（7 键 602,951 B）——逐位不变。
- C5 kept_indices 原子写——不变。
- C6 finalize/manifest/stats——import 边 + provenance "produced_by" 字符串改新路径
  （元数据字段，不在 kept_indices/token_emb bitexact 域）。
- C7 反向 import 护栏：IMPORT_ISOLATION 双向 0 泄漏（白名单仅 compare_online_memory）。
