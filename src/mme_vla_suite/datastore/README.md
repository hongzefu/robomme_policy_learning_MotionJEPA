# datastore：framesamp packed 特征库 + motion 离线表 格式层

对应 `v2-framesamp-restructure-plan.md`（A.1 布局契约、B.2 Store 行为）与 `motion-memory-plan.md`（第二部分一节 1.1 motion 表契约）。本包是格式层
**唯一实现**：打包工具（`scripts/dataset/pack_framesamp_store.py`、`scripts/dataset/pack_motion_store.py`）、`FrameSampDataset`、对拍工具
一律从这里 import，绝不复制；本包不 import 任何 training/model 模块（单向依赖，B.1）。

| 文件 | 内容 |
|---|---|
| `manifest.py` | `load_manifest`（sha256 fail-loud）与 `manifest_sha256`（与建库侧 `scan_manifest.py` 同口径的独立实现——scripts 目录名含连字符无法跨目录 import） |
| `motion_store.py` | motion 离线表布局常量（`motion-768-grid16-v1`、stride 16 / 窗 33 / 段起点网格 / 不截尾 / frame_size 256）、网格公式（`seg_num_chunks` / `seg_num_grid` / `visible_motion_rows` 三侧同式）、`build_index_entries` / `parse_index`（motion_index.json 行序契约）、`MotionMeta`（含 index sha256 现场重算）、`run_fast_checks` / `run_full_checks` / `require_verified` / `require_no_pack_lock` / `check_same_source`（双 store 同源闸）、只读整表 `MotionStore` |
| `framesamp_store.py` | 布局常量（`framesamp-4x4-v1`、三张表形制、源 npy 偏移常量）、`row_of()` 行号公式（写读共用）、`build_exec_lookup()`（执行样本 idx → (g,t) 查表数组，必须带 `exec_start_idx`）、`StoreMeta`（store_meta.json 结构校验）、`run_fast_checks` / `run_full_checks` 两档校验、`require_verified` / `require_no_pack_lock`（packed 分派层守卫）、只读 `FrameSampStore` |

## 布局速览

```
<store 根>/
├── meta/store_meta.json        唯一契约（两阶段写：pack→"packed"、verify→"verified"）
├── meta/pack_progress.jsonl    断点续跑记录（打包父进程单写）
├── meta/row_digests.blake2b.bin  逐行 blake2b-128（verify 时产出，image‖pos‖state）
├── image_emb_4x4/part_XXX.bf16.bin  (rows,16,2048) bf16 裸字节，按 episode 边界切分
├── pos_emb_4x4.f32.bin         (num_pos_rows,16,768) f32（pos 是 t 的纯函数，只存一份）
└── state_emb.f32.bin           (num_rows,8) f32
```

行号 = `row_of(total_sample_offset[g], t)`，t 为全 timestep 域帧号（含 demo 前缀）。
禁 `.npy` 容器：`np.save` 对 ml_dtypes bf16 写 `V2` descr、`np.load` 丢类型（A.1 定论），
一律裸 `.bin` + meta 声明 dtype。

## FrameSampStore 契约（B.2 摘要）

- 构造即打开全部 part fd（`O_RDONLY|O_CLOEXEC`）+ 两张小表 `np.fromfile` 整表读入
  （进程内副本、非映射——热路径不留任何 NFS mmap），记录 `owner_pid`；
- `read_image_rows`：先对全部行发 `posix_fadvise(WILLNEED)`（不可用则打一次 WARNING
  后永久跳过），再按连续行游程合并 `os.preadv` 直读进预分配数组；短读循环补齐，
  EOF/越界立即 raise；
- **禁止 pickle**（`__reduce__` 直接 raise）：跨进程懒构造与 pid 校验由
  `FrameSampDataset` 负责（S3）；
- 不看 `meta.status` / `pack.lock`——那是 `create_data_loader` 分派层的闸
  （`require_verified` / `require_no_pack_lock`，S3 接线）；verify 子命令持锁期间
  仍需经 Store 读库；
- fast 档（每进程懒构造时）：layout / 清单指纹重算 / parts 连续覆盖 / 逐 part st_size /
  抽 1 part 首尾 1 MiB digest / 抽 1 条源库抽样指纹；full 档（全部 sha256，能抓同尺寸
  中部翻转）**禁止在性能 allocation 内执行**（会预热 page cache，B.2）。

守卫测试：`scripts/training/tests/test_pack_guards.py`（Store 组 G1/G4/G5/G7/
G11/G12/G14 + Dataset 组 G2/G3/G6a/G8/G9/G10/G13 与分派闸）；spawn 生命周期验收另见
`scripts/training/tests/spawn_matrix.py`。消费侧装配层见
`src/mme_vla_suite/training/framesamp_dataset.py`；packed 是唯一训练数据路径
（commitV4.1 起，backend 三态与 `_resolve_backend` 已删除），构造入口见
`training/dataloader.py` 的 `_create_framesamp_dataset`。

## history_config schema（R6 补偿：`models/config/base.yaml` 已删，此处为唯一 schema 文档）

训练唯一配置 `models/config/robomme/perceptual-framesamp-context.yaml`（文件本体
一字不动——`FrameSampDataset` 的形制断言与 `_EXPECTED_HISTORY_CONFIG` 依赖其字段与
文件名）。各键含义（modul/expert 变体仅 `integration_type` 不同）：

- `representation_type: perceptual` — 记忆表征类型；v4 重构后训练/在线仅存 perceptual
  一种（recurrent/symbolic 已删，见 git 历史 commitV4.2/V4.3）。
- `integration_type: context` — 记忆接入方式，∈ {context, modulation, expert}；
  训练链固定 context（`FrameSampDataset` 形制断言），modul/expert 仅在线评估/部署。
- `perceptual_memory.type: frame_sampling` — perceptual 记忆的采样策略；v4 后仅存
  frame_sampling（token_dropping 已删）。
- `budget` — 记忆 token 预算（frame_sampling 下 = 采样帧数 × token_per_image）。
- `token_per_image` — 每帧图像池化后 token 数（4x4 池化 → 16）。
- `num_views` — 视角数（本数据集为 1）。
- `streaming_obs_horizon` — 流式观测窗口长度；train.py 断言其为 16 且
  `action_horizon==20`。
- `pool_type: mean` — 建库侧特征池化方式。
- `memory_feature.{img,pos,state}.input_dim` — 三路特征的输入维度（SigLIP 图像
  emb / 3D 位置编码 / 关节状态）；`img.net: identity` 表示图像特征不再过投影网络；
  `pos.hidden_dim`、`state.hidden_dim` 为 `FeatureEncoder` 两路投影的输出维度。
- `memory_token_dim` — `FeatureEncoder.encoder_static` 输出的记忆 token 维度
  （= 主干 LLM width）。
- `use_pos_emb` / `use_state_emb` — 是否拼接 pos/state 投影（影响
  `FeatureEncoder` 的参数树与 RNG 消耗序，禁改）。
