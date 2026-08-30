# 附录 D：AGENTS 18 数据逐跳链路图（v5.0 改动前/改动后）

> 落笔规则（计划附录 D）：`[推导]` = 读代码得出，注明稳定锚点，不得冒充实测；`[实测]` =
> 机器记录，注明来源文件与 step。字节量口径：per-batch 张量占用 = `prod(shape) × itemsize`；
> NFS 实际读字节走 `nfs_read.csv` 另一通道，不进本图。
>
> **两图关系**：本次 v5.0 改动不在数据链路上（入口层改动），改动前/改动后第 1–6 跳
> 逐格相同，故合并为一张表 + 观测旁支差异说明；「链路相同」本身是**待证结论、由 4.1 的
> `BATCH_DIGEST_CANONICAL` 与 `INDEX_SEQ` 两分项机器裁决**，不是免证前提。
>
> 勘误（相对计划附录 D 的预设）：`batch_digests.jsonl` 的 per_key 只存哈希、**不含逐键
> shape/dtype**（G3 实物核实），故第 5 跳的 shape/dtype 实测来源改为 C3 驱动日志中
> `train.main` 打印的 `Initialized data loader:`（`training_utils.array_tree_to_info(batch)`）
> 一节；内容一致性仍由 `batch_digests.jsonl` 裁决，两来源并注。

## 数据链路（第 1–6 跳）

| # | 跳 | 锚点 | shape/dtype/字节量 | 这一跳有没有改数？ |
|---|---|---|---|---|
| 1 | 全局原始 H5 → turbo 暂存副本 | `/data/hongzefu` 原件；AGENTS 15 | [推导] 建库输入，非训练期路径 | 否——逐文件 sha256 同源核对（AGENTS 15） |
| 2 | packed store 落盘（建库一次性） | `src/mme_vla_suite/datastore/framesamp_store.py`（`read_image_rows`/`pos_rows`/`state_rows`）+ `data/<idx>.pkl` 逐样本包 | [推导] 行式三表：token_emb 每帧 `(16,2048)` bf16、pos 每帧 `(16,768)` f32、state 每帧 `(8,)` f32；训练期只读，锚 `meta/store_meta.json` sha256=`3990165c…`（本 run 前置门已人工比对命中） | 建库时一次性转换（G2 已位级裁决），训练期只读不改 |
| 3 | `FrameSampDataset.__getitem__` | `src/mme_vla_suite/training/framesamp_dataset.py::__getitem__`（身份互校 raise、`even_sampling_indices`、`_pad`、reshape/repeat 注释「与旧路径逐字对齐」） | [推导] 单样本：`static_image_emb` `(512,2048)` bf16（=32 帧×16 token reshape，C-order 字节不变）、`static_pos_emb` `(512,768)` f32、`static_state_emb` `(512,8)` f32、`static_mask` `(512,)`；`actions` 截断到 `action_horizon`；pkl 其余键透传 | **是（设计语义内）**：`_pad` 不足 32 帧补零并给 mask；`_normalize_state` 状态归一化；reshape/repeat 仅重排字节。身份互校失败即 raise（不静默换样本） |
| 4 | transforms | `openpi/training/data_loader.py::transform_dataset` → `TransformedDataset`（norm_stats 来自 `data_config.norm_stats`，缺失即 raise） | [推导] 键集不变，逐键做 data transforms + norm_stats 归一化；norm_stats 文件 sha256=`709f22ff…`（与 G0b 指纹同源，本轮 P3 已核对） | **是（设计语义内）**：按 norm_stats 归一化（缺 stats fail-loud，不静默跳过） |
| 5 | collate → `TorchDataLoader.__iter__` 输出 host batch | `openpi/training/data_loader.py::TorchDataLoader`（G 链量具唯一输入观测点即在此包装） | **[实测]** 待 C3 补：shape/dtype 取 C3 驱动日志 `Initialized data loader:` 段（`array_tree_to_info`）；内容一致性取 `records/batch_digests.jsonl`（首行 `n_keys=12`，逐步 raw+canonical 双口径） | collate 堆叠 batch 维（重排不改值）；raw 口径已知 2 键（`static_image_emb`/`static_pos_emb`）dtype 统一差异（G1 位级裁决过的预期），canonical 口径下为「否」 |
| 6 | sharding → 进模型 | `train.py::main` 的 `jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))` | [推导] 与第 5 跳同 shape/dtype，仅按 batch 维分到 2 卡 | 否——只分片不改值 |

## 观测旁支（改动前后唯一差异处）

```
              ┌─（改动前）bench/prod 的 _WandbProxy（G 链量具装载，import 路径）
train.main ── wandb.log(reduced_info) 只读转发、float.hex() 取自 reduced_info
              └─（改动后新增）train.py 内置 _MetricsProxy——仅 __main__ 路径且设
                 TRAIN_RECORD_DIR 时装载；bench 路径（本 run）结构性不装载
```

- 旁支**只读转发不改数**：两代理的 `log()` 均为「记 jsonl 后原样转发真 wandb」，行 schema
  逐字段相同（C2.5 smoke 已验 analyzer 可读）。
- 本 run（C3）经 `g0/bench_train_steps.py` import 启动，`__main__` 不执行 → 内置记录器
  构造性不装载，数据链路与观测旁支均与 G3 完全同构。

## 实测数字回填（C3 收官，2026-08-30）

**第 5 跳逐键 shape/dtype**（`[实测]` 来源：`v1-store/logs/v1-singlerun-g0-driver.log` 的
`Initialized data loader:` 段，`train.main` 首 batch `array_tree_to_info` 打印；字节量 =
`prod(shape) × itemsize` 换算）：

| 键 | shape@dtype | per-batch 字节量（b8） |
|---|---|---|
| `[0].images['base_0_rgb']` / `['left_wrist_0_rgb']` | `(8,224,224,3)@float32` ×2 | 各 4,816,896 B ≈ 4.59 MiB |
| `[0].image_masks[...]` ×2 | `(8,)@bool` ×2 | 各 8 B |
| `[0].state` | `(8,32)@float32` | 1,024 B |
| `[0].tokenized_prompt` / `_mask` | `(8,64)@int32` / `@bool` | 2,048 B / 512 B |
| `[0].static_image_emb` | `(8,512,2048)@bfloat16` | 16,777,216 B = 16 MiB |
| `[0].static_mask` | `(8,512)@bool` | 4,096 B |
| `[0].static_pos_emb` | `(8,512,768)@float32` | 12,582,912 B = 12 MiB |
| `[0].static_state_emb` | `(8,512,8)@float32` | 131,072 B = 128 KiB |
| `[1]`（actions） | `(8,20,32)@float32` | 20,480 B = 20 KiB |

合计 ≈ 38.9 MiB / batch；键数 = 12（11 个观测键 + actions），与 `batch_digests.jsonl`
首行 `n_keys=12` 一致。

**内容一致性机器裁决**（`[实测]` 来源：`records/compare_vs_g0_r1.txt`）：
`BATCH_DIGEST_CANONICAL rows=14 mismatch=0`、`CANON_CHECK=PASS steps=14`、
`INDEX_SEQ=PASS n=8072`；raw `BATCH_DIGEST rows=14 mismatch=4`（`static_image_emb`/
`static_pos_emb` dtype 统一已知预期，first_bad_step=100 bad_keys=2，与 G2/G3 逐字吻合）。
**「改动前后数据链路第 1–6 跳逐格相同」的待证结论已由上述两分项裁决成立。**
