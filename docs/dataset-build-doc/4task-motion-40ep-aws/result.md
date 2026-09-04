# 4task-motion-40ep（环境 B / AWS 复刻）构建结果留档

起跑口径见同目录 [`launch.md`](launch.md)；环境 A 原始结果见 [`../4task-motion-40ep/result.md`](../4task-motion-40ep/result.md)。
判定行原文汇总在 `records/judgement_lines.txt`；`v1-store/` 不进 git，关键行内联于此。**日期 2026-09-04，HEAD `8093ebd`，8×A100-SXM4-80GB，介质 AWS 本地 NVMe RAID（`/dev/md0`）。**
本机数字与环境 A（RTX 6000 Ada / turbo NFS）**不得混比**。

## 一、原始 H5 同源（A5 的替代证据）

公开集 `Yinpei/robomme_data_h5` 四个目标任务解压后（`sha256sum`，4 路并行 194 s）：

```
6b100414429e3417f2afd600ae708406bc20b1a37ef92734ff593af6bdb70575  record_dataset_ButtonUnmask.h5       17,751,535,444 B
7c0441210bb1ec63aa60cfc30c5080a5f09c54f02bd0004714eba120df089274  record_dataset_ButtonUnmaskSwap.h5   26,627,642,804 B
05a653a8f8232882f82c84057f328e045f4a875ff5cfcf068738c429c6081427  record_dataset_VideoUnmask.h5        14,445,442,112 B
4e83aca373b2adb469cf78d338223e41559fc6ad19d435de5c88d99d2fe49a7e  record_dataset_VideoUnmaskSwap.h5    23,205,606,404 B
```

环境 A 留档（`../4task-motion-40ep/launch.md` S1 节）只记了 8 位前缀 `6b100414… / 7c044121… / 05a653a8… / 4e83aca3…` 与字节数——四个前缀与四个字节数**全部命中**，
`finalize_checks.py hash-inputs` 再算一遍写入 `meta/input_manifest.json`（`records/input_manifest.json`，全值）。结论：公开版与环境 A 本机原件同源，
环境 A 的 A5「原始帧同源」结论可传递到本库；40 ep 的全部锚点数字（13,756 / 11,530 / 772 / 658 / 114）无需重估。

## 二、清单与环境 A 逐字相同（sha 差异只来自 raw_dir 绝对路径）

| 清单 | 本机 sha256 | 把 `raw_dir` 字段替换为环境 A 路径后重算 | 环境 A 留档值 |
|---|---|---|---|
| `meta/episode_manifest.json`（40 ep / 13,756 / 11,530） | `d7cfb137b6ba01c42894e2d6d421a8c3f87dc1afeef1ac650563609bd7501d05` | `/data/hongzefu/robomme_data_h5` → `fee2777f58bf0e83b20fc95fff98a6b5871bfb2de10f967da39aecfccba892b6` | `fee2777f…` ✓ |
| `oracle/manifest-4task-100ep.json`（400 ep / 123,044 / 101,066） | `92fa17e97fba9434ee75302de12556319d8ce6d3feeb3adb9a397e830f477223` | `/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/raw-link-4task` → `4de8a0fc…` | `4de8a0fc…` ✓ |

`scan_manifest.manifest_sha256` 覆盖 `raw_dir` 字段（绝对路径），故两环境的清单 sha 天然不同；`episodes / totals / canonical_order` 逐字节相同。
下游一切「manifest_sha256」锚点（framesamp / motion store_meta、oracle 报告、a6）在本机以 `d7cfb137…` 为准。

## 三、判定行（全部 PASS；顺序按执行时间）

```
PROBE_BENCH=PASS windows=20 ms_per_window=1413.5 peak_mib=1714 rerun_bitwise=20/20                       （A2，GPU2，与 A3/A4/O1/O2/hash 并发）
A3_CROSSGPU=PASS compared=64 latent_bitwise=64 token_bitwise=64 max_abs_diff=0.000e+00                  （GPU0 vs GPU1）
A4_DUALVENV=PASS compared=64 latent_bitwise=64 token_bitwise=64 max_abs_diff=0.000e+00                  （MotionJEPA .venv 原版 vs wan venv 复制件，GPU1）
SHARD_DONE shard=0 episodes=40 skipped=0 steps=13756 elapsed=303.7s rate=45.291 step/s steady_steps=13465 rate_steady=80.292 step/s   （O1，GPU7）
Time taken: 5.44 minutes                                                                                  （O2，GPU6）
STAGE_DONE stage=siglip workers=6 items=40 elapsed=88s
FINALIZE_EXIT_CODE=0                                （四 h5 sha256 同源；sidecar=6 覆盖 40、残留 claim=0；抽检 256/256 max|diff|=0）
PACK_DONE=1
VERIFY_PACK=PASS scanned=13756 mismatches=0
COMPARE_RESULT=bitexact PASS                        （D1 O1：build_shard 单卡 vs 6 worker 库，--all_pkl）
COMPARE_RESULT=bitexact PASS                        （D1 O2：未改动 build_dataset.py，listdir 序映射交叉验证通过）
STAGE_DONE stage=wan workers=6 items=60 elapsed=199s                （≈1.43 s/窗/卡）
STAGE_DONE stage=encode workers=8 items=60 elapsed=21s
PACK_MOTION_DONE=1                                  （rows=772 = exec 658 + demo 114，index_sha256=4183a6e78297476a…）
VERIFY_MOTION=PASS scanned=772 mismatches=0
ORACLE_VAE=DONE windows=772 frame_mismatches=0 metadata_mismatches=0 elapsed=231s shards=8      （8 片：64/103/104/159/64/88/74/105 窗，aggregate 合成）
ORACLE_ENCODER=DONE rows=772 elapsed=10s
WAN_BITEXACT=PASS compared=772 frame_mismatches=0 latent_mismatches=0 metadata_mismatches=0 oracle_windows=772      （D2）
ENCODER_BITEXACT=PASS compared=772 mismatches=0 order_ok=1 state_sha_ok=1 prov_ok=1 ckpt_ok=1 no_latent_stats_call=1 finite=1   （D3）
A6_MANIFEST=PASS episodes=40 field_mismatches=0 manifest_sha_same=1 motion_index_sha256=4183a6e78297476a…
A7_BYTES=PASS segments=60 mismatches=0 table_rows=772 table_ok=1
A8_TABLE_BITEXACT=PASS sampled=128 mismatches=0 rows_total=772 ckpt=bae960373041629e…
A9_INDEXSET=PASS samples=500 mismatches=0
A9_ROWENC=PASS samples=500 rows=5071 unique_windows=687 mismatches=0
A10_ROWS=PASS rows=772 exec=658 demo=114 formula_or_rowbase_mismatches=0 expect=772=658+114
```

## 四、库坐标

| 项 | 值 |
|---|---|
| 清单 | `meta/episode_manifest.json` sha256 `d7cfb137…`（40 ep / 13,756 / 11,530；内容 == 环境 A `fee2777f…`，见二节） |
| 输入 | `meta/input_manifest.json`：四 h5 sha256 全值（一节） |
| `source/` | 13 GB；`meta/stats.json` = `{execution_samples: 11530, total_samples: 13756}`；`meta/provenance.json` 跨 6 worker 指纹唯一（jax 0.5.3、A100-SXM4-80GB、commit `8093ebd`） |
| `framesamp/` | 888 MB，`status=verified`，`num_rows=13756`、`num_exec_samples=11530`、`num_pos_rows=586`、pos 表 28,803,072 B、state 表 440,192 B（三项与环境 A 同值）；`store_meta.json` sha256 `56c4faf71213228624618874ee39a7905d5827b8746d73dabde9bf86a34f29f2` |
| `wan-latents/` | 436 MB，60 段 / 772 窗，每段 `.bin == num_grid × 589,824`，`metadata.json` schema 2 |
| `motion-tokens/` | 3.4 MB，60 段 / 772 行 |
| `motion/` | `status=verified`，772 行 = exec 658 + demo 114，表 2,371,584 B sha256 `d374aff255688a699f281d7a821d68cbcbfdd9d535c146f707229f9ab8f32bb3`，`motion_index_sha256 4183a6e78297476a313f116d25fef6a6153fb5f0cd875d41014a3c2c9bea4f91`，LAYOUT `motion-768-grid16-v1` |
| `oracle/` | 25 GB（两个 SigLIP oracle 库各 13 GB + `wan-mj` + `probe` + 100 ep 清单） |

**跨环境字节说明**：motion 表 sha（本机 `d374aff2…` vs 环境 A `708129f5…`）与 `motion_index_sha256`（`4183a6e7…` vs `313d4549…`）不同——前者是 A100 与 Ada 的 VAE / encoder
卷积与 bf16 算法实现差异（同架构跨卡逐位、跨架构不逐位，与环境 A 的 A11 crossarch 结论同性质），后者含 `manifest_sha256`（二节）；两者都**不是**链路差异：D2 / D3 在本机
对本机原版 oracle 逐位，A6–A10 全过。framesamp 的 pos / state 表字节数与 `num_pos_rows` 与环境 A 同值。

## 五、A100 实测数字（只记录；与 Ada 数字不混比）

- A2：合计 **1413.5 ms/窗**（VAE 段 mean 1403.9 / min 1401.5 / max 1409.1；encoder 段 9.6）；`max_memory_allocated` 1,714 MiB；同设置重跑 20/20 逐位。
  A2 漂移（只记录，生产不启用）：VAE bf16 autocast latent min_cos 0.99999273 / max rel 3.055e-2，token min_cos 0.99976308 / max|Δ| 9.375e-2，逐位 0/20；TF32 档见 `records/a2-bench-gpu2.json`。
  注：A2 与 A3（两卡）、A4、O1、O2、hash-inputs 同时在跑，数字偏慢；A3 单卡 64 窗含落盘 1453 / 1459 ms/窗。
- Wan 抽取 6 卡 199 s（≈1.43 s/窗/卡）；D2 oracle 8 片墙钟 231 s（最长片 159 窗）；encode 8 卡 21 s；O1 单卡 303.7 s（稳态 80.3 step/s）；O2 5.44 min；SigLIP 6 卡 88 s。
- `dataloader_bench.py`（`records/dataloader_bench_aws.json`；b64、warmup 5、measure 40；**与 P5 / t3common / T3 两条训练 run 并行**，绝对值只作参考）：
  关闭态 w4 75.8 样本/s、w8 85.0；开启态 w4 77.5、w8 74.0；每批 pickle 262.3 MB → 287.6 MB；`Pipe` 往返带四键 722.1 ms（398 MB/s 单向）/ 不带 648.8 ms。

## 六、与环境 A 留档的偏差 / 意外

- **encode 阶段首起 FAIL**：`GPU 0 空闲显存 19834 MiB < 要求 20000 MiB`——此时 `finalize_checks.py check` 的 JAX 进程正占着 GPU0（默认预分配），`run_local.py --require-free-mib 20000` 起跑预检拒绝；
  finalize 结束后重起即过（`p1-encode.attempt1-fail.log` 保留）。教训：finalize 的抽检要么显式给它一张不参与后续阶段的卡，要么串行。
- `extra_checks.py` 的 A9_ROWENC 子命令名是 `a9enc`（不是留档判定行前缀 `a9rowenc`），第一次调用报 `invalid choice`，改名重跑。
- `motion_checks.py a6 --manifest-400ep` 用本机新建的 100 ep 清单（`92fa17e9…`），不再依赖环境 A 的 `4de8a0fc…` 文件；a6 只比五字段与 `manifest_sha_same`，不受 raw_dir 影响。
- 不可做项：A5 / A11 / A12 / `crosscheck.py --vae_check`（对照物在 `/data/hongzefu` 或 turbo，本机不存在）；A5 以一节的 sha256 全等替代。

## 七、结论

40 ep 库在环境 B 交付：`framesamp/`（`status=verified`）与 `motion/`（`status=verified`）绑定同一清单 `d7cfb137…`（内容 == 环境 A `fee2777f…`）；
D1（两条 SigLIP oracle）、D2（原版 `encode_chunk` 772 窗 8 片）、D3（原版 `motion_token` 772 行）全部逐位；A6–A10 全过；A3 证 A100 跨卡逐位。
本库随后用于 `docs/training-doc/aws-*/` 八个 100 步 run 与全部 M / P / T 闸门（见各 run 留档与 `motion-memory-plan.md`「环境 B 复刻」节）。
