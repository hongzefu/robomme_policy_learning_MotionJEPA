# 4task-motion-400ep 构建结果留档（环境 B / AWS 8×A100）

起跑口径见同目录 [`launch.md`](launch.md)。判定行原文汇总在 `records/judgement_lines.txt`；`v1-store/` 不进 git，关键行内联于此。
**日期 2026-09-04；介质 AWS 本地 NVMe RAID（`/dev/md0`）；本机数字与环境 A 不混比。**

## 一、判定行（按执行顺序）

```
STAGE_DONE stage=siglip workers=3 items=400 elapsed=551s                    （GPU2,3,7；HEAD 8093ebd）
FINALIZE_EXIT_CODE=0                        （四 h5 sha256 同源；sidecar=3 覆盖 400、残留 claim=0；抽检 1024/1024 max|diff|=0；provenance 3 节点全体同源）
PACK_DONE=1
VERIFY_PACK=PASS scanned=123044 mismatches=0
norm_stats → v1-store/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json   sha256 750a8e9bd6e1e5a3cf5c294864c44564153309ef92492eb083fa361096d470d2（789 批 × 128，64 s）
STAGE_DONE stage=wan workers=6 items=600 elapsed=1639s                      （第三次，GPU2–7；HEAD c0e13aa；Σ 6,832 窗）
STAGE_DONE stage=encode workers=7 items=600 elapsed=127s                    （GPU0,2–7）
PACK_MOTION_DONE=1                          （rows=6832 = exec 5707 + demo 1125，index_sha256=74185921690cd26c…）
VERIFY_MOTION=PASS scanned=6832 mismatches=0
ORACLE_ENCODER=DONE rows=6832 elapsed=109s
ENCODER_BITEXACT=PASS compared=6832 mismatches=0 order_ok=1 state_sha_ok=1 prov_ok=1 ckpt_ok=1 no_latent_stats_call=1 finite=1   （D3 全量）
A6_MANIFEST=PASS episodes=40 field_mismatches=0 manifest_sha_same=1 motion_index_sha256=74185921690cd26c…   （a6 只比清单前 40 条五字段）
A7_BYTES=PASS segments=600 mismatches=0 table_rows=6832 table_ok=1
A9_INDEXSET=PASS samples=500 mismatches=0
A10_ROWS=PASS rows=6832 exec=5707 demo=1125 formula_or_rowbase_mismatches=0 expect=6832=5707+1125
A8_TABLE_BITEXACT=PASS sampled=128 mismatches=0 rows_total=6832 ckpt=bae960373041629e…
A9_ROWENC=PASS samples=500 rows=4611 unique_windows=3129 mismatches=0
[m1 real] samples=101066 mismatches=0 有效数分布 {k_median 9.0, k_mean 10.31, k_max 34, p25 5 / p75 15 / p90 20 / p95 23 / p99 27, zero_frac 0.0633, fill_rate 0.107}
A19_VALID_DIST=FAIL median=9.0 mean=10.31 max=34 zero_frac=0.0633 fill_rate=0.107     （见三节：判据写死 40 ep 的分布期望）
MOTION_DELIVERY=FAIL samples=101066 mismatches=0 helper_checked=1863                   （核心逐样本对拍 mismatches=0；FAIL 只由 A19 触发）
ORACLE_VAE=DONE windows=6832 frame_mismatches=0 metadata_mismatches=0 elapsed=3566s shards=8   （8 片：722/874/642/1045/721/729/725/1374 窗；shard 1 在 GPU1 与 open 侧评估争用故最慢）
WAN_BITEXACT=PASS compared=6832 frame_mismatches=0 latent_mismatches=0 metadata_mismatches=0 oracle_windows=6832   （D2 全量）
```

## 二、库坐标

| 项 | 值 |
|---|---|
| 清单 | `meta/episode_manifest.json` sha256 `92fa17e97fba9434ee75302de12556319d8ce6d3feeb3adb9a397e830f477223`（400 ep / 123,044 timestep / 101,066 exec；与 40 ep 库的 `oracle/manifest-4task-100ep.json` 逐字节相同；raw_dir 换成环境 A 的 `raw-link-4task` 路径后 == 环境 A 的 `4de8a0fc…`） |
| 输入 | `meta/input_manifest.json`：四 h5 sha256 全值（同 40ep-aws） |
| `source/` | **107 GB**；`meta/stats.json` = `{execution_samples: 101066, total_samples: 123044}`；provenance 3 worker 同源（A100 / jax 0.5.3 / commit `8093ebd`） |
| `framesamp/` | **7.6 GB**，`status=verified`，`num_rows=123044`、`num_exec_samples=101066`、`num_pos_rows=586`；`store_meta.json` sha256 `dffdd47b09aad2812bc46201231e49cd120a4498828d836c9d2695311a815642` |
| `wan-latents/` | 3.8 GB，600 段 / 6,832 窗，每段 `.bin == num_grid × 589,824` |
| `motion-tokens/` | 31 MB，600 段 / 6,832 行 |
| `motion/` | `status=verified`，**6,832 行 = exec 5,707 + demo 1,125**，表 20,987,904 B sha256 `6e70604da518c15647d69b5ecafdd74c16b20dad315290ba2a4f3b105c75e30f`，`motion_index_sha256 74185921690cd26cfd78d309b2d5f89c71c56c0a0ab43cd92b57534d4f8390f6`，LAYOUT `motion-768-grid16-v1` |
| `oracle/wan-mj` | 1.4 GB（8 片 VAE oracle + encoder oracle） |
| norm_stats | `train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json` sha256 `750a8e9b…`（交付件；测试用 `robomme/norm_stats.json` `f332bbd3…` 不动） |

计划 2.6 表估的「4env400ep 全量 26,777 行」是环境 A 私有 4 任务 × 400 ep 录制版的数字；公开版 4 任务 × 100 ep 为 6,832 行，量级一致（每 episode ≈17 行）。

## 三、偏差 / 意外

- **Wan 抽了三次**（留档 `v1-store/attic/400ep-attempt{1,2}-mixedcommit/` 与 `logs/p3-wan{,2,3}*.log`）：
  1. 第一次（`8093ebd`，GPU0–5，1638 s）后 encode 在 `cbf24e9` 跑 → `pack_motion_store.gather_provenance` 报 `跨 worker git_commit 不唯一`（它把 latents 与 tokens 两阶段 worker 指纹合并要求唯一）；
  2. 第二次在 `e94285c` 重抽（GPU2–7）到 240/600 段时，被本人误用 `tmux kill-server` 杀掉（同时杀掉了机器上原有的用户 tmux 会话，见 motion-memory-plan.md「环境 B 复刻」五节），
     清残留 claim / tmp 后 8 卡续抽完成，但 encode 又落在新 commit `c0e13aa` → 再次不唯一；
  3. 第三次在 `c0e13aa` 全量重抽（GPU2–7，1639 s），**打包完成前冻结 HEAD**，encode / pack / oracle 全在 `c0e13aa`。教训：Wan 抽取 → encode → pack 期间不得有任何 commit。
- M1 起早了一次（motion 表未建时 `FileNotFoundError`），重排到 pack 之后。
- `A19_VALID_DIST` 的判据把 40 ep 库的四个分布量（`k_mean 11.46±0.05`、`k_median==11`、`k_max==34`、`zero_frac 0.0555±0.001`）写死在 `motion_gates_model.py` 里，
  400 ep 库的分布（均值 10.31、中位 9.0）与计划 2.6 估的「4env 10.08」同量级，本身合理；M1 的核心判据（101,066 样本逐样本 vs oracle）`mismatches=0`。
  **本轮不改判据**：`MOTION_DELIVERY` 在 400 ep 库记 FAIL（仅 A19 触发），要过需把 A19 期望改为按清单独立重算或按库传参——留给用户拍板。
- `motion_checks.py a6` 只比对清单前 40 条（`episodes=40`），对 400 ep 库是弱检查；A7 / A10 / D3 覆盖全部 600 段 / 6,832 行。
- GPU 分配受并行任务牵制：SigLIP 3 卡（其余卡被 T2 ref / t3 gates 占）、Wan 6 卡（GPU0,1 跑 T2 cand 重跑 / 评估）、encode 7 卡（GPU1 被 open 侧评估占）。

## 五、A100 数字（只记录；与 Ada / turbo 不混比）

- SigLIP 3 卡 551 s（400 ep / 123,044 步 ≈ 74 步/s/卡，与 O1 单卡稳态 80 步/s 一致）；finalize（抽检 1024）4 min；framesamp pack 48 进程 5 s / verify 2 s；Wan 6 卡 1639 s（≈1.43 s/窗/卡，6,832 窗）；
  encode 7 卡 127 s；D2 oracle 8 片墙钟 3566 s（shard 1 在 GPU1 与评估争用；其余片 937–2023 s）；D3 encoder oracle 单卡 109 s；norm_stats 64 s；M1 全库 163 s。
- `dataloader_bench.py --lib $LIB4`（`records/dataloader_bench_400ep_aws.json`；b64、warmup 5、measure 40；与 open 侧评估、D2 oracle 并行）：关闭态 w4 82.0 / w8 82.4 样本/s，
  开启态 w4 79.0 / w8 80.4；每批 pickle 262.3 MB → 287.6 MB；`Pipe` 往返带四键 732.5 ms（392.6 MB/s 单向）。
- `test_padding_dtype.py` 以 `DTYPE_MANIFEST=$LIB4/meta/episode_manifest.json` 跑：14 passed（fixture 用例 `PER_STEP=200` 在 400 ep 库满足）；A22 式单步梯度也用本库 fixture（`docs/training-doc/aws-a22-grad/`）。

## 四、结论

400 ep 完整库交付：`framesamp/`（123,044 行 / 101,066 exec，`status=verified`）+ `motion/`（6,832 行，`status=verified`）绑定清单 `92fa17e9…`；
D3 全量逐位（6,832 行）、D2 全量逐位（6,832 窗，8 片 oracle）、A7 / A8 / A9 / A10 全过、M1 逐样本 101,066 无失配（A19 分布期望需按库参数化）；norm_stats 交付件 `750a8e9b…`。
