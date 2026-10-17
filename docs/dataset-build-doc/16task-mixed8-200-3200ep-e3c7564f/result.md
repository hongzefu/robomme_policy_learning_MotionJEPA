# mixed8 十六任务 3200 集 VLA 新库建成、全部对拍与两次上传通过

本文件是建库档案的第二段，在 VLA 新库上传 PASS、按口径 24 删除本地中间件之后补写。第一段 [launch.md](launch.md) 记起跑时的版本、命令与配置，本文件不重复。叙述中的时刻一律为美东；判定行与日志原文里的时间戳（多为 UTC）原样保留，不改写。

## 结论与指标速览

按 [mixed8 VLA 计划](/scratch/hongze/MotionJEPA-backup/1016-mixed8-vla-plan.html)（`dev/mixed8-vla-plan @ 952b8db`），从公开 bucket `HongzeFu/robomme-16task-mixed8-200-h5-20261014-v1` 的 100 个包全新构建 `16task-mixed8-200-3200ep-e3c7564f`：16 任务 × 200 集共 3200 条 primary，2,145,548 个源时步，1,383,846 个执行样本。launch.md 验收表里的每一项判定行都已 PASS，没有任何一项以「未执行」充数。两次上传都以 `RESULT=PASS`、读回 `SHA256_PRE_POST_DIFF=0`、`BUCKET_PUBLIC=True` 收尾。两条 80k 正式训练已在本库上起跑。

| 项目 | 实测 |
|---|---|
| episode / 源时步 / 执行样本 | 3200 / 2,145,548 / 1,383,846 |
| 合并 H5（16 个，input_manifest 合计） | 1,424,043,814,720 B |
| canonical manifest sha256 | `d6a3ddb2b579b5cf50cd66fbefedd36478766e2b4cc07833a7efae29769742ad` |
| 源 MANIFEST sha256 | `e3c7564f787781910f741d57b487d45565defbe8d3c2c17af9233c651093de11` |
| norm_stats sha256 | `36ee9acdea4c8383b50c7b07b45a740efb3fa073f6e60455977d5a6fb3204394`（64 个值全部有限） |
| 窗口预算 | `max_windows=123 budget=160 headroom=37`，与建库前实算相同 |
| motion 表 | 128,342 行（exec 81,611 + demo 46,731），4995 段，其中补帧段 1800 |
| HEAD 冻结窗口 | `6ebeab5632340328a3a62db836d1ab3b004eed57`，SigLIP 起跑到 motion verify 之间无提交 |
| latent 库 | `HongzeFu/robomme-16task-mixed8-200-motion-20261014-v1`：5120 文件，1,139,255,338,776 B，公开 |
| VLA 新库 | `HongzeFu/robomme-16task-mixed8-200-trainset-20261016-v1`：411 个 tar 分片加 124 个平文件，1405.0 GB，公开 |
| 两条 20 步短测 | 都 PASS，单卡显存峰值 39.3 / 38.3 GiB，限值 72 GiB |
| 端到端墙钟 | 美东 2026-10-16 08:06（A3 与 B1 判据起跑）到 2026-10-17 04:57（VLA 新库上传 `EXIT_CODE=0`），约 20 h 51 min |

实施中有 4 处偏离原计划，都已登记在计划进展登记第 21、22 条（P8、P9）：SigLIP 改 32 worker；oracle vae 首跑因 `HF_HOME` 被覆盖而失败，之后补跑；A11 的 `--expect-episodes` 口径错误，按用户「a」改为 1800 后重跑；wandb 不上传。前三项详见「计划外事件与处置」，wandb 一项见口径 44。

## 各阶段实测

### 起止时刻与耗时

起止时刻取各日志文件在 XFS 上的创建时刻与最后修改时刻，换算为美东；带 `elapsed=` 的阶段以判定行自报耗时为准。起跑 HEAD 列中，A 段是按提交时刻推断的（日志没有打印 `START_HEAD`，见 launch.md），SigLIP 到 motion 由日志首行的 `WINDOW_HEAD=` 给出。

| 会话 | 阶段 | 起（美东） | 止（美东） | 耗时 | 起跑 HEAD | 退出 |
|---|---|---|---|---|---|---|
| `m8v-fetch-smoke` | A3 两任务小样本（G1） | 10-16 08:06:06 | 08:43:26 | 37 min 20 s | `e944b69` 之后 | `EXIT_CODE=0` |
| `m8v-b1gate` | B1 判据与 encoder 拷贝 | 10-16 08:06:47 | 08:06:52 | 5 s | 同上 | `EXIT_CODE=0` |
| `m8v-fetch` | 全量下载、校验、解包 | 10-16 08:11:36 | 09:09:05 | 57 min 29 s | `44fb642` | `EXIT_CODE=0` |
| `m8v-merge` | plan、merge、full verify、cleanup | 10-16 09:09:22 | 10:26:54 | 1 h 17 min 32 s | `aa29828` | `EXIT_CODE=0` |
| `m8v-hash` | episode manifest 与输入 sha256 | 10-16 10:27:02 | 11:28:37 | 1 h 1 min 35 s | `6ebeab5` | `EXIT_CODE=0` |
| `m8v-esbound` | 建库后窗口预算复算（G2a） | 10-16 10:51:23 | 10:51:24 | 1 s | `6ebeab5` | `EXIT_CODE=0` |
| `m8v-siglip` | SigLIP 4 worker 版（已停） | 10-16 11:28:55 | 12:37:27 | 1 h 8 min 32 s | `6ebeab5` | 主动停止，无 `EXIT_CODE=` |
| `m8v-siglip-w32` | SigLIP 32 worker，`--resume` | 10-16 12:39:29 | 14:30:43 | `elapsed=6673s` | `6ebeab5` | `EXIT_CODE=0` |
| `m8v-finalize` | 五项检查与 1024 帧复算 | 10-16 14:31:09 | 14:48:59 | 17 min 50 s | `6ebeab5` | `FINALIZE_EXIT_CODE=0` |
| `m8v-pack4` | 4×4 打包、verify、norm_stats | 10-16 14:49:19 | 15:04:13 | 14 min 54 s | `6ebeab5` | `EXIT_CODE=0` |
| `m8v-pack8` | 8×8 打包、verify、xgrid | 10-16 15:04:27 | 15:13:42 | 9 min 15 s | `6ebeab5` | `EXIT_CODE=0` |
| `m8v-wan` | Wan latent 抽取，GPU 0–7 | 10-16 14:49:19 | 21:14:43 | `elapsed=23119s` | `6ebeab5` | `EXIT_CODE=0` |
| `m8v-encode` | e72 encoder 编码 motion token | 10-16 21:15:46 | 21:19:36 | `elapsed=229s` | `6ebeab5` | `EXIT_CODE=0` |
| `m8v-oracle` | oracle encoder 段，vae 段首跑 | 10-16 21:19:47 | 21:41:22 | encoder `elapsed=1281s` | `6ebeab5` | `EXIT_CODE=1`（vae 8 片失败） |
| `m8v-oracle-vae` | vae 8 片补跑与 aggregate | 10-16 21:43:11 | 22:44:24 | `elapsed=3664s` | `6ebeab5` | `EXIT_CODE=0` |
| `m8v-motion` | motion pack、verify、对拍、A6–A11 | 10-16 22:44:38 | 22:45:49 | 71 s | `6ebeab5` | `EXIT_CODE=1`（A11 口径错误） |
| `m8v-motion-a11` | 只重跑 A11 | 10-17 00:49:59 | 00:50:00 | 1 s | `6ebeab5` | `EXIT_CODE=0` |
| `m8v-try20-512` | 512 四卡 20 步短测，GPU 0–3 | 10-17 00:50:33 | 00:55:54 | 5 min 21 s | `6ebeab5` | `EXIT_CODE=0` |
| `m8v-try20-motion` | motion 四卡 20 步短测，GPU 4–7 | 10-17 00:50:33 | 00:56:00 | 5 min 27 s | `6ebeab5` | `EXIT_CODE=0` |
| `m8v-latent-upload-smoke` | latent 库上传 SMOKE（到阶段 3） | 10-16 15:17:45 | 15:20:42 | 2 min 57 s | `6ebeab5` | `EXIT_CODE=0` |
| `m8v-latent-upload` | latent 库全量上传与回读 | 10-16 15:21:17 | 18:35:41 | 3 h 14 min 24 s | `6ebeab5` | `EXIT_CODE=0` |
| `m8v-trainset-upload-smoke` | VLA 新库上传 SMOKE | 10-17 00:50:33 | 00:51:38 | 1 min 5 s | `6ebeab5` | `EXIT_CODE=0` |
| `m8v-trainset-upload` | VLA 新库全量打包、上传与分批回读 | 10-17 00:52:24 | 04:57:06 | 4 h 4 min 42 s | `6ebeab5` | `EXIT_CODE=0` |

`m8v-viz`（B1 的 mp4，在 MotionJEPA-backup 运行，GPU 0–3）的日志位于 `/scratch/hongze/logs/m8v-viz.log`，起止为 10-16 08:07:09 至 11:32:16，末尾有 `DELIVERY=PASS episodes=200 cells=100 per_cell=2 ckpt=epoch_72` 与 `EXIT_CODE=0`。它的完整结果归 MotionJEPA 侧档案，这里只用作 Wan 起跑前置条件的证据。

`m8v-esbound` 的日志创建于 10:51，当时 `m8v-hash` 还在串行计算 16 个 H5 的 sha256。esbound 只读 `episode_manifest.json` 与 data-raw，而 manifest 在 hash 阶段开头就已写出，所以它与 hash 的输入哈希段实际是并行的。这与 launch.md「随后 m8v-esbound」的顺序描述不同，但不影响结果。

### 下载、合并与源数据

A3 小样本取 InsertPeg 与 BinFill，共 11 个包、400 条；全量取 100 个包、3200 条。两者的每个包都是首次尝试即 `FETCH_ARCHIVE=PASS`，没有用到重试。合并逐字拷贝（口径 34），full verify 逐对象比对零失配；cleanup 删除解包件前，再次核对 verify 日志里的 PASS 和计数。

```text
SNAPSHOT_OK manifest_sha256=e3c7564f787781910f741d57b487d45565defbe8d3c2c17af9233c651093de11
PLAN_OK tasks=2 selected=400 role=primary:400 max_timesteps=1989 prechecks=PASS precheck_files=400
MERGE_VERIFY=PASS level=full tasks=2 episodes=400 timesteps=314737 role=primary:400 mismatches=0
FETCH_DONE tasks=16 archives=100 episodes=3200
PLAN_OK tasks=16 selected=3200 role=primary:3200 max_timesteps=1989 prechecks=PASS precheck_files=3200
MERGE_VERIFY=PASS level=full tasks=16 episodes=3200 timesteps=2145548 role=primary:3200 mismatches=0
CLEANUP=PASS removed=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/staging/mixed8-200-src/extracted files=3200 verify_episodes=3200 timesteps=2145548
TRAIN_ES_BOUND_WORST task=PickXtimes output_index=196 source=PickXtimes_ep175_seed18500.h5 g=1596 demo_len=0 exec_len=1989 demo_windows=0 exec_windows=123 windows=123
TRAIN_ES_BOUND=PASS episodes=3200 tasks=16 max_windows=123 budget=160 headroom=37
```

full verify 的子日志 `m8v-merge-verify` 起于 09:24:40，止于 10:26:23。据此推算，plan 加 merge 约 15 min 18 s，verify 约 61 min 43 s。MERGE_VERIFY 的 timesteps=2,145,548，与 launch.md 中按 16 份 episode map 汇总的数完全一致。hash 阶段串行重读 16 个 H5，最后一个文件完成时的累计耗时为 3509 s。

实际 `canonical_order` 按任务名字母序排列：BinFill、ButtonUnmask、ButtonUnmaskSwap、InsertPeg、MoveCube、PatternLock、PickHighlight、PickXtimes、RouteStick、StopCube、SwingXtimes、VideoPlaceButton、VideoPlaceOrder、VideoRepick、VideoUnmask、VideoUnmaskSwap。每个任务 200 条，global episode 区间依次为 0–199、200–399……3000–3199。最坏的窗口出现在 g=1596，即 PickXtimes 的输出索引 196，与建库前实算的那一集相同。

### SigLIP、finalize 与两档打包

SigLIP 先以 4 worker 起跑，完成 100 条后停止；改为 32 worker 后用 `--resume` 处理余下 3100 条（原因见「计划外事件与处置」）。收尾核对：`SHARD_DONE` 恰 32 行，`source/meta/_shard*of32.json` 恰 32 个，sidecar 的 episode 并集为 3200，fingerprint 唯一（`git_commit=6ebeab5`、jax 0.5.3、A100），残留 claim 为 0。32 个 worker 各自的稳态速率（`rate_steady`）在日志 SHARD_DONE 行逐片可查，例如 shard 21 为 9.281 step/s。本轮未对 SigLIP 阶段做 GPU 利用率采样。

```text
WINDOW_HEAD=6ebeab5632340328a3a62db836d1ab3b004eed57
STAGE_DONE stage=siglip workers=32 items=3200 elapsed=6673s
  ✓ 同架构复算全部逐位一致 PASS（1024 条）
全部检查通过 PASS
FINALIZE_EXIT_CODE=0
VERIFY_PACK=PASS scanned=2145548 mismatches=0
VERIFY_PACK=PASS scanned=2145548 mismatches=0
IMAGE_NPY_SPOT=PASS frames=512 mismatches=0
MOTION_POS_XGRID=PASS t=1989 npy=64 mismatches=0
```

两行 VERIFY_PACK 依次对应 4×4 和 8×8。8×8 的 verify 用 `reader=decode`、48 进程，`started_at` 为 `2026-10-16T19:10:32`，`finished_at` 为 `2026-10-16T19:13:40`（UTC）。norm_stats 由 `m8v-pack4` 在 4×4 verify 之后写出，路径为 `v1-store/train-assets/mme_vla_suite/16task-mixed8-200-3200ep-e3c7564f/robomme/norm_stats.json`。8×8 打包完成后执行 `FEATURES_PRUNE=PASS kept=8`，删除前后对 8 个抽样文件做 headtail blake2b-128 复核，结果 8/8 一致。framesamp-8x8 的 store_meta sha256 为 `ace85246419836d343b925ac45591e3fd75eb2f054c16eaafafb3b35d6ca9b79`，framesamp（4×4）的 store_meta sha256 为 `9a2cf938b62d6df6fd3d13e1771ebaff1bb99061bcf6d382ac080783bead2254`。

### Wan、encode、oracle 与 motion

Wan 与 4×4/8×8 打包同时在 14:49 起跑（GPU 链和 CPU 链并行），8 个 worker 合计 4995 个段、128,342 个窗。每个 worker 的显存峰值为 1439 MiB。encode 起跑时在首行断言 ckpt 的 sha256：`EXPECTED_CKPT_SHA256=bec423fdfa0c9494690a5118ab10f2cb6fec58eba4b8362b4af90c71cbcd9e04`。

```text
STAGE_WORKSET=PASS stage=wan processed=4995 initial_complete=0 duplicates=0 missing=0
STAGE_DONE stage=wan workers=8 items=4995 skipped=415 initial_complete=0 elapsed=23119s
STAGE_WORKSET=PASS stage=encode processed=4995 initial_complete=0 duplicates=0 missing=0
STAGE_DONE stage=encode workers=8 items=4995 skipped=1200 initial_complete=0 elapsed=229s
ORACLE_ENCODER=DONE rows=128342 elapsed=1281s
ORACLE_SHARDS=PASS ok=8/8
ORACLE_VAE=DONE metadata_rows=128342 windows=14480 frame_mismatches=0 metadata_mismatches=0 elapsed=3664s shards=8
ORACLE_AGGREGATE=DONE kinds=vae shards=8
PACK_MOTION_DONE=1 raw_dir_match=1 （pack.lock 保留，verify 回填后才释放）
VERIFY_MOTION=PASS scanned=128342 mismatches=0
WAN_BITEXACT=PASS compared=14480 padded_covered=1800 frame_mismatches=0 latent_mismatches=0 metadata_mismatches=0 oracle_windows=14480
ENCODER_BITEXACT=PASS compared=128342 mismatches=0 order_ok=1 state_sha_ok=1 prov_ok=1 ckpt_ok=1 no_latent_stats_call=1 finite=1
A6_SAMESOURCE=PASS episodes=3200 manifest_sha_same=1
A7_BYTES=PASS segments=4995 mismatches=0 table_rows=128342 table_ok=1
A9_INDEXSET=PASS samples=3700 cold=3200 mismatches=0
A10_ROWS=PASS rows=128342 exec=81611 demo=46731 episodes=3200 formula_or_rowbase_mismatches=0 expect=128342=81611+46731
A11_PAD=FAIL {'segments': 4995, 'rows': 128342, 'padded': 1800, 'eligible': 1800}
A11_PAD=PASS segs=4995 rows=128342 padded=1800 set_diff=0
MOTION_LAYOUT=motion-768-grid16-demopad17-v1
```

vae 补跑共 8 片：较小的 4 片（0、2、4、6）耗时 2366–2462 s，较大的 4 片（1、3、5、7）耗时 3586–3664 s；8 片的 `SHARD_EXIT_CODE=0` 都在 aggregate 日志里。motion 表本体 394,266,624 B，pack 耗时 10.5 s。motion store_meta 的 sha256 为 `c9f8275fe12d6395ea7cc29191e39a1fe5df7b569c579b5acdfea07a71bdde2e`，`status=verified`，provenance 里 16 个 worker 的 `git_commit` 去重后只有 `6ebeab5…` 一个。上面 A11 的两行依次是首跑 FAIL 与重跑 PASS，前者原样保留作为证据。VERIFY_MOTION PASS（美东 22:45）标志 HEAD 冻结窗口结束。窗口内 policy 仓库没有任何提交，下一次提交是 10-17 00:58:53 的 `e17c922`。

### 四卡 20 步短测与数据身份

两条短测同时 4+4 并行，用正式 runner、`mme_vla_suite_b128_80k`、`--no-wandb-enabled`，nvidia-smi 以 500 ms 间隔采样显存。两条都 PASS，按口径 9 不触发 8 卡回退。

```text
MOTION_4GPU_SMOKE=PASS run=512 steps=20 exit=0 finite_losses=20/20 peak_mem_gib=39.3 limit_gib=72 mesh=TRAIN_MESH shape=(1, 4) devices=4 fsdp_devices=4 batch_size=128 num_workers=8
MOTION_4GPU_SMOKE=PASS run=motion steps=20 exit=0 finite_losses=20/20 peak_mem_gib=38.3 limit_gib=72 mesh=TRAIN_MESH shape=(1, 4) devices=4 fsdp_devices=4 batch_size=128 num_workers=8
DATA_IDENTITY=PASS norm_sha=36ee9acdea4c8383b50c7b07b45a740efb3fa073f6e60455977d5a6fb3204394 steps=2145548 samples=1383846 manifest_sha=e3c7564f787781910f741d57b487d45565defbe8d3c2c17af9233c651093de11
```

稳态步时取第 6→18 步：512 约 1.84 s/步，motion 约 1.90 s/步。照此外推，80k 步分别约 40.9 h 和 42.2 h。没有观察到本机 GPU 中断不送达问题对 JAX 训练的明显拖累。据此决定按 4+4 并行起跑，不启用串行回退。20 步短测覆盖不到 ckpt 保存和 EMA，这两项留给正式训练的前 100 步复查。补写本档案时，`v1-store` 下已找不到两个 try20 的 run 或 ckpt 目录，只剩 `cache/jax/m8v-try20-*` 编译缓存；清理的确切时刻、以及 `m8v-try20-*-mem` 采样会话的停止时刻，都没有单独记录。

### 产物体积

下表数字来源：库内各目录在删除中间件之后实测 `du -sb`；已删的目录取 trainset 上传驱动阶段 2 的计数。

| 产物 | 字节 | 文件数 | 当前状态 |
|---|---:|---:|---|
| 合并 H5（16 个） | 1,424,043,814,720 | 16 | 已删（口径 24） |
| `source/data` | 547,443,716,927 | 1,383,846 | 保留 |
| `framesamp`（4×4） | 140,811,413,527 | 未记录 | 保留 |
| `framesamp-8x8` | 562,936,604,762 | 未记录 | 保留 |
| `motion` | 398,091,845 | 未记录 | 保留 |
| `meta` | 962,682 | 2 | 保留 |
| `wan-latents` | 75,801,094,862 | 14,986 | 已上传，本地已删 |
| `motion-tokens` | 458,162,497 | 14,986 | 已上传，本地已删 |
| `oracle` | 76,111,416,593 | 10,010 | 已上传，本地已删 |

### 两次上传

latent 库驱动按口径 25 原样复用，没有改动。它的闸门是 4×4、8×8、xgrid 都 PASS 且 `source/features` 已删，`m8v-pack8` 在 15:13 结束后满足。SMOKE 只跑到阶段 3，没有碰网络写。全量共分 137 批上传。阶段 7 第一次计数得 `files=5119 bytes=1138498764460`，与计划不符，按驱动既有逻辑 120 s 后复查，结果相符。

```text
HYGIENE=PASS scanned=132 hits=50085 ungated=0
BUCKET_CREATED=HongzeFu/robomme-16task-mixed8-200-motion-20261014-v1 PRIVATE=False
PLAN_BATCHES=137 PLAN_FILES=5120 PLAN_BYTES=1139255338776 ROOT_FILES=6 MAX_BATCH_BYTES=8589017088 MAX_BATCH_FILES=134 PLAN_SHA=9f23f52f6d156f20
BUCKET_FILES=5120 BUCKET_BYTES=1139255338776
SHA256_PRE_POST_DIFF=0
MEMBERS_SPOT=PASS tar_members=5000 checked=200 mismatches=0
RESULT=PASS
SRC_UNCHANGED=OK checked=5119
BUCKET_PUBLIC=True
PUBLIC_ANON_READ=OK
全部完成
EXIT_CODE=0
```

latent 驱动阶段 8 会把约 1.14 TB 全量回读到 `v1-store/exports/hf-motionjepa-mixed8-200/verify`，且不会自删。确认阶段 9–11 全部 PASS、`EXIT_CODE=0` 之后，主会话只删了这一个目录。两次大回读因此串行：VLA 新库上传排在其后，且要等 try20 之后才起。

VLA 新库上传的 SMOKE 只做本地打包、体检与 tar 成员抽查，没有碰网络与凭据。全量阶段 5 断点续用 SMOKE 的打包记账，16.1 分钟打出 411 个分片。回读按口径 38 分 14 批，单批上限 107,374,182,400 B；实测峰值 106,213,810,959 B，没有超限。

```text
STORE_STATUS=verified stores=3 pack_lock=0
HYGIENE=PASS scanned=25040 hits=40245 ungated=0
MEMBERS_SPOT=PASS tars=2 members=8192 mismatches=0
DF_CHECK avail=2953562669056 need=1205400165471 tar_est=701272390751 tar_have=3246407680 tar_need=698025983071 readback_batch_max=107374182400 margin=400000000000
BUCKET_CREATED=HongzeFu/robomme-16task-mixed8-200-trainset-20261016-v1 private=False
[05:09:28] PACK_DONE shards=411 plain=124 members=1423828 bytes=1404958086716
HYGIENE=PASS scanned=52 hits=211 ungated=0
PLAN_BATCHES=157 PLAN_FILES=544 PLAN_BYTES=1405076546975 ROOT_FILES=2 MAX_BATCH_BYTES=17962631168 MAX_BATCH_FILES=45 PLAN_SHA=f590ec87eca7a859
BUCKET_FILES=544 BUCKET_BYTES=1405076546975
READBACK_PLAN batches=14 files=544 bytes=1405076546975 max_batch_bytes=106213806590 budget=106300440576 max_bytes=107374182400 largest_object=17962631168
READBACK_PEAK_BYTES=106213810959
SHA256_PRE_POST_DIFF=0
READBACK=PASS batches=14 lines=543 peak=106213810959 max=107374182400
RESULT=PASS
SRC_UNCHANGED=OK
BUCKET_PUBLIC_FINAL private=False
BUCKET_PUBLIC=True
PUBLIC_ANON_READ=OK
全部完成
EXIT_CODE=0
```

阶段 8 第一次计数得 `files=543`，与计划不符；同样在 120 s 后复查，结果相符。按口径 44，两个 bucket 都不含 `wandb/`。

### 删除与磁盘

各项删除都在对应闸门之后执行，删除前列出清单交主会话核对：

| 时点（美东） | 删除对象 | 依据与判定 |
|---|---|---|
| 10-16 08:43 之后 | A3 smoke 目录，约 391 GB | G1 PASS（08:43）之后 |
| 10-16 10:26 | 解包件 `staging/mixed8-200-src/extracted`，约 1.4 TB | `CLEANUP=PASS … files=3200 verify_episodes=3200` |
| 10-16 15:13 之后 | `source/features` 中未被 store_meta 点名的文件，约 1.2 TB | `FEATURES_PRUNE=PASS kept=8` |
| 10-16 18:35 之后 | latent 上传的回读目录 `exports/hf-motionjepa-mixed8-200/verify` | latent 上传阶段 9–11 PASS、`EXIT_CODE=0` |
| 10-17 约 05:00 | 合并 H5 `raw-h5/16task-mixed8-200-20261014`（1.3 TB）、`wan-latents`（71 GB）、`motion-tokens`（0.47 GB）、`oracle`（71 GB）、VLA 导出 stage 与空的 verify 目录 | 口径 24：motion verify PASS 且 VLA 新库上传 PASS |

encoder ckpt 上传的回读目录也已删除，那次上传归 MotionJEPA 侧档案。导出 stage 中 tar 实体约 700.9 GB；其余是指向库内原件的硬链接，删除不影响原件，这一点已由上传驱动阶段 10 的 `SRC_UNCHANGED=OK` 证明。最后一次删除前，确认没有进程通过 maps 或 fd 引用这些文件。删除后 `/scratch` 空闲从约 2.1 TiB 回到约 4.1 TiB，补写本档案时实测为 4,505,756,684,288 B。A3 smoke 目录、features 与 latent 回读目录三次删除的确切时刻未记录。

长期保留的有：`framesamp/`、`framesamp-8x8/`、`meta/`、`motion/`、`source/`（`data/` 与 8 个抽样 features），以及 norm_stats。这些都已随 VLA 新库公开，可以从 HF 恢复。

## 计划外事件与处置

**A1 子代理卡死约 4.5 h（进展登记 21）**：美东约 03:08，主会话误调 EnterWorktree，把整个会话连同后台子代理锁进了计划 worktree。A1-merge 与 A1b 的工具调用一直挂起，到 07:47 才被发现，随后用 TaskStop 停掉，再用 SendMessage 续接完成。A 段因此按风险 8 顺延；mp4 与 SigLIP 的前置不受影响。

**全量下载与 A3 并行（进展登记 21）**：经用户同意、`ADJUST_REVIEW=PASS scale=same`，`m8v-fetch` 于 08:11 起，与 `m8v-fetch-smoke` 并行；`m8v-merge` 仍然等 G1 PASS 才起。

**SigLIP 改 32 worker（进展登记 21）**：4 worker 版每个 worker 只有约 9.9 step/s（9 月同代码为 76），按此全库约需 16.5 h。主会话实测查明根因：本机 48 个 nvidia 中断计数冻结，2 秒增量为 0，说明 GPU 完成中断没有送达，驱动以约 100 ms 超时轮询。于是 JAX 每次阻塞同步固定耗时约 100 ms，GPU 利用率只有 5%；torch 默认自旋等待，不受影响。排除项：关掉 `watch -n 1 nvidia-smi` 后复测，速率没有变化。用户回复「同意多worker+多gpu全用上」，`ADJUST_REVIEW=PASS scale=same findings=6`。处置没有改代码：

1. 停掉 `m8v-siglip`，确认 build_shard 进程全部退出、8 卡显存为 0。
2. 删除残留的 4 个 claim。
3. 新会话 `m8v-siglip-w32` 于 12:39 起，用同一个 `run_local`，每张卡重复 4 次共 32 worker，`XLA_PYTHON_CLIENT_MEM_FRACTION=0.15`，加 `--resume` 跳过 4 worker 版已完成的 100 条。

HEAD 冻结窗口不变，判定行改为 `workers=32`。`run_local` 以 GPU 号为键记录 rc，同卡 worker 的退出码可能互相覆盖，所以收尾另核 SHARD_DONE 行数、`_shard*of32.json` 文件数与 finalize 的全覆盖。

**oracle vae 首跑失败（进展登记 22）**：`$S/env.sh` 按红线 6 写了 `HF_HOME=/scratch/hongze/.cache/huggingface`，覆盖了 `scripts/dataset/paths.sh` 设定的 `$V1_STORE/cache/hf`，而 Wan VAE 快照就在后者。在 `HF_HUB_OFFLINE=1` 下，8 个 vae 分片都报 `LocalEntryNotFoundError`，`SHARD_EXIT_CODE=1` ×8；原日志改名为 `m8v-oracle-vae-[0-7].fail1.log` 留证。encoder oracle 已经 `ORACLE_ENCODER=DONE rows=128342`，不受影响。修正方法是让 env.sh 不再覆盖 `HF_HOME`（`paths.sh` 的路径仍在 `/scratch/hongze` 内），新会话 `m8v-oracle-vae` 只补跑 vae 8 片与 aggregate，补跑日志首行回显 `HF_HOME=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/hf`。第一轮修复即通过。训练 runner 自己 source `paths.sh`，不受此问题影响。

**A11 期望参数口径错误（进展登记 22）**：runbook 原写 `--expect-episodes 3200`，`m8v-motion` 因此判 `A11_PAD=FAIL`。核对 `motion_checks.py` 的 `cmd_a11` 发现，它用 eligible（`exec_start_idx>=17` 的 episode 数）与 `--expect-episodes` 比对。4 任务库的 1600 条全部带 demo，两者恰好重合；mixed8 只有 1800 条带 demo。`audit_padding` 的逐段断言都没有抛错，`padded==eligible`。这属于确定性对拍 FAIL，按停点 9 停下问用户。用户答复「a」，于是只重跑 A11 并传 `--expect-episodes 1800`，得到 PASS，runbook 也已同步改正。从 FAIL（22:45）到重跑（00:49）约 2 h 4 min，这段时间等待用户裁决，其间 GPU 链没有工作；用户答复的确切时刻未记录。

**小问题，不影响数据**：motion store_meta 的 `provenance.source_pin.mj_repo_path` 记为旧集群路径 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA`，不是本机的实际路径。它旁边的 `mj_repo_commit=2a484ad960ed6155321dc34def9011eb119f857f` 与 ckpt sha256 照常记录。数据内容与对拍结论（`ENCODER_BITEXACT … prov_ok=1 ckpt_ok=1`）都不依赖这个字段，本轮不改。

**计数接口滞后**：两次上传的计数层核对第一次都差 1 个文件，按驱动既有的「120 s 后复查一次」逻辑复查后一致。没有触发上传失败处理。

## 用户决策记录

编号行的用户原话从计划 `952b8db` 的「一、已定口径」逐字摘录；无编号的行出自进展登记。右列写本库的实际落地。

| 口径 | 用户原话 | 本库实际落地 |
|---|---|---|
| 开工令 | 「/scratch/hongze/MotionJEPA-backup/1016-mixed8-vla-plan.html 开工 有关键问题问用户但尽可能不要阻塞」 | 按停点停下问用户的只有 A11 一处（停点 9）；全量下载并行与 SigLIP 改 worker 数两项经用户同意后执行；其余按不阻塞规则自动处理并登记 |
| 2 | 「vla数据集还是用的encoder训练的那个数据集就是你相当于要重新做一次预处理」 | 从 3200 条源 H5 全新建库，库、norm_stats、合并 H5 全部是新输出 |
| 3 | 「我说的512和512+motion训练 是用的8帧8*8feature」「我也需要8乘8的」 | 8×8 打包、verify、xgrid 全 PASS；两条短测与正式训练都读 `framesamp-8x8/` |
| 12 | 「VLA 训练集范围全部」 | 3200 条全部入库，`DATA_IDENTITY` samples=1,383,846 |
| 13 | 「4×4 framesamp 打包同意」 | 4×4 打包并保留，140,811,413,527 B |
| 14 | 「推荐先不删，盘够。如果盘不够了立刻删除」 | 全程没有进入磁盘应急，`data-raw/` 未删 |
| 24 | D1：「1同意」 | 合并 H5 在 VLA 新库上传 PASS 之后（美东 10-17 约 05:00）才删 |
| 25 | D2：「2 encoder上传可以错开 往后推 飞出方案」；第二轮：「你直接给我这个prompt然后我直接输给他然后直接让他放弃做这件事情这件事情由我们这边做」 | latent 库驱动原样运行，在 features 删除后起跑，PASS；只删其 verify 目录，两次回读串行 |
| 29 | D6：「6 ckpt全部传不用管公开的事情无所谓 所有训练的产物都要上传 encoderdecoder什么的都要传。核心逻辑是所有需要训练的训练之后的产物公开数据集、公开的chekpoint里面没有的都需要」 | VLA 新库含 framesamp 两档、motion、motion-tokens、wan-latents、oracle、source（data、meta、8 个抽样 features）、meta、norm_stats、runner、README |
| 30 | D7：「7 无所谓你定」 | bucket 名分别为 `…-trainset-20261016-v1` 与沿用的 `…-motion-20261014-v1` |
| 34 | D11：「11评估一下严重性如果没有特别需要改变的你可以继续你需要报告用户。否则的话就要停下来」 | plan 预检两次 `prechecks=PASS`，未触发适配；merge 逐字拷贝 |
| 35 | 第三轮：「3 c」（TRAIN_ES_BOUND headroom<0 怎么办） | 建库后 headroom=37，没有触发 |
| 38 | 第四轮之后：「a. 分批回读（推荐） 同意」 | 14 批回读，峰值 106,213,810,959 B，不超过 100 GiB |
| 42 | 「这些都是很细节的内容了你自己来决定 然后你直接自己修改不用再走子代理的修改和推送了。」 | HEAD 冻结窗口从 SigLIP 起跑到 motion verify，全程 `6ebeab5`、无提交；launch.md 在 SigLIP 前提交 |
| 43 | 「a同意」 | build 交接只留 try20、`DATA_IDENTITY` 与建库档案；build done 的 handoff `692275e` 早于 V11.24Beta `62a8ef1` |
| 44 | 「按a执行」 | 两个 bucket 都不上传 `wandb/` |
| 进展登记 21 | 「全量下载是否和 A3 并行 同意」 | `m8v-fetch` 与 `m8v-fetch-smoke` 并行 |
| 进展登记 21 | 「同意多worker+多gpu全用上」 | SigLIP 改 32 worker，`--resume` 续跑 |
| 进展登记 22 | 「a」（A11 期望参数） | 只重跑 A11，`--expect-episodes 1800` |

## 结论与下一步

本库可以直接作为两条 80k 正式训练的输入，判据如下：

- 数值与结构层：合并 full verify、finalize 1024 帧复算、两档 VERIFY_PACK、xgrid、VERIFY_MOTION、Wan 与 encoder 对独立 oracle 的逐位对拍，全部零失配。
- 身份层：MANIFEST `e3c7564f…`、canonical manifest `d6a3ddb2…`、norm_stats `36ee9acd…`、motion store_meta `c9f8275f…` 互相绑定，`DATA_IDENTITY=PASS`。
- 可训练性层：两条四卡 20 步短测 loss 全部有限，显存约为限值的一半。

**下游**：两条 80k 正式训练已经起跑。

- `mixed8-200-m8x8-modul-b128-80k`：美东 10-17 01:00 起，commitV11.24Beta `62a8ef1`，GPU 0–3。档案见 [训练档案](../../training-doc/mixed8-200-m8x8-modul-b128-80k/launch.md)。
- `mixed8-200-m8x8-modul-motion-b128-80k`：01:01 起，commitV11.25Beta `ae6e9cd`，GPU 4–7。档案见 [训练档案](../../training-doc/mixed8-200-m8x8-modul-motion-b128-80k/launch.md)。

两条按短测外推各需约 41–42 h。训练期间按红线 18，policy 主检出的代码与依赖处于冻结和只读锁状态。

**下一步**（都不属于本档案范围）：

- 两条训练的盯盘、结束后的 result 回写与 ckpt 上传，归各自的训练档案。
- `.git/info/exclude` 中那 3 条路径行按口径 41 保留。
- motion store_meta 的 `mj_repo_path` 如需改正，留到训练结束、代码冻结解除之后另案处理；本库数据不需要重建。

## 归档文件清单

`records/` 只放 git 还原不出的东西：判定行汇总和各阶段的清洗日志，没有任何 .sh、.yaml 或数据产物。清洗命令为 `tr '\r' '\n' < <日志> | grep -vE '%\|'`，有两份例外：

- `latent-upload.summary.log`：另外删去空行。原日志的 `\r` 进度被拆成 133 万多个空行，不删空行就超过 1 MB。
- `siglip-w32.summary.log`：另外删去逐条的 `PROGRESS steps=` 进度行和逐集 `Episode N: timesteps=…` 回显行。逐集完成行 `EPISODE g=`、全部 `SHARD_DONE`、`STAGE_DONE` 和 `EXIT_CODE=` 都保留。

为通过 `git diff --check`，所有清洗日志另外去掉了行尾空白和文件末尾的空行，行内容不变。

oracle vae 首跑的 8 份分片日志（`*.fail1.log`）已经整体并入 `oracle.summary.log`，其中包含 8 行 `SHARD_EXIT_CODE=1` 与 traceback，因此不再单独归档。完整原始日志保留在 `v1-store/logs/m8v-*.log`。

| 文件 | 内容 |
|---|---|
| [verdicts-b1-b4.md](records/verdicts-b1-b4.md) | B1 到 B4 与上传的全部判定行汇总（第一版写于 VLA 新库上传进行中） |
| [b1gate.summary.log](records/b1gate.summary.log) | B1 判据、E72 EMA 断言、encoder 拷贝 G3 |
| [fetch-smoke.summary.log](records/fetch-smoke.summary.log) | A3 两任务下载、合并、full verify（G1） |
| [fetch.summary.log](records/fetch.summary.log) | 全量 100 包下载与 16 行 FETCH_VERIFY |
| [merge.summary.log](records/merge.summary.log) | 全量 plan、逐集 MERGE_EP、MERGE_VERIFY、CLEANUP |
| [merge-verify.summary.log](records/merge-verify.summary.log) | full verify 子日志（16 行 VERIFY_SHARD） |
| [hash.summary.log](records/hash.summary.log) | manifest scan 与 16 个 H5 的 sha256 |
| [esbound.summary.log](records/esbound.summary.log) | 逐任务窗口预算与 TRAIN_ES_BOUND |
| [siglip-w4.summary.log](records/siglip-w4.summary.log) | 4 worker 版 SigLIP（停止前完成 100 条，无 EXIT_CODE） |
| [siglip-w32.summary.log](records/siglip-w32.summary.log) | 32 worker 版 SigLIP 与 STAGE_DONE |
| [finalize.summary.log](records/finalize.summary.log) | finalize 五项检查 |
| [pack4x4.summary.log](records/pack4x4.summary.log) | 4×4 打包、verify、norm_stats |
| [pack8x8.summary.log](records/pack8x8.summary.log) | 8×8 打包、verify、IMAGE_NPY_SPOT、MOTION_POS_XGRID |
| [wan.summary.log](records/wan.summary.log) | Wan latent 抽取 |
| [encode.summary.log](records/encode.summary.log) | motion token 编码 |
| [oracle.summary.log](records/oracle.summary.log) | oracle encoder 段，以及 vae 首跑 8 片失败的证据 |
| [oracle-vae.summary.log](records/oracle-vae.summary.log) | vae 8 片补跑与 aggregate |
| [motion.summary.log](records/motion.summary.log) | motion pack、verify、对拍、A6–A10，以及 A11 首跑 FAIL |
| [motion-a11.summary.log](records/motion-a11.summary.log) | A11 重跑 PASS 与 motion store_meta sha256 |
| [try20-512.summary.log](records/try20-512.summary.log) | 512 四卡 20 步短测 |
| [try20-motion.summary.log](records/try20-motion.summary.log) | motion 四卡 20 步短测 |
| [try20-512.smoke.summary.log](records/try20-512.smoke.summary.log) | 512 短测的 MOTION_4GPU_SMOKE 判定行 |
| [try20-motion.smoke.summary.log](records/try20-motion.smoke.summary.log) | motion 短测的 MOTION_4GPU_SMOKE 判定行 |
| [identity.summary.log](records/identity.summary.log) | DATA_IDENTITY 判定行 |
| [latent-upload-smoke.summary.log](records/latent-upload-smoke.summary.log) | latent 库上传 SMOKE |
| [latent-upload.summary.log](records/latent-upload.summary.log) | latent 库全量上传、回读、公开性 |
| [trainset-upload-smoke.summary.log](records/trainset-upload-smoke.summary.log) | VLA 新库上传 SMOKE |
| [trainset-upload.summary.log](records/trainset-upload.summary.log) | VLA 新库全量打包、上传、分批回读、公开性 |

launch.md「归档与验收计划」原本拟归档的 JSON 文件，本轮都没有拷入 records：16 份 episode map、`merge_plan.json`、两份 manifest、SigLIP shard fingerprint、三份 store_meta、norm_stats 摘要。原因是这些文件都已随 VLA 新库进入公开 bucket，sha256 也已在上文和判定行中钉死。try20 的显存采样 `.mem.csv` 只取峰值写入判定行，原文件留在 `v1-store/logs/`。
