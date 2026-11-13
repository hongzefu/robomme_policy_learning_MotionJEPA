# v2-1600ep-m8x8-modul-wanraw9s32-b128-50k 起跑档案

本页在起跑那一刻写入（版本与启动配置），训练结束后另写 result.md。计划见仓库根目录 [1112-4task-modul-wanraw9-s32-50k-plan.html](../../../1112-4task-modul-wanraw9-s32-50k-plan.html)。

## ① 目的

在 4task 原训练库 `4task-v2-1600ep-604f16da`（BinFill、RouteStick、VideoRepick、VideoUnmaskSwap 各 400 局，共 1600 局）上，与 v11 `v2-1600ep-m8x8-modul-wantp768-b128-50k`（Wan raw latent 零参数池化 768、1 token/窗、stride 16；692 局 298/692 = 43.06%）对照：motion 改为**不池化**——每个 33 帧窗的 9 个 Wan raw latent 各展平 16384 维，经同一个可学习 `Linear(16384+768 → 1024)`（17,564,672 参数）各成 1 个记忆 token；窗口 stride 由 16 改为 32。其余（帧路 8×8、modulation、B128、8 卡、50001 步、超参）与 v11 相同。训完由链脚本自动接本机 692 局评估（seed 7、前缀库 `binfill-3b4de03a0b46`）。

用户原话：「我之前训了wan直接pool作为modulation的 4task每个task400 成功率43。1的 我现在训一个不pool的 直接9个latent都在 每个latent单读投影 然后stride原来是16 现在改为32 应该怎么做？」「训完也是直接eval」「同意写完整方案 1234都按照你说的 data成瓶颈的话你看能不能修复？」「bf16我同意 没问题」「登记」「开工」「同意开工，按照推荐开工，你可以直接开始了。不用再找我」。

## ② 版本与代码状态

- 分支 `v15-4task-modul-wanraw9-s32-50k`；起跑锚点为本档案所在的 Beta 提交（`commitV15.17Beta`；`commitV15.13Beta` fac3502 与 `commitV15.15Beta` 7ad7890 两次起跑已作废，见下）。代码锚点基线 `c55a6bc`（v11），计划 `1531e6d`→`bcb2fb2`。
- 合入：S1 `commitV15.4/15.5`（wan_raw9 共享模块、导出/校验、motion_store 步长与行形状参数化 + 只读 memmap、dataloader 闸、train.py 快照与诊断键）、S2 `commitV15.6`（YAML、percep_mem、framesamp_dataset raw9 取数、history_pi0 长度、DL 等价闸）、S3 `commitV15.7`（sidecar 不池化、协议、客户端、在线缓冲、check_shard、check_checkpoint RUNS、G4 对拍）、S4 `commitV15.8`（runner、链脚本、ckpt 导出）、S5-L1 `commitV15.9/15.10`（数据管线 L1 + 迭代器干净收尾）、S3-G5 `commitV15.11/15.12`（G5 单 episode H5 读法与 modulation 前向）。
- 依赖：主 `.venv` 按本分支 `uv.lock`；sidecar 用 `v1-store/venvs/wan`。

## ③ 启动与配置还原

- TrainConfig：`git show <Beta>:src/mme_vla_suite/training/config.py` 中 `mme_vla_suite_b128_80k`（batch 128、8 卡 FSDP8、num_workers 16、seed 42、EMA 0.999、AdamW 裁剪 1.0、warmup 5000、peak_lr = decay_lr = 5e-5、save/keep 5000）。
- 启动脚本覆盖（`git show <Beta>:scripts/training/run_4task_wanraw9s32_50k.sh`）：`--exp-name v2-1600ep-m8x8-modul-wanraw9s32-b128-50k`、`--model.history-config perceptual-framesamp-modul-8frame-8x8-wanraw9s32.yaml`、`--num_train_steps 50001`；`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`；诊断开关 `TRAIN_TIMING_STEPS=300`、`TRAIN_TIMING_PROFILE=0`。
- history YAML 文件 sha256 `21f8db0ef58179f5a6440bccb3a564117f70b91f2185c637034efa3acf7397c0`，resolved_sha256 `819a001c58b00d4a4110835b913ac1fb58238fb71d272eaa33a429f6dcb53062`；motion 节：dim 16384、tokens_per_window 9、budget 80（窗数，token 720）、stride 32、encoder_kind wan_raw9、readout wan_raw9_flat16384_s32_v1。
- 启动命令（tmux `r9-train`，主检出，HEAD 等于 Beta 且工作区 clean）：

```bash
tmux new-session -d -s r9-train "cd /scratch/hongze/robomme_policy_learning_MotionJEPA; ulimit -c 0; set -o pipefail; PYTHONUNBUFFERED=1 bash scripts/training/chain_wanraw9s32_train_eval.sh <Beta 全 sha> 2>&1 | tee v1-store/logs/r9-train.log; echo \"EXIT_CODE=\$?\" >> v1-store/logs/r9-train.log"
```

- 链脚本：runner EXIT_CODE=0 且 `CKPT_LOAD=PASS step=50000`、`TRAIN_FINAL_RECORD=PASS step=50000 ckpts=10` 后另起 tmux `r9-eval`：`POLICY=perceptual-framesamp-modul-8frame-8x8-wanraw9s32 RUN_NAME=local692-wanraw9s32-50k EXPECT_MOTION_STRIDE=32 EXPECT_MOTION_BUDGET=80 bash scripts/evaluation/local_692_pool.sh <RUN_ROOT>/50000 7`。
- ckpt 上传长尾（tmux `r9-ckptup`）：`bash scripts/dataset/hf_export/run_wanraw9s32_ckpt_export.sh`，bucket `HongzeFu/robomme-4task-modul-wanraw9-s32-50k-v1`，完成行 `HF_VERIFY=PASS ckpts=10`。

## ④ 数据与划分

- VLA 库：`v1-store/datasets/4task-v2-1600ep-604f16da`（同 v11），canonical manifest identity `4cd5a170…bb918`，input_manifest 文件 sha256 `655e64b4…`。
- motion 表 `motion-wan-raw9-s32`：取本机 stride16 Wan raw latent（71316 窗）的偶数窗，不重跑 VAE。`EXPORT_RAW9=PASS rows=36484 demo=18367 exec=18117 padded=821 budget_max=71 table_sha256=0e6657af118e65c762b86dfd86ce0794289108b3be37ad5017fe9c26d1442d19`（约 21.5 GB fp32，只读 memmap）。

## ⑤ 起跑前闸门

| 闸门 | 判定行 |
|---|---|
| 开工前审查 | `PLAN_CONSISTENCY_REVIEW=FAIL base=127072a findings=9 blocking=3`（用户裁决按推荐，计划 `bcb2fb2` 修订） |
| 规则同步 | `GUARD_FIX=PASS commit=cd5df94` |
| G0 前置 | `DATA_LINEAGE_PRE=PASS table=pending encoder=wanvae-9980d252+raw9s32 … files=4 mismatches=0` |
| G0 | `DATA_LINEAGE=PASS encoder=wanvae-9980d252+raw9s32 encoder_source_pin=41f4ca55… vla_input_pin=41f4ca55… files=4 mismatches=0 binding=input_manifest_sha256:655e64b4`（冒烟与测速 runner 内置，每次通过） |
| G1 | `RAW9_TABLE=PASS rows=36484 demo=18367 exec=18117 budget_max=71 mismatch=0` |
| G2 | pytest 112 passed（含 `OLD_STRIDE16_REGRESS=PASS baseline=c55a6bc`、`V11_PARAM_TREE=PASS n_leaves=65`、`RAW9_PARAM_TREE=PASS kernel=(17152, 1024) params=17564672`）；`V11_REGRESS=PASS samples=256 mismatch=0 leaves_equal=1` |
| G3 | `DL_EQUIV=PASS samples=256 mismatch=0` |
| G4 | `ONLINE_RAW9=PASS windows=64 padded=8 bitexact=64`（首跑缺 18 个单 episode H5，补下 `H5_SAMPLES=PASS episodes=36 bad=0` 后重跑） |
| G5 | `TIC_SIDECAR=PASS episodes=5 points=114 motion=sidecar windows=107 blocking=14/14 observe=7`；`MEM_S_VS_TRAINSET=PASS points=114 keys=8`、`MOTION_S_VS_SIDECAR=PASS windows=107 mismatches=0`、`OBS_T_VS_I=PASS points=114`、`VT_FULL_VS_CACHED_F32=PASS rel_fro=1.33e-07`、`ACT_T_VS_I=PASS seeds=3 mismatches=0`（bf16 `VT_FULL_VS_CACHED` rel_fro=0.00311 为观察项）。前 3 次未出判定：缺合并整任务 H5（改读单 episode H5，commitV15.11）、L3/L4 前向未传 modulation 记忆（commitV15.12）、主会话误用 worktree 路径 |
| G6 | `SMOKE20=PASS steps=20 save_load=PASS`；L1 后 `r9-smoke20-l1` `SMOKE20=PASS`、`CKPT_LOAD=PASS step=19 resolved_sha256=819a001c…` |
| 测速 L0 | `BENCH300=PASS steps=300 s_per_step=1.8740 dl_wait_frac=0.5987 gpu_util_mean=54.61` → 派 S5-L1 |
| 测速 L1（末级） | `BENCH300=PASS steps=300 s_per_step=1.1467 dl_wait_frac=0.2854 gpu_util_mean=94.44 gpu_util_zero_frac=0.0114` |
| G6b | `EVAL_LOAD_SMOKE=PASS episodes=1 handshake=PASS share2=PASS peak_mem_gb=78.3`（单片 372 s，同卡 2 片墙钟 487 s） |

## ⑥ 硬件与资源

AWS 8×A100-SXM4-80GB，本地 NVMe RAID `/dev/md0`；`/scratch` 余量约 2.6 T。L1 末级 1.147 s/步，预计 50001 步约 16 h，评估约 2.5–3 h。

## 实施中调整（摘要，详见计划「八」与 commit body）

1. 闸 G5 由 C 阶段挪到 RB-D3 判级后、RB-D4 前，用末级冒烟 ckpt（`ADJUST_REVIEW=PASS scale=same findings=2`）；G5 脚本改读单 episode H5（合并整任务 H5 本机没有）。
2. 数据管线触发 L1：主进程缺页 / make_array / 释放压在主线程，改后台取批 + 释放线程与共享内存直填；L1 后迭代器需干净收尾（首版退出 SIGABRT 134）。
3. G4 首跑缺 H5 小样，按 v11 口径流式补下 10 个 tar.xz 对象中所需 18 个 episode。

## 第一次起跑事故（commitV15.13Beta fac3502，已作废）

美东 2026-11-12 16:57 起跑，约 17:08 在第 299–300 步 `terminate called after throwing an instance of 'absl::lts_20230802::BadStatusOrAccess'` → `EXIT_CODE=134`，链脚本 `CHAIN=STOP reason=train_exit runner_exit=134`。前 300 步 s/步 1.0–1.2、无 nan/OOM，无 ckpt。根因判断：runner `TRAIN_TIMING_STEPS=300` 令 `scripts/training/step_timing.py::StepTiming` 前 300 步开 `jax.profiler` trace，第 300 步 `stop_trace()` 后训练继续，与 L1 数据管线后台线程冲突（v11 同设置无事；冒烟 20 步与测速 300 步即结束未覆盖）。修复 `commitV15.14`（FIX-PROF fc4a0e5）：新增 `TRAIN_TIMING_PROFILE`，raw9 runner 全模式置 0（只写计时 jsonl、不调 jax.profiler），训练数值不变。验证：`R9_TIMING_STEPS=10` 冒烟 `r9-smoke20-l2` → `STEP_TIMING_DONE steps=10 profile=0` 后继续训练，`CKPT_LOAD=PASS step=19`、`SMOKE20=PASS steps=20 save_load=PASS`、`EXIT_CODE=0`；`PRE_MERGE_REVIEW=PASS base=fac3502 tip=fc4a0e5 blocking=0`。第一次起跑产物改名留存：`v1-store/train-runs/mme_vla_suite_b128_80k/<run>.try1`、`v1-store/bench/<run>/train-try1/`、`v1-store/logs/r9-train.try1.log`、`<run>.try1.runner.log`。用户原话：「修复然后起跑，有这种问题你都要自己解决，你不能每次在这种情况下就停。」

## 第二次起跑事故（commitV15.15Beta 7ad7890，已作废）

美东 2026-11-12 17:42 起跑，1.1 s/步稳定跑到约第 7560 步（step 5000 ckpt 已存，`STEP_VERIFY=PASS sub=5000 files=20` 上传），约 20:06 DataLoader worker 报 `No space left on device (28)`：/dev/shm（561 G tmpfs）写满，`EXIT_CODE=1`、`CHAIN=STOP reason=train_exit`。根因：L1 `_PrefetchIter` 释放线程 `_Releaser` 队列无界（`queue.SimpleQueue`），释放稍慢于取批即无限积压用过的共享内存批（CPU 复现 240 批积压 178 批、/dev/shm 325 GB）。修复 `commitV15.16`（FIX-SHM 132d549）：队列改 `queue.Queue(maxsize=1)` 反压；`SHM_LEAK=PASS batches=240 rel_q_max=1`、`RELEASER_CHECK=PASS`；`PRE_MERGE_REVIEW=PASS base=7ad7890 tip=132d549 blocking=0`。train.py 禁用续训（ckpt 仅 params），从头重跑；产物改名为 `*.try2` 留存；导出目录重置（bucket 中 step 5000 为 try2 内容，新 run 覆盖上传并 sha256 读回校验）。新增盯盘：tmux `r9-shmmon` 每 60 s 记 `/dev/shm` 已用量与 tqdm 进度，>300 GB 打 `SHM_ALERT`。

## 本轮 tmux 会话清单（截至起跑）

已结束：`r9-g0`、`r9-export`、`r9-verify`、`r9-dlequiv`、`r9-online`、`r9-h5`、`r9-regress`、`r9-smoke20`、`r9-bench300`、`r9-smoke20-l1`、`r9-bench300-l1`、`r9-g5`、`r9-g6b`、`r9-train`（第一次，134 中止）、`r9-smoke20-l2`、`r9-train`（第二次，/dev/shm 写满中止）、`r9-ckptup`（第二次）。起跑：`r9-train`（之后链脚本起 `r9-eval`）、`r9-ckptup`、`r9-shmmon`。

## 盯盘项

Monitor 过滤 `DATA_LINEAGE=|CKPT_LOAD=|TRAIN_FINAL_RECORD=|CHAIN=|STEP_TIME step=(1000|5000|10000|20000|30000|40000|50000) |nan|NaN|out of memory|Killed|Traceback|terminate called|^EXIT_CODE=`；训练崩溃 / 发散（nan、OOM、Killed、Traceback）停下报用户；步速变慢只记录并告知（D8）。
