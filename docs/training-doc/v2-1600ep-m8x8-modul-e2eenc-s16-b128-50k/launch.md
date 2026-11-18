# launch.md — v2-1600ep-m8x8-modul-e2eenc-s16-b128-50k（E2E-Enc，任务 id e2eenc-s16-50k）

计划：`1117-e2e-enc-train-plan.html`，用户批准版本 = `c618671`（v15 分支 commitV15.21）。

## 起点（RB-A1）
- 开工令：2026-11-17 23:4x 美东，用户原话「开工」（此前定稿审阅约定：「先定稿，定稿完了之后，不要立刻开工，让我再看一眼通过」；停止规则：「我同意开工之后就尽可能一路跑到底，除非有训练语义上的不同……」）。
- 实施分支 `v18-4task-modul-e2eenc-s16-50k` 自 `c618671` 创建（不从 c0826be 直接起）；代码基线 = `c0826be`（c618671 的祖先）。
- 执行位置：主检出 `/scratch/hongze/robomme_policy_learning_MotionJEPA`（runner 写死 `MAIN=` 该路径），由 v17 分支切到 v18；v17 工作区干净、已推送。
- 规则子模块随 repo_guard 更新到 b91c20d（`6d9e36a`）。

## 资源与 A0 现状（RB-A0/RB-A2）
- 2026-11-17 23:41 美东：8 卡 memory.used 全 0 MiB；NativeMEM 23:09:43 美东结束（NM_PIPELINE=PASS、EVAL692=PASS 290/692、EXIT_CODE=0）。
- `df -B1 /scratch`：可用 566,394,560,512 B（93% 已用）。

## 偏离/调整记录（R8 第 (i) 档等只记不报的项写这里）
（暂无）

## RB-A3 核实：MotionMEM 80k 全局 grad_norm
- `v1-store/logs/v2-1600ep-m8x8-modul-motion-b128-80k.log`（1,233,164 B，2026-09-19）含 804 条 `grad_norm=`，每 100 步一条（如 `Step 0: grad_norm=1.1137, llm_grad_norm=1.1026, …`）。历史全局梯度范数轨迹可作参照；逐步裁剪系数仍只能由 e2e 自身逐步日志给出（log_interval 均值不等于逐步）。

## 实施合入记录
| 子任务 | TIP | 合并提交 | 合并前/后审查 |
|---|---|---|---|
| S5a 触发器 | b4895ab | 24bf94b | PASS/PASS findings=0 |
| S2 bf16 s16 表 | 5a62789 | 817551e | PASS/PASS findings=0 |
| S1 encoder 与对拍 | 436c28e | 22bc278 | PASS/PASS findings=0 |

## 主会话裁决（只记不报类，非训练语义）
- 标准化常量取 MotionJEPA ckpt buffer 的 fp32 位模式（与 VAE config 常数 max|Δmean|=4.17e-7、max|Δstd|=3.22e-5）；理由：depth=8 对拍逐位对齐，从头训练时差异可忽略。
- e2e-trig 触发器推迟到闸门表可填（S5 合入）后再起：开工时 A0 已满足（NativeMEM 已结束、8 卡 0 MiB），空表起触发器只会全部 not_configured。G1/G1b/G1c 由主会话直接在 1 卡串行跑（A0 现状已满足，见下）。
- 固定窗口清单 `scripts/training/g0/e2e_windows_v1.json` sha256 = 32e9aecc20d21bb68948a25f51657e1db2b3f50501dcbe2f0247ae28eae7b759（2048 窗，每任务 512）。
- 已知既有失败：`scripts/training/tests/test_s3_eval_local.py` c4/c5/c7 在不含 S4 的 v18 上同样失败（评估 slot 登记旧问题），非本轮引入。

## 闸门结果
- G2（tmux e2e-export，日志 v1-store/logs/e2e-export.log，HEAD 817551e，00:04–00:09 美东）：
  - `EXPORT_RAW16=PASS rows=71316 dtype=bf16 row_bytes=294912 demo=35913 exec=35403 budget_max=141 budget_limit=160 mismatch=0`，table_sha256=07f3766fdb28ab5f71bf30d4ff6eb5614747f350b978190cea658d5f8ccc5a2e
  - `INDEX_VS_MOTION768=PASS rows=71316 mismatches=0`
  - `S16_EVEN_VS_S32=PASS rows=36484 mismatches=0`
  - `K_DIST=43.906481,39,93,141 … over_budget=0 slot_util=27.44%`，`K_DIST_VS_AUDIT=PASS`（s16 index 文件 sha256=2b10c788…2151；审计的 d68696a0…7864 为 768 表 index，逐行相等已由 INDEX_VS_MOTION768 证明）
  - 表路径 `v1-store/datasets/4task-v2-1600ep-604f16da/motion-wan-raw9-s16`
- G1/G1b/G1c（tmux e2e-g1，GPU0，日志 v1-store/logs/e2e-g1.log，HEAD 22bc278，真 raw_bf16）：
  - `E2E_PORT_PARITY=PASS windows=2048 … depth=1 init=seed params=20252928`
  - `E2E_BF16_FLOOR=RECORDED f=5.2745e-03`
  - `E2E_PORT_PARITY=PASS windows=2048 … depth=8 init=ckpt params=69808896`
  - `E2E_BF16_REF=PASS windows=2048 fwd_rel=5.1494e-03 fwd_limit=1.5420e-02 grad_rel_max=6.5487e-03 grad_limit_min=1.5180e-02 neg_x2=FAIL_AS_EXPECTED neg_stopgrad=FAIL_AS_EXPECTED`

## 后续合入（续表）
| 子任务 | TIP | 合并提交 | 审查 |
|---|---|---|---|
| S4 在线侧 | 09f2147 | 2e4a24e | PASS/PASS findings=0 |
| S3 训练侧（替身对齐后） | 4a5e488 | 22b3eb3 | PASS findings=0 |
| S5 runner/链/评估/judge/闸门表 | 09b78f7 | 473789e | 首审 FAIL findings=1（G6 聚合吞阶段 FAIL）→ 修复复核 PASS；合并后 PASS |
| S3 DEVICE_PEAK 打印 | e9c008f | cfbde91 | PASS findings=0 |
| S5b G6 合成 benchckpt、SHM 移到正式训练 | 876d327 | c8558b5 | PASS findings=1（shmmon 判定静默丢失）→ 修复复核 PASS |
| F1 encoder 块改字符串键 | 4802d19 | 88fd4e2 | PASS findings=0；三次合并后审查 PASS（181 passed） |
| F2 runner XLA 显存预占 0.90 | 07beb7b | 488d2eb | PASS findings=0 |

## 用户中途裁定（2026-11-18 美东）
- 原话：「问题一和三能不能够放在一起来做？然后问题二能不能够在正式起跑的时候再来做，就不要单独跑这个」「你自己来裁决这个问题，尽可能不要单独的跑这个两千步。」→ G6 改为单次 `--benchckpt`（300 步、--save-interval 250 --keep-period 250、步速窗口 50–240）；SHM 判定移到正式训练前 2000 步（shmmon --formal-rec → judge shm-formal；FAIL 时主会话按 RB-D2 以 E2E_PREFETCH_FACTOR=1 从头重起，不问用户）。

## 计划外事件与处置（工程故障，自行修复）
1. G2 H5：G5 最长局 `BinFill_ep111_seed15100.h5` 本机缺失（原始目录 v1-store/raw-h5/4task-20260912-v2 已不在）。从公开 bucket `HongzeFu/robomme-4task-h5-20260912-v2` 只读下载 `record_dataset_BinFill_hard.h5.tar.xz`（18,411,696,984 B，LFS sha256 6e8fa8e6…5b53；注意 xet_manifest 里的 b167d790… 是 xetHash 不是 sha256），解出该局（sha256 9d57e21e…30a2，1,529,131,008 B，与 episode_map 一致）放回 raw-h5 原位，压缩包已删。日志 v1-store/logs/e2e-h5fetch.r2.log `H5_FETCH=PASS`。
2. G4 a01 失败：真实权重加载 `_merge_params` 的 `flatten_dict(sep="/")` 遇 encoder 块 list 整数键 TypeError（CPU 冒烟用 NoOpWeightLoader 未覆盖）。F1 改字符串键容器 `blocks/block_<i>`，encoder 初值改前改后逐位相同（ENC_INIT_BITEQ=PASS depth 1/8），真实 pi05_base 加载 PASS。G1/G1b/G1c/G3 在 88fd4e2 上重跑，数值与首轮逐位相同（v1-store/logs/e2e-g1g3-r2.log，E2E_G1G3_R2_ALL_RC=0）。
3. G4 a02 失败：第一训练步 8 卡 NCCL `Cuda failure 2 'out of memory'`，进程挂死占 649 GB，按确切 PID SIGKILL 清理（核实 cwd 与 --exp-name e2e-smoke20）。诊断 run `e2e-memdiag-a02`（同 TRAIN_ARGS、预占 0.90、3 步）通过，`DEVICE_PEAK` 每卡 19,759,631,360 B（19.76 GB）——程序本身很小，问题是 0.95 预占后池外（NCCL/cuBLAS）不足。F2 把 e2e runner 预占改为 `${E2E_XLA_MEM_FRACTION:-0.90}`（非训练语义；GPU_PEAK 判据本即 72 GB=80×0.9）。诊断第 0 步：loss=0.0888（MotionMEM 80k 步 0 同为 0.0888）、grad_norm=1.1248、e2e_enc_grad_norm=0.0652。（e2e-memdiag-a01 因诊断脚本漏 source paths.sh 找不到 pi05_base，是主会话脚本错误。）
4. 失败尝试的产物改名保留（未删）：`train-runs/.../e2e-smoke20.fail-a01|.fail-a02`、`bench/e2e-smoke20.fail-a01|.fail-a02`、`logs/e2e-smoke20.runner.log.fail-a01|.fail-a02`；诊断 run `train-runs/.../e2e-memdiag-a01|a02`。RB-E1 清理时一并列给用户。

## 闸门结果（续）
- G0（v1-store/logs/e2e-g0.log）：`DATA_LINEAGE=PASS encoder=wanvae-9980d252+e2e-s16 … files=4 mismatches=0`
- G9（v1-store/logs/e2e-g9.log，cfbde91；F1 后在 worktree 复跑同样 PASS）：`INIT_EQ=PASS leaves_compared=65 mismatches=0 encoder_leaves=10 default_rng_count_before=8 after=8`、`R3_OFF_EQ=PASS yamls=4 leaves=256 mismatches=0 base=dd39f3d`
- G3（v1-store/logs/e2e-g1g3-r2.log，88fd4e2）：`E2E_MEM_FP32=PASS rel_rmse=1.53039e-06 limit=1e-3`、`E2E_MEM_VS_MOTIONMEM=PASS windows=2048 eps=0.00496768 eps_ref_fwd=0.00522709 rel_rmse_bf16=0.00601303 limit=0.0206489 mask_mismatch=0 order_mismatch=0 depth=8`
- G4（v1-store/logs/e2e-g4-a03.log，488d2eb，XLA_MEM_FRACTION=0.90）：`CKPT_LOAD=PASS step=19 leaves=75 enc_leaves=10 resolved_sha256=15454805…a63b`、`E2E_TRAINABLE=PASS groups_grad_nonzero=4/4 frozen_hits=0`、`SMOKE20=PASS steps=20 save_load=PASS`、DEVICE_PEAK 19.76 GB/卡、EXIT_CODE=0

## 本轮 tmux 会话清单
e2e-export、e2e-g1、e2e-h5fetch、e2e-g9、e2e-g0、e2e-g3-a01、e2e-g4-a01、e2e-g4-a02（挂死，已 kill-session 并按 PID 清理）、e2e-g1g3-r2、e2e-memdiag-a01、e2e-memdiag-a02、e2e-g4-a03、e2e-benchckpt-a01

## G6 / G5b / G5 与 F3、F4（2026-11-18 美东凌晨）
- G6（v1-store/logs/e2e-benchckpt-a01.log，488d2eb，XLA_MEM_FRACTION=0.90，prefetch=2，num_workers=16）：
  - `BENCH300=PASS s_per_step=1.1166 limit=2.27 dl_wait_frac=0.0647 window=steps_50_240`
  - `GPU_UTIL=PASS util_mean=99.22 limit=85`
  - `GPU_PEAK=PASS peak_gb=18.40 limit=72`
  - `SMOKE_CKPT=PASS step=250 updates_after=49 bytes=11960266502`
  - `SHM_PEAK=DEFERRED reason=user_2026-11-18_formal_first2000`
  - `G6=PASS pathcfg_sha=d89d7351db6c31cb plan=benchckpt`
  - 外推 50,001 步 × 1.1166 s ≈ 15.5 h（Wan-9 1.133 s/步、16 h）。
- G5b（a01 全片 check_checkpoint None 叶 AttributeError → F3；a02 一片被 run_shard 归属检查误杀 → F4；a03 PASS，v1-store/logs/e2e-g5b-a03.log）：
  - `EVAL692_PLAN_IDENTITY=PASS manifest_sha=2d594d715b77d409 shards=32 union=692 per_group=14/14`
  - `E2E_ONLINE_PERF=PASS p95_ms=508.7 shards_per_gpu=4 peak_gb=77.38`（infer_p50 342.9 ms；最重 4 局墙钟 968–1043 s；motion_downsample 0）→ 评估按 32 片、每卡 4 片、EVAL_MEM_FRACTION=0.2。
- F3（60e9b60 → ca98ad2）：check_checkpoint 参数树比对两侧剔除 None 叶；真实 e2e-smoke20/19 `PARAM_TREE_EXACT=PASS n_model=75 n_ckpt=75 missing=0 extra=0 shape_mismatch=0 none_leaves_model=7 none_leaves_ckpt=7`。
- F4（2ad7c46 → ba6a264）：run_shard.sh 归属检查对「回溯失败且 /proc 已无」的已退出进程不再判外来（存活外来进程仍拒）；local_692_pool_e2e.sh run_slot 先登记后 exec 起跑。三次合并后审查 POST_MERGE_REVIEW=PASS（135 passed）。旧 local_692_pool.sh 未改。test_s3_eval_local.py c4/c5/c7 基线即失败（夹具仍按 10 片/每卡 2 片），未改。
- G5（v1-store/logs/e2e-g5-a01.log，GPU1 策略 + GPU2 sidecar）：`TIC_ACCEPTANCE_COVERAGE=PASS tasks=4/4 max_visible_k=141 long_c_env=1151/1151`；逐步 obs_mis=0、act T==I True（进行中，判定行待补）。

## G5 用户裁定提前结束（2026-11-18 美东约 03:30）
- 用户原话：「这个问题我们之前验证过嘛？验证过了就不用再跑了，直接开始吧。」
- 主会话如实答复：e2e 这条在线通路此前没有完全同口径验证过（Wan-9 的 G5 验过同一 sidecar 通路，但为 fp32、stride 32；bf16 转换另由 G1c 与单测逐字节对拍）。按用户裁定停掉 G5（tmux kill-session '=e2e-g5-a01'，无残留进程）。
- 已完成部分（v1-store/logs/e2e-g5-a01.log）：`TIC_ACCEPTANCE_COVERAGE=PASS tasks=4/4 tail_pad_windows>=1 max_visible_k=141 long_c_env=1151/1151`；逐步 159 个决策点（BinFill:367 72 步〔全库最长局，含补尾窗〕、BinFill:357 65 步、RouteStick:366 22 步）全部 `obs_mis=0 tok_eq=True act T==I True`，帧 rel_fro S-B≈0.0046；VideoRepick、VideoUnmaskSwap 两局未跑到。G5 判定记为 `RECORDED`（用户裁定提前结束，未出 TIC_SIDECAR 判定行），不是 PASS。

## 正式训练起跑（RB-E0/E2）
- 起跑前 G6 报告已向用户给出（见上 G6 节）；按一级授权不等回复。
- e2e-shmmon（--limit-gb 293、--formal-rec、--judge-step 2000）先起，随后 e2e-train 链脚本；TRAIN_HEAD = 本 Beta commit。
- 评估档：32 片、每卡 4 片、EVAL_MEM_FRACTION=0.2、CHUNK_EPISODES=5、EVAL_TIMEOUT=14400（链脚本缺省值）。

## 正式训练早期记录（2026-11-18 美东）
- 03:24:18 起跑（FORMAL_LAUNCHED head=e84f472，commitV18.4Beta）；tmux e2e-train（链脚本）、e2e-shmmon。
- Step 0：loss=0.0888（MotionMEM 80k 同为 0.0888）、grad_norm=1.1299（MotionMEM 1.1137）、e2e_enc_grad_norm=0.0656（input_proj 0.0629 / blocks 0.0130 / output_proj 0.0135 / final_norm 0.0003）、clip_coef=0.8850。
- Step 100：loss=0.0564、grad_norm=0.4595（MotionMEM 0.0565 / 0.4598）。Step 2000：loss=0.0115、grad_norm=0.0726、enc update/param=2e-4。
- 步速（metrics.jsonl 第 100–500 步）1.1161 s/步，外推剩余约 15.35 h。
- SHM（用户裁定的正式训练前 2000 步判定）：`SHM_PEAK=PASS peak_gb=230.1 limit=330 growth_last1500=0.00% window=formal_steps_0_2000 samples=167`，`SHM_FORMAL_JUDGED rc=0`；无需 RB-D2 退路。GPU 利用率采样 99%。
- 注意：训练期间只提交文档（runner 只在起跑时核 HEAD；评估分片要求工作区 clean）。
