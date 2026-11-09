# 双消融三 seed 官方评估结果

## ① 一句话结论与指标速览

两条 GreatLakes 训练的双消融模型（b64/80k，最终 checkpoint 79999）在官方 test 800 局上按官方三个模型 seed（7、42、0）各评一次，6 次全部完成。总体成功率三 seed 均值 ± 样本标准差（ddof=1，同官方 `compute_results.py` 与 10-31 口径）：

| 模型 | seed 42 | seed 7 | seed 0 | 三 seed 均值 ± std |
|---|---|---|---|---|
| N：noxyz（编码器去掉 depth/xyz，`gl-orig100-ctx32x4x4-noxyz-motion-s2split4x256-b64-80k`） | 276/800 = 34.50% | 299/800 = 37.38% | 308/800 = 38.50% | **36.79 ± 2.06** |
| S：nostate（编码器去掉 action/state，`gl-orig100-ctx32x4x4-nostate-motion-s2split4x256-b64-80k`） | 263/800 = 32.88% | 291/800 = 36.38% | 274/800 = 34.25% | **34.50 ± 1.76** |

探索性对照（训练制度不同，按计划 K6 不作严格比较）：10-31 的 context split（编码器全输入，b128/60k，AWS 训练）三 seed 为 **38.42 ± 1.25**。两个消融都低于它：noxyz 低 1.6 个百分点（在两边标准差之内），nostate 低 3.9 个百分点。两个消融之间相差 2.3 个百分点，与 seed 间波动同量级。

最终判定（逐行原文，见 `records/three_seed_verdicts.txt`）：

- `PLAN_CONSISTENCY_REVIEW=FAIL base=77a14a4 findings=9 blocking=2`（F1 交用户裁决，其余文字统一，修订 `857c746`）
- `PRE_MERGE_REVIEW=PASS base=8b52210… tip=4d70b1a… files=2 commits=3 findings=1`；`ADJUST_REVIEW=PASS scale=same findings=2`
- `EVAL_CODE=PASS sha=4d70b1a tests=seed_gate7+official_contract38+eval_target_contract12+runtime_protocol24+runtime_restore23+simple_queue3 eval_dirs_diff_vs_5fce5d3=0 cfg=mme_vla_suite_orig16_b64_80k`（放宽前原判 `FAIL tests=325passed`）
- `RELEASE_SNAPSHOT=PASS commit=4d70b1a files=4755 tree_diff=0`
- `CONTRACT=PASS model=N sha256=bb4d5603…`；`CONTRACT=PASS model=S sha256=b85d72c8…`；`BUILD_RUNS=PASS runs=8`
- `SMOKE2=PASS model=N accepted=2 errors=0 readout=stage2_pool4_split_v1`；`SMOKE2=PASS model=S accepted=2 errors=0 readout=stage2_pool4_split_v1`
- `LAYOUT_SPEED=PASS a_eps_per_h=236.2 b_eps_per_h=225.9 ratio=0.957 choose=1x`
- 6 行 `COMPUTE_DONE completed=800 missing=0 errors=0 uncertain=0`；`retries` 仅 N-s7 的 `VideoPlaceButton:19` 一次（基础设施中断后重试，计划允许的上限）
- 6 行 `SEED_ECHO=PASS records=800 identity_mismatch=0`
- `THREE_SEED=PASS model=N(noxyz) seeds=7,42,0 runs=3 overall=36.79±2.06`；`THREE_SEED=PASS model=S(nostate) seeds=7,42,0 runs=3 overall=34.50±1.76`
- `HOLDS_RELEASED=PASS cancelled=2 squeue_empty=1`

## ② 逐任务三 seed 均值 ± std（%）

每任务每 seed 50 局，单 seed 单任务二项标准误约 6–7 个百分点。原值见 `records/three_seed_summary.json`。

| 任务 | N（noxyz） | S（nostate） | 10-31 context split（参照） |
|---|---|---|---|
| BinFill | 24.00 ± 5.29 | 20.67 ± 4.62 | 21.33 ± 4.62 |
| PickXtimes | 86.67 ± 7.02 | 56.67 ± 11.02 | 67.33 ± 1.15 |
| SwingXtimes | 89.33 ± 1.15 | 66.67 ± 1.15 | 92.67 ± 5.77 |
| StopCube | 42.67 ± 5.03 | 57.33 ± 7.57 | 66.00 ± 8.72 |
| VideoUnmask | 30.00 ± 2.00 | 25.33 ± 3.06 | 30.67 ± 1.15 |
| ButtonUnmask | 28.00 ± 2.00 | 27.33 ± 5.77 | 31.33 ± 1.15 |
| VideoUnmaskSwap | 18.00 ± 6.00 | 23.33 ± 1.15 | 22.67 ± 3.06 |
| ButtonUnmaskSwap | 21.33 ± 6.11 | 20.00 ± 6.00 | 16.00 ± 2.00 |
| PickHighlight | 18.00 ± 3.46 | 17.33 ± 4.16 | 17.33 ± 1.15 |
| VideoRepick | 18.00 ± 2.00 | 16.67 ± 3.06 | 18.00 ± 0.00 |
| VideoPlaceButton | 32.00 ± 5.29 | 35.33 ± 4.16 | 37.33 ± 5.77 |
| VideoPlaceOrder | 24.67 ± 3.06 | 26.00 ± 7.21 | 18.67 ± 3.06 |
| MoveCube | 86.67 ± 2.31 | 83.33 ± 3.06 | 88.67 ± 3.06 |
| InsertPeg | 2.00 ± 0.00 | 6.00 ± 2.00 | 4.00 ± 3.46 |
| PatternLock | 28.00 ± 2.00 | 34.00 ± 3.46 | 32.00 ± 4.00 |
| RouteStick | 39.33 ± 2.31 | 36.00 ± 2.00 | 50.67 ± 3.06 |
| **总体** | **36.79 ± 2.06** | **34.50 ± 1.76** | **38.42 ± 1.25** |

差异最明显的几项（探索性，单任务波动大）：
- nostate 在计数/节奏类任务上掉得多：PickXtimes 56.67（noxyz 86.67、参照 67.33），SwingXtimes 66.67（noxyz 89.33、参照 92.67）。
- noxyz 在 StopCube 上掉得多：42.67（nostate 57.33、参照 66.00）。
- 两个消融在 RouteStick 上都明显低于参照（39.33 / 36.00 对 50.67）。

难度分档（成功/局数）：

| 运行 | easy | medium | hard |
|---|---|---|---|
| N-s42 | 179/416 | 57/192 | 40/192 |
| N-s7 | 193/416 | 60/192 | 46/192 |
| N-s0 | 198/416 | 67/192 | 43/192 |
| S-s42 | 172/416 | 58/192 | 33/192 |
| S-s7 | 181/416 | 68/192 | 42/192 |
| S-s0 | 177/416 | 54/192 | 43/192 |

超时局数（每次 800 局）：N 30／27／30（seed 42／7／0），S 24／31／32。

## ③ 口径说明

- seed 7／42／0 与 10-31 相同；seed 只控制模型采样噪声，三个 seed 跑的是同一批 800 局。
- 只评 79999；std 只含 seed 间方差，与论文 Table 3（另含 checkpoint 维）不完全可比。
- 与 10-31 context split 的对照是探索性的：batch（64 对 128）、步数（80k 对 60k）、训练硬件（A40 对 A100）、EMA 样本尺度都不同（训练计划 K12）。
- 训练侧 V10 的完整在线握手 `CKPT_DELIVERY_READY=FAIL ×2`（跨 A6000/A40 numerics 与离线来源不同）未裁决；用户原话「暂时不管这个问题了开始了」放行。评估的 runtime 合同本身为 `cross_hardware_numeric=not_required`，与 10-31 口径一致；本结果不构成对该 FAIL 的判定。

## ④ 运行过程

时间均为美东。

| 运行 | 起跑 | `COMPUTE_DONE` | 用时 | 布局 |
|---|---|---|---|---|
| 冒烟 N、S（各 2 局） | 11-08 18:16 | 约 18:40 | — | A 份卡 0／卡 1 各 1 worker |
| N-s42 | 18:46:42 | 20:41 | 6949 s | A 每卡 1（4 CPU）、B 每卡 2（2 CPU），12 worker，兼测速 |
| S-s42 | 20:42:32 | 22:35 | 6803 s | 每卡 1、4 CPU，8 worker |
| N-s7 | 22:35:56 | 11-09 01:09 | 9187 s（含两次故障停顿） | 同上；worker 重起两次（`-r1`、`-r2`） |
| S-s7 | 01:09:04 | 02:47 | 5906 s | 同上 |
| N-s0 | 02:47:51 | 04:22 | 5667 s | 同上 |
| S-s0 | 04:22:38 | 05:59 | 5785 s | 同上 |

资源：复用训练占位 65209321（gl1519）与 65209330（gl1503），各 4×A40 / 16 CPU / 128G；冒烟起至 06:00:30 逐个 `scancel`，评估实际占用约 11.7 小时 × 8 卡 ≈ 94 GPU·h。未新提交任何 job。

## ⑤ 计划外事件与处置（实施中调整）

1. **上游门放行**：用户「暂时不管这个问题了开始了」，V0 不等上游 `done`，以占位无训练 step 为准。
2. **V1 验收放宽**：`tests/evaluation` 31 条失败为既有（与 `5fce5d3` 逐字节相同），经 `ADJUST_REVIEW=PASS scale=same` 改为定向测试 + 四目录零差异。
3. **turbo 读写改在 sled-vail**：aspen 读 turbo 单个小文件实测 35.36 s、vail 0.51 s。执行副本改为本地导出再经 vail rsync；首次远端比对误用 `rsync -anc`（含权限）报 4681 处权限位差异，改 `-rlnc` 只比内容后为 0。
4. **一卡两个 worker 不划算**：V5 判 `ratio=0.957 choose=1x`。每个 worker 只有 2 CPU 时单局墙钟约为 4 CPU 时的 2.5 倍，仿真受 CPU 限制；全程无 Killed／OOM。后五次每卡 1 个 worker。
5. **vail 访问 turbo 卡顿（11-08 23:19–23:28）**：N-s7 控制器在 vail 上进入 `D` 状态，`state.json` 停更，8 个 worker 因 `TimeoutError: 控制器响应未确认`（`controller_timeout_seconds` 默认 120 s）先后退出；用新 worker-id（`-r1`）重起后又全部超时退出（原文见 `records/ns7-a2.fail-timeout.txt`、`records/ns7-a3-r1.fail-timeout.txt`）。用户随后授权：「控制器挪到别的机器可以作为fallback 也可以自行启动cpu job 作为控制器 不要问我 尽可能跑完！」。处置：等 vail 旧控制器进程完全退出（锁释放）后，在占位 A 内 `srun --overlap` 起控制器（`run-controller-gl.sh`，`--slurm-host ""` 本机调 `sacct`），worker 以 `-r2` 重起。评估代码未改。
6. **GL 控制器也曾 `D` 态（11-09 00:26–01:00）**：N-s7 剩 6 块挂在第一批已退出 worker 名下，控制器停更约 34 分钟。取消该控制器 step（`65209321.33`）后重起，并对 6 个已退出 worker 逐个执行 `official_queue recover`，全部 `RECOVERED`；其中 `VideoPlaceButton:19` 计 1 次基础设施重试，正常失败未重跑。根因未查清。
7. **换根链停不掉 GL 控制器 step**：`chain2.sh` 只 kill aspen 上的 tmux 会话，GreatLakes 上的 srun step 不随 ssh 断开退出。每次换根后按启动时间核对，手动 `scancel` 上一轮控制器 step（`.39` N-s7 重起版、`.40` S-s7、`.45` N-s0），最后一个 `.50` 随占位取消一并结束。
8. **launch.md 晚提交**：计划要求起跑前提交，实际在 N-s42 起跑后数分钟补交（`48332cf`）。

## ⑥ 本轮 tmux 会话（前缀 `evab-`，均已结束或清理）

aspen：`evab-release`（导出改道时停止）、`evab-release2`、`evab-build`、`evab-w-smN`、`evab-w-smS`、`evab-w-ns42-*`（12）、`evab-chain`（迁移控制器时停止）、`evab-w-ss42-*`、`evab-w-ns7-*`、`evab-w-ns7-*-r1`、`evab-w-ns7-*-r2`、`evab-chain2`、`evab-ctlgl-o16abl-{N-seed7,S-seed7,N-seed0,S-seed0}`、`evab-w-{ss7,ns0,ss0}-*`。sled-vail：`evab-ctl-{N,S}-smoke`、`evab-ctl-o16abl-N-seed42`、`evab-ctl-o16abl-S-seed42`、`evab-ctl-o16abl-N-seed7`。清理只按精确会话名逐个执行，删前删后 `tmux ls` 差集只含目标。

## ⑦ 产物位置

- 逐局记录与视频：`$O/o16abl-<N|S>-seed<s>/run800/accepted/*.json`、`attempts/*/client/`；`$O=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store/evaluation/o16abl-80k`。
- 本机日志：`/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/o16abl-3seed/`（各 worker、控制器、换根链、导出、建根）。

## ⑧ 下一步

无自动解锁的下游。可选：分析 nostate 在 PickXtimes／SwingXtimes、noxyz 在 StopCube 的退化；或与同制度（b64/80k、GL A40）的全输入 context 模型做严格对照——目前没有这样的模型（`gl-orig100-ctx-reproduce-*` 处于停工）。均需用户另行决定。训练侧 V10 在线握手 FAIL 的裁决仍在训练任务的 `pending_decisions` 中。

## ⑨ 归档文件清单（`records/`）

`three_seed_summary.json`、`three_seed_verdicts.txt`、`three_seed_summary.py`、`build_runs.py`、`build-runs.summary.txt`、`export-release.summary.txt`、`chain-rest.summary.txt`、`chain2.summary.txt`、`ns7-a2.fail-timeout.txt`、`ns7-a3-r1.fail-timeout.txt`。
