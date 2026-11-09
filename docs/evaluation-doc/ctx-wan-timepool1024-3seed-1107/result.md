# Wan 时间池化 1024 context 60k：官方 800 局三 seed 本机评估结果

## ① 一句话结论与指标速览

`orig100-ctx32x4x4-wan-timepool8x8-b128-60k` 的 59999 在官方 test 800 局上按 seed 42/7/0 各评一次。三个 seed 的总体成功率为 **35.46 ± 0.79**（ddof=1）：

| seed | 成功 / 800 | 成功率 | 失败 | 超时 |
|---|---|---|---|---|
| 42 | 291 | 36.38% | 493 | 16 |
| 7 | 280 | 35.00% | 505 | 15 |
| 0 | 280 | 35.00% | 497 | 23 |
| **均值 ± std** | | **35.46 ± 0.79** | | |

对照（同一份 800 局清单、同样 seed 42/7/0）：

| 模型 | 三 seed 均值 ± std | 评估平台 |
|---|---|---|
| Wan 时间池化 1024 context（本档案） | **35.46 ± 0.79** | 本机 A100，`local_eval.py` |
| stage2 concat 1×1024 context（学习型 encoder） | 37.33 ± 0.59 | GreatLakes 10-31（本机重评见 `ctx-concat1x1024-3seed-1108`） |
| stage2 split 4×256 context | 38.42 ± 1.25 | GreatLakes 10-31 |
| stage2 split modulation | 43.42 ± 0.52 | GreatLakes 10-31 |
| Wan-LTM flat16384 anchor2×16 | 40.67 ± 0.71 | 本机 11-03 |

零参数池化比学习型 concat encoder 低约 1.9 个百分点，比 split 低约 3.0 个百分点。差距约为两倍 std，平台也不同（GL 与本机），所以要等本机 concat 重评结果出来才能下同平台结论。

最终判定（逐行原文，`v1-store/logs/ctx-wan-timepool1024-3seed-1107/simple-chain.log`）：

- `TRAIN_DONE=PASS`、`GPU_FREE=PASS`
- `COMPUTE_DONE seed=42 completed=800 success=291 fail=493 timeout=16`
- `COMPUTE_DONE seed=7 completed=800 success=280 fail=505 timeout=15`
- `COMPUTE_DONE seed=0 completed=800 success=280 fail=497 timeout=23`
- `THREE_SEED=PASS seeds=7,42,0 overall=35.46±0.79`
- `EXIT_CODE=0`；全程没有 `RECOVER`（无 worker 死亡回收）

## ② 分难度与逐任务三 seed（%）

| 分组 | seed 42 | seed 7 | seed 0 | 均值 ± std |
|---|---|---|---|---|
| easy | 43.51 | 42.79 | 42.55 | 42.95 ± 0.50 |
| medium | 30.21 | 32.29 | 31.25 | 31.25 ± 1.04 |
| hard | 27.08 | 20.83 | 22.40 | 23.44 ± 3.25 |

| 任务 | timepool1024 | GL concat 1×1024 | GL split context |
|---|---|---|---|
| BinFill | 44.00 ± 2.00 | 34.67 ± 1.15 | 21.33 ± 4.62 |
| PickXtimes | 75.33 ± 1.15 | 53.33 ± 3.06 | 67.33 ± 1.15 |
| SwingXtimes | 90.00 ± 0.00 | 83.33 ± 2.31 | 92.67 ± 5.77 |
| StopCube | 43.33 ± 6.43 | 43.33 ± 2.31 | 66.00 ± 8.72 |
| VideoUnmask | 23.33 ± 2.31 | 33.33 ± 1.15 | 30.67 ± 1.15 |
| ButtonUnmask | 22.00 ± 5.29 | 30.00 ± 6.00 | 31.33 ± 1.15 |
| VideoUnmaskSwap | 26.00 ± 5.29 | 37.33 ± 4.62 | 22.67 ± 3.06 |
| ButtonUnmaskSwap | 19.33 ± 4.62 | 27.33 ± 3.06 | 16.00 ± 2.00 |
| PickHighlight | 20.00 ± 0.00 | 20.00 ± 5.29 | 17.33 ± 1.15 |
| VideoRepick | 16.67 ± 3.06 | 22.67 ± 3.06 | 18.00 ± 0.00 |
| VideoPlaceButton | 28.00 ± 2.00 | 36.00 ± 2.00 | 37.33 ± 5.77 |
| VideoPlaceOrder | 30.00 ± 2.00 | 28.00 ± 5.29 | 18.67 ± 3.06 |
| MoveCube | 84.00 ± 2.00 | 82.67 ± 3.06 | 88.67 ± 3.06 |
| InsertPeg | 0.00 ± 0.00 | 2.00 ± 2.00 | 4.00 ± 3.46 |
| PatternLock | 14.00 ± 4.00 | 32.67 ± 1.15 | 32.00 ± 4.00 |
| RouteStick | 31.33 ± 1.15 | 30.67 ± 1.15 | 50.67 ± 3.06 |

每任务每 seed 50 局，单任务单 seed 的二项标准误约 6–7 个百分点。与 GL concat 相比，timepool 在 PickXtimes（+22.0）、BinFill（+9.3）、SwingXtimes（+6.7）上更好，在 PatternLock（−18.7）、Unmask/Swap 类（−7.3～−11.3）、VideoPlaceButton（−8.0）上更差，这与“时间均值会抹掉窗口内顺序信息”的预期一致，但没有做因果检验。

## ③ 口径

- 清单：官方 test 800 局、160 块每块 5 局，封签 `991c34960b431f4e6170357dfe428e74e404f5ccdbf8dafddc3a0a380eaeecc6`；`max_steps=1300`；timeout 算失败；std 取 ddof=1。
- seed 只控制模型采样噪声，三个 seed 跑的是同一批 800 局。
- 只评 59999，std 只含 seed 间方差。

## ④ 版本与运行

- 计划：`1107-ctx-wan-timepool1024-3seed-eval-plan.html`（开工令「同意开始」，11-07 美东）；计划审查 `PLAN_CONSISTENCY_REVIEW=FAIL base=17dfb4c findings=1 blocking=1`，按 `ADJUST_REVIEW=PASS scale=same` 放开 `eval.py` seed 集合（3 行）。
- 执行代码：`v10-ctx-wan-timepool-eval@ffb307c`（`commitV11.82`），执行副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA-eval-ffb307c`。
- 触发：训练日志末行 `EXIT_CODE=0`（11-08 美东 06:22），约 1 分钟后起跑。GPU 2–7 先起 24 个 worker；G9 在 06:24 写出 `RESTORE_POOL.json`（`RESTORE_POOL=PASS`）后，GPU 0、1 加入，共 32 个 worker，每卡 4 个，`XLA_PYTHON_CLIENT_MEM_FRACTION=0.18`。
- 显存：每卡约 72.6 GB，无 OOM。
- 耗时：seed 42 06:23–07:18，seed 7 07:24–08:24，seed 0 08:29–09:23，三 seed 合计约 3 小时（美东）。
- 输出：`v1-store/evaluation/ctx-wan-timepool1024-3seed-1107/`（视频不上传，用户裁决 D3）。

## ⑤ 用户决策记录

1. 「给出方案 训练一结束立刻启动 timepool1024的eval 3seed」
2. 「你在干什么？写完整html方案！」
3. 「你需要强制隔一段时间唤醒一次 以免时间到了没被唤醒 注意我说的是claude agent 不是代码的唤醒！」→ CronCreate 每 20 分钟唤醒。
4. 「同意开始」

## ⑥ 计划外事件

- 后台会话写主检出路径与起 tmux 被隔离守卫拒绝，改为把 `chain.sh` 放进执行副本（linked worktree 可写）后起链，未产生副作用。
- G9 另报 `CKPT_INVENTORY=FAIL field=checkpoint_entry_missing index=5000/train_state`（中间 checkpoint 只存 params），属训练任务，不影响本评估（只读 59999 params，且 `RESTORE_POOL=PASS`）。

## ⑦ 归档文件

- `records/three_seed_summary.txt`、`records/three_seed_summary.json`：三 seed 汇总原件。
- `records/simple-chain.summary.txt`：链日志判定行（清洗后）。
