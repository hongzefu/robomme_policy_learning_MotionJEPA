# orig100 context+motion 59999 步官方 800 局评估结果

## ① 一句话结论

59999 步 context 注入 + motion 模型在官方 800 局上成功 **311/800 = 38.88%**，比论文 RoboMME（arXiv 2603.04639）Table 3 中 **FrameSamp + Context 30.68 ± 1.17** 高约 8.2 个百分点；16 个任务里 11 个高于论文、5 个低于论文。本轮只跑了一个 seed（42），每任务 50 局，单任务的二项标准误约 6–7 个百分点。

最终判定：`COMPUTE_DONE completed=800 missing=0 errors=0 uncertain=0`；10 个 worker 均 `EXIT_CODE=0`；`VLA_VERIFY=PASS step=59999 files=36 sha_mismatch=0`；10 份占位逐个 `CANCELLED`，squeue 为空。

## ② 逐任务成功率与论文对照

论文数值取自 arXiv HTML 版 Table 3（主表）与附录逐任务表的 FrameSamp / Context 行，经本机从原文 HTML 抽取核对。

| 任务 | 本轮成功 / 50 | 本轮 % | 论文 FrameSamp+Context % | 差（百分点） | 本轮超时局 |
|---|---|---|---|---|---|
| BinFill | 12 | 24.0 | 41.22 ± 4.27 | −17.2 | 0 |
| PickXtimes | 34 | 68.0 | 72.00 ± 3.74 | −4.0 | 0 |
| SwingXtimes | 48 | 96.0 | 73.67 ± 11.08 | +22.3 | 0 |
| StopCube | 36 | 72.0 | 13.67 ± 8.82 | +58.3 | 0 |
| VideoUnmask | 15 | 30.0 | 26.89 ± 5.28 | +3.1 | 0 |
| ButtonUnmask | 16 | 32.0 | 30.22 ± 3.52 | +1.8 | 0 |
| VideoUnmaskSwap | 13 | 26.0 | 20.89 ± 2.94 | +5.1 | 0 |
| ButtonUnmaskSwap | 7 | 14.0 | 15.22 ± 3.93 | −1.2 | 2 |
| PickHighlight | 9 | 18.0 | 17.67 ± 2.83 | +0.3 | 3 |
| VideoRepick | 9 | 18.0 | 15.22 ± 4.08 | +2.8 | 0 |
| VideoPlaceButton | 17 | 34.0 | 30.00 ± 4.68 | +4.0 | 0 |
| VideoPlaceOrder | 9 | 18.0 | 20.89 ± 6.28 | −2.9 | 2 |
| MoveCube | 44 | 88.0 | 77.22 ± 7.87 | +10.8 | 5 |
| InsertPeg | 1 | 2.0 | 1.22 ± 1.63 | +0.8 | 16 |
| PatternLock | 16 | 32.0 | 15.22 ± 2.34 | +16.8 | 0 |
| RouteStick | 25 | 50.0 | 19.67 ± 3.67 | +30.3 | 0 |
| **平均** | **311 / 800** | **38.88** | **30.68 ± 1.17** | **+8.2** | 28 |

论文同表的其他行供参照：Perceptual TokenDrop + Context 34.50 ± 0.89，FrameSamp + Modul 44.51 ± 0.77，FrameSamp + Expert 36.25。计划 K5 写的「官方 44.51±0.77」实为 FrameSamp + Modul 行；本模型是 context 注入，同口径对照应为 30.68 ± 1.17，此处更正。

## ③ 运行过程与计划外事件

- 冒烟（5000 步，2 局）通过后释放占位 65037493；59999 于美东 10-29 22:04 上传，随即提交 12 份占位并以 `--size-only` 下载，22:26 下载完成即开跑。
- worker 数 12 → 10：Aspen 到 GreatLakes 的 ControlMaster 只允许 10 个并发会话，第 11、12 个 worker 被拒（`Session open refused by peer`）。用户原话：「如果它最多就只能支持十个对话，那我们就用十个塔来跑，其他的就不要管」「其他任务就可以释放了」。65048197、65048198 经 sled-vail 的连接 scancel。
- 控制器 Aspen → sled-vail，首轮作废重跑：Aspen 本机读 turbo 小文件约 200 ms／个（45 个文件 9 s，sled-vail 0.2 s、GreatLakes 0.1 s），控制器回执超过 worker 默认 120 s 超时，10 个 worker 中 7 个以 `TimeoutError` 退出。对抗验证 `ADJUST_REVIEW=PASS scale=same findings=1` 后把同一控制器（代码、spec、参数不变）挪到 sled-vail。用户原话：「你修复完了之后重新开始跑，现在的结果作废」。首轮目录 `run800/` 保留但作废；正式结果来自全新目录 `run800-r2/`，22:41 起跑，美东 10-30 00:09 `COMPUTE_DONE`，约 1 小时 28 分。
- 后台 sha256 校验与评估并行，`VLA_VERIFY=PASS`，结果有效。

## ④ 产物位置

- 逐局记录：`$R/run800-r2/accepted/*.json`（800 个），汇总 `$R/run800-r2/queue/summary.json`，视频在 `$R/run800-r2/attempts/*/client/videos/`。
- 控制器日志 `$R/run800-r2/logs/controller-vail.log`；worker 日志本机 `$L/r2-worker-NN.log`。

其中 `$R=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store/evaluation/official-stage2-1026/o16ctx-60k-seed42`，`$L=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/o16ctx-60k`。
