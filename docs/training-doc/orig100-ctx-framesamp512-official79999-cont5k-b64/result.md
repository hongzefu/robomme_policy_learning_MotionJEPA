# 官方 FrameSamp+Context 512 续训 5k（b64）＋三 seed 评测：结果

## 一、一句话结论

官方 79999 权重用 batch 64 再训 5000 步（warmup 500、5e-5 常数）后，第 4999 步在本机官方 test 800 局 × 推理 seed 42/7/0 上为 **30.75 ± 0.25%**（31.00 / 30.50 / 30.75，样本标准差），与论文 Table 15 该行的 30.7（3 ckpt × 3 seed 均值）几乎相同。

## 二、训练

- 起止（美东 11-10）：冒烟 18:19–18:22，正式 18:23 起跑，19:47 `EXIT_CODE=0`；稳态约 1.0 s/步（8 卡 mesh 2×4，batch 64）。
- 冒烟判据：`TRAIN_MESH shape=(2, 4)`、`EXIT_CODE=0`、`Merging missing weight` 0 行（官方权重键全部命中）。
- loss（单 batch 记录值）：第 0 步 0.0009（grad_norm 0.0121）→ 1000 步 0.0017 → 2000 步 0.0017 → 3000 步 0.0017 → 4000 步 0.0018。第 0 步即收敛量级，确认是在官方权重上续训；warmup 结束后的平台略高于第 0 步，原因未单独诊断（AdamW 从零起步、单 batch 噪声均可能），未影响评测成绩。
- checkpoint：`v1-store/train-runs/mme_vla_suite/orig100-ctx-framesamp512-official79999-cont5k-b64/{2500,4999}`（各 11G，params + assets）。
- wandb：项目 openpi，run 名同上。

## 三、评测

- 代码 `e4b697a`（eval_queue.py 与主表 3ckpt×3seed 评测同实现，f0149d8），官方清单 sha256 `991c3496…`（800 局）；评测冒烟 BinFill:0 seed 42 `SMOKE=PASS outcome=success seconds=75`。
- 美东 19:48 起、21:45 `QUEUE_DONE done=3 blocked=0`、`EXIT_CODE=0`；三项均 errors=0 retries=0。
- 结果目录：`v1-store/evaluation/fs512cont5k-1110/fs512c-4999-seed{42,7,0}/accepted/`（每局一份 JSON）。

| 判定行 | 原文 |
|---|---|
| seed 42 | `EVAL_DONE item=fs512c-4999-seed42 success=248/800 errors=0 retries=0 status=COMPUTE_DONE completed=800` |
| seed 7 | `EVAL_DONE item=fs512c-4999-seed7 success=244/800 errors=0 retries=0 status=COMPUTE_DONE completed=800` |
| seed 0 | `EVAL_DONE item=fs512c-4999-seed0 success=246/800 errors=0 retries=0 status=COMPUTE_DONE completed=800` |

终局分布：seed 42 success 248 / fail 529 / timeout 23；seed 7 244 / 535 / 21；seed 0 246 / 533 / 21（timeout 计为失败）。

逐任务成功率（%，每格 50 局）：

| 任务 | seed 42 | seed 7 | seed 0 | 均值 |
|---|---:|---:|---:|---:|
| BinFill | 42.0 | 40.0 | 36.0 | 39.3 |
| ButtonUnmask | 26.0 | 24.0 | 26.0 | 25.3 |
| ButtonUnmaskSwap | 22.0 | 24.0 | 16.0 | 20.7 |
| InsertPeg | 0.0 | 0.0 | 2.0 | 0.7 |
| MoveCube | 82.0 | 82.0 | 80.0 | 81.3 |
| PatternLock | 22.0 | 20.0 | 20.0 | 20.7 |
| PickHighlight | 16.0 | 18.0 | 22.0 | 18.7 |
| PickXtimes | 70.0 | 62.0 | 52.0 | 61.3 |
| RouteStick | 2.0 | 8.0 | 6.0 | 5.3 |
| StopCube | 20.0 | 22.0 | 28.0 | 23.3 |
| SwingXtimes | 84.0 | 80.0 | 78.0 | 80.7 |
| VideoPlaceButton | 22.0 | 20.0 | 24.0 | 22.0 |
| VideoPlaceOrder | 18.0 | 16.0 | 18.0 | 17.3 |
| VideoRepick | 20.0 | 20.0 | 24.0 | 21.3 |
| VideoUnmask | 30.0 | 28.0 | 36.0 | 31.3 |
| VideoUnmaskSwap | 20.0 | 24.0 | 24.0 | 22.7 |
| **全部 16 任务** | **31.00** | **30.50** | **30.75** | **30.75** |

## 四、解读边界

- 三个推理 seed 共用同一训练轨迹与同一 800 个环境实例，±0.25 只反映推理采样随机性，不含训练随机性。
- 与论文 30.7 的口径不同：论文是官方 60000/70000/79999 三个 ckpt × 三 seed；本次是官方 79999 再训 5k 的单 ckpt × 三 seed、本机评测器。两者接近说明本机评测链与续训未引入明显偏差，但不能读成「续训 5k 无增益」的严格结论——缺少同一评测器下原版 79999 的对照。
- StopCube 23.3 / BinFill 39.3 与论文该行 13.67 / 41.22 同一区间，而自训 FrameSamp 1024（b64 80k）79999 为 48 / 18（seed 42）、36 / 20（seed 7）。

## 五、用户决策记录

- 「用 batch size 六十四继续训练，训练完了之后，在这个呃做三次评测。」（11-10 美东 18:0x）
- 「现在这个搞完要多久？」——答复后照原计划执行。

## 六、计划外事件

- 无。跨会话 `req=fs512cont-11101815-1` 的约定（eval3x3 队列暂停期间本会话独占 8 卡）已兑现，结束后已通知对方卡位释放。

## 七、下一步（未授权，仅建议）

- 如需判断续训本身的作用：在同一评测器上评一次官方原版 79999（三 seed 约 3 h）。
- 本地 ckpt 2500/4999（22G）与官方 zip（11.6G）+ 解压目录（11G）均保留在 `/scratch`（实例存储，stop 即丢）；是否上传 HF bucket 待用户决定。
