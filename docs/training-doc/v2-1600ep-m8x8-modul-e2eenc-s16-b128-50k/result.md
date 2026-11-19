# result.md — v2-1600ep-m8x8-modul-e2eenc-s16-b128-50k（E2E-Enc，任务 id e2eenc-s16-50k）

## ① 一句话结论与指标速览

端到端从头训练的 1 层因果 transformer motion encoder（20,252,928 参数，随机初始化、可训练，读 bf16 步长 16 的 Wan 原始 9 latent）接入 modulation VLA，b128 训练 50,001 步，本机 692 局评估 **337/692 = 48.70%**；对照 MotionMEM@50k（冻结预训练 8 层 encoder，GL 评估，跨环境参考）**378/692 = 54.62%**，低 41 局（−5.9 pp）。数值含义与稿件措辞由用户判断，本档只交结果。

| 判定 | 结果 |
|---|---|
| EVAL692 | `EVAL692=PASS episodes=692 success=337 rate=0.4870 preseeded_errors=9 new_errors=0 retried=0 expect_motion_stats=1` |
| E2E_EVAL | `E2E_EVAL=PASS episodes=692 groups=14 success=337 error=9 overflow_resample=2 motion_downsample_steps=48` |
| EVAL692_RESULT_IDENTITY | `PASS manifest_sha=2d594d715b77d409 episodes=692` |
| 训练收尾 | `CKPT_LOAD=PASS step=50000 leaves=75 enc_leaves=10`、`TRAIN_FINAL_RECORD=PASS step=50000 ckpts=10`、runner EXIT_CODE=0 |
| HF 备份 | `HF_VERIFY=PASS ckpts=10`（公开 bucket HongzeFu/robomme-4task-modul-e2eenc-s16-50k-v1；每 step 读回 sha256 对照通过后删本地 verify 副本） |

## ② 版本与代码状态
- 计划 1117-e2e-enc-train-plan.html @ c618671；实施分支 v18-4task-modul-e2eenc-s16-50k。
- 训练锚点 commitV18.4Beta = e84f47218cd167df61309269bc32843691afe799；本结论 commit 为 commitV18.4。
- 合入与修复清单、闸门判定原文见 launch.md。

## ⑥ 硬件与耗时
- 8 × A100-80GB；训练美东 2026-11-18 03:24:18 → 19:03（约 15.6 h），1.116 s/步，GPU 利用率约 99%，每卡显存峰值 18.4 GB（XLA 预占 0.90），/dev/shm 峰值 230.1 GB（SHM_PEAK=PASS，正式训练前 2000 步判定）。
- 评估 32 片、每卡 4 片、EVAL_MEM_FRACTION=0.2：美东 19:03 → 20:53（LOCAL692_DONE 2026-11-19T01:53:58Z），约 1 h 50 m（Wan-9 同配置 73 min）。慢在 motion 编码：每决策步 add_buffer 均值 11.7 s（Wan-9 4.1 s；步长 16 窗数翻倍 + 同卡 4 片争抢 GPU），infer 均值 430 ms（Wan-9 283 ms；每次推理重算 160 槽 encoder）。

## ⑦ 训练过程行为
- Step 0 loss=0.0888（MotionMEM 80k 同为 0.0888）、grad_norm=1.1299；Step 100 loss 0.0564 / grad_norm 0.4595（MotionMEM 0.0565 / 0.4598）。
- loss：2000 步 0.0115 → 10000 步 0.0052 → 25000 步 0.0029 → 50000 步 0.0019；500 步后不再裁剪。
- encoder 四组梯度全程非零：e2e_enc_grad_norm 0.0656（步 0）→ 0.0010（步 50000），update/param 比稳定在 4–5e-4。
- 逐 1000 步摘要 records/train.summary.log（50 行），逐 100 步指标 records/metrics.jsonl。

## ⑧ 训练后评估（14 组，逐局记录口径，与 MotionMEM@50k 同局对比）

逐局记录口径 E2E 合计 346、MotionMEM@50k 合计 387；官方统计把两边相同的 9 个预标 error 局计为失败，得 337 与 378，差距同为 41 局。

| 任务 / 难度 | 局数 | E2E-Enc | MotionMEM@50k | 差 |
|---|---|---|---|---|
| BinFill easy | 49 | 44（90%） | 44（90%） | 0 |
| BinFill medium | 49 | 33（67%） | 44（90%） | −11 |
| BinFill hard | 44 | 16（36%） | 23（52%） | −7 |
| RouteStick easy | 50 | 49（98%） | 50（100%） | −1 |
| RouteStick medium | 50 | 49（98%） | 47（94%） | +2 |
| RouteStick hard | 50 | 43（86%） | 41（82%） | +2 |
| RouteStick xhard | 50 | 22（44%） | 29（58%） | −7 |
| VideoRepick easy | 50 | 15（30%） | 23（46%） | −8 |
| VideoRepick medium | 50 | 11（22%） | 22（44%） | −11 |
| VideoRepick xhard | 50 | 25（50%） | 26（52%） | −1 |
| VideoUnmaskSwap easy | 50 | 17（34%） | 8（16%） | +9 |
| VideoUnmaskSwap medium | 50 | 12（24%） | 17（34%） | −5 |
| VideoUnmaskSwap hard | 50 | 3（6%） | 7（14%） | −4 |
| VideoUnmaskSwap xhard | 50 | 7（14%） | 6（12%） | +1 |

- MotionMEM@50k 逐局来源：非 BinFill 11 组 docs/training-doc/primary700-motion50k-gl/records/merged/progress.json，BinFill 3 组 docs/training-doc/binfilldemo-motion50k-gl/records/merged/progress.json（补 demo 口径）；官方 54.62% 见 primary700-motion50k-gl/result.md。评估环境不同（GL vs 本机），为跨环境参考。
- error 9 局为预标（VideoRepick 7、VideoUnmaskSwap 2），无新增；overflow 降采样 2 局、48 步。

## ⑨ 用户决策记录（原话另见 launch.md 与 handoff authorizations）
- 开工：「开工」；停止规则：「我同意开工之后就尽可能一路跑到底，除非有训练语义上的不同……」
- G6 合并与 SHM 后移：「问题一和三能不能够放在一起来做？然后问题二能不能够在正式起跑的时候再来做，就不要单独跑这个」「你自己来裁决这个问题，尽可能不要单独的跑这个两千步。」
- G5 提前结束：「这个问题我们之前验证过嘛？验证过了就不用再跑了，直接开始吧。」（G5 记为 RECORDED，159 步全部一致）
- 对比口径：「你只需要和我比较 motion map 五十 checkpoint」

## ⑩ 计划外事件与处置
见 launch.md：F1 encoder 字符串键、F2 XLA 预占 0.90、F3 check_checkpoint None 叶、F4 run_shard 并发起跑误判、G5 所需 H5 从公开 bucket 补回；评估比预估慢（原因见⑥）。

## ⑪ 结论与下一步
- 结果如上表；后续方向（encoder 深度、步长、初始化等）由用户决定。
- 待用户确认的清理（RB-E1，本轮自建、已登记）：train-runs/mme_vla_suite_b128_80k/ 下 e2e-smoke20（12G）、e2e-smoke20.fail-a01（20K）、e2e-smoke20.fail-a02（6.1M）、e2e-memdiag-a01（20K）、e2e-memdiag-a02（12G）、e2e-benchckpt-a01（23G）；bench/ 下同名记录目录（共约 200K）；evaluation/e2e-g5b-a01|a02|a03（约 52M）。未经确认不删。

## ⑫ 归档文件清单
records/：EVAL692.txt、merged-progress.json、merged-log.json、merged-motion_stats.json、preseeded-errors.json、train.summary.log、metrics.jsonl。
