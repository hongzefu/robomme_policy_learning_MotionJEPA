# 阶段结果：G2超阈值，正式VLA尚未起跑

截至美东2026-10-26 14:23，任务按计划R11停在G2，等待用户决定单窗诊断。库尚未packed/verified，G3及训练均未启动。

| 项 | 实测 |
|---|---|
| 代码 | S1至S4全部合审并同步GitHub；合后分组CPU共446项通过 |
| G0 | 64窗、16任务、16个demo尾窗，像素SHA零不匹配，13秒 |
| 冒烟 | 4段128窗、14秒，退出0 |
| 全量抽取 | `STAGE2_EXPORT=PASS segments=4995 rows=65419`，1010秒 |
| G1 | `DROPOUT_OFF=PASS windows=64 eval_repeat_max_abs=0.0 train_vs_eval_max_abs=1.684738039970398` |
| G2 | 第8窗 `ButtonUnmask_ep178_exec`，旧格10，帧160至192、无补帧；MSE `1.747542904391321e-7`≤`1e-6`，max `0.013544513720267467`>`0.01`；退出1 |
| G10 | `MJ_CKPT_UPLOAD=PASS source_runs=2 files=35 bytes=7545592817 sha256_mismatch=0 public=1`；退出0 |

全量流程美东14:03:19起跑，G2失败后14:20:42退出。MJ归档14:03:55开始、14:21:20结束；公开bucket `HongzeFu/motionjepa-stage2-pool4-3decoder-1025-v1` 含两源24份checkpoint及配套记录，35文件约7.55GB，已匿名SHA回读通过。两份tmux自然退出，GPU全部释放。

只读复核未发现G2归一化或轴序错误。但历史 `1e-6/0.01` 分别用于同FP16计算结果的落盘舍入、FP16 1+32与FP16整33；当前G2是历史FP32缓存与FP16整33，不能直接引用历史PASS。此外实际缓存为B2/max-autotune，当前G2为B1/default，其贡献尚未实测。当前批准阈值仍被违反，必须保留FAIL，不自行放宽。

已向用户提议GPU0最多10分钟、仅此失败窗的有界诊断：分离缓存复现、精度、时间分块和编译模式，记录逐组误差与最大差坐标；不重抽全库、不改阈值、不自动起训练。截至本文写入尚待回答。

数据Beta为 `7ff55d61640f6b5d983571ea5f23daf227a129cb`，MJ源为 `b5d23a5fab55be16e580c5b77a8f04d68b3170c2`。全量token仍在主仓数据父目录 `motion-tokens-stage2/`；export_record SHA为 `3d97062cbd904dbc96a6184867f43f912429cb2584f842878d9a917248c9b97e`。G2失败JSON SHA为 `88cc88fb69827f00763b5499e3aa3d39bdb411db3e186660eed596716d6dce72`。清洗日志、失败报告和MJ发布清单存records，不含图像或权重。

主检出保持Beta/HOLD，冻结期状态真源为 `/scratch/hongze/robomme_policy_learning_MotionJEPA-temp/docs/handoff/current.json`，分支 `worktree-agent-vla-s2-state-20261026`。用户原话「开工 有问题问用户 所有subagebt必须全部astra high」「同意推送 要和github同步」。原1–2小时起跑预估因本次G2失败暂时失效，未承诺绕过验收。
