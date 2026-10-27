# 用户停止于70k：严格交付恢复通过，最终备份核验进行中

用户原话：「停止现在的training 收尾 70k够了」。正式训练于美东2026-10-27 12:17:53停止，实际运行19小时8分54秒，真实退出143；原80k配置与run名保留。最终选定已完整提交的70000，保留5000至70000共14份。70k保存时间为美东10-27 12:02:16，其loss为0.0018；最后日志指标70900步loss0.0019，未保存的末段不作为交付权重。

70k远端37文件、12,218,974,941字节已重新匿名回读，SHA零不匹配；严格恢复65个参数叶、来源验证及真实在线infer通过：`CKPT_70K_READY=PASS step=70000 root_complete=1 sha256_mismatch=0 strict_restore=1 online_infer=1`。显式复用既有锁定sidecar环境、VAE缓存和tokenizer，encoder权重和归一化统计均取自隔离远端回读实体。没有重新训练或重抽encoder库。

432份源码/脚本已恢复原权限，HOLD已解除；R13授权旧测速step119已删除，正式14份checkpoint及短训诊断权重保留。当前仅final70最终备份核验运行，完成后补最终清单与判定。W&B summary/notes已标明用户主动停止、最终70000及真实143；服务端把外部SIGTERM显示为crashed，此界面标签不代表训练数值报错。

以下为历史过程记录，以本节用户终点变更为准。

# 正式80k运行中：启动后检查通过

美东2026-10-26 17:08:59从`5730a0cfc7a62036585362a03a8c922f14b4d94f`启动；17:13:20首个loss为0.0871，梯度有限。8卡FSDP8、B128、W16、80000步、pi05_base初始化、原学习率与EMA均符合计划。W&B为hongzefu，run `al3p8t77`。

启动后配置和数据/快照两路独立只读审查均PASS，未发现阻断问题。数据16任务/3200episodes/1383846执行样本，motion65419行/96×1024，encoder为ep105的EMA。G2按用户精确WAIVED保留原FAIL，G5明确CPU全表和1024窗encoder抽样；G7原EMA与动作逐位比较及真实在线infer通过，逐位结论限定同一训练编译缓存。

根快照和encoder依赖17件已上传并匿名回读PASS，增量会话等待首个5000步。正式checkpoint保存、60k严格交付、80k训练完成和最终全量上传尚未到时机，均不冒记PASS。主检出持续冻结；当前状态真源为开发副本`docs/handoff/current.json`。

以下为历史过程记录，以本节最新结果为准。

# 阶段结果：G2超阈值，正式VLA尚未起跑

**最新实测（美东16:16）**：用户批准的省时G5已完成。`PACKED_TOKEN_BYTES=PASS compared=65419 mismatches=0`、`ENCODER_SAMPLE_BITEXACT=PASS compared=1024 total=65419 mismatches=0 seed=0`、`VERIFY_MOTION=PASS scanned=65419 mismatches=0`；749个补帧窗、1051个丢弃短尾、5个合法零窗均核对通过。CPU完整性＋GPU抽样98秒，verify约10.8秒，总计109秒，退出0；库为verified。随后注册encoder/config两实体并通过G6：`TRAIN_ES_BOUND=PASS train_max=62 eval_max=79 budget=96`，2秒、退出0。认证明确scope=sampled，SHA为 `4345250e7a03244ca28162a870041e1648a708ebe7e28a79c50d9cfbcb22a4eb`，不代表全量encoder重算。下一步G4与20步/恢复，正式训练仍未起跑。

用户新增要求：「注意你的wandb 不要用tri的账号 用hongzefu的账号 正式训练」。已用现有凭据只读核API viewer=`hongzefu`、entity=`hongzefu-university-of-michigan`，当前训练配置project=`robomme-framesamp`；凭据不进入日志或git。

**G5最新裁决**：用户明确「同意按照你说过的更省时方案 做」「encoder只抽机小部分」，随后「同意继续」。CPU打包已完成：65,419行、267,956,224字节，table SHA为 `8de36e670d58c60859d48b37f29c989f2eed45344f5cd5fb2f7a96126394ee5c`。原全量GPU比较已按用户指令于美东15:43:53停止（退出130，最后进度至少17,408行一致），GPU释放且无PASS认证。后续改用CPU全量字节/文件SHA＋encoder分层1,024窗，每任务64、seed0，认证明确scope=sampled，不重pack或重抽、不声明全表encoder一致。

**最新裁决（2026-10-26美东15:21）**：用户明确「可以忽略 继续做」，接受本次G2数值例外并恢复原主run。原始FAIL证据保留，例外按精确失败/ckpt/G0三SHA以WAIVED记录传递，不降低其他验收要求。G3随后单次通过：65项EMA原张量精确一致、live负控被拒，32窗对拍max_abs=0，32秒、退出0。当前继续准备pack与全表独立重算，正式训练仍未起跑。以下14:23停工记录作为历史证据保留。

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
