# 节奏检查：35代表例与完整预算边界全部通过

用户原话「采用35例，保留其他全部覆盖（推荐）」。原入口沿用按全部不同demo长度枚举的逻辑，在新库变为411个回放案例。旧任务于2026-09-18 02:57:58 UTC从clean 81bd0217f890218a1134b135adcf3490cc68dd51启动；用户确认调整后，于03:20:12 UTC仅向本轮Python PID383996发送SIGTERM，正常记录EXIT_CODE=143。已运行1334秒，未产生最终判定，不能计为通过；日志保存在 [all-411-stopped.summary.log](records/all-411-stopped.summary.log)。

新增 --replay-cases representative 只影响RHYTHM_EQ与EVAL_TERMINATION的重复回放。按16种(es−17)%16余数各取最短/最长真实案例，并保留每任务最长demo，共35例；包括es100、最大es1152及g367，覆盖四任务。默认all保持旧行为。--expect-replay-cases 35与--expect-real-es 411是外部数量硬闸。

完整范围保留：real_es始终来自全部411种长度；ES_BOUNDARY继续扫描全部411种长度，1296/1297两侧各完整运行1300步控制流；TAU_LONG仍使用真实最大es1152。V4、V-online和200次真实reset的覆盖均不缩减。

工作区短测：pytest test_eval_rhythm_selection.py 共5项通过（2.37秒），验证已批准的35个真实g编号、默认全量行为、回放与预算扫描分派及错误期望数拒绝。实际 --gate esbound 短测设280秒上限并正常退出0：es1296不报错、82点、k_last160；es1297在全域时刻2593按预算报错，81个成功推理点；全411种长度按最晚决策时刻计算的最大k151、余量9。该余量对应训练集最长demo延长至1300步，真实test集200次reset的余量仍为57，两者口径不同。

这次短测在本轮未提交工作区执行，旧HEAD不锚定修改；完整35例四项闸门将在修补提交后的clean HEAD重跑。记录见 [selection-precheck.json](records/selection-precheck.json) 及其清洗日志。

## 35例首次完整重跑及终止参考修正

35例从clean 77a4dd46d352c42e08a5e61f274223301150292d完整执行，最终EXIT_CODE=1。唯一差异是BinFill g317（es1073、T2146、执行步数1072=16×67）：按eval.py控制流的主臂、通用_drive参考、独立时序分别产生67/68/67个推理点。主臂最后时刻2129，通用参考额外包含终止时刻2145；共享时刻的es、k、次序均无失配。旧终止公式同样多算一次。TAU_LONG及全411种长度、1296/1297边界检查均通过。原报告和日志保存在 [representative-first-failed.json](records/representative-first-failed.json) 与同名前缀清洗日志。

用户确认「修正验证参考并补用例（推荐）」。生产eval在记录终止观测后break，不再进入循环开头infer；因此只在gate_rhythm调用通用_drive时，将帧数截止到min(T−1, es+max_steps+1)。通用_drive本身仍是完整帧流重放器，生产eval、P5全量帧流及全部已有覆盖不变。环境终止且C=T−1−es>0时，推理次数改为ceil(C/16)，即floor((T−2−es)/16)+1。

19项测试通过（5.71秒）：真实EpisodeState/pack_buffer控制流覆盖C=15/16/17、31/32/33及1295/1296/1297/1300/1301/1400，并保留两个未截断参考的反例；代表名单与全量预算分派测试继续通过。g317实际memory/policy短测完成67/67/67点，rhythm/termination均PASS，19.22秒、EXIT_CODE=0，见 [terminal-precheck.json](records/terminal-precheck.json)。短测在本轮工作区执行；完整35例随后从修补提交后的独立路径重跑通过。

## 修复后完整重跑通过

从 clean `782696c231aace21c20200638ae302a5a1c7f277` 启动 tmux `mv2-rhythm-fixed`，2026-09-18 03:43:00→03:47:19 UTC，259秒，起止HEAD相同且porcelain为空，`EXIT_CODE=0`。环境为AWS本地NVMe RAID `/dev/md0` XFS，CPU验证，GPU未参与；代码与启动命令见 [launch-terminal-fixed.md](launch-terminal-fixed.md)。

四项全部PASS：RHYTHM_EQ覆盖35个代表例、1520个推理点，tau/es/k/order摘要失配均为0；EVAL_TERMINATION覆盖全部35个真实终止例，主臂、独立时序与修正公式相符，1300步上限例仍为82次推理；TAU_LONG的es0/1152分别得到k80/151、最小余量9；ES_BOUNDARY保留全部411种真实长度扫描，1296侧完整回放82点、k160，1297侧在全域时刻2593准确阻断第161个合法窗。代表例仍覆盖16种余数的最短/最长、四任务和最大长度。

g317现为67/67/67次推理，生产eval与通用完整帧流重放器均未修改。用户要求「修正验证参考并补用例（推荐）」已经完成；19项测试和真实g317短测记录见前文，最终结构化报告见 [terminal-fixed.json](records/terminal-fixed.json)，完整判定和退出码见 [terminal-fixed.summary.log](records/terminal-fixed.summary.log)。原411例中止与首次35例失败均保留，不用成功记录覆盖历史失败。
