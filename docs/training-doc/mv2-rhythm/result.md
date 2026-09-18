# 节奏检查：按用户决定改用代表回放，完整重跑待执行

用户原话「采用35例，保留其他全部覆盖（推荐）」。原入口沿用按全部不同demo长度枚举的逻辑，在新库变为411个回放案例。旧任务于2026-09-18 02:57:58 UTC从clean 81bd0217f890218a1134b135adcf3490cc68dd51启动；用户确认调整后，于03:20:12 UTC仅向本轮Python PID383996发送SIGTERM，正常记录EXIT_CODE=143。已运行1334秒，未产生最终判定，不能计为通过；日志保存在 [all-411-stopped.summary.log](records/all-411-stopped.summary.log)。

新增 --replay-cases representative 只影响RHYTHM_EQ与EVAL_TERMINATION的重复回放。按16种(es−17)%16余数各取最短/最长真实案例，并保留每任务最长demo，共35例；包括es100、最大es1152及g367，覆盖四任务。默认all保持旧行为。--expect-replay-cases 35与--expect-real-es 411是外部数量硬闸。

完整范围保留：real_es始终来自全部411种长度；ES_BOUNDARY继续扫描全部411种长度，1296/1297两侧各完整运行1300步控制流；TAU_LONG仍使用真实最大es1152。V4、V-online和200次真实reset的覆盖均不缩减。

工作区短测：pytest test_eval_rhythm_selection.py 共5项通过（2.37秒），验证已批准的35个真实g编号、默认全量行为、回放与预算扫描分派及错误期望数拒绝。实际 --gate esbound 短测设280秒上限并正常退出0：es1296不报错、82点、k_last160；es1297在全域时刻2593按预算报错，81个成功推理点；全411种长度按最晚决策时刻计算的最大k151、余量9。该余量对应训练集最长demo延长至1300步，真实test集200次reset的余量仍为57，两者口径不同。

这次短测在本轮未提交工作区执行，旧HEAD不锚定修改；完整35例四项闸门将在修补提交后的clean HEAD重跑。记录见 [selection-precheck.json](records/selection-precheck.json) 及其清洗日志。
