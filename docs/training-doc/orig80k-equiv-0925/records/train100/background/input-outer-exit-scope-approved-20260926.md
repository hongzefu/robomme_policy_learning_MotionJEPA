# CPU INPUT最外退出限制的批准与沿用说明

**用户已明确批准「同样补记限制，沿用INPUT结果（推荐）」。** 本说明据此记录旧CPU INPUT结果的沿用决定；六个任务的子命令返回与输入内容验收证据保留，同时明确最外整体进程真实退出码不可追补。批准不把缺失的退出回执变成已取得证据，也不新增训练轨迹、20步开关对照或100步等价通过结论。

## 原取证记录与版本保持不变

本说明追加于[原静态取证补充](input-outer-exit-scope-supplement-20260926.md)，其完整SHA256为 `bd8c37ba659b2dca0db52f1ebf54c5ff9e925416acbe9e7f8b0e3b46dfb2fd8a`。原文保留当时“INPUT处置待答”的历史状态，未回写成当时已有批准；本说明记录后续用户决定。原始命令、collector记录、manifest、judge输出、日志和既有归档检查链均不修改，不重跑任务或补造回执。

六个CPU任务实际起跑版本仍为 `3a1582db39c723c735e04752e5027bfe40ecc3e1`，上游A为 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`；本说明编写时主仓冻结版本为 `00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8`。后续文档或代码提交不能回填成这些任务的运行版本。

## 保留的证据与不能追补的边界

六个任务为 `orig80k-input-{full-a,full-b,count-a,count-b,full-judge,count-judge}-20260926T170654Z`。固定源码与实际保存命令已核对：主体子shell最后执行collector/judge命令，随即取得 `PIPESTATUS`。六份唯一的 `COMMAND_EXIT=0`记录各自实际子命令返回0；主体及footer的已捕获输出状态也各自唯一为0。这些证据与最外整体Bash进程的返回值分开表述。

四份collector的schema2 manifest及其各17个文件的原始SHA/bytes检查、四个 `INPUT_COLLECT=PASS` 和两组真实 `INPUT_EQ=PASS`继续沿用。full判据覆盖1600集、476857个执行样本、6906个边界样本、104批和2个epoch；count覆盖400集、189035个执行样本、2400个边界样本、104批和2个epoch。比较规则仍为 `host_numeric_exact_motion_none_v1`，终点为 `collate_before_jax`；主机精确数值及四个获准None键的规则不扩展为训练状态或标量的宽松判据。

旧封装最终direct append存在“先完整写出日志0、随后该append返回非零”的边界；六个已结束会话没有保存独立wait或 `pane_dead_status`。因此**最外整体进程真实退出码不可追补，日志 `EXIT_CODE=0`不能单独证明最外实际退出0**。会话消失、事后检查返回0及本次用户批准均不能补成这一回执。已有纯shell反例证明的是封装证据边界，没有这些CPU任务实际失败的新证据，也没有据此发现输入数值或来源失配。

## 批准后的后续要求

用户本次原话明确适用于旧INPUT结果的限制补记与沿用；此前「补记P1限制，补独立退出取证后继续（推荐）」仍按其P1范围记录，两项决定不相互替代。后续任务强制使用已冻结的[独立退出guard](tmux-exit-guard-candidate-20260926/tmux_exit_guard.py)，SHA256为 `f8dada4025dc49f260697e2fb768e922cbea7b8f98afe5cc8b98555648bff52b`：先为本任务保留窗口并持久化身份后释放启动门闩，结束后联合核验独立pane退出回执、父进程实际capture返回码及其sidecar绑定、命令/工具SHA和原有内容判据。PENDING或任一失败不能放行，日志中的成功行不能代替独立退出证据。

[timing20启动草案](timing20-launch-candidate-00bdabc-20260926/launch-draft.md)据此将INPUT处置状态更新为“已批准补记并沿用”。两库P1适用证据及限制、100步上游/当前与单跑/并跑的既定通过证据、独立Beta/clean锚点、唯一TAG及现场资源预算等前置要求仍须逐项满足；本说明不预写100步或后续阶段已经通过，也不启动任何任务。
