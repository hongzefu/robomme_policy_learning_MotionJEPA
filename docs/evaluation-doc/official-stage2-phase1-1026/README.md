# 官方评估阶段一实施

本轮只实施 [1026计划](../../../1026-official-benchmark-stage2-eval-plan.html) 阶段一。开工计划锚点为 `2789ae42f3166788ec7d5efc4baf29d8d5a1bc4f`，评估分支为 `v4-eval-official-1026`；源起点为计划锁定的 `d7912e4d4ccd6a158d2e0b64ccd657fc989d1a24`。阶段二正式800局未启动。

## 授权与环境

用户明确要求「开始实现这个阶段一,然后我授权你可以使用greatlakes」，并强调「你要用 Codex的映射来做」。随后确认「同意这个，同意是用这个Aspen来做的。」「连接这个Great Lakes。」「登记到状态机」「还有什么问题吗？你尽可能不要阻塞 能做的想做」「先做」。

实际开发主机为 `sled-aspen.eecs.umich.edu`，本机GPU为2张 `NVIDIA RTX A6000`，这一历史环境表差异已获用户确认。Aspen只开发、控制与做CPU检查；模型、VAE和仿真计算进入GreatLakes作业。Codex只读探索使用 `gpt-6-luna`，实现、运行和审查使用 `gpt-6.1-sol`，子代理均显式设置 `high`。并发配置现场核为32，不以此宣称32个代理同时实测。

开工同步原始判定：

```text
REPO_SYNC=OK branch=v4-eval-official-1026 upstream=origin/v4-eval-official-1026 dirty=0
PUSH_STATE=CLEAN
RULES_SYNC=OK
GUARD_FIX=PASS commit=none
PLAN_CONSISTENCY_REVIEW=PASS base=2789ae42f3166788ec7d5efc4baf29d8d5a1bc4f findings=0 blocking=0
```

交接真源为 `docs/handoff/current.json` 的 `official-benchmark-stage2-eval`，初始读取时本任务尚未登记；六项原任务逐字段保持一致，`HANDOFF_PRESERVE=PASS existing_tasks=6`。授权和running状态分别以 `6ed9341`、`0087c71` 提交并推送。

## 文件归属与验证

| 归属 | 范围 | 当前证据与边界 |
| --- | --- | --- |
| A | `examples/robomme/{eval,env_runner,utils}.py`与客户端测试 | 38项CPU测试通过，包含真实本地WebSocket；修正确定性格式错误被误标可重试的问题后独立复核通过；真实仿真未验 |
| B | server、policy恢复、motion client、运行合同与同机验证 | 作者早期98项CPU通过、1项原条件测试跳过；最新简化口径仍需整合审查及真实GPU验证；不再要求A40数值证书 |
| C | 固定清单、文件队列、控制器与汇总 | 作者24项CPU通过；独立审查指出骤死恢复、drain归还、固定分块验证缺口，原owner续修中 |
| G | 占位池、发布与常驻worker、ETA工具 | CPU mock与文件协议联调中；真实Slurm资源操作由独立R执行 |
| D | 独立窗口、位置选行、mask与装配oracle | 基线147通过、9失败；随后补充手算案例，生产修复后全套重测；编码器为替身，不能声称真实VAE／EMA通过 |
| E | 两个在线 `add_buffer` 的输入原子性修复 | 首版182项通过；合前发现过长首包及policy类型转换绕过，正在原owner最小续修 |

各写入owner在独立worktree提交 `sub/<编号>:`，不自行推送或合入；主会话合前核授权范围，合后整体验证、独立审查后推送。测试文件不能生成自身的期望数据。仍在读取阻塞中的复测不能写成通过。

## 实施中调整

固定基线 `FrameSampMemory::add_buffer` 对负索引、包内重复、乱序及缺帧未完整拒绝；部分拒绝发生在raw/history或段边界写入之后。`MME_VLA_Policy::add_buffer` 又在memory接受前修改段边界。独立oracle给出9项真实失败，不是放宽门槛或只看代码推测。

原只读范围调整为仅允许E修改上述两个 `add_buffer`：整批校验前置，policy既有段边界赋值后移。窗口、编码、合法数据数值和RNG不改；不引入回滚副本、事务对象、额外状态或开关。独立审查结论为 `ADJUST_REVIEW=PASS scale=same findings=2`。严格首包长度仍是恰好 `D+1`，policy必须在转换前拒绝布尔与非整数边界，合前发现的遗漏继续按原判据修复。

合法首包与16帧续包使用固定BASE类逐键字节对拍；非法包比较policy与memory失败前后状态。该结论只覆盖替身编码条件下的在线输入链路，任意编码器运行故障回滚、真实VAE／EMA数值、训练梯度均未由这些CPU测试证明。

## 待完成

用户已明确批准「同意这么做。然后这个A100和40的问题你不用再关心了。」「不需要做到跨硬件的数值性一致。」前案审查为 `ADJUST_REVIEW=PASS scale=same findings=4`，最新简化复核为 `ADJUST_REVIEW=PASS scale=same findings=0 blocking=0`。沿既有validation20000，不新增用途、参考数组pin、签发器或证书层。`eval-runtime-contract-v1`记录原话和 `cross_hardware_numeric=not_required`；run身份绑定 `runtime_contract_sha256`。全模型成员、resolved／normstats、motion身份、原料SHA／EMA／source／VAE／软件精度仍严格；实际GPU名称／driver／架构单独记录。无eval参数时训练strict不变，历史G2 FAIL／WAIVED及上游材料不改。

合入独立审查通过的子提交并发布最终固定执行commit。现有NFS `releases/420ae9801b4d23ed3a798e29a4580960301b9d20`只提供三环境初始import来源，未跑本轮实现，最终执行release尚未发布。epoch105已有 `ENCODER_DOWNLOAD=PASS files=3`、`EXIT_CODE=0`；20k36成员传输SHA全过，完整读回／READY未完成，仍有NFS等待。现有BinFill testep0真实reset通过（seed540000／easy，actions0，D0，初始1观测，RGB256×256×3／state8，3.9504s，退出0）；它只证明环境reset，不是策略rollout。

已批运行名 `official-stage2-20k-probe-1026-a`、两个独立1×A40／4CPU／24GiB／48h，以及先单局smoke1次、后两worker各固定五局chunk共10次，总11次。先静态／完整资产与单GPUstrict restore：真实131帧、D66首67＋四次16续包、真实state／wrist；seed42完整noise按model.action_dim生成锁SHA，num_steps10两次重复，动作(20,8)／执行(16,8)及有限值。单局通过后两chunk实采profile，不以五局profile阻挡首局；两阶段收据分开、十局分片无重叠，独立报告task_success与错误／基础设施重试。

`CROSS_HARDWARE_NUMERIC=NOT_REQUIRED reason=user_decision`，不能伪写PASS。真实恢复／IPC／策略rollout、资源profile和分片仍待实跑；早期小样本不是正式60k／800成绩。阶段二仍未开工，正式800覆盖、真实D预算及最终成绩无实测结论。
