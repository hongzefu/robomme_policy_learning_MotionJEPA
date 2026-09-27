# 原版80k前置：两库测速包装off/on真保存20步

## 1. 结论与指标速览

**两库off/on正式判定均为FAIL；四次20步训练、真实保存与CPU恢复均正常完成。** 实际Beta为`b0efbde61e38411fb1b9114eec8485d36a4ee9a0`，运行于2026-09-27 00:09:50—00:33:34 UTC。每库21次取批、20次模型输入与RNG全部相同，但100对标量有80对不同，末态201叶有147叶摘要不同。两正式judge均返回1，原生退出和capture亦为1；300步perf不放行。完整证据与用户追加off诊断决定见[result](result.md)。准备标签`20260926T231534Z`仍不是实际起跑时间。

| 库/模式 | run档案 | GPU |
|---|---|---|
| full-off | [timing20-full-off-0926-20260926T231534Z](../timing20-full-off-0926-20260926T231534Z/README.md) | 0,1,2,3 |
| full-on | [timing20-full-on-0926-20260926T231534Z](../timing20-full-on-0926-20260926T231534Z/README.md) | 0,1,2,3 |
| count-off | [timing20-count-off-0926-20260926T231534Z](../timing20-count-off-0926-20260926T231534Z/README.md) | 4,5,6,7 |
| count-on | [timing20-count-on-0926-20260926T231534Z](../timing20-count-on-0926-20260926T231534Z/README.md) | 4,5,6,7 |

## 2. 版本与代码状态

实际`commitV11.14Beta`为`b0efbde61e38411fb1b9114eec8485d36a4ee9a0`。四run与两judge共用该clean版本，运行及最终独立核验期间源码、主仓uv环境保持冻结。guard为f8、run模板45b4…、judge模板17c0…；完整SHA及Git源/工作树/runtime绑定按[launch](launch.md)执行。后续归档提交不替换此运行锚点。

旧INPUT运行HEAD为 `3a1582db39c723c735e04752e5027bfe40ecc3e1`；P1及100量具/B为 `00bdabc4dc3db10a8bc9b0dc6766dbf69fee98f8`；上游A为 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`。旧运行的启动版本永远不回填为本阶段新提交。Beta后沿用前核固定diff、取证/输入/训练模块及依赖指纹，不能把归档或再次运行旧judge称为新HEAD重训。

## 3. 启动与配置还原

[launch.md](launch.md)完整写入通用参数绑定、guard父适配、launch/capture/verify、日志联合验收及六任务命令。两正式Git模板在 `scripts/training/tests/`；生成runtime时加 `set -- "$1" <printf %q固定参数>`，后接模板原字节，guard绑定整份runtime新SHA。judge COMMON是未加前缀的run模板实体副本，不覆盖judge位置参数。

实际guard执行原`v1-store/bench/orig80k-build-preflight-0925/tmux-exit-guard-candidate-20260926/tmux_exit_guard.py`；每动作与实际Beta中的正式源及工作树逐字核对，source/runtime绑定写入launch/capture父记录。六会话均先门闩retain+identity再释放；四训练包装最终原生退出0，两失败judge原生退出1，原始回执不改。七个阶段调度载体由Beta中的对应代码块提取，实际文件SHA与父进程返回值另行留证。

## 4. 数据集与划分口径

沿用公开16任务×100集完整库与其中4个counting任务×100集子集，不改变数据或划分。full为1600集/768897帧/476857执行样本，count为400集/189035帧及执行样本；建库实际Beta均为 `49a333eb18e8d6ff1143bf7871ef7c498ab91579`。详见[full库](../../dataset-build-doc/16task-pub-1600ep/README.md)和[count库](../../dataset-build-doc/4task-counting-pub-400ep/README.md)。

full使用 `v1-store/datasets/16task-pub-1600ep`和原版norm `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`；count使用 `4task-counting-pub-400ep`及重算norm `a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9`。full不得替换为完整库自算c5d45b…统计量。source/manifest/packed及assets绝对路径、history摘要见launch。

## 5. 关键超参

四run均为mme_vla_suite、b64/fsdp4/workers4/seed42，仅smoke覆盖steps20。history为perceptual-framesamp-modul.yaml，modulation、budget512、4×4=16 token/帧、最多32 memory帧、memory_token_dim1024；原streaming_obs_horizon16保持。warmup10000、peak_lr=decay_lr=5e-5、decay_steps100000、AdamW clip1.0、EMA0.999、log100/save10000/keep10000不改。

W&B online（project openpi），初始化为本仓v1-store内pi05_base。只清遗留CPU平台，实际x64必须False且不能静默覆盖继承环境绕过拒绝。off直调原训练，on调用speed.run(mode=smoke)，两侧安装相同的额外只读取证层，真实保存行为不变。

## 6. 硬件、调度与耗时

使用AWS 8×A100-SXM4-80GB，/scratch为md0 XFS本地NVMe。两off均于00:09:50起跑，count/full分别于00:20:35/00:21:15结束；各自恢复与原生退出通过后，count-on于00:20:57、full-on于00:21:48起跑，分别于00:31:59/00:32:55结束。每库固定同一组四卡、两库独立推进；现场资源由本次预检记录，完整UTC/PID/UUID见结果档。

本阶段包含device_get/哈希和计时包装观测开销，20步只用于正确性，不计算吞吐优劣、正式ETA或GPU瓶颈结论。已有100步重叠时间也只证明调度有效，不能作为本阶段性能测量。

## 7. 实际执行过程与输出行为

每run实际取21批，模型使用其中20批并更新20次；两侧记录每次模型实参/RNG、完整20步五标量及末态params/EMA/optimizer/step。普通step0日志和尾窗1…19合起来覆盖全部20步。末步loop19必须真实save并等待异步提交完成，再CPU真正恢复checkpoint19。

on侧启用0.5秒磁盘元数据采样和smoke计时，保存窗口采样与原生commit绑定经独立核验；off无speed记录。四run均记录完整201叶末态，真实恢复与自身EMA一致，不能据此推出off/on相等。观测最大占用不是连续真实峰值；W&B ID、UUID、进程和实际状态由各run结果留档。

## 8. 已有前置与本阶段验收

CPU INPUT两组已通过，保留原3a范围与批准的最外退出限制，见[INPUT结果](../orig80k-equiv-0925/result.md)。P1四侧入口/finite/分段/来源记录通过，保留A两段状态且不称轨迹等价，见[P1四侧记录](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/p1-four-sides-verification-root-20260926.json)及[批准后的限制说明](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-p1-20260926T195426Z/archive-staging/p1-result-draft.md)。

100步六run及四judge均已真实完成，数据仍锚定00bd/ecf；下表是刚完成的原报告，不把它们迁移成新Beta运行：

| 证据 | 已完成范围 | SHA256 |
|---|---|---|
| [六run记录验收](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/eq100-six-runs-verification-root-20260926.json) | A各main100+tentative12，B的S1/S3/S2各main100，state50/99各201叶，独立退出完整 | `f23eb261938939958e122cf5138d604175ffc922f2f2552293351e3b0d379648` |
| [full upstream judge](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/eq100-full-upstream-judge-verification-root-20260926.json) | 100步五标量与完整状态严格比较通过 | `cb19125eea2f71f1536ac8b4490d6cd513861812470ec8c0b5a86dc0c11e8163` |
| [count upstream judge](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/eq100-count-upstream-judge-verification-root-20260926.json) | 同上，count库独立判定 | `382c9bf19f8ded3d15c6307a4c8d78141ba3a2786d8cd961fa807464e4ee6792` |
| [full same-entry judge](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/eq100-full-same-judge-verification-root-20260926.json) | hex_mismatch=0、state_mismatch=0，实际4+4训练重叠 | `f3f52bef9e75f3e3556adf950cfb912359369a1b240ea5e8cd406e603a7d5062` |
| [count same-entry judge](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/eq100-count-same-judge-verification-root-20260926.json) | count单跑/并跑独立通过 | `bfa8908f236974dbc2d362c16b06be06345b5f9cbf381ead8c1b9709d56fe063` |
| [两库same-entry独立复核](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/eq100-both-same-judge-verification-independent-20260926.json) | 实际checker返回0，两组完整结果及原生退出链通过 | `9fcf54005bdb5766eade0aea4aa89d18cefdf507c3bdded83691e92c9f255e3f` |

S2主机记录窗口10…99内两侧实际训练重叠631.992527秒，完整落入重叠的步骤数为90/85；仅是并跑有效性证据。100是在既定确定性验证档下的前100步等价，不承诺生产80k逐位轨迹。

四run各自完整性与恢复通过；两组正式判定原文均为`TIMING_EQ_JUDGE=FAIL reason=20步五标量不是逐位一致`。独立全量核对显示：每库loss、grad_norm、llm_grad_norm、mem_enc_norm各20项不等，param_norm的20项相等；状态treedef/keyset一致，但params有37/61、EMA有36/61、opt有74/78叶不等，step的1叶相同。两失败judge没有生成成功结果JSON，后续原生退出守卫按非零拒绝。原因尚未确定，策略评估不在范围内。

## 9. 用户决定记录

用户原始目标为「给出方案跑完整的80k 和80k纯counting任务 完全参照原版训练 4卡+4卡 先做对拍测试」，执行要求「开始实现该计划 有问题立刻问用户 不要自己决策」。本阶段范围经「两库各跑一对 20 步（推荐）」确认；「尽可能并行做」「你有8张卡」落实为两条独立4卡库链。

「允许仅对拍关闭 W&B」仅限P1/100，本阶段保持W&B online。「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」及四None键授权只保留其CPU输入范围，训练比较不降判据。「继续工作 一路做到起泡前 有问题问用户」仍把本轮终点限定为正式80k起跑前。

发现首步差异后，询问用户是否在当前取证结束后两库各追加一次相同配置20步off、按4+4并行，保持严格判据且仅用于诊断。用户明确答复「同意」。该复验不替代原off/on闸门，当前失败记录和原始SHA不改。

## 10. 历史限制与新阶段处置

V11.13结果提交`f7a9757656ae38391319c4001d823bc06e62621f`首次push被远端非快进拒绝后已暂停。用户于2026-09-27要求「给出合并方案」，在只读核实后明确「没有冲突是吧 继续工作」。远端`d710d8489b88aa75770af3452fd7c9b374deaef7`仅改AGENTS.md、CLAUDE.md和greatlakes.md；普通双父合并为`40de98a1b3b60072e80bcf82dc4fbdf64d329b6c`并已正常push，训练源码、依赖、旧运行记录和六份20步启动文档均未受该合并改动。新Beta沿用核验只对这三份规则增加固定d710内容登记，00bd基准、训练源码保护、六工具SHA和113项输入引用判据保留。

远端新增“生产入口不得依赖测试目录”的通用规则；本轮此前已批准的总计划明确指定tests中的speed/timing/completion量具及runner/预算调用，按已有精确授权保留其路径和源码SHA，不宣称其符合该通用新句，不把长期目录迁移混入当前训练准备。AWS本机、v1-store缓存和当前宿主实际并行容量同样沿用已确认范围。

旧INPUT/P1最外final append可能写出日志0后自身失败，已结束pane的最外实际退出不可追补；没有真实任务失败的新证据。用户分别批准「同样补记限制，沿用INPUT结果（推荐）」和「补记P1限制，补独立退出取证后继续（推荐）」，原始数值/来源/记录及历史限制保留，不补造回执。新100已补独立退出证据，原实际HEAD不变。

本阶段所有任务从新门闩起跑，f8留原生pane，父适配实测launch/capture返回并绑定stdout/stderr、identity、receipt及Git源/runtime；现场宿主另记父适配的实际返回。不能仅依赖旧11项wrapper测试或日志0。任何失败保留现场、停止依赖阶段并报告，不自动清理、重命名或放宽配置。

独立核验首次通过uv入口得到`sys.executable=.../python3`，与Beta父适配实际`.../python`的argv字面不同而被拒；按原解释器重核后通过，原失败调用与修正均保留，未改回执或判据。只读机理核查没有证明首步数学计算被包装改写，也没有证明后端非确定性是原因；完整初始state未被摘要，且param_norm只聚合特定kernel，相同norm不能替代完整初态逐位证据。

## 11. 当前结论与下一步

本阶段按严格判据失败，已完成四run真实保存恢复、两份正式失败判定及独立退出留证。按新增授权准备两库各一次同配置off诊断复验，保留旧b0基线与新仅文档Beta的来源边界，先区分关闭包装时是否也有运行间差异；不改确定性设置、容差或原判定器。

之后仍须真实300步4+4 perf、保存恢复与带明确余量来源的预算，完成两份正式80k launch共同Beta/clean准备后停在起跑前。本group不执行80k，不把100或本阶段工具测试替代后续实测。

## 12. 归档文件清单

起跑前文本由实际Beta还原，本README、[launch](launch.md)、四run README与[result](result.md)追加实测结论。代码、脚本和配置不另行复制，权重留run根；运行时命令、template/runtime SHA及Git绑定保留在`v1-store/bench/orig80k-timing20-commands-20260926T231534Z`，小型实测资料按结果档白名单归档。

每run的原始timing_eq目录保持metadata.json、fetch_inputs.jsonl、model_inputs.jsonl、full_state.json及manifest.json五个文件，未混入日志。两正式judge在标量比较失败后退出，预定的full.json/count.json成功结果没有生成，不补造空成功文件。原记录、清洗日志、完成器、on采样及六任务guard侧证和核验报告分别保留，原始绝对路径与字节不改。
