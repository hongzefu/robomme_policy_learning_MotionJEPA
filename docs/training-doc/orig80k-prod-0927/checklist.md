# 两个正式80k起跑前检查清单（仅汇总既定要求）

最新授权：用户明确「确认无误后可以直接开始两个训练」。共同V11.18Beta、clean状态和全部起跑前检查真实通过后，由root直接启动两条既定80k，无需再次请求批准。此前stop-before要求仅作为历史决定保留；检查器本身仍只读并返回STOP_BEFORE_PROD_LAUNCH，root随后独立调用f8 launch。当前两个prod尚未启动。

本清单不是新增验收标准，也不代替用户已授权前提下的真实检查。依据[原计划](/scratch/hongze/robomme_policy_learning_MotionJEPA/0925-orig-80k-full-counting-4plus4-plan.md)、现有runner/contract/完成器及[共享launch](launch.md)。当前两个prod尚未启动；未填项保持待填，不用工具测试替代真实训练/恢复/性能证据。

| 既定核对项 | 当前已知或待填证据 |
|---|---|
| 数据/资产 | 两库49a构建与packed/counting子集已验收；起跑前重核实体路径、manifest、norm与初始化资产，full固定f332原版/count固定a770重算 |
| INPUT | 3a真实内容/顺序/来源结果沿用；主机数值精确及四None键授权只在既定CPU范围，原始dtype/raw SHA保留 |
| P1/100 | P1按原入口/finite/分段范围；100六run/四judge已通过，保留00bd/ecf及S1/S3真单跑、S2主机记录窗口，不回填未来HEAD |
| 旧退出限制 | INPUT/P1按用户两条原话补记沿用；不能补造历史最外退出；新阶段都须独立原生/父进程/宿主真实退出 |
| 确定性20 | 实际Beta c71，四恢复/两schema2 det judge/六独立退出及七phase宿主wait已PASS，保留原证据 |
| normal300 perf | 实际Beta70a，八阶段已正式及独立PASS，最终八阶段索引/宿主快照引用待回填 |
| normal档位 | prod实际命令unset两ORIG selector、XLA_FLAGS、TRAIN_TIMING_STEPS与CPU遗留；normal descriptor必填，继承x64实际True拒绝 |
| 预算来源 | 两侧normal300/299真恢复、speed schema2、联合报告、原始0.5秒采样、原生保存窗口与measurement receipts及所有SHA；不只相信派生峰值 |
| 三项余量 | 用户已选B档16/32/64GiB；growth84798963712B、sampling34359738368B、logs68719476736B，basis及预算最终SHA已落证，未来现场可用量仍须复核，不缩水 |
| 两run共用预算 | 同一实体预算JSON与已审核期望SHA；与既定公式重算一致，现场可用量再次满足max(300GiB,完整预算) |
| 源码/环境沿用 | 正式Beta对实际perf、实现锚aec与所用数据/依赖逐项核指纹；原HEAD不改写，源码或环境变化需明确处理 |
| 版本与档案 | 两run README、完整启动/父适配正文、索引同批形成一个真实正式Beta；SHA尚未生成，两次起跑均精确HEAD+clean |
| 唯一输出 | 固定run名不改，run/records/driver/JAX/CUDA/W&B/XDG均为新实体目标；新两session、命令根与回执不复用；不resume/overwrite |
| 现场资源 | 0–3与4–7固定分组；GPU身份/空闲、CPU/内存/SHM/IO、scratch存储与可用量按实际填入，不凭历史快照 |
| f8与调用身份 | 正式Git源=工作树=原v1-store f8；命令正文与自身SHA、session/window/pane/PID、独立父返回链完整，先retain/identity后释放 |
| 启动身份 | 真正起跑UTC、run UUID、W&B ID待实际生成；准备标签不当起跑时间，W&B保持online |
| 收尾与归档预案 | 8份checkpoint/800条log100/尾99步/state80000/真恢复EMA及driver/原生退出均按现有判据，结果当前不存在 |

共用B预算路径/SHA和margin-source已固定，session准备标签为20260927-ready；仍需根代理填共同V11.18Beta、V11.17归档、最终索引/快照归档路径和两runtime整体SHA；不存在用动态计算值自动充当“期望值”的默认分支。

正式落档时复用[已审normal perf父适配](/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925/perf300-schema2-candidate-20260927/launch.md)的完整 `guard_perf_action()`正文，并将本两个prod runtime正文同时锁进同一Beta；其传参/日志/回执仍使用f8原协议，不新增工具或schema。不能只保留ignored引用、只凭RELEASED/日志0或session消失判成功。

已批准历史原话保持：「补记P1限制，补独立退出取证后继续（推荐）」「同样补记限制，沿用INPUT结果（推荐）」。最新资源要求「尽可能并行做」「你有8张卡」落实为两个run各4卡；此前「继续工作 一路做到起泡前 有问题问用户」仅保留历史；最新授权允许检查无误后直接起跑。清单及最终检查真实通过后，root按最新授权直接启动两个80k。


B档已由用户确认：每份checkpoint按16GiB、两侧共16份保留，即checkpoint_total_bytes=274877906944；增长余量84798963712B；保存采样余量34359738368B（32GiB），另加真实E合计5531541504B，save_temporary_peak_bytes=39891279872；日志/cache余量68719476736B（64GiB）。三项合计383488663552B，约357.151649GiB。预算实体固定为 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-perf300-results-20260926T232339Z/prod-disk-budget.json`，真实budget SHA `3653beccde73c9550639828d022edddffe98f7cc1b083cceab17699920306b20`，margin-source SHA `09ea4e4739e11e2851a013b00e9a2f3322d0b6d9b7859f9b749495be90d375bf`。第8阶段于04:20:41完成正式预算PASS，所需383488663552B、当时可用708412715008B，原生/capture/父宿主均0；该可用量不是未来正式起跑保证，最终独立复核已通过，已归档到 `bafecc658c45fbd940fb1655e7057d2b1d2c0593`。

70a CPU预览已PASS（原SHA4cf995473747d6ad9e462e4dda710f433fe6409b01e71566c54b8b2c3643f7a6），但未真正prod起跑。已批准本轮perf八stage wrapper保留原字节、只豁免COMMAND末尾1B检查；这不修改数值/来源/退出判据，也不自动豁免未来prod检查。

最终只读/CPU检查按[launch中的完整CLI](launch.md)执行，使用固定af8检查器，复用原contract解析及预算只读验收；检查器本身不创建输出或调用launch；全部检查通过后root独立启动两个80k。
