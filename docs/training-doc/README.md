# docs/training-doc/ 索引

> 每个 run 一个目录：`launch.md`（起跑前预提交：commit、命令、配置、数据来源、输出路径、判据）、`result.md`（判定行原文与结论）、`records/`（Git 无法还原的日志与指标）。
> 新 run 仍落本目录；环境 A 的吞吐 / 确定性 / dtype 中间产物 / v5 对拍与集群正式 run 留档已于 2026-09-07 迁入 [`../archive/training-doc/`](../archive/README.md)。
> 环境 A（GreatLakes / turbo + 2×RTX 6000 Ada）与环境 B（AWS 8×A100）的数字按 `AGENTS.md` 第 13 条不混比，下表按环境分档。

## 一、环境 B 现行（AWS 8×A100，2026-09-04 起）

### 原版80k公开全集与counting四卡并跑（2026-09-25）

| 目录 | 内容 | 判定 |
|---|---|---|
| [`smoke-orig80k-full-0925/`](smoke-orig80k-full-0925/launch.md) | 公开16任务正式库20步四卡可读性、真实保存与加载 | 预建，等待本库完整数据验收 |
| [`smoke-orig80k-count-0925/`](smoke-orig80k-count-0925/launch.md) | 公开counting正式库20步四卡可读性、真实保存与加载 | 预建，等待本库完整数据验收及新统计量 |
| [`orig80k-schema-0925/`](orig80k-schema-0925/result.md) | 既有40集库三样本CPU输入字段取证，保留dtype及精确数值摘要 | 共同字段精确数值相同；用户事后允许四键缺失/None等价，三样本按新规则通过，非正式全量验收 |
| [`orig80k-env-0925/`](orig80k-env-0925/result.md) | 上游独立uv环境P0、源码恢复和模块来源验证 | P0通过，208项依赖版本一致、退出0；已恢复建库并批准四键缺失/None等价，训练仍须全部前置闸门 |

### 相同取证方式下继续4096→1024（2026-09-23）

用户最终允许优化取证，但同组改前/改后与补跑必须同一取证器和设置。c批次先核验完整b基线与环境指纹，再推进新预算；此处均为预建记录。

| 目录 | 内容 | 判定 |
|---|---|---|
| [`modul-sweep-20260923-c/`](modul-sweep-20260923-c/launch.md) | 取证一致性守卫、选帧参考修正及顺序队列 | 已完成，通过对应验收 |
| [`modul-sweep-20260923-c-m4096-refnpy-20/`](modul-sweep-20260923-c-m4096-refnpy-20/launch.md) | 4096 refnpy 20步真实输入、五标量与201叶取证 | 已完成，通过对应验收 |
| [`modul-sweep-20260923-c-m4096-refnpy-1/`](modul-sweep-20260923-c-m4096-refnpy-1/launch.md) | 4096 refnpy 1步真实输入、五标量与201叶取证 | 已完成，通过对应验收 |
| [`modul-sweep-20260923-c-m4096-packed-20/`](modul-sweep-20260923-c-m4096-packed-20/launch.md) | 4096 packed 20步真实输入、五标量与201叶取证 | 已完成，通过对应验收 |
| [`modul-sweep-20260923-c-m4096-packed-1/`](modul-sweep-20260923-c-m4096-packed-1/launch.md) | 4096 packed 1步真实输入、五标量与201叶取证 | 已完成，通过对应验收 |
| [`modul-sweep-20260923-c-m4096-save-100/`](modul-sweep-20260923-c-m4096-save-100/launch.md) | 4096 100步真实保存与加载 | 已完成，通过对应验收 |
| [`modul-sweep-20260923-c-m4096-capacity/`](modul-sweep-20260923-c-m4096-capacity/launch.md) | 4096 八卡生产容量验证 | 已完成，通过对应验收 |
| [`modul-sweep-20260923-c-m4096-perf/`](modul-sweep-20260923-c-m4096-perf/launch.md) | 4096 八卡独立测速 | 已完成，通过对应验收 |
| [`modul-sweep-20260923-c-m4096-input/`](modul-sweep-20260923-c-m4096-input/launch.md) | 4096 有界输入、在线装配、padding、RoPE及初始化 | 已完成，通过对应验收 |
| [`modul-sweep-20260923-c-m1024-refnpy-20/`](modul-sweep-20260923-c-m1024-refnpy-20/launch.md) | 1024 refnpy 20步真实输入、五标量与201叶取证 | 已完成，通过对应验收 |
| [`modul-sweep-20260923-c-m1024-refnpy-1/`](modul-sweep-20260923-c-m1024-refnpy-1/launch.md) | 1024 refnpy 1步真实输入、五标量与201叶取证 | 已完成，通过对应验收 |
| [`modul-sweep-20260923-c-m1024-packed-20/`](modul-sweep-20260923-c-m1024-packed-20/launch.md) | 1024 packed 20步真实输入、五标量与201叶取证 | 已完成，通过对应验收 |
| [`modul-sweep-20260923-c-m1024-packed-1/`](modul-sweep-20260923-c-m1024-packed-1/launch.md) | 1024 packed 1步真实输入、五标量与201叶取证 | 已完成，通过对应验收 |
| [`modul-sweep-20260923-c-m1024-save-100/`](modul-sweep-20260923-c-m1024-save-100/launch.md) | 1024 100步真实保存与加载 | 已完成，通过对应验收 |
| [`modul-sweep-20260923-c-m1024-capacity/`](modul-sweep-20260923-c-m1024-capacity/launch.md) | 1024 八卡生产容量验证 | 已完成，通过对应验收 |
| [`modul-sweep-20260923-c-m1024-perf/`](modul-sweep-20260923-c-m1024-perf/launch.md) | 1024 八卡独立测速 | 已完成，通过对应验收 |
| [`modul-sweep-20260923-c-m1024-input/`](modul-sweep-20260923-c-m1024-input/launch.md) | 1024 有界输入、在线装配、padding、RoPE及初始化 | 已完成，通过对应验收 |

### 4096→1024顺序实验重跑（2026-09-23，取证并行化）

| 目录 | 内容 | 判定 |
|---|---|---|
| [`modul-sweep-20260923-b/`](modul-sweep-20260923-b/launch.md) | 保留完整判据的有界并行取证与顺序队列 | 2048三组20+1全过；边界暂停，队列130 |
| [`modul-sweep-20260923-b-m2048-before-20/`](modul-sweep-20260923-b-m2048-before-20/launch.md) | 2048 before 20步数值回归 | 完成，退出0；完整对照逐位一致 |
| [`modul-sweep-20260923-b-m2048-before-1/`](modul-sweep-20260923-b-m2048-before-1/launch.md) | 2048 before 1步数值回归 | 完成，退出0；完整对照逐位一致 |
| [`modul-sweep-20260923-b-m2048-after-20/`](modul-sweep-20260923-b-m2048-after-20/launch.md) | 2048 after 20步数值回归 | 完成，退出0；完整对照逐位一致 |
| [`modul-sweep-20260923-b-m2048-after-1/`](modul-sweep-20260923-b-m2048-after-1/launch.md) | 2048 after 1步数值回归 | 完成，退出0；完整对照逐位一致 |
| [`modul-sweep-20260923-b-m2048-after-final-20/`](modul-sweep-20260923-b-m2048-after-final-20/launch.md) | 2048 after-final 20步数值回归 | 完成，退出0；完整对照逐位一致 |
| [`modul-sweep-20260923-b-m2048-after-final-1/`](modul-sweep-20260923-b-m2048-after-final-1/launch.md) | 2048 after-final 1步数值回归 | 完成，退出0；完整对照逐位一致 |
| [`modul-sweep-20260923-b-m4096-refnpy-20/`](modul-sweep-20260923-b-m4096-refnpy-20/launch.md) | 4096 refnpy 20步真实输入、五标量与201叶取证 | 本批次未运行，转c批次 |
| [`modul-sweep-20260923-b-m4096-refnpy-1/`](modul-sweep-20260923-b-m4096-refnpy-1/launch.md) | 4096 refnpy 1步真实输入、五标量与201叶取证 | 本批次未运行，转c批次 |
| [`modul-sweep-20260923-b-m4096-packed-20/`](modul-sweep-20260923-b-m4096-packed-20/launch.md) | 4096 packed 20步真实输入、五标量与201叶取证 | 本批次未运行，转c批次 |
| [`modul-sweep-20260923-b-m4096-packed-1/`](modul-sweep-20260923-b-m4096-packed-1/launch.md) | 4096 packed 1步真实输入、五标量与201叶取证 | 本批次未运行，转c批次 |
| [`modul-sweep-20260923-b-m4096-save-100/`](modul-sweep-20260923-b-m4096-save-100/launch.md) | 4096 100步真实保存与加载 | 本批次未运行，转c批次 |
| [`modul-sweep-20260923-b-m4096-capacity/`](modul-sweep-20260923-b-m4096-capacity/launch.md) | 4096 八卡生产容量验证 | 本批次未运行，转c批次 |
| [`modul-sweep-20260923-b-m4096-perf/`](modul-sweep-20260923-b-m4096-perf/launch.md) | 4096 八卡独立测速 | 本批次未运行，转c批次 |
| [`modul-sweep-20260923-b-m4096-input/`](modul-sweep-20260923-b-m4096-input/launch.md) | 4096 有界输入、在线装配、padding、RoPE及初始化 | 本批次未运行，转c批次 |
| [`modul-sweep-20260923-b-m1024-refnpy-20/`](modul-sweep-20260923-b-m1024-refnpy-20/launch.md) | 1024 refnpy 20步真实输入、五标量与201叶取证 | 本批次未运行，转c批次 |
| [`modul-sweep-20260923-b-m1024-refnpy-1/`](modul-sweep-20260923-b-m1024-refnpy-1/launch.md) | 1024 refnpy 1步真实输入、五标量与201叶取证 | 本批次未运行，转c批次 |
| [`modul-sweep-20260923-b-m1024-packed-20/`](modul-sweep-20260923-b-m1024-packed-20/launch.md) | 1024 packed 20步真实输入、五标量与201叶取证 | 本批次未运行，转c批次 |
| [`modul-sweep-20260923-b-m1024-packed-1/`](modul-sweep-20260923-b-m1024-packed-1/launch.md) | 1024 packed 1步真实输入、五标量与201叶取证 | 本批次未运行，转c批次 |
| [`modul-sweep-20260923-b-m1024-save-100/`](modul-sweep-20260923-b-m1024-save-100/launch.md) | 1024 100步真实保存与加载 | 本批次未运行，转c批次 |
| [`modul-sweep-20260923-b-m1024-capacity/`](modul-sweep-20260923-b-m1024-capacity/launch.md) | 1024 八卡生产容量验证 | 本批次未运行，转c批次 |
| [`modul-sweep-20260923-b-m1024-perf/`](modul-sweep-20260923-b-m1024-perf/launch.md) | 1024 八卡独立测速 | 本批次未运行，转c批次 |
| [`modul-sweep-20260923-b-m1024-input/`](modul-sweep-20260923-b-m1024-input/launch.md) | 1024 有界输入、在线装配、padding、RoPE及初始化 | 本批次未运行，转c批次 |
| [`v2-1600ep-m64x8x8-modul-b128-80k/`](v2-1600ep-m64x8x8-modul-b128-80k/launch.md) | 4096八卡80000步正式训练 | 已完成，通过对应验收 |
| [`v2-1600ep-m16x8x8-modul-b128-80k/`](v2-1600ep-m16x8x8-modul-b128-80k/launch.md) | 1024八卡80000步正式训练 | 已完成，通过对应验收 |

### 首次4096→1024实施（2026-09-23，已中断取证）

a批次仅执行配置基线和2048 before-20的部分前缀，其余预建验证未运行；由b批次完整重跑，原记录保留。

| 目录 | 内容 | 判定 |
|---|---|---|
| [`modul-sweep-20260923-a/`](modul-sweep-20260923-a/launch.md) | 代码实施、旧档回归与完整顺序验证总档案 | 初态摘要615.913秒；主动中断，队列退出1 |
| [`modul-sweep-20260923-a-m2048-before-20/`](modul-sweep-20260923-a-m2048-before-20/launch.md) | 2048 before 20步数值回归 | 只有初态和前两步，退出130，不算通过 |
| [`modul-sweep-20260923-a-m2048-before-1/`](modul-sweep-20260923-a-m2048-before-1/launch.md) | 2048 before 1步数值回归 | 预建，未启动 |
| [`modul-sweep-20260923-a-m2048-after-20/`](modul-sweep-20260923-a-m2048-after-20/launch.md) | 2048 after 20步数值回归 | 预建，未启动 |
| [`modul-sweep-20260923-a-m2048-after-1/`](modul-sweep-20260923-a-m2048-after-1/launch.md) | 2048 after 1步数值回归 | 预建，未启动 |
| [`modul-sweep-20260923-a-m2048-after-final-20/`](modul-sweep-20260923-a-m2048-after-final-20/launch.md) | 2048 after-final 20步数值回归 | 预建，未启动 |
| [`modul-sweep-20260923-a-m2048-after-final-1/`](modul-sweep-20260923-a-m2048-after-final-1/launch.md) | 2048 after-final 1步数值回归 | 预建，未启动 |
| [`modul-sweep-20260923-a-m4096-refnpy-20/`](modul-sweep-20260923-a-m4096-refnpy-20/launch.md) | 4096 refnpy 20步真实输入、五标量与201叶取证 | 预建，未启动 |
| [`modul-sweep-20260923-a-m4096-refnpy-1/`](modul-sweep-20260923-a-m4096-refnpy-1/launch.md) | 4096 refnpy 1步真实输入、五标量与201叶取证 | 预建，未启动 |
| [`modul-sweep-20260923-a-m4096-packed-20/`](modul-sweep-20260923-a-m4096-packed-20/launch.md) | 4096 packed 20步真实输入、五标量与201叶取证 | 预建，未启动 |
| [`modul-sweep-20260923-a-m4096-packed-1/`](modul-sweep-20260923-a-m4096-packed-1/launch.md) | 4096 packed 1步真实输入、五标量与201叶取证 | 预建，未启动 |
| [`modul-sweep-20260923-a-m4096-save-100/`](modul-sweep-20260923-a-m4096-save-100/launch.md) | 4096 100步真实保存与加载 | 预建，未启动 |
| [`modul-sweep-20260923-a-m4096-capacity/`](modul-sweep-20260923-a-m4096-capacity/launch.md) | 4096 八卡生产容量验证 | 预建，未启动 |
| [`modul-sweep-20260923-a-m4096-perf/`](modul-sweep-20260923-a-m4096-perf/launch.md) | 4096 八卡独立测速 | 预建，未启动 |
| [`modul-sweep-20260923-a-m4096-input/`](modul-sweep-20260923-a-m4096-input/launch.md) | 4096 有界输入、在线装配、padding、RoPE及初始化 | 预建，未启动 |
| [`modul-sweep-20260923-a-m1024-refnpy-20/`](modul-sweep-20260923-a-m1024-refnpy-20/launch.md) | 1024 refnpy 20步真实输入、五标量与201叶取证 | 预建，未启动 |
| [`modul-sweep-20260923-a-m1024-refnpy-1/`](modul-sweep-20260923-a-m1024-refnpy-1/launch.md) | 1024 refnpy 1步真实输入、五标量与201叶取证 | 预建，未启动 |
| [`modul-sweep-20260923-a-m1024-packed-20/`](modul-sweep-20260923-a-m1024-packed-20/launch.md) | 1024 packed 20步真实输入、五标量与201叶取证 | 预建，未启动 |
| [`modul-sweep-20260923-a-m1024-packed-1/`](modul-sweep-20260923-a-m1024-packed-1/launch.md) | 1024 packed 1步真实输入、五标量与201叶取证 | 预建，未启动 |
| [`modul-sweep-20260923-a-m1024-save-100/`](modul-sweep-20260923-a-m1024-save-100/launch.md) | 1024 100步真实保存与加载 | 预建，未启动 |
| [`modul-sweep-20260923-a-m1024-capacity/`](modul-sweep-20260923-a-m1024-capacity/launch.md) | 1024 八卡生产容量验证 | 预建，未启动 |
| [`modul-sweep-20260923-a-m1024-perf/`](modul-sweep-20260923-a-m1024-perf/launch.md) | 1024 八卡独立测速 | 预建，未启动 |
| [`modul-sweep-20260923-a-m1024-input/`](modul-sweep-20260923-a-m1024-input/launch.md) | 1024 有界输入、在线装配、padding、RoPE及初始化 | 预建，未启动 |

### 无 motion、32 帧 × 8×8 扩容至 2048 token（2026-09-21）

| 目录 | 内容 | 判定 |
|---|---|---|
| [`m2048-implementation/`](m2048-implementation/result.md) | 实施、验证与用户决定总记录 | 正确性、容量、测速及正式80k完成，收尾验收与主仓合并完成 |
| [`m2048-dump-base-8x64/`](m2048-dump-base-8x64/result.md) | 旧 8×64 全量定点输入基线 | 完成：3200样本、200批、完整1600集身份遍历 |
| [`m2048-dump-base-32x16/`](m2048-dump-base-32x16/result.md) | 旧 32×16 全量定点输入基线 | 完成：3000样本、200批、完整1600集身份遍历 |
| [`m2048-b20-base-8x64/`](m2048-b20-base-8x64/result.md) | 旧 8×64 真实20步基线 | 与CAND完整20+1步逐位对拍通过 |
| [`m2048-b20-base-8x64-step1/`](m2048-b20-base-8x64-step1/result.md) | 旧 8×64 更新1补跑 | 完成：初态/首批/首步标量与主run逐位同 |
| [`m2048-b20-base-32x16/`](m2048-b20-base-32x16/result.md) | 旧 32×16 真实20步基线 | 与CAND完整20+1步逐位对拍通过 |
| [`m2048-b20-base-32x16-step1/`](m2048-b20-base-32x16-step1/result.md) | 旧 32×16 更新1补跑 | 完成：初态/首批/首步标量与主run逐位同 |
| [`m2048-dump-cand-8x64/`](m2048-dump-cand-8x64/result.md) | 旧 8×64 候选输入回归 | 与BASE的3200样本/200批逐位相同 |
| [`m2048-dump-cand-32x16/`](m2048-dump-cand-32x16/result.md) | 旧 32×16 候选输入回归 | 与BASE的3000样本/200批逐位相同 |
| [`m2048-b20-cand-8x64/`](m2048-b20-cand-8x64/result.md) | 旧 8×64 候选20步与补跑 | 完整状态0..20、201叶、标量与输入逐位通过 |
| [`m2048-b20-cand-8x64-step1/`](m2048-b20-cand-8x64-step1/result.md) | 旧 8×64 候选第1次更新补跑 | 与BASE及本侧主run相同，补齐状态1 |
| [`m2048-b20-cand-32x16/`](m2048-b20-cand-32x16/result.md) | 旧 32×16 候选20步与补跑 | 完整0..20状态、201叶、标量和输入逐位通过 |
| [`m2048-b20-cand-32x16-step1/`](m2048-b20-cand-32x16-step1/result.md) | 旧 32×16第1次更新补跑 | 初态/首批/首标量相同，更新1已补齐 |
| [`m2048-input/`](m2048-input/result.md) | 新档配置、装配、padding、在线、共享内存及oracle | 全通过，400真实样本、32种pad、worker0/4等 |
| [`m2048-init/`](m2048-init/result.md) | 512与2048完整初态逐叶比较 | 61叶shape/dtype/SHA完全相同 |
| [`m2048-func/`](m2048-func/result.md) | 功能门首次运行 | 验证器权重常量捕获失败，保留错误；retry1已闭合 |
| [`m2048-func-probe1/`](m2048-func-probe1/result.md) | 修复的探索性诊断 | 247秒全通过，不替代正式验收 |
| [`m2048-func-retry1/`](m2048-func-retry1/result.md) | clean修复提交的首次正式功能重测 | 原结果保留，帧带/有效位基线由retry2更新 |
| [`m2048-func-retry2/`](m2048-func-retry2/result.md) | 帧带与正负对照统一JIT路径的最终功能门 | 全部通过，32帧带最小delta=1.04189e-4>同路径噪声0 |
| [`m2048-dump-packed/`](m2048-dump-packed/result.md) | 新档packed完整输入取证 | 与源NPY的3000样本/200批逐位相同 |
| [`m2048-dump-refnpy/`](m2048-dump-refnpy/result.md) | 新档源NPY完整输入取证 | 完整1600集身份与1192918帧索引相同 |
| [`m2048-r20-refnpy/`](m2048-r20-refnpy/result.md) | 新档源NPY真实20步与补跑 | 与packed完整0..20状态、五标量和输入逐位同 |
| [`m2048-r20-refnpy-step1/`](m2048-r20-refnpy-step1/result.md) | 新参考链的第1次更新补跑 | 与主run初态、首批及首标量一致 |
| [`m2048-r20-packed/`](m2048-r20-packed/result.md) | 新档packed真实20步与补跑 | 双侧0..20/201叶通过，十叶更新，七反例拒绝 |
| [`m2048-r20-packed-step1/`](m2048-r20-packed-step1/result.md) | 新packed第1次更新补跑 | 状态1已补齐，全部真实训练对拍完成 |
| [`m2048-train100/`](m2048-train100/result.md) | 新档100步、原数组及checkpoint保存加载 | 全通过，201叶原数组/61叶加载一致，动作RMS=0、mem_len2048 |
| [`m2048-smoke/`](m2048-smoke/result.md) | b128/w16/FSDP8的20步容量检查 | 全通过，61叶加载正确，共享内存峰值10.60%，临时checkpoint已按归属清理 |
| [`perf-m32x8x8-modul-b128-20260921T051856Z/`](perf-m32x8x8-modul-b128-20260921T051856Z/result.md) | 独占八卡1000步测速与80k外推 | READY；1.030681秒/步、GPU均值98.6104%，80k约23小时 |
| [`v2-1600ep-m32x8x8-modul-b128-80k/`](v2-1600ep-m32x8x8-modul-b128-80k/result.md) | 正式80k运行 | 完成：22h58m52s、EXIT0，16份checkpoint及最终61叶加载通过；主仓已合并 |

### modulation 8×8 接入 motion（2026-09-17）

| 目录 | 内容 | 判定 |
|---|---|---|
| [`mv2-implementation/`](mv2-implementation/result.md) | 新旧契约、独立公式、初态、位置编码与计时验证 | 全部通过；八卡trace导出上限修复与实测见smoke档案 |
| [`mv2-evalbound/`](mv2-evalbound/result.md) | 四任务 test 集 200 次真实 reset 与 budget 160 上界 | PASS：最长 demo 383 帧，需求 103 窗，余量 57 |
| [`mv2-v4/`](mv2-v4/result.md) | 新库交付与独立手算逐位验证 | PASS：3232 样本逐位同，完整覆盖关键集合 |
| [`mv2-m4/`](mv2-m4/result.md) | 真实 modulation 结构的完整 V5 mask/梯度验证 | 六项 PASS：3 次探针、42/42 叶、无排除 |
| [`mv2-rhythm/`](mv2-rhythm/result.md) | 新配置客户端节奏与 1296/1297 边界 | PASS：35例1520点、411长度扫描及完整边界全部通过 |
| [`mv2-aa/`](mv2-aa/result.md) | 新库开启态两轮100步A/A及四叶更新 | PASS：100步五标量、5状态、7输入摘要与800训练索引逐位同，四叶均更新 |
| [`mv2-online/`](mv2-online/result.md) | 全量装配、stub及分层真实encoder三档 | PASS：全71316窗装配、1600集次序、3139真编码窗逐位同 |
| [`smoke-m8x8-modul-motion-20260918T050716Z/`](smoke-m8x8-modul-motion-20260918T050716Z/result.md) | 八卡b128/w16、20步开启态及trace验收 | PASS：20步有限，65叶/10记忆叶，配置同源，八卡完整trace已恢复并修复导出上限 |
| [`v2-1600ep-m8x8-modul-motion-b128-80k/`](v2-1600ep-m8x8-modul-motion-b128-80k/result.md) | 1600 集 demo 补帧 motion 表及 8 卡 b128 的 80k 训练 | 已完成：09-18 23:24 EDT结束，21小时37分30秒、EXIT0；16份checkpoint齐全，最终79999真实CPU加载65叶通过；分窗缺口保留，策略评估待另起计划 |
| [`mv2-v1-dump/`](mv2-v1-dump/result.md) | 关闭态 Dataset 完整取证 | PASS：3,200 样本 / 200 batch 前后逐位一致 |
| [`mv2-v2-legacy/`](mv2-v2-legacy/result.md) | 旧 YAML 与旧表完整取证 | PASS：1,200 样本 / 200 batch 逐位一致，历史 policy 构造通过 |
| [`mv2-v6-grad/`](mv2-v6-grad/result.md) | 关闭态固定 batch 梯度取证 | PASS：61 初态叶及三类各 38 梯度叶前后逐位一致 |
| [`mv2-v7-guard-base/`](mv2-v7-guard-base/result.md) | 关闭态真实 100 步基线 | BASE 留存完整，候选对拍已通过 |
| [`mv2-v7-guard-cand/`](mv2-v7-guard-cand/result.md) | 关闭态真实 100 步候选及逐位对拍 | PASS：100 步、5 状态、7 输入摘要及前 800 索引逐位相同 |

### modulation 8×8 起跑前验证（2026-09-15）

| 目录 | 内容 | 判定 |
|---|---|---|
| [`bench-collate-shm-8gpu/`](bench-collate-shm-8gpu/result.md) | collate 共享内存：20 批真实输入对拍、8 卡速度与 100 步三侧确定性验证 | 全部逐位 PASS；步时 1.773→0.973s（1.82×），GPU 利用率 53.32%→97.87% |
| [`t8-c8-guard-s100/`](t8-c8-guard-s100/result.md) | Dataset 成对白名单放宽；3454 样本轻量对拍及既有 context 8×8 的前 100 步梯度对照 | PASS：100 步五标量、800 个训练索引、6 份输入与完整状态均逐位一致 |
| [`m8-modul-retro/`](m8-modul-retro/result.md) | modulation 关闭态两档固定 batch；历史锚点 `07702f0` 的 A/A 自复现及对当前源码的逐叶梯度比较 | PASS：四组比较，61 个初态叶子、三类 loss 与 38 个梯度叶子逐位一致 |

### 新版四任务建库可读性验证

| 目录 | 内容 | 判定 |
|---|---|---|
| [`v2b-read20-20260914T174147Z/`](v2b-read20-20260914T174147Z/result.md) | 1600 集新库及独立 norm_stats；关闭 motion 的真实 dataloader 与 20 步训练检查 | PASS：20 步有限且正常收尾，统计量一致，临时 run 已清理 |

### 8帧×8×8支持与逐位验收（2026-09-14）

仅使用物理GPU4–7。正式训练轨迹、四组全梯度、两模型七关及各48集闭环均已完成并通过，详见[总览](t8-training/result.md)与[用户决定](t8-training/decisions.md)。

| 目录 | 内容 | 判定 |
|---|---|---|
| [`t8-reference-tools/`](t8-reference-tools/result.md) | 参考量具与既有100步轨迹逐位回归 | PASS，REF固定为99faacb |
| [`t8-fixture/`](t8-fixture/result.md) | 两库四profile完整输入、worker矩阵、独立手算与错配拒绝 | PASS |
| [`t8-infer-regress-4x4/`](t8-infer-regress-4x4/result.md) | 旧4×4在线完整回归 | PASS，772窗逐位一致 |
| [`t8-infer-m8/`](t8-infer-m8/result.md) | 真实8×8在线池化、M8七关及48集闭环 | PASS：14/14、7/7、48/48 |
| [`t8-infer-c8/`](t8-infer-c8/result.md) | C8 checkpoint七关及48集闭环 | PASS：13/13、7/7、48/48 |
| [`t8-gradient/`](t8-gradient/result.md) | 四profile的三类batch全梯度对拍 | PASS，全部32/36叶逐位同且有限 |
| [`t8-c32-a1/`](t8-c32-a1/result.md) | C32 A1：1000更新、batch8、fsdp2、GPU4,5 | 本组1000步与三batch全梯度均PASS |
| [`t8-c32-a2/`](t8-c32-a2/result.md) | C32 A2：1000更新、batch8、fsdp2、GPU4,5 | 本组1000步与三batch全梯度均PASS |
| [`t8-c32-b/`](t8-c32-b/result.md) | C32 B：1000更新、batch8、fsdp2、GPU4,5 | 本组1000步与三batch全梯度均PASS |
| [`t8-m32-a1/`](t8-m32-a1/result.md) | M32 A1：1000更新、batch8、fsdp2、GPU6,7 | 本组1000步与三batch全梯度均PASS |
| [`t8-m32-a2/`](t8-m32-a2/result.md) | M32 A2：1000更新、batch8、fsdp2、GPU6,7 | 本组1000步与三batch全梯度均PASS |
| [`t8-m32-b/`](t8-m32-b/result.md) | M32 B：1000更新、batch8、fsdp2、GPU6,7 | 本组1000步与三batch全梯度均PASS |
| [`t8-c8-a1/`](t8-c8-a1/result.md) | C8 A1：1000更新、batch8、fsdp2、GPU4,5 | 本组1000步与三batch全梯度均PASS |
| [`t8-c8-a2/`](t8-c8-a2/result.md) | C8 A2：1000更新、batch8、fsdp2、GPU4,5 | 本组1000步与三batch全梯度均PASS |
| [`t8-c8-b/`](t8-c8-b/result.md) | C8 B：1000更新、batch8、fsdp2、GPU4,5 | 本组1000步与三batch全梯度均PASS |
| [`t8-m8-a1/`](t8-m8-a1/result.md) | M8 A1：1000更新、batch8、fsdp2、GPU6,7 | 本组1000步与三batch全梯度均PASS |
| [`t8-m8-a2/`](t8-m8-a2/result.md) | M8 A2：1000更新、batch8、fsdp2、GPU6,7 | 本组1000步与三batch全梯度均PASS |
| [`t8-m8-b/`](t8-m8-b/result.md) | M8 B：1000更新、batch8、fsdp2、GPU6,7 | 本组1000步与三batch全梯度均PASS |

### motion 利用率评估（2026-09-08，commitV8.0 起；正本 `docs/motion-utilization.md`）

| 目录 | 内容 | 判定 |
|---|---|---|
| [`mv-openloop-40k/`](mv-openloop-40k/result.md) | 阶段 0 开环动作差（none / mask / swap，noise 标尺）+ 阶段 1 18 层机制（attention 富集、逐层门控/KV 干预最终动作差、固定-loss 逐层梯度）；adapter 自检 | PASS（149 点：mask/noise 9.9、swap/noise 6.4；直读集中 15–16 层，主通路第 0–1 层间接） |
| [`mv-matrix-40k/`](mv-matrix-40k/result.md) | 阶段 2 闭环矩阵：official / normal / mask / swap × 4 seed × (test+val) 400 集，pooled 配对统计；校准批与跨卡校验 | PASS 32/32 6400 集：normal−mask +5.00pp [t +3.44,+6.56] POSITIVE；mask−swap −2.56 ND；normal−official −0.81 ND；mask timeout 27%；MV_XGPU=BITEXACT |

### 训练 / 推理一致性对拍（2026-09-07，commitV7.1 起）

| 目录 | 内容 | 判定 |
|---|---|---|
| [`tic-l0-rhythm-40k/`](tic-l0-rhythm-40k/result.md) | 第 0 关配置与库同源（`check_config_provenance.py`）+ 评估真节奏 CPU 复刻（`eval_rhythm_gates.py`）+ A19 按库重算后在 400 ep 库重跑 M1 | PASS（`TIC_L0` / `TIC_RHYTHM` / 400 ep `A19_VALID_DIST`、`MOTION_DELIVERY` 全 PASS） |
| [`tic-obs-model-40k/`](tic-obs-model-40k/result.md) | 第 1–5 关：输入键 → 预处理后 → 模型内部 → 整段前向 vs 缓存分步 → 最终动作，motion 从库查表（`compare_train_infer_obs.py --motion store`） | 12/13 阻断 PASS；`VT_FULL_VS_CACHED` 超事先阈值待裁决（纯数值来源） |
| [`tic-t3-causal-40k/`](tic-t3-causal-40k/result.md) | `T3_MOTION_CAUSAL` 按收窄口径（单列不确定叶）在 40 ep 库原记录上重跑 | PASS（`covered=36/36 excluded=0`） |
| [`tic-sidecar-40k/`](tic-sidecar-40k/result.md) | 第 1–5 关再跑一遍，motion 换真 sidecar 现算（`--motion sidecar`） | 13/14 阻断 PASS；`MOTION_S_VS_SIDECAR` 140 窗逐位同；`VT_FULL_VS_CACHED` 同上 |
| [`tic-eval-probe-40k/`](tic-eval-probe-40k/result.md) | 第 6 关：探针版 policy server + 主线 `eval.py` 24 集单次仿真，汇总器核不变量 | 6/7 阻断 PASS；`EVAL_PROMPT=FAIL`（1/24 集 goal 组合训练未见，数据覆盖缺口） |
| [`tic-vulkan-makeenv/`](tic-vulkan-makeenv/result.md) | 单进程第 28 次 `make_env` 必崩（Vulkan）的根因排查与修法验证 | PASS（复现第 28 轮；两修法 35 轮不崩） |
| [`vulkan-fix-30ep/`](vulkan-fix-30ep/result.md) | commitV7.2 修法（`GLIBC_TUNABLES`）在真实评估上的验证：单进程连评 30 集 | PASS（第 28 次不再崩，30/30） |

方案、判据总表与结论见 [`../train-infer-consistency.md`](../train-infer-consistency.md)。

### 生产 run、基准与评估

| 目录 | 内容 | 判定 |
|---|---|---|
| [`v2-1600ep-m8x8-modul-b128-60k/`](v2-1600ep-m8x8-modul-b128-60k/result.md) | 新库 1600 集、modulation 8×8、b128、60k、lr 5e-5，GPU 4–7 | 完成：退出 0，12 个 checkpoint，最终 59999 参数树 61/61；V10 通过，耗时 39h05m04s |
| [`smoke-m8x8-modul-20260915T054007Z/`](smoke-m8x8-modul-20260915T054007Z/result.md) | 正式 b128 四卡配置的 20 步 smoke 与 checkpoint 参数树验收 | PASS：20 步有限、61/61 参数树精确匹配、六条 modulation 叶子和新 norm_stats 齐全 |
| [`awsprod40k-b128-motion/`](awsprod40k-b128-motion/result.md) | 生产训练：40k 步、global batch 128、400 ep 库、motion 接入 | 跑完，checkpoint 39999 |
| [`bench-b128-util/`](bench-b128-util/result.md) | 起跑前钉 b128 的卡数 / fsdp / util 档位 | 见 result.md |
| [`repro-4a100-fsdp4/`](repro-4a100-fsdp4/result.md) | 4×A100 fsdp4 复现基准（40 G 模拟与 80 G 两档） | 见 result.md |
| [`siglip-ab-replay-40k/`](siglip-ab-replay-40k/result.md) | 39999 checkpoint 下 `_prepare_history` 八键 vs 训练表逐位重放；SigLIP f32 离线表 vs bf16 checkpoint 差异量化 | 八键 PASS；SigLIP 差异 rel_fro 0.34% |
| [`eval-awsprod40k-b128-motion/`](eval-awsprod40k-b128-motion/result.md) | motion 组仿真评估 | 见 result.md |
| [`eval-official-framesamp-context/`](eval-official-framesamp-context/result.md) | 官方 framesamp+context 权重仿真评估；首次记录 Vulkan 第 28 次 `make_env` 崩溃与分片 ≤27 红线 | 见 result.md |

### motion 接入闸门（环境 B 复刻，40 ep 库）

| 目录 | 闸门 | 判定 |
|---|---|---|
| [`aws-a22-grad/`](aws-a22-grad/result.md) | A22 式单步定点梯度两侧互核 | PASS kinds=3 leaves=32 mismatches=0 |
| [`aws-p5-online/`](aws-p5-online/result.md) | P5 在线 sidecar 逐窗 vs 离线 motion 表 | PASS compared=772 mismatches=0 |
| [`aws-t2-ref-s100/`](aws-t2-ref-s100/result.md) / [`aws-t2-cand-s100/`](aws-t2-cand-s100/result.md) | T2 关闭态 100 步 A/B | ref PASS；cand 的 `BASELINE_ENV` 因 GPU 指纹不同报 FAIL（记录在案，见 result.md） |
| [`aws-t3-closed-s100/`](aws-t3-closed-s100/result.md) / [`aws-t3-open-s100/`](aws-t3-open-s100/result.md) | T3 closed / open 100 步 | PASS；`T3_MOTION_CAUSAL` 的 `pad_bitexact=0` 由 `tic-t3-causal-40k/` 按新口径闭合 |

## 二、环境 A：motion 接入依据（2×RTX 6000 Ada + turbo，只读）

| 目录 | 闸门 | 判定 |
|---|---|---|
| [`motion-a21-g0b-replay/`](motion-a21-g0b-replay/result.md) | A21：S2 起工前 HEAD 复跑 G0b 黄金基线 1000 步 | PASS，锚点同值 |
| [`motion-a22-grad/`](motion-a22-grad/result.md) | A22 单步定点梯度 | PASS kinds=3 mismatches=0 |
| [`motion-p5-online/`](motion-p5-online/result.md) | P5 真编码器在线链 vs 离线表 | PASS episodes=40 |
| [`motion-t1-closed/`](motion-t1-closed/result.md) | T1 关闭态训练等价（旧库 1000 步，对 G0b r1） | PASS |
| [`motion-t2-ref/`](motion-t2-ref/result.md) / [`motion-t2-cand/`](motion-t2-cand/result.md) | T2 新库 300 步严格 A/B | PASS |
| [`motion-t3-closed/`](motion-t3-closed/result.md) / [`motion-t3-open/`](motion-t3-open/result.md) | T3 真实训练端到端 1000 步 | PASS |

## 三、环境 A：dataloader 重构梯度对拍链（只读，仍是判据锚）

| 目录 | 角色 | 判定 |
|---|---|---|
| [`v1-grad-baseline-g0/`](v1-grad-baseline-g0/result.md) | G0 黄金基线（PG0） | 已被 G0b 取代，保留 |
| [`v1-grad-baseline-g0b/`](v1-grad-baseline-g0b/result.md) | G0b 黄金基线 1000 步（链头） | 两轮逐位一致 |
| [`v1-dtype-p5-grad/`](v1-dtype-p5-grad/result.md) | dtype 修复单步定点梯度 B 侧 | PASS |
| [`v1-framesamp-g2/`](v1-framesamp-g2/result.md) | G2：framesamp 三表读路径训练对拍 | PASS |
| [`v1-singlerun-g0/`](v1-singlerun-g0/result.md) | C3：`train.py` 单跑入口 G0 重锚 | PASS，锚点同值 |
| [`v1-postclean-g3/`](v1-postclean-g3/result.md) | G3：v4 破坏性重构正确性长跑 | PASS |
| [`v1-l0-gauge/`](v1-l0-gauge/result.md) | L0 修量具后的两档基准 | PASS |

## 四、环境 A：评估（只读，成功率数字标注 RTX 6000 Ada）

| 目录 | 内容 |
|---|---|
| [`eval-3seed-context-vs-motion/`](eval-3seed-context-vs-motion/result.md) | 三 seed 四任务成功率对照：motion 24.2% ± 1.3 vs 官方 24.5% ± 0.5，无可辨别差异 |
| [`eval-hard-patternlock-routestick/`](eval-hard-patternlock-routestick/result.md) / [`eval-medium-patternlock-routestick/`](eval-medium-patternlock-routestick/result.md) | PatternLock / RouteStick 难度分层评估 |
| [`eval-binfill-pickxtimes/`](eval-binfill-pickxtimes/result.md) | BinFill / PickXTimes 评估 |

## 五、已归档（`../archive/training-doc/`，34 项）

环境 A 的吞吐 / 瓶颈基准 15 项、确定性四档 8 项、dtype 中间产物 3 项、v5 对拍与集群正式 run / smoke 7 项、未完成空壳 1 项（`eval-official-framesamp-modul`）。清单与理由见 [`../archive/README.md`](../archive/README.md)。
