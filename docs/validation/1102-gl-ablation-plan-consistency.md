# 双消融 encoder 训练计划开工前一致性审查

审查锚点 `AUDIT_BASE=9551a288b2d4b5927a0aef6720a4e97b8c51893e`，计划自身提交 `a084aa273fe73c83ad1767c43cd03c4e6762f53e`。对象为根目录 `1101-gl-ablation-encoders-ctx-b64-80k-plan.md` 全文。用户于美东 2026 年 11 月 2 日给出原话：「/data/hongzefu/robomme_policy_learning_MotionJEPA/1101-gl-ablation-encoders-ctx-b64-80k-plan.md 开工」。

独立只读审查者 `/root/plan_consistency` 使用显式 `gpt-6.1-sol/high`，通过 `git show <AUDIT_BASE>:<计划路径>` 阅读固定版本。主会话逐项回原文核实。未提交占位，未启动下载、建表、冒烟或训练，未修改实施代码。开工令已在 `586be6c52023ce98e744b2cb175c8657d89e207a` 登记并推送；该提交只改交接文件，不改变被审计划。

`PLAN_CONSISTENCY_REVIEW=FAIL base=9551a288b2d4b5927a0aef6720a4e97b8c51893e findings=2 blocking=2`

## 一、统一交接与先到先起不能同时满足

第一处原文，K11 与第一部分「九、交接顺序与状态机登记」：「phase1 `done`（含 V6 PASS）→ train 自动 `running`」；第一阶段交接要求两个 encoder 的 V1–V5、V6a、V8 全部通过且 V6 判定行齐全。

第二处原文，第一部分「七、步骤」B1：「两个占位不同时 RUNNING 时先到先起」。K2 同时要求任一 64 档 OOM 后测完其他档位并「逐 JobID `scancel` 两份占位」。

冲突场景：甲占位已运行且冒烟通过，乙仍排队。统一第一阶段尚未完成，甲却被要求直接正式起跑；乙后来 OOM 时，取消两占位还会中止甲已开始的正式训练。当前计划不能唯一决定此时起跑和取消行为。

推荐的最小修订，待用户裁决：先到的占位可先冒烟、保存恢复和测时；两个 run 的 64 档都通过并完成第一阶段后，再统一正式起跑。任一 OOM 仍按原 K2 测完档位、取消两占位并交用户，不会中止另一份正式训练。

另一选择：保留先到先起，将状态与交接改为分 run，并明确另一 run OOM 时是否取消已经正式训练的 run。此选择需要改动状态机与失败处理，主会话不自行采用。

## 二、Beta 早于占位运行的时序不能保证

第一处原文，K10：「Beta commit 在代码合入、契约 PASS、两张表落地之后、占位 RUNNING 之前打好」。

第二处原文，K7：「开工即为每个 run 提交……占位 job」「排队与本机建表并行」；步骤 A1 提交占位，A3b 才提交 Beta。

冲突场景：调度器立即分配占位，而建表尚需数小时。实施者不能保证 Beta 早于调度器分配资源。

推荐的最小修订，待用户裁决：改为「Beta 在占位内首次运行冒烟或训练工作负载之前打好；占位可提前 RUNNING 等待」。保留立即排队、Beta 前完成代码与两张表、部署核对，以及冒烟和正式使用同一 Beta 的要求。

## 三、文字统一与边界

文首「唯一停点是 64 档 OOM」与第二部分「三、闸门总表」中 EMA 身份、H5 SHA 失败停点措辞不一致。后者已有明确身份约束，可将前者限定为「训练 batch 裁决停点」，不取消身份失败停点；本项不计入两个阻塞项。

状态机中原有下一步还使用 `sonnet`、三块写入和 240 小时模板，第一阶段验收亦未按 64 通过与失败分支区分。停止快照将其替换为本次待裁决项，保留其他任务与历史用户原话。

本轮只验证计划内部一致性及现场前置状态，不证明源码接口、输入字节、显存、吞吐或训练行为。由于涉及起跑、取消与交接口径，按 `AGENTS.md` 第 2 条「每次开工前派一个独立子代理审查计划有无自相矛盾」中的冲突处置要求，实施前停止并交用户裁决；没有擅自选择一条执行顺序。

## 四、复核方式

通过 `git show 9551a288b2d4b5927a0aef6720a4e97b8c51893e:1101-gl-ablation-encoders-ctx-b64-80k-plan.md` 可重读上述各节。交付前运行 `git diff --check`、用标准库解析交接 JSON，并核对仅本次两个任务的状态字段及报告变化；不运行训练测试。待用户裁决后，全文同步修订相关口径、步骤、闸门与状态条件，再完整呈现修订版供用户确认开工。

## 五、用户裁决与修订版开工审查

用户随后原话依次为「同意修订 继续做」「不需要再确认，你可以直接开始了。」。按这条明确授权，两项口径已全文统一，完整 v5 计划提交 `2e9b8dc09acf5a5518c95284da9658c1961fd1c9` 已呈现并推送；无需再次请求开工确认。

同一独立审查者重新只读审查该固定版本全文，结论为：

`PLAN_CONSISTENCY_REVIEW=PASS base=2e9b8dc09acf5a5518c95284da9658c1961fd1c9 findings=0 blocking=0`

两 run 共用第一阶段起跑条件、Beta 早于首次工作负载、等待后刷新剩余时间三处衔接一致。上述旧版 FAIL 为历史证据，未改写成 PASS；本节 PASS 仅属于修订版内部一致性，源码、资产与运行仍分别验收。

## 六、实施中最小调整审查

| 调整 | 必要性与实际改法 | 独立审查 |
|---|---|---|
| 工作树入口与落点 | 宿主 `create_worktree` 拒绝以实施副本作为来源；改从同一 Git 对象库的主检出创建，固定基线仍为 `f1dbe7c9c85ffbf23aaba2bc73321f696bdf4b69`。实际五树位于 `/home/hongzefu/.codex/worktrees/o16abl-sN/robomme_policy_learning_MotionJEPA`，仍只整合到实施副本 `v4-gl-orig100-1031`。 | `ADJUST_REVIEW=PASS scale=same findings=0` |
| latent 解包支持 | 现工具仅认 source tar，不能完成 B2。S6 只扩展既有下载器与对应测试，识别 latent tar 并复用既有逐成员 SHA、成员数和解包后删除本轮 tar 的流程；独立暂存后只用 `rsync -a` 同步 `wan-latents/`，不覆盖现库控制文件。 | `ADJUST_REVIEW=PASS scale=same findings=0` |

两项均由同一独立审查者分别确认必要、同等规模、未新增保护层或触及硬停。S6 由原 S2 写入代理同职责续用，独立分支 `worktree-codex-S6`，仅可写 `scripts/dataset/hf_export/fetch_orig100_trainset.py` 与 `tests/dataset/test_fetch_orig100_trainset.py`；下载总量和资源不扩张。

## 七、后续必要适配与真实来源预检

固定 4a 导出入口必读 AWS 绝对数据路径，现有 WAN 包实际锁 b5；两个核心模块在两提交中字节相同。独立审查确认 `ADJUST_REVIEW=PASS scale=same findings=0` 后，S7 仅将上游单 runner 入库并新增显式 `--lineage-build-spec`。原文件 SHA 为 `305fb59a497010ca7ef540092a7a2b3a8b6bd919ba4680c990d15b6eff67c53e`，撤去四处参数差异可完整还原；完整 16 份 H5 集合、字节数、SHA 比较保留，模型、数值、输出不变。实际包仍记录 b5，G3 按锁定 blob 核现有 MJ 源码。10 项 CPU 回归、固定版本独立审查及合后复测通过，已整合为实施分支 `76ce32e`。

真实输入预检日志为 `/data/hongzefu/robomme_policy_learning_MotionJEPA-gl/v1-store/logs/o16abl-portable-lineage.log`，包含：

```text
MJ_PACKAGE_PIN=PASS commit=b5d23a5fab55be16e580c5b77a8f04d68b3170c2 export_sha=c9e260ea768901989b81781cef9e2b4ad1e9950f06e249fb6b48db4c0d37090b encoder_sha=c9b967f3d2c67c0e996117ac1eec560472e8058fcb7419ecb85a6c5b8c8db15b
PORTABLE_LINEAGE=PASS encoder=noxyz files=16 mismatch=0
PORTABLE_LINEAGE=PASS encoder=nostate files=16 mismatch=0
EXIT_CODE=0
```

主会话第一次预检误将关键字参数作为位置参数，产生 `TypeError`、退出 1；原文保留，随后只修正调用写法重跑，未改代码或判据。

训练根快照另必读两份在线输入，库内原五证据不含它们。独立审查确认输入与 encoder 无关、同等规模后，主会话从本机既有真实输入及原成功冒烟根快照核对并暂存到两 `evidence-<tag>`。两份文件的 SHA 为 JSON `3ca016472b031ac5ef41faad5a2a2866f847860c2f8c3e3f54621339539ab9ac`、NPZ `d3447de2941f2d723ecd76417deeb9758b7d16701a8fdb0550a7e5074e7e9ae8`；用真实 `load_online_inputs` 校验 payload、三个数组的 shape、dtype 与 SHA。

```text
EVIDENCE_STAGE=PASS tag=noxyz files=6 sha_mismatch=0 online_arrays=3
EVIDENCE_STAGE=PASS tag=nostate files=6 sha_mismatch=0 online_arrays=3
EXIT_CODE=0
```

目前六文件为原 G0、G2、SOURCE_PIN、缓存收据加两份在线素材；各 tag 的新 G3 待实际生成，不宣称证据已全部齐全。

## 八、实现审查中修复的遗漏

S1 初版允许新 encoder 经旧八卡分支绕过配置组合，原 worker 在原写集内补全局绑定及完整核心反例。之后主会话合后发现 K2 的 B32/B16 诊断误被三处 B64 限制阻断；原独立审查者撤销该版本通过结论，原 worker 同步修正 helper、core、shell，只对 20 步 `o16gl-smoke-*` 放行 B64/B32/B16。100 步、80000 步仍仅 B64，旧八卡绕过继续拒绝。

最终固定 `ccbbda239535e887e2acd38f1fa0156117526ed0` 独立审查通过；原完整回归 342 通过、6 跳过，诊断修复后的完整 core 273 通过，24 个核心组合、10 个 shell 成功、14 个 shell 拒绝用例均通过；主会话合后再测 273 通过、127.11 秒并推送 `4b61f7f`。未提交的临时合并只含主会话自己的五文件差异，已撤回后按最终版本重合，未回滚他人内容。

其他块合后复测：S4 为 46 通过、1 跳过、17.29 秒；S3 初版 4 通过、2.19 秒；S5 恢复合同 132 通过、12.21 秒；S6 下载解包 12 通过、5.30 秒；S7 为 10 通过、0.07 秒。两份新 encoder 已真实 CPU 读回 `epoch=105 global_step=7770`，匹配导出契约。S1 初次测试曾误用主检出路径并由 uv 新建空 `.venv`，该空环境保留未清理，后续验证统一使用实施副本 `-gl/.venv`，未安装或变更依赖。

真实 G3 plan 选 32 窗、14 片、115 对象、32,476,096,020 字节，build_spec 发布 SHA 为 `5943608ef0f1a5febe79069d6081c7427c6f149f071cdf397c0d21a1b567ef7a`。固定清单下载采用显式四线程、默认仍为单线程，6 项本地 HTTP 并发、续传、重跑、篡改拒绝测试与独立审查通过，合后 6 项通过、3.31 秒，已推送 `2b3c5cd`。latent 恢复和 G3 fetch 分别从 clean `dc7ad38`、`7c02c9f` 启动，日志记录各自实际起跑版本；后续文档或无重叠代码提交不冒充其起跑锚点。
