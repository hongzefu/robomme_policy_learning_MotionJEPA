# 1114 4task VLA 计划 workflow 对抗验证留档

- 日期：2026-11-14（美东）；锚点：v1 计划 `59bc133`；workflow run `wf_a7e1b340-f81`。
- 规模：123 个 opus agent（12 维审查 + 每条 finding 一名独立反驳者 + 1 名完整性评审），全部成功。
- 结果：110 条 finding，成立 102（blocker 6 / major 30 / minor 66），驳回 8；完整性评审 4 条全部采纳。
- 用户原话：「写完完整完整方案之后，进行一次 workflow 对抗验证，尽可能详细，不限制这个 agent 的数量，然后不需要叫我批准，直接开始 workflow，然后最后交给我一份完整的计划。」
- 处置：见计划 v2 J 节；原始结构化结果 `records/workflow-result.json`（含每条 finding 的证据、反驳理由与修正后修法）。

## 成立项一览（严重度以反驳者复核后为准）

| 严重度 | 需用户裁决 | id | 标题 |
|---|---|---|---|
| blocker | false | CFG-01 | [内部产出] policy 侧 stage2 来源校验把 arch 写死为三路 tag，G0 证据写死 16 任务，B 节没有列出要改的文件 |
| blocker | false | CFG-02 | 评估移植与 benchmark gitlink 冲突：verify_sources 钉死 b4e97f2，v16 基线是 856bc3a，S4 又禁止改 gitlink |
| blocker | false | EV-1 | S4 写明「third_party/ gitlink 不改」，但 692 局评估硬依赖 fork benchmark b4e97f22，v4 的 gitlink 856bc3a 跑不起来 |
| blocker | false | F04 | [内部产出] 692 局评估移植依赖 third_party gitlink b4e97f2 和 docs 里的 preseeded-errors.json，但分配表禁止改 gitlink，逐文件清单用「等」带过 |
| blocker | false | PK-1 | motion_store 把 stage2 arch 写死为 three-decoder，新 rgb-state 表在打包、训练、评估各处都会被拒 |
| blocker | false | PK-2 | 新来源 local-built-latents 只改 pack_motion_store 不够：motion_store 与 motion_client 的 provenance/evidence 校验会拒绝未知 kind |
| major | false | CFG-03 | S4 只移植 scripts/evaluation/ 不够：eval.py、serve_policy、policies/* 在两条线上差异很大，preseed 文件也不在 v4 线上 |
| major | false | CFG-06 | 新 encoder 没有登记进 v1-store/external/motionjepa/<run>/，在线 sidecar 和启动合同都按这个路径找 ckpt 与 config.yaml |
| major | false | CFG-07 | wan_stage2 sidecar 子项目钉死旧 MJ commit b5d23a5，加载不了 rgb-state arch；B 节没有列 SOURCE_PIN/pyproject/uv.lock |
| major | false | CFG-08 | ckpt 交付检查、G4、上传脚本都写死 16task 配置名或库名，B 节没有列出 |
| major | false | CFG-09 | §4 承诺的 motion 激活方差和分支梯度范数，f7b2dc8 的 train.py 记不了 |
| major | false | EV-2 | 16task 的 wan-stage2 sidecar 环境钉死 MJ@b5d23a5，加载不了新 rgb-state encoder；S4 实际依赖 S2，不是「互不依赖」 |
| major | false | EV-4 | v4 的 policy 侧不读 MMEVLA_MOTION_ONLINE_GPU，32 个 sidecar 会全挤到 yaml 写死的 GPU 1 |
| major | false | EV-5 | S4 工作量被严重低估：v4 上缺整套 692 评估栈，客户端与 policy 源码也有冲突，验收只有 CPU dry-run |
| major | false | F01 | [内部产出] P-LINEAGE 计划阶段锚点缺 encoder ckpt sha 与 VLA pin 相等的命令和判定行，第 7 节阶段 0 的 PLAN_CONSISTENCY_REVIEW=PASS 按规则无法达成 |
| major | false | F02 | [内部产出] DATA_LINEAGE 的比对口径换成了 episode_map，但没有把 episode_map 绑到 VLA 库的 meta/input_manifest.json，vla_input_pin 的语义也和现有脚本不一致 |
| major | false | F03 | [内部产出] 血缘脚本路径写错，episode_map 目录在 runbook 里不存在，第一步 L0 命令会失败 |
| major | false | F08 | [内部产出] ckpt 上传链路（CKPT_UPLOAD/HF_VERIFY）写死了 orig16，逐文件清单和分配表里没人负责 |
| major | false | F10 | [内部产出] raw H5 恢复体量估错一个数量级，与 R2「只新增两个目录」和 R8 磁盘账互相冲突 |
| major | false | F11 | [内部产出] MOTION_XPATH 同一段里既说「不设拦截阈值」又说「超 10 倍停下」，基准值单位不一致、也查不到来源 |
| major | false | F4x4-1 | E 节 hf buckets download 子命令不存在，P0 拉 4×4 会直接失败 |
| major | false | LIN-2 | member-pin 模式没有锚定 meta/input_manifest.json，比对链有断口 |
| major | false | LIN-3 | 新模式 CLI 不接 encoder ckpt，encoder_ckpt=<sha8> 无来源，source_manifest 也没绑定到具体 ckpt |
| major | false | MJ-1 | 导出运行环境不成立：MJ 主 venv 没有导出模块，16task 用的 wan-stage2 venv 又钉死在旧 MJ 提交，不认新 arch |
| major | false | MJ-2 | runbook 里的导出命令参数名不对，缺多个必填参数 |
| major | false | MJ-3 | S2 写集漏了 load_ema_encoder 所在模块和 memexport 分支落后的模型代码；policy 侧也写死了旧 arch |
| major | false | MX-1 | [内部产出] E 节 MJ 导出命令的参数名与真实 CLI 不符，照抄会在 argparse 阶段直接失败 |
| major | false | MX-2 | [内部产出] 新 arch 加载的改动位置写错，且 memexport 分支比 v12-3decoder 落后 155 个提交，缺 rgb-state 的模型与工厂代码 |
| major | false | MX-3 | [内部产出] policy 侧 motion_store 和 framesamp_dataset 把 stage2 arch 写死成三路 tag，新 motion 表的 pack/verify/训练读取都会被拒 |
| major | false | MX-4 | [内部产出] stage2 venv 与 SOURCE_PIN 锁定的是 MJ b5d23a5，导出、pack 和在线 sidecar 都要换成新 MJ 提交，计划没写 |
| major | false | PK-4 | G0/G2/G3 证据读取把任务数写死为 16、窗数为 64，4task 生成不出能通过的 G0 |
| major | false | T1 | [内部产出] 串行链跨约 2.5 天，要求主检出一直停在 Beta HEAD 且 clean，但计划没写冻结（HOLD）和 -temp 开发副本 |
| major | false | T4 | [内部产出] ckpt 增量上传、最终导出和交付恢复（CKPT_DELIVERY_READY）用的脚本只认 orig16，计划文件表里没有对应条目和负责人 |
| major | false | T6 | [内部产出] L0 runbook 的导出、打包、验证命令参数和运行环境与现有脚本对不上 |
| major | false | UP-1 | [内部产出] 增量上传/最终导出工具没有 4task 入口，也不在逐文件清单和子代理分配里 |
| major | true | SEM-1 | 帧记忆 32×4×4 替换 4task 历史 8×8，未列为待裁决项 |
| minor | false | CFG-04 | 4task 692 局评估 max_steps=2000 会让 motion 窗超过预算 96，计划没写溢出口径 |
| minor | false | CFG-05 | check_4task_stage2_launch 不能只改常量：模板里写死了 split 配对、ep105#model、motion-stage2-1024 路径、cache 来源和 16 文件血缘 |
| minor | false | CFG-10 | F2 恢复 4 个合并 H5 的体量写成「约 200 GB 内」，input_manifest 实为约 791 GB，超过 R8 的磁盘余量规划 |
| minor | false | CFG-11 | 串行链设计与 orig16 模板的 formal-only 守卫、失败即 WAIT_USER 语义不一致 |
| minor | false | CFG-12 | runbook 的 episode_map 路径不存在；合同要的 policy 侧 manifest sha 没有写全 |
| minor | false | CFG-13 | 评估驱动里 POLICY/POLICY_CONFIG/RUN_NAME 的默认值是 v11 wantp768，S4 没要求改 |
| minor | false | EV-6 | run_slot「pending→pid 竞争」的诊断与留证矛盾；按这个口径修，未必修到真因 |
| minor | false | EV-7 | 评估必须在主检出上跑，主检出要切到 v16、保持 clean、子模块在 b4e97f22；计划没写 |
| minor | false | EV-8 | G4 在线对拍可能需要原始 H5，raw_dir 恢复实际是硬前置 |
| minor | false | EV-9 | 传给评估的 ckpt 路径没写死；check_checkpoint 按父目录名查 RUNS 表 |
| minor | false | F05 | [内部产出] runbook 命令与现有 CLI 对不上：hf buckets 没有 download，导出和打包的参数名不存在或缺必填项，SHA256SUMS 文件不存在 |
| minor | false | F06 | [内部产出] 第 6 节判定行名称和格式与现有输出脚本不一致，G4_ONLINE 没有任何脚本会打出 |
| minor | false | F07 | [内部产出] EVAL692 的基础设施故障处置与要移植的脚本相反 |
| minor | false | F09 | [内部产出] 待裁决项编号前后不一致，J5 默认登记状态机违反第 27 条「必须问用户」 |
| minor | false | F12 | [内部产出] MJ 导出脚本在哪个分支、哪份工作树运行，三处写法不一致；MJ 侧的合并流程没写 |
| minor | false | F13 | [内部产出] 短测放在 Beta 后的串行链里，与第 7 节阶段 3/4 的划分和 S3 验收的「4 段」不一致 |
| minor | false | F14 | [内部产出] 时长数字前后不一致 |
| minor | false | F15 | [内部产出] 第 27 条要求的「交接顺序与状态机登记」五项不全 |
| minor | false | F16 | [内部产出] 第一部分和第二部分的格式骨架有缺项 |
| minor | false | F17 | [内部产出] 用户原话没有逐字引用，口径表的指向也有误 |
| minor | false | F18 | [内部产出] 没写 encoder ckpt bucket 名和 PAIRED_RUNS_SAME_LIB 判定行 |
| minor | false | F4x4-2 | 按根目录加 --include 'framesamp/*' 下载会多出一层 framesamp/，mv 后目录结构不对 |
| minor | false | F4x4-3 | runbook 没有指定 hf 来源：PATH 上没有 hf，policy venv 只有旧版 huggingface-cli |
| minor | false | F4x4-4 | bucket 里没有名为 SHA256SUMS 的 4×4 校验文件，且校验行路径前缀是 framesamp/ 而不是 framesamp-4x4/ |
| minor | false | F4x4-6 | R2 列出的新增目录不全，和 E 节实际写入的目录对不上 |
| minor | false | F4x4-7 | 4×4 体量写成约 78.2 GB，实测 78.35 GB |
| minor | false | LIN-1 | member-pin 口径与 P-LINEAGE 字面比对对象不一致，需用户裁决或先改规则 |
| minor | false | LIN-4 | check_data_lineage.py 路径、接口和 episode_map 路径都写错，照 runbook 执行会直接失败 |
| minor | false | LIN-5 | 判定行 vla_input_pin 的含义被改成 member pin，与既有闸门和 PAIRED_RUNS_SAME_LIB 不一致 |
| minor | false | LIN-7 | MJ 导出器内置血缘检查的现状描述不准，且 runbook 没传必填的 --input-manifest |
| minor | false | LIN-8 | 起跑合同复查血缘的接线没有写明（16task 模板写死三源 lineage，并要求 files==tasks） |
| minor | false | MJ-4 | 导出血缘 verify_input_lineage 的新输入没有定义清楚：encoder 侧是逐成员记录，VLA 侧是合并后的每任务 H5 |
| minor | false | MJ-5 | F6 和 §I 的前提已过时：MJ 已经解冻，上游分支写成 freeze-4trs-state-1113 是错的，上游 current.json 也是陈旧状态 |
| minor | false | MJ-6 | runbook 里 MJ 侧命令用了相对路径 LIB/LOG，episode_map 路径也不存在 |
| minor | false | MX-5 | [内部产出] 导出器血缘检查要用 member-pin 改造，但计划没写清 CLI 要多收哪个文件 |
| minor | false | MX-6 | [内部产出] E 节 pack/verify 命令同样不能照跑：参数名不对，缺 stage2 必填项 |
| minor | false | MX-7 | [内部产出] E 节的 MJ 运行环境写的是「uv run python」+「MJ venv」，按 MJ 规则会出问题 |
| minor | false | PK-3 | 非 cache 路径的打包前置 check_raw_sources 要 oracle 报告和 raw_dir 三方绑定，4task 库里没有 oracle，raw_dir 路径本身也不存在 |
| minor | false | PK-5 | 非 cache 的 stage2 verify 要先有 compare_stage2.py 写的 encoder_bitexact.json，runbook 漏了这一步 |
| minor | false | PK-6 | runbook 里导出、pack、verify 的命令行参数和真实 CLI 不符 |
| minor | false | PK-7 | 起跑合同模板把 motion 子目录写死为 motion-stage2-1024，还强制要求 cache 来源 |
| minor | false | PK-8 | R2「打包脚本遇到已存在的输出根就拒绝」与代码不符 |
| minor | false | SEM-2 | MOTION_XPATH 的停工判据没法执行，且与「只记录」自相矛盾 |
| minor | false | SEM-3 | J1 备选项对 16task ep105 的描述与事实不符 |
| minor | false | SEM-4 | BinFill demo 窗在 VLA motion 表里怎么参与，计划没有交代 |
| minor | false | SEM-5 | J2「与 16task 统一」只统一了步数，数据遍数并不统一 |
| minor | false | SEM-6 | §1 与 §9 的 J 编号冲突，且「先 modulation 后评估」本是用户已定口径 |
| minor | false | T10 | [内部产出] Monitor 过滤词漏了链脚本和评估脚本的停机文案；会话清单不全；缓存落点与项目规则不一致 |
| minor | false | T11 | [内部产出] 短测在 Beta 前还是 Beta 后，计划前后说法不一 |
| minor | false | T2 | [内部产出] E1/E2 三个 seed 共用一个评估目录名，第二个 seed 会被 local_692_pool.sh 以「目录已存在」拒绝 |
| minor | false | T3 | [内部产出] 链脚本「失败即停在原地」，但没有从中间阶段接续的入口；E1 分片误判会把 31–40 h 的 C 一起卡住 |
| minor | false | T5 | [内部产出] stage2 在线 sidecar 每卡 4 片的显存没有预算，也没安排加载冒烟；v15 每卡 4 片已用到约 79 GB |
| minor | false | T7 | [内部产出] encoder ETA 偏保守约 2 h，而且对不上计划自己给的单轮耗时 |
| minor | false | T8 | [内部产出] M（modulation concat）时长和合计下限估算偏差 |
| minor | false | T9 | [内部产出] MJ 冻结状态和上游 current.json 指针已过时；MJ P5 的 GPU 评估可能和 L0 抢卡 |
| minor | false | UP-2 | R8 写 export stage「硬链接」不对：16task 工具是逐文件实体复制，磁盘峰值约翻倍；删除本地 stage 的步骤也没写 |
| minor | false | UP-3 | P0 拉 4×4 用的 `hf buckets download` 不是有效子命令；校验文件名 SHA256SUMS 也和 bucket 里的实际文件对不上 |
| minor | false | UP-4 | 交付闸门 check_stage2_checkpoint.py 按 yaml 后缀选 TrainConfig，4task yaml 会落到 mme_vla_suite_b128_80k，但它不在改动清单里 |
| minor | false | UP-5 | F5 说 motion-wan-pool768/motion-wan-raw9-s32 在 bucket 上「已有副本」，没有依据，而且删掉也只省约 21 GB |
| minor | false | UP-9 | 留档目录：local692-<run> 和先例的短名惯例不一致；README 索引、短测目录、评估结果上 bucket 都没写 |
| minor | false | XPATH-1 | 「16task 量级 10 倍」的阈值单位对不上：0.0055 是绝对差，MOTION_XPATH 报的是相对差 |
| minor | false | XPATH-2 | 对拍样本与窗口对齐规则没写死；1113 的 raw_episode 不是旧库的 raw_ep_idx，按它对齐会对错集 |
| minor | false | XPATH-4 | 16task 先例的写法不准：当时 VLA 训练用的是 encoder 自己的缓存 latent，本计划恰好相反 |
| minor | false | XPATH-5 | 两条路径的真实差异写错了：aug_id 0 根本不做 243/256 裁剪 |
| minor | false | XPATH-6 | 「只记录、不设拦截阈值」与「超 10 倍就停」互相矛盾 |
| minor | false | XPATH-7 | 「1113 的 vae_receipt.json」并不存在于 1113 数据根，且 S2 禁触与 xpath 读 1113 根的表述冲突 |

## 完整性评审

- [major] X1 [内部产出] 崩溃后「从 step 0 自修重跑」用不了：同名重跑会被单条入口拒绝，换名必须走 attempt 清单和用户决定记录
- [major] X2 [内部产出] 单条训练入口还有几处写死 orig16 的守卫，S3 只参数化库名、TrainConfig、合同脚本不够：新 yaml 名和短测 run 名都会被拒
- [minor] X3 [内部产出] bucket 里有 oracle_tars 和 wan-latents_tars，计划和前序结论都没利用；补 oracle 不需要原始 H5
- [minor] X4 [内部产出] 16task 的 modulation concat 1×1024 当时被用户放弃、没训过 60k，计划写成「待实测」，等于把首次 60k 当作有先例

## 驳回项

- 血缘闸门：成员 sha 应取 episode_map 的 h5_sha256_local，不应取 MANIFEST 抄来的 h5_sha256
- 4×4 帧表拉回：FRAMESAMP4X4_FETCH 判定没有钉住 manifest 绑定，也没用库自带的全量自检
- 692 局评估移植：显存占比 0.28 配每卡 4 片，总量超过 100%，必定 OOM 或抢不到显存
- 692 局评估移植：E2（context）评估时长直接套用 modulation 的 75 min/seed，缺依据
- 跨路径对拍可行性：latent 级对拍现在就能在 CPU 上做，已有实测量级；不能拿 G9 的 0.01 当闸门
- 上传、bucket 与留档：没有写「本机无集群，HF 读回校验在本机做」以及读回所需的磁盘余量
- 上传、bucket 与留档：没有估计公开配额的影响；单条体量估得偏低
- 上传、bucket 与留档：bucket 名去掉了 motion，和 run 名、16task 同类 bucket 的命名不一致
