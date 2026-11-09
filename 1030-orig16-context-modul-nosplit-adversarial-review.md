# context／modulation 不 split 方案对抗审查

审查时间：2026-10-30，美东。用户要求「做一次对抗炎症 全部astra high 检查有什么问题」，按对抗验证执行；随后追加「检查有什么用户没确定下来的」。

原审查对象：[训练方案](1030-orig16-context-modul-nosplit-60k-plan.md)，当时 SHA256=`185d4533609f247e15c823827e7d43c87c633d4931aa59f1d3628d3b32895540`。代码锚点为开发副本 `6d2a219c59210e61e9f0f05757136a6e51e3bf61`。下文 F1–F4 及所列行号保留原审查时的发现；当前方案已按文末修订记录更新，不再对应此旧摘要。

> 后续用户决定：「同意增量上传诊断失败后就问用户 改写md」。两项偏好现已确定：每 5000 步增量公开上传；诊断失败立即询问用户，在答复前暂停相关后续步骤。方案已据此修订；本文的原审查 FAIL 不代表修订版已完成实现验收。

三名独立审查代理均显式使用 `gpt-6-astra`、`high`：机制与数据、启动与恢复、交付与授权范围。主会话复核关键源码和现有产物。没有安装依赖、改训练代码、启动训练或执行 GPU 验证。

## 结论

核心 concat 机制已实现。发现 **1 项 P1 交付阻塞、2 项 P2 运行步骤缺口、1 项 P3 解释缺口**。正式启动校验与新串行入口原本已列在方案中，不重复作为发现。

```text
ADVERSARIAL_REVIEW=FAIL p1=1 p2=2 p3=1
REVIEW_AGENTS=3 model=gpt-6-astra reasoning_effort=high
IMPLEMENTATION_STARTED=NO
```

这里的 FAIL 表示方案需修订，不表示 concat 模型机制失败，也不表示本轮运行了 GPU 测试。

## F1／P1：本地交付只生成清单，缺少 encoder 依赖实体

**计划位置：**原方案第 146、229–231 行。拟新增脚本只读训练目录生成 manifest，然后对 `<原run>/59999` 执行 delivery 恢复。

**源码依据：**[训练快照写入](scripts/training/train.py)第 239–261 行只校验外部 encoder 文件并记录 `source_path`、`relative_path`，不复制权重和配置到 run 根。[恢复检查器](scripts/training/check_stage2_checkpoint.py)第 199–203 行在 delivery 模式明确忽略 `source_path`，要求 `checkpoint.parent / relative_path` 是可校验的实体文件；第 497–500 行将 checkpoint 的父目录作为恢复根。绝对路径、越界和符号链接不能替代实体依赖。

只读检查两个现有 split run，结果相同：

| 依赖 | 原 run 内的相对路径 | 注册的外部源 |
|---|---|---|
| `encoder_dependencies/stage2-full144-1025-a/checkpoint_epoch_105.pt` | 不存在 | 存在 |
| `encoder_dependencies/stage2-full144-1025-a/config.yaml` | 不存在 | 存在 |

交付审查代理提取检查器原 `inside()` 函数进行只读调用，复现 `ValueError: 交付实体文件缺失或越界`。即使 manifest 收录外部源的正确摘要，检查器仍然找不到恢复根内实体；漏掉该依赖也会失败。

**最小修正：**将新脚本改为构建独立本地 stage，原 run 保持只读。stage 中放入 ready 文件列出的根快照与来源证据、ready 本身、两份 encoder 依赖，以及 59999 下全部参数、assets 和元数据。沿用依赖声明的相对路径，核验原始身份与复制后摘要，再生成 manifest。恢复命令改用 `<stage>/59999`，报告放在 stage 外。

现有 [_stage2_orig16_vla_ckpt_export_common.sh](scripts/dataset/hf_export/_stage2_orig16_vla_ckpt_export_common.sh)第 217–227 行建立外部依赖映射，第 314–330 行实体复制，已有做法可参考。无需修改恢复器，也无需为本地交付接入公开上传。增加“外部源存在而 stage 缺依赖实体”的负例。

## F2／P2：失败后重试及串行链重新进入缺少可执行方式

**计划位置：**第 138、227、248 行同时要求固定正式 run 名、禁止覆盖，以及失败重试建立新记录。

**源码依据：**[单条入口](scripts/training/run_orig16_stage2_60k.sh)第 79–81 行在执行完整启动合同之前已独占创建日志；[启动合同](scripts/training/check_orig16_stage2_launch.py)第 152–155 行固定正式名称并拒绝已有 checkpoint；[训练器](scripts/training/train.py)禁止 resume／overwrite，现有 checkpoint 未保留可恢复训练所需的优化器状态。

触发场景之一：正式入口因合同失败退出，日志已存在，修复后原名再起即被防覆盖拒绝。另一场景：context 成功、modulation 失败；重新运行从 context 开始的链会撞上已经完成的 context。仅“另建记录”不能使这些入口接受重试。

**最小修正：**区分实验身份与实际 attempt。成功条通过完整证据复核后可以跳过，只启动未完成条；失败条保留现场，若决定重训，以显式登记的 attempt 后缀从 step 0 开始，合同仍严格校验对应实验配置。不要开放任意名称或假称恢复了优化器状态。额外整条重训的成本在发生失败后据证据决定，不自动反复启动。

该项是运行恢复设计缺口，无需用户现在替实现者设计 run 名或脚本参数。

## F3／P2：20 步短测恢复命令未写完整

**计划位置：**第 215 行只引用 `check_stage2_checkpoint.py --mode smoke`，没有列全两条的实际参数与环境。

**源码依据：**[恢复器](scripts/training/check_stage2_checkpoint.py)第 491–495 行要求父进程 `CUDA_VISIBLE_DEVICES=0`、`--online-gpu 1`，并拒绝 `JAX_PLATFORMS=cpu`；第 641–664 行要求 checkpoint、reference、out，且报告必须位于 run 根之外。旧 [smoke-stage.sh](v1-store/logs/o16-20261028/smoke-stage.sh)第 17 行绑定 split 的 `4x256`，不能原样拿来验证 concat。

**最小修正：**为两条 concat 明确各自绝对路径 `<run>/19`、`<验证目录>/final-record/pre_save_reference.json`、独立的 `<验证目录>/save-load.json`；列出清除 `JAX_PLATFORMS`、父 GPU0／在线 GPU1、`MMEVLA_V1_STORE`、`OPENPI_DATA_HOME` 的完整恢复命令，核对实际退出码和 `SAVE_LOAD=PASS`。

保存前参考机制已经实现：[train.py](scripts/training/train.py)第 637–640 行同时接受 concat 和 split，第 720–722 行在 step19 保存前生成参考。这项发现属于运行步骤漏写，不是要求新增模型机制。

## F4／P3：相同窗口时间码不代表相同注意力位置

**计划位置：**第 65、67、181 行应进一步限定“位置对齐”的含义。

**源码依据：**[MemoryAttention](src/mme_vla_suite/models/integration/history_gemma.py)第 84–91 行采用包含 padding 的 `mem_len` 生成 modulation 动作 query 的 RoPE 位置，key 位置为槽位索引；第 97–99 行的 mask 屏蔽无效 key，不消除 `mem_len` 对 query 位置的影响。context 路径在 [history_pi0.py](src/mme_vla_suite/models/integration/history_pi0.py)第 538 行按有效 mask 累加位置，两个融合路径也不能混为一谈。

例如 512 个帧 token 均有效、motion 有 44 窗：concat 有效记忆 556、padding 52，split 有效记忆 688、padding 208。modulation 首动作 query 分别位于 608、896；与最后有效 key 的相对距离分别为 53、209。

**最小修正：**说明 `motion_pos` 的窗口时间码和窗口身份保持一致，而投影参数、注意力序列位置以及 padding 预算会随 token 化一并变化。本实验比较的是原实现下 split／concat 的整体差异。保持原 RoPE 实现，不额外改模型，不增加实验；该说明无需用户重新选择。

## 原审查时的两项待定偏好（现已确定）

| 项目 | 原审查发现的差异 | 用户现已确定的内容 |
|---|---|---|
| 检查点公开备份 | 旧任务每 5000 步增量公开上传，共 12 份；原稿默认本地交付 | 同意两条 concat 增量公开上传，最终全量匿名回读 |
| 诊断失败后的接续 | 旧任务“失败只记录、继续可执行步骤”与原稿严格阻断不同 | 诊断失败立即问用户；留证并暂停相关后续步骤，按实际答复处理 |

原审查指出的两项范围偏好现已获得明确答复，不再列为待决，不重复请求授权。后续实际发生诊断失败时仍须就该次具体故障询问用户，不能用上述原则性决定替代具体故障处理决定。

修订版将询问规则覆盖数据检查、短测、训练诊断、打包、上传、回读和恢复。真实训练非零退出、没有成功日志或缺 59999 仍停止接续并报告；数据血缘不符同样不能绕过。

W&B 记录与 HF 检查点公开备份分别说明：现有单条入口检测已有凭据文件后启用 W&B，旧 split 已采用该行为；修订版沿用该记录方式，并明确加入已授权的 HF 增量备份。本轮没有读取凭据内容，也没有调用外发工具。

## 已确定或可直接继承的事项

- 用户明确：新增 context、modulation 两个不 split 实验；每条八卡串行；方案以 Markdown 放在 `-temp` 根目录；本轮审查代理全部 Astra high。
- 合理继承：60k、global batch 128、seed 42、同一官方 16×100 数据库、同 encoder 和初始化、96 窗、原学习率和冻结策略、保留 12 份检查点。无需重新询问。
- 顺序 context→modulation 沿用原顺序；表述为继承设置即可，不必扩大引用为用户额外说过的原话。
- 本地 stage、完整恢复命令、失败现场保留、测试文件与运行名组织属于实现责任；不应交给用户逐项设计。
- 当前要求是按已明确决定改写方案；训练代码尚未实施，训练及公开上传尚未启动。

## 核实范围与限制

已核对 concat 的训练读取、padding／mask、位置元数据、投影、context／modulation 融合、在线装配和保存恢复相关源码。两份 concat YAML 只差融合方式和目标宽度；与对应 split 相比恰差 motion 的四项配置。标准库重算真实 manifest 得到 1600 集、23253 窗、最大 44 窗，与方案一致。

未发现新的模型机制缺失；`1×1024`、96 窗和 608 总记忆槽位正确。机制代理还核实 motion 投影实际拼接 768 维位置特征：concat context／modulation 的输入 kernel 分别为 `[1792,2048]`、`[1792,1024]`。

本轮未执行 pytest、模型前向或完整 GPU 在线推理。原方案已记录的独立环境缺 `yaml`、当前 GPU 可见性未确认属于开训前环境事项，本报告不将其重复包装为新增发现。现有测试源码覆盖不等于本轮实测通过。

## 后续方案修订记录

| 发现 | 方案层修正 | 实施状态 |
|---|---|---|
| F1／P1 | 扩展现有导出链支持 concat，实体复制 encoder 依赖与 checkpoint 到独立 stage；从 stage 恢复，并将其用于已授权增量上传 | 待实现、待运行验证 |
| F2／P2 | 补已完成 context 的受控跳过、诊断失败待决、用户批准后显式登记 attempt 及独立 bucket；禁止假续训 | 待实现、待测试 |
| F3／P2 | 补全短测训练、step19 参考恢复、GPU／缓存环境和独立报告路径；失败由主会话立即询问 | 命令已写入方案，待实际运行 |
| F4／P3 | 区分窗口时间码与注意力 RoPE 位置，明确保留原机制并披露伴随差异 | 文档已补充 |

本次仅更新计划与本审查记录。原版审查结论保留，新版的结构、链接与命令语法检查不替代实施后的模型和交付验收。
