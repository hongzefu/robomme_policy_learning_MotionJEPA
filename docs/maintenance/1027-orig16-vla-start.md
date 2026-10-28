# 官方16×100建库与串行训练开工记录

用户批准的计划为 [`1027-orig16-vla-512-and-stage2-motion-4plus4-plan.md`](../../1027-orig16-vla-512-and-stage2-motion-4plus4-plan.md)，固定提交 `d5fe05d647a899e016bc513aa5fba03485a9b398`。本文件记录开工证据与实施中的等价解释，不改写原批准计划。

## 授权与状态

用户原话：「/scratch/hongze/robomme_policy_learning_MotionJEPA/1027-orig16-vla-512-and-stage2-motion-4plus4-plan.md 开工 有问题问用户 尽可能不要阻塞 所有subagent全部astra high」。随后选择「登记三项任务（推荐）」。

三项状态已登记到 [`docs/handoff/current.json`](../handoff/current.json)：建库、context 60k、modulation 60k。两条正式训练均为8卡、全局批量128、32帧4×4、每个motion窗口4个256维token，顺序为context完成后立即接modulation；两条59999恢复验收留到第二条训练结束后。全部子代理显式指定 `gpt-6-astra`、`high`，本轮用户指令覆盖仓库旧档位与计划中的历史角色名称。

## 环境与只读资产预核

美东2026年10月27日开工核实：环境B，8张`NVIDIA A100-SXM4-80GB`，均无GPU占用；本机NVMe RAID的可用空间为3,031,002,558,464字节，约2.757 TiB，高于计划1.5 TiB下限。不存在`/data/hongzefu`、`/nfs/turbo`及集群SSH配置。

```text
HANDOFF_READ=PASS
REPO_SYNC=OK branch=v4-256concat1 upstream=origin/v4-256concat1 dirty=0
PUSH_STATE=CLEAN
RULES_SYNC=OK
GUARD_FIX=PASS commit=none
```

MJ导出工作树位于`/scratch/hongze/worktrees/vla-s2-memexport-20261026`，分支`v12-3decoder-memexport`，起始提交`b5d23a5fab55be16e580c5b77a8f04d68b3170c2`，干净且与远端同步。初始化其规则子模块后同步检查通过，未改代码锚点。该树没有独立虚拟环境；执行时必须显式使用已核实的uv管理环境，并确认源码导入指向所跑工作树。

源H5、ep105、SigLIP、pi05_base、tokenizer与固定Wan VAE均存在。存在性检查不替代后续内容哈希验收。

## 独立计划审查

```text
PLAN_CONSISTENCY_REVIEW=PASS base=d5fe05d647a899e016bc513aa5fba03485a9b398 findings=2 blocking=0
```

审查覆盖固定计划全文、SVG、接口、步骤、计数、授权与固定代码。两项文字遗漏按正文已定口径执行：

1. 短测runbook只列了`DRY_RUN=1`。实际按正文阶段4，先跑启动契约，再执行context和modulation各20步真实训练及保存恢复。仅DRY_RUN通过不能记为`SMOKE_8GPU=PASS`。
2. 窗口示意没有区分物理缓存与消费间隔。Wan按grid16导入全量45,228窗；MJ导出只消费偶数格，得到grid32的23,253窗。窗口、补帧、截尾和计数均维持已批准口径。

## 同源与索引证据的边界

```text
SOURCE_PIN_RECORDS_MATCH=PASS encoder_files=16 published_files=16 mismatches=0
```

encoder的`build_spec.json::identity.source_pin.sha256`为`31d6ab0f845336e76dc25db64170a0bc4ac59ecf366c161025b9800285c74467`；发布`source_manifest.json::pin_sha256`为`9e109419d08828d46261d45ca88786207015752d4b3c54b67a1ce8554fcf96f6`。两个pin的载荷不同，字符串不可直接比较；逐个H5文件名、大小、SHA256集合相同。发布源清单整文件SHA256为`a2ff2c888f6b6f1f95b0f4866b063392576bfc38e94170e3ec7eb813333a998e`。

上述核对只消费已记录清单，未重算478GB H5；不得当作正式L0/L1/L2。新VLA input manifest仍须从实际源文件构造，并由新血缘检查器核验。

独立索引复算：grid16共45,228窗，其中copy42,404、demo-fill900、exec-fill1,924；grid32共23,253窗，非空目标段2,496。正式manifest对齐后，导出计数候选为`--expected-counts 2496,23253`。全量帧768,897，执行样本476,857，最大单集消费44窗，容量96窗。

整文件恢复需要2,485个发布bin，共388,310,630,400字节，另加元数据；本机bin为0。不能把42,404个命中窗的约25GB当成下载量。发布latent已经是`group_first`，导入须读实际轴约定，不能默认再次转轴。

## 实施中调整

```text
ADJUST_REVIEW=PASS scale=same findings=1
```

计划的新工具需要直接读取Parquet，policy主环境未安装`pyarrow`。根锁文件已含`pyarrow==20.0.0`，因此将其从仅可选依赖提升为直接依赖，并通过uv同步安装这一包。`wan-stage2`环境已有固定依赖带入的`pyarrow==26.0.0`，保留其既有锁与数值身份。两种版本只负责索引读取，不改变模型或张量数值路径，不新增回退机制。独立审查认可该同等规模依赖补齐；采用根锁原版本进一步避免升级。

`uv add --no-sync pyarrow==20.0.0`由新版uv重写锁文件元数据，已机械比较所有包版本，`changed_versions=[]`。同步安装采用`--frozen --inexact --no-dev --no-install-project`并排除其余锁包；预览仅安装`pyarrow==20.0.0`，避免同步触及并行测试使用的既有环境。

## 实际分工与后续

S6、S1、S2、S3、S4、S5分别在独立`worktree-agent-o16-*-20261028`分支实施，S7在S2和S4合入后启动。S2在既有`motion_store.py`定义唯一缓存收据校验接口，S1与S3直接复用；共享在线和恢复检查器仍由S2先改证据部分，S7随后接手。

合并保持计划顺序和`--no-ff`，每块先审固定提交，再合入测试与复审。训练库发布包含完整source特征、两种帧表、norm、Wan latent、逐段motion token、packed motion表及缓存证据；concat和split共用同一packed motion表。

当前记录仅证明开工检查通过与实现开始；建库、GPU短测、正式训练、上传和最终恢复均未完成。
