> 创建时间：2026-10-30，美东。当前交付为修订后的训练方案；本轮只改文档。
>
> 用户原话：「给出方案 现在训了context和modul split的模型 同时再训2个不split的」「写在根目录md temp文件夹」。调度已确认：「新增两组，仍按每条 8 卡串行（推荐，保持原实验口径）」。按本轮明确要求，文件使用 Markdown，位于 `robomme_policy_learning_MotionJEPA-temp` 根目录。
>
> 最新决定：「同意增量上传诊断失败后就问用户 改写md」。因此两条均做每 5000 步增量公开备份；诊断失败立即留证并询问用户，未经答复不继续依赖失败结果的后续步骤。此前审查中的两项待定偏好已经确定，不重复询问。
>
> 代码参考：现有 split 正式训练锚点 `9459859384589b3553148cf53f715c1318f1a533`；开发副本当前锚点 `6d2a219c59210e61e9f0f05757136a6e51e3bf61`，分支 `worktree-agent-o16-state-20261028`。后续训练使用完成最小改动、验收并固定后的新提交，不能把旧提交写成新训练的版本。

# 第一部分（给人看）

## 1. 方案与实验矩阵

补训 **context + concat、modulation + concat 各一条 60k**，与已有两条 split 组成 2×2 对照。这里的「不 split」是保留阶段二 encoder 每个窗口的 `1×1024` 输出，即 `stage2_pool4_concat_v1`。

| 模型 | motion 每窗形状 | motion 窗口上限 | motion token 上限 | 帧记忆 token | 总记忆 token | 任务状态 |
|---|---:|---:|---:|---:|---:|---|
| context + split | 4×256 | 96 | 384 | 512 | 896 | 已完成 60k |
| modulation + split | 4×256 | 96 | 384 | 512 | 896 | 已完成 60k |
| **context + concat** | **1×1024** | **96** | **96** | **512** | **608** | 本次新增计划 |
| **modulation + concat** | **1×1024** | **96** | **96** | **512** | **608** | 本次新增计划 |

用户已确认**每条独占 8 卡、两条串行**；顺序沿用原实验，context 先、modulation 后。本次新增两组实验，按相同资源口径串行补训；尚未启动任何训练。

固定口径如下：

1. 使用原版 RoboMME **16 任务×100 集**，复用 `16task-orig100-1600ep-9e109419`，共 476857 个执行样本；帧表为 `framesamp-4x4`，32 帧×16 token＝512。
2. 复用阶段二 encoder `stage2-full144-1025-a/checkpoint_epoch_105.pt#model` 和同一份 `motion-stage2-1024`；两条新实验的数据与 encoder 均与 split 对照同源。
3. 每条从同一份 `pi05_base` 初始化，seed＝42、global batch＝128、workers＝16、训练 60000 步，约 16.1 个数据遍历当量。保持原 AdamW、梯度裁剪 1.0、EMA＝0.999、warmup＝5000、peak_lr＝decay_lr＝5e-5。
4. 两条新模型之间仅融合方式及对应 `memory_token_dim` 不同：context 为 2048，modulation 为 1024。
5. 与各自 split 对照相比，仅改变 motion 的读出方式及其必然伴随的 token 数、输入维度和预算；保存周期、数据顺序配置、窗口定义、精度和冻结策略保持原口径。
6. 每 5000 步保留检查点，最终目录为 `59999`；与原实验一致共 12 份。正式模型从 step 0 训练，不从 split 检查点续训，也不接着 20 步短测训练。
7. 两条训练各自增量公开上传 `5000,10000,…,55000,59999`，每条 12 份；训练结束后完成最终清单、全量匿名回读和严格恢复。W&B 继续原入口的记录方式，HF 检查点公开备份单独验收。
8. 数据检查、短测、数值诊断、最终记录、打包、上传或恢复发生失败，均保留真实 FAIL 与退出码，立即向用户报告并询问处理方式。下一条训练和失败任务的重试不能在用户答复前自动开始；具体处理见第 11 节。

## 2. 现有实验已核实的状态

美东 10 月 30 日现场读取日志与检查点目录，结果如下；交接文件仍写着 running/planned，已经落后于日志。

| 现有模型 | 完成时间（美东） | 现场证据 |
|---|---|---|
| context + split | 10 月 29 日 21:59:12 | `EXIT_CODE=0`、`TRAIN_FINAL_RECORD=PASS`、`59999/` 存在 |
| modulation + split | 10 月 30 日 14:18:03 | `EXIT_CODE=0`、`TRAIN_FINAL_RECORD=PASS`、`59999/` 存在 |
| 两条最终恢复检查 | 10 月 30 日 14:23:22 全部结束 | 两次 `CKPT_DELIVERY_READY=PASS step=59999 root_complete=1 sha256_mismatch=0 strict_restore=1 online_infer=1`，各自退出 0 |

证据文件：[context 日志](v1-store/logs/orig100-ctx32x4x4-motion-s2split4x256-b128-60k.log)、[modulation 日志](v1-store/logs/orig100-m32x4x4-modul-motion-s2split4x256-b128-60k.log)、[原串行训练日志](v1-store/logs/o16-20261028/formal-chain.log)、[最终恢复日志](v1-store/logs/o16-20261028/delivery-restores.log)。以上证明训练与恢复通过，未据此认定公开备份长尾全部完成。

```text
HANDOFF_READ=STALE base=6fefaf56c0d536bc0d1ecc1842df4503beb6a93a tasks=14 done=10 running=3
```

路径判据符合环境 B（AWS 工作盘）；本会话 `nvidia-smi` 无法连接驱动，因此当前 GPU 型号、空闲状态和占用情况尚未现场确认。训练实施前复核硬件，文档整理可以继续。

## 3. 不 split 具体改变什么

落盘 motion 原本就是每窗 `float32[1,1024]`。`datastore/motion_adapter.py::split_stage2_tokens` 在 split 路径上只做无损 reshape；concat 路径直接保留这份 1024 维向量。两种读法都包含同一窗口内四个 pool 组的信息，concat 没有对四组求平均或丢掉其中三组。

| 配置／链路 | split 对照 | 新 concat 实验 |
|---|---|---|
| `motion.source_shape` | `[1,1024]` | `[1,1024]` |
| `motion.readout` | `stage2_pool4_split_v1` | `stage2_pool4_concat_v1` |
| `motion.num_tokens` | 4 | 1 |
| `motion.dim` | 256 | 1024 |
| `motion.budget` | 384 | 96 |
| 进入 motion 投影前的补齐张量 | `float32[384,256]` | `float32[96,1024]` |
| 上述张量每样本大小 | 393216 字节 | 393216 字节 |
| 投影后与帧记忆合并 | `[896,D]` | `[608,D]` |
| `D` | context＝2048；modulation＝1024 | context＝2048；modulation＝1024 |

`window_budget = motion.budget / motion.num_tokens`，两边都等于 96。已有库训练侧最大 44 窗，余量 52 窗；同一时刻的真实窗口集合、时间位置和有效性需在验收中对齐。若 concat 仍设置 `budget=384`，会变成 384 窗，破坏本实验的对照口径。

预期总记忆槽位减少 288，降幅约 32.1%；这不是端到端训练提速比例。输入投影维度会由 256 变成 1024，因此投影参数形状、模型参数量以及注意力计算量也会变化。报告时应同时记录参数量、显存和步时，不能把本实验描述为等参数或等 FLOPs 对照；也不要求两种模型的 loss 或动作输出逐位相等。

位置对齐指同一窗口的时间码、窗口起点和有效性元数据。`history_gemma.py::MemoryAttention` 用含 padding 的 `mem_len` 设置 modulation 动作 query 的 RoPE 位置：split 首位置为 896，concat 为 608；mask 屏蔽无效 key，不消除该位置变化。以 512 个有效帧 token、44 个 motion 窗为例，两者 query 到最后有效 key 的距离分别为 209、53。context 按有效 mask 累加位置，也会随有效 token 数改变位置。这些是原机制下 token 化变更的伴随差异，保留原实现并在结果中披露，不额外改 RoPE。

已有两份目标 YAML，无需新增模型算法或重建数据：

- [context concat 配置](src/mme_vla_suite/models/config/robomme/perceptual-framesamp-ctx-32frame-4x4-motion-stage2-1x1024-orig16.yaml)
- [modulation concat 配置](src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-32frame-4x4-motion-stage2-1x1024-orig16.yaml)

真正需要补的是启动管理：`check_orig16_stage2_launch.py::check_config` 当前将正式 60k 限定为 split，`core` 的配对检查也固定读取 split YAML；原串行入口同样固定了 split 名称。仅换 `HISTORY` 会被 `CFG_RUN` 拒绝。

## 4. 调度、运行名与时间预算

已确定继续使用 `mme_vla_suite_orig16_b128_60k`、`fsdp_devices=8`、GPU 0–7，每条串行运行。

| 模型 | 新运行名 | YAML 末段 | 正式顺序 |
|---|---|---|---|
| context concat | `orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k` | `ctx-32frame-4x4-motion-stage2-1x1024-orig16.yaml` | 1 |
| modulation concat | `orig100-m32x4x4-modul-motion-s2concat1x1024-b128-60k` | `modul-32frame-4x4-motion-stage2-1x1024-orig16.yaml` | 2 |

两个独立公开 bucket 按原命名惯例确定如下；创建及写入仅在后续实施时执行：

| 模型 | 增量公开备份目标 |
|---|---|
| context concat | `HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2concat1x1024-60k-v1` |
| modulation concat | `HongzeFu/robomme-vla-orig100-m32x4x4-modul-motion-s2concat1x1024-60k-v1` |

每条使用独立 `v1-store/exports/hf-ckpt-<run>/stage`，既是上传源也是最终恢复根；不复用旧 split bucket。实施前核实目标未被其他实验占用；同一 run 的断点传输只接受身份与摘要一致的已上传对象。

已有 split 两条端到端耗时约 40.57 小时、16.31 小时，合计约 56.88 小时，可作为八卡串行资源预留参考，不是 concat 的实测时长。20 步短测用于正确性与显存检查；正式起跑后按排除编译和保存阶段的稳定步时更新 ETA，计算 `剩余步数×稳定秒/步`，单独注明保存与交付开销。

## 5. 验收与实施步骤

以下是实施时的目标判定，除第 2 节历史结果外，本轮尚未执行。

| 阶段 | 验收内容 | 通过依据 |
|---|---|---|
| A：固定实验 | 两个 concat YAML、四组实验身份、库与 encoder 来源固定 | `NOSPLIT_CONFIG=PASS`、`DATA_LINEAGE=PASS files=16 mismatches=0` |
| B：启动合同 | 初次只放行指定四个正式组合；重试只接受用户决定后登记的 attempt；拒绝串名、改预算和覆盖旧 run | `LAUNCH_CONTRACT=PASS`，新负例测试通过 |
| C：输入与模型 | 相同数据顺序与窗口内容；concat 形状正确；在线输出消费同一读法 | `NOSPLIT_INPUT=PASS`、`MOTION_LAYOUT=PASS readout=concat tokens_per_window=1 budget=96 total_tokens=608` |
| D：真实短测 | 两条各 20 步、保存 step19、严格恢复及在线推理；loss/梯度有限、无 OOM | `SMOKE20=PASS`、现有恢复工具输出 `SAVE_LOAD=PASS`，实际退出 0 |
| E：正式训练 | 两条各 60000 步、各有 `59999` 与完整末段记录 | `TRAIN_FINAL_RECORD=PASS`、`EXIT_CODE=0` |
| F：交付 | 两条最终检查点严格恢复、根级依赖齐全，恢复后在线推理通过 | `CKPT_DELIVERY_READY=PASS step=59999` |
| G：公开备份 | 每条 12 份增量齐全、最终 manifest 完整、全量匿名回读零失配 | `INC_END run=<run> done=12`、`HF_VERIFY=PASS`、`CKPT_UPLOAD=PASS steps=12`，实际退出 0 |

执行顺序为 A→B→C→两条 D→context E→modulation E→两条 F；G 在每条实际起跑后独立增量执行，最后收齐。只要未发生失败，上一条上传正常运行中的长尾不阻塞下一条训练；出现任何诊断或上传失败则按第 11 节暂停接续、询问用户。最终交付必须同时完成 E、F、G。主副本原任务若仍有依赖冻结源码的备份长尾，先核实其状态再整合新代码。

后续分析按融合方式分别比较 split/concat，再比较两种融合方式；同时记录训练 loss、参数量、显存和吞吐。任务成功率需后续统一 benchmark、任务、seed、checkpoint 步数和评估预算才能比较。本方案交付范围包括训练、实体化检查点包、增量公开上传与最终匿名回读；正式 benchmark 评估另列任务。

## 6. 子代理分工与合并（简述）

若进入实施，将启动合同与负例测试、concat 串行入口、打包上传三块分别委派，各自负责独立文件。主会话负责配对配置检查、数据身份、版本整合、GPU 调度、实际训练及失败后询问用户。先合合同，再合串行入口和导出改动，最后检查训练与上传两侧身份完全一致。此前三路对抗审查已经完成；本轮尚未派发实现任务。

# 第二部分（技术细节，供 agent 追踪）

## 7. 前置边界与外部依赖

本轮工作为按用户决定修订文档；增量公开上传及诊断失败后询问用户已经明确授权纳入计划。本轮不据此启动训练或上传。后续实施沿用这些已定事项，不重复请求同一授权；数据、初始化、训练超参和两条实验身份以本文为准。

| 依赖 | 固定值／定位 |
|---|---|
| 主工作副本 | `/scratch/hongze/robomme_policy_learning_MotionJEPA` |
| 开发副本 | `/scratch/hongze/robomme_policy_learning_MotionJEPA-temp` |
| encoder SHA256 | `27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035` |
| encoder 源 pin | `31d6ab0f845336e76dc25db64170a0bc4ac59ecf366c161025b9800285c74467` |
| VLA input manifest SHA256 | `5d1bfed54377c22e68bc1883ebd35fdf77372129ac1565dd6a78601c1d9965f9` |
| episode manifest 文件 SHA256 | `93d7c5a0c1f34fd955de5f8c37370369e54457019f75e56aea8908cba894d999` |
| canonical manifest 身份 | `337c4fcde163974039b5c0f2ad1f46ebaacd272719f72818503e23c1360447e2` |
| norm_stats SHA256 | `c5d45b4bdb481ca46d18aeb32446bbedec34de435eef31d3212a179ee7868bcd` |
| motion 表 | `v1-store/datasets/16task-orig100-1600ep-9e109419/motion-stage2-1024`，23253 行、每行 `[1,1024]` |
| MJ 源清单 | `/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/source_manifest.json` |
| VLA 初始化 | `v1-store/models/openpi-assets/checkpoints/pi05_base/params`，实施时记录文件身份与原 run 对照 |

**两个源 pin 的字符串不相等是正常的**：它们摘要的清单格式不同。必须由 `check_data_lineage.py::check_lineage` 比较 encoder、VLA、MJ 三方的 H5 文件名、字节数、SHA256 集合完全一致，不能将 pin 字符串相等作为替代判据。两条 split 启动日志已经记录相同的 `DATA_LINEAGE=PASS`；本轮重新读取并核对了 input manifest、episode manifest、norm 三份文件摘要，与表中一致。

开发副本的 `v1-store` 指向主副本，任何写入都是主数据写入；新产物只写专属路径。主副本完成旧任务冻结期后才合入新代码，禁止在旧任务仍依赖其版本时移动 HEAD。开发副本使用自己的依赖环境；本轮发现其 `.venv` 缺少 `yaml`，运行血缘脚本在导入阶段失败，**没有获得一次新的血缘复验通过结果**。实施时按锁文件恢复环境后重新验收。

本轮对旧 Beta 到当前开发 HEAD 的 `src/`、`scripts/training/` 中 31 个发生变化的 Python 文件做了去注释／文档字符串后的 AST 比较，未发现可执行 AST 差异。该结果支持当前改动主要为文档引用迁移，不代替完整环境、依赖、配置或运行行为一致性验证。

## 8. 逐文件改动清单

| 文件／锚点 | 现有能力 | 拟改内容 |
|---|---|---|
| `scripts/training/check_orig16_stage2_launch.py::RUNS`、`check_config` | 正式仅允许两个 split run | 建立 `(integration_type, readout)` 到指定 run 名的四项显式映射；允许已登记决定支持的 attempt 后缀；帧布局仍限 32×4×4，保持其他超参固定 |
| 同文件 `core` | 配对 YAML 固定为 `4x256` | 根据当前有效 history 选择同读法的 ctx/modul 配对；两个配对都只允许 `integration_type,memory_token_dim` 不同 |
| 同文件 `check_history`、`check_dataset` | 已支持 concat 的 96 窗与 608 总 token、同源和来源校验 | 复用；保留原形状、摘要、清单、窗口与来源检查，不增跳过选项 |
| `scripts/training/run_orig16_concat_ctx_then_modul_60k.sh`（新增） | 无 concat 正式串行入口 | 固定两条实验身份；正常完成后接续；诊断失败写待决记录并返回主会话询问；支持复核已完成前条后从后条进入 |
| `scripts/training/run_orig16_stage2_60k.sh` | 参数化单条入口，已支持两种读法的配置文件名 | 八卡串行方案原则上无需修改；保留日志防覆盖与全配置合同 |
| `tests/training/test_check_orig16_stage2_launch.py` | 八组合短测、旧正式训练与串行检查 | 增 concat 正式正例、身份错配负例、诊断失败等待用户决定、无决定不能重试、已完成 context 不重跑、已登记 attempt；保持 split 回归 |
| `tests/training/test_paired_ctx_modul_yaml.py` | 只检查 split 配对 | 参数化检查 split、concat 两对；新增同融合前后仅 motion 四个字段不同的断言 |
| 两份 `*-32frame-4x4-motion-stage2-1x1024-orig16.yaml` | 已具备全部目标值 | 直接复用，不创建重复配置 |
| `scripts/dataset/hf_export/_stage2_orig16_vla_ckpt_export_common.sh::RUNS`、`history_name`、`check_history`、`run_checkpoints` | 已有实体复制、增量上传、最终匿名回读，但身份、读法和 README 固定 split | 增 concat 的 run／bucket／YAML／读法／README 显式配对；校验 1×1024、96 窗、608 槽；复用 `snapshot_sources`、`freeze_sources` 和现有逐文件核验；旧 split 行为不变 |
| 同目录 `run_orig16_ctx32x4x4_concat_60k_ckpt_export.sh`、`run_orig16_modul32x4x4_concat_60k_ckpt_export.sh`（新增） | 缺两条 concat 最终导出包装器 | 默认固定各自 RUN／BUCKET；有 `ATTEMPT_MANIFEST` 时仅接受其中已登记的同角色 run／bucket，调用既有 final 模式；增量入口复用 `run_orig16_stage2_ckpt_inc.sh` |
| 同目录 `orig16_ctx32x4x4_concat_60k_bucket_README.md`、`orig16_modul32x4x4_concat_60k_bucket_README.md`（新增） | 旧发布说明只描述 split | 写明配置、数据来源、12 步列表、恢复方式、依赖和真实完成状态 |
| 同目录 `test_orig16_export_drivers.py` | 已覆盖 split 导出、实体依赖和匿名回读失败 | 加 concat、run／bucket 错配、缺实体依赖、未原子提交的 checkpoint、上传或回读失败、重入身份与摘要核验；补 stage 对恢复器的兼容性检查 |
| `docs/handoff/current.json`、对应训练档案 | 原状态落后于完成日志 | 实施时先按证据同步旧状态，再登记两条新任务及其真实授权、运行句柄和阶段 |

原 split 串行入口保持原身份。通用单条入口保留真实训练退出码；新 concat 编排同时检查诊断结果，发现 FAIL 就写 `WAIT_USER_DECISION`，暂停后续并由主会话询问。即使底层程序退出 0，也不能把诊断失败写成 PASS；用户允许某后续动作后，原 FAIL 及决定均继续留档。

现已采用共享导出链构建独立 stage，替代旧方案的“只生成 manifest”脚本。`encoder_dependencies.json` 只描述外部源，不能代替实体；必须由 `snapshot_sources` 收集注册权重和配置，再由 `freeze_sources` 复制到 stage 的声明相对路径。最终恢复使用 `<stage>/59999`，不使用缺少依赖实体的原 run 根。

## 9. 子代理分配与整合

下表为后续实施分工；延续用户「全部astra high」要求，本任务审查与实施子代理使用 `gpt-6-astra`、`high`。

| 子任务 | 目标／可写文件 | 禁触路径 | 接口、依赖与合并顺序 | 验收与资源 |
|---|---|---|---|---|
| A1 | 合同四项映射和 attempt 身份；仅 `check_orig16_stage2_launch.py`、`test_check_orig16_stage2_launch.py` | 模型、数据、旧日志、启动脚本 | 先合；负责 A2 接续与待决行为的合同测试，其他代理不写该测试文件 | CPU 定向 pytest；无 GPU、端口、训练 run |
| A2 | 仅新增 `run_orig16_concat_ctx_then_modul_60k.sh` | A1 文件、模型、数据、旧入口 | 消费 A1 身份；故障报告和重入方式遵守第 10.4／11 节；第二个合 | `bash -n`；主会话合后两条 dry-run；无 GPU |
| A3 | 第 8 节列出的导出公共体、两个 concat 包装器、两个 README 和导出测试 | 训练合同、模型、原始数据、旧发布物 | 使用 A1 相同 run／attempt 身份；第三个合；独立 stage，旧 split 导出回归 | CPU 夹具、不联网的 SMOKE 合同与消费者检查；不读取凭据、不上传 |
| 主会话 | 配对 YAML 测试、任务状态、档案、整合与实际运行 | 既有输出目录及原数据不可覆盖 | 裁决共享接口；冻结版本；所有 GPU／公开上传／用户询问由主会话调度 | 两条 20 步与 60k；八卡串行；独立备份会话；tmux 前缀 `o16-concat-` |

每次合并前后审查具体差异和测试结果。子代理不得自行起训练、修改依赖环境或写共享 `v1-store`。合后统一生成 `PRE_MERGE_REVIEW`、`POST_MERGE_REVIEW` 与计划一致性记录；本轮没有生成这些实施验收结果。

## 10. 校验闸门与运行步骤

以下命令为后续实施操作说明，**本轮没有执行训练命令**。先完成第 8 节改动和环境恢复，再执行 CPU 检查；正式训练从主副本的固定新提交启动，避免现有入口对实体日志目录的检查拒绝开发副本共享链。

### 10.1 CPU 检查

```bash
UV_CACHE_DIR=/scratch/hongze/.cache/uv uv run --frozen --no-sync python -m pytest \
  tests/training/test_check_orig16_stage2_launch.py \
  tests/training/test_paired_ctx_modul_yaml.py \
  scripts/dataset/hf_export/test_orig16_export_drivers.py \
  tests/datastore/test_motion_split.py -q
bash -n scripts/training/run_orig16_concat_ctx_then_modul_60k.sh
git diff --check
```

需覆盖的负例：concat YAML 配 split run 名、融合方式与 run 名错配、`budget=384` 却只读 1 token、帧布局误用 8×8、global batch/seed/步数漂移、已有 checkpoint 或日志目录、首条训练退出非零、缺少 59999、最终记录失败、诊断 FAIL 后无用户决定而继续、未登记 attempt、上传身份错配、只有外部 encoder 源而 stage 缺实体依赖。另验证已完成 context 的合法重入不会重训它。拒绝原因应明确，不能把无关错误当成预期负例通过。

逐样本输入对拍使用真实库，覆盖 demo、exec、短尾和最大窗口样本；检查同一索引和窗口起点、帧路输入一致，motion 的 `split.reshape(k,1,1024)` 与 concat 原始向量逐位相同。窗口时间码与 mask 按窗口对应：split 同一窗的四个槽位在 concat 中对应一个槽位。padding 无效 key 被屏蔽；注意力 RoPE 位置变化按第 3 节解释，不要求两侧相同。检查训练与在线路径都识别 concat。

血缘命令在已准备好的项目环境中执行：

```bash
UV_CACHE_DIR=/scratch/hongze/.cache/uv uv run --frozen --no-sync python \
  scripts/training/check_data_lineage.py lineage \
  --encoder-ckpt v1-store/external/motionjepa/stage2-full144-1025-a/checkpoint_epoch_105.pt \
  --expected-ckpt-sha256 27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035 \
  --vla-input-manifest v1-store/datasets/16task-orig100-1600ep-9e109419/meta/input_manifest.json \
  --mj-source-manifest /scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/source_manifest.json
```

两条 dry-run 均还须输出 `PAIRED_RUNS_SAME_LIB=PASS input_pin=5d1bfed54377c22e68bc1883ebd35fdf77372129ac1565dd6a78601c1d9965f9`。新的 `NOSPLIT_CONFIG`、`NOSPLIT_INPUT` 为拟新增验收记录；现有 `LAUNCH_CONTRACT`、`MOTION_LAYOUT`、`DATA_LINEAGE`、`TRAIN_FINAL_RECORD` 与恢复工具输出沿用现有语义。

### 10.2 两条 dry-run、短测与正式训练

先设置实施时冻结的 **40 位新提交** `TRAIN_HEAD`，并核对 `git rev-parse HEAD` 与之相同。以下示例中的 `${TRAIN_HEAD:?}` 必须来自真实记录，不能填旧 split Beta 冒充新版本。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
env DRY_RUN=1 TRAIN_HEAD="${TRAIN_HEAD:?}" \
  HISTORY=perceptual-framesamp-ctx-32frame-4x4-motion-stage2-1x1024-orig16.yaml \
  RUN_NAME=orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k \
  FINAL_RECORD_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/validation/orig16-concat-60k/orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k/final-record \
  bash scripts/training/run_orig16_stage2_60k.sh

env DRY_RUN=1 TRAIN_HEAD="${TRAIN_HEAD:?}" \
  HISTORY=perceptual-framesamp-modul-32frame-4x4-motion-stage2-1x1024-orig16.yaml \
  RUN_NAME=orig100-m32x4x4-modul-motion-s2concat1x1024-b128-60k \
  FINAL_RECORD_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/validation/orig16-concat-60k/orig100-m32x4x4-modul-motion-s2concat1x1024-b128-60k/final-record \
  bash scripts/training/run_orig16_stage2_60k.sh
```

短测分别使用同一单条入口，学习率计划仍保持正式 60k 口径。以下为每条短测脚本中的完整训练及恢复命令，`FUSION` 分别取 `ctx`、`modul`，`SMOKE_STAMP` 为当次唯一美东时间戳。长任务实际落入独立 tmux，外层脚本保存日志、真实退出码并按第 11 节发出失败待决事件。旧 `smoke-stage.sh` 写死 split，不能原样复用。

```bash
set -euo pipefail
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
: "${TRAIN_HEAD:?}" "${FUSION:?取ctx或modul}" "${SMOKE_STAMP:?当次美东时间戳}"
[[ "$FUSION" == ctx || "$FUSION" == modul ]]
SMOKE_RUN="o16-smoke-concat-${FUSION}-${SMOKE_STAMP}"
SMOKE_VALIDATION="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/validation/orig16-concat-60k/$SMOKE_RUN"
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export MMEVLA_V1_STORE=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store
export OPENPI_DATA_HOME="$MMEVLA_V1_STORE/models"
export HF_HOME="$MMEVLA_V1_STORE/cache/hf" XDG_CACHE_HOME="$MMEVLA_V1_STORE/cache/xdg"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME JAX_PLATFORMS XLA_FLAGS JAX_COMPILATION_CACHE_DIR
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_ASSETS_CACHE TRANSFORMERS_CACHE

env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_STEPS=20 DRY_RUN=0 \
  TRAIN_HEAD="$TRAIN_HEAD" RUN_NAME="$SMOKE_RUN" \
  HISTORY="perceptual-framesamp-${FUSION}-32frame-4x4-motion-stage2-1x1024-orig16.yaml" \
  FINAL_RECORD_DIR="$SMOKE_VALIDATION/final-record" \
  bash scripts/training/run_orig16_stage2_60k.sh

JAX_PLATFORMS=cpu .venv/bin/python scripts/training/check_orig16_stage2_launch.py \
  --check-final-record "$SMOKE_VALIDATION/final-record/final.json" \
  --run "$SMOKE_RUN" --steps 20 --train-head "$TRAIN_HEAD"
if rg -q '^TRAIN_(NUMERICS|DIAGNOSTIC)=FAIL' "$MMEVLA_V1_STORE/logs/$SMOKE_RUN.log"; then
  printf 'WAIT_USER_DECISION run=%s reason=smoke_diagnostic\n' "$SMOKE_RUN"
  exit 3
fi

export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false HF_HUB_OFFLINE=1
export MMEVLA_JAX_CACHE_DIR="$MMEVLA_V1_STORE/cache/jax/$SMOKE_RUN"
export JAX_COMPILATION_CACHE_DIR="$MMEVLA_JAX_CACHE_DIR"
export CUDA_CACHE_PATH="$MMEVLA_V1_STORE/cache/cuda/$SMOKE_RUN"
uv run --frozen --no-sync python scripts/training/check_stage2_checkpoint.py \
  --mode smoke \
  --checkpoint "$MMEVLA_V1_STORE/train-runs/mme_vla_suite_orig16_b128_60k/$SMOKE_RUN/19" \
  --reference "$SMOKE_VALIDATION/final-record/pre_save_reference.json" \
  --online-gpu 1 --out "$SMOKE_VALIDATION/save-load.json"
```

恢复使用父进程物理 GPU0、在线 encoder GPU1，退出 0 且出现完整 `SAVE_LOAD=PASS` 才算通过；两条都通过后再创建正式 run。报告在验证目录，位于 checkpoint/run 根之外。外层监督脚本对非零退出同样立即留证并询问用户，不能因 `set -e` 退出而遗漏通知。

正式串行入口接口拟定如下，脚本由 A2 实现后才可运行：

```bash
env TRAIN_HEAD="${TRAIN_HEAD:?}" \
  FINAL_ROOT=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/validation/orig16-concat-60k \
  bash scripts/training/run_orig16_concat_ctx_then_modul_60k.sh
```

超过 5 分钟的短测、恢复和正式训练均由主会话放入独立 tmux；正式会话名建议 `o16-concat-train-chain-<美东时间戳>`。实际完整命令、版本、会话名、PID、GPU、开始时间写入 `launch.md`；日志使用无缓冲输出、`tee`、真实管道退出码，运行中固定源码。

正式产物：`v1-store/train-runs/mme_vla_suite_orig16_b128_60k/<run>/`；日志：`v1-store/logs/<run>.log`；验证记录：`v1-store/validation/orig16-concat-60k/<run>/`。启动前逐项确认这些新路径不存在。不得使用覆盖或强制清理旧 run 的参数。

### 10.3 增量公开上传与实体化交付

每条训练实际开始后，主会话为其启动独立备份 tmux，命名 `o16-concat-backup-ctx-<美东时间戳>`、`o16-concat-backup-modul-<美东时间戳>`。CPU 上传进程不占训练 GPU；该条尚未起跑时不创建等待无限期的上传任务。下列为完成第 8 节身份扩展后使用的两个增量命令，分别放入对应备份脚本：

```bash
env RUN=orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k \
  BUCKET=HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2concat1x1024-60k-v1 \
  bash scripts/dataset/hf_export/run_orig16_stage2_ckpt_inc.sh
```

```bash
env RUN=orig100-m32x4x4-modul-motion-s2concat1x1024-b128-60k \
  BUCKET=HongzeFu/robomme-vla-orig100-m32x4x4-modul-motion-s2concat1x1024-60k-v1 \
  bash scripts/dataset/hf_export/run_orig16_stage2_ckpt_inc.sh
```

只有检查点 `_CHECKPOINT_METADATA` 已存在且 `commit_timestamp_nsecs` 有效时才允许复制与发布该步；全套步数为 `5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,59999`。每份 step manifest 同时包含根快照及 encoder 依赖身份，单步恢复不依赖其他 step。

实体打包必须包括：ready 列出的实际 YAML、来源证据与在线恢复样本、ready 本身、两份 encoder 注册依赖，以及各检查点完整 params、assets 和元数据。原 run 及外部依赖只读；`freeze_sources` 复制到独立 stage 并核对大小／SHA，禁止通过符号链接替代依赖或覆盖内容不同的已有文件。磁盘余量按原检查保持至少 `待复制字节 + 40 GiB`；实施前计入两条本地检查点与 stage 的总空间，不自动删除原实验产物。

`INC_END done=12` 只证明增量步数齐全，不代表训练程序已经退出。备份编排还须核对本条训练日志 `EXIT_CODE=0`、run 身份、最终记录与当前无未处理失败，再串行调用对应新增 final 包装器：context 使用 `run_orig16_ctx32x4x4_concat_60k_ckpt_export.sh`，modulation 使用 `run_orig16_modul32x4x4_concat_60k_ckpt_export.sh`。同一 run 的 incremental／final 共用原单实例锁，不能同时持锁运行。final 负责最终清单、完整训练日志及所有对象匿名回读；确认 `HF_VERIFY=PASS`、`CKPT_UPLOAD=PASS steps=12` 和实际退出 0。

用户已授权这两条的增量公开上传；复用现有凭据和传输工具，凭据不进入命令、日志或发布清单。数据集重新上传、旧 bucket 覆盖、额外实验发布均不在此授权范围。

两条训练结束、GPU0/1 现场空闲后，从各自 stage 恢复最终检查点。以下为每条恢复脚本的完整核心命令，`DELIVERY_RUN` 分别取第 4 节两条运行名：

```bash
set -euo pipefail
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
: "${DELIVERY_RUN:?第4节指定运行名}"
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export MMEVLA_V1_STORE=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store
export OPENPI_DATA_HOME="$MMEVLA_V1_STORE/models"
export HF_HOME="$MMEVLA_V1_STORE/cache/hf" XDG_CACHE_HOME="$MMEVLA_V1_STORE/cache/xdg"
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORMS PYTHONPATH PYTHONHOME XLA_FLAGS
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_ASSETS_CACHE TRANSFORMERS_CACHE
export MMEVLA_JAX_CACHE_DIR="$MMEVLA_V1_STORE/cache/jax/$DELIVERY_RUN"
export JAX_COMPILATION_CACHE_DIR="$MMEVLA_JAX_CACHE_DIR"
export CUDA_CACHE_PATH="$MMEVLA_V1_STORE/cache/cuda/$DELIVERY_RUN"
DELIVERY_STAGE="$MMEVLA_V1_STORE/exports/hf-ckpt-$DELIVERY_RUN/stage"
uv run --frozen --no-sync python scripts/training/check_stage2_checkpoint.py \
  --mode delivery --delivery-step 59999 \
  --checkpoint "$DELIVERY_STAGE/59999" \
  --expected-manifest "$DELIVERY_STAGE/manifests/step-59999.json" \
  --online-gpu 1 \
  --out "$MMEVLA_V1_STORE/validation/orig16-concat-60k/$DELIVERY_RUN/delivery.json"
```

tokenizer、VAE 及运行环境仍使用现有按摘要核验的本机缓存；stage 包含 encoder 权重与配置，不宣称仅凭 stage 就包含所有运行软件和缓存。任何复制、上传、回读或恢复失败按第 11 节询问用户，不能把失败最终步骤降级为“可选”。

### 10.4 重新进入与用户决定后的重试

新增串行入口拟提供 `START_ROLE=ctx|modul`，默认 `ctx`。指定 `modul` 时，先复核本次 context 的同版本、同配置、退出日志、最终记录、59999 与未处理诊断状态；仅前条确实完成且不存在待决事项时跳过它。缺证据或存在 FAIL 则询问用户，不能把已有目录当作成功。未发生失败的正常链重入属于既有实验调度，不额外训练已完成条。

失败发生后先暂停、询问用户，保存决定记录到 `v1-store/validation/orig16-concat-60k/decisions/<事件ID>.json`，包括失败阶段、证据路径、原话、允许动作和适用 run。该记录只能由主会话根据真实答复写入；程序生成 `WAIT_USER_DECISION` 不构成许可，超时也不构成许可。

若用户允许从 step 0 重训，使用原实验名加 `-attempt02` 等显式后缀，保留旧日志与 checkpoint。实施时给合同及导出器增加 `ATTEMPT_MANIFEST` 输入，校验实验基名、实际 run、对应 YAML、TRAIN_HEAD、决定记录及独立 bucket；已登记的对应 bucket 使用同样 attempt 后缀，禁止混入首次尝试的发布物。没有已登记决定的后缀一律拒绝。原入口不支持完整优化器恢复，不通过加载旧权重冒充续训。

若仅上传网络中断且用户同意重试，可复用同一 run 的传输账本和 stage，先逐项核对远端身份／内容一致，再补传缺失对象；不重跑训练。所有诊断失败后的恢复动作都以用户的具体决定为限，不自动扩大重试次数或整条重训预算。

## 11. 失败处理、风险与尚未验证事项

| 情况 | 处理 |
|---|---|
| 当前 GPU 查询失败 | 规划阶段已登记；实施前复核驱动、8×A100-80GB 与占用，仍失败则报告并询问，不能启动 GPU 工作 |
| 开发环境缺 `yaml` | 按锁文件准备独立环境，真实检查仍失败则留证并询问；本轮不安装依赖 |
| 数据／encoder 身份不符 | 记录 FAIL，停止受影响工作并询问，不绕过血缘检查 |
| 20 步 OOM、非有限值或保存恢复失败 | 保留现场，暂停正式训练准入，立即询问；不得自行减 batch、改预算、精度或跳过检查 |
| 正式训练退出非零、末步缺失或诊断失败 | 写待决事件、暂停接续第二条并立即询问；真实退出码与诊断结果分别记录 |
| 打包、增量上传或匿名回读失败 | 停止故障发布链及后续新阶段，立即询问；不自动跳过步骤或无限重试 |
| 最终恢复失败 | 标注不可交付，询问修复／复验方式；不将 FAIL 改写为 PASS |
| 原任务仍依赖冻结版本 | 先核实其状态，不擅自删锁或移动 HEAD；需要改变原任务安排时询问用户 |

未知项：concat 的稳定步时、峰值显存、最终指标尚未实测；当前 GPU 可用性未知；既有公开备份长尾未做完整复核。本轮 AST 检查不证明完整运行环境相同；新模型的投影参数形状与 split 不同，不能严格恢复 split 权重作为续训。

**失败询问协议：**故障程序输出 `WAIT_USER_DECISION run=<run> stage=<stage> evidence=<path>` 并写入共享验证目录的待决记录；主会话持续接收日志事件，立即向用户说明“哪项失败、实际退出码／证据、影响哪些后续步骤、可选处理及代价”。两个训练之间、incremental 转 final、最终验收之前都检查未处理事件。用户未答复前保持相关后续步骤暂停，不能以等待时长或代理自行判断代替答复。

若备份失败时独立训练已在运行，立即向用户报告在跑进程及状态，不擅自杀停或重启该进程；是否暂停正在运行的训练由用户裁决。未发生故障、仅上传速度慢的正常长尾继续运行并报告进度，不伪造失败。收到允许继续、修复后重试或终止的决定后，按其范围执行并保留原 FAIL。

正常目标每条只启动一次正式 run。按旧训练合计约 57 小时规划资源观察窗口，正式开始后更新 ETA；超过预期而仍在有效前进时报告实际状态，不自动杀掉训练。出现失败时第 10.4 节只提供恢复方式，不代表预先获得额外整条重训许可。

## 12. 留档与文档验证

实施时每条建立 `docs/training-doc/<run>/launch.md` 与 `result.md`，记录配置差异、数据／代码／环境身份、起止美东时间、检查点列表、步时、参数量、峰值显存、上传目标、manifest、匿名回读及所有真实判定。训练完成、恢复通过和公开备份完成分别记录，三项都满足才标记整条任务完成。诊断失败的证据和用户处理决定单独归档。

版本纪律：功能变更沿用仓库 `commitV<大版本>.<小版本>:` 体例；纯文档用 `docs:`，交接用 `handoff:`。实际版本号在实施时按分支最新历史确定，不能在计划阶段虚构提交。当前文档落在用户指定的开发副本中，本轮不合并训练代码、不移动主副本 HEAD。

文档交付检查：恰好保留两个一级标题；本地已存在链接可解析；新入口明确标注为拟新增；命令中的现有参数与当前脚本一致；新增文件执行空白检查。两条新训练和其验收结果均不得提前标为完成。

本次修订对应[对抗审查记录](1030-orig16-context-modul-nosplit-adversarial-review.md)：F1 改为实体 stage 后恢复；F2 补受控重入及用户决定后的 attempt；F3 补全两条短测恢复命令与环境；F4 补充 RoPE 位置影响。以上为方案层修正，代码尚未实施、运行验证尚未完成，不能将原审查失败改写为已实测通过。
