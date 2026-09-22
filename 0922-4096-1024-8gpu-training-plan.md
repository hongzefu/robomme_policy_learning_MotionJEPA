# 4096 → 1024 token 八卡顺序训练计划

创建日期：2026-09-22（America/New_York）。本轮交付为本根目录计划文件，尚未修改代码、配置或启动训练。用户原话：「给出再训练4096和1024的计划 配置和2048一致 跑完4096跑1024 都是8卡 计划落在根目录 /scratch/hongze/robomme_policy_learning_MotionJEPA」。

## 第一部分（给人看）

### 1. 训练安排与参照版本

先训练 **4096 token（64 帧 × 每帧 64 token）**，完整完成 80,000 步并验收保存结果后，再训练 **1024 token（16 帧 × 每帧 64 token）**。两档均独占本机 GPU `0,1,2,3,4,5,6,7`，沿用 2048 档的全局 batch 128、worker 16、FSDP 8 和其余训练配置。每档独立从同一 `pi05_base` 初始化；4096 的权重不传给 1024。

参照是已完成的 `v2-1600ep-m32x8x8-modul-b128-80k`，正式起跑 Beta 为 `55647ff33c8ddb9ec324fdbcee8bd1491456725b`。以其 [实际配置记录](docs/training-doc/v2-1600ep-m32x8x8-modul-b128-80k/records/approval.actual.json)、[启动档案](docs/training-doc/v2-1600ep-m32x8x8-modul-b128-80k/launch.md) 和 [完成结果](docs/training-doc/v2-1600ep-m32x8x8-modul-b128-80k/result.md) 为配置依据；本计划核对时仓库 HEAD 为 `f711d68fcb23ac2cdd8b776c1f051f9916173e36`，工作区为空。

已实测本机为环境 B：8 × `NVIDIA A100-SXM4-80GB`，`/scratch` 为 `/dev/md0`、XFS、本地 NVMe RAID；编写时八卡显存均为 0 MiB，scratch 可用约 1.2 TiB，`/dev/shm` 总量约 561 GiB。仓库、数据、资产和新产物均使用 `/scratch/hongze/robomme_policy_learning_MotionJEPA/`，运行产物统一落 `v1-store/`。这些是编写时快照，每档起跑前仍重查。

本轮只新增这份文档，从本文件标题至第 9 节；下文的代码、验证和长训练都是后续实施安排。计划所描述的是训练实施，因此按两部分组织：第一部分给出训练安排、完整配置及关键保证的具体依据；第二部分列出实现文件、命令、验证和交付。新名称拟为 `v2-1600ep-m64x8x8-modul-b128-80k` 和 `v2-1600ep-m16x8x8-modul-b128-80k`，本次已检查两处训练输出均不存在。正式实施前依据 [AGENTS.md](AGENTS.md) 第 6 条一次确认两个名称和实施范围；若用户直接认可本计划及这两个名称，沿用该决定，4096 结束后不再重复确认。2048 档的历史同意记录不能代替本轮实施授权。

### 2. 三档配置对照：只改变记忆预算

正式配置仍为 [training/config.py](src/mme_vla_suite/training/config.py) 中 `_CONFIGS` 的 `mme_vla_suite_b128_80k`。不改这份具名配置和已有 2048 YAML；新增的两份 history YAML 从 [32frame 配置](src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-32frame-8x8.yaml) 派生，解析后只允许 `budget` 一个键不同。启动参数只切换新 YAML、run 名和相应产物路径，不另加学习率、batch、worker 或训练步数覆盖。

| 项目 | 2048 参照 | 先跑 4096 | 后跑 1024 |
|---|---|---|---|
| `budget` | 2048 | 4096 | 1024 |
| 最大历史帧数 | 32 | 64 | 16 |
| history YAML | `perceptual-framesamp-modul-32frame-8x8.yaml` | 拟新增 `perceptual-framesamp-modul-64frame-8x8.yaml` | 拟新增 `perceptual-framesamp-modul-16frame-8x8.yaml` |
| `token_per_image` / `num_views` | 64 / 1 | 相同 | 相同 |
| `memory_token_dim` / 图像输入维度 | 1024 / 2048 | 相同 | 相同 |
| 记忆类型 / 融合 / motion | perceptual、frame_sampling、modulation、关闭 | 相同 | 相同 |
| global batch / GPU / FSDP | 128 / 8 / 8 | 相同 | 相同 |
| 总步数 / seed / workers | 80000 / 42 / 16 | 相同 | 相同 |
| warmup / peak LR / decay LR / decay steps | 5000 / 5e-5 / 5e-5 / 80000 | 相同 | 相同 |
| EMA / 梯度裁剪 | 0.999 / 1.0 | 相同 | 相同 |
| 日志 / 保存 / 保留间隔 | 100 / 5000 / 5000 | 相同 | 相同 |
| 初始化 | `pi05_base/params`，独立初始化 | 相同 | 相同 |

完整固定项如下，不能把数值相近的其他配置当作参照：AdamW `b1=0.9,b2=0.95,eps=1e-8,weight_decay=1e-10`；模型 `pi05=true,paligemma_variant=gemma_2b,action_expert_variant=gemma_300m,memory_expert_variant=gemma_150m`，bf16 计算，`action_dim=32,action_horizon=20,max_token_len=64,use_history=true,discrete_state_input=false`，冻结 `.*img.*`。history 保持 `streaming_obs_horizon=16,pool_type=mean,use_pos_emb=true,use_state_emb=false`，位置输入及隐藏维度均为 768。`streaming_obs_horizon` 不随 64/16 帧档位改变。学习率仍使用 `CosineDecaySchedule`；因 peak 与 decay LR 相同，warmup 后为 5e-5。

数据加载沿用 [create_data_loader](src/mme_vla_suite/training/dataloader.py) 与 [TorchDataLoader](src/openpi/training/data_loader.py)：`spawn`、`persistent_workers=true`、`prefetch_factor=2`、`pin_memory=false`、`shuffle=true`、`drop_last=true`、`_collate_fn_shm`。运行方式是**单个 JAX 训练进程使用八卡**，`make_mesh` 得到 `batch=1,fsdp=8`，每设备数据分片 16 个样本；不改成八个 `torchrun` 进程，也不加入梯度累积。

正式配置比对必须比较完整解析对象，与 2048 的实际记录逐项核对；仅放行 `budget`、history 文件名、run 名、输出/日志/缓存路径等运行身份差异。模型、loss、优化器、冻结参数、数据和初始化资产都不应发生额外变化。两份 YAML 与完整 CLI 解析结果分别计算摘要并绑定自己的起跑记录，不能共用 2048 的 config SHA。

### 3. 数据、输入链路与两条核心保证

两档复用 `v1-store/datasets/4task-v2-1600ep-604f16da/`，其中 `--dataset-path` 为 `framesamp-8x8`，`MMEVLA_FRAMESAMP_SOURCE` 为 `source`，`MMEVLA_FRAMESAMP_MANIFEST` 为 `meta/episode_manifest.json`。规模仍为四任务 1600 episode、605611 个执行样本、1192918 帧，packed store 为 `verified/full`；沿用整个 manifest 的执行样本，不另造训练/验证划分。归一化根为 `v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da`，`asset_id=robomme`；初始权重位于 `v1-store/models/openpi-assets/checkpoints/pi05_base/params`。上述实体路径已只读检查。

本次对本地文件重新计算 SHA256，均与 2048 档案一致：

| 文件 | SHA256 |
|---|---|
| `meta/episode_manifest.json` | `df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482` |
| `framesamp-8x8/meta/store_meta.json` | `f7677e69e5c473ab2962a5ac05a5909348c2f96736152b7217b77f0d2eb4231a` |
| `robomme/norm_stats.json` | `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173` |

这些摘要核对不等于重新扫描全部大文件；起跑仍需使用现有 preflight 检查 store 状态、清单关联及资产。无需重建数据、重新抽特征或下载模型。

改前 2048 链路如下，箭头上的变换沿用现有实现：

```text
同一 source / manifest / framesamp-8x8
  → FrameSampDataset：每样本选最多32帧，每帧64 token；读取原始特征数值不变
  → 单样本 image(2048,2048) bf16 + pos(2048,768) f32
            + state(2048,8) f64（既有归一化结果）+ mask(2048,) bool
  → _collate_fn_shm：全局batch128，shape前加128；保留各键既有dtype语义
  → TorchDataLoader / JAX交付：八卡分片；state由f64转f32，沿用原有转换
  → PerceptualMemory / HistoryPi0：memory(128,2048,1024)
  → modulation动作分支：20个action query；同一动作监督、loss及优化器
```

改后使用相同数据、相同算法和同一交付链，仅预算决定选帧数量与记忆轴长度：

```text
同一 source / manifest / framesamp-8x8
  → FrameSampDataset：4096选最多64帧；1024选最多16帧；每帧仍64 token
  → 单样本 image(B,2048) bf16 + pos(B,768) f32
            + state(B,8) f64 + mask(B,) bool，B分别为4096、1024
  → 同一 _collate_fn_shm / JAX交付：batch128、八卡分片、state仍由f64转f32
  → 同一模型：memory(128,B,1024)，只扩缩token轴，不改通道或参数形状
  → 同一20个action query及动作监督；历史内容与query位置按预算产生预期变化
```

下表只计算Dataset归一化后、collate阶段四个history输入张量的逻辑字节量，其中state为f64；不含图像当前观测、动作、主机队列、参数、梯度或模型中间张量，不能当成实际显存：

| 预算 | image 字节/样本 | pos 字节/样本 | state 字节/样本 | mask 字节/样本 | 合计字节/样本 | batch128 合计 |
|---|---:|---:|---:|---:|---:|---:|
| 2048 | 8388608 | 6291456 | 131072 | 2048 | 14813184 | 1896087552 |
| 4096 | 16777216 | 12582912 | 262144 | 4096 | 29626368 | 3792175104 |
| 1024 | 4194304 | 3145728 | 65536 | 1024 | 7406592 | 948043776 |

JAX交付后state转成f32，三档每样本四键合计分别为14747648、29495296、7373824字节。按worker16×prefetch2只计算主机队列内四键，2048/4096/1024分别约56.51/113.02/28.25GiB；实际峰值还包含其他字段和副本，需实测。

同一被选帧的底层特征、观测和标签不改数；选帧集合、padding 数量和动作 query 的 RoPE 位置有意改变。`MemoryAttention` 的动作 query 位置随预算从 `2048..2067` 改为 `4096..4115` 或 `1024..1043`，因此不要求三档 loss、梯度或输出相等。等价验证只在**相同预算**的源 NPY 参考链与 packed 链之间进行。

2048 档案只证明该库样本均能提供 32 帧，不能据此声称 4096 均满 64 帧。实施时先由 episode/执行索引统计可用历史长度，记录 16/32/64 帧各档的满额比例；对 64 帧不足的真实样本验证既有 padding。合成案例再覆盖 0、1、15、16、17、31、32、33、63、64、65 帧及长历史边界，不伪造生产清单或重建库。

#### 保证一：两档继承2048的训练配置，额外差异在起跑前被拒绝

**配置文件层只扩预算。** 拟新增的 `perceptual-framesamp-modul-64frame-8x8.yaml`、`perceptual-framesamp-modul-16frame-8x8.yaml` 解析后，逐键与既有32frame配置比较，只放行 `budget=4096` 或 `budget=1024`。例如4096档的拟新增判定行为 `BUDGET_CONFIG=PASS budget=4096 baseline=2048 diff_keys=budget`。`memory_token_dim=1024`、`img.input_dim=2048`、`streaming_obs_horizon=16` 任一被顺带改变都应失败；否则会混入通道宽度或观察节奏的变化。

**实际命令层再次检查最终生效值。** [modul_launch_contract.py](scripts/training/modul_launch_contract.py) 的 `parse_config`、`validate_config` 目前只绑定2048，实施时扩展为三档映射，由 [preflight_train_launch.py](scripts/training/preflight_train_launch.py) 核对即将交给训练的同一份 `TRAIN_ARGS`。两档都必须输出既有格式 `LAUNCH_CONFIG=PASS mode=prod steps=80000 batch=128 workers=16 fsdp=8`，并在解析记录中证明seed42、warmup5000、lr5e-5、EMA0.999、pi05_base初始化及第2节其他固定值全部一致。YAML正确但CLI偷偷覆盖batch、学习率或步数，也不能起跑。

**数据与版本层绑定本次实际输入。** 同一preflight重新检查第3节三个文件摘要、verified/full数据规模、全新输出根及clean TRAIN_HEAD，把本档YAML、runner和测速报告摘要写入各自记录。不能借用2048的旧批准JSON或旧测速报告。这些检查的对象是即将执行的代码、配置和数据，而不只是文档里列出的值。

#### 保证二：4096完整成功并释放八卡后，1024才能启动

**先以真实训练产物判定完成。** 拟新增 `scripts/training/tests/check_modul_completion.py` 必须同时检查4096正常 `EXIT_CODE=0`、80000步完成、5000到75000每5000一步加79999共16份checkpoint、800条有限指标以及最终79999真实加载通过。训练入口 [train.py](scripts/training/train.py) 的 `main` 已在结束前调用 `checkpoint_manager.wait_until_finished()`；顺序调度器等待整个训练入口返回后才验收，不在看到最后一条日志时提前释放下一档。完成检查拟输出 `RUN_COMPLETED=PASS run=v2-1600ep-m64x8x8-modul-b128-80k budget=4096 steps=80000 checkpoints=16 final=79999`，并把最终文件摘要写入JSON。

**再由独立交接检查控制下一次调用。** 拟新增 `scripts/training/prod/run_modul4096_then1024.sh` 验证该完成JSON绑定的是本轮4096，而后检查训练/采样PID退出、八卡显存均为0、剩余空间至少400GB及代码/环境未变化，才开始1024的GPU验证、测速和正式训练。4096报错、缺最终checkpoint、加载失败或完成记录错配时，负例必须得到 `SECOND_TRAIN_CALLS=0`；本轮授权一旦覆盖完整顺序和两个run名，不在正常交接点重复请求确认。

**时间与容量以各档本机实测为准。** 2048的22小时58分52秒和约190GB检查点只是基线；4096的输入逻辑字节翻倍不意味着整模型耗时必然翻倍，也不证明显存能容纳。两档各做八卡20步容量和1000步测速，主统计窗口固定100–899步，报告GPU均值、0%占比及慢步分层。若原b128/w16/FSDP8不能通过容量，队列停止并报告，不自动改变已定配置。上述两条保证是待实施及验证的要求，不是本轮已运行的结果。

## 第二部分（技术细节，供 agent 追踪）

### 4. 待实施的最小改动

当前代码不能直接启动两档。[FrameSampDataset.__init__](src/mme_vla_suite/training/framesamp_dataset.py) 的新档白名单只接受 `(2048,64,1)`、modulation、无 motion；[run_modul2048.sh](scripts/training/prod/run_modul2048.sh) 和 [modul_launch_contract.py](scripts/training/modul_launch_contract.py) 还固定了 2048 YAML 和旧 run 名。计划按下表扩展，所有新增接口在本轮都只是设计。

| 文件/稳定锚点 | 拟改动与边界 |
|---|---|
| `src/mme_vla_suite/models/config/robomme/` | 新增 `perceptual-framesamp-modul-64frame-8x8.yaml`、`perceptual-framesamp-modul-16frame-8x8.yaml`；与32frame解析值只差budget。 |
| `training/framesamp_dataset.py::FrameSampDataset.__init__` | 精确加入 `(4096,64,1)`、`(1024,64,1)`，只允许modulation且无motion；旧512/2048接受集合与拒绝条件保持。 |
| `scripts/training/tests/ref_npy_dataset.py::RefNpyFrameSampDataset.__init__` | 独立参考链加入相同两档；参考采样与拼装不调用生产Dataset以免同源错误互相通过。 |
| `scripts/training/g0/bench_train_steps.py` | 扩展history文件白名单；复用现有refnpy/packed、索引、batch、初始化和参数记录能力。 |
| `scripts/training/tests/check_32frame_modul.py` | 将写死32/2048的检查参数化为显式预算，保留2048默认；新增有界真实样本入口，覆盖选帧、collate、padding、在线缓存、初始化、RoPE及保存加载。 |
| `scripts/training/tests/check_modul_train_records.py::validate` | 移除仅支持两卡/w4/fsdp2的验证器限制，新增明确的预算、worker、FSDP期望参数；校验八个不重复设备、mask实际形状与逐样本有效数，不能继续硬判mask全True。旧两卡默认仍可回归。 |
| `scripts/training/modul_launch_contract.py`、`preflight_train_launch.py` | 显式映射1024/2048/4096与16/32/64frame配置；正式run及报告按本次记录绑定。每个新档都强制进入launch-mode检查，防止换文件名绕过闸门；保留完整配置、数据、初始化、clean HEAD、资源及授权校验。 |
| `scripts/training/tests/check_modul_speed.py::report` | 将budget=2048断言和图像逻辑读取带宽公式改为从已校验预算推导；保持1000步、800步稳态与资源/报告完整性判据。 |
| 拟新增 `scripts/training/prod/run_modul_budget.sh` | 接受显式预算和本档run/HEAD/YAML SHA/报告，复用2048的环境及训练CLI纪律；保留原2048入口兼容，不让它静默启动新档。 |
| 拟新增 `scripts/training/prod/run_modul4096_then1024.sh`、`scripts/training/tests/check_modul_completion.py` | 严格顺序调度、失败停止，验收4096的真实产物后才进入1024；完成记录绑定run、预算、HEAD及checkpoint摘要。 |
| `test_pack_guards.py`、`test_modul_launch_guards.py`、`test_modul_speed.py`及对应新增用例 | 覆盖错误预算/文件名/配置/旧报告/旧run/失败交接，断言训练入口调用次数为0。 |

在线 [FrameSampMemory](src/mme_vla_suite/policies/framesamp_memory.py) 的 `get_frame_sampling_indices`、`_prepare_frame_sampling` 已按预算计算，不存在需要扩展的2048专属白名单；[HistoryPi0](src/mme_vla_suite/models/integration/history_pi0.py) 也读取配置预算。先增加两档回归，除非验证暴露缺陷，不修改这些实现。冻结的旧建库 memory buffer 不属于本轮生产改动。

### 5. 验证顺序与通过标准

所有输入验证先于相应训练。共同 CPU 检查可以提前覆盖两档；GPU任务严格按“基线回归 → 4096验证、测速、正式80k和验收 → 1024验证、测速、正式80k和验收”顺序执行，不在4096正式训练期间抢卡验证1024。每个真实训练验证也使用八卡、b128/w16/FSDP8；短验证只通过自己的启动覆盖缩短步数、增加取证和关闭W&B，不修改正式默认。

| 阶段 | 验证内容 | 通过标准 |
|---|---|---|
| CPU配置及拒绝测试 | 三档完整解析对照，旧512/2048守卫回归，错误融合方式、motion、浮点budget、错误run/数据/摘要、失败交接负例 | 两新YAML只差budget，输出第3节拟新增`BUDGET_CONFIG=PASS`；错误输入全部拒绝且训练调用0次。 |
| 真实输入对拍 | 固定seed42；四任务每个episode至少取首/中/末执行样本并去重；加入16/32/64帧边界及前20个实际batch；保存精确索引清单 | 同预算refnpy与packed的索引顺序、选帧索引、逐键shape/dtype/原始字节完全相同，mismatches=0。合成短历史与在线链另测。 |
| 模型语义检查 | 初始化参数树；mask垃圾值不影响输出；各预算独立RoPE oracle；短历史padding和collate交付 | 相同seed初态61叶逐位相同；mask无效位扰动不改输出；oracle `atol=1e-6,rtol=1e-5`。跨预算非退化前向允许不同。 |
| 2048旧档训练回归 | 在改前/改后代码、同一八卡环境各跑20步，另各补1步保存第一次更新现场 | 同一输入、相同初态；loss/梯度及params/optimizer/EMA摘要逐步一致，证明扩白名单未改旧档。 |
| 新档真实训练对拍 | 每档refnpy与packed各20步，各补1步；固定数据、seed、设备列表和运行环境 | 输入和初态逐位一致；五项标量及完整状态摘要一致，覆盖首次更新与第20次更新；发现差异先定位，不能按新预算放宽。 |
| 保存与加载 | 每档packed独立100步，真实保存末步，再经`check_config_provenance.py`及参数化加载检查真实读取 | 配置/数据/61叶参数/EMA内容与本run现场相符；可执行动作路径。旧run或初始化权重冒充时必须失败，不能只比较参数树形状。 |
| 八卡生产容量 | 每档独立20步，原生产模型、b128/w16/FSDP8及一次真实保存 | 无OOM、NaN/Inf、worker故障、共享内存错误；峰值记录齐全；不因容量失败自动降低batch、worker或预算。 |
| 八卡独立测速 | 每档独立1000步，0–99预热、100–899稳态、900–999收尾，记录真实保存 | 800步完整、`SPEED_REPORT=READY`；共享内存峰值占比≤0.70；报告吞吐、GPU/主机数据和本档80k预计时长。 |

正确性对拍使用既有 `XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`，容量、测速及正式阶段则显式unset，不能混用速度。20步主记录核对初始化 `state_step=0` 及更新后 `2..20`，独立1步补跑补齐 `state_step=1`；比较五标量hex与完整TrainState 201叶（参数61、可训练参数38、EMA61、Adam一二阶各38及计数3，按既有树结构核对），不是只比loss曲线。参数61叶与可训练38叶存在包含关系，201叶计数为61+61+38+38+3。

源NPY侧复用 `BENCH_DATASET_IMPL=refnpy`、`BENCH_REF_SOURCE`、`BENCH_REF_MANIFEST`，其 `--dataset-path` 指向source；packed侧指向framesamp-8x8。两侧都通过 `bench_train_steps.py` 明确覆盖 `--num-train-steps 20 --log-interval 1 --save-interval 1 --no-wandb-enabled`，b128/w16/FSDP8/seed42从同一具名配置取得并再核实。新增预算及八卡验证器接口完成前，不能直接使用旧验证命令宣称通过。

100步验证可沿用 `BENCH_SAVE_FINAL_CKPT=1 BENCH_FINAL_STEP=99 BENCH_STATE_DUMP_STEPS=0,99`，`BENCH_STATE_DUMP_DIR` 指向本轮新目录。既有harness将该验证checkpoint命名为 `999/`，真实状态是 `state_step=100`，不把目录名当更新次数。保存现场EMA转bf16后与加载叶逐位相同，10个记忆叶在fp32及加载bf16下均须区别于初始权重；固定noise、`num_steps=10` 的动作RMS差≤6.8e-5。以上是验证run的约定，正式末步仍为79999。

逐位判据是本计划的首选通过条件。若八卡重复执行存在非确定性，先做同实现的A/A对照并保留真实逐叶数组，用 [compare_baseline.py](scripts/training/g0/compare_baseline.py) 的既有噪声判据分析（标量绝对底线loss `1e-6`、其余四项 `1e-5`，噪声余量倍数2）；这只能提供诊断证据，未解释的参数/梯度差异仍阻断等价结论。不得拿2048与4096/1024之间的预期语义差异充当同预算误差的解释。

有界样本验证不直接使用 `dump_fixture_samples.py` 的 `DTYPE_DUMP_LIMIT`：该入口仍可能遍历全库帧及605611个pkl。应新增明确只读取指定样本的模式，并记录实际读取数量。这里得到的是有界真实输入证据，不声称所有样本的特征字节均已重扫。

所有预计超过5分钟的验证、基准、诊断和训练均从clean HEAD在独立detached tmux中执行，按run留档；预计短于5分钟的测试若超时也保留真实启动状态和结果。短测的临时run只有在确认归属且满足清理规则时才删除，较长验证结果不按冒烟清空。验证器修改本身须先用可判定的正负例检查，再用于生产验收。

### 6. 测速、资源和正式入口

2048实测总耗时为22小时58分52秒；正式step100→79900窗口为1.030577秒/步、124.2023 samples/s，含周期保存。同窗500ms采样的八卡GPU利用率均值98.4911%、0%占比0.2321%。这是同机同数据的参照，4096和1024的耗时尚未实测，不按token比例线性外推。2048显存读数包含`.95`的JAX预分配，也不能证明4096一定放得下。

两档分别复用 `check_modul_speed.py` 的1000步方法，明确记录8卡型号、`/dev/md0`存储、b128/w16/prefetch2、预热与稳态范围、500ms GPU采样、吞吐、逐步慢步分层、主机RAM、`/dev/shm`及真实磁盘读取。GPU报告含均值、0%占比、慢步/其他步均值，NVML重复值不视作新增独立证据。80k ETA分开列初始化/编译、稳态更新和16次保存估计，不称为已完成的时长；正式约300步后用原生log100时间戳复核，不往正式训练加入逐步同步或profiler。2048以前的“利用率大于50%可直接起跑”授权只描述旧run，不套用成新两档的同意记录。

2048的16份checkpoint合计190052280479字节，两新run按相同参数树估计共约380GB权重，另加验证保存、缓存和日志。每档起跑沿用至少400GB空闲空间闸，4096开始前还要预算两档累计保留量；1024开始前重新计算。不得通过删除已有run来满足空间闸。当前约1.2TiB空闲只能说明具备初步空间条件，不能代替执行时检查。

拟新增通用入口的接口如下，**当前文件尚不存在，不可直接执行**；占位符须在实现、验证及本轮实施授权落实后替换为真实值：

```bash
# 4096档；实际运行由顺序调度器放入detached tmux。
bash scripts/training/prod/run_modul_budget.sh \
  prod 4096 v2-1600ep-m64x8x8-modul-b128-80k \
  <TRAIN_HEAD> <4096_YAML_SHA256> <4096_APPROVAL_JSON> <4096_REPORT_JSON>

# 仅在4096完成验收及本档验证、测速通过后，由顺序调度器调用。
bash scripts/training/prod/run_modul_budget.sh \
  prod 1024 v2-1600ep-m16x8x8-modul-b128-80k \
  <TRAIN_HEAD> <1024_YAML_SHA256> <1024_APPROVAL_JSON> <1024_REPORT_JSON>
```

入口内部的真实训练仍为 `uv run --no-sync python scripts/training/train.py mme_vla_suite_b128_80k ...`。保留2048入口的 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`、`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`、`TRAIN_TIMING_STEPS=0`、`TZ=UTC`，正式W&B开启；显式unset `PYTHONPATH,JAX_PLATFORMS,XLA_FLAGS,MMEVLA_MOTION_STORE,MMEVLA_FRAMESAMP_ALLOW_SUBSET`。各类缓存逐项落 `v1-store/cache/`，每run独立JAX缓存；不覆盖HOME、不在训练期间同步依赖。入口内层 `set -euo pipefail`，外层用 `PYTHONUNBUFFERED=1`、`tee` 和管道整体退出码记录 `EXIT_CODE=`，日志写入失败也不得成为成功。

每档正式起跑由preflight重新解析与训练**同一份** `TRAIN_ARGS`，同时核对预算、YAML SHA、完整配置、初始化权重、manifest/norm/store摘要、自己的run和新输出根、runner/报告摘要、clean HEAD、八卡空闲及空间。测速HEAD至正式HEAD只可有文档变化，源码/启动器/依赖一旦改变则重新验证测速。新档的同意记录应忠实引用本轮实施决定及其范围，分别绑定本档的实际报告和入口，不伪造新的用户原话。

### 7. 4096完成后才转入1024的硬条件

顺序调度器按以下状态运行，所有状态和完成记录写入本轮唯一的 `v1-store/bench/modul-budget-sweep/<批次名>/`；正式权重分别进入 `v1-store/train-runs/mme_vla_suite_b128_80k/<run_name>/`。

```text
共同CPU检查与2048旧档回归
  → 4096输入/模型/20步对拍/100步保存加载
  → 4096八卡容量20步与测速1000步
  → 4096最终起跑检查 → 正式80000步 → 完成验收
  → 八卡释放、重查HEAD/环境/数据/磁盘
  → 1024输入/模型/20步对拍/100步保存加载
  → 1024八卡容量20步与测速1000步
  → 1024最终起跑检查 → 正式80000步 → 完成验收
```

4096交接必须同时满足：训练进程正常退出且唯一终态为 `EXIT_CODE=0`；`train.py::main` 的 `checkpoint_manager.wait_until_finished()` 已返回；保存目录集合为 `5000,10000,…,75000,79999` 共16份且元数据/数据完整；正常log100共800条记录、五项标量全部有限；最终79999经过真实加载和本档配置/来源检查。最后一条常规日志是79900，不能将它当成79999单步loss。预算不改变61叶参数形状，因而“目录存在、61叶形状正确”不足以证明拿到了本run的权重，完成检查还须绑定run来源、文件摘要和保存记录。

通过上述检查后，调度器核对训练PID和本轮GPU采样PID均退出、八卡释放，并生成绑定4096 run/HEAD/budget/最终checkpoint摘要的完成JSON及第3节的 `RUN_COMPLETED=PASS` 判定；1024起跑门重新验证该JSON及其产物。只有4096完成记录有效且1024自身检查全过，才能调用1024训练入口。任一失败均停止队列并保留现场；不跳过4096、不自动续训、不自动改超参、不用tmux会话消失判断成功。当前checkpoint主要保存params/assets，不完整保存优化器状态，不能把载入权重称为精确断点续训。

调度器的负例至少覆盖：4096非零退出、被终止而无退出码、最终checkpoint缺失或加载失败、仅有79900日志、旧run伪造完成JSON、预算/HEAD不匹配、1024配置变化、GPU未释放及磁盘不足。上述任一情况都应得到 `SECOND_TRAIN_CALLS=0`。零训练负例使用替身入口核对控制流；正向完成验收必须读真实训练产物。

预计长任务的tmux全名采用 `modul-sweep-<UTC>`、`m4096-<阶段>-<UTC>`、`m1024-<阶段>-<UTC>`，启动后归档展开后的完整名称和精确PID；存活检查用 `tmux has-session -t '=完整会话名'`。日志监听逐级行缓冲，只关注启动、判定、异常和完成事件。清理只针对本轮清单里的精确会话，不能全局结束tmux或清理其他代理现场。

### 8. 提交、留档与最终交付

实施时先固定BASE及环境指纹，完成新增配置、最小守卫和验证工具；代码验证通过后按当时 `git log` 顺延功能版本。两档代码和完整顺序入口提前提交，正式起跑前固定Beta和clean TRAIN_HEAD；可以由同一个Beta锁定这轮顺序实验，两档分别记录自己的实际启动时间、命令、预算和摘要，结束结论与该Beta配对，不amend或squash掉锚点。

按照目标仓库现有格式，为两个正式run分别建立 `docs/training-doc/<run_name>/launch.md`、`result.md`、`records/`，更新 [训练档案索引](docs/training-doc/README.md)。起跑信息在实际启动时记录，1024不能提前填为“已启动”；运行期间需要提交文档时使用独立开发副本及独立环境，主训练副本源码、依赖与HEAD保持稳定。完成检查和交接运行记录先落本轮 `v1-store/`，让两档顺序执行不因写入未提交文档而破坏clean HEAD。

每档归档本轮用户原话、代码完整hash、真实CLI与全部环境覆盖、预算/配置/数据摘要、硬件和存储、验证/容量/测速结果、初始化来源、500ms利用率数据、正式清洗日志和800条指标、16份checkpoint清单、最终真实加载结果、退出码及计划外事件。Git可还原的脚本/YAML只引用提交和路径，不复制到档案；权重留在 `v1-store/train-runs/`。完整保留2048及其他run，不自动导出到HuggingFace或开展策略rollout。

最终完成标准为两个新run均80000步正常退出、各16份checkpoint通过、各最终权重真实加载通过、配置差异仅为本计划所列预算及身份字段、4096结束早于1024启动且全程资源口径均为八卡、档案齐全。训练loss和利用率用于检查训练过程，不作为策略成功率结论；后续rollout范围另定。

### 9. 本次计划文档的验证与提交

本次只新增 `0922-4096-1024-8gpu-training-plan.md`。提交前核对所引现有文件、配置键、数据摘要、命令中拟新增接口标记、逻辑字节计算和顺序条件，执行 `git diff --check` 及暂存差异检查；纯文档阶段不启动仓库训练/验证脚本、不同步依赖。只用明确文件路径暂存，中文 `docs:` 提交记录用户原话、核对依据和验证结果；按目标仓库既有规则推送当前分支的既有upstream，不改远端或凭据。实施与正式训练仍待本轮计划后的明确指令。
