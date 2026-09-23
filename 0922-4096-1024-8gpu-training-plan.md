# 4096 → 1024 token 八卡顺序训练计划

创建日期：2026-09-22（America/New_York）。本轮交付为本根目录计划文件，尚未修改代码、配置或启动训练。用户原话：「给出再训练4096和1024的计划 配置和2048一致 跑完4096跑1024 都是8卡 计划落在根目录 /scratch/hongze/robomme_policy_learning_MotionJEPA」。

修订（2026-09-22）：用户原话「文本太多细节太少第一部分 参考…0920-32frame-8x8-modul-2048-plan.md对齐风格」「主要是每一节的叙述方式 不是章节表格这种大的排版」。第一部分保留原章节与表格，逐段改为结论先行加代码级细节的叙述；按 `exec_start_idx` 最小 100 的实测，更正 64 帧满额口径，并同步第二部分三处相应表述。

修订（2026-09-23，Codex 审计 `AUDIT_BASE=0f53f6390ab2e5b164c1d0f7a8ab21e293a8498a`）：采纳全部 6 项意见，引用的代码点已逐一核实。P1：最终保存现场记录绑定权重来源；尾窗 NaN 与含 NaN 的 checkpoint 判失败；测速报告按报告内配置语义比对预算；完整配置记录改为带版本格式，并从 2048 Beta 补建基线。P2：训练记录验证器改读实际解析配置；GPU 采样校验有限性与唯一时间戳。另将「0 帧」更正为零基帧号 `t=0` 已有 1 帧。落点在第 1、2 节，第 3 节两条保证，以及第二部分第 4–7 节。

## 第一部分（给人看）

### 1. 训练安排与参照版本

**两档都是 2048 run 的「只换预算」复刻，先 4096、后 1024，串行不并行。** 先训 4096 token（最多 64 帧 × 每帧 64 token），80,000 步跑完并通过完成验收后，再训 1024 token（最多 16 帧 × 64 token）。两档都独占 GPU `0–7`：单个 JAX 训练进程、`fsdp=8`、全局 batch 128、`num_workers=16`，各自从 `pi05_base/params` 重新初始化——4096 的权重不传给 1024。

**参照对象是刚跑完的 2048 run，不是任何数值相近的旧配置。** `v2-1600ep-m32x8x8-modul-b128-80k` 从 Beta `55647ff33c8ddb9ec324fdbcee8bd1491456725b` 起跑，2026-09-22 06:19:30 UTC 以 `EXIT_CODE=0` 结束，全程 22 小时 58 分 52 秒；step100→79900 平均 1.030577 秒/步、124.2023 samples/s，八卡 GPU 利用率均值 98.4911%。「配置和 2048 一致」以 Beta 提交上的具名配置、32frame YAML 与起跑 CLI 为准；它的 [实际配置记录](docs/training-doc/v2-1600ep-m32x8x8-modul-b128-80k/records/approval.actual.json)（config SHA `b2acd4a95c6c9fb5649fe0791b24ab891292ffeaa56befb747221cf90179bbb8`）只是 `PARSE_SCRIPT.serialize` 手工摘取的字段子集，不是完整解析配置，完整基线要从 Beta 补建（见第 2 节末段），[启动档案](docs/training-doc/v2-1600ep-m32x8x8-modul-b128-80k/launch.md) 与 [完成结果](docs/training-doc/v2-1600ep-m32x8x8-modul-b128-80k/result.md) 提供命令和结果依据。

**本机是环境 B，资源在编写时足够起步，但每档起跑前都要重查。** 8 × `NVIDIA A100-SXM4-80GB`；`/scratch` 是 `/dev/md0`（XFS，本地 NVMe RAID），编写时可用约 1.2 TiB；`/dev/shm` 总量约 561 GiB；八卡显存均为 0 MiB。核对时 HEAD 为 `f711d68fcb23ac2cdd8b776c1f051f9916173e36`，工作区干净。仓库、数据、资产和新产物都在 `/scratch/hongze/robomme_policy_learning_MotionJEPA/`，运行产物统一落 `v1-store/`。

**两个 run 名已查重，确认一次后 4096 → 1024 自动交接。** 拟定 `v2-1600ep-m64x8x8-modul-b128-80k`（4096）与 `v2-1600ep-m16x8x8-modul-b128-80k`（1024），两处训练输出目录编写时均不存在。按 [AGENTS.md](AGENTS.md) 第 6 条，实施前一次确认两个名称和本计划范围；确认后 4096 验收通过即转入 1024，不在交接点再问。2048 的历史同意记录不能代替本轮授权。本轮只交付这份文档，代码、验证和训练都未开始。

### 2. 三档配置对照：只改变记忆预算

**唯一的实质差异是 history YAML 里的 `budget`。** 具名配置仍是 [training/config.py](src/mme_vla_suite/training/config.py) 中 `_CONFIGS` 的 `mme_vla_suite_b128_80k`，本轮不改它，也不改已有的 [32frame 配置](src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-32frame-8x8.yaml)。新增 `perceptual-framesamp-modul-64frame-8x8.yaml` 与 `perceptual-framesamp-modul-16frame-8x8.yaml`，从 32frame 复制后只把 `budget: 2048` 改成 `4096` / `1024`。最大帧数由 `FrameSampDataset.__init__` 按 `budget // (token_per_image * num_views)` 推出（4096//64=64、1024//64=16），选帧、读行、展平都已按这个值参数化，不需要为新档另写分支。启动命令只换 YAML、run 名和输出路径，不加学习率、batch、worker 或步数覆盖。

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

**表外的固定项同样逐项继承。** AdamW `b1=0.9,b2=0.95,eps=1e-8,weight_decay=1e-10`；模型 `pi05=true,paligemma_variant=gemma_2b,action_expert_variant=gemma_300m,memory_expert_variant=gemma_150m`，bf16 计算，`action_dim=32,action_horizon=20,max_token_len=64,use_history=true,discrete_state_input=false`，冻结 `.*img.*`。history 保持 `streaming_obs_horizon=16,pool_type=mean,use_pos_emb=true,use_state_emb=false`，位置输入与隐藏维度均为 768；`streaming_obs_horizon` 不随 64/16 帧档改变。学习率仍是 `CosineDecaySchedule`，peak 与 decay 同为 5e-5，warmup 后恒为 5e-5。

**数据加载与并行方式不变。** [create_data_loader](src/mme_vla_suite/training/dataloader.py) 经 [TorchDataLoader](src/openpi/training/data_loader.py)：`spawn`、`persistent_workers=true`、`prefetch_factor=2`、`pin_memory=false`、`shuffle=true`、`drop_last=true`、`_collate_fn_shm`。单个 JAX 进程用八卡，`make_mesh` 得 `batch=1,fsdp=8`，每卡 128/8=16 个样本；不改成八个 `torchrun` 进程，不加梯度累积。

**现在直接跑会在三处被拦下，所以需要少量代码改动。** 一是 `FrameSampDataset.__init__` 的形制守卫：`legacy_shape` 只认 `(512,16,1)`、`(512,64,1)`，新档分支只认 `raw == (2048,64,1)` 且 modulation、无 motion，`budget: 4096` 会直接抛 `ValueError`。二是启动链写死 2048：[modul_launch_contract.py](scripts/training/modul_launch_contract.py) 的 `YAML="perceptual-framesamp-modul-32frame-8x8.yaml"` 与 `history_expected["budget"]=2048`，[run_modul2048.sh](scripts/training/prod/run_modul2048.sh) 固定 `HC=perceptual-framesamp-modul-32frame-8x8.yaml`、记录目录 `v1-store/bench/m2048/`。三是验收工具写死旧档：`check_modul_speed.py::report` 断言 `history_values["budget"]==2048`，`check_modul_train_records.py::validate` 要求 `--fsdp-devices == 2`、mask 摘要按 `(batch_size,2048)` 计算。在线 [FrameSampMemory](src/mme_vla_suite/policies/framesamp_memory.py) 与 [HistoryPi0](src/mme_vla_suite/models/integration/history_pi0.py) 都读配置里的预算，不在拦截之列。逐文件改法见第二部分第 4 节。

**「一致」靠完整配置记录逐字段比对来证明，现有记录不够用，要先补。** 现有 `modul_launch_contract.py::PARSE_SCRIPT.serialize` 手工摘取 `batch_size`、`lr_schedule`、`optimizer`、`model` 等字段：完整 `data` 配置、实际解析出的数据变换、`resume`/续训 checkpoint 相关字段都没有记，`lr_schedule`、`optimizer` 经 `dataclasses.asdict` 后也丢了类型（看不出是不是 `CosineDecaySchedule`、`AdamW`）。所以 2048 的 `approval.actual.json` 只能证明摘取到的那些键一致。实施时改为带版本号的配置记录格式（拟名 `config_record_version=2`），递归覆盖 `TrainConfig` 全部字段并保留每个 dataclass 的类型全名、解析后的数据变换链及资产文件摘要；再在 2048 Beta `55647ff` 的只读快照上用同一格式补建基线（CPU 解析，不训练），原档案不动。比对只放行按精确字段路径列出的差异（`model.history_config`、`exp_name`、`checkpoint_dir`、`wandb` 与日志/缓存路径等），未知字段、缺字段或清单外差异一律拒绝起跑。两档各自计算 YAML 与完整记录的摘要，绑定自己的起跑记录，不共用 2048 的 config SHA。配置记录只证明「配置没变」；训练语义没变仍靠代码差异审查与第 5 节的同预算训练对拍共同证明。

### 3. 数据、输入链路与两条核心保证

**数据原样复用，不重建、不重抽特征、不下载模型。** 两档都用 `v1-store/datasets/4task-v2-1600ep-604f16da/`：`--dataset-path` 为 `framesamp-8x8`，`MMEVLA_FRAMESAMP_SOURCE` 为 `source`，`MMEVLA_FRAMESAMP_MANIFEST` 为 `meta/episode_manifest.json`。规模为四任务 1600 episode、605611 个执行样本、1192918 帧，packed store 状态 `verified/full`，沿用整个 manifest 的执行样本，不另造训练/验证划分。归一化根 `v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da`（`asset_id=robomme`），初始权重 `v1-store/models/openpi-assets/checkpoints/pi05_base/params`，均已只读确认存在。

**三个关键文件本次重算 SHA256，与 2048 档案逐字一致：**

| 文件 | SHA256 |
|---|---|
| `meta/episode_manifest.json` | `df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482` |
| `framesamp-8x8/meta/store_meta.json` | `f7677e69e5c473ab2962a5ac05a5909348c2f96736152b7217b77f0d2eb4231a` |
| `robomme/norm_stats.json` | `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173` |

这只核了摘要，没有重扫大文件；store 状态、清单关联与资产由起跑 preflight 再查。

**本库每个样本在三档下都是满帧，补零分支是死代码。** 清单 1600 集的 `exec_start_idx` 最小 100、最大 1152（本次 `jq` 实测），执行样本的全域帧号 `step ≥ 100`。`even_sampling_indices(step, max_frames)` 只在 `step < max_frames` 时返回 `step+1` 帧，否则返回满额的 `linspace(0, step, max_frames)`；100 > 64，所以 4096 档每样本恰好 64 帧、1024 档恰好 16 帧，`static_mask` 恒全 True。「2048 满 32 帧不代表 4096 满 64 帧」的顾虑因此不成立；实施时仍按执行索引统计一次满额比例留档（期望 605611/605611）。补零与 mask 屏蔽在真实数据上碰不到，只能用合成案例覆盖：零基帧号 `t` 对应 `min(t+1, max_frames)` 个有效帧，所以 `t=0` 已有 1 帧，**不存在 0 帧样本**；合成案例按 `t=0、1、14、15、16、30、31、32、62、63、64` 覆盖 1 帧到满额及越过 16/32/64 上限的边界。mask 不变性（无效位写入垃圾值不改输出）只对至少含一个有效 token 的样本成立，判据也只在这个范围内声明。不伪造清单、不重建库。

**改前 2048 链路**（箭头上的变换沿用现有实现）：

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

**改后只动两处：选几帧、记忆轴多长。** 数据、算法和交付链全部相同：

```text
同一 source / manifest / framesamp-8x8
  → FrameSampDataset：4096恰选64帧；1024恰选16帧（本库恒满帧）；每帧仍64 token
  → 单样本 image(B,2048) bf16 + pos(B,768) f32
            + state(B,8) f64 + mask(B,) bool，B分别为4096、1024
  → 同一 _collate_fn_shm / JAX交付：batch128、八卡分片、state仍由f64转f32
  → 同一模型：memory(128,B,1024)，只扩缩token轴，不改通道或参数形状
  → 同一20个action query及动作监督；历史内容与query位置按预算产生预期变化
```

**字节账：history 四键的逻辑量随预算线性变化，4096 是 2048 的两倍。** 下表只算 Dataset 归一化后、collate 阶段的四个 history 张量（state 为 f64），不含当前观测、动作、主机队列、参数、梯度或模型中间张量，不能当显存看：

| 预算 | image 字节/样本 | pos 字节/样本 | state 字节/样本 | mask 字节/样本 | 合计字节/样本 | batch128 合计 |
|---|---:|---:|---:|---:|---:|---:|
| 2048 | 8388608 | 6291456 | 131072 | 2048 | 14813184 | 1896087552 |
| 4096 | 16777216 | 12582912 | 262144 | 4096 | 29626368 | 3792175104 |
| 1024 | 4194304 | 3145728 | 65536 | 1024 | 7406592 | 948043776 |

JAX 交付后 state 转 f32，三档每样本四键合计分别为 14747648、29495296、7373824 字节。按 worker16 × prefetch2 只算主机队列里的四键，2048/4096/1024 约 56.51/113.02/28.25 GiB；实际峰值还含其他字段和副本，要实测。每 epoch 图像行逻辑读取量为 4731.3/9462.7/2365.7 GiB（605611 样本 × budget × 2048 × 2 字节）；page cache 与预读会让磁盘实读不同，不能拿它推断步时。

**跨档输出不同是预期的，所以等价只在同一预算内证明。** 同一被选帧的底层特征、观测和标签不改数；改的是选帧集合和动作 query 的 RoPE 位置。`MemoryAttention` 把 20 个动作 query 接在记忆轴之后，位置从 `2048..2067` 变为 `4096..4115` 或 `1024..1043`，因此三档的 loss、梯度和输出都不要求相等。等价验证只在**同一预算**的源 NPY 参考链与 packed 链之间做。

#### 保证一：两档继承 2048 的训练配置，额外差异在起跑前被拒绝

**配置文件层：新 YAML 只能比 32frame 多改 `budget` 一个键。** 两份新 YAML 解析后与 32frame 逐键比较，只放行 `budget=4096` / `budget=1024`，拟新增判定行如 `BUDGET_CONFIG=PASS budget=4096 baseline=2048 diff_keys=budget`。`memory_token_dim=1024`、`img.input_dim=2048`、`streaming_obs_horizon=16` 任一被顺带改动即失败，避免通道宽度或观测节奏的变化混进来。

**实际命令层：检查的是即将交给训练的同一份 `TRAIN_ARGS`。** `modul_launch_contract.py` 的 `parse_config`、`validate_config` 由写死 2048 扩为 1024/2048/4096 三档映射，[preflight_train_launch.py](scripts/training/preflight_train_launch.py) 对即将执行的 CLI 做 CPU 解析。两档都必须输出既有格式 `LAUNCH_CONFIG=PASS mode=prod steps=80000 batch=128 workers=16 fsdp=8`，解析记录还要证明 seed 42、warmup 5000、lr 5e-5、EMA 0.999、`pi05_base` 初始化及第 2 节其余固定值一致。YAML 对了但 CLI 偷偷覆盖 batch、学习率或步数，同样起不了跑。比对对象是第 2 节末段的带版本完整配置记录，不是现有 `serialize` 的字段子集。

**测速报告层：报告的 SHA 对，不代表测的是本档预算。** 现有 `validate_approval` 只核报告文件摘要、`status=="READY"`、环境和 HEAD，不读报告里记录的配置。同一提交里三档配置并存时，拿一份合法的 1024 测速报告去生成 4096 的批准记录、SHA 全部填对，现有检查会放行。实施时把测速报告内记录的完整配置记录与正式配置逐字段比对：只放行 `num_train_steps` 1000→80000、W&B 开关以及按精确字段路径列出的身份字段（run 名、输出/日志/缓存路径）；`budget`、history YAML 摘要、模型、优化器、数据与初始化资产必须完全一致。负例必须是「另一预算的合法报告、所有 SHA 都正确」，不能只测损坏的摘要。

**数据与版本层：绑定本次实际输入，不借 2048 的记录。** 同一 preflight 重查上面三个文件摘要、`verified/full` 规模、全新输出根和 clean `TRAIN_HEAD`，把本档 YAML、runner 和测速报告的摘要写进本档记录。2048 的旧批准 JSON 与旧测速报告不能挪用。

#### 保证二：4096 完整成功并释放八卡后，1024 才能启动

**完成与否看真实训练产物，不看最后一条日志。** 拟新增 `scripts/training/tests/check_modul_completion.py` 同时要求：唯一终态 `EXIT_CODE=0`；checkpoint 目录恰为 `5000,10000,…,75000,79999` 共 16 份；log100 共 800 条记录、五项标量全部有限；最终 79999 经真实加载与本档配置/来源检查。[train.py](scripts/training/train.py) 的 `main` 结束前已调用 `checkpoint_manager.wait_until_finished()`，调度器等整个训练入口返回后才验收。通过时输出 `RUN_COMPLETED=PASS run=v2-1600ep-m64x8x8-modul-b128-80k budget=4096 steps=80000 checkpoints=16 final=79999`，并把最终文件摘要写进完成 JSON。

**「可加载、形状对」证明不了权重来自本次训练，所以保存现场要留一份摘要。** [checkpoints.py](src/openpi/training/checkpoints.py) 的 `save_state` 只写 `params` 和 `assets`（`train_state` 一项被注释掉），不记更新次数，也不记参数摘要；三档参数形状完全相同。反例：保留新 4096 run 的配置、日志和来源记录，把 `79999/params` 换成 2048 的权重，加载与结构检查照样通过，事后重算 SHA 只能证明文件当前内容自洽。修正：`train.py` 新增一个只在设置 `TRAIN_FINAL_RECORD_DIR`（拟名）时启用的保存现场记录。在 `step == config.num_train_steps - 1` 的 `save_state` 之后，它写入本次运行标识（启动时生成的 UUID 与 W&B run id）、run 名、HEAD、budget、`int(train_state.step)`（期望 80000）以及按 checkpoint 存储 dtype 逐叶计算的 EMA 参数摘要。完成器从 checkpoint 原样读回数组后逐叶比对这份摘要，不一致即 `RUN_COMPLETED=FAIL`。不设该变量时 `train.py` 行为零差异；设了也只多写一个文件，不改 `info`、不改 W&B、不改训练计算。

**最后 99 步出 NaN 也要拦住。** 常规日志只在 `step % log_interval == 0` 时输出，最后一条是 79900；79901–79999 这 99 步的 `info` 留在 `infos` 列表里、不再输出。现有 `check_config_provenance.py::gate_ckpt_param_tree` 只查参数树结构与 dtype，不查数值是否有限。所以「800 条指标有限 + 最终权重可加载」不足以判成功。修正：同一保存现场记录把 `infos` 里剩下的尾窗逐步五项标量及其均值、实际更新次数一并写进文件（只写文件、不进 W&B，原有 800 条 log100 保持不变）；完成器另外检查最终权重全部叶都是有限值，并用固定输入、固定 noise、`num_steps=10` 跑一次动作输出，要求结果有限。尾窗出现 NaN、checkpoint 可加载但含 NaN/Inf、摘要不符、更新次数不是 80000，任一情况都判 `RUN_COMPLETED=FAIL`，并使 `SECOND_TRAIN_CALLS=0`。

**交接由独立检查放行，任一不满足则 1024 训练入口调用 0 次。** 拟新增 `scripts/training/prod/run_modul4096_then1024.sh` 先核完成 JSON 绑定的是本轮 4096 的 run/HEAD/budget，再核训练与采样 PID 已退出、八卡显存均为 0 MiB、`/scratch` 空闲 ≥ 400 GB、代码与环境未变，然后才进入 1024 的验证、测速和正式训练。非零退出、被杀无退出码、缺最终 checkpoint、加载失败、只有 79900 日志、伪造的旧完成 JSON、只替换 `79999/params` 而保留新 run 其余记录、尾窗 NaN、可加载但含 NaN 的 checkpoint 等负例都必须得到 `SECOND_TRAIN_CALLS=0`。

**时间与容量按各档本机实测，不按 token 比例外推。** 2048 的 22 小时 58 分 52 秒与 16 份 checkpoint 合计 190052280479 字节只是基线：预算不改参数形状，两新 run 权重预计各约 190 GB、合计约 380 GB；但输入逻辑字节翻倍不代表整模型步时翻倍，也不证明 4096 放得进显存（2048 的显存读数含 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` 预分配）。每档各跑八卡 20 步容量检查和 1000 步测速，稳态窗口固定 100–899 步，报告 GPU 均值、0% 采样占比和慢步分层。原 b128/w16/FSDP8 容量不过时队列停下报告，不自动降配。以上两条保证都是待实施、待验证的要求，不是本轮已跑出的结果。

## 第二部分（技术细节，供 agent 追踪）

### 4. 待实施的最小改动

当前代码不能直接启动两档。[FrameSampDataset.__init__](src/mme_vla_suite/training/framesamp_dataset.py) 的新档白名单只接受 `(2048,64,1)`、modulation、无 motion；[run_modul2048.sh](scripts/training/prod/run_modul2048.sh) 和 [modul_launch_contract.py](scripts/training/modul_launch_contract.py) 还固定了 2048 YAML 和旧 run 名。计划按下表扩展，所有新增接口在本轮都只是设计。

| 文件/稳定锚点 | 拟改动与边界 |
|---|---|
| `src/mme_vla_suite/models/config/robomme/` | 新增 `perceptual-framesamp-modul-64frame-8x8.yaml`、`perceptual-framesamp-modul-16frame-8x8.yaml`；与32frame解析值只差budget。 |
| `training/framesamp_dataset.py::FrameSampDataset.__init__` | 精确加入 `(4096,64,1)`、`(1024,64,1)`，只允许modulation且无motion；旧512/2048接受集合与拒绝条件保持。 |
| `scripts/training/tests/ref_npy_dataset.py::RefNpyFrameSampDataset.__init__` | 独立参考链加入相同两档；参考采样与拼装不调用生产Dataset以免同源错误互相通过。 |
| `scripts/training/train.py::main` | 新增仅在设置 `TRAIN_FINAL_RECORD_DIR`（拟名）时启用的最终保存现场记录：末步 `save_state` 之后写运行 UUID/W&B run id、run、HEAD、budget、`int(train_state.step)`、按存储 dtype 的 EMA 逐叶摘要，以及 `infos` 中尾窗（79901–79999）逐步五标量与均值。不设变量时零行为差异；不改 `info` 字典、W&B 日志、训练计算与保存内容。按 AGENTS.md 第 18 条，此改动不影响训练输入与语义，由 2048 改前/改后 20+1 步回归（设与不设该变量两种情形）证明数值不变。 |
| `scripts/training/g0/bench_train_steps.py` | 扩展history文件白名单；复用现有refnpy/packed、索引、batch、初始化和参数记录能力；新增把实际解析配置（第 2 节带版本完整配置记录格式）写入 run 记录，供验证器与真实 loader、mesh、runtime 互校。 |
| `scripts/training/tests/check_32frame_modul.py` | 将写死32/2048的检查参数化为显式预算，保留2048默认；新增有界真实样本入口，覆盖选帧、collate、padding、在线缓存、初始化、RoPE及保存加载。 |
| `scripts/training/tests/check_modul_train_records.py::validate` | 现有实现用 `flag(meta["argv"], ...)` 强制 `--batch-size`/`--num-workers`/`--seed`/`--fsdp-devices` 显式且唯一出现在 CLI，并写死 `workers == 4`、`fsdp == 2`；第 5 节规定 b128/w16/FSDP8/seed42 从具名配置继承、CLI 不写，只扩常量仍会报「参数必须明确且唯一」。改为读 `bench_train_steps.py` 记录的实际解析配置，与真实 loader（batch、workers）、mesh（`fsdp=8`）、runtime 设备列表互校；新增「默认继承正确值通过、CLI 显式覆盖成错误值拒绝」两类用例。移除仅支持两卡/w4/fsdp2的验证器限制，新增明确的预算、worker、FSDP期望参数；校验八个不重复设备；mask摘要由写死的`(batch_size,2048)`改为`(batch_size,budget)`，本库恒满帧，全True判据对三档仍成立，同时记录逐样本有效数。旧两卡默认仍可回归。 |
| `scripts/training/modul_launch_contract.py`、`preflight_train_launch.py` | 显式映射1024/2048/4096与16/32/64frame配置；正式run及报告按本次记录绑定。每个新档都强制进入launch-mode检查，防止换文件名绕过闸门；保留完整配置、数据、初始化、clean HEAD、资源及授权校验。`PARSE_SCRIPT.serialize` 升级为带版本完整配置记录（递归全部字段、dataclass 类型全名、解析后数据变换、资产摘要），差异放行表写成精确字段路径，未知字段/缺字段/清单外差异拒绝；在 2048 Beta 只读快照上用同一格式补建基线，原 `approval.actual.json` 不动。`validate_approval` 新增测速报告内配置与正式配置的语义比对：只放行 1000→80000、W&B 与列明身份字段，预算、YAML 摘要、模型、优化器、数据、初始化资产须一致。 |
| `scripts/training/tests/check_modul_speed.py::gpu_rows`、`report` | 将budget=2048断言和图像逻辑读取带宽公式改为从已校验预算推导；保持1000步、800步稳态与资源/报告完整性判据。`gpu_rows` 现用 `float()` 直接解析，会接受 `nan`/`inf`，`ready` 也不查利用率是否有限，报告可同时出现 READY 与 NaN；且重复时间戳会虚增采样密度。改为校验 GPU 编号属于 0–7、值有限、`0≤util≤100`、显存非负、每卡时间戳严格递增且唯一，覆盖率按唯一时间点计算，报告 JSON 以 `allow_nan=False` 写出。不同时间点出现相同利用率属正常，保留。补 NaN、Inf、重复记录、稀疏采样四类负例。 |
| 拟新增 `scripts/training/prod/run_modul_budget.sh` | 接受显式预算和本档run/HEAD/YAML SHA/报告，复用2048的环境及训练CLI纪律；保留原2048入口兼容，不让它静默启动新档。 |
| 拟新增 `scripts/training/prod/run_modul4096_then1024.sh`、`scripts/training/tests/check_modul_completion.py` | 严格顺序调度、失败停止，验收4096的真实产物后才进入1024；完成记录绑定run、预算、HEAD及checkpoint摘要。完成器读取 `TRAIN_FINAL_RECORD_DIR` 的保存现场记录：checkpoint 原样读回数组逐叶比对 EMA 摘要，核 `state.step==80000`、尾窗 99 步五标量全部有限，再检查最终权重全叶有限，并用固定输入、固定 noise、`num_steps=10` 的动作输出须有限。 |
| `test_pack_guards.py`、`test_modul_launch_guards.py`、`test_modul_speed.py`及对应新增用例 | 覆盖错误预算/文件名/配置/旧报告/旧run/失败交接，断言训练入口调用次数为0。新增：另一预算的合法测速报告（全部 SHA 正确）、完整配置记录出现未知字段或清单外差异、只替换 `params` 而保留新 run 其余记录、尾窗 NaN、可加载但含 NaN 的 checkpoint、GPU 采样 NaN/Inf/重复/稀疏。 |

在线 [FrameSampMemory](src/mme_vla_suite/policies/framesamp_memory.py) 的 `get_frame_sampling_indices`、`_prepare_frame_sampling` 已按预算计算，不存在需要扩展的2048专属白名单；[HistoryPi0](src/mme_vla_suite/models/integration/history_pi0.py) 也读取配置预算。先增加两档回归，除非验证暴露缺陷，不修改这些实现。冻结的旧建库 memory buffer 不属于本轮生产改动。

### 5. 验证顺序与通过标准

所有输入验证先于相应训练。共同 CPU 检查可以提前覆盖两档；GPU任务严格按“基线回归 → 4096验证、测速、正式80k和验收 → 1024验证、测速、正式80k和验收”顺序执行，不在4096正式训练期间抢卡验证1024。每个真实训练验证也使用八卡、b128/w16/FSDP8；短验证只通过自己的启动覆盖缩短步数、增加取证和关闭W&B，不修改正式默认。

| 阶段 | 验证内容 | 通过标准 |
|---|---|---|
| CPU配置及拒绝测试 | 三档完整解析对照，旧512/2048守卫回归，错误融合方式、motion、浮点budget、错误run/数据/摘要、失败交接负例 | 两新YAML只差budget，输出第3节保证一的`BUDGET_CONFIG=PASS`；错误输入全部拒绝且训练调用0次。 |
| 完整配置记录与反例 | 2048 Beta 快照补建带版本基线；三档完整记录逐字段比对；另一预算合法测速报告（SHA全对）冒充本档；未知字段/缺字段；验证器「默认继承通过、显式错值拒绝」；GPU采样NaN/Inf/重复/稀疏 | 三档记录只在精确列出的字段路径上不同；冒充报告、未知字段、显式错值、无效采样全部拒绝，训练调用0次；报告 JSON 不含 NaN。 |
| 完成器反例（替身产物） | 只替换 `79999/params` 保留其余记录、尾窗 NaN、可加载但含 NaN 的 checkpoint、`state.step≠80000` | 全部 `RUN_COMPLETED=FAIL`、`SECOND_TRAIN_CALLS=0`。正向完成验收另读真实训练产物（见容量 20 步与正式 run）。 |
| 真实输入对拍 | 固定seed42；四任务每个episode至少取首/中/末执行样本并去重；加入episode接缝与尾部样本及前20个实际batch（本库恒满帧，16/32/64帧短历史边界只在合成案例中覆盖）；保存精确索引清单 | 同预算refnpy与packed的索引顺序、选帧索引、逐键shape/dtype/原始字节完全相同，mismatches=0。合成短历史与在线链另测。 |
| 模型语义检查 | 初始化参数树；mask垃圾值不影响输出；各预算独立RoPE oracle；短历史padding和collate交付 | 相同seed初态61叶逐位相同；mask无效位扰动不改输出；oracle `atol=1e-6,rtol=1e-5`。跨预算非退化前向允许不同。 |
| 2048旧档训练回归 | 在改前/改后代码、同一八卡环境各跑20步，另各补1步保存第一次更新现场；改后侧再分设与不设 `TRAIN_FINAL_RECORD_DIR` 两种 | 同一输入、相同初态；loss/梯度及params/optimizer/EMA摘要逐步一致，证明扩白名单与保存现场记录都未改旧档数值。 |
| 新档真实训练对拍 | 每档refnpy与packed各20步，各补1步；固定数据、seed、设备列表和运行环境 | 输入和初态逐位一致；五项标量及完整状态摘要一致，覆盖首次更新与第20次更新；发现差异先定位，不能按新预算放宽。 |
| 保存与加载 | 每档packed独立100步，真实保存末步，再经`check_config_provenance.py`及参数化加载检查真实读取 | 配置/数据/61叶参数/EMA内容与本run现场相符；可执行动作路径。旧run或初始化权重冒充时必须失败，不能只比较参数树形状。 |
| 八卡生产容量 | 每档独立20步，原生产模型、b128/w16/FSDP8及一次真实保存；开启 `TRAIN_FINAL_RECORD_DIR` | 无OOM、NaN/Inf、worker故障、共享内存错误；峰值记录齐全；不因容量失败自动降低batch、worker或预算。完成器按 20 步口径（`state.step==20`）读真实产物做正向验收，保存现场摘要与读回数组逐叶一致。 |
| 八卡独立测速 | 每档独立1000步，0–99预热、100–899稳态、900–999收尾，记录真实保存 | 800步完整、`SPEED_REPORT=READY`；GPU 采样全部有限、每卡时间戳唯一递增，覆盖率按唯一时间点计；共享内存峰值占比≤0.70；报告内记录本档完整配置记录；报告吞吐、GPU/主机数据和本档80k预计时长。 |

正确性对拍使用既有 `XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`，容量、测速及正式阶段则显式unset，不能混用速度。20步主记录核对初始化 `state_step=0` 及更新后 `2..20`，独立1步补跑补齐 `state_step=1`；比较五标量hex与完整TrainState 201叶（参数61、可训练参数38、EMA61、Adam一二阶各38及计数3，按既有树结构核对），不是只比loss曲线。参数61叶与可训练38叶存在包含关系，201叶计数为61+61+38+38+3。

源NPY侧复用 `BENCH_DATASET_IMPL=refnpy`、`BENCH_REF_SOURCE`、`BENCH_REF_MANIFEST`，其 `--dataset-path` 指向source；packed侧指向framesamp-8x8。两侧都通过 `bench_train_steps.py` 明确覆盖 `--num-train-steps 20 --log-interval 1 --save-interval 1 --no-wandb-enabled`，b128/w16/FSDP8/seed42从同一具名配置取得、CLI 不写，由验证器按记录的实际解析配置与真实 loader/mesh/runtime 再核实。新增预算及八卡验证器接口完成前，不能直接使用旧验证命令宣称通过。

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

每档正式起跑由preflight重新解析与训练**同一份** `TRAIN_ARGS`，同时核对预算、YAML SHA、带版本完整配置记录（与 2048 Beta 补建基线逐字段比对，只放行精确列出的字段路径）、初始化权重、manifest/norm/store摘要、自己的run和新输出根、runner/报告摘要、clean HEAD、八卡空闲及空间。报告摘要之外还要读报告内的配置记录与正式配置做语义比对，只放行 1000→80000、W&B 与列明身份字段；预算、模型、优化器、数据、初始化资产不同即拒绝，防止拿另一档的合法报告冒充。正式入口设置 `TRAIN_FINAL_RECORD_DIR` 指向本 run 的 `records/` 忽略目录。测速HEAD至正式HEAD只可有文档变化，源码/启动器/依赖一旦改变则重新验证测速。新档的同意记录应忠实引用本轮实施决定及其范围，分别绑定本档的实际报告和入口，不伪造新的用户原话。

### 7. 4096完成后才转入1024的硬条件

顺序调度器按以下状态运行，所有状态和完成记录写入本轮唯一的 `v1-store/bench/modul-budget-sweep/<批次名>/`；正式权重分别进入 `v1-store/train-runs/mme_vla_suite_b128_80k/<run_name>/`。

```text
2048 Beta快照补建完整配置基线 → 共同CPU检查与反例 → 2048旧档回归
  → 4096输入/模型/20步对拍/100步保存加载
  → 4096八卡容量20步与测速1000步
  → 4096最终起跑检查 → 正式80000步 → 完成验收
  → 八卡释放、重查HEAD/环境/数据/磁盘
  → 1024输入/模型/20步对拍/100步保存加载
  → 1024八卡容量20步与测速1000步
  → 1024最终起跑检查 → 正式80000步 → 完成验收
```

4096交接必须同时满足：训练进程正常退出且唯一终态为 `EXIT_CODE=0`；`train.py::main` 的 `checkpoint_manager.wait_until_finished()` 已返回；保存目录集合为 `5000,10000,…,75000,79999` 共16份且元数据/数据完整；正常log100共800条记录、五项标量全部有限；最终79999经过真实加载和本档配置/来源检查。最后一条常规日志是79900，不能将它当成79999单步loss。预算不改变61叶参数形状，因而“目录存在、61叶形状正确”不足以证明拿到了本run的权重，完成检查还须绑定run来源、文件摘要和保存记录。具体为：`TRAIN_FINAL_RECORD_DIR` 保存现场记录存在且运行 UUID、run、HEAD、budget 与本档一致；`state.step==80000`；checkpoint 原样读回的数组逐叶等于现场 EMA 摘要；尾窗 79901–79999 共 99 步五标量全部有限；最终权重全叶有限；固定输入、固定 noise、`num_steps=10` 的动作输出有限。

通过上述检查后，调度器核对训练PID和本轮GPU采样PID均退出、八卡释放，并生成绑定4096 run/HEAD/budget/最终checkpoint摘要的完成JSON及第3节保证二的 `RUN_COMPLETED=PASS` 判定；1024起跑门重新验证该JSON及其产物。只有4096完成记录有效且1024自身检查全过，才能调用1024训练入口。任一失败均停止队列并保留现场；不跳过4096、不自动续训、不自动改超参、不用tmux会话消失判断成功。当前checkpoint主要保存params/assets，不完整保存优化器状态，不能把载入权重称为精确断点续训。

调度器的负例至少覆盖：4096非零退出、被终止而无退出码、最终checkpoint缺失或加载失败、仅有79900日志、旧run伪造完成JSON、只替换 `79999/params` 而保留新 run 其余记录、尾窗 NaN、可加载但含 NaN 的checkpoint、`state.step≠80000`、预算/HEAD不匹配、1024配置变化、GPU未释放及磁盘不足。上述任一情况都应得到 `SECOND_TRAIN_CALLS=0`。零训练负例使用替身入口核对控制流；正向完成验收必须读真实训练产物。

预计长任务的tmux全名采用 `modul-sweep-<UTC>`、`m4096-<阶段>-<UTC>`、`m1024-<阶段>-<UTC>`，启动后归档展开后的完整名称和精确PID；存活检查用 `tmux has-session -t '=完整会话名'`。日志监听逐级行缓冲，只关注启动、判定、异常和完成事件。清理只针对本轮清单里的精确会话，不能全局结束tmux或清理其他代理现场。

### 8. 提交、留档与最终交付

实施时先固定BASE及环境指纹，完成新增配置、最小守卫和验证工具；代码验证通过后按当时 `git log` 顺延功能版本。两档代码和完整顺序入口提前提交，正式起跑前固定Beta和clean TRAIN_HEAD；可以由同一个Beta锁定这轮顺序实验，两档分别记录自己的实际启动时间、命令、预算和摘要，结束结论与该Beta配对，不amend或squash掉锚点。

按照目标仓库现有格式，为两个正式run分别建立 `docs/training-doc/<run_name>/launch.md`、`result.md`、`records/`，更新 [训练档案索引](docs/training-doc/README.md)。起跑信息在实际启动时记录，1024不能提前填为“已启动”；运行期间需要提交文档时使用独立开发副本及独立环境，主训练副本源码、依赖与HEAD保持稳定。完成检查和交接运行记录先落本轮 `v1-store/`，让两档顺序执行不因写入未提交文档而破坏clean HEAD。

每档归档本轮用户原话、代码完整hash、真实CLI与全部环境覆盖、预算/配置/数据摘要、硬件和存储、验证/容量/测速结果、初始化来源、500ms利用率数据、正式清洗日志和800条指标、16份checkpoint清单、最终真实加载结果、退出码及计划外事件。Git可还原的脚本/YAML只引用提交和路径，不复制到档案；权重留在 `v1-store/train-runs/`。完整保留2048及其他run，不自动导出到HuggingFace或开展策略rollout。

最终完成标准为两个新run均80000步正常退出、各16份checkpoint通过、各最终权重真实加载通过、配置差异仅为本计划所列预算及身份字段、4096结束早于1024启动且全程资源口径均为八卡、档案齐全。训练loss和利用率用于检查训练过程，不作为策略成功率结论；后续rollout范围另定。

### 9. 本次计划文档的验证与提交

本次只新增 `0922-4096-1024-8gpu-training-plan.md`。提交前核对所引现有文件、配置键、数据摘要、命令中拟新增接口标记、逻辑字节计算和顺序条件，执行 `git diff --check` 及暂存差异检查；纯文档阶段不启动仓库训练/验证脚本、不同步依赖。只用明确文件路径暂存，中文 `docs:` 提交记录用户原话、核对依据和验证结果；按目标仓库既有规则推送当前分支的既有upstream，不改远端或凭据。实施与正式训练仍待本轮计划后的明确指令。
