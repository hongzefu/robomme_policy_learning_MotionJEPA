# context 五组训练与通路验证计划

> 创建日期：2026-09-28，America/New_York。权威工作副本：`/scratch/hongze/robomme_policy_learning_MotionJEPA`，环境 B；代码核对锚点：`8aee9ced0bbc7bc4d2a863539fa8c14873bb9832`，分支 `v2-motionmem`，开始时工作区干净。本文是本轮训练方案，不是启动回执。本轮授权范围为核对、最小验证和根目录计划；尚未实施生产代码、配置与启动器改动，尚未开始正式训练。后续实施和正式起跑分别按用户授权推进，已有明确决定不重复询问。
>
> 提交体例沿用 `docs:` 与 `commitV<大版本>.<小版本>Beta/正式版`；本计划用 `docs:` 提交，后续 Beta 编号从实施时最新历史接续，不预占编号。外部依赖以本提交的 `uv.lock`、`scripts/assets/ASSETS_LOCK.json` 为准；benchmark gitlink 为 `856bc3a189d4172f3f47dbee4424d585f8d78db3`。不升级依赖、不下载或重建数据、不启动 Slurm、不访问 turbo。

用户决定按时间顺序原文保留：

1. 「根目录给出方案 我已经训练过了modulation的512 2048 4096 8192 和 512+motion 我要再训练context下的这几个 训练 data 等都要保持一致 先验证是否通路 并且有效 然后写计划到根目录 优先训练512+motion 8卡 其他的在512motion跑完后 跑 4卡一个 2个并行」
2. 「b」——环境判定由用户确认采用 B；随后沙箱外实查确认 8 张 A100-SXM4-80GB。
3. 「实际要的是 1024」——中途澄清，已被下面两条最终决定替代。
4. 「context要跑8192。」
5. 「1024不要了」
6. 「尽可能多做验证 验证完了再写计划」
7. 「你可以自由使用8gpu全都可以用 gogogo」
8. 「测试全部放行」
9. 「不要再问我你现在是full permission模式」
10. 「做你想做的实验」

按后续指令，早期未提交草稿移入诊断目录；所有模型、数据和容量实验结束后，才生成本根目录正式方案。测试授权已覆盖本轮实验，没有据此启动80k正式训练。

**实测结论先行：优先组512+motion的八卡、batch128两步真实训练通过；512、2048的四卡、batch128在逐位验证过的mask候选下通过。4096、8192在四卡、batch128下真实执行显存不足，不能直接排正式训练。** 五组数据与超参的保持方法、优先组独立推进方式，以及大预算的阻断与候选修法见下文。通过的是通路、参数更新及所注明的短测，不是训练收敛或任务成功率。

## 第一部分：方案与可核对的依据

### 1. 最终五组与执行顺序

**最终训练 context 512、2048、4096、8192、512+motion；不安排 context 1024。** 先独占八卡训练 `512+motion160`，正常完成且最终 checkpoint 通过验收后，释放八卡，进入四卡两组并行。根据实际容量结果，第一波安排 `512 + 2048`；第二波为 `4096 + 8192`，必须先完成省显存实现与目标容量复验。每波使用 GPU `0,1,2,3` 和 `4,5,6,7` 两个不重叠集合；上一波两组均验收成功且下一波门槛齐全才交接。这个次序优先推进已证明计算容量可行的档位，不改变任何组的全局batch、数据或训练步数。

**“保持一致”按已训练的对应 modulation 组逐项继承。** 本地 512 基线确为 60,000 步、workers8；2048、4096 与 512+motion 为 80,000 步、workers16，不能写成原来五组均为 80k。没有找到 modulation 8192 的正式训练记录；用户最终仍要求 context 8192，因此将它明确作为新增预算，按 4096 的 80k 口径扩展到 128 帧，不声称有已核实的同预算 modulation 对照。

| 顺序 | context 组 | 最大帧数 / motion token | 具名训练配置 | 步数 / workers | GPU / FSDP | 拟用 run_name |
|---|---|---|---|---|---|---|
| 优先组 | 512+motion | 8 / 160，总记忆 672 | `mme_vla_suite_b128_80k` | 80000 / 16 | `0–7` / 8 | `v2-1600ep-m8x8-context-motion-b128-80k` |
| 第一波甲 | 512 | 8 / 0 | `mme_vla_suite_b128_60k` | 60000 / 8 | `0–3` / 4 | `v2-1600ep-m8x8-context-b128-60k` |
| 第一波乙 | 2048 | 32 / 0 | `mme_vla_suite_b128_80k` | 80000 / 16 | `4–7` / 4 | `v2-1600ep-m32x8x8-context-b128-80k` |
| 第二波甲 | 4096 | 64 / 0 | `mme_vla_suite_b128_80k` | 80000 / 16 | `0–3` / 4 | `v2-1600ep-m64x8x8-context-b128-80k` |
| 第二波乙 | 8192 | 128 / 0 | `mme_vla_suite_b128_80k` | 80000 / 16 | `4–7` / 4 | `v2-1600ep-m128x8x8-context-b128-80k` |

这些是拟用名称，正式起跑前查重并随启动授权确认；不得覆盖已有 run。每组均从同一 `pi05_base/params` 独立初始化，不从 modulation checkpoint 或前一组 context 权重继续训练。四卡时仍是一个 JAX 进程、全局 batch128，每卡从八卡时的16样本变为32样本；不改为 batch64，不线性缩放学习率，不引入梯度累积。`sharding.py::make_mesh` 要求可见设备数能被 `fsdp_devices` 整除，80k 四卡组必须在新启动器显式传 `--fsdp-devices 4`。

新契约还须核对实际 `jax.device_count() == fsdp_devices`、物理GPU列表无重复且长度匹配；仅检查整除不够，否则8张可见卡配FSDP4会形成另一种mesh，不能算四卡运行。

**优先组不必等待4096/8192的显存问题解决。** 它可以在正确的新YAML、正式启动契约、真实dataloader验收、保存验收及本组空间门槛通过后先行启动。大预算优化是独立阶段；优先组长跑期间主副本按仓库规则冻结，开发只能在独立开发副本进行，不能热改在跑的源码或环境。后续四卡任务仍须在优先组完成之后开始。

### 2. 已训练基线与固定训练口径

| 基线组 | 已训练 run / 权威档案 | 起跑提交 | 完成依据 |
|---|---|---|---|
| 512 | [v2-1600ep-m8x8-modul-b128-60k](docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/result.md) | `dd07f18fc385b01eb52db7563fe5f202997b9706` | 60k、12份 checkpoint、最终59999、退出0；历史真实恢复61叶 |
| 2048 | [v2-1600ep-m32x8x8-modul-b128-80k](docs/training-doc/v2-1600ep-m32x8x8-modul-b128-80k/result.md) | `55647ff33c8ddb9ec324fdbcee8bd1491456725b` | 80k、16份 checkpoint、最终79999、退出0；历史真实恢复61叶 |
| 4096 | [v2-1600ep-m64x8x8-modul-b128-80k](docs/training-doc/v2-1600ep-m64x8x8-modul-b128-80k/result.md) | `0571fea5f626e822ca2b23b9e5f90c5bf31dd5eb` | 80k、16份 checkpoint、`completed.json::status=PASS` |
| 512+motion | [v2-1600ep-m8x8-modul-motion-b128-80k](docs/training-doc/v2-1600ep-m8x8-modul-motion-b128-80k/result.md) | `2f10473161b760f16d9240d3c2959ff326cde66b` | 80k、16份 checkpoint、最终79999、退出0；历史真实恢复65叶 |
| 8192 | 当前本地未找到同预算 modulation 档案 | 无 | 本计划按4096扩展；同预算历史对照未验证 |

上述完成结果来自既有归档，本轮只核实档案及本地产物在位，未重新读取所有大权重。额外发现的 modulation1024 历史 run 不属于本轮训练矩阵。

**除机制、预算和已指定的卡数外，训练条件保持如下。** 权威入口为 [training/config.py](src/mme_vla_suite/training/config.py) 的 `_CONFIGS`；60k 与80k分别继承自己的具名配置，后续不修改全局默认超参。

| 项目 | 固定值 |
|---|---|
| global batch / seed | 128 / 42 |
| 学习率 | `CosineDecaySchedule`；warmup5000；peak=`5e-5`；decay=`5e-5`；`decay_steps`按对应60000/80000 |
| 优化器 | AdamW；`b1=0.9,b2=0.95,eps=1e-8,weight_decay=1e-10`；clip1.0 |
| EMA / 日志 / 保存 / 保留 | 0.999 / 100步 / 5000步 / 5000步 |
| 模型 | `pi05=true`、`gemma_2b`主干、`gemma_300m`动作专家、bf16、`memory_expert_variant=gemma_150m` |
| 动作 / 文本 | `action_dim=32,action_horizon=20,max_token_len=64,discrete_state_input=false` |
| 冻结 | `get_freeze_filter` 的图像模块 `.*img.*`；帧与motion编码器保持可训练 |
| history | `perceptual/frame_sampling`、每帧64 token、单历史视角、`streaming_obs_horizon=16,pool_type=mean,use_pos_emb=true,use_state_emb=false` |
| dataloader | 同一 shuffle/seed/drop_last；spawn；persistent_workers；prefetch2；共享内存collate；不改变数据划分 |

warmup后因 peak=decay，学习率恒为 `5e-5`，但60k与80k的曝光量仍分别为768万与1024万样本。跨卡数归约顺序可以变化，不能把同一训练口径写成四卡与八卡逐位复现。context与modulation本来就改变模型计算，不要求两种机制的 loss 或梯度相等。

因此本方案首先保证每个context组与其对应modulation组的训练条件一致。若把context512和context2048直接作预算消融，60k/80k的曝光量也是变量，不能将差异全归因于token预算。8192没有同预算历史基线，只能按新档位报告。另，512起跑提交 `dd07f18` 的subject实际为 `docs: 四卡 b128 modulation smoke 与参数树验收通过`，不能把它追称为Beta。

### 3. 同一数据与motion身份

**复用现成1600集库，不重新抽特征或重建数据。** 数据根 `v1-store/datasets/4task-v2-1600ep-604f16da`；训练读 `framesamp-8x8`，源数据指向 `source`，清单为 `meta/episode_manifest.json`。真实任务是 **BinFill、RouteStick、VideoRepick、VideoUnmaskSwap各400集**，合计605611个执行样本、1192918帧，不使用旧项目scope里的四任务名称推断本次数据。各任务样本数为256929、107450、158025、83207；沿用完整清单及原 sampler，不引入新的留出划分。

本轮重算的小文件指纹如下；大二进制的身份由既有资产锁、store关联检查和后续起跑核验保证，下面不是整库重新哈希证明。

| 对象 | 相对位置 | SHA256 |
|---|---|---|
| 数据清单 | 数据根下 `meta/episode_manifest.json` | `df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482` |
| 帧库元数据 | 数据根下 `framesamp-8x8/meta/store_meta.json` | `f7677e69e5c473ab2962a5ac05a5909348c2f96736152b7217b77f0d2eb4231a` |
| 归一化 | `v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json` | `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173` |
| motion元数据 | 数据根下 `motion/meta/store_meta.json` | `d3a518011c80a115f7b398458a510a6f128f47e76c03fc251e59dab4c9c56268` |

**512+motion必须保留已训练的160-token motion语义。** `motion`库71316行，demo35913、exec35403，float32/768维，layout=`motion-768-grid16-demopad17-v1`；窗口33帧、stride16、位置256维，demo至少17真帧后 `repeat_last`，exec至少33真帧。来源为 `wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt#encoder`，编码器权重SHA为 `0c1986297ccc0ab1913910f33a09ec74ba4c208844d0f5d72dd7ba59e0d9e3ca`。该权重SHA取自既有来源记录，本轮不重新下载或重抽。

⚠ 现有 `perceptual-framesamp-context-8frame-8x8-motion.yaml` 不是本次的正确起点：它仍是motion96、40集库、旧 `wan-v8-filter10-72ep-a`，缺demo17补尾规则。正确做法是从已训 `perceptual-framesamp-modul-8frame-8x8-motion.yaml` 派生**新具名配置**，保留完整motion节，仅改 `integration_type: context` 和 `memory_token_dim: 2048`。旧context配置保留以便还原历史。

### 4. 通路怎样变化，哪些输入应保持逐位一致

**context把历史token接入主干前缀，所以不仅是改一个融合字符串。** [history_pi0.py](src/mme_vla_suite/models/integration/history_pi0.py) 的 `HistoryPi0.embed_prefix` 在当前两路图像和文本之前插入记忆；主干 `gemma_2b` 的宽度为2048，故 `memory_token_dim` 要从modulation的1024改成2048。modulation走 `HistoryBlock` 的 `MemoryAttention/MemoryRMSNorm` 动作调制分支；context走主干注意力。配置选择键是 `integration_type`，没有名为 `memory_mode` 的入口。

记 `B` 为全局batch、`L`为视觉记忆预算、`F=L/64`。同一预算对照时数据路径不改：

```text
改前：同一manifest/source/framesamp-8x8，按原linspace规则选F帧
  → FrameSampDataset.__getitem__
    image[B,L,2048] bf16，pos[B,L,768] f32
    state[B,L,8] f64，mask[B,L] bool；从库读值不变
  → 同一共享内存collate / JAX：state沿既有转换变f32，其余保留
  → 可训练帧编码器：concat(image, pos_proj(pos))，2816→1024；这里改数
  → 可选motion编码器：1536→1024；mem_order同步排序token/mask；排序不改值
  → modulation动作专家 → 同一20步动作监督 → 同一flow matching loss

改后：同一manifest/source/framesamp-8x8；同预算选帧和输入字节不变
  → 同一Dataset、collate和JAX交付，shape/dtype同上
  → 可训练帧编码器：2816→2048；此处是预期的参数形状与数值变化
  → 可选motion：motion_emb[B,160,768] f32 + motion_pos[B,160,256] f32
    pos_proj 256→768，再concat 1536→2048；motion_mask[B,160] bool
    → mem_order[B,L+160] i32；得到memory[B,L+160,2048] bf16
  → [memory | 两路当前图像token | 文本token]主干前缀
  → 动作专家 → 同一20步动作监督 → 同一flow matching loss
```

`FeatureEncoder.encode_perceptual_memory` 的pos投影和帧投影可训练；motion的两个线性层共四个参数叶。`make_attn_mask`让有效记忆可影响文本和动作，当前图像不读取记忆，padding由query/key有效mask隔离。`memory_order`按时刻与类型稳定交错，token和mask必须同步排列。不能只凭张量进入 `Observation` 或总体梯度非零就说motion有效，必须实际核对这四叶及输入扰动响应。

**8192还会首次触发真实短历史。** 当前清单最小 `exec_start_idx=100`，也就是首个执行样本只有101帧。512/2048/4096在本库均满帧；8192要128帧，有2650个样本不足128帧，涉及150集，满帧样本为602961/605611。数量由清单独立计算 `sum(max(0,min(num_timesteps,127)-exec_start_idx))` 得到。未来保持原 `even_sampling_indices` 与补零规则，不删掉这些样本，也不重复真实帧伪造全True mask。选帧仍用 `np.linspace(..., dtype=np.int32)` 的既有舍入；理想整数公式并不总等价。

逐样本视觉history四键字节数为 `L*(2048*2+768*4+8*8+1)=L*7233`；JAX交付后state为f32，对应 `L*7201`。motion额外含491520字节特征、163840字节位置、160字节mask、2688字节顺序，共658208字节/样本。上述仅是输入逻辑量，不能当作峰值显存。

| 视觉预算 | Dataset字节/样本 | JAX字节/样本 | Dataset batch128 | 对应workers×prefetch的history逻辑量 |
|---:|---:|---:|---:|---:|
| 512 | 3703296 | 3686912 | 0.441467 GiB | workers8×2：7.063477 GiB |
| 2048 | 14813184 | 14747648 | 1.765869 GiB | workers16×2：56.507813 GiB |
| 4096 | 29626368 | 29495296 | 3.531738 GiB | workers16×2：113.015625 GiB |
| 8192 | 59252736 | 58990592 | 7.063477 GiB | workers16×2：226.031250 GiB |

512+motion完整history为4361504字节/样本，workers16×2约16.637817GiB。按最终两波配对，纯history队列逻辑量分别约63.571290GiB和339.046875GiB，尚未包含当前图像、动作、worker私有数组、主进程batch及其他副本；第二波必须实测RAM和shm，不能只看GPU容量。

### 5. 当前已知阻断与拟改文件

**现有代码能装配512的context输入，但大预算生产入口还没有接通。** [framesamp_dataset.py](src/mme_vla_suite/training/framesamp_dataset.py) 的 `FrameSampDataset.__init__::new_shape` 只允许1024/2048/4096、modulation、无motion；context2048/4096/8192当前均显式抛错。本轮真实构造已复现，未修改守卫。后续只增加所需的context合法组合；新增分支继续要求原生int预算，错误维数、视角和未经批准的大预算motion继续拒绝。旧512的legacy转换语义不顺带收紧。

| 文件 / 稳定锚点 | 拟改内容 | 既有modulation行为 | context目标行为 |
|---|---|---|---|
| `models/config/robomme/` | 新增五份明确对应本轮数据的history YAML | 原文件不动 | 2048宽；motion组完整复用160-token契约 |
| `training/framesamp_dataset.py::FrameSampDataset.__init__` | 扩展成对形制白名单 | 保持原合法/非法组合 | 放行context2048/4096/8192且无motion；512两态保持 |
| `models/integration/history_pi0.py::make_attn_mask` | 一维ar/na先做前缀和再广播；非一维保留原逻辑 | 要求mask逐位和同预算训练回归通过 | 消除已观察到的大batch重复常量前缀和；候选48项对拍已通过 |
| `src/openpi/models/gemma.py::Attention.__call__` | 4096/8192需要另行验证的省显存实现 | 不直接改既有计算 | query分块原型目前反向不逐位，不能直接纳入生产 |
| `scripts/training/tests/test_pack_guards.py`及新context测试 | 调整context2048旧负例；增加128帧边界与非法组合 | 原路径回归 | 正例能加载，负例在读大数据前报错 |
| 拟新增 `scripts/training/context_launch_contract.py` | 绑定五档完整配置、数据身份、卡数和批准记录 | 旧契约不放宽 | 仅放行表1矩阵与明确差异 |
| `scripts/training/preflight_train_launch.py::main` | 显式选择context契约，复用通用检查 | modulation仍进入原契约 | context不能被硬派发到modulation契约 |
| 拟新增 `scripts/training/prod/run_context_sweep.sh` | 同一TRAIN_ARGS先解析再执行；八卡→四加四两波 | 旧runner不改 | 依赖成功回执交接；隔离run/cache/log |
| 拟新增 `scripts/training/tests/check_context_sweep.py` | 配置/输入/有效性/容量/完成验收 | 不挪用旧PASS | 支持60k/80k与motion，按真实参数树验收 |

上表未以`src/`或`scripts/`开头的配置/模型/数据路径相对于 `src/mme_vla_suite/`。新接口是拟实施内容，不代表生产文件已经存在。现有 `run_modul_budget.sh` 和 `modul_launch_contract.py` 固定八卡及modulation1024/2048/4096，不能拿掉检查后直接复用。`check_modul_completion.py`又固定无motion、80k、61叶，context验收必须从本run初始化树与保存EMA取得**精确叶集合和dtype**，不能照抄61或65。本轮实际context树为纯视觉55叶、motion59叶，但生产验证仍比较完整叶集合，不只比较数量。

### 6. 有效性、容量和存储的放行条件

**有效性分成“参与训练计算”和“最终策略收益”。** 本轮及后续起跑前的梯度、扰动和参数更新验证只能证明前者；训练后的成功率需要同任务、同种子、同评估口径的rollout，不能由loss下降推出。本计划不自动启动正式策略评估。

| 查什么 | 怎么查 / 为什么成立 | 拟用判定行 |
|---|---|---|
| 配置同源 | `config_record.complete_record/compare_records`全字段含类型比对；仅允许精确叶差异，缺键/新增未知键拒绝 | `CONTEXT_CONFIG=PASS variant=... unexpected_diffs=0` |
| 同预算数据未变 | 固定索引，经真实dataset与transform逐键对照shape/dtype/raw bytes；motion包括mem_order | `CONTEXT_INPUT=PASS variant=... mismatches=0` |
| 所有有效历史参与 | 固定模型/噪声/随机流，图像与位置逐token梯度非零；逐帧带扰动loss超过同路径A/A噪声 | `CONTEXT_FRAME_EFFECT=PASS tokens=... bands=...` |
| motion参与 | 非空真实motion四叶梯度各自有限且非零；有效motion扰动loss有响应；全遮时四叶梯度为零 | `CONTEXT_MOTION_EFFECT=PASS active_leaves=4 masked_leaves_zero=4` |
| padding隔离 | 无效位写垃圾，loss、固定噪声动作、全部可训练梯度保持相同；padding输入梯度零 | `CONTEXT_MASK=PASS loss_equal=1 actions_equal=1 all_grads_equal=1` |
| 真训练更新 | 正式pi05结构、同pi05_base来源、batch1单卡一轮真实train_step；有限loss/全梯度/优化器和EMA更新 | `CONTEXT_STEP=PASS variant=... finite=1 updated=1` |
| 目标容量 | 再按每组真实卡数、batch128、workers8/16跑目标形状；保存/恢复一次；不靠小batch冒充 | `CONTEXT_CAPACITY=PASS variant=... batch=128 fsdp=...` |
| 并发资源 | 分别实跑第一波与第二波4+4，记录RAM、shm、NVML、磁盘与两组吞吐 | `CONTEXT_CONCURRENCY=PASS wave=... runs=2` |
| 完成交接 | 唯一退出0、精确checkpoint集合、尾窗有限、最终EMA真实恢复一致、回执绑定run/HEAD/配置SHA | `CONTEXT_COMPLETE=PASS run=... state_step=...` |

以上是目标门槛，实际已完成的子集见第8节，不能把预期判定行当已通过。先单样本，再目标规模；任何失败保留现场，不能降低batch、缩短历史、改学习率或换dummy主干凑通过。新增8192的采样与mask用例必须含真实101–127帧样本及合成边界；现成16/32/64帧测试不覆盖128帧。

**context8192四卡是实质容量风险。** [gemma.py](src/openpi/models/gemma.py) 的 `Attention` 显式生成QK logits再softmax；RoPE动态计算，没有8192位置表截断，`max_token_len=64`只管文本。按两路当前图像各256token、文本64、动作20，总长 `N=L+596`；motion组 `N=672+596=1268`。单层f32 logits逻辑量 `local_batch*8*N*N*4` 如下：

| 组 | N | 每卡样本 | 单层logits逻辑量 |
|---|---:|---:|---:|
| 512 | 1108 | 32 | 1.17 GiB |
| 2048 | 2644 | 32 | 6.67 GiB |
| 4096 | 4692 | 32 | 21.00 GiB |
| 8192 | 8788 | 32 | 73.65 GiB |
| 512+motion160 | 1268 | 16 | 0.767 GiB |

这些是按代码形状估算，**不是实测峰值**；XLA融合、分片和重计算会改变实际分配。本轮已进一步取得真实结果：4096与8192在候选mask下、关闭GPU自动调优后，第一步仍发生运行期OOM，请求分配块分别为78246528072字节（72.872758GiB）和245612515016字节（228.744480GiB）。因此这两档当前不能按四卡/b128直接启动，必须先完成注意力/激活存储优化及新旧对拍；不能用改8卡或batch64代替用户要求。原版512/2048/8192短测的编译超时与此处运行期OOM是不同证据，详见8.5。

**优先组空间可独立放行，五组全保留空间尚不足。** 本地NVMe RAID共6.9T，shm561GiB；诊断后的只读快照可用479492767744字节，即446.562439GiB。本轮真实保存的motion EMA为12622947936字节/份（11.756036GiB），与历史context同宽参数元数据相同；纯视觉历史context元数据为12609567328字节/份（11.743575GiB），预算长度不改变参数形状。按原保存策略共 `12+16*4=76` 份，参数原dtype逻辑量为958541206656字节，即892.711064GiB，比该可用快照多446.148625GiB，尚未计日志、缓存、元数据和临时保存空间，不假定压缩或去重。

优先motion的16份约188.096582GiB，单组保留量可容纳，扣除后约258.465857GiB。后续第一波512+2048还需约328.820GiB，仅这两组就超过该余量约70.354GiB。因此正式起跑前逐阶段核算完整保留量、并发异步保存峰值和日志余量：可以先推进优先组，但第一波交接前必须解决空间，不能把五组队列标成可无人值守跑完。现有数据和旧权重不自动删除，保存频率不自行改变；本轮两份诊断checkpoint及日志也明确留在诊断目录。

### 7. 分阶段实施与判据

| 阶段 | 内容 | 结束判据 |
|---|---|---|
| 本轮 | 核对历史、真实最小验证、根目录方案 | 第8节如实记录通过项、拒绝项与未验证项；文档检查通过 |
| A | 先落优先组正确motion160配置、八卡契约、完成器；补真实dataloader与保存验收 | 本组 `CONTEXT_CONFIG/INPUT/MOTION_EFFECT/MASK/STEP/CAPACITY`、空间检查及原链路回归 |
| B | 确认优先组run名、建档、Beta提交并push，启动8卡512+motion | clean HEAD；本组完整身份与启动门槛齐全；不等待大预算优化完成 |
| C | 准备512/2048的配置、成对守卫和mask优化，做同预算回归与真实4+4验证 | 本组 `CONTEXT_CONFIG/INPUT/FRAME_EFFECT/MASK/STEP/CAPACITY/CONCURRENCY`；解决第一波存储缺口 |
| D | 优先组完成验收后，第一波512+2048各4卡并行 | `CONTEXT_COMPLETE`成功回执；第一波自身Beta与配置/容量/空间门槛 |
| E | 4096/8192省显存实现、数值对拍、真实四卡和并发验收；通过后第二波各4卡 | 所有原训练设定保留；数值与容量失败不能自动放宽；验收未过则不发起第二波 |
| F | 各run结果、清洗日志、原dtype恢复核对与配对正式提交 | 退出与权重来源完整；提交后立即push；不宣称未做的策略评估 |

### 8. 本轮实测记录（追加区）

#### 8.1 环境与数据通路

沙箱内首次 `nvidia-smi` 无法连接驱动，按规则停止并得到用户「b」裁决。随后沙箱外只读检查成功：8张A100-SXM4-80GB，检查时每张0MiB、0%利用率；未停止或修改任何已有tmux会话。

真实输入诊断位于 `v1-store/diagnostics/context-plan-20260928/input_probe.py`，完整日志为同目录 `input_probe.log`，源码锚点为文首HEAD。四任务各取首集/末集的首尾执行样本共16个，motion开关两态共32个；从已训modulation配置内存派生context，只改integration与输出维数，实际调用 `FrameSampDataset`。结果原文：

```text
CONTEXT_INPUT_EXACT=PASS motion=0 samples=16 tasks=4 keys=19 mismatches=0 static_valid=512 motion_valid_min=0 motion_valid_max=0
CONTEXT_INPUT_EXACT=PASS motion=1 samples=16 tasks=4 keys=19 mismatches=0 static_valid=512 motion_valid_min=9 motion_valid_max=127
CONTEXT_INPUT_PROBE=PASS samples=32 rejected_budgets=2048,4096,8192 elapsed_s=2.301847
EXIT_CODE=0
```

`rejected_budgets`表示探针如期复现当前生产守卫拒绝，不是这三档训练通过。2.301847秒是脚本主体耗时，不含依赖导入。全部19键比较shape/dtype/原字节；motion160和mem_order672正确，padding motion特征为零。结论仅覆盖所抽样本的dataset装配，不等同于整库或训练梯度等价。

探针SHA256=`63d4fa61df0aabe12758f57012c380ccd04840a46b0a9c61c73b49fe53cec3a9`，日志SHA256=`d11ae7cbf2569bd0d9d6564054e5b83a263809ec37793889dc827f6ee1d60ba4`。复现入口：

```bash
UV_CACHE_DIR="$PWD/v1-store/cache/uv" \
XDG_CACHE_HOME="$PWD/v1-store/cache/xdg" \
OPENPI_DATA_HOME="$PWD/v1-store/models" \
HF_HOME="$PWD/v1-store/cache/hf" \
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
timeout 240 uv run --frozen --no-sync python \
  v1-store/diagnostics/context-plan-20260928/input_probe.py
```

#### 8.2 基础测试与模型探针

全部诊断源码、日志、JSON和NVML采样位于 `v1-store/diagnostics/context-plan-20260928/`。生产源码、正式YAML、依赖和既有数据均未修改。除独立标注的mask候选进程外，计算使用本提交的生产模型函数；所有模型探针均使用真实 `gemma_2b/gemma_300m` 和pi05_base，不使用dummy或缩宽模型。

最终证据索引为同目录 `evidence_final.json`，绑定110份诊断文件的SHA，不包含大权重分片；索引SHA256=`59da66282862e3f4aac9154bb1e1cc6cd2412fe8ce4ecdc1a061eeef5255caae`。30个已记录源码HEAD均为文首8aee9ced。收尾只读检查原文：`TESTS_FINISHED=PASS active_ctx_sessions=0 active_test_processes=0 gpus=8 used_mib=0 util_pct=0`；本轮会话均自然结束，未停止任何既有会话。末次空间读数为479488401408字节（446.558372GiB），这一稍后快照不改变第6节的分阶段空间结论。

CPU三文件测试为 `test_modul_sampling_oracle.py`、`test_collate_shm.py`、`test_motion_v2_tools.py`：沙箱内真实spawn队列超时280秒、退出124；相同测试在宿主侧 **21 passed in 11.00s**、退出0，日志 `cpu-tests.log` 保留两次结果。独立 `assets_full_probe.log` 按资产锁对pi05_base与tokenizer的21文件、12445985954字节完整SHA核验：`ASSETS_FULL=PASS assets=2 files=21 bytes=12445985954 mismatches=0 elapsed_s=32.772981`，退出0。

`cpu_extended_probe.py`在19.667240秒内完成下列验证，退出0：

```text
CONTEXT_SAMPLING_ORACLE=PASS frames=8 timesteps=2304 mismatches=0
CONTEXT_SAMPLING_ORACLE=PASS frames=32 timesteps=2304 mismatches=0
CONTEXT_SAMPLING_ORACLE=PASS frames=64 timesteps=2304 mismatches=0
CONTEXT_SAMPLING_ORACLE=PASS frames=128 timesteps=2304 mismatches=0
CONTEXT_SHORT_HISTORY=PASS episodes=150 samples=2650 full_samples=602961 total_samples=605611
CONTEXT_PACKED_NPY=PASS budget=512 samples=24 tasks=4 keys=19 mismatches=0 production_assembly=context
CONTEXT_PACKED_NPY=PASS budget=2048 samples=24 tasks=4 keys=19 mismatches=0 production_assembly=modulation
CONTEXT_PACKED_NPY=PASS budget=4096 samples=24 tasks=4 keys=19 mismatches=0 production_assembly=modulation
CONTEXT_PACKED_NPY=PASS budget=8192 samples=24 tasks=4 keys=19 mismatches=0 production_assembly=rejected
CONTEXT_CPU_EXTENDED=PASS comparisons=96 config_variants=5 elapsed_s=19.667240
EXIT_CODE=0
```

9216个选帧例由独立float64步长/截断公式复算，不调用被测`np.linspace`；96对真实样本从packed与独立源NPY装配并比较19键。8192的真实边界index `257079/257105/257106/257107` 对应step `100/126/127/128`，有效token为 `6464/8128/8192/8192`。8192生产构造仍拒绝；大预算模型实验通过独立读取器或已合法的配对modulation读取器取得相同字节，**不冒充大预算context生产入口已经放行**。

五组 `complete_record` 配对是**当前HEAD参考modulation配置与内存派生context候选**的全字段比对；实际data变换、归一化、tokenizer及资产摘要相同，batch128篡改成64的五个负例全部拒绝。本轮没有重建所有历史Beta。另只读复核既存 [2048历史V2记录](docs/training-doc/modul-sweep-20260923-c/records/baseline-2048.json) 与 [4096正式保存现场](docs/training-doc/v2-1600ep-m64x8x8-modul-b128-80k/records/final/final.json)：14项固定训练字段、完整resolved_data/assets/history_values均与当前参考相同。全记录的差别限于诊断run身份、同实体资产目录的相对/绝对写法和history文件名/已解析字典表示。512和512+motion已核历史命令、runtime和指纹，但未作历史V2全字段重建；8192没有同预算历史记录。

#### 8.3 真实模型的记忆作用与mask

`model_matrix_probe.py`加载51个预训练参数叶，只随机初始化4个帧编码器叶；在真实样本、全宽模型上同时求记忆参数与图像/位置输入梯度。以下各行均为 `CONTEXT_MODEL_MATRIX=PASS` 且外层退出0：

| 预算 / 样本 | 有效图像token梯度 / 位置token梯度 | 真实padding | 耗时 | 结果标签 |
|---|---|---:|---:|---|
| 512 / 满历史 | 512 / 512全部非零 | 0；另合成末64位无效 | 88.552秒 | `model-matrix-b512-full-r2` |
| 2048 / 满历史 | 2048 / 2048全部非零 | 0；另合成末64位无效 | 90.926秒 | `model-matrix-b2048-full` |
| 4096 / 满历史 | 4096 / 4096全部非零 | 0；另合成末64位无效 | 95.806秒 | `model-matrix-b4096-full` |
| 8192 / 满历史 | 8192 / 8192全部非零 | 0；另合成末64位无效 | 114.660秒 | `model-matrix-b8192-full` |
| 8192 / 真实101帧 | 6464 / 6464全部非零 | 1728位输入梯度严格零 | 114.664秒 | `model-matrix-b8192-short` |

每档三次A/A的loss、记忆参数梯度及输入梯度相同；首、中、末有效帧带扰动均引起loss变化；mask外垃圾不改变loss、这4叶梯度及输入梯度。这里未遍历每个帧带的扰动，也未声称全部主干参数梯度的padding不变性。

优先组 `model_probe.py` 的真实样本有512帧token及42个有效motion：帧4叶、motion4叶均有非零梯度；有效motion内容置零，loss由0.5349911451变为0.4270034730，差0.1079876721；118个motion padding写垃圾不改变loss与这8叶梯度。该探针71.647秒、退出0。

进一步的 `motion-mask-action-r2` 在128.626秒内完成并退出0：三次A/A相同；42个有效motion图像/位置输入梯度均非零，118个padding梯度为零；全遮motion但保持672候选位置和mem_order不变时，4个motion参数叶梯度严格为零、4个帧参数叶仍非零。固定noise进行10步动作采样，padding垃圾不改变输出字节；有效motion内容置零和全遮的动作最大绝对差分别为0.8494262919和1.0485413373。原文判定项为 `MOTION_INPUT_GRADIENT=PASS`、`MOTION_ALL_MASK_GRADIENT=PASS`、`MOTION_ACTION_PADDING=PASS`、`MOTION_ACTION_EFFECT=PASS`。动作检查作用于模型的32维输出，不等于环境任务成功率或控制性能已验证。

#### 8.4 真实优化器更新与完整EMA保存恢复

`full_step_probe.py`直接调用生产 `train.init_train_state/train_step`，保留原AdamW、裁剪、冻结过滤和EMA；临时诊断覆盖为batch1、FSDP1、workers0、两步。纯视觉512在91.767秒内两步通过、退出0；55叶参数、66叶优化器状态、55叶EMA全部有限，4个记忆叶每步确实更新，主干与记忆梯度范数均非零。

motion两步同样通过：59叶参数、74叶优化器状态、59叶EMA全部有限，8个记忆叶更新。实际前两步学习率为约 `9.9971658e-9 / 1.9994332e-8`，不是第0步学习率为零。原生完整EMA保存和读取约11.76GiB；首次把训练、保存和两次全树摘要塞进240秒，最终摘要未完成即退出124，不记完整通过。

随后使用 `full_step_probe_v2.py`将保存前59叶摘要写入 `expected_ema.json`，原生保存完成并写出 `TRAIN_SAVE_PASS` 回执，探针主体238.277秒；进程收尾仍触发外层124，**保留该非零退出状态**。独立 `restore_probe.py`以该保存前摘要为期望：沙箱内未返回、退出124；宿主CPU同程序读取6.459秒，总69.062秒完成原dtype、shape和逐叶SHA精确对拍，退出0：

```text
CONTEXT_FULL_STEP_RESTORE=PASS leaves=59 bytes=12622947936 original_dtype=true
EXIT_CODE=0
```

结论是两步计算、保存现场与独立真实恢复各自有有效证据，不能把v2外层124改写成正常退出。保存前身份、原生提交元数据、HEAD、脚本/训练源码SHA均被恢复器复验。此checkpoint不包含AdamW动量，不证明可无损续训。

#### 8.5 目标batch、卡数与显存实测

以下均保持global batch128，调用真实训练两步。容量探针重复同一个真实样本，workers0，并显式`--skip-save`，因此它们**不证明生产dataloader吞吐、正常prefetch显存余量、长时稳定或正式4+4并发已经通过**。两步也不能当稳态测速，NVML数字为500ms采样观察峰值，包含初始化、编译/调优与验证。

| 组 / 条件 | 完成步数 / 退出码 | 实测结论 |
|---|---|---|
| 512+motion，8卡，原计算 | 2 / 0 | `cap512motion-b128-f8`：主体212.153秒；8记忆叶均更新；每卡峰值41909–41929MiB |
| 512，4卡，原mask | 0 / 124 | `cap512-b128-f4`：240秒内未完成首轮编译，不是OOM证据 |
| 512，4卡，候选mask | 2 / 0 | `cap512-mask-b128-f4`：主体174.974秒，4记忆叶均更新，所有状态有限；每卡峰值73017–73035MiB |
| 2048，4卡，原mask | 0 / 124 | `cap2048-b128-f4`：编译常量折叠超时，容量未获证明 |
| 2048，4卡，候选mask | 2 / 0 | `cap2048-mask-b128-f4`：主体199.693秒；每卡峰值79439–79453MiB，余量很小 |
| 4096，4卡，候选mask、默认调优 | 0 / 1 | `cap4096-mask-b128-f4`：调优日志明确OOM，请求20.995010GiB |
| 4096，4卡，候选mask、关闭调优 | 0 / 1 | `cap4096-mask-noauto-b128-f4`：实际执行OOM，请求72.872758GiB |
| 8192，4卡，原mask | 0 / 124 | `cap8192-b128-f4`：编译常量折叠超时，不是OOM证据 |
| 8192，4卡，候选mask、默认调优 | 0 / 1 | `cap8192-mask-b128-f4`：首次update失败，请求36.841255GiB |
| 8192，4卡，候选mask、关闭调优 | 0 / 1 | `cap8192-mask-noauto-b128-f4`：实际执行OOM，请求228.744480GiB |

关闭调优实验只增加 `XLA_FLAGS=--xla_gpu_autotune_level=0`，单独留档，不混作默认计算环境。4096/8192关闭调优后仍有 `Execution of replica ... RESOURCE_EXHAUSTED`，证明问题不只是调优时的额外内存。所有OOM数值是请求分配块大小，不是实际占用峰值。2048两步虽通过，但其近80GiB采样峰值要求正式输入预取与保存路径再做容量验收。

#### 8.6 编译候选、数值反例与诊断异常

**mask候选只改变整数/布尔计算顺序。** 原 `make_attn_mask`先将相同一维ar/na广播到batch128，再做前缀和；8192的StableHLO前缀和工作形状是 `[128,8788]`。`mask_cumsum_probe.py::candidate_make_attn_mask`一维时先做 `[8788]`前缀和再广播，非一维走原路径。五长度×六维数组合30组加18个边界全部逐位一致，12.737秒、退出0。CPU编译没有测得提速，不能声称CPU加速；GPU大图的实际候选结果以8.5为准。候选源码SHA为 `cf41c3296561677bdfc0968e1e763796feaaf0eb7a847303f75bdbd5d4b58377`，每次进程内覆盖均有独立 `.mask_override.json`，原生产文件未改。

**长位置RoPE参考曾失败，未改阈值抹平。** `mask_probe.py`的十组角色/有效性mask对照全部逐元素相同，但随后NumPy float32 RoPE参考有1/2249728元素超出预设 `atol=rtol=0.003`，整体退出1。`rope_diagnostic.py`定位为位置6603、频率1处NumPy/JAX的float32幂运算尺度相差1ULP，长位置使相位相差0.0009765625、输出相差0.00339097。改成同float32除法仍失败；独立float64数学参考在原阈值内全通过、最大差0.00187333。原FAIL保留，这不是8192位置表越界，也不是本轮修改生产RoPE的理由。

**query分块仅是尚未通过严格梯度等价的候选。** CPU组件保留原8Q/1KV、head256、完整key轴softmax，query分块C32/64并重计算；N128/256/512、float32/bf16共12组，前向、loss和dQ全部逐位相同，dK/dV全部不逐位。float32最大绝对差约3.87e-12；bf16最大相对L2差约0.473%。没有调整容差或把退出0的诊断脚本称为全梯度等价通过，不能直接据此修改生产attention或宣称解决了8192容量。

其他探针实现问题也保留原记录：首轮512模型矩阵本体通过，但主代理在等待期间给Bash脚本追加环境行，导致后续流式读入偏移、外层EOF退出2；稳定脚本以新标签r2复跑退出0。首轮motion动作实验数值检查完成后，函数名`sample`覆盖样本字典，JSON序列化退出1；保留初版，只重命名函数与调用点，静态序列化检查后新标签r2完整通过。没有覆盖首轮日志或把残缺JSON当完整结果。

#### 8.7 复现入口与会话清单

工作目录为文首仓库根；诊断文件都在 `D=v1-store/diagnostics/context-plan-20260928`。各wrapper已经显式设置uv、模型/缓存路径和平台，CPU脚本沿用8.1的环境。GPU wrapper和恢复结果拒绝覆盖；复跑应给新标签。CPU脚本中部分输出使用固定文件名，复跑前须复制到同层全新诊断目录或显式调整输出落点，不能覆盖本轮证据。

```bash
D=v1-store/diagnostics/context-plan-20260928
# 以下是本轮已执行的CPU入口；再次执行前按正文隔离其固定输出文件。
UV_CACHE_DIR="$PWD/v1-store/cache/uv" OPENPI_DATA_HOME="$PWD/v1-store/models" \
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
uv run --frozen --no-sync python "$D/cpu_extended_probe.py"
# 实际单样本矩阵形式；复现时标签须全新，GPU须已核实空闲。
bash "$D/run_model_matrix_probe.sh" 8192 5 full <新标签>
bash "$D/run_model_matrix_probe.sh" 8192 6 short <新标签>
bash "$D/run_motion_mask_action_probe.sh" 0 <新标签>
# 目标规模原计算与候选mask分别用不同入口、不同标签。
bash "$D/run_full_step_capacity.sh" 0,1,2,3,4,5,6,7 <新标签> \
  --batch-size 128 --fsdp-devices 8 --budget 512 --skip-save
bash "$D/run_full_step_mask_capacity.sh" 0,1,2,3 <新标签> \
  --batch-size 128 --fsdp-devices 4 --budget 2048 --no-motion --skip-save
bash "$D/run_full_step_mask_noautotune_capacity.sh" 0,1,2,3 <新标签> \
  --batch-size 128 --fsdp-devices 4 --budget 8192 --no-motion --skip-save \
  --sample-reader "$PWD/$D/cpu_extended_probe.py"
```

上述是命令形态，尖括号占位符必须替换后才可执行。本轮目标probe源码绑定的是8aee9ced锚点；用于当前计划版本重新复现时，先记录新的HEAD与脚本SHA，不把新结果冒充原锚点运行。初次模型与CPU测试由短进程执行；后续本轮创建过的tmux全名如下，未操作清单外会话：

```text
ctx-test-fullstep-motion-b1-0928
ctx-test-512-full-1-0928
ctx-test-2048-full-2-0928
ctx-test-4096-full-3-0928
ctx-test-8192-full-5-0928
ctx-test-8192-short-6-0928
ctx-test-motion-action-0-0928
ctx-test-fullstep-512-b1-7-0928
ctx-cap-8192-f4-0928
ctx-test-save-motion-v2-4-0928
ctx-test-motion-action-r2-0-0928
ctx-cap-2048-f4-0928
ctx-test-restore-cpu-0928
ctx-cap-8192-mask-f4-0928
ctx-cap-4096-mask-f4-0928
ctx-cap-2048-mask-f4-0928
ctx-cap-4096-mask-noauto-f4-0928
ctx-cap-8192-mask-noauto-f4-0928
ctx-cap-512motion-f8-0928
ctx-cap-512-f4-0928
ctx-cap-512-mask-f4-0928
```

## 第二部分：实施追踪细节

### 9. 前置边界与精确文件方案

1. 不改 `training/config.py::_CONFIGS` 的训练默认值；四卡FSDP只在新启动器覆盖，60k/80k及workers逐组继承。
2. 不改既有modulation YAML、历史context YAML、数据/归一化、pi05_base、motion表或encoder；独立新增配置供本轮使用。
3. 不改变冻结策略、模型层数/宽度、动作监督、文本截断、选帧舍入与数据划分；机制必要的记忆输出宽度2048属于批准目标。
4. 本轮诊断只在 `v1-store/diagnostics/context-plan-20260928/` 写独立探针和日志；不落正式训练产物。探针不进生产导入链。
5. 后续超过5分钟的验证/测速/训练均从clean HEAD、独立tmux和档案启动；本轮短探针设置硬超时，超时记失败，不算通过。
6. 训练失败停住后续依赖并保留现场。`train.py::main`禁止 `--resume/--overwrite`，现有checkpoint只有EMA、没有AdamW动量；不安排自动断点续训。

拟新增history文件均在 `src/mme_vla_suite/models/config/robomme/`：

| 新文件 | 派生源 | 解析后允许变化 |
|---|---|---|
| `perceptual-framesamp-context-8frame-8x8-1600ep.yaml` | 已训modul8frame | integration、memory_token_dim |
| `perceptual-framesamp-context-8frame-8x8-motion160-1600ep.yaml` | 已训modul8frame-motion | integration、memory_token_dim；motion节不变 |
| `perceptual-framesamp-context-32frame-8x8-1600ep.yaml` | 已训modul32frame | integration、memory_token_dim |
| `perceptual-framesamp-context-64frame-8x8-1600ep.yaml` | 已训modul64frame | integration、memory_token_dim |
| `perceptual-framesamp-context-128frame-8x8-1600ep.yaml` | 已训modul64frame | integration、memory_token_dim、budget4096→8192 |

完整配置比对使用精确路径白名单：`history_values.integration_type`、`history_values.memory_token_dim`、8192的 `history_values.budget`；模型history文件标识、run身份和产物目录的对应叶；2048/4096/8192的 `train_config.fields.fsdp_devices:8→4`。如果解析出的模型配置同时含相同history副本，也逐项列出对应路径，不用通配符放行。其余训练项及 `resolved_data`、资产身份必须相同；60k与80k分别选各自基线比对，不先把它们强行归一化。

### 10. 验证实施顺序与对拍

**先最小样本，再扩大，不越过失败项。** 扩展守卫后先跑batch1、单进程、单卡的各档真实输入和有限前后向；CPU负例覆盖新增档的字符串/浮点预算、非64 token_per_image、非1视角、context宽1024、无motion_root以及未批准的大预算motion组合；旧512 legacy语义保持。128帧独立oracle覆盖全域0–2303，含100、126、127、128及现有浮点舍入反例。本轮已取得的子项证据可用于定位，但落入生产源码后的导入、配置、启动器和实际dataloader仍要重新验收。

**输入等价与训练等价分开。** 同预算context/modulation应逐字节交付相同的帧、位置、状态、当前图像、动作和文本；机制输出不同是预期。对守卫/启动链改动则用改前与改后的**同一机制、同一预算**核对：先固定样本/批次摘要，再每侧20+1步真实训练记录，固定seed/噪声/数据顺序/精度和取证器；比较完整loss/梯度与参数状态，不排除“难比较”的叶。旧modulation至少覆盖512关闭、512motion开启和既有2048；context512两态也做改前后不变性。超过5分钟按独立诊断留档，不复用环境指纹不符的历史结果。

8192不能调用目前同样被拒绝的modulation8192充当生产对照，也不为此顺带开放modulation8192。它的输入参考采用独立源NPY+128帧oracle，像本轮96样本对拍那样验证；训练数值的改前/改后对照属于同一context模型与同一输入。

**优先组有效性应使用真实pi05宽度。** 旧 `motion_gates_model.py::_make_models` 的context分支是dummy模型、memory宽64，不能据它宣布正式2048宽主干通过。使用 `gemma_2b/gemma_300m`、正式pi05参数来源、真实样本；A/A至少三次，同一个编译函数建立噪声基线。对所有有效帧带逐带扰动；motion以同一模型保持tensor形状的全遮/扰动进行对照，不把不同长度下的RoPE变化混入内容作用。

**容量必须测试最终拓扑。** 依次单卡batch1 → 优先组八卡batch128 → 每个纯视觉组四卡batch128 → 每波两run重叠；先短smoke，再需要的稳态测速。稳态建议丢弃0–99步预热、统计100–299步，同时以500ms采样NVML，记录util均值、0%占比、慢步/非慢步均值、每组samples/s、RAM/shm峰值和保存停顿。记录介质为AWS本地NVMe RAID `/dev/md0`；不套用历史modulation步时估计context工期。ETA仅在真实稳态数据可用后计算。

**大预算省显存阶段须另有明确数值结论。** 优先调查query轴分块+分块重计算，保留Q/K/V、RoPE、dtype、角色mask和完整key轴softmax。8192四卡按query块256估算，单块logits由约73.65GiB降到约2.15GiB，但仍有dense bool mask、其他激活和反向存储；必须实际核对峰值，不能只算这一项。当前CPU原型dK/dV不逐位，需先解决反向累加策略或在后续方案中明确可接受的一致性标准；原FAIL不能自动改判为PASS。

本机JAX0.5.3的 `fused_attention_stablehlo.py::check_is_flash_attention` 对A100限制head_dim≤128，现有模型head_dim=256，不能直接换为cuDNN实现；默认 `jax.nn.dot_product_attention` 的XLA分支仍生成完整logits。本地Pallas现成接口也不直接覆盖本模型任意角色mask与8Q/1KV组合。因此本计划不把“换一个attention函数”写成已验证修法。

梯度累积仅列为后备设计，当前训练安排不启用：如果后续必须研究它，dataloader仍交付原128样本、原顺序；先按完整128生成增强观测、noise与time，再切微批，按样本数加权累积，最后仅裁剪一次、AdamW一次、step/LR/EMA推进一次。直接把batch改小会改变 `compute_loss` 的normal/beta随机数组和 `preprocess_observation` 的逐图随机键，不能称同一训练条件。即使保留这些，梯度加法次序仍需单独对拍。

### 11. 启动命令与依赖交接

**下面是拟新增runner应生成的命令形态，不是现在可直接执行的生产入口。** `CONFIG/RUN/HISTORY/DEVICES/FSDP`从第1节固定矩阵解析并拒绝其他值。启动器为每个run建立独立日志、JAX/CUDA/W&B缓存与记录目录，所有持久路径在本仓库 `v1-store/`；清除诊断遗留平台/调优覆盖，再显式选择CUDA。资源环境继承本轮容量测量的 `XLA_PYTHON_CLIENT_PREALLOCATE=false`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`，不能拿这些条件下的通过结果去放行不同的内存分配设置；该变化不修改训练超参或模型计算。

```bash
REPO=/scratch/hongze/robomme_policy_learning_MotionJEPA
STORE="$REPO/v1-store"
DS="$STORE/datasets/4task-v2-1600ep-604f16da"
unset JAX_PLATFORMS XLA_FLAGS PYTHONPATH MMEVLA_MOTION_STORE MMEVLA_FRAMESAMP_ALLOW_SUBSET
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export UV_CACHE_DIR="$STORE/cache/uv"
export XDG_CACHE_HOME="$STORE/cache/xdg"
export OPENPI_DATA_HOME="$STORE/models" HF_HOME="$STORE/cache/hf"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MMEVLA_FRAMESAMP_SOURCE="$DS/source"
export MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"
export CUDA_VISIBLE_DEVICES="$DEVICES"
export MMEVLA_JAX_CACHE_DIR="$STORE/cache/jax/$RUN"
export CUDA_CACHE_PATH="$STORE/cache/cuda/$RUN"
export WANDB_DIR="$STORE/logs/wandb/$RUN"
export WANDB_CACHE_DIR="$STORE/cache/wandb/$RUN"
export WANDB_CONFIG_DIR="$STORE/cache/wandb-config/$RUN"
export WANDB_DATA_DIR="$STORE/cache/wandb-data/$RUN"
export TRAIN_RECORD_DIR="$STORE/bench/context-sweep/$RUN"
export TRAIN_FINAL_RECORD_DIR="$TRAIN_RECORD_DIR/final"
TRAIN_ARGS=("$CONFIG" --exp-name "$RUN"
  --assets-base-dir "$STORE/train-assets"
  --data.assets.assets-dir "$STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
  --data.assets.asset-id robomme
  --checkpoint-base-dir "$STORE/train-runs"
  --dataset-path "$DS/framesamp-8x8"
  --weight-loader.params-path "$STORE/models/openpi-assets/checkpoints/pi05_base/params"
  --model.use-history --model.history-config "$HISTORY"
  --fsdp-devices "$FSDP")
# 新context契约先解析并核验同一数组；通过后才执行下面一行。
uv run --frozen --no-sync python scripts/training/train.py "${TRAIN_ARGS[@]}"
```

正式命令不加batch、学习率、步数或worker覆盖。单样本smoke的临时参数仅存在于诊断模式，审批记录和生产模式拒绝携带这些覆盖。`preflight_train_launch.py`当前在 `launch_mode` 非空时直接导入modulation契约，因此实施时必须显式接入新的context契约，不能声称上述片段已拥有完整preflight保护。

**控制器的成功依赖是验收回执，不是tmux退出。** 每run从本阶段已经验收并冻结的Beta执行，同一并行波使用同一代码锚点；优先组与后续阶段可有各自Beta，跨阶段源码差异必须逐项留档并重新验收，不要求五组强行使用同一HEAD。记录源码/配置/数据/权重身份、run UUID、PID和tmux全名。控制器等待子进程与日志统一完成信号，验收成功写入绑定HEAD、run、配置SHA和最终EMA身份的回执；读取并复验回执后才调度下一阶段。失败停止后续发起，不自动杀同波另一组，不碰任何非本轮会话。

拟用会话前缀为 `ctx-`，完整名字在实际 `launch.md`中确定。多变量命令先写运行脚本再交 `tmux new-session -d`；每个脚本使用 `set -o pipefail`、`PYTHONUNBUFFERED=1`、`tee`和唯一 `EXIT_CODE=`尾行。监控管道每一级行缓冲。只在确需清理本轮会话时按清单使用 `tmux kill-session -t '=完整名字'`并前后核对列表，严禁全局清理。

### 12. 完成验收、留档与尚未证明的事项

**完成验收逐run检查实际训练与权重，而非仅数目录。** 60k须有12份checkpoint：5000至55000每5000步及59999，保存现场 `final_record::state_step=60000`；80k须有16份：5000至75000及79999，现场state_step=80000。当前checkpoint仅保存params/assets，不能从恢复权重读出训练步数；须将目录号、原生提交元数据、绑定run/HEAD/UUID的现场step与EMA摘要联合核对。核对唯一退出0、普通日志和末尾99步全部有限、最终异步保存完成、真实恢复的原dtype参数逐叶等于现场EMA、初始化来源正确。用真实树判定精确叶集合，拒绝missing/extra及非有限值；固定noise动作检查另记，不当成功率评估。`TRAIN_RECORD_DIR/TRAIN_FINAL_RECORD_DIR`必须由启动契约检查，不能漏设后再假定有现场记录。

**留档按既有制度，两段写入。** 正式起跑前为五run分别建立 `docs/training-doc/<run_name>/launch.md`、`result.md`、`records/`并更新总索引。launch记录Beta、全部覆盖、数据指纹、GPU/并发顺序和本轮tmux清单；result补实测、完成结果和异常。records只保留Git不能还原的日志/指标/验收输出，不复制YAML或脚本，不提交checkpoint。配置由 `git show <Beta>:<配置路径>` 加启动覆盖还原。

本轮持久Git改动只新增此根目录计划；诊断探针和原始日志留在被忽略的 `v1-store/diagnostics/context-plan-20260928/`，不修改生产链路。提交前执行 `git diff --check`、文件链接核对、拟用名查重和 `git status --short`，仅逐路径暂存此文档，中文提交后立即裸 `git push`到既有upstream。若push被拒，保留原报错并停止，不改写历史。

**当前未完成事项：** 生产配置/守卫/契约尚未落地；4096和8192的四卡/b128已实测失败，需省显存实现后复验；512/2048的通过属于mask候选路径，尚需正式源码回归及真实workers/prefetch/保存和4+4稳定性验收；五组完整保留空间尚不足。所有通过项只证明所列通路和计算/存储行为，不证明训练收敛或策略成功率。按阶段门槛推进，优先组自身通过即可先行，不把大预算失败藏进自动交接逻辑。
