# context 五组训练与通路验证计划

> **当前范围：原训练入口验证、结果核对与方案记录，训练链路保持原样。** 用户要求「不要修改训练计算的机制 如果发现训练计算机制有改进空间只做记录」「dataloader也是 读取逻辑也不要改」。本轮没有修改或替换训练计算、数据加载器、sampler、读取配置、输入守卫或批次组装；没有启动正式60k/80k训练。守卫扩展、mask/attention优化、重计算和梯度累积只保留历史候选记录，不属于后续自动执行步骤。八轮原入口验证及收尾核验完成后更新本文，完整证据以[本轮结果](docs/training-doc/ctx-stock-scaling-20260928/result.md)和[运行口径](docs/training-doc/ctx-stock-scaling-20260928/launch.md)为准。
>
> 创建日期保持2026-09-28，America/New_York，文件名不改。权威工作副本为 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，环境B、分支 `v2-motionmem`。第8.1–8.7节保存最初 `8aee9ced0bbc7bc4d2a863539fa8c14873bb9832` 锚点下的历史实验；本次八轮原入口诊断实际运行干净源码快照 `6db0e0ab9ef266d0e80fc1dd8d4383705bd81ab0`，快照位于 `v1-store/bench/context-stock-scaling-20260928/source-6db0e0a`。后续文档提交不追称为起跑源码。
>
> 外部依赖以实际源码快照的 `uv.lock`、`scripts/assets/ASSETS_LOCK.json` 为准，benchmark gitlink为 `856bc3a189d4172f3f47dbee4424d585f8d78db3`。提交体例沿用 `docs:` 与仓库既有训练提交编号；本轮结果与根计划只做文档留档，不制造正式训练Beta。不升级依赖、不下载或重建数据、不使用Slurm或turbo。[64代理只读检查报告](docs/context-gpu-scaling-readonly-audit-20260928.md)及其排除项保留。

用户决定按时间顺序原文保留：

1. 「根目录给出方案 我已经训练过了modulation的512 2048 4096 8192 和 512+motion 我要再训练context下的这几个 训练 data 等都要保持一致 先验证是否通路 并且有效 然后写计划到根目录 优先训练512+motion 8卡 其他的在512motion跑完后 跑 4卡一个 2个并行」
2. 「b」——历史简短答复；环境B由本轮实际判定命令确认。
3. 「实际要的是 1024」——中途澄清，已被下面两条最终决定替代。
4. 「context要跑8192。」
5. 「1024不要了」
6. 「尽可能多做验证 验证完了再写计划」
7. 「你可以自由使用8gpu全都可以用 gogogo」
8. 「测试全部放行」
9. 「不要再问我你现在是full permission模式」
10. 「做你想做的实验」
11. 「不要这么干 评估8gpu的训练效果 2个4gpu并行 对比8gpu 会慢多少 不要修改内部训练机制 除非用户授权」
12. 「尽可能穷尽所有选项 但是不要修改训练链路 只是让你仔细检查」
13. 「不要修改训练计算的机制 如果发现训练计算机制有改进空间只做记录」
14. 「dataloader也是 读取逻辑也不要改」

**结果先行：512+motion160和纯512各自完成前八卡、双四卡、后八卡，共八个独立原入口run，均为global batch128、401步、原生保存及CPU完整恢复。** 单个四卡任务相对八卡的耗时分别增加94.23%–95.15%与88.02%–88.25%；双四卡合计吞吐分别提高2.7274%与6.3065%。两份等步数任务的稳态完工时间外推分别减少2.4258%与5.8748%，不是完整训练工期承诺。初始权重及数据内容核验见第8.8–8.9节。

**2048、4096、8192当前首先受原context读取守卫阻断，没有原入口四卡或八卡的容量与速度结论。** 早期候选mask、配对modulation读取器或独立装配产生的成功、超时和显存不足仅属于其历史实验条件；不把它们当成本轮原入口结果，不据此安排优化或删除8192目标。

## 第一部分：目标、结果与可核对的依据

### 1. 最终五组与原定训练顺序

**目标仍是context512、2048、4096、8192、512+motion160，不安排context1024。** 用户原定顺序为先用八卡训练512+motion，完成后其余组各用四卡、两个并行。本轮四卡motion只是与八卡比较的短程诊断，不改变这个正式训练目标。原计划把纯视觉组配成512+2048和4096+8192两波；由于三档大预算守卫仍拒绝，该配对目前只是未启动的历史排程。

| 目标顺序 | context组 | 最大帧数 / motion token | 具名训练配置 | 正式步数 / workers | 原定GPU / FSDP | 原拟run_name | 当前证据 |
|---|---|---|---|---|---|---|---|
| 优先组 | 512+motion | 8 / 160，总记忆672 | `mme_vla_suite_b128_80k` | 80000 / 16 | 八卡 / 8 | `v2-1600ep-m8x8-context-motion-b128-80k` | 原入口八卡及双四卡各401步和保存恢复通过；未跑80k |
| 后续纯视觉 | 512 | 8 / 0 | `mme_vla_suite_b128_60k` | 60000 / 8 | 四卡 / 4 | `v2-1600ep-m8x8-context-b128-60k` | 原入口八卡及双四卡各401步和保存恢复通过；未跑60k |
| 后续纯视觉 | 2048 | 32 / 0 | `mme_vla_suite_b128_80k` | 80000 / 16 | 四卡 / 4 | `v2-1600ep-m32x8x8-context-b128-80k` | 原context读取守卫拒绝；容量与速度未验证 |
| 后续纯视觉 | 4096 | 64 / 0 | `mme_vla_suite_b128_80k` | 80000 / 16 | 四卡 / 4 | `v2-1600ep-m64x8x8-context-b128-80k` | 原context读取守卫拒绝；容量与速度未验证 |
| 后续纯视觉 | 8192 | 128 / 0 | `mme_vla_suite_b128_80k` | 80000 / 16 | 四卡 / 4 | `v2-1600ep-m128x8x8-context-b128-80k` | 原守卫拒绝，且无现成可用生产配置；容量与速度未验证 |

表中名称与步数保留原目标，不表示已起跑、已确认可运行或可以覆盖已有run。本轮没有发起五组正式训练，也没有新增生产启动器、自动交接控制器或批准流程。每个实际诊断run独立从pi05_base初始化，没有承接其他run权重。

本地512历史基线为60,000步、workers8；2048、4096和512+motion为80,000步、workers16。未找到modulation8192正式训练档案；8192按原目标保留4096的80k口径及128帧预算，不能声称有已核实的同预算历史对照。全局batch保持128，四卡每卡32样本、八卡每卡16样本；不改batch64、不缩放学习率、不引入梯度累积。实际设备数与FSDP均由本轮前置和runtime记录核实，双四卡使用互斥的GPU0–3和GPU4–7。

### 2. 已训练基线与固定训练口径

| 基线组 | 已训练 run / 权威档案 | 起跑提交 | 完成依据 |
|---|---|---|---|
| 512 | [v2-1600ep-m8x8-modul-b128-60k](docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/result.md) | `dd07f18fc385b01eb52db7563fe5f202997b9706` | 60k、12份 checkpoint、最终59999、退出0；历史真实恢复61叶 |
| 2048 | [v2-1600ep-m32x8x8-modul-b128-80k](docs/training-doc/v2-1600ep-m32x8x8-modul-b128-80k/result.md) | `55647ff33c8ddb9ec324fdbcee8bd1491456725b` | 80k、16份 checkpoint、最终79999、退出0；历史真实恢复61叶 |
| 4096 | [v2-1600ep-m64x8x8-modul-b128-80k](docs/training-doc/v2-1600ep-m64x8x8-modul-b128-80k/result.md) | `0571fea5f626e822ca2b23b9e5f90c5bf31dd5eb` | 80k、16份 checkpoint、`completed.json::status=PASS` |
| 512+motion | [v2-1600ep-m8x8-modul-motion-b128-80k](docs/training-doc/v2-1600ep-m8x8-modul-motion-b128-80k/result.md) | `2f10473161b760f16d9240d3c2959ff326cde66b` | 80k、16份 checkpoint、最终79999、退出0；历史真实恢复65叶 |
| 8192 | 当前本地未找到同预算 modulation 档案 | 无 | 本计划按4096扩展；同预算历史对照未验证 |

上述完成结果来自既有归档，最初方案核对了档案及本地产物在位，没有重新读取所有历史modulation权重。额外发现的 modulation1024 历史 run 不属于本轮训练矩阵。

**除机制、预算和已指定的卡数外，训练条件保持如下。** 权威入口为 [training/config.py](src/mme_vla_suite/training/config.py) 的 `_CONFIGS`；60k 与80k分别继承自己的具名配置，本轮没有修改全局默认超参。

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

**本轮原入口诊断与正式训练的差异单独记录。** 每个模型做前八卡、双四卡两run、后八卡共四次独立运行，均从同一pi05_base初始化。只把诊断循环截为401步，保留原60k/80k学习率定义及5000步预热；原 `log_interval=100`、`save_interval=5000`、`keep_period=5000` 不变，最后一轮仍由原入口保存checkpoint400。W&B网络关闭，两拓扑统一使用 `XLA_PYTHON_CLIENT_PREALLOCATE=false`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`，没有修改mask、attention、loss、精度或梯度逻辑，也没有新增训练同步点。诊断差异不能混入历史正式训练速度比较。

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

**本轮motion使用已核对的160-token配置。** 旧 `perceptual-framesamp-context-8frame-8x8-motion.yaml` 仍是motion96、40集库、旧 `wan-v8-filter10-72ep-a`，缺demo17补尾规则，因此原入口诊断使用 `v1-store/diagnostics/context-gpu-throughput-20260928/history-context-motion160.yaml`。它来自已训modulation512+motion配置，仅选择 `integration_type: context` 和必要的 `memory_token_dim: 2048`，完整motion字典保留，SHA256为 `b69a4bd4398b08298c6dc12db7150a0873c327b9555ceee4a5166094c1a610ac`。现有正式YAML没有被修改。

纯512直接使用6db快照中的 `perceptual-framesamp-context-8frame-8x8.yaml`，SHA256为 `299b6d3f2b52c7b3c1894f8594183a3140d0d2a6b91250a2eb8e568c528344d8`。与历史modulation512相比，除融合方式和输出宽度外，它还多出 `motion.enabled=false` 字典；旧96预算、40集路径等关闭态字段仅进入配置记录，不触发motion表读取或模型层创建。本轮没有清理这些不生效的字段，也未改用modulation配对读取器。

最初小文件指纹及历史来源核查，与收尾的完整内容核验范围不同。初始权重/tokenizer完整哈希及八份EMA共享叶检查见第8.8节；训练特征35文件的完整核验见第8.9节。完整哈希证明所读文件符合既有固定指纹，不等于重新运行全部VAE或motion encoder。

### 4. 既有机制、实际输入与有效性边界

**context把历史token接入主干前缀，所以不仅是改一个融合字符串。** [history_pi0.py](src/mme_vla_suite/models/integration/history_pi0.py) 的 `HistoryPi0.embed_prefix` 在当前两路图像和文本之前插入记忆；主干 `gemma_2b` 的宽度为2048，故 `memory_token_dim` 要从modulation的1024改成2048。modulation走 `HistoryBlock` 的 `MemoryAttention/MemoryRMSNorm` 动作调制分支；context走主干注意力。配置选择键是 `integration_type`，没有名为 `memory_mode` 的入口。

记 `B` 为全局batch、`L`为视觉记忆预算、`F=L/64`。下图比较既有modulation与context两种机制，不表示本轮正在重构生产链路；实际测速直接调用原训练入口与原数据交付链：

```text
既有modulation：同一manifest/source/framesamp-8x8，按原linspace规则选F帧
  → FrameSampDataset.__getitem__
    image[B,L,2048] bf16，pos[B,L,768] f32
    state[B,L,8] f64，mask[B,L] bool；从库读值不变
  → 同一共享内存collate / JAX：state沿既有转换变f32，其余保留
  → 可训练帧编码器：concat(image, pos_proj(pos))，2816→1024；这里改数
  → 可选motion编码器：1536→1024；mem_order同步排序token/mask；排序不改值
  → modulation动作专家 → 同一20步动作监督 → 同一flow matching loss

既有context：同一manifest/source/framesamp-8x8；同预算按原规则交付
  → 同一Dataset、collate和JAX交付，shape/dtype同上
  → 可训练帧编码器：2816→2048；此处是预期的参数形状与数值变化
  → 可选motion：motion_emb[B,160,768] f32 + motion_pos[B,160,256] f32
    pos_proj 256→768，再concat 1536→2048；motion_mask[B,160] bool
    → mem_order[B,L+160] i32；得到memory[B,L+160,2048] bf16
  → [memory | 两路当前图像token | 文本token]主干前缀
  → 动作专家 → 同一20步动作监督 → 同一flow matching loss
```

`FeatureEncoder.encode_perceptual_memory` 的pos投影和帧投影可训练；motion的两个线性层共四个参数叶。`make_attn_mask`让有效记忆可影响文本和动作，当前图像不读取记忆，padding由query/key有效mask隔离。`memory_order`按时刻与类型稳定交错，token和mask必须同步排列。不能只凭张量进入 `Observation` 或总体梯度非零就说motion有效，必须实际核对这四叶及输入扰动响应。

**8192还会首次触发真实短历史。** 当前清单最小 `exec_start_idx=100`，也就是首个执行样本只有101帧。512/2048/4096在本库均满帧；8192要128帧，有2650个样本不足128帧，涉及150集，满帧样本为602961/605611。数量由清单独立计算 `sum(max(0,min(num_timesteps,127)-exec_start_idx))` 得到。本轮保持原 `even_sampling_indices` 与补零规则，不删掉这些样本，也不重复真实帧伪造全True mask。选帧仍用 `np.linspace(..., dtype=np.int32)` 的既有舍入；理想整数公式并不总等价。

逐样本视觉history四键字节数为 `L*(2048*2+768*4+8*8+1)=L*7233`；JAX交付后state为f32，对应 `L*7201`。motion额外含491520字节特征、163840字节位置、160字节mask、2688字节顺序，共658208字节/样本。上述仅是输入逻辑量，不能当作峰值显存。

| 视觉预算 | Dataset字节/样本 | JAX字节/样本 | Dataset batch128 | 对应workers×prefetch的history逻辑量 |
|---:|---:|---:|---:|---:|
| 512 | 3703296 | 3686912 | 0.441467 GiB | workers8×2：7.063477 GiB |
| 2048 | 14813184 | 14747648 | 1.765869 GiB | workers16×2：56.507813 GiB |
| 4096 | 29626368 | 29495296 | 3.531738 GiB | workers16×2：113.015625 GiB |
| 8192 | 59252736 | 58990592 | 7.063477 GiB | workers16×2：226.031250 GiB |

512+motion完整history为4361504字节/样本，workers16×2约16.637817GiB。按最终两波配对，纯history队列逻辑量分别约63.571290GiB和339.046875GiB，尚未包含当前图像、动作、worker私有数组、主进程batch及其他副本；这些大预算并发队列量目前没有原入口实测，不能由输入逻辑量或512档GPU结果推定可用。

**有效性结论按实际覆盖范围表述。** 本轮八次原入口运行的记忆编码器梯度均有有效记录，全部原生日志标量有限。重新核对的旧原模型探针显示：指定纯512真实样本的512个图像/位置token有输入梯度；指定motion样本的42个有效token有输入梯度、118个padding输入梯度为零，有效motion内容扰动影响所测loss和固定噪声动作。探针只覆盖记录的样本、所查记忆叶与输入梯度，不等于全部训练样本、全部主干梯度或环境成功率验收。

原 `build_exec_lookup` 的step是episode绝对帧号，当前最小为100；512只需8帧，所以全部605611个样本都没有真实静态padding，旧末64位屏蔽是合成边界测试。2048/4096同样都有足够真实历史。8192所需的2650个短历史样本由原公式与清单只读推导：RouteStick为1350个，VideoUnmaskSwap为1300个；index257079为6464个真实token加1728个padding，step126仍有64个padding，step127起满帧。这些推导不表示已绕过大预算守卫。当前 `use_state_emb=false`，不能把历史状态字段的零影响写成有效消费证据。

### 5. 当前原入口阻断与历史候选

**原入口的阻断发生在取得首批数据之前。** [framesamp_dataset.py](src/mme_vla_suite/training/framesamp_dataset.py) 的 `FrameSampDataset.__init__::new_shape` 仅接受既有1024/2048/4096、modulation、无motion组合，context大预算不在该集合；旧512走原legacy路径。本轮原构造器检查确认context512可构造605611个样本，context2048/4096/8192均按形制断言拒绝，原异常及退出0的检查记录见[原始守卫结果](docs/training-doc/ctx-stock-scaling-20260928/records/original-guards.json)。`ORIGINAL_CONTEXT_GUARDS=PASS`表示守卫行为被如实核实，不表示被拒绝的训练组合通过。

原 `train.py::main` 先取得真实首批数据，再初始化模型并编译训练步；守卫拒绝后没有现成原入口容量模式。现有 `inputs_spec`包含三路当前图像及float32历史，而实际交付是两路图像及bfloat16历史；`fake_obs/fake_act`生成假输入，`eval_shape`仅推形状，编译内存分析也不能替代真实读取与执行。当前不改守卫、不借用modulation读取器、不采用独立装配来伪称原入口通过，增加GPU数量也不能解除这个输入阻断。

以下保留最初方案的候选清单，**全部仅记录，不实施、不安排自动跟进**：

| 文件 / 稳定锚点 | 历史候选内容 | 当前状态与限制 |
|---|---|---|
| `src/mme_vla_suite/models/config/robomme/` | 新增五档具名history YAML | 未新增生产配置；本轮实际配置来源见第3节 |
| `training/framesamp_dataset.py::FrameSampDataset.__init__` | 扩展context大预算白名单 | 未修改，不绕过原守卫 |
| `models/integration/history_pi0.py::make_attn_mask` | 一维前缀和后广播 | 旧48项mask对拍仅属候选证据；本轮原入口未采用 |
| `src/openpi/models/gemma.py::Attention.__call__` | query分块与重计算 | 旧原型dK/dV不逐位；不落入训练计算 |
| `scripts/training/tests/test_pack_guards.py`及新测试 | 调整旧负例、增加128帧用例 | 未改现有测试来放行被拒绝组合 |
| `context_launch_contract.py`、`preflight_train_launch.py::main` | 新context契约与分发 | 未新增或改写生产契约 |
| `prod/run_context_sweep.sh`、`tests/check_context_sweep.py` | 五组自动启动和完成交接 | 未实现、未执行 |
| loss、精度、梯度累积、采样和数据交付 | 吞吐或容量改进设想 | 只记录限制，不修改或替换这些机制 |

上表未写完整前缀的模型/数据路径相对于 `src/mme_vla_suite/`，脚本相对于 `scripts/training/`。现有modulation启动与完成器固定自己的机制、预算、80k和参数叶口径，不能移除约束后用于context。本轮恢复按实际独立模型schema核对完整叶集合和dtype：纯512为55叶，motion为59叶，没有照抄历史modulation的61/65叶。

### 6. 已有验证、容量边界与存储约束

**短程验证证明通路和所测条件下的速度，不证明策略收益。** 八次原入口运行均完成401次更新、五条原生metrics、最后checkpoint400保存及CPU完整恢复，原生与外层退出均为0。每组采用对应原workers、batch128、两次独立八卡基线和真正并行的两个四卡进程，改进候选没有进入被测进程。具体指标和证据入口见第8.8节与第10节。

| 检查内容 | 已取得证据 | 结论边界 |
|---|---|---|
| 原生产配置与来源 | 前置、runtime、完整配置、源码/依赖和history交叉核验 | 拓扑比较只允许run身份、FSDP及派生输出路径差异 |
| 原输入守卫 | 512构造成功；2048/4096/8192如期拒绝 | 不将检查脚本PASS读成大预算通过 |
| 训练与并发 | 八run各401步；双四卡主窗口均有对侧原训练进度覆盖 | 没有完整批次字节与实际随机数组取证，不证明同轨迹 |
| 实际保存恢复 | 八份原生checkpoint400；55/59叶完整原dtype恢复、有限性及归一化通过 | 没有保存前现场EMA摘要，不声明与现场EMA逐位相等 |
| 初始化与冻结 | 初始21文件完整哈希；八run的23冻结共享叶相同、28非冻结共享EMA叶摘要变化 | 排除新增随机记忆叶；不证明每个元素均改变 |
| 训练特征内容 | 第8.9节的35文件完整只读核验 | 不等于重新运行VAE/encoder或全部源数据重扫 |

**大预算原入口显存峰值目前未知。** 既有 `Attention`显式生成QK logits，再沿完整key轴softmax；两路当前图像、文本64、动作20时，纯视觉总长 `N=L+596`。八卡每卡16样本对应单层float32 logits逻辑量为 `16*8*N*N*4`：2048约3.333GiB，4096约10.498GiB，8192约36.826GiB；四卡每卡32样本时分别约6.667、20.995、73.651GiB。它们只是代码形状的逻辑量，不是实测峰值或无条件的峰值下界；不能把历史OOM分配请求除二当八卡实测。

早期4096/8192候选mask及不同调优条件下的显存不足原样保留在第8.5节。当前原context读取守卫仍拒绝这两档，故不能写成“原四卡已OOM，必须优化”。512两态实测倍率也不能外推到大预算，或用它证明512+2048、4096+8192异组并发容量。

**保存空间仍按完整目标核算，不自动清理旧产物。** 原保存策略下512的60k需12份，其余四组80k各16份，共76份；纯视觉每份参数原dtype逻辑量12609567328字节，motion为12622947936字节，总计958541206656字节、约892.711GiB，尚未计日志、缓存、临时保存与元数据。不假定压缩、去重或自行改保存频率。最初方案446.56GiB可用空间是当时的历史快照；本轮新增八份401步检查点与留档之后，不能再用该快照宣称空间足够。收尾可用空间为374842220544字节、349.099022GiB，低于五组完整参数保留逻辑量958541206656字节；仅这两项差额已有583698986112字节、543.612042GiB，当前空间不能按该未压缩逻辑预算保证五组全保留，实际压缩后的物理占盘仍需核算。此为收尾快照，不是未来起跑时可复用的固定容量。

### 7. 本轮验证状态与目标阻断

本轮按“保持原机制验证 → 独立验收 → 更新根计划”收尾，没有沿旧A–F表实施生产改造或启动长训练。

| 阶段 | 实际内容 | 完成依据 |
|---|---|---|
| 原入口边界 | 干净6db源码、实际导入位置、原配置/资产与守卫检查 | `STOCK_PREFLIGHT=PASS`；`ORIGINAL_CONTEXT_GUARDS=PASS`，拒绝范围保留 |
| motion四run | 八卡前基线、双四卡、八卡后基线；每run401步及原生保存 | 四份退出0、恢复通过；`motion-comparison.json::status=PASS` |
| 纯512四run | 相同四轮结构，保留workers8和原60k学习率定义 | 四份退出0、恢复通过；`plain-comparison.json::status=PASS` |
| 初始权重参考 | 21文件12445985954字节完整哈希、八份共享叶与冻结核验 | `INITIAL_ASSETS_FULL=PASS`；`INITIAL_REFERENCE_CHECK=PASS shared=51 frozen=23 runs=8`；退出0 |
| 训练特征收尾 | GPU测速完成后独立只读检查35文件313426537152字节 | 第8.9节实际结果；未在性能窗口内预热整库 |
| 文档收尾 | 保留历史失败，回写原入口数字、当前阻断与剩余盲区 | 只更新计划和本轮证据留档；不修改训练计算、loader或读取 |

五组正式训练仍未启动。当前2048/4096/8192的状态是原输入守卫阻断；本轮到此记录，不把解除阻断或落地优化写成已安排的下一步，也不新增批准流程。

### 8. 实测记录与历史边界

**第8.1–8.7节逐字保留最初8aee9ced锚点下的历史记录。** 其中“本轮”“本提交”“后续复现”均指当时实验；旧候选覆盖、配对读取器和独立装配没有进入此次原入口诊断，历史命令不是当前重跑指令。8.5节中的超时与OOM分别按原条件保留，不能用于替代第5–6节的原入口阻断结论。原训练入口八轮结果和本次收尾核验另追加于第8.8–8.9节。

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

#### 8.8 原训练入口八轮性能、保存与初始化核验

本次实际源码为 `6db0e0ab9ef266d0e80fc1dd8d4383705bd81ab0`，运行原 `scripts/training/train.py`、原模型、原数据加载器和原读取守卫。AWS本地NVMe RAID `/dev/md0`、8张A100-SXM4-80GB、global batch128；motion为workers16，纯512为workers8；每run401步。没有运行三份冻结的 `scripts/training/tests/context_gpu_scaling.py`、`compare_context_gpu_scaling.py`、`run_context_gpu_scaling.sh`，没有向样本添加索引键，没有进程内替换mask、attention或loss。八份独立run全部原生及外层退出0，checkpoint400提交完成，CPU按实际模型schema完成原dtype恢复和有限性检查。

原 `train.py::_MetricsProxy`在log100同步后记录wall_time；主窗口使用 `(t300-t100)/200`，即更新101–300，八卡参照为前后两个独立200步窗口等权均值。双四卡各自主窗口内均有另一侧原训练进度完整覆盖；计算公式与边界见第10节。完整结果分别为[motion比较](docs/training-doc/ctx-stock-scaling-20260928/records/motion-comparison.json)、[纯512比较](docs/training-doc/ctx-stock-scaling-20260928/records/plain-comparison.json)。

| 原生主窗口 | 512+motion160秒/步 | 纯512秒/步 |
|---|---:|---:|
| 独立前八卡 | 1.979399133 | 1.727037694 |
| 双四卡左组 | 3.844594864 | 3.247254070 |
| 双四卡右组 | 3.862700729 | 3.251229017 |
| 独立后八卡 | 1.979331276 | 1.727116215 |
| 两次八卡等权均值 | 1.979365205 | 1.727076955 |
| 八卡前后漂移 | −0.003428% | +0.004547% |

| 比较指标 | 512+motion160 | 纯512 |
|---|---:|---:|
| 左／右四卡单任务耗时增加 | 94.2337%／95.1485% | 88.0202%／88.2504% |
| 八卡参考吞吐，样本/秒 | 64.667197 | 74.113663 |
| 双四卡合计吞吐，样本/秒 | 66.430931 | 78.787647 |
| 合计吞吐增加 | 2.7274% | 6.3065% |
| 两份等步数任务稳态完工时间外推减少 | 2.4258% | 5.8748% |

motion双四卡主窗口单卡平均GPU利用率为99.8403%–99.9195%，单卡零利用率采样占比最高0.0652%，采样显存峰值范围71965–71985MiB。纯512双四卡为99.8039%–99.9081%、零采样占比0、73005–73019MiB；纯512前后八卡均值范围99.4913%–99.7849%，零采样占比0，采样显存峰值41785–41803MiB。NVML间隔请求500毫秒，是采样观察，不代表瞬时精确峰值。原指标每100步一条，不能识别逐个慢步、计算逐步p95，不能据此宣称所有瓶颈消失。

八份checkpoint400均完整恢复：motion每份59叶、12622947936字节，纯512每份55叶、12609567328字节，精确路径、shape、原dtype、全部有限值及归一化比较通过。预期schema由原模型和freeze_filter独立生成，恢复使用CPU、`restore_type=np.ndarray,dtype=None`。本轮未启用 `TRAIN_FINAL_RECORD_DIR`，没有保存前现场EMA逐叶摘要；不能将完整恢复写成与现场EMA逐位相等，也不能声称没有AdamW状态的checkpoint可以无损续训。

**初始化参考另做独立核验。** `check_initial_reference.py`在八轮测速后检查pi05_base及tokenizer的21文件、12445985954字节完整指纹，只恢复一次初始权重；按原 `_merge_params`规则转换到独立schema要求的dtype，与八份恢复报告的原字节SHA比较。65.893秒完成，原日志为 `INITIAL_ASSETS_FULL=PASS`、`INITIAL_REFERENCE_CHECK=PASS shared=51 frozen=23 runs=8`，唯一退出0，证据见[初始化参考结果](docs/training-doc/ctx-stock-scaling-20260928/records/initial-reference-check.json)。八run的23个冻结图像共享叶逐字节匹配参考，28个共享非冻结EMA叶各自摘要变化；这不表示每个元素均改变或给出变化幅度。纯512新增4叶、motion新增8叶是随机记忆参数，不在该参考比较内。

同一八卡拓扑前后run也存在首步数值差异，本轮没有完整首批张量、初始随机记忆参数和实际随机数组取证，故不把差异单独归因于卡数，不宣称逐位训练轨迹一致。全部401步仍处于原5000步学习率预热；没有策略rollout、全程60k/80k稳定性或统计置信区间。初始化、编译、周期保存、长训练变化和一侧提前结束后的资源变化不包含在稳态完工时间外推中。

#### 8.9 当前训练特征内容的完整只读核验

本次核验在全部八轮原入口训练和保存结束后执行，进入时已核对八份唯一退出回执与GPU无计算进程；没有在测速窗口内全量预热数据。会话 `ctx-stock-data-fullhash-0928`从2026-09-29 00:37:11.535746 UTC运行至00:49:41.454804 UTC，总耗时749.919秒，其中帧库检查749.413秒。完整输出见[数据核验JSON](docs/training-doc/ctx-stock-scaling-20260928/records/data-content-recheck.json)和[原始核验日志](docs/training-doc/ctx-stock-scaling-20260928/records/data-content-recheck.log)。

外层 `check_current_data.py`调用原 `framesamp_store.run_full_checks(StoreMeta.load(...))`与 `motion_store.run_full_checks(MotionMeta.load(...))`，完整读取32个帧图像特征part、位置表、状态表和motion表，共35文件、313426537152字节，逐项SHA256与事先固定的元数据一致。帧库1192918行，motion库71316行；motion索引SHA另由元数据加载核验，不计入这35个数据文件。前后帧/motion元数据SHA均与第3节相同，没有创建pack.lock或改写元数据，原日志为：

```text
DATA_FULL_CHECK_START files=35 bytes=313426537152
FRAMESAMP_FULL_HASH=PASS rows=1192918 seconds=749.413
MOTION_FULL_HASH=PASS rows=71316
DATA_FULL_CHECK=PASS metadata_unchanged=1
EXIT_CODE=0
```

该结果证明当前已打包训练特征内容与固定元数据完整指纹一致，范围不含重新扫描原始RGB/pkl、重新运行VAE或motion encoder。历史帧库逐行源NPY对拍、motion整表独立oracle和9月20日更大payload集合的哈希仍按各自[数据档案](docs/dataset-build-doc/4task-v2-1600ep-604f16da/result.md)、[motion档案](docs/dataset-build-doc/4task-v2-1600ep-motion-demopad17/result.md)、[导出验证档案](docs/dataset-build-doc/hf-export-v2-1600ep-trainset-20260920-v1/result.md)引用；不能把这些不同范围相加后宣称本次重新验证了全部编码数值。

## 第二部分：实际运行口径与证据追踪

### 9. 源码、配置与数据交付保持范围

1. 被测源码始终来自干净6db快照；CPU前置核实 `openpi`、`openpi_client`、`mme_vla_suite`及模型/数据模块实际导入路径，避免editable环境退回主树。主仓uv管理的解释器只读复用，没有同步环境或安装依赖。
2. 本轮选择既有context机制与相应2048输出宽度，history配置来源见第3节。60k/80k具名配置、batch128、workers8/16、学习率、AdamW、冻结和EMA均继承原定义；401步仅为诊断停止点。
3. Dataset、sampler、shuffle/seed/drop_last、spawn、persistent workers、原prefetch与共享内存collate不变。没有独立大预算读取器、配对modulation读取配置、索引包装或小batch替身。
4. 原mask、attention、loss、dtype、RoPE、随机数调用和训练循环保持原样；原log100同步及最后checkpoint400保存仍执行，没有额外训练同步点或保持负载循环。
5. 原始实验在 `v1-store/diagnostics/context-plan-20260928/`；原入口诊断与外层核验在 `v1-store/bench/context-stock-scaling-20260928/`。两个目录各有独立身份，不能混用相同标签或将历史候选结果归入原入口运行。
6. 外层 `run_original.sh`、`preflight.py`、`run_dual4.sh`只负责启动、核查和外部采样；离线 `summarize_native.py`、`check_checkpoint.py`、`check_initial_reference.py`、`check_current_data.py`不注入被测进程。实际脚本SHA、环境和命令均在[launch](docs/training-doc/ctx-stock-scaling-20260928/launch.md)与[result](docs/training-doc/ctx-stock-scaling-20260928/result.md)留档。

最初拟新增的五份history生产YAML、context启动契约、守卫分支和完成控制器未实施；第5节保留其候选身份。主工作区三份未提交测量草稿冻结且排除，没有清理、提交或借用它们取得本轮结果。

### 10. 原入口性能比较方法与证据边界

**唯一主计时来自原metrics。** 每run包含step0、100、200、300、400五条原记录；主窗口100→300，后段复核200→400。两窗口各200次更新、重叠100次，不能视为独立重复。最后checkpoint400保存发生于metric400之后，未计入稳态窗口，但必须等待原异步保存完成才能接受整轮退出。

```text
t_run = (wall_time(step=300) - wall_time(step=100)) / 200
t8_ref = (t8_before + t8_after) / 2
单任务耗时增加 = (t4 / t8_ref - 1) × 100%
双四卡合计吞吐 = 128 / t4_left + 128 / t4_right
合计吞吐增加 = (t8_ref × (1/t4_left + 1/t4_right) - 1) × 100%
两份等步数任务稳态时间比 = max(t4_left,t4_right) / (2 × t8_ref)
八卡前后漂移 = (t8_after/t8_before - 1) × 100%
```

单任务慢倍数、合计吞吐和两份任务完工时间是三个不同指标。两个四卡run各自global128，相同seed和数据顺序下的合计吞吐统计训练样本处理次数，不是唯一数据条目数；不能将两run解释为一次global128更新。motion和纯512各自用本模型八卡参考，不能跨模型或跨介质套用。

**并发覆盖由原进度与外部记录核实。** 两侧metric0都早于两个主窗口的最早起点，metric400都晚于最晚终点，且物理UUID集合互斥、并集等于对应八卡集合；四份独立run的保存和启动顺序正确。该判据证明对侧处于原训练循环覆盖期间，不表示每毫秒都在执行GPU算子；真实数据等待计入墙钟时间。两个主窗口时间戳不必完全相同，合计速率是两个相互覆盖窗口速率之和，不能说同一个完全相同墙钟区间直接计数。motion的200→400窗口可能混入对侧收尾，因此不用于该并发倍率。

**这里只记录改进线索。** 原mask的批量常量前缀和编译耗时、dense attention的逻辑量、旧query分块的反向差异、梯度累积的随机数与归约顺序问题均保留为观察。第8.6节原FAIL不改判、不放宽阈值；本轮不实施mask重排、attention替换、重计算、梯度累积、精度或dataloader改造。原入口大预算的守卫阻断与实际容量未验证并列记录，不以候选优化替代本轮结果。

### 11. 已执行诊断入口与会话记录

**下面记录已经执行的诊断形态，不安排重跑或正式训练启动。** 完整命令、包版本、环境、源码与history指纹见[本轮launch](docs/training-doc/ctx-stock-scaling-20260928/launch.md)。外层入口为：

```text
BASE=v1-store/bench/context-stock-scaling-20260928
run_original.sh motion|plain 8before|4left|4right|8after GPU列表 FSDP
SOURCE/scripts/training/train.py CONFIG
  --exp-name RUN --dataset-path DATA/framesamp-8x8
  --model.history-config HISTORY
  --assets-base-dir MAIN/v1-store/train-assets
  --data.assets.assets-dir MAIN/v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da
  --data.assets.asset-id robomme
  --checkpoint-base-dir BASE/checkpoints
  --fsdp-devices FSDP --num-train-steps 401 --no-wandb-enabled
```

每次先核对快照干净、导入位置和原始配置，再启动原train.py；`PYTHONPATH`只指快照的src与openpi-client源目录，禁止产生源码字节码。所有持久路径落在本仓 `v1-store/`，uv、XDG、HF、CUDA、JAX和W&B缓存显式设置；不覆盖HOME。被测进程清除诊断遗留的XLA、精度与绕过验证开关，只启用原 `TRAIN_RECORD_DIR`指标留档，`TRAIN_TIMING_STEPS=0`。启动环境设置见launch，runtime记录实际设备/FSDP/batch/workers；前置环境仅属观察进程，不声称捕获训练进程全部生效环境。

八份run名称是 `ctx-stock-motion-8before-20260928`、`ctx-stock-motion-4left-20260928`、`ctx-stock-motion-4right-20260928`、`ctx-stock-motion-8after-20260928`，以及同样四个phase的 `ctx-stock-plain-…-20260928`。对应已起过的tmux为 `ctx-stock-motion-8before-0928`、`ctx-stock-motion-dual4-0928`、`ctx-stock-motion-8after-0928`、`ctx-stock-plain-8before-0928`、`ctx-stock-plain-dual4-0928`、`ctx-stock-plain-8after-0928`；收尾CPU核验会话为 `ctx-stock-initial-ref-0928`、`ctx-stock-data-fullhash-0928`。精确PID、时刻、退出码以留档为准，未操作本轮清单之外的用户会话。

日志使用 `PYTHONUNBUFFERED=1`、`set -o pipefail`与tee，原run有唯一 `EXIT_CODE=0`；双四卡分别保留左右原生与外层退出状态。外部NVML流式采样每500毫秒，按每run实际GPU过滤；它不修改训练进程。旧五组生产启动器、Beta依赖交接和自动完成回执仍是历史提案，不属于当前行动。

### 12. 完成验收、留档与尚未证明的事项

**本轮完成依据逐项独立。** `STOCK_PREFLIGHT=PASS`及runtime证明来源和实际配置；八份原生metrics与唯一退出0证明所记录循环及保存正常结束；每run `checkpoint-restore.json::status=PASS`证明参数实际可读、原dtype/完整schema/有限性与归一化符合检查；两份 `*-comparison.json::status=PASS`证明各自同口径基线、并发覆盖及比值；`INITIAL_REFERENCE_CHECK=PASS`证明51个共享叶中冻结与变化范围。数据内容判据及退出状态见第8.9节，不用一条笼统PASS替代这些不同结论。

本轮没有保存前现场EMA逐叶摘要，因此未声明磁盘EMA与训练现场逐位相等；checkpoint不含AdamW动量，不证明可无损续训。未来正式目标若执行，60k的12份checkpoint（5000至55000及59999）、80k的16份（5000至75000及79999）仍是原保存策略的验收口径，但当前八份checkpoint400不替代它们，也不以目录编号反推未记录的训练状态。

证据落在[本轮档案](docs/training-doc/ctx-stock-scaling-20260928/result.md)及其 `records/`，原始大日志、外层脚本与权重留在 `v1-store/`；仓库只保留Git不能还原的必要测量、恢复、比较和核验记录，不提交checkpoint、生产YAML副本或启动shell；每轮派生完整配置JSON保留在records。历史第8.1–8.7节的失败、超时和候选路径完整保留。根计划在已授权验证全部收尾后更新，不改源码、依赖、训练配置、loader或读取。

**剩余边界明确保留：** 2048/4096/8192原context守卫仍拒绝，8192现成生产配置缺失，这三档原入口四卡/八卡容量和速度未知；512+2048、4096+8192异组并发没有原入口测量。八次401步不证明完整60k/80k稳定、收敛、策略成功率或跨卡逐位轨迹；没有逐步慢步分布、全部样本梯度覆盖或完整训练工期置信区间。五组完整保存空间仍需以真实可用空间核对，不能自动删除历史产物或改保存频率。这些是本轮结束时的已知限制，不转成自动实施优化或启动训练的安排。
