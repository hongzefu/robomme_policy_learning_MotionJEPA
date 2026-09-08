# 8 帧 × 8×8 训练支持与前后对拍计划

本文记录 2026-09-08 经用户确认后落盘的实施计划。当前授权仅为编写本文件；代码修改、建库、测试和训练尚未开始，需后续实施授权。最终真实训练对拍采用每侧 1000 次参数更新，不能用短测代替。

静态审计锚点为 `e6f93466570c4df626060d75b81807528d276adc`。审计发起与结束时工作区均干净、HEAD 未变，未执行仓库脚本或训练。环境判定为 B：仓库位于 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，8 × `NVIDIA A100-SXM4-80GB`，无 turbo、无 GreatLakes。所有后续产物收敛到本仓库 `v1-store/`。本文的形状和字节量来自代码与公式推导，现有训练数字仅引用留档，不代表本次实测。

验收方法参照 [docs/dataloader-restructure.md](docs/dataloader-restructure.md) 的非训练轻量对拍与真实训练梯度对拍。本轮只新增根目录本文件，不重写该文档，也不修改其历史结论。按纯文档任务规则，本文不机械拆分为“给人看”和“技术细节”两部分。

## 1. 目标与范围

目标是在保留现有 `32 帧 × 每帧 4×4 token` 训练能力及默认配置的同时，新增独立的 `8 帧 × 每帧 8×8 token` 训练配置。两者的视觉记忆预算都是 512 个 token，单视角，图像特征宽度 2048，位置特征宽度 768。

这里的“8 帧”沿用 `shared/sampling.py::even_sampling_indices`：在截至当前帧的历史中均匀采样最多 8 帧，包含当前帧；不足 8 帧时右补零。它不是最近连续 8 帧，也不是把 `streaming_obs_horizon` 改为 8。现有 `streaming_obs_horizon=16`、`action_horizon=20` 保持。

实施范围包括：新增 8×8 packed 特征格式及打包/校验支持，令 Dataset 按明确规格校验，增加独立训练配置，补齐必要的对拍量具和本次验证入口的缓存落点。仅支持本次需要的 4×4、8×8 两档；不扩展多视角、任意网格、其他记忆表示或 `expert/modulation`。

motion 只做回归保护：复用原 motion 数据、窗口规则、预算、模型模块及训练配置，不重新抽取或训练 MotionJEPA，不研究 motion 效果，不做预算或结构消融。旧库、旧默认配置、源 pkl、归一化统计和初始化权重保留。

在线推理不在本次范围内。`policies/framesamp_memory.py::FrameSampMemory` 当前仍只构造和保存 4×4 特征；训练验收通过不等于 8×8 在线部署已支持，交付报告必须保留这一边界。

## 2. 当前支持情况与直接阻塞

| 层级 | 审计锚点下的实现 | 本次处理 |
|---|---|---|
| 源特征 | `dataset_builder/mem_buffer.py::MemoryBuffer.add_buffer` 同时生成 `image_emb_8x8/4x4/2x2` 与相应位置特征 | 先核实当前源 npy 的完整性；若齐全，直接重打包，不重抽 SigLIP |
| 独立参考装配 | 同类的 `get_frame_sampling_indices`、`_prepare_frame_sampling` 已按预算及 `sqrt(token_per_image)` 选帧、取空间键 | 固定该版本作为 8 帧参考计算依据，仅供验证，不加生产回退 |
| packed 格式 | `datastore/framesamp_store.py` 固定 `LAYOUT="framesamp-4x4-v1"`、`IMAGE_KEY="image_emb_4x4"`、`POS_KEY="pos_emb_4x4"` 以及行形状和字节数 | 保留旧格式，新增明确区分的 8×8 格式 |
| 打包工具 | `scripts/dataset/pack_framesamp_store.py::read_frame_decode` 取固定键，`read_frame_slice` 使用固定偏移，CLI 无网格参数 | 新档先走完整 decode；不能复用 4×4 的 slice 偏移 |
| Dataset | `training/framesamp_dataset.py::FrameSampDataset.__init__` 强制 `(budget, token_per_image, num_views)==(512,16,1)` | 接受两档白名单，并校验配置与实际 store 完全匹配 |
| 模型与批处理 | `HistoryPi0Config.inputs_spec`、`PerceptualMemory.__call__`、`FeatureEncoder` 主要依赖预算和通道宽度；`_collate_fn` 逐键 stack | 不新增模型参数、不改变模型计算式；实际入口形状与精度由对拍确认 |
| 验证入口 | `bench_train_steps.py` 等存在配置名白名单、固定键数、旧 dtype 例外 | 增加本次明确的验收规则，不能直接套旧通过条件 |

因此，当前仅把 `token_per_image` 改成 64 会在 Dataset 构造时失败；只放宽断言也不够，旧 store 每帧只有 16 个 token，无法交付目标的 64 个。

现有环境 B 留档 [awsprod40k-b128-motion](docs/training-doc/awsprod40k-b128-motion/result.md) 证明旧配置曾完成 8 卡训练，不能作为 8 帧配置已跑通的证据。两档模型输入总长相同也只支持“模型侧规模原则上不变”的静态判断，不能据此保证显存余量、吞吐或训练效果。

## 3. 配置与数据契约

| 验收档位 | 视觉帧上限 | `budget` | `token_per_image` | `num_views` | `motion.enabled` |
|---|---:|---:|---:|---:|---|
| C32 | 32 | 512 | 16 | 1 | false |
| M32 | 32 | 512 | 16 | 1 | true |
| C8 | 8 | 512 | 64 | 1 | false |
| M8 | 8 | 512 | 64 | 1 | true |

保留 `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-context.yaml` 和 `perceptual-framesamp-context-motion.yaml`。拟新增 `perceptual-framesamp-context-8frame-8x8.yaml` 与 `perceptual-framesamp-context-8frame-8x8-motion.yaml`；相对各自旧配置，仅改变本次视觉规格及必要的显式数据路径。配置文件名是拟议接口，实施时应与量具白名单一次性同步。

学习率、loss 权重、优化器、训练精度等训练语义不随本次重构修改。1000 步是本次验收要求；建议的 batch、worker、GPU 档位写在第 8 节，仅作为本次启动覆盖参数，不修改全局默认值，实际运行前确认。

建议在 `framesamp_store.py` 内建立唯一的规格定义，以 `framesamp-4x4-v1` 和拟新增 `framesamp-8x8-v1` 映射到网格边长、token 数、键名、文件名、行 shape、dtype、字节数。`StoreMeta.load`、reader、packer、verify 共同使用该定义；若 meta 同时记录显式规格字段，字段必须与布局相符。不得把 8×8 数据伪装成旧布局，也不得放过配置与库规格不一致。

旧 meta 无需迁移即可读取。以下检查继续生效：verified 状态、pack.lock、manifest 现场摘要、part 大小与边界、源库身份、pkl 内 episode/step 互校、motion 双 store 同源。与新规格不兼容时直接报错，不静默取前 16 个 token、不插值补足、不回退散文件。

候选主数据口径为已留档的四任务 400 episode 库，开发口径为四任务 40 episode 库；实际是否存在、源 npy 是否保留，均在实施 preflight 核实。建议新 packed 根分别为 `v1-store/datasets/4task-motion-400ep/framesamp-8x8/` 和对应 40 episode 目录。源 pkl、manifest、motion store 通过显式路径复用，不复制大份源库，不新建外链，不覆盖旧 `framesamp/`。

## 4. 前后链路图与字节账

下面两图分别描述现有 32 帧链和目标 8 帧链。它们之间的选帧及空间特征变化是本次明确目标，不能把两图末端数值相等作为验收要求。等价关系是第 5 节规定的“同一配置内参考链与候选链”。

### 4.1 重构前：32 帧 × 4×4

```text
源库 features/episode_g/token_emb_t.npy + data/idx.pkl + episode_manifest.json
  image_emb_4x4 (1,16,2048) bf16，65,536 B/帧
  pos_emb_4x4   (1,16,768)  f32，49,152 B/帧；state_emb (8,) f32，32 B/帧
  │ pack：完整解码、抽取指定键、按 episode/帧映射写裸表；不改数
  ▼
旧 framesamp-4x4-v1
  image 行 (16,2048) bf16；pos 行 (16,768) f32；state 行 (8,) f32
  │ even_sampling_indices(step,32) → row_of/gather；选取历史，不改选中元素
  ▼
FrameSampDataset：最多 32 帧 → 同 dtype 右补零 → reshape/repeat
  static_image_emb (512,2048) bf16，2,097,152 B
  static_pos_emb   (512,768)  f32，1,572,864 B
  static_state_emb (512,8)    f64，32,768 B（现有 quantile 统计口径）
  static_mask      (512,)    bool，512 B
  │ 补零/复制/重排不改有效值；state 经现有归一化公式改数，保留该行为
  ▼
现有 transforms → collate：static/motion 字段透传后 stack，添加 batch 维 B
  static 字段 dtype 不变，字节数为上述各项 × B；其它字段保留既有变换
  │ make_array_from_process_local_data → HistAugObservation
  ▼
JAX / 模型入口：图像 bf16、位置 f32、mask bool，形状分别为 (B,512,D)/(B,512)
  x64 关闭时 state 为 f32：16,384 × B B；开启时 f64：32,768 × B B
  dtype 收窄属于既有交付行为，按实测入口记录，前后必须相同
  → PerceptualMemory / HistoryPi0：继续现有投影、位置融合及模型计算
```

### 4.2 重构后：8 帧 × 8×8

```text
同一源库、同一 manifest、同一 data/idx.pkl
  image_emb_8x8 (1,64,2048) bf16，262,144 B/帧
  pos_emb_8x8   (1,64,768)  f32，196,608 B/帧；state_emb (8,) f32，32 B/帧
  │ 新 pack：从源 8×8 键完整解码、按原身份映射写独立裸表；不改数
  ▼
新 framesamp-8x8-v1
  image 行 (64,2048) bf16；pos 行 (64,768) f32；state 行 (8,) f32
  │ even_sampling_indices(step,8) → 同一行号规则/gather；不改选中元素
  ▼
FrameSampDataset：最多 8 帧 → 同 dtype 右补零 → reshape/repeat
  image (8,64,2048) → (512,2048) bf16，2,097,152 B
  pos   (8,64,768)  → (512,768)  f32，1,572,864 B
  state (8,8) → repeat 64 → (512,8)，同式归一化后 f64，32,768 B
  mask  (8,)  → repeat 64 → (512,) bool，512 B
  │ 与独立 8 帧参考链的有效值、填充位、dtype 必须完全一致
  ▼
同一 transforms → 同一 collate → 同一 JAX 交付 → 同一模型
  各跳形状、dtype、字节量沿用上图；每批字节数乘 B
  仅授权选帧数量和所取空间特征变化，其他变换与数值规则不变
```

图中的 state dtype 描述按当前 quantile 归一化代码及 JAX x64 开关给出。实施时必须分别抓取 Dataset、transform、collate 和 JAX 入口的实际 dtype，不能仅凭 `inputs_spec` 或静态图填写结果。原图、actions、prompt/tokenizer、当前 state 等旁路也必须逐键记录；它们经过原有变换可能改数，但同配置两侧必须一致。

motion 旁路在两图中复用同一来源：`motion store → visible_motion_rows(entry, step) → rows → 同 dtype padding → memory_order`。开启态每样本为 `motion_emb (96,768) f32 = 294,912 B`、`motion_pos (96,256) f32 = 98,304 B`、`motion_mask (96,) bool = 96 B`、`mem_order (608,) int32 = 2,432 B`；collate 后各乘 B，关闭态四键均为 None。读取、切片和有效值重排不改数，排序位置的跨视觉配置变化单列在第 6 节。

满长单样本图像读取仍为 2 MiB，位置特征仍为 1.5 MiB；但全帧 image 库及每 worker 的完整 pos 表扩大 4 倍。沿用留档的 123,044 帧、586 个位置行作公式估算，新 image 裸表为 `32,255,246,336 B`，pos 表为 `115,212,288 B`，state 表为 `3,937,408 B`，不含 meta、摘要及临时文件。这不是当前磁盘实测，也不代表初期短样本的 I/O 与旧配置相同。

pack 的空间预检应按实际行数、规格、临时文件和新旧库共存计算；`cmd_pack` 现有固定 40 GB 余量不能充当通用保证。运行任何指定输出根的命令前，先 `ls -ld <输出根>`，确认实体目录及位置，再按实际可用空间决定能否开始。

## 5. 两组等价关系与独立参考链

| 档位 | 参考 A | 候选 B | 必须成立的关系 |
|---|---|---|---|
| C32 / M32 | 重构前正式 4×4 packed 链 | 重构后保留的 4×4 链 | 同配置输入和训练轨迹逐位一致 |
| C8 / M8 | 冻结的独立源 npy 读取与 8 帧装配 | 新 8×8 packed 链 | 同配置输入和训练轨迹逐位一致 |

不存在“重构前正式 Dataset 已经能跑 8 帧”的前提。8 帧参考实现仅服务验证，不能作为生产 fallback。参考链直接读源 npy 的 `image_emb_8x8/pos_emb_8x8/state_emb`，沿用锚定版本的 `MemoryBuffer._prepare_frame_sampling` 及其明确的采样、padding 语义，并独立完成 pkl 身份与输出装配。

当前 `dataset_builder/data_utils.py::right_padding_token_emb` 三处补零均显式采用输入 dtype，因此参考链可交付 image bf16、pos f32、原始 state f32。state 归一化必须固定为审计锚点 `FrameSampDataset._normalize_state` 的同式运算，不能调用候选实现来定义参考答案；motion 装配也以冻结的现有语义为准。量具不能通过替换候选 reader 而同时改变两侧答案。

参考装配、量具及缓存路径修复先形成可复现提交，确认未改变现有数据/模型语义，再冻结参考 commit。候选格式与 reader 改动后另记候选 commit。两者都保留原始审计锚点的来源关系；不得复活已弃用的 `d951aef` 脚本，也不使用环境 A 的固化梯度或标量作为本次基线。

## 6. motion 一致性的精确定义

同一个 profile 内，参考与候选的 `motion_emb`、`motion_pos`、`motion_mask`、`mem_order` 四键必须逐位一致，包括有效元素、填充元素、dtype、shape 和 None 状态。M32 与 M8 各自完成该检查；C32 与 C8 均确认四键保持 None。

跨 32 帧与 8 帧配置，在相同 episode、step 下，motion 起点集合、行号、窗口可见性、前三个交付键必须不变。`motion_pos` 实际来自 `store.pos_rows(f_m)[:,0,:256]`，不能因 motion 源码未改就省略验证：应对全部有效时刻核对源 4×4 与 8×8 位置特征该切片的 dtype、shape 和 raw 字节完全一致。

`mem_order` 不要求跨视觉配置相等。`shared/sampling.py::memory_order` 用帧时刻、每帧 token 数及 motion 时刻生成置换；从 32×16 改为 8×64 后，视觉 token 的时间布局改变，motion 在混合序列中的位置可能随之改变。必须保持按时刻稳定排序、同刻视觉在 motion 前、motion 内部相对顺序、有效位与 padding 的处理规则；不得强行复用旧排列，也不得据此要求跨配置的全模型 loss 或 motion 模块梯度相等。

motion store 内容与 provenance、`visible_motion_rows`、预算 96、stride 16、窗口 33 帧、前视窗口边界、encoder/projection 模块及创建顺序均保持。只补接口和训练回归量具，不扩展整套 motion 内部机理测试。

## 7. 第一块验收：非训练轻量对拍

所有比较先核完整键集和结构，再核 dtype、shape、raw 字节；字符串和 None 单独核对。禁止仅比较双方键的交集。预期判据为零差异，不继承旧 dtype 重构曾允许的“四个 batch raw 失配”，也不通过统一转 f32 掩盖本轮差异。

| 检查 | 覆盖与方法 | 通过条件 |
|---|---|---|
| 数据身份 | 所有 episode、样本长度、`exec_start_idx`、源 pkl 的 episode/step 与清单映射 | 全部一致；非零演示长度和 part 边界均覆盖 |
| 选帧 | 对全部合法 step 比较参考与候选索引；8 帧重点取 `0,1,2,5,6,7,8,9`，旧档取 `29,30,31,32,33` | 列表顺序及元素完全一致 |
| 打包写读 | 完整 decode 每一源帧，经候选真实读 API 比 image、pos、state；核 pos 去重跨 episode 同值 | `scanned == num_rows`，三键失配数均为 0 |
| 定点样本 | 每集首尾、演示/执行交界、padding 临界、part 首尾及固定 seed 随机样本 | Dataset 与 transform 后所有字段零差异 |
| 真实 batch | 每个 profile 至少 200 个经真实 transforms 和 collate 的 batch，含全短、全满、混合、随机组合 | 所有字段、结构、mask 与 raw 字节一致 |
| 抽样顺序 | 同 seed、同 worker 档，记录真实 loader 实际消费的 index；不把预取尾部计入 | 前 `1000 × batch_size` 个消费 index 及次序一致 |
| worker 生命周期 | 40 episode 开发库上 w0/w1/w4/w16 各跑 2 个 epoch，两侧逐档比较 | 无重复/遗漏导致的差异，文件句柄关闭后回到基线 |
| 拒绝错误输入 | 错网格、错大小、未知布局、未 verified、pack.lock、错清单、错 motion 来源、损坏源身份 | 必须明确失败，不能回退或静默截断 |

源 npy 与新 packed 的全量验收以完整数据口径执行，抽样只作开发快检。开发小库应是完整独立 manifest 的 40 episode 库，不能用会绕过正式闸门的 subset 迷你库代替最终验收。`dump_index_seq.py` 使用的 ProbeDataset 只能作抽样器旁证，不能替代真实新旧链的内容与消费顺序检查。

跨 epoch 的 worker 行为按同 worker 数比较，不要求 w0 与 w4 的结果彼此相同。超过 5 分钟的非训练扫描或生命周期检查，同样按完整诊断运行留档、从 clean HEAD 启动并进入 detached tmux。

## 8. 第二块验收：同机真实训练 1000 次更新

四个 profile 各做参考自重复 A1/A2，再做候选 B。每条轨迹独立初始化、独立空编译缓存、完成 1000 次更新：共 12 条轨迹、12,000 次更新。每个 profile 先通过 A1=A2，才判断 A1=B。不能在基线自身不重复时放宽阈值或宣称重构等价。

建议档位为 global batch 8、seed 42、4 worker、固定同一对 A100、同一设备顺序和 mesh；这些是本次启动覆盖参数，运行前确认，不修改全局训练默认值。具体 GPU、FSDP/mesh 参数、全新 run_name 及预计耗时在实施 launch 中确定。四个 profile 各自内部保持语义配置一致，不跨 motion 开关比较参数叶数或训练轨迹。

确定性参数包含 `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`，与固定 JAX 精度、随机种子及依赖一起写入环境指纹。A1、A2、B 使用相同实际 GPU 集合顺序运行，不用不同卡组并行对拍。100 步可用于提前排错和估时，不能作为 1000 步通过证据。

### 8.1 步号与完整状态

“1000 步”明确指 1000 次参数更新：训练 metrics 的循环编号恰为 `0..999`，无重复、无缺步，终态 `TrainState.step == 1000`。每步比较 `loss`、`grad_norm`、`llm_grad_norm`、`mem_enc_norm`、`param_norm` 的浮点 hex；逐步记录实际消费输入的完整结构、dtype、shape、raw 摘要及 index。

现有 `_install_step0_checksum` 的 `param_checksums.step=0` 是尚未更新的初态；其他保存标签 `s` 是循环第 s 次更新之后，已完成 `s+1` 次更新。因此它与 `metrics.step=0` 并非同一时刻。量具需显式记录 `phase=init/post_update`、`loop_step` 和实际 `state_step`，不能只按一个 step 字段连接两份记录。

完整状态摘要建议固定为更新次数 `state_step ∈ {0,1,2,25,50,100,200,400,600,800,1000}`：0 是初态，正整数 k 对应 `loop_step=k-1`。若现有保存钩子无法覆盖，先补钩子再冻结基线，不用最近一次摘要代替缺失点。

每个摘要点比较全部参数（含冻结参数）、optimizer 状态与计数器、EMA 和 step 的完整树路径、dtype、shape、raw 哈希。EMA 是否启用也是契约；关闭时明确记录，不伪造空摘要。不得硬编码 177 个叶或 12 个输入键，也不得取两个树的交集。

补充全短、全满、混合三种真实 batch 的单步完整梯度逐叶对拍，定位可能被梯度范数掩盖的差异；同样要求同配置逐位相等。完整训练第二块未通过，只能报告已通过的输入检查，不能宣称训练等价。

### 8.2 环境指纹与允许差异

硬一致项包括实际 GPU 集合及顺序、驱动、软件依赖与 `uv.lock`、JAX/XLA 配置、精度与 x64 开关、seed、batch、worker、mesh、训练超参、模型初始化权重、tokenizer、norm stats、样本清单及源数据身份。resolved 配置与原配置都留档，不忽略整个 YAML 或全部 argv。

A1/A2 的参考 commit 必须相同，B 记录候选 commit。代码提交不同是必要来源信息，不能误当环境失败。可不同项仅限明确列出的 run_name、日志/输出/独立缓存目录、参考与候选 commit，以及本方案规定的读取后端和派生库路径。任何路径归一化均使用逐字段白名单。

源文件参考与 packed 的布局、store_meta 和物理文件哈希自然不同：分别留存，通过第一块全量写读建立联系，不能要求两份 meta 整体 sha 相等，也不能把存储来源从指纹里删除。motion store/index/provenance、库内 manifest 和 norm stats 必须纳入，不能沿用只看根 `v1-store/episode_manifest.json` 的旧指纹。

## 9. 拟修改文件与量具边界

下表是后续实施范围，不表示本次已经改动。新增参考、fixture 和比较器统一放在 `scripts/training/tests/`，采用本次专用且配置驱动的接口；生产 loader 始终只读 verified packed 库。

| 文件或稳定锚点 | 必要工作 |
|---|---|
| `src/mme_vla_suite/datastore/framesamp_store.py`，必要时同包 `__init__.py` | 两档规格定义、StoreMeta 校验、行大小/键/文件名解析，保留旧布局兼容 |
| `scripts/dataset/pack_framesamp_store.py` | 显式网格选择、decode 打包、独立输出、全量 verify、按规格计算空间需求 |
| `src/mme_vla_suite/training/framesamp_dataset.py::FrameSampDataset` | 配置白名单及 store 匹配，动态帧数/每帧 token 装配；不改采样、归一化与 motion 规则 |
| `src/mme_vla_suite/training/dataloader.py::_create_framesamp_dataset` | 仅在规格与元数据传递确有需要时调整，保留全部 fail-loud 闸 |
| 两个新增 `perceptual-framesamp-context-8frame-8x8*.yaml` | 独立目标配置，旧默认不变 |
| `scripts/training/tests/_common.py`、`dump_fixture_samples.py` | 显式配置/manifest、按帧预算生成边界样本，配额适配 40/400 episode 库 |
| `scripts/training/tests/test_pack_guards.py`、`spawn_matrix.py` | 两档规格、motion 四键、真实生命周期及错配拒绝检查 |
| `scripts/training/g0/bench_train_steps.py` | 配置白名单、完整结构含 None 的记录、精确更新次数摘要；保留已有逐树记录能力 |
| `scripts/training/g0/check_baseline_env.py` 与本次比较器 | 库内来源指纹、明确允许差异、完整步集/键集/树集及 raw 零差异检查 |
| `scripts/training/train.py::main`、本次单步/启动入口 | 让实际生效的 JAX/uv 等缓存落在 `v1-store/cache/`，不改训练数值逻辑 |

`g0_gate.py` 的旧 T1/T2、`compare_dtype_fix.py` 服务历史验证，不应为本次随意放宽。可以新增明确的本次 profile 或独立严格比较器，但不能继承“恰有 4 个 raw 失配”、固定叶数、仅允许 `motion:false` 配置差等历史条件。`motion_gates_model.py` 的全套内部机理门不在本次泛化范围内。

## 10. 环境 B 的执行与留档要求

现有 `train.py::main`、`single_step_grad.py::main` 会设置 `~/.cache/jax_<exp_name>`，旧 `run_2gpu_epoch_bench.sh` 还会创建 HOME 缓存及软链。只导出 `XDG_CACHE_HOME` 无法覆盖这些硬编码；实施时必须先修正本次实际入口，再验证最终生效路径。不能照抄旧 runner，也不修改或覆盖 HOME。

所有缓存逐项显式设置到本仓库 `v1-store/cache/`，包括 `UV_CACHE_DIR`、`XDG_CACHE_HOME`、`HF_HOME`、JAX 编译缓存及需要的 W&B 目录。执行 Python 前确认 uv 管理方式，使用 `uv run`；若确需正式依赖变更，须同时更新 `pyproject.toml` 与 `uv.lock`，本计划不预设新增依赖。

实施 preflight 先确认实际源库、manifest、8×8 源键、motion 数据、初始化权重和 tokenizer 可用，以及输出根为实体目录。若缺失源特征，停止在该项并与用户确认数据取得或重建方案，不能自行开始原始 H5 大下载或重抽取。禁止访问 turbo、ssh 集群、Slurm 或 `gl_submit.py`。

所有预计超过 5 分钟的建库、取证、基准、诊断、训练均从 clean HEAD 启动，放入 detached tmux。命令使用 `PYTHONUNBUFFERED=1`、`set -o pipefail` 和 `tee`，结束写 `EXIT_CODE=`。正式建库留档于 `docs/dataset-build-doc/<dataset_name>/`；训练及完整诊断留档于 `docs/training-doc/<run_name>/{launch.md,result.md,records/}`，记录 commit、精确命令、配置、数据和输出路径。

启动前确认新的 run_name，绝不覆盖旧 run。具体命令在入口适配完成后以真实 CLI 写入 launch，不把尚不存在的新参数当作可执行命令。短 smoke 可自行命名，验证后按本轮清单清理；长取证的必要证据按留档规则保留，不归档大权重。

tmux 会话名使用可辨识前缀并逐个留档。禁止任何全局清理；只可对本轮明确登记的完整会话名执行 `tmux kill-session -t <确切会话名>`，删除前后分别 `tmux ls` 核对恰少该会话。日志监测管道各级行缓冲，存活检查使用 `tmux has-session`。

吞吐与显存检查在数值验收之外独立报告，存储介质写明“AWS 本地 NVMe RAID（`/dev/md0`）”，记录 batch、worker、warmup、稳态窗口及并行负载。使用 `nvidia-smi -lms 500` 的稳态 util 均值、0% 占比、慢步/非慢步分层均值；不以中位数宣称 GPU 吃满，不把 JAX 预分配池当真实显存需求。不承诺性能收益，不与环境 A 数字混比，不自动扩展 worker/batch 大扫描。

## 11. 实施顺序与交付判定

1. **确认实施授权与资源。** 确认库口径和实际文件、验证档位、GPU/mesh、新 run_name；1000 更新为已确定最终口径。缺数据时先处理来源决策，不开始重构后的正式运行。
2. **固定量具和参考。** 完成本次参考适配、缓存落点与严格比较器，验证不改变旧语义，提交并推送；在 clean HEAD 冻结参考 commit。先检查量具能拒绝缺键、缺步、重复步和人为扰动，防止“两侧漏看同一字段”假通过。
3. **实现两档格式与读取。** 保留旧 4×4，新增独立 8×8；每次代码改动先跑覆盖核心路径的最小真实验证，目标每轮约 5 分钟内，长项按完整运行另记。分段提交时保留可复现来源，不动其他人在途工作。
4. **建库并完成第一块。** 先在开发库检查，再对最终口径完整 pack/verify；完成身份、选帧、样本、batch、worker 及 motion 接口的逐位对拍，产出按实测更新的前后链路图。
5. **完成第二块。** 在固定 GPU 上按各 profile 的 A1/A2/B 顺序运行和比较，所有轨迹从 clean HEAD 启动。必要时先 100 步排错，再执行完整 1000 更新；任何失败保留首次差异和原始证据，不改阈值过关。
6. **交付结论。** 分开报告旧能力回归、新 8 帧训练支持、motion 接口保护、实际资源表现及在线未覆盖项；只有两块全部通过，才宣称对应 profile 的训练交付等价。

最终验收记录至少明确以下判定及实际扫描数量，不允许以一条笼统 PASS 代替：`SOURCE_IDENTITY`、`FRAME_INDEX_EXACT`、`PACK_VERIFY_ALL`、`SAMPLE_RAW_EXACT`、`BATCH_RAW_EXACT`、`MOTION_INTERFACE_EXACT`、`BASELINE_REPEAT_EXACT`、`TRAIN_1000_EXACT`、`FINAL_STATE_EXACT`。这些是拟议报告项，当前没有任何实测 PASS。

## 12. 本次文档验证与提交

本轮文件范围仅为根目录新增 `8frame-8x8-training-plan.md`，从标题到本节全部新建，不替换其他文档段落。正文引用代码使用函数、类、配置键等稳定锚点，不写硬编码行号；前后链路图、公式字节账和验收步骤按静态审计核对。

落盘后运行 Markdown 结构检查与 `git diff --check`，核对新增文件范围、链接目标、代码锚点、代码围栏、无尾随空白和末尾换行。本文只定义未来实施，不为纯文档改动执行仓库脚本、uv、建库或训练测试。

提交前运行 `git status --short`，仅逐文件暂存本文件，检查 staged diff 与空白。计划提交主题为 `docs: 增加8帧8×8训练支持与对拍计划`；commit 完成后立即向当前分支既有 upstream 执行 `git push`，不创建远端分支、不强推。推送完成后用 `git status -sb` 确认无 ahead。若推送失败，保留原始错误交用户处理，不重试、不改写历史。
