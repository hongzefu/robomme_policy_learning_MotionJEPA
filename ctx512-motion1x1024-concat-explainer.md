# 给 VLA 加 motion memory：怎么拼、怎么建库、窗口怎么补帧

> 写给要把同一套 motion memory 搬到别的 VLA（例如 Qwen 系）上的人。只讲机制，不讲本仓库的工程细节。
> 参考实现：本仓库在 pi0.5 上的 `context 512 + motion 1×1024（concat）` 模型，AWS 8×A100 训练，官方 800 局三 seed 37.33 ± 0.59%。

## 一、总体思路

VLA 原本只看当前帧 + 文本。我们给它加一段「记忆」，放在主干（这里是 PaliGemma）输入序列的最前面，让语言 token 和动作 token 能回看历史。记忆由两路 token 组成：

| 路 | 一个 token 代表什么 | 怎么得到 | 每样本多少个 | 原始维度 |
|---|---|---|---|---|
| 帧路（context） | 某一历史帧「长什么样」 | 历史帧过 VLA 自带的视觉编码器（SigLIP），再池化到 4×4 = 16 个 token | 等距取 32 帧 × 16 = 512 | 2048 |
| 运动路（motion） | 某段 33 帧「发生了什么运动」 | 33 帧 → Wan2.1 VAE → MotionJEPA 阶段二 encoder，输出 (4,256) 展平成一个 1024 维向量 | 最多 96 个窗口，每窗 1 个 | 1024 |

两路各自接一个小的线性投影，投到主干宽度（2048），然后**按时间交错**拼成一段 608 位的记忆区。两个 encoder 都冻结，训练只更新 VLA 本体和这两个投影。

## 二、memory 怎么拼接

```
[ 记忆区 608 位（帧 token 与 motion token 按时刻交错） | 当前帧图像 512 | 文本 64 ]  →  主干  →  动作专家 20 步
```

1. **投影**。帧 token：`concat(视觉特征 2048, silu(Linear(时间码 768→768)))` → `Linear(2816→2048)`。motion token：`concat(motion 1024, silu(Linear(时间码 256→768)))` → `Linear(1792→2048)`。时间码就是该帧 / 该窗口起点帧的位置编码，让主干知道「这是什么时候的记忆」。
2. **交错**。每个 token 带一个「时刻」（帧号或窗口起点帧号），两路合在一起按时刻稳定排序：同一时刻帧 token 排在 motion token 前。效果是时间上相邻的「样子」和「运动」靠在一起，而不是 512 个帧 token 后面跟 96 个 motion token。
3. **固定形状 + mask**。为了 jit 形状固定，帧路填满 512 位、运动路填满 96 位；不够的位置填零向量并把 mask 置 False。排序时 padding 的时刻记成最大值，所以 padding 全部落在记忆区尾部。位置编号按有效 token 累加，padding 不占位置号。
4. **attention 规则**。记忆 token 之间双向互看，看不到当前图像和文本；当前图像 token **看不到记忆**（不让历史污染当前视觉流）；文本 token 和动作 token 能看到全部记忆。记忆区只在 prefix 里算一次并进 KV cache，动作去噪的多步只重算动作那一段。
5. **预算与截断**。motion 位置数 96 按全部数据集定标，保证最长 episode 也不截断；合法窗口数超过 96 直接报错，不抽稀。

搬到 Qwen 时需要对应改的只有三处：投影输出维度换成 Qwen 隐宽；记忆区插在 Qwen 的视觉 token 之前；attention mask 按上面第 4 条写。

## 三、motion token 的两种获取方式

同一个 encoder、同一份代码，但训练和推理两侧拿 token 的方式不同：

| | 训练：离线建库预存 | 推理：在线现算 |
|---|---|---|
| 做什么 | 对整个训练集提前算好所有窗口的 motion token，存成一张表；帧路同理存一张 SigLIP 特征表 | 评估时 policy 进程旁边起一个 encoder 子进程，环境每送来新帧就把凑齐的窗口现场编码 |
| 输入 | 原始 H5 视频帧（只在建库这一步读 H5） | 环境实时帧（256×256 原图） |
| 存储 | 每窗 1024 维 f32，整个 1600 集训练库只有 23k 行、约 95 MB；中间产物 Wan latent 另存可复用 | 不落盘，按 episode 缓存在内存里 |
| 流程 | 帧 → VAE latent → encoder token → 打包成表（三步各自可断点续跑、可逐位对拍） | 首批送整段 demo，之后每 16 帧送一批；每批到后用 while 把已凑齐 33 帧的起点全部编掉，一个起点只编一次 |
| 一致性 | 两侧 encoder 入口是同一个函数（fp32、关 TF32、batch 1），启动时核对权重哈希与数值环境，同一窗口离线表与在线现算逐位相同 | |

训练不用在线算是因为太慢（一个窗口 VAE 约 1 s）；推理不用预存表是因为推理时的帧是新的。所以必须保证两侧算出来的东西一样，核心就是「同一函数、同一权重、同一窗口定义」。

## 四、窗口怎么取、怎么补帧

窗口 = 从起点往后连续 33 帧。起点钉在**段内绝对网格** 0, 32, 64, …（demo 段与 exec 段各自从 0 起算，窗口不跨段），当前时刻只决定哪些起点「已经可见」。这样每过 32 帧才多一个窗口，离线表小，在线每隔两个 action chunk 才编一次。

补帧规则（训练建库与在线现算完全一致）：

- **demo 段**：尾部窗口不足 33 帧时，用 **demo 最后一帧重复补到 33 帧**；要求至少 17 个真实帧，不足 17 的起点不算窗口；**绝不借用 exec 的帧**，避免 demo 窗口泄露执行阶段信息。
- **exec 段**：**永不补帧**，必须凑齐 33 个真实帧才算可见；没凑齐就等后面的帧到了再编。

换一个 encoder 或窗口长度时，这条规则和 stride 要同时写进离线建库脚本与在线编码器，否则不会报错，只会静默地训练与推理不一致。

## 五、参考代码位置（本仓库）

| 内容 | 位置 |
|---|---|
| 记忆拼接、attention mask | `src/mme_vla_suite/models/integration/history_pi0.py::HistoryPi0.embed_prefix / embed_memory / make_attn_mask` |
| 两路投影 | `src/mme_vla_suite/models/representation/percep_mem.py::PerceptualMemory` |
| 交错排序 | `src/mme_vla_suite/shared/sampling.py::memory_order` |
| 窗口网格与补帧契约 | `src/mme_vla_suite/datastore/motion_store.py::visible_motion_rows / LAYOUT_SPECS` |
| 离线建库 | `scripts/dataset/wan/extract_wan.py`（帧→latent）、MotionJEPA `stage2_memory_export.py`（latent→token）、`scripts/dataset/pack_motion_store.py`（打包） |
| 在线现算 | `scripts/dataset/wan_stage2/stage2_sidecar.py`（encoder 子进程）、`src/mme_vla_suite/policies/framesamp_memory.py::FrameSampMemory`（窗口状态机与补帧） |
| encoder 本体 | MotionJEPA `src/motion_jepa/models/wan_latent_encoder.py::Stage2WanLatentMotionEncoder`，导出入口 `src/motion_jepa/export/stage2_memory.py::encode_windows` |
