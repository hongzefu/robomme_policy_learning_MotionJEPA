# v6 计划：motion memory 接入——framesample 记忆双路化（帧路 + 运动路）

> **本文件是 v6 工作的权威计划**（2026-09-01；只陈述当前定稿设计，历次修订见 git log）。
> **锚点**：分支 `v1-dataloader-Restructure`，HEAD = `4503ea2`（工作区 clean）。
> **commit 编号**：代码切片 **commitV6.x**；本文件本身按 `docs:` 提交。
> **外部依赖仓库**：MotionJEPA 单副本 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA`
> （HEAD 与 checkpoint 选型见第二部分〇节，**起工前须锚定并写死**）。
> **本计划只规划、不实施**：S0–S4 每步须逐步获批后动手。

---

# 第一部分（给人看）

## 一、Context 与方案总览

`AGENTS.md` 的项目 scope 写明「仓库总体目标：修改 MME-VLA 的 `perceptual-framesamp-context`，
并在后续阶段接入 MotionJEPA motion token」。v1–v5 已把 dataloader 与训练入口收敛到 packed
framesamp 单一路径（v5.2 收官，60k 全量 run 在跑），本计划是那个「后续阶段」。

当前记忆只有**一路**：`even_sampling_indices(step, 32)` 在 `[0, t]` 上变长间隔地选 32 个历史帧，
每帧给 16 个 4×4 池化的 SigLIP token，共 512 个 memory token。这一路描述的是**静态外观**
（那一帧长什么样），不描述**运动**（那一段时间里在发生什么）。v6 并联第二路。

**一句话方案**：memory 从「512 个外观 token」变成「512 个外观 token **并列** 80 个运动 token」，
prefix 记忆区 512 → **592**。运动特征来自 MotionJEPA 的两级链路：Wan VAE（离线冻结）→
`WanLatentMotionEncoder`；接入形态按用户拍板「**作为 memory 的一部分**」——记忆序列的
第二路，不是插单个 token 进 prefix。帧路照旧——用户明确「**逻辑不变，你不用管**」，
`even_sampling_indices` 一字不动、变长间隔铺满全历史；运动路与帧路**完全独立**：按段内
绝对网格每 20 帧取一个起点、每个起点往后 33 帧编一个运动向量、窗口尾端不得越过当前帧。
训练读离线表，在线评估每 20 帧增量现编一次。

五个已定死的口径（依据分别在 2.2、2.1、2.3–2.4、3.5、3.4）：

1. **段内绝对网格**（起点 = 段起点 + 20m）。
2. **前视窗口 + 尾端 ≤ 当前帧**（起点往后 33 帧）。
3. **预算 N=80，零截断**；容量按 16 任务全集 `/data/hongzefu/robomme_data_h5` 定标（全集最大需 69），
   **不按当前 4 任务训练集定标**；代价是平均填充率仅 4env 10.1% / 16env 19.1%。
4. **并列拼接**（512 + 80）。
5. **缺失走 `motion_mask`**（与 `static_mask` 同款）。

下文按 窗口（二节）→ 链路（三节）→ 对齐（四节）展开，再给实施步骤（五节）与影响面（六节）。

## 二、窗口：运动路怎么采样

### 2.1 窗口定义：前视 33 帧，尾端不越当前帧

一个运动窗口 = 从起点 `f` 往后连续 33 帧 `[f, f+32]`，经 Wan VAE + `WanLatentMotionEncoder`
编成一个 768 维 motion token。**前视**方向与 encoder 的训练语义完全一致（motion 描述
「相对锚点 z0 之后发生的运动」）。训练时窗口必须整体位于已发生的历史内，约束为
**尾端 ≤ 当前帧**（`f + 32 ≤ t`），出自用户原话「训练时候 f+32 窗口必须小于等于当前帧」。

起点的语义出自用户原话「以 VLA 训练时每个 action chunk 的开始作为 f」
「间隔一个 action chunk 抽取一次」——这句话如何落成具体的起点集合，见 2.2。

### 2.2 起点集合：段内绝对网格

起点**钉死在段内绝对位置** `0, 20, 40, …`（= 段起点 + 20m），不随当前帧平移；当前帧 `t`
只决定网格上哪些起点「已经可见」。间隔 20 走**独立配置键 `motion.stride`**——默认值写 20
（= 当前 `action_horizon`），但**不自动跟随**该超参。

设当前样本的全 timestep 域帧号为 `t`，该 episode 的 `exec_start_idx = es`，起点集合为：

```
exec 段网格： u = 0, 20, 40, …        （u 是 exec 段内偏移）
              合法条件： u + 32 ≤ t − es      且   u < num_chunks_exec

demo 段网格： s = 0, 20, 40, …        （s 是 demo 段内偏移；demo 段整段已见）
              合法条件： s + 32 ≤ es − 1      且   s < num_chunks_demo
```

（`num_chunks_*` 是 MotionJEPA 侧每段的合法窗口起点数，实测口径见 4.1；起点如何换算成
latent 库的 chunk 行号，见 4.2。）

⚠ **demo / exec 两段各自成网格、互不延续，窗口一律不跨 demo/exec 边界**——用户拍板
「独立的，也是一样采样不到 33 窗口都不补」（latent 分段抽，跨界窗口不存在）。
⚠ 起点集合与 `even_sampling_indices` 选出的 32 个帧号**没有任何对齐关系**——两路各采各的。

训练样本的当前帧 `t` 是**逐帧 dense** 的，多数不落在 20 的倍数上；网格不随 `t` 平移，
`t` 只把「可见边界」往右推——已经算过的窗口永远有效。下面用段起点 = 0、当前帧从
**205 走到 206** 的实际数轴演示：

```
【起点钉死在段内绝对位置 0,20,40,…，网格不动，t 只决定哪些已经可见】

t = 205
  帧号   0    20   40   60   80   100  120  140  160  180        205
         ●────●────●────●────●────●────●────●────●────●·········┤
         ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✗
         └──── 这 9 个的窗口尾端都 ≤ 205 ────┘    180+32=212 > 205，还看不见

t = 206
  帧号   0    20   40   60   80   100  120  140  160  180         206
         ●────●────●────●────●────●────●────●────●────●··········┤
         ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✗
         └──── 还是同样这 9 个，全部原样复用，一次都不用重编 ────┘

  要等到 t = 212（= 180+32），第 10 个起点才进入可见范围：
  帧号   0    20   40   60   80   100  120  140  160  180          212
         ●────●────●────●────●────●────●────●────●────●···········┤
         ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓ ← 新增第 10 个
                                    ⇒ 每 20 帧只新编 1 个窗口
                                      1.57 s ÷ 20 步 ≈ 0.079 s/step
```

绝对网格的两项直接收益（实测口径见 2.3 与 3.7）：

- **离线表小、在线增量少**：全数据集的网格窗口只有 **20,958 行 = 61.4 MiB**；在线每
  20 步只新编 1 个窗口，摊薄 **0.079 s/step**（实测 1.57 s/窗口）。
- **训练与部署共用同一套网格**：上线时 policy 每 20 步推理一次、真实 chunk 边界就是
  绝对网格，训练口径与部署口径天然一致。

### 2.3 预算 N=80：零截断，按 16 任务全集定标

**定标原则（明文口径）**：`motion.budget` 以及一切「随数据分布定的容量上限」类超参，一律以
16 任务完整数据集 **`/data/hongzefu/robomme_data_h5`**（16 任务 × 100 ep = 1600 ep）的统计定标，
**不以当前 v1 的 4 任务训练集** `robomme_data_h5_v2_4env400ep` 定标。理由两条：4 任务集是全集的
窄子集（只含 ButtonUnmask / ButtonUnmaskSwap / VideoUnmask / VideoUnmaskSwap，且这四个恰好都是
短 demo 任务），按它定的容量在 scope 扩到全集时必然溢出；两集段长口径同源——4 个共同任务
ep0–99 的 `(num_timesteps, exec_start_idx)` 400 条逐条相同，所以全集统计可以直接用作上界。
该原则在第二部分〇节列为红线 8。

**16 任务全集实测**（1600 ep，476,857 个 exec 样本；按 2.2 网格换算 `num_grid = ceil((段帧数 − 32) / 20)`，
一个样本的起点上界 = `num_grid(demo) + num_grid(exec)`）：

```
  P25 = 6    中位 = 12    P75 = 20    P90 = 36    P95 = 43    P99 = 56    最大 = 69
  均值 15.31    一个合法起点都没有的样本：4.72%
```

| 预算 N | 32 | 48 | 56 | 64 | 69 | 72 | **80** |
|---|---|---|---|---|---|---|---|
| 截断样本数 | 56,235 | 10,821 | 4,085 | 207 | 0 | 0 | **0** |
| 截断样本占比 | 11.793% | 2.269% | 0.857% | 0.043% | 0.000% | 0.000% | **0.000%** |
| 平均填充率 | 47.9% | 31.9% | 27.3% | 23.9% | 22.2% | 21.3% | **19.1%** |

零截断最小 N = **69**。最长纪录三条：最长 episode 是 VideoPlaceOrder ep4（1411 帧 = demo 1118 +
exec 293 → 55 + 14 = 69）；最长 demo 段 VideoPlaceOrder ep90（1145 帧 → 56）；最长 exec 段
BinFill（1044 帧 → 51）。逐任务上界超 32 的有五个：VideoPlaceOrder 69、VideoPlaceButton 52、
BinFill 51、VideoRepick 50、PickXtimes 50；RouteStick 32 正好卡满；v1 四任务分别为
ButtonUnmaskSwap 27、VideoUnmaskSwap 27、ButtonUnmask 21、VideoUnmask 18。

**溢出根因是 demo 段，不是 exec 段**：VideoPlace* 系列 demo 动辄 1000+ 帧，而按 2.2 的定义
demo 段整段已见、与 `t` 无关，贡献的是「从 episode 第 0 步就顶满」的常数项——200/1600 = 12.5%
的 episode 仅 demo 网格起点数就 > 32（demo 网格 MAX = 56），这些 episode 的每个样本都会截断，
丢的正是最早的历史。exec 网格单独看 MAX = 51、P90 = 25。

**为什么是 80 而不是 69 / 72**：裕度按同一把尺子——4env 上 MAX 27 取 32 是 18.5% 裕度，同比例
套到 69 是 81.8，80 落在同一档（15.9%）；69 / 72 贴着观测最大值走，数据集再动一下就顶穿。
另附一条弱证据：4 个共同任务上 400ep 版的各任务最长 exec 与前 100ep 逐个相同
（452/452、555/555、333/333、370/370），长尾大概率已采到，但无法证明饱和，这也是不取 72 的原因。

**硬地板 51**：若想压预算，demo 段单独放大 stride 到 40 可把零截断线从 69 降到 51，再放
（60 / 80 / 100）不再下降——瓶颈换成 BinFill 1044 帧 / PickXtimes 1025 帧的 exec 段（这两个任务
没有 demo 段）。守住「exec 每 20 帧一采 + 零截断」两条，N 不可能低于 51；要再往下压只能动
exec stride，那正是「间隔一个 action chunk」的本意所在。**本计划不采用 demo 独立 stride**，
只把它列为 S4 消融备选。

**当前 4 任务训练集在 N=80 下的实况**（395,289 个样本全量）：

```
  P25 = 4    中位 = 7    P75 = 12    P90 = 16    P95 = 18    P99 = 21    最大 = 27
  均值 8.09    一个合法起点都没有的样本：6.48%
```

| 预算 N | 32 | 48 | 64 | **80** |
|---|---|---|---|---|
| 截断样本占比 | 0.000% | 0.000% | 0.000% | **0.000%** |
| 平均填充率 | 25.3% | 16.9% | 12.6% | **10.1%** |

**与 framesample 的对齐关系**：N=80 不等于帧路的 `_max_frames = 32`，「两路同预算」这一层对齐
**不成立**；「尽可能和 framesample 对齐」只保留在 padding + mask 同款这一层（3.4、3.5）。用户的
三条拍板为「尽可能不截断任何样本」「容量按全集定标」「padding / mask 与 framesample 同款」。

### 2.4 ⚠ 零截断的代价：4env 上平均只有 8 个、16env 上平均只有 15 个位置是真数据

**这是本方案最需要清醒认识的一点，用户明确要求「写入 plan 让用户清楚认知」，
不藏在技术细节里：**

- 运动路固定占 **80 个 memory 位置**，但 4env 上平均只有 **8.09 个**是真 motion token，
  16env 上平均 **15.31 个**；
- 其余 **约 72 个位置（89.9%）是 padding**（16env 为 64.7 个 / 80.9%），靠 `motion_mask=False` 屏蔽；
- **6.48% 的样本（约 25,600 个）一个真 motion token 都没有**（16env 为 4.72%）——整条运动路全是
  padding，这些样本等价于「motion 功能未启用」；
- 分布很偏：4env P25 只有 4 个真数据，中位 7 个，要到 P90 才有 16 个；16env P25 6、中位 12、
  P75 20——四分之三的样本连 20 个位置都填不满，却要为 12.5% 的长 demo episode 全程背 80 个位置。

**为什么仍然接受**：这是「零截断 + 按全集定标」的直接代价。要提高填充率只能降预算（4env 上
N=16 时填充率 48.9%，但 8.88% 的样本被截断，丢的是最早的历史；全集上零截断硬地板是 51，
见 2.3），或改用变间隔采样（违背「间隔一个 action chunk」的本意）。用户已明确选择零截断优先。

**三个后果需要在实验中盯住**：
1. attention 里 89.9%（4env）/ 80.9%（16env）的运动位置被 mask，计算恒定支出但无信息——形状
   固定是 JAX jit 的硬约束，省不掉（详见 3.4）。
2. 早期样本（`t` 小）与晚期样本（`t` 大）的运动路信息量差异极大（0 个 vs 69 个，4env 内
   0 个 vs 27 个），模型可能学成「按 motion 有效数判断 episode 进度」的捷径——S4 需要一个消融
   专门测这个。
3. 6.48% 全空样本使得「motion 到底有没有用」的评估必须**分层看**（按有效数分桶），
   整体平均会被全空样本稀释。

### 2.5 当前帧附近的空白：不补

窗口 33 帧比网格步长 20 帧长，所以当前帧附近必然留一段空白：网格上最靠前的合法起点，
其窗口尾端离当前帧最近 8 帧、最远 27 帧（取决于 `t` 落在网格的哪个相位）——
**当前时刻的运动始终缺席**。

**用户已拍板：不补**（「采样不到 33 窗口都不补」）。不加网格外的起点（例如紧贴当前帧的
`t−32`），不做钳位回退。凑不齐完整 33 帧窗口的位置就是缺失，走 padding + mask（3.4）。

## 三、链路：运动特征怎么送入模型

### 3.1 改动前后链路图（`AGENTS.md` 第 18 条）

分两个子节：改动前全图（3.1.1）、改动后全图（3.1.2）。图中两处标注的 `static_pos_emb`
/ `pos_f` 与 padding 分别在 3.2、3.4 单独展开。图中 `b` = batch size。

#### 3.1.1 改动前：memory 单路（帧路）

```
                     样本 = (episode g, 当前帧 t)
                               │
                               │ even_sampling_indices(t, 32)   [shared/sampling.py]
                               ▼
              32 个历史帧号（变长间隔铺满 [0, t]；历史不足时只有 n = t+1 < 32 个）
                               │
                               │ 逐帧号查离线表（FrameSampStore）
                               ▼
   ┌───────────────────────────┼────────────────────────────────┐
 image_emb_4x4            pos_emb_4x4                       state_emb
 (n,16,2048) bf16         (n,16,768) f32                    (n,8) f32
 每帧 16 个外观 token      每 token 一个 3D 位置编码         （use_state_emb=False，
   │                           │                             不进链路，只随行交付）
   │        _pad 右填充到 32 帧 + 帧级 mask (32,)（3.4）      │
   │        reshape / repeat：32 帧 × 16 token = 512            │
   ▼                           ▼                                ▼
 static_image_emb         static_pos_emb                  static_state_emb   static_mask
 (b,512,2048) bf16        (b,512,768) f32                 (b,512,8)          (b,512) bool
   │                           │
   │                           │ pos_proj: Linear(768→768) + silu
   │                           ▼
   │                      (b,512,768)
   └───────────┬───────────────┘
               │ 最后一维 concat：2048 ⊕ 768 = 2816
               ▼
        (b, 512, 2816)
               │ encoder_static: Linear(2816→2048)
               ▼
      mem tokens (b, 512, 2048)      ar=F  na=F  input_mask = static_mask
               │
               ▼
 prefix ┌─ mem 512 ─┬─ img 2×256=512 ─┬─ prompt ≤64 ─┐ = 1088   (b,1088,2048)
        │ ar=F na=F │   ar=T/F na=T   │  ar=F na=F   │
        └── cumsum(na)==0 的「记忆区」：image 看不到，prompt / action 看得到 ──┘
 suffix  20 个 action token（pi05=True，无 state token；状态/时间走 adarms_cond）
         → 全序列 1108，attention 1108×1108
```

#### 3.1.2 改动后：memory 双路并列（左列逐位不动，右列整列新增）

```
                          样本 = (episode g, 当前帧 t)
              ┌────────────────────┴──────────────────────────────┐
     路1 帧路（3.1.1 整列逐位不变）              路2 运动路（★新增，独立采样）
              │                                                   │
  even_sampling_indices(t, 32)               段内绝对网格起点 0, 20, 40, …（2.2）
  变长间隔铺满 [0, t]                         合法条件：起点+32 ≤ 当前帧（前视 33 帧窗口）
              │                              取最近 ≤80 个（4env 最多 27、16env 最多 69）
              │                                                   │
  查 FrameSampStore                     ┌─ 训练：查离线表 motion_token.f32.bin
  (32,16,2048) bf16                     │      (20958,768) f32 = 61.4 MiB
  (32,16,768)  f32                      │      每起点 seek(row×3072) 读 1 行
              │                         └─ 在线：33 帧原图 (33,256,256,3)
  reshape 512                                  → Wan VAE 冻结 → (9,16,32,32)
              │                                → WanLatentMotionEncoder 冻结 → (768,)
              ▼                                                   ▼
  static_image_emb (b,512,2048)               motion_emb  (b,80,768) f32  padding 行填 0
  static_pos_emb   (b,512,768)                motion_mask (b,80) bool     padding 位 False
  static_mask      (b,512)                    pos_f       (b,80,768) f32  padding 行填 0
              │                               ↑ 起点帧 16 个 pos 的均值（3.2）
              │                                                   │
  pos_proj(768→768)+silu                      motion_pos_proj(768→768)+silu      ★新参数
  concat → (b,512,2816)                       concat → (b,80,1536)
  encoder_static(2816→2048)                   motion_encoder_static(1536→2048)   ★新参数
              │                                                   │
              ▼                                                   ▼
  帧 tokens (b,512,2048)                      motion tokens (b,80,2048)
              └────────────────────┬──────────────────────────────┘
                                   │ 长度轴 concat：512 + 80 = 592
                                   ▼
            memory (b,592,2048)    input_mask (b,592) = [static_mask ⊕ motion_mask]
                                   │ ar_mask / na_mask 各追加 80 个 False
                                   ▼
 prefix ┌─ mem 512 ─┬─ motion 80 ★─┬─ img 2×256=512 ─┬─ prompt ≤64 ─┐ = 1168（原 1088）
        │ ar=F na=F │  ar=F na=F   │   ar=T/F na=T   │  ar=F na=F   │
        └──── 记忆区扩到 592：image 看不到，prompt / action 看得到 ────┘
 suffix  不变 → 全序列 1188（原 1108），attention 1188²（+14.96%）
```

读图抓三条对应关系：

1. **右列是左列的镜像**：都是「特征 ⊕ 位置编码 → silu 投影 → concat → 一个 Linear 压到
   2048」。区别只在输入粒度——帧路一帧出 16 个外观 token，运动路一个 33 帧窗口只出
   1 个运动 token。
2. **两列在 concat 之前互不相干**：采样各采各的（变长间隔 vs 绝对网格）、表各查各的、
   投影各用各的参数；唯一交汇点就是最后那次 `512 + 80` 的长度轴 concat。
3. **重活全在训练环外**：右列的 Wan VAE 与 `WanLatentMotionEncoder` 只在离线抽表 /
   在线评估时跑（在线按 2.2 的网格每 20 帧才增量编 1 个窗口，见 3.7）；训练时右列就是
   「seek 读几行 f32 + 两个小 Linear」，新可训练参数只有打 ★ 的两层，合计 3.74 M（六节）。

### 3.2 `static_pos_emb` 与 `pos_f` 的实现

**底表唯一**。`pos_emb_4x4.f32.bin` 按全 timestep 域逐帧存一行 `(16, 768)` f32，由
`PosEmb3D(dim=768)` 对 `arange(全域帧数)` 一次性预计算（在线侧 `FrameSampMemory.__init__`
用同一 `PosEmb3D`、同一 `arange` 构表，两侧逐位同表）。任意帧号都能经
`FrameSampStore.pos_rows(frames)` 直接取行。

**768 维的内部构成**（`dim // 6 = 128`）：

```
768 = [ 时间 sin 128 | 时间 cos 128 | y sin 128 | y cos 128 | x sin 128 | x cos 128 ]
       └── 前 256：帧号 t 的时间编码，同一帧 16 行完全相同 ──┘└─ 后 512：4×4 网格点
                                                y,x ∈ {2,6,10,14}（4*y+2），16 行各不同 ─┘
```

**帧路 `static_pos_emb`**：`FrameSampDataset.__getitem__` 拿 `even_sampling_indices` 选出的
帧号调 `store.pos_rows(frames_arr)` → `(n,16,768)` → `_pad` 右填充 → `(32,16,768)` →
`reshape(-1, 768)` → `(512, 768)`。逐 token 使用，每个向量都是某个真实网格点的合法编码。

**运动路 `pos_f`**：每个合法起点先换算成全 timestep 域帧号（exec 段起点 `u` → `es + u`；
demo 段起点 `s` → `s`），调同一个 `pos_rows` 查出该帧 `(16, 768)`，**沿 16 那个轴取均值**
得 `(768,)`；80 个起点堆成 `(80, 768)`，padding 行填 0。均值的真实语义要说清楚：

- 前 256 维（时间）：16 行本来就相同，均值 = 原值，**无损保留**——这正是运动路需要 pos
  的原因（motion token 只描述「窗口里发生了什么」，不含「发生在什么时候」）；
- 后 512 维（空间）：4 个网格点 sin/cos 的平均**不等于任何位置的编码**（要在 128 个频率上
  同时成立，做不到），且空间编码与帧号无关——这 512 维对所有起点、所有样本都是**同一个
  常数向量**，零信息但无害（下游 `motion_pos_proj` 可学会忽略恒定输入）。

所以 `pos_f` 实际有效内容 = 起点帧时间编码 256 维 + 全局常数 512 维。`pos_source:
frame_mean` 因此是权宜（盲区清单第 3 条），列为 S4 可选消融项。

**投影互不共享**：帧路走 `pos_proj`、运动路走新建的 `motion_pos_proj`，参数树互不沾边；
两路只共享 `pos_emb_4x4` 这张只读底表——这也是 `motion.enabled=false` 能逐位退回的前提。

### 3.3 三条 mask 在 token 数轴上的取值与效果

![三条 mask 在 token 数轴上的取值与效果](docs/v6-mask-axis.svg)

- `input_mask` 是唯一随样本变化的一行。记忆段前 16k 位为 True，k 是有效帧数。运动段前 m 位为 True，m 是采到的真起点数。文本段前 L 位为 True，L 是指令 token 数。图像和动作全 True。
- `ar_mask` 全序列只有两个 True，位置分别是第 592 位和第 1168 位，即图像段第一个 token 和动作段第一个 token。
- 对 `ar_mask` 做累加得到块号：记忆和运动是块号 0，图像和文本是块号 1，动作是块号 2。
- `na_mask` 只有图像段 512 位为 True。
- 第一个 `na=True` 之前的范围是第 0 到 591 位，正好是记忆加运动两段。条件 C 只在 k 落在这一段时才可能删格子。

### 3.4 两路 padding 与 mask 的实现

**这一节解决什么问题。** 模型走 JAX jit，每个输入张量的形状在编译期固定：帧路恒为 512 个
memory 位置、运动路恒为 `motion.budget = 80` 个位置。但每个样本的真数据个数是变的：帧路在
episode 开头只有 `t+1 < 32` 帧，运动路的合法起点在当前 4 任务集上平均 8.09 个、最多 27 个、
6.48% 的样本为 0 个（16 任务全集：平均 15.31、最多 69、4.72% 为 0；预算 80 按全集定标，2.3）。
所以要做两件事，缺一不可：不够的补齐到固定长度（**padding**），再告诉模型哪些位置是补的、让
它们对输出零贡献（**mask**）。

**阅读路线。** 先给两张调用链图（调用链 A 训练、调用链 B 推理），每个节点写清它在干什么、
负责数据流里的哪几步；再给一张两路数据流图，每一跳标形状、dtype、哪些位是 0 或 False；最后
固定一个样本逐站走到 attention 权重为 0。样本选法：帧路跟 `(g, t=5)`，运动路跟 `(g, t=200)`。
两路不能用同一个 `t`，因为帧路 `t ≥ 31` 起恒满 32 帧，而运动路 `t ≥ 32` 才有第一个合法起点，
同一时刻两路不会同时出现部分填充。

```
════════════════════ 调用链 A：训练（scripts/training/train.py）════════════════════

main(config)
│   训练入口：读配置、建模型、建数据加载器，然后进入「取一个 batch → 算一次梯度 →
│   更新一次参数」的循环。
│
├─ create_data_loader                         [src/mme_vla_suite/training/dataloader.py]
│  │   建数据加载器：一个不停产出 batch 的东西。训练循环只管从它手里拿，不管怎么来的。
│  │
│  ├─ _create_framesamp_dataset  三闸
│  │      建数据集对象之前先查三样：离线特征库有没有正在被写（require_no_pack_lock）、库的
│  │      元信息能不能读（StoreMeta.load）、库是否通过过校验（require_verified）。任一不过
│  │      拒绝启动，防止拿半成品数据训练。
│  │
│  └─ TorchDataLoader(num_workers=N)
│     │   开 N 个子进程并行准备样本。主进程只训练，子进程只读数据、拼数据，两边互不等待。
│     │
│     └─ 每个 worker 进程里：FrameSampDataset.__getitem__(idx)      ← numpy，CPU
│        │   「给我第 idx 个样本」的实现。一个样本 = 某条 episode 的某一帧 t，任务是把
│        │   这一帧的历史记忆准备好、形状固定、能直接进模型。
│        │
│        ├─ ① even_sampling_indices(step=t, 32)
│        │      选帧：决定回看哪 32 帧。t ≥ 31 时在 [0, t] 上均匀取 32 个；t < 31 时历史
│        │      不够，有多少拿多少，返回 t+1 个。例：t=5 → [0,1,2,3,4,5]，6 个，缺 26。
│        │
│        ├─ ② store.read_image_rows / pos_rows / state_rows
│        │      查表：按选出的帧号去离线特征库读每帧的外观特征与位置编码。行数 = 选出的
│        │      帧数，是变的。例：img (6,16,2048) bf16、pos (6,16,768) f32。
│        │
│        ├─ ③ _pad(img, pos, stt, n)
│        │      补齐：开一块固定 32 行的空间，前 n 行放真数据，后 32−n 行写 0；同时记一个
│        │      32 位布尔向量 mask，前 n 位 True。例：img → (32,16,2048)，第 6–31 帧全 0，
│        │      mask = [T×6, F×26]。这一步只解决「形状固定」，不负责「让模型忽略」。
│        │
│        ├─ ④ reshape(-1, d) + np.repeat(mask, 16)
│        │      摊平：一帧 16 个 token，32 帧摊成 512 个位置；mask 每位复制 16 份对齐成
│        │      512 位。产出 static_image_emb (512,2048)、static_pos_emb (512,768)、
│        │      static_mask (512,) = [T×96, F×416]。
│        │
│        └─ ★motion memory 接入：①′–③′ 运动路（同一个 __getitem__ 里另做一遍）
│               ①′ 选起点：网格 0,20,40,… 里满足 f+32 ≤ t 的起点，取最近 ≤80 个。
│                  例：t=200 → 0,20,…,160 共 9 个，缺 71；t=5 → 0 个。
│               ②′ 查表：每个起点去 motion_token.f32.bin 读一行 (768,) f32；起点帧的 16 个
│                  位置编码取均值得 pos_f (768,)。
│               ③′ 补齐：另写的填充函数补到 80 行，motion_emb/pos_f 后 80−k 行填 0，
│                  motion_mask (80,) 前 k 位 True。一个起点 = 一个 token，没有 ④ 那步摊平。
│
├─ collate
│      把 64 个 worker 产出的单样本摞成一个 batch：(512,2048) → (64,512,2048)，其余键同理。
│      装进 HistAugObservation。
│
├─ batch = next(data_iter)
│      训练循环每轮从加载器取一个 batch。
│
└─ ptrain_step = jax.jit(train_step)                                    ← 以下 jit 内，GPU
   │   把「一步训练」编译成 GPU 程序：第一次运行时把整个计算图编好，之后每次直接跑编译
   │   产物。编译产物要求所有输入形状固定——这就是 ③ 必须补齐到 32、不能交变长数据的原因。
   │
   └─ loss_fn → model.compute_loss(rng, obs, actions)     [models/integration/history_pi0.py]
      │   真正的前向计算。
      │
      ├─ ⑤ embed_memory(obs)
      │      把 ④ 交来的 512 个位置的原始特征变成 512 个 2048 维「记忆 token」：
      │      static_pos_emb 过 pos_proj(768→768)+silu，与 static_image_emb 拼成 2816 维，
      │      过 encoder_static(2816→2048)。补齐的零行也照过这两层，出来是非零向量，
      │      不做任何分支。static_mask 原样往下传。
      │      ★motion memory 接入：运动路走独立的 motion_pos_proj / motion_encoder_static 得
      │      (b,80,2048)，两路 token 长度轴接成 (b,592,2048)，mask 接成
      │      [static_mask ⊕ motion_mask] (b,592)。
      │
      ├─ embed_prefix(obs)
      │      当前观测的两张图像编成 512 个 token，文本指令编成 ≤64 个 token，各带自己的 mask。
      │
      ├─ embed_suffix(obs, x_t, t)
      │      待预测的 20 步动作（加了噪声）编成 20 个 token。
      │
      ├─ ⑥ input_mask = concat([mem, prefix, suffix]) → (b,1188)
      │      三段布尔向量按 记忆 → 图像文本 → 动作 的顺序首尾相接。motion_mask 在这里没有
      │      任何特殊待遇。同时 positions = cumsum(input_mask) − 1：真数据位加一、补齐位不加，
      │      所以补齐位不占位置编号。
      │
      ├─ ⑦ make_attn_mask(input_mask, ar_mask, na_mask) → (b,1188,1188)
      │      把一条 1188 位布尔向量变成一张 1188×1188 的「第 q 个 token 能不能看第 k 个」表。
      │      核心是外积 valid_mask[q,k] = input_mask[q] ∧ input_mask[k]：补齐位所在的整列
      │      变 False，不管谁当 query 都看不到它。再与结构规则（prefix 双向 / action 因果 /
      │      image 不看 memory）按位与。
      │
      └─ ⑧ PaliGemma.llm([mem, prefix, suffix], mask, positions)      [src/openpi/models/gemma.py]
         │   把三段 token、这张表、位置编号交给语言模型主干。主干是一层层 Block，每层里的
         │   Attention.__call__ 是真正算注意力的地方：
         │      logits = q·k                         (b, heads, 1188, 1188) f32
         │      masked = where(mask, logits, −2.38e38) False 格子换成 f32 最小值
         │      probs  = softmax(masked)             exp(−2.38e38) 精确为 0 → 补齐列权重 0
         │      out    = probs @ v                   补齐位的 value 乘 0，对输出零贡献
         │
         └─ suffix_out → action_out_proj → loss
                主干输出里动作那 20 个位置的向量过线性层变回动作维度，与真实动作比较得 loss。
```

```
════════════════════ 调用链 B：推理（src/mme_vla_suite/policies/policy.py）════════════════════

推理没有离线特征库，记忆是边跑边攒的。每一步做两件事：先把当前帧存进记忆，再用记忆算动作。

阶段一（每步）：MME_VLA_Policy.add_buffer(obs)
│
└─ FrameSampMemory.add_buffer                              [policies/framesamp_memory.py]
       把当前帧图像过一次视觉编码器，得到与训练离线表同款的外观特征和位置编码，按步号存进
       内存字典 _history_feats[step]。这个字典就是推理时的「离线表」，一步步长出来。
       ★motion memory 接入：另存一份 256 域原图滚动缓冲（运动编码器要 256 域，视觉编码器用
       的是 224）。每当「下一个网格起点 + 32」这 33 帧全部到齐，把它们过运动编码器得一个
       768 维 motion token，存 _history_feats_motion[f]。每 20 帧才触发一次（3.7）。

阶段二（每步）：MME_VLA_Policy.infer(obs)
│
├─ _prepare_history                                                     ← numpy，CPU
│  │   对应训练 worker 的 ①–④，数据来源从离线表换成内存字典。
│  │
│  └─ FrameSampMemory.prepare_frame_sampling(step_idx, budget=512, token_per_image=16)
│        ① even_sampling_indices(step_idx, 32)     同一个选帧函数。
│        ② _load_emb(history_feats, indices)       从字典按帧号取特征。
│        ③ right_padding_token_emb(…, 32)          补齐到 32 行，训练侧 _pad 的等价老写法：
│                                                   用 concatenate 拼零块而不是原地填，数值相同。
│        ④ reshape + np.repeat(mask, 16)            同样摊平成 512 位。
│     ★motion memory 接入：另加 _prepare_motion，给运动路做 ①′–③′。不塞进
│        prepare_frame_sampling，因为该函数注释明记「只换模块、不换数值路径」，不许动。
│
├─ _input_transform → HistAugObservation.from_dict
│      做与训练同样的归一化和格式转换，加一个 batch 维，b = 1。
│
└─ _sample_actions = jit(model.sample_actions)                          ← 以下 jit 内，GPU
   │   与训练的差别：训练一次前向算 loss；推理先算一次前缀、再跑多轮去噪。
   │
   ├─ ⑤ embed_memory(obs) + embed_prefix(obs)
   │      与训练完全一样，得 592 个记忆 token 和 576 个图像文本 token。
   │
   ├─ ⑥ prefix_mask = [mem 592 | vlm 576] → (b,1168)；positions = cumsum − 1
   │
   ├─ ⑦ make_attn_mask(prefix_mask, ar, na) → (b,1168,1168)
   │      对 1168 位做外积，补齐列整列 False。
   │
   ├─ ⑧ 前缀 pass：PaliGemma.llm([mem, vlm, None], mask, positions) → kv_cache
   │      把 1168 个 token 交给主干算一遍，不要输出，只要每层算出的 key/value，存成 kv_cache。
   │      前缀在整个去噪过程中不变，所以只算这一次。kv_cache 里补齐位的 K/V 存在，
   │      但 ⑦ 已把它们的列封死。
   │
   └─ step(carry) 去噪循环 × num_steps
         embed_suffix(obs, x_t, time)
             把当前带噪声的动作编成 20 个 token。
         full_attn_mask = [repeat(prefix_mask, 20 行) | make_attn_mask(suffix)] → (b, 20, 1188)
             这 20 个 query 对前缀 1168 位的可见性，直接拿 prefix_mask 复制 20 行——
             prefix_mask 里 False 的位自然成了整列 False，补齐位第二次被封，不需要再做外积。
             对动作自己的 20 位用 make_attn_mask 得 20×20 因果表，拼在右边。
         positions = sum(prefix_mask) + cumsum(suffix_mask) − 1
             动作的位置编号接在「前缀真数据个数」之后，补齐位同样不占号。
         PaliGemma.llm([None, None, suffix], mask=full_attn_mask, kv_cache)
             只算这 20 个 token 的 query；key/value = 缓存的 1168 个 + 新的 20 个。
             Attention.__call__ 里 where / softmax / @v 与训练侧 ⑧ 同一段代码。
         → 循环结束输出 actions
```

**两路数据流图 ①–⑧**（编号与上面两张调用链图一致；左列 `t=5` 的帧路样本，右列 `t=200` 的
运动路样本，汇合后单列）：

```
                          样本 = (episode g, 当前帧 t)
          ┌────────────────────────┴────────────────────────────┐
     帧路，取 t=5 的样本                              运动路，取 t=200 的样本
          │                                                     │
 ① 选帧 even_sampling_indices(5, 32)              ① 选起点：网格 0,20,40,… 中满足 f+32 ≤ 200 者
    → 帧号 [0,1,2,3,4,5]，n=6，缺 26                  → f ∈ {0,20,…,160}，k=9，缺 71
          │                                                     │
 ② 查表 FrameSampStore                            ② 查表 motion_token.f32.bin + pos_rows 均值
    img (6,16,2048) bf16                             motion 行 (9,768) f32
    pos (6,16,768)  f32                              pos_f    (9,768) f32
    stt (6,8)       f32
          │                                                     │
 ③ _pad(…, n=6)  目标长度 _max_frames=32          ③ 另写填充函数  目标长度 motion.budget=80
    img (32,16,2048)  第 6–31 帧 = 0                 motion_emb (80,768)  第 9–79 行 = 0
    pos (32,16,768)   第 6–31 帧 = 0                 pos_f      (80,768)  第 9–79 行 = 0
    mask (32,) bool   [T×6, F×26]  ← 帧级            motion_mask (80,) bool [T×9, F×71] ← 已是 token 级
          │                                                     │
 ④ reshape(-1, d) + np.repeat(mask, 16)           ④ （无此步：一个起点 = 一个 token）
    static_image_emb (512,2048) bf16  位 96–511 = 0
    static_pos_emb   (512,768)  f32   位 96–511 = 0
    static_mask      (512,) bool      [T×96, F×416]
          │                                                     │
 ═════════╪═══════════ dataloader 结束 / collate 成 batch ═══════╪═════════════
          │                                                     │
 ⑤ embed_memory（padding 行照算，不分支）
    pos_proj(768→768)+silu → concat 2816             motion_pos_proj(768→768)+silu → concat 1536
    encoder_static(2816→2048)                        motion_encoder_static(1536→2048)
    → (b,512,2048)  第 96–511 行非零                 → (b,80,2048)  第 9–79 行非零
          └──────────────────┬──────────────────────────────────┘
                             │ 长度轴 concat
                    memory (b,592,2048)
                    input_mask (b,592) bool = [static_mask ⊕ motion_mask]
                             │
 ⑥ compute_loss：三段 mask 首尾相接
    input_mask = [mem 592 | img 512 + prompt ≤64 | action 20] → (b,1188) bool
    positions  = cumsum(input_mask) − 1 → (b,1188) int，padding 位不推进
                             │
 ⑦ make_attn_mask(input_mask, ar_mask, na_mask) → (b,1188,1188) bool
    = 结构规则 attn_mask  ∧  valid_mask  ∧  ¬mask_not_attend
      valid_mask = input_mask[:,None,:] * input_mask[:,:,None]   ← padding 屏蔽的全部
      → 帧路第 96–511 列、运动路第 521–591 列 整列 False
                             │
 ⑧ gemma Attention.__call__
    logits (b,heads,1188,1188) f32
    masked = where(attn_mask, logits, −2.3819763e38)
    probs  = softmax(masked)         ← False 列 exp(−2.38e38) 精确为 0
    out    = probs @ v               ← padding 位 value × 0，对任何输出零贡献
```

下面跟着这两个样本逐站走。第一至四站在 dataloader 侧（训练时是 worker 进程里的
`FrameSampDataset.__getitem__`，推理时是主进程的 `FrameSampMemory`，都是 numpy / CPU）；
第五至七站在 JAX jit 内（GPU）。每站给输入输出的形状与 dtype。

**第一站（dataloader 侧）：这个时刻能取到多少真数据**

帧路，样本 `(g, t=5)`。`FrameSampDataset.__getitem__` 调 `even_sampling_indices(step=5,
token_budget=32)`（`shared/sampling.py`），`step < 32` 走 `range(step+1)` 分支，返回帧号
`[0,1,2,3,4,5]`，`n = 6`。`t ≥ 31` 起走 `linspace(0, t, 32)` 分支恒返回 32 个，所以帧路 padding
只出现在每条 episode 的前 31 帧。

运动路，样本 `(g, t=200)`。一个运动窗口 = 从起点 `f` 往后连续 33 帧 `[f, f+32]`，经 Wan VAE +
`WanLatentMotionEncoder` 编成一个 768 维 motion token。前视方向与 encoder 的训练语义一致
（motion 描述「相对锚点 z0 之后发生的运动」）。训练时窗口必须整体位于已发生的历史内，约束为
尾端 ≤ 当前帧，即 `f + 32 ≤ t`（2.1）。起点只能取段内绝对网格 `0, 20, 40, …`（2.2）。`t = 200`
时 `f ≤ 168`，合法起点为 `0, 20, …, 160` 共 `k = 9` 个，缺 71 个。同一条 episode 若在 `t = 5`，
一个合法起点都没有，`k = 0`，80 位全是 padding，这就是 2.4 说的 6.48%。

**第二站（dataloader 侧）：逐帧 / 逐起点取单帧特征，行数随真数据个数变**

帧路。6 个帧号先加 `row_base[g]` 得到全局行号，然后三次查表（`FrameSampStore`）：

| 键 | 调用 | 形状 | dtype | 字节 |
|---|---|---|---|---|
| img | `store.read_image_rows(rows)` | `(6, 16, 2048)` | bf16 | 393,216 |
| pos | `store.pos_rows(frames_arr)` | `(6, 16, 768)` | f32 | 294,912 |
| stt | `store.state_rows(rows)` | `(6, 8)` | f32 | 192 |

每帧 16 个外观 token 与 16 个 3D 位置编码（768 维的构成见 3.2），state 只随行交付、不进链路
（`use_state_emb=False`）。

运动路。9 个起点先换算为全 timestep 域帧号（exec 段 `es + u`，demo 段 `s`，见 4.2），然后：

| 键 | 来源 | 形状 | dtype |
|---|---|---|---|
| motion 行 | `motion_token.f32.bin (20958, 768)`，按 `(段, 网格序号)` 定位行号，`seek(row × 3072)` 读 1 行 | 每起点 `(768,)`，堆成 `(9, 768)` | f32 |
| pos_f | 同一张 `pos_rows`，取起点帧 `(16, 768)` 沿 16 轴均值（3.2） | 每起点 `(768,)`，堆成 `(9, 768)` | f32 |

训练时 motion 行从离线表读，在线评估时现编（3.7）。

**第三站（dataloader 侧）：拼成这一个时刻的完整 memory，不足补 0 并记下 mask**

帧路走 `FrameSampDataset._pad(img, pos, stt, n=6)`：按目标长度 `m = _max_frames = 32` 一次性
`np.empty` 预分配，`out[:6]` 放真数据，`out[6:] = 0`，三键同式；同时
`mask = np.zeros(32, bool); mask[:6] = True`。输出：

| 键 | 形状 | dtype | 第 0–5 帧 | 第 6–31 帧 |
|---|---|---|---|---|
| img | `(32, 16, 2048)` | bf16 | 真数据 | 全 0 |
| pos | `(32, 16, 768)` | f32 | 真数据 | 全 0 |
| stt | `(32, 8)` | f32 | 真数据 | 全 0 |
| mask | `(32,)` | bool | True | False |

在线侧同一步由 `right_padding_token_emb`（`shared/data_utils.py`）完成，用 `np.concatenate` 把
零块拼到尾部而不是原地填，数值结果与 `_pad` 相同；`framesamp_memory.py` 注释明记「必须复用
——只换模块、不换数值路径」，`_pad` 是它在训练侧的等价预分配版。

运动路另写一个填充函数，目标长度是配置项 `motion.budget = 80`，不复用 `_pad`（`_pad` 的目标
长度是类内常量 `_max_frames`，签名是 img/pos/stt 三键，运动路只有 emb/pos_f 两键，长度语义与
签名都不同，第二部分 2.6 已定）。输出：

| 键 | 形状 | dtype | 第 0–8 行 | 第 9–79 行 |
|---|---|---|---|---|
| motion_emb | `(80, 768)` | f32 | 9 个起点的 token，按时间序 | 全 0 |
| pos_f | `(80, 768)` | f32 | 9 个起点帧的 pos 均值 | 全 0 |
| motion_mask | `(80,)` | bool | True | False |

填 0 的 dtype 不需要特判：bf16 与 f32 的 0 位型都是全零字节，新旧链路对拍可以逐字节比。填 0
本身不是屏蔽手段，模型不是靠「看到 0」来忽略这些位置的，真正起作用的是从第五站开始的 mask。

**第四站（dataloader 侧）：帧路把帧级 memory 摊成 token 级，mask 跟着 ×16**

帧路一帧 16 个 token，`__getitem__` 把 `(32,16,2048)` `reshape(-1, 2048)` 铺成 512 个位置，mask
必须跟着每位复制 16 份：`np.repeat(mask, 16)`，`(32,) → (512,)`。交付：

| 键 | 形状 | dtype | 位 0–95 | 位 96–511 |
|---|---|---|---|---|
| static_image_emb | `(512, 2048)` | bf16 | 6 帧 × 16 token | 全 0 |
| static_pos_emb | `(512, 768)` | f32 | 同上 | 全 0 |
| static_mask | `(512,)` | bool | True | False |

运动路一个起点只对应 1 个 token，`(80,)` 的 `motion_mask` 天然就是 token 级，没有这一步。

到这里 dataloader 结束。collate 后进模型的是 `static_image_emb (b,512,2048)`、
`static_pos_emb (b,512,768)`、`static_mask (b,512)`、`motion_emb (b,80,768)`、`pos_f (b,80,768)`、
`motion_mask (b,80)`。关于 padding 的全部信息只有两条布尔向量。

**第五站（JAX，jit 内）：帧路 img ⊕ pos 2816→2048、运动路 motion ⊕ pos_f 1536→2048，各自压成 2048 维记忆 token；再把各段 mask 接成一条**

模型主干只认 2048 维 token，所以这一站前半段是一次「翻译」：帧路每个位置把外观特征（2048）
和位置编码（768）拼成 2816 维，过一个线性层压到 2048；运动路把 motion token（768）和 `pos_f`
（768）拼成 1536 维，用另一组线性层扩到 2048。两者终点都是 2048，因为那是 gemma 的隐层宽度，
帧路恰好是压缩、运动路恰好是扩张，只是特征本体宽度不同，不是设计上的取舍。补齐的零行照样过
这两层，出来是普通的非零向量，模型此刻分不出哪些是真的。后半段是「拼装」：记忆、图像、文本、
动作各段自带的布尔向量按顺序首尾接成一条 1188 位的 `input_mask`，`motion_mask` 没有任何特殊
待遇，接完之后下游不再知道哪一位来自哪一路。

`embed_memory`（`history_pi0.py`）不对 padding 位做任何分支：

- 帧路：`static_pos_emb` 过 `pos_proj: Linear(768→768) + silu` → `(b,512,768)`，与
  `static_image_emb` 在最后一维 concat → `(b,512,2816)`，过 `encoder_static: Linear(2816→2048)`
  → `(b,512,2048)`。第 96–511 行输入是零向量，输出是 bias 决定的**非零**向量。
- 运动路：`pos_f` 过 `motion_pos_proj: Linear(768→768) + silu` → `(b,80,768)`，与 `motion_emb`
  concat → `(b,80,1536)`，过 `motion_encoder_static: Linear(1536→2048)` → `(b,80,2048)`。
  第 9–79 行同样是非零向量。`pos_f` 是起点帧 16 个位置编码沿 16 轴的均值（3.2）：前 256 维
  时间编码 16 行本就相同，均值无损保留；后 512 维 xy 编码取平均后对所有起点、所有样本都是
  同一个常数向量——**xy 那些维度还在，但不再携带信息**，下游 `motion_pos_proj` 会学会忽略它。
- 两段在长度轴 concat → memory `(b,592,2048)`；`input_mask = [static_mask ⊕ motion_mask]` →
  `(b,592)` bool。`motion_mask` 在这里没有任何特殊处理，只是接在后面。

`compute_loss` 再把三段拼成整条序列：`input_mask = concat([mem 592, prefix 512 img + ≤64 prompt,
suffix 20 action], axis=1)` → `(b,1188)` bool（现状 1108）。`ar_mask`、`na_mask` 同长同序。拼完后
下游不知道也不关心哪一位来自帧路、哪一位来自运动路。

对 `t=5` 的帧路样本，这 1188 位里第 96–511 位 False；对 `t=200` 的运动路样本，第 521–591 位
False。

**第六站（JAX，jit 内）：`make_attn_mask` 把一条 1188 位向量变成 1188 × 1188「谁能看谁」表，补齐位整列封死**

这一站把「哪些位是补的」这条一维信息变成「谁不能看谁」的二维规则。

attention 里每个 token（query，行 `q`）要决定看序列里每个 token（key，列 `k`）多少。控制这件事
的是 `(b,1188,1188)` 的布尔表，`(q,k)` 为 True 表示允许。`make_attn_mask(input_mask, ar_mask,
na_mask)` 用三部分按位与得到它：

1. 结构规则 `attn_mask = cumsum(ar_mask)[k] <= cumsum(ar_mask)[q]`，决定 prefix 双向、action
   因果，与 padding 无关；
2. `mask_na` 的 where，决定 image 不看 memory（3.6），与 padding 无关；
3. **`valid_mask = input_mask[:, None, :] * input_mask[:, :, None]`**，`(b,1188) → (b,1188,1188)`
   的外积：`(q,k)` 为 True 当且仅当第 `q` 位和第 `k` 位都是真数据。

第 3 项是 padding 屏蔽的全部。用 6 位迷你例子，`input_mask = [T,T,T,T,F,F]`：

```
valid_mask     k=0  k=1  k=2  k=3  k=4  k=5
   q=0 真       T    T    T    T    F    F
   q=1 真       T    T    T    T    F    F
   q=2 真       T    T    T    T    F    F
   q=3 真       T    T    T    T    F    F
   q=4 补       F    F    F    F    F    F
   q=5 补       F    F    F    F    F    F
```

第 4、5 **列**整列 False：不管谁当 query，都不允许看这两个位置，这就是「padding 对任何 query
不可见」。第 4、5 **行**也整行 False：padding 自己谁也看不到，但 padding 位的输出无人消费，这
一半无关紧要。回到我们的样本：`t=5` 时第 96–511 列、`t=200` 时第 521–591 列，在 1188 × 1188
的表里整列 False，不论结构规则怎么允许。

**第七站（JAX，jit 内，gemma）：False 格子在 softmax 里权重严格为 0，补齐位对任何输出零贡献**

这一站把第六站的「不能看」落实成数值上的 0。

`src/openpi/models/gemma.py` 的 `Attention.__call__` 收到的 mask 形状是 `(b,1,1188,1188)`：

```
logits = q · k                                  (b, heads, 1188, 1188) f32
masked = where(attn_mask, logits, −2.3819763e38)   False 格子换成 f32 最小值
probs  = softmax(masked, axis=-1)                  exp(−2.38e38) 精确为 0，不是很小的正数
out    = probs @ v                                 padding 列的 value 乘的是 0
```

所以整件事收口在这里：padding 位的零向量确实经过了 `motion_pos_proj` / `motion_encoder_static`，
算出了非零的 key 和 value，但没有任何 query 能给它非零权重。模型的每一个输出，与这 26 帧、这
71 个运动位置是否存在完全无关。

**两条补充**

- 位置编号也跳过 padding。`compute_loss` 里 `positions = cumsum(input_mask, axis=1) − 1`，
  `(b,1188)` int，padding 位不推进计数，RoPE 因此不受影响。运动路填 71 位还是 80 位，后面
  image / prompt / action 拿到的位置编号一样；`motion.enabled=false` 时全 False 的 80 位对
  positions 零贡献，这是逐位退回的又一个前提。
- 两路在这条链上完全对称。从第三站填 0 到第七站权重为 0，帧路和运动路走的是同一条路，没有
  任何一处按来源分支。3.5 说运动路 padding 与 `static_mask`「完全同款」，指的就是这个。

**推理时 padding 被封两次**

推理的前缀 pass 和去噪步是两次独立的注意力计算，各自要有自己的 mask（调用链 B 的 ⑦ 与
`step`）。第一次：前缀 pass 对 `prefix_mask (b,1168)` 做外积，帧路第 96–511 列、运动路第
521–591 列在 `(b,1168,1168)` 里整列 False；⑧ 算出的 `kv_cache` 里这些位置的 K/V 存在但从未被
读到。第二次：每个去噪步里 action 的 20 个 query 用的是
`full_attn_mask (b, 20, 1168 + 20) = [prefix_mask 广播成 20 行 | suffix 自己的 20 × 20]`，
`prefix_mask` 里 False 的位直接成了整列 False，所以 action token 在每一步都看不到 padding 列，
不需要再算一次外积；`positions` 用 `sum(prefix_mask)` 起算，padding 位同样不占位置编号。

### 3.5 并列拼接 + motion_mask

运动路以**并列拼接**进入记忆序列：`[512 帧路] + [80 运动路]` = 592。运动路完全独立——
独立采样、独立投影、独立 mask，`motion.enabled=false` 一键退回**逐位等价**的旧链路。
缺失位置（padding）喂 0 向量并置 `motion_mask=False`，语义与 framesample 的 `static_mask`
**完全同款**——帧路在 `step<32` 时同样是 padding + mask，用户要求的「尽可能和 framesample
对齐」也指向这一边。

### 3.6 注入点与冻结边界

**运动段放在 img 之前**：`make_attn_mask` 的
`mask_not_attend = (na[k] | na[q]) & (cumsum(na) <= 0)`，第一个 `na=True` 的位置是 image token。
放在 img 之前 ⇒ 运动段落入记忆区，沿用原设计「image token 不 attend memory」对预训练 VLM
视觉-语言对齐的保护；放在 img 之后则与所有 token 双向可见——列为 S4 消融项，不做默认。

**两级冻结边界**：Wan VAE 与 `WanLatentMotionEncoder` 都在训练环外、无梯度回传；训练环内
只有 `motion_pos_proj` 与 `motion_encoder_static` 两个新投影。

### 3.7 在线推理：增量编码与延迟账

MotionJEPA v8 全量抽取的**实测吞吐 0.635 chunk/s**（单 A40，fp32、**关 TF32**、窗口 batch 恒 1；
`docs/dataset-build-doc/v8-400ep-full/README.md` 记 396,302 chunk ÷ 8 分片，sacct Elapsed
22:21:51–22:56:48，全部 `COMPLETED 0:0`）→ **≈1.57 s/窗口**。

段内绝对网格下**每 20 帧才新增一个起点**，所以：

```
  1.57 s / 20 步 ≈ 0.079 s/step（摊薄）
  实际形态是「19 步零开销 + 第 20 步一次 1.57 s 的尖峰」
```

`MME_VLA_Policy.infer` 现有 `infer_time_ms` 是几十到几百毫秒量级，摊薄后的 79 ms 属同量级，
**「测试时同步推理」可行**。

两个仍需处理的细节：

1. **尖峰而非均摊**：第 20 步会出现 1.57 s 的单步延迟尖峰。若控制回路对单步延迟敏感，
   可在 policy 侧把编码放到后台线程 / 提前一步预编（起点可见性是可预测的），S3 决定。
2. **可选提速**：抽取时关 TF32、batch=1 是为了 finalize 语义 oracle 逐位复现
   （`extract_wan_chunk_latents_all.py` 头部注释写明这不是可选优化）；**在线不需要这个保证**，
   开 TF32 + bf16 预计再快 3~5 倍。代价是与 fp32 离线表的数值漂移，**启用前必须实测漂移量**。

## 四、对齐：数据集与模型怎么对上

整个方案建立在「MotionJEPA 已抽好的 latent 能被 MME-VLA 的帧号直接命中」之上。
**已实测确认，不需要重抽 261 GB latent。**

### 4.1 两库同源性核对（2026-09-01 全量跑过）

| 项 | 实测结果 |
|---|---|
| MME-VLA 侧 | `v1-store/episode_manifest.json`：1600 episode（4 任务 × 400 ep），483,291 timesteps，**395,289 exec 样本**；`raw_dir = /nfs/turbo/.../robomme_data_h5_v2_4env400ep` |
| MotionJEPA 侧 | `dataset-4env-v8/dataset-token/wan_chunk_latents/metadata.json`：2400 条目 = ButtonUnmask / ButtonUnmaskSwap 各 400 个 `_exec`（无 demo）+ VideoUnmask / VideoUnmaskSwap 各 400 个 `_exec` + 400 个 `_demo` |
| **demo 段帧数** | **1600 个 episode 全部差 0** —— `MME.exec_start_idx == MJ.demo.frames`，帧号 1:1 同源 |
| **exec 段帧数** | MotionJEPA 恒短 4~12 帧（分布 `{4:231, 5:537, 6:237, 7:127, 8:142, 9:180, 10:99, 11:41, 12:6}`），成因是 `build_data_raw_from_h5.py` 的 `EXEC_TRUNCATE_TAIL = 2`（首个 `is_completed=True` 后保留一帧即截断） |
| chunk 密度 | `num_chunks = frames − 32`，**stride = 1**——段内每一帧都是合法窗口起点（v6 只用其中 1/20） |
| 体量 | latent 261 GB（同一 turbo 文件系统，`df` 余 6.0 TB）；exec chunk 333,900 + demo chunk 62,402 = 396,302 |
| exec 段长度 | P0=96，P25=115，**中位 240**，P75=316，P90=450，P99=478，max=555，均值 247.1 |

**结论**：两库是同一批原始数据的两个派生副本，demo 前缀长度逐 episode 完全一致、exec 起点一致，
只是 MotionJEPA 在 exec 尾部多截了 4~12 帧。帧号可直接换算，**零重抽**。

### 4.2 帧号换算与 chunk 命中

2.2 的起点集合定义在 MME-VLA 的帧域上；每个合法起点按下式换算成 latent 库的 chunk 行
（MotionJEPA 的 chunk 是 stride=1 的稠密窗口，段内偏移即 chunk 序号）：

```
exec 段网格： u = 0, 20, 40, …        （u 是 exec 段内偏移）
              合法条件： u + 32 ≤ t − es      且   u < num_chunks_exec
              → 读 <Task>_ep<j>_exec.bin 第 u 个 chunk

demo 段网格： s = 0, 20, 40, …        （s 是 demo 段内偏移；demo 段整段已见）
              合法条件： s + 32 ≤ es − 1      且   s < num_chunks_demo
              → 读 <Task>_ep<j>_demo.bin 第 s 个 chunk
```

两段各自成网格、互不延续，窗口不跨 demo/exec 边界；起点集合与帧路无对齐关系——两条 ⚠
的原文见 2.2。

### 4.3 离线表与训练/在线双路一致性

训练不在线跑 encoder，而是读离线表：从已有 latent 只取 **20,958 个网格窗口**跑 encoder，
每行 768 维 f32，共 **61.4 MiB**，落 `v1-store/datasets/4task-gl-motion/`（格式契约见第二部分
一节）。在线评估则按 2.2 的网格增量现编，每 20 帧一次。两条路取数口径的一致由三道闸门
保证：M2（抽表逐位相等）、M3（索引映射双实现对拍）、M10（在线/离线同一起点特征一致），
判据见第二部分四节。

## 五、实施步骤（S0–S4）

| 阶段 | 内容 | 判据 |
|---|---|---|
| **S0 延迟与漂移先验** | 单窗口在线编码基准：fp32/关TF32 vs TF32+bf16 的 ms/窗口与两口径数值漂移 | 拿到 ms/窗口实测；漂移余弦 ≥ 阈值（阈值起工前拍板）；决定 S3 是否需提速 |
| **S1 特征就位** | 冻结 encoder，从已有 latent 读 **20,958 个网格窗口**（11.5 GiB / 261 GiB）跑 encoder，落 `v1-store/datasets/4task-gl-motion/`（**61.4 MiB**），沿用 framesamp 的 packed→verified 两阶段 + 逐行 digest | 200 条在线跑 encoder vs 表逐位相等；500 样本索引映射对拍；行数账 20,958 = exec 17,514 + demo 3,444 |
| **S2 model 接线** | 双路 memory + `motion.enabled` 开关 + `motion_mask`（本计划主体） | 关闭态与 HEAD **逐位等价**；开启态 smoke 跑通；`‖motion_tok‖/‖mem_tok‖` 同量级；有效数分布与 2.3 一致 |
| **S3 在线接线** | `FrameSampMemory` 绝对网格增量编码 + Wan VAE 常驻 + 尖峰处理 | 在线/离线同一起点特征一致；端到端 ms/step 实测 |
| **S4 消融** | ① 预算 N（80 / 64 / 48 / 32）+ demo 独立 stride（2.3「硬地板 51」） ② 叠加 adaRMS 调制 ③ 运动段放 img 之后 ④ `motion.stride` ⑤ 冻结 vs JAX 移植微调 ⑥ **按有效数分桶的分层评估**（2.4 后果 3） | 训练曲线 + 在线成功率 |

S1 与 S3 属「预计超过 5 分钟的全量数据构建 / 评估」，按 `AGENTS.md` 第 12、17 条从 clean HEAD
起跑并留档（`docs/dataset-build-doc/4task-gl-motion/` 与 `docs/training-doc/<run_name>/`）。
⚠ S1 规模仅 20,958 个窗口，单卡不到 1 小时，**无需上集群**。

## 六、影响面结论

| 项 | 现在 | 之后 | 增幅 |
|---|---|---|---|
| memory 段 | 512 | **592**（512 帧路 + 80 运动路） | +15.63% |
| prefix 总长 | 1088（mem 512 + img 2×256 + prompt 64） | 1168 | +7.35% |
| 全序列（含 20 个 action token） | 1108 | 1188 | +7.22% |
| attention 计算量（O(L²)） | 1108² | 1188² | **+14.96%** |
| 每样本数据字节 | 3.52 MiB（`static_*` 四键） | +240 KiB（`motion_emb` 80×768 f32） | +6.7% |
| batch=64 每批额外 | — | +15 MiB | — |
| 离线表 | — | 61.4 MiB | — |
| 新增训练参数 | — | `motion_pos_proj` 589,824 + `motion_encoder_static` 3,145,728 = **3,735,552 ≈ 3.74 M** | 可忽略 |

- **训练语义**：`motion.enabled=false` 时零影响（逐位等价，M5 判据）；`true` 时 prefix 记忆区
  512 → 592，其后所有 token 的 RoPE 位置整体右移 80。
- **冻结**：两个新投影挂在 `HistoryPi0.mem_encoder`（`PerceptualMemory`）下，路径形如
  `mem_encoder.motion_encoder_static`。当前 `HistoryPi0Config.get_freeze_filter` 返回
  `PathRegex(".*img.*")`（`paligemma_variant="gemma_2b"`，无 lora），不匹配 → **默认可训练**。
  若日后启用 lora，filters 为 `Any(All(".*llm.*", Not(".*lora.*"), Not(".*mem.*")), ".*img.*")`，
  路径含 `mem` 恰被 `Not(".*mem.*")` 排除出冻结集 → **仍可训练**。两种情形都安全。
- **数据**：新增 61.4 MiB 离线表（`v1-store/` 内，不进 git，符合第 14 条）；不动 261 GB latent。
- **在线评估**：多背一个 Wan VAE（PyTorch）常驻，延迟见 3.7。
- **不影响**：正在跑的 `v1-prod-60k` 全量 run（本计划一行代码都还没动）。

---

# 第二部分（技术细节，供 agent 追踪）

## 〇、前置声明与红线

1. **本计划只规划不实施**。S0–S4 每一步动手前须单独获批（`AGENTS.md` 第 2 条）。
2. **外部仓库锚定（起工第一件事）**：记录并写死
   - MotionJEPA 仓库 HEAD：`git -C /nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA rev-parse HEAD`
   - **checkpoint 选型待用户拍板**：候选 `runs/wan-v8-filter10-72ep-a` / `runs/wan-v8-armw01-72ep-b` /
     `runs/wan-v8-filter2-72ep-b`，各含 `checkpoint_epoch_*.pt`（单个 ~1.1 GB）。
     取 `ckpt["encoder"]`（EMA）还是 `ckpt["encoder_live"]`（live）**一并拍板**；
     `scripts/train.py` 的保存逻辑为 `has_live_weights` 时两者都存。
   - 选定后把 run 名 + epoch + `sha256` 写进 `store_meta.json` 的 `provenance` 块。
3. **归一化常数不得二次读取**。`WanLatentMotionEncoder` 继承 `LatentAffineMixin`，
   `latents_mean` / `latents_std` 是 **persistent buffer、随 checkpoint 存档**；抽取脚本必须
   `load_state_dict(..., strict=True)` 让 buffer 从 ckpt 填充，**禁止**再调
   `load_wan_latent_stats(vae_id)`（MotionJEPA 规定它是全仓库唯一读取点，二次读取会绕过
   「strict load 失败即在第一次前向炸」的保护——`normalize()` 首行的 finite 断言为此存在）。
4. **抽取口径必须与 MotionJEPA 一致**：fp32、`torch.backends.cuda.matmul.allow_tf32 = False`、
   窗口 batch 恒 1。这三条是 finalize 语义 oracle 逐位可复现的前提，抽表阶段不得放开
   （在线阶段可放开，见 3.7 与 S0）。
5. **新参数必须在所有现有模块之后创建**。`HistoryPi0.__init__` / `PerceptualMemory.__init__` 里
   `rngs` 的消耗序决定既有模块的初始化值（`datastore/README.md` 明记 `use_pos_emb` /
   `use_state_emb` 影响「`FeatureEncoder` 的参数树与 RNG 消耗序，禁改」）。运动路的两个新投影
   一律在现有 `feature_encoder` **之后**建，否则 `motion.enabled=true` 会连带改变帧路的初始化值，
   M5 等价判据失去意义。
6. **禁止 `git clean -x` / `-X`**（`AGENTS.md` 第 19 条附则），会删掉 `v1-store/` 全部产物。
7. **禁止引入两类已废弃设计**：① motion 与 framesample 采样帧一一对齐；
   ② `missing_motion_emb` + 恒 True 的 `input_mask`。
8. **容量类超参按 16 任务全集定标**。`motion.budget` 及一切随数据分布定的容量上限，一律以
   `/data/hongzefu/robomme_data_h5`（16 任务 × 100 ep）的统计定标，**不以当前 4 任务训练集**
   `robomme_data_h5_v2_4env400ep` 定标（2.3 定标原则）。改预算前必须先在全集上重跑 2.3 的起点
   统计（`num_grid = ceil((段帧数 − 32) / 20)`，样本上界 = demo + exec 两段之和），在零截断线
   之上留裕度；4env 上的统计只用于描述当前训练集的填充率实况，不作为容量依据。

## 一、离线 motion 表格式契约（S1）

新建独立 store，**不混进** `v1-store/datasets/4task-gl-framesamp/`（帧路的 `row_of()` 与运动路的
段内网格公式不同，混放会让两套索引互相污染）：

```
v1-store/datasets/4task-gl-motion/
├── meta/store_meta.json          唯一契约，两阶段写：pack→"packed"、verify→"verified"
├── meta/motion_index.json        段基址表（唯一身份来源）
├── meta/row_digests.blake2b.bin  逐行 blake2b-128（verify 产出）
├── meta/pack_progress.jsonl      断点续跑记录
└── motion_token.f32.bin          (20958, 768) f32 裸字节 = 61.4 MiB
```

布局常量（照 `datastore/framesamp_store.py` 的 `LAYOUT` 体例，新增到同包内新模块
`motion_store.py`，**不改 `framesamp_store.py`**）：

```python
LAYOUT = "motion-768-grid20-v1"
META_SCHEMA = 1
MOTION_KEY = "motion_token"
MOTION_ROW_SHAPE = (768,)
MOTION_DTYPE = np.float32
MOTION_ROW_BYTES = 768 * 4            # 3,072
MOTION_TABLE_RELPATH = "motion_token.f32.bin"
WINDOW_FRAMES = 33                    # 与 MotionJEPA 的 WINDOW 同值，verify 时核对
GRID_STRIDE = 20                      # 段内绝对网格步长（= motion.stride）
GRID_ORIGIN = "segment_start"         # 网格锚点：每段各自从段起点起算，两段互不延续
WINDOW_DIRECTION = "forward"          # 前视：窗口 = [起点, 起点+32]
```

⚠ 沿用 framesamp 的**禁 `.npy` 容器**定论（`np.save` 对 ml_dtypes bf16 写 `V2` descr），
一律裸 `.bin` + meta 声明 dtype。

**行序（写进 `store_meta.json`）**：按 `episode_manifest.json` 的 `canonical_order` 遍历 1600 个
episode，每 episode 先 `demo` 段后 `exec` 段，段内按网格序 `0, 20, 40, …` 升序。
实测行数 **20,958 = exec 17,514 + demo 3,444**。

`motion_index.json`：

```json
{"schema": 1, "grid_stride": 20, "window_frames": 33,
 "entries": [{"g": 0, "task": "ButtonUnmask", "raw_ep_idx": 0,
              "demo": {"row_base": null, "num_grid": 0, "num_chunks": 0},
              "exec": {"row_base": 0, "num_grid": 12, "num_chunks": 227}}, ...],
 "totals": {"rows": 20958, "exec_rows": 17514, "demo_rows": 3444},
 "mj_metadata_sha256": "<MotionJEPA metadata.json 的 sha256>"}
```

**查表**（`t` = 当前样本全 timestep 域帧号，`es = exec_start_idx`）：

```
exec 段： for m in range(entries[g].exec.num_grid):
              u = 20*m
              if u + 32 <= t - es:  取 row = entries[g].exec.row_base + m
demo 段： for m in range(entries[g].demo.num_grid):
              s = 20*m
              if s + 32 <= es - 1:  取 row = entries[g].demo.row_base + m
          （demo 段整段已见，该条件与 t 无关，可在 __init__ 预计算成每 episode 的定值）
合并后按起点的全局帧号升序排列，取最近 80 个（4env 最大 27、16env 最大 69，永不触发），右填充到 80
```

`num_grid = ceil(num_chunks / 20)`，即 `len(range(0, num_chunks, 20))`。

**读取实现**：表只有 61.4 MiB，**每 worker 整表 `np.fromfile` 读入进程内**即可（8 workers 也只占
491 MiB），不必走 `FrameSampStore` 的 pread 游程合并。仍照抄它的三条纪律：记录 `owner_pid`、
`__reduce__` 直接 raise 禁 pickle、跨进程懒构造。

**起点帧的 pos**：`pos_emb_4x4.f32.bin` 是按全 timestep 域 `t` 存的全表，任意起点帧都能直接查
（`FrameSampStore.pos_rows` 现成）。`pos_f` 取该帧 16 个 4×4 pos 沿空间轴的均值。

## 二、model 侧逐文件改动清单（S2）

按 `AGENTS.md` 第 9 条，以下全部用函数 / 类 / 配置键作锚点，不写行号。

### 2.1 `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-context.yaml`

新增 `motion` 节（**只加节，不动任何既有键**）：

```yaml
motion:
  enabled: false            # 总开关；false 时链路逐位等价于当前 HEAD
  dim: 768                  # = MotionJEPA config 的 motion.dim
  budget: 80                # 运动路 memory 位置数。零截断；按 16 任务全集定标
                            #   （全集最大需 69，4env 最大需 27），见红线 8
  stride: 20                # 段内绝对网格步长。⚠ 独立配置键：默认值取自当前
                            #   action_horizon=20，但**不自动跟随**——改 action_horizon
                            #   不改本键，避免离线表语义绑定到训练超参上
  window_frames: 33
  window_direction: forward # 前视：窗口 = [起点, 起点+32]，尾端 ≤ 当前帧
  grid_origin: segment_start  # demo / exec 各自从段起点起算，窗口不跨界
  store_path: v1-store/datasets/4task-gl-motion
  source_run: ???           # MotionJEPA run 名 + epoch，S1 锚定后填
  pos_source: frame_mean    # 起点帧 16 个 4×4 pos 的均值
```

已核对：`scripts/training/g0/bench_train_steps.py` 与 `scripts/training/tests/dump_fixture_samples.py`
的 `_EXPECTED_HISTORY_CONFIG` 只断言**文件名**（`"perceptual-framesamp-context.yaml"`），
不校验内容；`FrameSampDataset.__init__` 的 `_req(...)` 形制断言只查既有键的值。**加节不触发
任何现有断言**，但**必须新增对 `motion.*` 的同款 `_req` 断言**（显式 `raise`，禁 `assert`
——`PYTHONOPTIMIZE=1` 会剥离 `assert`，见该文件头部注释的 R6），至少覆盖：
`motion.dim == 768`、`motion.budget == 80`、`motion.stride >= 1`、`motion.window_frames == 33`、
`motion.window_direction == "forward"`、`motion.grid_origin == "segment_start"`。

### 2.2 `src/mme_vla_suite/models/integration/history_observation.py`

`HistAugObservation` 新增两字段（`@at.typecheck` + `@struct.dataclass` 下必须同步四处）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `motion_emb` | `at.Float[at.Array, "b l4 d4"] \| None` | `l4 = motion.budget = 80`，`d4 = 768` |
| `motion_mask` | `at.Bool[at.Array, "b l4"] \| None` | padding 位 False，语义与 `static_mask` 同款 |

同步改动：`from_dict`（`data.get(..., None)`）、`to_dict`、`from_base_obs` 形参与传递、
模块级 `preprocess_observation` 的透传（它调完基类 `_preprocess_observation` 后重建
`HistAugObservation`，漏传即静默丢特征）。

### 2.3 `src/mme_vla_suite/models/integration/history_pi0.py`

- `HistoryPi0Config.inputs_spec`：在 `with at.disable_typechecking():` 块内补两个
  `jax.ShapeDtypeStruct`——`[batch_size, motion.budget, motion.dim] float32` 与
  `[batch_size, motion.budget] bool_`。**从 config 键推导，不写死字面量。**
- `HistoryPi0.embed_memory`：现签名返回 `(tokens, input_mask, ar_mask, na_mask)`，内部调
  `self.mem_encoder(obs.static_image_emb, obs.static_pos_emb, obs.static_state_emb)`。
  改为把 `obs.motion_emb` / `obs.motion_mask` 一并传入 `PerceptualMemory.__call__`；
  返回值三处拼接——`tokens` 沿 axis=1 concat、`input_mask` 变
  `jnp.concatenate([obs.static_mask, obs.motion_mask], axis=1)`、`ar_mask` 与 `na_mask` 各追加
  `motion.budget` 个 `False`。
  **`motion.enabled=false` 时三处一个元素都不追加**，返回值与当前 HEAD 逐位相同。
- `HistoryPi0.embed_prefix`：**无需改动**——它只把 `embed_memory` 的四元组 append 进列表，
  长度变化自动透传。这是选并列拼接的直接收益。
- `HistoryPi0.compute_loss` / `sample_actions`：`integration_type == "context"` 分支不碰
  `embed_memory` 之外的东西，无需改动；`expert` / `modulation` 两分支本轮**不接 motion**
  （训练链固定 `context`，`FrameSampDataset` 形制断言已挡住其余两种）。

### 2.4 `src/mme_vla_suite/models/representation/percep_mem.py` / `mem_encoder.py`

`PerceptualMemory.__init__` 在现有 `self.feature_encoder` **之后**（红线 5）新建两件：

```python
self.motion_pos_proj       = nnx.Linear(pos.input_dim, pos.hidden_dim, rngs=rngs, dtype=dtype,
                                        kernel_init=kernel_init)          # 768 → 768
self.motion_encoder_static = nnx.Linear(motion.dim + pos.hidden_dim, memory_token_dim,
                                        rngs=rngs, dtype=dtype,
                                        kernel_init=kernel_init)          # 1536 → 2048
```

`PerceptualMemory.__call__` 现有 `assert static_image_emb.shape[1] == self.config.budget` 保留不动
（帧路仍是 512）；新增运动路分支，形制断言同款显式 `raise`：

```
motion_emb (b,80,768) ──────────────────────────────────────────┐
pos_f      (b,80,768) ── silu(motion_pos_proj(pos_f)) ──────────┼→ concat(-1) → (b,80,1536)
                                                                └→ motion_encoder_static → (b,80,2048)
```

padding 位不做特殊处理（`motion_emb` 该位为 0），屏蔽完全交给 `input_mask` —— 与帧路对
padding 帧的处理逐字同构（`FrameSampDataset._pad` 也是填 0 + `static_mask=False`）。

返回值仍为 `(hidden_states, None, None)` 三元组以保持 `embed_memory` 的解包不变，
运动段作为 `hidden_states` 的后 80 个位置拼接返回。

`mem_encoder.py` 的 `FeatureEncoder` **一字不动**——运动路不复用它（复用会共享 `use_pos_emb`
分支与参数树，破坏可退性）。

### 2.5 `src/mme_vla_suite/policies/robomme_policy.py`

`RoboMMEInputs.__call__` 的 `inputs` 字典补两键，写法与既有四个 `static_*` 键完全一致：
```python
"motion_emb":  data.get("motion_emb", None),   # (80, 768)
"motion_mask": data.get("motion_mask", None),  # (80,)
```

### 2.6 数据侧（本轮只定契约，实现归 S1/S3）

- `src/mme_vla_suite/training/framesamp_dataset.py`：`FrameSampDataset.__getitem__` 已有
  `g` 与 `step`，运动路查表**不复用** `frames`（两路独立采样，见 2.2）；起点集合按一节的公式
  现算（纯整数运算，`num_grid` 与 demo 段的合法集合可在 `__init__` 预计算）。
  `_NONE_KEYS` 尾部补空键列表加 `motion_emb` / `motion_mask` 两项；运动路的右填充**另写**
  （目标长度 `motion.budget`，不复用 `_pad`——后者的目标长度是 `_max_frames`，两者语义不同）。
- `src/mme_vla_suite/training/dataloader.py`：`_create_framesamp_dataset` 的三闸
  （`require_no_pack_lock` / `StoreMeta.load` / `require_verified`）对 motion store 照做一遍。

## 三、在线侧改动（S3）

`src/mme_vla_suite/policies/framesamp_memory.py` 的 `FrameSampMemory`：

- `__init__` 注入 `motion_enc_fn`（同 `vision_enc_fn` 的注入范式），内部持 Wan VAE + encoder。
- **新缓一份 256 域原始帧**：现有 `add_buffer` 把 `images` 经 `resize_with_pad` 成 224 后就丢了
  原图，而 Wan VAE 要 256 域。必须另存一个滚动缓冲（只需保留最近 33 帧 + 尚未编码的网格窗口）。
- **增量编码触发条件**（绝对网格的直接落地）：维护 `next_grid_start`（下一个待编起点，
  段内绝对位置，初值 0，每编完一个 `+= motion.stride`）。每步 `add_buffer` 后检查
  `next_grid_start + 32 <= 当前段内帧号`；成立则编一个窗口、存
  `_history_feats_motion[next_grid_start]`，然后 `next_grid_start += stride`。
  **每 20 帧才触发一次**，其余步零开销。
- `_prepare_frame_sampling` 之外**另加**一个 `_prepare_motion`：按一节的查表公式取合法起点、
  取最近 `motion.budget` 个、右填充 + mask。**不塞进 `_prepare_frame_sampling`**——该函数的
  数值路径（含 `right_padding_token_emb`）注释明记「只换模块、不换数值路径」，不得改动。
- `MME_VLA_Policy._prepare_history`：补 `inputs["motion_emb"]` / `inputs["motion_mask"]`。
- ⚠ 注释里那条红线仍然有效：**禁把 encode 与 pool 包进新的 `jax.jit`**（融合边界变了，
  bf16 累加序可能变位）。motion 编码走 PyTorch、在 jit 之外，天然不违反。
- **尖峰处理**（3.7 细节 1）：第 20 步的 1.57 s 尖峰若不可接受，可提前一步预编——起点的可见
  时刻 `起点 + 32` 完全可预测，可在 `起点 + 32` 到来前的空闲步里后台编好。S3 决定是否需要。

## 四、对拍闸门总表

| 闸 | 阶段 | 判据 | 失败处置 |
|---|---|---|---|
| **M0** 环境指纹 | S0 前 | 引用既有基线 run 时先过指纹 preflight（`AGENTS.md` 第 18 条末款） | 指纹不符即基线失效，重跑基线 |
| **M1** 延迟与漂移 | S0 | ms/窗口实测（fp32/关TF32 与 TF32+bf16 两档）；两档输出余弦 ≥ 阈值 | 漂移超阈值则在线也用 fp32/关TF32（1.57 s/20 步可接受） |
| **M2** 抽表逐位 | S1 | 随机 200 个 `(段, 网格序号)`，在线跑 encoder vs 表逐位相等（f32 位型 `np.array_equal`） | 任一不等即停，查 dtype / TF32 / batch 口径 |
| **M3** 索引映射 | S1 | 随机 500 个 `(g, t)`，按一节公式解出的起点集合 == 独立实现（直接遍历 metadata）解出的集合；且 `row_base + m` 读出的行 == 按 episode 名直读 `.bin` 第 `20m` 个 chunk 过 encoder 的输出 | 不等即查 `motion_index.json` 定序 |
| **M4** 行数账 | S1 | 表行数 == **20,958**；exec 17,514 + demo 3,444；逐段 `num_grid == len(range(0, num_chunks, 20))` | 不符即 metadata 与实际 `.bin` 不配套 |
| **M5** 关闭态等价 | S2 | `motion.enabled=false`，`embed_prefix` 的四个返回张量与当前 HEAD **逐位相同**（同 rng、同输入 fixture） | 不等即红线 5 被违反（RNG 消耗序变了） |
| **M6** 开启态形制 | S2 | prefix 序列长 == 1168；`ar_mask` / `na_mask` 在运动段全 False；运动段 `input_mask` == `motion_mask` | — |
| **M7** 有效数分布 | S2 | 逐 batch 统计 `motion_mask.sum(axis=1)`，分布须与 2.3 的 4env 实测一致（中位 7、均值 8.09、6.48% 全零；若训练集换成 16env，应为中位 12、均值 15.31、4.72% 全零） | 不一致即起点集合算错 |
| **M8** 尺度 | S2 | `‖motion_tok‖₂ / ‖mem_tok‖₂`（**只在 valid 位上算**）的 batch 均值落在 [0.3, 3.0] | 越界则在 `motion_encoder_static` 后补 RMSNorm |
| **M9** 梯度一致 | S2 收尾 | `motion.enabled=false` 本机跑前 N 步，逐步 loss / grad-norm / 参数摘要与既有基线一致（N 起工前商定） | 不一致即 S2 不得宣称等价 |
| **M10** 在线/离线一致 | S3 | 同一 `(g, 段, 网格序号)` 的在线编码 vs 离线表，余弦 ≥ M1 阈值；且在线解出的起点集合 == 离线解出的集合 | — |

## 五、第一块：非训练轻量对拍明细（`AGENTS.md` 第 18 条第一块）

不启动训练，四项：

1. **M5 关闭态逐位**：用 `scripts/training/tests/dump_fixture_samples.py` 现成的 fixture 机制
   dump 同一批样本，改动前后各跑一次 `embed_prefix`，比 `tokens` / `input_mask` / `ar_mask` /
   `na_mask` 四个张量的原始字节。
2. **M2/M3/M4 表与索引**：独立小脚本，不进训练环。M3 的关键是**用两个独立实现解同一个起点
   集合**（一个走 `motion_index.json` 的预计算，一个直接遍历 MotionJEPA metadata 现算），
   互为对照。
3. **逐样本内容对拍**：`FrameSampDataset.__getitem__` 改动前后，对同一批 `idx` 比全部键的
   dtype / shape / 字节——`motion.enabled=false` 时新增两键应为 `None`（走 `_NONE_KEYS`），
   其余键逐位不变。
4. **index 序列对拍**：`scripts/training/tests/dump_index_seq.py` 同款，确认 shuffle 序不受影响。

## 六、第二块：本机训练梯度一致 runbook（`AGENTS.md` 第 18 条第二块）

- 本机可跑档位启动真实训练，`motion.enabled=false` 跑前 N 步，与既有基线逐步比对
  loss / grad-norm / 参数摘要。
- **若复用既有基线 run 的固化产物**（而非同场次重跑对照侧），必须先过环境指纹 preflight，
  并在留档写明所引用基线的 `run_name`、commit 与指纹比对结论；**指纹不符即该基线失效，
  必须重跑基线后再对拍**。
- 第二块不通过不得宣称改动等价（M9）。
- `motion.enabled=true` 的梯度**不做等价对拍**——那是新语义，只做 M6/M7/M8 的形制、
  分布与尺度检查。

## 七、风险登记

| # | 风险 | 概率 | 影响 | 处置 |
|---|---|---|---|---|
| R1 | 在线延迟 | 低 | 低 | 绝对网格下摊薄 0.079 s/step；剩余的是第 20 步 1.57 s 尖峰，S3 可预编化解 |
| R2 | **填充率仅 10.1%（4env）/ 19.1%（16env），运动路信号被 padding 稀释** | **高** | **中** | 已在 2.4 显式记账；M7 盯有效数分布；S4 增「按有效数分桶的分层评估」与预算 N 消融 |
| R3 | 6.48% 样本运动路全空，模型可能学成「按有效数猜 episode 进度」的捷径 | 中 | 中 | S4 专项消融：对比「全空样本参与训练」与「全空样本的运动路整体 mask 掉」两档 |
| R4 | TF32+bf16 在线口径与 fp32 离线表漂移过大 | 中 | 低 | M1 定量；可直接退回 fp32/关TF32（每 20 步 1.57 s 仍可接受），不必冒漂移风险 |
| R5 | 新参数插入位置错误改变 RNG 消耗序 | 低 | 高 | 红线 5 明写；M5 是它的探测器 |
| R6 | MotionJEPA encoder 的已知缺陷（§3 R3：对 32×32 网格零权重共享、两种编码模式共用一个投影） | 已知 | 未知 | 该缺陷在 MotionJEPA 侧由用户拍板不动，检测靠 motion std / 余弦仪表；本计划沿用其 checkpoint，不修 |
| R7 | 在线侧多背一个 Wan VAE 的显存 | 中 | 中 | S3 实测；Wan2.1-T2V-1.3B 的 VAE 部分显存可控，但与 pi05 主干共卡需实测 |
| R8 | 窗口跨 demo/exec 边界被误判为合法 | 低 | 高——喂错数据静默训坏 | M3 专项覆盖跨界样本；`motion_index.json` 按段独立记 `row_base` / `num_grid`，使跨界在结构上不可表达 |

## 八、盲区诚实清单（写入 S1/S2 的 `result.md`）

1. **motion token 的语义未经独立验证**。它是 MotionJEPA 为「从 z0 预测未来 8 段 latent」训练出来的，
   在 VLA 里当历史运动特征用属于跨任务迁移，本计划不含对该迁移有效性的先验证据。
2. **本方案用的是前视窗口**（起点往后 33 帧），与 encoder 训练时的语义完全一致（motion 描述
   「相对锚点 z0 之后发生的运动」）；但对 VLA 而言这些窗口全部位于历史，最靠前的窗口尾端
   离当前帧 8~27 帧——**当前时刻的运动始终缺席**（2.5，用户已知并拍板不补）。
3. **`pos_source: frame_mean` 是权宜**。16 个 4×4 空间 pos 取均值得到的向量，在 `PosEmb3D`
   的 sin/cos 结构里没有明确语义（均值不是某个网格点的编码）。列为 S4 可选项。
4. **填充率 10.1%（4env）/ 19.1%（16env）的影响未经实验量化**，2.4 的三条后果都是推理不是实测。
5. **消融覆盖不全**：S4 六项互相有交互，本计划不承诺跑满全矩阵。
6. **未覆盖 `expert` / `modulation` 两种 integration_type**。

## 九、留档与 commit 纪律

- S1 属正式数据集构建 → `docs/dataset-build-doc/4task-gl-motion/`（`AGENTS.md` 第 12 条）：
  记 commit、命令、配置、数据来源、输出路径、M2–M4 判据结果；不归档 encoder 权重。
  ⚠ S1 规模仅 20,958 个窗口（读 11.5 GiB latent，单卡 <1 h），**无需上集群**。
- S2 的等价对拍 run 若超 5 分钟 → 视作完整运行，`docs/training-doc/<run_name>/`
  （launch.md / result.md / records/，第 17 条）。
- 正式 run 起跑前按第 6 条向用户确认全新 `run_name`；从 clean HEAD 起跑（第 12 条）。
- 代码切片按 `commitV6.<小版本>` 编号，文档 / 修补用 `docs:` / `fix:`；逐文件 `git add`，
  禁 `git add .` / `-A` / `commit -a`；每次 commit 后立即 `git push` 同步 origin（第 11 条）。
- 集群作业一律先读仓库根 `greatlakes.md`（第 8 条）；ssh 前必须问用户 Okta 验证方式。
