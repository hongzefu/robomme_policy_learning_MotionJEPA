# X4：任务文字的词看向哪些历史帧，以及帧内空间位置

脚本：[`scripts/interpret/analysis/x4_text_spatial.py`](../../../../../scripts/interpret/analysis/x4_text_spatial.py)；数字全部来自同目录 `x4_summary.json`。
数据（只读）：`v1-store/evaluation/interpret-1103/stage1/{ctx,mot}/attn/<局>/infer_00xx.npz`，三局：

| 标签 | 模型 | 局目录 | episode | 结果 | 交换次数 k | SWAP 区间 |
|---|---|---|---|---|---|---|
| `ctx_ep4` | ctx512 基线 | `1d677dad…` | 4 | 失败 | 1 | [64,114) |
| `ctx_ep8` | ctx512 基线 | `3625db1f…` | 8 | 成功 | 2 | [64,114)、[114,164) |
| `mot_ep8` | ctx512+motion | `af10850c…` | 8 | 成功 | 2 | 同上 |

k 由 `aggregate_attn.compute_swap_times` 从元数据 seed 复算（ep4=1、ep8=2）。VIS 统一为 [0,32)。

## 一句话结论

文本词 query 在 prefill 阶段只把 5%～8% 的注意力给记忆区，而且这一点质量大多落在少数「sink token」上（#15～#19 三局 green query 帧流中占比 ≥10% 的帧，单个 token 占该帧 73.4%～99.4%）；各词（green／cube／container／table…）之间几乎没有差别，**看不到「green 这个词去找方块裸露那几帧」的词特异检索**。motion 模型 green query 的 VIS 富集（#15～#19 占帧流 37.8%，均匀 13.1%）主要来自第 0 帧：t=0 帧占帧流 25.9%～39.8%，其中单个 token 占 73.4%～98.8%；剔除每帧最大单 token 后 VIS 降到 20.8%（仍为均匀的 1.6 倍，其中 #15/#18/#19 为 14.2%/13.1%/12.1%，已与均匀 15.6%/12.5%/12.5% 持平，剩余富集来自 t=0 单 token 只占 73% 的 #16/#17）。另：motion 模型的动作 query 在 VIS 上为 17.2%（均匀 13.1%），去 sink 后不变，基线动作 query 只有 6.7%～7.0%。动作 query 在 VIS／SWAP 里看得最多的历史帧，4×4 帧内分布接近均匀（最大块只有均匀的 1.07～1.6 倍），也不对准绿方块或容器；当前图像上动作 query 集中在夹爪／机械臂附近与 base 左下角固定的几个 patch（三局同位置，疑为 sink）。

## 一、token 定位方法与表

- policy 的文本 transform 是 `src/mme_vla_suite/training/config.py::TokenizePromptWithState` + 同文件 `PaligemmaTokenizer`；本仓库 PI05 配置均 `discrete_state_input=False`，走 pi0 格式：`encode(prompt, add_bos=True) + encode("\n")`，**state 不进文本**。
- 脚本直接实例化同一个 `PaligemmaTokenizer(64)`（sentencepiece 模型读本机已有缓存 `~/.cache/openpi/big_vision/paligemma_tokenizer.model`，未联网、未写新缓存），还原出 19 个 token，与三局 npz 的 `txt_len=19` 一致（脚本里有断言），三局序列逐 id 相同。
- 文本 query 行 = prefix 文本段第 r 行（`probs_text[:, :, r, :]`）。每个词恰好 1 个 token：

| 行 | id | piece | 行 | id | piece |
|---|---|---|---|---|---|
| 0 | 2 | `<bos>` | 10 | 4788 | `▁pick` |
| 1 | 4878 | `first` | 11 | 908 | `▁up` |
| 2 | 3379 | `▁press` | 12 | 573 | `▁the` |
| 3 | 2145 | `▁both` | 13 | 11254 | `▁container` |
| 4 | 20811 | `▁buttons` | 14 | 33120 | `▁hiding` |
| 5 | 611 | `▁on` | 15 | 573 | `▁the` |
| 6 | 573 | `▁the` | 16 | 4433 | `▁green` |
| 7 | 3037 | `▁table` | 17 | 28660 | `▁cube` |
| 8 | 235269 | `,` | 18 | 108 | `\n` |
| 9 | 1492 | `▁then` | | | |

分析的 8 个词：press(2)、buttons(4)、table(7)、pick(10)、container(13)、hiding(14)、green(16)、cube(17)，外加「全部文本」（19 行平均）。

**分流方法**：对每个词，把 `probs_text` 该词行（f32）复制到全部 19 行，交给 `aggregate_attn.split_streams(query="text")`（只 import 不改），于是 split_streams 内部的 query 平均就等于该词；逐层用 `layers="<l>"`、全层用 `"all"`，头一律 8 头平均。单层单词时 f16 存盘让行和偏离 1 最多 3.8e-4，超过 split_streams 的 1e-4 断言，所以复制前逐（层,头）把行和归一到 1（只是量化修正，不改流内占比），偏差记在 `text_rowsum_max_dev_before_renorm`。
- 帧流占比 = 区间内帧（`frame_times` ∈ 区间）质量 / 帧流总质量，均匀期望 = 区间内帧数 / 有效帧数；motion 流按窗口 [t, t+33) 与区间相交计（同 `key_step_shares`）。

## 二、patch 网格核实

- **记忆帧**：`framesamp_memory.py::add_buffer` 把 SigLIP 输出 (t, v, 256, 2048) 交 `shared/data_utils.py::pool_tokens_to_size(target=16)`：`(h w)` 行优先还原 16×16，`avg_pool` 4×4 窗口、步长 4，再 `(h w)` 行优先展平。所以帧内序号 j = `mem_order % 16`，对应 4×4 网格（行 j//4、列 j%4），每块 = 256 图上 64×64 像素。
- **记忆帧视角**：`num_views=1`（`get_frame_sampling_indices` 按 `token_per_image*num_views=16` 一帧）；服务端存的 `frames/<t>.jpg` 是 256×256 单图，目视与视频左半 base 视角一致 → **记忆只存 base 视角**。
- **当前图像**：base 256 + wrist 256，各是 SigLIP 在 224×224 上的 16×16 patch（`ResizeImages(224,224)`，原图 256×256 正方，`resize_with_pad` 不留边），行优先，每块 = 256 图上 16×16 像素。
- **原图来源**：记忆帧有 `frames/<t>.jpg` 时用它（精确）；否则用视频帧（视频 512×464，上 208 行文字区，下方左 base 右 wrist）。粗核：44+31 张 jpg 与视频帧 t+d 比对 MSE，中位数在 d=0／±1 处最小（ctx8：d=−10/0/+10 → 616/534/596），所以视频帧号 ≈ 全域帧号，误差 ±1 帧以内。

## 三、问题 1：文本词看向哪些历史帧（全层平均，数字 = 窗口占比 / 均匀期望）

| 局 | 词 | VIS #15～#19 | VIS 全局 | 帧流 SWAP #15～#19 | 帧流 SWAP 全局 | motion 流 SWAP #15～#19 | 记忆区占全部 #15～#19 |
|---|---|---|---|---|---|---|---|
| ctx_ep8 | green | 2.6% / 13.1% | 19.1% / 35.9% | 24.0% / 35.6% | 51.2% / 40.0% | — | 5.9% |
| ctx_ep8 | cube | 2.7% / 13.1% | 21.9% / 35.9% | 22.5% / 35.6% | 48.1% / 40.0% | — | 5.9% |
| ctx_ep8 | container | 3.3% / 13.1% | 20.7% / 35.9% | 22.6% / 35.6% | 50.9% / 40.0% | — | 5.9% |
| ctx_ep8 | 全部文本 | 4.0% / 13.1% | 22.4% / 35.9% | 24.6% / 35.6% | 49.1% / 40.0% | — | 5.7% |
| mot_ep8 | green | 37.8% / 13.1% | 51.1% / 35.9% | 8.7% / 35.6% | 23.2% / 40.0% | 47.7% / 61.5% | 7.1% |
| mot_ep8 | cube | 34.9% / 13.1% | 50.6% / 35.9% | 16.8% / 35.6% | 26.5% / 40.0% | 51.4% / 61.5% | 7.4% |
| mot_ep8 | container | 37.6% / 13.1% | 50.7% / 35.9% | 14.2% / 35.6% | 25.3% / 40.0% | 56.6% / 61.5% | 7.7% |
| mot_ep8 | 全部文本 | 33.3% / 13.1% | 45.3% / 35.9% | 13.2% / 35.6% | 30.0% / 40.0% | 52.8% / 61.5% | 7.2% |
| ctx_ep4 | green | 5.1% / 13.1% | 19.8% / 35.9% | 7.0% / 17.5% | 29.8% / 24.4% | — | 6.1% |

全部 8 词见 `fig_x4_text_vis_swap.png` 与 json。「全局」包含 #0～#2 这种历史帧全在 [0,32) 的早期推理（占比恒为 1），均匀期望因此被抬高；SWAP「全局」只对已有 SWAP 帧的推理求平均（`n_infer_frame_swap`）。

要点：

1. **词之间几乎没有差别**。同一局里 8 个词的 VIS 占比在 ctx_ep8 #15～#19 为 1.8%～4.7%，在 mot_ep8 为 3.4%（press）与 34.9%～41.1%（其余 7 词）；green 不比 table、pick 更偏向 VIS。逐层看（`fig_x4_text_layers.png`）green 与 cube 曲线形状几乎相同：基线第 2 层 VIS 富集 2.2～3.8 倍、其余层多在 1 以下；motion 模型第 0 层 3.5～3.7 倍、第 14 层 1.6 倍。
2. **文本 query 很少看记忆**：#15～#19 记忆区只拿 5.4%～8.0%，文本自身 60%～72%，当前图像 21%～34%。motion 模型里文本给 motion 流的质量（约 5%）是给帧流（约 2%）的 2.5 倍。
3. **质量落在 sink token 上**（`fig_x4_green_frames.png` 与 json `spatial.*.top_frames`）：#18 的「green」query，mot_ep8 帧流 39.8% 在第 0 帧、39.1% 在 t=269，两帧里各有单个 token 占 97.4%／99.1%；ctx_ep8 前三帧（t=176/288/260）各约 12%，单 token 占 76.6%～98.8%；ctx_ep4 同样（86.7%～95.3%）。扩到 #15～#19（json `sink_focus`）：基线两局 top-3 帧单 token 占 76.6%～99.4%；mot_ep8 两根大柱单 token 占 73.4%～99.1%。基线 #15～#19 的帧流分布是 8～10 根等高（≈0.12）的尖柱，motion 模型是固定两根（第 0 帧 + 最近某帧）。motion 模型的 VIS「富集」主要是第 0 帧的单 token：#15～#19 中 t=0 帧占 green 帧流 25.9%～39.8%，该帧单 token 占 73.4%～98.8%，不能读成「看了方块裸露的画面」。
4. **SWAP**：帧流里两模型的文本词都低于均匀（基线 #15～#19 约 0.62～0.72 倍，motion 0.23～0.68 倍）；motion 流 SWAP 窗口 #15～#19 也低于均匀（green 47.7% / 61.5% = 0.78 倍），全局高于均匀（81.8% / 68.1%）是早期推理撑起来的。
5. ep4（失败）与 ep8（成功）的基线在文本 query 上看不出区别（#15～#19 green VIS 5.1% vs 2.6%，都远低于均匀）。
6. **去 sink 稳健对照**（每帧剔除帧内最大单 token 后重算 VIS 占比；均匀期望不变；#15～#19 平均）：

| 局 | green VIS | green 去 sink VIS | 动作 VIS | 动作 去 sink VIS | 均匀 |
|---|---|---|---|---|---|
| ctx_ep4 | 5.1% | 10.6% | 6.7% | 6.9% | 13.1% |
| ctx_ep8 | 2.6% | 6.5% | 7.0% | 7.3% | 13.1% |
| mot_ep8 | 37.8% | 20.8% | 17.2% | 17.2% | 13.1% |

去 sink 后基线 green 仍低于均匀；mot_ep8 green 从 2.9 倍降到 1.6 倍，逐次推理看 #15/#18/#19 已回到均匀附近（14.2%/13.1%/12.1%），#16/#17 仍有 31.8%/32.6%（这两次 t=0 单 token 只占 73.4%～73.9%，帧内其余 token 还留约 10% 质量）。动作 query 本来就没有 sink 集中（top 帧单 token 7%～26%），去 sink 几乎不变；motion 模型动作 query 的 VIS 1.3 倍是稳健的，基线约 0.5 倍。

## 四、问题 2：帧内空间位置（推理 #18，step 288）

动作 query = `probs_action` 10 去噪步 × 18 层 × 8 头 × 20 动作 query 全平均（`aggregate_attn.aggregate(query="action")`）。VIS／SWAP 中各取帧流质量最大的历史帧，显示其 16 token 的帧内分布（÷ 均匀 1/16）；当前 base／wrist 显示 256 patch 的分布（÷ 1/256）。叠图一律 `cv2.INTER_NEAREST` 放大 + `imshow(interpolation="nearest")`，目视已确认为整块无渐变。下排用「green」文本 query 在同一批帧上作对照。实线框 = HSV 检出的红／绿／蓝方块，白虚线框 = 白色容器（S<60、V>140、面积 ≥320 px，在夹爪里或与按钮粘连的容器会漏检）。

| 局 | VIS 选中帧（占帧流） | 动作 query 最大块 | SWAP 选中帧（占帧流） | 动作 query 最大块 | 当前 base 最大 patch／视角占 prefix | 当前 wrist 最大 patch／视角占 prefix |
|---|---|---|---|---|---|---|
| ctx_ep4 | t=9（1.6%） | (1,0) 9.8% | t=102（1.9%） | (3,3) 8.5% | (14,1) 6.9%／14.9% | (14,1) 5.1%／47.5% |
| ctx_ep8 | t=0（1.1%） | (2,3) 7.4% | t=92（3.1%） | (3,3) 10.0% | (14,1) 8.2%／14.8% | (13,14) 9.9%／36.2% |
| mot_ep8 | t=27（4.7%） | (1,0) 7.4% | t=111（3.5%） | (0,2) 7.2% | (14,1) 6.9%／12.6% | (14,1) 4.8%／39.4% |

目视判断（`fig_x4_overlay_<局>.png`）：

1. **记忆帧内几乎均匀**：动作 query 的 4×4 最大块只有均匀 6.25% 的 1.07～1.6 倍。ep8 VIS 帧里绿方块在第 1 行第 2 列（256 图 x≈175、y≈98），该块倍数 ctx_ep8 ×1.06、mot_ep8 ×1.05，不突出；红、蓝方块所在块同样接近 1。SWAP 帧里容器所在第 1 行中间两块在基线反而偏低（ctx_ep8 ×0.35／×0.44，ctx_ep4 ×0.45／×0.52），基线的最大块落在右下角桌面（×1.36～×1.59）；motion 模型 SWAP 帧最大块在顶部机械臂处（×1.15）。
2. **选中的历史帧本身权重就很小**：基线动作 query 帧流的 63%～64% 落在最新帧 t=288（≈ 当前帧），VIS／SWAP 选中帧只有 1%～3%；motion 模型帧流平铺（top 帧 4.8%），VIS 帧 t=27 与 t=250 并列最高。
3. **当前图像**：三局 base 视角都在 (14,1) 等左下角桌面 patch 和右上方机械臂／夹爪附近有高亮，前者三局同位置、画面上没有物体，疑为 SigLIP 位置性 sink；wrist 视角拿 36%～48% 的 prefix 质量，集中在被抓容器边缘和夹爪指尖。
4. 「green」文本 query 的对照：ctx_ep4 VIS 帧 t=9 里「green」最大两块 (2,1)×2.07、(2,2)×1.93 正好包住绿方块，**这是唯一一处对上绿方块的情形**，但该帧只占 green 帧流 0.2%；ctx_ep4 SWAP 帧 t=102 的 (0,2) 单块占 97.9%（机械臂处的 sink）。其余局的 green 最大块在画面底部桌面（ctx_ep8 VIS (3,1)×1.98）或左下角，与方块无关。当前 base 上 green query 最高的是夹爪与被抓容器（ctx_ep8 (5,10) 9.0%、mot_ep8 (2,12) 8.0%）。

## 五、局限

- 注意力不等于因果贡献；sink token 的大质量可能只是数值平衡，没有做值向量范数加权或消融。
- 帧流占比是「帧内 16 token 求和」，因此一个 sink token 就能决定一帧的排名；去 sink 对照只剔除每帧最大的一个 token，且只做了 green 与动作 query、#15～#19。
- 每个模型只有 ep8 一局（基线另加失败的 ep4），#15～#19 只有 5 次推理，无置信区间。
- 只用了 #18 一次推理做空间叠图；VIS／SWAP 选帧以动作 query 为准，green 行沿用同一批帧。
- 当前图像取视频帧（±1 帧），记忆帧部分取视频帧；容器 HSV 检出会漏掉在夹爪中或与按钮粘连者，SWAP 中哪个容器藏绿方块未逐帧追踪。
- motion token 是一窗一个，没有帧内空间信息，问题 2 不涉及。

## 六、文件

- `fig_x4_text_vis_swap.png`：8 词 + 全部文本，VIS／帧流 SWAP／motion 流 SWAP 占比柱与均匀期望（#15～#19、全局）。
- `fig_x4_text_layers.png`：green、cube 逐层富集倍数（#15～#19）。
- `fig_x4_green_frames.png`：「green」query 在 #15～#19 的帧流分布（ep8 两模型）。
- `fig_x4_overlay_ctx_ep4.png`、`fig_x4_overlay_ctx_ep8.png`、`fig_x4_overlay_mot_ep8.png`：#18 空间叠图（每张 8 格，共 24 格）。
- `x4_summary.json`：token 表、逐局逐词逐层逐推理占比、空间网格与 sink 诊断。

复现：

```bash
UV_PROJECT_ENVIRONMENT=/data/hongzefu/robomme_policy_learning_MotionJEPA/.venv \
PYTHONPATH=<worktree>:<worktree>/src:<worktree>/packages/openpi-client/src \
uv run --no-sync python scripts/interpret/analysis/x4_text_spatial.py
# 末行：X4_TEXT_SPATIAL=PASS words=8 overlays=24 figs=6
```
