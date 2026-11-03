# R1：attention rollout——基线「只看最近帧」是否借 prefill 间接读到 VIS／SWAP（ButtonUnmaskSwap，三局）

脚本 `scripts/interpret/analysis/r1_rollout.py`。只读 stage1b 三局的 `infer_*.npz` 与 `prefix_*.npz`（prefill 阶段 prefix 内部完整 attention），不跑仿真、不用 GPU。每局 20 次推理全算，逐次流式读、f64 计算。分流复用 `scripts/interpret/aggregate_attn.py::split_streams`（只 import），区间与 motion 窗计权沿用 X3：VIS=[0,32)，SWAP1=[64,114)，SWAP2=[114,164)（基线 ep4 只有 SWAP1）；motion 窗按与区间重叠的帧数／33 计权。全部数字在 `r1_summary.json`（严格 JSON，缺项写 null）。

**字段约定**：下文用 `E` 指 `episodes.<局>.infers.<推理序号>`，用 `B` 指 `block_diagnostics.<局>`。局名分别为 `ctx_ep8`、`ctx_ep4`、`mot_ep8`。凡写「#15～#19 平均」的数字，都是对 `E`（n=15..19）里对应字段取算术平均；百分比是字段值乘以 100。

判定行（脚本末尾打印）：
- `ROW_SUM_CHECK max_err=2.44e-15 tol=0.0001`（字段 `row_err_max`）。R 按构造就是行归一的，这项检查只验证数值误差，**不验证层对齐**（见 `row_check_note`）。
- `C0_ALIGN_CHECK max_err=0.00e+00 tol=1e-12`（字段 `c0_err_max`）。这是层对齐的单元式自检：R_0 = I，所以 c_0 的 prefix 部分应该逐位等于 a_0 的 prefix 部分。
- `BLOCK_CLOSURE_CHECK=PASS mem_to_imgtxt_max=0.00e+00 img_to_mem_max=0.00e+00`（字段 `B.all_infers_mem_to_imgtxt_max`、`B.all_infers_img_to_mem_max`，阈值 `block_tol`=1e-6）。脚本对每次推理都做了断言。
- `R1_ROLLOUT=PASS episodes=3 figs=5`

| 局 | 目录 |
|---|---|
| 基线 ctx512 ep8（成功） | `stage1b/ctx/attn/0621439bab554595b3ee0f0f9414ba84` |
| 基线 ctx512 ep4（失败，只交换 1 次） | `stage1b/ctx/attn/3a79ca2d563c4a679d51013e15394130` |
| ctx512+motion ep8（成功） | `stage1b/mot/attn/afb7438311e44dfab40a03e709c2ab16` |

## 一句话结论
**在这个 attention rollout 近似下，看不到基线经 prefill 间接读取 VIS／SWAP。** 这个近似会过平滑，并且忽略了 MLP 和残差范数，所以不能排除信息走了其他通路。

把间接通路算进来以后，基线记忆区的有效质量确实变大了：#18 帧流从直接 attention 的 0.9% 涨到 8.4%（`E.direct.frame_total` 与 `E.rollout.frame_total`，ctx_ep8 n=18）。但多出来的质量几乎都落在**近期帧的单个 sink token** 上：#15～#19 rollout 帧流里最大的那一帧占 21～31%，该帧有 51～98% 的质量来自一个 token（`E.rollout.top_frame_share` 与 `E.rollout.top_frame_single_token`）。

VIS 的有效占比反而更低。#15～#19 平均，基线 ep8 从直接的 7.0% 降到 rollout 的 1.7%，均匀期望是 13.1%（`E.direct.vis_frame_share`、`E.rollout.vis_frame_share`、`E.direct.vis_frame_uniform`）。去掉每帧最大的 token 后是 6.2%（`E.rollout.vis_frame_share_desink`），仍然低于均匀。只走记忆区内部混合的那条通路，VIS 是 6.0%（`E.via.mem.vis_frame_share`），也低于均匀。

motion 模型的同一条记忆区内部通路，VIS 是 21.0%，高于均匀的 13.1%。

## 1. 口径与假设
- **有效位置**：取记忆区中 `mem_mask` 为真的位置、当前图像的 512 个位置，以及前 `txt_len` 个文本位，padding 的行和列都剔除。prefix 里 padding query 的行全是 0。#18 每层的 padding 行数：基线为 45，motion 为 132（`B.focus_n_pad_rows`）。全部推理、全部层的零行数都等于预期的 padding 行数（`B.all_infers_zero_rows_eq_pad`=true，所出现的取值见 `B.all_infers_zero_rows_set`；早期推理的记忆区未填满，padding 更多）。
- **每层图**：先对 8 个头取平均，得到 `probs_prefix_l[V,V]`，行归一后记作 A_l。令 Ã_l = 0.5·A_l + 0.5·I，再做一次行归一。递推为 R_0 = I、R_{l+1} = Ã_l·R_l。R_l[k,j] 表示第 l 层输入在位置 k 上，对原始输入 token j 的近似贡献。
- **动作有效贡献与层对齐**：动作在第 l 层读的是 prefill 第 l 层的 KV，也就是第 l 层的输入表示，所以 c_l[j] = Σ_k a_l[k]·R_l[k,j]。a_l 是第 l 层动作 query 对 prefix 的 attention，在 10 个去噪步、20 个动作 query、8 个头上取平均。代码在循环里**先用当前 R（即 R_l）算 c_l，再把 R 更新成 R_{l+1}**；`C0_ALIGN_CHECK` 验证了 l=0 时 c_0 等于 a_0。最终的 c 是 18 层的平均，逐层的 c_l 统计在 `E.layers`。动作对动作 key 的质量原样留在 action 流，所以 c 的总和仍为 1。
- **直接版**：d = 18 层 a_l 的平均，与 `split_streams` 的全层全头平均一致。
- **联合 rollout（补充）**：把 20 个动作 token 也放进图里。第 l 层的动作行是 0.5·[a_l 对 prefix, a_l 对动作] + 0.5·I。取第 18 层之后 20 个动作行的平均，这样能包含「动作→动作→prefix」的路径（`E.joint.*`）。
- **中介分解（对照）**：把 c_l[j] 按中介 k 的类别拆成三份：经记忆区、经当前图像、经文本（`E.via.{mem,image,text}`）。记忆区的有效行对图像和文本的质量为 0（`B.all_infers_mem_to_imgtxt_max`=0，脚本已断言），所以记忆子块在矩阵乘法下封闭，「经记忆区」这一份就等于「只用记忆区子块做的 rollout」。
- **去 sink 对照**（与 X4 同口径）：每帧剔除帧内最大的单个 token，再重算帧流的 VIS／SWAP 占比，均匀期望不变（`E.*.vis_frame_share_desink`、`E.*.swap_frame_share_desink`）。
- **假设与近似**：
  1. 残差按 0.5 恒等近似，实际残差与注意力输出的范数比未知。
  2. 忽略 MLP 与 LayerNorm 的混合。
  3. 头取平均，丢掉了头之间的差异。
  4. 没有按值向量范数加权，attention 质量不等于信息量。
  5. 去噪步和 query 都取了平均。

  rollout 是对「信息可能从哪里来」的近似估计，不是因果贡献。
- **实测可见关系的补充**：
  - 当前图像确实看不到记忆区：全部推理、全部层的 img→mem 最大行质量都是 0（`B.all_infers_img_to_mem_max`，脚本已断言）。
  - 但当前图像**能看文本**。#18 每层图像行对文本的平均质量为 7～75%（`B.focus_img_to_txt_mean_range`：ctx_ep8 为 [0.068, 0.751]，mot_ep8 为 [0.078, 0.761]）；全部推理的范围是 5～88%（`B.all_infers_img_to_txt_mean_range`）。
  - 文本在第 0、1 层把大量 attention 给了记忆区。#18 的数值：基线 ep8 为 72%／17%，ep4 为 71%／22%，motion 为 60%／56%（`B.focus_txt_to_mem_layer0`、`B.focus_txt_to_mem_layer1`）。第 2 层以后最大只有 4.7%／5.9%／4.1%（`B.focus_txt_to_mem_layer2plus_max`）。
  - 因此存在「图像→文本→记忆区」的两跳通路，下面「经图像中介」的质量就从这里来。

## 2. 直接 attention 与 rollout 对比（fig_r1_1、fig_r1_2）
帧流占比，单位是占帧流的百分比；表中先列 #15～#19 平均，括号里是 #18。字段为 `E.{direct,rollout,joint}.vis_frame_share` 与 `swap_frame_share`，去 sink 列取 `E.rollout.*_desink`，均匀列取 `E.direct.*_uniform`。

| 局 | VIS 直接 | VIS rollout | VIS 联合 | VIS rollout 去 sink | VIS 均匀 | SWAP 直接 | SWAP rollout | SWAP rollout 去 sink | SWAP 均匀 |
|---|---|---|---|---|---|---|---|---|---|
| 基线 ep8 | 7.0（3.6） | 1.7（1.8） | 2.3（2.8） | 6.2（5.8） | 13.1（12.5） | 22.7（13.5） | 13.0（12.0） | 25.0（22.4） | 35.6（34.4） |
| 基线 ep4 | 6.7（5.2） | 1.8（1.5） | 2.3（2.4） | 6.5（5.6） | 13.1（12.5） | 11.2（7.0） | 4.1（2.8） | 4.1（5.3） | 17.5（18.8） |
| motion ep8 | 17.2（17.0） | 5.9（4.6） | 9.2（7.0） | 14.0（20.7） | 13.1（12.5） | 33.8（30.6） | 6.5（4.7） | 14.6（20.0） | 35.6（34.4） |

- **记忆区有效质量放大约 7～10 倍**（`E.{direct,rollout}.frame_total`，motion 另加 `motion_total`）。基线 ep8 #18 的帧流从直接的 0.9% 变成 rollout 的 8.4%。#15～#19 三局 rollout 的记忆区质量为 5～10%，直接 attention 只有 0.2～2.2%。
- **放大出来的质量去了 sink**（`E.rollout.top_frame_t`、`top_frame_share`、`top_frame_single_token`）。#15～#19 rollout 帧流的最大帧都是近期帧：
  - 基线 ep8：t=224/239/254/278/284，占帧流 30.7/27.4/25.7/21.5/21.5%，该帧最大单 token 占 94.0/51.2/86.1/89.3/56.2%。
  - 基线 ep4：t=224/239/254/269/294，占 30.9～20.2%，单 token 占 84.4～98.3%。
  - motion ep8：t=232/247/263/278/294，占 68.2/76.9/79.9/78.2/63.5%，单 token 占 51.1～98.6%。

  逐帧数值见 `E.frames`（只存 n=18）。在 fig_r1_2 中，基线 #18 的 rollout 在 t=269、278 各约 21%，motion 在 t=278 达 78%。直接 attention 里，基线的最大帧是 t=288，也就是当前步那一帧，占帧流 64%（`E.direct.top_frame_t`／`top_frame_share`）。
- **去 sink 后基线仍低于均匀**：VIS 为 6.2%／6.5%，均匀 13.1%，与直接 attention 的 7.0%／6.7% 相近。motion 模型去 sink 后 VIS 为 14.0%，约等于均匀；它在直接 attention 下的 17.2% 是均匀的 1.3 倍。
- **整局曲线**（`E.*.swap1_frame_share`、`vis_frame_share`，n=5..10 取平均）：
  - 早期 #5～#10 基线两局对 SWAP1 帧的偏好，在 rollout 下保持不变：基线 ep8 直接 58.3%、rollout 58.9%；ep4 为 56.7%、57.4%；均匀 33.9%。
  - 这一阶段的 sink 帧本身就落在 SWAP1 内（例如 #7 的 sink 在 t=104）。去 sink 后，基线 ep8 #5～#10 的 SWAP 仍有 38.8～89.9%（`E.rollout.swap_frame_share_desink`），所以早期对第 1 次交换帧的偏好在 rollout 下同样成立。
  - #11 之后，各条曲线回落到均匀以下。
  - motion 模型 #5～#10 的 rollout VIS 为 14.6%，低于它直接 attention 的 38.9%（均匀 28.6%）。
- **motion 流（motion ep8，`E.*.vis_motion_share`、`swap_motion_share`、`*_motion_uniform`）**：rollout 后，#18 的 motion 流总质量从 0.3% 涨到 2.0%。按窗计权，VIS 从直接的 8.2% 变为 rollout 的 4.3%（均匀 10.8%），SWAP 从 30.5% 变为 22.3%（均匀 35.0%）。两种算法都没有高于均匀。
- 联合 rollout（含动作→动作的路径）与层平均 rollout 结论一致，数值略高。例如 motion 的 VIS 为 9.2%，层平均 rollout 为 5.9%。

## 3. 间接通路主要走哪类 token（fig_r1_3、fig_r1_5）
下表为 #15～#19 平均：rollout 落在记忆区的质量（占动作 attention 的百分比，`E.via.<中介>.mem_mass`），以及各中介对应的帧流 VIS／SWAP 占比（`E.via.<中介>.vis_frame_share`／`swap_frame_share`）。

| 局 | 经记忆区：质量 | 经文本：质量 | 经图像：质量 | 经记忆区：VIS／SWAP | 经文本：VIS／SWAP | 经图像：VIS／SWAP | 均匀 VIS／SWAP |
|---|---|---|---|---|---|---|---|
| 基线 ep8 | 0.59 | 3.05 | 3.02 | 6.0／28.7 | 1.5／11.8 | 1.1／11.5 | 13.1／35.6 |
| 基线 ep4 | 0.62 | 2.98 | 3.03 | 6.0／12.1 | 1.5／3.4 | 1.2／3.2 | 13.1／17.5 |
| motion ep8 | 1.90 | 3.28 | 2.98 | 21.0／25.9 | 3.4／3.5 | 3.4／2.9 | 13.1／35.6 |

- **间接质量大部分走文本或图像（图像→文本→记忆）**。按表中质量列算：基线 ep8 为 (3.05+3.02)/6.66≈91%，motion 为 6.26/8.16≈77%。其余部分是「动作直接读记忆 token，再经记忆区内部混合」。motion 模型经记忆区的份额更大（1.90 对 0.6）。
- 经文本或图像过来的质量几乎全落在 sink 帧上，VIS 只占 1.1～3.4%。
- **只用记忆区子块（即经记忆区通路）**：基线两局 VIS 为 6.0%，低于均匀 13.1%，与直接 attention 持平。motion ep8 为 21.0%，是均匀的 1.6 倍，高于它直接 attention 的 17.2%。也就是说，motion 模型的记忆区内部混合确实把 VIS 信息往动作能读到的位置汇集，基线没有。
- **末层记忆区 R 的形态**（fig_r1_5，#18，字段 `E.mem_R`，n=18）：
  - 18 层之后，记忆区各位置的来源分布几乎相同。帧级行间最大绝对差：基线 ep8 为 9.4e-5、ep4 为 1.0e-4、motion 为 1.8e-5（`row_absdiff_max`）。对比单列最大质量 0.117／0.156／0.031（`col_mass_max`），行间差别不到它的 0.1%；行间标准差最大为 5.2e-5／5.1e-5／6.7e-6（`row_std_max`）。帧级对角均值为 0.031／0.031／0.013（`self_frame`），基线恰好等于 1/32。这是典型的 rollout 过平滑。
  - 基线 ep8 的来源前三帧是 t=185（11.7%）、288（9.1%）、176（8.9%）；ep4 是 t=288（15.6%）、185（10.3%）、195（9.9%）（`top_source_frames`）。VIS 4 帧合计：基线 ep8 为 5.3%，ep4 为 5.5%，均匀 12.5%（`vis_cols_sum`，基线 `frame_part_total`=1）。
  - motion ep8 帧 token 的行有 **59.8% 的质量落到仅 9 个有效 motion token 上**（`frame_rows_to_motion`、`n_motion_tokens`），motion token 在记忆区内部充当汇聚点。帧部分只剩 40.2%（`frame_part_total`），其中 VIS 占 17.8%（`vis_share_in_frame_part`），高于均匀。

## 4. 逐层（fig_r1_4，#18，字段 `E.layers[l]`，n=18）
- 直接 a_l 的帧流质量集中在第 15 层：基线 ep8 为该层动作 attention 的 10.3%，ep4 为 8.5%（`direct_frame_total`）。这一层的 VIS 只有 0.2～0.3%（`direct_vis`），几乎全给了当前步那一帧，「只看最近帧」主要来自这一层。motion ep8 的帧流质量峰值在第 16 层（7.1%），VIS 为 22%。
- rollout 的 c_l 从第 1 层起 VIS 就只有 0.2～2%（`rollout_vis`），说明早层文本→记忆的 sink 已经决定了大部分有效来源。往后 VIS 缓慢回升，基线到 3～4%，motion 到 7～11%。

## 关键发现
1. **在这个 attention rollout 近似下，看不到基线「只看最近帧」借 prefill 间接读取 VIS／SWAP。** 这个近似会过平滑，并且忽略了 MLP 和残差范数，不能排除信息走了其他通路。#15～#19 rollout 的 VIS 为 1.7%／1.8%，去 sink 后为 6.2%／6.5%，都低于均匀 13.1%。基线 ep8 的 SWAP 为 13.0%，去 sink 后 25.0%，低于均匀 35.6%。
2. 间接通路把记忆区的有效质量放大了 7～10 倍（#18 基线 ep8：0.9% → 8.4%），但大部分质量经文本或图像中介（基线约 91%，motion 约 77%），最终落在近期帧的单个 sink token 上（单 token 占该帧 51～98%）。这更像位置性的 sink，而不是内容检索。
3. 只看记忆区内部混合那部分：基线 VIS 是 6.0%，低于均匀；motion 模型是 21.0%，为均匀的 1.6 倍。motion 模型对 VIS 的偏好在记忆区内部通路上依然存在，而且比直接 attention 更强。
4. 早期 #5～#10，基线对第 1 次交换帧的偏好在 rollout 下保持（SWAP1 58.9%，均匀 33.9%）。也就是说，基线确实「看过」交换，只是到 #15～#19 抬起阶段不再回看。
5. motion 模型记忆区的帧 token，在末层约有 60% 的 rollout 来源是 9 个 motion token，motion token 在记忆区内部是汇聚点。但动作端最终读到的 motion 流仍不高于均匀（#18 SWAP 22.3%，均匀 35.0%）。

## 局限
- 只有 3 局（2 局 ep8 成功、1 局 ep4 失败），且是单 seed。attention 和 rollout 都不是因果贡献，没有做消融或梯度归因。
- 0.5 残差、忽略 MLP、头平均、不按值范数加权，都是近似。多层连乘会过平滑（末层记忆区 R 的行间几乎相同），还会放大 sink。所以「rollout 占比低」只能说明在这个近似下看不到间接读取，不能证明信息完全没有走别的通道，例如 MLP，或残差流中不经 attention 的部分。
- 去 sink 只剔除每帧最大的一个 token，与 X4 相同，是粗粒度的稳健性对照。
- 文中的分段占比都是帧流或 motion 流内部的比例。rollout 落在记忆区的总质量只有 5～10%，主体仍是当前图像（24～39%）、文本（23～39%）和动作自身（`E.rollout.image`、`text`）。

## 图
- `fig_r1_1_share_curves.png`：三局，每局三栏（帧流 VIS、帧流 SWAP、motion 流），按推理序号画直接 attention、rollout、联合 rollout、rollout 去 sink 和均匀期望的曲线，灰底为 #15～#19。
- `fig_r1_2_focus18_frames.png`：#18 帧流按全域帧号画柱图，直接与 rollout 并排（帧流内归一，绿底为 VIS，黄底为 SWAP）。
- `fig_r1_3_routing.png`：rollout 落在记忆区的质量按中介类别堆叠，以及各中介通路的 VIS 占比。
- `fig_r1_4_layers_focus18.png`：#18 逐层的直接 a_l 与 rollout c_l 帧流 VIS 占比，灰柱为直接 a_l 的帧流总质量。
- `fig_r1_5_mem_R_focus18.png`：#18 末层 R 的记忆区帧×帧热图（`interpolation="nearest"`）。

五张图都已渲染，并用图片读取逐张目视复查，没有重叠、截断或缺字。初版有三张图需要改，已修正后重出：
- fig_r1_3：图例压住柱子，已调高 y 轴上限，并把图例改为两列。
- fig_r1_1：motion 栏因 #0 没有 motion 窗，被误判为「无 motion 流」；图例压住曲线；motion 栏标题被截断。已修正判定、上移图例并缩短标题。
- fig_r1_5：基线标题多写了「0 个 motion token」，已去掉。

本轮续改只增加了 json 字段与断言，五张 PNG 与上一版逐字节相同（md5 一致）。
