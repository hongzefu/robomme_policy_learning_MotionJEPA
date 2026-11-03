# X1：回看裸露窗与交换区间的是哪几层、哪几个头

## 问题

ButtonUnmaskSwap 上，pi0.5 动作专家在去噪时回看记忆区的「裸露窗 VIS=[0,32)」（唯一能看到绿方块位置的时刻）和「交换区间 SWAP=[64,64+50k)」，分别由哪些 (层, 头) 承担？有没有专门的「记忆头」？

## 数据与做法

- 三局原始 attention（只读）：ctx512 基线 ep4（失败，k=1）、ctx512 基线 ep8（成功，k=2）、ctx512+motion ep8（成功，k=2），每局 20 次推理；关键推理 #18（step 288，「揭开」）。
- 对每次推理的 `probs_action` (10 去噪步, 18 层, 8 头, 20 动作 query, P+20 key)，先在全部去噪步与 20 个动作 query 上平均，得到每 (层, 头) 一条 key 向量，再原样交给 `scripts/interpret/aggregate_attn.py::split_streams` 分流（复用计划 §2.3 映射与 pad／求和断言）。
- 每 (层, 头) 的指标：
  - 帧流质量 M_f、motion 流质量 M_m（占该头全部注意力的比例）；
  - 帧流 VIS 富集 = (VIS 帧质量 / M_f) ÷ (VIS 帧数 / 有效帧数)，#18 时 VIS 帧 4/32，均匀期望 12.5%；帧流 SWAP 富集同理；
  - motion 流 SWAP 富集：窗 w 权重 o_w = |[t_w, t_w+33) ∩ SWAP| / 33（按重叠帧数加权，不是沾边整窗算），#18 均匀期望 35.0%（原 fig_key_step 按沾边计为 55.6%）；
  - 排序指标「超额质量」= M × (占比 − 均匀期望)：该头全部注意力中「比均匀多投给区间」的部分。它是富集倍数与质量的乘积型指标，富集 = 1 时为 0、质量极小时自动趋零。流质量 < 1% 的头在热图上加斜线，富集倍数不解读。
- 口径核对：把 144 个头按质量加权合回，#18 帧流 VIS 占比基线 3.57%、motion 17.05%，与 `result.md` 的 KEY_STEP 行（0.0357／0.1705）一致。

复现命令（worktree 根目录）：

```bash
R=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/interpret-1103/stage1
UV_PROJECT_ENVIRONMENT=/data/hongzefu/robomme_policy_learning_MotionJEPA/.venv PYTHONPATH=$PWD:$PWD/src \
uv run --no-sync python scripts/interpret/analysis/x1_layer_head.py \
  --ep "base_ep4=$R/ctx/attn/1d677dadd3f34872a5779dbe8ab815e4:1:ctx512 基线 ep4（失败）" \
  --ep "base_ep8=$R/ctx/attn/3625db1fb81a48af8aa5a136b3a7aa68:2:ctx512 基线 ep8" \
  --ep "mot_ep8=$R/mot/attn/af10850c1f7d47549fcd8934178a7497:2:ctx512+motion ep8" \
  --out docs/evaluation-doc/attn-ctx512-vs-motion-1103/analysis/x1
# 末行：X1_LAYER_HEAD=PASS models=2 episodes=3 figs=4
```

## 关键数字

**记忆注意力集中在极少数头上。** #18 时基线 ep8 有 133/144 个头的帧流质量 < 1%，motion 模型为 109/144；记忆质量（帧流 + motion 流）≥ 0.3 的头在三局中各只有 1 个。

**基线的「记忆头」L15H1 看的是当前帧，不是记忆。** 基线 ep8 #18 的 L15H1 把 75.9% 的注意力放在记忆区，其中 99.8% 落在最近 4 帧（峰值 t=288，也就是当前步），VIS 富集 0.002、SWAP 富集 0.001。第二大的 L16H0（13.7%）SWAP 富集 1.31、VIS 富集 0.33，峰值在 t=288 和 139。这两个头在 20 次推理中有 14 次都排在记忆质量前 5，是固定的结构性头。

**基线回看 VIS 的是一批低质量、浅中层的头，总量很小。** ep8 #18 的 VIS 超额质量 top-10 是 L5H3、L7H4、L3H3、L3H6、L2H0、L0H3、L5H5、L4H3、L6H4、L7H6，富集 1.7～4.0 倍，但帧流质量只有 0.2%～1.1%，单头超额质量 ≤ 0.0021。整体加权后基线 #18 帧流 VIS 占比只有 3.6%，低于均匀 12.5%。L7H4、L5H3、L3H3 在 #15～#19 五次推理里 5 次都在 top-10，20 次里有 17～18 次在。

**motion 模型出现了大质量的 VIS 头。** ctx512+motion ep8 #18：L16H0 的记忆质量 0.498（帧 0.464、motion 0.034），VIS 富集 1.81，超额质量 0.047；L3H0 帧流质量 0.110，VIS 富集 3.65，超额质量 0.037，峰值帧 t=0、9、27，全在裸露窗内。这两个头占全部正 VIS 超额质量的约 69%（top-10 合计占 86%）。L3H0 在 #15～#19 五次全在 top-10，20 次里 18 次在。L16H0 的 VIS 超额在 #8～#11 最高（约 0.31～0.44），#17 跌到富集 0.12，#18 又回到 1.81，不平稳。

**两模型 VIS 头基本不是同一批。** ep8 #18 两模型每头 VIS 超额质量的 Pearson r = 0.091，top-10 只有 L4H3、L7H4 两个重合。同一个 (层, 头) 编号在 motion 模型里承担的角色已经不同（例如 L16H0：基线是偏 SWAP／当前帧，motion 模型是 VIS）。

**motion 流对 SWAP 几乎没有专门头。** motion 流质量最大的头也只有 0.034（L16H0）；#18 时 134/144 个头 motion 流质量 < 1%。按重叠加权后，全头 motion 流 SWAP 占比 30.5%，低于均匀 35.0%。SWAP 超额 top 头（L15H7 富集 1.73、L7H4 1.70、L15H1 1.34）的超额质量都 ≤ 0.0032，排名逐次推理变动大（L15H7 在 20 次里只有 7 次进 top-10）。帧流 SWAP 方面，L14H3（帧 0.182，富集 1.26，超额 0.016）最大，峰值帧 t=111～148，落在交换期内。

**基线失败局 ep4 与成功局 ep8 的头结构几乎一样。** 整局平均每头 VIS 超额质量 ep4 对 ep8 的 Pearson r = 0.996，log2 富集 r = 0.988，top-10 VIS 头 10/10 重合，逐推理曲线基本重叠。所以基线哪几个头回看 VIS 是模型的固定属性，看不出和这一局成败有关。

## 结论

1. 「有没有专门的记忆头」：两个模型都有极少数大质量记忆头（基线 L15H1／L16H0，motion 模型 L16H0／L14H3／L15H1／L3H0），集中在 14～16 层（另有一个 3 层头）。但基线里最大的 L15H1 实际上是「当前帧头」：几乎只看最近 4 帧。
2. 回看 VIS：基线靠 0～7 层若干低质量头，富集 2～4 倍，但总量太小，整体合起来 VIS 反而低于均匀。motion 模型新出现两个大质量 VIS 头 L16H0（深层）和 L3H0（浅层），整体 VIS 占比因此升到 17%。
3. 回看 SWAP：帧流里有中深层头（基线 L16H0，motion 模型 L14H3）对交换期有 1.3 倍左右的富集；motion 流对 SWAP 没有稳定的专门头。
4. 基线失败局与成功局的头结构一致，所以基线成败的差别不在「哪个头看 VIS」。

## 图

- `fig_x1_heat_ep8_infer18.png`：两模型 ep8 #18 的 18×8 热图，五列依次为帧流质量、帧流 VIS 富集、帧流 SWAP 富集、motion 流质量、motion 流 SWAP 富集。最近邻显示，斜线表示流质量 < 1%。
- `fig_x1_memory_heads_infer18.png`：两模型 #18 记忆质量 top-3 头在各帧、各 motion 窗上的注意力分布。
- `fig_x1_stability_ep8.png`：top-10 头在 20 次推理中的富集倍数与超额质量曲线，#15～#19 橙底，#18 虚线；依次为基线帧流 VIS、motion 帧流 VIS、motion 流 SWAP、基线帧流 SWAP。
- `fig_x1_base_ep4_vs_ep8.png`：基线 ep4 与 ep8 同头散点（超额质量用 symlog 轴；log2 富集），以及 top-5 头逐推理对比。
- `x1_summary.json`：全部 top-10 清单（#18 与整局平均）、稳定性计数、记忆头画像、相关系数。

四张图都已渲染成 PNG 目视复查，没有文字重叠、截断或字体缺失。

## 局限

- 每个模型只有一局（ep8），基线另加一局 ep4；结论都是单局观察，不能推广成统计规律。
- 注意力只说明「看了哪里」，不代表因果，也不说明信息真的被用上（没有做消融或 patch）。
- 全去噪步、全 20 个动作 query 平均会掩盖步间与 query 间的差异；pi0.5（Gemma）是多 query 头共享 KV，头之间不是完全独立的读取通道。
- 前两次推理（#0、#1）全部帧都在 VIS 内，富集无定义，记为 NaN；step < 64 时 SWAP 也无定义。
- 1% 低质量阈值是人为设定；超额质量排序会偏向大质量头，所以热图同时给出富集倍数，便于看低质量但高富集的头。
