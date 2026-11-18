# 新版 Fig. 4：下一步走法（next move）线性解码，A 对 B

- 日期：2026-11-17（美东），子任务 F4。
- 数据出处：`../records/rsprog-probe-summary.json`（RouteStick 200 局专家局，5 折按局交叉验证，按局 bootstrap 2000 次）。脚本只读该 JSON，图中数字无硬编码。
- A = MotionMem（本方法），B = Frozen Wan pooling（对照）。任务进度 τ 读不出比时间更多的信息，故本图只画「下一步走法」4 分类准确率（随机水平 0.25）。

## 复现

```bash
uv run --no-project --with matplotlib --with numpy python \
  docs/evaluation-doc/rs-progress-probe-1117/fig4-next/plot_fig4_next.py
```

参数 `--summary`（JSON 路径）与 `--out`（输出目录），默认按脚本位置定位。产出 `fig4-next.png`（dpi 200）与 `fig4-next.pdf`，末行判定：

```
FIG4_NEXT=PASS a_sel=0.668 b_sel=0.552 diffs=+0.112,+0.195,+0.118
```

（`diffs` 依次为 medium、hard、xhard 的 A−B；easy 为 +0.038。）

## 面板含义

- **(a) 逐层曲线**：横轴为层，纵轴为准确率。蓝 = A，灰 = B。
  - 粗实线加阴影：记忆读出（`memmod`，MemoryAttention 输出）及 95% CI；共 18 层（0–17）。
  - 细虚线：动作 token 隐状态（`action`），18 层。
  - 黑色点线：前缀 token（`prefix_cat`，结构上看不到记忆）。A、B 两条几乎相同（选层后均为 0.38），图中画的是两者逐层均值的平均，合并为一条。
  - 橙色点划线：只用时间的基线（0.257，与随机水平 0.25 接近，故不另画随机线）。
  - 空心圆：各折选中最多的层（A 五折全为 13 层，B 五折全为 9 层）；旁边数字是训练折内选层后的准确率（A 0.67，B 0.55）。
- **(b) 分难度**：记忆读出的选层后准确率，误差线为 95% CI，每组上的橙色短线是该难度档的时间基线。柱上数字是 A−B（两个独立 bootstrap 均值之差）；蓝色加粗表示两者 95% CI 不重叠（medium、hard、xhard），灰色表示 CI 重叠（easy）。不含 p 值。

## 读图须知（如实）

- A 的记忆读出优势集中在第 13 层的一个尖峰；其余各层 A 与 B 的 CI 大多重叠，第 9 层 B 反而高于 A。
- 分难度的差值为 easy +0.04、medium +0.11、hard +0.19、xhard +0.12：hard 档最大，xhard 回落，并非随难度单调增大。
- 动作 token 上 A 0.50 对 B 0.47，差距小。
