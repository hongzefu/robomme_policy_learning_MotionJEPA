# 四任务模型同一局 motion steering 结果（steer-4task-1115）

## ① 一句话结论

在 MotionMem（四任务×400，modulation + encoder 768，step 50000）上，把 RouteStick 第 j 步示范段的 motion token 换成**同一局**只差这一步的变体 token，模型的第 j 步几乎总是跟着变：只翻绕向时 **89%** 跟随供体绕向（第 1 步 98%），只翻左右时 **85%** 走向供体那一侧按钮（第 1 步 100%），两者都翻时按钮 87%、物理弧侧 90%；对照的换采样 seed 翻转率只有 **8.3%**，置零翻转率 67%。即 MotionMem 的示范 motion token 同时携带「去哪一侧」与「怎么绕」两种信息，且对执行起决定作用。Frozen Wan pooling 与 VideoRepick 本轮按降级顺序未跑。

## ② 判定行（原文）

- 资产：`VLA_SYNC=PASS model=A files=20 sha_mismatch=0`（turbo、aspen 本地各一次）；`VLA_SYNC=PASS model=B files=25 size_bad=0 sha_mismatch=0`；`GL_RELEASE=PASS sha=6d8edc8893ef1fecdd9981b6a2c995090162ad8c`。
- 局池：`SP_SCAN=PASS task=RouteStick seeds=79`；`SP_POOL=PASS task=RouteStick eps=60 d=easy:8,medium:16,hard:16,xhard:20`；`FP_POOL=PASS task=VideoRepick eps=50`（本轮未用）。
- 同局变体：`RS_SEGLEN=PASS eps=60 variants=903 bad=0`；`RS_SPEC_REBUILD=FAIL eps=60 frame_mismatch=1`（seed 3104700 xhard 整局剔除）；`RS_VARIANT=FAIL variants=903 early_diff=21 swing_mismatch=0 skipped_endpoint=180 usable=882`（21 个 early_diff 全出自被剔除那局）。
- 冒烟：`RUN_LOCAL=PASS`（A、B 各 1 局，GL A40）；`NOOP_EQ=PASS model=A differ=0 n_replaced=11/11`。
- 供体：`SAMEP_VREC entries=941 records=882 missing=59`；`DONOR_ENC=FAIL model=motion windows=1029 bad=456 maxabs=0.03125`；`SP_SAMEP_GROUPS S_swing=265 S_side=198 S_both=201 Z=285 no_record=191 excluded=1`；`DONORS_SAMEP=PASS runs=949 specs=68`。
- 读数（`records/motion/readout.md`）：`SWAP_SCOPE=PASS runs=923 scope_bad=0 count_mismatch=0`；`NOISE_BASE=PASS model=motion pairs=448 flip_rate=0.083`；`RS_SAMEP=FAIL model=motion groups=4 invalid=51`（invalid = 08:45 停发时未跑完的 26 次干预 + 1 次 UncertainRPC + 24 次未跑的噪声对照；组数 4 = 计划 B 口径，core 组按降级未生成）。

## ③ MotionMem 不干预基线（本局池）

RouteStick 60 局：easy 8/8、medium 12/16、hard 10/16、xhard 7/20，合计 37/60（62%）。低于 692 评估（100/94/82/58%），局池为新 seed 且每档仅 8–20 局。N 可达干预位置 291/361：j=1..10 为 60/59/52/47/26/18/14/9/5/1。

## ④ 主分析：第 j 步是否跟随同局变体供体

净跟随率口径：分母 = 原 N 走到第 j 步起点的 (局, j)；干预未走到第 j 步记为非跟随；按接收局 bootstrap 2000 次 95% 区间；「有效应」= 区间下界 > max(置零翻转率, 换 seed 翻转率)（同维度）。

| 组（供体只改） | 主读数 | 第 1 步 | 第 2 步以后 | 合计 | 仍做原步 own | 未走到 |
|---|---|---|---|---|---|---|
| 只翻绕向 S_swing | 绕向跟供体 | 0.98 [0.95,1.00] n=57 | 0.87 [0.82,0.91] n=208 | **0.89** [0.85,0.93] n=265 | 0.04 | 0.05 |
| 只翻左右 S_side | 按钮跟供体 | 1.00 [1.00,1.00] n=31 | 0.81 [0.75,0.87] n=151 | **0.85** [0.79,0.90] n=182 | 0.03 | 0.06 |
| 两者都翻 S_both | 按钮 / 物理弧侧 | 1.00 / 1.00 n=31 | 0.84 / 0.88 n=170 | **0.87 / 0.90** n=201 | 0.05 | 0.06 |
| 置零 Z | 翻转率 | 0.69 n=59 | 0.66 n=216 | 0.67 n=275 | 0.28 | 0.12 |
| 换采样 seed 不干预 | 翻转率 | 0.03 n=94 | 0.10 n=354 | 0.08 n=448 | — | — |

- 三个翻转组在第 1 步与第 2 步以后两档均判「有效应」：跟随率比置零翻转（0.29–0.57）与换 seed 翻转（0.00–0.10）都高得多。
- 只翻绕向时按钮仍正确（button_follow 0.91 合计），说明换绕向 token 只改了绕法、不改去向；只翻左右时 0.85 走向另一侧，绕向标签与物理弧侧随之一致（几何上二者在去另一侧时同时翻，见 ⑥）。
- 独立抽查：S_swing 8 局第 1 步实际绕向全部等于供体方向、到达正确按钮（环境按原设定判绕向错，整局记 fail，属预期）。
- 置零：仅 28% 仍按原样完成该步，67% 行为改变，说明该段 token 不可或缺；但置零不给方向，跟随率恒 0。

## ⑤ 硬件与耗时

- GreatLakes 4 个单卡 A40 占位（66195002–66195005，8 CPU/96G）+ aspen 2×A6000（与他人共用）。美东 23:47 开工，01:55 起正式运行，08:45 停发新局。
- 局数：不干预 60、变体录制 941（落盘 882）、干预 923/949、换 seed 噪声 94/118。

## ⑥ 局限与说明

1. **只翻左右 / 两者都翻不是严格单变量**：去另一侧按钮必然绕另一根立柱、弧在桌面上的一侧随之改变，且之后路线不同；core 窗口对照本轮按降级未跑，跨界窗口的影响未单独隔离。只翻绕向是严格只差一处的组，结论以它为最干净证据。
2. `DONOR_ENC=FAIL`：原样重建局录制的示范窗编码与自然局在线编码 456/1029 窗不逐位一致，最大差 0.03125（bf16 量级）；录制与自然局可能落在不同卡型（用户「暂时不考虑不同硬件」）。差值远小于翻转组与原 token 的差别，不影响上述方向性结论，但供体 token 不是逐位的「模型自己会编出的值」。
3. 191 个（位置×组）因变体录制缺失（失败 spec 未补录）记 no_record；08:45 停发时 26 次高 j 干预未跑。
4. 第 2 步以后读数受执行误差累积影响（own/unreached 合计 ≈ 9–11%），跟随率仍在 0.81–0.88。
5. 本局池 MotionMem 成功率低于 692 评估，说明局池偏难或抽样波动；跟随率按位置统计，不依赖整局成功。
6. **未做**：Frozen Wan pooling（RouteStick、VideoRepick spec 已生成未跑）、MotionMem VideoRepick（spec 已生成未跑）、core 窗口组。均按计划 §8 降级顺序砍去，可下一轮用现成 spec 直接跑。

## ⑦ 用户决策记录

见 launch.md ②；开工令「好的，开工吧，开工吧，然后尽可能不要再阻塞了，然后有问题尽可能留到明天早上。」

## ⑧ 结论与下一步

- MotionMem 的 RouteStick 示范 motion token 逐步编码了「绕向」与「去向」，替换即可定向操纵对应步的行为（第 1 步近 100%，后续步 81–88%）。
- 下一步（需用户决定）：Frozen Wan pooling 同口径对照（判断这种可操纵性是否 MotionMem 特有）；MotionMem VideoRepick 同局交换段替换；只翻左右的 core 窗口隔离；是否追加 GL 占位。

## ⑨ 归档

- `records/motion/readout.{md,json}`：最终同局读数（读数代码 6d8edc8 `scripts/interpret/analysis/sp_swap_readout.py --same-episode --expect-groups 4`）。
- `records/spec-probe/`：episode_spec 实测脚本与输出。
- 运行产物：NFS `…-eval/v1-store/evaluation/steer-4task-1115/{rs,vr}/`；日志 `…-eval/v1-store/logs/steer-4task-1115/` 与 aspen `v1-store/logs/steer-4task-1115/`。
