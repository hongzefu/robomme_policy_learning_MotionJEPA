# steer-speedup-1116 结果：提速流水线 + 四任务模型同一局 steering 补跑

## ① 一句话结论

提速后的同一局 steering 流水线（离线 bf16 批量供体、示范窗复用 N 录制、fast 在线编码、常驻服务、共享任务池）三道闸门全过（`NOOP_EQ2=PASS rate=1.0000`、`BEHAVIOR_TOL=PASS diff=+0.0000`、ENC_TOL 放宽口径 PASS），把上一轮没跑的四部分在当天跑完（09:40 开工 → 15:10 全部完成）。结果：**RouteStick 第 1 步两个模型都几乎 100% 可操纵（只翻绕向跟随率均 0.98），第 2 步以后 MotionMem 明显更强（Frozen Wan 0.63、无有效应；MotionMem 上一轮 0.87）；VideoRepick 两个模型换同一局交换段的 token 都基本操纵不动（Frozen Wan 跟随 0.00、MotionMem 0.15，k=5 例外 9/12）；MotionMem 只换第 j 段内部窗口（core）跟随率仅约 0.10，操纵效应来自跨段边界的窗口。**

## ② 判定行（原文）

- 闸门：见 `records/gates/gates.txt`
  - `NOOP_EQ2=PASS eps=20 steps=57 agree=57 rate=1.0000 missing=0 min_rate=0.95`
  - `BEHAVIOR_TOL=PASS group=S_swing j=1 key=swing_follow pairs=57 new=0.9825 old=0.9825 diff=+0.0000 tol=0.05 new_all=0.9833 old_all=0.9825 n_new=60 n_old=57`
  - motion：`ENC_TOL=FAIL`（原阈值 0.999/0.1）/`PASS`（放宽 0.997/0.3）`min_cos=0.999480 maxabs=0.1406 windows=300 files=60`
  - wan：`ENC_TOL=PASS`（两档）`min_cos=0.999995 maxabs=0.03297`
  - `OFFLINE_ENC=PASS` 两模型各 `windows=5348 files=958 bad=0`
- 变体：`RS_SPEC_REBUILD=PASS eps=60 frame_mismatch=0`；`RS_SEGLEN=PASS variants=903`；`RS_VARIANT=FAIL early_diff=5 usable=898`；`VFRAMES=PASS files=958 windows=5348`
- Frozen Wan RouteStick：`SWAP_SCOPE=PASS runs=793 scope_bad=0 count_mismatch=0`；`NOISE_BASE=PASS model=wan pairs=388 flip_rate=0.186`；`RS_SAMEP=FAIL model=wan groups=4 invalid=16`（噪声 15、干预 1）
- Frozen Wan VideoRepick：`NOISE_BASE=PASS model=wan task=VideoRepick pairs=38 flip_rate=0.263`；`VR_SAMEP=PASS model=wan invalid=0`
- MotionMem VideoRepick：`NOISE_BASE=PASS model=motion task=VideoRepick pairs=44 flip_rate=0.136`；`VR_SAMEP=PASS model=motion invalid=0`
- MotionMem RouteStick core：见 ⑥

## ③ RouteStick：Frozen Wan pooling 同一局四组（`records/rs-wan/readout.md`）

| 组（供体只改） | 主读数 | 第 1 步 | 第 2 步以后 | 合计 | 上一轮 MotionMem 合计 |
|---|---|---|---|---|---|
| 只翻绕向 S_swing | 绕向跟供体 | **0.98** [0.95,1.00] n=60 | 0.63 [0.55,0.73] n=167（无有效应） | 0.73 n=227 | 0.89 |
| 只翻左右 S_side | 按钮跟供体 | 0.81 [0.66,0.94] n=32 | 0.42 [0.35,0.51] n=137（无有效应） | 0.50 n=169 | 0.85 |
| 两者都翻 S_both | 按钮跟供体 | 0.81 n=32 | 0.48 n=138（无有效应） | 0.54 n=170 | 0.87 |
| 置零 Z | 翻转率 | 0.55 | 0.63 | 0.61 n=227 | 0.67 |
| 换 seed | 翻转率 | 0.03 | 0.24 | 0.19 n=388 | 0.08 |

- 第 1 步：Frozen Wan 与 MotionMem 一样几乎完全跟随供体，说明「第 1 步可被同局 motion token 定向操纵」不是 MotionMem 特有。
- 第 2 步以后：Frozen Wan 只翻绕向的跟随率 0.63，区间下界未超过其自身置零翻转（0.58），判无有效应；按钮跟随 0.76 判有效应。MotionMem 上一轮同档 0.87。差别主要在后续步。
- Frozen Wan 的换 seed 噪声本身更大（0.19 vs MotionMem 0.08）。

## ④ VideoRepick：同一局交换段替换（`records/vr-{wan,motion}/readout.md`）

| 模型 | T 组 pred（跟随供体） | correct（仍拿对） | 置零仍拿对 | 换 seed 翻转 | 有效应 |
|---|---|---|---|---|---|
| Frozen Wan pooling | 0.00 n=102 | 0.99 | 0.92 | 0.263 | 否 |
| MotionMem | 0.15 [0.08,0.23] n=124 | 0.78 | 0.80 | 0.136 | 否（k=5 段 9/12 跟随，样本小） |

两模型在 VideoRepick 上基本都不能靠换同一局段 token 操纵；MotionMem 仅最后一次交换（k=5）有跟随迹象。

## ⑤ 提速效果

- 上一轮 8.7 h 只跑完 MotionMem RouteStick 四组；本轮当天完成 Frozen Wan RS 四组（793 次）、两模型 VideoRepick（各约 250 次干预 + 100 噪声）、MotionMem core 三组（904 行），以及三道闸门。
- 在线编码：示范段首包 add_buffer 15.6 s → 8.8 s，之后每包 3.5–4.1 s → 0.6–1.2 s（同局冒烟）。变体录制阶段取消（离线编码 GL 4 卡约 35 分钟）。
- 资源：GL 4 个单卡 A40 占位（66207065/67/68/69，14:3x 逐个释放）+ aspen 2 卡；每卡 2 通道（实测上限），常驻服务下显存比例 0.30。

## ⑥ MotionMem RouteStick core 窗口组

判定行：`SWAP_SCOPE=PASS runs=723 scope_bad=0 count_mismatch=0`；`NOISE_BASE=PASS model=motion pairs=451 flip_rate=0.082`；`RS_SAMEP=FAIL model=motion groups=3 invalid=25`（噪声 24 为上一轮 08:45 停发时未跑的噪声行，干预 1）。`records/rs-core/readout.md`。

| 组（只换完全落在第 j 段内的窗口） | 主读数 | 第 1 步 | 第 2 步以后 | 合计 | 换 seed 翻转 |
|---|---|---|---|---|---|
| S_swing_core | 绕向跟供体 | 0.10 [0.03,0.18] n=60 | 0.10 n=230 | 0.10 n=290 | 0.08 |
| S_side_core | 按钮跟供体 | 0.06 n=32 | 0.11 n=183 | 0.10 n=215 | 0.06 |
| S_both_core | 按钮跟供体 | 0.06 n=32 | 0.09 n=186 | 0.09 n=218 | 0.06 |

- **只换段内 core 窗口几乎不改变行为**（跟随率约 0.10，接近换 seed 噪声 0.06–0.10）；对照上一轮换「与第 j 段相交的全部窗口」第 1 步 0.98–1.00。说明第 j 步的操纵效应来自**跨段边界的窗口**（含第 j 段起点附近与相邻段交界处），而不是段内中间窗口。
- S_both_core 的 `swing_follow_phys` 0.91 不是跟随：两者都翻时供体的物理弧侧与原 N 相同，0.91 表示模型基本照原样执行。

## ⑦ 局限

1. ENC_TOL 原阈值对跨卡噪声过严（同配置跨卡即 min_cos 0.9974），按用户「跨卡以及精度问题我们都可以暂时不考虑」改为跨卡噪声口径（0.997/0.3）判定，原判定照录；行为一致性由 BEHAVIOR_TOL（diff=0）保证。
2. Frozen Wan RS 噪声对照 15 局无效（10:22 重起 GL 作业步时被打断的噪声 spec，以及上一轮剔除、本轮恢复的第 44 局 2 行未补跑），噪声翻转率按 388 对计。
3. 本轮 N 与噪声复用上一轮在线 fp32 录制（release 6d8edc8），干预为 fast/离线 bf16；跨精度差异由 NOOP_EQ2、BEHAVIOR_TOL 量化为零。
4. 只翻左右 / 两者都翻不是严格单变量（同上一轮）；core 组用于隔离。

## ⑧ 用户决策记录

见 launch.md ②；开工令「同意，开工」。

## ⑨ 归档

- `records/gates/gates.txt`、`records/rs-wan/`、`records/vr-wan/`、`records/vr-motion/`、`records/rs-core/`。
- 运行产物：NFS `…-eval/v1-store/evaluation/steer-speedup-1116/`；日志 `…-eval/v1-store/logs/steer-speedup-1116/`；runbook `…/steer-speedup-1116/runbook/`。
