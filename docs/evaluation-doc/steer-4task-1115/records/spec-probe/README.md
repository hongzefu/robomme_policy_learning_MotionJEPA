# RouteStick episode_spec 实测（美东 2026-11-15 晚）

- 目的：确认新 benchmark（`b4e97f2`）的 `episode_spec` 能否从自然局重建一局、并只改第 j 步绕向或左右，用作同一局供体（计划 `1115-4task-routestick-videorepick-steering-plan.html` 第二部分 1.1）。
- 会话：aspen 策略仓库会话「VLA steering with motion models testing」。
- 环境：benchmark 源码 `git archive b4e97f2 src` 导出到会话 scratchpad；仿真解释器 `v1-store/envs/official-sim/bin/python`；单卡 A6000（CUDA_VISIBLE_DEVICES=1）。
- 复现：`PYTHONPATH=<bench-b4e>/src <sim-python> -I rs_spec_probe.py <seed> <difficulty> <j>`，实测参数 `3100003 hard 2`、`3100010 medium 2`、`3100011 easy 1`。
- `probe-run1-layout-frames.txt`：自然局 vs 重建 spec 局路线、绕向、按钮坐标 md5 与示范帧逐位比对（351/351、201/201、101/101 一致）；翻第 j 步绕向/左右后首个不同帧 = 第 j 段首帧。该次输出末尾的 DEMO 行分段取错（偏一段），以 run2 为准。
- `probe-run2-demo-swing.txt`：修正分段（第 j 段 = 示范第 50(j−1)+1…50j 帧）后，示范实际绕向与 spec 逐段一致（原样 13 段、翻转后各段均一致）。
