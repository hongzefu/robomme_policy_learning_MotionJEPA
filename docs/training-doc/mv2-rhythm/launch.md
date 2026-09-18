# 新配置的客户端节奏与预算边界检查

本任务在 CPU 上按客户端每 16 步推理、最多 1300 环境步的节奏验证状态机，采用已建成的新库和 perceptual-framesamp-modul-8frame-8x8-motion.yaml。budget=160、demo 最少 17 帧，边界 es=1296 合法、es=1297 超预算；真实 test 集 200 次 EnvRunner reset 已另行完成并通过，见 [实测记录](../mv2-evalbound/result.md)。

从本页提交后的 clean CHECK_HEAD 启动，起止检查完整提交与空 porcelain。共用外壳见 [V4 启动记录](../mv2-v4/launch.md)。tmux 全名 mv2-rhythm，命令 `bash v1-store/logs/mv2-open-runner.sh rhythm <CHECK_HEAD>`，日志 v1-store/logs/mv2-rhythm.driver.log，报告 v1-store/reports/motion/mv2-rhythm.json。全部 gate 须通过、EXIT_CODE=0；不得用 None 或未执行边界冒充通过。任务尚未启动。
