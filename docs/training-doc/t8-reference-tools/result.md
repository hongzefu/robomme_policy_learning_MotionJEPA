# 阶段一结果

参考代码固定为 `REF=99faacb1319adfc63c0cf9a15187e24c34e38fd1`（`commitV9.0`）。GPU 4、5 上的 `t8-c32-s100` 从 clean HEAD 起跑，2026-09-14 01:51:57 UTC 开始、02:20:05 UTC 结束，共 28 分 08 秒，`EXIT_CODE=0`。参考 worktree 为 `v1-store/worktrees/ref-8x8`，后续 A 侧必须显式设置该 worktree 的 `PYTHONPATH=…/src`，并共用主树 uv 环境。

100 步 × 5 个标量的 `float.hex()` 与历史 `aws-t2-ref-s100` 全部相同；完整投影文件 SHA 为 `85b8fe376729259cf25bb3f56c409eaa55806b0b7497e6a2955cf6d2f05b9e34`。五份完整 TrainState、各 177 个叶子，与历史对应记录逐叶相同且全部有限。这里按历史调度记录的 `state_step` 为 `{0,26,51,76,100}`，不能混作正式 1000 步验收的十一点调度。主进程 index 序列共有 872 项。

本次用于复刻旧基线的资产是 40ep norm_stats，文件 SHA 为 `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`，实际 loader 接收的数组与该文件解析结果一致。正式十二条轨迹仍按用户决定使用 400ep 数据和 `750a8e9b…` 的统计文件。

五次状态摘要分别耗时 357.246、206.072、192.977、387.474、278.060 秒。这里记录的是取证开销，不是稳态训练吞吐；不能将历史约 169 秒直接当作本轮固定成本。训练首、末 loss 为 0.580677330493927、0.09695563465356827，仅作该回归运行的记录。

轻量检查补充：8×8 参考链在 motion 开、关两态各取 8 个真实边界样本，step `{0,6,7,8,9,31,32,33}` 的 image/pos 直接对源 npy、mask 数量与首个运动窗口数量均符合独立公式。前述 4×4 原始输入与 transforms/collate 检查也已通过，详见 [launch.md](launch.md)。这些开发检查不替代后续完整 fixture、1000 步三轨迹及推理验收。

仿真设备绑定另做了短探针：`CUDA_VISIBLE_DEVICES=7` 时，SAPIEN 的逻辑 `cuda_id=0` 对应 `pci=0000:a0:1d.0`，与 `nvidia-smi --id=7` 的物理 PCI 地址相同。探针结束后 GPU 4–7 的已用显存均回到 0 MiB。

用户进一步确认：“同意 让评估脚本读取 checkpoint 保存的配置，自动算出正确长度”。据此在后续 `serve_policy_mv.py` 中按保存的配置设置 adapter 前缀长度，motion 开启为 1184、关闭为 1088，保留原采样实现。

临时 run 的 records、checkpoint 根和两份开发 fixture 在核验后清理，编译缓存保留；原始日志改名为 `v1-store/logs/t8-stage1-c32-s100.log`，避免与后续 400ep 排错运行同名。这里保留阶段一的结论与可还原命令，不归档临时权重或 Bash/YAML 副本。
