# 20 步真实训练可读性验收通过

按用户“开始实现该计划 有问题提前问用户 越早越好”的指令，完成新库的真实输入和训练检查。20 个 step（0–19）全部执行，逐步 loss、梯度等记录全部有限，checkpoint 19 收尾成功，`EXIT_CODE=0`。实际运行 `2026-09-14T21:11:15Z` 至 `21:14:36.544738Z`，201.54 秒。验收证明新库和新统计量能被当前训练链路使用，不据这 20 步判断策略效果。

## 版本与配置

起跑 HEAD 为 `81a6c7580507f82a4ad19cf4c651ccd8a3c80336`，工作区干净，实际消费代码与本轮实现版本一致。tmux 为 `v2b-read20-20260914T174147Z`，pane PID 1262054，训练 PID 1262180。完整命令见 [launch.md](launch.md)，机器可读参数见 [records/run_meta.json](records/run_meta.json) 和 [records/launch.actual.json](records/launch.actual.json)。

仅使用 GPU 4–7，默认 batch 64、worker 4、FSDP 4、seed 42、bfloat16、pi05_base；history 为 `perceptual-framesamp-context.yaml`，motion 关闭。步数 20 与逐步日志为 CLI 覆盖，学习率和其他全局默认未改。数据源和清单通过 `MMEVLA_FRAMESAMP_SOURCE`、`MMEVLA_FRAMESAMP_MANIFEST` 显式绑定到本轮库，与 store_meta 内路径一致。

## 输入与统计量

库为 `4task-v2-1600ep-604f16da`，manifest SHA256 `4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918`。新 norm_stats SHA256 为 `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`。checkpoint 19 的 `assets/robomme/norm_stats.json` 与输入统计量逐字节摘要相同，确认训练保存的是这份新统计量。

两档 packed 分别以对应 history 配置进行了真实 CPU batch 检查（worker 0，seed 42）：state `(64,32)`、actions `(64,20,32)` 均为 float32 且有限，static_image_emb `(64,512,2048)` 为 bfloat16，`motion_emb/motion_pos/motion_mask/mem_order` 均为 None。证据为 [batch4.json](records/batch4.json)、[batch8.json](records/batch8.json)。真实训练使用默认 worker 4，运行 provenance 确认 motion 关闭且 framesamp 清单为本轮版本。

## 实测指标与结束判据

`metrics.jsonl` 恰有 20 条记录，step 严格为 0..19。loss 范围 0.4258049428462982–0.6334579586982727；日志中的梯度范数范围 62.78167724609375–130.35836791992188，均为有限值。首/末 loss 分别为 0.617760181427002、0.42620348930358887，仅作为运行记录，不据此评价训练收益。完整数值及十六进制表示保留在 [metrics.jsonl](records/metrics.jsonl)。

```text
Step 19: grad_norm=85.1399, llm_grad_norm=65.3807, loss=0.4262, mem_enc_norm=54.5108, param_norm=1803.0919
CheckpointManager Save Finalize is done on all hosts.
EXIT_CODE=0
TRAIN20_ACCEPT=PASS
```

编译期间出现 XLA 常量折叠耗时提示（其中一次约 3.625 秒），之后原命令完成全部 step 和保存收尾，未发生运行失败。训练时长未覆盖原 MotionJEPA 训练的完整新 epoch，守卫仅采到起跑前 epoch 69 基线；不单独以本短测宣称共处吞吐结论。

最终复核从原训练日志补读到 epoch 70，吞吐 870.7770815382041 samples/s，仍高于 865.26 暂停线。其时间窗口覆盖本短测和相邻时间段；该记录在建库档案的 `records/coexistence.train-perf.json`，不改写短测期间的原始守卫日志。

## 归档与清理

归档包括原始逐步指标、清洗训练日志、实际参数、motion provenance、checkpoint 元数据、验收 JSON 以及两档 batch 记录。未归档模型权重或配置脚本副本。清洗日志保留所有 20 条 Step 行和退出记录。

归档后核对进程均已退出、run_meta 的 checkpoint_dir 与本轮名称一致、指标与归档字节相同、provenance 绑定新库且 motion 关闭，随后仅清理本轮 checkpoint run 与 bench 目录。清理证据见 [cleanup.json](records/cleanup.json)。正式数据、源 extracted、新 norm_stats、原始日志和编译缓存均保留。
