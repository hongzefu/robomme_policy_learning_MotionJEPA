# v1-2gpu-epoch-bench 结果留档

第 3 轮（commit `22baa1c`，clean HEAD，`BATCHES="8 4 2"`）成功，run_name = `v1-2gpu-epoch-bench-b8`。2026-08-24 14:25 起跑，14:45 完成，`EXIT_CODE=0`。

## 核心结果

| 项 | 值 |
|---|---|
| 实际可跑最大 batch（2 卡） | **8**（global；per-device 4。64/32/16 均 OOM，见 launch.md 第 1 轮） |
| 稳态步时（剔除前 50 步 warmup 与校验和步及其后一步，n=230，中位数） | **1.060 s/step**（p10=1.029，p90=1.200） |
| steps_per_epoch（395,289 // 8） | 49,411 |
| **1 epoch 外推** | **52,372 s ≈ 14.55 小时** |
| loss（300 步，全有限） | min 0.0236 / max 0.7580 / 末值 0.0437（首步 0.7580 → 快速收敛） |
| 参数校验和 | 12 次（step 25…275 每 25 步 + step 299），110 叶子/次，单次中位 47.3 s |
| 300 步总墙钟 | 15.3 min（其中校验和约 9.4 min，正式训练无此开销） |

⚠ 按 AGENTS.md 第 13 条：本机数字只作估算，不作正式吞吐结论；300 步稳态可能受 page cache 影响偏乐观。batch 8 是非官方口径（官方 4 卡 batch 64），该 epoch 时长不能直接换算官方口径吞吐。

## 重要附带发现：同配置重跑非 bitwise 确定

第 2 轮（commit `891d6e3`）与第 3 轮（commit `22baa1c`）除记录链路（不进计算图）外计算完全同配置、同 seed、同数据，但每个记录步的参数校验和均不同（如 step 25：`23e643fb…` vs `02bd9a4d…`）。即**本机 2 卡默认设置下，同配置重跑不满足 bitwise 一致**。最可能来源：每轮结束删除了 `~/.cache/jax_<exp_name>` 编译缓存，XLA 重新 autotune 可能选中不同 kernel/归约实现。

对一致性检验的含义：做 A/B 逐位对比前，必须先把「同配置重跑 bitwise 稳定」的前提立住——候选手段：A/B 两边同开 `--xla_gpu_deterministic_ops=true`、关闭/固定 autotune（如 `--xla_gpu_autotune_level=0`）、保留并共用同一份 jax 编译缓存；先跑两次相同 run 验证校验和逐步一致，再开始 dataloader 改动的对比。

## 产物位置

- 记录（保留）：`v1-store/bench/2gpu-epoch-bench/v1-2gpu-epoch-bench-b8/`（`metrics.jsonl` 300 行、`param_checksums.jsonl` 12 行、`env.json`；格式见 `scripts/smoke-local/README.md`）
- 日志（保留）：`v1-store/logs/2gpu-epoch-bench-driver.log`（第 3 轮）、`*.round1/round2.log`（前两轮）、`v1-store/logs/v1-2gpu-epoch-bench-b{64,32,16,8}*.log`
- run 目录与 `~/.cache/jax_*`：已按脚本自动清理，无 checkpoint 产物
