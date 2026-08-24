# 本机 2 卡 epoch 时长基准报告（v1-2gpu-epoch-bench）

**结论：本机 2× RTX 6000 Ada、数据走 NFS turbo、2 卡能跑的最大档 batch 8 下，1 个 epoch（395,289 样本 = 49,411 步）外推约 14.5 小时（稳态 1.060 s/step）。官方口径 batch 64 在 2 卡上跑不起来（OOM）。**

- 机器/存储：本机 2× RTX 6000 Ada 46 GB（驱动 570.211.01 / CUDA 12.8），数据集 `v1-store/datasets/4task-gl`（NFS turbo）
- 配置：尽可能对齐 `scripts/finetune_mme_vla_suite.sh`（workers 4、use_history + `perceptual-framesamp-context.yaml`、seed 42、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`），差异逐项见 `scripts/smoke-local/README.md`；wandb 关闭；不落 checkpoint
- 显存实测：batch 64/32/16 全部 OOM（失败张量 17.62/12.61/10.38 GiB）——2 卡 FSDP 每卡驻留约 28 GB 参数+优化器+EMA 状态，装不下固定底座约 8 GiB 的激活；**batch 8（per-device 4）可跑**
- 稳态口径：300 步中剔除前 50 步 warmup 与参数校验和步及其后一步，n=230 逐步墙钟差取中位数 = 1.060 s/step（p10 1.029 / p90 1.200）
- loss：300 步全有限，0.758 → 0.044
- ⚠ 本机估算，不作正式吞吐结论（AGENTS.md 第 13 条）；page cache 可能使稳态偏乐观；batch 8 为非官方口径

**重要附带发现**：同配置同 seed 重跑两轮，参数校验和逐步全不相同——本机默认设置下训练**非 bitwise 确定**（疑 XLA autotune 重编译所致）。未来一致性 A/B 前须先加确定性设置（`--xla_gpu_deterministic_ops`、固定/关闭 autotune、共用 jax 编译缓存，两边同设）并用两次重跑验证校验和逐步一致。

一致性检验记录（`metrics.jsonl` 逐步 loss/梯度范数 hex 精度、`param_checksums.jsonl` 逐叶子 sha256、`env.json`）保留在 `v1-store/bench/2gpu-epoch-bench/v1-2gpu-epoch-bench-b8/`，格式与三级比较协议见 `scripts/smoke-local/README.md`；起跑/轮次/完整数字见 `docs/training-doc/v1-2gpu-epoch-bench/`。
