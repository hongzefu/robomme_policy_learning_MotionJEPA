# 4 卡 batch 64 的 NFS 瓶颈判定——四实验汇总（2026-08-24）

**结论：NFS turbo 存储侧对 4×A40、全局 batch 64 的官方口径训练没有瓶颈——供给（398-628 MB/s）是需求（251 MB/s）的 1.6-2.5 倍。但端到端实测暴露了另一个真瓶颈：job 内 CPU 配比。官方口径 `num_workers=4` 配 8 CPU 时，dataloader worker 被 jax 主进程挤压，e2e 步时 6.93 s/step（epoch ≈ 11.9 h），比纯计算的 4.78 s/step（epoch ≈ 8.2 h）慢 45%。把 `--cpus-per-task` 提到 ≥12-16（或 worker 提到 8-16）即可释放，无需改代码。**

## 判据链与四个实验

```
需求 = 1.20 GB/步 ÷ compute-only 步时 = 251 MB/s     ← 实验 3（4×A40 重复 batch）
供给 = dataloader-only 实测 398-628 MB/s             ← 实验 1（1×A40 × 9 个数据点）
供给 ≥ 1.6× 需求 ⇒ 存储侧无瓶颈
e2e 实测 6.93 s/step，期间 NFS 仅 122 MB/s           ← 实验 4（4×A40 端到端）
   → 慢的 45% 不来自 NFS，来自 worker CPU 饥饿
本机旁证：2 卡 b8 冷缓存仅慢 8%，GPU 99%             ← 实验 2（计算主导）
```

| 实验 | run/job | 关键数字 |
|---|---|---|
| 1 供给 | 58619536 + 58631511-15/40（9 档，3 节点） | **398-628 MB/s**（server_read 实测）；对 worker 数不敏感、对每 worker CPU 预算敏感（w4c6→w4c10 +30%）、w16c10 超订档最高 563 |
| 2 本机旁证 | v1-coldcache-b8 | 冷/热 1.08×，GPU 99%，本机口径计算主导 |
| 3 需求 | 58619547 | **4.778 s/step**（p10-p90 仅 0.015 s）→ 需求 251 MB/s；epoch 计算下限 **8.2 h** |
| 4 e2e | 58627883 | **6.933 s/step**（p10 5.01 / p90 8.54）→ **epoch 11.9 h**；NFS 实际仅 122 MB/s；GPU util 均值仅 69-70%（约 1/3 采样 <100%、大量为 0%，空转段与慢步一一对应——「中位 100%」是假象） |

## 归因与建议

1. **e2e 慢 45% 的机制**：8 CPU 里 jax 主进程（驱动 4 卡）与 4 个 spawn worker 抢核。standalone 同 4 worker 配足 CPU 能供 27-36 样本/s，e2e 里连在线需求 13.4 样本/s 都供不稳（p10 步 ≈ 纯计算步时、p90 步在等数据）。
2. **正式训练建议**（启动参数级，无代码改动）：`--cpus-per-task=16`、`num_workers=8`（供给侧已证明 NFS 余量充足、worker CPU 超订无损）；预期步时压回 ~4.8-5.2 s、epoch ~8.2-8.9 h、80k 步全程 ~4.4-4.8 天。
3. **数据形态注记**：每样本 = 1 pkl（395 KB）+ ≤32 个 token_emb（603 KB），公式上界 18.7 MB/样本；同 episode 相邻样本文件高度重叠，客户端缓存实际吃掉 6-27% 的重复读——真实需求比公式还低，判定更稳。
4. **内存教训**：4 卡 run 若触发参数校验和（device_get ~28 GB），host 内存须 ≥96G——实验 4 的 64G 在末步校验和被 OOM-kill（训练数据无损）；吞吐类 run 应干脆不开校验和。
5. **历史锚点修正**：本机到 turbo 的 132/320 MB/s 不能外推集群——GL 节点实测 4-6 倍于此。

## 溯源

各实验的起跑/结果/记录归档：`docs/training-doc/{v1-gl-dlbench, v1-coldcache-b8, v1-computeonly-b64, v1-e2e-b64}/`；脚本与判据说明：`scripts/bottleneck-bench/README.md`；工作副本记录：`v1-store/bench/bottleneck/`。
