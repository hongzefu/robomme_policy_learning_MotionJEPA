# 4×4 在线回归结果

全部通过。运行代码和全程 HEAD 均为 `c08ec2060a544af1869c1e24f755e536150569ca`，起跑及收尾工作区干净；2026-09-14 02:33:38 至 03:22:45 UTC，共 49 分 07 秒，仅使用物理 GPU 7，整体 `EXIT_CODE=0`。

| 检查 | 实测结果 | 耗时 |
|---|---|---:|
| REF 与候选在线记忆 | `ONLINE_MEM_REGRESS=PASS frames=256 keys=4 mismatches=0` | 160 秒 |
| 旧实现、建库副本、候选三方 | `POS_TABLE=PASS`、`ENC_LAYER=PASS steps=13 keys=3 mismatch=0`、`ASSEMBLY=PASS`、`OOB_PROBE=PASS`、`ONLINE_MEM=PASS` | 79 秒 |
| 已有 40k 真模型 S 臂 | 5 集、120 个决策点、140 个真 sidecar 窗；`TIC_SIDECAR=PASS blocking=14/14` | 1537 秒 |
| 40ep 完整 motion 回归 | `ONLINE_ENC_BITEXACT=PASS compared=772 mismatches=0`；起点、位置、排列、provenance 全部通过，738 个决策时刻 | 1170 秒 |

历史 `tic-obs-model-40k` 的 12 个 PASS 标签在本次全部保留。bf16 `VT_FULL_VS_CACHED` 仍超过原报告阈值，`rel_fro=0.0014338662385844179`，按用户裁决作为观察项；本次有效前缀 KV 恰好逐位相同。f32 阻断检查共 15 点（每集 3 点），`rel_fro=1.3940050310904495e-7`、`prefix_kv_max_abs=0`，满足约定。原始记录中的 dtype、mask、完整判定行均保留，不把 bf16 观察项写成通过。

本次使用显式确定性 XLA 配置和物理 GPU 7，历史回归使用 GPU 0，历史 bf16 观察数字本身也随编译而变；这里只按计划比较 PASS 标签及 bf16 差异的数量级。独立的 REF/candidate 同进程记忆对拍则使用同一个真实 SigLIP 编码器，256 帧逐位相等。

用户已明确同意文件 SHA 与实际数组摘要双重检查，以及按 checkpoint 保存的配置自动计算评估前缀长度。本阶段只证明 4×4 回归通过；C8/M8 的训练输入、1000 步轨迹与推理验收仍待后续阶段完成。

归档 `records/` 包含在线记忆逐帧摘要、三方对拍明细、TIC 全部判定与逐点统计、772 窗口回归统计，以及清洗后的总日志。未归档模型权重、配置副本或从 Git 提取的参考源码。
