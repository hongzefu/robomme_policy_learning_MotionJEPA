# v1-framesamp-dl-w{2,4,8,16}（S8a GL dataloader-only 四档）result 记录

对应 `v2-framesamp-restructure-plan.md` 阶段 4 S8a。launch 记录见同目录
`launch.md`。四档全部 `DLBENCH_PASS` + `EXIT_CODE=0`。

## job 与节点

| run_name | job id | 节点 | 状态 | 用时 |
|---|---|---|---|---|
| `v1-framesamp-dl-w2` | 58995916 | gl1508 | COMPLETED | 3m23s |
| `v1-framesamp-dl-w4` | 58995917 | gl1512 | COMPLETED | 3m55s |
| `v1-framesamp-dl-w8` | 58996753（三提） | gl1501 | COMPLETED | 4m15s |
| `v1-framesamp-dl-w16` | 58995919 | gl1512 | COMPLETED | 6m41s |

- w8 前两次提交（58995918、58996161）均落在故障节点 gl1514 被 root
  `CANCELLED`（~2 min 零日志），三提加 `--exclude=gl1514` 后在 gl1512 之外的
  gl1501 正常完成。此后本计划全部 GL job 一律 `--exclude=gl1514`。
- **档间条件不严格同一**（判读时注意）：w4 与 w16 先后落在同一节点 gl1512
  （w16 起跑时该节点页缓存已被 w4 部分焐热）；w2（gl1508）与 w8（gl1501）
  各自在未碰过 packed 库的冷节点。四档均为「冷态自证」口径（allocation 内
  不跑 full 校验、无本地复制预热），但节点历史造成的缓存差异未受控。

## 吞吐结果（batch 64、warmup 5、measure 40，packed 2.43 MB/样本现场推导）

| workers | cpus | 样本/s | 批时 s/批 | MB/s 公式 | MB/s server_read | MB/s normal_read | pkl_ms med/p90 | gather_ms med/p90 |
|---|---|---|---|---|---|---|---|---|
| 2 | 4 | 57.30 | 1.117 | 139.1 | 114.7 | 136.2 | 5.41/41.21 | 3.96/4.90 |
| 4 | 6 | 97.74 | 0.655 | 237.2 | 190.0 | 233.1 | 6.49/10.21 | 4.95/6.99 |
| 8 | 10 | 89.88 | 0.712 | 218.1 | 172.5 | 218.9 | 5.04/5.90 | 3.59/4.16 |
| 16 | 18 | 116.00 | 0.552 | 281.5 | 149.5 | 260.1 | 6.31/8.03 | 3.69/5.17 |

- 记录目录：`v1-store/bench/bottleneck/v1-framesamp-dl-w{2,4,8,16}/`
  （summary.jsonl / batches.jsonl / seg_timing.jsonl / env.json，env.json 均
  记 backend=packed 显式 + resolved 双根 + store_meta sha）。
- 公式口径与实测口径吻合良好：normal_read 与公式差 −2%～−8%（w16 −8% 与
  server_read 149.5 明显低于公式，与该节点页缓存部分预热一致——缓存命中不再
  走 NFS server 读）。

## 对照 legacy 历史（v1-dlb-*，同 batch 64/measure 40 口径）

| 配比 | legacy 样本/s | packed 样本/s | 倍率 |
|---|---|---|---|
| w4c6 | 27.41 | 97.74 | **3.57×** |
| w8c10 | 26.25 | 89.88 | **3.42×** |
| w16c18 | 33.16 | 116.00 | **3.50×** |
| legacy 最好档 w16c10=40.89 vs packed 最好档 w16c18=116.00 | — | — | **2.84×** |

## 判读

1. **同配比 packed 一致取得 3.4–3.6× dataloader 吞吐**，与字节帐预期方向一致
   （每样本读盘 19.08 MB → 2.43 MB，缩 7.9×；吞吐增幅小于字节缩幅，说明瓶颈
   已从 NFS 读字节转移到 pkl 反序列化/CPU 侧——SEGPROBE 显示 pkl_ms 中位
   5–6.5 ms 与 gather_ms 3.6–5.0 ms 同量级）。
2. **训练需求余量充足**：e2e 目标步时 ≤5.00 s、batch 64 → 需 ≥12.8 样本/s；
   最低档 w2 已达 57.3（4.5×需求），官方默认档 w4 达 97.7（7.6×需求）。
   dataloader 侧不再是 e2e 步时的一阶瓶颈（待 S8b e2e 验证收口）。
3. **w8 略低于 w4（−8%）为条件噪声**，与冷节点（gl1501 无缓存）+ 40 批短窗口
   一致，SEGPROBE 分段（w8 的 pkl/gather 中位反而最低）不支持 w8 存在结构性
   退化；S8b e2e 的 w4/w8 步时差 ≤3% 附加判据将在真实训练负载下复核。
4. S8a 无过/不过阈值（计划如此），数据落档供 S8b 对照。

## 状态

S8a 完结。S8b（v1-framesamp-e2e 四 run）判读见
`docs/training-doc/v1-framesamp-e2e/`。
