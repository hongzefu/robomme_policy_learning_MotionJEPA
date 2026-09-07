# v1-g1-speed 结果（G1-speed vs G0-speed-r2）

`BENCH_PASS`、`EXIT_CODE=0`，1000 步跑满。

## 一、结论（AGENTS 16 口径：以**均值**为准，中位数只作附列）

**dtype 修复带来 步时均值 −7.21%、GPU util 均值 +5.64pp、0% 采样占比 −3.69pp。**

| 指标 | G1-speed | G0-speed-r2（锚点） | 变化 |
|---|---|---|---|
| **步时均值** | **1.1003 s** | **1.1858 s** | **−7.21%** |
| **util 均值** | **92.10%** | **86.46%** | **+5.64pp** |
| **0% 采样占比** | **1.20%** | **4.89%** | **−3.69pp** |
| 慢步数（>1.5× 中位） | **0** | 3（@122/335/419） | −3 |
| 非慢步均值 | 1.1003 s | 1.1834 s | −7.02% |
| 步时中位（附列） | 1.0809 s | 1.1520 s | −6.17% |
| p10 / p90（附列） | 1.0483 / 1.1798 | 1.0967 / 1.2753 | −4.41% / −7.49% |
| 稳态样本数 | 950 | 950 | — |
| epoch 外推（均值口径） | 15.10 h | 16.28 h | −1.17 h |

per-GPU util 均值：G1 侧 GPU0 / GPU1 分别 91.6% / 92.6%（锚点侧 85.8% / 87.1%）。

**本机口径，不作最终吞吐结论**（AGENTS 13）——GL 侧吞吐验收仍是 v4 计划 D 节的主判据，
speed 链是本机口径的链式对账。

## 二、绝对收益与计划预期的吻合度

1.1858 − 1.1003 = **85.5 ms/step**。

计划（`v1-dtype-unify-plan.md` Context）按实测分解预估的收益是 **约 80 ms/step**
（collate 52→19 ms 省 33 ms + device_put 73→23 ms 省 50 ms，合计约 83 ms）。实测
85.5 ms/step 与该预估几乎完全吻合——收益来源清楚地就是「短样本被抬成 f64 后在
worker→collate→IPC→device_put 全程白搬运的那部分字节」被消除。

## 三、util 结构的改善

- util 均值 86.46% → 92.10%，**0% 采样占比从 4.89% 降到 1.20%**：0% 采样即 GPU 完全
  空转的采样点，它的大幅减少说明喂数据的间隙被填上了——与「batch 载荷 757 → 257 MB、
  device_put 73 → 23 ms」的机理一致；
- **慢步从 3 个降到 0**：锚点侧的 3 个慢步（@122/335/419，慢步均值 1.9594 s）在本轮
  消失。慢步通常对应一次特别大的 IO/H2D 尖峰，载荷减小后不再触发；
- 两卡 util 更均衡（91.6/92.6 vs 85.8/87.1）。

## 四、口径核对（可比性前提）

两侧 `env.json` 逐项一致：`XLA_FLAGS=''`（均为生产档、autotune 默认开）、
`XLA_PYTHON_CLIENT_MEM_FRACTION='0.95'`、steps=1000、workers=4、seed=42、fsdp=2、
b8。统计由 `scripts/dtype-unify/analyze_util.py` 对**两侧同法**重算（稳态窗步 50–999、
慢步阈值 1.5× 中位），不直接引用对方留档数字；该脚本对锚点的复算与其 result.md
逐项吻合，可比性成立。

bench 自报：`RESULT batch=8 稳态=1.081s/step (n=948, p10=1.048, p90=1.180)`、
`epoch估算=53407s ≈ 14.84 小时`（中位口径，按 AGENTS 16 不作标题结论）。

## 五、产物

`records/`：`metrics.jsonl`(1000) / `env.json` / `run_meta.json` / `util-lms500.csv` /
`util-legacy15s.csv` / `BASELINE_MANIFEST.json`(5 条)。无 `param_checksums.jsonl` 与
`batch_digests.jsonl`——speed 口径下 `SAVE_INTERVAL=0` 联动禁掉了两种摘要，这正是
性能族 run 与正确性族 run 的分野。
