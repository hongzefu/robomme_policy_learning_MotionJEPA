# v1-singlerun-g0 结果（C3：G0 完整重锚）——G0_EQ=PASS，第六份锚点同值

- **结论先行**：`G0_EQ=PASS`（`tests/g0_gate.py` fail-closed 总闸唯一判定行）。
  `scalars_hex.tsv` sha256 = `c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757`，
  与 G0b r1/r2、G1、G2、G3 五份同值，**本轮为第六份**。v5.0 对 `train.main` 体内的改动
  （删三处 tentative 死代码）与目录拆分（bench→g0/util）经 1000 步位级裁决为零数值影响。
- **起跑 commit**：`f641f40`（launch.md 先行提交于 `cc0d88d`）。2026-08-30 起跑，
  loop 墙钟（step0→step999，metrics wall_time 差）**47.55 min**，全程含收尾约 64 min，
  在 1–2 h 预算内；`BENCH_PASS`、driver `EXIT_CODE=0`。
- **基线引用**（AGENTS 18 末句）：G0b-r1（run_name `v1-grad-baseline-g0b-r1`，commit
  `570287f`）固化产物；起跑前环境指纹 preflight `BASELINE_ENV=PASS`
  （`records/preflight_baseline_env.txt`）；packed 库锚点 `store_meta_sha256=3990165c…`
  与 G2/G3 一致（env.json 实测同值）。

## 判定行实录（全部达标）

```
BASELINE_ENV=PASS
SCALARS steps=1000 keys=5 hex_mismatch_steps=0 first_mismatch_step=None
REL 五键 median/p95/max 全 0.000e+00
STATE_DIGEST rows=12 mismatch=0
BATCH_DIGEST rows=14 mismatch=4 first_bad_step=100 bad_keys=2 首个: ["['static_image_emb']"]   # 非判据，与 G2/G3 逐字吻合
BATCH_DIGEST_CANONICAL rows=14 mismatch=0
CANON_CHECK=PASS steps=14
INDEX_SEQ=PASS n=8072（共同前缀逐个一致, steps≈1000）
sha256(scalars_hex.tsv) = c799a0b2…   # PROJECTED rows=1000（tests/project_scalars.py 投影）
G0_EQ=PASS
```

`DET_CHECK=FAIL`（batch_digest_diff=4）照既定纪律不作判据。过程中增量 state 对拍
（Monitor 挂 `param_checksums.jsonl`，逐摘要步与 G0b-r1 比 `global_digest`+`state_digest`）
12 个摘要步（0/100…900/299/999）**全程 MATCH、零分叉**，fail-fast 通道未触发。

## 过程与性能旁注（非判据）

稳态 1.942 s/step（n=929，p10 1.841 / p90 1.982；本机数字按 AGENTS 13 只作估算）。
本轮为冷编译（EXP_NAME 全新）；跨编译 bitwise 由 D2-cold 实证背书，结果亦再次印证。

## C2.5 记录器 smoke 结果归档（RECORDER_SMOKE=PASS cases=4）

计划 4.4/第八节四用例，2026-08-30 本机完成（临时 run `v5-recorder-smoke-*`，已清理）：

| # | 用例 | 结果 |
|---|---|---|
| S1 | 设 `TRAIN_RECORD_DIR` 跑 2 步 | 训练 RC=0；`metrics.jsonl` 2 行×5 键 + `run_meta.json` 齐全；`load_metrics` 解析通过。补充 S1b（同款 dense 采样器伴跑）：analyzer `--steps 2 --warmup 0` 出全部 `RESULT` 行、`loss n=2`（解析步数=2）、RC=0 |
| S2 | 预置 sbatch 式 6 文件后首跑 / 同目录二跑 | 首跑 RC=0（「目录非空」不误伤）；二跑必 `FileExistsError`（按 `metrics.jsonl` 判），RC=1 |
| S3 | main 中途异常（预置 ckpt 目录触发 `FileExistsError`） | RC=1、Traceback 完整未被吞、`finally` 已写 `run_meta.json`。另：首轮 OOM 崩溃路径同样观测到 run_meta 落盘，异常路径收尾双重验证 |
| S4 | 不设 `TRAIN_RECORD_DIR` | RC=0、零文件产出（`find` 计数 0） |

**两处与计划 4.4 的偏离（均已处置）**：

1. **档位**：计划「本机 1 卡 `--fsdp-devices 1`」实测 OOM（`RESOURCE_EXHAUSTED`，训练状态
   全量放单卡超 46 GB 显存，与 g0/README A3「2 卡 fsdp2 每卡约 28 GB」一致）。经用户
   2026-08-30 拍板改 **2 卡 fsdp2 batch8** 重跑；smoke 只验记录器功能，拓扑不影响验收语义。
2. **S3 注入方式**：`kill -TERM` 在 Python 默认信号语义下不执行 `finally`（进程直接终止），
   计划提供的备选「注入异常」路线改用「预置 ckpt 目录触发 `FileExistsError`」实现，无需
   临时改代码。集群 Slurm 超时 kill 的真实路径上 `run_meta.json` 可能缺失，但 analyzer
   唯一硬依赖 `metrics.jsonl` 为逐步追加写、不受影响。
3. **analyzer 判据补写**：2 步数据需 `--warmup 0`（默认 warmup=50 直接断言失败）且需要
   稳态窗口内的 GPU 采样行（`gpu_util.csv`/`gpu_util_dense.csv` 同为硬依赖——计划 1.3
   「metrics.jsonl 唯一硬依赖」的说法不完全准确；生产路径 sbatch 采样器必在，S1b 以同款
   dense 采样器伴跑复现生产拓扑后判据全过）。

## AGENTS 18 两块一致性讨论（计划 4.6 适配口径）

- **链路图**：见 `appendix-d-datapath.md`（前后两图因数据链路零改动合并为一表 + 观测旁支
  差异说明；第 5 跳 shape/dtype 已按驱动日志实测回填，字节量合计 ≈ 38.9 MiB/batch）。
- **第一块（非训练轻量对拍）**：按计划 4.6 据实说明——本次改动不在数据链路上，字面的
  「新旧链路对拍」无对应物；以本 run vs G0b-r1 固化基线的 `INDEX_SEQ=PASS n=8072` 与
  `BATCH_DIGEST_CANONICAL rows=14 mismatch=0` 充当（index 序列 + 逐 batch 内容 + 逐键
  canonical 口径）。
- **第二块（本机训练梯度一致）**：不挂本 run（bench 路径结构性绕开新 `__main__`），挂
  C4 B 侧 1000 步真实训练（`v1-upstream-eq`），见该 run 留档。

## 产物清单

`records/`：`metrics.jsonl`、`param_checksums.jsonl`、`batch_digests.jsonl`、
`index_sequence.json`、`env.json`、`run_meta.json`（以上逐字节拷贝自
`v1-store/bench/2gpu-epoch-bench/v1-singlerun-g0/`）、`compare_vs_g0_r1.txt`、
`scalars_hex.tsv`、`preflight_baseline_env.txt`、`BASELINE_MANIFEST.json`（manifest 9 产物）。
driver 日志留 `v1-store/logs/v1-singlerun-g0-driver.log`（不进 git）。

## 下一步

C4 上游 main 对拍（`v1-upstream-eq`，launch.md 已具草待提交）：P2/P3 已 PASS，
先跑 P1（A 侧 2 步 harness 冒烟）再双侧各 1000 步串行。
