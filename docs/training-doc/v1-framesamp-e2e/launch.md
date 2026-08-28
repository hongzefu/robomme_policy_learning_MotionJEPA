# v1-framesamp-e2e（S8b GL e2e 收官测试）launch 记录

对应 `v2-framesamp-restructure-plan.md` 阶段 4 S8b（D 节）。留档体例沿
`v1-framesamp-dl` 先例：一目录覆盖本批全部 run。

## 拍板与豁免（2026-08-27，用户 AskUserQuestion）

- 「GL e2e 收官测试 评估是否可以同时提交」→ 拍板**依赖链串行**：现在全部提交、
  `--dependency=afterany` 链 T1→T2→T3→COLDHOT 严格串行执行——立即排队但一次只跑
  一个（S8b 是负载敏感的性能验收，与 S5/S6 bitwise 并行是本质不同场景；避免同节点
  共驻/互相预热污染 `E2E_ACCEPT`）。与 S8a 的时间重叠风险接受（S8a 单 job ≤30 min，
  4 卡 job 排队大概率更久）。
- 资源审批「四个全批」：放行记录见仓库根 `greatlakes.md`「放行记录（robomme
  framesamp v2 计划，2026-08-27）」。
- **S9 G2-speed 用户决策暂时不跑**（计划 T8/E 表同步标注）。
- 条件档 T4 w16 / T5 w4c8 未批未提交，视 T1–T3 结果另行审批。

## run 表（全部 4×A40/16C/96G，packed 库，`ANALYZE_ACCEPT=1`）

| run_name | workers | 步数 | walltime | seed | 依赖 |
|---|---|---|---|---|---|
| `v1-framesamp-e2e-w4c16`（T1，最重要：官方默认档还需不需要调） | 4 | 600 | 2h | 320 | 无 |
| `v1-framesamp-e2e-w8c16`（T2，直接对 v1-e2efix-w8c16） | 8 | 600 | 2h | 321 | afterany:T1 |
| `v1-framesamp-e2e-w2c16`（T3，探底） | 2 | 600 | 2h | 322 | afterany:T2 |
| `v1-framesamp-e2e-w4c16-coldlike` + `…-hot`（COLDHOT=1 同 allocation 先 C1 后 H1） | 4 | 300+300 | 4h | 323 | afterany:T3 |

- 入口 `gl_e2e_fix.sbatch`（S7.5 参数化）：`MMEVLA_DATA_BACKEND=packed`（显式）、
  `DATASET_PATH=…/4task-gl-framesamp`（status=verified）、`SAVE_INTERVAL=1000`
  （600 步内零摘要，性能口径不受污染）、`MMEVLA_FRAMESAMP_VERIFY` 缺省 fast——
  **冷态自证：allocation 内不跑 full 校验、无本地复制预热**（env.json 硬字段）。
- 遥测：dense 500ms + legacy 15s 双通道 GPU util、NFS server_read、meminfo
  （Cached+pgmajfault）、compute_apps（worker CUDA context 存证）。
- seed 320–323（避开已用 42/200–205/210–212/310–313）。

## 判据（D 节）

- T1–T3：`analyze_gpu_util.py --accept` 机器判定单行
  `E2E_ACCEPT=PASS|FAIL`——必达五项：步时中位 ≤5.00 s、util 稳态均值 ≥90%、
  0% 采样 ≤5%、慢步(>8s)墙钟 ≤5%、epoch(6,176 步) ≤8.6 h；报告须附「距 100%
  残差分解」。附加判据（`--extra`）：w4 与 w8 步时差 ≤3%；NFS server_read 稳态
  期望 ≈17–25 MB/s，>65 MB/s 视为读放大信号。
- COLDHOT：`(C1稳态−H1稳态)/H1 ≤ 15%`（cold-like 口径，meminfo/pgmajfault 证据链）。
- 对照组（历史口径实测，不重测）：v1-e2e-b64 6.933 s/69.7%、w8c16 5.301 s/71.2%、
  w12c16 5.319 s、w16c16 5.327 s、compute-only 4.778 s。

## 起跑环境

- **起跑 HEAD**：本 launch.md 的 docs commit 本身（结果留档回填全 sha + 各 job id）
- ControlMaster 复用零认证；提交器 `gl_submit.py`；提交命令模板：

```bash
uv run --no-project --with pexpect python scripts/data-preprocess-GL/gl_submit.py \
  "sbatch --parsable --job-name=<run_name> --time=<walltime> [--dependency=afterany:<prev>] \
   --export=ALL,WORKERS=<W>,BENCH_SEED=<S>,TAG=<run_name 基名>,ANALYZE_ACCEPT=1[,COLDHOT=1],DATASET_PATH=…/4task-gl-framesamp,MMEVLA_DATA_BACKEND=packed \
   scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch"
```
