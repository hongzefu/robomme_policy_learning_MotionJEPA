# v1-l0-gauge：L0 修量具后的两档基准（S0c + S0）

计划权威源：[`v1-95util.md`](../../../v1-95util.md) L0 节。本 job 是 v1-95util 顺序执行的
第 2 步：量具修复（commitV3.6）后，同一 allocation 同节点先后跑两档 600 步 e2e 基准，
把「量具口径差异」归因出来，并以 log100 生产档重立 speed 基准锚点 S0。

## 两档定义

| 档 | run_name | seed | 口径 |
|---|---|---|---|
| S0c（对照） | `v1-l0-gauge-log1` | 330 | `LOG_INTERVAL=1`，与旧 89.2% 同口径；不判 accept，只出报表与 log1 诊断区（慢步分层/相位分组） |
| S0（新锚） | `v1-l0-gauge-log100` | 331 | `LOG_INTERVAL=100` 生产口径（`BENCH_PERF_MODE=1`，摘要全禁）；判 `E2E95_ACCEPT` 五项 |

共同配置：w8c16、packed（`4task-gl-framesamp`，status=verified）、prefetch=2（torch 现状
默认，实效值入 env.json）、线程现状（OMP/MKL=16，OPENBLAS/NUMEXPR 未设——L0 不动线程，
这是 L1 的事）、batch 64、fsdp 4、600 步、`SAVE_INTERVAL=1000`、dense 500ms + legacy 15s
双通道采样。seed 330/331 互不相同且避开已用 42/200-205/210-212/320-325（防同节点 page
cache 跨 run 污染归因）。

## 判据

- S0 判 `E2E95_ACCEPT` 五项（analyzer `--accept`）：util_mean ≥95%、zero_pct ≤3.8%、
  active_util ≥98%、step_mean ≤5.013 s（稳态真实墙钟均值）、epoch_mean ≤8.6 h。
  **全过 → 双 seed 复验 → 收官（L1–L4 全不做）；未过 → 进 L1。**
- S0c 不判 accept（旧口径预期不达标，判了会污染 job 退出码）；其价值是与 S0 的差值
  =「量具打断」的真实幅度，以及 log1 诊断区（慢步分层、`step%8` 相位分组）的细节归因。
- 新旧 analyzer 对拍：job 完成后在本机用旧版 analyzer（`git show` 取 commitV3.6 前一版）
  对 `v1-l0-gauge-log1` 记录目录双跑，把统计口径差异（中位数→墙钟均值等）单独归因。

## 资源与放行记录

- 资源：4×A40 / 16C / 96G / **04:00:00**（两档各 600 步 ≈ 各 50-60 min + 双次冷启动），
  超出 `greatlakes.md` 调试硬限（≤2 GPU / ≤30 min）。
- **放行**：用户 2026-08-28 本轮原话「greatlakes你有完整授权 可以自由提交任意长度的job」；
  run_name 与 seed 同日经 AskUserQuestion 逐项确认（「按此提交」）。

## 提交命令

起跑 HEAD：本 launch.md 所在 commit（含 L0 三个量具文件改动）。ControlMaster 复用零认证。

```bash
uv run --no-project --with pexpect python scripts/data-preprocess-GL/gl_submit.py \
  "sbatch --parsable --job-name=v1-l0-gauge --time=04:00:00 \
   --export=ALL,WORKERS=8,TAG=v1-l0-gauge,BENCH_SEED=330,BENCH_SEED2=331,GAUGE_DUAL=1,ANALYZE_ACCEPT=1,DATASET_PATH=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-gl-framesamp,MMEVLA_DATA_BACKEND=packed \
   scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch"
```

提交后回填：jobid、节点、起跑 commit 全 sha（见 result.md）。
