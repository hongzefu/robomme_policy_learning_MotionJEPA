# v1-prod-trend-10h：首次正式训练（10 小时利用率趋势观测）

计划权威源：`~/.claude/plans/worker8-packed-run-name-smooth-flute.md`（用户 2026-08-28 批准）。

## 这是什么 run，不是什么

- **是**：GPU 利用率在**长时间尺度**上能否稳住的趋势探针；同时是本仓库**正式训练链路
  的首次通电**（此前 29 个留档全是 ≤1000 步的基准与一致性验证，`v1-store/train-runs/`
  下零个 checkpoint）。
- **不是**：一个训好的模型。lr schedule 为 `warmup_steps=10000`、
  `peak_lr=decay_lr=5e-5`（warmup 之后恒定，非真正余弦衰减），只跑 7000 步意味着学习率
  从 5e-9 线性爬到 **3.5e-5 就停了、从未到达峰值**，区间均值仅峰值的 35%。7000 步 =
  配置量 80k 步的 8.75% = 1.13 个 epoch。**不得用于 manual_evaluation 或对外比较。**
- **不做续跑**（用户 2026-08-28：「不考虑继续跑」）。

## 为什么不能用现成的 finetune 脚本

`scripts/finetune_mme_vla_suite.sh` 三条全踩：

| 坑 | 官方脚本 | 本 run | 后果 |
|---|---|---|---|
| worker 数 | `--num-workers=4` | **8** | 历史 4 worker 档实测 util 仅 69–70% |
| 数据后端 | 不设 → 默认 legacy | **packed**（显式 export） | packed 快 3.4–3.6× |
| 入口 | `scripts/train.py` 双 `main()` | **`prod_train_once.py` 单 `main()`** | 双 main 第二次必 `FileExistsError` |

`scripts/train.py` **一字未改**——薄启动器只替换模块级 `train.wandb` 为指标记录代理
（只读观测），不 patch `save_state`（bench 那样会把 checkpoint 变成空操作、白跑）。

## 配置

| 项 | 值 |
|---|---|
| run_name / exp_name | `v1-prod-trend-10h` |
| seed | 335（避开已用 42 / 200–205 / 210–212 / 320–325 / 330–331 / 334） |
| 步数 | 7000（跨 epoch 边界 6176，最后 824 步在第 2 个 epoch） |
| batch / fsdp / workers | 64 / 4 / 8 |
| log_interval | **100**（生产口径；L0 已证明这是维持 99% 的必要条件） |
| save_interval | **2000**（保存 4 次：2000/4000/6000/6999） |
| 后端 / 数据集 | packed / `v1-store/datasets/4task-gl-framesamp` |
| wandb | 关（`WANDB_MODE=disabled` + `--no-wandb-enabled`） |
| 资源 | 4×A40 / 16 CPU / **128G** / `--time=12:00:00` |

**时间预算**（按 S0 实测 4.756 s/步；`sacct` 反推启动开销仅 325 s）：启动 5.4 min +
训练 9.25 h + 4 次保存约 4 min + 收尾分析约 5 min ≈ **9.5 h**，对 12 h walltime 留 2.5 h 裕度。

## 判据

**`TREND_OK=PASS`**（用户 2026-08-28 拍板）：
1. 全程 util 均值 **≥95%**（dense 500ms 稳态口径，同 E2E95 阈值）；
2. **无单调下滑**——按稳态窗口时间中点切两半，后半段 util 均值不得低于前半段 **1pp** 以上。

`analyze_gpu_util.py --trend --trend-step 1000` 输出分段走势表与该判定行。
**长训不带 `--accept`**：`E2E95_ACCEPT` 的五项阈值是按「600 步、全程不落 checkpoint」
标定的，对长训判 FAIL 会让脚本非零退出、掩盖训练本身的成败。

附带观测（非判据）：4 次 checkpoint 的实测停顿秒数（此前全是推算）、跨 epoch 边界
前后有无掉档、cgroup anon 峰值。

## 资源放行记录

- 4×A40 / 128G / 12 h **远超** `greatlakes.md` 调试包络（≤2 GPU、≤30 min）。
- **放行**：用户 2026-08-28 原话「greatlakes你有完整授权 可以自由提交任意长度的job」；
  同日就本 run 追加拍板「提交128g版本」「只跑10小时 看趋势」。
- 提交前实测复核：chaijy2 mem 配额 960G / 已用 456G；spgpu 上同时满足「≥128G 空闲内存
  + ≥16 空闲 CPU + ≥4 张空闲 A40」的节点有 5 个（gl1513 / gl1525 / gl1512 / gl1524 / gl1529）。

## 提交命令

```bash
uv run --no-project --with pexpect python scripts/data-preprocess-GL/gl_submit.py \
  "sbatch --parsable --job-name=v1-prod-trend-10h \
   --export=ALL,RUN_NAME=v1-prod-trend-10h \
   scripts/train-prod/gl_train_prod.sbatch"
```

其余参数（步数 7000、seed 335、workers 8、save_interval 2000 等）均为 sbatch 内默认值。

## 产物落法

| 产物 | 位置 |
|---|---|
| checkpoint（EMA 权重，约 13 GB×常驻 1–2 份） | `v1-store/train-runs/mme_vla_suite/v1-prod-trend-10h/` |
| 全分辨率记录（含 dense CSV 约 12 MB） | `v1-store/train-records/v1-prod-trend-10h/` |
| 归档（dense 降采样 10×） | `docs/training-doc/v1-prod-trend-10h/records/` |
| jax 编译缓存（软链，不删） | `v1-store/cache/jax/v1-prod-trend-10h/` |

起跑 HEAD：本 launch.md 所在 commit（结果回填见 result.md）。
