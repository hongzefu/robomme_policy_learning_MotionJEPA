# v1-prod-100k 起跑档（100,000 步正式训练，其余口径逐项同 v1-prod-60k）

> 计划：`~/.claude/plans/job-v1-prod-60k-job-snazzy-floyd.md`（用户 2026-09-01 批准）。
> 用户指令原话：「再提交一个100k的训练的任务 其他配置都保持一致」；三项拍板：run_name
> `v1-prod-100k`、walltime `168:00:00`、wandb 继续上报 `robomme-framesamp`。
> 提交时 `v1-prod-60k`（job 59345249）仍在 gl1524 运行（30.5k/60k 步），两 job 并行。

## 可复现锚

- 代码切片：与 `v1-prod-60k` 完全相同的训练代码（commitV5.1 `baeaee7` 之后只有 docs 提交，
  `scripts/training/` 与 `src/openpi/` 零改动）。`EXPECTED_GIT_HEAD` 为本 launch 档提交后的
  HEAD，经 `--export` 传入、sbatch 起跑闸断言 HEAD 未漂移 + 工作区 clean。
- 入口：`scripts/training/prod/gl_train_prod.sbatch` → `scripts/training/train.py`
  （单跑、`TRAIN_RECORD_DIR` 记录器、wandb 条件块 online）。**本轮不改 sbatch、不改任何
  代码/配置默认值**；`STEPS=100000` 走 `--export` 覆盖（AGENTS 10，全局默认不动）。
- 数据：`v1-store/datasets/4task-gl-framesamp`（framesamp packed 库，sbatch 默认；
  1600 episodes / 483,291 timesteps / 395,289 exec samples，VERIFY_PACK=PASS）+
  `perceptual-framesamp-context.yaml`；权重 `pi05_base/params`；assets `v1-store/train-assets`。

## 提交命令（钉死口径，实际执行实录落 records/）

```bash
uv run --no-project --with pexpect python scripts/dataset/gl/gl_submit.py \
  "sbatch --job-name=v1-prod-100k --time=168:00:00 \
     --export=ALL,RUN_NAME=v1-prod-100k,STEPS=100000,SAVE_INTERVAL=10000,LOG_INTERVAL=100,BATCH=64,FSDP=4,WORKERS=8,SEED=42,EPOCH_STEPS=6176,EXPECTED_GIT_HEAD=<launch 后 HEAD>,WANDB_PROJECT=robomme-framesamp,WANDB_ENTITY=hongzefu-university-of-michigan,WANDB_API_KEY=<key 不落档> \
     scripts/training/prod/gl_train_prod.sbatch"
```

与 `v1-prod-60k` 提交串逐字对照，**差异只有三处**：`--job-name` / `RUN_NAME` 改为
`v1-prod-100k`、`STEPS=100000`、`--time=168:00:00`。其余（batch 64、FSDP 4、workers 8、
seed 42、log 100、save 10k、epoch_steps 6176、数据路径、wandb project/entity）逐字相同。

- 资源：spgpu 4×A40 / 16C / 128G / **168h**（`--time` CLI 覆盖 sbatch 头部 12h；分区
  MaxTime 14-00:00:00）。速率底座取 `v1-prod-60k` 本身：metrics.jsonl 0→30,500 步
  wall_time 差 149,674 s → **4.907 s/步**；100k 外推 **136.3 h**，walltime 裕度 1.23×
  （60k 为 1.18×）。**168h 远超 greatlakes.md 调试包络（≤2 GPU / ≤30 min），2026-09-01
  经用户 AskUserQuestion 显式放行**（三选一选 168:00:00）。
- checkpoint：`SAVE_INTERVAL=10000`，具名配置自带 `keep_period=10_000`（`checkpoints.py`
  的 `max_to_keep=1` 只删非 keep_period 整倍数的），10 个 ckpt 全保留；单 ckpt 按 60k 实测
  13 GB → 合计 ≈ 130 GB。turbo 提交前剩余 6.0 TB。
- wandb：project `robomme-framesamp`、entity `hongzefu-university-of-michigan`；API key
  只经 `--export` 环境变量传递，不落文件不进 git，gl_submit 的 `[remote cmd]` 回显经
  `sed` 打码后才落到会话输出。

## 语义说明（用户知情）

同 seed 42、同配置、同数据顺序，100k run 的前 60k 个更新与 `v1-prod-60k` 构成**语义前缀**
（lr 为 warmup 10k 后恒定 5e-5，`decay_steps=1_000_000`、`decay_lr=peak_lr`，schedule 不
依赖 `num_train_steps`）。这是从头重算而非续跑——checkpoint 只存 EMA、丢 AdamW 动量与
warmup 计数，`train.py` 已 fail-loud 禁用 `--resume` / `--overwrite`。跨节点非位级一致，
只主张语义前缀。

## 提交前 preflight（2026-09-01 实测）

- ControlMaster 存活（`ssh -O check greatlakes` → `Master running`），全程零认证，未触发 Okta。
- chaijy2 配额（skill `greatlakes-usage`）：上限 GPU 20 / MEM 960 G / CPU 80；RUNNING 占用
  GPU 8 / 384 G / 36 CPU（本人 60k 占 4 / 128 G / 16），**剩余 GPU 12 / 576 G / 44 CPU**，
  本 job 4 / 128 G / 16 装得下，不会 `AssocGrp*`。
- spgpu 全局：240 张 A40 已分配 212、空闲 28；能凑 4 张的节点只有 gl1514（5 GPU 空闲，
  但空闲 CPU 仅 11 < 16），**预期提交后 `PENDING (Resources)` 排队，起跑时间不可预测**；
  全局 746 个 job 排队、真抢卡约 92 GPU。
- NFS 互扰：60k 稳态 `server_read` 仅 5.3 MB/s（36 GB 库已进 128 G 页缓存后近乎零读盘），
  第二个 job 在各自节点各自暖缓存，互扰可忽略。
- run_name 全新：`v1-store/train-records/` 与 `train-runs/mme_vla_suite/` 下无
  `v1-prod-100k`；用户已按 AGENTS 6 确认拟名。
- 队列：`squeue -u hongzefu` 仅 59345249（v1-prod-60k，RUNNING 1-17:45）。

## 排队期间 commit 冻结（硬规则）

带 `EXPECTED_GIT_HEAD` 的 job 在排队期间任何 commit 都会让起跑闸 fail-fast
（2026-08-31 job 59345261 实测）。**从 sbatch 提交到日志出现 `ENV_RC=0`（起跑闸已过）
之间，仓库冻结一切 commit**；`records/submit.md` 先写盘、起跑后再提交。prod sbatch 只在
起跑瞬间查 HEAD，之后 commit 安全（60k 运行期间已有多次 docs 提交为先例）。

## 起跑首小时 Monitor 判据（同 v1-prod-60k）

- `wandb.init` 成功行（run URL 出现、无 `wandb: ERROR`）。
- `ENV_RC=0`。
- `metrics.jsonl` 前 500 步逐 100 步区间步时 ≈ 4.9 s，100k 外推 < 168 h。

## 判读与收官（写 result.md 时核对）

- `TREND_OK`（util 均值 + 前后半段趋势 + 0% 采样占比）+ dense 500ms 通道分段趋势；
  wandb 曲线与 `metrics.jsonl` 交叉核对。
- log100 下慢步/非慢步分层不可算：按 `v1-prod-trend-10h` 同款降级口径留档说明
  （AGENTS 16 该分项，用户已拍板接受）。
- 与 `v1-prod-60k` 前 60k 步曲线并排对照，核对语义前缀主张（loss / grad_norm 走势一致）。
