# v1-prod-60k 起跑档（D3：首次全量正式训练，60,000 步 + wandb online）

> 计划：`v5.1-prod-60k-wandb-plan.md`（第二版）D3 节。用户 2026-08-30 授权：
> 「直接提交 greatlakes 所有任务包括长训练，不需要再请示；先提 60k 完整训练，然后同时
> 提交 ab 对拍，抢出时间保证 60k 先开始」——即 D3 先于 D2 入队，D2 的放行闸语义由用户
> 明示改为事后核验（若 D2 FAIL，处置回用户拍板，不自行动 60k）。

## 可复现锚

- 代码切片：commitV5.1 `baeaee7`（train.py entity 覆盖、prod sbatch 四件闭环）；
  EXPECTED_GIT_HEAD 为本 launch 档提交后的 HEAD，经 `--export` 传入、sbatch 起跑闸
  断言 HEAD 未漂移 + 工作区 clean。
- 入口：`scripts/training/prod/gl_train_prod.sbatch` → `scripts/training/train.py`
  （单跑、`TRAIN_RECORD_DIR` 记录器、wandb 条件块 online）。
- 数据：`v1-store/datasets/4task-gl-framesamp`（framesamp packed 库，sbatch 默认）+
  `perceptual-framesamp-context.yaml`；权重 `pi05_base/params`；assets
  `v1-store/train-assets`。

## 提交命令（钉死口径，实际执行实录落 records/）

```bash
uv run --no-project --with pexpect python scripts/dataset/gl/gl_submit.py \
  "cd $REPO && sbatch --job-name=v1-prod-60k --time=96:00:00 \
     --export=ALL,RUN_NAME=v1-prod-60k,STEPS=60000,SAVE_INTERVAL=10000,LOG_INTERVAL=100,BATCH=64,FSDP=4,WORKERS=8,SEED=42,EPOCH_STEPS=6176,EXPECTED_GIT_HEAD=<launch 后 HEAD>,WANDB_PROJECT=robomme-framesamp,WANDB_ENTITY=hongzefu-university-of-michigan,WANDB_API_KEY=<key 不落档> \
     scripts/training/prod/gl_train_prod.sbatch"
```

- 资源：spgpu 4×A40 / 16C / 128G / **96h**（`--time` CLI 覆盖 sbatch 头部 12h）。
  速率底座 `v1-prod-trend-10h` 实测步时均值 4.814 s → 60k ≈ 80.23 h，walltime 1.20×。
- 超参口径（用户逐项拍板）：seed 42（覆盖脚本默认 335）、log_interval 100、workers 8、
  save/keep 10000（keep 由具名配置自带，6 ckpt ≈ 78 GB，单 ckpt 实测 13 GB）；
  `STEPS=60000` 走 `--export` 覆盖、全局默认不动（AGENTS 10）。
- wandb：project `robomme-framesamp`、entity `hongzefu-university-of-michigan`
  （train.py 经 `WANDB_ENTITY` 环境变量覆盖，默认仍 daiyp_umich）；API key 只经
  `--export` 环境变量传递，不落文件不进 git。

## 提交前 preflight（2026-08-30 实测，全绿）

- turbo 剩余 **6.0 TB**（`df -h`，需 ≥ 200 GB ✓）。
- `scontrol show partition spgpu` → **MaxTime=14-00:00:00** ≥ 96h ✓。
- 本机 wandb 探针：真 key 上报 `robomme-framesamp` PASS（probe run 即删）；顺手清理
  既有 `wandb-probe-delete-me`（run id oqhie0y8）✓。
- run_name 全新：`v1-store/train-records/` 与 `train-runs/` 无同名目录，wandb project
  无同名 run；用户已按 AGENTS 6 确认拟名 ✓。
- 队列：squeue 本人 0 在队 ✓；ControlMaster 存活、零认证提交 ✓。

## 起跑首小时 Monitor 判据

- `wandb.init` 成功行（真实 `init_wandb→_MetricsProxy` 链路首验；init 位于权重加载与
  JIT 之前，失败分钟级即死）。
- 首步步时 ≈ 4.8 s；`ENV_RC=0`。

## 判读与收官（写 result.md 时核对）

- `TREND_OK`（util 均值 + 前后半段趋势 + 0% 采样占比）+ dense 500ms 通道分段趋势；
  wandb 曲线与 `metrics.jsonl` 交叉核对。
- log100 下慢步/非慢步分层不可算：按 `v1-prod-trend-10h` 同款降级口径留档说明
  （AGENTS 16 该分项，用户已拍板接受）。
- 60k 语义前缀主张（非位级）：其余参数不变、仅延长 `num_train_steps` 时前 60k 个更新
  构成语义前缀（lr 为 warmup 10k 后恒定 5e-5，schedule 不依赖 num_train_steps）。
