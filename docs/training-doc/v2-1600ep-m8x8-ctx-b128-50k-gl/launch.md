# 四任务×400 context 512 训练（GreatLakes 4×A40，bs128 = 每卡 16 × 累积 2，50k 步）

> 计划：仓库根 `1111-gl-4task400-ctx512-1024-b128-50k-plan.html`（r3 + 第二部分「九 实施中调整」）。本文件随 `commitV4.6.5Beta` 提交，起跑后补「实际起跑」一节；结果写 `result.md`。

## 目的

在 modulation 对照所用的同一训练库上补训 context 模式 budget 512，与 `v2-1600ep-m8x8-modul-b128-60k（第 50000 步，30.5%）` 直接比较。用户原话：「我要训练的是这个四 task……context 模式下的这个呃 memory budget 五一二和一零二四，然后都是 batch size 一二八，然后都是五十 k step。」「然后两个都 query 四卡」「对的，要用梯度累积来」「好的就保持吧」。

## 版本与代码状态

- 训练代码：`commitV4.6.5Beta`（本提交）。梯度累积 `commitV4.6.1`/`4.6.4`、context 1024 守卫 `commitV4.6.0`、完整状态续跑 `commitV4.6.2`、GL 入口 `commitV4.6.3`。
- 集群执行副本 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA` 以 detached HEAD checkout 到 Beta，`.agentmetarules` 子模块同步到 Beta 指针；`uv.lock` 与 `pyproject.toml` 相对其原 HEAD `124b145` 无差异，沿用该副本 `.venv`（钉死 `uv-python/cpython-3.11.14`）。
- 本机前置闸门：`GRAD_ACCUM_IDENTITY=PASS steps=20 values=100 hex_mismatch=0`（新旧代码 K=1 逐位）；`GRAD_ACCUM_LINEAR=PASS ks=2,4 max_rel=2.751e-07`；`MICRO_SPLIT=PASS k=2 per_device=16 dup=0 miss=0`；`CTX1024_GUARD=PASS tests=38`、`REJECT_INPUTS=PASS cases=90`；`RESUME_UNIT=PASS tests=4`；V10 `RESUME_STATE=PASS step=81 leaves=177 sha_verified=1 resumed_steps=157 epochs_crossed=2 nonfinite=0`（迷你库、K=1，见计划「九」A3）。V4（K=1 对 K=2 行为）在 GL 冒烟前跑。

## 启动与配置还原

- 配置：`git show <Beta>:src/mme_vla_suite/training/config.py` 中 `mme_vla_suite_b128_60k`（batch 128、peak/decay lr 5e-5、warmup 5000、decay 60000、seed 42、save/keep 5000），覆盖项：`--fsdp-devices 4 --num-workers 8 --grad-accum-steps 2 --num-train-steps 50001`（最后 ckpt 50000，学习率曲线与对照 run 第 50000 步相同）。
- history：`perceptual-framesamp-context-8frame-8x8.yaml`，sha256 `299b6d3f2b52c7b3c1894f8594183a3140d0d2a6b91250a2eb8e568c528344d8`；`integration_type: context`、`memory_token_dim: 2048`、`motion.enabled: false`。
- 数据：`<GL_REPO>/v1-store/datasets/4task-v2-1600ep-604f16da`（HF bucket `HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1` 训练子集，GL CPU job 65945136 拉取：`TURBO_LIB=PASS files=202 mismatches=0 pkl=605611`，553,230,306,097 B，下载 3383 s），`--dataset-path <DS>/framesamp-8x8`；norm_stats sha256 `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`。
- 起跑合同：`DATA_LINEAGE=NA encoder=none motion_enabled=false vla_input_pin=655e64b4fd033aea1ff79a1e5ad47f886a148592496759ed2a667fe167c57d86`（用户裁决「可以按 NA」），`PAIRED_RUNS_SAME_LIB=PASS pin=655e64b4…`（两条 run 同库，EXPECT_LIB_PIN 同值）。
- 资源：占位 `65943150`（chaijy2/spgpu，4×A40，16 CPU，128G，192 h，`gpu_cmode=shared`）。
- 命令（经 aspen tmux 内 `ssh greatlakes`）：

```bash
srun --jobid=65943150 --overlap --exact --ntasks=1 --cpus-per-task=16 --gpu_cmode=shared \
  /usr/bin/env RUN_NAME=v2-1600ep-m8x8-ctx-b128-50k-gl CONFIG=mme_vla_suite_b128_60k HISTORY=perceptual-framesamp-context-8frame-8x8.yaml \
  GRAD_ACCUM=2 MAX_STEPS=50001 FSDP=4 WORKERS=8 TRAIN_HEAD=<Beta> \
  EXPECT_LIB_PIN=655e64b4fd033aea1ff79a1e5ad47f886a148592496759ed2a667fe167c57d86 \
  bash /nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/scripts/training/gl/run_in_hold_4task_ctx.sh
```

- 产物：ckpt `<GL_REPO>/v1-store/train-runs/mme_vla_suite_b128_60k/v2-1600ep-m8x8-ctx-b128-50k-gl/{5000,…,50000}` 与 `resume/`（完整状态，只留最新 1 份，单份约 45 GiB）；日志 `<GL_REPO>/v1-store/logs/v2-1600ep-m8x8-ctx-b128-50k-gl.log`；指标 `<GL_REPO>/v1-store/train-records/v2-1600ep-m8x8-ctx-b128-50k-gl/attempt01/metrics.jsonl`。
- 盯盘：`EXIT_CODE=`、`Step [0-9]+000:`、`RESOURCE_EXHAUSTED`、`Traceback`、`nan`、`Killed`、`=FAIL`。
- tmux 会话清单（aspen）：`g4c-train-512`（正式）、`g4c-smoke-512`/`g4c-v4`（冒烟）。
