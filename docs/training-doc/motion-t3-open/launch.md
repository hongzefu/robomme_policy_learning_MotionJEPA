# motion-t3-open（T3 真实训练端到端，open 侧 1000 步，保存最终 EMA checkpoint）launch

- **目的**：`motion-memory-plan.md` 5.3 T3 / 2.10——closed / open 两份不可变 YAML 各在 `4task-motion-40ep/framesamp` 跑 1000 步 × batch 8（同 seed、同样本顺序，
  8,000 样本在 11,530 的单 epoch 内），保存最终 EMA checkpoint（目录 999）。硬闸：`T3_COMMON_INIT`（起跑前）→ `T3_SMOKE` → `T3_TOKEN_TRACE` → `T3_MECHANISM`；
  `T3_PHASE_REPORT` 完整性硬条件、均值只报告；最后 200 步 loss 只记 `T3_EFFECT_OBS`。跨侧 verdict 与比较只归档在 open run 的 `result.md` / `records/comparison/`。
- **run_name**：`motion-t3-open`；`EXP_NAME=RUN_TAG=motion-t3-open`；history config **固定** `perceptual-framesamp-context-motion.yaml`（红线 16，不手改开关）。
- **commit**：S2 合入后的 clean HEAD（sha 在 `result.md` 回填）；A20 已由用户拍板降为观察项（比值 0.166，不改模型）。
- **口径**：`run_2gpu_epoch_bench.sh`，b8 / seed 42 / fsdp 2 / WORKERS 4 / STEPS 1000 / SAVE_INTERVAL 100 + EXTRA_DIGEST_STEPS 299（摘要步 {0,100,…,900,299,999}，
  输入摘要 {0,1,2,100,…,900,299,999} 共 14）/ 确定性档 / CUDA 0,1；`BENCH_SAVE_FINAL_CKPT=1 BENCH_FINAL_STEP=999`（末步经原版 save_state 写目录 999，EMA，
  `final_checkpoint.json` 记 `checkpoint_id=999,state_step=1000,param_kind=ema`）；run 根写 `history_config.resolved.yaml` / `.sha256` / `motion_provenance.json`。
  数据根本机 NVMe，epoch 样本数从 store meta 读 11,530；1000×8 = 8,000 < 11,530。

## 起跑命令

```bash
tmux new-session -d -s motion-t3-open "set -o pipefail; cd /data/hongzefu/robomme_policy_learning_MotionJEPA; echo HEAD=\$(git rev-parse HEAD); \
  STEPS=1000 SAVE_INTERVAL=100 EXTRA_DIGEST_STEPS=299 WORKERS=4 WARMUP_STEPS=50 HISTORY_CONFIG=perceptual-framesamp-context-motion.yaml \
  BENCH_SAVE_FINAL_CKPT=1 BENCH_FINAL_STEP=999 EXP_NAME=motion-t3-open RUN_TAG=motion-t3-open \
  DATASET_PATH=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-40ep/framesamp \
  XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
  bash scripts/training/g0/run_2gpu_epoch_bench.sh 2>&1 | tee v1-store/logs/motion-t3-open-driver.log; echo \"EXIT_CODE=\$?\" >> v1-store/logs/motion-t3-open-driver.log"
```

## 判读（`scripts/training/tests/motion_gates_model.py`）

- 起跑前：`--gate t3common`（GPU，同进程初始化两态 TrainState；须显式 `JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1`——脚本为 M 系列默认 `setdefault(JAX_PLATFORMS, cpu)`，不设则 fsdp=2 报 `Number of devices 1`）→ `T3_COMMON_INIT=PASS common_mismatches=0 open_only_params=4`，reference 写 `v1-store/reports/motion/t3_common_init_reference.json`。
- 训练后：`--gate t3verifyinit`（step-0 记录命中 reference）；`T3_SMOKE`：1000 步 loss 全有限、四个新参数叶初/末态 sha 不同（open）、`n_keys=16/12`、`n_leaves=193/177`、
  8,000 个实际 index 的有效窗口数由 oracle 重算与训练记录逐项相同；`--gate t3trace`（open/closed records）；`--gate t3mechanism`（open records + reference，GPU）；
  两侧都完成后 `--gate t3phase --open-ckpt … --closed-ckpt …`（run `motion-t3-phase`）。
