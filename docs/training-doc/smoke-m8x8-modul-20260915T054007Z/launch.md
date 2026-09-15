# modulation 8×8 的 b128 四卡 smoke

本 run_name 为 `smoke-m8x8-modul-20260915T054007Z`，实施正式计划 B 节的 20 步功能验收，物理 GPU 4,5,6,7，batch 128、worker 8、fsdp 4、seed 42。仅以 CLI 覆盖 `num_train_steps=20`、`log_interval=1`、wandb 关闭，其他参数来自 `mme_vla_suite_b128_60k`。本模型形状首次编译可能超过 5 分钟，因此预先按完整运行留档，不依赖短测豁免。

数据与 SHA、正式参数及隔离方案见 [正式起跑留档](../v2-1600ep-m8x8-modul-b128-60k/launch.md)。F2/F3 和两档 modulation 追溯验证均已通过。候选源码含 `63858f5`；起跑 HEAD 为包含本留档的 clean HEAD，准确值由日志 `TRAIN_HEAD=` 和 preflight 记录。

完整命令见正式目录的 `records/smoke-runner.sh`。主副本运行，不设置 PYTHONPATH；沿用现有 .venv，uv 不同步依赖；缓存位于仓库 v1-store。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
RUN=smoke-m8x8-modul-20260915T054007Z
TRAIN_HEAD=$(git rev-parse HEAD)
HC_SHA=$(sha256sum src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8.yaml | cut -d' ' -f1)
tmux new-session -d -s m8-smoke "bash /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/m8-smoke-runner.sh '$RUN' '$TRAIN_HEAD' '$HC_SHA'"
tmux has-session -t m8-smoke
```

本次会话清单仅 `m8-smoke`。日志 `v1-store/logs/m8-smoke.log`，checkpoint 根 `v1-store/train-runs/mme_vla_suite_b128_60k/smoke-m8x8-modul-20260915T054007Z`，记录根 `v1-store/bench/smoke-m8x8-modul-20260915T054007Z`；固定编译缓存 `v1-store/cache/jax/m8-smoke` 留作复用。

验收必须包含 `PREFLIGHT=PASS n=25`、20 步全部有限、`Integration Type: modulation`、`EXIT_CODE=0`。checkpoint 19 的 norm_stats SHA 必须为新库 `856c75ea…`；provenance 的 motion 关闭且 manifest 为 `4cd5a170…`。参数树必须 `n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0`，并正面核对六条 modulation 参数路径存在。归档日志、指标、provenance 和参数树结果后，只清理本 run 的 checkpoint 与 bench 根。
