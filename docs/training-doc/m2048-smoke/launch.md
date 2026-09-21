# 2048八卡生产容量检查

用户已要求完整实施并允许使用全部八卡。本run名为m2048-smoke，会话同名；只在全部正确性门通过、其余本轮GPU任务结束后起跑。必须从主副本clean HEAD执行，实际提交与UTC由runner写入日志；不预填本文自己的提交。环境为8×A100-SXM4-80GB、AWS /dev/md0 XFS NVMe RAID，依赖uv.lock保持不变。

入口为已提交scripts/training/prod/run_modul2048.sh的smoke模式。实际配置仍为mme_vla_suite_b128_80k，2048新YAML、b128/w16/FSDP8/seed42、LR5e-5/warmup5000/EMA0.999；仅启动覆盖20步、log_interval1、关闭W&B。显式unset XLA_FLAGS，TRAIN_TIMING_STEPS=0，不启用profiler和完整状态摘要。该20步仅验编译、容量、有限性和真实保存，不用于80k ETA。

数据只用4task-v2-1600ep-604f16da/framesamp-8x8；清单file SHA=df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482，norm_stats SHA=856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173，YAML SHA=42813c7e06840e511f8a382f14954edcaad8d4fd5b320b4323fea94979cda4de。输出根为v1-store/train-runs/mme_vla_suite_b128_80k/m2048-smoke，记录为v1-store/bench/m2048/m2048-smoke，拒绝已存在的目录、覆盖和续训。

从detached tmux m2048-smoke运行下列入口。runner内部已有pipefail、PYTHONUNBUFFERED、tee及EXIT_CODE，日志为v1-store/logs/m2048-smoke.driver.log；500ms GPU采样器记录精确PID并在退出时回收，主机RSS、shm及磁盘采样由包装器记录。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
M2048_HEAD=$(git rev-parse HEAD)
bash scripts/training/prod/run_modul2048.sh smoke m2048-smoke "$M2048_HEAD" 42813c7e06840e511f8a382f14954edcaad8d4fd5b320b4323fea94979cda4de
```

正常退出后CPU执行check_32frame_modul.py capacity，传本run records、driver日志及全新capacity.json。要求20步五标量有限、实际state_step20保存到19、加载后完整61叶及十个记忆叶、配置budget2048/motion关闭、shm峰值比例≤0.70。日志退出码、清洗结果和原始指标全部归档；核实归属后只清理本run临时checkpoint目录，保留记录。
