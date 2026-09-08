# vulkan-fix-30ep —— 起跑记录（launch）

> 目的：验证 commitV7.2 的修法——评估启动加 `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192`——在**真实评估路径**上让单进程连评超过 27 集不崩。用生产 policy server（`scripts/training/serve_policy.py`，`mme_vla_suite` 配置，sidecar 自动拉起）+ 主线 `examples/robomme/eval.py` 单进程连评 ButtonUnmask test split 30 集。
> 根因与排查：`docs/training-doc/tic-vulkan-makeenv/result.md`（每集 make_env 重建渲染 Context，NVIDIA Vulkan ICD 反复 dlopen/dlclose 每轮净漏 64 B 静态 TLS，默认 512 B 撑到第 27 集）。
> 起跑 commit：`cfe2f75`（launch 预提交后的 clean HEAD；修法在 commitV7.2 8c20c78）。2026-09-08。

## 环境与介质

- **环境 B**：仓库根 `/scratch/hongze/robomme_policy_learning_MotionJEPA`；`/nfs/turbo`、`/data/hongzefu`、`~/.ssh/config` 均不存在；8 × A100-SXM4-80GB。介质 AWS 本地 NVMe RAID（`/dev/md0`）。
- 仿真：micromamba `robomme`（`/scratch/hongze/micromamba/envs/robomme/bin/python`），驱动 595.71.05、glibc 2.34、SAPIEN 3.0.3。

## 命令（tmux `tic-vulkanfix`，policy + 仿真 GPU 0，sidecar 按快照落 GPU 1，端口 8124）

```bash
bash v1-store/reports/tic/run_G.sh   # 日志 v1-store/reports/tic/G/{eval-probe.log,server.log,eval.log}
# G1 CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run --no-sync python scripts/training/serve_policy.py --seed=42 --port=8124 policy:checkpoint --policy.dir=v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999 --policy.config=mme_vla_suite
# G2 cd examples/robomme && CUDA_VISIBLE_DEVICES=0 GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192 <robomme python> eval.py --args.port=8124 --args.model_seed=42 --args.policy_name=vulkan-fix-30ep --args.model_ckpt_id=39999 --args.only_tasks=ButtonUnmask --args.max_episodes=30 --args.save_dir=v1-store/evaluation/vulkan-fix-30ep
```

- 与 `eval_shard.local.sh` 的差别只有两点：不分片（主线 `eval.py` 无 `episode_start`），环境变量直接写死而非 `${GLIBC_TUNABLES:-…}` 默认值；驱动库加载路径完全相同。
- `save_dir` 起跑前不存在，单次运行不续评。

## 判据

`VULKAN_FIX_30EP=PASS episodes_setup=<n≥30> progress_entries=<≥30> incompatible_driver=0 error_lines=0 eval_rc=0`。对照：未加环境变量时同一路径第 28 集必抛 `ErrorIncompatibleDriver`（`eval-official-framesamp-context`、`tic-vulkan-makeenv`）。成功率不作结论（单 seed、单任务）。

## tmux 会话清单

- `tic-vulkanfix`（内含 server 后台进程，由脚本 kill 自己起的 PID）。
