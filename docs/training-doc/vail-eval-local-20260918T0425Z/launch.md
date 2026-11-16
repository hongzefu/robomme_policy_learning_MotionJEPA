# 本机 RouteStick 单回合启动留档

本轮仅使用 /nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA，分支 v2-vail-eval-0917。从提交本留档之后的同一个 clean HEAD 起跑，实际 SHA 由运行器输出。benchmark 官方来源、锁定 856bc3a189d4172f3f47dbee4424d585f8d78db3，未改源码。

模型为 bucket HongzeFu/robomme-vla-modul-60k-v1 的 59999。下载后核验 28 项 SHA256；冻结配置 sha256 为 804b25382af668d67c8f8ea2d1cca414aee9a184ec737dcad704bdff253a2f92，norm_stats 为 856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173。冻结配置为 8×64 token、budget 512、modulation、motion 关闭，使用本仓库现有严格恢复实现。bucket README 任务列表与训练留档冲突，训练留档明确包含 RouteStick，保留差异不修改远端。

测试条件：RouteStick/test/episode 0，策略 seed 7，环境 seed 660000、easy，max_steps 1300、obs_horizon 16。sled-vail 空闲 GPU 1。不重试挑成功，不复用任何旧评估结果。

两个独立 uv 环境在 v1-store/envs 下，均使用 NFS Python 3.11.14。启动核对实际 import 路径和锁定版本，渲染探针、参数树、服务健康、结果及完整视频全部通过才认定链路通过。

## 命令与会话

会话名为 `vail-eval-local-20260918T0425Z`，在 detached tmux 中执行以下命令。准备会话另有 mv-eval-uv-0917、mv-eval-ckpt-0917；只记录和清理自己的进程，不释放用户交互分配。

```bash
export REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
cd "$REPO"
unset ROBOMME_GPU_RASTER SAPIEN_DISABLE_RAY_TRACING
export PYTHONUNBUFFERED=1
set -o pipefail
CUDA_VISIBLE_DEVICES=1 bash scripts/evaluation/run.sh vail-eval-local-20260918T0425Z "$REPO/v1-store/models/robomme-vla-modul-60k-v1/59999" RouteStick 1 7 2>&1 | tee "v1-store/logs/policy-eval/vail-eval-local-20260918T0425Z.log"
echo "EXIT_CODE=$?"
```

结果与完整视频位于 v1-store/evaluation/vail-eval-local-20260918T0425Z/perceptual-framesamp-modul-8frame-8x8/ckpt59999/seed7/。留档保存轻量原始日志和 JSON，不提交权重与视频。收官分别报告任务结果和链路验收。

本次为资源释放后的新启动；首次 0400Z 启动在渲染探针阶段因 GPU 被其他任务占用而主动停止，没有策略回合。用户要求等待空闲，本次起跑前与模型服务启动前均检查 GPU 计算进程为空。
