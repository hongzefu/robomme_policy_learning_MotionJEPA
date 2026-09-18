# 本仓库双 uv 单回合评估计划

## 第一部分：目标与边界

用户批准在本 NFS MotionJEPA 工作副本，从 f6915f2 创建 v2-vail-eval-0917；本轮所有主仓库改动和留档均提交到该分支。两端只调用本仓库代码，不使用外部评估仓库。

benchmark 保持官方 RoboMME/robomme_benchmark 来源、锁定 856bc3a，不改源码、不建分支。仅确需修改时才使用 hongzefu/robomme_benchmark_MotionJEPA，并创建 PolicyEvalThirdParty-<主仓库任务分支>；此机制已写入 AGENTS.md。

权重为 bucket HongzeFu/robomme-vla-modul-60k-v1 的 59999，8 帧 × 64 token、budget 512、modulation、motion 关闭。两端各运行 RouteStick/test/episode 0，策略 seed 7；元数据环境 seed 660000、easy。本机成功执行后再使用既有 GreatLakes 作业 61331555。此前其他目录的结果不算本轮验收。

## 第二部分：实现与验收

服务端使用根项目与原锁文件，客户端使用 scripts/evaluation/client 的独立项目和锁文件。环境分别位于 v1-store/envs/policy-eval-server 与 policy-eval-client，使用同一 NFS Python 3.11.14；不覆盖训练 .venv。ManiSkill fork 固定 07be6fbc66350ddca200abfb0a11b692f078f7fd。所有运行通过 uv run --frozen --no-sync，缓存和产物进入本仓库 v1-store。

scripts/evaluation/run.sh 依次核对源码、渲染、参数树和归一化，启动 scripts/training/serve_policy.py，再运行 examples/robomme/eval.py。使用既有 max_episodes=1；修复 evaluate 的建环境和执行异常收尾、错误成功率汇总及无限循环。模型与训练链路不修改。

下载仅包含 59999 与根目录来源文件，逐项验证 SHA256。保留 create_trained_policy 的冻结配置、来源验证及严格恢复。独立结果检查器拒绝 error、额外回合和无法完整解码的视频。

静态测试覆盖 shell、无效 checkpoint、单回合、异常、恢复及结果验收；两端从同一个 clean HEAD 运行，新运行目录拒绝覆盖。GreatLakes 使用既有 61331555，shared、一 GPU、两 CPU、30 分钟上限。detached tmux 使用 pipefail、tee、EXIT_CODE，收尾仅回收自己的服务进程，不释放交互作业。

结果留档位于 docs/training-doc/<run_name>，记录命令、提交、环境和模型指纹、日志、结果和视频路径。分别报告链路与任务结果，不为成功率重试，不要求两端帧数相同。提交使用中文，逐文件暂存，提交后立即推送同名分支。
