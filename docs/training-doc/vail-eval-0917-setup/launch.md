# 本仓库双 uv 环境与权重准备

用户批准从 f6915f2 创建 v2-vail-eval-0917，在当前 NFS MotionJEPA 工作副本实现并执行两端评估。benchmark 使用官方来源及 856bc3a，不改源码、不建立分支。

准备入口为 `scripts/evaluation/setup.sh` 与 `download.sh`，均在本仓库根目录以 detached tmux 运行，使用 PYTHONUNBUFFERED=1、pipefail、tee 并记录 EXIT_CODE。会话清单：`mv-eval-uv-0917`、`mv-eval-ckpt-0917`。日志位于 `v1-store/logs/policy-eval/setup.log`、`download.log`。启动前提交本留档；客户端首次解析生成的 uv.lock 在真实评估前提交。

两个环境使用 NFS Python 3.11.14，分别位于 `v1-store/envs/policy-eval-server` 与 `policy-eval-client`；训练 `.venv` 不变。只下载 bucket 末步 59999 与根目录来源文件至 `v1-store/models/robomme-vla-modul-60k-v1`，验证源 SHA256 清单。准备阶段不运行策略回合。
