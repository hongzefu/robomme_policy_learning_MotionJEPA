# 本仓库策略评估

主入口为 `scripts/evaluation/run.sh 运行名 checkpoint绝对路径 任务 episode上限 [策略seed=7]`。本轮仅验收指定 bucket 末步 59999。旧入口 scripts/training/eval.sh 直接转发相同参数。

环境准备：`bash scripts/evaluation/setup.sh`。权重准备：`bash scripts/evaluation/download.sh`。两套环境都使用已提交的锁文件执行 frozen 安装。准备脚本将本仓库已有的 `v1-store/models/big_vision/paligemma_tokenizer.model` 复制进独立缓存并核验 SHA256；该源文件必须存在。服务端和客户端依赖隔离，全部包和评估产物位于当前 MotionJEPA 工作副本，解释器使用已固定的 NFS Python；训练 .venv 不变。

单回合从 clean HEAD 启动：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/evaluation/run.sh 新运行名 "$PWD/v1-store/models/robomme-vla-modul-60k-v1/59999" RouteStick 1 7
```

实际运行必须在 detached tmux 内，使用 PYTHONUNBUFFERED=1、pipefail、tee 和 EXIT_CODE。集群复用已分配资源，srun 显式加 --gpu_cmode=shared，工作目录指向本仓库。不可自动申请新作业，也不退回兼容渲染。

本机起跑及模型服务启动前都会检查所选 GPU 上的计算进程，占用即停止。用户已明确要求等待空闲 GPU，不与已有任务同卡运行。

测试：在已安装 uv 环境中执行 `scripts/evaluation/test_control.py` 和 `test_result.py`。来源检查额外核对官方 benchmark 锁定 SHA、editable 路径、ManiSkill fork 及 RouteStick 元数据。模型检查使用当前源码形状与 Orbax 元数据，不跳过多余参数。

产物位于 v1-store/evaluation/<运行名>。EVAL_PASS 表示指定回合正常完成且视频完整可解码，任务成功由 task_successes 单独报告。error 不能作为正常失败通过验收。不得覆盖运行名或换 seed 挑选成功结果。
