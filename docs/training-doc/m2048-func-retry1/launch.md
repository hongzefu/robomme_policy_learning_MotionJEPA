# 2048功能检查的正式重测

用户要求「能并行的尽可能并行 8个gpu你可以用」，同时保留「保持原计划，完整取证（推荐）」。主副本的旧档与新参考训练继续冻结在CAND `0c877c7495dfe5db8b83f033442013c6d6fd8552`。现有 `-temp` 有其他代理进程，本次未改变其HEAD、源码或环境；另建本轮独立副本 `v1-store/workspaces/m2048-tool-fix`，从相同CAND克隆，使用独立 `.venv`。`uv sync --offline --frozen` 只在新副本中安装208包；主副本环境与依赖文件不变。所有新增产物仍在主仓库的 `v1-store/` 范围内。

原功能检查 `m2048-func` 在GPU2的三次A/A全部38叶通过后失败：JIT闭包捕获权重，CompilationResultProto约10GB，超出2GB序列化上限，随后新增72MiB常量分配失败。探索性 `m2048-func-probe1` 把权重改成显式参数后247秒全通过，但没有替代正式验收。本次只将同一修复落入检查器的 `input_function` 与 `act`，生产 `src/`、模型、输入、判据、容差和叶集合均不改。

新副本只复制功能检查需要的pi05_base权重，使用 `cp -a --reflink=auto`，没有复制数据集或创建存储外链；运行前按既有资产锁全量SHA核验。验证仍使用已定gemma_150m替身、完整gemma_300m动作专家及兼容预训练参数，seed42。该模型用于功能门，不代表生产测速模型。

正式运行从本次工具修复提交后的clean HEAD开始；完整提交、Python环境与导入路径在实际日志中记录。会话全名 `m2048-func-retry1`，物理GPU2，确定性XLA设置沿用。检查覆盖三次全部梯度A/A、2048个image和pos位置各自非零、每帧删63位置的反例、32帧带扰动、n=1/8/31短历史mask下loss/动作及全部38梯度叶相等、无效位置梯度零与有效位置负对照。失败不放宽任何判据。

命令体如下。外层detached tmux使用pipefail、tee和EXIT_CODE，输出根全新；这是启动档案，运行结果另写result.md。

```bash
set -euo pipefail
cd /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/workspaces/m2048-tool-fix
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/uv
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export CUDA_CACHE_PATH="$V1_STORE/cache/cuda"
export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/m2048-func-retry1"
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
unset JAX_PLATFORMS MMEVLA_MOTION_STORE
test -z "$(git status --porcelain)"
REC=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/m2048/m2048-func-retry1
test ! -e "$REC"
mkdir "$REC"
printf 'START_HEAD=%s\nSTART_UTC=%s\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
uv run --no-sync python scripts/training/tests/check_32frame_modul.py func --out "$REC/func.json"
printf 'CHECKS_DONE run=m2048-func-retry1 utc=%s\n' "$(date -u +%FT%TZ)"
```

用户另已授权：「1000步测速后如果占用率高于50% 并且没有其他的问题 可以直接启动训练」。本次功能重测只是前置门之一；全部正确性和容量检查通过、测速报告完整且800步稳态窗口八卡总体平均util严格大于50%、无未解决问题时，才执行最终起跑门并启动已定正式run。不会把探索性结果或单项通过代替全部条件。
