# m2048-func 启动口径

用户授权实现至测速报告，并明确「保持原计划，完整取证（推荐）」。本档从同一clean CAND运行；实际完整HEAD、UTC起点由下列命令现场输出。依赖不变，源码导入主副本。环境B，A100-SXM4-80GB，底层/dev/md0 XFS本地NVMe RAID。该项不是正式训练，不创建额外数据集。

仅GPU4，modulation替身沿既有gemma_150m/300m与pi05_base兼容加载口径；至少三次A/A全叶探针，不排除叶。逐token image/pos梯度2048位置均非零，删除每帧63位置故障须拒绝；32帧带扰动、合成1/8/31帧mask的loss/固定noise动作/全部可训练梯度不变及有效位负例均须通过。static_state_emb因use_state_emb=false不参与输入梯度判据，其交付仍在输入检查覆盖。

数据仅为1600ep，清单file SHA=df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482，norm_stats file SHA=856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173。原始记录落v1-store/bench/m2048/m2048-func，完成后归档可复核结果和清洗日志；不把预期PASS当实测。

唯一会话m2048-func，detached tmux，外层pipefail+tee写v1-store/logs/m2048-func.log，结束记录EXIT_CODE。CPU组合检查保守按超过5分钟留档；逐项执行失败即停。独立oracle保持atol1e-6/rtol1e-5，numpy求幂先高精度再舍入f32，避免已定位的numpy f32幂1 ULP误差，不改被测模型或验收容差。

```bash
set -euo pipefail
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/training/paths.sh
export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1 CUDA_CACHE_PATH="$V1_STORE/cache/cuda"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DS="$V1_STORE/datasets/4task-v2-1600ep-604f16da"
export MMEVLA_FRAMESAMP_SOURCE="$DS/source" MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"
export DTYPE_MANIFEST="$MMEVLA_FRAMESAMP_MANIFEST"
unset MMEVLA_MOTION_STORE MMEVLA_FRAMESAMP_ALLOW_SUBSET
test -z "$(git status --porcelain)"
printf 'START_HEAD=%s\nSTART_UTC=%s\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
export CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
unset JAX_PLATFORMS
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
REC="$V1_STORE/bench/m2048/m2048-func"
test ! -e "$REC"; mkdir "$REC"
export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/m2048-func"
uv run --no-sync python scripts/training/tests/check_32frame_modul.py func --out "$REC/func.json"
printf 'CHECKS_DONE run=m2048-func utc=%s\n' "$(date -u +%FT%TZ)"
```
