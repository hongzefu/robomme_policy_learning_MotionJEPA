# 新版四任务 28 集冒烟启动记录

## 范围与判据

用户要求“开始实现该计划 有问题提前问用户 越早越好”，并明确“继续工作 忽略huggingface的任务 但是要注意git”。沿用根目录 `v2-4task-h5-merge-plan.md`，先以 `--per-group 2` 选取 14 个任务难度组各两条，共 28 条，跑通物理合并、full 验真、SigLIP、finalize、4×4 与 8×8 打包及跨网格校验。必须取得 `PLAN_OK`、`MERGE_VERIFY=PASS`、`STAGE_DONE stage=siglip`、`FINALIZE_EXIT_CODE=0`、两档 `VERIFY_PACK=PASS` 和跨网格判定。此冒烟预计超过五分钟，因此保留本档案和清洗日志；通过后仅清理本轮新建的两个数据目录。

## 版本与环境

实现提交为 `a0cdfe751fcd5bd4ce9fc7e4d5037dee03a4246e`（`commitV9.4`）。HF 工作由原作者同期提交并占用 V9.3，本轮接续 V9.4，未修改或暂存其内容。实际每阶段启动 HEAD 和工作区状态由运行日志记录；本档案提交后启动。SigLIP 起跑到 finalize 通过期间本轮不提交。

环境 B：AWS，8×A100-SXM4-80GB，`/scratch` 位于 `/dev/md0` XFS 本地 NVMe RAID。本轮 GPU 仅使用 4–7。GPU 0–3 的 MotionJEPA 正式训练保持运行，参考吞吐 874.0 samples/s；观察到新增 epoch 吞吐低于 865.26 时暂停本轮活动阶段。该比较仅用于资源共处，不判读模型效果。

SigLIP、PaliGemma tokenizer、pi05_base 三项本地权重的 `cheap` 内容检查均通过。代码测试 74 项通过（9.93 秒）；真实 RouteStick easy episode_1 的 200 步前检、源摘要、对象拷贝及 full 数值/结构检查通过（3.022 秒），临时数据已清理。

## 来源与路径

源为 `/scratch/hongze/robomme-4task-h5-20260912-v2`，仓库 `HongzeFu/robomme-4task-h5-20260912-v2`，revision `604f16da36d6b6d175884df8fb687dc08e0a36eb`，manifest sha256 `df992cdf5a768ae0f368e520b0d7be28a68ed4ce6ca6201967cf5d6654cf2bb9`。这是私有 sim 录制集，不是公开 RoboMME 数据的重新打包。这里只验证建库，不执行现有官方 sim 评估。

合并根 `v1-store/raw-h5/4task-20260912-v2-smoke`、派生库 `v1-store/datasets/4task-v2-smoke28` 均为本轮全新目录。源 `extracted/` 保留。BinFill/RouteStick/VideoRepick/VideoUnmaskSwap 分别选中 6/8/6/8 条。生产完整源检查仍要求每任务 400 条，总计 1600 条，不以 smoke 缩小源全集完整性要求。

## 启动命令

tmux 会话：`v2b-smoke-20260914T174147Z`。总日志：`v1-store/logs/v2b-smoke-20260914T174147Z.log`。外层使用 `set -o pipefail`、`PYTHONUNBUFFERED=1`、`tee`，无论命令成功或失败均写 `EXIT_CODE=`。每段命令前打印 `V2_STAGE=` 与 UTC 时间。

```bash
source scripts/dataset/paths.sh
export UV_CACHE_DIR="$V1_STORE/cache/uv"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SRC=/scratch/hongze/robomme-4task-h5-20260912-v2
RAW=$V1_STORE/raw-h5/4task-20260912-v2-smoke
LIB=$V1_STORE/datasets/4task-v2-smoke28
TASKS=BinFill,RouteStick,VideoUnmaskSwap,VideoRepick
MANI=$LIB/meta/episode_manifest.json
INMANI=$LIB/meta/input_manifest.json
uv run --no-sync python scripts/dataset/merge_v2_h5.py plan --snapshot "$SRC/snapshot" --control "$SRC/control" --extracted "$SRC/extracted" --out "$RAW" --tasks "$TASKS" --per-group 2 --procs 48
uv run --no-sync python scripts/dataset/merge_v2_h5.py merge --out "$RAW" --extracted "$SRC/extracted" --procs 4 --check-source-sha256
uv run --no-sync python scripts/dataset/merge_v2_h5.py verify --out "$RAW" --extracted "$SRC/extracted" --level full --procs 48
uv run --no-sync python scripts/dataset/scan_manifest.py build --raw_dir "$RAW" --tasks "$TASKS" --num_shards 1 --out "$MANI"
uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs --raw_dir "$RAW" --out "$INMANI"
uv run --no-sync python scripts/dataset/run_local.py --stage siglip --lib "$LIB" --gpus 4,5,6,7 --raw-dir "$RAW"
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/dataset/finalize_checks.py check --manifest "$MANI" --out "$LIB/source" --raw_dir "$RAW" --input_manifest "$INMANI" --input_level sha256 --spot_check 1024
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp" --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --store "$LIB/framesamp" --resume --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --layout framesamp-8x8-v1 --reader decode --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp-8x8" --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --store "$LIB/framesamp-8x8" --resume --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py report --store "$LIB/framesamp-8x8"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/xgrid_pos_check.py --store-4x4 "$LIB/framesamp" --store-8x8 "$LIB/framesamp-8x8" --source "$LIB/source" --image-spot 512
```

## 已发现的问题与待验结果

真实 MANIFEST 的 `counts` 包含 `primary/spare/smoke/videos/archives/files` 六项；实现只将 episode role 分布与前三项逐项比较，后三项不是 episode role。此修正已纳入回归。阶段 7 的四文件并行 sha256 由用户明确批准，函数是 `finalize_checks.check_inputs`。本档案记录启动计划，最终结果与实际耗时另写 `result.md`。
