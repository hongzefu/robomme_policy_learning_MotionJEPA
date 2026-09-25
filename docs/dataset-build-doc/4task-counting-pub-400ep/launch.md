# 公开counting四任务400集正式库：起跑前记录

## 范围与启动条件

**尚未启动，不预填Beta提交或成功结果。** 用户最新原话：「恢复计划中的建库，严格按前置闸门推进」。本库为同一公开全集中的BinFill、PickXtimes、SwingXtimes、StopCube各100集，独立构建source、4×4 packed和自算norm_stats；不从完整库切分重编号，不生成8×8、Wan或motion。

必须先完成[16集冒烟](../16task-pub-smoke16-0925/launch.md)及[16任务正式库](../16task-pub-1600ep/launch.md)对应数据验收，并在完整库落盘后重新核算剩余预算。既有审查估算的full/count/smoke加10%为938.1027 GiB、临时等项目128 GiB、保留300 GiB，合计1366.1027 GiB；这是全流程估算，不是进入本阶段时仍可照抄的可用容量或剩余预算。实际以冒烟和完整库分配字节、当前可用字节及未完成阶段需求重算。

用户已明确counting是公开集合的4×100子集，并对其重算norm_stats；训练超参仍为原版。主机dtype许可只改变输入判定，数值要求精确、原始dtype/raw SHA保留，训练标量/状态bitwise不变。最新原话「允许这四键缺失与 None 等价」只覆盖`motion_emb/motion_pos/motion_mask/mem_order`的缺失与显式None，其他键和非None值不在许可内。本档案不启动训练，数据验收不能代替训练前置闸门。

## 版本、输入与输出

正式raw/SigLIP起跑前先完成本轮Beta提交，传完整40位`BUILD_HEAD`字面量且工作区clean。实际启动SHA、会话、UTC和退出码待起跑日志记录；不以事后提交或占位符声称已有Beta。源码按`git show <实际Beta完整SHA>:<路径>`还原，入口为`scan_manifest.cmd_build()`、`check_orig80k_sources.check_sources()`、`run_local.worker_cmd()`、finalize的hash/check、packed的pack/verify、`compute_norm_stats.main()`及`check_subset_eq.py`，不复制脚本或yaml到档案。

公开原始根为`/scratch/hongze/robomme_data_h5`；新硬链接目录为`/scratch/hongze/robomme_data_h5_counting4`。库根为`/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-counting-pub-400ep`，统计量根为`v1-store/train-assets/mme_vla_suite/4task-counting-pub-400ep`。三处新输出均必须不存在（含悬空链接）；核对同盘和实体父目录后才能创建。硬链接只指向四个原H5，共享同一设备/inode，不修改或删除原文件。

验收期望为400集、189035总帧和189035执行样本，来源是已提交全集manifest的四任务投影；counting无demo段。全局episode和样本编号允许按新库重新排列，物理身份`(h5_file, raw_ep_idx, step_idx)`及真实内容不能变化。

## 环境与阶段命令

所有预计超过5分钟的阶段在独立记录的detached tmux中执行，外围按[冒烟档案](../16task-pub-smoke16-0925/launch.md)写tee日志、任务/tee退出状态和`EXIT_CODE=`，失败即停。代码块是阶段命令体，不是绕过前置检查的一键入口。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
set -euo pipefail
source scripts/dataset/paths.sh
export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_CACHE_PATH="$V1_STORE/cache/cuda" PYTHONDONTWRITEBYTECODE=1
PUBLIC=/scratch/hongze/robomme_data_h5
RAW=/scratch/hongze/robomme_data_h5_counting4
LIB="$V1_STORE/datasets/4task-counting-pub-400ep"
FULL="$V1_STORE/datasets/16task-pub-1600ep"
MANI="$LIB/meta/episode_manifest.json"
INMANI="$LIB/meta/input_manifest.json"
STATS="$V1_STORE/train-assets/mme_vla_suite/4task-counting-pub-400ep"
REFERENCE="$REPO_ROOT/docs/dataset-build-doc/16task-h5-scan/records"
TASKS=BinFill,PickXtimes,SwingXtimes,StopCube
: "${BUILD_HEAD:?必须传已提交的完整Beta SHA字面量}"
test "$BUILD_HEAD" = "$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
test ! -e "$RAW" && test ! -L "$RAW"
test ! -e "$LIB" && test ! -L "$LIB"
test ! -e "$STATS" && test ! -L "$STATS"
test "$(stat -c %d "$PUBLIC")" = "$(stat -c %d "$(dirname "$RAW")")"
```

**硬链接与来源阶段：**创建后逐文件核对设备/inode，再重新hash这四个H5。`check_orig80k_sources.py`使用已提交16文件pin及全集manifest作参考，只按明确四任务名单投影，禁止以现场两个派生清单自证来源。

```bash
mkdir "$RAW"
for task in BinFill PickXtimes SwingXtimes StopCube; do
  name="record_dataset_$task.h5"
  ln -- "$PUBLIC/$name" "$RAW/"
  test "$(stat -c '%d:%i' "$PUBLIC/$name")" = "$(stat -c '%d:%i' "$RAW/$name")"
done
uv run --no-sync python scripts/dataset/scan_manifest.py build \
  --raw_dir "$RAW" --tasks "$TASKS" --episodes-per-task 100 --num_shards 1 --out "$MANI"
uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs \
  --raw_dir "$RAW" --out "$INMANI"
uv run --no-sync python scripts/dataset/check_orig80k_sources.py \
  --input-manifest "$INMANI" --reference-input "$REFERENCE/input_manifest.json" \
  --manifest "$MANI" --reference-manifest "$REFERENCE/episode_manifest.json" --tasks "$TASKS"
```

要求`INPUT_PIN=PASS files=4`、`EPISODE_IDENTITY=PASS episodes=400 timesteps=189035 samples=189035`及阶段退出0，实际清单SHA和`SOURCE_PROVENANCE`留档；这里仍是预定判据。

**构建、验证和独立统计量：**与完整库串行执行，SigLIP用GPU0–7、finalize用GPU7；CPU阶段显式隔离GPU。

```bash
uv run --no-sync python scripts/dataset/run_local.py --stage siglip \
  --lib "$LIB" --gpus 0,1,2,3,4,5,6,7 --raw-dir "$RAW" --require-free-mib 70000
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/dataset/finalize_checks.py check \
  --manifest "$MANI" --out "$LIB/source" --raw_dir "$RAW" \
  --input_manifest "$INMANI" --input_level sha256 --spot_check 1024
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack \
  --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp" --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify \
  --store "$LIB/framesamp" --resume --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/compute_norm_stats.py \
  --output-dir "$STATS" --config-name mme_vla_suite --repo-id robomme --dataset-path "$LIB/source"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/check_subset_eq.py \
  --full "$FULL" --subset "$LIB"
```

统计量实际输出为`$STATS/robomme/norm_stats.json`，后续训练的`assets-dir`传`$STATS`，不重复添加robomme。必须按物理episode及局部step全覆盖比较两个库对应的features和全部pkl键，包括两幅图像、完整actions、state、prompt等；只有`epis_idx`按manifest显式映射，不豁免其他差异。主机训练输入的dtype与四motion None许可，不放宽建库子集内容的dtype/字节守卫。

## 会话清单、验收与归档

当前实际创建会话清单为空；使用`cnt-<阶段>-<UTC>`模板，起跑时填完整实例名，不把模板当实际会话。日志为`v1-store/logs/<完整会话名>.<阶段>.log`，worker日志在`$LIB/logs/`。保存命令、HEAD、clean状态、精确PID、起止UTC及任务/tee退出码；仅对本轮清单用`tmux has-session -t '=完整名称'`检查，必要清理逐个完整匹配且前后核对，不执行任何全局kill。

| 阶段 | 实际完整tmux名称 | 当前状态 |
|---|---|---|
| 硬链接、scan/hash/pin绑定 | 待起跑填写 | 未启动 |
| SigLIP | 待起跑填写 | 未启动 |
| finalize | 待起跑填写 | 未启动 |
| pack/verify | 待起跑填写 | 未启动 |
| 独立norm_stats | 待起跑填写 | 未执行 |
| 与全集的全覆盖子集比较 | 待起跑填写 | 未执行 |

数据阶段要求所有阶段成功、packed为`framesamp-4x4-v1`且全量verified、规模与pin绑定，子集比较满足400集/189035样本和零失配。任一失败保留现场并报告，不自行放宽或删除输出。归档实际清单/摘要、norm_stats、守卫判定、清洗日志、资源与字节量；不归档H5、features、packed、权重或脚本/yaml副本。结果按[十三节README](README.md)回填，训练可读性及后续对拍另列，未执行不得记为通过。
