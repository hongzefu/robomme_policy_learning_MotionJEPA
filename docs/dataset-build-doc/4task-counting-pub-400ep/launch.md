# 公开counting四任务400集正式库：启动实录

## 范围、起跑前约定与实际状态

**本库数据阶段、full/counting严格子集与独立count20均已PASS。** 本文件由起跑前记录回填实际会话、环境和退出状态。用户原话：「纯counting跑4*100 公开集合的子集合」及「恢复计划中的建库，严格按前置闸门推进」。范围为同一公开全集中的BinFill、PickXtimes、SwingXtimes、StopCube各100集，独立构建source、4×4 packed和本库norm_stats，不从full库切分改号，也不生成8×8、Wan或motion。

起跑前约定先完成[16集冒烟](../16task-pub-smoke16-0925/result.md)及[full数据验收](../16task-pub-1600ep/README.md)，再按实际剩余空间、最大feature/pkl分配量和未建内容重算预算。原历史初估1366.1027 GiB覆盖full/count/smoke及余量，未照抄为counting剩余门槛。实际在full成功后只读stat其已完成768897个feature、476857个pkl，F/P最大分配量606208/397312 B，与冒烟下限比较取大；counting数据基数202175603728 B，加10%、128 GiB缓存/临时/日志和300 GiB保留量，所需681954664773 B。2026-09-26 00:50:31 UTC可用1065860091904 B，预算通过，见[正式预算记录](records/post_full_counting_budget.json)。

用户允许主机dtype不同但数值精确一致、训练标量/状态逐位一致，另允许 `motion_emb/motion_pos/motion_mask/mem_order` 仅缺失与严格None等价。此许可不放宽建库子集的dtype/raw字节判据；原始dtype/raw SHA、其他键、非None及signed zero要求保留。

## 版本、输入输出与日志保护

实际启动及结束HEAD均为 **`49a333eb18e8d6ff1143bf7871ef7c498ab91579`**，提交主题 `commitV11.11Beta: 锚定两个公开正式库及20步可读性检查`，与full和两次20步一致。来源与16集冒烟结果提交 `de6354fd7c0f78b347c7b07c4b8ee0c8b9b37418` 只作已完成前置，不当作本次启动版本。生产代码与依赖未为本轮改算法或参数。

公开输入 `/scratch/hongze/robomme_data_h5`，新硬链接根 `/scratch/hongze/robomme_data_h5_counting4`；库根 `v1-store/datasets/4task-counting-pub-400ep`、统计根 `v1-store/train-assets/mme_vla_suite/4task-counting-pub-400ep`，均在本仓库内。起跑前拒绝这三个新根及预算记录已存在或为悬空链接，确认父目录为实体且同设备。四个H5逐一硬链接，逐一核对device/inode；实际共同device为2304，四文件size合计126024202016 B，没有复制第二份H5。准确inode见[README原始来源](README.md#4-原始来源与输入pin)。

预检要求full最终唯一 `EXIT_CODE=0`、十个阶段按顺序退出0、唯一 `FULL_DATA_BUILD=PASS`与报告完全相同，以及manifest/packed/norm绑定一致。检查通过后才构造硬链接、scan和建库。counting实际400集、189035帧和189035执行样本，无demo段；跨库物理身份为 `(h5_file, raw_ep_idx, step_idx)`，仅 `epis_idx`按两清单映射。

起跑前模板按阶段说明日志纪律；实际是一份detached tmux串行十二阶段、共享构建日志，worker各有独立日志，严格子集另有独立tee日志。外层pipefail、body严格模式、每阶段退出后显式失败返回，任一失败停止下游；任务/tee分别记录退出码，日志noclobber拒绝复用。结束重核HEAD/clean，整个运行过程保持同一Beta。

还原起跑前约定与默认：

```bash
git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:docs/dataset-build-doc/4task-counting-pub-400ep/launch.md
git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:scripts/dataset/paths.sh
```

## 实际环境与阶段命令

实际会话为 `cnt-pub400-20260926T004948Z`，pane `%540`，包装PID **3924000**、body PID **3924005**。会话在同一shell中按 `run_stage()` 串行执行十二阶段，并记录每阶段UTC与退出码。运行时保存的命令正文为 `v1-store/bench/orig80k-build-preflight-0925/cnt-pub400-20260926T004948Z.command.txt`，SHA256为 `35c0af03c1a8f58e3022eece8b0231521c97dcc863b6e3fab2c2d3a9c7fb6f2f`。该文本用于还原实际调用，不作为独立bash副本复制到正式docs。

起跑前launch的CUDA缓存示例是 `$V1_STORE/cache/cuda`，实跑明确覆盖为 `$V1_STORE/cache/cuda/orig80k-build`；它不是paths.sh提供的默认值。`RAW_H5_DIR`则在source前覆盖为四文件硬链接根，不使用paths.sh的AWS公开全集默认输入。

| 来源 | 环境项 | 实际值或范围 |
|---|---|---|
| 手动export，source前 | `RAW_H5_DIR` | `/scratch/hongze/robomme_data_h5_counting4` |
| 手动export | `BUILD_HEAD` | `49a333eb18e8d6ff1143bf7871ef7c498ab91579` |
| 手动export | `UV_CACHE_DIR` | `$V1_STORE/cache/uv` |
| 手动export | `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS` | 均为`1` |
| 手动export | `PYTHONUNBUFFERED`、`PYTHONDONTWRITEBYTECODE` | 均为`1` |
| 手动export | `CUDA_CACHE_PATH` | `$V1_STORE/cache/cuda/orig80k-build` |
| 手动unset | `XLA_FLAGS`、`JAX_PLATFORMS`、`JAX_PLATFORM_NAME` | 清除继承的验证/CPU平台设置 |
| CPU阶段局部覆盖 | `CUDA_VISIBLE_DEVICES=''`、`JAX_PLATFORMS=cpu` | preflight、scan、hash、pin、pack、verify、norm_stats、subset_eq、report |
| GPU阶段局部覆盖 | `CUDA_VISIBLE_DEVICES` | SigLIP为0–7，worker各单卡；finalize为7。两阶段额外用`env -u`清除上述三个变量 |
| paths.sh固定Beta默认 | `OPENPI_DATA_HOME`、`XDG_CACHE_HOME`、`HF_HOME` | 分别为`$V1_STORE/models`、`$V1_STORE/cache/xdg`、`$V1_STORE/cache/hf` |
| paths.sh固定Beta默认 | `JAX_COMPILATION_CACHE_DIR`、`HF_HUB_OFFLINE`、`UV_LINK_MODE` | 分别为`$V1_STORE/cache/jax`、`1`、`copy` |

固定提交默认与手动覆盖明确区分；`run_local.base_env()` 的同值worker设置按该Beta源码还原。以下为实际命令正文中的实质调用，外层还有前置、新输出、预算、tmux、tee与退出状态保护。现有目录已经创建，下列文本仅供还原，不得重放覆盖。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
set -euo pipefail
export RAW_H5_DIR=/scratch/hongze/robomme_data_h5_counting4
source scripts/dataset/paths.sh
export BUILD_HEAD=49a333eb18e8d6ff1143bf7871ef7c498ab91579
export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_CACHE_PATH="$V1_STORE/cache/cuda/orig80k-build" PYTHONDONTWRITEBYTECODE=1
unset XLA_FLAGS JAX_PLATFORMS JAX_PLATFORM_NAME
PUBLIC=/scratch/hongze/robomme_data_h5
RAW=/scratch/hongze/robomme_data_h5_counting4
LIB="$V1_STORE/datasets/4task-counting-pub-400ep"
FULL="$V1_STORE/datasets/16task-pub-1600ep"
MANI="$LIB/meta/episode_manifest.json"
INMANI="$LIB/meta/input_manifest.json"
STATS="$V1_STORE/train-assets/mme_vla_suite/4task-counting-pub-400ep"
REFERENCE="$REPO_ROOT/docs/dataset-build-doc/16task-h5-scan/records"
TASKS=BinFill,PickXtimes,SwingXtimes,StopCube

mkdir "$RAW"
for task in BinFill PickXtimes SwingXtimes StopCube; do
  name="record_dataset_$task.h5"
  ln -- "$PUBLIC/$name" "$RAW/"
  test "$(stat -c '%d:%i' "$PUBLIC/$name")" = "$(stat -c '%d:%i' "$RAW/$name")"
done
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/scan_manifest.py build --raw_dir "$RAW" --tasks "$TASKS" --episodes-per-task 100 --num_shards 1 --out "$MANI"
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs --raw_dir "$RAW" --out "$INMANI"
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/check_orig80k_sources.py --input-manifest "$INMANI" --reference-input "$REFERENCE/input_manifest.json" --manifest "$MANI" --reference-manifest "$REFERENCE/episode_manifest.json" --tasks "$TASKS"
env -u JAX_PLATFORMS -u JAX_PLATFORM_NAME -u XLA_FLAGS CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --no-sync python scripts/dataset/run_local.py --stage siglip --lib "$LIB" --gpus 0,1,2,3,4,5,6,7 --raw-dir "$RAW" --require-free-mib 70000
env -u JAX_PLATFORMS -u JAX_PLATFORM_NAME -u XLA_FLAGS CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/dataset/finalize_checks.py check --manifest "$MANI" --out "$LIB/source" --raw_dir "$RAW" --input_manifest "$INMANI" --input_level sha256 --spot_check 1024
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp" --procs 48
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --store "$LIB/framesamp" --resume --procs 48
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/compute_norm_stats.py --output-dir "$STATS" --config-name mme_vla_suite --repo-id robomme --dataset-path "$LIB/source"
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/check_subset_eq.py --full "$FULL" --subset "$LIB"
```

preflight和最终report是实际命令正文中的CPU内联Python，分别完成前置/预算与结果/体积验收；没有将它们伪称为Beta中的独立脚本。前者在full完成后扫描已完成文件的stat取得F/P最大分配量，后者检查norm的shape/有限性、源/packed计数、严格子集记录绑定与结束空间后写出 `build_sizes.json`。`strict_subset()` 另用tee保留原始 `meta/subset_eq.log`，分别记录工具和tee退出码。

代码恢复使用完整Beta与语义入口，不依赖易漂移行号：

```bash
git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:scripts/dataset/scan_manifest.py
git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:scripts/dataset/check_orig80k_sources.py
git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:scripts/dataset/run_local.py
git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:scripts/dataset/finalize_checks.py
git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:scripts/dataset/pack_framesamp_store.py
git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:scripts/dataset/check_subset_eq.py
git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:scripts/training/compute_norm_stats.py
```


## 实际会话、阶段退出与交接

| 项目 | 实际记录 |
|---|---|
| 构建会话/pane | `cnt-pub400-20260926T004948Z` / `%540` |
| 包装/body PID | `3924000` / `3924005` |
| UTC起止 | 2026-09-26 00:50:24 → 01:44:56 |
| 总跨度 | 3272秒，54分32秒 |
| 原始日志 | `v1-store/logs/cnt-pub400-20260926T004948Z.build.log` |
| 任务/tee/整体终态 | `TASK_EXIT=0`、`TEE_EXIT=0`、唯一外层 `EXIT_CODE=0` |
| 子集工具/tee | `SUBSET_TASK_EXIT=0`、`SUBSET_TEE_EXIT=0` |

| 阶段 | UTC开始→结束，2026-09-26 | 秒级跨度 | 退出 |
|---|---|---:|---:|
| preflight | 00:50:24→00:50:31 | 7 | 0 |
| hardlinks | 00:50:31→00:50:31 | 0 | 0 |
| scan400 | 00:50:31→00:50:35 | 4 | 0 |
| hash4 | 00:50:35→00:55:42 | 307 | 0 |
| source_pin | 00:55:42→00:55:42 | 0 | 0 |
| siglip | 00:55:42→01:36:40 | 2458 | 0 |
| finalize | 01:36:40→01:40:55 | 255 | 0 |
| pack | 01:40:55→01:41:04 | 9 | 0 |
| verify | 01:41:04→01:41:07 | 3 | 0 |
| norm_stats | 01:41:07→01:41:37 | 30 | 0 |
| subset_eq | 01:41:37→01:44:49 | 192 | 0 |
| report | 01:44:49→01:44:56 | 7 | 0 |


SigLIP工具报告2457秒，表中秒级起止跨度2458秒，分别保留口径。来源pin四文件、episode400集/189035帧/189035样本均通过；finalize1024条最大绝对差0；packed全量verify189035行、失配0。`SUBSET_EQ=PASS episodes=400 samples=189035 feature_mismatch=0 sample_mismatch=0`与[两库规范manifest和日志SHA](records/build_sizes.json)绑定。`COUNTING_DATA_BUILD=PASS`包含source＋packed实测分配量202152935424 B和全部结构/有限性验收；没有只看成功字符串跳过内容检查。

独立norm实际纳入1476个batch128，共188928样本，尾107个不足完整batch的样本未纳入统计；全部189035训练样本保留。norm SHA为 `a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9`；counting的训练 `assets-dir`为 `v1-store/train-assets/mme_vla_suite/4task-counting-pub-400ep`，asset-id为robomme，不重复添加robomme。full自算与原版norm均未改写。

count20另在 `orig80k-count-read20-20260926T014535Z`（包装PID3957023）使用GPU4–7、本库norm及相同Beta，2026-09-26 01:45:54→01:51:37 UTC整体PASS；20次更新、末步19、61个EMA叶实际恢复一致，见[独立训练档案](../../training-doc/smoke-orig80k-count-0925/launch.md)。其结果独立于建库日志验收。

记录已归档于[十三节README](README.md)及[正式核验清单](records/archive_checks.json)，共28文件、530983 B。首次过长内联tmux命令未启动成功，确认无会话/日志后改为执行相同已核SHA命令文件；实质参数未改，见README计划外事件。本次没有删除原件、硬链接或库；仅对本轮明确清单的完整tmux名称精确检查，不全局kill。两库正式INPUT、P1、100步对拍、300步perf及80k均尚未通过，不能用本次20步代替。
