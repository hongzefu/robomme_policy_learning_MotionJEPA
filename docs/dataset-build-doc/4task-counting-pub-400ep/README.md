# 公开counting四任务400集正式库：十三节构建档案

> 本档案在构建与两次20步检查结束后回填实际结果。起跑前范围、参数和闸门由完整Beta `49a333eb18e8d6ff1143bf7871ef7c498ab91579` 锚定；实际启动与退出记录见 [launch](launch.md)，小型实测记录及清洗日志已归档于 [records核验清单](records/archive_checks.json)。

## 1. 结论与指标速览

**counting400数据阶段、与full库的严格子集内容对拍、count20真实保存与恢复均已独立通过。** 实际会话从 2026-09-26 00:50:24 UTC 运行至01:44:56 UTC，共3272秒（54分32秒）。十二个阶段全部退出0，任务、内层子集tee和外层tee均成功。

| 指标 | counting本次实测 |
|---|---:|
| 任务与episode | BinFill、PickXtimes、StopCube、SwingXtimes，各100集，共400集 |
| 总帧与执行样本 | 均为189035；全部episode的exec_start_idx为0 |
| 原始硬链接 | 4个，与各自公开原件同设备/inode |
| SigLIP | 8个worker、400集，工具记录2457秒 |
| finalize复算 | 1024/1024条，最大绝对差0 |
| packed全量verify | 189035行，失配0，4×4布局 |
| 严格子集对拍 | 400集、189035样本，feature/sample失配均为0 |
| source＋packed文件分配量 | 202152935424 B，约188.270 GiB |
| 自算统计量 | 1476个完整batch、188928个样本；尾部不足batch的107个样本未纳入统计 |
| count20更新与checkpoint | `state_step=20`、`loop_step=checkpoint_step=19`；CPU实际恢复61个EMA叶逐项一致 |

本结论来自counting自己的完成日志、`build_sizes.json`、source/packed元数据和独立的 `subset_eq.log`；full的PASS只作为前置和对照来源，没有替代counting的验收。

## 2. 用户原话与范围

用户对数据明确：「纯counting跑4*100 公开集合的子集合」，执行授权为「恢复计划中的建库，严格按前置闸门推进」。依据[原版80k计划](../../../0925-orig-80k-full-counting-4plus4-plan.md)，本库使用同一公开全集中四个counting任务的全部100集，独立构造source、4×4 packed与本库norm_stats，不从full库切分改号，不生成8×8 packed、Wan或motion。

用户同时决定「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」，以及「允许这四键缺失与 None 等价」。后一条仅限 `motion_emb/motion_pos/motion_mask/mem_order` 缺失与严格None，其他键、非None值和signed zero要求不变，原始dtype/raw SHA仍留证。这些许可仅适用于指定训练输入的比较口径；本次full/counting子集比较仍严格核对feature文件字节及pkl字段的dtype、shape与原始字节，未借主机dtype许可放宽它。

本档案包含已完成的数据构建、严格子集验收及count20完成摘要；真实训练、保存和恢复的独立记录见[训练档案](../../training-doc/smoke-orig80k-count-0925/launch.md)。正式 `INPUT_EQ`、P1、100步轨迹对拍、300步perf及80k仍分别验收。

## 3. 版本与代码状态

实际Beta为 **`49a333eb18e8d6ff1143bf7871ef7c498ab91579`**，主题 `commitV11.11Beta: 锚定两个公开正式库及20步可读性检查`。counting日志的 `BUILD_HEAD/BUILD_END_HEAD`、source构建分片指纹、finalize provenance及packer均记录该提交。起末均执行HEAD相等和工作区clean检查，最终成功退出包含这些检查。

同组full库也是这个Beta；counting预检读取full已经完成的日志和报告，核对full完整阶段退出、规模、清单绑定、packed verified和统计量摘要后才启动。更早的16集冒烟Beta `42b91cd96499bd51fcb6acaeedd642b368dbefeb` 及结果提交 `de6354fd7c0f78b347c7b07c4b8ee0c8b9b37418` 仅作为已有前置，不能误记为counting起跑版本。

本组运行期间保持该Beta不变；本档案和正式records在两库及两次20步全部完成后回填。起跑前档案可由该Beta还原，结果回填不改变启动版本，也不将事后结果提交当成起跑锚点。

## 4. 原始来源与输入pin

公开原始目录为 `/scratch/hongze/robomme_data_h5`，新硬链接目录为 `/scratch/hongze/robomme_data_h5_counting4`。建立前检查新根不存在、父目录为实体且设备相同；每次 `ln` 后立即比较源和目标的 `stat -c '%d:%i'`，没有复制第二份126 GB H5内容。

| H5任务 | 源和目标共同device:inode | 单文件size B |
|---|---|---:|
| BinFill | `2304:24427642016` | 40186950704 |
| PickXtimes | `2304:24427642012` | 35814958880 |
| StopCube | `2304:24427642013` | 21074856224 |
| SwingXtimes | `2304:24427642017` | 28947436208 |

四文件size合计 **126024202016 B**。实际构建日志记录 `HARDLINKS=PASS files=4`；归档核对时又只读复核了这八个路径的stat，源/目标仍逐对同设备、同inode、同size，没有重新读取大型H5内容。

新四文件hash与已提交16文件pin的对应投影一致，finalize以sha256档再次复核。参考input文件SHA为 `36c33bd6fdb012b72b692596c83982d2a8dc0f56e6ef7e45781ec9ebc954c640`，参考episode规范内容SHA为 `34a259ea364bbc642b2b63ab22da4497056116196bede6cc09929a5b9548d5b2`。本库新input文件SHA为 `39f12dffa5dcebc60a366e3994b75058701a62ed185556166a3229742261dd4d`。

本库episode文件自身SHA为 `fa0535335751910451c483d198cda151f04d25df36b75467c7460d118e25ad52`，规范内容SHA为 **`6b309822aa604a31f6eb0dfb5f02d5482573dc5fa11dc5c4f917880aa92ba986`**。日志中的 `SOURCE_PROVENANCE` 同时记录本轮与参考摘要；没有仅比较现场派生清单来循环自证来源。

## 5. 任务、划分与样本口径

scan显式传四任务名单和每任务100集，实际规范顺序为BinFill、PickXtimes、StopCube、SwingXtimes；CLI名单中的顺序不用于推断全局编号。四任务清单如下，所有 `exec_start_idx` 均为0，故总帧数等于执行样本数。

| 任务 | episode数 | 总帧 | 执行样本 |
|---|---:|---:|---:|
| BinFill | 100 | 60282 | 60282 |
| PickXtimes | 100 | 53720 | 53720 |
| StopCube | 100 | 31614 | 31614 |
| SwingXtimes | 100 | 43419 | 43419 |
| 合计 | 400 | 189035 | 189035 |

最长episode为1044帧。source为每个时步生成feature及执行pkl；实际文件计数分别都是189035，另有400份 `kept_indices.json`。新库的全局episode和执行样本编号独立连续，跨库身份按 `(h5_file, raw_ep_idx, step_idx)` 对齐。

norm_stats沿用固定Beta的 `compute_norm_stats.main()/create_data_loader()`：batch128、workers4、shuffle及该函数默认seed0，取 `len(dataset)//128=1476` 个完整批次。实际统计 **188928个样本**，尾部不足一个batch的107个shuffle样本不进入统计；训练库仍保留全部189035个执行样本。这里没有更改统计脚本或训练seed42。

## 6. 启动命令与配置还原

实际会话为 `cnt-pub400-20260926T004948Z`，pane `%540`，包装PID **3924000**、body PID **3924005**。会话在同一shell中按 `run_stage()` 串行执行十二阶段，并记录每阶段UTC与退出码。运行时保存的命令正文为 `v1-store/bench/orig80k-build-preflight-0925/cnt-pub400-20260926T004948Z.command.txt`，SHA256为 `35c0af03c1a8f58e3022eece8b0231521c97dcc863b6e3fab2c2d3a9c7fb6f2f`。该文本用于还原实际调用，不作为独立bash副本复制到正式docs。

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

## 7. 构建参数与格式

manifest参数为 `--episodes-per-task 100 --num_shards 1`，四任务名单显式给出。实际SigLIP使用8个动态worker和GPU0–7，预检每卡空闲至少70000 MiB；finalize在GPU7上以 `--input_level sha256 --spot_check 1024` 运行。CPU pack/verify都为48进程，`reader=decode`，没有使用verify抽样档。

source保留构造器原样多尺度feature和执行pkl，另生成 `framesamp-4x4-v1` packed。实际有31个image part、1044个位置行、189035个总行；`status=verified`、`manifest_scope=full`。这里的full表示完整覆盖本库400集清单，不代表它变成了16任务全集。逐行摘要覆盖189035行，大小3024560 B，元数据记录SHA为 `4ed01f58bab7def1c953957fef4634dd58793c0dad05abdf3da0133448d2fc9b`。

本库独立统计量为 `v1-store/train-assets/mme_vla_suite/4task-counting-pub-400ep/robomme/norm_stats.json`，完整SHA为 **`a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9`**。state/actions均含mean/std/q01/q99四字段，每字段8个数，全部有限，std非负，q01不大于q99。后续counting训练的 `assets-dir` 指向该文件上方的 `4task-counting-pub-400ep` 目录，asset-id仍为robomme；不重复拼接robomme，也不使用full自算或原版文件代替。

最终report再次确认full自算norm SHA `c5d45b4bdb481ca46d18aeb32446bbedec34de435eef31d3212a179ee7868bcd` 和原版norm SHA `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5` 与counting起跑前一致，本流程没有改写两份已有统计量。

## 8. 硬件、存储、预算与耗时

环境为AWS单机8×`NVIDIA A100-SXM4-80GB`，本地NVMe RAID `/dev/md0`、XFS。8个source分片指纹均为同一主机、JAX/jaxlib 0.5.3和本组Beta；packer记录Python 3.11.15、NumPy 1.26.4、ml_dtypes 0.4.1。8条host记录表示8个worker，Slurm字段为空，未使用集群。

counting预检发生在full成功结束之后。预算记录表明当时只读stat检查了full的768897个feature和476857个pkl，最大文件分配量分别为606208 B和397312 B，与smoke下限相同，取两者较大值继续预算counting。full与smoke已占用的空间没有再次计入“剩余待构建”需求。

| counting剩余预算项目 | 字节 |
|---|---:|
| source feature | 114594529280 |
| source pkl | 75105873920 |
| packed图像、state及逐行摘要 | 12397671440 |
| packed位置表 | 51314688 |
| episode元数据余量 | 26214400 |
| 数据基数 | 202175603728 |
| 数据加10%余量 | 222393164101 |
| 缓存环境64 GiB＋验证临时48 GiB＋日志其他16 GiB | 137438953472 |
| 预留300 GiB | 322122547200 |
| 所需可用空间 | **681954664773** |
| 预检时可用空间，2026-09-26 00:50:31 UTC | **1065860091904** |

预算文件SHA为 `66b440a7c628c2123471a75c1059fe1d27c0858e83ae8428211f2d1c26ddfc3d`，绑定full的PASS报告、日志与规范manifest。以上为起跑前外推及余量；本次实际文件体积另列：

| 对象 | 逻辑字节 | 文件分配字节 | 文件数 |
|---|---:|---:|---:|
| source | 188750826996 | 189703917568 | 378480 |
| packed及其meta | 12449010901 | 12449017856 | 36 |
| 两者合计 | 201199837897 | 202152935424 | 378516 |

source实有189035个feature、189035个pkl及400个kept_indices文件；实测最大F/P仍为606208/397312 B。分配量来自构建report逐文件累加 `st_blocks*512`，不包含全部目录/inode开销，表中也不包含库根日志、manifest及独立stats目录。report结束时可用 **863485992960 B**，约804.184 GiB，高于300 GiB保留量；它是当时快照，不代替后续count20或正式训练起跑检查。

本轮未独立采集构建临时写盘峰值，也没有配套稳态GPU利用率；只报告阶段耗时和最终体积，不据此给出性能瓶颈或正式80k预算结论。

## 9. 构建过程与阶段日志

原始构建日志为 `v1-store/logs/cnt-pub400-20260926T004948Z.build.log`。下表使用日志秒级UTC边界，十二个阶段退出均为0。SigLIP程序内部报告2457秒，秒级开始/结束差为2458秒，保留各自口径而不改写成同一精度。

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

finalize确认四个H5的sha256同源、feature目录缺失0、pkl实得189035/期望189035、8份sidecar覆盖400集、残留claim为0。1024条复算的最大绝对差为0；该浮点最大差判据不称为原始文件字节比较。packed的verify则实际进行三键字节对拍，189035行零失配。

关键判定原文：

```text
HARDLINKS=PASS files=4
INPUT_PIN=PASS files=4
EPISODE_IDENTITY=PASS episodes=400 timesteps=189035 samples=189035
STAGE_DONE stage=siglip workers=8 items=400 elapsed=2457s
FINALIZE_EXIT_CODE=0
VERIFY_PACK=PASS scanned=189035 mismatches=0
SUBSET_EQ=PASS episodes=400 samples=189035 feature_mismatch=0 sample_mismatch=0
SUBSET_TASK_EXIT=0
SUBSET_TEE_EXIT=0
NORM_STATS=PASS training_uses=counting_self_norm_stats batches=1476 samples=188928 dropped=107
BUILD_END_HEAD=49a333eb18e8d6ff1143bf7871ef7c498ab91579
TASK_EXIT=0
TEE_EXIT=0
EXIT_CODE=0
```

`COUNTING_DATA_BUILD=PASS` 后的完整JSON与现存 `meta/build_sizes.json` 一致。原始构建日志SHA为 `4074acb1c1659c54881c7bf76805e2136b5f821c891542b2b99aeeb51e38757f`。

## 10. 验收与内容一致性

本库的source stats为 `execution_samples=total_samples=189035`；packed为本库规范manifest **`6b309822aa604a31f6eb0dfb5f02d5482573dc5fa11dc5c4f917880aa92ba986`** 的全量verified结果。其 `source_provenance_sha256` 与实际source provenance文件SHA **`4271e7e646760a5d4efa1fc4949ffed706e5842e9939bddc66751ff033a59258`** 相同，位置与清单指向本库，pack锁已经释放。

严格对拍使用已经PASS的 `16task-pub-1600ep` 作为full参照，其规范manifest SHA为 **`fb1bdbcbcc176d15475332b728fa218bb545b4a8c9dcaa9ce1ec9e0bb17f174f`**。`check_subset_eq.check_subset()` 先确认两清单的规定集合、数量和连续编号，再按物理episode/局部step全覆盖比较：400集、189035个feature时步、189035个执行样本。对应feature文件（含kept_indices）按字节比较；pkl解码后逐键核对类型、dtype、shape和原始字节，覆盖两幅图像、完整actions、state、prompt和身份字段。唯一映射豁免是两库各自的 `epis_idx`，其值仍须匹配对应manifest。

`subset_eq.log` 的完整SHA为 **`cff7d4aa356454895095006b9b7dbc501146b858db8a6b0e30d4d26861e4a1fc`**；日志含唯一成功判定，以及明确绑定上述full/counting两份规范SHA的 `SUBSET_MANIFEST`。counting报告再保存该日志SHA与400集/189035样本计数。构建结束复核及归档核验均确认这些绑定，没有只看一个PASS字符串。

这项结果证明本轮两库对应source内容按上述范围严格一致。它没有比较两种训练loader的全部模型输入，也没有比较训练标量、EMA、优化器状态或单跑/并跑轨迹。

**独立count20已完成。** 会话 `orig80k-count-read20-20260926T014535Z`，包装PID3957023、body PID3957028；2026-09-26 01:45:54 UTC启动，01:51:37 UTC整体PASS。run为 `smoke-orig80k-count-0925`，UUID为 `ba82f509-8189-4002-809a-021f11d58d97`，使用同一Beta、GPU4–7、global batch64、fsdp4、workers4、seed42及本库独立norm SHA `a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9`。W&B保持开启，原版训练参数不变，仅smoke步数覆盖为20。

最终 `state_step=20`、`loop_step=checkpoint_step=19`，末步保存等待完成。CPU完成器真实恢复61个EMA叶，其中23个bfloat16、38个float32；全部叶的dtype、shape、原始字节SHA及有限性记录与末步摘要一致。恢复结果 `completion.json` SHA为 **`97cd2cdc2c6feb6c9c5612813c64f871cadcb53259a44829ba58bef72549b268`**。driver训练终态与wrapper整体终态分别验收，训练、tee、CPU完成器均退出0，最后为 `READABILITY20=PASS`。此项只证明当前库的20次更新和一次保存恢复，不构成正式输入、100步对拍或性能证据。

## 11. 计划外事件与处置

控制器保存的启动记录表明，首次将较长命令正文以内联方式交给tmux时出现 `command too long`。当时确认没有对应会话或日志，随后改为在tmux中执行已经保存且SHA核对一致的命令文件。改变的是命令传输方式，实质阶段、环境、参数与内容校验未改变；成功运行的实际会话和起止时间按本档案记录，不将未启动的尝试算作一轮构建。

成功运行的十二阶段、子集工具与tee、外层任务与tee全部退出0，没有本次数据失败。uv弃用提示保留在原日志，未据此改依赖。实际是一份detached tmux中的串行阶段，不为满足预建模板而补造多个会话。

本次未覆盖旧库、full统计量或原版统计量，没有复制第二份原始H5，也没有自动删除新库、硬链接目录或已有产物。主机dtype和四motion键许可未用于规避严格子集比较。

## 12. 当前结论与下一步

counting数据、独立norm、严格子集验收和count20真实保存恢复已完成；同组[full数据及full20](../16task-pub-1600ep/README.md)也已完成。两个库分别保存输入pin、manifest、source/packed来源指纹和各自成功记录，没有拿full结论替代counting结果。

两库正式 `INPUT_EQ`、P1、100步上游与同入口单跑/并跑对拍、300步perf及80k均尚未通过。20步可读性不能替代这些检查。P1/100步对拍驱动目前关闭W&B，是否仅允许对拍关闭仍待用户答复；20步、perf与80k的W&B保持开启。perf所需保存期间磁盘峰值证据尚未采集，是否增加0.5秒只读checkpoint目录分配量与scratch可用量采样，或暂停perf，也仍待用户决定；没有自行改变判据或将最终文件体积当峰值。

后续清理仅在相应归档和归属确认完成后按授权范围处理。原始公开H5继续保留；硬链接是同一文件的另一目录项，不能因为counting完成就把原件或其他输出自动删除。

## 13. 归档文件清单

本目录已归档 **28个文件，530983 B**，包括27个payload和1个正式 `archive_checks.json`；README与launch不计入该文件数。正式[核验清单](records/archive_checks.json) SHA256为 **`eec6e85003a79fdd7a6f7997b87f6481d259c1199495ff7745202f144aea1351`**，登记原始来源、暂存核验、正式目标、逐文件SHA和bytes。它是正式目标的清单，未把先前ignored暂存清单的SHA当成正式清单SHA。

18份普通记录复制后字节相同且源SHA复核不变；9份构建/worker日志先拆CR、去 `%|` 进度态，删除168条进度态并完整保留逐份关键行。正式落档另采用仅去行尾空格/Tab的规范化，核验全部行在相同规范化后逐行相等；本组数据日志实际没有因此改变字节。`subset_eq.log` 无需清洗，保持原字节及其已绑定SHA。凭据检查未发现匹配。清洗前后行数、删除数、关键行一致性和时间均由核验清单留证。

| 已归档内容 | 正式位置 |
|---|---|
| 400集episode与四文件输入pin | [episode_manifest.json](records/episode_manifest.json)、[input_manifest.json](records/input_manifest.json) |
| 规模、体积、统计及严格子集结果绑定 | [build_sizes.json](records/build_sizes.json)、[subset_eq.log](records/subset_eq.log) |
| source统计与来源指纹 | [source_stats.json](records/source_stats.json)、[source_provenance.json](records/source_provenance.json) |
| 8个worker分片指纹 | [source_shards/](records/source_shards/) 中的 `_shard0of8.json`…`_shard7of8.json` |
| packed元数据与打包进展 | [packed_store_meta.json](records/packed_store_meta.json)、[pack_progress.jsonl](records/pack_progress.jsonl) |
| 本库独立统计量 | [self_norm_stats.json](records/self_norm_stats.json) |
| full完成后的counting剩余预算 | [post_full_counting_budget.json](records/post_full_counting_budget.json) |
| 全构建及8个worker清洗日志 | [counting-build.summary.log](records/logs/counting-build.summary.log)、[logs/](records/logs/) 中8份 `siglip-gpu*.summary.log` |
| 复制、日志清洗与正式目标核验 | [archive_checks.json](records/archive_checks.json) |

关键原始记录SHA如下；这些普通JSON及子集日志与正式归档字节相同，其他清洗日志的新SHA以正式核验清单为准。

| 对象 | SHA256 |
|---|---|
| count `meta/build_sizes.json` | `69f36813bf9168ff08f57b64f6e6587c5746e3466caf84d529f38bde84b40f7e` |
| count `source/meta/stats.json` | `e55e1c44c347dcd3d9dfb735f019a00ce00dcb8884501368d6b65225a332d417` |
| count `source/meta/provenance.json` | `4271e7e646760a5d4efa1fc4949ffed706e5842e9939bddc66751ff033a59258` |
| count `framesamp/meta/store_meta.json` | `6c0ffe3968149fd666091e1809fa55a4c1540132db906ec27a40a584cb6aa748` |
| count `meta/subset_eq.log` | `cff7d4aa356454895095006b9b7dbc501146b858db8a6b0e30d4d26861e4a1fc` |
| 本库自算norm_stats | `a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9` |
| counting预算记录 | `66b440a7c628c2123471a75c1059fe1d27c0858e83ae8428211f2d1c26ddfc3d` |

count20的训练、保存与恢复资料归入[独立训练档案](../../training-doc/smoke-orig80k-count-0925/launch.md)，不计入本构建目录的28文件。没有归档H5、feature数组、packed二进制、权重或缓存；没有把 `.sh`、`.yaml` 或 `.command.txt` 的脚本正文作为独立副本搬入docs。已有代码、配置及参考pin通过固定Beta还原，必要实际命令留在正文代码块。
