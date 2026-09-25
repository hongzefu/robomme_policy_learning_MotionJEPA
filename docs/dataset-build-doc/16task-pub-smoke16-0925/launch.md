# 公开16任务每任务首集构建冒烟：起跑前记录

## 目标、授权与当前状态

**尚未启动，本文不代表任何构建闸门已经通过。** 用户最新原话：「恢复计划中的建库，严格按前置闸门推进」。依据[原版80k计划](../../../0925-orig-80k-full-counting-4plus4-plan.md)，先重新核对公开16个H5的摘要与全部1600集身份，再用每任务首集共16集验证 source、4×4 packed 和全量verify链路。该冒烟覆盖任务首集，不宣称覆盖每任务所有难度。

P0已从clean `5489e4b3a92197d0e9a37421b1e6415b3022b613`通过，见[环境结果](../../training-doc/orig80k-env-0925/result.md)。另行完成的3样本CPU字段取证显示共有字段精确数值一致，motion缺键/None在当时严格schema口径下FAIL；用户查看后新增原话：「允许这四键缺失与 None 等价」。该许可仅限`motion_emb/motion_pos/motion_mask/mem_order`缺失与显式None，不扩展到其他键或任何非None值；主机数值精确、原始dtype/raw SHA留证及训练标量/状态bitwise判据不变。新许可需按对应工具修复和验证落地，不回写旧取证为当时PASS。本档案不启动训练，不把构建成功当作模型输入或训练对拍通过。

## 版本与起跑前闸门

本档案必须先提交，实际任务再从clean HEAD起跑。**实际启动提交、起止时间和会话完整名称尚未产生，待起跑日志写入，不能把当前开发HEAD冒充未来启动版本。** 启动时传入已提交的完整40位 `BUILD_HEAD` 字面量，并记录 `git rev-parse HEAD` 与 `git status --porcelain`；不为空或不等于该锚点就停止。正式库另须起跑前Beta锚点，不能以事后提交追认。

源码与参数的稳定还原锚点为 `scan_manifest.cmd_build()`、`finalize_checks.cmd_hash_inputs()/cmd_check()`、`run_local.main()/worker_cmd()` 和 `pack_framesamp_store.py` 的 `pack/verify` 子命令，均使用实际启动提交读取：`git show <实际启动完整SHA>:scripts/dataset/<对应文件>`。不在档案复制脚本或yaml。

环境为AWS本机8×A100-SXM4-80GB，`/scratch`为`/dev/md0` XFS本地NVMe RAID。最近只读快照可用`1792014577664 B`、使用率77%，八卡空闲；两个正式库和counting原始硬链接目录不存在。此为当时快照，执行前重测。原始目录为`/scratch/hongze/robomme_data_h5`，既有公开来源为`Yinpei/robomme_data_h5`；不下载新数据、不自行删除任何旧产物。

起跑前核对原始根和输出父目录实体路径、没有外链，下面两个输出根必须不存在（包括悬空符号链接）。磁盘按“剩余构建峰值预算 + 300 GiB”核算；以下预算由既有产物抽样外推，**不是三个新库已经构建后的体积实测**。

| 项目 | 数量或字节 | 证据性质 |
|---|---:|---|
| 单总帧feature文件组的实际分配量F | `606208 B` | 既有产物抽取4个样本实测，用于当前链路外推 |
| 单执行样本pkl的实际分配量P | `397312 B` | 既有产物抽取4个样本实测，用于当前链路外推 |
| 16任务正式库 | `657.6759 GiB` | 按总帧/执行样本及packed项外推，未构建实测 |
| counting正式库 | `188.2907 GiB` | 同口径外推，未构建实测 |
| 16集冒烟库 | `6.8541 GiB` | 同口径外推，待冒烟校正 |
| 三库合计再加10% | `938.1027 GiB` | 数据分配量的保守估算 |
| 缓存与独立环境 | `64 GiB` | 保守预算，不是实测占用 |
| 验证checkpoint及临时产物 | `48 GiB` | 保守预算，不是实测峰值 |
| 日志与其他开销 | `16 GiB` | 保守预算，不是实测占用 |
| 保留空间 | `300 GiB` | 计划既定保留量 |
| 当前完整预算门槛 | `1466841656330 B`，约`1366.1027 GiB` | 本次分项预算之和，不是通用固定门槛 |
| 最近可用空间 | `1792014577664 B`，约`1668.9436 GiB` | 只读磁盘快照，较本次门槛多约`302.84 GiB` |

按当前估算和快照，预算足够；但每阶段起跑前必须重测，冒烟后以实际分配量和临时峰值校正。完整分项、样本来源和字节计算记录随结果归档，实际闸门使用上表完整字节值或更新后的已记录预算，不从舍入GiB反算，也不以1.5T等固定数替代。

## 路径与环境

| 项目 | 本轮明确路径 |
|---|---|
| 全量来源前检证据 | `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925` |
| 冒烟库 | `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-smoke16-0925` |
| 已提交来源pin与全集清单 | `docs/dataset-build-doc/16task-h5-scan/records/input_manifest.json`、`episode_manifest.json` |
| 阶段原始日志 | `v1-store/logs/<完整会话名>.<阶段>.log` |
| worker日志 | `v1-store/datasets/16task-pub-smoke16-0925/logs/` |
| 归档 | `docs/dataset-build-doc/16task-pub-smoke16-0925/records/`，运行后创建并归档实测数据 |

以下环境在每个实际阶段所在的detached tmux中设置；`paths.sh`只source一次。缓存不覆盖HOME，不使用旧`/data`或NFS路径。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
set -euo pipefail
source scripts/dataset/paths.sh
export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_CACHE_PATH="$V1_STORE/cache/cuda" PYTHONDONTWRITEBYTECODE=1
RAW=/scratch/hongze/robomme_data_h5
CHECKS="$V1_STORE/bench/orig80k-build-preflight-0925"
LIB="$V1_STORE/datasets/16task-pub-smoke16-0925"
REFERENCE="$REPO_ROOT/docs/dataset-build-doc/16task-h5-scan/records"
MANI="$LIB/meta/episode_manifest.json"
: "${BUILD_HEAD:?必须传已提交的完整启动SHA字面量}"
test "$BUILD_HEAD" = "$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
```

## 阶段命令与验收

下列代码块是分阶段执行的命令体，必须先套用下一节的tmux与日志纪律；不是未经检查就整段启动的脚本。任一阶段失败即停止后续阶段，保留现场并向用户报告；重试前根据本轮阶段记录判断已完成范围，不覆盖重跑输出根。

**阶段1：全量公开来源前检。** 首次要求`CHECKS`和`LIB`不存在，然后重新hash全部16个H5、扫描每任务100集，与已提交pin逐文件size/SHA和全部episode身份绑定。`check_orig80k_sources.py`只接受全集1600集或counting400集，不能拿16集冒烟清单调用它制造通过。

```bash
test ! -e "$CHECKS" && test ! -L "$CHECKS"
test ! -e "$LIB" && test ! -L "$LIB"
uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs \
  --raw_dir "$RAW" --out "$CHECKS/input_manifest.json"
uv run --no-sync python scripts/dataset/scan_manifest.py build \
  --raw_dir "$RAW" --episodes-per-task 100 --num_shards 1 --out "$CHECKS/episode_manifest.json"
uv run --no-sync python scripts/dataset/check_orig80k_sources.py \
  --input-manifest "$CHECKS/input_manifest.json" --reference-input "$REFERENCE/input_manifest.json" \
  --manifest "$CHECKS/episode_manifest.json" --reference-manifest "$REFERENCE/episode_manifest.json"
```

通过要求`INPUT_PIN=PASS files=16`，`EPISODE_IDENTITY=PASS episodes=1600 timesteps=768897 samples=476857`及该阶段`EXIT_CODE=0`。保留`SOURCE_PROVENANCE`中的参考及新清单摘要。这些是预定判据，尚未实跑。

**阶段2：冒烟清单。** 仅选择各H5按`raw_ep_idx`升序的第一集，不从相同集号推断跨任务身份；核对16个`(h5_file, raw_ep_idx)`恰为阶段1清单中每任务首个身份，逐项核对`num_timesteps/exec_start_idx/exec_samples`。本清单全局编号和offset因子集重排，不与全集逐字节比。总帧数、执行样本数以本次清单实算归档，不预填未经实测的数字。

```bash
uv run --no-sync python scripts/dataset/scan_manifest.py build \
  --raw_dir "$RAW" --episodes-per-task 1 --num_shards 1 --out "$MANI"
```

**阶段3–6：真实构建及检查。** `run_local`用GPU0–7、8个worker；`--num_shards 1`是manifest设置，不等于只使用一个GPU。finalize单独用GPU7复算，CPU打包与verify隔离GPU；只生成4×4 packed，不生成8×8或motion。

```bash
uv run --no-sync python scripts/dataset/run_local.py --stage siglip \
  --lib "$LIB" --gpus 0,1,2,3,4,5,6,7 --raw-dir "$RAW" --require-free-mib 70000
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/dataset/finalize_checks.py check \
  --manifest "$MANI" --out "$LIB/source" --raw_dir "$RAW" \
  --input_manifest "$CHECKS/input_manifest.json" --input_level sha256 --spot_check 1024
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack \
  --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp" --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify \
  --store "$LIB/framesamp" --resume --procs 48
```

要求SigLIP所有worker成功及`STAGE_DONE stage=siglip`，finalize所有检查通过，packed全量verify完成且`store_meta`记录`framesamp-4x4-v1`、`verified`、与当前manifest绑定；所有阶段退出0。按每任务记录实际构造结果，不能仅凭元数据scan通过就声称图像、动作等字段均可构建。

**阶段7：实测预算和结果交接。** 记录source特征、pkl、packed各自逻辑字节与实际分配字节、临时峰值和耗时；分别按全集/子集的实际总帧T及执行样本E外推，数据开销至少加10%，另列缓存、日志、独立环境及后续验证权重预算。冒烟成功且更新预算通过后，才允许[16任务正式库](../16task-pub-1600ep/launch.md)起跑。此处不算吞吐结论，不预填实际峰值；若要报告GPU利用率，需留500ms有效采样、稳态窗口、均值/0%占比及慢步分层。

## tmux、日志与精确会话清单

预计超过5分钟的阶段都放detached tmux，使用`PYTHONUNBUFFERED=1`、`set -o pipefail`和`tee`。会话前缀使用`pub16-`；**当前实际创建会话清单为空**。`pub16-pin-<UTC>`、`pub16-smoke16-<阶段>-<UTC>`只是命名模板，不是已创建会话。实际起跑时逐项记录完整名称、精确PID、阶段、`BUILD_HEAD`、起止UTC、日志路径及退出码。

| 阶段 | 实际完整会话名 | 实际启动HEAD/日志/退出码 |
|---|---|---|
| 全量hash、scan与pin绑定 | 待起跑填入 | 未启动 |
| 16集scan与身份检查 | 待起跑填入 | 未启动 |
| SigLIP | 待起跑填入 | 未启动 |
| finalize | 待起跑填入 | 未启动 |
| pack与verify | 待起跑填入 | 未启动 |
| 预算实测与交接 | 待起跑填入 | 未执行 |

每阶段命令体由会话中的shell函数或子shell执行。外围关闭`errexit`后执行`阶段命令 2>&1 | tee <全新日志>`，立即保存整个`PIPESTATUS`；分别打印任务与tee退出码，最终`EXIT_CODE`在两者任一个非零时也必须非零。最终追加日志本身失败不能算成功；阶段命令内部仍使用`set -euo pipefail`或显式逐项失败返回，禁止失败后继续后续阶段。不得复用/截断已有日志。

存活检查为`tmux has-session -t '=实际完整会话名'`；自然退出后核对日志退出码。必要清理时仅对本轮清单内的完整名称逐个操作，前后各执行`tmux ls`确认只少目标会话；禁止全局kill、前缀猜测或批量删除。每份阶段日志独立流式监听，`tr`用`stdbuf -oL`、`grep`用`--line-buffered`、`sed`用`-u`、`awk`显式`fflush()`。

## 归档与当前下一步

仅归档实际生成的input/episode清单摘要与必要快照、守卫判定、清洗后的阶段/worker日志、资源采样及预算实测，不复制已提交pin、脚本或yaml，不归档H5、features、packed实体或模型权重。清洗按`tr '\r' '\n'`后剔除tqdm中间态，核对每阶段退出记录和关键判定未丢失。P0代码及旧pin由其Git提交还原。

当前全部构建项待执行；预算估算已有审查数字，完整分项记录及冒烟实测尚待归档。成功后先归档结果与预算再讨论本轮冒烟大产物清理，不自动删除；来源前检证据继续供两个正式库溯源。任何构建FAIL都不改变用户批准的四键缺失/None边界，构建与输入取证分别报告。
