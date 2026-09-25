# 公开16任务每任务首集构建冒烟：实际起跑记录

## 目标、授权与当前状态

**来源前检与16集构建冒烟已全部通过，结果见[result.md](result.md)。** 用户原话：「恢复计划中的建库，严格按前置闸门推进」。依据[原版80k计划](../../../0925-orig-80k-full-counting-4plus4-plan.md)，本轮重新核对公开16个H5的摘要与全部1600集身份，并完成每任务首集共16集的source、4×4 packed和全量verify。实际为7898总帧、5018执行样本；只覆盖任务首集，不宣称覆盖每任务所有难度。本文件先在Beta提交中建档，以下为运行后回填，未改写实际启动版本。

P0已从clean `5489e4b3a92197d0e9a37421b1e6415b3022b613`通过，见[环境结果](../../training-doc/orig80k-env-0925/result.md)。另行完成的3样本CPU字段取证显示共有字段精确数值一致，motion缺键/None在当时严格schema口径下FAIL；用户查看后新增原话：「允许这四键缺失与 None 等价」。该许可仅限`motion_emb/motion_pos/motion_mask/mem_order`缺失与显式None，不扩展到其他键或任何非None值；主机数值精确、原始dtype/raw SHA留证及训练标量/状态bitwise判据不变。新许可需按对应工具修复和验证落地，不回写旧取证为当时PASS。本档案不启动训练，不把构建成功当作模型输入或训练对拍通过。

## 版本与起跑前闸门

本档案已包含在起跑Beta `42b91cd96499bd51fcb6acaeedd642b368dbefeb`（`commitV11.10Beta`）中；来源前检和冒烟均从该clean HEAD启动，起末检查HEAD一致且porcelain为空。来源前检为2026-09-25 20:32:35Z→20:56:40Z，冒烟为20:58:03Z→21:09:35Z；实际会话与PID见下表。8个构建worker、finalize及packed元数据均记录同一完整Beta。后续归档提交不能替代该启动锚点。

源码与参数的稳定还原锚点为 `scan_manifest.cmd_build()`、`finalize_checks.cmd_hash_inputs()/cmd_check()`、`run_local.main()/worker_cmd()` 和 `pack_framesamp_store.py` 的 `pack/verify` 子命令，使用 `git show 42b91cd96499bd51fcb6acaeedd642b368dbefeb:scripts/dataset/<对应文件>`读取。tmux外围是运行时内联shell正文，没有独立wrapper文件；下文还原其实际环境、实质调用与退出处理，不声称该内联正文的原始字节可由Git还原。不在档案复制脚本或yaml。

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

上表保留起跑前估算。来源前检实际起跑可用`1791913361408 B`，冒烟起跑可用`1791912865792 B`，均高于该门槛；相关`BUDGET_GATE=PASS`在两份阶段日志。冒烟后实测及两正式库剩余预算见[result.md](result.md)和[post_smoke_budget.json](records/post_smoke_budget.json)，不把估算改写为实际峰值，也不以固定容量替代逐阶段预算。

## 路径与环境

| 项目 | 本轮明确路径 |
|---|---|
| 全量来源前检证据 | `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-build-preflight-0925` |
| 冒烟库 | `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-smoke16-0925` |
| 已提交来源pin与全集清单 | `docs/dataset-build-doc/16task-h5-scan/records/input_manifest.json`、`episode_manifest.json` |
| 阶段原始日志 | `v1-store/logs/<完整会话名>.<阶段>.log` |
| worker日志 | `v1-store/datasets/16task-pub-smoke16-0925/logs/` |
| 归档 | `docs/dataset-build-doc/16task-pub-smoke16-0925/records/`，已保存实测数据与清洗核验 |

以下为两个实际detached tmux使用的共同环境；`RAW_H5_DIR`在source前显式设置，每个会话只source一次`paths.sh`。缓存不覆盖HOME，不使用旧`/data`或NFS路径。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
set -euo pipefail
export RAW_H5_DIR=/scratch/hongze/robomme_data_h5
source scripts/dataset/paths.sh
export BUILD_HEAD=42b91cd96499bd51fcb6acaeedd642b368dbefeb
export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_CACHE_PATH="$V1_STORE/cache/cuda/orig80k-build" PYTHONDONTWRITEBYTECODE=1
unset XLA_FLAGS JAX_PLATFORMS
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

下列代码块还原本次实际执行的实质命令，外围使用下一节记录的tmux与日志处理。原起跑时输出根全新，现在已经存在，不应重放这些创建命令。任一阶段失败都须停止后续阶段，保留现场并报告，不覆盖重跑输出根。

**阶段1：全量公开来源前检。** 实际顺序为输出/预算检查→核参考文件SHA→SigLIP资产full校验→scan1600→hash16→pin与身份绑定。与最初命令骨架的hash/scan先后不同，未改变检查内容；两者完成后才消费结果。`check_orig80k_sources.py`只接受全集1600集或counting400集，本次传入的是全集清单。

```bash
test ! -e "$CHECKS" && test ! -L "$CHECKS"
test ! -e "$LIB" && test ! -L "$LIB"
mkdir "$CHECKS"
printf '%s  %s\n' \
  36c33bd6fdb012b72b692596c83982d2a8dc0f56e6ef7e45781ec9ebc954c640 "$REFERENCE/input_manifest.json" \
  43d9a6fbcef7eb259c12e74b7c62898d33961a6470c40586cbed1be51d1024c8 "$REFERENCE/episode_manifest.json" | sha256sum -c -
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/assets/fetch_assets.py \
  verify --level full --assets siglip_params
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/scan_manifest.py build \
  --raw_dir "$RAW" --episodes-per-task 100 --num_shards 1 --out "$CHECKS/episode_manifest.json"
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs \
  --raw_dir "$RAW" --out "$CHECKS/input_manifest.json"
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/check_orig80k_sources.py \
  --input-manifest "$CHECKS/input_manifest.json" --reference-input "$REFERENCE/input_manifest.json" \
  --manifest "$CHECKS/episode_manifest.json" --reference-manifest "$REFERENCE/episode_manifest.json"
```

实际得到`INPUT_PIN=PASS files=16`、`EPISODE_IDENTITY=PASS episodes=1600 timesteps=768897 samples=476857`、`SOURCE_PREFLIGHT_DONE=PASS`及`EXIT_CODE=0`。参考与新清单摘要随`SOURCE_PROVENANCE`保存在[清洗日志](records/logs/source-preflight.summary.log)和归档JSON中。

**阶段2：冒烟清单。** 冒烟会话先核前检日志的`SOURCE_PREFLIGHT_DONE=PASS`和末行`EXIT_CODE=0`，复测预算及LIB不存在，再执行scan16。CPU内联Python以全集清单中`first.setdefault(h5_file, full_ep)`选择每任务首集，核对`h5_file/raw_ep_idx/num_timesteps/exec_start_idx/exec_samples`五字段和总账，得到`SMOKE_IDENTITY=PASS episodes=16 timesteps=7898 samples=5018`。本清单全局编号和offset按子集重排，不与全集字节比较；[smoke_identity.json](records/smoke_identity.json)保存了实际结果。

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/scan_manifest.py build \
  --raw_dir "$RAW" --episodes-per-task 1 --num_shards 1 --out "$MANI"
```

**阶段3–6：真实构建及检查。** `run_local`用GPU0–7、8个worker；`--num_shards 1`是manifest设置，不等于只使用一个GPU。finalize单独用GPU7复算，CPU打包与verify隔离GPU；只生成4×4 packed，不生成8×8或motion。

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --no-sync python scripts/dataset/run_local.py --stage siglip \
  --lib "$LIB" --gpus 0,1,2,3,4,5,6,7 --raw-dir "$RAW" --require-free-mib 70000
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/dataset/finalize_checks.py check \
  --manifest "$MANI" --out "$LIB/source" --raw_dir "$RAW" \
  --input_manifest "$CHECKS/input_manifest.json" --input_level sha256 --spot_check 1024
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack \
  --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp" --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify \
  --store "$LIB/framesamp" --resume --procs 48
```

实际SigLIP完成8个worker、16个episode，finalize同架构复算1024条全部零差，packed全量扫描7898行、mismatch=0。`store_meta`为`framesamp-4x4-v1/verified`并绑定本次manifest；所有阶段退出0，详见[result.md](result.md)。这些结果覆盖本次16集，不能替代两正式库的独立检查。

**阶段7：实测预算和结果交接。** CPU内联Python用`os.walk`分别累计source/packed文件的`st_size`、`st_blocks*512`及文件数，并核对7898个feature、5018个pkl及F/P分配量最大值。source实际分配`6781734912 B`、packed `576983040 B`，最大F/P分别为`606208/397312 B`。没有独立临时峰值测量或GPU稳态采样，不据阶段时长作性能结论。按这些实测更新两正式库剩余预算后，要求可用`1458746144941 B`，实测`1784546611200 B`，预算PASS；该外推仍不是正式库实际体积，完整记录见[result.md](result.md)。

## tmux、日志与精确会话清单

实际通过`tmux new-session -d -s <完整名称> -c /scratch/hongze/robomme_policy_learning_MotionJEPA bash -c <内联正文>`创建以下两个会话，使用`PYTHONUNBUFFERED=1`、`set -o pipefail`和`tee`。来源前检单独一个会话；smoke的7个阶段在同一会话内串行，并各打印`BUILD_STAGE`及`STAGE_EXIT_CODE`。两会话均结束、退出0；没有据前缀清理其他会话。

| 阶段 | 实际完整会话名 | pane/包装PID | 原始日志 | 退出 |
|---|---|---:|---|---:|
| 资产、scan1600、hash16、pin绑定 | `pub16-pin-20260925T203220Z` | 3787720 | `v1-store/logs/pub16-pin-20260925T203220Z.preflight.log` | 0 |
| scan16、identity16、SigLIP、finalize、pack、verify、measure | `pub16-smoke16-20260925T205655Z` | 3798932 | `v1-store/logs/pub16-smoke16-20260925T205655Z.build.log` | 0 |

实际外围用`set -o noclobber; : > <全新日志>`独占日志，body内`set -euo pipefail`，执行`body 2>&1 | tee -a <日志>`后立即保存`PIPESTATUS`。末尾分别打印`TASK_EXIT`、`TEE_EXIT`、`END_UTC`和`EXIT_CODE`，任一非零时总退出非零，最后追加日志的tee也检查成功。smoke的`run_stage`逐次捕获argv返回码并打印阶段完成UTC，失败停止后续。两任务和所有已记录阶段均为0。

存活检查为`tmux has-session -t '=实际完整会话名'`；自然退出后核对日志退出码。必要清理时仅对本轮清单内的完整名称逐个操作，前后各执行`tmux ls`确认只少目标会话；禁止全局kill、前缀猜测或批量删除。每份阶段日志独立流式监听，`tr`用`stdbuf -oL`、`grep`用`--line-buffered`、`sed`用`-u`、`awk`显式`fflush()`。

## 归档与当前下一步

仅归档实际生成的input/episode清单摘要与必要快照、守卫判定、清洗后的阶段/worker日志、资源采样及预算实测，不复制已提交pin、脚本或yaml，不归档H5、features、packed实体或模型权重。清洗按`tr '\r' '\n'`后剔除tqdm中间态，核对每阶段退出记录和关键判定未丢失。P0代码及旧pin由其Git提交还原。

来源前检、16集构建及剩余预算检查已通过，实测记录和清洗核验已经归档。冒烟source+packed约7.36GB仍保留在原库，不自动删除；来源前检证据继续供两个正式库溯源。后续只在重新核实当前HEAD、输出和预算后推进正式库，训练仍须全部输入/对拍闸门。任何构建结果都不扩大用户批准的四键缺失/None边界。
