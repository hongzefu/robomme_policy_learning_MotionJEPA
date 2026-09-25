# 公开16任务1600集正式库：起跑前记录

## 范围与启动条件

**尚未启动，不预填Beta提交或成功结果。** 用户最新原话：「恢复计划中的建库，严格按前置闸门推进」。本库使用公开`Yinpei/robomme_data_h5`的16任务×100集，交付source、4×4 packed和仅供比较的自算norm_stats，不建8×8、Wan或motion。原版正式训练仍使用既有`v1-store/train-assets/mme_vla_suite/robomme/norm_stats.json`，禁止用本库自算版静默替换。

P0已通过；必须先完成[来源前检与16集构建冒烟](../16task-pub-smoke16-0925/launch.md)，归档全部阶段成功、来源pin和冒烟实际分配字节，再重新计算预算。当前估算的full/count/smoke加10%为938.1027 GiB，临时等分项保守估计128 GiB，加300 GiB保留量共1366.1027 GiB；可用1668.9436 GiB仅为近期快照。估算不得写成实测，完整字节分项记录为闸门依据；冒烟后校正，阶段前重测。

用户另已明确「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」及「允许这四键缺失与 None 等价」。后者只限`motion_emb/motion_pos/motion_mask/mem_order`缺失与显式None，其他键、非None值仍严格；原始dtype/raw SHA留证、signed zero和训练bitwise要求不变。既有3样本取证不是全库输入验收；本档案不启动训练，后续训练仍须全部前置闸门。

## 版本与可复现面

正式SigLIP/raw构造前必须先完成本轮代码及档案的Beta提交，从该完整40位字面量`BUILD_HEAD`的clean HEAD启动。**实际Beta SHA、会话、UTC和退出码尚未产生，起跑现场填入日志及本档案；不把事后提交当启动版本。** 若来源前检与正式构建不在同一提交，分别记各阶段HEAD和差异，不声称全过程为一个未变版本。

还原使用`git show <实际Beta完整SHA>:scripts/dataset/scan_manifest.py`、`finalize_checks.py`、`check_orig80k_sources.py`、`run_local.py`、`pack_framesamp_store.py`及`git show <实际Beta完整SHA>:scripts/training/compute_norm_stats.py`。稳定入口为`scan_manifest.cmd_build()`、`run_local.worker_cmd()`、`finalize_checks`的hash/check、packed的pack/verify和`compute_norm_stats.main()`；全部覆盖参数见下方命令。配置、脚本及yaml不另行拷贝入档案。

## 路径、环境与阶段命令

输入`/scratch/hongze/robomme_data_h5`。库根`/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep`，统计量根`v1-store/train-assets/mme_vla_suite/16task-pub-1600ep`，两处起跑时必须全新、实体路径且无外链。规范期望为1600集、768897总帧、476857执行样本；这些来自已提交参考清单，是验收目标，不是本次已生成结果。

各阶段必须在已记录的detached tmux内按下面顺序执行，遵守[冒烟档案的日志与会话纪律](../16task-pub-smoke16-0925/launch.md)。下方代码块是阶段命令体，外围为每阶段独立tee日志和`EXIT_CODE=`；不能不带外围就粘贴启动。任一失败停止后续阶段。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
set -euo pipefail
source scripts/dataset/paths.sh
export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_CACHE_PATH="$V1_STORE/cache/cuda" PYTHONDONTWRITEBYTECODE=1
RAW=/scratch/hongze/robomme_data_h5
LIB="$V1_STORE/datasets/16task-pub-1600ep"
MANI="$LIB/meta/episode_manifest.json"
INMANI="$LIB/meta/input_manifest.json"
STATS="$V1_STORE/train-assets/mme_vla_suite/16task-pub-1600ep"
REFERENCE="$REPO_ROOT/docs/dataset-build-doc/16task-h5-scan/records"
: "${BUILD_HEAD:?必须传已提交的完整Beta SHA字面量}"
test "$BUILD_HEAD" = "$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
test ! -e "$LIB" && test ! -L "$LIB"
test ! -e "$STATS" && test ! -L "$STATS"
```

**来源与全量清单阶段：**不传`--tasks`；恰好收录已提交pin的16个H5。即使前检已成功，本库实际消费的input/episode清单仍须重新生成并绑定参考，不把现场清单互相比较当来源证明。

```bash
uv run --no-sync python scripts/dataset/scan_manifest.py build \
  --raw_dir "$RAW" --episodes-per-task 100 --num_shards 1 --out "$MANI"
uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs \
  --raw_dir "$RAW" --out "$INMANI"
uv run --no-sync python scripts/dataset/check_orig80k_sources.py \
  --input-manifest "$INMANI" --reference-input "$REFERENCE/input_manifest.json" \
  --manifest "$MANI" --reference-manifest "$REFERENCE/episode_manifest.json"
```

**构建与验证阶段：**SigLIP用本机八卡；finalize用GPU7复算1024条，pack/verify及norm_stats用CPU。source保留原版多尺度feature，packed只生成4×4布局；所有阶段为独立记录的实际命令，不能因前段成功跳过后段验收。

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
```

norm_stats实际输出为`$STATS/robomme/norm_stats.json`。与原版文件逐键比较最大绝对差、非零参考项最大相对差和参考零值项，只归档差异和摘要，不因不一致自行更换训练资产。

## 会话清单、判据与交接

当前实际创建会话清单为空；使用`pub16-<阶段>-<UTC>`命名模板，运行时填完整名称。每阶段记录`BUILD_HEAD`、`git status --porcelain`、命令、起止UTC、精确PID和`v1-store/logs/<完整会话名>.<阶段>.log`。保存任务及tee退出码；任一非零使最终`EXIT_CODE`非零。每份日志行缓冲独立监听。

| 阶段 | 实际完整tmux名称 | 当前状态 |
|---|---|---|
| scan/hash/pin绑定 | 待起跑填写 | 未启动 |
| SigLIP | 待起跑填写 | 未启动 |
| finalize | 待起跑填写 | 未启动 |
| pack/verify | 待起跑填写 | 未启动 |
| norm_stats及原版比较 | 待起跑填写 | 未执行 |

只用`tmux has-session -t '=实际完整名称'`检查；清理只按实际清单完整匹配，一次一个，前后核对其他会话未变，不全局kill或按前缀猜测。源文件不动，不使用覆盖或强制清理选项。

数据阶段交付要求来源pin、全部episode身份和顺序、SigLIP worker、finalize、packed全量verify及norm_stats全部满足，实际规模和摘要与参考绑定。清单分片调度字段可以与历史不同，真实内容及全量身份不能放宽。保留实际字节量，重算counting阶段剩余预算后再进入[counting库](../4task-counting-pub-400ep/launch.md)。训练可读性检查和模型输入/训练对拍单独记录，未执行不得写成“全部交付通过”。正式结果按[十三节README](README.md)回填。
