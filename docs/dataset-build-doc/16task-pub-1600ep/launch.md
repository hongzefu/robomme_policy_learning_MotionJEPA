# 公开16任务1600集正式库：启动实录

## 范围、起跑前约定与实际状态

**本库数据阶段及独立full20均已PASS。** 本文件由起跑前记录回填实际会话、环境和退出状态，起跑前正文仍可通过下方Beta还原。用户原话：「恢复计划中的建库，严格按前置闸门推进」。范围保持公开 `Yinpei/robomme_data_h5` 的16任务×100集，交付source、4×4 packed及仅供比较的自算norm_stats；不生成8×8、Wan或motion。full正式训练与本次full20使用原版 `v1-store/train-assets/mme_vla_suite/robomme/norm_stats.json`，没有用新自算统计量替换。

起跑前约定依次通过P0、[来源与16集冒烟](../16task-pub-smoke16-0925/result.md)、实际文件分配量预算、全新实体输出和clean Beta检查。历史初估为两库加smoke数据10%余量938.1027 GiB、缓存/临时/日志128 GiB及保留300 GiB，共1366.1027 GiB；该估算不是最终体积。冒烟得到F=606208 B、P=397312 B后，[校正预算](../16task-pub-smoke16-0925/records/post_smoke_budget.json)将所需可用字节定为1458746144941 B；本次正式起跑实测1784545525760 B，`BUDGET_GATE=PASS`。输入原件、两个新输出不存在、精确HEAD与clean检查均成功，才进入十阶段。

用户已允许主机dtype不同但数值精确一致、训练标量/状态逐位一致；另允许 `motion_emb/motion_pos/motion_mask/mem_order` 仅缺失与严格None等价。原始dtype/raw SHA留证，其他键、非None和signed zero要求不变。这些许可不放宽建库字节判据，既有3样本取证也不能代替两库正式输入验收。

## 版本、路径与日志保护

实际启动及结束HEAD均为 **`49a333eb18e8d6ff1143bf7871ef7c498ab91579`**，提交主题 `commitV11.11Beta: 锚定两个公开正式库及20步可读性检查`。来源与冒烟结果提交为 `de6354fd7c0f78b347c7b07c4b8ee0c8b9b37418`，冒烟Beta为 `42b91cd96499bd51fcb6acaeedd642b368dbefeb`，不是本库的启动版本。生产构建代码未为本轮改算法或参数；实际环境与操作记录由本次日志补齐。

输入为 `/scratch/hongze/robomme_data_h5`；库为 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep`；自算统计根为 `v1-store/train-assets/mme_vla_suite/16task-pub-1600ep`。全新输出检查包含悬空链接，原H5不改。实际规模为1600集、768897总帧、476857执行样本，已与已提交来源pin绑定。

起跑前模板要求逐阶段留退出记录；实际为一个detached tmux串行十阶段、共享一份构建日志，各stage保留独立开始/结束及退出行。外层 `set -o pipefail`、body `set -euo pipefail`，`run_stage()` 调用失败即返回并停止下游，tee与任务的 `PIPESTATUS` 分别记录；日志用noclobber拒绝复用。结束再次核对相同HEAD与clean。不是十个独立tmux或十份外层tee日志。

还原起跑前约定与默认：

```bash
git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:docs/dataset-build-doc/16task-pub-1600ep/launch.md
git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:scripts/dataset/paths.sh
```

## 实际环境与阶段命令

完整构建实际在一个detached tmux会话 `pub16-full-20260925T212238Z` 内，通过 `run_stage()` 串行执行十个阶段。包装PID为 **3815822**，原始日志为 `v1-store/logs/pub16-full-20260925T212238Z.build.log`。运行时保存的命令正文位于 `v1-store/bench/orig80k-build-preflight-0925/full_build_command.txt`，SHA256为 `02e47a92d5753f2e649860bdd72a5bf74c2b256ee422ab93ef27f6b91b0182bb`；它是运行时保存的shell文本，不是Beta里新增的脚本，也不应作为独立脚本拷入正式档案。

下面还原该正文的实际环境与实质阶段命令。原运行另有新输出检查、预算检查、`set -euo pipefail`、逐阶段退出码记录及外层tee；现在这些输出已经存在，下列文本仅供还原，不能重放来覆盖它们。

起跑前launch的CUDA缓存示例是 `$V1_STORE/cache/cuda`，实跑明确覆盖为 `$V1_STORE/cache/cuda/orig80k-build`；它不是paths.sh提供的默认值。实际环境口径如下。表中的“固定提交默认”来自 `scripts/dataset/paths.sh @ 49a333eb18e8d6ff1143bf7871ef7c498ab91579`，没有混称为运行命令手动覆盖；`run_local.base_env()` 在worker环境中再次设置的同值项也按该Beta源码还原。`LIB/MANI/INMANI/STATS/REFERENCE` 是命令正文的普通shell路径变量，见下方代码块。

| 来源 | 环境项 | 实际值或作用范围 |
|---|---|---|
| 手动export，在source前 | `RAW_H5_DIR` | `/scratch/hongze/robomme_data_h5` |
| 手动export | `BUILD_HEAD` | `49a333eb18e8d6ff1143bf7871ef7c498ab91579` |
| 手动export | `UV_CACHE_DIR` | `$V1_STORE/cache/uv` |
| 手动export | `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS` | 均为`1` |
| 手动export | `PYTHONUNBUFFERED`、`PYTHONDONTWRITEBYTECODE` | 均为`1`；前者虽在paths中已有同值，命令仍明确export |
| 手动export | `CUDA_CACHE_PATH` | `$V1_STORE/cache/cuda/orig80k-build` |
| 手动unset | `XLA_FLAGS`、`JAX_PLATFORMS` | 在实质阶段开始前清除继承值 |
| CPU阶段命令的局部覆盖 | `CUDA_VISIBLE_DEVICES=''`、`JAX_PLATFORMS=cpu` | asset_verify、scan1600、hash16、source_pin、pack、verify、norm_stats、report；不导出到后续GPU阶段 |
| SigLIP阶段命令的局部覆盖 | `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` | 调度器可见八卡；`run_local.worker_cmd()` 再将每个worker限制到其指定单卡 |
| finalize阶段命令的局部覆盖 | `CUDA_VISIBLE_DEVICES=7` | 使用GPU7复算；该阶段没有CPU平台覆盖 |
| paths.sh固定提交默认 | `OPENPI_DATA_HOME`、`XDG_CACHE_HOME`、`HF_HOME` | 分别为`$V1_STORE/models`、`$V1_STORE/cache/xdg`、`$V1_STORE/cache/hf` |
| paths.sh固定提交默认 | `JAX_COMPILATION_CACHE_DIR`、`HF_HUB_OFFLINE`、`UV_LINK_MODE` | 分别为`$V1_STORE/cache/jax`、`1`、`copy` |

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
set -euo pipefail
export RAW_H5_DIR=/scratch/hongze/robomme_data_h5
source scripts/dataset/paths.sh
export BUILD_HEAD=49a333eb18e8d6ff1143bf7871ef7c498ab91579
export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_CACHE_PATH="$V1_STORE/cache/cuda/orig80k-build" PYTHONDONTWRITEBYTECODE=1
unset XLA_FLAGS JAX_PLATFORMS
LIB="$V1_STORE/datasets/16task-pub-1600ep"
MANI="$LIB/meta/episode_manifest.json"
INMANI="$LIB/meta/input_manifest.json"
STATS="$V1_STORE/train-assets/mme_vla_suite/16task-pub-1600ep"
REFERENCE="$REPO_ROOT/docs/dataset-build-doc/16task-h5-scan/records"

env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/assets/fetch_assets.py verify --level full --assets siglip_params
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/scan_manifest.py build --raw_dir "$RAW_H5_DIR" --episodes-per-task 100 --num_shards 1 --out "$MANI"
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs --raw_dir "$RAW_H5_DIR" --out "$INMANI"
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/check_orig80k_sources.py --input-manifest "$INMANI" --reference-input "$REFERENCE/input_manifest.json" --manifest "$MANI" --reference-manifest "$REFERENCE/episode_manifest.json"
env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --no-sync python scripts/dataset/run_local.py --stage siglip --lib "$LIB" --gpus 0,1,2,3,4,5,6,7 --raw-dir "$RAW_H5_DIR" --require-free-mib 70000
env CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/dataset/finalize_checks.py check --manifest "$MANI" --out "$LIB/source" --raw_dir "$RAW_H5_DIR" --input_manifest "$INMANI" --input_level sha256 --spot_check 1024
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp" --procs 48
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --store "$LIB/framesamp" --resume --procs 48
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/compute_norm_stats.py --output-dir "$STATS" --config-name mme_vla_suite --repo-id robomme --dataset-path "$LIB/source"
```

第十个 `report` 是实际命令正文中的CPU内联Python：检查规模和packed状态，核对自算统计量形状与有限性，逐键比较原版文件，再以 `os.walk/stat` 累计source与packed的逻辑/分配字节，独占写出 `meta/norm_stats_comparison.json` 和 `meta/build_sizes.json`。这段报告代码没有被误称为Beta中的既有入口；其实际输出和SHA在本档案中保留，归档不复制shell/yaml载体。

生产代码与依赖由以下固定提交还原；其余入口同理：

```bash
git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:scripts/dataset/run_local.py
git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:scripts/dataset/finalize_checks.py
git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:scripts/dataset/pack_framesamp_store.py
git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:scripts/training/compute_norm_stats.py
git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:pyproject.toml
git show 49a333eb18e8d6ff1143bf7871ef7c498ab91579:uv.lock
```


## 实际会话、阶段退出与交接

| 项目 | 实际记录 |
|---|---|
| 构建会话 | `pub16-full-20260925T212238Z` |
| 包装PID | `3815822` |
| UTC起止 | 2026-09-25 21:23:38 → 2026-09-26 00:40:16 |
| 总跨度 | 11798秒，3小时16分38秒 |
| 原始日志 | `v1-store/logs/pub16-full-20260925T212238Z.build.log` |
| 任务/tee/整体终态 | `TASK_EXIT=0`、`TEE_EXIT=0`、唯一外层 `EXIT_CODE=0` |

| 阶段 | UTC开始→结束 | 秒级跨度 | 阶段退出 |
|---|---|---:|---:|
| asset_verify | 09-25 21:23:38→21:23:42 | 4 | 0 |
| scan1600 | 09-25 21:23:42→21:24:44 | 62 | 0 |
| hash16 | 09-25 21:24:44→21:44:40 | 1196 | 0 |
| source_pin | 09-25 21:44:40→21:44:40 | 0 | 0 |
| siglip | 09-25 21:44:40→09-26 00:27:27 | 9767 | 0 |
| finalize | 09-26 00:27:27→00:35:37 | 490 | 0 |
| pack | 09-26 00:35:37→00:37:46 | 129 | 0 |
| verify | 09-26 00:37:46→00:37:59 | 13 | 0 |
| norm_stats | 09-26 00:37:59→00:40:02 | 123 | 0 |
| report | 09-26 00:40:02→00:40:16 | 14 | 0 |


实际source pin为 `INPUT_PIN=PASS files=16`、`EPISODE_IDENTITY=PASS episodes=1600 timesteps=768897 samples=476857`。SigLIP8 worker完成1600集；finalize1024条最大绝对差0；全量packed verify768897行、零失配；source＋packed实测分配量706084982784 B。`FULL_DATA_BUILD=PASS`绑定[实际报告](records/build_sizes.json)，自算norm差异已[单独记录](records/norm_stats_comparison.json)，训练继续使用原版。

后续[counting构建](../4task-counting-pub-400ep/launch.md)在full结束后重新统计最大F/P及剩余预算，再开始独立构造，现已完成严格子集与count20。full20另在 `orig80k-full-read20-20260926T004046Z`（包装PID3915445）运行，2026-09-26 00:41:03→00:47:20 UTC整体PASS；20次更新、末步19、61个EMA叶实际恢复一致，见[独立训练档案](../../training-doc/smoke-orig80k-full-0925/launch.md)。

记录已归档于[十三节README](README.md)及[正式核验清单](records/archive_checks.json)，共27文件、1634943 B。本轮只按完整tmux名称精确检查，会话随任务自然退出，未执行tmux清理；本次没有删除源数据、库或权重。两库正式INPUT、P1、100步对拍、300步perf及80k均尚未通过；20步PASS不能代替它们。
