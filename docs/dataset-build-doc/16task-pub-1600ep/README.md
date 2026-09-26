# 公开16任务1600集正式库：十三节构建档案

> 本档案在构建与两次20步检查结束后回填实际结果。起跑前范围、参数和闸门由完整Beta `49a333eb18e8d6ff1143bf7871ef7c498ab91579` 锚定；实际启动与退出记录见 [launch](launch.md)，小型实测记录及清洗日志已归档于 [records核验清单](records/archive_checks.json)。

## 1. 结论与指标速览

**公开16任务×100集正式库的数据阶段及 full 库20步可读性、真实保存与恢复检查均已通过。** 数据阶段从 2026-09-25 21:23:38 UTC 到次日 00:40:16 UTC，10个阶段全部退出0，构建主体任务与正文tee返回也均为0；最外层退出记录的证据边界见第9节。随后独立的 `smoke-orig80k-full-0925` 更新20次，保存唯一末步 `19`，CPU真实恢复的61个EMA叶与本次末步摘要逐项一致。

| 指标 | 本次实测 |
|---|---:|
| 原始任务与episode | 16任务，各100集，共1600集 |
| 总帧数 | 768897 |
| 执行样本数 | 476857 |
| demo帧数 | 292040，由总帧减执行样本计算 |
| SigLIP | 8个worker，1600集，9767秒 |
| finalize复算 | 1024/1024条，最大绝对差0 |
| packed全量验证 | 768897行，失配0，`framesamp-4x4-v1` |
| source文件分配量 | 655588253696 B |
| packed文件分配量 | 50496729088 B |
| 两者文件分配量合计 | 706084982784 B，约657.593 GiB |
| full20更新与checkpoint | `state_step=20`，`loop_step=checkpoint_step=19` |
| 恢复后的EMA叶 | 61个：23个bfloat16、38个float32，全部有限且逐叶摘要相同 |

完整run按计划使用原版 `norm_stats.json`，本次full20也已实际使用该文件。自算统计量只记录差异，没有替换训练资产。20步结果覆盖当前代码、当前full库及一次保存恢复；正式 `INPUT_EQ`、上游100步对拍、单跑/并跑对拍、300步perf和80k均尚无本次通过结论。

## 2. 用户原话与范围

用户要求：「给出方案跑完整的80k 和80k纯counting任务 完全参照原版训练 4卡+4卡 先做对拍测试」。随后明确完整数据口径：「完整run跑公开 Yinpei/robomme_data_h5：16 任务 × 100 集 = 1600 集」，counting口径为「纯counting跑4*100 公开集合的子集合」。本阶段执行依据是最新原话：「恢复计划中的建库，严格按前置闸门推进」。

用户另批准「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」及「允许这四键缺失与 None 等价」。后一条仅适用于 `motion_emb/motion_pos/motion_mask/mem_order` 缺失与严格 `None` 的训练输入比较投影；其他键、非None值和signed zero要求不变，原始ALL字段、dtype/raw SHA仍留证。它不放宽源数据、packed或 full/counting 子集的字节守卫，也不代替正式输入与训练对拍。

本库使用同一公开全集，生成 source、4×4 packed及仅供比较的自算统计量；未生成8×8 packed、Wan或motion。后续依赖顺序遵循[原版80k计划](../../../0925-orig-80k-full-counting-4plus4-plan.md)。本档案确认full数据阶段和full20；同组counting数据、严格子集与count20的独立结果见[counting档案](../4task-counting-pub-400ep/README.md)。

## 3. 版本与代码状态

本组正式构建与full20共同使用 **`49a333eb18e8d6ff1143bf7871ef7c498ab91579`**，提交主题为 `commitV11.11Beta: 锚定两个公开正式库及20步可读性检查`，提交时间为 2026-09-25 21:21:09 UTC。full构建日志的 `BUILD_HEAD/BUILD_END_HEAD`、source的构建分片指纹、packer记录以及full20的launch/start/final/completion均指向该完整提交。启动与结束按相同HEAD和clean工作区执行硬检查；构建主体和read20主体的实际返回0包含这些检查，不以最末退出文本单独证明最外层进程状态。

来源前检与16集冒烟的起跑Beta为 `42b91cd96499bd51fcb6acaeedd642b368dbefeb`，其结果已在 `de6354fd7c0f78b347c7b07c4b8ee0c8b9b37418` 归档，详见[冒烟结果](../16task-pub-smoke16-0925/result.md)。P0环境起跑提交 `5489e4b3a92197d0e9a37421b1e6415b3022b613` 是更早的环境准备锚点，不能作为本次构建版本。

本组运行期间保持该Beta不变；本档案和正式records在两库及两次20步全部完成后回填。起跑前档案可由该Beta还原，结果回填不改变启动版本，也不将事后结果提交当成起跑锚点。

## 4. 原始来源与输入pin

输入实体根为 `/scratch/hongze/robomme_data_h5`，来自公开 `Yinpei/robomme_data_h5`。本轮重新hash全部16个H5，文件size合计 **512595968744 B**；`check_orig80k_sources.check_sources()` 将新输入清单的文件集合、size/SHA以及episode身份绑定到已提交的 `docs/dataset-build-doc/16task-h5-scan/records/` 参考。finalize又以 `--input_level sha256` 重核16文件，全部同源。

下表区分文件自身SHA与episode清单内部的规范内容SHA。分片安排不同可以改变清单内部SHA，不能把参考与本轮的整个JSON强求字节相同。

| 对象 | 本次核对的完整SHA256 |
|---|---|
| 已提交参考input文件 | `36c33bd6fdb012b72b692596c83982d2a8dc0f56e6ef7e45781ec9ebc954c640` |
| 已提交参考episode文件 | `43d9a6fbcef7eb259c12e74b7c62898d33961a6470c40586cbed1be51d1024c8` |
| 参考episode规范内容 | `34a259ea364bbc642b2b63ab22da4497056116196bede6cc09929a5b9548d5b2` |
| 本轮input文件 | `3ec2bf9a7df4bdd035476574a07ba2da243eb165f44a13ace16dd8be9abf4847` |
| 本轮episode文件 | `222d84f58fb82073603fda19992c4f82e542475294176bea99e8868036b8a11a` |
| 本轮episode规范内容 | `fb1bdbcbcc176d15475332b728fa218bb545b4a8c9dcaa9ce1ec9e0bb17f174f` |

参考文件随本次Beta可还原；新文件位于 `v1-store/datasets/16task-pub-1600ep/meta/`，当前存在且摘要已核对。来源判定原文为 `INPUT_PIN=PASS files=16`、`EPISODE_IDENTITY=PASS episodes=1600 timesteps=768897 samples=476857`。原始H5没有被本阶段删除或改作派生输出。

## 5. 任务、划分与样本口径

采用已提交来源档案所确认的公开train集合，每任务100集，本轮重新核验文件身份和episode清单，不另改划分。规范H5任务序为：BinFill、ButtonUnmask、ButtonUnmaskSwap、InsertPeg、MoveCube、PatternLock、PickHighlight、PickXtimes、RouteStick、StopCube、SwingXtimes、VideoPlaceButton、VideoPlaceOrder、VideoRepick、VideoUnmask、VideoUnmaskSwap；每个H5内部按raw episode序排列。

source为全部768897个时步保存feature，只为476857个执行时步保存pkl；执行样本按 `Σ(num_timesteps − exec_start_idx)` 计。episode全局编号与两个offset连续，最长episode为1411帧。最终 `source/meta/stats.json` 为 `execution_samples=476857`、`total_samples=768897`。总帧、执行样本和训练更新步数采用不同计数，不相互替代。

统计量脚本沿用 `compute_norm_stats.create_data_loader()` 的 batch128、完整批次和shuffle口径：`476857 // 128 = 3725` 批，实际纳入 **476800个执行样本**，余下57个不足完整batch的样本未参与这次统计。训练数据仍保留全部476857个执行样本，未为计算统计量删样本，也未修改该脚本的既有取法。

## 6. 启动命令与配置还原

完整构建实际在一个detached tmux会话 `pub16-full-20260925T212238Z` 内，通过 `run_stage()` 串行执行十个阶段。包装PID为 **3815822**，原始日志为 `v1-store/logs/pub16-full-20260925T212238Z.build.log`。运行时保存的命令正文位于 `v1-store/bench/orig80k-build-preflight-0925/full_build_command.txt`，SHA256为 `02e47a92d5753f2e649860bdd72a5bf74c2b256ee422ab93ef27f6b91b0182bb`；它是运行时保存的shell文本，不是Beta里新增的脚本，也不应作为独立脚本拷入正式档案。

下面还原该正文的实际环境与实质阶段命令。原运行另有新输出检查、预算检查、`set -euo pipefail`、逐阶段退出码记录及外层tee；现在这些输出已经存在，下列文本仅供还原，不能重放来覆盖它们。

实际环境口径如下。表中的“固定提交默认”来自 `scripts/dataset/paths.sh @ 49a333eb18e8d6ff1143bf7871ef7c498ab91579`，没有混称为运行命令手动覆盖；`run_local.base_env()` 在worker环境中再次设置的同值项也按该Beta源码还原。`LIB/MANI/INMANI/STATS/REFERENCE` 是命令正文的普通shell路径变量，见下方代码块。

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

## 7. 构建参数与格式

scan为每任务100集、`num_shards=1`；实际SigLIP通过 `run_local.worker_cmd()` 起8个动态worker，分别绑定GPU0–7，`--require-free-mib 70000`。manifest中的分片数1不代表实际只使用一张GPU。finalize使用GPU7复算1024条；pack/verify为CPU48进程，`reader=decode`，默认4×4布局，全量verify没有使用抽样参数。

source保留当前构造器输出的多尺度feature与执行pkl；packed仅保留4×4历史图像、按时刻共享的位置表和state表。实际有32个image part、1411个位置行，`manifest_scope=full`、`status=verified`，并绑定本次规范manifest SHA。逐行摘要覆盖768897行，`row_digests`为12302352 B，记录的SHA为 `dba2598148f3f29837220a204d1eafa725ae2a4efaed37a2b1e9f737e7f12013`。

本库自算统计量位于 `v1-store/train-assets/mme_vla_suite/16task-pub-1600ep/robomme/norm_stats.json`，SHA为 `c5d45b4bdb481ca46d18aeb32446bbedec34de435eef31d3212a179ee7868bcd`。完整run计划使用、本次full20已实际使用 `v1-store/train-assets/mme_vla_suite/robomme/norm_stats.json`，SHA为 `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`。两者不能混写成同一资产。

`norm_stats_comparison.json`记录的差异如下；相对差只在原版参考非零位置计算，原版零值另列。差异用于溯源，不作为替换原版训练统计量的理由。

| 统计字段 | 最大绝对差 | 非零参考项最大相对差 |
|---|---:|---:|
| actions.mean | 1.8000602722e-5 | 1.6414794159e-2 |
| actions.std | 5.3048133850e-6 | 7.0401098257e-5 |
| actions.q01 | 1.1207661510e-3 | 2.2115607851e-3 |
| actions.q99 | 1.1207661510e-3 | 2.7042943871e-3 |
| state.mean | 2.0742416382e-5 | 6.1455803445e-4 |
| state.std | 2.8282403946e-5 | 8.8267779703e-5 |
| state.q01 | 4.9409391880e-4 | 1.7277159329e-3 |
| state.q99 | 9.8818783760e-4 | 1.2526202505e-3 |

只有 `state.q01` 含1个参考零值，该位置最大绝对差为0；其余字段参考零值数量均为0。state/actions各字段形状均为8维，有限性、std非负和q01不大于q99均由本次report检查通过。

## 8. 硬件、存储、预算与耗时

环境为AWS单机8×`NVIDIA A100-SXM4-80GB`，底层存储为 `/dev/md0`、XFS、本地NVMe RAID。source分片及finalize记录JAX/jaxlib 0.5.3；packer记录Python 3.11.15、NumPy 1.26.4、ml_dtypes 0.4.1。8个worker的host相同，是同一台机器上的8个进程；旧字段中Slurm job均为空，不能写成集群构建。

full起跑日志实际记录：可用 **1784545525760 B**，门槛 **1458746144941 B**，`BUDGET_GATE=PASS`，依据是冒烟后尚未构建两正式库的预算。该门槛包含两库数据估算及10%余量、128 GiB缓存/验证临时/日志、300 GiB保留量；它不是full单库实测体积，也不是counting后续可直接照抄的固定门槛。

| 存储对象 | 逻辑字节 | 文件分配字节 | 文件数 |
|---|---:|---:|---:|
| 已有原始H5输入，按pin的size合计 | 512595968744 | 本阶段未测原始H5分配量 | 16 |
| 新建source | 652240332312 | 655588253696 | 1247364 |
| 新建packed及其meta | 50496720024 | 50496729088 | 37 |
| 新建source＋packed | 702737052336 | 706084982784 | 1247401 |

分配量是报告逐文件累加 `st_blocks*512` 的结果，不包含所有目录/inode开销；该表的两新目录也不包括库根其他manifest、worker日志或自算统计量。没有把既存H5再次计入新增预算。本次归档只读取这些已经完成的报告，没有重扫大数据。

构建总体墙钟跨度为 **11798秒，3小时16分38秒**。SigLIP实际9767秒；其余阶段耗时见下一节。full20外层总跨度377秒，其中runner阶段316秒、CPU恢复验收61秒；均包含相应初始化、检查和收尾，不能除以步数就当稳态步时。构建没有配套的稳态GPU利用率采样，本档案不据worker速率字段给出瓶颈或跨环境性能结论。

## 9. 构建过程与阶段日志

所有阶段位于实际会话 `pub16-full-20260925T212238Z`；其完整名称与包装PID3815822应保留，不能把预建launch里的会话模板当成实际实例。以下为日志直接记录的UTC边界；耗时取秒级边界差，source_pin显示0秒只说明它在同一秒完成。

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

资产full校验输出 `ASSETS=PASS assets=1 mismatches=0`，工具自身记录SigLIP校验3.82秒。SigLIP输出 `STAGE_DONE stage=siglip workers=8 items=1600 elapsed=9767s`。finalize确认feature目录缺失0、pkl实得476857/期望476857、8份sidecar覆盖1600集、残留claim为0；抽检1024/1024条全部最大差0。pack完成后经过独立verify，才形成verified交付状态。

关键成功判定保持原文：

```text
INPUT_PIN=PASS files=16
EPISODE_IDENTITY=PASS episodes=1600 timesteps=768897 samples=476857
STAGE_DONE stage=siglip workers=8 items=1600 elapsed=9767s
FINALIZE_EXIT_CODE=0
VERIFY_PACK=PASS scanned=768897 mismatches=0
NORM_STATS_COMPARE=RECORDED training_uses=original
BUILD_END_HEAD=49a333eb18e8d6ff1143bf7871ef7c498ab91579
TASK_EXIT=0
TEE_EXIT=0
EXIT_CODE=0
```

`FULL_DATA_BUILD=PASS` 后附完整报告JSON，已与现有 `meta/build_sizes.json` 一致。原始构建日志SHA256为 `9ac411726e04cc5ead58d2dc5a940956d7824a1e21d6ec8e36abd7766bbda6ef`。[构建清洗日志](records/logs/full-build.summary.log)已保留10条 `STAGE_EXIT_CODE`、来源判定、finalize/verify、报告和唯一外层退出记录；归档核验逐份确认关键行序列与原始日志一致。

补记[历史最外层退出记录审计](../../training-doc/orig80k-equiv-0925/exit-record-audit.md)：上述`TASK_EXIT/TEE_EXIT`记录构建主体和正文tee的真实返回码；末尾footer先写`EXIT_CODE=0`，再检查自身printf/tee，最后状态未独立持久化。因此日志末行0不能单独证明最外层包装实际退出0。目前没有证据表明历史footer失败，子阶段和数据内容PASS保留，原始records不改写。

## 10. 验收与内容一致性

数据阶段分别证实了：16个原H5与公开pin一致；1600集身份及规范顺序、样本offset和规模一致；source完整；1024条feature同架构复算的最大绝对差为0；packed经真实读取API与source对应4×4三键全量逐位对拍，768897行零遗漏、零失配。packed元数据文件SHA为 `a554a7859ba35c883ec805244c98764a9eb10c1ddbc244850c5f213db4da669b`，其 `source_provenance_sha256` 与实际source provenance文件 `42a852f89673939bb556887ecdaf51bc83285fb3cbdf0711ddd2b5c2bf827ff8` 相同。

这些检查的范围不同：finalize抽样复算不能称为全库每个H5→pkl字段的独立逐字节复核，packed验证也不能替代full/counting全覆盖子集对拍或上游训练loader对拍。

**独立full20已完成。** 会话为 `orig80k-full-read20-20260926T004046Z`，包装PID3915445、body PID3915450，真实训练记录PID3915984，run UUID为 `a56dd89a-ffbe-415d-9dd1-d5e205e14c25`，W&B run ID为 `770cbaaz`。外层从 2026-09-26 00:41:03 UTC 到00:47:20 UTC；训练driver在00:46:19 UTC结束，随后CPU完成器真实加载并验收。

本次模式为smoke，实际runtime为GPU0–3、device_count4、mesh `(batch=1, fsdp=4)`、global batch64、workers4、seed42、`jax_enable_x64=false`。只覆盖步数20；modul 512/4×4、原版学习率/warmup/EMA以及log100/save10000/keep10000不变，W&B保持开启。启动契约实际绑定本库manifest文件SHA `222d84f58fb82073603fda19992c4f82e542475294176bea99e8868036b8a11a`、上述packed元数据SHA及原版norm SHA。

核心实际调用如下；完整环境和argv保存在run的 `launch.json` 与已保存命令正文中。启动时清除CPU平台变量，没有通过设置 `JAX_ENABLE_X64=0` 掩盖继承值；runner按实际x64状态验收。

```bash
env -u JAX_PLATFORMS -u JAX_PLATFORM_NAME \
  TRAIN_HEAD=49a333eb18e8d6ff1143bf7871ef7c498ab91579 \
  HISTORY_CONFIG_SHA256=823c3948e75a9335ace3f250d0255e6a8618e8ecf0bb77c65af65077349d199a \
  NORM_STATS_SHA256=f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5 \
  bash scripts/training/prod/run_orig80k.sh smoke smoke-orig80k-full-0925 0,1,2,3 \
  /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-pub-1600ep \
  /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite

env -u JAX_PLATFORM_NAME CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  UV_CACHE_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/uv \
  XDG_CACHE_HOME=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/xdg \
  HF_HOME=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/hf \
  OPENPI_DATA_HOME=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/models \
  uv run --no-sync python scripts/training/tests/check_orig80k_completion.py \
  --mode smoke --run smoke-orig80k-full-0925 --head 49a333eb18e8d6ff1143bf7871ef7c498ab91579 \
  --records /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k/smoke-orig80k-full-0925 \
  --run-root /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-runs/mme_vla_suite/smoke-orig80k-full-0925 \
  --log /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/smoke-orig80k-full-0925.driver.log \
  --out /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k/smoke-orig80k-full-0925/completion.json
```

普通指标恰为step0一条、五个标量；末步记录完整保留tail的step1…19，五标量全有限。最终 `state_step=20`，`loop_step=checkpoint_step=19`，唯一checkpoint为 `19`，异步保存等待完成记录与run UUID、HEAD一致。`completion.json.restored_leaves` 与 `final.json.ema_leaves` 对全部61个叶的dtype、shape、字节数、原始SHA及有限性记录逐项相同；归档前已用小JSON独立核对两份字典相等，没有再次读取大权重。

```text
TRAIN_PIPE_EXIT=0
TEE_EXIT=0
EXIT_CODE=0
RUN_COMPLETED=PASS run=smoke-orig80k-full-0925 checkpoints=1 final=19
COMPLETION_TASK_EXIT=0
COMPLETION_TEE_EXIT=0
WRAPPER_TASK_EXIT=0
WRAPPER_TEE_EXIT=0
READABILITY20=PASS
```

driver日志只有自己的唯一 `EXIT_CODE=0`，整个runner的真实返回0另由wrapper捕获。wrapper转录driver输出后，在末尾记录预先计算的 `EXIT_CODE=0` 与 `READABILITY20=PASS`；其footer自身的证据边界同第9节，两类终态不能混合计数。恢复结果JSON SHA256为 `340f8b53543041c36456f8c4aebacf6182139711caf5a3121ca8b307c248f897`。

checkpoint `19` 的控制器已记录最终逻辑字节 **11879851680 B**、文件分配字节 **11879907328 B，约11.064 GiB**；本档案引用该小型记录，没有重新遍历权重目录。原生 `_CHECKPOINT_METADATA` 文件SHA `bd70f6fff3fbc2ae5144b081ae34d247ffa49fbf06f32aeaf5f5f1deb02187fa` 已与completion文件清单核对；其初始化到提交差为14491721657 ns，即 **14.491722秒**（保留六位小数）。这是本次真实保存提交区间，不能替代保存期间磁盘占用峰值；峰值尚未测量。

GPU采样成功覆盖记录窗口，每卡519条窗口内记录，最大相邻间隔约0.508秒、均值约0.50025秒，采样错误文件为空。这证明本次采样覆盖有效，不构成稳态利用率、吞吐或300步perf结论。

## 11. 计划外事件与处置

full构建十阶段、full20 driver、CPU完成器和相应tee均退出0，当前证据没有本阶段构建或恢复失败。uv关于 `tool.uv.dev-dependencies` 的弃用提示保留在原始日志，未据此修改依赖或同步正在使用的环境。

起跑前launch按阶段模板说明会话纪律；实际full使用一个已记录tmux串行执行全部阶段，full20使用另一个tmux。现有launch已回填真实会话和阶段边界，没有将十个串行阶段记成十个会话。

自算norm与原版不同是预先约定的记录项，已按协议保存差异并继续使用原版。full20最终checkpoint大小已记录，但没有保存磁盘峰值采样；该缺口保留，不能以最终大小或14.491722秒提交耗时补造峰值。本阶段未调整训练算法、超参或精度来换取成功。

## 12. 当前结论与下一步

本组已完成两库数据交付、counting全覆盖严格子集对拍及两次20步真实保存恢复。full使用原版norm，counting使用本库独立norm，各自的清单、资产摘要与完成记录独立绑定。counting及其20步结果见[独立档案](../4task-counting-pub-400ep/README.md)。

两库正式 `INPUT_EQ`、P1、100步上游与同入口单跑/并跑对拍、300步perf及80k均尚未通过。已有三样本诊断、主机dtype/四motion键许可和本次20步检查不替代这些闸门。P1/100步对拍驱动目前关闭W&B，是否仅允许对拍关闭仍待用户答复；20步、perf与80k的W&B保持开启。perf所需保存期间磁盘峰值证据也尚未采集，是否增加0.5秒只读checkpoint目录分配量与scratch可用量采样，或暂停perf，仍待用户决定。没有用最终checkpoint大小代替峰值，也未自行放宽要求。

本次没有清理原始H5、full库、checkpoint或缓存。后续临时产物清理须在归档核验与归属确认后按既有授权处理，不删改其他run。

## 13. 归档文件清单

本目录已归档 **27个文件，1634943 B**，包括26个payload和1个正式 `archive_checks.json`；README与launch不计入该文件数。正式[核验清单](records/archive_checks.json) SHA256为 **`1ade435d24e0c8c04de1d5894a88a09280458febc745583d8cda65d216c97128`**，登记原始来源、暂存核验、正式目标、逐文件SHA和bytes。它是正式目标的清单，未把先前ignored暂存清单的SHA当成正式清单SHA。

| 已归档内容 | 正式位置 |
|---|---|
| 1600集episode及16文件输入pin | [episode_manifest.json](records/episode_manifest.json)、[input_manifest.json](records/input_manifest.json) |
| 实际规模、体积与自算/原版统计差异 | [build_sizes.json](records/build_sizes.json)、[norm_stats_comparison.json](records/norm_stats_comparison.json) |
| source统计与来源指纹 | [source_stats.json](records/source_stats.json)、[source_provenance.json](records/source_provenance.json) |
| 8个worker分片指纹 | [source_shards/](records/source_shards/) 中的 `_shard0of8.json`…`_shard7of8.json` |
| packed元数据与打包进展 | [packed_store_meta.json](records/packed_store_meta.json)、[pack_progress.jsonl](records/pack_progress.jsonl) |
| 本库自算统计量，仅供比较 | [self_norm_stats.json](records/self_norm_stats.json) |
| 全构建及8个worker清洗日志 | [full-build.summary.log](records/logs/full-build.summary.log)、[logs/](records/logs/) 中8份 `siglip-gpu*.summary.log` |
| 复制、日志清洗与正式目标核验 | [archive_checks.json](records/archive_checks.json) |

普通记录逐字节复制，源SHA复核不变；构建与8个worker日志先拆CR，再去掉 `%|` 进度态，保留完成、失败、关键指标和退出记录。正式落档另采用仅去行尾空格/Tab的规范化，核验全部行在相同规范化后逐行相等；本组数据日志实际没有因此改变字节。凭据检查未发现匹配。每份日志清洗前后行数、删除数、关键行一致性和时间均在核验清单中，而非只以PASS字符串代替清洗验收。

关键原始记录SHA如下；这些普通JSON与正式归档字节相同，清洗日志的新SHA以正式核验清单为准。

| 文件 | SHA256 |
|---|---|
| full `meta/build_sizes.json` | `a9a6b9772c51979b36face7fb85603a118801409e457421e3ffc53b7b6e1d9ef` |
| full `meta/norm_stats_comparison.json` | `91c1bd37db7be145592c762b5d825fb5b2778066b11f744eee3f6f2af52dce35` |
| full `source/meta/stats.json` | `f48b9c92d9c7eaa389490130be85889ef2b6d7b94a9691c72061767d6d3df57a` |
| full `source/meta/provenance.json` | `42a852f89673939bb556887ecdaf51bc83285fb3cbdf0711ddd2b5c2bf827ff8` |
| full `framesamp/meta/store_meta.json` | `a554a7859ba35c883ec805244c98764a9eb10c1ddbc244850c5f213db4da669b` |

full20的launch/runtime、末步摘要、恢复结果、GPU采样、清洗driver/wrapper日志和小型checkpoint元数据归入[独立训练档案](../../training-doc/smoke-orig80k-full-0925/launch.md)，不混入本构建目录的27文件。对应[训练归档核验清单](../../training-doc/smoke-orig80k-full-0925/records/archive_checks.json)保存源与正式目标绑定。

已提交的参考pin、配置、脚本和yaml通过Beta还原，不重复归档；运行时 `.command.txt` 与 `full_build_command.txt` 虽然后缀是txt，本质仍是bash正文，没有作为独立脚本副本搬入docs。必要实际命令保留在正文代码块。H5、features、packed数组、checkpoint权重与缓存留在现有scratch实体目录。
