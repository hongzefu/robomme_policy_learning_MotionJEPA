# 1027-orig16-vla-512-and-stage2-motion-4plus4-plan.md — 用原版 RoboMME 16×100 重建 VLA 训练库，4+4 卡并行训 512（8 帧 8×8）与 512+stage2 motion（1024 单 token）两条 80k

> **裁决状态**：草案，等开工令。本文只是计划，不是代码实施、建库、下载、训练或上传的开工令（`AGENTS.md` 第 2 条）。按用户长期裁决「计划 md 放仓库根目录并合进工作分支」写成 Markdown；计划将提交到新分支 `v5-orig16`（主会话从 `b25fd99` 切出并合并本文）。
>
> **用户原话（2026-10-27，逐字）**：
> ①「上次的训练应该错误了。你的这个 motion encoder 的这个训练是用了官方数据集来训练的，所以说你 policy 仓库也应该用官方的训练。」
> ②「写完整方案 我需要同时跑512 8帧8*8 和 512+motion」
> ③「加入一个强约束，以后 encoder 和 VI 必须是同一个数据集，不可以出现混用的情况。」（VI 即 VLA）
> 既往长期裁决沿用、不再询问：bucket 名由我们自定；计划 md 放仓库根目录并合进工作分支；对用户报时一律美东；所有训练产物上传 HF；长任务 detached tmux + Monitor；写入型子代理独立 worktree + `sub/` 提交 + sonnet 审查。
>
> **代码锚点**：policy `v4-256concat1@b25fd992f8ed4df74726888a452d87d33a28c86e`（`commitV11.56`；其后 `256b70b`/`7dd98dd`/`6d2f1d5` 只有 handoff 与 70k 收尾记录），工作副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，本计划及全部实施落新分支 `v5-orig16`；MotionJEPA `v12-3decoder@6c79ebb3d9c2f8a6a45dfcd012954af688c962b0`，抽取 CLI 在 MJ worktree `/scratch/hongze/worktrees/vla-s2-memexport-20261026`（分支 `v12-3decoder-memexport`）。
>
> **外部依赖锚点**：encoder ckpt `/scratch/hongze/MotionJEPA/runs/stage2-full144-1025-a/checkpoint_epoch_105.pt`（EMA 在 `model` 键）sha256 `27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035`；源 H5 `/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/source/h5/record_dataset_<Task>.h5`（16 个，约 512 GB），上游 `Yinpei/robomme_data_h5@a5e4e25ffe8af34f64944f9533d06455ce5f8337`，MJ `dataset/source_manifest.json` 的 `pin_sha256`（下称 `<pin8>` 取其前 8 位，开工第一步实读回填；编写时只读预读值为 `9e109419…`，未作开工复核）；Wan VAE `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`；G2 对拍 oracle 为 MJ 已公开的 stride-1 库 `HongzeFu/robomme-original-16x100-wan21-front-33f-dense-v1@384d5db410be713a49357bf3f55ca4f4d2f559a0`（MJ `logs/public-pipeline-20261023.log` 末行 `PUBLIC_PIPELINE=COMPLETE`）。
>
> **commit 体例**：功能改动 `commitV<x>.<y>: <中文>`，接续 `git log` 最近号（当前 `commitV11.56`）；文档 `docs:`，交接 `handoff:`；子代理 `sub/<编号>: <中文>`，经主会话 `git merge --no-ff <TIP sha>` 合入。

# 第一部分（给人看）

## 一、为什么重做

**错配事实**：阶段 III motion encoder（MotionJEPA run `stage2-full144-1025-a` 的 ep105 EMA）训练根是 `/scratch/hongze/dataset-rgb-gtxyz-16views-stride2-v1`（MJ `configs/stage2_pool4.yaml` 的 `data_root`），该缓存的 `control/build_spec.json` 的 `source_pin.files[]` 逐文件列出 16 个原版 H5：`/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/source/h5/record_dataset_<Task>.h5`（原版 RoboMME 16 任务 × 100 集 = 1600 集，每任务 easy 50／medium 25／hard 25，无 xhard；总帧 768,897，demo 帧 292,040，exec 样本 476,857；9 个任务有 video demo：InsertPeg、MoveCube、PatternLock、RouteStick、VideoPlaceButton、VideoPlaceOrder、VideoRepick、VideoUnmask、VideoUnmaskSwap；集长 48–1411 帧；encoder 验证集 = 每任务 ep94–99）。而 10-26 的 VLA run `mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k` 用的是 policy 仓 `v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f`（本地录制 mixed8-200，16×200 = 3200 集，含 xhard）。两库是同一任务参数的不同批次录制，**不是子集关系**。1026 计划风险登记第一行把它记成「encoder 域外，只能由 VLA 结果判断」而没有拦下，用户裁定这是错误（原话①）。该 run 已按用户指令在 70k 停止（SIGTERM，`EXIT_CODE=143`），交付 70000 步，HF 导出在 tmux `cc-vla-s2-ckptup` 进行中（日志 `v1-store/logs/vla-s2-20261026/close70-export.log`，期望 `CKPT_UPLOAD=PASS steps=14 ...`）；其结果此后**仅作历史记录**。

**强约束（原话③）落地成红线 + 机械闸门，不是风险项**：

| 项 | 内容 |
|---|---|
| 红线 R-LINEAGE | motion encoder 的训练源数据集与 VLA 训练库必须是同一数据集：同一组源 H5、逐文件 sha256 相同。任何混用一律 FAIL 停工，不得 WAIVE，不得登记成「已知风险」继续。项目规则见 `AGENTS.md` 项目专属规则 P-LINEAGE（由另两个子代理在 policy 与 MotionJEPA 两仓落地，本计划只引用）。 |
| 闸门 `DATA_LINEAGE` | 新脚本 `scripts/training/check_data_lineage.py`：encoder ckpt 旁 `config.yaml` 的 `data.data_root` → `<data_root>/control/build_spec.json` 的 `source_pin.files[]`（逐文件 sha256）↔ MJ `dataset/source_manifest.json`（`pin_sha256` + 逐文件 sha）交叉核对 → 与 VLA 库 `meta/input_manifest.json` 的 `files{}` 逐文件比较，集合必须相等。判定行 `DATA_LINEAGE=PASS encoder_ckpt=27daa832 encoder_source_pin=<pin_sha256> vla_input_pin=<sha> files=16 mismatches=0`。 |
| 接入点 | 阶段 0 对参考清单预检（L0）、建库 hash 后正式核（L1）、motion run 启动契约必检项 `CHECK_ENCODER_DATA_LINEAGE`（L2）；两条 run 同批起跑前 `PAIRED_RUNS_SAME_LIB=PASS`（P1）。 |

**已核实的同源性**：MJ 副本 16 个 H5 与 policy 仓 `docs/dataset-build-doc/16task-h5-scan/records/input_manifest.json` 登记的 sha256 逐文件相同（16/16），即与 2026-09 从 `Yinpei/robomme_data_h5` 下载、后来已删的 `/scratch/hongze/robomme_data_h5` 是同一份数据（例：BinFill `1b3a7139…`，40,186,950,704 B）。**不必重新下载**，raw 目录直接用 MJ 副本路径、只读、不复制。policy 仓 2026-09-25 曾用同一份数据建过 `16task-pub-1600ep`（source + 4×4，SigLIP 8 worker 9767 s），所以本次不需要 mixed8 的 fetch/merge 阶段，直接从 H5 建 manifest。

**泄漏说明（信息，不是问题）**：encoder 验证集 ep94–99 落在 VLA 训练集内；encoder 冻结、不参与 VLA 梯度，不构成泄漏。VLA 的 sim 评估按 RoboMME 官方 test/val seed（test 560000、val 1060000 起），与训练集无交集。

## 二、要做什么与运行一览

**一句话**：从原版 16×100 H5 照搬 mixed8 建库链（去掉 fetch/merge）建新库 `16task-orig100-1600ep-<pin8>`（SigLIP source、4×4、8×8、自算 norm_stats、Wan grid16 latent），在新库上用 ep105 EMA encoder 重抽 stage2 1024 单 token motion 库，然后 4+4 卡并行训两条 80k：512（8 帧 8×8 modulation，无 motion）与 512+stage2 motion（与 10-26 run 同一 motion 机制，只换库）。

| 运行 | run 名 | GPU | 配置 | 步数 | 预计时长 | bucket（公开） |
|---|---|---|---|---|---|---|
| 建库 | 库 `16task-orig100-1600ep-<pin8>` | SigLIP／Wan 0–7 | 照搬 mixed8 链 | — | 约 5 h（估算） | `HongzeFu/robomme-16task-orig100-1600ep-trainset-20261027-v1` |
| 512 | `orig100-m8x8-modul-b128-80k` | 0–3 | `mme_vla_suite_b128_80k`，HC `perceptual-framesamp-modul-8frame-8x8.yaml`，FSDP 4、workers 8 | 80,000 | 1.7183 s/步 × 80k ≈ 38.2 h（10-17 实测速率外推） | `HongzeFu/robomme-vla-orig100-modul-512-80k-v1` |
| 512+motion | `orig100-m8x8-modul-motion-stage2-1024-b128-80k` | 4–7 | 同上 + 新 yaml `perceptual-framesamp-modul-8frame-8x8-motion-stage2-orig16.yaml`（budget 96、stride 32、`[1,1024]`） | 80,000 | ≤1.75 s/步 × 80k ≈ 39 h（估算） | `HongzeFu/robomme-vla-orig100-modul-motion-stage2-1024-80k-v1` |
| MJ ckpt | 阶段 II＋III | — | 已 PASS | — | 不重传 | `HongzeFu/motionjepa-stage2-pool4-3decoder-1025-v1`（G10） |

两条并行总墙钟 ≈ 39 h；若串行 8 卡各约 21 h（8 卡 0.96 s/步实测外推）≈ 43 h，所以按用户原话②并行。tmux 前缀一律 `o16-`；运行脚本与日志在 `v1-store/logs/o16-20261027/`。

## 三、数据：原版库规格、与 mixed8 的差异、目录与磁盘

**库名**：`16task-orig100-1600ep-<pin8>`，`<pin8>` = MJ `dataset/source_manifest.json` 的 `pin_sha256` 前 8 位（体例同 mixed8 取源 MANIFEST sha 前 8 位）。库根 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-orig100-1600ep-<pin8>`，norm_stats `v1-store/train-assets/mme_vla_suite/16task-orig100-1600ep-<pin8>/robomme/norm_stats.json`。

| 项 | mixed8（10-16 库） | 原版（本次） |
|---|---|---|
| 来源 | 本地录制 mixed8-200，HF bucket 100 个 tar.xz → fetch → `merge_v2_h5.py` | `Yinpei/robomme_data_h5@a5e4e25f`，MJ 副本按任务一个 H5、`episode_<N>` 分组，直接 `scan_manifest.py build` |
| 集数 | 16×200 = 3200（含 xhard 档） | 16×100 = 1600（easy 50／medium 25／hard 25） |
| 帧／exec 样本 | 2,145,548 ／ 1,383,846 | 768,897 ／ 476,857 |
| 带 demo 的集 | 1800 | 9 个任务（集数以 manifest 为准） |
| source/data | 547,443,716,927 B | ≈ 189 GB（估算，按样本线性折算） |
| framesamp（4×4） | 140.8 GB | ≈ 50 GB（估算） |
| framesamp-8x8 | 562.9 GB | ≈ 202 GB（估算） |
| wan-latents（grid16） | 75.8 GB，128,342 窗 | ≈ 27 GB、约 4.6 万窗（估算） |
| motion-stage2-1024（grid32） | 65,419 行 | 约 2.3 万行（估算，G5 实测为准） |
| encoder 同源 | 否（错配） | 是（`DATA_LINEAGE` 机械核） |

**H5 内部结构（已核实）**：顶层 100 个 `episode_<N>`；每组 `setup/{difficulty, seed, task_goal(2,), front_camera_intrinsic, ...}` 与 `timestep_0…timestep_{T-1}`；每 timestep 有 `obs/front_rgb (256,256,3) uint8`、`obs/front_depth`、`obs/joint_state (7,)`、`obs/eef_state (6,)`、`obs/gripper_state (2,)`、`action/{eef_action(7,), joint_action(8,), ...}`、`info/{is_video_demo, is_completed, is_subgoal_boundary, ...}`。这正是 `scan_manifest.py build`／`build_shard.py`（SigLIP 建库）直接消费的形态。

**磁盘（2026-10-27 美东约 12:40 实测）**：`/scratch` 6.9 T 总、5.7 T 已用、**1.3 T 可用**；GPU 0–7 全空。

| 对象 | 大小 | 处置 |
|---|---|---|
| `dataset-rgb-gtxyz-16views-stride2-v1`（encoder 训练缓存，HF 上传 `HF_VERIFY=FAIL`） | 2.8 T | 不可删 |
| `dataset-original-robomme-16x100-wan21-v1`（本计划的源） | 564 G | 不可删、只读 |
| `v1-store/restore/mixed8-200-vla-latents-20261021-a`（来自 PASS bucket 的恢复副本） | 71 G | 阶段 0 删（第一档） |
| 10-26 导出 `v1-store/exports/hf-ckpt-mixed8-200-vla-stage2-1024-80k` | 156 G | `cc-vla-s2-ckptup` 打出 `CKPT_UPLOAD=PASS steps=14` 后删（第一档） |
| 10-26 run 本地 ckpt | 155 G | 同上判定行后删（第一档） |
| mixed8 VLA 库（HF `RESULT=PASS`，但 `motion-stage2-1024/` 261 M 与 `motion-tokens-stage2/` 363 M 未上传） | 1.2 T | **默认保留**；只有建库前可用 <1.5 T 时作第二档，先补传两子目录并回读、经用户裁决才删 |
| `dataset-local-a100-16task-mixed8-200`（mixed8 原帧，备份未核实） | 324 G | 不动 |

需求估算：库 ≈ 190 + 50 + 202 + 27 ≈ 470 GB（features 剪枝后），建库峰值另加 features ≈ 465 GB（pub16 实测量级，SigLIP 后、剪枝前），两条 run ckpt 各 ≈ 190 GB（16 份），JAX cache／日志 ≈ 50 GB，G2 oracle 抽样下载 ≤ 15 GB → 峰值约 1.0–1.2 TB，训练期稳态约 0.9 TB。第一档删除后可用约 1.7 T。每阶段起跑前打 `DF_CHECK` 判定行，阈值见第二部分三节 B0。

## 四、memory 抽取链在原版库上的差异点

抽取口径与 1026 完全相同：窗 33 帧（锚点 + 未来 32）、`GRID_STRIDE` 32、`segment_start`、`forward`；demo 末窗真实帧 ≥17 保留并 `repeat_last` 补到 33、≤16 丢弃；exec 全真实；每窗 4×256 按源时间片 [0,3,5,8] 拼 1 个 1024 维 token；budget 96；抽取 `eval()` + `no_grad()` + fp32；memory 不增强；encoder = ep105 EMA。差异只有以下五处：

| 差异 | 原因（已核代码） | 本次做法 |
|---|---|---|
| Wan latent 需重新编码 | 1026 用的是 mixed8 恢复副本；原版库没有现成的 grid32 整 33 帧补帧 latent。MJ 的 stride-1 库（658,350 窗、不补帧、exec 段 `[D,C)` 排除首个 `is_completed` 帧）**本机只剩 index 与 metadata**（`dataset/` 4.3 MB，`wan_chunk_latents/` 已不在本机，全量在 HF repo），且口径不同，只能作对拍 oracle | 用 policy 自己的 `run_local.py --stage wan`（`scripts/dataset/wan/extract_wan.py`）在新库上编码 |
| grid32 还是 grid16 | `extract_wan.py` 的 argparse 只有 `--manifest/--raw-dir/--out/--worker/--segments/--max-items`，步长取模块常量 `wan_common.GRID_STRIDE = 16`，**没有 grid 参数**；而 MJ 抽取 CLI（`--grid-subsample`，默认 2）与 `pack_motion_store.py pack --layout stage2`（强制 `--latent-grid-subsample 2`）都按「grid16 取偶数格」实现 | **路径 B（默认）**：沿用 grid16 编码、下游 `--grid-subsample 2`，零改 Wan 代码，代价是 Wan 多算一倍（约 2.3 h，估算）。路径 A：给 `wan_common`/`extract_wan` 加 grid 32 参数，并同步改 MJ 抽取、pack、metadata 合同。判据：路径 A 要动 ≥4 个文件且改 stage2 来源 schema 的 `source_grid` 语义，超出同等规模，不采用 |
| G0 读帧来源 | `check_stage2_validation_inputs.py` 的 demo 帧从 mixed8 `data-raw/<Task>_ep<N>/video_demo.h5` 读、并依赖 `control/scope.json`；原版没有 data-raw 形态 | 新增 `--h5-root` 模式：demo 原帧直接读源 H5 `episode_<N>/timestep_<t>/obs/front_rgb`，身份绑定 `meta/input_manifest.json`；exec 仍读 `source/data/<sample>.pkl` |
| G2 对拍对象 | 1026 的 G2 对 encoder 训练缓存（fp16）FAIL 后被 WAIVED；本次不允许 WAIVE | G2 改对 MJ stride-1 库同起点窗；**数值口径冲突（以代码为准）**：MJ 库 `provenance.json` 记 fp32 权重、batch 1、`cudnn_conv_tf32: true`、`matmul_tf32: false`；policy `wan_motion_infer.py::pin_numerics` 钉 `cudnn.allow_tf32 = False`。故 G2 判据定为「用 policy 同一 `encode_chunk` 在 MJ provenance 数值模式下重编抽样窗，须与 MJ bin 逐位相等」，生产 latent（TF32 关）与 MJ bin 的差另出信息行，不作放宽 |
| 来源链 | `pack_motion_store.py::cmd_pack` 的 stage2 分支强制 `--restore-receipt`（mixed8 恢复副本收据），`motion_store.py`、`motion_client.py`、`train.py` 的 stage2_compat 都校验 `restore_receipt`；MJ 抽取 CLI `FULL_COUNTS = (4995, 65419)`、`compare_stage2.py` 写死 `num_rows != 65419` | 新增「fresh-extract」来源模式（mixed8 旧路径逐字节不变）；计数改为由 manifest 推导或显式参数 |

## 五、训练：4+4 配置、速度、epoch 口径

**512 run**：runner 照搬 `scripts/dataset/hf_export/mixed8_200_runner/train-512-80k.sh` 的结构，但 `_train_common.sh` 把库名写死为 `NAME=16task-mixed8-200-3200ep-e3c7564f`（`DS`、`MMEVLA_FRAMESAMP_SOURCE`、`MMEVLA_FRAMESAMP_MANIFEST`、`ASSETS_DIR` 全由它派生），所以新建平行目录 `scripts/dataset/hf_export/orig100_runner/`，只改库名，不改 mixed8 旧文件。CLI 必须带 `--dataset-path $DS/framesamp-8x8 --assets-base-dir ... --data.assets.assets-dir v1-store/train-assets/mme_vla_suite/16task-orig100-1600ep-<pin8> --data.assets.asset-id robomme`（config 默认 assets 指向 4task，漏写会静默读错 norm stats）。启动：`CUDA_VISIBLE_DEVICES=0,1,2,3 FSDP_DEVICES=4 NUM_WORKERS=8 bash scripts/dataset/hf_export/orig100_runner/train-512-80k.sh orig100-m8x8-modul-b128-80k`；先过 `scripts/training/preflight_train_launch.py`（`CHECK_*` 与 `PREFLIGHT=PASS n=`，不传 `--launch-mode`）。

**512+motion run**：入口 `scripts/training/run_mixed8_stage2_1024_80k.sh` 写死 `HC`、`PARENT`、`ASSETS_DIR`、正式 run 名、`CUDA_VISIBLE_DEVICES=0,…,7` 与短测前缀 `cc-vla-s2-smoke-`；契约 `scripts/training/check_mixed8_stage2_1024_launch.py` 写死 `HISTORY_CONFIG`/`FRAME_CONFIG`、`Contract` 的 `parent_rel/assets_rel/norm_sha256/episode_manifest_sha256/input_manifest_sha256/canonical_manifest_sha256/motion_rows=65419/train_max=62/eval_max=79/fsdp_devices=8/num_workers=16/gpus=0..7`、`CFG_RUN_NAME`，且 `check_config` 要求 `fsdp_devices` 与基础配置相等（基础配置为 8），`VALUE_FLAGS` 不含 `--fsdp-devices`/`--num-workers`。方案：**新建**平行文件 `scripts/training/run_orig16_stage2_1024_80k.sh`、`scripts/training/check_orig16_stage2_1024_launch.py`、`src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion-stage2-orig16.yaml`；契约里 GPU 改 `4,5,6,7`、FSDP 4、workers 8，并新增必检项 `CHECK_ENCODER_DATA_LINEAGE` 与「G2 证据 status 必须 PASS、不得 WAIVED」。

**速度（10-17 4+4 先例实测）**：稳态 512 1.7183 s/步、motion 1.7346 s/步（38 h 15 min／38 h 37 min），短测峰值显存 39.3／38.3 GiB（limit 72）；8 卡单跑 stage2 motion 0.96 s/步。本次 motion 为单 token、budget 96、memory 608，4 卡估 ≤1.75 s/步、80k ≈ 39 h（估算，G7 短测更新）。

**epoch 口径提醒（请用户看见）**：原版库 exec 样本 476,857，b128 × 80k = 10.24 M 样本 ≈ **21.5 个 epoch**；对照 mixed8 3200 集 ≈ 7.4 个 epoch、4task-v2-1600ep（605,611 样本）80k ≈ 16.9 epoch。60k ≈ 16.1 epoch，40k ≈ 10.7 epoch。默认沿用 80k（用户要求并行跑同样两条），是否改 60k/40k 列入第八节裁决。

## 六、验收闸门总表

| 编号 | 判定行 | 查什么／过了说明什么 |
|---|---|---|
| B0 | `DF_CHECK=PASS stage=<s> avail_bytes=<n> need_bytes=<n>` | 每阶段起跑前磁盘够用；不够即停 |
| L0 | `DATA_LINEAGE=PASS encoder_ckpt=27daa832 encoder_source_pin=<pin_sha256> vla_input_pin=<sha> files=16 mismatches=0`（对参考清单） | 开工前确认 encoder 源 = 参考 16 文件 |
| B1 | `INPUT_PIN=PASS files=16`、`EPISODE_IDENTITY=PASS episodes=1600 timesteps=768897 …` | `check_orig80k_sources.py`：新 input_manifest 与参考逐文件、逐 episode 同一 |
| L1 | `DATA_LINEAGE=PASS …`（对新库 `meta/input_manifest.json`） | 新库输入 = encoder 训练源，R-LINEAGE 机械成立 |
| B2 | `STAGE_DONE stage=siglip workers=32 items=1600 elapsed=<s>` | SigLIP 全覆盖 |
| B3 | `全部检查通过 PASS` | finalize 1024 帧复算与输入 sha256 级核验 |
| B4 | `VERIFY_PACK=PASS scanned=768897 mismatches=0`（4×4） | 4×4 打包逐帧一致 |
| B5 | `VERIFY_PACK=PASS scanned=768897 mismatches=0`（8×8）、`IMAGE_NPY_SPOT=PASS frames=512 mismatches=0`、`MOTION_POS_XGRID=PASS …` | 8×8 打包与跨档位置一致 |
| B6 | `STAGE_DONE stage=wan workers=8 items=<n> …` | Wan grid16 全段完成 |
| B7 | `ORACLE_VAE=DONE … frame_mismatches=0`、`WAN_BITEXACT=PASS …` | policy Wan 与 MJ 环境独立 oracle 逐位一致（pack 前置） |
| B8 | `FEATURES_PRUNE=PASS kept=8` | features 只留 8 个抽样 |
| B9 | `DATA_IDENTITY=PASS norm_sha=<…> steps=768897 samples=476857 manifest_sha=<…>`（主会话手工行） | 库身份四方绑定 |
| G0 | `RAW_WINDOWS_READY=PASS windows=64 tasks=16 demo_tail=<≥16> sha_mismatches=0` | 抽样原帧（改读源 H5）与 latent 输入 SHA 一致 |
| G1 | `DROPOUT_OFF=PASS windows=64 eval_repeat_max_abs=0 train_vs_eval_max_abs=<正数>` | dropout 真关 |
| G2 | `WAN_EQUIV=PASS windows=64 oracle=mj-stride1@384d5db4 numerics=mj-provenance max_abs=0`（另出 `WAN_NUMERICS_DRIFT=INFO …`） | 同帧、同窗、同 VAE、同预处理；**不允许 WAIVE** |
| G3 | `EMA_SOURCE=PASS leaves=65 state_key=model exact=1 live_negative_rejected=1`、`EXPORT_PARITY=PASS windows=32 seed=0 max_abs=0 geometry_all_true=1` | 真加载 EMA、导出路径数值一致 |
| G4 | `ONLINE_OFFLINE_PARITY=PASS windows=64 handshake=1 response_bytes=4096 encoder_max_abs=0 e2e_max_abs=<x> …` | 在线 sidecar 与离线库一致（x≤1e-5） |
| G5 | `PACKED_TOKEN_BYTES=PASS compared=<rows> mismatches=0`、`ENCODER_SAMPLE_BITEXACT=PASS compared=1024 total=<rows> mismatches=0 seed=0`、`VERIFY_MOTION=PASS scanned=<rows> …` | motion 库完整、补帧正确 |
| G6 | `TRAIN_ES_BOUND=PASS train_max=<n> eval_max=<n> budget=96` | 训练与 1300 步评估零截断 |
| L2／P1 | `CHECK_ENCODER_DATA_LINEAGE=PASS`（契约内）、`PAIRED_RUNS_SAME_LIB=PASS lib=16task-orig100-1600ep-<pin8> runs=2` | 起跑时再核血缘；两条 run 同一父库 |
| G7 | `LAUNCH_CONTRACT=PASS checks=<n> failed=0`、`PREFLIGHT=PASS n=<n>`、`SMOKE_4GPU=PASS run=512 …`、`SMOKE_4GPU=PASS run=s2 …`、`SAVE_LOAD=PASS …` | 两条均能启动、20 步有限、motion 能保存恢复 |
| G8 | 两份训练日志 `EXIT_CODE=0` | 80k 训完，16 份 ckpt |
| G9 | `CKPT_UPLOAD=PASS steps=16 …` × 2 | 远端全量匿名回读 sha256 一致 |
| G10 | `MJ_CKPT_UPLOAD=PASS source_runs=2 …`（10-26 已 PASS） | 引用，不重传 |
| G11 | 不使用（编号保留，与 1026 对齐） | — |
| G12 | `check_stage2_checkpoint.py --mode delivery` 判定行（step 79999） | motion run 交付 ckpt 严格恢复 + 真实在线推理 |
| B10 | 训练库上传 `RESULT=PASS`（长尾） | 新库公开备份 |
| S0 | `SOURCE_RO=PASS files=16 size_mtime_unchanged=16` | 收尾确认源 H5 未被触碰 |

## 七、步骤与时间线（T0 = 开工令时刻，均为美东相对时刻；耗时为估算）

| 阶段 | 内容 | 预计耗时 | 相对时刻 | 判据 |
|---|---|---|---|---|
| 0 磁盘与句柄 | 开工审查 `PLAN_CONSISTENCY_REVIEW`；等 `CKPT_UPLOAD=PASS steps=14`；实读 `<pin8>` 回填；L0；按判定行删第一档三项；状态机登记 | 0.5–1 h（不含等 ckptup） | T0 → T0+1 h | L0、B0、`CKPT_UPLOAD=PASS steps=14` |
| 1 代码改动与审查 | S1–S6 并行开发（独立 worktree）、逐块 sonnet 合前审、`--no-ff` 合入、合后审、push；并行跑不依赖新代码的 CPU 前置：`scan_manifest`、`hash-inputs`、`check_orig80k_sources`、L1 | 5–6 h | T0+1 → T0+7 h | `PRE/POST_MERGE_REVIEW=PASS`、B1、L1 |
| 2 建库（含 Wan） | 建库 Beta、HOLD 冻结 → SigLIP 32 worker → finalize → Wan grid16（GPU 0–7）∥ 4×4／norm／8×8／xgrid（CPU）→ oracle vae → 剪枝 features | 约 4.5 h | T0+7 → T0+11.5 h | B2–B9 |
| 3 memory 抽取与闸门 | G0 → MJ 抽取（G1）→ G2（HF 抽样 bin）→ G3 → pack → compare_stage2 sampled → verify（冻结结束）→ G4 → G6 → 回填契约实测常量 commit | 约 2 h | T0+11.5 → T0+13.5 h | G0–G6 |
| 4 短测 | 两条 20 步并行（0–3、4–7）→ motion 保存恢复 | 约 1 h | T0+13.5 → T0+14.5 h | G7、P1、L2 |
| 5 正式 4+4 与增量上传 | 训练 Beta、只读锁、两条同时起跑；两路增量上传；训练库长尾上传 | 约 39 h | T0+15 → T0+54 h | G8、`INC_UPLOADED` |
| 6 收尾 | 两条最终导出 + 全量回读（G9）、motion 79999 严格恢复（G12）、档案、恢复只读锁、S0、handoff | 约 4–8 h | T0+54 → T0+62 h | G9、G12、`PERM_RESTORED=OK`、S0 |

## 八、需用户裁决项（括号内为默认）

1. **步数**：两条是否仍训 80k（默认 80k，≈21.5 epoch）；可改 60k（≈16.1）或 40k（≈10.7），改动落在启动脚本覆盖还是全局配置需按 `AGENTS.md` 第 10 条确认。
2. **mixed8 VLA 库（1.2 T）**：是否删除（默认保留；只有建库前可用 <1.5 T 时提请，先补传 `motion-stage2-1024/` 与 `motion-tokens-stage2/` 并回读）。
3. **是否等 `cc-vla-s2-ckptup` 打出 `CKPT_UPLOAD=PASS steps=14` 再开工**（默认等）。
4. **norm_stats 用新库自算**（默认是，与 mixed8 做法一致）。
5. **G2 oracle 抽样下载**：从公开 repo `HongzeFu/robomme-original-16x100-wan21-front-33f-dense-v1@384d5db4` 只下 G2 样本所在的段 bin 与 index（默认同意，上限 15 GB，落 `v1-store/validation/orig16-g2/mj-oracle/`）。
6. **生产 Wan 数值口径**：保持 policy 钉死的 `cudnn.allow_tf32=False`（默认，与在线 sidecar 一致；G2 改用 MJ 数值模式重编证明同源），还是改成 MJ 库的 conv TF32 开（会破坏 G4 在线一致）。

## 九、停点清单

- `CKPT_UPLOAD` 非 PASS 或 tmux 消失无完成行：不删 10-26 产物，交用户。
- 任一 `DF_CHECK` 低于阈值：停，提请第二档。
- `DATA_LINEAGE` 任一次 FAIL 或缺行：立即停工，不得 WAIVE（R-LINEAGE）。
- 任一 B／G 闸门 FAIL：停，留证交用户，不放宽阈值；G2 FAIL 不得走 `accept_stage2_g2_exception.py`。
- 合前审两轮仍 FAIL、合后审 FAIL：写 `AGENTMETARULES_HOLD`，交用户。
- 训练崩溃、NaN、OOM：停下问用户，不 resume。
- G2 下载需要超出 15 GB，或任何未列入的下载：先问。
- push 被拒：停，原文交用户。

## 十、子代理分工与合并（简述）

拆六块写入型（全部 `opus` + 独立 worktree）：S1 管 G0 改读 H5 与新血缘检查脚本；S2 管 stage2 来源链的「fresh-extract」模式（pack／motion_store／motion_client／train.py／compare_stage2 计数参数化）；S3 管 G2 新对拍脚本；S4 管两条 run 的启动入口、契约与新 yaml；S5 管三套上传驱动；S6 在 MotionJEPA 抽取 CLI 去掉 mixed8 计数硬编码。合并顺序 S6 → S1 → S2 → S3 → S4 → S5：每块合前派一个 `sonnet` 只读审查钉死 BASE／TIP，合后主会话跑定向测试与范围核对再 push。运行另派 `opus` 运行代理在主检出启动、交回句柄，不改代码；探索用 `haiku` 只读。逐文件写集、验收命令见第二部分二节。

# 第二部分（技术细节，供 agent 追踪）

## 〇、前置声明与红线

- **R0 R-LINEAGE**：encoder 训练源与 VLA 库必须同一组源 H5、逐文件 sha256 相同；`DATA_LINEAGE` 在 L0／L1／L2 三处机械核，任何 FAIL 立即停工，不得 WAIVE、不得写成风险项继续（`AGENTS.md` P-LINEAGE；`CLAUDE.md` 项目专属补充「P-LINEAGE 的 Claude Code 落地」：主会话派发 VLA 建库／起跑子代理前先贴 `DATA_LINEAGE` 判定行，审查代理对涉及起跑脚本／契约的改动必须核到该闸门存在且 FAIL 即停）。
- **R1 源目录只读**：`/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/` 整体不写不动（MJ 1024 计划 R6）；policy 侧只以 `--raw_dir`/`--raw-dir` 只读引用 `source/h5/`，不复制、不建 symlink 进 `v1-store/`；收尾以 build_spec 记录的 `bytes`/`mtime_ns` 核 S0。
- **R2 不改 encoder 权重**：只读 `checkpoint_epoch_105.pt` 的 `model` 键，strict、`eval()`、`no_grad()`、autocast 关、两个 TF32 开关关（与 1026 R2 相同）。
- **R3 memory 不增强**：`geometry_valid` 全 True，等价 aug_id=0。
- **R4 旧口径逐字节不变**：mixed8 的入口、契约、yaml、导出驱动、runner 目录零 diff；`motion_store`/`motion_client`/`pack_motion_store`/`train.py` 中的 mixed8 restore-receipt 路径与旧 layout 逐字节不变，由回归测试盯住。
- **R5 只在新分支**：policy 只在 `v5-orig16` 及其 `worktree-agent-*` 子分支；MotionJEPA 只在 `v12-3decoder-memexport` 新增提交，不动 `v12-3decoder`/`main`。
- **R6 建库冻结窗口**：建库 Beta 后，从 SigLIP 起跑到 `VERIFY_MOTION=PASS`，policy 不 commit、不合并、不回流规则（SigLIP 各 shard 的 fingerprint 与 `pack_motion_store.py` 都要求 `git_commit` 唯一）；主会话写 `$(git rev-parse --git-dir)/AGENTMETARULES_HOLD`（`freeze` + 锚点 sha），窗口结束删除。
- **R7 训练期代码冻结**：从训练 Beta 起到两条 G8 与最终导出完成，`src/`、`pyproject.toml`、`uv.lock` 不动；被跟踪的 `src/ scripts/ packages/` 文件 `chmod a-w`，原权限记到 `v1-store/o16-scripts/chmod-before.txt`，结束按记录恢复并打 `PERM_RESTORED=OK`。
- **R8 tmux**：只用前缀 `o16-`，完整名单见四节；只按清单 `tmux kill-session -t '=名'` 逐个清理，删前删后 `tmux ls`，禁止 `kill-server` 等全局杀法。
- **R9 不 resume**：训练崩溃、NaN、OOM 一律停下问用户；不带 `--force`、不 `overwrite`；新 run 目录、新库目录、新 bucket 起跑前确认不存在。
- **R10 W&B**：hongzefu 账号，entity `hongzefu-university-of-michigan`，project `robomme-framesamp`；凭据 `v1-store/secrets/wandb.env`、`v1-store/secrets/hf.env` 只经 `set -a; . file; set +a` 读，不进日志、档案、命令行。
- **R11 存储边界**：一切持久化落 `/scratch/hongze/` 下；缓存 `UV_CACHE_DIR=/scratch/hongze/.cache/uv`，HF 缓存沿用 `paths.sh` 的 `$V1_STORE/cache/hf`（mixed8 oracle 首跑因覆盖 `HF_HOME` 失败的教训，env 不覆盖 `HF_HOME`）。
- **R12 删除纪律**：第一档删除只在对应 HF 判定行原文核到后由主会话执行，删前列清单、`ls -ld` 确认实体目录，显式列目录不 glob；第二档需用户裁决。
- **R13 不 WAIVE**：G2 及一切闸门无例外通道；`scripts/dataset/wan_stage2/accept_stage2_g2_exception.py` 本计划禁用，契约核 G2 证据 `status == "PASS"`。
- **R14 子代理环境**：worktree 内只跑 ≤5 分钟 CPU 测试，policy 用 `UV_PROJECT_ENVIRONMENT=/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv PYTHONPATH=<worktree>/src uv run --no-sync python -m pytest <定向> -q`（先打印 `mme_vla_suite.__file__` 确认指向 worktree）；MJ 用 `UV_PROJECT_ENVIRONMENT=/scratch/hongze/MotionJEPA/.venv PYTHONPATH=<worktree>/src uv run --no-sync python -m pytest <定向> -q`。GPU 验收合并后由运行代理在主检出跑。

## 一、逐文件改动清单

所有锚点用符号名（`AGENTS.md` 第 9 条），以 `b25fd99` 为准。

### A. MotionJEPA（S6，分支 `v12-3decoder-memexport`，worktree 内新提交）

| 文件 | 锚点 | 改什么 | 验收 |
|---|---|---|---|
| `scripts/stage2_memory_export.py` | 常量 `FULL_COUNTS = (4995, 65419)`、`main` 中 `require_equal(counts, FULL_COUNTS, "mixed8 全表段数与行数")` | 新增 `--expected-counts <segments>,<rows>`（必填于 full scope）；不传时报错而不是回落 mixed8 常量；mixed8 用 `4995,65419` 调用时行为逐字节不变 | `tests/test_stage2_memory_export.py` 新增：错计数拒绝、正确计数通过、smoke scope 不受影响；`STAGE2_EXPORT=PASS scope=smoke …` |

### B. policy

**S1 `scripts/dataset/check_stage2_validation_inputs.py`**（G0 改读 H5）
- 新增 `--h5-root <dir>` 与 `--input-manifest <LIB/meta/input_manifest.json>`；给出 `--h5-root` 时不读 `control/scope.json`/`run_manifest.json`，demo 帧改为 `h5py.File(<h5-root>/<ep.h5_file>, "r")[f"episode_{raw_ep_idx}/timestep_{t}/obs/front_rgb"]`，H5 文件 sha256 必须等于 input_manifest 登记值（首次读取时全量算并缓存，或只读引用 `hash-inputs` 已算值——实施时按代码确认最省且仍「不信文件信现场」的写法）；exec 路径不变（`source/data/<sample>.pkl`）。
- 模块常量 `GRID_STRIDE = 16` 不变（路径 B 下 G0 仍按 grid16 旧格选窗）；`select_windows` 的 64 窗／16 任务／≥16 demo 尾窗合同不变。
- 不给 `--h5-root` 时 mixed8 行为逐字节不变。测试 `tests/dataset/test_check_stage2_validation_h5.py`（新）：合成 2 任务 H5 夹具，demo 补帧窗 SHA 与按定义计算一致、H5 sha 不符拒绝、mixed8 默认分支对夹具输出不变。
- 判定行沿用 `RAW_WINDOWS_READY=PASS windows=64 tasks=16 demo_tail=<n> sha_mismatches=0`。

**S1 `scripts/training/check_data_lineage.py`（新）**
- 子命令 `lineage --encoder-ckpt <pt> --expected-ckpt-sha256 <64hex> --vla-input-manifest <json> [--mj-source-manifest /scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/source_manifest.json]`：
  1. 现场算 ckpt sha256 并与期望比较；
  2. 读 ckpt 同目录 `config.yaml` 的 `data.data_root`（ep105 实为 `/scratch/hongze/dataset-rgb-gtxyz-16views-stride2-v1`）；若 ckpt `fingerprint` 内另有 data_root 字段则一并比较（实施时按 MJ 代码确认，读 ckpt 需 torch 时改在 MJ 环境只读 `config.yaml` 足够）；
  3. 读 `<data_root>/control/build_spec.json` 的 `source_pin.files[]`（每项 `task/path/bytes/sha256`），得 encoder 源集合；
  4. 读 MJ `source_manifest.json` 的 `files.<Task>.h5.{file,sha256,size}` 与 `pin_sha256`，要求与 3 完全相同（链条：`stage2_pool4.yaml` data_root → 1024 计划 R6 源目录 → source_manifest）；
  5. 读 VLA `input_manifest.json` 的 `files{<name>:{size,sha256}}`，按文件名 `record_dataset_<Task>.h5` 对齐，集合必须相等（多、少、sha 不同、size 不同都计 mismatch）；
  6. `vla_input_pin` = 按文件名排序的 `[name, size, sha256]` 列表的 canonical JSON sha256。
  输出 `DATA_LINEAGE=PASS encoder_ckpt=<sha8> encoder_source_pin=<pin_sha256> vla_input_pin=<sha> files=16 mismatches=0`，否则 `DATA_LINEAGE=FAIL … mismatches=<k>` 并逐项列出、退出码非 0。
- 子命令 `paired --argv-a <file> --argv-b <file>`：两份文件是两条 run 的最终 `TRAIN_CMD=` 行（runner `DRY_RUN=1` 输出或契约打印），要求 `--dataset-path` 的父目录、`--data.assets.assets-dir`、以及 motion yaml `store_path` 的父库（若有）全部相同；输出 `PAIRED_RUNS_SAME_LIB=PASS lib=<库名> runs=2`。
- 提供可导入函数 `check_lineage(...) -> (ok, detail)` 供 S4 契约调用。测试 `tests/training/test_check_data_lineage.py`（新）：合成 build_spec／source_manifest／input_manifest 三方一致 PASS；改一个 sha、少一个文件、多一个文件、mixed8 式 100 文件清单均 FAIL；paired 父库不同 FAIL。

**S2 来源链 fresh-extract 模式**
- `scripts/dataset/pack_motion_store.py`：`cmd_pack` 中 stage2 分支「必须 `--restore-receipt`」改为二选一：`--restore-receipt`（历史 mixed8，原逻辑不变）或新 `--latents-origin fresh`（latent 由本库 `run_local.py --stage wan` 现场编码）。fresh 模式下 VAE 身份取 `wan-latents/metadata.json` 记录的当前 `scripts/dataset/wan/SOURCE_PIN.json` 与 VAE 权重 sha，不经 `load_vae_compat`；`check_raw_sources` 仍要求 `oracle/wan-mj/vae_report.json`（B7）与 input_manifest、Wan metadata、manifest 的 `raw_dir` 四方一致；`--expected-ckpt-sha256` 在 fresh 模式同样必填。`gather_provenance` 的 stage2 分支写 `stage2_compat.latent_origin = {"kind": "fresh-extract", ...}`，证据键由 `restore_receipt` 换成 `wan_extract`（指向 `wan-latents/metadata.json` 与 B7 oracle 报告的 sha）。
- `src/mme_vla_suite/datastore/motion_store.py`：stage2_compat 校验（现 `require_sha256(compat.get("restore_receipt_sha256"), …)` 与 `for gate in ("G0", "G2", "G3", "restore_receipt", "SOURCE_PIN")`）按 `latent_origin.kind` 分支；mixed8（无 `latent_origin` 或 kind=historical-restore-receipt）逐字节不变；fresh 分支要求 `G0/G2/G3/wan_extract/SOURCE_PIN` 五项且 G2 `status == "PASS"`。
- `src/mme_vla_suite/policies/motion_client.py`：`check_provenance` 的 stage2 证据循环（同上五键）同样按 kind 分支；fresh 模式不读 restore receipt，VAE 当前实现即在线实现，无历史→当前映射。
- `scripts/training/train.py::init_history_config`：stage2 快照按 kind 保存对应证据集（`EVIDENCE_FILES` 的键集合随 kind 变化），`motion_snapshot.ready.json` 机制不变。
- `scripts/dataset/wan_stage2/compare_stage2.py`：写死的 `meta.num_rows != 65419` 改为 `--expect-rows <n>`（必填），mixed8 调用传 65419 行为不变。
- 测试：`tests/dataset/test_pack_motion_stage2_fresh.py`（新）两非空段夹具跑 pack(fresh) → compare 记录 → verify → loader；`tests/training/test_stage2_snapshot.py`、`tests/policies/test_motion_client_stage2.py` 增 fresh 分支用例；既有 mixed8 用例原样通过。
- 实施提示：`restore_receipt` 的全部引用点以 `grep -rn restore_receipt src scripts --include=*.py` 为准（当前命中 `motion_store.py`、`motion_client.py`、`pack_motion_store.py`、`train.py` 及 mixed8 专用脚本），只改前四个。

**S3 `scripts/dataset/wan_stage2/orig16_g2.py`（新，torch 子环境 `scripts/dataset/wan_stage2` 运行）**
- `select --manifest <MANI> --latents <LIB/wan-latents> --mj-index <oracle>/index --out <VALID>/G2-windows.json`：seed 0，每任务 4 个**全真实**窗（demo 段至少覆盖 9 个有 demo 的任务各 1 窗；exec 窗要求 `start+33 ≤ C`，C 为 MJ `segments.parquet` 记录的首个 `is_completed` 帧，即只选 MJ 库实际存在的窗），把 policy 的（段, grid16 旧格 m）映射到 MJ 的（task, episode, segment, start），并列出所需 bin 文件名清单 `G2-bins.txt`；policy 段起点与 MJ `lo` 的对应关系（demo `[0,es)` ↔ `[0,D)`、exec 起点 `es` ↔ `D`）逐窗断言，不符即 FAIL。
- `compare --windows <VALID>/G2-windows.json --mj-oracle <oracle> --raw-dir <RAW> --vae-snapshot <…> --out <VALID>/G2.json`：① 逐 bin 核 MJ `SHA256SUMS`；② 用 `wan_motion_infer.py::encode_chunk`（与生产同一函数）在 **MJ provenance 数值模式**（fp32、batch 1、`cudnn.allow_tf32=True`、`matmul.allow_tf32=False`、benchmark 关）下从源 H5 原帧重编 64 窗，与 MJ bin 对应 block 逐位比较 → `WAN_EQUIV=PASS windows=64 oracle=mj-stride1@384d5db4 numerics=mj-provenance max_abs=0`；③ 同时读本库生产 latent（policy 钉死模式）对应 block 与 MJ bin 比较，只输出 `WAN_NUMERICS_DRIFT=INFO windows=64 norm_mse=<x> max_abs=<y>`，不作 PASS/FAIL、不放宽 ②；④ 环境（torch/cuDNN/driver/GPU）与 MJ `provenance.json.numeric_mode` 不符时直接 FAIL（同型号 A100、torch 2.9.0+cu128 是逐位前提）。
- 产物 `G2.json` 的 `status` 只能是 PASS／FAIL；`validate_stage2.py` 与 `accept_stage2_g2_exception.py` 不改、不用。测试 `scripts/dataset/wan_stage2/tests/test_orig16_g2.py`（新，CPU）：映射与 `start+33 ≤ C` 过滤、bin 清单去重、SHA256SUMS 不符拒绝。

**S4 训练入口、契约、yaml**
- 新 yaml `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion-stage2-orig16.yaml`：motion 节之前与 `perceptual-framesamp-modul-8frame-8x8-motion-stage2-mixed8.yaml` 逐字节相同；motion 节只改 `store_path: v1-store/datasets/16task-orig100-1600ep-<pin8>/motion-stage2-1024` 与 `online_gpu: 1`（训练不读该键；与 G7/G12 sidecar `--online-gpu 1` 及 `policy_config.py` 缺省值一致，评估部署可在调用时覆盖）；`checkpoint_sha256` 直接写 `27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035`。
- 新 `scripts/training/run_orig16_stage2_1024_80k.sh`：复制 `run_mixed8_stage2_1024_80k.sh`，改 `HC`、`PARENT`、`ASSETS_DIR` 到 orig 库；正式 run 名 `orig100-m8x8-modul-motion-stage2-1024-b128-80k`；短测前缀 `o16-smoke-s2-`；`CUDA_VISIBLE_DEVICES=4,5,6,7`；`TRAIN_ARGS` 增 `--fsdp-devices 4 --num-workers 8`；`MMEVLA_FRAMESAMP_SOURCE`/`_MANIFEST` 指向新库；调用新契约。
- 新 `scripts/training/check_orig16_stage2_1024_launch.py`：复制 mixed8 契约，改 `HISTORY_CONFIG`（新 yaml）、`FRAME_CONFIG` 不变；`Contract` 字段 `parent_rel/assets_rel` 指向新库，`norm_sha256/episode_manifest_sha256/input_manifest_sha256/canonical_manifest_sha256/motion_rows/train_max/eval_max` 先写占位 `__FILL_AFTER_BUILD__`（占位即 FAIL），阶段 3 由主会话单独 commit 回填实测值；`fsdp_devices=4`、`num_workers=8`、`gpus="4,5,6,7"`；`check_config` 对 fsdp/workers 改为「等于契约值」而不再要求等于基础配置（batch/seed/ema 仍三方相等）；`VALUE_FLAGS` 加 `--fsdp-devices`、`--num-workers` 并要求各恰出现一次且值为 4／8；`CFG_RUN_NAME` 改新 run 名与 `o16-smoke-s2-` 前缀；`MOTION_FIXED` 的 `store_path`/`online_gpu` 随新 yaml；新增 `CHECK_ENCODER_DATA_LINEAGE`（调用 S1 `check_lineage`，VLA 侧取 `parent/meta/input_manifest.json`）与 `CHECK_G2_NOT_WAIVED`（store_meta 中 G2 证据 status 为 PASS）；末行仍 `LAUNCH_CONTRACT=PASS checks=<n> failed=0`。
- 新目录 `scripts/dataset/hf_export/orig100_runner/`：`_train_common.sh`（复制 mixed8 版，`NAME=16task-orig100-1600ep-<pin8>`，其余不变）与 `train-512-80k.sh`（同 mixed8 版，`HC=perceptual-framesamp-modul-8frame-8x8.yaml`）。
- 测试 `tests/training/test_check_orig16_stage2_launch.py`（新）：合成 Contract 跑通；占位、错 GPU、缺 `--fsdp-devices`、WAIVED 的 G2、血缘 mismatch 均 FAIL；`bash -n` 三个新 sh；`DRY_RUN=1` 打印的 512 `TRAIN_CMD=` 含 orig 库三处路径。

**S5 上传驱动**（全部新建，mixed8 驱动零改动）
- 训练库：`scripts/dataset/hf_export/run_orig100_trainset_export.sh`（复制 `run_mixed8_200_trainset_export.sh`，`NAME`、`BUCKET_ID=HongzeFu/robomme-16task-orig100-1600ep-trainset-20261027-v1`）、`orig100_trainset_bucket_README.md`、`hygiene_allowlist_orig100_trainset.json`；上传范围含 `motion-stage2-1024/`、`motion-tokens-stage2/`、`oracle/`（吸取 mixed8 两子目录漏传教训，驱动显式列入并在回读清单里计数）。
- 512：`_orig100_vla_ckpt_export_common.sh`（复制 mixed8 common，`NAME` 换 orig 库，`EXPECT_MOTION=false` 路径）、`run_orig100_vla_512_80k_ckpt_export.sh`（`RUN_NAME=orig100-m8x8-modul-b128-80k`、`BUCKET_ID=HongzeFu/robomme-vla-orig100-modul-512-80k-v1`、`EXPORT_NAME=hf-ckpt-orig100-vla-512-80k`）、`orig100_vla_512_80k_bucket_README.md`、`hygiene_allowlist_orig100_vla_512_80k.json`。
- stage2：`_stage2_orig16_vla_ckpt_export_common.sh`（复制 `_stage2_vla_ckpt_export_common.sh`，常量 `RUN`、`EXPORT_NAME=hf-ckpt-orig100-vla-stage2-1024-80k`、`VLA_BUCKET=HongzeFu/robomme-vla-orig100-modul-motion-stage2-1024-80k-v1`、`DATASET`、`HISTORY` 换新值；`EVIDENCE_FILES` 键集合按 S2 fresh 模式 `G0/G2/G3/wan_extract/SOURCE_PIN`；删去 70k 停止专用常量；`STEPS` 保持 5000…75000 + 79999）、`run_orig16_stage2_1024_ckpt_inc.sh`（训练期增量）、`run_orig16_vla_stage2_1024_80k_ckpt_export.sh`（最终）、`orig16_stage2_vla_bucket_README.md`、`hygiene_allowlist_orig16_stage2_vla.json`。
- 512 训练期增量沿用 10-17 的非 git 脚本 `v1-store/m8v-scripts/m8v-ckpt-inc.sh`（参数 `<RUN> <BUCKET> <EXPORT> <ALLOWLIST> <训练 tmux>`，与正式驱动共用 `logs/uploaded/` 标记根），由主会话复制为 `v1-store/o16-scripts/o16-ckpt-inc.sh`，内容不改。
- 测试 `scripts/dataset/hf_export/test_orig16_export_drivers.py`（新）：三套驱动常量与 run/bucket/库名一致、`bash -n`、`SMOKE=1` 合同夹具；mixed8 驱动文件对 BASE 零 diff。

## 二、子代理分配表

`BASE` = 主会话合入本计划后的 `v5-orig16` HEAD（开工时记录 40 位 sha）；MJ 侧 BASE = `v12-3decoder-memexport` 当前 HEAD。所有写入型 `model: "opus"` + `isolation: "worktree"`；审查 `model: "sonnet"` 只读；探索 `model: "haiku"` 只读；运行型 `model: "opus"`、主检出、不加 worktree、不改代码。

| 编号 | 目标 | 可写集合 | 禁触 | 依赖／接口契约 | 合并顺序 | 验收命令与判定行（环境） | 资源 |
|---|---|---|---|---|---|---|---|
| E1（haiku） | 只读核实：`check_orig80k_sources.py` 是否比较 `raw_dir`；`check_stage2_checkpoint.py --mode delivery --delivery-step` 是否支持 79999；MJ 抽取 CLI 读 policy wan-latents metadata 的字段要求 | 无 | 全部写入 | 结论交主会话，作 S2/S4/S6 输入 | — | 报告原文摘录 | CPU |
| S6（opus，MJ worktree） | MJ 抽取计数参数化 | `scripts/stage2_memory_export.py`、`tests/test_stage2_memory_export.py` | 其余 MJ 文件、`runs/` | — | 1，主会话 push MJ 分支 | MJ R14 环境 pytest；`STAGE2_EXPORT=PASS scope=smoke` 夹具 | CPU |
| S1（opus） | G0 改读 H5；血缘与配对检查 | `scripts/dataset/check_stage2_validation_inputs.py`、`scripts/training/check_data_lineage.py`（新）、`tests/dataset/test_check_stage2_validation_h5.py`（新）、`tests/training/test_check_data_lineage.py`（新） | R4 所列旧文件；源目录 | 输出 `check_lineage` 函数签名供 S4 | 2 | policy R14 环境 pytest 两新测试；对真实只读文件跑 `lineage`（参考清单）出 L0 行 | CPU |
| S2（opus） | fresh-extract 来源链 | `scripts/dataset/pack_motion_store.py`、`src/mme_vla_suite/datastore/motion_store.py`、`src/mme_vla_suite/policies/motion_client.py`、`scripts/training/train.py`、`scripts/dataset/wan_stage2/compare_stage2.py`、`tests/dataset/test_pack_motion_stage2_fresh.py`（新）、`tests/training/test_stage2_snapshot.py`、`tests/policies/test_motion_client_stage2.py` | mixed8 restore 路径语义；`scripts/dataset/wan/**` | 证据键集合 `G0/G2/G3/wan_extract/SOURCE_PIN` 供 S4/S5 | 3 | 新旧 stage2 测试全过；旧 datastore 定向回归 | CPU |
| S3（opus） | G2 新对拍 | `scripts/dataset/wan_stage2/orig16_g2.py`（新）、`scripts/dataset/wan_stage2/tests/test_orig16_g2.py`（新） | `validate_stage2.py`、`accept_stage2_g2_exception.py`、`scripts/dataset/wan/**` | G2.json schema 供 S2 pack 读取 | 4 | wan_stage2 子环境 CPU pytest | CPU |
| S4（opus） | 两条 run 入口、契约、yaml、512 runner | 一节 S4 所列 3 个新文件 + `scripts/dataset/hf_export/orig100_runner/**`（新）+ `tests/training/test_check_orig16_stage2_launch.py`（新） | mixed8 入口／契约／yaml／runner | 依赖 S1 `check_lineage`、S2 证据键 | 5 | pytest；`bash -n`；`DRY_RUN=1` 打印 | CPU |
| S5（opus） | 三套上传驱动 | 一节 S5 所列新文件 + `scripts/dataset/hf_export/test_orig16_export_drivers.py`（新） | mixed8 驱动、`ckpt_upload_common.py`、`scan_hygiene.py` | 依赖 S2 证据键 | 6 | pytest；`bash -n`；`SMOKE=1` 只本地 | CPU |
| RV-*（sonnet） | 每块合前只读审查 | 无 | 全部写入 | 钉死 `REVIEW_BASE`/`REVIEW_TIP`；涉及起跑脚本／契约的改动必须核到 `DATA_LINEAGE` 闸门存在且 FAIL 即停，缺失即 FAIL | 每块合前 | `PRE_MERGE_REVIEW=PASS\|FAIL base=<sha> tip=<sha> files=<n> commits=<n> findings=<n>` | 无 |
| X1（opus 运行） | 阶段 2 建库各段 | 只写新库、norm 目录、`v1-store/logs/o16-20261027/` | 代码、源 H5 | 四节 runbook 2.x | 阶段 2 | 首个判定行或 STAGE 起跑行 | GPU 0–7／CPU |
| X2（opus 运行） | 阶段 3 抽取与闸门 | `motion-tokens-stage2/`、`motion-stage2-1024/`、`v1-store/validation/orig16-*`、日志 | 代码 | runbook 3.x | 阶段 3 | G0–G6 各行 | GPU 0（G4 sidecar GPU 1） |
| X3（opus 运行） | 阶段 4–5 短测、正式、增量上传 | 两 run 目录、exports、日志 | 代码、旧 run | runbook 4.x–5.x | 阶段 4–5 | 第 1 步 loss 有限、`INC_UPLOADED` | GPU 0–3、4–7 |
| X4（opus 运行） | 阶段 6 最终导出、G12 | exports、validation、日志 | 代码 | runbook 6.x | 阶段 6 | G9、G12 | CPU／GPU 0–1 |
| 主会话 | 删除、L0/L1、契约常量回填 commit、Beta、只读锁、档案、handoff、合并与 push | 档案、`docs/handoff/current.json`、契约常量 | 不替运行代理改代码 | — | — | 各判定行汇总 | — |

合并流程：每块 `git diff --name-only BASE..TIP` ⊆ 可写集合，越界 `SCOPE=FAIL`；合前审 PASS → `git merge --no-ff <TIP sha>` → 合后定向测试 → `POST_MERGE_REVIEW=PASS merge=<sha> tests=<结果> files=<n>` → push → 清理 worktree。全部 S 块须在建库 Beta（R6 冻结起点）之前合入。

## 三、闸门总表

| 闸门 | 命令／入口 | 判定行与阈值 | FAIL 处置 |
|---|---|---|---|
| B0 | `df -B1 --output=avail /scratch`，主会话按阶段需求比较 | `DF_CHECK=PASS stage=<s> avail_bytes=<n> need_bytes=<n>`；阈值：阶段 2 起跑前 ≥1.5 T；阶段 5 起跑前 ≥0.6 T | 停，提请第二档 |
| L0 | `check_data_lineage.py lineage --vla-input-manifest docs/dataset-build-doc/16task-h5-scan/records/input_manifest.json` | `DATA_LINEAGE=PASS encoder_ckpt=27daa832 encoder_source_pin=<pin_sha256> vla_input_pin=<sha> files=16 mismatches=0` | 立即停工（R0） |
| B1 | `check_orig80k_sources.py --input-manifest $INMANI --reference-input <records>/input_manifest.json --manifest $MANI --reference-manifest <records>/episode_manifest.json` | `INPUT_PIN=PASS files=16`、`EPISODE_IDENTITY=PASS episodes=1600 timesteps=768897 …`（余字段按代码原样） | 停 |
| L1 | 同 L0，`--vla-input-manifest $LIB/meta/input_manifest.json` | 同 L0 格式，`vla_input_pin` 须等于 L0 值 | 立即停工 |
| B2 | `run_local.py --stage siglip`（32 worker，`--resume`） | `STAGE_DONE stage=siglip workers=32 items=1600 elapsed=<s>`；另核 `SHARD_DONE` 恰 32 行、`_shard*of32.json` 恰 32 个 | 停 |
| B3 | `finalize_checks.py check … --spot_check 1024` | `全部检查通过 PASS` | 停 |
| B4 | `pack_framesamp_store.py verify --store $LIB/framesamp` | `VERIFY_PACK=PASS scanned=768897 mismatches=0` | 停 |
| B5 | 8×8 `verify` + `xgrid_pos_check.py` | `VERIFY_PACK=PASS scanned=768897 mismatches=0`、`IMAGE_NPY_SPOT=PASS frames=512 mismatches=0`、`MOTION_POS_XGRID=PASS t=<n> npy=64 mismatches=0` | 停 |
| B6 | `run_local.py --stage wan` | `STAGE_DONE stage=wan workers=8 items=<段数> skipped=<n> initial_complete=0 elapsed=<s>` | 停 |
| B7 | `oracle_driver.py vae`（8 片）+ `aggregate --kind vae` + `compare_wan.py latents` | `ORACLE_VAE=DONE … frame_mismatches=0 …`、`ORACLE_AGGREGATE=DONE kinds=vae shards=8`、`WAN_BITEXACT=PASS compared=<n> padded_covered=<n> frame_mismatches=0 …` | 停 |
| B8 | 主会话按 store_meta `source_spot_sha256.entries` 删 features | `FEATURES_PRUNE=PASS kept=8` | 停 |
| B9 | 主会话 `sha256sum norm_stats.json`＋`report`＋manifest sha | `DATA_IDENTITY=PASS norm_sha=<…> steps=768897 samples=476857 manifest_sha=<…>` | 停 |
| G0 | `check_stage2_validation_inputs.py --h5-root $RAW --input-manifest $INMANI …` | `RAW_WINDOWS_READY=PASS windows=64 tasks=16 demo_tail=<≥16> sha_mismatches=0` | 停 |
| G1 | MJ 抽取 `--check-dropout 64` | `DROPOUT_OFF=PASS windows=64 eval_repeat_max_abs=0 train_vs_eval_max_abs=<正数>` | 停 |
| G2 | `orig16_g2.py select` → 按 `G2-bins.txt` 下载 → `orig16_g2.py compare` | `WAN_EQUIV=PASS windows=64 oracle=mj-stride1@384d5db4 numerics=mj-provenance max_abs=0`；信息行 `WAN_NUMERICS_DRIFT=INFO …` 不判 | 停，**不 WAIVE** |
| G3 | `validate_stage2.py --gate G3 --rgb-root /scratch/hongze/dataset-rgb-gtxyz-16views-stride2-v1` | `EMA_SOURCE=PASS leaves=65 state_key=model exact=1 live_negative_rejected=1`、`EXPORT_PARITY=PASS windows=32 seed=0 max_abs=0 geometry_all_true=1 ckpt_sha256=27daa832…` | 停 |
| G5 | `pack --layout stage2 --latents-origin fresh` → `compare_stage2.py --scope sampled --expect-rows <rows>` → `verify --resume --check-padding` | `PACKED_TOKEN_BYTES=PASS compared=<rows> mismatches=0`、`ENCODER_SAMPLE_BITEXACT=PASS compared=1024 total=<rows> mismatches=0 seed=0`、`VERIFY_MOTION=PASS scanned=<rows> mismatches=0 nonfinite=0 segment_set_mismatch=0 real_plus_pad_ne_33=0 exec_padded=0 pad_not_last_demo=0 pad_source_mismatch=0 demo_tail_padded=<n>` | 停 |
| G4 | `check_stage2_online.py --store … --samples <VALID>/G0.json --history <新 yaml> --online-gpu 0` | `ONLINE_OFFLINE_PARITY=PASS windows=64 handshake=1 response_bytes=4096 encoder_max_abs=0 e2e_max_abs=<x≤1e-5> frame_ids_mismatch=0 state_machine_mismatch=0` | 停 |
| G6 | `train_es_bound.py --manifest $MANI --layout motion-stage2-1x1024-grid32-demopad17-v1 --budget 96` + 参数化 `eval_es_bound.py`（1300 步） | `TRAIN_ES_BOUND=PASS train_max=<n> eval_max=<n> budget=96` | 停，不改预算 |
| L2 | 契约内 | `CHECK_ENCODER_DATA_LINEAGE=PASS …` | 契约 FAIL 即不起跑 |
| P1 | `check_data_lineage.py paired` | `PAIRED_RUNS_SAME_LIB=PASS lib=16task-orig100-1600ep-<pin8> runs=2` | 停 |
| G7 | 512：`preflight_train_launch.py`；motion：`run_orig16_stage2_1024_80k.sh`（`MAX_STEPS=20`，内含契约）；`check_stage2_checkpoint.py --mode smoke` | `PREFLIGHT=PASS n=<n>`、`LAUNCH_CONTRACT=PASS checks=<n> failed=0`、`SMOKE_4GPU=PASS run=512 steps=20 exit=0 finite_losses=20/20 peak_mem_gib=<x> limit_gib=72`、`SMOKE_4GPU=PASS run=s2 …`（主会话据日志手工写）、`SAVE_LOAD=PASS missing=0 extra=0 shape_mismatch=0 ema_mismatch=0 reference_output_max_abs=0 online_infer=1` | 停 |
| G8 | 两份训练日志尾行 | `EXIT_CODE=0`，各 16 份 ckpt（5000…75000、79999）Orbax 提交完整 | 停，不 resume |
| G9 | `run_orig100_vla_512_80k_ckpt_export.sh`、`run_orig16_vla_stage2_1024_80k_ckpt_export.sh`（先 `SMOKE=1`） | 各 `CKPT_UPLOAD=PASS steps=16 … sha256_mismatch=0 public=1` | 长尾失败按第 27 条写 `pending_decisions` |
| G10 | 引用 10-26 记录 | `MJ_CKPT_UPLOAD=PASS source_runs=2 …`（已 PASS，不重传） | — |
| G11 | 不使用 | — | — |
| G12 | `check_stage2_checkpoint.py --mode delivery --delivery-step 79999 --online-gpu 1`（参数支持度由 E1 核实） | 按代码实际判定行（70k 先例为 `CKPT_70K_READY=PASS`） | 停 |
| B10 | `run_orig100_trainset_export.sh` | `RESULT=PASS …` | 长尾，失败交用户 |
| S0 | 主会话按 build_spec `bytes/mtime_ns` 比对 16 个源 H5 | `SOURCE_RO=PASS files=16 size_mtime_unchanged=16` | 交用户 |

## 四、runbook

所有阶段脚本写成 `v1-store/logs/o16-20261027/run-<会话名>.sh`（首行 `source v1-store/logs/o16-20261027/env.sh`），日志 `v1-store/logs/o16-20261027/<会话名>.log`，统一启动模板：

```bash
D=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/o16-20261027
tmux new-session -d -s <会话名> "set -o pipefail; PYTHONUNBUFFERED=1 bash $D/run-<会话名>.sh 2>&1 | tee $D/<会话名>.log; echo \"EXIT_CODE=\$?\" >> $D/<会话名>.log"
```

公共环境 `env.sh`：

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/dataset/paths.sh            # V1_STORE、HF_HOME=$V1_STORE/cache/hf、HF_HUB_OFFLINE=1；不覆盖 HF_HOME
export UV_CACHE_DIR=/scratch/hongze/.cache/uv OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
P=/scratch/hongze/robomme_policy_learning_MotionJEPA
MJ=/scratch/hongze/MotionJEPA
MJW=/scratch/hongze/worktrees/vla-s2-memexport-20261026
RAW=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/source/h5
PIN8=<阶段 0 实读回填>
NAME=16task-orig100-1600ep-$PIN8
LIB=$V1_STORE/datasets/$NAME
MANI=$LIB/meta/episode_manifest.json; INMANI=$LIB/meta/input_manifest.json
REF=$P/docs/dataset-build-doc/16task-h5-scan/records
CKPT=$MJ/runs/stage2-full144-1025-a/checkpoint_epoch_105.pt
CKPT_SHA=27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035
ENC=$V1_STORE/external/motionjepa/stage2-full144-1025-a      # 10-26 已建注册副本，复核 sha 后复用
VALID=$V1_STORE/validation/orig16-stage2
G2D=$V1_STORE/validation/orig16-g2
RUN512=orig100-m8x8-modul-b128-80k
RUNS2=orig100-m8x8-modul-motion-stage2-1024-b128-80k
```

Monitor 过滤词（每份日志一个 Monitor，`tail -n +1 -F <log> | stdbuf -oL tr '\r' '\n' | grep --line-buffered -E …`）：建库 `STAGE_DONE|SHARD_DONE|VERIFY_PACK=|全部检查通过|IMAGE_NPY_SPOT=|MOTION_POS_XGRID=|ORACLE_|WAN_BITEXACT=|DATA_LINEAGE=|INPUT_PIN=|EPISODE_IDENTITY=|FAIL|Traceback|Killed|EXIT_CODE=`；抽取与闸门 `RAW_WINDOWS_READY=|STAGE2_EXPORT=|DROPOUT_OFF=|WAN_EQUIV=|WAN_NUMERICS_DRIFT=|EMA_SOURCE=|EXPORT_PARITY=|PACKED_TOKEN_BYTES=|ENCODER_SAMPLE_BITEXACT=|VERIFY_MOTION=|ONLINE_OFFLINE_PARITY=|TRAIN_ES_BOUND=|FAIL|Traceback|EXIT_CODE=`；训练 `EXIT_CODE=|Traceback|out of memory|Killed|NaN|nan|step [0-9]+000|LAUNCH_CONTRACT=|PREFLIGHT=`；上传 `INC_UPLOADED|ROOT_UPLOAD=|CKPT_UPLOAD=|RESULT=|FAIL|EXIT_CODE=`。

**0. 阶段 0（主会话，T0 → T0+1 h）**
1. `grep -E '^CKPT_UPLOAD=PASS steps=14' v1-store/logs/vla-s2-20261026/close70-export.log` 且末行 `EXIT_CODE=0`；未出现则等（Monitor 挂该日志），不删任何东西。
2. `grep pin_sha256 /scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/source_manifest.json` → 取前 8 位回填本文全部 `<pin8>` 与 `env.sh`（文档 commit `docs:`，在 S 派发前）。
3. L0：`uv run --no-sync python scripts/training/check_data_lineage.py lineage --encoder-ckpt "$CKPT" --expected-ckpt-sha256 "$CKPT_SHA" --vla-input-manifest "$REF/input_manifest.json" --mj-source-manifest /scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/source_manifest.json`（S1 合入后才有该脚本；S1 合入前由主会话用只读 Python 片段按同一定义先核，正式 L0 行在 S1 合入后补跑）。
4. 第一档删除（R12，逐项判定行原文核到后）：`v1-store/restore/mixed8-200-vla-latents-20261021-a`（`hf buckets list HongzeFu/robomme-16task-mixed8-200-trainset-20261016-v1 -R` 与其 `RESULT=PASS` 记录）、`v1-store/exports/hf-ckpt-mixed8-200-vla-stage2-1024-80k`、`v1-store/train-runs/mme_vla_suite_b128_80k/mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k`（均以 `CKPT_UPLOAD=PASS steps=14` 为据）；删前 `ls -ld`、`du -sb` 列清单，删后 `DF_CHECK`。
5. 状态机：主会话在 `docs/handoff/current.json` 登记 `orig16-vla-build`、`orig16-vla-train-4plus4`（`planned`），并给 `mixed8-vla-stage2-1024-train` 的 `state_reason` 补「数据错配，结果仅作历史记录」。

**1. 阶段 1（T0+1 → T0+7 h）**：按二节派 E1、S6、S1–S5；合入顺序 S6 → S1 → S2 → S3 → S4 → S5。并行（不依赖新代码、CPU）：
```bash
# o16-hash（CPU，约 20 min：pub16 hash16 实测 1196 s）
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/scan_manifest.py build --raw_dir "$RAW" --episodes-per-task 100 --num_shards 1 --out "$MANI"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs --raw_dir "$RAW" --out "$INMANI"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/check_orig80k_sources.py --input-manifest "$INMANI" --reference-input "$REF/input_manifest.json" --manifest "$MANI" --reference-manifest "$REF/episode_manifest.json"
```
（`scan_manifest`/`hash-inputs` 是否记录 git HEAD 由 E1 核实；若记录，则推迟到建库 Beta 后作为阶段 2 第一步。）S1 合入后主会话跑 L1（`--vla-input-manifest "$INMANI"`）。

**2. 阶段 2 建库（X1 运行，T0+7 → T0+11.5 h）**：前置——全部 S 块已合入并 push；建库档案 `docs/dataset-build-doc/16task-orig100-1600ep-<pin8>/launch.md` 已提交；打建库 Beta（`commitV11.xxBeta`），`git status --short` 为空；写 `AGENTMETARULES_HOLD`（freeze）；`DF_CHECK stage=build` ≥1.5 T。
```bash
# o16-siglip（32 worker = 8 卡 × 4，XLA_PYTHON_CLIENT_MEM_FRACTION=0.15；mixed8 实测 4 worker 因 GPU 中断不送达过慢）
# 每卡重复 4 次的 --gpus 写法沿用 m8v-siglip-w32.sh 原文（实施时照抄）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --no-sync python scripts/dataset/run_local.py --stage siglip --lib "$LIB" --gpus <8 卡各重复 4 次> --raw-dir "$RAW" --require-free-mib 70000
# o16-finalize
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/dataset/finalize_checks.py check --manifest "$MANI" --out "$LIB/source" --raw_dir "$RAW" --input_manifest "$INMANI" --input_level sha256 --spot_check 1024
# o16-wan（GPU 0–7，grid16，路径 B；约 2.3 h 估算）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --no-sync python scripts/dataset/run_local.py --stage wan --lib "$LIB" --gpus 0,1,2,3,4,5,6,7 --raw-dir "$RAW" --require-free-mib 70000
# o16-pack4（CPU，与 o16-wan 并行）
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp" --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --store "$LIB/framesamp" --resume --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/compute_norm_stats.py --output-dir "$V1_STORE/train-assets/mme_vla_suite/$NAME" --config-name mme_vla_suite --repo-id robomme --dataset-path "$LIB/source"
# o16-pack8（CPU，o16-pack4 之后）
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --layout framesamp-8x8-v1 --reader decode --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp-8x8" --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --store "$LIB/framesamp-8x8" --resume --procs 48
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py report --store "$LIB/framesamp-8x8"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/xgrid_pos_check.py --store-4x4 "$LIB/framesamp" --store-8x8 "$LIB/framesamp-8x8" --source "$LIB/source" --image-spot 512
# 主会话：B8 删 features（store_meta 点名的 8 个除外，删前列清单）；B9 DATA_IDENTITY
# o16-oracle（o16-wan EXIT_CODE=0 后；MJ 环境；ORACLE 目录不得已存在；8 片写法照抄 mixed8 launch.md m8v-oracle 段）
ORACLE="$LIB/oracle/wan-mj"; test ! -e "$ORACLE"; mkdir -p "$ORACLE"
# 8 片：CUDA_VISIBLE_DEVICES=$i uv run --project "$MJ" --no-sync python "$P/scripts/dataset/wan/oracle_driver.py" --mj-repo "$MJ" vae --manifest "$MANI" --raw-dir "$RAW" --latents "$LIB/wan-latents" --out "$ORACLE" --shard-idx $i --num-shards 8 --demo-min-real 17 --exec-min-real 33 --sample-spec 'padded:all,rest:0.10,seed:0'
uv run --project "$MJ" --no-sync python "$P/scripts/dataset/wan/oracle_driver.py" aggregate --manifest "$MANI" --out "$ORACLE" --num-shards 8 --kind vae --demo-min-real 17 --exec-min-real 33
uv run --no-sync python scripts/dataset/wan/compare_wan.py latents --latents "$LIB/wan-latents" --oracle "$ORACLE" --sampled "$ORACLE/sampled_windows.json" --sample-spec 'padded:all,rest:0.10,seed:0' --expect-padded <主会话按 manifest 现场算>
```

**3. 阶段 3 抽取与闸门（X2 运行，T0+11.5 → T0+13.5 h）**
```bash
# o16-g0
uv run --no-sync python scripts/dataset/check_stage2_validation_inputs.py --manifest "$MANI" --latents "$LIB/wan-latents" --source "$LIB/source" --h5-root "$RAW" --input-manifest "$INMANI" --out "$VALID/G0.json"
# o16-s2export（MJ worktree，GPU 0；含 G1；先 --max-segments 4 冒烟到独立临时输出）
cd "$MJW" && CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/stage2_memory_export.py --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --latents "$LIB/wan-latents" --manifest "$MANI" --out "$LIB/motion-tokens-stage2" --grid-subsample 2 --batch 1 --check-dropout 64 --expected-counts <段数>,<行数>
# o16-g2（主会话先跑 select，按 G2-bins.txt 从公开 repo 下载到 $G2D/mj-oracle，unset HF_HUB_OFFLINE 只在本段）
UV_PROJECT_ENVIRONMENT="$V1_STORE/venvs/wan-stage2" uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/orig16_g2.py select --manifest "$MANI" --latents "$LIB/wan-latents" --mj-index "$G2D/mj-oracle/index" --out "$VALID/G2-windows.json"
# hf download HongzeFu/robomme-original-16x100-wan21-front-33f-dense-v1 --repo-type dataset --revision 384d5db410be713a49357bf3f55ca4f4d2f559a0 --include <G2-bins.txt 所列> index/* provenance.json SHA256SUMS --local-dir "$G2D/mj-oracle"（repo 类型与 CLI 写法实施时按 hf 版本确认；合计 ≤15 GB）
CUDA_VISIBLE_DEVICES=0 UV_PROJECT_ENVIRONMENT="$V1_STORE/venvs/wan-stage2" uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/orig16_g2.py compare --windows "$VALID/G2-windows.json" --mj-oracle "$G2D/mj-oracle" --raw-dir "$RAW" --latents "$LIB/wan-latents" --out "$VALID/G2.json"
# o16-g3
CUDA_VISIBLE_DEVICES=0 UV_PROJECT_ENVIRONMENT="$V1_STORE/venvs/wan-stage2" uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/validate_stage2.py --gate G3 --samples "$VALID/G0.json" --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --rgb-root /scratch/hongze/dataset-rgb-gtxyz-16views-stride2-v1 --out "$VALID/G3.json"
# o16-s2pack（CPU pack → GPU 抽样认证 → CPU verify；HOLD 在 VERIFY_MOTION=PASS 后由主会话删除）
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_motion_store.py pack --layout stage2 --latent-grid-subsample 2 --latents-origin fresh --manifest "$MANI" --tokens "$LIB/motion-tokens-stage2" --latents "$LIB/wan-latents" --out "$LIB/motion-stage2-1024" --encoder-run-dir "$ENC" --raw-dir "$RAW" --expected-ckpt-sha256 "$CKPT_SHA" --stage2-validation "$VALID" --stage2-source-pin scripts/dataset/wan_stage2/SOURCE_PIN.json
CUDA_VISIBLE_DEVICES=0 UV_PROJECT_ENVIRONMENT="$V1_STORE/venvs/wan-stage2" uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/compare_stage2.py --store "$LIB/motion-stage2-1024" --latents "$LIB/wan-latents" --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --scope sampled --batch 1 --expect-rows <行数>
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_motion_store.py verify --store "$LIB/motion-stage2-1024" --resume --check-padding
# o16-g4（GPU 0 父进程，sidecar --online-gpu 0；核实际物理卡号）
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/training/check_stage2_online.py --store "$LIB/motion-stage2-1024" --samples "$VALID/G0.json" --history "$P/src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion-stage2-orig16.yaml" --online-gpu 0 --out "$VALID/G4.json"
# G6
uv run --no-sync python scripts/training/tests/train_es_bound.py --manifest "$MANI" --layout motion-stage2-1x1024-grid32-demopad17-v1 --budget 96
```
（`compare_stage2.py`/`validate_stage2.py` 其余必填参数以 `--help` 为准，实施时照抄 1026 runbook 原文并只换路径。）G6 后主会话：L1 复跑、把实测 `norm_sha256/episode_manifest_sha256/input_manifest_sha256/canonical_manifest_sha256/motion_rows/train_max/eval_max` 回填 `check_orig16_stage2_1024_launch.py` 的 `Contract`（`commitV11.xx: 回填 orig16 启动契约实测常量`），push。

**4. 阶段 4 短测（X3 运行，T0+13.5 → T0+14.5 h）**
```bash
# o16-pre512
MMEVLA_FRAMESAMP_SOURCE="$LIB/source" MMEVLA_FRAMESAMP_MANIFEST="$MANI" uv run --no-sync python scripts/training/preflight_train_launch.py --repo "$P" --train-head <40 位 HEAD> --history-config perceptual-framesamp-modul-8frame-8x8.yaml --history-config-sha256 <sha256> --assets-dir "$V1_STORE/train-assets/mme_vla_suite/$NAME" --asset-id robomme --norm-stats-sha256 <B9 norm_sha> --dataset-path "$LIB/framesamp-8x8" --run-root "$V1_STORE/train-runs/mme_vla_suite_b128_80k/o16-smoke-512-<美东MMDDHHMM>" -- mme_vla_suite_b128_80k --exp-name o16-smoke-512-<美东MMDDHHMM> --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/$NAME" --data.assets.asset-id robomme --dataset-path "$LIB/framesamp-8x8" --model.history-config perceptual-framesamp-modul-8frame-8x8.yaml --fsdp-devices 4 --num-workers 8
# P1：两条 runner 的 DRY_RUN／契约 TRAIN_CMD= 行写入 $VALID/argv-512.txt、argv-s2.txt
uv run --no-sync python scripts/training/check_data_lineage.py paired --argv-a "$VALID/argv-512.txt" --argv-b "$VALID/argv-s2.txt"
# o16-smoke512（GPU 0–3）与 o16-smokes2（GPU 4–7）同时起
CUDA_VISIBLE_DEVICES=0,1,2,3 FSDP_DEVICES=4 NUM_WORKERS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false bash scripts/dataset/hf_export/orig100_runner/train-512-80k.sh o16-smoke-512-<美东MMDDHHMM> --num-train-steps 20 --no-wandb-enabled
RUN_NAME=o16-smoke-s2-<美东MMDDHHMM> MAX_STEPS=20 TRAIN_HEAD=<40 位 HEAD> FINAL_RECORD_DIR="$VALID/smoke/final-record" bash scripts/training/run_orig16_stage2_1024_80k.sh
# 两条退出后：o16-saveload（父 JAX GPU 0，sidecar GPU 1）
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/training/check_stage2_checkpoint.py --mode smoke --checkpoint <step19 绝对路径> --reference <保存前参考清单> --online-gpu 1 --out "$VALID/smoke/G7-saveload.json"
```
短测 ≤5 分钟且验收后只删本轮 `o16-smoke-*` 自有 run 目录；超 5 分钟按诊断 run 留档。主会话据日志写两行 `SMOKE_4GPU=PASS …`（含 `sec_per_step`，更新 39 h 估算）。

**5. 阶段 5 正式 4+4（X3 运行，T0+15 → T0+54 h）**：前置——两份训练档案 `launch.md` 已提交；训练 Beta `commitV11.xxBeta`、`git status --short` 空；`chmod` 记录到 `v1-store/o16-scripts/chmod-before.txt` 后 `chmod a-w`；`DF_CHECK stage=train` ≥0.6 T；L2／P1 在契约与 preflight 中再出一次。
```bash
# o16-train512（GPU 0–3）
CUDA_VISIBLE_DEVICES=0,1,2,3 FSDP_DEVICES=4 NUM_WORKERS=8 bash scripts/dataset/hf_export/orig100_runner/train-512-80k.sh "$RUN512"
# o16-trains2（GPU 4–7）
RUN_NAME="$RUNS2" TRAIN_HEAD=<Beta 40 位 sha> FINAL_RECORD_DIR="$VALID/$RUNS2/final-record" bash scripts/training/run_orig16_stage2_1024_80k.sh
# o16-inc512：bash v1-store/o16-scripts/o16-ckpt-inc.sh "$RUN512" HongzeFu/robomme-vla-orig100-modul-512-80k-v1 hf-ckpt-orig100-vla-512-80k scripts/dataset/hf_export/hygiene_allowlist_orig100_vla_512_80k.json o16-train512
# o16-incs2：bash scripts/dataset/hf_export/run_orig16_stage2_1024_ckpt_inc.sh（入口内固定 RUN/bucket/EXPORT_NAME）
# o16-trainset-up（长尾，ionice）：SMOKE=1 bash scripts/dataset/hf_export/run_orig100_trainset_export.sh → 全量
```
起跑判据：两份日志第 1 个 loss 有限、`LAUNCH_CONTRACT=PASS`／`PREFLIGHT=PASS` 原文在日志内。主会话每份日志挂一个 Monitor。

**6. 阶段 6 收尾（X4 运行 + 主会话，T0+54 → T0+62 h）**：两条 G8 后——`o16-up512`：`SMOKE=1` 再全量 `run_orig100_vla_512_80k_ckpt_export.sh`；`o16-ups2`：同法 `run_orig16_vla_stage2_1024_80k_ckpt_export.sh`；`o16-g12`：`check_stage2_checkpoint.py --mode delivery --delivery-step 79999 --checkpoint <远端回读 step 79999> --expected-manifest <发布清单> --online-gpu 1`（父 GPU 0）；主会话按记录恢复权限打 `PERM_RESTORED=OK`、跑 S0、写 result.md／records、更新状态机、按 tmux 清单逐个清理。

**本计划 tmux 清单（R8）**：`o16-hash`、`o16-siglip`、`o16-finalize`、`o16-wan`、`o16-pack4`、`o16-pack8`、`o16-oracle`、`o16-g0`、`o16-s2export`、`o16-g2`、`o16-g3`、`o16-s2pack`、`o16-g4`、`o16-pre512`、`o16-smoke512`、`o16-smokes2`、`o16-saveload`、`o16-train512`、`o16-trains2`、`o16-inc512`、`o16-incs2`、`o16-trainset-up`、`o16-up512`、`o16-ups2`、`o16-g12`。

## 五、风险登记

| 项 | 说明 | 处置 |
|---|---|---|
| G2 数值口径 | MJ 库 conv TF32 开、policy 钉死关；生产 latent 与 MJ bin 不会逐位 | G2 判据用 MJ 数值模式重编证明同源；生产差另出 INFO 行；不放宽、不 WAIVE；口径选择列入裁决 6 |
| encoder 训练缓存是 fp16 | encoder 训练用的 `dataset-rgb-gtxyz-16views-stride2-v1` 是整 33 帧 fp16 编码，与 VLA 生产 fp32 latent 仍有数值差（1026 G2 实测 max 0.0135） | 同数据集已满足 R-LINEAGE；数值差属编码精度，不是混用数据；记录于盲区，不作闸门 |
| SigLIP 慢 | 本机 GPU 完成中断不送达，JAX 每次阻塞同步约 100 ms | 直接 32 worker（mixed8 实测 6673 s／3200 集） |
| 冻结窗口内需要修代码 | R6 窗口内不能 commit | 全部 S 块在建库 Beta 前合入；窗口内发现缺陷即停，交用户决定是否作废已产出的分片重跑 |
| 两条 run 共享 CPU／IO | 各 8 workers、96 核，10-17 先例可行 | 增量与训练库上传 `ionice -c 3 nice -n 19` |
| 磁盘峰值 | features 剪枝前 ≈1.0–1.2 TB | 第一档删除后约 1.7 T；每阶段 `DF_CHECK` |
| 只读锁误伤 | chmod 范围过大会挡日志写 | 只锁被跟踪的 `src/ scripts/ packages/` 文件，清单留档 |
| 21.5 epoch 过拟合 | 样本量约 mixed8 的 1/3 | 每 5000 步存档，用户可按中间 ckpt 评估；步数列入裁决 1 |
| G2 下载失败 | 公开 repo 网络问题 | 同对象最多 3 次、间隔 60 s，仍失败停下报告 |
| restore_receipt 改动面 | 四个文件都校验它 | S2 用 `latent_origin.kind` 分支，mixed8 回归测试逐字节盯住 |

## 六、盲区诚实清单

- 原版库 Wan 编码吞吐未实测；2.3 h 是按 mixed8 23,119 s × 768,897/2,145,548 线性折算。窗口数、motion 行数同为估算，以 B6／G5 实测为准。
- features 剪枝前峰值 ≈465 GB 取自 pub16 量级，未在本次配置复测。
- 4 卡 stage2 单 token 速度未实测（≤1.75 s/步为估算），以 G7 短测为准；两条并行时彼此 IO 影响未实测。
- `scan_manifest`/`hash-inputs` 是否把 git HEAD 写进产物（决定能否在阶段 1 并行跑）未核，交 E1。
- `check_orig80k_sources.py` 是否比较 input_manifest 的 `raw_dir`（MJ 路径 ≠ 参考记录的 `/scratch/hongze/robomme_data_h5`）未核，交 E1；若比较则需加参数而不是放宽。
- `check_stage2_checkpoint.py --mode delivery --delivery-step` 是否接受 79999 未核，交 E1。
- MJ 抽取 CLI 读 policy 新生成 `wan-latents/metadata.json` 的字段要求（10-26 用的是 HF 恢复副本，同 schema 但非同一批）未逐字段核，交 E1；S6 只改计数。
- policy 段起点 `es` 与 MJ `D`、exec 截止 `C` 的逐集对应仅从 README 与代码语义推断，由 `orig16_g2.py select` 逐窗断言。
- MJ stride-1 库 HF repo 的类型（dataset repo）与路径布局 `wan_chunk_latents/<key>.bin` 取自本机 `SHA256SUMS` 与 publish 日志，下载命令写法实施时按 hf 版本确认。
- G4 端到端阈值 1e-5 沿用 1026，未在原版库实测。
- online_gpu 取 1 只影响评估部署，训练不读；评估方案另行授权。

## 七、状态机登记与交接

- 任务 id：`orig16-vla-build`（阶段 0–3）、`orig16-vla-train-4plus4`（阶段 4–6，`depends_on: [orig16-vla-build]`）；跨仓库依赖 MotionJEPA `stage3-formal-training-v1`（只读复核 ep105 存在且 sha = `27daa832…`）。登记由主会话在用户确认「是否登记进状态机」后写入，状态 `planned`；开工令到达时写 `authorizations[]`（`kind: go`）。
- `mixed8-vla-stage2-1024-train` 的 `state_reason` 补一句「数据错配，结果仅作历史记录」。
- `orig16-vla-build` 交接前必须完成：B1–B9、L1、G0–G6 判定行；所有建库 GPU 任务退出；HOLD 已删。长尾：B10 训练库上传（`o16-trainset-up`，`v1-store/logs/o16-20261027/o16-trainset-up.log`，完成行 `RESULT=PASS`）。
- `orig16-vla-train-4plus4` 交接前必须完成：G7、两条 G8、G12、`PERM_RESTORED=OK`、S0；长尾：两条 G9（`o16-up512`、`o16-ups2`，完成行 `CKPT_UPLOAD=PASS`）。
- 本任务 `done` 后，另获授权的 sim 评估可消费两条 79999（或中间 step）ckpt。

## 八、留档与 commit 纪律

- 建库档案 `docs/dataset-build-doc/16task-orig100-1600ep-<pin8>/{launch.md,result.md,records/}`；训练档案 `docs/training-doc/orig100-m8x8-modul-b128-80k/` 与 `docs/training-doc/orig100-m8x8-modul-motion-stage2-1024-b128-80k/`（`launch.md` 起跑前随 Beta 提交，`result.md` 跑完回写，同小版本号配对）。records 只放清洗后日志、判定行、G0／G2 样本清单与 SHA、耗时表，不放脚本、yaml、权重、latent。
- 每块 `--no-ff` 合并占一个项目号（接续 `commitV11.56`），body 按 `AGENTS.md` 第 11 条六项，含本文全部用户原话；子代理 `sub/<编号>:` 提交原样保留。
- 建库 Beta 与训练 Beta 各自独立，Beta 与回写结论 commit 禁止 squash/amend。冻结期内不 commit（R6、R7）。
- 每次 commit 后裸 `git push`；`v5-orig16` 首次推送若无 upstream 先问用户（第 11 条例外）。
- 本轮 tmux 清单写入各 `launch.md`。

## 九、进展登记

| 时刻（美东） | 阶段 | 事件 | 判定行原文 | commit |
|---|---|---|---|---|
| | | | | |
