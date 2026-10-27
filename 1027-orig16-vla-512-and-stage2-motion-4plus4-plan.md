# 1027-orig16-vla-512-and-stage2-motion-4plus4-plan.md — 用原版 RoboMME 16×100 重建 VLA 训练库，4+4 卡并行训 512（8 帧 8×8）与 512+stage2 motion（1024 单 token）两条 80k

> **裁决状态**：草案，等开工令。本文只是计划，不是代码实施、建库、下载、训练或上传的开工令（`AGENTS.md` 第 2 条）。修订 r2 吸收开工前一致性审查 `PLAN_CONSISTENCY_REVIEW=FAIL base=6d2f1d5 tip=41059d8 findings=12 blocking=3` 的全部条目。
>
> **格式例外（用户指定）**：`AGENTS.md` 第 2 条要求进 git 的计划写成 HTML；本计划按用户原话「你写一个 Markdown 的计划」与「计划写在根目录」（2026-10-26）写成根目录 `.md`，属用户指定例外。计划提交到新分支 `v5-orig16`，由主会话从 `v4-256concat1` **当前 HEAD**（≥`4029b1f`，`commitV11.57`）切出并合入本文。
>
> **用户原话（2026-10-27，逐字）**：
> ①「上次的训练应该错误了。你的这个 motion encoder 的这个训练是用了官方数据集来训练的，所以说你 policy 仓库也应该用官方的训练。」
> ②「写完整方案 我需要同时跑512 8帧8*8 和 512+motion」
> ③「加入一个强约束，以后 encoder 和 VI 必须是同一个数据集，不可以出现混用的情况。」（VI 即 VLA）
> 既往长期裁决沿用、不再询问：bucket 名由我们自定；计划 md 放仓库根目录并合进工作分支；对用户报时一律美东；所有训练产物上传 HF；长任务 detached tmux + Monitor；写入型子代理独立 worktree + `sub/` 提交 + sonnet 审查。
>
> **代码锚点**：policy `v4-256concat1` HEAD（≥`4029b1f`，`commitV11.57`；含 `d0445ce` 落地的 `AGENTS.md` 项目专属规则 P-LINEAGE 与 Codex 会话的 70k 收尾记录；`git diff --stat b25fd99 4029b1f -- src scripts` 为空，即代码与 `b25fd99`／`commitV11.56` 相同），工作副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，实施全部落 `v5-orig16`。MotionJEPA `v12-3decoder@d16966e`（含 MJ 侧 P3 同源规则），工作副本 `/scratch/hongze/MotionJEPA`；抽取 CLI 在 MJ worktree `/scratch/hongze/worktrees/vla-s2-memexport-20261026`，分支 `v12-3decoder-memexport`（基点 `6c79ebb`，HEAD `b5d23a5`）。
>
> **外部依赖锚点（P-LINEAGE 要求同时写 encoder 与两侧源 pin）**：
> - encoder ckpt `/scratch/hongze/MotionJEPA/runs/stage2-full144-1025-a/checkpoint_epoch_105.pt`（EMA 在 `model` 键），sha256 `27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035`。
> - encoder 训练源 pin：训练根 `/scratch/hongze/dataset-rgb-gtxyz-16views-stride2-v1` 的 `control/build_spec.json` → `identity.source_pin.files[]`（16 项，每项 `task/path/bytes/sha256/mtime_ns`），并与 MJ `/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/source_manifest.json` 的逐文件 sha 交叉一致；该文件的 `pin_sha256` 前 8 位记作 `<pin8>`（编写时只读预读值 `9e109419…`，开工第一步复核回填）。
> - VLA 库源 pin：新库 `meta/input_manifest.json` 的 `files{}`；同源证据为 L0／L1 的 `DATA_LINEAGE=PASS … files=16 mismatches=0`。
> - 源 H5：`/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/source/h5/record_dataset_<Task>.h5`（16 个，约 512 GB），上游 `Yinpei/robomme_data_h5@a5e4e25ffe8af34f64944f9533d06455ce5f8337`。
> - Wan VAE `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`。G2a 对拍 oracle 为 MJ 已公开的 stride-1 库 `HongzeFu/robomme-original-16x100-wan21-front-33f-dense-v1@384d5db410be713a49357bf3f55ca4f4d2f559a0`。出处是日志 `/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/logs/public-pipeline-20261023.log`：倒数第二行为 `PUBLIC_PIPELINE=COMPLETE repo=… revision=384d5db4… EXIT_CODE=0`，末行为 `EXIT_CODE=0`。
>
> **commit 体例**：功能改动 `commitV<x>.<y>: <中文>`；最新为 `commitV11.57`（`4029b1f`），本计划各块合并从 **`commitV11.58`** 起依次编号（S6 无 policy 号；S1→V11.58、S2→V11.59、S3→V11.60、S4→V11.61、S5→V11.62；契约常量回填 V11.63；建库 Beta `commitV11.64Beta`／结论 `commitV11.64`；训练 Beta `commitV11.65Beta`／结论 `commitV11.65`；实际号以合入时 `git log` 为准顺延）。文档 `docs:`，交接 `handoff:`；子代理 `sub/<编号>: <中文>`，经主会话 `git merge --no-ff <TIP sha>` 合入。

# 第一部分（给人看）

## 一、为什么重做

**错配事实**：阶段 III motion encoder（MotionJEPA run `stage2-full144-1025-a` 的 ep105 EMA）训练根是 `/scratch/hongze/dataset-rgb-gtxyz-16views-stride2-v1`（MJ `configs/stage2_pool4.yaml` 的 `data_root`）。该缓存 `control/build_spec.json` 的 `identity.source_pin.files[]` 逐文件列出 16 个原版 H5：`/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/source/h5/record_dataset_<Task>.h5`。原版 RoboMME 规格：16 任务 × 100 集 = 1600 集，每任务 easy 50／medium 25／hard 25，无 xhard；总帧 768,897，demo 帧 292,040，exec 样本 476,857；9 个任务有 video demo（InsertPeg、MoveCube、PatternLock、RouteStick、VideoPlaceButton、VideoPlaceOrder、VideoRepick、VideoUnmask、VideoUnmaskSwap）；集长 48–1411 帧；encoder 验证集为每任务 ep94–99。

10-26 的 VLA run `mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k` 用的却是 policy 仓 `v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f`（本地录制 mixed8-200，16×200 = 3200 集，含 xhard）。两库是同一任务参数的不同批次录制，**不是子集关系**。1026 计划风险登记第一行把它记成「encoder 域外，只能由 VLA 结果判断」而没有拦下，用户裁定这是错误（原话①）。

该 run 已按用户指令在 70k 停止（SIGTERM，`EXIT_CODE=143`），交付 70000 步。HF 导出在 tmux `cc-vla-s2-ckptup` 进行中（日志 `v1-store/logs/vla-s2-20261026/close70-export.log`，期望 `CKPT_UPLOAD=PASS steps=14 ...`）。其结果此后**仅作历史记录**。

**强约束（原话③）落地成红线 + 机械闸门，不是风险项**：

| 项 | 内容 |
|---|---|
| 红线 R-LINEAGE | encoder 训练源数据集与 VLA 训练库必须是同一数据集：同一组源 H5，逐文件 sha256 相同。任何混用一律 FAIL 停工，不得 WAIVE，不得登记成「已知风险」继续。规则本体是 `AGENTS.md` 项目专属规则 P-LINEAGE（`d0445ce`），本计划只引用。 |
| 闸门 `DATA_LINEAGE` | 新脚本 `scripts/training/check_data_lineage.py`，三步：① encoder ckpt 旁 `config.yaml` 的 `data.data_root` → `<data_root>/control/build_spec.json` 的 `identity.source_pin.files[]`；② 与 MJ `dataset/source_manifest.json` 的 `files.<Task>.h5.{file,sha256,size}` 交叉核对；③ 与 VLA 库 `meta/input_manifest.json` 的 `files{}` 按文件名逐个比 `size+sha256`，集合必须相等。判定行 `DATA_LINEAGE=PASS encoder_ckpt=27daa832 encoder_source_pin=<pin_sha256> vla_input_pin=<sha> files=16 mismatches=0`。 |
| 两个 pin 不比值 | `encoder_source_pin`（取 MJ `pin_sha256`）与 `vla_input_pin`（VLA 文件集 canonical JSON 的 sha256）算法不同，只作身份记录、不互相比较。build_spec 的 `identity.source_pin.sha256` 是 source_receipts 的哈希，也不等于 MJ `pin_sha256`。**同源只以逐文件 sha256 集合相等判定。** |
| 接入点 | L0：阶段 0 对参考清单预检。L1：建库 hash 后正式核，并在 `o16-siglip` 脚本首行再核一次。L2：两条 run 起跑时硬核，包括 512 runner 启动前自检、stage2 契约必检项 `CHECK_ENCODER_DATA_LINEAGE`、两条正式起跑脚本首行。P1：`PAIRED_RUNS_SAME_LIB`，内部先对每条 run 各自 `DATA_LINEAGE=PASS`，再比库路径相同。任何一处 FAIL 即 `exit≠0`。 |

**已核实的同源性**：MJ 副本 16 个 H5 与 policy 仓 `docs/dataset-build-doc/16task-h5-scan/records/input_manifest.json` 登记的 sha256 逐文件相同（16/16）。它们与 2026-09 从 `Yinpei/robomme_data_h5` 下载、后来已删的 `/scratch/hongze/robomme_data_h5` 是同一份数据（例：BinFill `1b3a7139…`，40,186,950,704 B）。因此：
- **不必重新下载**；raw 目录直接用 MJ 副本路径，只读、不复制。
- policy 仓 2026-09-25 曾用同一份数据建过 `16task-pub-1600ep`（source + 4×4，SigLIP 8 worker 9767 s），所以本次不需要 mixed8 的 fetch/merge 阶段，直接从 H5 建 manifest。

**泄漏说明（信息，不是问题）**：encoder 验证集 ep94–99 落在 VLA 训练集内，但 encoder 冻结、不参与 VLA 梯度，不构成泄漏。VLA 的 sim 评估按 RoboMME 官方 test/val seed（test 560000、val 1060000 起），与训练集无交集。

## 二、要做什么与运行一览

**一句话**：从原版 16×100 H5 照搬 mixed8 建库链（去掉 fetch/merge），建新库 `16task-orig100-1600ep-<pin8>`（SigLIP source、4×4、8×8、自算 norm_stats、Wan grid16 latent）。然后在新库上用 ep105 EMA encoder 重抽 stage2 1024 单 token motion 库。最后 4+4 卡并行训两条 80k：
- 512：8 帧 8×8 modulation，无 motion；
- 512+stage2 motion：与 10-26 run 同一 motion 机制，只换库。

| 运行 | run 名 | GPU | 配置 | 步数 | 预计时长 | bucket（公开） |
|---|---|---|---|---|---|---|
| 建库 | 库 `16task-orig100-1600ep-<pin8>` | SigLIP／Wan 0–7 | 照搬 mixed8 链 | — | 约 5 h（估算） | `HongzeFu/robomme-16task-orig100-1600ep-trainset-20261027-v1` |
| 512 | `orig100-m8x8-modul-b128-80k` | 0–3 | `mme_vla_suite_b128_80k`，HC `perceptual-framesamp-modul-8frame-8x8.yaml`，FSDP 4、workers 8 | 80,000 | 1.7183 s/步 × 80k ≈ 38.2 h（10-17 实测速率外推） | `HongzeFu/robomme-vla-orig100-modul-512-80k-v1` |
| 512+motion | `orig100-m8x8-modul-motion-stage2-1024-b128-80k` | 4–7 | 同上 + 新 yaml `perceptual-framesamp-modul-8frame-8x8-motion-stage2-orig16.yaml`（budget 96、stride 32、`[1,1024]`） | 80,000 | ≤1.75 s/步 × 80k ≈ 39 h（估算） | `HongzeFu/robomme-vla-orig100-modul-motion-stage2-1024-80k-v1` |
| MJ ckpt | 阶段 II＋III | — | 已 PASS | — | 不重传 | `HongzeFu/motionjepa-stage2-pool4-3decoder-1025-v1`（G10） |

两条并行总墙钟约 39 h。若改为串行，8 卡各约 21 h（按 8 卡 0.96 s/步实测外推），合计约 43 h，所以按用户原话②并行。tmux 前缀一律 `o16-`；运行脚本与日志在 `v1-store/logs/o16-20261027/`。

## 三、数据：原版库规格、与 mixed8 的差异、目录与磁盘

**库名**：`16task-orig100-1600ep-<pin8>`。`<pin8>` = MJ `dataset/source_manifest.json` 的 `pin_sha256` 前 8 位（体例同 mixed8 取源 MANIFEST sha 前 8 位）。
- 库根：`/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-orig100-1600ep-<pin8>`。
- norm_stats：`v1-store/train-assets/mme_vla_suite/16task-orig100-1600ep-<pin8>/robomme/norm_stats.json`。

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

**H5 内部结构（已核实）**：
- 顶层 100 个 `episode_<N>`。
- 每组有 `setup/{difficulty, seed, task_goal(2,), front_camera_intrinsic, ...}` 与 `timestep_0…timestep_{T-1}`。
- 每 timestep 有：
  - `obs/front_rgb (256,256,3) uint8`、`obs/front_depth`、`obs/joint_state (7,)`、`obs/eef_state (6,)`、`obs/gripper_state (2,)`；
  - `action/{eef_action(7,), joint_action(8,), ...}`；
  - `info/{is_video_demo, is_completed, is_subgoal_boundary, ...}`。

这正是 `scan_manifest.py build`／`build_shard.py`（SigLIP 建库）直接消费的形态。

**磁盘（2026-10-27 美东约 12:40 实测）**：`/scratch` 6.9 T 总、5.7 T 已用、**1.3 T 可用**；GPU 0–7 全空。

| 对象 | 大小 | 处置 |
|---|---|---|
| `dataset-rgb-gtxyz-16views-stride2-v1`（encoder 训练缓存，HF 上传 `HF_VERIFY=FAIL`） | 2.8 T | 不可删 |
| `dataset-original-robomme-16x100-wan21-v1`（本计划的源） | 564 G | 不可删、只读 |
| `v1-store/restore/mixed8-200-vla-latents-20261021-a`（来自 PASS bucket 的恢复副本） | 71 G | 第一档（裁决 ⑦） |
| 10-26 导出 `v1-store/exports/hf-ckpt-mixed8-200-vla-stage2-1024-80k` | 156 G | 第一档，以 `CKPT_UPLOAD=PASS steps=14` 为据（裁决 ⑦） |
| 10-26 run 本地 ckpt | 155 G | 第一档，同上（裁决 ⑦） |
| mixed8 VLA 库（HF `RESULT=PASS`，但 `motion-stage2-1024/` 261 M 与 `motion-tokens-stage2/` 363 M 未上传） | 1.2 T | **默认保留**。仅当建库前可用 <1.5 T 时作第二档，先补传这两个子目录并回读，再经用户裁决才删（裁决 ②） |
| `dataset-local-a100-16task-mixed8-200`（mixed8 原帧，备份未核实） | 324 G | 不动 |

**需求估算**：
- 库 ≈ 190 + 50 + 202 + 27 ≈ 470 GB（features 剪枝后）。
- 建库峰值另加 features ≈ 465 GB（pub16 实测量级，出现在 SigLIP 后、剪枝前）。
- 两条 run 的 ckpt 各 ≈ 190 GB（16 份）；JAX cache／日志 ≈ 50 GB；G2a oracle 抽样下载 ≤ 15 GB。
- 合计：峰值约 1.0–1.2 TB，训练期稳态约 0.9 TB。

第一档约 382 GB 删除后，可用约 1.7 T。每阶段起跑前打 `DF_CHECK` 判定行，阈值见第二部分三节 B0。

## 四、memory 抽取链在原版库上的差异点

抽取口径与 1026 完全相同：
- 窗：33 帧（锚点 + 未来 32）、`GRID_STRIDE` 32、`segment_start`、`forward`。
- 补帧：demo 末窗真实帧 ≥17 的保留并 `repeat_last` 补到 33，≤16 的丢弃；exec 全真实。
- token：每窗 4×256 按源时间片 [0,3,5,8] 拼成 1 个 1024 维 token；budget 96。
- 数值：抽取 `eval()` + `no_grad()` + fp32；memory 不增强；encoder = ep105 EMA。

差异只有以下五处：

| 差异 | 原因（已核代码） | 本次做法 |
|---|---|---|
| Wan latent 需重新编码 | 1026 用的是 mixed8 恢复副本，原版库没有现成 latent。MJ stride-1 库（658,350 窗、不补帧、exec 段 `[D,C)` 排除首个 `is_completed` 帧）**本机只剩 index 与 metadata**：`dataset/` 只有 4.3 MB，`index/{episodes,segments,windows,train_points}.parquet` 在本机，`wan_chunk_latents/` 不在本机、全量在 HF repo；口径也不同 | 用 policy 自己的 `run_local.py --stage wan`（`scripts/dataset/wan/extract_wan.py`）在新库上编码 |
| grid32 还是 grid16 | `extract_wan.py` 的 argparse 只有 `--manifest/--raw-dir/--out/--worker/--segments/--max-items`，步长取模块常量 `wan_common.GRID_STRIDE = 16`，**没有 grid 参数**。MJ 抽取 CLI（`--grid-subsample`，默认 2）与 `pack_motion_store.py pack --layout stage2`（强制 `--latent-grid-subsample 2`）都按「grid16 取偶数格」实现 | **路径 B（默认，裁决 ⑧）**：沿用 grid16 编码，下游 `--grid-subsample 2`。不改 Wan 代码，代价是 Wan 多算一倍（约 2.3 h，估算）。路径 A 要给 `wan_common`/`extract_wan` 加 grid 32 参数，并同步改 MJ 抽取、pack、metadata 合同与 stage2 来源 schema 的 `source_grid` 语义，超出同等规模 |
| G0 读帧来源 | `check_stage2_validation_inputs.py` 的 demo 帧从 mixed8 `data-raw/<Task>_ep<N>/video_demo.h5` 读，并依赖 `control/scope.json`；原版没有 data-raw 形态 | 新增原版模式 `--h5-root`：demo 原帧直接读源 H5 `episode_<N>/timestep_<t>/obs/front_rgb`，身份绑定 `meta/input_manifest.json`；exec 仍读 `source/data/<sample>.pkl`。另加 `--mj-index`：exec 候选窗限定 `start+33 ≤ C`，并逐集断言 policy `exec_start_idx` = MJ `D`，使 G0 的全部全真实窗都落在 MJ 库覆盖内 |
| G2 对拍对象与数值口径 | 两处冲突。① MJ 库 `provenance.json` 记录 fp32 权重、batch 1、`cudnn_conv_tf32: true`、`matmul_tf32: false`，而 policy `wan_motion_infer.py::pin_numerics` 钉 `cudnn.allow_tf32 = False`。② `motion_store.py::validate_stage2_g2_evidence` 要求 `samples_sha256` 等于 G0 清单 sha、`windows == 64`、`demo_tail` 等于 G0 的且 ≥16、`worst_mse ≤ 1e-6`、`worst_max ≤ 0.01`、`numerics.cudnn_allow_tf32 == False`、`online_vae.tf32 == "off"`，以及 SOURCE_PIN 五键 | **G2 分两段，窗集即 G0 的 64 窗**（`samples_sha256` 对齐 G0）：G2a 对 G0 中全部全真实窗，在 MJ 数值模式下重编，与 MJ bin 逐位比较；G2b 对 G0 全部 64 窗（含 ≥16 个 demo 补帧尾窗），在 policy 生产数值模式下重编，与本库 wan-latents 逐位比较，证明补帧链路。两段合成一份 G2.json。顶层 `numerics`／`online_vae` 仍是生产口径（TF32 关），MJ 口径只放 `oracle.numerics`。fresh 分支的校验规则由 S2 实现，原 mse/max 阈值一律不放宽，不允许 WAIVE |
| 来源链 | `pack_motion_store.py::cmd_pack` 的 stage2 分支强制 `--restore-receipt`（mixed8 恢复副本收据）；`motion_store.py`、`motion_client.py`、`train.py` 的 stage2_compat 都校验 `restore_receipt`。MJ 抽取 CLI 写死 `FULL_COUNTS = (4995, 65419)`，`compare_stage2.py` 写死 `num_rows != 65419` | 新增「fresh-extract」来源模式，mixed8 旧路径语义不变、旧测试仍绿；计数改为显式参数 |

## 五、训练：4+4 配置、速度、epoch 口径

**512 run**：runner 照搬 `scripts/dataset/hf_export/mixed8_200_runner/train-512-80k.sh` 的结构。但 `_train_common.sh` 把库名写死为 `NAME=16task-mixed8-200-3200ep-e3c7564f`，`DS`、`MMEVLA_FRAMESAMP_SOURCE`、`MMEVLA_FRAMESAMP_MANIFEST`、`ASSETS_DIR` 全由它派生。所以新建平行目录 `scripts/dataset/hf_export/orig100_runner/`，不改 mixed8 旧文件，新 `_train_common.sh` 做两件事：
- 库名换成新库；
- 在打印 `TRAIN_CMD=` 之后、执行训练之前，**硬调** `check_data_lineage.py lineage`（encoder 取 ep105，VLA 侧取 `$DS/meta/input_manifest.json`），把 `DATA_LINEAGE=PASS …` 打进 run 日志，FAIL 即 `exit 3`。`preflight_train_launch.py` 不改。

CLI 必须带 `--dataset-path $DS/framesamp-8x8 --assets-base-dir ... --data.assets.assets-dir v1-store/train-assets/mme_vla_suite/16task-orig100-1600ep-<pin8> --data.assets.asset-id robomme`。config 默认 assets 指向 4task，漏写会静默读错 norm stats。

启动命令：`CUDA_VISIBLE_DEVICES=0,1,2,3 FSDP_DEVICES=4 NUM_WORKERS=8 bash scripts/dataset/hf_export/orig100_runner/train-512-80k.sh orig100-m8x8-modul-b128-80k`。起跑前先过 `scripts/training/preflight_train_launch.py`（`CHECK_*` 与 `PREFLIGHT=PASS n=`，不传 `--launch-mode`）。

**512+motion run**：mixed8 的入口与契约都写死了不能复用的值。
- 入口 `scripts/training/run_mixed8_stage2_1024_80k.sh` 写死 `HC`、`PARENT`、`ASSETS_DIR`、正式 run 名、`CUDA_VISIBLE_DEVICES=0,…,7` 与短测前缀 `cc-vla-s2-smoke-`。
- 契约 `scripts/training/check_mixed8_stage2_1024_launch.py`：
  - 写死 `HISTORY_CONFIG`/`FRAME_CONFIG`；
  - 写死 `Contract` 的 `parent_rel/assets_rel/norm_sha256/episode_manifest_sha256/input_manifest_sha256/canonical_manifest_sha256/motion_rows=65419/train_max=62/eval_max=79/fsdp_devices=8/num_workers=16/gpus=0..7`；
  - 写死 `CFG_RUN_NAME`；
  - `check_config` 要求 `fsdp_devices` 等于基础配置（基础配置为 8）；
  - `VALUE_FLAGS` 不含 `--fsdp-devices`/`--num-workers`。

方案是**新建**三个平行文件：`scripts/training/run_orig16_stage2_1024_80k.sh`、`scripts/training/check_orig16_stage2_1024_launch.py`、`src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion-stage2-orig16.yaml`。新契约的改动：
- GPU 改 `4,5,6,7`、FSDP 4、workers 8；
- 新增必检项 `CHECK_ENCODER_DATA_LINEAGE`（内部调用同一个 `check_data_lineage.check_lineage`）；
- 新增 `CHECK_G2_NOT_WAIVED`。

**速度（10-17 4+4 先例实测）**：
- 稳态 512 1.7183 s/步、motion 1.7346 s/步（38 h 15 min／38 h 37 min）；
- 短测峰值显存 39.3／38.3 GiB（limit 72）；
- 8 卡单跑 stage2 motion 0.96 s/步。

本次 motion 为单 token、budget 96、memory 608，4 卡估 ≤1.75 s/步、80k ≈ 39 h（估算，G7 短测时更新）。

**epoch 口径提醒（请用户看见）**：原版库 exec 样本 476,857，b128 × 80k = 10.24 M 样本 ≈ **21.5 个 epoch**。对照：
- mixed8 3200 集，80k ≈ 7.4 个 epoch；
- 4task-v2-1600ep（605,611 样本），80k ≈ 16.9 epoch；
- 本库 60k ≈ 16.1 epoch，40k ≈ 10.7 epoch。

默认沿用 80k（用户要求并行跑同样两条）。是否改成 60k/40k 列入第八节裁决 ①。

## 六、验收闸门总表

| 编号 | 判定行 | 查什么／过了说明什么 |
|---|---|---|
| B0 | `DF_CHECK=PASS stage=<s> avail_bytes=<n> need_bytes=<n>` | 每阶段起跑前磁盘够用；不够即停 |
| L0 | `DATA_LINEAGE=PASS encoder_ckpt=27daa832 encoder_source_pin=<pin_sha256> vla_input_pin=<sha> files=16 mismatches=0`（对参考清单） | 开工前确认 encoder 源 = 参考 16 文件 |
| B1 | `INPUT_PIN=PASS files=16`、`EPISODE_IDENTITY=PASS episodes=1600 timesteps=768897 …` | `check_orig80k_sources.py`：新 input_manifest 与参考逐文件（size/sha256）、逐 episode 同一 |
| L1 | 同 L0 格式（对新库 `meta/input_manifest.json`）；`o16-siglip` 脚本首行再打一次 | 新库输入 = encoder 训练源，R-LINEAGE 机械成立；建库 GPU 阶段起跑前再确认 |
| B2 | `STAGE_DONE stage=siglip workers=32 items=1600 elapsed=<s>` | SigLIP 全覆盖 |
| B3 | `全部检查通过 PASS` | finalize 1024 帧复算与输入 sha256 级核验 |
| B4 | `VERIFY_PACK=PASS scanned=768897 mismatches=0`（4×4） | 4×4 打包逐帧一致 |
| B5 | `VERIFY_PACK=PASS scanned=768897 mismatches=0`（8×8）、`IMAGE_NPY_SPOT=PASS frames=512 mismatches=0`、`MOTION_POS_XGRID=PASS …` | 8×8 打包与跨档位置一致 |
| B6 | `STAGE_DONE stage=wan workers=8 items=<n> …` | Wan grid16 全段完成 |
| B7 | `ORACLE_VAE=DONE … frame_mismatches=0`、`WAN_BITEXACT=PASS …` | policy Wan 与 MJ 环境独立 oracle 逐位一致（pack 前置） |
| B8 | `FEATURES_PRUNE=PASS kept=8` | features 只留 8 个抽样 |
| B9 | `DATA_IDENTITY=PASS norm_sha=<…> steps=768897 samples=476857 manifest_sha=<…>`（主会话手工行） | 库身份四方绑定 |
| G0 | `RAW_WINDOWS_READY=PASS windows=64 tasks=16 demo_tail=<≥16> sha_mismatches=0` | 抽样原帧（改读源 H5）与 latent 输入 SHA 一致；全真实窗均在 MJ 覆盖内 |
| G1 | `DROPOUT_OFF=PASS windows=64 eval_repeat_max_abs=0 train_vs_eval_max_abs=<正数>` | dropout 真关 |
| G2 | `WAN_EQUIV=PASS windows=64 demo_tail=<=G0> g2a_windows=<64−demo_tail> g2a_max_abs=0 g2b_windows=64 g2b_max_abs=0 worst_mse=0 worst_max=0 oracle=mj-stride1@384d5db4`；信息行 `WAN_NUMERICS_DRIFT=INFO …` 不判 | G2a 证同帧、同窗、同 VAE、同预处理；G2b 证生产补帧链路；G2.json 须过 S2 fresh 校验器；**不允许 WAIVE** |
| G3 | `EMA_SOURCE=PASS leaves=65 state_key=model exact=1 live_negative_rejected=1`、`EXPORT_PARITY=PASS windows=32 seed=0 max_abs=0 geometry_all_true=1` | 真加载 EMA、导出路径数值一致 |
| G4 | `ONLINE_OFFLINE_PARITY=PASS windows=64 handshake=1 response_bytes=4096 encoder_max_abs=0 e2e_max_abs=<x> …` | 在线 sidecar 与离线库一致（x≤1e-5） |
| G5 | `PACKED_TOKEN_BYTES=PASS compared=<rows> mismatches=0`、`ENCODER_SAMPLE_BITEXACT=PASS compared=1024 total=<rows> mismatches=0 seed=0`、`VERIFY_MOTION=PASS scanned=<rows> …` | motion 库完整、补帧正确 |
| G6 | `TRAIN_ES_BOUND=PASS episodes=1600 tasks=16 max_windows=<n> budget=96 headroom=<≥0>`（`train_es_bound.py`）；契约内 `CHECK_BUDGET_TRAIN=PASS train_max=<n> budget=96`、`CHECK_BUDGET_EVAL=PASS eval_max=<n> …` | 训练与 1300 步评估零截断 |
| L2 | 512 runner 日志内 `DATA_LINEAGE=PASS …`；stage2 契约内 `CHECK_ENCODER_DATA_LINEAGE=PASS …`；两条正式起跑脚本首行 `DATA_LINEAGE=PASS …` | 起跑时硬核血缘，FAIL 即 `exit≠0` 不起训练 |
| P1 | `PAIRED_RUNS_SAME_LIB=PASS lib=16task-orig100-1600ep-<pin8> runs=2` | 两条 run 各自 `DATA_LINEAGE=PASS` 且同一父库、同一 assets |
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
| 0 磁盘与句柄 | 开工审查 `PLAN_CONSISTENCY_REVIEW`；等 `CKPT_UPLOAD=PASS steps=14`；实读 `<pin8>` 回填；L0（S1 合入前用只读片段预核）；经裁决 ⑦ 删第一档三项；状态机登记（裁决 ⑨） | 0.5–1 h（不含等 ckptup） | T0 → T0+1 h | L0、B0、`CKPT_UPLOAD=PASS steps=14` |
| 1 代码改动与审查 | 第一批 S1、S2、S3、S6 并行；S1/S2 合入后第二批 S4、S5；逐块 sonnet 合前审、`--no-ff` 合入、合后审、push。并行跑 CPU 前置 `scan_manifest`、`hash-inputs`、`check_orig80k_sources`（两者不写 git HEAD，不受冻结影响），S1 合入后跑正式 L0 与 L1 | 6–7 h | T0+1 → T0+8 h | `PRE/POST_MERGE_REVIEW=PASS`、B1、L0、L1 |
| 2 建库（含 Wan） | 建库 Beta、HOLD 冻结 → `o16-siglip` 首行 L1 → SigLIP 32 worker → finalize → Wan grid16（GPU 0–7）∥ 4×4／norm／8×8／xgrid（CPU）→ oracle vae → 剪枝 features | 约 4.5 h | T0+8 → T0+12.5 h | B2–B9 |
| 3 memory 抽取与闸门 | G0 → MJ 抽取（G1）→ G2（HF 抽样 bin）→ G3 → pack → compare_stage2 sampled → verify（冻结结束）→ G4 → G6 → 回填契约实测常量 commit | 约 2 h | T0+12.5 → T0+14.5 h | G0–G6 |
| 4 短测 | P1；两条 20 步并行（0–3、4–7）→ motion 保存恢复 | 约 1 h | T0+14.5 → T0+15.5 h | G7、P1、L2 |
| 5 正式 4+4 与增量上传 | 训练 Beta、只读锁；两条起跑脚本首行 L2＋P1 后同时起跑；两路增量上传；训练库长尾上传 | 约 39 h | T0+16 → T0+55 h | G8、`INC_UPLOADED` |
| 6 收尾 | 两条最终导出 + 全量回读（G9）、motion 79999 严格恢复（G12）、档案、恢复只读锁、S0、handoff | 约 4–8 h | T0+55 → T0+63 h | G9、G12、`PERM_RESTORED=OK`、S0 |

## 八、需用户裁决项（括号内为默认）

1. **步数**：两条是否仍训 80k（默认 80k，≈21.5 epoch）；可改 60k（≈16.1）或 40k（≈10.7）。改动落在启动脚本覆盖还是全局配置，需按 `AGENTS.md` 第 10 条确认。
2. **mixed8 VLA 库（1.2 T）是否删除**（默认保留）。仅当建库前可用 <1.5 T 时才提请；先补传 `motion-stage2-1024/` 与 `motion-tokens-stage2/` 并回读。
3. **是否等 `cc-vla-s2-ckptup` 打出 `CKPT_UPLOAD=PASS steps=14` 再开工**（默认等）。
4. **norm_stats 用新库自算**（默认是，与 mixed8 做法一致）。
5. **G2a oracle 抽样下载**（默认同意）：从公开 repo `HongzeFu/robomme-original-16x100-wan21-front-33f-dense-v1@384d5db4` 只下 G0 全真实窗所在的段 bin，上限 15 GB，落 `v1-store/validation/orig16-g2/mj-oracle/`。
6. **生产 Wan 数值口径**：默认保持 policy 钉死的 `cudnn.allow_tf32=False`，与在线 sidecar 一致，也满足现有 G2 校验器；同源性由 G2a 在 MJ 模式下重编来证明。另一选项是改成 MJ 库的 conv TF32 开，但会破坏 G4 与 G2 校验器。
7. **第一档删除约 382 GB 是否授权**：恢复副本 71 G、10-26 导出 156 G、10-26 本地 run 155 G。每项都以 HF 判定行为据：恢复副本对应 mixed8 trainset bucket 的 `RESULT=PASS`，后两项对应 `CKPT_UPLOAD=PASS steps=14`。默认请授权。
8. **Wan 路径**：默认路径 B（grid16 + `--grid-subsample 2`，零改 Wan 代码）；另一选项是路径 A（改 Wan 代码直出 grid32）。
9. **状态机与推送**：是否同意把 `orig16-vla-build`、`orig16-vla-train-4plus4` 登记进 `docs/handoff/current.json`；`v5-orig16` 首次推送是否同意 `git push -u origin v5-orig16`。默认请同意。

## 九、停点清单

- `CKPT_UPLOAD` 非 PASS，或 tmux 消失且无完成行：不删 10-26 产物，交用户。
- 任一 `DF_CHECK` 低于阈值：停，提请第二档。
- `DATA_LINEAGE` 或 `PAIRED_RUNS_SAME_LIB` 任一次 FAIL 或缺行（L0／L1／L2／P1）：立即停工，不得 WAIVE（R-LINEAGE）。
- 任一 B／G 闸门 FAIL：停，留证交用户，不放宽阈值。G2 FAIL 不得走 `accept_stage2_g2_exception.py`。
- 合前审两轮仍 FAIL、或合后审 FAIL：写 `AGENTMETARULES_HOLD`，交用户。
- 训练崩溃、NaN、OOM：停下问用户，不 resume。
- G2a 下载超出 15 GB，或出现任何未列入的下载：先问。
- push 被拒：停，原文交用户。

## 十、子代理分工与合并（简述）

拆六块写入型，全部 `opus` + 独立 worktree：

| 块 | 内容 |
|---|---|
| S1 | G0 原版模式与血缘／配对检查脚本 |
| S2 | stage2 来源链 fresh-extract 模式与 G2 fresh 校验规则 |
| S3 | G2 两段对拍脚本 |
| S4 | 两条 run 的启动入口、契约、yaml、512 runner（含血缘硬调） |
| S5 | 三套上传驱动 |
| S6 | 在 MotionJEPA 抽取 CLI 去掉 mixed8 计数硬编码 |

派发分两批：
- **第一批**：S1、S2、S3、S6 同时并行。
- **第二批**：S4、S5，等 S1 的 `check_lineage` 接口与 S2 的证据键合入后，从合并后的 `v5-orig16` HEAD 切出再派。

合并顺序 S6 → S1 → S2 → S3 → S4 → S5。每块合前派一个 `sonnet` 只读审查，钉死 BASE／TIP。凡涉及起跑脚本或契约，审查须核到 `DATA_LINEAGE` 闸门存在且 FAIL 即停，缺失即判 FAIL。合后主会话跑定向测试与范围核对，再 push。

运行另派 `opus` 运行代理，在主检出启动、交回句柄，不改代码；探索用 `haiku` 只读。逐文件写集与验收命令见第二部分二节。

# 第二部分（技术细节，供 agent 追踪）

## 〇、前置声明与红线

- **R0 R-LINEAGE**：encoder 训练源与 VLA 库必须是同一组源 H5、逐文件 sha256 相同。`DATA_LINEAGE` 在 L0／L1／L2 三处、`PAIRED_RUNS_SAME_LIB` 在 P1 机械核，任何 FAIL 即 `exit≠0`、立即停工，不得 WAIVE、不得写成风险项继续。规则来源：`AGENTS.md` P-LINEAGE（`d0445ce`）；`CLAUDE.md` 项目专属补充「P-LINEAGE 的 Claude Code 落地」——主会话派发 VLA 建库／起跑子代理前，先贴 `DATA_LINEAGE` 判定行；审查代理对涉及起跑脚本／契约的改动，必须核到该闸门存在且 FAIL 即停。
- **R1 源目录只读**：`/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/` 整体不写不动（MJ 1024 计划 R6）。policy 侧只以 `--raw_dir`/`--raw-dir`/`--h5-root` 只读引用 `source/h5/`，以 `--mj-index` 只读引用 `dataset/index/`；不复制、不建 symlink 进 `v1-store/`。收尾以 build_spec 记录的 `bytes`/`mtime_ns` 核 S0。
- **R2 不改 encoder 权重**：只读 `checkpoint_epoch_105.pt` 的 `model` 键，strict、`eval()`、`no_grad()`、autocast 关、两个 TF32 开关关（与 1026 R2 相同）。
- **R3 memory 不增强**：`geometry_valid` 全 True，等价 aug_id=0。
- **R4 旧路径不变**：
  - mixed8 的入口、契约、yaml、导出驱动、runner 目录零 diff。
  - `motion_store`／`motion_client`／`pack_motion_store`／`train.py`／`compare_stage2`／`check_stage2_validation_inputs` 中的 mixed8 restore-receipt 路径与旧 layout，**语义不变、旧测试全部仍绿**；新增分支不改变既有参数缺省行为。
- **R5 只在新分支**：policy 只在 `v5-orig16` 及其 `worktree-agent-*` 子分支；MotionJEPA 只在 `v12-3decoder-memexport` 新增提交，不动 `v12-3decoder`/`main`。
- **R6 建库冻结窗口**：建库 Beta 后，从 SigLIP 起跑到 `VERIFY_MOTION=PASS`，policy 不 commit、不合并、不回流规则。原因是 SigLIP 各 shard 的 fingerprint（`finalize_checks check` 核其唯一）与 `pack_motion_store.py` 都要求 `git_commit` 唯一。窗口内主会话写 `$(git rev-parse --git-dir)/AGENTMETARULES_HOLD`（`freeze` + 锚点 sha），窗口结束删除。`scan_manifest` 与 `hash-inputs` 不写 git HEAD，可在冻结前跑。
- **R7 训练期代码冻结**：从训练 Beta 起，到两条 G8 与最终导出完成，`src/`、`pyproject.toml`、`uv.lock` 不动。被跟踪的 `src/ scripts/ packages/` 文件 `chmod a-w`，原权限记到 `v1-store/o16-scripts/chmod-before.txt`；结束按记录恢复并打 `PERM_RESTORED=OK`。
- **R8 tmux**：只用前缀 `o16-`，完整名单见四节。清理只按清单逐个 `tmux kill-session -t '=名'`，删前删后各 `tmux ls`；禁止 `kill-server` 等全局杀法。
- **R9 不 resume**：训练崩溃、NaN、OOM 一律停下问用户。不带 `--force`、不 `overwrite`；新 run 目录、新库目录、新 bucket 起跑前确认不存在。
- **R10 W&B 与凭据**：W&B 用 hongzefu 账号，entity `hongzefu-university-of-michigan`，project `robomme-framesamp`。凭据 `v1-store/secrets/wandb.env`、`v1-store/secrets/hf.env` 只经 `set -a; . file; set +a` 读，不进日志、档案或命令行。
- **R11 存储边界**：一切持久化落 `/scratch/hongze/` 下。缓存 `UV_CACHE_DIR=/scratch/hongze/.cache/uv`；HF 缓存沿用 `paths.sh` 的 `$V1_STORE/cache/hf`，env 不覆盖 `HF_HOME`（mixed8 oracle 首跑因覆盖 `HF_HOME` 失败的教训）。
- **R12 删除纪律**：第一档删除须经裁决 ⑦，并在核到对应 HF 判定行原文后由主会话执行。删前列清单、`ls -ld` 确认是实体目录，显式列目录、不 glob。第二档需另行裁决。
- **R13 不 WAIVE**：G2 及一切闸门没有例外通道。本计划禁用 `scripts/dataset/wan_stage2/accept_stage2_g2_exception.py`；fresh 分支校验器拒绝 `status=WAIVED`，契约另核 `CHECK_G2_NOT_WAIVED`。
- **R14 子代理环境**：worktree 内只跑 ≤5 分钟的 CPU 测试。
  - policy：`UV_PROJECT_ENVIRONMENT=/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv PYTHONPATH=<worktree>/src uv run --no-sync python -m pytest <定向> -q`，先打印 `mme_vla_suite.__file__` 确认指向 worktree。
  - MJ：`UV_PROJECT_ENVIRONMENT=/scratch/hongze/MotionJEPA/.venv PYTHONPATH=<worktree>/src uv run --no-sync python -m pytest <定向> -q`。
  - GPU 验收在合并后由运行代理在主检出跑。

## 一、逐文件改动清单

所有锚点用符号名（`AGENTS.md` 第 9 条），以 `v4-256concat1` HEAD（代码同 `b25fd99`）为准。

### A. MotionJEPA（S6，分支 `v12-3decoder-memexport`，worktree 内新提交）

| 文件 | 锚点 | 改什么 | 验收 |
|---|---|---|---|
| `scripts/stage2_memory_export.py` | 常量 `FULL_COUNTS = (4995, 65419)`；`main` 中 `require_equal(counts, FULL_COUNTS, "mixed8 全表段数与行数")` | 新增 `--expected-counts <segments>,<rows>`，full scope 必填；不传时报错，不回落到 mixed8 常量。mixed8 以 `4995,65419` 调用时语义不变 | `tests/test_stage2_memory_export.py` 新增三个用例：错计数拒绝、正确计数通过、smoke scope 不受影响；冒烟出 `STAGE2_EXPORT=PASS scope=smoke …` |

### B. policy

**S1 `scripts/dataset/check_stage2_validation_inputs.py`（G0 原版模式）**
- 新增参数 `--h5-root <dir>`、`--input-manifest <LIB/meta/input_manifest.json>`、`--mj-index <dir>`。三者须同时给出；任一缺失时走原 mixed8 路径，语义不变。
- **demo 帧**：原版模式不读 `control/scope.json`／`run_manifest.json`，改为 `h5py.File(<h5-root>/<ep.h5_file>, "r")[f"episode_{raw_ep_idx}/timestep_{t}/obs/front_rgb"]`。H5 的 size 与 sha256 须等于 input_manifest 登记值（sha 现场全量计算并在进程内缓存）。
- **exec 帧**：仍读 `source/data/<sample>.pkl`。
- **MJ 对齐**：从 `<mj-index>/segments.parquet` 取每集 demo 末端 `D` 与 exec 截止 `C`（首个 `is_completed` 帧）。
  - 逐集断言 policy `exec_start_idx == D`；
  - `select_windows` 的 exec 候选窗限定 `start+33 ≤ C`；
  - demo 全真实窗本身就在 `[0,D)` 内；
  - 每个全真实窗写出 `mj_window: {task, episode, segment, start}`，补帧窗写 `null`。
  - 原有合同不变：64 窗、16 任务、≥16 个 demo 尾窗、grid16 旧格（模块常量 `GRID_STRIDE = 16` 不变，对应路径 B）。
- **测试**：
  - 新增 `tests/dataset/test_check_stage2_validation_h5.py`：合成 2 任务 H5 + parquet 夹具；验证补帧窗 SHA、H5 sha 不符拒绝、`D` 不等拒绝、exec 窗全部满足 `start+33 ≤ C`。
  - 既有 `tests/dataset/test_stage2_validation_inputs.py` 原样通过。
- 判定行沿用：`RAW_WINDOWS_READY=PASS windows=64 tasks=16 demo_tail=<n> sha_mismatches=0`。

**S1 `scripts/training/check_data_lineage.py`（新）**
- 可导入函数 `check_lineage(encoder_ckpt, expected_ckpt_sha256, vla_input_manifest, mj_source_manifest) -> (ok: bool, line: str, detail: dict)`，供 S4 契约与 runner 调用。
- 子命令 `lineage --encoder-ckpt <pt> --expected-ckpt-sha256 <64hex> --vla-input-manifest <json> [--mj-source-manifest /scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/source_manifest.json]`，步骤如下：
  1. 现场计算 ckpt sha256，与期望值比较。
  2. 读 ckpt 同目录 `config.yaml` 的 `data.data_root`（ep105 实为 `/scratch/hongze/dataset-rgb-gtxyz-16views-stride2-v1`）。
  3. 读 `<data_root>/control/build_spec.json` 的 `identity.source_pin.files[]`（每项 `task/path/bytes/sha256/mtime_ns`），得到 encoder 源集合 `{basename(path): (bytes, sha256)}`。`identity.source_pin.sha256` 是 source_receipts 的哈希，不参与比较。
  4. 读 MJ `source_manifest.json` 的 `files.<Task>.h5.{file,sha256,size}`，要求与第 3 步集合完全相同；取其 `pin_sha256` 作 `encoder_source_pin`（只记录）。链条为：`stage2_pool4.yaml` 的 data_root → 1024 计划 R6 源目录 → source_manifest。
  5. 读 VLA `input_manifest.json` 的 `files{<name>:{size,sha256}}`，按文件名对齐，集合必须相等。多、少、sha 不同、size 不同都计入 mismatch。
  6. `vla_input_pin` = 按文件名排序的 `[name, size, sha256]` 列表 canonical JSON 的 sha256，只记录。**不与 `encoder_source_pin` 比值，同源只以第 4、5 步的集合相等判定。**
  - 输出：通过时打 `DATA_LINEAGE=PASS encoder_ckpt=<sha8> encoder_source_pin=<pin_sha256> vla_input_pin=<sha> files=16 mismatches=0`、退出码 0；否则打 `DATA_LINEAGE=FAIL … mismatches=<k>`，逐项列出差异，退出码 3。
- 子命令 `paired --encoder-ckpt <pt> --expected-ckpt-sha256 <64hex> --argv-a <file> --argv-b <file>`：两份文件分别是两条 run 的最终 `TRAIN_CMD=` 行。
  1. 对每条 argv 取 `--dataset-path` 的父目录作库根，分别以 `<库根>/meta/input_manifest.json` 调 `check_lineage`，**两条都须 `DATA_LINEAGE=PASS`**，并各打一行。
  2. 比较两条的库根、`--data.assets.assets-dir`、motion yaml `store_path` 的父库（若有），必须全部相同。
  3. 通过才输出 `PAIRED_RUNS_SAME_LIB=PASS lib=<库名> runs=2`，否则 FAIL 并以退出码 3 退出。
- **测试** `tests/training/test_check_data_lineage.py`（新）：
  - 合成 build_spec（`identity.source_pin.files[]` 形态）、source_manifest、input_manifest 三方一致时 PASS；
  - 改一个 sha、少一个文件、多一个文件、size 不同、build_spec 顶层误放 `source_pin` 但 `identity` 下缺失，均 FAIL；
  - paired：任一侧血缘 FAIL 即 FAIL，父库不同也 FAIL。

**S2 来源链 fresh-extract 模式与 G2 fresh 校验**
- **`scripts/dataset/pack_motion_store.py`**
  - `cmd_pack` 中 stage2 分支「必须 `--restore-receipt`」改为二选一：`--restore-receipt`（历史 mixed8，原逻辑）或新参数 `--latents-origin fresh`。
  - fresh 模式下，VAE 身份取 `wan-latents/metadata.json` 记录的当前 `scripts/dataset/wan/SOURCE_PIN.json` 与 VAE 权重 sha，不经 `load_vae_compat`。
  - `check_raw_sources` 仍要求 `oracle/wan-mj/vae_report.json`（B7），并要求 input_manifest、Wan metadata、manifest 三者的 `raw_dir` 与 `--raw-dir` 一致。
  - `--expected-ckpt-sha256` 在 fresh 模式同样必填。
  - `gather_provenance` 的 stage2 分支写 `stage2_compat.latent_origin = {"kind": "fresh-extract", ...}`；证据键由 `restore_receipt` 换成 `wan_extract`，指向 `wan-latents/metadata.json` 与 B7 oracle 报告的 sha。
- **`src/mme_vla_suite/datastore/motion_store.py`**
  - stage2_compat 校验按 `latent_origin.kind` 分支。当前代码 `require_sha256(compat.get("restore_receipt_sha256"), …)` 与 `for gate in ("G0", "G2", "G3", "restore_receipt", "SOURCE_PIN")` 两处需分支；mixed8（无 `latent_origin`，或 kind=historical-restore-receipt）语义不变。
  - fresh 分支要求 `G0/G2/G3/wan_extract/SOURCE_PIN` 五项。
  - `validate_stage2_g2_evidence` 新增关键字参数 `latent_origin="historical-restore-receipt"`（缺省即旧行为）。传 `latent_origin="fresh-extract"` 时，在**全部原有检查之上**另加以下规则：
    - **原有检查保持**：schema `stage2-validation-gate-v1`、gate G2、`checkpoint_sha256`、`samples_sha256` 等于 G0 清单 sha、SOURCE_PIN 五键、`numerics` 的 `dtype=float32/autocast=False/matmul_allow_tf32=False/cudnn_allow_tf32=False/batch=1` 与环境身份、`online_vae` 固定合同（`tf32="off"` 等）、`metrics.windows == 64`、`metrics.demo_tail == G0 的值且 ≥ 16`、`0 ≤ worst_mse ≤ 1e-6`、`0 ≤ worst_max ≤ 0.01`。**一个阈值都不放宽。**
    - **新增必填**：
      - `metrics.g2a_windows == 64 − demo_tail` 且 `metrics.g2a_max_abs == 0`；
      - `metrics.g2b_windows == 64` 且 `metrics.g2b_max_abs == 0`；
      - `oracle.repo == "HongzeFu/robomme-original-16x100-wan21-front-33f-dense-v1"`、`oracle.revision == "384d5db410be713a49357bf3f55ca4f4d2f559a0"`、`oracle.sha256sums_sha256` 为 64 位 hex；
      - `oracle.numerics` 至少含 `weights="fp32"`、`batch=1`、`cudnn_conv_tf32=True`、`matmul_tf32=False`、`amp=False`，并与 MJ `provenance.json.numeric_mode` 的这几项一致。
    - **只认 PASS**：fresh 分支 `status` 只接受 `PASS`；`WAIVED` 或带 `exception` 字段一律拒绝。
- **`src/mme_vla_suite/policies/motion_client.py`**：`check_provenance` 的 stage2 证据循环（上述五键）同样按 kind 分支，并以 `latent_origin="fresh-extract"` 调用 G2 校验器；fresh 模式不读 restore receipt，VAE 的当前实现即在线实现，不需要历史→当前映射。
- **`scripts/training/train.py::init_history_config`**：stage2 快照按 kind 保存对应证据集（`EVIDENCE_FILES` 的键集合随 kind 变化）；`motion_snapshot.ready.json` 机制不变。
- **`scripts/dataset/wan_stage2/compare_stage2.py`**：把写死的 `meta.num_rows != 65419` 改为必填参数 `--expect-rows <n>`；mixed8 调用传 65419，语义不变。
- **测试**：
  - 新增 `tests/dataset/test_pack_motion_stage2_fresh.py`：两非空段夹具跑 pack(fresh) → compare 记录 → verify → loader。
  - 修改既有 `tests/dataset/test_pack_motion_stage2.py`、`tests/dataset/test_stage2_sampled_compare.py`（`--expect-rows`）、`tests/dataset/test_stage2_numeric_contract.py`（fresh 校验：缺 g2a/g2b、max_abs≠0、oracle.revision 不符、WAIVED、mse 超阈值均拒绝；合法 fresh 记录通过）、`tests/datastore/test_motion_store_stage2.py`、`tests/training/test_stage2_snapshot.py`、`tests/policies/test_motion_client_stage2.py`，加入 fresh 分支用例。
  - 其余 `tests/dataset/test_stage2_*.py` 原样全绿。
- **实施提示**：`restore_receipt` 的全部引用点以 `grep -rn restore_receipt src scripts --include=*.py` 为准。当前命中 `motion_store.py`、`motion_client.py`、`pack_motion_store.py`、`train.py` 及 mixed8 专用脚本，只改前四个。

**S3 `scripts/dataset/wan_stage2/orig16_g2.py`（新，在 torch 子环境 `scripts/dataset/wan_stage2` 运行）**
- **子命令 `bins`**：`bins --g0 <VALID>/G0.json --mj-index <dir> --out <G2D>/G2-bins.txt`，从 G0 全真实窗的 `mj_window` 列出所需 bin 文件名（`wan_chunk_latents/<Task>_ep<N>_<seg>.bin`，去重），并核对每个窗都在 `windows.parquet` 中。
- **子命令 `compare`**：`compare --g0 <VALID>/G0.json --mj-oracle <G2D>/mj-oracle --raw-dir <RAW> --source <LIB/source> --latents <LIB/wan-latents> --ckpt <CKPT> --expected-sha256 <CKPT_SHA> --out <VALID>/G2.json`。
  - **原帧来源**：64 窗原帧按 G0 清单的 `frame_ids` 与 `sources` 重新读取，并复核每窗 `input_frames_sha256`。
  - **G2a**：对 G0 中 `pad_frames == 0` 的全部窗，调用 `wan_motion_infer.py::encode_chunk`（与生产同一函数），在 **MJ provenance 数值模式**下重编：fp32、batch 1、`cudnn.allow_tf32=True`、`matmul.allow_tf32=False`、benchmark 关。结果与 MJ bin 对应 block 逐位比较，得 `g2a_max_abs`。比较前逐 bin 核对 MJ `SHA256SUMS`；若环境（torch/cuDNN/driver/GPU）与 MJ `provenance.json.numeric_mode` 不符，直接 FAIL。
  - **G2b**：对 G0 全部 64 窗，先调 `pin_numerics()` 进入生产模式（TF32 关），再用 `encode_chunk` 重编，与本库 `wan-latents/<段>.bin` 的 `source_grid` 块逐位比较，得 `g2b_max_abs`。其中 ≥16 个是 demo 补帧尾窗，因此这一段证明了补帧链路。
  - **信息行**：读出本库生产 latent 对应 G2a 窗的 block 与 MJ bin 比较，只输出 `WAN_NUMERICS_DRIFT=INFO windows=<g2a> norm_mse=<x> max_abs=<y>`，写入 G2.json 的 `drift` 字段，不判 PASS/FAIL。
  - **输出 G2.json**：
    - 固定字段：schema `stage2-validation-gate-v1`、gate `G2`、`producer="orig16-g2-v1"`。
    - 来源与环境：`checkpoint_sha256`、`samples_sha256 = sha256(G0.json)`；`source_pin` 取 `scripts/dataset/wan_stage2/SOURCE_PIN.json` 的五键；`numerics` 与 `online_vae` 记录生产口径（与现有校验器字段完全一致）。
    - `metrics{windows:64, demo_tail:<G0 值>, worst_mse, worst_max, g2a_windows, g2a_max_abs, g2b_windows:64, g2b_max_abs}`：两段都逐位时 `worst_mse = worst_max = 0`。
    - `oracle{repo, revision, sha256sums_sha256, bins[], numerics}`、`drift`。
    - `status` 只能是 PASS 或 FAIL。
  - **判定行**：`WAN_EQUIV=PASS windows=64 demo_tail=<n> g2a_windows=<n> g2a_max_abs=0 g2b_windows=64 g2b_max_abs=0 worst_mse=0 worst_max=0 oracle=mj-stride1@384d5db4`。
- `validate_stage2.py` 与 `accept_stage2_g2_exception.py` 不改、不用。
- **测试**：
  - `scripts/dataset/wan_stage2/tests/test_orig16_g2.py`（新，CPU）：bin 清单去重、SHA256SUMS 不符拒绝、G2.json 字段齐全、`samples_sha256` 绑定 G0。
  - `tests/dataset/test_orig16_g2_validator_compat.py`（新）：用 `test_orig16_g2.py` 的夹具产出 G2.json，调用 S2 的 `motion_store.validate_stage2_g2_evidence(..., latent_origin="fresh-extract")` 必须通过；篡改任一 metrics 必须被拒。
  - 该兼容测试依赖 S2：S3 在第一批开发时先只跑第一个测试；S2 合入后，主会话用 SendMessage 续接 S3，在其子分支上 `git merge` 合入当前 `v5-orig16` HEAD（只把工作分支并入自己的子分支，不 rebase、不动工作分支），再跑兼容测试，得到 `G2_VALIDATOR_COMPAT=PASS` 后才进入 S3 合前审。

**S4 训练入口、契约、yaml、512 runner（第二批，从含 S1/S2 的 HEAD 切出）**
- **新 yaml** `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion-stage2-orig16.yaml`：
  - motion 节之前与 `perceptual-framesamp-modul-8frame-8x8-motion-stage2-mixed8.yaml` 逐字节相同；
  - motion 节只改两处：`store_path: v1-store/datasets/16task-orig100-1600ep-<pin8>/motion-stage2-1024`，以及 `online_gpu: 1`（训练不读该键；与 G7/G12 sidecar `--online-gpu 1` 及 `policy_config.py` 缺省值一致，评估部署可在调用时覆盖）；
  - `checkpoint_sha256` 直接写 `27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035`。
- **新入口 `scripts/training/run_orig16_stage2_1024_80k.sh`**：复制 `run_mixed8_stage2_1024_80k.sh`，改动如下：
  - `HC`、`PARENT`、`ASSETS_DIR` 指向原版库；
  - 正式 run 名 `orig100-m8x8-modul-motion-stage2-1024-b128-80k`，短测前缀 `o16-smoke-s2-`；
  - `CUDA_VISIBLE_DEVICES=4,5,6,7`；`TRAIN_ARGS` 增加 `--fsdp-devices 4 --num-workers 8`；
  - `MMEVLA_FRAMESAMP_SOURCE`/`_MANIFEST` 指向新库；
  - 调用新契约。契约内含 `CHECK_ENCODER_DATA_LINEAGE`，契约 FAIL 时 `body` 在 `set -e` 下退出，不 exec 训练。
- **新契约 `scripts/training/check_orig16_stage2_1024_launch.py`**：复制 mixed8 契约后修改。
  - `HISTORY_CONFIG` 换新 yaml，`FRAME_CONFIG` 不变。
  - `Contract`：
    - `parent_rel/assets_rel` 指向新库；
    - `norm_sha256/episode_manifest_sha256/input_manifest_sha256/canonical_manifest_sha256/motion_rows/train_max/eval_max` 先写占位 `__FILL_AFTER_BUILD__`（占位即 FAIL），阶段 3 由主会话单独 commit 回填实测值；
    - `fsdp_devices=4`、`num_workers=8`、`gpus="4,5,6,7"`。
  - `check_config`：fsdp/workers 改为「等于契约值」，不再要求等于基础配置；batch/seed/ema 仍三方相等。
  - `VALUE_FLAGS` 增加 `--fsdp-devices`、`--num-workers`，各须恰好出现一次，值分别为 4、8。
  - `CFG_RUN_NAME` 改为新 run 名与 `o16-smoke-s2-` 前缀；`MOTION_FIXED` 的 `store_path`/`online_gpu` 随新 yaml。
  - **新增必检项**：
    - `ENCODER_DATA_LINEAGE`：调用 S1 `check_lineage`，VLA 侧取 `parent/meta/input_manifest.json`，输出 `CHECK_ENCODER_DATA_LINEAGE=PASS|FAIL <DATA_LINEAGE 行>`；
    - `G2_NOT_WAIVED`：store_meta 中 G2 证据 status 必须为 PASS，并且以 `latent_origin="fresh-extract"` 重跑 S2 校验器。
  - `BUDGET_ROWS/BUDGET_TRAIN/BUDGET_EVAL` 沿用，输出 `CHECK_BUDGET_TRAIN=PASS train_max=<n> budget=96` 等；末行仍为 `LAUNCH_CONTRACT=PASS checks=<n> failed=0`。
- **新目录 `scripts/dataset/hf_export/orig100_runner/`**：
  - `_train_common.sh` 复制 mixed8 版，改两处：`NAME=16task-orig100-1600ep-<pin8>`；在 `body()` 里打印 `TRAIN_CMD=` 之后、执行 `"${CMD[@]}"` 之前（`DRY_RUN=1` 时跳过），插入 `uv run --no-sync python scripts/training/check_data_lineage.py lineage --encoder-ckpt /scratch/hongze/MotionJEPA/runs/stage2-full144-1025-a/checkpoint_epoch_105.pt --expected-ckpt-sha256 27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035 --vla-input-manifest "$DS/meta/input_manifest.json" || exit 3`。判定行随 `tee` 进 run 日志。
  - `train-512-80k.sh` 同 mixed8 版，`HC=perceptual-framesamp-modul-8frame-8x8.yaml`。
- **测试** `tests/training/test_check_orig16_stage2_launch.py`（新）：
  - 合成 Contract 跑通；
  - 以下情形均 FAIL：占位未回填、错 GPU、缺 `--fsdp-devices`、G2 为 WAIVED、血缘 mismatch；
  - 三个新 sh 过 `bash -n`；
  - `DRY_RUN=1` 打印的 512 `TRAIN_CMD=` 含 orig 库三处路径；
  - 用伪造 input_manifest 运行 512 runner 的血缘段时，以退出码 3 退出、不起训练（以 stub 替换 train.py 验证）。

**S5 上传驱动（第二批；全部新建，mixed8 驱动零改动）**
- **训练库**：
  - 新增 `scripts/dataset/hf_export/run_orig100_trainset_export.sh`（复制 `run_mixed8_200_trainset_export.sh`，改 `NAME`，`BUCKET_ID=HongzeFu/robomme-16task-orig100-1600ep-trainset-20261027-v1`）、`orig100_trainset_bucket_README.md`、`hygiene_allowlist_orig100_trainset.json`。
  - 上传范围显式列入 `motion-stage2-1024/`、`motion-tokens-stage2/`、`oracle/`，并在回读清单里计数（吸取 mixed8 漏传这两个子目录的教训）。
- **512**：
  - `_orig100_vla_ckpt_export_common.sh`：复制 mixed8 common，`NAME` 换 orig 库，走 `EXPECT_MOTION=false` 路径。
  - `run_orig100_vla_512_80k_ckpt_export.sh`：`RUN_NAME=orig100-m8x8-modul-b128-80k`、`BUCKET_ID=HongzeFu/robomme-vla-orig100-modul-512-80k-v1`、`EXPORT_NAME=hf-ckpt-orig100-vla-512-80k`。
  - 配套 `orig100_vla_512_80k_bucket_README.md`、`hygiene_allowlist_orig100_vla_512_80k.json`。
- **stage2**：
  - `_stage2_orig16_vla_ckpt_export_common.sh`：复制 `_stage2_vla_ckpt_export_common.sh`，并做以下修改：
    - 常量 `RUN`、`EXPORT_NAME=hf-ckpt-orig100-vla-stage2-1024-80k`、`VLA_BUCKET=HongzeFu/robomme-vla-orig100-modul-motion-stage2-1024-80k-v1`、`DATASET`、`HISTORY` 换新值；
    - `EVIDENCE_FILES` 键集合按 S2 fresh 模式改为 `G0/G2/G3/wan_extract/SOURCE_PIN`；
    - 删去 70k 停止专用常量；
    - `STEPS` 保持 5000…75000 + 79999。
  - `run_orig16_stage2_1024_ckpt_inc.sh`：训练期增量上传。
  - `run_orig16_vla_stage2_1024_80k_ckpt_export.sh`：最终导出。
  - 配套 `orig16_stage2_vla_bucket_README.md`、`hygiene_allowlist_orig16_stage2_vla.json`。
- **512 训练期增量**：沿用 10-17 的非 git 脚本 `v1-store/m8v-scripts/m8v-ckpt-inc.sh`，参数为 `<RUN> <BUCKET> <EXPORT> <ALLOWLIST> <训练 tmux>`，与正式驱动共用 `logs/uploaded/` 标记根。主会话将其复制为 `v1-store/o16-scripts/o16-ckpt-inc.sh`，内容不改。
- **测试** `scripts/dataset/hf_export/test_orig16_export_drivers.py`（新）：三套驱动常量与 run/bucket/库名一致；`bash -n` 通过；`SMOKE=1` 合同夹具通过；mixed8 驱动文件对 BASE 零 diff。

## 二、子代理分配表

**基线**：
- `BASE1`：主会话合入本计划后的 `v5-orig16` HEAD（第一批 S1/S2/S3 的基线；开工时记录 40 位 sha）。
- `BASE2`：S1、S2 合入并 push 后的 `v5-orig16` HEAD（第二批 S4/S5 的基线）。
- MJ 侧 BASE = `v12-3decoder-memexport` HEAD（当前 `b5d23a5`）。

**模型与隔离**：写入型 `model: "opus"` + `isolation: "worktree"`；审查 `model: "sonnet"`，只读；探索 `model: "haiku"`，只读；运行型 `model: "opus"`，在主检出运行、不加 worktree、不改代码。

| 编号（批次） | 目标 | 可写集合 | 禁触 | 依赖／接口契约 | 合并顺序 | 验收命令与判定行（环境） | 资源 |
|---|---|---|---|---|---|---|---|
| E1（haiku，按需） | 只读核实：`check_stage2_checkpoint.py --mode delivery --delivery-step` 是否支持 79999；MJ 抽取 CLI 读 policy 新生成 `wan-latents/metadata.json` 的字段要求 | 无 | 全部写入 | 结论交主会话，作 S4/S6 与 runbook 输入 | — | 报告原文摘录 | CPU |
| S6（opus，MJ worktree，第一批） | MJ 抽取计数参数化 | `scripts/stage2_memory_export.py`、`tests/test_stage2_memory_export.py` | 其余 MJ 文件、`runs/` | — | 1，主会话 push MJ 分支 | MJ R14 环境 pytest；`STAGE2_EXPORT=PASS scope=smoke` 夹具 | CPU |
| S1（opus，第一批） | G0 原版模式；血缘与配对检查 | `scripts/dataset/check_stage2_validation_inputs.py`、`scripts/training/check_data_lineage.py`（新）、`tests/dataset/test_check_stage2_validation_h5.py`（新）、`tests/dataset/test_stage2_validation_inputs.py`、`tests/training/test_check_data_lineage.py`（新） | R4 所列旧语义；源目录 | 输出 `check_lineage(...)` 签名、G0 的 `mj_window` 字段 | 2 | policy R14 环境 pytest 上列测试；对真实只读文件跑 `lineage --vla-input-manifest docs/dataset-build-doc/16task-h5-scan/records/input_manifest.json` 出 L0 行 | CPU |
| S2（opus，第一批） | fresh-extract 来源链与 G2 fresh 校验 | `scripts/dataset/pack_motion_store.py`、`src/mme_vla_suite/datastore/motion_store.py`、`src/mme_vla_suite/policies/motion_client.py`、`scripts/training/train.py`、`scripts/dataset/wan_stage2/compare_stage2.py`、`tests/dataset/test_pack_motion_stage2_fresh.py`（新）、`tests/dataset/test_pack_motion_stage2.py`、`tests/dataset/test_stage2_sampled_compare.py`、`tests/dataset/test_stage2_numeric_contract.py`、`tests/datastore/test_motion_store_stage2.py`、`tests/training/test_stage2_snapshot.py`、`tests/policies/test_motion_client_stage2.py` | `scripts/dataset/wan/**`、`accept_stage2_g2_exception.py` | 证据键 `G0/G2/G3/wan_extract/SOURCE_PIN`；`validate_stage2_g2_evidence(..., latent_origin=)` | 3 | `tests/dataset/test_stage2_*.py`、`tests/datastore/`、`tests/training/test_stage2_snapshot.py`、`tests/policies/test_motion_client_stage2.py` 全绿 | CPU |
| S3（opus，第一批；S2 合入后续接） | G2 两段对拍 | `scripts/dataset/wan_stage2/orig16_g2.py`（新）、`scripts/dataset/wan_stage2/tests/test_orig16_g2.py`（新）、`tests/dataset/test_orig16_g2_validator_compat.py`（新） | `validate_stage2.py`、`accept_stage2_g2_exception.py`、`scripts/dataset/wan/**` | 读 G0 的 `mj_window`；产出符合 S2 fresh 校验器的 G2.json | 4 | wan_stage2 子环境 CPU pytest；S2 合入后在子分支并入 `v5-orig16` HEAD 跑兼容测试，判定行 `G2_VALIDATOR_COMPAT=PASS` | CPU |
| S4（opus，第二批，BASE2） | 两条 run 入口、契约、yaml、512 runner（含血缘硬调） | 一节 S4 所列 3 个新文件 + `scripts/dataset/hf_export/orig100_runner/**`（新）+ `tests/training/test_check_orig16_stage2_launch.py`（新） | mixed8 入口／契约／yaml／runner；`preflight_train_launch.py` | 调 S1 `check_lineage`；读 S2 证据键与 fresh 校验器 | 5 | pytest；`bash -n`；`DRY_RUN=1` 打印；伪造 manifest 时 runner 以退出码 3 退出 | CPU |
| S5（opus，第二批，BASE2） | 三套上传驱动 | 一节 S5 所列新文件 + `scripts/dataset/hf_export/test_orig16_export_drivers.py`（新） | mixed8 驱动、`ckpt_upload_common.py`、`scan_hygiene.py` | 依赖 S2 证据键 | 6 | pytest；`bash -n`；`SMOKE=1` 只本地 | CPU |
| RV-*（sonnet） | 每块合前只读审查 | 无 | 全部写入 | 钉死 `REVIEW_BASE`/`REVIEW_TIP`；S1/S4 涉及起跑与契约，必须核到 `DATA_LINEAGE`／`CHECK_ENCODER_DATA_LINEAGE` 存在且 FAIL 即 `exit≠0`，缺失即判 FAIL | 每块合前 | `PRE_MERGE_REVIEW=PASS\|FAIL base=<sha> tip=<sha> files=<n> commits=<n> findings=<n>` | 无 |
| X1（opus 运行） | 阶段 2 建库各段 | 只写新库、norm 目录、`v1-store/logs/o16-20261027/` | 代码、源 H5 | 四节 runbook 阶段 2 | 阶段 2 | `o16-siglip` 首行 `DATA_LINEAGE=PASS`、首个 STAGE 起跑行 | GPU 0–7／CPU |
| X2（opus 运行） | 阶段 3 抽取与闸门 | `motion-tokens-stage2/`、`motion-stage2-1024/`、`v1-store/validation/orig16-*`、日志 | 代码 | runbook 阶段 3 | 阶段 3 | G0–G6 各行 | GPU 0（G4 父进程与 sidecar 同为 GPU 0，`check_stage2_online.py` 要求二者相等） |
| X3（opus 运行） | 阶段 4–5 短测、正式、增量上传 | 两个 run 目录、exports、日志 | 代码、旧 run | runbook 阶段 4–5 | 阶段 4–5 | 起跑脚本首行 `DATA_LINEAGE=PASS`、`PAIRED_RUNS_SAME_LIB=PASS`；第 1 个 loss 有限；`INC_UPLOADED` | GPU 0–3、4–7；saveload 父 GPU 0、sidecar GPU 1 |
| X4（opus 运行） | 阶段 6 最终导出、G12 | exports、validation、日志 | 代码 | runbook 阶段 6 | 阶段 6 | G9、G12 | CPU／GPU 0（父）+1（sidecar） |
| 主会话 | 删除、L0/L1、契约常量回填 commit、Beta、只读锁、档案、handoff、合并与 push | 档案、`docs/handoff/current.json`、契约常量 | 不替运行代理改代码 | — | — | 各判定行汇总 | — |

**合并流程**（每块）：
1. 核对写集：`git diff --name-only BASE..TIP` ⊆ 可写集合，越界即 `SCOPE=FAIL`。
2. 合前审 PASS 后，`git merge --no-ff <TIP sha>`，提交号按引言块从 `commitV11.58` 依次编。
3. 合后跑定向测试，得 `POST_MERGE_REVIEW=PASS merge=<sha> tests=<结果> files=<n>`。
4. push，清理 worktree。

全部 S 块须在建库 Beta（R6 冻结起点）之前合入。

## 三、闸门总表

| 闸门 | 命令／入口 | 判定行与阈值 | FAIL 处置 |
|---|---|---|---|
| B0 | `df -B1 --output=avail /scratch`，主会话按阶段需求比较 | `DF_CHECK=PASS stage=<s> avail_bytes=<n> need_bytes=<n>`；阈值：阶段 2 起跑前 ≥1.5 T，阶段 5 起跑前 ≥0.6 T | 停，提请第二档 |
| L0 | `check_data_lineage.py lineage --vla-input-manifest docs/dataset-build-doc/16task-h5-scan/records/input_manifest.json` | `DATA_LINEAGE=PASS encoder_ckpt=27daa832 encoder_source_pin=<pin_sha256> vla_input_pin=<sha> files=16 mismatches=0` | 立即停工（R0） |
| B1 | `check_orig80k_sources.py --input-manifest $INMANI --reference-input <records>/input_manifest.json --manifest $MANI --reference-manifest <records>/episode_manifest.json`（只比 size/sha256，不比 `raw_dir`） | `INPUT_PIN=PASS files=16`、`EPISODE_IDENTITY=PASS episodes=1600 timesteps=768897 …`（余字段按代码原样） | 停 |
| L1 | 同 L0，`--vla-input-manifest $LIB/meta/input_manifest.json`；`run-o16-siglip.sh` 首行再跑一次 | 同 L0 格式，`files=16 mismatches=0`；L0 与 L1 的 `vla_input_pin` 应相同（同一文件集） | 立即停工，SigLIP 不起 |
| B2 | `run_local.py --stage siglip`（32 worker） | `STAGE_DONE stage=siglip workers=32 items=1600 elapsed=<s>`；另核 `SHARD_DONE` 恰 32 行、`_shard*of32.json` 恰 32 个 | 停 |
| B3 | `finalize_checks.py check … --spot_check 1024` | `全部检查通过 PASS` | 停 |
| B4 | `pack_framesamp_store.py verify --store $LIB/framesamp` | `VERIFY_PACK=PASS scanned=768897 mismatches=0` | 停 |
| B5 | 8×8 `verify` + `xgrid_pos_check.py` | `VERIFY_PACK=PASS scanned=768897 mismatches=0`、`IMAGE_NPY_SPOT=PASS frames=512 mismatches=0`、`MOTION_POS_XGRID=PASS t=<n> npy=64 mismatches=0` | 停 |
| B6 | `run_local.py --stage wan` | `STAGE_DONE stage=wan workers=8 items=<段数> skipped=<n> initial_complete=0 elapsed=<s>` | 停 |
| B7 | `oracle_driver.py vae`（8 片）+ `aggregate --kind vae` + `compare_wan.py latents` | `ORACLE_VAE=DONE … frame_mismatches=0 …`、`ORACLE_AGGREGATE=DONE kinds=vae shards=8`、`WAN_BITEXACT=PASS compared=<n> padded_covered=<n> frame_mismatches=0 …` | 停 |
| B8 | 主会话按 store_meta `source_spot_sha256.entries` 删 features | `FEATURES_PRUNE=PASS kept=8` | 停 |
| B9 | 主会话：`sha256sum norm_stats.json`＋`report`＋manifest sha | `DATA_IDENTITY=PASS norm_sha=<…> steps=768897 samples=476857 manifest_sha=<…>` | 停 |
| G0 | `check_stage2_validation_inputs.py --h5-root $RAW --input-manifest $INMANI --mj-index $MJIDX …` | `RAW_WINDOWS_READY=PASS windows=64 tasks=16 demo_tail=<≥16> sha_mismatches=0` | 停 |
| G1 | MJ 抽取 `--check-dropout 64` | `DROPOUT_OFF=PASS windows=64 eval_repeat_max_abs=0 train_vs_eval_max_abs=<正数>` | 停 |
| G2 | `orig16_g2.py bins` → 按 `G2-bins.txt` 下载 → `orig16_g2.py compare` → S2 fresh 校验器（pack 内与契约内再验） | `WAN_EQUIV=PASS windows=64 demo_tail=<=G0> g2a_windows=<64−demo_tail> g2a_max_abs=0 g2b_windows=64 g2b_max_abs=0 worst_mse=0 worst_max=0 oracle=mj-stride1@384d5db4`；信息行 `WAN_NUMERICS_DRIFT=INFO …` 不判 | 停，**不 WAIVE** |
| G3 | `validate_stage2.py --gate G3 --rgb-root /scratch/hongze/dataset-rgb-gtxyz-16views-stride2-v1` | `EMA_SOURCE=PASS leaves=65 state_key=model exact=1 live_negative_rejected=1`、`EXPORT_PARITY=PASS windows=32 seed=0 max_abs=0 geometry_all_true=1 ckpt_sha256=27daa832…` | 停 |
| G5 | `pack --layout stage2 --latents-origin fresh` → `compare_stage2.py --scope sampled --expect-rows <rows>` → `verify --resume --check-padding` | `PACKED_TOKEN_BYTES=PASS compared=<rows> mismatches=0`、`ENCODER_SAMPLE_BITEXACT=PASS compared=1024 total=<rows> mismatches=0 seed=0`、`VERIFY_MOTION=PASS scanned=<rows> mismatches=0 nonfinite=0 segment_set_mismatch=0 real_plus_pad_ne_33=0 exec_padded=0 pad_not_last_demo=0 pad_source_mismatch=0 demo_tail_padded=<n>` | 停 |
| G4 | `CUDA_VISIBLE_DEVICES=0 … check_stage2_online.py … --online-gpu 0`（父进程与 sidecar 同为 GPU 0，脚本要求相等） | `ONLINE_OFFLINE_PARITY=PASS windows=64 handshake=1 response_bytes=4096 encoder_max_abs=0 e2e_max_abs=<x≤1e-5> frame_ids_mismatch=0 state_machine_mismatch=0` | 停 |
| G6 | `train_es_bound.py --manifest $MANI --layout motion-stage2-1x1024-grid32-demopad17-v1 --budget 96`；契约内 BUDGET 组 | `TRAIN_ES_BOUND=PASS episodes=1600 tasks=16 max_windows=<n> budget=96 headroom=<≥0>`；契约 `CHECK_BUDGET_TRAIN=PASS train_max=<n> budget=96`、`CHECK_BUDGET_EVAL=PASS eval_max=<n> …`（数值回填进 `Contract`） | 停，不改预算 |
| L2 | 512：`orig100_runner/_train_common.sh` 启动前硬调 `check_data_lineage.py lineage`；s2：契约 `CHECK_ENCODER_DATA_LINEAGE`；两条正式起跑脚本 `run-o16-train512.sh`／`run-o16-trains2.sh` 首行 `lineage` | runner 日志与起跑脚本日志内 `DATA_LINEAGE=PASS …`；契约内 `CHECK_ENCODER_DATA_LINEAGE=PASS …` | `exit≠0`，训练不起 |
| P1 | `check_data_lineage.py paired`（内部先对两条各自 `DATA_LINEAGE=PASS`） | 两行 `DATA_LINEAGE=PASS …` + `PAIRED_RUNS_SAME_LIB=PASS lib=16task-orig100-1600ep-<pin8> runs=2` | `exit≠0`，停 |
| G7 | 512：`preflight_train_launch.py` + runner 20 步；motion：`run_orig16_stage2_1024_80k.sh`（`MAX_STEPS=20`，内含契约）；`check_stage2_checkpoint.py --mode smoke` | `PREFLIGHT=PASS n=<n>`、`LAUNCH_CONTRACT=PASS checks=<n> failed=0`、`SMOKE_4GPU=PASS run=512 steps=20 exit=0 finite_losses=20/20 peak_mem_gib=<x> limit_gib=72`、`SMOKE_4GPU=PASS run=s2 …`（主会话据日志手工写）、`SAVE_LOAD=PASS missing=0 extra=0 shape_mismatch=0 ema_mismatch=0 reference_output_max_abs=0 online_infer=1` | 停 |
| G8 | 两份训练日志尾行 | `EXIT_CODE=0`，各 16 份 ckpt（5000…75000、79999）Orbax 提交完整 | 停，不 resume |
| G9 | `run_orig100_vla_512_80k_ckpt_export.sh`、`run_orig16_vla_stage2_1024_80k_ckpt_export.sh`（先 `SMOKE=1`） | 各 `CKPT_UPLOAD=PASS steps=16 … sha256_mismatch=0 public=1` | 长尾失败按第 27 条写 `pending_decisions` |
| G10 | 引用 10-26 记录 | `MJ_CKPT_UPLOAD=PASS source_runs=2 …`（已 PASS，不重传） | — |
| G11 | 不使用 | — | — |
| G12 | `check_stage2_checkpoint.py --mode delivery --delivery-step 79999 --online-gpu 1`，父 GPU 0（参数支持度由 E1 核实） | 按代码实际判定行（70k 先例为 `CKPT_70K_READY=PASS`） | 停 |
| B10 | `run_orig100_trainset_export.sh` | `RESULT=PASS …` | 长尾，失败交用户 |
| S0 | 主会话按 build_spec `identity.source_pin.files[]` 的 `bytes/mtime_ns` 比对 16 个源 H5 | `SOURCE_RO=PASS files=16 size_mtime_unchanged=16` | 交用户 |

## 四、runbook

所有阶段脚本写成 `v1-store/logs/o16-20261027/run-<会话名>.sh`（首行 `source v1-store/logs/o16-20261027/env.sh`），日志为 `v1-store/logs/o16-20261027/<会话名>.log`。统一启动模板：

```bash
D=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/o16-20261027
tmux new-session -d -s <会话名> "set -o pipefail; PYTHONUNBUFFERED=1 bash $D/run-<会话名>.sh 2>&1 | tee $D/<会话名>.log; echo \"EXIT_CODE=\$?\" >> $D/<会话名>.log"
```

公共环境 `env.sh`（含血缘函数，各脚本按需调用）：

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/dataset/paths.sh            # V1_STORE、HF_HOME=$V1_STORE/cache/hf、HF_HUB_OFFLINE=1；不覆盖 HF_HOME
export UV_CACHE_DIR=/scratch/hongze/.cache/uv OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
P=/scratch/hongze/robomme_policy_learning_MotionJEPA
MJ=/scratch/hongze/MotionJEPA
MJW=/scratch/hongze/worktrees/vla-s2-memexport-20261026
RAW=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/source/h5
MJIDX=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/index
MJSRC=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/source_manifest.json
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
lineage() {  # 判定行进日志；FAIL 退出码 3，set -e 下整段脚本停止
  uv run --no-sync python scripts/training/check_data_lineage.py lineage --encoder-ckpt "$CKPT" --expected-ckpt-sha256 "$CKPT_SHA" --vla-input-manifest "$1" --mj-source-manifest "$MJSRC"
}
```

**Monitor 过滤词**：每份日志挂一个 Monitor，写法 `tail -n +1 -F <log> | stdbuf -oL tr '\r' '\n' | grep --line-buffered -E …`。
- 建库：`DATA_LINEAGE=|STAGE_DONE|SHARD_DONE|VERIFY_PACK=|全部检查通过|IMAGE_NPY_SPOT=|MOTION_POS_XGRID=|ORACLE_|WAN_BITEXACT=|INPUT_PIN=|EPISODE_IDENTITY=|FAIL|Traceback|Killed|EXIT_CODE=`
- 抽取与闸门：`RAW_WINDOWS_READY=|STAGE2_EXPORT=|DROPOUT_OFF=|WAN_EQUIV=|WAN_NUMERICS_DRIFT=|EMA_SOURCE=|EXPORT_PARITY=|PACKED_TOKEN_BYTES=|ENCODER_SAMPLE_BITEXACT=|VERIFY_MOTION=|ONLINE_OFFLINE_PARITY=|TRAIN_ES_BOUND=|FAIL|Traceback|EXIT_CODE=`
- 训练：`DATA_LINEAGE=|PAIRED_RUNS_SAME_LIB=|CHECK_ENCODER_DATA_LINEAGE=|LAUNCH_CONTRACT=|PREFLIGHT=|EXIT_CODE=|Traceback|out of memory|Killed|NaN|nan|step [0-9]+000`
- 上传：`INC_UPLOADED|ROOT_UPLOAD=|CKPT_UPLOAD=|RESULT=|FAIL|EXIT_CODE=`

**0. 阶段 0（主会话，T0 → T0+1 h）**
1. 核对 10-26 导出已完成：`grep -E '^CKPT_UPLOAD=PASS steps=14' v1-store/logs/vla-s2-20261026/close70-export.log`，且末行为 `EXIT_CODE=0`。未出现就挂 Monitor 等待该日志，期间不删任何东西。
2. 回填 `<pin8>`：`grep pin_sha256 "$MJSRC"`，取前 8 位回填本文全部 `<pin8>` 与 `env.sh`，作为 `docs:` 提交，在派发 S 块之前完成。
3. L0 预核：S1 合入前，由主会话用只读 Python 片段按一节 S1 的同一定义先核（读 `identity.source_pin.files[]`），结果贴进汇报。S1 合入后补跑正式 L0：`lineage "$REF/input_manifest.json"`。
4. 第一档删除（R12，须裁决 ⑦）：
   - 删除对象：`v1-store/restore/mixed8-200-vla-latents-20261021-a`、`v1-store/exports/hf-ckpt-mixed8-200-vla-stage2-1024-80k`、`v1-store/train-runs/mme_vla_suite_b128_80k/mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k`。
   - 依据：恢复副本先跑 `hf buckets list HongzeFu/robomme-16task-mixed8-200-trainset-20261016-v1 -R` 并核其 `RESULT=PASS` 记录；后两项以 `CKPT_UPLOAD=PASS steps=14` 为据。
   - 操作：删前 `ls -ld`、`du -sb` 列清单，删后打 `DF_CHECK`。
5. 状态机（须裁决 ⑨）：在 `docs/handoff/current.json` 登记 `orig16-vla-build`、`orig16-vla-train-4plus4`，状态 `planned`；给 `mixed8-vla-stage2-1024-train` 的 `state_reason` 补「数据错配，结果仅作历史记录」。

**1. 阶段 1（T0+1 → T0+8 h）**

按二节派发：第一批 S6、S1、S2、S3（可同时派 E1）；S1、S2 合入后，从 BASE2 派第二批 S4、S5；S2 合入后再续接 S3 跑兼容测试。合入顺序 S6 → S1 → S2 → S3 → S4 → S5。

以下 CPU 前置不依赖新代码，与派发并行。`scan_manifest` 与 `hash-inputs` 不写 git HEAD，不受冻结影响：

```bash
# o16-hash（CPU，约 20 min：pub16 hash16 实测 1196 s）
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/scan_manifest.py build --raw_dir "$RAW" --episodes-per-task 100 --num_shards 1 --out "$MANI"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs --raw_dir "$RAW" --out "$INMANI"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/check_orig80k_sources.py --input-manifest "$INMANI" --reference-input "$REF/input_manifest.json" --manifest "$MANI" --reference-manifest "$REF/episode_manifest.json"
```

S1 合入后，主会话跑正式 L0 与 L1（`lineage "$INMANI"`）。

**2. 阶段 2 建库（X1 运行，T0+8 → T0+12.5 h）**

前置：
- 全部 S 块已合入并 push；
- 建库档案 `docs/dataset-build-doc/16task-orig100-1600ep-<pin8>/launch.md` 已提交；
- 已打建库 Beta `commitV11.64Beta`，且 `git status --short` 为空；
- 已写 `AGENTMETARULES_HOLD`（freeze）；
- `DF_CHECK stage=build` ≥1.5 T。

```bash
# run-o16-siglip.sh（首行硬检 L1；32 worker = 8 卡 × 4，XLA_PYTHON_CLIENT_MEM_FRACTION=0.15；mixed8 实测 4 worker 因 GPU 中断不送达过慢）
set -euo pipefail; source "$D/env.sh"
lineage "$INMANI"                           # 打印 DATA_LINEAGE=PASS …；FAIL 退出码 3，SigLIP 不起
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

**3. 阶段 3 抽取与闸门（X2 运行，T0+12.5 → T0+14.5 h）**

```bash
# o16-g0
uv run --no-sync python scripts/dataset/check_stage2_validation_inputs.py --manifest "$MANI" --latents "$LIB/wan-latents" --source "$LIB/source" --h5-root "$RAW" --input-manifest "$INMANI" --mj-index "$MJIDX" --out "$VALID/G0.json"
# o16-s2export（MJ worktree，GPU 0；含 G1；先 --max-segments 4 冒烟到独立临时输出）
cd "$MJW" && CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/stage2_memory_export.py --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --latents "$LIB/wan-latents" --manifest "$MANI" --out "$LIB/motion-tokens-stage2" --grid-subsample 2 --batch 1 --check-dropout 64 --expected-counts <段数>,<行数>
# o16-g2：bins → 下载（unset HF_HUB_OFFLINE 只在本段；合计 ≤15 GB；repo 类型与 CLI 写法实施时按 hf 版本确认）→ compare
UV_PROJECT_ENVIRONMENT="$V1_STORE/venvs/wan-stage2" uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/orig16_g2.py bins --g0 "$VALID/G0.json" --mj-index "$MJIDX" --out "$G2D/G2-bins.txt"
# hf download HongzeFu/robomme-original-16x100-wan21-front-33f-dense-v1 --repo-type dataset --revision 384d5db410be713a49357bf3f55ca4f4d2f559a0 --include <G2-bins.txt 所列> provenance.json SHA256SUMS --local-dir "$G2D/mj-oracle"
CUDA_VISIBLE_DEVICES=0 UV_PROJECT_ENVIRONMENT="$V1_STORE/venvs/wan-stage2" uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/orig16_g2.py compare --g0 "$VALID/G0.json" --mj-oracle "$G2D/mj-oracle" --raw-dir "$RAW" --source "$LIB/source" --latents "$LIB/wan-latents" --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --out "$VALID/G2.json"
# o16-g3
CUDA_VISIBLE_DEVICES=0 UV_PROJECT_ENVIRONMENT="$V1_STORE/venvs/wan-stage2" uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/validate_stage2.py --gate G3 --samples "$VALID/G0.json" --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --rgb-root /scratch/hongze/dataset-rgb-gtxyz-16views-stride2-v1 --out "$VALID/G3.json"
# o16-s2pack（CPU pack，内含 G2 fresh 校验 → GPU 抽样认证 → CPU verify；VERIFY_MOTION=PASS 后由主会话删除 HOLD）
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_motion_store.py pack --layout stage2 --latent-grid-subsample 2 --latents-origin fresh --manifest "$MANI" --tokens "$LIB/motion-tokens-stage2" --latents "$LIB/wan-latents" --out "$LIB/motion-stage2-1024" --encoder-run-dir "$ENC" --raw-dir "$RAW" --expected-ckpt-sha256 "$CKPT_SHA" --stage2-validation "$VALID" --stage2-source-pin scripts/dataset/wan_stage2/SOURCE_PIN.json
CUDA_VISIBLE_DEVICES=0 UV_PROJECT_ENVIRONMENT="$V1_STORE/venvs/wan-stage2" uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/compare_stage2.py --store "$LIB/motion-stage2-1024" --latents "$LIB/wan-latents" --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --scope sampled --batch 1 --expect-rows <行数>
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_motion_store.py verify --store "$LIB/motion-stage2-1024" --resume --check-padding
# o16-g4（父进程与 sidecar 同为 GPU 0，check_stage2_online.py 要求二者相等；核实际物理卡号）
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/training/check_stage2_online.py --store "$LIB/motion-stage2-1024" --samples "$VALID/G0.json" --history "$P/src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion-stage2-orig16.yaml" --online-gpu 0 --out "$VALID/G4.json"
# G6
uv run --no-sync python scripts/training/tests/train_es_bound.py --manifest "$MANI" --layout motion-stage2-1x1024-grid32-demopad17-v1 --budget 96
```

`compare_stage2.py`/`validate_stage2.py` 其余必填参数以 `--help` 为准，实施时照抄 1026 runbook 原文、只换路径。

G6 之后主会话做三件事：
1. 复跑 L1。
2. 把实测 `norm_sha256/episode_manifest_sha256/input_manifest_sha256/canonical_manifest_sha256/motion_rows/train_max/eval_max` 回填 `check_orig16_stage2_1024_launch.py` 的 `Contract`，提交 `commitV11.63: 回填 orig16 启动契约实测常量`。其中 `train_max/eval_max` 取契约 BUDGET 组算法的实测值，并与 `TRAIN_ES_BOUND` 的 `max_windows` 互相核对。
3. push。

**4. 阶段 4 短测（X3 运行，T0+14.5 → T0+15.5 h）**

```bash
# o16-pre512
MMEVLA_FRAMESAMP_SOURCE="$LIB/source" MMEVLA_FRAMESAMP_MANIFEST="$MANI" uv run --no-sync python scripts/training/preflight_train_launch.py --repo "$P" --train-head <40 位 HEAD> --history-config perceptual-framesamp-modul-8frame-8x8.yaml --history-config-sha256 <sha256> --assets-dir "$V1_STORE/train-assets/mme_vla_suite/$NAME" --asset-id robomme --norm-stats-sha256 <B9 norm_sha> --dataset-path "$LIB/framesamp-8x8" --run-root "$V1_STORE/train-runs/mme_vla_suite_b128_80k/o16-smoke-512-<美东MMDDHHMM>" -- mme_vla_suite_b128_80k --exp-name o16-smoke-512-<美东MMDDHHMM> --data.assets.assets-dir "$V1_STORE/train-assets/mme_vla_suite/$NAME" --data.assets.asset-id robomme --dataset-path "$LIB/framesamp-8x8" --model.history-config perceptual-framesamp-modul-8frame-8x8.yaml --fsdp-devices 4 --num-workers 8
# P1：两条 runner 的 DRY_RUN／契约 TRAIN_CMD= 行写入 $VALID/argv-512.txt、argv-s2.txt；paired 内部先各自跑血缘
uv run --no-sync python scripts/training/check_data_lineage.py paired --encoder-ckpt "$CKPT" --expected-ckpt-sha256 "$CKPT_SHA" --argv-a "$VALID/argv-512.txt" --argv-b "$VALID/argv-s2.txt"
# o16-smoke512（GPU 0–3；runner 内部启动前打 DATA_LINEAGE）与 o16-smokes2（GPU 4–7；契约内 CHECK_ENCODER_DATA_LINEAGE）同时起
CUDA_VISIBLE_DEVICES=0,1,2,3 FSDP_DEVICES=4 NUM_WORKERS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false bash scripts/dataset/hf_export/orig100_runner/train-512-80k.sh o16-smoke-512-<美东MMDDHHMM> --num-train-steps 20 --no-wandb-enabled
RUN_NAME=o16-smoke-s2-<美东MMDDHHMM> MAX_STEPS=20 TRAIN_HEAD=<40 位 HEAD> FINAL_RECORD_DIR="$VALID/smoke/final-record" bash scripts/training/run_orig16_stage2_1024_80k.sh
# 两条退出后：o16-saveload（父 JAX GPU 0，sidecar GPU 1）
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/training/check_stage2_checkpoint.py --mode smoke --checkpoint <step19 绝对路径> --reference <保存前参考清单> --online-gpu 1 --out "$VALID/smoke/G7-saveload.json"
```

- 短测 ≤5 分钟且验收后，只删本轮 `o16-smoke-*` 自有 run 目录；超过 5 分钟的按诊断 run 留档。
- 主会话据日志写两行 `SMOKE_4GPU=PASS …`，含 `sec_per_step`，用来更新 39 h 估算。

**5. 阶段 5 正式 4+4（X3 运行，T0+16 → T0+55 h）**

前置：
- 两份训练档案 `launch.md` 已提交；
- 已打训练 Beta `commitV11.65Beta`，且 `git status --short` 为空；
- 已把权限记录到 `v1-store/o16-scripts/chmod-before.txt` 后 `chmod a-w`；
- `DF_CHECK stage=train` ≥0.6 T。

```bash
# run-o16-train512.sh（GPU 0–3）
set -euo pipefail; source "$D/env.sh"
lineage "$INMANI"                           # 首行 L2，FAIL 退出码 3
uv run --no-sync python scripts/training/check_data_lineage.py paired --encoder-ckpt "$CKPT" --expected-ckpt-sha256 "$CKPT_SHA" --argv-a "$VALID/argv-512.txt" --argv-b "$VALID/argv-s2.txt"
CUDA_VISIBLE_DEVICES=0,1,2,3 FSDP_DEVICES=4 NUM_WORKERS=8 bash scripts/dataset/hf_export/orig100_runner/train-512-80k.sh "$RUN512"   # runner 内部再打一次 DATA_LINEAGE
# run-o16-trains2.sh（GPU 4–7）
set -euo pipefail; source "$D/env.sh"
lineage "$INMANI"
uv run --no-sync python scripts/training/check_data_lineage.py paired --encoder-ckpt "$CKPT" --expected-ckpt-sha256 "$CKPT_SHA" --argv-a "$VALID/argv-512.txt" --argv-b "$VALID/argv-s2.txt"
RUN_NAME="$RUNS2" TRAIN_HEAD=<Beta 40 位 sha> FINAL_RECORD_DIR="$VALID/$RUNS2/final-record" bash scripts/training/run_orig16_stage2_1024_80k.sh   # 契约内 CHECK_ENCODER_DATA_LINEAGE
# o16-inc512：bash v1-store/o16-scripts/o16-ckpt-inc.sh "$RUN512" HongzeFu/robomme-vla-orig100-modul-512-80k-v1 hf-ckpt-orig100-vla-512-80k scripts/dataset/hf_export/hygiene_allowlist_orig100_vla_512_80k.json o16-train512
# o16-incs2：bash scripts/dataset/hf_export/run_orig16_stage2_1024_ckpt_inc.sh（入口内固定 RUN/bucket/EXPORT_NAME）
# o16-trainset-up（长尾，ionice）：SMOKE=1 bash scripts/dataset/hf_export/run_orig100_trainset_export.sh → 全量
```

起跑判据（两份日志都要满足）：
- 首行 `DATA_LINEAGE=PASS`；
- `PAIRED_RUNS_SAME_LIB=PASS`；
- `LAUNCH_CONTRACT=PASS` 或 runner 内 `DATA_LINEAGE=PASS` 原文在日志内；
- 第 1 个 loss 有限。

主会话给每份日志挂一个 Monitor。

**6. 阶段 6 收尾（X4 运行 + 主会话，T0+55 → T0+63 h）**

两条 G8 之后：
1. `o16-up512`：先 `SMOKE=1`，再全量跑 `run_orig100_vla_512_80k_ckpt_export.sh`。
2. `o16-ups2`：同法跑 `run_orig16_vla_stage2_1024_80k_ckpt_export.sh`。
3. `o16-g12`：`check_stage2_checkpoint.py --mode delivery --delivery-step 79999 --checkpoint <远端回读 step 79999> --expected-manifest <发布清单> --online-gpu 1`，父进程用 GPU 0。
4. 主会话收尾：按记录恢复权限并打 `PERM_RESTORED=OK`；跑 S0；写 result.md 与 records；更新状态机；按 tmux 清单逐个清理。

**本计划 tmux 清单（R8）**：`o16-hash`、`o16-siglip`、`o16-finalize`、`o16-wan`、`o16-pack4`、`o16-pack8`、`o16-oracle`、`o16-g0`、`o16-s2export`、`o16-g2`、`o16-g3`、`o16-s2pack`、`o16-g4`、`o16-pre512`、`o16-smoke512`、`o16-smokes2`、`o16-saveload`、`o16-train512`、`o16-trains2`、`o16-inc512`、`o16-incs2`、`o16-trainset-up`、`o16-up512`、`o16-ups2`、`o16-g12`。

## 五、风险登记

| 项 | 说明 | 处置 |
|---|---|---|
| G2 数值口径 | MJ 库 conv TF32 开，policy 钉死关；生产 latent 与 MJ bin 不会逐位 | G2a 在 MJ 模式下重编证同源；G2b 在生产模式下证补帧链路；生产与 MJ 的差另出 INFO 行；不放宽、不 WAIVE；口径选择列入裁决 ⑥ |
| G0 选窗受 MJ 截止约束 | exec 候选窗限定 `start+33 ≤ C` 后，个别短 exec 段可能无候选 | 合同不变（64 窗／16 任务／≥16 尾窗）；凑不满就 FAIL 交用户，不放宽 |
| encoder 训练缓存是 fp16 | encoder 训练用的 `dataset-rgb-gtxyz-16views-stride2-v1` 是整 33 帧 fp16 编码，与 VLA 生产 fp32 latent 仍有数值差（1026 G2 实测 max 0.0135） | 同数据集已满足 R-LINEAGE；这是编码精度差，不是数据混用；记入盲区，不作闸门 |
| SigLIP 慢 | 本机 GPU 完成中断不送达，JAX 每次阻塞同步约 100 ms | 直接上 32 worker（mixed8 实测 6673 s／3200 集） |
| 冻结窗口内需要修代码 | R6 窗口内不能 commit | 全部 S 块在建库 Beta 前合入；窗口内发现缺陷即停，交用户决定是否作废已产出分片重跑 |
| S3 与 S2 的接口依赖 | S3 第一批开发时还没有 S2 校验器 | S3 按一节字段表开发；S2 合入后续接跑兼容测试，`G2_VALIDATOR_COMPAT=PASS` 才进合前审 |
| 两条 run 共享 CPU／IO | 各 8 workers、共 96 核，10-17 先例可行 | 增量与训练库上传加 `ionice -c 3 nice -n 19` |
| 磁盘峰值 | features 剪枝前 ≈1.0–1.2 TB | 第一档删除后约 1.7 T；每阶段打 `DF_CHECK` |
| 只读锁误伤 | chmod 范围过大会挡住日志写入 | 只锁被跟踪的 `src/ scripts/ packages/` 文件，清单留档 |
| 21.5 epoch 过拟合 | 样本量约为 mixed8 的 1/3 | 每 5000 步存档，用户可按中间 ckpt 评估；步数列入裁决 ① |
| G2a 下载失败 | 公开 repo 网络问题 | 同一对象最多 3 次、间隔 60 s，仍失败即停下报告 |
| restore_receipt 改动面 | 四个文件都校验它 | S2 用 `latent_origin.kind` 分支；mixed8 旧测试全绿把关 |

## 六、盲区诚实清单

**仍未核实**：
- 原版库 Wan 编码吞吐未实测。2.3 h 是按 mixed8 23,119 s × 768,897/2,145,548 线性折算；窗口数、motion 行数同为估算，以 B6／G5 实测为准。
- features 剪枝前峰值 ≈465 GB 取自 pub16 量级，未在本次配置复测。
- 4 卡 stage2 单 token 速度未实测（≤1.75 s/步是估算），以 G7 短测为准；两条并行时彼此的 IO 影响也未实测。
- `check_stage2_checkpoint.py --mode delivery --delivery-step` 是否接受 79999，交 E1。
- MJ 抽取 CLI 读 policy 新生成 `wan-latents/metadata.json` 的字段要求未逐字段核对，交 E1。10-26 用的是 HF 恢复副本，与本次同 schema 但不是同一批；S6 只改计数。
- policy `exec_start_idx` 与 MJ `D`、exec 截止 `C` 的逐集对应，目前只从 README 与代码语义推断；由 G0 原版模式逐集断言。
- MJ stride-1 库 HF repo 的类型（dataset repo）与路径布局 `wan_chunk_latents/<key>.bin`，取自本机 `SHA256SUMS` 与 publish 日志；下载命令写法实施时按 hf 版本确认。
- G2a 逐位相等的前提是同型号 A100、torch 2.9.0+cu128、相同 cuDNN／driver。MJ 记录 driver `595.71.05`、cudnn `91002`；若本机现驱动不同，G2a 会 FAIL，届时交用户，不放宽。
- G4 端到端阈值 1e-5 沿用 1026，未在原版库实测。
- `online_gpu` 取 1 只影响评估部署，训练不读；评估方案另行授权。

**已核实（一致性审查 A3 解决，从盲区移出）**：
- `check_orig80k_sources.py` 不比较 input_manifest 的 `raw_dir`，只比逐文件 size/sha256。因此 MJ 路径与参考记录的 `/scratch/hongze/robomme_data_h5` 不同不影响 B1。
- `scan_manifest.py build` 与 `finalize_checks.py hash-inputs` 不写 git HEAD，只有 `finalize_checks.py check` 写。因此这两步在阶段 1 并行跑，不受 R6 冻结窗口影响。
- `build_spec.json` 的源清单键路径是 `identity.source_pin.files[]`，顶层没有 `source_pin`；`identity.source_pin.sha256` 是 source_receipts 的哈希，不等于 MJ `pin_sha256`。

## 七、状态机登记与交接

- **任务 id**：`orig16-vla-build`（阶段 0–3）、`orig16-vla-train-4plus4`（阶段 4–6，`depends_on: [orig16-vla-build]`）。
  - 跨仓库依赖：MotionJEPA `stage3-formal-training-v1`，只读复核 ep105 存在且 sha = `27daa832…`。
  - 登记由主会话在用户确认裁决 ⑨ 后写入，状态 `planned`；开工令到达时写 `authorizations[]`（`kind: go`）。
- **旧任务备注**：`mixed8-vla-stage2-1024-train` 的 `state_reason` 补一句「数据错配，结果仅作历史记录」。
- **`orig16-vla-build` 交接**：
  - 交接前必须完成：B1–B9、L0、L1、G0–G6 判定行；所有建库 GPU 任务退出；HOLD 已删。
  - 长尾：B10 训练库上传（tmux `o16-trainset-up`，日志 `v1-store/logs/o16-20261027/o16-trainset-up.log`，完成行 `RESULT=PASS`）。
- **`orig16-vla-train-4plus4` 交接**：
  - 交接前必须完成：L2、P1、G7、两条 G8、G12、`PERM_RESTORED=OK`、S0。
  - 长尾：两条 G9（tmux `o16-up512`、`o16-ups2`，完成行 `CKPT_UPLOAD=PASS`）。
- **下游**：本任务 `done` 后，另获授权的 sim 评估可以消费两条 79999（或中间 step）的 ckpt。

## 八、留档与 commit 纪律

- **档案位置**：
  - 建库档案：`docs/dataset-build-doc/16task-orig100-1600ep-<pin8>/{launch.md,result.md,records/}`。
  - 训练档案：`docs/training-doc/orig100-m8x8-modul-b128-80k/` 与 `docs/training-doc/orig100-m8x8-modul-motion-stage2-1024-b128-80k/`。`launch.md` 起跑前随 Beta 提交，`result.md` 跑完回写，二者用同一小版本号配对。
  - records 只放清洗后日志、判定行（含全部 `DATA_LINEAGE`／`PAIRED_RUNS_SAME_LIB` 原文）、G0／G2 样本清单与 SHA、耗时表；不放脚本、yaml、权重、latent。
- **编号**：
  - 每块 `--no-ff` 合并占一个项目号，从 `commitV11.58` 起依次编（见引言块）；body 按 `AGENTS.md` 第 11 条六项写，含本文全部用户原话。
  - 子代理的 `sub/<编号>:` 提交原样保留。
- **Beta**：建库 Beta 与训练 Beta 各自独立；Beta 与回写结论的 commit 禁止 squash/amend。冻结期内不 commit（R6、R7）。
- **推送**：每次 commit 后裸 `git push`。`v5-orig16` 首次推送需 upstream，按裁决 ⑨ 由用户同意后执行 `git push -u origin v5-orig16`（第 11 条例外）。
- **tmux**：本轮 tmux 清单写入各 `launch.md`。

## 九、进展登记

| 时刻（美东） | 阶段 | 事件 | 判定行原文 | commit |
|---|---|---|---|---|
| | | | | |
