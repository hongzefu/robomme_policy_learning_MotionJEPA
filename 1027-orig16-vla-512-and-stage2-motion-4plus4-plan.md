# 官方 RoboMME 16×100：GPU 0–3 单条 32帧4×4＋stage2 motion 训练计划（只写改动）

> **授权**：只规划，未实施；开工需用户另行明确指令。代码锚点 policy `v4-256concat1@eb11df0`；MJ 抽取 CLI `v12-3decoder-memexport@b5d23a5`（`/scratch/hongze/worktrees/vla-s2-memexport-20261026`）。
>
> **外部锚点**：encoder `MotionJEPA/runs/stage2-full144-1025-a/checkpoint_epoch_105.pt`（sha256 `27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035`，取 `model` 键）；源 H5 `/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/source/h5/`（`Yinpei/robomme_data_h5@a5e4e25`）；Wan 缓存 `HongzeFu/robomme-original-16x100-wan21-front-33f-dense-v1@384d5db`。
>
> **用户原话**：①「修改这个计划 改为只用gpu0123 只跑512+motion 且是 32frame4*4」②「Wan 的抽取不是已经有缓存了吗」③「不考虑精度的额外不一致 只要输入是一样的就不考虑」④「修改这个计划 只说修改的部分 dataset怎么构造 model的训练配置是什么 32frame4*4+motion 需要改什么 你现在很多不改动的地方也在说」。
>
> 本文只列本次要新建或改动的东西；未提到的代码、旧 mixed8 路径一律不动。

# 第一部分（给人看）

## 一、dataset 怎么构造

新库 `v1-store/datasets/16task-orig100-1600ep-<pin8>/`，输入是官方 16 任务×100 集（1600 集、768,897 帧、exec 训练样本 476,857）。encoder 与 VLA 必须同一组源 H5，`DATA_LINEAGE=PASS files=16 mismatches=0` 在开工、建库、起跑三处核。

| 步骤 | 产物（库内子目录） | 做法 | GPU |
|---|---|---|---|
| 1 清单 | `meta/episode_manifest.json`、`meta/input_manifest.json` | `scan_manifest.py build` + `finalize_checks.py hash-inputs`，与 `16task-h5-scan` 参考清单比对 | 无 |
| 2 视觉 | `source/` → `framesamp/`（layout `framesamp-4x4-v1`） | SigLIP 16 worker（GPU0–3 各 4 个）→ finalize 复算 1024 帧 → `pack_framesamp_store.py` 打 4×4 表并全量 verify | 0–3 |
| 3 norm | `train-assets/mme_vla_suite/<库名>/` | `compute_norm_stats.py` 在新库自算 | 无 |
| 4 Wan latent | `wan-latents/`（grid16，raw `float32[9,16,32,32]`） | **不全量重抽**：从发布缓存复制 42,404 窗，只补算 2,824 缺窗（demo 尾补帧 900 + exec 越过 completed 1,924），共 45,228 窗 / 约 24.84 GiB | 只补窗用 0–3 |
| 5 motion token | `motion-tokens-stage2/` → `motion-stage2-1024/` | MJ ep105 EMA 冻结抽取，`--grid-subsample 2` 取 grid32 共 23,253 窗，每窗 `[4,256]` 按时间片拼成 `[1,1024]`；再 `pack_motion_store.py --layout stage2` | 0 |

窗口规则：33 帧、步长 32、每 segment 从 0 起算；demo 末窗真实帧≥17 重复末帧补满，<17 丢弃；exec 只收完整 33 帧并到原始 T。上面计数是索引推算，正式以新 manifest 为准。

Wan 精度：缓存窗是 FP32＋TF32 开，补窗是 FP32＋TF32 关；按用户③只要输入相同即不比数值，记 `NUMERIC_EQUIVALENCE=NOT_REQUIRED reason=user_decision_same_input`。输入身份（源 H5、帧号、像素 SHA、补帧来源）仍硬核。

## 二、model 的训练配置

| 项 | 值 |
|---|---|
| 基础配置 | `mme_vla_suite_b128_80k`，`pi05_base` 初始化，action_horizon 20，EMA 0.999，每 5000 步存 |
| 新 history yaml | `perceptual-framesamp-modul-32frame-4x4-motion-stage2-orig16.yaml` |
| 视觉记忆 | `budget=512`、`token_per_image=16`（4×4）、`num_views=1` → 32 帧；`streaming_obs_horizon=16`（增量喂包上限） |
| motion 记忆 | `source_shape=[1,1024]`、`budget=96`、`window=33`、`stride=32`、ep105 EMA 冻结 |
| 融合 | modulation，`memory_token_dim=1024`，最多 512+96=608 个记忆位置 |
| 资源 | `CUDA_VISIBLE_DEVICES=0,1,2,3`，`--fsdp-devices 4 --num-workers 8`，global batch 128 |
| 数据 | `--dataset-path $LIB/framesamp`，assets 指向新库 norm_stats |
| 步数 | 推荐 80k（≈21.5 epoch），待裁决；60k≈16.1、40k≈10.7 |
| run 名 | `orig100-m32x4x4-modul-motion-stage2-1024-b128-80k` |

步时与显存未实测，G7 短测后回填（显存上限 72 GiB）。

## 三、32帧4×4＋motion 需要改什么

| 工作块 | 改什么 | 文件 |
|---|---|---|
| 训练入口 | 新 yaml、新启动脚本（GPU0–3/FSDP4/workers8/4×4 库）、新启动契约（核 4×4 layout、512/16/1、608、GPU 范围、血缘） | 新 yaml、`run_orig16_stage2_1024_80k.sh`、`check_orig16_stage2_1024_launch.py` |
| 在线与恢复验证 | G4 支持 4×4 framesamp 表与原版 H5 读 demo 帧；checkpoint 验证支持 `--delivery-step 79999` | `check_stage2_online.py`、`check_stage2_checkpoint.py` |
| 数据血缘 | 新 `check_data_lineage.py`；G0 支持原版 H5 + MJ 索引 + 缓存收据 | `check_data_lineage.py`（新）、`check_stage2_validation_inputs.py` |
| Wan 缓存复用 | 新工具 plan/fetch/import/check：复制命中窗、只补缺窗 | `wan_stage2/orig16_wan_cache.py`（新） |
| 缓存来源合同 | 新 `latent_origin.kind=published-cache-with-local-fill`：只核输入与完整性，数值记 NOT_REQUIRED | `pack_motion_store.py`、`motion_store.py`、`motion_client.py`、`train.py`、`validate_stage2.py`、`compare_stage2.py` |
| MJ 抽取 | `--expected-counts` 替代 mixed8 硬编码计数；必填 `--input-manifest` 核源 | MJ `scripts/stage2_memory_export.py` |
| 上传 | 新训练库 / ckpt 导出驱动；`pack_and_hash.py` 加 4×4 layout 分支 | `hf_export/` 下新驱动与 allowlist |

## 四、验收（只列新增判定行）

| 编号 | 判定行 |
|---|---|
| L0/L1/L2 | `DATA_LINEAGE=PASS encoder_ckpt=27daa832 encoder_source_pin=<sha> vla_input_pin=<sha> files=16 mismatches=0` |
| C0 | `WAN_CACHE_AUDIT=PASS candidates=<n> route=published-cache-with-local-fill`（copy 42404 + fill 2824 = 45228） |
| B4 | `VERIFY_PACK=PASS scanned=768897 mismatches=0`（4×4 表） |
| B5 | `FRAME_LAYOUT=PASS frames=32 tokens_per_frame=16 budget=512`、`STAGE2_FRAME4_COMPAT=PASS total_tokens=608` |
| B6/B7 | `WAN_CACHE_IMPORT=PASS copied=<n> filled=<n> total=<n> input_mismatches=0 nonfinite=0`、`WAN_INPUT_COVERAGE=PASS windows=<n> missing=0 duplicate=0 mismatches=0` |
| G2 | `G2_INPUT_IDENTITY=PASS windows=64 mismatches=0`、`G2_NUMERICS=NOT_REQUIRED reason=user_decision_same_input` |
| G4 | `ONLINE_INPUT_IDENTITY=PASS windows=64 handshake=1 input_sha_mismatch=0 nonfinite=0` |
| G6 | `TRAIN_ES_BOUND=PASS episodes=1600 tasks=16 max_windows=<n> budget=96 headroom=<≥0>` |
| G7 | `GPU_SCOPE=PASS allowed=0,1,2,3`、`LAUNCH_CONTRACT=PASS checks=<n> failed=0`、`SMOKE_4GPU=PASS`、`SAVE_LOAD=PASS` |
| G8/G9/G12 | 训练 `EXIT_CODE=0`；`CKPT_UPLOAD=PASS steps=16`；`CKPT_DELIVERY_READY=PASS step=79999` |

## 五、步骤

| 阶段 | 内容 | 判据 |
|---|---|---|
| 1 代码 | S6→S1→S2→S3→S4→S5 实现并合入 | 各块定向测试、合前/合后审查 |
| 2 建库 | 清单 → SigLIP/4×4/norm → Wan 缓存恢复与补窗 | L1、B1–B9、C0 |
| 3 motion | G0 → ep105 抽取 → G2/G3 → pack/verify → G4/G6 → 回填契约常量 | G0–G6 |
| 4 短测 | DRY_RUN 契约 + 4 卡 20 步 + 保存恢复 | L2、G7 |
| 5 正式 | 唯一 run，增量上传 | G8 |
| 6 收尾 | 导出回读、79999 恢复验证、档案、handoff | G9、G12 |

## 六、子代理分工与合并（简述）

六块写入型子代理各在独立 worktree：S6 MJ 计数与源核；S1 血缘与 G0；S2 缓存来源合同；S3 缓存工具；S4 32帧4×4 入口/契约/在线/恢复；S5 上传驱动。S4 依赖 S2 对 `check_stage2_online.py`、`check_stage2_checkpoint.py` 的改动，从 S2 合入后的 HEAD 续做。按 S6→S1→S2→S3→S4→S5 逐个 `--no-ff` 合入，每次合前 sonnet 审固定 BASE/TIP，合后跑定向测试再 push。

## 七、待用户裁决

| # | 事项 | 推荐 |
|---|---|---|
| 1 | 训练步数 80k / 60k / 40k | 80k |
| 2 | 旧产物清理：重复导出与恢复副本 227 GB / 再加 70k ckpt 共 382 GB / 全保留 | 清 227 GB 副本，保留 70k ckpt |
| 3 | 分支与状态机登记：新 `v5-orig16` 并登记 / 沿用现分支 / 不登记 | 新分支并登记 |

# 第二部分（技术细节，供 agent 追踪）

## 〇、本次新增红线

- **R1 GPU 只 0–3**：`CUDA_VISIBLE_DEVICES`、`run_local.py --gpus`、补窗 `--gpus`、sidecar GPU 全部核；CPU 步骤 `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu`。
- **R2 P-LINEAGE**：`DATA_LINEAGE` 在 L0/L1/L2 三处跑，FAIL 即停，不 WAIVE。
- **R3 数值 NOT_REQUIRED 只覆盖精度差**：源 H5、帧号、像素 SHA、补帧、增强、权重、shape/finite、文件完整性仍硬核。
- **R4 冻结**：建库 Beta 到 `VERIFY_MOTION=PASS` 不 commit（写 `AGENTMETARULES_HOLD`）；训练 Beta 到导出完成 `src/` 只读。
- **R5 tmux 前缀 `o16-`**，日志根 `v1-store/logs/o16-20261027/`。

## 一、逐文件改动清单

### MotionJEPA（S6）

| 文件 / 锚点 | 改动 | 测试 |
|---|---|---|
| `scripts/stage2_memory_export.py::main` 中 `require_equal(counts, FULL_COUNTS, …)` | 新增 full scope 必填 `--expected-counts <segments>,<rows>`，不再回落 mixed8 常量 | 错计数拒绝、正确通过、smoke 不受影响 |
| 同文件读 latent 前 | 新增必填 `--input-manifest`：读 ckpt 旁 `config.yaml::data.data_root` → `control/build_spec.json::identity.source_pin.files[]`，与 input_manifest 逐文件比 size+sha256；输出 `MJ_CLI_LINEAGE=PASS files=16 mismatches=0`，无跳过开关 | 改 sha / 少文件 / 缺参数均失败 |

### policy

**S1**
- `scripts/training/check_data_lineage.py`（新）：`check_lineage(encoder_ckpt, expected_ckpt_sha256, vla_input_manifest, mj_source_manifest)`；子命令 `lineage`。encoder 源集合（build_spec `files[]`）= MJ `source_manifest.json::files.<Task>.h5` = VLA `input_manifest.json::files`，按文件名比 size+sha256；两个 pin 只记录不互比。退出码 PASS 0 / FAIL 3。测试 `tests/training/test_check_data_lineage.py`。
- `scripts/dataset/check_stage2_validation_inputs.py`：新增 `--h5-root --input-manifest --mj-index --cache-import-receipt`（四项须齐全）。demo 帧从 `h5[f"episode_{n}/timestep_{t}/obs/front_rgb"]` 读并核 H5 sha；从 `episodes.parquet` 取 `demo_end`/`completed_index`，断言 `exec_start_idx == D`；`start+33≤C` 为 copy，越过 C 到 T 为 fill；样本 64 窗、16 任务、≥16 demo 尾窗，copy/demo-fill/exec-fill 各≥1。测试 `tests/dataset/test_check_stage2_validation_h5.py`。

**S2 新来源 kind `published-cache-with-local-fill`**

| 文件 / 符号 | 改动 |
|---|---|
| `pack_motion_store.py::cmd_pack` 等 | 新增 `--latents-origin published-cache-with-local-fill --cache-import-receipt <json>`；按 copy/fill 分别记录数值环境 |
| `motion_store.py::validate_stage2_provenance/validate_stage2_g2_evidence` | 新 kind 证据集 `G0/G2/G3/wan_cache_import/SOURCE_PIN`；G2 只核输入、shape、finite |
| `motion_client.py::validate_stage2_evidence` | 同步新证据集，不因数值配置差异拒绝 |
| `train.py::write_stage2_motion_snapshot`、`check_stage2_checkpoint.py::verify_snapshot` | 保存/恢复同一证据集 |
| `validate_stage2.py::gate_g2/gate_g3/reference_g4` | 新增 `--cache-import-receipt`；G3 跨路径输出相等改 NOT_REQUIRED |
| `check_stage2_online.py::reference_subprocess/run` | 新 kind 去掉 `encoder_max_abs=0`/`e2e≤1e-5` 准入 |
| `compare_stage2.py` | 固定 65419 改 `--expect-rows` |
| `pack_motion_store.py::cmd_verify`、`motion_store.validate_stage2_encoder_validation` | 新 kind 消费完整性证据（packed 全量 memcmp + SHA + finite） |

收据字段：repo+revision、SHA256SUMS 自身 SHA、整文件 SHA/字节、源 H5 清单、目标 manifest、importer 源码 SHA、逐窗 copy/fill 映射与源 row/offset、33 帧输入 SHA、实际精度配置、最终文件 SHA。测试 `tests/dataset/test_pack_motion_stage2_cached.py`：仅精度不同通过并标 NOT_REQUIRED；换帧/换 H5/增强/错轴/缺映射/篡改/NaN 均 FAIL。

**S3 `scripts/dataset/wan_stage2/orig16_wan_cache.py`（新）**
- `plan --manifest --input-manifest --published-index --out`：列全 grid16 窗与 copy/fill、恢复文件清单与字节；参考 42404+2824=45228，grid32 23253。
- `fetch --plan --revision 384d5db… --out`：固定 revision 整文件恢复，逐文件核发布 SHA。
- `import --plan --cache-root --raw-dir --out --receipt --gpus 0,1,2,3`：copy 只做可逆轴序重排；fill 调 `wan_motion_infer.py::encode_chunk` 只算缺窗；输出 589,824 B/窗，核 SHA/finite/无缺重越界。
- `check --samples --receipt --raw-dir --out`：64 窗原 RGB/帧序/补帧 SHA 重算，产出 G2。
- 测试 `wan_stage2/tests/test_orig16_wan_cache.py`、`tests/dataset/test_orig16_cache_validator_compat.py`（S2 合入后续跑）。

**S4 32帧4×4＋motion 入口（从 S2 合入后 HEAD 续做）**
- 新 yaml `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-32frame-4x4-motion-stage2-orig16.yaml`：帧路取第一部分第二节的值；motion 节 `[1,1024]`/96/32/33、ep105 sha；`store_path=v1-store/datasets/16task-orig100-1600ep-<pin8>/motion-stage2-1024`；`online_gpu=1`（仅恢复验证用）。
- 新入口 `scripts/training/run_orig16_stage2_1024_80k.sh`：固定 `CUDA_VISIBLE_DEVICES=0,1,2,3`、`--fsdp-devices 4 --num-workers 8`、`--dataset-path $PARENT/framesamp`、新库 norm；`DRY_RUN=1` 只打印 `TRAIN_CMD=` 并跑契约；短测前缀 `o16-smoke-s2-`。
- 新契约 `scripts/training/check_orig16_stage2_1024_launch.py`：`store_meta.json` layout 必须 `framesamp-4x4-v1`；固定 512/16/1；fsdp/workers 与契约值 4/8 比较；`CHECK_ENCODER_DATA_LINEAGE`、`CHECK_INPUT_EQUIVALENCE`、`CHECK_FRAME_LAYOUT=PASS frames=32 tokens_per_frame=16 budget=512 total_tokens=608`、`GPU_SCOPE=PASS allowed=0,1,2,3`；库路径、norm/manifest 哈希、motion_rows、train_max/eval_max 先填 `__FILL_AFTER_BUILD__`（出现即 FAIL），阶段 3 回填单独提交。
- `check_stage2_online.py::prepare_restore_inputs`：新增 `--framesamp-store --h5-root --input-manifest`；校验 4×4 layout 与 state 表 sha；demo 帧从源 H5 读；状态机按首包 D+1、后续每次 ≤16 帧。
- `check_stage2_checkpoint.py`：`run_online` 适配 D+1 + 四批 16 帧；`--delivery-step` 增加 79999，输出 `CKPT_DELIVERY_READY=PASS step=79999`。
- 测试：新 `tests/training/test_check_orig16_stage2_launch.py`（8×8 库/错 GPU/缺 FSDP/占位符/G2 缺失/血缘不符均 FAIL；`bash -n` 与 DRY_RUN 无副作用）；`test_framesamp_motion_stage2.py` 增 16 token 矩阵、32 帧边界、`mem_order[608]`；`test_stage2_restore.py` 增 4×4 状态来源、D+1、79999。

**S5 上传**
- 新 `hf_export/run_orig100_trainset_export.sh`（bucket `HongzeFu/robomme-16task-orig100-1600ep-trainset-20261027-v1`）+ README + allowlist；范围 source、framesamp 4×4、norm、wan-latents、motion 两表、缓存导入证明。
- 新 `_stage2_orig16_vla_ckpt_export_common.sh`、`run_orig16_stage2_1024_ckpt_inc.sh`、`run_orig16_vla_stage2_1024_80k_ckpt_export.sh`（bucket `HongzeFu/robomme-vla-orig100-m32x4x4-modul-motion-stage2-1024-80k-v1`，STEPS 5000…75000+79999，证据集同 S2）+ README + allowlist。
- `pack_and_hash.py::build_jobs` 新增 4×4 layout 分支，不要求 fs8 目录。
- 测试 `hf_export/test_orig16_export_drivers.py`。

## 二、子代理分配表

| 编号 | 可写集合 | 依赖 / 合并序 | 验收 | 资源 |
|---|---|---|---|---|
| S6 | MJ `stage2_memory_export.py` 及其测试 | 1 | 计数与源错配负例 | CPU |
| S1 | `check_data_lineage.py`、`check_stage2_validation_inputs.py` 及测试 | 2 | 血缘与 G0 用例 | CPU |
| S2 | 一节 S2 表内文件及测试 | 3 | 新 kind 正负例 + 旧测试全绿 | CPU |
| S3 | `orig16_wan_cache.py` 及两份测试 | 4（兼容测试待 S2） | copy/fill、SHA、S2 兼容 | CPU |
| S4 | 新 yaml/入口/契约、`check_stage2_online.py`、`check_stage2_checkpoint.py`、列出测试 | 5（基于 S2 合入后 HEAD） | 608、D+1、79999、GPU 越界 | CPU |
| S5 | 新导出驱动、`pack_and_hash.py` 4×4 分支及测试 | 6 | fs4 夹具打包回读 | CPU |
| X1–X4 | 新库 / motion / run / 导出产物与日志（运行型，不改代码） | 代码全合入后 | B/G 闸门 | GPU 0–3 |
| 主会话 | Beta、HOLD、契约常量回填、删除裁决、档案、handoff | — | — | CPU |

## 三、runbook（开工后模板）

```bash
P=/scratch/hongze/robomme_policy_learning_MotionJEPA; cd $P; source scripts/dataset/paths.sh
RAW=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/source/h5
MJIDX=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/index
MJSRC=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/source_manifest.json
MJW=/scratch/hongze/worktrees/vla-s2-memexport-20261026
NAME=16task-orig100-1600ep-<pin8>; LIB=$V1_STORE/datasets/$NAME
MANI=$LIB/meta/episode_manifest.json; INMANI=$LIB/meta/input_manifest.json
VALID=$V1_STORE/validation/orig16-stage2; CACHE_PLAN=$VALID/wan-cache-plan.json; CACHE_RECEIPT=$VALID/wan-cache-import.json
CKPT=/scratch/hongze/MotionJEPA/runs/stage2-full144-1025-a/checkpoint_epoch_105.pt; CKPT_SHA=27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035
CPU="CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu"
lineage() { uv run --no-sync python scripts/training/check_data_lineage.py lineage --encoder-ckpt "$CKPT" --expected-ckpt-sha256 "$CKPT_SHA" --vla-input-manifest "$1" --mj-source-manifest "$MJSRC"; }

# 1 清单
env $CPU uv run --no-sync python scripts/dataset/scan_manifest.py build --raw_dir "$RAW" --episodes-per-task 100 --num_shards 1 --out "$MANI"
env $CPU uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs --raw_dir "$RAW" --out "$INMANI"
lineage "$INMANI"
env $CPU uv run --no-sync python scripts/dataset/wan_stage2/orig16_wan_cache.py plan --manifest "$MANI" --input-manifest "$INMANI" --published-index "$MJIDX" --out "$CACHE_PLAN"
# 2 视觉 4×4 与 norm
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-sync python scripts/dataset/run_local.py --stage siglip --lib "$LIB" --gpus 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3 --raw-dir "$RAW" --require-free-mib 70000
CUDA_VISIBLE_DEVICES=3 uv run --no-sync python scripts/dataset/finalize_checks.py check --manifest "$MANI" --out "$LIB/source" --raw_dir "$RAW" --input_manifest "$INMANI" --input_level sha256 --spot_check 1024
env $CPU uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp" --procs 48
env $CPU uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --store "$LIB/framesamp" --resume --procs 48
env $CPU uv run --no-sync python scripts/training/compute_norm_stats.py --output-dir "$V1_STORE/train-assets/mme_vla_suite/$NAME" --config-name mme_vla_suite --repo-id robomme --dataset-path "$LIB/source"
# 4 Wan：恢复缓存 + 只补缺窗
HF_HUB_OFFLINE=0 CUDA_VISIBLE_DEVICES= uv run --no-sync python scripts/dataset/wan_stage2/orig16_wan_cache.py fetch --plan "$CACHE_PLAN" --revision 384d5db410be713a49357bf3f55ca4f4d2f559a0 --out "$LIB/cache-source/official-wan"
CUDA_VISIBLE_DEVICES=0,1,2,3 UV_PROJECT_ENVIRONMENT="$V1_STORE/venvs/wan-stage2" uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/orig16_wan_cache.py import --plan "$CACHE_PLAN" --cache-root "$LIB/cache-source/official-wan" --raw-dir "$RAW" --out "$LIB/wan-latents" --receipt "$CACHE_RECEIPT" --gpus 0,1,2,3
# 5 motion
env $CPU uv run --no-sync python scripts/dataset/check_stage2_validation_inputs.py --manifest "$MANI" --latents "$LIB/wan-latents" --source "$LIB/source" --h5-root "$RAW" --input-manifest "$INMANI" --mj-index "$MJIDX" --cache-import-receipt "$CACHE_RECEIPT" --out "$VALID/G0.json"
CUDA_VISIBLE_DEVICES=0 uv run --project "$MJW" --no-sync python "$MJW/scripts/stage2_memory_export.py" --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --latents "$LIB/wan-latents" --manifest "$MANI" --out "$LIB/motion-tokens-stage2" --grid-subsample 2 --batch 1 --check-dropout 64 --expected-counts <段数>,<grid32行数> --input-manifest "$INMANI"
CUDA_VISIBLE_DEVICES= uv run --no-sync python scripts/dataset/wan_stage2/orig16_wan_cache.py check --samples "$VALID/G0.json" --receipt "$CACHE_RECEIPT" --raw-dir "$RAW" --out "$VALID/G2.json"
env $CPU uv run --no-sync python scripts/dataset/pack_motion_store.py pack --layout stage2 --latent-grid-subsample 2 --latents-origin published-cache-with-local-fill --cache-import-receipt "$CACHE_RECEIPT" --manifest "$MANI" --tokens "$LIB/motion-tokens-stage2" --latents "$LIB/wan-latents" --out "$LIB/motion-stage2-1024" --encoder-run-dir "$V1_STORE/external/motionjepa/stage2-full144-1025-a" --raw-dir "$RAW" --expected-ckpt-sha256 "$CKPT_SHA" --stage2-validation "$VALID" --stage2-source-pin scripts/dataset/wan_stage2/SOURCE_PIN.json
env $CPU uv run --no-sync python scripts/dataset/pack_motion_store.py verify --store "$LIB/motion-stage2-1024" --resume --check-padding
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/training/check_stage2_online.py --store "$LIB/motion-stage2-1024" --samples "$VALID/G0.json" --history "$P/src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-32frame-4x4-motion-stage2-orig16.yaml" --framesamp-store "$LIB/framesamp" --h5-root "$RAW" --input-manifest "$INMANI" --online-gpu 0 --out "$VALID/G4.json"
uv run --no-sync python scripts/training/tests/train_es_bound.py --manifest "$MANI" --layout motion-stage2-1x1024-grid32-demopad17-v1 --budget 96
# 6 短测与正式
CUDA_VISIBLE_DEVICES=0,1,2,3 DRY_RUN=1 RUN_NAME=o16-smoke-s2-<美东MMDDHHMM> MAX_STEPS=20 TRAIN_HEAD=<sha> FINAL_RECORD_DIR="$VALID/smoke/final-record" bash scripts/training/run_orig16_stage2_1024_80k.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 RUN_NAME=orig100-m32x4x4-modul-motion-stage2-1024-b128-80k TRAIN_HEAD=<训练Beta sha> FINAL_RECORD_DIR="$VALID/orig100-m32x4x4-modul-motion-stage2-1024-b128-80k/final-record" bash scripts/training/run_orig16_stage2_1024_80k.sh
```

G3（`validate_stage2.py --gate G3 --cache-import-receipt`）、保存恢复（`check_stage2_checkpoint.py --mode smoke`）、G12（`--mode delivery --delivery-step 79999`）命令同上式样，GPU 父 0 / sidecar 1。tmux 名：`o16-siglip`、`o16-finalize`、`o16-pack4`、`o16-cache-fetch`、`o16-cache-import`、`o16-s2export`、`o16-s2pack`、`o16-g0/g2/g3/g4`、`o16-smokes2`、`o16-trains2`、`o16-incs2`、`o16-trainset-up`、`o16-ups2`、`o16-g12`。

## 四、风险与未验证

- 缺窗计数（2824 物理 / 1439 消费）是索引推算，正式 manifest 不符须先解释，不改补帧或截尾规则凑数。
- 缓存恢复峰值约 388 GB，与 SigLIP features 约 465 GB 错开；B0 build 前 ≥1.5 T、train 前 ≥0.6 T。
- 4 卡步时、显存、79999 恢复均未实测；发布缓存逐文件 SHA 未下载核验。
- 80k 约 21.5 epoch 可能过拟合，步数待裁决；选定后同步改 run/bucket 名、STEPS 与交付步。

## 五、留档与提交

建库档案 `docs/dataset-build-doc/16task-orig100-1600ep-<pin8>/`，训练档案 `docs/training-doc/orig100-m32x4x4-modul-motion-stage2-1024-b128-80k/`。建库 Beta、训练 Beta 各自配对结论提交；状态机任务 id `orig16-vla-build`、`orig16-vla-train-m32x4x4-motion-4gpu`，是否登记按第七节第 3 项。
