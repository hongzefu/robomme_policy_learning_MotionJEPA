# scripts/dataset —— 本机数据处理链路（SigLIP 帧路 + Wan VAE / MotionJEPA 运动路）

v2-motionmem（2026-09-03）起的建库链路，全部在本机双卡（RTX 6000 Ada ×2）跑，不再提交集群
（旧 `gl/` 与 `pack/` 目录已删除，`gl_submit.py` 搬到 `scripts/training/`）。权威设计与判据：
仓库根 [`motion-memory-plan.md`](../../motion-memory-plan.md) 第一部分四节与第二部分一节；本次 40 ep 库的
起跑与结果留档：[`docs/dataset-build-doc/4task-motion-40ep/`](../../docs/dataset-build-doc/4task-motion-40ep/)。

## 目录

| 文件 | 作用 |
|---|---|
| `paths.sh` | 唯一路径 / 环境源（`RAW_H5_DIR`、`V1_STORE`、`OPENPI_DATA_HOME`、`HF_HOME`…；禁覆盖 `HOME`；仓库位置 fail-loud） |
| `scan_manifest.py` | 清单 `meta/episode_manifest.json`（`build --tasks … --episodes-per-task N`，schema 与旧版逐字段相同；`sample` 抽子集） |
| `run_local.py` | ★ 本机多 GPU 调度器：`--stage siglip|wan|encode --gpus 0,1`，每卡一常驻 worker、动态领任务（`_claims/`），收尾 `STAGE_DONE` |
| `build_shard.py` | SigLIP worker（`--worker-mode` 由 run_local 起；旧的分片模式保留）；`_process_episode` 逻辑原样继承 `DatasetProcessor` |
| `finalize_checks.py` | SigLIP 守卫：`hash-inputs`（h5 sha256 → `meta/input_manifest.json`）、`check`（完整性 / stats / provenance / 同卡 256 条零容差抽检） |
| `compare_datasets.py` | SigLIP 对拍（`--mode bitexact`；`--b_manifest` 跨清单按物理身份匹配；`--all_pkl` 全量 pkl） |
| `pack_framesamp_store.py` | framesamp 三表打包 / verify（逻辑不改，自 `pack/` 上提） |
| `pack_motion_store.py` | ★ motion 表 pack / verify（锁、两阶段 meta、逐行 digest；行序契约见 `datastore/motion_store.py`） |
| `test_guards.py` | 守卫单测（`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= uv run pytest scripts/dataset/test_guards.py -q`） |
| `wan/` | ★ torch 侧 uv 子项目（独立 `pyproject.toml` + `uv.lock`，venv 落 `v1-store/venvs/wan`，torch 2.9.0+cu128 / diffusers 0.39.0） |
| `wan/wan_motion_infer.py` | MotionJEPA `scripts/inference-example/wan_motion_infer.py` 整文件复制件 + `SOURCE_PIN.json`（commit / sha256） |
| `wan/wan_common.py` | 三个 worker 的公共件：清单、段工作项、网格公式（与 `datastore/motion_store.py` 互证）、claim、原子落盘、provenance |
| `wan/extract_wan.py` | 网格窗抽取 worker：33 帧 → 复制件 `encode_chunk` → `wan-latents/<段>.bin` + `.sha256` + `.metadata.json` |
| `wan/encode_motion.py` | encoder worker：latent 块 → 复制件 `motion_token` → `motion-tokens/<段>.f32.bin` |
| `wan/oracle_driver.py` | 经 MotionJEPA uv 环境调**原版**函数产 D2 / D3 真值（独立重算起点、逐窗核 33 帧 sha256） |
| `wan/compare_wan.py` | D2 `WAN_BITEXACT` / D3 `ENCODER_BITEXACT` 逐位比对 |
| `wan/probe_wan.py` | S0 探针（A2 计时 / 显存 / 漂移，A3 跨卡，A4 双 venv） |
| `build_dataset.py`、`tarxz_h5.py`、`unzip_data.py`、`finetune_vlm_subgoal_predictor.sh`、`hf_export/` | 非抽取件，原地不动 |

## 40 ep 库的构建顺序（数字与判定行以留档为准）

```bash
cd <仓库根>            # 环境 A：/data/hongzefu/robomme_policy_learning_MotionJEPA；环境 B：/scratch/hongze/robomme_policy_learning_MotionJEPA
source scripts/dataset/paths.sh; v1_prepare_dirs; v1_validate_raw_h5; v1_require_models; v1_require_wan
LIB=$DATASETS_DIR/4task-motion-40ep    # RAW_H5_DIR 已按仓库前缀分叉（A：/data/hongzefu/robomme_data_h5 16 任务全集；B：/scratch/hongze/robomme_data_h5 只含 4 个目标 h5）
# 一律 uv run --no-sync（多 worker 并发触发 uv 同步会争锁；run_local.py 三个 stage 内部也都带 --no-sync）

# 1. 清单（40 ep）+ 输入 sha256（hash-inputs 只取目录内 *.h5；环境 A 用只含 4 个 h5 的 symlink 目录 $V1_STORE/raw-link-4task，环境 B 直接指 $RAW_H5_DIR）
uv run --no-sync python scripts/dataset/scan_manifest.py build --raw_dir $RAW_H5_DIR --tasks $TARGET_TASKS_CSV \
    --episodes-per-task 10 --num_shards 1 --out $LIB/meta/episode_manifest.json
uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs --raw_dir $RAW_H5_DIR --out $LIB/meta/input_manifest.json

# 2. SigLIP 阶段（主 venv，双卡）→ 守卫 → framesamp 打包
uv run --no-sync python scripts/dataset/run_local.py --stage siglip --lib $LIB --gpus 0,1 --raw-dir $RAW_H5_DIR     # --gpus 按机器给（8 卡机 0,1,…,7；worker 数不影响字节）
uv run --no-sync python scripts/dataset/finalize_checks.py check --manifest $LIB/meta/episode_manifest.json --out $LIB/source \
    --raw_dir $RAW_H5_DIR --input_manifest $LIB/meta/input_manifest.json --spot_check 256
uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --source $LIB/source --manifest $LIB/meta/episode_manifest.json --out $LIB/framesamp
uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --store $LIB/framesamp --resume

# 3. Wan 抽取（子 venv；只依赖清单 + 原始 h5，可与 SigLIP 后处理并行，但 finalize 的 JAX 抽检会占一张卡——run_local 的 20 GB 空闲显存预检会拒绝该卡）→ encoder → motion 表
uv run --no-sync python scripts/dataset/run_local.py --stage wan --lib $LIB --gpus 0,1 --raw-dir $RAW_H5_DIR
uv run --no-sync python scripts/dataset/run_local.py --stage encode --lib $LIB --gpus 0,1
uv run --no-sync python scripts/dataset/pack_motion_store.py pack --manifest $LIB/meta/episode_manifest.json \
    --tokens $LIB/motion-tokens --latents $LIB/wan-latents --out $LIB/motion
uv run --no-sync python scripts/dataset/pack_motion_store.py verify --store $LIB/motion --resume

# 4. oracle 与对拍（D1 SigLIP / D2 Wan / D3 encoder；命令见 docs/dataset-build-doc/4task-motion-40ep/launch.md；
#    多卡机 D2 oracle 用 oracle_driver.py vae --shard-idx i --num-shards n 分片后 aggregate，见 4task-motion-40ep-aws/launch.md）
```

每阶段超过 5 分钟的都放 detached tmux（`PYTHONUNBUFFERED=1` + `pipefail` + `tee` + 尾行 `EXIT_CODE=`）。

## 库结构

```
v1-store/datasets/4task-motion-40ep/
├── meta/{episode_manifest.json, input_manifest.json}   40 ep 清单 + 四个 h5 的 sha256
├── source/{features/, data/, meta/}                    散 npy + pkl（形制同 4task-gl）
├── framesamp/                                          packed 三表 + meta（LAYOUT framesamp-4x4-v1）
├── motion/                                             motion_token.f32.bin + meta（LAYOUT motion-768-grid16-v1）
├── wan-latents/                                        <段>.bin + .sha256 + .metadata.json + 汇总 metadata.json
├── motion-tokens/                                      <段>.f32.bin + .sha256 + .metadata.json + 汇总 metadata.json
├── oracle/{siglip-shard1, siglip-serial, wan-mj, probe}/  SigLIP oracle 两库、原版 Wan/encoder 真值、S0 探针
└── logs/                                               各 worker 日志
```
