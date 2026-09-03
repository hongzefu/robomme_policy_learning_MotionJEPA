# 4task-motion-40ep 数据集构建 launch 记录（S0 先验与 oracle / S1 重抽与建库）

对应 `motion-memory-plan.md` 第一部分四节与第二部分一节、〇节 S0 / S1。按 `AGENTS.md` 第 12 条起跑前预提交：
commit、命令、配置、数据来源、输出路径与判据。结果随各阶段结束以 docs commit 补写进同目录 `result.md`。
本文件先写 S0；S1 起跑前追加「S1」一节。

## S0 起跑环境

- **起跑 HEAD**：本 launch.md 与 `scripts/dataset/wan/` 子项目骨架一起提交的 commitV6.1（实际 sha 在 `result.md` 回填）。
  代码锚点 `46ba9540f785970fb336ed05a32a7542679624ce`：相对它，`src/` 与 SigLIP oracle 用到的旧脚本
  `scripts/dataset/gl/scan_manifest.py` / `scripts/dataset/gl/build_shard.py` / `scripts/dataset/build_dataset.py` 零改动，
  新增的只有 `scripts/dataset/wan/`（torch 侧子项目）与本文件。工作区 clean。
- **机器**：本机 sled-vail，2×RTX 6000 Ada（48 GB），driver 570.211.01；**目标卡 GPU1**（`CUDA_VISIBLE_DEVICES=1`），
  两条 oracle 与被测都在 GPU1 产出；GPU0 只用于 A3 跨卡探针与 S1 的双卡并行抽取。GPU0 上另有两个本会话看不到的外来进程各占 498 MiB，
  用户 2026-09-03 拍板两张卡都可用。
- **执行方式**：超过 5 分钟的任务走 detached tmux（`PYTHONUNBUFFERED=1` + `pipefail` + `tee` + 尾行 `EXIT_CODE=`）；短探针直接后台跑并落日志。
  日志根 `v1-store/logs/motion/`（`v1-store/` 不进 git，判定行内联进 `result.md`）。
- **主 venv**：`.venv`（python 3.11.14 / jax 0.5.3 / torch 2.7.1+cu126），`pyproject.toml` 与根 `uv.lock` 一字不动（红线 10）。
- **wan 子 venv**：`scripts/dataset/wan/pyproject.toml` + 独立 `uv.lock`（进 git），venv 落 `v1-store/venvs/wan`。锁定版本与 MotionJEPA `2a484ad`
  的 `uv.lock` 逐项同值：torch 2.9.0（PyPI，cu128）/ diffusers 0.39.0 / numpy 2.4.4 / h5py 3.16.0 / pyyaml 6.0.3 / einops 0.8.2 /
  nvidia-cudnn-cu12 9.10.2.21 / nvidia-cublas-cu12 12.8.4.1 / torchvision 0.24.0；`motion-jepa` 以 git 依赖钉在
  `2a484ad960ed6155321dc34def9011eb119f857f`（`uv lock` 解析 91 包 12.9 s，`uv sync` 在 tmux `motion-wan-venv`）。
- **MotionJEPA 只读副本**：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA`，分支 `v6.1.1-slurmWanExtract`，
  HEAD `2a484ad960ed6155321dc34def9011eb119f857f`，工作区 clean；其 `.venv` 为 python 3.11.14 / torch 2.9.0 / diffusers 0.39.0（A4 的 oracle 侧）。

## 外部资产（S0 ③④⑤，已落地并核指纹）

| 资产 | 源 | 落点 | 指纹 |
|---|---|---|---|
| `wan_motion_infer.py` 复制件 | MotionJEPA `scripts/inference-example/wan_motion_infer.py` | `scripts/dataset/wan/wan_motion_infer.py` | sha256 `af67fdd913543aee416a9fe5df797f707ff159165c468d687fcf8e7347941b34`，与源逐字节相同；`SOURCE_PIN.json` 记 commit / 路径 / sha |
| encoder checkpoint | MotionJEPA `runs/wan-v8-filter10-72ep-a/checkpoint_epoch_72.pt`（954,853,147 B） | `v1-store/external/motionjepa/wan-v8-filter10-72ep-a/checkpoint_epoch_72.pt` | sha256 `bae960373041629e976a1f4a7d6d48ca3c51786c827146a3ee10bf7b034bc15a`，两侧相同 |
| encoder run 配置 | 同 run 的 `config.yaml`（`motion.dim=768`、`training.precision=bf16`、`data.max_horizon=8`、`wan.vae_id=Wan-AI/Wan2.1-T2V-1.3B-Diffusers`） | 同目录 `config.yaml` | sha256 `99548a6ca23522c235281e45819ae6d5e96a916709cb4b9c0b47142832c90946`，两侧相同 |
| Wan2.1 VAE 权重 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/hf-cache/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers`（snapshot `0fad780a534b6463e45facd96134c9f345acfa5b`） | `v1-store/cache/hf/hub/`（`cp -a`，578 MB） | safetensors blob sha256 `d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793` 两侧相同；state_dict 指纹须 `== 9980d252230c265cc2869466a74f85f5ee45b01ea9521bbb31159f90b75fe6d0`（`load_vae` 起手断言，S0 探针与 crosscheck 实测） |

`HF_HOME=v1-store/cache/hf`、`HF_HUB_OFFLINE=1`，不覆盖 `HOME`（红线 11）。

## 数据来源与清单（S0 ②）

- **原始 H5**：`/data/hongzefu/robomme_data_h5`（16 任务 × 100 ep，本机 NVMe，永久保留）。四个目标 h5：
  `record_dataset_{ButtonUnmask,ButtonUnmaskSwap,VideoUnmask,VideoUnmaskSwap}.h5`；帧 `episode_<i>/timestep_<t>/obs/front_rgb`
  `(256,256,3) uint8`。逐文件 sha256（合计约 82 GB）在 S1 建库时记入库内 `meta/input_manifest.json`，S0 不重复算。
- **只含符号链接的目录** `v1-store/raw-link-4task/`（4 条 symlink → 上述 h5；2026-09-03 11:48 建成，**到 D1 对拍结束不得重建**——
  `build_dataset.py` 用 `os.listdir` 遍历，目录序决定它的 `global_episode_idx`）。
- **清单（旧工具，clean HEAD）**：`scripts/dataset/gl/scan_manifest.py build --raw_dir v1-store/raw-link-4task --out
  v1-store/datasets/4task-motion-40ep/oracle/manifest-4task-100ep.json --num_shards 1` → 400 ep / 123,044 timestep / 101,066 exec 样本，
  sha256 `4de8a0fcac2bf1c6867b625edefab581c92e26fc16f160182959e2bdb46498b1`（规范序 ButtonUnmask → ButtonUnmaskSwap → VideoUnmask → VideoUnmaskSwap）；
  `scan_manifest.py sample --mode prefix --n 10` → `oracle/subset-prefix10.json`：40 ep（每 h5 前 10 个 raw_ep），**13,756 timestep**，与计划 4.1 一致。
  S1 的新清单 `meta/episode_manifest.json` 以 `--tasks … --episodes-per-task 10` 独立生成后与这 40 条逐字段核对（A6）。

## S0 命令

### ② SigLIP oracle O1 / O2（主 venv，GPU1，tmux `motion-siglip-oracle`，串行同卡）

```bash
cd /data/hongzefu/robomme_policy_learning_MotionJEPA
V1=$PWD/v1-store; LIB=$V1/datasets/4task-motion-40ep
export OPENPI_DATA_HOME=$V1/models XDG_CACHE_HOME=$V1/cache/xdg HF_HOME=$V1/cache/hf \
       JAX_COMPILATION_CACHE_DIR=$V1/cache/jax UV_LINK_MODE=copy PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1
tmux new-session -d -s motion-siglip-oracle "set -o pipefail; { echo HEAD=\$(git rev-parse HEAD); \
  uv run python scripts/dataset/gl/build_shard.py --manifest $LIB/oracle/manifest-4task-100ep.json \
    --raw_dir $V1/raw-link-4task --out $LIB/oracle/siglip-shard1 --shard_idx 0 --num_shards 1 \
    --subset $LIB/oracle/subset-prefix10.json --require_empty_output && \
  uv run python scripts/dataset/build_dataset.py --dataset_type robomme_pkl --raw_data_path $V1/raw-link-4task \
    --preprocessed_data_path $LIB/oracle/siglip-serial --max_episodes 10 2>&1 | tee $LIB/oracle/siglip-serial.build.log; } \
  2>&1 | tee $V1/logs/motion/s0-siglip-oracle.log; echo EXIT_CODE=\$? >> $V1/logs/motion/s0-siglip-oracle.log"
```

- O1（主 oracle）：`build_shard.py --num_shards 1 --shard_idx 0 --subset`，编号体系沿 400 ep 清单（只跑其中 40 条）。
- O2（旁证）：未改动 builder `build_dataset.py --max_episodes 10`（每 h5 前 10 个 episode，`os.listdir` 序），
  stdout 单独 tee 到 `oracle/siglip-serial.build.log` 供 `compare_datasets.py --a_untouched_log` 反查编号。
- 预估：13,756 帧 × 2 库 @ ≈67 step/s（本机 NVMe）≈ 7 min 合计。

### ② MotionJEPA `crosscheck.py --vae_check`（MotionJEPA uv 环境，GPU1，O1/O2 结束后跑）

```bash
MJ=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA; V1=$PWD/v1-store; LIB=$V1/datasets/4task-motion-40ep
mkdir -p $LIB/oracle/wan-mj
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 HF_HOME=$V1/cache/hf HF_HUB_OFFLINE=1 UV_LINK_MODE=copy \
  uv run --project $MJ --no-sync python $MJ/scripts/inference-example/crosscheck.py \
    --data_root /data/hongzefu/dataset-4env-v8 --vae_check --out_json $LIB/oracle/wan-mj/crosscheck.json \
  2>&1 | tee $V1/logs/motion/s0-crosscheck.log
```

（`crosscheck.py` 无 `--device`，选卡只靠 `CUDA_VISIBLE_DEVICES`；`--out_json` 不建父目录；需 `/data/hongzefu/dataset-4env-v8`
与 `/data/hongzefu/motionjepa-v7/data-raw`，均已在位。）

### ① Ada 探针 A2 + A3 + A4（`scripts/dataset/wan/probe_wan.py`，子 venv 建成后跑）

窗口源：`record_dataset_ButtonUnmask.h5` `episode_0`（291 帧，es = 0）。

```bash
V1=$PWD/v1-store; LIB=$V1/datasets/4task-motion-40ep; H5=/data/hongzefu/robomme_data_h5/record_dataset_ButtonUnmask.h5
MJ=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA; P=scripts/dataset/wan/probe_wan.py
RUN_WAN="env UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=$V1/venvs/wan uv run --project scripts/dataset/wan --no-sync python"
RUN_MJ="env PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy uv run --project $MJ --no-sync python"
# A2：20 窗（起点 0,12,…,228）计时 + max_memory_allocated + TF32 / VAE-bf16 两档漂移（只记录）
HF_HOME=$V1/cache/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 $RUN_WAN $P --module copy bench --h5 $H5 --episode 0 --stride 12 --n 20 --out $LIB/oracle/probe/a2-bench-gpu1.json
# A3：同 64 窗（起点 0,4,…,252）GPU0 vs GPU1，复制件，max|diff| 须为 0
HF_HOME=$V1/cache/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 $RUN_WAN $P --module copy encode --h5 $H5 --stride 4 --n 64 --out $LIB/oracle/probe/a3-copy-gpu1.npz
HF_HOME=$V1/cache/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 $RUN_WAN $P --module copy encode --h5 $H5 --stride 4 --n 64 --out $LIB/oracle/probe/a3-copy-gpu0.npz
$RUN_WAN $P compare --a $LIB/oracle/probe/a3-copy-gpu1.npz --b $LIB/oracle/probe/a3-copy-gpu0.npz --tag A3_CROSSGPU --require-bitwise --allow-prov-diff
# A4：同 64 窗 MotionJEPA .venv（原版模块）vs 子 venv（复制件），同卡 GPU1，max|diff| 须为 0 且 provenance 白名单逐键相等
HF_HOME=$V1/cache/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 $RUN_MJ $P --module orig encode --h5 $H5 --stride 4 --n 64 --out $LIB/oracle/probe/a4-orig-mjvenv-gpu1.npz
$RUN_WAN $P compare --a $LIB/oracle/probe/a4-orig-mjvenv-gpu1.npz --b $LIB/oracle/probe/a3-copy-gpu1.npz --tag A4_DUALVENV --require-bitwise
```

## S0 判据

| 项 | 判定行 / 判据 |
|---|---|
| O1 | `SHARD_DONE shard=0 episodes=40 …` 且日志尾行 `EXIT_CODE=0`；`meta/_shard0of1.json` 的 `manifest_sha256 == 4de8a0fc…`、`episodes` 为子集 40 条 |
| O2 | `Time taken: … minutes`、`meta/stats.json` 的 `execution_samples == 11530`、`features/` 下 40 个 `episode_*` |
| crosscheck | 末行 `CROSSCHECK=PASS`，`[V7]` 指纹 `9980d252…`，encoder 段 [0]–[10] 与 VAE 段 [V1]–[V7] 全过；json 归档 `oracle/wan-mj/crosscheck.json` |
| A2 | `PROBE_BENCH=PASS … rerun_bitwise=20/20`；ms/窗与 peak MiB 只记录（先验：A40 1.57 s/窗）；TF32 / bf16 漂移只记录不设线 |
| A3 | `A3_CROSSGPU=PASS compared=64 latent_bitwise=64 token_bitwise=64 max_abs_diff=0.000e+00`；不过则 S1 全部单卡 |
| A4 | `A4_DUALVENV=PASS compared=64 …`（含 provenance 白名单逐键相等，`module_sha256` 两侧 == `SOURCE_PIN.source_sha256`）；不过即重锁子项目或复制件被改 |
| 复制件 | `sha256(scripts/dataset/wan/wan_motion_infer.py) == SOURCE_PIN.source_sha256` |
