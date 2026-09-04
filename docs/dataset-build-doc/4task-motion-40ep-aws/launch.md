# 4task-motion-40ep（环境 B / AWS 复刻）构建留档 —— launch

> 本目录记录 **2026-09-04 在环境 B（AWS 单机 8×A100-SXM4-80GB）上从零重建 40 ep 测试库**的口径。库名沿用
> `v1-store/datasets/4task-motion-40ep`（测试脚本默认路径都指它）；环境 A（2×RTX 6000 Ada + turbo）原始建库留档见同级
> [`../4task-motion-40ep/`](../4task-motion-40ep/)，链路本身零改动，本文件只写与环境 A 不同的部分。结果见 [`result.md`](result.md)。

## 环境判定（AGENTS.md「运行环境判定」）

```
repo=/scratch/hongze/robomme_policy_learning_MotionJEPA
/nfs/turbo/coe-chaijy-unreplicated/hongzefu: 不存在   /data/hongzefu: 不存在   /scratch/hongze: 存在   ~/.ssh/config: 不存在
      8 NVIDIA A100-SXM4-80GB        （driver 595.71.05；/scratch = /dev/md0 6.9 T）
```

→ **环境 B**。无 GreatLakes、无 turbo、无本机 `/data/hongzefu` 原件；一切持久化落 `/scratch/hongze/`。

## 起跑 commit 与代码适配

- **起跑 HEAD**：`8093ebda23ec566533067e319bab506baaf80de5`（commitV6.12，干净）。该 commit 是环境 B 的前置适配，不触碰训练语义：
  两份 `paths.sh` 前缀白名单加 `AWS_WORK_PREFIX="/scratch/hongze/"`、`RAW_H5_DIR` 默认按前缀分叉、`MJ_REPO` 可被环境变量覆盖；
  `run_local.py` siglip 分支 `uv run` 加 `--no-sync`；`oracle_driver.py` 加 `--shard-idx/--num-shards` 与 `aggregate` 子命令（D2 8 卡并行）；
  `fetch_assets.py` 在钉 sha 的 `snapshot_download` 后补写 `refs/main`（异地从零复刻首次踩中：`wan_vae` 缺 `refs/main` → `ASSETS=FAIL`）。
- 与 `4task-motion-40ep/launch.md` 相比，`scan_manifest / build_shard / finalize_checks / pack_framesamp_store / extract_wan / encode_motion / pack_motion_store / compare_*`
  的**计算路径零改动**，只有 CLI 入口的 `uv run --no-sync` 与分片参数。

## 外部资产与原始数据（全部落 /scratch/hongze/）

| 物 | 来源 / 命令 | 结果 |
|---|---|---|
| 主 venv `.venv` | `UV_CACHE_DIR=/scratch/hongze/.cache/uv UV_LINK_MODE=copy uv sync` | jax 0.5.3，8 个 cuda 设备 |
| wan 子 venv `v1-store/venvs/wan` | `GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519_hongzefu …' GIT_CONFIG_*=url.git@github.com:.insteadOf=https://github.com/ UV_PROJECT_ENVIRONMENT=$PWD/v1-store/venvs/wan uv sync --project scripts/dataset/wan` | torch 2.9.0+cu128 / diffusers 0.39.0 |
| MotionJEPA 只读副本 | `git clone git@github.com:hongzefu/MotionJEPA.git /scratch/hongze/MotionJEPA && git checkout 2a484ad960ed6155321dc34def9011eb119f857f && uv sync` | torch 2.9.0+cu128 / diffusers 0.39.0（与 wan venv 同版） |
| 六条外部权重 | `HF_TOKEN=<HongzeFu read token，只在命令 env> uv run --no-sync python scripts/assets/fetch_assets.py plan / fetch`，`verify --level full` | `ASSETS_PLAN total=14.5GB assets=6 missing=6` → 首次 `ASSETS=FAIL`（wan_vae 缺 refs/main）→ 修 fetch_assets 后 `fetch --force --assets wan_vae` → **`ASSETS=PASS assets=6 mismatches=0`** |
| 原始 H5 | 公开集 `Yinpei/robomme_data_h5` 只下 4 个目标任务 tar.xz（2.01 / 2.95 / 1.17 / 1.54 GiB）到 `/scratch/hongze/robomme_data_h5/`，`scripts/dataset/tarxz_h5.py decompress --input_dir … --jobs 4`（4m49s，保留压缩包） | 见 result.md「H5 同源」 |
| norm_stats（测试用替身） | `cp assets/norm_stats.json v1-store/train-assets/mme_vla_suite/robomme/norm_stats.json` | sha256 `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5` |

`RAW_H5_DIR=/scratch/hongze/robomme_data_h5`（目录里只有这 4 个 `.h5` + 4 个 `.tar.xz`；`hash-inputs` / `build_dataset.py` / `compare_datasets.py` 都只取 `*.h5`）。

## 命令序列（`LIB=v1-store/datasets/4task-motion-40ep`，全部 `UV_LINK_MODE=copy uv run --no-sync`，长任务 detached tmux + Monitor）

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA; source scripts/dataset/paths.sh; v1_prepare_dirs
LIB=$V1_STORE/datasets/4task-motion-40ep; RAW=$RAW_H5_DIR; MJ=/scratch/hongze/MotionJEPA
# 探针（A100；A2 GPU2 / A3 GPU0 vs GPU1 / A4 MotionJEPA venv vs wan venv 同卡 GPU1；与 O1/O2/hash-inputs 并发）
uv run --project scripts/dataset/wan --no-sync python scripts/dataset/wan/probe_wan.py --module copy bench --h5 $RAW/record_dataset_ButtonUnmask.h5 --episode 0 --stride 12 --n 20 --out $LIB/oracle/probe/a2-bench-gpu2.json
… encode --stride 4 --n 64 → a3-copy-gpu0.npz / a3-copy-gpu1.npz；--module orig --mj-repo $MJ → a4-orig-mjvenv-gpu1.npz；compare --require-bitwise
# 清单 + 输入 sha256
uv run --no-sync python scripts/dataset/scan_manifest.py build --raw_dir $RAW --tasks $TARGET_TASKS_CSV --episodes-per-task 10 --num_shards 1 --out $LIB/meta/episode_manifest.json
uv run --no-sync python scripts/dataset/scan_manifest.py build --raw_dir $RAW --tasks $TARGET_TASKS_CSV --episodes-per-task 100 --num_shards 1 --out $LIB/oracle/manifest-4task-100ep.json   # a6 用
uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs --raw_dir $RAW --out $LIB/meta/input_manifest.json
# SigLIP oracle（GPU7 / GPU6）与 SigLIP 阶段（GPU0–5）并行
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/dataset/build_shard.py --manifest $LIB/meta/episode_manifest.json --raw_dir $RAW --out $LIB/oracle/siglip-shard1 --num_shards 1 --shard_idx 0 --require_empty_output
CUDA_VISIBLE_DEVICES=6 uv run --no-sync python scripts/dataset/build_dataset.py --dataset_type robomme_pkl --raw_data_path $RAW --preprocessed_data_path $LIB/oracle/siglip-serial --max_episodes 10 | tee $LIB/oracle/siglip-serial.build.log
uv run --no-sync python scripts/dataset/run_local.py --stage siglip --lib $LIB --gpus 0,1,2,3,4,5 --raw-dir $RAW
uv run --no-sync python scripts/dataset/finalize_checks.py check --manifest $LIB/meta/episode_manifest.json --out $LIB/source --raw_dir $RAW --input_manifest $LIB/meta/input_manifest.json --input_level sha256 --spot_check 256
uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --source $LIB/source --manifest $LIB/meta/episode_manifest.json --out $LIB/framesamp --procs 32; … verify --store $LIB/framesamp --resume --procs 32
# D1
uv run --no-sync python scripts/dataset/compare_datasets.py --mode bitexact --manifest $LIB/meta/episode_manifest.json --a_lib $LIB/oracle/siglip-shard1 --b_lib $LIB/source --b_manifest $LIB/meta/episode_manifest.json --steps_per_episode 0 --all_pkl --report $LIB/oracle/compare-o1.json
uv run --no-sync python scripts/dataset/compare_datasets.py --mode bitexact --manifest $LIB/oracle/manifest-4task-100ep.json --a_lib $LIB/oracle/siglip-serial --a_untouched_log $LIB/oracle/siglip-serial.build.log --a_max_episodes 10 --raw_dir $RAW --b_lib $LIB/source --b_manifest $LIB/meta/episode_manifest.json --steps_per_episode 0 --report $LIB/oracle/compare-o2.json
# Wan（GPU0–5，与 SigLIP 后处理并行）→ encode（8 卡）→ motion 表
uv run --no-sync python scripts/dataset/run_local.py --stage wan --lib $LIB --gpus 0,1,2,3,4,5 --raw-dir $RAW
uv run --no-sync python scripts/dataset/run_local.py --stage encode --lib $LIB --gpus 0,1,2,3,4,5,6,7
uv run --no-sync python scripts/dataset/pack_motion_store.py pack --manifest $LIB/meta/episode_manifest.json --tokens $LIB/motion-tokens --latents $LIB/wan-latents --out $LIB/motion; … verify --store $LIB/motion --resume
# D2 / D3 oracle（MotionJEPA venv；VAE 8 片 + aggregate，encoder 单进程 GPU7）
for i in 0..7: CUDA_VISIBLE_DEVICES=$i PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 uv run --project $MJ --no-sync python scripts/dataset/wan/oracle_driver.py --mj-repo $MJ vae --manifest $LIB/meta/episode_manifest.json --raw-dir $RAW --latents $LIB/wan-latents --out $LIB/oracle/wan-mj --shard-idx $i --num-shards 8
uv run --project $MJ --no-sync python scripts/dataset/wan/oracle_driver.py aggregate --manifest $LIB/meta/episode_manifest.json --out $LIB/oracle/wan-mj --num-shards 8 --kind vae
CUDA_VISIBLE_DEVICES=7 … oracle_driver.py --mj-repo $MJ encoder --manifest … --latents $LIB/wan-latents --out $LIB/oracle/wan-mj --expected-ckpt-sha256 bae960373041629e976a1f4a7d6d48ca3c51786c827146a3ee10bf7b034bc15a
uv run --no-sync python scripts/dataset/wan/compare_wan.py latents --latents $LIB/wan-latents --oracle $LIB/oracle/wan-mj
uv run --no-sync python scripts/dataset/wan/compare_wan.py tokens --store $LIB/motion --oracle $LIB/oracle/wan-mj
# 附加检查：motion_checks.py a6（--manifest-400ep $LIB/oracle/manifest-4task-100ep.json）/ a7 / a9set / a10；extra_checks.py a8 / a9enc（wan venv，GPU6/7）
```

## 8 卡分配

SigLIP 阶段 6 worker（GPU0–5）+ O1（GPU7）+ O2（GPU6）同时跑；Wan 抽取 6 worker（GPU0–5，此时 GPU6/7 仍被 O1/O2 占用）；encode 8 worker；D2 oracle VAE 8 片各占一卡。
A3 已证 A100 跨卡逐位（`max_abs_diff=0`），故 worker 数与卡号不影响任何字节。

## 不可做项（环境 B 永久失效，见 motion-memory-plan.md「环境 B 复刻」节）

A5（vs `/data/hongzefu/robomme_data_h5_v2_4env400ep` 与 MotionJEPA `data-raw`）、A11（vs turbo `4task-gl`）、A12（v7 latent）、MotionJEPA `crosscheck.py --vae_check`（需 `/data/hongzefu/dataset-4env-v8`）——对照物都在环境 A 的 `/data` 或 turbo。
A5 的替代证据：公开版四个 h5 与环境 A 留档的 sha256 前缀与字节数全部命中（result.md）。
