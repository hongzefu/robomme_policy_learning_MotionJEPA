# 40ep 的 8×8 特征库构建

本次按用户“开始实现 有问题越早问用户越好 一口气全做完！”的指令执行已确认计划。代码为 `CAND=c08ec2060a544af1869c1e24f755e536150569ca`；实际起跑 HEAD 由日志及 `store_meta.json` 的 `packer.git_commit` 同时记录。4×4 在线回归已全部通过，当前阶段只从现有源特征打包，不抽取模型特征，不修改旧库。

环境为 AWS 本地 NVMe RAID `/dev/md0`，本阶段纯 CPU，显式 `CUDA_VISIBLE_DEVICES=''`。源目录为 `v1-store/datasets/4task-motion-40ep/source`，清单为同库 `meta/episode_manifest.json`，输出为尚不存在的实体目录 `v1-store/datasets/4task-motion-40ep/framesamp-8x8`。清单 SHA 为 `d7cfb137b6ba01c42894e2d6d421a8c3f87dc1afeef1ac650563609bd7501d05`，共 40 集、13,756 帧、11,530 个执行样本，位置表 586 行。

采用 `framesamp-8x8-v1`、`--reader decode`、48 个进程。预期 image 为 3,606,052,864 B，pos 为 115,212,288 B，state 为 440,192 B，逐行摘要为 220,096 B；part 边界应与旧库的 22 个 part 相同。image/pos/state 的 dtype 分别为 bf16/f32/f32，打包与读回不改数值。

```bash
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
L="$V1_STORE/datasets/4task-motion-40ep"
uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack \
  --layout framesamp-8x8-v1 --reader decode --source "$L/source" \
  --manifest "$L/meta/episode_manifest.json" --out "$L/framesamp-8x8" --procs 48
uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify \
  --store "$L/framesamp-8x8" --resume --procs 48
uv run --no-sync python scripts/dataset/pack_framesamp_store.py report --store "$L/framesamp-8x8"
uv run --no-sync python scripts/dataset/xgrid_pos_check.py \
  --store-4x4 "$L/framesamp" --store-8x8 "$L/framesamp-8x8" --source "$L/source" --image-spot 512
```

pack 与 verify 分别放入 detached tmux 会话 `p8-pack-40`、`p8-verify-40`，使用 `set -o pipefail`、`tee`，日志分别为 `v1-store/logs/p8-pack-40.log`、`p8-verify-40.log`，尾行记录 `EXIT_CODE=`。启动前要求 clean HEAD，并确认输出不存在；verify 只接管本轮 pack 完成后留下的锁。

判据为全量 `VERIFY_PACK=PASS scanned=13756 mismatches=0`、独立 `IMAGE_NPY_SPOT=PASS frames=512 mismatches=0`、完整 `MOTION_POS_XGRID=PASS t=586 npy=64 mismatches=0`，以及新旧 part 边界完全一致。后续完整 fixture 和 worker 生命周期另行验收，不用建库成功替代训练交付等价。
