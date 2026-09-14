# 400ep 的 8×8 特征库构建

按用户“一口气全做完”的指令，在40ep全部库层和装配层验收通过后构建正式400ep新库。候选代码固定为 `c08ec2060a544af1869c1e24f755e536150569ca`，实际启动HEAD为其后仅含档案提交的clean HEAD，由日志与store_meta同时记录。

源为 `v1-store/datasets/4task-motion-400ep/source`，清单为同库 `meta/episode_manifest.json`，清单SHA `92fa17e97fba9434ee75302de12556319d8ce6d3feeb3adb9a397e830f477223`；400集、123044帧、101066个执行样本、586行位置表。输出为新增实体目录 `framesamp-8x8/`，不覆盖现有 `framesamp/`、motion或源库。

纯CPU，48进程，AWS本地NVMe RAID `/dev/md0`，显式使GPU不可见。decode完整反序列化源npy，按 `framesamp-8x8-v1` 打包；image/pos/state的dtype为bf16/f32/f32，全过程不改数值。预期image 32255246336 B、pos 115212288 B、state 3937408 B、行摘要1968704 B；31个part与旧库边界一致。

```bash
source scripts/training/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
L="$V1_STORE/datasets/4task-motion-400ep"
uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack \
  --layout framesamp-8x8-v1 --reader decode --source "$L/source" \
  --manifest "$L/meta/episode_manifest.json" --out "$L/framesamp-8x8" --procs 48
uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify \
  --store "$L/framesamp-8x8" --resume --procs 48
uv run --no-sync python scripts/dataset/pack_framesamp_store.py report --store "$L/framesamp-8x8"
uv run --no-sync python scripts/dataset/xgrid_pos_check.py \
  --store-4x4 "$L/framesamp" --store-8x8 "$L/framesamp-8x8" --source "$L/source" --image-spot 512
```

会话分别为 `p8-pack-400`、`p8-verify-400`，日志为 `v1-store/logs/<会话名>.log`，外层pipefail、tee、EXIT_CODE。启动前检查新输出不存在，verify只接管本轮pack锁。

要求 `VERIFY_PACK=PASS scanned=123044 mismatches=0`、`IMAGE_NPY_SPOT=PASS frames=512 mismatches=0`、`MOTION_POS_XGRID=PASS t=586 npy=64 mismatches=0`，以及31个part边界一致、image字节四倍、state摘要相同。随后在本库上进行四profile完整输入验收，不能以40ep结果替代400ep覆盖。
