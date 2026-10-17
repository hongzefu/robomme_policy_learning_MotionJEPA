# 建库判定行汇总（B1 → B4 试训，美东 2026-10-16 08:06 → 2026-10-17 00:56）

本文件只收 git 还原不出的判定行原文；完整日志在 `v1-store/logs/`，第二段叙述见 result.md（上传完成后补写）。窗口 HEAD（SigLIP 起跑 → motion verify）全程 `6ebeab5632340328a3a62db836d1ab3b004eed57`，期间无提交。

## B1 与源数据

```
B1_GATE=PASS exit=0 epochs=72 ckpt_sha=bec423fdfa0c9494690a5118ab10f2cb6fec58eba4b8362b4af90c71cbcd9e04 manifest_sha=e3c7564f tmux_gone=1 gpu_idle=1 ema_only=1
G3=PASS enc_copy_sha=bec423fdfa0c9494690a5118ab10f2cb6fec58eba4b8362b4af90c71cbcd9e04
MERGE_VERIFY=PASS level=full tasks=2 episodes=400 timesteps=314737 role=primary:400 mismatches=0      # A3 / G1
FETCH_VERIFY=PASS ×16，archives 合计 100                                                              # m8v-fetch
PLAN_OK tasks=16 selected=3200 role=primary:3200 max_timesteps=1989 prechecks=PASS precheck_files=3200
MERGE_TASK=PASS ×16
MERGE_VERIFY=PASS level=full tasks=16 episodes=3200 timesteps=2145548 role=primary:3200 mismatches=0   # G2
CLEANUP=PASS removed=…/staging/mixed8-200-src/extracted files=3200 verify_episodes=3200 timesteps=2145548
TRAIN_ES_BOUND=PASS episodes=3200 tasks=16 max_windows=123 budget=160 headroom=37                      # G2a；最坏 PickXtimes:196 / PickXtimes_ep175_seed18500.h5
```

canonical manifest `d6a3ddb2b579b5cf50cd66fbefedd36478766e2b4cc07833a7efae29769742ad`；episode_manifest.json 文件 sha256 `c6a063dfbfc6da404dd8f1d3ddc38e9ba6e25abca81417fe742872401e7f97df`。

## SigLIP / finalize / 打包

```
STAGE_DONE stage=siglip workers=32 items=3200 elapsed=6673s      # 32 worker 为实施中调整，见计划进展登记 21
SHARD_DONE ×32；_shard*of32.json ×32；sidecar episode 并集 3200；指纹唯一（git_commit=6ebeab5、jax 0.5.3、A100）；残留 claim 0
✓ 同架构复算全部逐位一致 PASS（1024 条）；全部检查通过 PASS；FINALIZE_EXIT_CODE=0
VERIFY_PACK=PASS scanned=2145548 mismatches=0      # 4×4
VERIFY_PACK=PASS scanned=2145548 mismatches=0      # 8×8；num_rows=2145548 num_exec_samples=1383846 parts=32
IMAGE_NPY_SPOT=PASS frames=512 mismatches=0
MOTION_POS_XGRID=PASS t=1989 npy=64 mismatches=0
FEATURES_PRUNE=PASS kept=8      # 删除前后 headtail blake2b-128 复核 8/8
```

norm_stats `36ee9acdea4c8383b50c7b07b45a740efb3fa073f6e60455977d5a6fb3204394`（64 个值全部有限）；framesamp-8x8 store_meta `ace85246419836d343b925ac45591e3fd75eb2f054c16eaafafb3b35d6ca9b79`。

## Wan / encode / oracle / motion

```
STAGE_WORKSET=PASS stage=wan processed=4995 duplicates=0 missing=0；STAGE_DONE stage=wan workers=8 items=4995 elapsed=23119s；窗合计 128342
STAGE_WORKSET=PASS stage=encode processed=4995 duplicates=0 missing=0；STAGE_DONE stage=encode workers=8 items=4995 elapsed=229s
ORACLE_ENCODER=DONE rows=128342 elapsed=1281s
ORACLE_SHARDS=PASS ok=8/8；ORACLE_VAE=DONE metadata_rows=128342 windows=14480 frame_mismatches=0 metadata_mismatches=0；ORACLE_AGGREGATE=DONE kinds=vae shards=8
VERIFY_MOTION=PASS scanned=128342 mismatches=0
WAN_BITEXACT=PASS compared=14480 padded_covered=1800 frame_mismatches=0 latent_mismatches=0 metadata_mismatches=0 oracle_windows=14480
ENCODER_BITEXACT=PASS compared=128342 mismatches=0 order_ok=1 state_sha_ok=1 prov_ok=1 ckpt_ok=1 no_latent_stats_call=1 finite=1
A6_SAMESOURCE=PASS episodes=3200 manifest_sha_same=1
A7_BYTES=PASS segments=4995 mismatches=0 table_rows=128342 table_ok=1
A9_INDEXSET=PASS samples=3700 cold=3200 mismatches=0
A10_ROWS=PASS rows=128342 exec=81611 demo=46731 episodes=3200 formula_or_rowbase_mismatches=0
A11_PAD=PASS segs=4995 rows=128342 padded=1800 set_diff=0      # 首跑 --expect-episodes 3200 判 FAIL（口径错误），用户「a」后改 1800 只重跑 A11
```

oracle vae 首跑因 env.sh 覆盖 HF_HOME 失败、修正后补跑通过（计划进展登记 22）。motion store_meta `c9f8275fe12d6395ea7cc29191e39a1fe5df7b569c579b5acdfea07a71bdde2e`，layout `motion-768-grid16-demopad17-v1`。

## B4 试训与数据身份

```
MOTION_4GPU_SMOKE=PASS run=512 steps=20 exit=0 finite_losses=20/20 peak_mem_gib=39.3 limit_gib=72 mesh=TRAIN_MESH shape=(1, 4) devices=4 fsdp_devices=4 batch_size=128 num_workers=8
MOTION_4GPU_SMOKE=PASS run=motion steps=20 exit=0 finite_losses=20/20 peak_mem_gib=38.3 limit_gib=72 mesh=TRAIN_MESH shape=(1, 4) devices=4 fsdp_devices=4 batch_size=128 num_workers=8
DATA_IDENTITY=PASS norm_sha=36ee9acdea4c8383b50c7b07b45a740efb3fa073f6e60455977d5a6fb3204394 steps=2145548 samples=1383846 manifest_sha=e3c7564f787781910f741d57b487d45565defbe8d3c2c17af9233c651093de11
```

两条短测同时 4+4 并行（含互扰），`XLA_PYTHON_CLIENT_PREALLOCATE=false`，nvidia-smi 500 ms 采样。稳态步时（第 6→18 步）512 约 1.84 s/步、motion 约 1.90 s/步，80k 分别约 40.9 h / 42.2 h；未见本机 GPU 中断不送达对 JAX 训练的明显拖累。决定：4+4 并行，无串行回退。

## 上传

- latent 库 `HongzeFu/robomme-16task-mixed8-200-motion-20261014-v1`：HYGIENE=PASS scanned=132 hits=50085 ungated=0 count_mismatch=[]；SHA256_PRE_POST_DIFF=0；RESULT=PASS；BUCKET_PUBLIC=True；全部完成；EXIT_CODE=0（5120 文件 1,139,255,338,776 B，private=false），回读目录已删。
- VLA 新库 `HongzeFu/robomme-16task-mixed8-200-trainset-20261016-v1`：SMOKE=PASS（MEMBERS_SPOT=PASS tars=2 members=8192 mismatches=0；HYGIENE=PASS scanned=25040 hits=40245 ungated=0）；全量 tmux `m8v-trainset-upload` 美东 00:52 起，进行中。
