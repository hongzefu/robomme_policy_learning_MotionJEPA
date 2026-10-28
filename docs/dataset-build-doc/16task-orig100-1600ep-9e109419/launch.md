# 官方16×100：双帧表与阶段二motion建库启动档案

## 一、目的与当前状态

按[批准计划](../../../1027-orig16-vla-512-and-stage2-motion-4plus4-plan.md)构建全新库`16task-orig100-1600ep-9e109419`，供先context、后modulation两条8卡60k训练消费。两种帧表全量构建；motion只落一张`[1,1024]`表，concat与split共用。当前为建库Beta启动档案，数据验收及训练尚未完成。

库名后缀取发布源清单`pin_sha256=9e109419d08828d46261d45ca88786207015752d4b3c54b67a1ce8554fcf96f6`的前八位；不拿这一字符串代替实际源H5核验。

## 二、版本与代码状态

| 对象 | 固定版本或身份 |
|---|---|
| 批准计划 | `d5fe05d647a899e016bc513aa5fba03485a9b398` |
| 全部代码整合完成 | policy `58d3b5c47731f68fd085b81f68595e641b5ec6fc`，工作区干净，已推送 |
| 建库代码 | 本档案所在`commitV11.66Beta`；实际完整SHA在起跑时写入`v1-store/logs/o16-20261028/build-start.json`及各日志`BUILD_HEAD=` |
| MJ导出CLI | `50a853492df32e6aa91e2d523acc33bc34257f4a`，`/scratch/hongze/worktrees/vla-s2-memexport-20261026` |
| MJ共享库 | stage2独立环境内固定安装`b5d23a5fab55be16e580c5b77a8f04d68b3170c2`；不设`PYTHONPATH`覆盖它 |
| encoder | `stage2-full144-1025-a/checkpoint_epoch_105.pt`，EMA，SHA256 `27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035` |
| encoder源pin | `31d6ab0f845336e76dc25db64170a0bc4ac59ecf366c161025b9800285c74467` |
| 官方H5 | `Yinpei/robomme_data_h5@a5e4e25ffe8af34f64944f9533d06455ce5f8337` |
| 发布Wan缓存 | `HongzeFu/robomme-original-16x100-wan21-front-33f-dense-v1@384d5db410be713a49357bf3f55ca4f4d2f559a0` |

Beta至`VERIFY_MOTION=PASS`冻结主检出HEAD并写`AGENTMETARULES_HOLD`。期间状态写入`/scratch/hongze/robomme_policy_learning_MotionJEPA-temp/docs/handoff/current.json`，专属本地分支`worktree-agent-o16-state-20261028`；该状态分支随解冻后合并一起推送，不修改冻结主检出。

起跑前归档复核：首份Beta `a0c1948`之后发现复制的矩阵日志含原程序打印的行尾空白，尚未启动任何数据步骤。本次仅清理归档副本的行尾空白并重新固定实际起跑Beta；判定字段、数值、代码与运行参数不变，原日志仍保留在S7工作树。实际起跑SHA以`build-start.json`及日志`BUILD_HEAD=`为准。

## 三、启动、配置与完整命令

所有命令在主检出执行。长步骤使用独立`o16-`前缀tmux、`PYTHONUNBUFFERED=1`、`tee`和真实`EXIT_CODE=`。运行包装脚本只是以下命令的调度载体，位于日志根，不复制进档案；固定SHA为：

| 载体 | SHA256 |
|---|---|
| `prepare-manifest.sh` | `042c427723a85aeaf7a573628c17e270ca67e3a8125176ac35022135cfaecb2b` |
| `fetch-cache.sh` | `81de6a0f6028980e927aec331dacc7c021fd4abc9f78d7c680920a9ca1722cde` |
| `import-cache.sh` | `b167ba56e2890fcb3cc27ab682fc0b77c1bb8169d34e0965d0af31d9deac318c` |

```bash
P=/scratch/hongze/robomme_policy_learning_MotionJEPA
cd "$P"
export MJ_REPO=/scratch/hongze/MotionJEPA
source scripts/dataset/paths.sh
export UV_CACHE_DIR=/scratch/hongze/.cache/uv PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RAW=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/source/h5
MJIDX=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/index
MJSRC=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/source_manifest.json
MJW=/scratch/hongze/worktrees/vla-s2-memexport-20261026
RGBROOT=/scratch/hongze/dataset-rgb-gtxyz-16views-stride2-v1
NAME=16task-orig100-1600ep-9e109419
LIB="$V1_STORE/datasets/$NAME"
MANI="$LIB/meta/episode_manifest.json"
INMANI="$LIB/meta/input_manifest.json"
VALID="$V1_STORE/validation/orig16-stage2"
CKPT=/scratch/hongze/MotionJEPA/runs/stage2-full144-1025-a/checkpoint_epoch_105.pt
CKPT_SHA=27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035
G8=0,1,2,3,4,5,6,7

# CPU阶段：实际读源文件，随后检查三源同源。
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/scan_manifest.py build --raw_dir "$RAW" --episodes-per-task 100 --num_shards 1 --out "$MANI"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/finalize_checks.py hash-inputs --raw_dir "$RAW" --out "$INMANI"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/check_data_lineage.py lineage --encoder-ckpt "$CKPT" --expected-ckpt-sha256 "$CKPT_SHA" --vla-input-manifest "$INMANI" --mj-source-manifest "$MJSRC"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/wan_stage2/orig16_wan_cache.py plan --manifest "$MANI" --input-manifest "$INMANI" --published-index "$MJIDX" --out "$VALID/wan-cache-plan.json"
HF_HUB_OFFLINE=0 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/wan_stage2/orig16_wan_cache.py fetch --plan "$VALID/wan-cache-plan.json" --revision 384d5db410be713a49357bf3f55ca4f4d2f559a0 --out "$LIB/cache-source/official-wan"

# import-cache.sh先用read_window/make_encoder对首个真实fill窗做一窗shape/finite/VAE身份短测。
CUDA_VISIBLE_DEVICES=$G8 UV_PROJECT_ENVIRONMENT="$V1_STORE/venvs/wan-stage2" uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/orig16_wan_cache.py import --plan "$VALID/wan-cache-plan.json" --cache-root "$LIB/cache-source/official-wan" --raw-dir "$RAW" --out "$LIB/wan-latents" --receipt "$VALID/wan-cache-import.json" --gpus "$G8"
# WAN_CACHE_IMPORT与WAN_INPUT_COVERAGE通过后，核确切目录再删除本轮临时cache-source/official-wan；不删其他缓存或源H5。

# SigLIP先单episode短测，随后按原计划每GPU两个worker构建完整source。
CUDA_VISIBLE_DEVICES=$G8 uv run --no-sync python scripts/dataset/run_local.py --stage siglip --lib "$LIB" --gpus 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7 --raw-dir "$RAW" --require-free-mib 70000
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python scripts/dataset/finalize_checks.py check --manifest "$MANI" --out "$LIB/source" --raw_dir "$RAW" --input_manifest "$INMANI" --input_level sha256 --spot_check 1024
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --layout framesamp-4x4-v1 --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp-4x4" --procs 48
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py pack --layout framesamp-8x8-v1 --source "$LIB/source" --manifest "$MANI" --out "$LIB/framesamp-8x8" --procs 48
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --layout framesamp-4x4-v1 --store "$LIB/framesamp-4x4" --resume --procs 48
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_framesamp_store.py verify --layout framesamp-8x8-v1 --store "$LIB/framesamp-8x8" --resume --procs 48
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/compute_norm_stats.py --output-dir "$V1_STORE/train-assets/mme_vla_suite/$NAME" --config-name mme_vla_suite --repo-id robomme --dataset-path "$LIB/source"

CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/check_stage2_validation_inputs.py --manifest "$MANI" --latents "$LIB/wan-latents" --source "$LIB/source" --h5-root "$RAW" --input-manifest "$INMANI" --mj-index "$MJIDX" --cache-import-receipt "$VALID/wan-cache-import.json" --out "$VALID/G0.json"
# 2496,23253是索引推算值，执行前与实际manifest/cache plan复核；先另目录max-segments=1短测。
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 "$V1_STORE/venvs/wan-stage2/bin/python" "$MJW/scripts/stage2_memory_export.py" --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --latents "$LIB/wan-latents" --manifest "$MANI" --out "$LIB/motion-tokens-stage2" --grid-subsample 2 --batch 1 --check-dropout 64 --expected-counts 2496,23253 --input-manifest "$INMANI"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/wan_stage2/orig16_wan_cache.py check --samples "$VALID/G0.json" --receipt "$VALID/wan-cache-import.json" --raw-dir "$RAW" --out "$VALID/G2.json"
CUDA_VISIBLE_DEVICES=0 UV_PROJECT_ENVIRONMENT="$V1_STORE/venvs/wan-stage2" uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/validate_stage2.py --gate G3 --samples "$VALID/G0.json" --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --rgb-root "$RGBROOT" --mj-root "$MJW" --cache-import-receipt "$VALID/wan-cache-import.json" --out "$VALID/G3.json"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_motion_store.py pack --layout stage2 --latent-grid-subsample 2 --latents-origin published-cache-with-local-fill --cache-import-receipt "$VALID/wan-cache-import.json" --manifest "$MANI" --tokens "$LIB/motion-tokens-stage2" --latents "$LIB/wan-latents" --out "$LIB/motion-stage2-1024" --encoder-run-dir "$V1_STORE/external/motionjepa/stage2-full144-1025-a" --raw-dir "$RAW" --expected-ckpt-sha256 "$CKPT_SHA" --stage2-validation "$VALID" --stage2-source-pin scripts/dataset/wan_stage2/SOURCE_PIN.json
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python scripts/dataset/pack_motion_store.py verify --store "$LIB/motion-stage2-1024" --resume --check-padding
```

全部环境覆盖由上表/命令及`paths.sh@Beta`还原：HF、XDG、JAX等缓存均在主检出`v1-store`；不覆盖HOME、不访问集群。两份帧表可以各48个CPU进程并行，GPU步骤串行。完整模型短测与正式训练另归两份训练档案。

## 四、数据口径与验收

预核索引为16任务×100集、768,897帧、476,857执行样本；grid16共45,228窗（copy42,404、demo-fill900、exec-fill1,924），grid32为23,253窗/2,496个非空段，最大每集44窗。以上仍须实际清单确认。缓存恢复为2,485个完整bin、388,310,630,400字节，不能按命中窗口的25GB估计下载量。

正式关口依次为磁盘≥1.5TiB、实际`DATA_LINEAGE`、缓存整文件与输入覆盖、SigLIP finalize、两种帧表全量verify、G0/G2/G3、motion pack/verify、全表split逐位、两条G4、G6和配置常量回填。输入/权重/shape/finite/完整性仍严格检查；跨路径精度差记`NOT_REQUIRED reason=user_decision_same_input`。

`VERIFY_MOTION=PASS`后解除构建冻结，回填实际库路径、norm/manifest/input摘要、行数与窗口上界。G4两条分别用正式yaml及`--framesamp-store "$LIB/framesamp-4x4" --h5-root "$RAW" --input-manifest "$INMANI"`，父GPU0、sidecar GPU1。G6消费实际manifest，预算96窗。两条20步必须真实执行并严格保存恢复，不能以DRY_RUN替代。

## 五、硬件、资源与现场句柄

AWS本地NVMe RAID，8×A100-SXM4-80GB。开工预核空闲2.757TiB、8卡空闲；每个GPU步骤起跑时重新确认。没有GreatLakes/Slurm任务。当前尚无本轮运行句柄，计划依次使用`o16-manifest`、`o16-cache-fetch`、`o16-cache-import`、`o16-siglip-smoke`、`o16-siglip`、`o16-finalize`、`o16-pack4`、`o16-pack8`、`o16-norm`、`o16-g0`、`o16-s2export`、`o16-g2`、`o16-g3`、`o16-s2pack`、`o16-g4-ctx`、`o16-g4-mod`；实际启动逐项写入冻结期状态副本，未启动的名字不算运行句柄。

日志根`v1-store/logs/o16-20261028/`，验证根`v1-store/validation/orig16-stage2/`。此轮只清理成功导入后的确切临时发布缓存；不扩展旧数据清理范围。

## 六、代码验收与用户决定

`PLAN_CONSISTENCY_REVIEW=PASS findings=2 blocking=0`；所有代码块完成固定提交合前、合后审查，全部子提交通过`--no-ff`保留。合后关键测试：S6 63项；S1 92项；S2 53项及实际独立环境17项；S3 29项；S4 67项；S7联合261项；S5 66项。集合有重叠，不将这些数字相加当唯一测试数。`MECH_MATRIX=PASS combos=8 contract_ok=8 forward_ok=8`，范围为实际模块单层/FFN64，保留正式长度和维度；完整VLA由后续8卡短测验证。

用户原话：「开工 有问题问用户 尽可能不要阻塞 所有subagent全部astra high」「登记三项任务（推荐）」。沿用计划K1–K6：其他六组合仅CPU验证；全部正式前置通过即自动训练；训练库与checkpoint公开上传；不评估；两条固定run名；四个motion token共用窗口起点时间码。

实施中的依赖补齐、文字统一及各审查修复见[开工记录](../../maintenance/1027-orig16-vla-start.md)和各合并提交。实际数据/资源/失败/耗时及上传结果完成后写`result.md`与`records/`；当前不把预核、CPU夹具或存在性检查写成真实建库通过。
