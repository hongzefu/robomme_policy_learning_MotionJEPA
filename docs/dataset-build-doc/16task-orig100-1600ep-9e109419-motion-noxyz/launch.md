# orig100 noxyz motion 表启动记录

## 正式建表：版本与代码状态

完整代码已整合，最终 CPU 范围为 962 通过、7 跳过，两组分别 140.58 秒、96.00 秒，均退出 0。默认旧身份入口及契约在同一当前环境、旧真实资产下 stdout/stderr 逐字节一致。业务代码为 `2b3c5cd`（后续仅启动档案），正式建表起跑前提交 `commitV11.82Beta`；实际 Beta 全 SHA 由日志 `BUILD_START` 记录，起跑时工作区干净。VLA 的训练 Beta 仍在两表完成后单独提交。

两 encoder 的源配置、权重 SHA 不变。导出 runner 来自固定 `4a16ab42a0ef5d70b9658e296b4944180842f248`，入库版本只增加显式 `--lineage-build-spec`；来源清单与四处可还原差异随 Beta 还原。实际 WAN 包提交为 `b5d23a5fab55be16e580c5b77a8f04d68b3170c2`，两核心模块 SHA 匹配仓库 `SOURCE_PIN.json`。来源元数据完整 16 份 H5 名字、字节数、SHA 已预检一致。

## 正式建表：启动与配置还原

本任务 tmux 为 `o16abl-build-noxyz`，日志 `/data/hongzefu/robomme_policy_learning_MotionJEPA-gl/v1-store/logs/o16abl-build-noxyz.log`，运行载体 `/data/hongzefu/robomme_policy_learning_MotionJEPA-gl/v1-store/logs/o16abl-build.sh` SHA256 为 `256969bba47332cb5df6791dd1f15d0245d5e1446016725039c0c052decf2768`，静态 `RUNTIME_REVIEW=PASS findings=0`。主会话先启动 noxyz 并核到锁已取得，再启动 nostate。完整启动命令：

```bash
tmux new-session -d -s o16abl-build-noxyz "set -o pipefail; PYTHONUNBUFFERED=1 bash /data/hongzefu/robomme_policy_learning_MotionJEPA-gl/v1-store/logs/o16abl-build.sh noxyz 2>&1 | tee /data/hongzefu/robomme_policy_learning_MotionJEPA-gl/v1-store/logs/o16abl-build-noxyz.log; rc=\$?; echo EXIT_CODE=\$rc >> /data/hongzefu/robomme_policy_learning_MotionJEPA-gl/v1-store/logs/o16abl-build-noxyz.log; exit \$rc"
```

载体只按下列确定参数顺序调用 Beta 中的生产入口（所有原始命令可由此重建，档案不另存脚本或 YAML）：

```bash
P=/data/hongzefu/robomme_policy_learning_MotionJEPA-gl
LIB=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-orig100-1600ep-9e109419
TAG=noxyz
GPU=0
SHA=6acc92af239e9b9a4c58ced4c5123bc25183188ec518b10734e917879240d00d
RUN=aws-stage2-noxyz-1101-a
LATENTS="$P/v1-store/external/o16abl-latents-import/wan-latents"
RGB="$P/v1-store/external/rgb-gtxyz-g3-subset"
ENCODER="$P/v1-store/external/motionjepa/$RUN"
EVIDENCE="$LIB/evidence-$TAG"
TOKENS="$LIB/motion-tokens-stage2-$TAG"
TABLE="$LIB/motion-stage2-1024-$TAG"
WANPY="$P/v1-store/venvs/wan-stage2/bin/python"
export UV_CACHE_DIR="$P/v1-store/cache/uv" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$P"
exec 9> "$P/v1-store/logs/o16abl-g3.lock"
flock 9
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$GPU" "$WANPY" scripts/dataset/wan_stage2/export_stage2_memory.py --lineage-build-spec "$RGB/control/build_spec.json" --ckpt "$ENCODER/checkpoint_epoch_105.pt" --expected-sha256 "$SHA" --latents "$LATENTS" --manifest "$LIB/meta/episode_manifest.json" --out "$LIB/motion-tokens-stage2-$TAG-smoke-1102-a" --grid-subsample 2 --batch 1 --check-dropout 0 --max-segments 1 --expected-counts 2496,23253 --input-manifest "$LIB/meta/input_manifest.json"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$GPU" "$WANPY" scripts/dataset/wan_stage2/export_stage2_memory.py --lineage-build-spec "$RGB/control/build_spec.json" --ckpt "$ENCODER/checkpoint_epoch_105.pt" --expected-sha256 "$SHA" --latents "$LATENTS" --manifest "$LIB/meta/episode_manifest.json" --out "$TOKENS" --grid-subsample 2 --batch 1 --check-dropout 64 --expected-counts 2496,23253 --input-manifest "$LIB/meta/input_manifest.json"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 "$WANPY" scripts/dataset/wan_stage2/validate_stage2.py --gate G3 --samples "$EVIDENCE/G0.json" --ckpt "$ENCODER/checkpoint_epoch_105.pt" --expected-sha256 "$SHA" --rgb-root "$RGB" --mj-root /data/hongzefu/MotionJEPA --cache-import-receipt "$EVIDENCE/wan_cache_import.json" --out "$EVIDENCE/G3.json"
flock -u 9
PYTHONPATH="$P/src" "$P/.venv/bin/python" scripts/dataset/pack_motion_store.py pack --layout stage2 --latent-grid-subsample 2 --latents-origin published-cache-with-local-fill --cache-import-receipt "$EVIDENCE/wan_cache_import.json" --manifest "$LIB/meta/episode_manifest.json" --tokens "$TOKENS" --latents "$LATENTS" --out "$TABLE" --encoder-run-dir "$ENCODER" --raw-dir /data/hongzefu/data_0226 --expected-ckpt-sha256 "$SHA" --stage2-validation "$EVIDENCE" --stage2-source-pin scripts/dataset/wan_stage2/SOURCE_PIN.json
PYTHONPATH="$P/src" "$P/.venv/bin/python" scripts/dataset/pack_motion_store.py verify --store "$TABLE" --latents "$LATENTS" --resume --check-padding
```

## 正式建表：输入划分、关键参数与预期判定

R1 的 13 tar、7,489 成员全量 SHA、共享盘同步后 SHA 已通过且退出 0；G3 的 32 窗、14 分片、115 对象、32,476,096,020 字节也全部通过且退出 0。两输入实际完成后才启动本次 GPU 工作。按本机快盘优先要求，B4/B7/B8 的三个已有 latent 参数读已核验的本机副本，表与证据输出仍在共享盘；该位置调整已独立 `ADJUST_REVIEW=PASS scale=same findings=0`。

先单段实际编码冒烟（不做 dropout 诊断），成功后才完整导出 2,496 段、23,253 行并做 64 窗 dropout 诊断。输出 `<f4`、每行 `(1,1024)`、总 95,244,288 字节。G3 沿用户批准的缓存路线检查 32 窗形状、有限性与 65 叶 EMA 身份，输出数值仍按既定口径标 `NOT_REQUIRED`。原四份证据和在线输入两文件已各 tag 暂存；G3 后七份齐全。打包读取本机永久原始 H5，仅做完整 SHA，不写原件。

验收分别记录 `ENCODER_SMOKE=PASS`、`STAGE2_EXPORT=PASS segments=2496 rows=23253`、`EMA_SOURCE=PASS`、`PACK_PROVENANCE_PRECHECK=PASS`、`VERIFY_MOTION=PASS scanned=23253 mismatches=0 nonfinite=0`，末行 `BUILD_DONE=PASS tag=noxyz` 和外层 `EXIT_CODE=0`。任一命令非零，该 tag 不继续后续步骤。

## 正式建表：硬件与盯盘

Aspen 2×RTX A6000，导出 noxyz 使用物理 GPU 0；G3 统一 GPU0，noxyz 持锁至其 G3 完成，nostate 导出后取锁。CPU 线程变量为 1，两个 tag 的原始 H5 与 latent 只读、输出目录独立。现场另有他人服务的小显存上下文，本轮不干预，耗时按共享现场记录。预计两 encoder 导出并行，G3 串行，pack/verify 各自完成；实际耗时、故障原文、退出码和最终指标写 result.md。当前只登记启动口径，不预写实际完成。


## 目的、输入与版本

对应 encoder 为 `aws-stage2-noxyz-1101-a/checkpoint_epoch_105.pt#model`，已下载且原始 config、权重和共享副本 SHA 核验通过。目标表为训练库内 `motion-stage2-1024-noxyz`，目标证据目录为 `evidence-noxyz`；另一消融的输出目录独立。

本轮只准备两个已批准建表任务共用的输入，尚未执行 GPU 导出、G3、打包或训练。执行宿主为 Aspen 环境 A，两张 RTX A6000；输入下载不占 GPU。实施副本为 `/data/hongzefu/robomme_policy_learning_MotionJEPA-gl`，分支 `v4-gl-orig100-1031`，latent 恢复代码锚点为 `77f1061323b015daa4f3d4a050160b30b0bab162`。起跑时的实际 HEAD 由日志 `ASSET_START_HEAD=` 记录，起跑前工作区必须干净；后续阶段各记实际确定版本，不把本次下载提交冒充后续建表版本。

训练库根为 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-orig100-1600ep-9e109419`。源公开 bucket 为 `HongzeFu/robomme-16task-orig100-1600ep-trainset-20261028-v1`，沿库内 `delivery_manifest.json` 选择 13 份 latent tar、逐成员 SHA 文件和上传成员清单，共 15 对象、26,917,887,511 字节。独立暂存根为 `/data/hongzefu/robomme_policy_learning_MotionJEPA-gl/v1-store/external/o16abl-latents-import`；核验后只同步 `wan-latents/` 至训练库，不覆盖现库 `delivery_manifest.json` 或 `.fetch-state`。

两个任务共享且只启动一次 tmux `o16abl-fetch-latents`，日志 `/data/hongzefu/robomme_policy_learning_MotionJEPA-gl/v1-store/logs/o16abl-fetch-latents.log`。主会话编写的运行载体为同目录 `o16abl-fetch-latents.sh`；真实生产逻辑来自已提交的 `scripts/dataset/hf_export/fetch_orig100_trainset.py`，顺序如下：

```bash
P=/data/hongzefu/robomme_policy_learning_MotionJEPA-gl
LIB=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-orig100-1600ep-9e109419
IMPORT="$P/v1-store/external/o16abl-latents-import"
TOOL="$P/scripts/dataset/hf_export/fetch_orig100_trainset.py"
export UV_CACHE_DIR="$P/v1-store/cache/uv" PYTHONUNBUFFERED=1
"$P/.venv/bin/python" "$TOOL" --ipv4 plan --manifest "$LIB/delivery_manifest.json" --include wan-latents_tars/ checksums/sha256-source-files.txt manifest/upload_manifest.json --out "$IMPORT/plan.json"
"$P/.venv/bin/python" "$TOOL" --ipv4 fetch --plan "$IMPORT/plan.json" --out "$IMPORT" --workers 8
"$P/.venv/bin/python" "$TOOL" unpack --lib "$IMPORT" --workers 4
"$P/.venv/bin/python" "$TOOL" verify --lib "$IMPORT" --deep
rsync -a --info=progress2 "$IMPORT/wan-latents/" "$LIB/wan-latents/"
# 同步后按已发布逐成员 SHA 再核共享副本；日志保留对象和成员总数。
```

上块的 `P`、`LIB`、`IMPORT` 分别就是上文绝对路径；完整执行命令是 `bash /data/hongzefu/robomme_policy_learning_MotionJEPA-gl/v1-store/logs/o16abl-fetch-latents.sh`，由 detached tmux 配 `pipefail`、`PYTHONUNBUFFERED=1`、`tee` 和 `EXIT_CODE=` 包装。无需新增依赖，不用裸系统 Python。预期判定为原工具 `TRAINSET_FETCH=PASS`、`TRAINSET_UNPACK=PASS`、`TRAINSET_VERIFY=PASS`，以及共享副本 `LATENTS_SYNC=PASS members=7489 mismatch=0`，退出码必须为 0。未验证前不写成完成。

## 用户决策与后续阶段

用户原话依次为「/data/hongzefu/robomme_policy_learning_MotionJEPA/1101-gl-ablation-encoders-ctx-b64-80k-plan.md 开工」「同意修订 继续做」「不需要再确认，你可以直接开始了。」。计划 v5 `2e9b8dc09acf5a5518c95284da9658c1961fd1c9` 独立一致性审查通过。现有工具无法解 latent 的缺口经独立 `ADJUST_REVIEW=PASS scale=same findings=0` 后最小修复；source 原默认行为和 SHA 校验不变。G3 子集、显式来源路径、固定 runner、两表构建与 pack/verify 的确切版本和命令在各自起跑前补充，不在本次资产准备中伪填已启动。完成后另写 result.md 和不可由 Git 还原的 records；权重、数据、日志原件仍在 v1-store，不归档脚本或 YAML 副本。


## G3 子集输入准备起跑

真实规划为32窗、14片、115对象、32,476,096,020字节；规划阶段只下载了约4.3MB元数据。`control/build_spec.json` 发布SHA为 `5943608ef0f1a5febe79069d6081c7427c6f149f071cdf397c0d21a1b567ef7a`，四索引与14份DONE也逐文件核对通过。下载工具代码锚点 `2b3c5cd`，原四测试及四线程HTTP回归共6项通过，合前独立审查通过。

两个建表任务共用一次 `o16abl-g3fetch` tmux，日志 `P/v1-store/logs/o16abl-g3fetch.log`；完整入口为 `bash P/v1-store/logs/o16abl-g3fetch.sh`，载体先记录实际启动HEAD，再只运行下列命令（P是上文绝对实施副本路径）：

```bash
UV_CACHE_DIR=/data/hongzefu/robomme_policy_learning_MotionJEPA-gl/v1-store/cache/uv PYTHONUNBUFFERED=1 \
/data/hongzefu/robomme_policy_learning_MotionJEPA-gl/.venv/bin/python \
/data/hongzefu/robomme_policy_learning_MotionJEPA-gl/scripts/dataset/wan_stage2/plan_g3_subset.py \
fetch --root /data/hongzefu/robomme_policy_learning_MotionJEPA-gl/v1-store/external/rgb-gtxyz-g3-subset --workers 4
```

运行使用pipefail、tee与EXIT_CODE尾行，终点 `G3_SUBSET=PASS shards=14 bytes=32476096020 sha_mismatch=0` 且退出0；预计1–2小时，下载对象和预算不扩张。完成前保留为进行中。两个encoder的输入来源已用固定runner显式build_spec路径做CPU预检，完整16份H5名字/字节数/SHA集合一致；一次主会话调用误用了位置参数，TypeError原文与改为关键字后的真实结果保存在 `P/v1-store/logs/o16abl-portable-lineage.log`，代码与判据未改。
