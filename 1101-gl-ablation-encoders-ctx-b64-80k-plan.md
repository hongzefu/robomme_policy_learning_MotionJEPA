# GreatLakes 4×A40 并行训练两个消融 encoder（去 depth／去 action）的 context VLA 计划（b64、80k）

> **权威性与授权边界**：本计划只规划、不实施。每一步（下载、本机建表、提交占位 job、冒烟、正式起跑）都是开工步骤，须等用户对本计划不改而直接认可后才执行（`AGENTS.md` 第 2 条）。GL 占位 job 的提交也在开工令之内（本计划第一部分 §一 K7 已写明「开工即提交两份长占位」），不再逐份另问。
>
> **代码锚点**：训练代码固定为分支 `origin/v4-gl-orig100-1031` 的 `f1dbe7c9c85ffbf23aaba2bc73321f696bdf4b69`（commitV11.73，含 1031 GL 复现计划已合入的 S1 卡数可变入口、S2 占位运行器、S3 训练库拉取工具、S4 小 batch 冒烟放行）。本计划新增代码在该分支上继续（commitV11.74 起），起跑前打 `commitV11.<n>Beta`。
>
> **工作副本**：本机实施副本 `$P=/data/hongzefu/robomme_policy_learning_MotionJEPA-gl`（环境 A，sled-aspen，2×RTX A6000 48 GB，自带 `.venv` 与 `v1-store/venvs/wan-stage2`）；主检出 `/data/hongzefu/robomme_policy_learning_MotionJEPA`（评估分支 `v4-eval-official-1026`）只放本计划与 `docs/handoff/current.json`；集群副本 `<GL_REPO>=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA`（计算节点唯一可见路径，1031 第一阶段已在 NFS 上建好主 venv 与 wan-stage2 子环境）。训练库 `$LIB=<GL_REPO>/v1-store/datasets/16task-orig100-1600ep-9e109419`（本机 `$P/v1-store/datasets/` 下是指向它的只读 symlink）。
>
> **外部依赖锚点**：训练库公开 bucket `HongzeFu/robomme-16task-orig100-1600ep-trainset-20261028-v1`（K7 子集已在 turbo，`TRAINSET_FETCH/UNPACK=PASS`；本计划再取其 `wan-latents_tars/` 13 个 tar 共 26.80 GB）；两个 encoder 公开 bucket `HongzeFu/motionjepa-aws-stage2-noxyz-ep105-v1`、`HongzeFu/motionjepa-aws-stage2-nostate-ep105-v1`（各 26 对象，`HF_VERIFY=PASS public_anon_read=1`，MotionJEPA commitV8.136 `bb9f9ab`）；G3 子集来自 `HongzeFu/robomme-16task-rgb-gtxyz-stride2-16views-fp16-v1`（11759 文件、2.996 TB，只取索引与 32 片）；MotionJEPA 锁定提交 `b5d23a5fab55be16e580c5b77a8f04d68b3170c2`（`scripts/dataset/wan_stage2/SOURCE_PIN.json` 五键，本机 clone 已含该提交）；原始 H5 本机原件 `/data/hongzefu/data_0226/`（16 文件字节数与 `$LIB/meta/input_manifest.json` 逐一相同，SHA 由打包脚本现场核）。
>
> **commit 体例**：计划与留档 `docs:`；代码 `commitV11.<n>: `（从 11.73 接续）；子代理分支 `sub/<编号>: `；起跑前 `commitV11.<n>Beta`。本计划按用户原话「写md就可以」写成 Markdown，是对 `AGENTS.md` 第 2 条「进仓库 git 的计划一律 HTML」的本次用户显式豁免。

# 第一部分（给人看）

## 一、要做什么与已定口径

一句话方案：把 AWS 刚训完的两个阶段 III 消融 encoder（去 depth、去 action，各取 ep105 EMA）拿到本机，对 orig100 训练库各重算一张 stage2 motion 表（95 MB），然后在 GreatLakes 上用两份 4×A40 长占位 job 并行训两版 context 融合、4 token×256 维 motion 的 VLA，配置沿用 AWS 原 run `orig100-ctx32x4x4-motion-s2split4x256-b128-60k`，只把 global batch 改成 64、步数改成 80 000（warmup 10 000、decay_steps 80 000，lr 5e-5 不动）。

| 编号 | 已定口径 | 依据 |
|---|---|---|
| K1 | 两个 run 并行，各一份 4×A40 占位；不做梯度累积 | 用户 2026-11-01 美东原话「同意并行。不做任何的梯度累积。」 |
| K2 | global batch 先测 64（每卡 16）。**64 档通过 → 不等、不再问，直接在同一占位内起正式 80k 训练**；**64 档 OOM → 同占位内补测 32、16 两档各 20 步取显存峰值，然后 `scancel` 两份占位，停下交用户裁决**，不自动降档、不自动训练 | 用户原话「在 Great Lakes 上先测 batch size，总共六十四，如果不够的话就问用户」「先跑这个六十四的档位」；2026-11-01 修订原话「第二阶段的开工令不需要再问用户，如果六十四这一档不通过的话，就停下来，然后取消你的占位 job，然后问用户。但是在问用户之前，需要把六十四、三十二、十六这三档都测了。如果六十四这档通过了，就直接开始正式训练，就不要在中间等了。」；§三显存事实 |
| K3 | 步数 80 000；warmup 10 000；decay_steps 80 000；peak_lr = decay_lr = 5e-5；EMA 0.999；seed 42；save_interval = keep_period = 5000（16 份 ckpt 含 79999）；fsdp_devices 4、num_workers 8 | 用户原话「步数从一百二十 K 降到八十 K step」；lr 不缩放沿用 `config.py` 既有注释口径（b64→b128 时「不做线性缩放，用延长步数补偿」）；warmup 按样本量等价 5000×128 = 10000×64，且与官方 `mme_vla_suite`（b64、warmup 10000）同值；§二 |
| K4 | 两个新 encoder 身份与表名：去 depth `aws-stage2-noxyz-1101-a/checkpoint_epoch_105.pt` SHA256 `6acc92af239e9b9a4c58ced4c5123bc25183188ec518b10734e917879240d00d` → 表 `$LIB/motion-stage2-1024-noxyz`；去 action `aws-stage2-nostate-1101-a/checkpoint_epoch_105.pt` SHA256 `be7cdec7d96c889193f27cd925977c7ff47423d4321d1fd7446799d38495e3d4` → 表 `$LIB/motion-stage2-1024-nostate` | MotionJEPA `bb9f9ab` 两份 `records/upload-SHA256SUMS.txt`；两个 encoder 的 `config.yaml` SHA `583e48c0…`／`869ac778…` |
| K5 | motion 表全部在本机重算，不与 AWS 沟通：MotionJEPA `b5d23a5` 的 `scripts/stage2_memory_export.py` 单卡导出 → G3 数值闸门 → `pack_motion_store.py pack/verify`；G0、G2 复用库内 `evidence/`（缓存路线与 encoder 无关，`motion_store.validate_stage2_g2_evidence` 带 `cache_import_receipt` 分支不核 ckpt） | 用户原话「不和 AWS 沟通，你在本地进行。」；§四 |
| K6 | G3 不整下 3 TB：只下 `control/build_spec.json` 与 `index/` 四文件（约 3 MB），本地复现 `gate_g3` 的 `default_rng(0).choice(val 候选, 32)` 得到确切 32 窗，再只下所在分片（每满片 2.45 GB，上限 32 片约 78.5 GB）与对应 `masks/source_valid/<task>/epNNN.bits` | reader `rgb_gtxyz_reader.py::PairedWanCacheReader._init_v2` 只校验 4 个索引文件，`_load_shard_full` 按 `row_start // 4096` 打开单片，rgb／gtxyz 只核大小；§四 |
| K7 | 占位方式：**开工即为每个 run 提交一份 4×A40、walltime 8 天（192 h）的占位 job**（`--wrap='sleep infinity'`），排队与本机建表并行；64 档 20 步冒烟、32／16 档显存冒烟、100 步测时、正式 80k 训练全部用 `srun --overlap` 塞进同一份占位，64 档冒烟通过后正式训练**紧接着起跑，中间不等用户**；起跑失败在占位内修复重起，不再排第二次队；64 档 OOM 时按 K2 测完三档后 `scancel` 两份占位；不续跑、不实现 `--resume`。8 天 = 估算 5.1～6.0 天 × 1.3～1.6 余量（步时实测到 8 s/步仍放得下），低于 `spgpu MaxTime=14-00:00:00`（2026-11-01 `scontrol` 实测）；V7 实测步时折算总时长超过占位剩余时间 90% 即停下问用户 | 用户原话「直接申请一个有安全余量的 Great Lakes 的 job，然后万一启动失败了，就不用再排队了，就是你启动的时候以占位 job 的形式启动」；2026-11-02 修订原话「能不能稍微再少一点，就是余量稍微留的稍微少一点」；1031 口径「不续跑」沿用 |
| K8 | run 名 `gl-orig100-ctx32x4x4-motion-s2split4x256-noxyz-b64-80k`、`gl-orig100-ctx32x4x4-motion-s2split4x256-nostate-b64-80k`；冒烟前缀 `o16gl-smoke-`，测时前缀 `o16gl-tier-`；tmux 前缀 `o16abl-` | 沿用 1031 命名法，batch 段按实际值 |
| K9 | ckpt 存 turbo `<GL_REPO>/v1-store/train-runs/mme_vla_suite_orig16_b64_80k/<run>/`，每 5000 步一份共 16 份（约 12 GB × 16 × 2 run ≈ 390 GB，turbo 余量 12 T）；跑完各上传公开 bucket `HongzeFu/robomme-vla-gl-orig100-ctx-s2split-<noxyz|nostate>-b64-80k-v1` 作长尾，读回校验在集群 CPU job 做 | 同 1031 K8 |
| K10 | **一次开工令覆盖全程**：本机建表 + 代码 + 两份占位提交 + 占位内 64 档冒烟 + （通过时）Beta commit 与正式 80k 起跑 + 收尾上传。唯一的停点是 K2 的 64 档 OOM。Beta commit 在占位 RUNNING 前、代码合入与契约 PASS 后就打好，冒烟与正式用同一个 Beta sha | 用户 2026-11-01 修订原话（见 K2）；取代 1031 的两阶段口径 |
| K11 | 任务已登记进 `docs/handoff/current.json`（b019081）：`gl-abl-encoders-phase1`、`gl-abl-encoders-train`（后者 `depends_on` 前者）；K10 修订后 train 任务的开工令与 phase1 同一条，phase1 `done` 即自动进入 train（第 27 条「已有 kind: go 记录可以上游 done 为开工时点」） | 用户原话「登记」；第 27 条 |

## 二、被复现的实验与本次的差别

| 项 | AWS 原 run（`orig100-ctx32x4x4-motion-s2split4x256-b128-60k`，2026-10-29） | 本次两个 run | 差别 |
|---|---|---|---|
| encoder | `stage2-full144-1025-a` ep105，SHA `27daa832…` | K4 两个消融 ep105 | 唯一的实验变量 |
| history yaml | `perceptual-framesamp-ctx-32frame-4x4-motion-stage2-4x256-orig16.yaml`：budget 512、token_per_image 16、32 帧、context、memory_token_dim 2048、motion 4 token×256、budget 384、stride 32、window 33 | 两份新 yaml，只改 `motion.store_path`、`motion.source_run`、`motion.checkpoint_sha256` 三键 | 记忆位 512 + 384 = 896 不变 |
| 训练配置 | `mme_vla_suite_orig16_b128_60k`：B128、FSDP8、workers16、60k、warmup 5000、lr 5e-5、decay 60000 | 新条目 `mme_vla_suite_orig16_b64_80k`：B64、FSDP4、workers8、80k、warmup 10000、lr 5e-5、decay 80000 | 总样本 5.12M = 原 7.68M 的 2/3，与官方 80k×b64 相同；每步样本减半、更新次数增 1/3 |
| 硬件 | 8×A100-80GB，2.43 s/步，40 h 34 min | 4×A40-48GB，估 5.5～6.5 s/步，5～6 天 | §三 |
| 数据 | 库 `16task-orig100-1600ep-9e109419`，1600 集、476857 执行样本、motion 表 23253 行 | 同库同帧表，motion 表换成两张新表（行数仍 23253） | — |
| 对照 | 原 run 59999 官方 800 局 38.88%（1029 评估）；三 seed ctx split 38.42±1.25 | 训完后另立评估计划 | 对照时须带 batch／步数差 |

## 三、显存与耗时事实（决定 K2、K7）

| 卡型 / 分片 | 每卡样本 | 结果 | 出处 |
|---|---|---|---|
| 本机 2×A6000 48G，FSDP2 | 2（global 4） | 跑通，`memory.used` 峰值 39 020 MiB | `$P/v1-store/logs/o16gl-smoke-b4-11011852.mem.csv` |
| 本机 2×A6000 48G，FSDP2 | 4／8／16 | 全部 `RESOURCE_EXHAUSTED`（分别多要 7.15／11.47／14.10 GiB） | `o16gl-smoke-b{8,16,32}-*.log` |
| AWS 8×A100 80G，FSDP8 | 16 | 峰值 42 045 MiB | `docs/training-doc/o16-smoke-ctx-10280431`（分支 v4-256concat1） |
| GL 4×A40 48G，FSDP4 | 16（global 64） | **未测**；参数／优化器分片比本机减半，但激活是本机 OOM 档的 4 倍，按上两行推断大概率 OOM | 本计划 V6 实测 |

⚠ 结论先写明：64 档很可能过不了。按 K2，V6 的 64 档 FAIL 不是本计划失败，而是测完 64／32／16 三档峰值、取消两份占位后把数字交用户裁决；若用户改选 32 或 16，步数等效口径（总样本 5.12M）分别为 160 000／320 000 步，占位要重新提交，但那是另一次裁决与另一次开工令，本计划不预设。64 档 PASS 则按 K10 直接起正式训练。

耗时锚点：历史 `v1-prod-trend-10h`（4×A40、b64、512 位 context、无 motion）4.81 s/步；本次 896 位，按 Gemma prefix 长度近似线性外推 5.5～6.5 s/步；80 000 步 = 122～144 h（5.1～6.0 天）；K7 的 192 h 占位余量 1.3～1.6 倍。两个 run 并行占 8 张 A40，约 1 000～1 150 GPU·h（含冒烟测时）。V7 实测步时若使 80k 总时长超过占位剩余时间的 90%（约 170 h，对应 7.6 s/步），停下交用户，不起正式训练。

## 四、本机建表链路（每个 encoder 一遍，两个可并行各占一张 A6000）

| 步 | 命令锚点 | 输入 | 输出 | 估时 |
|---|---|---|---|---|
| B1 下 encoder | `hf download`（或 `fetch_orig100_trainset.py` 同式样逐对象核 SHA） | bucket 的 `checkpoint_epoch_105.pt`（314 265 827 B）、`config.yaml`、`source_pin/source_receipts.json`、`acceptance/identity.json` | `$P/v1-store/external/motionjepa/<run>/` 与 turbo 同路径各一份 | 分钟级 |
| B2 下 wan-latents | `fetch_orig100_trainset.py fetch/unpack` 加 `wan-latents_tars/` 过滤 | 13 tar、26.80 GB | `$LIB/wan-latents/`（2496 段 `.bin` + metadata，26 676 559 872 B，按 `checksums/` 核成员） | 约 10～20 min |
| B3 MJ 检出 | `git -C /data/hongzefu/MotionJEPA worktree add --detach /data/hongzefu/MotionJEPA-b5d23a5 b5d23a5` | — | `$MJW`，三个源文件 SHA 等于 SOURCE_PIN（已核：`src/motion_jepa/export/stage2_memory.py` = `c9e260ea…`、`wan_latent_encoder.py` = `c9b967f3…`） | 分钟级 |
| B4 导出 | `env -u PYTHONPATH CUDA_VISIBLE_DEVICES=<0|1> $P/v1-store/venvs/wan-stage2/bin/python $MJW/scripts/stage2_memory_export.py --ckpt <ckpt> --expected-sha256 <sha> --latents $LIB/wan-latents --manifest $LIB/meta/episode_manifest.json --out $LIB/motion-tokens-stage2-<tag> --grid-subsample 2 --batch 1 --check-dropout 64 --expected-counts 2496,23253 --input-manifest $LIB/meta/input_manifest.json` | wan-latents、encoder | 2496 个 `.f32.bin` 共 95 244 288 B | AWS A100 8 min，A6000 估 15～20 min |
| B5 G3 子集 | 新工具 `scripts/dataset/wan_stage2/plan_g3_subset.py plan/fetch`（S3）：下 `control/build_spec.json` + `index/` 四文件 → 复现 `gate_g3` 抽样 → 下 32 窗所在分片 7 文件 + 对应 `.bits` | rgb-gtxyz bucket | `/data/hongzefu/dataset-rgb-gtxyz-g3-subset/`（本机 NVMe，≤ 80 GB，两个 encoder 共用，抽样只依赖 seed 0 与 val 候选集） | 下载 1～2 h |
| B6 G3 | `validate_stage2.py --gate G3 --samples $LIB/evidence/G0.json --ckpt <ckpt> --expected-sha256 <sha> --rgb-root <B5 目录> --mj-root $MJW --cache-import-receipt $LIB/evidence/wan_cache_import.json --out $VALID/<tag>/G3.json` | B5、encoder | `G3.json`，须 `windows=32 max_abs=0 leaves=65 exact=1 live_negative_rejected=1` | 分钟级 |
| B7 pack | `pack_motion_store.py pack --layout stage2 --latent-grid-subsample 2 --latents-origin published-cache-with-local-fill --cache-import-receipt $LIB/evidence/wan_cache_import.json --manifest $LIB/meta/episode_manifest.json --tokens $LIB/motion-tokens-stage2-<tag> --latents $LIB/wan-latents --out $LIB/motion-stage2-1024-<tag> --encoder-run-dir $P/v1-store/external/motionjepa/<run> --raw-dir /data/hongzefu/data_0226 --expected-ckpt-sha256 <sha> --stage2-validation $VALID/<tag> --stage2-source-pin scripts/dataset/wan_stage2/SOURCE_PIN.json` | B4、B6、G0/G2 复制到 `$VALID/<tag>/`、本机 H5 | 表 + meta（status=packed） | AWS 24 min；本机含 16 个 H5 全量 SHA（约 500 GB）估 1～2 h，两个 encoder 的 H5 核对各做一遍 |
| B8 verify | `pack_motion_store.py verify --store $LIB/motion-stage2-1024-<tag> --resume --check-padding` | B7 | status=verified，`VERIFY_MOTION=PASS scanned=23253 mismatches=0` | AWS 22 min |

说明：`load_cache_import` 只按 `raw_dir / <文件名>` 打开 H5 核字节数与 SHA，不与 `input_manifest.raw_dir`（AWS 绝对路径）比对，所以 `--raw-dir /data/hongzefu/data_0226` 可用；provenance 会记下本机 raw_dir，训练侧不读它。`numerics`（torch／cuda／gpu_name 等）由 B4 导出记录并与 B6 的 G3 绑定，两步都在 A6000 上做，自洽；训练侧 `validate_stage2_evidence` 只核证据文件 SHA，不核 GPU 型号。

## 五、要改什么（为什么非改不可）

| 块 | 现状（阻塞点） | 改法 | 文件 |
|---|---|---|---|
| S1 启动契约与 GL 入口 | `check_orig16_stage2_launch.py` 把 encoder 身份写死为 `ENCODER_SHA=27daa832…`／`Contract.encoder_run="stage2-full144-1025-a"`，`store_path` 写死 `motion-stage2-1024`，history 白名单只认 8 个固定名，`GL_COMBOS` 正式只认 (64,32)@60000，`CFG_SCHEDULE` 写死 (5000,5e-5,60000,5e-5)，`--steps` choices=(20,100,60000)，正式 run 名精确等于 `gl-orig100-…-b{batch}-60k`；`run_orig16_stage2_60k_gl.sh` 的 `GL_HC` 只认原 yaml、`STEPS=60000`、`CONFIG` 写死 60k 条目 | 契约加 `--encoder-run`／`--encoder-sha`／`--motion-store` 三个显式参数（默认值 = 原身份，默认行为逐位不变），history 白名单与 `HISTORIES` 加两份新 yaml，`GL_COMBOS` 加 `(64,)@80000`，`CFG_SCHEDULE` 按 config 名分派 `(10000,5e-5,80000,5e-5)`，choices 加 80000，正式 run 名按 K8 两个；入口加 `CONFIG`／`HISTORY`／`ENCODER_RUN` 三个环境变量并转发契约参数，`STEPS` 按 config 取 80000 | `scripts/training/check_orig16_stage2_launch.py`、`scripts/training/run_orig16_stage2_60k_gl.sh`、`tests/training/test_orig16_gl_contract.py`、`tests/training/test_check_orig16_stage2_launch.py` |
| S2 训练配置与 yaml | `config.py` 没有 b64/80k 的 orig16 条目；没有新 encoder 的 yaml | 新增 `TrainConfig(name="mme_vla_suite_orig16_b64_80k", batch_size=64, fsdp_devices=4, num_workers=8, num_train_steps=80_000, lr_schedule=CosineDecaySchedule(warmup_steps=10_000, peak_lr=5e-5, decay_steps=80_000, decay_lr=5e-5)`，其余逐键复制 60k 条目）；复制原 ctx 4×256 yaml 为 `perceptual-framesamp-ctx-32frame-4x4-motion-stage2-4x256-noxyz-orig16.yaml`／`…-nostate-orig16.yaml`，只改 motion 三键 | `src/mme_vla_suite/training/config.py`、`src/mme_vla_suite/models/config/robomme/` 两份新 yaml |
| S3 G3 子集工具 | 没有现成工具能只下 rgb-gtxyz 的索引与指定分片 | 新增 `scripts/dataset/wan_stage2/plan_g3_subset.py`：`plan`（下 build_spec + index 四文件并按 build_spec.index_files 核 SHA、复现 gate_g3 抽样、打印 32 窗→分片→字节）／`fetch`（逐对象下载、核 bucket SHA256SUMS、落 K6 目录）；只读 bucket，不改 reader 与 validate_stage2.py | `scripts/dataset/wan_stage2/plan_g3_subset.py`、`tests/dataset/test_plan_g3_subset.py` |
| 不改 | `framesamp_dataset.py` 形制白名单 (512,16,1) 已放行 context+motion；`pack_motion_store.py`、`validate_stage2.py`、`motion_store.py`、模型前向、loss、精度、`save_state` | — | — |

## 六、验收（查什么 / 怎么查 / 过了说明什么 / 判定行）

| 编号 | 查什么 | 怎么查 | 过了说明什么 | 判定行 |
|---|---|---|---|---|
| V0 | 计划无自相矛盾 | 开工令后一个 sonnet 只读审查，锚定本计划 sha | 可照着实施 | `PLAN_CONSISTENCY_REVIEW=PASS base=<sha> findings=<n> blocking=0` |
| V1 | encoder 字节同一 | B1 下载后 `sha256sum` 等于 K4 两值；`config.yaml` SHA 等于 `583e48c0…`／`869ac778…` | 用的就是 AWS 训出的 ep105 | `ENCODER_FETCH=PASS run=<run> ckpt_sha=<8> config_sha=<8>` ×2 |
| V2 | wan-latents 同一 | B2 逐对象 SHA + 解包后按 `checksums/` 核成员数与 SHA | 与 AWS 建库 latent 字节同一 | `LATENTS_FETCH=PASS objects=13 bytes=26800… sha_mismatch=0`、`LATENTS_UNPACK=PASS members=<n> mismatch=0` |
| V3 | 导出完整 | B4 日志 | 2496 段 23253 行、dropout 诊断 `eval_repeat_max_abs=0` | `STAGE2_EXPORT=PASS segments=2496 rows=23253` ×2 |
| V4 | G3 数值闸门 | B6 | 新 encoder 在 32 个独立 val 窗上两条调用路径逐位相等、EMA 65 叶严格 | `EMA_SOURCE=PASS leaves=65 state_key=model exact=1 live_negative_rejected=1` 与 G3 record `windows=32 max_abs=0` ×2；G3 子集下载 `G3_SUBSET=PASS shards=<n> bytes=<n> sha_mismatch=0` |
| V5 | 表打包与验证 | B7、B8；H5 核对由 `load_cache_import` 内置 | 表行序、字节、padding 与 provenance 全部闭合 | `PACK_PROVENANCE_PRECHECK=PASS mode=vae_compat segments=2496`、`VERIFY_MOTION=PASS scanned=23253 mismatches=0 nonfinite=0 …` ×2 |
| V6a | S1/S2 契约（CPU） | `DRY_RUN=1` 对两份新 yaml × GPUS=4 × GLOBAL_BATCH=64 各一次；拒绝用例：旧 yaml 配新 config、b128、60000、旧 encoder 身份配新表；既有测试 `tests/training/test_check_orig16_stage2_launch.py tests/training/test_orig16_gl_contract.py tests/training/test_stage2_launch.py tests/datastore/` 全过；默认参数（旧身份）行为逐位不变 | 除 batch／步数／日程／encoder 三键外与 AWS 契约逐键相同 | `LAUNCH_CONTRACT_GL=PASS gpus=4 fsdp=4 batch=64 steps=80000 encoder=<run> checks=<n> failed=0` ×2、`CONTRACT_REJECTS=PASS cases=<n>`、`GL_CONTRACT_TESTS=PASS` |
| V6 | 占位内 64 档冒烟 | 占位 RUNNING 后，每个 run 在自己占位里 `o16gl-smoke-` 20 步 global 64，nvidia-smi 500 ms 采样；PASS → 接 V6b、V7 后立即 V9；OOM → 记 FAIL，再各跑 32、16 档 20 步只取峰值，然后 `scancel` 两份占位，停下交用户 | A40 上 64 档是否可行；决定直接起跑还是停下 | `SMOKE_GL=PASS|FAIL gpus=4 batch=64 steps=20 peak_mib=<n> nonfinite=0 [reason=oom]`，另两档 `SMOKE_GL_TIER=… batch=<32|16> peak_mib=<n>` |
| V6b | 保存恢复 | 20 步冒烟的 step 19 用 `check_stage2_checkpoint.py --mode smoke`（父 GPU 0、sidecar GPU 1）；跨卡型握手失败如实记录不改判据（1031 已知） | 保存与恢复链路通 | `SAVE_LOAD=PASS|FAIL …` |
| V7 | 100 步测时 | 同占位 `o16gl-tier-`？不用测时 yaml，用正式 yaml 与 V6 定下的 batch 跑 100 步 `TRAIN_TIMING_STEPS=100`，稳态取第 21–98 步 `host_step_s` 均值；超过 5 分钟按第 17 条诊断 run 留档 | 得到正式步时，核 K7 余量（est_total_h ≤ 0.9 × 占位剩余小时才放行 V9） | `TIMING_GL=PASS|FAIL gpus=4 batch=<b> steps=100 window=21-98 steady_step_s=<x> est_total_h=<x> hold_left_h=<x> margin=<x>` |
| V8 | 占位与 walltime | `sbatch` 回执、`squeue`；`scontrol show partition spgpu` 的 MaxTime | 两份 8 天占位已拿到或在排 | `HOLD_SUBMIT=PASS jobid=<id> gpus=4 time=192:00:00` ×2、`PARTITION_MAXTIME=14-00:00:00` |
| V9 | 正式起跑（V6 PASS 后立即） | Beta commit 已在占位 RUNNING 前打好、`git status --short` 为空、档案 launch.md 第一段写好、`srun` 起训练、日志出现 `TRAIN_CMD=` 与 `Step 100` | 跑的是锚点代码，与冒烟同一 sha | `TRAIN_START=PASS run=<run> head=<sha> jobid=<id>` ×2 |
| V10 | 正式完成与交付可读 | 日志尾 `EXIT_CODE=0`；`--check-final-record`；步集合 {5000,…,75000,79999} 共 16；同占位内 `check_stage2_checkpoint.py --mode delivery --delivery-step 79999` | 80k 跑完、79999 可读回 | `TRAIN_FINAL_RECORD=PASS run=<run> …`、`CKPT_STEPS=PASS … count=16`、`CKPT_DELIVERY_READY=PASS step=79999 …` ×2 |
| V11 | 上传长尾 | 集群 CPU job 逐对象读回 | 公开备份可信 | `HF_VERIFY=PASS bucket=<b> objects=<n> mismatches=0` ×2 |

## 七、步骤

| 阶段 | 内容 | 判据 |
|---|---|---|
| A0 | 开工审查 | V0 |
| A1（开工令后立即） | **先提交两份 4×A40 192 h 占位**（K7，排队与下面并行）；同时派三个写入型子代理在 `$P` 分支上建 worktree 写 S1、S2、S3（§八）；同时 detached tmux 下 encoder（B1）、wan-latents（B2）、建 MJ worktree（B3） | V8、V1、V2 |
| A2 | S3 合入后跑 G3 子集下载（B5）；S1/S2 合入后跑 CPU 契约（V6a） | V4 子集行、V6a |
| A3 | 两个 encoder 并行：B4 导出（GPU 0／GPU 1）→ B6 G3（串行共用 GPU 0）→ B7 pack → B8 verify；表与 `$VALID/<tag>/` 证据落 turbo；两份 yaml 的 `checkpoint_sha256`/`store_path` 与表 provenance 互核（契约 `MOTION_VERIFIED`） | V3、V4、V5 |
| A3b | 表与代码齐备后、占位 RUNNING 前：写两份 `docs/training-doc/<run>/launch.md` 第一段 → **Beta commit**（`git status --short` 为空） | 工作区为空、Beta sha 记入 launch.md |
| A4 | 占位 RUNNING 后：计算节点 venv 核对（1031 V4b 同式）→ 64 档冒烟。**PASS** → SAVE_LOAD → 100 步测时 → 直接进 B1，不问用户；**FAIL** → 32／16 档峰值 → `scancel` 两份占位 → 停下交用户裁决 batch（K2） | V6、V6b、V7 |
| B1（V6 PASS 后立即，不另要开工令） | 两个 run 各在自己占位里 `srun` 起训练（tmux `o16abl-train-noxyz`、`o16abl-train-nostate`）→ 每份日志挂 Monitor；两个占位不同时 RUNNING 时，先到的先起 | V9 |
| B2 | 跑完核 EXIT_CODE、final record、步集合、79999 严格恢复；`scancel` 两份占位；写 result.md、清洗日志、回写结论 commit、handoff；起上传长尾 | V10、V11 |

## 八、子代理分工与合并（简述）

三块写入型子代理（opus，各自独立 worktree，主会话从 `$P` 的分支 `v4-gl-orig100-1031` 建好再派，同一批派出）：S1 改启动契约与 GL 入口，S2 加训练配置条目与两份 yaml，S3 写 G3 子集工具；三者文件不重叠。S1 的契约测试要引用 S2 的新 yaml 文件名与 config 名，所以合入顺序 S2 → S1 → S3（S1 的测试在 S2 合入后的主检出跑）。每块合前一个 sonnet 审固定 BASE/TIP（范围、`sub/` 前缀、默认行为逐位不变、K3 数字），合后主会话跑该块定向短测再 push。B1～B8 建表、占位提交、冒烟测时、正式起跑由 opus 运行代理在 `$P` 或 turbo 副本启动，只起跑、验起跑、交回句柄；主会话负责 Monitor、占位登记与释放、合并审查、建档与交接；运行代理发现代码问题只交证据，主会话另派修复代理走同一审查合并流程。

## 九、交接顺序与状态机登记（第 27 条）

| 项 | 内容 |
|---|---|
| 任务 id | `gl-abl-encoders-phase1`（本计划 A0–A4）、`gl-abl-encoders-train`（B1–B2，`depends_on: [gl-abl-encoders-phase1]`） |
| phase1 交接前必须完成 | V1–V5、V6a、V8 全部 PASS；V6 判定行齐全（64 PASS 一行，或 64 FAIL + 32/16 两行并 `scancel` 回执）；V6b、V7 判定行齐全；出处：`$P/v1-store/logs/o16abl-*.log`、turbo `v1-store/logs/o16abl-*.log`、本计划「实施中实测」小节 |
| 长尾 | phase1 无；train 两条 ckpt 上传（tmux `o16abl-upload-<tag>`，日志 turbo `v1-store/logs/o16abl-upload-<tag>.log`，完成行 `HF_VERIFY=PASS`） |
| 依赖的上游 | 无（两个 encoder 已 `done`，取自公开 bucket，不读 AWS 状态机） |
| 本任务 done 后可开工的下游 | phase1 `done`（含 V6 PASS）→ `gl-abl-encoders-train` 自动 `running`，不另问；train `done` → 两个 run 的官方 800 局评估（另立计划） |
| 是否登记 | 已登记（b019081，K11） |

## 十、用户已裁决清单与剩余待裁决

已裁决（2026-11-01 美东）：① 并行、不累积（K1）；② 先测 64（K2）；③ 80k 步（K3）；④ 不和 AWS 沟通、本地建表（K5）；⑤ 开工即提交带余量的占位、起跑失败不再排队（K7）；⑥ 计划写 md；⑦ 登记状态机（K11）；⑧ 64 档通过直接正式训练、不另要第二阶段开工令；64 档不过则测完三档、取消占位再问（K2、K10 修订）。

待裁决：⑨ 64 档 OOM 后的 batch 与步数（只在 A4 FAIL 时问）。

# 第二部分（技术细节，供 agent 追踪）

## 〇 前置声明与红线

- R1 不改训练语义：不改模型前向、loss、精度、数据格式、既有 yaml、`config.py` 既有条目、`framesamp_dataset.py`、`pack_motion_store.py`、`validate_stage2.py`、`motion_store.py`、`save_state`；新 config 条目除 K3 四项（batch、fsdp/workers、步数、日程）外逐键等于 60k 条目。
- R2 不续跑：不启用 `--resume`；占位一次申请 192 h（K7）。64 档 OOM 时按 K2 `scancel` 两份占位（逐 JobID），正式训练只在 64 档 PASS 的占位内起。
- R3 不下载原始 H5（本机已有原件 `/data/hongzefu/data_0226`，只读）；不写 `/scratch/hongze` 进任何新默认值；不在本机新建 `/scratch`。
- R4 G3 子集目录 `/data/hongzefu/dataset-rgb-gtxyz-g3-subset/` 只放下载物，验证完成后保留到两个 run 评估结束，删前问用户。
- R5 占位 job 两份 4 GPU 规格超默认，K7 已放行；同批两份是「同批多 job 并行」，S3 起跑时记进 `greatlakes.md` 项目节；`scancel` 只按本计划登记的两个 JobID 逐个做。
- R6 本机 GPU 只用于建表（B4、B6），不在本机跑 VLA 冒烟（FSDP2 不能代表 FSDP4）。
- R7 全部 tmux 前缀 `o16abl-`，清单写进 launch.md 与 current.json；清理只按清单。

## 一、逐文件改动清单

| 块 | 文件 | 锚点 | 改什么 | 关闭态（默认） | 开启态 |
|---|---|---|---|---|---|
| S1 | `scripts/training/check_orig16_stage2_launch.py` | 顶层 `ENCODER_SHA`、`Contract`、`check_history`、`check_dataset`、`check_config`、`GL_COMBOS`、`gl_formal_run`、`HISTORIES`、`main` argparse | 加 `ENCODERS = {run: (ckpt_sha, config_sha, store_dir, yaml)}` 表含旧身份与两新身份；`Contract` 字段改为按 `--encoder-run` 取值；`store_path` 由表给；`CFG_SCHEDULE`/`CFG_STEPS` 按 `--config-name` 分派 60k 或 80k 口径；`--steps` choices 加 80000；`GL_COMBOS` 加 formal `(64,)@80000`；正式 run 名表加 K8 两项 | 不传新参数时所有值等于原常量，输出逐位相同 | 新身份被完整校验 |
| S1 | `scripts/training/run_orig16_stage2_60k_gl.sh` | `CONFIG=`、`GL_HC`、`STEPS=60000`、formal 分支、契约调用 | 读 `CONFIG`（默认原值）、`ENCODER_RUN`（默认原值）环境变量；`HISTORY` 允许集合 = 原 yaml + 两新 yaml；`STEPS` 按 `CONFIG` 取 60000/80000；把 `--config-name`、`--encoder-run` 转给契约 | 不设即原行为 | — |
| S1 | `tests/training/test_orig16_gl_contract.py`、`tests/training/test_check_orig16_stage2_launch.py` | 夹具 | 加两新身份的 PASS 用例与 V6a 四个拒绝用例 | — | — |
| S2 | `src/mme_vla_suite/training/config.py` | `mme_vla_suite_orig16_b128_60k` 条目之后 | 新增 `mme_vla_suite_orig16_b64_80k`（K3 数值），注释写明「只改 batch/fsdp/workers/步数/日程五项」 | 既有条目不动 | — |
| S2 | `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-ctx-32frame-4x4-motion-stage2-4x256-{noxyz,nostate}-orig16.yaml` | 新文件 | 复制原 yaml，`motion.store_path: v1-store/datasets/16task-orig100-1600ep-9e109419/motion-stage2-1024-<tag>`、`motion.source_run: <run>/checkpoint_epoch_105.pt#model`、`motion.checkpoint_sha256: <K4>`；其余逐键相同（`diff` 只许三行） | — | — |
| S3 | `scripts/dataset/wan_stage2/plan_g3_subset.py` | 新文件 | `plan`：匿名下 `control/build_spec.json`、`index/{windows.parquet,source_valid_episodes.json,state_episodes.json,episodes.parquet}`，按 `build_spec.index_files` 核字节与 SHA；用与 `validate_stage2.py::gate_g3` 第 315–318 行逐字相同的表达式（`sorted` 候选 + `np.random.default_rng(0).choice(…, 32, replace=False)`）得 32 窗；`row_start // 4096` 得分片号；输出 `g3-subset-plan.json`（窗、分片、episode bits、字节和）。`fetch`：逐对象下载分片 7 文件与 bits，核 bucket `SHA256SUMS` 中的 SHA（rgb/gtxyz 以 SHA256SUMS 为准，比 reader 更严）；末行 `G3_SUBSET=PASS shards=<n> bytes=<n> sha_mismatch=0` | — | — |
| S3 | `tests/dataset/test_plan_g3_subset.py` | 新文件 | 用伪造 build_spec/windows 表测抽样复现与分片映射；不联网 | — | — |

## 二、子代理分配表

| 子任务 | 模型／隔离 | 目标 | 可写文件集合 | 禁触路径 | 接口契约与依赖 | 合并顺序 | 验收命令与判定行（在哪里、什么环境） | 资源 |
|---|---|---|---|---|---|---|---|---|
| S2 | opus，worktree（自 `$P` 分支 HEAD） | 新 TrainConfig + 两份 yaml | `src/mme_vla_suite/training/config.py`（只增条目）、两份新 yaml | 既有 yaml、既有条目、其余全部 | yaml 文件名与 config 名固定如上表，S1 引用 | 1 | `UV_PROJECT_ENVIRONMENT=$P/.venv PYTHONPATH=<worktree>/src uv run --no-sync python -c "from mme_vla_suite.training import config as c; cfg=c.get_config('mme_vla_suite_orig16_b64_80k'); assert (cfg.batch_size,cfg.fsdp_devices,cfg.num_workers,cfg.num_train_steps)==(64,4,8,80000) and (cfg.lr_schedule.warmup_steps,cfg.lr_schedule.decay_steps)==(10000,80000)"` → `CFG_B64_80K=PASS`；`diff` 新旧 yaml 恰 3 行 → `YAML_DIFF=PASS lines=3` ×2 | CPU |
| S1 | opus，worktree | 契约与入口参数化 | `scripts/training/check_orig16_stage2_launch.py`、`scripts/training/run_orig16_stage2_60k_gl.sh`、两份测试 | config.py、yaml、模型、`pack_motion_store.py`、`motion_store.py`、`train.py` | 依赖 S2 的文件名；默认参数行为逐位不变（用 S4 的既有 DRY_RUN 日志对拍） | 2 | 同环境 `uv run --no-sync python -m pytest tests/training/test_check_orig16_stage2_launch.py tests/training/test_orig16_gl_contract.py tests/training/test_stage2_launch.py tests/datastore -q` → `GL_CONTRACT_TESTS=PASS`；合后主会话在 `$P` 跑 V6a 的 DRY_RUN 与拒绝用例 | CPU |
| S3 | opus，worktree | G3 子集工具 | `scripts/dataset/wan_stage2/plan_g3_subset.py`、`tests/dataset/test_plan_g3_subset.py` | `validate_stage2.py`、`rgb_gtxyz_reader.py`（MJ 侧）、其余 | 抽样表达式与 `gate_g3` 逐字相同（在文件头注明来源行）；只读 bucket | 3 | `uv run --no-sync python -m pytest tests/dataset/test_plan_g3_subset.py -q` → `G3_SUBSET_TESTS=PASS`；合后主会话真跑 `plan` → `G3_SUBSET_PLAN=PASS windows=32 shards=<n> bytes=<n>` | CPU，网络 |
| R1 | opus，运行型，`$P` | B1、B2、B3 | turbo `$LIB/wan-latents/`、`$P` 与 turbo 的 `v1-store/external/motionjepa/<run>/`、`/data/hongzefu/MotionJEPA-b5d23a5`（worktree）、日志 `$P/v1-store/logs/o16abl-fetch-*.log` | 代码 | 无 | — | V1、V2；tmux `o16abl-fetch-enc`、`o16abl-fetch-latents` | 网络 |
| R2 | opus，运行型，`$P` | B5 | `/data/hongzefu/dataset-rgb-gtxyz-g3-subset/`、日志 `o16abl-g3fetch.log` | 代码 | 依赖 S3 合入 | — | `G3_SUBSET=PASS …`；tmux `o16abl-g3fetch` | 网络，≤80 GB 本机盘 |
| R3a／R3b | opus，运行型 ×2，`$P` | 各一个 encoder 的 B4→B6→B7→B8 | `$LIB/motion-tokens-stage2-<tag>`、`$LIB/motion-stage2-1024-<tag>`、`$P/v1-store/validation/orig16-stage2-<tag>/`（G0/G2 复制 + G3）、日志 `o16abl-build-<tag>.log` | 代码；另一个 tag 的目录 | 依赖 R1、R2；B6 两者串行共用 GPU 0（先 noxyz 后 nostate） | — | V3、V4、V5；tmux `o16abl-build-noxyz`、`o16abl-build-nostate` | GPU 0／GPU 1 各一张（B4），CPU 48 线程（B7/B8） |
| R4 | opus，运行型，turbo 副本 | 两份占位提交 | turbo `v1-store/logs/o16abl-hold-*.log` | 代码 | 开工令 | — | V8；JobID 立即交回 | 2 × (4×A40, 192 h, `--cpus-per-task=16 --mem=192G`，规格沿用 `scripts/training/gl/hold_o16.sbatch`) |
| R5a／R5b | opus，运行型 ×2，turbo 副本 | 占位内 V4b、V6、V6b、V7 | turbo `train-runs/o16gl-smoke-*`、`train-runs/o16gl-tier-*`、`logs/o16abl-smoke-*.log` | 代码 | 依赖 R3 表落地、S1/S2 合入、占位 RUNNING | — | V6、V6b、V7；tmux `o16abl-smoke-<tag>` | 各自占位 |
| R6a／R6b | opus，运行型 ×2，turbo 副本 | 正式起跑（V6 PASS 后立即） | turbo `train-runs/<run>`、`logs/<run>.log` | 代码 | V6 PASS、V6b/V7 判定行、Beta 已打、建档第一段已写 | — | V9；tmux `o16abl-train-<tag>` | 各自占位 |
| 主会话 | — | 占位登记／释放、Monitor、合并审查、G0/G2 证据复制与 SHA 核、建档、handoff、batch 裁决转交 | — | — | — | — | V0、V6a、V8、V10、V11 | — |

共享文件归属：`check_orig16_stage2_launch.py` 与 GL 入口只属 S1；`config.py` 只属 S2；两份 yaml 只属 S2；`$LIB` 下按 `<tag>` 子目录分属 R3a／R3b；turbo `v1-store/logs/` 按文件名前缀分属。

## 三、闸门总表

顺序：V0 → (V8 提交占位 ∥ V1、V2 ∥ S1/S2/S3 写与审) → S2 合 → S1 合 → S3 合 → V6a → (V4 子集下载) → V3 ×2 → V4 ×2 → V5 ×2 → A3b Beta → 占位 RUNNING → V4b（计算节点 venv，同 1031）→ V6 → [PASS] V6b → V7 → V9 → V10 → V11 ／ [FAIL] 32、16 档峰值 → `scancel` ×2 → 停下交用户。任一 FAIL 只记证据不改判据；V6 的 64 档 FAIL 按 K2 不算计划失败；V7 余量不足 1.2 倍、V4 任一 encoder G3 不过（说明 encoder 与锁定源码不同源）、V5 H5 SHA 不符（本机原件与 AWS 不同源）三种情形停下交用户。

## 四、runbook（开工后模板，变量按 §一 展开）

```bash
# A1-占位（turbo 副本，每个 tag 一份；规格沿用 scripts/training/gl/hold_o16.sbatch，只改 --time 与 --job-name）
sbatch --account=chaijy2 --partition=spgpu --nodes=1 --ntasks-per-node=1 --gres=gpu:4 \
  --cpus-per-task=16 --mem=192G --time=192:00:00 --job-name=o16abl-hold-<tag> --wrap='sleep infinity'
# A1-数据（$P，detached tmux，日志三件套）
tmux new-session -d -s o16abl-fetch-enc "set -o pipefail; PYTHONUNBUFFERED=1 bash v1-store/logs/o16abl-fetch-enc.sh 2>&1 | tee v1-store/logs/o16abl-fetch-enc.log; echo \"EXIT_CODE=\$?\" >> v1-store/logs/o16abl-fetch-enc.log"
# A3-建表（$P，每 tag 一个 tmux，驱动脚本按 §四 B4→B8 串行；GPU 由脚本参数给）
tmux new-session -d -s o16abl-build-noxyz "bash v1-store/logs/o16abl-build.sh noxyz 0"
tmux new-session -d -s o16abl-build-nostate "bash v1-store/logs/o16abl-build.sh nostate 1"
# A4-占位内冒烟（turbo 副本，占位 RUNNING 后）
srun --jobid=<hold> --overlap --exact --ntasks=1 --cpus-per-task=16 --gpu_cmode=shared \
  env RUN_NAME=o16gl-smoke-<tag>-b64 CONFIG=mme_vla_suite_orig16_b64_80k ENCODER_RUN=<run> \
      HISTORY=perceptual-framesamp-ctx-32frame-4x4-motion-stage2-4x256-<tag>-orig16.yaml \
      GPUS=4 GLOBAL_BATCH=64 MAX_STEPS=20 TRAIN_HEAD=<sha> FINAL_RECORD_DIR=<...>/final-record \
  bash scripts/training/run_orig16_stage2_60k_gl.sh
# B1-正式（V6 PASS 后立即，同占位，不另问）
srun --jobid=<hold> --overlap --exact --ntasks=1 --cpus-per-task=16 --gpu_cmode=shared \
  env RUN_NAME=gl-orig100-ctx32x4x4-motion-s2split4x256-<tag>-b64-80k CONFIG=mme_vla_suite_orig16_b64_80k \
      ENCODER_RUN=<run> HISTORY=<新 yaml> GPUS=4 GLOBAL_BATCH=64 TRAIN_HEAD=<Beta sha> FINAL_RECORD_DIR=<...>/final-record \
  bash scripts/training/run_orig16_stage2_60k_gl.sh
```

每份日志一个 Monitor：`tail -n +1 -F <log> | stdbuf -oL tr '\r' '\n' | grep --line-buffered -E "EXIT_CODE=|Step [0-9]+:|RESOURCE_EXHAUSTED|Traceback|nan|NaN|Killed|=PASS|=FAIL"`。

## 五、风险登记

| 风险 | 处置 |
|---|---|
| 64 档 OOM（概率高，§三） | K2：记 FAIL，补 32／16 峰值，停下交用户；占位不释放，等裁决后在同占位继续 |
| spgpu 常年满卡，两份 4 卡 192 h 占位排队时间不可控 | 开工即提交（K7），排队期间完成本机建表与代码；排到一份先跑该 run 的 A4 |
| 192 h 对 32／16 档可能不够 | 若用户改选小 batch，步数与 walltime 另算，占位不够则由用户决定是否重交 |
| G3 子集抽样复现偏差（`windows.parquet` 过滤条件与 `gate_g3` 不一致） | S3 用逐字相同表达式并在测试里对拍；真跑 B6 时 reader 若要的分片不在本地会直接报错，按 FAIL 记并补下 |
| 本机 H5 原件与 AWS 不同源（只核过字节数） | B7 的 `load_cache_import` 逐文件 SHA 核对，不符即 FAIL 停下 |
| `numerics.gpu_name` 绑定：导出与 G3 必须同一卡型 | B4、B6 都在 A6000 上做；B6 两 encoder 串行用 GPU 0 |
| SAVE_LOAD 跨卡型握手（A100 产 reference 对 A40） | 1031 已知，V6b 如实记录不改判据 |
| NFS 读 wan-latents 与 H5 的 IO | 26.8 GB latent 在 turbo，本机 NFS 实测约 130 MB/s，B4 读取 3～4 min；H5 在本机 NVMe |

## 六、盲区诚实清单

- 4×A40 FSDP4 的任何显存与步时数字都未实测，§三 全部是推断；V6、V7 替换。
- G3 子集的分片数（1～32）与字节（2.5～78.5 GB）在 S3 `plan` 跑完前不知道。
- `stage2_memory_export.py` 在 A6000 上的耗时、`--batch 1` 是否可放大未测；按 AWS 8 min 外推。
- 两个 bucket 的对象列表只从 upload-SHA256SUMS 读到，`hf buckets list` 两次超时未直接看远端树。
- 评估侧（800 局）不在本计划内。

## 七、留档与 commit 纪律

- phase1 的 V1–V8 齐全后在本计划末追加「实施中实测」小节（V1–V8 判定行原文、三档峰值表、`PLAN_CONSISTENCY_REVIEW` 行）；不改写上文口径。
- 两个 run 各建 `docs/training-doc/<run>/{launch.md,result.md,records/}`，两段式写入；B4～B8 建表按第 13 条在 `docs/dataset-build-doc/16task-orig100-1600ep-9e109419-motion-<tag>/` 留压缩版档案（新表不是新库，只记 motion 表增量）。
- 代码 commit `commitV11.74+`，合并提交 body 按第 11 条六项；起跑前 `commitV11.<n>Beta`；状态变化只碰 `docs/handoff/current.json` 的 `handoff:` 小 commit；每次 commit 后裸 `git push`。
