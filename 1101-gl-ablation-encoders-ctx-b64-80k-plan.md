# GreatLakes 4×A40 并行训练两个消融 encoder（去 depth／去 action）的 context VLA 计划（b64、80k）

> **修订记录**：v1（29841c1）首版；v2（29d9dea、fdbebc6、ea231ca）按用户裁决改为一次开工令覆盖全程、占位 176 h、余量不足 10% 自动重申请；**v3 按 Astra 对抗审查（审计锚点 125feda，`PLAN_ADVERSARIAL_REVIEW=FAIL scope=static`）逐条回代码核实后修订**：10 条阻塞 9 条成立或部分成立（第 2 条「A100 守卫」成立；第 4 条「逐位比较」部分成立），处置见第二部分「八、对抗审查处置」；用户 2026-11-02 裁决「1 2 同意 3 不要」（G3 沿用 AWS 缓存先例、放宽 A100 守卫、不加旧 encoder 对照 run）；**v4：执行宿主改为 Codex**（用户原话「这个计划需要改成用 Codex 来做，你把所有的 agent 都映射成 Codex 的要求」），全部子代理按正本 `docs/codex-aspen-gpt56-roles.md` §二与 `AGENTS.md` 第 26 条 Codex 专节映射，见第一部分 §八与第二部分「〇-C Codex 执行专章」。**v5按开工审查与用户 2026-11-02 裁决同步两项时序**：两个 run 全部通过 64 档及第一阶段后再正式起跑；Beta 早于首次冒烟或训练工作负载，占位可提前 RUNNING 等待。用户先说「同意修订 继续做」，随后明确「不需要再确认，你可以直接开始了。」；本版修订后直接完成独立开工审查并实施，不再请求开工确认。 **v6（当前执行口径）**：用户原话「不用再测速了，可以正式训练启动。你再看一下这个，你先看一下这个冗余的时间还够不够，不够的话要重新启动任务。」取消独立100步测时。V7记NOT_REQUIRED，按原预算上界核实际余量；两个64/20均PASS、余量约19%后，已于美东11月2日10:23:07在两原占位正式起跑。
>
> **权威性与授权边界**：本计划只规划、不实施。下载、本机建表、提交占位 job、冒烟、正式起跑都是开工步骤，用户已明确授权本次两项修订后直接实施：「不需要再确认，你可以直接开始了。」无需再请求开工确认；独立开工审查仍执行（`AGENTS.md` 第 2 条）。**一次开工令覆盖全程**（K10）：两个 run 的 64 档都通过并完成第一阶段后，直接正式起跑，不再另问；训练 batch 裁决停点是 64 档 OOM（K2），身份与原始数据 SHA 失败停点仍按第二部分 §三。
>
> **代码锚点**：训练代码固定为分支 `origin/v4-gl-orig100-1031` 的 `f1dbe7c9c85ffbf23aaba2bc73321f696bdf4b69`（commitV11.73，含 1031 计划已合入的 S1 卡数可变入口、S2 占位运行器、S3 训练库拉取工具、S4 小 batch 冒烟放行）。本计划新增代码在该分支继续（commitV11.74 起），起跑前打 `commitV11.<n>Beta`。MotionJEPA 导出入口来源锁定为 `4a16ab42a0ef5d70b9658e296b4944180842f248` 的 `scripts/stage2_memory_export.py`，原文件 SHA256 为 `305fb59a497010ca7ef540092a7a2b3a8b6bd919ba4680c990d15b6eff67c53e`；按实施中 S7 以单文件 vendoring 放到 `scripts/dataset/wan_stage2/export_stage2_memory.py`，只增加 `--lineage-build-spec` 显式来源落点。实际导入的 MotionJEPA 包提交为 `b5d23a5fab55be16e580c5b77a8f04d68b3170c2`，其 export 模块 SHA `c9e260ea…`、encoder 模块 SHA `c9b967f3…` 与 4a 对应文件逐字节相同，保持仓库 `SOURCE_PIN.json`。入口来源、实际包提交和本地 Beta 分别如实记录。
>
> **工作副本**：本机实施副本 `$P=/data/hongzefu/robomme_policy_learning_MotionJEPA-gl`（环境 A，sled-aspen，2×RTX A6000 48 GB，自带 `.venv` 与 `v1-store/venvs/wan-stage2`，torch 2.9.0+cu128）；主检出 `/data/hongzefu/robomme_policy_learning_MotionJEPA`（评估分支 `v4-eval-official-1026`）只放本计划与 `docs/handoff/current.json`；集群副本 `<GL_REPO>=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA`（计算节点唯一可见路径，现为 detached HEAD `f1dbe7c`、干净，NFS 上已有主 venv 与 wan-stage2 子环境）。训练库 `$LIB=<GL_REPO>/v1-store/datasets/16task-orig100-1600ep-9e109419`（本机 `$P/v1-store/datasets/` 下是指向它的只读 symlink）。三处的读取关系：计划与状态机只在主检出评估分支；代码与 Beta 只在 `$P` 的 `v4-gl-orig100-1031`；集群执行副本在 Beta 后 `git fetch` 并 `checkout <Beta sha>`（detached），运行器核 HEAD 与 clean。
>
> **外部依赖锚点**：训练库公开 bucket `HongzeFu/robomme-16task-orig100-1600ep-trainset-20261028-v1`（K7 子集已在 turbo；本计划再取 `wan-latents_tars/` 13 个 tar 共 26.80 GB）；两个 encoder 公开 bucket `HongzeFu/motionjepa-aws-stage2-noxyz-ep105-v1`、`HongzeFu/motionjepa-aws-stage2-nostate-ep105-v1`（各 26 对象，`HF_VERIFY=PASS public_anon_read=1`，MotionJEPA commitV8.136 `bb9f9ab`）；G3 子集来自 `HongzeFu/robomme-16task-rgb-gtxyz-stride2-16views-fp16-v1`（11759 文件、2.996 TB，只取 `control/build_spec.json`、`index/` 四文件与 32 窗所在分片）；原始 H5 本机原件 `/data/hongzefu/data_0226/`（16 文件字节数与 `$LIB/meta/input_manifest.json` 逐一相同，SHA 由打包脚本现场核）。
>
> **commit 体例**：计划与留档 `docs:`；代码 `commitV11.<n>: `；子代理分支 `sub/<编号>: `；起跑前 `commitV11.<n>Beta`。本计划按用户原话「写md就可以」写成 Markdown，是对 `AGENTS.md` 第 2 条「进仓库 git 的计划一律 HTML」的本次用户显式豁免。

# 第一部分（给人看）

## 一、要做什么与已定口径

一句话方案：把 AWS 训完的两个阶段 III 消融 encoder（去 depth、去 action，各取 ep105 EMA）下到本机，对 orig100 训练库各重算一张 stage2 motion 表（95 MB）并经缓存路线的 G3 与 pack/verify 闭合证据；GreatLakes 上开工即为每个 run 提交一份 4×A40、176 h 的占位 job，两个占位各做 64 档 20 步冒烟，全部通过并完成第一阶段后就直接起正式 80k 训练（global 64、warmup 10000、decay 80000、lr 5e-5），两个 run 并行。

| 编号 | 已定口径 | 依据 |
|---|---|---|
| K1 | 两个 run 并行，各一份 4×A40 占位；不做梯度累积 | 用户 2026-11-01 原话「同意并行。不做任何的梯度累积。」 |
| K2 | global batch 先测 64（每卡 16）。**每个 run 的 64 档 PASS → 记录该 run 的 SAVE_LOAD 原始判定；独立100步测时按用户最新指令取消，核实际余量后继续。两个 run 的 64 档都 PASS 且第一阶段完成后，直接正式起跑，不再等待用户确认**；**64 档 OOM → 同占位内补测 32、16 两档各 20 步只取显存峰值（V7现已取消），然后逐 JobID `scancel` 两份占位，停下交用户裁决**，不自动降档、不自动训练 | 用户原话「在 Great Lakes 上先测 batch size，总共六十四，如果不够的话就问用户」「先跑这个六十四的档位」；2026-11-01 修订原话「第二阶段的开工令不需要再问用户，如果六十四这一档不通过的话，就停下来，然后取消你的占位 job，然后问用户。但是在问用户之前，需要把六十四、三十二、十六这三档都测了。如果六十四这档通过了，就直接开始正式训练，就不要在中间等了。」；§三；用户 2026-11-02「同意修订 继续做」「不需要再确认，你可以直接开始了。」批准两个 run 的共同起跑条件 |
| K3 | 步数 80 000；warmup 10 000；decay_steps 80 000；peak_lr = decay_lr = 5e-5；EMA 0.999；seed 42；save_interval = keep_period = 5000（16 份 ckpt 含 79999）；fsdp_devices 4、num_workers 8 | 用户原话「步数从一百二十 K 降到八十 K step」；lr 不缩放沿用 `config.py` 既有注释口径；warmup 按样本量等价 5000×128 = 10000×64，与官方 `mme_vla_suite`（b64、warmup 10000）同值 |
| K4 | encoder 身份与表名：去 depth `aws-stage2-noxyz-1101-a/checkpoint_epoch_105.pt` SHA256 `6acc92af239e9b9a4c58ced4c5123bc25183188ec518b10734e917879240d00d`（config.yaml SHA `583e48c0…`）→ 表 `$LIB/motion-stage2-1024-noxyz`、证据目录 `$LIB/evidence-noxyz/`；去 action `aws-stage2-nostate-1101-a/checkpoint_epoch_105.pt` SHA256 `be7cdec7d96c889193f27cd925977c7ff47423d4321d1fd7446799d38495e3d4`（config.yaml SHA `869ac778…`）→ `$LIB/motion-stage2-1024-nostate`、`$LIB/evidence-nostate/`。两份新 yaml 名 `perceptual-framesamp-ctx-32frame-4x4-noxyz-motion-stage2-4x256-orig16.yaml`、`…-4x4-nostate-motion-…-orig16.yaml`（标签放中段，`history_key` 的 `parts[2]=ctx`、`parts[-2]=4x256`、结尾 `-orig16.yaml` 三处解析都不变） | MotionJEPA `bb9f9ab` 两份 `records/upload-SHA256SUMS.txt`；审查第 5 条命名解析核实 |
| K5 | motion 表全部在本机重算，不与 AWS 沟通：固定 4a 入口的本地副本（S7，显式 `--lineage-build-spec`） 单卡导出 → 缓存路线 G3 → `pack_motion_store.py pack/verify`；G0、G2、缓存收据复用库内 `evidence/`（缓存路线下三者都不绑定 encoder；G2 带收据分支不核 ckpt） | 用户原话「不和 AWS 沟通，你在本地进行。」；§四 |
| K6 | G3 子集：只下 `control/build_spec.json` 与 `index/` 四文件（约 3 MB），本地复现 `gate_g3` 的 `default_rng(0).choice(val 候选, 32)` 得到确切 32 窗，再只下所在分片（每满片 2.45 GB，上限 32 片约 78.5 GB）与对应 `masks/source_valid/<task>/epNNN.bits`；落点 `$P/v1-store/external/rgb-gtxyz-g3-subset/`（在 `<STORE_ROOT>` 内，不落仓库外） | reader `_init_v2` 只校验 4 个索引文件、`_load_shard_full` 按 `row_start // 4096` 开单片；审查「G3 目录落在仓库外」项 |
| K6a | **G3 验收口径沿用 AWS 缓存先例**：带 `--cache-import-receipt`，不做两路径逐位比较，判定为 `EMA_SOURCE=PASS leaves=65 state_key=model exact=1 live_negative_rejected=1`、record `metrics` 固定为 `windows=32 seed=0 geometry_all_true=1 shape_mismatches=0 nonfinite=0`（无 `max_abs`）、`output_numerics=NOT_REQUIRED`，`EXPORT_PARITY=NOT_REQUIRED reason=user_decision_same_input`；训练侧 `validate_stage2_cache_evidence` 核的正是这组字段 | 用户 2026-11-02「1 同意」；AWS 原 run `evidence/G3.json` 即此形制 |
| K6b | **放宽 A100 守卫**：`validate_stage2.py::setup_runtime` 的「数值闸门要求同型号 A100」改为接受 `A100` 或 `RTX A6000`，`numerics.gpu_name` 照实记录；导出（B4）与 G3（B6）必须在同一卡型（本机 A6000）完成，训练侧只比对 G3 与表 provenance 的 `numerics` 一致 | 用户 2026-11-02「2 同意」；GL 只有 A40、本机只有 A6000，没有 A100 可用 |
| K7 | 占位方式：**开工即为每个 run 提交一份 4×A40、walltime 176 h（7 天 8 小时）的占位 job**（`--wrap='sleep infinity'`，规格沿用 `scripts/training/gl/hold_o16.sbatch`：`--cpus-per-task=16 --mem=128G --gpu_cmode=shared`），排队与本机建表并行，占位可提前 RUNNING 等待；冒烟、正式训练全部用 `srun --overlap` 经占位运行器 `run_in_hold_o16.sh` 塞进同一份占位；176 h =（上限估算 144 h + 冒烟测时约 2 h）× 1.2，余量约 20%；取消V7后，起跑前以原计划6.5s/步上界估80k为144.44h，并用**该 JobID 当时的实际剩余时间**算余量；正式运行后以其实际同步日志复核：**不足 10%（`est_total_h > 0.9 × hold_left_h`）时不问用户，立即按 `est_total_h × 1.2` 另交一份同规格新占位，新占位 RUNNING 且两个 run 已满足 K2 的共同起跑条件后在其中起正式训练，旧占位随即 `scancel`**；≥ 10% 且共同起跑条件满足则直接在原占位起；先到者等待期间消耗的时间须计入正式起跑前重新读取的 `hold_left_h`。不续跑、不实现 `--resume` | 用户原话「直接申请一个有安全余量的 Great Lakes 的 job……以占位 job 的形式启动」「余量保持在这个百分之二十左右就可以了，因为它这个任务太长了」「如果实测步数的余量小于百分之十，你就要重新再申请 job，不需要再问用户」；`spgpu MaxTime=14-00:00:00` |
| K8 | run 名 `gl-orig100-ctx32x4x4-noxyz-motion-s2split4x256-b64-80k`、`gl-orig100-ctx32x4x4-nostate-motion-s2split4x256-b64-80k`；**失败重起用 `-attemptNN`（NN ≥ 02）后缀**，经 `ATTEMPT_MANIFEST` 清单登记，新 `FINAL_RECORD_DIR`，失败产物原样保留；冒烟前缀 `o16gl-smoke-`，tmux 前缀 `o16abl-` | 沿用 1031 命名法；审查第 8 条 |
| K9 | ckpt 存 turbo `<GL_REPO>/v1-store/train-runs/mme_vla_suite_orig16_b64_80k/<run>/`，每 5000 步一份共 16 份（约 12 GB × 16 × 2 ≈ 390 GB，turbo 余量 12 T）；交付步 79999；跑完各上传公开 bucket `HongzeFu/robomme-vla-gl-orig100-ctx-s2split-<noxyz|nostate>-b64-80k-v1` 作长尾，读回校验在集群 CPU job 做 | 同 1031 K8；审查第 7 条 |
| K10 | **一次开工令覆盖全程**：本机建表 + 代码 + 两份占位提交 + Beta + 占位内 64 档冒烟 + （PASS 时）正式 80k 起跑 + 收尾上传。Beta commit 在代码合入、契约 PASS、两张表落地之后、占位内首次运行冒烟或训练工作负载之前打好；占位可提前 RUNNING 等待，冒烟与正式用同一 Beta sha | 用户 2026-11-01 修订原话（见 K2） |
| K11 | 任务已登记 `docs/handoff/current.json`（b019081）：`gl-abl-encoders-phase1`、`gl-abl-encoders-train`（`depends_on` 前者）；phase1 `done`（含两个 run 的 V6 64 档 PASS）即自动进入 train，不另问 | 用户原话「登记」；第 27 条 |
| K13 | **执行宿主 = 本机（sled-aspen）Codex**，主代理 `gpt-6.1-sol/high`；探索→`explorer`（`gpt-6-luna/high`，read-only）、写代码→`worker`（`gpt-6.1-sol/high`，主代理预建独立 worktree）、运行型→`tester`（`gpt-6.1-sol/high`，只起跑不改码）、审查与对抗验证→`reviewer`（`gpt-6.1-sol/high`，read-only）；每次派发显式写 `model` 与 `reasoning_effort="high"`，禁 GPT-5.6；持久代理同职责续用 `followup_task`；无 hook，开工手动 `repo_guard.py start/fix`、每次 commit 后手动 `git status -sb` | 用户原话「用 Codex 来做……映射成 Codex 的要求」；正本 `docs/codex-aspen-gpt56-roles.md` §二；第 26 条 |
| K12 | **不加旧 encoder 同制度对照 run**。两个消融互比成立；与旧全量 encoder（B128×60k）成绩的差异只作探索性比较，结论里注明制度不同 | 用户 2026-11-02「3 不要」；审查「唯一实验变量」项 |

## 二、被复现的实验与本次的差别

| 项 | AWS 原 run（`orig100-ctx32x4x4-motion-s2split4x256-b128-60k`，2026-10-29） | 本次两个 run | 差别 |
|---|---|---|---|
| encoder | `stage2-full144-1025-a` ep105，SHA `27daa832…` | K4 两个消融 ep105 | 实验变量 |
| history yaml | `…-ctx-32frame-4x4-motion-stage2-4x256-orig16.yaml`：budget 512、token_per_image 16、32 帧、context、memory_token_dim 2048、motion 4 token×256、budget 384、stride 32、window 33 | 两份新 yaml，只改 `motion.store_path`、`motion.source_run`、`motion.checkpoint_sha256` 三键 | 记忆位 512 + 384 = 896 不变 |
| 训练配置 | `mme_vla_suite_orig16_b128_60k`：B128、FSDP8、workers16、60k、warmup 5000、lr 5e-5、decay 60000 | 新条目 `mme_vla_suite_orig16_b64_80k`：B64、FSDP4、workers8、80k、warmup 10000、lr 5e-5、decay 80000 | 总样本 5.12M = 原 7.68M 的 2/3（与官方 80k×b64 相同）；每步样本减半、更新次数增 1/3；EMA 0.999 的样本尺度随之变化 |
| 硬件 | 8×A100-80GB，2.43 s/步，40 h 34 min | 4×A40-48GB，估 5.5～6.5 s/步，5～6 天 | §三 |
| 数据 | 库 `16task-orig100-1600ep-9e109419`，1600 集、476857 执行样本、motion 表 23253 行 | 同库同帧表，motion 表换两张新表（23253 行） | — |
| 对照 | 原 run 59999 官方 800 局 38.88%；三 seed ctx split 38.42±1.25 | 训完后另立评估计划 | **制度不同（batch、步数、硬件、EMA 样本尺度），与旧 run 的差异只作探索性比较（K12）** |

## 三、显存与耗时事实（决定 K2、K7）

| 卡型 / 分片 | 每卡样本 | 结果 | 出处 |
|---|---|---|---|
| 本机 2×A6000 48G，FSDP2 | 2（global 4） | 跑通，`memory.used` 峰值 39 020 MiB | `$P/v1-store/logs/o16gl-smoke-b4-11011852.mem.csv` |
| 本机 2×A6000 48G，FSDP2 | 4／8／16 | 全部 `RESOURCE_EXHAUSTED`（分别多要 7.15／11.47／14.10 GiB） | `o16gl-smoke-b{8,16,32}-*.log` |
| AWS 8×A100 80G，FSDP8 | 16 | 峰值 42 045 MiB | `docs/training-doc/o16-smoke-ctx-10280431`（分支 v4-256concat1） |
| GL 4×A40 48G，FSDP4 | 16（global 64） | **未测**；参数／优化器分片比本机减半，但激活是本机 OOM 档的 4 倍，推断大概率 OOM | V6 实测 |

⚠ 64 档很可能过不了。按 K2，V6 的 64 档 FAIL 不是本计划失败，而是测完 64／32／16 三档峰值、取消两份占位后把数字交用户；若用户改选 32 或 16，步数等效口径（总样本 5.12M）分别为 160 000／320 000 步，占位重交，那是另一次裁决与开工令。两个 run 的 64 档都 PASS 且完成第一阶段后，按 K10 直接起正式训练。

耗时锚点：历史 `v1-prod-trend-10h`（4×A40、b64、512 位 context、无 motion）4.81 s/步；本次 896 位，按 Gemma prefix 长度近似线性外推 5.5～6.5 s/步；80 000 步 = 122～144 h（5.1～6.0 天）；176 h 占位对上限估算余量约 20%。两个 run 并行占 8 张 A40，约 1 000～1 150 GPU·h。用户已取消V7独立测时；起跑前采用原6.5s/步上界，实际剩171.40/171.83h、运行预留18.66%/18.96%，足够复用。正式step0到100的同步日志给出6.1363/6.3915s每步，80k估136.36/142.03h；这是正式运行日志，未新增测速任务。正式训练期间 Monitor 每 5000 步按日志时间戳核进度，预计超时（剩余步数 × 稳态步时 > 剩余时间）立即报用户（ckpt 只存 EMA，不能续跑）。

## 四、本机建表链路（每个 encoder 一遍，B4 两个并行各占一张 A6000，B6 串行共用 GPU 0）

| 步 | 命令锚点 | 输入 | 输出 | 估时 |
|---|---|---|---|---|
| B1 下 encoder | 逐对象匿名下载并核 SHA（同 `fetch_orig100_trainset.py` 式样） | bucket 的 `checkpoint_epoch_105.pt`（314 265 827 B）、`config.yaml`、`source_pin/source_receipts.json`、`acceptance/identity.json` | `$P/v1-store/external/motionjepa/<run>/` 与 turbo 同路径各一份 | 分钟级 |
| B2 下 wan-latents | `fetch_orig100_trainset.py fetch/unpack` 加 `wan-latents_tars/` 过滤 | 13 tar、26.80 GB | `$LIB/wan-latents/`（2496 段 `.bin` + metadata，26 676 559 872 B，按 `checksums/` 核成员） | 10～20 min |
| B3 源码身份 | 核 S7 来源清单与固定 4a 原 runner SHA；实际 WAN 包的 `source_pin()` 三字段匹配 `SOURCE_PIN.json`，G3 通过 `--mj-root /data/hongzefu/MotionJEPA` 按锁定 b5 blob 核两个实际加载文件 | 已有包与源码，不新增 MJ worktree | 包提交 b5、模块 SHA 不变；runner 单独锁原 4a 与本地 Beta | 分钟级 |
| B4 导出 | `env -u PYTHONPATH CUDA_VISIBLE_DEVICES=<0|1> $P/v1-store/venvs/wan-stage2/bin/python $P/scripts/dataset/wan_stage2/export_stage2_memory.py --lineage-build-spec $P/v1-store/external/rgb-gtxyz-g3-subset/control/build_spec.json --ckpt <ckpt> --expected-sha256 <sha> --latents $LIB/wan-latents --manifest $LIB/meta/episode_manifest.json --out $LIB/motion-tokens-stage2-<tag> --grid-subsample 2 --batch 1 --check-dropout 64 --expected-counts 2496,23253 --input-manifest $LIB/meta/input_manifest.json`（编码参数沿用 AWS；唯一新增参数是已按发布 SHA 核验的来源规格落点，不改原 config 或来源比较） | wan-latents、encoder、B5 已核 SHA 的 build_spec | 2496 个 `.f32.bin` 共 95 244 288 B，export_record `scope=full` | AWS A100 8 min，A6000 估 15～20 min |
| B5 G3 子集 | S3 新工具 `plan_g3_subset.py plan/fetch`，fetch 显式 `--workers 4` | rgb-gtxyz bucket | `$P/v1-store/external/rgb-gtxyz-g3-subset/`（真实为 14 片、115 对象、32.48 GB，两 encoder 共用） | 下载 1～2 h |
| B6 G3（缓存路线，K6a/K6b） | `CUDA_VISIBLE_DEVICES=0 UV_PROJECT_ENVIRONMENT=$P/v1-store/venvs/wan-stage2 uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/validate_stage2.py --gate G3 --samples $LIB/evidence/G0.json --ckpt <ckpt> --expected-sha256 <sha> --rgb-root <B5 目录> --mj-root /data/hongzefu/MotionJEPA --cache-import-receipt $LIB/evidence/wan_cache_import.json --out $LIB/evidence-<tag>/G3.json` | B5、encoder、库内 G0 与收据 | `G3.json`（K6a 形制） | 分钟级 |
| B6b 证据目录 | 把库内 `evidence/{G0,G2,SOURCE_PIN,wan_cache_import}.json` 与本机 `v1-store/validation/orig16-stage2/online_restore_inputs.{json,npz}` 逐字节复制进 `$LIB/evidence-<tag>/`，加 B6 的新 G3；在线输入先核原成功冒烟根快照、payload SHA、逐数组 shape/dtype/SHA | 原证据与两份 encoder 无关的真实在线输入，SHA 不变 | 新 G3 完成后五份闸门证据与两份在线素材齐全，供 pack、训练根快照及 SAVE_LOAD 读 | 分钟级 |
| B7 pack（S4 路径别名） | `pack_motion_store.py pack --layout stage2 --latent-grid-subsample 2 --latents-origin published-cache-with-local-fill --cache-import-receipt $LIB/evidence-<tag>/wan_cache_import.json --manifest $LIB/meta/episode_manifest.json --tokens $LIB/motion-tokens-stage2-<tag> --latents $LIB/wan-latents --out $LIB/motion-stage2-1024-<tag> --encoder-run-dir $P/v1-store/external/motionjepa/<run> --raw-dir /data/hongzefu/data_0226 --expected-ckpt-sha256 <sha> --stage2-validation $LIB/evidence-<tag> --stage2-source-pin scripts/dataset/wan_stage2/SOURCE_PIN.json` | B4、B6b、本机 H5 | 表 + meta（status=packed），provenance 的 evidence 路径指向 `$LIB/evidence-<tag>/` | AWS 24 min；本机含 16 个 H5 全量 SHA（约 500 GB）估 1～2 h |
| B8 verify | `pack_motion_store.py verify --store $LIB/motion-stage2-1024-<tag> --resume --check-padding` | B7 | status=verified | AWS 22 min |

说明：① 缓存路线下 `pack` 只对 G0/G2/G3/收据/SOURCE_PIN 取 SHA 写进 provenance，不校验 G3 指标（`_stage2_validation` 只在非缓存分支调用）；训练侧 `validate_stage2_cache_evidence` 校验 K6a 那组字段并要求 `g3.numerics == provenance.numerics`，所以 B4 与 B6 都在 A6000 上做。② 收据 `latents.root` 写的是 AWS 绝对路径，`load_cache_import` 现要求它 resolve 后等于 `--latents`，本机必 raise；S4 把这一处比较改为「以 `wan-latents` 相对路径为别名」，`metadata.json` SHA 与逐窗 SHA 比对不动，收据字节不动，G0/G2/旧表的 SHA 绑定全部保持。③ `load_cache_import` 按 `raw_dir / <文件名>` 打开 H5 核字节数与 SHA，不比对 `input_manifest.raw_dir`，所以 `/data/hongzefu/data_0226` 可用。④ 表与证据直接写在 turbo `$LIB` 下，训练侧无需再传递；本机 `$P` 经 symlink 读同一份。

## 五、要改什么（为什么非改不可）

| 块 | 现状（阻塞点） | 改法 | 文件 |
|---|---|---|---|
| S1 启动契约与 GL 入口 | `check_orig16_stage2_launch.py`：encoder 身份常量（`ENCODER_SHA`、`Contract.encoder_run`、`store_path`）、`HISTORIES` 八个固定名、`pair_name` 硬编码、`GL_COMBOS` 无 `(smoke,20):64` 与 `(formal,80000)`、`CFG_SCHEDULE/CFG_STEPS/ARGV_STEPS/CFG_RUN` 写死 60000、`--steps` choices、`gl_formal_run` 精确名、`check_attempt` 只认四个旧 run、`check_evidence_lineage` 读库根共享 `evidence/`；`run_orig16_stage2_60k_gl.sh`：`GL_HC` 单一、`20:64` 缺、`STEPS=60000`、`CONFIG` 写死、无 `ATTEMPT_MANIFEST` 处理；`run_in_hold_o16.sh` 不核 HEAD 与 clean | 契约加 `ENCODERS` 表（旧身份 + 两新身份：run、ckpt SHA、config SHA、store 目录、evidence 目录、yaml）按 `--encoder-run` 分派（第一部分与第二部分统一为这一个接口），默认值 = 旧身份、默认行为逐位不变；`HISTORIES`、`pair_name`、`GL_HC` 加两新 yaml；`GL_COMBOS` 加 `(smoke,20):(64,…)` 与 `(formal,80000):(64,)`；`CFG_SCHEDULE/CFG_STEPS/ARGV_STEPS/CFG_RUN` 按 `--config-name` 分派；choices 加 80000；`gl_formal_run` 表加 K8 两名并接受 `-attemptNN`；`check_attempt` 的 base 集合加 GL 名；血缘检查读 `ENCODERS[run].evidence_dir`；入口读 `CONFIG`／`ENCODER_RUN`／`HISTORY` 环境变量、`STEPS` 按 config、透传 `ATTEMPT_MANIFEST`；运行器加 `git rev-parse HEAD == TRAIN_HEAD` 与 `git status --porcelain` 为空 | `scripts/training/check_orig16_stage2_launch.py`、`scripts/training/run_orig16_stage2_60k_gl.sh`、`scripts/training/gl/run_in_hold_o16.sh`、`tests/training/test_orig16_gl_contract.py`、`tests/training/test_check_orig16_stage2_launch.py` |
| S2 训练配置与 yaml | 没有 b64/80k 的 orig16 条目；没有新 encoder 的 yaml | 新增 `TrainConfig(name="mme_vla_suite_orig16_b64_80k", batch_size=64, fsdp_devices=4, num_workers=8, num_train_steps=80_000, lr_schedule=CosineDecaySchedule(warmup_steps=10_000, peak_lr=5e-5, decay_steps=80_000, decay_lr=5e-5)`，其余逐键复制 60k 条目）；两份新 yaml 按 K4 命名，只改 motion 三键 | `src/mme_vla_suite/training/config.py`、两份新 yaml |
| S3 G3 子集工具 | 无 | `scripts/dataset/wan_stage2/plan_g3_subset.py plan/fetch`：下 `control/build_spec.json` + `index/` 四文件并按 `build_spec.index_files` 核 SHA；用与 `gate_g3` 逐字相同的抽样表达式得 32 窗；按 `row_start // 4096` 列分片；`fetch` 逐对象下载并核 bucket `SHA256SUMS`；末行 `G3_SUBSET=PASS shards=<n> bytes=<n> sha_mismatch=0` | `scripts/dataset/wan_stage2/plan_g3_subset.py`、`tests/dataset/test_plan_g3_subset.py` |
| S4 打包路径别名与 A100 守卫 | `pack_motion_store.py::load_cache_import` 要求收据 root == `--latents`；verify 分支读 provenance 里的 `latents_root`（AWS 路径）；`validate_stage2.py::setup_runtime` 只认 A100 | `load_cache_import` 的 root 比较改为按 `wan-latents` 相对路径别名（SHA 比对全部保留）；verify 分支以 `--latents`（或 store 同级库根）重基 `latents_root`；`setup_runtime` 接受 `A100` 或 `RTX A6000`（K6b） | `scripts/dataset/pack_motion_store.py`、`scripts/dataset/wan_stage2/validate_stage2.py`、对应测试 |
| S5 79999 交付与上传 | `check_stage2_checkpoint.py --delivery-step` 只认 59999/60000/70000，标签 `CKPT_79K_READY`；`--expected-manifest` 由 `_stage2_orig16_vla_ckpt_export_common.sh` 生产，但它锁定四个旧 run、旧 encoder、旧表与 60k 步集合 | choices 加 79999，`delivery_label` 对 79999 也返回 `CKPT_DELIVERY_READY`；导出脚本按 `ENCODERS` 表泛化 run／encoder／表／步集合（16 步含 79999）；`tests/policies/test_stage2_restore.py` 同步 | `scripts/training/check_stage2_checkpoint.py`、`scripts/dataset/hf_export/_stage2_orig16_vla_ckpt_export_common.sh` 及其 README、`tests/policies/test_stage2_restore.py` |
| 不改 | `framesamp_dataset.py`（(512,16,1)+context+motion 已放行）、`motion_store.py`、模型前向、loss、精度、`save_state`、`rgb_gtxyz_reader.py`（MJ 侧） | — | — |

## 六、验收（查什么 / 怎么查 / 过了说明什么 / 判定行）

| 编号 | 查什么 | 怎么查 | 过了说明什么 | 判定行 |
|---|---|---|---|---|
| V0 | 计划无自相矛盾 | 开工令后一个 `reviewer`（`gpt-6.1-sol/high`，read-only）审查，锚定本计划 sha | 可照着实施 | `PLAN_CONSISTENCY_REVIEW=PASS base=<sha> findings=<n> blocking=0` |
| V1 | encoder 字节同一 | B1 后 `sha256sum` 等于 K4 两值；config.yaml SHA 等于 `583e48c0…`／`869ac778…` | 用的就是 AWS 训出的 ep105 | `ENCODER_FETCH=PASS run=<run> ckpt_sha=<8> config_sha=<8>` ×2 |
| V2 | wan-latents 同一 | B2 逐对象 SHA + 按 `checksums/` 核成员 | 与 AWS latent 字节同一 | `LATENTS_FETCH=PASS objects=13 bytes=<n> sha_mismatch=0`、`LATENTS_UNPACK=PASS members=<n> mismatch=0` |
| V3 | 导出完整 | B4 日志 | 2496 段 23253 行、`scope=full`、dropout 诊断 `eval_repeat_max_abs=0` | `STAGE2_EXPORT=PASS segments=2496 rows=23253` ×2 |
| V4 | G3 缓存路线（K6a） | B5、B6 | encoder 在 32 个 val 窗上形状有限、EMA 65 叶严格一致 | `G3_SUBSET=PASS shards=<n> bytes=<n> sha_mismatch=0`；`EMA_SOURCE=PASS leaves=65 state_key=model exact=1 live_negative_rejected=1`、`EXPORT_PARITY=NOT_REQUIRED reason=user_decision_same_input`、record `windows=32 shape_mismatches=0 nonfinite=0` ×2 |
| V5 | 表打包与验证 | B7、B8；H5 核对由 `load_cache_import` 内置 | 表行序、字节、padding、provenance 闭合，证据目录自洽 | `PACK_PROVENANCE_PRECHECK=PASS mode=vae_compat segments=2496`、`VERIFY_MOTION=PASS scanned=23253 mismatches=0 nonfinite=0 …` ×2 |
| V6a | S1/S2/S4/S5 契约（CPU） | `DRY_RUN=1` 对两份新 yaml 的 **20 步 b64、100 步 b64、80000 步 b64 三条完整启动命令**各一次；拒绝用例：旧 yaml 配新 config、b128、60000、旧身份配新表、`-attempt01`；既有测试 `tests/training/ tests/datastore/ tests/dataset/ tests/policies/test_stage2_restore.py` 全过；默认参数（旧身份、60k）DRY_RUN 输出与 S4 的既有日志逐位相同 | 契约闭合、默认行为不变 | `LAUNCH_CONTRACT_GL=PASS gpus=4 fsdp=4 batch=64 steps=<20|100|80000> encoder=<run> checks=<n> failed=0` ×6、`CONTRACT_REJECTS=PASS cases=<n>`、`GL_CONTRACT_TESTS=PASS`、`DEFAULT_BITEXACT=PASS` |
| V6 | 占位内 64 档冒烟 | 占位 RUNNING 后，每个 run 在自己占位里 `o16gl-smoke-<tag>-b64` 20 步，nvidia-smi 500 ms 单持久进程采样（只查本 job 4 卡；采样对 20 步步时的干扰以 1031 本机冒烟有／无采样的步时对照为参考，写进留档）；PASS → 该 run 的 V6b记录与余量核算（V7用户取消），等两个 run 都通过并完成第一阶段后才 V9；OOM → 32、16 档 20 步只取峰值 → `scancel` ×2 → 停下交用户 | A40 上 64 档是否可行 | `SMOKE_GL=PASS|FAIL gpus=4 batch=64 steps=20 peak_mib=<n> nonfinite=0 [reason=oom]`；FAIL 时另两行 `SMOKE_GL_TIER=… batch=<32|16> peak_mib=<n>` 与 `HOLD_CANCEL=PASS jobids=<a>,<b>` |
| V6b | 保存恢复 | 20 步冒烟的 step 19 用 `check_stage2_checkpoint.py --mode smoke`（父 GPU 0、sidecar GPU 1）；跨卡型握手失败如实记录不改判据 | 链路通 | `SAVE_LOAD=PASS|FAIL …` |
| V7 | 独立100步测时取消 | 用户最新明确指令；仅精确取消65209321.6、65209330.6，保留原占位。起跑预算按K7，正式进度仍照常读日志 | 不再把额外测时作为正式前置，不伪填测时PASS | `TIMING_GL=NOT_REQUIRED reason=user_skip_timing`；`HOLD_MARGIN=PASS basis=plan_upper_6.5s est_train_h=144.4444 …` |
| V8 | 占位与部署 | `sbatch` 回执、`squeue`；集群副本 `git rev-parse HEAD` == Beta sha 且 `status --porcelain` 空；`.venv/bin/python -c "import mme_vla_suite;print(mme_vla_suite.__file__)"` 指向 `<GL_REPO>/src` | 两份占位在排，集群跑的是锚点代码 | `HOLD_SUBMIT=PASS jobid=<id> gpus=4 time=176:00:00` ×2、`DEPLOY_GL=PASS head=<sha> clean=1 pkg=<GL_REPO>/src` |
| V9 | 正式起跑（两个 run 的第一阶段完成后立即） | 两个 run 的 V6 均 PASS、V6b判定已记录、V7用户取消且余量够用，phase1 已 done；Beta 已打、launch.md 第一段已写、`srun` 起训练、日志 `TRAIN_CMD=` 与 `Step 100` | 跑的是锚点代码，与冒烟同 sha | `TRAIN_START=PASS run=<run> head=<sha> jobid=<id>` ×2 |
| V10 | 正式完成与交付 | 日志尾 `EXIT_CODE=0`；`--check-final-record`；步集合 {5000,…,75000,79999} 共 16；导出脚本产 `manifests/step-79999.json`；同占位内 `check_stage2_checkpoint.py --mode delivery --delivery-step 79999 --expected-manifest <manifest>` | 80k 跑完、79999 可读回 | `TRAIN_FINAL_RECORD=PASS run=<run> …`、`CKPT_STEPS=PASS … count=16`、`CKPT_DELIVERY_READY=PASS step=79999 …` ×2 |
| V11 | 上传长尾 | 集群 CPU job 逐对象读回 | 公开备份可信 | `HF_VERIFY=PASS bucket=<b> objects=<n> mismatches=0` ×2 |

## 七、步骤

| 阶段 | 内容 | 判据 |
|---|---|---|
| A0 | 开工审查 | V0 |
| A1（开工令后立即） | **先提交两份 4×A40 176 h 占位**（K7）；同时由主代理在 `$P` 分支为 S1～S5 各建一个独立 worktree，派五个 `worker` 写（§八）；同时 detached tmux 起 B1、B2、B3 | V8（占位行）、V1、V2 |
| A2 | S3 合入后跑 B5；S4 合入后 B6 可跑；S1/S2/S5 合入后跑 V6a | V4 子集行、V6a |
| A3 | 两个 encoder：B4 并行（GPU 0／1）→ B6 串行（GPU 0）→ B6b → B7 → B8；表与证据直接落 turbo `$LIB` | V3、V4、V5 |
| A3b | 写两份 `docs/training-doc/<run>/launch.md` 第一段 → **Beta commit**（`git status --short` 为空）→ 集群副本 `git fetch && git checkout <Beta sha>` → V8 部署行；占位可早于 Beta RUNNING，但部署前不得启动冒烟或训练 | V8 |
| A4 | 占位 RUNNING 后：V4b 计算节点 venv 核对（1031 同式）→ 64 档冒烟。**PASS** → 该 run 的 SAVE_LOAD原判定记录；100步测时已由用户取消。两个 run 的 64 档都通过后统一完成第一阶段，起跑前刷新剩余时间，余量 < 10% 自动重申请，否则原占位 B1；**FAIL** → 32／16 档峰值 → `scancel` ×2 → 停下交用户 | V6、V6b记录、K7余量（V7取消） |
| B1（两个 run 的第一阶段完成后立即，不另问） | 两个 run 各在自己占位里 `srun` 起训练（tmux `o16abl-train-<tag>`）→ 每份日志挂 Monitor；两个 run 的 64 档都通过、V6b原判定已记录、V7取消且余量够用且 phase1 done 后才起正式训练；先到的占位在此之前只做冒烟及规定判定并等待 | V9 |
| B2 | 跑完核 EXIT_CODE、final record、步集合；导出脚本产 manifest；同占位内 79999 严格恢复；按2026-11-08用户新指令保留65209321、65209330并在状态机记明转交`gl-abl-encoders-eval`，不取消；写 result.md、清洗日志、回写 commit、handoff；起上传长尾 | V10、V11 |

## 八、子代理分工与合并（简述，Codex 映射）

执行宿主是本机 Codex（K13）。角色对应：原计划的「haiku 探索」→ `explorer`，「opus worktree 写入」→ `worker`，「opus 运行型」→ `tester`，「sonnet 审查」→ `reviewer`；主代理 `gpt-6.1-sol/high`，所有子代理显式 `reasoning_effort="high"`，模型不高于主代理档位。

五块写入型任务（S2、S1、S4、S3、S5）：Codex 没有自动的子代理 worktree 参数，**主代理先在 `$P` 的 `v4-gl-orig100-1031` 上 `git worktree add` 五个独立目录（`$P/.claude/worktrees/codex-<S编号>/`，分支 `worktree-codex-<S编号>`），再各派一个 `worker`**，委派写明工作树绝对路径、基线 sha、可写集合、禁触路径、验收命令；worker 只在自己的 worktree 提交 `sub/<编号>: ` 前缀的 commit，不 push、不合分支。五者文件不重叠，合入顺序 S2 → S1 → S4 → S3 → S5。每块合前一个 `reviewer` 审固定 BASE/TIP（范围、前缀、默认行为逐位不变、K3 数字、SHA 比对未削弱），合后主代理在 `$P` 跑该块定向短测，`--no-ff` 合入、push。审查 FAIL 时用 `followup_task` 把 findings 交回原 worker 续改，不新开。

B1～B8 建表、占位提交、冒烟测时、正式起跑由 `tester` 启动：只起跑、验起跑、交回句柄，不改码、不改参数；主代理负责 tmux 监听（Codex 无 Monitor 工具，用 `tail -F` 管道加宿主等待机制）、占位登记与释放、合并审查、建档与交接。tester 发现代码问题只交证据，主代理另派 `worker` 修复并走同一审查合并流程。探索按需派 `explorer`，无需预列。

## 九、交接顺序与状态机登记（第 27 条）

| 项 | 内容 |
|---|---|
| 任务 id | `gl-abl-encoders-phase1`（A0–A4）、`gl-abl-encoders-train`（B1–B2，`depends_on` 前者） |
| phase1 交接前必须完成 | V1–V5、V6a、V8 全部 PASS；两个 run 的 V6 64 档均 PASS，且各自V6b原判定已记录（允许FAIL且不放宽），V7由用户取消记NOT_REQUIRED，起跑前实际剩余时间检查完成；任一 64 档 FAIL 时记录 32/16 档与 `HOLD_CANCEL`，phase1 保持 stopped 并交用户，不解锁 train；出处 `$P/v1-store/logs/o16abl-*.log`、turbo `v1-store/logs/o16abl-*.log`、本计划「实施中实测」小节 |
| 长尾 | phase1 无；train 两条 ckpt 上传（tmux `o16abl-upload-<tag>`，日志 turbo `v1-store/logs/o16abl-upload-<tag>.log`，完成行 `HF_VERIFY=PASS`） |
| 依赖的上游 | 无（两个 encoder 已 `done`，取自公开 bucket，不读 AWS 状态机） |
| done 后可开工的下游 | phase1 `done`（含两个 run 的 V6 64 档 PASS）→ train 自动 `running`，不另问；**train `done` 后**才可另立两个 run 的官方 800 局评估计划 |
| 登记 | 已登记（b019081），K11 |

## 十、用户已裁决清单与剩余待裁决

已裁决（2026-11-01／11-02 美东）：① 并行、不累积（K1）；② 先测 64（K2）；③ 80k 步（K3）；④ 不和 AWS 沟通、本地建表（K5）；⑤ 开工即提交带余量占位、起跑失败不再排队（K7）；⑥ 计划写 md；⑦ 登记状态机（K11）；⑧ 一次开工令覆盖全程、64 PASS 直接起跑、64 FAIL 测完三档取消占位再问（K2、K10）；⑨ 余量 20%（K7）；⑩ 余量不足 10% 自动重申请（K7）；⑪ G3 沿用 AWS 缓存先例（K6a）；⑫ 放宽 A100 守卫（K6b）；⑬ 不加对照 run（K12）；⑭ 两个 run 均通过 64 档及第一阶段后再正式起跑，Beta 早于首次冒烟／训练而非占位 RUNNING（用户原话「同意修订 继续做」）；⑮ 修订后无需再次确认、直接实施（用户原话「不需要再确认，你可以直接开始了。」）。

待裁决：⑯ 64 档 OOM 后的 batch 与步数（只在 A4 FAIL 时问）。

# 第二部分（技术细节，供 agent 追踪）

## 〇 前置声明与红线

- R1 不改训练语义：不改模型前向、loss、精度、数据格式、既有 yaml、`config.py` 既有条目、`framesamp_dataset.py`、`motion_store.py`、`save_state`；新 config 条目除 K3 五项外逐键等于 60k 条目。S4 对 `pack_motion_store.py`／`validate_stage2.py` 的改动只限路径别名与卡型白名单，任何 SHA 比对不得削弱。
- R2 不续跑：不启用 `--resume`；占位 176 h（K7）；64 档 OOM 时逐 JobID `scancel`；余量不足 10% 自动重申请新占位后再 `scancel` 旧占位。
- R3 不下载原始 H5（本机已有原件 `/data/hongzefu/data_0226`，只读）；不写 `/scratch/hongze` 进任何新默认值；不在本机新建 `/scratch`。
- R4 G3 子集放 `$P/v1-store/external/rgb-gtxyz-g3-subset/`，两 run 评估结束前保留，删前问用户。
- R5 两份 4 GPU 占位规格超默认，K7 已放行；同批两份是「同批多 job 并行」，A1 提交时记进 `greatlakes.md` 项目节；`scancel` 只按登记的 JobID 逐个做。
- R6 本机 GPU 只用于 B4、B6；不在本机跑 VLA 冒烟。
- R7 全部 tmux 前缀 `o16abl-`，清单写进 launch.md 与 current.json；清理只按清单。
- R8 证据目录按表分离（`evidence-<tag>/`），库根 `evidence/` 原件不改不删。

## 〇-C Codex 执行专章（第 26 条 Codex 专节与正本 `docs/codex-aspen-gpt56-roles.md` 的落地）

| 项 | 要求 | 核对命令／判定行 |
|---|---|---|
| 主代理 | `gpt-6.1-sol`，`model_reasoning_effort = "high"`（`~/.codex/config.toml` 现值一致） | 新任务 `/status` 核对；`CODEX_MAIN=PASS model=gpt-6.1-sol effort=high` |
| 并发 | `[agents] max_concurrent_threads_per_session = 32`（现值一致，开工时仍按第 26 条重查） | `grep -A3 '^\[agents\]' ~/.codex/config.toml`；`CODEX_THREADS=PASS n=32` |
| 角色预设 | `~/.codex/agents/{default,planner,explorer,worker,reviewer,tester}.toml` 全为 GPT-6／6.1 系列：explorer `gpt-6-luna/high` read-only；planner `gpt-6.1-sol/xhigh` read-only；reviewer `gpt-6.1-sol/high` read-only；worker、tester、default `gpt-6.1-sol/high`。注意：正本文档 §二 把 worker／tester 写成 `gpt-6-sol`，本机实际安装为 `gpt-6.1-sol`，本计划按实际安装值写，不改配置 | 开工时逐文件 `grep ^model`；任一绑 GPT-5.6 即停止派发并报告；`CODEX_ROLES=PASS roles=6 gpt6_series=6` |
| 派发纪律 | 每次 spawn 显式 `model` 与 `reasoning_effort="high"`，不依赖默认；全历史 fork 不接受覆盖时 `fork_turns="none"` 并给自含委派；子代理档位不高于主代理 | 委派文本留档 |
| 角色映射 | 探索→explorer；写代码／解冲突→worker；运行型（下载、建表、占位、冒烟、起跑）→tester；审查／对抗验证／V0→reviewer；计划修订（若需）→planner | 分配表「Codex 角色」列 |
| 写入隔离 | Codex 无自动 worktree：主代理 `git -C $P worktree add --detach $P/.claude/worktrees/codex-<S> <HEAD>` 后 `git checkout -b worktree-codex-<S>`，再派 worker；worker 每次写入前核 `git rev-parse --show-toplevel` 与 `git branch --show-current`；commit 前缀 `sub/<S>: `；不 push、不合分支、不 `--amend`／`reset --hard` | 合前 reviewer 核 `git diff --name-only BASE..TIP ⊆ 可写集合` |
| 生命周期 | 持久代理：同职责续用 `followup_task`，审查 FAIL 的续改交回原 worker；`wait_agent` 等待，不忙轮询；`list_agents` 核状态；不长期空挂，职责结束按宿主工具关闭 | — |
| 运行／修码分工 | tester 不改码、不改参数、不自派修复；证据交主代理 → 主代理派 worker → reviewer → 合入 → 交 tester 确定版本续跑 | — |
| 无 hook | 开工手动 `uv run --no-project python .agentmetarules/scripts/repo_guard.py start` 与 `fix`，三行判定写进第一条回复；每次 commit 后手动 `git status -sb` 核无 `ahead` | `REPO_SYNC`／`PUSH_STATE`／`RULES_SYNC` |
| 补丁工具 | 编辑优先 `apply_patch`；仅 Bubblewrap 权限故障时按第 20 条 `patch --dry-run --fuzz=0` 回退，禁整文件覆盖 | — |
| 监听 | Codex 无 Monitor 工具：每份日志 `tail -n +1 -F <log> \| stdbuf -oL tr '\r' '\n' \| grep --line-buffered …` 挂在宿主等待机制上，禁 `sleep` 忙轮询；tmux 规范照第 7 条 | — |
| 中文 | 子代理内部可英文，交回首行标「[内部产出，英文]」；主代理对用户一律中文 | — |

## 一、逐文件改动清单

| 块 | 文件 | 锚点 | 改什么 | 关闭态（默认） |
|---|---|---|---|---|
| S1 | `scripts/training/check_orig16_stage2_launch.py` | `ENCODER_SHA`、`Contract`、`HISTORIES`、`pair_name`、`history_key`、`GL_COMBOS`、`gl_formal_run`、`gl_combo_ok`、`check_attempt`、`check_history`、`check_config`（CFG_STEPS/CFG_SCHEDULE/CFG_RUN）、`check_argv`（ARGV_STEPS）、`check_dataset`／`check_evidence_lineage`、`main` argparse | `ENCODERS` 表与 `--encoder-run`／`--config-name` 分派；`HISTORIES` 与 `pair_name` 加两新 yaml；组合表加 `(smoke,20):64`、`(formal,80000):64`；日程与步数按 config 分派；choices 加 80000；正式名表加 K8 两名并接受 `-attemptNN`（`check_attempt` 的 base 集合加 GL 名、bucket 按 K9）；血缘检查读 `ENCODERS[run].evidence_dir` | 不传新参数时所有值等于原常量，输出逐位相同（V6a `DEFAULT_BITEXACT`） |
| S1 | `scripts/training/run_orig16_stage2_60k_gl.sh` | `CONFIG=`、`GL_HC`、组合表、`STEPS=60000`、`STEP_ARGS`、契约调用 | 读 `CONFIG`／`ENCODER_RUN`（默认原值）；`HISTORY` 允许集合加两新 yaml；`20:64` 放行；`STEPS` 按 `CONFIG`；转发 `--config-name`、`--encoder-run`；显式透传或 unset `ATTEMPT_MANIFEST` | 不设即原行为 |
| S1 | `scripts/training/gl/run_in_hold_o16.sh` | 计算节点判定段 | 加 `git -C "$REPO" rev-parse HEAD == $TRAIN_HEAD`、`git status --porcelain` 为空、`.venv/bin/python -c "import mme_vla_suite"` 的 `__file__` 在 `$REPO/src` | 新增检查，失败 exit 2 |
| S1 | 两份契约测试 | 夹具 | 加两新身份三步数 PASS 用例、拒绝用例、attempt 用例 | — |
| S2 | `src/mme_vla_suite/training/config.py` | 60k 条目之后 | 新增 `mme_vla_suite_orig16_b64_80k`（K3），注释写明只改五项 | 既有条目不动 |
| S2 | 两份新 yaml（K4 命名） | 新文件 | 复制原 yaml，只改 `motion.store_path`／`source_run`／`checkpoint_sha256`（`diff` 恰 3 行） | — |
| S3 | `scripts/dataset/wan_stage2/plan_g3_subset.py`、`tests/dataset/test_plan_g3_subset.py` | 新文件 | 见第一部分 §五 S3；文件头注明抽样表达式来源 `validate_stage2.py::gate_g3` 行号 | — |
| S4 | `scripts/dataset/pack_motion_store.py` | `load_cache_import`（root 比较）、verify 分支的 `latents_root` 三处 | root 比较改为按 `wan-latents` 相对路径别名；verify 以 `--latents` 或 store 同级库根重基；SHA 比对不动 | 收据 root 与 `--latents` 相同时逐位同原行为 |
| S4 | `scripts/dataset/wan_stage2/validate_stage2.py` | `setup_runtime` | 卡型白名单 `("A100", "RTX A6000")`，错误文案同步 | A100 上行为不变 |
| S5 | `scripts/training/check_stage2_checkpoint.py`、`tests/policies/test_stage2_restore.py` | `--delivery-step` choices、`delivery_label` | 加 79999；79999 也返回 `CKPT_DELIVERY_READY` | 既有步数行为不变 |
| S5 | `scripts/dataset/hf_export/_stage2_orig16_vla_ckpt_export_common.sh`（与 README） | `RUNS`／`ENC_RUN`／表／步集合常量 | 按 `ENCODERS` 约定支持 K8 两 run、K4 两 encoder 与表、16 步集合含 79999、K9 两 bucket | 旧四 run 行为不变 |

## 二、子代理分配表

| 子任务 | Codex 角色／隔离 | 目标 | 可写文件集合 | 禁触路径 | 接口契约与依赖 | 合并顺序 | 验收命令与判定行（`$P` 的 `.venv`，`UV_PROJECT_ENVIRONMENT=$P/.venv PYTHONPATH=<worktree>/src uv run --no-sync …`） | 资源 |
|---|---|---|---|---|---|---|---|---|
| S2 | `worker` gpt-6.1-sol/high，主代理预建 worktree | 新 TrainConfig + 两份 yaml | `config.py`（只增条目）、两份新 yaml | 其余全部 | 文件名与 config 名固定如 K4／§五 | 1 | `python -c` 断言五项数值 → `CFG_B64_80K=PASS`；`diff` 新旧 yaml 恰 3 行 → `YAML_DIFF=PASS lines=3` ×2 | CPU |
| S1 | `worker` gpt-6.1-sol/high，主代理预建 worktree | 契约、入口、运行器参数化 | `check_orig16_stage2_launch.py`、`run_orig16_stage2_60k_gl.sh`、`gl/run_in_hold_o16.sh`、两份契约测试 | config.py、yaml、模型、`pack_motion_store.py`、`motion_store.py`、`train.py`、`check_stage2_checkpoint.py` | 依赖 S2 文件名；`ENCODERS` 字段名写进文件头供 S5 引用 | 2 | `pytest tests/training/test_check_orig16_stage2_launch.py tests/training/test_orig16_gl_contract.py tests/training/test_stage2_launch.py tests/datastore -q` → `GL_CONTRACT_TESTS=PASS`；合后主会话跑 V6a 六条 DRY_RUN、拒绝用例、`DEFAULT_BITEXACT` | CPU |
| S4 | `worker` gpt-6.1-sol/high，主代理预建 worktree | 打包路径别名、A100 守卫 | `scripts/dataset/pack_motion_store.py`、`scripts/dataset/wan_stage2/validate_stage2.py`、对应测试 | `motion_store.py`、其余 | 不动任何 SHA 比对；root 相同时逐位同原行为 | 3 | `pytest tests/dataset -q -k "motion or stage2"` → `PACK_ALIAS_TESTS=PASS`；用旧表 provenance 构造 root 相同／不同两用例 | CPU |
| S3 | `worker` gpt-6.1-sol/high，主代理预建 worktree | G3 子集工具 | `plan_g3_subset.py`、`test_plan_g3_subset.py` | `validate_stage2.py`、MJ reader、其余 | 抽样表达式与 `gate_g3` 逐字相同；只读 bucket | 4 | `pytest tests/dataset/test_plan_g3_subset.py -q` → `G3_SUBSET_TESTS=PASS`；合后主会话真跑 `plan` → `G3_SUBSET_PLAN=PASS windows=32 shards=<n> bytes=<n>` | CPU，网络 |
| S5 | `worker` gpt-6.1-sol/high，主代理预建 worktree | 79999 交付与上传泛化 | `check_stage2_checkpoint.py`、`tests/policies/test_stage2_restore.py`、`_stage2_orig16_vla_ckpt_export_common.sh` 与 README | 其余 | 引用 S1 文件头的 `ENCODERS` 约定（不 import，只对齐名字） | 5 | `pytest tests/policies/test_stage2_restore.py -q` → `DELIVERY_79999_TESTS=PASS`；导出脚本 `DRY_RUN` 对 K8 两 run → `EXPORT_CONTRACT=PASS runs=2 steps=16` | CPU |
| R1 | `tester` gpt-6.1-sol/high，`$P` | B1、B2 | turbo `$LIB/wan-latents/`、`$P` 与 turbo 的 `v1-store/external/motionjepa/<run>/`、日志 `$P/v1-store/logs/o16abl-fetch-*.log` | 代码 | 无 | — | V1、V2；B3 实际源码身份由主会话核并记录；tmux `o16abl-fetch-enc`、`o16abl-fetch-latents` | 网络 |
| R2 | `tester` gpt-6.1-sol/high，`$P` | B5 | `$P/v1-store/external/rgb-gtxyz-g3-subset/`、日志 `o16abl-g3fetch.log` | 代码 | 依赖 S3 合入 | — | `G3_SUBSET=PASS …`；tmux `o16abl-g3fetch` | 网络，4 workers，≤80 GB（实测 32.48 GB） |
| R3a／R3b | `tester` ×2 gpt-6.1-sol/high，`$P` | 各一个 encoder 的 B4→B6→B6b→B7→B8 | `$LIB/motion-tokens-stage2-<tag>`、`$LIB/evidence-<tag>/`、`$LIB/motion-stage2-1024-<tag>`、日志 `o16abl-build-<tag>.log` | 代码；另一 tag 目录；库根 `evidence/` | 依赖 R1、R2、S4/S7 合入；B4 另须 B5 build_spec 发布 SHA 已核；B6 两者串行共用 GPU 0（noxyz 先，驱动脚本内用锁文件交接） | — | V3、V4、V5；tmux `o16abl-build-<tag>` | GPU 0／1 各一张（B4），CPU 48 线程（B7/B8） |
| R4 | `tester` gpt-6.1-sol/high，集群副本 | 两份占位提交 | turbo `v1-store/logs/o16abl-hold-*.log` | 代码 | 开工令 | — | V8 占位行；JobID 立即交回 | 2 × (4×A40, 176 h, 16 CPU, 128G) |
| R5a／R5b | `tester` ×2 gpt-6.1-sol/high，集群副本 | 占位内 V4b、V6、V6b原判定、K7余量（V7用户取消） | turbo `train-runs/o16gl-smoke-*`、`logs/o16abl-smoke-*.log` | 代码 | 依赖 R3 表落地、S1～S5 合入、Beta、部署 V8、占位 RUNNING | — | V6、V6b记录、余量；tmux `o16abl-smoke-<tag>` | 各自占位 |
| R6a／R6b | `tester` ×2 gpt-6.1-sol/high，集群副本 | 正式起跑 | turbo `train-runs/<run>`、`logs/<run>.log` | 代码 | 两个 run 的 V6 均 PASS、V6b已记录、V7取消且余量够用、phase1 done、Beta、建档第一段 | — | V9；tmux `o16abl-train-<tag>` | 各自占位 |
| 主会话 | — | 占位登记／释放、Monitor、合并审查、Beta、集群副本 checkout 与部署核对、建档、handoff、batch 裁决转交 | — | — | — | — | V0、V6a、V8、V10、V11 | — |

共享文件归属：`check_orig16_stage2_launch.py`／GL 入口／运行器只属 S1；`config.py` 与 yaml 只属 S2；`pack_motion_store.py`／`validate_stage2.py` 只属 S4；`check_stage2_checkpoint.py`／导出脚本只属 S5；`$LIB` 下按 `<tag>` 子目录分属 R3a／R3b；turbo `v1-store/logs/` 按前缀分属。

## 三、闸门总表

顺序：V0 → (V8 占位提交 ∥ V1、V2、B3 ∥ S1～S5 写与审) → S2 合 → S1 合 → S4 合 → S3 合 → S5 合 → V6a → B5（V4 子集行）→ V3 ×2 → V4 ×2 → V5 ×2 → A3b Beta → V8 部署行 →（占位已 RUNNING 或等待其 RUNNING）→ 各 run 的 V4b → V6 → [PASS] 各自V6b原判定记录 → V7用户取消 → 等两个run的64档全部PASS并完成规定记录 → 起跑前重新读取各占位剩余时间，余量 < 10% 则 `HOLD_RESUBMIT` → phase1 done → V9 ×2 → V10 → V11 ／ [任一 64 档 FAIL] 32、16 档峰值 → `HOLD_CANCEL` → 停下交用户。任一 FAIL 只记证据不改判据；V6 的 64 档 FAIL 按 K2 不算计划失败；V4 任一 encoder 的 `EMA_SOURCE` 不过（encoder 与锁定源码不同源）、V5 H5 SHA 不符（本机原件与 AWS 不同源）两种情形停下交用户。

## 四、runbook（开工后模板）

```bash
# A1-占位（集群副本，每 tag 一份；只改 --time 与 --job-name，其余沿用 hold_o16.sbatch）
sbatch --account=chaijy2 --partition=spgpu --nodes=1 --ntasks-per-node=1 --gres=gpu:4 --gpu_cmode=shared \
  --cpus-per-task=16 --mem=128G --time=176:00:00 --job-name=o16abl-hold-<tag> --wrap='sleep infinity'
# A1-数据（$P，detached tmux，日志三件套）
tmux new-session -d -s o16abl-fetch-enc "set -o pipefail; PYTHONUNBUFFERED=1 bash v1-store/logs/o16abl-fetch-enc.sh 2>&1 | tee v1-store/logs/o16abl-fetch-enc.log; echo \"EXIT_CODE=\$?\" >> v1-store/logs/o16abl-fetch-enc.log"
# A3-建表（$P，每 tag 一个 tmux；B6 经 $P/v1-store/logs/o16abl-g3.lock 串行）
tmux new-session -d -s o16abl-build-noxyz "bash v1-store/logs/o16abl-build.sh noxyz 0"
tmux new-session -d -s o16abl-build-nostate "bash v1-store/logs/o16abl-build.sh nostate 1"
# A3b-部署（集群副本，Beta 后；占位可提前 RUNNING 等待，部署完成前不运行冒烟）
git -C <GL_REPO> fetch -q origin && git -C <GL_REPO> checkout -q --detach <Beta sha> && git -C <GL_REPO> status --porcelain
# A4-占位内冒烟（经占位运行器，占位 RUNNING 后）
srun --jobid=<hold> --overlap --exact --ntasks=1 --cpus-per-task=16 --gpu_cmode=shared \
  env RUN_NAME=o16gl-smoke-<tag>-b64 CONFIG=mme_vla_suite_orig16_b64_80k ENCODER_RUN=<run> \
      HISTORY=perceptual-framesamp-ctx-32frame-4x4-<tag>-motion-stage2-4x256-orig16.yaml \
      GPUS=4 GLOBAL_BATCH=64 MAX_STEPS=20 TRAIN_HEAD=<Beta sha> \
      MMEVLA_STAGE2_EVIDENCE_ROOT=<GL_REPO>/v1-store/datasets/16task-orig100-1600ep-9e109419/evidence-<tag> \
      FINAL_RECORD_DIR=<GL_REPO>/v1-store/validation/o16abl/<run>/final-record \
  bash scripts/training/gl/run_in_hold_o16.sh
# B1-正式（两个 run 的 64 档均 PASS、V6b已记录、V7取消且余量够用且 phase1 done 后，同占位，不另问；失败重起换 RUN_NAME=<run>-attempt02 与新 FINAL_RECORD_DIR，并给 ATTEMPT_MANIFEST）
srun --jobid=<hold> --overlap --exact --ntasks=1 --cpus-per-task=16 --gpu_cmode=shared \
  env RUN_NAME=gl-orig100-ctx32x4x4-<tag>-motion-s2split4x256-b64-80k CONFIG=mme_vla_suite_orig16_b64_80k \
      ENCODER_RUN=<run> HISTORY=<新 yaml> GPUS=4 GLOBAL_BATCH=64 TRAIN_HEAD=<Beta sha> \
      MMEVLA_STAGE2_EVIDENCE_ROOT=<同上> FINAL_RECORD_DIR=<GL_REPO>/v1-store/validation/o16abl/<run>/final-record \
  bash scripts/training/gl/run_in_hold_o16.sh
```

每份日志一个 Monitor：`tail -n +1 -F <log> | stdbuf -oL tr '\r' '\n' | grep --line-buffered -E "EXIT_CODE=|Step [0-9]+:|RESOURCE_EXHAUSTED|Traceback|nan|NaN|Killed|=PASS|=FAIL"`。

## 五、风险登记

| 风险 | 处置 |
|---|---|
| 64 档 OOM（概率高，§三） | K2：记 FAIL，补 32／16 峰值，`scancel` 两占位，停下交用户 |
| spgpu 常年满卡，两份 4 卡 176 h 占位排队不可控 | 开工即提交，排队期间完成建表、代码、Beta、部署；排到一份且 Beta 已部署后先跑该 run 的 A4；待两个 run 都通过并完成第一阶段才正式起跑 |
| 实际步时可能高于初始预算，checkpoint只存EMA不能续跑 | 用户已取消V7；按原上界核余量并起跑后，持续以正式日志核进度，沿K7处理不足10%的余量，不伪造未完成测速结果 |
| 176 h 对 32／16 档可能不够 | 用户改选小 batch 时步数与 walltime 另算、占位重交 |
| G3 子集抽样复现偏差 | S3 逐字相同表达式并测试对拍；B6 若报缺分片按 FAIL 记并补下 |
| 本机 H5 原件与 AWS 不同源（只核过字节数） | B7 `load_cache_import` 逐文件 SHA，不符即 FAIL 停下 |
| `numerics.gpu_name` 绑定 | B4、B6 都在 A6000；B6 串行 GPU 0 |
| 收据 root 别名改动误伤 SHA 比对 | S4 审查项「任何 SHA 比对不得削弱」；root 相同用例逐位同原行为 |
| SAVE_LOAD 跨卡型握手 | 1031 已知，V6b 如实记录不改判据 |
| 集群副本与 Beta 不同 sha | A3b 显式 checkout，运行器核 HEAD 与 clean（V8 部署行） |
| 500 ms 采样干扰 | 单持久进程只查本 job 4 卡；步时对照写留档，不宣称零影响 |

## 六、盲区诚实清单

- 已实际通过两个B64/20：显存峰38489/38491MiB、nonfinite=0；正式首100步同步日志间隔6.1363/6.3915秒每步。未来整段吞吐及最终完成仍须以真实记录验收。
- G3 子集分片数（1～32）与字节（2.5～78.5 GB）在 S3 `plan` 跑完前不知道。
- 固定 4a 入口的本地副本（S7）在 A6000 上的耗时未测，按 AWS 8 min 外推；本次包实际提交 b5，两核心模块与 4a 字节相同。AWS 实际使用 4a 是历史参数推断，未读 AWS 日志。
- 两个 bucket 的对象列表只从 upload-SHA256SUMS 读到。
- 评估侧（800 局）不在本计划内。

## 七、留档与 commit 纪律

- phase1 的 V1–V8 齐全后在本计划末追加「实施中实测」小节（判定行原文、三档峰值表、`PLAN_CONSISTENCY_REVIEW` 行）；不改写上文口径。
- 两个 run 各建 `docs/training-doc/<run>/{launch.md,result.md,records/}`，两段式写入；建表按第 13 条在 `docs/dataset-build-doc/16task-orig100-1600ep-9e109419-motion-<tag>/` 留压缩版档案。
- 代码 commit `commitV11.74+`，合并提交 body 按第 11 条六项；起跑前 `commitV11.<n>Beta`；状态变化只碰 `docs/handoff/current.json` 的 `handoff:` 小 commit；每次 commit 后裸 `git push`。

## 八、对抗审查处置（Astra，AUDIT_BASE=125feda，2026-11-02 美东）

| 条 | 核实 | 处置 |
|---|---|---|
| 1 导出脚本缺 `--expected-counts` | 部分成立：`b5d23a5` 无此参数且写死 mixed8 计数；AWS 实际用 `4a16ab4`，模块 SHA 不变 | 历史修订先固定 4a 入口；实施中 S7 保留其字节及四处显式路径参数差异，实际包仍为 b5，详见 §十 |
| 2 G3 入口强制 A100 | 成立：`setup_runtime` 对带收据的 G3 仍生效 | K6b、S4 白名单加 `RTX A6000`（用户同意） |
| 3 收据 root 绑 AWS 路径 | 成立 | S4 路径别名，SHA 不动 |
| 4 「逐位相等」不会产生 | 部分成立：缓存路线本就不比较；AWS 先例 `EXPORT_PARITY=NOT_REQUIRED` | K6a、V4 改写为缓存口径（用户同意沿用先例） |
| 5 契约漏项 | 成立 | K4 命名、S1 全部触点、V6a 六条完整命令 |
| 6 证据根不贯通 | 成立 | 按表证据目录 `evidence-<tag>/`，S1 血缘读表目录，runbook 传 `MMEVLA_STAGE2_EVIDENCE_ROOT` |
| 7 79999 交付不支持 | 成立 | S5 |
| 8 失败不能重起 | 成立 | K8 `-attemptNN`，S1 放行 |
| 9 FAIL 分支顺序 | 成立 | K2：FAIL 不跑 V7；§九条件改写 |
| 10 部署缺步骤 | 部分成立：两边现同 sha，无同步机制 | A3b 显式 checkout，运行器核 HEAD 与 clean |
| 「唯一实验变量」 | 成立 | K12、§二措辞（用户裁决不加对照 run） |
| 测时与余量算法 | 成立 | 历史V7双口径设计；当前用户取消V7，保留`squeue %L`实际余量与正式日志核算 |
| B6 串行、运行器、128G、采样干扰 | 成立 | R3 锁文件；经运行器；128G；V6 干扰说明 |
| K11 陈旧、接口不统一、读取关系、「done 后可评估」、G3 目录 | 成立 | 已改 |
| 保存步集合、三 decoder 兼容、`chunk_motion.parquet` | 审查自己排除的误报 | 不改 |

## 九、开工审查两项时序裁决（2026-11-02 美东）

用户先说「同意修订 继续做」，随后明确「不需要再确认，你可以直接开始了。」。本节记录用户已批准的 v5 修订，不把修订本身伪称运行证据。原审查 `PLAN_CONSISTENCY_REVIEW=FAIL base=9551a288b2d4b5927a0aef6720a4e97b8c51893e findings=2 blocking=2` 保留在 `docs/validation/1102-gl-ablation-plan-consistency.md`。

| 项 | 原口径 | 已批准的新口径 | 核对场景 |
|---|---|---|---|
| 正式起跑与交接 | phase1 统一 done 与先到先正式起跑同时存在 | 先到者只先做冒烟与原保存恢复判定记录（V7已取消）；两个 run 都通过 64 档并完成第一阶段后统一解锁正式训练 | 甲 PASS、乙 PENDING：甲等待；乙后来 OOM：尚无正式训练，按 K2 测完档位并取消两占位 |
| Beta 时点 | Beta 必须早于占位 RUNNING | Beta 必须早于占位内首次冒烟或训练工作负载；占位可提前运行等待 | 占位立即 RUNNING：保持空占位，代码、建表、Beta、部署完成后才冒烟 |

共同起跑前按用户最新指令取消V7，使用原计划6.5s/步上界144.44h估算并使用当时重新读取的 `squeue %L` 核余量，沿 K7 不足 10% 自动重申请；不因等待而使用陈旧剩余时间。文首「唯一停点」仅指训练 batch 裁决，EMA 身份与 H5 SHA 失败仍按既有规则停工。两个一级部分、原定 S1～S5 写入任务与资源规模、80k 超参和身份判据均保持既定口径。按用户本轮明确授权，修订提交后直接做一次开工审查，通过后实施，不再次询问开工。

## 十、实施中调整

| 项 | 旧 → 新与理由 | 文件或归属 | 审查与规模 |
|---|---|---|---|
| 子代理工作树 | `$P/.claude/worktrees/` → 宿主工具返回的 `/home/hongzefu/.codex/worktrees/o16abl-sN/robomme_policy_learning_MotionJEPA`。宿主只接受主检出为创建入口，主检出与 `$P` 共用 Git 对象库；基线、写集和合入目标不变。 | S1～S5，原专属分支 `worktree-codex-SN`；仍整合至 `$P` 的 `v4-gl-orig100-1031` | `ADJUST_REVIEW=PASS scale=same findings=0`；仅改创建入口与实际路径 |
| B2 latent 解包 | 原以为只加过滤即可复用；实际 `TAR_DIRS`、解包根和校验入口仅认 source。补 latent tar 分支，成员 SHA 已在原 `sha256-source-files.txt`，复用现有校验；先下载至 `$P/v1-store/external/o16abl-latents-import/`，核验解包后只 `rsync -a` 同步 `wan-latents/` 至 `$LIB`，保留库原控制文件。 | 新增 S6，由 S2 同职责续用，专属分支 `worktree-codex-S6`；仅 `scripts/dataset/hf_export/fetch_orig100_trainset.py`、`tests/dataset/test_fetch_orig100_trainset.py`，其余禁触；CPU 本地 HTTP/tar 回归，独立审查后整合，网络运行另交 tester | `ADJUST_REVIEW=PASS scale=same findings=0`；下载总量、资源、原默认过滤及 source 行为不变；不增加状态、回退或保护层 |

原开工版本和两次调整审查出处见 `docs/validation/1102-gl-ablation-plan-consistency.md`。子代理宿主禁止自行提交时，由主会话在其独立工作树逐路径代作 `sub/<编号>` 提交，仍保留合前审查与 `--no-ff` 整合，不直接把改动带入主检出。

### S7 与在线素材的实际处置

| 项 | 旧 → 新、必要性与实际文件 | 审查与验证 |
|---|---|---|
| S7 固定入口的来源路径 | 4a runner 原先必读 config 中 AWS 绝对路径；改为入库单文件副本显式 `--lineage-build-spec`，不改原config/ckpt、编码、数值或完整16份H5来源比较。只新增 runner、`.source.json` 和对应测试；S4原写入代理在独立 `worktree-codex-S7` 同职责续用。G3直接按 b5 blob 核现有MJ两源码文件，无需新worktree或新包。 | `ADJUST_REVIEW=PASS scale=same findings=0`；`PRE_MERGE_REVIEW=PASS task=S7 ... findings=0`；10 CPU测试与撤去4处改动后的原文件SHA/AST还原通过。真实两encoder来源预检 `MJ_CLI_LINEAGE=PASS files=16 mismatches=0`。 |
| 在线恢复素材 | 原B6b仅五份闸门证据，训练根快照另必读两份在线输入。新增逐字节暂存 `online_restore_inputs.json/npz`，输入与encoder无关；原20步成功冒烟根快照的两份文件SHA、payload和逐数组shape/dtype/SHA均匹配。 | `ADJUST_REVIEW=PASS scale=same findings=0`；`EVIDENCE_STAGE=PASS tag=noxyz files=6 sha_mismatch=0 online_arrays=3` 与nostate同项，退出0；新G3尚待生成。 |
| G3下载worker细节 | 真规划32窗/14片/115对象/32476096020字节，默认workers1不变，运行显式4。先核SUMS与全部对象身份，再并发唯一对象；不改对象清单、预算或源字节。 | `G3_PARALLEL_FETCH=PASS workers=4`；6项HTTP回归、独立 `PRE_MERGE_REVIEW=PASS task=S3-fetch ... findings=0`；实际fetch以tmux `o16abl-g3fetch`起跑。 |

S1合前与合后复核曾分别发现旧8卡分支绕过、K2诊断小batch误拦，均由原S1代理在原五文件集合内最小续改，保留失败报告与全部sub提交。最终固定 `ccbbda239535e887e2acd38f1fa0156117526ed0` 的独立审查通过，完整core/shell成功与拒绝矩阵通过；正式100/80000步仍仅B64，20步诊断允许B64/B32/B16。以上属于把原计划实现正确，未改变用户口径。

## 十一、用户取消测时与正式启动实测

用户原话：「不用再测速了，可以正式训练启动。你再看一下这个，你先看一下这个冗余的时间还够不够，不够的话要重新启动任务。」随后要求「赶紧开始了你还在等什么！！！」。主会话仅停止两个100步测时step `.6`，保留两份hold；当时剩171.40/171.83h，对原计划144.44h估计分别留18.66%/18.96%，超过10%更换门槛，复用原占位。该起跑前值是预算估计，没有伪称完成精确测速。

两条正式任务在美东2026-11-02 10:23:07真正起跑，Beta `8b52210ff0048a9ea22776090fda3d76cd181c89`，step `65209321.7/gl1519`、`65209330.7/gl1503`，各4×A40、global64、workers8、80000步；MAX_STEPS和TRAIN_TIMING_STEPS均未设置、无NVML采样。各54项合同及实际mesh `(1,4)`通过。正式同步step100已到，step0→100日志间隔分别6.1363/6.3915秒每步，对全程估计136.36/142.03h；loss100分别0.36810/0.36983、grad_norm有限。两条`TRAIN_START=PASS`已可按实际step100记录，未把正式运行完成或策略性能写成PASS。

`SAVE_LOAD=FAIL`源于真实A40握手numerics与A6000离线来源不同，按既定V6b口径保留原判定，不改阈值。第一次post64因PATH把env解析到HOME不可执行路径，目标程序未起；独立`ADJUST_REVIEW=PASS scale=same findings=0`后仅改两个命令为`/usr/bin/env`，保留旧载体及失败日志，做一次基础设施重试。后续100步测时按用户要求取消，记`NOT_REQUIRED`。完整建表、代码、缓存及输入适配结果见数据档案；实际正式日志与句柄登记`docs/handoff/current.json`。

## 十二、正式训练完成与收尾中的调整（美东2026-11-08）

两条正式训练已完成80000步，训练代码仍为Beta `8b52210ff0048a9ea22776090fda3d76cd181c89`。noxyz于美东11月8日03:02:33退出0、记录耗时136小时36分；nostate于07:14:00退出0、耗时140小时47分。两条内日志均有`TRAIN_FINAL_RECORD=PASS tail_steps=99 ema_leaves=59`，各检查点目录集合精确为5000到75000每5000一步，加79999，共16份。训练完成和目录齐全不替代79999严格恢复或公开备份验收；当前任务仍为`running`。

| 收尾调整 | 旧 → 新与理由 | 独立审查、授权与实际范围 |
|---|---|---|
| 四文件的路径披露 | 原导出器离线模式在`encoder_dependencies.json P1`拒绝。现场扫描两run的交付根来源与训练日志，缺少7项披露裁决：三文件`encoder_dependencies.json`、`motion_provenance.json`、`_run-meta/train.log`的P1/P2，以及`motion_evidence/G3.json`的P3。用户在完整清单后明确「同意开始」，故只新增7项精确文件匹配，保留原字节与SHA，不改凭据硬停或恢复判据。 | `ADJUST_REVIEW=PASS scale=same findings=0`；披露提交`385fd43dd64b680e7aa738cff9768a7a4db63978`已push。`DISCLOSURE_SCOPE=PASS added=7 old_unchanged=1 exact_paths=4`。 |
| 上传与纯CPU读回分离 | 既有`Publisher::transfer/readback/verify_all`把上传与落盘回读混合，不能直接满足本计划V11及`greatlakes.md`标准分区纯CPU匿名流式要求。保留原离线导出器，新增小入口`stage2_bucket_delivery.py`：Aspen只上传；GL预装CLI列远端完整集合，无Authorization的HTTP请求逐对象流式核SHA与字节数、不落对象文件。清单自身也纳入交付身份，未知既存bucket拒绝接管。 | `ADJUST_REVIEW=PASS scale=same findings=0`；S8独立工作树仅新增入口与定向测试，10项真实本地HTTP测试通过，固定子提交`06ce58adb981de016901ad077eda5aa9a5f5fcb1`独立审查通过。合并及真实HF/GL结果另据现场回写，不把fixture当备份通过。 |

S8由`gpt-6.1-sol/medium`修改与独立审查，探索由`gpt-6-luna/high`执行，采用用户2026-11-04更新的通用映射。共享文件归属不变，S8仅有两个新文件。原主检出已由其他会话切到`v4.1-interpretMotion`且存在他人的未跟踪文档，因此本次计划与状态更新使用原任务分支`v4-eval-official-1026`的独立交接工作树，不切换、不清理他人的现场。

初次收尾载体未消费标签参数而让公共入口拒绝额外参数，退出2；消费参数后一次重试才得到真实披露失败，全部日志保留。原导出驱动短测59通过、9项被其硬编码AWS缓存目录`/scratch/hongze/.cache/uv`的子进程权限错误阻断；没有修改无关测试，也未虚报全套通过。79999严格恢复、占位释放、公开上传与集群CPU读回仍按各自实际判定留档。

### 本次最新用户覆盖与已启动句柄

用户原话：「Aspen如果传不上去就用vail来传试一下」。Aspen直接访问NFS部署的`git fetch`与共享工具预装卡在RPC，主会话核实进程归属后只停止该两项部署进程；在Aspen制作从训练Beta到`124b145f3f410a9e59b8b4372f90f51626319c57`的增量bundle，经SSH/rsync传到G，再由Aspen发起GL本地bundle导入和detached checkout，GL不需要Git凭据或网络fetch。共享HF工具固定1.30.0，在GL登录节点按共享Python3.11.14正常预装，计算节点不安装。独立`ADJUST_REVIEW=PASS scale=same findings=0`，G最终代码和clean已读回。

为尽早得到终态恢复结果，保持16步全量打包并行，用同一固定导出器原始代码块和原函数，在独立`delivery-terminal/stage`准备完整根依赖＋79999一步的恢复包；保留原16份集合检查、扫描、源文件稳定性复核与所有摘要判据。额外约24GB，`ADJUST_REVIEW=PASS scale=same findings=0`。两份`TERMINAL_PACKAGE=PASS`已生成。首次严格恢复因本轮遗漏创建编译缓存目录而拒绝；已补齐缓存，保留原FAIL报告，复用终态包并用新`report-attempt02.json`重试，不重跑训练、不覆盖失败。

Vail现场确认共享存储、预装HF工具和既有用户凭据可用。两条归档控制器已经在Vail的detached tmux `o16abl-upload-noxyz`、`o16abl-upload-nostate`启动：流式等待准确导出与恢复日志；导出必须退出0，恢复退出码原样记录；原计划仅V4的EMA来源及V5的H5 SHA失败硬停，两者已由实际前序证据通过，本次备份不把恢复FAIL伪写成PASS。随后Vail只上传，通过Aspen已有GreatLakes连接提交标准分区4CPU/8G/12h纯CPU任务，同一脚本先验一个小对象，再逐对象匿名流式核对集合、SHA和字节数。`RUNTIME_REVIEW=PASS findings=0`；不在计算节点装依赖，不在工作机回读，不将待上传写成已备份。

用户随后明确：「10 完成后不取消 65209321、65209330，在它的 current.json 里写明占位已转交 gl-abl-encoders-eval」。这条覆盖原B2释放指令：两份占位始终保留，不执行任何`scancel`；当前先在状态机标为`retained_for_transfer`，V10检查结束后记录`transferred`及实际验收结果，不能把仍运行中的验收标为完成。
