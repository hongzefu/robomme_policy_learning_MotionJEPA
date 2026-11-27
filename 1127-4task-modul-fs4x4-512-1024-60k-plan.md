# 1127 4task modulation 帧记忆每帧 4×4：预算 512（32 帧）与 1024（64 帧）各 50k 训练与三 seed 692 局评估方案

状态：裁决已落定（2026-11-27 美东），等开工令。分支 `v20-4task-modul-fs4x4-512-1024-60k`（自 `v18-4task-modul-e2eenc-s16-50k@11d84600`，该提交已在 origin；分支名里的 60k 是建分支时的旧值，保留不改，以本文为准）。

用户原话（2026-11-27 美东）：
- 「给出方案 跑这两组 512 1024的freamesample modul 注意是4task 给出gpu占用的方案 并行还是串」
- 「都用60k step　4×4 库从h5重抽要多久　写完整的2部分md计划！」
- 「数据集桶拉回原生 4×4 表 同意　评估改为3seed 先全部训完 再启动评估」
- 评估 ckpt 选 50k（与 Table I 同口径），随后改为：「都只训50k然后评估」

已定裁决：
- 4×4 表：从数据集桶拉回原生表，不重抽。
- 步数：两组都训 **50k**（取代早先的 60k）。
- 评估：两组都训完后再统一评估；ckpt **50000**；三 seed **7/0/42**。

来由：稿件仓 `motionjepa-paper-CoRLWorkshop` 的 `docs/1122-review-concerns-todo.md` 中的 B3。四个 motion-centric 任务上的帧记忆预算扫描（512→4096）只改帧数，每帧固定 8×8＝64 token，所以「加帧在 45% 左右饱和」「约六分之一的 token 超过 8 倍的帧」这两条结论只对这一种分配成立。RoboMME 设置用的是 32 帧×16 token。本方案补同预算、高时间密度（每帧 4×4＝16 token）的两档。

例外声明：11-14 用户约定「4task 每帧一律 8×8」。本方案是 B3 明确要求的 4×4 对照，作为该约定的例外，只适用于本分支这两个 run。

外部依赖锚点：
- 训练库 `v1-store/datasets/4task-v2-1600ep-604f16da`：manifest canonical id `4cd5a170…`，1600 ep，605,611 个执行样本，1,192,918 帧。
- norm_stats `v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json`，sha256 `856c75ea…`。
- 4×4 帧表来源：公开数据集桶 `hf://buckets/HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1/framesamp/`，37 个文件，73.0 GiB。属于 HF 禁删清单中的数据集桶。
- 起训权重 `pi05_base`。
- 训练代码：本分支（v18 基线）。
- 评估代码：**与 Table I fs 行同一套 v42 评估栈** `t1s-1120-evalstack-v42@c3876b9f`（worktree `/scratch/hongze/worktrees/t1s-ev-d6ece8b`，入口 `scripts/evaluation/local_692_pool.sh`，配对调度参照 `v1-store/jobs/t1s-1120/t1s-lib.sh` 的 `pair()`）。本方案从 c3876b9f 新开评估分支 `v20-evalstack-fs4x4`，只做登记类改动（见第四节第 5 条）。
- 对照数据：v18 `docs/training-doc/t1s-3seed-1120/final.md` 与 `table1_3seed.json`。

提交号沿用 `commitV<x>.<y>` 体例，从 v18 的 `commitV18.9` 往后接，本分支从 `commitV20.0` 起。

# 第一部分（给人看）

## 一、两个 run 与对照行

| | A（本方案） | B（本方案） | Table I 对照行 A′ | Table I 对照行 B′ |
|---|---|---|---|---|
| 帧记忆 | **32 帧×4×4＝512** | **64 帧×4×4＝1024** | 8 帧×8×8＝512 | 16 帧×8×8＝1024 |
| `token_per_image` / `budget` | 16 / 512 | 16 / 1024 | 64 / 512 | 64 / 1024 |
| 帧表 | `framesamp-4x4`（新拉回） | 同 A | `framesamp-8x8` | 同 A′ |
| 训练步数 | **50k** | **50k** | 60k | 80k |
| 评估 ckpt | 50000 | 50000 | 50k | 50k |
| 三 seed（7/0/42）692 局 | 待测 | 待测 | 211/231/223 局，**32.03 ± 1.45%** | 291/279/279 局，**40.90 ± 1.00%** |
| run 名 | `v2-1600ep-m32x4x4-modul-b128-50k` | `v2-1600ep-m64x4x4-modul-b128-50k` | `v2-1600ep-m8x8-modul-b128-60k` | `v2-1600ep-m16x8x8-modul-b128-80k` |

对照口径说明：
- A′、B′ 训练更长，但评估用的都是 50k ckpt。两份训练配置在 5k 预热后都是恒定 lr 5e-5，所以 50k 时两边经历的 lr 曲线完全相同，评估 ckpt 的步数也对齐。
- **ckpt 同位（用户 11-27 确认）**：Table I 两行用的是 step `50000`（`robomme-vla-modul-60k-v1/50000`、`robomme-vla-modul-1024-80k-v1/50000`，见 v18 `docs/training-doc/t1s-fs512-s0/README.md:11-12`、`t1s-fs1024-s42/README.md:11-12`）。训练循环是 `range(start_step, num_train_steps)`，在 `step % 5000 == 0` 或最后一步时存（`scripts/training/train.py:711,761-762`）。所以 `--num_train_steps=50001` 的最后一步就是 step 50000，存下 `50000` 后立即结束，不多训。设 50000 的话最后一份是 `49999`，和 Table I 差一步，不采用。
- 卡数差异：A′ 用 `mme_vla_suite_b128_60k`，即 4 卡 FSDP4、workers 8（v18 `docs/training-doc/v2-1600ep-m8x8-modul-b128-60k/launch.md`）；B′ 与本方案用 `mme_vla_suite_b128_80k`，即 8 卡 FSDP8、workers 16。global batch、预热、lr、EMA、seed 都相同，数据顺序与数值细节不保证逐位一致，只标注、不归因。
- 评估栈：A、B 与 Table I 的 A′、B′ seed 0/42 用同一套 v42 评估栈和本机前缀库评；A′、B′ 的 seed 7 是 GreatLakes 上的结果。平台差异在表里标注，不归因。
- 只交结果，不做显著性判读。
- 早先计划引用的 32.95%、41.33% 是 1028 汇总里的单 seed、最终 ckpt 数字，口径不同，本文不再使用。

其余全部沿用 8×8 扫描与 v11–v18 的 4task modulation 口径，不改：
- 模型 pi05（action_horizon 20，图像塔冻结），从 pi05_base 起训。
- modulation 接入，`memory_token_dim` 1024，`pool_type: mean`，`use_pos_emb: true`，`use_state_emb: false`，motion 关闭。
- batch 128，预热 5k，lr 5e-5 恒定（cosine 的 `decay_lr` 等于 peak），EMA 0.999。
- 8 卡 FSDP8，workers 16，seed 42。
- 每 5000 步存一份 ckpt，每个 run 10 份（5000…50000）。

A 的 yaml 就是现有基础配置 `perceptual-framesamp-modul.yaml`（budget 512，`token_per_image` 16）。为了留痕，复制成显式文件名 `perceptual-framesamp-modul-32frame-4x4.yaml`。B 再复制一份 `perceptual-framesamp-modul-64frame-4x4.yaml`，只改 `budget: 1024`。训练配置用 `mme_vla_suite_b128_80k` 加 `--num_train_steps=50001`，做法与 v11/v18 的 50k run 完全相同。

## 二、4×4 帧表：从桶里拉回（用户已同意）

**现状**：本机 4task 库只有 `framesamp-8x8`（292 GB）。9 月建库时 4×4 和 8×8 两档都打过包（`docs/dataset-build-doc/4task-v2-1600ep-604f16da/result.md`，4×4 pack+verify 用 210 秒，32 个 part 零差异），之后本机删掉了 4×4，原始 SigLIP 特征 npy 也只剩 8 个探针文件。`token_per_image` 读的是离线表，训练时不会现场池化，所以必须有 4×4 表。

**做法**
- 桶里的 `framesamp/` 就是当时上传的原生 4×4 表：32 个 part 加 pos、state 两张小表和 meta，共 37 个文件、73.0 GiB。每个文件的字节数和 sha256 都记在库内 `bucket-top/upload_manifest.json` 的 `plain_files` 里（`rel` 前缀为 `framesamp/`）。
- 只用网络和 CPU，估计 15–40 分钟（取决于 HF 带宽），不占 GPU。
- 落点 `v1-store/datasets/4task-v2-1600ep-604f16da/framesamp-4x4/`。直接同步桶内 `framesamp/` 子目录到这个目录，不能按桶根加 `--include` 下载，否则会多出一层 `framesamp/`（1114 审查 F4x4-2）。
- CLI 用 `/scratch/hongze/MotionJEPA/.venv/bin/hf`（1.9.2，有 `hf buckets sync/cp`；没有 `download` 子命令，见 1114 审查 F4x4-1），`HF_HOME=/scratch/hongze/.cache/huggingface-hongzefu`。

**为什么拉回的表不能用库自带的 `verify`**：`pack_framesamp_store.py verify` 要逐行和 source npy 对拍，全量 npy 已经不在了。改用第五节 G1/G2 的四项独立校验：逐文件 sha、行摘要自证、和本机 8×8 表的跨网格对拍、和 8 个探针 npy 的逐位对拍。

已做的预核：8 个探针上，原生 4×4 图像嵌入与 8×8 做 2×2 平均池化的结果余弦为 0.999997–0.999998，最大相对误差 0.4–0.5%，在 bf16 舍入量级。说明两档出自同一前向，跨网格对拍可以作为「行和样本对得上」的独立证据。位置嵌入 4×4 是 `PosEmb3D(ranges, 4)` 单独算的表，不是平均，所以只做逐位对拍。

**备选（未采用）：从 h5 重抽的耗时**
- SigLIP 在 8 卡上约 30–40 分钟（9 月 4 卡跑全量用了 66 分钟），之后 finalize 约 15 分钟、打包加校验约 4 分钟，再加短测和闸门。合计约 1.2–1.5 h。
- 另需约 950 GB 临时盘，且必须另建库根。
- 只在拉回失败、或 G1/G2 不过时才回到这条，届时先问用户。

## 三、GPU 方案：串行，每个 run 独占 8 卡（推荐）

| 方案 | 训练 | 评估（6 次：2 个 run × 3 seed） | 全部完成 | 风险 |
|---|---|---|---|---|
| **串行 8 卡（推荐）**：A 训 → B 训 → 3 对评估 | A：50k×约 0.95 秒/步≈**13.2 h**（参照 1122 那条 32 帧×4×4 modulation 8 卡实测 0.971 秒/步，它的记忆更长，有 704 token）；B：50k×约 0.99 秒/步≈**13.8 h**（参照 8×8 的 1024 档 8 卡实测 0.986 秒/步）；合计约 27 h | 按 Table I 的同卡双评估配对（两次评估同时跑，各 16 片、每卡 2 片，共用占位、每卡上限 4），每对约 1.16 h（Table I fs 行实测）；3 对约 **3.5 h** | **约 31 h** | 低：8 卡 FSDP8 b128 是 1122、v11–v18 一直在用的配置；A、B 的记忆 token 数和 8×8 的 512、1024 两档相同；评估配对是 Table I 跑过的方式 |
| 并行 4+4 卡：A、B 同时训，然后评估 | 4 卡每步约慢一倍（8×8 的 512 档 4 卡实测 2.34 秒/步；按 1.9–2.0 秒/步算），两个 run 同时约 27 h | 同左，约 3.5 h | 约 31 h | 中：1024 档在 4 卡 b128 上没跑过，显存未知；两条训练日志、两套句柄同时盯 |

改成「全部训完再评估」之后，两种方案的完成时间基本一样，串行原来能提前出 A 结果的优势也没有了。仍推荐串行，理由只剩风险：每一步都是验证过的 8 卡配置，监控也只有一条训练线。顺序 A→B。

训练和评估不能叠在同一批卡上：训练每卡预占 0.95 显存，评估每份占 0.2。

## 四、代码改动（开工后只用 CPU，约 1–2 小时）

在本分支的 worktree 里改，不碰主检出的其他分支工作树。

1. **配置**：新增 `perceptual-framesamp-modul-32frame-4x4.yaml`（等于 `perceptual-framesamp-modul.yaml`，只加注释）和 `perceptual-framesamp-modul-64frame-4x4.yaml`（只改 `budget: 1024`）。不改任何已有 yaml。
2. **数据集白名单（阻塞项，审查 train-F1）**：v18 的 `src/mme_vla_suite/training/framesamp_dataset.py:96` 只放行 `(budget, token_per_image, num_views)` 为 (512,16,1) 和 (512,64,1)，B 组 (1024,16,1) 在数据集初始化时就会抛 ValueError。改为在集合里加上 (1024,16,1)，其余档位照旧拒绝；补定向 pytest。顺带核对 `mem_buffer`、`policies/framesamp_memory.py`、`history_pi0` 里没有其他按 512 写死的地方。
3. **拉回与校验**：新增 `scripts/dataset/fetch_framesamp_4x4.sh` 和 `scripts/dataset/check_framesamp_4x4_fetch.py`，判定行见第五节 G1/G2。
4. **runner**：新增 `scripts/training/run_4task_fs4x4_50k.sh <TRAIN_HEAD> --budget 512|1024 [--smoke20] [--dry-run]`，**以 v11 的 `run_4task_wantp768_50k.sh` 为主底**（体量小，没有 e2e 专属项）：
   - `--dataset-path $DS/framesamp-4x4`；`--model.history-config` 按 budget 取上面两份 yaml，并钉死 yaml 的 sha256；`STEPS=50001`；`EXPECT_CKPTS` 为 5000…50000；`TRAIN_TIMING_STEPS=300`。
   - 显式 `export MMEVLA_FRAMESAMP_SOURCE=$DS/source MMEVLA_FRAMESAMP_MANIFEST=$DS/meta/episode_manifest.json`，不靠桶里 store_meta 记录的 9 月绝对路径兜底（双根契约是环境变量优先）。
   - 删除 v11 里的 motion 专属项：`MOTION`、`LATENTS_META_SHA`、wan 表 provenance 检查、`MMEVLA_MOTION_*`，并 unset 全部 motion 环境变量；保留 HEAD 与工作区干净检查，以及「拒绝覆盖本轮日志」。
   - 不参照 v18 的 `run_4task_e2eenc_50k.sh`：它有 e2e/Wan9 专属闸门（G0 环 3/4、`E2E_ROWS`、`WAN9_LEAVES`、`e2e_enc_grad_norm`、probe 模式）以及写死的 `framesamp-8x8` meta 读取。
5. **启动合同**：新增 `scripts/training/check_4task_fs4x4_launch.py`，判定行 `LAUNCH_CONTRACT=PASS`。要求：
   - A 的 yaml 相对 `perceptual-framesamp-modul-8frame-8x8.yaml` 只差 `token_per_image`（16 对 64）和注释，B 相对 A 只差 `budget`（1024 对 512）；
   - 训练配置为 `mme_vla_suite_b128_80k`，`num_train_steps` 覆盖为 50001；
   - 帧表 meta 的 layout 为 `framesamp-4x4-v1`、`status=verified`、`num_rows=1192918`、`manifest_sha256=4cd5a170…`；`budget // token_per_image` 等于 32 或 64；
   - 两个 `MMEVLA_FRAMESAMP_*` 路径存在；
   - **用真实 4×4 meta 在 CPU 上实例化 `FrameSampDataset`**，budget 512/1024 都通过，取 1 个样本，核对帧数为 32/64、每帧 16 token。
6. **评估（在 v42 评估栈上，审查 eval-EV-1/2/3）**：从 `t1s-1120-evalstack-v42@c3876b9f` 新开分支 `v20-evalstack-fs4x4`，独立 worktree `/scratch/hongze/worktrees/fs4x4-ev-<sha>`，只做三类改动：
   - 加入与本分支逐字节相同的两份 4×4 yaml，供评估 server 加载；
   - `scripts/evaluation/check_checkpoint.py`：把写死的 `token_per_image == 64`（第 56 行）改成按 run 条目登记的值，缺省 64，旧条目行为不变；在 `RUNS` 里登记 A、B 两个正式 run 根名和 A 的冒烟 run 根名，字段写明 budget 512/1024、`token_per_image` 16、motion 关闭、steps 50000/19、`resolved_sha256`（用 `--resolved-sha256` 对新 yaml 复算）、`norm_stats_sha256` `856c75ea…`；
   - 评估调度脚本 `scripts/evaluation/fs4x4_eval_pairs.sh`，照 `t1s-lib.sh` 的 `cfg()`/`run()`/`pair()` 写：`PYTHONPATH` 指向评估 worktree，`EXPECT_MOTION_STATS=0 MMEVLA_MOTION_OVERFLOW=`，`POLICY` 取新 yaml 名，`POLICY_CONFIG=mme_vla_suite_b128_80k`，`RUN_NAME=fs4x4-{a,b}-s{7,0,42}`；配对为 A 的 s0+s42、B 的 s0+s42、A 的 s7+B 的 s7。
   - v42 栈已经支持关闭 motion 的判据（Table I fs 行就是这样评的），所以**不再改 v18 的 `local_692_pool.sh`/`run_shard.sh`/`check_shard.py`/`merge_shards.py`**。
   - 另加三 seed 汇总脚本 `scripts/evaluation/fs4x4_3seed_summary.py`，输出与 `table1_3seed.json` 同格式的两行，判定行 `FS4X4_3SEED=PASS rows=2 seeds=3`。
7. **触发链与上传**：新增 `scripts/training/chain_4task_fs4x4.sh`，按第六节串起闸门、训练和评估（评估一步调用评估 worktree 里的 `fs4x4_eval_pairs.sh`）。链脚本只有仓库里这一份，不在 `v1-store/jobs` 下另放副本。新增增量上传包装器 `scripts/dataset/hf_export/run_fs4x4_ckpt_export.sh`（以 v18 的 `run_e2eenc_ckpt_export.sh` 为底，每出一份 ckpt 就上传并读回校验）。
8. **测试**：给 2、3、5、6 加定向 pytest，覆盖白名单正反例、校验脚本、启动合同正反例、`check_checkpoint` 的 4×4 登记（旧条目不变）、三 seed 汇总。

分工：写入用 opus 子代理，各自开 worktree：S1 负责 1、2、4、5，S2 负责 3，S3 负责 6（评估分支），S4 负责 7。每份合入前派 sonnet 审查，主会话按 `--no-ff` 合入。

## 五、开训前闸门（全部 PASS 才起正式训练）

| 闸门 | 判定行 | 何时 |
|---|---|---|
| G0 单测 | 两条分支的定向 pytest 全过；`check_checkpoint.py` 对旧 RUNS 条目的结果不变 | 代码合入后 |
| G1 拉回 | `FS4X4_FETCH=PASS files=37 bytes=<总字节> mismatches=0`：37 个文件逐个比对 `upload_manifest.json` 的字节数和 sha256 | 拉回后，CPU |
| G2 帧表自证 | ① `FS4X4_ROWDIGEST=PASS rows=1192918 mismatches=0`：按 `ROW_DIGEST_COVERAGE`（image‖pos‖state 三键 store 侧原始字节，依此顺序拼接）逐行重算 blake2b-128，与 `meta/row_digests.blake2b.bin` 比对。image 和 state 按全局行号取；**pos 表只有 2304 行，按帧号 t 取**，t＝行号减去该 episode 的 `total_sample_offset`，episode 切分取自 `meta/episode_manifest.json`，做法同 `pack_framesamp_store.py` 的 `_verify_part_worker`；② `FS4X4_XGRID=PASS rows=1192918 min_cos=<x>`：每行 4×4 图像与本机 8×8 同行做 2×2 平均池化后比对，余弦全部 ≥ 0.9999，任一行不满足即 FAIL；③ `FS4X4_PROBE=PASS frames=8 mismatches=0`：8 个探针 npy 的 `image_emb_4x4`、`pos_emb_4x4` 与表中对应行逐位相等；④ `FS4X4_META=PASS`：layout、status、num_rows、manifest_sha256 四项 | G1 后，CPU，约 20–30 分钟（并行进程池，按核数留余量） |
| G3 启动合同 | `LAUNCH_CONTRACT=PASS`（A、B 各一次，含 CPU 实例化 `FrameSampDataset`） | CPU |
| G4 8 卡 20 步短测 | A、B 各一次 `SMOKE20=PASS save_load=PASS`，并记录显存峰值和首 300 步之外的步速 | GPU，每次约 10 分钟 |
| G5 评估加载冒烟 | 在评估 worktree 上，用 A 的短测 ckpt 跑 `check_checkpoint.py` PASS 后，再跑 `local_692_pool.sh --g6b`，得到 `EVAL_LOAD_SMOKE=PASS episodes=1 handshake=PASS share2=PASS`（`EXPECT_MOTION_STATS=0`） | GPU，约 10 分钟 |

## 六、起跑与时间线（串行，全部训完再评估）

触发链 tmux `fs4-chain`，在 detach 到 Beta 提交的主检出上执行 `bash scripts/training/chain_4task_fs4x4.sh <Beta> <评估 worktree 路径>`，按序执行：

1. 等 8 卡显存都低于 2000 MiB，否则不起跑并报告用户；
2. G4（A、B 各一次）→ G5；
3. 正式训练 A（`v2-1600ep-m32x4x4-modul-b128-50k`）；
4. 正式训练 B（`v2-1600ep-m64x4x4-modul-b128-50k`）；
5. 两个 run 都有 `EXIT_CODE=0` 且 `50000` 都在之后，在评估 worktree 上跑 `fs4x4_eval_pairs.sh`，顺序为 A 的 s0+s42、B 的 s0+s42、A 的 s7+B 的 s7，共 3 对 6 次。每对两次评估同时跑，各 16 片、每卡 2 片、每份 0.2 显存，共用占位目录、每卡上限 4，与 Table I 一致；
6. 三 seed 汇总，得到 `FS4X4_3SEED=PASS`。

任一闸门 FAIL 或任一步非 0 退出即停链。

预计（从开工令算起）：代码与拉回、G0–G3 约 2–3 h；G4–G5 约 0.5 h；训练 A 约 13.2 h、B 约 13.8 h；3 对评估约 3.5 h。**全部结果约在开工后 33–34 h 出来。** 实际步速以 G4 和正式训练前 300 步的计时为准，届时修正本节。

## 七、ckpt 与磁盘

- 现剩 1.6 TB。4×4 表 73 GiB；每个 run 10 份 ckpt，按 8×8 的 512 档每份约 11 GiB 估算，两个 run 约 220 GiB，本地全部保留。6 次评估的视频和分片产物每次约 1 GB。
- 增量上传：每出一份就上传到公开桶 `HongzeFu/robomme-vla-4task-modul-m32x4x4-50k-v1` 和 `HongzeFu/robomme-vla-4task-modul-m64x4x4-50k-v1`（不带 `--private`），逐份读回校验 sha。tmux 分别为 `fs4-incA` 和 `fs4-incB`。
- 磁盘余量低于 200 GB 时停下报告。

## 八、评估与交付

- 4task 测试集 692 局，每局最多 2000 步，ckpt 50000，seed 7/0/42，全部在本机评。
- 交付：A、B 各自三 seed 的成功局数、均值 ± 样本标准差和逐任务表，与 Table I 的 A′（32.03 ± 1.45）、B′（40.90 ± 1.00）以及 8×8 的 2048/4096 两档（44.85 ± 0.96、45.13 ± 1.59）并列，标注评估平台和训练总步数；同时给 A、B 的实测训练步速与评估墙钟。只交结果，不做显著性判读。
- 结果文档：训练写到 `docs/training-doc/v2-1600ep-m{32,64}x4x4-modul-b128-50k/result.md`；评估写到 `docs/training-doc/fs4x4-{a,b}-s{7,0,42}/README.md`（注明评估栈 sha），逐局 CSV 进 `records/`；三 seed 汇总写到 `docs/training-doc/fs4x4-3seed-1127/final.md` 和 `fs4x4_3seed.json`。

## 九、裁决与剩余默认项

已定（用户 11-27 美东）：
1. 4×4 表从数据集桶拉回原生表。
2. 两组都只训 50k。
3. 两个 run 全部训完后再评估；ckpt 50000；三 seed 7/0/42。

按默认处理（用户未另行指示）：
4. GPU 方案：串行 8 卡，A→B。
5. 不在本机补评 A′、B′ 的 seed 7（Table I 的 seed 7 是 GL 结果，seed 0/42 已是本机结果）。
6. B3 的其余部分不在本方案内：128 帧×16（预算 2048）需另训一个约 14 h 的 run；推理延迟或 FLOPs 对比只需一次 GPU 测速，不用训练。
7. 状态机登记任务 `4t-fs4x4-50k`，本分支已 push。

开工令覆盖：第四节代码、拉回与 G0–G3（CPU/网络）＋ G4–G5 ＋ 第六节自动串行跑完两次训练和六次评估。中途只在以下情况停下：闸门 FAIL、训练崩溃且属于数值语义问题、磁盘不足、卡长时间不空；其余工程故障自行修复后继续。

## 九′、开工前审查与处置（2026-11-27 美东）

workflow `wf_09b13830-86b` 审查 `decd3e26`：6 个维度各派 1 个 sonnet 审查，每条发现再派 1 个 sonnet 复核。共 15 条，13 条成立、1 条驳回、1 条重复。运维维度没有交出发现。均不涉及训练语义，已全部改进本文：
- 阻塞 2 条：数据集白名单不放行 (1024,16,1) → 第四节第 2 条；`check_checkpoint.py` 写死 token_per_image==64 且 run 名未登记 → 第四节第 6 条。
- 评估改用 Table I 同一套 v42 栈和同卡配对：耗时从 10.8 h 降到约 3.5 h，评估栈口径与 Table I 一致。
- 其余已改：不存在的 `16frame-8x8.yaml` 引用；A′ 的 4 卡 FSDP4 差异；runner 改以 v11 为底并列出删除项；显式导出 `MMEVLA_FRAMESAMP_*`；链脚本路径统一；G2① 的 pos 按 t 索引。
- 驳回 1 条：S3 依赖与并行派发的写法本来不矛盾。

## 十、交接顺序与状态机登记

- 任务 id：`4t-fs4x4-50k`，无前置依赖（8 卡当前空闲，tmux 上没有本项目的在跑会话）。
- 交接前必须完成：G0–G5 全部 PASS；两份训练日志都有 `EXIT_CODE=0`；6 次评估都有 `EVAL692=PASS episodes=692`；三 seed 汇总 `FS4X4_3SEED=PASS rows=2 seeds=3`；两个 run 的 50000 都已上传，且读回 `HF_VERIFY=PASS`。
- 长尾：增量上传的读回校验（`fs4-incA`、`fs4-incB`，完成行 `INC_END`）。
- 本任务 `done` 之后：第九节第 5、6 项由用户决定。

# 第二部分（技术细节，供 agent 追踪）

## 〇、红线

1. 不改任何已有 yaml、训练配置、`framesamp-8x8` 表或 `source/`；只新增。代码只允许第四节列出的改动：数据集白名单加一档、`check_checkpoint.py` 的按条目登记。A 的 yaml 相对 `perceptual-framesamp-modul-8frame-8x8.yaml` 只允许 `token_per_image`（和注释）不同，B 相对 A 只允许 `budget` 不同。
2. 不删除、不覆盖数据集桶（HF 禁删清单）；拉回只读。本地 ckpt 不删。
3. 不 force push；子代理不 push、不合主分支。
4. tmux 只建以 `fs4-` 为前缀的会话，清理只用 `tmux kill-session -t '=确切名'`，并逐个列清单。
5. 不用 worktree 里的脚本跑主数据：触发链在主检出 `git checkout --detach <Beta 提交>` 之后、执行子模块更新和 `uv sync --frozen` 再起跑（1122 实施中的教训）。
6. 评估一律在两个训练都结束后才起，不在训练间隙插评估。

## 一、子代理分配表

| 编号 | 目标 | 可写文件 | 依赖 | 验收 |
|---|---|---|---|---|
| S1 | 两份 yaml、数据集白名单、runner、启动合同 | `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-{32,64}frame-4x4.yaml`、`src/mme_vla_suite/training/framesamp_dataset.py`（只改第 96 行白名单集合）、`scripts/training/run_4task_fs4x4_50k.sh`、`scripts/training/check_4task_fs4x4_launch.py`、`scripts/training/tests/test_check_4task_fs4x4_launch.py`、`scripts/training/tests/test_framesamp_dataset_budget.py` | 无 | runner `--dry-run` 对 A、B 打印的 `TRAIN_ARGS` 正确（dataset-path、history-config、`--num_train_steps=50001`、两个 `MMEVLA_FRAMESAMP_*`），且不含 motion/e2e 项；白名单对 (512,16)/(512,64)/(1024,16) 放行、其他拒绝；合同脚本反例（改 lr、改 pool_type、budget 不整除、步数不是 50001）均 FAIL；pytest 全过 |
| S2 | 4×4 表拉回与四项校验 | `scripts/dataset/fetch_framesamp_4x4.sh`、`scripts/dataset/check_framesamp_4x4_fetch.py`、`scripts/dataset/test_check_framesamp_4x4_fetch.py` | 无 | 用合成小表（2 个 episode、pos 按 t 索引，其中一行故意篡改）验证 ROWDIGEST/XGRID/PROBE 都能抓到；校验用进程池并行；按 `upload_manifest.json` 里 `framesamp/` 前缀映射到 `framesamp-4x4/` |
| S3 | v42 评估分支：yaml、`check_checkpoint` 登记、配对调度、三 seed 汇总 | 评估分支 `v20-evalstack-fs4x4`（基线 c3876b9f）上的：`src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-{32,64}frame-4x4.yaml`（与 S1 逐字节相同）、`scripts/evaluation/check_checkpoint.py`、`scripts/evaluation/fs4x4_eval_pairs.sh`、`scripts/evaluation/fs4x4_3seed_summary.py`，以及对应 test 文件（文件名由 S3 在 `scripts/evaluation/` 下新建，并在交付里列出） | 无（run 根名、ckpt 路径、yaml 内容由主会话在派发说明里写死） | `check_checkpoint.py` 对全部旧 RUNS 条目的结果不变，对 4×4 新条目在 mock 快照上 PASS、token_per_image 不符时 FAIL；`fs4x4_eval_pairs.sh --dry-run` 打印 3 对 6 次的 RUN_NAME、seed、shard_base、POLICY；三 seed 汇总在 mock 数据上输出与 `table1_3seed.json` 同格式 |
| S4 | 触发链与增量上传 | 本分支的 `scripts/training/chain_4task_fs4x4.sh`、`scripts/dataset/hf_export/run_fs4x4_ckpt_export.sh`（以 `run_e2eenc_ckpt_export.sh` 为底） | S1、S3（接口由主会话写死：runner 参数、`fs4x4_eval_pairs.sh` 的调用方式） | 链脚本 `--dry-run` 打印完整步骤，评估排在两个训练都完成之后；起跑前核 8 卡空闲；上传包装器 dry-run 通过 |

写入子代理用 opus，各自开 worktree：S1、S2、S4 以本分支 HEAD 为基线，S3 以 c3876b9f 为基线。commit subject 前缀为 `sub/S<n>: `。每份合入前派 sonnet 审查，钉死 BASE/TIP，判定行 `PRE_MERGE_REVIEW=PASS`。主会话按 S1→S2→S4 的顺序 `--no-ff` 合入本分支；S3 合入评估分支 `v20-evalstack-fs4x4`。每次合入后跑定向测试、做合并后审查，然后 push。

## 二、runbook

1. 主会话：在 `docs/handoff/current.json` 登记 `4t-fs4x4-50k` 并写 `kind: go`，提交并 push。
2. 主会话从 c3876b9f 建评估分支 `v20-evalstack-fs4x4` 并 push。并行派 S1–S4。同时在主检出（只读脚本、写数据目录）起 tmux `fs4-fetch` 拉回 4×4 表：
   `HF_HOME=/scratch/hongze/.cache/huggingface-hongzefu /scratch/hongze/MotionJEPA/.venv/bin/hf buckets sync hf://buckets/HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1/framesamp $DS/framesamp-4x4`
   日志写到 `v1-store/logs/fs4x4-fetch.log`，挂 Monitor。目标目录预先确认不存在，存在就停下。拉回后在 S2 合入前不动这份数据。
3. S1–S4 合入后跑 G0；用评估分支合入后的提交建评估 worktree `/scratch/hongze/worktrees/fs4x4-ev-<sha>`，执行 `uv sync --frozen`，`venvs/wan` 等共用环境不动；跑 G1、G2（`check_framesamp_4x4_fetch.py --store $DS/framesamp-4x4 --ref8 $DS/framesamp-8x8 --probes $DS/source/features --manifest-json $DS/bucket-top/upload_manifest.json --procs <按核数留余量>`）。
4. G3：`check_4task_fs4x4_launch.py --budget 512` 和 `--budget 1024`。
5. 写 `docs/training-doc/v2-1600ep-m32x4x4-modul-b128-50k/launch.md` 和 `…/m64x4x4…/launch.md`，打 `commitV20.xBeta`，push，确认工作区干净。
6. 起跑前 `git worktree list`、`tmux ls`、`nvidia-smi` 确认主检出与 GPU 没有被其他会话占用。主检出 `git checkout --detach <Beta>`，执行子模块更新和 `uv sync --frozen`。起 tmux `fs4-chain`（`bash scripts/training/chain_4task_fs4x4.sh <Beta> <评估 worktree>`），依次跑 G4(A)、G4(B)、G5、训练 A、训练 B、3 对评估、三 seed 汇总。训练 A 起跑后起 `fs4-incA`，训练 B 起跑后起 `fs4-incB`。
7. 监控：每份日志挂一个 Monitor，过滤词 `拒绝|失败|FAIL|错误|找不到|Traceback|out of memory|EXIT_CODE=|PASS`，加上 runner 和链脚本里所有 `echo` 文案；另用 CronCreate 每小时挂一次巡检（避开 :00/:30），输出 `HEALTH=OK|ALERT`，检查 tmux 存活、日志久无更新、训练期 GPU 空转、非 0 退出码。
8. 每个训练结束：写训练 result.md，打 commit 并 push。全部评估结束：写评估 README、三 seed 汇总和 final.md，打正式 commit 并 push；状态机改为 `done`；长尾读回完成后改为 `closed`，CronDelete。

## 三、风险

- 拉回带宽慢或中断：`hf buckets sync` 可重入；超过 2 小时仍未完成就报告用户，不自动改成重抽。
- XGRID 个别行余弦低于阈值：说明行没对齐或数据损坏，停下报告，不放宽阈值。
- B 的 64 帧 4×4 在 8 卡上的显存与步速：8×8 的 1024 档（同样 1024 token）跑过，风险低；G4 实测。
- 评估脚本关闭 motion 的路径在本机还是第一次用：G5 先验证。
- 3 对评估约 3.5 h；某次出现新 error 时按脚本规则只重跑该局一次，不整次重评。配对中一次失败不影响同对另一次，失败那次单独补跑。
- 在跑期间有其他会话要卡：按跨会话约定答复，不让出正在训练或评估的卡。

## 四、盲区

- 桶里 4×4 表的 `meta/store_meta.json` 记录的 `source_dataset_root` 和 `manifest_path` 是 9 月的绝对路径（与当前库根同名）。runner 已用 `MMEVLA_FRAMESAMP_SOURCE/MANIFEST` 显式覆盖，G4 再核对训练记录的 env.json 里解析出的路径。
- v42 评估栈的评估 server 用的是评估 worktree 里的模型代码；训练用的是 v18 基线代码。两边的 framesample 读帧与 modulation 实现是否逐位一致，靠 G5 加载冒烟和 `check_checkpoint` 的快照核对兜底（Table I 的 8×8 行也是这种组合）。
- 4 卡并行方案下 1024 档的显存没有实测（只影响备选方案）。
- Table I 的 seed 7 来自 GL、前缀库与本机有 40 局 BinFill 前缀不同（t1s-3seed-1120 已记录，用户确认不处理）；本方案三 seed 都用本机前缀库。
