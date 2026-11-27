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
- 评估入口：v18 的 `scripts/evaluation/local_692_pool.sh`。
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
- A′、B′ 的 seed 7 是 GreatLakes 上的结果，seed 0/42 在本机 AWS 上评；本方案三个 seed 都在本机评。平台差异在表里标注，不归因。
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
| **串行 8 卡（推荐）**：A 训 → B 训 → 6 次评估 | A：50k×约 0.95 秒/步≈**13.2 h**（参照 1122 那条 32 帧×4×4 modulation 8 卡实测 0.971 秒/步，它的记忆更长，有 704 token）；B：50k×约 0.99 秒/步≈**13.8 h**（参照 8×8 的 1024 档 8 卡实测 0.986 秒/步）；合计约 27 h | 每次约 1.8 h（v17 本机 692 局，24 片、每卡 3 份，1 h 47 m），合计约 10.8 h | **约 38 h** | 低：8 卡 FSDP8 b128 是 1122、v11–v18 一直在用的配置；A、B 的记忆 token 数和 8×8 的 512、1024 两档相同 |
| 并行 4+4 卡：A、B 同时训，然后评估 | 4 卡每步约慢一倍（8×8 的 512 档 4 卡实测 2.34 秒/步；按 1.9–2.0 秒/步算），两个 run 同时约 27 h | 同左，约 10.8 h | 约 38 h | 中：1024 档在 4 卡 b128 上没跑过，显存未知；两条训练日志、两套句柄同时盯 |

改成「全部训完再评估」之后，两种方案的完成时间基本一样，串行原来能提前出 A 结果的优势也没有了。仍推荐串行，理由只剩风险：每一步都是验证过的 8 卡配置，监控也只有一条训练线。顺序 A→B。

训练和评估不能叠在同一批卡上：训练每卡预占 0.95 显存，评估每份占 0.2。

## 四、代码改动（开工后只用 CPU，约 1–2 小时）

在本分支的 worktree 里改，不碰主检出的其他分支工作树。

1. **配置**：新增 `perceptual-framesamp-modul-32frame-4x4.yaml`（等于 `perceptual-framesamp-modul.yaml`，只加注释）和 `perceptual-framesamp-modul-64frame-4x4.yaml`（只改 `budget: 1024`）。不改任何已有 yaml。
2. **拉回与校验**：新增 `scripts/dataset/fetch_framesamp_4x4.sh` 和 `scripts/dataset/check_framesamp_4x4_fetch.py`，判定行见第五节 G1/G2。
3. **runner**：新增 `scripts/training/run_4task_fs4x4_50k.sh <TRAIN_HEAD> --budget 512|1024 [--smoke20] [--dry-run]`，以 v18 的 `run_4task_e2eenc_50k.sh` 和 v11 的 runner 为底：`--dataset-path $DS/framesamp-4x4`；`--model.history-config` 按 budget 取上面两份 yaml，并钉死 yaml 的 sha256；`STEPS=50001`；`EXPECT_CKPTS` 为 5000…50000；motion 环境变量全部 unset；`TRAIN_TIMING_STEPS=300`。保留 HEAD 与工作区干净检查，以及「拒绝覆盖本轮日志」。
4. **启动合同**：新增 `scripts/training/check_4task_fs4x4_launch.py`，判定行 `LAUNCH_CONTRACT=PASS`。要求：新 yaml 相对 `perceptual-framesamp-modul-{8,16}frame-8x8.yaml` 只差 `token_per_image`（和注释）；B 相对 A 只差 `budget`；训练配置为 `mme_vla_suite_b128_80k`，`num_train_steps` 覆盖为 50001；帧表 meta 的 layout 为 `framesamp-4x4-v1`、`status=verified`、`num_rows=1192918`、`manifest_sha256=4cd5a170…`；`budget // token_per_image` 等于 32 或 64。
5. **评估**：`scripts/evaluation/local_692_pool.sh` 里写死的 `EXPECT_MOTION_STATS=1`、`EXPECT_DEMO_INJECT=1`、`MMEVLA_MOTION_OVERFLOW` 改成可由环境覆盖，缺省值不变；同时核对 `run_shard.sh`、`check_shard.py`、`merge_shards.py` 在关闭 motion 时的判据（不期待 motion 统计，判定行写 `expect_motion_stats=0`）。`POLICY` 已经可以覆盖，本方案传新 yaml 名。`RUN_NAME` 为 `local692-m{32,64}x4x4-50k-s{7,0,42}`，共 6 个。另加三 seed 汇总脚本，输出与 `table1_3seed.json` 同格式的两行，判定行 `FS4X4_3SEED=PASS rows=2 seeds=3`。
6. **触发链与上传**：新增 `scripts/training/chain_4task_fs4x4.sh`，按第六节串起闸门、训练和评估；新增增量上传包装器 `scripts/dataset/hf_export/run_fs4x4_ckpt_export.sh`（以 v18 的 `run_e2eenc_ckpt_export.sh` 为底，每出一份 ckpt 就上传并读回校验）。
7. **测试**：给 2、4、5 加定向 pytest，包括关闭 motion 时 check_shard 的判据、三 seed 汇总和启动合同的正反例。

分工：写入用 opus 子代理，各自开 worktree：S1 负责 1、3、4，S2 负责 2，S3 负责 5、6。每份合入前派 sonnet 审查，主会话按 `--no-ff` 合入。

## 五、开训前闸门（全部 PASS 才起正式训练）

| 闸门 | 判定行 | 何时 |
|---|---|---|
| G0 单测 | 定向 pytest 全过，且原有 `local_692_pool` 相关用例不变 | 代码合入后 |
| G1 拉回 | `FS4X4_FETCH=PASS files=37 bytes=<总字节> mismatches=0`：37 个文件逐个比对 `upload_manifest.json` 的字节数和 sha256 | 拉回后，CPU |
| G2 帧表自证 | ① `FS4X4_ROWDIGEST=PASS rows=1192918 mismatches=0`：按 `ROW_DIGEST_COVERAGE`（image‖pos‖state 三键原始字节）逐行重算 blake2b-128，与 `meta/row_digests.blake2b.bin` 比对；② `FS4X4_XGRID=PASS rows=1192918 min_cos=<x>`：每行 4×4 图像与本机 8×8 同行做 2×2 平均池化后比对，余弦全部 ≥ 0.9999，任一行不满足即 FAIL；③ `FS4X4_PROBE=PASS frames=8 mismatches=0`：8 个探针 npy 的 `image_emb_4x4`、`pos_emb_4x4` 与表中对应行逐位相等；④ `FS4X4_META=PASS`：layout、status、num_rows、manifest_sha256 四项 | G1 后，CPU，约 20–30 分钟（并行进程池，按核数留余量） |
| G3 启动合同 | `LAUNCH_CONTRACT=PASS`（A、B 各一次） | CPU |
| G4 8 卡 20 步短测 | A、B 各一次 `SMOKE20=PASS save_load=PASS`，并记录显存峰值和首 300 步之外的步速 | GPU，每次约 10 分钟 |
| G5 评估加载冒烟 | `EVAL_LOAD_SMOKE=PASS episodes=1 handshake=PASS share2=PASS`（`local_692_pool.sh --g6b`，用 A 的短测 ckpt，关闭 motion 的判据） | GPU，约 10 分钟 |

## 六、起跑与时间线（串行，全部训完再评估）

触发链 tmux `fs4-chain`，脚本放 `v1-store/jobs/fs4x4-1127/chain.sh`，按序执行：

1. 等 8 卡显存都低于 2000 MiB，否则不起跑并报告用户；
2. G4（A、B 各一次）→ G5；
3. 正式训练 A（`v2-1600ep-m32x4x4-modul-b128-50k`）；
4. 正式训练 B（`v2-1600ep-m64x4x4-modul-b128-50k`）；
5. 两个 run 都有 `EXIT_CODE=0` 且 50000 都在后，依次评估 A@50000 的 seed 7/0/42 和 B@50000 的 seed 7/0/42，共 6 次。每次都是 24 片、每卡 3 份、每份 0.2 显存，用满 8 卡；
6. 三 seed 汇总，得到 `FS4X4_3SEED=PASS`。

任一闸门 FAIL 或任一步非 0 退出即停链。

预计（从开工令算起）：代码与拉回、G0–G3 约 2–3 h；G4–G5 约 0.5 h；训练 A 约 13.2 h、B 约 13.8 h；6 次评估约 10.8 h。**全部结果约在开工后 40–41 h 出来。** 实际步速以 G4 和正式训练前 300 步的计时为准，届时修正本节。

## 七、ckpt 与磁盘

- 现剩 1.6 TB。4×4 表 73 GiB；每个 run 10 份 ckpt，按 8×8 的 512 档每份约 11 GiB 估算，两个 run 约 220 GiB，本地全部保留。6 次评估的视频和分片产物每次约 1 GB。
- 增量上传：每出一份就上传到公开桶 `HongzeFu/robomme-vla-4task-modul-m32x4x4-50k-v1` 和 `HongzeFu/robomme-vla-4task-modul-m64x4x4-50k-v1`（不带 `--private`），逐份读回校验 sha。tmux 分别为 `fs4-incA` 和 `fs4-incB`。
- 磁盘余量低于 200 GB 时停下报告。

## 八、评估与交付

- 4task 测试集 692 局，每局最多 2000 步，ckpt 50000，seed 7/0/42，全部在本机评。
- 交付：A、B 各自三 seed 的成功局数、均值 ± 样本标准差和逐任务表，与 Table I 的 A′（32.03 ± 1.45）、B′（40.90 ± 1.00）以及 8×8 的 2048/4096 两档（44.85 ± 0.96、45.13 ± 1.59）并列，标注评估平台和训练总步数；同时给 A、B 的实测训练步速与评估墙钟。只交结果，不做显著性判读。
- 结果文档：训练写到 `docs/training-doc/v2-1600ep-m{32,64}x4x4-modul-b128-50k/result.md`；评估写到 `docs/training-doc/local692-m{32,64}x4x4-50k-s{7,0,42}/README.md`，逐局 CSV 进 `records/`；三 seed 汇总写到 `docs/training-doc/fs4x4-3seed-1127/final.md` 和 `fs4x4_3seed.json`。

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

## 十、交接顺序与状态机登记

- 任务 id：`4t-fs4x4-50k`，无前置依赖（8 卡当前空闲，tmux 上没有本项目的在跑会话）。
- 交接前必须完成：G0–G5 全部 PASS；两份训练日志都有 `EXIT_CODE=0`；6 次评估都有 `EVAL692=PASS episodes=692`；三 seed 汇总 `FS4X4_3SEED=PASS rows=2 seeds=3`；两个 run 的 50000 都已上传，且读回 `HF_VERIFY=PASS`。
- 长尾：增量上传的读回校验（`fs4-incA`、`fs4-incB`，完成行 `INC_END`）。
- 本任务 `done` 之后：第九节第 5、6 项由用户决定。

# 第二部分（技术细节，供 agent 追踪）

## 〇、红线

1. 不改任何已有 yaml、训练配置、`framesamp-8x8` 表或 `source/`；只新增。新 yaml 相对 8×8 同预算 yaml 只允许 `token_per_image`（和注释）不同。
2. 不删除、不覆盖数据集桶（HF 禁删清单）；拉回只读。本地 ckpt 不删。
3. 不 force push；子代理不 push、不合主分支。
4. tmux 只建以 `fs4-` 为前缀的会话，清理只用 `tmux kill-session -t '=确切名'`，并逐个列清单。
5. 不用 worktree 里的脚本跑主数据：触发链在主检出 `git checkout --detach <Beta 提交>` 之后、执行子模块更新和 `uv sync --frozen` 再起跑（1122 实施中的教训）。
6. 评估一律在两个训练都结束后才起，不在训练间隙插评估。

## 一、子代理分配表

| 编号 | 目标 | 可写文件 | 依赖 | 验收 |
|---|---|---|---|---|
| S1 | 两份 yaml、runner、启动合同 | `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-{32,64}frame-4x4.yaml`、`scripts/training/run_4task_fs4x4_50k.sh`、`scripts/training/check_4task_fs4x4_launch.py`、`scripts/training/tests/test_check_4task_fs4x4_launch.py` | 无 | runner `--dry-run` 对 A、B 打印的 `TRAIN_ARGS` 正确（dataset-path、history-config、`--num_train_steps=50001`）；合同脚本反例（改 lr、改 pool_type、budget 不整除、步数不是 50001）均 FAIL；pytest 全过 |
| S2 | 4×4 表拉回与四项校验 | `scripts/dataset/fetch_framesamp_4x4.sh`、`scripts/dataset/check_framesamp_4x4_fetch.py`、`scripts/dataset/test_check_framesamp_4x4_fetch.py` | 无 | 用 4 行合成小表（含一行故意篡改）验证 ROWDIGEST/XGRID/PROBE 能抓到；校验用进程池并行；按 `upload_manifest.json` 里 `framesamp/` 前缀映射到 `framesamp-4x4/` |
| S3 | 评估关闭 motion 的参数化、三 seed 汇总、触发链、增量上传 | `scripts/evaluation/local_692_pool.sh`、`scripts/evaluation/run_shard.sh`、`scripts/evaluation/check_shard.py`、`scripts/evaluation/merge_shards.py`（仅在关闭 motion 判据需要时改）、`scripts/evaluation/fs4x4_3seed_summary.py`、`scripts/training/chain_4task_fs4x4.sh`、增量上传包装器 `scripts/dataset/hf_export/run_fs4x4_ckpt_export.sh`（以 `run_e2eenc_ckpt_export.sh` 为底）、`scripts/evaluation/test_*` 下对应用例 | S1（run 名与 ckpt 路径由主会话在派发说明里写死） | 缺省（不设环境变量）时行为与改前逐字一致；`DRY_RUN=1 EXPECT_MOTION_STATS=0` 打印 24 片计划；mock 关闭 motion 的分片能通过 check_shard；三 seed 汇总在 mock 数据上输出与 `table1_3seed.json` 同格式；链脚本 `--dry-run` 打印完整步骤，并且评估步骤排在两个训练都完成之后 |

写入子代理用 opus，各自开 worktree，基线为本分支 HEAD，commit subject 前缀为 `sub/S<n>: `。每份合入前派 sonnet 审查，钉死 BASE/TIP，判定行 `PRE_MERGE_REVIEW=PASS`。主会话按 S1→S2→S3 的顺序 `--no-ff` 合入，每次合入后跑定向测试、做合并后审查，然后 push。

## 二、runbook

1. 主会话：在 `docs/handoff/current.json` 登记 `4t-fs4x4-50k` 并写 `kind: go`，提交并 push。
2. 并行派 S1–S3。同时在主检出（只读脚本、写数据目录）起 tmux `fs4-fetch` 拉回 4×4 表：
   `HF_HOME=/scratch/hongze/.cache/huggingface-hongzefu /scratch/hongze/MotionJEPA/.venv/bin/hf buckets sync hf://buckets/HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1/framesamp $DS/framesamp-4x4`
   日志写到 `v1-store/logs/fs4x4-fetch.log`，挂 Monitor。目标目录预先确认不存在，存在就停下。拉回后在 S2 合入前不动这份数据。
3. S1–S3 合入后跑 G0；跑 G1、G2（`check_framesamp_4x4_fetch.py --store $DS/framesamp-4x4 --ref8 $DS/framesamp-8x8 --probes $DS/source/features --manifest-json $DS/bucket-top/upload_manifest.json --procs <按核数留余量>`）。
4. G3：`check_4task_fs4x4_launch.py --budget 512` 和 `--budget 1024`。
5. 写 `docs/training-doc/v2-1600ep-m32x4x4-modul-b128-50k/launch.md` 和 `…/m64x4x4…/launch.md`，打 `commitV20.xBeta`，push，确认工作区干净。
6. 主检出 `git checkout --detach <Beta>`，执行子模块更新和 `uv sync --frozen`。起 tmux `fs4-chain`（`bash v1-store/jobs/fs4x4-1127/chain.sh <Beta>`），依次跑 G4(A)、G4(B)、G5、训练 A、训练 B、6 次评估、三 seed 汇总。训练 A 起跑后起 `fs4-incA`，训练 B 起跑后起 `fs4-incB`。
7. 监控：每份日志挂一个 Monitor，过滤词 `拒绝|失败|FAIL|错误|找不到|Traceback|out of memory|EXIT_CODE=|PASS`，加上 runner 和链脚本里所有 `echo` 文案；另用 CronCreate 每小时挂一次巡检（避开 :00/:30），输出 `HEALTH=OK|ALERT`，检查 tmux 存活、日志久无更新、训练期 GPU 空转、非 0 退出码。
8. 每个训练结束：写训练 result.md，打 commit 并 push。全部评估结束：写评估 README、三 seed 汇总和 final.md，打正式 commit 并 push；状态机改为 `done`；长尾读回完成后改为 `closed`，CronDelete。

## 三、风险

- 拉回带宽慢或中断：`hf buckets sync` 可重入；超过 2 小时仍未完成就报告用户，不自动改成重抽。
- XGRID 个别行余弦低于阈值：说明行没对齐或数据损坏，停下报告，不放宽阈值。
- B 的 64 帧 4×4 在 8 卡上的显存与步速：8×8 的 1024 档（同样 1024 token）跑过，风险低；G4 实测。
- 评估脚本关闭 motion 的路径在本机还是第一次用：G5 先验证。
- 6 次评估串行约 10.8 h；某次出现新 error 时按脚本规则只重跑该局一次，不整次重评。
- 在跑期间有其他会话要卡：按跨会话约定答复，不让出正在训练或评估的卡。

## 四、盲区

- 桶里 4×4 表的 `meta/store_meta.json` 记录的 `source_dataset_root` 和 `manifest_path` 是 9 月的绝对路径（与当前库根同名），训练读表时是否会校验这两个路径，需要 G4 实测。
- 4 卡并行方案下 1024 档的显存没有实测（只影响备选方案）。
- Table I 的 seed 7 来自 GL、前缀库与本机有 40 局 BinFill 前缀不同（t1s-3seed-1120 已记录，用户确认不处理）；本方案三 seed 都用本机前缀库。
