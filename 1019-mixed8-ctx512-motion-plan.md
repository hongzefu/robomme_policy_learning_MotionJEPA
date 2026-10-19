# mixed8-200 8 帧 8×8 context 512 + motion 160（全部走 context）GPU 0–3 80k 训练计划

> 创建：2026-10-19 美东 00:40。权威工作副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，环境 B（AWS p4de，8×A100-80GB），分支 `v2-motionmem`，起草时 HEAD `f6a57ff`（代码最后一次改动在 `c27514a` commitV11.27；工作区 clean）。
>
> **当前状态：只是计划，未改代码、未起任何短测或训练。** 开工令按 AGENTS 第 2 条由用户对本文件不改而直接认可给出；开工后、动手前先派一个 sonnet 只读审查计划自洽性，`PLAN_CONSISTENCY_REVIEW=PASS … blocking=0` 才动手。
>
> **计划格式**：用户本轮明确要求「直接用 plan 模式的 md 不用再写 html 但是要留在根目录」，按用户当前指令覆盖 AGENTS 第 2 条「根目录计划一律 HTML」，与 `1018-mixed8-2048-context-modul-plan.md` 的先例相同。
>
> **与 `1018-ctx-frame-modul-motion-plan.html` 的关系**：那份「帧走 context、motion 走 modulation」的分路计划理解错了用户意图，**作废、不实施**（用户 2026-10-19：「不是！」「我的意思是全部走conetxxt」）。本计划不改任何模型代码，分路计划里的 `motion.integration` 键、S1/S2 模型改动、V11.29/V11.30 编号预留全部取消，编号由本计划接用。
>
> **commit 编号**：代码合并 `commitV11.29`；起跑前档案 `commitV11.30Beta`；训练结束回写 `commitV11.30`。（V11.28 是在跑的 modulation 2048 的 Beta，其正式回写归 `mixed8-2048-train` 的 owner，本计划不碰。）

用户原话（按时间顺序）：

1. 「gpu0-3跑context 512+motion 写计划 直接用plan模式的md 不用再写html 但是要留在根目录」
2. 「不是！」「我的意思是全部走conetxxt」「现在已经实现过了」（针对分路理解的纠正）

# 第一部分（给人看）

## 1. 总览与已定口径

**一句话方案**：在上一轮 mixed8-200 两条 512 run（commitV11.24 纯 512 modulation、commitV11.25 512+motion modulation）的同一套库、超参、卡数上，把**帧记忆 512 与 motion 记忆 160 都拼进 PaliGemma 主干前缀（context，宽 2048）**，在 GPU 0–3 上训一条 80k。这条链路代码已经存在（0928 在 4 任务库上原入口跑通 401 步并完整恢复），本轮只新增一份指向 mixed8 库的 yaml、一个 runner 和一套上传驱动，**不改任何已跟踪文件**。

| # | 已定口径 | 依据 |
|---|---|---|
| 1 | 接入方式：顶层 `integration_type: context`、`memory_token_dim: 2048`；motion 节与 V11.25 的 mixed8 版逐字相同（enabled、dim 768、budget 160、demo 17 帧 repeat_last、stride 16、window 33、forward、segment_start、mixed8 motion 库与 `wan-mixed8-200-filter2-stop075-b88x8-72ep-a/checkpoint_epoch_72.pt#encoder`） | §4、第二部分 §1 |
| 2 | 新 yaml `perceptual-framesamp-context-8frame-8x8-motion-mixed8.yaml` = 复制 `perceptual-framesamp-modul-8frame-8x8-motion-mixed8.yaml`（sha256 `213bbe80…`），**只改两行** `integration_type`、`memory_token_dim`，与 0928 诊断 yaml `history-context-motion160.yaml` 的派生方式相同 | 0928 [result.md](docs/training-doc/ctx-stock-scaling-20260928/result.md)「历史配置交叉核对」 |
| 3 | 训练配置不变：`mme_vla_suite_b128_80k`（80000 步、global batch 128、每卡 32、warmup 5000、lr 5e-5、AdamW clip 1.0、EMA 0.999、每 5000 步保存、seed 42、`pi05_base` 初始化），`FSDP_DEVICES=4 NUM_WORKERS=8`，与 V11.24/V11.25 相同 | V11.25 [launch.md](docs/training-doc/mixed8-200-m8x8-modul-motion-b128-80k/launch.md) |
| 4 | 数据：`v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/framesamp-8x8`（3200 集、2145548 帧、1383846 执行样本）+ 同目录 `motion/`（128342 行，store_meta `c9f8275f…`），norm stats `36ee9acd…`，与 V11.25 同一份 | 同上 |
| 5 | GPU 0–3（用户指定）；GPU 4–7 是在跑的 `m2k-train-modul`（modulation 2048，美东 10-19 00:19 在 12.6k/80k，1.8 s/步，约剩 34 h），不得占用 | `nvidia-smi`、`v1-store/logs/mixed8-200-m32x8x8-modul-b128-80k.log` |
| 6 | run 名 `mixed8-200-m8x8-context-motion-b128-80k`（沿用 0928 原拟名 `v2-1600ep-m8x8-context-motion-b128-80k` 的体例）；公开 bucket `HongzeFu/robomme-vla-mixed8-200-context-motion-80k-v1`，训练中增量上传，结束后正式驱动全量回读 | §6 Q1 |
| 7 | 不能热启动：59 叶参数树（`encoder_static` (2816,2048)、`motion_encoder_static` (1536,2048)）与 V11.25 的 65 叶不同形，从 `pi05_base` 冷启动 | 0928 result.md「59叶12622947936字节」 |
| 8 | 失败处置沿用 2048 计划裁决 D：OOM 即停下报错，不自行换卡数或改 mask；步时慢不停；80k 中途崩溃停下问，不 resume | `1018-mixed8-2048-context-modul-plan.md` §6 D |
| 9 | tmux 前缀 `cmo-`；不动 `m2k-train-modul`、`m2k-inc-modul` 及其他任何会话 | 第二部分 R7 |
| 10 | 评估另立计划；本计划只训练与上传 | — |

## 2. 现状核查（2026-10-19 美东 00:20–00:35，主会话只读）

| 项 | 结论 | 依据 |
|---|---|---|
| 模型侧 | **不用改**。`PerceptualMemory` 的 `motion_encoder_static` 输出维跟随 `memory_token_dim`（context 即 2048）；`HistoryPi0.embed_memory` 把帧 512 + motion 160 concat 成 672 位，按 `mem_order` (B,672) 交错；`HistoryPi0.__init__` 只拒 `expert` 接 motion | `percep_mem.py::PerceptualMemory`、`history_pi0.py::HistoryPi0.__init__ / embed_memory / _motion_specs` |
| 数据侧守卫 | **放行**。`FrameSampDataset.__init__` 的 `(integration_type, memory_token_dim)` 白名单含 `("context",2048)`；形制 `(512,64,1)` 走 `legacy_shape`，不检查接入方式与 motion | `framesamp_dataset.py::FrameSampDataset.__init__` |
| 代码漂移 | 0928 原入口快照 `6db0e0a` 到 HEAD，`src/mme_vla_suite/{models,training,policies}` 只多两份 yaml 和守卫里附加的 `ctx_shape` 分支，模型代码一字未动 → 0928 的结构、显存、恢复证据适用于同一代码路径 | `git diff --stat 6db0e0a..HEAD -- src/mme_vla_suite/models src/mme_vla_suite/training src/mme_vla_suite/policies` |
| 只读锁 | 2048 计划 R4 的锁只锁**已跟踪文件**（如 `history_pi0.py` 为 `r--r--r--`），目录 `src/…/config/robomme`、`scripts/dataset/hf_export`、`mixed8_200_runner` 均 `drwxr-xr-x` → 本计划只新增文件，合并不需要临时解锁 | `ls -l` / `ls -ld` |
| 资源 | 主机内存可用 992 GiB；`/scratch` 可用 3.8 T（本 run 16 份 ckpt 约 203 GB + 回读副本同量） | `free -g`、`df -h` |
| 可复用件 | runner 共用体 `_train_common.sh`（含 mixed8 库、assets、`FSDP_DEVICES`/`NUM_WORKERS` 透传、`DRY_RUN=1`）；上传共用体 `_mixed8_200_vla_ckpt_export_common.sh`；增量上传 `v1-store/m8v-scripts/m8v-ckpt-inc.sh`；起跑前自检 `scripts/training/preflight_train_launch.py`（`PREFLIGHT=PASS n=<k>`，带四个 `--motion-*`）；短测汇总 `scripts/training/tests/check_32frame_context.py smoke-summary`（只解析日志与 NVML csv，不读 yaml） | 各文件头注释与 `m8v-preflight-motion.sh` |

## 3. 最大风险：步时与工期（0928 有原入口实测，库不同）

- 0928 原入口实测（4 任务 1600ep 库，同为 context 512+motion160、batch 128，**两条 4 卡同时跑**，与本轮 GPU 4–7 有 2048 在跑的情形相同）：四卡 **3.845 / 3.863 s/步**，主窗口采样显存峰值 **71965–71985 MiB**，单卡利用率 99.8%；八卡 1.979 s/步、约 41.8 GiB/卡。当时 `num_workers=16`，本轮按 V11.25 用 8。
- 外推本轮 80k ≈ 3.85 s × 80000 ≈ **85.5 h（约 3.6 天）**；若美东 10-19 白天起跑，约美东 10-22 晚间结束。mixed8 库样本更多，但每步计算形状相同，步时以 G4 短测实测为准。
- 对照（只告知、不改用户决定）：等 2048 结束（约美东 10-20 上午）再用 8 卡跑，按 1.98 s/步约 44 h，总完工时间约 78 h，与现在 4 卡起跑差不多；4 卡方案的好处是马上开始、与 V11.24/V11.25 同为 4 卡可直接比。2048 结束后 GPU 4–7 空出，**不自动挪用**。
- 显存峰值约 72 GiB，余约 8 GiB；context 前缀含 672 位记忆，`num_workers` 只影响主机内存，不影响显存。若 OOM 按口径 8 停下。

## 4. 机制：数据链路（与 V11.25 只差最后两跳）

| 跳 | V11.25（modulation 同路） | 本轮（context 同路） | 这一跳有没有改数 |
|---|---|---|---|
| 数据侧八键 | `static_image_emb` (512,2048) bf16、`static_pos_emb` (512,768) f32、`static_state_emb` (512,8)、`static_mask` (512,)、`motion_emb` (160,768) f32、`motion_pos` (160,256) f32、`motion_mask` (160,)、`mem_order` (672,) int32；约 4.36 MB/样本 | 同左，一个字节不变 | 否 |
| 帧路编码 | `concat(img 2048, silu(pos_proj) 768)` → `encoder_static` 2816→**1024** | 2816→**2048** | 输出宽变 |
| motion 路编码 | `concat(motion_emb 768, silu(motion_pos_proj 256→768))` → `motion_encoder_static` 1536→**1024** | 1536→**2048** | 输出宽变 |
| 合并与重排 | concat (b,672,1024)，按 `mem_order` 交错 | concat (b,672,2048)，同样交错 | 只变宽 |
| 进模型 | 不进前缀；作为 `mem_seq` 进动作专家 18 层 `mem_attn` + `mem_rms_norm_ffn` | 拼进主干前缀最前 `[记忆 672 \| 图像 \| 文本]`，`make_attn_mask` 三参数：图像不看记忆，文本与动作 token 看记忆；不建 `mem_attn` | 是：接入点不同 |
| 参数树 | 65 叶 | 59 叶（23 冻结 bf16 + 32 可训练 + 4 motion），12622947936 字节 | 不能热启动 |

推理侧不改：`policy.py::MME_VLA_Policy._prepare_history` 与 `framesamp_memory.py::FrameSampMemory._prepare_motion` 产出同样八键，策略层没有 `integration_type` 分支；在线 motion 由 sidecar 编码（评估另立计划）。

## 5. 步骤总览

| 步 | 做什么 | 谁做 | 判定 | 预计 |
|---|---|---|---|---|
| 1 | 写入 5 个新文件：新 yaml、runner `train-ctx-motion-80k.sh`、上传驱动、hygiene allowlist、bucket README | 写入子代理 S1（opus，worktree） | worktree 内 `S1_STATIC=PASS`（见 §7） | 20 min |
| 2 | 合并前审查 → `--no-ff` 合并（`commitV11.29`）→ 合并后审查 → push；新增的 5 个文件随即 `chmod a-w` 并登记 | 审查 RV1（sonnet）；主会话合并 | `PRE_MERGE_REVIEW=PASS`、`POST_MERGE_REVIEW=PASS`、`LOCK_ADDED=OK files=5` | 20 min |
| 3 | 主检出 CPU 闸门：G1 runner 干跑、G2 yaml 两行差异、G3 dataset 构造 | 主会话 | `DRY_RUN_CMO=PASS`、`YAML_DIFF=PASS lines=2`、`DATASET_BUILD=PASS` | 10 min |
| 4 | G4：GPU 0–3 20 步短测（`XLA_PYTHON_CLIENT_PREALLOCATE=false`，NVML 500 ms 采样，只查 GPU 0–3） | 运行子代理 X1（opus，主检出） | `SMOKE20=PASS oom=0 loss_finite=1 …` | 15–25 min |
| 5 | 打 `commitV11.30Beta`（`docs/training-doc/<run>/launch.md`，只动 `docs/`）→ G5 起跑前自检 `PREFLIGHT=PASS` | 主会话 | `BETA_CLEAN=PASS`、`PREFLIGHT=PASS n=<k>` | 15 min |
| 6 | 起跑 80k（`cmo-train`）+ 增量上传 `cmo-inc` | X1 起跑；主会话挂 Monitor、起上传 | G6：300 步内无 `Traceback`/`RESOURCE_EXHAUSTED`、loss 有限 | 约 85 h（外推） |
| 7 | 收尾：核 `EXIT_CODE=0`、16 份 ckpt → `INC_END` 后正式驱动 `SMOKE=1` 再全量 → `result.md` 回写 `commitV11.30` → 权限恢复（见 Q3）→ 状态机 done | 主会话 | G7 `TRAIN_DONE=PASS`、G8 `SHA256_PRE_POST_DIFF=0` | 数小时 |

**开工令覆盖范围**：一次开工令覆盖步骤 1–7；G4、G5 通过后直接打 Beta 起跑，不再二次请示；只有闸门 FAIL（含 OOM）才停下报错。

## 6. 用户待裁决项

| 编号 | 问题 | 推荐 |
|---|---|---|
| Q1 | run 名 / bucket 名 | 按口径 6：`mixed8-200-m8x8-context-motion-b128-80k`、`HongzeFu/robomme-vla-mixed8-200-context-motion-80k-v1`（公开） |
| Q2 | 是否登记进状态机 `docs/handoff/current.json` | 登记，任务 id `mixed8-ctxmotion-train`，状态 `planned` |
| Q3 | 只读锁交接：2048 计划收尾时要按 `m2k-chmod-before.txt` 恢复权限，而本 run 会比 2048 晚结束约 2 天 | 2048 收尾时**不恢复**，改由本 run 结束后按 `m2k-chmod-before.txt` + `cmo-chmod-added.txt` 一次恢复；需通知 `mixed8-2048-train` 的 owner 会话 |
| Q4 | 作废的 `1018-ctx-frame-modul-motion-plan.html` 怎么处理 | 文件头加「已作废」标注保留（`docs:` 提交），不删 |

## 7. 子代理分工与合并（简述）

改动只有 5 个新文件、互相独立又很小，不再拆两块：一个 Opus 写入子代理 S1 在独立 worktree 里写完、自检、提交；一个 Sonnet 审查者 RV1 钉死 BASE/TIP 看 diff，主会话 `--no-ff` 合入后 RV1 再审一次合并结果，然后 push。短测与正式起跑由一个 Opus 运行子代理 X1 在主检出执行，只跑已合并的脚本、核起跑、交回 tmux 名，不改任何文件；CPU 闸门、Beta、起跑前自检、监听、上传与收尾由主会话负责。

## 8. 验收表

| 判定 | 查什么 / 怎么查 | 过了说明什么 |
|---|---|---|
| `S1_STATIC=PASS yaml_diff=2 bash_n=2 json=1 stale=0` | 新 yaml 对 `…-modul-8frame-8x8-motion-mixed8.yaml` 的 diff 恰为两行；runner 与驱动 `bash -n`；allowlist `json.load`；5 个新文件 `grep -c 'modul-motion\|modul_motion\|vla_motion_80k'` 为 0 | 新文件只改了该改的、没有残留旧 run 名 |
| `DRY_RUN_CMO=PASS` | `DRY_RUN=1 FSDP_DEVICES=4 NUM_WORKERS=8` 打印的 `HC=`、`--dataset-path …/framesamp-8x8`、`--fsdp-devices 4 --num-workers 8` | 起跑命令与 V11.25 只差 `HC` |
| `DATASET_BUILD=PASS samples=1383846 motion=1` | 用新 yaml 实际构造 `FrameSampDataset`（CPU），取首尾各 2 个样本核八键形状 | 守卫放行、motion 库身份校验通过 |
| `SMOKE20=PASS oom=0 loss_finite=1 peak_mib=<n> steady_s_per_step=<x>` | 20 步日志 + NVML csv 交给 `check_32frame_context.py smoke-summary --gpus 0,1,2,3` | 能起跑；峰值与步时只记录、不设门槛 |
| `PREFLIGHT=PASS n=<k>` | `preflight_train_launch.py` 钉 Beta sha、yaml sha、norm sha、motion 库四项 | 起跑时 HEAD、配置、数据身份全部钉死 |
| `TRAIN_DONE=PASS exit=0 ckpts=16` | 日志末行 `EXIT_CODE=0`，step 5000…75000 与 79999 共 16 份 | 训练完整 |
| `SHA256_PRE_POST_DIFF=0`、`BUCKET_PUBLIC=True` | 正式驱动全量回读、上传前后 sha256 逐行对照 | 上传完成（`INC_END` 不算） |

第二部分是逐文件清单、分配表、闸门、runbook、风险、盲区与交接登记。

# 第二部分（技术细节，供 agent 追踪）

## 〇 前置声明与红线

- **R1** 只新增第二部分 §1 的 5 个文件；不改任何已跟踪文件（含 `_train_common.sh`、`_mixed8_200_vla_ckpt_export_common.sh`、`test_*.py`、`training/config.py::_CONFIGS`）。
- **R2** 新 yaml 相对 `perceptual-framesamp-modul-8frame-8x8-motion-mixed8.yaml` 只允许两行差异（`integration_type: context`、`memory_token_dim: 2048`）加头注释；motion 节一个键都不动。
- **R3** run 名全新；`_train_common.sh` 的「拒绝覆盖日志」保留；短测 run 名带 `cmo-smoke-<UTC>`，结束即删其 ckpt 目录。
- **R4** 正式训练从 clean HEAD（`commitV11.30Beta`）起跑；训练期间沿用 2048 计划 R4 的只读锁，新增 5 个文件合并后同样 `chmod a-w`，原权限登记 `v1-store/m8v-scripts/cmo-chmod-added.txt`；恢复时点按 Q3 裁决。
- **R5** 80k 中途崩溃：停下问用户，不 resume、不改参数。
- **R6** OOM：停下报错，不换卡数、不改 mask、不改 workers；步时慢不停。
- **R7** tmux 前缀 `cmo-`；只按本轮清单逐个 `tmux kill-session -t '=名'` 清理，删前删后各 `tmux ls`；`m2k-*` 及其他会话一律不动。
- **R8** 只用 GPU 0–3：所有本轮命令显式 `CUDA_VISIBLE_DEVICES=0,1,2,3`；NVML 采样只查 `-i 0,1,2,3`。
- **R9** 长期文档不写代码行号，用符号锚点。

## 1. 逐文件改动清单（S1，全部新建）

| 文件 | 内容 |
|---|---|
| `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-context-8frame-8x8-motion-mixed8.yaml` | 复制 `perceptual-framesamp-modul-8frame-8x8-motion-mixed8.yaml`，改 `integration_type: context`、`memory_token_dim: 2048`；头注释写「context 8×8 开启 motion（mixed8 库）：帧 512 与 motion 160 都进主干前缀；相对 modul 版只改接入方式与输出宽；派生方式同 0928 `history-context-motion160.yaml`」 |
| `scripts/dataset/hf_export/mixed8_200_runner/train-ctx-motion-80k.sh` | 照 `train-motion-80k.sh`，`HC=perceptual-framesamp-context-8frame-8x8-motion-mixed8.yaml`，头注释改 run 名与接入方式 |
| `scripts/dataset/hf_export/run_mixed8_200_vla_ctx_motion_80k_ckpt_export.sh` | 照 `run_mixed8_200_vla_motion_80k_ckpt_export.sh`：`RUN_NAME=mixed8-200-m8x8-context-motion-b128-80k`、`BUCKET_ID=HongzeFu/robomme-vla-mixed8-200-context-motion-80k-v1`、`EXPORT_NAME=hf-ckpt-mixed8-200-vla-ctx-motion-80k`、`README_SRC`/`ALLOWLIST`/`DRIVER_NAME` 指新文件，`EXPECT_MOTION=true` 保留 |
| `scripts/dataset/hf_export/hygiene_allowlist_mixed8_200_vla_ctx_motion_80k.json` | 照 motion 版全文替换 run 名、bucket 名、驱动名（含 `_doc` 字段） |
| `scripts/dataset/hf_export/mixed8_200_vla_ctx_motion_80k_bucket_README.md` | 照 motion 版改 run 名、bucket 名、驱动名、yaml 名，写明「context 同路：帧 512 + motion 160 进主干前缀」；sha 与字节数沿用 `@@…@@` 占位由驱动填 |

不新增测试文件：守卫与模型都不改，`DATASET_BUILD` 与 `SMOKE20` 在真实入口上覆盖新 yaml。

## 2. 子代理分配表

| 编号 | 目标 | 模型 / 隔离 | 可写集合 | 禁触 | 依赖与合并顺序 | 验收命令与判定行 | 资源 |
|---|---|---|---|---|---|---|---|
| S1 | §1 的 5 个新文件 | `opus`，`isolation: worktree`，基线 = 开工时 HEAD | §1 表 5 个文件 | 其他一切；主检出任何路径 | 无依赖；唯一一次合并 | worktree 内：`diff` 新旧 yaml 恰 2 行；`bash -n` 两个 sh；`uv run --no-project python -c "import json;json.load(open(...))"`；残名 grep 为 0 → `S1_STATIC=PASS yaml_diff=2 bash_n=2 json=1 stale=0` | CPU |
| RV1 | 合并前、合并后各审一次 | `sonnet`，只读 | 无 | 全部 | 钉死 `REVIEW_BASE`/`REVIEW_TIP` | `PRE_MERGE_REVIEW=PASS\|FAIL base=… tip=… files=5 commits=<n> findings=<n> blocking=<n>`；`POST_MERGE_REVIEW=PASS\|FAIL merge=… files=5 blocking=<n>` | — |
| X1 | G4 短测、正式起跑 | `opus`，主检出，不加 worktree | 无（只写 `v1-store/` 日志与 run 产物） | 任何代码、配置、脚本、参数 | 合并 push、G1–G3 PASS 后跑 G4；正式起跑在 Beta 与 `PREFLIGHT=PASS` 之后 | §3 G4、G6 判定行 + `tmux ls` 原文 | GPU 0–3；tmux `cmo-smoke`/`cmo-smoke-nvml`/`cmo-train` |
| 主会话 | G1–G3、G5、Beta、锁登记、增量与正式上传、Monitor、收尾、handoff | — | `docs/training-doc/mixed8-200-m8x8-context-motion-b128-80k/`、`docs/handoff/current.json`、`v1-store/m8v-scripts/cmo-*` | — | — | — | tmux `cmo-inc`/`cmo-upload-smoke`/`cmo-upload` |

S1 commit subject 前缀 `sub/S1: `，不 push、不合主分支。`docs/handoff/current.json` 只由主会话写。

## 3. 闸门总表

| 编号 | 内容 | 判定行 | 时机 |
|---|---|---|---|
| G1 | runner 干跑 | `DRY_RUN_CMO=PASS`（`HC=perceptual-framesamp-context-8frame-8x8-motion-mixed8.yaml`、`--dataset-path …/framesamp-8x8`、`--fsdp-devices 4 --num-workers 8`） | 合并后 |
| G2 | yaml 差异 | `YAML_DIFF=PASS lines=2`（`diff <(grep -v '^#' 新) <(grep -v '^#' modul 版)`） | 合并后 |
| G3 | dataset 构造（CPU） | `DATASET_BUILD=PASS samples=1383846 motion=1` | 合并后 |
| G4 | 20 步短测（GPU 0–3） | `SMOKE20=PASS oom=0 loss_finite=1 peak_mib=<n> steady_s_per_step=<x>` | G1–G3 后 |
| G5 | 起跑前自检 | `BETA_CLEAN=PASS head=<V11.30Beta sha>`；`PREFLIGHT=PASS n=<k>` | Beta 后 |
| G6 | 起跑 | 300 步内无 `Traceback`/`RESOURCE_EXHAUSTED`，loss 有限 | 起跑后 |
| G7 | 训练结束 | `TRAIN_DONE=PASS exit=0 ckpts=16` | 训练末 |
| G8 | 上传 | `SHA256_PRE_POST_DIFF=0`、`BUCKET_PUBLIC=True`、`PUBLIC_ANON_READ=OK` | 收尾 |

任一 FAIL（含 OOM）即停，原文交用户；步时慢不算 FAIL。

## 4. runbook

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
R=scripts/dataset/hf_export/mixed8_200_runner
C=src/mme_vla_suite/models/config/robomme
# G1
DRY_RUN=1 FSDP_DEVICES=4 NUM_WORKERS=8 bash $R/train-ctx-motion-80k.sh mixed8-200-m8x8-context-motion-b128-80k
# G2
diff <(grep -v '^ *#' $C/perceptual-framesamp-context-8frame-8x8-motion-mixed8.yaml) <(grep -v '^ *#' $C/perceptual-framesamp-modul-8frame-8x8-motion-mixed8.yaml)
# G3：主会话一次性 CPU 脚本（放 $CLAUDE_JOB_DIR/tmp，不进 git），环境同 _train_common.sh 的 MMEVLA_FRAMESAMP_SOURCE/MANIFEST，
#     JAX_PLATFORMS=cpu，构造 FrameSampDataset(新 yaml, motion_root=…/motion)，打印 len 与首尾样本八键形状
# G4：短测（tmux，GPU 0–3）
tmux new-session -d -s cmo-smoke "cd $PWD && CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_PREALLOCATE=false FSDP_DEVICES=4 NUM_WORKERS=8 bash $R/train-ctx-motion-80k.sh cmo-smoke-<UTC> --num-train-steps 20 --no-wandb-enabled"
tmux new-session -d -s cmo-smoke-nvml "nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used,utilization.gpu --format=csv -lms 500 > v1-store/logs/cmo-smoke-nvml-<UTC>.csv"
uv run --no-sync python scripts/training/tests/check_32frame_context.py smoke-summary --log v1-store/logs/cmo-smoke-<UTC>.log --nvml v1-store/logs/cmo-smoke-nvml-<UTC>.csv --gpus 0,1,2,3
#   之后：kill cmo-smoke-nvml（精确名，删前删后 tmux ls）；删 v1-store/train-runs/mme_vla_suite_b128_80k/cmo-smoke-<UTC>
# 步骤 2 末：新增 5 个文件登记原权限后上锁
stat -c '%a %n' <5 个新文件> > v1-store/m8v-scripts/cmo-chmod-added.txt; chmod a-w <5 个新文件>
# 步骤 5：commitV11.30Beta（只动 docs/）→ G5
#   v1-store/m8v-scripts/cmo-preflight.sh 照 m8v-preflight-motion.sh：--train-head <V11.30Beta 全 sha>、
#   --history-config perceptual-framesamp-context-8frame-8x8-motion-mixed8.yaml --history-config-sha256 <新 yaml sha>、
#   --norm-stats-sha256 36ee9acd…、--motion-store …/motion --motion-store-meta-sha256 c9f8275f… --motion-layout motion-768-grid16-demopad17-v1 --motion-rows 128342、
#   --run-root …/mixed8-200-m8x8-context-motion-b128-80k、-- 后接与 runner 相同的 train.py 参数
# 步骤 6：起跑
tmux new-session -d -s cmo-train "cd $PWD && CUDA_VISIBLE_DEVICES=0,1,2,3 FSDP_DEVICES=4 NUM_WORKERS=8 bash $R/train-ctx-motion-80k.sh mixed8-200-m8x8-context-motion-b128-80k"
tmux new-session -d -s cmo-inc "cd $PWD; set -o pipefail; bash v1-store/m8v-scripts/m8v-ckpt-inc.sh mixed8-200-m8x8-context-motion-b128-80k HongzeFu/robomme-vla-mixed8-200-context-motion-80k-v1 hf-ckpt-mixed8-200-vla-ctx-motion-80k scripts/dataset/hf_export/hygiene_allowlist_mixed8_200_vla_ctx_motion_80k.json cmo-train 2>&1 | tee v1-store/logs/cmo-inc.log; echo \"EXIT_CODE=\$?\" >> v1-store/logs/cmo-inc.log"
# 步骤 7：训练 EXIT_CODE=0、cmo-inc 出 INC_END 且会话已退出后
tmux new-session -d -s cmo-upload-smoke "cd $PWD; SMOKE=1 bash scripts/dataset/hf_export/run_mixed8_200_vla_ctx_motion_80k_ckpt_export.sh 2>&1 | tee v1-store/logs/cmo-upload-smoke.log"
tmux new-session -d -s cmo-upload "cd $PWD; bash scripts/dataset/hf_export/run_mixed8_200_vla_ctx_motion_80k_ckpt_export.sh 2>&1 | tee v1-store/logs/cmo-upload.log"
```

`_train_common.sh` 已 tee 日志并写 `EXIT_CODE=`，runbook 不重复。每份日志各挂一个 Monitor：训练日志过滤 `EXIT_CODE=|Traceback|RESOURCE_EXHAUSTED|out of memory|nan|NaN|Killed`；inc 日志过滤 `INC_UPLOADED|INC_END|UPLOAD_FAIL|HYGIENE=FAIL|错误|Traceback|EXIT_CODE=`；正式驱动过滤 `RESULT=|SHA256_PRE_POST_DIFF|全部完成|错误|Traceback|EXIT_CODE=`。

本轮 tmux 会话清单：`cmo-smoke`、`cmo-smoke-nvml`、`cmo-train`、`cmo-inc`、`cmo-upload-smoke`、`cmo-upload`。

## 5. 风险登记

| 编号 | 风险 | 应对 |
|---|---|---|
| K1 | 4 卡约 85 h，比 V11.25（38.6 h）长一倍多 | 用户已定 GPU 0–3；G4 后只报实测工期；2048 结束后 GPU 4–7 空出不挪用 |
| K2 | 显存约 72 GiB/卡（0928 实测），余量约 8 GiB | G4 实测峰值；OOM 按 R6 停 |
| K3 | 与 2048 modulation 同读一份库，IO 与主机内存争用 | 上一轮 4+4 同库无异常；主机内存可用 992 GiB；G4 看步时 |
| K4 | 只读锁恢复时点冲突（Q3） | 按裁决执行；未裁决前 2048 收尾不得恢复本 run 依赖的文件权限 |
| K5 | 增量上传与训练抢 IO | `m8v-ckpt-inc.sh` 已 `ionice -c 3 nice -n 19` |
| K6 | 训练崩溃时公开 bucket 停在部分 step | 按 V11.25 F5 处置：写进 `pending_decisions` 交用户裁决保留或清理 |

## 6. 盲区诚实清单

- 本轮步时与显存是从 0928（4 任务库、workers16）外推，mixed8 库上的原入口实测要等 G4。
- 不做 100 步保存加载核对：依据是 0928 在同一模型代码路径上已做 59 叶原 dtype 完整 CPU 恢复（`12622947936` 字节），且模型代码此后未改；第一份 step 5000 ckpt 的可加载性要到评估计划才实际验证。
- context 512+motion 从未完整训练过 80k；与 V11.25 的效果对比要等评估。
- 推理侧不做在线装配冒烟；评估另立计划。

## 7. 交接顺序与状态机登记

- **任务 id**：`mixed8-ctxmotion-train`（登记与否见 Q2）。
- **上游**：`mixed8-vla-build`（closed；假设 `framesamp-8x8` `VERIFY_PACK=PASS scanned=2145548 mismatches=0`、motion `VERIFY_MOTION=PASS scanned=128342`、`TRAIN_ES_BOUND=PASS max_windows=123 budget=160`）。与 `mixed8-2048-train` 无数据或代码依赖，只共享机器（GPU 4–7）与只读锁（Q3）。
- **交接前必须完成（pre_handoff）**：① `TRAIN_DONE=PASS exit=0 ckpts=16`（`v1-store/logs/mixed8-200-m8x8-context-motion-b128-80k.log`）；② `result.md` 回写并以 `commitV11.30` push；③ 权限按 Q3 裁决恢复 `PERM_RESTORED=OK`。
- **长尾**：`cmo-inc`（`INC_END` 只表示会话结束）；`cmo-upload` 正式驱动（`SHA256_PRE_POST_DIFF=0`、`BUCKET_PUBLIC=True`，这才是上传完成）。
- **下游**：本 run 的在线评估（另立计划）。

## 8. 留档与 commit 纪律

- 计划本身 `docs:`；S1 合并 `commitV11.29: …`（保留 `sub/S1`）；起跑前 `commitV11.30Beta`（建档 `docs/training-doc/mixed8-200-m8x8-context-motion-b128-80k/launch.md`，含 tmux 清单与完整命令）；结束后 `result.md` 回写 `commitV11.30`。
- Q4 若按推荐：作废标注作为单独 `docs:` 提交。
- 一律 `git add <逐个路径>`，commit 后裸 `git push`。
