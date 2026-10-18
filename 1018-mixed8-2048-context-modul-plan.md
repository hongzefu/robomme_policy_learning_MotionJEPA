# mixed8-200 32 帧 8×8（2048）context 与 modulation 两条 80k 并行训练计划

> 创建：2026-10-18 美东 16:30。权威工作副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，环境 B，分支 `v2-motionmem`，起草时 HEAD `4fd7271`（工作区 clean，`docs/handoff/current.json` 中 `mixed8-vla-train` 已 closed、无在跑句柄）。
>
> **当前状态：只是计划，未改代码、未起任何短测或训练。** 第 6 节待定项已于 2026-10-18 美东 16:35 全部由用户裁决；开工须等用户开工令。

用户原话（按时间顺序）：

1. 「给出方案 训练2048contetxt和modulation 先用subagent检查一下是否都可以实现 还是4卡分别跑 其他各个配置都都是一样的」
2. 「然后也是边训练边上传。」
3. 「本次不再需要建 HTML 计划，直接建 Markdown 计划。还是写在policy库的根目录」
4. 「都是三十二帧的八乘八。」
5. 「再审查一下，RAM 名和 GPU 名和上传位置都你定，太慢或太慢就不要管了。然后 OOM 是不可能的，应该是都会在的我之前跑过。如果oom就报错。上一轮回读副本删除，状态机登登记。」（「RAM 名」「GPU 名」按语境为 run 名、GPU 分配）

# 第一部分（给人看）

## 1. 总览与已定口径

在上一轮 mixed8-200 两条 512 训练（commitV11.24 / V11.25）的同一套库、同一套超参上，把历史记忆从「8 帧 × 64 token = 512」扩到「32 帧 × 64 token = 2048」，并按记忆接入方式分成两条 run，4+4 卡同时跑：

| 项 | context 2048 | modulation 2048 |
|---|---|---|
| 记忆形制 | 32 帧 × 8×8（每帧 64 token），无 motion | 同左 |
| 接入方式 | `integration_type: context`，记忆 token 拼进主干前缀，`memory_token_dim 2048` | `integration_type: modulation`，`memory_token_dim 1024` |
| history yaml | **新建** `perceptual-framesamp-context-32frame-8x8.yaml` | 现成 `perceptual-framesamp-modul-32frame-8x8.yaml`（不改） |
| run 名（裁决 A） | `mixed8-200-m32x8x8-context-b128-80k` | `mixed8-200-m32x8x8-modul-b128-80k` |
| GPU（裁决 B） | GPU0–3，`FSDP_DEVICES=4 NUM_WORKERS=8` | GPU4–7，同左 |
| 训练配置 | `mme_vla_suite_b128_80k`：80000 步、global batch 128（每卡 32）、warmup 5000、lr 5e-5、AdamW clip 1.0、EMA 0.999、每 5000 步保存、seed 42、`pi05_base` 初始化 | 同左 |
| 数据 | `v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/framesamp-8x8`（16 任务 × 200 集 = 3200 集，2145548 帧，1383846 执行样本），norm stats 同上一轮 | 同左 |
| 上传（裁决 C） | 训练中增量上传到公开 bucket `HongzeFu/robomme-vla-mixed8-200-context-2048-80k-v1`，结束后正式驱动全量回读 | `HongzeFu/robomme-vla-mixed8-200-modul-2048-80k-v1`，同左 |

**除接入方式及其必然附带的 `memory_token_dim` 外，两条 run 的数据、超参、卡数、workers、seed 全部一致，也和上一轮 512 一致（只差 budget）。**

## 2. 子代理核查结论（2026-10-18 美东 16:20–16:28，三路 Haiku 只读）

| 项 | 结论 | 依据 |
|---|---|---|
| modulation 2048 | **可直接训**。现成 yaml 只比 8 帧档多一处 `budget: 2048`、不含数据路径；`FrameSampDataset.__init__` 形制守卫已放行 `(2048,64,1)+modulation+无 motion`；mixed8 库按集存全部帧，选帧 `even_sampling_indices(step, budget//64)` 从全集取 32 帧。缺的只是一个指向 mixed8 库的 runner。旧 `scripts/training/prod/run_modul2048.sh` 等入口写死 4 任务库与 SHA，**不能用** | 守卫 `new_shape` 分支；`framesamp_store` 布局；旧 4 任务 2048 正式档案 `v2-1600ep-m32x8x8-modul-b128-80k` |
| context 2048 | **当前被守卫拒绝，改动小但要解锁**。守卫 `new_shape` 分支要求 `integration_type == "modulation"`；而同一函数前段 `(integration_type, memory_token_dim)` 白名单早已允许 `("context", 2048)`——commitV9.6 加白名单时没同步形制守卫。模型与推理侧都按 `budget` 参数化（`percep_mem`、`history_pi0` 的形状检查、`policy.py` 与 `framesamp_memory` 的 `max_frames`），位置由 mask 前缀和推出，没有写死 512 | 0928 计划与 `docs/training-doc/ctx-stock-scaling-20260928/result.md` 均记「2048 受原 context 读取守卫阻断」 |
| 起跑资源 | 8 卡 A100-80GB 全空、无他人进程；`/scratch` 可用 3.5T；只读锁已恢复；上一轮 `mixed8_200_runner/`、`m8v-ckpt-inc.sh`、`_mixed8_200_vla_ckpt_export_common.sh` 可照搬 | `nvidia-smi`、`df -h`、`m8v-chmod-before.txt` 逐行比对 |

## 3. 最大风险：context 2048 在 4 卡上的显存与速度（均未在原入口实测）

- 9-28 的唯一数据是**候选 mask**（非原入口）下 4 卡、每卡 32 样本、2 步探针：context 2048 每卡峰值 79439–79453 MiB，几乎顶满 80 GiB；同条件 context 512 是 73 GiB。原 mask 在 4 卡的 240 s 探针时限内未完成编译（512 也一样，所以这更像探针时限不够，不是容量结论）。
- context 把 2048 个记忆 token 拼进前缀，前缀从约 1108 变为约 2644，注意力项按平方粗估约 5.7 倍；**整步时间不能外推**，必须实测。
- modulation 2048 在 4 卡上同样没有实测：旧 4 任务库 8 卡 1.031 s/步（22 h 59 min），按 0928 实测「四卡单任务约为八卡 1.88 倍」推算 4 卡约 1.9–2.0 s/步、约 43–45 h，**只是推断**。
- 用户裁决（第 6 节 D）：「OOM 是不可能的……我之前跑过。如果oom就报错」「太慢……就不要管了」。所以短测（第 5 节步骤 4–5）只用来确认能起跑、记录显存与步时，**不以步时设门槛**；若真 OOM，停下报错交用户，不自行换卡数、不改训练计算机制。

## 4. 机制（数据链路，两条 run 共用）

逐跳只改「读多少帧」，不改数：

| 跳 | 上一轮 512 | 本轮 2048 | 这一跳有没有改数 |
|---|---|---|---|
| 库行 `image_emb_8x8` | (64,2048) bf16 / 帧 | 同左 | 否（同一份库） |
| 选帧 | `even_sampling_indices(step, 8)` | `even_sampling_indices(step, 32)` | 只改帧数；`step<31` 时取 `range(step+1)` 并补零 |
| 拼记忆 | (512, 2048) + mask(512) | (2048, 2048) + mask(2048) | 只改长度 |
| 进模型 | modulation：调制动作分支；context：拼进主干前缀 | 同左 | 否 |

短历史补零路径（`_pad` / `pad_times`）在 512 下就存在，mixed8 库里 `step<31` 的执行样本数要在步骤 3 核实（旧 4 任务库「全部满 32 帧」，mixed8 未核）。

## 5. 步骤总览

| 步 | 做什么 | 谁做 | 判定 | 预计 |
|---|---|---|---|---|
| 1 | 写入：放开 context 2048 守卫分支、新建 context 32 帧 yaml、补守卫测试、新增两个 mixed8 runner | 写入子代理 S1（worktree） | 守卫测试全过、两条 `DRY_RUN=1` 打印正确命令 | 30 min |
| 2 | 写入：两条上传驱动、hygiene allowlist、bucket README | 写入子代理 S2（worktree） | `bash -n`、allowlist JSON 可解析、`SMOKE` 干跑 | 30 min |
| 3 | 合并前审查 → 合并 → 合并后审查 → push，记 `commitV11.26`；同时只读核 mixed8 短历史样本数与 context 在线装配（CPU） | 审查 R1；只读 Haiku | `PRE/POST_MERGE_REVIEW=PASS`；`SHORT_HIST=… ` 与 `ONLINE_ASM_CTX2048=PASS` | 30 min |
| 4 | 4+4 同时 20 步短测（`XLA_PYTHON_CLIENT_PREALLOCATE=false`，NVML 500 ms 采样） | 运行子代理 X1 | 两条均无 OOM，读峰值显存与稳态步时 | 15–30 min |
| 5 | context 2048 再做 100 步 + 保存 + `create_trained_policy` 加载（mem_len=2048） | X1 | 保存加载叶相等、动作一致 | 30–40 min |
| 6 | 打 Beta 锚点（V11.27Beta context / V11.28Beta modulation）、建档 `launch.md`，实测步时与推算工期只记录并告知，不再等用户确认 | 主会话 | G4、G5 通过 | 10 min |
| 7 | 两条 80k 起跑 + 两条增量上传会话 | X1 起跑；主会话挂 Monitor | 起跑后 300 步内 loss 有限、无 OOM | context 待测；modul 约 43–45 h（推断） |
| 8 | 收尾：核 `EXIT_CODE=0` → 正式上传驱动全量回读 `SHA256_PRE_POST_DIFF=0` → `result.md` 回写 → handoff closed | 主会话 + 写入子代理 | 同上一轮验收 | 数小时 |

## 6. 用户裁决（2026-10-18 美东 16:35）

- **A. run 名**（用户「你定」）：`mixed8-200-m32x8x8-context-b128-80k` 与 `mixed8-200-m32x8x8-modul-b128-80k`（沿用旧 4 任务 2048 的 `m32x8x8` 体例）。
- **B. GPU 分配**（用户「你定」）：context GPU0–3、modulation GPU4–7。
- **C. bucket**（用户「你定」）：新建两个公开 bucket `HongzeFu/robomme-vla-mixed8-200-context-2048-80k-v1`、`HongzeFu/robomme-vla-mixed8-200-modul-2048-80k-v1`（与上一轮同为公开）。
- **D. 显存与速度**：用户「OOM 是不可能的，应该是都会在的我之前跑过。如果oom就报错」「太慢……就不要管了」。短测不设步时门槛；任一条 OOM 即停下报错交用户，不自行回退。
- **E. 上一轮回读副本**：用户「上一轮回读副本删除」。已于 2026-10-18 美东 16:33 删除 `v1-store/exports/hf-ckpt-mixed8-200-vla-{512,motion}-80k/verify`（各 178G；原 ckpt 仍在 `train-runs`，HF 上传已全量回读 PASS），`/scratch` 可用 3.5T → 3.8T。
- **状态机**：用户「状态机登登记」，登记任务 `mixed8-2048-train`，状态 `planned`。

## 7. 子代理分工与合并（简述）

改动拆成两块：S1 管训练侧（守卫、context yaml、测试、两个 runner），S2 管上传侧（两条导出驱动、allowlist、README），各在独立 worktree、互不碰对方文件。S1 先合、S2 后合，每次合并前由一个 Sonnet 审查者钉死 BASE/TIP 看 diff，合并后再审一次合并结果，然后 push。短测与起跑由一个 Opus 运行子代理在主检出执行，只启动、核起跑、交回 tmux 名，不改任何文件；监听、上传会话和收尾由主会话负责。

## 8. 验收表

| 判定 | 通过条件 |
|---|---|
| 守卫 | context 2048 放行；context 1024/4096、context 2048+motion、多视角、浮点/字符串/布尔类型反例全拒；旧 512 两档与 modulation 1024/2048/4096 接受条件不变 |
| 短测 | 两条 20 步无 OOM、loss 有限；记录每卡 NVML 峰值与稳态步时 |
| context 保存加载 | 100 步 ckpt 加载后叶与保存现场一致、记忆相关叶不同于初态、固定 noise 动作一致、`mem_len=2048` |
| 正式训练 | 两条各 80000 步 `EXIT_CODE=0`，16 份 ckpt（5000…75000、79999） |
| 上传 | 每条 16 份 ckpt 公开、全量回读 `SHA256_PRE_POST_DIFF=0` |

# 第二部分（技术细节，供 agent 追踪）

## 〇 前置声明与红线

- **R1** 只改第 1 节列出的文件；不改训练计算、dataloader 读取逻辑、sampler、collate、优化器、全局默认超参（`training/config.py::_CONFIGS` 不动）。
- **R2** 守卫只新增一个精确分支：`raw == (2048,64,1)`、三元组每项 `type(v) is int`、`integration_type == "context"`、`memory_token_dim == 2048`、motion 关闭。不放行 context 1024/4096/8192、不放行 context+motion、不放行 4×4。
- **R3** run 名全新、不覆盖；`_train_common.sh` 的「拒绝覆盖日志」保留。
- **R4** 正式训练从 clean HEAD 起跑，起跑前打 Beta 锚点；起跑后到训练结束，被跟踪的 `src/ scripts/ packages/` 照上一轮红线 18 做 `chmod a-w` 只读锁，原权限另存 `v1-store/m8v-scripts/m2k-chmod-before.txt`，结束后恢复。
- **R5** 80k 中途崩溃：停下问用户，不自动 resume、不自动改参数。
- **R6** 任一条 OOM：停下报错交用户（裁决 D），不自行换卡数、不改 mask；步时慢不停。
- **R7** tmux 前缀 `m2k-`；只按本轮会话名清单清理，禁止任何批量杀 tmux。
- **R8** 长期文档不写代码行号，用符号锚点。

## 1. 逐文件改动清单

### S1（训练侧）

| 文件 | 改动 |
|---|---|
| `src/mme_vla_suite/training/framesamp_dataset.py` | `FrameSampDataset.__init__` 形制守卫：在 `new_shape` 之外新增 `ctx_shape`（R2 条件），`_req(legacy_shape or new_shape or ctx_shape, …)`；报错文字补上新档位。不动守卫之后任何计算 |
| `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-context-32frame-8x8.yaml`（新建） | 复制 `perceptual-framesamp-context-8frame-8x8.yaml`，只改 `budget: 2048`；头注释写明来源与唯一差异 |
| `scripts/training/tests/test_pack_guards.py` | 现有「context 大预算应拒」的用例改为：2048 接受、1024/4096 拒；新增类型反例（`2048.0` / `"2048"` / `num_views: true`）与组合反例（+motion、+4×4、多视角、`memory_token_dim 1024`）；旧用例保留 |
| `scripts/dataset/hf_export/mixed8_200_runner/train-ctx2048-80k.sh`（新建） | 照 `train-512-80k.sh`，`HC=perceptual-framesamp-context-32frame-8x8.yaml` |
| `scripts/dataset/hf_export/mixed8_200_runner/train-modul2048-80k.sh`（新建） | 照 `train-512-80k.sh`，`HC=perceptual-framesamp-modul-32frame-8x8.yaml` |

`_train_common.sh` 不改（已含 mixed8 库、assets、`FSDP_DEVICES`/`NUM_WORKERS` 透传、`DRY_RUN`）。

### S2（上传侧）

| 文件 | 改动 |
|---|---|
| `scripts/dataset/hf_export/run_mixed8_200_vla_ctx2048_80k_ckpt_export.sh`、`…_modul2048_80k_ckpt_export.sh`（新建） | 照 `run_mixed8_200_vla_512_80k_ckpt_export.sh`，改 RUN_NAME、BUCKET_ID、EXPORT_NAME（`hf-ckpt-mixed8-200-vla-{ctx,modul}2048-80k`）、allowlist、README |
| `scripts/dataset/hf_export/hygiene_allowlist_mixed8_200_vla_{ctx,modul}2048_80k.json`（新建） | 照 512 版改 run 名 |
| `scripts/dataset/hf_export/mixed8_200_vla_{ctx,modul}2048_80k_bucket_README.md`（新建） | 照 512 版，写明 2048、接入方式、yaml 名与 sha256（合并后回填） |

`_mixed8_200_vla_ckpt_export_common.sh` 不改。增量上传沿用不进 git 的 `v1-store/m8v-scripts/m8v-ckpt-inc.sh`（参数 `<RUN_NAME> <BUCKET_ID> <EXPORT_NAME> <ALLOWLIST> <训练 tmux 名>`，内含 `EXPECT_STEPS=5000…75000 79999`，适用 80k）。

## 2. 子代理分配表

| 编号 | 目标 | 模型 / 隔离 | 可写集合 | 禁触 | 依赖与合并顺序 | 验收命令与判定行（主检出外的 worktree，uv 环境） | 资源 |
|---|---|---|---|---|---|---|---|
| S1 | 训练侧改动 | `opus`，worktree，基线为开工时 HEAD | 第 1 节 S1 表 5 个文件 | 其他一切；`_train_common.sh` | 无依赖；第 1 个合 | `uv run python -m pytest scripts/training/tests/test_pack_guards.py -q` 全过；两 runner `DRY_RUN=1` 输出 `HC=` 与 `TRAIN_CMD=` 正确；`GUARD_CTX2048=PASS accept=1 reject_type=3 reject_combo≥4 legacy_accept 不变` | CPU |
| S2 | 上传侧改动 | `opus`，worktree，同基线 | 第 1 节 S2 表 6 个文件 | 其他一切；`_mixed8_200_vla_ckpt_export_common.sh` | 无依赖；S1 之后合 | `bash -n` 两驱动；`python -I -c json.load` 两 allowlist；`UPLOAD_LINT=PASS` | CPU |
| R1 | 合并前后审查（S1、S2 各一次前审、一次后审） | `sonnet`，只读 | 无 | 全部 | 钉死 `REVIEW_BASE`/`REVIEW_TIP` | `PRE_MERGE_REVIEW=PASS|FAIL …`、`POST_MERGE_REVIEW=PASS|FAIL …` | — |
| Q1 | 只读核 mixed8 库 `step<31` 执行样本数；context 2048 在线装配（`FrameSampMemory` + `policy.py`，CPU） | `haiku`，只读 | 无（输出到 scratch） | 全部 | S1 合并后 | `SHORT_HIST=<n>/<1383846>`；`ONLINE_ASM_CTX2048=PASS steps=5 mem_len=2048` | CPU |
| X1 | 短测（步骤 4–5）与正式起跑（步骤 7） | `opus`，主检出，不加 worktree | 无（只写 `v1-store/` 产物与日志） | 任何代码、配置、脚本、参数 | 合并 push 后；正式起跑等用户开工令 | 见第 4 节 runbook 判定行 | 8 卡；tmux `m2k-smoke-ctx`/`m2k-smoke-modul`/`m2k-train-ctx`/`m2k-train-modul` |
| 主会话 | Beta 锚点、建档、只读锁、增量上传会话、Monitor、收尾、handoff | — | `docs/training-doc/<run>/`、`docs/handoff/current.json`、`v1-store/m8v-scripts/m2k-*` | — | — | — | tmux `m2k-inc-ctx`/`m2k-inc-modul` |

共享文件归属：`docs/handoff/current.json` 只由主会话写。S1/S2 commit subject 前缀 `sub/S1: ` / `sub/S2: `，不 push、不合主分支。

## 3. 闸门总表

| 编号 | 内容 | 判定行 | 时机 |
|---|---|---|---|
| G0 | 守卫正反例 | `GUARD_CTX2048=PASS …`；pytest 全过 | S1 内 + 合并后主检出复跑 |
| G1 | runner 干跑 | 两条 `DRY_RUN=1` 的 `HC=`、`--dataset-path …/framesamp-8x8`、`--fsdp-devices 4 --num-workers 8` | 合并后 |
| G2 | 短历史统计 | `SHORT_HIST=…`（只记录，不设门槛） | 合并后 |
| G3 | context 在线装配 | `ONLINE_ASM_CTX2048=PASS` | 合并后 |
| G4 | 20 步短测 ×2（同时，4+4） | 每条 `SMOKE20=PASS oom=0 loss_finite=1 peak_mib=<…> steady_s_per_step=<…>` | 合并后 |
| G5 | context 100 步保存加载 | `CKPT_LOAD_CTX2048=PASS leaves_equal=1 mem_leaves_changed≥1 actions_match=1 mem_len=2048` | G4 通过后 |
| G6 | 起跑 | 两条日志 300 步内无 `Traceback`/`RESOURCE_EXHAUSTED`，loss 有限 | 用户开工令后 |
| G7 | 训练结束 | `EXIT_CODE=0`、16 份 ckpt | 训练末 |
| G8 | 上传 | `SHA256_PRE_POST_DIFF=0`、bucket `private=False` | 收尾 |

任一 FAIL 即停，原文交用户。

## 4. runbook

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
R=scripts/dataset/hf_export/mixed8_200_runner
# G1
DRY_RUN=1 FSDP_DEVICES=4 NUM_WORKERS=8 bash $R/train-ctx2048-80k.sh  mixed8-200-m32x8x8-context-b128-80k
DRY_RUN=1 FSDP_DEVICES=4 NUM_WORKERS=8 bash $R/train-modul2048-80k.sh mixed8-200-m32x8x8-modul-b128-80k
# G4：短测 run 名带 smoke 后缀，结束即删（≤5 min 的部分不留档；超时按第 17 条补档）
tmux new-session -d -s m2k-smoke-ctx "cd $PWD && CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_PREALLOCATE=false FSDP_DEVICES=4 NUM_WORKERS=8 bash $R/train-ctx2048-80k.sh m2k-smoke-ctx-<UTC> --num-train-steps 20 --no-wandb-enabled; echo EXIT_CODE=\$? >> v1-store/logs/m2k-smoke-ctx-<UTC>.log"
tmux new-session -d -s m2k-smoke-modul "cd $PWD && CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_PYTHON_CLIENT_PREALLOCATE=false FSDP_DEVICES=4 NUM_WORKERS=8 bash $R/train-modul2048-80k.sh m2k-smoke-modul-<UTC> --num-train-steps 20 --no-wandb-enabled; echo EXIT_CODE=\$? >> v1-store/logs/m2k-smoke-modul-<UTC>.log"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -lms 500 > v1-store/logs/m2k-smoke-nvml-<UTC>.csv   # 另起 tmux m2k-smoke-nvml
# 步骤 7（用户开工令 + Beta 锚点之后）
tmux new-session -d -s m2k-train-ctx   "cd $PWD && CUDA_VISIBLE_DEVICES=0,1,2,3 FSDP_DEVICES=4 NUM_WORKERS=8 bash $R/train-ctx2048-80k.sh  mixed8-200-m32x8x8-context-b128-80k"
tmux new-session -d -s m2k-train-modul "cd $PWD && CUDA_VISIBLE_DEVICES=4,5,6,7 FSDP_DEVICES=4 NUM_WORKERS=8 bash $R/train-modul2048-80k.sh mixed8-200-m32x8x8-modul-b128-80k"
# 增量上传（每条一个会话）
tmux new-session -d -s m2k-inc-ctx   "bash v1-store/m8v-scripts/m8v-ckpt-inc.sh mixed8-200-m32x8x8-context-b128-80k HongzeFu/robomme-vla-mixed8-200-context-2048-80k-v1 hf-ckpt-mixed8-200-vla-ctx2048-80k scripts/dataset/hf_export/hygiene_allowlist_mixed8_200_vla_ctx2048_80k.json m2k-train-ctx 2>&1 | tee v1-store/logs/m2k-inc-ctx.log"
tmux new-session -d -s m2k-inc-modul "bash v1-store/m8v-scripts/m8v-ckpt-inc.sh mixed8-200-m32x8x8-modul-b128-80k HongzeFu/robomme-vla-mixed8-200-modul-2048-80k-v1 hf-ckpt-mixed8-200-vla-modul2048-80k scripts/dataset/hf_export/hygiene_allowlist_mixed8_200_vla_modul2048_80k.json m2k-train-modul 2>&1 | tee v1-store/logs/m2k-inc-modul.log"
```

`--num-train-steps` 与 `--no-wandb-enabled` 的确切 CLI 名由 S1 在 DRY_RUN 中对照 `train.py` 的 tyro 参数核实；G5 的 100 步保存加载复用 0920 计划 V8 的加载核对写法（`create_trained_policy`），由 X1 执行前把命令写进 `launch.md`。每个日志各挂一个 Monitor，过滤 `EXIT_CODE=|Traceback|RESOURCE_EXHAUSTED|out of memory|nan`。

## 5. 风险登记

| 编号 | 风险 | 应对 |
|---|---|---|
| K1 | context 2048 4 卡 OOM（候选 mask 下已 79.4 GiB；用户称以前跑过不会 OOM） | G4 实测；OOM 即停下报错（裁决 D） |
| K2 | context 步时远超 modulation，两条结束时间差很大 | 用户裁决不管；G4 后只告知两条工期；先结束的那条 4 卡空出不自动挪用 |
| K3 | 原 mask 首次编译很慢 | 短测不设 240 s 时限；Monitor 过滤词含编译卡死特征（长时间无 step 行时人工核） |
| K4 | 两条同时读同一份 525G 库，IO 争用 | 上一轮 512+motion 4+4 已同库并行无异常；G4 看步时 |
| K5 | 磁盘：两条 ckpt 约 360G + 两条新回读副本约 360G | 上一轮回读副本已删，可用 3.8T，足够 |
| K6 | 增量上传与训练抢 IO | `m8v-ckpt-inc.sh` 已 `ionice -c 3 nice -n 19` |

## 6. 盲区诚实清单

- 两条 2048 在 4 卡 b128 下的显存与步时都没有原入口实测；modulation 的 43–45 h 是从 8 卡和 512 倍率推的。
- context 2048 从未完整训练过；数值质量与 modulation 的对比只能等训练结束、评估后再说。
- mixed8 库短历史比例未核（G2）。
- 推理侧 context 2048 只做 CPU 在线装配冒烟（G3），不做在线评估；评估另立计划。

## 7. 交接顺序与状态机登记

- **任务 id**：`mixed8-2048-train`（用户裁决登记，`planned`）。
- **依赖的上游**：`mixed8-vla-build`（closed；假设库 `16task-mixed8-200-3200ep-e3c7564f` 的 `framesamp-8x8` 为 `status verified`、`num_rows 2145548`、`VERIFY_PACK … mismatches=0`）。不依赖 `mixed8-vla-train`。
- **交接前必须完成（pre_handoff）**：两条训练 `EXIT_CODE=0`（日志 `v1-store/logs/<run>.log`）；两条 `result.md` 回写并以 `commitV11.27` / `commitV11.28` push；只读锁按 `m2k-chmod-before.txt` 恢复 `PERM_RESTORED=OK`。
- **长尾（long_tail）**：`m2k-inc-ctx`、`m2k-inc-modul` 增量上传（完成行 `INC_END`）；两条正式上传驱动全量回读（完成行 `SHA256_PRE_POST_DIFF=0`）。
- **下游**：本任务 `done` 后可开两条 2048 run 的在线评估（另立计划）。

## 8. 留档与 commit 纪律

- S1、S2 合并后 `commitV11.26: …`（训练侧 + 上传侧，合并提交保留 `sub/S1`、`sub/S2`）；计划本身 `docs:`。
- 正式起跑前分别打 `commitV11.27Beta`（context）、`commitV11.28Beta`（modulation），建档 `docs/training-doc/<run_name>/launch.md`；结束后 `result.md` 回写并以 `commitV11.27` / `commitV11.28` 提交。
- handoff 新任务 `mixed8-2048-train`（含两条 run 与两条上传长尾）；会话末 `handoff:` 提交。
- 一律 `git add <逐个路径>`，commit 后裸 `git push`。
