# 1026 计划：用阶段 III 最后 ckpt 抽 1024 维 memory，训一条 VLA（窗 33 / 步长 32）

状态：**草案，等用户确认开工**。确认前不改任何代码、不占 GPU、不上传。
计划放在 policy 仓库（本事完全是 policy 侧的事），MotionJEPA 侧只新增一个只读抽取脚本。
写于 2026-10-26 美东中午；用户已定：EMA 权重、预算 96、不跑 4 token 对照臂。
MotionJEPA 基线 `v12-3decoder@6c79ebb`，policy 基线 `v3-motionmem-pair4@8c319aa`。

---

## 1. 现状（已核实）

| 项 | 事实 | 证据 |
|---|---|---|
| 最后 ckpt | `runs/stage2-full144-1025-a/checkpoint_epoch_105.pt`，第 106 轮停训；含 EMA（`model`）、live（`model_live`）、optimizer；EMA decay 0.99；314 MB | `docs/handoff/current.json` 任务 `stage3-formal-training-v1`；ckpt header |
| encoder 输出 | 每个 33 帧窗（9 个 Wan 时间片）→ 4 个 readout token × 256 维，对应源时间片 [0,3,5,8]；encoder 与三个 decoder 在同一文件 | `src/motion_jepa/models/wan_latent_encoder.py:269-290,430-431` |
| dropout | encoder / dit / state 各 0.05，训练态生效；旧推理脚本 `scripts/inference-example/wan_motion_infer.py:349` 硬断言 `num_tokens==8`，不能直接用于本模型 | `runs/stage2-full144-1025-a/config.yaml` |
| 输入归一化 | encoder 输入 fp32 cast 后做 affine，统计量 `docs/training-doc/stage2-model-phase2-20261025/step1-full-root/qstats.json` | config `qstats_path` |
| Wan 缓存 | **已有**：policy 仓 `v1-store/restore/mixed8-200-vla-latents-20261021-a/wan-latents/`，16 任务 × 200 集，33 帧窗、grid16（起点 0,16,32,…）、fp32、无增强、同一 VAE 权重 SHA、约 70.5 GiB；exec 3195 段（PatternLock 缺 5 集 exec，与上次建库一致）、demo 1800 段 | 各 `*.metadata.json` |
| Wan 编码等价 | 阶段 II/III 训练缓存用「整 33 帧」fp16 编码，恢复副本用标准 1+32 fp32 编码；两者归一化 MSE 6.9e-8，可互换 | `docs/training-doc/wan-online-preprocessing-20261023/round4-fp16.md:63`、`1024-rgb-gtxyz-16views-stride2-plan.md:435` |
| 策略 memory 口径 | motion 路现为 33 帧窗、GRID_STRIDE 16（相邻窗重叠 17 帧）、每窗 (8,256) 行、投到 `memory_token_dim=1024`；帧路 8 帧 × 64 token 不变 | policy `src/mme_vla_suite/datastore/motion_store.py:45-60`、`models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion-mixed8.yaml` |
| 增强 | 官方 OpenPI 只在训练期对**当前帧**图像增强（主相机 RandomCrop 0.95→Resize→Rotate ±5°→ColorJitter；腕相机只 ColorJitter），eval `train=False` 全关，官方无 memory；本 fork 的 memory 一直离线从干净帧抽取，不增强、无 dropout、无噪声 | policy `src/openpi/models/model.py:144-190`、`training/framesamp_dataset.py` |
| VLA 库 | `v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/framesamp-8x8`（525 GB）在本地；mixed8 留出集 = 每个任务×难度格最后 2 集 | 1016 计划口径 |
| 泄漏 | encoder 训练源是原版 16×100，VLA 库是 mixed8-200，源 episode 不同，无 encoder 泄漏；但 encoder 未见过 mixed8 画面（域外） | 两库 scope |
| ckpt 上传 | 阶段 II（`runs/stage2-scale-20261025-d`：ep4、ep8、final）与阶段 III（21 份）**均未上传** | handoff 长尾 pending |
| 旁注 | rgb-gtxyz 数据库公开上传实际失败（`HF_VERIFY=FAIL error=ValueError`、`EXIT_CODE=1`，进程已退），handoff 仍写 running；属 Codex 那条任务，本计划不处理，只报告 | `/scratch/hongze/logs/rgb-gtxyz-full-20261024/upload.log` 末行 |
| 资源 | 8 卡全空；/scratch 可用 1.6 TB | `nvidia-smi`、`df` |

## 2. 口径（已定）

1. **权重**：ep105 的 **EMA** 权重（用户 10-26 确认）。
2. **token 合并**：每窗 4×256 按源时间片顺序 [0,3,5,8] **拼接**为 1 个 1024 维 token（不是平均）。策略侧 `memory_token_dim=1024` 不变。
3. **窗口**：33 帧（锚点 + 未来 32 步），**GRID_STRIDE 32**，起点 segment_start，forward。相邻窗只共享 1 个边界帧，不再重叠 16。
   - **demo 段 padding 规则不变（用户 10-26 要求）**：demo 窗起点 0,32,64,…；末窗真实帧数 ≥ 17 时保留，用最后一帧复制补到 33 帧（repeat_last）；真实帧数 ≤ 16 的末窗丢弃。exec 段窗必须 33 帧全真实、不补帧（与现库 `min_real` exec=33 / demo=17、`truncation_policy=none` 一致）。
   - 恢复副本是 grid16，其 demo 尾窗已按同一规则补好；步长 32 取 g 为偶数的行，等价于直接在 stride 32 上按该规则枚举（32k 处的尾窗存在当且仅当真实帧 ≥ 17）。打包时按 metadata 的 `real_frames` 重新断言这一点，判定行并入 G5。
   - 在线评估端 `pack_buffer` 的 demo 窗枚举改为同规则（stride 32、≥17 补帧），由 G4 对拍覆盖。
4. **预算**：motion budget **96**（用户 10-26 确认）。依据：上次 stride16 时训练集最坏 123 窗 / 预算 160、评估侧 1300 步上限最坏 157；步长翻倍后约 62 / 79，80 在评估侧只剩 1 的余量，96 保险且 token 开销差别很小。G6 实算后若最坏窗数 > 96 停下问。
5. **增强**：memory **不增强**（与官方及既有口径一致），用干净 latent（等价 aug_id=0）。
6. **dropout**：抽取与在线 sidecar 都显式 `model.eval()` + `torch.no_grad()`，并用闸门证明（§4）。
7. **数据**：直接用恢复副本中 g 为偶数的窗，不重过 Wan VAE；PatternLock 缺 exec 的 5 集沿用上次处理（无 motion 行即不可见）。
8. **训练配置**：沿用 `mme_vla_suite_b128_80k`（80k 步、batch 128、pi05_base 初始化、EMA 0.999、每 5000 步存，共 16 份），一条 run 独占 8 卡（fsdp 8）。与已上传的 512、512+motion 两条基线同配置可比。
9. **不跑对照臂**（4 个 256 不拼接作 4 个 token 的版本，用户 10-26 确认不跑）。

## 2a. 官方评估 1300 步上限对本方案的影响

官方 eval（`examples/robomme/eval.py:30,88`）每集 exec 最多 1300 步（超过即判失败），每步取 16 步动作块执行。对本方案的影响只有 memory 预算一项：

| 段 | 最坏长度 | stride 16 窗数（旧） | stride 32 窗数（新） |
|---|---|---|---|
| demo（最长 VideoPlaceOrder 1247 帧） | 1247 | 77 | 39 |
| exec（1300 步上限） | 1300 | 80 | 40 |
| 合计 | — | 157（旧预算 160） | **79**（新预算 96，余量 17） |

结论：1300 步上限给出的评估侧最坏 motion 窗数是 79，预算 96 够用，不会出现评估到后段 memory 装不下、被截断的情况；训练集最坏约 62 窗更小。其余影响与 memory 无关：帧路仍是 8 帧均匀采样，不随步数增长；超过 1300 步未完成的集按官方规则记失败，对所有模型一样。G6 用新库实算训练集最坏窗数并复算上表，任一超过 96 停下问。

## 3. 分工与产物

### A. MotionJEPA 侧（新分支，从 6c79ebb 切；只加一个脚本）

- 新脚本 `scripts/stage2_memory_export.py`：
  - 加载 ckpt（`weight_type=ema`，取 `model`），构造 encoder，`eval()`，`no_grad`，fp32 cast + affine（qstats 路径取自 ckpt 内 config，断言文件 sha 一致）。
  - 读恢复副本每段 `.bin`（按 metadata 行表，只取 `m % 2 == 0` 的行），输出每窗 float32[1024]，逐段写 `npz`（含 segment、m、start_global_frame、encoder 指纹、ckpt sha256）。
  - 提供可复用函数 `encode_windows(latents[B,9,16,32,32]) -> [B,1024]`，给 policy 的在线 sidecar 复用（同一函数、同一权重）。
- 不改既有训练/评估代码。

### B. policy 侧（新分支，从 8c319aa 切）

- `datastore/motion_store.py`：新 layout `motion-stage2-1x1024-grid32-demopad17-v1`，行形状 (1,1024)，GRID_STRIDE 32；`visible_motion_rows` 的 `16m` 改成读 layout 的 stride；旧 layout 保持可读。
- `scripts/dataset/pack_motion_store.py`：支持新 layout，输入 A 的 npz。
- `models/representation/percep_mem.py`：motion 维度从 yaml 读（768→1024），投影 Linear(1024+768→1024)。
- 新 yaml `models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion-stage2-mixed8.yaml`：window 33、stride 32、dim 1024、pos_dim 256、budget 96，motion store 指向新库。
- 在线 sidecar（`MotionEncoderClient` 所用的 `wan_motion_infer.py` 或其替代）改调 A 的 `encode_windows`，去掉 `num_tokens==8` 断言；eval 端 `pack_buffer` 的窗口枚举按 stride 32。
- 训练中增量上传脚本 `v1-store/m8v-scripts/m8v-ckpt-inc.sh` 不改，只换参数。

### C. 产物

- 新 motion 库：`v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/motion-stage2-1024/`（约 6.4 万行 × 4 KB ≈ 260 MB）。
- 训练 run：`v1-store/train-runs/mme_vla_suite_b128_80k/mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k/`。
- 文档：policy `docs/training-doc/mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k/{launch,result}.md`（含抽取记录与全部判定行）；MotionJEPA 只在 `docs/training-doc/stage2-full144-1025-a/` 留一行指针。
- 状态真源：policy `docs/handoff/current.json` 新任务 `mixed8-vla-stage2-1024-train`。

## 4. 闸门（判定行；任一 FAIL 停下问用户）

| # | 闸门 | 判定行 | 做法 |
|---|---|---|---|
| G1 | dropout 真关了 | `DROPOUT_OFF=PASS` | 同一批窗 `eval()` 跑两遍逐位相等；再在 `train()` 下跑一遍结果不同；两条都成立才 PASS |
| G2 | Wan 两种编码可互换 | `WAN_EQUIV=PASS worst_mse≤1e-6 worst_max≤0.01` | 抽 64 个恢复副本窗（分层含 demo 尾窗），用 `wan_fp16_full33.py` 重编，按 `latents_std` 归一化比对 |
| G3 | 导出函数与训练评估同路 | `EXPORT_PARITY=PASS max_abs≤1e-5` | 在 rgb-gtxyz 验证集 32 个 aug0 窗上，`encode_windows` 与 `stage2_eval.py` 的 encoder 输出比对 |
| G4 | 离线库 = 在线 sidecar | `ONLINE_OFFLINE_PARITY=PASS max_abs≤1e-5` | 随机 64 窗，sidecar 输出 vs 库中行 |
| G5 | 库完整 | `VERIFY_MOTION=PASS scanned=<n> mismatches=0` | 行数 = 偶数格窗数，全 finite，ckpt sha256 与指纹写进 meta |
| G6 | 窗口预算 | `TRAIN_ES_BOUND=PASS max_windows=<n> budget=96` | 遍历训练集与评估侧 1300 步上限，最坏窗数 ≤ 96 |
| G7 | 短测 | `SMOKE_TRAIN=PASS steps=20 loss_finite=1` | 8 卡 20 步，不 OOM、loss 有限、吞吐记录 |
| G8 | 正式训练 | `EXIT_CODE=0` 且 16 份 ckpt 齐 | 见 §5 |
| G9 | VLA ckpt 上传 | `INC_END … done=16` 后正式驱动 `CKPT_UPLOAD=PASS` | 见 §6 |
| G10 | encoder/decoder ckpt 上传 | `MJ_CKPT_UPLOAD=PASS files=<n>` | 见 §6 |

## 5. 训练怎么跑

1. 代码：A、B 各由一个写入子代理在独立 worktree 完成，sonnet 审查 PASS 后合并到各自新分支并 push（不进 main）。
2. 抽取：单卡（GPU 0）跑 `stage2_memory_export.py`，预计十几分钟（IO 为主，未实测），依次过 G1–G3；打包过 G4–G6。
3. 短测：8 卡 20 步（G7），记录 s/步。上次 4 卡 1.73 s/步；8 卡估约 1.0 s/步、80k 步约 22–24 h。**估值，以短测为准**。
4. 正式起跑（detached tmux，前缀 `cc-vla-s2-`，日志三件套，Monitor 盯 `EXIT_CODE=|Traceback|out of memory|nan`）：

   ```bash
   # 在 policy 主检出；具体 CLI 以 B 合并后的 launch.md 为准
   tmux new-session -d -s cc-vla-s2-train "bash <launch 脚本>"
   ```

5. 训练期：沿用 1016 红线（训练期两仓代码冻结、被跟踪 src/scripts 只读锁，训完按记录恢复权限）。崩溃停下问，不自动 resume。
6. 训完：按 1016 收尾（result.md、吞吐、val loss 曲线），评估另立计划。

## 6. ckpt 怎么传

### 6.1 VLA ckpt（边训边传）

- 复用 `v1-store/m8v-scripts/m8v-ckpt-inc.sh`（每 300 s 轮询 `_CHECKPOINT_METADATA` 已提交的 step、hygiene 体检、`hf sync` 最多 3 次重试、写 `.done`；训练 tmux 退出后打 `INC_END`）。
- 新公开 bucket：`HongzeFu/robomme-vla-mixed8-200-modul-motion-stage2-1024-80k-v1`。
- 训练起跑后立刻起 `tmux cc-vla-s2-inc`；`INC_END` 后按既有驱动做全量 sha256 回读 → `CKPT_UPLOAD=PASS`，回写 handoff 长尾。

### 6.2 encoder/decoder ckpt（现在就能传，不占卡，与训练并行）

- 内容：阶段 III `runs/stage2-full144-1025-a/`（21 份 ckpt ≈ 6.6 GB、config.yaml、train_metrics_epoch.jsonl、motion_ablation/）＋阶段 II `runs/stage2-scale-20261025-d/`（ep4、ep8、final、config）＋两份 README 指明 EMA/live 键与指纹。
- 新公开 bucket：`HongzeFu/motionjepa-stage2-pool4-3decoder-1025-v1`。
- 流程：本地 sha256 清单 → 上传 → 匿名全字节回读比对 → `MJ_CKPT_UPLOAD=PASS files=<n>` → handoff `stage3-formal-training-v1` 长尾改 pass。
- tmux `cc-mj-ckpt-upload`，日志 `/scratch/hongze/logs/mj-ckpt-upload-20261026/`。

## 7. 时间线（相对开工令 T0，均为估值）

| 阶段 | 耗时 | 卡 |
|---|---|---|
| A、B 代码 + 审查 + 合并 | 2–3 h | 0 |
| 抽取 + G1–G6 | 0.5–1 h | 1 |
| 短测 G7 | 10 min | 8 |
| 正式训练 | 22–24 h（待短测） | 8 |
| 6.2 上传 | 1–2 h，与上面并行 | 0 |

## 8. 待用户裁决

1. bucket 名两处（§6.1、§6.2）是否可用默认名。
2. **开工令**。已定项：EMA、预算 96、不跑对照臂。

## 9. 不做的事

- 不重过 Wan VAE；不做 memory 增强；不改 encoder 权重；不动 v12-3decoder / main；不处理 rgb-gtxyz 上传失败（只报告）；不在本计划内做 benchmark 评估。
