# docs/training-doc/ 索引

> 每个 run 一个目录：`launch.md`（起跑前预提交：commit、命令、配置、数据来源、输出路径、判据）、`result.md`（判定行原文与结论）、`records/`（Git 无法还原的日志与指标）。
> 新 run 仍落本目录；环境 A 的吞吐 / 确定性 / dtype 中间产物 / v5 对拍与集群正式 run 留档已于 2026-09-07 迁入 [`../archive/training-doc/`](../archive/README.md)。
> 环境 A（GreatLakes / turbo + 2×RTX 6000 Ada）与环境 B（AWS 8×A100）的数字按 `AGENTS.md` 第 13 条不混比，下表按环境分档。

## 一、环境 B 现行（AWS 8×A100，2026-09-04 起）

### 训练 / 推理一致性对拍（2026-09-07，commitV7.1 起）

| 目录 | 内容 | 判定 |
|---|---|---|
| [`tic-l0-rhythm-40k/`](tic-l0-rhythm-40k/result.md) | 第 0 关配置与库同源（`check_config_provenance.py`）+ 评估真节奏 CPU 复刻（`eval_rhythm_gates.py`）+ A19 按库重算后在 400 ep 库重跑 M1 | PASS（`TIC_L0` / `TIC_RHYTHM` / 400 ep `A19_VALID_DIST`、`MOTION_DELIVERY` 全 PASS） |
| [`tic-obs-model-40k/`](tic-obs-model-40k/result.md) | 第 1–5 关：输入键 → 预处理后 → 模型内部 → 整段前向 vs 缓存分步 → 最终动作，motion 从库查表（`compare_train_infer_obs.py --motion store`） | 12/13 阻断 PASS；`VT_FULL_VS_CACHED` 超事先阈值待裁决（纯数值来源） |
| [`tic-t3-causal-40k/`](tic-t3-causal-40k/result.md) | `T3_MOTION_CAUSAL` 按收窄口径（单列不确定叶）在 40 ep 库原记录上重跑 | PASS（`covered=36/36 excluded=0`） |
| [`tic-sidecar-40k/`](tic-sidecar-40k/result.md) | 第 1–5 关再跑一遍，motion 换真 sidecar 现算（`--motion sidecar`） | 13/14 阻断 PASS；`MOTION_S_VS_SIDECAR` 140 窗逐位同；`VT_FULL_VS_CACHED` 同上 |
| [`tic-eval-probe-40k/`](tic-eval-probe-40k/result.md) | 第 6 关：探针版 policy server + 主线 `eval.py` 24 集单次仿真，汇总器核不变量 | 6/7 阻断 PASS；`EVAL_PROMPT=FAIL`（1/24 集 goal 组合训练未见，数据覆盖缺口） |
| [`tic-vulkan-makeenv/`](tic-vulkan-makeenv/result.md) | 单进程第 28 次 `make_env` 必崩（Vulkan）的根因排查与修法验证 | PASS（复现第 28 轮；两修法 35 轮不崩） |

方案、判据总表与结论见 [`../train-infer-consistency.md`](../train-infer-consistency.md)。

### 生产 run、基准与评估

| 目录 | 内容 | 判定 |
|---|---|---|
| [`awsprod40k-b128-motion/`](awsprod40k-b128-motion/result.md) | 生产训练：40k 步、global batch 128、400 ep 库、motion 接入 | 跑完，checkpoint 39999 |
| [`bench-b128-util/`](bench-b128-util/result.md) | 起跑前钉 b128 的卡数 / fsdp / util 档位 | 见 result.md |
| [`repro-4a100-fsdp4/`](repro-4a100-fsdp4/result.md) | 4×A100 fsdp4 复现基准（40 G 模拟与 80 G 两档） | 见 result.md |
| [`siglip-ab-replay-40k/`](siglip-ab-replay-40k/result.md) | 39999 checkpoint 下 `_prepare_history` 八键 vs 训练表逐位重放；SigLIP f32 离线表 vs bf16 checkpoint 差异量化 | 八键 PASS；SigLIP 差异 rel_fro 0.34% |
| [`eval-awsprod40k-b128-motion/`](eval-awsprod40k-b128-motion/result.md) | motion 组仿真评估 | 见 result.md |
| [`eval-official-framesamp-context/`](eval-official-framesamp-context/result.md) | 官方 framesamp+context 权重仿真评估；首次记录 Vulkan 第 28 次 `make_env` 崩溃与分片 ≤27 红线 | 见 result.md |

### motion 接入闸门（环境 B 复刻，40 ep 库）

| 目录 | 闸门 | 判定 |
|---|---|---|
| [`aws-a22-grad/`](aws-a22-grad/result.md) | A22 式单步定点梯度两侧互核 | PASS kinds=3 leaves=32 mismatches=0 |
| [`aws-p5-online/`](aws-p5-online/result.md) | P5 在线 sidecar 逐窗 vs 离线 motion 表 | PASS compared=772 mismatches=0 |
| [`aws-t2-ref-s100/`](aws-t2-ref-s100/result.md) / [`aws-t2-cand-s100/`](aws-t2-cand-s100/result.md) | T2 关闭态 100 步 A/B | ref PASS；cand 的 `BASELINE_ENV` 因 GPU 指纹不同报 FAIL（记录在案，见 result.md） |
| [`aws-t3-closed-s100/`](aws-t3-closed-s100/result.md) / [`aws-t3-open-s100/`](aws-t3-open-s100/result.md) | T3 closed / open 100 步 | PASS；`T3_MOTION_CAUSAL` 的 `pad_bitexact=0` 由 `tic-t3-causal-40k/` 按新口径闭合 |

## 二、环境 A：motion 接入依据（2×RTX 6000 Ada + turbo，只读）

| 目录 | 闸门 | 判定 |
|---|---|---|
| [`motion-a21-g0b-replay/`](motion-a21-g0b-replay/result.md) | A21：S2 起工前 HEAD 复跑 G0b 黄金基线 1000 步 | PASS，锚点同值 |
| [`motion-a22-grad/`](motion-a22-grad/result.md) | A22 单步定点梯度 | PASS kinds=3 mismatches=0 |
| [`motion-p5-online/`](motion-p5-online/result.md) | P5 真编码器在线链 vs 离线表 | PASS episodes=40 |
| [`motion-t1-closed/`](motion-t1-closed/result.md) | T1 关闭态训练等价（旧库 1000 步，对 G0b r1） | PASS |
| [`motion-t2-ref/`](motion-t2-ref/result.md) / [`motion-t2-cand/`](motion-t2-cand/result.md) | T2 新库 300 步严格 A/B | PASS |
| [`motion-t3-closed/`](motion-t3-closed/result.md) / [`motion-t3-open/`](motion-t3-open/result.md) | T3 真实训练端到端 1000 步 | PASS |

## 三、环境 A：dataloader 重构梯度对拍链（只读，仍是判据锚）

| 目录 | 角色 | 判定 |
|---|---|---|
| [`v1-grad-baseline-g0/`](v1-grad-baseline-g0/result.md) | G0 黄金基线（PG0） | 已被 G0b 取代，保留 |
| [`v1-grad-baseline-g0b/`](v1-grad-baseline-g0b/result.md) | G0b 黄金基线 1000 步（链头） | 两轮逐位一致 |
| [`v1-dtype-p5-grad/`](v1-dtype-p5-grad/result.md) | dtype 修复单步定点梯度 B 侧 | PASS |
| [`v1-framesamp-g2/`](v1-framesamp-g2/result.md) | G2：framesamp 三表读路径训练对拍 | PASS |
| [`v1-singlerun-g0/`](v1-singlerun-g0/result.md) | C3：`train.py` 单跑入口 G0 重锚 | PASS，锚点同值 |
| [`v1-postclean-g3/`](v1-postclean-g3/result.md) | G3：v4 破坏性重构正确性长跑 | PASS |
| [`v1-l0-gauge/`](v1-l0-gauge/result.md) | L0 修量具后的两档基准 | PASS |

## 四、环境 A：评估（只读，成功率数字标注 RTX 6000 Ada）

| 目录 | 内容 |
|---|---|
| [`eval-3seed-context-vs-motion/`](eval-3seed-context-vs-motion/result.md) | 三 seed 四任务成功率对照：motion 24.2% ± 1.3 vs 官方 24.5% ± 0.5，无可辨别差异 |
| [`eval-hard-patternlock-routestick/`](eval-hard-patternlock-routestick/result.md) / [`eval-medium-patternlock-routestick/`](eval-medium-patternlock-routestick/result.md) | PatternLock / RouteStick 难度分层评估 |
| [`eval-binfill-pickxtimes/`](eval-binfill-pickxtimes/result.md) | BinFill / PickXTimes 评估 |

## 五、已归档（`../archive/training-doc/`，34 项）

环境 A 的吞吐 / 瓶颈基准 15 项、确定性四档 8 项、dtype 中间产物 3 项、v5 对拍与集群正式 run / smoke 7 项、未完成空壳 1 项（`eval-official-framesamp-modul`）。清单与理由见 [`../archive/README.md`](../archive/README.md)。
