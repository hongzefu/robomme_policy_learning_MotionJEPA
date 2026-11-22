# 1122 官方16×100 modulation + motion（concat 1×1024，stride 16）60k 训练与单 seed 评估方案

状态：定稿（2026-11-21 美东），等开工令。分支 `v19-orig16-modul-s2concat-s16-60k`（自 `origin/v4-gl-orig100-1031@1c652dc7`）。

用户原话（2026-11-21 美东）：「给出方案 重新训modulation + motion 但是这次stride 16 且不要split4 先只eval一个seed　在评估结束后立刻开始训 之前可以写代码　还是60k」

外部依赖锚点：encoder `/scratch/hongze/MotionJEPA/runs/stage2-full144-1025-a/checkpoint_epoch_105.pt`（sha256 `27daa8320c9a…`，训练源为官方 16×100 源 H5 `/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/source/h5`）；VLA 库 `v1-store/datasets/16task-orig100-1600ep-9e109419`（建库时已有 `DATA_LINEAGE=PASS`，开工后对新 run 重跑）；Wan latent `v1-store/cache/orig16-wan-timepool1024-recovery/wan-latents/`（共 45,228 窗）。提交号沿用 `commitV<x>.<y>` 体例，从本分支最近的编号接续。

# 第一部分（给人看）

## 一、和上一条（10-30 modulation split）相比只改三处

| 项 | 上一条 `orig100-m32x4x4-modul-motion-s2split4x256-b128-60k` | 本条 `orig100-m32x4x4-modul-motion-s2concat1x1024-s16-b128-60k` |
|---|---|---|
| motion 读出 | split：每窗 4 个 256 维 token | **concat：每窗 1 个 1024 维 token**（4 个 pool 位按第 0/3/5/8 组首尾相接） |
| 窗口步长 | 32 | **16**（窗长仍 33 帧，相邻窗重叠 17 帧） |
| motion budget | 96 | **192**（96 翻倍，用户 11-21 指定；理由见第三节） |

其余完全不变：训练库 `16task-orig100-1600ep-9e109419`；帧记忆 32 帧×4×4＝512；encoder `stage2-full144-1025-a/checkpoint_epoch_105.pt` 的 EMA（sha256 `27daa832…`，冻结）；pi05_base 起训；8 卡 FSDP8，batch 128，workers 16，seed 42；60k 步，预热 5k，lr 5e-5，EMA 0.999；norm_stats 沿用库内 `c5d45b4b…`；每 5000 步存 1 份 ckpt，共 12 份。DATA_LINEAGE：encoder 训练源和 VLA 库都是官方 16×100，同源。

对照组：10-30 的 modulation split（stride 32）。它的 3 seed 成绩是 43.42±0.52，其中 seed 42 为 346/800＝43.25%。本条只评 seed 42。注意对照组是在 GreatLakes A40 上评的，本条在本机 A100 上用移植的 `local_eval.py` 评，**只作跨平台参考**，差值不归因于 readout、stride 或 budget。可选：8 卡空闲时在本机补评旧 split 59999 的 seed 42，作为同平台对照（不挡训练）。

## 二、数据：不重跑 Wan，只重跑 encoder

- 本机已有 grid16 的全量 Wan latent：`v1-store/cache/orig16-wan-timepool1024-recovery/wan-latents/`，共 45,228 窗（9×16×32×32 f32，25 GB），metadata 写 grid_stride=16，导入回执 PASS。
- 旧的 `motion-tokens-stage2/` 只对偶数格做过 encoder，因此对全部 45,228 窗重新跑一遍 stage2 encoder（单卡；上次 2.3 万窗约 7 分钟，这次估计约 15 分钟）。导出后打包成新库 `motion-stage2-1024-grid16/`（layout `motion-stage2-1x1024-grid16-demopad17-v1`，约 4.6 万行，约 190 MB）。旧的 grid32 库不动。
- demo 尾窗规则不变：≥17 个真实帧才保留，用最后一帧补满（repeat_last）；执行段窗口必须全是真实帧。

## 三、budget 192（96 翻倍，用户指定；开工后实测，结果写进本文）

- 训练侧粗算：最坏的 VideoPlaceOrder ep4（demo 1118 帧、exec 293 帧）在 stride 16 下有 69＋17＝86 窗，96 只剩 10 的余量。
- 评估侧每局最多 1300 步，执行段最多 (1300−32)//16＋1＝80 窗，再加 demo 段，会超过 96。超预算在代码里是直接报错（raise），不会截断。
- 用户 11-21 美东原话：「budget 96翻倍！」——步长减半、窗口数约翻倍，budget 同比翻倍到 192。参照：mixed8 上 stride 16 用 160 时，评估侧最坏已到 157。
- 闸门：`train_es_bound.py --layout <grid16> --budget 192` 和 `eval_es_bound.py --budget 192`（已按 stride 16 写死）都必须 PASS。若评估侧最坏超过 192，**停下问用户**，不自动放大。
- 代价：记忆序列从 608 变成 704 个 token（512＋192），比上一条 split 的 896 还短，速度预计不慢于上一条的 0.97 s/步。

## 四、代码改动（评估结束前完成，只用 CPU；需要 GPU 的 G2/G5 放到 t1s 结束后）

在本分支的 worktree 中改（主检出 v18 和正在跑的 t1s 评估都不碰）：

1. **布局**：`motion_store.py` 和 `motion_adapter.py` 新增 `motion-stage2-1x1024-grid16-demopad17-v1`（grid_stride=16），保留原 grid32 布局。
2. **导出/打包**：`export_stage2_memory.py` 和 `pack_motion_store.py` 中「取偶数格」「32*j」「grid_subsample=2」的硬编码改成可传参，本条传 subsample=1、stride=16；原值作为默认，旧行为不变。
3. **在线与检查脚本**：`check_stage2_online.py`、`check_stage2_checkpoint.py` 中写死的 `stride=32` 与 `window_budget==96` 改成读 yaml/spec；`validate_stage2.py`、`compare_stage2.py` 的布局名比较改成「已知 stage2 布局集合」（grid32 与 grid16 都认）。`motion_client.py` 的 `source_grid % 2` 是 G0 证据校验，不在在线推理路径上，按 subsample 参数化即可。
4. **配置与 runner**：新 yaml `perceptual-framesamp-modul-32frame-4x4-motion-stage2-1x1024-s16-orig16.yaml`（复制已有的 modul concat yaml，只改 stride、budget、store_path）；在 `check_orig16_stage2_launch.py` 登记新 run 名和 bucket；新增 runner 和 ckpt 上传包装器。
5. **评估**：把 `t1s-1120-evalstack-v42` 上的本机官方 800 局入口（`local_eval.py`、`resident_worker.py`、`official_queue`、`examples/robomme/eval.py` 中的 OFFICIAL_SEEDS）移植进本分支，新增 profile `modulcat16`，只跑 seed 42。保证评估 server 和训练用同一份模型代码。
6. **测试**：同步更新 `test_pack_motion_stage2.py`、`test_export_stage2_memory_portable.py`、`test_guards.py` 中关于 grid32 和偶数格的断言，并新增 grid16 用例。

分工：写入用 opus 子代理（各自 worktree，分别负责 S1 布局与导出、S2 在线与检查脚本、S3 配置/runner/上传、S4 评估移植），每份合入前由 sonnet 审查，主会话按 `--no-ff` 合入。

## 五、开训前闸门（全部 PASS 才排队）

| 闸门 | 判定行 | 何时 |
|---|---|---|
| G0 同源 | `DATA_LINEAGE=PASS` | 代码合入后 |
| G1 单测 | 全部 pytest 通过 | 代码合入后 |
| G2 encoder 导出 | `STAGE2_EXPORT=PASS windows=45228`；偶数格行与旧 grid32 表逐字节相等（`EVEN_GRID_BITEXACT=PASS`） | t1s 结束后、触发链第一步，约 15 分钟 |
| G3 打包校验 | `MOTION_PACK_VERIFY=PASS mismatches=0` | CPU |
| G4 预算 | `TRAIN_ES_BOUND=PASS budget=192`、`EVAL_ES_BOUND=PASS budget=192` | CPU |
| G5 在线离线一致 | `ONLINE_STAGE2=PASS`（在线 sidecar 算出的 token 与离线表逐位一致，抽样 ≥64 窗，其中奇数格 ≥16 窗且含 demo 尾补帧窗） | t1s 结束后、触发链第二步；同时实测在线编码耗时，修正评估时间估计 |
| G6 启动合同 | `LAUNCH_CONTRACT=PASS`（相对已有 modul concat yaml 只差 stride/budget/store_path 三项） | CPU |
| G7 8 卡 20 步短测 | `SMOKE20=PASS save_load=PASS` | **评估结束后**，正式链的第一步 |

## 六、起跑：评估一结束立刻开训

- 当前正在跑的 `t1s-3seed-1120`（tmux `t1s-next`，日志 `v1-store/logs/t1s-batch2.log`）还剩约 4 对，预计**美东 11-22 凌晨 0:30 前后**结束。
- 触发链 tmux `o16s16-chain`：只读等待 `t1s-batch2.log` 出现 `T1S_CHAIN_DONE` 和 `EXIT_CODE=0`，**并且** 8 卡显存都低于 2000 MiB。判定写法：`grep -q '^T1S_CHAIN_DONE batch2'` 且 `grep -qx 'EXIT_CODE=0'`（锚定行首，避免误配 `SHARD_EXIT_CODE=`）。然后依次跑 G2 导出 → G3 打包 → G4 预算 → G5 在线一致 → G6 启动合同 → G7 短测 → 正式 60k → 训完自动起 seed 42 评估；任一闸门 FAIL 即停链。不动 t1s 的会话和句柄。本方案的导出与评估用 `v1-store/venvs/wan-stage2`，与 t1s 用的 `venvs/wan` 互不共用。
- 预计耗时：训练约 16 h（参照上一条 0.97 s/步），G2–G7 约 40 分钟；评估约 1 h（本机 8 卡×4 worker，参照 timepool 单 seed 55–60 分钟；在线编码窗口数约翻倍，G5 实测后修正）。**预计美东 11-22 傍晚出结果。**
- 卡长时间不空：不起跑，报告用户。

## 七、ckpt 与磁盘

- 11-21 美东用户批准清理：删除本地 `/scratch/hongze/dataset-4task-rgb-state-stride2-16views-v1`（约 1.8 TB，HF 公开 bucket `robomme-4task-rgb-state-stride2-16views-fp16-v1` 有完整副本，`HF_VERIFY=PASS files=11038`；记录文件留档 `/scratch/hongze/logs/4task-rgb-state-1113/local-deleted-20261122/`）。删后余量约 1.9 TB，12 份 ckpt（约 143 GB）可全部留在本地。
- 做法：增量上传 tmux `o16s16-inc`。每出 1 份 ckpt 就上传到公开 bucket `HongzeFu/robomme-vla-orig100-modul-s2concat-s16-60k-v1`，读回校验 sha 通过即可；本地 12 份全部保留，不删。
- 磁盘余量低于 200 GB 时停下报告。

## 八、评估

- 官方 test 800 局（16 任务×50），每局最多 1300 步，用 ckpt 59999，**只跑 seed 42**。
- 交付：成功率和逐任务表，与 split@seed42（43.25%，GL A40）以及官方权重（44.51±0.77）并列，表中标注评估平台。只报结果，不做显著性判读。seed 7 和 0 等用户决定。

## 九、需要用户确认（括号内是默认）

1. budget 192（用户已定：96 翻倍；实测超过 192 时停下问）。
2. 评估用哪个 seed（默认 42，可直接和 split 那条的 seed 42 比）。
3. 磁盘：已按用户批准清掉 1.8 TB，ckpt 本地全留并增量上传（已定）。
4. 状态机登记任务 `o16-modul-s2cat-s16-60k`，并 `git push -u` 本分支（默认登记并推送）。

开工令覆盖：第四节写代码与不占 GPU 的检查 ＋ t1s 结束后的 G2–G7 ＋ 第六节自动起跑的 60k 训练和 seed 42 评估。中途只在以下情况停下：闸门 FAIL、训练崩溃、预算超 192、磁盘不足；其余工程故障自行修复后继续。

## 九′、开工前对抗审查与处置（2026-11-21 美东）

workflow `wf_b11709d8-7d3`（5 个维度各 1 个 sonnet 审查，每条发现 1 个 sonnet 复核）：共 24 条，复核后成立 15 条，**均不涉及训练语义**，按用户授权（「除非有影响训练语义的内容 否则你自己决策了」）由主会话处置，已改进本文：代码清单补漏（评估客户端 `validate_budget`、`compare_stage2.py`、`orig16_wan_cache.py`、启动合同两脚本、检查脚本的 budget/layout、导出包装器）；S4 加 `--seeds` 单 seed 汇总；G2/G5 改到 t1s 结束后、作为触发链第一步（不与评估抢卡）；触发判定锚定行首；对照组改为跨平台参考；G5 加奇数格抽样与在线编码耗时实测。驳回 9 条（理由见 workflow 记录）。

`PLAN_CONSISTENCY_REVIEW=PASS base=8a632f85 findings=15 blocking=0`（原报 2 条 blocking 经复核降级，均已在本文处置）。

## 十、交接顺序与状态机登记

- 任务 id：`o16-modul-s2cat-s16-60k`；依赖：`t1s-3seed-1120`（本仓库 v18 分支）的 `T1S_CHAIN_DONE` 和 `EXIT_CODE=0`。
- 交接前必须完成：G0–G7 全部 PASS；训练日志有 `EXIT_CODE=0`；seed 42 评估 `COMPUTE_DONE completed=800 errors=0`；本地 59999 已上传且读回 `HF_VERIFY=PASS`。
- 长尾：上传完成后的读回校验（tmux `o16s16-inc`，完成行 `INC_END`）。
- 本任务 `done` 之后：seed 7 和 0 的评估是否要跑，由用户决定。

# 第二部分（技术细节，供 agent 追踪）

## 〇、红线

1. 不碰主检出 v18 的工作树，不碰 t1s 评估的 tmux、worktree 和日志；触发链只读等待。
2. 旧 grid32 布局、旧 motion 表、已有 yaml 一律不改，只新增。
3. 不 force push，不删除任何 bucket；本地 ckpt 只有在 `HF_VERIFY=PASS` 之后才删。
4. 只允许新增 yaml 中相对已有 modul concat yaml 的 stride、budget、store_path 三项差异，其余超参不得改动。
5. `DATA_LINEAGE` FAIL 就停（P-LINEAGE）。

## 一、子代理分配表

| 编号 | 目标 | 可写文件 | 依赖 | 验收 |
|---|---|---|---|---|
| S1 | grid16 布局与导出/打包 | `src/mme_vla_suite/datastore/{motion_store,motion_adapter}.py`、`scripts/dataset/wan_stage2/export_stage2_memory.py`、`scripts/dataset/pack_motion_store.py`、`scripts/dataset/wan_stage2/compare_stage2.py`、`scripts/dataset/wan_stage2/orig16_wan_cache.py`、`scripts/dataset/check_stage2_validation_inputs.py`、`scripts/dataset/hf_export/_stage2_orig16_vla_ckpt_export_common.sh`（LAYOUT/budget/store_dir 按 run 登记）、对应 tests（`test_pack_motion_stage2*`、`test_stage2_validation_inputs.py`、`wan_stage2/tests/test_orig16_wan_cache.py`、`test_orig16_export_drivers.py` 等） | 无（布局常量名由主会话派发前约定） | `pytest tests/dataset/test_pack_motion_stage2.py tests/dataset/test_export_stage2_memory_portable.py scripts/dataset/test_guards.py -q` 全过，且 grid32 旧用例不变 |
| S2 | 在线与检查脚本读 yaml 的 stride/budget/layout | `src/mme_vla_suite/policies/motion_client.py`、`scripts/dataset/wan_stage2/validate_stage2.py`、`scripts/training/check_stage2_online.py`、`scripts/training/check_stage2_checkpoint.py`、对应 tests | 与 S1 共享布局常量名（S1 定义、S2 只读） | 定向 pytest 全过 |
| S3 | 新 yaml、启动合同、runner、上传包装器 | `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-32frame-4x4-motion-stage2-1x1024-s16-orig16.yaml`、`scripts/training/check_orig16_stage2_launch.py`、`scripts/training/check_mixed8_stage2_1024_launch.py`（stride/budget/layout/motion_rows/train_max/byte_count 按 run 参数化，旧 run 默认不变）、`tests/training/test_check_orig16_stage2_launch.py`、新 runner `scripts/training/run_orig16_modul_s16_60k.sh`、新导出包装器 | S1 | 合同脚本对旧两条 run 仍 PASS；新 run 的 yaml 相对已有 modul concat yaml 只差 stride/budget/store_path；新导出包装器 dry-run 通过 |
| S4 | 移植本机官方 800 局评估，加 `modulcat16` profile | `scripts/evaluation/{local_eval,resident_worker}.py`、`official_queue` 相关、`examples/robomme/eval.py`（OFFICIAL_SEEDS 与 `validate_budget` 的 stride/budget/exec 上限改为随 spec 取值）、`tests/evaluation/test_testx_client.py` | S2、S3（ckpt 路径与 run 名由主会话写进派发说明） | 新增 `--seeds`（默认三 seed）与 profile 字段（ckpt/out/prefix/port_base/wait_log/POLICY_CONFIG）；单 seed 走单 seed 汇总、不调 stdev；dry-run 打印 seeds=(42,)、32 worker；mock 单 seed 汇总不抛异常；`test_official_seed_gate.py` 通过 |

写入子代理用 opus，各自开 worktree（基线是本分支 HEAD）；每份合入前派 sonnet 审查，钉死 BASE/TIP，判定行 `PRE_MERGE_REVIEW=PASS`；主会话按 S1→S2→S3→S4 的顺序 `--no-ff` 合入，每次合入后跑定向测试并 push。G2–G6 由主会话在合入之后串行跑。

## 二、runbook

1. 主会话：`git push -u origin v19-orig16-modul-s2concat-s16-60k`；在状态机登记并写 `kind: go`。
2. 主会话先约定布局常量名 `STAGE2_GRID16_LAYOUT="motion-stage2-1x1024-grid16-demopad17-v1"`、run 名与 ckpt 路径，再并行派 S1–S4；合入顺序 S1→S2→S3→S4。
3. G2（在触发链里，t1s 结束后）：`CUDA_VISIBLE_DEVICES=0 uv run python scripts/dataset/wan_stage2/export_stage2_memory.py --grid-subsample 1 ...`，输出到 `v1-store/datasets/16task-orig100-1600ep-9e109419/motion-tokens-stage2-grid16/`；再把偶数格行与旧表逐字节比对。
4. G3：`pack_motion_store.py pack --layout stage2 --latent-grid-subsample 1` 生成 `motion-stage2-1024-grid16/`，然后跑 verify。
5. G4：`train_es_bound.py --layout motion-stage2-1x1024-grid16-demopad17-v1 --budget 192`、`eval_es_bound.py aggregate --budget 192`。
6. G5/G6：`check_stage2_online.py`（stride 16）、`check_orig16_stage2_launch.py`。
7. 写 `docs/training-doc/orig100-m32x4x4-modul-motion-s2concat1x1024-s16-b128-60k/launch.md`，打 Beta commit，确认工作区干净。
8. 起 tmux `o16s16-chain`（脚本落 `v1-store/jobs/o16s16/chain.sh`）：等 t1s 结束且 8 卡空闲 → G2–G6 → 20 步短测 → 正式 60k → `local_eval.py --profile modulcat16 --seeds 42`。起 tmux `o16s16-inc` 做增量上传。每份日志单独挂一个 Monitor。
9. 训完、评完：写 result.md，打正式 commit；状态机改 done；长尾读回完成后改 closed。

## 三、风险

- 评估侧窗口数超过 192：停下问用户（G4 会提前暴露）。
- 磁盘：清理后余量约 1.9 TB，12 份 ckpt 约 143 GB，不构成风险；余量低于 200 GB 才停。
- t1s 评估异常拖长：训练顺延，不抢卡。

## 四、盲区

- 官方 test 800 局的最坏 demo 长度还没实测，budget 192 是否够要等 G4。
- 本机 `local_eval.py` 还没跑过 modulation + 在线 stage2 的组合，G5 和评估冒烟是第一次验证。
- stride 16 的步速是按记忆长度更短推算的，还没实测。

## 实施中调整（2026-11-21 美东追加，不改写上文）

| 调整 | 旧 → 新 | 理由与审查 |
|---|---|---|
| S2 合入 | 两轮 PRE_MERGE_REVIEW FAIL 交用户 → 主会话合入 S2@173ad032，policy_config.py 传 grid_stride 转整合子任务 S5 | 第二轮唯一阻塞在 S2 写集外、非训练语义；按用户「除非有影响训练语义的内容 否则你自己决策了」自决 |
| 新增整合子任务 S5 | 无 → S5：train.py stage2 快照判定改为 in STAGE2_LAYOUTS、policy_config 传 grid_stride、删过渡回退、新增闸门脚本 `scripts/training/o16s16_gates.sh` | train.py 布局判等不在任何原写集，S1/S2 审查均指出 |
| G3 证据 | 沿用旧证据 → 在 `v1-store/validation/orig16-stage2-grid16` 重建 G0/G2/G3；G3 的 RGB 用 `plan_g3_subset.py` 从公开桶只读下载 32 窗子集（14 片、32.48 GB，`G3_SUBSET_PLAN=PASS windows=32 shards=14 bytes=32476096020`） | 旧证据绑定已不在盘的旧 latent metadata；RGB 源本地已删。`ADJUST_REVIEW=PASS scale=same findings=2` |
| G0 抽样 | 只抽偶数格 → `check_stage2_validation_inputs.py --grid-subsample 1` 从全部格抽样（缺省 2 不变）；`validate_stage2.py` 增 `--grid-subsample` | 使 G5「奇数格 ≥16 窗」可达；同上审查 |
| G2 判据 | 逐字节 → 先逐字节（`mode=bitexact`），不等时 max_abs ≤ 1e-4 判 `mode=tol` 并记录 mismatched_rows/max_abs/`note=cross_latent_copy`，非有限值即 FAIL | 新旧表跨 latent 副本（恢复版 vs 旧版）；旧表不参与新训练；同上审查（建议先严后宽） |
| G2 判定行格式 | `STAGE2_EXPORT=PASS windows=45228` → 以导出脚本实际输出 `segments=… rows=…` 为准（rows=45228） | 文字与实际输出对齐 |
| 起跑信号 | 等 `T1S_CHAIN_DONE` + `EXIT_CODE=0` → 等 t1s 会话写 `v1-store/jobs/t1s-1120/T1S_CLOSEOUT_DONE` | t1s 收尾仍要在主检出 v18 上提交 8 次归档与 handoff；跨会话约定 `req=f8ace2-11212205-1` |
| 主检出 | v18 → 触发链在标记后 `git checkout --detach <Beta>` + 子模块 + `uv sync --frozen` | runner 按脚本位置推仓库根、用根 .venv；v19 与 v18 的 uv.lock/子模块不同 |
