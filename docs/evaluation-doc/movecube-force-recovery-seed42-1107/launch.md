# MoveCube 强推恢复实验起跑档案

计划：[`1107-movecube-force-recovery-plan.html`](../../../1107-movecube-force-recovery-plan.html)，开工令所指版本 `3bc7c5f`，开工前文字统一后为 `0acfab9`。

## 目的与用户决策原话

开局把机械臂强行带到错误物体的正上方（z=0.2），在 0/16/32/48/64/80 步放手交还策略，看它能否回到正确物体并抓起。方向 A：拿 cube 的 21 局强推到 peg 上方；方向 B：拿 peg 的 12 局强推到 cube 上方。两模型、模型 seed 42，共 396 局。不记录任何注意力。

用户原话按时间顺序：

1. 「给出方案 做latent steering的实验 对movecube任务 做2分 拿/推cube和拿peg的任务二分 对于拿cube推cube的任务 往peg方向走执行实际 DLA 的动作，就是强行把它往强行呃给一串动作让它往往右边走，然后走完之后，然后走完之后，你看它在每一个点位上，就是每一个点位上是不是都能够恢复，就是它有没有呃它有没有这个这个恢复的可能性，然后两个模型都要测。」
2. 选择题：本局规划到 peg；走向本局 peg；只测行为恢复；38局×6点×2模型。
3. 「这次完全不测内部的激活，就是我只看它被拉到一个方向之后还能不能恢复。」「权重都不需要保留。」「我完全不需要看内部注意力的机制，所以说你都不需要有任何的这个呃记录的功能」
4. 「我我不需要，就是这个往右边走，不是不是这个意思，是是指的是能不能从 peg 上面走到这个 cube 上面。哦，反过来也要测，就是如果往 cube 这边走，它反过来能不能走向走向peg」「然后两个模型都要测。就如果被判断为失败的，就就还是判断为失败，就不用去管它。」
5. 「完整的 high level 的方案，你现在第一部分这个太多了，你把它缩减」
6. 「我们降低一些难度，它只要能够正确地回到物体并且抓取成功就可以了，不需要任务成功。」选择题：去掉这 17 局（gripper_push）。
7. 「同意开工」；开工审查裁决：共用服务分批；只验到达放手的局。「同意开工」。

本任务不登记 `docs/handoff/current.json`（用户「我我不需要」），交接走人工。

## 版本与代码状态

- 执行树 `v1-store/worktrees/movecube-force-run`，detached 固定在 `b625c1d8fa7972c6a754ce316c9c3a8c1bf0e1f2`（commitV4.1.46），`env.sh` 起跑时校验 HEAD 与干净树。
- 本轮代码：S1 合入 `3c7286f`（commitV4.1.45，动作表、客户端、`run_local.py` 的 `movecube_force` 模式），S2 合入 `b625c1d`（commitV4.1.46，分析脚本）。
- benchmark 锁定 `856bc3a189d4172f3f47dbee4424d585f8d78db3`。
- 审查：`PLAN_CONSISTENCY_REVIEW=FAIL base=872f091 findings=5 blocking=2`（用户裁决后修订）；`PLAN_CONSISTENCY_REVIEW=FAIL base=3bc7c5f findings=3 blocking=1`（主会话核实 blocking 不成立，见 `0acfab9`）；`PRE_MERGE_REVIEW=PASS`（S1 findings=1，S2 findings=2）；`POST_MERGE_REVIEW=PASS merge=3c7286f tests=96passed`；S2 合并后 82 passed。

## 运行环境与被跑对象

环境 A，sled-aspen，2×RTX A6000（每卡 46068 MiB），不用集群。起跑时 GPU0 约 5.9 GiB、GPU1 约 2.4 GiB 为其他进程占用，未动。

| 标记 | checkpoint | 配置 | model_kind |
|---|---|---|---|
| mot | `v1-store/assets/vla-o16cc-59999/59999` | `mme_vla_suite_orig16_b128_60k` | `stage2_pool4_concat` |
| ctx | `v1-store/models/official-mme-vla/perceptual-framesamp-context/79999` | `mme_vla_suite_b128` | `history` |

服务为官方 `scripts.training.serve_policy`，客户端 `scripts/interpret/movecube_force_runner.py`，每卡 2 个 worker，`XLA_PYTHON_CLIENT_MEM_FRACTION=0.30`，`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`。

## 局与分片

官方 test 划分。方向 A（grasp_putdown）：0,2,6,9,10,17,18,19,21,27,28,30,34,35,36,40,42,43,44,45,48；方向 B（peg_push）：1,3,4,5,8,11,15,25,31,32,33,47。按局号排序后轮流分 4 片（9/8/8/8 局），每批至多 5 局。s0、s1 在 GPU0，s2、s3 在 GPU1。

## 动作表

`FORCE_TABLE=PASS episodes=33 plan_fail=0`；方向 A 路径中位数 42 步（38～52），方向 B 中位数 57 步（36～89），B 有 2 局（ep15 为 88 步、ep47 为 89 步）在 80 步截断；开环途中终局 0。表 sha256 前缀 `552fd198d192ff79`，454954 字节。

## 冒烟（G3）

2 局（MoveCube:0 方向 A、MoveCube:1 方向 B）× k∈{0,48} × 2 模型：

```
FORCE_EXEC=PASS runs=8 mismatches=0
FORCE_RELEASE=PASS reached_release=8 mismatches=0
NO_ATTN=PASS npz=0
MFR_RUN=PASS completed=8 missing=0 errors=0
```

k=0 四局全部抓起成功；k=48 四局全部失败。轨迹显示放手后两模型都下降并闭合夹爪抓了正下方的错误物体（A 末端到 peg 尾部水平 2～7 mm，B 抓起 cube），环境判 fail。录像截帧目视复查与轨迹一致：放手帧腕部相机正对错误物体，末帧方向 A 停在 peg 上、方向 B cube 在爪中。

## 完整命令

```bash
O=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/movecube-force-recovery-seed42-1107
# 清单与 spec
cd /data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/worktrees/movecube-force-run && PYTHONPATH=$PWD:$PWD/src /data/hongzefu/robomme_policy_learning_MotionJEPA/.venv/bin/python $O/gen_specs.py
# 动作表
tmux new-session -d -s mfr-table "set -o pipefail; source $O/env.sh mot 1; PYTHONUNBUFFERED=1 \$sim scripts/interpret/movecube_force_table.py --manifest $O/manifest-all.json --out $O/force-table.json 2>&1 | tee $O/logs/table.log; echo \"EXIT_CODE=\$?\" >> $O/logs/table.log"
# 冒烟
tmux new-session -d -s mfr-smoke-mot "set -o pipefail; bash $O/run_one.sh mot 0 s0 smoke 0,48 2>&1 | tee $O/logs/smoke-mot.log; echo \"EXIT_CODE=\$?\" >> $O/logs/smoke-mot.log"
tmux new-session -d -s mfr-smoke-ctx "set -o pipefail; bash $O/run_one.sh ctx 1 s0 smoke 0,48 2>&1 | tee $O/logs/smoke-ctx.log; echo \"EXIT_CODE=\$?\" >> $O/logs/smoke-ctx.log"
# 全量（美东 2026-11-07 15:42 起跑），i=0..3，GPU 为 i<2 ? 0 : 1
tmux new-session -d -s mfr-s$i "set -o pipefail; bash $O/run_shard.sh <gpu> s$i 2>&1 | tee $O/logs/full-s$i.log; echo \"EXIT_CODE=\$?\" >> $O/logs/full-s$i.log"
```

`run_shard.sh` 依次执行 `run_one.sh mot <gpu> s<i> all 0,16,32,48,64,80` 与同样的 ctx。

## 产物与盯盘

产物根 `v1-store/evaluation/movecube-force-recovery-seed42-1107/`（不进 git）：`force-table.json`、`manifest-*.json`、`specs/`、`<mot|ctx>/k<k>/s<i>/`（回执、逐步轨迹、录像）、`smoke/`、`logs/`。每份日志一个 Monitor，过滤 `RUN_END|RUN_FAIL|SHARD_DONE|EXIT_CODE=|RUN_LOCAL=FAIL|Traceback|out of memory|Killed|NaN`。

本轮 tmux 会话清单：`mfr-table`、`mfr-smoke-mot`、`mfr-smoke-ctx`（均已自然退出，EXIT_CODE=0）、`mfr-s0`、`mfr-s1`、`mfr-s2`、`mfr-s3`。

## 实施中调整

- 开工审查第二轮的 blocking（强推期间无推理导致 `infers=0`）经核实不成立：首个动作前必有一次推理。
- 抓取判定只在放手后做；回执新增 `grasped_object`；`load_force_spec` 拒绝官方服务不支持的键；动作表与 `force_k` 并入 `run_identity`。
- S2 首版按 `forced` 读 `control_source`，与 S1 定稿的 `force` 不符，已续改。
- S1 审查的非阻塞项（抓起与环境超时同一步记抓取成功）不改，符合判据。
- 合并提交 subject 首次误写 `commitV11.86`，推送前改为 `commitV4.1.45`。
