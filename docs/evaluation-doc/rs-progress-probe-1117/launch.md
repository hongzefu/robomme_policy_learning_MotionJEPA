# RouteStick 进度探针与注意力图：起跑与运行记录

计划：[1117-routestick-progress-probe-attn-plan.html](../../../1117-routestick-progress-probe-attn-plan.html)（开工审查后版本 `637adb5`）。环境 A：sled-aspen 2×RTX A6000 + GreatLakes spgpu 4×A40 占位。全部时间为美东 2026-11-17。

## 用户指令原话（按时间）

1. 「我之前做的论文(C)decode task progess这个问题 能不能在4task的routestick任务上做 给出方案」「是在aspen上做的吗」
2. 「给出完整html方案 然后query4卡」
3. 「图a能不能一起做了？也改为routestick任务」「最新的rss」
4. 「图a也是attn map不是tsne」「你现在modulation模式 要怎么画？」
5. 「你先给我一个假设的出图a和b让我看明白」（假图见 `mock/`）
6. 「你现在要测多少个任务 为什么？」「我问的是你测多少集 为什么？ 之前是怎么做的？」
7. 「同意开始」（开工令，对 `9a96b36` 版计划不改直接同意）
8. 「所有的gpu任务全部改到gl上跑」（实施中，美东约 16:10）

## 版本与代码

| 项 | 值 |
|---|---|
| 开工审查 | `PLAN_CONSISTENCY_REVIEW=FAIL base=9a96b36 findings=11 blocking=4`，按用户「除了会影响评估结果语义的问题，其他的你都自己来决策」修订为 `637adb5` |
| 代码合入 | R1 `6f28377`（`commitV4.9.0`）、R3 `a4a6a98`（`commitV4.9.1`）、R2 `3fa6893`（`commitV4.9.2`）、R3 出图修正 `765b474`（`commitV4.9.3`） |
| 回放运行树 | aspen `v1-store/worktrees/rsp-run`（detached `3fa68930dbd9bbf47eaf3661289a6392236209af`，clean）；GL release `…MotionJEPA-eval/releases/3fa68930dbd9bbf47eaf3661289a6392236209af`（含子模块 `b4e97f2`，`.release_sha` 核对） |
| 出图运行树 | 探针 `rsp-run`（`3fa6893`，探针代码此后未改）；注意力图 `v1-store/worktrees/rsp-run2`（`765b474`） |
| benchmark | 子模块 `b4e97f22fe007078e297205898c07c1acbc69165`；生成在其 `git archive` 副本 `v1-store/evaluation/rs-progress-probe-1117/bench/` 里跑（生成脚本要求输出目录在 benchmark 仓库根内） |
| 模型 A | MotionMem 四任务×400，`v2-1600ep-m8x8-modul-motion-b128-80k@50000`；aspen `v1-store/assets/vla-4task-modul-motion-50000/50000`，GL `…MotionJEPA/v1-store/models/robomme-vla-modul-motion-80k-v1/50000`（steering 时 `VLA_SYNC=PASS model=A files=20 sha_mismatch=0`） |
| 模型 B | Frozen Wan pooling，`v2-1600ep-m8x8-modul-wantp768-b128-50k@50000`；aspen `v1-store/assets/vla-4task-modul-wantp768-50000/50000`，GL `…MotionJEPA-eval/v1-store/models/robomme-vla-4task-modul-wantp768-50k-v1/50000`（`VLA_SYNC=PASS model=B files=25`） |

## 数据生成

- 局池：`rsprog_points build-pool`，源 seed `3400000+100k`，k=0..199，四难度各 50，局号 1000–1049；L 分布 easy {2:26, 3:24}、medium {4:17, 5:33}、hard {4:16, 5:13, 6:15, 7:6}、xhard {8:21, 9:12, 10:17}；`CHECK_SPEC=PASS routes=200 bad=0`、`SPEC_LOAD=PASS groups=4 episodes=200`。spec sha256 前 16 位：easy `a283643a0791f11b`、medium `c0c90e8f9f74d516`、hard `d006d457e1c9c399`、xhard `10b7e27318648817`、manifest `b9d8f8b16a2d03b2`、pool `8aba295424835fe2`（主检出代码重算一致）。
- 生成：aspen tmux `rsp-gen-smoke`，`v1-store/envs/bench-b4e97f2/bin/python scripts/generate_dataset_newseed.py --episode-specs …/specs/manifest.json --episode-start 1000 --episodes 1 --layout train --workers 4 --gpus 0,1 --output-dir rs-gen-smoke`。清单模式不理会 `--episodes 1`，直接生成全部 200 局，12:49–13:20，全部 attempt 0，`EXIT_CODE=0`。
- 合并：每难度 `--merge-only` 到 `v1-store/datasets/rs-progress-1117/RouteStick/<d>/record_dataset_RouteStick.h5`（7.7 / 14.4 / 16.1 / 27.6 GiB）。sha256：easy `3f4ce693…`、medium `86ce1cf2…`、hard `410f96bf…`、xhard `fc779661…`；turbo 副本逐文件相同（`DATA_SYNC=PASS files=4 mismatches=0`）。
- 取点：`GEN_SPEC=PASS eps=200 mismatches=0 dropped=0 irregular=0 missing=0`；3421 点（easy 422、medium 749、hard 833、xhard 1417）；`PAIRS n=566 kept=0`。

## 回放

- 冒烟（aspen，k=0）：A `MEM_BLIND=PASS prefix_maxabs=0 action_maxabs=73.8765`、`REPLAY_PARITY=PASS points=7 maxabs=1.78813934e-07 tol=0.001`、`HIDDEN_PRECISION=PASS` 六组（`path_maxabs=0`）、`REPLAY_PARITY_BF16_INFO maxabs=0.00280261`；B `MEM_BLIND=PASS prefix_maxabs=0 action_maxabs=63.8488`、`REPLAY_PARITY=PASS maxabs=2.38418579e-07`、`HIDDEN_PRECISION=PASS` 六组。
- 全量：每模型 200 局按 k 的 sha256 切 3 片。第 0 片先在 aspen（A GPU1 比例 0.75，B GPU0 比例 0.62），第 1、2 片在 GL 占位 `66304774`/`66304775`（A）、`66304776`/`66305145`（B），`XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`。用户 16:10 要求 GPU 任务全部改到 GL 后，停掉 aspen 两片（A 已完成 36 局、B 38 局），剩余局按点数均分为 x1、x2，接在 GL 对应占位上跑（`s0_remaining.json`）。
- 判定：A s1 `REPLAY_RUN=PASS episodes=66 points=1131`、s2 `episodes=62 points=1009`、x1 `episodes=16 points=388`、x2 `episodes=17 points=398`；B s1 `episodes=66 points=1131`、s2 `episodes=62 points=1009`、x1 `episodes=17 points=404`、x2 `episodes=17 points=404`；汇总 `REPLAY_SET=PASS model=A episodes=200/200 points=3421 expected=3421 memmod_maxabs=0 attn_rowsum_maxerr=2.38e-07`，B 同。
- 环境变量（aspen `env-aspen.sh`，GL `run_gl.sh`）：`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`、`MMEVLA_MOTION_PROV_RELAX=gpu_name,compute_cap,sm_count,driver`、`MMEVLA_MOTION_ONLINE_GPU=0`、`HF_HUB_OFFLINE=1`；K=5 噪声样本，seed 42。
- 耗时：GL A40 easy 约 49 s/局、medium 约 81 s/局；aspen 同局 140–300 s（CPU 单核瓶颈，机器负载 29/16 核）。全部回放 18:05 结束。

## 分析

aspen CPU：`rsprog_attn --points … --replay-root replay-all --output analysis-attn2`（`765b474`）与 `rsprog_probe … --workers 8`（`3fa6893`）。产物与清洗日志在 `records/`。

## 本轮 tmux 会话（均已退出）

`rsp-gen-smoke`、`rsp-smoke-A/B`、`rsp-smoke2-A/B`、`rsp-smoke3-A/B`、`rsp-smoke4-B`、`rsp-full-A`、`rsp-full-A-s0`、`rsp-full-B-s0`（按用户指令停止）、`rsp-rsync`、`rsp-gl-A-s1/s2`、`rsp-gl-B-s1/s2`、`rsp-chain-A-x1/x2`、`rsp-chain-B-x1/x2`、`rsp-rerun-A-x2`、`rsp-analysis`、`rsp-attn2`。

## GL 占位

`66304774`、`66304775`、`66304776`（8 CPU/96G）、`66305145`（4 CPU/64G，替换因 `AssocGrpCpuLimit` 排不上的 `66304777`），12:18 起 RUNNING，18:10 按清单 `scancel`，前后 `squeue --me` 核对，训练占位 `65943150`、`65943151` 未动。
