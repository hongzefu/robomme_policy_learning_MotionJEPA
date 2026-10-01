# eval-orig80k-modul-vs-official — 起跑记录

**环境 B（AWS 单机 8×A100-SXM4-80GB，`/dev/md0` 本地 NVMe RAID）**。正式评估，按 AGENTS 第 12 条留档（两段式：本文件起跑前写，`result.md` 与 `records/` 跑完后补）。

## 一、目的与用户指令原话

在同一套 RoboMME test split 仿真评测上，把 commitV11.18Beta（`0b904ab`）起跑、已训完的两条原版 80k run 与官方发布的 perceptual-framesamp-modul 80k 权重逐集配对比较，看差距；每集回放视频全部保留。

- 「给出方案 跑run bucket 文件 / 字节 v2-orig-16task-pub1600ep-modul-b64-80k HongzeFu/robomme-vla-orig-16task-1600ep-modul-b64-80k-v1 190 / 95,073,680,531 B v2-orig-counting-pub400ep-modul-b64-80k HongzeFu/robomme-vla-orig-counting-400ep-modul-b64-80k-v1 202 / 95,075,858,231 B 和官方80k的modul的eval 8卡并行 看差距 跑完保留回放」
- 「没看懂 还是分两部分 评估代码从哪里来 训练结果放在policy learning这个repo？」（方案改为两部分体例后获批）
- 执行前 AskUserQuestion 定下的两条口径：
  1. counting 模型评哪些任务 →「只评 4 个 counting 任务」
  2. 评估驱动代码怎么落地 →「拷回 examples/robomme 并提交」

计划正文：`~/.claude-personal/plans/rippling-rolling-whistle.md`（harness 计划文件，不进 git；本文件是它的落地版）。

## 二、被评对象（三份 checkpoint）

| 评估名（RUN_NAME） | checkpoint 目录 | 来源与身份 | 任务 | 集数 |
|---|---|---|---|---|
| `eval-orig80k-official-modul` | `v1-store/models/official-mme-vla/perceptual-framesamp-modul/79999` | HF `Yinpei/perceptual-framesamp-modul` @ `c0f565dd40082f32863b3ee9db99de5ed5d3d4e0`，`79999.zip` 11,878,950,895 B，sha256 `2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62`（本地重算 = HF 下载元数据记录值）；本轮 2026-09-28 用 `scripts/training/unzip_ckpt.py::unzip_one` 只解这一个 zip，47 s，解出 11,877,156,757 B；run 根 `history_config.txt` = `perceptual-framesamp-modul.yaml`（无换行） | 16 全集 | 800 |
| `eval-orig80k-full16` | `v1-store/train-runs/mme_vla_suite/v2-orig-16task-pub1600ep-modul-b64-80k/79999` | 训练 HEAD `0b904ab`，`v1-store/bench/orig80k/<run>/completion.json`：`status=PASS mode=prod state_step=80000 final=79999` | 16 全集 | 800 |
| `eval-orig80k-count4` | `v1-store/train-runs/mme_vla_suite/v2-orig-counting-pub400ep-modul-b64-80k/79999` | 同上，`status=PASS mode=prod state_step=80000 final=79999` | BinFill, StopCube, PickXtimes, SwingXtimes | 200 |

**加载口径**：两个本地 run 根目录有 `history_config.resolved.yaml`（无 motion 块；`motion_provenance.json` 为 `motion_enabled=false`；resolved sha 前缀 `823c3948` = 仓库 `perceptual-framesamp-modul.yaml`），走严格快照分支加 `_assert_param_tree_exact`。官方只有 `history_config.txt`，走旧兼容的非严格分支（`remove_extra_params=True`），所以起跑前三份都跑了参数树核对（CPU，`records/param_tree_{official,full16,count4}.json`）：

```
official  PARAM_TREE_EXACT=PASS config=mme_vla_suite history_config=perceptual-framesamp-modul.yaml yaml_sha256=823c3948e75a9335 n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0
full16    PARAM_TREE_EXACT=PASS config=mme_vla_suite history_config=perceptual-framesamp-modul.yaml yaml_sha256=823c3948e75a9335 n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0
count4    PARAM_TREE_EXACT=PASS config=mme_vla_suite history_config=perceptual-framesamp-modul.yaml yaml_sha256=823c3948e75a9335 n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0
```

官方这一行与远端历史记录 `eval-hard-patternlock-routestick/records/modul/param_tree.json` 完全相同。

## 三、评估代码从哪来、本轮改了什么

- **仿真环境**：submodule `third_party/robomme_benchmark` @ `856bc3a`，editable 安装在 `/scratch/hongze/micromamba/envs/robomme`（已核 `robomme.__file__` 指向该 submodule 的 `src/robomme/`）。test split 每任务 50 集，种子表在 `env_metadata/test/record_dataset_<Task>_metadata.json`。
- **模型服务**：`scripts/training/serve_policy.py` 加 `src/mme_vla_suite/`（主仓 `.venv`），每个 worker 起一个。
- **评测主循环**：`examples/robomme/{eval.py,utils.py,env_runner.py}`，本轮用 `scripts/training/legacy-eval/robomme-local/` 同名文件覆盖（即 `c704bf5` commitV5.6 那一版）。补回的是 `--args.dataset`、`--args.episode_start`、`--args.episode_stride`，以及逐集自证行 `EVAL_EPISODE split= task= ep= env_seed= difficulty=`。`src/mme_vla_suite/policies/*` 没有拷：两份只差 motion sidecar 卡号覆盖，本轮三个模型都没有 motion。
- **调度**：`scripts/training/legacy-eval/eval_all_shards.local.sh` 负责拆片，`eval_shard.local.sh` 负责单片执行（clean HEAD 硬闸、端口就绪轮询 + server 死亡检测、`trap cleanup EXIT`、`GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192`、`EXIT_CODE=` 尾行）。本轮只改了一处：`eval_all_shards.local.sh` 增加转发 `POLICY_MEM_FRACTION`，默认 0.4，旧命令行为不变。
- **对比**：新增 `scripts/training/legacy-eval/compare_eval_runs.py`。按 (task, ep) 配对，先核 seed 一致，给出 overall / suite / 难度 / 任务四个层面的 delta、配对 bootstrap 95% CI（10000 次，seed 0）和精确 McNemar p。自测结果 `COMPARE_SELFTEST=PASS delta=+0.2500 p=0.0129395 exp_p=0.0129395`。

## 四、分片与资源

- 三个模型同时起，各 `MODE=stride WORKERS=8`：worker k 用 GPU k、端口 `PORT_BASE+k`，评每个任务的 ep ∈ {k, k+8, …}（w0、w1 每任务 7 集，其余 6 集）。16 任务的 worker 各 96 或 112 集，count4 的 worker 各 24 或 28 集，均低于 `GLIBC_TUNABLES` 支撑的约 147 轮。
- 每卡同时驻留 3 个 policy server + 3 个仿真进程，`POLICY_MEM_FRACTION=0.28`（约 22.5 GB/server）。本机 96 核、1.1 TB 内存。
- split=test，seed=42，max_steps=1300（eval.py 默认），ckpt_id=79999。
- 起跑前实测：8 卡 0 MiB 占用、无计算进程；24 个正式端口与 3 个 smoke 端口均空闲；`v1-store/evaluation/` 下没有 `*orig80k*` 目录；`/scratch` 余 483 GB。

## 五、执行顺序与完整命令

顺序调整说明：计划原定「smoke → Beta commit → 正式」。但 `eval_shard.local.sh` 的 clean HEAD 硬闸对 smoke 同样生效，所以改为「Beta commit（本文件随之提交）→ 同一 HEAD 上 smoke → smoke 通过后直接正式起跑」，其间不做任何 commit。smoke 实测数（显存峰值、每集秒数）补写在 `result.md`。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
# smoke（GPU 0 三片并发，各 1 集；RUN_NAME=smoke-orig80k-*，跑完删除本轮这 3 个评估目录与日志）
for spec in "off RouteStick v1-store/models/official-mme-vla/perceptual-framesamp-modul/79999 8091" \
            "full BinFill v1-store/train-runs/mme_vla_suite/v2-orig-16task-pub1600ep-modul-b64-80k/79999 8092" \
            "cnt BinFill v1-store/train-runs/mme_vla_suite/v2-orig-counting-pub400ep-modul-b64-80k/79999 8093"; do
  read -r n task ckpt port <<<"$spec"
  TASK=$task K=0 GPU=0 PORT=$port EP_COUNT=1 RUN_NAME=smoke-orig80k-$n CKPT_ID=79999 CKPT_DIR=$PWD/$ckpt \
    SEED=42 DATASET=test LOG_PREFIX=smoke-orig80k-$n POLICY_MEM_FRACTION=0.28 \
    bash scripts/training/legacy-eval/eval_shard.local.sh 2>&1 | tee v1-store/logs/smoke-orig80k-$n.log &
done; wait

# 正式
COMMON="MODE=stride WORKERS=8 CKPT_ID=79999 SEED=42 DATASET=test POLICY_MEM_FRACTION=0.28"
T16=BinFill,StopCube,PickXtimes,SwingXtimes,ButtonUnmask,VideoUnmask,VideoUnmaskSwap,ButtonUnmaskSwap,PickHighlight,VideoRepick,VideoPlaceButton,VideoPlaceOrder,MoveCube,InsertPeg,PatternLock,RouteStick
env $COMMON RUN_NAME=eval-orig80k-official-modul LOG_PREFIX=ev16off PORT_BASE=8031 TASKS_ALL=$T16 \
  CKPT_DIR=$PWD/v1-store/models/official-mme-vla/perceptual-framesamp-modul/79999 \
  bash scripts/training/legacy-eval/eval_all_shards.local.sh
env $COMMON RUN_NAME=eval-orig80k-full16 LOG_PREFIX=ev16full PORT_BASE=8051 TASKS_ALL=$T16 \
  CKPT_DIR=$PWD/v1-store/train-runs/mme_vla_suite/v2-orig-16task-pub1600ep-modul-b64-80k/79999 \
  bash scripts/training/legacy-eval/eval_all_shards.local.sh
env $COMMON RUN_NAME=eval-orig80k-count4 LOG_PREFIX=ev4cnt PORT_BASE=8071 TASKS_ALL=BinFill,StopCube,PickXtimes,SwingXtimes \
  CKPT_DIR=$PWD/v1-store/train-runs/mme_vla_suite/v2-orig-counting-pub400ep-modul-b64-80k/79999 \
  bash scripts/training/legacy-eval/eval_all_shards.local.sh
```

## 六、产物路径

- 每片结果：`v1-store/evaluation/<RUN_NAME>-w<k>/ckpt79999/seed42/{progress.json,log.json,videos/}`。
- **回放视频**：同目录 `videos/<Task>_ep<n>_<True|False>_<goal>_<difficulty>.mp4`，成功、失败、超时每集都存；本轮不删、不移动。
- 驱动日志：`v1-store/logs/<LOG_PREFIX>-w<k>.log`；server 日志：`v1-store/logs/<LOG_PREFIX>-w<k>.server.log`。
- 合并结果：`merge_eval_shards.py` 写 `v1-store/evaluation/<RUN_NAME>/ckpt79999/seed42/{progress,log,shards}.json`，summary / per_episode 写进本目录 `records/<名>/`。

## 七、本轮 tmux 会话清单（清理只按此清单、按确切名）

- `ev16off-w0` … `ev16off-w7`
- `ev16full-w0` … `ev16full-w7`
- `ev4cnt-w0` … `ev4cnt-w7`

smoke 不起 tmux，由 agent 会话直接后台起。现有的 `0`、`1`、`codex*`、`claude-private`、`orig80k-*` 等会话一律不动。

## 八、盯盘项

- 24 份驱动日志各挂一个 Monitor：`tail -n +1 -F <log> | stdbuf -oL tr '\r' '\n' | grep --line-buffered -E "EXIT_CODE=|EVAL_RC=|Traceback|Error|out of memory|ErrorIncompatibleDriver|server 提前退出|Killed"`。
- GPU 用 `nvidia-smi -lms 500`（显存 + 利用率）流式采样，写入 `records/gpu.csv`。
- 失败处置：先定位原因。只有基础设施故障（端口、Vulkan、OOM）才同参数重跑该片，由 eval.py 按 `progress.json` 续评，并记录原因与次数；绝不为成绩重跑。

## 九、验收判定项

- `PARAM_TREE_EXACT=PASS` ×3（已过，见第二节）
- smoke：`EXIT_CODE=0` ×3，GPU 0 峰值显存 < 75 GB
- 正式：`EXIT_CODE=0` ×24
- `merge_eval_shards.py`：三个模型 DONE，分别 800 / 800 / 200 集；`SPLIT_SEED_MATCH=PASS` ×3
- `VIDEOS=PASS`：每个模型的 mp4 数等于集数
- `CMP_FULL16_VS_OFFICIAL`（16 任务）、`CMP_COUNT4_VS_OFFICIAL`、`CMP_COUNT4_VS_FULL16`（后两组只比 4 个 counting 任务）
