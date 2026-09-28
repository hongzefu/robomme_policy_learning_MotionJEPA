# eval-orig40k-ckpt40000 — 起跑记录

**环境 B（AWS 单机 8×A100-SXM4-80GB，`/dev/md0` 本地 NVMe RAID）**。正式评估，按 AGENTS 第 12 条留档。本轮是 [eval-orig80k-modul-vs-official](../eval-orig80k-modul-vs-official/launch.md) 的续评：同样两条 run，但换成第 **40000 步**的 checkpoint，其余口径全部沿用。

## 一、目的与用户指令原话

- 上一轮的三个模型评的都是第 79999 步（80k）的 checkpoint。用户看完分任务结果后，要求：「再评一下两个run的40k checkpoint」。
- 目的：拿到两条 run 训练到一半（40k）时的分任务成功率，与同 run 的 80k 和官方 modul 80k 对比。

## 二、被评对象

| 评估名（RUN_NAME） | checkpoint 目录 | 任务 | 集数 |
|---|---|---|---|
| `eval-orig40k-full16` | `v1-store/train-runs/mme_vla_suite/v2-orig-16task-pub1600ep-modul-b64-80k/40000` | 16 全集 | 800 |
| `eval-orig40k-count4` | `v1-store/train-runs/mme_vla_suite/v2-orig-counting-pub400ep-modul-b64-80k/40000` | BinFill, StopCube, PickXtimes, SwingXtimes | 200 |

- 两条 run 的训练 HEAD 都是 `0b904ab`，训练总长 80000 步（batch 64），完成回执 `checkpoints=[10000,…,70000,79999]`。本轮评的是其中的 40000，**不是一条单独训练 40k 步的 run**；在 40k 时学习率已处于 warmup（前 10000 步）之后的恒定 5e-5 段。
- 加载走严格快照分支。起跑前在 CPU 上做了参数树核对，结果在 `records/param_tree_{full16,count4}.json`：

```
full16 @40000  PARAM_TREE_EXACT=PASS config=mme_vla_suite history_config=perceptual-framesamp-modul.yaml yaml_sha256=823c3948e75a9335 n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0
count4 @40000  PARAM_TREE_EXACT=PASS config=mme_vla_suite history_config=perceptual-framesamp-modul.yaml yaml_sha256=823c3948e75a9335 n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0
```

## 三、代码与口径

- 评估代码没有任何改动，与上一轮 Beta `1af91f9` 相同。`655dfc6` 与本轮 Beta 之间只多了文档提交。
- test split，seed 42，max_steps 1300。
- 两个模型同时起跑，各 `MODE=stride WORKERS=8`，每卡 2 个 policy server + 2 个仿真进程。`POLICY_MEM_FRACTION=0.28`，和上一轮一样，保证每集的推理环境一致。
- 回放视频全部保留：`v1-store/evaluation/<RUN_NAME>-w<k>/ckpt40000/seed42/videos/`。

## 四、执行顺序与完整命令

smoke 在 `655dfc6`（clean）上先跑：GPU 0 两片，各跑 1 集 BinFill。smoke 通过后提交本文件作为 Beta，再在 Beta HEAD 上正式起跑。两个 HEAD 之间只差本文件与 records 里的参数树 json。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
COMMON="MODE=stride WORKERS=8 CKPT_ID=40000 SEED=42 DATASET=test POLICY_MEM_FRACTION=0.28"
T16=BinFill,StopCube,PickXtimes,SwingXtimes,ButtonUnmask,VideoUnmask,VideoUnmaskSwap,ButtonUnmaskSwap,PickHighlight,VideoRepick,VideoPlaceButton,VideoPlaceOrder,MoveCube,InsertPeg,PatternLock,RouteStick
env $COMMON RUN_NAME=eval-orig40k-full16 LOG_PREFIX=ev40full PORT_BASE=8131 TASKS_ALL=$T16 \
  CKPT_DIR=$PWD/v1-store/train-runs/mme_vla_suite/v2-orig-16task-pub1600ep-modul-b64-80k/40000 \
  bash scripts/training/legacy-eval/eval_all_shards.local.sh
env $COMMON RUN_NAME=eval-orig40k-count4 LOG_PREFIX=ev40cnt PORT_BASE=8151 TASKS_ALL=BinFill,StopCube,PickXtimes,SwingXtimes \
  CKPT_DIR=$PWD/v1-store/train-runs/mme_vla_suite/v2-orig-counting-pub400ep-modul-b64-80k/40000 \
  bash scripts/training/legacy-eval/eval_all_shards.local.sh
tmux new-session -d -s ev40-gpu "nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -lms 500 > v1-store/logs/ev40-gpu.csv"
```

## 五、本轮 tmux 会话清单

`ev40full-w0` … `ev40full-w7`，`ev40cnt-w0` … `ev40cnt-w7`，以及 GPU 采样会话 `ev40-gpu`。只按这份清单、按确切名清理；其他会话一律不动。

## 六、盯盘与验收

- 逐文件轮询 16 份驱动日志，过滤 `EXIT_CODE=|EVAL_RC=|Traceback|Error|out of memory|ErrorIncompatibleDriver|server 提前退出|Killed`。
- 验收判定：`PARAM_TREE_EXACT=PASS` ×2（已过）；smoke `EXIT_CODE=0` ×2；正式 `EXIT_CODE=0` ×16；`EVAL_ORIG40K_FULL16=DONE 800/800`、`EVAL_ORIG40K_COUNT4=DONE 200/200`；`SPLIT_SEED_MATCH=PASS` ×2；视频数等于集数。
- 对比（`compare_eval_runs.py`，逐集配对）：full16 40k vs 80k、full16 40k vs 官方 80k、count4 40k vs 80k、count4 40k vs 官方。

## 七、smoke 实测（起跑前，HEAD `655dfc6`）

两片都是 `EXIT_CODE=0`，各跑 1 集 BinFill ep0，结果都成功。`EVAL_EPISODE split=test task=BinFill ep=0 env_seed=540000` 自证行齐全。server 在 16:00:14 就绪，smoke 总墙钟 183 s，GPU 0 峰值 47,973 MiB（2 个 server）。smoke 产物已删，清洗后的日志与 GPU 采样留在 `records/smoke/`。

附注：smoke 脚本里的 awk 最初按字符串比较取峰值，误报 859 MiB，改成数值比较后得到 47,973。上一轮留档的 71,932 MiB 用数值比较复核，数值不变。
