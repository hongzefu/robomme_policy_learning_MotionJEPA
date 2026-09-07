# eval-binfill-pickxtimes —— 结果

## 结论

官方两变体在 **BinFill / PickXtimes（Counting / Temporal memory suite）** 的 hard 与 medium 两档，
val + test 两 split，单 seed 42，每格 48 集：

| 变体 | hard | medium |
|---|---|---|
| `perceptual-framesamp-context` | 4/48 = **8.33%** | 21/48 = **43.75%** |
| `perceptual-framesamp-modul` | 21/48 = **43.75%** | 31/48 = **64.58%** |
| 组间 Fisher 单尾 p | 6.5 × 10⁻⁵ | 3.2 × 10⁻² |

modul 在两档都显著优于 context，但**这个总体结论掩盖了任务级的结构**，见下。

## 一、变体差异是逐任务的，不是全局的

拆到任务级别（每格 24 集 = test + val 合并）：

| 任务 | 难度 | context | modul | Fisher 单尾 p |
|---|---|---|---|---|
| **BinFill** | hard | 0/24 = 0.0% | 1/24 = 4.2% | **0.500** |
| **BinFill** | medium | 9/24 = 37.5% | 10/24 = 41.7% | **0.500** |
| **PickXtimes** | hard | 4/24 = 16.7% | 20/24 = **83.3%** | **< 0.0001** |
| **PickXtimes** | medium | 12/24 = 50.0% | 21/24 = **87.5%** | **0.0057** |

**BinFill 上两个变体统计上毫无差异**（两档 p 均为 0.50），且都很弱；差距全部来自 PickXtimes
（hard 档 5.0 倍、medium 档 1.75 倍）。因此「modul 优于 context」必须限定到具体任务，
把两个任务平均成一个 suite 数字会把这个结构完全抹掉。

## 二、与前两轮 Imitation suite 合并：难度效应是关键判别量

配合 `../eval-hard-patternlock-routestick/` 与 `../eval-medium-patternlock-routestick/`
（同口径、同 ckpt、同 seed，仅任务与难度不同），八组数据如下：

| suite | 变体 | hard | medium | 难度效应 Fisher p |
|---|---|---|---|---|
| Imitation（PatternLock + RouteStick） | context | 0.00% | 4.17% | **0.247（不显著）** |
| Imitation | modul | 16.67% | 47.92% | 0.00099（显著） |
| Counting（BinFill + PickXtimes） | context | 8.33% | 43.75% | 6.5e-05（显著） |
| Counting | modul | 43.75% | 64.58% | 0.032（显著） |

**四组里唯有 context-on-Imitation 失去了难度效应**。其余三组降低难度都带来显著提升，是模型
正常工作的表现；而 context 在 PatternLock / RouteStick 上无论难度都接近零。

这修正了前两轮的结论边界：前两轮写的「context 在这两个任务上基本不具备能力」成立，但**不可
推广为 context 能力弱**——同一份权重在 Counting suite 的 medium 档达 43.75%，与 modul 在
Imitation medium 的 47.92% 相当。context 的问题是**在 Imitation suite 上失灵**，而非普遍无能。

## 三、判据

四批各自独立汇总，判据全部 PASS：

```
# hard / context
HARD_EVAL=DONE units=4 episodes=48/48 BinFill-test=0/12 BinFill-val=0/12 PickXtimes-test=3/12 PickXtimes-val=1/12 mean_rate=0.0833
# hard / modul
HARD_EVAL=DONE units=4 episodes=48/48 BinFill-test=1/12 BinFill-val=0/12 PickXtimes-test=11/12 PickXtimes-val=9/12 mean_rate=0.4375
# medium / context
HARD_EVAL=DONE units=4 episodes=48/48 BinFill-test=4/12 BinFill-val=5/12 PickXtimes-test=8/12 PickXtimes-val=4/12 mean_rate=0.4375
# medium / modul
HARD_EVAL=DONE units=4 episodes=48/48 BinFill-test=5/12 BinFill-val=5/12 PickXtimes-test=11/12 PickXtimes-val=10/12 mean_rate=0.6458
```

四批的 `DIFFICULTY_ONLY=PASS`（expect=hard / medium，各 checked=48 mismatched=0）、
`EP_SET=PASS`、`SPLIT_SEED=PASS`（日志自证 4/4 相符、跨 split 逐集 seed 100 集全不同）。
共 192 集，`no_video=0`，全部 `EXIT_CODE=0`。

modul 权重的参数树核对沿用 `../eval-hard-patternlock-routestick/records/modul/param_tree.json`
（同一份权重，`PARAM_TREE_EXACT=PASS n_model=61 n_ckpt=61 missing=0 extra=0`）。

## 四、一次显存事故与两处编排修正

首次起跑（16:03）两个 BinFill 单元双双失败：`EVAL_RC=139`（SIGSEGV）与 `EVAL_RC=1`、`infer n=0`，
报 `CUDA error at .../sapien-vulkan-2/src/core/buffer.cpp 251: out of memory`。

根因**不是** Counting 任务仿真需求大，而是**卡上另有用户自己的任务**：进程表显示 16 个约 740 MiB
的进程来自另一仓库 `/data/hongzefu/robomme_benchmark_MotionJEPA` 的
`scripts/data-generation-newSeed/generate_dataset_newseed.py`（17 个子进程、两卡各约 6 GB），
恰在本轮起跑前 12 秒启动。叠加本轮 2 × policy(18.6 GB) 后 GPU1 达 44341 / 46068 MiB、仅剩 1.7 GB。

处置：立即停掉评估单元让出显存（GPU1 降至 24.8 GB，随后全部释放），清空全部半截产物，
等用户任务结束后重跑。用户另有一个 `plrs-test` 会话不在本轮清单内，按 AGENTS 第 7 条未作处理；
两次 `kill-session` 均逐个指名、删前删后各查一次 `tmux ls` 并核对差集。

**编排修正两处**：
1. **起跑前必须先 `nvidia-smi` 查实际占用再定并发**，不照搬上一轮档位。重跑前实测 GPU0 1017 MiB、
   GPU1 9 MiB；后续三批起跑前同样各查一次。
2. **两个 BinFill 单元拆到不同卡**，每卡配一个 PickXtimes——首次事故中 BinFill 两个同压 GPU0
   双双崩溃、而 GPU1 的两个 PickXtimes 存活，提示 BinFill 仿真更吃显存。

重跑后 16 个单元全部 `EXIT_CODE=0`。

## 五、资源与耗时

| 批 | 变体 | 难度 | 起跑 commit | 起跑时刻 | 端口 |
|---|---|---|---|---|---|
| 1 | context | hard | `f612792` | 16:21:43 | 9403 / 9423 / 9413 / 9433 |
| 2 | context | medium | `f612792` | 16:31:22 | 9442 / 9462 / 9452 / 9472 |
| 3 | modul | hard | `3ea8b8a` | 16:42:10 | 9483 / 9503 / 9493 / 9513 |
| 4 | modul | medium | `3ea8b8a` | 16:52:37 | 9522 / 9542 / 9532 / 9552 |

每批四单元并行、2 × RTX 6000 Ada 每卡 2 worker、`POLICY_MEM_FRACTION=0.40`，四批串行约 10 分钟/批。
16 个 tmux 会话全部自然退出，未执行任何 `kill-session`（首次事故那两次除外，已如实记于第四节）。

窗数远高于前两轮（这两个任务无 demo 前缀、episode 更长）：`infer` 296–599 次/单元，
而 PatternLock / RouteStick 是 31–138 次/单元。日志中 `add_buffer(首批>16帧) n=0` 证实
BinFill / PickXtimes 不在 `examples/robomme/utils.py::TASK_WITH_VIDEO_DEMO` 内。
`infer` 中位 70–74 ms，与前两轮一致。

## 六、局限

1. **单 seed 42**。每格 48 集、逐任务 24 集，基数不大；BinFill 的组间比较（p = 0.50）只能说
   「未观察到差异」，不能断言两变体等价。
2. **easy 档未测**。三档曲线仍缺一点。
3. **两变体是各自独立训练的 checkpoint**，差异不能单纯归因于 `integration_type` 这一处结构差别。
4. 结论限于 BinFill / PickXtimes 两个 Counting suite 任务，不推广到该 suite 的另两个
   （SwingXtimes、StopCube）——注意 `StopCube` 的 val split 难度分布是 easy 26 / medium 11 / hard 13，
   与其余任务不同，若要评它必须重新核对集号。
