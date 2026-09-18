# primary700-scan 启动留档

资源探针阶段：把 10 路 array 的 CPU / MEM 档位压到实测下限，同时完成三件附带验收——
在 A40 上验 Vulkan 静态 TLS 修法、预热 JAX 编译缓存、冒烟确认这 700 条候选**能被 step**
（fork 只对它们做过 make → reset → close，从未跑过完整 rollout）。

## 评测集与模型

评测集是 fork `hongzefu/robomme_benchmark_MotionJEPA` 的注入候选 test/primary 700 条，
**不是官方 test split**。权威载体
`third_party/robomme_benchmark/artifacts/injection/20260912-contract-v3-10/candidates/candidates.jsonl`，
筛 `split=="test" and role=="primary"` 得 700 条 = 14 个「任务/难度」组 × 50：
BinFill{easy,medium,hard}、RouteStick{easy,medium,hard,xhard}、
VideoUnmaskSwap{easy,medium,hard,xhard}、VideoRepick{easy,medium,xhard}。
`identity_sha256=3b4de03a0b46…`、`contract_sha256=bfcf4c5984fb…`、`sampling_config_sha256=97c9af660ca5…`。

模型为 bucket `HongzeFu/robomme-vla-modul-60k-v1` 的 59999（run `v2-1600ep-m8x8-modul-b128-60k`），
8 帧 × 64 token、budget 512、modulation、motion 关闭；策略 seed 7。
`max_steps=2000`（用户 2026-09-18 定，改自默认 1300——oracle 在 VideoUnmaskSwap/xhard 最大要
1312 步、BinFill/hard 真实最大 1152 步，1300 会误判 timeout）。**与上一轮单回合基线（1300）
不是同一口径，成功率不可直接相比。**

## 来源锁定

- 主仓库分支 `v2-vail-eval-0917`，起跑 HEAD 见各 job 日志的 `git rev-parse HEAD`。
- benchmark 锁 `PolicyEvalThirdParty-v2-vail-eval-0917` 的 `b4e97f22fe007078e297205898c07c1acbc69165`
  （从 `newtask-v2.1refractor` 的 `77681e106f005f1ff93157f36b973d1005cf0ebe` 切出，
  加了 `make_env_for_spec` 与 `robomme.injection_candidates`，未触碰候选快照校验的那 7 份采样源码）。
- `.gitmodules` url 改为 `https://github.com/hongzefu/robomme_benchmark_MotionJEPA.git`。
- 客户端 `uv.lock` 重新解析：只新增 `pebble v5.2.2`（fork 根 pyproject 的主依赖），其余 235 包零版本变动。

## 四个探针

两条一维扫描，不做笛卡尔积（CPU 与 mem 的瓶颈机理不耦合）。
`OMP_NUM_THREADS` 由 `env.sh` 跟随 `SLURM_CPUS_PER_TASK`，随档位自动变，避免混淆
「档位不够」与「线程打架」。

| 探针 | CPU | MEM | 计划 | 集数 | 职责 |
|---|---:|---:|---|---:|---|
| A 基准 | 2 | 32G | probe-A.json | 32 | anon+shmem 峰值、核当量、步时基线；跨过第 28 次 make_env 验 TLS；预热 JAX 缓存 |
| B 压 CPU | 1 | 32G | probe-BCD.json | 8 | 对照 A 的稳态步时，判 1 核够不够 |
| C 压 MEM | 2 | 16G | probe-BCD.json | 8 | 16G 会不会 OOM |
| D 双压 | 1 | 8G | probe-BCD.json | 8 | 极限档，跑通就直接用 |

probe-A 覆盖全部 14 组（各 2 条 + 最重四组各再 1 条 = 32 集）；probe-BCD 只取最重四组
（VideoUnmaskSwap/xhard、VideoRepick/xhard、RouteStick/xhard、BinFill/hard）各 2 条——
内存峰值由最长的集决定，`RolloutRecorder` 把整局帧连同 demo 帧全攒在内存里。

判据：
- **CPU**：B 的稳态 `infer_ms` / `add_buffer_ms` / 每集墙钟相对 A 退化 < 10% → 取 1 核，否则 2 核。
- **MEM**：`MEM_G = max(ceil4(A_peak × 1.25 + 2), min(跑通的 C/D 档 mem))`，
  `A_peak = max(anon + shmem)`。OOM 本身是有效结论，不重跑不上调重试。
- 判 OOM 只看 cgroup `anon`(+`shmem`)，**不看 MaxRSS**（它会精确贴死 `--mem` 上限，是 NFS 页缓存读数）。
- `step_ok` 与 `demo_frames_ok` 任一不达标即停止，不发正式 array——那是评测口径问题，不是档位问题。

## 命令

A 与 B 走既有交互作业 **61331555**（gl1519，2×A40，2 CPU / 32G，2 天时限）的 srun step——
它的档位正好等于 A（2C/32G），B 只需把 `--cpus-per-task` 降到 1，内存仍是 job 的 32G，
符合 B「只压 CPU」的设计。这样 CPU 维的结论**不用排队**就能拿到（当时 spgpu 240 张 A40
已分配 238，新 job 要等）。C / D 需要更小的 cgroup 内存限制，只能走 sbatch。

⚠ gl1519 的两张 A40 是 `Exclusive_Process`，而 `run_shard.sh` 在 Slurm 下断言
`compute_mode == Default`，所以 srun **必须显式加 `--gpu_cmode=shared`**。

```bash
export REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
export CKPT="$REPO/v1-store/models/robomme-vla-modul-60k-v1/59999"
export PLANS="$REPO/v1-store/evaluation/primary700-scan/plans"

# 探针 A：2 CPU / 32G（= job 原档），32 集
ssh -o BatchMode=yes greatlakes "srun --jobid=61331555 --account=chaijy2 --partition=spgpu \
  --gpu_cmode=shared --overlap --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --gpus-per-node=1 --time=02:30:00 --chdir='$REPO' \
  /usr/bin/env -u ROBOMME_GPU_RASTER -u SAPIEN_DISABLE_RAY_TRACING \
  RUN_NAME=primary700-scan CKPT='$CKPT' PLAN='$PLANS/probe-A.json' PROBE=A \
  /usr/bin/bash scripts/evaluation/gl_eval_scan.sbatch"

# 探针 B：1 CPU / 32G，8 集（只改 --cpus-per-task 与 PLAN/PROBE）
#   --cpus-per-task=1 ... PLAN='$PLANS/probe-BCD.json' PROBE=B

# 探针 C / D：需要更小的 cgroup 内存限制，走 sbatch
sbatch --gpu_cmode=shared --cpus-per-task=2 --mem=16G \
  --export=ALL,RUN_NAME=primary700-scan,CKPT="$CKPT",PLAN="$PLANS/probe-BCD.json",PROBE=C,EXPECTED_GIT_HEAD=<HEAD> \
  scripts/evaluation/gl_eval_scan.sbatch
sbatch --gpu_cmode=shared --cpus-per-task=1 --mem=8G \
  --export=ALL,RUN_NAME=primary700-scan,CKPT="$CKPT",PLAN="$PLANS/probe-BCD.json",PROBE=D,EXPECTED_GIT_HEAD=<HEAD> \
  scripts/evaluation/gl_eval_scan.sbatch
```

sbatch 经 `scripts/training/gl_submit.py` 提交（ControlMaster 存活时零 MFA）。

## 会话与产物

（待填）

## 前置：本机两集端到端冒烟（已通过）

探针之前先在本机 RTX 6000 Ada（GPU1，空闲）跑了两集，覆盖有 / 无 conditioned demo 两条路径，
`CHUNK_EPISODES=1` 顺带验证分块循环。tmux 会话 `ev-smoke-local`，
起跑 HEAD `ccac38c`，2026-09-18 06:54:42Z → 06:59:02Z（4 分 20 秒），`EXIT_CODE=0`。

| 条目 | 结果 | demo 帧 | 视频帧 | 视频字节 |
|---|---|--:|--:|--:|
| RouteStick/xhard ep115 | fail（走 48 步） | 451 | 499 | 1,275,373 |
| BinFill/hard ep153 | success（走 879 步） | 0 | 879 | 2,673,149 |

`SHARD_PASS pipeline_pass=true episodes=2 evaluated=2 successes=1 errors=[]`。
**这是这 700 条候选第一次被完整 step**——fork 此前只对它们做过 make → reset → close，
其 `reset_check.py` 的 docstring 明写「reset 通过 ≠ 能出 h5」。计划里标注的最大未知项到此清掉。

server 侧时序（本机）：首次 `add_buffer`(451 帧) 9862 ms、首次 `infer` 2327 ms、
16 帧 `add_buffer` 首次 1331 ms、1 帧 `add_buffer` 首次 1864 ms（BinFill 无 demo，
`exec_start_idx=0` 又是一个新形状）；**稳态 `infer` 68 ms、稳态 `add_buffer` 29 ms**。
58 次 infer × 16 步 = 928 步，与两集实走的 48 + 879 = 927 步吻合。
JAX 编译缓存 7 → 9 条（7.3M → 8.4M）：**每个不同的 demo 长度都是一次新编译**，
这正是让探针 A 覆盖全部 14 组的意义——把各组的形状先编出来给 10 路 array 用。

冒烟过程中打死两个 bug，已分别提交（见 `ccac38c`）：
1. 端口归属断言误报——`$!` 拿到的是 wrapper，实测进程链 env → uv → python 三层，
   socket fd 挂在孙进程上，改为遍历整棵进程树后 `SERVER_READY ... own=1 tree=1069052 1069064`。
2. `setup_log_dict()` 的「重启即重试 error」与分块跑相冲突：块数按计划集数固定算，
   删掉 error 会让下一块重跑它而不是推进，计划末尾的集永远轮不到。改为注入候选路径下
   error 是终态，补跑交给 `retry_plan.json`。

冒烟产物已按 AGENTS 6 清理，关键日志与 JSON 存于本目录 `records/`。
