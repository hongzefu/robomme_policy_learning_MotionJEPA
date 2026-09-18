# 本仓库策略评估

主入口为 `scripts/evaluation/run.sh 运行名 checkpoint绝对路径 任务 episode上限 [策略seed=7]`。本轮仅验收指定 bucket 末步 59999。旧入口 scripts/training/eval.sh 直接转发相同参数。

环境准备：`bash scripts/evaluation/setup.sh`。权重准备：`bash scripts/evaluation/download.sh`。两套环境都使用已提交的锁文件执行 frozen 安装。准备脚本将本仓库已有的 `v1-store/models/big_vision/paligemma_tokenizer.model` 复制进独立缓存并核验 SHA256；该源文件必须存在。服务端和客户端依赖隔离，全部包和评估产物位于当前 MotionJEPA 工作副本，解释器使用已固定的 NFS Python；训练 .venv 不变。

单回合从 clean HEAD 启动：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/evaluation/run.sh 新运行名 "$PWD/v1-store/models/robomme-vla-modul-60k-v1/59999" RouteStick 1 7
```

实际运行必须在 detached tmux 内，使用 PYTHONUNBUFFERED=1、pipefail、tee 和 EXIT_CODE。集群复用已分配资源，srun 显式加 --gpu_cmode=shared，工作目录指向本仓库。不可自动申请新作业，也不退回兼容渲染。

本机起跑及模型服务启动前都会检查所选 GPU 上的计算进程，占用即停止。用户已明确要求等待空闲 GPU，不与已有任务同卡运行。

测试：在已安装 uv 环境中执行 `scripts/evaluation/test_control.py` 和 `test_result.py`。来源检查额外核对官方 benchmark 锁定 SHA、editable 路径、ManiSkill fork 及 RouteStick 元数据。模型检查使用当前源码形状与 Orbax 元数据，不跳过多余参数。

产物位于 v1-store/evaluation/<运行名>。EVAL_PASS 表示指定回合正常完成且视频完整可解码，任务成功由 task_successes 单独报告。error 不能作为正常失败通过验收。不得覆盖运行名或换 seed 挑选成功结果。

## test/primary 700 条评测（2026-09-18 起）

评测集换成 fork `hongzefu/robomme_benchmark_MotionJEPA` 的注入候选，**不是官方 test split**。
submodule 锁在 `b4e97f2`（从 `newtask-v2.1refractor` 的 `77681e1` 切出，加了 `make_env_for_spec` 与
`robomme.injection_candidates`）；fork 上 `PolicyEvalThirdParty-v2-eval-0917` 与 `PolicyEvalThirdParty-v2-eval-0918-motion`
两个分支都指向这一个 commit（motion 轮未改 benchmark 源码，只按「`PolicyEvalThirdParty-<主仓库任务分支>`」机制建名）。

### 700 条是什么

权威载体是 `third_party/robomme_benchmark/artifacts/injection/20260912-contract-v3-10/candidates/candidates.jsonl`，
筛 `split=="test" and role=="primary"` 得 700 条 = 14 个「任务/难度」组 × 50：
BinFill{easy,medium,hard}、RouteStick{easy,medium,hard,xhard}、VideoUnmaskSwap{easy,medium,hard,xhard}、
VideoRepick{easy,medium,xhard}。每条自带完整 `episode_spec`，header 自带 `sampling_config` 与
`runtime.kwargs`，一个文件就能建全部 700 个环境。

**唯一键是 `(task, difficulty, episode)`**：四个难度组共用同一批 episode 号，同一 episode 号在不同
难度下 seed 还完全相同（seed 只是 task+episode 的函数）。所以 `progress.json` 的组键是
`"任务/难度"`，不是任务名——按 `(task, episode)` 索引会把 700 条压成 204 条互相覆盖。

### 入口

```bash
# 1) 出分片计划（10 片 × 70 集，每片 14 组各 5 条）与探针计划
uv run ... scripts/evaluation/make_shard_plan.py shards v1-store/evaluation/<run>/plans
uv run ... scripts/evaluation/make_shard_plan.py probe  v1-store/evaluation/<run>/plans

# 2) 跑一个分片（本机需显式 CUDA_VISIBLE_DEVICES；集群由 sbatch 调）
bash scripts/evaluation/run_shard.sh <运行名> <ckpt绝对路径> <分片计划json> [seed=7] [分片标识]

# 3) 合并 10 片
uv run ... scripts/evaluation/merge_shards.py v1-store/evaluation/<run>
```

集群侧：`gl_eval_scan.sbatch`（四个资源探针，档位由提交行覆盖）→ 定档 →
`gl_eval_shard.sbatch --array=0-9`。两者都带 `EXPECTED_GIT_HEAD` 起跑闸，**array 排队期间主仓库
不能 commit、也不能留未跟踪文件**。

### 与单回合入口的关系

`run.sh` / `check_result.py` **一个字没改**，它们承载 `vail-eval-gl-20260918T0400Z` 那份已绿留档的
可复现口径。`eval.py` 的官方 `env_metadata` 路径行为不变，`Args.max_steps` 默认仍是 1300；
注入候选路径由 `run_shard.sh` 显式传 `--args.max_steps=2000`（用户 2026-09-18 定，因为 oracle 在
VideoUnmaskSwap/xhard 最大要 1312 步、BinFill/hard 真实最大 1152 步，1300 会误判 timeout）。
**两个口径的成功率不可直接相比。**

### 三个坑与对应处理

1. **Vulkan 静态 TLS**：每次 `make_env`+`close_env` 净泄漏 64 字节，默认第 28 次必崩
   （`docs/training-doc/tic-vulkan-makeenv/result.md`）。两层处理：`run_shard.sh` 把客户端分块，
   每个 `eval.py` 进程只新评 ≤20 集（`--args.max_new_episodes`，靠 `progress.json` 续跑衔接）；
   再给客户端那一行加 `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=65536` 做纵深防御。
   server 不建 Vulkan Context，不加这个变量。
2. **端口 TOCTOU**：`/dev/tcp` 探测是先探后 bind，同节点两片可能同时认为空闲，历史上造成过
   「连上别人的服务、12 秒 EXIT_CODE=0、跑了 0 集」。`run_shard.sh` 按 array 下标错开端口基准，
   并在 healthz 通过后读 `/proc/net/tcp` 的 inode 与 `/proc/$SERVER_PID/fd` 比对，确认监听者是自己。
3. **`--frozen` 不校验依赖**：它只是「不更新 lock」（`--locked` 才校验），叠加 `--no-sync` 后
   缺依赖要到 `import` 才炸。fork 的根 `pyproject.toml` 多了 `pebble` 主依赖，所以切 gitlink 后
   重新 lock 并提交了 `client/uv.lock`。

### 验收

`check_shard.py` 断言计划全覆盖、取值 ∈ {bool, "error"}、**视频数 == 非 error 集数**
（`success_flag=="unknown"` 与抛异常两条路都不落视频，都记 `"error"`），并逐帧解码每个 mp4。
`merge_shards.py` 断言 10 片并集 700、两两不交，给出 14 组成功率、宏平均（14 组均值）与
微平均（成功数/700），error 条目另写 `retry_plan.json` 供单 job 补跑。

## motion 轮（2026-09-18 起，分支 `v2-eval-0918-motion`）

被评权重换成 bucket `HongzeFu/robomme-vla-modul-motion-80k-v1` 的 `50000`（run `v2-1600ep-m8x8-modul-motion-b128-80k`，
`motion.enabled=true`、`motion.budget=160`），口径与 `primary700-gl` 基线逐项相同（同一套 10 个分片计划、seed 7、`max_steps=2000`）。
`src/` 里的在线 motion 路（`FrameSampMemory` 滚动缓存 + `MotionEncoderClient` sidecar）本来就在本分支，评估侧只补了两个口子：

- `MMEVLA_MOTION_ONLINE_GPU`：sidecar 的卡号，`run_shard.sh` 自动设成本进程可见的第一张卡（Slurm 单卡 step 里是 0），
  覆盖 run 快照写死的 `motion.online_gpu: 4`（训练机的卡号）。
- `MMEVLA_MOTION_PROV_RELAX=gpu_name,compute_cap,sm_count`：sidecar 与 run 内 `motion_provenance.json` 逐键比对时只放行这
  三个硬件键（训练 A100 → 评估 A40 / Ada 必然不等；driver / torch / cuda / cudnn / diffusers 已核实逐值相同），不等改为打
  `MOTION_PROV_RELAXED` 行进 server.log；其余键仍硬拒。用户 2026-09-18 批准。

多出来的三份资产（`setup.sh wan` + `download.sh` 落位，`scripts/assets/fetch_assets.py verify --assets wan_vae,motionjepa_ckpt,motionjepa_config` 核验）：

| 资产 | 落点 | 来源 |
|---|---|---|
| sidecar venv（torch 2.9.0+cu128 / diffusers 0.39.0） | `v1-store/venvs/wan` | `scripts/dataset/wan/uv.lock`，NFS 3.11.14 解释器 |
| Wan2.1 VAE | `v1-store/cache/hf/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers` | 本机 `/data` 同路径缓存复制 |
| MotionJEPA encoder `checkpoint_epoch_72.pt` + `config.yaml` | `v1-store/external/motionjepa/wan-full1600-filter2-b176x4-72ep-a/` | bucket `HongzeFu/motionjepa-wan-full1600-72ep-v1`，按其 `SHA256SUMS.pre.txt` 核验 |

入口全部经环境变量切换，非 motion 轮默认值不变：

```bash
bash scripts/evaluation/download.sh HongzeFu/robomme-vla-modul-motion-80k-v1 50000 "$PWD/v1-store/models/robomme-vla-modul-motion-80k-v1"
POLICY=perceptual-framesamp-modul-8frame-8x8-motion POLICY_CONFIG=mme_vla_suite_b128_80k EXPECT_CKPT_ID=50000 \
EPISODE_WALL_S=2400 EVAL_TIMEOUT=14400 MMEVLA_MOTION_PROV_RELAX=gpu_name,compute_cap,sm_count \
bash scripts/evaluation/run_shard.sh <运行名> "$PWD/v1-store/models/robomme-vla-modul-motion-80k-v1/50000" <分片计划json> 7 <分片标识>
```

集群侧不再发 array，而是在**既有** gpu-hold 作业里直接跑（`gl_hold_queue.sh`，登录节点 detached tmux 内对一个 jobid 串行
`srun --jobid --overlap --exact --gpu_cmode=shared` 若干分片；`gl_eval_shard.sbatch` 用 `SHARD_INDEX` 代替 array 下标）。
`EPISODE_WALL_S` 只兜死锁不属口径：sidecar 每窗约 1.5 s（A100 实测）会把 2000 步的集推近默认 900 s，motion 轮放宽到 2400 s。
