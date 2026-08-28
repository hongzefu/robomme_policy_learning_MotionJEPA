# v1：GPU 利用率冲刺 ≥95% —— 先修量具重立基准（L0），配置收敛（L1）随后，代码级优化（L2–L4）按决策门后置

> **本文件是本轮（2026-08-28 起，Codex 审计后重排版）的单一权威计划文档。** 它接续 v2 计划 S8b 的
> 收官结论与 [`docs/training-doc/v1-framesamp-e2e/result.md`](docs/training-doc/v1-framesamp-e2e/result.md)
> 第五节待办 3，并吸收 Codex 审计（锚点 `1a76fef`）的核心发现后**全面改写**：
> 先修量具、按生产口径重立 speed 基准，而不是继续改 DataLoader。
>
> **与既有文档的关系（只引用、不复制）**：
> - IO 重构本体、G0/G2 定义、红线表（G 节 R1–R17）、白名单（T1）、基线链登记簿（T8）
>   的权威源是 [`v2-framesamp-restructure-plan.md`](v2-framesamp-restructure-plan.md)；本文件只引用其条目编号。
> - L3/L4 的 scope 权威源是 [`v1-post-restructure-roadmap.md`](v1-post-restructure-roadmap.md) 的项 2/项 3；
>   本文件只补该文件未覆盖的实施坑，不重述其规格。
> - S8b 各档实测数字与 epoch 口径缺陷的权威源是上述 `result.md`。
>
> **目标**：GL 4×A40 e2e 的 dense 稳态 **util 均值 ≥95%**（用户 2026-08-28 指定），
> 且全程可证明与 G0 训练语义逐位等价。

## 量具缺陷发现（本次改写的起因）

Codex 审计发现一个此前所有性能结论共同踩着的地基缺陷：

- GL 性能 benchmark 的入口 `scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch` 在 `run_bench()` 里
  **硬编码 `--log-interval 1`**；
- `--log-interval 1` 迫使 `scripts/train.py::main` 的日志分支**每步**执行一次
  `jax.device_get(stack_forest(infos))`——阻塞主线程直到 GPU 算完本步，然后才 `next(data_iter)`；
- 而正式训练的默认值是 **`log_interval=100`**（`src/mme_vla_suite/training/config.py::log_interval`）。
  已核实 `log_interval>1` 时非日志步**没有任何** `device_get` / `block_until_ready` 同步点，
  取数与 GPU 计算之间只剩 JAX 异步队列的天然背压。

也就是说：**现有 89.2% util 基线是「诊断量具主动打断流水线」之后测出来的数字，不代表生产形态。**
第一节诊断出的「取数时间完整暴露成 GPU 空窗」机制真实存在，但它在生产口径（log100）下的
**幅度是未知的**——很可能大部分空窗在 log100 下已被异步队列天然藏住。

因此本轮重排为：**L0 先修量具并按两档（log1 对照 + log100 生产）重立 speed 基准**；
此前的一切性能族量具口径与 speed 链对齐**全部作废**（见 0.3）；原 L0/L1/L2/L3 顺延为 L1/L2/L3/L4。

---

# 第一部分（给人看）

## Context：为什么做这件事

v2 计划的 packed IO 重构已经收官——正确性侧 G2 对拍 bitwise 全过，性能侧 S8a/S8b 全部 run 跑完，
真实墙钟口径下 packed 比 legacy 快约 30%（`result.md` 四节 4.2）。但绝对目标没达到：
旧量具口径下 GPU 利用率停在 89.2%（w8），真实 epoch 9.28 h 距 8.6 h 阈值差约 7.4%。

本轮 Codex 审计确认：**这些数字全部是 log_interval=1 量具口径的产物**（见上「量具缺陷发现」）。
在继续做任何优化之前，必须先回答一个更基本的问题——**生产口径下缺口到底还剩多少**。
所以本计划的第一步不是改 DataLoader，而是修量具、重立基准（L0）；若生产口径直接达标，
本项目即收官；不达标再按 L1（配置收敛）→ L2（取数/计算重叠）→ L3/L4（削载荷/换通道）逐层推进。

## 〇、口径订正与基线作废

### 0.1 真正过 IPC 的 batch 是 256.7 MB

一度出现过「313.2 MB」的说法，那是**上卡后 Observation** 口径，不是 worker→主进程的 IPC 载荷。
两张 224² 原图在 IPC 这一段是 **uint8 共 19.3 MB**；f32（共 77 MB）是 `device_put` 之后
`openpi/models/model.py::Observation.from_dict` 里 `astype(f32)/255*2-1` 才产生的。
**v2 计划 1.4 节字节帐的「~257 MB」是对的。** 连带结论：原图在 IPC 上已是最窄的 uint8，无窄化空间。

真实 IPC 载荷逐键：

| 键 | host 侧 dtype | 字节 |
|---|---|---|
| `static_image_emb` | bf16 | 134.2 MB |
| `static_pos_emb` | f32 | 100.7 MB |
| `static_state_emb` | **f64** | 2.10 MB |
| 两张原图 | **uint8** | 19.3 MB |
| `static_mask` / `actions` / `state` / `tokenized_prompt` 等 | bool/f64/i32 | 约 0.4 MB |
| **合计** | | **≈ 256.7 MB** |

（`static_state_emb` 在 host 侧是 f64，上卡后显示的 f32 是 jax 按 x64-disabled 规范化的结果。）

### 0.2 COLDHOT 与「节点漂移」的说法作废

分析初期曾用**步时中位数**比较 C1（冷）与 H1（热），得出「C1 比 H1 快 11–13%、冷态惩罚为负、
疑 A40 热降频或跨节点漂移」的结论。该结论**不成立**：两侧分布形状不同（C1 有 2.9% 慢步、p90=6.681；
H1 零慢步、中位≈均值），中位数比较正是 AGENTS 16 禁止的做法。按稳态墙钟均步时，
**C1 5.224 s vs H1 5.180 s = +0.7%，COLDHOT 判据通过、冷态惩罚可忽略**（`result.md` 二节）。
cold/hot 已证明 NFS/cache 惩罚只有 0.7%——继续优化磁盘不会带来 5–6pp util。

另注（Codex 审计核实）：COLDHOT 的对比判据 `(C1稳态−H1稳态)/H1 ≤ 15%` 目前**只写在
`gl_e2e_fix.sbatch` 的注释里，sbatch 与 `analyze_gpu_util.py` 双方都没有代码实现**——跑完 COLDHOT
只得到两个独立 record_dir，判据靠人工算。本轮如实记载，不扩 scope 去实现它。

### 0.3 性能族基线与旧量具全部作废（本次新增）

**作废清单**（一律降为「历史参照」，不再作为任何对齐/对比的基线）：

| 作废项 | 原地位 | 作废理由 |
|---|---|---|
| `v1-g0-speed-r2` 锚点（1.152 s/step、util 86.5%） | 本机 speed 锚点，V3 先导对比基线 | log1 量具口径 |
| S8b 各档数字（w8 89.2% / w12 89.5% / w16 等） | GL 性能基线 | 同上 |
| `v1-e2e-b64`、`v1-e2efix-w{8,12,16}c16`、`v1-framesamp-e2e`、`v1-framesamp-dl` 的 util/步时结论 | GL e2e 性能族 | 同上 |
| 旧 `E2E_ACCEPT` 五项判据（`util_mean_pct≥90` 等，`analyze_gpu_util.py::ACCEPT_THRESHOLDS`） | GL 验收判据 | 换 `E2E95_ACCEPT` 六项（见 3.3） |
| `E2E_EXTRA`（w4/w8 步时**中位**差 3%） | 附加判据 | 中位口径，违反 AGENTS 16 |
| 性能族 speed 链的一切「对齐」要求（无 TrainState 摘要、无 batch_digests、生产 XLA 档的历史可比性） | 跨轮对比 | 量具口径已换，重新开始 |

**新 speed 基准由 L0 重立**（见 2.0）。T8 登记簿中上述条目的作废标记留待 L0 实施收官后回填 v2 计划。

**【硬性原则，用户两次强调】正确性族一项不减、原样保留**：G0/G2 确定性档 bitwise 对拍链
（200 batch shape/dtype/raw bits 零失配、1000 步五标量逐步 hex、`state_digest`、canonical batch
digest、index 序列判据）是全程**不漂移**的关键保证。正确性族 run 照旧跑 `log_interval=1`
确定性档——它的量具、判据、固化产物、`scalars_hex.tsv` 的 sha256 锚
（`c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757`）本轮完全不动。
废弃范围严格限定在性能族；任何量具修改不得削弱正确性族判据。

> 为什么正确性族不需要 log100：`train.py` 日志步记录的是 `stack_forest` + `jnp.mean` 的**区间均值**，
> log100 下天然没有逐步五标量。正确性族维持 log1 即可完整保留逐步 hex 证据链，且正确性 run 的
> util/步时本来就禁作性能结论（T9-B5），量具打断流水线对它无害。**因此无须为取证动 `scripts/train.py`。**

## 一、诊断：缺口到底在哪里

> **限定框（本次新增）**：本节全部数字为 **log_interval=1 旧量具口径**下的实测。其中的**机制结论**
> ——空窗落在步头、慢步周期恒等于 num_workers、IPC 通道约 520 MB/s、H2D 只占 2%、compute-only
> 可到 99.9%——仍然有效；但**幅度**（89.2%、10.8pp 缺口、270 s 总延迟等）是量具每步打断流水线
> 之后的数字，生产口径（log100）下的真实缺口待 L0 重测。本节保留作为机制档案与 L2–L4 的立项依据。

以下全部为本轮实测，数据源是 `v1-store/bench/bottleneck/` 下各 run 的 `metrics.jsonl` 与
`gpu_util_dense.csv`，复现命令见第二部分 D 节。

### 1.1 天花板不是 90%

`v1-computeonly-b64` 走的是**同一套训练循环**（同样每步 `jax.device_get`），唯一差别是
`create_data_loader` 被换成「首 batch 缓存后无限重复」的 wrapper
（`scripts/bottleneck-bench/gl-compute-only/compute_only_train_steps.py::_RepeatFirstBatchLoader`），
于是 `next(data_iter)` 变成零成本。结果：**util 均值 99.9%、0% 采样 0.0%**。

→ 计算图、FSDP、每步的 `device_get` 同步屏障**都不是**利用率缺口的原因。缺口只可能来自 `next(data_iter)`。

### 1.2 快步 / 慢步分层：缺口的 84% 集中在 25% 的步上

把稳态窗口按「步时 ≤ p10×1.15」分成快步与慢步，再分别做步内相位分析（把每个 dense 500 ms 采样点
按它在所属步中的相对位置分桶），`v1-framesamp-e2e-w8c16`：

| 分层 | 步数 | 墙钟占比 | 步时均值 | util 均值 | 0% 采样 | 空窗落点 |
|---|---|---|---|---|---|---|
| 快步 | 409（75%） | 68.1% | 4.948 s | **97.4%** | 1.5% | 相位 0.0–0.2 轻微（81%→93%→100%） |
| 慢步 | 140（25%） | 31.9% | 6.767 s | **71.6%** | 27.4% | 相位 0.0–0.3 深度停摆（27%→1%→8%） |

- 慢步贡献的缺口 = 28.4pp × 31.9% 墙钟 = **9.1pp**，占总缺口 10.8pp 的 **84%**。
- 两个分层的空窗**都落在步的开头**，步的后半段（快步相位 ≥0.2、慢步相位 ≥0.4）是**恒定 100.0%**、
  零个 0% 采样。GPU 计算段本身干净利落。

w12c16 同构（快步 97.9% / 慢步 71.2%），w4c16 因 worker 更少而快步也被拖到 92.0%。

### 1.3 慢步严格周期性，周期恰等于 num_workers

| run | 慢步间隔模式 | 周期和 | 慢步占比 | 慢步超快步基线的总延迟 |
|---|---|---|---|---|
| w4c16 | 4,4,4,4,… | 4 = W | 29.9% | 293 s / 3067 s |
| w8c16 | 5,3,5,3,… | 8 = W | 25.5% | 269 s / 2971 s |
| w12c16 | 5,5,2,5,5,2,… | 12 = W | 25.0% | 269 s / 2964 s |

**周期恒等于 worker 数、比例恒约 25%、总延迟恒约 270 s**——这是主进程与 worker 之间数据交接的节奏特征
（torch DataLoader 的 round-robin 取数 + `prefetch_factor=2`），既不是 NFS 抖动（S8a 实测 w8 的
dataloader 吞吐是训练需求的 3.4×），也不是 worker 产能不足。

### 1.4 交接为什么慢：IPC 通道只有约 520 MB/s

本机微基准（fake dataset，worker 只造零数组，零 IO、零反序列化；batch 形状/dtype 逐键照抄 GL 日志的
`Initialized data loader` 段）：

| 环节 | 实测 | 等效带宽 |
|---|---|---|
| worker→主进程 IPC（num_workers=4 / 8） | 591 / 601 ms | **约 520 MB/s** |
| 折算真实 IPC 载荷 256.7 MB | **约 490 ms** | 同上 |
| `jax.make_array_from_process_local_data`（H2D） | **9.5 ms** | 33 GB/s |

根因：`openpi/training/data_loader.py::_collate_fn` 返回的是 **numpy** 数组，而 torch 的
shared-memory 零拷贝 reduction 只对 `torch.Tensor` 生效，于是整批走标准 pickle + 管道，
并且在**主进程单线程**完成反序列化。这解释了两件事：为什么 w4 与 w8 的 IPC 耗时完全一致；
为什么加 worker 对 util 毫无帮助。**H2D 只占 2%，不是问题。**

### 1.5 根因（log1 口径下）：主循环里的串行结构 + 量具强制逐步同步

`scripts/train.py::main` 的 `for step in pbar:` 循环体，关键三步的现有顺序是：

1. `ptrain_step(...)` —— 异步 dispatch，立刻返回；
2. `jax.device_get(reduced_info)` —— **仅日志步执行**，阻塞直到 GPU 算完本步；
   `log_interval=1` 时即**每步**执行；
3. `batch = next(data_iter)` —— **屏障之后**才去取下一个 batch，此时 GPU 完全空转。

log1 下取数（含慢步那 ~1.9 s 的交接阻塞）100% 暴露在 GPU 空窗里，与计算零重叠。
**但 log100 下第 2 步 100 步才发生一次**，其余 99 步取数与 GPU 计算之间由 JAX 异步队列衔接
——这正是「生产口径下缺口可能远小于 10.8pp」的原因，也是 L0 必须先重测的原因。

### 1.6 顺带查出的一件事：roadmap 项 1 的前提不成立

v2 计划 1.5 节第 5 条与 roadmap 项 1 都以「每个 spawn worker 在 GPU0 建约 442 MiB CUDA context」
为前提。本轮核对 GL packed run 的 `compute_apps.csv`（`nvidia-smi --query-compute-apps` 采样存证）：
**全程只有主进程一个 pid**，worker 侧 CUDA context **没有出现**。

→ roadmap 项 1 应从「性能补救首选项」上撤下，其立项前提需重新采集后再议。

## 二、五层方案（L0 / L1 / L2 / L3 / L4）

五层按「改动量」从小到大排列。**先给结论**：L0 是本轮唯一授权实施的层——修量具、重立基准；
它的 log100 档若直接过 `E2E95_ACCEPT`，本项目立即收官。L1 是配置层收敛，L2–L4 是代码级优化，
全部按决策门后置（见五节）。

| 层 | 一句话 | 改哪些文件 | 性质 | 与 G0 的对齐强度 |
|---|---|---|---|---|
| **L0** | **修量具 + 两档重立基准** | 3 个量具文件（`analyze_gpu_util.py`、`bench_train_steps.py`、`gl_e2e_fix.sbatch`） | **本轮唯一授权** | 不触训练语义；性能模式默认关闭、log1 路径逐字节不变 |
| **L1** | 配置打包收敛（log100 + w8 + 线程=1 + prefetch=4） | sbatch 环境变量 + `dataloader.py` 一个形参透传 | 决策门后置 | prefetch 透传须补 index 序列对拍（R3） |
| **L2** | 让取数与 GPU 计算重叠 | 1 个文件（落法不同则不同） | 决策门后置 | ✅ **逐位 bitwise** |
| **L3** | 不传 pos/state，GPU 侧查表 | **9 个 `src/` 文件**（5 个在红线内）+ 4 个验证资产 | 决策门后置 | ⚠ 输出侧 bitwise **不可得** |
| **L4** | IPC 换 shared memory 通道 | 1 个自建 loader + **改动量具本体** | 决策门后置 | ⚠ 理论可得，**但尺子要换** |

---

### 2.0 L0：修量具 + 重立基准 —— 本轮唯一授权实施的层

#### 2.0.1 量具现状缺陷与逐条修法（Codex 审计核实）

| 现状（已核实） | 修法 |
|---|---|
| `analyze_gpu_util.py` 的 `epoch_h`/吞吐用 `step_med`（中位数），与自己文件头「禁止中位数标题结论」及 AGENTS 16 自相矛盾 | epoch/吞吐/主步时改**稳态均值 + 真实 elapsed 反推**；中位数降为附列 |
| `ACCEPT_THRESHOLDS` 旧五项（`util_mean_pct≥90` 等），标记 `E2E_ACCEPT`；另有中位口径的 `E2E_EXTRA` | 新增 **`E2E95_ACCEPT` 六项**（见 3.3）；旧标记与 `E2E_EXTRA` 废弃 |
| 无 active_util（非零采样条件均值）、无 `step % workers` 分组、真实 elapsed 算了（`t_hi - t_lo`）但不输出 | 三项全部新增输出 |
| `EPOCH_STEPS=6176`、吞吐里的 batch=64 硬编码 | 从 `env.json` 取 |
| `bench_train_steps.py::main` 对 `log_interval != 1` 直接 `raise ValueError` | 新增**性能模式**（显式环境变量开启）放行 `--log-interval 100`；默认行为不变、正确性模式仍强制 1 |
| `gl_e2e_fix.sbatch::run_bench` 硬编码 `--log-interval 1` | 改 `LOG_INTERVAL=${LOG_INTERVAL:-1}` 可覆盖、透传并写入 `env.json` |
| 全链路无 `prefetch_factor` 参数（sbatch、`dataloader.py` 均无） | 不属 L0——是 L1 的前置改动（见 2.1） |
| COLDHOT 判据无代码实现（见 0.2） | 如实记载，不扩 scope |

**log100 不得为逐步计时重新引入每步 `device_get`**；末段不足 100 步的指标必须强制 flush
（`train.py` 现有逻辑在 `step == num_train_steps-1` 非日志步时会丢掉尾段 infos——分析器须
以真实 elapsed 为准，不依赖末行 metrics；实施时核实尾段行为并在留档写明）。

#### 2.0.2 log100 档的口径限制（必须写清，避免误读）

log100 下 `metrics.jsonl` 每 100 步一行、记录的是**区间均值**，没有逐步步时。因此：

- **慢步分层、`step % workers` 相位分组只能在 log1 对照档做**；log100 档做不了也不需要做；
- log100 档的 `slow_wall_pct` 用 dense NVML csv（500 ms 采样）的**低 util 连续段**近似判读；
- **util 三项（均值 / 0% 采样 / active_util）来自 `gpu_util_dense.csv`，与 metrics 粒度无关，
  两档口径完全一致**——这是两档可以互相印证的地方；
- log100 档的步时均值 = 稳态窗口真实 elapsed ÷ 稳态步数，**这是新的主步时口径**。

#### 2.0.3 基准两档 run

量具修完后，**同一配置**（w8c16、packed backend、prefetch=2、线程现状 OMP/MKL=16）跑两档：

| 档 | LOG_INTERVAL | 作用 |
|---|---|---|
| **log1 对照档** | 1 | 与旧 89.2% 数字同口径可比，**把量具差异单独归因出来**（新旧 analyzer 对同一 run 双跑核对） |
| **log100 生产档** | 100 | **新 speed 基准**，判 `E2E95_ACCEPT`；达标即收官 |

两档各 600 步、4×A40/16C/96G，`ANALYZE_ACCEPT=1` 只挂 log100 档。共用一次申请或分两 job 实施时定。

---

### 2.1 L1：配置打包收敛 —— L0 未达标时的第一步

**改的是什么（人话）**：不改任何训练逻辑，只把四件配置一次打包收敛（用户 2026-08-28 拍板打包跑，
不逐项单变量）：

1. **`LOG_INTERVAL=100` 沿用**（L0 生产档口径）；
2. **`WORKERS=8` 固定**。依据 S8b：w8→w12 只 +0.3pp 且 w12 慢步墙钟升到 10.7%——加 worker
   已无增益且引入慢步（此机制结论不受量具口径影响）；
3. **CPU 线程池收敛为 1**：现状 sbatch 把 `OMP_NUM_THREADS`/`MKL_NUM_THREADS` 设为 `$NCPU`（=16），
   8 worker + 主进程共享 16 核、潜在 16×16 线程超订；改默认 1，并**补设现状完全缺失的
   `OPENBLAS_NUM_THREADS` / `NUMEXPR_NUM_THREADS`**；
4. **`prefetch_factor` 2→4**：需先给 `src/mme_vla_suite/training/dataloader.py` 加形参透传
   （该文件不在 R2 不动清单，v2 B.0 表列于「修改·接线」行；torch 默认 2，现链路无处可调）。
   `in_order` 恒 True，**禁止 False**——False 会改变 batch 到达顺序与优化器更新轨迹，无法与 G0 对齐。

**约束与回退**：w8 在途 batch 上限 16→32，按 ~257 MB/batch 名义增量约 4.1 GB；要求 MaxRSS <80 GB、
`meminfo.csv` 无 major fault 激增。prefetch 一项收益不足（util <1pp 或步时 <2%）即回退 2。

**对齐要求**：第 3/4 两项不触任何训练语义；第 4 项因动了 loader 装配代码，须补 **index 序列对拍**
（R3：共同前缀逐位不变，尾部超前量只增不重排），并补两 epoch 实际消费 index 对拍。

**预期（Codex 评估）**：线程收敛收益 0–4pp、中等置信。原文虽主张单变量，用户已拍板打包一个 GL run
省排队；若打包后达标但需归因，可事后补拆解档，默认不安排。

---

### 2.2 L2：让取数与 GPU 计算重叠 —— 代码级修复的首选（原 L1 顺延）

**改的是什么（人话）**：log1 诊断显示主循环是「发计算任务 → 等它算完 → 再去拿下一批数据」。
L2 把「去拿数据」挪到「等它算完」之前（或交给一个后台线程），让搬数据和算数据同时发生。
**不改任何一个数、不改任何一个张量、不改喂给模型的顺序**，只改「什么时候去搬」。

> ⚠ 顺延后的重要限定：L2 的立项依据（1.2/1.3/1.5 节）是 log1 口径的。log100 下日志同步 100 步
> 才一次，队头等待的暴露程度未知——**L2 是否还有立项价值，完全取决于 L0/L1 的实测结果**。

#### 2.2.1 三种落法，各自改哪些文件（原 L1a/L1b/L1c 顺延更名）

| | **L2a** | **L2b（若进入 L2，首选）** | **L2c** |
|---|---|---|---|
| **改哪个文件** | `scripts/train.py` | `scripts/smoke-local/bench_train_steps.py` | `src/mme_vla_suite/training/dataloader.py` |
| **改成什么样** | 把 `main()` 的 `for step in pbar:` 里那一行 `batch = next(data_iter)` 从 `device_get` **之后**移到**之前**（合法插入点：`infos.append(info)` 之后、`if step % log_interval` 块之前） | 新增一个 monkeypatch：包住 `_train._data_loader.create_data_loader`，在它返回的 `DataLoaderImpl` 外面再套一层后台预取层（有界队列 + 单线程）。由 `BENCH_PREFETCH` 环境变量控制，**默认关闭** | 在 `DataLoaderImpl.__iter__` 里加同样的后台预取层（同样默认关闭） |
| **改动规模** | **一行位置调换** | 约一个类 + 一处安装函数 | 约一个类 + 改一个方法 |
| **谁受益** | 三条链全部（`train.py` 自身、bench、compute-only 都跑同一个 `train.main`） | **只有 bench / GL 验收链**（正式训练入口不受益） | 三条链全部（都经 `create_data_loader` → `DataLoaderImpl`） |
| **红线** | ❌ **撞 R1 + R2 双硬红线**（R2 首位点名「`scripts/train.py` 不动」），且会让 `G0_SCOPE` 断言输出非空，需显式解禁 | ✅ **T1 白名单内整目录**，且被 R2 的「验证验收资产参数化」除外条款覆盖 | ✅ 不在 R2 不动清单（v2 的 B.0 表把它列在「修改·接线」行），走三块验证即可 |
| **index 序列** | **逐位不变（n=8072）** | 尾部超前量 `pf×w` → `pf×w + k`，**前缀逐位不变** | 同 L2b |
| **实现复杂度** | 最低 | 线程 + 有界队列 + 与 `finalize()` 的 join 时序 | 同 L2b，但落在生产代码里 |

另一等价形态（Codex 提出）：不做预取层，直接在主循环把 `next(data_iter)` 移到 `device_get` 之前
——即 L2a 的一行移位，或严格保序的 host/device 双缓冲。落法选择在进入 L2 时再定。

**L2 属于训练交付路径改动，必须走 AGENTS 18 全套**：前后链路图、第一块轻量对拍、
第二块本机训练梯度一致（1000 步确定性档，判据见四节）。

#### 2.2.2 本机实测：重叠确实能把空转吃掉（log1 口径）

微基准把 GPU 计算段用 `jax.lax.fori_loop` 调到 4.258 s（与 GL 同量级），配一个与真实 batch
逐键同形同 dtype 的 fake dataset（worker 只造零数组，零 IO 零反序列化），本机 2×RTX 6000 Ada：

| 调度 | 步时均值 | max | 相对纯计算的开销 |
|---|---|---|---|
| 纯 GPU 计算（参照） | 4.258 s | — | — |
| **串行（复刻 log1 现状）** | **5.272 s** | **9.31 s** | +23.8% |
| **同线程重叠**（取数移到屏障前） | **4.629 s** | **4.74 s** | +8.7% |
| **后台线程预取** | **4.691 s** | **4.81 s** | +10.2% |

三点读数：① 省 0.64 s（−12.2%）；② **长尾从 9.31 s 塌到 4.74 s，周期性慢步完全消失**——
这正是 1.3 节那个「每 W 步一次」的交接阻塞被藏进计算的直接证据；③ 两条路径效果相当，
说明 **GIL 不构成障碍**（主线程等 GPU 时会释放 GIL，预取线程能拿满）。

> 口径注记：该微基准复刻的是**每步同步（log1）**的串行结构；log100 生产口径下「现状」侧
> 本身就没有每步屏障，上表的收益幅度**不能直接外推**，仅证明重叠机制本身有效。

#### 2.2.3 为什么 L2 能保住逐位 bitwise

memory token 由四个因素完全决定（v2 计划三节），L2 一个都没碰：① 每 step 取哪些样本
——同一个 `data_iter`、抽取顺序不变；② 每样本选哪 32 帧——`even_sampling_indices` 不动；
③ 每帧特征的字节——packed 库不动；④ 交付 dtype——不动。

另外 `train_rng` 是循环外常量、在 `train_step` 内按 `jax.random.fold_in(rng, state.step)` 折入，
**host 侧的取数时机与记录频率对数值零影响**。因此 L2 可以直接要求最强判据：G2 那套四分项
原样复用，一行收官是 `scalars_hex.tsv` 的 sha256 仍等于 `c799a0b2…`。

---

### 2.3 L3：不传 pos/state，GPU 侧查表 —— 把 IPC 载荷砍掉 40%（原 L2 顺延）

即 roadmap 的**项 2（pos_emb 移出 batch）+ 项 3（`static_state_emb` 清除）**。roadmap 已把 scope
规格化，本轮核对确认其规格仍准确，并补出两条它没写的坑。

**改的是什么（人话）**：`static_pos_emb` 是 100.7 MB 的大张量，但它其实是「第几帧」的纯函数
——同一个帧号在任何 episode 里都是同一串字节，packed 库里已经存成一张 586 行的小表。
所以完全不必每步把它从 worker 搬到主进程再搬上卡，只要搬 32 个帧号（8 KB），
让 GPU 从常驻的小表里查出来就行。`static_state_emb` 更简单：`use_state_emb=false`，
GPU 根本不用它，纯属白算白传。

**收益**：IPC 载荷 256.7 MB → **153.9 MB（−40%）**，IPC 耗时约 490 ms → 约 294 ms。
与 roadmap 项 2 原文的「约 40%」一致。

#### 2.3.1 要改哪些文件（9 个 `src/` + 4 个验证资产）

| 文件 | 改什么 | 红线 |
|---|---|---|
| `training/framesamp_dataset.py` | `__getitem__` 不再交付 pos 张量，改交付定长 int32 帧号（`-1` 哨兵补位）；`_pad` 同步调整 | ✅ 可改（走三块验证） |
| `datastore/framesamp_store.py` | pos 小表不再逐样本 `pos_rows`；表本身改为供设备端加载 | ✅ 可改 |
| `training/config.py` | **两处** `RepackTransform` 结构表增删键 | ⚠ **不在 v2 的 B.0 授权范围** |
| `models/integration/history_observation.py` | `HistAugObservation` 字段增删、`from_dict`/`to_dict`/`from_base_obs` 同步 | ❌ **R2 硬红线** |
| `models/integration/history_pi0.py` | `HistoryPi0Config.inputs_spec` 的 perceptual 分支（决定 init 与 HLO）、`embed_memory` 调用点 | ❌ **R2 硬红线** |
| `models/representation/percep_mem.py` | `__call__` 形参与断言 | ❌ **R2 硬红线** |
| `models/representation/mem_encoder.py` | `encode_perceptual_memory` 形参；新增设备端 gather + mask 清零 | ❌ **R2 硬红线** |
| `policies/robomme_policy.py` | `RoboMMEInputs.__call__` 的键透传 | ⚠ 灰区（R2 未点名，也不在 B.0 内） |
| `policies/policy.py` | `MME_VLA_Policy._prepare_history` 的在线侧生产 | ⚠ 灰区，且其 pos 来自 R2 不动的 `MemoryBuffer` |
| 验证资产 ×4 | `scripts/dtype-unify/_common.py` 的 `MEMORY_KEYS`、`compare_dtype_fix.py`、`test_padding_dtype.py`、`scripts/data-pack-framesamp/test_pack_guards.py` 的 dtype 断言 | ✅ 白名单内 |

**5 个在 R2 硬红线内**（`src/mme_vla_suite/models/**` 整目录），**v2 期间绝对禁改**
——L3 必须等 v2 收官后按 roadmap 项 2 独立立项。

#### 2.3.2 三个必须先解决的前置

1. **判据得先自己造**（roadmap 原文）：G0 的量具只记录 collate 后、上卡前的 host batch；
   L3 落地后 pos 不再出现在 host batch，「与 G0 的 pos 摘要对拍」**目前无从采集**。
   第一步必须先落地**设备端观测点**（在 gather、mask 清零、展开之后、进投影层之前把有效 pos
   取回 host 做逐位摘要），这部分工作量计入 L3 本身。
2. **`-1` 哨兵与 mask 粒度**（后者是本轮新补的坑）：短样本现语义是 pos 补**零向量**；改交付索引后
   若用 `0` 补位，gather 会取回真实 `pos[0]` 而**不是零**，**不等价**——必须定长 int32 + `-1` 哨兵
   + gather 后按 mask 显式清零 + 越界测试。而且现有 `static_mask` 是 **token 级 `(b,512)`**
   （帧级 mask 经 `np.repeat` 展开而来），gather 索引却是**帧级 `(b,32)`**，
   清零时必须显式处理这个粒度差——**roadmap 未涉及此点**。
3. **legacy 回滚链与在线推理链的裁决**（本轮新补的坑）：`MMEVLA_DATA_BACKEND=legacy` 是
   `_resolve_backend` 的**默认值**且仍交付 pos 张量，模型侧一改签名 legacy 立刻崩；而
   `training/dataset.py` 与 `shared/**` 都在 R2 不动清单、R7 又要求旧链路原地保留。
   在线侧同理——`policies/policy.py::_prepare_history` 显式写这两个键，其 pos 来自
   R2 不动的 `shared/mem_buffer.py::MemoryBuffer`，且在线表按 4096 步现算、训练表 586 行，
   **长度口径不同，step 超界即越界**。→ L3 计划必须显式裁决：同步迁移（破 R2/R7），
   还是宣布 legacy backend 与在线旧路径报废（破 v2 的回滚保障）。

#### 2.3.3 对齐强度：输出侧 bitwise 天然不可得

L3 改的是**模型输入签名**，`ptrain_step` 是 `jax.jit` 的 → **HLO 必变、编译缓存必失效**，
输出侧的逐位等价从原理上就拿不到。主判据只能退到「输入侧逐位 + 输出侧量化复核」，
而输入侧的 pos 摘要还得先造观测点才采得到。**这是 L3 排在 L2 之后的根本原因。**

---

### 2.4 L4：把 IPC 通道换成 shared memory —— 根治，但要换尺子（原 L3 顺延）

**改的是什么（人话）**：现在 worker 把整批数据 pickle 一遍、经管道发给主进程，主进程再单线程
反序列化，实测只有 520 MB/s。torch 对 `torch.Tensor` 有共享内存的零拷贝通道，走通了这段就从
约 490 ms 降到约 10 ms。

#### 2.4.1 三个技术阻断点（本轮已用仓库自带 venv 实测，jax 0.5.3 / torch 2.7.1）

| 实测项 | 结果 |
|---|---|
| `torch.as_tensor(ml_dtypes.bfloat16 数组)` | ❌ `can't convert np.ndarray of type bfloat16` |
| `np.asarray(torch bf16 tensor)` | ❌ `Got unsupported ScalarType BFloat16` |
| `jax.make_array_from_process_local_data(sharding, torch.Tensor)` | ❌ `Cannot interpret 'torch.bfloat16' as a data type` |
| `uint16` 位视图双向桥 | ✅ 可用 |
| `jax.dlpack.from_dlpack(cpu torch bf16)` → `jax.device_put` | ✅ 可用 |

即：collate 一旦改出 torch tensor，① bf16 在 collate 内就转不过去（必须走 uint16 位视图）、
② 主进程的接收端会直接 raise（不是慢，是不能用）、③ bench 的摘要链会当场炸。
唯一自洽的接线是「collate 出 torch（bf16 走 uint16 位视图）→ 主进程 dlpack 还原 → `device_put`」。

#### 2.4.2 要改哪些文件

- **不能改 `src/openpi/training/data_loader.py`**：它是 R2 硬红线，且「给 `TorchDataLoader` 加
  `prefetch_factor` 形参」已被三份文档（v2 计划 5.1、B.0 表、roadmap「已裁定不做」）裁定不实施，
  理由正是「与不动 `src/openpi/**` 红线矛盾」——该理由对 `pin_memory`、对改 `_collate_fn` 同样适用。
  （注意与 L1 的区分：L1 的 prefetch 透传落在 `src/mme_vla_suite/training/dataloader.py`，
  不动 `src/openpi/**`，不受此裁定约束。）
- **唯一合规落点**：在 `src/mme_vla_suite/training/dataloader.py` 里自建一个 `TorchDataLoader`
  等价类，自带 collate（torch + uint16 桥）与 dlpack 交付。但必须**逐字复刻 index 序列的全部恒等要素**
  才能满足 R3：`shuffle`/`sampler` 互斥、`persistent_workers=num_workers>0`、`drop_last=True`、
  `generator.manual_seed(seed)`、spawn context、`worker_init_fn`、以及「`StopIteration` 即重建
  `iter()`」的跨 epoch 外层循环。
- **连带必须改量具**：`scripts/smoke-local/bench_train_steps.py` 的三处安装点
  （patch `TorchDataLoader.__iter__`、其源码守卫、`_install_idx_probe` 取 `torch_loader`）全部要跟着改。

#### 2.4.3 三个代价

1. **要动 G0/G2 的量具本体**：`_install_batch_digest_recorder` 的 `record()` 依赖
   `np.asarray(leaf)` / `arr.tobytes()` / `dtype.kind in "fV"`，三处对 torch tensor 全部失效；
   `main()` 对 `TorchDataLoader.__iter__` 的源码守卫也会先炸。`bench_train_steps.py` **是**
   G0/G2 对拍的尺子，改它等于改判据口径——**这是 L4 与 L2 的本质差别**：L2 用原尺子量，
   L4 得先换尺子再量。
2. **GL 内存帐要重算**：torch shm 计入 cgroup 的 shmem，而 `greatlakes.md` 已实证 anon+shmem
   **不可回收**。256.7 MB × `prefetch_factor` × workers 是新增的常驻不可回收内存，
   w16 档约 8–10 GB，现申请的 `--mem=96G` 需按档位重新核算。
3. **既有验收判据失效**：`scripts/data-pack-framesamp/spawn_matrix.py` 的 PASS 判据是
   「主进程 fd 计数回到基线」，torch 默认 `file_descriptor` sharing 会新增数十~上百常驻 fd，
   该判据需重定基线。

---

### 2.5 分水岭：五层与 G0 的对齐强度

`ptrain_step` 是 `jax.jit` 的：**模型输入签名一旦变化，HLO 必变、编译缓存必失效，输出侧的
逐位等价就天然不可得**。这条把五层劈成两类：

| 层 | 输入签名 | HLO | 量具是否要改 | **与 G0 的最强可得判据** |
|---|---|---|---|---|
| **L0** | 不变 | 不变 | 是（但只改性能族口径，正确性族量具不动） | 不适用（不触训练语义；性能模式默认关闭 + log1 路径逐字节不变） |
| **L1** | 不变 | 不变 | 否 | index 序列对拍（prefetch 透传项）；其余纯环境变量无需对拍 |
| **L2** | **不变** | **不变** | **否** | ✅ **逐位 bitwise**：G2 全套四分项 + `scalars_hex.tsv` sha256 相等 |
| **L3** | **必变** | **必变** | 是（要造设备端观测点） | ⚠ 输入侧逐位 + 输出侧量化，**输出侧 bitwise 不可得** |
| **L4** | 不变 | 不变 | **是（尺子本身）** | ⚠ 理论可得，但判据口径需重新定义后才能主张 |

**L2 是唯一既可能显著提升、又能用原尺子证明逐位等价的代码级方案**——这是它在代码级三层中
排第一的根本原因。但在它之前，L0 要先回答「生产口径下还缺多少」，L1 要先把免费的配置收益拿完。

## 三、本轮要做的事：L0

**本轮唯一授权实施的是 L0**：修三个量具文件 + 跑两档基准 run。L1 起跑前另行确认 run_name 与资源。

### 3.1 量具改动清单

| 文件 | 改什么 |
|---|---|
| `scripts/bottleneck-bench-v2/analyze_gpu_util.py` | ① `epoch_h`/吞吐/主步时改**稳态均值 + 真实 elapsed**（`t_hi - t_lo` 现算而未输出，改为输出并作主口径）；中位数降为附列。② 新增 `active_util`（非零采样条件均值）。③ 新增 `step % workers` 分组均值/p90/慢步数（log1 档专用；workers 从 `env.json` 取）。④ `EPOCH_STEPS`、吞吐 batch 从 `env.json` 取，废除 6176/64 硬编码。⑤ 判据换 `E2E95_ACCEPT` 六项（见 3.3），旧 `E2E_ACCEPT`/`E2E_EXTRA` 废弃。⑥ 兼容 log100 输入：`metrics.jsonl` 行距 >1 步时自动切换到「elapsed 均值 + dense csv」口径，不再假设逐步记录 |
| `scripts/smoke-local/bench_train_steps.py` | 新增性能模式开关（建议 `BENCH_PERF_MODE=1`，默认 `0`）：开启时放行 `--log-interval 100`（现状 `main()` 里 `if config.log_interval != 1: raise` 保持为默认行为）；开启时联动校验 `BENCH_CHECKSUM=0` 且 `BENCH_BATCH_DIGESTS=0`（性能模式禁摘要，防误跑出 T9-B5 违规 run）。**log1 路径逐字节不变** |
| `scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch` | `--log-interval 1` 硬编码改 `LOG_INTERVAL=${LOG_INTERVAL:-1}`，随 `--export` 可覆盖；`LOG_INTERVAL != 1` 时置 `BENCH_PERF_MODE=1`；`LOG_INTERVAL` 写入 `env.json` 供 analyzer 与留档使用 |

**明确不动**：`scripts/train.py`、`src/openpi/**`、`src/mme_vla_suite/models/**`、
`src/mme_vla_suite/training/dataset.py`、`src/mme_vla_suite/shared/**`（均属 v2 计划 G 节 R2 硬红线）；
`src/mme_vla_suite/training/dataloader.py`（L0 不动，留给 L1 的 prefetch 透传）；
正确性族量具路径（`_install_batch_digest_recorder`、`_install_idx_probe`、`compare_baseline.py`
及全部判据）**一行不改**。

### 3.2 基准两档 run（规格见 2.0.3）

- 配置：w8c16、packed（`MMEVLA_DATA_BACKEND=packed`）、prefetch=2（现状不可调）、线程现状（OMP/MKL=16）；
- log1 对照档 600 步：与旧 89.2% 同口径，**用新旧两版 analyzer 对同一 record_dir 双跑**，
  把「量具计算口径差异」与「run 间波动」分开归因；
- log100 生产档 600 步：新 speed 基准，`ANALYZE_ACCEPT=1` 判 `E2E95_ACCEPT`。

### 3.3 新判据：`E2E95_ACCEPT` 六项（性能族唯一现行判据）

| 项 | 阈值 | 口径 |
|---|---|---|
| `util_mean` | **≥ 95%** | dense 500 ms 稳态均值（AGENTS 16） |
| `zero_pct` | **≤ 3.8%**（工程目标 ≤2%） | dense 稳态 0% 采样占比 |
| `active_util` | **≥ 98%** | 非零采样条件均值（GPU 工作时是否吃满） |
| `step_mean` | **≤ 5.013 s**（建议直接要求 ≤5.00 s） | 稳态真实 elapsed ÷ 稳态步数 |
| `epoch_mean` | **≤ 8.6 h** | `EPOCH_STEPS × step_mean`，EPOCH_STEPS 从 `env.json` 取 |
| `slow_wall_pct` | **≤ 5%** | log1 档：>8 s 慢步墙钟占比；log100 档：dense 低 util 连续段近似 |

阈值推导（Codex，基于旧 w8 实测 89.2%/9.7%/98.8%/5.412 s）：要总体 util ≥95%，0% 采样须
9.7%→≤3.8%（留余量取 ≤2%，对应总 util 约 96.8%）；步时须 5.412→≤5.013 s（再快 7.4%）。
中位数一律只作附列，禁作任何判据与标题结论（AGENTS 16）。

### 3.4 正确性侧要求

L0 不触训练语义，正确性验证为**代码级**：

1. 性能模式默认关闭（`BENCH_PERF_MODE` 未设时全部行为逐字节等价现状，含 raise 路径）；
2. 正确性族量具与判据零改动（`git diff` 逐 hunk 核对不触 digest/index/compare 路径）；
3. log1 对照档跑出的 `metrics.jsonl` 与历史 run 同 schema，可被旧版 analyzer 正常消费（可比性核对）。

G0/G2 bitwise 链本身不需要因 L0 重跑——量具改动不在其链路上。**首个需要重新对拍的是 L1 的
prefetch 透传**（index 序列），首个需要全套 AGENTS 18 的是 L2。

## 四、验证阶梯

| 步 | 内容 | 判据 | 成本 |
|---|---|---|---|
| **V-L0-code** | 量具改动代码级验证：`BENCH_PERF_MODE` 未设时 log1 路径逐字节等价（含 raise）；新 analyzer 对既有历史 record_dir 重算，均值/附列口径输出齐全 | 默认路径 diff 审查 + 历史目录重算无异常 | 分钟级 |
| **V-L0a GL 对照档** | log1 600 步 w8c16，新旧 analyzer 双跑同一 record_dir | 只报数不设阈值：量具口径差异归因 + 与 89.2% 的可比性 | 1 h + 排队 |
| **V-L0b GL 生产档** | log100 600 步 w8c16，`ANALYZE_ACCEPT=1` | **`E2E95_ACCEPT` 六项**；全过 → 收官分叉（五节） | 1 h + 排队 |
| **V-L1**（若开） | 配置打包档（log100 + w8 + 线程1 + prefetch4） | `E2E95_ACCEPT` 六项 + MaxRSS<80G + index 序列对拍（prefetch 项） | 1 h + 排队 |
| **V-L2**（若开） | AGENTS 18 全套：链路图×2 + 第一块轻量对拍 + 第二块本机 1000 步确定性档 bitwise + GL 性能档 | 五标量逐步 hex 全等（`scalars_hex.tsv` sha256 == `c799a0b2…`）+ `state_digest` 全等 + canonical batch digest 全等 + index 全等（n=8072）+ `E2E95_ACCEPT` | ~33 min 本机 + GL |
| **最终复验** | 达标候选在**两个独立 seed/节点**各跑 600 步 GL | 两个 run 都 `util_mean ≥95%` 才算通过 | 2 job |

性能 run 纪律：生产 XLA（不注入确定性 flags、autotune 默认开）、`BENCH_CHECKSUM=0`、
`BENCH_BATCH_DIGESTS=0`、dense 500 ms NVML、只用均值/0%/真实 elapsed 判读（T9-B5）。
正确性 run 纪律：log1 确定性档、摘要全开，util/步时禁作性能结论。两族永不混用一个 run。

其余纪律：起跑前 `BASELINE_ENV=PASS` preflight 必过（T5 指纹不含仓库代码 sha，量具改动不破坏
`BASELINE_ENV`）；每个 >5 min 的 run 按 AGENTS 17 留档 `docs/training-doc/<run_name>/`；
run_name 起跑前逐个交用户确认（AGENTS 6）；GL 的 4×A40 / 2 h 超出 `greatlakes.md` 硬限，
提交前逐个走资源审批并留放行记录。

## 五、决策门与后续分叉

```
L0-log100 全过 E2E95_ACCEPT ──→ 收官：不做任何 DataLoader 优化，回填 T8 与 roadmap
        │未过
        ▼
L1 配置打包（log100 + w8 + 线程1 + prefetch4）──全过──→ 收官（同上）
        │未过
        ▼
L2 取数/计算重叠（bitwise 可对齐，AGENTS 18 全套）──全过──→ 收官
        │未过
        ▼
按缺口性质再评估 L4（IPC 通道，bitwise 理论可得但要换尺子）→ L3（削载荷，输出侧 bitwise 不可得）
```

- 每层「全过」= `E2E95_ACCEPT` 六项 + 最终双 seed/节点复验；
- L1 的 prefetch 项达标即止、收益不足即回退（2.1）；
- L3 前置三件事（设备端 pos 观测点、`-1` 哨兵与 mask 粒度、legacy/在线链裁决）不解决不开工（2.3.2）；
- 旧版预期数字（L2 折算 96–98% util、epoch 8.3–8.6 h 等）**全部标注为 log1 口径预估**，
  生产口径的预期待 L0 实测后重估——本文件不再给未实测的生产口径预测值。

epoch 一项按旧数据推算余量很小（阈值 8.6 h）。util ≥95% 是用户指定的主目标，
epoch 若卡线需单独判读，不因其失守而否定 util 结论。

---

# 第二部分（技术细节，供 agent 追踪）

## A. 改动清单（本轮授权范围 = L0，超出即越界）

| 文件 | 改什么 | 红线归属 |
|---|---|---|
| `scripts/bottleneck-bench-v2/analyze_gpu_util.py` | 3.1 表第一行全部六项（均值/elapsed 主口径、active_util、`step%workers` 分组、env.json 取参、`E2E95_ACCEPT`、log100 兼容） | 不在 T1 白名单但在 v2 的 B.0 表内（S7.5 已参数化过），`G0_SCOPE` 输出需逐 hunk 说明 |
| `scripts/smoke-local/bench_train_steps.py` | 仅新增 `BENCH_PERF_MODE` 开关与联动校验（3.1 表第二行）；**不触** digest/index/compare 任何路径 | T1 白名单内；R2 除外条款（默认值等价现状） |
| `scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch` | `LOG_INTERVAL` 可覆盖 + 透传 + 写 `env.json`（3.1 表第三行） | 同 analyze 行 |

**明确不动**（3.1 已列，此处重申执行口径）：`scripts/train.py`、`src/openpi/**`、
`src/mme_vla_suite/models/**`、`training/dataset.py`、`shared/**`（R2 硬红线）；
`src/mme_vla_suite/training/dataloader.py`（留给 L1）；正确性族量具与判据零改动。

**L1 预告（本轮不实施，另行审批）**：`gl_e2e_fix.sbatch` 线程默认值改 1 并补
`OPENBLAS_NUM_THREADS`/`NUMEXPR_NUM_THREADS`；`src/mme_vla_suite/training/dataloader.py`
加 `prefetch_factor` 透传（补 index 序列对拍）。

## B. 关键实现点

### B.1 L0 量具

- **`E2E95_ACCEPT` 的判定输入**：util 三项永远来自 `gpu_util_dense.csv`（缺 dense 退 legacy 时
  必须显式告警且判定标 `DEGRADED`）；step_mean 来自 `metrics.jsonl` 的 `wall_time` 首末差
  （稳态窗口裁剪后），log1/log100 两档同式；`slow_wall_pct` 按档分口径（3.3 表）。
- **log100 兼容**的判别：相邻 metrics 行的 `step` 差 >1 即进入 log100 口径，逐步统计
  （p10/p90、慢步分层、相位分组）自动跳过并在输出中写明「log100 档无逐步步时」。
- **`env.json` 增写 `log_interval`**（sbatch 侧 `write_env()`），analyzer 读它决定口径，
  不靠猜 metrics 行距。
- **性能模式联动校验**放在 `bench_train_steps.py::main` 现有护栏区：`BENCH_PERF_MODE=1` 时
  若 `BENCH_CHECKSUM != 0` 或 `BENCH_BATCH_DIGESTS != 0` 立即 raise（防 T9-B5 违规 run）；
  `BENCH_PERF_MODE` 未设/为 0 时保持 `log_interval != 1` 即 raise 的现行为。
- **尾段 flush**：`train.py` 在 `num_train_steps-1` 非日志步时尾段 infos 不落盘（不动 train.py，
  接受此损失）——analyzer 的 step_mean 以真实 elapsed 为准，不依赖末行 metrics；600 步 % 100 == 0
  时日志步恰落在末步附近，实施时以实测行为留档为准。

### B.2 L2b 预取层（顺延保留，进入 L2 时启用）

- 结构模板照抄仓库已有先例
  `scripts/bottleneck-bench/gl-compute-only/compute_only_train_steps.py::_RepeatFirstBatchLoader`
  ——「包一层 loader、透传 `data_config()`、自定义迭代、经 `create_data_loader` monkeypatch 装配」。
- 四条硬要求：① `BENCH_PREFETCH` 默认关闭、显式开启（R2 除外条款「默认值等价现状」）；
  ② 不做成 generator，用显式迭代器对象（`__iter__`/`__next__`/`close()`）自管线程与队列生命周期，
  队列深度 k 从 1 起；③ 预取线程异常必须在主线程重新 raise；④ `finalize()` 之前必须 join 预取线程，
  否则 `_LoggingSampler` 的 `index_log` 被并发 append、`index_sequence.json` 的 `n` 非确定。
- 与现有 monkeypatch 的关系：`_install_batch_digest_recorder` 替换的是
  `_openpi_dl.TorchDataLoader.__iter__`（内层），预取层套在 `DataLoaderImpl`（外层）之外，两者不冲突。
  正确性 run 里 batch 摘要的 `record()` 会在预取线程执行——单线程串行写文件无竞态，但
  `digest_seconds` 语义从「主线程停顿」变成「后台线程耗时」，须在留档写明。
- `openpi/training/sharding.py::_MeshState.active_mesh` 是类级全局、非 thread-local，但预取线程
  不调 `set_mesh`，且 `make_array_from_process_local_data` 不读 `active_mesh`，当前不冲突
  ——实现里需注释记一笔。
- **绝不能把取数挪到 `ptrain_step` 之前**（L2a/L2c 同样适用）：那样 step *s* 用的 batch 会从
  idx *s* 变成 idx *s+1*，是「看起来只是重排」的静默错帧，G0/G2 对拍会全线 FAIL。
  L2a 的唯一合法插入点是 `infos.append(info)` 之后、`if step % log_interval` 块之前。
- 现有 fail-loud 断言（`inspect.getsource` 子串检查）探测不到语句顺序变化——若走 L2a 改 train.py，
  应同时加一条顺序敏感断言。
- 显存与 index 前提（已核实，顺延不变）：batch 按 B 轴 sharded 到 4 卡，多驻留一个 batch 只
  +64 MB/卡；`donate_argnums=(1,)` 只捐 `train_state`，无 donation-after-use 风险；预取只让尾部
  超前量 `pf×w → pf×w + k`，前缀逐位不变（满足 R3），`compare_baseline.py::compare_index_seq`
  取共同前缀、天然容忍。
- T5 指纹清单（`check_baseline_env.py`）不含仓库代码文件 sha，改代码不破坏 `BASELINE_ENV=PASS`，
  G0 固化产物可直接引用对拍——这是 L2 验证方案成立的前提（已核实，对 L0/L1 同样适用）。

## C. run 命名与留档（起跑前逐个交用户确认）

| 用途 | 建议 run_name | 族 |
|---|---|---|
| V-L0a GL log1 对照档 | `v1-l0-gauge-log1` | 性能族（新量具，log1 口径） |
| V-L0b GL log100 生产档 | `v1-l0-gauge-log100` | 性能族（**新 speed 基准**） |
| V-L1 配置打包档（若开） | `v1-l1-cfg-w8` | 性能族 |
| V-L2 本机 bitwise（若开） | `v1-l2-overlap-g3` | 正确性族（确定性档 + 摘要，log1） |
| V-L2 GL 验收（若开） | `v1-l2-overlap-e2e` | 性能族 |

旧命名表（`v1-prefetch-g3` / `v1-prefetch-g3-speed` / `v1-prefetch-e2e-w8c16`）随原 V2/V3/V4
阶梯一并废弃。commit 切分沿用 v2 体例：本文件落档走 `docs:` → L0 量具改动走 `commitV3.6` →
launch.md 预提交 → clean HEAD 起跑 → 结果留档提交。收官后回填 v2 计划 T8 登记簿
（含 0.3 节作废标记）与 roadmap。

## D. 本轮诊断的可复现命令

> **注意**：以下命令与流程产出的判定基于**旧量具**（中位 epoch、旧 `E2E_ACCEPT`），仅供历史复现
> 与机制档案核对，不再作为任何现行结论的依据。

全部为只读分析，数据源是既有 run 的记录目录，无需重跑训练。

```bash
# 各档 util / 步时 / accept 判定（--steps 按 run 实际步数：600 或 300）
uv run scripts/bottleneck-bench-v2/analyze_gpu_util.py \
  v1-store/bench/bottleneck/v1-framesamp-e2e-w8c16 --steps 600 --accept
```

- **步内相位分析**与**快步/慢步分层**：把 `metrics.jsonl` 的逐步 `wall_time` 差分作为步边界，
  将 `gpu_util_dense.csv` 的每个采样点按其在所属步中的相对位置分桶统计 util 均值与 0% 占比；
  分层阈值取「步时 ≤ 稳态窗口 p10 × 1.15」为快步、其余为慢步。
- **慢步周期性**：对同一组步时差分取「> p10×1.15」的步号序列，看相邻步号间隔——三档分别呈现
  周期 4 / 8 / 12，恰等于各自的 `num_workers`。
- **IPC / H2D 微基准**：构造与 GL 日志逐键同形同 dtype 的 fake dataset（worker 只造零数组），
  用 `torch.utils.data.DataLoader`（spawn、`persistent_workers`、自定义 collate 做 `np.stack`）
  测稳态 `next()` 耗时；H2D 单独用 `jax.make_array_from_process_local_data` + `block_until_ready` 测。
- **重叠可行性**：把一段 GPU 计算（`jax.lax.fori_loop` 包住的矩阵乘链，调到与 GL 同量级的约 4.3 s）
  与上述 DataLoader 组合，分别测「串行（复刻 log1 现状）」「同线程重叠」「后台线程预取」三种调度的
  步时分布。
