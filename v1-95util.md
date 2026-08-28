# v1：GPU 利用率冲刺 ≥95% —— 先修量具重立基准（L0），再按 L1→L4 顺序推进

> **本文件是本轮（2026-08-28 起，Codex 审计后重排版）的单一权威计划文档。**
> G0/G2 定义、红线表（R1–R17）、白名单（T1）、登记簿（T8）的权威源是
> [`v2-framesamp-restructure-plan.md`](v2-framesamp-restructure-plan.md)；L3/L4 的 scope 权威源是
> [`v1-post-restructure-roadmap.md`](v1-post-restructure-roadmap.md) 项 2/项 3；本文件只引用不复制。
>
> **目标**：GL 4×A40 e2e 的 dense 稳态 **util 均值 ≥95%**（用户 2026-08-28 指定）。
>
> **两条全局原则**：
> 1. **旧性能数字全部作废，重新开始**。此前所有性能族基线与量具口径——`v1-g0-speed-r2` 锚点、
>    S8b 各档数字（含 89.2%）、旧 `E2E_ACCEPT`/`E2E_EXTRA` 判据、speed 链的一切历史对齐——
>    一律降为历史参照，不再作为任何对比基线。新基准由 L0 重立。
> 2. **【硬性原则，用户两次强调】G0/G2 正确性对拍链原样保留、一项不减**——这是全程**不漂移**
>    的关键保证（200 batch raw bits 零失配、1000 步五标量逐步 hex、`state_digest`、canonical
>    batch digest、index 序列判据；`scalars_hex.tsv` sha256 锚 `c799a0b2…`）。正确性族 run 照旧跑
>    `log_interval=1` 确定性档，其量具、判据、固化产物本轮完全不动。废弃范围严格限定在性能族。

## 量具缺陷发现（本次改写的起因，Codex 审计核实）

- GL 性能 benchmark 的 sbatch **硬编码 `--log-interval 1`**：训练每走一步都强制停下来把指标从
  GPU 拉回主机（`jax.device_get`），拉完才去取下一批数据。
- 而**正式训练默认 100 步才记一次日志**（`config.py::log_interval = 100`），平时 GPU 不会被打断
  ——已核实非日志步没有任何同步点。

**结论：89.2% 很可能是量尺自己打断流水线量出来的数字，不代表生产形态。** 所以第一步不是改
DataLoader，而是修量具、按生产口径重测——生产口径下缺口可能远小于想象，甚至可能直接达标。

---

# 第一部分（给人看）

## 一、诊断：缺口到底在哪里（说人话）

以下是旧量具（每步打断）口径下查明的事实。机制都是真的，但**幅度**要等 L0 用生产口径重测。

1. **GPU 本身没问题**。把数据供应换成「同一批数据无限重复」（零取数成本），util 立刻到 99.9%。
   计算图、多卡切分、每步同步都不背锅——**缺口 100% 来自「等数据」**。
2. **等待都堆在每一步的开头**。步的后半段 GPU 恒定 100%；四分之三的步很快（util 97.4%，
   本身已达标），四分之一的步很慢（util 71.6%），这些慢步贡献了 84% 的缺口。
3. **慢步有严格的节奏**：每隔「worker 数」那么多步准时出现一次。这是主进程轮流找各个 worker
   收数据的节奏特征——说明卡在**交接**上，不是 worker 干不完活。
4. **交接为什么慢**：worker 把一批 257 MB 的数据打包（pickle）经管道发给主进程、主进程单线程
   拆包，这条进程间通道实测只有约 520 MB/s——每批要等约 0.5 秒。因为瓶颈在主进程这一端，
   **加 worker 完全没用**（实测 w8→w12 只 +0.3pp）。数据上卡（H2D）只要 9.5 ms，不是问题。
5. **硬盘/NFS 不背锅**：冷缓存 vs 热缓存实测只差 0.7%，继续优化磁盘拿不到 5–6pp。
6. **但以上等待暴露多少，取决于量法**：旧量具每步强制「等 GPU 算完才取数」，0.5 秒的交接
   全部变成 GPU 空转。生产口径 100 步才同步一次，其余 99 步取数和计算天然可以重叠——
   所以真实缺口必须先重测（L0），再决定要不要动代码。
7. 顺带查清一件事：roadmap 项 1 的前提（「每个 worker 在 GPU0 建 442 MiB CUDA context」）
   实测不成立——全程只有主进程一个 pid。该项从补救清单撤下。

## 二、五层方案：每层的实际作用、修改的文件、改完要跑什么

| 层 | 一句话 | 改动量 | 与 G0 的对齐 |
|---|---|---|---|
| **L0** | 修量具，按生产口径重立基准 | 3 个量具文件 | 不触训练语义 |
| **L1** | 配置收敛（log100 + w8 + 线程1 + 预取4） | 环境变量 + 1 个形参透传 | index 序列对拍即可 |
| **L2** | 取数与 GPU 计算重叠 | 1 个文件 | ✅ 逐位 bitwise 可证 |
| **L3** | 不传 pos/state，GPU 侧查表 | 9 个 src/ 文件（5 个在红线内） | ⚠ 输出侧 bitwise 不可得 |
| **L4** | 进程间通道换共享内存 | 自建 loader + 换量具尺子 | ⚠ 理论可得，尺子要换 |

---

### L0：修量具 + 重立基准

**实际作用**：让量尺不再打断被测对象（允许按生产方式 100 步才同步一次），让统计口径正确
（epoch/步时用真实墙钟均值，不再用中位数；补上「GPU 工作时是否吃满」的 active_util 等缺失
统计），然后重测基准——回答「生产口径下到底还缺多少」。**如果直接达标，整个项目就此收官。**

**修改的文件（3 个，全部是量具，不碰任何训练代码）**：

| 文件 | 改什么 |
|---|---|
| `scripts/bottleneck-bench-v2/analyze_gpu_util.py` | epoch/吞吐/主步时改**稳态均值 + 真实墙钟**（现状用中位数，且真实 elapsed 算了不输出）；中位数降为附列。新增 `active_util`（非零采样条件均值）、`step % workers` 分组统计（log1 档专用）。`EPOCH_STEPS=6176`、batch=64 两处硬编码改从 `env.json` 取。判据换 **`E2E95_ACCEPT` 六项**（见第二部分判据表），旧 `E2E_ACCEPT`/`E2E_EXTRA` 废弃。兼容 log100 输入（metrics 行距 >1 步时自动切「真实墙钟 + dense 采样」口径） |
| `scripts/smoke-local/bench_train_steps.py` | 新增性能模式开关 `BENCH_PERF_MODE`（默认 0）：开启才放行 `--log-interval 100`（现状对 ≠1 直接报错，该默认行为保留）；开启时强制 `BENCH_CHECKSUM=0`、`BENCH_BATCH_DIGESTS=0`（性能 run 禁摘要）。**log1 路径逐字节不变，正确性族量具零改动** |
| `scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch` | 硬编码的 `--log-interval 1` 改为可覆盖变量 `LOG_INTERVAL`（默认仍 1）；≠1 时自动置 `BENCH_PERF_MODE=1`；`LOG_INTERVAL` 写入 `env.json` |

不得为逐步计时重新引入每步 `device_get`；log100 下没有逐步步时（日志记的是 100 步区间均值），
慢步分层、相位分组只在 log1 档做；util 三项统计来自 500 ms dense 采样，与日志粒度无关，两档一致。

**改完要跑什么**：同一配置（w8c16、packed、prefetch=2、线程现状）两档 GL run，各 600 步、
4×A40/16C/96G：

1. **log1 对照档**：与旧 89.2% 同口径，新旧 analyzer 对同一记录目录双跑，把「量具口径差异」
   单独归因出来；
2. **log100 生产档**：**新 speed 基准**，判 `E2E95_ACCEPT`——全过即收官，后面所有层都不做。

---

### L1：配置收敛（不改任何训练逻辑）

**实际作用**：把四项配置调对，纯配置层把剩余空窗再压一压：

1. 沿用 `LOG_INTERVAL=100`（生产口径）；
2. **worker 固定 8 个**——实测加到 12 只快 0.3pp 反而多出慢步；
3. **每进程线程数收敛为 1**——现状每个进程都开 16 线程，8 个 worker + 主进程在 16 核上
   16×16 互相打架；改 OMP/MKL 为 1，并补设现状**完全缺失**的 OPENBLAS/NUMEXPR 两个变量；
4. **预取从 2 提到 4**——让 worker 手里多备一批货（in_order 恒 True，禁 False——False 会改变
   batch 到达顺序、改变训练轨迹，无法与 G0 对齐）。

**修改的文件**：

| 文件 | 改什么 |
|---|---|
| `scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch` | `OMP_NUM_THREADS`/`MKL_NUM_THREADS` 默认由 16 改 1，补设 `OPENBLAS_NUM_THREADS`/`NUMEXPR_NUM_THREADS`=1（均可覆盖） |
| `src/mme_vla_suite/training/dataloader.py` | 加 `prefetch_factor` 形参透传（现在全链路无处可调；该文件不在 R2 不动清单） |
| 提交参数（不改文件） | `WORKERS=8` 固定 |

**改完要跑什么**：

1. prefetch 透传属 loader 装配改动 → 先补 **index 序列对拍**（共同前缀逐位不变、尾部超前量
   只增不重排，R3）+ 两 epoch 实际消费 index 对拍；
2. 一个 GL 打包档 600 步（四项配置一起上，用户拍板打包跑），判 `E2E95_ACCEPT`；
   约束 MaxRSS <80 GB、无 major fault 激增；prefetch 一项收益不足（util <1pp 或步时 <2%）回退 2。
   全过即收官。

---

### L2：取数与 GPU 计算重叠（代码级首选）

**实际作用**：现在主循环是「发计算 → 等算完 → 才去拿下一批」；改成拿数据和算数据**同时发生**，
把交接等待藏进 GPU 计算。**不改任何一个数、不改喂给模型的顺序**，只改「什么时候去搬」。
本机实测（复刻每步同步的旧结构）：步时 5.27→4.63 s，最长步 9.31→4.74 s，周期性慢步完全消失。
注意：log100 生产口径下「现状」侧本就没有每步屏障，此收益幅度不能直接外推——L2 是否还值得做，
完全取决于 L0/L1 的实测结果。

**修改的文件（三选一落法）**：

| 落法 | 改哪个文件 | 怎么改 | 红线 |
|---|---|---|---|
| L2a | `scripts/train.py` | 把 `batch = next(data_iter)` 一行从 `device_get` 之后移到之前（唯一合法插入点：`infos.append` 之后、日志块之前——放到 `ptrain_step` 之前就是静默错帧，对拍全线 FAIL） | ❌ 撞 R1+R2 硬红线，需显式解禁 |
| **L2b（首选）** | `scripts/smoke-local/bench_train_steps.py` | monkeypatch 包住 `create_data_loader`，套一层后台预取（有界队列+单线程），`BENCH_PREFETCH` 默认关闭 | ✅ T1 白名单内 |
| L2c | `src/mme_vla_suite/training/dataloader.py` | 同样的预取层落进生产 loader | ✅ 不在不动清单 |

**改完要跑什么**（训练交付路径改动，AGENTS 18 全套）：

1. 重构前后两张链路图 + 第一块轻量对拍（index 序列：前缀逐位不变）；
2. **本机 1000 步确定性档 bitwise 硬闸**：五标量逐步 hex 全等（`scalars_hex.tsv` sha256 ==
   `c799a0b2…`）、`state_digest` 全等、canonical batch digest 全等、index 全等（n=8072）；
3. GL 性能档 600 步判 `E2E95_ACCEPT`。全过即收官。

---

### L3：不传 pos/state，GPU 侧查表（削载荷 40%）

**实际作用**：每批数据里 100.7 MB 的「位置编码」其实由「第几帧」唯一决定，库里已存成一张
586 行的小表——不必每步整块搬运，只传 32 个帧号（8 KB）让 GPU 自己查表；另外 2.1 MB 的
`static_state_emb` 模型根本不用（`use_state_emb=false`），纯属白算白传，直接删。
进程间搬运量 256.7 → 153.9 MB（−40%），交接时间约 490 → 294 ms。

**修改的文件**：9 个 `src/` 文件 + 4 个验证资产。其中 **5 个在 R2 硬红线内**
（`models/**` 整目录：`history_observation.py`、`history_pi0.py`、`percep_mem.py`、
`mem_encoder.py` 等），v2 期间绝对禁改——**L3 必须独立立项**；其余为
`training/framesamp_dataset.py`、`datastore/framesamp_store.py`、`training/config.py`、
`policies/robomme_policy.py`、`policies/policy.py` 与 4 个白名单内验证脚本。

三个必须先解决的前置：① G0 量具只记 host batch，pos 移到设备端后**判据无处采集**，得先造
设备端观测点；② 短样本补位必须用 `-1` 哨兵 + gather 后按 mask 显式清零（用 0 补位会查回真数据，
不等价），且 mask 是 token 级 (b,512)、索引是帧级 (b,32)，粒度差要显式处理；③ legacy 回滚链与
在线推理链会因签名改变当场崩，必须先裁决迁移还是报废。

**改完要跑什么**：先落设备端观测点 → 输入侧逐位对拍 + 输出侧量化复核（模型输入签名变了，
HLO 必变，**输出侧 bitwise 原理上拿不到**——这是 L3 排最后的根本原因）→ GL 性能档判
`E2E95_ACCEPT`。

---

### L4：进程间通道换共享内存（根治通道，但要换尺子）

**实际作用**：worker→主进程那条 520 MB/s 的 pickle 管道，换成 torch 对 `torch.Tensor` 的
共享内存零拷贝通道，每批交接从约 490 ms 降到约 10 ms。实测可行接线只有一条：
collate 出 torch（bf16 走 uint16 位视图桥）→ 主进程 dlpack 还原 → `device_put`
（bf16 直转、numpy 互转、jax 直收 torch 三条路都实测报错）。

**修改的文件**：

| 文件 | 改什么 |
|---|---|
| `src/mme_vla_suite/training/dataloader.py` | 自建 `TorchDataLoader` 等价类（torch collate + uint16 桥 + dlpack 交付），逐字复刻 index 恒等要素（shuffle/seed/spawn/drop_last/跨 epoch 重建）满足 R3。不能改 `src/openpi/**`（硬红线） |
| `scripts/smoke-local/bench_train_steps.py` | 三处量具安装点（`TorchDataLoader.__iter__` patch、源码守卫、idx probe）全部跟着改——**这是 L4 与 L2 的本质差别：L2 用原尺子量，L4 得先换尺子再量** |

**改完要跑什么**：先重新定义判据口径（尺子换了）→ bitwise 对拍 → GL 性能档判 `E2E95_ACCEPT`。
另有两笔帐必须先算：torch shm 是**不可回收**内存（约 `257 MB × prefetch × workers`，
`--mem=96G` 要按档重核）；fd 泄漏判据（`spawn_matrix.py`）需重定基线。

---

# 第二部分：L0→L4 顺序执行计划（技术细节，供 agent 追踪）

## 判据：`E2E95_ACCEPT` 六项（性能族唯一现行判据）

| 项 | 阈值 | 口径 |
|---|---|---|
| `util_mean` | **≥ 95%** | dense 500 ms 稳态均值 |
| `zero_pct` | **≤ 3.8%**（工程目标 ≤2%） | dense 稳态 0% 采样占比 |
| `active_util` | **≥ 98%** | 非零采样条件均值 |
| `step_mean` | **≤ 5.013 s**（建议直接要求 ≤5.00 s） | 稳态真实墙钟 ÷ 稳态步数 |
| `epoch_mean` | **≤ 8.6 h** | `EPOCH_STEPS × step_mean`，参数从 `env.json` 取 |
| `slow_wall_pct` | **≤ 5%** | log1 档：>8 s 慢步墙钟占比；log100 档：dense 低 util 连续段近似 |

中位数一律只作附列，禁作判据与标题结论（AGENTS 16）。每层「全过」= 六项全过；
**任何达标候选最后须在两个独立 seed/节点各跑 600 步 GL，两个 run 都 util ≥95% 才算最终通过。**

## 顺序执行

```
第 1 步  L0 修量具        改 3 个量具文件 → commit（commitV3.6）
第 2 步  L0 两档基准       log1 对照档 + log100 生产档，各 600 步 GL
            ├─ log100 全过 E2E95_ACCEPT ──→ 双 seed 复验 → 收官（回填 T8/roadmap，L1–L4 全不做）
            └─ 未过 ↓
第 3 步  L1 配置收敛       改 sbatch 线程 + dataloader.py prefetch 透传
                          → index 序列对拍 → GL 打包档 600 步
            ├─ 全过 ──→ 双 seed 复验 → 收官
            └─ 未过 ↓
第 4 步  L2 取数重叠       L2b 落地 → 本机 1000 步 bitwise 硬闸（sha256 == c799a0b2…）
                          → GL 性能档 600 步
            ├─ 全过 ──→ 双 seed 复验 → 收官
            └─ 未过 ↓
第 5 步  按缺口性质再评估   L4（通道，bitwise 理论可得）优先于 L3（削载荷，输出侧 bitwise 不可得，
                          且三个前置不解决不开工）
```

## 各步执行要点

1. **L0 量具实现点**：
   - `E2E95_ACCEPT` 的 util 三项永远来自 `gpu_util_dense.csv`（缺 dense 退 legacy 须显式告警并标
     `DEGRADED`）；`step_mean` 来自 `metrics.jsonl` 的 `wall_time` 首末差（稳态窗口裁剪后），两档同式。
   - log100 口径判别靠 `env.json` 新增的 `log_interval` 字段（sbatch `write_env()` 写入），不靠猜
     metrics 行距；log100 下逐步统计（p10/p90、慢步分层、相位分组）自动跳过并在输出写明。
   - `BENCH_PERF_MODE=1` 时若 `BENCH_CHECKSUM≠0` 或 `BENCH_BATCH_DIGESTS≠0` 立即报错
     （性能 run 禁摘要，防 T9-B5 违规）；未设时保持现行为（`log_interval≠1` 即报错）。
   - 尾段：`train.py` 末步非日志步时尾段指标不落盘——不动 `train.py`，接受此损失，`step_mean`
     以真实墙钟为准，不依赖末行 metrics。
   - **正确性族为何不需要 log100**：`train.py` 日志步记录的是 100 步**区间均值**，log100 天然没有
     逐步五标量；正确性族维持 log1 确定性档即可完整保留逐步 hex 证据链（其 util/步时本就禁作
     性能结论，量具打断对它无害），**无须为取证动 `scripts/train.py`**。
2. **L2b 实现点**（进入 L2 时启用）：结构照抄
   `scripts/bottleneck-bench/gl-compute-only/compute_only_train_steps.py::_RepeatFirstBatchLoader`
   的包装写法。四条硬要求：`BENCH_PREFETCH` 默认关闭；不做 generator，显式迭代器自管线程与队列
   （深度从 1 起）；预取线程异常在主线程重 raise；`finalize()` 前必须 join（否则 index 记录被并发
   append）。已核实前提：预取层与 digest recorder 分属外/内层不冲突；`active_mesh` 全局但预取线程
   不触；显存只 +64 MB/卡；T5 环境指纹不含代码 sha，改代码不破坏 `BASELINE_ENV=PASS`，
   G0 固化产物可直接引用对拍。
3. **run 纪律（每步通用）**：
   - 性能族 run：生产 XLA（不注入确定性 flags、autotune 默认开）、`BENCH_CHECKSUM=0`、
     `BENCH_BATCH_DIGESTS=0`、dense 500 ms NVML、只用均值/0%/真实墙钟判读；
   - 正确性族 run：log1 确定性档、摘要全开，util/步时禁作性能结论；两族永不混用一个 run；
   - 起跑前 `BASELINE_ENV=PASS` preflight 必过；clean HEAD 起跑；每个 >5 min 的 run 按 AGENTS 17
     留档 `docs/training-doc/<run_name>/`；run_name 起跑前逐个交用户确认（AGENTS 6）；
     4×A40/2h 超出 `greatlakes.md` 硬限，提交前逐个走资源审批并留放行记录。
4. **run 命名占位**（起跑前逐个确认）：L0 两档 `v1-l0-gauge-log1` / `v1-l0-gauge-log100`；
   L1 `v1-l1-cfg-w8`；L2 本机 `v1-l2-overlap-g3`、GL `v1-l2-overlap-e2e`。
5. **明确不动**：`scripts/train.py`、`src/openpi/**`、`src/mme_vla_suite/models/**`、
   `training/dataset.py`、`shared/**`（R2 硬红线，L2a/L3 若要触碰须先显式解禁/独立立项）；
   正确性族量具与判据（digest/index/compare 路径）在 L0–L2 期间一行不改。
6. **收官动作**：回填 v2 计划 T8 登记簿（含旧性能族基线的作废标记）与 roadmap 决策门结论。
