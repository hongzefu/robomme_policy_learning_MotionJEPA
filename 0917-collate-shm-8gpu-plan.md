# 0917 计划：collate 改走共享内存，吃满 8 卡（基准：8 帧 × 8×8 modulation 关闭态）

> 创建日期 2026-09-17（America/New_York）。计划首次入库 commit `5b2e7ba`，之后只有 docs commit 修订本文件。**「改前」源码 = `v2-motionmem` 上 `e4dc733` 的源码**（`e4dc733` 之后全是 docs commit，`git diff --stat e4dc733 HEAD -- src scripts packages pyproject.toml uv.lock` 为空）；起跑时主副本的 clean HEAD 记作 `A_HEAD`，具体 SHA 在第 4 节第 1 步记录，本文件不钉死它。
> 本计划只覆盖 dataloader 交付路径的一处改动、它的一致性检验与 8 卡速度对比；不碰 motion、不碰模型、不碰超参。
> 可行性验证（不启动训练）已在写本计划前完成，结果内联于第一部分第 3 节；产物固化在 `v1-store/bench/collate-shm-feasibility/`（不进 git）。
> **2026-09-17 第二版**：按 Codex 静态审计（锚定 `96b86b5`）的五条意见修订，逐条处置见第 0.2 节。

---

## 第一部分（给人看）

### 0. 结论先行

- **问题**：8 卡 mesh (1,8) 训练步时 1.82 s、GPU util 均值 51%、45% 采样为 0%，比 4 卡只快 22%（`docs/training-doc/bench-m8x8-8gpu-worker/result.md`）。
- **假设的根因（待第 7 步 A/B 实测裁决）**：`src/openpi/training/data_loader.py::_collate_fn` 返回 **numpy** 数组，torch `DataLoader` 对 numpy 只能整批 pickle 后经队列管道送到主进程；b128 一个 batch **513 MB**，pickle 的 `dumps` 在 worker、`loads` 在主进程**串行**发生，主进程每批平均等 **0.98 s**（真实数据集、CPU 环境实测，峰值 13 s）。8 卡 GPU 计算按 4 卡 2.34 s 折半**估算**约 1.2 s；若估算成立则搬运长于计算、GPU 等数据。CPU 环境量到的 0.98 s 不等于真实训练循环里的等待（真实循环还有日志同步与 device 交付），所以「搬运是瓶颈」在第 6 步 A/B 步时对比出来之前只是强支持的假设，不是结论。旁证：worker 数、mesh 形状都不改变这一段，w8/w16、(1,8)/(2,4) 全无差别。
- **最简修法**：collate 末尾把 numpy 转成 **torch tensor**（`torch.from_numpy`，零拷贝），torch `DataLoader` 对 tensor 走**共享内存**、主进程只收句柄；`TorchDataLoader.__iter__` 收到后 `.numpy()` 转回（共享内存上的零拷贝视图），后续 `jax.make_array_from_process_local_data` 一字不动。bf16 键（`static_image_emb`，numpy 侧是 `ml_dtypes.bfloat16`）先 `.view(np.uint16)` 再 `.view(torch.bfloat16)`，回程反向，位模式不变。数据路改动集中在 `data_loader.py` 一个文件、两处，约 20 行；`_collate_fn` 本身保持 numpy 语义不动，所有引用它的取证工具不受影响。
- **可行性验证结论（已完成，不启动训练）**：bf16 视图往返逐位一致；合成传输隔离下主进程每批等待 0.816 s → 0.061 s；**真实数据集** b128 w16 下 0.983 s → 0.073 s，且 20 批 × 12 个数组键逐键 sha256(dtype‖shape‖bytes) **全部一致、0 处不匹配**（`F3_COLLATE_EQUIV=PASS batches=20 keys=12 none_keys=4 mismatches=0`）。这些是 CPU 侧数据路的数字，不是训练步时。
- **预期**：若假设成立，8 卡步时从 1.82 s 回到计算主导的约 1.2–1.3 s，80k 步约 41 h → 约 27–29 h。以第 7 步实测为准。
- **后续流程（用户 09-17 定，第二版按审计改）**：切一个**干净的改前副本**（主副本有别人的在途文件，过不了 preflight 的 `REPO_CLEAN`）和一个**改后副本**，主副本锁只读只当权威仓库 → 先提交**基准工具**（preflight 副本模式、bench 白名单、完整性守卫，`C0`）→ 再提交**数据路改动**并过第一块对拍（`C1`）→ 起跑留档（`C2`）→ 8 卡 300 步冒烟 A/B → 8 卡 100 步确定性梯度对拍（改前两次 + 改后一次，**不设量化退路**，改前两次不逐位即停下问用户）→ 结果留档（`C3`）→ push、并回、解锁由用户裁决。

### 0.1 用户决策记录（2026-09-17）

| # | 事项 | 用户答复 | 落地 |
|---|---|---|---|
| 1 | 改后副本分支名 `v2-motionmem-collate-shm`、路径 `/scratch/hongze/robomme_policy_learning_MotionJEPA-temp` | **同意** | 第 4 节第 2 步 |
| 2 | 主副本源码锁只读，直到第 9 步解锁 | **同意** | 第二版里主副本不再是运行侧（见第 7 项），锁只读仍执行：它是权威仓库与实体 `v1-store` 的所在地，锁住防止对比期间被误改 |
| 3 | 8 卡独占约 60 分钟（冒烟 25 + 梯度对拍约 30），现在就开 | **同意** | 第 6、7 步不必再问 |
| 4 | `scripts/dataset/hf_export/` 下两个在途未跟踪文件（另有 tmux 会话 `hf-modul60k-export` 在跑） | **不用管** | 不 add、不动、不等它结束；但它们让主副本过不了 `REPO_CLEAN`，这是第 7 项的直接原因；冒烟留档注明它同期占 CPU/IO |
| 5 | 第 8 步改前两次若不逐位一致，是否自动退回量化判据 | 未答复 | **第二版取消量化退路**（审计第 4 条）：改前两次不逐位一致即停下，把原始判定行交用户裁决，不出任何 PASS |
| 6 | 第 9 步 push 新分支 / 并回 / 解锁 | 届时再问 | — |
| 7 | **（第二版新增，待确认）** 改前副本改为独立干净 worktree `/scratch/hongze/robomme_policy_learning_MotionJEPA-base`（detached 在 `A_HEAD`，`v1-store` symlink，自己的 `.venv`），主副本不再作为运行侧 | 待确认，默认按此执行 | 审计第 2 条：主副本 porcelain 非空过不了 preflight；又不能动别人的在途文件。先例：`m8-modul-retro` 的改前侧也是独立 worktree `v1-store/worktrees/s2-base` |
| 8 | **（第二版新增，待确认）** commit 从 3 个变 4 个：新增 `C0 commitV9.8` 基准工具（preflight 副本模式、bench 白名单加 modul、bench 记源码副本 HEAD、guard 完整性守卫），数据路改动顺延为 `C1 commitV9.9` | 待确认，默认按此执行 | 审计第 1、3、5 条都要改工具；工具与被测改动分开提交，`git diff C0 C1` 只含数据路改动，审计时一眼可核 |

**执行状态**：用户 09-17 指示「不要开始」，本计划**尚未执行**。此前已做的只有第 1 步的**只读核验**（在 HEAD `96b86b5` 上）：源码与 `e4dc733` 零差异、工作区只含第 4 项两个文件、8 卡显存全 0、tmux 现有会话 `0`、`1`、`claude-private`、`codex`、`codex-repo`、`codex2`、`hf-modul60k-export`（均非本计划的，一律不动）。没有建分支、没有改权限、没有改任何仓库文件。正式开始时第 1 步需重做一次核验并以当时 HEAD 为 `A_HEAD`。

### 0.2 Codex 审计（09-17，锚定 `96b86b5`）五条意见与处置

| # | 审计意见 | 核实 | 处置 |
|---|---|---|---|
| 1 | 改后副本必被 preflight 拒绝：`V1_STORE_REAL` 要求 `v1-store` 是实体目录、`NOT_DEV_COPY` 拒绝目录名以 `-temp` 结尾 | 属实（`scripts/training/preflight_train_launch.py` 「起跑位置」段） | C0 给 preflight 加显式 `--bench-copy --v1-store-realpath <主副本>/v1-store` 模式：这两条换成 `V1_STORE_LINK_TARGET`（`v1-store` 必须是 symlink 且 realpath 等于给定主副本实体目录）与 `BENCH_COPY_IS_WORKTREE`（该目录出现在主副本 `git worktree list --porcelain` 里）；其余 24 条（`CWD`、`TRAIN_PY`、`YAML_*`、`PYTHONPATH`、`PKG_*`、`SYS_PREFIX`、`REPO_HEAD`、`REPO_CLEAN`、数据与 CLI 各条）**一条不放宽**，`wt` 仍取 `--repo`。不带 `--bench-copy` 时行为与现在逐字节相同 |
| 2 | 改前侧「允许两个未跟踪文件」与 `REPO_CLEAN`（porcelain 必须为空）冲突，主副本无法起跑 | 属实 | 改前侧改为独立干净 worktree `-base`（决策第 7 项）；worktree 有自己的 index，主副本的未跟踪文件不出现在它的 porcelain 里；`v1-store`、`.venv` 均已 gitignore（`.gitignore` 第 214、140 行），symlink 与 venv 不弄脏它。两侧都是 worktree、都走同一个 `--bench-copy` preflight，对称 |
| 3 | 100 步梯度对拍入口 `bench_train_steps.py` 的 `_EXPECTED_HISTORY_CONFIGS` 只有四个 context 配置，modulation YAML 直接 `ValueError` | 属实（第 107 行附近） | C0 白名单加 `perceptual-framesamp-modul-8frame-8x8.yaml`（只加本次要用的这一个精确文件名）。三侧都用 C0/C1 的工具跑各自副本的源码：工具从改后副本取，源码由各副本自己的 `.venv`（editable 指向该副本 `src/`）决定；C0 再加 `BENCH_SOURCE_ROOT` 环境变量，run_meta 同时记 `tool_head`（工具副本）与 `source_head`（源码副本），并 fail-loud 断言 `openpi.training.data_loader.__file__` 在 `BENCH_SOURCE_ROOT` 下——防止「以为在跑改前、实际 import 了改后」 |
| 4 | 量化退路没接通（未传 `--null-pair`、状态数组未落盘），且 `compare_baseline.py::leaf_numeric_stats` 只打印逐叶误差、无阈值裁决，退出码只跟确定性判定 | 属实 | **本轮取消量化退路**。第 7 步判据只有逐位：A1/A2 与 A1/B 都要 `DET_CHECK=PASS`。A1/A2 自身不逐位 ⇒ 停下，把两份 `SCALARS` / `STATE_DIGEST` 原文交用户，不出 PASS、不降级 |
| 5 | 汇总 PASS 缺完整性守卫：比较器只比共同步集合、索引只比最短公共前缀且失败不影响退出码，退出 0 不保证覆盖 100 步 / 12800 个索引 / 六个摘要步；先例另用了 `finish_check.py` | 属实 | C0 把 `docs/training-doc/t8-c8-guard-s100/records/finish_check.py` 移植为参数化的 `scripts/training/g0/guard_finish_check.py`：对 a1、a2、b **逐侧**断言 `metrics.jsonl` 步集合 == `range(100)`、`param_checksums.jsonl` 与 `batch_digests.jsonl` 步集合 == `[0,1,2,24,49,99]`、`index_sequence.json` 长度 ≥ 12800 且三侧前 12800 个逐个相等、两份比较日志各含全部判定行且 `DET_CHECK=PASS`；全部成立才打 `COLLATE_SHM_GUARD=PASS …`，否则非零退出 |
| — | 性能论证收紧：CPU loader 等待 0.983 s 不能证明搬运长于推算的 1.2 s 计算 | 接受 | 第 0、2 节措辞改为「假设，待第 7 步裁决」，1.2 s 标明是估算 |

### 1. Context：为什么做这件事

`0916-motion-modul-8x8-plan.md` 的正式 80k run 拟用 8 卡。`bench-m8x8-8gpu-worker`（起跑 commit `9cbb94e`，配置 `mme_vla_suite_b128_60k` + `perceptual-framesamp-modul-8frame-8x8.yaml` 关闭态，新库 `4task-v2-1600ep-604f16da/framesamp-8x8`，各 300 步）三档实测：

| 档 | mesh | worker | 步时均值 | util 均值 | 0% 采样占比 | 80k 外推 |
|---|---|---|---|---|---|---|
| f8-w8 | (1,8) | 8 | 1.847 s | 51.0% | 45.5% | 41.0 h |
| f8-w16 | (1,8) | 16 | **1.818 s** | 51.6% | 43.5% | 40.4 h |
| f4-w16 | (2,4) | 16 | 1.833 s | 52.6% | 43.6% | 40.7 h |
| 4 卡基线 `v2-1600ep-m8x8-modul-b128-60k` | (1,4) | 8 | 2.340 s | 72.4% | 25.4% | 52.0 h |

三档相差 ≤ 1.6%，说明瓶颈不在 worker 侧也不在通信，在所有配置共用的主进程数据路。

### 2. 假设的根因机制（三层，每层给依据；最终由第 7 步 A/B 裁决）

**层 1：collate 交付 numpy ⇒ worker→主进程走 pickle。** `_collate_fn` 是
`jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)`。torch `DataLoader` 的 worker 把 collate 结果 `put` 进 `multiprocessing.Queue`，`ForkingPickler` 只对 **torch tensor** 注册了共享内存归约（把 storage 搬到 `/dev/shm`、只传 fd），numpy 数组按普通对象整块序列化成字节流。b128 host batch 逐键实测（CPU、b2 换算）：

| 键 | dtype | 每样本形状 | 每样本字节 |
|---|---|---|---|
| `static_image_emb` | bfloat16（ml_dtypes） | (512, 2048) | 2,097,152 |
| `static_pos_emb` | float32 | (512, 768) | 1,572,864 |
| `image/base_0_rgb`、`image/left_wrist_0_rgb` | uint8 | (224, 224, 3) ×2 | 301,056 |
| `static_state_emb` | float64 | (512, 8) | 32,768 |
| `actions` | float64 | (20, 32) | 5,120 |
| `tokenized_prompt` / `_mask`、`state`、`static_mask`、`image_mask/*` | int64 / bool / float64 / bool / bool | 小 | < 1,500 |
| `mem_order`、`motion_emb`、`motion_mask`、`motion_pos` | None（关闭态） | — | 0 |
| **合计** | 12 个数组键 + 4 个 None | | **4,010,306 B/样本 ⇒ b128 = 513.3 MB/batch** |

进程内实测该 batch `ForkingPickler.dumps` 0.720 s、`loads` 0.344 s（`v1-store/bench/collate-shm-feasibility/f2_transport.log`）。

**层 2：主进程串行收包。** `TorchDataLoader.__iter__` 在主线程 `next(data_iter)` → 队列 `get`（读字节流 + `loads`，0.34 s 起，管道读取还要排队）→ `jax.make_array_from_process_local_data`（拷 8 卡）。没有 `pin_memory` 线程、没有后台预取线程，这一段无法与任何东西重叠。真实数据集 b128 w16 在 CPU 环境实测主进程每批等待均值 **0.983 s、峰值 13.2 s**（`f3_real_loader.log`）。

**层 3：训练循环里搬运与计算已经重叠，但搬运（推测）更长。** `scripts/training/train.py` 主循环是 `ptrain_step(...)`（JAX 异步派发，立即返回）→ `batch = next(data_iter)`，所以 GPU 算第 N 步时主机在取第 N+1 批；每步 ≈ max(计算, 搬运 + 拷卡 + 每 10 步一次 `device_get` 同步)。4 卡计算 2.34 s 盖住搬运，util 72%；8 卡计算**估算**约 1.2 s（2.34 s 折半），盖不住，util 51%。这一层的 1.2 s 与「盖不住」都是推算，不是实测，第 7 步 A/B 才是裁决。

### 3. 改动方案与可行性验证

**数据路改动（C1，全部在 `src/openpi/training/data_loader.py`，外加 bench 入口一行）**：

1. 新增两个纯函数：`_to_shared_torch(batch)`——对每个 numpy 叶子 `torch.from_numpy`（bf16 先 `.view(np.uint16)` 再 `.view(torch.bfloat16)`），None 叶子原样；`_from_shared_torch(batch)`——每个 tensor 叶子 `.numpy()`（bf16 先 `.view(torch.uint16)` 再 `.numpy().view(ml_dtypes.bfloat16)`）。两者都是 `jax.tree.map`，Observation dataclass 结构原样保留。
2. 新增 `_collate_fn_shm(items) = _to_shared_torch(_collate_fn(items))`，`TorchDataLoader.__init__` 的 `collate_fn=_collate_fn` 改为 `collate_fn=_collate_fn_shm`。**`_collate_fn` 本身不改**（`scripts/training/tests/single_step_grad.py`、`dump_fixture_samples.py` 等直接 import 它做取证，语义保持 numpy）。
3. `TorchDataLoader.__iter__` 里 `batch = next(data_iter)` 之后紧接 `batch = _from_shared_torch(batch)`，其余（`make_array_from_process_local_data` / `torch.as_tensor` 两个分支）一字不动。
4. 连带一处：`scripts/training/g0/bench_train_steps.py::iter_with_digest` 重实现了 `__iter__`（在 `make_array` 前插摘要），同样在 `next(data_iter)` 后加 `batch = getattr(_openpi_dl, "_from_shared_torch", lambda b: b)(batch)`——改前副本的 `data_loader.py` 没有这个函数、batch 本来就是 numpy，恒等即可；若函数缺失而 batch 却含 tensor，下一行 `np.asarray(bf16 tensor)` 会直接抛错，不会静默错判。

`num_workers=0` 时 collate 在主进程执行，`from_numpy` 与 `.numpy()` 都是零拷贝视图、不经 pickle，交付内容与现状逐位相同。

**基准工具改动（C0，不含任何数据路改动）**：见第 0.2 节第 1、3、5 条处置与第二部分 B 节。

**可行性验证（F1–F4，全部不启动训练、仓库源码零改动，monkeypatch 只在脚本内；脚本与原始输出在 `v1-store/bench/collate-shm-feasibility/`，未入 git、Codex 审计未复验）**：

| 编号 | 内容 | 结果 |
|---|---|---|
| F1 | bf16 `ml_dtypes → uint16 view → torch.bfloat16 → 反向` 逐位往返（512×2048 随机） | `BF16_VIEW_ROUNDTRIP=PASS`，`tobytes()` 相等 |
| F2 | 合成传输隔离：worker 返回缓存样本（零生成成本）、同键同 dtype 同形状、b128、8 worker、10 批 | 主进程等待 numpy **0.816 s**（峰 3.96）→ shm **0.061 s**（峰 0.19）；进程内 dumps 0.720 / loads 0.344 s，`share_memory_` 0.284 s |
| F3 | 真实数据集：`mme_vla_suite.training.dataloader.create_data_loader` 建的 loader（新库、modul 8×8 关闭态、b128、w16、seed 42、`JAX_PLATFORMS=cpu`），旧 collate 与新 collate 各 20 批 | 主进程稳态等待 **0.983 s**（峰 13.2）→ **0.073 s**（峰 0.098）；`F3_COLLATE_EQUIV=PASS batches=20 keys=12 none_keys=4 mismatches=0` |
| F4 | 资源边界 | `/dev/shm` 561 G（16 worker × prefetch 2 × 513 MB ≈ 16 G）；`ulimit -n` 1,048,576（每批 12 个 fd）；torch 2.7.1 默认 `file_descriptor` 共享策略 |

F3 判定同时证明「抽样顺序相同」：两侧独立构造、同一 seed 42 的 `torch.Generator`，20 批摘要逐批一致，顺序若不同摘要不可能相等。第 4 步会把 F3 正式化为仓库内工具并重跑。

### 4. 完整步骤流程与 commit 清单

**一句话说整个流程**：从主副本切两份干净副本——一份「改前」原样不动、一份「改后」用来改；先提交基准工具，再提交那 20 行改动；先在 CPU 上证明喂进模型的数据和改前一模一样；再在 8 卡上改前、改后各跑 300 步比速度；再各跑 100 步证明 loss 和梯度也一模一样；最后留档，由你决定要不要合并回主线。

**三个目录、两个运行侧**——改前和改后是磁盘上两个不同的目录、两个不同的 git 位置，同时存在；主副本本身不再运行任何东西，只当权威仓库和数据所在地：

| | 主副本（权威） | 改前副本（A 侧） | 改后副本（B 侧） |
|---|---|---|---|
| 目录 | `/scratch/hongze/robomme_policy_learning_MotionJEPA` | `/scratch/hongze/robomme_policy_learning_MotionJEPA-base` | `/scratch/hongze/robomme_policy_learning_MotionJEPA-temp` |
| git | 分支 `v2-motionmem`，HEAD 记作 `A_HEAD` | `git worktree add --detach … A_HEAD`，detached，clean | `git worktree add -b v2-motionmem-collate-shm … A_HEAD`，加 4 个 commit |
| 源码 | 第 3 步起锁只读，全程不加 commit | 与 `A_HEAD` 逐字节相同，全程不动 | C0 工具 + C1 数据路改动 |
| `v1-store` | 实体目录（唯一一份数据） | symlink → 主副本 `v1-store` | symlink → 主副本 `v1-store` |
| `.venv` | 现有的，不动 | 自己一份（`uv sync --frozen`） | 自己一份（`uv sync --frozen`） |
| 为什么需要 | 别人的两个在途文件在这里，porcelain 非空过不了 `REPO_CLEAN`，又不能动它们 | 干净、可被 preflight 接受的「改前」 | 干净、可被 preflight 接受的「改后」 |
| 角色 | 不运行 | 跑冒烟 A 档、梯度对拍 a1/a2 | 改代码、跑冒烟 B 档、梯度对拍 b、写留档 |

**九步，每步一句话**（命令、参数、判定行原文见第二部分对应小节）：

| # | 在哪个副本 | 做什么 | 为什么 | 产出 commit | 通过条件 |
|---|---|---|---|---|---|
| 1 | 主副本 | 确认起点：HEAD 与 `e4dc733` 源码一致、除那两个在途文件外无其他改动、8 卡空闲、记下 tmux 会话清单；记下 `A_HEAD` | 后面所有「改前」都以这个 HEAD 为准 | 无 | 不符就停下来问你 |
| 2 | 主副本 → 建两个副本 | 从 `A_HEAD` 切出改前副本 `-base`（detached）和改后副本 `-temp`（新分支）；两边 `v1-store` 用 symlink 共享，各装自己的 `.venv` | 两侧物理隔离、都干净、数据不复制第二份 | 无 | 两边 HEAD 都等于 `A_HEAD`、porcelain 都为空 |
| 3 | 主副本 | 把主副本源码目录设成只读，记下三份关键文件的 sha256 与 `A_HEAD` | 权威仓库在对比期间不被误改 | 无 | 目录权限变成只读 |
| 4 | 改后副本 | 提交**基准工具**：preflight 加 `--bench-copy` 模式、bench 白名单加 modul、bench 记 `BENCH_SOURCE_ROOT`、新增 `guard_finish_check.py`；跑工具自检 | 审计第 1、3、5 条；工具与被测改动分开提交 | **C0 `commitV9.8`** | preflight 默认模式对主副本只挂 `REPO_CLEAN` 一条、对 `-base`/`-temp` 带 `--bench-copy` 全过；bench 以 modul YAML `--num-train-steps 1` 能起 |
| 5 | 改后副本 | 落那 20 行数据路改动，加两个小测试；在 CPU 上跑「改前副本 vs 改后副本」逐键字节对拍 | 先不碰 GPU 就证明喂进模型的数据完全一样（第 18 条第一块） | **C1 `commitV9.9`** | 对拍 0 处不匹配，测试全绿；否则不提交 |
| 6 | 改后副本 | 写起跑留档 | 记下两侧 commit、命令、配置，满足 preflight 的 clean HEAD 要求 | **C2 `docs:`** | 改后副本工作区干净 |
| 7 | 两个副本各起一次，8 卡独占 | 先从改前副本起 300 步，跑完再从改后副本起 300 步，同配置，比步时和 GPU 利用率 | 这是你要的速度对比，也是对第 2 节假设的裁决；改前那档还要复现上次的 1.82 s 作校验 | 无 | 两档 `PREFLIGHT=PASS`、正常退出；得到 old / new 步时与提速倍数 |
| 8 | 两个副本各起，8 卡独占 | 改前副本起两次、改后副本起一次，各 100 步、确定性模式，逐步比 loss / 梯度范数 / 参数摘要 / 输入摘要 / 样本索引，再跑完整性守卫 | 证明训练本身没被改变（第 18 条第二块）；改前跑两次是为了知道「同代码重跑」本身是否逐位一致 | 无 | 两组比较都逐位一致且守卫通过；**改前两次自身不逐位 ⇒ 停下问你，不降级** |
| 9 | 改后副本 → 你 | 写结果留档，收原始记录，删临时 run 产物；然后问你三件事：推不推新分支到远端、合不合并回 `v2-motionmem`、解不解锁主副本并删两个副本 | 第 12、17 条要求；新分支没有 upstream、合并与解锁都是你的决定 | **C3 `docs:`** | 分支上依次是 C0、C1、C2、C3；你答复前主副本保持只读、分支留在本地 |

**三条规则**：
- 改前侧永远是改前副本的 `A_HEAD`，改后侧永远是改后副本的 C2（C2 与 C1 源码相同，只多一份起跑留档）；三个目录全程并存，直到第 9 步你决定后才删两个副本。
- 四个 commit 先留本地，第 9 步一次性请示 push（新分支没有 upstream，按第 11 条不能自行 `-u`）。
- 任一步的通过条件不满足就停在那一步，不跳步、不放宽判据，把原始判定行给你；第 7、8 步失败也照样写结果留档。

### 5. 冒烟速度对比（8 卡、300 步、同会话 A/B）

照抄 `bench-m8x8-8gpu-worker` 的 runner 形制（`v1-store/logs/bw-8gpu-runner.sh`），改三处：run 名、`MAIN` 按侧取副本路径、preflight 带 `--bench-copy`：

- **配置**：`mme_vla_suite_b128_60k` + `perceptual-framesamp-modul-8frame-8x8.yaml`（sha256 `5b5ac2f8…`，关闭态）、新库 `4task-v2-1600ep-604f16da/framesamp-8x8`、norm_stats `856c75ea…`、`CUDA_VISIBLE_DEVICES=0-7`、`--fsdp-devices 8`（mesh (1,8)）、`--num-workers 16`、`--num-train-steps 300 --log-interval 10 --no-wandb-enabled`、500 ms 密采。与 f8-w16 档完全同口径。
- **A 侧（改前）**：改前副本 `-base`，HEAD `A_HEAD`；run 名 `bench-collate-shm-old`。preflight 脚本用**改后副本 C2 那份**（工具），`--repo` 指向 `-base`、`--train-head A_HEAD`、`--bench-copy --v1-store-realpath <主副本>/v1-store`，在 `-base` 目录下用 `-base/.venv` 运行（`SYS_PREFIX`、`PKG_*`、`TRAIN_PY` 都按 `-base` 判）；train.py 是 `-base` 自己的。
- **B 侧（改后）**：改后副本 `-temp`，clean HEAD = C2；run 名 `bench-collate-shm-new`；preflight 与 train.py 都是它自己的。
- **顺序**：A → B 串行、8 卡独占（bench-b128-util 实测并行互扰 +11–13%）。
- **判读**：稳态窗口 step 100→290；步时均值为主判据；util 按 AGENTS.md 第 16 条给均值、0% 占比、慢步/非慢步分层均值，禁止以中位数作结论。A 侧应复现 1.82 s ± 5%（与 f8-w16 互为校验）。
- **判定行**：每档 `PREFLIGHT=PASS n=<N>`（`--bench-copy` 模式下条数以实际为准，C0 自检时记下）、`EXIT_CODE=0`；分析输出 `BENCH_8GPU workers=16 step_mean_s=… util_mean=… zero_share=…`；汇总 `COLLATE_SHM_SPEED old=… new=… speedup=…`。
- **留档**：两档各约 12 min，超过 5 min，按第 17 条在 `docs/training-doc/bench-collate-shm-8gpu/` 留档；末步 ckpt 与 `v1-store/bench/bench-collate-shm-*` 验收后删除。

### 6. 一致性检验（AGENTS.md 第 18 条两块）

**链路图（改前 → 改后）**，每跳标「有没有改数」：

```
改前  FrameSampDataset.__getitem__ ─numpy─▶ transform_dataset ─numpy─▶ _collate_fn(np.stack) ─numpy 513 MB─▶
      [worker: ForkingPickler.dumps 0.72 s] ═══ 队列管道 ═══▶ [主进程: loads 0.34 s] ─▶ make_array_from_process_local_data ─▶ 8 卡
改后  FrameSampDataset.__getitem__ ─numpy─▶ transform_dataset ─numpy─▶ _collate_fn(np.stack) ─numpy─▶ _to_shared_torch(零拷贝视图)
      ─torch─▶ [worker: storage 搬入 /dev/shm 0.28 s] ═══ 只传 fd ═══▶ [主进程: _from_shared_torch 零拷贝视图 <1 ms] ─numpy─▶ make_array… ─▶ 8 卡
```
所有跳里唯一「改数」的位置是 bf16 的 dtype 视图（uint16 ⇄ bfloat16），字节不变；其余跳都是搬运或视图。

- **第一块（非训练轻量对拍，第 5 步）**：把 F3 脚本正式化为 `scripts/training/tests/compare_collate_paths.py`（C1 新增），对拍口径不变：同库同 seed、b128 w16、N=20 批、逐键 `sha256(dtype‖shape‖bytes)` 与抽样序列；旧侧在改前副本 `.venv` 下跑、新侧在改后副本 `.venv` 下跑，各自 dump jsonl，再由脚本 `compare` 子命令比对。再加一份 pytest `scripts/training/tests/test_collate_shm.py`：合成 dict（bf16/f32/f64/uint8/int64/bool/None 各一）经 2 个 spawn worker 的 `DataLoader` 走新 collate，与单进程 `_collate_fn` 逐键 `tobytes()` 相等。判据逐位。
- **第二块（8 卡真实训练梯度一致，第 8 步，收尾检验）**：`bench_train_steps.py`（改后副本 C2 那份，作为工具）起 100 步，`mme_vla_suite_b128_60k --fsdp-devices 8 --batch-size 128 --num-workers 16 --seed 42 --log-interval 1 --save-interval 1 --no-wandb-enabled --model.history-config perceptual-framesamp-modul-8frame-8x8.yaml`，`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`，`BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1 BENCH_DIGEST_INTERVAL=1000 BENCH_EXTRA_DIGEST_STEPS=1,2,24,49`，`BENCH_SOURCE_ROOT=<该侧副本>`。三侧同会话串行：a1、a2（在 `-base` 目录、`-base/.venv` 下运行）、b（在 `-temp`、`-temp/.venv` 下）。每侧先 `check_baseline_env.py dump`，a2、b 各对 a1 `check`（数据指纹同源）。`compare_baseline.py a1 a2`、`compare_baseline.py a1 b`，两组都要 `SCALARS steps=100 keys=5 hex_mismatch_steps=0`、`INDEX_SEQ=PASS`、`BATCH_DIGEST rows=6 mismatch=0`、`STATE_DIGEST rows=6 mismatch=0`、`CANON_CHECK=PASS steps=6`、`DET_CHECK=PASS`。最后 `guard_finish_check.py` 逐侧核完整性，打 `COLLATE_SHM_GUARD=PASS scalars_steps=100 index_n=12800 batch_digest_rows=6 state_digest_rows=6 sides=3 pairs=2`。**不传 `--null-pair`、不落状态数组、没有量化退路**：a1/a2 自身 `DET_CHECK=FAIL` ⇒ 停下交用户。第二块不通过不得宣称等价。

### 7. 风险与边界

- **不改训练语义**：collate 的 `np.stack` 顺序、dtype、shape、抽样 generator 全部不动；不改超参、不改 config 默认值。
- **工具与源码来自不同副本**：a1/a2 与冒烟 A 档用改后副本的 preflight / bench 工具跑改前副本的源码。防呆两道：preflight `PKG_*`/`SYS_PREFIX`/`TRAIN_PY` 按 `--repo` 判、bench 断言 `data_loader.__file__` 在 `BENCH_SOURCE_ROOT` 下；run_meta 同时记 `tool_head` 与 `source_head`。先例：`m8-modul-retro` 改前侧即此形制。
- **`--force` 红线**：两个副本的 `v1-store` 都是指向主副本的可写 symlink（AGENTS.md 第 14 条例外），本计划所有命令都不带 `--force`、不传输出根参数；run 产物只写 `v1-store/train-runs/mme_vla_suite_b128_60k/{bench-collate-shm-*,cs-guard-*}` 与 `v1-store/bench/{bench-collate-shm-*,8x8/cs-guard-*}`。
- **两个在途未跟踪文件**与 `.claude/worktrees/b128cfg`、`sgab`、`v1-store/worktrees/official-89efeaab` 等别的 worktree 一律不动；`git worktree add` 不影响它们。
- **F2 里看到的 `terminate called without an active exception`**：torch persistent worker 在解释器退出时的收尾噪声，退出码 0；冒烟日志里若出现同样文本，判定以 `EXIT_CODE=` 为准。
- **共享内存策略**：默认 `file_descriptor`；若出现 `Too many open files`，改 `torch.multiprocessing.set_sharing_strategy("file_system")`，作为记录在案的备选而非默认。
- **同期 `hf-modul60k-export`** 占 CPU/IO，可能让两档步时都略高；A/B 同期跑、影响对称，留档注明。
- **不在本计划内**：`static_pos_emb` 改传行号在 device 上索引、后台线程 device 预取——共享内存生效后主进程等待已降到 0.07 s，另立任务。

---

## 第二部分（技术细节，供 agent 追踪）

### A. 三个目录（第 1–3 步）

```bash
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
BASE=/scratch/hongze/robomme_policy_learning_MotionJEPA-base
TEMP=/scratch/hongze/robomme_policy_learning_MotionJEPA-temp
cd "$MAIN"
git status --porcelain            # 仅允许 scripts/dataset/hf_export/ 下两个 ??，其余非空即停
git diff --stat e4dc733 HEAD -- src scripts packages pyproject.toml uv.lock   # 须为空
A_HEAD=$(git rev-parse HEAD); echo "$A_HEAD"
test ! -e "$BASE" && test ! -e "$TEMP"
git worktree add --detach "$BASE" "$A_HEAD"
git worktree add -b v2-motionmem-collate-shm "$TEMP" "$A_HEAD"
for d in "$BASE" "$TEMP"; do
  ln -s "$MAIN/v1-store" "$d/v1-store"                       # 可写链，第 14 条例外；已 gitignore
  ( cd "$d" && UV_CACHE_DIR="$MAIN/v1-store/cache/uv" uv sync --frozen --python "$MAIN/.venv/bin/python" )   # 各自 .venv，保留 dev group（pytest）
  ( cd "$d" && test "$(git rev-parse HEAD)" = "$A_HEAD" && test -z "$(git status --porcelain)" && .venv/bin/python -c "import openpi,sys;print(openpi.__file__, sys.prefix)" )
done
git worktree list                 # 两个新条目 + 既有的 b128cfg / sgab / tic / official 不变
# 主副本锁只读
sha256sum src/openpi/training/data_loader.py scripts/training/g0/bench_train_steps.py scripts/training/preflight_train_launch.py scripts/training/train.py > v1-store/bench/collate-shm-feasibility/lock_sha256.txt
echo "$A_HEAD" > v1-store/bench/collate-shm-feasibility/A_HEAD.txt
chmod -R a-w src scripts packages
ls -ld src scripts packages       # dr-xr-xr-x
```

`uv sync` 约 2–3 min/副本，各占约 7 G（`/scratch` 余 1.1 T）。两个副本的 `openpi.__file__` 必须分别在各自 `src/` 下。

### B. C0 基准工具（第 4 步，改后副本）

1. `scripts/training/preflight_train_launch.py`：加 `--bench-copy`（flag）与 `--v1-store-realpath`（与 `--bench-copy` 必须同时给）。
   - `--bench-copy` 下：`V1_STORE_REAL` → `V1_STORE_LINK_TARGET`（`wt/v1-store` 是 symlink 且 `realpath == --v1-store-realpath` 且后者是实体目录）；`NOT_DEV_COPY` → `BENCH_COPY_IS_WORKTREE`（`git -C <realpath 的父目录> worktree list --porcelain` 的 `worktree <path>` 行含 `wt`）。其余检查与非 bench 模式共用同一段代码，不复制、不放宽。
   - 非 `--bench-copy`：代码路径与现在完全相同（diff 只在两处 `if a.bench_copy` 分支）。
   - 自检（第 4 步通过条件）：在主副本以默认模式跑一次 → 期望恰好 `PREFLIGHT=FAIL failed=REPO_CLEAN`；在 `-base` 与 `-temp` 各以 `--bench-copy` 跑一次 → `PREFLIGHT=PASS n=<N>`，记下 N。
2. `scripts/training/g0/bench_train_steps.py`：`_EXPECTED_HISTORY_CONFIGS` 追加 `"perceptual-framesamp-modul-8frame-8x8.yaml"`；新增环境变量 `BENCH_SOURCE_ROOT`（默认 `_REPO_ROOT`）：启动时 `import openpi.training.data_loader as m; assert pathlib.Path(m.__file__).resolve().is_relative_to(BENCH_SOURCE_ROOT)`，否则 `SystemExit`；`run_meta.json` 记 `tool_head`（`_REPO_ROOT` 的 HEAD）、`source_head`（`BENCH_SOURCE_ROOT` 的 HEAD 与 porcelain）。原 `start_head` 字段保留为 `tool_head` 的别名不删。
3. `scripts/training/g0/guard_finish_check.py`（新增，参数化移植 `docs/training-doc/t8-c8-guard-s100/records/finish_check.py`）：`--sides a1=<dir> a2=<dir> b=<dir> --compare-logs <a1a2.log> <a1b.log> --steps 100 --batch-size 128 --digest-steps 0,1,2,24,49,99 --summary-prefix COLLATE_SHM_GUARD`。逐侧断言见第 0.2 节第 5 条；任一断言失败以非零退出并打 `COLLATE_SHM_GUARD=FAIL reason=…`。
4. 自检不动 GPU：`bench_train_steps.py` 用 modul YAML、`JAX_PLATFORMS=cpu`、`--num-train-steps 1 --batch-size 2 --num-workers 0` 起一步即退（只证白名单与 `BENCH_SOURCE_ROOT` 断言生效）。
5. **C0 `commitV9.8: 基准工具——preflight 副本模式、bench 白名单加 modul 与源码副本断言、guard 完整性守卫`**，逐文件 add 上述三个文件。`git diff A_HEAD C0 -- src/` 必须为空（C0 不含数据路改动）。

### C. C1 数据路改动与第一块工具（第 5 步，改后副本）

`src/openpi/training/data_loader.py`：

```python
import ml_dtypes  # 新增 import

def _to_shared_torch(batch):
    """collate 后（worker 侧）numpy → torch，零拷贝视图；tensor 经 DataLoader 队列时走共享内存只传 fd。
    bf16 在 numpy 侧是 ml_dtypes.bfloat16，torch.from_numpy 不认，先 uint16 视图再 bfloat16 视图，位模式不变。"""
    def f(x):
        if x is None:
            return None
        x = np.asarray(x)
        if x.dtype == ml_dtypes.bfloat16:
            return torch.from_numpy(x.view(np.uint16)).view(torch.bfloat16)
        return torch.from_numpy(x)
    return jax.tree.map(f, batch, is_leaf=lambda x: x is None)

def _from_shared_torch(batch):
    """主进程侧 torch → numpy，共享内存上的零拷贝视图；bf16 反向还原 ml_dtypes.bfloat16。"""
    def f(t):
        if t is None:
            return None
        if t.dtype == torch.bfloat16:
            return t.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
        return t.numpy()
    return jax.tree.map(f, batch, is_leaf=lambda x: x is None)

def _collate_fn_shm(items):
    return _to_shared_torch(_collate_fn(items))
```

`TorchDataLoader.__init__`：`collate_fn=_collate_fn` → `collate_fn=_collate_fn_shm`。
`TorchDataLoader.__iter__`：`batch = next(data_iter)` 之后加 `batch = _from_shared_torch(batch)`。
`scripts/training/g0/bench_train_steps.py::iter_with_digest`：同位置加 `batch = getattr(_openpi_dl, "_from_shared_torch", lambda b: b)(batch)`。

第一块工具：
- `scripts/training/tests/compare_collate_paths.py`：`dump --out <dir> --batches 20 --workers 16` 在当前副本 `.venv` 下用 `create_data_loader(...)._data_loader.torch_loader` 取 N 批（改前副本得 numpy、改后副本得 tensor 后经 `_from_shared_torch`；脚本按 `hasattr` 分支），逐键 `sha256(dtype‖shape‖bytes)` 写 `digests.jsonl`，同时记主进程等待；`compare <dir_old> <dir_new>` 输出 `COLLATE_EQUIV=PASS|FAIL batches=N keys=K none_keys=… mismatches=…` 与两侧等待均值。运行：

  ```bash
  for d in "$BASE" "$TEMP"; do ( cd "$d" && source scripts/training/paths.sh && \
    MMEVLA_FRAMESAMP_SOURCE="$V1_STORE/datasets/4task-v2-1600ep-604f16da/source" \
    MMEVLA_FRAMESAMP_MANIFEST="$V1_STORE/datasets/4task-v2-1600ep-604f16da/meta/episode_manifest.json" \
    JAX_PLATFORMS=cpu uv run --no-sync python "$TEMP/scripts/training/tests/compare_collate_paths.py" dump \
      --out "$V1_STORE/bench/collate-shm-equiv/$(basename $d)" --batches 20 --workers 16 ); done
  ( cd "$TEMP" && JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/tests/compare_collate_paths.py compare \
      "$V1_STORE/bench/collate-shm-equiv/robomme_policy_learning_MotionJEPA-base" "$V1_STORE/bench/collate-shm-equiv/robomme_policy_learning_MotionJEPA-temp" )
  ```
- `scripts/training/tests/test_collate_shm.py`（pytest，< 1 min）：合成 dict 七种叶子经 2 个 spawn worker 的新 collate 与单进程 `_collate_fn` 逐键 `tobytes()` 相等；bf16 往返逐位；None 键保留。`( cd "$TEMP" && uv run --no-sync pytest scripts/training/tests/test_collate_shm.py -q )`。

**C1 `commitV9.9: collate 交付改走 torch 共享内存，主进程数据路等待 0.98 s→0.07 s`**，逐文件 add：`src/openpi/training/data_loader.py`、`scripts/training/g0/bench_train_steps.py`、`scripts/training/tests/compare_collate_paths.py`、`scripts/training/tests/test_collate_shm.py`。`git diff C0 C1 -- src/` 只含 `data_loader.py`。

### D. 冒烟 runner（第 7 步，8 卡、300 步）

复制 `v1-store/logs/bw-8gpu-runner.sh` 为 `v1-store/logs/cs-8gpu-runner.sh`，参数 `<SIDE old|new> <TRAIN_HEAD> <HC_SHA256>`：

- `old`：`MAIN=$BASE`、`TRAIN_HEAD=A_HEAD`；`new`：`MAIN=$TEMP`、`TRAIN_HEAD=C2`。`cd "$MAIN"`，`uv run --no-sync` 用该副本自己的 `.venv`。
- preflight 一律用 `$TEMP/scripts/training/preflight_train_launch.py`（C2 那份），参数 `--repo "$MAIN" --train-head "$TRAIN_HEAD" --bench-copy --v1-store-realpath /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store …`，其余参数与 bw 版相同；train 用 `$MAIN/scripts/training/train.py`（各侧自己的）。
- `RUN=bench-collate-shm-$SIDE`、`--num-workers 16 --fsdp-devices 8`、`MMEVLA_JAX_CACHE_DIR=$V1_STORE/cache/jax/cs-8gpu-$SIDE`（两侧各自缓存）；GPU_IDLE 检查、500 ms 密采、`EXIT_CODE=` 照抄。驱动 `cs-8gpu-driver.sh`：`for SIDE in old new`，串行，末尾 `DRIVER_ALL_DONE`。

```bash
tmux new-session -d -s cs-8gpu "bash /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/cs-8gpu-driver.sh"
tmux has-session -t cs-8gpu
```

Monitor 挂驱动日志、每级行缓冲，过滤 `PREFLIGHT=|EXIT_CODE=|DRIVER_ALL_DONE|Traceback|out of memory|Too many open files`。分析复用 `docs/training-doc/bench-m8x8-8gpu-worker/records/bw_analyze.py`（起跑记录完整版在 commit `f2c9dc7`），窗口 step 100→290。

### E. 第二块 runner（第 8 步，8 卡、100 步、确定性）

`v1-store/logs/cs-guard-runner.sh`，形制照 `docs/training-doc/t8-c8-guard-s100/records/runner.sh`，三侧串行：

| side | cd / .venv | `BENCH_SOURCE_ROOT` | 工具 |
|---|---|---|---|
| a1 | `$BASE` | `$BASE` | `$TEMP/scripts/training/g0/bench_train_steps.py` |
| a2 | `$BASE` | `$BASE` | 同上 |
| b | `$TEMP` | `$TEMP` | 同上 |

每侧：`BENCH_RECORD_DIR=$V1_STORE/bench/8x8/cs-guard-$SIDE`、`MMEVLA_JAX_CACHE_DIR=$V1_STORE/cache/jax/cs-guard-$SIDE`、`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`、`BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1 BENCH_DIGEST_INTERVAL=1000 BENCH_EXTRA_DIGEST_STEPS=1,2,24,49`、`unset BENCH_STATE_DUMP_STEPS BENCH_STATE_DUMP_DIR BENCH_PERF_MODE`；`check_baseline_env.py dump --record-dir …`；然后 `bench_train_steps.py mme_vla_suite_b128_60k --exp-name cs-guard-$SIDE --batch-size 128 --num-workers 16 --num-train-steps 100 --log-interval 1 --save-interval 1 --seed 42 --fsdp-devices 8 --no-wandb-enabled --model.history-config perceptual-framesamp-modul-8frame-8x8.yaml`（数据集 / assets 参数与 D 相同）；`project_scalars.py` 出 `scalars_hex.tsv`。a2、b 各 `check_baseline_env.py check --base <a1 dir> --record-dir <own> --steps 100 --batch-size 128 --dataset …` → `BASELINE_ENV=PASS`。收尾：

```bash
compare_baseline.py "$A1" "$A2" --tier cs-aa | tee "$REC/compare-a1a2.log"
compare_baseline.py "$A1" "$B"  --tier cs-ab | tee "$REC/compare-a1b.log"
guard_finish_check.py --sides a1="$A1" a2="$A2" b="$B" --compare-logs "$REC/compare-a1a2.log" "$REC/compare-a1b.log" \
  --steps 100 --batch-size 128 --digest-steps 0,1,2,24,49,99 --summary-prefix COLLATE_SHM_GUARD
```

`compare-a1a2.log` 的 `DET_CHECK=FAIL` ⇒ 驱动立即停止（不跑 a1/b 比较的 PASS 判定），把两份日志交用户。tmux 会话 `cs-guard`。

### F. 留档、提交、清理（第 6、9 步）

- `docs/training-doc/bench-collate-shm-8gpu/`：`launch.md`（`A_HEAD`、C0/C1/C2 SHA、三个目录、命令、会话清单 `cs-8gpu`、`cs-guard`，C2 提交）；`result.md`（速度表、util 分层、第一块与第二块判定行、链路图、`hf-modul60k-export` 同期说明）；`records/`（`metrics.jsonl` × 5、`judgement_lines.txt`、`analysis.json`、对拍 jsonl、`compare-*.log`、`guard_finish.log`、三个 runner 副本），C3 提交。
- 四个 commit 全在 `v2-motionmem-collate-shm`，逐文件 `git add`，不碰两个在途 `??` 文件。push 需先问用户（新分支无 upstream）。
- 并回 `v2-motionmem` 与否由用户决定（四 commit 线性、可 `--ff-only`）；并回前主副本解锁 `chmod -R u+w src scripts packages`，先 `sha256sum -c v1-store/bench/collate-shm-feasibility/lock_sha256.txt` 四份文件全 OK、`git status --porcelain` 仍只含那两个在途文件；`-base`、`-temp` 用 `git worktree remove` 删除（先确认里面没有未提交内容；`v1-store` symlink 随目录删，主副本数据不受影响）。
- 清理：`v1-store/train-runs/mme_vla_suite_b128_60k/{bench-collate-shm-old,bench-collate-shm-new,cs-guard-a1,cs-guard-a2,cs-guard-b}`；`v1-store/bench/bench-collate-shm-*`、`v1-store/bench/8x8/cs-guard-*` 拷入 records 后删；tmux 会话随驱动自然退出，如需手动只允许 `tmux kill-session -t cs-8gpu` / `-t cs-guard`，删前删后各 `tmux ls`。

### G. 验证清单（完成判据）

1. C0 自检：主副本默认模式 `PREFLIGHT=FAIL failed=REPO_CLEAN`（且仅此一条）；`-base`、`-temp` `--bench-copy` 均 `PREFLIGHT=PASS n=<N>`；bench 以 modul YAML 在 CPU 起 1 步成功；`git diff A_HEAD C0 -- src/` 为空。
2. C1：`COLLATE_EQUIV=PASS batches=20 keys=12 none_keys=4 mismatches=0`（改前副本 vs 改后副本）+ pytest 全绿；`git diff C0 C1 -- src/` 只含 `data_loader.py`。
3. 冒烟：两侧 `PREFLIGHT=PASS`、`EXIT_CODE=0`；old 步时与 f8-w16 的 1.818 s 相差 ≤ 5%；new 步时、util 均值、0% 占比三项同时给出；`COLLATE_SHM_SPEED old=… new=… speedup=…`。
4. 第二块：a1/a2 与 a1/b 两组 `DET_CHECK=PASS` + 两组 `BASELINE_ENV=PASS` + `COLLATE_SHM_GUARD=PASS scalars_steps=100 index_n=12800 batch_digest_rows=6 state_digest_rows=6 sides=3 pairs=2`。a1/a2 不逐位 ⇒ 无 PASS，交用户。
5. 解锁前 `lock_sha256.txt` 四份文件 sha256 不变、主副本 `git status --porcelain` 只含那两个在途文件、`git worktree list` 里两个副本条目干净。
