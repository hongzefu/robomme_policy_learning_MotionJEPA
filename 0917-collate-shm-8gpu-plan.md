# 0917 计划：collate 改走共享内存，吃满 8 卡（基准：8 帧 × 8×8 modulation 关闭态）

> 创建日期 2026-09-17（America/New_York）。计划入库 commit `5b2e7ba`（`v2-motionmem`，clean；源码与 `e4dc733` 完全相同，`5b2e7ba` 只多本计划文件）。后续所有「改前 / A 侧」一律指 `5b2e7ba`。
> 本计划只覆盖 dataloader 交付路径的一处改动、它的一致性检验与 8 卡速度对比；不碰 motion、不碰模型、不碰超参。
> 可行性验证（不启动训练）已在写本计划前完成，结果内联于第一部分第 3 节；产物固化在 `v1-store/bench/collate-shm-feasibility/`（不进 git）。

---

## 第一部分（给人看）

### 0. 结论先行

- **问题**：8 卡 mesh (1,8) 训练步时 1.82 s、GPU util 均值 51%、45% 采样为 0%，比 4 卡只快 22%（`docs/training-doc/bench-m8x8-8gpu-worker/result.md`）。
- **根因**：`src/openpi/training/data_loader.py::_collate_fn` 返回 **numpy** 数组，torch `DataLoader` 对 numpy 只能整批 pickle 后经队列管道送到主进程；b128 一个 batch **513 MB**，pickle 的 `dumps` 在 worker、`loads` 在主进程**串行**发生，主进程每批平均等 **0.98 s**（真实数据集实测，峰值 13 s）。8 卡下 GPU 计算约 1.2 s，搬运比计算长，GPU 等数据。worker 数、mesh 形状都不改变这一段，所以 w8/w16、(1,8)/(2,4) 全无差别。
- **最简修法**：collate 末尾把 numpy 转成 **torch tensor**（`torch.from_numpy`，零拷贝），torch `DataLoader` 对 tensor 走**共享内存**、主进程只收句柄；`TorchDataLoader.__iter__` 收到后 `.numpy()` 转回（共享内存上的零拷贝视图），后续 `jax.make_array_from_process_local_data` 一字不动。bf16 键（`static_image_emb`，numpy 侧是 `ml_dtypes.bfloat16`）先 `.view(np.uint16)` 再 `.view(torch.bfloat16)`，回程反向，位模式不变。改动集中在 `data_loader.py` 一个文件、两处，约 20 行；`_collate_fn` 本身保持 numpy 语义不动，所有引用它的取证工具不受影响。
- **可行性验证结论（已完成，不启动训练）**：bf16 视图往返逐位一致；合成传输隔离下主进程每批等待 0.816 s → 0.061 s（13.3×）；**真实数据集** b128 w16 下 0.983 s → 0.073 s（13.4×），且 20 批 × 12 个数组键逐键 sha256(dtype‖shape‖bytes) **全部一致、0 处不匹配**（`F3_COLLATE_EQUIV=PASS batches=20 keys=12 none_keys=4 mismatches=0`）。
- **预期**：8 卡步时从 1.82 s 回到计算主导的约 1.2–1.3 s，80k 步约 41 h → 约 27–29 h。这是推算，以第 4 节冒烟对比的实测为准。
- **后续流程（用户 09-17 定）**：本计划获准后按第 4 节的九步走：切独立分支到 `-temp` 开发副本、主副本源码锁只读 → 在分支上落改动并过第一块对拍（`commitV9.8`）→ 起跑留档（`docs:`）→ 8 卡 300 步冒烟 A/B 对比速度 → 8 卡 100 步确定性梯度对拍（第二块）→ 结果留档（`docs:`）→ push 分支、并回 `v2-motionmem` 与解锁均由用户裁决。

### 1. Context：为什么做这件事

`0916-motion-modul-8x8-plan.md` 的正式 80k run 拟用 8 卡。`bench-m8x8-8gpu-worker`（起跑 commit `9cbb94e`，配置 `mme_vla_suite_b128_60k` + `perceptual-framesamp-modul-8frame-8x8.yaml` 关闭态，新库 `4task-v2-1600ep-604f16da/framesamp-8x8`，各 300 步）三档实测：

| 档 | mesh | worker | 步时均值 | util 均值 | 0% 采样占比 | 80k 外推 |
|---|---|---|---|---|---|---|
| f8-w8 | (1,8) | 8 | 1.847 s | 51.0% | 45.5% | 41.0 h |
| f8-w16 | (1,8) | 16 | **1.818 s** | 51.6% | 43.5% | 40.4 h |
| f4-w16 | (2,4) | 16 | 1.833 s | 52.6% | 43.6% | 40.7 h |
| 4 卡基线 `v2-1600ep-m8x8-modul-b128-60k` | (1,4) | 8 | 2.340 s | 72.4% | 25.4% | 52.0 h |

三档相差 ≤ 1.6%，说明瓶颈不在 worker 侧也不在通信，在所有配置共用的主进程数据路。

### 2. 根因机制（三层，每层给判定依据）

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

**层 2：主进程串行收包。** `TorchDataLoader.__iter__` 在主线程 `next(data_iter)` → 队列 `get`（读字节流 + `loads`，0.34 s 起，管道读取还要排队）→ `jax.make_array_from_process_local_data`（拷 8 卡）。没有 `pin_memory` 线程、没有后台预取线程，这一段无法与任何东西重叠。真实数据集 b128 w16 实测主进程每批等待均值 **0.983 s、峰值 13.2 s**（`f3_real_loader.log`）。

**层 3：训练循环里搬运与计算已经重叠，但搬运更长。** `scripts/training/train.py` 主循环是 `ptrain_step(...)`（JAX 异步派发，立即返回）→ `batch = next(data_iter)`，所以 GPU 算第 N 步时主机在取第 N+1 批；每步 ≈ max(计算, 搬运)。4 卡计算 2.3 s 盖住搬运，util 72%；8 卡计算约 1.2 s 盖不住，util 51%。

### 3. 改动方案与可行性验证

**改动（全部在 `src/openpi/training/data_loader.py`）**：

1. 新增两个纯函数：`_to_shared_torch(batch)`——对每个 numpy 叶子 `torch.from_numpy`（bf16 先 `.view(np.uint16)` 再 `.view(torch.bfloat16)`），None 叶子原样；`_from_shared_torch(batch)`——每个 tensor 叶子 `.numpy()`（bf16 先 `.view(torch.uint16)` 再 `.numpy().view(ml_dtypes.bfloat16)`）。两者都是 `jax.tree.map`，Observation dataclass 结构原样保留。
2. 新增 `_collate_fn_shm(items) = _to_shared_torch(_collate_fn(items))`，`TorchDataLoader.__init__` 的 `collate_fn=_collate_fn` 改为 `collate_fn=_collate_fn_shm`。**`_collate_fn` 本身不改**（`scripts/training/tests/single_step_grad.py`、`dump_fixture_samples.py` 等直接 import 它做取证，语义保持 numpy）。
3. `TorchDataLoader.__iter__` 里 `batch = next(data_iter)` 之后紧接 `batch = _from_shared_torch(batch)`，其余（`make_array_from_process_local_data` / `torch.as_tensor` 两个分支）一字不动。
4. 连带一处：`scripts/training/g0/bench_train_steps.py::iter_with_digest` 重实现了 `__iter__`（在 `make_array` 前插摘要），同样在 `next(data_iter)` 后加一行 `_from_shared_torch`，否则摘要器会拿到 torch bf16 tensor、`np.asarray` 失败。

`num_workers=0` 时 collate 在主进程执行，`from_numpy` 与 `.numpy()` 都是零拷贝视图、不经 pickle，交付内容与现状逐位相同。

**可行性验证（F1–F4，全部不启动训练、仓库源码零改动，monkeypatch 只在脚本内；脚本与原始输出在 `v1-store/bench/collate-shm-feasibility/`）**：

| 编号 | 内容 | 结果 |
|---|---|---|
| F1 | bf16 `ml_dtypes → uint16 view → torch.bfloat16 → 反向` 逐位往返（512×2048 随机） | `BF16_VIEW_ROUNDTRIP=PASS`，`tobytes()` 相等 |
| F2 | 合成传输隔离：worker 返回缓存样本（零生成成本）、同键同 dtype 同形状、b128、8 worker、10 批 | 主进程等待 numpy **0.816 s**（峰 3.96）→ shm **0.061 s**（峰 0.19），**13.3×**；进程内 dumps 0.720 / loads 0.344 s，`share_memory_` 0.284 s |
| F3 | 真实数据集：`mme_vla_suite.training.dataloader.create_data_loader` 建的 loader（新库、modul 8×8 关闭态、b128、w16、seed 42、`JAX_PLATFORMS=cpu`），旧 collate 与新 collate 各 20 批 | 主进程稳态等待 **0.983 s**（峰 13.2）→ **0.073 s**（峰 0.098），**13.4×**；`F3_COLLATE_EQUIV=PASS batches=20 keys=12 none_keys=4 mismatches=0` |
| F4 | 资源边界 | `/dev/shm` 561 G（16 worker × prefetch 2 × 513 MB ≈ 16 G）；`ulimit -n` 1,048,576（每批 12 个 fd）；torch 2.7.1 默认 `file_descriptor` 共享策略 |

F3 判定同时证明「抽样顺序相同」：两侧独立构造、同一 seed 42 的 `torch.Generator`，20 批摘要逐批一致，顺序若不同摘要不可能相等。

### 4. 完整步骤流程与 commit 清单（从 HEAD `5b2e7ba` 起，顺序固定）

约定：**主副本** = `/scratch/hongze/robomme_policy_learning_MotionJEPA`（分支 `v2-motionmem`，A 侧 / 改前）；**开发副本** = `/scratch/hongze/robomme_policy_learning_MotionJEPA-temp`（分支 `v2-motionmem-collate-shm`，B 侧 / 改后）。本计划共产出 **3 个 commit**，全部落在 `v2-motionmem-collate-shm` 分支；`v2-motionmem` 在整个过程中**不再新增 commit**，直到用户裁决并回。

| # | 在哪里 | 做什么 | 产出 commit | 判定行 / 通过条件 |
|---|---|---|---|---|
| 1 | 主副本 | 起点核验：`git rev-parse HEAD` = `5b2e7ba`；`git status --porcelain` 只含 `scripts/dataset/hf_export/` 下两个在途 `??` 文件；8 卡显存全 0；`tmux ls` 记录现有会话清单 | 无 | 任一不符即停，把原始输出交用户 |
| 2 | 主副本 → 开发副本 | `git worktree add -b v2-motionmem-collate-shm <开发副本路径> HEAD`；开发副本内 `ln -s <主副本>/v1-store v1-store`（可写链，第 14 条例外）；`uv sync --frozen` 建独立 `.venv`（`UV_CACHE_DIR` 指 `v1-store/cache/uv`） | 无 | `git -C <开发副本> rev-parse HEAD` = `5b2e7ba`；`ls -ld v1-store` 是 symlink；`.venv/bin/python` 存在且不是主副本的 |
| 3 | 主副本 | 锁只读：`sha256sum src/openpi/training/data_loader.py scripts/training/g0/bench_train_steps.py scripts/training/train.py > v1-store/bench/collate-shm-feasibility/lock_sha256.txt`；`chmod -R a-w src scripts packages` | 无 | `ls -ld src scripts packages` 为 `dr-xr-xr-x`；此后主副本不改文件、不 pull、不跑 `uv sync/add` |
| 4 | 开发副本 | 落代码（第二部分 B 节）：`data_loader.py` 新增 `_to_shared_torch` / `_from_shared_torch` / `_collate_fn_shm`、`__init__` 换 collate、`__iter__` 加转回一行；`bench_train_steps.py::iter_with_digest` 加同一行。新增第一块工具 `scripts/training/tests/test_collate_shm.py`、`scripts/training/tests/compare_collate_paths.py`（第二部分 C 节）。跑 `uv run --no-sync pytest scripts/training/tests/test_collate_shm.py -q` 与 `compare_collate_paths.py --batches 20 --workers 16`（CPU，约 3 min） | **C1 `commitV9.8: collate 交付改走 torch 共享内存，主进程数据路等待 0.98 s→0.07 s`**，逐文件 add 四个文件：`src/openpi/training/data_loader.py`、`scripts/training/g0/bench_train_steps.py`、`scripts/training/tests/test_collate_shm.py`、`scripts/training/tests/compare_collate_paths.py` | pytest 全绿；`COLLATE_EQUIV=PASS batches=20 keys=12 none_keys=4 mismatches=0`；不通过则不提交、回报用户 |
| 5 | 开发副本 | 写 `docs/training-doc/bench-collate-shm-8gpu/launch.md`（两侧 commit：A = `5b2e7ba`，B = C1；命令、配置 sha、会话清单 `cs-8gpu`、`cs-guard`）；把 `cs-8gpu-runner.sh` / `cs-8gpu-driver.sh` / `cs-guard-runner.sh` 落 `v1-store/logs/`（不进 git，副本随第 8 步进 `records/`） | **C2 `docs: bench-collate-shm-8gpu 起跑留档`**，只 add `launch.md` | C2 之后开发副本 `git status --porcelain` 为空（preflight 的 `CHECK_REPO_CLEAN` 要求 B 侧 clean HEAD = C2） |
| 6 | 两侧，8 卡独占 | 冒烟速度 A/B（第 5 节）：`tmux new-session -d -s cs-8gpu "bash <主副本>/v1-store/logs/cs-8gpu-driver.sh"`，驱动串行跑 A（主副本 `5b2e7ba`，run `bench-collate-shm-old`）→ B（开发副本 C2，run `bench-collate-shm-new`），各 300 步 (1,8) w16 b128；Monitor 挂驱动日志 | 无 | 每档 `PREFLIGHT=PASS n=25`、`EXIT_CODE=0`；`DRIVER_ALL_DONE`；分析 `BENCH_8GPU workers=16 step_mean_s=…`×2 与 `COLLATE_SHM_SPEED old=… new=… speedup=…`；old 与 f8-w16 的 1.818 s 相差 ≤ 5% |
| 7 | 两侧，8 卡独占 | 第二块梯度对拍（第 6 节）：`tmux new-session -d -s cs-guard "bash <主副本>/v1-store/logs/cs-guard-runner.sh"`，串行 a1、a2（主副本）、b（开发副本），各 100 步确定性 XLA、`BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1`；`compare_baseline.py a1 a2`、`compare_baseline.py a1 b` | 无 | A1/A2 与 A1/B 均 `SCALARS steps=100 keys=5 hex_mismatch_steps=0`、`INDEX_SEQ=PASS`、`BATCH_DIGEST rows=6 mismatch=0`、`STATE_DIGEST rows=6 mismatch=0`；汇总 `COLLATE_SHM_GUARD=PASS`。A1/A2 自身不逐位则退回 `QUANT_EQUIV`（零假设 A/A、margin 2.0）并在留档注明 |
| 8 | 开发副本 | 写 `result.md`（速度表、util 分层、两块判定行、链路图）；`records/` 收 `metrics.jsonl`（old/new/a1/a2/b）、`judgement_lines.txt`、`analysis.json`、对拍 jsonl、`comparisons.log`、三个 runner 副本；清理 `v1-store/train-runs/mme_vla_suite_b128_60k/{bench-collate-shm-old,bench-collate-shm-new,cs-guard-a1,cs-guard-a2,cs-guard-b}`、`v1-store/bench/bench-collate-shm-*`、`v1-store/bench/8x8/cs-guard-*` | **C3 `docs: bench-collate-shm-8gpu 结果——…`**，add `result.md` + `records/` 逐文件 | 提交后 `git log --oneline -3` 为 C3 / C2 / C1，其下是 `5b2e7ba` |
| 9 | 主副本 + 用户 | 三件事都**先问用户**：(a) `git push -u origin v2-motionmem-collate-shm`（新分支无 upstream，第 11 条禁止自行 `-u`）；(b) 是否 `git merge --ff-only v2-motionmem-collate-shm` 并回 `v2-motionmem`（三 commit 线性、可快进）；(c) 解锁主副本 `chmod -R u+w src scripts packages`——解锁前先 `sha256sum -c v1-store/bench/collate-shm-feasibility/lock_sha256.txt` 三份文件全 OK、`git status --porcelain` 仍只含那两个在途文件 | 无新 commit | 用户答复前主副本保持只读、分支保持本地 |

三点补充：

- **改前 / 改后两侧的 commit 固定为 `5b2e7ba` / C2**，冒烟与第二块都用这两个；C1 与 C2 源码相同（C2 只加 `launch.md`），留档写 C2 是因为 preflight 要求 clean HEAD。
- **每次 `git commit` 后照第 11 条应立即 push**，但本分支远端不存在，`git push` 会因无 upstream 失败，这是第 11 条明文要求先问用户的情形；因此三个 commit 先留本地，在第 9 步一次性请示。
- **中途任一判定失败**：停在该步，不跳步、不改判据、不放宽阈值，把原始判定行交用户；第 4 步失败不提交 C1；第 6 或 7 步失败照样写 `result.md` 留档（C3 记失败原因）。

### 5. 冒烟速度对比（8 卡、300 步、同会话 A/B）

照抄 `bench-m8x8-8gpu-worker` 的 runner 形制（`v1-store/logs/bw-8gpu-runner.sh`），只改 run 名与代码副本：

- **配置**：`mme_vla_suite_b128_60k` + `perceptual-framesamp-modul-8frame-8x8.yaml`（sha256 `5b5ac2f8…`，关闭态）、新库 `4task-v2-1600ep-604f16da/framesamp-8x8`、norm_stats `856c75ea…`、`CUDA_VISIBLE_DEVICES=0-7`、`--fsdp-devices 8`（mesh (1,8)）、`--num-workers 16`、`--num-train-steps 300 --log-interval 10 --no-wandb-enabled`、500 ms 密采。与 f8-w16 档完全同口径。
- **A 侧（改前）**：主副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，HEAD `5b2e7ba`（源码已锁只读，只读不妨碍运行）；run 名 `bench-collate-shm-old`。
- **B 侧（改后）**：开发副本 `/scratch/hongze/robomme_policy_learning_MotionJEPA-temp`，分支 `v2-motionmem-collate-shm` 的 clean HEAD（= C2）；run 名 `bench-collate-shm-new`。
- **顺序**：A → B 串行、8 卡独占（bench-b128-util 实测并行互扰 +11–13%）。
- **判读**：稳态窗口 step 100→290；步时均值为主判据；util 按 AGENTS.md 第 16 条给均值、0% 占比、慢步/非慢步分层均值，禁止以中位数作结论。A 侧应复现 1.82 s ± 噪声（与 f8-w16 互为校验）。
- **判定行**：每档 `PREFLIGHT=PASS n=25`、`EXIT_CODE=0`；分析输出 `BENCH_8GPU workers=16 step_mean_s=… util_mean=… zero_share=…`；汇总 `COLLATE_SHM_SPEED old=… new=… speedup=…`。
- **留档**：两档各约 12 min（3 min 编译 + 300 步），超过 5 min，按第 17 条在 `docs/training-doc/bench-collate-shm-8gpu/` 留档；末步 ckpt 与 `v1-store/bench/bench-collate-shm-*` 验收后删除。

### 6. 一致性检验（AGENTS.md 第 18 条两块）

**链路图（改前 → 改后）**，每跳标「有没有改数」：

```
改前  FrameSampDataset.__getitem__ ─numpy─▶ transform_dataset ─numpy─▶ _collate_fn(np.stack) ─numpy 513 MB─▶
      [worker: ForkingPickler.dumps 0.72 s] ═══ 队列管道 ═══▶ [主进程: loads 0.34 s] ─▶ make_array_from_process_local_data ─▶ 8 卡
改后  FrameSampDataset.__getitem__ ─numpy─▶ transform_dataset ─numpy─▶ _collate_fn(np.stack) ─numpy─▶ _to_shared_torch(零拷贝视图)
      ─torch─▶ [worker: storage 搬入 /dev/shm 0.28 s] ═══ 只传 fd ═══▶ [主进程: _from_shared_torch 零拷贝视图 <1 ms] ─numpy─▶ make_array… ─▶ 8 卡
```
所有跳里唯一「改数」的位置是 bf16 的 dtype 视图（uint16 ⇄ bfloat16），字节不变；其余跳都是搬运或视图。

- **第一块（非训练轻量对拍）**：把 F3 脚本正式化为 `scripts/training/tests/compare_collate_paths.py`（在分支上新增），对拍口径不变：同库同 seed、b128 w16、N=20 批、逐键 `sha256(dtype‖shape‖bytes)` 与抽样序列；再加一份 pytest `scripts/training/tests/test_collate_shm.py`：合成 dict（bf16/f32/f64/uint8/int64/bool/None 各一）经 2 个 spawn worker 的 `DataLoader` 走新 collate，与单进程 `_collate_fn` 逐键 `tobytes()` 相等。判据逐位。
- **第二块（8 卡真实训练梯度一致，收尾检验）**：`scripts/training/g0/bench_train_steps.py` 起 100 步，`--fsdp-devices 8 --batch-size 128 --num-workers 16 --seed 42 --log-interval 1 --save-interval 1`，`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`，`BENCH_CHECKSUM=1 BENCH_BATCH_DIGESTS=1 BENCH_DIGEST_INTERVAL=1000 BENCH_EXTRA_DIGEST_STEPS=1,2,24,49`。三侧同会话串行：A1、A2（主副本改前，A/A 零假设）、B（开发副本改后）。`scripts/training/g0/compare_baseline.py` 判 A1/A2 与 A1/B：`SCALARS steps=100 keys=5 hex_mismatch_steps=0`、`INDEX_SEQ=PASS`、`BATCH_DIGEST rows=6 mismatch=0`、`STATE_DIGEST rows=6 mismatch=0`。8 卡 (1,8) 从未做过确定性 A/A，若 A1/A2 自身不逐位（FSDP 归约次序），退回 `QUANT_EQUIV` 量化判据（以 A/A 为零假设、margin 2.0），并在留档写明。第二块不通过不得宣称等价。

### 7. 风险与边界

- **不改训练语义**：collate 的 `np.stack` 顺序、dtype、shape、抽样 generator 全部不动；不改超参、不改 config 默认值。
- **`--force` 红线**：开发副本的 `v1-store` 是指向主副本的可写 symlink（AGENTS.md 第 14 条例外），本计划所有命令都不带 `--force`、不传输出根参数；run 产物只写 `v1-store/train-runs/mme_vla_suite_b128_60k/bench-collate-shm-*` 与 `v1-store/bench/bench-collate-shm-*`。
- **两个在途未跟踪文件**（`scripts/dataset/hf_export/modul60k_bucket_README.md`、`run_modul60k_ckpt_export.sh`）不是本计划的，不 add、不动；`.claude/worktrees/b128cfg`、`sgab` 两个别的任务的 worktree 不动。
- **F2 里看到的 `terminate called without an active exception`**：torch persistent worker 在解释器退出时的收尾噪声，退出码 0，不影响结果；冒烟日志里若出现同样文本，判定以 `EXIT_CODE=` 为准。
- **共享内存策略**：默认 `file_descriptor`；若冒烟出现 `Too many open files`（fd 上限 1,048,576，理论不会），改 `torch.multiprocessing.set_sharing_strategy("file_system")`，作为记录在案的备选而非默认。
- **不在本计划内**：`static_pos_emb` 改传行号在 device 上索引（每样本再省 1.6 MB）、后台线程 device 预取——共享内存生效后主进程等待已降到 0.07 s，这两项收益变小，另立任务。

---

## 第二部分（技术细节，供 agent 追踪）

### A. 分支与副本（用户获准后执行，顺序固定）

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
git status --porcelain            # 仅允许出现上面两个在途 ?? 文件，其余非空即停
git rev-parse HEAD                # 须为 5b2e7ba…
git worktree add -b v2-motionmem-collate-shm /scratch/hongze/robomme_policy_learning_MotionJEPA-temp HEAD
cd /scratch/hongze/robomme_policy_learning_MotionJEPA-temp
ln -s /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store v1-store     # 可写链，第 14 条例外
UV_CACHE_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/uv uv sync --frozen   # 独立 .venv
# 主副本源码锁只读（照抄 v2-1600ep-m8x8-modul-b128-60k 形制）
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
sha256sum src/openpi/training/data_loader.py scripts/training/g0/bench_train_steps.py scripts/training/train.py > v1-store/bench/collate-shm-feasibility/lock_sha256.txt
chmod -R a-w src scripts packages
ls -ld src scripts packages       # 应为 dr-xr-xr-x
```

分支名 `v2-motionmem-collate-shm` 与开发副本路径以用户确认为准。远端无该分支，按第 11 条**不自行 `git push -u`**，push 前先问用户。

### B. 代码改动（在 `-temp` 上）

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
`scripts/training/g0/bench_train_steps.py::iter_with_digest`：同位置加同一行（`_openpi_dl._from_shared_torch(batch)`）。

注意 `jax.tree.map` 默认把 None 当空子树、不会把 None 交给 f；显式 `is_leaf` 只是防御。DataLoader 的 `_worker_init_fn`、`generator`、`persistent_workers`、`drop_last` 等参数不动。

### C. 第一块工具（在 `-temp` 上新增）

- `scripts/training/tests/compare_collate_paths.py`：以 `v1-store/bench/collate-shm-feasibility/f3_real_loader.py` 为底本正式化——旧侧用 `_collate_fn` 手建 `torch.utils.data.DataLoader`（与 `TorchDataLoader.__init__` 同参数、同 seed 构造 generator），新侧直接用 `create_data_loader(...)._data_loader.torch_loader`（已是新 collate）+ `_from_shared_torch`；N 批逐键摘要写 jsonl；输出 `COLLATE_EQUIV=PASS|FAIL batches=N keys=K none_keys=… mismatches=…` 与两侧主进程等待均值。运行：

  ```bash
  cd /scratch/hongze/robomme_policy_learning_MotionJEPA-temp && source scripts/training/paths.sh
  export MMEVLA_FRAMESAMP_SOURCE="$V1_STORE/datasets/4task-v2-1600ep-604f16da/source" MMEVLA_FRAMESAMP_MANIFEST="$V1_STORE/datasets/4task-v2-1600ep-604f16da/meta/episode_manifest.json"
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/tests/compare_collate_paths.py --batches 20 --workers 16 --out "$V1_STORE/bench/collate-shm-equiv"
  ```
- `scripts/training/tests/test_collate_shm.py`（pytest，< 1 min）：合成 dict 七种叶子经 2 个 spawn worker 的新 collate 与单进程 `_collate_fn` 逐键 `tobytes()` 相等；bf16 往返逐位；None 键保留。运行 `uv run --no-sync pytest scripts/training/tests/test_collate_shm.py -q`。

### D. 冒烟 runner（8 卡、300 步）

复制 `v1-store/logs/bw-8gpu-runner.sh` 为 `v1-store/logs/cs-8gpu-runner.sh`，参数 `<SIDE old|new> <TRAIN_HEAD> <HC_SHA256>`：`old` 侧 `MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA`，`new` 侧 `MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA-temp`；`RUN=bench-collate-shm-$SIDE`；`--num-workers 16 --fsdp-devices 8`；`MMEVLA_JAX_CACHE_DIR=$V1_STORE/cache/jax/cs-8gpu-$SIDE`（两侧各自缓存，避免互相命中）；其余（GPU_IDLE 检查、preflight 同 argv、500 ms 密采、`EXIT_CODE=`）照抄。驱动 `cs-8gpu-driver.sh`：`for SIDE in old new`，串行。

```bash
tmux new-session -d -s cs-8gpu "bash /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/cs-8gpu-driver.sh"
tmux has-session -t cs-8gpu
```

本轮 tmux 会话清单：仅 `cs-8gpu`（前缀 `cs-`）。Monitor 挂驱动日志、行缓冲过滤 `PREFLIGHT=|EXIT_CODE=|DRIVER_ALL_DONE|Traceback|out of memory|Too many open files`。分析复用 `docs/training-doc/bench-m8x8-8gpu-worker/records/bw_analyze.py`（起跑记录完整版在 commit `f2c9dc7`），窗口 step 100→290。

### E. 第二块 runner（8 卡、100 步、确定性）

复制 `docs/training-doc/t8-c8-guard-s100/records/runner.sh` 形制为 `v1-store/logs/cs-guard-runner.sh`：三侧 `a1`、`a2`（主副本）、`b`（开发副本）串行，各自 `BENCH_RECORD_DIR=$V1_STORE/bench/8x8/cs-guard-$SIDE`、`MMEVLA_JAX_CACHE_DIR` 独立；`bench_train_steps.py mme_vla_suite_b128_60k --exp-name cs-guard-$SIDE --batch-size 128 --num-workers 16 --num-train-steps 100 --log-interval 1 --save-interval 1 --seed 42 --fsdp-devices 8 --no-wandb-enabled ...`（数据集 / assets / history-config 参数与 D 相同）；每侧后 `project_scalars.py`，最后 `compare_baseline.py a1 a2` 与 `compare_baseline.py a1 b`。判定行见第一部分第 5 节；汇总 `COLLATE_SHM_GUARD=PASS scalars_steps=100 index_n=12800 batch_digest_rows=6 state_digest_rows=6`。tmux 会话 `cs-guard`。

### F. 留档、提交、清理

- `docs/training-doc/bench-collate-shm-8gpu/`：`launch.md`（起跑 commit 两侧、命令、会话清单）、`result.md`（速度表、util 分层、第一块与第二块判定行、链路图）、`records/`（`metrics.jsonl` 两侧、`judgement_lines.txt`、`analysis.json`、对拍 jsonl、`comparisons.log`）。
- commit 在 `v2-motionmem-collate-shm`：C1 `commitV9.8: collate 交付改走 torch 共享内存，主进程数据路等待 0.98 s→0.07 s`（四个文件）、C2 `docs: bench-collate-shm-8gpu 起跑留档`、C3 `docs: bench-collate-shm-8gpu 结果——…`；对应第一部分第 4 节第 4、5、8 步。逐文件 `git add`，不碰两个在途 `??` 文件。push 需先问用户（新分支无 upstream）。
- 并回 `v2-motionmem` 与否由用户决定；并回前主副本解锁 `chmod -R u+w src scripts packages` 并核对 `lock_sha256.txt` 三份文件未变。
- 清理：`v1-store/train-runs/mme_vla_suite_b128_60k/bench-collate-shm-*`、`cs-guard-*` 末步 ckpt；`v1-store/bench/bench-collate-shm-*` 拷入 records 后删；tmux 会话随驱动自然退出，如需手动只允许 `tmux kill-session -t cs-8gpu` / `-t cs-guard`，删前删后各 `tmux ls`。

### G. 验证清单（完成判据）

1. `F3_COLLATE_EQUIV=PASS`（已有）→ 正式化后 `COLLATE_EQUIV=PASS batches=20 keys=12 mismatches=0` + pytest 全绿。
2. 冒烟：两侧 `PREFLIGHT=PASS n=25`、`EXIT_CODE=0`；old 步时与 f8-w16 的 1.818 s 相差 ≤ 5%；new 步时、util 均值、0% 占比三项同时给出；`COLLATE_SHM_SPEED old=… new=… speedup=…`。
3. 第二块：`COLLATE_SHM_GUARD=PASS`（或量化判据 `QUANT_EQUIV=PASS` 并注明 A/A 不逐位的原因）。
4. 主副本解锁前 `lock_sha256.txt` 三份文件 sha256 不变、`git status --porcelain` 只含那两个在途文件。
