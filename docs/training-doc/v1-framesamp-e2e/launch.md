# v1-framesamp-e2e（S8b GL e2e 收官测试）launch 记录

对应 `v2-framesamp-restructure-plan.md` 阶段 4 S8b（D 节）。留档体例沿
`v1-framesamp-dl` 先例：一目录覆盖本批全部 run。

## 拍板与豁免（2026-08-27，用户 AskUserQuestion）

- 「GL e2e 收官测试 评估是否可以同时提交」→ 初版拍板**依赖链串行**：现在全部提交、
  `--dependency=afterany` 链 T1→T2→T3→COLDHOT 严格串行执行——立即排队但一次只跑
  一个（S8b 是负载敏感的性能验收，与 S5/S6 bitwise 并行是本质不同场景；避免同节点
  共驻/互相预热污染 `E2E_ACCEPT`）。与 S8a 的时间重叠风险接受（S8a 单 job ≤30 min，
  4 卡 job 排队大概率更久）。
- **改并行（2026-08-27 二次拍板，用户授权「可以自由提交这些 job」）**：T1 运行期间
  撤销 T2/T3/CH 依赖链（scancel 58996750/58996751/58996752），T2/T3 无依赖重提
  （58996987/58997004，均 `--exclude=gl1514`），Slurm 一有节点释放即起。评估结论：
  - **配额账**：chaijy2 上限 GPU 20/CPU 80/MEM 960G，组内他人占 GPU 6/CPU 18/
    MEM 256G；T1+T2+T3 并行 = 我方 12 GPU/48C/288G，放得下；**四个同时超配额**
    （22 GPU/82C）→ CH 必须等 T1 退出，改为 T2/T3 起跑后带全量冷节点排除清单
    另行提交（见下）。
  - **节点现状**：提交时 spgpu 无任何节点同时空出 4GPU+16C+96G（有 4 空卡节点
    CPU/内存均被占满），「锁定空节点立即并行」不可行，故不用 `-w` 钉节点、只用
    exclude 消极隔离；T1 结束释放 gl1512 是确定空位。
  - **同节点共驻风险**：当前无 8 空卡节点，T2/T3 落同一节点需大 job 恰好退出，
    概率低；由 squeue Monitor 盯节点分配，一旦同节点即 scancel 后起的一个并加
    exclude 重提（代价 ~10 min）。
  - **暖缓存口径**：T2/T3 允许落 gl1512（T1 刚跑过）——packed 库 30 GiB、每 run
    读 ~93 GB（全库重读 ~3 遍），稳态中位对起跑缓存状态不敏感；且 T1 自身起跑时
    gl1512 已被 dl-w4/w16 与 OOM run 焐过（pgmajfault=685 极低），条件对称。真冷
    代价由 COLDHOT 专项量化，其节点必须未碰过 packed 库（exclude
    gl1501/gl1508/gl1512/gl1514 + T2/T3 实际节点）。
  - **NFS 交叉负载**：并行 2–3 个 run 合计 ~50–75 MB/s，对 turbo 聚合带宽可忽略；
    server_read 遥测按节点采样，不受他 job 混淆。
- **96G 内存评估（用户问「96G 节点 OOM 评估是否需要增大」）：不需要。** OOM 根因
  （步 0 TrainState 摘要 device_get 45.4 GiB）已被 beb464f
  （`BENCH_CHECKSUM=0 BENCH_BATCH_DIGESTS=0`）移除；T1 实测 sstat MaxRSS=46.3 GiB
  （AveRSS 41.9 GiB），96G 余量近半；维持 96G 亦保持与 legacy 基线 v1-e2efix 同
  资源包络可比。
- 资源审批「四个全批」：放行记录见仓库根 `greatlakes.md`「放行记录（robomme
  framesamp v2 计划，2026-08-27）」。
- **S9 G2-speed 用户决策暂时不跑**（计划 T8/E 表同步标注）。
- 条件档 T4 w16 / T5 w4c8 未批未提交，视 T1–T3 结果另行审批。

## run 表（全部 4×A40/16C/96G，packed 库，`ANALYZE_ACCEPT=1`）

| run_name | workers | 步数 | walltime | seed | 调度（二次拍板后） |
|---|---|---|---|---|---|
| `v1-framesamp-e2e-w4c16`（T1，最重要：官方默认档还需不需要调） | 4 | 600 | 2h | 320 | job 58996749 @gl1512 |
| `v1-framesamp-e2e-w8c16`（T2，直接对 v1-e2efix-w8c16） | 8 | 600 | 2h | 321 | 无依赖重提 58996987（原 58996750 已撤） |
| `v1-framesamp-e2e-w2c16`（T3，探底） | 2 | 600 | 2h | 322 | 无依赖重提 58997004（原 58996751 已撤） |
| `v1-framesamp-e2e-w4c16-coldlike` + `…-hot`（COLDHOT=1 同 allocation 先 C1 后 H1） | 4 | 300+300 | 4h | 323 | job 59001191（排除 gl1514/1501/1508/1512/1519） |
| `v1-framesamp-e2e-w12c16`（T4′，补齐 legacy 四档对照） | 12 | 600 | 2h | 324 | job 59001192 |
| `v1-framesamp-e2e-w16c16`（T5′，同上 + 验证超订拐点） | 16 | 600 | 2h | 325 | job 59001193 |

### 三次拍板（2026-08-27 深夜，用户「不改动训练逻辑 把现有的实验跑完…先全部提交 job 排队」）

不改任何训练/dataloader 代码，仅把剩余档位跑完拿完整结论。新增两档理由：

- **packed 侧只有 w4/w8，legacy 侧有 w4/w8/w12/w16 完整四档**——补齐 w12/w16 才能给出
  同口径的「worker 量级 → util」全曲线对比。
- **w12 是不改代码前提下唯一可能达标的候选**：packed w4→w8 的 util 是 85.2%→89.2%
  （+4.0pp，与 legacy「加 worker 反而降」的走向相反，因瓶颈已从 NFS 带宽换成 worker
  供给）；若 w12 延续增益即可越过 90% 阈值。
- **w16 验证超订拐点**：慢步归因分析（见 result.md）推断瓶颈在 worker 侧 CPU 超订
  （`_worker_init_fn` 未收敛线程数、worker 继承 `OMP_NUM_THREADS=16`），预测 w16 会
  退化；实测可证实/证伪，且 legacy w16 有对照。
- **COLDHOT 意义不减反增**：T1/T2 实测 NFS 仅 8–10 MB/s 对公式口径 29–32 MB/s，
  证实约 2/3 的读命中 page cache——即 T1–T3 数字本身偏热态、偏乐观；而 T2 的
  epoch 8.35 h 距 8.6 h 阈值仅剩 0.25 h 余量，冷态惩罚哪怕 3% 就击穿。COLDHOT 正是
  量化这个惩罚的唯一实验，直接决定 T2 的 epoch ✓ 作不作数。
- 资源包络与已批四 job 完全一致（4×A40/16C/96G，600 步 2h、COLDHOT 4h），未扩。
- 排队策略（用户指定）：集群紧张，全部先提交排队；若出现同节点共驻干扰，scancel
  后起者加 exclude 重提。

- 入口 `gl_e2e_fix.sbatch`（S7.5 参数化）：`MMEVLA_DATA_BACKEND=packed`（显式）、
  `DATASET_PATH=…/4task-gl-framesamp`（status=verified）、`SAVE_INTERVAL=1000`
  （600 步内零摘要，性能口径不受污染）、`MMEVLA_FRAMESAMP_VERIFY` 缺省 fast——
  **冷态自证：allocation 内不跑 full 校验、无本地复制预热**（env.json 硬字段）。
- 遥测：dense 500ms + legacy 15s 双通道 GPU util、NFS server_read、meminfo
  （Cached+pgmajfault）、compute_apps（worker CUDA context 存证）。
- seed 320–323（避开已用 42/200–205/210–212/310–313）。

## 判据（D 节）

- T1–T3：`analyze_gpu_util.py --accept` 机器判定单行
  `E2E_ACCEPT=PASS|FAIL`——必达五项：步时中位 ≤5.00 s、util 稳态均值 ≥90%、
  0% 采样 ≤5%、慢步(>8s)墙钟 ≤5%、epoch(6,176 步) ≤8.6 h；报告须附「距 100%
  残差分解」。附加判据（`--extra`）：w4 与 w8 步时差 ≤3%；NFS server_read 稳态
  期望 ≈17–25 MB/s，>65 MB/s 视为读放大信号。
- COLDHOT：`(C1稳态−H1稳态)/H1 ≤ 15%`（cold-like 口径，meminfo/pgmajfault 证据链）。
- 对照组（历史口径实测，不重测）：v1-e2e-b64 6.933 s/69.7%、w8c16 5.301 s/71.2%、
  w12c16 5.319 s、w16c16 5.327 s、compute-only 4.778 s。

## 起跑环境

- **起跑 HEAD**：本 launch.md 的 docs commit 本身（结果留档回填全 sha + 各 job id）
- ControlMaster 复用零认证；提交器 `gl_submit.py`；提交命令模板：

```bash
uv run --no-project --with pexpect python scripts/data-preprocess-GL/gl_submit.py \
  "sbatch --parsable --job-name=<run_name> --time=<walltime> [--dependency=afterany:<prev>] \
   --export=ALL,WORKERS=<W>,BENCH_SEED=<S>,TAG=<run_name 基名>,ANALYZE_ACCEPT=1[,COLDHOT=1],DATASET_PATH=…/4task-gl-framesamp,MMEVLA_DATA_BACKEND=packed \
   scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch"
```

## 四次拍板（2026-08-28 午间）：COLDHOT 的 H1 缺失——脚本修复 + 同节点补跑

用户原话：先问「目前排队的 job 跑完要多久」，答复中报出本问题后追加两条指令——
「COLDHOT 修复重新跑」、「w16 取消不跑了」。

### 问题：H1 被 C1 的 accept 判定挡掉，4h allocation 只会产出一半数据

`gl_e2e_fix.sbatch` 的 `run_bench` 在训练成功后，用 `analyze_gpu_util.py` 的退出码
**覆盖**了自己的返回码；`--accept`（`ANALYZE_ACCEPT=1`）判 FAIL 时 analyzer 非零退出
（见 `analyze_gpu_util.py` 的 `--accept` 分支「FAIL 非零退出」）。而 COLDHOT 分支写的是
`if [ "$RC" -eq 0 ]` 才跑 H1，于是：

- C1 是 **cold-like 冷态**，`E2E_ACCEPT` 五项必达档几乎必然 FAIL——热态的 T2 w8c16
  （job 58996987）都已经 `E2E_ACCEPT=FAIL util_mean_pct=89.178✗ zero_pct=9.739✗`；
- → C1 返回 1 → H1 被 `if` 挡掉 → job 在 C1 结束后直接 `EXIT_CODE=1` 退出；
- → COLDHOT 判据 `(C1稳态−H1稳态)/H1 ≤ 15%` 缺对照组，**整个 4h run 白跑**。

判据本身与 `E2E_ACCEPT` 五项无关，C1 判 FAIL 属预期内，不该作为流程闸门——这是纯粹
的脚本缺陷，不是实验设计问题。

### 修复：训练退出码与 accept 判定解耦（不改任何训练/dataloader 语义）

`gl_e2e_fix.sbatch` 三处改动，均在 `run_bench` 与其后的 COLDHOT 分支内：

1. `run_bench` 新增副作用全局量 `BENCH_TRAIN_RC`，在 `local rc=$?` 捕获训练退出码后
   立即存住（analyzer 覆盖 `rc` 之前）；函数入口初始化为 1，使「记录目录已存在」等
   提前 `return` 的分支也有确定值、不残留上一轮。
2. COLDHOT 分支改判 `C1_TRAIN_RC="$BENCH_TRAIN_RC"`——**只看训练是否跑完，不看 accept
   判定**；C1 训练本身失败（OOM 等）仍跳过 H1 并打印原因。
3. `run_bench` 的返回值语义不变（仍是含 accept 的综合码），因此非 COLDHOT 路径
   （T1/T2/T3/T4′/T5′ 各档）的 `EXIT_CODE` 口径与既有 run 完全一致，历史数据可比。

验证（不启动训练）：`bash -n` 语法检查通过；把脚本内 COLDHOT 分支原文抽出、套 mock
`run_bench` 跑三场景——「C1 训练成功+accept FAIL（本次实况）」与「C1 训练成功+accept
PASS」均调用到 `t-hot`，「C1 训练本身失败」不调用 `t-hot`，三项符合预期。

### 补跑策略：不重跑 C1，H1 用 `-w gl1515` 钉同节点单独跑

发现问题时 C1 已跑到 233/300（4.8–5.1 s/it，冷态数据完整有效），5 分钟后即结束。
权衡后**不 scancel、不整体重跑**：

- **同 allocation 的实质是同节点 + C1 焐热的 page cache**，而 page cache 是 OS 级的、
  不随 job 退出清空。C1 job 退出到 H1 起跑之间只隔 analyzer 的约 1 分钟，`-w gl1515`
  钉回同一节点即可复现 H1 所需的热态现场。
- 整体重跑的代价：作废 25 分钟已完成的冷态数据，且需另找一个**从未读过 packed 库**的
  冷节点（gl1512/1515/1517 及原 exclude 清单已全部污染），排队到 w12 释放（约 14:33）
  才起，收尾推迟约 1 小时。
- 为保住这个不可逆的时间窗，在 C1 结束前先行提交 H1 排队（job 59044123，
  `PENDING (AssocGrpGRES)` 等 C1 释放配额）；w16 已由用户于 13:38:49 自行 scancel
  （`CANCELLED by 114466650`）并明示「取消不跑了」，无其他 job 竞争 gl1515。
- **H1 走非 COLDHOT 路径**（`COLDHOT` 缺省 0、`STEPS=300`、`TAG=v1-framesamp-e2e-w4c16-hot`），
  故 record_dir 落在 `v1-store/bench/bottleneck/v1-framesamp-e2e-w4c16-hot`——与原
  COLDHOT 路径下 `${TAG}-hot` 的命名完全一致，与 C1 的 `-coldlike` 天然配对，
  对拍脚本无需特判。seed 仍 323，与 C1 同 seed 冻结 index 序列（D 节要求）。

  ```bash
  sbatch --parsable --job-name=v1-framesamp-e2e-ch-hot --time=01:00:00 -w gl1515 \
    --export=ALL,WORKERS=4,BENCH_SEED=323,TAG=v1-framesamp-e2e-w4c16-hot,STEPS=300,\
ANALYZE_ACCEPT=1,DATASET_PATH=…/4task-gl-framesamp,MMEVLA_DATA_BACKEND=packed \
    scripts/bottleneck-bench-v2/gl_e2e_fix.sbatch
  ```

### 结果判读前必须先验的前提（热态自证）

`-w gl1515` 补跑相对同 allocation 的唯一风险是 job 间隙 page cache 漂移（含集群
epilog 可能 drop_caches）。该风险**可事后证伪**，判读 `(C1−H1)/H1` 之前必须先验：

- H1 的 meminfo 采样起跑即高 `Cached`、`pgmajfault` 显著低于 C1 → 热态成立，结论有效；
- 若 H1 的 pgmajfault 与 C1 同量级 → 缓存已被清，H1 实为第二次冷跑，
  `(C1−H1)/H1` 偏小、判据偏乐观，**该次结果作废**，须按修复后的脚本另找冷节点整体重跑。

### 本轮 job 台账

| job | run | 结果 |
|---|---|---|
| 59001191 | `v1-framesamp-e2e-w4c16-coldlike`（C1，gl1515） | 300 步跑完，H1 被缺陷挡掉；C1 数据保留 |
| 59044123 | `v1-framesamp-e2e-w4c16-hot`（H1，`-w gl1515`） | 本次补跑 |
| 59001192 | `v1-framesamp-e2e-w12c16` | 正常运行中，不受影响 |
| 59001193 | `v1-framesamp-e2e-w16c16` | 用户 13:38:49 scancel，**取消不跑** |
