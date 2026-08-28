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
