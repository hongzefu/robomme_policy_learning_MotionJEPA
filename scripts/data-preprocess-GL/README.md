# scripts/data-preprocess-GL —— 四任务数据处理的集群链路

四任务（`ButtonUnmask` / `VideoUnmask` / `ButtonUnmaskSwap` / `VideoUnmaskSwap`）各 400 episodes
的预处理，在 GreatLakes 上以 8 个 1-GPU job array 并行完成。
**集群相关的一切实现都在本目录内**，仓库其余部分只被调用、不被修改。

- 方案与全部实测数字：[`docs/v1-gl-dataset-consistency-report.md`](../../docs/v1-gl-dataset-consistency-report.md)
- CPU/mem 档位实测结论：[`docs/v1-gl-resource-tier-bench.md`](../../docs/v1-gl-resource-tier-bench.md)
- 集群提交硬规则：仓库根 [`greatlakes.md`](../../greatlakes.md)

## 三个入口：step0 → step1 → step2

真正需要手跑的只有这三个脚本，按编号顺序执行。时长均为 2026-08-23 实测值。

| 入口 | 功能 | 实测时长 |
|---|---|---|
| `step0_setup_turbo.sh` | 一次性置备：NFS venv 重建 / H5 暂存到 turbo + 两侧 sha256 同源核验 / **episode 清单生成（`manifest` 子命令，全流程唯一真值源）** / 模型内联 / 自检。幂等，可反复跑 | 首跑约 2.5–3 h（rsync 321 GB 约 54 min + 两侧 sha256 并行约 48 min + 清单 NFS 扫描约 48 min + venv 约 11 min + 模型约 4 min）；复跑全命中复用为分钟级 |
| `step1_submit.sh` | 九项 pre-flight（清单/输入同源/输出洁净/claim/ControlMaster/模型/walltime 裕度/配额/审批闸门）→ 提交 8×1GPU array + afterok finalize（五道守卫） | 提交秒级；array 实测 33:40–36:45（八片极差 9%）、finalize 约 4 min，端到端约 40 min |
| `step2_verify.sh` | 本地对照：本机建 47 个分层随机 episode 参照库（`build_shard.py`）→ 第二层跨架构逐 key 分类对拍 + 第三层下游等价（`compare_datasets.py`） | 参照库构建约 12 min（约 14,200 步 @ ~20 step/s NFS）+ 两层对拍约 15 min，合计约 30 min |

```bash
cd /nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
S=scripts/data-preprocess-GL

bash $S/step0_setup_turbo.sh all          # venv / H5 暂存+sha256 / 清单 / 模型 / 自检

# 【审批点】超出 greatlakes.md 调试限额，须用户明示放行。
# 档位与速率为 2026-08-23 实测定案值；数据形制变化后先用 legacy/step_bench.sh 重测。
CONFIRM_FULL=yes RATE=28.913 TIER_CPUS=2 TIER_MEM_GB=24 WALLTIME=04:00:00 \
  bash $S/step1_submit.sh

bash $S/step2_verify.sh                   # 【第二、三层】→ VERIFY_PASS
bash scripts/smoke-local/run_gl_dataset_training_smoke.sh   # 【第四层】首次接入训练前建议跑
```

## GreatLakes 构建一致性

集群产物与本地产物的差异有两个独立来源，必须分开验证，混在一起测出了差异说不清是 bug
还是硬件噪声：**① 我们把串行 builder 改成了 8 分片（自己的代码改造）；
② A40（sm_86）与本机 RTX 6000 Ada（sm_89）是不同 GPU 架构（硬件事实）。**

### a. 跨架构：A40 与 RTX 6000 Ada 导致的 token 差距——差多少、为什么可接受

产物里**只有 SigLIP 的 `image_emb_*` 真正过了 GPU 归约**，也只有它跨架构不一致。差距实测：

| 指标 | 实测值 |
|---|---|
| bf16 位完全相同的元素占比 | 仅 **15.8%** |
| 平均绝对误差 | ≈ **2–3 个 bf16 ULP**（bf16 尾数 7 位，1 ULP ≈ 0.4% 相对误差） |
| 误差地板（平均绝对误差 ÷ 非零中位幅值） | **0.019**（阈值 0.05，裕度 2.6×） |
| 最小逐 token 余弦 | **0.999842**（阈值 0.999，裕度 6.3×） |
| p5 逐 token 余弦 | **0.999975**（阈值 0.9999，裕度 4.1×） |

为什么可接受（四条论证，每条都有实测支撑）：

1. **成因锁定为归约累加顺序，不是 bug**：SigLIP So400m/14 共 27 层全程 bf16 计算，每层矩阵乘是
   1152/4304 项点积；两种架构的 tensor core 分块形状与 split-K 划分不同 → 同一点积的累加顺序不同
   → 逐层 ULP 级舍入差异累积。决定性对照组是 `pos_emb_*`：同走 GPU/JAX 但是秩一外积**无归约**
   → 跨架构**逐位相同**，把「GPU/驱动/JAX 版本不同」全部排除。determinism 三档对此无效
   （不是 cuDNN 非确定性，是硬件分块策略），任何人跨这两种卡都会得到同量级差异。
2. **除 `image_emb` 外一切逐位相同**：`kept_indices.json`（numpy 像素差）、`data/*.pkl`（H5 直读）、
   `state_emb`、`pos_emb_*` 跨架构全部零容差通过。
3. **数值噪声没有改变任何离散决策**：下游 `prepare_frame_sampling` 的选帧索引与 padding mask
   逐位相同——训练读到的「选哪些帧、哪些 token 有效」与本地完全一致。
4. **判据本身经过判别力验证**：先在已知答案的合成用例上标定——人为造 3 ULP 舍入型差异
   vs 把一个值 2.0→2.5 的结构型差异，误差地板拉开 **21 倍**，判据能拦真错误，不是「调到刚好能过」。

因此交付按「**换合同**」口径：跨架构逐位一致本来就做不到，集群产物自成一份数据集，
`meta/provenance.json` 逐条带硬件/软件指纹并由 finalize 断言全体同源，
**机制上杜绝与本地字节混用**；验收标准是上述等价判据，而不是「和本地一模一样」。
成因的分层指纹归因全文见一致性报告 4.3.2 节。

### b. 分片：为什么要求且能做到 bit-by-bit 一致

**为什么要求零容差**：分片是我们自己的代码改造，第一层对拍在**同机同架构**下进行，
硬件变量为零——此时出现任何差异都必然是 bug（错号、偏移算错、覆盖），
给容差就等于给自己的 bug 留藏身处。所以判据是**逐字节相同，无任何阈值**。

**为什么能做到**（三个机制保证，对拍只是兜底证据）：

1. **计算本体一行不动**：`build_shard.py` 子类化 `DatasetProcessor`，只覆盖 `__init__`
   （跳过会互删产物的 `shutil.rmtree`）与 `run()`（按清单遍历 + 喂偏移量），
   `_process_episode` 原样继承——语义同构由构造方式保证。
2. **清单把并行错号的根源掐死**：原版三个计数器（`global_episode_idx` / `exec_sample_id` /
   `total_sample_id`）从 0 跨文件累加、遍历还用非确定序的 `os.listdir`，直接并行必然错号覆盖。
   `scan_manifest.py` 按规范序 `sorted(*.h5) × sorted(episode_i)` 把每个 episode 的三个 ID
   起点用前缀和算死，8 片写出的文件名与「串行跑一遍」逐个同构。
3. **同机跨进程 XLA 实测确定**：单进程跑 4 个 episode vs 4 个独立进程各跑 1 个，
   `image_emb_*` 逐位相同——排除了「XLA autotuning 在不同进程选到不同算法」这类同机非确定性。

**实测结论**：12 episode / 3,862 步全覆盖，9 个 key（`image_emb_*`×3、`pos_emb_*`×3、
`state_emb`、`kept_indices.json`、`data/*.pkl`）全部逐字节相同，`COMPARE_RESULT=bitexact PASS`。
验证脚本已归档为 `legacy/step_local_baseline.sh`；**若改动 `build_shard.py` 或
`scan_manifest.py`，须重跑它重新取得「本地真值」资格**。

集群侧另有第五道自证：finalize 在同一节点随机复算 256 条 `token_emb`，断言 `max|diff|=0`
（同架构可以零容差），排除线程调度、cuDNN 算法选择这类非确定性。

## 资源档位（resource）

定案 **2 CPU / 24G / walltime 04:00:00**（`step1_submit.sh` 已作默认值）。
选档判据、本机九档扫描、集群四档 A40 探针、全量 8 分片复核（最重分片 anon 峰 14.78 GiB，
证明 16G 档几乎必然 OOM）的完整实测过程见
[`docs/v1-gl-resource-tier-bench.md`](../../docs/v1-gl-resource-tier-bench.md)。
复测入口：`legacy/step_bench.sh`。

## 为什么需要预扫描（这条决定了整个设计）

`DatasetProcessor.run()` 是**严格串行**的：`global_episode_idx`（决定 `features/episode_{g}/`
目录名）、`exec_sample_id`（决定 `data/{id}.pkl` 文件名）、`total_sample_id` 三个计数器从 0
一路**跨文件累加**，而文件遍历用的还是非确定序的 `os.listdir`。直接并行分片必然错号覆盖。

所以先用 `scan_manifest.py` 只读 metadata 扫一遍，按**规范序** `sorted(*.h5) × sorted(episode_i)`
把每个 episode 的三个 ID 起点算死，写进 `episode_manifest.json`。它是全流程的**唯一真值源**：
分片 worker、finalize 守卫、一致性比对工具都从它取 episode 身份 `(h5_file, raw_ep_idx)` 与偏移量，
不再依赖任何目录名或遍历顺序。

## 文件

### 活跃链路（主目录）

入口：`step0_setup_turbo.sh` / `step1_submit.sh` / `step2_verify.sh`（见上表）。

被入口调用的组件（不直接手跑）：

| 文件 | 作用 |
|---|---|
| `paths.sh` | 唯一路径与环境源。turbo 前缀 fail-loud 校验；**刻意不覆盖 `HOME`** |
| `scan_manifest.py` | `build` 预扫描生成清单 + LPT 装箱；`sample` 抽 episode 子集 |
| `build_shard.py` | `DatasetProcessor` 子类，按清单跑单分片，支持 `--resume`；也用于本机建对照参照库 |
| `compare_datasets.py` | 分层一致性比对（`bitexact` / `crossarch` / `downstream`） |
| `finalize_checks.py` | `hash-inputs` 算 H5 sha256；`check` 完整性/stats/provenance/零容差抽检 |
| `stage_models.sh` | SigLIP / tokenizer / pi05_base → `v1-store/models/`，带逐项校验 |
| `check_quota.py` | 解析组配额输出，pre-flight 判断本次提交是否超出 chaijy2 剩余额度 |
| `gl_submit.py` | 提交器（ControlMaster 复用，免 2FA） |
| `gl_build_dataset.sbatch` | `--array=0-7` 分片构建 job |
| `gl_finalize.sbatch` | `afterok` 收尾守卫 job |

### legacy/（一次性工作，结论已定案，仅复测时使用）

| 文件 | 作用 | 定案结论 |
|---|---|---|
| `step_local_baseline.sh` | 第一层：分片语义无损（未改动 builder vs 分片实现，逐字节零容差） | 已 PASS；改动 `build_shard.py`/`scan_manifest.py` 后须重跑 |
| `step_bench.sh` | 档位实测入口（`local` / `cluster` / `report`） | 定案 **2 CPU / 24G**；数据形制变化后须重测 |
| `bench_resources.py` | 本机档位扫描器（`systemd-run` 限内存 + `taskset` 限核） | 只被 step_bench 调用 |
| `sample_summary.py` | 采样汇总器，本机与集群共用，保证指标定义逐字相同 | 只被 bench_resources 与 gl_probe 调用 |
| `gl_probe.sbatch` | 多档位探针 job，1 GPU / ≤30 min（限额内，无需放行） | 定案 RATE=28.913 step/s（A40 稳态） |

## 取代关系

本目录取代了已弃用的 `scripts/v1_dataloader_restructure/`（commit `d951aef`，经判定不可靠，
已 `git rm`）。那批脚本定义的路径约定（`.openpi-data/`、`data/robomme_preprocessed_4task_*`、
`artifacts/v1_dataloader_restructure/`）一并作废，勿从 git 历史里翻出重新采用。

## walltime 怎么算（GPU 与 I/O 两条估算取大者）

单分片探针**测不出 8 路并发下的 turbo 带宽争用**，只按探针速率反算会严重低估。
`step1_submit.sh` 的第 ⑦ 项因此同时算两条：

- **GPU 侧**：`总步数 ÷ 分片数 ÷ 稳态 step/s`
- **I/O 侧**：`(读 321 GB 原始 H5 + 写 总步数×每步字节) ÷ 卷带宽`。
  8 个分片**并发共享同一个 turbo 卷**，所以 I/O 侧的耗时对每个分片都是「全量字节 ÷ 卷带宽」，
  **不再除以分片数**。

取两者的大者 ×1.5 作为申请值，并硬断言裕度 ≥1.2×。这条流水线是 I/O 受限而非 GPU 受限
（GPU 利用率实测仅约 21%）。全量实测：按 `IO_BW_MBPS=132` 估 1h37m，实际 36m45s——
8 路并发聚合带宽实测约 320 MB/s，估算刻意保守。

## 仓库搬迁（一次性）

仓库单副本在 turbo，本机不留。首次搬迁：

```bash
rsync -a --exclude='.venv/' --exclude='.ruff_cache/' --exclude='__pycache__/' \
      --exclude='v1-store/' /data/hongzefu/robomme_policy_learning_MotionJEPA/ \
      /nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/
```

`.venv` 必须排除后在 turbo 上重建（`step0_setup_turbo.sh venv`）——uv 默认把解释器装在
本机 home，`.venv/bin/python` symlink 过去，在计算节点上是**死链**。

## 续跑与故障处理

**某分片超时/被杀**（三步，缺一不可）：

```bash
rm <OUT>/_claims/_claim_shard<i>of8.json
gl_submit.py "sbatch --parsable --array=<i> --cpus-per-task=<C> --mem=<M>G --time=<T> \
  --job-name=v1-4task-build --export=ALL,REQUIRE_EMPTY=0,<三路径> \
  scripts/data-preprocess-GL/gl_build_dataset.sbatch"
gl_submit.py "sbatch --dependency=afterok:<原AID>:<新JOBID> ... gl_finalize.sbatch"
```

第三步不可省：**任一分片失败，原 finalize 会因 afterok 不满足被 SLURM 以
`kill_invalid_depend` 自动 CANCELLED，此时连日志文件都不会生成**——判死只能靠
`sacct -j <AID>,<FID> --format=JobID,State,ExitCode -X`。

`build_shard.py --resume` 会跳过已完整落盘的 episode（判据是 `kept_indices.json` 存在
且 `token_emb_*.npy` 数量对得上），中断损失上界 = 单 episode ≤ 586 步。
残缺 episode 会被**先清后重做**——不清的话 `_process_episode` 里的
`assert not os.path.exists(pkl_path)` 会撞上半截 pkl 直接炸。

## 收尾（验收通过后）

按 AGENTS.md 第 15 条，turbo 上的 H5 暂存副本是临时的，验收通过后删除；
本机 `/data/hongzefu/robomme_data_h5_v2_4env400ep` 的原件永久保留。
⚠ 4task-gl 这一轮的 turbo H5 副本经用户 2026-08-24 明示决定**保留不删**
（见一致性报告第 7.3 节），勿按第 15 条误删。

## 已知坑

- **绝不要在 Bash 脚本执行期间编辑它本身。** bash 是**增量读取**脚本文件的：长命令
  （如 14 GB 的 rsync）执行完返回后，它会从**保存的字节偏移**继续往下读。此时若文件
  已被编辑、偏移对应的位置变了，就会读到半截语句并报
  `syntax error near unexpected token '('`。本轮实测踩过一次：`stage_models.sh` 的全部
  校验其实都已通过、日志写到「项目内模型准备完成」，却仍以 `EXIT_CODE=2` 收尾。
  判断办法是看日志里业务步骤是否走完；补救办法是原样重跑（本目录脚本都设计成幂等的）。
- **仓库放在 turbo 上必须 `git config core.filemode false`。** turbo 的默认 ACL 会给每个文件
  强制加上属主执行位（`664` → `774`），git 于是把**每一个文件**都报成
  `mode change 100644 => 100755`，`git status` 里 400 多个文件全变 modified、内容却零差异。
  `step0_setup_turbo.sh check` 会自动把它设上。代价是 git 不再感知可执行位，
  所以新增需要 +x 的脚本要显式 `git update-index --chmod=+x <file>`。
- **`--qos=interactive` 在 chaijy2/spgpu 报 `Invalid qos specification`**，不要用。
- **spgpu 强制每个 job 至少 1 GPU**，纯 CPU 的 job 也要带 `--gpus-per-node=1`。
- **`systemd-run --user -p AllowedCPUs` 不生效**（user slice 没下放 cpuset 控制器，
  进程 affinity 仍是全核），本机限核必须用 `taskset`；`-p MemoryMax` 则确实生效。
- **`scripts/train.py` 的 `__main__` 会连跑两次 `main()`**（tentative 之后紧接 80k step
  正式训练），smoke 必须走 `scripts/smoke-local/smoke_train_once.py`。
- **抽取期间 8 个 task 同读写一个 turbo 卷**（实测天花板 ~132 MB/s 为保守值，
  8 路并发聚合实测约 320 MB/s），期间勿并行起集群训练。
