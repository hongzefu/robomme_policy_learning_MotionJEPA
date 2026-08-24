# scripts/data-preprocess-GL —— 四任务数据处理的集群链路

四任务（`ButtonUnmask` / `VideoUnmask` / `ButtonUnmaskSwap` / `VideoUnmaskSwap`）各 400 episodes
的预处理，从「本机单卡串行」改为「GreatLakes 上 8 个 1-GPU job array 并行」。
**集群相关的一切实现都在本目录内**，仓库其余部分只被调用、不被修改。

- 方案与全部实测数字：[`docs/v1-gl-dataset-consistency-report.md`](../../docs/v1-gl-dataset-consistency-report.md)
- CPU/mem 档位实测结论：[`docs/v1-gl-resource-tier-bench.md`](../../docs/v1-gl-resource-tier-bench.md)
- 集群提交硬规则：仓库根 [`greatlakes.md`](../../greatlakes.md)

## 取代关系

本目录取代了已弃用的 `scripts/v1_dataloader_restructure/`（commit `d951aef`，经判定不可靠，
已 `git rm`）。那批脚本定义的路径约定（`.openpi-data/`、`data/robomme_preprocessed_4task_*`、
`artifacts/v1_dataloader_restructure/`）一并作废，勿从 git 历史里翻出重新采用。

## 为什么需要预扫描（这条决定了整个设计）

`DatasetProcessor.run()` 是**严格串行**的：`global_episode_idx`（决定 `features/episode_{g}/`
目录名）、`exec_sample_id`（决定 `data/{id}.pkl` 文件名）、`total_sample_id` 三个计数器从 0
一路**跨文件累加**，而文件遍历用的还是非确定序的 `os.listdir`。直接并行分片必然错号覆盖。

所以先用 `scan_manifest.py` 只读 metadata 扫一遍，按**规范序** `sorted(*.h5) × sorted(episode_i)`
把每个 episode 的三个 ID 起点算死，写进 `episode_manifest.json`。它是全流程的**唯一真值源**：
分片 worker、finalize 守卫、一致性比对工具都从它取 episode 身份 `(h5_file, raw_ep_idx)` 与偏移量，
不再依赖任何目录名或遍历顺序。

`build_shard.py` 则**子类化** `DatasetProcessor` 而非复制其逻辑，只覆盖 `__init__`
（跳过会互删产物的 `shutil.rmtree`）与 `run()`（按清单遍历 + 喂偏移量），
`_process_episode` 本体一行不动——**语义同构由构造方式保证**。

## 文件

| 文件 | 作用 |
|---|---|
| `paths.sh` | 唯一路径与环境源。turbo 前缀 fail-loud 校验；**刻意不覆盖 `HOME`** |
| `stage_models.sh` | SigLIP / tokenizer / pi05_base → `v1-store/models/`，带逐项校验 |
| `scan_manifest.py` | `build` 预扫描 + LPT 装箱；`sample` 抽 episode 子集 |
| `build_shard.py` | `DatasetProcessor` 子类，按清单跑单分片，支持 `--resume` |
| `bench_resources.py` | 本机档位扫描器（`systemd-run` 限内存 + `taskset` 限核） |
| `sample_summary.py` | 采样汇总器，**本机与集群共用**，保证指标定义逐字相同 |
| `finalize_checks.py` | `hash-inputs` 算 H5 sha256；`check` 完整性/stats/provenance/零容差抽检 |
| `compare_datasets.py` | 分层一致性比对（`bitexact` / `crossarch` / `downstream`） |
| `gl_probe.sbatch` | 多档位探针，1 GPU / ≤30 min（限额内，无需放行） |
| `gl_build_dataset.sbatch` | `--array=0-7`，档位由实测填入 |
| `gl_finalize.sbatch` | `afterok` 收尾守卫 |
| `gl_submit.py` | 提交器（ControlMaster 复用，免 2FA） |
| `step0_setup_turbo.sh` | venv / H5 暂存与 sha256 / 模型 / 自检 |
| `step_local_baseline.sh` | 第一层：分片语义无损（逐字节） |
| `step_bench.sh` | 档位实测（`local` / `cluster` / `report`） |
| `step_submit.sh` | 九项 pre-flight + 提交 array + afterok finalize |
| `step_verify.sh` | 第二、三层：跨架构分类对拍 + 下游等价 |

## 逐段流程

```bash
cd /nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
S=scripts/data-preprocess-GL

bash $S/step0_setup_turbo.sh all          # venv / H5 暂存+sha256 / 模型 / 自检
bash $S/step_local_baseline.sh            # 【第一层】逐字节，不过就地停
bash $S/step_bench.sh local               # 本机扫档 → 候选档位
bash $S/step_bench.sh cluster             # 候选各提一个 ≤30min 探针
bash $S/step_bench.sh report              # 汇总 → 定档 + 拿 A40 实测 step/s

# 【审批点】超出 greatlakes.md 调试限额，须用户明示放行
# RATE 取探针的 rate_steady；BYTES_PER_STEP 取基线脚本打印的 CALIBRATION_BYTES_PER_STEP
CONFIRM_FULL=yes RATE=<A40稳态step/s> BYTES_PER_STEP=<实测> \
  TIER_CPUS=<定档> TIER_MEM_GB=<定档> bash $S/step_submit.sh

bash $S/step_verify.sh                    # 【第二、三层】→ VERIFY_PASS
bash scripts/smoke-local/run_gl_dataset_training_smoke.sh   # 【第四层】
```

## walltime 怎么算（GPU 与 I/O 两条估算取大者）

单分片探针**测不出 8 路并发下的 turbo 带宽争用**，只按探针速率反算会严重低估。
`step_submit.sh` 的第 ⑦ 项因此同时算两条：

- **GPU 侧**：`总步数 ÷ 分片数 ÷ 稳态 step/s`
- **I/O 侧**：`(读 321 GB 原始 H5 + 写 总步数×每步字节) ÷ 卷带宽`。
  8 个分片**并发共享同一个 turbo 卷**，所以 I/O 侧的耗时对每个分片都是「全量字节 ÷ 卷带宽」，
  **不再除以分片数**。

取两者的大者 ×1.5 作为申请值，并硬断言裕度 ≥1.2×。按 483,291 步 × 588 KiB ≈ 284 GB 写
加 321 GB 读、卷带宽 132 MB/s 估，I/O 侧约 76 分钟——**这条流水线大概率是 I/O 受限而非 GPU 受限**，
也正因如此 CPU/mem 档位有下压空间。

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

```bash
rm -rf /nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_data_h5_v2_4env400ep
```

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
- **抽取期间 8 个 task 同读写一个 turbo 卷**（实测天花板 ~132 MB/s），期间勿并行起集群训练。
