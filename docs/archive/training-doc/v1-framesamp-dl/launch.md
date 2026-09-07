# v1-framesamp-dl-w{2,4,8,16}（S8a GL dataloader-only 四档）launch 记录

对应 `v2-framesamp-restructure-plan.md` 阶段 4 S8a（D 节）。留档体例沿
`v1-gl-dlbench` 先例：四个单档 job 合并一个目录。

## 档位（2026-08-27 经用户确认——「四档全跑」，提交前档位确认为用户新增硬性要求）

| run_name | workers | cpus | mem | seed | 资源 |
|---|---|---|---|---|---|
| `v1-framesamp-dl-w2` | 2 | 4 | 24G | 310 | 1×A40 / 30 min |
| `v1-framesamp-dl-w4` | 4 | 6 | 24G | 311 | 1×A40 / 30 min |
| `v1-framesamp-dl-w8` | 8 | 10 | 32G | 312 | 1×A40 / 30 min |
| `v1-framesamp-dl-w16` | 16 | 18 | 48G | 313 | 1×A40 / 30 min |

- 配比沿历史口径 cpus≈workers+2；seed 310–313（避开已用 42/200–205/210–212，防
  page cache 串扰）；**全部在 greatlakes.md debug 包络内（1 GPU、≤30 min），无超限
  特批项**。
- **backend/库**：`MMEVLA_DATA_BACKEND=packed`（显式，R16）+
  `DATASET_PATH=v1-store/datasets/4task-gl-framesamp`（S4 交付，status=verified）；
  `MMEVLA_FRAMESAMP_VERIFY` 缺省 fast——**冷态自证：本 allocation 不跑 full 校验、
  无本地复制预热**（D 节）。
- 入口：`gl_dlbench_single.sbatch`（S7.5 参数化 + RUN_NAME 覆盖）→
  `dataloader_bench.py`（batch 64、warmup 5、measure 40、SEG_PROBE_N=200 分段计时）。

## 起跑环境

- **起跑 HEAD**：本 launch.md 的 docs commit 本身（结果留档回填全 sha）
- ControlMaster 存活复用（零认证）；提交器 `gl_submit.py`（greatlakes.md 规约）

## 提交命令（每档一条，经 gl_submit 复用 master）

```bash
uv run --no-project --with pexpect python scripts/data-preprocess-GL/gl_submit.py \
  "sbatch --parsable --job-name=v1-framesamp-dl-w<W> --cpus-per-task=<C> --mem=<M> \
   --export=ALL,WORKERS=<W>,BENCH_SEED=<S>,TAG=framesamp-w<W>,RUN_NAME=v1-framesamp-dl-w<W>,DATASET_PATH=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-gl-framesamp,MMEVLA_DATA_BACKEND=packed \
   scripts/bottleneck-bench/gl-dataloader/gl_dlbench_single.sbatch"
```

## 判据与产物

- 每 job：`DLBENCH_PASS` + `RESULT workers=… 样本/s=… MB/s(公式/实测)`（公式口径
  packed 2.43 MB/样本现场推导）+ `SEGPROBE`（gather/pkl 分段）+ `EXIT_CODE=0`；
  env.json 记 backend=packed 显式与 resolved 双根。
- S8a 无过/不过阈值（吞吐数据落档供 S8b 对照）；对照组：legacy 历史
  v1-dlb-w{4,8,16}c* 与 v1-gl-dlbench。
- 记录目录 `v1-store/bench/bottleneck/v1-framesamp-dl-w{2,4,8,16}/`；结果留档
  result.md 随 docs 收官提交。
