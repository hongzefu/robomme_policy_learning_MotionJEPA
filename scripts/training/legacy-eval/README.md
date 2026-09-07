# legacy-eval：评估链路归档

主线（`v2-motionmem`）自 `commitV6.2` 起不再维护评估执行链路 —— `examples/robomme/` 与
`src/mme_vla_suite/policies/` 已回到 `4b7a710`（40k 训练收尾）那一刻的状态，`scripts/training/prod/`
下不再有任何评估脚本。**评估的东西全部集中在本目录**，并且**仍然可以直接运行**，不是封存的死代码。

## 为什么有 `.local` / `.remote` 两套

两条 `v2-motionmem` 分支在 2026-09-06 各自独立实现了同一个「benchmark split 选择」功能，
参数名与语义都不同，无法合并成一套：

| | `.local`（本机 8×A100 那条线） | `.remote`（另一台 2×RTX 6000 Ada 那条线） |
|---|---|---|
| 来源 commit | `c704bf5` commitV5.6 | `7fb5206` commitV5.9 |
| `eval.py` 参数 | `Args.dataset` → `--args.dataset` | `Args.dataset_split` → `--args.dataset_split` |
| shell 环境变量 | `DATASET=test\|val` | `SPLIT=test\|val\|train` |
| 结果目录 | 非 test 时另开 `<split>-seed<n>` 与 test 分家（否则会被续评逻辑当成「已评过」整体跳过） | 不分家，`setup_save_directory` 未改 |
| 逐集自证 | `make_env` 打 `EVAL_EPISODE split=… task=… ep=… env_seed=… difficulty=…` | 无 |
| 多卡适配 | 无（worker 号 = 卡号，固定 8 卡） | `GPU_LIST` 取模映射 + `POLICY_MEM_FRACTION` / `ROBOMME_PY` 转发 |
| 端口占用守卫 | 无 | 有（`baa93fa` + `1573ab9`：端口被占会被误判为 server 就绪，eval 静默跑 0 集仍退出码 0） |

**选哪套**：复现本地那三轮留档（`eval-awsprod40k-b128-motion`、`eval-official-framesamp-context`、
以及带 `DATASET=` 的命令）用 `.local`；复现远端留档（`eval-3seed-context-vs-motion`、
`eval-hard-patternlock-routestick`、`eval-medium-patternlock-routestick`、`eval-binfill-pickxtimes`，
以及一切带 `SPLIT=` / `GPU_LIST=` 的命令）用 `.remote`。少于 8 卡的机器只能用 `.remote`。

## 目录内容

| 文件 | 属于 | 说明 |
|---|---|---|
| `eval_shard.local.sh` / `eval_shard.remote.sh` | 各一份 | 单分片驱动：起 policy server → 跑 `eval.py` → 收 server → 汇总 TIMING |
| `eval_all_shards.local.sh` / `eval_all_shards.remote.sh` | 各一份 | 多分片总起（`MODE=task\|stride`），一片一个 detached tmux 会话 |
| `merge_eval_shards.py` | local | 合并分片结果，出 `summary.txt` / `per_episode.json` 与 `SPLIT_SEED_MATCH` 判定行 |
| `check_test_seeds.py` | local | 核 test/val split 的 seed 与训练 H5 seed 不相交 |
| `plot_eval_success_by_length.py` | local | 四任务成功率 × 任务长度对照图，出 `docs/eval-success-by-task-length.png` |
| `eval_seed_sweep.sh` | remote | 一个 (组, seed) 四任务按批串行跑完，逐片核 `EXIT_CODE=0` |
| `aggregate_seed_runs.py` | remote | 跨 seed / 跨任务批次汇总（`merge_eval_shards.py` 只处理单 run 单 seed） |
| `summarize_hard_eval.py` | remote | 难度档（hard / medium）专用汇总与 Fisher 检验 |
| `check_ckpt_param_tree.py` | 共有 | 官方 checkpoint 参数树严格核对 |
| `robomme-local/` `robomme-remote/` | 各一份 | `examples/robomme/{eval.py,utils.py,env_runner.py}` 两套实现的**只读副本** |
| `mme_vla_suite/policies/` | 共有 | `policy_config.py`（两侧同源）与 `motion_client.py`（远端版）的**只读副本** |

`robomme-*/` 与 `mme_vla_suite/` 下的副本**不能原地运行**（不在各自的包路径上，import 解析不到），
它们只是被主线回退掉的那 5 个文件的归档。

## 怎么跑

脚本的相对路径已适配本目录（`source ../paths.sh` 指向 `scripts/training/paths.sh`，
py 的 `parents[3]` 仍指仓库根），直接跑即可：

```bash
# 官方 framesamp+context 组（不走 motion sidecar），开箱即用
MODE=stride WORKERS=8 RUN_NAME=official-framesamp-context CKPT_ID=79999 \
  LOG_PREFIX=evX PORT_BASE=8031 DRY_RUN=1 \
  bash scripts/training/legacy-eval/eval_all_shards.remote.sh     # 先 DRY 看分片表，去掉 DRY_RUN 才真起
```

**motion 组有两个前置**，不做会静默出错：

```bash
# 1) eval.py 那套改动必须先拷回原位（主线已回退，没有 --args.dataset / --args.dataset_split）
cp scripts/training/legacy-eval/robomme-remote/*.py examples/robomme/    # 或 robomme-local/，与所选那套对应

# 2) policies 两个文件必须拷回 src/，否则：
#    - policy_config.py 不读 MMEVLA_MOTION_ONLINE_GPU → 8 个分片的 sidecar 全挤到快照写死的那张卡
#    - motion_client.py 恢复了物理 GPU 指纹硬拦 → 换机器跑会被直接拒绝
cp scripts/training/legacy-eval/mme_vla_suite/policies/*.py src/mme_vla_suite/policies/
```

拷回后工作区就不干净了，而 `eval_shard.*.sh` 有 AGENTS 12 的 clean HEAD 硬闸 —— 正式评估前需要把这些
改动提交成一个正式 commit（并在 `docs/training-doc/<run_name>/` 留档），不要用改工作区的方式绕过。

## 留档里的旧路径怎么对应

各评估留档 `launch.md` 里记录的是当时真实执行过的命令，路径一字未改。对应关系：

| 留档里的旧路径 | 现在的位置 |
|---|---|
| `scripts/training/prod/eval_shard.sh` | `eval_shard.local.sh` 或 `eval_shard.remote.sh`（按上表选） |
| `scripts/training/prod/eval_all_shards.sh` | `eval_all_shards.local.sh` 或 `eval_all_shards.remote.sh` |
| `scripts/training/prod/merge_eval_shards.py` 等其余 prod 脚本 | 本目录同名文件 |
| `scripts/analysis/plot_eval_success_by_length.py` | 本目录同名文件（`scripts/analysis/` 已不存在） |

## 演化来历

`7867dcd` commitV5.3 建链路（8 卡按集区间分片）→ `f210b40` commitV5.4 改 stride 交错负载均衡 →
两条线分叉：本地 `c704bf5` commitV5.6 加 split 选择与逐集自证；远端 `1f67cd6` commitV5.6 多 seed 链路、
`4f08c26` commitV5.7 批次 driver、`baa93fa`/`1573ab9` 端口守卫、`469fbfa` commitV5.8 GPU 指纹降级、
`7fb5206` commitV5.9 另一套 split 选择、`4a1574e` commitV6.0 难度档参数化。
两条线在 `commitV6.1` 合并，本目录在 `commitV6.2` 成形。

跑过的评估（结果见 `docs/training-doc/` 对应留档）：motion 40k **28.0%**、官方 context **24.0%**、
三 seed 复测 官方 **24.5%±0.5** vs motion **24.2%±1.3**、hard 档 modul **16.67%** vs context **0%**
(p=0.00285)、medium 档 **47.92%** vs **4.17%** (p=4.9e-07)、BinFill/PickXTimes 两变体无差异 (p=0.50)。
