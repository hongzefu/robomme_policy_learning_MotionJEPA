# v1-upstream-eq 起跑档（C4：上游 main 分支对拍，A/B 各 1000 步）

- **run_name**：`v1-upstream-eq`（用户 2026-08-30 经 AskUserQuestion 确认采计划拟名；两侧
  exp-name `entry-eq-a` / `entry-eq-b`，临时 run，PASS 后清理）。
- **计划**：`v5.0-train-entry-restructure-plan.md` 4.2/第十节；用户已授权（A 侧 1000 步 +
  12 步预热 + 冷编译、B 侧 1000 步 + 冷编译，本机 2 卡，两侧串行约半天 + 2 h）。
- **上游锚点**：`ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`（RoboMME main = fork main =
  本地 main 同 sha，且为本分支分叉点）。可比性机器实证见计划 4.2（G0b 起跑 commit 的
  `scripts/train.py` 与 main tip 逐字节同 blob `a8be7a5`）。
- **A 侧 worktree**：`v1-store/entryeq/worktree-main` @ `ecf086c`（已建，独立 `uv sync`
  完成，`.venv` 就绪；`uv.lock` 分叉至今零 diff → 环境同一）。
- **harness**：`scripts/training/tests/entry_equiv.py`（run/judge）+ `run_entry_equiv.sh`
  （run-a/run-b/judge），commit `f641f40` 落地；投影与 C3 收官共用
  `tests/project_scalars.py`（`--selftest` 已实证逐字节复现 G3 产物）。

## 两侧口径

| | A 侧（上游原版） | B 侧（本分支） |
|---|---|---|
| 代码 | worktree @ `ecf086c` | V5.0 tip |
| 入口 | `scripts/train.py` 官方 `__main__` 双跑（runpy，加 `--overwrite`） | `scripts/training/train.py` 官方 `__main__` 单跑 |
| 数据 | legacy 库 `v1-store/datasets/4task-gl` | packed 库 `v1-store/datasets/4task-gl-framesamp` |
| 步数/口径 | 1000 步、b8 / seed 42 / workers 4 / fsdp 2 / log-interval 1 / save-interval 100 / context 变体 / 确定性档 | 同左 |

argv 差异**五处**（其余逐字符同，由 `run_entry_equiv.sh::common_args` 机器保证）：入口文件、
`--exp-name`（连带 `~/.cache/jax_{exp_name}` 编译缓存目录，两侧缓存天然分离）、
`--dataset-path`（被测对象）、`--checkpoint-base-dir`、A 侧独有 `--overwrite`。
B 侧不设 `TRAIN_RECORD_DIR`（内置记录器不装，观测全由 harness 代理承担）。

## preflight 三项

- **P1**（起跑前执行）：A 侧 `--num-train-steps 2` 冒烟走 harness（`save_state` 已被摘要器
  替换，无落权重问题）：`STEPS=2 RECORD_ROOT=<tmp> bash tests/run_entry_equiv.sh run-a`，
  判据：跑通出 2 步 loss、`SEGMENTS tentative_rows=2 main_rows=2`（2 步下 tentative 段
  `step>10` 不 break、跑满 2 步）、`ENTRY_RUN=OK`；跑完删临时目录。
- **P2**（已完成，2026-08-30）：`ecf086c` 版 `config.py` 静态核对——A 侧 argv 全部 15 个
  config 字段 + `weight_loaders.py::params_path` + `history_pi0.py::use_history/history_config`
  逐一存在。**PASS**。
- **P3**（已完成，2026-08-30）：两侧实际加载 norm_stats 同源——两侧 argv 同指
  `v1-store/train-assets`，`mme_vla_suite/robomme/norm_stats.json` sha256 实测
  `709f22ff5cd43c08c43d6485c032178a3f5b57cfb1a39a446f934cb34636fc98`，命中 G0b 指纹
  `norm_stats_sha256`。**PASS**。

## 长任务规范（AGENTS 6/7/12/17）

两侧各起 detached tmux（`v1-upstream-eq-a` / `-b`，串行），各一份 `tee` 日志、结束写
`EXIT_CODE=`、各挂一个 Monitor（管道每级行缓冲）；本 launch.md 起跑前提交。

## 判据（4.5）

```
A_SIDE_SEGMENTS tentative_rows=12 main_rows=1000        # 留档非判据
ENTRY_SCALARS steps=1000 keys=5 hex_mismatch=0
ENTRY_STATE_DIGEST rows=<首轮实测钉死> mismatch=0        # save-interval 100 + 末步，预期约 10–11 行
ENTRY_RESOLVED_CFG mismatch=0 whitelist=4
ENTRY_PROVENANCE=PASS
A_SCALARS_SHA256 == c799a0b2…                            # 第六/七份同值之一；A 命中是待验非预期必然
B_SCALARS_SHA256 == c799a0b2…
ENTRY_EQ=PASS                                            # 定义 = 上面五条子判据全过
```

四象限判读：A✅B✅ 收工；A❌B✅ = tentative 预热污染正式轨迹（正是从未实测的怀疑点）；
A✅B❌ = 新 train.py 改坏；A❌B❌ = harness 自己在改数，先修工具。
A 侧 tentative 段（预期 step 0..11 共 12 行）单独留档不判，tentative 段零 save。

## 收尾

PASS 后清理 worktree（`git worktree remove --force` + `prune`）、两侧临时 ckpt 目录与
`v1-store/entryeq/`；FAIL 记录按 `.failed-<n>` 惯例保留。留档 `docs/training-doc/v1-upstream-eq/`。
