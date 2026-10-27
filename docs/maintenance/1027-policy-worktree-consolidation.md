# policy 工作目录合并与清理

2026-10-27 美东，按用户原话「你现在policy learning有3个repo 你看一下是否可以合并 然后删除另外两个copy的」完成。

保留 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，分支 `v4-256concat1`。
已通过 `git worktree remove` 删除本机 `robomme_policy_learning_MotionJEPA-temp` 和 `robomme_policy_learning_MotionJEPA-eval` 两个目录；三个目录本来共享同一个 Git 对象库，分支与提交历史继续保留。

| 来源 | 清理前提交 | 处理结果 |
|---|---|---|
| 主目录 | `c1a794ee205504d868e349aea1ae3ed80d2f41f4` | 保留为合并第一父 |
| `-temp` | `7dd98dd262966af0610ddd939dbf7c3466edea53` | 已在原主分支历史中 |
| `-eval` 本地 | `db9c3eec5bfc23f4e4c5ffa2cedb9606aafd0db6` | 已包含在评估远端历史中 |
| `-eval` 远端 | `3a4c32a43a5b539931eab38f122eaeb110367c35` | 比本地新65个提交，以第二父完整合入 |

合并提交为 `6126d494a7488a2f597054d9706d6007df854d3f`，已推送 `origin/v4-256concat1`。仅两个文件发生冲突：

- `motion_client.py` 保留两边功能；脚本身份检查按启动入口记录参数偏移，兼容 `uv` 与显式 Python。真实进程、socket 身份、会话归属和清理检查均保留。
- `docs/handoff/current.json` 保留主分支6项任务，完整加入评估分支4项任务。70k停止、同源失败、长尾上传及另一台机器的评估运行状态均保留；清理后仅将一条本机证据路径从旧 `-temp` 改为主目录的同一文件。

删除前两个目录都没有未提交代码或未跟踪文件，也没有活动进程引用其中的目录或打开文件。`-temp` 没有忽略产物；`-eval` 的忽略产物仅为 `v1-store/envs/eval-cpu`，包含9个标准安装包、4451个文件，清单已保存在本机证据目录。两个工作目录原占用共812400640字节（约775 MiB）。主目录的 `v1-store`、模型权重、训练数据和产物保留；没有启动或停止既有训练、评估或集群作业。

## 验证与已知限制

- `POST_MERGE_REVIEW=PASS tree=cb08d8971294b1f21677641eba02a98daf57b7f1 findings=0 blocking=0`：独立审查确认兼容修复与双方状态保全。
- `REQUIRED_CPU_TESTS=PASS passed=289 failed=0 skipped=0 seconds=38.00`：真实进程/socket、两种解释器入口、身份负例、恢复、70k导出、评估运行合同与现行队列通过；退出码0。
- `WORKTREE_CONSOLIDATION=PASS copies_removed=2 unique_history_preserved=1 main_store_preserved=1`：两个目录及对应工作树登记均已移除，全部源提交仍是主分支祖先。
- `git diff --check` 通过；主分支已同步 GitHub。

测试使用主目录现有 `.venv/bin/python -m pytest -p no:cacheprovider -q`，环境为 `CUDA_VISIBLE_DEVICES=''`、`JAX_PLATFORMS=cpu`、`UV_OFFLINE=1`、`UV_CACHE_DIR=/scratch/hongze/.cache/uv`，并设置 `MMEVLA_STAGE2_IMPORT_ENV=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/venvs/wan-stage2`，真实 `uv` 用例没有跳过。必要测试文件：

```text
tests/policies/test_motion_client_stage2.py
tests/policies/test_stage2_restore.py
tests/dataset/test_stage2_export_contract.py
tests/evaluation/test_runtime_restore.py
tests/evaluation/test_eval_target_contract.py
tests/evaluation/test_simple_queue.py
```

扩展测试有524项通过、1项失败，退出码1：`test_testx_manifest_from_real_jsonl` 依赖另一环境的 `/nfs/turbo/.../eval-testx-1200.jsonl`，AWS没有该文件。全评估测试收集另有两项错误：`test_gl_pool.py` 与 `test_resident_worker.py` 引用已经删除的 `Pool`、`atomic_json` 等旧接口。相关实现与测试逐字节等于评估远端锚点，独立审查确认属于既有问题；本次没有改写旧队列或补造集群数据，不声明全套测试通过。

本机证据位于 `v1-store/maintenance/policy-consolidation-20261027/`：`before-cleanup.json`、`after-cleanup.json`、`required-tests.log`、`targeted-tests.log`、`tests.log`。提交历史和上述清理前锚点可用于恢复代码；CPU临时环境的包版本清单在 `before-cleanup.json`。
