# 原版80k环境P0启动记录

本档案在提交 `5489e4b3a92197d0e9a37421b1e6415b3022b613` 中预先建立，起跑时为“尚未执行”状态。P0现已完成，实际命令、版本、会话和验收见 [result.md](result.md)。正式建库、输入差异判定与两个80k训练仍由根计划中的独立闸门控制。

## 用户决定与范围

用户原话：「开始实现该计划 有问题立刻问用户 不要自己决策」。执行依据为根目录 [原版80k计划](../../../0925-orig-80k-full-counting-4plus4-plan.md)。当前只准备固定上游的独立uv环境，验证主副本与上游各自使用自身代码及一致依赖；不修改训练源码或超参，不读取全量H5，不启动GPU训练。

用户已被询问三项：上游短历史padding的dtype差异；本机磁盘预算不足；上游缺失四motion字段与当前显式None是否可视为等价。答复前不放宽严格输入判据，不启动全量建库。环境P0与这些决策独立。

## 版本与起跑状态

工具开发基线为 `a433f4e36b9475198c5ead57aa1578f9a40ff30b`，上游固定 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`。本档案和工具必须提交后，才从clean HEAD启动；实际完整实施SHA作为 `--head` 字面量传入，记录在运行 `environment.json` 和清洗日志中。本文不提前声称尚未产生的提交就是起跑版本。

已核实主副本为 `/scratch/hongze/robomme_policy_learning_MotionJEPA`，`v1-store` 为实体目录；8张A100-SXM4-80GB当前空闲；存储为 `/dev/md0` XFS。本机Python为uv管理的3.11.15，主环境 `uv run --no-sync` 可执行。首次磁盘快照为 `1221847916544 B` 可用，低于全量建库预算；P0不建立两个大数据集。

## 启动与配置还原

入口为 `scripts/training/tests/prepare_orig80k_env.py`。启动命令模板如下；实际起跑前必须把 `<实施提交完整SHA>` 替换成已提交的40位字面量，并将展开后的命令写入日志。唯一输出根均要求不存在，失败保留现场，不覆盖重试。

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
UV_CACHE_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/uv \
  uv run --no-sync python scripts/training/tests/prepare_orig80k_env.py \
  --head <实施提交完整SHA> \
  --worktree /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/worktrees/orig-ecf086c \
  --records /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-env-0925
```

任务使用本轮唯一 detached tmux，会话完整名称在实际起跑时记入结果；`PYTHONUNBUFFERED=1`、`set -o pipefail`、tee日志及 `EXIT_CODE=` 必须齐全。日志落 `v1-store/logs/orig80k-env-0925.log`。仅检查或清理本轮记录的完整会话名，不清理其他会话。

上游 `pyproject.toml` 声明缺失的 `sandbox2/flash_attn_jax`，但该成员不在上游锁文件workspace清单中。工具仅在uv安装期间用可逆补丁去掉这项成员，`uv sync --frozen` 完成或失败后反向应用完全相同补丁；核对原 `pyproject.toml` 与 `uv.lock` 摘要恢复、A/B工作区clean。独立A `.venv` 指向同一已存在的Python解释器，不下载或修改主环境。uv缓存和任何新产物都落主副本 `v1-store`，不覆盖HOME。

## 验收与结果保留

成功要求唯一 `ORIG80K_ENV=PASS`、`EXIT_CODE=0`，源码及锁文件摘要恢复，A实际 `sys.prefix` 为其独立 `.venv`，两侧Python构建信息和关键依赖版本一致，A的 `mme_vla_suite/openpi/openpi_client` 模块全部来自A worktree，B模块全部来自主副本且不穿入A。失败立即向用户报告；不自行改变依赖、上游提交或源码以凑通过。

运行后将真实 `environment.json`、清洗日志及判定归档到本目录 `records/`，补 `result.md` 并更新索引。大体积 `.venv` 和worktree留在 `v1-store/`，不进Git；本P0不自动删除失败现场。
