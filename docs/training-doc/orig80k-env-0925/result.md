# P0通过，正式建库与训练仍待前置问题解决

上游独立uv环境已建立，源码补丁已完全恢复；A/B均为Python 3.11.15，关键依赖一致，补充全量分发包版本比较为208项对208项、差异0。`ORIG80K_ENV=PASS`、`P0_PROCESS_EXIT=0`、`TEE_EXIT=0`、`EXIT_CODE=0`。这只证明环境准备完成，未运行GPU训练或真实数据输入对拍。

## 用户决定与未决项

用户原话：「开始实现该计划 有问题立刻问用户 不要自己决策」。本轮已实现根计划的验证和启动工具，严格输入判据未放宽。已提出的三项问题仍待答复：上游短历史padding的dtype差异如何处理；由用户清理磁盘的安排；是否认可上游缺失四motion字段与当前四键全为None等价。未收到答复前，保持严格失败，不进行全量建库或正式训练。

## 版本与实际启动

主副本起跑及结束均为clean `5489e4b3a92197d0e9a37421b1e6415b3022b613`，上游worktree为clean `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`。P0使用该提交中的 `prepare_orig80k_env.py`；运行期间未修改主副本代码，归档在任务结束后进行。

实际tmux会话完整名称为 `orig80k-p0-20260925T184222Z`。日志记录启动时间 `2026-09-25T18:42:38Z`、结束时间 `2026-09-25T18:42:41Z`；Python准备器的起止时间差为2.33秒。任务按预计较长的环境准备使用detached tmux，缓存命中使实际耗时很短；会话自然退出，未调用kill，也未清理其他六个原有会话。

实际命令如下，外围使用 `PYTHONUNBUFFERED=1`、`set -o pipefail`、tee，并分别记录准备器与tee退出码：

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
UV_CACHE_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/cache/uv \
  uv run --no-sync python scripts/training/tests/prepare_orig80k_env.py \
  --head 5489e4b3a92197d0e9a37421b1e6415b3022b613 \
  --worktree /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/worktrees/orig-ecf086c \
  --records /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/orig80k-env-0925
```

## 环境与恢复验证

上游 `.venv` 独立创建在其worktree内，使用与主副本相同的已存在Python解释器；新下载目录、uv缓存与工作树均按计划限定在本仓库 `v1-store/`，没有修改HOME或主副本 `.venv`。uv输出安装208个包，其中两个editable项目重新构建。A的三个项目模块分别落在A的src/packages目录，B均落在主副本，具体路径和摘要见原始JSON记录。

| 项目 | 两侧共同值 |
|---|---|
| Python | 3.11.15，Clang 22.1.3 |
| JAX / jaxlib | 0.5.3 / 0.5.3 |
| PyTorch | 2.7.1 |
| NumPy / ml-dtypes | 1.26.4 / 0.4.1 |
| Flax / Optax | 0.10.2 / 0.2.4 |
| Orbax checkpoint | 0.11.13 |

临时移除缺失workspace成员的补丁SHA256为 `e44f6e26dfc83eebbd344406c965374f6a51a388d40c0b761c0a73eb320bd7ad`；恢复后上游 `pyproject.toml` SHA256为 `6c347c23df7e77ab44ff1db986c128126e4ae375700e99c191700a1cb174e0a1`，`uv.lock`为 `02cbc3ba67a9024f8afb9e31f60661c9abdcc3eb680ae80a8ee464c639327221`，均与准备前一致。记录 `temporary_patch_applied=false`，两侧 `git status --porcelain` 均为空。

P0工具验证八个关键依赖后，又用两侧解释器分别执行 `importlib.metadata.distributions()`，将分发包名转为小写并把下划线替换为连字符，逐项比较版本；两边各208项，差异字典为空。该补充检查未安装或修改任何依赖，结果单独归档。

## 工具验证与资源状态

提交前合并执行八个本轮测试文件，命令为 `uv run --no-sync pytest -q`，目标为 `scripts/dataset/test_orig80k_sources.py`、`scripts/dataset/test_subset_eq.py` 及 `scripts/training/tests/test_orig80k_{inputs,entry_equiv,launch,completion,env,speed}.py`。实测 **271 passed，40.64秒**，仅有既有beartype类型注解弃用警告。测试包含真实CLI、torch持久worker、runpy/JAX叶摘要、shell失败传播和微型Orbax保存/生产恢复；不代表大型模型训练已通过。定向Ruff、两个shell入口语法及暂存空白检查通过。

环境为AWS本地 `/dev/md0` XFS，8×A100-SXM4-80GB；P0没有使用GPU训练。结束后 `df -B1` 可用 `1221779390464 B`。全量建库仍低于计划预算，未自行删除任何数据、checkpoint或用户会话。

## 归档与下一步

归档文件为 [records/environment.json](records/environment.json)、[records/dependency_full_check.json](records/dependency_full_check.json) 和 [records/environment.summary.log](records/environment.summary.log)。源码、锁文件和启动配置由固定Git提交及本文命令还原，不额外复制脚本或yaml。上游worktree和独立 `.venv` 保留在 `v1-store/`，供后续已授权对拍使用。

后续等待用户解决三个已提问题，再按计划复核磁盘、源文件摘要，推进构建冒烟、正式库、输入及训练对拍。未获答复前不自行接受 dtype/schema 差异，也不降低磁盘余量。
