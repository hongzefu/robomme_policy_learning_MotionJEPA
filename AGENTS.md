# AGENTS.md

本文件规定 agent 在本仓库中的工作方式。所有仓库任务都必须先遵守本文件，再结合用户当前明确指令确定本轮范围。

## 强制规则（最高优先级）

1. 所有计划、提问、进度、解释和最终总结必须使用简体中文。代码、命令、路径、标识符、库名和 API 名保留原文。仓库内新增或修改的注释与文档也必须使用中文。

2. 所有计划必须用中文书写。文档中的项目目标、未来 scope 或 roadmap 不等于当前实施授权；只执行用户本轮明确要求的工作。遇到范围、实现方式或破坏性操作存在歧义时，必须先询问用户，不得擅自扩展。

3. 执行任何 Python 命令前，必须先确认工作区是否提供 `uv`、`uv.lock` 或由 uv 管理的 `pyproject.toml`。uv 可用时：
   - 使用 `uv run` 执行脚本，禁止裸 `python` 或 `python3`。
   - 使用 `uv add` 管理正式依赖；依赖变更必须同时落入 `pyproject.toml` 和 `uv.lock`。
   - 禁止裸 `pip install`。正式依赖也禁止只用 `uv pip install` 临时安装而不更新项目配置。
   - NFS 上执行 uv 操作必须设置 `UV_LINK_MODE=copy`。

4. 每次代码改动后必须运行覆盖核心路径的验证，总耗时尽量控制在 5 分钟以内。全量测试过长时选择能够覆盖改动的最小真实子集，不能直接跳过。纯文档改动至少运行 Markdown/diff 空白检查并核对最终文件范围。

5. patch 级特征图或热力图的放大只能使用 `cv2.INTER_NEAREST`。真实照片帧的缩放不受此限制。

6. 启动正式长训练前必须向用户确认一个全新的 `run_name`。禁止通过复用名称或覆盖参数清空已有 run。跑完即删的 smoke 或短测可以自行命名，但验证完成后必须清理对应临时 run。

7. 预计超过 5 分钟的训练、抽取、评估或全量数据构建必须放入 detached tmux session。日志使用 `PYTHONUNBUFFERED=1`、`set -o pipefail` 和 `tee`，结束时写入 `EXIT_CODE=`。禁止使用裸 `pgrep -f` 判断进程存活；tmux 任务使用 `tmux has-session`，其他任务记录精确 PID。

8. 向 GreatLakes 提交 Slurm 作业前必须遵守仓库根目录的 `greatlakes.md`。如果该文件尚不存在，必须先向用户确认集群 account、partition、资源上限和 NFS 路径，不得直接复制其他仓库的集群配置。

9. 仓库文档引用代码时禁止使用易漂移的硬编码行号。应使用函数名、类名、方法名、配置键或语义段落作为稳定锚点。

10. 修改学习率、batch size、训练步数、loss 权重等训练超参前，必须先让用户确认改动应落在全局默认配置还是具体启动脚本的覆盖参数中。

11. 完成一轮改动并验证后必须提交 Git commit：
    - commit subject 和 body 使用简体中文。
    - 功能性改动使用 `commitV<大版本>.<小版本>: <中文描述>`；文档、修补和撤销使用 `docs:`、`fix:`、`revert:`。
    - 提交前运行 `git status --short`，只对本轮明确文件逐个执行 `git add`。
    - 禁止 `git add .`、`git add -A` 和 `git commit -a`。
    - 不得提交、stash、删除或回滚用户及其他 agent 的在途改动。

12. 正式训练或评估必须从 clean HEAD 启动，并在 `docs/training-doc/<run_name>/` 留档。正式全量数据集构建必须在 `docs/dataset-build-doc/<dataset_name>/` 留档。起跑前记录可复现的 commit、命令、配置、数据来源和输出路径；只归档 Git 无法还原的日志、指标和结果，不归档大模型权重。

13. 本机 `/data` 只用于数据处理、正确性检查和 smoke run。本机结果不得作为 dataloader 最终吞吐结论；正式吞吐基准必须在 NFS 数据副本上运行，并记录存储位置、batch size、worker 数、warmup 和稳定态统计。

14. 除最初的全局原始 H5 外，派生数据、索引、缓存、模型、tokenizer、checkpoint、日志和 smoke 产物都必须放在本仓库目录内。不得自行把新的外部目录作为长期依赖。

## 项目 scope（未来工作，不代表当前实施授权）

- 仓库总体目标：修改 MME-VLA 的 `perceptual-framesamp-context`，并在后续阶段接入 [MotionJEPA](https://github.com/hongzefu/MotionJEPA) motion token。
- `v1-dataloader-Restructure` 分支只用于 dataloader 重构，目标是在不改变训练语义的前提下尽可能提升训练吞吐。
- v1 只关注 `ButtonUnmask`、`VideoUnmask`、`ButtonUnmaskSwap`、`VideoUnmaskSwap` 四个任务。
- 未来的数据处理和 smoke run 在本机 `/data/hongzefu` 完成；本机吞吐不作为最终指标，最终吞吐在 NFS 上验证。
- 除全局原始 H5 外，后续生成的文件和模型全部放在 `/data/hongzefu/robomme_policy_learning_MotionJEPA` 项目内。

## 规则来源

通用规则迁移自 MotionJEPA commit `a9a467e3a4536e68f620283703e331ed469a561d` 的 [`CLAUDE.md`](https://github.com/hongzefu/MotionJEPA/blob/a9a467e3a4536e68f620283703e331ed469a561d/CLAUDE.md)。Wan/v7 专属架构、旧评估指标和旧项目命令不属于通用规则，未迁移到本仓库。Claude Workflow 与 Agent 模型选择规则属于 Claude Code 独有机制，不写在本文件，已落在仓库根目录 [`CLAUDE.md`](CLAUDE.md)。
