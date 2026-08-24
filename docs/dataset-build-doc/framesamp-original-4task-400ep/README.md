# framesamp-original-4task-400ep（已弃用）

> **状态：弃用（2026-08-23）。本档案连同其固定入口脚本一并作废，请勿再参照。**

## 弃用说明

本档案对应的实现是 commit `d951aef` 引入的 `scripts/v1_dataloader_restructure/`
与 `scripts/smoke_train_once.py`。这批脚本**经用户判定不可靠**，已于本轮 `git rm` 删除
（历史保留在 git 中，不做 revert）。**它们从未实际运行过**——档案里描述的
`.openpi-data/`、`data/robomme_preprocessed_4task_original*`、
`runs/assets/...`、`artifacts/v1_dataloader_restructure/` 等路径都不曾生成。

因此本档案记录的以下内容**全部作废**，不得作为任何后续工作的依据：

- 「项目内路径」一节列出的目录约定；
- 「固定入口」一节列出的四个 Bash 启动器；
- 「起跑与验收」一节描述的本地单卡全量构建流程。

## 现行替代

四任务全量数据处理已改为在 GreatLakes 上以 8×1GPU job array 完成，路径布局全部重新定义。
现行方案、逐段流程与全部实测数字见：

- 方案与实测报告：[`docs/v1-gl-dataset-consistency-report.md`](../../v1-gl-dataset-consistency-report.md)
- 集群链路实现：`scripts/data-preprocess-GL/`
- 构建档案：[`docs/dataset-build-doc/4task-gl-400ep/`](../4task-gl-400ep/README.md)

## 保留原因

仅作为「此路不通」的记录保留，避免后来者从 git 历史里翻出这批脚本重新采用。
