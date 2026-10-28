# 官方16×100 ctx：8卡20步与严格保存恢复

## 一、目的与当前状态

按[已批准计划](../../../1027-orig16-vla-512-and-stage2-motion-4plus4-plan.md)执行原G7短测。当前只创建起跑档案，尚未运行。本条为20步完整VLA训练及step19严格保存恢复；不把CPU单层矩阵替代为真实训练。用户最新裁决：“不要后置任何步骤 该做的还是做 但是失败了只记录 让用户决策”。所有检查执行，真实失败留档，继续技术上可执行工作。

## 二、版本与代码状态

模型/启动逻辑及实际数据常量以`941fcfde1571e3c164937b4f817e38a7229f8d07`为代码基点，后续至本档案仅有文档提交。实际启动前取clean HEAD并作为`TRAIN_HEAD`传入；完整SHA由入口写入该条日志及运行记录，结果档案据此回填。计划批准锚点`d5fe05d647a899e016bc513aa5fba03485a9b398`，本短测名称创建于美东2026-10-28 04:31，实际起跑时刻以日志为准。

## 三、启动与配置还原

运行载体`v1-store/logs/o16-20261028/smoke-stage.sh`的SHA为`6c1724f61245e981740ddf4c760b931d3a892bde5378f845e17b69d8c417ee2f`，调用既有正式入口及恢复器，不复制脚本进入本目录。以下三阶段逐项执行并保留真实退出码：

```bash
cd /scratch/hongze/robomme_policy_learning_MotionJEPA
export TRAIN_HEAD=$(git rev-parse HEAD)
export RUN_NAME=o16-smoke-ctx-10280431 FUSION=ctx
MODE=contract bash v1-store/logs/o16-20261028/smoke-stage.sh
MODE=train bash v1-store/logs/o16-20261028/smoke-stage.sh
MODE=restore bash v1-store/logs/o16-20261028/smoke-stage.sh
```

训练：`MAX_STEPS=20`，`HISTORY=perceptual-framesamp-ctx-32frame-4x4-motion-stage2-4x256-orig16.yaml`，`FINAL_RECORD_DIR=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/validation/orig16-stage2/o16-smoke-ctx-10280431/final-record`。入口其余路径/缓存/GPU环境覆盖以相同HEAD的`run_orig16_stage2_60k.sh`为准；用`git show <日志TRAIN_HEAD>:src/mme_vla_suite/models/config/robomme/perceptual-framesamp-ctx-32frame-4x4-motion-stage2-4x256-orig16.yaml`还原原配置。运行期间主会话不修改训练源代码或环境。恢复检查沿用本条真实checkpoint19与`final-record/pre_save_reference.json`，父GPU0、sidecar1。

## 四、数据与来源

库`16task-orig100-1600ep-9e109419`：1600集、768897帧、476857执行样本；4×4和8×8帧表均已全量verified。消费4×4帧表、23253行stage2动作表；同一encoder源H5逐SHA匹配。encoder为ep105/EMA，SHA `27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035`。已完成数据身份、padding、全表split及窗口预算44/96验证；两种G4均须先完整执行，使用真实portable恢复输入。完整数据身份见[build-pins](../../dataset-build-doc/16task-orig100-1600ep-9e109419/records/build-pins.json)。

## 五、固定超参

配置`mme_vla_suite_orig16_b128_60k`：global batch128、FSDP8、workers16、seed42。仅训练步数由同一入口的既定短测覆盖改为20，60k学习率日程不缩短；32帧×4×4共512视觉位，每窗4×256、96窗/384动作位，合计896位。ctx对应融合宽度为2048，与另一条仅融合方式及相应宽度不同。

## 六、硬件、句柄、产物与监督

环境B：8×A100-SXM4-80GB，本地NVMe RAID。训练tmux `o16-smoke-ctx`，恢复tmux `o16-restore-ctx`；GPU训练占0–7，恢复父0/sidecar1。训练、恢复各2小时监督预算，主会话监听并保留失败，不自动改batch/模型/seed。

- 训练日志：`v1-store/logs/o16-smoke-ctx-10280431.log`。
- 契约/恢复日志：`v1-store/logs/o16-20261028/o16-smoke-ctx-10280431-contract.log`、`o16-smoke-ctx-10280431-restore.log`。
- checkpoint：`/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-runs/mme_vla_suite_orig16_b128_60k/o16-smoke-ctx-10280431/19`。
- 最终记录和保存前参考：`v1-store/validation/orig16-stage2/o16-smoke-ctx-10280431/final-record/`。
- 步指标和实际运行信息：同级`train-record/`。
- 恢复结果：同级`save-load.json`。

原判据为实际20步、8GPU/B128、有限标量及严格`SAVE_LOAD`；任何失败保持真实结果，不能改写PASS。正式60k仍是后续两条固定名称，本短测不是正式训练，也不做评估。
