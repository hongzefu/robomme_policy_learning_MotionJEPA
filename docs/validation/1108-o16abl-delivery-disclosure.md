# 双消融训练完成与交付路径披露裁决

训练代码永久锚定 `8b52210ff0048a9ea22776090fda3d76cd181c89`。两条正式训练均完成80000步，最终79999步检查点已完成异步保存，内外日志均为 `EXIT_CODE=0`，内日志均有 `TRAIN_FINAL_RECORD=PASS tail_steps=99 ema_leaves=59`。美东2026-11-08，noxyz于03:02:33退出，nostate于07:14:00退出。每条检查点目录集合精确为5000到75000每5000一步，加79999，共16份；此项不替代严格恢复。

## 用户裁决与实施中调整

用户先问「训练完毕了吗」。主会话现场复核后尝试原导出入口的离线模式，两个run均报告 `ORIG16_EXPORT=FAIL error=ValueError detail=公开前未裁决命中: encoder_dependencies.json P1`。初次运行载体把标签参数传给只接受零参数的公共入口，退出2；已在调用前消费标签，一次重试后才得到上述真实披露失败。全部旧日志保留。

主会话枚举交付根来源与完整训练日志，两个run均无凭据模式命中，缺少披露裁决的项目各为7项：`encoder_dependencies.json`、`motion_provenance.json`、`_run-meta/train.log`各自的P1/P2，以及`motion_evidence/G3.json`的P3。P1是集群绝对路径，P2是集群组织层级，P3是本机绝对路径。

主会话把这四文件与三类路径完整交用户，用户明确回复：「同意开始」。本次只给上述7组精确文件名与模式增加`accept`裁决，既有条目不改，不增加路径通配，不改扫描器、凭据硬停、恢复或数值判据。独立只读审查结论：`ADJUST_REVIEW=PASS scale=same findings=0`；`PLAN_CONSISTENCY_REVIEW=PASS base=1dbcc2cfa3fb04a46f143b4c91a9b709dde8aa86 findings=0 blocking=0 scope=disclosure_adjustment`。

## 验收边界与后续

补丁静态校验：`DISCLOSURE_SCOPE=PASS added=7 old_unchanged=1 exact_paths=4`，`git diff --check`退出0。执行`uv run --frozen --no-sync python -m pytest scripts/dataset/hf_export/test_orig16_export_drivers.py -q`，59项通过、9项失败，耗时5.47秒；9项均为数据集打包测试的子进程环境硬编码`UV_CACHE_DIR=/scratch/hongze/.cache/uv`，在Aspen无权创建而退出2。没有修改这些无关测试，也未将整套测试写成通过；本轮实际双消融离线导出仍作为真实验收。

披露补丁提交与训练Beta分别记录。继续离线生成完整交付清单，再在原占位65209321/65209330内执行79999步严格恢复。此前A6000/A40来源握手的`SAVE_LOAD=FAIL`原文保留，不因此改成通过。公开上传后的逐对象SHA、字节数和集合校验必须由GreatLakes标准分区纯CPU任务执行；未通过前不称已备份，不删除任何训练产物。
