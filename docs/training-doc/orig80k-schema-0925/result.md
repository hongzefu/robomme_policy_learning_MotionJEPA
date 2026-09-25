# 共同字段数值相同，四个None字段仍使严格schema比较失败

对既有40集库的索引 `0/31/32`，两侧raw的15个共同字段、transforms后的10个共同字段都精确一致，无误差容差。当前侧独有 `mem_order`、`motion_emb`、`motion_mask`、`motion_pos`，三个样本两个阶段均为 `None`；上游均缺失，因此严格schema判据仍失败。该结果只覆盖三个真实样本，不代表正式两个新库或100步训练对拍通过。

## 用户决定与版本

用户原话：「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」；对字段差异则为「先保留严格判据并取证，结果出来后再决定」。本轮按这两个决定取证，未自行接受缺键与None等价。

A来自clean `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b` 的独立 `.venv`；B来自clean `1b504b2e4d2c3be0a0b92b5cf0c2b287bca1e53e`。共同量具为主仓库该版本的 `probe_orig80k_schema.py` 和 `check_orig80k_inputs.py`；两侧项目模块归属写入JSON的起止provenance。

## 输入与真实执行

数据为 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-40ep`，仅读已有source和verified 4x4 packed。索引0/31/32属于ButtonUnmask的raw episode 0，step相同。norm_stats为原版 `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`，两侧共用相同文件。实际只构造Dataset/transforms并读取样本，不迭代worker、不初始化SigLIP、不计算训练梯度。

两侧均设置 `CUDA_VISIBLE_DEVICES=''`、`JAX_PLATFORMS=cpu`、`OPENPI_DATA_HOME=<主仓库>/v1-store/models`、`PYTHONUNBUFFERED=1` 及scratch缓存，清除 `PYTHONPATH/PYTHONHOME`。A在上游根用其 `.venv/bin/python` 执行主仓库工具，B在主根用 `uv run --no-sync python` 执行；共同参数为 `--side <侧> --expect-root <侧根> --expect-head <上述完整SHA> --out <主仓库>/v1-store/bench/orig80k-schema-0925/<侧>.json`，B额外 `--forbid-root <A根>`。这两次调用的输出路径均为本轮全新文件。

两侧从 `2026-09-25T20:23:05Z` 开始，B于 `20:23:13Z` 结束、A于 `20:23:22Z` 结束，均小于5分钟。精确shell PID分别为A `3779677`、B `3779683`；日志均有 `SCHEMA_PROBE_COLLECT=PASS` 和 `EXIT_CODE=0`。采集PASS只表示成功读取并留证，不是完整输入判据PASS。

## 实测字段与数值

| 样本step | 阶段 | 共同字段 | 数值失配 | dtype差异 | 当前侧额外字段 |
|---|---|---:|---:|---|---|
| 0 | raw / transformed | 15 / 10 | 0 / 0 | static_image_emb：A float64、B bfloat16；static_pos_emb：A float64、B float32 | 四motion键，全部None |
| 31 | raw / transformed | 15 / 10 | 0 / 0 | 无 | 四motion键，全部None |
| 32 | raw / transformed | 15 / 10 | 0 / 0 | 无 | 四motion键，全部None |

完整 `raw_all/transformed_all` 同时保留上游四个 `recur_* = None`；它们按计划已明确的历史废弃键规则单独留证，不进入共同模型输入比较。四个motion键未排除，未补到A侧，未将严格总判改为通过。

精确数值摘要使用schema 2的无损实数编码，仍保留原始dtype、shape、字节数和SHA；训练侧标量/状态对拍工具未改变。这轮对应的输入工具与训练判定器测试为123项通过、12.10秒，含独立数学oracle和精度损失反例。

## 原始记录与下一步

原始JSON为 [records/upstream.json](records/upstream.json)、[records/current.json](records/current.json)，比较汇总为 [records/comparison.json](records/comparison.json)，两份运行日志为 [records/upstream.summary.log](records/upstream.summary.log) 和 [records/current.summary.log](records/current.summary.log)。只归档摘要与测量，不复制原始图像、特征或权重。

上述结果交用户后，用户明确回复「允许这四键缺失与 None 等价」。仅对 `motion_emb/motion_pos/motion_mask/mem_order` 缺失或严格为None放行，任何非None、其他新增/缺失键或数值差异仍失败，原始记录与真实训练输入不改。

按新授权对同一份原始 `raw_all/transformed_all` 摘要重新判读，三个样本两个阶段均满足获准等价规则，见 [records/comparison_after_decision.json](records/comparison_after_decision.json)。该文件绑定两份原始JSON的SHA，不改写当时的 `comparison.json`；其中 `formal_input_eq=false` 明确这不是正式两库完整验收。相应输入工具与训练判定器128项测试通过、12.32秒，训练bitwise要求不变。

建库按最新恢复授权推进独立前置；正式训练仍须正式库输入、100步训练对拍及测速/保存加载等全部闸门。
