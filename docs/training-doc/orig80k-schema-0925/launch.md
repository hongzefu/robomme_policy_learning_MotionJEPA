# 原版80k三样本输入字段取证

用户对主机dtype的原话为「允许主机 dtype 不同，但要求数值一致且训练标量/状态逐位一致」；对motion四键的原话为「先保留严格判据并取证，结果出来后再决定」。用户后续已恢复建库授权，但本次取证不建库、不训练，仅读取既有已验证库中的三个样本，不能作为新两个正式库的全量验收。

## 版本与真实输入

取证工具为 `scripts/training/tests/probe_orig80k_schema.py`，复用 `check_orig80k_inputs.py` 的schema 2、实际侧loader与精确数值摘要。先提交工具与本档案，再从clean HEAD运行；A固定 `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`，B为本次工具提交，实际完整SHA记录在JSON的起止provenance中。训练侧对拍器不改，仍严格比较标量和状态。

输入库为本机 `v1-store/datasets/4task-motion-40ep/`，40集、13756帧、11530执行样本，4x4 packed状态为verified，manifest SHA为 `d7cfb137b6ba01c42894e2d6d421a8c3f87dc1afeef1ac650563609bd7501d05`。样本索引固定 `0/31/32`，均为 `ButtonUnmask` raw episode 0，对应step `0/31/32`，覆盖短历史与32帧边界。只构造实际loader并读取其原始Dataset与transforms，不启动worker迭代或模型前向。

两侧使用同一原版norm_stats，父目录为 `v1-store/train-assets/mme_vla_suite`，asset-id为robomme，文件SHA为 `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`。tokenizer和模型资产根均显式指向主仓库 `v1-store/models`；本次不读取大模型权重，也不重算SigLIP特征。

## 命令与输出

A从 `v1-store/worktrees/orig-ecf086c` 运行其独立uv管理 `.venv/bin/python`，B从主仓库 `uv run --no-sync python` 执行同一主仓库工具。两侧均清除 `PYTHONPATH/PYTHONHOME`，设 `CUDA_VISIBLE_DEVICES=''`、`JAX_PLATFORMS=cpu`、`OPENPI_DATA_HOME` 为主副本models，并显式设置本仓库uv/HF/XDG缓存。工具参数为 `--side upstream|current --expect-root <该侧根> --expect-head <完整SHA> --out <输出JSON>`，B另传 `--forbid-root <A根>`。

唯一输出分别为 `v1-store/bench/orig80k-schema-0925/upstream.json` 和 `current.json`，不存在才可写入；预计不超过5分钟。若异常超时则保留真实启动状态与输出，按长诊断留档，不能改称另一个版本起跑。

## 判据与归档

每个样本保留raw/transformed的全部字段、原始dtype/shape/字节数/SHA及精确数值摘要，不改变输入。比较数值使用无损实数编码，正负零仍区分，非有限值拒绝；motion四键缺失与None仍视为schema差异。真实结果应分别报告共同字段数值结论和完整字段集合结论，禁止因共同部分相同而把总判写成通过。采集成功只表示读取和记录完成。

结果与JSON归档在本目录；原始pkl/npy/packed不复制。实际执行命令、完整版本、实测耗时和待用户裁决项在结果文档回填。
