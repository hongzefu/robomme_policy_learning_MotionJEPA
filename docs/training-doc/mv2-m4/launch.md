# V5：modulation 模型开启态检查

用户原话「采用兼容权重加载，保留 gemma_150m 替身」。本任务使用真新库和真 modulation 结构验证 mask、梯度、内容与次序；先前 205 秒单 batch 预检通过，本页对应的完整四类样本检查尚未执行。

## 版本与模型

从本页提交后的 clean CHECK_HEAD 启动，起止均核对完整 SHA 与空 porcelain。共用外壳原文见 [V4 启动记录](../mv2-v4/launch.md)，数值实现仍为 f6915f2a09443d48c1bb57e9b5f83e95400704fd。仅占 GPU 4（A100-SXM4-80GB），CUDA_VISIBLE_DEVICES=4，JAX_PLATFORMS=cuda，XLA_FLAGS 为 --xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0，XLA 内存比例 0.95。存储为本地 NVMe RAID /dev/md0 XFS，uv --no-sync，不更改依赖。

模型为 pi05，VLM gemma_150m，动作专家 gemma_300m，真实 18 层、1024 维 memory、4 头×256 头维，帧记忆512、motion budget160。按名称与形状加载 pi05_base 兼容叶，19 个动作专家参数必须完整；随机保留的叶及门参数证据写入报告。正式训练仍用完整生产 VLM。

## 输入与判据

输入为合成零 motion、真库中位、真库最大 k=141 和合成满 budget=160 四类。固定参数、随机流和动作，至少三次确定性探针；逐叶比较全部可训练梯度。padding 的 loss、动作和梯度不得改变，padding 输入梯度必须为零，真实 motion 内容与次序必须影响 loss；motion 全遮时四叶梯度为零，有效输入时非零。行内容和对应时间次序一起置换后 loss 须逐位相同。

nondeterministic_leaves 单列，loss 或 mem_encoder/motion 参数叶不确定即失败；其余叶按计划的确定性纪律记录覆盖和排除集合。行置换判据比较 loss 的逐位一致性。modulation 不同记忆长度的全遮等价明确记 LEN_EQUIV_NA。

## 启动与产物

tmux 全名 mv2-m4。命令为 `bash v1-store/logs/mv2-open-runner.sh m4 <CHECK_HEAD>`，可与 CPU 的 mv2-v4 并行；此后必须等两者结束再启动 V8。

日志 v1-store/logs/mv2-m4.driver.log，结构化报告 v1-store/reports/motion/mv2-m4.json。全部判据须 PASS 且 EXIT_CODE=0；本任务不运行优化器，motion 参数随训练更新由后续 V8 验证。模型质量与策略成功率不由本任务判定。
