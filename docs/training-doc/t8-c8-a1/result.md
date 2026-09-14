# t8-c8-a1 训练结果

本条完成1000次参数更新，退出码0；对应C8组的A1/A2自重复及A1/B候选比较全部逐位一致。完整判定见[本组gate记录](../t8-training/records/gate-c8.log)。本页记录1000步轨迹验收，全梯度与checkpoint推理在各自档案补充。

## 版本、数据与用户决定

实际启动HEAD为 `99faacb1319adfc63c0cf9a15187e24c34e38fd1`，起跑工作区干净，四个源码导入路径已验证。REF为 `99faacb1319adfc63c0cf9a15187e24c34e38fd1`，CAND为 `c08ec2060a544af1869c1e24f755e536150569ca`；候选启动HEAD只含其后的档案提交。完整命令和启动覆盖见[launch.md](launch.md)，真实argv与来源见[run_meta.json](records/run_meta.json)。

数据为400ep、101066个执行样本，本条实现为 `refnpy`，输入路径 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-motion-400ep/source`，配置 `perceptual-framesamp-context-8frame-8x8.yaml`。batch8、worker4、seed42、fsdp2、log/save interval1均为启动覆盖。norm_stats显式使用 `/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json`，文件SHA `750a8e9bd6e1e5a3cf5c294864c44564153309ef92492eb083fa361096d470d2`；实际loader收到的8个统计数组与文件解析摘要逐项相同。

用户要求“开始实现 有问题越早问用户越好 一口气全做完！”。四GPU限制、双重norm_stats检查、checkpoint配置驱动前缀以及仅-s100关闭状态摘要的决定，见[用户决定](../t8-training/decisions.md)。本条正式轨迹保留全部11份完整状态摘要。

## 数值与完整性

| 项 | 实测 |
|---|---|
| 更新次数与索引 | 1000次更新使用8000个样本；含预取的索引记录8072项 |
| loss | 0.547706306 → 0.0318128765 |
| 参数范数 | 1803.091919 → 1803.096436 |
| 完整TrainState | 177叶；state_step=0,2,3,25,50,100,200,400,600,800,1000 |
| 输入取证 | 11份batch摘要，显式记录键集、None状态及sample_indices |
| 有限性与活性 | 五项标量及全部状态叶有限；loss降低、参数变化、记忆梯度非零 |
| 五标量hex文件SHA | `7ca21f0881f875a3e0afda2ab3330dab9e76d61af829b8580367bb1d4e4c6030` |
| 最终完整状态摘要 | `3381cd87cd671b9cd5897d0b92729c5f507820da000170fbf6d968e11957e371` |

比较只在同profile内部进行，32帧与8帧、motion开与关之间不比较数值。前8000个采样索引、全部1000步五标量hex、11个时刻完整状态逐叶SHA及输入摘要均零差异。不能用本条有限步数替代已完成的库层全量verify。

## 硬件、耗时与资源

AWS A100-SXM4-80GB，物理GPU `4,5`，存储为 `/dev/md0` XFS本地NVMe RAID。开始 `2026-09-14T08:02:14Z`，结束 `2026-09-14T09:06:52Z`，墙钟1小时4分38秒；11份状态摘要累计2222.5秒。

稳态统计取日志步50–999，排除摘要步及下一步，共938个步时；GPU每500ms采样。相同NVML读数不视为更高时间分辨率，性能按共享主机环境记录。本轮运行中曾观测到GPU0–3有其他计算，见[环境观测](../t8-training/records/host-observation-first-a1.json)，该记录不是启动时刻快照。

| 项 | 实测 |
|---|---:|
| 稳态步时均值 | 1.5550秒 |
| 吞吐 | 5.1448 samples/s |
| GPU util均值／0%采样占比 | 96.39%／0.00% |
| 慢步分层 | n=0，均值无样本 |
| 其他步分层 | n=5830，均值96.39% |
| 整卡显存峰值（含预分配） | GPU4: 78639 MiB；GPU5: 78639 MiB |

慢步阈值为该run稳态步时中位数的1.5倍（2.3310秒），中位数只用于分层。完整计算方法和排除步列表保存在[性能记录](records/performance_stratified.json)，原始采样保留于records；未用单一利用率断言无瓶颈。

## 产物与后续

本条只记录完整参数、优化器与EMA摘要，没有额外保存大权重。

records保存全部逐步指标、hex投影、11份状态/输入摘要、完整索引、环境与run元数据、原始GPU采样、性能统计及BASELINE_MANIFEST。复制后逐文件SHA与源记录一致。清洗日志保留1000条Step行、所有状态摘要与结束记录，剔除tqdm中间态和行尾空白。后续仍需完成计划中的全梯度与推理闭环。
