# V8 开启态两轮100步逐位一致，四个motion参数叶均更新

用户要求按0916计划实施并“有问题尽早问用户”。本档验证同一开启态实现的可复现性，使用新库、完整生产VLM与动作专家，两轮100步、batch8、fsdp2、workers4、seed42，GPU4,5；不使用V5的gemma_150m替身。两轮同源HEAD为 `782696c231aace21c20200638ae302a5a1c7f277`，起止porcelain为空，exp_name同为mv2-aa、共用同一JAX缓存，记录标签分别为mv2-aa-a和mv2-aa-b。

2026-09-18 03:47:27→04:27:56 UTC，共2429秒（40分29秒），两轮及总入口均EXIT_CODE=0。环境B，A100-SXM4-80GB，存储AWS本地NVMe RAID `/dev/md0` XFS。驱动、四处新库norm路径覆盖和完整命令见 [launch.md](launch.md)。两轮起跑均核norm、motion元数据和整表SHA；第二轮前的环境/基线完整性检查通过。真实loader的norm数组与固定文件相等，文件SHA为 `856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173`。

## 完整判定

```text
SCALARS steps=100 keys=5 hex_mismatch_steps=0 first_mismatch_step=None
STATE_DIGEST rows=5 mismatch=0
BATCH_DIGEST rows=7 mismatch=0
BATCH_DIGEST_CANONICAL rows=7 mismatch=0
CANON_CHECK=PASS steps=7
INDEX_SEQ=PASS n=872（共同前缀逐个一致, steps≈100）
INDEX_TRAIN=PASS n=800 recorded_n=872
MOTION_PARAMS_UPDATED=PASS n=4 first_step=0 last_step=99
AA_100=PASS scalars_steps=100 index_n=800 batch_digest_rows=7 state_digest_rows=5
```

五项标量为loss、grad_norm、llm_grad_norm、mem_enc_norm、param_norm，每步十六进制值完全相等且有限。完整TrainState摘要在0/25/50/75/99步各覆盖217个叶，包含params、EMA、Adam状态与step；第0份是初始化状态，最后一份state_step=100。原始与canonical输入摘要在0/1/2/25/50/75/99步均一致，motion_emb、motion_pos、motion_mask、mem_order四键非null。前800个实际训练索引一致，包含预取的共同872索引也逐项一致。motion_pos_proj和motion_encoder_static的kernel/bias四叶从初态到第99步全部改变。

## 结论边界与留档

本结果证明当前实现开启态A/A可复现，以及motion参数进入了优化过程；不证明与关闭态或context实现等价，也不证明策略成功率收益。八卡batch128实际生产档位由后续20步smoke覆盖，正式吞吐由80k run前300步单独测量。

本档包含确定性设置和完整状态拷回/哈希，耗时不用于生产吞吐结论。例如第一轮初态摘要311.484秒，后四次各约189–191秒。驱动已按预设清理本轮没有checkpoint的run空壳，全部记录及共享编译缓存保留。完整指标、状态与输入摘要、环境指纹、实际norm和索引序列分别位于 [records/a/](records/a/) 与 [records/b/](records/b/)，完整判定与退出码见 [driver.summary.log](records/driver.summary.log)。V8通过后按既定顺序推进V-online三档。
