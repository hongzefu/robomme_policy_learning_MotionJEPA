# 2048的100步训练与保存加载验收

V8全部通过。100次真实更新完成，五标量全程有限且mem_enc_norm严格为正；初态与packed20步的初态逐201叶相同。保存现场原数组、实际checkpoint与加载模型之间的比较全部成立，固定noise动作RMS差为0，实际MemoryAttention长度为2048、motion关闭。

源码为主副本clean CAND 0c877c7495dfe5db8b83f033442013c6d6fd8552，05:22:53Z→05:51:05Z，总1692秒。按用户「能并行的尽可能并行 8个gpu你可以用」使用GPU6、7，b8/w4/FSDP2/seed42；环境指纹引用已完成refnpy，只有卡号和store_meta两个明确差异，其余软件、资产、数据源、清单与精度一致。完整初态数值比较没有豁免，见[启动记录](records/launch.actual.json)。

100行五标量、100批输入摘要齐全；状态摘要按原save_interval25取0、26、51、76、100五份，每份201叶。源码原样保留，未把该间隔改成每步落盘。初/末原数组分别为50092740044字节，两份合计100185480088字节（约93.305GiB），逐叶SHA与现场摘要相同。文件名state_step_99按循环号命名，其内部真实state.step=100；记录与位置见[数组清单](records/arrays_inventory.json)，大bin继续保存在v1-store，未进Git。

真实checkpoint目录为v1-store/train-runs/m2048-train100/mme_vla_suite/m2048-train100/999。999是既有bench保存编号，不是更新次数；[保存元数据](records/final_checkpoint.json)确认state_step100、param_kind=ema、61个参数叶。

| 验收 | 实测 |
|---|---|
| TIC L0 | 归一化8数组相同、库provenance相同、关闭态不读motion库、61叶无缺失或额外项，全部通过 |
| 保存dtype | 23个image叶bf16，38个可训练叶f32；只观察既有保存行为 |
| bf16加载一致性 | 原始EMA转bf16后与加载61叶逐字节相同，mismatches=0 |
| 区别于初态 | 十个记忆叶在f32及加载bf16精度下均改变，10/10 |
| 动作比较 | 固定noise、10个采样步，rms_diff=0.0≤6.8e-5；A/A RMS同为0 |
| 实际形制 | budget2048、token_per_image64、memory_token_dim1024、真实前向mem_len2048、motion_enabled0 |

TIC L0沿用原check_config_provenance.py，本轮未修改该工具；完整[判定日志](records/driver.summary.log)、[L0结果](records/l0.json)及[逐叶变化和动作结果](records/ckpt_checks.json)已归档。加载校验内部耗时379.464秒，已由同一detached会话留档；驱动EXIT_CODE=0，GPU6、7释放。

用户要求「保持原计划，完整取证（推荐）」。本项验证更新、保存与加载，不宣称策略成功率改善，也不支持恢复优化器状态继续训练。吞吐与80k时间另由独占八卡1000步测速给出；正式起跑按最新条件授权和V10处理。
