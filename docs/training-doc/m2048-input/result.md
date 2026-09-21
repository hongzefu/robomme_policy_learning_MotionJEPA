# 新2048档输入与装配验收结果

全部输入与CPU功能门通过，CAND `0c877c7495dfe5db8b83f033442013c6d6fd8552`，04:08:00Z→04:10:08Z，EXIT_CODE=0。

- YAML解析值仅budget从512变2048；新组合接受，3个类型、4个组合反例拒绝；四键形状与mask广播反例在普通及-O下均拒绝。
- 独立选帧12例；四任务400个真实样本（各50集首/末，共200接缝与200尾部）逐键对拍全等。
- n=1..32的pad与在线right_padding_token_emb逐字节相同；在线5时刻、256→64池化及64-token直通两分支均通过，位置表805306368字节。
- 短/混合/满三类合成batch经过spawn与真实共享内存交付后相同；image bf16、pos f32、归一化state f64、mask bool保持，JAX仅沿用state f64→f32。
- 独立numpy attention oracle六组通过预设atol=1e-6、rtol=1e-5，并通过单key、mask外垃圾、非退化RoPE敏感性与统一query位置检验。
- worker0与spawn4的真实4批、32索引逐位相同；每批120961552字节。此处等待时间不作为性能结论。

各项JSON与[判定日志](records/driver.summary.log)已保存，[归档清单](records/archive_manifest.json)核对原字节。真实1600ep恒满32帧，短历史覆盖来自明确的合成协议；本档不宣称使用了真实短历史样本，也不改变模型对static_state的既有消费范围。

用户明确选择「保持原计划，完整取证（推荐）」。环境为AWS 8×A100-SXM4-80GB，底层 `/dev/md0` XFS本地NVMe RAID；依赖锁未改。执行口径见[实施计划](../../../0920-32frame-8x8-modul-2048-plan.md)，实际版本、时刻与退出记录见[运行记录](records/launch.actual.json)。
