# 2048参考链20步与补跑结果

本侧完整通过，尚待packed侧双侧逐位比较，不能单凭本侧完成宣布V7b通过。主run从clean CAND 0c877c7495dfe5db8b83f033442013c6d6fd8552运行，04:09:38Z→05:13:26Z，3828秒；补跑05:13:26Z→05:21:21Z，475秒。GPU0、1、b8/w4/FSDP2/seed42，均为已授权启动覆盖，驱动EXIT_CODE=0。

20行五标量、20行状态（0及2..20）和201叶均完整有限；主232索引/29批，160个参与更新。补跑初态、首批、首标量与主run逐位相同，合并后状态0..20完整。所有20批mask_sum=16384，只作为本库恒满32帧的形制记录。

十个记忆叶逐叶满足Adam nu改变、参数改变与有限性，见[单侧判定](records/single-side.summary.log)。记录检查器用显式--expected-devices 0,1验证两侧与补跑的卡对；七个缺失、错相位、非有限、同漏叶重算摘要等反例全部拒绝。[完整状态集合](records/complete-states.json)和[原SHA清单](records/archive_manifest.json)已保存。这里没有用全True mask代替位置参与，功能证据见[最终V5](../m2048-func-retry2/result.md)。

用户要求「保持原计划，完整取证（推荐）」「能并行的尽可能并行 8个gpu你可以用」。源NPY来自同一1600ep库，norm_stats和依赖不变；AWS 8×A100、/dev/md0 XFS本地NVMe RAID。实际GPU调整、提交、时刻和退出证据见[运行记录](records/launch.actual.json)。完整状态摘要成本不作为生产吞吐或80k ETA。
