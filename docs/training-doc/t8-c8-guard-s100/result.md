# 守卫放宽的百步梯度等价验证通过

F2 与 F3 均通过。`FrameSampDataset.__init__` 的成对白名单放宽，在本次覆盖范围内不改变数据交付与既有 context 8×8 的训练数值。候选功能提交 `81ba002`；实际起跑 HEAD 为 `75fb1d3a425b84b669efb9009d112ee606ad8eba`，`start_status` 为空。完整命令、前后链路图及一致性边界见 [launch.md](launch.md)。

第一块在 1600 集新库比较 modulation-8x8 与 context-8x8 的 3454 个样本、19 个原始键，逐键 dtype、shape、原始字节及字符串/None 全等，耗时 77.913 秒。modulation 两档构造通过，expert 与错配被拒绝。该比较限定在当前源码下两份配置，不声称两种模型数值相同。

第二块使用 GPU 4,5、batch 8、worker 4、fsdp 2、seed 42 与 XLA 确定性档，在 400 集旧库复跑 `t8-c8-b` 前 100 步。基线起跑 HEAD 为 `38f0db46db19a3b645613e04fba3f6e635b47054`；本次基线环境指纹和 10 个固化产物均通过严格校验。

```text
BASELINE_ENV=PASS
SCALARS steps=100 keys=5 hex_mismatch_steps=0 first_mismatch_step=None
STATE_DIGEST rows=6 mismatch=0
BATCH_DIGEST rows=6 mismatch=0
BATCH_DIGEST_CANONICAL rows=6 mismatch=0
CANON_CHECK=PASS steps=6
INDEX_SEQ=PASS n=872（共同前缀逐个一致, steps≈100）
INDEX_TRAIN=PASS n=800 recorded_n=872
DET_CHECK=PASS tier=adhoc steps=100 scalar_hex_diff=0 state_digest_diff=0 batch_digest_diff=0
GUARD_GRAD_100=PASS scalars_steps=100 index_n=800 batch_digest_rows=6 state_digest_rows=6
EXIT_CODE=0
```

索引数量说明：`compare_baseline.py::compare_index_seq` 实际比较两侧记录的整个共同前缀，并不裁到 `steps × batch_size`，因此打印 872，包含预取的 72 个额外索引。另由 [finish_check.py](records/finish_check.py) 独立断言前 800 个训练索引完全相同；两种口径分别保留，未修改比较器或放宽判据。五标量 TSV 恰好 101 行，与基线前 101 行 `cmp` 一致。

完整状态摘要取 step `{0,1,2,24,49,99}`，每份 177 叶，覆盖 params、EMA、optimizer 状态与 step。运行于 `2026-09-15T04:25:12Z` 至 `04:59:58Z`，墙钟 34 分 46 秒；六次摘要分别耗时 504.352、290.825、215.044、245.768、295.266、278.995 秒。该 run 用于数值等价，不以其被取证开销支配的步时报告训练吞吐。存储为 AWS 本地 NVMe RAID（`/dev/md0`），不与环境 A 混比。

记录见 [records/train.log](records/train.log) 和七份原始数值/环境文件，复制后与源文件逐字节核对。`m8-guard` 正常结束并自动消失，没有执行 tmux 清理命令。完成归档后仅清理本 run 的 bench、checkpoint 根和专用 JAX 缓存，基线 `t8-c8-b` 原始产物保留。后续仍须完成 modulation A/A、自历史锚点的梯度追溯、正式配置 smoke 和起跑隔离；本结果不替代那些验证。
