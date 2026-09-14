# 40ep 的 8×8 特征库结果

库层全部通过。起跑提交为 `d61c47ba4e9d6c53c8079ada6a8c78e83140e87f`，工作区干净，纯 CPU、48 个进程，源与输出均为 AWS `/dev/md0` 本地 NVMe RAID。pack 于 2026-09-14 03:26:38–03:26:42 UTC 完成，脚本内部计时 3 秒；verify 内部计时 1 秒，随后 report 与独立交叉检查于 03:26:54 全部完成。两份日志均 `EXIT_CODE=0`。该耗时受已缓存源数据影响，不作为冷缓存性能结论。

```text
PACK_DONE=1
VERIFY_PACK=PASS scanned=13756 mismatches=0
IMAGE_NPY_SPOT=PASS frames=512 mismatches=0
MOTION_POS_XGRID=PASS t=586 npy=64 mismatches=0
PART_BOUNDARY=PASS parts=22 bytes_ratio=4 state_sha_exact=1
```

新库为 `status=verified`、全量 scope，锁已释放。image 共 3,606,052,864 B，pos 115,212,288 B，state 440,192 B，行摘要 220,096 B，全部符合启动前字节账。旧库和源库未修改；state 表 SHA 与旧库完全一致，每个 part 的起点、行数和 episode 边界完全一致，image 字节数恰为旧库四倍。

`records/` 留存最终 store_meta、pack 与 verify/report/xgrid 的清洗日志及分片核对结果。用户“一口气全做完”的指令继续执行；本次只完成库层，后续四种配置完整 fixture、worker 矩阵、1000 步训练和推理仍需分别通过。
