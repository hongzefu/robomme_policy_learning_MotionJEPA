# 400ep 的 8×8 特征库结果

全部库层检查通过。起跑HEAD为 `099578174e698f5b437d2a26612ba16006f47112`，工作区干净，候选代码为 `c08ec2060a544af1869c1e24f755e536150569ca`。纯CPU、48进程、AWS本地NVMe RAID `/dev/md0`；pack于2026-09-14 03:51:40–03:51:58 UTC完成，内部计时17秒，verify于03:52:04–03:52:09完成，独立检查于03:52:10完成。两份日志均 `EXIT_CODE=0`。源数据已经过多轮读取，耗时不作为冷缓存性能结论。

```text
PACK_DONE=1
VERIFY_PACK=PASS scanned=123044 mismatches=0
IMAGE_NPY_SPOT=PASS frames=512 mismatches=0
MOTION_POS_XGRID=PASS t=586 npy=64 mismatches=0
PART_BOUNDARY=PASS parts=31 bytes_ratio=4 state_sha_exact=1
```

库状态为verified、scope为full，pack锁已释放。image共32255246336 B、pos 115212288 B、state 3937408 B、行摘要1968704 B，完全符合预期；31个part的起点、行数、episode边界与旧库相同，image字节四倍，state摘要一致。旧库、motion库与source未修改。

`records/`保存最终store_meta、分片核对结果及pack和verify/report/xgrid清洗日志。用户“一口气全做完”的指令继续执行，本库四profile完整装配验收另见 [t8-fixture结果](../../training-doc/t8-fixture/result.md)，真实训练仍独立验收。
