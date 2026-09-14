# 8×8 数据装配验收结果（分阶段记录）

40ep与400ep的完整装配验收全部通过，两种worker矩阵及独立边界检查也通过；1000步训练尚未开始，当前结论仅覆盖输入交付。

## 40ep完整输入与独立边界

取证于2026-09-14 03:27:36–03:50:44 UTC运行，23分08秒；worker矩阵同期运行至03:44:48，17分12秒。两个会话均从clean `304e2225a3d82530edb8b7aef43bc215acaf407e` 启动，全部纯CPU，最终 `EXIT_CODE=0`。运行中主树仅增加真实8×8在线池化的启动与结果档案，候选源码始终为 `c08ec2060a544af1869c1e24f755e536150569ca`，每份dump记录实际HEAD。参考侧始终为 `99faacb1319adfc63c0cf9a15187e24c34e38fd1`，显式PYTHONPATH指向REF/src。

| 配置 | 每侧样本数 | 每档定点配额 | 每侧batch数 | 结果 |
|---|---:|---:|---:|---|
| C32 | 1200 | 20 | 200 | 全部raw、transformed键逐位相同 |
| M32 | 1200 | 20 | 200 | 全部raw、transformed键逐位相同 |
| C8 | 1220 | 20 | 200 | 全部raw、transformed键逐位相同 |
| M8 | 1220 | 20 | 200 | 全部raw、transformed键逐位相同 |

四组分别核对全部40集、11530个执行样本身份与13756个合法帧索引。C32/M32边界分档存在重复时刻，去重后10档；C8/M8为11档，均另取1000个随机样本。两侧真实归一化均使用用户指定400ep norm_stats，文件SHA为 `750a8e9bd6e1e5a3cf5c294864c44564153309ef92492eb083fa361096d470d2`。空值键也显式参与比较，关闭motion时四键全部为None。

逐组均输出SOURCE_IDENTITY、FRAME_INDEX_EXACT、SAMPLE_RAW_EXACT和BATCH_RAW_EXACT的PASS行，完整原文见清洗日志。独立整数公式另输出：

```text
HAND_CALC_8FRAME=PASS samples=27 mismatches=0 profile=c8
HAND_CALC_8FRAME=PASS samples=27 mismatches=0 profile=m8
```

## 40ep两种读取方式的完整worker矩阵

每档固定batch 8、2个完整epoch，共2882个batch。8档均为fd 5→5、leak=0，两种实现分别输出 `MATRIX=PASS workers=0,1,4,16 epochs=2 batches_per_run=2882`。

| 实现 | w0秒 | w1秒 | w4秒 | w16秒 |
|---|---:|---:|---:|---:|
| packed | 59.3 | 181.5 | 138.6 | 128.1 |
| refnpy | 88.5 | 155.7 | 138.1 | 120.4 |

这些耗时来自与fixture并行的生命周期测试，只记录完成成本，不用于两种实现的性能结论。存储均为AWS本地NVMe RAID `/dev/md0`。

## 归档与范围

`records/40ep/`按profile与参考/候选分别保存DUMP_MANIFEST、identity、fixture_plan，以及无损gzip压缩的逐样本和逐batch摘要。解压后的SHA与原DUMP_MANIFEST一致。数组本体仍在 `v1-store/fixtures/8x8/40ep/`，不提交GB级数组；两份清洗日志保留所有比较、手算、矩阵与退出判定行。

用户要求“一口气全做完”，并已确认双重norm_stats检查与checkpoint配置驱动评估长度。40ep开发覆盖之外的400ep结果如下；后续继续真实训练验收。

## 400ep完整输入与独立边界

四个独立CPU会话于2026-09-14 03:52:26 UTC从clean `099578174e698f5b437d2a26612ba16006f47112` 同时启动，最后一组于04:06:07结束，总墙钟13分41秒，全部EXIT_CODE=0。运行期间只增加12条训练启动档案；四份候选dump实际HEAD均为 `df7e5307c0fef8266b5a1d35f3ce636d68778d18`，源码仍为固定CAND。参考侧均为REF，并显式加载REF/src。

| 配置 | 每侧样本 | 每档配额 | 每侧batch | 总耗时 | 结果 |
|---|---:|---:|---:|---:|---|
| C32 | 3000 | 200 | 200 | 12分06秒 | 全部逐位一致 |
| M32 | 3000 | 200 | 200 | 13分02秒 | 全部逐位一致 |
| C8 | 3200 | 200 | 200 | 12分45秒 | 全部逐位一致 |
| M8 | 3200 | 200 | 200 | 13分41秒 | 全部逐位一致 |

每组完整核对400集、101066执行样本身份及123044个合法帧索引；每组SOURCE_IDENTITY、FRAME_INDEX_EXACT、SAMPLE_RAW_EXACT、BATCH_RAW_EXACT均PASS，原文见各日志。独立手算于04:06:19–04:06:25在clean候选HEAD上运行：

```text
HAND_CALC_8FRAME=PASS samples=27 mismatches=0 profile=c8
HAND_CALC_8FRAME=PASS samples=27 mismatches=0 profile=m8
EXIT_CODE=0
```

错配拒绝沿用已完成的阶段2真实测试：`REJECT_INPUTS=PASS cases=29`，29 passed、15.51秒；完整命令与结果由 `236765fdb7b35f8fc96fb7d71133870b70b76271` 的commit body固定，覆盖两档库、目录/配置/字节账错配及写前拒绝，未重复运行。

另外以已取证C8真实数组直接经过 `jax.make_array_from_process_local_data` 核对交付dtype。`jax_enable_x64=False` 时，batch8的image仍为bf16/16777216 B，pos仍为f32/12582912 B，mask仍为bool/4096 B；static_state在host为f64/262144 B，进入JAX为f32/131072 B。这是两侧共用的既有转换，未修改精度。该轻量检查在CPU后端执行，GPU实训仍由第二块验证。

`records/400ep/`采用与40ep相同的摘要归档方式，解压SHA与原DUMP_MANIFEST一致；原始数组保留在v1-store。至此第一块的两库全量verify、独立xgrid、四profile完整fixture、worker矩阵、29项拒绝和两库手算均通过，下一步执行四组100步排错及12条1000更新轨迹。
