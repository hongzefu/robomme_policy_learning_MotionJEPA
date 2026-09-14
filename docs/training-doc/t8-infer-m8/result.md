# M8 推理验收结果（分阶段记录）

目前完成共用帧编码路径的真实8×8核验；M8 checkpoint尚未训练，七关中依赖该checkpoint的部分与48集闭环未执行。

## 真实8×8位置表、编码与装配

于2026-09-14 03:36:05–03:38:55 UTC运行，耗时2分50秒，启动HEAD为 `c8aba47eef6c31266df985ff5c7a675418522df1`，工作区干净；仅使用物理GPU7。使用真实H5帧和真实SigLIP，同一个jitted编码器供三方使用，无stub。52个实际输入帧覆盖17个边界时刻。

```text
POS_TABLE=PASS rows=4096 a==b=True a==c=True
ENC_LAYER_8X8=PASS steps=17 keys=3 mismatch=0
ASSEMBLY=PASS steps=17 mismatch=0
OOB_PROBE=PASS a=empty-slice-detected b=empty-slice-detected c=raise
ONLINE_MEM=PASS base=732fae3b
EXIT_CODE=0
```

三类特征为8×8 image、8×8 pos与state。4096行位置表全量逐位一致，17个时刻的编码和装配均零差异，越界由候选显式拒绝。这个检查同时覆盖C8/M8共用的帧路径，后续无需重复；它没有替代真实checkpoint上的动作、motion或闭环验收。

用户要求“一口气全做完”；当前仍继续后续数据、训练与推理阶段。`records/online_memory/`保存逐步摘要、判定行及清洗日志。
