# modulation 关闭态追溯验证通过

两档的 A1/A2 自复现及 A1/B 追溯比较全部通过。参考源码为 `07702f0076e640724e9516e911983d7810b423d3`，候选源码为 `63858f5fa39db26b3c587e9bf7df433b5be5bed4`。每组 61 个初态参数叶子、三类固定 batch 的输入摘要、loss hex 和每类全部 38 个可训练梯度叶子逐位相同；没有放宽容差。

```text
COMPARE profile=c32 pair=A1/A2
INIT_EQ=PASS leaves=61 mismatches=0
GRAD_EQ=PASS kinds=3 leaves=38 mismatches=0
COMPARE profile=c32 pair=A1/B
INIT_EQ=PASS leaves=61 mismatches=0
GRAD_EQ=PASS kinds=3 leaves=38 mismatches=0
COMPARE profile=c8 pair=A1/A2
INIT_EQ=PASS leaves=61 mismatches=0
GRAD_EQ=PASS kinds=3 leaves=38 mismatches=0
COMPARE profile=c8 pair=A1/B
INIT_EQ=PASS leaves=61 mismatches=0
GRAD_EQ=PASS kinds=3 leaves=38 mismatches=0
MODUL_RETRO=PASS profiles=2 comparisons=4 init_leaves=61 grad_leaves=38
EXIT_CODE=0
```

| 布局 | mixed1 loss hex | allshort loss hex | allfull loss hex |
|---|---|---|---|
| c32：32 帧 4×4 | `0x1.c168f40000000p-4` | `0x1.891e8e0000000p-4` | `0x1.0159440000000p-3` |
| c8：8 帧 8×8 | `0x1.0c91bc0000000p-3` | `0x1.80d27e0000000p-4` | `0x1.041d4a0000000p-3` |

各行内部 A1、A2、B 三侧完全相同，不跨布局比较数值。两档在模型侧走同一段代码，用不同真实输入增加数值覆盖；本结论限定于记录的固定输入与初态，不替代 Dataset 装配、采样语义或完整训练轨迹验收。后两者对应本轮 F2/F3 及既有 t8 记录，见 [launch.md](launch.md)。

六次运行均使用物理 GPU 4,5、batch 8、fsdp 2、seed 42、同一主副本 `.venv`，并分别使用独立 JAX 缓存；XLA 为确定性档。初态、梯度和 loss 全部有限。软件版本、uv.lock SHA、GPU 驱动、可见卡号、源码实际导入路径和 YAML SHA 均在各 `grad_summary.json` 中；比较器还逐项核对环境相同。历史树只添加同字节 8×8 YAML，没有修改 `src/` 代码。

运行从 `2026-09-15T05:03:11Z` 至 `05:38:52Z`，墙钟 35 分 41 秒，退出 0。完整日志见 [records/train.log](records/train.log)，四组复核原文见 [records/comparisons.log](records/comparisons.log)，六侧原始初态/梯度摘要保存在对应子目录，复制后逐字节核对；没有生成梯度数组或训练 checkpoint。

`tr-m8-modul-retro` 正常结束并自动消失，没有执行 tmux 清理命令。归档后移除本轮复制的 YAML，以普通 `git worktree remove` 删除 `s2-base`，其余五个 worktree 保持原样；本轮 bench 与六个专用缓存目录已清理。现成 fixture 与预训练权重均保留。
