# 2048功能门正式重测结果

V5正式通过，原验证包装问题已闭合。源码为clean提交 `31d4bd0b03cc1031eee04b95a69cd6a8aa6a1222`；GPU2，04:56:02Z→05:00:38Z，驱动276秒、检查器内部271.694秒，EXIT_CODE=0，tmux自然退出且GPU释放。

| 判据 | 实测 |
|---|---|
| 三次A/A | 全部38个可训练梯度叶SHA及loss相同，未排除任何叶 |
| 逐token参与 | image与pos各2048个位置的梯度分别非零 |
| 删token故障 | 每帧只保留1位置时只剩32位置非零，2048位置门拒绝该反例 |
| 帧带扰动 | 32/32高于A/A噪声0，最小delta=0.00010776519775390625 |
| 短历史mask | n=1/8/31；垃圾前后loss和固定noise动作逐字节相同，38/38梯度SHA同 |
| 梯度与负对照 | mask外输入梯度严格零、有效位置非零；向有效位置注入垃圾改变loss |

运行来自本轮独立副本v1-store/workspaces/m2048-tool-fix，独立.venv与三个项目模块导入均经现场核验，见[runtime](records/runtime.json)。生产src树为4a16a51e8fd57b25c60d47836a2ac31250b71b89，与CAND相同；uv.lock未改，全部208个分发版本与主环境相同。pi05_base副本按资产锁全量SHA通过，未复制数据集或创建存储外链。gemma_150m替身与完整动作专家只用于功能门，生产训练与测速仍用完整模型。

实际命令除已提交launch外，增加来源核验并显式unset PYTHONPATH，记录在runtime中；没有改变计算与判据。[完整日志](records/driver.summary.log)、[压缩结果](records/func.json.gz)与[原SHA](records/archive_manifest.json)已归档。原失败及探针另行保留。此项证明位置参与和mask行为；真实训练更新、保存加载与容量/测速仍由各自门负责。

用户明确选择「保持原计划，完整取证（推荐）」。环境为AWS 8×A100-SXM4-80GB，底层 `/dev/md0` XFS本地NVMe RAID；依赖锁未改。执行口径见[实施计划](../../../0920-32frame-8x8-modul-2048-plan.md)，实际版本、时刻与退出记录见[运行记录](records/launch.actual.json)。
