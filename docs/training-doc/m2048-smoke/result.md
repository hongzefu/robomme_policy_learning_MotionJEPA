# 八卡2048生产容量验收结果

V9全部通过。完整生产模型在8×A100-SXM4-80GB上以b128/w16/FSDP8/seed42完成20步、五标量全有限并真实保存第20次更新；加载后的61叶形状全部匹配，十个记忆叶齐全，没有OOM。

主副本clean HEAD为3e5f365849e7fca58fe6d8d4018c0f91615a9604。driver于06:37:51Z→06:43:05Z运行，314秒；训练Python进程计时295.437秒。CPU容量校验06:44:20Z→06:44:41Z，内部19.420秒，两份日志均EXIT_CODE=0。起跑26项preflight全通过，实际CLI配置及来源均绑定；会话和精确PID见[实际记录](records/launch.actual.json)。八卡与采样器均已释放。

| 资源/行为 | 实测 |
|---|---|
| 实际loader | batch128、worker16、prefetch_factor2、persistent_workers=true、pin_memory=false |
| 共享内存峰值 | 63868461056 B / 602265600000 B，10.6047%，低于70% |
| 主进程RSS峰值 | 70334951424 B；不与worker RSS相加冒充物理独占内存 |
| 主机采样 | 553条；原始逐进程RSS、内存与IO记录保留 |
| checkpoint | 实际step19目录、内部state_step20，完整61叶、十记忆叶 |

[GPU显存峰值](records/gpu_memory_peaks.json)来自500ms原始CSV，包含XLA 0.95预分配，不能当作模型活跃显存。JIT初始化期间出现cumsum常量折叠慢操作提示，随后编译和20步均完成，不是运行失败；未调整XLA参数回避提示。此短窗包含初始化，不据此外推80k或判定稳态GPU利用率。

完整[训练日志](records/driver.summary.log)、[容量判定](records/capacity.summary.log)、[结果JSON](records/capacity.json)、指标/主机/GPU采样与checkpoint元数据已归档，压缩回读SHA见[清单](records/archive_manifest.json)。按原计划，核实归属、无外链、23个文件共11879611583字节后，仅清理本轮临时smoke checkpoint根；原始及Git内记录保留，见[清理记录](records/cleanup.json)。没有删除其他run、验证100步checkpoint或原数组。

用户要求「保持原计划，完整取证（推荐）」并允许使用八卡；正式参数原样，仅本次启动覆盖20步/log1/关闭W&B。下一步是独占八卡1000步测速，报告READY且稳态总体util>50%、无其他问题时按既有用户条件授权执行V10并启动正式80k。
