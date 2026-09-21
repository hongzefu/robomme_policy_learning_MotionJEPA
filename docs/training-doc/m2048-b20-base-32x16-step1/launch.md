# m2048-b20-base-32x16-step1 启动记录

本补跑的完整命令、参数和独立输出名已在起跑前提交的[父档案](../m2048-b20-base-32x16/launch.md)中声明：同seed、同配置，只将num_train_steps改为1，以真实末步保存调用记录state_step=1。实际起跑HEAD为29a27c0be9f218ca0cfe4ab2abf1d3372ed308b6，UTC为2026-09-21T03:13:06Z，启动状态干净，实测结束于2026-09-21T03:20:44Z。本文件是对预先声明的补跑拆出单独导航，并非事后伪造启动版本。

GPU4、5，batch8、worker4、FSDP2、seed42，XLA确定性设置与父run相同。完整实际argv、导入路径、数据与归一化摘要见[现场记录](records/launch.actual.json)。补跑不作为新正式训练，不复用或覆盖现有run目录。
