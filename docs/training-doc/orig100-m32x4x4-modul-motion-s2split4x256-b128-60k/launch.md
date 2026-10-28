# 官方16×100 modulation训练启动档案

当前状态：已获开工授权，尚未启动；由[context串行入口](../orig100-ctx32x4x4-motion-s2split4x256-b128-60k/launch.md)接续。正式起跑时补训练Beta SHA、实际时间、设备和句柄。

固定run为`orig100-m32x4x4-modul-motion-s2split4x256-b128-60k`。除`integration_type=modulation`及`memory_token_dim=1024`外，与前序context配置完全一致：官方16×100同库、ep105 EMA、32帧4×4、motion每窗4×256、总896 token、8卡FSDP8、全局批量128、16 workers、seed42、60k步、同一学习率日程与数据顺序。

前序必须真实退出0、最终记录有限且59999保存，才由`scripts/training/run_orig16_ctx_then_modul_60k.sh`立即接续；不等待前序上传/G12。此run日志为`v1-store/logs/orig100-m32x4x4-modul-motion-s2split4x256-b128-60k.log`，checkpoint根为`v1-store/train-runs/mme_vla_suite_orig16_b128_60k/orig100-m32x4x4-modul-motion-s2split4x256-b128-60k`。

公开bucket为`HongzeFu/robomme-vla-orig100-m32x4x4-modul-motion-s2split-60k-v1`，5000至55000及59999共12份checkpoint。训练结束后进行两条G12严格恢复及最终匿名读回；不做评估。数据、代码与完整启动命令见前序档案及[建库档案](../../dataset-build-doc/16task-orig100-1600ep-9e109419/launch.md)。

用户原话：「开工 有问题问用户 尽可能不要阻塞 所有subagent全部astra high」「登记三项任务（推荐）」。此档案仅登记已批准运行，不声明训练或上传已完成；实际结果、耗时和日志随后归档。
