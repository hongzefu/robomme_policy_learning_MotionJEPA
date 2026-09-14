# C8推理启动记录

使用`t8-c8-b`的999 EMA checkpoint；完整命令、资源、判据与会话清单见[共同启动口径](../t8-infer/launch.md)。本模型motion关闭，评估从checkpoint配置推导1088前缀长度。关0–6与48集闭环将各自从clean HEAD启动，实际提交、时间和结果记入日志并回写result.md。
