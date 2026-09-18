# 终止参考修正后的完整节奏重跑

用户确认「修正验证参考并补用例（推荐）」。修补和失败证据见 [result.md](result.md)。本轮保持35个真实回放、全411种长度的预算扫描、TAU_LONG和1296/1297两侧完整1300步；仅修正通用重放参考的截止点及终止推理次数公式。

从本页及修补提交后的clean CHECK_HEAD启动，起止核完整提交和空porcelain。CPU、JAX_PLATFORMS=cpu、CUDA_VISIBLE_DEVICES为空，uv --no-sync；缓存落v1-store。数据、YAML和完整公共外壳见 [原启动记录](launch-representative.md)。

新tmux全名mv2-rhythm-fixed，避免复用已有日志：
```bash
bash v1-store/logs/mv2-open-runner.sh rhythm-fixed <CHECK_HEAD>
```
新增rhythm-fixed分支执行与rhythm-rep相同的四项gate及35/411数量硬闸，输出改为v1-store/reports/motion/mv2-rhythm-fixed.json；日志为v1-store/logs/mv2-rhythm-fixed.driver.log。所有判据须PASS且EXIT_CODE=0。AA入口只读取本次通过的日志，旧失败结果保留。
