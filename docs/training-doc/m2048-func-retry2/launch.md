# 同一编译路径的功能交叉检查

在retry1通过后复核发现，帧带扰动使用loss_only，而基线来自loss_grads；两条JIT图的舍入差不应被计作帧带作用。仅修复验证器：额外用同一loss_only重复三次建立A/A基线，记录它与loss_grads的差值；帧带与有效位垃圾的正负对照各自在同一路径比较，并显式拒绝扰动结果非有限。三次全部梯度A/A、2048位置梯度、删token反例、mask和所有原阈值保持。

从本次工具修复的clean提交启动，完整HEAD及UTC现场记录；生产src和uv.lock与CAND相同，独立副本与依赖/资产核验沿用[retry1](../m2048-func-retry1/launch.md)。GPU2，seed42，确定性XLA配置不变；输出全新m2048-func-retry2，旧失败、探针与retry1均保留。该补测与主副本三组固定CAND训练并行，不改其代码或版本。

实际执行路径为v1-store/workspaces/m2048-tool-fix，命令为 `uv run --no-sync python scripts/training/tests/check_32frame_modul.py func --out /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/m2048/m2048-func-retry2/func.json`。环境、来源自证与retry1相同，只替换run名和独立JAX缓存名；detached tmux全名m2048-func-retry2，pipefail/tee/EXIT_CODE日志位于主仓v1-store/logs/m2048-func-retry2.log。需确认GPU2空闲、记录与日志全新且无外链。

新增判定LOSS_PATH_BASELINE=PASS要求loss_only三次重复有限且逐位同；FRAME_BAND_PERTURB继续要求32带全部高于本路径A/A噪声。通过后以本run作为最终V5证据，不放宽判据、不将不同编译路径的差值混入扰动量。用户完整取证及全部门通过后利用率>50%可直接起跑的决定不变。
