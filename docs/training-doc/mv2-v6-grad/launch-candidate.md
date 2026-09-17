# 固定 batch 的逐叶梯度对拍：候选侧起跑准备

用户原话「开始实施 有问题尽早问用户」。本轮按 [0916 计划](../../../0916-motion-modul-8x8-plan.md) 步骤 2d 执行；改前 BASE 固定为 `2126b1b1c436166662ff89a629985ecd4524fc42`。候选 CAND 为完成生产与验证工具两个提交后的 clean HEAD，运行前显式传入完整 SHA；起跑记录另写 `v1-store/bench/mv2/v6-candidate-launch.json`，不把事后提交冒充启动版本。

## 口径与判据

GPU 4、5，batch 8、fsdp 2、seed 42，c8-b 的 mixed1/allshort/allfull 三组；61 个公共初态叶及三组各 38 个梯度叶须逐位相同。

数据、依赖与数值口径沿用 BASE；`uv.lock` SHA 为 `02cbc3ba67a9024f8afb9e31f60661c9abdcc3eb680ae80a8ee464c639327221`。V6/V7 固定 `CUDA_VISIBLE_DEVICES=4,5`、`XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`、显存比例 0.95。V7 的 `SAVE_INTERVAL=25 EXTRA_DIGEST_STEPS=99 WORKERS=4 WARMUP_STEPS=20 STEPS=100` 与 BASE 相同，`EXP_NAME=mv2-v7` 共用编译缓存，`RUN_TAG=mv2-v7-cand KEEP_JAX_CACHE=1`；不启用本轮新增的性能 trace。

V7 的环境检查显式使用 BASE 驱动实际采指纹的 source 根，再由驱动保存 packed 库的顶层身份字段。比较前另用机器断言两侧源码 SHA 分别为固定 BASE 与实际 CAND，且二者不同；不依赖原比较器自动检查源码身份。

## 启动与输出

```bash
CAND=$(git rev-parse HEAD)
test -z "$(git status --porcelain)"
tmux new-session -d -s mv2-v6 "bash /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/mv2-candidate-runner.sh v6 '$CAND'"
```

外壳沿用 BASE 的缓存、离线资产、uv 与日志配置，补齐 dump 的 `--exp-name`；只替换本阶段候选输出目录和源码锚点。实际产物为 `v1-store/bench/mv2/v6-cand`，外层日志为 `v1-store/logs/mv2-v6-candidate-driver.log`。V7 驱动内层写 `mv2-v7-cand.log`，不会重现 BASE 的日志重名问题。全程使用 pipefail、tee 与 EXIT_CODE；只使用上面列出的本轮 tmux 会话，不触碰其他会话。

源码和依赖在取证期间保持不变。模型数值失配按计划只撤销生产提交并交用户；命令、目录或环境口径失配先排查，不改判据。完整结果待运行结束后归档。
