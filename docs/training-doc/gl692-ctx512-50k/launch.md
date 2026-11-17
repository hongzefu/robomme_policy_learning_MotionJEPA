# gl692-ctx512-50k 起跑留档（第一段）

## ① 目的
在 GL 原占位 job 65943150 里，对训练 run `v2-1600ep-m8x8-ctx-b128-50k-gl` 的 50000 步 ckpt 做 AWS 同口径 692 局评估（seed 7），与对照组（GL A40、modulation、@50000：211/692（30.49%））比较。计划：[`1116-gl-4task400-ctx-692-eval-plan.html`](../../1116-gl-4task400-ctx-692-eval-plan.html)。

用户决策原话：11-16「给出方案 跑完了直接在本地原job评估 参考aws的评估方式」「加上并开工」「按推荐改」；11-17「这些问题你自己来修理。就除了会影响评估结果语义的问题，其他的你都自己来决策。」「不要每次都阻塞来问我」。

## ② 版本与代码状态
- 评估代码：commit `97f237ac7de713fe95847627da9b3f7a6a7249ae`（commitV4.8.5，合入 E1），评估副本 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/eval-wt/97f237ac`（detached worktree，子模块 `third_party/robomme_benchmark` @ b4e97f22）。
- 训练代码：`<GL_REPO>` HEAD af56267（训练 Beta，未动）。
- 独立环境：`$W/v1-store/envs/policy-eval-{server,client}`（setup.sh，11-17 07:56–09:33 美东，server 209 包、client 233 包，EXIT_CODE=0）。
- 起跑前验收（11-17 09:33 美东）：`EVAL_WT=PASS head=97f237ac editable=wt`、`POOL_DEFAULT_EQ=PASS lines=33`、V2 `SHARD_MAP` 8 行 gpu∈{0..3}、`EVAL_SHARDS=12` 退 2、`SUMMARY_SMOKE=PASS ems_default=1 ems_set=0`；合并闸门 `PRE_MERGE_REVIEW=PASS base=8d6ce5b tip=1d1fb27 files=3 commits=1 findings=2`、合并后 pytest 12 passed、resolved_sha256 复算两条逐位一致。

## ③ 启动与配置还原
链脚本 `<GL_REPO>/v1-store/jobs/g4c-ev512-chain.sh` + 公共部分 `g4c-ev-chain-common.sh`（不进 git），在 aspen tmux `g4c-ev512-chain` 里 09:34 美东起跑，日志 `<GL_REPO>/v1-store/logs/g4c-ev512-chain.log`，各关输出 `<GL_REPO>/v1-store/logs/g4c-ev512-stages/`。

每关前缀：
```
ssh -o BatchMode=yes greatlakes srun --jobid=65943150 --overlap --exact --ntasks=1 --cpus-per-task=16 --gpu_cmode=shared --chdir=$W \
  env -u SLURM_JOB_ID -u SLURM_CPUS_PER_TASK EXPECT_MOTION_STATS=0 MMEVLA_MOTION_OVERFLOW= \
  POLICY=perceptual-framesamp-context-8frame-8x8 POLICY_CONFIG=mme_vla_suite_b128_60k DEMO_PREFIX_ORIGINAL=1
```
- 第 1 关：等训练日志 `EXIT_CODE=0`、tmux `g4c-train-512` 退出、ckpt 5000..50000 共 10 份、占位只剩 .batch/.extern、就绪标记 `$W/v1-store/.g4c-eval-ready` 存在 → `TRAIN_DONE=PASS ckpts=10`。
- 第 2 关：`RUN_NAME=gl692-ctx512-g6b G6B_SHARE_FRACTION=0.38 local_692_pool.sh --g6b <CK> 7`。
- 第 3 关：`RUN_NAME=gl692-ctx512-cap2 EVAL_PER_GPU=2 EVAL_MEM_FRACTION=0.38 … --g6b-capacity`；不过退 `cap1 / 1 / 0.70`；链算 `HOST_MEM`（×4 ≤ 115）与 `CAP_TIMEOUT`（20×wall×2 < 14400，不过由会话裁定后 `START_STAGE=4` 重起）。
- 第 4 关：`RUN_NAME=gl692-ctx512-50k EVAL_NGPU=4 EVAL_SHARDS=<8|4> EVAL_PER_GPU=<2|1> EVAL_MEM_FRACTION=<0.38|0.70> local_692_pool.sh <CK> 7`。

## ④ 数据与划分
GL 原前缀库 `<GL_REPO>/v1-store/demo-prefix/binfill-3b4de03a0b46`（index sha256 `3ef6b165…3f27`），692 局、seed 7，预标 9 局（`finalckpt-nomotion59999-gl/records/preseeded-errors.json`）。

## ⑥ 硬件
GL spgpu 占位 65943150（4×A40），与训练同 job，训练退出后才起评估 step。

## 本轮 tmux 会话清单
`g4c-ev-setup`（已退出，EXIT_CODE=0）、`g4c-ev512-chain`、`g4c-ev1024-chain`。

（第二段 result.md 在 EVAL692 出来后补写。）
