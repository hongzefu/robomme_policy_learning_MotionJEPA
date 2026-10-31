# 官方16×100 modulation 与 context 两模型三 seed 官方评估启动档案

## ① 目的

用官方三个模型 seed（7、42、0），在官方 test 16×50 = 800 局上评估两条官方 16×100 训练模型的最终 checkpoint 59999：modulation split（`orig100-m32x4x4-modul-motion-s2split4x256-b128-60k`）与 context split（`orig100-ctx32x4x4-motion-s2split4x256-b128-60k`）。context 的 seed 42 复用 10-29 的 `run800-r2`（311/800），本轮新跑 5 次共 4000 局。计划为根目录 `1031-orig100-modul-ctx-3seed-official-eval-plan.html`（修订版 `5db7311`，措辞统一 `dfe8a9a`）。

用户决策原话（按时间）：
1. 「给出 evaluate 的方案，官方的 checkpoint evaluate 了三次，是哪三个 seed 你找一下。然后我现在手上应该有两个 model，一个是 modulation 的 model，一个是 context model，都要 evaluate，都要 evaluate 三次。」
2. 「modulation robomme-vla-orig100-m32x4x4-modul-motion-s2split-60k-v1 …… context（split） robomme-vla-orig100-ctx32x4x4-motion-s2split-60k-v1 …… eval这两个」
3. 「开工并登记」（针对 `e60dd99`）
4. 「优先先跑 modulation，现在同样的 seed42，然后再跑其其他的两个。」（计划据此修订，原开工令作废）
5. 开工令：「开工」（针对 `5db7311`）

开工前审查：首轮 `PLAN_CONSISTENCY_REVIEW=FAIL base=e60dd99 findings=4 blocking=2`，处置写入修订版；修订部分复审 `PLAN_CONSISTENCY_REVIEW=PASS base=5db7311 findings=4 blocking=0`，四条均为措辞，统一于 `dfe8a9a`（N3 可选加强未采纳）。

## ② 版本与代码状态

- 执行代码：`5fce5d3c153956708d1470132c22201c456756a7`（分支 `v4-eval-3seed-1031`，`commitV11.70: 官方评估 seed 放开到 0／7／42`，从 `9459859` 合入子代理提交 `2cb5d8e`）。benchmark 子模块 `856bc3a189d4172f3f47dbee4424d585f8d78db3`。与 10-29 所用 `9459859` 的差别只有 `examples/robomme/eval.py` 的 seed 判断和新增测试 `tests/evaluation/test_official_seed_gate.py`。
- 判定：`SEED_GATE=PASS allowed=0,7,42 rejected=1`（45 passed）；`PRE_MERGE_REVIEW=PASS base=9459859… tip=2cb5d8e… files=2 commits=1 findings=0`；`POST_MERGE_REVIEW=PASS merge=5fce5d3 tests=45passed files=2`。
- 执行副本 `$E/releases/5fce5d3c153956708d1470132c22201c456756a7/`，`RELEASE_SNAPSHOT=PASS files=4674 tree_diff=0`。
- modulation checkpoint：bucket `HongzeFu/robomme-vla-orig100-m32x4x4-modul-motion-s2split-60k-v1` 的 `manifests/step-59999.json`，清单 sha256 `05f0b3dc04f745298a675b46213688bf118b4f372743540e14a677042eb1a814`，42 个文件 12,271,321,532 字节；资产目录 `$E/v1-store/assets/vla-o16modul-59999-05f0b3dc…/`，只核字节下载，sha256 后台核对，结果写进 `result.md`。
- context checkpoint：沿用 10-29 已核对的 `vla-o16ctx-59999-ada0653d…`（`VLA_VERIFY=PASS step=59999 files=36 sha_mismatch=0`）。
- runtime 合同：modulation 用 59999 的 `motion_provenance.json`（`d135718b…`，与 5000 的 17 个根文件逐个相同）经 `runtime_contract.py::make_eval_runtime_contract` 生成，`$O/o16modul-60k-seed42/contracts/runtime.json` sha256 `8de0642ec52d2161995dc31fbf41f658da7e29ec8000353253f326f25895e6cd`，冒烟与三个 seed 共用；context 沿用 `e254f115…`。同法对 context 复算与 10-29 合同内容完全相等。

其中 `$E=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval`，`$O=$E/v1-store/evaluation/official-stage2-1026`，`$L=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/o16-3seed`。

## ③ 启动与配置还原

run 身份 schema `simple-run-identity-1027`，`run_identity_sha256 = identity_sha256(run-identity.json)`（同法复算 10-29 得 `3ebc1c3f…`，与当时一致）。共同字段：`checkpoint_step=59999`、`eval_code_commit=5fce5d3…`、`model_kind=stage2_pool4_concat`、`policy_config=mme_vla_suite_orig16_b128_60k`。800 局清单内部签封 `991c3496…`（文件 `8f1cb76d…`，与 10-29 相同）；冒烟 2 局（`BinFill:0`、`VideoPlaceOrder:18`）签封 `f2f98f25…`（文件 `9ebebe78…`）。

| 编号 | run 根（`$O/…`） | seed | purpose | run_identity_sha256 | spec-input 文件 sha256 |
|---|---|---|---|---|---|
| M-smoke | `o16modul-60k-seed42/smoke2-60k` | 0 | validation | `09d1cc5dcba7c4335d4cccee59157254182a09db5fe03e8a373a790616678d91` | `3b09e1a7…` |
| M-s42 | `o16modul-60k-seed42/run800` | 42 | production | `4b7b235abe439210b75677b96e51f819f9de7720117260b6fc1becca0f80362f` | `daa29bfa…` |
| M-s7 | `o16modul-60k-seed7/run800` | 7 | production | `e827df8d8b16dd78d26b8e7c4bfaf3bf6644d6326bbc3060a424eecd415e7e58` | `678006f6…` |
| M-s0 | `o16modul-60k-seed0/run800` | 0 | production | `e9876c99cb9046ac25f19867658991296db11a26dc36d23cb559596cd21affbf` | `cae1835d…` |
| C-s7 | `o16ctx-60k-seed7/run800` | 7 | production | `c166dd5c7b2e3618e3d7778dae01f85946f98c08649ced689134fde1edae2e97` | `42641ea9…` |
| C-s0 | `o16ctx-60k-seed0/run800` | 0 | production | `5fdef9f0bd73f5997e02040794f41ac1b037f8b5d7f418987f3835ef11a27149` | `6cdf0d25…` |

run 身份与 spec-input 在各 run 根父目录的 `inputs/<run 名>.run-identity.json`、`inputs/<run 名>.spec-input.json`。spec 比 10-29 多一个 `model_seed` 字段，经 `resident_worker.py` 传给 server `--seed` 与 client `--model-seed`。

命令（每次运行相同，只换 run 根与 tag）：

```bash
# 下载（Aspen，tmux ev3-fetch60k）；完成后起后台校验 tmux ev3-verify60k
python3 -u $O/o16modul-60k-seed42/fetch.py 59999 --size-only | tee $L/fetch60k.log
python3 -u $O/o16modul-60k-seed42/verify.py 59999 | tee $L/verify60k.log
# 初始化（执行副本内）
python -m scripts.evaluation.official_queue initialize --run-root $O/<rel> --manifest <清单> --spec $O/<base>/inputs/<tag>.spec-input.json
# 控制器（sled-vail，tmux ev3-ctl-<编号>）
bash $O/<base>/run-controller-vail-<tag>.sh
# worker（Aspen，tmux ev3-w-<编号>-NN）
bash $L/start-worker.sh <JobID> worker-NN $O/<base>/run-worker-<tag>.sh <日志名>
#   = ssh greatlakes srun --jobid=<JobID> --overlap --exact --ntasks=1 --cpus-per-task=4 --gpu_cmode=shared \
#       /usr/bin/bash $O/<base>/run-worker-<tag>.sh 5fce5d3c153956708d1470132c22201c456756a7 worker-NN
```

`run-worker-<tag>.sh`、`run-controller-vail-<tag>.sh` 由 10-29 `run-worker-r2.sh`、`run-controller-vail-r2.sh` 逐行复制，只替换 run 根与执行 sha（diff 已核）。

## ④ 数据集与划分

官方 test，16 任务 × 50 局 = 800 局，160 块 × 5 局，动态队列。冒烟 2 局另成一块。

## ⑤ 关键超参

官方 profile，1300 步上限，motion 窗口超 96 即 raise，不重试正常失败，基础设施错误每局最多重试 1 次；seed 按上表。

## ⑥ 硬件与资源

10 份 spgpu 占位，各 1×A40 / 4 CPU / 24 G / 48 h，美东 10-31 18:58 提交，5 次运行与冒烟共用：

| JobID | 节点（19:10 美东） |
|---|---|
| 65146920 | gl1509 |
| 65146921 | gl1520 |
| 65146922 | gl1513 |
| 65146923 | gl1513 |
| 65146924 | gl1514 |
| 65146925 | gl1510 |
| 65146926 | PENDING（Resources） |
| 65146927 | PENDING（Priority） |
| 65146928 | PENDING（Priority） |
| 65146929 | PENDING（Priority） |

本轮 tmux 会话清单（前缀 `ev3-`）：Aspen `ev3-fetch60k`；其余在起跑时补入 `result.md`。

日志：本机 `$L/`；控制器与 server 日志在各 run 根 `logs/`，汇总在各 run 根 `queue/summary.json`。
