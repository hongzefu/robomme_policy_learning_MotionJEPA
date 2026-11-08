# 双消融三 seed 官方评估启动档案

## ① 目的

两条 GreatLakes 训练的双消融模型只评最终 checkpoint 79999，按官方三个模型 seed（7、42、0）在官方 test 16×50 = 800 局上各评一次，共 6 次 4800 局：

- N：`gl-orig100-ctx32x4x4-noxyz-motion-s2split4x256-b64-80k`（编码器去掉 depth/xyz）
- S：`gl-orig100-ctx32x4x4-nostate-motion-s2split4x256-b64-80k`（编码器去掉 action/state）

计划为根目录 `1108-gl-ablation-encoders-3seed-eval-plan.html`（`77a14a4`，开工前审查修订 `857c746`）。

用户决策原话（按时间）：
1. 「这两个训练都结束了 怎么做eval？在greatlakes的 之前是怎么做的？单卡job还是多卡 一个卡几个worker？」
2. 「能不能实现1卡多worker？」「写进计划」
3. 选择题：三 seed 7/42/0；复用两份训练占位；再开第二个 ControlMaster。
4. 「controlmaster你直接起 不在计划内 先定下来」；`greatlakes2` 于美东 11-08 建好（gl-login3）。
5. 「训练验收后接手占位」「登记」
6. 开工令：「开工」（针对 `77a14a4`）
7. 「暂时不管这个问题了开始了」：上游 `gl-abl-encoders-train` 仍为 running（V10 `CKPT_DELIVERY_READY=FAIL ×2`，跨 A6000/A40 在线握手 numerics 不一致），不等上游 done。
8. 开工前审查 F1 裁决：「推荐口径，V6 FAIL 照跑」。

开工前审查：`PLAN_CONSISTENCY_REVIEW=FAIL base=77a14a4 findings=9 blocking=2`；F1 交用户裁决，F2–F9 文字统一，修订 `857c746`。

## ② 版本与代码状态

- 执行代码 `4d70b1a5a02e2e1b4cfda51e8b6d2573dc453d42`（分支 `v4-eval-abl-1108`）= 训练锚点 `8b52210` `--no-ff` 合并 `5fce5d3`（10-31 seed 门）。只带进 `examples/robomme/eval.py` 与 `tests/evaluation/test_official_seed_gate.py`。
  - `PRE_MERGE_REVIEW=PASS base=8b52210… tip=4d70b1a… files=2 commits=3 findings=1`（finding：子提交 body 验证一项只写「见交回」，判定行补在此处）。
  - V1 原判 `EVAL_CODE=FAIL sha=4d70b1a tests=325passed`：`tests/evaluation` 另有 31 条失败、2 个收集错误、1 个挂起（`test_queue_faults` 29、`test_manifest` 1 的 `QueueStore.initialize` 签名不符；`test_gl_pool`/`test_resident_worker` 导入已删符号；`test_testx_client` 1 条要求拒绝 seed 7；`test_testx_pipeline` 挂起）。这些测试与被测代码在 `5fce5d3` 上逐字节相同，属既有失败。
  - 放宽后判定：`EVAL_CODE=PASS sha=4d70b1a tests=seed_gate7+official_contract38+eval_target_contract12+runtime_protocol24+runtime_restore23+simple_queue3 eval_dirs_diff_vs_5fce5d3=0 cfg=mme_vla_suite_orig16_b64_80k`（零差异限 `scripts/evaluation`、`tests/evaluation`、`examples`、`third_party` 四处；`src/` 与 `scripts/training` 有训练侧差异）。对抗验证 `ADJUST_REVIEW=PASS scale=same findings=2`。
- benchmark 子模块 `856bc3a189d4172f3f47dbee4424d585f8d78db3`。
- 执行副本 `$E/releases/4d70b1a…/`：aspen 本地导出 `LOCAL_SNAPSHOT=PASS files=4755 tree_diff=0`，经 sled-vail rsync 到 turbo，按内容比对 `RELEASE_SNAPSHOT=PASS commit=4d70b1a files=4755 tree_diff=0`（另 4681 个文件仅权限位不同，来自 turbo setgid/ACL；首次用 `rsync -anc` 含权限比对误报 FAIL，已改为 `-rlnc` 只比内容）。
- checkpoint：训练会话 V10 离线导出目录（只读引用）
  - N：`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/exports/hf-ckpt-gl-orig100-ctx32x4x4-noxyz-motion-s2split4x256-b64-80k/stage/79999`，`manifests/step-79999.json` sha256 `1d1e4b4e8310b5b9925e0000a7e651fa1a7bfdc93384029a5857111cfdfe20c4`，`motion_provenance.json` `03dc1d30a8c95e9488096e2c8bfb9feb4953cf74a77cf447cc0bf4d02de56d16`
  - S：同上目录名换 `nostate`，清单 `3e1f86de7be99232799c5af5f1324efcf01d626ad96a9531569c5b87855cf570`，provenance `75a483a746c13bc5e2f67b3fb994a523b0968b7a3a4b5127e6fc5d2740b25073`
- runtime 合同（`measure_target_runtime`，`EVAL_RUNTIME_CONTRACT=RECORDED cross_hardware_numeric=not_required`）：N `bb4d56034f79f452707144140f81ad33e63183e36f3bd970cb277477af5de907`，S `b85d72c8df0dba64a0a5862c00c0595e43df350b8a97fecfb6c820672fdec38d`。

记号：`$E=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval`，`$O=$E/v1-store/evaluation/o16abl-80k`，`$L=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/o16abl-3seed`。

## ③ 启动与配置还原

run 身份 schema `simple-run-identity-1027`，共同字段 `checkpoint_step=79999`、`eval_code_commit=4d70b1a…`、`model_kind=stage2_pool4_concat`、`policy_config=mme_vla_suite_orig16_b64_80k`。800 局清单复制自 10-29（内部签封 `991c3496…`，文件 `8f1cb76d…`）；冒烟 2 局（`BinFill:0`、`VideoPlaceOrder:18`，签封 `f2f98f25…`，文件 `9ebebe78…`）。全部由 `$L/build_runs.py` 在 sled-vail 上生成（`BUILD_RUNS=PASS runs=8`）。

| 编号 | run 根（`$O/…`） | seed | purpose | run_identity_sha256 |
|---|---|---|---|---|
| N-smoke | `o16abl-N-smoke/smoke2` | 0 | validation | `2a9ce485249784a7e669407d74a9cea81553c8b58f0245009fd8ebed7492e593` |
| N-s42 | `o16abl-N-seed42/run800` | 42 | production | `bfffa7a5596618a05818bb19da43d6a225454c7b2cabd40815e1494228cda5b8` |
| N-s7 | `o16abl-N-seed7/run800` | 7 | production | `27c2ad2067c435d5fd50355e2a9b16921e944d71870173832b5279c97507d88a` |
| N-s0 | `o16abl-N-seed0/run800` | 0 | production | `cc9cc10d0096b6f272fe8be26a7b9d1ecff9ac8c9f79861b6d7be2529ac757e8` |
| S-smoke | `o16abl-S-smoke/smoke2` | 0 | validation | `fa011e83cbceb7dd09285527b01bd4a5c3d53d6d9d77bef73758525f92992a77` |
| S-s42 | `o16abl-S-seed42/run800` | 42 | production | `3eb8480f6d2220d7754e62838fac01c9886aa4758a17b7263ead74c1292f852e` |
| S-s7 | `o16abl-S-seed7/run800` | 7 | production | `0749ca8a9c977109ba3b3caf841f915967b2983f3cd32364377c0846702dcec7` |
| S-s0 | `o16abl-S-seed0/run800` | 0 | production | `9190d6995b4ef07b650612e5fa5210249d408f48f1f349667791739574ddeea2` |

命令（每次运行相同，只换 `<名>` 与 worker-id）：

```bash
# 控制器（sled-vail，tmux evab-ctl-<名>）
bash $O/<名>/run-controller-vail.sh
# worker（aspen，tmux evab-w-<wid>）；参数：<ssh别名> <JobID> <worker-id> <卡号> <cpus> <run-worker.sh> <日志名>
bash $L/start-worker.sh <greatlakes|greatlakes2> <JobID> <wid> <卡号> <cpus> $O/<名>/run-worker.sh <wid>
#   = ssh <别名> srun --jobid=<JobID> --overlap --exact --ntasks=1 --cpus-per-task=<cpus> --gpu_cmode=shared \
#       /usr/bin/bash $O/<名>/run-worker.sh 4d70b1a… <wid> <卡号>      # run-worker.sh 内 export CUDA_VISIBLE_DEVICES=<卡号>
```

`run-worker.sh` 由 10-31 `run-worker-run800.sh` 逐行复制，改动只有 run 根、执行 sha、新增 `export CUDA_VISIBLE_DEVICES="$3"`，以及 `WORKER_START` 行多打 `gpu=`。

## ④ 数据集与划分

官方 test，16 任务 × 50 局 = 800 局，160 块 × 5 局，动态队列。冒烟 2 局另成一块。

## ⑤ 关键超参

官方 profile，1300 步上限，正常失败不重试，基础设施错误每局最多重试 1 次；seed 按上表。

## ⑥ 硬件与资源

复用训练占位（训练会话保留未取消），本任务结束后逐个 `scancel`：

| JobID | 节点 | 规格 | 到期（美东） | ssh 别名 |
|---|---|---|---|---|
| 65209321（A） | gl1519 | 4×A40 / 16 CPU / 128G | 11-09 13:38 | `greatlakes` |
| 65209330（B） | gl1503 | 4×A40 / 16 CPU / 128G | 11-09 14:04 | `greatlakes2` |

起跑前现场：两份占位只剩 `.batch`／`.extern` step，无训练 step（`UPSTREAM` 门按用户原话 7 放行，`holds_running=2 foreign_steps=0`）。

冒烟（美东 18:16 起，A 份卡 0／卡 1 各 1 个 worker、4 CPU）：

- `SMOKE2=PASS model=N accepted=2 errors=0 readout=stage2_pool4_split_v1`（success 1 / fail 1）
- `SMOKE2=PASS model=S accepted=2 errors=0 readout=stage2_pool4_split_v1`（fail 2）
- server 回显 `integration_type=context`、`num_tokens=4`。

N-s42 于美东 11-08 18:46:42 起跑（第 1 次正式运行兼测速）：A 份 `ns42-a0…a3` 每卡 1 个、4 CPU；B 份 `ns42-b{0..3}{x,y}` 每卡 2 个、2 CPU。

本轮 tmux 会话（前缀 `evab-`）：aspen `evab-release`（导出改道时停止）、`evab-release2`、`evab-build`、`evab-w-smN`、`evab-w-smS`、`evab-w-ns42-*`（12 个）；sled-vail `evab-ctl-N-smoke`、`evab-ctl-S-smoke`（冒烟后已停）、`evab-ctl-o16abl-N-seed42`。其余在起跑时补入 `result.md`。

## 实施中调整（已告知用户）

- 涉及 turbo 的大量读写改在 sled-vail 上做：aspen 读 turbo 单个小文件实测 35.36 s，vail 读三个 0.51 s。执行副本先在 aspen 本地导出再 rsync，合同与 run 根在 vail 生成。
- launch.md 在 N-s42 控制器起跑后约数分钟补写提交（计划要求起跑前），内容按起跑时状态如实记录。
