# v1-framesamp-g2（S6 第二块·G2 训练对拍）launch 记录

对应 `v2-framesamp-restructure-plan.md` 阶段 3 S6（C.3）：`MMEVLA_DATA_BACKEND=packed`
从 clean HEAD 起跑一轮 1000 步，离线对拍 G0 固化产物（A 侧不重跑）。run_name
2026-08-27 经用户确认。

## 并行豁免声明（2026-08-27 用户批准）

本 run 与 S5（`v1-framesamp-cmp`，本机 CPU 对拍）**同机并行**——严格串行与红线 B6
在本对 run 上经用户批准豁免（计划 E 节已注记）。依据：两 run 均不测速、判据与负载
无关（S6 为确定性档 bitwise；其 util/步时按 B5 本就禁作性能结论）；资源不冲突
（S5 纯 CPU 单进程 / S6 两卡 + 4 worker；NFS 合计需求 ≪ 实测供给）。风险仅为
S5 若败则本轮作废重跑；放行规则仍要求两块都过。

## 起跑环境与前置断言（C.3）

- **起跑 HEAD**：本 launch.md 的 docs commit 本身（结果留档回填全 sha）
- **preflight**：`BASELINE_ENV=PASS`（vs G0 r1 `docs/training-doc/v1-grad-baseline-g0b/records/r1`，
  `--steps 1000 --batch-size 8`，确定性档 XLA_FLAGS 注入下现场实测）
- backend **显式** `packed`；库 `status=verified`（S4：`VERIFY_PACK=PASS scanned=483291
  mismatches=0`）；`MMEVLA_FRAMESAMP_VERIFY` 默认 fast；`ALLOW_UNVERIFIED`/`ALLOW_SUBSET` 未设
- 单 epoch 约束：1000×8 = 8,000 < 395,289 ✓；EXP_NAME 独立（不与任何历史 run 共用
  编译缓存；确定性档关 autotune，共用无收益）

## G0_SCOPE 反向白名单断言（T1）

```
$ git diff --name-only 55e6e5bf8ef38b780902d0e63257ea859a432a2c HEAD | grep -Ev '<白名单>'
pyproject.toml
src/mme_vla_suite/datastore/README.md
src/mme_vla_suite/datastore/__init__.py
src/mme_vla_suite/datastore/framesamp_store.py
src/mme_vla_suite/datastore/manifest.py
src/mme_vla_suite/shared/data_utils.py
src/mme_vla_suite/training/dataloader.py
src/mme_vla_suite/training/framesamp_dataset.py
```

逐项说明（`git status --porcelain` 为空）：

| 文件 | 说明 |
|---|---|
| `datastore/` 四件 + `training/framesamp_dataset.py` | B.0 授权新增（V3.0/V3.1）。**新增文件，legacy 模式不构造**；模块 import 无副作用（datastore 仅 numpy/ml_dtypes；framesamp_dataset 经 shared.data_utils 拉的 flax 本就在旧链 import 图内） |
| `training/dataloader.py` | B.0 授权接线（V3.1）。**legacy 路径逐字未动**：`MMEVLA_DATA_BACKEND` 未设默认 legacy、legacy 分支 RoboMMEDataset 构造参数逐字同源、`transform_dataset`/`TorchDataLoader`/`DataLoaderImpl` 零改动；e2e 实证 tmp-v31-smoke（默认 backend STEPS=5）BENCH_PASS。本 run 显式 `packed`，训练语义等价正由本 run 的 bitwise 判据证明（T1 明文：非白名单豁免） |
| `shared/data_utils.py` | dtype 修复 commitV2.4b（三行显式 dtype，**已验收收官**）——即 C.3 所记「G0 固化后交付 dtype 经过一次已验收统一」，raw 输入摘要预期失配的来源；bitwise 判据（五标量/state_digest/canonical/index）已计入该差异 |
| `pyproject.toml` | dtype 计划 V2.4a 的 ruff 目录豁免一行（lint 配置，无训练语义） |

## 命令

```bash
tmux new-session -d -s s6-g2 "set -o pipefail; cd <REPO_ROOT>; \
  STEPS=1000 SAVE_INTERVAL=100 EXTRA_DIGEST_STEPS=299 WORKERS=4 WARMUP_STEPS=50 \
  EXP_NAME=v1-framesamp-g2 RUN_TAG=v1-framesamp-g2 \
  DATASET_PATH=<REPO_ROOT>/v1-store/datasets/4task-gl-framesamp \
  MMEVLA_DATA_BACKEND=packed \
  XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
  PYTHONUNBUFFERED=1 bash scripts/smoke-local/run_2gpu_epoch_bench.sh \
    2>&1 | tee v1-store/logs/v1-framesamp-g2-driver.log; \
  echo \"EXIT_CODE=\$?\" >> v1-store/logs/v1-framesamp-g2-driver.log"
```

本机 2×RTX 6000 Ada、b8、seed 42、`bench_train_steps.py` 入口；摘要步集与 G0 完全
对齐（步 0/每 100/299/末步，共 12 次完整 TrainState 摘要）。预计 2–2.5 h（含摘要停顿）。

## 判据（C.3；对拍 `compare_baseline.py` 离线 vs G0 r1 records，分项判读——
## 2026-08-27 用户拍板不修总行聚合，raw 预期失配不计入）

1. 逐步五标量 hex 列 diff 为空（1000 步全对齐）；
2. 12 次摘要步 `state_digest` diff 为空；
3. canonical 输入摘要（14 记录步）+ 全步 index 序列与 G0 逐位一致；
4. raw 输入摘要**不计入判据**（已知预期失配，来源见上表 `data_utils.py` 行）。

失败处置按 C.3/三节第二块；其 util/步时仅留档参考、禁作性能结论（B5）。
