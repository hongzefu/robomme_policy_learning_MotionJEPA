# v1-postclean-g3（G3：v4 破坏性重构正确性长跑）launch

- **目的**：v3 破坏性单一化重构计划（v3-destructive-restructure-plan.md）的终局闸门
  N5——七刀（commitV4.0–V4.6）全部冻结后，从 clean HEAD 以最终两域布局跑 1000 步
  确定性档，离线对拍 G0（G0b r1 固化产物）bitwise。红线：收敛后训练侧交付给模型
  的字节与重构前逐位一致。
- **run_name**：`v1-postclean-g3`（AGENTS 6，用户 2026-08-29 拍板；EXP_NAME=RUN_TAG
  独立编译缓存、不复用任何烟测缓存——R23，现场重编译对拍成立由 D2-cold 授权）。
- **commit**：`3eccb10`（= commitV4.6 `b30be80` + 两个 docs 留档 commit；porcelain 空）。
- **执行日期**：2026-08-30。本机 2×RTX 6000 Ada，b8，seed 42，1000 步。

## 前置门（全过才起跑，逐项实测）

- 工作区 porcelain 空、clean HEAD 起跑 ✓
- 第一块五判定行全 PASS（COPY_DIFF / IMPORT_ISOLATION / GRAD_FIXTURE / ONLINE_MEM /
  预 G3 SMOKE5）✓，见 `records/block1/first_block_verdicts.txt`
- 7.5 三条迁移判定行全 PASS（RELOCATION_REFS / ROOT / COLLECT）✓
- bench 四道源码护栏在新 train.py 上成立（v46-smoke5 在新布局 BENCH_PASS 即实证）✓
- `train_step` info 含 `mem_enc_norm`（metrics 五 hex 键齐全）✓
- packed 库 `status=verified` ✓；**库本体锚点**：v46-smoke5 的 env.json 顶层
  `store_meta_sha256=3990165c…`、`manifest_sha256=20da0dfe…` 与
  `docs/training-doc/v1-framesamp-g2/records/env.json` 逐字一致（G2→G3 间库未重建）✓
- `env | grep MMEVLA_FRAMESAMP` 输出为空（SOURCE/MANIFEST/VERIFY/ALLOW_* 全未设）✓
- preflight（必带三环境变量）：`BASELINE_ENV=PASS` ✓
  ——判读纪律：指纹不含仓库代码 sha，PASS 只证「引用 G0 的资格还在」，不能当
  「代码没改坏」的证据；后者只由四维位级对比回答。
- 单 epoch 约束：1000×8=8,000 < 395,289 ✓；`tmux has-session` 无撞名 ✓

## 起跑命令（tmux detached；照 G2 模板去掉 MMEVLA_DATA_BACKEND 行）

```bash
tmux new-session -d -s v1-postclean-g3 "set -o pipefail; cd <REPO_ROOT>; \
  STEPS=1000 SAVE_INTERVAL=100 EXTRA_DIGEST_STEPS=299 WORKERS=4 WARMUP_STEPS=50 \
  EXP_NAME=v1-postclean-g3 RUN_TAG=v1-postclean-g3 \
  DATASET_PATH=<REPO_ROOT>/v1-store/datasets/4task-gl-framesamp \
  XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
  UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
  bash scripts/training/bench/run_2gpu_epoch_bench.sh \
    2>&1 | tee v1-store/logs/v1-postclean-g3-driver.log; \
  echo \"EXIT_CODE=\$?\" >> v1-store/logs/v1-postclean-g3-driver.log"
```

- 摘要步集与 G0 对齐：state 12 次（0/每 100/299/999）、batch 14 行；driver 端
  `SAVE_INTERVAL=100` + `EXTRA_DIGEST_STEPS=299` 自动装记录器。
- env.json 自 commitV4.1 起不再含 backend 字段（量具字段变更，preflight 不比对该字段）。
- **过程中增量对拍**：另挂监视进程 tail `param_checksums.jsonl`，每出一行摘要即与
  G0b-r1 同步骤比 `state_digest`，首次分叉立刻停跑（12 个摘要步天然二分粒度）。
  本轮 12/12 全 MATCH、全程未触发停跑。

## 对拍判读命令与纪律（四分项逐行人工核对；退出码与 DET_CHECK 总行不具判据资格）

```bash
UV_LINK_MODE=copy uv run scripts/training/bench/compare_baseline.py \
  docs/training-doc/v1-grad-baseline-g0b/records/r1 \
  v1-store/bench/2gpu-epoch-bench/v1-postclean-g3 --tier g3-vs-g0b
```

- 判据：① `SCALARS steps=1000 keys=5 hex_mismatch_steps=0`；② `STATE_DIGEST rows=12
  mismatch=0`；③ `BATCH_DIGEST_CANONICAL rows=14 mismatch=0` + `CANON_CHECK=PASS`；
  ④ `INDEX_SEQ=PASS`（前 8,000 条前缀；n 与 G2 同为 8072 即逐字吻合）。
- **不作判据**：raw `BATCH_DIGEST mismatch=4 first_bad_step=100 bad_keys=2
  (static_image_emb/static_pos_emb)`——V2.4b dtype 统一的已知预期失配，与 G2 逐字
  吻合即正常；`DET_CHECK=FAIL` 总行为已拍板不修的工具聚合缺口。
- **三处 fail-open 人工补位**（量具本轮不改，二节第 13 条）：
  1. `keys=5` 是常量、缺标量静默跳过——另核 scalars_hex.tsv 表头恰为六列
     `step\tloss.hex\tgrad_norm.hex\tllm_grad_norm.hex\tmem_enc_norm.hex\tparam_norm.hex`；
  2. `INDEX_SEQ` 只比最短公共前缀——另核 G3 侧 n 不小于 G0b 侧且 ≥8000；
  3. canonical 与 index 不进 verdict 与退出码——四分项逐行为准。
- `--tier` 只是打印标签，不切换任何判据严格度。
- 额外必检：本轮 `batch_digests.jsonl` 首行 `n_keys=12`（比较器不产出该字段）。
- 收官一行：投影 `records/scalars_hex.tsv`（六列、1001 行、末尾单换行）的 sha256
  必须等于 `c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757`
  （G0b r1/r2、G1、G2 四份同值）。
