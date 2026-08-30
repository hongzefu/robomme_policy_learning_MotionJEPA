# v1-singlerun-g0 起跑档（C3：G0 完整重锚）

- **run_name**：`v1-singlerun-g0`（用户 2026-08-30 经 AskUserQuestion 确认采计划拟名；AGENTS 6）。
- **计划**：`v5.0-train-entry-restructure-plan.md` 第三版 4.1/第九节；本 run 属用户已授权的三 run 之一（「本机 2 卡，约 1 h loop，预算 1–2 h」）。
- **起跑 commit**：`f641f40`（commitV5.0 训练入口重构；工作区 clean，AGENTS 12/17）。
- **被裁决的两个变量**：`train.main` 体内删三处 tentative 死代码；目录拆分 `bench/`→`g0/`+`util/`（train.py 内置记录器在 bench 路径根本不装，构造性零影响）。

## 定位（第三版修订，重要）

本 run 经 `g0/bench_train_steps.py` 的 `import train as _train` → `_train.main(config)` 启动，被导入模块 `__name__` 是 `"train"`，**`if __name__ == "__main__":` 块结构性不执行**。因此它证明的是「`train.main` 函数体删 tentative 死代码后位级等价」，对新 `__main__` 与新记录器零覆盖——那两样由 C2.5 记录器 smoke（功能验收）与 C4 B 侧（1000 步真实训练，AGENTS 18 第二块）分别覆盖。

## 口径（照 G3 runbook 逐项）

1000 步、batch 8、2 卡 fsdp2、workers 4、seed 42、确定性档 `XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'`、packed 库 `v1-store/datasets/4task-gl-framesamp`、`SAVE_INTERVAL=100`（驱动转译 train 侧 `--save-interval 1` + 记录器侧 `BENCH_DIGEST_INTERVAL=100`）、`EXTRA_DIGEST_STEPS=299`、`WARMUP_STEPS=50`。参考墙钟约 54 min（同配方基线 G0b-r1 loop 实测 54.83 min），预算 1–2 h。

## 前置门

- porcelain 空、clean HEAD = `f641f40`；`env | grep MMEVLA_FRAMESAMP` 为空；`tmux has-session -t v1-singlerun-g0` 无撞名；本 launch.md 已提交。
- packed 库锚点：本轮 env.json 顶层 `store_meta_sha256` 须与 G2/G3 的 `3990165c9cebffdadaceb01cc88470645a3d62f9af65eabccf4331d3fcd5b556` 一致。
- preflight（缺任一 env 必 FAIL 三项；`--dataset` 必须指 legacy 源库，packed 库无 `data/` 会炸）：

```bash
XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 CUDA_VISIBLE_DEVICES=0,1 \
UV_LINK_MODE=copy uv run scripts/training/g0/check_baseline_env.py check \
  --baseline docs/training-doc/v1-grad-baseline-g0b/records/r1 \
  --dataset v1-store/datasets/4task-gl --steps 1000 --batch-size 8
# 判定 BASELINE_ENV=PASS
# 判读纪律：指纹不含仓库代码 sha，PASS 只证「引用 G0 的资格还在」，不证「代码没改坏」
```

- AGENTS 18 末句：本 run 引用既有基线 **G0b-r1**（run_name `v1-grad-baseline-g0b-r1`，commit `570287f`）而非同场次重跑对照侧，preflight 指纹比对结论以上述 `BASELINE_ENV=PASS` 为准，结论在 result.md 留档。

## 起跑命令（G3 模板，路径改 g0/）

```bash
tmux new-session -d -s v1-singlerun-g0 "set -o pipefail; cd <REPO>; \
  STEPS=1000 SAVE_INTERVAL=100 EXTRA_DIGEST_STEPS=299 WORKERS=4 WARMUP_STEPS=50 \
  EXP_NAME=v1-singlerun-g0 RUN_TAG=v1-singlerun-g0 \
  DATASET_PATH=<REPO>/v1-store/datasets/4task-gl-framesamp \
  XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
  UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
  bash scripts/training/g0/run_2gpu_epoch_bench.sh \
    2>&1 | tee v1-store/logs/v1-singlerun-g0-driver.log; \
  echo \"EXIT_CODE=\$?\" >> v1-store/logs/v1-singlerun-g0-driver.log"
```

Monitor 挂 driver.log（每级行缓冲）；过程中每出一次 state 摘要即与 G0b-r1 同步骤比 digest，首次分叉立停（fail-fast，故不单独跑 SMOKE5）。失败按阶梯 GRAD_FIXTURE(15min) → SMOKE5(35min) → 增量二分定位，不得直接重跑。

## 判据（4.1 四分项 + G0_EQ 总闸）

```
BASELINE_ENV=PASS
SCALARS steps=1000 keys=5 hex_mismatch_steps=0
STATE_DIGEST rows=12 mismatch=0
BATCH_DIGEST_CANONICAL rows=14 mismatch=0 ；CANON_CHECK=PASS steps=14
INDEX_SEQ=PASS n=8072
sha256(scalars_hex.tsv) == c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757   # 第六份同值
G0_EQ=PASS   # tests/g0_gate.py fail-closed 总闸（反向自测已于 C2 完成：正样本 PASS + 四负样本全 FAIL）
```

**不作判据**：raw `BATCH_DIGEST mismatch=4`（`static_image_emb`/`static_pos_emb` dtype 统一已知预期失配，须与 G2/G3 逐字吻合——由 g0_gate A 类校验）与 `DET_CHECK=FAIL` 总行/退出码。

判读命令（先四分项、再总闸）：

```bash
UV_LINK_MODE=copy uv run scripts/training/g0/compare_baseline.py \
  docs/training-doc/v1-grad-baseline-g0b/records/r1 \
  v1-store/bench/2gpu-epoch-bench/v1-singlerun-g0 --tier singlerun-vs-g0b \
  | tee docs/training-doc/v1-singlerun-g0/records/compare_vs_g0_r1.txt

UV_LINK_MODE=copy uv run scripts/training/tests/g0_gate.py \
  --compare-out docs/training-doc/v1-singlerun-g0/records/compare_vs_g0_r1.txt \
  --run-dir v1-store/bench/2gpu-epoch-bench/v1-singlerun-g0 \
  --scalars docs/training-doc/v1-singlerun-g0/records/scalars_hex.tsv \
  --expect-sha256 c799a0b299f243c1740f1594b62aec920cf7ad0033a29d37b851051d52105757 \
  --env-out <preflight 输出文件>
```

收官：`tests/project_scalars.py` 投影 `scalars_hex.tsv`（与 C4 共用同一份投影实现，其 `--selftest` 已用 G3 固化产物自证逐字节复现）→ sha256 对锚点；`check_baseline_env.py manifest` 生成防腐清单；留档含附录 D 两图（数据逐跳链路，C3 后补实测数字）与 4.6 两块一致性讨论。C2.5 记录器 smoke 结果一并归档于本 run 的 result.md。
