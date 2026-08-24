#!/usr/bin/env bash
# ── 本机 2 GPU epoch 时长基准（官方口径 + 一致性检验记录底座）───────────────────
#
# 目标：配置尽可能对齐 scripts/finetune_mme_vla_suite.sh（batch 64、num-workers 4、
# use_history + framesamp-context），在 2 卡 + NFS turbo 数据上跑 300 步，
# 用稳态 s/step 外推 1 个 epoch（395,289 样本）的时长；同时逐步记录 loss/梯度范数、
# 每 SAVE_INTERVAL 步记录参数校验和，为将来 dataloader 改动的一致性检验留底。
# 与官方默认训练的逐项差异及记录文件格式见同目录 README.md。
#
# ⚠ 本机数字按 AGENTS.md 第 13 条只作估算，不作正式吞吐结论。
# ⚠ 不落任何 checkpoint（save_state 已被 bench 入口替换为校验和记录器）。
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../data-preprocess-GL" && pwd)/paths.sh"

v1_prepare_dirs
v1_require_venv
v1_require_models 1                       # 需要 tokenizer 与 pi05_base

DATASET_PATH="${DATASET_PATH:-${GL_DATASET}}"
STEPS="${STEPS:-300}"
WORKERS="${WORKERS:-4}"                   # 官方口径
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"      # 参数校验和间隔（不落 ckpt）
WARMUP_STEPS="${WARMUP_STEPS:-50}"        # 稳态统计丢弃的头部步数（JIT 编译 + worker 起步）
EPOCH_SAMPLES=395289                      # meta/stats.json 的 execution_samples，drop_last
NORM_STATS="${TRAIN_ASSETS}/mme_vla_suite/robomme/norm_stats.json"
BENCH_ROOT="${V1_STORE}/bench/2gpu-epoch-bench"

[[ -f "${DATASET_PATH}/meta/stats.json" ]] || {
  echo "错误: 数据集不存在或未 finalize: ${DATASET_PATH}/meta/stats.json" >&2; exit 1; }
[[ -f "${NORM_STATS}" ]] || {
  echo "错误: norm stats 缺失: ${NORM_STATS}（用 scripts/compute_norm_stats.py 先生成）" >&2; exit 1; }

STATUS=1
FINAL_BATCH=""
FINAL_RECORD_DIR=""

# OOM 自动降档：64 是官方全局 batch 口径；2 卡下 per-device 翻倍，显存无历史实测
for BATCH in 64 32 16; do
  RUN_NAME="v1-2gpu-epoch-bench-b${BATCH}"
  CKPT_DIR="${TRAIN_RUNS}/mme_vla_suite/${RUN_NAME}"
  RECORD_DIR="${BENCH_ROOT}/${RUN_NAME}"
  LOG="${LOGS_DIR}/${RUN_NAME}.log"

  [[ -e "${CKPT_DIR}" ]] && {
    echo "错误: run 目录已存在, 禁止 overwrite: ${CKPT_DIR}" >&2; exit 1; }
  [[ -e "${RECORD_DIR}" ]] && {
    echo "错误: 记录目录已存在, 禁止覆盖既有记录: ${RECORD_DIR}" >&2; exit 1; }
  mkdir -p "${RECORD_DIR}"

  # 环境留档：将来一致性 A/B 的对照 run 必须逐项同设（见 README.md）
  "${PY}" - "$RECORD_DIR" <<EOF
import json, subprocess, sys, platform
import jax
d = {
    "argv_batch": ${BATCH}, "steps": ${STEPS}, "workers": ${WORKERS},
    "save_interval": ${SAVE_INTERVAL}, "seed": 42, "fsdp_devices": 2,
    "history_config": "perceptual-framesamp-context.yaml",
    "dataset_path": "${DATASET_PATH}",
    "git_head": subprocess.run(["git", "-C", "${REPO_ROOT}", "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
    "git_dirty": bool(subprocess.run(["git", "-C", "${REPO_ROOT}", "status", "--porcelain"],
                                     capture_output=True, text=True).stdout.strip()),
    "XLA_FLAGS": "${XLA_FLAGS:-}",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "hostname": platform.node(),
    "python": sys.version,
    "jax": jax.__version__,
    "nvidia_smi": subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().splitlines(),
}
json.dump(d, open("$RECORD_DIR/env.json", "w"), indent=2, ensure_ascii=False)
EOF

  echo "=== 2 GPU epoch 基准: ${RUN_NAME} (${STEPS} steps, batch ${BATCH}, workers ${WORKERS}) ==="
  echo "  数据集: ${DATASET_PATH}"
  echo "  记录目录: ${RECORD_DIR}"
  set +e
  (
    set -e
    cd "${REPO_ROOT}"
    BENCH_RECORD_DIR="${RECORD_DIR}" \
    CUDA_VISIBLE_DEVICES=0,1 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    PYTHONUNBUFFERED=1 \
    WANDB_MODE=disabled \
    "${PY}" "${REPO_ROOT}/scripts/smoke-local/bench_train_steps.py" mme_vla_suite \
      --exp-name "${RUN_NAME}" \
      --assets-base-dir "${TRAIN_ASSETS}" \
      --checkpoint-base-dir "${TRAIN_RUNS}" \
      --batch-size "${BATCH}" \
      --num-workers "${WORKERS}" \
      --num-train-steps "${STEPS}" \
      --log-interval 1 \
      --save-interval "${SAVE_INTERVAL}" \
      --seed 42 \
      --fsdp-devices 2 \
      --dataset-path "${DATASET_PATH}" \
      --weight-loader.params-path "${MODELS_DIR}/openpi-assets/checkpoints/pi05_base/params" \
      --model.use-history \
      --model.history-config perceptual-framesamp-context.yaml \
      --no-wandb-enabled
  ) 2>&1 | tee "${LOG}"
  RC="${PIPESTATUS[0]}"
  set -e

  # 跑完（无论成败）清理：run 目录只剩 orbax 的空壳；jax 编译缓存被 train.main
  # 硬编码写到 ~/.cache/jax_<exp_name>（jax_compilation_cache_dir），一并删
  if [[ -e "${CKPT_DIR}" ]]; then
    case "${CKPT_DIR}" in
      "${TRAIN_RUNS}/mme_vla_suite/${RUN_NAME}") rm -rf -- "${CKPT_DIR}" ;;
      *) echo "错误: 拒绝清理非预期路径 ${CKPT_DIR}" >&2; exit 1 ;;
    esac
  fi
  rm -rf -- "${HOME}/.cache/jax_${RUN_NAME}"

  if [[ "${RC}" -eq 0 ]]; then
    STATUS=0; FINAL_BATCH="${BATCH}"; FINAL_RECORD_DIR="${RECORD_DIR}"
    break
  fi
  # ⚠ 日志匹配须先 tr '\r' '\n'：tqdm 用回车不换行，直接 grep 会漏行
  if tr '\r' '\n' < "${LOG}" | grep -qE "RESOURCE_EXHAUSTED|[Oo]ut of memory|OOM"; then
    echo "!!! batch ${BATCH} OOM，降档重试" | tee -a "${LOG}"
    rm -rf -- "${RECORD_DIR}"             # 失败档的记录不保留，避免半截文件误导
    continue
  fi
  echo "错误: batch ${BATCH} 非 OOM 失败（退出码 ${RC}），不降档，人工排查: ${LOG}" >&2
  exit "${RC}"
done

[[ "${STATUS}" -eq 0 ]] || { echo "错误: 全部 batch 档（64/32/16）均失败" >&2; exit 1; }

# ── 结果判定与外推：直接吃 metrics.jsonl（比解析 tqdm 日志可靠）──────────────────
"${PY}" - "${FINAL_RECORD_DIR}" "${STEPS}" "${SAVE_INTERVAL}" "${WARMUP_STEPS}" \
         "${FINAL_BATCH}" "${EPOCH_SAMPLES}" <<'EOF'
import json, math, statistics, sys
record_dir, steps, save_iv, warmup, batch, epoch_samples = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), int(sys.argv[6]))

rows = [json.loads(l) for l in open(f"{record_dir}/metrics.jsonl")]
rows = [r for r in rows if r.get("loss") is not None]
if len(rows) != steps:
    raise SystemExit(f"BAD metrics.jsonl 行数 {len(rows)} != 预期步数 {steps}")

losses = [r["loss"]["dec"] for r in rows]
bad = [v for v in losses if not math.isfinite(v)]
if bad:
    raise SystemExit(f"BAD 出现非有限 loss: {bad[:5]}")
for r in rows[:3] + rows[-3:]:
    assert float.fromhex(r["loss"]["hex"]) == r["loss"]["dec"], "hex 精度回读不一致"

cks = [json.loads(l) for l in open(f"{record_dir}/param_checksums.jsonl")]
expect_ck = len([s for s in range(1, steps) if s % save_iv == 0] + [steps - 1])
if len(cks) != expect_ck:
    raise SystemExit(f"BAD 校验和条数 {len(cks)} != 预期 {expect_ck}")

by_step = {r["step"]: r["wall_time"] for r in rows}
deltas = []
for s in range(warmup + 1, steps):
    if s % save_iv == 0 or (s - 1) % save_iv == 0:
        continue      # 剔除校验和步本身及其下一步（device_get ~14GB 的开销，正式训练没有）
    if s in by_step and s - 1 in by_step:
        deltas.append(by_step[s] - by_step[s - 1])

steady = statistics.median(deltas)
spe = epoch_samples // batch
epoch_s = steady * spe
print(f"OK loss n={len(losses)} min={min(losses):.4f} max={max(losses):.4f} 末值={losses[-1]:.4f}")
print(f"OK 校验和 {len(cks)} 次, 全局摘要末值 {cks[-1]['global_digest'][:16]}…, "
      f"单次耗时中位 {statistics.median(c['checksum_seconds'] for c in cks):.1f}s")
print(f"RESULT batch={batch} 稳态={steady:.3f}s/step (n={len(deltas)}, "
      f"p10={sorted(deltas)[len(deltas)//10]:.3f}, p90={sorted(deltas)[len(deltas)*9//10]:.3f})")
print(f"RESULT steps_per_epoch={spe}  epoch估算={epoch_s:.0f}s ≈ {epoch_s/3600:.2f} 小时")
EOF
RC=$?
[[ "${RC}" -eq 0 ]] || { echo "错误: 结果判定失败" >&2; exit "${RC}"; }

echo "记录文件保留在: ${FINAL_RECORD_DIR}"
echo "EXIT_CODE=0"
echo "BENCH_PASS 基准完成"
