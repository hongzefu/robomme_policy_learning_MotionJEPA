#!/usr/bin/env bash
# ── 本机 2 GPU epoch 时长基准（官方口径 + 一致性检验记录底座）───────────────────
#
# 目标：配置尽可能对齐 scripts/finetune_mme_vla_suite.sh（num-workers 4、
# use_history + framesamp-context；batch 固定 8——官方 64 在 2 卡 OOM，见下），
# 在 2 卡 + NFS turbo 数据上跑 STEPS 步，
# 用稳态 s/step 外推 1 个 epoch（395,289 样本）的时长；同时逐步记录 loss/梯度范数、
# 每 SAVE_INTERVAL 步记录完整 TrainState 摘要与输入 batch 摘要，
# 为 dataloader 改动的一致性检验留底。
# 与官方默认训练的逐项差异及记录文件格式见同目录 README.md。
#
# 身份拆分（v1-gradient-baseline.md P1）：
#   EXP_NAME —— 决定 jax 编译缓存目录（A/B 对拍两轮共用它即共享编译产物与
#               per-fusion autotune 结论）；
#   RUN_TAG  —— 决定记录目录 / 日志 / checkpoint run 目录（每轮各异，避免
#               initialize_checkpoint_dir 的 FileExistsError）。
#
# 可调环境变量：
#   STEPS（≤1200）、WORKERS、WARMUP_STEPS、DATASET_PATH
#   SAVE_INTERVAL   TrainState 摘要间隔；0 = 完全禁摘要（speed 链口径）
#   BATCH_DIGESTS   记输入摘要开关；未设时默认联动 SAVE_INTERVAL（P1b：
#                   SAVE_INTERVAL=0 → 0，否则 1）；显式设 0/1 可覆盖
#   EXTRA_DIGEST_STEPS  逗号分隔附加摘要步（如 299——对齐旧 300 步基线末步摘要）；
#                   实现：给 train 传 --save-interval 1、记录器按
#                   BENCH_DIGEST_INTERVAL=SAVE_INTERVAL + 附加步自选
#   STATE_DUMP_STEPS  逗号分隔 TrainState 数组落盘步（须是摘要步；单步约 14 GB，
#                   落 <记录目录>/state_dump/，G1 逐叶数值裁决的参照）
#   KEEP_JAX_CACHE  1 = 收官保留编译缓存（确定性 A/B 共用缓存用）；默认 0 删除
#   XLA_FLAGS       原样注入训练进程并留档 env.json（确定性档由调用方给定）
#   BENCH_DUMP_IDX  1 = batch_sampler 层 index 记录（S0'，v2 计划 C.1 端到端旁证），
#                   每 batch 追加写 <记录目录>/idx_seq.jsonl；默认 0（现行为不变）
#   BENCH_GPUS      这条 run 用的两张卡（默认 0,1；必须恰好两张，与 --fsdp-devices 2 配套）。
#                   只为在 8 卡机上并行多条 2 卡 run 互不抢显存，不改任何训练口径
#
# runner（P1b）：一律 UV_LINK_MODE=copy uv run（AGENTS 3；同一 .venv 解释器，
# 计算行为不变——由 G0b 重跑 vs 旧 G0 前 300 步逐位对拍实证）。
#
# ⚠ 本机数字按 AGENTS.md 第 13 条只作估算，不作正式吞吐结论；带摘要/确定性档的
#   run 其 util/步时按基线计划红线 B7 禁作任何性能结论。
# ⚠ 不落任何 checkpoint（save_state 已被 bench 入口替换为摘要记录器）。
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/paths.sh"

v1_prepare_dirs
v1_require_venv
v1_require_models 1                       # 需要 tokenizer 与 pi05_base

# runner 收敛 uv run（P1b，AGENTS 3）：同一 .venv 解释器，NFS 上强制 copy 链接
export UV_LINK_MODE=copy
UVPY=(uv run --project "${REPO_ROOT}" python)

DATASET_PATH="${DATASET_PATH:-${GL_DATASET}}"
STEPS="${STEPS:-300}"
WORKERS="${WORKERS:-4}"                   # 官方口径
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"      # TrainState 摘要间隔（不落 ckpt）；0 = 禁摘要
BATCH_DIGESTS_REQUESTED="${BATCH_DIGESTS:-auto}"   # 留档：显式值或 auto（联动）
if [[ -z "${BATCH_DIGESTS:-}" ]]; then    # P1b：未显式设置时联动 SAVE_INTERVAL——
  if [[ "${SAVE_INTERVAL}" -eq 0 ]]; then #   speed 链口径要求两摘要一起关，此前须
    BATCH_DIGESTS=0                       #   调用方手动同时设置、易漏
  else
    BATCH_DIGESTS=1
  fi
fi
EXTRA_DIGEST_STEPS="${EXTRA_DIGEST_STEPS:-}"   # 附加摘要步（逗号分隔；P1b）
BENCH_GPUS="${BENCH_GPUS:-0,1}"                # 恰好两张卡（fsdp 2）；8 卡机上多条 run 并行时各给一对
if [[ ! "${BENCH_GPUS}" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "错误: BENCH_GPUS 必须是恰好两张卡的逗号分隔列表(如 0,1), 实为 '${BENCH_GPUS}'" >&2
  exit 1
fi
STATE_DUMP_STEPS="${STATE_DUMP_STEPS:-}"       # TrainState 数组落盘步（逗号分隔；P1b）
KEEP_JAX_CACHE="${KEEP_JAX_CACHE:-0}"     # 1 = 保留编译缓存供下一轮共用
WARMUP_STEPS="${WARMUP_STEPS:-50}"        # 稳态统计丢弃的头部步数（JIT 编译 + worker 起步）
# epoch 样本数不再硬编码（motion-memory-plan.md 2.8 / R14）：packed 根读 meta/store_meta.json.num_exec_samples，
# 旧 source 根读 meta/stats.json.execution_samples；两者同时存在却不等、或都读不到即报错
_es_sm=""; _es_st=""
[[ -f "${DATASET_PATH}/meta/store_meta.json" ]] && _es_sm="$(jq -r '.num_exec_samples' "${DATASET_PATH}/meta/store_meta.json")"
[[ -f "${DATASET_PATH}/meta/stats.json" ]] && _es_st="$(jq -r '.execution_samples' "${DATASET_PATH}/meta/stats.json")"
if [[ -n "${_es_sm}" && -n "${_es_st}" && "${_es_sm}" != "${_es_st}" ]]; then
  echo "错误: epoch 样本数两处不等: store_meta=${_es_sm} stats=${_es_st}" >&2; exit 1
fi
EPOCH_SAMPLES="${_es_sm:-${_es_st}}"
[[ "${EPOCH_SAMPLES}" =~ ^[0-9]+$ ]] || {
  echo "错误: epoch 样本数无法从 ${DATASET_PATH}/meta/{store_meta,stats}.json 读出（得到: '${EPOCH_SAMPLES}'）" >&2; exit 1; }
# history config 只接受 closed / open 两个精确文件名并原样写入记录（motion-memory-plan.md 2.1）
HISTORY_CONFIG="${HISTORY_CONFIG:-perceptual-framesamp-context.yaml}"
case "${HISTORY_CONFIG}" in
  perceptual-framesamp-context.yaml|perceptual-framesamp-context-motion.yaml) ;;
  *) echo "错误: HISTORY_CONFIG 必须是 perceptual-framesamp-context.yaml 或 perceptual-framesamp-context-motion.yaml, 当前 ${HISTORY_CONFIG}" >&2; exit 1 ;;
esac
NORM_STATS="${TRAIN_ASSETS}/mme_vla_suite/robomme/norm_stats.json"
BENCH_ROOT="${V1_STORE}/bench/2gpu-epoch-bench"

BENCH_DUMP_IDX="${BENCH_DUMP_IDX:-0}"     # S0'：batch_sampler 层 index 记录开关

# S0'：preflight 兼容 packed 库——legacy 库有 meta/stats.json，打包库有 meta/store_meta.json
[[ -f "${DATASET_PATH}/meta/stats.json" || -f "${DATASET_PATH}/meta/store_meta.json" ]] || {
  echo "错误: 数据集不存在或未 finalize: ${DATASET_PATH}/meta/ 下 stats.json 与 store_meta.json 均缺失" >&2; exit 1; }
[[ -f "${NORM_STATS}" ]] || {
  echo "错误: norm stats 缺失: ${NORM_STATS}（用 scripts/compute_norm_stats.py 先生成）" >&2; exit 1; }

# batch 固定为 8：2026-08-24 实测 2 卡下唯一确认可跑的档位——64/32/16 全部 OOM
# （失败张量 17.62/12.61/10.38 GiB，激活固定底座约 8 GiB，每卡还驻留约 28 GB
# 参数+优化器+EMA 状态；官方 4 卡每卡状态减半故能跑 64），batch 8 以 300 步
# 全程验证通过（稳态 1.060 s/step）。改档位属超参变更：须按 AGENTS.md 第 10 条
# 先与用户确认，且需重新实测显存，不提供环境变量覆盖。
BATCH=8
(( STEPS * BATCH < EPOCH_SAMPLES )) || {
  echo "错误: 单 epoch 约束违反: STEPS×BATCH=$((STEPS * BATCH)) ≥ EPOCH_SAMPLES=${EPOCH_SAMPLES}" >&2; exit 1; }
EXP_NAME="${EXP_NAME:-v1-2gpu-epoch-bench-b${BATCH}}"
RUN_TAG="${RUN_TAG:-${EXP_NAME}}"
CKPT_BASE="${TRAIN_RUNS}/${RUN_TAG}"      # 按 RUN_TAG 分目录：共用 EXP_NAME 的两轮不撞名
CKPT_DIR="${CKPT_BASE}/mme_vla_suite/${EXP_NAME}"
RECORD_DIR="${BENCH_ROOT}/${RUN_TAG}"
LOG="${LOGS_DIR}/${RUN_TAG}.log"

# jax 编译缓存收敛进 v1-store（AGENTS 14，不动 train.py、不覆盖 HOME）：
# train.main 硬编码写 ~/.cache/jax_<exp_name>，用软链把它指到 v1-store/cache/jax/
JAX_CACHE_DIR="${CACHE_DIR}/jax/${EXP_NAME}"
JAX_CACHE_LINK="${HOME}/.cache/jax_${EXP_NAME}"
mkdir -p "${JAX_CACHE_DIR}" "${HOME}/.cache"
[[ -e "${JAX_CACHE_LINK}" && ! -L "${JAX_CACHE_LINK}" ]] && {
  echo "错误: ${JAX_CACHE_LINK} 已存在且不是软链, 拒绝覆盖" >&2; exit 1; }
ln -sfn "${JAX_CACHE_DIR}" "${JAX_CACHE_LINK}"

[[ -e "${CKPT_DIR}" ]] && {
  echo "错误: run 目录已存在, 禁止 overwrite: ${CKPT_DIR}" >&2; exit 1; }
[[ -e "${RECORD_DIR}" ]] && {
  echo "错误: 记录目录已存在, 禁止覆盖既有记录: ${RECORD_DIR}" >&2; exit 1; }
mkdir -p "${RECORD_DIR}"

# SAVE_INTERVAL=0（speed 链口径）：入口层禁摘要；train.main 的 step%interval 不能吃 0，
# 传一个大于步数上限的值让周期触发失效（末步触发由 BENCH_CHECKSUM=0 变 no-op）。
# EXTRA_DIGEST_STEPS/STATE_DUMP_STEPS（P1b）：给 train 传 --save-interval 1（save
# 分支只调已被换成记录器的 save_state，每步空调用零开销），记录器按
# BENCH_DIGEST_INTERVAL=SAVE_INTERVAL + 附加步自选摘要步。
BENCH_DIGEST_INTERVAL=""
if [[ "${SAVE_INTERVAL}" -eq 0 ]]; then
  [[ -n "${EXTRA_DIGEST_STEPS}${STATE_DUMP_STEPS}" ]] && {
    echo "错误: SAVE_INTERVAL=0（禁摘要）与 EXTRA_DIGEST_STEPS/STATE_DUMP_STEPS 互斥" >&2
    exit 1; }
  SAVE_INTERVAL_ARG=1000000
  BENCH_CHECKSUM=0
elif [[ -n "${EXTRA_DIGEST_STEPS}" || -n "${STATE_DUMP_STEPS}" ]]; then
  SAVE_INTERVAL_ARG=1
  BENCH_CHECKSUM=1
  BENCH_DIGEST_INTERVAL="${SAVE_INTERVAL}"
else
  SAVE_INTERVAL_ARG="${SAVE_INTERVAL}"
  BENCH_CHECKSUM=1
fi
STATE_DUMP_DIR=""
[[ -n "${STATE_DUMP_STEPS}" ]] && STATE_DUMP_DIR="${RECORD_DIR}/state_dump"

# 训练命令的唯一真值源：env.json 的 argv 与实际执行的是同一个数组，不再手抄字面量
ARGS=(
  "${REPO_ROOT}/scripts/training/g0/bench_train_steps.py" mme_vla_suite
  --exp-name "${EXP_NAME}"
  --assets-base-dir "${TRAIN_ASSETS}"
  --checkpoint-base-dir "${CKPT_BASE}"
  --batch-size "${BATCH}"
  --num-workers "${WORKERS}"
  --num-train-steps "${STEPS}"
  --log-interval 1
  --save-interval "${SAVE_INTERVAL_ARG}"
  --seed 42
  --fsdp-devices 2
  --dataset-path "${DATASET_PATH}"
  --weight-loader.params-path "${MODELS_DIR}/openpi-assets/checkpoints/pi05_base/params"
  --model.use-history
  --model.history-config "${HISTORY_CONFIG}"
  --no-wandb-enabled
)

# 环境留档：将来一致性 A/B 的对照 run 必须逐项同设（见 README.md）。
# argv 直接来自上面的 ARGS 数组，经位置参数传入（不能走管道——heredoc 已占用 stdin）
"${UVPY[@]}" - "$RECORD_DIR" "${ARGS[@]}" <<EOF
import json, subprocess, sys, platform
import jax
argv = sys.argv[2:]
d = {
    "argv": argv,
    "argv_batch": ${BATCH}, "steps": ${STEPS}, "workers": ${WORKERS},
    "exp_name": "${EXP_NAME}", "run_tag": "${RUN_TAG}",
    "save_interval_requested": ${SAVE_INTERVAL},
    "save_interval_effective": ${SAVE_INTERVAL_ARG},
    "batch_digests": ${BATCH_DIGESTS},
    "batch_digests_requested": "${BATCH_DIGESTS_REQUESTED}",
    "extra_digest_steps": "${EXTRA_DIGEST_STEPS}",
    "state_dump_steps": "${STATE_DUMP_STEPS}",
    "keep_jax_cache": ${KEEP_JAX_CACHE},
    "seed": 42, "fsdp_devices": 2,
    "history_config": "${HISTORY_CONFIG}",
    "epoch_samples": ${EPOCH_SAMPLES},
    "dataset_path": "${DATASET_PATH}",
    "jax_cache_dir": "${JAX_CACHE_DIR}",
    "git_head": subprocess.run(["git", "-C", "${REPO_ROOT}", "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
    "git_dirty": bool(subprocess.run(["git", "-C", "${REPO_ROOT}", "status", "--porcelain"],
                                     capture_output=True, text=True).stdout.strip()),
    "XLA_FLAGS": "${XLA_FLAGS:-}",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
    "CUDA_VISIBLE_DEVICES": "${BENCH_GPUS}",
    "hostname": platform.node(),
    "python": sys.version,
    "jax": jax.__version__,
    "nvidia_smi": subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().splitlines(),
    # ── S0'：framesamp packed 库 provenance（v2 计划 D 节清单；commitV4.1 起
    #    packed 为唯一数据链，backend 字段已随 MMEVLA_DATA_BACKEND 三态一并删除）──
    "MMEVLA_FRAMESAMP_VERIFY": "${MMEVLA_FRAMESAMP_VERIFY:-}",
    "MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED": "${MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED:-}",
    "MMEVLA_FRAMESAMP_ALLOW_SUBSET": "${MMEVLA_FRAMESAMP_ALLOW_SUBSET:-}",
    "MMEVLA_FRAMESAMP_SOURCE": "${MMEVLA_FRAMESAMP_SOURCE:-}",
    "MMEVLA_FRAMESAMP_MANIFEST": "${MMEVLA_FRAMESAMP_MANIFEST:-}",
    "MMEVLA_FRAMESAMP_LOCAL_CACHE": "${MMEVLA_FRAMESAMP_LOCAL_CACHE:-}",
    "BENCH_DUMP_IDX": "${BENCH_DUMP_IDX}",
}
# resolved 双根（v2 计划 B.4）：打包库读 store_meta.json（env 可覆盖），legacy 库
# 源库根即 dataset_path 自身、无独立清单依赖
import hashlib, pathlib
_sm = pathlib.Path("${DATASET_PATH}") / "meta" / "store_meta.json"
if _sm.exists():
    _raw = _sm.read_bytes()
    _meta = json.loads(_raw)
    d["store_meta_sha256"] = hashlib.sha256(_raw).hexdigest()
    d["store_meta_status"] = _meta.get("status")
    d["manifest_sha256"] = _meta.get("manifest_sha256")
    d["source_dataset_root_resolved"] = ("${MMEVLA_FRAMESAMP_SOURCE:-}"
                                        or _meta.get("source_dataset_root"))
    d["manifest_path_resolved"] = ("${MMEVLA_FRAMESAMP_MANIFEST:-}"
                                   or _meta.get("manifest_path"))
else:
    d["store_meta_sha256"] = None
    d["store_meta_status"] = None
    d["manifest_sha256"] = None
    d["source_dataset_root_resolved"] = "${DATASET_PATH}"
    d["manifest_path_resolved"] = None
json.dump(d, open("$RECORD_DIR/env.json", "w"), indent=2, ensure_ascii=False)
EOF

# 环境指纹并入 env.json（preflight 同一套采集代码，保证跨期可比对；
# 环境变量与训练进程逐项同口径）。
# S6 补：packed 库无 data/ 目录（pkl/原图仍从源库读），dataset_spot 指纹一律
# 锚在源库——store_meta.source_dataset_root（可被 MMEVLA_FRAMESAMP_SOURCE 覆盖），
# 与 check 侧「vs G0 的源数据未动」口径一致
DUMP_DATASET="${DATASET_PATH}"
if [[ -f "${DATASET_PATH}/meta/store_meta.json" ]]; then
  DUMP_DATASET="$("${UVPY[@]}" - "${DATASET_PATH}" <<'PYEOF'
import json, os, sys
meta = json.load(open(os.path.join(sys.argv[1], "meta", "store_meta.json")))
print(os.environ.get("MMEVLA_FRAMESAMP_SOURCE") or meta["source_dataset_root"])
PYEOF
)"
fi
XLA_FLAGS="${XLA_FLAGS:-}" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
CUDA_VISIBLE_DEVICES="${BENCH_GPUS}" \
"${UVPY[@]}" "${REPO_ROOT}/scripts/training/g0/check_baseline_env.py" dump \
  --record-dir "${RECORD_DIR}" --dataset "${DUMP_DATASET}"

echo "=== 2 GPU epoch 基准: ${RUN_TAG} (exp=${EXP_NAME}, ${STEPS} steps, batch ${BATCH}, workers ${WORKERS}) ==="
echo "  数据集: ${DATASET_PATH}"
echo "  记录目录: ${RECORD_DIR}"
echo "  XLA_FLAGS: ${XLA_FLAGS:-<未设>}"
set +e
(
  set -e
  cd "${REPO_ROOT}"
  BENCH_RECORD_DIR="${RECORD_DIR}" \
  BENCH_CHECKSUM="${BENCH_CHECKSUM}" \
  BENCH_BATCH_DIGESTS="${BATCH_DIGESTS}" \
  BENCH_DIGEST_INTERVAL="${BENCH_DIGEST_INTERVAL}" \
  BENCH_EXTRA_DIGEST_STEPS="${EXTRA_DIGEST_STEPS}" \
  BENCH_STATE_DUMP_STEPS="${STATE_DUMP_STEPS}" \
  BENCH_STATE_DUMP_DIR="${STATE_DUMP_DIR}" \
  BENCH_DUMP_IDX="${BENCH_DUMP_IDX}" \
  CUDA_VISIBLE_DEVICES="${BENCH_GPUS}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  XLA_FLAGS="${XLA_FLAGS:-}" \
  PYTHONUNBUFFERED=1 \
  WANDB_MODE=disabled \
  "${UVPY[@]}" "${ARGS[@]}"
) 2>&1 | tee "${LOG}"
RC="${PIPESTATUS[0]}"
set -e

# 跑完（无论成败）清理 run 目录空壳；编译缓存按 KEEP_JAX_CACHE 处置（软链总是拆掉）
# ⚠ BENCH_SAVE_FINAL_CKPT=1（T3 两侧 run）时 run 目录里有最终 checkpoint 999 与配置快照，必须保留、不得清理
#   （2026-09-03 motion-t3-closed 跑到一半发现此处会连 999 一起 rm，临时以硬链接看门狗保出；本条修补对之后的 run 生效）
if [[ -e "${CKPT_DIR}" ]]; then
  if [[ "${BENCH_SAVE_FINAL_CKPT:-0}" == "1" ]]; then
    echo "保留 run 目录（BENCH_SAVE_FINAL_CKPT=1，含最终 checkpoint）: ${CKPT_DIR}"
  else
    case "${CKPT_DIR}" in
      "${TRAIN_RUNS}/${RUN_TAG}/mme_vla_suite/${EXP_NAME}") rm -rf -- "${CKPT_BASE}" ;;
      *) echo "错误: 拒绝清理非预期路径 ${CKPT_DIR}" >&2; exit 1 ;;
    esac
  fi
fi
rm -f -- "${JAX_CACHE_LINK}"
if [[ "${KEEP_JAX_CACHE}" != "1" ]]; then
  case "${JAX_CACHE_DIR}" in
    "${CACHE_DIR}/jax/${EXP_NAME}") rm -rf -- "${JAX_CACHE_DIR}" ;;
    *) echo "错误: 拒绝清理非预期缓存路径 ${JAX_CACHE_DIR}" >&2; exit 1 ;;
  esac
fi

# 任何失败（含 OOM）直接 fail-loud，不降档不重试——batch 8 是实测钉死的档位，
# 在它上面再出 OOM 说明环境变了（驱动/常驻占用/代码），须人工排查而不是掩盖
if [[ "${RC}" -ne 0 ]]; then
  echo "错误: 基准失败（退出码 ${RC}），人工排查: ${LOG}" >&2
  # 失败记录原子改名保留（commitV4.1，计划二节第 9 条）：它是定位分叉/失败的
  # 唯一证据，禁止删除；后缀取现存最大值 +1，绝不覆盖历史失败记录
  _fn=1
  while [[ -e "${RECORD_DIR}.failed-${_fn}" ]]; do _fn=$((_fn + 1)); done
  mv -- "${RECORD_DIR}" "${RECORD_DIR}.failed-${_fn}"
  echo "失败记录已保留: ${RECORD_DIR}.failed-${_fn}" >&2
  exit "${RC}"
fi

# 缓存事件计数（run_meta.json，bench 入口落盘）并进 env.json
"${UVPY[@]}" - "${RECORD_DIR}" <<'EOF'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
env = json.load(open(d / "env.json"))
meta = json.load(open(d / "run_meta.json"))
env["run_meta"] = meta
assert env["argv"] == meta["argv"][1:] or env["argv"] == meta["argv"], \
    f"env.json argv 与进程实际 argv 不一致: {env['argv'][:2]} vs {meta['argv'][:3]}"
json.dump(env, open(d / "env.json", "w"), indent=2, ensure_ascii=False)
print("OK run_meta 并入 env.json, 缓存事件:",
      {k: v for k, v in meta["monitoring_event_counts"].items() if "cache" in k or "compil" in k})
EOF

# ── 结果判定与外推：直接吃 metrics.jsonl（比解析 tqdm 日志可靠）──────────────────
"${UVPY[@]}" - "${RECORD_DIR}" "${STEPS}" "${SAVE_INTERVAL}" "${WARMUP_STEPS}" \
         "${BATCH}" "${EPOCH_SAMPLES}" "${BENCH_CHECKSUM}" "${BATCH_DIGESTS}" \
         "${EXTRA_DIGEST_STEPS}" "${STATE_DUMP_STEPS}" <<'EOF'
import json, math, os, statistics, sys
record_dir, steps, save_iv, warmup, batch, epoch_samples, ck_on, dg_on = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]))
extra = {int(s) for s in sys.argv[9].split(",") if s.strip()}
dump_steps = {int(s) for s in sys.argv[10].split(",") if s.strip()}
# save_iv 是请求值（SAVE_INTERVAL 原值）；摘要步集合 = 0/末步/间隔倍数/附加步
ck_steps = ({0, steps - 1} | extra
            | ({s for s in range(1, steps) if s % save_iv == 0} if save_iv > 0 else set()))

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

if ck_on:
    cks = [json.loads(l) for l in open(f"{record_dir}/param_checksums.jsonl")]
    expect_steps = sorted(ck_steps)
    got_steps = [c["step"] for c in cks]
    if got_steps != expect_steps:
        raise SystemExit(f"BAD 摘要步序列 {got_steps} != 预期 {expect_steps}")
    assert all("state_digest" in c for c in cks), "BAD 缺 state_digest（完整 TrainState 摘要）"
    print(f"OK TrainState 摘要 {len(cks)} 次 @steps={got_steps}, "
          f"末值 state={cks[-1]['state_digest'][:16]}…, "
          f"单次耗时中位 {statistics.median(c['checksum_seconds'] for c in cks):.1f}s")
    for s in sorted(dump_steps):   # TrainState 数组落盘完整性（P1b）
        meta_p = f"{record_dir}/state_dump/state_step_{s}.json"
        bin_p = f"{record_dir}/state_dump/state_step_{s}.bin"
        if not (os.path.exists(meta_p) and os.path.exists(bin_p)):
            raise SystemExit(f"BAD STATE_DUMP 步 {s} 缺产物: {meta_p}")
        m = json.load(open(meta_p))
        if os.path.getsize(bin_p) != m["total_bytes"]:
            raise SystemExit(f"BAD STATE_DUMP 步 {s} bin 大小与 meta 不符")
    if dump_steps:
        print(f"OK TrainState 数组落盘 {len(dump_steps)} 步 @{sorted(dump_steps)}")
else:
    assert not os.path.exists(f"{record_dir}/param_checksums.jsonl"), \
        "BAD BENCH_CHECKSUM=0 却写了 param_checksums.jsonl"
    print("OK speed 口径: 无 TrainState 摘要")

if dg_on:
    dgs = [json.loads(l) for l in open(f"{record_dir}/batch_digests.jsonl")]
    expect_dg = sorted({0, 1, 2} | ck_steps)
    got_dg = [d["step"] for d in dgs]
    if got_dg != expect_dg:
        raise SystemExit(f"BAD 输入摘要步序列 {got_dg} != 预期 {expect_dg}")
    # P1b schema 2：canonical 双口径 + 逐步样本 index + index 全序列
    for d_ in dgs:
        if d_.get("schema") != 2 or "batch_digest_canonical" not in d_ \
                or "per_key_canonical" not in d_:
            raise SystemExit(f"BAD 步 {d_['step']} 缺 canonical 摘要（schema 2）")
        if not d_.get("sample_indices"):
            raise SystemExit(f"BAD 步 {d_['step']} 缺 sample_indices")
    seq = json.load(open(f"{record_dir}/index_sequence.json"))
    if seq["n"] < steps * batch:
        raise SystemExit(f"BAD index 序列长度 {seq['n']} < steps×batch {steps*batch}")
    print(f"OK 输入摘要 {len(dgs)} 次 @steps={got_dg}, raw 末值 {dgs[-1]['batch_digest'][:16]}…, "
          f"canonical 末值 {dgs[-1]['batch_digest_canonical'][:16]}…, "
          f"index 序列 {seq['n']} 个 sha={seq['indices_sha256'][:16]}…")
else:
    assert not os.path.exists(f"{record_dir}/batch_digests.jsonl"), \
        "BAD BATCH_DIGESTS=0 却写了 batch_digests.jsonl"
    print("OK speed 口径: 无输入摘要")

by_step = {r["step"]: r["wall_time"] for r in rows}
deltas = []
for s in range(warmup + 1, steps):
    if s in ck_steps or s - 1 in ck_steps:
        continue      # 剔除摘要步本身及其下一步（device_get 的开销，正式训练没有）
    if s in by_step and s - 1 in by_step:
        deltas.append(by_step[s] - by_step[s - 1])

steady = statistics.median(deltas)
spe = epoch_samples // batch
epoch_s = steady * spe
print(f"OK loss n={len(losses)} min={min(losses):.4f} max={max(losses):.4f} 末值={losses[-1]:.4f}")
print(f"RESULT batch={batch} 稳态={steady:.3f}s/step (n={len(deltas)}, "
      f"p10={sorted(deltas)[len(deltas)//10]:.3f}, p90={sorted(deltas)[len(deltas)*9//10]:.3f})")
print(f"RESULT steps_per_epoch={spe}  epoch估算={epoch_s:.0f}s ≈ {epoch_s/3600:.2f} 小时")
EOF
RC=$?
[[ "${RC}" -eq 0 ]] || { echo "错误: 结果判定失败" >&2; exit "${RC}"; }

echo "记录文件保留在: ${RECORD_DIR}"
echo "EXIT_CODE=0"
echo "BENCH_PASS 基准完成"
