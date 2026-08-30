#!/usr/bin/env bash
# ── 第二/三层验收：跨架构逐 key 分类对拍 + 下游等价 ────────────────────────────
#
# 前置：第一层（legacy/step_local_baseline.sh，已归档）已 PASS——那层证明了分片实现与未改动 builder
# 逐字节相同，所以本机跑 build_shard.py 产出的 ref-crossarch 才有资格当「本地真值」，
# 也才能摆脱未改动 builder「只能取前缀」的限制，在全 1600 里分层随机抽样。
#
# 第二层按「是否过 GPU」逐 key 分类，不搞一刀切阈值：
#   kept_indices / data/*.pkl / state_emb  → 零容差逐位（它们压根没碰 GPU）
#   pos_emb_*                              → 走 JAX，桶归属由实测判定
#   image_emb_*（bf16）                    → 量化等价：位相同占比 / 最大 ULP 差 / 余弦
# 第三层比训练实际怎么用它：选帧索引与 mask 必须逐位相同。
#
# ⚠ 跨架构逐位一致是做不到的（greatlakes.md 已实证 A40 vs Ada 不逐位、determinism
#   三档全部无效）。所以交付按「换合同」口径：集群产物自成一份数据集，每个分片在自己的
#   sidecar 里记硬件/软件指纹、finalize 断言跨片同源，**机制上杜绝**与本地字节混用；
#   验收标准是等价判据，而不是「和本地一模一样」。
#   ⚠ provenance.json 的**顶层**字段描述的是 finalize 节点，构建节点的指纹在
#     shard_fingerprints 里（两层语义见 README「provenance 的两层语义」）。
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"

v1_prepare_dirs
v1_require_venv
v1_require_models 0

STRAT_N="${STRAT_N:-10}"
SUBSET_STRAT="${V1_STORE}/subset_strat${STRAT_N}.json"
REPORT_DIR="${V1_STORE}/reports"
JOB="${JOB:-v1-4task-build}"
mkdir -p "${REPORT_DIR}"

export CUDA_VISIBLE_DEVICES="${BENCH_GPU:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

echo "=== [1/5] finalize 判定行 ==="
FINAL_LOG="$(ls -t "${LOGS_DIR}"/${JOB}-final-*.log 2>/dev/null | head -1 || true)"
if [[ -z "${FINAL_LOG}" ]]; then
  echo "✗ 找不到 finalize 日志。注意：任一分片失败会让 finalize 因 afterok 不满足被"
  echo "  kill_invalid_depend 自动 CANCELLED，此时**连日志文件都不会生成**——用 sacct 判死。"
  exit 1
fi
echo "  日志：${FINAL_LOG}"
grep -q "FINALIZE_EXIT_CODE=0" "${FINAL_LOG}" || { echo "✗ finalize 退出码非 0"; exit 1; }
grep -q "同架构复算全部逐位一致 PASS" "${FINAL_LOG}" || { echo "✗ 同架构零容差抽检判定行缺失"; exit 1; }
echo "  ✓ finalize 全绿（输入同源 / 完整性 / provenance / 同架构零容差抽检）"

echo "=== [2/5] 本地真值 ref-crossarch（分层随机 ${STRAT_N}/任务 + 边界样本）==="
[[ -f "${SUBSET_STRAT}" ]] || "${PY}" "${V1_SCRIPT_DIR}/scan_manifest.py" sample \
    --manifest "${MANIFEST_PATH}" --out "${SUBSET_STRAT}" --mode strat --n "${STRAT_N}"
"${PY}" "${V1_SCRIPT_DIR}/build_shard.py" \
    --manifest "${MANIFEST_PATH}" --raw_dir "${RAW_H5_DIR}" --out "${REF_CROSSARCH}" \
    --subset "${SUBSET_STRAT}" --shard_idx 0 --num_shards 1 --resume --report_every 1000

echo "=== [3/5] 第二层：跨架构逐 key 分类对拍 ==="
"${PY}" "${V1_SCRIPT_DIR}/compare_datasets.py" --mode crossarch \
    --manifest "${MANIFEST_PATH}" \
    --a_lib "${REF_CROSSARCH}" --b_lib "${GL_DATASET}" \
    --subset "${SUBSET_STRAT}" --steps_per_episode "${STEPS_PER_EP:-24}" \
    --min_cosine "${MIN_COSINE:-0.999}" \
    --min_p5_cosine "${MIN_P5_COSINE:-0.9999}" \
    --max_err_floor_rel "${MAX_ERR_FLOOR_REL:-0.05}" \
    --report "${REPORT_DIR}/layer2_crossarch.json"

echo "=== [4/5] 第三层：下游等价（prepare_frame_sampling 的选帧索引与 mask）==="
"${PY}" "${V1_SCRIPT_DIR}/compare_datasets.py" --mode downstream \
    --manifest "${MANIFEST_PATH}" \
    --a_lib "${REF_CROSSARCH}" --b_lib "${GL_DATASET}" \
    --subset "${SUBSET_STRAT}" \
    --min_cosine "${MIN_COSINE:-0.999}" \
    --min_p5_cosine "${MIN_P5_COSINE:-0.9999}" \
    --max_err_floor_rel "${MAX_ERR_FLOOR_REL:-0.05}" \
    --report "${REPORT_DIR}/layer3_downstream.json"

echo "=== [5/5] 全量库自洽复核 ==="
STATS_JSON="${GL_DATASET}/meta/stats.json"
PROV_JSON="${GL_DATASET}/meta/provenance.json"
# ⚠ 原写法是 `jq -e ... && echo ✓`：jq 失败时它只是不打印那行 ✓，脚本继续往下跑到
#   最后无条件打印 VERIFY_PASS —— 唯一一道 stats 数值合理性判据被 bash 语义吞掉了。
#   一律改成 if/else + exit 1。
if jq -e '.execution_samples > 0 and .total_samples > 0' "${STATS_JSON}" >/dev/null 2>&1; then
  echo "  ✓ stats.json 合理：$(jq -c . "${STATS_JSON}")"
else
  echo "  ✗ stats.json 缺失、非法 JSON 或数值不合理：${STATS_JSON}"; exit 1
fi

if ! jq -e 'has("host") and has("gpu_device_kind") and has("jax") and has("git_commit")' \
      "${PROV_JSON}" >/dev/null 2>&1; then
  echo "  ✗ provenance.json 缺失、非法 JSON 或缺必需字段：${PROV_JSON}"; exit 1
fi
# 顶层字段描述的是 **finalize 节点**，不是产出数据的分片节点（见 finalize_checks.py
# 的模块 docstring）。历史上这里把顶层 resource_tier 当构建档位打印，实际打的是
# finalize job 的 4CPU/32G，而分片是 2CPU/24G —— 档位改从 shard_fingerprints 取。
jq -r '"  provenance(finalize 节点): host=\(.host) gpu=\(.gpu_device_kind) jax=\(.jax) commit=\(.git_commit[0:12])"' \
  "${PROV_JSON}"
# 分片指纹是 2026-08-24 才引入的；旧库没有它只提示、不判死，否则本来全绿的复测会
# 因为一个装饰性字段失败。
if jq -e 'has("shard_fingerprints")' "${PROV_JSON}" >/dev/null 2>&1; then
  jq -r '"  provenance(构建分片): gpu=\(.shard_fingerprints.gpu_device_kind) jax=\(.shard_fingerprints.jax) commit=\(.shard_fingerprints.git_commit[0:12]) 节点=\(.shard_fingerprints.hosts|length) 个 档位=\(.shard_fingerprints.resource_tiers[0].cpus_per_task)CPU/\(.shard_fingerprints.resource_tiers[0].mem_per_node_mb)M"' \
    "${PROV_JSON}"
else
  echo "  · 该库产出于分片指纹字段引入之前，无 shard_fingerprints；"
  echo "    注意顶层 resource_tier 是 finalize job 的档位，不是分片档位（分片实为 2CPU/24G）"
fi
du -sh "${GL_DATASET}" 2>/dev/null || true

echo
echo "VERIFY_PASS  第二、三层全绿。"
echo "下一步：训练侧烟测见 scripts/training/bench/run_2gpu_epoch_bench.sh"
