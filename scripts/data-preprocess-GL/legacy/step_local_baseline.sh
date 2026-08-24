#!/usr/bin/env bash
# ── 第一层：清零「分片」这个变量（本机，同机同架构，零容差）─────────────────────
# 同一批 episode 跑两遍：
#   参照系 = 仓库**未改动的** scripts/build_dataset.py --max_episodes 3
#   被测   = 本目录的 build_shard.py --num_shards 4
# 硬件变量为零 ⇒ 两份产物必须**逐字节相同**，没有任何容差。
#
# 为什么取 --max_episodes 3（12 个 episode）而不是 1（4 个）：分片改造最容易错的地方
# 是 exec_sample_id / global_episode_idx 的跨文件累加偏移，而每个任务的 episode_0
# 偏移基本是平凡的——只拿它们对拍，恰好避开了最该测的部分。12 个 episode 里有 11 个
# 带非零偏移，且 4 个分片能压到分片边界。
#
# 这一层过了，「分片」被彻底排除，后续差异只可能来自硬件；同时 build_shard.py 取得
# **「本地真值」资格**——第二/三层的跨架构对拍就不再受未改动 builder「只能取前缀」
# 的限制，可以在全 1600 里分层随机抽样。
#
# ⚠ 已归档进 legacy/：第一层已 PASS（12 episode / 3,862 步逐字节相同，见
#   docs/v1-gl-dataset-consistency-report.md 第 6.1 节）。只要 build_shard.py /
#   scan_manifest.py 不再改动，结论持续有效；改动后须重跑本脚本重新取得资格。
#   清单生成一步已上移为 step0_setup_turbo.sh manifest。
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../paths.sh"

v1_prepare_dirs
v1_require_cmd jq
v1_require_venv
v1_require_models 0

PREFIX_N="${PREFIX_N:-3}"
STRAT_N="${STRAT_N:-10}"
GPU="${BENCH_GPU:-0}"
SUBSET_PREFIX="${V1_STORE}/subset_prefix${PREFIX_N}.json"
SUBSET_STRAT="${V1_STORE}/subset_strat${STRAT_N}.json"
UNTOUCHED_LOG="${LOGS_DIR}/ref_untouched_build.log"
REPORT_DIR="${V1_STORE}/reports"
mkdir -p "${REPORT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

echo "=== [1/6] 分片清单（若已存在则复用）==="
if [[ -f "${MANIFEST_PATH}" ]]; then
  echo "  复用 ${MANIFEST_PATH}"
else
  "${PY}" "${V1_SCRIPT_DIR}/scan_manifest.py" build \
      --raw_dir "${RAW_H5_DIR}" --out "${MANIFEST_PATH}" --num_shards 8
fi

echo "=== [2/6] 抽 episode 子集 ==="
"${PY}" "${V1_SCRIPT_DIR}/scan_manifest.py" sample \
    --manifest "${MANIFEST_PATH}" --out "${SUBSET_PREFIX}" --mode prefix --n "${PREFIX_N}"
"${PY}" "${V1_SCRIPT_DIR}/scan_manifest.py" sample \
    --manifest "${MANIFEST_PATH}" --out "${SUBSET_STRAT}" --mode strat --n "${STRAT_N}"

echo "=== [3/6] 参照系：未改动的 build_dataset.py --max_episodes ${PREFIX_N} ==="
# 原版 builder 在 __init__ 里 shutil.rmtree(输出目录)，所以这里先做一道路径护栏：
# 只允许它指向 datasets/ 下的 ref-untouched，绝不可能误删别的东西。
case "${REF_UNTOUCHED}" in
  "${DATASETS_DIR}"/ref-untouched) ;;
  *) echo "错误: 参照系输出路径异常, 拒绝让原版 builder 执行 rmtree: ${REF_UNTOUCHED}" >&2; exit 1 ;;
esac
"${PY}" "${REPO_ROOT}/scripts/build_dataset.py" \
    --dataset_type robomme_pkl \
    --raw_data_path "${RAW_H5_DIR}" \
    --preprocessed_data_path "${REF_UNTOUCHED}" \
    --max_episodes "${PREFIX_N}" 2>&1 | tee "${UNTOUCHED_LOG}"

echo "=== [4/6] 被测：build_shard.py --num_shards 4（同一批 episode）==="
rm -rf "${REF_SHARD}"
for i in 0 1 2 3; do
  echo "--- shard ${i}/4 ---"
  "${PY}" "${V1_SCRIPT_DIR}/build_shard.py" \
      --manifest "${MANIFEST_PATH}" --raw_dir "${RAW_H5_DIR}" --out "${REF_SHARD}" \
      --subset "${SUBSET_PREFIX}" --shard_idx "${i}" --num_shards 4 --report_every 500
done

echo "=== [5/6] 第一层判定：逐字节相同（零容差）==="
"${PY}" "${V1_SCRIPT_DIR}/compare_datasets.py" --mode bitexact \
    --manifest "${MANIFEST_PATH}" \
    --a_lib "${REF_UNTOUCHED}" --b_lib "${REF_SHARD}" \
    --a_untouched_log "${UNTOUCHED_LOG}" --a_max_episodes "${PREFIX_N}" \
    --raw_dir "${RAW_H5_DIR}" --steps_per_episode 0 \
    --report "${REPORT_DIR}/layer1_bitexact.json"

echo "=== [6/6] 产物体积实测（校准 588 KB/step 的估算）==="
du -sh "${REF_UNTOUCHED}" "${REF_SHARD}" || true
"${PY}" - "${MANIFEST_PATH}" "${SUBSET_PREFIX}" "${REF_SHARD}" <<'PYCAL'
import json, subprocess, sys
manifest = json.load(open(sys.argv[1]))
subset = set(json.load(open(sys.argv[2]))["global_episode_idx"])
steps = sum(e["num_timesteps"] for e in manifest["episodes"]
            if e["global_episode_idx"] in subset)
out = subprocess.run(["du", "-sb", sys.argv[3]], capture_output=True, text=True).stdout
size = int(out.split()[0])
per = size / steps
total = manifest["totals"]["timesteps"]
print(f"CALIBRATION steps={steps} bytes={size} per_step={per / 1024:.0f} KiB "
      f"→ 全量 {total} step 外推 {per * total / 1e9:.0f} GB")
print(f"CALIBRATION_BYTES_PER_STEP={int(per)}   # 提交时用: BYTES_PER_STEP={int(per)} bash step1_submit.sh")
PYCAL

echo "LAYER1_PASS  第一层通过：分片实现与未改动 builder 逐字节相同，"
echo "             build_shard.py 已取得「本地真值」资格。"
echo "下一步：bash ${V1_SCRIPT_DIR}/legacy/step_bench.sh   （CPU/mem 档位实测，已定案 2C/24G，仅需复测时跑）"
