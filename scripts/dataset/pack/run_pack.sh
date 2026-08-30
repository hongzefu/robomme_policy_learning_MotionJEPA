#!/usr/bin/env bash
# S4 全量打包 + 全量校验的 tmux 驱动（v2 计划 A.2；AGENTS 7 三件套内嵌）。
#
# 用法（在 detached tmux 里起，模板见 AGENTS.md 第 7 条）：
#   tmux new-session -d -s pack-framesamp \
#     "bash scripts/dataset/pack/run_pack.sh"
# 可调环境变量：
#   PACK_SOURCE   源库根（默认 v1-store/datasets/4task-gl）
#   PACK_OUT      打包库根（默认 v1-store/datasets/4task-gl-framesamp）
#   PACK_MANIFEST 清单（默认 v1-store/episode_manifest.json）
#   PACK_READER   decode|slice（默认 decode——首跑零布局假设）
#   PACK_PROCS    进程数（默认 16）
#   PACK_RESUME   1 = pack 带 --resume 续跑
# 日志：v1-store/logs/pack-framesamp.log；结束写 EXIT_CODE=。
# 判定行：PACK_DONE=1 → VERIFY_PACK=PASS scanned=483291 mismatches=0
set -o pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "错误: 仓库根解析失败 ${REPO_ROOT}（缺 pyproject.toml）" >&2; exit 1
fi
cd "${REPO_ROOT}"

PACK_SOURCE="${PACK_SOURCE:-${REPO_ROOT}/v1-store/datasets/4task-gl}"
PACK_OUT="${PACK_OUT:-${REPO_ROOT}/v1-store/datasets/4task-gl-framesamp}"
PACK_MANIFEST="${PACK_MANIFEST:-${REPO_ROOT}/v1-store/episode_manifest.json}"
PACK_READER="${PACK_READER:-decode}"
PACK_PROCS="${PACK_PROCS:-16}"
LOG="${REPO_ROOT}/v1-store/logs/pack-framesamp.log"
mkdir -p "${REPO_ROOT}/v1-store/logs"

RESUME_FLAG=()
[[ "${PACK_RESUME:-0}" == "1" ]] && RESUME_FLAG=(--resume)

{
  set -e
  export UV_LINK_MODE=copy PYTHONUNBUFFERED=1
  echo "=== pack: source=${PACK_SOURCE} out=${PACK_OUT} reader=${PACK_READER} procs=${PACK_PROCS} ==="
  uv run --project "${REPO_ROOT}" python \
    scripts/dataset/pack/pack_framesamp_store.py pack \
    --source "${PACK_SOURCE}" --manifest "${PACK_MANIFEST}" --out "${PACK_OUT}" \
    --reader "${PACK_READER}" --procs "${PACK_PROCS}" "${RESUME_FLAG[@]}"
  echo "=== verify（全量零遗漏；--resume 接管 pack 留下的锁）==="
  uv run --project "${REPO_ROOT}" python \
    scripts/dataset/pack/pack_framesamp_store.py verify \
    --store "${PACK_OUT}" --procs "${PACK_PROCS}" --resume
  echo "全部完成"
} 2>&1 | tee "${LOG}"
RC="${PIPESTATUS[0]}"
echo "EXIT_CODE=${RC}" >> "${LOG}"
exit "${RC}"
