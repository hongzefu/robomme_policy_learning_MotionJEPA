#!/usr/bin/env bash
# ── 第 0 段：turbo 侧一次性置备（幂等，可反复跑）────────────────────────────────
# 前置：仓库本体已 rsync 到 turbo（一次性动作，见本目录 README「仓库搬迁」一节）。
# 本脚本负责剩下五件事，每件都先检测再决定要不要做：
#   ① 在 NFS 上重建 venv —— greatlakes.md 的硬规则：uv 默认把解释器装在本机 home，
#      .venv/bin/python 会 symlink 过去，在计算节点上是**死链**。必须显式指定
#      装在 NFS 的解释器绝对路径重建，这样 pyvenv.cfg 与 bin/python 天然落在 NFS，
#      本机与集群双端可用、无需任何手术。
#   ② 原始 H5 暂存到 turbo（计算节点看不见本机 /data）+ 两侧各算一次 sha256。
#      ⚠ 这份 turbo 副本是**临时暂存**，全流程验收通过后按 AGENTS.md 第 15 条删除。
#   ③ 生成 episode 清单（scan_manifest.py build）——全流程唯一真值源。
#      口径与既往逐字一致（原先由 legacy/step_local_baseline.sh 生成，上移到此）。
#   ④ 模型内联到 v1-store/models/。
#   ⑤ 自检：git 干净、jax 能 import、四个 H5 齐全。
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"

v1_prepare_dirs
v1_require_cmd rsync jq sha256sum uv

STEP="${1:-all}"

do_venv() {
  echo "=== [venv] 在 NFS 上重建（解释器必须来自 NFS，否则集群上是死链）==="
  [[ -x "${NFS_UV_PYTHON}" ]] || {
    echo "错误: 缺少 NFS 解释器 ${NFS_UV_PYTHON}" >&2
    echo "      先跑: UV_PYTHON_INSTALL_DIR=${GL_ROOT}/uv-python uv python install 3.11.14" >&2
    exit 1
  }
  if [[ -x "${PY}" ]] && "${PY}" -c 'import jax' >/dev/null 2>&1; then
    local real; real="$(readlink -f "${PY}")"
    case "${real}" in
      "${GL_ROOT}"/*) echo "  venv 已就绪且解释器在 NFS: ${real}"; return 0 ;;
      *) echo "  ⚠ 现有 venv 的解释器在 ${real}（不在 NFS），重建" ;;
    esac
  fi
  ( cd "${REPO_ROOT}" && uv venv --python "${NFS_UV_PYTHON}" && UV_LINK_MODE=copy uv sync )
  echo "  解释器 -> $(readlink -f "${PY}")"
}

do_h5() {
  echo "=== [H5] 暂存到 turbo 并两侧 sha256 核对 ==="
  v1_validate_raw_h5 "${RAW_H5_LOCAL}"
  mkdir -p "${RAW_H5_TURBO}"
  rsync -a --info=stats2 "${RAW_H5_LOCAL}/" "${RAW_H5_TURBO}/"
  v1_validate_raw_h5 "${RAW_H5_TURBO}"

  local local_manifest="${V1_STORE}/input_manifest_local.json"
  # 两侧并行算：本机 NVMe 约 4 分钟、turbo NFS 读 321 GB 约 40 分钟，串行是纯浪费。
  echo "  并行计算两侧 sha256（本机约 4 分钟 / turbo 约 40 分钟）..."
  "${PY}" "${V1_SCRIPT_DIR}/finalize_checks.py" hash-inputs \
      --raw_dir "${RAW_H5_LOCAL}" --out "${local_manifest}" \
      > "${LOGS_DIR}/hash_local.log" 2>&1 &
  local pid_local=$!
  "${PY}" "${V1_SCRIPT_DIR}/finalize_checks.py" hash-inputs \
      --raw_dir "${RAW_H5_TURBO}" --out "${INPUT_MANIFEST_PATH}" \
      > "${LOGS_DIR}/hash_turbo.log" 2>&1 &
  local pid_turbo=$!
  wait "${pid_local}" || { echo "错误: 本机 sha256 计算失败, 见 ${LOGS_DIR}/hash_local.log" >&2; exit 1; }
  wait "${pid_turbo}" || { echo "错误: turbo sha256 计算失败, 见 ${LOGS_DIR}/hash_turbo.log" >&2; exit 1; }

  # 只比 files 字段：raw_dir 本来就不同，比它必然失败
  if ! diff <(jq -S '.files' "${local_manifest}") <(jq -S '.files' "${INPUT_MANIFEST_PATH}"); then
    echo "错误: turbo H5 副本与本机原件不同源——一致性验证的地基塌了，拒绝继续" >&2
    exit 1
  fi
  echo "  ✓ 四个 H5 逐文件 sha256 同源（本机原件 == turbo 副本）"
}

do_manifest() {
  echo "=== [manifest] 分片清单（若已存在则复用；NFS 全量扫描约 48 分钟）==="
  if [[ -f "${MANIFEST_PATH}" ]]; then
    echo "  复用 ${MANIFEST_PATH}（重扫请先手动删除它）"
  else
    v1_require_venv
    "${PY}" "${V1_SCRIPT_DIR}/scan_manifest.py" build \
        --raw_dir "${RAW_H5_DIR}" --out "${MANIFEST_PATH}" --num_shards 8
  fi
}

do_models() { bash "${V1_SCRIPT_DIR}/stage_models.sh"; }

do_check() {
  echo "=== [自检] ==="
  # turbo 的默认 ACL 强制给每个文件加属主执行位（664 -> 774），git 会把全部文件报成
  # mode change 100644 => 100755（内容零差异）。关掉 filemode 感知是唯一干净的解法。
  if [[ "$(git -C "${REPO_ROOT}" config --get core.filemode || echo true)" != "false" ]]; then
    git -C "${REPO_ROOT}" config core.filemode false
    echo "  · 已设 core.filemode=false（turbo 默认 ACL 会强制加执行位）"
  else
    echo "  ✓ core.filemode=false 已就位"
  fi
  git -C "${REPO_ROOT}" fsck --no-progress --no-dangling >/dev/null && echo "  ✓ git fsck 通过"
  echo "  git HEAD = $(git -C "${REPO_ROOT}" rev-parse HEAD)"
  v1_require_venv && echo "  ✓ venv 可 import jax"
  "${PY}" - <<'PYCHK'
import jax, jaxlib
print(f"  jax={jax.__version__} jaxlib={jaxlib.__version__} devices={jax.devices()}")
PYCHK
  v1_validate_raw_h5 "${RAW_H5_TURBO}" && echo "  ✓ turbo H5 四件齐全"
  v1_require_models 0 && echo "  ✓ SigLIP 就位"
  echo "  ✓ 第 0 段完成"
}

case "${STEP}" in
  venv)     do_venv ;;
  h5)       do_h5 ;;
  manifest) do_manifest ;;
  models)   do_models ;;
  check)    do_check ;;
  all)      do_venv; do_h5; do_manifest; do_models; do_check ;;
  *) echo "用法: $0 [all|venv|h5|manifest|models|check]" >&2; exit 1 ;;
esac
