#!/usr/bin/env bash
# ── step1 第 ⑧ 项（配额闸门）的桩测试 ─────────────────────────────────────────
# 验的是 2026-08-24 审查确认的 fail-open：配额查询失败时，原实现只打印一行提示、
# **不置 FAIL=1**，于是配额真不够也照样提交 8 个 GPU job。
#
# 安全性（为什么可以带 CONFIRM_FULL=yes 跑）：本测试在 PATH 前面放一个恒退出 1 的假
# `uv`。step1 里**一切与集群交互的路径都要经过 `uv run ... gl_submit.py`**
# （第 ⑧ 项的配额查询、array 提交、finalize 提交三处），假 uv 一挡，sbatch 根本无从
# 发出；本机 Python 走的是 ${PY}，不受影响。另外还叠了三个独立 FAIL 源
# （假 ssh → ⑤、RATE=0 → ⑦、真实 OUT 非空 → ③），任一生效都会在提交前拒绝。
#
# 跑法： bash scripts/dataset/gl/legacy/test_step1_quota_gate.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S1="${HERE}/../step1_submit.sh"
STUB="$(mktemp -d)"
trap 'rm -rf "${STUB}"' EXIT

# 假 uv / 假 ssh：都恒退出 1
printf '#!/usr/bin/env bash\nexit 1\n' > "${STUB}/uv";  chmod +x "${STUB}/uv"
printf '#!/usr/bin/env bash\nexit 1\n' > "${STUB}/ssh"; chmod +x "${STUB}/ssh"

PASS=0; FAILED=0
check() {  # check <用例名> <应出现的串> <日志文件>
  if grep -qF -- "$2" "$3"; then echo "  ✓ $1：找到「$2」"; PASS=$((PASS + 1))
  else echo "  ✗ $1：未找到「$2」"; FAILED=$((FAILED + 1)); fi
}
check_absent() {
  if grep -qF -- "$2" "$3"; then echo "  ✗ $1：不应出现「$2」"; FAILED=$((FAILED + 1))
  else echo "  ✓ $1：确认未出现「$2」"; PASS=$((PASS + 1)); fi
}

echo "=== A. 配额查询失败 + 未审批 → 第 ⑧ 项必须亮 ✗ ==="
PATH="${STUB}:${PATH}" RATE=0 bash "${S1}" > "${STUB}/a.log" 2>&1 || true
check A "✗ 配额查询失败" "${STUB}/a.log"
check A "⛔ 全量构建是审批点" "${STUB}/a.log"
check_absent A "[提交] array..." "${STUB}/a.log"

echo "=== B. 已审批但 pre-flight 有失败 → 必须拒绝提交、绝不 sbatch ==="
PATH="${STUB}:${PATH}" RATE=0 CONFIRM_FULL=yes bash "${S1}" > "${STUB}/b.log" 2>&1 || true
check B "pre-flight 未全绿，拒绝提交" "${STUB}/b.log"
check_absent B "[提交] array..." "${STUB}/b.log"

echo "=== C. 显式豁免 → 提示豁免且不把该项算作绿灯 ==="
PATH="${STUB}:${PATH}" RATE=0 ALLOW_QUOTA_SKIP=yes bash "${S1}" > "${STUB}/c.log" 2>&1 || true
check C "已按 ALLOW_QUOTA_SKIP=yes 显式豁免" "${STUB}/c.log"
check_absent C "✗ 配额查询失败" "${STUB}/c.log"

echo
if [[ "${FAILED}" -eq 0 ]]; then
  echo "STEP1_GATE_PASS  ${PASS} 项断言全过"
else
  echo "STEP1_GATE_FAIL  通过 ${PASS} / 失败 ${FAILED}（日志留在 ${STUB}，本脚本退出后自动清理）"
  exit 1
fi
