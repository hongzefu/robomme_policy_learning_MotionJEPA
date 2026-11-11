#!/usr/bin/env bash
# 官方 FrameSamp+Context 512 续训 5k 的三 seed 评测链：冒烟 1 局 → SMOKE=PASS 后正式 3×800 局。
# 用法（在执行副本根目录）：tmux new-session -d -s fs5c-eval "bash scripts/evaluation/fs512cont5k_1110_eval.sh"
set -o pipefail
cd "$(dirname "$0")/../.."
PY=/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv/bin/python
Q=scripts/evaluation/fs512cont5k_1110_queue.json
L=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/fs512cont5k-1110
mkdir -p "$L"
export PYTHONUNBUFFERED=1
"$PY" scripts/evaluation/eval_queue.py --queue "$Q" --smoke 2>&1 | tee -a "$L/smoke.log"
if ! grep -q '^SMOKE=PASS' "$L/smoke.log" || grep -q '^SMOKE=FAIL' "$L/smoke.log"; then
  echo "CHAIN_STOP reason=smoke_not_pass" | tee -a "$L/chain.log"
  echo "EXIT_CODE=3" >> "$L/chain.log"; exit 3
fi
"$PY" scripts/evaluation/eval_queue.py --queue "$Q" 2>&1 | tee -a "$L/chain.log"
rc=$?
echo "EXIT_CODE=$rc" >> "$L/chain.log"
exit "$rc"
