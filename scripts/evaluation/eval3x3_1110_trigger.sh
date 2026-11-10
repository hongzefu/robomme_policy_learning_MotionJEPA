#!/usr/bin/env bash
# 3ckpt×3seed 评测触发链：fs8 训练结束 → 79999 落盘 → 8 卡空闲 → 冒烟 → 正式队列。
# 用法（在执行副本根目录）：tmux new-session -d -s ev3-trigger "bash scripts/evaluation/eval3x3_1110_trigger.sh"
set -u -o pipefail
E="$(cd "$(dirname "$0")/../.." && pwd)"
STORE=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store
PY=/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv/bin/python
L=$STORE/logs/eval3x3-1110
TL=$STORE/logs/orig100-ctx-framesamp1024-b64-80k-8gpu.log
FS_FINAL=$STORE/train-runs/mme_vla_suite/orig100-ctx-framesamp1024-b64-80k-8gpu/79999
T=$L/trigger.log
mkdir -p "$L"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$T"; }

log "TRIGGER_ARMED exec=$E commit=$(git -C "$E" rev-parse HEAD)"
until grep -q '^EXIT_CODE=' "$TL" 2>/dev/null; do sleep 60; done
log "TRAIN_DONE $(grep '^EXIT_CODE=' "$TL" | tail -1)"

for i in $(seq 1 60); do [ -f "$FS_FINAL/_CHECKPOINT_METADATA" ] && break; sleep 30; done
if [ -f "$FS_FINAL/_CHECKPOINT_METADATA" ]; then log "FS_FINAL=present"; else log "FS_FINAL=missing（FS 79999 项将 BLOCKED，其余照跑）"; fi

for i in $(seq 1 120); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>=2000' | wc -l)
  [ "$busy" -eq 0 ] && break; sleep 30
done
log "GPU_FREE busy=$busy"

cd "$E"
PYTHONUNBUFFERED=1 "$PY" scripts/evaluation/eval_queue.py --smoke 2>&1 | tee -a "$L/smoke.log"
rc=$?; echo "EXIT_CODE=$rc" >> "$L/smoke.log"; log "SMOKE_EXIT=$rc"

PYTHONUNBUFFERED=1 "$PY" scripts/evaluation/eval_queue.py 2>&1 | tee -a "$L/chain.log"
rc=$?; echo "EXIT_CODE=$rc" >> "$L/chain.log"; log "CHAIN_EXIT=$rc"
log "TRIGGER_DONE"
