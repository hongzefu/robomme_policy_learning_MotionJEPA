#!/usr/bin/env bash
# /dev/shm 占用采样（1117-e2e-enc-train-plan.html 第二部分一节 shmmon.sh 行、RB-D1 / RB-E2）。
# 每 --interval 秒（缺省 15）写一行：
#   SHM_SAMPLE ts=<美东 HH:MM:SS> shm_used_gb=<n> epoch=<unix 秒> gpu_util=<8 卡 utilization.gpu，逗号分隔或 na> gpu_util_mean=<x|na>
# shm_used_gb 超过 --limit-gb 时另打一行 SHM_ALERT used_gb=<n> limit_gb=<m>（告警动作 = Monitor 上报主会话，由主会话决定是否停训；
# 本脚本不停训、不杀任何进程）。GPU 利用率只是顺带记一列便于 Monitor 一眼看到，权威采样仍是 runner 的 15 s nvidia-smi csv。
# 用法：bash scripts/training/shmmon.sh [--limit-gb 330] [--interval 15] [--path /dev/shm] [--max-samples N]
#        [--formal-rec <正式记录目录> --samples-log <新文件> [--formal-run-log <正式 runner 日志>] [--judge-step 2000]]
#   --shmprobe 期间由 runner 后台起（日志 v1-store/logs/e2e-shmmon-aNN.log；旧模式，闸门表已不用）。
#   被 SIGTERM/SIGINT 结束或采满 --max-samples 时写 EXIT_CODE= 尾行。
# 正式训练期判定（用户 2026-11-18 美东裁定：「问题二能不能够在正式起跑的时候再来做，就不要单独跑这个」，取代 --shmprobe）：
#   给 --formal-rec 时，每条 SHM_SAMPLE 另追加到 --samples-log（必须是新文件，已存在即拒），每次采样后调用
#   e2e_g6_judge.py shm-formal：正式记录目录 metrics.jsonl（与旧 shmprobe 同一步数来源）出现 step ≥ --judge-step 的行后，
#   按「TRAIN_START_EPOCH → 第 2000 步」窗口打印一次
#     SHM_PEAK=<PASS|FAIL> peak_gb=<x> limit=330 growth_last1500=<y>% window=formal_steps_0_2000 …
#   FAIL 时另打印 SHM_ALERT reason=formal_first2000 … 与 RB-D2 退路提示（主会话停训、E2E_PREFETCH_FACTOR=1 从头重起）；
#   之后打印 SHM_FORMAL_JUDGED rc=<0|1|…>，继续照常采样告警。本脚本不停训。
#   与 RB-E2 的 e2e-shmmon 会话合并为一个：告警线 --limit-gb 初值 293（原「G6 平台 × 1.35」，G6 不再测平台），
#   判定用的绝对上限 330 写死在 judge（SHM_LIMIT_GB）。主会话在正式 runner 起跑后（主检出根目录）起：
#     V1=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store; F=v2-1600ep-m8x8-modul-e2eenc-s16-b128-50k
#     tmux new-session -d -s e2e-shmmon "set -o pipefail; bash scripts/training/shmmon.sh --limit-gb 293 --interval 15 \
#       --formal-rec $V1/bench/$F --formal-run-log $V1/logs/$F.runner.log --samples-log $V1/logs/e2e-shmmon.samples.log \
#       2>&1 | tee $V1/logs/e2e-shmmon.log; echo \"EXIT_CODE=\$?\" >> $V1/logs/e2e-shmmon.log"
# 单测：SHMMON_DF_CMD / SHMMON_SMI_CMD 可替换 df / nvidia-smi 的取值命令（输出格式同真实命令）；
#   SHMMON_PY 替换调用 judge 的解释器（缺省 <仓库根>/.venv/bin/python；judge 只用标准库）。
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

LIMIT_GB=330
INTERVAL=15
SHM_PATH=/dev/shm
MAX_SAMPLES=0
FORMAL_REC=""; FORMAL_RUN_LOG=""; SAMPLES_LOG=""; JUDGE_STEP=2000
while [ $# -gt 0 ]; do
  case "$1" in
    --limit-gb) LIMIT_GB="${2:?}"; shift 2 ;;
    --interval) INTERVAL="${2:?}"; shift 2 ;;
    --path) SHM_PATH="${2:?}"; shift 2 ;;
    --max-samples) MAX_SAMPLES="${2:?}"; shift 2 ;;
    --formal-rec) FORMAL_REC="${2:?}"; shift 2 ;;
    --formal-run-log) FORMAL_RUN_LOG="${2:?}"; shift 2 ;;
    --samples-log) SAMPLES_LOG="${2:?}"; shift 2 ;;
    --judge-step) JUDGE_STEP="${2:?}"; shift 2 ;;
    *) printf '拒绝：未知参数 %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [ -n "$FORMAL_REC" ]; then
  [ -n "$SAMPLES_LOG" ] || { printf '拒绝：--formal-rec 必须同时给 --samples-log\n' >&2; exit 2; }
  [ ! -e "$SAMPLES_LOG" ] || { printf '拒绝覆盖已存在的 samples 文件 %s\n' "$SAMPLES_LOG" >&2; exit 2; }
  [[ "$JUDGE_STEP" =~ ^[0-9]+$ ]] || { printf '拒绝：judge-step 非法 %s\n' "$JUDGE_STEP" >&2; exit 2; }
elif [ -n "$SAMPLES_LOG$FORMAL_RUN_LOG" ]; then
  printf '拒绝：--samples-log / --formal-run-log 只能与 --formal-rec 同用\n' >&2; exit 2
fi
JUDGE_PY="${SHMMON_PY:-$REPO/.venv/bin/python}"
JUDGED=0
[[ "$LIMIT_GB" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { printf '拒绝：limit-gb 非法 %s\n' "$LIMIT_GB" >&2; exit 2; }
[[ "$INTERVAL" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { printf '拒绝：interval 非法 %s\n' "$INTERVAL" >&2; exit 2; }
[[ "$MAX_SAMPLES" =~ ^[0-9]+$ ]] || { printf '拒绝：max-samples 非法 %s\n' "$MAX_SAMPLES" >&2; exit 2; }

used_bytes() {
  if [ -n "${SHMMON_DF_CMD:-}" ]; then bash -c "$SHMMON_DF_CMD"; else df -B1 --output=used "$SHM_PATH" | tail -n 1; fi
}
gpu_utils() {
  if [ -n "${SHMMON_SMI_CMD:-}" ]; then bash -c "$SHMMON_SMI_CMD"
  else nvidia-smi --id=0,1,2,3,4,5,6,7 --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null; fi
}

RC=0
finish() { echo "SHMMON_END samples=${n:-0} alerts=${alerts:-0}"; echo "EXIT_CODE=$RC"; exit "$RC"; }
trap 'RC=0; finish' TERM INT
echo "SHMMON_START path=$SHM_PATH limit_gb=$LIMIT_GB interval=${INTERVAL}s ts=$(TZ=America/New_York date '+%F %T %Z')"
n=0; alerts=0
while :; do
  b="$(used_bytes | tr -d ' ')"
  if ! [[ "$b" =~ ^[0-9]+$ ]]; then echo "SHMMON_ERROR 取不到 $SHM_PATH 用量：$b 失败"; RC=1; finish; fi
  gb="$(awk -v b="$b" 'BEGIN{printf "%.2f", b/1e9}')"
  u="$(gpu_utils | tr -d ' ' | paste -sd, -)"
  if [[ "$u" =~ ^[0-9.,]+$ ]]; then
    um="$(awk -v s="$u" 'BEGIN{n=split(s,a,","); t=0; for(i=1;i<=n;i++)t+=a[i]; printf "%.1f", t/n}')"
  else u=na; um=na; fi
  sample="SHM_SAMPLE ts=$(TZ=America/New_York date +%H:%M:%S) shm_used_gb=$gb epoch=$(date +%s) gpu_util=$u gpu_util_mean=$um"
  echo "$sample"
  [ -z "$SAMPLES_LOG" ] || printf '%s\n' "$sample" >> "$SAMPLES_LOG"
  if awk -v g="$gb" -v l="$LIMIT_GB" 'BEGIN{exit !(g>l)}'; then
    alerts=$((alerts + 1))
    echo "SHM_ALERT used_gb=$gb limit_gb=$LIMIT_GB"
  fi
  if [ -n "$FORMAL_REC" ] && [ "$JUDGED" = 0 ]; then
    jargs=(shm-formal --rec "$FORMAL_REC" --shmmon-samples "$SAMPLES_LOG" --judge-step "$JUDGE_STEP")
    [ -z "$FORMAL_RUN_LOG" ] || jargs+=(--run-log "$FORMAL_RUN_LOG")
    if "$JUDGE_PY" -I "$SCRIPT_DIR/g0/e2e_g6_judge.py" "${jargs[@]}"; then jrc=0; else jrc=$?; fi
    if [ "$jrc" != 3 ]; then JUDGED=1; echo "SHM_FORMAL_JUDGED rc=$jrc"; fi   # 3 = 未到第 JUDGE_STEP 步
  fi
  n=$((n + 1))
  if [ "$MAX_SAMPLES" -gt 0 ] && [ "$n" -ge "$MAX_SAMPLES" ]; then finish; fi
  sleep "$INTERVAL" & wait $!
done
