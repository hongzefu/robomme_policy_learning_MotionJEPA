#!/usr/bin/env bash
# /dev/shm 占用采样（1117-e2e-enc-train-plan.html 第二部分一节 shmmon.sh 行、RB-D1 / RB-E2）。
# 每 --interval 秒（缺省 15）写一行：
#   SHM_SAMPLE ts=<美东 HH:MM:SS> shm_used_gb=<n> epoch=<unix 秒> gpu_util=<8 卡 utilization.gpu，逗号分隔或 na> gpu_util_mean=<x|na>
# shm_used_gb 超过 --limit-gb 时另打一行 SHM_ALERT used_gb=<n> limit_gb=<m>（告警动作 = Monitor 上报主会话，由主会话决定是否停训；
# 本脚本不停训、不杀任何进程）。GPU 利用率只是顺带记一列便于 Monitor 一眼看到，权威采样仍是 runner 的 15 s nvidia-smi csv。
# 用法：bash scripts/training/shmmon.sh [--limit-gb 330] [--interval 15] [--path /dev/shm] [--max-samples N]
#   --shmprobe 期间由 runner 后台起（日志 v1-store/logs/e2e-shmmon-aNN.log）；正式训练期间主会话起 tmux e2e-shmmon
#   （告警线 = G6 实测平台 × 1.35，绝对上限 330）。被 SIGTERM/SIGINT 结束或采满 --max-samples 时写 EXIT_CODE= 尾行。
# 单测：SHMMON_DF_CMD / SHMMON_SMI_CMD 可替换 df / nvidia-smi 的取值命令（输出格式同真实命令）。
set -uo pipefail

LIMIT_GB=330
INTERVAL=15
SHM_PATH=/dev/shm
MAX_SAMPLES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --limit-gb) LIMIT_GB="${2:?}"; shift 2 ;;
    --interval) INTERVAL="${2:?}"; shift 2 ;;
    --path) SHM_PATH="${2:?}"; shift 2 ;;
    --max-samples) MAX_SAMPLES="${2:?}"; shift 2 ;;
    *) printf '拒绝：未知参数 %s\n' "$1" >&2; exit 2 ;;
  esac
done
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
  echo "SHM_SAMPLE ts=$(TZ=America/New_York date +%H:%M:%S) shm_used_gb=$gb epoch=$(date +%s) gpu_util=$u gpu_util_mean=$um"
  if awk -v g="$gb" -v l="$LIMIT_GB" 'BEGIN{exit !(g>l)}'; then
    alerts=$((alerts + 1))
    echo "SHM_ALERT used_gb=$gb limit_gb=$LIMIT_GB"
  fi
  n=$((n + 1))
  if [ "$MAX_SAMPLES" -gt 0 ] && [ "$n" -ge "$MAX_SAMPLES" ]; then finish; fi
  sleep "$INTERVAL" & wait $!
done
