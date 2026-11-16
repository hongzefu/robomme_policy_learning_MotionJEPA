#!/usr/bin/env bash
# 共享任务池的一条通道（契约 C4，计划 1116-steer-speedup-fix-plan S-D）。池由 scripts/interpret/pool_make.py make 生成。
#
# 用法：
#   pool_runner.sh --pool-root <O> --stage <池名> [--stage <池名2> …] --site gl|aspen
#                  [--env-file <env-gl.sh|env-aspen.sh>] [--gpu <卡号>] [--mem-frac <比例>]
#                  [--run-cmd '<命令前缀>'] [--channel <通道名>] [--log <文件>]
#
# 行为：
#   - 按 --stage 给出的顺序领：前一个池 todo 里没有本站点可领的 spec 了，才领下一个池。
#   - 在 <O>/pool/<池名>/todo/ 里按文件名序（= 成本序号，长者先领）找 *.<site>.json，
#     用 mkdir <池>/claim/<键> 原子领取，领到才跑；领不到就看下一个。
#   - 跑 <命令前缀> --spec <json>（缺省命令前缀 "$py" -m scripts.interpret.run_local，$py 由 env 文件设置），
#     输出写 <池>/logs/<键>.<site>.log；rc=0 把该键的全部 json 移入 done/，否则移入 fail/，并写 <键>.result（rc、日志路径）。
#     某 spec 失败只记录、继续领下一个。
#   - 给了 --gpu / --mem-frac 时，把领到的 spec 复制到 claim/<键>/spec.<site>.json 并覆盖 cuda（及已有的
#     motion_online_gpu）/ xla_mem_fraction 再跑（run_local 从 spec 读卡号与显存比例）；--gpu 与 --mem-frac
#     也作为 env 文件的两个参数（source <env> <gpu> <frac>），XLA_PYTHON_CLIENT_MEM_FRACTION 同时导出。
#   - <池>/STOP 存在时不再领新 spec（已在跑的跑完），用于平滑收尾。
#   - 每 spec 一行开始 POOL_SPEC_START、一行结束 POOL_SPEC_END（含 rc、耗时、日志）；结束写 POOL_RUNNER_DONE
#     与 EXIT_CODE=（全部成功为 0，有失败为 1，env 文件失败为其退出码）。
set -o pipefail

POOL_ROOT=""; SITE=""; ENV_FILE=""; GPU=""; MEM_FRAC=""; RUN_CMD=""; CHANNEL="${HOSTNAME:-host}-$$"; LOG=""
STAGES=()

say() {
  echo "$*"
  if [ -n "$LOG" ]; then echo "$*" >> "$LOG"; fi
}
die() { say "POOL_RUNNER_ERROR $*"; say "EXIT_CODE=2"; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --pool-root) POOL_ROOT=$2; shift 2;;
    --stage) STAGES+=("$2"); shift 2;;
    --site) SITE=$2; shift 2;;
    --env-file) ENV_FILE=$2; shift 2;;
    --gpu) GPU=$2; shift 2;;
    --mem-frac) MEM_FRAC=$2; shift 2;;
    --run-cmd) RUN_CMD=$2; shift 2;;
    --channel) CHANNEL=$2; shift 2;;
    --log) LOG=$2; shift 2;;
    -h|--help) sed -n '2,25p' "$0"; exit 0;;
    *) die "未知参数 $1";;
  esac
done

[ -n "$POOL_ROOT" ] || die "缺 --pool-root"
[ "${#STAGES[@]}" -gt 0 ] || die "缺 --stage"
case "$SITE" in gl|aspen) ;; *) die "--site 只能是 gl|aspen，得到 '$SITE'";; esac
POOL_ROOT=$(cd "$POOL_ROOT" && pwd) || die "池根不存在 $POOL_ROOT"
case "$LOG" in ""|/*) ;; *) LOG=$PWD/$LOG;; esac   # env 文件会 cd 到运行树，先转绝对路径
for st in "${STAGES[@]}"; do
  [ -d "$POOL_ROOT/pool/$st/todo" ] && [ -d "$POOL_ROOT/pool/$st/claim" ] || die "池不存在或不完整：$POOL_ROOT/pool/$st"
done

if [ -n "$ENV_FILE" ]; then
  [ -f "$ENV_FILE" ] || die "env 文件不存在 $ENV_FILE"
  # env 文件用 return 报错（如 ENV_CHECK=FAIL），在本 shell 里 source 才能拿到 $py 与导出的环境
  # shellcheck disable=SC1090
  source "$ENV_FILE" "${GPU:-0}" ${MEM_FRAC:+"$MEM_FRAC"}
  rc=$?
  if [ "$rc" -ne 0 ]; then say "POOL_RUNNER_ERROR env_file=$ENV_FILE rc=$rc"; say "EXIT_CODE=$rc"; exit "$rc"; fi
fi
if [ -n "$MEM_FRAC" ]; then export XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRAC; fi
if [ -z "$RUN_CMD" ]; then
  [ -n "${py:-}" ] || die "未给 --run-cmd，且 env 文件没有设置 \$py"
  RUN_CMD='"$py" -m scripts.interpret.run_local'
fi
JSON_PY=${POOL_JSON_PY:-${py:-python3}}

ran=0; ok=0; failed=0
current_key=""
on_signal() {
  say "POOL_RUNNER_INTERRUPTED channel=$CHANNEL key=${current_key:-none}（该键保持已领取，不会被重跑）"
  say "EXIT_CODE=130"
  exit 130
}
trap on_signal INT TERM

# 写覆盖后的 spec：cuda、motion_online_gpu（仅当原 spec 有该键）、xla_mem_fraction
override_spec() {
  local src=$1 dst=$2
  POOL_SRC=$src POOL_DST=$dst POOL_GPU=$GPU POOL_FRAC=$MEM_FRAC "$JSON_PY" - <<'PYEOF'
import json, os
src, dst = os.environ["POOL_SRC"], os.environ["POOL_DST"]
spec = json.load(open(src, encoding="utf-8"))
gpu, frac = os.environ.get("POOL_GPU", ""), os.environ.get("POOL_FRAC", "")
if gpu:
    spec["cuda"] = gpu
    if "motion_online_gpu" in spec:
        spec["motion_online_gpu"] = gpu
if frac:
    spec["xla_mem_fraction"] = float(frac)
tmp = dst + ".tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(spec, fh, ensure_ascii=False, indent=2)
os.replace(tmp, dst)
PYEOF
}

# 领取：按文件名序扫本站点 json，mkdir 成功者得；输出领到的键，领不到输出空
claim_next() {
  local pool=$1 f key
  while IFS= read -r f; do
    key=${f%".$SITE.json"}
    [ -d "$pool/claim/$key" ] && continue
    if mkdir "$pool/claim/$key" 2>/dev/null; then echo "$key"; return 0; fi
  done < <(find "$pool/todo" -maxdepth 1 -type f -name "*.$SITE.json" -printf '%f\n' | LC_ALL=C sort)
  return 1
}

for st in "${STAGES[@]}"; do
  pool=$POOL_ROOT/pool/$st
  say "POOL_STAGE_BEGIN stage=$st site=$SITE channel=$CHANNEL"
  while true; do
    if [ -e "$pool/STOP" ]; then say "POOL_STOP_SEEN stage=$st channel=$CHANNEL"; break; fi
    key=$(claim_next "$pool") || break
    [ -n "$key" ] || break
    current_key=$key
    echo "channel=$CHANNEL site=$SITE host=${HOSTNAME:-} pid=$$ t=$(date -u +%FT%TZ)" > "$pool/claim/$key/owner"
    src=$pool/todo/$key.$SITE.json
    spec=$src
    if [ -n "$GPU" ] || [ -n "$MEM_FRAC" ]; then
      spec=$pool/claim/$key/spec.$SITE.json
      if ! override_spec "$src" "$spec"; then
        say "POOL_RUNNER_ERROR key=$key 覆盖 spec 失败"; spec=""
      fi
    fi
    log=$pool/logs/$key.$SITE.log
    t0=$(date +%s)
    say "POOL_SPEC_START stage=$st key=$key site=$SITE channel=$CHANNEL t=$(date -u +%FT%TZ)"
    if [ -n "$spec" ]; then
      eval "$RUN_CMD --spec \"\$spec\"" > "$log" 2>&1 < /dev/null
      rc=$?
    else
      rc=97
    fi
    secs=$(( $(date +%s) - t0 ))
    ran=$((ran+1))
    if [ "$rc" -eq 0 ]; then dest=done; ok=$((ok+1)); else dest=fail; failed=$((failed+1)); fi
    for f in "$pool/todo/$key".*.json; do
      [ -e "$f" ] && mv -f "$f" "$pool/$dest/"
    done
    printf 'key=%s\nstage=%s\nsite=%s\nchannel=%s\nrc=%s\nseconds=%s\nlog=%s\nspec=%s\n' \
      "$key" "$st" "$SITE" "$CHANNEL" "$rc" "$secs" "$log" "$spec" > "$pool/$dest/.$key.result.tmp"
    mv -f "$pool/$dest/.$key.result.tmp" "$pool/$dest/$key.result"
    say "POOL_SPEC_END stage=$st key=$key site=$SITE channel=$CHANNEL rc=$rc seconds=$secs dest=$dest log=$log"
    current_key=""
  done
  say "POOL_STAGE_END stage=$st channel=$CHANNEL"
done

say "POOL_RUNNER_DONE channel=$CHANNEL site=$SITE ran=$ran ok=$ok failed=$failed"
if [ "$failed" -eq 0 ]; then code=0; else code=1; fi
say "EXIT_CODE=$code"
exit "$code"
