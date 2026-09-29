#!/usr/bin/env bash
# 官方路线 xhard0 单片评估（在占位 job 的 srun 步骤内运行）：官方 test 元数据的 hard 子集，按清单只评本片 (task, source_episode)。
# 同一张卡：先起 policy server，再跑 examples/robomme/eval.py（清单模式），结束收 server。
# 与官方的差别只有：server 启动环境 XLA_PYTHON_CLIENT_MEM_FRACTION=0.75（长演示 add_buffer 需要）、清单选局与逐局记录、视频命名。
# 用法（环境变量）：
#   MANIFEST=<清单 jsonl，每行 {task, source_episode, seed, shard}> SHARD=<i> PORT=<端口> RUN_TAG=<标签> \
#   [VIDEO_DIR=<目录>] [SAVE_ROOT=<目录>] [ONLY_TASKS=a,b] [ROBOMME_ENV=<客户端 venv>] bash scripts/gl_eval_official_xhard0.sh
# srun 包装：srun --overlap --jobid=<占位JobID> --ntasks=1 --cpus-per-task=4 ... bash scripts/gl_eval_official_xhard0.sh
# 客户端环境：默认复用 testhard-v7 克隆的 robomme_env（装的是 fork 的 robomme＋robomme_hard）；本脚本用 PYTHONPATH
#   把本工作树官方子模块（856bc3a）的 src 放到最前，使 import robomme 解析到官方源码，起跑前核对且 robomme_hard 未被导入。
# 断点续评：eval.py 按 episodes.jsonl 每个 (task, source_episode) 最后一条记录判，正常终态跳过；最多跑 3 遍重评 error；
# 单局无进展（episodes.jsonl mtime）30 分钟即杀掉 eval 重起。
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
: "${MANIFEST:?}" "${SHARD:?}" "${PORT:?}" "${RUN_TAG:?}"
# 与 testhard 分支相同的权重 run 根（官方 perceptual-framesamp-modul 79999 ＋ history_config.txt）
CKPT=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/eval-out/mmevla-ckpt/perceptual-framesamp-modul/79999
SAVE_ROOT="${SAVE_ROOT:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/eval-out/mmevla-official-xhard0-${RUN_TAG}-s${SHARD}}"
SAVE_DIR="$SAVE_ROOT/mmevla-official-xhard0/ckpt79999/seed7"
VIDEO_DIR="${VIDEO_DIR:-$SAVE_ROOT/videos-xhard0}"
ROBOMME_ENV="${ROBOMME_ENV:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning-testhard-v7/robomme_env}"
OFFICIAL_SRC="$REPO/third_party/robomme_benchmark/src"
# server 在 testhard-v7 克隆里起（复用其 .venv）：该分支相对 ecf086c 未改 src/、serve_policy.py、pyproject.toml、uv.lock、packages/，
# server 代码与官方逐字节相同；本工作树不另建 JAX 环境
SERVER_REPO="${SERVER_REPO:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning-testhard-v7}"
mkdir -p "$SAVE_ROOT"
SERVER_LOG="$SAVE_ROOT/server.log"
[[ -f "$MANIFEST" ]] || { echo "错误: 清单不存在: $MANIFEST"; echo "EXIT_CODE=1"; exit 1; }
[[ -d "$CKPT/params" ]] || { echo "错误: checkpoint 缺 params: $CKPT"; echo "EXIT_CODE=1"; exit 1; }
[[ -x "$SERVER_REPO/.venv/bin/python" ]] || { echo "错误: server 环境不存在: $SERVER_REPO/.venv"; echo "EXIT_CODE=1"; exit 1; }
git -C "$SERVER_REPO" diff --quiet ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b -- src scripts/serve_policy.py pyproject.toml uv.lock packages \
  || { echo "错误: $SERVER_REPO 的 server 代码与官方 ecf086c 不同"; echo "EXIT_CODE=1"; exit 1; }
[[ -x "$ROBOMME_ENV/bin/python" ]] || { echo "错误: 客户端环境不存在: $ROBOMME_ENV"; echo "EXIT_CODE=1"; exit 1; }
[[ -d "$OFFICIAL_SRC/robomme" ]] || { echo "错误: 官方子模块未初始化: $OFFICIAL_SRC"; echo "EXIT_CODE=1"; exit 1; }
SHARD_N=$(python3 -c "
import json,sys
print(sum(1 for l in open(sys.argv[1]) if l.strip() and int(json.loads(l)['shard'])==int(sys.argv[2])))" "$MANIFEST" "$SHARD")
[[ "$SHARD_N" -gt 0 ]] || { echo "错误: 清单里 shard=$SHARD 没有行"; echo "EXIT_CODE=1"; exit 1; }
# 客户端公共环境：去掉代理变量（GL 节点代理拒绝 websocket，HTTP 403）、官方 src 置顶
CLIENT_ENV=(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY
  NO_PROXY=127.0.0.1,localhost PYTHONUNBUFFERED=1 GLIBC_TUNABLES=glibc.rtld.optional_static_tls=16384
  PYTHONPATH="$OFFICIAL_SRC")
# 起跑前核对：robomme 必须解析到本工作树官方子模块，且导入 env_runner 后 robomme_hard 不在 sys.modules
( cd examples/robomme && "${CLIENT_ENV[@]}" "$ROBOMME_ENV/bin/python" -c "
import sys, robomme, env_runner
assert robomme.__file__.startswith(sys.argv[1]), robomme.__file__
assert 'robomme_hard' not in sys.modules
print('OFFICIAL_ROBOMME', robomme.__file__)" "$OFFICIAL_SRC" ) || { echo "错误: 客户端未解析到官方 robomme 或导入了 robomme_hard"; echo "EXIT_CODE=1"; exit 1; }
echo "=== MMEVLA_OFFICIAL_XHARD0 host=$(hostname) job=${SLURM_JOB_ID:-none} HEAD=$(git -C "$REPO" rev-parse HEAD) server_repo=$SERVER_REPO@$(git -C "$SERVER_REPO" rev-parse --short HEAD) submodule=$(git -C "$REPO/third_party/robomme_benchmark" rev-parse HEAD) shard=$SHARD shard_n=$SHARD_N port=$PORT ckpt=$CKPT start=$(date -Is) ==="
nvidia-smi --query-gpu=name,driver_version,compute_mode --format=csv,noheader
# 端口占用守卫：已被占用就会把别人的服务误判为就绪、静默产出空结果
if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
  exec 3>&-
  echo "错误: 端口 ${PORT} 起跑前已被占用"; echo "EXIT_CODE=1"; exit 1
fi
# 显存上限 0.75 只改 server 启动环境（官方默认 0.4 时长演示 add_buffer OOM）
( cd "$SERVER_REPO" && exec env XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
  uv run --frozen --no-sync scripts/serve_policy.py --seed=7 --port="$PORT" \
    policy:checkpoint --policy.dir="$CKPT" --policy.config=mme_vla_suite ) >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
EVAL_PID=""
cleanup() {
  [[ -n "$EVAL_PID" ]] && kill "$EVAL_PID" 2>/dev/null
  if kill -0 "$SERVER_PID" 2>/dev/null; then kill "$SERVER_PID"; sleep 2; kill -9 "$SERVER_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
for _ in $(seq 1 600); do   # server 就绪上限 20 分钟；server 先死立即退出
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "错误: server 提前退出"; tail -30 "$SERVER_LOG"; echo "EXIT_CODE=1"; exit 1; fi
  if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then exec 3>&-; break; fi
  sleep 2
done
grep -m1 -E "Restoring checkpoint|checkpoint" "$SERVER_LOG" | sed 's/^/SERVER_CKPT_LINE /'
echo "server 端口就绪 $(date +%T)"
RC=0
unresolved=0
for pass in 1 2 3; do
  errors=$(grep -c '"status": "error"' "$SAVE_DIR/episodes.jsonl" 2>/dev/null); errors=${errors:-0}
  if [[ "$errors" -gt "$SHARD_N" ]]; then echo "RETRY_CAP_HIT shard=$SHARD errors=$errors"; RC=4; break; fi
  echo "EVAL_PASS $pass errors_so_far=$errors $(date -Is)"
  ( cd examples/robomme && "${CLIENT_ENV[@]}" \
      "$ROBOMME_ENV/bin/python" eval.py --args.host=127.0.0.1 --args.port="$PORT" --args.model_seed=7 --args.model_ckpt_id=79999 \
      --args.policy_name=mmevla-official-xhard0 --args.episode_manifest="$MANIFEST" --args.shard="$SHARD" \
      --args.video_dir="$VIDEO_DIR" --args.save_dir="$SAVE_ROOT" ${ONLY_TASKS:+--args.only_tasks="$ONLY_TASKS"} ) &
  EVAL_PID=$!
  started=$(date +%s)
  stalled=0
  while kill -0 "$EVAL_PID" 2>/dev/null; do   # 无进展看门狗：首局放宽 10 分钟，之后 30 分钟
    sleep 30
    last=$(stat -c %Y "$SAVE_DIR/episodes.jsonl" 2>/dev/null || echo "$started")
    [[ "$last" -lt "$started" ]] && last=$started
    limit=1800; [[ ! -f "$SAVE_DIR/episodes.jsonl" ]] && limit=2400
    if (( $(date +%s) - last > limit )); then
      echo "EVAL_STALL pass=$pass idle_s=$(( $(date +%s) - last )) 杀掉重起（记基础设施 error）"
      pkill -TERM -P "$EVAL_PID" 2>/dev/null; kill "$EVAL_PID" 2>/dev/null; stalled=1; break
    fi
  done
  wait "$EVAL_PID"; RC=$?; EVAL_PID=""
  [[ "$stalled" = 1 ]] && { RC=124; continue; }
  unresolved=$(python3 -c "
import json,sys
last={}
for l in open(sys.argv[1]):
    if l.strip():
        r=json.loads(l); last[(r['task'],r['source_episode'])]=r['status']
print(sum(v=='error' for v in last.values()))" "$SAVE_DIR/episodes.jsonl" 2>/dev/null || echo 0)
  echo "EVAL_PASS_END $pass rc=$RC unresolved_errors=$unresolved"
  [[ "$unresolved" = 0 ]] && break
done
# 有未解决的 error 身份时不得以 0 退出
[[ "$RC" = 0 && "${unresolved:-0}" != 0 ]] && RC=3
echo "EVAL_RC=$RC end=$(date -Is)"
echo "EXIT_CODE=$RC"
exit "$RC"
