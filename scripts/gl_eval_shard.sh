#!/usr/bin/env bash
# test-hard 单片评估（在占位 job 的 srun 步骤内运行；benchmark 0927 计划 §8.2 第 4 项、第二部分 §3.3）。
# 同一张卡：先起 policy server，再跑 examples/robomme/eval.py（robomme_env 环境），结束收 server。
# 显存：JAX 上限 0.75（第一轮 0.4 时 VideoPlace 新值档约 1419 帧演示的 add_buffer 需 6.8～8.5 GB，server OOM）；仿真渲染只占数 GB。
# 用法（环境变量）：SHARD=<0..9> ROUND=<1|2> PORT=<端口> RUN_TAG=<标签> bash scripts/gl_eval_shard.sh
#   片 SHARD、轮 ROUND：episode_start = 10×(ROUND−1)+SHARD、episode_stride = 20、max_episodes = 0，
#   即 80 局的任务取 {s, s+20, s+40, s+60}、20 局的任务取 {s}，每片 13×4 + 3×1 = 55 局。
# 断点续评：eval.py 按 episodes.jsonl 里已有正常终态的身份跳过；本脚本最多跑 3 遍，error 身份在下一遍重评，
# 每片每轮 error 合计超过 55 即 RETRY_CAP_HIT 停止。单局无进展（episodes.jsonl mtime）30 分钟即杀掉 eval 重起。
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
: "${SHARD:?}" "${ROUND:?}" "${PORT:?}" "${RUN_TAG:?}"
# 官方 Yinpei/perceptual-framesamp-modul@c0f565dd 的 79999（与本机官方下载逐字节相同，diff -rq 18 文件）；
# 官方加载器从 checkpoint_dir.parent/history_config.txt 读 history 配置，NFS 那份副本缺该文件，故另建 run 根：
# history_config.txt 取自官方下载（内容 perceptual-framesamp-modul.yaml），79999 为指向 NFS 权重的 symlink。
CKPT=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/eval-out/mmevla-ckpt/perceptual-framesamp-modul/79999
SAVE_ROOT="${SAVE_ROOT:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/eval-out/mmevla-${RUN_TAG}-r${ROUND}-s${SHARD}}"
SAVE_DIR="$SAVE_ROOT/mmevla-testhard/ckpt79999/seed7"
LIMIT="${LIMIT:-0}"
mkdir -p "$SAVE_ROOT"
SERVER_LOG="$SAVE_ROOT/server.log"
EP_START=$((10 * (ROUND - 1) + SHARD))
[[ -d "$CKPT/params" ]] || { echo "错误: checkpoint 缺 params: $CKPT"; echo "EXIT_CODE=1"; exit 1; }
[[ -x "$REPO/robomme_env/bin/python" ]] || { echo "错误: robomme_env 不存在"; echo "EXIT_CODE=1"; exit 1; }
echo "=== MMEVLA_SHARD host=$(hostname) job=${SLURM_JOB_ID:-none} HEAD=$(git -C "$REPO" rev-parse HEAD) round=$ROUND shard=$SHARD port=$PORT ckpt=$CKPT ep_start=$EP_START start=$(date -Is) ==="
nvidia-smi --query-gpu=name,driver_version,compute_mode --format=csv,noheader
# 端口占用守卫：已被占用就会把别人的服务误判为就绪、静默产出空结果
if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
  exec 3>&-
  echo "错误: 端口 ${PORT} 起跑前已被占用"; echo "EXIT_CODE=1"; exit 1
fi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
  uv run --frozen --no-sync scripts/serve_policy.py --seed=7 --port="$PORT" \
    policy:checkpoint --policy.dir="$CKPT" --policy.config=mme_vla_suite >> "$SERVER_LOG" 2>&1 &
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
for pass in 1 2 3; do
  errors=$(grep -c '"status": "error"' "$SAVE_DIR/episodes.jsonl" 2>/dev/null || echo 0)
  if [[ "$errors" -gt 55 ]]; then echo "RETRY_CAP_HIT round=$ROUND shard=$SHARD errors=$errors"; RC=4; break; fi
  echo "EVAL_PASS $pass errors_so_far=$errors $(date -Is)"
  # GL 计算节点设有 HTTP 代理变量，websocket 客户端连本机 server 会走代理被拒（proxy rejected connection: HTTP 403）：
  # 评估进程去掉代理变量并显式连 127.0.0.1
  ( cd examples/robomme && env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
      NO_PROXY=127.0.0.1,localhost PYTHONUNBUFFERED=1 GLIBC_TUNABLES=glibc.rtld.optional_static_tls=16384 \
      "$REPO/robomme_env/bin/python" eval.py --args.host=127.0.0.1 --args.port="$PORT" --args.model_seed=7 --args.model_ckpt_id=79999 \
      --args.policy_name=mmevla-testhard --args.episode_start="$EP_START" --args.episode_stride=20 \
      --args.max_episodes="$LIMIT" --args.save_dir="$SAVE_ROOT" ${ONLY_TASKS:+--args.only_tasks="$ONLY_TASKS"} ) &
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
  unresolved=$("$REPO/robomme_env/bin/python" -c "
import json,sys
last={}
for l in open(sys.argv[1]):
    r=json.loads(l); last[(r['task'],r['episode'])]=r['status']
print(sum(v=='error' for v in last.values()))" "$SAVE_DIR/episodes.jsonl" 2>/dev/null || echo 0)
  echo "EVAL_PASS_END $pass rc=$RC unresolved_errors=$unresolved"
  [[ "$unresolved" = 0 ]] && break
done
# 有未解决的 error 身份时不得以 0 退出（第一轮实测：只记最后一遍 eval 的返回码会把中止的片报成成功）
[[ "$RC" = 0 && "${unresolved:-0}" != 0 ]] && RC=3
echo "EVAL_RC=$RC end=$(date -Is)"
echo "EXIT_CODE=$RC"
exit "$RC"
