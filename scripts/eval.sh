# 在线评估启动（commitV4.5 收敛后仅存 framesamp 三变体）：
# perceptual-framesamp-context, perceptual-framesamp-modul, perceptual-framesamp-expert

#### set your own parameters ####
MODEL_TYPE="perceptual-framesamp-modul"
SEED=7          # model seed for evaluation; change this to use different seeds for multiple runs
CKPT_ID=79999   # ckpt id for evaluation; change this to use different checkpoints
GPU_ID_server=0 # gpu id for server; when set, the VLA policy server will run on this GPU
GPU_ID_client=1 # gpu id for client; when set, the RoboMME environment will run on this GPU
#--------------------------------#



find_free_port() {
  local min=${1:-2000}
  local max=${2:-30000}
  local port
  local tries=5000  # max tries to find a free port

  for ((i=0; i<tries; i++)); do
    port=$(shuf -i"${min}"-"${max}" -n1)
    if ! lsof -iTCP:"${port}" -sTCP:LISTEN &>/dev/null; then
      echo "${port}"
      return 0
    fi
  done

  echo "ERROR: not found free port in range ${min}-${max}" >&2
  return 1
}
PORT=$(find_free_port)


CONFIG_TYPE="mme_vla_suite"
EXTRA_ARGS=""


session_name="${MODEL_TYPE}_ckpt${CKPT_ID}_seed${SEED}_port${PORT}"
echo "Evaluating $MODEL_TYPE with seed $SEED and ckpt id $CKPT_ID on port $PORT"


# Check if tmux session already exists
tmux has-session -t $session_name 2>/dev/null

if [ $? != 0 ]; then
    # Create new tmux session with first window for serve_policy
    tmux new-session -d -s $session_name -n "serve_policy"
    tmux send-keys -t $session_name:serve_policy "CUDA_VISIBLE_DEVICES=$GPU_ID_server uv run scripts/serve_policy.py --seed=$SEED  --port=$PORT policy:checkpoint --policy.dir=v1-store/train-runs/$CONFIG_TYPE/$MODEL_TYPE/$CKPT_ID --policy.config=$CONFIG_TYPE" Enter

    sleep 30

    # Create second window for eval in the same session
    tmux new-window -t $session_name -n "eval"
    tmux send-keys -t $session_name:eval "micromamba activate robomme" Enter
    # 评估客户端跑在独立的 micromamba robomme 环境（RoboMME 仿真器依赖装在该环境、
    # 不在仓库 uv venv），故此处保留该环境的 python、不换 uv run——非 uv 管理环境，
    # 不适用「禁裸 python」（2026-08-30 用户拍板）
    tmux send-keys -t $session_name:eval "CUDA_VISIBLE_DEVICES=$GPU_ID_client python examples/robomme/eval.py --args.model_seed=$SEED --args.port=$PORT --args.policy_name=$MODEL_TYPE --args.model_ckpt_id=$CKPT_ID ${EXTRA_ARGS}; tmux wait-for -S eval-done" Enter

    # Wait for eval to complete, or exit if tmux session is killed
    tmux wait-for eval-done &
    wait_pid=$!
    while kill -0 $wait_pid 2>/dev/null; do
        tmux has-session -t $session_name 2>/dev/null || { kill $wait_pid 2>/dev/null; echo "Tmux session killed, exiting."; exit 1; }
        sleep 2
    done
    tmux kill-session -t $session_name 2>/dev/null || true

else
    echo "Tmux session ${session_name} already exists. Change the port or use a different session."
fi
