#!/usr/bin/env bash
# 本机数据处理链路（scripts/dataset/）的**唯一**路径与环境源。建库域脚本一律 source 本文件，
# 不再各自硬编码路径；训练域（scripts/training/）自带同构的 paths.sh，两域互不引用。
#
# v2-motionmem（2026-09-03）起的口径：
#   1. **工作副本在本机** /data/hongzefu/robomme_policy_learning_MotionJEPA，turbo 那份转为只读归档
#      （AGENTS.md 第 13 条）；仓库位置 fail-loud：必须位于本机工作副本前缀或 turbo 归档前缀下。
#   2. **不覆盖 HOME**。缓存类环境变量逐项显式指向 v1-store/cache/。
#   3. **原始 H5 = 16 任务全集** /data/hongzefu/robomme_data_h5（16 任务 × 100 ep，本机 NVMe，永久保留）；
#      建库只取 4 个目标任务的前 N 个 episode（scan_manifest.py --tasks … --episodes-per-task N）。
#      旧版「恰好 4 个 h5 + 各带 _metadata.json sidecar + 400 ep」的校验不再适用（16 任务目录无 sidecar、各 100 ep），
#      改为「4 个目标 h5 存在 + 各 ≥ N 个 episode + 逐文件 sha256 记入库内 meta/input_manifest.json」。
#   4. 集群链路（gl/ 目录、Slurm 提交）已删除；本文件不再提供任何 turbo 暂存 / NFS venv 变量。
#
# 环境 B（AWS 单机，2026-09-04 起，AGENTS.md「运行环境判定」）：
#   仓库工作副本在 /scratch/hongze/robomme_policy_learning_MotionJEPA，本机没有 /data/hongzefu 与 /nfs/turbo。
#   前缀白名单加第三项 AWS_WORK_PREFIX；RAW_H5_DIR 默认值按仓库前缀分叉（AWS 下默认
#   /scratch/hongze/robomme_data_h5——只含 4 个目标 h5 的目录）；MJ_REPO 允许环境变量覆盖（本机只读副本
#   在 /scratch/hongze/MotionJEPA）。turbo / /data 两个前缀与其默认值原样保留给环境 A。

set -euo pipefail

readonly TURBO_PREFIX="/nfs/turbo/coe-chaijy-unreplicated/hongzefu/"
readonly LOCAL_WORK_PREFIX="/data/hongzefu/"
readonly AWS_WORK_PREFIX="/scratch/hongze/"

V1_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly V1_SCRIPT_DIR
REPO_ROOT="$(cd "${V1_SCRIPT_DIR}/../.." && pwd)"
readonly REPO_ROOT

# 根自证断言：以 pyproject.toml 存在性判根，层数数错时 fail-loud，不在错误位置造假目录树
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "错误: 仓库根解析失败 ${REPO_ROOT}（缺 pyproject.toml）" >&2
  exit 1
fi

case "${REPO_ROOT}/" in
  "${LOCAL_WORK_PREFIX}"*) ;;
  "${TURBO_PREFIX}"*) ;;
  "${AWS_WORK_PREFIX}"*) ;;
  *)
    echo "错误: 仓库必须位于 ${LOCAL_WORK_PREFIX}(环境 A 本机工作副本)、${TURBO_PREFIX}(turbo 只读归档) 或 ${AWS_WORK_PREFIX}(环境 B AWS 单机) 下, 当前为 ${REPO_ROOT}" >&2
    echo "      产物根 v1-store/ 随仓库走, 不得落到这三处之外(见 AGENTS.md 第 13、14 条)。" >&2
    exit 1
    ;;
esac

# ── 项目内产物根（整体不进 Git，见 .gitignore 的 /v1-store/）────────────────────
readonly V1_STORE="${REPO_ROOT}/v1-store"
readonly MODELS_DIR="${V1_STORE}/models"
readonly DATASETS_DIR="${V1_STORE}/datasets"
readonly CACHE_DIR="${V1_STORE}/cache"
readonly LOGS_DIR="${V1_STORE}/logs"
readonly EXTERNAL_DIR="${V1_STORE}/external"
readonly WAN_VENV="${V1_STORE}/venvs/wan"

# ── 原始 H5 与四个目标任务 ─────────────────────────────────────────────────────
# 环境 A：16 任务全集 /data/hongzefu/robomme_data_h5（本机原件）；
# 环境 B：/scratch/hongze/robomme_data_h5（从 Yinpei/robomme_data_h5 只下 4 个目标任务，见 external-assets-lock.md 第五节）
case "${REPO_ROOT}/" in
  "${AWS_WORK_PREFIX}"*) _raw_h5_default="/scratch/hongze/robomme_data_h5" ;;
  *) _raw_h5_default="/data/hongzefu/robomme_data_h5" ;;
esac
RAW_H5_DIR="${RAW_H5_DIR:-${_raw_h5_default}}"
unset _raw_h5_default
readonly TARGET_TASKS=(ButtonUnmask ButtonUnmaskSwap VideoUnmask VideoUnmaskSwap)
readonly TARGET_TASKS_CSV="ButtonUnmask,ButtonUnmaskSwap,VideoUnmask,VideoUnmaskSwap"

# ── MotionJEPA 只读副本与 encoder 资产（motion-memory-plan.md 红线 2 / 9 / 10）───────
# 环境 A 默认 turbo 只读副本；环境 B 用环境变量 MJ_REPO 指向 /scratch/hongze/MotionJEPA（同一 commit）
MJ_REPO="${MJ_REPO:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA}"
readonly MJ_COMMIT="2a484ad960ed6155321dc34def9011eb119f857f"
readonly ENCODER_RUN_DIR="${EXTERNAL_DIR}/motionjepa/wan-v8-filter10-72ep-a"
readonly ENCODER_CKPT="checkpoint_epoch_72.pt"

# ── 环境变量：显式逐项指向 v1-store/cache，**不动 HOME** ────────────────────────
export OPENPI_DATA_HOME="${MODELS_DIR}"
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"
export HF_HOME="${CACHE_DIR}/hf"
export HF_HUB_OFFLINE=1
export JAX_COMPILATION_CACHE_DIR="${CACHE_DIR}/jax"
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1

v1_prepare_dirs() {
  mkdir -p "${MODELS_DIR}" "${DATASETS_DIR}" "${CACHE_DIR}" "${LOGS_DIR}" "${EXTERNAL_DIR}" \
    "${XDG_CACHE_HOME}" "${HF_HOME}" "${JAX_COMPILATION_CACHE_DIR}"
}

v1_require_cmd() {
  local name
  for name in "$@"; do
    command -v "${name}" >/dev/null 2>&1 || { echo "错误: 缺少命令 ${name}" >&2; exit 1; }
  done
}

# 原始 H5 自洽性：4 个目标 h5 存在、可读、各 ≥ N 个 episode（N 默认 10）。sha256 由 finalize_checks.py
# hash-inputs 写进库内 meta/input_manifest.json，本函数不重复算（82 GB 约 5–10 min）。
v1_validate_raw_h5() {
  local dir="${1:-${RAW_H5_DIR}}"
  local need="${2:-10}"
  local t p
  [[ -d "${dir}" ]] || { echo "错误: 原始 H5 目录不存在: ${dir}" >&2; exit 1; }
  for t in "${TARGET_TASKS[@]}"; do
    p="${dir}/record_dataset_${t}.h5"
    [[ -f "${p}" ]] || { echo "错误: 缺少目标 H5: ${p}" >&2; exit 1; }
  done
  UV_LINK_MODE=copy uv run --no-sync python - "${dir}" "${need}" "${TARGET_TASKS[@]}" <<'PY'
import sys, h5py
d, need, tasks = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
for t in tasks:
    p = f"{d}/record_dataset_{t}.h5"
    with h5py.File(p, "r") as f:
        n = sum(1 for k in f.keys() if k.startswith("episode_"))
    if n < need:
        raise SystemExit(f"错误: {p} 只有 {n} 个 episode < {need}")
    print(f"  ✓ {p}: {n} episode")
PY
}

# 项目内模型齐全性（SigLIP；训练侧还需 tokenizer 与 pi05_base）。
v1_require_models() {
  local need_train="${1:-0}"
  local siglip="${MODELS_DIR}/pi05_vision_encoder/siglip_params.pkl"
  [[ -f "${siglip}" ]] || { echo "错误: 缺少项目内 SigLIP 权重: ${siglip}" >&2; exit 1; }
  local names="siglip_params"
  if [[ "${need_train}" = "1" ]]; then
    local tok="${MODELS_DIR}/big_vision/paligemma_tokenizer.model"
    local pi05="${MODELS_DIR}/openpi-assets/checkpoints/pi05_base/params"
    [[ -f "${tok}" ]] || { echo "错误: 缺少项目内 PaliGemma tokenizer: ${tok}" >&2; exit 1; }
    [[ -f "${pi05}/commit_success.txt" ]] || { echo "错误: 缺少项目内 pi05_base: ${pi05}" >&2; exit 1; }
    names="${names},paligemma_tokenizer,pi05_base"
  fi
  # 内容级校验：上面的 [[ -f ]] 只答「文件在不在」，这一段答「是不是 ASSETS_LOCK.json 钉死的那份」。
  # cheap 档 = 逐文件字节数 + 首尾各 1 MiB 摘要（O(ms)）；它挡不住保持长度的中段字节篡改，
  # 那一层由 fetch_assets.py verify --level full 与各重载点（load_encoder/load_vae）兜。
  if [[ "${V1_SKIP_ASSET_VERIFY:-0}" = "1" ]]; then
    echo "⚠ V1_SKIP_ASSET_VERIFY=1: 跳过资产内容校验(${names})，本次运行不保证用的是钉死的那份权重" >&2
  else
    UV_LINK_MODE=copy uv run --no-sync python "${REPO_ROOT}/scripts/assets/fetch_assets.py" \
      verify --level cheap --assets "${names}" || exit 1
  fi
}

# wan 子 venv 与 encoder 资产齐全性（S0 落地件）
v1_require_wan() {
  [[ -x "${WAN_VENV}/bin/python" ]] || { echo "错误: 缺少 wan 子 venv: ${WAN_VENV}（uv sync --project scripts/dataset/wan）" >&2; exit 1; }
  [[ -f "${ENCODER_RUN_DIR}/${ENCODER_CKPT}" && -f "${ENCODER_RUN_DIR}/config.yaml" ]] || {
    echo "错误: 缺少 encoder 资产: ${ENCODER_RUN_DIR}/{${ENCODER_CKPT},config.yaml}" >&2; exit 1; }
  [[ -d "${HF_HOME}/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers" ]] || {
    echo "错误: 缺少 HF 缓存里的 Wan2.1 VAE: ${HF_HOME}/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers" >&2; exit 1; }
  # 同 v1_require_models：存在性之外再核内容（VAE snapshot revision + blob 字节、encoder ckpt 与 config）
  if [[ "${V1_SKIP_ASSET_VERIFY:-0}" = "1" ]]; then
    echo "⚠ V1_SKIP_ASSET_VERIFY=1: 跳过 wan 侧资产内容校验" >&2
  else
    UV_LINK_MODE=copy uv run --no-sync python "${REPO_ROOT}/scripts/assets/fetch_assets.py" \
      verify --level cheap --assets wan_vae,motionjepa_ckpt,motionjepa_config || exit 1
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "错误: paths.sh 只能被 source, 不能直接执行" >&2
  exit 1
fi
