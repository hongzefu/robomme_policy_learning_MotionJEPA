#!/usr/bin/env bash
# 异地机器：把 bucket robomme-4task-v2-1600ep-trainset-20260920-v1 还原成可直接开训的库。
#
# 做四件事：全量下载 → sha256 全量校验 → 解开 source/data_tars → norm_stats 归位，
# 再取两个训练必需的外部权重（pi05_base 与 paligemma_tokenizer，均匿名可拉）。
# 每步幂等，整脚本可重跑。
#
# 前提：仓库 clone 在 /scratch/hongze/robomme_policy_learning_MotionJEPA。
# 换别的路径也能跑，但要多做两件事：一是在起跑脚本里显式 export MMEVLA_FRAMESAMP_SOURCE
# 与 MMEVLA_FRAMESAMP_MANIFEST（两个 store_meta.json 里写死的是本仓库这条绝对路径），
# 二是 scripts/training/paths.sh 里有一张顶层前缀白名单，本路径的前缀在名单内、别的不在，
# 不在名单内 paths.sh 会直接 exit 1，需要在那里加一条常量前缀。
set -euo pipefail

BUCKET="hf://buckets/HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO（缺 pyproject.toml）"; exit 1; }
DS="$REPO/v1-store/datasets/4task-v2-1600ep-604f16da"
ASSETS="$REPO/v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"

# 缓存类环境变量逐项指向 v1-store，**禁止覆盖 HOME**
export HF_HOME="$REPO/v1-store/cache/hf"
export HF_XET_CACHE="$REPO/v1-store/cache/hf-xet"
export UV_LINK_MODE=copy
unset HF_HUB_OFFLINE || true

# 项目 .venv 的 huggingface_hub 由 openpi 依赖钉死在 0.32.3，没有 bucket 子命令；
# 用 uvx 拉一份独立 CLI，pyproject/uv.lock/.venv 一律不碰。公开 bucket 不需要 token。
hf() { uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf "$@"; }
command -v uvx >/dev/null || { echo "错误: PATH 里找不到 uvx"; exit 1; }

echo "=== 1/5 下载（约 669 GiB；只要训练可加 --include 跳过 155 GiB 的对拍材料）"
mkdir -p "$DS"
hf sync "$BUCKET" "$DS"

echo "=== 2/5 全量 sha256 校验（覆盖除清单自身外的全部字节）"
( cd "$DS" && sha256sum -c --strict SHA256SUMS.pre.txt )
echo "RESTORE_SHA256=OK"

echo "=== 3/5 解开 source/data_tars → source/data/（605,611 个 pkl，约 225 GiB）"
mkdir -p "$DS/source"
for t in "$DS"/source/data_tars/*.tar; do
  tar -xf "$t" -C "$DS/source"
done
n=$(find "$DS/source/data" -name '*.pkl' | wc -l)
[ "$n" = "605611" ] || { echo "错误: source/data 解出 $n 个 pkl，期望 605611"; exit 1; }
echo "RESTORE_PKL=605611"

echo "=== 4/5 norm_stats 归位"
mkdir -p "$ASSETS"
cp -f "$DS/assets/robomme/norm_stats.json" "$ASSETS/robomme/norm_stats.json" 2>/dev/null \
  || { mkdir -p "$ASSETS/robomme"; cp -f "$DS/assets/robomme/norm_stats.json" "$ASSETS/robomme/norm_stats.json"; }
echo -n "NORM_STATS_SHA256="; sha256sum "$ASSETS/robomme/norm_stats.json" | cut -d' ' -f1

echo "=== 5/5 取外部权重（只要这两个；均从 gs:// 匿名可拉，不需要任何 token）"
cd "$REPO"
uv run python scripts/assets/fetch_assets.py fetch  --assets pi05_base,paligemma_tokenizer
uv run python scripts/assets/fetch_assets.py verify --assets pi05_base,paligemma_tokenizer --level full

cat <<'TXT'

还原完成。接着跑：
  bash scripts/dataset/hf_export/v2_1600ep_runner/train-60k.sh <run_name>   # 512 modulation
  bash scripts/dataset/hf_export/v2_1600ep_runner/train-80k.sh <run_name>   # 512+motion modulation

不需要 SigLIP / Wan VAE / MotionJEPA encoder：motion token 已离线烘焙进
motion/motion_token.f32.bin，history config 里的 source_run 只是身份字符串。
TXT
