#!/usr/bin/env bash
# 从 HF 桶拉回原生 4×4 framesamp 表（1127-4task-modul-fs4x4-512-1024-60k-plan.md 第二节 G1）。
#
# 留档脚本：固化 2026-11-27 主会话实际使用的拉回命令（tmux fs4-fetch）。目标目录必须不存在
# 才执行，绝不覆盖或续写已有目录。拉回后用 check_framesamp_4x4_fetch.py 做 G1/G2 判定。
#
# 用法（约 1 小时，按长任务规则放 detached tmux）：
#   bash scripts/dataset/fetch_framesamp_4x4.sh [DST]
# DST 默认 $DS/framesamp-4x4，DS=v1-store/datasets/4task-v2-1600ep-604f16da。
set -euo pipefail

DS="${DS:-/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/4task-v2-1600ep-604f16da}"
DST="${1:-${DS}/framesamp-4x4}"
HF_BIN="${HF_BIN:-/scratch/hongze/MotionJEPA/.venv/bin/hf}"
SRC="hf://buckets/HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1/framesamp"

if [[ -e "${DST}" ]]; then
  echo "拒绝：目标已存在 ${DST}（本脚本只写全新目录，不覆盖、不续写）" >&2
  exit 2
fi
if [[ ! -x "${HF_BIN}" ]]; then
  echo "找不到 hf 可执行文件: ${HF_BIN}" >&2
  exit 2
fi

echo "[fetch] ${SRC} -> ${DST}  开始 $(date -Is)"
HF_HOME=/scratch/hongze/.cache/huggingface-hongzefu "${HF_BIN}" buckets sync "${SRC}" "${DST}"
echo "[fetch] 完成 $(date -Is)；下一步: uv run python scripts/dataset/check_framesamp_4x4_fetch.py --store ${DST} ..."
