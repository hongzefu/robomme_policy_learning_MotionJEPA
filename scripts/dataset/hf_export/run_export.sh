#!/usr/bin/env bash
# driver：打包 → 上传 bucket → 全量回读 → sha256 校验。在 detached tmux 里跑。
# 每步幂等，可中断后整脚本重跑续传。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXPORT_ROOT="/data/hongzefu/hf-export-robomme-vla-motionjepa-v1"
BUCKET="hf://buckets/HongzeFu/robomme-vla-motionjepa-v1"
export HF_HOME="$EXPORT_ROOT/hf-home"   # 缓存不写 NFS home
export UV_LINK_MODE=copy
mkdir -p "$EXPORT_ROOT/logs" "$HF_HOME"
# HF_HOME 改指后 CLI 找不到默认位置的登录 token，须带过来（缺这步上传会 401）
for f in token stored_tokens; do
  [ -f "$HF_HOME/$f" ] || cp "$HOME/.cache/huggingface/$f" "$HF_HOME/$f"
done

echo "阶段1开始：打包+哈希（NFS 单遍读 → $EXPORT_ROOT/stage）"
cd "$REPO"
uv run scripts/dataset/hf_export/pack_and_hash.py --workers 8
cp scripts/dataset/hf_export/bucket_README.md "$EXPORT_ROOT/stage/README.md"
echo "阶段1完成"

LOCAL_FILES=$(find "$EXPORT_ROOT/stage" -type f | wc -l)
LOCAL_BYTES=$(du -sb "$EXPORT_ROOT/stage" | cut -f1)
echo "本地暂存：$LOCAL_FILES 个文件，$LOCAL_BYTES 字节"

echo "阶段2开始：上传 bucket（增量 sync，重试至多 3 次）"
ok=0
for attempt in 1 2 3; do
  if hf buckets sync "$EXPORT_ROOT/stage" "$BUCKET"; then ok=1; break; fi
  echo "上传第 $attempt 次中断，60s 后重试续传"
  sleep 60
done
[ "$ok" = 1 ] || { echo "上传三次仍失败"; exit 1; }
hf buckets info HongzeFu/robomme-vla-motionjepa-v1
echo "阶段2完成"

echo "阶段3开始：全量回读到 $EXPORT_ROOT/verify"
mkdir -p "$EXPORT_ROOT/verify"
ok=0
for attempt in 1 2 3; do
  if hf buckets sync "$BUCKET" "$EXPORT_ROOT/verify"; then ok=1; break; fi
  echo "回读第 $attempt 次中断，60s 后重试续传"
  sleep 60
done
[ "$ok" = 1 ] || { echo "回读三次仍失败"; exit 1; }
echo "阶段3完成"

echo "阶段4开始：sha256 校验（sha256sum -c，并行分块）"
uv run scripts/dataset/hf_export/verify_download.py --workers 6
echo "阶段4完成"
