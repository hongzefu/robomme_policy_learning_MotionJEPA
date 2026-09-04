#!/usr/bin/env bash
# driver：把 MotionJEPA run wan-v8-filter10-72ep-a 的 encoder+wan_decoder（整份 checkpoint_epoch_72.pt）
# 上传到 HF **private** model repo HongzeFu/MotionJEPA，并做元数据 + 回读两层验收。在 detached tmux 里跑。
#
# 与同目录 run_export.sh（bucket + 88 万小文件）的关键差异：
#   1. 目标是 model repo，走 `hf upload`，不是 `hf buckets sync`；
#   2. **不复制 token**。run_export.sh 把 HF_HOME 改指后再 cp ~/.cache/huggingface/{token,stored_tokens}，
#      那会把密钥复制成第二份落盘。这里改用 HF_TOKEN_PATH 指回原处：实测 HF_HOME 单独改指会让
#      `hf auth whoami` 报 Not logged in（XDG_CACHE_HOME 改指同样会，因为 HF_HOME 默认值由它派生），
#      而 HF_HOME + HF_TOKEN_PATH 组合能正常返回 HongzeFu；HF_STORED_TOKENS_PATH 自动跟随
#      dirname(HF_TOKEN_PATH)，不必单独设。
#   3. 每条 `hf upload` 都带 --private：CLI 内部硬调 create_repo(exist_ok=True, private=private)，
#      而 --private 默认 None → 新建时是 public。repo_id 打错一个字母就会静默新建 public repo
#      并把 954 MB 权重传上去。repo 已存在时该参数被忽略，无副作用。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }

REPO_ID="HongzeFu/MotionJEPA"
RUN_NAME="wan-v8-filter10-72ep-a"
TAG="${RUN_NAME}-e72"
SRC="$REPO/v1-store/external/motionjepa/$RUN_NAME"
DOC="$REPO/docs/dataset-build-doc/hf-export-motionjepa-encoder-v1"
EXPORT_ROOT="$REPO/v1-store/exports/hf-model-motionjepa-encoder-v1"
STAGE="$EXPORT_ROOT/stage"
VERIFY="$EXPORT_ROOT/verify"
HF=/home/hongzefu/.local/bin/hf

CKPT_SHA=bae960373041629e976a1f4a7d6d48ca3c51786c827146a3ee10bf7b034bc15a
CFG_SHA=99548a6ca23522c235281e45819ae6d5e96a916709cb4b9c0b47142832c90946
CKPT_BYTES=954853147

export HF_HOME="$REPO/v1-store/cache/hf"                  # AGENTS 第 14 条：缓存进 v1-store
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"     # 凭据留在原处，不复制第二份
export HF_XET_CACHE="$REPO/v1-store/cache/hf-xet"         # xet 上传侧 shard-cache/staging 也收进 v1-store
export UV_LINK_MODE=copy
unset HF_HUB_OFFLINE || true

echo "阶段0开始：预检"
# 落点必须是本机实体目录，不能穿透 symlink 往 turbo 只读归档写（AGENTS 第 13/14 条）
for d in "$EXPORT_ROOT" "$REPO/v1-store/exports"; do
  [ -L "$d" ] && { echo "错误: $d 是符号链接，拒绝写入"; exit 1; }
done
mkdir -p "$STAGE/$RUN_NAME" "$VERIFY" "$EXPORT_ROOT/logs" "$HF_XET_CACHE"

# 源文件指纹必须等于仓库钉死的常量
got_ckpt="$(sha256sum "$SRC/checkpoint_epoch_72.pt" | cut -d' ' -f1)"
got_cfg="$(sha256sum "$SRC/config.yaml" | cut -d' ' -f1)"
got_bytes="$(stat -c '%s' "$SRC/checkpoint_epoch_72.pt")"
[ "$got_ckpt" = "$CKPT_SHA" ] || { echo "错误: ckpt sha256 $got_ckpt != $CKPT_SHA"; exit 1; }
[ "$got_cfg" = "$CFG_SHA" ]   || { echo "错误: config sha256 $got_cfg != $CFG_SHA"; exit 1; }
[ "$got_bytes" = "$CKPT_BYTES" ] || { echo "错误: ckpt 字节 $got_bytes != $CKPT_BYTES"; exit 1; }
echo "  ✓ 源文件指纹与常量一致"

"$HF" auth whoami
echo "阶段0完成"

echo "阶段1开始：搭 stage 树（ckpt 走硬链接，不复制 954 MB）"
cp "$DOC/model-card.md"          "$STAGE/README.md"
cp "$DOC/records/SHA256SUMS.txt" "$STAGE/SHA256SUMS.txt"
cp "$SRC/config.yaml"                "$STAGE/$RUN_NAME/config.yaml"
cp "$SRC/SHA256SUMS.src-vs-copy.txt" "$STAGE/$RUN_NAME/SHA256SUMS.src-vs-copy.txt"
ln -f "$SRC/checkpoint_epoch_72.pt"  "$STAGE/$RUN_NAME/checkpoint_epoch_72.pt"
# stage 里只能有实体文件或符号链接文件：hf upload 用 Path.glob("**/*")，而 Path.walk 默认
# follow_symlinks=False，符号链接**目录**的内容会被静默整体漏传
find "$STAGE" -type l -print | grep . && { echo "错误: stage 内出现符号链接"; exit 1; } || true
find "$STAGE" -type f | sort
echo "阶段1完成"

echo "阶段2开始：commit 1（小文件先行，--exclude '*.pt' 跨目录生效）"
ok=0
for attempt in 1 2 3; do
  if "$HF" upload "$REPO_ID" "$STAGE" . --private --exclude "*.pt" \
       --commit-message "docs: model card + ${RUN_NAME} 冻结 config 与 SHA256SUMS"; then ok=1; break; fi
  echo "commit 1 第 $attempt 次失败，60s 后重试"; sleep 60
done
[ "$ok" = 1 ] || { echo "commit 1 三次仍失败"; exit 1; }
echo "阶段2完成"

echo "阶段3开始：commit 2（954 MB ckpt）"
ok=0
for attempt in 1 2 3; do
  if "$HF" upload "$REPO_ID" "$STAGE/$RUN_NAME/checkpoint_epoch_72.pt" \
       "$RUN_NAME/checkpoint_epoch_72.pt" --private \
       --commit-message "上传 checkpoint_epoch_72.pt（EMA encoder+wan_decoder，sha256 ${CKPT_SHA:0:8}…）"; then ok=1; break; fi
  echo "commit 2 第 $attempt 次失败，60s 后重试（xet 已传分块服务端去重，等价续传）"; sleep 60
done
[ "$ok" = 1 ] || { echo "commit 2 三次仍失败"; exit 1; }
echo "阶段3完成"

echo "阶段4开始：打 tag $TAG"
"$HF" repos tag create "$REPO_ID" "$TAG" \
  -m "MotionJEPA wan-latent-v7 · run $RUN_NAME · epoch 72（36/36，最后一个）" || \
  echo "  tag 已存在或创建失败，见上方报错（重打须先 hf repos tag delete）"
echo "阶段4完成"

echo "阶段5开始：L1 元数据验收（不下载，内容寻址）"
uv run --no-sync python "$REPO/scripts/dataset/hf_export/verify_model_repo.py" \
  --repo-id "$REPO_ID" --src "$SRC" --stage "$STAGE" --json "$DOC/records/blobs_api.json"
echo "阶段5完成"

echo "阶段6开始：L2 回读闭环（禁 xet + 全新空目录 + force-download）"
rm -rf "$VERIFY"; mkdir -p "$VERIFY"
HF_HUB_DISABLE_XET=1 "$HF" download "$REPO_ID" --revision "$TAG" --local-dir "$VERIFY" --dry-run
ok=0
for attempt in 1 2 3; do
  if HF_HUB_DISABLE_XET=1 "$HF" download "$REPO_ID" --revision "$TAG" \
       --local-dir "$VERIFY" --force-download --max-workers 4; then ok=1; break; fi
  echo "回读第 $attempt 次失败，60s 后重试"; sleep 60
done
[ "$ok" = 1 ] || { echo "回读三次仍失败"; exit 1; }
( cd "$VERIFY" && sha256sum -c --strict SHA256SUMS.txt )
cmp "$VERIFY/README.md" "$DOC/model-card.md"
cmp "$VERIFY/$RUN_NAME/SHA256SUMS.src-vs-copy.txt" "$SRC/SHA256SUMS.src-vs-copy.txt"
echo "RESULT=PASS"
echo "阶段6完成"

echo "阶段7开始：收尾断言（源文件未被 stage 硬链接连带改动）"
after_ckpt="$(sha256sum "$SRC/checkpoint_epoch_72.pt" | cut -d' ' -f1)"
[ "$after_ckpt" = "$CKPT_SHA" ] || { echo "错误: 源 ckpt sha256 变化 $after_ckpt"; exit 1; }
echo "  ✓ 源 ckpt sha256 未变"
echo "全部完成"
