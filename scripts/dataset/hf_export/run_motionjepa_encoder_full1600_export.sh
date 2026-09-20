#!/usr/bin/env bash
# driver：把 MotionJEPA encoder 训练 run wan-full1600-filter2-b176x4-72ep-a 的**全部 36 个**
# epoch checkpoint（epoch 2…72，每 2 epoch 一存，各 456 MiB，合计 17.26 GB / 49 文件）
# 上传到 HF **private** bucket HongzeFu/motionjepa-wan-full1600-72ep-v1，
# 并做「上传前 / 上传后」两遍独立 sha256 逐行对照验收。
#
# 这条 encoder 正是 VLA run v2-1600ep-m8x8-modul-motion-b128-80k 的 motion 特征来源
# （用的是其中的 epoch 72）。在此之前 HF 上**只有上一代** wan-v8-filter10-72ep-a
# （私有模型库 HongzeFu/MotionJEPA），这一版一个副本都没有。
#
# 蓝本是同目录 run_modul60k_ckpt_export.sh，六阶段骨架照抄。三处差异：
#   1. 源在**仓库外**（/scratch/hongze/MotionJEPA/runs/...），但与 EXPORT_ROOT 同在 /dev/md0，
#      硬链接照样成立（脚本里有同设备断言兜底）。
#   2. 这条 run 2026-09-14 就跑完了（train.log 尾部 TRAIN_DONE / EXIT_CODE=0），
#      **不需要分两趟**，一次跑到底。
#   3. 36 个 ckpt 是**平铺在根下**的，没有 step 目录那种天然批次边界。但它们**大小完全相同**
#      （各 477,432,433 字节），固定条数分组就是稳定分组，重跑时 --include 模式不会漂移，
#      因此不需要 plan_upload_batches.py。每批 8 个 = 3.82 GB，远低于实测会挂的 20 GB 阈值。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
cd "$REPO"

RUN_NAME="wan-full1600-filter2-b176x4-72ep-a"
BUCKET_ID="HongzeFu/motionjepa-wan-full1600-72ep-v1"
BUCKET="hf://buckets/$BUCKET_ID"
SRC="/scratch/hongze/MotionJEPA/runs/$RUN_NAME"
EXPORT_ROOT="$REPO/v1-store/exports/hf-motionjepa-encoder-full1600-72ep"
STAGE="$EXPORT_ROOT/stage"
VERIFY="$EXPORT_ROOT/verify"
TMPD="$EXPORT_ROOT/tmp"
BATCH_SIZE=8          # 每批 8 个 ckpt = 3.82 GB
EXPECT_CKPTS=36

export HF_HOME="$REPO/v1-store/cache/hf"
export HF_XET_CACHE="$REPO/v1-store/cache/hf-xet"
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export UV_PYTHON_INSTALL_DIR=/scratch/hongze/.cache/uv-python
export UV_LINK_MODE=copy
unset HF_HUB_OFFLINE || true

# 80k VLA 训练可能还在跑：在跑就限流，不跑就全速。
if pgrep -f "train\.py mme_vla_suite_b128_80k" > /dev/null 2>&1; then
  THROTTLE=(ionice -c 3 nice -n 19); NPROC=2
  echo "检测到 80k VLA 训练在跑 => 限流模式（ionice idle / nice 19 / NPROC=2）"
else
  THROTTLE=(); NPROC=8
  echo "无训练在跑 => 全速"
fi

[ -f "$REPO/v1-store/secrets/hf.env" ] || { echo "错误: 缺少 v1-store/secrets/hf.env"; exit 1; }
set -a; . "$REPO/v1-store/secrets/hf.env"; set +a
[ -n "${HF_TOKEN:-}" ] || { echo "错误: hf.env 未提供 HF_TOKEN"; exit 1; }

hf() { "${THROTTLE[@]}" uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf "$@"; }

bucket_stat() {  # $1 = size|totalFiles
  curl -sf -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/api/buckets/HongzeFu" \
    | python3 -c "
import sys, json
for b in json.load(sys.stdin):
    if b['id'] == '$BUCKET_ID':
        print(b.get('$1')); break
else:
    sys.exit('目标 bucket $BUCKET_ID 不在返回列表里')
"
}

hash_tree() {  # $1 = 目录, $2 = 输出文件（绝对路径，不能落在被扫描目录内）
  ( cd "$1" && find . -type f -not -name 'SHA256SUMS.*' -printf '%P\n' | sort \
      | "${THROTTLE[@]}" xargs -P "$NPROC" -n 16 sha256sum | sort -k 2 > "$2" )
}

# ---------------------------------------------------------------- 阶段 0
echo "阶段0开始：预检 + 建 bucket"
for d in "$REPO/v1-store/exports" "$EXPORT_ROOT"; do
  [ -L "$d" ] && { echo "错误: $d 是符号链接，拒绝写入"; exit 1; }
done
[ -d "$SRC" ] || { echo "错误: 源目录不存在 $SRC"; exit 1; }
[ "$STAGE" != "$SRC" ] || { echo "错误: STAGE 与 SRC 相同，拒绝继续"; exit 1; }
case "$STAGE" in "$EXPORT_ROOT"/*) : ;; *) echo "错误: STAGE 不在 EXPORT_ROOT 下"; exit 1 ;; esac
mkdir -p "$STAGE" "$VERIFY" "$TMPD" "$EXPORT_ROOT/logs" "$EXPORT_ROOT/logs/uploaded" "$HF_XET_CACHE"

# 源在仓库外，硬链接要求同设备。不同设备就直接停——否则 ln -f 报错，或被误改成 cp 白吃 17 GB。
SRC_DEV=$(stat -c %d "$SRC"); STAGE_DEV=$(stat -c %d "$STAGE")
[ "$SRC_DEV" = "$STAGE_DEV" ] || { echo "错误: 源($SRC_DEV)与 stage($STAGE_DEV)不同设备，硬链接不成立"; exit 1; }

# 这条 run 必须是已经跑完的
grep -q '^EXIT_CODE=0$' "$SRC/train.log" || { echo "错误: $SRC/train.log 未见 EXIT_CODE=0"; exit 1; }
echo "  源 train.log 见到 EXIT_CODE=0"

who="$(hf auth whoami 2>&1 | tr '\n' ' ' | tr -s ' ')"
echo "  whoami: $who"
case "$who" in *HongzeFu*) : ;; *) echo "错误: 身份不是 HongzeFu（$who）"; exit 1 ;; esac

if hf buckets create "$BUCKET_ID" --private 2>&1 | tee "$TMPD/create.log"; then
  echo "  bucket create 成功"
else
  echo "  bucket create 非零退出（可能已存在），继续走空库断言"
fi

pre_files="$(bucket_stat totalFiles)"; pre_size="$(bucket_stat size)"
echo "  目标 bucket 现状: files=$pre_files size=$pre_size"
if [ "$pre_files" != "0" ] && [ ! -f "$EXPORT_ROOT/logs/uploaded.marker" ]; then
  echo "错误: 目标 bucket 非空（files=$pre_files）且本地无本轮上传标记，停止并交用户裁决"
  exit 1
fi
echo "阶段0完成"

# ---------------------------------------------------------------- 阶段 1
echo "阶段1开始：搭 stage 树（硬链接，不复制 17.26 GB）+ 上传前 sha256"
( cd "$SRC" && find . -mindepth 1 -type d -printf '%P\n' ) | while read -r d; do
  mkdir -p "$STAGE/$d"
done
( cd "$SRC" && find . -mindepth 1 -type f -printf '%P\n' ) | while read -r f; do
  ln -f "$SRC/$f" "$STAGE/$f"
done
# 源 README 是 444，重跑时 cp 覆盖不了（Permission denied）。先删再拷，stage 里落成 644。
rm -f "$STAGE/README.md"
cp "$REPO/scripts/dataset/hf_export/motionjepa_encoder_full1600_bucket_README.md" "$STAGE/README.md"
chmod 644 "$STAGE/README.md"

if find "$STAGE" -type l -print | grep -q .; then
  echo "错误: stage 内出现符号链接"; find "$STAGE" -type l -print; exit 1
fi

mapfile -t CKPTS < <( cd "$STAGE" && find . -maxdepth 1 -name 'checkpoint_epoch_*.pt' -printf '%P\n' \
  | sed 's/checkpoint_epoch_//; s/\.pt//' | sort -n | sed 's/^/checkpoint_epoch_/; s/$/.pt/' )
[ "${#CKPTS[@]}" = "$EXPECT_CKPTS" ] || { echo "错误: ckpt 数应为 $EXPECT_CKPTS，实为 ${#CKPTS[@]}"; exit 1; }
# 分组稳定性来自「文件名按 epoch 数值排序」这一确定性，与大小无关，重跑时 --include 不会漂移。
# 要额外保证的是**每批字节量**低于实测会挂的阈值：2026-09-08 实测 16.2 GB 过、20 GB 挂，
# 这里按最大单文件 × BATCH_SIZE 取上界，卡在 16 GiB。
MAXSZ=$( cd "$STAGE" && printf '%s\n' "${CKPTS[@]}" | xargs stat -c %s | sort -n | tail -1 )
BATCH_MAX_BYTES=$(( MAXSZ * BATCH_SIZE ))
echo "  ${#CKPTS[@]} 个 ckpt，最大单个 $MAXSZ 字节，每批上界 $BATCH_MAX_BYTES 字节"
[ "$BATCH_MAX_BYTES" -le $((16 * 1024 * 1024 * 1024)) ] \
  || { echo "错误: 每批上界 $BATCH_MAX_BYTES 超过 16 GiB，调小 BATCH_SIZE"; exit 1; }

hash_tree "$STAGE" "$TMPD/SHA256SUMS.pre.txt"
mv "$TMPD/SHA256SUMS.pre.txt" "$STAGE/SHA256SUMS.pre.txt"

LOCAL_FILES=$(find "$STAGE" -type f | wc -l)
LOCAL_BYTES=$(find "$STAGE" -type f -printf '%s\n' | awk '{s+=$1} END{print s}')
PRE_LINES=$(wc -l < "$STAGE/SHA256SUMS.pre.txt")
echo "LOCAL_FILES=$LOCAL_FILES LOCAL_BYTES=$LOCAL_BYTES PRE_LINES=$PRE_LINES"
[ "$((PRE_LINES + 1))" = "$LOCAL_FILES" ] || { echo "错误: 清单行数+1 != stage 文件数"; exit 1; }
echo "阶段1完成"

# ---------------------------------------------------------------- 阶段 2
echo "阶段2开始：36 个 ckpt 按 $BATCH_SIZE 个一批上传 + wandb/ 子树 + 根下小文件"
touch "$EXPORT_ROOT/logs/uploaded.marker"

upload_batch() {  # $1 = 批号；$2.. = --include 模式（stage 根下的文件名）
  local tag="$1"; shift
  local args=() ok=0 attempt pat
  for pat in "$@"; do args+=(--include "$pat"); done
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf sync "$STAGE" "$BUCKET" "${args[@]}"; then ok=1; break; fi
    echo "  批 [$tag] 第 $attempt 次中断，60s 后重试续传"; sleep 60
  done
  [ "$ok" = 1 ] || { echo "错误: 批 [$tag] 八次仍失败"; exit 1; }
}

NBATCH=$(( (${#CKPTS[@]} + BATCH_SIZE - 1) / BATCH_SIZE ))
for ((b = 0; b < NBATCH; b++)); do
  DONE="$EXPORT_ROOT/logs/uploaded/ckpt-batch-$b.done"
  if [ -f "$DONE" ]; then echo "  [$((b+1))/$NBATCH] 已完成，跳过"; continue; fi
  GROUP=( "${CKPTS[@]:$((b * BATCH_SIZE)):$BATCH_SIZE}" )
  echo "  [$((b+1))/$NBATCH] ${#GROUP[@]} 个: ${GROUP[*]}"
  upload_batch "$b" "${GROUP[@]}"
  touch "$DONE"
done

if [ ! -f "$EXPORT_ROOT/logs/uploaded/wandb.done" ]; then
  echo "  上传 wandb/ 子树"
  ok=0
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf sync "$STAGE/wandb" "$BUCKET/wandb"; then ok=1; break; fi
    echo "  wandb 第 $attempt 次中断，60s 后重试"; sleep 60
  done
  [ "$ok" = 1 ] || { echo "错误: wandb 八次仍失败"; exit 1; }
  touch "$EXPORT_ROOT/logs/uploaded/wandb.done"
fi

# 根下平文件（config.yaml / train.log / *.csv / *.jsonl / README.md / SHA256SUMS.pre.txt）
mapfile -t ROOTFILES < <( cd "$STAGE" && find . -maxdepth 1 -type f ! -name 'checkpoint_epoch_*.pt' -printf '%P\n' | sort )
echo "  根下平文件 ${#ROOTFILES[@]} 个"
for f in "${ROOTFILES[@]}"; do
  ok=0
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf buckets cp "$STAGE/$f" "$BUCKET/$f"; then ok=1; break; fi
    echo "  $f 第 $attempt 次中断，60s 后重试"; sleep 60
  done
  [ "$ok" = 1 ] || { echo "错误: $f 八次仍失败"; exit 1; }
done
echo "UPLOAD_DONE batches=$NBATCH"
echo "阶段2完成"

# ---------------------------------------------------------------- 阶段 3
echo "阶段3开始：计数层核对"
BUCKET_FILES="$(bucket_stat totalFiles)"; BUCKET_BYTES="$(bucket_stat size)"
if [ "$BUCKET_FILES" != "$LOCAL_FILES" ] || [ "$BUCKET_BYTES" != "$LOCAL_BYTES" ]; then
  echo "  首次不符（files=$BUCKET_FILES bytes=$BUCKET_BYTES），统计接口可能异步滞后，120s 后复查"
  sleep 120
  BUCKET_FILES="$(bucket_stat totalFiles)"; BUCKET_BYTES="$(bucket_stat size)"
fi
echo "BUCKET_FILES=$BUCKET_FILES BUCKET_BYTES=$BUCKET_BYTES"
[ "$BUCKET_FILES" = "$LOCAL_FILES" ] || { echo "错误: 文件数不符 $BUCKET_FILES != $LOCAL_FILES"; exit 1; }
[ "$BUCKET_BYTES" = "$LOCAL_BYTES" ] || { echo "错误: 字节数不符 $BUCKET_BYTES != $LOCAL_BYTES"; exit 1; }
echo "阶段3完成"

# ---------------------------------------------------------------- 阶段 4
echo "阶段4开始：全量回读到 $VERIFY（17.26 GB）"
mkdir -p "$VERIFY"
ok=0
for attempt in 1 2 3; do
  if hf sync "$BUCKET" "$VERIFY"; then ok=1; break; fi
  echo "回读第 $attempt 次中断，60s 后重试续传"; sleep 60
done
[ "$ok" = 1 ] || { echo "回读三次仍失败"; exit 1; }
echo "阶段4完成"

# ---------------------------------------------------------------- 阶段 5
echo "阶段5开始：上传后 sha256（POST）+ 与上传前逐行对照"
hash_tree "$VERIFY" "$TMPD/SHA256SUMS.post.txt"
cp "$TMPD/SHA256SUMS.post.txt" "$VERIFY/SHA256SUMS.post.txt"
if diff "$STAGE/SHA256SUMS.pre.txt" "$TMPD/SHA256SUMS.post.txt"; then
  echo "SHA256_PRE_POST_DIFF=0"
else
  echo "错误: 上传前后 sha256 不一致（上方 diff 即差异行）"; exit 1
fi
( cd "$VERIFY" && sha256sum -c --strict SHA256SUMS.pre.txt > "$TMPD/check.txt" 2>&1 ) \
  || { echo "错误: sha256sum -c 失败"; grep -v ': OK$' "$TMPD/check.txt" | head -20; exit 1; }
echo "  sha256sum -c 校验行数: $(grep -c ': OK$' "$TMPD/check.txt")"

# 与 VLA bucket 的交叉锚点：epoch 72 必须等于 motion_provenance.json 记的那个 encoder
ENC_EXPECT="$(python3 -c "
import json
print(json.load(open('$REPO/v1-store/train-runs/mme_vla_suite_b128_80k/v2-1600ep-m8x8-modul-motion-b128-80k/motion_provenance.json'))['encoder']['checkpoint_sha256'])
")"
ENC_ACTUAL="$(grep '  checkpoint_epoch_72\.pt$' "$STAGE/SHA256SUMS.pre.txt" | cut -d' ' -f1)"
[ "$ENC_ACTUAL" = "$ENC_EXPECT" ] || {
  echo "错误: epoch 72 的 sha256 与 80k run 的 motion_provenance 不符"
  echo "  期望 $ENC_EXPECT"; echo "  实际 $ENC_ACTUAL"; exit 1
}
echo "ENCODER_MATCHES_VLA_RUN=OK $ENC_ACTUAL"
echo "RESULT=PASS"
echo "阶段5完成"

# ---------------------------------------------------------------- 阶段 6
echo "阶段6开始：收尾断言（源侧第三遍 sha256，证明硬链接未改动原件）"
hash_tree "$SRC" "$TMPD/SHA256SUMS.src-after.txt"
cp "$TMPD/SHA256SUMS.src-after.txt" "$EXPORT_ROOT/SHA256SUMS.src-after.txt"
grep -v '  README\.md$' "$STAGE/SHA256SUMS.pre.txt" > "$TMPD/pre.src-only.txt"
PRE_ONLY_LINES=$(wc -l < "$TMPD/pre.src-only.txt")
SRC_LINES=$(wc -l < "$TMPD/SHA256SUMS.src-after.txt")
echo "  pre(剔 README)=$PRE_ONLY_LINES 行, src-after=$SRC_LINES 行"
[ "$((PRE_LINES - PRE_ONLY_LINES))" = "1" ] || { echo "错误: 剔除行数不是 1"; exit 1; }
if diff "$TMPD/pre.src-only.txt" "$TMPD/SHA256SUMS.src-after.txt"; then
  echo "SRC_UNCHANGED=OK"
else
  echo "错误: 源文件 sha256 发生变化（上方 diff 即差异行）"; exit 1
fi
echo "阶段6完成"
echo "全部完成"
