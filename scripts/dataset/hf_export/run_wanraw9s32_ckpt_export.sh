#!/usr/bin/env bash
# driver：把训练 run v2-1600ep-m8x8-modul-wanraw9s32-b128-50k 的 10 份 orbax checkpoint（step 5000…50000）
# 逐份上传到 HF **公开** bucket HongzeFu/robomme-4task-modul-wanraw9-s32-50k-v1，并在本机读回逐文件 sha256 对照
# （1112-4task-modul-wanraw9-s32-50k-plan.html 第二部分 1.2 导出行、runbook RB-E3 的 tmux r9-ckptup；本机无集群，读回校验在本机做）。
# 由 run_wantp768_ckpt_export.sh（v11）复制，只换 RUN_NAME、BUCKET_ID 与 TRAIN_LOG 默认值（链脚本所在 tmux r9-train 的日志）。
#
# 蓝本是同目录 run_ckpt_export.sh（骨架照抄：硬链接 stage、不打 tar、凭据走 v1-store/secrets/hf.env、
# uvx 拉独立 1.30.0 CLI、上传前 / 读回后两遍独立 sha256 逐行 diff、源侧第三遍 sha256 证明硬链接未改原件）。
# 与蓝本的差异：
#   1. **不等 50k、每份存完即传**：按 step 5000…50000 依次等该份 ckpt 落盘完成——orbax 先写
#      <step>.orbax-checkpoint-tmp-* 再改名为 <step>，改名后 <step>/_CHECKPOINT_METADATA 存在、且同号 tmp 目录已消失，
#      才算落盘完成——随即 stage（硬链接）+ 上传该 step 目录 + 本机读回该 step + sha256 对照，一份一份推进；
#   2. **建公开 bucket（不带 --private）+ 空库断言**沿用 run_modul60k_ckpt_export.sh 的做法（uploaded.marker 区分续传）；
#   3. 按 step 目录分批上传（每份约 12 GB，落在 run_modul60k 实测会过的批量级）；
#   4. NPROC=48（沿用 v11 P7：哈希并行 32–48）；
#   5. run 根下的平文件与非 step 目录（history_config.*、motion_provenance.json 等）在 10 份都校验完后最后上传并读回校验。
# 每份完成写 logs/verified/<step>.done，可整脚本重跑续传。末行判定 HF_VERIFY=PASS ckpts=10。
# 可覆盖：SRC（run 根，默认 v1-store/train-runs/mme_vla_suite_b128_80k/<RUN_NAME>）、
#        TRAIN_LOG（训练日志，默认 v1-store/logs/r9-train.log：其中已出现 EXIT_CODE= 而某份 ckpt 仍未落盘即报错退出）
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
cd "$REPO"

RUN_NAME="v2-1600ep-m8x8-modul-wanraw9s32-b128-50k"
BUCKET_ID="HongzeFu/robomme-4task-modul-wanraw9-s32-50k-v1"
BUCKET="hf://buckets/$BUCKET_ID"
SRC="${SRC:-$REPO/v1-store/train-runs/mme_vla_suite_b128_80k/$RUN_NAME}"
TRAIN_LOG="${TRAIN_LOG:-$REPO/v1-store/logs/r9-train.log}"
EXPORT_ROOT="$REPO/v1-store/exports/hf-ckpt-$RUN_NAME"
STAGE="$EXPORT_ROOT/stage"
VERIFY="$EXPORT_ROOT/verify"
TMPD="$EXPORT_ROOT/tmp"
NPROC=48
STEPS=(5000 10000 15000 20000 25000 30000 35000 40000 45000 50000)
POLL_S=300

# AGENTS 第 14 条：缓存类环境变量逐项指向 v1-store / scratch，**禁止覆盖 HOME**
export HF_HOME="$REPO/v1-store/cache/hf"
export HF_XET_CACHE="$REPO/v1-store/cache/hf-xet"
export HF_XET_HIGH_PERFORMANCE=1
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export UV_PYTHON_INSTALL_DIR=/scratch/hongze/.cache/uv-python
export UV_LINK_MODE=copy
unset HF_HUB_OFFLINE || true

# 凭据：只从 v1-store/secrets/hf.env 取（600、在 .gitignore 的 /v1-store/ 下）。
# 环境里原有的 HF_TOKEN 属于 yinpei-tri，写不进 HongzeFu 命名空间，必须被这里覆盖掉。
[ -f "$REPO/v1-store/secrets/hf.env" ] || { echo "错误: 缺少 v1-store/secrets/hf.env"; exit 1; }
set -a; . "$REPO/v1-store/secrets/hf.env"; set +a
[ -n "${HF_TOKEN:-}" ] || { echo "错误: hf.env 未提供 HF_TOKEN"; exit 1; }

hf() { uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf "$@"; }

bucket_stat() {  # $1 = 字段名 size|totalFiles
  curl -sf -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/api/buckets/HongzeFu" \
    | uv run --no-project python -c "
import sys, json
for b in json.load(sys.stdin):
    if b['id'] == '$BUCKET_ID':
        print(b.get('$1')); break
else:
    sys.exit('目标 bucket $BUCKET_ID 不在返回列表里')
"
}

# 在 $1 目录内并行算 sha256，输出按路径排序（xargs -P 并行会乱序，不排序则无法 diff）
hash_tree() {  # $1 = 目录, $2 = 输出文件（绝对路径，不能落在被扫描目录内）
  ( cd "$1" && find . -type f -not -name 'SHA256SUMS.*' -printf '%P\n' | sort \
      | xargs -P "$NPROC" -n 16 sha256sum | sort -k 2 > "$2" )
}

retry_sync() {  # $1 = 源, $2 = 目标；commit 超时是间歇性的（run_modul60k 实测），重试 8 次
  local attempt
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf sync "$1" "$2"; then return 0; fi
    echo "  sync [$1 → $2] 第 $attempt 次中断，60s 后重试续传"; sleep 60
  done
  echo "错误: sync [$1 → $2] 八次仍失败"; exit 1
}

# 一份（或一个子目录）的完整流程：stage 硬链接 → pre sha256 → 上传 → 读回 → post sha256 diff → 源侧 sha256 diff
export_sub() {  # $1 = SRC 下的相对子目录
  local sub="$1"
  rm -rf "${STAGE:?}/$sub"; mkdir -p "$STAGE/$sub"
  ( cd "$SRC/$sub" && find . -mindepth 1 -type d -printf '%P\n' ) | while read -r d; do mkdir -p "$STAGE/$sub/$d"; done
  ( cd "$SRC/$sub" && find . -mindepth 1 -type f -printf '%P\n' ) | while read -r f; do ln -f "$SRC/$sub/$f" "$STAGE/$sub/$f"; done
  if find "$STAGE/$sub" -type l -print | grep -q .; then
    echo "错误: stage 内出现符号链接"; find "$STAGE/$sub" -type l -print; exit 1
  fi
  hash_tree "$STAGE/$sub" "$TMPD/$sub.pre.txt"
  local files bytes
  files=$(find "$STAGE/$sub" -type f | wc -l)
  bytes=$(find "$STAGE/$sub" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
  echo "  [$sub] stage files=$files bytes=$bytes pre_lines=$(wc -l < "$TMPD/$sub.pre.txt")"
  retry_sync "$STAGE/$sub" "$BUCKET/$sub"
  rm -rf "${VERIFY:?}/$sub"; mkdir -p "$VERIFY/$sub"
  retry_sync "$BUCKET/$sub" "$VERIFY/$sub"
  hash_tree "$VERIFY/$sub" "$TMPD/$sub.post.txt"
  diff "$TMPD/$sub.pre.txt" "$TMPD/$sub.post.txt" || { echo "错误: [$sub] 上传前后 sha256 不一致（上方 diff）"; exit 1; }
  hash_tree "$SRC/$sub" "$TMPD/$sub.src-after.txt"
  diff "$TMPD/$sub.pre.txt" "$TMPD/$sub.src-after.txt" || { echo "错误: [$sub] 源文件 sha256 发生变化（上方 diff）"; exit 1; }
  cp "$TMPD/$sub.pre.txt" "$EXPORT_ROOT/logs/sums/$sub.SHA256SUMS.txt"
  echo "STEP_VERIFY=PASS sub=$sub files=$files bytes=$bytes"
}

echo "阶段0开始：预检 + 建 bucket"
for d in "$REPO/v1-store/exports" "$EXPORT_ROOT"; do
  [ -L "$d" ] && { echo "错误: $d 是符号链接，拒绝写入"; exit 1; }
done
[ -d "$SRC" ] || { echo "错误: 源目录不存在 $SRC"; exit 1; }
[ "$STAGE" != "$SRC" ] || { echo "错误: STAGE 与 SRC 相同，拒绝继续"; exit 1; }
mkdir -p "$STAGE" "$VERIFY" "$TMPD" "$EXPORT_ROOT/logs/verified" "$EXPORT_ROOT/logs/sums" "$HF_XET_CACHE"
who="$(hf auth whoami 2>&1 | tr '\n' ' ' | tr -s ' ')"
echo "  whoami: $who"
case "$who" in *HongzeFu*) : ;; *) echo "错误: 身份不是 HongzeFu（$who）"; exit 1 ;; esac
echo "HF_WHOAMI=HongzeFu"
# 建为**公开** bucket（2026-11-13 用户：「hugging face 上传的是公开 bucket」；私有存储额度不足曾报 403）。
# hf 1.30.0 的 `buckets create` 只有 --private 开关、没有 --public 选项，不带 --private 即公开；
# 已存在则 create 非零退出，由下面的空库断言兜底（已存在的私有 bucket 需另用 `hf buckets settings <id> --public` 改公开）
if hf buckets create "$BUCKET_ID" 2>&1 | tee "$TMPD/create.log"; then
  echo "  bucket create 成功"
else
  echo "  bucket create 非零退出（可能已存在），继续走空库断言"
fi
pre_files="$(bucket_stat totalFiles)"
echo "  目标 bucket 现状: files=$pre_files"
if [ "$pre_files" != "0" ] && [ ! -f "$EXPORT_ROOT/logs/uploaded.marker" ]; then
  echo "错误: 目标 bucket 非空（files=$pre_files）且本地无本轮上传标记，停止并交用户裁决"; exit 1
fi
touch "$EXPORT_ROOT/logs/uploaded.marker"
echo "阶段0完成"

echo "阶段1开始：逐份等落盘 → 上传 → 本机读回校验"
for step in "${STEPS[@]}"; do
  if [ -f "$EXPORT_ROOT/logs/verified/$step.done" ]; then echo "  step $step 已校验，跳过"; continue; fi
  # 落盘完成：<step>/_CHECKPOINT_METADATA 在、且同号 orbax tmp 目录已不在
  until [ -f "$SRC/$step/_CHECKPOINT_METADATA" ] && ! compgen -G "$SRC/$step.orbax-checkpoint-tmp*" > /dev/null; do
    if [ -f "$TRAIN_LOG" ] && grep -q '^EXIT_CODE=' "$TRAIN_LOG"; then
      echo "错误: 训练日志已有 EXIT_CODE= 而 step $step 未落盘完成（$TRAIN_LOG）"; exit 1
    fi
    echo "  等 step $step 落盘（$(date -u +%FT%TZ)）"; sleep "$POLL_S"
  done
  echo "CKPT_READY step=$step $(date -u +%FT%TZ)"
  export_sub "$step"
  touch "$EXPORT_ROOT/logs/verified/$step.done"
done
echo "阶段1完成"

echo "阶段2开始：run 根下的平文件与非 step 目录"
# 训练已结束（10 份都在）后 run 根不再变化；非 step 目录逐个走 export_sub，平文件收进 stage/_root_files 后逐个 cp
mapfile -t ROOT_DIRS < <( cd "$SRC" && find . -mindepth 1 -maxdepth 1 -type d -printf '%P\n' | sort \
                          | grep -vxE '[0-9]+|[0-9]+\.orbax-checkpoint-tmp.*' || true )
for d in "${ROOT_DIRS[@]}"; do export_sub "$d"; done
mapfile -t ROOT_FILES < <( cd "$SRC" && find . -mindepth 1 -maxdepth 1 -type f -printf '%P\n' | sort )
: > "$TMPD/root.pre.txt"; : > "$TMPD/root.post.txt"
for f in "${ROOT_FILES[@]}"; do
  ok=0
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf buckets cp "$SRC/$f" "$BUCKET/$f"; then ok=1; break; fi
    echo "  $f 第 $attempt 次中断，60s 后重试"; sleep 60
  done
  [ "$ok" = 1 ] || { echo "错误: $f 八次仍失败"; exit 1; }
  rm -f "${VERIFY:?}/$f"
done
# 读回：整库 sync 到 VERIFY（增量，已读回校验过的 step 目录不重下），再对平文件逐个算 sha256 对照
retry_sync "$BUCKET" "$VERIFY"
for f in "${ROOT_FILES[@]}"; do
  ( cd "$SRC" && sha256sum "$f" ) >> "$TMPD/root.pre.txt"
  ( cd "$VERIFY" && sha256sum "$f" ) >> "$TMPD/root.post.txt"
done
diff "$TMPD/root.pre.txt" "$TMPD/root.post.txt" || { echo "错误: run 根平文件读回 sha256 不一致"; exit 1; }
cp "$TMPD/root.pre.txt" "$EXPORT_ROOT/logs/sums/root-files.SHA256SUMS.txt"
echo "ROOT_VERIFY=PASS dirs=${#ROOT_DIRS[@]} files=${#ROOT_FILES[@]}"
echo "阶段2完成"

NDONE=$(find "$EXPORT_ROOT/logs/verified" -name '*.done' | wc -l)
[ "$NDONE" = "${#STEPS[@]}" ] || { echo "错误: 已校验份数 $NDONE != ${#STEPS[@]}"; echo "HF_VERIFY=FAIL ckpts=$NDONE"; exit 1; }
echo "全部完成"
echo "HF_VERIFY=PASS ckpts=$NDONE"
