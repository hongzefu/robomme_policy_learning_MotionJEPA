#!/usr/bin/env bash
# 增量上传：1127 4task modulation 帧记忆每帧 4×4 两个正式 run 的 10 份 orbax checkpoint（step 5000…50000）
# 每出一份就上传到 HF **公开** bucket、本机读回逐文件 sha256 对照
# （1127-4task-modul-fs4x4-512-1024-60k-plan.md 第一部分七节、第二部分一节 S4 行；tmux fs4-incA / fs4-incB 由主会话起）。
#   bash scripts/dataset/hf_export/run_fs4x4_ckpt_export.sh <RUN_NAME> [--dry-run]
#   RUN_NAME 只接受：
#     v2-1600ep-m32x4x4-modul-b128-50k（预算 512）→ HongzeFu/robomme-vla-4task-modul-m32x4x4-50k-v1
#     v2-1600ep-m64x4x4-modul-b128-50k（预算 1024）→ HongzeFu/robomme-vla-4task-modul-m64x4x4-50k-v1
# 以 run_e2eenc_ckpt_export.sh（v18）为底，骨架照抄：硬链接 stage、不打 tar、上传前 / 读回后两遍独立 sha256 逐行 diff、
#   源侧第三遍 sha256 证明硬链接未改原件；落盘判据 <step>/_CHECKPOINT_METADATA 在且同号 orbax tmp 目录不在；
#   每份完成写 logs/verified/<step>.done 可整脚本重跑续传；step 读回校验通过即删 $VERIFY/<step> 检验副本
#   （只删 $VERIFY 下的读回副本，源 ckpt 与 stage 硬链接不动），末行判定后删 $VERIFY 剩余检验副本。
# 与底本的差异：
#   1. RUN_NAME 作为参数，bucket 按 RUN_NAME 映射；TRAIN_LOG 默认 v1-store/logs/<RUN_NAME>.log（runner 的日志，末行 EXIT_CODE=）；
#   2. CLI 改用 /scratch/hongze/MotionJEPA/.venv/bin/hf（1.9.2：hf sync、hf buckets cp/create/info/list），
#      凭据走 HF_HOME=/scratch/hongze/.cache/huggingface-hongzefu 的 token（HongzeFu），并 unset 环境里的 HF_TOKEN
#      （原有 HF_TOKEN 属于其他账号，会覆盖 token 文件）；不再读 v1-store/secrets/hf.env、不再 uvx；
#   3. 建桶 `hf buckets create <id> --exist-ok`（不带 --private = 公开），再用 `hf buckets info` 断言 private=false
#      （已存在的私有桶不改、直接报错退出）；空库断言与 uploaded.marker 续传判据沿用底本；bucket_stat 改读 info 的 JSON；
#   4. 末行判定改为 HF_VERIFY=PASS objects=<n> bytes=<n> mismatches=0（n 为本轮逐文件读回校验过的对象数与字节数，
#      并与 bucket info 的 total_files 对照），随后打印 INC_END；任何 sha 不一致即退出非零（mismatches 只可能为 0 才打印 PASS）；
#   5. --dry-run：只做本地预检并打印将执行的步骤与命令，不调 hf、不写任何文件；
#   6. 正式运行要求脚本位于主检出（不用 worktree 里的脚本跑主数据）。
# 测试钩子（仅单测用，正式运行不设）：FS4X4_EXPORT_FAKE_HF=<假 hf 可执行文件> 时 hf 与 bucket_stat 改调它；
#   另可覆盖 SRC、EXPORT_ROOT、TRAIN_LOG、FS4X4_EXPORT_STEPS（空格分隔）、POLL_S。
# 不删任何本地 ckpt、不删桶。
set -euo pipefail

usage() { echo "用法: $0 <v2-1600ep-m32x4x4-modul-b128-50k|v2-1600ep-m64x4x4-modul-b128-50k> [--dry-run]" >&2; exit 2; }
RUN_NAME="${1:-}"; [ -n "$RUN_NAME" ] || usage
DRY=0
if [ $# -eq 2 ]; then [ "$2" = "--dry-run" ] || usage; DRY=1; elif [ $# -ne 1 ]; then usage; fi
case "$RUN_NAME" in
  v2-1600ep-m32x4x4-modul-b128-50k) BUCKET_ID="HongzeFu/robomme-vla-4task-modul-m32x4x4-50k-v1" ;;
  v2-1600ep-m64x4x4-modul-b128-50k) BUCKET_ID="HongzeFu/robomme-vla-4task-modul-m64x4x4-50k-v1" ;;
  *) echo "错误: RUN_NAME 只接受两个正式 run，收到 $RUN_NAME" >&2; exit 2 ;;
esac

MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
FAKE_HF="${FS4X4_EXPORT_FAKE_HF:-}"
if [ "$DRY" = 0 ] && [ -z "$FAKE_HF" ] && [ "$REPO" != "$MAIN" ]; then
  echo "错误: 正式上传须从主检出 $MAIN 运行（当前脚本在 $REPO）"; exit 1
fi

BUCKET="hf://buckets/$BUCKET_ID"
SRC="${SRC:-$MAIN/v1-store/train-runs/mme_vla_suite_b128_80k/$RUN_NAME}"
TRAIN_LOG="${TRAIN_LOG:-$MAIN/v1-store/logs/$RUN_NAME.log}"
EXPORT_ROOT="${EXPORT_ROOT:-$MAIN/v1-store/exports/hf-ckpt-$RUN_NAME}"
STAGE="$EXPORT_ROOT/stage"
VERIFY="$EXPORT_ROOT/verify"
TMPD="$EXPORT_ROOT/tmp"
DELETED_TSV="$EXPORT_ROOT/logs/verify_deleted.tsv"
COUNTS="$EXPORT_ROOT/logs/counts"
NPROC=48
if [ -n "${FS4X4_EXPORT_STEPS:-}" ]; then read -r -a STEPS <<< "$FS4X4_EXPORT_STEPS"
else STEPS=(5000 10000 15000 20000 25000 30000 35000 40000 45000 50000); fi
POLL_S="${POLL_S:-300}"
HF_BIN=/scratch/hongze/MotionJEPA/.venv/bin/hf

# 缓存与凭据：HF_HOME 指向 HongzeFu 令牌所在目录（/scratch/hongze/.cache 下），禁止覆盖 HOME
if [ -n "$FAKE_HF" ]; then export HF_HOME="$EXPORT_ROOT/cache/hf"
else export HF_HOME=/scratch/hongze/.cache/huggingface-hongzefu; fi
export HF_XET_HIGH_PERFORMANCE=1
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
unset HF_TOKEN HF_HUB_OFFLINE || true

if [ -n "$FAKE_HF" ]; then hf() { "$FAKE_HF" "$@"; }; else hf() { "$HF_BIN" "$@"; }; fi

bucket_info() {  # 打印 bucket info 的 JSON
  if [ -n "$FAKE_HF" ]; then "$FAKE_HF" buckets info "$BUCKET_ID"; else "$HF_BIN" buckets info "$BUCKET_ID"; fi
}
bucket_stat() {  # $1 = JSON 字段名 total_files|size|private
  bucket_info | uv run --no-project python -c "
import sys, json
d = json.load(sys.stdin)
v = d.get('$1')
print(str(v).lower() if isinstance(v, bool) else v)
"
}

hash_tree() {  # $1 = 目录, $2 = 输出文件（不能落在被扫描目录内）；按路径排序
  ( cd "$1" && find . -type f -not -name 'SHA256SUMS.*' -printf '%P\n' | sort \
      | xargs -r -P "$NPROC" -n 16 sha256sum | sort -k 2 > "$2" )
}

retry_sync() {  # $1 源, $2 目标；commit 超时是间歇性的，重试 8 次
  local attempt
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf sync "$1" "$2"; then return 0; fi
    echo "  sync [$1 → $2] 第 $attempt 次中断，60s 后重试续传"; sleep 60
  done
  echo "错误: sync [$1 → $2] 八次仍失败"; exit 1
}

retry_cp() {  # $1 源, $2 目标（单文件）
  local attempt
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf buckets cp "$1" "$2"; then return 0; fi
    echo "  cp [$1 → $2] 第 $attempt 次中断，60s 后重试"; sleep 60
  done
  echo "错误: cp [$1 → $2] 八次仍失败"; exit 1
}

delete_verified() {  # $1 = $VERIFY 下的绝对路径（只删读回检验副本）
  local path="$1" bytes
  case "$path" in */../*|*/..) echo "错误: 拒绝删除含 .. 的路径 $path"; exit 1 ;; esac
  case "$path" in "$VERIFY"/?*) : ;; *) echo "错误: 拒绝删除 \$VERIFY 以外的路径 $path"; exit 1 ;; esac
  [ -e "$path" ] || return 0
  bytes=$(find "$path" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
  rm -rf -- "$path"
  echo "VERIFY_DELETED path=$path bytes=$bytes"
  printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$path" "$bytes" >> "$DELETED_TSV"
}

export_sub() {  # $1 = SRC 下的相对子目录：stage 硬链接 → pre sha → 上传 → 读回 → post sha diff → 源侧 sha diff
  local sub="$1" files bytes
  rm -rf "${STAGE:?}/$sub"; mkdir -p "$STAGE/$sub"
  ( cd "$SRC/$sub" && find . -mindepth 1 -type d -printf '%P\n' ) | while read -r d; do mkdir -p "$STAGE/$sub/$d"; done
  ( cd "$SRC/$sub" && find . -mindepth 1 -type f -printf '%P\n' ) | while read -r f; do ln -f "$SRC/$sub/$f" "$STAGE/$sub/$f"; done
  if find "$STAGE/$sub" -type l -print | grep -q .; then
    echo "错误: stage 内出现符号链接"; find "$STAGE/$sub" -type l -print; exit 1
  fi
  hash_tree "$STAGE/$sub" "$TMPD/$sub.pre.txt"
  files=$(find "$STAGE/$sub" -type f | wc -l)
  bytes=$(find "$STAGE/$sub" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
  echo "  [$sub] stage files=$files bytes=$bytes pre_lines=$(wc -l < "$TMPD/$sub.pre.txt")"
  retry_sync "$STAGE/$sub" "$BUCKET/$sub"
  rm -rf "${VERIFY:?}/$sub"; mkdir -p "$VERIFY/$sub"
  retry_sync "$BUCKET/$sub" "$VERIFY/$sub"
  hash_tree "$VERIFY/$sub" "$TMPD/$sub.post.txt"
  diff "$TMPD/$sub.pre.txt" "$TMPD/$sub.post.txt" || { echo "错误: [$sub] 上传前后 sha256 不一致（上方 diff）"; echo "HF_VERIFY=FAIL sub=$sub reason=readback_mismatch"; exit 1; }
  hash_tree "$SRC/$sub" "$TMPD/$sub.src-after.txt"
  diff "$TMPD/$sub.pre.txt" "$TMPD/$sub.src-after.txt" || { echo "错误: [$sub] 源文件 sha256 发生变化（上方 diff）"; echo "HF_VERIFY=FAIL sub=$sub reason=source_changed"; exit 1; }
  cp "$TMPD/$sub.pre.txt" "$EXPORT_ROOT/logs/sums/$sub.SHA256SUMS.txt"
  printf '%s\t%s\n' "$files" "$bytes" > "$COUNTS/$sub.tsv"
  echo "STEP_VERIFY=PASS sub=$sub files=$files bytes=$bytes"
  if [[ "$sub" =~ ^[0-9]+$ ]]; then delete_verified "$VERIFY/$sub"; fi
}

if [ "$DRY" = 1 ]; then
  echo "FS4X4_EXPORT_DRY run=$RUN_NAME bucket=$BUCKET_ID（公开，不带 --private）"
  echo "  SRC=$SRC exists=$([ -d "$SRC" ] && echo yes || echo no)"
  echo "  TRAIN_LOG=$TRAIN_LOG exists=$([ -f "$TRAIN_LOG" ] && echo yes || echo no)"
  echo "  EXPORT_ROOT=$EXPORT_ROOT  HF_HOME=$HF_HOME  CLI=$HF_BIN  repo=$REPO main_ok=$([ "$REPO" = "$MAIN" ] && echo yes || echo no)"
  echo "  阶段0: hf auth whoami（须 HongzeFu）；hf buckets create $BUCKET_ID --exist-ok；hf buckets info → private=false、空库或有 uploaded.marker"
  for step in "${STEPS[@]}"; do
    st=pending
    [ -f "$EXPORT_ROOT/logs/verified/$step.done" ] && st=done
    { [ -f "$SRC/$step/_CHECKPOINT_METADATA" ] && ! compgen -G "$SRC/$step.orbax-checkpoint-tmp*" > /dev/null; } && st="$st,ready" || st="$st,not_ready"
    echo "  阶段1 step=$step [$st]: 等落盘（每 ${POLL_S}s；训练日志出现 EXIT_CODE= 仍未落盘即报错）→ 硬链接 stage → sha256 → hf sync $STAGE/$step $BUCKET/$step → hf sync $BUCKET/$step $VERIFY/$step → sha256 diff → 源侧 sha256 diff → 删 $VERIFY/$step"
  done
  echo "  阶段2: run 根非 step 目录逐个同上；平文件 hf buckets cp 上传 + 读回 + sha256 diff"
  echo "  末行: HF_VERIFY=PASS objects=<n> bytes=<n> mismatches=0（对照 bucket info total_files）→ INC_END"
  echo "FS4X4_EXPORT_DRY=DONE"
  exit 0
fi

echo "阶段0开始：预检 + 建 bucket（$RUN_NAME → $BUCKET_ID）"
for d in "$MAIN/v1-store/exports" "$EXPORT_ROOT"; do
  [ -L "$d" ] && { echo "错误: $d 是符号链接，拒绝写入"; exit 1; }
done
# SRC 可能在训练刚起时尚未创建：等它出现（与等 step 同一判据：训练已退出仍无则报错）
until [ -d "$SRC" ]; do
  if [ -f "$TRAIN_LOG" ] && grep -q '^EXIT_CODE=' "$TRAIN_LOG"; then echo "错误: 训练日志已有 EXIT_CODE= 而源目录不存在 $SRC"; exit 1; fi
  echo "  等源目录 $SRC 出现（$(date -u +%FT%TZ)）"; sleep "$POLL_S"
done
[ "$STAGE" != "$SRC" ] || { echo "错误: STAGE 与 SRC 相同，拒绝继续"; exit 1; }
mkdir -p "$STAGE" "$VERIFY" "$TMPD" "$EXPORT_ROOT/logs/verified" "$EXPORT_ROOT/logs/sums" "$COUNTS"
who="$(hf auth whoami 2>&1 | tr '\n' ' ' | tr -s ' ')"
echo "  whoami: $who"
case "$who" in *HongzeFu*) : ;; *) echo "错误: 身份不是 HongzeFu（$who）"; exit 1 ;; esac
echo "HF_WHOAMI=HongzeFu"
hf buckets create "$BUCKET_ID" --exist-ok 2>&1 | tee "$TMPD/create.log"
priv="$(bucket_stat private)"
[ "$priv" = "false" ] || { echo "错误: bucket $BUCKET_ID private=$priv（须公开；不自动改，交主会话）"; exit 1; }
pre_files="$(bucket_stat total_files)"
echo "  目标 bucket 现状: private=$priv files=$pre_files"
if [ "$pre_files" != "0" ] && [ ! -f "$EXPORT_ROOT/logs/uploaded.marker" ]; then
  echo "错误: 目标 bucket 非空（files=$pre_files）且本地无本轮上传标记，停止并交用户裁决"; exit 1
fi
touch "$EXPORT_ROOT/logs/uploaded.marker"
echo "阶段0完成"

echo "阶段1开始：逐份等落盘 → 上传 → 本机读回校验"
for step in "${STEPS[@]}"; do
  if [ -f "$EXPORT_ROOT/logs/verified/$step.done" ]; then echo "  step $step 已校验，跳过"; continue; fi
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
mapfile -t ROOT_DIRS < <( cd "$SRC" && find . -mindepth 1 -maxdepth 1 -type d -printf '%P\n' | sort \
                          | grep -vxE '[0-9]+|[0-9]+\.orbax-checkpoint-tmp.*' || true )
for d in "${ROOT_DIRS[@]}"; do export_sub "$d"; done
mapfile -t ROOT_FILES < <( cd "$SRC" && find . -mindepth 1 -maxdepth 1 -type f -printf '%P\n' | sort )
: > "$TMPD/root.pre.txt"; : > "$TMPD/root.post.txt"
for f in "${ROOT_FILES[@]}"; do retry_cp "$SRC/$f" "$BUCKET/$f"; rm -f "${VERIFY:?}/$f"; done
for f in "${ROOT_FILES[@]}"; do retry_cp "$BUCKET/$f" "$VERIFY/$f"; done
for f in "${ROOT_FILES[@]}"; do
  ( cd "$SRC" && sha256sum "$f" ) >> "$TMPD/root.pre.txt"
  ( cd "$VERIFY" && sha256sum "$f" ) >> "$TMPD/root.post.txt"
done
diff "$TMPD/root.pre.txt" "$TMPD/root.post.txt" || { echo "错误: run 根平文件读回 sha256 不一致"; echo "HF_VERIFY=FAIL sub=_root reason=readback_mismatch"; exit 1; }
cp "$TMPD/root.pre.txt" "$EXPORT_ROOT/logs/sums/root-files.SHA256SUMS.txt"
root_bytes=0
for f in "${ROOT_FILES[@]}"; do root_bytes=$((root_bytes + $(stat -c %s "$SRC/$f"))); done
printf '%s\t%s\n' "${#ROOT_FILES[@]}" "$root_bytes" > "$COUNTS/_root_files.tsv"
echo "ROOT_VERIFY=PASS dirs=${#ROOT_DIRS[@]} files=${#ROOT_FILES[@]}"
echo "阶段2完成"

NDONE=$(find "$EXPORT_ROOT/logs/verified" -name '*.done' | wc -l)
[ "$NDONE" = "${#STEPS[@]}" ] || { echo "错误: 已校验份数 $NDONE != ${#STEPS[@]}"; echo "HF_VERIFY=FAIL ckpts=$NDONE"; exit 1; }
# 对象数 / 字节数：本轮各 step、非 step 目录与平文件的逐文件校验计数之和（logs/counts/*.tsv，续传时含先前已校验的份）
read -r objects bytes < <( cat "$COUNTS"/*.tsv | awk -F'\t' '{o+=$1; b+=$2} END{print o+0, b+0}' )
remote_files="$(bucket_stat total_files)"
remote_size="$(bucket_stat size)"
echo "  bucket info: total_files=$remote_files size=$remote_size；本地校验 objects=$objects bytes=$bytes"
[ "$remote_files" = "$objects" ] || { echo "错误: bucket total_files=$remote_files != 校验对象数 $objects"; echo "HF_VERIFY=FAIL objects=$objects remote_files=$remote_files"; exit 1; }
echo "全部完成 ckpts=$NDONE"
echo "HF_VERIFY=PASS objects=$objects bytes=$bytes mismatches=0"
if [ -d "$VERIFY" ]; then
  mapfile -t LEFT < <( cd "$VERIFY" && find . -mindepth 1 -maxdepth 1 -printf '%P\n' | sort )
  for x in "${LEFT[@]}"; do delete_verified "$VERIFY/$x"; done
fi
echo "INC_END run=$RUN_NAME bucket=$BUCKET_ID $(date -u +%FT%TZ)"
