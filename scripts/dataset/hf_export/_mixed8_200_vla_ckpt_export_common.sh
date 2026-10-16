#!/usr/bin/env bash
# 两条 mixed8-200 VLA 80k run 的 ckpt 上传共用体。由 run_mixed8_200_vla_512_80k_ckpt_export.sh /
# run_mixed8_200_vla_motion_80k_ckpt_export.sh 设好下列变量后 source，不单独运行：
#   RUN_NAME BUCKET_ID EXPORT_NAME README_SRC ALLOWLIST EXPECT_MOTION(true|false) DRIVER_NAME
#
# 计划：1016-mixed8-vla-plan.html 3.1 表 A1b 行、口径 29（两条 VLA：全部 16 份 ckpt、config、日志）/
# 30（bucket 名）、G12、runbook 长尾「每条训练 EXIT_CODE=0 后起对应的 ckpt 上传，先 SMOKE=1 再全量」。
#
# 蓝本 run_modul60k_ckpt_export.sh（六阶段骨架）与 run_motion80k_ckpt_export.sh（16 份 step、_run-meta/、
# encoder 交叉锚点）。与蓝本的差别：
#   1. **公开**：建 bucket 不带私有开关（两份蓝本都写死了私有），建后与收尾两次断言 private==false
#      （口径 21、审查报告 H7）。
#   2. **一趟跑完**：只在训练已退出（进程不在、日志末行 EXIT_CODE=0）后运行，不做 motion80k 那种「训练中先传
#      已写完的 step」的趟 A；16 个 step 目录必须恰为 5000…75000（步长 5000）加 79999。
#   3. **公开前体检**：在建 bucket 之前对 stage 跑 scan_hygiene.py。
#   4. **_run-meta/train.log 现场清洗**（去 tqdm 中间态，AGENTS 第 12 条 (3) 同一条命令）；wandb 本地目录不传
#      （元数据含主机名、GPU UUID 与本机路径，计划口径 29 的 VLA 清单是「ckpt、config、日志」）。
#   5. **回读前 df 余量断言**：照抄 run_motionjepa_mixed8_200_export.sh 阶段 8 的 READBACK_NEED / DF_MARGIN 写法，
#      余量取 0.4 TB（G12）。
#   6. **provenance 交叉锚点**：motion_provenance.json 的 motion_enabled 必须等于本驱动的预期；framesamp
#      store_meta sha256 必须等于训练库现场重算值；motion 那条另核 encoder.checkpoint_sha256 等于 VLA 侧拷入件。
#   7. 完成信号是末尾的「全部完成」；RESULT=PASS 只是中途的内容层判定行。
#
# 路径覆盖（只为离线 dry-run / 单测；正式运行一律不设）：M8X_SRC、M8X_TRAIN_LOG、M8X_LIB、M8X_ENC_COPY、
# M8X_EXPORT_ROOT。设了哪个就打印一行 OVERRIDE；全量模式下 EXPORT_ROOT 仍必须落在 v1-store/exports/ 下。
set -euo pipefail

: "${RUN_NAME:?}" "${BUCKET_ID:?}" "${EXPORT_NAME:?}" "${README_SRC:?}" "${ALLOWLIST:?}" "${EXPECT_MOTION:?}" "${DRIVER_NAME:?}"
case "$EXPECT_MOTION" in true|false) : ;; *) echo "错误: EXPECT_MOTION 只能是 true|false"; exit 1 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
cd "$REPO"

NAME="16task-mixed8-200-3200ep-e3c7564f"
ENC_RUN="wan-mixed8-200-filter2-stop075-b88x8-72ep-a"
BUCKET="hf://buckets/$BUCKET_ID"
SRC="${M8X_SRC:-$REPO/v1-store/train-runs/mme_vla_suite_b128_80k/$RUN_NAME}"
TRAIN_LOG="${M8X_TRAIN_LOG:-$REPO/v1-store/logs/$RUN_NAME.log}"
LIB="${M8X_LIB:-$REPO/v1-store/datasets/$NAME}"
ENC_COPY="${M8X_ENC_COPY:-$REPO/v1-store/external/motionjepa/$ENC_RUN/checkpoint_epoch_72.pt}"
EXPORT_ROOT="${M8X_EXPORT_ROOT:-$REPO/v1-store/exports/$EXPORT_NAME}"
STAGE="$EXPORT_ROOT/stage"
VERIFY="$EXPORT_ROOT/verify"
TMPD="$EXPORT_ROOT/tmp"
EXPECT_STEPS="$(seq 5000 5000 75000 | tr '\n' ' ')79999"
REQUIRED_ROOT=(history_config.resolved.yaml history_config.resolved.sha256 motion_provenance.json)
STEP_MAX_BYTES=$((16 * 1024 * 1024 * 1024))   # 单批（一个 step 目录）上界：实测 20 GB 一批会挂
DF_MARGIN_BYTES=400000000000                    # 0.4 TB（十进制，G12）
NPROC=8
# 网络重试口径（计划口径 27「重试 3 次」）：首试 + 重试 UPLOAD_RETRIES 次，仍失败即非 0 退出并点名失败对象，
# 由主会话写进 pending_decisions（status=fail）。上传、断点续传回读一律用这一个常量。
UPLOAD_RETRIES=3
MAX_ATTEMPTS=$((UPLOAD_RETRIES + 1))
SMOKE="${SMOKE:-0}"

for v in M8X_SRC M8X_TRAIN_LOG M8X_LIB M8X_ENC_COPY M8X_EXPORT_ROOT; do
  [ -n "${!v:-}" ] && echo "OVERRIDE $v=${!v}"
done
if [ "$SMOKE" != 1 ]; then
  case "$EXPORT_ROOT" in
    "$REPO/v1-store/exports/"*) : ;;
    *) echo "错误: 全量模式下 EXPORT_ROOT 必须在 $REPO/v1-store/exports/ 下（实为 $EXPORT_ROOT）"; exit 1 ;;
  esac
fi

export HF_HOME="$REPO/v1-store/cache/hf"
export HF_XET_CACHE="$REPO/v1-store/cache/hf-xet"
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export UV_PYTHON_INSTALL_DIR=/scratch/hongze/.cache/uv-python
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE || true

hf() { uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf "$@"; }

bucket_stat() {  # $1 = size|totalFiles|private
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
assert_public() {  # $1 = 判定行前缀
  local p; p="$(bucket_stat private)"
  echo "$1 private=$p"
  case "$p" in
    False|false) : ;;
    *) echo "错误: bucket 不是公开的（private=$p），要求 private==false"; exit 1 ;;
  esac
}
hash_tree() {  # $1 = 目录, $2 = 输出文件（绝对路径，不能落在被扫描目录内）
  ( cd "$1" && find . -type f -not -name 'SHA256SUMS.*' -printf '%P\n' | sort \
      | xargs -P "$NPROC" -n 16 sha256sum | sort -k 2 > "$2" )
}
sync_dir() {  # $1 = STAGE 下的子目录（同时也是 bucket 内前缀）
  local sub="$1" ok=0 attempt
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    if hf sync "$STAGE/$sub" "$BUCKET/$sub"; then ok=1; break; fi
    echo "  批 [$sub] 第 $attempt/$MAX_ATTEMPTS 次尝试中断"
    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then echo "  60s 后重试续传"; sleep 60; fi
  done
  [ "$ok" = 1 ] || { echo "UPLOAD_FAIL object=$sub/ attempts=$MAX_ATTEMPTS"; echo "错误: 批 [$sub] 首试 + 重试 $UPLOAD_RETRIES 次仍失败"; exit 1; }
}
cp_file() {  # $1 = STAGE 下的相对文件路径（sync 的源必须是目录，根下平文件走 cp）
  local f="$1" ok=0 attempt
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    if hf buckets cp "$STAGE/$f" "$BUCKET/$f"; then ok=1; break; fi
    echo "  $f 第 $attempt/$MAX_ATTEMPTS 次尝试中断"
    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then echo "  60s 后重试"; sleep 60; fi
  done
  [ "$ok" = 1 ] || { echo "UPLOAD_FAIL object=$f attempts=$MAX_ATTEMPTS"; echo "错误: 根文件 $f 首试 + 重试 $UPLOAD_RETRIES 次仍失败"; exit 1; }
}

# ---------------------------------------------------------------- 阶段 0：预检
echo "阶段0开始：预检（$DRIVER_NAME；run=$RUN_NAME motion=$EXPECT_MOTION）"
for d in "$(dirname "$EXPORT_ROOT")" "$EXPORT_ROOT"; do
  [ -L "$d" ] && { echo "错误: $d 是符号链接，拒绝写入"; exit 1; }
done
[ -d "$SRC" ] || { echo "错误: 源 run 目录不存在 $SRC"; exit 1; }
[ -f "$ALLOWLIST" ] && [ -f "$README_SRC" ] || { echo "错误: 缺体检账本或 README 模板"; exit 1; }
mkdir -p "$STAGE" "$VERIFY" "$TMPD" "$EXPORT_ROOT/logs/uploaded"

# 训练必须已经退出：进程不在（括号技巧防 pgrep 自匹配），日志末行 EXIT_CODE=0（runner 尾行）
if pgrep -f "[s]cripts/training/train.py mme_vla_suite_b128_80k --exp-name $RUN_NAME " > /dev/null 2>&1; then
  echo "错误: 训练进程仍在运行（--exp-name $RUN_NAME），本驱动只在训练退出后运行"; exit 1
fi
[ -f "$TRAIN_LOG" ] || { echo "错误: 训练日志不存在 $TRAIN_LOG"; exit 1; }
LAST_LINE="$(tail -n 1 "$TRAIN_LOG")"
[ "$LAST_LINE" = "EXIT_CODE=0" ] || { echo "错误: $TRAIN_LOG 末行为「$LAST_LINE」，不是 EXIT_CODE=0"; exit 1; }
LOG_RUN="$(grep -m1 '^RUN=' "$TRAIN_LOG" | cut -d= -f2- || true)"
[ "$LOG_RUN" = "$RUN_NAME" ] || { echo "错误: 训练日志的 RUN= 为「$LOG_RUN」，不是 $RUN_NAME"; exit 1; }
GIT_HEAD="$(grep -m1 '^GIT_HEAD=' "$TRAIN_LOG" | cut -d= -f2- || true)"
echo "  训练日志: 末行 $LAST_LINE，RUN=$LOG_RUN，GIT_HEAD=${GIT_HEAD:-<缺>}"

# step 目录：恰为 5000…75000 + 79999，且每个都已由 orbax 提交（_CHECKPOINT_METADATA 带 commit_timestamp_nsecs）
GOT_DIRS="$(cd "$SRC" && find . -mindepth 1 -maxdepth 1 -type d -printf '%P\n' | sort -n | tr '\n' ' ' | sed 's/ $//')"
[ "$GOT_DIRS" = "$EXPECT_STEPS" ] || { echo "错误: run 目录下的子目录不符"; echo "  期望 $EXPECT_STEPS"; echo "  实为 $GOT_DIRS"; exit 1; }
for s in $EXPECT_STEPS; do
  python3 -c "
import json, sys
m = json.load(open('$SRC/$s/_CHECKPOINT_METADATA'))
sys.exit(0 if m.get('commit_timestamp_nsecs') else 1)
" || { echo "错误: step $s 未见 orbax 提交标记（_CHECKPOINT_METADATA 缺 commit_timestamp_nsecs）"; exit 1; }
  sz=$(find "$SRC/$s" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
  [ "$sz" -le "$STEP_MAX_BYTES" ] || { echo "错误: step $s 共 $sz B，超过单批上界 $STEP_MAX_BYTES"; exit 1; }
done
echo "  16 个 step 目录齐全且均已提交: $EXPECT_STEPS"
for f in "${REQUIRED_ROOT[@]}"; do
  [ -f "$SRC/$f" ] || { echo "错误: run 根缺 $f"; exit 1; }
done
# 快照自洽：history_config.resolved.yaml 的 sha256 必须等于同目录 .sha256 与 motion_provenance.json 的记录
RESOLVED_SHA="$(sha256sum "$SRC/history_config.resolved.yaml" | cut -d' ' -f1)"
[ "$RESOLVED_SHA" = "$(cut -d' ' -f1 < "$SRC/history_config.resolved.sha256" | tr -d '[:space:]')" ] \
  || { echo "错误: history_config.resolved.yaml 与 .sha256 不符"; exit 1; }
[ -f "$LIB/framesamp-8x8/meta/store_meta.json" ] || { echo "错误: 缺训练库 $LIB/framesamp-8x8/meta/store_meta.json"; exit 1; }
FS8_SHA="$(sha256sum "$LIB/framesamp-8x8/meta/store_meta.json" | cut -d' ' -f1)"
MO_SHA=""; ENC_SHA=""
if [ "$EXPECT_MOTION" = true ]; then
  [ -f "$ENC_COPY" ] || { echo "错误: 缺 VLA 侧 encoder 拷入件 $ENC_COPY"; exit 1; }
  MO_SHA="$(sha256sum "$LIB/motion/meta/store_meta.json" | cut -d' ' -f1)"
  ENC_SHA="$(sha256sum "$ENC_COPY" | cut -d' ' -f1)"
fi
python3 - "$SRC/motion_provenance.json" "$EXPECT_MOTION" "$RESOLVED_SHA" "$FS8_SHA" "$MO_SHA" "$ENC_SHA" <<'PY'
import json, sys
prov_path, expect_motion, resolved, fs8, mo, enc = sys.argv[1:7]
p = json.load(open(prov_path))
bad = []
if p.get("motion_enabled") is not (expect_motion == "true"):
    bad.append(f"motion_enabled={p.get('motion_enabled')}，期望 {expect_motion}")
if p.get("resolved_sha256") != resolved:
    bad.append("resolved_sha256 与 history_config.resolved.yaml 不符")
if p.get("framesamp_store_meta_sha256") != fs8:
    bad.append(f"framesamp_store_meta_sha256={p.get('framesamp_store_meta_sha256')} 与训练库现场值 {fs8} 不符")
if expect_motion == "true":
    if p.get("motion_store_meta_sha256") != mo:
        bad.append(f"motion_store_meta_sha256 与训练库现场值 {mo} 不符")
    if (p.get("encoder") or {}).get("checkpoint_sha256") != enc:
        bad.append(f"encoder.checkpoint_sha256 与拷入件 {enc} 不符")
print(f"PROVENANCE_MATCH={'PASS' if not bad else 'FAIL'} motion={p.get('motion_enabled')} "
      f"fs8_store_meta={fs8[:16]} resolved={resolved[:16]}" + (f" encoder={enc[:16]}" if enc else ""))
for b in bad:
    print("  " + b)
sys.exit(1 if bad else 0)
PY
echo "阶段0完成"

# ---------------------------------------------------------------- 阶段 1：stage（硬链接）+ _run-meta + README
echo "阶段1开始：搭 stage 树（ckpt 走硬链接，不复制）+ _run-meta/train.log 清洗 + README"
rm -rf "$STAGE"; mkdir -p "$STAGE"
( cd "$SRC" && find . -mindepth 1 -type d -printf '%P\n' ) | while read -r d; do mkdir -p "$STAGE/$d"; done
( cd "$SRC" && find . -mindepth 1 -type f -printf '%P\n' ) | while read -r f; do ln -f "$SRC/$f" "$STAGE/$f"; done
mapfile -t ROOT_SRC_FILES < <(cd "$SRC" && find . -mindepth 1 -maxdepth 1 -type f -printf '%P\n' | sort)
echo "  run 根下平文件 ${#ROOT_SRC_FILES[@]} 个: ${ROOT_SRC_FILES[*]}"
mkdir -p "$STAGE/_run-meta"
tr '\r' '\n' < "$TRAIN_LOG" | grep -vE '%\|' > "$STAGE/_run-meta/train.log"
[ "$(tail -n 1 "$STAGE/_run-meta/train.log")" = "EXIT_CODE=0" ] || { echo "错误: 清洗后的 train.log 丢了 EXIT_CODE 尾行"; exit 1; }
echo "  _run-meta/train.log 清洗: $(wc -c < "$TRAIN_LOG") B → $(wc -c < "$STAGE/_run-meta/train.log") B"

python3 - "$README_SRC" "$STAGE" "$TRAIN_LOG" "$GIT_HEAD" "$FS8_SHA" "$MO_SHA" "$ENC_SHA" "$RESOLVED_SHA" "$EXPECT_STEPS" <<'PY'
import json, re, sys
from pathlib import Path
src, stage, log, head, fs8, mo, enc, resolved, steps = sys.argv[1:10]
stage = Path(stage)
lines = Path(log).read_text(errors="replace").splitlines()
grab = lambda k: next((l.split("=", 1)[1] for l in lines if l.startswith(k + "=")), "")
steps = steps.split()
step_bytes = {s: sum(p.stat().st_size for p in (stage / s).rglob("*") if p.is_file()) for s in steps}
vals = {
    "GIT_HEAD": head or "（训练日志未记录）",
    "START_UTC": grab("START_UTC") or "（未记录）",
    "END_UTC": grab("END_UTC") or "（未记录）",
    "FSDP_DEVICES": grab("FSDP_DEVICES") or "（未记录）",
    "NUM_WORKERS": grab("NUM_WORKERS") or "（未记录）",
    "HC": grab("HC") or "（未记录）",
    "FS8_META_SHA": fs8,
    "MOTION_META_SHA": mo or "—（motion 关闭）",
    "ENC_SHA": enc or "—（motion 关闭）",
    "RESOLVED_SHA": resolved,
    "NUM_STEPS": str(len(steps)),
    "STEPS": "、".join(steps),
    "CKPT_BYTES_TOTAL": f"{sum(step_bytes.values()):,}",
    "STEP_BYTES_RANGE": f"{min(step_bytes.values()):,}–{max(step_bytes.values()):,}",
}
text = Path(src).read_text()
for k, v in vals.items():
    text = text.replace(f"@@{k}@@", v)
if "@@" in text:
    sys.exit("错误: README 仍有未填占位 " + str(sorted({t for t in text.split("@@")[1::2]})))
(stage / "README.md").write_text(text)
print(f"README_RENDERED steps={len(steps)} ckpt_bytes={sum(step_bytes.values())} git_head={head[:12] or '-'}")
PY

if find "$STAGE" -type l -print | grep -q .; then
  echo "错误: stage 内出现符号链接"; find "$STAGE" -type l -print; exit 1
fi
echo "阶段1完成"

# ---------------------------------------------------------------- 阶段 2：上传前 sha256
echo "阶段2开始：上传前 sha256"
hash_tree "$STAGE" "$TMPD/SHA256SUMS.pre.txt"
mv "$TMPD/SHA256SUMS.pre.txt" "$STAGE/SHA256SUMS.pre.txt"
LOCAL_FILES=$(find "$STAGE" -type f | wc -l)
LOCAL_BYTES=$(find "$STAGE" -type f -printf '%s\n' | awk '{s+=$1} END{print s}')
PRE_LINES=$(wc -l < "$STAGE/SHA256SUMS.pre.txt")
SRC_FILES=$(find "$SRC" -type f | wc -l)
echo "LOCAL_FILES=$LOCAL_FILES LOCAL_BYTES=$LOCAL_BYTES PRE_LINES=$PRE_LINES SRC_FILES=$SRC_FILES"
# stage = 源全部文件 + _run-meta/train.log + README.md + SHA256SUMS.pre.txt
[ "$LOCAL_FILES" = "$((SRC_FILES + 3))" ] || { echo "错误: stage 文件数应为源 $SRC_FILES + 3，实为 $LOCAL_FILES"; exit 1; }
[ "$((PRE_LINES + 1))" = "$LOCAL_FILES" ] || { echo "错误: 清单行数+1 != stage 文件数"; exit 1; }
echo "阶段2完成"

# ---------------------------------------------------------------- 阶段 3：公开前体检
echo "阶段3开始：公开前体检（扫 stage，在建 bucket 之前）"
uv run --no-sync python scripts/dataset/hf_export/scan_hygiene.py \
  --root "$STAGE" --scope stage --allowlist "$ALLOWLIST" \
  --report "$EXPORT_ROOT/logs/hygiene-stage.json"
echo "阶段3完成"

if [ "$SMOKE" = 1 ]; then echo "SMOKE=PASS（到阶段 3 为止，未碰网络与凭据）"; echo "全部完成"; exit 0; fi

# ---------------------------------------------------------------- 阶段 4：凭据 + 建 bucket（公开）
echo "阶段4开始：凭据与身份 + 创建公开 bucket $BUCKET_ID"
[ -f "$REPO/v1-store/secrets/hf.env" ] || { echo "错误: 缺少 v1-store/secrets/hf.env"; exit 1; }
set -a; . "$REPO/v1-store/secrets/hf.env"; set +a
[ -n "${HF_TOKEN:-}" ] || { echo "错误: hf.env 未提供 HF_TOKEN"; exit 1; }
command -v uvx >/dev/null || { echo "错误: PATH 里找不到 uvx。PATH=$PATH"; exit 1; }
WHO="$(hf auth whoami --format agent 2>&1 | tr '\n' ' ' | tr -s ' ')"
echo "  whoami: $WHO"
case "$WHO" in *HongzeFu*) : ;; *) echo "错误: 身份不是 HongzeFu（$WHO）"; exit 1 ;; esac
echo "HF_WHOAMI=HongzeFu"
if bucket_stat totalFiles >/dev/null 2>&1; then
  echo "  bucket 已存在，跳过创建（幂等重跑）"
else
  hf buckets create "$BUCKET_ID"     # 不带私有开关即公开（口径 21）
fi
hf buckets settings "$BUCKET_ID" --public
assert_public "BUCKET_CREATED=$BUCKET_ID"
pre_files="$(bucket_stat totalFiles)"; pre_size="$(bucket_stat size)"
echo "BUCKET_FILES_AT_START=$pre_files BUCKET_BYTES_AT_START=$pre_size"
if [ "$pre_files" != "0" ] && [ "$pre_files" != "None" ] && [ ! -f "$EXPORT_ROOT/logs/uploaded.marker" ]; then
  echo "错误: 目标 bucket 非空（files=$pre_files）且本地无本轮上传标记，停止并交用户裁决"; exit 1
fi
echo "阶段4完成"

# ---------------------------------------------------------------- 阶段 5：按 step 目录分批上传
echo "阶段5开始：按 step 目录分 16 批 + _run-meta/ + 根下平文件"
touch "$EXPORT_ROOT/logs/uploaded.marker"
b=0
for s in $EXPECT_STEPS _run-meta; do
  b=$((b + 1))
  DONE="$EXPORT_ROOT/logs/uploaded/$s.done"
  if [ -f "$DONE" ]; then echo "  [$b/17] $s 已完成，跳过"; continue; fi
  echo "  [$b/17] $s  $(find "$STAGE/$s" -type f | wc -l) 件 / $(find "$STAGE/$s" -type f -printf '%s\n' | awk '{s+=$1} END{print s}') 字节"
  sync_dir "$s"
  touch "$DONE"
done
mapfile -t ROOTFILES < <(cd "$STAGE" && find . -mindepth 1 -maxdepth 1 -type f -printf '%P\n' | sort)
echo "  根下平文件 ${#ROOTFILES[@]} 个"
for f in "${ROOTFILES[@]}"; do cp_file "$f"; done
echo "UPLOAD_DONE batches=17 root_files=${#ROOTFILES[@]}"
echo "阶段5完成"

# ---------------------------------------------------------------- 阶段 6：计数层
echo "阶段6开始：计数层核对"
BUCKET_FILES="$(bucket_stat totalFiles)"; BUCKET_BYTES="$(bucket_stat size)"
if [ "$BUCKET_FILES" != "$LOCAL_FILES" ] || [ "$BUCKET_BYTES" != "$LOCAL_BYTES" ]; then
  echo "  首次不符（files=$BUCKET_FILES bytes=$BUCKET_BYTES），统计接口可能异步滞后，120s 后复查"
  sleep 120
  BUCKET_FILES="$(bucket_stat totalFiles)"; BUCKET_BYTES="$(bucket_stat size)"
fi
echo "BUCKET_FILES=$BUCKET_FILES BUCKET_BYTES=$BUCKET_BYTES"
[ "$BUCKET_FILES" = "$LOCAL_FILES" ] || { echo "错误: 文件数不符 $BUCKET_FILES != $LOCAL_FILES"; exit 1; }
[ "$BUCKET_BYTES" = "$LOCAL_BYTES" ] || { echo "错误: 字节数不符 $BUCKET_BYTES != $LOCAL_BYTES"; exit 1; }
echo "阶段6完成"

# ---------------------------------------------------------------- 阶段 7：df 断言 + 全量回读
echo "阶段7开始：df 余量断言 + 全量回读到 $VERIFY"
# 写法照抄 run_motionjepa_mixed8_200_export.sh 阶段 8：需要量 = stage 总字节 − verify 已有字节，另留 0.4 TB
mkdir -p "$VERIFY"
VERIFY_HAVE=$(find "$VERIFY" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
READBACK_NEED=$(( LOCAL_BYTES - VERIFY_HAVE )); [ "$READBACK_NEED" -ge 0 ] || READBACK_NEED=0
DF_AVAIL=$(df -B1 --output=avail "$VERIFY" | tail -n 1 | tr -d ' ')
echo "DF_CHECK avail=$DF_AVAIL readback_need=$READBACK_NEED verify_have=$VERIFY_HAVE margin=$DF_MARGIN_BYTES"
if [ "$DF_AVAIL" -lt $(( READBACK_NEED + DF_MARGIN_BYTES )) ]; then
  echo "错误: 磁盘余量不足（avail=$DF_AVAIL < 回读 $READBACK_NEED + 0.4 TB），停止回读交用户裁决"; exit 1
fi
ok=0
for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
  if hf sync "$BUCKET" "$VERIFY"; then ok=1; break; fi
  echo "回读第 $attempt/$MAX_ATTEMPTS 次尝试中断"
  if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then echo "60s 后重试续传"; sleep 60; fi
done
[ "$ok" = 1 ] || { echo "READBACK_FAIL object=$BUCKET attempts=$MAX_ATTEMPTS"; echo "错误: 全量回读首试 + 重试 $UPLOAD_RETRIES 次仍失败"; exit 1; }
echo "阶段7完成"

# ---------------------------------------------------------------- 阶段 8：内容层验收
echo "阶段8开始：上传后 sha256（POST）+ 与上传前逐行对照"
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
echo "RESULT=PASS"
echo "阶段8完成"

# ---------------------------------------------------------------- 阶段 9：源侧第三遍
echo "阶段9开始：收尾断言（源侧第三遍 sha256，证明硬链接未改动原件）"
hash_tree "$SRC" "$TMPD/SHA256SUMS.src-after.txt"
cp "$TMPD/SHA256SUMS.src-after.txt" "$EXPORT_ROOT/SHA256SUMS.src-after.txt"
grep -vE '  (README\.md|_run-meta/train\.log)$' "$STAGE/SHA256SUMS.pre.txt" > "$TMPD/pre.src-only.txt"
[ "$((PRE_LINES - $(wc -l < "$TMPD/pre.src-only.txt")))" = 2 ] || { echo "错误: 剔除行数不是 2"; exit 1; }
if diff "$TMPD/pre.src-only.txt" "$TMPD/SHA256SUMS.src-after.txt"; then
  echo "SRC_UNCHANGED=OK checked=$(wc -l < "$TMPD/SHA256SUMS.src-after.txt")"
else
  echo "错误: 源文件 sha256 发生变化（上方 diff 即差异行）"; exit 1
fi
tr '\r' '\n' < "$TRAIN_LOG" | grep -vE '%\|' | cmp -s - "$STAGE/_run-meta/train.log" \
  || { echo "错误: 源训练日志重新清洗后与上传的 _run-meta/train.log 不同"; exit 1; }
echo "  训练日志重新清洗一致"
echo "阶段9完成"

# ---------------------------------------------------------------- 阶段 10：公开性确认
echo "阶段10开始：公开性确认（private==false + 匿名读）"
assert_public "BUCKET_PUBLIC_FINAL"
echo "BUCKET_PUBLIC=True"
rm -rf "$TMPD/anon"; mkdir -p "$TMPD/anon"
if env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HF_HUB_OFFLINE \
     uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf \
     sync "$BUCKET" "$TMPD/anon" --include 'README.md' \
   && [ -s "$TMPD/anon/README.md" ]; then
  echo "PUBLIC_ANON_READ=OK"
else
  echo "错误: 匿名读取失败，bucket 可能并未真正公开"; exit 1
fi
echo "阶段10完成"
echo "全部完成"
