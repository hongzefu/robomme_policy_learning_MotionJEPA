#!/usr/bin/env bash
# v9-st 编码器增量上传：与最终驱动共用 bucket 根路径、归属标记和逐 epoch 的 SHA 标记。
# 用法：bash mj_v9st_ckpt_inc.sh [--dry-run] <RUN_NAME> <BUCKET_ID> <EXPORT_NAME> <TRAIN_TMUX>
# 干跑只读 CPU 权重，内部扫描两轮，不读凭据、不调用 HF；状态写到源旁的专用干跑目录。
set -euo pipefail
DRY_RUN=0
if [ "${1:-}" = --dry-run ]; then DRY_RUN=1; shift; fi
[ "$#" = 4 ] || { echo "用法：$0 [--dry-run] <RUN_NAME> <BUCKET_ID> <EXPORT_NAME> <TRAIN_TMUX>"; exit 2; }
RUN_NAME="$1"; BUCKET_ID="$2"; EXPORT_NAME="$3"; TRAIN_TMUX="$4"
for name in "$RUN_NAME" "$EXPORT_NAME"; do
  [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "错误: 非法运行名或导出名"; exit 2; }
done
[[ "$BUCKET_ID" =~ ^HongzeFu/[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "错误: bucket 必须属于 HongzeFu"; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
SRC="/scratch/hongze/MotionJEPA/runs/$RUN_NAME"
EXPORT_ROOT="$REPO/v1-store/exports/$EXPORT_NAME"
if [ "$DRY_RUN" = 1 ]; then
  SRC="${SRC_OVERRIDE:-$SRC}"
  EXPORT_ROOT="$SRC.inc-dry-run"
  [[ "${DRYRUN_ALIVE:-1}" =~ ^[01]$ ]] || { echo "错误: DRYRUN_ALIVE 只能为 0 或 1"; exit 2; }
elif [ "${SRC_OVERRIDE+x}" = x ] || [ "${DRYRUN_ALIVE+x}" = x ]; then
  echo "错误: SRC_OVERRIDE/DRYRUN_ALIVE 只允许用于 --dry-run"; exit 2
fi
BUCKET="hf://buckets/$BUCKET_ID"
PY="/scratch/hongze/MotionJEPA/.venv/bin/python"
UPLOAD_RETRIES=3; MAX_ATTEMPTS=$((UPLOAD_RETRIES + 1))
export HF_HOME="$REPO/v1-store/cache/hf"
export HF_XET_CACHE="$REPO/v1-store/cache/hf-xet"
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export UV_PYTHON_INSTALL_DIR=/scratch/hongze/.cache/uv-python
export UV_LINK_MODE=copy PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE || true

hf() { ionice -c 3 nice -n 19 uvx --python 3.11 --from 'huggingface_hub[cli]==1.30.0' hf "$@"; }
bucket_stat() {
  local body
  body="$(curl -sf -H "Authorization: Bearer $HF_TOKEN" 'https://huggingface.co/api/buckets/HongzeFu')" || return 2
  printf '%s' "$body" | "$PY" -c '
import json, sys
items = json.load(sys.stdin)
for item in items:
    if item["id"] == sys.argv[1]:
        print(item[sys.argv[2]])
        break
else:
    sys.exit(3)
' "$BUCKET_ID" "$1"
}
assert_public() {
  local private; private="$(bucket_stat private)"
  echo "$1 private=$private"
  case "$private" in False|false) : ;; *) echo "错误: bucket 未公开"; exit 1 ;; esac
}
upload_marker() {
  # 两个驱动共用严格身份格式；旧空标记和其他 run/bucket 的标记均不得迁移为本轮归属。
  "$PY" - "$1" "$EXPORT_ROOT/logs/uploaded.marker" "$RUN_NAME" "$BUCKET_ID" <<'PYMARKER'
import json, os, sys, tempfile
from pathlib import Path
mode, filename, run_name, bucket_id = sys.argv[1:]
path = Path(filename)
expected = dict(schema_version=1, run_name=run_name, bucket_id=bucket_id)
if mode == "check":
    try:
        actual = json.loads(path.read_text())
    except (OSError, ValueError):
        sys.exit(1)
    matches = isinstance(actual, dict) and type(actual.get("schema_version")) is int and actual == expected
    sys.exit(0 if matches else 1)
if mode != "write":
    sys.exit(2)
fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
try:
    with os.fdopen(fd, "w") as stream:
        json.dump(expected, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PYMARKER
}
BUCKET_READY=0
ensure_bucket() {
  [ "$BUCKET_READY" = 1 ] && return 0
  local count rc
  if { [ -e "$EXPORT_ROOT/logs/uploaded.marker" ] || [ -L "$EXPORT_ROOT/logs/uploaded.marker" ]; } && ! upload_marker check; then
    echo "UPLOAD_MARKER_MISMATCH 本地标记无效或属于其他 run/bucket，停止"; exit 1
  fi
  if count="$(bucket_stat totalFiles)"; then
    [[ "$count" =~ ^[0-9]+$ ]] || { echo "错误: bucket 文件数无效"; exit 1; }
    if [ "$count" != 0 ] && ! upload_marker check; then
      echo "错误: 目标 bucket 非空且本地无本轮上传标记，停止"; exit 1
    fi
  else
    rc=$?
    [ "$rc" = 3 ] || { echo "错误: 无法查询 bucket，拒绝创建或修改公开性"; exit 1; }
    hf buckets create "$BUCKET_ID"
  fi
  hf buckets settings "$BUCKET_ID" --public
  assert_public "BUCKET_CREATED=$BUCKET_ID"
  upload_marker write
  BUCKET_READY=1
}
validate_ckpt() {
  "$PY" -I - "$1" "$2" <<'PY'
import sys
import torch
try:
    item = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    assert item.get("epoch") == int(sys.argv[2])
    assert item.get("arch") == "wan-latent-v9-st"
    assert item.get("ema_only") is True
    assert not any("optim" in str(key).lower() for key in item)
except Exception as exc:
    print(f"检查点校验失败: {type(exc).__name__}")
    sys.exit(1)
PY
}
first_readback() {
  local target="$EXPORT_ROOT/tmp/first-readback" actual attempt ok=0
  mkdir -p "$target"
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    if env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HF_HUB_OFFLINE HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME="$target/hf-home" \
      ionice -c 3 nice -n 19 uvx --python 3.11 --from 'huggingface_hub[cli]==1.30.0' hf \
      buckets cp "$BUCKET/checkpoint_epoch_2.pt" "$target/checkpoint_epoch_2.pt"; then ok=1; break; fi
    [ "$attempt" -ge "$MAX_ATTEMPTS" ] || sleep 60
  done
  [ "$ok" = 1 ] || { echo "FIRST_READBACK=FAIL epoch=2 attempts=$MAX_ATTEMPTS"; exit 1; }
  actual="$(sha256sum "$target/checkpoint_epoch_2.pt" | cut -d' ' -f1)"
  [ "$actual" = "$1" ] || { echo "FIRST_READBACK=FAIL epoch=2"; exit 1; }
  echo "FIRST_READBACK=PASS epoch=2 sha256=$actual"
  printf '%s\n' "$actual" > "$EXPORT_ROOT/logs/first-readback.done"
}
write_inc_manifest() {
  # 每轮从已提交标记重建，恢复「已写 done、尚未更新清单」间隙的中断。
  local epoch file digest
  : > "$EXPORT_ROOT/logs/inc-sha256.txt.tmp"
  for epoch in $(seq 2 2 60); do
    file="$EXPORT_ROOT/logs/uploaded/ckpt-epoch-$epoch.done"
    [ -f "$file" ] || continue
    digest="$(cat "$file")"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { echo "INC_SHA_MISMATCH epoch=$epoch 标记内容无效"; exit 1; }
    printf '%s  checkpoint_epoch_%s.pt\n' "$digest" "$epoch" >> "$EXPORT_ROOT/logs/inc-sha256.txt.tmp"
  done
  mv "$EXPORT_ROOT/logs/inc-sha256.txt.tmp" "$EXPORT_ROOT/logs/inc-sha256.txt"
}

[ -d "$SRC" ] && [ -x "$PY" ] || { echo "错误: 缺源目录或 CPU 解释器"; exit 1; }
for d in "$(dirname "$EXPORT_ROOT")" "$EXPORT_ROOT"; do
  [ ! -L "$d" ] || { echo "错误: 导出目录为符号链接"; exit 1; }
done
mkdir -p "$EXPORT_ROOT/logs/uploaded"
if [ "$DRY_RUN" = 0 ]; then
  # 启动即核身份，已有全部 done 的重跑也不能把另一 run/bucket 的完成记录据为己有。
  if { [ -e "$EXPORT_ROOT/logs/uploaded.marker" ] || [ -L "$EXPORT_ROOT/logs/uploaded.marker" ]; } && ! upload_marker check; then
    echo "UPLOAD_MARKER_MISMATCH 本地标记无效或属于其他 run/bucket，停止"; exit 1
  fi
  set -a; . "$REPO/v1-store/secrets/hf.env"; set +a
  [ -n "${HF_TOKEN:-}" ] || { echo "错误: hf.env 未提供 HF_TOKEN"; exit 1; }
  WHO="$(hf auth whoami --format agent)"
  case "$WHO" in *HongzeFu*) echo "HF_WHOAMI=HongzeFu" ;; *) echo "错误: HF 身份不符"; exit 1 ;; esac
fi
echo "INC_START run=$RUN_NAME bucket=$BUCKET_ID dry_run=$DRY_RUN"
round=0
while true; do
  round=$((round + 1))
  if [ "$DRY_RUN" = 1 ]; then
    alive="${DRYRUN_ALIVE:-1}"
  else
    alive=0; tmux has-session -t "=$TRAIN_TMUX" 2>/dev/null && alive=1
  fi
  for epoch in $(seq 2 2 60); do
    file="$SRC/checkpoint_epoch_$epoch.pt"
    done_file="$EXPORT_ROOT/logs/uploaded/ckpt-epoch-$epoch.done"
    if [ -f "$done_file" ] && [ "$DRY_RUN" = 0 ]; then
      if [ "$epoch" = 2 ] && [ ! -f "$EXPORT_ROOT/logs/first-readback.done" ]; then
        ensure_bucket; first_readback "$(cat "$done_file")"
      fi
      continue
    fi
    [ -f "$file" ] || continue
    size="$(stat -c %s "$file")"
    if [ "$alive" = 1 ]; then
      previous="$(cat "$EXPORT_ROOT/logs/inc-size-$epoch" 2>/dev/null || true)"
      printf '%s\n' "$size" > "$EXPORT_ROOT/logs/inc-size-$epoch"
      age=$(( $(date +%s) - $(stat -c %Y "$file") ))
      if [ "$age" -lt 600 ] || [ "$size" != "$previous" ]; then
        echo "INC_WAIT epoch=$epoch round=$round age=$age stable=$([ "$size" = "$previous" ] && echo 1 || echo 0)"
        continue
      fi
    fi
    if ! validate_ckpt "$file" "$epoch"; then
      echo "INC_CKPT_INVALID epoch=$epoch"
      [ "$DRY_RUN" = 1 ] && continue
      exit 1
    fi
    if [ "$DRY_RUN" = 1 ]; then echo "INC_DRYRUN=PASS epoch=$epoch round=$round"; continue; fi
    ensure_bucket
    digest="$(sha256sum "$file" | cut -d' ' -f1)"
    ok=0
    for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
      if hf buckets cp "$file" "$BUCKET/checkpoint_epoch_$epoch.pt"; then ok=1; break; fi
      [ "$attempt" -ge "$MAX_ATTEMPTS" ] || sleep 60
    done
    [ "$ok" = 1 ] || { echo "UPLOAD_FAIL object=checkpoint_epoch_$epoch.pt attempts=$MAX_ATTEMPTS"; exit 1; }
    # e2 匿名读回也成功才落逐件完成标记，避免重启后漏掉这道验收。
    if [ "$epoch" = 2 ]; then first_readback "$digest"; fi
    printf '%s\n' "$digest" > "$done_file.tmp"; mv "$done_file.tmp" "$done_file"
    echo "INC_UPLOADED epoch=$epoch bytes=$size sha256=$digest"
  done
  if [ "$DRY_RUN" = 1 ]; then
    [ "$round" -lt 2 ] || { echo "INC_DRYRUN_END rounds=2"; exit 0; }
    sleep 1; continue
  fi
  write_inc_manifest
  if [ "$alive" = 0 ]; then
    completed=0
    for epoch in $(seq 2 2 60); do
      [ ! -f "$EXPORT_ROOT/logs/uploaded/ckpt-epoch-$epoch.done" ] || completed=$((completed + 1))
    done
    if [ "$completed" = 30 ]; then echo "INC_END run=$RUN_NAME done=30"; exit 0; fi
    echo "INC_END_INCOMPLETE run=$RUN_NAME done=$completed"; exit 1
  fi
  sleep 300
done
