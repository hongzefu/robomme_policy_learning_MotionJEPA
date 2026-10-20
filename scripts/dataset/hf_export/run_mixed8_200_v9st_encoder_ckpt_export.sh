#!/usr/bin/env bash
# v9-st 编码器正式导出，派生自 mixed8-200 旧驱动。
# 30 个 EMA 检查点、训练档案、三模型臂四种消融的 192 段视频和 5+k 份报告。
# 成功计算但四臂不通过仍如实上传；RGB 两条 PASS 与评估正常退出是导出前提。
# 用法：SMOKE=1 bash 本脚本（只到阶段3，不读凭据、不触网）；正式模式不设 SMOKE。
# 离线路径覆盖：M8X_SRC / M8X_TRAIN_LOG / M8X_MJ / M8X_COMPARE / M8X_MIDPROBE /
# M8X_EVAL_LOG / M8X_EXPORT_ROOT。最终完成信号为 RESULT=PASS 与「全部完成」。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
cd "$REPO"

RUN_NAME="wan-mixed8-200-st512q16-b88x8-60ep-a"
BUCKET_ID="HongzeFu/motionjepa-wan-mixed8-200-st512q16-60ep-v1"
BUCKET="hf://buckets/$BUCKET_ID"
SRC="${M8X_SRC:-/scratch/hongze/MotionJEPA/runs/$RUN_NAME}"
# 训练原始日志（tmux tee 落盘的那份，B1_GATE 读的也是它）。stage 里的 train.log 由本驱动现场清洗生成：
# 去掉 tqdm 中间态（AGENTS 第 12 条 (3) 的同一条命令），保留 Epoch 汇总行、横幅与 EXIT_CODE= 尾行。
TRAIN_LOG="${M8X_TRAIN_LOG:-/scratch/hongze/logs/$RUN_NAME.train.log}"
MJ="${M8X_MJ:-/scratch/hongze/MotionJEPA}"     # 只借它的 .venv 读 ckpt 元数据（torch），不写
COMPARE="${M8X_COMPARE:-/scratch/hongze/MotionJEPA/runs/$RUN_NAME-compare}"
MIDPROBE_DIR="${M8X_MIDPROBE:-/scratch/hongze/MotionJEPA/runs/$RUN_NAME-midprobe}"
EVAL_LOG="${M8X_EVAL_LOG:-/scratch/hongze/logs/cc-v9st-eval.log}"
EXPORT_ROOT="${M8X_EXPORT_ROOT:-$REPO/v1-store/exports/hf-mixed8-200-v9st-encoder-ckpt}"
ALLOWLIST="$REPO/scripts/dataset/hf_export/hygiene_allowlist_mixed8_200_v9st_encoder_ckpt.json"
README_SRC="$REPO/scripts/dataset/hf_export/mixed8_200_v9st_encoder_ckpt_bucket_README.md"
STAGE="$EXPORT_ROOT/stage"
VERIFY="$EXPORT_ROOT/verify"
TMPD="$EXPORT_ROOT/tmp"
MP4_PLAN="$EXPORT_ROOT/logs/eval-fourarm_upload_plan.json"
EXPECT_CKPTS=30
EXPECT_MP4=192
ROOT_WHITELIST=(config.yaml train_metrics_epoch.jsonl sig_loss_epoch1.csv gpu_samples.csv train.log)
DF_MARGIN_BYTES=400000000000          # 0.4 TB（十进制，G12）
NPROC=8
# 网络重试口径（计划口径 27「重试 3 次」）：首试 + 重试 UPLOAD_RETRIES 次，仍失败即非 0 退出并点名失败对象，
# 由主会话写进 pending_decisions（status=fail）。上传、断点续传回读一律用这一个常量。
UPLOAD_RETRIES=3
MAX_ATTEMPTS=$((UPLOAD_RETRIES + 1))
SMOKE="${SMOKE:-0}"

for v in M8X_SRC M8X_TRAIN_LOG M8X_MJ M8X_COMPARE M8X_MIDPROBE M8X_EVAL_LOG M8X_EXPORT_ROOT; do
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

hf() { ionice -c 3 nice -n 19 uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf "$@"; }

bucket_stat() {
  local body
  body="$(curl -sf -H "Authorization: Bearer $HF_TOKEN" 'https://huggingface.co/api/buckets/HongzeFu')" || return 2
  printf '%s' "$body" | "$MJ/.venv/bin/python" -c '
import json, sys
for item in json.load(sys.stdin):
    if item["id"] == sys.argv[1]:
        print(item[sys.argv[2]])
        break
else:
    sys.exit(3)
' "$BUCKET_ID" "$1"
}
assert_public() {  # $1 = 判定行前缀
  local p; p="$(bucket_stat private)"
  echo "$1 private=$p"
  case "$p" in
    False|false) : ;;
    *) echo "错误: bucket 不是公开的（private=$p），要求 private==false"; exit 1 ;;
  esac
}

upload_marker() {
  # 两个驱动共用严格身份格式；旧空标记和其他 run/bucket 的标记均不得迁移为本轮归属。
  "$MJ/.venv/bin/python" - "$1" "$EXPORT_ROOT/logs/uploaded.marker" "$RUN_NAME" "$BUCKET_ID" <<'PYMARKER'
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

clean_train_log() {
  tr '\r' '\n' < "$TRAIN_LOG" | sed -E 's/^Epoch .*\](MOTION_)/\1/' | grep -vE '%\|'
}

hash_tree() {  # $1 = 目录, $2 = 输出文件（绝对路径，不能落在被扫描目录内）
  ( cd "$1" && find . -type f -not -name 'SHA256SUMS.*' -printf '%P\n' | sort \
      | xargs -P "$NPROC" -n 16 sha256sum | sort -k 2 > "$2" )
}

# 硬链接要求同设备；不同设备时退回复制（复制的字节也纳入后续源侧复核），并在日志里写明
link_or_copy() {  # $1 = 源, $2 = 目标
  mkdir -p "$(dirname "$2")"
  ln -f "$1" "$2" 2>/dev/null || cp -f "$1" "$2"
}

# ---------------------------------------------------------------- 阶段 0：预检
echo "阶段0开始：预检（源 run 已收尾、四臂与 RGB 已完成、白名单外条目）"
for d in "$(dirname "$EXPORT_ROOT")" "$EXPORT_ROOT"; do
  [ -L "$d" ] && { echo "错误: $d 是符号链接，拒绝写入"; exit 1; }
done
[ -d "$SRC" ] || { echo "错误: 源 run 目录不存在 $SRC"; exit 1; }
[ -d "$COMPARE/videos" ] || { echo "错误: 缺四臂视频目录 $COMPARE/videos"; exit 1; }
[ -f "$COMPARE/summary.json" ] || { echo "错误: 缺四臂 summary.json"; exit 1; }
[ -f "$ALLOWLIST" ] && [ -f "$README_SRC" ] || { echo "错误: 缺体检账本或 README 模板"; exit 1; }
mkdir -p "$STAGE" "$VERIFY" "$TMPD" "$EXPORT_ROOT/logs" "$EXPORT_ROOT/logs/uploaded"

# 这条 run 必须已经跑完：判据与 runbook 的 B1_GATE 逐字相同（末行 EXIT_CODE=0、清洗后 60 行 Epoch）
[ -f "$TRAIN_LOG" ] || { echo "错误: 训练日志不存在 $TRAIN_LOG"; exit 1; }
LAST_LINE="$(tail -n 1 "$TRAIN_LOG")"
[ "$LAST_LINE" = "EXIT_CODE=0" ] || { echo "错误: $TRAIN_LOG 末行为「$LAST_LINE」，不是 EXIT_CODE=0"; exit 1; }
NEPOCH_LINES=$(clean_train_log | grep -c '^Epoch ' || true)
echo "  训练日志: 末行 $LAST_LINE，清洗后 Epoch 行 $NEPOCH_LINES 条"
[ "$NEPOCH_LINES" = 60 ] || { echo "错误: 清洗后 Epoch 行应为 60，实为 $NEPOCH_LINES"; exit 1; }

# 数值质量不通过允许归档，计算未成功或 RGB 产物不齐则拒绝。
[ -f "$EVAL_LOG" ] || { echo "错误: 缺评估日志 $EVAL_LOG"; exit 1; }
[ "$(tail -n 1 "$EVAL_LOG")" = EXIT_CODE=0 ] || { echo "错误: 评估日志末行不是 EXIT_CODE=0"; exit 1; }
for pattern in '^FOURARM=(PASS|FAIL)( |$)' '^FOURARM_ROBUST=(PASS|FAIL)( |$)' '^RGB_COMPARE=PASS( |$)' '^RGB_ABLATION_VIDEOS=PASS( |$)'; do
  grep -Eq "$pattern" "$EVAL_LOG" || { echo "错误: 缺评估判定行 $pattern"; exit 1; }
done
NMP4=$(find "$COMPARE/videos" -mindepth 2 -maxdepth 2 -type f -name '*.mp4' | wc -l)
[ "$NMP4" = "$EXPECT_MP4" ] || { echo "错误: 视频应为 $EXPECT_MP4 个，实为 $NMP4"; exit 1; }
for arm in new old ref; do
  [ -f "$COMPARE/$arm/result.json" ] || { echo "错误: 缺 $arm/result.json"; exit 1; }
  count=$(find "$COMPARE/videos/$arm" -maxdepth 1 -type f -name '*.mp4' | wc -l)
  [ "$count" = 64 ] || { echo "错误: $arm 视频应为 64 个，实为 $count"; exit 1; }
done
[ -f "$COMPARE/windows.json" ] || { echo "错误: 缺 windows.json"; exit 1; }
# 只认看守的五个目标终态；失败/跳过/缺失均记录，但不计入成功报告数 k。
uv run --no-sync python - "$MIDPROBE_DIR" "$TMPD/midprobe-selection.json" <<'PY'
import json, sys
from pathlib import Path
root, target = map(Path, sys.argv[1:])
selected, notes, lines = [], [], []
for epoch in (10, 20, 30, 40, 50):
    status_path, result_path = root / f"e{epoch}.status.json", root / f"e{epoch}.json"
    label = "missing"
    try:
        status = json.loads(status_path.read_text())
        label = status.get("status", "unknown") if status.get("terminal") is True else "unfinished"
        if label == "ok":
            result = json.loads(result_path.read_text())
            if result.get("status") != "PASS" or result.get("epoch") != epoch:
                label = "invalid_result"
            else:
                selected.append(epoch)
                line = result.get("midprobe_line") or status.get("midprobe_line")
                if isinstance(line, str) and line.startswith(f"MIDPROBE epoch={epoch} ") and "\n" not in line:
                    lines.append(line)
        elif label not in ("fail", "skip", "unfinished"):
            label = "unknown"
    except (OSError, ValueError, AttributeError, TypeError):
        label = "missing" if not status_path.exists() else "invalid_result"
    note = f"MIDPROBE_EXPORT epoch={epoch} status={label} included={int(epoch in selected)}"
    notes.append(note)
    print(note)
target.write_text(json.dumps(dict(selected=selected, notes=notes, lines=lines), ensure_ascii=False, indent=2) + "\n")
PY
mapfile -t MIDPROBE_EPOCHS < <(uv run --no-sync python -c 'import json,sys; print(*json.load(open(sys.argv[1]))["selected"],sep="\n")' "$TMPD/midprobe-selection.json")
# 空列表不能通过 print() 变成一个空元素。
[ "${MIDPROBE_EPOCHS[*]:-}" ] || MIDPROBE_EPOCHS=()
NREP=$((5 + ${#MIDPROBE_EPOCHS[@]}))

# run 目录根：除 30 个 ckpt、白名单文件与刻意不传的 wandb/ 之外不得有别的条目
mapfile -t CKPT_NAMES < <(for e in $(seq 2 2 60); do echo "checkpoint_epoch_$e.pt"; done)
UNEXPECTED=()
while IFS= read -r name; do
  case " ${CKPT_NAMES[*]} ${ROOT_WHITELIST[*]} wandb " in
    *" $name "*) : ;;
    *) UNEXPECTED+=("$name") ;;
  esac
done < <(cd "$SRC" && find . -mindepth 1 -maxdepth 1 -printf '%P\n' | sort)
if [ "${#UNEXPECTED[@]}" -gt 0 ]; then
  echo "错误: run 目录根下有白名单外条目（既不传也未裁决）: ${UNEXPECTED[*]}"; exit 1
fi
for n in "${CKPT_NAMES[@]}"; do
  [ -f "$SRC/$n" ] || { echo "错误: 缺 $SRC/$n"; exit 1; }
done
for n in config.yaml train_metrics_epoch.jsonl sig_loss_epoch1.csv gpu_samples.csv; do
  [ -f "$SRC/$n" ] || { echo "错误: 缺 $SRC/$n"; exit 1; }
done
NMETRIC=$(grep -c . "$SRC/train_metrics_epoch.jsonl" || true)
[ "$NMETRIC" = 60 ] || { echo "错误: train_metrics_epoch.jsonl 应为 60 行，实为 $NMETRIC"; exit 1; }

# README 写「EMA-only，没有 live 权重与优化器状态」：30 个 ckpt 逐个用 B1 同一判据核实，不只核 epoch 60
[ -x "$MJ/.venv/bin/python" ] || { echo "错误: 找不到 $MJ/.venv/bin/python（读 ckpt 元数据要 torch）"; exit 1; }
"$MJ/.venv/bin/python" -I - "$SRC" "${CKPT_NAMES[@]}" <<'PY'
import sys, torch
src, names = sys.argv[1], sys.argv[2:]
bad = []
for n in names:
    epoch = int(n.removeprefix("checkpoint_epoch_").removesuffix(".pt"))
    c = torch.load(f"{src}/{n}", map_location="cpu", weights_only=False)
    if not (c.get("epoch") == epoch and c.get("arch") == "wan-latent-v9-st"
            and c.get("ema_only") is True and c.get("has_live_weights") is False
            and not any("optim" in str(k).lower() for k in c)):
        bad.append(n)
print(f"CKPT_EMA_ONLY={'PASS' if not bad else 'FAIL'} checked={len(names)} bad={len(bad)}")
if bad:
    print("  不满足 EMA-only 的 ckpt:", bad); sys.exit(1)
PY
echo "阶段0完成"

# ---------------------------------------------------------------- 阶段 1：stage（白名单，硬链接）
echo "阶段1开始：按白名单搭 stage（硬链接，零拷贝）+ README"
rm -rf "$STAGE"; mkdir -p "$STAGE"
for n in "${CKPT_NAMES[@]}" config.yaml train_metrics_epoch.jsonl sig_loss_epoch1.csv gpu_samples.csv; do
  link_or_copy "$SRC/$n" "$STAGE/$n"
done
# 清洗：同一条命令（AGENTS 第 12 条 (3)），清洗后再核一次 60 行 Epoch 与 EXIT_CODE=0 尾行零丢失
clean_train_log > "$STAGE/train.log"
[ "$(grep -c '^Epoch ' "$STAGE/train.log")" = 60 ] && [ "$(tail -n 1 "$STAGE/train.log")" = "EXIT_CODE=0" ] \
  || { echo "错误: 清洗后的 train.log 丢了 Epoch 行或 EXIT_CODE 尾行"; exit 1; }
echo "  train.log 清洗: $(wc -c < "$TRAIN_LOG") B → $(wc -c < "$STAGE/train.log") B"
mkdir -p "$STAGE/eval-fourarm/videos" "$STAGE/eval-fourarm/reports"
for arm in new old ref; do
  while IFS= read -r file; do
    name="$(basename "$file")"
    case "$name" in *_"$arm"_normal.mp4|*_"$arm"_zero.mp4|*_"$arm"_cross.mp4|*_"$arm"_neighbor.mp4) : ;;
      *) echo "错误: 视频名不符合任务/模型臂/消融契约 $name"; exit 1 ;; esac
    [ ! -e "$STAGE/eval-fourarm/videos/$name" ] || { echo "错误: 拍平视频重名 $name"; exit 1; }
    link_or_copy "$file" "$STAGE/eval-fourarm/videos/$name"
  done < <(find "$COMPARE/videos/$arm" -maxdepth 1 -type f -name '*.mp4' | sort)
  link_or_copy "$COMPARE/$arm/result.json" "$STAGE/eval-fourarm/reports/result-$arm.json"
done
for name in summary windows; do
  link_or_copy "$COMPARE/$name.json" "$STAGE/eval-fourarm/reports/$name.json"
done
for epoch in "${MIDPROBE_EPOCHS[@]}"; do
  link_or_copy "$MIDPROBE_DIR/e$epoch.json" "$STAGE/eval-fourarm/reports/midprobe-e$epoch.json"
done

# README：把锚点 sha 与实测数字现填进模板（不经人手），填完断言没有残留占位
uv run --no-sync python - "$README_SRC" "$STAGE" "$NMP4" "$NREP" "$EVAL_LOG" "$TMPD/midprobe-selection.json" <<'PY'
import json, sys
from pathlib import Path
src, stage = Path(sys.argv[1]), Path(sys.argv[2])
nmp4, nrep = sys.argv[3:5]
eval_lines = Path(sys.argv[5]).read_text().splitlines()
midprobe = json.loads(Path(sys.argv[6]).read_text())
rows = [json.loads(l) for l in (stage / "train_metrics_epoch.jsonl").read_text().splitlines() if l.strip()]
ck = sorted(stage.glob("checkpoint_epoch_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
sizes = sorted({p.stat().st_size for p in ck})
vals = {
    "NUM_CKPT": str(len(ck)),
    "CKPT_BYTES_TOTAL": f"{sum(p.stat().st_size for p in ck):,}",
    "CKPT_SIZES": " / ".join(f"{s:,}" for s in sizes),
    "NUM_EPOCH_ROWS": str(len(rows)),
    "NUM_MP4": nmp4,
    "NUM_REPORTS": nrep,
}
if [r["epoch"] for r in rows] != list(range(1, 61)):
    sys.exit("错误: train_metrics_epoch.jsonl 的 epoch 列不是 1..60")
for tag, r in (("E1", rows[0]), ("E60", rows[-1])):
    vals[f"{tag}_TOTAL"] = f"{r['loss_total']:.8f}"
    vals[f"{tag}_LATENT"] = f"{r['loss_raw/latent']:.8f}"
    vals[f"{tag}_SIGREG"] = f"{r['loss_raw/sigreg']:.5f}"
    vals[f"{tag}_STD"] = f"{r['motion/std_batch']:.5f}"
    vals[f"{tag}_COS"] = f"{r['motion/cos_cross_chunk']:.7f}"
for key, prefix in (("FOURARM", "FOURARM="), ("FOURARM_ROBUST", "FOURARM_ROBUST="), ("RGB", "RGB_COMPARE=")):
    vals[key] = [line for line in eval_lines if line.startswith(prefix)][-1]
vals["RGB_VIDEOS"] = [line for line in eval_lines if line.startswith("RGB_ABLATION_VIDEOS=")][-1]
vals["MIDPROBE"] = "\n".join(midprobe["lines"] + midprobe["notes"])
vals["NUM_MIDPROBE"] = str(len(midprobe["selected"]))
text = src.read_text()
for k, v in vals.items():
    text = text.replace(f"@@{k}@@", v)
if "@@" in text:
    sys.exit("错误: README 仍有未填占位 " + str(sorted({t for t in text.split("@@")[1::2]})))
(stage / "README.md").write_text(text)
print(f"README_RENDERED ckpts={len(ck)} epochs={len(rows)} mp4={nmp4} reports={nrep}")
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
EXPECT_FILES=$((EXPECT_CKPTS + ${#ROOT_WHITELIST[@]} + NMP4 + NREP + 2))   # +README +SHA256SUMS.pre
echo "LOCAL_FILES=$LOCAL_FILES LOCAL_BYTES=$LOCAL_BYTES PRE_LINES=$PRE_LINES"
[ "$LOCAL_FILES" = "$EXPECT_FILES" ] || { echo "错误: stage 文件数应为 $EXPECT_FILES，实为 $LOCAL_FILES"; exit 1; }
[ "$((PRE_LINES + 1))" = "$LOCAL_FILES" ] || { echo "错误: 清单行数+1 != stage 文件数"; exit 1; }
echo "阶段2完成"

# ---------------------------------------------------------------- 阶段 3：公开前体检
echo "阶段3开始：公开前体检（扫 stage，在建 bucket 之前）"
uv run --no-sync python scripts/dataset/hf_export/scan_hygiene.py \
  --root "$STAGE" --scope stage --allowlist "$ALLOWLIST" \
  --report "$EXPORT_ROOT/logs/hygiene-stage.json"
echo "阶段3完成"

# 重跑必须沿用相同评估产物，避免旧装箱计划和批标记漏掉后来补齐的报告。
EVAL_SHA=$(awk '$2 ~ /^eval-fourarm\//' "$STAGE/SHA256SUMS.pre.txt" | sha256sum | cut -d' ' -f1)
if [ -f "$MP4_PLAN" ]; then
  [ "$(cat "$MP4_PLAN.inputs.sha256" 2>/dev/null || true)" = "$EVAL_SHA" ] \
    || { echo "EVAL_STAGE_CHANGED 评估产物与原装箱计划不同，请核对旧计划和批标记"; exit 1; }
fi

if [ "$SMOKE" = 1 ]; then echo "SMOKE=PASS（到阶段 3 为止，未碰网络与凭据）"; exit 0; fi

# ---------------------------------------------------------------- 阶段 4：凭据 + 建 bucket（公开）
echo "阶段4开始：凭据与身份 + 创建公开 bucket $BUCKET_ID"
if { [ -e "$EXPORT_ROOT/logs/uploaded.marker" ] || [ -L "$EXPORT_ROOT/logs/uploaded.marker" ]; } && ! upload_marker check; then
  echo "UPLOAD_MARKER_MISMATCH 本地标记无效或属于其他 run/bucket，停止"; exit 1
fi
[ -f "$REPO/v1-store/secrets/hf.env" ] || { echo "错误: 缺少 v1-store/secrets/hf.env"; exit 1; }
set -a; . "$REPO/v1-store/secrets/hf.env"; set +a
[ -n "${HF_TOKEN:-}" ] || { echo "错误: hf.env 未提供 HF_TOKEN"; exit 1; }
command -v uvx >/dev/null || { echo "错误: PATH 里找不到 uvx。PATH=$PATH"; exit 1; }
WHO="$(hf auth whoami --format agent 2>&1 | tr '\n' ' ' | tr -s ' ')"
echo "  whoami: $WHO"
case "$WHO" in *HongzeFu*) : ;; *) echo "错误: 身份不是 HongzeFu（$WHO）"; exit 1 ;; esac
echo "HF_WHOAMI=HongzeFu"
# 先核归属/非空，确认可以操作后才设置公开；查询失败不等同于 bucket 不存在。
if pre_files="$(bucket_stat totalFiles)"; then
  [[ "$pre_files" =~ ^[0-9]+$ ]] || { echo "错误: bucket 文件数无效"; exit 1; }
  if [ "$pre_files" != 0 ] && ! upload_marker check; then
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
echo "BUCKET_FILES_AT_START=$(bucket_stat totalFiles) BUCKET_BYTES_AT_START=$(bucket_stat size)"
echo "阶段4完成"

# ---------------------------------------------------------------- 阶段 5：分批上传
echo "阶段5开始：30 个 ckpt 逐件核 SHA + eval-fourarm/ 按装箱计划 + 根下小文件"
sync_retry() {  # $1 = 标签；其余 = hf sync 参数
  local tag="$1"; shift
  local ok=0 attempt
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    if hf sync "$@"; then ok=1; break; fi
    echo "  批 [$tag] 第 $attempt/$MAX_ATTEMPTS 次尝试中断"
    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then echo "  60s 后重试续传"; sleep 60; fi
  done
  [ "$ok" = 1 ] || { echo "UPLOAD_FAIL object=$tag sync=$* attempts=$MAX_ATTEMPTS"; echo "错误: 批 [$tag] 首试 + 重试 $UPLOAD_RETRIES 次仍失败"; exit 1; }
}
# 与增量脚本共用逐 epoch 标记；任何已传件字节变化都停止，不能静默跳过。
for epoch in $(seq 2 2 60); do
  name="checkpoint_epoch_$epoch.pt"
  digest="$(sha256sum "$STAGE/$name" | cut -d' ' -f1)"
  DONE="$EXPORT_ROOT/logs/uploaded/ckpt-epoch-$epoch.done"
  if [ -f "$DONE" ]; then
    [ "$(cat "$DONE")" = "$digest" ] || { echo "INC_SHA_MISMATCH epoch=$epoch"; exit 1; }
    echo "INC_SHA_MATCH epoch=$epoch 跳过已传件"
    continue
  fi
  ok=0
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    if hf buckets cp "$STAGE/$name" "$BUCKET/$name"; then ok=1; break; fi
    [ "$attempt" -ge "$MAX_ATTEMPTS" ] || sleep 60
  done
  [ "$ok" = 1 ] || { echo "UPLOAD_FAIL object=$name attempts=$MAX_ATTEMPTS"; exit 1; }
  printf '%s\n' "$digest" > "$DONE.tmp"; mv "$DONE.tmp" "$DONE"
done

# eval-fourarm/：装箱计划落盘、幂等读盘（每批 ≤ 8 GiB、≤ 64 件）
if [ ! -f "$MP4_PLAN" ]; then
  uv run --no-sync python scripts/dataset/hf_export/plan_upload_batches.py \
    --stage "$STAGE/eval-fourarm" --out "$MP4_PLAN" --max-files 64
  printf '%s\n' "$EVAL_SHA" > "$MP4_PLAN.inputs.sha256"
fi
uv run --no-sync python - "$MP4_PLAN" "$STAGE/eval-fourarm" <<'PY'
import json, sys
from pathlib import Path
plan, root = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
seen = []
for batch in plan["batches"]:
    files = [root / batch["sub"] / name for name in batch["files"]]
    if len(files) > 64 or sum(path.stat().st_size for path in files) > 8 * 1024**3:
        sys.exit("错误: 评估装箱批次超过 64 件或 8 GiB")
    seen.extend(str(path.relative_to(root)) for path in files)
expected = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()]
if plan["root_files"] or len(seen) != len(set(seen)) or sorted(seen) != sorted(expected):
    sys.exit("错误: 评估装箱计划没有恰好覆盖全部文件")
PY
NMB=$(uv run --no-sync python -c "import json;print(len(json.load(open('$MP4_PLAN'))['batches']))")
[ "$(uv run --no-sync python -c "import json;print(len(json.load(open('$MP4_PLAN'))['root_files']))")" = 0 ] \
  || { echo "错误: eval-fourarm/ 根下不应有平文件"; exit 1; }
for ((b = 0; b < NMB; b++)); do
  BID=$(uv run --no-sync python -c "import json;print(json.load(open('$MP4_PLAN'))['batches'][$b]['id'])")
  DONE="$EXPORT_ROOT/logs/uploaded/eval-fourarm-$BID.done"
  if [ -f "$DONE" ]; then
    [ "$(cat "$DONE")" = "$EVAL_SHA" ] || { echo "EVAL_STAGE_CHANGED 批标记身份不符 $BID"; exit 1; }
    echo "  eval-fourarm [$((b+1))/$NMB] $BID 已完成，跳过"; continue
  fi
  SUB=$(uv run --no-sync python -c "import json;print(json.load(open('$MP4_PLAN'))['batches'][$b]['sub'])")
  mapfile -t FILES < <(uv run --no-sync python -c "
import json
for f in json.load(open('$MP4_PLAN'))['batches'][$b]['files']: print(f)
")
  INC=(); for f in "${FILES[@]}"; do INC+=(--include "$f"); done
  echo "  eval-fourarm [$((b+1))/$NMB] $BID  ${#FILES[@]} 件"
  sync_retry "eval-fourarm-$BID" "$STAGE/eval-fourarm/$SUB" "$BUCKET/eval-fourarm/$SUB" "${INC[@]}"
  printf '%s\n' "$EVAL_SHA" > "$DONE"
done

mapfile -t ROOTFILES < <(cd "$STAGE" && find . -maxdepth 1 -type f ! -name 'checkpoint_epoch_*.pt' -printf '%P\n' | sort)
echo "  根下平文件 ${#ROOTFILES[@]} 个"
for f in "${ROOTFILES[@]}"; do
  ok=0
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    if hf buckets cp "$STAGE/$f" "$BUCKET/$f"; then ok=1; break; fi
    echo "  $f 第 $attempt/$MAX_ATTEMPTS 次尝试中断"
    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then echo "  60s 后重试"; sleep 60; fi
  done
  [ "$ok" = 1 ] || { echo "UPLOAD_FAIL object=$f attempts=$MAX_ATTEMPTS"; echo "错误: 根文件 $f 首试 + 重试 $UPLOAD_RETRIES 次仍失败"; exit 1; }
done
echo "UPLOAD_DONE ckpts=$EXPECT_CKPTS mp4_batches=$NMB"
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
# 回读是真实复制（stage 是硬链接、不占额外空间，verify 不是）。需要量 = stage 总字节 − verify 已有字节
# （续跑时不重复计），另留 0.4 TB（G12）。写法照抄 run_motionjepa_mixed8_200_export.sh 阶段 8。
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
echo "阶段8完成"

# ---------------------------------------------------------------- 阶段 9：源侧第三遍
echo "阶段9开始：源侧第三遍 sha256（stage 是源文件的硬链接，重算 stage 即重算源）"
# 退回复制的条目（跨设备时）重算 stage 只证明副本没变；逐条再对一次源文件，覆盖两种情形
hash_tree "$STAGE" "$TMPD/SHA256SUMS.src-after.txt"
cp "$TMPD/SHA256SUMS.src-after.txt" "$EXPORT_ROOT/SHA256SUMS.src-after.txt"
diff "$STAGE/SHA256SUMS.pre.txt" "$TMPD/SHA256SUMS.src-after.txt" \
  || { echo "错误: stage 侧 sha256 与上传前不一致"; exit 1; }
while read -r digest rel; do
  case "$rel" in
    README.md) continue ;;
    train.log) continue ;;      # 现场清洗生成，不是源文件的链接；源日志在循环后单独复核
    eval-fourarm/videos/*)
      name="${rel#eval-fourarm/videos/}"; stem="${name%.mp4}"; without_ablation="${stem%_*}"; arm="${without_ablation##*_}"
      case "$arm" in new|old|ref) : ;; *) echo "错误: 未知模型臂 $arm"; exit 1 ;; esac
      src="$COMPARE/videos/$arm/$name" ;;
    eval-fourarm/reports/result-*.json)
      arm="${rel#eval-fourarm/reports/result-}"; arm="${arm%.json}"; src="$COMPARE/$arm/result.json" ;;
    eval-fourarm/reports/midprobe-e*.json)
      name="${rel#eval-fourarm/reports/midprobe-}"; src="$MIDPROBE_DIR/$name" ;;
    eval-fourarm/reports/summary.json|eval-fourarm/reports/windows.json)
      src="$COMPARE/${rel#eval-fourarm/reports/}" ;;
    *) src="$SRC/$rel" ;;
  esac
  [ "$(sha256sum "$src" | cut -d' ' -f1)" = "$digest" ] || { echo "错误: 源文件 $src 与上传前清单不符"; exit 1; }
done < "$STAGE/SHA256SUMS.pre.txt"
clean_train_log | cmp -s - "$STAGE/train.log" \
  || { echo "错误: 源训练日志重新清洗后与上传的 train.log 不同（日志被改过）"; exit 1; }
echo "SRC_UNCHANGED=OK checked=$((PRE_LINES - 2)) train_log_reclean=OK"
echo "阶段9完成"

# ---------------------------------------------------------------- 阶段 10：公开性确认
echo "阶段10开始：公开性确认（private==false + 匿名读）"
assert_public "BUCKET_PUBLIC_FINAL"
echo "BUCKET_PUBLIC=True"
rm -rf "$TMPD/anon"; mkdir -p "$TMPD/anon"
if env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HF_HUB_OFFLINE HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME="$TMPD/anon-hf-home" \
     uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf \
     sync "$BUCKET" "$TMPD/anon" --include 'README.md' \
   && [ -s "$TMPD/anon/README.md" ]; then
  echo "PUBLIC_ANON_READ=OK"
else
  echo "错误: 匿名读取失败，bucket 可能并未真正公开"; exit 1
fi
echo "阶段10完成"
echo "RESULT=PASS"
echo "全部完成"
