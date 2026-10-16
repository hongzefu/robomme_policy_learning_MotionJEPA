#!/usr/bin/env bash
# driver：把 MotionJEPA encoder 训练 run wan-mixed8-200-filter2-stop075-b88x8-72ep-a 的全部 36 个 epoch
# checkpoint（epoch 2…72，每 2 epoch 一存，EMA-only）、config、逐 epoch 指标、清洗后的训练日志，以及
# 200 集留出集可视化（eval-mp4/：mp4 与 reports/）上传到**公开** HF bucket
# HongzeFu/motionjepa-wan-mixed8-200-72ep-v1，做「上传前 / 上传后」两遍独立 sha256 逐行对照验收。
#
# 计划：1016-mixed8-vla-plan.html 3.1 表 A1b 行、口径 29（内容）/ 30（bucket 名）、G12、runbook 长尾
# m8v-encoder-ckpt-upload（闸门：mixed8-encoder-train done，且 DELIVERY=PASS）。
#
# 蓝本 run_motionjepa_encoder_full1600_export.sh（上一代 encoder 的 private 备份，已 PASS），骨架照抄。差别：
#   1. **公开**：建 bucket 不带私有开关（蓝本写死了私有），建后与收尾两次断言 private==false（口径 21、H7）。
#   2. **公开前体检**：蓝本是 private、没有体检；本驱动在建 bucket 之前对 stage 跑 scan_hygiene.py。
#   3. **stage 是显式白名单**，不是整目录硬链接：36 个 ckpt + config.yaml + 三份指标 + train.log +
#      eval-mp4/{videos,reports}。run 目录里的 wandb/ 刻意不传——其 wandb-metadata.json 带私有仓库的
#      GitHub remote（P6）与本机解释器路径，且计划口径 29 的 encoder 清单不含 wandb；逐 epoch 指标已在
#      train_metrics_epoch.jsonl。run 目录根下出现白名单外的任何其他条目，驱动停下交人裁决，不静默漏传。
#   4. **eval-mp4/**（口径 29「200 集 mp4 可视化 传」）：取交付目录的 output/videos/*.mp4（恰 200 个）与
#      reports/ 下的文件，按 plan_upload_batches.py 落盘装箱分批上传。
#   5. **交叉锚点**：epoch 72 的 sha256 必须等于 VLA 侧拷入件 v1-store/external/motionjepa/<run>/checkpoint_epoch_72.pt
#      （B1 拷入并记 sha256 的那一份，motion 表正是用它算的）。
#   6. **回读前 df 余量断言**：照抄 run_motionjepa_mixed8_200_export.sh 阶段 8 的 READBACK_NEED / DF_MARGIN 写法，
#      余量取本计划阈值 0.4 TB（G12）。
#   7. 完成信号是末尾的「全部完成」；RESULT=PASS 只是中途的内容层判定行。
#
# 用法（正式运行在主检出）：
#   SMOKE=1 bash scripts/dataset/hf_export/run_mixed8_200_encoder_ckpt_export.sh  # 预检 + stage + pre sha + 体检，不碰网络
#   bash scripts/dataset/hf_export/run_mixed8_200_encoder_ckpt_export.sh          # 全量
# 路径覆盖（只为离线 dry-run / 单测；正式运行一律不设）：M8X_SRC、M8X_TRAIN_LOG、M8X_MJ、M8X_DELIVERY、
# M8X_ENC_COPY、M8X_VIZ_LOG、M8X_EXPORT_ROOT。设了哪个就打印一行 OVERRIDE；全量模式下 EXPORT_ROOT 仍必须落在 v1-store/exports/ 下。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
cd "$REPO"

RUN_NAME="wan-mixed8-200-filter2-stop075-b88x8-72ep-a"
BUCKET_ID="HongzeFu/motionjepa-wan-mixed8-200-72ep-v1"
BUCKET="hf://buckets/$BUCKET_ID"
SRC="${M8X_SRC:-/scratch/hongze/MotionJEPA/runs/$RUN_NAME}"
# 训练原始日志（tmux tee 落盘的那份，B1_GATE 读的也是它）。stage 里的 train.log 由本驱动现场清洗生成：
# 去掉 tqdm 中间态（AGENTS 第 12 条 (3) 的同一条命令），保留 Epoch 汇总行、横幅与 EXIT_CODE= 尾行。
TRAIN_LOG="${M8X_TRAIN_LOG:-/scratch/hongze/logs/$RUN_NAME.train.log}"
MJ="${M8X_MJ:-/scratch/hongze/MotionJEPA}"     # 只借它的 .venv 读 ckpt 元数据（torch），不写
DELIVERY="${M8X_DELIVERY:-/scratch/hongze/eval-deliveries/motionjepa-mixed8-200-e72}"
VIZ_LOG="${M8X_VIZ_LOG:-/scratch/hongze/logs/m8v-viz.log}"    # runbook m8v-viz 的 tee 日志（DELIVERY 判定行在这里）
DELIVERY_LINE="DELIVERY=PASS episodes=200 cells=100 per_cell=2 ckpt=epoch_72"
ENC_COPY="${M8X_ENC_COPY:-$REPO/v1-store/external/motionjepa/$RUN_NAME/checkpoint_epoch_72.pt}"
EXPORT_ROOT="${M8X_EXPORT_ROOT:-$REPO/v1-store/exports/hf-mixed8-200-encoder-ckpt}"
ALLOWLIST="$REPO/scripts/dataset/hf_export/hygiene_allowlist_mixed8_200_encoder_ckpt.json"
README_SRC="$REPO/scripts/dataset/hf_export/mixed8_200_encoder_ckpt_bucket_README.md"
STAGE="$EXPORT_ROOT/stage"
VERIFY="$EXPORT_ROOT/verify"
TMPD="$EXPORT_ROOT/tmp"
MP4_PLAN="$EXPORT_ROOT/logs/eval-mp4_upload_plan.json"
BATCH_SIZE=8          # 每批 8 个 ckpt ≈ 3.8 GB
EXPECT_CKPTS=36
EXPECT_MP4=200
ROOT_WHITELIST=(config.yaml train_metrics_epoch.jsonl sig_loss_epoch1.csv gpu_samples.csv train.log)
DF_MARGIN_BYTES=400000000000          # 0.4 TB（十进制，G12）
NPROC=8
SMOKE="${SMOKE:-0}"

for v in M8X_SRC M8X_TRAIN_LOG M8X_MJ M8X_DELIVERY M8X_ENC_COPY M8X_VIZ_LOG M8X_EXPORT_ROOT; do
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

# 硬链接要求同设备；不同设备时退回复制（只有 ckpt 量级 17 GB，可接受），并在日志里写明
link_or_copy() {  # $1 = 源, $2 = 目标
  mkdir -p "$(dirname "$2")"
  ln -f "$1" "$2" 2>/dev/null || cp -f "$1" "$2"
}

# ---------------------------------------------------------------- 阶段 0：预检
echo "阶段0开始：预检（源 run 已收尾、交付已 PASS、交叉锚点、白名单外条目）"
for d in "$(dirname "$EXPORT_ROOT")" "$EXPORT_ROOT"; do
  [ -L "$d" ] && { echo "错误: $d 是符号链接，拒绝写入"; exit 1; }
done
[ -d "$SRC" ] || { echo "错误: 源 run 目录不存在 $SRC"; exit 1; }
[ -d "$DELIVERY/output/videos" ] || { echo "错误: 交付目录缺 output/videos/：$DELIVERY"; exit 1; }
[ -d "$DELIVERY/reports" ] || { echo "错误: 交付目录缺 reports/：$DELIVERY"; exit 1; }
[ -f "$ALLOWLIST" ] && [ -f "$README_SRC" ] || { echo "错误: 缺体检账本或 README 模板"; exit 1; }
mkdir -p "$STAGE" "$VERIFY" "$TMPD" "$EXPORT_ROOT/logs" "$EXPORT_ROOT/logs/uploaded"

# 这条 run 必须已经跑完：判据与 runbook 的 B1_GATE 逐字相同（末行 EXIT_CODE=0、清洗后 72 行 Epoch）
[ -f "$TRAIN_LOG" ] || { echo "错误: 训练日志不存在 $TRAIN_LOG"; exit 1; }
LAST_LINE="$(tail -n 1 "$TRAIN_LOG")"
[ "$LAST_LINE" = "EXIT_CODE=0" ] || { echo "错误: $TRAIN_LOG 末行为「$LAST_LINE」，不是 EXIT_CODE=0"; exit 1; }
NEPOCH_LINES=$(tr '\r' '\n' < "$TRAIN_LOG" | grep -vE '%\|' | grep -c '^Epoch ' || true)
echo "  训练日志: 末行 $LAST_LINE，清洗后 Epoch 行 $NEPOCH_LINES 条"
[ "$NEPOCH_LINES" = 72 ] || { echo "错误: 清洗后 Epoch 行应为 72，实为 $NEPOCH_LINES"; exit 1; }

# 交付必须已经 PASS（runbook 闸门：viz 日志含原样判定行且末行 EXIT_CODE=0），且恰 200 个 mp4
[ -f "$VIZ_LOG" ] || { echo "错误: 缺 viz 日志 $VIZ_LOG"; exit 1; }
grep -qxF "$DELIVERY_LINE" "$VIZ_LOG" || { echo "错误: $VIZ_LOG 里没有「$DELIVERY_LINE」"; exit 1; }
[ "$(tail -n 1 "$VIZ_LOG")" = "EXIT_CODE=0" ] || { echo "错误: $VIZ_LOG 末行不是 EXIT_CODE=0"; exit 1; }
echo "  $DELIVERY_LINE（$VIZ_LOG）"
NMP4=$(find "$DELIVERY/output/videos" -maxdepth 1 -type f -name '*.mp4' | wc -l)
[ "$NMP4" = "$EXPECT_MP4" ] || { echo "错误: 交付 mp4 应为 $EXPECT_MP4 个，实为 $NMP4"; exit 1; }
if find "$DELIVERY/reports" -mindepth 1 -type d -print -quit | grep -q .; then
  echo "错误: reports/ 下有子目录，本驱动只认平铺文件，先核对交付结构"; exit 1
fi
NREP=$(find "$DELIVERY/reports" -maxdepth 1 -type f | wc -l)
[ "$NREP" -ge 1 ] || { echo "错误: reports/ 为空"; exit 1; }

# run 目录根：除 36 个 ckpt、白名单文件与刻意不传的 wandb/ 之外不得有别的条目
mapfile -t CKPT_NAMES < <(for e in $(seq 2 2 72); do echo "checkpoint_epoch_$e.pt"; done)
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
[ "$NMETRIC" = 72 ] || { echo "错误: train_metrics_epoch.jsonl 应为 72 行，实为 $NMETRIC"; exit 1; }

# 交叉锚点：epoch 72 必须与 VLA 侧拷入件逐字节同 sha256
[ -f "$ENC_COPY" ] || { echo "错误: 缺 VLA 侧拷入件 $ENC_COPY（B1 未完成？）"; exit 1; }
E72_SRC="$(sha256sum "$SRC/checkpoint_epoch_72.pt" | cut -d' ' -f1)"
E72_COPY="$(sha256sum "$ENC_COPY" | cut -d' ' -f1)"
[ "$E72_SRC" = "$E72_COPY" ] || { echo "错误: epoch 72 与拷入件 sha256 不同（$E72_SRC vs $E72_COPY）"; exit 1; }
echo "ENCODER_MATCHES_VLA_COPY=OK $E72_SRC"

# README 写「EMA-only，没有 live 权重与优化器状态」：36 个 ckpt 逐个用 B1 同一判据核实，不只核 epoch 72
[ -x "$MJ/.venv/bin/python" ] || { echo "错误: 找不到 $MJ/.venv/bin/python（读 ckpt 元数据要 torch）"; exit 1; }
"$MJ/.venv/bin/python" -I - "$SRC" "${CKPT_NAMES[@]}" <<'PY'
import sys, torch
src, names = sys.argv[1], sys.argv[2:]
bad = []
for n in names:
    c = torch.load(f"{src}/{n}", map_location="cpu", weights_only=False)
    if not (c.get("ema_only") is True and c.get("has_live_weights") is False
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
# 清洗：同一条命令（AGENTS 第 12 条 (3)），清洗后再核一次 72 行 Epoch 与 EXIT_CODE=0 尾行零丢失
tr '\r' '\n' < "$TRAIN_LOG" | grep -vE '%\|' > "$STAGE/train.log"
[ "$(grep -c '^Epoch ' "$STAGE/train.log")" = 72 ] && [ "$(tail -n 1 "$STAGE/train.log")" = "EXIT_CODE=0" ] \
  || { echo "错误: 清洗后的 train.log 丢了 Epoch 行或 EXIT_CODE 尾行"; exit 1; }
echo "  train.log 清洗: $(wc -c < "$TRAIN_LOG") B → $(wc -c < "$STAGE/train.log") B"
while IFS= read -r f; do
  link_or_copy "$DELIVERY/output/videos/$f" "$STAGE/eval-mp4/videos/$f"
done < <(cd "$DELIVERY/output/videos" && find . -maxdepth 1 -type f -name '*.mp4' -printf '%P\n' | sort)
while IFS= read -r f; do
  link_or_copy "$DELIVERY/reports/$f" "$STAGE/eval-mp4/reports/$f"
done < <(cd "$DELIVERY/reports" && find . -maxdepth 1 -type f -printf '%P\n' | sort)

# README：把锚点 sha 与实测数字现填进模板（不经人手），填完断言没有残留占位
uv run --no-sync python - "$README_SRC" "$STAGE" "$E72_SRC" "$NMP4" "$NREP" <<'PY'
import json, sys
from pathlib import Path
src, stage, e72, nmp4, nrep = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
rows = [json.loads(l) for l in (stage / "train_metrics_epoch.jsonl").read_text().splitlines() if l.strip()]
ck = sorted(stage.glob("checkpoint_epoch_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
sizes = sorted({p.stat().st_size for p in ck})
vals = {
    "E72_SHA": e72,
    "NUM_CKPT": str(len(ck)),
    "CKPT_BYTES_TOTAL": f"{sum(p.stat().st_size for p in ck):,}",
    "CKPT_SIZES": " / ".join(f"{s:,}" for s in sizes),
    "NUM_EPOCH_ROWS": str(len(rows)),
    "NUM_MP4": nmp4,
    "NUM_REPORTS": nrep,
}
if [r["epoch"] for r in rows] != list(range(1, 73)):
    sys.exit("错误: train_metrics_epoch.jsonl 的 epoch 列不是 1..72")
for tag, r in (("E1", rows[0]), ("E72", rows[-1])):
    vals[f"{tag}_TOTAL"] = f"{r['loss_total']:.8f}"
    vals[f"{tag}_LATENT"] = f"{r['loss_raw/latent']:.8f}"
    vals[f"{tag}_SIGREG"] = f"{r['loss_raw/sigreg']:.5f}"
    vals[f"{tag}_STD"] = f"{r['motion/std_batch']:.5f}"
    vals[f"{tag}_COS"] = f"{r['motion/cos_cross_chunk']:.7f}"
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

# ---------------------------------------------------------------- 阶段 5：分批上传
echo "阶段5开始：36 个 ckpt 按 $BATCH_SIZE 个一批 + eval-mp4/ 按装箱计划 + 根下小文件"
touch "$EXPORT_ROOT/logs/uploaded.marker"
sync_retry() {  # $1 = 标签；其余 = hf sync 参数
  local tag="$1"; shift
  local ok=0 attempt
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf sync "$@"; then ok=1; break; fi
    echo "  批 [$tag] 第 $attempt 次中断，60s 后重试续传"; sleep 60
  done
  [ "$ok" = 1 ] || { echo "错误: 批 [$tag] 八次仍失败"; exit 1; }
}
# 分组稳定性来自「文件名按 epoch 数值排序」这一确定性；每批字节上界卡在 16 GiB（实测 20 GB 会挂）
MAXSZ=$(cd "$STAGE" && printf '%s\n' "${CKPT_NAMES[@]}" | xargs stat -c %s | sort -n | tail -1)
[ $((MAXSZ * BATCH_SIZE)) -le $((16 * 1024 * 1024 * 1024)) ] || { echo "错误: 每批上界超过 16 GiB，调小 BATCH_SIZE"; exit 1; }
NBATCH=$(( (EXPECT_CKPTS + BATCH_SIZE - 1) / BATCH_SIZE ))
for ((b = 0; b < NBATCH; b++)); do
  DONE="$EXPORT_ROOT/logs/uploaded/ckpt-batch-$b.done"
  if [ -f "$DONE" ]; then echo "  [$((b+1))/$NBATCH] 已完成，跳过"; continue; fi
  GROUP=( "${CKPT_NAMES[@]:$((b * BATCH_SIZE)):$BATCH_SIZE}" )
  INC=(); for g in "${GROUP[@]}"; do INC+=(--include "$g"); done
  echo "  [$((b+1))/$NBATCH] ${#GROUP[@]} 个: ${GROUP[*]}"
  sync_retry "ckpt-$b" "$STAGE" "$BUCKET" "${INC[@]}"
  touch "$DONE"
done

# eval-mp4/：装箱计划落盘、幂等读盘（每批 ≤ 8 GiB、≤ 64 件）
if [ ! -f "$MP4_PLAN" ]; then
  uv run --no-sync python scripts/dataset/hf_export/plan_upload_batches.py \
    --stage "$STAGE/eval-mp4" --out "$MP4_PLAN" --max-files 64
fi
NMB=$(python3 -c "import json;print(len(json.load(open('$MP4_PLAN'))['batches']))")
[ "$(python3 -c "import json;print(len(json.load(open('$MP4_PLAN'))['root_files']))")" = 0 ] \
  || { echo "错误: eval-mp4/ 根下不应有平文件"; exit 1; }
for ((b = 0; b < NMB; b++)); do
  BID=$(python3 -c "import json;print(json.load(open('$MP4_PLAN'))['batches'][$b]['id'])")
  DONE="$EXPORT_ROOT/logs/uploaded/eval-mp4-$BID.done"
  if [ -f "$DONE" ]; then echo "  eval-mp4 [$((b+1))/$NMB] $BID 已完成，跳过"; continue; fi
  SUB=$(python3 -c "import json;print(json.load(open('$MP4_PLAN'))['batches'][$b]['sub'])")
  mapfile -t FILES < <(python3 -c "
import json
for f in json.load(open('$MP4_PLAN'))['batches'][$b]['files']: print(f)
")
  INC=(); for f in "${FILES[@]}"; do INC+=(--include "$f"); done
  echo "  eval-mp4 [$((b+1))/$NMB] $BID  ${#FILES[@]} 件"
  sync_retry "eval-mp4-$BID" "$STAGE/eval-mp4/$SUB" "$BUCKET/eval-mp4/$SUB" "${INC[@]}"
  touch "$DONE"
done

mapfile -t ROOTFILES < <(cd "$STAGE" && find . -maxdepth 1 -type f ! -name 'checkpoint_epoch_*.pt' -printf '%P\n' | sort)
echo "  根下平文件 ${#ROOTFILES[@]} 个"
for f in "${ROOTFILES[@]}"; do
  ok=0
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf buckets cp "$STAGE/$f" "$BUCKET/$f"; then ok=1; break; fi
    echo "  $f 第 $attempt 次中断，60s 后重试"; sleep 60
  done
  [ "$ok" = 1 ] || { echo "错误: $f 八次仍失败"; exit 1; }
done
echo "UPLOAD_DONE ckpt_batches=$NBATCH mp4_batches=$NMB"
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
for attempt in 1 2 3; do
  if hf sync "$BUCKET" "$VERIFY"; then ok=1; break; fi
  echo "回读第 $attempt 次中断，60s 后重试续传"; sleep 60
done
[ "$ok" = 1 ] || { echo "回读三次仍失败"; exit 1; }
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
POST_E72="$(grep '  checkpoint_epoch_72\.pt$' "$TMPD/SHA256SUMS.post.txt" | cut -d' ' -f1)"
[ "$POST_E72" = "$E72_SRC" ] || { echo "错误: 回读的 epoch 72 sha256 与拷入件不符"; exit 1; }
echo "ENCODER_READBACK_MATCHES_VLA_COPY=OK $POST_E72"
echo "RESULT=PASS"
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
    eval-mp4/videos/*) src="$DELIVERY/output/videos/${rel#eval-mp4/videos/}" ;;
    eval-mp4/reports/*) src="$DELIVERY/reports/${rel#eval-mp4/reports/}" ;;
    *) src="$SRC/$rel" ;;
  esac
  [ "$(sha256sum "$src" | cut -d' ' -f1)" = "$digest" ] || { echo "错误: 源文件 $src 与上传前清单不符"; exit 1; }
done < "$STAGE/SHA256SUMS.pre.txt"
tr '\r' '\n' < "$TRAIN_LOG" | grep -vE '%\|' | cmp -s - "$STAGE/train.log" \
  || { echo "错误: 源训练日志重新清洗后与上传的 train.log 不同（日志被改过）"; exit 1; }
echo "SRC_UNCHANGED=OK checked=$((PRE_LINES - 2)) train_log_reclean=OK"
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
