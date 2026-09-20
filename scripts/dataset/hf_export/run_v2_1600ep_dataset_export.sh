#!/usr/bin/env bash
# driver：把这一版训练库 v1-store/datasets/4task-v2-1600ep-604f16da（BinFill / RouteStick /
# VideoRepick / VideoUnmaskSwap，669 GiB / 63.2 万文件）上传到**公开** HF bucket
# HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1，做「上传前 / 上传后」两遍独立
# sha256 逐行对照验收，末尾再用回读副本跑一次 20 步冷启动彩排。
# detached tmux 里跑，每阶段幂等、整脚本可重跑续传。
#
# 与同目录三条老链路的关系：
#   run_dataset400ep_export.sh  —— 阶段骨架、公开性体检闸门、tar 成员抽样、源侧第三遍
#     全部照抄它（上一版同类数据集，122 对象 / 130 GB，已 PASS）。
#   run_motionjepa_full1600_export.sh —— 分批改用它那套 plan_upload_batches.py 落盘装箱
#     （400ep 是手写 25 批；源目录一变贪心分组就漂移，--include 模式随之失配）。
#   pack_and_hash.py —— 共用，本轮新增 layout v2-1600ep-v1。
#
# 本脚本相对 400ep 那条多出来的三件事：
#   1. **阶段 0 腾盘**：本机 /scratch 已用 93%，打 tar 305 GiB + 回读 669 GiB 的峰值放不下。
#      删掉 v1-store/exports/*/verify/（都是已验收 PASS 的回读副本，可从对应 bucket 重下）。
#   2. **source/features 只带 8 个文件**：整目录 674 GiB 训练不读，但 framesamp-8x8 的
#      source_spot_sha256 有 8 条抽样点落在 features/ 下，少一个异地就起不来（见 layout）。
#   3. **阶段 11 冷启动彩排**：字节层全过只证明「传对了」，证明不了「下完能训」。
#      用回读副本真跑 20 步，两条 config 各一次。
#
# 用法：
#   bash scripts/dataset/hf_export/run_v2_1600ep_dataset_export.sh          # 全量
#   SMOKE=1 bash scripts/dataset/hf_export/run_v2_1600ep_dataset_export.sh  # 只跑阶段 -1
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
cd "$REPO"

LIB="$REPO/v1-store/datasets/4task-v2-1600ep-604f16da"
NORM_STATS="$REPO/v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json"
RUNNER_SRC="$REPO/scripts/dataset/hf_export/v2_1600ep_runner"
ALLOWLIST="scripts/dataset/hf_export/hygiene_allowlist_v2_1600ep.json"
BUCKET_ID="HongzeFu/robomme-4task-v2-1600ep-trainset-20260920-v1"
BUCKET="hf://buckets/$BUCKET_ID"
EXPORT_ROOT="$REPO/v1-store/exports/hf-v2-1600ep-trainset"
STAGE="$EXPORT_ROOT/stage"
VERIFY="$EXPORT_ROOT/verify"
TMPD="$EXPORT_ROOT/tmp"
PLAN="$EXPORT_ROOT/logs/upload_plan.json"
NPROC=16
SMOKE="${SMOKE:-0}"

# AGENTS 第 14 条：缓存类环境变量逐项指向 v1-store / scratch，**禁止覆盖 HOME**
export HF_HOME="$REPO/v1-store/cache/hf"
export HF_XET_CACHE="$REPO/v1-store/cache/hf-xet"
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export UV_PYTHON_INSTALL_DIR=/scratch/hongze/.cache/uv-python
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1
# scripts/dataset/paths.sh 会 export HF_HUB_OFFLINE=1。本脚本不 source 它，但上游 shell
# 可能带进来——不 unset 的话所有网络操作会静默走缓存，看起来"成功"实际什么都没传。
unset HF_HUB_OFFLINE || true

# 凭据：只从 v1-store/secrets/hf.env 取（600、在 .gitignore 的 /v1-store/ 下）。
# 环境里原有的 HF_TOKEN 属于 yinpei-tri，写不进 HongzeFu 命名空间，必须被这里覆盖掉。
[ -f "$REPO/v1-store/secrets/hf.env" ] || { echo "错误: 缺少 v1-store/secrets/hf.env"; exit 1; }
set -a; . "$REPO/v1-store/secrets/hf.env"; set +a
[ -n "${HF_TOKEN:-}" ] || { echo "错误: hf.env 未提供 HF_TOKEN"; exit 1; }

# 项目 .venv 的 huggingface_hub 是 0.32.3（openpi 钉死、不能升，且无 bucket 子命令），
# 故用 uvx 拉一份完全独立的 1.30.0 CLI，只在本脚本内使用，pyproject/uv.lock/.venv 一律不碰。
hf() { uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf "$@"; }

# 精确字节/文件数/可见性只信 buckets REST API：`hf repos ls` 给人类可读值，且该类统计接口
# **异步滞后**（480 GB 那次传完一度显示 312 GB / 200 文件；87 G 那次首查 0/0）。
bucket_stat() {  # $1 = 字段名 size|totalFiles|private
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

# 在 \$1 目录内并行算 sha256，输出按路径排序（xargs -P 并行会乱序，不排序则无法 diff）。
# 必须并行：单进程 Python hashlib 只有 38 MB/s，669 GiB 要 5 小时；xargs -P16 实测 ~6.5 GB/s。
hash_tree() {  # $1 = 目录, $2 = 输出文件（绝对路径，不能落在被扫描目录内）
  ( cd "$1" && find . -type f -not -name 'SHA256SUMS.*' -printf '%P\n' | sort \
      | xargs -P "$NPROC" -n 16 sha256sum | sort -k 2 > "$2" )
}

mkdir -p "$STAGE" "$VERIFY" "$TMPD" "$EXPORT_ROOT/logs" "$HF_XET_CACHE"

# ---------------------------------------------------------------- 阶段 -1：smoke
if [ "$SMOKE" = 1 ]; then
  echo "阶段-1开始：smoke（本地打 2 个 data 片，跳过 64 个 part 大 bin，不碰网络）"
  uv run --no-sync python scripts/dataset/hf_export/pack_and_hash.py \
    --layout v2-1600ep-v1 --lib "$LIB" --stage-root "$EXPORT_ROOT" \
    --bucket "$BUCKET" --norm-stats "$NORM_STATS" --workers 8 --limit-shards 2

  cat "$EXPORT_ROOT/logs/parts/"*.sha256 > "$TMPD/smoke-members.txt"
  uv run --no-sync python scripts/dataset/hf_export/verify_tar_members.py \
    --verify-root "$STAGE" --checksums "$TMPD/smoke-members.txt" --sample-tars 2

  # 上一步的 sha 与 tar 出自同一遍读，自洽但不能证明与源一致。这一步把成员解出来
  # 跟源文件真比一次，才是「打包没改数」的独立证据。
  uv run --no-sync python - "$LIB" "$STAGE" <<'PY'
import random, sys, tarfile
from pathlib import Path
lib, stage = Path(sys.argv[1]), Path(sys.argv[2])
rng = random.Random(20260920)
bad, checked = [], 0
for t in sorted(stage.rglob("*.tar")):
    with tarfile.open(t, "r") as tf:
        names = [ti.name for ti in tf if ti.isfile()]
        for nm in rng.sample(names, min(10, len(names))):
            got = tf.extractfile(nm).read()
            checked += 1
            if got != (lib / "source" / nm).read_bytes():   # 成员名 data/N.pkl
                bad.append(nm)
print(f"SMOKE_BYTEWISE checked={checked} mismatches={len(bad)}")
if bad:
    print("差异成员:", bad[:10]); sys.exit(1)
PY
  echo "阶段-1完成：smoke 通过（产物由 pack_progress.jsonl 记账，全量跑时直接续用）"
  exit 0
fi

# ---------------------------------------------------------------- 阶段 0：腾盘 + 预检
echo "阶段0开始：腾盘 + 预检（此时 bucket 尚未创建）"
# 峰值需求：tar 实体 305 GiB + 回读 669 GiB ≈ 974 GiB。开工前把旧导出的 verify/ 删掉，
# 它们都是当时验收 PASS 之后留下的回读副本（400ep 的 result.md 明写「验收后可整目录删」），
# 真要复查可从对应 bucket 重下。stage/ 是硬链接，删了不回收空间，不动。
FREED=0
for d in "$REPO"/v1-store/exports/*/verify; do
  [ -d "$d" ] || continue
  case "$d" in "$VERIFY") continue ;; esac          # 本轮自己的回读目录不能删
  sz=$(du -sb "$d" | cut -f1)
  echo "  删除旧回读副本 $d（$((sz / 1024 / 1024 / 1024)) GiB）"
  rm -rf "$d"
  FREED=$((FREED + sz))
done
echo "FREED_BYTES=$FREED"
AVAIL=$(df -B1 --output=avail "$REPO" | tail -1)
echo "AVAIL_BYTES=$AVAIL"
# 974 GiB = 1045971107840；留 5% 余量后要求至少 1.0 TiB 可用
[ "$AVAIL" -ge 1045971107840 ] || { echo "错误: 可用空间 $AVAIL 不足 974 GiB，停止"; exit 1; }

# 落点必须是本环境实体目录（AGENTS 第 13/14 条：环境 B 的 v1-store 下没有任何 symlink 外链）
for d in "$REPO/v1-store/exports" "$EXPORT_ROOT"; do
  [ -L "$d" ] && { echo "错误: $d 是符号链接，拒绝写入"; exit 1; }
done
[ -d "$LIB" ] || { echo "错误: 源库不存在 $LIB"; exit 1; }
[ -f "$NORM_STATS" ] || { echo "错误: norm_stats 不存在 $NORM_STATS"; exit 1; }
[ -d "$RUNNER_SRC" ] || { echo "错误: runner 源目录不存在 $RUNNER_SRC"; exit 1; }
# pack.lock 是读侧硬闸且无逃生阀，带进包等于交付一个异地一加载就拒绝的库
for sub in framesamp-8x8 framesamp motion; do
  [ -e "$LIB/$sub/meta/pack.lock" ] && { echo "错误: $sub/meta/pack.lock 存在"; exit 1; }
done

command -v uvx >/dev/null || { echo "错误: PATH 里找不到 uvx。PATH=$PATH"; exit 1; }
# whoami 的输出格式随是否有 tty 而变（非 tty 单行 `user=X orgs=Y`，tty 下多行）。
# tmux 里有 tty，**不能按行取字段**，须整体合并后再子串匹配。
who="$(hf auth whoami 2>&1 | tr '\n' ' ' | tr -s ' ')"
echo "  whoami: $who"
case "$who" in *HongzeFu*) : ;; *) echo "错误: 身份不是 HongzeFu（$who）"; exit 1 ;; esac
echo "HF_WHOAMI=HongzeFu"
echo "阶段0完成"

# ---------------------------------------------------------------- 阶段 1：公开性体检（闸门）
echo "阶段1开始：公开性体检（★ 闸门：不过不建库）"
# 放在 create 之前：目标是公开 bucket，一旦建出来就没有「发现问题还来得及」的窗口。
uv run --no-sync python scripts/dataset/hf_export/scan_hygiene.py \
  --root "$LIB" --extra "$NORM_STATS" --scope source \
  --allowlist "$ALLOWLIST" --report "$EXPORT_ROOT/logs/hygiene-source.json"
echo "阶段1完成"

# ---------------------------------------------------------------- 阶段 2：建 bucket（公开）
echo "阶段2开始：创建公开 bucket $BUCKET_ID"
if bucket_stat totalFiles >/dev/null 2>&1; then
  echo "  bucket 已存在，跳过创建（幂等重跑）"
else
  hf buckets create "$BUCKET_ID"
fi
# 显式设一次 public 而不是依赖 create 的默认值（create 不带 --private 才是公开，
# 这一行让终态成为结构性成立的事实而不是推断）。settings 幂等。
hf buckets settings "$BUCKET_ID" --public
BUCKET_PRIVATE="$(bucket_stat private)"
echo "BUCKET_CREATED=$BUCKET_ID PRIVATE=$BUCKET_PRIVATE"
case "$BUCKET_PRIVATE" in
  False|false|None) : ;;
  *) echo "错误: bucket 仍非公开（private=$BUCKET_PRIVATE）"; exit 1 ;;
esac

# 绝不往一个已有内容的 bucket 上 sync：repo_id 打错一个字母就会静默污染别的库
pre_files="$(bucket_stat totalFiles)"; pre_size="$(bucket_stat size)"
echo "BUCKET_FILES_AT_START=$pre_files BUCKET_BYTES_AT_START=$pre_size"
if [ "$pre_files" != "0" ] && [ "$pre_files" != "None" ]; then
  if [ ! -f "$EXPORT_ROOT/logs/uploaded.marker" ]; then
    echo "错误: 目标 bucket 非空（files=$pre_files）且本地无上传记录，停止并交用户裁决"; exit 1
  fi
  echo "  bucket 非空但本地有上传记录，按续传处理"
fi
echo "阶段2完成"

# ---------------------------------------------------------------- 阶段 3：打包
echo "阶段3开始：打包 → $STAGE（断点续跑，已完成分片自动跳过）"
uv run --no-sync python scripts/dataset/hf_export/pack_and_hash.py \
  --layout v2-1600ep-v1 --lib "$LIB" --stage-root "$EXPORT_ROOT" \
  --bucket "$BUCKET" --norm-stats "$NORM_STATS" --workers 8
echo "阶段3完成"

# ---------------------------------------------------------------- 阶段 4：stage 收尾 + 上传前 sha256
echo "阶段4开始：README + runner + 断言 + 上传前 sha256 + stage 侧第二遍体检"
cp scripts/dataset/hf_export/v2_1600ep_bucket_README.md "$STAGE/README.md"
mkdir -p "$STAGE/runner"
cp "$RUNNER_SRC"/restore.sh "$RUNNER_SRC"/train-60k.sh "$RUNNER_SRC"/train-80k.sh \
   "$RUNNER_SRC"/_train_common.sh "$STAGE/runner/"

# stage 里只能有实体文件：上传端遍历不跟随符号链接**目录**，其内容会被静默整体漏传
if find "$STAGE" -type l -print | grep -q .; then
  echo "错误: stage 内出现符号链接"; find "$STAGE" -type l -print; exit 1
fi
if find "$STAGE" -name 'pack.lock' -print | grep -q .; then
  echo "错误: stage 内出现 pack.lock"; exit 1
fi

# 第二遍体检扫的是 stage —— 这一遍才看得到我们自己新写的文件（README.md、runner/*.sh、
# manifest/upload_manifest.json、checksums/*.txt）。400ep 首轮就停在 README 自身命中。
uv run --no-sync python scripts/dataset/hf_export/scan_hygiene.py \
  --root "$STAGE" --scope stage \
  --allowlist "$ALLOWLIST" --report "$EXPORT_ROOT/logs/hygiene-stage.json"

hash_tree "$STAGE" "$TMPD/SHA256SUMS.pre.txt"
mv "$TMPD/SHA256SUMS.pre.txt" "$STAGE/SHA256SUMS.pre.txt"

LOCAL_FILES=$(find "$STAGE" -type f | wc -l)
LOCAL_BYTES=$(find "$STAGE" -type f -printf '%s\n' | awk '{s+=$1} END{print s}')
PRE_LINES=$(wc -l < "$STAGE/SHA256SUMS.pre.txt")
echo "LOCAL_FILES=$LOCAL_FILES LOCAL_BYTES=$LOCAL_BYTES"
echo "PRE_LINES=$PRE_LINES"
echo "阶段4完成"

# ---------------------------------------------------------------- 阶段 5：分批上传
echo "阶段5开始：分批上传"
# **为什么分批**（2026-09-08 实测）：一次 sync 提交全部对象时字节全部传完，却在
# `_batch_bucket_files → session.new_upload_commit` 抛 TimeoutError，三次重试同因失败、
# bucket 始终 files=0。速率 546 MB/s，瓶颈不在带宽而在服务端一次性提交大量 Xet 对象的
# batch commit。阈值经验值：1.62 GB×10 过、2 GB×5 稳、2 GB×10 挂。
# ⚠ 本轮新变量：framesamp-8x8 的 32 个 part 单片就 9.78–10.21 GB，装箱后一片一批，
#   正好顶在已知阈值上。若某片 8 次仍失败，改用 hf buckets cp 单文件传该片。
upload_batch() {  # $1 = stage 下的子目录（同时也是 bucket 内前缀）；$2.. = --include 模式
  local sub="$1"; shift
  local args=() pat
  for pat in "$@"; do args+=(--include "$pat"); done
  local ok=0 attempt
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf sync "$STAGE/$sub" "$BUCKET/$sub" "${args[@]}"; then ok=1; break; fi
    echo "  批 [$sub] 第 $attempt 次中断，60s 后重试续传"; sleep 60
  done
  [ "$ok" = 1 ] || { echo "错误: 批 [$sub] 八次仍失败"; exit 1; }
  echo "  批完成: $sub"
}

touch "$EXPORT_ROOT/logs/uploaded.marker"

# 装箱计划落盘、幂等读盘：分组漂移是续传的头号杀手——源目录任何变动都会让贪心装箱重排，
# 已传的批和新分组对不上，--include 模式随之失配。
if [ -f "$PLAN" ]; then
  echo "  上传计划已存在，读盘复用（不重算装箱）"
else
  uv run --no-sync python scripts/dataset/hf_export/plan_upload_batches.py \
    --stage "$STAGE" --out "$PLAN"
fi
python3 -c "
import json
p = json.load(open('$PLAN'))
print(f\"PLAN_BATCHES={len(p['batches'])} PLAN_FILES={p['total_files']} \"
      f\"PLAN_BYTES={p['total_bytes']} MAX_BATCH_BYTES={max(b['bytes'] for b in p['batches'])}\")
"
PLAN_FILES=$(python3 -c "import json;print(json.load(open('$PLAN'))['total_files'])")
[ "$PLAN_FILES" = "$LOCAL_FILES" ] || { echo "错误: 计划文件数 $PLAN_FILES != stage $LOCAL_FILES"; exit 1; }

mkdir -p "$EXPORT_ROOT/logs/uploaded"
NBATCH=$(python3 -c "import json;print(len(json.load(open('$PLAN'))['batches']))")
for ((b = 0; b < NBATCH; b++)); do
  BID=$(python3 -c "import json;print(json.load(open('$PLAN'))['batches'][$b]['id'])")
  DONE="$EXPORT_ROOT/logs/uploaded/$BID.done"
  if [ -f "$DONE" ]; then echo "  [$((b+1))/$NBATCH] $BID 已完成，跳过"; continue; fi
  SUB=$(python3 -c "import json;print(json.load(open('$PLAN'))['batches'][$b]['sub'])")
  mapfile -t FILES < <(python3 -c "
import json
for f in json.load(open('$PLAN'))['batches'][$b]['files']: print(f)
")
  BBYTES=$(python3 -c "import json;print(json.load(open('$PLAN'))['batches'][$b]['bytes'])")
  echo "  [$((b+1))/$NBATCH] $BID  ${#FILES[@]} 件 / $BBYTES 字节"
  upload_batch "$SUB" "${FILES[@]}"
  touch "$DONE"
done
echo "UPLOAD_DONE batches=$NBATCH"

# bucket 根下的平文件：sync 的源必须是目录，这些走 cp 单传
mapfile -t ROOTFILES < <(python3 -c "
import json
for f in json.load(open('$PLAN'))['root_files']: print(f)
")
echo "  根下平文件 ${#ROOTFILES[@]} 个"
for f in "${ROOTFILES[@]}"; do
  [ -f "$STAGE/$f" ] || { echo "错误: stage 缺少根文件 $f"; exit 1; }
  ok=0
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf buckets cp "$STAGE/$f" "$BUCKET/$f"; then ok=1; break; fi
    echo "  $f 第 $attempt 次中断，60s 后重试"; sleep 60
  done
  [ "$ok" = 1 ] || { echo "错误: $f 八次仍失败"; exit 1; }
done
echo "阶段5完成"

# ---------------------------------------------------------------- 阶段 6：计数层
echo "阶段6开始：计数层核对（REST API，异步滞后则复查一次）"
BUCKET_FILES="$(bucket_stat totalFiles)"; BUCKET_BYTES="$(bucket_stat size)"
if [ "$BUCKET_FILES" != "$LOCAL_FILES" ] || [ "$BUCKET_BYTES" != "$LOCAL_BYTES" ]; then
  echo "  首次不符（files=$BUCKET_FILES bytes=$BUCKET_BYTES），统计接口可能异步滞后，120s 后复查"
  sleep 120
  BUCKET_FILES="$(bucket_stat totalFiles)"; BUCKET_BYTES="$(bucket_stat size)"
fi
echo "BUCKET_FILES=$BUCKET_FILES BUCKET_BYTES=$BUCKET_BYTES"
# 计数层不单独作判据（只是便宜的早停信号，真判据在阶段 8），但不符仍要停下来看
[ "$BUCKET_FILES" = "$LOCAL_FILES" ] || { echo "错误: 文件数不符 $BUCKET_FILES != $LOCAL_FILES"; exit 1; }
[ "$BUCKET_BYTES" = "$LOCAL_BYTES" ] || { echo "错误: 字节数不符 $BUCKET_BYTES != $LOCAL_BYTES"; exit 1; }
echo "阶段6完成"

# ---------------------------------------------------------------- 阶段 7：全量回读
echo "阶段7开始：全量回读到 $VERIFY（669 GiB）"
# 不 rm -rf：hf sync 本身增量，保留已回读部分可让中断后续传
mkdir -p "$VERIFY"
ok=0
for attempt in 1 2 3; do
  if hf sync "$BUCKET" "$VERIFY"; then ok=1; break; fi
  echo "回读第 $attempt 次中断，60s 后重试续传"; sleep 60
done
[ "$ok" = 1 ] || { echo "回读三次仍失败"; exit 1; }
echo "阶段7完成"

# ---------------------------------------------------------------- 阶段 8：内容层验收
echo "阶段8开始：上传后 sha256（POST）+ 与上传前逐行对照 + tar 成员抽样"
# 独立重算，不是拿 pre 去 -c：两份清单都留档，任何时候能指出是哪一行、哪个文件变了
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

# tar 成员级：SHA256SUMS 只看得见 tar 本身，看不见成员名写错或清单与内容错位。
# 按顶层类别分组抽样（data / wan-latents / oracle / motion-tokens 四类命名规则各不相同）。
uv run --no-sync python scripts/dataset/hf_export/verify_tar_members.py \
  --verify-root "$VERIFY" --checksums "$VERIFY/checksums/sha256-source-files.txt" \
  --sample-tars 6 --seed 20260920
echo "RESULT=PASS"
echo "阶段8完成"

# ---------------------------------------------------------------- 阶段 9：源侧第三遍
echo "阶段9开始：收尾断言（源侧第三遍 sha256，证明硬链接未改动原件）"
# 只核直传件（硬链接那些）——tar 分片没有对应的单个源文件，其内容由
# checksums/sha256-source-files.txt 的逐成员清单覆盖。
# 注意 hash_tree 用 find -printf '%P'，清单里的路径**不带 ./ 前缀**。
grep -v -E "  (README\.md|SHA256SUMS\.pre\.txt|manifest/|checksums/|runner/)" \
  "$STAGE/SHA256SUMS.pre.txt" \
  | grep -v -E "  [A-Za-z0-9/-]*_tars/" | sort -k 2 > "$TMPD/pre-plain.txt"
while read -r _digest rel; do
  if [ "$rel" = "assets/robomme/norm_stats.json" ]; then src="$NORM_STATS"; else src="$LIB/$rel"; fi
  echo "$(sha256sum "$src" | cut -d' ' -f1)  $rel"
done < "$TMPD/pre-plain.txt" | sort -k 2 > "$TMPD/src-plain.txt"
cp "$TMPD/src-plain.txt" "$EXPORT_ROOT/SHA256SUMS.src-after.txt"
echo "  直传件核对行数: $(wc -l < "$TMPD/pre-plain.txt")"
if diff "$TMPD/pre-plain.txt" "$TMPD/src-plain.txt"; then
  echo "SRC_UNCHANGED=OK"
else
  echo "错误: 源文件 sha256 发生变化（上方 diff 即差异行）"; exit 1
fi
echo "阶段9完成"

# ---------------------------------------------------------------- 阶段 10：公开性确认
echo "阶段10开始：公开性确认"
# 注意：verify_model_repo.py 写死 `info.private is not True → return 1`，那是给 model repo
# 用的、口径是「必须保持 private」。本轮是 bucket 且目标就是 public，方向相反。
# 两处断言各自显式、不共用函数——不要为了「统一」把这里改成强制 private。
FINAL_PRIVATE="$(bucket_stat private)"
case "$FINAL_PRIVATE" in
  False|false|None) echo "BUCKET_PUBLIC=True" ;;
  *) echo "错误: bucket 不是公开的（private=$FINAL_PRIVATE）"; exit 1 ;;
esac

# 匿名读验证必须清掉 token，否则用自己的凭据拉下来，证明不了任何「公开」的事
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

# ---------------------------------------------------------------- 阶段 11：冷启动彩排
echo "阶段11开始：冷启动彩排（只用回读副本跑 20 步，两条 config 各一次）"
# 字节层全过只证明「传对了」，证明不了「下完能训」。这一步用 $VERIFY —— 一份真从 HF
# 下回来的副本 —— 起训练，命中的是读侧的全部闸：两库的 pack.lock / status=verified、
# manifest sha256 现场重算、32 个 part 的 st_size 与首尾 blake2b、源库 16 选 1 抽样指纹
# （专验 source/ 传全了没有）、motion_index sha256、双库同源、source_run ↔ provenance.encoder。
#
# 腾盘：阶段 9 已证明源件未动，stage 里的 tar 实体（305 GiB）此刻可以删，给解包让位。
find "$STAGE" -name '*.tar' -delete
echo "  已删除 stage 内 tar 实体，为解包腾出空间"
for t in "$VERIFY"/source/data_tars/*.tar; do tar -xf "$t" -C "$VERIFY/source"; done
NPKL=$(find "$VERIFY/source/data" -name '*.pkl' | wc -l)
[ "$NPKL" = "605611" ] || { echo "错误: 回读副本解出 $NPKL 个 pkl，期望 605611"; exit 1; }
echo "  回读副本解包完成：$NPKL 个 pkl"

rehearse() {  # $1 = config 条目, $2 = history config, $3 = 是否带 motion（1/0）
  local cfg="$1" hc="$2" with_motion="$3" run="coldstart-rehearsal-$1"
  rm -rf "$REPO/v1-store/train-runs/$cfg/$run"
  (
    set -euo pipefail
    cd "$REPO"
    export MMEVLA_FRAMESAMP_SOURCE="$VERIFY/source"
    export MMEVLA_FRAMESAMP_MANIFEST="$VERIFY/meta/episode_manifest.json"
    if [ "$with_motion" = 1 ]; then
      export MMEVLA_MOTION_STORE="$VERIFY/motion"   # 彩排专用；正式起跑不设这条
    else
      unset MMEVLA_MOTION_STORE || true
    fi
    export OPENPI_DATA_HOME="$REPO/v1-store/models"
    uv run --no-sync python scripts/training/train.py "$cfg" \
      --exp-name "$run" --num-train-steps 20 --no-wandb-enabled \
      --assets-base-dir "$REPO/v1-store/train-assets" \
      --data.assets.assets-dir "$VERIFY/assets" --data.assets.asset-id robomme \
      --checkpoint-base-dir "$REPO/v1-store/train-runs" \
      --dataset-path "$VERIFY/framesamp-8x8" \
      --model.history-config "$hc"
  )
  echo "  彩排通过: $cfg"
  rm -rf "$REPO/v1-store/train-runs/$cfg/$run"      # AGENTS 第 6 条：临时 run 跑完即删
}
rehearse mme_vla_suite_b128_60k perceptual-framesamp-modul-8frame-8x8.yaml 0
rehearse mme_vla_suite_b128_80k perceptual-framesamp-modul-8frame-8x8-motion.yaml 1
echo "COLD_START=PASS steps=20"
echo "阶段11完成"
echo "全部完成"
