#!/usr/bin/env bash
# driver：把训练 run v2-1600ep-m8x8-modul-b128-60k 的 12 个 orbax checkpoint
# （132.76 GiB / 272 文件）上传到 HF **private** bucket HongzeFu/robomme-vla-modul-60k-v1，
# 并做「上传前 / 上传后」两遍独立 sha256 逐行对照验收。在 detached tmux 里跑，
# 每阶段幂等、可整脚本重跑续传。
#
# 蓝本是同目录 run_ckpt_export.sh（awsprod40k-b128-motion 的 87 GB / 164 文件），骨架照抄。
# 与它的两处差异：
#   1. **阶段 0 建 bucket**：那条 run 的 bucket 是事先手工建好的，这里脚本内建，幂等。
#      `hf buckets create` **不带 --private 即公开**，务必带上。
#   2. **阶段 2 按 step 目录分 12 批**，不再一次性 sync 整个 stage。理由见阶段 2 注释。
#      分批方式比 run_motionjepa_full1600_export.sh 简单得多：那条链路的源是 2991 个重尾
#      .bin，必须贪心装箱并把 upload_plan.json 落盘（否则重跑时分组漂移、--include 失配）；
#      这里 orbax 的 step 目录本身就是稳定的天然边界，每个恰好 11.06 GiB，
#      直接按目录切批即可，不需要 plan_upload_batches.py。
#
# 沿用蓝本的三条做法（都是本环境实测踩出来的）：
#   - **stage 走硬链接**：stage 与源同在 /dev/md0，ln -f 额外占盘为 0，不是再拷 132.76 GiB。
#     代价是必须在收尾阶段重算源文件 sha256，断言没被连带改坏（阶段 6）。
#   - **不打 tar**：保持「下载下来直接 orbax.restore」这个性质。
#   - **不复制 token**：凭据走 v1-store/secrets/hf.env 注入 HF_TOKEN 环境变量，
#     huggingface_hub 优先读它，不需要任何 token 文件落第二份盘。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
cd "$REPO"

RUN_NAME="v2-1600ep-m8x8-modul-b128-60k"
BUCKET_ID="HongzeFu/robomme-vla-modul-60k-v1"
BUCKET="hf://buckets/$BUCKET_ID"
# 注意中间层是 mme_vla_suite_b128_60k，**不是** mme_vla_suite_b128（后者是上一条 40k run）
SRC="$REPO/v1-store/train-runs/mme_vla_suite_b128_60k/$RUN_NAME"
EXPORT_ROOT="$REPO/v1-store/exports/hf-ckpt-$RUN_NAME"
STAGE="$EXPORT_ROOT/stage"
VERIFY="$EXPORT_ROOT/verify"
TMPD="$EXPORT_ROOT/tmp"
NPROC=8

# AGENTS 第 14 条：缓存类环境变量逐项指向 v1-store / scratch，**禁止覆盖 HOME**
export HF_HOME="$REPO/v1-store/cache/hf"
export HF_XET_CACHE="$REPO/v1-store/cache/hf-xet"
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export UV_PYTHON_INSTALL_DIR=/scratch/hongze/.cache/uv-python
export UV_LINK_MODE=copy
unset HF_HUB_OFFLINE || true

# 凭据：只从 v1-store/secrets/hf.env 取（600、在 .gitignore 的 /v1-store/ 下）。
# 环境里原有的 HF_TOKEN 属于 yinpei-tri，写不进 HongzeFu 命名空间，必须被这里覆盖掉。
[ -f "$REPO/v1-store/secrets/hf.env" ] || { echo "错误: 缺少 v1-store/secrets/hf.env"; exit 1; }
set -a; . "$REPO/v1-store/secrets/hf.env"; set +a
[ -n "${HF_TOKEN:-}" ] || { echo "错误: hf.env 未提供 HF_TOKEN"; exit 1; }

# 项目 .venv 的 huggingface_hub 是 0.32.3，**没有 bucket 子命令**，且由 openpi 依赖钉住
# 不能升。用 uvx 拉一份完全独立的 1.30.0 CLI，只在本脚本内使用，
# 项目 pyproject.toml / uv.lock / .venv 一律不碰。
hf() { uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf "$@"; }

# 精确字节/文件数只信 buckets REST API：`hf repos ls` 给的是 "132.8 GB" 这种人类可读值，
# 而且该类统计接口**异步滞后**（上次 480 GB 传完一度显示 312 GB / 200 文件）。
bucket_stat() {  # $1 = 字段名 size|totalFiles
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
# -not -name 'SHA256SUMS.*' 让清单文件自身不进清单——pre/post/src 三份才能逐行对齐。
hash_tree() {  # $1 = 目录, $2 = 输出文件（绝对路径，不能落在被扫描目录内）
  ( cd "$1" && find . -type f -not -name 'SHA256SUMS.*' -printf '%P\n' | sort \
      | xargs -P "$NPROC" -n 16 sha256sum | sort -k 2 > "$2" )
}

# ---------------------------------------------------------------- 阶段 0
echo "阶段0开始：预检 + 建 bucket"
# 落点必须是本环境实体目录（AGENTS 第 13/14 条：环境 B 的 v1-store 下没有任何 symlink 外链）
for d in "$REPO/v1-store/exports" "$EXPORT_ROOT"; do
  [ -L "$d" ] && { echo "错误: $d 是符号链接，拒绝写入"; exit 1; }
done
[ -d "$SRC" ] || { echo "错误: 源目录不存在 $SRC"; exit 1; }
# rm -rf "$STAGE" 在阶段 1 执行，这里先断言它不可能等于源目录
[ "$STAGE" != "$SRC" ] || { echo "错误: STAGE 与 SRC 相同，拒绝继续"; exit 1; }
case "$STAGE" in "$EXPORT_ROOT"/*) : ;; *) echo "错误: STAGE 不在 EXPORT_ROOT 下"; exit 1 ;; esac
mkdir -p "$STAGE" "$VERIFY" "$TMPD" "$EXPORT_ROOT/logs" "$HF_XET_CACHE"

# 注意：whoami 的输出格式随是否有 tty 而变（非 tty 是单行 `user=X orgs=Y`，tty 下是多行
# `user: X` / `  orgs: Y`）。tmux 里有 tty，**不能按行取字段**，须整体合并后再匹配。
who="$(hf auth whoami 2>&1 | tr '\n' ' ' | tr -s ' ')"
echo "  whoami: $who"
case "$who" in *HongzeFu*) : ;; *) echo "错误: 身份不是 HongzeFu（$who）"; exit 1 ;; esac
echo "HF_WHOAMI=HongzeFu"

# 建 bucket（幂等）：已存在则 create 报错，忽略后继续——下面的空库断言会兜底。
# **必须带 --private**：不带即公开。
if hf buckets create "$BUCKET_ID" --private 2>&1 | tee "$TMPD/create.log"; then
  echo "  bucket create 成功"
else
  echo "  bucket create 非零退出（可能已存在），继续走空库断言"
fi

# 绝不往一个已有内容的 bucket 上 sync：repo_id 打错一个字母就会静默污染别的库。
# 续跑时靠 uploaded.marker 区分「首次跑」与「本轮续传」。
pre_files="$(bucket_stat totalFiles)"; pre_size="$(bucket_stat size)"
echo "  目标 bucket 现状: files=$pre_files size=$pre_size"
if [ "$pre_files" != "0" ] && [ ! -f "$EXPORT_ROOT/logs/uploaded.marker" ]; then
  echo "错误: 目标 bucket 非空（files=$pre_files）且本地无本轮上传标记，停止并交用户裁决"
  exit 1
fi
echo "阶段0完成"

# ---------------------------------------------------------------- 阶段 1
echo "阶段1开始：搭 stage 树（ckpt 走硬链接，不复制 132.76 GiB）+ 上传前 sha256"
rm -rf "$STAGE"; mkdir -p "$STAGE"
( cd "$SRC" && find . -mindepth 1 -type d -printf '%P\n' ) | while read -r d; do
  mkdir -p "$STAGE/$d"
done
( cd "$SRC" && find . -mindepth 1 -type f -printf '%P\n' ) | while read -r f; do
  ln -f "$SRC/$f" "$STAGE/$f"
done
cp "$REPO/scripts/dataset/hf_export/modul60k_bucket_README.md" "$STAGE/README.md"

# stage 里只能有实体文件：上传端遍历是 os.walk(followlinks=False)，
# 符号链接**目录**的内容会被静默整体漏传
if find "$STAGE" -type l -print | grep -q .; then
  echo "错误: stage 内出现符号链接"; find "$STAGE" -type l -print; exit 1
fi

hash_tree "$STAGE" "$TMPD/SHA256SUMS.pre.txt"
mv "$TMPD/SHA256SUMS.pre.txt" "$STAGE/SHA256SUMS.pre.txt"

# LOCAL_FILES 在 mv 之后统计，故含 README.md 与 SHA256SUMS.pre.txt 自身（272+2=274）；
# PRE_LINES 是清单行数，含 README.md 但不含清单自身（272+1=273）。
LOCAL_FILES=$(find "$STAGE" -type f | wc -l)
LOCAL_BYTES=$(find "$STAGE" -type f -printf '%s\n' | awk '{s+=$1} END{print s}')
PRE_LINES=$(wc -l < "$STAGE/SHA256SUMS.pre.txt")
echo "LOCAL_FILES=$LOCAL_FILES LOCAL_BYTES=$LOCAL_BYTES"
echo "PRE_LINES=$PRE_LINES"
[ "$LOCAL_FILES" = "274" ] || { echo "错误: stage 文件数应为 274，实为 $LOCAL_FILES"; exit 1; }
[ "$PRE_LINES" = "273" ] || { echo "错误: pre 清单应为 273 行，实为 $PRE_LINES"; exit 1; }
echo "阶段1完成"

# ---------------------------------------------------------------- 阶段 2
echo "阶段2开始：按 step 目录分 12 批上传"
# **为什么分批**（2026-09-08 实测）：一次 sync 提交大量对象时，字节全部传完却在
# `_batch_bucket_files → session.new_upload_commit` 抛 TimeoutError，重试同因失败、
# bucket 始终 files=0。失败阈值在**批内字节量**：1.62 GB×10=16.2 GB 过、2 GB×5=10 GB 稳、
# 2 GB×10=20 GB 挂。本次每个 step 目录 11.06 GiB / 20–25 文件，落在已验证会过的量级。
# 传输速率 546 MB/s，瓶颈不在带宽。已传字节不浪费——Xet 内容寻址，重跑时相同的块服务端已有。
touch "$EXPORT_ROOT/logs/uploaded.marker"
mkdir -p "$EXPORT_ROOT/logs/uploaded"

# 批次即 step 目录名，从 stage 实际枚举（不硬编码），按数值排序
mapfile -t STEPS < <( cd "$STAGE" && find . -mindepth 1 -maxdepth 1 -type d -printf '%P\n' | sort -n )
NBATCH=${#STEPS[@]}
[ "$NBATCH" = "12" ] || { echo "错误: step 目录数应为 12，实为 $NBATCH"; exit 1; }

upload_batch() {  # $1 = stage 下的子目录（同时也是 bucket 内前缀）
  local sub="$1" ok=0 attempt
  # 重试 8 次而不是 3 次：实测该 commit 超时是**间歇性**的，不是确定性失败
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf sync "$STAGE/$sub" "$BUCKET/$sub"; then ok=1; break; fi
    echo "  批 [$sub] 第 $attempt 次中断，60s 后重试续传"; sleep 60
  done
  [ "$ok" = 1 ] || { echo "错误: 批 [$sub] 八次仍失败"; exit 1; }
}

for ((b = 0; b < NBATCH; b++)); do
  SUB="${STEPS[$b]}"
  DONE="$EXPORT_ROOT/logs/uploaded/$SUB.done"
  if [ -f "$DONE" ]; then echo "  [$((b+1))/$NBATCH] step $SUB 已完成，跳过"; continue; fi
  BBYTES=$(find "$STAGE/$SUB" -type f -printf '%s\n' | awk '{s+=$1} END{print s}')
  BFILES=$(find "$STAGE/$SUB" -type f | wc -l)
  echo "  [$((b+1))/$NBATCH] step $SUB  $BFILES 件 / $BBYTES 字节"
  upload_batch "$SUB"
  touch "$DONE"
  echo "  批完成: $SUB"
done
echo "UPLOAD_DONE batches=$NBATCH"

# --- bucket 根下的平文件：sync 的源必须是目录，这些用 cp 单传 ---
mapfile -t ROOTFILES < <( cd "$STAGE" && find . -mindepth 1 -maxdepth 1 -type f -printf '%P\n' | sort )
echo "  根下平文件 ${#ROOTFILES[@]} 个（sync 的源必须是目录，这些走 cp）"
for f in "${ROOTFILES[@]}"; do
  [ -f "$STAGE/$f" ] || { echo "错误: stage 缺少根文件 $f"; exit 1; }
  ok=0
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf buckets cp "$STAGE/$f" "$BUCKET/$f"; then ok=1; break; fi
    echo "  $f 第 $attempt 次中断，60s 后重试"; sleep 60
  done
  [ "$ok" = 1 ] || { echo "错误: $f 八次仍失败"; exit 1; }
done
echo "阶段2完成"

# ---------------------------------------------------------------- 阶段 3
echo "阶段3开始：计数层核对（buckets REST API，异步滞后则复查一次）"
BUCKET_FILES="$(bucket_stat totalFiles)"; BUCKET_BYTES="$(bucket_stat size)"
if [ "$BUCKET_FILES" != "$LOCAL_FILES" ] || [ "$BUCKET_BYTES" != "$LOCAL_BYTES" ]; then
  echo "  首次不符（files=$BUCKET_FILES bytes=$BUCKET_BYTES），统计接口可能异步滞后，120s 后复查"
  sleep 120
  BUCKET_FILES="$(bucket_stat totalFiles)"; BUCKET_BYTES="$(bucket_stat size)"
fi
echo "BUCKET_FILES=$BUCKET_FILES BUCKET_BYTES=$BUCKET_BYTES"
# 计数层只是便宜的早停信号，真判据在阶段 5；但不符仍要停下来看
[ "$BUCKET_FILES" = "$LOCAL_FILES" ] || { echo "错误: 文件数不符 $BUCKET_FILES != $LOCAL_FILES"; exit 1; }
[ "$BUCKET_BYTES" = "$LOCAL_BYTES" ] || { echo "错误: 字节数不符 $BUCKET_BYTES != $LOCAL_BYTES"; exit 1; }
echo "阶段3完成"

# ---------------------------------------------------------------- 阶段 4
echo "阶段4开始：全量回读到 $VERIFY（132.76 GiB）"
# 不 rm -rf：hf sync 本身增量，保留已回读部分可让中断后续传
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
echo "RESULT=PASS"
echo "阶段5完成"

# ---------------------------------------------------------------- 阶段 6
echo "阶段6开始：收尾断言（源侧第三遍 sha256，证明硬链接未改动原件）"
hash_tree "$SRC" "$TMPD/SHA256SUMS.src-after.txt"
cp "$TMPD/SHA256SUMS.src-after.txt" "$EXPORT_ROOT/SHA256SUMS.src-after.txt"
# pre 清单比源多 README.md 一行（stage 专有）；SHA256SUMS.pre.txt 本就被 hash_tree 排除，
# 不在清单内，无需额外剔除。比对前剔掉 README.md 并断言行数差恰为 1。
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
