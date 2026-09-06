#!/usr/bin/env bash
# driver：把训练 run awsprod40k-b128-motion 的 8 个 orbax checkpoint（87 GB / 164 文件）
# 上传到 HF **private** bucket HongzeFu/robomme-motionjepa-vla-v1，并做「上传前 / 上传后」
# 两遍独立 sha256 逐行对照验收。在 detached tmux 里跑，每阶段幂等、可整脚本重跑续传。
#
# 与同目录 run_export.sh（480 GB 数据集 + 88 万小文件）的关键差异：
#   1. **不打 tar**：源是 164 个 1–2.2 GB 的 OCDBT 大块，打包只多一遍读写，而且会毁掉
#      「下载下来直接 orbax.restore」这个性质。stage 保持 step 目录原结构。
#   2. **stage 走硬链接**：stage 与源同在 /dev/md0，ln -f 建链接额外占盘为 0，不是再拷 87 GB。
#      代价是必须在收尾阶段重算源文件 sha256，断言没被连带改坏（阶段 6）。
#   3. **不复制 token**：run_export.sh 把 ~/.cache/huggingface/{token,stored_tokens} 拷进
#      改指后的 HF_HOME（否则 401），等于让密钥落第二份盘。这里改用 v1-store/secrets/hf.env
#      注入 HF_TOKEN 环境变量，huggingface_hub 优先读它，不需要任何 token 文件。
#   4. **CLI 版本与子命令**：项目 .venv 的 huggingface_hub 是 0.32.3，其 huggingface-cli
#      **没有 bucket 相关子命令**，而该版本由 openpi 依赖钉住、不能升。故用 uvx 拉一份完全
#      独立的 1.30.0 CLI，只在本脚本内使用，项目 pyproject.toml / uv.lock / .venv 一律不碰。
#      注意 1.30 相对老脚本用的版本已重构：`hf buckets sync` → 顶层 `hf sync`，
#      `hf buckets info` → `hf repos ls`（本脚本的精确字节数改走 buckets REST API，见阶段 3）。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
cd "$REPO"

RUN_NAME="awsprod40k-b128-motion"
BUCKET_ID="HongzeFu/robomme-motionjepa-vla-v1"
BUCKET="hf://buckets/$BUCKET_ID"
SRC="$REPO/v1-store/train-runs/mme_vla_suite_b128/$RUN_NAME"
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

hf() { uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf "$@"; }

# 精确字节/文件数只信 buckets REST API：`hf repos ls` 给的是 "480.2 GB" 这种人类可读值，
# 而且上次实测该类统计接口**异步滞后**（480 GB 传完一度显示 312 GB / 200 文件）。
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

# 在 \$1 目录内并行算 sha256，输出按路径排序（xargs -P 并行会乱序，不排序则无法 diff）
hash_tree() {  # $1 = 目录, $2 = 输出文件（绝对路径，不能落在被扫描目录内）
  ( cd "$1" && find . -type f -not -name 'SHA256SUMS.*' -printf '%P\n' | sort \
      | xargs -P "$NPROC" -n 16 sha256sum | sort -k 2 > "$2" )
}

echo "阶段0开始：预检"
# 落点必须是本环境实体目录（AGENTS 第 13/14 条：环境 B 的 v1-store 下没有任何 symlink 外链）
for d in "$REPO/v1-store/exports" "$EXPORT_ROOT"; do
  [ -L "$d" ] && { echo "错误: $d 是符号链接，拒绝写入"; exit 1; }
done
[ -d "$SRC" ] || { echo "错误: 源目录不存在 $SRC"; exit 1; }
mkdir -p "$STAGE" "$VERIFY" "$TMPD" "$EXPORT_ROOT/logs" "$HF_XET_CACHE"

who="$(hf auth whoami 2>&1 | tail -1)"
echo "  whoami: $who"
case "$who" in *HongzeFu*) : ;; *) echo "错误: 身份不是 HongzeFu（$who）"; exit 1 ;; esac
echo "HF_WHOAMI=HongzeFu"

# 绝不往一个已有内容的 bucket 上 sync：repo_id 打错一个字母就会静默污染别的库
pre_files="$(bucket_stat totalFiles)"; pre_size="$(bucket_stat size)"
echo "  目标 bucket 现状: files=$pre_files size=$pre_size"
if [ "$pre_files" != "0" ]; then
  echo "错误: 目标 bucket 非空（files=$pre_files），停止并交用户裁决"; exit 1
fi
echo "阶段0完成"

echo "阶段1开始：搭 stage 树（ckpt 走硬链接，不复制 87 GB）+ 上传前 sha256"
rm -rf "$STAGE"; mkdir -p "$STAGE"
( cd "$SRC" && find . -mindepth 1 -type d -printf '%P\n' ) | while read -r d; do
  mkdir -p "$STAGE/$d"
done
( cd "$SRC" && find . -mindepth 1 -type f -printf '%P\n' ) | while read -r f; do
  ln -f "$SRC/$f" "$STAGE/$f"
done
cp "$REPO/scripts/dataset/hf_export/ckpt_bucket_README.md" "$STAGE/README.md"

# stage 里只能有实体文件：上传端遍历不跟随符号链接**目录**，其内容会被静默整体漏传
if find "$STAGE" -type l -print | grep -q .; then
  echo "错误: stage 内出现符号链接"; find "$STAGE" -type l -print; exit 1
fi

hash_tree "$STAGE" "$TMPD/SHA256SUMS.pre.txt"
mv "$TMPD/SHA256SUMS.pre.txt" "$STAGE/SHA256SUMS.pre.txt"

LOCAL_FILES=$(find "$STAGE" -type f | wc -l)
LOCAL_BYTES=$(find "$STAGE" -type f -printf '%s\n' | awk '{s+=$1} END{print s}')
PRE_LINES=$(wc -l < "$STAGE/SHA256SUMS.pre.txt")
echo "LOCAL_FILES=$LOCAL_FILES LOCAL_BYTES=$LOCAL_BYTES"
echo "PRE_LINES=$PRE_LINES"
echo "阶段1完成"

echo "阶段2开始：上传 bucket（增量 sync，重试至多 3 次）"
ok=0
for attempt in 1 2 3; do
  if hf sync "$STAGE" "$BUCKET"; then ok=1; break; fi
  echo "上传第 $attempt 次中断，60s 后重试续传"; sleep 60
done
[ "$ok" = 1 ] || { echo "上传三次仍失败"; exit 1; }
echo "阶段2完成"

echo "阶段3开始：计数层核对（buckets REST API，异步滞后则复查一次）"
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

echo "阶段4开始：全量回读到 $VERIFY（87 GB）"
rm -rf "$VERIFY"; mkdir -p "$VERIFY"
ok=0
for attempt in 1 2 3; do
  if hf sync "$BUCKET" "$VERIFY"; then ok=1; break; fi
  echo "回读第 $attempt 次中断，60s 后重试续传"; sleep 60
done
[ "$ok" = 1 ] || { echo "回读三次仍失败"; exit 1; }
echo "阶段4完成"

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

echo "阶段6开始：收尾断言（源侧第三遍 sha256，证明硬链接未改动原件）"
hash_tree "$SRC" "$TMPD/SHA256SUMS.src-after.txt"
cp "$TMPD/SHA256SUMS.src-after.txt" "$EXPORT_ROOT/SHA256SUMS.src-after.txt"
# pre 清单比源多 README.md 一行（stage 专有），比对前剔除
grep -v '  README\.md$' "$STAGE/SHA256SUMS.pre.txt" > "$TMPD/pre.src-only.txt"
if diff "$TMPD/pre.src-only.txt" "$TMPD/SHA256SUMS.src-after.txt"; then
  echo "SRC_UNCHANGED=OK"
else
  echo "错误: 源文件 sha256 发生变化（上方 diff 即差异行）"; exit 1
fi
echo "阶段6完成"
echo "全部完成"
