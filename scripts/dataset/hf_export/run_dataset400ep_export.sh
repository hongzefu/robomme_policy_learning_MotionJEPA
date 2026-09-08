#!/usr/bin/env bash
# driver：把派生数据集 v1-store/datasets/4task-motion-400ep（130 GB / 229,390 个文件）
# 上传到**公开** HF bucket HongzeFu/robomme-4task-motion-400ep-20260904-v1，
# 做「上传前 / 上传后」两遍独立 sha256 逐行对照验收。detached tmux 里跑，每阶段幂等、
# 整脚本可重跑续传。
#
# 与同目录两条老链路的关系：
#   run_export.sh（环境 A，480 GB）—— 打 tar 的方法论来自它，pack_and_hash.py 是共用的，
#     但它的三个路径常量已改为 CLI 参数，两条链路各自在自己的 driver 里声明用哪个库。
#   run_ckpt_export.sh（环境 B，87 GB ckpt）—— 本脚本的阶段骨架、凭据注入、whoami 匹配、
#     bucket_stat REST 查询、hash_tree 都照抄它，那是本环境验证过的做法。
#
# 本脚本相对 ckpt 那条多出来的三件事：
#   1. **打 tar**：22.4 万个平均 400–600 KB 的小文件直传会退化成逐文件 HTTP 往返。
#      framesamp/ 的 31 个 260 MB part bin 与 motion/ 反而原样直传——打包只多一遍读写，
#      而且会毁掉「下下来即训练可直读 store」这个性质。
#   2. **公开性体检闸门（阶段 1）**：目标是公开 bucket，且用户拍板「创建时即公开」，
#      所以 bucket 一建出来就没有「发现问题还来得及」的窗口。体检必须跑在 create 之前。
#   3. **tar 成员级抽样复算（阶段 8）**：SHA256SUMS 只看得见 tar 本身，看不见成员。
#
# 用法：
#   bash scripts/dataset/hf_export/run_dataset400ep_export.sh            # 全量
#   SMOKE=1 bash scripts/dataset/hf_export/run_dataset400ep_export.sh    # 只跑阶段 -1
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
cd "$REPO"

LIB="$REPO/v1-store/datasets/4task-motion-400ep"
NORM_STATS="$REPO/v1-store/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json"
BUCKET_ID="HongzeFu/robomme-4task-motion-400ep-20260904-v1"
BUCKET="hf://buckets/$BUCKET_ID"
EXPORT_ROOT="$REPO/v1-store/exports/hf-dataset-4task-motion-400ep"
STAGE="$EXPORT_ROOT/stage"
VERIFY="$EXPORT_ROOT/verify"
TMPD="$EXPORT_ROOT/tmp"
HYGIENE_REPORT="$EXPORT_ROOT/logs/hygiene.json"
NPROC=8
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
# 本机 ~/.cache/huggingface/ 下没有 token 文件，所以只能走环境变量注入这一条路。
[ -f "$REPO/v1-store/secrets/hf.env" ] || { echo "错误: 缺少 v1-store/secrets/hf.env"; exit 1; }
set -a; . "$REPO/v1-store/secrets/hf.env"; set +a
[ -n "${HF_TOKEN:-}" ] || { echo "错误: hf.env 未提供 HF_TOKEN"; exit 1; }

# 项目 .venv 的 huggingface_hub 是 0.32.3（openpi 钉死、不能升，且无 bucket 子命令），
# 故用 uvx 拉一份完全独立的 1.30.0 CLI，只在本脚本内使用，pyproject/uv.lock/.venv 一律不碰。
hf() { uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf "$@"; }

# 精确字节/文件数/可见性只信 buckets REST API：`hf repos ls` 给的是 "130.4 GB" 这种人类可读值，
# 且该类统计接口**异步滞后**（480 GB 那次传完一度显示 312 GB / 200 文件；87 G 那次首查 0/0）。
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

# 在 \$1 目录内并行算 sha256，输出按路径排序（xargs -P 并行会乱序，不排序则无法 diff）
hash_tree() {  # $1 = 目录, $2 = 输出文件（绝对路径，不能落在被扫描目录内）
  ( cd "$1" && find . -type f -not -name 'SHA256SUMS.*' -printf '%P\n' | sort \
      | xargs -P "$NPROC" -n 16 sha256sum | sort -k 2 > "$2" )
}

mkdir -p "$STAGE" "$VERIFY" "$TMPD" "$EXPORT_ROOT/logs" "$HF_XET_CACHE"

# ---------------------------------------------------------------- 阶段 -1：smoke
if [ "$SMOKE" = 1 ]; then
  echo "阶段-1开始：smoke（本地打 2 个 data 片 + 1 个 features 片，不碰网络）"
  uv run --no-sync python scripts/dataset/hf_export/pack_and_hash.py \
    --layout motion400ep-v1 --lib "$LIB" --stage-root "$EXPORT_ROOT" \
    --bucket "$BUCKET" --norm-stats "$NORM_STATS" --workers 8 --limit-shards 2

  # 成员级复算：拿打包时逐成员算的 sha 对 tar 里实际内容重算一遍
  cat "$EXPORT_ROOT/logs/parts/"*.sha256 > "$TMPD/smoke-members.txt"
  uv run --no-sync python scripts/dataset/hf_export/verify_tar_members.py \
    --verify-root "$STAGE" --checksums "$TMPD/smoke-members.txt" --sample-tars 3

  # 逐字节对拍：上一步的 sha 与 tar 出自同一遍读，自洽但不能证明与源一致。
  # 这一步把成员解出来跟源文件真比一次，才是「打包没改数」的独立证据。
  uv run --no-sync python - "$LIB" "$STAGE" <<'PY'
import random, sys, tarfile
from pathlib import Path
lib, stage = Path(sys.argv[1]), Path(sys.argv[2])
rng = random.Random(20260908)
tars = sorted(stage.rglob("*.tar"))
bad, checked = [], 0
for t in tars:
    with tarfile.open(t, "r") as tf:
        names = [ti.name for ti in tf if ti.isfile()]
        for nm in rng.sample(names, min(10, len(names))):
            got = tf.extractfile(nm).read()
            src = lib / "source" / nm          # 成员名 data/N.pkl、features/episode_X/...
            want = src.read_bytes()
            checked += 1
            if got != want:
                bad.append(nm)
print(f"SMOKE_BYTEWISE checked={checked} mismatches={len(bad)}")
if bad:
    print("差异成员:", bad[:10]); sys.exit(1)
PY
  echo "阶段-1完成：smoke 通过（产物由 pack_progress.jsonl 记账，全量跑时直接续用）"
  exit 0
fi

# ---------------------------------------------------------------- 阶段 0：预检
echo "阶段0开始：预检（此时 bucket 尚未创建）"
# 落点必须是本环境实体目录（AGENTS 第 13/14 条：环境 B 的 v1-store 下没有任何 symlink 外链）
for d in "$REPO/v1-store/exports" "$EXPORT_ROOT"; do
  [ -L "$d" ] && { echo "错误: $d 是符号链接，拒绝写入"; exit 1; }
done
[ -d "$LIB" ] || { echo "错误: 源库不存在 $LIB"; exit 1; }
[ -f "$NORM_STATS" ] || { echo "错误: norm_stats 不存在 $NORM_STATS"; exit 1; }

# 1.30.0 的 bucket 子命令签名在本环境尚未实证过，跑之前先把 help 原样打进日志、写进留档
echo "--- hf buckets --help ---"; hf buckets --help 2>&1 || true
echo "--- hf buckets create --help ---"; hf buckets create --help 2>&1 || true
echo "--- hf buckets settings --help ---"; hf buckets settings --help 2>&1 || true

# whoami 的输出格式随是否有 tty 而变（非 tty 是单行 `user=X orgs=Y`，tty 下是多行
# `user: X` / `  orgs: Y`）。tmux 里有 tty，**不能按行取字段**，须整体合并后再匹配。
who="$(hf auth whoami 2>&1 | tr '\n' ' ' | tr -s ' ')"
echo "  whoami: $who"
case "$who" in *HongzeFu*) : ;; *) echo "错误: 身份不是 HongzeFu（$who）"; exit 1 ;; esac
echo "HF_WHOAMI=HongzeFu"
echo "阶段0完成"

# ---------------------------------------------------------------- 阶段 1：公开性体检（闸门）
echo "阶段1开始：公开性体检（★ 闸门：不过不建库）"
# 放在 create 之前而不是最后：目标是公开 bucket，一旦建出来就没有「发现问题还来得及」的
# 窗口。发现问题的成本必须在写下任何字节之前付清。
uv run --no-sync python scripts/dataset/hf_export/scan_hygiene.py \
  --root "$LIB" --extra "$NORM_STATS" --scope source \
  --allowlist scripts/dataset/hf_export/hygiene_allowlist.json \
  --report "$HYGIENE_REPORT"
echo "阶段1完成"

# ---------------------------------------------------------------- 阶段 2：建 bucket（公开）
echo "阶段2开始：创建公开 bucket $BUCKET_ID"
if bucket_stat totalFiles >/dev/null 2>&1; then
  echo "  bucket 已存在，跳过创建（幂等重跑）"
else
  hf buckets create "$BUCKET_ID"
fi
# 显式设一次 public 而不是依赖 create 的默认值：用户要的终态是公开，这一行让它成为
# 结构性成立的事实，而不是「create 默认应该是公开吧」的推断。settings 幂等。
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
  # 幂等重跑时 bucket 里可能已有上一轮传上去的内容，这是正常的；只有「首次跑却非空」才危险
  if [ ! -f "$EXPORT_ROOT/logs/uploaded.marker" ]; then
    echo "错误: 目标 bucket 非空（files=$pre_files）且本地无上传记录，停止并交用户裁决"; exit 1
  fi
  echo "  bucket 非空但本地有上传记录，按续传处理"
fi
echo "阶段2完成"

# ---------------------------------------------------------------- 阶段 3：打包
echo "阶段3开始：打包 → $STAGE（断点续跑，已完成分片自动跳过）"
uv run --no-sync python scripts/dataset/hf_export/pack_and_hash.py \
  --layout motion400ep-v1 --lib "$LIB" --stage-root "$EXPORT_ROOT" \
  --bucket "$BUCKET" --norm-stats "$NORM_STATS" --workers 8
echo "阶段3完成"

# ---------------------------------------------------------------- 阶段 4：stage 收尾 + 上传前 sha256
echo "阶段4开始：README + 符号链接断言 + 上传前 sha256 + stage 侧第二遍体检"
cp scripts/dataset/hf_export/dataset400ep_bucket_README.md "$STAGE/README.md"

# stage 里只能有实体文件：上传端遍历不跟随符号链接**目录**，其内容会被静默整体漏传
if find "$STAGE" -type l -print | grep -q .; then
  echo "错误: stage 内出现符号链接"; find "$STAGE" -type l -print; exit 1
fi

# 第二遍体检扫的是 stage —— 这一遍才看得到我们自己新写的三个文件（README.md、
# manifest/upload_manifest.json、checksums/*.txt）。upload_manifest 曾经会写构建机绝对路径。
uv run --no-sync python scripts/dataset/hf_export/scan_hygiene.py \
  --root "$STAGE" --scope stage \
  --allowlist scripts/dataset/hf_export/hygiene_allowlist.json \
  --report "$EXPORT_ROOT/logs/hygiene-stage.json"

hash_tree "$STAGE" "$TMPD/SHA256SUMS.pre.txt"
mv "$TMPD/SHA256SUMS.pre.txt" "$STAGE/SHA256SUMS.pre.txt"

LOCAL_FILES=$(find "$STAGE" -type f | wc -l)
LOCAL_BYTES=$(find "$STAGE" -type f -printf '%s\n' | awk '{s+=$1} END{print s}')
PRE_LINES=$(wc -l < "$STAGE/SHA256SUMS.pre.txt")
echo "LOCAL_FILES=$LOCAL_FILES LOCAL_BYTES=$LOCAL_BYTES"
echo "PRE_LINES=$PRE_LINES"
echo "阶段4完成"

# ---------------------------------------------------------------- 阶段 5：上传
echo "阶段5开始：上传 bucket（增量 sync，重试至多 3 次）"
ok=0
for attempt in 1 2 3; do
  if hf sync "$STAGE" "$BUCKET"; then ok=1; break; fi
  echo "上传第 $attempt 次中断，60s 后重试续传"; sleep 60
done
[ "$ok" = 1 ] || { echo "上传三次仍失败"; exit 1; }
touch "$EXPORT_ROOT/logs/uploaded.marker"
echo "阶段5完成"

# ---------------------------------------------------------------- 阶段 6：计数层
echo "阶段6开始：计数层核对（buckets REST API，异步滞后则复查一次）"
BUCKET_FILES="$(bucket_stat totalFiles)"; BUCKET_BYTES="$(bucket_stat size)"
if [ "$BUCKET_FILES" != "$LOCAL_FILES" ] || [ "$BUCKET_BYTES" != "$LOCAL_BYTES" ]; then
  echo "  首次不符（files=$BUCKET_FILES bytes=$BUCKET_BYTES），统计接口可能异步滞后，120s 后复查"
  sleep 120
  BUCKET_FILES="$(bucket_stat totalFiles)"; BUCKET_BYTES="$(bucket_stat size)"
fi
echo "BUCKET_FILES=$BUCKET_FILES BUCKET_BYTES=$BUCKET_BYTES"
# 计数层不单独作判据（它只是个便宜的早停信号，真判据在阶段 8），但不符仍要停下来看
[ "$BUCKET_FILES" = "$LOCAL_FILES" ] || { echo "错误: 文件数不符 $BUCKET_FILES != $LOCAL_FILES"; exit 1; }
[ "$BUCKET_BYTES" = "$LOCAL_BYTES" ] || { echo "错误: 字节数不符 $BUCKET_BYTES != $LOCAL_BYTES"; exit 1; }
echo "阶段6完成"

# ---------------------------------------------------------------- 阶段 7：全量回读
echo "阶段7开始：全量回读到 $VERIFY（130 GB）"
# 不 rm -rf：hf sync 本身增量，保留已回读部分可让中断后续传；只有内容层失败时才需要重来
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

uv run --no-sync python scripts/dataset/hf_export/verify_tar_members.py \
  --verify-root "$VERIFY" --checksums "$VERIFY/checksums/sha256-source-files.txt" \
  --sample-tars 4 --seed 20260908
echo "RESULT=PASS"
echo "阶段8完成"

# ---------------------------------------------------------------- 阶段 9：收尾断言
echo "阶段9开始：收尾断言（源侧第三遍 sha256，证明硬链接未改动原件）"
# 只核直传件（硬链接那些）——tar 分片没有对应的单个源文件，其内容由
# checksums/sha256-source-files.txt 的 229,323 行逐成员清单覆盖。
# 注意 hash_tree 用 find -printf '%P'，清单里的路径**不带 ./ 前缀**。
grep -v -E "  (README\.md|SHA256SUMS\.pre\.txt|manifest/|checksums/)" "$STAGE/SHA256SUMS.pre.txt" \
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
# 两处断言各自显式、不共用函数——不要为了「统一」把这里改回强制 private。
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
echo "全部完成"
