#!/usr/bin/env bash
# driver：把公开 dataset repo HongzeFu/robomme-4task-h5-20260912-v2（2057 个文件 / 126.6 GB）
# 原样搬进**同名公开 bucket**，做六层校验。**本脚本不含删除**——删除是独立的
# delete_h5v2_dataset_repo.sh，要显式 CONFIRM_DELETE=1 才执行。
#
# 为什么搬：用户判定当初选 dataset repo 的形式错了，要改成 bucket（对象存储）。
#
# 为什么几乎不传字节：实测 repo 的 2057 个文件里 **1978 个（126.59 GB）带 xetHash**，可以走
# batch_bucket_files(copy=[...]) 的服务端按内容哈希复制（文档原文 "no data is downloaded or
# re-uploaded"）；只有 79 个普通 git blob（22 MB）需要客户端中转。附带好处是逐位同一性由
# 内容寻址直接给出——但它不替代全量回读，两层都做。
#
# 与 run_motionjepa_full1600_export.sh 的差别：那条是「本地产物 → bucket」，这条是
# 「Hub repo → bucket」，所以没有 stage、没有硬链接，校验对象换成 repo 自带的两份账本
# （SHA256SUMS 2055 行、MANIFEST.json 的 archives+videos 共 1978 条 sha256）。
#
# 用法：
#   bash scripts/dataset/hf_export/run_h5v2_rehost.sh          # 全量
#   SMOKE=1 bash scripts/dataset/hf_export/run_h5v2_rehost.sh  # 只到阶段 3（临时 bucket 探测）
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
cd "$REPO"

SRC_REPO="HongzeFu/robomme-4task-h5-20260912-v2"
SRC_REV="604f16da36d6b6d175884df8fb687dc08e0a36eb"
BUCKET_ID="HongzeFu/robomme-4task-h5-20260912-v2"
BUCKET="hf://buckets/$BUCKET_ID"
TMP_BUCKET="HongzeFu/tmp-rehost-smoke-20260914"
SNAPSHOT="/scratch/hongze/robomme-4task-h5-20260912-v2/snapshot"
EXPORT_ROOT="$REPO/v1-store/exports/hf-h5v2-rehost"
VERIFY="$EXPORT_ROOT/verify"
TMPD="$EXPORT_ROOT/tmp"
RECORDS="$REPO/docs/dataset-build-doc/hf-export-h5v2-rehost-20260914/records"
MANIFEST_JSON="$EXPORT_ROOT/logs/xet_manifest.json"
NPROC=16
SMOKE="${SMOKE:-0}"

export PATH="/home/ec2-user/.local/bin:$PATH"   # tmux server 的 PATH 不含 ~/.local/bin
export HF_HOME="$REPO/v1-store/cache/hf"
export HF_XET_CACHE="$REPO/v1-store/cache/hf-xet"
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export UV_PYTHON_INSTALL_DIR=/scratch/hongze/.cache/uv-python
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE || true

[ -f "$REPO/v1-store/secrets/hf.env" ] || { echo "错误: 缺少 v1-store/secrets/hf.env"; exit 1; }
set -a; . "$REPO/v1-store/secrets/hf.env"; set +a
[ -n "${HF_TOKEN:-}" ] || { echo "错误: hf.env 未提供 HF_TOKEN"; exit 1; }

hf() { uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf "$@"; }
pyhf() { uvx --python 3.11 --with "huggingface_hub==1.30.0" python "$@"; }

bucket_stat() {  # $1 = size|totalFiles|private
  curl -sf -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/api/buckets/HongzeFu" \
    | python3 -c "
import sys, json
for b in json.load(sys.stdin):
    if b['id'] == '$BUCKET_ID':
        print(b.get('$1')); break
else:
    sys.exit('目标 bucket $BUCKET_ID 不在返回列表里')
"
}

hash_tree() {  # $1 = 目录, $2 = 输出文件
  ( cd "$1" && find . -type f -not -name 'SHA256SUMS.post*' -printf '%P\n' | sort \
      | xargs -P "$NPROC" -n 16 sha256sum | sort -k 2 > "$2" )
}

mkdir -p "$VERIFY" "$TMPD" "$EXPORT_ROOT/logs" "$RECORDS" "$HF_XET_CACHE"

# ---------------------------------------------------------------- 阶段 0：凭据
echo "阶段0开始：凭据与身份"
command -v uvx >/dev/null || { echo "错误: PATH 里找不到 uvx。PATH=$PATH"; exit 1; }
# whoami 输出格式随 tty 而变，整体合并后子串匹配（见 full1600 driver 的同处注释）
WHO="$(hf auth whoami --format agent 2>&1 | tr '\n' ' ' | tr -s ' ')"
echo "  whoami: $WHO"
case "$WHO" in *HongzeFu*) : ;; *) echo "错误: 身份不是 HongzeFu（$WHO）"; exit 1 ;; esac
echo "HF_WHOAMI=HongzeFu"
echo "阶段0完成"

# ---------------------------------------------------------------- 阶段 1：源元数据留档
# 必须在删除之前、也在做任何改动之前抓：revision 历史、discussions、croissant、README 的
# YAML front matter、.gitattributes——这些 bucket 结构上装不下，repo 一删就永久没有。
echo "阶段1开始：源 repo 元数据留档 → $RECORDS"
if [ -f "$RECORDS/summary.json" ]; then
  echo "  已存在，跳过（幂等）"
else
  uv run --no-sync python scripts/dataset/hf_export/archive_h5v2_metadata.py \
    --repo "$SRC_REPO" --revision "$SRC_REV" --out "$RECORDS"
fi
echo "阶段1完成"

# ---------------------------------------------------------------- 阶段 2：probe
echo "阶段2开始：取全部文件的 xetHash"
if [ -f "$MANIFEST_JSON" ]; then
  echo "  清单已存在，读盘复用"
else
  uv run --no-sync python scripts/dataset/hf_export/h5v2_copy_to_bucket.py probe \
    --repo "$SRC_REPO" --revision "$SRC_REV" --out "$MANIFEST_JSON"
fi
python3 -c "
import json
m = json.load(open('$MANIFEST_JSON'))
print(f\"PROBE_FILES={len(m['sizes'])} XET={len(m['xet'])} PLAIN={len(m['plain'])}\")
"
echo "阶段2完成"

# ---------------------------------------------------------------- 阶段 3：smoke
# 两件事一次验完：(a) 服务端复制在本环境到底能不能用（文档说 remote-to-remote 只在同一
# storage region 内有效，而新建 bucket 落在哪个区不由我们控制）；(b) 本 repo 里 basename
# 最长的那个文件（247 字节，xfs 的 NAME_MAX 是 255，只剩 8 字节）能不能原样落盘——
# 若下载端给临时文件加后缀，这个会在回读阶段才炸，那时已经搬完 126 GB 了。
echo "阶段3开始：smoke（临时 bucket，验服务端复制 + 最长文件名）"
if [ -f "$EXPORT_ROOT/logs/smoke.done" ]; then
  echo "  smoke 已通过过，跳过（幂等）"
else
  pyhf scripts/dataset/hf_export/h5v2_copy_to_bucket.py smoke \
    --manifest "$MANIFEST_JSON" --tmp-bucket "$TMP_BUCKET" --workdir "$TMPD"
  touch "$EXPORT_ROOT/logs/smoke.done"
fi
echo "阶段3完成"

if [ "$SMOKE" = 1 ]; then echo "SMOKE=1，到此为止"; echo "全部完成"; exit 0; fi

# ---------------------------------------------------------------- 阶段 4：建 bucket
echo "阶段4开始：创建公开 bucket $BUCKET_ID（与 dataset repo 同名）"
if bucket_stat totalFiles >/dev/null 2>&1; then
  echo "  bucket 已存在，跳过创建（幂等重跑）"
else
  hf buckets create "$BUCKET_ID"
fi
hf buckets settings "$BUCKET_ID" --public
BP="$(bucket_stat private)"
echo "BUCKET_CREATED=$BUCKET_ID PRIVATE=$BP"
case "$BP" in False|false|None) : ;; *) echo "错误: bucket 仍非公开（private=$BP）"; exit 1 ;; esac
pre_files="$(bucket_stat totalFiles)"
echo "BUCKET_FILES_AT_START=$pre_files"
if [ "$pre_files" != "0" ] && [ "$pre_files" != "None" ] && [ ! -f "$EXPORT_ROOT/logs/copied.marker" ]; then
  echo "错误: 目标 bucket 非空（files=$pre_files）且本地无搬迁记录，停止交用户裁决"; exit 1
fi
echo "阶段4完成"

# ---------------------------------------------------------------- 阶段 5：搬迁
echo "阶段5开始：服务端复制 1978 件 + 客户端中转 79 件"
touch "$EXPORT_ROOT/logs/copied.marker"
pyhf scripts/dataset/hf_export/h5v2_copy_to_bucket.py copy \
  --manifest "$MANIFEST_JSON" --bucket "$BUCKET_ID" --workdir "$EXPORT_ROOT"

# 迁移说明：本轮唯一新增的对象。原 repo 的 2057 个文件一律逐位原样，不改 README——
# 那是「删掉的东西在 bucket 里一件不缺」这句话能成立的前提。
cp scripts/dataset/hf_export/h5v2_bucket_MIGRATION.md "$TMPD/MIGRATION.md"
ok=0
for attempt in 1 2 3 4 5 6 7 8; do
  if hf buckets cp "$TMPD/MIGRATION.md" "$BUCKET/MIGRATION.md"; then ok=1; break; fi
  echo "  MIGRATION.md 第 $attempt 次中断，30s 后重试"; sleep 30
done
[ "$ok" = 1 ] || { echo "错误: MIGRATION.md 八次仍失败"; exit 1; }
echo "阶段5完成"

# ---------------------------------------------------------------- 阶段 6：L1 清单层
echo "阶段6开始：L1 清单层（bucket 文件清单 ⟷ repo 文件清单，逐条路径与字节）"
uv run --no-sync python - "$MANIFEST_JSON" <<'PY'
import json, os, sys, urllib.request
man = json.load(open(sys.argv[1]))
tok = os.environ["HF_TOKEN"]
bucket = "HongzeFu/robomme-4task-h5-20260912-v2"
got, cursor = {}, None
while True:
    url = f"https://huggingface.co/api/buckets/{bucket}/tree?recursive=true&limit=1000"
    if cursor:
        url += f"&cursor={cursor}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
    with urllib.request.urlopen(req) as r:
        page = json.load(r)
        link = r.headers.get("Link", "")
    for e in page:
        if e.get("type") == "file" or "size" in e:
            got[e["path"]] = e.get("size", 0)
    if 'rel="next"' not in link:
        break
    cursor = link.split("cursor=")[1].split(">")[0].split("&")[0]
want = dict(man["sizes"])
extra = sorted(set(got) - set(want))          # 预期只有 MIGRATION.md
missing = sorted(set(want) - set(got))
badsize = sorted(p for p in want if p in got and got[p] != want[p])
print(f"INVENTORY_DIFF={len(missing) + len(badsize)} BUCKET_FILES={len(got)} "
      f"REPO_FILES={len(want)} MISSING={len(missing)} BADSIZE={len(badsize)} EXTRA={extra}")
if missing:
    print("  缺失:", missing[:10])
if badsize:
    print("  字节不符:", [(p, want[p], got[p]) for p in badsize[:10]])
if missing or badsize or extra != ["MIGRATION.md"]:
    sys.exit(1)
PY
echo "阶段6完成"

# ---------------------------------------------------------------- 阶段 7：L2 XetHash 层
echo "阶段7开始：L2 XetHash 层（服务端内容寻址同一性，零字节）"
uv run --no-sync python - "$MANIFEST_JSON" <<'PY'
import json, os, sys, urllib.request
man = json.load(open(sys.argv[1]))
tok = os.environ["HF_TOKEN"]
bucket = "HongzeFu/robomme-4task-h5-20260912-v2"
got, cursor = {}, None
while True:
    url = f"https://huggingface.co/api/buckets/{bucket}/tree?recursive=true&expand=true&limit=1000"
    if cursor:
        url += f"&cursor={cursor}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
    with urllib.request.urlopen(req) as r:
        page = json.load(r)
        link = r.headers.get("Link", "")
    for e in page:
        h = e.get("xetHash") or (e.get("xet") or {}).get("hash")
        if h:
            got[e["path"]] = h
    if 'rel="next"' not in link:
        break
    cursor = link.split("cursor=")[1].split(">")[0].split("&")[0]
want = man["xet"]
miss = [p for p in want if p not in got]
bad = [p for p in want if p in got and got[p] != want[p]]
print(f"XETHASH_MATCH={len(want) - len(miss) - len(bad)} EXPECTED={len(want)} "
      f"NOT_REPORTED={len(miss)} MISMATCH={len(bad)}")
if bad:
    print("  不符:", bad[:10]); sys.exit(1)
if miss:
    print(f"  注意: bucket 未回报 xetHash 的 {len(miss)} 条（不判失败，由 L3 全量回读覆盖）:",
          miss[:5])
PY
echo "阶段7完成"

# ---------------------------------------------------------------- 阶段 8：L3 全量回读
echo "阶段8开始：L3 全量回读 126.6 GB → $VERIFY"
ok=0
for attempt in 1 2 3; do
  if hf sync "$BUCKET" "$VERIFY"; then ok=1; break; fi
  echo "回读第 $attempt 次中断，60s 后重试续传"; sleep 60
done
[ "$ok" = 1 ] || { echo "回读三次仍失败"; exit 1; }
VF=$(find "$VERIFY" -type f | wc -l); VB=$(find "$VERIFY" -type f -printf '%s\n' | awk '{s+=$1} END{print s}')
echo "VERIFY_FILES=$VF VERIFY_BYTES=$VB"
echo "阶段8完成"

# ---------------------------------------------------------------- 阶段 9：L4 账本层
echo "阶段9开始：L4 账本层（repo 自带的两份独立账本对回读侧重算结果）"
hash_tree "$VERIFY" "$TMPD/SHA256SUMS.post.txt"
cp "$TMPD/SHA256SUMS.post.txt" "$VERIFY/SHA256SUMS.post.txt"

# 账本一：SHA256SUMS（2055 行，repo 自带）
( cd "$VERIFY" && sha256sum -c --strict SHA256SUMS > "$TMPD/check.txt" 2>&1 ) \
  || { echo "错误: SHA256SUMS 校验失败"; grep -v ': OK$' "$TMPD/check.txt" | head -20; exit 1; }
echo "SHA256SUMS_OK=$(grep -c ': OK$' "$TMPD/check.txt")"

# 账本二：MANIFEST.json 的 archives[] 与 videos[]（1978 条，与 SHA256SUMS 互相独立）
uv run --no-sync python - "$VERIFY" <<'PY'
import json, sys
from pathlib import Path
v = Path(sys.argv[1])
man = json.loads((v / "MANIFEST.json").read_text())
post = {}
for line in (v / "SHA256SUMS.post.txt").read_text().splitlines():
    d, rel = line.split("  ", 1)
    post[rel] = d
bad_a = [a["path"] for a in man["archives"] if post.get(a["path"]) != a["sha256"]]
bad_v = [x["path"] for x in man["videos"] if post.get(x["path"]) != x["sha256"]]
print(f"MANIFEST_ARCH_OK={len(man['archives']) - len(bad_a)}/{len(man['archives'])} "
      f"MANIFEST_VIDEO_OK={len(man['videos']) - len(bad_v)}/{len(man['videos'])}")
if bad_a or bad_v:
    print("  archives 不符:", bad_a[:5]); print("  videos 不符:", bad_v[:5]); sys.exit(1)
PY
echo "阶段9完成"

# ---------------------------------------------------------------- 阶段 10：L5 异地副本对拍
# 本地 snapshot/ 里有 09-13 独立下载的 14 个 primary tar.xz（90.5 GB）。拿它和 bucket 取回
# 的那份逐字节比，是一次**异地独立副本对拍**：两份都对，才排除「源 repo 本身在某个时点被
# 改过、而我们把改过的版本原样搬走了」。
echo "阶段10开始：L5 与本地独立副本逐字节对拍"
SAME=0; DIFFC=0
for f in "$SNAPSHOT"/record_dataset_*.h5.tar.xz; do
  b=$(basename "$f")
  if [ -f "$VERIFY/$b" ] && cmp -s "$f" "$VERIFY/$b"; then SAME=$((SAME+1)); else
    echo "  差异或缺失: $b"; DIFFC=$((DIFFC+1)); fi
done
echo "SNAPSHOT_IDENTICAL=$SAME DIFFERENT=$DIFFC"
[ "$DIFFC" = "0" ] || { echo "错误: 与本地独立副本不一致"; exit 1; }
echo "阶段10完成"

# ---------------------------------------------------------------- 阶段 11：L6 匿名层
echo "阶段11开始：L6 公开性 + 匿名读（清 token、用全新空缓存）"
FP="$(bucket_stat private)"
case "$FP" in False|false|None) echo "BUCKET_PUBLIC=True" ;;
  *) echo "错误: bucket 不是公开的（private=$FP）"; exit 1 ;; esac
rm -rf "$TMPD/anon"; mkdir -p "$TMPD/anon"
if env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HF_HUB_OFFLINE HF_XET_CACHE="$TMPD/anon-xet" \
     uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf \
     sync "$BUCKET" "$TMPD/anon" --include 'SHA256SUMS' --include 'MIGRATION.md' \
   && [ -s "$TMPD/anon/SHA256SUMS" ] && [ -s "$TMPD/anon/MIGRATION.md" ]; then
  echo "PUBLIC_ANON_READ=OK anon_files=2"
else
  echo "错误: 匿名读取失败，bucket 可能并未真正公开"; exit 1
fi
echo "REHOST_RESULT=PASS"
echo "阶段11完成"
echo "全部完成"
