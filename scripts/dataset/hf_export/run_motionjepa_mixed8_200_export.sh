#!/usr/bin/env bash
# driver：把 MotionJEPA 本机 8×A100 建库产物（dataset-local-a100-16task-mixed8-200）上传到**公开**
# HF bucket HongzeFu/robomme-16task-mixed8-200-motion-20261014-v1，做「上传前 / 上传后」两遍独立
# sha256 逐行对照验收。detached tmux 里跑，每阶段幂等、整脚本可重跑续传。
#
# 本脚本由 run_motionjepa_full1600_export.sh 复制而来，骨架（11 阶段）不变，只改：源 / 桶 / 落点、
# README 与体检账本名、清单层多一道结构断言、阶段 8 前多一道 df 余量断言（本次规模是上次的 2.4 倍）。
# 授权：1013 计划决策 5「5 同意」、1015 计划决策 3、用户 2026-10-16「同意继续做」——公开 bucket、
# 按上次 latent bucket 体例、读回校验 sha256、本地不删。
#
# 范围照上次「只传最终产物 + 溯源」：dataset-token/PUBLISHED.json 清单的 4990 个文件
# （4986 个 .bin + metadata.json + chunk_motion.npz + DATASET_NOTES.json + FINALIZE_DONE.json，
# 共 1,139,145,192,226 B）+ PUBLISHED.json 自身 + control/ 溯源（126 个直传件 + content_hashes/
# 5000 个小文件打成一个 tar）。不传 data-raw/、reference*/、smoke100/、logs/、.staging/、
# .archive-staging/、launch/、output-chunk-filter*/，也不传 dataset-token 下的清单外边车
# （*.bin.complete.json / *.bin.sha256 / numeric_mode/ / performance/ / wan_chunk_latents/_shards_done/）。
#
# stage 组装复用 stage_motionjepa_full1600.py：它只依赖「PUBLISHED.json 的 files 是 {相对路径: sha256}」
# 与 control/ 目录，二者本库结构相同（见阶段 2 的结构断言）。硬链接要求 stage 与源同一文件系统
# （都在 /dev/md0 的 /scratch 上）；跨盘时 os.link 会直接报 EXDEV 失败，不会静默退化成复制。
#
# 与 run_dataset400ep_export.sh 的关系：阶段骨架、凭据注入、whoami 匹配、bucket_stat REST 查询、
# hash_tree、upload_batch 的分批与 8 次重试全部照抄那条链路——那是本环境 2026-09-08 实测验证过的
# 做法（一次 sync 122 个对象必挂 new_upload_commit 超时；带宽 546 MB/s 不是瓶颈）。
#
# 本脚本相对那条多出来的一件事：
#   **清单层校验**——PUBLISHED.json 自带 4990 个文件各自的 sha256（建库侧当时写下的），
#   stage 组装后重算一遍逐条对。这层能抓「建库之后源文件被动过」，是回读校验抓不到的。
#
# 用法：
#   bash scripts/dataset/hf_export/run_motionjepa_mixed8_200_export.sh          # 全量
#   SMOKE=1 bash scripts/dataset/hf_export/run_motionjepa_mixed8_200_export.sh  # 只到阶段 3（不碰网络写）
# 正式运行在主检出（v1-store 与 secrets 在那里）；worktree 里没有 v1-store，跑不起来是预期。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
cd "$REPO"

SRC="${SRC:-/scratch/hongze/dataset-local-a100-16task-mixed8-200}"
BUCKET_ID="HongzeFu/robomme-16task-mixed8-200-motion-20261014-v1"
BUCKET="hf://buckets/$BUCKET_ID"
EXPORT_ROOT="$REPO/v1-store/exports/hf-motionjepa-mixed8-200"
STAGE="$EXPORT_ROOT/stage"
VERIFY="$EXPORT_ROOT/verify"
TMPD="$EXPORT_ROOT/tmp"
ALLOWLIST="$REPO/scripts/dataset/hf_export/hygiene_allowlist_motionjepa_mixed8_200.json"
NPROC=16
SMOKE="${SMOKE:-0}"

# AGENTS 第 14 条：缓存类环境变量逐项指向 v1-store / scratch，**禁止覆盖 HOME**
export HF_HOME="$REPO/v1-store/cache/hf"
export HF_XET_CACHE="$REPO/v1-store/cache/hf-xet"
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export UV_PYTHON_INSTALL_DIR=/scratch/hongze/.cache/uv-python
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1
# scripts/dataset/paths.sh 会 export HF_HUB_OFFLINE=1。本脚本不 source 它，但上游 shell 可能带
# 进来——不 unset 的话所有网络操作会静默走缓存，看起来"成功"实际什么都没传。
unset HF_HUB_OFFLINE || true

# 凭据：只从 v1-store/secrets/hf.env 取。环境里原有的 HF_TOKEN 属于 yinpei-tri，写不进 HongzeFu
# 命名空间，必须被这里覆盖掉。
[ -f "$REPO/v1-store/secrets/hf.env" ] || { echo "错误: 缺少 v1-store/secrets/hf.env"; exit 1; }
set -a; . "$REPO/v1-store/secrets/hf.env"; set +a
[ -n "${HF_TOKEN:-}" ] || { echo "错误: hf.env 未提供 HF_TOKEN"; exit 1; }

# 项目 .venv 的 huggingface_hub 是 0.32.3（openpi 钉死、不能升，且无 bucket 子命令），
# 故用 uvx 拉一份独立的 1.30.0 CLI，只在本脚本内使用，pyproject/uv.lock/.venv 一律不碰。
hf() { uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf "$@"; }

# 精确字节/文件数/可见性只信 buckets REST API：`hf repos ls` 给人类可读值，且该类统计接口
# **异步滞后**（400ep 那次传完一度显示 312 GB / 200 文件）。
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

# 在 $1 目录内并行算 sha256，按路径排序（xargs -P 并行输出乱序，不排序无法 diff）
hash_tree() {  # $1 = 目录, $2 = 输出文件（绝对路径，不能落在被扫描目录内）
  ( cd "$1" && find . -type f -not -name 'SHA256SUMS.*' -printf '%P\n' | sort \
      | xargs -P "$NPROC" -n 16 sha256sum | sort -k 2 > "$2" )
}

mkdir -p "$STAGE" "$VERIFY" "$TMPD" "$EXPORT_ROOT/logs" "$HF_XET_CACHE"

# ---------------------------------------------------------------- 阶段 0：凭据
echo "阶段0开始：凭据与身份"
# 显式断言 uvx 可用：tmux server 的 PATH 可能不含 ~/.local/bin，而下面 whoami 的 2>/dev/null
# 会把 "command not found" 一起吞掉，只剩一个没有上下文的 exit 1。
command -v uvx >/dev/null || { echo "错误: PATH 里找不到 uvx。PATH=$PATH"; exit 1; }
# whoami 的输出格式随**是否有 tty** 而变：非 tty 是单行 `user=X orgs=Y`，tty 下是多行的
# `✓ Logged in` / `  user: X` / `  orgs: Y`。tmux 里有 tty——run_dataset400ep_export.sh 的
# 第 140 行就记着这条，本脚本一度改写成按行 grep '^user=' 又踩了一遍（起跑即挂在阶段 0，
# 且因为 2>/dev/null 连错误都看不到）。两道保险：显式 --format agent 钉死格式，再整体
# 合并后做子串匹配，不依赖行结构。
WHO="$(hf auth whoami --format agent 2>&1 | tr '\n' ' ' | tr -s ' ')"
echo "  whoami: $WHO"
case "$WHO" in *HongzeFu*) : ;; *) echo "错误: 身份不是 HongzeFu（$WHO），凭据注入失败"; exit 1 ;; esac
echo "HF_WHOAMI=HongzeFu"
echo "阶段0完成"

# ---------------------------------------------------------------- 阶段 1：组装 stage
echo "阶段1开始：组装 stage（硬链接，零拷贝）"
uv run --no-sync python scripts/dataset/hf_export/stage_motionjepa_full1600.py \
  --src "$SRC" --stage "$STAGE"
cp scripts/dataset/hf_export/motionjepa_mixed8_200_bucket_README.md "$STAGE/README.md"

# 硬断言：清单外的边车绝不能进 stage。dataset-token 下每个 .bin 都有 *.sha256 / *.complete.json
# 边车，其中 .complete.json 承载着构建机内部主机名——如果有人图省事把 stage 写成
# `cp -al dataset-token stage`，这些就会上公开库，而公开过就是公开过、不可逆。
SIDECARS=$(find "$STAGE" \( -name '*.complete.json' -o -name '*.sha256' \) | wc -l)
echo "STAGE_SIDECARS=$SIDECARS"
[ "$SIDECARS" = "0" ] || { echo "错误: stage 里出现了 $SIDECARS 个清单外边车，stage 不是清单驱动的"; exit 1; }

# 硬断言：stage 里不得有符号链接。`hf sync` 的本地遍历是 os.walk(followlinks=False)，
# 符号链接目录的内容会被静默整体漏传（400ep 那轮踩过）。
SYMS=$(find "$STAGE" -type l | wc -l)
[ "$SYMS" = "0" ] || { echo "错误: stage 内有 $SYMS 个符号链接"; find "$STAGE" -type l | head; exit 1; }
echo "阶段1完成"

# ---------------------------------------------------------------- 阶段 2：清单层 + 上传前 sha256
echo "阶段2开始：上传前 sha256 + 对 PUBLISHED.json 自带清单逐条核对"
hash_tree "$STAGE" "$TMPD/SHA256SUMS.pre.txt"
mv "$TMPD/SHA256SUMS.pre.txt" "$STAGE/SHA256SUMS.pre.txt"

uv run --no-sync python - "$STAGE" <<'PY'
import json, sys
from pathlib import Path
stage = Path(sys.argv[1])
pub = json.loads((stage / "PUBLISHED.json").read_text())
# 结构断言：本库 PUBLISHED.json 顶层是 acceptance_sha256 / files / identity / policy 四键，
# files 为 {相对路径: 64 位 sha256}——与 full1600 那份同构。若建库侧换了格式（例如 files 变成
# 列表或带 size 的对象），下面的逐条比对会以难懂的方式出错，所以先在这里明确拒绝。
keys = sorted(pub)
want = pub.get("files")
assert keys == ["acceptance_sha256", "files", "identity", "policy"], f"PUBLISHED.json 顶层键异常: {keys}"
assert isinstance(want, dict), f"files 不是 dict: {type(want).__name__}"
hexd = set("0123456789abcdef")
badfmt = [r for r, h in want.items()
          if r.startswith("/") or ".." in r.split("/") or r == "PUBLISHED.json"
          or not (isinstance(h, str) and len(h) == 64 and set(h) <= hexd)]
print(f"PUBLISHED_FORMAT=OK entries={len(want)} bad_entries={len(badfmt)} "
      f"acceptance_sha256={pub['acceptance_sha256'][:16]} policy={pub['policy'].get('name')}")
if badfmt:
    print("  异常条目:", badfmt[:10]); sys.exit(1)
got = {}
for line in (stage / "SHA256SUMS.pre.txt").read_text().splitlines():
    digest, rel = line.split("  ", 1)
    got[rel] = digest
missing = [r for r in want if r not in got]
bad = [(r, want[r], got[r]) for r in want if r in got and got[r] != want[r]]
print(f"PUBLISHED_SHA_MATCH={len(want) - len(missing) - len(bad)} "
      f"expected={len(want)} missing={len(missing)} mismatches={len(bad)}")
if missing or bad:
    for r in missing[:10]:
        print("  缺失:", r)
    for r, w, g in bad[:10]:
        print(f"  不符: {r}\n    清单 {w}\n    实测 {g}")
    sys.exit(1)
PY

LOCAL_FILES=$(find "$STAGE" -type f | wc -l)
LOCAL_BYTES=$(find "$STAGE" -type f -printf '%s\n' | awk '{s+=$1} END{print s}')
PRE_LINES=$(wc -l < "$STAGE/SHA256SUMS.pre.txt")
echo "LOCAL_FILES=$LOCAL_FILES LOCAL_BYTES=$LOCAL_BYTES"
echo "PRE_LINES=$PRE_LINES"
echo "阶段2完成"

# ---------------------------------------------------------------- 阶段 3：公开前体检
# 放在 create 之前而不是最后：目标是公开 bucket，一旦建出来就没有「发现问题还来得及」的窗口。
# 发现问题的成本必须在写下任何字节之前付清。
echo "阶段3开始：公开前体检（扫 stage，在 create 之前）"
# --scope source 而不是 stage：本轮的「库」就是 PUBLISHED.json 挑出来的那个子集，stage 即
# 恰好要公开的那批文件（4990 + PUBLISHED.json + control/ 127 项 + README + SHA256SUMS.pre.txt，
# 其中文本文件 132 个），所以 expected_count 的严格计数在这里是有意义的闸门——源侧任何
# 变化导致命中数对不上就停机。
uv run --no-sync python scripts/dataset/hf_export/scan_hygiene.py \
  --root "$STAGE" --scope source --allowlist "$ALLOWLIST" \
  --report "$EXPORT_ROOT/logs/hygiene-stage.json"
echo "阶段3完成"

if [ "$SMOKE" = 1 ]; then echo "SMOKE=1，到此为止（未碰网络写）"; echo "全部完成"; exit 0; fi

# ---------------------------------------------------------------- 阶段 4：建 bucket（公开）
echo "阶段4开始：创建公开 bucket $BUCKET_ID"
if bucket_stat totalFiles >/dev/null 2>&1; then
  echo "  bucket 已存在，跳过创建（幂等重跑）"
else
  hf buckets create "$BUCKET_ID"
fi
# 显式设一次 public 而不是依赖 create 的默认值：用户要的终态是公开，这一行让它成为结构性成立的
# 事实，而不是「create 默认应该是公开吧」的推断。settings 幂等。
hf buckets settings "$BUCKET_ID" --public
BUCKET_PRIVATE="$(bucket_stat private)"
echo "BUCKET_CREATED=$BUCKET_ID PRIVATE=$BUCKET_PRIVATE"
case "$BUCKET_PRIVATE" in
  False|false|None) : ;;
  *) echo "错误: bucket 仍非公开（private=$BUCKET_PRIVATE）"; exit 1 ;;
esac

# 绝不往一个已有内容的 bucket 上 sync：repo_id 打错一个字母就会静默污染别的库
pre_files="$(bucket_stat totalFiles)"
echo "BUCKET_FILES_AT_START=$pre_files BUCKET_BYTES_AT_START=$(bucket_stat size)"
if [ "$pre_files" != "0" ] && [ "$pre_files" != "None" ]; then
  if [ ! -f "$EXPORT_ROOT/logs/uploaded.marker" ]; then
    echo "错误: 目标 bucket 非空（files=$pre_files）且本地无上传记录，停止并交用户裁决"; exit 1
  fi
  echo "  bucket 非空但本地有上传记录，按续传处理"
fi
echo "阶段4完成"

# ---------------------------------------------------------------- 阶段 5：dry-run 预演
echo "阶段5开始：上传计划 dry-run（不写字节）"
hf sync "$STAGE" "$BUCKET" --dry-run --format json > "$EXPORT_ROOT/logs/sync-plan.jsonl" 2>&1 || {
  echo "  dry-run 返回非零，保留计划文件供查"; }
DEL_CNT=$(grep -c '"action"[[:space:]]*:[[:space:]]*"delete"' "$EXPORT_ROOT/logs/sync-plan.jsonl" || true)
echo "SYNC_PLAN_LINES=$(wc -l < "$EXPORT_ROOT/logs/sync-plan.jsonl") SYNC_PLAN_DELETES=$DEL_CNT"
[ "$DEL_CNT" = "0" ] || { echo "错误: dry-run 计划里出现 delete 动作，目标可能写错"; exit 1; }
echo "阶段5完成"

# ---------------------------------------------------------------- 阶段 6：分批上传
echo "阶段6开始：分批上传"
# **为什么分批**（2026-09-08 实测）：一次 sync 提交全部对象时，字节全部传完却在
# `_batch_bucket_files → session.new_upload_commit` 抛 TimeoutError，三次重试同因失败、
# bucket 始终 files=0。传输速率 546 MB/s，瓶颈不在带宽而在服务端一次性提交大量 Xet 对象的
# batch commit。已传字节不浪费——Xet 内容寻址，重跑时相同的块服务端已有。
upload_batch() {  # $1 = stage 下的子目录（同时也是 bucket 内前缀）；$2.. = --include 模式
  local sub="$1"; shift
  local args=() pat
  for pat in "$@"; do args+=(--include "$pat"); done
  # 重试 8 次而不是 3 次：实测该 commit 超时是**间歇性**的，不是确定性失败。
  local ok=0 attempt
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf sync "$STAGE/$sub" "$BUCKET/$sub" "${args[@]}"; then ok=1; break; fi
    echo "  批 [$sub ${*:-全部}] 第 $attempt 次中断，60s 后重试续传"; sleep 60
  done
  [ "$ok" = 1 ] || { echo "错误: 批 [$sub ${*:-全部}] 八次仍失败"; exit 1; }
  echo "  批完成: $sub (${#@} 项)"
}

# 标记「本轮已开始往这个 bucket 写」。阶段 4 的非空断言靠它区分首次跑与续跑。
touch "$EXPORT_ROOT/logs/uploaded.marker"

# --- 装箱：按字节预算切批，计划落盘（幂等：已存在则读盘，永不重算）---
# 分组漂移是续传的头号杀手：源目录任何变动都会让贪心装箱重排，已传的批和新分组对不上，
# --include 模式随之失配。首跑写盘、重跑读盘，分组才是稳定的。
PLAN="$EXPORT_ROOT/logs/upload_plan.json"
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
# 计划覆盖的文件数必须与 stage 实际文件数相等，否则有东西被漏在计划外
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

# --- bucket 根下的平文件：sync 的源必须是目录，这些用 cp 单传 ---
mapfile -t ROOTFILES < <(python3 -c "
import json
for f in json.load(open('$PLAN'))['root_files']: print(f)
")
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
echo "阶段6完成"

# ---------------------------------------------------------------- 阶段 7：计数层
echo "阶段7开始：计数层核对（REST API，异步滞后则复查一次）"
BUCKET_FILES="$(bucket_stat totalFiles)"; BUCKET_BYTES="$(bucket_stat size)"
if [ "$BUCKET_FILES" != "$LOCAL_FILES" ] || [ "$BUCKET_BYTES" != "$LOCAL_BYTES" ]; then
  echo "  首次不符（files=$BUCKET_FILES bytes=$BUCKET_BYTES），统计接口可能异步滞后，120s 后复查"
  sleep 120
  BUCKET_FILES="$(bucket_stat totalFiles)"; BUCKET_BYTES="$(bucket_stat size)"
fi
echo "BUCKET_FILES=$BUCKET_FILES BUCKET_BYTES=$BUCKET_BYTES"
# 计数层只是便宜的早停信号，真判据在阶段 9；但不符仍要停下来看
[ "$BUCKET_FILES" = "$LOCAL_FILES" ] || { echo "错误: 文件数不符 $BUCKET_FILES != $LOCAL_FILES"; exit 1; }
[ "$BUCKET_BYTES" = "$LOCAL_BYTES" ] || { echo "错误: 字节数不符 $BUCKET_BYTES != $LOCAL_BYTES"; exit 1; }
echo "阶段7完成"

# ---------------------------------------------------------------- 阶段 8：全量回读
echo "阶段8开始：全量回读到 $VERIFY（约 1.14 TB）"
# 不 rm -rf：hf sync 本身增量，保留已回读部分可让中断后续传
mkdir -p "$VERIFY"
# 余量断言（本次新增，仅因规模放大）：回读是**真实复制**（stage 是硬链接、不占额外空间，verify
# 不是），1.14 TB 要落到同一块 /scratch 上，且训练与其他产物也在写这块盘。余量不足时 hf sync 会在
# 半途 ENOSPC，留下一个残缺的 verify 树，所以先算清楚再动手。需要量 = stage 总字节 − verify 里已有
# 字节（续跑时已回读的部分不重复计），另留 200 GiB 安全垫。
VERIFY_HAVE=$(find "$VERIFY" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
READBACK_NEED=$(( LOCAL_BYTES - VERIFY_HAVE ))
[ "$READBACK_NEED" -ge 0 ] || READBACK_NEED=0
DF_AVAIL=$(df -B1 --output=avail "$VERIFY" | tail -n 1 | tr -d ' ')
DF_MARGIN=$(( 200 * 1024 * 1024 * 1024 ))
echo "DF_CHECK avail=$DF_AVAIL readback_need=$READBACK_NEED verify_have=$VERIFY_HAVE margin=$DF_MARGIN"
if [ "$DF_AVAIL" -lt $(( READBACK_NEED + DF_MARGIN )) ]; then
  echo "错误: 磁盘余量不足（avail=$DF_AVAIL < 回读 $READBACK_NEED + 200 GiB），停止回读交用户裁决"; exit 1
fi
ok=0
for attempt in 1 2 3; do
  if hf sync "$BUCKET" "$VERIFY"; then ok=1; break; fi
  echo "回读第 $attempt 次中断，60s 后重试续传"; sleep 60
done
[ "$ok" = 1 ] || { echo "回读三次仍失败"; exit 1; }
echo "阶段8完成"

# ---------------------------------------------------------------- 阶段 9：内容层验收
echo "阶段9开始：上传后 sha256（POST）+ 与上传前逐行对照"
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

# 成员层：content_hashes.tar 解包逐成员对源复算（tar 本身的 sha 只证明 tar 没变、不证明成员对）
uv run --no-sync python - "$SRC" "$VERIFY" <<'PY'
import hashlib, random, sys, tarfile
from pathlib import Path
src, verify = Path(sys.argv[1]), Path(sys.argv[2])
tar = verify / "control" / "content_hashes.tar"
if not tar.is_file():
    print("MEMBERS_SPOT=SKIP (无 content_hashes.tar)"); sys.exit(0)
rng = random.Random(20260914)
with tarfile.open(tar, "r") as tf:
    names = [ti.name for ti in tf if ti.isfile()]
    picked = rng.sample(names, min(200, len(names)))
    bad = []
    for nm in picked:
        got = tf.extractfile(nm).read()
        want = (src / "control" / "content_hashes" / nm).read_bytes()
        if got != want:
            bad.append(nm)
print(f"MEMBERS_SPOT={'PASS' if not bad else 'FAIL'} tar_members={len(names)} "
      f"checked={len(picked)} mismatches={len(bad)}")
if bad:
    print("差异成员:", bad[:10]); sys.exit(1)
PY
echo "RESULT=PASS"
echo "阶段9完成"

# ---------------------------------------------------------------- 阶段 10：源侧第三遍
echo "阶段10开始：源侧第三遍 sha256（证明硬链接未改动原件）"
# **必须并行。** 这段一度写成单进程 Python 逐文件 hashlib，实测只有 38 MB/s——full1600 的 479.5 GB
# 要跑 3.5 小时，而阶段 2 用 `xargs -P 16 sha256sum` 算同样的 479.5 GB 只花 74 秒（约 6.5 GB/s）。
# 慢 170 倍换不到任何信息增量，2026-09-14 那轮实测后改为复用 hash_tree。
#
# 为什么重算 stage 就等于重算源：stage 里每一项都是源文件的**硬链接**（同 inode），读 stage
# 就是读源文件本身。而且这样覆盖 stage 全部项（含 control/ 与 README），比只核发布清单的 4990 项更宽。
# 判据是与阶段 2 落盘的 pre 清单逐行相同——pre 是上传前算的、已固化，两者相等即证明原件在
# 整个上传与回读过程中一字未动。
hash_tree "$STAGE" "$TMPD/SHA256SUMS.src-after.txt"
cp "$TMPD/SHA256SUMS.src-after.txt" "$EXPORT_ROOT/SHA256SUMS.src-after.txt"
SRC_LINES=$(wc -l < "$TMPD/SHA256SUMS.src-after.txt")
if diff "$STAGE/SHA256SUMS.pre.txt" "$TMPD/SHA256SUMS.src-after.txt" > "$TMPD/src-diff.txt" 2>&1; then
  echo "SRC_UNCHANGED=OK checked=$SRC_LINES"
else
  echo "错误: 源侧 sha256 与上传前不一致（硬链接指向的原件被改动过）"
  head -10 "$TMPD/src-diff.txt"; exit 1
fi
echo "阶段10完成"

# ---------------------------------------------------------------- 阶段 11：公开性确认
echo "阶段11开始：公开性确认"
FINAL_PRIVATE="$(bucket_stat private)"
case "$FINAL_PRIVATE" in
  False|false|None) echo "BUCKET_PUBLIC=True" ;;
  *) echo "错误: bucket 不是公开的（private=$FINAL_PRIVATE）"; exit 1 ;;
esac

# 匿名读必须清掉 token，否则用自己的凭据拉下来，证明不了任何「公开」的事
rm -rf "$TMPD/anon"; mkdir -p "$TMPD/anon"
if env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HF_HUB_OFFLINE \
     uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf \
     sync "$BUCKET" "$TMPD/anon" --include 'README.md' \
   && [ -s "$TMPD/anon/README.md" ]; then
  echo "PUBLIC_ANON_READ=OK"
else
  echo "错误: 匿名读取失败，bucket 可能并未真正公开"; exit 1
fi
echo "阶段11完成"
echo "全部完成"
