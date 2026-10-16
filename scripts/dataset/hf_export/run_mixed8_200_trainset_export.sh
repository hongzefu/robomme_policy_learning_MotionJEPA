#!/usr/bin/env bash
# driver：把 16 任务 mixed8-200 VLA 训练库 v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f
# 上传到**公开** HF bucket HongzeFu/robomme-16task-mixed8-200-trainset-20261016-v1，做「上传前 /
# 上传后」两遍独立 sha256 逐行对照验收。detached tmux 里跑，每阶段幂等、整脚本可重跑续传。
#
# 计划：1016-mixed8-vla-plan.html 3.1 表 A1b 行、口径 29（内容）/ 30（bucket 名）/ 31（不做冷启动）/
# 38（分批回读）、G12（df 断言）、风险 17。
#
# 蓝本 run_v2_1600ep_dataset_export.sh（4 任务同类库，已 PASS）。与蓝本的差别，全部来自计划：
#   1. **没有蓝本阶段 0 的通配腾盘**（把 v1-store/exports 下其他导出的 verify 目录整批删掉）。那一步会误删 encoder latent
#      驱动的回读目录（风险 17、审查报告 H1）。本驱动只动自己的 EXPORT_ROOT。
#   2. **没有蓝本阶段 11 的冷启动彩排**（口径 31「8 同意」）：上传只占 CPU 与网络，不占 GPU。
#   3. **分批回读**（口径 38，用户 2026-10-16「a. 分批回读（推荐） 同意」）：蓝本阶段 7 的整桶
#      `hf sync "$BUCKET" "$VERIFY"` 与阶段 8 的整树 sha256 改成 batched_readback.py 逐批执行——
#      下载一批 → 逐文件 sha256 对上传前清单 → 该批 tar 成员抽样 → 删除该批 → 下一批。回读目录
#      任何时刻不超过一批（READBACK_BATCH_MAX_BYTES，默认 100 GiB）；全部批次汇总后 post 与 pre
#      逐行相同才打印 SHA256_PRE_POST_DIFF=0，另打印 READBACK_PEAK_BYTES=<n>（n 超上限即非 0 退出）。
#   4. **df 余量断言前移到打 tar 之前**（G12）：空闲 ≥ tar 实体体积 + 单批上限 + 0.4 TB，否则非 0 退出。
#   5. bucket 建时不带私有开关，建后与收尾两次断言 private==false（口径 21、审查报告 H7）。
#   6. 完成信号是末尾的「全部完成」（tmux 尾行另有 EXIT_CODE=0）；RESULT=PASS 只是中途的内容层判定行。
#   7. layout 用 pack_and_hash.py 的 mixed8-200-v1（与 v2-1600ep-v1 同布局，另断言 8 个抽样 features 在盘上）。
#
# 用法（正式运行在主检出：v1-store 与 secrets 在那里）：
#   SMOKE=1 bash scripts/dataset/hf_export/run_mixed8_200_trainset_export.sh  # 只做本地检查，不碰网络与凭据
#   bash scripts/dataset/hf_export/run_mixed8_200_trainset_export.sh          # 全量
# SMOKE=1：本地预检 → 源侧体检 → df 余量报告 → 打 2 个 data 片 → tar 成员与源逐字节对照 → README 渲染，
# 到此 0 退出。打出的分片记在 pack_progress.jsonl，全量跑时续用。
#
# 路径覆盖（只为离线 dry-run / 单测；正式运行一律不设）：M8X_LIB、M8X_NORM_STATS、M8X_RUNNER_SRC、
# M8X_EXPORT_ROOT。设了哪个就打印一行 OVERRIDE，全量模式下 EXPORT_ROOT 仍必须落在 v1-store/exports/ 下。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
cd "$REPO"

NAME="16task-mixed8-200-3200ep-e3c7564f"
LIB="${M8X_LIB:-$REPO/v1-store/datasets/$NAME}"
NORM_STATS="${M8X_NORM_STATS:-$REPO/v1-store/train-assets/mme_vla_suite/$NAME/robomme/norm_stats.json}"
RUNNER_SRC="${M8X_RUNNER_SRC:-$REPO/scripts/dataset/hf_export/mixed8_200_runner}"
EXPORT_ROOT="${M8X_EXPORT_ROOT:-$REPO/v1-store/exports/hf-mixed8-200-trainset}"
ALLOWLIST="$REPO/scripts/dataset/hf_export/hygiene_allowlist_mixed8_200_trainset.json"
README_SRC="$REPO/scripts/dataset/hf_export/mixed8_200_trainset_bucket_README.md"
BUCKET_ID="HongzeFu/robomme-16task-mixed8-200-trainset-20261016-v1"
BUCKET="hf://buckets/$BUCKET_ID"
STAGE="$EXPORT_ROOT/stage"
VERIFY="$EXPORT_ROOT/verify"
TMPD="$EXPORT_ROOT/tmp"
PLAN="$EXPORT_ROOT/logs/upload_plan.json"
RB_PLAN="$EXPORT_ROOT/logs/readback_plan.json"
RB_STATE="$EXPORT_ROOT/logs/readback"
# 单批回读上限：100 GiB（口径 38）。须不小于最大单对象（4 任务实测最大是 framesamp-8x8 的 part，
# 9.78–10.21 GB）；batched_readback.py plan 会现场核对，最大单对象放不进一批就在回读前报错退出。
READBACK_BATCH_MAX_BYTES="${READBACK_BATCH_MAX_BYTES:-107374182400}"
DF_MARGIN_BYTES=400000000000          # 0.4 TB（十进制，G12）
TAR_MEMBER_OVERHEAD=1024              # 每个 tar 成员的头 + 对齐填充上界（512 B 头 + ≤511 B 填充）
NPROC=16
# 网络重试口径（计划口径 27「重试 3 次」）：首试 + 重试 UPLOAD_RETRIES 次，仍失败即非 0 退出并点名失败对象，
# 由主会话写进 pending_decisions（status=fail）。上传、断点续传回读一律用这一个常量。
UPLOAD_RETRIES=3
MAX_ATTEMPTS=$((UPLOAD_RETRIES + 1))
SMOKE="${SMOKE:-0}"

for v in M8X_LIB M8X_NORM_STATS M8X_RUNNER_SRC M8X_EXPORT_ROOT; do
  [ -n "${!v:-}" ] && echo "OVERRIDE $v=${!v}"
done
if [ "$SMOKE" != 1 ]; then
  case "$EXPORT_ROOT" in
    "$REPO/v1-store/exports/"*) : ;;
    *) echo "错误: 全量模式下 EXPORT_ROOT 必须在 $REPO/v1-store/exports/ 下（实为 $EXPORT_ROOT）"; exit 1 ;;
  esac
fi

# AGENTS 第 14 条：缓存类环境变量逐项指向 v1-store / scratch，**禁止覆盖 HOME**
export HF_HOME="$REPO/v1-store/cache/hf"
export HF_XET_CACHE="$REPO/v1-store/cache/hf-xet"
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export UV_PYTHON_INSTALL_DIR=/scratch/hongze/.cache/uv-python
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1
# scripts/dataset/paths.sh 会 export HF_HUB_OFFLINE=1。本脚本不 source 它，但上游 shell 可能带进来
# ——不 unset 的话所有网络操作会静默走缓存，看起来"成功"实际什么都没传。
unset HF_HUB_OFFLINE || true

# 项目 .venv 的 huggingface_hub 是 0.32.3（openpi 钉死、不能升，且无 bucket 子命令），
# 故用 uvx 拉一份独立的 1.30.0 CLI，只在本脚本内使用，pyproject/uv.lock/.venv 一律不碰。
HF_CMD='uvx --python 3.11 --from huggingface_hub[cli]==1.30.0 hf'
hf() { uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf "$@"; }

# 精确字节/文件数/可见性只信 buckets REST API（`hf repos ls` 是人类可读值且异步滞后）
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

# 在 $1 目录内并行算 sha256，按路径排序（xargs -P 并行输出乱序，不排序无法 diff）
hash_tree() {  # $1 = 目录, $2 = 输出文件（绝对路径，不能落在被扫描目录内）
  ( cd "$1" && find . -type f -not -name 'SHA256SUMS.*' -printf '%P\n' | sort \
      | xargs -P "$NPROC" -n 16 sha256sum | sort -k 2 > "$2" )
}

mkdir -p "$STAGE" "$VERIFY" "$TMPD" "$EXPORT_ROOT/logs"

# ---------------------------------------------------------------- 阶段 0：本地预检
echo "阶段0开始：本地预检（只读，不删任何东西）"
for d in "$(dirname "$EXPORT_ROOT")" "$EXPORT_ROOT" "$STAGE" "$VERIFY"; do
  [ -L "$d" ] && { echo "错误: $d 是符号链接，拒绝写入"; exit 1; }
done
[ -d "$LIB" ] || { echo "错误: 源库不存在 $LIB"; exit 1; }
[ -f "$NORM_STATS" ] || { echo "错误: norm_stats 不存在 $NORM_STATS"; exit 1; }
for f in _train_common.sh train-512-80k.sh train-motion-80k.sh; do
  [ -f "$RUNNER_SRC/$f" ] || { echo "错误: runner 缺 $RUNNER_SRC/$f（A1-train 未合入？）"; exit 1; }
done
[ -f "$ALLOWLIST" ] || { echo "错误: 缺体检账本 $ALLOWLIST"; exit 1; }
[ -f "$README_SRC" ] || { echo "错误: 缺 README 模板 $README_SRC"; exit 1; }
# 口径 29 的内容清单：缺任何一项就不是计划要传的那个库
for sub in framesamp-8x8 framesamp motion motion-tokens wan-latents oracle meta source/data source/meta; do
  [ -d "$LIB/$sub" ] || { echo "错误: 源库缺 $sub/"; exit 1; }
done
# pack.lock 是读侧硬闸且无逃生阀，带进包等于交付一个异地一加载就拒绝的库
for sub in framesamp-8x8 framesamp motion; do
  [ -e "$LIB/$sub/meta/pack.lock" ] && { echo "错误: $sub/meta/pack.lock 存在"; exit 1; }
  # 读侧另一道硬闸：status != "verified" 即拒绝加载（README 对此有明文承诺，这里逐个断言）
  st="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('status'))" "$LIB/$sub/meta/store_meta.json")"
  [ "$st" = verified ] || { echo "错误: $sub/meta/store_meta.json 的 status=$st，不是 verified"; exit 1; }
done
echo "STORE_STATUS=verified stores=3 pack_lock=0"
# 回读目录只该装本驱动自己的半批（batched_readback.py 续跑时会清掉）。有别的东西说明路径接错了。
if find "$VERIFY" -mindepth 1 -maxdepth 1 -print -quit | grep -q . && [ ! -d "$RB_STATE" ]; then
  echo "错误: $VERIFY 非空但本驱动没有回读状态目录，拒绝在别人的目录上回读"; exit 1
fi
echo "阶段0完成"

# ---------------------------------------------------------------- 阶段 1：公开性体检（闸门）
echo "阶段1开始：源侧公开性体检（★ 闸门：不过不建库）"
# 放在 create 之前：目标是公开 bucket，一旦建出来就没有「发现问题还来得及」的窗口。
uv run --no-sync python scripts/dataset/hf_export/scan_hygiene.py \
  --root "$LIB" --extra "$NORM_STATS" --scope source \
  --allowlist "$ALLOWLIST" --report "$EXPORT_ROOT/logs/hygiene-source.json"
echo "阶段1完成"

# ---------------------------------------------------------------- 阶段 2：df 余量断言（打 tar 前）
echo "阶段2开始：df 余量断言（G12：空闲 ≥ tar 实体 + 单批回读上限 + 0.4 TB）"
# tar 实体 = source/data 与 oracle / wan-latents / motion-tokens 打成的 tar（其余全是硬链接，不占盘）。
# 体积按源目录现场 du -sb（表观字节）+ 每成员 1 KiB 头与填充上界估算；续跑时扣掉 stage 里已打好的 tar。
TAR_SRC_BYTES=0; TAR_SRC_FILES=0
for sub in source/data oracle wan-latents motion-tokens; do
  b=$(du -sb "$LIB/$sub" | cut -f1); n=$(find "$LIB/$sub" -type f | wc -l)
  echo "  $sub: bytes=$b files=$n"
  TAR_SRC_BYTES=$((TAR_SRC_BYTES + b)); TAR_SRC_FILES=$((TAR_SRC_FILES + n))
done
TAR_EST=$((TAR_SRC_BYTES + TAR_SRC_FILES * TAR_MEMBER_OVERHEAD))
TAR_HAVE=$(find "$STAGE" -type f -name '*.tar' -printf '%s\n' | awk '{s+=$1} END{print s+0}')
TAR_NEED=$((TAR_EST - TAR_HAVE)); [ "$TAR_NEED" -ge 0 ] || TAR_NEED=0
DF_AVAIL=$(df -B1 --output=avail "$EXPORT_ROOT" | tail -n 1 | tr -d ' ')
DF_NEED=$((TAR_NEED + READBACK_BATCH_MAX_BYTES + DF_MARGIN_BYTES))
echo "DF_CHECK avail=$DF_AVAIL need=$DF_NEED tar_est=$TAR_EST tar_have=$TAR_HAVE tar_need=$TAR_NEED" \
     "readback_batch_max=$READBACK_BATCH_MAX_BYTES margin=$DF_MARGIN_BYTES"
if [ "$DF_AVAIL" -lt "$DF_NEED" ]; then
  if [ "$SMOKE" = 1 ]; then
    echo "  警告: 此刻空闲不够全量（SMOKE 只报告不拦；全量起跑时同一断言会非 0 退出）"
  else
    echo "错误: 磁盘余量不足（avail=$DF_AVAIL < need=$DF_NEED），按计划第四节应急顺序处理后重跑"; exit 1
  fi
fi
echo "阶段2完成"

# ---------------------------------------------------------------- SMOKE：本地打 2 片 + README 渲染
render_readme() {  # $1 = 输出路径。从本库自己的 meta 现读锚点，填进 README 模板的 @@…@@ 占位
  uv run --no-sync python - "$LIB" "$NORM_STATS" "$README_SRC" "$1" <<'PY'
import hashlib, json, sys
from pathlib import Path
lib, norm, src, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
man = json.loads((lib / "meta/episode_manifest.json").read_text())
fs8 = json.loads((lib / "framesamp-8x8/meta/store_meta.json").read_text())
eps = man["episodes"]
tasks = man["canonical_order"]
if len(tasks) != 16:
    sys.exit(f"错误: canonical_order 应为 16 个任务，实为 {len(tasks)}")
spot = sorted(e["relpath"] for e in fs8.get("source_spot_sha256", {}).get("entries", [])
              if e["relpath"].startswith("features/"))
vals = {
    "TASKS": " / ".join(tasks),
    "NUM_TASKS": str(len(tasks)),
    "NUM_EPISODES": f"{len(eps):,}",
    "NUM_TIMESTEPS": f"{sum(e['num_timesteps'] for e in eps):,}",
    "NUM_EXEC": f"{fs8['num_exec_samples']:,}",
    "NUM_SPOT_FEATURES": str(len(spot)),
    "MANIFEST_SELF_SHA": man["sha256"],
    "MANIFEST_FILE_SHA": sha(lib / "meta/episode_manifest.json"),
    "FS8_META_SHA": sha(lib / "framesamp-8x8/meta/store_meta.json"),
    "FS4_META_SHA": sha(lib / "framesamp/meta/store_meta.json"),
    "MOTION_META_SHA": sha(lib / "motion/meta/store_meta.json"),
    "MOTION_INDEX_SHA": sha(lib / "motion/meta/motion_index.json"),
    "NORM_STATS_SHA": sha(norm),
}
text = src.read_text()
for k, v in vals.items():
    text = text.replace(f"@@{k}@@", v)
if "@@" in text:
    left = sorted({t.split("@@")[0] for t in text.split("@@")[1::2]})
    sys.exit(f"错误: README 仍有未填占位 {left}")
out.write_text(text)
print(f"README_RENDERED tasks={len(tasks)} episodes={len(eps)} exec={fs8['num_exec_samples']} "
      f"spot_features={len(spot)}")
PY
}

if [ "$SMOKE" = 1 ]; then
  echo "阶段S开始：smoke（本地打 2 个 data 片，跳过大 bin 与扁平目录，不碰网络）"
  uv run --no-sync python scripts/dataset/hf_export/pack_and_hash.py \
    --layout mixed8-200-v1 --lib "$LIB" --stage-root "$EXPORT_ROOT" \
    --bucket "$BUCKET" --norm-stats "$NORM_STATS" --workers 8 --limit-shards 2

  cat "$EXPORT_ROOT/logs/parts/"*.sha256 > "$TMPD/smoke-members.txt"
  uv run --no-sync python scripts/dataset/hf_export/verify_tar_members.py \
    --verify-root "$STAGE" --checksums "$TMPD/smoke-members.txt" --sample-tars 2

  # 上一步的 sha 与 tar 出自同一遍读，自洽但不能证明与源一致。这一步把成员解出来跟源文件真比一次。
  uv run --no-sync python - "$LIB" "$STAGE" <<'PY'
import random, sys, tarfile
from pathlib import Path
lib, stage = Path(sys.argv[1]), Path(sys.argv[2])
rng = random.Random(20261016)
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
if bad or not checked:
    print("差异成员:", bad[:10]); sys.exit(1)
PY
  render_readme "$TMPD/README.smoke.md"
  echo "阶段S完成"
  echo "SMOKE=PASS（未碰网络与凭据；产物由 pack_progress.jsonl 记账，全量跑时续用）"
  echo "全部完成"
  exit 0
fi

# ---------------------------------------------------------------- 阶段 3：凭据与身份
echo "阶段3开始：凭据与身份"
# 凭据只从 v1-store/secrets/hf.env 取。环境里原有的 HF_TOKEN 属于 yinpei-tri，写不进 HongzeFu
# 命名空间，必须被这里覆盖掉。
[ -f "$REPO/v1-store/secrets/hf.env" ] || { echo "错误: 缺少 v1-store/secrets/hf.env"; exit 1; }
set -a; . "$REPO/v1-store/secrets/hf.env"; set +a
[ -n "${HF_TOKEN:-}" ] || { echo "错误: hf.env 未提供 HF_TOKEN"; exit 1; }
command -v uvx >/dev/null || { echo "错误: PATH 里找不到 uvx。PATH=$PATH"; exit 1; }
# whoami 的输出格式随是否有 tty 而变：--format agent 钉死格式，再整体合并后子串匹配
WHO="$(hf auth whoami --format agent 2>&1 | tr '\n' ' ' | tr -s ' ')"
echo "  whoami: $WHO"
case "$WHO" in *HongzeFu*) : ;; *) echo "错误: 身份不是 HongzeFu（$WHO）"; exit 1 ;; esac
echo "HF_WHOAMI=HongzeFu"
echo "阶段3完成"

# ---------------------------------------------------------------- 阶段 4：建 bucket（公开）
echo "阶段4开始：创建公开 bucket $BUCKET_ID"
if bucket_stat totalFiles >/dev/null 2>&1; then
  echo "  bucket 已存在，跳过创建（幂等重跑）"
else
  hf buckets create "$BUCKET_ID"     # 不带私有开关即公开（口径 21）
fi
# 显式设一次 public，让终态成为结构性成立的事实而不是对 create 默认值的推断。settings 幂等。
hf buckets settings "$BUCKET_ID" --public
assert_public "BUCKET_CREATED=$BUCKET_ID"
# 绝不往一个已有内容的 bucket 上 sync：repo_id 打错一个字母就会静默污染别的库
pre_files="$(bucket_stat totalFiles)"; pre_size="$(bucket_stat size)"
echo "BUCKET_FILES_AT_START=$pre_files BUCKET_BYTES_AT_START=$pre_size"
if [ "$pre_files" != "0" ] && [ "$pre_files" != "None" ]; then
  if [ ! -f "$EXPORT_ROOT/logs/uploaded.marker" ]; then
    echo "错误: 目标 bucket 非空（files=$pre_files）且本地无上传记录，停止并交用户裁决"; exit 1
  fi
  echo "  bucket 非空但本地有上传记录，按续传处理"
fi
echo "阶段4完成"

# ---------------------------------------------------------------- 阶段 5：打包
echo "阶段5开始：打包 → $STAGE（断点续跑，已完成分片自动跳过）"
uv run --no-sync python scripts/dataset/hf_export/pack_and_hash.py \
  --layout mixed8-200-v1 --lib "$LIB" --stage-root "$EXPORT_ROOT" \
  --bucket "$BUCKET" --norm-stats "$NORM_STATS" --workers 8
echo "阶段5完成"

# ---------------------------------------------------------------- 阶段 6：stage 收尾 + 上传前 sha256
echo "阶段6开始：README + runner + 断言 + stage 侧第二遍体检 + 上传前 sha256"
rm -f "$STAGE/README.md"
render_readme "$STAGE/README.md"
mkdir -p "$STAGE/runner"
cp "$RUNNER_SRC"/_train_common.sh "$RUNNER_SRC"/train-512-80k.sh "$RUNNER_SRC"/train-motion-80k.sh \
   "$STAGE/runner/"

# stage 里只能有实体文件：上传端遍历不跟随符号链接**目录**，其内容会被静默整体漏传
if find "$STAGE" -type l -print | grep -q .; then
  echo "错误: stage 内出现符号链接"; find "$STAGE" -type l -print; exit 1
fi
if find "$STAGE" -name 'pack.lock' -print | grep -q .; then
  echo "错误: stage 内出现 pack.lock"; exit 1
fi
# 口径 29 的顶层内容清单（stage 侧）：多一项少一项都停
EXPECT_TOP="README.md assets checksums framesamp framesamp-8x8 manifest meta motion motion-tokens_tars oracle_tars runner source wan-latents_tars"
GOT_TOP="$(cd "$STAGE" && find . -mindepth 1 -maxdepth 1 -not -name 'SHA256SUMS.*' -printf '%P\n' | sort | tr '\n' ' ' | sed 's/ $//')"
[ "$GOT_TOP" = "$EXPECT_TOP" ] || { echo "错误: stage 顶层内容不符"; echo "  期望 $EXPECT_TOP"; echo "  实为 $GOT_TOP"; exit 1; }
GOT_SRC="$(cd "$STAGE/source" && find . -mindepth 1 -maxdepth 1 -printf '%P\n' | sort | tr '\n' ' ' | sed 's/ $//')"
[ "$GOT_SRC" = "data_tars features meta" ] || { echo "错误: stage/source 下应恰为 data_tars features meta，实为 $GOT_SRC"; exit 1; }
NSPOT=$(find "$STAGE/source/features" -type f | wc -l)
[ "$NSPOT" = 8 ] || { echo "错误: stage/source/features 应恰有 8 个抽样文件，实为 $NSPOT"; exit 1; }
echo "STAGE_LAYOUT=OK spot_features=$NSPOT"

# 第二遍体检扫 stage：这一遍才看得到本驱动新写的 README、runner、manifest、checksums
uv run --no-sync python scripts/dataset/hf_export/scan_hygiene.py \
  --root "$STAGE" --scope stage \
  --allowlist "$ALLOWLIST" --report "$EXPORT_ROOT/logs/hygiene-stage.json"

hash_tree "$STAGE" "$TMPD/SHA256SUMS.pre.txt"
mv "$TMPD/SHA256SUMS.pre.txt" "$STAGE/SHA256SUMS.pre.txt"
LOCAL_FILES=$(find "$STAGE" -type f | wc -l)
LOCAL_BYTES=$(find "$STAGE" -type f -printf '%s\n' | awk '{s+=$1} END{print s}')
PRE_LINES=$(wc -l < "$STAGE/SHA256SUMS.pre.txt")
echo "LOCAL_FILES=$LOCAL_FILES LOCAL_BYTES=$LOCAL_BYTES PRE_LINES=$PRE_LINES"
[ "$((PRE_LINES + 1))" = "$LOCAL_FILES" ] || { echo "错误: 清单行数+1 != stage 文件数"; exit 1; }
echo "阶段6完成"

# ---------------------------------------------------------------- 阶段 7：分批上传
echo "阶段7开始：分批上传"
# **为什么分批**（2026-09-08 实测）：一次 sync 提交全部对象时字节全部传完，却在
# `_batch_bucket_files → session.new_upload_commit` 抛 TimeoutError，bucket 始终 files=0。
# 瓶颈在服务端一次性提交大量 Xet 对象的 batch commit。阈值经验值：2 GB×5 稳、2 GB×10 挂。
upload_batch() {  # $1 = stage 下的子目录（同时也是 bucket 内前缀）；$2.. = --include 模式
  local sub="$1"; shift
  local args=() pat ok=0 attempt
  for pat in "$@"; do args+=(--include "$pat"); done
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    if hf sync "$STAGE/$sub" "$BUCKET/$sub" "${args[@]}"; then ok=1; break; fi
    echo "  批 [$sub] 第 $attempt/$MAX_ATTEMPTS 次尝试中断"
    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then echo "  60s 后重试续传"; sleep 60; fi
  done
  [ "$ok" = 1 ] || { echo "UPLOAD_FAIL object=$sub/ files=$* attempts=$MAX_ATTEMPTS"; echo "错误: 批 [$sub] 首试 + 重试 $UPLOAD_RETRIES 次仍失败"; exit 1; }
  echo "  批完成: $sub"
}

touch "$EXPORT_ROOT/logs/uploaded.marker"
# 装箱计划落盘、幂等读盘：分组漂移是续传的头号杀手（--include 模式会失配）。分批回读也基于这份计划。
if [ -f "$PLAN" ]; then
  echo "  上传计划已存在，读盘复用（不重算装箱）"
else
  uv run --no-sync python scripts/dataset/hf_export/plan_upload_batches.py --stage "$STAGE" --out "$PLAN"
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
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    if hf buckets cp "$STAGE/$f" "$BUCKET/$f"; then ok=1; break; fi
    echo "  $f 第 $attempt/$MAX_ATTEMPTS 次尝试中断"
    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then echo "  60s 后重试"; sleep 60; fi
  done
  [ "$ok" = 1 ] || { echo "UPLOAD_FAIL object=$f attempts=$MAX_ATTEMPTS"; echo "错误: 根文件 $f 首试 + 重试 $UPLOAD_RETRIES 次仍失败"; exit 1; }
done
echo "阶段7完成"

# ---------------------------------------------------------------- 阶段 8：计数层
echo "阶段8开始：计数层核对（REST API，异步滞后则复查一次）"
BUCKET_FILES="$(bucket_stat totalFiles)"; BUCKET_BYTES="$(bucket_stat size)"
if [ "$BUCKET_FILES" != "$LOCAL_FILES" ] || [ "$BUCKET_BYTES" != "$LOCAL_BYTES" ]; then
  echo "  首次不符（files=$BUCKET_FILES bytes=$BUCKET_BYTES），统计接口可能异步滞后，120s 后复查"
  sleep 120
  BUCKET_FILES="$(bucket_stat totalFiles)"; BUCKET_BYTES="$(bucket_stat size)"
fi
echo "BUCKET_FILES=$BUCKET_FILES BUCKET_BYTES=$BUCKET_BYTES"
[ "$BUCKET_FILES" = "$LOCAL_FILES" ] || { echo "错误: 文件数不符 $BUCKET_FILES != $LOCAL_FILES"; exit 1; }
[ "$BUCKET_BYTES" = "$LOCAL_BYTES" ] || { echo "错误: 字节数不符 $BUCKET_BYTES != $LOCAL_BYTES"; exit 1; }
echo "阶段8完成"

# ---------------------------------------------------------------- 阶段 9：分批回读 + 内容层验收
echo "阶段9开始：分批回读（单批 ≤ $READBACK_BATCH_MAX_BYTES B）+ 逐批 sha256 对照 + tar 成员抽样"
# 回读计划由阶段 7 已落盘的上传装箱计划合并相邻批得到，同样落盘复用；最大单对象 > 单批预算时这里就停。
uv run --no-sync python scripts/dataset/hf_export/batched_readback.py plan \
  --upload-plan "$PLAN" --stage "$STAGE" --out "$RB_PLAN" --max-bytes "$READBACK_BATCH_MAX_BYTES"
# 逐批：下载本批 → 逐文件 sha256 对 SHA256SUMS.pre.txt 该批的行 → 抽本批 1 个 tar 逐成员对
# checksums/sha256-source-files.txt → 删本批 → 下一批。全部批次汇总后 post 与 pre 逐行 diff，
# 为空才打印 SHA256_PRE_POST_DIFF=0；同时打印 READBACK_PEAK_BYTES=<n>，n 超上限非 0 退出。
uv run --no-sync python scripts/dataset/hf_export/batched_readback.py run \
  --readback-plan "$RB_PLAN" --pre "$STAGE/SHA256SUMS.pre.txt" \
  --verify "$VERIFY" --state "$RB_STATE" --post-out "$EXPORT_ROOT/SHA256SUMS.post.txt" \
  --max-bytes "$READBACK_BATCH_MAX_BYTES" \
  --members-checksums "$STAGE/checksums/sha256-source-files.txt" \
  --sample-tars-per-batch 1 --seed 20261016 --nproc "$NPROC" \
  --source hf --bucket "$BUCKET" --hf-cmd "$HF_CMD" --retries "$UPLOAD_RETRIES" --retry-sleep 60
echo "RESULT=PASS"
echo "阶段9完成"

# ---------------------------------------------------------------- 阶段 10：源侧第三遍
echo "阶段10开始：收尾断言（源侧第三遍 sha256，证明硬链接未改动原件）"
# 只核直传件（硬链接那些）——tar 分片没有对应的单个源文件，其内容由 checksums/sha256-source-files.txt
# 的逐成员清单覆盖。hash_tree 用 find -printf '%P'，清单里的路径**不带 ./ 前缀**。
grep -v -E "  (README\.md|manifest/|checksums/|runner/)" "$STAGE/SHA256SUMS.pre.txt" \
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
echo "阶段10完成"

# ---------------------------------------------------------------- 阶段 11：公开性确认
echo "阶段11开始：公开性确认（private==false + 匿名读）"
# verify_model_repo.py 是给 model repo 的、口径是「必须 private」，方向相反，这里不复用它。
assert_public "BUCKET_PUBLIC_FINAL"
echo "BUCKET_PUBLIC=True"
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
