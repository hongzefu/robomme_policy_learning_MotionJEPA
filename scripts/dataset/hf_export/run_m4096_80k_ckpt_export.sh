#!/usr/bin/env bash
# driver：把训练 run v2-1600ep-m64x8x8-modul-b128-80k（modulation + **motion 关闭**、
# 64 帧 × 8×8 = budget 4096）的全部 16 个 orbax checkpoint
# （5000…75000 共 15 个 + 收尾 79999，316 文件 / 190,052,719,306 B ≈ 177 GiB）
# 上传到 HF **private** bucket HongzeFu/robomme-vla-modul-4096-80k-v1，
# 并做「上传前 / 上传后」两遍独立 sha256 逐行对照验收。
#
# 蓝本是同目录 run_m2048_80k_ckpt_export.sh（同一 config 条目 mme_vla_suite_b128_80k、
# 同样 motion 关闭、同样 16 个 checkpoint、同样 <run>.driver.log），六阶段骨架照抄。
# 与它的**四处差异**：
#   1. run / bucket 名、train-cmd 里的 history YAML（64frame）、README 模板换成 4096。
#   2. **删掉趟 A 与 pgrep 判定，只走趟 B。** 蓝本用 pgrep "[t]rain\.py mme_vla_suite_b128_80k"
#      判断「本 run 还在训练」，但 1024 run 用的是同一个 config 名：本脚本起跑时 1024 正在训练，
#      照抄会把它误判成 4096 在训练 => 走趟 A，丢掉最大 step 79999、不做回读却打印 PASS=A。
#      改为按本 run 自身硬断言训练已完成：driver.log 尾部 EXIT_CODE=0，且完成器
#      <run>.completed.json 的 status=PASS、run_name/budget 对得上、checkpoints 恰 16 个。
#   3. **限流由 EXPORT_THROTTLE 决定（默认 1）**，不再由「训练是否在跑」决定：
#      1 => ionice -c 3 + nice 19 + NPROC=2（与同机在跑的 1024 训练共存）；0 => 不限流、NPROC=8。
#   4. **_run-meta 多带完成证据**：completed.json 与 runs/<run>/ 下的保存现场记录
#      （final/、run_meta.json、runtime.json、metrics.jsonl），合计 <0.5 MB。
# 另外：本脚本允许从 -temp 开发副本执行（1024 训练期间主副本锁定）。开发副本的 v1-store
# 是指向主副本的 symlink，这里一律先 readlink -f 成实体路径 STORE 再用，写入落点与在主副本跑完全相同；
# 本脚本不含任何 --force / rmtree。
#
# 沿用蓝本的四条做法（都是本环境实测踩出来的）：
#   - **stage 走硬链接**：stage 与源同在 /dev/md0，ln -f 额外占盘为 0。
#     代价是必须在收尾阶段重算源文件 sha256，断言没被连带改坏（阶段 6）。
#   - **不打 tar**：保持「下载下来直接 orbax.restore」这个性质。
#   - **不复制 token**：凭据走 v1-store/secrets/hf.env 注入 HF_TOKEN 环境变量。
#   - **按 step 目录分批 sync**：一次提交 >20 GB 会在 new_upload_commit 抛 TimeoutError。
#
# stage 为幂等增量（只 mkdir -p + ln -f，不删），每个 step 目录落一份**独立的**
# pre 清单分片 tmp/pre-parts/<step>.txt，全局 SHA256SUMS.pre.txt 由分片合成。中断后原样重跑即续传。
# 每 step 文件数 20–25 不等（ocdbt 的 d/ 分块数随内容变），故不硬编码总文件数。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
cd "$REPO"

RUN_NAME="v2-1600ep-m64x8x8-modul-b128-80k"
BUDGET=4096
BUCKET_ID="HongzeFu/robomme-vla-modul-4096-80k-v1"
BUCKET="hf://buckets/$BUCKET_ID"
# v1-store 解析成实体路径：主副本下它本身就是实体目录，-temp 开发副本下它是指向主副本的 symlink。
# 下面所有数据路径都走 $STORE，两种副本执行时读写的是同一批实体文件。
STORE="$(readlink -f "$REPO/v1-store")"
# 中间层是 mme_vla_suite_b128_80k（1024/2048/4096 三档与 motion 组同一 config 条目，run 并列其下）
SRC="$STORE/train-runs/mme_vla_suite_b128_80k/$RUN_NAME"
EXPORT_ROOT="$STORE/exports/hf-ckpt-$RUN_NAME"
STAGE="$EXPORT_ROOT/stage"
VERIFY="$EXPORT_ROOT/verify"
TMPD="$EXPORT_ROOT/tmp"
PARTS="$TMPD/pre-parts"
# 本 run 由 run_modul4096_then1024.sh 顺序调度器起跑，日志带 .driver 中缀
TRAIN_LOG="$STORE/logs/$RUN_NAME.driver.log"
# 完成器产物：顺序调度器批次 modul-sweep-20260923-c 写下的完成 JSON 与保存现场记录目录
COMPLETED_JSON="$STORE/bench/modul-budget-sweep/modul-sweep-20260923-c/$RUN_NAME.completed.json"
RECORD_DIR="$STORE/bench/modul-budget-sweep/runs/$RUN_NAME"
RECORD_ITEMS=(final run_meta.json runtime.json metrics.jsonl)
WANDB_ID_FILE="$SRC/wandb_id.txt"
EXPECT_STEPS=16
EXPECT_LAST="79999"
# 根下 sidecar：训练启动时一次性写完的静态文件
ROOT_SIDECARS=(history_config.resolved.sha256 history_config.resolved.yaml history_config.txt motion_provenance.json wandb_id.txt)

# AGENTS 第 14 条：缓存类环境变量逐项指向 v1-store / scratch，**禁止覆盖 HOME**
export HF_HOME="$STORE/cache/hf"
export HF_XET_CACHE="$STORE/cache/hf-xet"
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export UV_PYTHON_INSTALL_DIR=/scratch/hongze/.cache/uv-python
export UV_LINK_MODE=copy
unset HF_HUB_OFFLINE || true

# ---------------------------------------------------------------- 限流档位
# 不看「有没有训练在跑」（同 config 的 1024 run 会让 pgrep 误判），只看 EXPORT_THROTTLE。
# 同机 8 卡训练的 dataloader 对 /dev/md0 读带宽敏感：ionice class 3（idle）只在磁盘空闲时调度，
# nice 19 让 sha256 不抢 CPU。THROTTLE 包住 hf()（上传与回读）和全部 sha256，子进程继承。
EXPORT_THROTTLE="${EXPORT_THROTTLE:-1}"
case "$EXPORT_THROTTLE" in
  1) THROTTLE=(ionice -c 3 nice -n 19); NPROC=2; echo "THROTTLE_MODE=idle NPROC=$NPROC" ;;
  0) THROTTLE=(); NPROC=8; echo "THROTTLE_MODE=off NPROC=$NPROC" ;;
  *) echo "错误: EXPORT_THROTTLE 只能是 0 或 1，实为 $EXPORT_THROTTLE"; exit 1 ;;
esac

# 凭据：只从 v1-store/secrets/hf.env 取（600、在 .gitignore 的 /v1-store/ 下）。
# 环境里原有的 HF_TOKEN 可能属于别的账号，写不进 HongzeFu 命名空间，必须被这里覆盖掉。
[ -f "$STORE/secrets/hf.env" ] || { echo "错误: 缺少 v1-store/secrets/hf.env"; exit 1; }
set -a; . "$STORE/secrets/hf.env"; set +a
[ -n "${HF_TOKEN:-}" ] || { echo "错误: hf.env 未提供 HF_TOKEN"; exit 1; }

# 项目 .venv 的 huggingface_hub 是 0.32.3，**没有 buckets 子命令**，且由 openpi 依赖钉住
# 不能升。用 uvx 拉一份完全独立的 1.30.0 CLI，只在本脚本内使用，
# 项目 pyproject.toml / uv.lock / .venv 一律不碰。
hf() { "${THROTTLE[@]}" uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf "$@"; }

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

# 在 STAGE 内对相对路径 $1（一个子目录名，或 . 下的若干文件）算 sha256，写到 $2。
# 输出按路径排序（xargs -P 并行会乱序，不排序则无法与 post 清单逐行 diff）。
hash_subdir() {  # $1 = STAGE 下的子目录名, $2 = 输出分片文件（绝对路径）
  ( cd "$STAGE" && find "$1" -type f -printf '%p\n' | sort \
      | "${THROTTLE[@]}" xargs -P "$NPROC" -n 16 sha256sum | sort -k 2 > "$2" )
}
hash_files() {  # $1 = 输出分片文件（绝对路径）; 其余参数 = STAGE 下的相对文件路径
  local out="$1"; shift
  ( cd "$STAGE" && printf '%s\n' "$@" | sort \
      | "${THROTTLE[@]}" xargs -P "$NPROC" -n 16 sha256sum | sort -k 2 > "$out" )
}
# 阶段 5/6 用：整棵树重算。-not -name 'SHA256SUMS.*' 让清单文件自身不进清单。
hash_tree() {  # $1 = 目录, $2 = 输出文件（绝对路径，不能落在被扫描目录内）
  ( cd "$1" && find . -type f -not -name 'SHA256SUMS.*' -printf '%P\n' | sort \
      | "${THROTTLE[@]}" xargs -P "$NPROC" -n 16 sha256sum | sort -k 2 > "$2" )
}

# 带重试的整目录上传。重试 8 次而不是 3 次：实测 commit 超时是**间歇性**的，不是确定性失败。
sync_dir() {  # $1 = STAGE 下的子目录（同时也是 bucket 内前缀）
  local sub="$1" ok=0 attempt
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf sync "$STAGE/$sub" "$BUCKET/$sub"; then ok=1; break; fi
    echo "  批 [$sub] 第 $attempt 次中断，60s 后重试续传"; sleep 60
  done
  [ "$ok" = 1 ] || { echo "错误: 批 [$sub] 八次仍失败"; exit 1; }
}
cp_file() {  # $1 = STAGE 下的相对文件路径（sync 的源必须是目录，根下平文件走 cp）
  local f="$1" ok=0 attempt
  [ -f "$STAGE/$f" ] || { echo "错误: stage 缺少文件 $f"; exit 1; }
  for attempt in 1 2 3 4 5 6 7 8; do
    if hf buckets cp "$STAGE/$f" "$BUCKET/$f"; then ok=1; break; fi
    echo "  $f 第 $attempt 次中断，60s 后重试"; sleep 60
  done
  [ "$ok" = 1 ] || { echo "错误: $f 八次仍失败"; exit 1; }
}

# ---------------------------------------------------------------- 阶段 0
echo "阶段0开始：预检 + 建 bucket"
# 落点必须是本环境实体目录（AGENTS 第 13/14 条）：STORE 已 readlink -f，必须是 /scratch/hongze 下的
# 实体目录；它下面的 exports 与导出根不得再是 symlink（主副本 v1-store 下没有任何 symlink 外链）。
case "$STORE" in /scratch/hongze/*) : ;; *) echo "错误: v1-store 解析到 $STORE，不在 /scratch/hongze 下"; exit 1 ;; esac
[ -d "$STORE" ] && [ ! -L "$STORE" ] || { echo "错误: $STORE 不是实体目录"; exit 1; }
for d in "$STORE/exports" "$EXPORT_ROOT"; do
  [ -L "$d" ] && { echo "错误: $d 是符号链接，拒绝写入"; exit 1; }
done
echo "  STORE=$STORE"
[ -d "$SRC" ] || { echo "错误: 源目录不存在 $SRC"; exit 1; }
# 训练必须已正常结束，且完成器判 PASS——按本 run 自身判，不看进程（同 config 的 1024 在跑）
tail -n 5 "$TRAIN_LOG" | grep -qx 'EXIT_CODE=0' \
  || { echo "错误: $TRAIN_LOG 尾部没有 EXIT_CODE=0，训练未正常结束"; exit 1; }
python3 - "$COMPLETED_JSON" "$RUN_NAME" "$BUDGET" "$EXPECT_STEPS" "$EXPECT_LAST" <<'PY' \
  || { echo "错误: 完成记录 $COMPLETED_JSON 不满足要求"; exit 1; }
import json, sys
path, run, budget, n, last = sys.argv[1:]
d = json.load(open(path))
steps = sorted(int(s) for s in d.get("checkpoints") or [0])   # 完成器写的是 step 整数列表
errs = []
if d.get("status") != "PASS": errs.append(f"status={d.get('status')}")
if d.get("run_name") != run: errs.append(f"run_name={d.get('run_name')}")
if int(d.get("budget", -1)) != int(budget): errs.append(f"budget={d.get('budget')}")
if len(steps) != int(n) or steps[-1] != int(last): errs.append(f"checkpoints={steps}")
if errs:
    sys.exit("  完成记录不符: " + "; ".join(errs))
print(f"  完成记录: status=PASS run={run} budget={budget} checkpoints={len(steps)} final={steps[-1]}")
PY
echo "TRAIN_COMPLETED=OK"
[ "$STAGE" != "$SRC" ] || { echo "错误: STAGE 与 SRC 相同，拒绝继续"; exit 1; }
case "$STAGE" in "$EXPORT_ROOT"/*) : ;; *) echo "错误: STAGE 不在 EXPORT_ROOT 下"; exit 1 ;; esac
# 本 run motion 必须是关闭的——若这里读出 true，说明 RUN_NAME 指错了 run
MOTION_ENABLED="$(python3 -c "
import json
print(json.load(open('$SRC/motion_provenance.json'))['motion_enabled'])
")"
[ "$MOTION_ENABLED" = "False" ] || { echo "错误: 本脚本只导出 motion 关闭的 run，实际 motion_enabled=$MOTION_ENABLED"; exit 1; }
echo "  motion_enabled=False 确认，本 bucket 不含 _motion-encoder/"
mkdir -p "$STAGE" "$VERIFY" "$TMPD" "$PARTS" "$EXPORT_ROOT/logs" "$EXPORT_ROOT/logs/uploaded" "$HF_XET_CACHE"

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
echo "阶段1开始：枚举 step 目录 + 增量 stage（硬链接）+ 分片 sha256"
mapfile -t ALL < <( cd "$SRC" && find . -mindepth 1 -maxdepth 1 -type d -printf '%P\n' | sort -n )
# 完整性守门：num_train_steps=80_000 / save_interval=keep_period=5_000
# ⇒ 5000…75000 共 15 个 + 收尾 79999，合计 16 个。训练已结束（阶段 0 已断言），全部可传。
[ "${#ALL[@]}" = "$EXPECT_STEPS" ] || { echo "错误: step 目录数应为 $EXPECT_STEPS，实为 ${#ALL[@]}（${ALL[*]}）"; exit 1; }
[ "${ALL[-1]}" = "$EXPECT_LAST" ] || { echo "错误: 末尾 step 应为 $EXPECT_LAST，实为 ${ALL[-1]}"; exit 1; }
SAFE=( "${ALL[@]}" )
echo "  本趟上传 ${#SAFE[@]} 个 step: ${SAFE[*]}"

stage_subtree() {  # $1 = SRC 下的相对子目录
  local sub="$1"
  ( cd "$SRC/$sub" && find . -type d -printf '%P\n' ) | while read -r d; do
    mkdir -p "$STAGE/$sub/${d}"
  done
  ( cd "$SRC/$sub" && find . -type f -printf '%P\n' ) | while read -r f; do
    ln -f "$SRC/$sub/$f" "$STAGE/$sub/$f"
  done
}

for s in "${SAFE[@]}"; do
  if [ -f "$PARTS/$s.txt" ]; then echo "  step $s 已 stage 且已有分片清单，跳过"; continue; fi
  echo "  stage step $s"
  stage_subtree "$s"
  hash_subdir "$s" "$PARTS/.$s.txt.partial"
  mv "$PARTS/.$s.txt.partial" "$PARTS/$s.txt"   # 原子落盘：中断不会留下半份分片
done

# 根下 sidecar（训练启动时一次性写完的静态文件）
for f in "${ROOT_SIDECARS[@]}"; do
  [ -f "$SRC/$f" ] || { echo "错误: 源缺少 sidecar $f"; exit 1; }
  ln -f "$SRC/$f" "$STAGE/$f"
done
[ -f "$PARTS/_root.txt" ] || {
  hash_files "$PARTS/._root.txt.partial" "${ROOT_SIDECARS[@]}"
  mv "$PARTS/._root.txt.partial" "$PARTS/_root.txt"
}

# stage 里只能有实体文件：上传端遍历是 os.walk(followlinks=False)，
# 符号链接**目录**的内容会被静默整体漏传
if find "$STAGE" -type l -print | grep -q .; then
  echo "错误: stage 内出现符号链接"; find "$STAGE" -type l -print; exit 1
fi
echo "阶段1完成"

# ---------------------------------------------------------------- 阶段 2
echo "阶段2开始：按 step 目录分批上传"
# **为什么分批**（2026-09-08 实测）：一次 sync 提交大量对象时，字节全部传完却在
# `_batch_bucket_files → session.new_upload_commit` 抛 TimeoutError，重试同因失败、
# bucket 始终 files=0。失败阈值在**批内字节量**：1.62 GB×10=16.2 GB 过、2 GB×5=10 GB 稳、
# 2 GB×10=20 GB 挂。本次每个 step 目录 ~11 GiB / 20–25 文件，落在已验证会过的量级。
# 已传字节不浪费——Xet 内容寻址，重跑时相同的块服务端已有。
touch "$EXPORT_ROOT/logs/uploaded.marker"

for ((b = 0; b < ${#SAFE[@]}; b++)); do
  SUB="${SAFE[$b]}"
  DONE="$EXPORT_ROOT/logs/uploaded/$SUB.done"
  if [ -f "$DONE" ]; then echo "  [$((b+1))/${#SAFE[@]}] step $SUB 已完成，跳过"; continue; fi
  BBYTES=$(find "$STAGE/$SUB" -type f -printf '%s\n' | awk '{s+=$1} END{print s}')
  BFILES=$(find "$STAGE/$SUB" -type f | wc -l)
  echo "  [$((b+1))/${#SAFE[@]}] step $SUB  $BFILES 件 / $BBYTES 字节"
  sync_dir "$SUB"
  touch "$DONE"
  echo "  批完成: $SUB"
done

# 根下 sidecar：sync 的源必须是目录，这些走 cp
if [ ! -f "$EXPORT_ROOT/logs/uploaded/_root.done" ]; then
  echo "  上传根下 ${#ROOT_SIDECARS[@]} 个 sidecar"
  for f in "${ROOT_SIDECARS[@]}"; do cp_file "$f"; done
  touch "$EXPORT_ROOT/logs/uploaded/_root.done"
fi
echo "阶段2完成"

# ---------------------------------------------------------------- 阶段 2b
echo "阶段2b开始：补齐 _run-meta（训练日志 + wandb + 完成证据）与 README、pre 清单"

# _run-meta/：训练已退出（阶段 0 已断言 EXIT_CODE=0 与完成记录 PASS），这些文件不会再被追加写。
WANDB_ID="$(tr -d '[:space:]' < "$WANDB_ID_FILE")"
WANDB_SRC="$(find "$STORE/logs/wandb/wandb" -maxdepth 1 -type d -name "run-*-$WANDB_ID" | head -1)"
[ -n "$WANDB_SRC" ] || { echo "错误: 找不到 wandb run 目录（id=$WANDB_ID）"; exit 1; }
echo "  wandb run 目录: $WANDB_SRC"
mkdir -p "$STAGE/_run-meta"
ln -f "$TRAIN_LOG" "$STAGE/_run-meta/train.log"
( cd "$WANDB_SRC/.." && find "$(basename "$WANDB_SRC")" -type d -printf '%p\n' ) | while read -r d; do
  mkdir -p "$STAGE/_run-meta/wandb/$d"
done
( cd "$WANDB_SRC/.." && find "$(basename "$WANDB_SRC")" -type f -printf '%p\n' ) | while read -r f; do
  ln -f "$WANDB_SRC/../$f" "$STAGE/_run-meta/wandb/$f"
done
# 完成证据：完成器 JSON + 保存现场记录（final/ 里是末步 EMA 逐叶摘要与尾窗标量等）。
# 目录逐文件硬链接——hf sync 的 os.walk(followlinks=False) 会静默漏传 symlink 目录。
ln -f "$COMPLETED_JSON" "$STAGE/_run-meta/completed.json"
for it in "${RECORD_ITEMS[@]}"; do
  [ -e "$RECORD_DIR/$it" ] || { echo "错误: 保存现场记录缺少 $RECORD_DIR/$it"; exit 1; }
done
( cd "$RECORD_DIR" && find "${RECORD_ITEMS[@]}" -type d -printf '%p\n' ) | while read -r d; do
  mkdir -p "$STAGE/_run-meta/train-record/$d"
done
( cd "$RECORD_DIR" && find "${RECORD_ITEMS[@]}" -type f -printf '%p\n' ) | while read -r f; do
  mkdir -p "$(dirname "$STAGE/_run-meta/train-record/$f")"
  ln -f "$RECORD_DIR/$f" "$STAGE/_run-meta/train-record/$f"
done
{
  echo "commit=$(git -C "$REPO" rev-parse HEAD)"
  echo "describe=$(git -C "$REPO" describe --always --dirty 2>/dev/null || true)"
  echo "--- git status --porcelain ---"
  git -C "$REPO" status --porcelain || true
} > "$STAGE/_run-meta/git-commit.txt"
{
  echo "# 训练命令行（顺序调度器 scripts/training/prod/run_modul4096_then1024.sh 经 run_modul_budget.sh 起跑，"
  echo "# 下面是实际拼出的 train.py 调用，与 _run-meta/train-record/run_meta.json 的 argv 一致；"
  echo "# 训练 HEAD 与完整环境覆盖见 _run-meta/completed.json 与 train-record/run_meta.json）"
  echo "uv run --no-sync python scripts/training/train.py mme_vla_suite_b128_80k \\"
  echo "  --exp-name $RUN_NAME \\"
  echo "  --assets-base-dir \$REPO/v1-store/train-assets \\"
  echo "  --data.assets.assets-dir \$REPO/v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da \\"
  echo "  --data.assets.asset-id robomme \\"
  echo "  --checkpoint-base-dir \$REPO/v1-store/train-runs \\"
  echo "  --dataset-path \$REPO/v1-store/datasets/4task-v2-1600ep-604f16da/framesamp-8x8 \\"
  echo "  --model.history-config perceptual-framesamp-modul-64frame-8x8.yaml"
} > "$STAGE/_run-meta/train-cmd.txt"

# README 的「训练结果」一节等训练跑完才有实测值，由 motion80k_result_block.py 从训练日志
# **现读现生成**（起止时间、耗时、EXIT_CODE、loss 里程碑表、末条吞吐），替换掉模板里的
# TODO 哨兵。该生成脚本只依赖日志格式与哨兵、与 motion 开关无关，本脚本直接复用、不另立一份。
RM_SRC="$REPO/scripts/dataset/hf_export/m4096_80k_bucket_README.md"
python3 "$REPO/scripts/dataset/hf_export/motion80k_result_block.py" \
  "$TRAIN_LOG" "$RM_SRC" "$STAGE/README.md"
grep -q 'TODO-PASS-B' "$STAGE/README.md" && { echo "错误: 生成的 README 仍含 TODO 哨兵"; exit 1; }
: # 上一行 grep 无匹配返回 1，set -e 下靠这个 no-op 吃掉

# _run-meta 与 README 的分片清单每次运行都重算（这些文件是本脚本刚生成的，不是不可变的）
hash_subdir "_run-meta" "$PARTS/_run-meta.txt"
hash_files "$PARTS/_readme.txt" "README.md"

if find "$STAGE" -type l -print | grep -q .; then
  echo "错误: stage 内出现符号链接"; find "$STAGE" -type l -print; exit 1
fi

# 全局 pre 清单 = 所有分片合并后按路径排序（与 hash_tree 的 sort -k 2 同序）
cat "$PARTS"/*.txt | sort -k 2 > "$TMPD/SHA256SUMS.pre.txt"
mv "$TMPD/SHA256SUMS.pre.txt" "$STAGE/SHA256SUMS.pre.txt"

# LOCAL_FILES 在 mv 之后统计，故含 SHA256SUMS.pre.txt 自身；PRE_LINES 是清单行数，不含它自身。
LOCAL_FILES=$(find "$STAGE" -type f | wc -l)
LOCAL_BYTES=$(find "$STAGE" -type f -printf '%s\n' | awk '{s+=$1} END{print s}')
PRE_LINES=$(wc -l < "$STAGE/SHA256SUMS.pre.txt")
echo "LOCAL_FILES=$LOCAL_FILES LOCAL_BYTES=$LOCAL_BYTES PRE_LINES=$PRE_LINES"
[ "$((PRE_LINES + 1))" = "$LOCAL_FILES" ] || {
  echo "错误: 清单行数 +1 应等于 stage 文件数（$PRE_LINES+1 != $LOCAL_FILES）——有文件没进任何分片"
  exit 1
}
# 分片与 stage 树的独立交叉核对：整棵树重算一遍，必须与合并出来的 pre 清单逐行相同
hash_tree "$STAGE" "$TMPD/SHA256SUMS.pre-recheck.txt"
if diff "$STAGE/SHA256SUMS.pre.txt" "$TMPD/SHA256SUMS.pre-recheck.txt"; then
  echo "PRE_PARTS_CONSISTENT=OK"
else
  echo "错误: 分片合并出的 pre 清单与整树重算不一致（上方 diff 即差异行）"; exit 1
fi

# 补传：阶段 2 中断未完成的 step（正常时全已 .done）+ _run-meta + README + pre 清单
for ((b = 0; b < ${#ALL[@]}; b++)); do
  SUB="${ALL[$b]}"
  DONE="$EXPORT_ROOT/logs/uploaded/$SUB.done"
  if [ -f "$DONE" ]; then continue; fi
  echo "  补传 step $SUB"
  sync_dir "$SUB"; touch "$DONE"
done
sync_dir "_run-meta"
cp_file "README.md"
cp_file "SHA256SUMS.pre.txt"
echo "UPLOAD_DONE batches=${#ALL[@]}"
echo "阶段2b完成"

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
echo "阶段4开始：全量回读到 $VERIFY（约 177 GiB）"
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
# pre 清单比源多 README.md 与 _run-meta/*（stage 专有）；本 run motion 关闭，
# 没有蓝本那一项 _motion-encoder/。比对前剔掉并断言剔除条数对得上。
grep -vE '  (README\.md|_run-meta/)' "$STAGE/SHA256SUMS.pre.txt" > "$TMPD/pre.src-only.txt"
PRE_ONLY_LINES=$(wc -l < "$TMPD/pre.src-only.txt")
SRC_LINES=$(wc -l < "$TMPD/SHA256SUMS.src-after.txt")
RUNMETA_LINES=$(wc -l < "$PARTS/_run-meta.txt")
echo "  pre(剔 README/_run-meta)=$PRE_ONLY_LINES 行, src-after=$SRC_LINES 行, _run-meta=$RUNMETA_LINES 行"
[ "$((PRE_LINES - PRE_ONLY_LINES))" = "$((RUNMETA_LINES + 1))" ] \
  || { echo "错误: 剔除行数应为 $((RUNMETA_LINES + 1))，实为 $((PRE_LINES - PRE_ONLY_LINES))"; exit 1; }
if diff "$TMPD/pre.src-only.txt" "$TMPD/SHA256SUMS.src-after.txt"; then
  echo "SRC_UNCHANGED=OK"
else
  echo "错误: 源文件 sha256 发生变化（上方 diff 即差异行）"; exit 1
fi
echo "阶段6完成"
echo "全部完成"
