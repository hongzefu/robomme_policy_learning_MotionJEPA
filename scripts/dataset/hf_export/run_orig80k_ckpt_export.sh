#!/usr/bin/env bash
# driver：把 commitV11.18Beta 起跑的两条原版 80k 正式 run 之一的全部 8 个 orbax checkpoint
# （10000…70000 共 7 个 + 收尾 79999，每个约 11.88 GB）上传到 HF **private** bucket，
# 并做「上传前 / 上传后」两遍独立 sha256 逐行对照验收。
#
# 用法: run_orig80k_ckpt_export.sh full|count
#   full  → v2-orig-16task-pub1600ep-modul-b64-80k  → HongzeFu/robomme-vla-orig-16task-1600ep-modul-b64-80k-v1
#   count → v2-orig-counting-pub400ep-modul-b64-80k → HongzeFu/robomme-vla-orig-counting-400ep-modul-b64-80k-v1
# 可选 DRY_RUN=1：只跑阶段 0–1（闸门、身份、stage 硬链接、分片 sha256），不建 bucket、不上传。
#
# 蓝本是同目录 run_m1024_80k_ckpt_export.sh，逻辑逐段相同，只改：
#   - run / bucket 由参数选择，SRC 中间层是 mme_vla_suite；
#   - 完成证据改为 orig80k 完成器 check_orig80k_completion.py 的 <records>/completion.json；
#     该文件不存在时按 docs/training-doc/orig80k-prod-0927/launch.md 写定的命令在 CPU 上跑一次（独占新建）；
#   - EXPECT_STEPS=8；README 由 orig80k_result_block.py 从现场文件整份生成；
#   - _run-meta 附 records 目录下的 launch/runtime/metrics/gpu 采样/final 与实际启动命令文件。
# stage 走硬链接、不打 tar、不复制 token、按 step 目录分批 sync，四条做法与蓝本相同。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[ -f "$REPO/pyproject.toml" ] || { echo "错误: 仓库根解析失败 $REPO"; exit 1; }
cd "$REPO"

case "${1:-}" in
  full)  RUN_NAME="v2-orig-16task-pub1600ep-modul-b64-80k"
         BUCKET_ID="HongzeFu/robomme-vla-orig-16task-1600ep-modul-b64-80k-v1"
         CMD_TAG="orig80k-prod-full-20260927-ready" ;;
  count) RUN_NAME="v2-orig-counting-pub400ep-modul-b64-80k"
         BUCKET_ID="HongzeFu/robomme-vla-orig-counting-400ep-modul-b64-80k-v1"
         CMD_TAG="orig80k-prod-count-20260927-ready" ;;
  *) echo "用法: $0 full|count"; exit 1 ;;
esac
DRY_RUN="${DRY_RUN:-0}"
BUCKET="hf://buckets/$BUCKET_ID"
# v1-store 解析成实体路径：主副本下它本身就是实体目录，-temp 开发副本下它是指向主副本的 symlink。
# 下面所有数据路径都走 $STORE，两种副本执行时读写的是同一批实体文件。
STORE="$(readlink -f "$REPO/v1-store")"
SRC="$STORE/train-runs/mme_vla_suite/$RUN_NAME"
EXPORT_ROOT="$STORE/exports/hf-ckpt-$RUN_NAME"
STAGE="$EXPORT_ROOT/stage"
VERIFY="$EXPORT_ROOT/verify"
TMPD="$EXPORT_ROOT/tmp"
PARTS="$TMPD/pre-parts"
TRAIN_LOG="$STORE/logs/$RUN_NAME.driver.log"
# records 目录 = 完成器 --records；completion.json 由完成器独占新建
RECORD_DIR="$STORE/bench/orig80k/$RUN_NAME"
COMPLETED_JSON="$RECORD_DIR/completion.json"
RECORD_ITEMS=(final launch.json runtime.json metrics.jsonl gpu.csv gpu.csv.err)
CMD_FILE="$STORE/bench/orig80k-prod-commands-20260927-ready/$CMD_TAG.command"
WANDB_ID_FILE="$SRC/wandb_id.txt"
EXPECT_STEPS=8
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
# 不看「有没有训练在跑」（pgrep 易误判），只看 EXPORT_THROTTLE。
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
# 训练必须已正常结束（driver 日志唯一终态 EXIT_CODE=0），且 orig80k 完成器判 PASS
[ "$(grep -c '^EXIT_CODE=' "$TRAIN_LOG")" = 1 ] && grep -qx 'EXIT_CODE=0' "$TRAIN_LOG" \
  || { echo "错误: $TRAIN_LOG 没有唯一的 EXIT_CODE=0，训练未正常结束"; exit 1; }
TRAIN_HEAD="$(sed -n 's/^TRAIN_HEAD=//p' "$TRAIN_LOG" | tail -1)"
[[ "$TRAIN_HEAD" =~ ^[0-9a-f]{40}$ ]] || { echo "错误: 日志里取不到 TRAIN_HEAD"; exit 1; }
echo "  TRAIN_HEAD=$TRAIN_HEAD"
if [ ! -e "$COMPLETED_JSON" ]; then
  # 与 launch.md 写定的命令一致：独立 CPU 任务、--out 独占新建；stdout 与退出码另存在导出根
  mkdir -p "$EXPORT_ROOT/logs"
  echo "  completion.json 不存在，在 CPU 上运行完成器"
  set +e
  ( cd "$REPO" && env -u JAX_PLATFORM_NAME -u PYTHONPATH -u PYTHONHOME \
      CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
      uv run --no-sync python "$REPO/scripts/training/tests/check_orig80k_completion.py" \
      --mode prod --records "$RECORD_DIR" --run-root "$SRC" --log "$TRAIN_LOG" \
      --run "$RUN_NAME" --head "$TRAIN_HEAD" --out "$COMPLETED_JSON" ) 2>&1 \
    | tee "$EXPORT_ROOT/logs/completion.stdout"
  rc=${PIPESTATUS[0]}
  set -e
  echo "COMPLETION_EXIT_CODE=$rc" | tee -a "$EXPORT_ROOT/logs/completion.stdout"
  [ "$rc" = 0 ] || { echo "错误: 完成器未通过（退出码 $rc），停止上传"; exit 1; }
fi
python3 - "$COMPLETED_JSON" "$RUN_NAME" "$TRAIN_HEAD" "$EXPECT_STEPS" "$EXPECT_LAST" <<'PY' \
  || { echo "错误: 完成记录 $COMPLETED_JSON 不满足要求"; exit 1; }
import json, sys
path, run, head, n, last = sys.argv[1:]
d = json.load(open(path))
steps = sorted(int(s) for s in d.get("checkpoints") or [0])
errs = []
if d.get("status") != "PASS": errs.append(f"status={d.get('status')}")
if d.get("mode") != "prod": errs.append(f"mode={d.get('mode')}")
if d.get("run_name") != run: errs.append(f"run_name={d.get('run_name')}")
if d.get("head") != head: errs.append(f"head={d.get('head')}")
if d.get("state_step") != 80000: errs.append(f"state_step={d.get('state_step')}")
if len(steps) != int(n) or steps[-1] != int(last): errs.append(f"checkpoints={steps}")
if errs:
    sys.exit("  完成记录不符: " + "; ".join(errs))
print(f"  完成记录: status=PASS run={run} checkpoints={len(steps)} final={steps[-1]}")
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

if [ "$DRY_RUN" = 1 ]; then
  echo "  DRY_RUN=1：跳过建 bucket 与空库断言"
else
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
fi
echo "阶段0完成"

# ---------------------------------------------------------------- 阶段 1
echo "阶段1开始：枚举 step 目录 + 增量 stage（硬链接）+ 分片 sha256"
mapfile -t ALL < <( cd "$SRC" && find . -mindepth 1 -maxdepth 1 -type d -printf '%P\n' | sort -n )
# 完整性守门：num_train_steps=80_000 / save_interval=keep_period=10_000
# ⇒ 10000…70000 共 7 个 + 收尾 79999，合计 8 个。训练已结束（阶段 0 已断言），全部可传。
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
if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN=PASS steps=${#SAFE[@]}（到此为止，未建 bucket、未上传）"; exit 0; fi

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
WANDB_SRC="$(find "$STORE/logs/wandb/$RUN_NAME/wandb" -maxdepth 1 -type d -name "run-*-$WANDB_ID" | head -1)"
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
ln -f "$COMPLETED_JSON" "$STAGE/_run-meta/completion.json"
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
  echo "train_head=$TRAIN_HEAD"
  echo "export_head=$(git -C "$REPO" rev-parse HEAD)"
  echo "describe=$(git -C "$REPO" describe --always --dirty 2>/dev/null || true)"
  echo "--- git status --porcelain ---"
  git -C "$REPO" status --porcelain || true
} > "$STAGE/_run-meta/git-commit.txt"
[ -f "$CMD_FILE" ] || { echo "错误: 缺少实际启动命令文件 $CMD_FILE"; exit 1; }
ln -f "$CMD_FILE" "$STAGE/_run-meta/train-cmd.txt"

# README 由 orig80k_result_block.py 从 launch.json / completion.json / driver 日志现读、整份生成
rm -f "$STAGE/README.md"
python3 "$REPO/scripts/dataset/hf_export/orig80k_result_block.py" \
  "$RUN_NAME" "$BUCKET_ID" "$RECORD_DIR" "$SRC" "$TRAIN_LOG" "$STAGE/README.md"

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
echo "阶段4开始：全量回读到 $VERIFY（约 89 GiB）"
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
