#!/usr/bin/env bash
# GL standard 分区纯 CPU job：把一条训练 run 的 step ckpt 目录与 run 根平文件上传到 HF bucket，再逐对象流式读回校验。
# 用法（经 aspen：ssh greatlakes 'read -r HF_TOKEN; export HF_TOKEN; sbatch --export=ALL ... <本脚本> <RUN_DIR> <BUCKET_ID> <OUT_DIR>' < token）
#   RUN_DIR   turbo 上的 run 根（含 5000…50000 与 history_config.* 等）；resume/ 不上传
#   BUCKET_ID HongzeFu/<名>，公开
#   OUT_DIR   turbo 上本次导出的记录目录（SHA256SUMS、SIZES、标记文件）
# 读回按 greatlakes.md「HF 上传校验 job」：hf buckets cp <uri> - 流式读出，边读边算 sha256 与字节数，不落盘。
# 凭据只经环境变量 HF_TOKEN 进入，不写任何文件。末行 HF_VERIFY=PASS|FAIL …，日志由 sbatch --output 落 NFS。
set -uo pipefail
RUN_DIR=${1:?RUN_DIR}; BUCKET_ID=${2:?BUCKET_ID}; OUT=${3:?OUT_DIR}
BUCKET="hf://buckets/$BUCKET_ID"
G=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA
export HF_HOME=$G/v1-store/cache/hf HF_XET_CACHE=$G/v1-store/cache/hf-xet HF_XET_HIGH_PERFORMANCE=1 UV_LINK_MODE=copy
unset HF_HUB_OFFLINE || true
NPROC=${SLURM_CPUS_PER_TASK:-8}
[ -n "${HF_TOKEN:-}" ] || { echo "错误: 环境里没有 HF_TOKEN"; echo "HF_VERIFY=FAIL reason=no_token"; exit 1; }
hf() { uvx --python 3.11 --from 'huggingface_hub==1.30.0' hf "$@"; }
export -f hf
export BUCKET OUT
echo "JOB=${SLURM_JOB_ID:-} host=$(hostname) run=$RUN_DIR bucket=$BUCKET_ID start=$(TZ=America/New_York date '+%F %T') 美东"
[ -d "$RUN_DIR" ] || { echo "错误: RUN_DIR 不存在"; exit 1; }
mkdir -p "$OUT"
cd "$RUN_DIR"
mapfile -t STEPS < <(find . -mindepth 1 -maxdepth 1 -type d -printf '%P\n' | grep -xE '[0-9]+' | sort -n)
mapfile -t ROOTF < <(find . -mindepth 1 -maxdepth 1 -type f -printf '%P\n' | sort)
[ "${#STEPS[@]}" = 10 ] || { echo "错误: step 目录数 ${#STEPS[@]} != 10"; exit 1; }
for s in "${STEPS[@]}"; do [ -f "$s/_CHECKPOINT_METADATA" ] || { echo "错误: $s 缺 _CHECKPOINT_METADATA"; exit 1; }; done
echo "STEPS=${STEPS[*]} ROOT_FILES=${ROOTF[*]}"

echo "== 阶段0 身份与建 bucket"
who=$(hf auth whoami 2>&1 | tr '\n' ' ')
case "$who" in *HongzeFu*) echo "HF_WHOAMI=HongzeFu" ;; *) echo "错误: 身份不对（$who）"; exit 1 ;; esac
hf buckets create "$BUCKET_ID" 2>&1 | tail -2 || true
# 一律公开（用户 11-20：「上传公开的！！！！」）；private 存储有 1 TB 上限，已存在的 private bucket 也在这里改成公开
hf buckets settings "$BUCKET_ID" --public 2>&1 | tail -2 || { echo "错误: 改公开失败"; exit 1; }

echo "== 阶段1 源侧 sha256（$NPROC 并行）"
if [ ! -s "$OUT/SHA256SUMS" ]; then
  { find "${STEPS[@]}" -type f; printf '%s\n' "${ROOTF[@]}"; } | sort > "$OUT/files.txt"
  xargs -a "$OUT/files.txt" -P "$NPROC" -n 4 sha256sum | sort -k 2 > "$OUT/SHA256SUMS.tmp" && mv "$OUT/SHA256SUMS.tmp" "$OUT/SHA256SUMS"
  xargs -a "$OUT/files.txt" stat -c '%s %n' | sort -k 2 > "$OUT/SIZES"
fi
N=$(wc -l < "$OUT/SHA256SUMS"); BYTES=$(awk '{s+=$1} END{print s}' "$OUT/SIZES")
echo "SRC_SUMS files=$N bytes=$BYTES"

echo "== 阶段2 上传"
retry() { local a; for a in 1 2 3 4 5 6 7 8; do "$@" && return 0; echo "  [$*] 第 $a 次失败，60s 后重试"; sleep 60; done; return 1; }
for s in "${STEPS[@]}"; do
  [ -f "$OUT/up.$s.done" ] && { echo "  $s 已上传，跳过"; continue; }
  t0=$(date +%s)
  retry hf sync "$RUN_DIR/$s" "$BUCKET/$s" > "$OUT/up.$s.log" 2>&1 || { echo "错误: $s 上传八次仍失败"; tail -5 "$OUT/up.$s.log"; echo "HF_VERIFY=FAIL reason=upload step=$s"; exit 1; }
  touch "$OUT/up.$s.done"; echo "UPLOADED step=$s seconds=$(( $(date +%s)-t0 )) $(TZ=America/New_York date '+%T') 美东"
done
for f in "${ROOTF[@]}"; do retry hf buckets cp "$RUN_DIR/$f" "$BUCKET/$f" > /dev/null 2>&1 || { echo "错误: $f 上传失败"; exit 1; }; done
retry hf buckets cp "$OUT/SHA256SUMS" "$BUCKET/SHA256SUMS" > /dev/null 2>&1 || { echo "错误: SHA256SUMS 上传失败"; exit 1; }
echo "UPLOAD_DONE files=$((N+1)) $(TZ=America/New_York date '+%F %T') 美东"

echo "== 阶段3 流式读回校验"
check_one() {  # $1=sha $2=path ；输出 "OK|BAD path got_sha got_bytes"
  local exp=$1 p=$2 got n
  read -r got n < <(hf buckets cp "$BUCKET/$p" - 2>/dev/null | python3 -c "
import sys,hashlib
h=hashlib.sha256();n=0
while True:
    b=sys.stdin.buffer.read(1<<22)
    if not b: break
    h.update(b);n+=len(b)
print(h.hexdigest(),n)")
  [ "$got" = "$exp" ] && echo "OK $p $got $n" || echo "BAD $p $got $n"
}
export -f check_one
# 冒烟：1 个小对象
smoke=$(grep ' history_config.txt$' "$OUT/SHA256SUMS")
r=$(check_one $smoke); echo "SMOKE $r"; case "$r" in OK*) ;; *) echo "HF_VERIFY=FAIL reason=smoke"; exit 1 ;; esac
awk '{print $1, $2}' "$OUT/SHA256SUMS" | xargs -P "$NPROC" -n 2 bash -c 'check_one "$0" "$1"' > "$OUT/verify.txt"
OK=$(grep -c '^OK ' "$OUT/verify.txt"); BAD=$(grep -c '^BAD ' "$OUT/verify.txt")
SZ=$(awk 'NR==FNR{s[$2]=$1;next} $1=="OK" && s[$2]==$4{c++} END{print c+0}' "$OUT/SIZES" "$OUT/verify.txt")
MISS=$(( N - OK - BAD ))
# 清单外多出对象：用 bucket 统计的 totalFiles 对照（应为 N+1，含 SHA256SUMS；统计异步滞后，最多等 10 分钟）
EXP=$((N+1)); TF=""
for i in $(seq 1 20); do
  TF=$(curl -sf -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/api/buckets/HongzeFu?limit=1000" \
       | python3 -c "import sys,json;print(next((b['totalFiles'] for b in json.load(sys.stdin) if b['id']=='$BUCKET_ID'),''))")
  [ "$TF" = "$EXP" ] && break; sleep 30
done
EXTRA=$(( ${TF:-0} - EXP )); [ "$EXTRA" -lt 0 ] && EXTRA=0
V=PASS; [ "$OK" = "$N" ] && [ "$SZ" = "$N" ] && [ "$BAD" = 0 ] && [ "$TF" = "$EXP" ] || V=FAIL
grep '^BAD ' "$OUT/verify.txt" | head
echo "BUCKET_TOTALFILES=$TF expected=$EXP"
echo "end=$(TZ=America/New_York date '+%F %T') 美东"
echo "HF_VERIFY=$V objects=$N sha_match=$OK size_equal=$SZ missing=$MISS extra=$EXTRA bucket=$BUCKET_ID bytes=$BYTES"
[ "$V" = PASS ]
