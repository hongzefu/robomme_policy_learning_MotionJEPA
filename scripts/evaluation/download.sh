#!/usr/bin/env bash
# 从 HF bucket 下载一个 step 目录 + 根目录来源文件；认证经 stdin 传给 curl，不输出密钥。
# 用法：download.sh <bucket 例 HongzeFu/robomme-vla-modul-60k-v1> <step 例 59999> <落点绝对路径> [只取根文件白名单,逗号分隔]
#   - 第 4 参数给出时只下载白名单内的根文件（用于 encoder bucket：checkpoint_epoch_72.pt,config.yaml,SHA256SUMS.pre.txt），step 传 -
#   - bucket 根有 SHA256SUMS.pre.txt 时按它 sha256sum --check；没有时按 bucket tree 的 size 逐文件核对（mode=size）
set -euo pipefail
source "$(dirname "$0")/env.sh"
BUCKET="${1:?需要 bucket 名}"
STEP="${2:?需要 step（不取 step 目录时传 -）}"
DST="${3:?需要落点绝对路径}"
ONLY="${4:-}"
[[ "$DST" == /* ]] || { echo '落点必须是绝对路径'; exit 2; }
[[ ! -e "$DST" ]] || { echo "目标已存在，拒绝覆盖：$DST"; exit 2; }
mkdir -p "$DST"
auth_curl() {
    { printf 'header = "Authorization: Bearer '; tr -d '\r\n' < "$HOME/.cache/huggingface/token"; printf '"\n'; } |
        curl --config - --fail --location --silent --show-error --retry 3 "$@"
}
auth_curl "https://huggingface.co/api/buckets/$BUCKET/tree" > "$DST/bucket-tree.json"
jq -r --arg step "$STEP" --arg only "$ONLY" '
    .[] | select(.type == "file") | .path
    | select( ($step != "-" and startswith($step + "/"))
              or ( (contains("/") | not) and ($only == "" or (. as $p | ($only | split(",")) | index($p) != null)) ) )' \
    "$DST/bucket-tree.json" > "$DST/download-list.txt"
[[ -s "$DST/download-list.txt" ]] || { echo '下载清单为空'; exit 2; }
while IFS= read -r path; do
    [[ "$path" != /* && "$path" != *..* ]] || exit 2
    mkdir -p "$(dirname "$DST/$path")"
    echo "下载 $path"
    auth_curl "https://huggingface.co/buckets/$BUCKET/resolve/$path" -o "$DST/$path.part"
    mv "$DST/$path.part" "$DST/$path"
done < "$DST/download-list.txt"
cd "$DST"
if [[ -f SHA256SUMS.pre.txt ]]; then
    # 只校验本次下载到的文件
    awk 'NR==FNR{want[$0]=1; next} ($2 in want){print}' download-list.txt SHA256SUMS.pre.txt > SHA256SUMS.selected.txt
    [[ -s SHA256SUMS.selected.txt ]] || { echo 'SHA256SUMS.pre.txt 里没有本次下载的文件'; exit 2; }
    sha256sum --check --strict SHA256SUMS.selected.txt
    echo "CHECKPOINT_DOWNLOAD_PASS mode=sha256 files=$(wc -l < SHA256SUMS.selected.txt)"
else
    # bucket 根没有 SHA256SUMS：按 bucket tree 的 size 逐文件核对，并本地算一份 sha256 供留档
    N=0
    while IFS= read -r path; do
        WANT="$(jq -r --arg p "$path" '.[] | select(.path == $p) | .size' bucket-tree.json)"
        GOT="$(stat -c %s "$path")"
        [[ "$WANT" == "$GOT" ]] || { echo "大小不符 $path: bucket=$WANT local=$GOT"; exit 2; }
        N=$((N + 1))
    done < download-list.txt
    xargs -a download-list.txt sha256sum > SHA256SUMS.local.txt
    echo "CHECKPOINT_DOWNLOAD_PASS mode=size files=$N"
fi
