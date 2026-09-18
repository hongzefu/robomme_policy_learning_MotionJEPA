#!/usr/bin/env bash
# 从 HF bucket 下载一个 step 目录 + 根目录来源文件；认证经 stdin 传给 curl，不输出密钥。
# 用法：download.sh <bucket 例 HongzeFu/robomme-vla-modul-60k-v1> <step 例 59999> <落点绝对路径> [只取根文件白名单,逗号分隔]
#   - 第 4 参数给出时只下载白名单内的根文件（用于 encoder bucket：checkpoint_epoch_72.pt,config.yaml,SHA256SUMS.pre.txt），step 传 -
#   - bucket 根有 SHA256SUMS.pre.txt 时按它 sha256sum --check；没有时按 bucket tree 的 size 逐文件核对（mode=size）
#   - 落点已存在（首轮已下过别的 step 与根文件）时进入增量模式：只下载 <step>/ 下的文件、不碰根文件，
#     用本地已有的 SHA256SUMS.pre.txt 校验；<step> 目录已存在或本地没有 SHA256SUMS.pre.txt 一律拒绝
set -euo pipefail
source "$(dirname "$0")/env.sh"
BUCKET="${1:?需要 bucket 名}"
STEP="${2:?需要 step（不取 step 目录时传 -）}"
DST="${3:?需要落点绝对路径}"
ONLY="${4:-}"
[[ "$DST" == /* ]] || { echo '落点必须是绝对路径'; exit 2; }
APPEND=0
if [[ -e "$DST" ]]; then
    [[ "$STEP" != - && ! -e "$DST/$STEP" ]] || { echo "目标已存在，拒绝覆盖：$DST（step=$STEP）"; exit 2; }
    [[ -f "$DST/SHA256SUMS.pre.txt" ]] || { echo "已有落点缺 SHA256SUMS.pre.txt，不能增量下载：$DST"; exit 2; }
    APPEND=1
    echo "增量模式：只下载 $STEP/ 下的文件到已有落点 $DST"
fi
mkdir -p "$DST"
auth_curl() {
    { printf 'header = "Authorization: Bearer '; tr -d '\r\n' < "$HOME/.cache/huggingface/token"; printf '"\n'; } |
        curl --config - --fail --location --silent --show-error --retry 3 "$@"
}
# 增量模式下 bucket tree / 清单 / 校验选集都带 step 后缀，不覆盖首轮留下的文件
SUFFIX=""; [[ "$APPEND" == 0 ]] || SUFFIX=".$STEP"
TREE="$DST/bucket-tree$SUFFIX.json"; LIST="$DST/download-list$SUFFIX.txt"; SELECTED="$DST/SHA256SUMS.selected$SUFFIX.txt"
auth_curl "https://huggingface.co/api/buckets/$BUCKET/tree" > "$TREE"
jq -r --arg step "$STEP" --arg only "$ONLY" --argjson append "$APPEND" '
    .[] | select(.type == "file") | .path
    | select( ($step != "-" and startswith($step + "/"))
              or ( ($append == 0) and (contains("/") | not)
                   and ($only == "" or (. as $p | ($only | split(",")) | index($p) != null)) ) )' \
    "$TREE" > "$LIST"
[[ -s "$LIST" ]] || { echo '下载清单为空'; exit 2; }
while IFS= read -r path; do
    [[ "$path" != /* && "$path" != *..* ]] || exit 2
    mkdir -p "$(dirname "$DST/$path")"
    echo "下载 $path"
    auth_curl "https://huggingface.co/buckets/$BUCKET/resolve/$path" -o "$DST/$path.part"
    mv "$DST/$path.part" "$DST/$path"
done < "$LIST"
cd "$DST"
if [[ -f SHA256SUMS.pre.txt ]]; then
    # 只校验本次下载到的文件
    awk 'NR==FNR{want[$0]=1; next} ($2 in want){print}' "$LIST" SHA256SUMS.pre.txt > "$SELECTED"
    [[ -s "$SELECTED" ]] || { echo 'SHA256SUMS.pre.txt 里没有本次下载的文件'; exit 2; }
    sha256sum --check --strict "$SELECTED"
    echo "CHECKPOINT_DOWNLOAD_PASS mode=sha256 files=$(wc -l < "$SELECTED")"
else
    # bucket 根没有 SHA256SUMS：按 bucket tree 的 size 逐文件核对，并本地算一份 sha256 供留档
    N=0
    while IFS= read -r path; do
        WANT="$(jq -r --arg p "$path" '.[] | select(.path == $p) | .size' "$TREE")"
        GOT="$(stat -c %s "$path")"
        [[ "$WANT" == "$GOT" ]] || { echo "大小不符 $path: bucket=$WANT local=$GOT"; exit 2; }
        N=$((N + 1))
    done < "$LIST"
    xargs -a "$LIST" sha256sum > SHA256SUMS.local.txt
    echo "CHECKPOINT_DOWNLOAD_PASS mode=size files=$N"
fi
