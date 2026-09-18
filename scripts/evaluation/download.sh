#!/usr/bin/env bash
# 仅下载指定末步与来源文件；认证经 stdin 传递，不输出密钥。
set -euo pipefail
source "$(dirname "$0")/env.sh"
DST="$EVAL_REPO/v1-store/models/robomme-vla-modul-60k-v1"
[[ ! -e "$DST" ]] || { echo "目标已存在，拒绝覆盖：$DST"; exit 2; }
mkdir -p "$DST"
auth_curl() {
    { printf 'header = "Authorization: Bearer '; tr -d '\r\n' < "$HOME/.cache/huggingface/token"; printf '"\n'; } |
        curl --config - --fail --location --silent --show-error --retry 3 "$@"
}
auth_curl 'https://huggingface.co/api/buckets/HongzeFu/robomme-vla-modul-60k-v1/tree' > "$DST/bucket-tree.json"
jq -r '.[] | select(.type == "file") | .path | select(startswith("59999/") or (contains("/") | not))' "$DST/bucket-tree.json" |
while IFS= read -r path; do
    [[ "$path" != /* && "$path" != *..* ]] || exit 2
    mkdir -p "$(dirname "$DST/$path")"
    echo "下载 $path"
    auth_curl "https://huggingface.co/buckets/HongzeFu/robomme-vla-modul-60k-v1/resolve/$path" -o "$DST/$path.part"
    mv "$DST/$path.part" "$DST/$path"
done
cd "$DST"
awk '$2 ~ /^59999\// || $2 !~ /\// {print}' SHA256SUMS.pre.txt > SHA256SUMS.selected.txt
sha256sum --check --strict SHA256SUMS.selected.txt
echo CHECKPOINT_DOWNLOAD_PASS
