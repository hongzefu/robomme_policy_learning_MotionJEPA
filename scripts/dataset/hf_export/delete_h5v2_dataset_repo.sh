#!/usr/bin/env bash
# 删除已搬迁完毕的 dataset repo HongzeFu/robomme-4task-h5-20260912-v2。
#
# **这是本轮唯一不可逆的动作。** 单列一个脚本、要求显式 CONFIRM_DELETE=1，刻意不写进
# run_h5v2_rehost.sh 的收尾——校验通过与执行删除之间必须有一个人的决定。
#
# 为什么每一项都重查而不是信任上一轮的输出：搬迁与删除之间可能隔了很久，中间 bucket 可能
# 被改过、被清过。删除前的断言必须基于「此刻」的事实。
#
# **最危险的一点**：`hf repos delete <id> --repo-type dataset` 与 `hf buckets delete <id>`
# 的 <id> 字符串**完全相同**，而 `hf repos delete` 的 --repo-type 默认值是 model。打错一个
# 参数就是把刚搬好的 126.6 GB 目的地删掉。所以下面：
#   - 删除命令里的 --repo-type dataset 显式写死；
#   - 删除前断言 dataset 与 bucket **两者都在**；
#   - 删除后断言 dataset 没了、**bucket 还在且文件数与字节数一个不少**。
#
# 用法：
#   bash scripts/dataset/hf_export/delete_h5v2_dataset_repo.sh              # 只做前置断言，不删
#   CONFIRM_DELETE=1 bash scripts/dataset/hf_export/delete_h5v2_dataset_repo.sh   # 真删
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

SRC_REPO="HongzeFu/robomme-4task-h5-20260912-v2"
BUCKET_ID="HongzeFu/robomme-4task-h5-20260912-v2"
BUCKET="hf://buckets/$BUCKET_ID"
RECORDS="$REPO/docs/dataset-build-doc/hf-export-h5v2-rehost-20260914/records"
EXPECT_FILES="${EXPECT_FILES:-2058}"     # repo 的 2057 件 + MIGRATION.md
TMPD="$REPO/v1-store/exports/hf-h5v2-rehost/tmp"
CONFIRM_DELETE="${CONFIRM_DELETE:-0}"

export PATH="/home/ec2-user/.local/bin:$PATH"
export HF_HOME="$REPO/v1-store/cache/hf"
export HF_XET_CACHE="$REPO/v1-store/cache/hf-xet"
export UV_CACHE_DIR=/scratch/hongze/.cache/uv
export PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE || true
set -a; . "$REPO/v1-store/secrets/hf.env"; set +a
hf() { uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf "$@"; }

http_code() {  # $1 = url
  curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $HF_TOKEN" "$1"
}
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

echo "=== 前置断言 ==="
DS_CODE=$(http_code "https://huggingface.co/api/datasets/$SRC_REPO")
BK_FILES=$(bucket_stat totalFiles); BK_BYTES=$(bucket_stat size); BK_PRIV=$(bucket_stat private)
echo "  dataset repo HTTP=$DS_CODE"
echo "  bucket files=$BK_FILES bytes=$BK_BYTES private=$BK_PRIV"

[ "$DS_CODE" = "200" ] || { echo "错误: dataset repo 不可达（HTTP=$DS_CODE），可能已删，停止"; exit 1; }
[ "$BK_FILES" = "$EXPECT_FILES" ] || { echo "错误: bucket 文件数 $BK_FILES != 期望 $EXPECT_FILES，停止"; exit 1; }
case "$BK_PRIV" in False|false|None) : ;; *) echo "错误: bucket 不是公开的，停止"; exit 1 ;; esac

# 留档必须已经抓好——repo 一删，revision 历史 / discussions / README front matter 就永久没了
for f in revision.json tree.json commits.json paths_info.json README.raw.md summary.json; do
  [ -s "$RECORDS/$f" ] || { echo "错误: 缺少元数据留档 $RECORDS/$f，停止"; exit 1; }
done
echo "  元数据留档齐备: $(ls "$RECORDS" | tr '\n' ' ')"

# 匿名再读一次：证明此刻陌生人确实能从 bucket 取到东西，再去删 repo
rm -rf "$TMPD/predel-anon"; mkdir -p "$TMPD/predel-anon"
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HF_HUB_OFFLINE HF_XET_CACHE="$TMPD/predel-anon-xet" \
  uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf \
  sync "$BUCKET" "$TMPD/predel-anon" --include 'SHA256SUMS' >/dev/null 2>&1
[ -s "$TMPD/predel-anon/SHA256SUMS" ] || { echo "错误: 匿名读 bucket 失败，停止"; exit 1; }
echo "  匿名读 bucket: OK（$(wc -l < "$TMPD/predel-anon/SHA256SUMS") 行 SHA256SUMS）"
echo "DELETE_PRECHECK=PASS"

if [ "$CONFIRM_DELETE" != "1" ]; then
  echo
  echo "未设置 CONFIRM_DELETE=1，**没有删除任何东西**。"
  echo "确认要删时执行： CONFIRM_DELETE=1 bash $0"
  exit 0
fi

echo
echo "=== 执行删除（不可逆）==="
echo "  目标: dataset repo $SRC_REPO （--repo-type dataset 显式写死）"
hf repos delete "$SRC_REPO" --repo-type dataset -y

echo "=== 后置断言 ==="
sleep 5
DS_AFTER=$(http_code "https://huggingface.co/api/datasets/$SRC_REPO")
BK_FILES_AFTER=$(bucket_stat totalFiles); BK_BYTES_AFTER=$(bucket_stat size)
BK_PRIV_AFTER=$(bucket_stat private)
echo "  dataset repo HTTP=$DS_AFTER （期望 404 或 401）"
echo "  bucket files=$BK_FILES_AFTER bytes=$BK_BYTES_AFTER private=$BK_PRIV_AFTER"
case "$DS_AFTER" in 404|401) echo "DATASET_DELETED=OK" ;;
  *) echo "错误: dataset repo 仍返回 HTTP=$DS_AFTER，删除可能未生效"; exit 1 ;; esac
[ "$BK_FILES_AFTER" = "$BK_FILES" ] || { echo "错误: bucket 文件数变了 $BK_FILES → $BK_FILES_AFTER"; exit 1; }
[ "$BK_BYTES_AFTER" = "$BK_BYTES" ] || { echo "错误: bucket 字节数变了 $BK_BYTES → $BK_BYTES_AFTER"; exit 1; }
echo "BUCKET_INTACT=$BK_FILES_AFTER PRIVATE=$BK_PRIV_AFTER"

# 删完再匿名读一次：证明 repo 消失没有连带影响 bucket 的可达性
rm -rf "$TMPD/postdel-anon"; mkdir -p "$TMPD/postdel-anon"
if env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HF_HUB_OFFLINE HF_XET_CACHE="$TMPD/postdel-anon-xet" \
     uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf \
     sync "$BUCKET" "$TMPD/postdel-anon" --include 'MIGRATION.md' >/dev/null 2>&1 \
   && [ -s "$TMPD/postdel-anon/MIGRATION.md" ]; then
  echo "POST_DELETE_ANON_READ=OK"
else
  echo "错误: 删除后匿名读 bucket 失败"; exit 1
fi
echo "DELETE_RESULT=PASS"
