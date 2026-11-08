#!/usr/bin/env bash
# 本机 BinFill demo 前缀生成驱动（1108-4task-modul-wantp768-50k-plan.html §4 F6/U4、§5 E0、runbook C5/C6）。
# 把 GL 的 gl_demo_pool.sh 换成本机 8 卡：分片 k 用 GPU k%8，每片后台调既有 run_demo_shard.sh（gen 路径一行不动），
# wait 逐片收 SHARD_EXIT_CODE（R10，不吞码）；全部为 0 才 finalize --max-fail-rate 0.10，finalize 为 0 才做 E0。
#
# 用法：local_demo_pool.sh [输出前缀库根，默认 v1-store/demo-prefix/binfill-3b4de03a0b46（相对仓库根或绝对路径）]
#   正式：SHARDS=<16..32>（必填）
#   小样：PROBE=1 PROBE_GPU=<g> PROBE_SHARDS=<n> [SHARDS=<切片数，默认 16>]
#         只在 PROBE_GPU 一张卡上跑切片 0..n-1（与正式同一切法 rows[k::SHARDS]），不 finalize、不做 E0；
#         打印每片耗时供定 SHARDS；小样库为跑完即删的临时产物（由主会话删）
#   DRY_RUN=1：只打印片→卡映射与将执行的命令
# E0 判定行写 $STORE/E0.txt：
#   DEMO_PREFIX_SRC=PASS code_same=1 benchmark=b4e97f2 shards_ok=<n>/<n> ok=<n> failed=<n> diff_vs_old=<k>
#   （diff_vs_old 只记录：与旧 demo-prefix-index.json 成功集合对称差 + 共同成功条目中 D 不同的条数，明细写 demo-prefix-diff.json）
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$EVAL_REPO"

BASE_SHA=ba52d79
BENCH_SHA=b4e97f22fe007078e297205898c07c1acbc69165
OLD_INDEX="$EVAL_REPO/docs/training-doc/binfilldemo-nomotion50k-gl/records/demo-prefix-index.json"
DRY="${DRY_RUN:-0}"
PROBE="${PROBE:-0}"
STORE="${1:-v1-store/demo-prefix/binfill-3b4de03a0b46}"
[[ "$STORE" == /* ]] || STORE="$EVAL_REPO/$STORE"
if [[ "$PROBE" == 1 ]]; then
    SHARDS="${SHARDS:-16}"
    PROBE_GPU="${PROBE_GPU:?PROBE=1 需要 PROBE_GPU}"
    PROBE_SHARDS="${PROBE_SHARDS:?PROBE=1 需要 PROBE_SHARDS}"
    [[ "$PROBE_GPU" =~ ^[0-7]$ && "$PROBE_SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "PROBE_GPU / PROBE_SHARDS 非法"; exit 2; }
    (( PROBE_SHARDS <= SHARDS )) || { echo "PROBE_SHARDS=$PROBE_SHARDS > SHARDS=$SHARDS"; exit 2; }
    RUN_SHARDS="$PROBE_SHARDS"
else
    SHARDS="${SHARDS:?正式模式需要 SHARDS（16..32）}"
    [[ "$SHARDS" =~ ^[0-9]+$ ]] && (( SHARDS >= 16 && SHARDS <= 32 )) || { echo "SHARDS 须在 16..32：$SHARDS"; exit 2; }
    RUN_SHARDS="$SHARDS"
fi
gpu_of() { if [[ "$PROBE" == 1 ]]; then echo "$PROBE_GPU"; else echo $(( $1 % 8 )); fi; }

echo "DEMO_POOL_START probe=$PROBE dry=$DRY store=$STORE shards=$SHARDS run_shards=$RUN_SHARDS $(date -u +%FT%TZ)"
for ((K=0; K<RUN_SHARDS; K++)); do
    echo "DEMO_SHARD_MAP shard=$K/$SHARDS gpu=$(gpu_of "$K")"
done
if [[ "$DRY" == 1 ]]; then
    echo "DRY 每片：CUDA_VISIBLE_DEVICES=<gpu> STORE=$STORE SHARDS=$SHARDS bash scripts/evaluation/run_demo_shard.sh <k> $STORE"
    [[ "$PROBE" == 1 ]] || echo "DRY 之后：pregen_binfill_demo.py finalize --out $STORE --shards $SHARDS --max-fail-rate 0.10 → E0 → $STORE/E0.txt"
    exit 0
fi

mkdir -p "$STORE/logs"
PIDS=(); T0S=()
for ((K=0; K<RUN_SHARDS; K++)); do
    env CUDA_VISIBLE_DEVICES="$(gpu_of "$K")" STORE="$STORE" SHARDS="$SHARDS" \
        bash scripts/evaluation/run_demo_shard.sh "$K" "$STORE" > "$STORE/logs/driver-s$K.log" 2>&1 &
    PIDS+=("$!"); T0S+=("$(date +%s)")
done
NOK=0
for ((K=0; K<RUN_SHARDS; K++)); do
    wait "${PIDS[$K]}"; RC=$?
    echo "SHARD_EXIT_CODE=$RC shard=$K seconds=$(( $(date +%s) - T0S[K] ))"
    [[ "$RC" == 0 ]] && NOK=$((NOK + 1))
done
echo "DEMO_SHARDS_DONE ok=$NOK/$RUN_SHARDS"
if [[ "$PROBE" == 1 ]]; then
    echo "DEMO_PROBE_DONE shards_ok=$NOK/$RUN_SHARDS（小样不 finalize、不做 E0）"
    [[ "$NOK" == "$RUN_SHARDS" ]] || exit 1
    exit 0
fi
[[ "$NOK" == "$SHARDS" ]] || { echo "DEMO_PREFIX_SRC=FAIL reason=gen_shard_nonzero shards_ok=$NOK/$SHARDS（R10，不进 finalize）"; exit 1; }

"${CLIENT_UV[@]}" scripts/evaluation/pregen_binfill_demo.py finalize --out "$STORE" --shards "$SHARDS" --max-fail-rate 0.10 \
    2>&1 | tee "$STORE/logs/finalize.log"
FRC=${PIPESTATUS[0]}
echo "FINALIZE_EXIT_CODE=$FRC"
[[ "$FRC" == 0 ]] || { echo "DEMO_PREFIX_SRC=FAIL reason=finalize rc=$FRC（失败率 > 10% 或其他 finalize 错误）"; exit 1; }

# ── E0：同一份生成代码（runbook C6）──
CODE_SAME=1
git diff --quiet "$BASE_SHA" -- scripts/evaluation/run_demo_shard.sh examples/robomme/demo_prefix.py || CODE_SAME=0
# pregen_binfill_demo.py 相对 ba52d79 的改动只许落在 cmd_finalize 与 main（argparse --max-fail-rate）：按 hunk 头的函数上下文核
HUNKS="$(git diff -U0 "$BASE_SHA" -- scripts/evaluation/pregen_binfill_demo.py | grep '^@@')"
while read -r H; do
    [[ -z "$H" ]] && continue
    case "$H" in
        *"@@ def cmd_finalize(args):"|*"@@ def main():") ;;
        *) echo "E0 pregen 改动落在 cmd_finalize/main 之外：$H"; CODE_SAME=0 ;;
    esac
done <<< "$HUNKS"
git diff "$BASE_SHA" -- scripts/evaluation/pregen_binfill_demo.py | grep -q -- '--max-fail-rate' || CODE_SAME=0
BENCH_HEAD="$(git -C third_party/robomme_benchmark rev-parse HEAD)"
echo "E0_CODE code_same=$CODE_SAME benchmark_head=$BENCH_HEAD"
read -r -d '' PYDIFF <<'PY'
import json, sys
from pathlib import Path
store, old_path = Path(sys.argv[1]), Path(sys.argv[2])
new = json.loads((store / "index.json").read_text(encoding="utf-8"))
old = json.loads(old_path.read_text(encoding="utf-8"))
man = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
ok_new = {k for k, v in new.items() if v.get("demo_status") == "ok"}
ok_old = {k for k, v in old.items() if v.get("demo_status") == "ok"}
d_diff = sorted(k for k in ok_new & ok_old if new[k]["D"] != old[k]["D"])
diff = {"only_new": sorted(ok_new - ok_old), "only_old": sorted(ok_old - ok_new),
        "d_differs": {k: {"new": new[k]["D"], "old": old[k]["D"]} for k in d_diff},
        "failed_keys": man["failed_keys"], "planned": man["planned"], "ok": man["ok"], "failed": man["failed"],
        "fail_rate": man["failed"] / man["planned"], "old_index": str(old_path)}
(store / "demo-prefix-diff.json").write_text(json.dumps(diff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(man["ok"], man["failed"], len(diff["only_new"]) + len(diff["only_old"]) + len(d_diff))
PY
read -r N_OK N_FAILED DIFF_VS_OLD <<< "$("${CLIENT_UV[@]}" -c "$PYDIFF" "$STORE" "$OLD_INDEX")"
V=PASS
[[ "$CODE_SAME" == 1 && "$BENCH_HEAD" == "$BENCH_SHA" && -n "${DIFF_VS_OLD:-}" ]] || V=FAIL
LINE="DEMO_PREFIX_SRC=$V code_same=$CODE_SAME benchmark=${BENCH_HEAD:0:7} shards_ok=$NOK/$SHARDS ok=${N_OK:-?} failed=${N_FAILED:-?} diff_vs_old=${DIFF_VS_OLD:-?}"
printf '%s\n' "$LINE" > "$STORE/E0.txt.tmp" && mv "$STORE/E0.txt.tmp" "$STORE/E0.txt"
echo "$LINE"
[[ "$V" == PASS ]] || exit 1
echo "DEMO_POOL_DONE $(date -u +%FT%TZ)"
