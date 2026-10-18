#!/usr/bin/env bash
# 1017-test-xhard5-12per 的 ckpt 下载与核验（计划 4.3 第 6 项、第二部分二节）。主会话在 NFS 检出上手动执行；不在评估单元里跑。
# bucket 名一律从参数读入——两条新 run 的 bucket 由训练侧交付后才知道，本脚本不写死、不猜。
#
# 用法：
#   fetch_ckpt.sh run <bucket_id> <run_dir_name> <step>
#       自家 run（512 / 512+motion）：落点 $MODELS_ROOT/<run_dir_name>/（MODELS_ROOT 默认 $EVAL_REPO/v1-store/models）。顺序：
#       ① `hf buckets list <bucket> -R` 断言含 <step>/、history_config.resolved.yaml、history_config.resolved.sha256、
#          SHA256SUMS.pre.txt，motion run 另需 motion_provenance.json（缺任一即停）；
#          motion run 的判法沿用仓库约定：run 目录名含 `-motion-`（如 mixed8-200-m8x8-modul-motion-b128-80k、robomme-vla-modul-motion-80k-v1）；
#       ② `hf buckets sync hf://buckets/<bucket> <落点> --include … --dry-run`；③ 同一命令去掉 --dry-run 实际下载；
#       ④ 本次 <step>/ 下每个文件都必须在 SHA256SUMS.pre.txt 里并 `sha256sum --check --strict`（覆盖不全即失败）；
#          run 级根文件（history_config.resolved.*、motion_provenance.json）在清单里就核，不在清单里则列为豁免并打印；
#       ⑤ check_checkpoint.py（RUNS 未登记该 run 时这一步会失败——先登记再重跑本步即可，前四步不重下）。
#       hf CLI 写法按 huggingface_hub 1.30.0 `hf buckets {list,sync} --help` 核过：`hf buckets list <id> -R`、
#       `hf buckets sync <src> <dst> --include <pat> [--dry-run]`。
#       判定行：CKPT_SHA=PASS run=<r> step=<s> files=<n>（之后 check_checkpoint.py 打 PARAM_TREE_EXACT 等）。
#   fetch_ckpt.sh encoder <bucket_id> <encoder_run_name> <文件白名单,逗号分隔>
#       motion 模型的 encoder run 目录：落点 $EVAL_REPO/v1-store/external/motionjepa/<encoder_run_name>/，只取白名单根文件
#       （如 checkpoint_epoch_72.pt,config.yaml,SHA256SUMS.pre.txt），有 SHA256SUMS.pre.txt 就逐文件核。
#   fetch_ckpt.sh official <源 run 根目录> <zip 文件> <zip sha256 前缀>
#       官方 79999：先核 zip 的 sha256 以给定前缀开头（计划记 HF 前缀 387d4bd5）→ rsync 源 run 根（含 history_config.txt 与 79999/）
#       到 $EVAL_REPO/v1-store/models/official-mme-vla/perceptual-framesamp-modul/ → 逐文件 sha256，打印可直接粘进
#       check_checkpoint.py::RUNS['perceptual-framesamp-modul']['file_sha256'] 的字典。判定行 OFFICIAL_CKPT_SHA=PASS|FAIL。
#   fetch_ckpt.sh prov-dryrun <step 目录绝对路径>
#       只握手不评估：按 run 快照与 motion_provenance.json 起一次 sidecar 并做 provenance 比对（MMEVLA_MOTION_PROV_RELAX
#       照常生效），握手完即关；需要 GPU，在占位 job 的 srun step 里跑。判定行 MOTION_PROV=PASS exempt=<n> other_diffs=0。
# 环境：HF_BIN（默认 hf）；token 由 hf 自己从 $HF_HOME 读，不经命令行。
set -euo pipefail
source "$(dirname "$0")/env.sh"
HF_BIN="${HF_BIN:-hf}"
MODELS_ROOT="${MODELS_ROOT:-$EVAL_REPO/v1-store/models}"
MODE="${1:?需要模式 run|encoder|official|prov-dryrun}"; shift

verify_sums() {   # $1 = 落点；$2 = 必须被 SHA256SUMS.pre.txt 覆盖的文件清单；$3 = 豁免清单（在清单里就核，不在只打印）
    local dst="$1" list="$2" exempt="${3:-/dev/null}" selected missing
    selected="$dst/SHA256SUMS.selected.$$.txt"
    missing="$(awk 'NR==FNR{have[$2]=1; next} !($0 in have){print}' "$dst/SHA256SUMS.pre.txt" "$list")"
    if [[ -n "$missing" ]]; then
        echo "CKPT_SHA=FAIL reason=not-in-SHA256SUMS.pre.txt files=$(printf '%s\n' "$missing" | wc -l)" >&2
        printf '  %s\n' $missing >&2
        exit 2
    fi
    while IFS= read -r f; do
        grep -q -- "  $f\$" "$dst/SHA256SUMS.pre.txt" || echo "SHA_EXEMPT file=$f（run 级根文件，不在 SHA256SUMS.pre.txt 里）" >&2
    done < "$exempt"
    cat "$list" "$exempt" | awk 'NR==FNR{want[$0]=1; next} ($2 in want){print}' - "$dst/SHA256SUMS.pre.txt" > "$selected"
    [[ -s "$selected" ]] || { echo 'SHA256SUMS.pre.txt 里没有本次文件'; exit 2; }
    (cd "$dst" && sha256sum --check --strict "$selected") >&2
    wc -l < "$selected"
}

case "$MODE" in
run)
    BUCKET_ID="${1:?需要 bucket id}"; RUN="${2:?需要 run 目录名}"; STEP="${3:?需要 step}"
    [[ "$RUN" =~ ^[A-Za-z0-9_.-]+$ && "$STEP" =~ ^[0-9]+$ ]] || exit 2
    DST="$MODELS_ROOT/$RUN"
    [[ ! -e "$DST/$STEP" ]] || { echo "step 目录已存在，拒绝覆盖：$DST/$STEP"; exit 2; }
    ROOT_FILES=(history_config.resolved.yaml history_config.resolved.sha256)
    [[ "$RUN" != *-motion-* ]] || ROOT_FILES+=(motion_provenance.json)
    mkdir -p "$DST"
    LISTING="$DST/bucket-list.$STEP.txt"
    "$HF_BIN" buckets list "$BUCKET_ID" -R > "$LISTING"
    for need in "$STEP/" "${ROOT_FILES[@]}" SHA256SUMS.pre.txt; do
        grep -q -- "$need" "$LISTING" || { echo "bucket $BUCKET_ID 缺 $need，停止"; exit 2; }
    done
    INCLUDES=(--include "$STEP/*" --include SHA256SUMS.pre.txt)
    for f in "${ROOT_FILES[@]}"; do INCLUDES+=(--include "$f"); done
    "$HF_BIN" buckets sync "hf://buckets/$BUCKET_ID" "$DST" "${INCLUDES[@]}" --dry-run
    "$HF_BIN" buckets sync "hf://buckets/$BUCKET_ID" "$DST" "${INCLUDES[@]}"
    LIST="$DST/fetch-list.$STEP.txt"; EXEMPT="$DST/fetch-exempt.$STEP.txt"
    (cd "$DST" && find "$STEP" -type f | sort) > "$LIST"
    printf '%s\n' "${ROOT_FILES[@]}" > "$EXEMPT"
    N="$(verify_sums "$DST" "$LIST" "$EXEMPT" | tail -n 1)"
    echo "CKPT_SHA=PASS run=$RUN step=$STEP files=$N"
    JAX_PLATFORMS=cpu "${SERVER_UV[@]}" "$EVAL_REPO/scripts/evaluation/check_checkpoint.py" "$DST/$STEP"
    ;;
encoder)
    BUCKET_ID="${1:?需要 bucket id}"; NAME="${2:?需要 encoder run 名}"; ONLY="${3:?需要文件白名单}"
    [[ "$NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
    DST="$EVAL_REPO/v1-store/external/motionjepa/$NAME"
    mkdir -p "$DST"
    INCLUDES=(); IFS=',' read -r -a FILES <<< "$ONLY"
    for f in "${FILES[@]}"; do
        [[ "$f" != */* && "$f" != *..* ]] || exit 2
        [[ ! -e "$DST/$f" ]] || { echo "已存在，拒绝覆盖：$DST/$f"; exit 2; }
        INCLUDES+=(--include "$f")
    done
    "$HF_BIN" buckets sync "hf://buckets/$BUCKET_ID" "$DST" "${INCLUDES[@]}" --dry-run
    "$HF_BIN" buckets sync "hf://buckets/$BUCKET_ID" "$DST" "${INCLUDES[@]}"
    if [[ -f "$DST/SHA256SUMS.pre.txt" ]]; then
        LIST="$DST/fetch-list.encoder.txt"; printf '%s\n' "${FILES[@]}" | grep -vx SHA256SUMS.pre.txt > "$LIST"
        N="$(verify_sums "$DST" "$LIST" | tail -n 1)"
        echo "ENCODER_SHA=PASS run=$NAME files=$N"
    else
        echo "ENCODER_SHA=UNVERIFIED run=$NAME reason=no-SHA256SUMS.pre.txt"
    fi
    ;;
official)
    SRC="${1:?需要源 run 根目录}"; ZIP="${2:?需要 zip 文件}"; PREFIX="${3:?需要 zip sha256 前缀}"
    [[ "$PREFIX" =~ ^[0-9a-f]{8,64}$ && -f "$SRC/history_config.txt" && -d "$SRC/79999" ]] || exit 2
    GOT="$(sha256sum "$ZIP" | cut -d' ' -f1)"
    if [[ "$GOT" != "$PREFIX"* ]]; then
        echo "OFFICIAL_CKPT_SHA=FAIL zip_sha256=$GOT expect_prefix=$PREFIX"; exit 1
    fi
    DST="$MODELS_ROOT/official-mme-vla/perceptual-framesamp-modul"
    [[ ! -e "$DST/79999" ]] || { echo "已存在，拒绝覆盖：$DST/79999"; exit 2; }
    mkdir -p "$DST"
    rsync -a "$SRC/history_config.txt" "$SRC/79999" "$DST/"
    (cd "$DST/79999" && find . -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$DST/SHA256SUMS.official.txt"
    echo "OFFICIAL_CKPT_SHA=PASS zip_sha256=$GOT files=$(wc -l < "$DST/SHA256SUMS.official.txt")"
    echo "# 粘进 check_checkpoint.py::RUNS['perceptual-framesamp-modul']['file_sha256']："
    awk '{printf "    %c%s%c: %c%s%c,\n", 39, $2, 39, 39, $1, 39}' "$DST/SHA256SUMS.official.txt"
    ;;
prov-dryrun)
    CKPT="${1:?需要 step 目录绝对路径}"
    [[ "$CKPT" == /* && -d "$CKPT" ]] || exit 2
    "${SERVER_UV[@]}" - "$CKPT" <<'PY'
import json, os, pathlib, sys
from mme_vla_suite.policies.policy_config import _load_resolved_snapshot, _motion_online_gpu
from mme_vla_suite.policies.motion_client import MotionEncoderClient
run_root = pathlib.Path(sys.argv[1]).resolve().parent
cfg, enabled = _load_resolved_snapshot(run_root)
assert enabled, f"{run_root} 不是 motion run"
prov = json.loads((run_root / "motion_provenance.json").read_text(encoding="utf-8"))
store = {"vae": prov["vae"], "encoder": prov["encoder"]}
client = MotionEncoderClient(source_run=str(cfg.motion.source_run), online_gpu=_motion_online_gpu(cfg.motion),
                             store_provenance=store, expected_ckpt_sha256=store["encoder"].get("checkpoint_sha256"))
relaxed = list(getattr(client, "relaxed_diffs", []))
client.close()
print(f"MOTION_PROV=PASS exempt={len({line.split(':')[0].split('.', 1)[1] for line in relaxed})} "
      f"relaxed_lines={len(relaxed)} other_diffs=0", flush=True)
PY
    ;;
*) echo "未知模式 $MODE"; exit 2 ;;
esac
