#!/usr/bin/env bash
# 官方16×100 modulation + motion concat 1×1024、窗口步长16、budget192 的开训前闸门 G2→G6（1122 方案第五节、第二部分 runbook）。
# 参数以旧 orig100 建库档案 docs/dataset-build-doc/16task-orig100-1600ep-9e109419/launch.md 与
# v1-store/logs/o16-20261028/post-build-checks.sh 的真实命令为准，只把 subsample/stride/budget/目录换成 grid16 新值。
#
# 调用时点：须在 t1s 结束且 8 卡空闲后由 o16s16-chain 调用（E/G2/G4/G5 要占卡）。
# 用法（须在主检出运行；worktree 没有 v1-store 与 third_party 子模块）：
#   bash scripts/training/o16s16_gates.sh              # 依次跑 E→G2→G3→G4→G5→G6，任一 FAIL 即非零退出
#   DRY_RUN=1 bash scripts/training/o16s16_gates.sh    # 只打印全部命令，不执行、不建目录
#   GATES="G4 G6" bash scripts/training/o16s16_gates.sh  # 只跑列出的闸门（续跑用；默认全部）
# 环境变量：GATE_GPU（父进程/导出/G3 证据用卡，默认 0）、ONLINE_GPU（G5 sidecar 物理卡号，默认 1）、
#           RGBROOT（G3 证据所需配对训练缓存根，默认主会话下载的子集根 v1-store/external/rgb-gtxyz-g3-subset）。
#
# E（证据重建）：grid16 库只能用恢复版 latent（v1-store/cache/orig16-wan-timepool1024-recovery/wan-latents，
#   metadata sha 3650385…），其导入收据是 wan-cache-import.restored.json；旧 validation/orig16-stage2 的
#   G0/G2/G3 绑定旧收据（1a97618…）与旧 latent metadata（824ef22…），打包时 validate_stage2_cache_evidence
#   会拒绝，所以先在新目录 validation/orig16-stage2-grid16 重建 G0/G2/G3（命令同旧档案，只换收据/latent/输出，
#   并传 --grid-subsample 1：G0 从全部 grid16 格抽样，自然含奇数格）。
set -euo pipefail
[[ $# == 0 ]] || { echo '本脚本不接受参数' >&2; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
DRY_RUN=${DRY_RUN:-0}
GATES=${GATES:-"E G2 G3 G4 G5 G6"}
GATE_GPU=${GATE_GPU:-0}
ONLINE_GPU=${ONLINE_GPU:-1}

# ── 固定输入（旧档案同值）──
V1="$ROOT/v1-store"
NAME=16task-orig100-1600ep-9e109419
LIB="$V1/datasets/$NAME"
MANI="$LIB/meta/episode_manifest.json"
INMANI="$LIB/meta/input_manifest.json"
RAW=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/source/h5
MJIDX=/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/index
MJW=/scratch/hongze/worktrees/vla-s2-memexport-20261026
CKPT=/scratch/hongze/MotionJEPA/runs/stage2-full144-1025-a/checkpoint_epoch_105.pt
CKPT_SHA=27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035
ENC_RUN_DIR="$V1/external/motionjepa/stage2-full144-1025-a"
STAGE2_VENV="$V1/venvs/wan-stage2"
ROBOMME_PY=/scratch/hongze/micromamba/envs/robomme/bin/python
# encoder config.data.data_root（dataset-rgb-gtxyz-16views-stride2-v1）已删，血缘 build_spec 用保留的 keep-meta 副本。
LINEAGE_SPEC=/scratch/hongze/dataset-rgb-gtxyz-16views-stride2-v1-keep-meta/control/build_spec.json
RGBROOT=${RGBROOT:-$ROOT/v1-store/external/rgb-gtxyz-g3-subset}
# ── grid16 新值 ──
RECOVERY="$V1/cache/orig16-wan-timepool1024-recovery"
LATENTS="$RECOVERY/wan-latents"
RECEIPT="$RECOVERY/wan-cache-import.restored.json"
VALID16="$V1/validation/orig16-stage2-grid16"
TOKENS16="$LIB/motion-tokens-stage2-grid16"
OLD_TOKENS="$LIB/motion-tokens-stage2"
STORE16="$LIB/motion-stage2-1024-grid16"
LAYOUT16=motion-stage2-1x1024-grid16-demopad17-v1
HISTORY="$ROOT/src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-32frame-4x4-motion-stage2-1x1024-s16-orig16.yaml"
SEGMENTS=2496            # 非空段数（恢复版 metadata totals.segments，与旧 grid32 导出同为 2496）
ROWS=45228               # grid16 全部窗数（导入收据 counts.windows）
BUDGET=192
ES_OUT="$V1/reports/motion/o16s16-evalbound"
ES_MAX_STEPS=1300        # 沿用 mv2-evalbound 档案口径
TASKS=(BinFill ButtonUnmask ButtonUnmaskSwap InsertPeg MoveCube PatternLock PickHighlight PickXtimes
       RouteStick StopCube SwingXtimes VideoPlaceButton VideoPlaceOrder VideoRepick VideoUnmask VideoUnmaskSwap)
FINAL_ROOT="$V1/validation/orig16-modul-s16-60k"
LOGDIR="$V1/logs/o16s16-gates"

# ── 运行环境（同 post-build-checks.sh）──
export UV_CACHE_DIR=/scratch/hongze/.cache/uv UV_PROJECT_ENVIRONMENT="$ROOT/.venv"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MMEVLA_V1_STORE="$V1" XDG_CACHE_HOME="$V1/cache/xdg" HF_HOME="$V1/cache/hf" OPENPI_DATA_HOME="$V1/models"
unset PYTHONPATH PYTHONHOME

want() { [[ " $GATES " == *" $1 "* ]]; }

# 打印并执行一条命令，输出 tee 到 $LOGDIR/<名>.log；DRY_RUN=1 只打印。返回命令本身的退出码。
step() {
  local name=$1; shift
  printf '+ [%s] ' "$name"; printf '%q ' "$@"; printf '\n'
  [[ $DRY_RUN == 1 ]] && return 0
  local rc
  set +e
  "$@" 2>&1 | tee "$LOGDIR/$name.log"
  rc=${PIPESTATUS[0]}
  set -e
  echo "STEP_EXIT_CODE=$rc step=$name"
  return "$rc"
}

# 判定：日志里须出现锚定行首的期望行，否则打印 FAIL 判定行并退出。
expect_line() {
  local name=$1 pattern=$2 verdict=$3
  [[ $DRY_RUN == 1 ]] && { echo "  （DRY_RUN）期望 $LOGDIR/$name.log 含: $pattern"; return 0; }
  if ! grep -aqE "$pattern" "$LOGDIR/$name.log"; then
    echo "$verdict=FAIL reason=missing_line log=$LOGDIR/$name.log pattern=$pattern"
    echo "O16S16_GATES=FAIL at=$verdict"
    exit 1
  fi
}

run_or_fail() {
  local verdict=$1; shift
  if ! step "$@"; then
    echo "$verdict=FAIL reason=exit_code step=$1 log=$LOGDIR/$1.log"
    echo "O16S16_GATES=FAIL at=$verdict"
    exit 1
  fi
}

echo "O16S16_GATES_START root=$ROOT head=$(git rev-parse HEAD) gates=\"$GATES\" dry_run=$DRY_RUN gate_gpu=$GATE_GPU online_gpu=$ONLINE_GPU"
# ── 输入存在性（只读）：缺任何一项即 FAIL；DRY_RUN 只报告 ──
missing=()
for path in "$MANI" "$INMANI" "$RAW" "$MJIDX" "$MJW" "$CKPT" "$ENC_RUN_DIR/config.yaml" "$STAGE2_VENV/bin/python" \
    "$ROBOMME_PY" "$LINEAGE_SPEC" "$LATENTS/metadata.json" "$RECEIPT" "$OLD_TOKENS/export_record.json" "$HISTORY" \
    "$LIB/source" "$LIB/framesamp-4x4"; do
  [[ -e $path ]] || missing+=("$path")
done
if want E && [[ ! -d $RGBROOT ]]; then missing+=("$RGBROOT"); fi
if ((${#missing[@]})); then
  echo "O16S16_INPUTS=FAIL missing=${#missing[@]} paths=${missing[*]}"
  [[ $DRY_RUN == 1 ]] || { echo "O16S16_GATES=FAIL at=O16S16_INPUTS"; exit 1; }
else
  echo "O16S16_INPUTS=PASS"
fi
[[ $DRY_RUN == 1 ]] || mkdir -p "$LOGDIR"

# ── E：grid16 证据重建（G3 打包的前置；CPU + 1 卡）──
if want E; then
  [[ $DRY_RUN == 1 ]] || mkdir -p "$VALID16"
  run_or_fail O16S16_EVIDENCE e-g0 env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python \
    scripts/dataset/check_stage2_validation_inputs.py --manifest "$MANI" --latents "$LATENTS" --source "$LIB/source" \
    --h5-root "$RAW" --input-manifest "$INMANI" --mj-index "$MJIDX" --cache-import-receipt "$RECEIPT" \
    --grid-subsample 1 --out "$VALID16/G0.json"
  run_or_fail O16S16_EVIDENCE e-g2 env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python \
    scripts/dataset/wan_stage2/orig16_wan_cache.py check --samples "$VALID16/G0.json" --receipt "$RECEIPT" \
    --raw-dir "$RAW" --out "$VALID16/G2.json" --grid-subsample 1
  run_or_fail O16S16_EVIDENCE e-g3 env CUDA_VISIBLE_DEVICES="$GATE_GPU" UV_PROJECT_ENVIRONMENT="$STAGE2_VENV" \
    uv run --project scripts/dataset/wan_stage2 --no-sync python scripts/dataset/wan_stage2/validate_stage2.py \
    --gate G3 --samples "$VALID16/G0.json" --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" --rgb-root "$RGBROOT" \
    --mj-root "$MJW" --cache-import-receipt "$RECEIPT" --grid-subsample 1 --out "$VALID16/G3.json"
  [[ $DRY_RUN == 1 ]] || echo "O16S16_EVIDENCE=PASS root=$VALID16"
fi

# ── G2：encoder 导出（grid-subsample 1，全部 45228 窗）+ 偶数格与旧 grid32 表逐字节比对 ──
if want G2; then
  run_or_fail STAGE2_EXPORT g2-export env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$GATE_GPU" "$STAGE2_VENV/bin/python" \
    scripts/dataset/wan_stage2/export_stage2_memory.py --ckpt "$CKPT" --expected-sha256 "$CKPT_SHA" \
    --latents "$LATENTS" --manifest "$MANI" --input-manifest "$INMANI" --out "$TOKENS16" --grid-subsample 1 \
    --batch 1 --check-dropout 64 --expected-counts "$SEGMENTS,$ROWS" --lineage-build-spec "$LINEAGE_SPEC"
  expect_line g2-export "^STAGE2_EXPORT=PASS segments=$SEGMENTS rows=$ROWS\$" STAGE2_EXPORT
  # grid16 第 2j 行对旧 grid32 第 j 行：先逐字节（mode=bitexact）；不等时 max_abs≤1e-4 判 mode=tol
  # （旧表 latent 与恢复版为跨次拷贝，note=cross_latent_copy），否则 FAIL 停链。
  run_or_fail EVEN_GRID_BITEXACT g2-even env CUDA_VISIBLE_DEVICES= uv run --no-sync python - "$OLD_TOKENS" "$TOKENS16" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
old_root, new_root = map(Path, sys.argv[1:])
old_keys = json.loads((old_root / "export_record.json").read_text())["segment_keys"]
new_keys = json.loads((new_root / "export_record.json").read_text())["segment_keys"]
if sorted(old_keys) != sorted(new_keys):
    print(f"EVEN_GRID_BITEXACT=FAIL reason=segment_set old={len(old_keys)} new={len(new_keys)}")
    raise SystemExit(1)
TOL = 1e-4
compared = mismatches = 0
max_abs = 0.0
for key in old_keys:
    old = np.fromfile(old_root / f"{key}.f32.bin", dtype="<f4").reshape(-1, 1024)
    new = np.fromfile(new_root / f"{key}.f32.bin", dtype="<f4").reshape(-1, 1024)
    even = new[0::2]
    if even.shape != old.shape:
        print(f"EVEN_GRID_BITEXACT=FAIL reason=rows key={key} old={old.shape[0]} new_even={even.shape[0]}")
        raise SystemExit(1)
    # 非有限值会让容差比较失真：逐行比较前先查新旧两表，任一非有限即 FAIL（row 为该段内行号）。
    for name, arr in (("old", old), ("new_even", even)):
        finite = np.isfinite(arr).all(axis=1)
        if not finite.all():
            print(f"EVEN_GRID_BITEXACT=FAIL reason=nonfinite row={int(np.argmin(finite))} key={key} table={name}")
            raise SystemExit(1)
    compared += old.shape[0]
    rows = (even.view(np.uint32) != old.view(np.uint32)).any(axis=1)
    if rows.any():
        mismatches += int(rows.sum())
        segment_max = float(np.max(np.abs(even[rows].astype(np.float64) - old[rows].astype(np.float64))))
        if not np.isfinite(segment_max):
            print(f"EVEN_GRID_BITEXACT=FAIL reason=nonfinite_diff key={key}")
            raise SystemExit(1)
        max_abs = float(np.max([max_abs, segment_max]))
if mismatches == 0:
    print(f"EVEN_GRID_BITEXACT=PASS mode=bitexact segments={len(old_keys)} compared={compared} mismatches=0")
elif np.isfinite(max_abs) and max_abs <= TOL:
    print(f"EVEN_GRID_BITEXACT=PASS mode=tol max_abs={max_abs!r} mismatched_rows={mismatches} note=cross_latent_copy "
          f"segments={len(old_keys)} compared={compared}")
else:
    print(f"EVEN_GRID_BITEXACT=FAIL max_abs={max_abs!r} mismatched_rows={mismatches} tol={TOL} compared={compared}")
    raise SystemExit(1)
PY
  expect_line g2-even '^EVEN_GRID_BITEXACT=PASS ' EVEN_GRID_BITEXACT
fi

# ── G3：打包 grid16 库并全量 verify（CPU）──
if want G3; then
  run_or_fail MOTION_PACK_VERIFY g3-pack env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python \
    scripts/dataset/pack_motion_store.py pack --layout stage2 --latent-grid-subsample 1 \
    --latents-origin published-cache-with-local-fill --cache-import-receipt "$RECEIPT" --manifest "$MANI" \
    --tokens "$TOKENS16" --latents "$LATENTS" --out "$STORE16" --encoder-run-dir "$ENC_RUN_DIR" --raw-dir "$RAW" \
    --expected-ckpt-sha256 "$CKPT_SHA" --stage2-validation "$VALID16" \
    --stage2-source-pin scripts/dataset/wan_stage2/SOURCE_PIN.json
  run_or_fail MOTION_PACK_VERIFY g3-verify env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python \
    scripts/dataset/pack_motion_store.py verify --store "$STORE16" --resume --check-padding
  expect_line g3-verify "^VERIFY_MOTION=PASS scanned=$ROWS mismatches=0" MOTION_PACK_VERIFY
fi

# ── G4：预算（训练侧读 manifest，CPU；评估侧经真实 EnvRunner.reset 采集官方 test 集 es，需仿真环境与 1 卡渲染）──
if want G4; then
  run_or_fail TRAIN_ES_BOUND g4-train env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync python \
    scripts/training/tests/train_es_bound.py --manifest "$MANI" --layout "$LAYOUT16" --budget "$BUDGET"
  expect_line g4-train "^TRAIN_ES_BOUND=PASS .* budget=$BUDGET " TRAIN_ES_BOUND
  # 本机此前只有 4 任务（mv2-evalbound）的 test 集 es 记录，16 任务须现采；环境同 mv2-evalbound 档案。
  run_or_fail EVAL_ES_BOUND g4-eval-check env CUDA_VISIBLE_DEVICES= uv run --no-sync python \
    scripts/training/tests/eval_es_bound.py check --tasks "${TASKS[@]}"
  expect_line g4-eval-check '^ES_BOUND_TESTSET=READY ' EVAL_ES_BOUND
  [[ $DRY_RUN == 1 ]] || mkdir -p "$ES_OUT"
  reports=()
  for task in "${TASKS[@]}"; do
    run_or_fail EVAL_ES_BOUND "g4-eval-$task" env CUDA_VISIBLE_DEVICES="$GATE_GPU" \
      CUDA_CACHE_PATH="$V1/cache/cuda" MS_ASSET_DIR="$V1/cache/maniskill/data" \
      GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192 XDG_DATA_HOME="$V1/cache/xdg-data" \
      uv run --no-project --python "$ROBOMME_PY" python scripts/training/tests/eval_es_bound.py reset \
      --task "$task" --out "$ES_OUT/$task.json" --budget "$BUDGET" --max-steps "$ES_MAX_STEPS"
    reports+=("$ES_OUT/$task.json")
  done
  run_or_fail EVAL_ES_BOUND g4-eval-aggregate env CUDA_VISIBLE_DEVICES= uv run --no-sync python \
    scripts/training/tests/eval_es_bound.py aggregate --tasks "${TASKS[@]}" --reports "${reports[@]}" \
    --out "$ES_OUT/aggregate.json" --budget "$BUDGET" --max-steps "$ES_MAX_STEPS"
  expect_line g4-eval-aggregate "^EVAL_ES_BOUND=PASS .* budget=$BUDGET " EVAL_ES_BOUND
fi

# ── G5：在线 sidecar 与离线 grid16 表一致（父 GATE_GPU、sidecar ONLINE_GPU；同旧 g4-mod）──
# 输出须落在 G0 所在目录：online_restore_inputs.json 由此生成，train.py 写 stage2 快照时从 G0 父目录读取。
if want G5; then
  run_or_fail ONLINE_STAGE2 g5-online env CUDA_VISIBLE_DEVICES="$GATE_GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --no-sync python scripts/training/check_stage2_online.py --store "$STORE16" --samples "$VALID16/G0.json" \
    --history "$HISTORY" --framesamp-store "$LIB/framesamp-4x4" --h5-root "$RAW" --input-manifest "$INMANI" \
    --online-gpu "$ONLINE_GPU" --out "$VALID16/G4-mod.json"
  expect_line g5-online '^ONLINE_INPUT_IDENTITY=PASS windows=64 ' ONLINE_STAGE2
  # 方案要求抽样 ≥64 窗、奇数格 ≥16、含 demo 尾补帧窗；统计 G5 报告里实际在线编码的窗口，
  # 段类型与补帧数按 (key, source_grid) 回查 G0 窗口记录。
  run_or_fail ONLINE_STAGE2 g5-sample env CUDA_VISIBLE_DEVICES= uv run --no-sync python - "$VALID16/G0.json" "$VALID16/G4-mod.json" <<'PY'
import json, sys
g0, report = (json.load(open(p)) for p in sys.argv[1:])
by_id = {(w["key"], w["source_grid"]): w for w in g0["windows"]}
windows = [by_id[(r["key"], r["source_grid"])] for r in report.get("windows", [])]
odd = sum(w["source_grid"] % 2 == 1 for w in windows)
tail = sum(w["segment_kind"] == "demo" and w["pad_frames"] > 0 for w in windows)
ok = report.get("status") == "PASS" and len(windows) >= 64 and odd >= 16 and tail >= 1
print(f"ONLINE_STAGE2={'PASS' if ok else 'FAIL'} windows={len(windows)} odd_grid={odd} demo_tail={tail} "
      f"e2e_max_abs={report.get('metrics', {}).get('e2e_max_abs')}")
raise SystemExit(int(not ok))
PY
  expect_line g5-sample '^ONLINE_STAGE2=PASS ' ONLINE_STAGE2
fi

# ── G6：启动合同（用正式单条入口的 DRY_RUN：打印完整 TRAIN_CMD 并跑 check_orig16_stage2_launch.py，含 DATA_LINEAGE）──
if want G6; then
  run_or_fail LAUNCH_CONTRACT g6-launch env DRY_RUN=1 TRAIN_HEAD="$(git rev-parse HEAD)" FINAL_ROOT="$FINAL_ROOT" \
    bash scripts/training/run_orig16_modul_s16_60k.sh
  expect_line g6-launch '^DATA_LINEAGE=PASS ' LAUNCH_CONTRACT
  expect_line g6-launch '^LAUNCH_CONTRACT=PASS ' LAUNCH_CONTRACT
fi

if [[ $DRY_RUN == 1 ]]; then
  echo "O16S16_GATES=DRY_RUN gates=\"$GATES\" （只打印命令，未执行任何闸门）"
else
  echo "O16S16_GATES=PASS gates=\"$GATES\""
fi
