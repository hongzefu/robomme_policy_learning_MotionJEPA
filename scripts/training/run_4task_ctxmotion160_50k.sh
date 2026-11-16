#!/usr/bin/env bash
# 4task 原训练库 + 512 帧记忆 + motion160（旧 v7 encoder wan-full1600-filter2-b176x4-72ep-a ep72，每窗 1×768）
# context 版 50k 正式训练 / 20 步冒烟 runner。
# 依据：根目录 1115-4task-ctx-motion160-v7-50k-plan.html §5 第 1–10 道、第二部分 B3、runbook E3/E4/E6。
# 由 c55a6bc:scripts/training/run_4task_wantp768_50k.sh 复制后改：run 名、history YAML、motion 表根、G0 环 3 删除、
# 环 4 重写为输入 sha（第 2 道）、环 5 调四环血缘脚本（第 1 道）、新增配置交叉核对（第 3 道）与编码环境检查（第 4 道）、
# preflight 带 --motion-*（第 5 道）、训练后读回改为 context 断言、新增 --check-only。
# 覆盖项照抄 54.62% 那条 v2-1600ep-m8x8-modul-motion-b128-80k：TrainConfig mme_vla_suite_b128_80k 全部超参不变，
# 只覆盖 --num_train_steps=50001（最后一份 checkpoint 为 50000），history YAML 换成 …-ctx-8frame-8x8-motion-v7ep72.yaml。
#
# 用法：
#   bash scripts/training/run_4task_ctxmotion160_50k.sh <TRAIN_HEAD 40 位 sha>                          # 正式 50001 步
#   bash scripts/training/run_4task_ctxmotion160_50k.sh <TRAIN_HEAD> --smoke20 --run-name v17-smoke20   # 第 6 道冒烟 20 步
#   bash scripts/training/run_4task_ctxmotion160_50k.sh <任意> [--smoke20] [--run-name N] --dry-run      # 只打印 TRAIN_ARGS
#   bash scripts/training/run_4task_ctxmotion160_50k.sh <TRAIN_HEAD> --check-only                       # HEAD/clean + 第 1–5 道后退出
#
# --check-only：不写 run 日志、不建记录目录、不查显卡空闲、不训练；末行 CHECK_ONLY=PASS（任一道 FAIL 即非零退出）。
# 正式与冒烟起跑前照常再跑一遍同样的第 1–5 道检查（P-LINEAGE：DATA_LINEAGE=FAIL 即 exit 4，不得豁免）。
set -o pipefail                     # 外层记录正常返回和报错的退出码；外部强杀除外。

usage() { printf '用法: %s <TRAIN_HEAD> [--smoke20] [--run-name NAME] [--dry-run|--check-only]\n' "$0" >&2; exit 2; }
TRAIN_HEAD="${1:-}"; [ -n "$TRAIN_HEAD" ] || usage; shift
SMOKE=0; DRY=0; CHECK_ONLY=0; RUN_NAME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke20) SMOKE=1 ;;
    --dry-run) DRY=1 ;;
    --check-only) CHECK_ONLY=1 ;;
    --run-name) [ $# -ge 2 ] || usage; RUN_NAME="$2"; shift ;;
    *) usage ;;
  esac
  shift
done
[ "$DRY" = 1 ] && [ "$CHECK_ONLY" = 1 ] && usage

MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
FORMAL_RUN=v2-1600ep-m8x8-ctx-motion-b128-50k
SMOKE_RUN=v17-smoke20
if [ "$SMOKE" = 1 ]; then
  RUN="${RUN_NAME:-$SMOKE_RUN}"; STEPS=20; EXPECT_CKPTS="19"
  [ "$RUN" = "$SMOKE_RUN" ] || { printf '冒烟 run 名固定为 %s，拒绝 %s\n' "$SMOKE_RUN" "$RUN" >&2; exit 2; }
else
  RUN="${RUN_NAME:-$FORMAL_RUN}"; STEPS=50001
  EXPECT_CKPTS="5000 10000 15000 20000 25000 30000 35000 40000 45000 50000"
fi
CONFIG=mme_vla_suite_b128_80k
HC=perceptual-framesamp-ctx-8frame-8x8-motion-v7ep72.yaml
HC_SHA=2e98e191db566c2a0f5c107e28a871493f2207cd65d00600b88ae5eab7b6d2d3   # 定稿 YAML 文件字节 sha256（B1）
HC_ORIG_REF=2f10473:src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion.yaml   # 54.62% 原 run 的 YAML
NORM_SHA=856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173
MANIFEST_ID=4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918   # load_manifest 重算的 canonical identity（不是文件字节摘要）
MOTION_BIN_SHA=03fb46e150dd9015f264f9bd8f08b35883d84da40e9e8080147d2af5fef9b6c5     # motion/motion_token.f32.bin 完整文件 sha256
MOTION_META_SHA=d3a518011c80a115f7b398458a510a6f128f47e76c03fc251e59dab4c9c56268    # motion/meta/store_meta.json 文件 sha256
MOTION_LAYOUT=motion-768-grid16-demopad17-v1                                         # 表的 layout 字面量
MOTION_ROWS=71316
ENCODER_SHA=0c1986297ccc0ab1913910f33a09ec74ba4c208844d0f5d72dd7ba59e0d9e3ca        # wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt
V1_STORE="$MAIN/v1-store"
DS="$V1_STORE/datasets/4task-v2-1600ep-604f16da"
MOTION="$DS/motion"
ENCODER_CKPT="$V1_STORE/external/motionjepa/wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt"
LINEAGE_RECORDS="$MAIN/docs/dataset-build-doc/4task-v2-1600ep-604f16da/records"
WAN_VENV="$V1_STORE/venvs/wan"
ASSETS_DIR="$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
RUN_ROOT="$V1_STORE/train-runs/$CONFIG/$RUN"
LOG="$V1_STORE/logs/$RUN.log"
REC="$V1_STORE/bench/$RUN"

TRAIN_ARGS=(
  "$CONFIG"
  --exp-name "$RUN"
  --assets-base-dir "$V1_STORE/train-assets"
  --data.assets.assets-dir "$ASSETS_DIR"
  --data.assets.asset-id robomme
  --checkpoint-base-dir "$V1_STORE/train-runs"
  --dataset-path "$DS/framesamp-8x8"
  --model.history-config "$HC"
  "--num_train_steps=$STEPS"
)

if [ "$DRY" = 1 ]; then
  printf 'RUN=%s STEPS=%s SMOKE=%s EXPECT_CKPTS=%s\n' "$RUN" "$STEPS" "$SMOKE" "$EXPECT_CKPTS"
  printf 'TRAIN_ARGS='; printf ' %q' "${TRAIN_ARGS[@]}"; printf '\n'
  exit 0
fi

# ── 第 1–5 道检查（--check-only 与起跑前共用；在 body 子 shell 内 set -e 下调用，任一道失败即退出）──
gates() {
  # 第 2 道 INPUT_SHA（G0 环 1/2/4）：输入逐字节同原 run
  JAX_PLATFORMS=cpu uv run --no-sync python - "$DS" "$MOTION" "$MANIFEST_ID" "$MOTION_BIN_SHA" "$MOTION_META_SHA" \
      "$MOTION_LAYOUT" "$MOTION_ROWS" "$ENCODER_CKPT" "$ENCODER_SHA" "$ASSETS_DIR/robomme/norm_stats.json" "$NORM_SHA" <<'PY'
import json, os, pathlib, sys
from mme_vla_suite.datastore.manifest import load_manifest
from mme_vla_suite.datastore import motion_store as ms

(ds, motion, manifest_id, bin_sha, meta_sha, layout, rows, enc_ckpt, enc_sha, norm_path, norm_sha) = sys.argv[1:]
ds, motion, rows = pathlib.Path(ds), pathlib.Path(motion), int(rows)
fails = []
def need(ok, ring, msg):
    print(f"G0_RING={'PASS' if ok else 'FAIL'} ring={ring} {msg}", flush=True)
    if not ok:
        fails.append(f"{ring}:{msg}")
def sha(p):
    return ms.sha256_file(p) if pathlib.Path(p).is_file() else None

# 环 1：输入 4 个 H5 sha256——本地 meta/input_manifest.json 与 bucket 那份逐字节相同
local_im, bucket_im = ds / "meta/input_manifest.json", ds / "meta.bucket/input_manifest.json"
same = local_im.is_file() and bucket_im.is_file() and local_im.read_bytes() == bucket_im.read_bytes()
need(same, 1, f"input_manifest 本地与 bucket 逐字节相同={same}")
im = json.loads(local_im.read_text()) if local_im.is_file() else {"files": {}, "raw_dir": None}
files = im.get("files", {})
need(len(files) == 4 and all(len(str(v.get("sha256", ""))) == 64 for v in files.values()), 1, f"H5 文件数={len(files)}")
im_sha = sha(local_im)
# 环 2：canonical manifest identity 用 load_manifest 重算（不是文件字节摘要）
man = load_manifest(ds / "meta/episode_manifest.json")
need(man["sha256"] == manifest_id, 2, f"manifest_identity={man['sha256']}")
need(man.get("raw_dir") == im.get("raw_dir"), 2, f"manifest.raw_dir 与 input_manifest.raw_dir 相同={man.get('raw_dir') == im.get('raw_dir')}")
# 环 4：离线 motion 表（54.62% 原 run 同一份）
need(motion.is_dir() and not motion.is_symlink() and motion.resolve().parent == ds.resolve(), 4, f"motion 根={motion} 实体目录")
need("MMEVLA_MOTION_STORE" not in os.environ, 4, "MMEVLA_MOTION_STORE 未设置")
got_meta = sha(motion / "meta/store_meta.json")
need(got_meta == meta_sha, 4, f"store_meta.json sha256={got_meta}")
got_bin = sha(motion / "motion_token.f32.bin")
need(got_bin == bin_sha, 4, f"motion_token.f32.bin sha256={got_bin}")
mm = ms.MotionMeta.load(motion)
prov = mm.provenance
print(f"G0_INFO ms.LAYOUT={ms.LAYOUT} 表 layout={mm.raw.get('layout')}（按字面量 {layout} 比）", flush=True)
need(mm.raw.get("layout") == layout and mm.num_rows == rows, 4, f"layout={mm.raw.get('layout')} rows={mm.num_rows}")
got_enc = sha(enc_ckpt)
enc = prov.get("encoder") or {}
need(got_enc == enc_sha and enc.get("checkpoint_sha256") == got_enc, 4,
     f"encoder ckpt sha256={got_enc} provenance.encoder.checkpoint_sha256={enc.get('checkpoint_sha256')}")
need(mm.manifest_sha256 == manifest_id and prov.get("manifest_sha256") == manifest_id, 4,
     f"table manifest_sha256={prov.get('manifest_sha256')}")
need(prov.get("input_manifest_sha256") is not None and prov.get("input_manifest_sha256") == im_sha, 4,
     f"provenance.input_manifest_sha256={prov.get('input_manifest_sha256')} sha(input_manifest)={im_sha}")
need(prov.get("raw_dir") is not None and prov.get("raw_dir") == im.get("raw_dir"), 4, f"provenance.raw_dir={prov.get('raw_dir')}")
need("encoder_kind" not in prov, 4, f"provenance 无 encoder_kind（非池化表）={'encoder_kind' not in prov}")
got_norm = sha(norm_path)
need(got_norm == norm_sha, 4, f"norm_stats sha256={got_norm}")
ok = not fails
print(f"INPUT_SHA={'PASS' if ok else 'FAIL'} motion_bin={str(got_bin)[:8]} store_meta={str(got_meta)[:8]} "
      f"norm={str(got_norm)[:8]} encoder={str(got_enc)[:8]} rows={mm.num_rows}"
      + ("" if ok else f" failed={len(fails)}"), flush=True)
sys.exit(0 if ok else 1)
PY

  # 第 1 道 DATA_LINEAGE（G0 环 5，P-LINEAGE）：四环血缘脚本，FAIL 即 exit 4
  uv run --no-sync python scripts/training/check_lineage_v7_4task.py \
    --encoder-ckpt "$ENCODER_CKPT" --expected-ckpt-sha256 "$ENCODER_SHA" \
    --vla-input-manifest "$DS/meta/input_manifest.json" \
    --merge-done "$LINEAGE_RECORDS/MERGE_DONE.json" --episode-maps "$LINEAGE_RECORDS" \
    || { echo "DATA_LINEAGE 未通过，停工（P-LINEAGE 不得豁免）"; exit 4; }

  # 第 3 道 CONFIG_CROSSCHECK：新 YAML 与原 run YAML 解析后逐键比，差集必须恰为两键
  git show "$HC_ORIG_REF" > "$CHECK_TMP/orig.yaml"
  uv run --no-sync python - "$CHECK_TMP/orig.yaml" "src/mme_vla_suite/models/config/robomme/$HC" "$HC_SHA" <<'PY'
import hashlib, pathlib, sys
import yaml
orig_p, new_p, want_sha = sys.argv[1:]
got_sha = hashlib.sha256(pathlib.Path(new_p).read_bytes()).hexdigest()
def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        key = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out
a, b = flat(yaml.safe_load(open(orig_p))), flat(yaml.safe_load(open(new_p)))
diff = sorted(k for k in set(a) | set(b) if a.get(k, "<缺>") != b.get(k, "<缺>"))
ok = (got_sha == want_sha and diff == ["integration_type", "memory_token_dim"]
      and a.get("integration_type") == "modulation" and b.get("integration_type") == "context"
      and a.get("memory_token_dim") == 1024 and b.get("memory_token_dim") == 2048)
print(f"CONFIG_CROSSCHECK={'PASS' if ok else 'FAIL'} yaml_diff={','.join(diff)} yaml_sha256={got_sha[:8]}"
      + ("" if ok else f" want_sha256={want_sha[:8]} new_values={[b.get(k) for k in diff]}"), flush=True)
sys.exit(0 if ok else 1)
PY

  # 第 4 道 WAN_ENV：评估用运动编码环境与离线表 provenance 一致（训练不用此环境，放这里是起跑前一次拦住）
  uv run --no-sync python - "$WAN_VENV" "$MOTION/meta/store_meta.json" "scripts/dataset/wan/wan_motion_infer.py" <<'PY'
import glob, hashlib, json, pathlib, sys
venv, meta_p, infer_p = sys.argv[1:]
def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() if pathlib.Path(p).is_file() else None
prov = json.loads(pathlib.Path(meta_p).read_text())["provenance"]
vae, enc = prov.get("vae") or {}, prov.get("encoder") or {}
want_src, want_mod, want_commit = vae.get("encoder_src_sha256"), vae.get("module_sha256"), prov.get("mj_repo_commit")
fails = []
srcs = glob.glob(f"{venv}/lib/python3*/site-packages/motion_jepa/models/wan_latent_encoder.py")
src_sha = sha(srcs[0]) if len(srcs) == 1 else None
if not (src_sha and src_sha == want_src and enc.get("encoder_src_sha256") == want_src):
    fails.append(f"encoder_src={src_sha} want={want_src} files={len(srcs)}")
urls = glob.glob(f"{venv}/lib/python3*/site-packages/motion_jepa*.dist-info/direct_url.json")
commit = None
if len(urls) == 1:
    commit = (json.loads(pathlib.Path(urls[0]).read_text()).get("vcs_info") or {}).get("commit_id")
if not (commit and want_commit and commit == want_commit):
    fails.append(f"motion_jepa commit={commit} want={want_commit} dist_info={len(urls)}")
mod_sha = sha(infer_p)
if not (mod_sha and mod_sha == want_mod and enc.get("module_sha256") == want_mod):
    fails.append(f"module={mod_sha} want={want_mod}")
ok = not fails
print(f"WAN_ENV={'PASS' if ok else 'FAIL'} motion_jepa={str(commit)[:7]} encoder_src={str(src_sha)[:8]} "
      f"module={str(mod_sha)[:8]}" + ("" if ok else " " + " ; ".join(fails)), flush=True)
sys.exit(0 if ok else 1)
PY

  # 第 5 道 PREFLIGHT：起跑位置、HEAD/clean、norm、YAML sha、motion 表（帧库同级 motion/）
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/preflight_train_launch.py \
    --repo "$MAIN" --train-head "$TRAIN_HEAD" \
    --history-config "$HC" --history-config-sha256 "$HC_SHA" \
    --assets-dir "$ASSETS_DIR" --asset-id robomme \
    --norm-stats-sha256 "$NORM_SHA" \
    --dataset-path "$DS/framesamp-8x8" \
    --run-root "$RUN_ROOT" \
    --motion-store "$MOTION" --motion-store-meta-sha256 "$MOTION_META_SHA" \
    --motion-layout "$MOTION_LAYOUT" --motion-rows "$MOTION_ROWS" \
    -- "${TRAIN_ARGS[@]}"
}

common_env() {
  cd "$MAIN"
  source "$MAIN/scripts/training/paths.sh"
  test "$(git rev-parse HEAD)" = "$TRAIN_HEAD" || { echo "拒绝：$MAIN HEAD=$(git rev-parse HEAD) != TRAIN_HEAD=$TRAIN_HEAD"; exit 1; }
  test -z "$(git status --porcelain)" || { echo "拒绝：$MAIN 工作区不干净"; exit 1; }
  unset PYTHONPATH
  export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  unset JAX_PLATFORMS MMEVLA_MOTION_STORE WANDB_MODE
  export MMEVLA_FRAMESAMP_SOURCE="$DS/source"
  export MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"
  CHECK_TMP="$(mktemp -d)"
  trap 'rm -rf "$CHECK_TMP"' EXIT
}

if [ "$CHECK_ONLY" = 1 ]; then
  (
    set -euo pipefail
    common_env
    printf 'CHECK_ONLY_START RUN=%s TRAIN_HEAD=%s START_UTC=%s\n' "$RUN" "$TRAIN_HEAD" "$(date -u +%FT%TZ)"
    gates
    echo "CHECK_ONLY=PASS run=$RUN head=$TRAIN_HEAD"
  )
  rc=$?
  [ "$rc" = 0 ] || echo "CHECK_ONLY=FAIL rc=$rc"
  exit "$rc"
fi

test ! -e "$LOG" || { printf '拒绝覆盖本轮日志 %s\n' "$LOG"; exit 2; }
mkdir -p "$(dirname "$LOG")" "$REC"

body() (
  set -euo pipefail
  common_env
  set -a; . "$MAIN/v1-store/secrets/wandb.env"; set +a   # WANDB_API_KEY / WANDB_ENTITY，不进日志

  export CUDA_CACHE_PATH="$V1_STORE/cache/cuda"
  export WANDB_DATA_DIR="$V1_STORE/cache/wandb-data" XDG_DATA_HOME="$V1_STORE/cache/xdg-data"
  export TRAIN_TIMING_STEPS=300
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN"
  export TRAIN_RECORD_DIR="$REC"

  test "$(nvidia-smi --id=0,1,2,3,4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)" = "0"

  printf 'RUN=%s\nTRAIN_HEAD=%s\nREPO=%s\nV1_STORE=%s\nSTEPS=%s\nSMOKE=%s\nSTART_UTC=%s\n' \
    "$RUN" "$TRAIN_HEAD" "$MAIN" "$V1_STORE" "$STEPS" "$SMOKE" "$(date -u +%FT%TZ)"
  printf 'TRAIN_ARGS='; printf ' %q' "${TRAIN_ARGS[@]}"; printf '\n'

  # ── 起跑前第 1–5 道（与 --check-only 同一函数）──
  gates

  # 全程 15 s GPU 采样：stdout 与 stderr 都重定向到文件，绝不留在 tee 管道上（否则 tee 不退出）
  nvidia-smi --id=0,1,2,3,4,5,6,7 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 15 \
    > "$REC/gpu_util_15s_full.csv" 2> "$REC/gpu_util_15s_full.err" &
  sampler_pid=$!
  echo "GPU_SAMPLER_PID=$sampler_pid"
  trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; rm -rf "$CHECK_TMP"' EXIT

  uv run --no-sync python scripts/training/train.py "${TRAIN_ARGS[@]}"
  kill "$sampler_pid" 2>/dev/null || true

  # ── 训练后：checkpoint 齐全 + context 快照断言 + 真实加载最后一份（第 10 道 / 第 6 道）──
  JAX_PLATFORMS=cpu uv run --no-sync python - "$RUN_ROOT" "$SMOKE" "$ENCODER_SHA" $EXPECT_CKPTS <<'PY'
import json, pathlib, sys
import numpy as np
import jax, jax.numpy as jnp
import openpi.models.model as _model
from mme_vla_suite.policies.policy_config import _load_resolved_snapshot, _params_have_motion

run_root, smoke, enc_sha, expect = pathlib.Path(sys.argv[1]), sys.argv[2] == "1", sys.argv[3], sys.argv[4:]
dirs = sorted((d.name for d in run_root.iterdir() if d.is_dir() and d.name.isdigit()), key=int)
assert dirs == expect, f"checkpoint 集合 {dirs} != 期望 {expect}"
for d in expect:
    assert (run_root / d / "params").is_dir(), f"缺 {d}/params"
cfg, enabled = _load_resolved_snapshot(run_root)
assert cfg.integration_type == "context", f"快照 integration_type={cfg.integration_type}"
assert int(cfg.memory_token_dim) == 2048, f"快照 memory_token_dim={cfg.memory_token_dim}"
assert enabled, "快照 motion 未开启"
assert cfg.motion.get("encoder_kind", None) is None, f"快照出现 encoder_kind={cfg.motion.get('encoder_kind')}"
prov = json.loads((run_root / "motion_provenance.json").read_text())
assert "encoder_kind" not in prov, "motion_provenance 出现 encoder_kind（池化分支）"
assert (prov.get("encoder") or {}).get("checkpoint_sha256") == enc_sha, \
    f"motion_provenance encoder sha={(prov.get('encoder') or {}).get('checkpoint_sha256')}"
assert prov.get("vae"), "motion_provenance 缺 vae"
last = run_root / expect[-1]
params = _model.restore_params(last / "params", dtype=jnp.bfloat16)
assert _params_have_motion(params), "最后一份 checkpoint 不含 motion 参数"
leaves = jax.tree_util.tree_leaves(params)
assert leaves and all(bool(np.isfinite(np.asarray(x, dtype=np.float32)).all()) for x in leaves), "参数含非有限值"
print(f"CKPT_LOAD=PASS step={expect[-1]} integration=context motion=1 leaves={len(leaves)} "
      f"resolved_sha256={prov['resolved_sha256']}", flush=True)
if smoke:
    print(f"SMOKE20=PASS steps=20 save_load=PASS", flush=True)
else:
    print(f"TRAIN_FINAL_RECORD=PASS step={expect[-1]} ckpts={len(dirs)}", flush=True)
PY
)

body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
