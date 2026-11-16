#!/usr/bin/env bash
# 4task 原训练库 + Wan raw latent 零参数池化 768 维 motion（modulation）50k 正式训练 / 20 步冒烟 runner。
# 依据：根目录 1108-4task-modul-wantp768-50k-plan.html §3.3、§5 G0/G6/G7、第二部分 1.2 runner 行与 runbook D1/D3。
# 覆盖项照抄上次 v2-1600ep-m8x8-modul-motion-b128-80k 的 mv2-prod-runner.sh：TrainConfig mme_vla_suite_b128_80k
# 全部超参不变，只覆盖 --num_train_steps=50001（最后一份 checkpoint 为 50000），history YAML 换成 …-8x8-wantp768.yaml。
#
# 用法：
#   bash scripts/training/run_4task_wantp768_50k.sh <TRAIN_HEAD 40 位 sha>                         # 正式 50001 步
#   bash scripts/training/run_4task_wantp768_50k.sh <TRAIN_HEAD> --smoke20 --run-name w768-smoke20  # G6 冒烟 20 步
#   bash scripts/training/run_4task_wantp768_50k.sh <任意> [--smoke20] [--run-name N] --dry-run     # 只打印 TRAIN_ARGS
#
# 冒烟与正式的 TRAIN_ARGS 只差 --exp-name 与 --num_train_steps 两处；history YAML 不含步数，两者的
# history_config.resolved.yaml（train.py::init_history_config 写出）逐字节相同。
# 起跑前内置 G0 复核（runbook C2 各环 + 第 4 环新表 provenance + VLA 侧一环），FAIL 即退出；
# 训练结束核 checkpoint 齐全并真实加载最后一份，末行写 G7（正式）或 G6（冒烟）判定行。
set -o pipefail                     # 外层记录正常返回和报错的退出码；外部强杀除外。

usage() { printf '用法: %s <TRAIN_HEAD> [--smoke20] [--run-name NAME] [--dry-run]\n' "$0" >&2; exit 2; }
TRAIN_HEAD="${1:-}"; [ -n "$TRAIN_HEAD" ] || usage; shift
SMOKE=0; DRY=0; RUN_NAME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke20) SMOKE=1 ;;
    --dry-run) DRY=1 ;;
    --run-name) [ $# -ge 2 ] || usage; RUN_NAME="$2"; shift ;;
    *) usage ;;
  esac
  shift
done

MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
FORMAL_RUN=v2-1600ep-m8x8-modul-wantp768-b128-50k
SMOKE_RUN=w768-smoke20
if [ "$SMOKE" = 1 ]; then
  RUN="${RUN_NAME:-$SMOKE_RUN}"; STEPS=20; EXPECT_CKPTS="19"
  [ "$RUN" = "$SMOKE_RUN" ] || { printf '冒烟 run 名固定为 %s，拒绝 %s\n' "$SMOKE_RUN" "$RUN" >&2; exit 2; }
else
  RUN="${RUN_NAME:-$FORMAL_RUN}"; STEPS=50001
  EXPECT_CKPTS="5000 10000 15000 20000 25000 30000 35000 40000 45000 50000"
fi
CONFIG=mme_vla_suite_b128_80k
HC=perceptual-framesamp-modul-8frame-8x8-wantp768.yaml
HC_SHA=e049ed6243c7a226796a1e0f1ed346eff2cd586e3348c87e4a91b644ba2af5eb   # 定稿 YAML 文件字节 sha256
NORM_SHA=856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173
MANIFEST_ID=4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918   # load_manifest 重算的 canonical identity（不是文件字节摘要）
LATENTS_META_SHA=c92643178e71dc0bbe7a406770d5ca84c800bb902785118bb3dd62c42efa6eff   # wan-latents/metadata.json，旧表 provenance.latents_metadata_sha256
V1_STORE="$MAIN/v1-store"
DS="$V1_STORE/datasets/4task-v2-1600ep-604f16da"
MOTION="$DS/motion-wan-pool768"
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

test ! -e "$LOG" || { printf '拒绝覆盖本轮日志 %s\n' "$LOG"; exit 2; }
mkdir -p "$(dirname "$LOG")" "$REC"

body() (
  set -euo pipefail
  cd "$MAIN"
  source "$MAIN/scripts/training/paths.sh"
  set -a; . "$MAIN/v1-store/secrets/wandb.env"; set +a   # WANDB_API_KEY / WANDB_ENTITY，不进日志

  test "$(git rev-parse HEAD)" = "$TRAIN_HEAD"
  test -z "$(git status --porcelain)"
  export CUDA_CACHE_PATH="$V1_STORE/cache/cuda"
  export WANDB_DATA_DIR="$V1_STORE/cache/wandb-data" XDG_DATA_HOME="$V1_STORE/cache/xdg-data"
  export TRAIN_TIMING_STEPS=300

  unset PYTHONPATH
  export UV_CACHE_DIR="$V1_STORE/cache/uv" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  unset JAX_PLATFORMS MMEVLA_MOTION_STORE WANDB_MODE
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  export MMEVLA_JAX_CACHE_DIR="$V1_STORE/cache/jax/$RUN"
  export TRAIN_RECORD_DIR="$REC"
  export MMEVLA_FRAMESAMP_SOURCE="$DS/source"
  export MMEVLA_FRAMESAMP_MANIFEST="$DS/meta/episode_manifest.json"

  test "$(nvidia-smi --id=0,1,2,3,4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)" = "0"

  printf 'RUN=%s\nTRAIN_HEAD=%s\nREPO=%s\nV1_STORE=%s\nSTEPS=%s\nSMOKE=%s\nSTART_UTC=%s\n' \
    "$RUN" "$TRAIN_HEAD" "$MAIN" "$V1_STORE" "$STEPS" "$SMOKE" "$(date -u +%FT%TZ)"
  printf 'TRAIN_ARGS='; printf ' %q' "${TRAIN_ARGS[@]}"; printf '\n'

  # ── 内置 G0（计划 §5 G0、runbook C2；P-LINEAGE：任一环缺证据或不符即 DATA_LINEAGE=FAIL 并退出）──
  JAX_PLATFORMS=cpu uv run --no-sync python - "$DS" "$MOTION" "$MANIFEST_ID" "$LATENTS_META_SHA" <<'PY'
import hashlib, json, os, pathlib, sys
from mme_vla_suite.datastore.manifest import load_manifest
from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.shared import wan_pool768 as wp

ds, motion, manifest_id, latents_meta_sha = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
fails = []
def need(ok, ring, msg):
    print(f"G0_RING={'PASS' if ok else 'FAIL'} ring={ring} {msg}", flush=True)
    if not ok:
        fails.append(f"{ring}:{msg}")
def sha(p):
    return ms.sha256_file(p) if pathlib.Path(p).is_file() else None
def walk_values(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from walk_values(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk_values(v)
    else:
        yield o
def pin_of(files):
    return hashlib.sha256("\n".join(f"{k}:{files[k]['sha256']}" for k in sorted(files)).encode()).hexdigest()

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
# 环 3：wan-latents 元数据（G1a/G1b 已核 tar 对象与解包成员摘要）
lat_meta_path = ds / "wan-latents/metadata.json"
lat = json.loads(lat_meta_path.read_text()) if lat_meta_path.is_file() else {}
need(sha(lat_meta_path) == latents_meta_sha, 3, f"wan-latents/metadata.json sha256={sha(lat_meta_path)}")
need(lat.get("manifest_sha256") == manifest_id, 3, f"latent manifest_sha256={lat.get('manifest_sha256')}")
need(lat.get("raw_dir") is not None and lat.get("raw_dir") == im.get("raw_dir"), 3, f"latent raw_dir={lat.get('raw_dir')}")
# 环 4：新表 provenance（池化身份 + latent 根文件摘要 + manifest identity + 输入清单摘要）
need(motion.is_dir() and not motion.is_symlink() and motion.resolve().parent == ds.resolve(), 4, f"motion 根={motion}")
mm = ms.MotionMeta.load(motion)
prov = mm.provenance
need(mm.raw.get("layout") == ms.LAYOUT and mm.num_rows == 71316, 4, f"layout={mm.raw.get('layout')} rows={mm.num_rows}")
need("MMEVLA_MOTION_STORE" not in os.environ, 4, "MMEVLA_MOTION_STORE 未设置")
need(prov.get("encoder_kind") == wp.POOL768_ENCODER_KIND and prov.get("readout") == wp.POOL768_READOUT, 4,
     f"encoder_kind={prov.get('encoder_kind')} readout={prov.get('readout')}")
need("encoder_ckpt" in prov and prov["encoder_ckpt"] is None and prov.get("parameter_count") == 0, 4,
     f"encoder_ckpt={prov.get('encoder_ckpt', '<缺>')} parameter_count={prov.get('parameter_count')}")
need(prov.get("pool_source_sha256") == wp.pool_source_sha256(), 4, f"pool_source_sha256={prov.get('pool_source_sha256')}")
need(mm.manifest_sha256 == manifest_id and prov.get("manifest_sha256") == manifest_id, 4, f"table manifest_sha256={prov.get('manifest_sha256')}")
need(latents_meta_sha in set(walk_values(prov.get("latent_root"))), 4, "latent_root 记录 wan-latents/metadata.json 摘要")
vae = prov.get("vae") or {}
need(bool(vae.get("vae_state_sha256")), 4, f"vae_state_sha256={vae.get('vae_state_sha256')}")
# 环 5（P-LINEAGE）：两侧 pin 各取独立记录，逐文件比较 H5 sha256 集合，任一侧取不到即 FAIL。
# VLA 侧：VLA 库 meta/input_manifest.json 的 4 个 H5 sha256（规则原文定义的 vla_input_pin）；并核 framesamp store_meta 与
# source/meta/provenance.json 的 manifest identity、raw_dir / source_dataset_root 一致，source_provenance_sha256 = 文件摘要。
sm_path = ds / "framesamp-8x8/meta/store_meta.json"
sm = json.loads(sm_path.read_text()) if sm_path.is_file() else {}
sp_path = ds / "source/meta/provenance.json"
sp = json.loads(sp_path.read_text()) if sp_path.is_file() else {}
need(sm.get("manifest_sha256") == manifest_id and sp.get("manifest_sha256") == manifest_id, 5,
     f"store_meta/source manifest_sha256={sm.get('manifest_sha256')}/{sp.get('manifest_sha256')}")
need(sm.get("source_provenance_sha256") is not None and sm.get("source_provenance_sha256") == sha(sp_path), 5,
     "store_meta.source_provenance_sha256 = sha256(source/meta/provenance.json)")
need(sp.get("raw_dir") is not None and sp.get("raw_dir") == im.get("raw_dir"), 5, f"source raw_dir={sp.get('raw_dir')}")
need(sm.get("source_dataset_root") is not None and pathlib.Path(sm["source_dataset_root"]).resolve() == (ds / "source").resolve(), 5,
     f"store_meta.source_dataset_root={sm.get('source_dataset_root')}")
def h5_set(manifest):
    return {k: v.get("sha256") for k, v in (manifest.get("files") or {}).items()}
vla_files = h5_set(im)
# encoder 侧：从 encoder 侧独立记录取 input_manifest 文件摘要，按摘要内容寻址找清单，再取 H5 sha256 集合
records = []
rec = prov.get("input_manifest_sha256")
if rec:
    records.append(("motion-wan-pool768.provenance", rec))
old_meta = ds / "motion/meta/store_meta.json"
if old_meta.is_file():
    records.append(("motion.provenance", json.loads(old_meta.read_text()).get("provenance", {}).get("input_manifest_sha256")))
if "input_manifest_sha256" in lat:
    records.append(("wan-latents.metadata", lat.get("input_manifest_sha256")))
need(bool(rec), 5, f"新表 provenance.input_manifest_sha256={rec}")
need(all(v == rec for _, v in records), 5, f"encoder 侧各记录一致: {records}")
found = None
for cand in (ds / "meta/input_manifest.json", ds / "meta.bucket/input_manifest.json"):
    if rec and sha(cand) == rec:
        found = cand
        break
need(found is not None, 5, f"按摘要 {rec} 寻址到的清单={found}")
enc_files = h5_set(json.loads(found.read_text())) if found is not None else {}
diff_files = sorted(k for k in set(vla_files) | set(enc_files) if vla_files.get(k) is None or vla_files.get(k) != enc_files.get(k))
need(len(vla_files) == 4 and len(enc_files) == 4 and not diff_files, 5,
     f"逐文件比较 vla={len(vla_files)} enc={len(enc_files)} 不符={diff_files}")
vla_pin = pin_of(im.get("files") or {}) if len(vla_files) == 4 else None
enc_pin = pin_of(json.loads(found.read_text())["files"]) if found is not None and len(enc_files) == 4 else None
ok = not fails and vla_pin is not None and enc_pin is not None and vla_pin == enc_pin
print(f"DATA_LINEAGE={'PASS' if ok else 'FAIL'} encoder=wanvae-{str(vae.get('vae_state_sha256'))[:8]}+pool768 "
      f"encoder_source_pin={enc_pin} vla_input_pin={vla_pin} files={len(vla_files)} mismatches={len(fails)} "
      f"binding=input_manifest_sha256:{str(rec)[:8]} enc_record={'+'.join(n for n, _ in records) or 'none'}", flush=True)
print(f"MOTION_STORE_META=PASS sha256={sha(motion / 'meta/store_meta.json')} layout={mm.raw.get('layout')} rows={mm.num_rows}", flush=True)
sys.exit(0 if ok else 1)
PY

  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/preflight_train_launch.py \
    --repo "$MAIN" --train-head "$TRAIN_HEAD" \
    --history-config "$HC" --history-config-sha256 "$HC_SHA" \
    --assets-dir "$ASSETS_DIR" --asset-id robomme \
    --norm-stats-sha256 "$NORM_SHA" \
    --dataset-path "$DS/framesamp-8x8" \
    --run-root "$RUN_ROOT" \
    -- "${TRAIN_ARGS[@]}"
  # motion 表检查不走 preflight 的 --motion-*（它要求表在帧库同级的 motion/）；同等检查已在上面 G0 环 4 完成

  # 全程 15 s GPU 采样：stdout 与 stderr 都重定向到文件，绝不留在 tee 管道上（否则 tee 不退出）
  nvidia-smi --id=0,1,2,3,4,5,6,7 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 15 \
    > "$REC/gpu_util_15s_full.csv" 2> "$REC/gpu_util_15s_full.err" &
  sampler_pid=$!
  echo "GPU_SAMPLER_PID=$sampler_pid"
  trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT

  uv run --no-sync python scripts/training/train.py "${TRAIN_ARGS[@]}"
  kill "$sampler_pid" 2>/dev/null || true

  # ── 训练后：checkpoint 齐全 + 真实加载最后一份（G7 / G6）──
  JAX_PLATFORMS=cpu uv run --no-sync python - "$RUN_ROOT" "$SMOKE" $EXPECT_CKPTS <<'PY'
import json, pathlib, sys
import numpy as np
import jax, jax.numpy as jnp
import openpi.models.model as _model
from mme_vla_suite.policies.policy_config import _load_resolved_snapshot, _params_have_motion
from mme_vla_suite.shared import wan_pool768 as wp

run_root, smoke, expect = pathlib.Path(sys.argv[1]), sys.argv[2] == "1", sys.argv[3:]
dirs = sorted((d.name for d in run_root.iterdir() if d.is_dir() and d.name.isdigit()), key=int)
assert dirs == expect, f"checkpoint 集合 {dirs} != 期望 {expect}"
for d in expect:
    assert (run_root / d / "params").is_dir(), f"缺 {d}/params"
cfg, enabled = _load_resolved_snapshot(run_root)
assert enabled and cfg.motion.encoder_kind == wp.POOL768_ENCODER_KIND and cfg.motion.readout == wp.POOL768_READOUT, "快照 motion kind/readout 不符"
prov = json.loads((run_root / "motion_provenance.json").read_text())
assert prov.get("encoder_kind") == wp.POOL768_ENCODER_KIND and prov.get("readout") == wp.POOL768_READOUT, "motion_provenance kind/readout 不符"
assert prov.get("pool_source_sha256") == wp.pool_source_sha256() and prov.get("vae"), "motion_provenance 池化源码或 vae 缺失"
last = run_root / expect[-1]
params = _model.restore_params(last / "params", dtype=jnp.bfloat16)
assert _params_have_motion(params), "最后一份 checkpoint 不含 motion 参数"
leaves = jax.tree_util.tree_leaves(params)
assert leaves and all(bool(np.isfinite(np.asarray(x, dtype=np.float32)).all()) for x in leaves), "参数含非有限值"
print(f"CKPT_LOAD=PASS step={expect[-1]} leaves={len(leaves)} resolved_sha256={prov['resolved_sha256']}", flush=True)
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
