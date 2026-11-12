#!/usr/bin/env bash
# 4task 原训练库 + Wan raw latent 不池化（每窗 9 token × 16384 维、窗口步长 32）modulation 50k runner。
# 依据：根目录 1112-4task-modul-wanraw9-s32-50k-plan.html 第二部分「1.2 逐文件清单」runner 行、闸门总表 G0/G6/G7、
# runbook RB-B6 / RB-D1 / RB-D2 / RB-E2。由 run_4task_wantp768_50k.sh（v11）复制，只换 history YAML（…-8x8-wanraw9s32.yaml）、
# HC_SHA、motion 表路径（motion-wan-raw9-s32）、run 名；TrainConfig mme_vla_suite_b128_80k 全部超参不变，
# TRAIN_ARGS 仍只覆盖 --num_train_steps（不加 --fsdp-devices，沿用条目的 8）。
#
# 用法：
#   bash run_4task_wanraw9s32_50k.sh <TRAIN_HEAD 40 位 sha>                              # 正式 50001 步（最后一份 ckpt 50000）
#   bash run_4task_wanraw9s32_50k.sh <TRAIN_HEAD> --smoke20 [--run-name r9-smoke20]      # 闸 G6 冒烟 20 步（ckpt step 19）
#   bash run_4task_wanraw9s32_50k.sh <TRAIN_HEAD> --bench300 [--run-name r9-bench300]    # 闸 G6 测速 300 步（统计去掉前 20 步）
#   bash run_4task_wanraw9s32_50k.sh <任意> --g0-only                                    # RB-B6：只跑 DATA_LINEAGE 各环（只读）
#   bash run_4task_wanraw9s32_50k.sh <任意> [任一模式] [--run-name N] --dry-run          # 只打印 TRAIN_ARGS 与诊断开关
#
# 各模式的 TRAIN_ARGS 只差 --exp-name 与 --num_train_steps；history YAML 不含步数。
# --g0-only：不要求 GPU 空闲、不做 HEAD / clean 检查、不写任何文件；新表目录不存在时只核 VLA 侧与 encoder 源侧
#   （第 1、2、3、5 环，encoder 侧取新表将沿用的 motion-wan-pool768 provenance），末行 DATA_LINEAGE_PRE=PASS table=pending …
#   （不输出 DATA_LINEAGE=PASS，由冒烟与正式 runner 内置 G0 在表存在后再核）；表存在时跑含新表一环的完整检查，
#   末行 DATA_LINEAGE=PASS …。任一环 FAIL 即 DATA_LINEAGE=FAIL 并非零退出（P-LINEAGE，不得 WAIVE）。
#   单测专用：R9_G0_MOTION=<路径> 只在 --g0-only 下生效，用来替换新表路径（表缺失夹具）；其他模式设了即拒绝。
# 冒烟 / 测速 / 正式：起跑前内置完整 G0（含新表一环），FAIL 即退出；训练中后台 follower 把
#   TRAIN_RECORD_DIR 下 step_timing.jsonl 逐步打印成 STEP_TIME 行、metrics.jsonl 每个 log_interval 打印成 DIAG 行（只记录不判停）；
#   训练后：正式 / 冒烟核 checkpoint 齐全并真实加载最后一份（CKPT_LOAD=PASS、TRAIN_FINAL_RECORD=PASS 或 SMOKE20=PASS），
#   测速按去掉前 20 步的窗口算 BENCH300=PASS steps=300 s_per_step=… dl_wait_frac=… gpu_util_mean=…。
#   注意：train.py 在最后一步无条件存 ckpt，测速 run 因此会在 train-runs/<config>/<测速 run 名>/299 留一份，
#   runner 不加载、不核验它，由主会话按显式路径删除（S1 拥有 train.py，本脚本不改训练入口）。
set -o pipefail                     # 外层记录正常返回和报错的退出码；外部强杀除外。

usage() { printf '用法: %s <TRAIN_HEAD> [--g0-only|--smoke20|--bench300] [--run-name NAME] [--dry-run]\n' "$0" >&2; exit 2; }
TRAIN_HEAD="${1:-}"; [ -n "$TRAIN_HEAD" ] || usage; shift
MODE=formal; DRY=0; RUN_NAME=""
set_mode() { [ "$MODE" = formal ] || { printf '模式互斥：已是 %s，又给了 %s\n' "$MODE" "$1" >&2; exit 2; }; MODE="$1"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --g0-only) set_mode g0 ;;
    --smoke20) set_mode smoke ;;
    --bench300) set_mode bench ;;
    --dry-run) DRY=1 ;;
    --run-name) [ $# -ge 2 ] || usage; RUN_NAME="$2"; shift ;;
    *) usage ;;
  esac
  shift
done
if [ -n "${R9_G0_MOTION:-}" ] && [ "$MODE" != g0 ]; then
  printf 'R9_G0_MOTION 只允许与 --g0-only 同用（单测夹具），拒绝在 %s 模式下替换表路径\n' "$MODE" >&2; exit 2
fi

MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
FORMAL_RUN=v2-1600ep-m8x8-modul-wanraw9s32-b128-50k
SMOKE_RUN=r9-smoke20
BENCH_RUN=r9-bench300
BENCH_SKIP=20                       # 测速统计去掉前 20 步（编译）
case "$MODE" in
  smoke)
    RUN="${RUN_NAME:-$SMOKE_RUN}"; STEPS=20; EXPECT_CKPTS="19"
    [[ "$RUN" =~ ^r9-smoke20(-l[123])?$ ]] || { printf '冒烟 run 名只允许 r9-smoke20 / r9-smoke20-l<k>，拒绝 %s\n' "$RUN" >&2; exit 2; } ;;
  bench)
    RUN="${RUN_NAME:-$BENCH_RUN}"; STEPS=300; EXPECT_CKPTS=""
    [[ "$RUN" =~ ^r9-(bench300(-l[123])?|memprobe)$ ]] || { printf '测速 run 名只允许 r9-bench300 / r9-bench300-l<k> / r9-memprobe，拒绝 %s\n' "$RUN" >&2; exit 2; } ;;
  *)
    RUN="${RUN_NAME:-$FORMAL_RUN}"; STEPS=50001
    EXPECT_CKPTS="5000 10000 15000 20000 25000 30000 35000 40000 45000 50000" ;;
esac
[[ "$RUN" =~ ^[A-Za-z0-9._-]+$ ]] || { printf 'run 名非法：%s\n' "$RUN" >&2; exit 2; }
CONFIG=mme_vla_suite_b128_80k
HC=perceptual-framesamp-modul-8frame-8x8-wanraw9s32.yaml
# 定稿 YAML 文件字节 sha256。占位：YAML 由 S2 并行写，S2 合并后 rebase 时按真实文件算好写死（主会话复核）；
# 占位期间 preflight 的 --history-config-sha256 必然 FAIL，冒烟 / 测速 / 正式起不来（--g0-only 不用它）。
HC_SHA=PENDING_S2_YAML_SHA256
NORM_SHA=856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173
MANIFEST_ID=4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918   # load_manifest 重算的 canonical identity（不是文件字节摘要）
LATENTS_META_SHA=c92643178e71dc0bbe7a406770d5ca84c800bb902785118bb3dd62c42efa6eff   # wan-latents/metadata.json，新表 provenance.latent_root.metadata_sha256
RAW9_ROWS=36484
V1_STORE="$MAIN/v1-store"
DS="$V1_STORE/datasets/4task-v2-1600ep-604f16da"
MOTION="$DS/motion-wan-raw9-s32"
[ "$MODE" = g0 ] && [ -n "${R9_G0_MOTION:-}" ] && MOTION="$R9_G0_MOTION"
ASSETS_DIR="$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
RUN_ROOT="$V1_STORE/train-runs/$CONFIG/$RUN"
LOG="$V1_STORE/logs/$RUN.runner.log"     # 不用 <run>.log：tmux 外壳 tmx 已把整段输出 tee 到 logs/<tmux 名>.log，冒烟同名会撞
REC="$V1_STORE/bench/$RUN"
TIMING_STEPS=300                         # 诊断开关：step_timing 前 300 步（冒烟 20 步时取 min）；TRAIN_RECORD_DIR=$REC 打开 metrics 记录

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

if [ -e "$MOTION" ] || [ -L "$MOTION" ]; then TABLE_STATE=present; else TABLE_STATE=pending; fi

if [ "$DRY" = 1 ]; then
  printf 'MODE=%s RUN=%s STEPS=%s EXPECT_CKPTS=%s\n' "$MODE" "$RUN" "$STEPS" "$EXPECT_CKPTS"
  printf 'HC=%s HC_SHA=%s MOTION=%s TABLE=%s\n' "$HC" "$HC_SHA" "$MOTION" "$TABLE_STATE"
  printf 'TRAIN_ARGS='; printf ' %q' "${TRAIN_ARGS[@]}"; printf '\n'
  if [ "$MODE" = g0 ]; then
    printf 'G0_ONLY=1 checks=%s\n' "$([ "$TABLE_STATE" = present ] && echo full || echo pre)"
  else
    printf 'DIAG_SWITCH TRAIN_TIMING_STEPS=%s TRAIN_RECORD_DIR=%s follower=STEP_TIME+DIAG gpu_sample_s=15 LOG=%s\n' \
      "$TIMING_STEPS" "$REC" "$LOG"
  fi
  exit 0
fi

# ── 闸 G0（计划闸门总表 G0、R3；P-LINEAGE：任一环缺证据或不符即 DATA_LINEAGE=FAIL 并非零退出）──
# $1 = full（含新表一环）| pre（新表尚不存在：只核 VLA 侧与 encoder 源侧）
run_g0() (
  cd "$MAIN"
  env -u MMEVLA_MOTION_STORE JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 "$MAIN/.venv/bin/python" - \
    "$DS" "$MOTION" "$MANIFEST_ID" "$LATENTS_META_SHA" "$RAW9_ROWS" "$1" <<'PY'
import hashlib, json, os, pathlib, sys
from mme_vla_suite.datastore.manifest import load_manifest
from mme_vla_suite.datastore import motion_store as ms

ds, motion = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
manifest_id, latents_meta_sha, raw9_rows, mode = sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
assert mode in ("full", "pre"), mode
fails = []
def need(ok, ring, msg):
    print(f"G0_RING={'PASS' if ok else 'FAIL'} ring={ring} {msg}", flush=True)
    if not ok:
        fails.append(f"{ring}:{msg}")
def sha(p):
    return ms.sha256_file(p) if pathlib.Path(p).is_file() else None
def jload(p):
    p = pathlib.Path(p)
    return json.loads(p.read_text()) if p.is_file() else {}
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
im = jload(local_im) or {"files": {}, "raw_dir": None}
files = im.get("files", {})
need(len(files) == 4 and all(len(str(v.get("sha256", ""))) == 64 for v in files.values()), 1, f"H5 文件数={len(files)}")
# 环 2：canonical manifest identity 用 load_manifest 重算（不是文件字节摘要）
man = load_manifest(ds / "meta/episode_manifest.json")
need(man["sha256"] == manifest_id, 2, f"manifest_identity={man['sha256']}")
need(man.get("raw_dir") == im.get("raw_dir"), 2, f"manifest.raw_dir 与 input_manifest.raw_dir 相同={man.get('raw_dir') == im.get('raw_dir')}")
# 环 3：wan-latents 元数据（新表的源字节就是它的 stride-16 偶数窗）
lat_meta_path = ds / "wan-latents/metadata.json"
lat = jload(lat_meta_path)
need(sha(lat_meta_path) == latents_meta_sha, 3, f"wan-latents/metadata.json sha256={sha(lat_meta_path)}")
need(lat.get("manifest_sha256") == manifest_id, 3, f"latent manifest_sha256={lat.get('manifest_sha256')}")
need(lat.get("raw_dir") is not None and lat.get("raw_dir") == im.get("raw_dir"), 3, f"latent raw_dir={lat.get('raw_dir')}")
# 环 4：新表 provenance（raw9 身份 + 源码摘要 + latent 根摘要 + manifest identity + 行数）；pre 模式表尚不存在，记 pending
pool_prov = (jload(ds / "motion-wan-pool768/meta/store_meta.json").get("provenance") or {})
if mode == "full":
    from mme_vla_suite.shared import wan_raw9 as wr
    need(motion.is_dir() and not motion.is_symlink() and motion.resolve().parent == ds.resolve(), 4, f"motion 根={motion}")
    need("MMEVLA_MOTION_STORE" not in os.environ, 4, "MMEVLA_MOTION_STORE 未设置")
    try:
        mm = ms.MotionMeta.load(motion)
        prov, layout, rows = mm.provenance, mm.raw.get("layout"), mm.num_rows
        table_manifest = mm.manifest_sha256
    except Exception as e:  # 表读不出来也必须落成 FAIL 判定行，不能只留 Traceback
        need(False, 4, f"MotionMeta.load 失败: {type(e).__name__}: {e}")
        prov, layout, rows, table_manifest = {}, None, None, None
    need(layout == ms.LAYOUT_RAW9 and rows == raw9_rows, 4, f"layout={layout} rows={rows}")
    need(prov.get("encoder_kind") == wr.RAW9_ENCODER_KIND and prov.get("readout") == wr.RAW9_READOUT, 4,
         f"encoder_kind={prov.get('encoder_kind')} readout={prov.get('readout')}")
    need("encoder_ckpt" in prov and prov["encoder_ckpt"] is None and prov.get("parameter_count") == 0, 4,
         f"encoder_ckpt={prov.get('encoder_ckpt', '<缺>')} parameter_count={prov.get('parameter_count')}")
    need(prov.get("raw9_source_sha256") == wr.source_sha256(), 4, f"raw9_source_sha256={prov.get('raw9_source_sha256')}")
    need(table_manifest == manifest_id and prov.get("manifest_sha256") == manifest_id, 4, f"table manifest_sha256={prov.get('manifest_sha256')}")
    need(latents_meta_sha in set(walk_values(prov.get("latent_root"))), 4, "latent_root 记录 wan-latents/metadata.json 摘要")
    vae = prov.get("vae") or {}
else:
    print(f"G0_RING=PENDING ring=4 table=pending motion={motion}（新表未导出，由冒烟 / 正式 runner 内置 G0 再核）", flush=True)
    prov = {}
    vae = pool_prov.get("vae") or {}   # 新表沿用 motion-wan-pool768 的 vae 块（计划 1.1 第 3 行）
need(bool(vae.get("vae_state_sha256")), 4, f"vae_state_sha256={vae.get('vae_state_sha256')}")
# 环 5（P-LINEAGE）：两侧 pin 各取独立记录，逐文件比较 H5 sha256 集合，任一侧取不到即 FAIL。
# VLA 侧：VLA 库 meta/input_manifest.json 的 4 个 H5 sha256；并核 framesamp store_meta 与 source/meta/provenance.json
# 的 manifest identity、raw_dir / source_dataset_root 一致，source_provenance_sha256 = 文件摘要。
sm = jload(ds / "framesamp-8x8/meta/store_meta.json")
sp_path = ds / "source/meta/provenance.json"
sp = jload(sp_path)
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
# encoder 侧：从 encoder 侧独立记录取 input_manifest 文件摘要，按摘要内容寻址找清单，再取 H5 sha256 集合。
# full：主记录是新表 provenance（motion-wan-raw9-s32.provenance）；pre：新表未出，主记录取它将沿用的 motion-wan-pool768.provenance。
records = []
if mode == "full":
    rec = prov.get("input_manifest_sha256")
    rec_name = "motion-wan-raw9-s32.provenance"
else:
    rec = pool_prov.get("input_manifest_sha256")
    rec_name = "motion-wan-pool768.provenance"
if rec:
    records.append((rec_name, rec))
if mode == "full" and pool_prov:
    records.append(("motion-wan-pool768.provenance", pool_prov.get("input_manifest_sha256")))
old_meta = ds / "motion/meta/store_meta.json"
if old_meta.is_file():
    records.append(("motion.provenance", jload(old_meta).get("provenance", {}).get("input_manifest_sha256")))
if "input_manifest_sha256" in lat:
    records.append(("wan-latents.metadata", lat.get("input_manifest_sha256")))
need(bool(rec), 5, f"{rec_name}.input_manifest_sha256={rec}")
need(all(v == rec for _, v in records), 5, f"encoder 侧各记录一致: {records}")
found = None
for cand in (ds / "meta/input_manifest.json", ds / "meta.bucket/input_manifest.json"):
    if rec and sha(cand) == rec:
        found = cand
        break
need(found is not None, 5, f"按摘要 {rec} 寻址到的清单={found}")
enc_files = h5_set(jload(found)) if found is not None else {}
diff_files = sorted(k for k in set(vla_files) | set(enc_files) if vla_files.get(k) is None or vla_files.get(k) != enc_files.get(k))
need(len(vla_files) == 4 and len(enc_files) == 4 and not diff_files, 5,
     f"逐文件比较 vla={len(vla_files)} enc={len(enc_files)} 不符={diff_files}")
vla_pin = pin_of(im.get("files") or {}) if len(vla_files) == 4 else None
enc_pin = pin_of(jload(found)["files"]) if found is not None and len(enc_files) == 4 else None
ok = not fails and vla_pin is not None and enc_pin is not None and vla_pin == enc_pin
tail = (f"encoder=wanvae-{str(vae.get('vae_state_sha256'))[:8]}+raw9s32 "
        f"encoder_source_pin={enc_pin} vla_input_pin={vla_pin} files={len(vla_files)} mismatches={len(fails)} "
        f"binding=input_manifest_sha256:{str(rec)[:8]} enc_record={'+'.join(n for n, _ in records) or 'none'}")
if mode == "pre":
    # 新表未出时不得输出 DATA_LINEAGE=PASS（计划 RB-B6 补充口径）；FAIL 照常输出 DATA_LINEAGE=FAIL
    print(f"{'DATA_LINEAGE_PRE=PASS' if ok else 'DATA_LINEAGE=FAIL'} table=pending {tail}", flush=True)
else:
    print(f"DATA_LINEAGE={'PASS' if ok else 'FAIL'} {tail}", flush=True)
    if ok:
        print(f"MOTION_STORE_META=PASS sha256={sha(motion / 'meta/store_meta.json')} layout={layout} rows={rows}", flush=True)
sys.exit(0 if ok else 1)
PY
)

if [ "$MODE" = g0 ]; then
  # 只读：不写日志（tmx 外壳负责 tee）、不建目录、不查 GPU / HEAD
  printf 'G0_ONLY_START table=%s motion=%s TRAIN_HEAD=%s START_UTC=%s\n' "$TABLE_STATE" "$MOTION" "$TRAIN_HEAD" "$(date -u +%FT%TZ)"
  if [ "$TABLE_STATE" = present ]; then run_g0 full; else run_g0 pre; fi
  rc=$?
  printf 'EXIT_CODE=%s\n' "$rc"
  exit "$rc"
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
  export TRAIN_TIMING_STEPS="$TIMING_STEPS"

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

  printf 'RUN=%s\nMODE=%s\nTRAIN_HEAD=%s\nREPO=%s\nV1_STORE=%s\nSTEPS=%s\nSTART_UTC=%s\n' \
    "$RUN" "$MODE" "$TRAIN_HEAD" "$MAIN" "$V1_STORE" "$STEPS" "$(date -u +%FT%TZ)"
  printf 'TRAIN_ARGS='; printf ' %q' "${TRAIN_ARGS[@]}"; printf '\n'
  printf 'DIAG_SWITCH TRAIN_TIMING_STEPS=%s TRAIN_RECORD_DIR=%s follower=STEP_TIME+DIAG gpu_sample_s=15\n' "$TIMING_STEPS" "$REC"

  # 内置完整 G0（含新表一环），FAIL 即退出（set -e）
  run_g0 full

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

  # 记录项 follower：step_timing.jsonl → STEP_TIME 行；metrics.jsonl → DIAG 行（只记录、不判停）。
  # 读到停止标记后把剩余行读完再退出；它的输出走本 body 的 stdout（进 tee 日志）。
  stop_flag="$REC/.follower-stop"
  rm -f "$stop_flag"
  JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 "$MAIN/.venv/bin/python" -u - "$REC" "$stop_flag" <<'PY' &
import json, pathlib, sys, time
rec, stop = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
DIAG_KEYS = ("loss", "grad_norm", "motion_enc_kernel_norm", "motion_enc_grad_norm", "frame_enc_grad_norm", "mem_enc_norm")
pos = {"step_timing.jsonl": 0, "metrics.jsonl": 0}
dl_wait = {}
last = None
def lines(name):
    p = rec / name
    if not p.is_file():
        return []
    with p.open("rb") as f:
        f.seek(pos[name])
        data = f.read()
    cut = data.rfind(b"\n") + 1      # 只消费完整行
    pos[name] += cut
    return [json.loads(x) for x in data[:cut].decode().splitlines() if x.strip()]
def fmt(v):
    return "NA" if v is None else f"{v:.6g}"
def drain():
    global last
    for r in lines("step_timing.jsonl"):
        ph = r.get("phases_s", {})
        s = r["host_step_s"] - ph.get("checkpoint", 0.0)     # 存 ckpt 的那一步扣掉 checkpoint 段
        dl_wait[r["step"]] = ph.get("data_next")
        print(f"STEP_TIME step={r['step']} s={s:.4f} dl_wait={fmt(ph.get('data_next'))} completed={int(bool(r.get('completed')))}", flush=True)
    for r in lines("metrics.jsonl"):
        step, wall = r.get("step"), r.get("wall_time")
        sps = None
        if last is not None and step is not None and step > last[0]:
            sps = (wall - last[1]) / (step - last[0])
        if step is not None:
            last = (step, wall)
        vals = " ".join(f"{k}={fmt(r[k]['dec']) if k in r else 'NA'}" for k in DIAG_KEYS)
        print(f"DIAG step={step} {vals} s_per_step={fmt(sps)} dl_wait={fmt(dl_wait.get(step))}", flush=True)
while True:
    done = stop.exists()
    drain()
    if done:
        break
    time.sleep(2)
PY
  follower_pid=$!
  echo "FOLLOWER_PID=$follower_pid"
  trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; touch "$stop_flag"; wait "$follower_pid" 2>/dev/null || true' EXIT

  uv run --no-sync python scripts/training/train.py "${TRAIN_ARGS[@]}"
  kill "$sampler_pid" 2>/dev/null || true
  touch "$stop_flag"; wait "$follower_pid"

  if [ "$MODE" = bench ]; then
    # ── 测速判定（闸 G6）：去掉前 BENCH_SKIP 步；s = host_step_s − checkpoint 段；dl_wait = data_next 段 ──
    JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 "$MAIN/.venv/bin/python" - "$REC" "$STEPS" "$BENCH_SKIP" <<'PY'
import csv, datetime, json, pathlib, sys
rec, steps, skip = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rows = sorted((json.loads(x) for x in (rec / "step_timing.jsonl").read_text().splitlines() if x.strip()), key=lambda r: r["step"])
complete = len(rows) == steps and all(r.get("completed") for r in rows)
win = rows[skip:]
s = [r["host_step_s"] - r.get("phases_s", {}).get("checkpoint", 0.0) for r in win]
dl = [r.get("phases_s", {}).get("data_next", 0.0) for r in win]
s_per_step = sum(s) / len(s) if s else float("nan")
dl_frac = sum(dl) / sum(s) if s and sum(s) > 0 else float("nan")
t0, t1 = (win[0]["wall_start"], win[-1]["wall_end"]) if win else (0.0, 0.0)
util = []
with (rec / "gpu_util_15s_full.csv").open() as f:
    for row in csv.reader(f):
        try:
            ts = datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()   # nvidia-smi 本地时间
            u = float(row[2])
        except (ValueError, IndexError):
            continue
        if t0 <= ts <= t1:
            util.append(u)
gmean = sum(util) / len(util) if util else float("nan")
zero = sum(1 for u in util if u == 0.0) / len(util) if util else float("nan")
ok = complete and bool(win) and bool(util)
print(f"BENCH300={'PASS' if ok else 'FAIL'} steps={len(rows)} s_per_step={s_per_step:.4f} dl_wait_frac={dl_frac:.4f} "
      f"gpu_util_mean={gmean:.2f} gpu_util_zero_frac={zero:.4f} gpu_samples={len(util)} window_steps={len(win)} skip={skip}", flush=True)
sys.exit(0 if ok else 1)
PY
    exit 0
  fi

  # ── 训练后：checkpoint 齐全 + 真实加载最后一份（G7 / G6）──
  JAX_PLATFORMS=cpu uv run --no-sync python - "$RUN_ROOT" "$MODE" $EXPECT_CKPTS <<'PY'
import json, pathlib, sys
import numpy as np
import jax, jax.numpy as jnp
import openpi.models.model as _model
from mme_vla_suite.policies.policy_config import _load_resolved_snapshot, _params_have_motion
from mme_vla_suite.shared import wan_raw9 as wr

run_root, smoke, expect = pathlib.Path(sys.argv[1]), sys.argv[2] == "smoke", sys.argv[3:]
dirs = sorted((d.name for d in run_root.iterdir() if d.is_dir() and d.name.isdigit()), key=int)
assert dirs == expect, f"checkpoint 集合 {dirs} != 期望 {expect}"
for d in expect:
    assert (run_root / d / "params").is_dir(), f"缺 {d}/params"
cfg, enabled = _load_resolved_snapshot(run_root)
assert enabled and cfg.motion.encoder_kind == wr.RAW9_ENCODER_KIND and cfg.motion.readout == wr.RAW9_READOUT, "快照 motion kind/readout 不符"
assert int(cfg.motion.stride) == wr.RAW9_STRIDE and int(cfg.motion.tokens_per_window) == wr.RAW9_TOKENS_PER_WINDOW, "快照 stride/tokens_per_window 不符"
prov = json.loads((run_root / "motion_provenance.json").read_text())
assert prov.get("encoder_kind") == wr.RAW9_ENCODER_KIND and prov.get("readout") == wr.RAW9_READOUT, "motion_provenance kind/readout 不符"
assert prov.get("raw9_source_sha256") == wr.source_sha256() and prov.get("vae"), "motion_provenance raw9 源码摘要或 vae 缺失"
assert "encoder" in prov and prov["encoder"] is None, "motion_provenance encoder 应为 null"
last = run_root / expect[-1]
params = _model.restore_params(last / "params", dtype=jnp.bfloat16)
assert _params_have_motion(params), "最后一份 checkpoint 不含 motion 参数"
leaves = jax.tree_util.tree_leaves(params)
assert leaves and all(bool(np.isfinite(np.asarray(x, dtype=np.float32)).all()) for x in leaves), "参数含非有限值"
print(f"CKPT_LOAD=PASS step={expect[-1]} leaves={len(leaves)} resolved_sha256={prov.get('resolved_sha256')}", flush=True)
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
