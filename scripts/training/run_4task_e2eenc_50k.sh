#!/usr/bin/env bash
# 4task 原训练库 + 端到端训练的 Wan latent encoder（E2E-Enc：步长 16 bf16 raw 表、策略内 1 层 encoder）modulation 50k runner。
# 依据：根目录 1117-e2e-enc-train-plan.html 第二部分「一」runner 行、「二」S5 行、三节 G0/G4/G6/G7、四节 RB-C1/RB-D1/RB-D2/RB-E2。
# 由 run_4task_wanraw9s32_50k.sh（c0826be）复制：换 history YAML（…-8x8-e2eenc.yaml）、HC_SHA、motion 表路径（motion-wan-raw9-s16）、
# run 名与冒烟 / 探测 run 名白名单；G0 常量与收尾断言改 e2e 口径；新增 --shmprobe / --ckptprobe。
# TrainConfig mme_vla_suite_b128_80k 全部超参不变（R2）：TRAIN_ARGS 的超参类开关只有 --exp-name / --model.history-config /
# --num_train_steps 三项，路径类参数与 Wan-9 相同；只有 --ckptprobe 追加 --save-interval 50 --keep-period 50（R8 (i) 吞吐/留存档）。
#
# 用法：
#   bash run_4task_e2eenc_50k.sh <TRAIN_HEAD 40 位 sha>                                     # 正式 50001 步（最后一份 ckpt 50000）
#   bash run_4task_e2eenc_50k.sh <TRAIN_HEAD> --smoke20 [--run-name e2e-smoke20]             # 闸 G4 冒烟 20 步（ckpt step 19，8 卡）
#   bash run_4task_e2eenc_50k.sh <TRAIN_HEAD> --bench300 --run-name e2e-bench-aNN            # 闸 G6 ① 测速 300 步
#   bash run_4task_e2eenc_50k.sh <TRAIN_HEAD> --shmprobe --run-name e2e-shmprobe-aNN         # 闸 G6 ② 2000 步 + shmmon.sh
#   bash run_4task_e2eenc_50k.sh <TRAIN_HEAD> --ckptprobe --run-name e2e-ckptprobe-aNN       # 闸 G6 ③ 80 步，存 step 50 并读回
#   bash run_4task_e2eenc_50k.sh <任意> --g0-only                                            # 只跑 DATA_LINEAGE 各环（只读）
#   bash run_4task_e2eenc_50k.sh <任意> [任一模式] [--run-name N] --dry-run                 # 只打印 TRAIN_ARGS 与诊断开关
#
# 探测 run 名：三种探测模式必须显式给 --run-name，只接受 ^e2e-(bench|shmprobe|ckptprobe)-a[0-9]{2}$ 且模式与名字一致；
#   attempt id（aNN）同时绑定 runner 日志、checkpoint、记录目录、shmmon 日志与 G6 证据；本轮日志或 run 目录已存在即拒（重跑递增 NN）。
# 路径配置摘要 PATHCFG = "prefetch=<E2E_PREFETCH_FACTOR 或 2> compact=off remat=off"（紧凑计算 / remat 尚未实现，恒为 off；
#   采用 R8 (ii) 档时须先改这里），PATHCFG_SHA = 其 sha256 前 16 位；G6 证据只认与当前摘要相同的尝试。
# E2E_PREFETCH_FACTOR：默认不设（torch 默认 2）；RB-D2 退路第一档设 1。train.py 打印 DATALOADER_PREFETCH_FACTOR= 供 G6 读回比对。
# 单测专用（只在 --dry-run 生效，其他情形设了即拒）：E2E_DRY_STATE_DIR=<目录> 把「本轮日志 / run 目录已存在」检查改到该目录下
#   （logs/<run>.runner.log、train-runs/<run>），TRAIN_ARGS 照旧用真实路径。
# 单测专用（只在 --g0-only 生效）：E2E_G0_MOTION=<表路径>、E2E_G0_YAML=<history YAML 路径> 替换 G0 读的表与 YAML。
# 诊断开关 TRAIN_TIMING_PROFILE=0（不开 jax.profiler，见 Wan-9 runner 文件头的 abort 记录）；计时行照写，TRAIN_TIMING_STEPS=300。
# 所有训练模式保留 15 s nvidia-smi 采样（utilization.gpu、memory.used）与 15 s `free -g` 采样，写在记录目录。
# GPU_PEAK 证据：e2e_g6_judge.py 读日志里的 `DEVICE_PEAK device=<i> peak_bytes_in_use=<n>` 行（由训练进程打印；缺则 GPU_PEAK=FAIL）。
set -o pipefail                     # 外层记录正常返回和报错的退出码；外部强杀除外。

usage() { printf '用法: %s <TRAIN_HEAD> [--g0-only|--smoke20|--bench300|--shmprobe|--ckptprobe] [--run-name NAME] [--dry-run]\n' "$0" >&2; exit 2; }
TRAIN_HEAD="${1:-}"; [ -n "$TRAIN_HEAD" ] || usage; shift
MODE=formal; DRY=0; RUN_NAME=""
set_mode() { [ "$MODE" = formal ] || { printf '模式互斥：已是 %s，又给了 %s\n' "$MODE" "$1" >&2; exit 2; }; MODE="$1"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --g0-only) set_mode g0 ;;
    --smoke20) set_mode smoke ;;
    --bench300) set_mode bench ;;
    --shmprobe) set_mode shmprobe ;;
    --ckptprobe) set_mode ckptprobe ;;
    --dry-run) DRY=1 ;;
    --run-name) [ $# -ge 2 ] || usage; RUN_NAME="$2"; shift ;;
    *) usage ;;
  esac
  shift
done
for v in E2E_G0_MOTION E2E_G0_YAML; do
  if [ -n "${!v:-}" ] && [ "$MODE" != g0 ]; then
    printf '%s 只允许与 --g0-only 同用（单测夹具），拒绝在 %s 模式下替换\n' "$v" "$MODE" >&2; exit 2
  fi
done
if [ -n "${E2E_DRY_STATE_DIR:-}" ] && [ "$DRY" != 1 ]; then
  printf 'E2E_DRY_STATE_DIR 只允许与 --dry-run 同用（单测夹具），拒绝\n' >&2; exit 2
fi
if [ -n "${E2E_PREFETCH_FACTOR:-}" ] && ! [[ "$E2E_PREFETCH_FACTOR" =~ ^[1-9][0-9]*$ ]]; then
  printf 'E2E_PREFETCH_FACTOR 必须是正整数，收到 %s\n' "$E2E_PREFETCH_FACTOR" >&2; exit 2
fi

MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
FORMAL_RUN=v2-1600ep-m8x8-modul-e2eenc-s16-b128-50k
SMOKE_RUN=e2e-smoke20               # 与 check_checkpoint.py RUNS 的冒烟条目同名（step 19）
BENCH_SKIP=20                       # 测速统计去掉前 20 步（编译）
PROBE_RE='^e2e-(bench|shmprobe|ckptprobe)-a[0-9]{2}$'
EXTRA_ARGS=()
ATTEMPT=a00
case "$MODE" in
  smoke)
    RUN="${RUN_NAME:-$SMOKE_RUN}"; STEPS=20; EXPECT_CKPTS="19"
    [[ "$RUN" =~ ^e2e-smoke20(-a[0-9]{2})?$ ]] || { printf '冒烟 run 名只允许 e2e-smoke20 / e2e-smoke20-aNN，拒绝 %s\n' "$RUN" >&2; exit 2; }
    [[ "$RUN" =~ -(a[0-9]{2})$ ]] && ATTEMPT="${BASH_REMATCH[1]}" ;;
  bench|shmprobe|ckptprobe)
    RUN="$RUN_NAME"
    [ -n "$RUN" ] || { printf '%s 模式必须显式给 --run-name e2e-%s-aNN（尝试编号）\n' "$MODE" "$MODE" >&2; exit 2; }
    [[ "$RUN" =~ $PROBE_RE ]] || { printf '探测 run 名必须匹配 %s，拒绝 %s\n' "$PROBE_RE" "$RUN" >&2; exit 2; }
    [[ "$RUN" =~ ^e2e-${MODE}-(a[0-9]{2})$ ]] || { printf '%s 模式只接受 e2e-%s-aNN，拒绝 %s\n' "$MODE" "$MODE" "$RUN" >&2; exit 2; }
    ATTEMPT="${BASH_REMATCH[1]}"
    case "$MODE" in
      bench) STEPS=300; EXPECT_CKPTS="" ;;
      shmprobe) STEPS=2000; EXPECT_CKPTS="" ;;
      ckptprobe) STEPS=80; EXPECT_CKPTS="50 79"; EXTRA_ARGS=(--save-interval 50 --keep-period 50) ;;
    esac ;;
  *)
    RUN="${RUN_NAME:-$FORMAL_RUN}"; STEPS=50001
    [ "$RUN" = "$FORMAL_RUN" ] || { printf '正式 / --g0-only 模式 run 名只允许 %s，拒绝 %s\n' "$FORMAL_RUN" "$RUN" >&2; exit 2; }
    EXPECT_CKPTS="5000 10000 15000 20000 25000 30000 35000 40000 45000 50000" ;;
esac
[[ "$RUN" =~ ^[A-Za-z0-9._-]+$ ]] || { printf 'run 名非法：%s\n' "$RUN" >&2; exit 2; }
CONFIG=mme_vla_suite_b128_80k
HC=perceptual-framesamp-modul-8frame-8x8-e2eenc.yaml
# 定稿 YAML（src/mme_vla_suite/models/config/robomme/<HC>）的文件字节 sha256：按 S3 worktree 当前版本写入（S3 尚未合入），
# S3 合入后主会话须以合入版本重算（单测 test_e2e_runner_chain.py 在 YAML 存在时与现文件比对）。
HC_SHA=63f27aa6f3ea1fbdd3ae81d4f62598a9483673c554da3916a66f89e6d54350ae
NORM_SHA=856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173
MANIFEST_ID=4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918   # load_manifest 重算的 canonical identity
LATENTS_META_SHA=c92643178e71dc0bbe7a406770d5ca84c800bb902785118bb3dd62c42efa6eff   # wan-latents/metadata.json
E2E_ROWS=71316                      # s16 bf16 新表行数（G2 EXPORT_RAW16=PASS rows=71316）
WAN9_LEAVES=65                      # Wan-9 冒烟 ckpt 参数叶数（r9-smoke20 日志 CKPT_LOAD=PASS step=19 leaves=65）
V1_STORE="$MAIN/v1-store"
DS="$V1_STORE/datasets/4task-v2-1600ep-604f16da"
MOTION="$DS/motion-wan-raw9-s16"
HC_FILE="$MAIN/src/mme_vla_suite/models/config/robomme/$HC"
if [ "$MODE" = g0 ]; then
  [ -n "${E2E_G0_MOTION:-}" ] && MOTION="$E2E_G0_MOTION"
  [ -n "${E2E_G0_YAML:-}" ] && HC_FILE="$E2E_G0_YAML"
fi
ASSETS_DIR="$V1_STORE/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
RUN_ROOT="$V1_STORE/train-runs/$CONFIG/$RUN"
LOG="$V1_STORE/logs/$RUN.runner.log"     # 不用 <run>.log：tmux 外壳已把整段输出 tee 到 logs/<tmux 名>.log
REC="$V1_STORE/bench/$RUN"
G6_DIR="$V1_STORE/bench/g6"              # G6 分阶段证据 g6-<mode>-aNN.json
SHMMON_LOG="$V1_STORE/logs/e2e-shmmon-$ATTEMPT.log"
CHK_LOG="$LOG"; CHK_ROOT="$RUN_ROOT"
if [ -n "${E2E_DRY_STATE_DIR:-}" ]; then
  CHK_LOG="$E2E_DRY_STATE_DIR/logs/$RUN.runner.log"; CHK_ROOT="$E2E_DRY_STATE_DIR/train-runs/$RUN"
fi
TIMING_STEPS=300
TIMING_PROFILE=0
PREFETCH_EXPECT="${E2E_PREFETCH_FACTOR:-2}"
PATHCFG="prefetch=$PREFETCH_EXPECT compact=off remat=off"
PATHCFG_SHA="$(printf '%s' "$PATHCFG" | sha256sum | cut -c1-16)"

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
  "${EXTRA_ARGS[@]}"
)

if [ -e "$MOTION" ] || [ -L "$MOTION" ]; then TABLE_STATE=present; else TABLE_STATE=pending; fi

# 本轮日志 / run 目录已存在即拒（探测与冒烟必须换 attempt id；正式 run 名不得复用）
if [ "$MODE" != g0 ]; then
  test ! -e "$CHK_LOG" || { printf '拒绝覆盖本轮日志 %s（重跑请递增尝试编号）\n' "$CHK_LOG" >&2; exit 2; }
  test ! -e "$CHK_ROOT" || { printf '拒绝复用已存在的 run 目录 %s（重跑请递增尝试编号）\n' "$CHK_ROOT" >&2; exit 2; }
fi

if [ "$DRY" = 1 ]; then
  printf 'MODE=%s RUN=%s STEPS=%s EXPECT_CKPTS=%s ATTEMPT=%s\n' "$MODE" "$RUN" "$STEPS" "$EXPECT_CKPTS" "$ATTEMPT"
  printf 'HC=%s HC_SHA=%s MOTION=%s TABLE=%s\n' "$HC" "$HC_SHA" "$MOTION" "$TABLE_STATE"
  printf 'PATHCFG=%s PATHCFG_SHA=%s\n' "$PATHCFG" "$PATHCFG_SHA"
  printf 'TRAIN_ARGS='; printf ' %q' "${TRAIN_ARGS[@]}"; printf '\n'
  if [ "$MODE" = g0 ]; then
    printf 'G0_ONLY=1 checks=%s\n' "$([ "$TABLE_STATE" = present ] && echo full || echo pre)"
  else
    printf 'DIAG_SWITCH TRAIN_TIMING_STEPS=%s TRAIN_TIMING_PROFILE=%s TRAIN_RECORD_DIR=%s follower=STEP_TIME+DIAG gpu_sample_s=15 LOG=%s shmmon=%s g6_evidence=%s\n' \
      "$TIMING_STEPS" "$TIMING_PROFILE" "$REC" "$LOG" "$([ "$MODE" = shmprobe ] && echo "$SHMMON_LOG" || echo off)" \
      "$(case "$MODE" in bench|shmprobe|ckptprobe) echo "$G6_DIR/g6-$MODE-$ATTEMPT.json" ;; *) echo off ;; esac)"
  fi
  exit 0
fi

# ── 闸 G0（三节 G0、R6；P-LINEAGE：任一环缺证据或不符即 DATA_LINEAGE=FAIL 并非零退出）──
# $1 = full（含新表一环）| pre（新表尚不存在：只核 VLA 侧与 encoder 源侧）
run_g0() (
  cd "$MAIN"
  env -u MMEVLA_MOTION_STORE JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 "$MAIN/.venv/bin/python" - \
    "$DS" "$MOTION" "$MANIFEST_ID" "$LATENTS_META_SHA" "$E2E_ROWS" "$1" "$HC_FILE" "$MAIN" <<'PY'
import hashlib, json, os, pathlib, sys
from mme_vla_suite.datastore.manifest import load_manifest
from mme_vla_suite.datastore import motion_store as ms

ds, motion = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
manifest_id, latents_meta_sha, e2e_rows, mode = sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
hc_file, main_root = pathlib.Path(sys.argv[7]), pathlib.Path(sys.argv[8])
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
# 环 3：wan-latents 元数据（新表的源字节就是它的全部步长 16 窗）
lat_meta_path = ds / "wan-latents/metadata.json"
lat = jload(lat_meta_path)
need(sha(lat_meta_path) == latents_meta_sha, 3, f"wan-latents/metadata.json sha256={sha(lat_meta_path)}")
need(lat.get("manifest_sha256") == manifest_id, 3, f"latent manifest_sha256={lat.get('manifest_sha256')}")
need(lat.get("raw_dir") is not None and lat.get("raw_dir") == im.get("raw_dir"), 3, f"latent raw_dir={lat.get('raw_dir')}")
# 环 4：新表 provenance（check_e2e_table_provenance：table_kind/table_op/parameter_count 0/raw9 源码摘要/布局）
#   + latent 根摘要 + manifest identity + 行数 + YAML wan_e2e ↔ 表 wan_raw_bf16_s16_v1 配对
pool_prov = (jload(ds / "motion-wan-pool768/meta/store_meta.json").get("provenance") or {})
if mode == "full":
    need(motion.is_dir() and not motion.is_symlink() and motion.resolve().parent == ds.resolve(), 4, f"motion 根={motion}")
    need("MMEVLA_MOTION_STORE" not in os.environ, 4, "MMEVLA_MOTION_STORE 未设置")
    try:
        mm = ms.MotionMeta.load(motion)
        prov, layout, rows = mm.provenance, mm.raw.get("layout"), mm.num_rows
        table_manifest = mm.manifest_sha256
    except Exception as e:  # 表读不出来也必须落成 FAIL 判定行，不能只留 Traceback
        need(False, 4, f"MotionMeta.load 失败: {type(e).__name__}: {e}")
        prov, layout, rows, table_manifest = {}, None, None, None
    need(layout == ms.LAYOUT_RAW9BF16 and rows == e2e_rows, 4, f"layout={layout} rows={rows}")
    try:
        ms.check_e2e_table_provenance(prov, ctx=str(motion), layout=layout)
        need(True, 4, f"check_e2e_table_provenance table_kind={prov.get('table_kind')} table_op={prov.get('table_op')} parameter_count={prov.get('parameter_count')}")
    except Exception as e:
        need(False, 4, f"check_e2e_table_provenance 失败: {e}")
    need(table_manifest == manifest_id and prov.get("manifest_sha256") == manifest_id, 4, f"table manifest_sha256={prov.get('manifest_sha256')}")
    need(latents_meta_sha in set(walk_values(prov.get("latent_root"))), 4, "latent_root 记录 wan-latents/metadata.json 摘要")
    vae = prov.get("vae") or {}
    # YAML ↔ 表配对（S3 契约：encoder_kind wan_e2e、readout wan_e2e_enc768_s16_v1、stride 16、store_path 指向本表）
    ycfg = {}
    if hc_file.is_file():
        from omegaconf import OmegaConf
        ycfg = OmegaConf.to_container(OmegaConf.load(str(hc_file)), resolve=True).get("motion") or {}
    want_store = os.path.relpath(str(motion), str(main_root))
    pair = (ycfg.get("encoder_kind") == "wan_e2e" and ycfg.get("readout") == "wan_e2e_enc768_s16_v1"
            and int(ycfg.get("stride", -1)) == 16 and int(ycfg.get("latents_per_window", -1)) == 9
            and int(ycfg.get("raw_dim", -1)) == 16384 and ycfg.get("store_path") == want_store
            and prov.get("table_kind") == ms.E2E_TABLE_KIND)
    need(pair, 4, f"YAML↔表配对 yaml={hc_file.name} present={hc_file.is_file()} encoder_kind={ycfg.get('encoder_kind')} "
                  f"readout={ycfg.get('readout')} stride={ycfg.get('stride')} store_path={ycfg.get('store_path')} "
                  f"table_kind={prov.get('table_kind')}")
else:
    print(f"G0_RING=PENDING ring=4 table=pending motion={motion}（新表未导出，由冒烟 / 正式 runner 内置 G0 再核）", flush=True)
    prov = {}
    vae = pool_prov.get("vae") or {}
need(bool(vae.get("vae_state_sha256")), 4, f"vae_state_sha256={vae.get('vae_state_sha256')}")
# 环 5（P-LINEAGE）：两侧 pin 各取独立记录，逐文件比较 H5 sha256 集合，任一侧取不到即 FAIL。
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
records = []
if mode == "full":
    rec = prov.get("input_manifest_sha256")
    rec_name = "motion-wan-raw9-s16.provenance"
else:
    rec = pool_prov.get("input_manifest_sha256")
    rec_name = "motion-wan-pool768.provenance"
if rec:
    records.append((rec_name, rec))
if mode == "full" and pool_prov:
    records.append(("motion-wan-pool768.provenance", pool_prov.get("input_manifest_sha256")))
s32_meta = ds / "motion-wan-raw9-s32/meta/store_meta.json"
if s32_meta.is_file():
    records.append(("motion-wan-raw9-s32.provenance", jload(s32_meta).get("provenance", {}).get("input_manifest_sha256")))
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
tail = (f"encoder=wanvae-{str(vae.get('vae_state_sha256'))[:8]}+e2e-s16 "
        f"encoder_source_pin={enc_pin} vla_input_pin={vla_pin} files={len(vla_files)} mismatches={len(fails)} "
        f"binding=input_manifest_sha256:{str(rec)[:8]} enc_record={'+'.join(n for n, _ in records) or 'none'}")
if mode == "pre":
    print(f"{'DATA_LINEAGE_PRE=PASS' if ok else 'DATA_LINEAGE=FAIL'} table=pending {tail}", flush=True)
else:
    print(f"DATA_LINEAGE={'PASS' if ok else 'FAIL'} {tail}", flush=True)
    if ok:
        print(f"MOTION_STORE_META=PASS sha256={sha(motion / 'meta/store_meta.json')} layout={layout} rows={rows}", flush=True)
sys.exit(0 if ok else 1)
PY
)

if [ "$MODE" = g0 ]; then
  # 只读：不写日志（tmux 外壳负责 tee）、不建目录、不查 GPU / HEAD
  printf 'G0_ONLY_START table=%s motion=%s yaml=%s TRAIN_HEAD=%s START_UTC=%s\n' "$TABLE_STATE" "$MOTION" "$HC_FILE" "$TRAIN_HEAD" "$(date -u +%FT%TZ)"
  if [ "$TABLE_STATE" = present ]; then run_g0 full; else run_g0 pre; fi
  rc=$?
  printf 'EXIT_CODE=%s\n' "$rc"
  exit "$rc"
fi

if [ "$MODE" = shmprobe ]; then
  test ! -e "$SHMMON_LOG" || { printf '拒绝覆盖 shmmon 日志 %s（重跑请递增尝试编号）\n' "$SHMMON_LOG"; exit 2; }
fi
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
  export TRAIN_TIMING_STEPS="$TIMING_STEPS" TRAIN_TIMING_PROFILE="$TIMING_PROFILE"

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

  printf 'RUN=%s\nMODE=%s\nATTEMPT=%s\nTRAIN_HEAD=%s\nREPO=%s\nV1_STORE=%s\nSTEPS=%s\nSTART_UTC=%s\n' \
    "$RUN" "$MODE" "$ATTEMPT" "$TRAIN_HEAD" "$MAIN" "$V1_STORE" "$STEPS" "$(date -u +%FT%TZ)"
  printf 'PATHCFG=%s PATHCFG_SHA=%s E2E_PREFETCH_FACTOR=%s\n' "$PATHCFG" "$PATHCFG_SHA" "${E2E_PREFETCH_FACTOR:-unset}"
  printf 'TRAIN_ARGS='; printf ' %q' "${TRAIN_ARGS[@]}"; printf '\n'
  printf 'DIAG_SWITCH TRAIN_TIMING_STEPS=%s TRAIN_TIMING_PROFILE=%s TRAIN_RECORD_DIR=%s follower=STEP_TIME+DIAG gpu_sample_s=15\n' \
    "$TIMING_STEPS" "$TIMING_PROFILE" "$REC"

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

  # 全程 15 s GPU 采样（所有模式含正式训练）：stdout 与 stderr 都重定向到文件，绝不留在 tee 管道上
  nvidia-smi --id=0,1,2,3,4,5,6,7 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 15 \
    > "$REC/gpu_util_15s_full.csv" 2> "$REC/gpu_util_15s_full.err" &
  sampler_pid=$!
  echo "GPU_SAMPLER_PID=$sampler_pid"
  # 15 s 主机内存采样（RB-D1 记 free -g available）
  ( while :; do printf '%s %s\n' "$(date +%s)" "$(free -g | awk 'NR==2{print "total="$2" used="$3" available="$7}')"; sleep 15; done ) \
    > "$REC/free_g_15s.log" 2>&1 &
  mem_pid=$!
  shm_pid=""
  if [ "$MODE" = shmprobe ]; then
    bash "$MAIN/scripts/training/shmmon.sh" --limit-gb "${E2E_SHM_LIMIT_GB:-330}" --interval 15 > "$SHMMON_LOG" 2>&1 &
    shm_pid=$!
    echo "SHMMON_PID=$shm_pid log=$SHMMON_LOG"
  fi

  stop_flag="$REC/.follower-stop"
  rm -f "$stop_flag"
  JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 "$MAIN/.venv/bin/python" -u - "$REC" "$stop_flag" <<'PY' &
import json, pathlib, sys, time
rec, stop = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
DIAG_KEYS = ("loss", "grad_norm", "e2e_enc_grad_norm", "e2e_enc_update_param_ratio", "clip_coef", "clip_coef_ex_enc")
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
        s = r["host_step_s"] - ph.get("checkpoint", 0.0)
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
  trap 'kill "$sampler_pid" "$mem_pid" 2>/dev/null || true; [ -z "$shm_pid" ] || kill "$shm_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; touch "$stop_flag"; wait "$follower_pid" 2>/dev/null || true' EXIT

  echo "TRAIN_START_EPOCH=$(date +%s)"
  uv run --no-sync python scripts/training/train.py "${TRAIN_ARGS[@]}"
  echo "TRAIN_DONE_EPOCH=$(date +%s)"
  kill "$sampler_pid" "$mem_pid" 2>/dev/null || true
  if [ -n "$shm_pid" ]; then kill "$shm_pid" 2>/dev/null || true; wait "$shm_pid" 2>/dev/null || true; fi
  touch "$stop_flag"; wait "$follower_pid"

  JUDGE=(env JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 "$MAIN/.venv/bin/python" "$MAIN/scripts/training/g0/e2e_g6_judge.py")
  STAGE_ARGS=(--attempt "$ATTEMPT" --run "$RUN" --pathcfg "$PATHCFG" --pathcfg-sha "$PATHCFG_SHA"
              --prefetch-expect "$PREFETCH_EXPECT" --run-log "$LOG" --rec "$REC" --evidence-dir "$G6_DIR"
              --steps "$STEPS" --skip "$BENCH_SKIP")
  case "$MODE" in
    bench)
      "${JUDGE[@]}" stage --mode bench "${STAGE_ARGS[@]}"
      exit 0 ;;
    shmprobe)
      "${JUDGE[@]}" stage --mode shmprobe "${STAGE_ARGS[@]}" --shmmon-log "$SHMMON_LOG"
      exit 0 ;;
    ckptprobe)
      # SMOKE_CKPT 证据：读回 50/params（不得用最终 ckpt 代替），叶数 = Wan-9 叶数 + encoder 叶数
      JAX_PLATFORMS=cpu uv run --no-sync python - "$RUN_ROOT" "$WAN9_LEAVES" <<'PY'
import pathlib, sys
import numpy as np
import jax, jax.numpy as jnp
import openpi.models.model as _model
from mme_vla_suite.models.representation.percep_mem import E2E_ENCODER_ATTR
run_root, wan9 = pathlib.Path(sys.argv[1]), int(sys.argv[2])
p50 = run_root / "50" / "params"
nbytes = sum(f.stat().st_size for f in p50.rglob("*") if f.is_file()) if p50.is_dir() else 0
params = _model.restore_params(p50, dtype=jnp.bfloat16)
flat = jax.tree_util.tree_flatten_with_path(params)[0]
paths = ["/".join(str(getattr(k, "key", getattr(k, "name", k))) for k in kp) for kp, _ in flat]
enc = sum(1 for p in paths if E2E_ENCODER_ATTR in p)
finite = all(bool(np.isfinite(np.asarray(x, dtype=np.float32)).all()) for _, x in flat)
print(f"CKPT50_READBACK step=50 path={p50} bytes={nbytes} leaves={len(flat)} enc_leaves={enc} wan9_leaves={wan9} finite={int(finite)}", flush=True)
PY
      "${JUDGE[@]}" stage --mode ckptprobe "${STAGE_ARGS[@]}" --run-root "$RUN_ROOT" --wan9-leaves "$WAN9_LEAVES"
      "${JUDGE[@]}" aggregate --pathcfg-sha "$PATHCFG_SHA" --evidence-dir "$G6_DIR"
      exit 0 ;;
  esac

  # ── 训练后：checkpoint 齐全 + 真实加载最后一份（G7 / G4），e2e 口径收尾断言 ──
  JAX_PLATFORMS=cpu uv run --no-sync python - "$RUN_ROOT" "$MODE" "$REC" "$CONFIG" "$WAN9_LEAVES" $EXPECT_CKPTS <<'PY'
import json, math, pathlib, sys
import numpy as np
import jax, jax.numpy as jnp
import flax.nnx as nnx
import openpi.models.model as _model
from mme_vla_suite.policies.policy_config import _load_resolved_snapshot, _params_have_motion, check_e2e_provenance
from mme_vla_suite.models.representation import percep_mem as pm
from mme_vla_suite.shared import wan_raw9 as wr

run_root, mode, rec, config_name, wan9 = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4], int(sys.argv[5])
expect = sys.argv[6:]
smoke = mode == "smoke"
dirs = sorted((d.name for d in run_root.iterdir() if d.is_dir() and d.name.isdigit()), key=int)
assert dirs == expect, f"checkpoint 集合 {dirs} != 期望 {expect}"
for d in expect:
    assert (run_root / d / "params").is_dir(), f"缺 {d}/params"
cfg, enabled = _load_resolved_snapshot(run_root)
m = cfg.motion
assert enabled and str(m.encoder_kind) == pm.E2E_ENCODER_KIND and str(m.readout) == pm.E2E_READOUT, "快照 motion kind/readout 不符"
assert int(m.stride) == 16 and int(m.budget) == 160 and int(m.latents_per_window) == pm.E2E_LATENTS_PER_WINDOW, "快照 stride/budget/latents_per_window 不符"
assert int(m.encoder.layers) == 1, f"快照 encoder.layers={m.encoder.layers} != 1"
prov = json.loads((run_root / "motion_provenance.json").read_text())
check_e2e_provenance(m, prov, run_root)
pe = prov["policy_encoder"]
assert prov["table_kind"] == "wan_raw_bf16_s16_v1" and prov.get("parameter_count", 0) == 0, "表 table_kind / parameter_count 不符"
assert prov["raw9_source_sha256"] == wr.source_sha256() and prov.get("vae"), "motion_provenance raw9 源码摘要或 vae 缺失"
assert prov.get("encoder", None) is None, "motion_provenance encoder（表侧）应为 null"
assert (pe["init"], pe["trainable"], int(pe["layers"]), int(pe["parameter_count"]), pe["spec_src"]) == \
    ("random", True, 1, pm.E2E_POLICY_ENCODER_PARAMS, pm.E2E_SPEC_SRC), f"policy_encoder 不符: {pe}"
last = run_root / expect[-1]
params = _model.restore_params(last / "params")          # 保留存盘 dtype，用于核 encoder 叶为 f32
flat = jax.tree_util.tree_flatten_with_path(params)[0]
paths = [tuple(str(getattr(k, "key", getattr(k, "name", k))) for k in kp) for kp, _ in flat]
enc = [(p, x) for p, (_, x) in zip(paths, flat) if pm.E2E_ENCODER_ATTR in p]
assert _params_have_motion(params), "最后一份 checkpoint 不含 motion 参数"
assert enc, "checkpoint 不含 encoder 叶"
assert len(flat) == wan9 + len(enc), f"叶数 {len(flat)} != Wan-9 {wan9} + encoder {len(enc)}"
bad_dtype = [("/".join(p), str(x.dtype)) for p, x in enc if np.dtype(x.dtype) != np.float32]
assert not bad_dtype, f"encoder 叶 dtype 非 f32: {bad_dtype[:4]}"
norm_leaves = [p for p in paths if any(("latents_mean" in s or "latents_std" in s) for s in p)]
assert not norm_leaves, f"标准化常量不应在参数树: {norm_leaves[:4]}"
assert all(bool(np.isfinite(np.asarray(x, dtype=np.float32)).all()) for _, x in flat), "参数含非有限值"
print(f"CKPT_LOAD=PASS step={expect[-1]} leaves={len(flat)} enc_leaves={len(enc)} resolved_sha256={prov.get('resolved_sha256')}", flush=True)
if smoke:
    # E2E_TRAINABLE（三节 G4）：逐组梯度范数非零有限（metrics.jsonl 第 0 步的 e2e_enc_grad_norm/<组>）、路径不含 img、
    # 不被 freeze filter 命中
    from mme_vla_suite.training import config as _cfg
    rows = [json.loads(x) for x in (rec / "metrics.jsonl").read_text().splitlines() if x.strip()]
    row = next((r for r in rows if any(k.startswith("e2e_enc_grad_norm/") for k in r)), {})
    groups = {k: r["dec"] for k, r in row.items() if k.startswith("e2e_enc_grad_norm/")}
    nz = sum(1 for v in groups.values() if math.isfinite(v) and v > 0)
    pred = nnx.filterlib.to_predicate(_cfg.get_config(config_name).freeze_filter)
    hits = sum(1 for p, x in enc if pred(p, x)) + sum(1 for p, _ in enc if any("img" in s for s in p))
    ok = len(groups) == 4 and nz == 4 and hits == 0
    print(f"E2E_TRAINABLE={'PASS' if ok else 'FAIL'} leaves={len(flat)} enc_leaves={len(enc)} groups_grad_nonzero={nz}/{len(groups)} "
          f"frozen_hits={hits} groups={sorted(groups)} step={row.get('step')}", flush=True)
    assert ok, "E2E_TRAINABLE 不过"
    print(f"SMOKE20=PASS steps=20 save_load=PASS", flush=True)
else:
    print(f"TRAIN_FINAL_RECORD=PASS step={expect[-1]} ckpts={len(dirs)}", flush=True)
PY
)

body 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" | tee -a "$LOG"
exit "$rc"
