#!/usr/bin/env bash
# 本机 692 局评估驱动（1108-4task-modul-wantp768-50k-plan.html §4、§5 E1/G6b、第二部分 1.2、runbook D2/E）。
# 把上次的 GL 外壳（gl_hold_pool.sh → gl_eval_shard.sbatch → run_shard.sh）换成本机 8 卡同时跑 10 个逻辑分片调 run_shard.sh。
#
# 用法：
#   local_692_pool.sh <checkpoint 绝对路径> [seed=7]          正式评估（runbook E，ckpt …/50000）
#   local_692_pool.sh --g6b <冒烟 checkpoint 绝对路径> <seed>   G6b 评估加载冒烟（runbook D2，ckpt …/w768-smoke20/19）
# 可覆盖：RUN_NAME（评估目录名，正式默认 local692-wantp768-50k，G6b 默认 w768-g6b）、
#        STORE（demo 前缀库根，默认 $EVAL_REPO/v1-store/demo-prefix/binfill-3b4de03a0b46）、
#        DRY_RUN=1（只打印计划、片→卡/端口映射与预标清单，不起任何分片、不写评估目录）
#
# 流程（正式）：
#   ① $STORE/E0.txt 首行必须是 DEMO_PREFIX_SRC=PASS（E0，训练前已完成），否则不起跑；
#   ② make_shard_plan.py shards --shards 10 --demo-store $STORE 生成本次 10 片计划（PLAN_OK，C6 检查）；
#   ③ C5：从 docs/training-doc/finalckpt-nomotion59999-gl/records/preseeded-errors.json 只取去重后的 9 个
#      (task, difficulty, episode) 身份（不复用其 by_shard），按本次计划查所属分片，写该片 progress.json 预标 "error"
#      并 ALLOW_RESUME=1；判定行 PRESEED=PASS episodes=9 placed=9 duplicates=0，不过即不起跑；
#   ④ C4：分片 k 用 GPU k%8；每卡一个占位文件 $EVAL_ROOT/gpu-slots/gpu<g>（同名 .lock 做 flock）：
#      先在锁内预占卡位（含本片 <= 2，按分片计数），再后台起 run_shard.sh，把它自身的 PID 作为进程树根写入
#      （server、sidecar 都是它的子孙），片结束在锁内释放；run_shard.sh 的 LOCAL_GPU_SHARE=2 分支只认这份登记；
#   ⑤ wait 全部分片，逐片记 SHARD_EXIT_CODE；只含预标 error 的分片非零退出属预期；
#   ⑥ C7：新 error（不在预标清单）只重跑该局一次：重跑写独立运行目录（结束后移入 s<k>/retry1/），
#      首次尝试的该局视频与 progress/motion_stats/client.log 快照移入 s<k>/attempt0/ 留档、不参与验收，
#      重跑结果替换该局最终记录；正常跑完但未成功的局不重试；
#   ⑦ 按最终目录逐片调 check_shard.py（判据不改）→ merge_shards.py → 判定行
#      EVAL692=PASS episodes=<n> success=<k> rate=<x> preseeded_errors=9 new_errors=<n> retried=<n> expect_motion_stats=1
# G6b 模式：GPU 0 上先单片 1 局（LOCAL_GPU_SHARE=1、显存占比 0.7 默认），再 LOCAL_GPU_SHARE=2、0.45 同时起 2 片各 1 局；
#   全程单个 `nvidia-smi -i 0 -lms 500` 进程取显存峰值；判定行
#   EVAL_LOAD_SMOKE=PASS episodes=1 handshake=PASS share2=PASS peak_mem_gb=<x>
# 退出码按 R10 逐片收齐，不吞码：基础设施失败（分片目录不完整、非 error 类非零退出、check_shard 不过）即非零退出，不合并。
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$EVAL_REPO"

DRY="${DRY_RUN:-0}"
MODE=eval
if [[ "${1:-}" == --g6b ]]; then MODE=g6b; shift; fi
CKPT="${1:?需要 checkpoint 绝对路径}"
SEED="${2:-7}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed 必须是非负整数：$SEED"; exit 2; }
[[ "$CKPT" == /* ]] || { echo "checkpoint 必须是绝对路径：$CKPT"; exit 2; }
[[ "$DRY" == 1 || -d "$CKPT/params" ]] || { echo "checkpoint 缺 params/：$CKPT"; exit 2; }
CKPT_ID="$(basename "$CKPT")"
SHARDS=10
if [[ "$MODE" == g6b ]]; then RUN_NAME="${RUN_NAME:-w768-g6b}"; else RUN_NAME="${RUN_NAME:-local692-wantp768-50k}"; fi
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "RUN_NAME 非法：$RUN_NAME"; exit 2; }
STORE="${STORE:-$EVAL_REPO/v1-store/demo-prefix/binfill-3b4de03a0b46}"
PRESEED_SRC="$EVAL_REPO/docs/training-doc/finalckpt-nomotion59999-gl/records/preseeded-errors.json"
EVAL_ROOT="$EVAL_REPO/v1-store/evaluation/$RUN_NAME"
RETRY_RUN="$RUN_NAME-retry1"
RETRY_ROOT="$EVAL_REPO/v1-store/evaluation/$RETRY_RUN"

# 计划 §4 F8 必要参数（与历史 GL 口径一致，gl_hold_pool.sh 同样写死 EVAL_TIMEOUT=14400）
export MMEVLA_MOTION_OVERFLOW=resample EXPECT_MOTION_STATS=1 EXPECT_DEMO_INJECT=1 CHUNK_EPISODES=20 \
       MMEVLA_MOTION_PROV_RELAX=gpu_name,compute_cap,sm_count EPISODE_WALL_S=2400 EVAL_TIMEOUT=14400 \
       LOCAL_GPU_SHARE=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 ALLOW_RESUME=1 \
       POLICY=perceptual-framesamp-modul-8frame-8x8-wantp768 POLICY_CONFIG=mme_vla_suite_b128_80k \
       EXPECT_CKPT_ID="$CKPT_ID" DEMO_PREFIX_STORE="$STORE"
unset ROBOMME_GPU_RASTER SAPIEN_DISABLE_RAY_TRACING
echo "LOCAL692_START mode=$MODE dry=$DRY run=$RUN_NAME ckpt=$CKPT seed=$SEED store=$STORE $(date -u +%FT%TZ)"
echo "EVAL_ENV MMEVLA_MOTION_OVERFLOW=$MMEVLA_MOTION_OVERFLOW EXPECT_MOTION_STATS=$EXPECT_MOTION_STATS EXPECT_DEMO_INJECT=$EXPECT_DEMO_INJECT" \
     "CHUNK_EPISODES=$CHUNK_EPISODES MMEVLA_MOTION_PROV_RELAX=$MMEVLA_MOTION_PROV_RELAX seed=$SEED max_steps=2000(run_shard.sh)" \
     "EPISODE_WALL_S=$EPISODE_WALL_S EVAL_TIMEOUT=$EVAL_TIMEOUT LOCAL_GPU_SHARE=$LOCAL_GPU_SHARE" \
     "XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION POLICY=$POLICY POLICY_CONFIG=$POLICY_CONFIG EXPECT_CKPT_ID=$EXPECT_CKPT_ID"

# ── 内嵌 Python（经 CLIENT_UV 运行；只用标准库）──────────────────────────────────────────
read -r -d '' PYHELP <<'PY'
import json, shutil, sys
from pathlib import Path

def plan_groups(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["groups"]

def result_root(shard_dir, policy, ckpt, seed):
    return Path(shard_dir) / policy / f"ckpt{ckpt}" / f"seed{seed}"

def preseed_keys(src):
    # 只取身份：各 run 的 by_shard 并集去重，不复用分片号（C5）
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    keys = set()
    for run in data.values():
        for groups in run["by_shard"].values():
            for group, eps in groups.items():
                keys |= {(group, int(e)) for e in eps}
    return sorted(keys)

def cmd_preseed(plan_dir, src, out_root, policy, ckpt, seed, shards):
    keys = preseed_keys(src)
    owner = {}
    for k in range(int(shards)):
        for group, eps in plan_groups(Path(plan_dir) / f"shard{k}.json").items():
            for e in eps:
                owner.setdefault((group, int(e)), []).append(k)
    placed, dup, by_shard = 0, 0, {}
    for key in keys:
        hits = owner.get(key, [])
        if len(hits) > 1:
            dup += 1
        if len(hits) == 1:
            placed += 1
            by_shard.setdefault(f"s{hits[0]}", {}).setdefault(key[0], []).append(str(key[1]))
        print(f"PRESEED_PLACE key={key[0]}/{key[1]} shards={','.join(f's{h}' for h in hits) or 'none'}", flush=True)
    ok = len(keys) == 9 and placed == 9 and dup == 0
    line = f"PRESEED={'PASS' if ok else 'FAIL'} episodes={len(keys)} placed={placed} duplicates={dup}"
    if ok and out_root != "-":
        for tag, groups in by_shard.items():
            root = result_root(Path(out_root) / tag, policy, ckpt, seed)
            root.mkdir(parents=True, exist_ok=True)
            path = root / "progress.json"
            if path.exists():
                sys.exit(f"预标目标已存在，拒绝覆盖：{path}")
            path.write_text(json.dumps({g: {e: "error" for e in eps} for g, eps in groups.items()}, indent=2),
                            encoding="utf-8")
        record = {"source": str(src), "note": "身份引自历史清单，按本次 10 片计划重定位，未复用其分片号",
                  "keys": [f"{g}/{e}" for g, e in keys], "by_shard": by_shard, "verdict": line}
        (Path(out_root) / "preseeded-errors.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                                             encoding="utf-8")
    print(line, flush=True)
    sys.exit(0 if ok else 1)

def cmd_scan(eval_root, plan_dir, policy, ckpt, seed, shards):
    """逐片判定：ok / expected（只含预标 error）/ retry（含新 error，写 retry1-plan.json）/ infra（不可重试）。"""
    rec = json.loads((Path(eval_root) / "preseeded-errors.json").read_text(encoding="utf-8"))
    pre = set(rec["keys"])
    for k in range(int(shards)):
        tag = f"s{k}"
        rc = int((Path(eval_root) / "logs" / f"{tag}.rc").read_text().strip())
        plan = json.loads((Path(plan_dir) / f"shard{k}.json").read_text(encoding="utf-8"))
        planned = {(g, int(e)) for g, eps in plan["groups"].items() for e in eps}
        path = result_root(Path(eval_root) / tag, policy, ckpt, seed) / "progress.json"
        progress = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        done = {(g, int(e)) for g, entries in progress.items() for e in entries}
        errors = sorted(f"{g}/{e}" for g, entries in progress.items() for e, v in entries.items() if v == "error")
        new = [key for key in errors if key not in pre]
        complete = done == planned
        if not complete or (rc != 0 and not errors):
            verdict = "infra"
        elif new:
            verdict = "retry"
            groups = {}
            for key in new:
                task, diff, ep = key.split("/")
                groups.setdefault(f"{task}/{diff}", []).append(int(ep))
            retry = {"shard_tag": f"{tag}-retry1", "candidates": plan["candidates"],
                     "identity_sha256": plan["identity_sha256"],
                     "groups": {g: sorted(v) for g, v in sorted(groups.items())}}
            (Path(eval_root) / tag / "retry1-plan.json").write_text(
                json.dumps(retry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            verdict = "expected" if rc != 0 else "ok"
        print(f"SHARD_SCAN shard={tag} rc={rc} complete={int(complete)} errors={len(errors)} "
              f"preseeded={len(errors) - len(new)} new_errors={len(new)} verdict={verdict} new={','.join(new) or '-'}",
              flush=True)

def _videos_of(videos_dir, key):
    task, diff, ep = key.split("/")
    return sorted(p for p in Path(videos_dir).glob(f"{task}_ep{ep}_*_{diff}.mp4"))

def cmd_integrate(eval_root, retry_root, policy, ckpt, seed, shards):
    """C7：把 retry1 的结果并回最终目录；首次尝试的该局产物移入 attempt0/。"""
    records = []
    for k in range(int(shards)):
        tag = f"s{k}"
        shard_dir = Path(eval_root) / tag
        plan_path = shard_dir / "retry1-plan.json"
        if not plan_path.is_file():
            continue
        keys = [f"{g}/{e}" for g, eps in plan_groups(plan_path).items() for e in eps]
        first_rc = int((Path(eval_root) / "logs" / f"{tag}.rc").read_text().strip())
        retry_rc = int((Path(eval_root) / "logs" / f"{tag}-retry1.rc").read_text().strip())
        final = result_root(shard_dir, policy, ckpt, seed)
        retry_dir = Path(retry_root) / tag
        rres = result_root(retry_dir, policy, ckpt, seed)
        rpath = rres / "progress.json"
        rprog = json.loads(rpath.read_text(encoding="utf-8")) if rpath.is_file() else {}
        rvals = {f"{g}/{e}": v for g, entries in rprog.items() for e, v in entries.items()}
        if set(rvals) != set(keys):
            print(f"RETRY_INFRA shard={tag} rc={retry_rc} 重跑目录不完整：期望 {keys} 实得 {sorted(rvals)}", flush=True)
            sys.exit(1)
        attempt0 = shard_dir / "attempt0"
        (attempt0 / "videos").mkdir(parents=True, exist_ok=False)
        prog = json.loads((final / "progress.json").read_text(encoding="utf-8"))
        shutil.copy2(final / "progress.json", attempt0 / "progress.json")
        mpath = final / "motion_stats.json"
        mstats = json.loads(mpath.read_text(encoding="utf-8")) if mpath.is_file() else {}
        if mpath.is_file():
            shutil.copy2(mpath, attempt0 / "motion_stats.json")
        if (shard_dir / "client.log").is_file():
            shutil.copy2(shard_dir / "client.log", attempt0 / "client.log")
        rm_path = rres / "motion_stats.json"
        rmstats = json.loads(rm_path.read_text(encoding="utf-8")) if rm_path.is_file() else {}
        rlog = retry_dir / "client.log"
        rlines = rlog.read_text(encoding="utf-8", errors="replace").splitlines() if rlog.is_file() else []
        appended = []
        for key in keys:
            group, ep = key.rsplit("/", 1)
            for v in _videos_of(final / "videos", key):
                shutil.move(str(v), str(attempt0 / "videos" / v.name))
            value = rvals[key]
            prog[group][ep] = value
            if value != "error":
                vids = _videos_of(rres / "videos", key)
                if len(vids) != 1:
                    print(f"RETRY_INFRA shard={tag} key={key} 重跑结果非 error 但视频数={len(vids)}", flush=True)
                    sys.exit(1)
                shutil.copy2(vids[0], final / "videos" / vids[0].name)
            if key in rmstats:
                mstats[key] = rmstats[key]
            else:
                mstats.pop(key, None)
            appended += [line for line in rlines if line.startswith(f"DEMO_PREFIX_INJECTED key={key} ")
                         or line == f"DEMO_PREFIX_INJECTED key={key}"]
            records.append({"shard": tag, "key": key, "first_shard_exit_code": first_rc,
                            "retry_shard_exit_code": retry_rc, "attempts": 2, "final": value})
            print(f"RETRY_MERGED shard={tag} key={key} first=error final={value}", flush=True)
        (final / "progress.json").write_text(json.dumps(prog, indent=2), encoding="utf-8")
        if mstats or mpath.is_file():
            (final / "motion_stats.json").write_text(json.dumps(mstats, indent=2), encoding="utf-8")
        if appended:
            # check_shard.py 的 DEMO_INJECT 闸逐行读 client.log：把重跑那次的注入行带上来源标记追加
            with open(shard_dir / "client.log", "a", encoding="utf-8") as f:
                f.write(f"# RETRY1_MERGED from {retry_dir}/client.log\n")
                f.write("\n".join(appended) + "\n")
        shutil.move(str(retry_dir), str(shard_dir / "retry1"))
    (Path(eval_root) / "retries.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
    print(f"RETRY_INTEGRATE=PASS retried={len(records)}", flush=True)

def cmd_summary(eval_root):
    log = json.loads((Path(eval_root) / "merged" / "log.json").read_text(encoding="utf-8"))
    errors = json.loads((Path(eval_root) / "merged" / "errors.json").read_text(encoding="utf-8"))
    pre = set(json.loads((Path(eval_root) / "preseeded-errors.json").read_text(encoding="utf-8"))["keys"])
    retries = json.loads((Path(eval_root) / "retries.json").read_text(encoding="utf-8"))
    err_keys = {f"{e['group']}/{e['episode']}" for e in errors}
    n, k = int(log["episodes"]), int(log["successes"])
    print(f"EVAL692=PASS episodes={n} success={k} rate={k / n:.4f} preseeded_errors={len(err_keys & pre)} "
          f"new_errors={len(err_keys - pre)} retried={len(retries)} expect_motion_stats=1", flush=True)

def cmd_g6b_plans(plan_dir, out_dir):
    # G6b：shard0 / shard1 各取第一条 BinFill（走 demo 注入与 motion demo 窗），各写 1 局计划
    for k, names in ((0, ("g6b-single.json", "g6b-a.json")), (1, ("g6b-b.json",))):
        plan = json.loads((Path(plan_dir) / f"shard{k}.json").read_text(encoding="utf-8"))
        group = sorted(g for g in plan["groups"] if g.startswith("BinFill/"))[0]
        ep = plan["groups"][group][0]
        for name in names:
            one = {"shard_tag": name[:-5], "candidates": plan["candidates"],
                   "identity_sha256": plan["identity_sha256"], "groups": {group: [ep]}}
            (Path(out_dir) / name).write_text(json.dumps(one, indent=2) + "\n", encoding="utf-8")
            print(f"G6B_PLAN file={name} episode={group}/{ep}", flush=True)

def cmd_episode_ok(shard_dir, policy, ckpt, seed):
    path = result_root(shard_dir, policy, ckpt, seed) / "progress.json"
    prog = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    vals = [v for entries in prog.values() for v in entries.values()]
    print(1 if len(vals) == 1 and type(vals[0]) is bool else 0)

def cmd_peak(csv_path):
    vals = [float(x) for x in Path(csv_path).read_text().split() if x.replace(".", "", 1).isdigit()]
    print(f"{max(vals) / 1024:.1f}" if vals else "nan")

cmds = {"preseed": cmd_preseed, "scan": cmd_scan, "integrate": cmd_integrate, "summary": cmd_summary,
        "g6b_plans": cmd_g6b_plans, "episode_ok": cmd_episode_ok, "peak": cmd_peak}
cmds[sys.argv[1]](*sys.argv[2:])
PY
py() { "${CLIENT_UV[@]}" -c "$PYHELP" "$@"; }

# ── 每卡占位（C4）：占位文件每行「<分片标识> <run_shard.sh PID|pending>」，读写都在 <文件>.lock 的 flock 下 ──
SLOT_DIR="$EVAL_ROOT/gpu-slots"
slot_reserve() {  # $1=gpu $2=tag：锁内读已登记数，<2 才登记（含本片 <= 2）
    local f="$SLOT_DIR/gpu$1"
    ( flock 9 || exit 1
      touch "$f"
      n="$(grep -c . "$f")"
      (( n < 2 )) || { echo "SLOT_REFUSED gpu=$1 tag=$2 registered=$n"; exit 3; }
      echo "$2 pending" >> "$f" ) 9>"$f.lock"
}
slot_set_pid() {  # $1=gpu $2=tag $3=run_shard.sh PID
    local f="$SLOT_DIR/gpu$1"
    ( flock 9 || exit 1; sed -i "s/^$2 pending\$/$2 $3/" "$f" ) 9>"$f.lock"
}
slot_release() {  # $1=gpu $2=tag
    local f="$SLOT_DIR/gpu$1"
    ( flock 9 || exit 1; sed -i "/^$2 /d" "$f" ) 9>"$f.lock"
}
# 一片的完整生命周期（后台调用）：预占 → 起 run_shard.sh（重定向而非管道，$! 即 run_shard.sh 自身 PID）→ 登记 PID → wait → 释放 → 写 rc
run_slot() {  # $1=gpu $2=tag $3=shard_index $4=plan $5=run_name $6=log $7=rc 文件
    local gpu="$1" tag="$2" idx="$3" plan="$4" run="$5" log="$6" rcf="$7" pid rc
    if ! slot_reserve "$gpu" "$tag"; then echo 97 > "$rcf"; return; fi
    env CUDA_VISIBLE_DEVICES="$gpu" SHARD_INDEX="$idx" LOCAL_GPU_SLOT_FILE="$SLOT_DIR/gpu$gpu" \
        bash scripts/evaluation/run_shard.sh "$run" "$CKPT" "$plan" "$SEED" "$tag" > "$log" 2>&1 &
    pid=$!
    slot_set_pid "$gpu" "$tag" "$pid"
    echo "SLOT_REGISTERED gpu=$gpu tag=$tag pid=$pid run=$run"
    wait "$pid"; rc=$?
    slot_release "$gpu" "$tag"
    echo "$rc" > "$rcf"
}

# ── ① E0 闸 ──
E0_LINE="$(head -n 1 "$STORE/E0.txt" 2>/dev/null)"
if [[ "$E0_LINE" != "DEMO_PREFIX_SRC=PASS "* ]]; then
    echo "E0_GATE=FAIL store=$STORE line=${E0_LINE:-missing}（前缀库未按同代码生成或未完成，不起跑）"; exit 2
fi
echo "E0_GATE=PASS $E0_LINE"

# ── 评估目录：全新 run 名，不覆盖（第 6 条）；DRY_RUN 只写临时目录 ──
if [[ "$DRY" == 1 ]]; then
    PLAN_DIR="$(mktemp -d)"
    echo "DRY_PLAN_DIR=$PLAN_DIR"
else
    [[ ! -e "$EVAL_ROOT" && ! -e "$RETRY_ROOT" ]] || { echo "评估目录已存在，拒绝覆盖：$EVAL_ROOT / $RETRY_ROOT"; exit 2; }
    PLAN_DIR="$EVAL_ROOT/plans"
    mkdir -p "$PLAN_DIR" "$EVAL_ROOT/logs" "$SLOT_DIR"
fi

# ── ② 本次 10 片计划 ──
"${CLIENT_UV[@]}" scripts/evaluation/make_shard_plan.py shards "$PLAN_DIR" --shards "$SHARDS" --demo-store "$STORE" \
    2>&1 | tee "$PLAN_DIR/make_shard_plan.log"
PLAN_RC=${PIPESTATUS[0]}
PLAN_LINE="$(grep '^PLAN_OK ' "$PLAN_DIR/make_shard_plan.log")"
[[ "$PLAN_RC" == 0 && -n "$PLAN_LINE" ]] || { echo "PLAN_OK 缺失或 make_shard_plan.py rc=$PLAN_RC，不起跑"; exit 2; }
TOTAL="$(sed -n 's/^PLAN_OK .* total=\([0-9]*\) .*/\1/p' <<< "$PLAN_LINE")"
for ((K=0; K<SHARDS; K++)); do
    echo "SHARD_MAP shard=s$K gpu=$((K % 8)) port_base=$((18011 + K * 7)) plan=$PLAN_DIR/shard$K.json"
done

if [[ "$MODE" == g6b ]]; then
    # ════ G6b：评估加载冒烟（runbook D2）════
    py g6b_plans "$PLAN_DIR" "$PLAN_DIR" || { echo "G6B 计划生成失败"; exit 2; }
    if [[ "$DRY" == 1 ]]; then
        echo "DRY g6b ① LOCAL_GPU_SHARE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 CUDA_VISIBLE_DEVICES=0 SHARD_INDEX=0 run_shard.sh $RUN_NAME $CKPT g6b-single.json $SEED g1"
        echo "DRY g6b ② LOCAL_GPU_SHARE=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 GPU 0：g2a(SHARD_INDEX=0, g6b-a.json) + g2b(SHARD_INDEX=1, g6b-b.json) 同时起"
        echo "DRY g6b 显存：nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits -lms 500（单进程）"
        exit 0
    fi
    nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$EVAL_ROOT/logs/g6b-mem.csv" 2>&1 &
    MON=$!
    T0=$(date +%s)
    env LOCAL_GPU_SHARE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 CUDA_VISIBLE_DEVICES=0 SHARD_INDEX=0 \
        bash scripts/evaluation/run_shard.sh "$RUN_NAME" "$CKPT" "$PLAN_DIR/g6b-single.json" "$SEED" g1 \
        > "$EVAL_ROOT/logs/g1.log" 2>&1
    RC1=$?
    T1=$(date +%s)
    echo "SHARD_EXIT_CODE=$RC1 shard=g1 seconds=$((T1 - T0))"
    run_slot 0 g2a 0 "$PLAN_DIR/g6b-a.json" "$RUN_NAME" "$EVAL_ROOT/logs/g2a.log" "$EVAL_ROOT/logs/g2a.rc" &
    P_A=$!
    run_slot 0 g2b 1 "$PLAN_DIR/g6b-b.json" "$RUN_NAME" "$EVAL_ROOT/logs/g2b.log" "$EVAL_ROOT/logs/g2b.rc" &
    P_B=$!
    wait "$P_A"; wait "$P_B"
    T2=$(date +%s)
    kill "$MON" 2>/dev/null; wait "$MON" 2>/dev/null
    RC_A="$(cat "$EVAL_ROOT/logs/g2a.rc")"; RC_B="$(cat "$EVAL_ROOT/logs/g2b.rc")"
    echo "SHARD_EXIT_CODE=$RC_A shard=g2a"; echo "SHARD_EXIT_CODE=$RC_B shard=g2b"
    echo "G6B_TIME single_s=$((T1 - T0)) share2_wall_s=$((T2 - T1))"
    OK1="$(py episode_ok "$EVAL_ROOT/g1" "$POLICY" "$CKPT_ID" "$SEED")"
    OK_A="$(py episode_ok "$EVAL_ROOT/g2a" "$POLICY" "$CKPT_ID" "$SEED")"
    OK_B="$(py episode_ok "$EVAL_ROOT/g2b" "$POLICY" "$CKPT_ID" "$SEED")"
    PEAK="$(py peak "$EVAL_ROOT/logs/g6b-mem.csv")"
    HS=FAIL; [[ "$RC1" == 0 && "$OK1" == 1 ]] && HS=PASS
    S2=FAIL; [[ "$RC_A" == 0 && "$RC_B" == 0 && "$OK_A" == 1 && "$OK_B" == 1 ]] && S2=PASS
    V=FAIL; [[ "$HS" == PASS && "$S2" == PASS ]] && V=PASS
    echo "EVAL_LOAD_SMOKE=$V episodes=1 handshake=$HS share2=$S2 peak_mem_gb=$PEAK" | tee "$EVAL_ROOT/EVAL_LOAD_SMOKE.txt"
    [[ "$V" == PASS ]] || exit 1
    exit 0
fi

# ════ 正式评估 ════
# ── ③ C5 按身份重定位预标 ──
if [[ "$DRY" == 1 ]]; then
    py preseed "$PLAN_DIR" "$PRESEED_SRC" - "$POLICY" "$CKPT_ID" "$SEED" "$SHARDS" || exit 2
    echo "DRY 正式评估：10 片按 SHARD_MAP 同时起（每卡 <= 2 片，flock 占位），之后 scan → retry1 → check_shard → merge"
    exit 0
fi
py preseed "$PLAN_DIR" "$PRESEED_SRC" "$EVAL_ROOT" "$POLICY" "$CKPT_ID" "$SEED" "$SHARDS" \
    | tee "$EVAL_ROOT/logs/preseed.log"
[[ "${PIPESTATUS[0]}" == 0 ]] || { echo "PRESEED 不过，不起跑"; exit 2; }

# ── ④⑤ 10 片同跑、收齐退出码 ──
PIDS=()
for ((K=0; K<SHARDS; K++)); do
    run_slot "$((K % 8))" "s$K" "$K" "$PLAN_DIR/shard$K.json" "$RUN_NAME" "$EVAL_ROOT/logs/s$K.log" "$EVAL_ROOT/logs/s$K.rc" &
    PIDS+=("$!")
done
for P in "${PIDS[@]}"; do wait "$P"; done
for ((K=0; K<SHARDS; K++)); do echo "SHARD_EXIT_CODE=$(cat "$EVAL_ROOT/logs/s$K.rc") shard=s$K"; done

# ── ⑥ 判定 + 新 error 重跑一次 ──
py scan "$EVAL_ROOT" "$PLAN_DIR" "$POLICY" "$CKPT_ID" "$SEED" "$SHARDS" | tee "$EVAL_ROOT/logs/scan.log"
[[ "${PIPESTATUS[0]}" == 0 ]] || { echo "分片判定脚本失败"; exit 1; }
if grep -q 'verdict=infra' "$EVAL_ROOT/logs/scan.log"; then
    echo "EVAL692=FAIL reason=infra（见上方 verdict=infra 的分片，留证交用户，不重跑、不合并）"; exit 1
fi
PIDS=()
for ((K=0; K<SHARDS; K++)); do
    [[ -f "$EVAL_ROOT/s$K/retry1-plan.json" ]] || continue
    run_slot "$((K % 8))" "s$K" "$K" "$EVAL_ROOT/s$K/retry1-plan.json" "$RETRY_RUN" \
        "$EVAL_ROOT/logs/s$K-retry1.log" "$EVAL_ROOT/logs/s$K-retry1.rc" &
    PIDS+=("$!")
done
for P in "${PIDS[@]}"; do wait "$P"; done
for ((K=0; K<SHARDS; K++)); do
    [[ -f "$EVAL_ROOT/logs/s$K-retry1.rc" ]] && echo "SHARD_EXIT_CODE=$(cat "$EVAL_ROOT/logs/s$K-retry1.rc") shard=s$K-retry1"
done
py integrate "$EVAL_ROOT" "$RETRY_ROOT" "$POLICY" "$CKPT_ID" "$SEED" "$SHARDS" | tee "$EVAL_ROOT/logs/integrate.log"
[[ "${PIPESTATUS[0]}" == 0 ]] || { echo "EVAL692=FAIL reason=retry_integrate"; exit 1; }
[[ ! -d "$RETRY_ROOT" ]] || rmdir "$RETRY_ROOT" || { echo "重跑根目录未清空：$RETRY_ROOT"; exit 1; }

# ── ⑦ 按最终目录验收、合并 ──
FAILED=0
for ((K=0; K<SHARDS; K++)); do
    "${CLIENT_UV[@]}" scripts/evaluation/check_shard.py "$EVAL_ROOT/s$K" "$POLICY" "$CKPT_ID" "$SEED" "$PLAN_DIR/shard$K.json" \
        > "$EVAL_ROOT/logs/check-s$K.log" 2>&1
    RC=$?
    echo "CHECK_SHARD_EXIT_CODE=$RC shard=s$K $(grep -o '^SHARD_PASS' "$EVAL_ROOT/logs/check-s$K.log")"
    [[ "$RC" == 0 ]] || FAILED=$((FAILED + 1))
done
[[ "$FAILED" == 0 ]] || { echo "EVAL692=FAIL reason=check_shard failed_shards=$FAILED"; exit 1; }
"${CLIENT_UV[@]}" scripts/evaluation/merge_shards.py "$EVAL_ROOT" --policy "$POLICY" --ckpt "$CKPT_ID" --seed "$SEED" \
    --expect-total "$TOTAL" 2>&1 | tee "$EVAL_ROOT/logs/merge.log"
[[ "${PIPESTATUS[0]}" == 0 ]] || { echo "EVAL692=FAIL reason=merge"; exit 1; }
py summary "$EVAL_ROOT" | tee "$EVAL_ROOT/EVAL692.txt"
[[ "${PIPESTATUS[0]}" == 0 ]] || exit 1
echo "LOCAL692_DONE $(date -u +%FT%TZ)"
