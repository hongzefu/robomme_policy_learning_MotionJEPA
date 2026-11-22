"""本机8×A100跑官方800局，--profile 选模型，--seeds 选 seed（缺省 42,7,0 三 seed）：
timepool（context Wan 时间池化1024，先用GPU 2-7，LATE_MARKER出现后加入GPU 0、1）、ctxcat（context concat 1×1024，8卡）、
modulcat16（modulation + motion concat 1×1024，stride 16、budget 192，8卡）。弹性队列，每卡4个worker；
多 seed 出均值±std，单 seed 出成功率与逐任务表。--dry-run 只打印计划，不启动任何进程、不写任何文件。
来源：t1s-1120-evalstack-v42 分支的同名入口（最早由 v10-ctx-wan-timepool-eval 分支 sub/E1、sub/E2 移植）。"""
import argparse
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import time

try:
    from .manifest import build_official_manifest
    from .queue_store import QueueStore, read_json, replace_json
    from .runtime_contract import identity_sha256
except ImportError:
    from manifest import build_official_manifest
    from queue_store import QueueStore, read_json, replace_json
    from runtime_contract import identity_sha256

# 评估 server/client 跑本脚本所在检出的代码（spec["repo"]）；数据与产物固定落主检出的 v1-store（worktree 里没有 v1-store）。
REPO = Path(__file__).resolve().parents[2]
V1 = Path("/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store")
POLICY_PY = "/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv/bin/python"
SIM_PY = "/scratch/hongze/micromamba/envs/robomme/bin/python"
STAGE2_PY = str(V1 / "venvs/wan-stage2/bin/python")
BENCHMARK = "/scratch/hongze/robomme_policy_learning_MotionJEPA/third_party/robomme_benchmark"
MANIFEST_SHA = "991c34960b431f4e6170357dfe428e74e404f5ccdbf8dafddc3a0a380eaeecc6"
DEFAULT_SEEDS = (42, 7, 0)
PER_GPU = 4
MEM_FRACTION = "0.18"
START_GAP, POLL = 10, 30
QUEUE, WORKER = REPO / "scripts/evaluation/official_queue.py", REPO / "scripts/evaluation/resident_worker.py"
# tmux新会话不继承当前shell，模型与缓存路径写进命令串（与训练runbook同值）。
CACHE_ENV = {"OPENPI_DATA_HOME": V1 / "models", "HF_HOME": V1 / "cache/hf", "XDG_CACHE_HOME": V1 / "cache/xdg",
             "TORCH_HOME": V1 / "cache/torch"}
TRAIN_ROOT = V1 / "train-runs/mme_vla_suite_orig16_b128_60k"
# 模型相关取值按profile切换；late_marker为None时不加卡；wait_log为None时只等ckpt出现；
# motion为None时不往spec写motion步长／预算（客户端缺省32／96，旧命令行不变）。
# history_yaml只作记录：server从run目录的history_config快照恢复模型，不按名字读yaml。
PROFILES = {
    "timepool": dict(ckpt=TRAIN_ROOT / "orig100-ctx32x4x4-wan-timepool8x8-b128-60k/59999",
                     model_kind="wan_fixedpool1024", wait_log=V1 / "logs/orig100-ctx32x4x4-wan-timepool8x8-b128-60k.log",
                     out=V1 / "evaluation/ctx-wan-timepool1024-3seed-1107", gpus=range(2, 8), late_gpus=[0, 1],
                     late_marker=V1 / "validation/orig16-wan-timepool1024/RESTORE_POOL.json", prefix="evl-tp",
                     port_base={42: 9300, 7: 9400, 0: 9500}, policy_config="mme_vla_suite_orig16_b128_60k",
                     history_yaml=None, stage2_python=STAGE2_PY, motion=None),
    "ctxcat": dict(ckpt=TRAIN_ROOT / "orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k/59999",
                   model_kind="stage2_pool4_concat", wait_log=V1 / "logs/ctx-wan-timepool1024-3seed-1107/simple-chain.log",
                   out=V1 / "evaluation/ctx-concat1x1024-3seed-1108", gpus=range(0, 8), late_gpus=[], late_marker=None,
                   prefix="evl-cc", port_base={42: 9600, 7: 9700, 0: 9800}, policy_config="mme_vla_suite_orig16_b128_60k",
                   history_yaml="perceptual-framesamp-ctx-32frame-4x4-motion-stage2-1x1024-orig16.yaml",
                   stage2_python=STAGE2_PY, motion=None),
    # 1122 方案：stride 16、budget 192；exec 窗 (1300−32)//16+1=80。由触发链在训练成功后调用，只等 ckpt 59999 与 8 卡空闲。
    "modulcat16": dict(ckpt=TRAIN_ROOT / "orig100-m32x4x4-modul-motion-s2concat1x1024-s16-b128-60k/59999",
                       model_kind="stage2_pool4_concat", wait_log=None,
                       out=V1 / "evaluation/orig100-modul-s2concat1x1024-s16-60k-1122", gpus=range(0, 8), late_gpus=[],
                       late_marker=None, prefix="o16s16-ev", port_base={42: 9900, 7: 10000, 0: 10100},
                       policy_config="mme_vla_suite_orig16_b128_60k",
                       history_yaml="perceptual-framesamp-modul-32frame-4x4-motion-stage2-1x1024-s16-orig16.yaml",
                       stage2_python=STAGE2_PY, motion={"motion_stride": 16, "motion_budget": 192}),
}


def select(name):
    """把选中profile写成模块级常量，其余函数照旧读这些名字。"""
    global PROFILE, CKPT, MODEL_KIND, TRAIN_LOG, OUT, LOGS, GPUS, LATE_GPUS, LATE_MARKER, PREFIX, PORT_BASE
    global POLICY_CONFIG, STAGE2_PYTHON, MOTION
    p = PROFILES[name]
    PROFILE = name
    CKPT, MODEL_KIND, TRAIN_LOG, OUT = p["ckpt"], p["model_kind"], p["wait_log"], p["out"]
    LOGS = OUT / "logs"
    GPUS, LATE_GPUS, LATE_MARKER, PREFIX, PORT_BASE = (p["gpus"], p["late_gpus"], p["late_marker"], p["prefix"],
                                                        p["port_base"])
    POLICY_CONFIG, STAGE2_PYTHON, MOTION = p["policy_config"], p["stage2_python"], p["motion"]


select("timepool")


def say(line):
    print(line, flush=True)


def start_session(name, argv, log, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    variables = [f"{k}={v}" for k, v in {"PYTHONUNBUFFERED": "1", **(env or {})}.items()]
    command = shlex.join(["env", *variables, *map(str, argv)])
    path = shlex.quote(str(log))
    body = f'cd {shlex.quote(str(REPO))}; set -o pipefail; {command} 2>&1 | tee {path}; echo "EXIT_CODE=$?" >> {path}'
    subprocess.run(["tmux", "new-session", "-d", "-s", name, "bash", "-c", body], check=True)


def alive(name):
    return subprocess.run(["tmux", "has-session", "-t", f"={name}"], capture_output=True).returncode == 0


def session(seed, worker):
    return f"{PREFIX}-s{seed}-{worker}"


def port(seed, gpu, k):
    return PORT_BASE[seed] + gpu * 10 + k


def start_workers(seed, root, gpus):
    workers = []
    for gpu in gpus:
        for k in range(PER_GPU):
            worker = f"g{gpu}k{k}"
            if workers:
                time.sleep(START_GAP)
            start_session(session(seed, worker),
                          [POLICY_PY, WORKER, "--run-root", root, "--spec", root / "spec.json", "--worker-id", worker,
                           "--local", "--port", port(seed, gpu, k)], LOGS / f"seed{seed}-{worker}.log",
                          {"CUDA_VISIBLE_DEVICES": gpu, "XLA_PYTHON_CLIENT_MEM_FRACTION": MEM_FRACTION, **CACHE_ENV})
            workers.append(worker)
    return workers


def recover_dead(seed, root, workers, recovered):
    """会话已不在且还占着块的worker归还一次；不重起，其他worker会领走。"""
    state = read_json(root / "queue/state.json")
    busy = {a["grant"]["worker_id"] for a in state["assignments"].values() if not a["closed"]}
    for worker in workers:
        if worker in busy and worker not in recovered and not alive(session(seed, worker)):
            recovered.add(worker)
            result = subprocess.run([POLICY_PY, str(QUEUE), "recover", "--run-root", str(root), "--worker-id", worker,
                                     "--job-id", "local", "--step-id", state["workers"][worker]["step_id"]],
                                    capture_output=True, text=True)
            say(f"RECOVER seed={seed} worker={worker} code={result.returncode} {result.stdout.strip()}")


def make_spec(seed, manifest_sha, code):
    spec = {"repo": str(REPO), "policy_python": POLICY_PY, "stage2_python": STAGE2_PYTHON, "sim_python": SIM_PY,
            "checkpoint": str(CKPT),
            "policy_config": POLICY_CONFIG, "model_kind": MODEL_KIND, "model_seed": seed, "purpose": "production",
            "run_identity_sha256": identity_sha256({"checkpoint": str(CKPT), "episodes": manifest_sha,
                                                    "code": code, "seed": seed, "model_kind": MODEL_KIND})}
    if MOTION is not None:
        spec.update(MOTION)
    return spec


def run_seed(seed, manifest, code):
    root = OUT / f"seed{seed}/run800"
    QueueStore(root).initialize(manifest, make_spec(seed, manifest["sha256"], code))
    controller = f"{PREFIX}-ctl-s{seed}"
    start_session(controller, [POLICY_PY, QUEUE, "controller", "--run-root", root, "--local"],
                  LOGS / f"seed{seed}-controller.log")
    workers = start_workers(seed, root, GPUS)
    late_started, recovered = False, set()
    while True:
        time.sleep(POLL)
        path = root / "queue/summary.json"
        summary = read_json(path) if path.exists() else {}
        status, outcomes = summary.get("compute_status"), summary.get("outcomes", {})
        running = sum(alive(session(seed, w)) for w in workers)
        say(f"PROGRESS seed={seed} completed={summary.get('completed', 0)}/800 "
            f"success={outcomes.get('success', 0)} workers_alive={running}")
        if status == "COMPUTE_DONE":
            say(f"COMPUTE_DONE seed={seed} completed={summary['completed']} success={outcomes['success']} "
                f"fail={outcomes['fail']} timeout={outcomes['timeout']}")
            end = time.monotonic() + 300
            while time.monotonic() < end and any(alive(session(seed, w)) for w in workers):
                time.sleep(5)
            subprocess.run(["tmux", "kill-session", "-t", f"={controller}"])
            return True
        if not late_started and LATE_MARKER is not None and LATE_MARKER.exists():
            late_started = True
            workers += start_workers(seed, root, LATE_GPUS)
        recover_dead(seed, root, workers, recovered)
        if status == "STOPPED_ERROR" or not any(alive(session(seed, w)) for w in workers):
            say(f"CHAIN_STOP seed={seed} status={status} missing={len(summary.get('missing', []))} "
                f"errors={json.dumps(summary.get('errors', {}), ensure_ascii=False)[:500]}")
            subprocess.run(["tmux", "kill-session", "-t", f"={controller}"])
            return False


def load_records(seeds):
    return {s: [read_json(p) for p in (OUT / f"seed{s}/run800/accepted").glob("*.json")] for s in seeds}


def rate(records, name=None, field=None):
    chosen = [r for r in records if field is None or r[field] == name]
    return 100 * sum(r["outcome"] == "success" for r in chosen) / len(chosen)


def groups(records):
    """总体、按难度、按任务三组（名称取自给定记录）。"""
    out = [("overall", "overall", None)]
    for group, field in (("difficulty", "difficulty"), ("task", "task_name")):
        out += [(group, name, field) for name in sorted({r[field] for r in records})]
    return out


def multi_seed_summary(seeds):
    """成功只算outcome==success，timeout算失败；std为ddof=1（需≥2个seed）。"""
    records = load_records(seeds)
    counts = {s: len(records[s]) for s in seeds}
    if set(counts.values()) != {800}:
        say(f"MULTI_SEED=FAIL counts={counts}")
        return 1
    rows = []
    for group, name, field in groups(records[seeds[0]]):
        rates = {s: rate(records[s], name, field) for s in seeds}
        rows.append({"group": group, "name": name, **{f"seed{s}": rates[s] for s in seeds},
                     "mean": statistics.mean(rates.values()), "std": statistics.stdev(rates.values())})
    (OUT / "three_seed_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    lines = [f"{'分组':<12}{'名称':<20}" + "".join(f"seed{s:<8}" for s in seeds) + "均值±std"]
    lines += [f"{r['group']:<12}{r['name']:<20}" + "".join(f"{r[f'seed{s}']:<12.2f}" for s in seeds)
              + f"{r['mean']:.2f}±{r['std']:.2f}" for r in rows]
    (OUT / "three_seed_summary.txt").write_text("\n".join(lines) + "\n")
    say(f"MULTI_SEED=PASS seeds={','.join(map(str, seeds))} overall={rows[0]['mean']:.2f}±{rows[0]['std']:.2f}")
    return 0


def single_seed_summary(seed):
    """单 seed：成功率与逐难度、逐任务表；成功只算outcome==success，timeout算失败；不算std。"""
    records = load_records((seed,))[seed]
    if len(records) != 800:
        say(f"SINGLE_SEED=FAIL seed={seed} count={len(records)}")
        return 1
    rows = []
    for group, name, field in groups(records):
        chosen = [r for r in records if field is None or r[field] == name]
        rows.append({"group": group, "name": name, "seed": seed, "episodes": len(chosen),
                     "success": sum(r["outcome"] == "success" for r in chosen), "rate": rate(records, name, field)})
    (OUT / f"seed{seed}_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    lines = [f"{'分组':<12}{'名称':<20}{'成功/局数':<12}seed{seed}成功率(%)"]
    lines += [f"{r['group']:<12}{r['name']:<20}{str(r['success']) + '/' + str(r['episodes']):<12}{r['rate']:.2f}"
              for r in rows]
    (OUT / f"seed{seed}_summary.txt").write_text("\n".join(lines) + "\n")
    say(f"SINGLE_SEED=PASS seed={seed} overall={rows[0]['success']}/800={rows[0]['rate']:.2f}")
    return 0


def summarize(seeds):
    return single_seed_summary(seeds[0]) if len(seeds) == 1 else multi_seed_summary(seeds)


def wait_training():
    """有wait_log时等其EXIT_CODE；没有时（由上游链在训练成功后调用）只等ckpt的_CHECKPOINT_METADATA出现。"""
    if TRAIN_LOG is None:
        while not (CKPT / "_CHECKPOINT_METADATA").exists():
            time.sleep(POLL)
        say(f"CKPT_READY=PASS ckpt={CKPT}")
        return True
    while True:
        lines = TRAIN_LOG.read_bytes()[-4096:].decode(errors="replace").strip().splitlines()
        last = lines[-1] if lines else ""
        if last.startswith("EXIT_CODE="):
            say("TRAIN_DONE=PASS" if last == "EXIT_CODE=0" else "TRAIN_DONE=FAIL")
            return last == "EXIT_CODE=0"
        time.sleep(POLL)


def wait_gpus():
    end = time.monotonic() + 600
    while True:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                       "--format=csv,noheader,nounits"], text=True)
        used = {int(i): int(m) for i, m in (line.split(",") for line in raw.strip().splitlines())}
        if all(used[g] < 2000 for g in GPUS):
            say("GPU_FREE=PASS")
            return True
        if time.monotonic() >= end:
            say(f"GPU_FREE=FAIL used={used}")
            return False
        time.sleep(5)


def parse_seeds(text):
    seeds = tuple(int(s) for s in text.split(","))
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(f"seed 重复：{text}")
    return seeds


def dry_run(seeds):
    """只打印计划：不启动tmux/进程、不写文件、不碰GPU。"""
    gpus = list(GPUS)
    say(f"DRY_RUN profile={PROFILE} seeds={seeds} workers={len(gpus) * PER_GPU} gpus={gpus} per_gpu={PER_GPU} "
        f"late_gpus={list(LATE_GPUS)}")
    say(f"ckpt={CKPT} exists={(CKPT / '_CHECKPOINT_METADATA').exists()}")
    say(f"model_kind={MODEL_KIND} policy_config={POLICY_CONFIG} history_yaml={PROFILES[PROFILE]['history_yaml']} "
        f"motion={MOTION}")
    say(f"repo={REPO} stage2_python={STAGE2_PYTHON} wait_log={TRAIN_LOG} out={OUT}")
    for seed in seeds:
        ports = [port(seed, g, k) for g in gpus for k in range(PER_GPU)]
        say(f"seed={seed} ports={min(ports)}-{max(ports)} tmux_prefix={PREFIX} controller={PREFIX}-ctl-s{seed} "
            f"workers={session(seed, f'g{gpus[0]}k0')}..{session(seed, f'g{gpus[-1]}k{PER_GPU - 1}')}")
    say(f"manifest_sha_lock={MANIFEST_SHA} summary={'single_seed' if len(seeds) == 1 else 'multi_seed'}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), default="timepool")
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS, help="逗号分隔，缺省 42,7,0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    select(args.profile)
    seeds = args.seeds
    unknown = [s for s in seeds if s not in PORT_BASE]
    if unknown:
        parser.error(f"profile {args.profile} 未登记这些 seed 的端口段：{unknown}")
    if args.dry_run:
        return dry_run(seeds)
    if not wait_training() or not (CKPT / "_CHECKPOINT_METADATA").exists() or not wait_gpus():
        say(f"CHAIN_STOP before_eval ckpt={CKPT}")
        return 1
    manifest = build_official_manifest(BENCHMARK, utils_path=REPO / "examples/robomme/utils.py")
    replace_json(OUT / "episodes.json", manifest)
    assert manifest["sha256"] == MANIFEST_SHA
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    for seed in seeds:
        if not run_seed(seed, manifest, code):
            return 1
    return summarize(seeds)


if __name__ == "__main__":
    raise SystemExit(main())
