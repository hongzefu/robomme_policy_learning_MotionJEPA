"""本机8×A100跑 context Wan 时间池化1024（orig16，60k）官方800局×三seed：
弹性队列，每卡4个worker；先用GPU 2-7，LATE_MARKER出现后加入GPU 0、1；最后出均值±std。"""
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

REPO = Path(__file__).resolve().parents[2]
V1 = REPO / "v1-store"
CKPT = V1 / "train-runs/mme_vla_suite_orig16_b128_60k/orig100-ctx32x4x4-wan-timepool8x8-b128-60k/59999"
POLICY_CONFIG = "mme_vla_suite_orig16_b128_60k"
MODEL_KIND = "wan_fixedpool1024"
POLICY_PY = "/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv/bin/python"
SIM_PY = "/scratch/hongze/micromamba/envs/robomme/bin/python"
STAGE2_PY = "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/venvs/wan-stage2/bin/python"
BENCHMARK = "/scratch/hongze/robomme_policy_learning_MotionJEPA/third_party/robomme_benchmark"
TRAIN_LOG = V1 / "logs/orig100-ctx32x4x4-wan-timepool8x8-b128-60k.log"
OUT = V1 / "evaluation/ctx-wan-timepool1024-3seed-1107"
LOGS = OUT / "logs"
MANIFEST_SHA = "991c34960b431f4e6170357dfe428e74e404f5ccdbf8dafddc3a0a380eaeecc6"
SEEDS = (42, 7, 0)
GPUS = range(2, 8)
LATE_GPUS = [0, 1]
LATE_MARKER = V1 / "validation/orig16-wan-timepool1024/RESTORE_POOL.json"
PER_GPU = 4
MEM_FRACTION = "0.18"
PORT_BASE = {42: 9300, 7: 9400, 0: 9500}
START_GAP, POLL = 10, 30
QUEUE, WORKER = REPO / "scripts/evaluation/official_queue.py", REPO / "scripts/evaluation/resident_worker.py"
# tmux新会话不继承当前shell，模型与缓存路径写进命令串（与训练runbook同值）。
CACHE_ENV = {"OPENPI_DATA_HOME": V1 / "models", "HF_HOME": V1 / "cache/hf", "XDG_CACHE_HOME": V1 / "cache/xdg",
             "TORCH_HOME": V1 / "cache/torch"}


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
    return f"evl-tp-s{seed}-{worker}"


def start_workers(seed, root, gpus):
    workers = []
    for gpu in gpus:
        for k in range(PER_GPU):
            worker = f"g{gpu}k{k}"
            if workers:
                time.sleep(START_GAP)
            start_session(session(seed, worker),
                          [POLICY_PY, WORKER, "--run-root", root, "--spec", root / "spec.json", "--worker-id", worker,
                           "--local", "--port", PORT_BASE[seed] + gpu * 10 + k], LOGS / f"seed{seed}-{worker}.log",
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


def run_seed(seed, manifest, code):
    root = OUT / f"seed{seed}/run800"
    spec = {"repo": str(REPO), "policy_python": POLICY_PY, "stage2_python": STAGE2_PY, "sim_python": SIM_PY,
            "checkpoint": str(CKPT),
            "policy_config": POLICY_CONFIG, "model_kind": MODEL_KIND, "model_seed": seed, "purpose": "production",
            "run_identity_sha256": identity_sha256({"checkpoint": str(CKPT), "episodes": manifest["sha256"],
                                                    "code": code, "seed": seed, "model_kind": MODEL_KIND})}
    QueueStore(root).initialize(manifest, spec)
    controller = f"evl-tp-ctl-s{seed}"
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
        if not late_started and LATE_MARKER.exists():
            late_started = True
            workers += start_workers(seed, root, LATE_GPUS)
        recover_dead(seed, root, workers, recovered)
        if status == "STOPPED_ERROR" or not any(alive(session(seed, w)) for w in workers):
            say(f"CHAIN_STOP seed={seed} status={status} missing={len(summary.get('missing', []))} "
                f"errors={json.dumps(summary.get('errors', {}), ensure_ascii=False)[:500]}")
            subprocess.run(["tmux", "kill-session", "-t", f"={controller}"])
            return False


def row(group, name, records, field=None):
    rates = {}
    for seed in SEEDS:
        chosen = [r for r in records[seed] if field is None or r[field] == name]
        rates[seed] = 100 * sum(r["outcome"] == "success" for r in chosen) / len(chosen)
    return {"group": group, "name": name, **{f"seed{s}": rates[s] for s in SEEDS},
            "mean": statistics.mean(rates.values()), "std": statistics.stdev(rates.values())}


def three_seed_summary():
    """成功只算outcome==success，timeout算失败；std为ddof=1。"""
    records = {s: [read_json(p) for p in (OUT / f"seed{s}/run800/accepted").glob("*.json")] for s in SEEDS}
    counts = {s: len(records[s]) for s in SEEDS}
    if set(counts.values()) != {800}:
        say(f"THREE_SEED=FAIL counts={counts}")
        return 1
    rows = [row("overall", "overall", records)]
    for group, field in (("difficulty", "difficulty"), ("task", "task_name")):
        rows += [row(group, name, records, field) for name in sorted({r[field] for r in records[SEEDS[0]]})]
    (OUT / "three_seed_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    lines = [f"{'分组':<12}{'名称':<20}" + "".join(f"seed{s:<8}" for s in SEEDS) + "均值±std"]
    lines += [f"{r['group']:<12}{r['name']:<20}" + "".join(f"{r[f'seed{s}']:<12.2f}" for s in SEEDS)
              + f"{r['mean']:.2f}±{r['std']:.2f}" for r in rows]
    (OUT / "three_seed_summary.txt").write_text("\n".join(lines) + "\n")
    say(f"THREE_SEED=PASS seeds=7,42,0 overall={rows[0]['mean']:.2f}±{rows[0]['std']:.2f}")
    return 0


def wait_training():
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


def main():
    if not wait_training() or not (CKPT / "_CHECKPOINT_METADATA").exists() or not wait_gpus():
        say(f"CHAIN_STOP before_eval ckpt={CKPT}")
        return 1
    manifest = build_official_manifest(BENCHMARK, utils_path=REPO / "examples/robomme/utils.py")
    replace_json(OUT / "episodes.json", manifest)
    assert manifest["sha256"] == MANIFEST_SHA
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    for seed in SEEDS:
        if not run_seed(seed, manifest, code):
            return 1
    return three_seed_summary()


if __name__ == "__main__":
    raise SystemExit(main())
