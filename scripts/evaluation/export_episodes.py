"""把官方评测运行根的 accepted 记录导出成逐局 CSV、逐任务成功率与 run 身份（纯读取，不改任何运行根）。

用法：
  export_episodes.py --queue scripts/evaluation/eval3x3_1110_queue.json --out-dir <目录>   # 队列 out_root 下全部项
  export_episodes.py --run-root <运行根> [--run-root ...] --out-dir <目录>                 # 任意运行根（如 1107）
产物：episodes.csv（model,checkpoint,model_seed,task,episode_id,success,outcome,retries）、per_task.csv、identity.json。
成功只算 outcome == success，timeout 记失败；episode_id 即官方清单原始局号。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.evaluation.queue_store import QueueStore, read_json  # noqa: E402
from scripts.evaluation.runtime_contract import sha256_file  # noqa: E402

COLUMNS = ("model", "checkpoint", "model_seed", "task", "episode_id", "success", "outcome", "retries")
TOTAL = 800


def code_commit(spec, identity):
    if identity and identity.get("eval_code_commit"):
        return identity["eval_code_commit"]
    repo = Path(spec["repo"])
    if repo.exists():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return "unknown"


def load_run(root):
    """读一个运行根：spec、可选 run_identity.json、accepted 与 state.retries；不要求控制器在跑。"""
    root = Path(root).resolve()
    spec = read_json(root / "spec.json")
    identity = read_json(root / "run_identity.json") if (root / "run_identity.json").exists() else None
    checkpoint = Path(spec["checkpoint"])
    model = identity["model"] if identity else checkpoint.parent.name
    accepted = QueueStore(root).accepted()
    state_path = root / "queue/state.json"
    state = read_json(state_path) if state_path.exists() else {"retries": {}, "errors": {}}
    manifest = read_json(root / "manifest.json")
    episodes = manifest.get("episodes") or [e for c in manifest["chunks"] for e in c["episodes"]]
    order = {e["episode_key"]: i for i, e in enumerate(episodes)}
    rows = []
    for key in sorted(accepted, key=lambda k: order[k]):
        record = accepted[key]
        if record["run_identity_sha256"] != spec["run_identity_sha256"]:
            raise ValueError(f"{root} 的 accepted 记录 run 身份与 spec 不符: {key}")
        rows.append({"model": model, "checkpoint": checkpoint.name, "model_seed": spec["model_seed"],
                     "task": record["task_name"], "episode_id": record["episode_id"],
                     "success": int(record["outcome"] == "success"), "outcome": record["outcome"],
                     "retries": int(state["retries"].get(key, 0))})
    info = {"root": str(root), "model": model, "checkpoint_path": str(checkpoint), "checkpoint_step": checkpoint.name,
            "model_seed": spec["model_seed"], "model_kind": spec["model_kind"], "policy_config": spec["policy_config"],
            "eval_code_commit": code_commit(spec, identity), "run_identity_sha256": spec["run_identity_sha256"],
            "manifest_sha256": manifest.get("sha256"), "episodes": len(episodes), "accepted": len(rows),
            "success": sum(r["success"] for r in rows), "retries": sum(state["retries"].values()),
            "errors": len(state.get("errors", {})),
            "spec_sha256": sha256_file(root / "spec.json")}
    return rows, info


def per_task(rows):
    table = {}
    for r in rows:
        key = (r["model"], r["checkpoint"], r["model_seed"], r["task"])
        n, k = table.get(key, (0, 0))
        table[key] = (n + 1, k + r["success"])
    return [{"model": m, "checkpoint": c, "model_seed": s, "task": t, "episodes": n, "success": k,
             "success_rate": round(100 * k / n, 2)} for (m, c, s, t), (n, k) in table.items()]


def queue_roots(queue_path):
    queue = json.loads(Path(queue_path).read_text(encoding="utf-8"))
    v1 = Path(os.environ.get("MMEVLA_V1_STORE") or REPO / "v1-store")
    out = Path(queue["out_root"])
    out = out if out.is_absolute() else (v1.joinpath(*out.parts[1:]) if out.parts[0] == "v1-store" else REPO / out)
    entries = list(queue["items"]) + (list(queue["optional_tail"]) if queue.get("optional_tail_enabled") else [])
    names = [f"{e['model']}-{e['step']}-seed{e['seed']}" for e in entries]
    return [out / n for n in names if (out / n / "spec.json").exists()]


def export(roots, out_dir, *, allow_partial=False):
    out_dir = Path(out_dir)
    all_rows, infos = [], []
    for root in roots:
        rows, info = load_run(root)
        if info["accepted"] != info["episodes"] and not allow_partial:
            raise ValueError(f"{root} 只有 {info['accepted']}/{info['episodes']} 局 accepted（加 --allow-partial 才导出）")
        all_rows += rows
        infos.append(info)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    tasks = per_task(all_rows)
    with (out_dir / "per_task.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tasks[0]) if tasks else ["model"])
        writer.writeheader()
        writer.writerows(tasks)
    (out_dir / "identity.json").write_text(json.dumps(infos, ensure_ascii=False, indent=1) + "\n")
    for info in infos:
        print(f"EXPORT_ITEM model={info['model']} step={info['checkpoint_step']} seed={info['model_seed']} "
              f"success={info['success']}/{info['accepted']} retries={info['retries']} errors={info['errors']} "
              f"code={info['eval_code_commit'][:12]} run_identity_sha256={info['run_identity_sha256']}", flush=True)
    print(f"EXPORT=PASS items={len(infos)} rows={len(all_rows)} out={out_dir}", flush=True)
    return all_rows, infos


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--queue", type=Path)
    source.add_argument("--run-root", type=Path, action="append")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    roots = queue_roots(args.queue) if args.queue else args.run_root
    export(roots, args.out_dir, allow_partial=args.allow_partial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
