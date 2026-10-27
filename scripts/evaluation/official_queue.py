"""弹性worker队列的控制、归还、明确恢复和成绩汇总命令。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading
import time
import uuid

try:
    from .queue_store import Controller, QueueStore, publish, read_json, replace_json, summarize
except ImportError:
    from queue_store import Controller, QueueStore, publish, read_json, replace_json, summarize


def aggregate(root):
    store = QueueStore(root)
    path = store.root / "manifest.json"
    manifest = read_json(path if path.exists() else store.root / "inputs/episodes.json")
    episodes = manifest.get("episodes") or [e for chunk in manifest["chunks"] for e in chunk["episodes"]]
    return summarize(store, read_json(store.root / "queue/state.json"), {e["episode_key"]: e for e in episodes})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("initialize")
    init.add_argument("--run-root", required=True)
    init.add_argument("--manifest", type=Path, required=True)
    init.add_argument("--spec", type=Path, required=True)
    control = sub.add_parser("controller")
    control.add_argument("--run-root", required=True)
    control.add_argument("--once", action="store_true")
    control.add_argument("--slurm-host", default="greatlakes")
    report = sub.add_parser("aggregate")
    report.add_argument("--run-root", required=True)
    report.add_argument("--out", type=Path)
    for operation in ("drain", "recover"):
        item = sub.add_parser(operation)
        item.add_argument("--run-root", required=True)
        item.add_argument("--worker-id", required=True)
        item.add_argument("--request-id")
        if operation == "recover":
            item.add_argument("--job-id", required=True)
            item.add_argument("--step-id", required=True)
        item.add_argument("--wait-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    store = QueueStore(args.run_root)
    if args.command == "initialize":
        store.initialize(read_json(args.manifest), read_json(args.spec))
        return 0
    if args.command == "aggregate":
        result = aggregate(args.run_root)
        if args.out:
            publish(args.out, result)
        print(json.dumps(result, ensure_ascii=False))
        return 2 if result["missing"] else 0
    if args.command in {"drain", "recover"}:
        request_id = args.request_id or f"{args.command}-{uuid.uuid4().hex}"
        payload = {"worker_id": args.worker_id}
        if args.command == "recover":
            payload.update(job_id=args.job_id, step_id=args.step_id)
        store.request(request_id, args.command, payload)
        print(f"REQUEST_ID={request_id}", flush=True)
        end = time.monotonic() + args.wait_seconds
        while time.monotonic() < end:
            response = store.query(request_id)
            if response is not None:
                print(json.dumps(response, ensure_ascii=False))
                return 2 if response["status"] == "REJECTED" else 0
            threading.Event().wait(0.1)
        print("控制器回执未确认，请用原request-id查询或重发。")
        return 3
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    from functools import partial
    try:
        from .queue_store import confirm_step_terminal
    except ImportError:
        from queue_store import confirm_step_terminal
    with Controller(store, terminal_lookup=partial(confirm_step_terminal, slurm_host=args.slurm_host)) as controller:
        while not stop.is_set():
            controller.process_requests()
            result = controller.summary()
            replace_json(store.root / "queue/summary.json", result)
            # 保持响应到操作方停止，让最后一块结束后的worker也能领到完成回执。
            if args.once:
                print(json.dumps(result, ensure_ascii=False), flush=True)
                return 2 if result["compute_status"] == "STOPPED_ERROR" else 0
            stop.wait(0.25)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
