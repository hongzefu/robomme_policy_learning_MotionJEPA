"""官方队列控制与计算、成绩、交付三个独立状态的核算入口。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading

try:
    from .queue_store import Controller, QueueStore, publish, read_json
    from .manifest import validate_episodes
except ImportError:
    from queue_store import Controller, QueueStore, publish, read_json
    from manifest import validate_episodes


def aggregate(run_root, *, expect_tasks=16, expect_episodes=800):
    """取得唯一控制锁，只读原receipt并从不可变accepted重建核算。"""
    store = QueueStore(run_root)
    with Controller(store, readonly=True) as controller:
        validate_episodes(controller.manifest, official=controller.config["mode"] == "production")
        if len(controller.episodes) != expect_episodes or len({e["task_name"] for e in controller.episodes.values()}) != expect_tasks:
            raise ValueError("预期母集数量不符")
        return controller.summary()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("initialize")
    init.add_argument("--run-root", type=Path, required=True)
    init.add_argument("--manifest", type=Path, required=True)
    init.add_argument("--run-identity-sha256", required=True)
    init.add_argument("--mode", choices=["production", "validation", "measurement_bootstrap"], default="production")
    control = sub.add_parser("controller")
    control.add_argument("--run-root", type=Path, required=True)
    control.add_argument("--hold-jobs", type=Path)
    control.add_argument("--pool-policy", type=Path)
    control.add_argument("--max-infra-retries", type=int, choices=[1], default=1)
    control.add_argument("--no-auto-renew", action="store_true")
    control.add_argument("--once", action="store_true")
    control.add_argument("--slot", type=Path, action="append", default=[])
    control.add_argument("--recovery-proofs", type=Path)
    report = sub.add_parser("aggregate")
    report.add_argument("--run-root", type=Path, required=True)
    report.add_argument("--expect-tasks", type=int, default=16)
    report.add_argument("--expect-episodes", type=int, default=800)
    report.add_argument("--require-zero-errors", action="store_true")
    report.add_argument("--require-no-uncertain-terminal", action="store_true")
    report.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    store = QueueStore(args.run_root)
    if args.command == "initialize":
        store.initialize(read_json(args.manifest), args.run_identity_sha256, mode=args.mode,
                         official=args.mode == "production")
        print("QUEUE_INITIALIZE=PASS")
        return 0
    if args.command == "aggregate":
        result = aggregate(args.run_root, expect_tasks=args.expect_tasks, expect_episodes=args.expect_episodes)
        if args.out:
            publish(args.out, result)
        print(json.dumps(result, ensure_ascii=False))
        label = "VALIDATION_SCORE" if result["purpose"] == "validation" else "OFFICIAL_SCORE"
        print(f"{label}={result['score_status']} U={result['U']} coverage={result['execution_coverage']}")
        return 0 if result["score_status"] == "PASS" else 2
    stopped = threading.Event()
    for name in (signal.SIGINT, signal.SIGTERM):
        signal.signal(name, lambda *_: stopped.set())
    with Controller(store) as controller:
        for slot in args.slot:
            controller.register_slot(read_json(slot))
        if args.recovery_proofs:
            controller.recover(read_json(args.recovery_proofs))
        # 资源取消由G的已核账本执行；本控制器从不根据账号作业列表猜取消对象。
        while not stopped.is_set():
            controller.process_requests()
            result = controller.summary()
            if args.once or result["compute_status"] in {"COMPUTE_DONE", "STOPPED_ERROR"}:
                print(json.dumps(result, ensure_ascii=False))
                return 2 if result["compute_status"] == "STOPPED_ERROR" else 0
            stopped.wait(1)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
