"""从已交付的只读进度估算到位时刻；不提交作业、不读取检查点。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import time


def timestamp(value):
    if isinstance(value, (int, float)):
        return float(value)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def estimate_ready(events, delivery_samples, *, run_id, now=None, explicit_eta_ready=None,
                   max_age_seconds=900, target_step=60000, max_rate_ratio=3.0):
    """样本必须携带来源身份；训练结束与传输完成保持两种含义。"""
    now = time.time() if now is None else now
    def unknown(reason):
        return {"status": "UNKNOWN", "reason": reason, "run_id": run_id, "submit_due": False}
    if explicit_eta_ready is not None:
        eta = timestamp(explicit_eta_ready)
        if not math.isfinite(eta):
            return unknown("到位时刻无效")
        return {"status": "KNOWN", "run_id": run_id, "source": "user_explicit",
                "eta_ready": eta, "submit_at": eta - 7200, "submit_due": now >= eta - 7200}
    if len(events) < 3:
        return unknown("至少需要三个有来源的进度事件")
    if any(e.get("run_id") != run_id or not e.get("source_identity") for e in events):
        return unknown("进度run或来源身份不匹配")
    try:
        times = [timestamp(e["timestamp"]) for e in events]
        steps = [int(e["step"]) for e in events]
        if any(not math.isfinite(value) for value in times) or max_age_seconds <= 0:
            return unknown("时间或新鲜度阈值无效")
        if any(e.get("paused", False) for e in events):
            return unknown("训练暂停")
        if now - times[-1] > max_age_seconds or times[-1] > now + 30:
            return unknown("进度陈旧或来自未来")
        if any(b <= a for a, b in zip(times, times[1:])) or any(b <= a for a, b in zip(steps, steps[1:])):
            return unknown("时间或step不严格单调")
        rates = [(b-a)/(d-c) for a, b, c, d in zip(times, times[1:], steps, steps[1:])]
        if min(rates) <= 0 or max(rates)/min(rates) > max_rate_ratio:
            return unknown("近期速率不稳")
        if steps[-1] > target_step or steps[0] < 0:
            return unknown("进度越界")
        if not delivery_samples:
            return unknown("缺保存发布传输校验估时")
        durations = []
        for item in delivery_samples:
            if item.get("run_id") != run_id or not item.get("source_identity"):
                return unknown("交付耗时来源不匹配")
            values = [float(item[key]) for key in ("save_seconds", "publish_seconds", "transfer_seconds", "verify_seconds")]
            if any(not math.isfinite(v) or v < 0 for v in values):
                return unknown("交付耗时无效")
            durations.append(sum(values))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return unknown("进度或交付样本格式错误")
    eta_train = times[-1] + (target_step - steps[-1]) * statistics.median(rates)
    eta_ready = eta_train + max(durations)
    return {"status": "KNOWN", "run_id": run_id, "source": "delivered_progress",
            "eta_train": eta_train, "delivery_seconds": max(durations), "eta_ready": eta_ready,
            "submit_at": eta_ready - 7200, "submit_due": now >= eta_ready - 7200,
            "actual_ready": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--delivery-samples", type=Path)
    parser.add_argument("--eta-ready")
    parser.add_argument("--max-age-seconds", type=float, default=900)
    args = parser.parse_args(argv)
    events = json.loads(args.progress.read_text()) if args.progress else []
    samples = json.loads(args.delivery_samples.read_text()) if args.delivery_samples else []
    result = estimate_ready(events, samples, run_id=args.run_id, explicit_eta_ready=args.eta_ready,
                            max_age_seconds=args.max_age_seconds)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    print(f"ETA_READY={result['status']} submit_due={int(result['submit_due'])}")
    return 0 if result["status"] == "KNOWN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
