"""沿用 f2c9dc7 的 100→290 稳态窗口与 500 ms 采样统计口径。"""
import bisect
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

store = Path(__file__).resolve().parents[1]


def analyze(side):
    rec = store / "bench" / f"bench-collate-shm-{side}"
    rows = [json.loads(line) for line in (rec / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == list(range(0, 300, 10))
    points = {r["step"]: r["wall_time"] for r in rows}
    steady = sorted((s, t) for s, t in points.items() if 100 <= s <= 290)
    intervals = [(a, b, ta, tb, (tb-ta)/(b-a)) for (a, ta), (b, tb) in zip(steady, steady[1:])]
    threshold = 1.5 * statistics.median(x[4] for x in intervals)
    starts = [x[2] for x in intervals]
    groups = {"all": [], "slow": [], "other": []}
    per_gpu = {}
    for line in (rec / "gpu_util_lms500.csv").read_text().splitlines():
        timestamp, gpu, util, memory = [x.strip() for x in line.split(",")]
        timestamp = datetime.strptime(timestamp, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
        gpu, util = int(gpu), float(util)
        i = bisect.bisect_right(starts, timestamp) - 1
        if i < 0 or timestamp >= intervals[i][3]:
            continue
        groups["all"].append(util)
        groups["slow" if intervals[i][4] > threshold else "other"].append(util)
        per_gpu.setdefault(gpu, []).append(util)
    assert set(per_gpu) == set(range(8))

    def stats(values):
        return None if not values else {"n": len(values), "mean_pct": statistics.fmean(values),
                                       "zero_pct_share": 100 * sum(x == 0 for x in values) / len(values)}

    step_mean = (points[290] - points[100]) / 190
    return {"side": side, "step_mean_s": step_mean, "samples_per_second": 128 / step_mean,
            "batch_size": 128, "workers": 16, "mesh": [1, 8], "gpu": "8 × A100-SXM4-80GB",
            "storage": "AWS 本地 NVMe RAID /dev/md0 XFS", "warmup": [0, 100], "steady": [100, 290],
            "sample_interval_ms": 500, "intervals": intervals, "slow_threshold_s": threshold,
            "slow_intervals": sum(x[4] > threshold for x in intervals),
            "util_all": stats(groups["all"]), "util_slow": stats(groups["slow"]),
            "util_other": stats(groups["other"]), "per_gpu": {g: stats(v) for g, v in per_gpu.items()}}


results = {side: analyze(side) for side in ("old", "new")}
for side, r in results.items():
    print(f"BENCH_8GPU side={side} workers=16 step_mean_s={r['step_mean_s']:.6f} "
          f"samples_per_s={r['samples_per_second']:.3f} util_mean={r['util_all']['mean_pct']:.3f} "
          f"zero_share={r['util_all']['zero_pct_share']:.3f} slow_util={r['util_slow']} other_util={r['util_other']}")
speedup = results["old"]["step_mean_s"] / results["new"]["step_mean_s"]
print(f"COLLATE_SHM_SPEED old={results['old']['step_mean_s']:.6f} new={results['new']['step_mean_s']:.6f} speedup={speedup:.6f}")
deviation = abs(results["old"]["step_mean_s"] / 1.818 - 1)
results.update(speedup=speedup, old_reference_relative_deviation=deviation)
(store / "bench/bench-collate-shm-new/analysis.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
print(f"OLD_REPRO={'PASS' if deviation <= .05 else 'FAIL'} relative_deviation={deviation:.6f}")
raise SystemExit(0 if deviation <= .05 else 1)
