"""bench-m8x8-8gpu-w<N>：按 metrics.jsonl 的 wall_time（每 10 步一行）取 it100→末行稳态窗口，
配 500 ms GPU 密采按第 16 条口径出均值 / 0% 占比 / 慢步-非慢步分层；8 卡分表。"""
import bisect, json, statistics, sys
from datetime import datetime, timezone
from pathlib import Path

MAIN = Path('/scratch/hongze/robomme_policy_learning_MotionJEPA')

def analyze(workers: int, fsdp: int = 8) -> dict:
    run = f'bench-m8x8-8gpu-f{fsdp}-w{workers}'
    rec = MAIN / 'v1-store/bench' / run
    rows = [json.loads(l) for l in (rec / 'metrics.jsonl').read_text().splitlines() if l.strip()]
    pts = {r['step']: r['wall_time'] for r in rows}
    steady = sorted((s, t) for s, t in pts.items() if s >= 100)
    assert len(steady) >= 3 and steady[-1][0] >= 290, f'{run}: 稳态点不足 {steady}'
    (s0, t0), (s1, t1) = steady[0], steady[-1]
    mean_step = (t1 - t0) / (s1 - s0)
    intervals = [(a, b, ta, tb, (tb - ta) / (b - a)) for (a, ta), (b, tb) in zip(steady, steady[1:])]
    med = statistics.median(iv[4] for iv in intervals)
    thr = 1.5 * med
    starts = [iv[2] for iv in intervals]
    groups = {'all': [], 'slow': [], 'other': []}
    per_gpu = {}
    mem_peak = {}
    for line in (rec / 'gpu_util_lms500.csv').read_text().splitlines():
        f = [x.strip() for x in line.split(',')]
        if len(f) != 4:
            continue
        try:
            st = datetime.strptime(f[0], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
            g, u, m = int(f[1]), float(f[2]), float(f[3])
        except ValueError:
            continue
        mem_peak[g] = max(mem_peak.get(g, 0), m)
        i = bisect.bisect_right(starts, st) - 1
        if i < 0 or st >= intervals[i][3]:
            continue
        groups['all'].append(u); groups['slow' if intervals[i][4] > thr else 'other'].append(u)
        per_gpu.setdefault(g, []).append(u)
    def util(xs):
        return None if not xs else {'n': len(xs), 'mean_pct': round(statistics.fmean(xs), 2),
                                    'zero_pct_share': round(100 * sum(x == 0 for x in xs) / len(xs), 2)}
    assert set(per_gpu) == set(range(8)), f'采样卡集合 {sorted(per_gpu)}'
    loss = [(r['step'], r['loss']['dec']) for r in rows]
    return {'run': run, 'num_workers': workers, 'fsdp_devices': fsdp, 'mesh': [8 // fsdp, fsdp], 'gpus': '0-7',
            'batch_size': 128, 'per_device_batch': 16, 'storage': 'AWS 本地 NVMe RAID（/dev/md0）',
            'steady_step_range': [s0, s1], 'steady_wall_seconds': round(t1 - t0, 3),
            'step_mean_s': round(mean_step, 4), 'samples_per_second': round(128 / mean_step, 2),
            'step_median_s': round(med, 4), 'n_intervals': len(intervals),
            'slow_intervals': sum(iv[4] > thr for iv in intervals),
            'first100_wall_seconds': round(pts[100] - pts[0], 1),
            'util_all': util(groups['all']), 'util_slow': util(groups['slow']), 'util_other': util(groups['other']),
            'util_per_gpu': {g: util(v) for g, v in sorted(per_gpu.items())},
            'mem_peak_mib': dict(sorted(mem_peak.items())),
            'loss_first_last': [loss[0], loss[-1]],
            'eta_80k_hours': round(80000 * mean_step / 3600, 2)}

if __name__ == '__main__':
    specs = [tuple(int(x) for x in a.split(':')) for a in (sys.argv[1:] or ('8:8', '8:16', '4:16'))]  # fsdp:workers
    out = {f'f{f}w{w}': analyze(w, f) for f, w in specs}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    for w, r in out.items():
        print(f"BENCH_8GPU mesh={r['mesh']} fsdp={r['fsdp_devices']} workers={r['num_workers']} step_mean_s={r['step_mean_s']} samples_per_s={r['samples_per_second']} "
              f"util_mean={r['util_all']['mean_pct']} zero_share={r['util_all']['zero_pct_share']} "
              f"slow_util={r['util_slow'] and r['util_slow']['mean_pct']} other_util={r['util_other']['mean_pct']} "
              f"eta_80k_h={r['eta_80k_hours']}")
