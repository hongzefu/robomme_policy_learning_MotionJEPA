"""按实际 tqdm 进度区间分析起跑稳态，使用 500ms GPU 采样。"""
import bisect
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import statistics

MAIN = Path('/scratch/hongze/robomme_policy_learning_MotionJEPA')
RUN = 'v2-1600ep-m8x8-modul-b128-60k'
record = MAIN/'v1-store/bench'/RUN
text = (MAIN/'v1-store/logs'/f'{RUN}.log').read_text().replace('\r','\n')
started = datetime.fromisoformat(re.search(r'^START_UTC=(.+)$',text,re.M)[1].replace('Z','+00:00'))
points = {}
for line in text.splitlines():
    match = re.search(r'^(\d{2}:\d{2}:\d{2}\.\d+).*Progress on: ([\d,]+)it/',line)
    if not match:
        continue
    step = int(match[2].replace(',',''))
    stamp = datetime.combine(started.date(),datetime.strptime(match[1],'%H:%M:%S.%f').time(),timezone.utc)
    if stamp < started - timedelta(hours=12):
        stamp += timedelta(days=1)
    points.setdefault(step,stamp.timestamp())
assert points and max(points)>=300, '尚未取得 300 步进度记录'
end_step = min(s for s in points if s>=300)
steady = sorted((s,t) for s,t in points.items() if 100<=s<=end_step)
assert len(steady)>=3
intervals = [(s0,s1,t0,t1,(t1-t0)/(s1-s0)) for (s0,t0),(s1,t1) in zip(steady,steady[1:]) if s1>s0 and t1>t0]
threshold = 1.5*statistics.median(row[4] for row in intervals)
groups = {'all':[], 'slow':[], 'other':[]}
starts = [row[2] for row in intervals]
for line in (record/'gpu_util_lms500_first30min.csv').read_text().splitlines():
    fields = [f.strip() for f in line.split(',')]
    if len(fields)!=4:
        continue
    try:
        stamp = datetime.strptime(fields[0],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
        row = (stamp,int(fields[1]),float(fields[2]),float(fields[3]))
    except ValueError:
        continue
    i = bisect.bisect_right(starts,stamp)-1
    if i<0 or stamp>=intervals[i][3]:
        continue
    groups['all'].append(row)
    groups['slow' if intervals[i][4]>threshold else 'other'].append(row)

def util(rows):
    if not rows:
        return {'n':0,'mean_pct':None,'zero_pct_share':None}
    return {'n':len(rows),'mean_pct':statistics.fmean(r[2] for r in rows),
            'zero_pct_share':100*sum(r[2]==0 for r in rows)/len(rows)}

assert {r[1] for r in groups['all']}=={4,5,6,7}
mean_step = (steady[-1][1]-steady[0][1])/(steady[-1][0]-steady[0][0])
remaining = (60000-steady[-1][0])*mean_step
out = {'run':RUN,'train_head':'dd07f18fc385b01eb52db7563fe5f202997b9706',
       'storage':'AWS 本地 NVMe RAID（/dev/md0）','batch_size':128,'num_workers':8,'fsdp_devices':4,
       'warmup_steps':100,'steady_step_range':[steady[0][0],steady[-1][0]],
       'steady_wall_seconds':steady[-1][1]-steady[0][1], 'step_mean_s':mean_step,
       'samples_per_second':128/mean_step,'sampling_ms':500,'n_intervals':len(intervals),
       'slow_interval_threshold_s':threshold,
       'stratification':'按 tqdm 相邻进度区间的平均步时分层；超过区间步时中位数的 1.5 倍为慢区间，不能声称逐个训练步分层。',
       'slow_intervals':sum(r[4]>threshold for r in intervals),
       'util':{name:util(rows) for name,rows in groups.items()},
       'per_gpu':{str(g):util([r for r in groups['all'] if r[1]==g]) for g in (4,5,6,7)},
       'estimated_total_compute_hours':60000*mean_step/3600,
       'estimated_finish_utc':datetime.fromtimestamp(steady[-1][1]+remaining,timezone.utc).isoformat(),
       'eta_note':'仅按起跑稳态外推，不含后续 checkpoint 开销、epoch 边界和资源争用变化。'}
path = Path(__file__).parent/'startup_performance.json'
path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
