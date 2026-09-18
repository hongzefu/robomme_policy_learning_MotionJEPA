"""在真实GPU上以外部上限1检验导出恢复，记录3步且只收尾同步一次。"""
import gzip
import json
import os
import pathlib
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0,str(pathlib.Path.cwd()/'scripts/training'))
from step_timing import StepTiming

print(f'DIAG_PID={os.getpid()}',flush=True)
root=pathlib.Path('v1-store/bench/mv2-trace-export-fix')
if root.exists(): raise FileExistsError(root)
os.environ['TF_PROFILER_TRACE_VIEWER_MAX_EVENTS']='1'
timer=StepTiming(str(root),3)
x=jnp.ones((512,512),jnp.float32)
compute=jax.jit(lambda a:jnp.tanh(a@a*0.001))
flushed=[]
def flush():
    flushed.append(True)
    x.block_until_ready()
for i in range(5):
    with timer.step(i,flush=flush):
        with timer.phase('train_dispatch'): x=compute(x)
        with timer.phase('data_next'): pass
x.block_until_ready()
assert flushed==[True] and os.environ['TF_PROFILER_TRACE_VIEWER_MAX_EVENTS']=='1'
p,=timer.trace_dir.rglob('*.trace.json.gz')
with gzip.open(p) as f: events=json.load(f)['traceEvents']
steps=[int(e['args']['step_num']) for e in events if e.get('ph')=='X' and e.get('name')=='motionjepa_train']
gpu={e['pid'] for e in events if e.get('name')=='process_name' and e.get('args',{}).get('name','').startswith('/device:GPU:')}
device_events=sum(e.get('ph')=='X' and e.get('pid') in gpu for e in events)
assert steps==[0,1,2] and device_events>0 and len(gpu)==1
meta=json.loads(timer.trace_metadata.read_text())
assert meta['trace_viewer_event_limit']==100000000 and meta['steps']==3
result=dict(passed=True,steps=steps,gpu_count=len(gpu),device_events=device_events,
            exported_events=len(events),flush_calls=len(flushed),external_limit_restored=1,
            exported_limit=meta['trace_viewer_event_limit'])
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print('TRACE_EXPORT_GPU=PASS '+json.dumps(result),flush=True)
