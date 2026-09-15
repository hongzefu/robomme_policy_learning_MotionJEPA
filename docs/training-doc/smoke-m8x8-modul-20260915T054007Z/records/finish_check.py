"""验收本次 smoke 的实际记录，排除共享日志中的历史运行。"""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

root = Path.cwd()
run = sys.argv[1]
record = root/'v1-store/bench'/run
run_root = root/'v1-store/train-runs/mme_vla_suite_b128_60k'/run
target = root/'docs/training-doc'/run/'records'
raw = (root/'v1-store/logs/m8-smoke.log').read_text()
marker = f'RUN={run}\n'
assert raw.count(marker) == 1
log = marker + raw.split(marker,1)[1].split('\nRUN=',1)[0]
assert log.rstrip().endswith('EXIT_CODE=0')
assert 'PREFLIGHT=PASS n=25' in log
assert len(re.findall(r'^CHECK_.*=PASS ',log,re.M)) == 25
assert 'Integration Type: modulation' in log
assert 'Norm stats not found in' not in log
assert 'Loaded norm stats from '+str(root/'v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme') in log
rows = [json.loads(line) for line in (record/'metrics.jsonl').read_text().splitlines()]
assert [r['step'] for r in rows] == list(range(20))
for r in rows:
    for key in ('loss','grad_norm','llm_grad_norm','mem_enc_norm','param_norm'):
        assert math.isfinite(r[key]['dec']), (r['step'],key)
prov = json.loads((run_root/'motion_provenance.json').read_text())
assert prov['motion_enabled'] is False
assert prov['framesamp_manifest_sha256']=='4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918'
yaml = (run_root/'history_config.resolved.yaml').read_text()
assert 'token_per_image: 64' in yaml and 'integration_type: modulation' in yaml
norm = run_root/'19/assets/robomme/norm_stats.json'
assert hashlib.sha256(norm.read_bytes()).hexdigest()=='856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173'
tree = json.loads((record/'param_tree.json').read_text())
assert tree['n_model']==tree['n_ckpt']==61
assert not tree['missing'] and not tree['extra'] and not tree['shape_mismatch']
assert tree['assert_param_tree_exact'] is True
metadata = (run_root/'19/params/_METADATA').read_text()
needles = ["q_einsum_mem', 'w'", "kv_einsum_mem', 'w'", "out_einsum_mem', 'w'",
           "mem_rms_norm', 'scale'", "mem_rms_norm_ffn', 'Dense_0', 'kernel'",
           "mem_rms_norm_ffn', 'Dense_0', 'bias'"]
assert all(n in metadata for n in needles)
target.mkdir(parents=True,exist_ok=True)
for name in ('metrics.jsonl','run_meta.json','param_tree.json'):
    shutil.copyfile(record/name,target/name)
for name in ('motion_provenance.json','history_config.resolved.yaml','history_config.resolved.sha256'):
    shutil.copyfile(run_root/name,target/name)
shutil.copyfile(norm,target/'norm_stats.json')
clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',log.replace('\r','\n'))
(target/'train.log').write_text('\n'.join(s.rstrip() for s in clean.splitlines() if s.strip() and 'Progress on:' not in s)+'\n')
checks=['SMOKE20=PASS steps=20 finite=1 exit_code=0','MEM_PARAMS=PASS n=6',tree['line'],'NORM_STATS=PASS sha256=856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173']
(target/'checks.log').write_text('\n'.join(checks)+'\n')
print('\n'.join(checks))
