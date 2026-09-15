"""追溯对拍收尾：复核四组比较并归档六侧原始摘要。"""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import re
import runpy
import shutil

root = Path.cwd()
source = root/'v1-store/bench/m8-modul-retro'
target = root/'docs/training-doc/m8-modul-retro/records'
log = (root/'v1-store/logs/m8-modul-retro.log').read_text()
assert log.rstrip().endswith('EXIT_CODE=0')
assert 'RETRO_PROFILE=PASS profile=c32' in log
assert 'RETRO_PROFILE=PASS profile=c8' in log
compare = runpy.run_path('scripts/training/tests/compare_fixed_grad.py')['compare']
out = io.StringIO()
data = {}
for profile in ('c32','c8'):
    for side in ('a1','a2','b'):
        key = f'{profile}-{side}'
        row = json.loads((source/key/'grad_summary.json').read_text())
        expected = '07702f0076e640724e9516e911983d7810b423d3' if side != 'b' else '63858f5fa39db26b3c587e9bf7df433b5be5bed4'
        assert row['source']['head'] == expected
        assert len(row['initial_params']) == 61
        assert all(r['n_leaves']==38 for r in row['results'].values())
        data[key] = row
        (target/key).mkdir(parents=True,exist_ok=True)
        for name in ('init_params.json','grad_summary.json'):
            shutil.copyfile(source/key/name,target/key/name)
            assert (source/key/name).read_bytes()==(target/key/name).read_bytes()
    with redirect_stdout(out):
        print(f'COMPARE profile={profile} pair=A1/A2')
        compare(data[f'{profile}-a1'],data[f'{profile}-a2'])
        print(f'COMPARE profile={profile} pair=A1/B')
        compare(data[f'{profile}-a1'],data[f'{profile}-b'])
(target/'comparisons.log').write_text(out.getvalue())
clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',log.replace('\r','\n'))
(target/'train.log').write_text('\n'.join(s.rstrip() for s in clean.splitlines() if s.strip())+'\n')
manifest={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.rglob('*')) if p.is_file() and p.name!='sha256.json'}
(target/'sha256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(out.getvalue(),end='')
print('MODUL_RETRO=PASS profiles=2 comparisons=4 init_leaves=61 grad_leaves=38')
