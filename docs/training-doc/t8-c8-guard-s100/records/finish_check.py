"""F3 收尾：带数量验收，清洗日志，逐字节归档已完成记录。"""
import hashlib
import json
from pathlib import Path
import re
import shutil

root = Path.cwd()
source = root/'v1-store/bench/8x8/t8-c8-guard-s100'
target = root/'docs/training-doc/t8-c8-guard-s100/records'
raw = (root/'v1-store/logs/t8-c8-guard-s100.log').read_text()
for expected in (
    'BASELINE_ENV=PASS',
    'SCALARS steps=100 keys=5 hex_mismatch_steps=0',
    'INDEX_SEQ=PASS',
    'BATCH_DIGEST rows=6 mismatch=0',
    'STATE_DIGEST rows=6 mismatch=0',
    'CANON_CHECK=PASS steps=6',
    'DET_CHECK=PASS',
    'EXIT_CODE=0',
):
    if expected not in raw:
        raise SystemExit(f'F3_FINISH=FAIL 缺少判定行：{expected}')
metrics = [json.loads(s) for s in (source/'metrics.jsonl').read_text().splitlines()]
assert [m['step'] for m in metrics] == list(range(100))
base_indices = json.loads((root/'v1-store/bench/8x8/t8-c8-b/index_sequence.json').read_text())['indices']
cand_indices = json.loads((source/'index_sequence.json').read_text())['indices']
assert len(base_indices) >= 800 and len(cand_indices) >= 800
assert base_indices[:800] == cand_indices[:800]
index_line = f'INDEX_TRAIN=PASS n=800 recorded_n={len(cand_indices)}'
for name in ('param_checksums.jsonl', 'batch_digests.jsonl'):
    rows = [json.loads(s) for s in (source/name).read_text().splitlines()]
    assert [r['step'] for r in rows] == [0,1,2,24,49,99]
summary = 'GUARD_GRAD_100=PASS scalars_steps=100 index_n=800 batch_digest_rows=6 state_digest_rows=6'
target.mkdir(parents=True, exist_ok=True)
for name in ('metrics.jsonl','scalars_hex.tsv','batch_digests.jsonl','param_checksums.jsonl',
             'index_sequence.json','env.json','run_meta.json'):
    shutil.copyfile(source/name, target/name)
    assert (source/name).read_bytes() == (target/name).read_bytes()
clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', raw.replace('\r','\n'))
lines = [s.rstrip() for s in clean.splitlines() if s.strip() and 'Progress on:' not in s]
(target/'train.log').write_text('\n'.join(lines) + '\n' + index_line + '\n' + summary + '\n')
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.iterdir()) if p.is_file() and p.name != 'sha256.json'}
(target/'sha256.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
print(summary)
print(index_line)
print('F3_ARCHIVE=PASS files=7')
