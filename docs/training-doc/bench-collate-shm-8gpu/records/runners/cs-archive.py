"""逐字节保存本轮已存在记录并清洗日志；不删除原件、不改变运行目录。"""
import hashlib
import json
from pathlib import Path
import re
import shutil

main = Path('/scratch/hongze/robomme_policy_learning_MotionJEPA')
store = main / 'v1-store'
target = Path('/scratch/hongze/robomme_policy_learning_MotionJEPA-temp/docs/training-doc/bench-collate-shm-8gpu/records')
target.mkdir(exist_ok=False)
manifest = {}


def copy(source, relative):
    if not source.is_file():
        return
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    original = source.read_bytes()
    assert destination.read_bytes() == original
    manifest[str(relative)] = {'sha256': hashlib.sha256(original).hexdigest(),
                               'bytes': len(original), 'source': str(source),
                               'source_mtime_ns': source.stat().st_mtime_ns}


for side in ('old', 'new'):
    source = store / 'bench' / f'bench-collate-shm-{side}'
    if source.is_dir():
        for p in sorted(source.iterdir()):
            if p.is_file():
                copy(p, Path(side) / p.name)
for side in ('a1', 'a2', 'b', 'summary'):
    source = store / 'bench/8x8' / f'cs-guard-{side}'
    if source.is_dir():
        for p in sorted(source.iterdir()):
            if p.is_file():
                copy(p, Path(f'guard-{side}') / p.name)
for side in ('base', 'temp'):
    source = store / 'bench/collate-shm-equiv' / f'robomme_policy_learning_MotionJEPA-{side}'
    for name in ('digests.jsonl', 'indices.json', 'meta.json'):
        copy(source / name, Path(f'input-{side}') / name)
for name in ('c0_main.log', 'c0_base.log', 'c0_temp.log', 'c0_dirty.log', 'c0_selfcheck.log',
             'c0_wrong_source.log', 'c1_old.log', 'c1_old_retry.log', 'c1_new.log', 'c1_compare.log',
             'lock_sha256.txt', 'A_HEAD.txt', 'c0_selfcheck.py', 'c0_preflight.sh'):
    copy(store / 'bench/collate-shm-feasibility' / name, Path('checks') / name)
for name in ('cs-8gpu-runner.sh', 'cs-8gpu-driver.sh', 'cs-guard-runner.sh', 'cs-analyze.py', 'cs-archive.py'):
    copy(store / 'logs' / name, Path('runners') / name)

judgements = []
for name in ('cs-8gpu-old.log', 'cs-8gpu-new.log', 'cs-8gpu-driver.log',
             'cs-guard-a1.log', 'cs-guard-a2.log', 'cs-guard-b.log', 'cs-guard.log'):
    source = store / 'logs' / name
    if not source.is_file():
        continue
    raw = source.read_text(errors='replace')
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', raw.replace('\r', '\n'))
    lines = [line.rstrip() for line in text.splitlines()
             if line.strip() and (('%|' not in line and 'Progress on:' not in line)
                                  or re.search(r'\bStep \d+:|EXIT_CODE=|TrainState 摘要', line))]
    clean = '\n'.join(lines) + '\n'
    # 退出记录无论是否附在进度条后都不得丢失。
    assert re.findall(r'EXIT_CODE=\d+', text) == re.findall(r'EXIT_CODE=\d+', clean)
    assert re.findall(r'\bStep \d+:', text) == re.findall(r'\bStep \d+:', clean)
    assert text.count('TrainState 摘要') == clean.count('TrainState 摘要')
    destination = target / name.replace('.log', '.summary.log')
    destination.write_text(clean)
    manifest[destination.name] = {'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                                  'source': str(source), 'cleaned': True,
                                  'source_mtime_ns': source.stat().st_mtime_ns,
                                  'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    for line in lines:
        if re.search(r'PREFLIGHT=|EXIT_CODE=|BASELINE_ENV=|DET_CHECK=|COLLATE_SHM|OLD_REPRO=|BENCH_8GPU|GUARD_ALL_DONE|DRIVER_ALL_DONE|SCALARS |STATE_DIGEST |BATCH_DIGEST |CANON_CHECK=|INDEX_SEQ=', line):
            judgements.append(f'{name}: {line}')
judgements.extend('c1_compare.log: ' + line for line in
                  (store / 'bench/collate-shm-feasibility/c1_compare.log').read_text().splitlines())
(target / 'judgement_lines.txt').write_text('\n'.join(judgements) + '\n')
(target / 'archive_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(f'ARCHIVE=PASS files={len(manifest)}')
for relative in sorted(manifest):
    print(relative)
