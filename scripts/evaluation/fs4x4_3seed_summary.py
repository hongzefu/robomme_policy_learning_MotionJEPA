"""1127 计划（1127-4task-modul-fs4x4-512-1024-60k-plan.md 第四节第 6 条、第八节）的三 seed 汇总。

读 <eval-root>/fs4x4-{a,b}-s{7,0,42}/EVAL692.txt 首行与 merged/progress.json，输出与
docs/training-doc/t1s-3seed-1120/table1_3seed.json 同格式的两行（键 fs512x4x4、fs1024x4x4）：
每个 seed 一个 {run, eval692, success, pass}，行上 mean_pct / sd_pct = 3 个 success/692×100 的均值与样本标准差（保留 2 位）。
每次评估都要求：EVAL692 首行 EVAL692=PASS 且 episodes=692；merged/progress.json 共 692 局、成功（true）局数等于首行 success。
判定行 FS4X4_3SEED=PASS rows=2 seeds=3；任一项不过打印 FS4X4_3SEED=FAIL … 并非 0 退出（JSON 仍写出，便于排查）。

用法：uv run python scripts/evaluation/fs4x4_3seed_summary.py --out <json> [--eval-root <目录>]
"""
import argparse
import json
from pathlib import Path
import re
import statistics
import sys

EPISODES = 692
DEFAULT_EVAL_ROOT = Path('/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/evaluation')
SEEDS = (7, 0, 42)
ROWS = {  # 行键 -> (名字, RUN_NAME 前缀)
    'fs512x4x4': ('FrameSamp 512 (32 frames x 4x4) modulation', 'fs4x4-a'),
    'fs1024x4x4': ('FrameSamp 1024 (64 frames x 4x4) modulation', 'fs4x4-b'),
}


def read_item(eval_root: Path, run: str):
    """返回 (条目 dict, 失败原因或 None)。"""
    f = eval_root / run / 'EVAL692.txt'
    if not f.is_file():
        return {'run': run, 'eval692': None, 'success': None, 'pass': False}, 'eval692_missing'
    lines = f.read_text(encoding='utf-8').splitlines()
    first = lines[0].strip() if lines else ''
    m_s = re.search(r'\bsuccess=(\d+)\b', first)
    m_e = re.search(r'\bepisodes=(\d+)\b', first)
    success = int(m_s.group(1)) if m_s else None
    item = {'run': run, 'eval692': first, 'success': success, 'pass': False}
    if not first.startswith('EVAL692=PASS'):
        return item, 'eval692_not_pass'
    if not m_e or int(m_e.group(1)) != EPISODES or success is None:
        return item, 'eval692_bad_fields'
    p = eval_root / run / 'merged' / 'progress.json'
    if not p.is_file():
        return item, 'progress_missing'
    prog = json.loads(p.read_text(encoding='utf-8'))
    vals = [v for task in prog.values() for v in task.values()]
    n_true = sum(1 for v in vals if v is True)
    if len(vals) != EPISODES:
        return item, f'progress_episodes={len(vals)}'
    if n_true != success:
        return item, f'progress_success={n_true}!=eval692_success={success}'
    item['pass'] = True
    return item, None


def summarize(eval_root: Path):
    rows, failed = {}, []
    for key, (name, prefix) in ROWS.items():
        row = {'name': name}
        pcts = []
        for seed in SEEDS:
            run = f'{prefix}-s{seed}'
            item, why = read_item(eval_root, run)
            row[f'seed{seed}'] = item
            if why:
                failed.append(f'{run}:{why}')
            else:
                pcts.append(item['success'] / EPISODES * 100)
        if len(pcts) == len(SEEDS):
            row['mean_pct'] = round(statistics.mean(pcts), 2)
            row['sd_pct'] = round(statistics.stdev(pcts), 2)
        else:
            row['mean_pct'] = None
            row['sd_pct'] = None
        rows[key] = row
    verdict = (f'FS4X4_3SEED=PASS rows={len(ROWS)} seeds={len(SEEDS)}' if not failed
               else f'FS4X4_3SEED=FAIL rows={len(ROWS)} seeds={len(SEEDS)} failed={",".join(failed)}')
    return {
        'schema': 'fs4x4-3seed-1127 final',
        'episodes': EPISODES,
        'sd': 'sample stdev over 3 values',
        'seed_rows': rows,
        'verdict': verdict,
    }, not failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--eval-root', type=Path, default=DEFAULT_EVAL_ROOT)
    a = ap.parse_args(argv)
    doc, ok = summarize(a.eval_root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for key, row in doc['seed_rows'].items():
        print(f"FS4X4_ROW key={key} mean_pct={row['mean_pct']} sd_pct={row['sd_pct']} "
              + ' '.join(f"s{s}={row[f'seed{s}']['success']}" for s in SEEDS), flush=True)
    print(doc['verdict'], flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
