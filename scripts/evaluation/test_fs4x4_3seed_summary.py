"""fs4x4_3seed_summary.py 单测（1127 计划第四节第 8 条）：mock 6 个评估目录，核输出与 table1_3seed.json 同格式。

运行：uv run --frozen --no-sync python -m pytest scripts/evaluation/test_fs4x4_3seed_summary.py -q
"""
import importlib.util
import json
from pathlib import Path
import statistics

import pytest

_PATH = Path(__file__).resolve().parent / 'fs4x4_3seed_summary.py'
_spec = importlib.util.spec_from_file_location('fs4x4_3seed_summary', _PATH)
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)

SUCC = {'fs4x4-a-s7': 230, 'fs4x4-a-s0': 241, 'fs4x4-a-s42': 225,
        'fs4x4-b-s7': 250, 'fs4x4-b-s0': 262, 'fs4x4-b-s42': 244}


def write_run(root, run, success, episodes=692, verdict='PASS', progress_true=None, progress_n=692):
    d = root / run
    (d / 'merged').mkdir(parents=True)
    (d / 'EVAL692.txt').write_text(
        f'EVAL692={verdict} episodes={episodes} success={success} rate={success / 692:.4f} preseeded_errors=9 '
        f'new_errors=0 retried=0 expect_motion_stats=0\n', encoding='utf-8')
    t = success if progress_true is None else progress_true
    vals = [True] * t + ['error'] * 9 + [False] * (progress_n - t - 9)
    prog = {'BinFill/easy': {str(i): v for i, v in enumerate(vals[:300])},
            'RouteStick/hard': {str(i): v for i, v in enumerate(vals[300:])}}
    (d / 'merged/progress.json').write_text(json.dumps(prog), encoding='utf-8')


@pytest.fixture
def full(tmp_path):
    for run, s in SUCC.items():
        write_run(tmp_path, run, s)
    return tmp_path


def test_pass_format(full, tmp_path, capsys):
    out = tmp_path / 'o' / 'fs4x4_3seed.json'
    assert sm.main(['--out', str(out), '--eval-root', str(full)]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == 'FS4X4_3SEED=PASS rows=2 seeds=3'
    doc = json.loads(out.read_text())
    # 与 table1_3seed.json 同顶层键（本汇总不评中间 ckpt，故无 ckpt_rows）
    assert doc['episodes'] == 692 and doc['sd'] == 'sample stdev over 3 values'
    assert doc['verdict'] == 'FS4X4_3SEED=PASS rows=2 seeds=3'
    assert list(doc['seed_rows']) == ['fs512x4x4', 'fs1024x4x4']
    for key, prefix in (('fs512x4x4', 'fs4x4-a'), ('fs1024x4x4', 'fs4x4-b')):
        row = doc['seed_rows'][key]
        assert set(row) == {'name', 'seed7', 'seed0', 'seed42', 'mean_pct', 'sd_pct'}
        pcts = []
        for s in (7, 0, 42):
            it = row[f'seed{s}']
            run = f'{prefix}-s{s}'
            assert set(it) == {'run', 'eval692', 'success', 'pass'}
            assert it['run'] == run and it['success'] == SUCC[run] and it['pass'] is True
            assert it['eval692'].startswith(f'EVAL692=PASS episodes=692 success={SUCC[run]} ')
            pcts.append(SUCC[run] / 692 * 100)
        assert row['mean_pct'] == round(statistics.mean(pcts), 2)
        assert row['sd_pct'] == round(statistics.stdev(pcts), 2)


def test_matches_table1_arithmetic(tmp_path):
    # Table I fs512 行：211/231/223 → 32.03 ± 1.45（table1_3seed.json 记录值）
    for run, s in (('fs4x4-a-s7', 211), ('fs4x4-a-s0', 231), ('fs4x4-a-s42', 223),
                   ('fs4x4-b-s7', 211), ('fs4x4-b-s0', 231), ('fs4x4-b-s42', 223)):
        write_run(tmp_path, run, s)
    doc, ok = sm.summarize(tmp_path)
    assert ok and doc['seed_rows']['fs512x4x4']['mean_pct'] == 32.03 and doc['seed_rows']['fs512x4x4']['sd_pct'] == 1.45


@pytest.mark.parametrize('kw,why', [
    (dict(verdict='FAIL'), 'eval692_not_pass'),
    (dict(episodes=600), 'eval692_bad_fields'),
    (dict(progress_true=200), 'progress_success'),
    (dict(progress_n=691), 'progress_episodes'),
])
def test_fail_cases(tmp_path, kw, why):
    for run, s in SUCC.items():
        write_run(tmp_path, run, s, **(kw if run == 'fs4x4-b-s42' else {}))
    doc, ok = sm.summarize(tmp_path)
    assert not ok and doc['verdict'].startswith('FS4X4_3SEED=FAIL') and f'fs4x4-b-s42:{why}' in doc['verdict']
    assert doc['seed_rows']['fs1024x4x4']['mean_pct'] is None
    assert doc['seed_rows']['fs512x4x4']['mean_pct'] is not None


def test_missing_run(tmp_path, capsys):
    for run, s in SUCC.items():
        if run != 'fs4x4-a-s7':
            write_run(tmp_path, run, s)
    assert sm.main(['--out', str(tmp_path / 'x.json'), '--eval-root', str(tmp_path)]) == 1
    assert 'fs4x4-a-s7:eval692_missing' in capsys.readouterr().out
