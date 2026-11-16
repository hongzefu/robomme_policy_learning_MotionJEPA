"""check_checkpoint.assert_integration 的单测（1115 计划 B5 / C 节 S2 行）：按 run 登记的 integration 核对快照融合方式。

运行：UV_PROJECT_ENVIRONMENT=<主 .venv> PYTHONPATH=<仓库>/src JAX_PLATFORMS=cpu \
      uv run --no-sync python -m pytest scripts/evaluation/test_check_checkpoint_integration.py -q
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parent / 'check_checkpoint.py'
_spec = importlib.util.spec_from_file_location('check_checkpoint', _PATH)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

CONTEXT_CFG = SimpleNamespace(token_per_image=64, integration_type='context')
MODUL_CFG = SimpleNamespace(token_per_image=64, integration_type='modulation')
CTX_RUNS = ('v2-1600ep-m8x8-ctx-motion-b128-50k', 'v17-smoke20')
MODUL_RUN = 'robomme-vla-modul-motion-80k-v1'


@pytest.mark.parametrize('name', CTX_RUNS)
def test_context_snapshot_passes_context_entry(name):
    cc.assert_integration(CONTEXT_CFG, cc.RUNS[name])


def test_modulation_snapshot_passes_existing_entries():
    # 已有条目都不写 integration，缺省按 modulation，行为与改动前一致
    for name, run in cc.RUNS.items():
        if name in CTX_RUNS:
            continue
        assert 'integration' not in run
        cc.assert_integration(MODUL_CFG, run)


def test_context_snapshot_fails_modulation_entry():
    with pytest.raises(AssertionError):
        cc.assert_integration(CONTEXT_CFG, cc.RUNS[MODUL_RUN])


def test_modulation_snapshot_fails_context_entry():
    with pytest.raises(AssertionError):
        cc.assert_integration(MODUL_CFG, cc.RUNS['v2-1600ep-m8x8-ctx-motion-b128-50k'])


def test_token_per_image_still_pinned():
    with pytest.raises(AssertionError):
        cc.assert_integration(SimpleNamespace(token_per_image=16, integration_type='context'), cc.RUNS['v17-smoke20'])


@pytest.mark.parametrize('name', CTX_RUNS)
def test_context_entries_fields(name):
    run = cc.RUNS[name]
    ref = cc.RUNS[MODUL_RUN]
    assert run['integration'] == 'context' and run['motion'] is True and run['budget'] == 512
    assert run['config'] == 'mme_vla_suite_b128_80k' and run['motion_budget'] == 160
    assert run['motion_source_run'] == ref['motion_source_run'] and 'motion_kind' not in run
    assert run['norm_stats_sha256'] == '856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173'
    # 已回填（v17 a085c88 复算）：必须是 64 位小写十六进制
    assert run['resolved_sha256'] == '21791b2964c868119784d086f5ce2334405596d8fa20586e217c4703f018d0e3'
    assert len(run['resolved_sha256']) == 64 and all(c in '0123456789abcdef' for c in run['resolved_sha256'])
