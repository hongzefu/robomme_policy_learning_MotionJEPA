"""check_checkpoint 的 1127 计划 4×4 登记单测（1127-4task-modul-fs4x4-512-1024-60k-plan.md 第四节第 6、8 条）。

覆盖：旧条目（fs512 = robomme-vla-modul-60k-v1、fs1024 = robomme-vla-modul-1024-80k-v1 等）的 token_per_image 仍缺省 64、
结果不变；4 个新条目在 mock 快照上 check_registration PASS，token_per_image / budget / resolved 不符 FAIL；
新 YAML 的 resolved_sha256 复算与登记值一致。

运行：UV_PROJECT_ENVIRONMENT=<policy-eval-server 环境> PYTHONPATH=<worktree>/src:<worktree> JAX_PLATFORMS=cpu \
      uv run --frozen --no-sync python -m pytest scripts/evaluation/test_check_checkpoint_fs4x4.py -q
"""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

_PATH = Path(__file__).resolve().parent / 'check_checkpoint.py'
_spec = importlib.util.spec_from_file_location('check_checkpoint', _PATH)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

REPO = Path(__file__).resolve().parents[2]
CFG_DIR = REPO / 'src/mme_vla_suite/models/config/robomme'
NORM = '856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173'
NEW = {  # run 根名 -> (budget, step, yaml)
    'v2-1600ep-m32x4x4-modul-b128-50k': (512, '50000', 'perceptual-framesamp-modul-32frame-4x4.yaml'),
    'v2-1600ep-m64x4x4-modul-b128-50k': (1024, '50000', 'perceptual-framesamp-modul-64frame-4x4.yaml'),
    'fs4a-smoke20': (512, '19', 'perceptual-framesamp-modul-32frame-4x4.yaml'),
    'fs4b-smoke20': (1024, '19', 'perceptual-framesamp-modul-64frame-4x4.yaml'),
}
# 旧条目（Table I fs 行 + 8×8 全系）：均不写 token_per_image，缺省 64
OLD_FS = {
    'robomme-vla-modul-60k-v1': (512, '50000', 'perceptual-framesamp-modul-8frame-8x8.yaml'),
    'robomme-vla-modul-1024-80k-v1': (1024, '50000', 'perceptual-framesamp-modul-16frame-8x8.yaml'),
    'robomme-vla-modul-2048-80k-v1': (2048, '50000', 'perceptual-framesamp-modul-32frame-8x8.yaml'),
    'robomme-vla-modul-4096-80k-v1': (4096, '50000', 'perceptual-framesamp-modul-64frame-8x8.yaml'),
}
T1S_MODELS = Path('/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/models/t1s-1120')


def make_mock(tmp_path, run_name, step, yaml_name, mutate=None):
    """按训练 init_history_config 同口径写 run 根快照（.yaml/.sha256/motion_provenance.json）与 ckpt/assets/norm_stats.json。"""
    cfg = OmegaConf.load(str(CFG_DIR / yaml_name))
    if mutate:
        mutate(cfg)
    raw = OmegaConf.to_yaml(cfg, resolve=True).encode('utf-8')
    root = tmp_path / run_name
    (root / step / 'assets/robomme').mkdir(parents=True)
    (root / 'history_config.resolved.yaml').write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    (root / 'history_config.resolved.sha256').write_text(sha + '\n')
    (root / 'motion_provenance.json').write_text(json.dumps({'motion_enabled': False, 'resolved_sha256': sha}))
    norm = b'{"mock": true}'
    (root / step / 'assets/robomme/norm_stats.json').write_bytes(norm)
    return root / step, hashlib.sha256(norm).hexdigest()


def run_check(ckpt, run, norm_sha):
    # mock 的 norm_stats 字节不可能等于真 856c…，只替换这一字段，其余登记字段原样
    cfg, enabled = cc._load_resolved_snapshot(ckpt.parent)
    cc.check_registration(ckpt, dict(run, norm_stats_sha256=norm_sha), cfg, enabled)


# ── 旧条目不变 ──
def test_old_entries_have_no_token_per_image():
    for name, run in cc.RUNS.items():
        if name in NEW:
            continue
        assert 'token_per_image' not in run, name
        with pytest.raises(AssertionError):  # 缺省仍是 64：16 不放行
            cc.assert_integration(SimpleNamespace(token_per_image=16, integration_type=run.get('integration', 'modulation')), run)


@pytest.mark.parametrize('name', sorted(OLD_FS))
def test_old_fs_entry_mock_pass(tmp_path, name):
    budget, step, yml = OLD_FS[name]
    run = cc.RUNS[name]
    assert run['budget'] == budget and run['motion'] is False
    ckpt, norm_sha = make_mock(tmp_path, name, step, yml)
    run_check(ckpt, run, norm_sha)  # 8×8 yaml 复算出已登记 resolved_sha256，算法与旧值一致


@pytest.mark.parametrize('name', ['robomme-vla-modul-60k-v1', 'robomme-vla-modul-1024-80k-v1'])
def test_old_fs_entry_mock_4x4_fails(tmp_path, name):
    budget, step, yml = OLD_FS[name]
    ckpt, norm_sha = make_mock(tmp_path, name, step, yml, mutate=lambda c: setattr(c, 'token_per_image', 16))
    with pytest.raises(AssertionError):
        run_check(ckpt, cc.RUNS[name], norm_sha)


@pytest.mark.parametrize('name', ['robomme-vla-modul-60k-v1', 'robomme-vla-modul-1024-80k-v1'])
def test_old_fs_entry_real_snapshot(name):
    """真实 Table I fs512 / fs1024 权重（只读）：登记核对整段 PASS（不建模型、不读参数树）。"""
    ckpt = T1S_MODELS / name / '50000'
    if not (ckpt / 'assets/robomme/norm_stats.json').is_file():
        pytest.skip(f'本机无 {ckpt}')
    cfg, enabled = cc._load_resolved_snapshot(ckpt.parent)
    cc.check_registration(ckpt, cc.RUNS[name], cfg, enabled)


# ── 新条目 ──
@pytest.mark.parametrize('name', sorted(NEW))
def test_new_entry_fields(name):
    budget, step, yml = NEW[name]
    run = cc.RUNS[name]
    assert run['steps'] == (step,) and run['motion'] is False and run['budget'] == budget
    assert run['token_per_image'] == 16 and run['config'] == 'mme_vla_suite_b128_80k'
    assert run['norm_stats_sha256'] == NORM and 'integration' not in run
    assert not any(k.startswith('motion_') for k in run)
    assert run['resolved_sha256'] == cc.resolved_sha256_of_yaml(CFG_DIR / yml)
    assert len(run['resolved_sha256']) == 64 and all(c in '0123456789abcdef' for c in run['resolved_sha256'])


def test_new_resolved_sha_values():
    assert cc.resolved_sha256_of_yaml(REPO / cc.FS4X4A_YAML) == cc.FS4X4A_RESOLVED_SHA256
    assert cc.resolved_sha256_of_yaml(REPO / cc.FS4X4B_YAML) == cc.FS4X4B_RESOLVED_SHA256
    assert cc.FS4X4A_RESOLVED_SHA256 != cc.FS4X4B_RESOLVED_SHA256


@pytest.mark.parametrize('name', sorted(NEW))
def test_new_entry_mock_pass(tmp_path, name):
    budget, step, yml = NEW[name]
    ckpt, norm_sha = make_mock(tmp_path, name, step, yml)
    run_check(ckpt, cc.RUNS[name], norm_sha)


@pytest.mark.parametrize('name', sorted(NEW))
def test_new_entry_token_per_image_mismatch_fails(tmp_path, name):
    budget, step, yml = NEW[name]
    ckpt, norm_sha = make_mock(tmp_path, name, step, yml, mutate=lambda c: setattr(c, 'token_per_image', 64))
    with pytest.raises(AssertionError):
        run_check(ckpt, cc.RUNS[name], norm_sha)
    with pytest.raises(AssertionError):
        cc.assert_integration(SimpleNamespace(token_per_image=64, integration_type='modulation'), cc.RUNS[name])


def test_new_entry_budget_swap_fails(tmp_path):
    # A 的 ckpt 拿 B 的 yaml（budget 1024）快照：budget 断言拦住
    ckpt, norm_sha = make_mock(tmp_path, 'fs4a-smoke20', '19', 'perceptual-framesamp-modul-64frame-4x4.yaml')
    with pytest.raises(AssertionError):
        run_check(ckpt, cc.RUNS['fs4a-smoke20'], norm_sha)


def test_new_entry_wrong_step_fails(tmp_path):
    ckpt, norm_sha = make_mock(tmp_path, 'v2-1600ep-m32x4x4-modul-b128-50k', '19',
                               'perceptual-framesamp-modul-32frame-4x4.yaml')
    with pytest.raises(AssertionError):
        run_check(ckpt, cc.RUNS['v2-1600ep-m32x4x4-modul-b128-50k'], norm_sha)


def test_new_entry_integration_context_fails(tmp_path):
    ckpt, norm_sha = make_mock(tmp_path, 'fs4b-smoke20', '19', 'perceptual-framesamp-modul-64frame-4x4.yaml',
                               mutate=lambda c: setattr(c, 'integration_type', 'context'))
    with pytest.raises(AssertionError):
        run_check(ckpt, cc.RUNS['fs4b-smoke20'], norm_sha)
