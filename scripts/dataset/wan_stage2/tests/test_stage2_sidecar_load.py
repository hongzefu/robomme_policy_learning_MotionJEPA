"""回归：sidecar 按路径加载 stage2_project.py（含 frozen dataclass）不得因未登记 sys.modules 而抛 AttributeError。

1114 计划 L0b G4 实跑在 rgbstate venv（Python 3.11）里报
``AttributeError: 'NoneType' object has no attribute '__dict__'``（dataclasses._is_type 查 sys.modules）。
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
SIDECAR_DIR = HERE.parent
REPO = SIDECAR_DIR.parents[2]
MAIN_CHECKOUT_VENV = Path("/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/venvs/wan-stage2-rgbstate")

CODE = (
    "import sys; sys.path.insert(0, sys.argv[1]); import stage2_sidecar as s; "
    "p = s.stage2_project('wan_stage2_rgbstate'); "
    "assert p.name == 'wan_stage2_rgbstate', p; print('OK', p.arch)"
)


def _rgbstate_python():
    for venv in (REPO / "v1-store/venvs/wan-stage2-rgbstate", MAIN_CHECKOUT_VENV):
        python = venv / "bin/python"
        if python.exists():
            return python
    return None


def _run(python):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    result = subprocess.run([str(python), "-c", CODE, str(SIDECAR_DIR)], cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "AttributeError" not in result.stderr
    assert result.stdout.startswith("OK wan-latent-stage2-blockwise-pool4-rgb-state")


def test_sidecar_stage2_project_loads_in_current_interpreter():
    _run(sys.executable)


def test_sidecar_stage2_project_loads_in_rgbstate_venv():
    python = _rgbstate_python()
    if python is None:
        pytest.skip("rgbstate 独立 venv 不存在（worktree 与主检出 v1-store 下均无 bin/python），跳过该解释器用例")
    _run(python)
