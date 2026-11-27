"""FrameSampDataset (budget, token_per_image, num_views) 白名单（1127 计划第四节第 2 条，审查 train-F1）。

放行 (512,16,1) / (512,64,1) / (1024,16,1)，其余档位在 __init__ 形制断言处拒绝。白名单在读任何库文件之前判定，
所以用不存在的库根做最小 mock：放行档位会越过白名单、在随后的 StoreMeta.load 处因库不存在而失败
（错误信息不含白名单文案），拒绝档位在白名单处 ValueError。另核两份 4×4 yaml 的实际取值落在放行档位。
运行：``JAX_PLATFORMS=cpu uv run --no-sync python -m pytest scripts/training/tests/test_framesamp_dataset_budget.py -q``
"""

from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from omegaconf import OmegaConf

WT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT / "src"))
YAML_DIR = WT / "src/mme_vla_suite/models/config/robomme"

from mme_vla_suite.training import framesamp_dataset as fsd  # noqa: E402
from mme_vla_suite.training.framesamp_dataset import FrameSampDataset  # noqa: E402

WHITELIST_MSG = "不在支持的"


def test_module_from_worktree():
    assert fsd.__file__.startswith(str(WT / "src")), fsd.__file__


def _hc(budget, tpi, nv):
    base = OmegaConf.to_container(OmegaConf.load(YAML_DIR / "perceptual-framesamp-modul.yaml"))
    base.update(budget=budget, token_per_image=tpi, num_views=nv)
    return OmegaConf.create(base)


def _construct(hc, tmp_path):
    return FrameSampDataset(str(tmp_path / "no-such-store"), data_config=None, history_config=hc, action_horizon=20)


@pytest.mark.parametrize("budget,tpi,nv", [(512, 16, 1), (512, 64, 1), (1024, 16, 1)])
def test_whitelist_allows(budget, tpi, nv, tmp_path):
    with pytest.raises(Exception) as ei:
        _construct(_hc(budget, tpi, nv), tmp_path)
    assert WHITELIST_MSG not in str(ei.value), f"放行档位被白名单拒绝: {ei.value}"
    assert "形制断言失败" not in str(ei.value), f"放行档位撞上其他形制断言: {ei.value}"


@pytest.mark.parametrize("budget,tpi,nv", [
    (1024, 64, 1), (2048, 16, 1), (768, 16, 1), (256, 16, 1), (512, 4, 1), (512, 16, 2), (1024, 16, 2), (1000, 16, 1),
])
def test_whitelist_rejects(budget, tpi, nv, tmp_path):
    with pytest.raises(ValueError, match=WHITELIST_MSG) as ei:
        _construct(_hc(budget, tpi, nv), tmp_path)
    assert "(1024,16,1)" in str(ei.value)   # 错误文案与集合同步


@pytest.mark.parametrize("name,budget", [("perceptual-framesamp-modul-32frame-4x4.yaml", 512),
                                         ("perceptual-framesamp-modul-64frame-4x4.yaml", 1024)])
def test_fs4x4_yamls_pass_whitelist(name, budget, tmp_path):
    hc = OmegaConf.load(YAML_DIR / name)
    assert (int(hc.budget), int(hc.token_per_image), int(hc.num_views)) == (budget, 16, 1)
    with pytest.raises(Exception) as ei:
        _construct(hc, tmp_path)
    assert "形制断言失败" not in str(ei.value), ei.value
