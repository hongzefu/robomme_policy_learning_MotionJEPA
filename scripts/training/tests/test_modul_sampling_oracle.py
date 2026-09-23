"""参考选帧必须保留既有浮点舍入，不能把整数理想公式当作生产结果。"""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_32frame_modul import sample_indices_oracle
from mme_vla_suite.shared.sampling import even_sampling_indices


@pytest.mark.parametrize("count", [16, 32, 64])
def test_full_observed_frame_domain(count):
    for step in range(2304):
        assert sample_indices_oracle(step, count) == even_sampling_indices(step, count)


@pytest.mark.parametrize("step", [153, 161, 177, 259])
def test_rounding_counterexample_is_not_silently_changed(step):
    observed = sample_indices_oracle(step, 64)
    assert observed != [step * i // 63 for i in range(64)]
    assert observed[-1] == step
