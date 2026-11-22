"""官方评估 seed 闸门：只放行官方三个模型 seed 0／7／42；不创建仿真环境、不启动模型。"""

import pytest

from examples.robomme import eval as candidate


SEED_MSG = "seed 须为官方"


def _evaluate_error(**overrides):
    """调用 evaluate 并返回抛出的异常；seed 判断之后缺身份参数等错误也会被捕获。"""
    with pytest.raises(Exception) as info:
        candidate.evaluate(candidate.Args(**overrides))
    return info.value


def test_official_seeds_constant():
    assert candidate.OFFICIAL_SEEDS == (0, 7, 42)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_official_profile_allows_official_seeds(seed):
    # 缺 run_identity 等参数会在 seed 判断之后抛错，只要不是 seed 闸门的错误即视为通过。
    error = _evaluate_error(profile="official", model_seed=seed)
    assert SEED_MSG not in str(error)


@pytest.mark.parametrize("seed", [1, -1])
def test_official_profile_rejects_other_seeds(seed):
    error = _evaluate_error(profile="official", model_seed=seed)
    assert isinstance(error, candidate.ContractError)
    assert SEED_MSG in str(error)


def test_testx_profile_still_rejects_seed_zero():
    error = _evaluate_error(profile="testx", model_seed=0)
    assert isinstance(error, candidate.ContractError)
    assert "model_seed" in str(error)
