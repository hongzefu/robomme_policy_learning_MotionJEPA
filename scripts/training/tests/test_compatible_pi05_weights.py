"""验证兼容加载不会吞掉动作专家缺参、错形状或零门。"""

import copy
import numpy as np
import pytest
from flax import traverse_util
from compatible_pi05_weights import merge_compatible


def fixture():
    params = {}
    for prefix in ("layers/pre_attention_norm_1", "layers/pre_ffw_norm_1", "final_norm_1"):
        batch = (18,) if prefix.startswith("layers") else ()
        for leaf, middle in (("bias", ()), ("kernel", (1,))):
            params[f"PaliGemma/llm/{prefix}/Dense_0/{leaf}"] = np.ones(batch + middle + (3072,), np.float32)
    params["PaliGemma/llm/embedder/input_embedding"] = np.ones((3, 2), np.float32)
    params["action_in_proj/kernel"] = np.ones((2, 2), np.float32)
    source = traverse_util.unflatten_dict(params, sep="/")
    ref = copy.deepcopy(params)
    ref["PaliGemma/llm/embedder/input_embedding"] = np.full((3, 1), 9, np.float32)
    ref["mem_encoder/motion_emb/kernel"] = np.full((2, 1), 7, np.float32)
    return traverse_util.unflatten_dict(ref, sep="/"), source


def test_compatible_load_keeps_only_explicit_random_parameters():
    ref, source = fixture()
    merged, report = merge_compatible(ref, source)
    assert {r["reason"] for r in report["random"]} == {"shape", "missing"}
    assert len(report["random"]) == 2
    assert np.all(merged["mem_encoder"]["motion_emb"]["kernel"] == 7)
    assert np.all(merged["PaliGemma"]["llm"]["embedder"]["input_embedding"] == 9)
    assert len(report["required_action"]) == 7


@pytest.mark.parametrize("damage", ["missing", "shape", "zero_gate"])
def test_compatible_load_rejects_broken_action_expert(damage):
    ref, source = fixture()
    gate = source["PaliGemma"]["llm"]["layers"]["pre_ffw_norm_1"]["Dense_0"]
    if damage == "missing":
        gate.pop("kernel")
    elif damage == "shape":
        gate["kernel"] = gate["kernel"][:1]
    else:
        gate["kernel"][7, ..., 2048:] = 0
        gate["bias"][7, ..., 2048:] = 0
    with pytest.raises(ValueError):
        merge_compatible(ref, source)
