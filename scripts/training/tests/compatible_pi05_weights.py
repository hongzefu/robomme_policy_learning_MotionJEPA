"""仅供 V5 替身验证：按名称、形状加载 pi05_base，动作专家必须完整。"""

import numpy as np
from flax import traverse_util


def merge_compatible(reference, pretrained):
    ref = traverse_util.flatten_dict(reference, sep="/")
    source = traverse_util.flatten_dict(pretrained, sep="/")
    required = {name for name in ref if
                (name.startswith("PaliGemma/llm/") and any(p.endswith("_1") for p in name.split("/")))
                or name.split("/")[0] in {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}}
    # 名单独立于 checkpoint 推导，缺少任何门都不能被普通的缺参合并吞掉。
    gates = {f"PaliGemma/llm/{prefix}/Dense_0/{leaf}" for prefix in
             ("layers/pre_attention_norm_1", "layers/pre_ffw_norm_1", "final_norm_1")
             for leaf in ("bias", "kernel")}
    if not gates <= required:
        raise ValueError(f"替身动作专家缺少 AdaRMS 门: {sorted(gates-required)}")
    result, loaded, random = {}, [], []
    for name, value in ref.items():
        candidate = source.get(name)
        reason = "missing" if candidate is None else ("shape" if candidate.shape != value.shape else None)
        if reason:
            if name in required:
                raise ValueError(f"动作专家预训练参数不兼容: {name} ({reason})")
            result[name] = value
            random.append({"name": name, "reason": reason, "shape": list(value.shape),
                           "source_shape": None if candidate is None else list(candidate.shape)})
        else:
            result[name] = candidate.astype(value.dtype, copy=False)
            loaded.append(name)
    gate_maxima = {}
    for name in sorted(gates):
        value = np.asarray(result[name])
        if value.shape[-1] != 3072 or not np.isfinite(value).all():
            raise ValueError(f"AdaRMS 门宽度或数值非法: {name} {value.shape}")
        # 最后一维第三段是 gate；逐层检查，不能用其他层的非零掩盖零门。
        gate = value[..., 2048:]
        maxima = np.max(np.abs(gate.astype(np.float32)), axis=tuple(range(1, gate.ndim))) if gate.ndim > 1 else np.max(np.abs(gate))
        gate_maxima[name] = np.asarray(maxima).tolist()
    for prefix in ("layers/pre_attention_norm_1", "layers/pre_ffw_norm_1"):
        kernel = np.asarray(gate_maxima[f"PaliGemma/llm/{prefix}/Dense_0/kernel"])
        bias = np.asarray(gate_maxima[f"PaliGemma/llm/{prefix}/Dense_0/bias"])
        if kernel.shape != (18,) or bias.shape != (18,) or np.any((kernel == 0) & (bias == 0)):
            raise ValueError(f"18 层动作专家存在全零预训练门: {prefix}")
    report = {"loaded": sorted(loaded), "random": random, "required_action": sorted(required),
              "gate_max_abs": gate_maxima}
    return traverse_util.unflatten_dict(result, sep="/"), report
