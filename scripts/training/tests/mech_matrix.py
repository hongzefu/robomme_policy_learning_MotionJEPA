#!/usr/bin/env python3
"""八组合CPU机制矩阵：夹具启动契约与单层、小FFN真实融合前向。

保留完整608/896记忆、2048/1024宽度及所有记忆投影形状。视觉编码输入使用
确定性小夹具；不加载SigLIP/词表/预训练权重，不代表完整VLA前向或GPU短测。
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

# 必须在导入JAX之前设定；矩阵永远不使用GPU或下载。
os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests/training"))


def forward(history):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from omegaconf import OmegaConf
    import openpi.shared.array_typing as at
    from mme_vla_suite.datastore.motion_adapter import motion_policy_spec, split_stage2_tokens
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0, make_attn_mask
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    from mme_vla_suite.models.integration.history_gemma import HistoryBlock
    from mme_vla_suite.models.integration.utils import get_config
    from mme_vla_suite.models.representation.percep_mem import PerceptualMemory
    from mme_vla_suite.shared.sampling import memory_order

    assert all(device.platform == "cpu" for device in jax.devices())
    hc = OmegaConf.create(history)
    spec = motion_policy_spec(history["motion"])
    # 帧身份与四pool槽身份均不恒定，能暴露转置/重复/重排错误。
    frame_times = np.arange(512 // hc.token_per_image)
    raw = np.arange(96 * 1024, dtype=np.float32).reshape(96, 1, 1024) / np.float32(100000)
    motion = split_stage2_tokens(raw).reshape(384, 256) if spec.num_tokens == 4 else raw[:, 0]
    motion_times = np.repeat(np.arange(96) * 32, spec.num_tokens)
    order = memory_order(frame_times, hc.token_per_image, motion_times)
    np.testing.assert_array_equal(np.sort(order), np.arange(512 + spec.budget))
    obs = HistAugObservation(images={"front": jnp.zeros((1, 1, 1, 3), jnp.float32)},
        image_masks={"front": jnp.array([True])}, state=jnp.zeros((1, 32)),
        static_image_emb=jnp.broadcast_to(jnp.arange(512)[None, :, None] / 512, (1, 512, 2048)),
        static_pos_emb=jnp.broadcast_to(jnp.repeat(jnp.asarray(frame_times), hc.token_per_image)[None, :, None] / 32, (1, 512, 768)),
        static_state_emb=jnp.zeros((1, 512, 8)), static_mask=jnp.ones((1, 512), bool),
        motion_emb=jnp.asarray(motion)[None], motion_pos=jnp.broadcast_to(jnp.asarray(motion_times)[None, :, None] / 3072,
            (1, spec.budget, 256)), motion_mask=jnp.ones((1, spec.budget), bool), mem_order=jnp.asarray(order)[None])
    encoder = PerceptualMemory(hc, nnx.Rngs(42))
    assert encoder.motion_encoder_static.kernel.value.shape == (spec.dim + 768, hc.memory_token_dim)
    model = SimpleNamespace(mem_encoder=encoder, history_config=hc, integration_type=hc.integration_type,
        PaliGemma=SimpleNamespace(img=lambda image, train=False: (jnp.ones((1, 2, 2048), jnp.float32) * 0.01, None)))
    model.embed_memory = lambda observation, owner=model: HistoryPi0.embed_memory(owner, observation)
    with at.disable_typechecking():
        mem, mem_mask, _, _ = model.embed_memory(obs)
        prefix, prefix_mask, prefix_ar, prefix_na = HistoryPi0.embed_prefix(model, obs)
    assert mem.shape == (1, 512 + spec.budget, hc.memory_token_dim)
    assert prefix.shape[1] == (512 + spec.budget + 2 if hc.integration_type == "context" else 2)
    configs = tuple(dataclasses.replace(get_config(name), depth=1, mlp_dim=64) for name in ("gemma_2b", "gemma_300m"))
    block = HistoryBlock(configs=configs, integration_type=hc.integration_type)
    actions = jnp.ones((1, 2, 1024), jnp.float32) * 0.02
    mask = jnp.concatenate([prefix_mask, jnp.ones((1, 2), bool)], axis=1)
    attn = make_attn_mask(mask, jnp.concatenate([prefix_ar, jnp.array([True, False])]),
        jnp.concatenate([prefix_na, jnp.zeros((2,), bool)]))[:, None]
    positions = jnp.cumsum(mask, axis=1) - 1
    args = ([prefix, actions], None, positions, attn, [None, jnp.zeros((1, 1024))],
            [None, mem] if hc.integration_type == "modulation" else None,
            [None, mem_mask] if hc.integration_type == "modulation" else None)
    variables = block.init(jax.random.key(7), *args)
    outputs, _ = block.apply(variables, *args)
    jax.block_until_ready(outputs)
    assert outputs[0].shape == prefix.shape and outputs[1].shape == actions.shape
    assert all(np.isfinite(np.asarray(value)).all() for value in outputs)
    # 冷初始化的AdaRMS残差门为0；先验证原始前向，再以非零门参数夹具验证记忆确实连到动作。
    for key in ("pre_attention_norm_1", "pre_ffw_norm_1"):
        bias = variables["params"][key]["Dense_0"]["bias"]
        variables["params"][key]["Dense_0"]["bias"] = bias.at[2048:].set(1)
    probe, _ = block.apply(variables, *args)
    if hc.integration_type == "modulation":
        altered_args = (*args[:5], [None, mem + 0.1], args[6])
    else:
        changed = prefix.at[:, :mem.shape[1], :].add(0.1)
        altered_args = ([changed, actions], *args[1:])
    altered, _ = block.apply(variables, *altered_args)
    assert not np.array_equal(np.asarray(probe[1]), np.asarray(altered[1]))
    result = dict(tokens=mem.shape[1], width=mem.shape[2], motion_projection=list(encoder.motion_encoder_static.kernel.value.shape),
                  depth=1, ffn=64, memory_changes_actions=1, nonzero_gate_probe=1)
    del variables, outputs, altered, encoder, model
    gc.collect()
    jax.clear_caches()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", type=Path, help="仅记录调用者提供的库定位；本CPU矩阵始终只读隔离夹具")
    args = parser.parse_args(argv)
    print(f"MECH_SCOPE=single_layer_small_ffn requested_lib={args.lib} data=isolated_fixture")
    from test_check_orig16_stage2_launch import make_fixture, arguments, run_contract, launch
    # 临时目录不保留；即使给--lib也绝不读取真实库、encoder或基础权重。
    contract_ok = forward_ok = 0
    with tempfile.TemporaryDirectory(prefix="o16-mech-matrix-") as temporary:
        fx = make_fixture(Path(temporary))
        for name in launch.HISTORIES:
            record = run_contract(fx, name)
            if record.failed:
                raise ValueError(f"夹具DRY_RUN契约失败: {name} {record.failed}")
            contract_ok += 1
            _, _, _, history = arguments(fx, name)
            result = forward(history)
            forward_ok += 1
            print(f"MECH_FORWARD=PASS history={name} scope=single_layer_small_ffn result={result}", flush=True)
    print(f"MECH_MATRIX=PASS combos=8 contract_ok={contract_ok} forward_ok={forward_ok}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
