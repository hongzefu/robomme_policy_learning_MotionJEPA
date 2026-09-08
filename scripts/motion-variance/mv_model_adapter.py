#!/usr/bin/env python3
"""motion-variance 模型侧手术：复刻 `HistoryPi0.sample_actions` 加 attention 门控、18 层手工展开、固定-loss 梯度。

三块能力（计划第二部分 §3）：
  (a) `f_sample`：逐字复刻 sample_actions 的 context 分支（prefill → while_loop 10 步），在两处 attention mask
      上插 key 列门控（`apply_gate=True`）；`apply_gate=False` 为无算子的逐字复刻（只用于验收 `MV_NOOP_BITEXACT`）。
      四个条件（normal / mask / swap 与开环三臂）全走 `apply_gate=True` 这一份编译产物。
  (b) `f_sample_layered`：把 `nn.scan` 的 18 层手工展开（prefill 与去噪 10 步都展开），逐层给 attention 门控、
      逐层抓 attention 概率（monkeypatch `history_gemma.Attention`）、可选逐层 KV 替换 / V 置零。
  (c) `f_layer_grads` / `f_input_grads`：固定 (obs, x_t, t, u_t) 的 loss 对第 l 层入口隐状态 / 对 motion_emb·motion_pos 的梯度。

**两条硬纪律**：
  1. 屏蔽只改 attention mask 的 key 列，绝不动 `motion_mask` / `input_mask` / `mem_order`——
     `positions = cumsum(input_mask) - 1`，改 mask 会把其后所有 token（含动作）的 RoPE 位置整体前移。
  2. 源码护栏 `_guard_model_source` 在构造时对生产代码做指纹核对（空白规范化后匹配），漂移即拒跑。

用法（自检，≤5 min，1 张 GPU）：
  CUDA_VISIBLE_DEVICES=0 XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
    UV_LINK_MODE=copy uv run --no-sync python scripts/motion-variance/mv_model_adapter.py --selftest --max-points 1 [--unroll] [--grads] [--dump-param-tree]
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import json
import os
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import mv_common as C  # noqa: E402

os.environ.setdefault("OPENPI_DATA_HOME", str(C.V1 / "models"))

_CAPTURE: list = []          # 被 patch 的 Attention 把 f32 softmax 概率追加到这里（trace 期内消费）


# ══════════════════════════════════════════ 护栏 ══════════════════════════════════════════
def _guard_model_source() -> None:
    from mme_vla_suite.models.integration import history_gemma as hg
    from mme_vla_suite.models.integration import history_pi0 as hp
    from openpi.models import gemma
    C.guard_source(hp.HistoryPi0.sample_actions, (
        "observation = preprocess_observation(None, observation, train=False)",
        "dt = -1.0 / num_steps",
        "prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask, prefix_na_mask)",
        "positions = jnp.cumsum(prefix_mask, axis=1) - 1",
        "[prefix_tokens, None], mask=prefix_attn_mask, positions=positions",
        "prefix_attn_mask = einops.repeat(",
        "full_attn_mask = jnp.concatenate(",
        "jnp.sum(prefix_mask, axis=-1)[:, None]",
        "kv_cache=kv_cache,",
        "adarms_cond=[None, adarms_cond],",
        "v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])",
        "return time >= -dt / 2",
        "x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))"), "HistoryPi0.sample_actions")
    C.guard_source(hp.HistoryPi0.embed_memory, (
        "input_mask = jnp.concatenate([obs.static_mask, obs.motion_mask], axis=1)",
        "tokens = jnp.take_along_axis(tokens, obs.mem_order[:, :, None], axis=1)",
        "input_mask = jnp.take_along_axis(input_mask, obs.mem_order, axis=1)"), "HistoryPi0.embed_memory")
    C.guard_source(hp.HistoryPi0.compute_loss, (
        "x_t = time_expanded * noise + (1 - time_expanded) * actions",
        "u_t = noise - actions",
        "positions = jnp.cumsum(input_mask, axis=1) - 1",
        "return jnp.mean(jnp.square(v_t - u_t), axis=-1)"), "HistoryPi0.compute_loss")
    C.guard_source(hp.make_attn_mask, (
        "cumsum = jnp.cumsum(mask_ar, axis=1)",
        "attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]",
        "valid_mask = input_mask[:, None, :] * input_mask[:, :, None]",
        "mask_not_attend"), "make_attn_mask")
    C.guard_source(gemma.Attention.__call__, (
        'q = _apply_rope(q, positions=positions)',
        'q *= self.configs[0].head_dim ** -0.5',
        'k = _apply_rope(k, positions=positions)',
        'k = jnp.concatenate([cache_k, k], axis=1)',
        'logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)',
        'big_neg = -2.3819763e38',
        'masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)',
        'probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)',
        'encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)',
        'return out, (k, v)'), "gemma.Attention.__call__")
    C.guard_source(hg.HistoryBlock.__call__, (
        'attn = Attention(configs=self.configs, name="attn")',
        'post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)',
        '_gated_residual(x, y, gate)',
        'name=_name("pre_ffw_norm", i)'), "history_gemma.HistoryBlock.__call__")
    C.guard_source(hg.Module.setup, (
        'variable_axes={"params": 0}',
        'length=self.configs[0].depth'), "history_gemma.Module.setup")
    C.guard_source(hg.Module.__call__, (
        "embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)",
        "mask = jnp.asarray(mask)[:, None, :, :]",
        "f(e, a)[0] if e is not None else e"), "history_gemma.Module.__call__")


# ═════════════════════════════ 被 patch 的 Attention（父类 __call__ 全文 + 一行追加） ═════════════════════════════
def _make_patched_attention():
    import einops
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import openpi.models.lora as lora
    from openpi.models import gemma
    from openpi.models.gemma import _apply_rope, _name

    class _PatchedAttention(gemma.Attention):
        """与 gemma.Attention.__call__ 逐字同式，只把 softmax 拆成 f32 概率 + astype 两句，并把 f32 概率追加到 _CAPTURE。"""

        @nn.compact
        def __call__(self, xs, positions, attn_mask, kv_cache):
            assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
            assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
            assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)
            dtype = next(x.dtype for x in xs if x is not None)
            qkvs = []
            for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
                if x is None:
                    continue
                if config.num_kv_heads == config.num_heads:
                    qkv_einsum = lora.Einsum(
                        shape=(3, config.num_heads, config.width, config.head_dim),
                        name=_name("qkv_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                        lora_config=config.lora_configs.get("attn"))
                    qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
                else:
                    q_einsum = lora.Einsum(
                        shape=(config.num_heads, config.width, config.head_dim),
                        name=_name("q_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                        lora_config=config.lora_configs.get("attn"))
                    q = q_einsum("BTD,NDH->BTNH", x)
                    kv_einsum = lora.Einsum(
                        shape=(2, config.num_kv_heads, config.width, config.head_dim),
                        name=_name("kv_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                        lora_config=config.lora_configs.get("attn"))
                    k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                    qkvs.append((q, k, v))
            q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))
            q = _apply_rope(q, positions=positions)
            q *= self.configs[0].head_dim ** -0.5
            k = _apply_rope(k, positions=positions)
            assert q.dtype == k.dtype == v.dtype == dtype
            if kv_cache is not None:
                cache_k, cache_v = kv_cache
                k = jnp.concatenate([cache_k, k], axis=1)
                v = jnp.concatenate([cache_v, v], axis=1)
            q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
            logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
            if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
                raise ValueError(f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}")
            big_neg = -2.3819763e38
            masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
            probs_f32 = jax.nn.softmax(masked_logits, axis=-1)
            _CAPTURE.append(probs_f32)                         # ← 唯一的追加行
            probs = probs_f32.astype(dtype)
            encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
            encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")
            out = []
            start = 0
            for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
                if x is not None:
                    end = start + x.shape[1]
                    out_einsum = lora.Einsum(
                        shape=(config.num_heads, config.head_dim, config.width),
                        name=_name("attn_vec_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                        lora_config=config.lora_configs.get("attn"))
                    out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                    start = end
                else:
                    out.append(None)
            return out, (k, v)

    return _PatchedAttention


@contextlib.contextmanager
def _capture_attn(hg_module, patched_cls):
    from unittest import mock
    with mock.patch.object(hg_module, "Attention", patched_cls):
        _CAPTURE.clear()
        yield _CAPTURE


# ══════════════════════════════════════════ policy 构造 ══════════════════════════════════════════
class ZeroMotion:
    """与 MotionEncoderClient 同形的零向量编码句柄：可调用 (frames, start_frame) -> (768,) f32，有 close()。"""

    def __init__(self):
        self.calls = 0

    def __call__(self, frames, start_frame: int = 0):
        self.calls += 1
        return np.zeros(C.MOTION_DIM, np.float32)

    def close(self) -> None:
        pass


def build_policy(train_config, ckpt_dir, *, seed: int, motion_factory):
    """create_trained_policy 的包装：把 `motion_client.MotionEncoderClient` 换成工厂，构造期**不起** stub 子进程。

    `motion_factory(**kw)` 收到 create_trained_policy 传给 MotionEncoderClient 的全部关键字
    （online_gpu / stub / store_provenance / expected_ckpt_sha256），返回任意可调用编码句柄（需有 close()）。
    """
    from unittest import mock
    import mme_vla_suite.policies.motion_client as mc_mod
    from mme_vla_suite.policies import policy_config as _policy_config
    with mock.patch.object(mc_mod, "MotionEncoderClient", motion_factory):
        return _policy_config.create_trained_policy(train_config, ckpt_dir, seed=seed, motion_stub=False)


# ══════════════════════════════════════════ 适配器 ══════════════════════════════════════════
class MVAdapter:
    def __init__(self, policy, *, unroll: bool = False):
        import einops
        import jax
        import jax.numpy as jnp
        import flax.nnx as nnx
        from mme_vla_suite.models.integration import history_gemma as hg
        from mme_vla_suite.models.integration.history_observation import preprocess_observation
        from mme_vla_suite.models.integration.history_pi0 import make_attn_mask
        from mme_vla_suite.models.integration.utils import get_config
        from openpi.models import gemma

        _guard_model_source()
        self.jax, self.jnp, self.nnx, self.einops = jax, jnp, nnx, einops
        self.hg, self.gemma = hg, gemma
        self.policy = policy
        self.model = policy._model
        self.gdef, self.gstate = nnx.split(self.model)
        self.AH, self.AD = int(self.model.action_horizon), int(self.model.action_dim)
        self.embed_dtype = self.model.config.dtype
        self.cfgs = (get_config(self.model.config.paligemma_variant), get_config(self.model.config.action_expert_variant))
        self.P = C.PREFIX_LEN
        self._is_motion_par = jnp.asarray(C.IS_MOTION_PAR)
        self._preprocess = preprocess_observation
        self._make_attn_mask = make_attn_mask
        self._build_sample()
        self.unrolled = False
        if unroll:
            self._build_unrolled()

    # ── motion 列（与 embed_memory 的 take_along_axis 同向）────────────────────────────────
    def motion_cols(self, mem_order):
        jnp = self.jnp
        par = jnp.broadcast_to(self._is_motion_par, mem_order.shape)
        m = jnp.take_along_axis(par, mem_order, axis=1)
        return jnp.concatenate([m, jnp.zeros((mem_order.shape[0], self.P - C.MEM_LEN), bool)], axis=1)

    def gates(self, obs, kind: str):
        """kind ∈ {none, mask}：返回 (gate_prefill, gate_step)，bool[b, P]，True = 该列可被读。"""
        jnp = self.jnp
        b = obs.state.shape[0]
        if kind == "none":
            g = jnp.ones((b, self.P), bool)
            return g, g
        if kind == "mask":
            g = ~self.motion_cols(obs.mem_order)
            return g, g
        raise ValueError(kind)

    # ── (a) 复刻 sample_actions ──────────────────────────────────────────────────────────
    def _build_sample(self):
        jax, jnp, nnx, einops = self.jax, self.jnp, self.nnx, self.einops
        gdef = self.gdef
        preprocess_observation, make_attn_mask = self._preprocess, self._make_attn_mask

        @functools.partial(jax.jit, static_argnames=("num_steps", "apply_gate"))
        def f_sample(state, rng, obs, gate_prefill, gate_step, noise, num_steps, apply_gate):
            """照抄 HistoryPi0.sample_actions（context 分支）；apply_gate=True 时在两处 mask 插 key 列门控。"""
            m = nnx.merge(gdef, state)
            observation = preprocess_observation(None, obs, train=False)
            dt = -1.0 / num_steps
            batch_size = observation.state.shape[0]
            if noise is None:
                noise = jax.random.normal(rng, (batch_size, m.action_horizon, m.action_dim))
            prefix_tokens, prefix_mask, prefix_ar_mask, prefix_na_mask = m.embed_prefix(observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask, prefix_na_mask)
            if apply_gate:
                # 干预点 1（prefill）：非 motion 的 query 行 × 被门控的 key 列 → False；motion 行读自己照旧
                mc = ~gate_prefill
                prefix_attn_mask = prefix_attn_mask & ~(mc[:, None, :] & ~mc[:, :, None])
            positions = jnp.cumsum(prefix_mask, axis=1) - 1               # 用原 prefix_mask，绝不用 gate
            _, kv_cache = m.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

            def step(carry):
                x_t, time = carry
                suffix_tokens, suffix_mask, suffix_ar_mask, _, adarms_cond = m.embed_suffix(
                    observation, x_t, jnp.broadcast_to(time, batch_size))
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn_mask_s = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                if apply_gate:
                    prefix_attn_mask_s = prefix_attn_mask_s & gate_step[:, None, :]   # 干预点 2（去噪）
                full_attn_mask = jnp.concatenate([prefix_attn_mask_s, suffix_attn_mask], axis=-1)
                positions_s = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
                (prefix_out, suffix_out), _ = m.PaliGemma.llm(
                    [None, suffix_tokens], mask=full_attn_mask, positions=positions_s,
                    kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
                assert prefix_out is None
                v_t = m.action_out_proj(suffix_out[:, -m.action_horizon:])
                return x_t + dt * v_t, time + dt

            def cond(carry):
                x_t, time = carry
                return time >= -dt / 2

            x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
            return x_0

        @jax.jit
        def f_prefix(state, obs):
            """prefill 一次：返回 prefix_mask / 三参 attn mask / positions / KV（与 g0/compare_train_infer_obs.py::f_prefix 同式）。"""
            m = nnx.merge(gdef, state)
            o = preprocess_observation(None, obs, train=False)
            tok, mask, ar, na = m.embed_prefix(o)
            attn = make_attn_mask(mask, ar, na)
            pos = jnp.cumsum(mask, axis=1) - 1
            _, kv = m.PaliGemma.llm([tok, None], mask=attn, positions=pos)
            return {"mask": mask, "attn": attn, "pos": pos, "k": kv[0], "v": kv[1]}

        self.f_sample = f_sample
        self.f_prefix = f_prefix

    def make_sample_fns(self) -> dict:
        """返回可直接顶掉 policy._sample_actions 的可调用（签名 (rng, observation, **kw)，kw 支持 noise / num_steps）。"""
        gs = self.gstate
        f = self.f_sample

        def _mk(kind: str, apply_gate: bool):
            def fn(rng, observation, **kw):
                gp, gsp = self.gates(observation, kind)
                return f(gs, rng, observation, gp, gsp, kw.get("noise"), int(kw.get("num_steps", 10)), apply_gate)
            fn.__name__ = f"f_sample_{kind}{'' if apply_gate else '_ref'}"
            return fn

        return {"none": _mk("none", True), "mask": _mk("mask", True), "ref": _mk("none", False)}

    # ── (b) 18 层手工展开 ─────────────────────────────────────────────────────────────────
    def _llm_pure(self) -> dict:
        pure = self.gstate.to_pure_dict()
        llm = pure["PaliGemma"]["llm"]
        for k in ("layers", "final_norm", "final_norm_1"):
            if k not in llm:
                raise SystemExit(f"错误: llm 参数树缺 {k!r}，现有键 {sorted(llm)}；先用 --dump-param-tree 钉键名")
        return llm

    def dump_param_tree(self) -> str:
        import jax
        llm = self._llm_pure()
        flat = jax.tree_util.tree_flatten_with_path(llm["layers"])[0]
        n_ax0 = sum(1 for _, v in flat if np.asarray(v).shape[:1] == (C.N_LAYERS,))
        lines = [f"PARAM_TREE layers={C.N_LAYERS} leaves={len(flat)} layer_axis0={n_ax0}/{len(flat)} "
                 f"llm_keys={sorted(llm)} final_norm_keys={sorted(llm['final_norm'])}/{sorted(llm['final_norm_1'])}"]
        for path, v in flat:
            lines.append(f"  {jax.tree_util.keystr(path)} {tuple(np.asarray(v).shape)} {np.asarray(v).dtype}")
        return "\n".join(lines)

    def _build_unrolled(self):
        jax, jnp, nnx, einops = self.jax, self.jnp, self.nnx, self.einops
        hg, gemma = self.hg, self.gemma
        gdef = self.gdef
        preprocess_observation, make_attn_mask = self._preprocess, self._make_attn_mask
        self._llm_pure()                                   # 只做键名校验；参数树在 jit 内从入参 state 取（闭包常量会把 2.6B 参数嵌进每个编译产物）
        cfgs = self.cfgs
        ed = self.embed_dtype
        P, L, AH = self.P, C.N_LAYERS, self.AH
        patched = _make_patched_attention()
        block = hg.HistoryBlock(configs=cfgs, integration_type="context")
        final_norm_1 = gemma.RMSNorm()

        def llm_of(state):
            return state.to_pure_dict()["PaliGemma"]["llm"]

        def block_apply(layers, l, xs, kv, positions, mask3d, adarms):
            p_l = jax.tree.map(lambda x: x[l], layers)
            with _capture_attn(hg, patched) as cap:
                out, kv_out = block.apply({"params": p_l}, xs, kv, positions, mask3d[:, None, :, :],
                                          adarms, None, None, True)
                probs = cap[0]
            return out, kv_out, probs

        def share_stats(probs, mc, row_sel, key_mask):
            """probs (b,1,G,T,S) f32；mc bool[b,S] motion 列；row_sel bool[b,T] 参与统计的 query 行；
            key_mask bool[b,T,S] 该 query 的合法 key（未门控的结构 mask）。返回 (share (b,G), U (b,))，
            share = 该行 motion 列概率质量之和在 row_sel 上取均值；U = |合法 motion key| / |合法 key| 在 row_sel 上取均值。"""
            p = probs[:, 0]                                                  # (b,G,T,S)
            share_row = jnp.sum(p * mc[:, None, None, :], axis=-1)           # (b,G,T)
            n_sel = jnp.maximum(jnp.sum(row_sel, axis=-1), 1)                # (b,)
            share = jnp.sum(share_row * row_sel[:, None, :], axis=-1) / n_sel[:, None]
            legal = key_mask.astype(jnp.float32)
            u_row = jnp.sum(legal * mc[:, None, :], axis=-1) / jnp.maximum(jnp.sum(legal, axis=-1), 1.0)   # (b,T)
            U = jnp.sum(u_row * row_sel, axis=-1) / n_sel
            return share, U

        @functools.partial(jax.jit, static_argnames=("num_steps",))
        def f_sample_layered(state, obs, gate_prefill_layers, gate_step_layers, noise, num_steps,
                             kv_override, override_layers, v_zero_layers):
            """展开版 sample_actions：gate_*_layers bool[L,b,P]；kv_override=(k,v) 或 None；override_layers / v_zero_layers bool[L]。
            返回 x_0、prefill 与去噪的 motion 份额统计、KV。"""
            m = nnx.merge(gdef, state)
            llm = llm_of(state)
            layers, fn1 = llm["layers"], llm["final_norm_1"]
            o = preprocess_observation(None, obs, train=False)
            dt = -1.0 / num_steps
            b = o.state.shape[0]
            pt, pm, par_, pna = m.embed_prefix(o)
            base = make_attn_mask(pm, par_, pna)                             # (b,P,P) 结构 mask
            pos = jnp.cumsum(pm, axis=1) - 1
            mc = self.motion_cols(o.mem_order)                               # (b,P)
            # query 行分组（prefill）：frame = 记忆段非 motion 且有效；txt = 文本段有效；padding 行剔除
            idx = jnp.arange(P)[None, :]
            row_frame = pm & (idx < C.MEM_LEN) & ~mc
            row_txt = pm & (idx >= C.MEM_LEN + C.IMG_LEN)
            h = pt.astype(ed)
            ks, vs, sh_f, sh_t, U_f, U_t = [], [], [], [], [], []
            for l in range(L):
                mcl = ~gate_prefill_layers[l]
                mask_l = base & ~(mcl[:, None, :] & ~mcl[:, :, None])
                (h, _), (k, v), probs = block_apply(layers, l, [h, None], None, pos, mask_l, [None, None])
                s_f, u_f = share_stats(probs, mc, row_frame, base)
                s_t, u_t = share_stats(probs, mc, row_txt, base)
                ks.append(k); vs.append(v); sh_f.append(s_f); sh_t.append(s_t); U_f.append(u_f); U_t.append(u_t)
            K = jnp.stack(ks); Vv = jnp.stack(vs)                            # (L,b,P,1,H)
            if kv_override is not None:
                sel = (override_layers[:, None, None, None, None] & mc[None, :, :, None, None])
                K = jnp.where(sel, kv_override[0], K)
                Vv = jnp.where(sel, kv_override[1], Vv)
            zsel = (v_zero_layers[:, None, None, None, None] & mc[None, :, :, None, None])
            Vv = jnp.where(zsel, jnp.zeros_like(Vv), Vv)
            if noise is None:
                noise = jax.random.normal(jax.random.key(0), (b, AH, m.action_dim))
            spos_base = jnp.sum(pm, axis=-1)[:, None]
            # 去噪 query 的合法 key（结构）：prefix 的 pm 列 + suffix 20 列
            n_valid = jnp.sum(pm, axis=-1).astype(jnp.float32) + AH
            U_step = jnp.sum((pm & mc), axis=-1).astype(jnp.float32) / n_valid              # (b,)

            def step(carry):
                x_t, time, acc = carry
                st, sm, sar, _, ad = m.embed_suffix(o, x_t, jnp.broadcast_to(time, b))
                sattn = make_attn_mask(sm, sar)
                spos = spos_base + jnp.cumsum(sm, axis=-1) - 1
                hs = st.astype(ed)
                shares = []
                for l in range(L):
                    prefix_attn = einops.repeat(pm, "b p -> b s p", s=st.shape[1]) & gate_step_layers[l][:, None, :]
                    full = jnp.concatenate([prefix_attn, sattn], axis=-1)
                    (_, hs), _, probs = block_apply(layers, l, [None, hs], (K[l], Vv[l]), spos, full, [None, ad])
                    p = probs[:, 0]                                                          # (b,G,T,S)
                    shares.append(jnp.sum(p[..., :P] * mc[:, None, None, :], axis=-1))       # (b,G,T)
                hs, _ = final_norm_1.apply({"params": fn1}, hs, ad)
                v_t = m.action_out_proj(hs[:, -AH:])
                return x_t + dt * v_t, time + dt, acc + jnp.stack(shares)

            def cond(carry):
                return carry[1] >= -dt / 2

            acc0 = jnp.zeros((L, b, cfgs[0].num_heads, AH), jnp.float32)
            x_0, _, acc = jax.lax.while_loop(cond, step, (noise, 1.0, acc0))
            return {"x_0": x_0, "k": K, "v": Vv,
                    "share_step": acc / num_steps, "U_step": U_step,
                    "share_prefill_frame": jnp.stack(sh_f), "U_prefill_frame": jnp.stack(U_f),
                    "share_prefill_txt": jnp.stack(sh_t), "U_prefill_txt": jnp.stack(U_t),
                    "k_valid": jnp.sum(pm & mc, axis=-1)}

        @jax.jit
        def f_attn_uniform_check(state, obs):
            """把「概率」换成合法 mask 的均匀分布，走同一套 share 归约，结果必须等于 U（验证归约与列索引口径）。"""
            m = nnx.merge(gdef, state)
            o = preprocess_observation(None, obs, train=False)
            pt, pm, par_, pna = m.embed_prefix(o)
            base = make_attn_mask(pm, par_, pna)
            mc = self.motion_cols(o.mem_order)
            idx = jnp.arange(P)[None, :]
            row_frame = pm & (idx < C.MEM_LEN) & ~mc
            row_txt = pm & (idx >= C.MEM_LEN + C.IMG_LEN)
            legal = base.astype(jnp.float32)
            uni = legal / jnp.maximum(jnp.sum(legal, axis=-1, keepdims=True), 1.0)      # (b,P,P)
            probs = jnp.broadcast_to(uni[:, None, None], (b_ := pm.shape[0], 1, cfgs[0].num_heads, P, P))
            s_f, u_f = share_stats(probs, mc, row_frame, base)
            s_t, u_t = share_stats(probs, mc, row_txt, base)
            # 去噪侧：合法 key = pm 列 + 20 suffix 列，均匀分布下 share 应 = U_step
            n_valid = jnp.sum(pm, axis=-1).astype(jnp.float32) + AH
            U_step = jnp.sum(pm & mc, axis=-1).astype(jnp.float32) / n_valid
            share_step = jnp.sum(pm & mc, axis=-1).astype(jnp.float32) / n_valid
            return {"share_f": s_f, "U_f": u_f, "share_t": s_t, "U_t": u_t, "share_step": share_step, "U_step": U_step}

        self.f_sample_layered = f_sample_layered
        self.f_attn_uniform_check = f_attn_uniform_check
        self.unrolled = True

    def layered_gates(self, obs, prefill_kind: str = "none", step_kind: str = "none"):
        """所有层同一种门控：kind ∈ {none, mask}。返回 (gate_prefill_layers, gate_step_layers) bool[L,b,P]。"""
        jnp = self.jnp
        gp, _ = self.gates(obs, prefill_kind)
        _, gs = self.gates(obs, step_kind)
        L = C.N_LAYERS
        return jnp.broadcast_to(gp[None], (L,) + gp.shape), jnp.broadcast_to(gs[None], (L,) + gs.shape)

    def run_layered(self, obs, *, gate_prefill_layers, gate_step_layers, noise, num_steps=10,
                    kv_override=None, override_layers=None, v_zero_layers=None):
        jnp = self.jnp
        L = C.N_LAYERS
        if override_layers is None:
            override_layers = jnp.zeros((L,), bool)
        if v_zero_layers is None:
            v_zero_layers = jnp.zeros((L,), bool)
        return self.f_sample_layered(self.gstate, obs, gate_prefill_layers, gate_step_layers, noise, num_steps,
                                     kv_override, override_layers, v_zero_layers)

    # ── (c) 固定-loss 梯度 ────────────────────────────────────────────────────────────────
    def _build_grads(self):
        jax, jnp, nnx = self.jax, self.jnp, self.nnx
        hg, gemma = self.hg, self.gemma
        gdef = self.gdef
        preprocess_observation, make_attn_mask = self._preprocess, self._make_attn_mask
        AH = self.AH
        ed = self.embed_dtype

        def loss_fixed_prod(m, obs, x_t, tstep, u_t):
            """固定 (x_t, t, u_t) 的 flow-matching loss，走生产 scan 路径（compute_loss 主体，train=False，无增广无随机）。"""
            o = preprocess_observation(None, obs, train=False)
            pt, pm, par_, pna = m.embed_prefix(o)
            st, sm, sar, sna, ad = m.embed_suffix(o, x_t, tstep)
            input_mask = jnp.concatenate([pm, sm], axis=1)
            ar_mask = jnp.concatenate([par_, sar], axis=0)
            na_mask = jnp.concatenate([pna, sna], axis=0)
            attn = make_attn_mask(input_mask, ar_mask, na_mask)
            pos = jnp.cumsum(input_mask, axis=1) - 1
            (_, so), _ = m.PaliGemma.llm([pt, st], mask=attn, positions=pos, adarms_cond=[None, ad])
            v_t = m.action_out_proj(so[:, -AH:])
            return jnp.mean(jnp.square(v_t - u_t))

        @jax.jit
        def f_loss_fixed(state, obs, x_t, tstep, u_t):
            return loss_fixed_prod(nnx.merge(gdef, state), obs, x_t, tstep, u_t)

        @jax.jit
        def f_input_grads(state, obs, x_t, tstep, u_t):
            m = nnx.merge(gdef, state)

            def f(me, mp):
                return loss_fixed_prod(m, dataclasses.replace(obs, motion_emb=me, motion_pos=mp), x_t, tstep, u_t)

            (g_me, g_mp) = jax.grad(f, argnums=(0, 1))(obs.motion_emb, obs.motion_pos)
            return {"g_motion_emb": g_me, "g_motion_pos": g_mp}

        self._llm_pure()
        block = hg.HistoryBlock(configs=self.cfgs, integration_type="context")
        final_norm_1 = gemma.RMSNorm()
        L, P = C.N_LAYERS, self.P

        @jax.jit
        def f_layer_grads(state, obs, x_t, tstep, u_t):
            """整段（prefix+suffix 一次前向）展开 18 层；第 l 层入口 h0 += deltas[l]（恒零），jax.grad 一次出 18 层。
            返回 grad 的逐列 L2 范数 (L,b,P) 与 loss。"""
            m = nnx.merge(gdef, state)
            llm = state.to_pure_dict()["PaliGemma"]["llm"]
            layers, fn1 = llm["layers"], llm["final_norm_1"]
            o = preprocess_observation(None, obs, train=False)
            pt, pm, par_, pna = m.embed_prefix(o)
            st, sm, sar, sna, ad = m.embed_suffix(o, x_t, tstep)
            input_mask = jnp.concatenate([pm, sm], axis=1)
            attn = make_attn_mask(input_mask, jnp.concatenate([par_, sar], axis=0), jnp.concatenate([pna, sna], axis=0))
            pos = jnp.cumsum(input_mask, axis=1) - 1
            b = pm.shape[0]

            def loss_of(deltas):
                h0 = pt.astype(ed)
                h1 = st.astype(ed)
                for l in range(L):
                    h0 = (h0.astype(jnp.float32) + deltas[l]).astype(ed)
                    p_l = jax.tree.map(lambda x: x[l], layers)

                    def one(h0_, h1_, p_l=p_l):
                        (a, c), _ = block.apply({"params": p_l}, [h0_, h1_], None, pos, attn[:, None, :, :],
                                                [None, ad], None, None, True)
                        return a, c

                    h0, h1 = jax.checkpoint(one)(h0, h1)
                h1, _ = final_norm_1.apply({"params": fn1}, h1, ad)
                v_t = m.action_out_proj(h1[:, -AH:])
                return jnp.mean(jnp.square(v_t - u_t))

            d0 = jnp.zeros((L, b, P, self.cfgs[0].width), jnp.float32)
            loss, g = jax.value_and_grad(loss_of)(d0)
            return {"loss": loss, "col_norm": jnp.linalg.norm(g, axis=-1), "mc": self.motion_cols(o.mem_order), "pm": pm}

        self.f_loss_fixed = f_loss_fixed
        self.f_input_grads = f_input_grads
        self.f_layer_grads = f_layer_grads

    def build_grads(self):
        self._build_grads()
        return self


# ══════════════════════════════════════════ 自检 ══════════════════════════════════════════
def _obs_with(obs, **kw):
    return dataclasses.replace(obs, **kw)


def selftest(args) -> int:
    import jax
    import jax.numpy as jnp
    from mv_points import PointSource

    C.require_gpu()
    C.require_det_flags()
    V = C.Verdict()
    src = PointSource(lib=pathlib.Path(args.lib), ckpt_dir=pathlib.Path(args.ckpt), train_config_name=args.train_config,
                      episodes_spec=args.episodes, max_points=args.max_points)
    ad = MVAdapter(src.policy, unroll=args.unroll)
    if args.dump_param_tree:
        print(ad.dump_param_tree())
    fns = ad.make_sample_fns()
    seeds = [int(s) for s in args.noise_seeds.split(",")]
    NOISE = {s: jax.random.normal(jax.random.key(s), (1, ad.AH, ad.AD), dtype=jnp.float32) for s in seeds}
    key0 = jax.random.key(0)
    rng = np.random.default_rng(0)

    n_pts = 0
    ref_prod = mv_ref = 0
    pad_ok = mask_null = 0
    n_pad = n_null = 0
    unroll_x0 = unroll_kv = self_x0 = 0
    n_unroll = 0
    attn_ok = 0
    n_attn = 0
    grad_finite = grad_pad0 = 0
    n_grad = 0
    numeric = C.DiffAcc()
    t0 = time.perf_counter()
    for pt in src.iter_points(max_total=args.max_total_points):
        obs = pt["obs"]
        n_pts += 1
        k = int(pt["k"])
        for s in seeds:
            a_prod = np.asarray(jax.block_until_ready(src.policy._sample_actions(key0, obs, noise=NOISE[s])))
            a_ref = np.asarray(jax.block_until_ready(fns["ref"](key0, obs, noise=NOISE[s])))
            a_mv = np.asarray(jax.block_until_ready(fns["none"](key0, obs, noise=NOISE[s])))
            ref_prod += int(C.bytes_equal(a_prod, a_ref))
            mv_ref += int(C.bytes_equal(a_mv, a_ref))
            numeric.add(a_ref, a_mv)
            # MV_PAD_INVARIANT：padding 槽灌垃圾
            me = np.asarray(obs.motion_emb); mp = np.asarray(obs.motion_pos); mm = np.asarray(obs.motion_mask)[0]
            for variant in ("zero", "gauss10", "huge"):
                me2, mp2 = me.copy(), mp.copy()
                if variant == "zero":
                    me2[0, ~mm] = 0; mp2[0, ~mm] = 0
                elif variant == "gauss10":
                    me2[0, ~mm] = rng.normal(0, 10, me2[0, ~mm].shape); mp2[0, ~mm] = rng.normal(0, 10, mp2[0, ~mm].shape)
                else:
                    me2[0, ~mm] = 1e4; mp2[0, ~mm] = -1e4
                o2 = _obs_with(obs, motion_emb=jnp.asarray(me2, jnp.float32), motion_pos=jnp.asarray(mp2, jnp.float32))
                a2 = np.asarray(jax.block_until_ready(fns["none"](key0, o2, noise=NOISE[s])))
                n_pad += 1; pad_ok += int(C.bytes_equal(a2, a_mv))
            # MV_MASK_CONTENT_NULL：mask_all 下换有效槽内容
            a_mask = np.asarray(jax.block_until_ready(fns["mask"](key0, obs, noise=NOISE[s])))
            for variant in ("zero", "gauss", "gauss2"):
                me2 = me.copy()
                if variant == "zero":
                    me2[0, mm] = 0
                else:
                    me2[0, mm] = rng.normal(0, 1, me2[0, mm].shape)
                o2 = _obs_with(obs, motion_emb=jnp.asarray(me2, jnp.float32))
                a2 = np.asarray(jax.block_until_ready(fns["mask"](key0, o2, noise=NOISE[s])))
                n_null += 1; mask_null += int(C.bytes_equal(a2, a_mask))
            if args.unroll and s == seeds[0]:
                gp, gs = ad.layered_gates(obs, "none", "none")
                out = jax.block_until_ready(ad.run_layered(obs, gate_prefill_layers=gp, gate_step_layers=gs, noise=NOISE[s]))
                pre = jax.block_until_ready(ad.f_prefix(ad.gstate, obs))
                n_unroll += 1
                unroll_x0 += int(C.bytes_equal(np.asarray(out["x_0"]), a_mv))
                pv = np.asarray(pre["mask"][0], bool)
                unroll_kv += int(C.bytes_equal(np.asarray(out["k"])[:, :, pv], np.asarray(pre["k"])[:, :, pv])
                                 and C.bytes_equal(np.asarray(out["v"])[:, :, pv], np.asarray(pre["v"])[:, :, pv]))
                gp2, gs2 = ad.layered_gates(obs, "mask", "mask")
                out2 = jax.block_until_ready(ad.run_layered(obs, gate_prefill_layers=gp2, gate_step_layers=gs2, noise=NOISE[s]))
                self_x0 += int(C.bytes_equal(np.asarray(out2["x_0"]), a_mask))
                if not C.bytes_equal(np.asarray(out["x_0"]), a_mv):
                    numeric_u = C.DiffAcc(); numeric_u.add(a_mv, np.asarray(out["x_0"]))
                    V.observe(numeric_u.line(f"UNROLL_VS_SCAN_NUMERIC point={n_pts}"))
                u = jax.block_until_ready(ad.f_attn_uniform_check(ad.gstate, obs))
                n_attn += 1
                ok = (np.allclose(np.asarray(u["share_f"]), np.asarray(u["U_f"])[:, None], atol=1e-6)
                      and np.allclose(np.asarray(u["share_t"]), np.asarray(u["U_t"])[:, None], atol=1e-6)
                      and np.allclose(np.asarray(u["share_step"]), np.asarray(u["U_step"]), atol=1e-6))
                attn_ok += int(ok)
            if args.grads and s == seeds[0]:
                if not hasattr(ad, "f_layer_grads"):
                    ad.build_grads()
                acts = jnp.asarray(pt["actions"], jnp.float32)[None]
                tt = 0.5
                x_t = tt * NOISE[s] + (1.0 - tt) * acts
                u_t = NOISE[s] - acts
                tstep = jnp.asarray([tt], jnp.float32)
                g = jax.block_until_ready(ad.f_layer_grads(ad.gstate, obs, x_t, tstep, u_t))
                gi = jax.block_until_ready(ad.f_input_grads(ad.gstate, obs, x_t, tstep, u_t))
                lp = float(jax.block_until_ready(ad.f_loss_fixed(ad.gstate, obs, x_t, tstep, u_t)))
                n_grad += 1
                cn = np.asarray(g["col_norm"])
                gme = np.asarray(gi["g_motion_emb"])[0]; gmp = np.asarray(gi["g_motion_pos"])[0]
                grad_finite += int(np.all(np.isfinite(cn)) and np.all(np.isfinite(gme)) and np.isfinite(float(g["loss"])))
                grad_pad0 += int(np.all(gme[~mm] == 0) and np.all(gmp[~mm] == 0))
                V.observe(f"GRAD_PROBE point={n_pts} k={k} loss_unrolled={float(g['loss']):.6g} loss_prod={lp:.6g} "
                          f"motion_col_norm_l0={float(cn[0, 0][np.asarray(g['mc'])[0]].mean()) if k else 0:.4g} "
                          f"g_motion_emb_valid_rms={C.rms(gme[mm]) if k else 0:.4g}")
        print(f"[mv] point {n_pts} task={pt['task']} g={pt['g']} t={pt['t']} k={k} done {time.perf_counter() - t0:.0f}s", flush=True)

    n = n_pts * len(seeds)
    ok1 = ref_prod == n and n > 0
    V.block(ok1, f"MV_NOOP_BITEXACT={'PASS' if ok1 and mv_ref == n else 'FAIL'} points={n} ref_vs_prod={ref_prod}/{n} "
                 f"mv_vs_ref={mv_ref}/{n} num_steps=10 note=同一 obs/同一 noise")
    if mv_ref != n:
        V.observe(numeric.line("MV_NOOP_NUMERIC(ref_vs_mv)") + " note=带门控算子的编译产物与无算子复刻不逐位；四臂全在 mv 内比")
    V.block(pad_ok == n_pad and n_pad > 0, f"MV_PAD_INVARIANT={'PASS' if pad_ok == n_pad else 'FAIL'} points={n} variants=3 bitexact={pad_ok}/{n_pad}")
    V.block(mask_null == n_null and n_null > 0, f"MV_MASK_CONTENT_NULL={'PASS' if mask_null == n_null else 'FAIL'} points={n} variants=3 "
                                                 f"bitexact={mask_null}/{n_null} masked_cols={C.MOTION_SLOTS}")
    if args.unroll:
        V.observe(f"UNROLL_VS_SCAN_BF16 points={n_unroll} x0_bitexact={unroll_x0}/{n_unroll} kv_bitexact={unroll_kv}/{n_unroll} "
                  f"all_mask_x0_bitexact={self_x0}/{n_unroll} layers={C.N_LAYERS} "
                  f"note=bf16 下展开与 nn.scan 是两份 XLA 编译产物，逐位不保证（与仓库 VT_FULL_VS_CACHED 3e-3 同性质）；语义等价由下面 f32 闸判定")
        V.block(attn_ok == n_attn and n_attn > 0, f"ATTN_UNIFORM_SELFCHECK={'PASS' if attn_ok == n_attn else 'FAIL'} points={n_attn} ok={attn_ok}/{n_attn} atol=1e-6")
        if args.f32_check:
            f32 = _f32_semantic_check(src, ad, NOISE[seeds[0]], key0, args)
            V.block(f32["ok_none"], f"UNROLL_VS_SCAN_F32={'PASS' if f32['ok_none'] else 'FAIL'} points={f32['n']} "
                                     f"x0_rel_fro={f32['x0_rel']:.3g} x0_max_abs={f32['x0_max']:.3g} kv_rel_fro={f32['kv_rel']:.3g} "
                                     f"thr_rel_fro={f32['thr']} dtype=f32(params+embed_dtype) matmul_precision=highest bf16_ref_rel_fro={f32['bf16_ref']:.3g}")
            V.block(f32["ok_mask"], f"LAYER_SELFCHECK_F32={'PASS' if f32['ok_mask'] else 'FAIL'} points={f32['n']} "
                                     f"all_layers_prefill+step_mask_vs_stage0_mask_rel_fro={f32['mask_rel']:.3g} max_abs={f32['mask_max']:.3g} thr_rel_fro={f32['thr']}")
    if args.grads:
        V.block(grad_finite == n_grad and grad_pad0 == n_grad and n_grad > 0,
                f"GRAD_SELFCHECK={'PASS' if grad_finite == n_grad and grad_pad0 == n_grad else 'FAIL'} points={n_grad} "
                f"finite={grad_finite}/{n_grad} pad_zero={grad_pad0}/{n_grad} layers={C.N_LAYERS}")
    lines = V.texts() + [V.final("MV_ADAPTER_SELFTEST", f"points={n_pts} seeds={len(seeds)}")]
    print("\n".join(lines))
    if args.out:
        p = pathlib.Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"argv": sys.argv, "lines": lines}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    src.close()
    return 0 if V.all_ok else 1


class _ModelShim:
    def __init__(self, model):
        self._model = model


def _f32_semantic_check(src, ad_bf16, noise, key0, args, thr: float = 1e-5) -> dict:
    """把参数树与 embed_dtype 一起升 f32、matmul_precision=highest，比较 nn.scan 路径与 18 层展开路径：
    (i) gate=none：x_0 与 prefill KV；(ii) 全层 prefill+step mask vs 阶段 0 mask 的 x_0。f32 下差异应掉到 1e-6 量级
    （仓库先例 VT_FULL_VS_CACHED_F32 = 1.35e-7），坐实展开实现与生产路径数学同式；bf16 下的 3e-3 是 kernel/归约序差。"""
    import gc
    import jax
    import jax.numpy as jnp
    import omegaconf
    from openpi.models import model as _model
    t0 = time.perf_counter()
    train_config = src.train_config
    hc = omegaconf.OmegaConf.load(src.run_root / "history_config.resolved.yaml")
    params_f32 = _model.restore_params(src.ckpt_dir / "params", dtype=jnp.float32)
    tc = dataclasses.replace(train_config, model=dataclasses.replace(train_config.model, history_config=hc, use_history=True, dtype="float32"))
    model_f32 = tc.model.load(params_f32, remove_extra_params=False)
    ad32 = MVAdapter(_ModelShim(model_f32), unroll=True)
    acc_x0, acc_kv, acc_mask = C.DiffAcc(), C.DiffAcc(), C.DiffAcc()
    acc_bf16 = C.DiffAcc()
    n = 0
    with jax.default_matmul_precision("highest"):
        for pt in src.iter_points(max_total=args.max_total_points):
            obs = pt["obs"]
            fns32 = ad32.make_sample_fns()
            a_none = np.asarray(jax.block_until_ready(fns32["none"](key0, obs, noise=noise)))
            a_mask = np.asarray(jax.block_until_ready(fns32["mask"](key0, obs, noise=noise)))
            pre = jax.block_until_ready(ad32.f_prefix(ad32.gstate, obs))
            gp, gs = ad32.layered_gates(obs, "none", "none")
            out = jax.block_until_ready(ad32.run_layered(obs, gate_prefill_layers=gp, gate_step_layers=gs, noise=noise))
            gp2, gs2 = ad32.layered_gates(obs, "mask", "mask")
            out2 = jax.block_until_ready(ad32.run_layered(obs, gate_prefill_layers=gp2, gate_step_layers=gs2, noise=noise))
            pv = np.asarray(pre["mask"][0], bool)
            acc_x0.add(a_none, np.asarray(out["x_0"]))
            acc_kv.add(np.asarray(pre["k"])[:, :, pv], np.asarray(out["k"])[:, :, pv])
            acc_kv.add(np.asarray(pre["v"])[:, :, pv], np.asarray(out["v"])[:, :, pv])
            acc_mask.add(a_mask, np.asarray(out2["x_0"]))
            # bf16 参照（同一点、生产 bf16 路径 vs bf16 展开）
            fnsb = ad_bf16.make_sample_fns()
            ab = np.asarray(jax.block_until_ready(fnsb["none"](key0, obs, noise=noise)))
            gpb, gsb = ad_bf16.layered_gates(obs, "none", "none")
            ob = jax.block_until_ready(ad_bf16.run_layered(obs, gate_prefill_layers=gpb, gate_step_layers=gsb, noise=noise))
            acc_bf16.add(ab, np.asarray(ob["x_0"]))
            n += 1
    sx, sk, sm = acc_x0.summary(), acc_kv.summary(), acc_mask.summary()
    res = {"n": n, "x0_rel": sx["rel_fro"], "x0_max": sx["max_abs"], "kv_rel": sk["rel_fro"], "mask_rel": sm["rel_fro"],
           "mask_max": sm["max_abs"], "thr": thr, "bf16_ref": acc_bf16.summary()["rel_fro"],
           "ok_none": n > 0 and sx["rel_fro"] <= thr and sk["rel_fro"] <= thr, "ok_mask": n > 0 and sm["rel_fro"] <= thr,
           "wall_s": time.perf_counter() - t0}
    del params_f32, model_f32, ad32
    gc.collect()
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--f32-check", action="store_true", help="展开 vs scan 的 f32/highest 语义等价闸（--unroll 时）")
    ap.add_argument("--unroll", action="store_true")
    ap.add_argument("--grads", action="store_true")
    ap.add_argument("--dump-param-tree", action="store_true")
    ap.add_argument("--lib", default=str(C.LIB))
    ap.add_argument("--ckpt", default=str(C.CKPT_MOTION))
    ap.add_argument("--train-config", default=C.TRAIN_CONFIG)
    ap.add_argument("--episodes", default="ButtonUnmaskSwap:3,VideoUnmask:3,VideoUnmaskSwap:5,VideoUnmaskSwap:31,VideoUnmaskSwap:3")
    ap.add_argument("--max-points", type=int, default=1, help="每集前 N 个决策点")
    ap.add_argument("--max-total-points", type=int, default=4)
    ap.add_argument("--noise-seeds", default="0,1,2")
    ap.add_argument("--out", default=str(C.REPORT_ROOT / "adapter_selftest.json"))
    args = ap.parse_args()
    if not args.selftest:
        ap.error("只支持 --selftest（适配器由 serve_policy_mv.py / run_open_loop.py import 使用）")
    return selftest(args)


if __name__ == "__main__":
    sys.exit(main())
