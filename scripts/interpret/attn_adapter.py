#!/usr/bin/env python3
"""attention 采集适配器（计划 1103-interpret-motion-attention-plan.html 第二部分 §1.1）。

`AttnAdapter(policy)` 复刻 `HistoryPi0.sample_actions` 的 context 分支，把 `nn.scan` 的 18 层手工展开：
  - prefill：逐层 `HistoryBlock` 前向、存 K/V，并捕获每层 softmax 概率，只切文本 query 行 `[MEM+IMG : P]`；
  - 去噪 while_loop：每步每层捕获动作 query 的概率 `probs[:, 0]`（b, 8, AH, P+AH），按步累进，原样不平均（K10）。
不加任何门控、不做 KV 替换；只读 probs，不改任何 mask / positions / input_mask / mem_order（§〇 红线）。

复用 `scripts/motion-variance/mv_model_adapter.py` 的三个符号（import 复用，不复制）：
  `_make_patched_attention`（与 gemma.Attention.__call__ 逐字同式、只追加一行抓 f32 概率）、
  `_capture_attn`（trace 期把 history_gemma.Attention 换成上面那个类）、
  `_guard_model_source`（生产源码指纹护栏，漂移即 SystemExit）。
注意 mv_common 的 PREFIX_LEN=1184 / MEM_LEN=608 是写死给 motion 模型的，本文件不用；
P / MEM / IMG / TXT 一律从当次观测经 `embed_prefix` / `embed_memory` 的抽象求值形状得到
（官方 ctx512：P=1088、MEM=512；motion concat：P=1184、MEM=608；IMG=2 路图像×256=512、TXT=64）。

接口（S2 依赖，名字定死）：
  adapter = AttnAdapter(policy)                 # policy: MME_VLA_Policy
  policy._sample_actions = adapter.make_sample_fn()   # fn(rng, observation, **kw)，kw 支持 noise / num_steps（默认 10）
  adapter.last_attn = {"probs_action": f16 (num_steps, 18, 8, 20, P+20),
                       "probs_text":   f16 (18, 8, 64, P),        # padding 文本行置 0（npz 契约 §1.3）
                       "P": int, "MEM": int}
  # 可选 AttnAdapter(policy, save_prefix=True)：last_attn 另含 "probs_prefix": f16 (18, 8, P, P)，
  # 即 prefill 每层完整 prefix 自注意力（padding query 行置 0）；默认关闭时 jit 不产出该张量、其余输出不变。

因果遮挡（S7，默认全关；关闭时走与原来逐字相同的 trace，编译产物与输出不变）：
  - key 遮挡 mask_keys：bool[P]（或 bool[MEM]，自动右补 False 到 P），True = 该 key 列被遮；每次推理动态传入
    （jit 普通参数，非 static）。prefill：非被遮 query 行 × 被遮 key 列 → False，被遮 token 之间照常互读
    （同 mv_model_adapter.MVAdapter._build_sample 的 apply_gate）；去噪：动作 query 不能读被遮列。
    只改 attention mask 的 key 列，不改 input_mask / positions / mem_order（positions 仍按原 prefix_mask 求）。
  - 头遮挡 ablate_heads：static 元组 ((layer, head), ...)，把该层该头的 attention 输出（out_einsum 之前的
    encoded，B T K G H 里 query 头 head = k*G+g）置零，prefill 与去噪都生效；由本文件 _make_ablate_attention 实现。
  用法：AttnAdapter(policy, ablate_heads=((3, 0),))；fn = adapter.make_sample_fn()；fn(rng, obs, mask_keys=<bool[P]>)。
  last_attn 另含 "n_masked_keys"（本次被遮 key 列数）与 "ablate_heads"。

隐状态捕获（计划 1105-stopcube-real-rollout-plan.html，默认关闭；关闭时调用形式与编译产物不变）：
  output = adapter.run(rng, obs, noise=..., capture_hidden=True, group_masks=<bool[G,P]>)
  output[-1] = {"hidden_prefix": f32 (19, G, W),      # 第 0 项是 embed_prefix 的 pt，第 1～18 项是各层输出；按组对有效 token 取均值
                "hidden_action": f32 (18, W_a),       # 各层后 AH 个 token 里前 16 个的均值，再对去噪步求平均
                "group_tokens":  int32 (G,)}          # 每组实际参与均值的有效 token 数；空组对应的均值全 0
  只读隐状态，不改采样计算；只支持 batch=1；与 mask_keys / ablate_heads 同时给时直接报错。

只做前缀的隐状态捕获（计划 1105-stopcube-hover-probe-plan.html W2）：
  adapter.run_prefix(obs, group_masks=<bool[G,P]>) -> {"hidden_prefix": f32 (19, G, W), "group_tokens": int32 (G,)}
  前缀段的计算与 capture_hidden=True 相同，不做去噪；只支持 batch=1；由独立的 jit 函数 f_prefix 执行，不影响 f_sample_attn。

用法：
  # CPU、不加载权重，抽象求值两种 P 下的输出形状
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/interpret/attn_adapter.py --dry-shapes
  # GPU 自检（主会话在 GPU0 跑，≤5 min）
  CUDA_VISIBLE_DEVICES=0 XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
    uv run --no-sync python scripts/interpret/attn_adapter.py --selftest \
      --ckpt v1-store/assets/vla-o16cc-59999/59999 --config mme_vla_suite_orig16_b128_60k
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import os
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_MV_DIR = _HERE.parent / "motion-variance"
sys.path.insert(0, str(_MV_DIR))
import mv_common as C  # noqa: E402
import mv_model_adapter as _MVA  # noqa: E402
from mv_model_adapter import _capture_attn, _guard_model_source, _make_patched_attention  # noqa: E402

N_LAYERS = 18
DEFAULT_NUM_STEPS = 10
HIDDEN_ACTION_TOKENS = 16        # 动作隐状态只取后 AH 个 token 里的前 16 个（计划第一部分第三节第 5 小节）
# RoboMME 推理只喂两路图像（robomme_policy.py 去掉了 right_wrist_0_rgb），dry-shapes 的抽象观测与之一致
ROBOMME_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb")


def normalize_heads(ablate_heads, n_layers: int = N_LAYERS, n_heads: int = 8) -> tuple:
    """((layer, head), ...) → 去重排序的 int 元组；越界即 ValueError。"""
    out = set()
    for item in ablate_heads or ():
        layer, head = (int(x) for x in item)
        if not (0 <= layer < n_layers and 0 <= head < n_heads):
            raise ValueError(f"头遮挡越界 L{layer}H{head}（层 0..{n_layers - 1}，头 0..{n_heads - 1}）")
        out.add((layer, head))
    return tuple(sorted(out))


_ABLATE_CLS: dict = {}


def _make_ablate_attention(zero_heads: tuple):
    """与 mv_model_adapter._make_patched_attention 的 _PatchedAttention 逐字同式（同样把 f32 概率追加到 _CAPTURE），
    只在 out_einsum 之前把 zero_heads 这些 query 头的 encoded 置零（jnp.where，未遮的头原值透传）。"""
    zero_heads = tuple(sorted(set(int(h) for h in zero_heads)))
    if zero_heads in _ABLATE_CLS:
        return _ABLATE_CLS[zero_heads]
    import einops
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import openpi.models.lora as lora
    from openpi.models import gemma
    from openpi.models.gemma import _apply_rope, _name

    capture = _MVA._CAPTURE

    class _AblateAttention(gemma.Attention):
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
            capture.append(probs_f32)
            probs = probs_f32.astype(dtype)
            encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
            # ← 头遮挡：query 头 h = k*G + g 的输出置零（out_einsum 之前）
            n_k, n_g = encoded.shape[2], encoded.shape[3]
            keep = np.ones((n_k, n_g), bool)
            for h in zero_heads:
                if not 0 <= h < n_k * n_g:
                    raise ValueError(f"头号 {h} 越界（共 {n_k * n_g} 个 query 头）")
                keep[h // n_g, h % n_g] = False
            encoded = jnp.where(jnp.asarray(keep)[None, None, :, :, None], encoded, jnp.zeros((), encoded.dtype))
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

    _ABLATE_CLS[zero_heads] = _AblateAttention
    return _AblateAttention


def head_out_leaves(state) -> list:
    """llm 里 attn_vec_einsum（两路 expert）的参数叶子 (keystr, shape)；自检用权重置零的定位。"""
    import jax
    found = []

    def visit(path, x):
        ks = jax.tree_util.keystr(path)
        if "llm" in ks and "attn_vec_einsum" in ks:
            found.append((ks, tuple(x.shape)))
        return x

    jax.tree_util.tree_map_with_path(visit, state)
    return found


def zero_head_out_weights(state, ablate_heads, n_layers: int = N_LAYERS):
    """把 attn_vec_einsum 权重的 [layer, head] 行置零（与 encoded 该头置零在数学上等价）；返回 (新 state, 置零叶子数)。"""
    import jax
    import jax.numpy as jnp
    heads = normalize_heads(ablate_heads, n_layers)
    n = [0]

    def f(path, x):
        ks = jax.tree_util.keystr(path)
        if "llm" in ks and "attn_vec_einsum" in ks:
            if x.ndim != 4 or x.shape[0] != n_layers:
                raise SystemExit(f"错误: {ks} 形状 {x.shape} 不是 (L, N, H, D)")
            n[0] += 1
            for layer, head in heads:
                x = x.at[layer, head].set(jnp.zeros((), x.dtype))
        return x

    return jax.tree_util.tree_map_with_path(f, state), n[0]


def group_token_mean(tokens, group_masks, valid):
    """tokens (P,W)、group_masks bool[G,P]、valid bool[P] → (每组有效 token 的 f32 均值 (G,W), 每组 token 数 int32 (G,))。
    组外 token 不参与求和（先按 mask 取值，而不是乘 0），空组均值为 0，不产生 NaN。"""
    import jax.numpy as jnp
    members = group_masks & valid[None, :]                                    # (G,P)
    counts = members.sum(axis=1).astype(jnp.int32)
    picked = jnp.where(members[:, :, None], tokens[None].astype(jnp.float32), jnp.zeros((), jnp.float32))
    return picked.sum(axis=1) / jnp.maximum(counts, 1)[:, None].astype(jnp.float32), counts


def action_token_mean(hs, action_horizon: int, n_tokens: int = HIDDEN_ACTION_TOKENS):
    """hs (T,W_a) → 后 action_horizon 个 token 里前 n_tokens 个的 f32 均值 (W_a,)。"""
    return hs[-action_horizon:][:n_tokens].astype("float32").mean(axis=0)


class _ModelShim:
    """只带 `_model` 的替身（dry-shapes 与 f32 自检用），与 MME_VLA_Policy 暴露同一属性。"""

    def __init__(self, model):
        self._model = model


class AttnAdapter:
    def __init__(self, policy, *, require_flags: bool = True, save_prefix: bool = False, ablate_heads=()):
        import einops
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        from mme_vla_suite.models.integration import history_gemma as hg
        from mme_vla_suite.models.integration.history_observation import preprocess_observation
        from mme_vla_suite.models.integration.history_pi0 import make_attn_mask
        from mme_vla_suite.models.integration.utils import get_config
        from openpi.models import gemma

        if require_flags:
            C.require_det_flags()
        _guard_model_source()
        self.jax, self.jnp, self.nnx, self.einops = jax, jnp, nnx, einops
        self.hg, self.gemma = hg, gemma
        self.policy = policy
        self.model = policy._model
        if not getattr(self.model, "use_history", False) or getattr(self.model, "integration_type", None) != "context":
            raise SystemExit(f"错误: 只支持 HistoryPi0 context 分支（use_history={getattr(self.model, 'use_history', None)}, "
                             f"integration_type={getattr(self.model, 'integration_type', None)!r}）")
        self.gdef, self.gstate = nnx.split(self.model)
        self.AH, self.AD = int(self.model.action_horizon), int(self.model.action_dim)
        self.embed_dtype = self.model.config.dtype
        self.cfgs = (get_config(self.model.config.paligemma_variant), get_config(self.model.config.action_expert_variant))
        if self.cfgs[0].depth != N_LAYERS:
            raise SystemExit(f"错误: llm 深度 {self.cfgs[0].depth} != {N_LAYERS}")
        self.n_heads = int(self.cfgs[0].num_heads)
        self._preprocess = preprocess_observation
        self._make_attn_mask = make_attn_mask
        self._layout_cache: dict = {}
        self.last_attn: dict | None = None
        self.save_prefix = bool(save_prefix)
        self.ablate_heads = normalize_heads(ablate_heads, N_LAYERS, self.n_heads)
        self._check_param_tree()
        self._build()

    # ── 参数树键名校验（参数在 jit 内从入参 state 取，不闭包常量）─────────────────────────────
    def _check_param_tree(self) -> None:
        llm = self.gstate.to_pure_dict()["PaliGemma"]["llm"]
        for k in ("layers", "final_norm_1"):
            if k not in llm:
                raise SystemExit(f"错误: llm 参数树缺 {k!r}，现有键 {sorted(llm)}")

    # ── 布局：P / MEM / IMG / TXT 从当次观测形状抽象求值得到，不写死 ─────────────────────────
    def layout(self, obs) -> dict:
        jax = self.jax
        sig = tuple((tuple(x.shape), str(x.dtype)) for x in jax.tree.leaves(obs))
        if sig in self._layout_cache:
            return self._layout_cache[sig]
        gdef, nnx, pre = self.gdef, self.nnx, self._preprocess

        def f(state, o):
            m = nnx.merge(gdef, state)
            o = pre(None, o, train=False)
            mem_tok, _, _, _ = m.embed_memory(o)
            pt, _, _, _ = m.embed_prefix(o)
            return mem_tok, pt, o.tokenized_prompt

        mem_tok, pt, tp = jax.eval_shape(f, self.gstate, obs)
        P, MEM, TXT = int(pt.shape[1]), int(mem_tok.shape[1]), int(tp.shape[1])
        IMG = P - MEM - TXT
        n_img = len(obs.images)
        if IMG <= 0 or IMG % max(n_img, 1) != 0:
            raise SystemExit(f"错误: 前缀布局异常 P={P} MEM={MEM} TXT={TXT} IMG={IMG} n_img={n_img}")
        out = {"P": P, "MEM": MEM, "IMG": IMG, "TXT": TXT, "n_img": n_img}
        self._layout_cache[sig] = out
        return out

    # ── 展开版 sample_actions（context 分支，无门控、无 KV override）─────────────────────────
    def _build(self):
        jax, jnp, nnx, einops = self.jax, self.jnp, self.nnx, self.einops
        hg, gemma = self.hg, self.gemma
        gdef = self.gdef
        preprocess_observation, make_attn_mask = self._preprocess, self._make_attn_mask
        cfgs, ed, L, AH = self.cfgs, self.embed_dtype, N_LAYERS, self.AH
        patched = _make_patched_attention()
        block = hg.HistoryBlock(configs=cfgs, integration_type="context")
        final_norm_1 = gemma.RMSNorm()

        def llm_of(state):
            return state.to_pure_dict()["PaliGemma"]["llm"]

        def block_apply(layers, l, xs, kv, positions, mask3d, adarms, attn_cls=patched):
            p_l = jax.tree.map(lambda x: x[l], layers)
            with _capture_attn(hg, attn_cls) as cap:
                out, kv_out = block.apply({"params": p_l}, xs, kv, positions, mask3d[:, None, :, :],
                                          adarms, None, None, True)
                probs = cap[0]
            return out, kv_out, probs

        @functools.partial(jax.jit, static_argnames=("num_steps", "txt_lo", "out_f16", "save_prefix", "ablate_heads",
                                                    "capture_hidden"))
        def f_sample_attn(state, rng, obs, noise, num_steps, txt_lo, out_f16=True, save_prefix=False,
                          mask_keys=None, ablate_heads=(), capture_hidden=False, group_masks=None):
            """逐字复刻 HistoryPi0.sample_actions（context 分支）的 18 层展开版；返回 (x_0, probs_action, probs_text, n_done)。
            probs_action (num_steps, L, b, G, AH, P+AH)；probs_text (L, b, G, TXT, P)，padding 文本行置 0。
            save_prefix=True（static）时末尾多返回 probs_prefix (L, b, G, P, P)：prefill 每层完整概率，padding query 行置 0。
            mask_keys bool[b,P]（动态，None=关）：被遮 key 列；ablate_heads（static）((layer, head), ...)：头输出置零。
            capture_hidden=True（static）时末尾再多返回 dict：hidden_prefix f32 (L+1,G,W)、hidden_action f32 (L,W_a)、
            group_tokens int32 (G,)，group_masks bool[G,P] 为动态参数；只读 h / hs，不参与采样计算。"""
            if capture_hidden and (mask_keys is not None or ablate_heads):
                raise ValueError("capture_hidden 不能与 mask_keys / ablate_heads 同时使用")
            if capture_hidden and group_masks is None:
                raise ValueError("capture_hidden=True 时必须给 group_masks")
            by_layer: dict = {}
            for layer, head in ablate_heads:
                by_layer.setdefault(int(layer), []).append(int(head))
            cls_of = {layer: _make_ablate_attention(tuple(hs)) for layer, hs in by_layer.items()}
            m = nnx.merge(gdef, state)
            llm = llm_of(state)
            layers, fn1 = llm["layers"], llm["final_norm_1"]
            o = preprocess_observation(None, obs, train=False)
            dt = -1.0 / num_steps
            b = o.state.shape[0]
            if noise is None:
                noise = jax.random.normal(rng, (b, m.action_horizon, m.action_dim))
            pt, pm, par_, pna = m.embed_prefix(o)
            base = make_attn_mask(pm, par_, pna)                        # (b,P,P)
            if mask_keys is not None:
                # 干预点 1（prefill）：非被遮 query 行 × 被遮 key 列 → False；被遮 token 之间照常互读
                mk = jnp.broadcast_to(mask_keys.astype(bool), pm.shape)
                base = base & ~(mk[:, None, :] & ~mk[:, :, None])
            pos = jnp.cumsum(pm, axis=1) - 1                             # 用原 prefix_mask，绝不用遮挡
            P = pt.shape[1]
            h = pt.astype(ed)
            if capture_hidden:
                if b != 1 or group_masks.ndim != 2 or group_masks.shape[1] != P:
                    raise ValueError(f"隐状态捕获只支持 batch=1 且 group_masks 为 bool[G,{P}]，"
                                     f"得到 batch={b}、group_masks.shape={group_masks.shape}")
                hidden0, group_tokens = group_token_mean(pt[0], group_masks, pm[0])
                hidden_rows = [hidden0]
            ks, vs, txt, pre = [], [], [], []
            for l in range(L):
                (h, _), (k, v), probs = block_apply(layers, l, [h, None], None, pos, base, [None, None],
                                                    cls_of.get(l, patched))
                ks.append(k); vs.append(v)
                if capture_hidden:
                    hidden_rows.append(group_token_mean(h[0], group_masks, pm[0])[0])
                txt.append(probs[:, 0, :, txt_lo:P, :])                  # (b,G,TXT,P)
                if save_prefix:
                    pre.append(probs[:, 0])                              # (b,G,P,P)：完整 prefix 自注意力
            K = jnp.stack(ks); V = jnp.stack(vs)                         # (L,b,P,1,H)
            txt_valid = pm[:, txt_lo:P].astype(jnp.float32)              # padding 文本 query 行（全遮蔽 → softmax 均匀）置 0
            probs_text = jnp.stack(txt) * txt_valid[None, :, None, :, None]
            spos_base = jnp.sum(pm, axis=-1)[:, None]

            def step(carry):
                x_t, time, i, acc = carry[:4]
                st, sm, sar, _, ad = m.embed_suffix(o, x_t, jnp.broadcast_to(time, b))
                sattn = make_attn_mask(sm, sar)
                prefix_attn = einops.repeat(pm, "b p -> b s p", s=st.shape[1])
                if mask_keys is not None:
                    prefix_attn = prefix_attn & ~mk[:, None, :]          # 干预点 2（去噪）：动作 query 不读被遮列
                full = jnp.concatenate([prefix_attn, sattn], axis=-1)
                spos = spos_base + jnp.cumsum(sm, axis=-1) - 1
                hs = st.astype(ed)
                rows, action_rows = [], []
                for l in range(L):
                    (_, hs), _, probs = block_apply(layers, l, [None, hs], (K[l], V[l]), spos, full, [None, ad],
                                                    cls_of.get(l, patched))
                    rows.append(probs[:, 0])                             # (b,G,AH,P+AH)：不平均，原样保留
                    if capture_hidden:
                        action_rows.append(action_token_mean(hs[0], AH))
                hs, _ = final_norm_1.apply({"params": fn1}, hs, ad)
                v_t = m.action_out_proj(hs[:, -AH:])
                done = (x_t + dt * v_t, time + dt, i + 1, acc.at[i].set(jnp.stack(rows)))
                return (*done, carry[4] + jnp.stack(action_rows)) if capture_hidden else done

            def cond(carry):
                return carry[1] >= -dt / 2

            acc0 = jnp.zeros((num_steps, L, b, cfgs[0].num_heads, AH, P + AH), jnp.float32)
            carry0 = (noise, 1.0, jnp.int32(0), acc0)
            if capture_hidden:                                           # 和 acc 一样预分配，循环内累加
                carry0 = (*carry0, jnp.zeros((L, cfgs[1].width), jnp.float32))
            x_0, _, n_done, acc, *action_sum = jax.lax.while_loop(cond, step, carry0)
            tail = ()
            if capture_hidden:
                tail = ({"hidden_prefix": jnp.stack(hidden_rows),
                         "hidden_action": action_sum[0] / n_done.astype(jnp.float32),
                         "group_tokens": group_tokens},)
            if save_prefix:                                              # padding query 行（全遮蔽 → softmax 均匀）置 0，同 probs_text
                probs_prefix = jnp.stack(pre) * pm.astype(jnp.float32)[None, :, None, :, None]
                if out_f16:
                    return (x_0, acc.astype(jnp.float16), probs_text.astype(jnp.float16), n_done,
                            probs_prefix.astype(jnp.float16), *tail)
                return (x_0, acc, probs_text, n_done, probs_prefix, *tail)
            if out_f16:
                return (x_0, acc.astype(jnp.float16), probs_text.astype(jnp.float16), n_done, *tail)
            return (x_0, acc, probs_text, n_done, *tail)

        @functools.partial(jax.jit, static_argnames=("txt_lo",))
        def f_masks(state, obs, txt_lo):
            """自检用：文本 query 行的合法 key（结构 mask）与去噪 query 的合法 key，形状同 probs 的末两维。"""
            m = nnx.merge(gdef, state)
            o = preprocess_observation(None, obs, train=False)
            pt, pm, par_, pna = m.embed_prefix(o)
            base = make_attn_mask(pm, par_, pna)
            b = pm.shape[0]
            x0 = jnp.zeros((b, m.action_horizon, m.action_dim), jnp.float32)
            st, sm, sar, _, _ = m.embed_suffix(o, x0, jnp.ones((b,), jnp.float32))
            sattn = make_attn_mask(sm, sar)
            full = jnp.concatenate([einops.repeat(pm, "b p -> b s p", s=st.shape[1]), sattn], axis=-1)
            return {"text": base[:, txt_lo:], "text_valid": pm[:, txt_lo:], "action": full, "prefix_valid": pm}

        @jax.jit
        def f_prefix(state, obs, group_masks):
            """f_sample_attn 在 capture_hidden=True 时的前缀段，不做去噪；返回 hidden_prefix f32 (L+1,G,W)、group_tokens int32 (G,)。"""
            m = nnx.merge(gdef, state)
            layers = llm_of(state)["layers"]
            o = preprocess_observation(None, obs, train=False)
            b = o.state.shape[0]
            pt, pm, par_, pna = m.embed_prefix(o)
            base = make_attn_mask(pm, par_, pna)
            pos = jnp.cumsum(pm, axis=1) - 1
            P = pt.shape[1]
            if b != 1 or group_masks.ndim != 2 or group_masks.shape[1] != P:
                raise ValueError(f"隐状态捕获只支持 batch=1 且 group_masks 为 bool[G,{P}]，"
                                 f"得到 batch={b}、group_masks.shape={group_masks.shape}")
            h = pt.astype(ed)
            hidden0, group_tokens = group_token_mean(pt[0], group_masks, pm[0])
            hidden_rows = [hidden0]
            for l in range(L):
                (h, _), _, _ = block_apply(layers, l, [h, None], None, pos, base, [None, None])
                hidden_rows.append(group_token_mean(h[0], group_masks, pm[0])[0])
            return {"hidden_prefix": jnp.stack(hidden_rows), "group_tokens": group_tokens}

        self.f_sample_attn = f_sample_attn
        self.f_masks = f_masks
        self.f_prefix = f_prefix

    def pad_mask_keys(self, mask_keys, lay):
        """bool[P] / bool[MEM] / bool[b,·] → jnp bool[b,P]（MEM 长度右补 False）；None 透传。"""
        if mask_keys is None:
            return None
        mk = np.asarray(mask_keys, dtype=bool)
        if mk.ndim == 1:
            mk = mk[None]
        P, MEM = lay["P"], lay["MEM"]
        if mk.shape[-1] == MEM and MEM != P:
            mk = np.concatenate([mk, np.zeros(mk.shape[:-1] + (P - MEM,), bool)], axis=-1)
        if mk.shape[-1] != P:
            raise ValueError(f"mask_keys 长度 {mk.shape[-1]} 既不是 P={P} 也不是 MEM={MEM}")
        return self.jnp.asarray(mk)

    def run(self, rng, observation, *, noise=None, num_steps: int = DEFAULT_NUM_STEPS, out_f16: bool = True, state=None,
            save_prefix: bool | None = None, mask_keys=None, ablate_heads=None,
            capture_hidden: bool = False, group_masks=None):
        """调一次展开版；返回 (x_0, probs_action, probs_text, n_done, layout)，probs 仍在 device 上、保留 batch 轴。
        save_prefix（缺省取构造参数）为真时末尾多一个 probs_prefix (L, b, G, P, P)。
        mask_keys（缺省 None=关）见 pad_mask_keys；ablate_heads（缺省取构造参数）((layer, head), ...)。
        遮挡全关时调用形式与 S7 之前逐字相同（不传 mask_keys / ablate_heads），trace 与编译产物不变。
        capture_hidden=True 时在上述返回值之后再多一个 dict（hidden_prefix、hidden_action、group_tokens，见文件头），
        group_masks 为 bool[G,P]；默认关闭时不向 f_sample_attn 传这两个参数，调用与编译产物不变。"""
        if group_masks is not None and not capture_hidden:
            raise ValueError("group_masks 只在 capture_hidden=True 时使用")
        lay = self.layout(observation)
        st = self.gstate if state is None else state
        heads = self.ablate_heads if ablate_heads is None else normalize_heads(ablate_heads, N_LAYERS, self.n_heads)
        extra = {}
        if mask_keys is not None:
            extra["mask_keys"] = self.pad_mask_keys(mask_keys, lay)
        if heads:
            extra["ablate_heads"] = heads
        if capture_hidden:
            extra["capture_hidden"] = True
            extra["group_masks"] = None if group_masks is None else self.jnp.asarray(np.asarray(group_masks, dtype=bool))
        if self.save_prefix if save_prefix is None else bool(save_prefix):
            x_0, pa, ptxt, n_done, pp, *hidden = self.f_sample_attn(st, rng, observation, noise, int(num_steps),
                                                                    lay["MEM"] + lay["IMG"], out_f16, save_prefix=True, **extra)
            return (x_0, pa, ptxt, n_done, lay, pp, *hidden)
        x_0, pa, ptxt, n_done, *hidden = self.f_sample_attn(st, rng, observation, noise, int(num_steps),
                                                            lay["MEM"] + lay["IMG"], out_f16, **extra)
        return (x_0, pa, ptxt, n_done, lay, *hidden)

    def run_prefix(self, observation, group_masks):
        """只做前缀前向；返回 {"hidden_prefix": f32 (19,G,W), "group_tokens": int32 (G,)}，仍在 device 上。只支持 batch=1。"""
        return self.f_prefix(self.gstate, observation, self.jnp.asarray(np.asarray(group_masks, dtype=bool)))

    def make_sample_fn(self, ablate_heads=None):
        """返回可顶掉 policy._sample_actions 的可调用：fn(rng, observation, **kw) -> actions（与生产同形同 dtype）。
        kw 支持 noise / num_steps / mask_keys（每次推理由调用方按观测算好，bool[P] 或 bool[MEM]，None=关）。
        ablate_heads 缺省取构造参数。每次调用后 self.last_attn 存本次推理的原始概率（batch=1，去 batch 轴，f16，不平均）。"""
        jax = self.jax
        heads = self.ablate_heads if ablate_heads is None else normalize_heads(ablate_heads, N_LAYERS, self.n_heads)

        def fn(rng, observation, **kw):
            extra = set(kw) - {"noise", "num_steps", "mask_keys"}
            if extra:
                raise TypeError(f"AttnAdapter sample_fn 不支持的参数 {sorted(extra)}")
            num_steps = int(kw.get("num_steps", DEFAULT_NUM_STEPS))
            if observation.state.shape[0] != 1:
                raise ValueError(f"只支持 batch=1，得到 {observation.state.shape[0]}")
            mask_keys = kw.get("mask_keys")
            out = self.run(rng, observation, noise=kw.get("noise"), num_steps=num_steps, mask_keys=mask_keys,
                           ablate_heads=heads)
            x_0, pa, ptxt, n_done, lay = out[:5]
            pa_h, ptxt_h, n_h = jax.device_get((pa, ptxt, n_done))
            if int(n_h) != num_steps:
                raise RuntimeError(f"去噪循环实际步数 {int(n_h)} != num_steps {num_steps}")
            self.last_attn = {"probs_action": np.asarray(pa_h[:, :, 0]),       # (num_steps, L, G, AH, P+AH) f16
                              "probs_text": np.asarray(ptxt_h[:, 0]),          # (L, G, TXT, P) f16
                              "P": int(lay["P"]), "MEM": int(lay["MEM"]),
                              "n_masked_keys": int(np.asarray(mask_keys, bool).sum()) if mask_keys is not None else 0,
                              "ablate_heads": heads}
            if self.save_prefix:
                self.last_attn["probs_prefix"] = np.asarray(jax.device_get(out[5])[:, 0])   # (L, G, P, P) f16
            return x_0

        fn.__name__ = "f_sample_attn"
        return fn


# ══════════════════════════════════════════ dry-shapes（CPU、不加载权重）══════════════════════════════════════════
def _resolve_history_config(ckpt: pathlib.Path, fallback_yaml: str | None):
    """与 create_trained_policy 同序：run 内 resolved yaml 优先、其次 history_config.txt；都没有时用仓库 yaml（注明来源）。"""
    import omegaconf
    from mme_vla_suite.models.config.utils import get_history_config
    run_root = ckpt.parent
    if (run_root / "history_config.resolved.yaml").is_file():
        return omegaconf.OmegaConf.load(run_root / "history_config.resolved.yaml"), "run_resolved_yaml"
    if (run_root / "history_config.txt").is_file():
        return get_history_config((run_root / "history_config.txt").read_text().strip()), "run_history_config_txt"
    if fallback_yaml is None:
        raise SystemExit(f"错误: {run_root} 下没有 history 配置快照")
    return get_history_config(fallback_yaml), f"repo_yaml:{fallback_yaml}"


def _robomme_obs_spec(model_cfg):
    """inputs_spec 带三路图像，RoboMME 推理只喂两路；抽象观测与真实推理同布局才能得到真实的 P。"""
    spec, _ = model_cfg.inputs_spec(batch_size=1)
    keep = {k: v for k, v in spec.images.items() if k in ROBOMME_IMAGE_KEYS}
    masks = {k: v for k, v in spec.image_masks.items() if k in ROBOMME_IMAGE_KEYS}
    if len(keep) != len(ROBOMME_IMAGE_KEYS):
        raise SystemExit(f"错误: inputs_spec 图像键 {sorted(spec.images)} 不含 {ROBOMME_IMAGE_KEYS}")
    from openpi.shared import array_typing as at
    with at.disable_typechecking():                 # 与 inputs_spec 同：ShapeDtypeStruct 过不了 jaxtyping
        return dataclasses.replace(spec, images=keep, image_masks=masks)


def dry_shapes(args) -> int:
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from mme_vla_suite.training import config as _config

    os.chdir(C.REPO_ROOT)          # get_history_config 的仓库 yaml 是相对路径
    print(f"[dry] backend={jax.default_backend()} repo={C.REPO_ROOT}", flush=True)
    specs = [("official_ctx512", pathlib.Path(args.official_ckpt), args.official_config, "perceptual-framesamp-context.yaml"),
             ("motion_concat", pathlib.Path(args.motion_ckpt), args.motion_config, None)]
    ok_all = True
    summary = []
    for tag, ckpt, cfg_name, fb in specs:
        t0 = time.perf_counter()
        try:
            tc = _config.get_config(cfg_name)
            cfg_used = cfg_name
        except Exception as e:                                  # noqa: BLE001
            tc = _config.get_config("mme_vla_suite_b128")
            cfg_used = f"mme_vla_suite_b128(fallback:{cfg_name} 未注册:{type(e).__name__})"
        hc, hc_src = _resolve_history_config(ckpt, fb)
        model_cfg = dataclasses.replace(tc.model, history_config=hc, use_history=True)
        model = nnx.eval_shape(model_cfg.create, jax.random.key(0))
        ad = AttnAdapter(_ModelShim(model), require_flags=False)
        obs = _robomme_obs_spec(model.config)
        lay = ad.layout(obs)
        steps = int(args.num_steps)
        out = jax.eval_shape(lambda s, r, o: ad.f_sample_attn(s, r, o, None, steps, lay["MEM"] + lay["IMG"], True),
                             ad.gstate, jax.random.key(0), obs)
        x_0, pa, ptxt, _ = out
        out_p = jax.eval_shape(lambda s, r, o: ad.f_sample_attn(s, r, o, None, steps, lay["MEM"] + lay["IMG"], True,
                                                                save_prefix=True),
                               ad.gstate, jax.random.key(0), obs)
        P, MEM = lay["P"], lay["MEM"]
        # S7 遮挡态：mask_keys（动态 bool[1,P]）+ 头遮挡（static）同时开，形状 / dtype 与关闭态一致
        heads = ((3, 0), (16, 0))
        out_a = jax.eval_shape(lambda s, r, o, mk: ad.f_sample_attn(s, r, o, None, steps, lay["MEM"] + lay["IMG"], True,
                                                                    mask_keys=mk, ablate_heads=heads),
                               ad.gstate, jax.random.key(0), obs, jax.ShapeDtypeStruct((1, P), jnp.bool_))
        ablate_same = all(tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype for a, b in zip(out, out_a))
        hol = head_out_leaves(ad.gstate)
        hol_ok = (len(hol) == 2 and all(len(sh) == 4 and sh[:2] == (N_LAYERS, ad.n_heads) for _, sh in hol))
        pp = out_p[4]
        want_pp = (N_LAYERS, 1, ad.n_heads, P, P)
        same_rest = all(tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype for a, b in zip(out, out_p[:4]))
        want_pa = (steps, N_LAYERS, 1, ad.n_heads, ad.AH, P + ad.AH)
        want_pt = (N_LAYERS, 1, ad.n_heads, lay["TXT"], P)
        want_x = (1, ad.AH, ad.AD)
        ok = (tuple(pa.shape) == want_pa and tuple(ptxt.shape) == want_pt and tuple(x_0.shape) == want_x
              and pa.dtype == jnp.float16 and ptxt.dtype == jnp.float16 and x_0.dtype == jnp.float32
              and lay["TXT"] == 64 and ad.AH == 20 and ad.n_heads == 8
              and tuple(pp.shape) == want_pp and pp.dtype == jnp.float16 and same_rest
              and ablate_same and hol_ok)
        ok_all &= ok
        print(f"DRY_ABLATE {tag} ablate_shapes_same={int(ablate_same)} head_out_leaves={len(hol)} "
              f"shapes={[sh for _, sh in hol]} ok={int(ablate_same and hol_ok)}", flush=True)
        la = (steps, N_LAYERS) + want_pa[3:]
        lt = (N_LAYERS,) + want_pt[2:]
        print(f"DRY_SHAPES {tag} ok={int(ok)} config={cfg_used} history={hc_src} "
              f"motion={int(bool(getattr(model.mem_encoder, 'motion_enabled', False)))} "
              f"P={P} MEM={MEM} IMG={lay['IMG']} TXT={lay['TXT']} n_img={lay['n_img']} "
              f"actions={tuple(x_0.shape)}/{x_0.dtype} last_attn.probs_action={la}/float16 "
              f"last_attn.probs_text={lt}/float16 last_attn.probs_prefix={(N_LAYERS,) + want_pp[2:]}/float16(save_prefix) "
              f"wall={time.perf_counter() - t0:.0f}s", flush=True)
        summary.append(f"{tag}:P={P},MEM={MEM}")
    print(f"ATTN_DRY_SHAPES={'PASS' if ok_all else 'FAIL'} {' '.join(summary)} num_steps={args.num_steps} L={N_LAYERS} AH=20")
    return 0 if ok_all else 1


# ══════════════════════════════════════════ selftest（GPU）══════════════════════════════════════════
def _legal_obs(policy, *, n_frames: int, seed: int = 0):
    """走生产 infer 的同一条装配链（add_buffer → _prepare_history → _input_transform → HistAugObservation），
    帧与状态为随机合法值；两种 checkpoint 都能用（motion 用本文件 _ZeroMotion 零句柄、维度取 checkpoint 的 motion.dim，attention 机制与 motion 内容无关）。
    分包与真实 eval 客户端同形：首包只喂第 0 帧（motion 开启时 FrameSampMemory.add_buffer 要求首包恰好
    exec_start_idx+1 帧，无 demo 即 1 帧），之后每包 16 帧（exec_start_idx=0 表示沿用）；基线无此约束、分包不影响结果。"""
    import jax
    import jax.numpy as jnp
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    rng = np.random.default_rng(seed)
    nv = int(policy.config.num_views)
    sd = int(policy.config.memory_feature.state.input_dim)
    frames = rng.integers(0, 256, size=(n_frames, nv, 256, 256, 3), dtype=np.uint8)
    wrists = rng.integers(0, 256, size=(n_frames, 256, 256, 3), dtype=np.uint8)
    states = rng.normal(0, 0.5, size=(n_frames, sd)).astype(np.float32)
    policy.reset()
    policy.add_buffer({"images": frames[:1], "state": states[:1], "exec_start_idx": 0})
    for s in range(1, n_frames, 16):
        policy.add_buffer({"images": frames[s:s + 16], "state": states[s:s + 16], "exec_start_idx": 0})
    t = n_frames - 1
    element = {"observation/image": frames[t, 0], "observation/wrist_image": wrists[t],
               "observation/state": states[t], "prompt": "press the button to reveal the cube"}
    inputs = policy._input_transform(policy._prepare_history(dict(element)))
    return HistAugObservation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs))


def _rowsum_err(pa, ptxt, text_valid) -> float:
    ea = float(np.abs(pa.astype(np.float64).sum(-1) - 1.0).max())
    s = ptxt.astype(np.float64).sum(-1)                           # (L,b,G,TXT)
    tv = np.asarray(text_valid, bool)[None, :, None, :]           # (1,b,1,TXT)
    et = float(np.abs(np.where(tv, s - 1.0, 0.0)).max())
    return max(ea, et)


def _uniform_err(pa, ptxt, masks) -> tuple[float, float]:
    """q 置零后：合法 key 上 max−min（应 ≈0）与非法 key 上最大概率（应为 0）。"""
    act = np.asarray(masks["action"], bool)                       # (b,AH,P+AH)
    txt = np.asarray(masks["text"], bool)                         # (b,TXT,P)
    tv = np.asarray(masks["text_valid"], bool)                    # (b,TXT)
    spread, illegal = 0.0, 0.0
    a = pa.astype(np.float64)                                     # (S,L,b,G,AH,K)
    m = act[None, None, :, None]
    hi = np.where(m, a, -np.inf).max(-1); lo = np.where(m, a, np.inf).min(-1)
    spread = max(spread, float((hi - lo).max())); illegal = max(illegal, float(np.where(m, 0.0, a).max()))
    t = ptxt.astype(np.float64)                                   # (L,b,G,TXT,P)
    mt = (txt & tv[:, :, None])[None, :, None]
    rows = np.broadcast_to(tv[None, :, None, :], t.shape[:-1])
    hi = np.where(mt, t, -np.inf).max(-1); lo = np.where(mt, t, np.inf).min(-1)
    spread = max(spread, float(np.where(rows, hi - lo, 0.0).max())); illegal = max(illegal, float(np.where(mt, 0.0, t).max()))
    return spread, illegal


class _ZeroMotion:
    """自检用零向量编码句柄（与 MotionEncoderClient 同调用约定：(frames, start_frame) -> (dim,) f32，有 close()）。

    输出维度 `dim` 在 policy 构造后取自 `policy._motion_cfg["dim"]`——即 FrameSampMemory 校验用 motion_dim 的同源值
    （run 的 history_config.resolved.yaml 里 motion.dim）；未赋值即调用会直接报错。
    """

    def __init__(self):
        self.calls = 0
        self.dim: int | None = None

    def __call__(self, frames, start_frame: int = 0):
        self.calls += 1
        return np.zeros(self.dim, np.float32)

    def close(self) -> None:
        pass


def selftest(args) -> int:
    import gc
    import jax
    import jax.numpy as jnp
    from mme_vla_suite.training import config as _config
    from openpi.models import model as _model
    from openpi.shared import nnx_utils
    from mv_model_adapter import build_policy

    C.require_gpu()
    C.require_det_flags()
    os.chdir(C.REPO_ROOT)          # train_config.data.assets.assets_dir 是仓库相对路径
    ckpt = pathlib.Path(args.ckpt).resolve()
    t0 = time.perf_counter()
    train_config = _config.get_config(args.config)
    zm_calls = []

    def factory(**kw):
        zm = _ZeroMotion()
        zm_calls.append(zm)
        return zm

    policy = build_policy(train_config, ckpt, seed=42, motion_factory=factory)
    motion = bool(policy.motion_enabled)
    motion_dim = int(policy._motion_cfg["dim"]) if motion else 0
    for zm in zm_calls:
        zm.dim = motion_dim
    print(f"[attn] policy 构造 {time.perf_counter() - t0:.1f}s ckpt={ckpt} config={args.config} motion={int(motion)}", flush=True)
    ad = AttnAdapter(policy)
    obs = _legal_obs(policy, n_frames=int(args.n_frames))
    lay = ad.layout(obs)
    P, MEM, txt_lo = lay["P"], lay["MEM"], lay["MEM"] + lay["IMG"]
    print(f"[attn] layout P={P} MEM={MEM} IMG={lay['IMG']} TXT={lay['TXT']} n_img={lay['n_img']} "
          f"motion_windows={int(np.asarray(obs.motion_mask).sum()) if obs.motion_mask is not None else 0}", flush=True)
    key0 = jax.random.key(0)
    noise = jax.random.normal(jax.random.key(1), (1, ad.AH, ad.AD), dtype=jnp.float32)

    # ① bf16：生产 _sample_actions vs 展开版（观察项），以及接口形状
    a_prod = np.asarray(jax.block_until_ready(policy._sample_actions(key0, obs, noise=noise)))
    fn = ad.make_sample_fn()
    a_mine = fn(key0, obs, noise=noise)
    a_mine_np = np.asarray(jax.block_until_ready(a_mine))
    la = ad.last_attn
    shape_ok = (a_mine_np.shape == a_prod.shape and a_mine_np.dtype == a_prod.dtype
                and la["probs_action"].shape == (DEFAULT_NUM_STEPS, N_LAYERS, ad.n_heads, ad.AH, P + ad.AH)
                and la["probs_text"].shape == (N_LAYERS, ad.n_heads, lay["TXT"], P)
                and la["probs_action"].dtype == np.float16 and la["probs_text"].dtype == np.float16
                and la["P"] == P and la["MEM"] == MEM)
    acc = C.DiffAcc(); acc.add(a_prod, a_mine_np)
    bf16_relfro = acc.summary()["rel_fro"]
    # noise=None 路径与生产同法生成噪声
    a_prod_n = np.asarray(jax.block_until_ready(policy._sample_actions(key0, obs)))
    a_mine_n = np.asarray(jax.block_until_ready(fn(key0, obs)))
    acc_n = C.DiffAcc(); acc_n.add(a_prod_n, a_mine_n)
    print(f"[attn] bf16 prod_vs_unrolled rel_fro={bf16_relfro:.4g} max_abs={acc.summary()['max_abs']:.4g} | "
          f"noise=None rel_fro={acc_n.summary()['rel_fro']:.4g} | shape_ok={int(shape_ok)}", flush=True)

    # ② rowsum：f32 概率（out_f16=False）每行求和 = 1；另记 f16 落盘口径
    masks = jax.device_get(ad.f_masks(ad.gstate, obs, txt_lo))
    _, pa32, pt32, _, _ = ad.run(key0, obs, noise=noise, out_f16=False)
    pa32, pt32 = jax.device_get((pa32, pt32))
    rowsum_err = _rowsum_err(pa32, pt32, masks["text_valid"])
    rowsum16 = _rowsum_err(la["probs_action"][:, :, None], la["probs_text"][:, None], masks["text_valid"])
    del pa32, pt32

    # ③ uniform：把 llm 所有 q_einsum 参数置零 → logits 恒 0 → 合法 key 上均匀、非法 key 为 0
    n_zero = [0]

    def zq(path, x):
        ks = jax.tree_util.keystr(path)
        if "llm" in ks and "q_einsum" in ks:
            n_zero[0] += 1
            return jnp.zeros_like(x)
        return x

    st_q0 = jax.tree_util.tree_map_with_path(zq, ad.gstate)
    _, pa_u, pt_u, _, _ = ad.run(key0, obs, noise=noise, out_f16=False, state=st_q0)
    pa_u, pt_u = jax.device_get((pa_u, pt_u))
    spread, illegal = _uniform_err(pa_u, pt_u, masks)
    uniform_ok = int(n_zero[0] > 0 and spread <= 1e-5 and illegal <= 1e-6)
    del st_q0, pa_u, pt_u

    # ③′ save_prefix：f32 口径 probs_prefix 有效行行和 = 1、padding 行为 0；开启/关闭两态的其余输出逐位比对（f32 与 f16 落盘口径）
    _, pa32b, pt32b, _, _ = ad.run(key0, obs, noise=noise, out_f16=False)
    _, pa32p, pt32p, _, _, pp32 = ad.run(key0, obs, noise=noise, out_f16=False, save_prefix=True)
    pa32b, pt32b, pa32p, pt32p, pp32 = jax.device_get((pa32b, pt32b, pa32p, pt32p, pp32))
    pv = np.asarray(masks["prefix_valid"], bool)                  # (b,P)
    ps = pp32.astype(np.float64).sum(-1)                          # (L,b,G,P)
    prefix_rowsum_err = float(np.abs(np.where(pv[None, :, None, :], ps - 1.0, 0.0)).max())
    prefix_pad_max = float(np.abs(np.where(pv[None, :, None, :, None], 0.0, pp32)).max())
    f32_same = int(np.array_equal(pa32b, pa32p) and np.array_equal(pt32b, pt32p))
    del ps, pp32
    x_off, pa_off, pt_off, _, _ = ad.run(key0, obs, noise=noise)
    x_on, pa_on, pt_on, _, _, pp_on = ad.run(key0, obs, noise=noise, save_prefix=True)
    pp_shape, pp_dtype = tuple(pp_on.shape), pp_on.dtype
    x_off, pa_off, pt_off, x_on, pa_on, pt_on = jax.device_get((x_off, pa_off, pt_off, x_on, pa_on, pt_on))
    del pp_on

    def _maxd(a, b):
        return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())

    d_x, d_pa, d_pt = _maxd(x_off, x_on), _maxd(pa_off, pa_on), _maxd(pt_off, pt_on)
    d_pa32, d_pt32 = _maxd(pa32b, pa32p), _maxd(pt32b, pt32p)
    del pa32b, pt32b, pa32p, pt32p
    prefix_same = int(np.array_equal(x_off, x_on) and np.array_equal(pa_off, pa_on) and np.array_equal(pt_off, pt_on)
                      and f32_same == 1)
    prefix_shape_ok = int(pp_shape == (N_LAYERS, 1, ad.n_heads, P, P) and pp_dtype == jnp.float16)
    print(f"[attn] save_prefix rowsum_err_f32={prefix_rowsum_err:.3g} pad_rows_max={prefix_pad_max:.3g} "
          f"shape={pp_shape}/{pp_dtype} on_vs_off max_abs f16: actions={d_x:.3g} probs_action={d_pa:.3g} "
          f"probs_text={d_pt:.3g} | f32: probs_action={d_pa32:.3g} probs_text={d_pt32:.3g} same={prefix_same}", flush=True)
    print(f"[attn] rowsum_err_f32={rowsum_err:.3g} rowsum_err_f16={rowsum16:.3g} uniform spread={spread:.3g} "
          f"illegal_max={illegal:.3g} q_einsum_leaves_zeroed={n_zero[0]}", flush=True)

    # ⑤ 因果遮挡（S7）：mask 预设按服务端同一函数从观测算出（motion → mask_motion，基线 → mask_vis）
    if str(_HERE.parents[1]) not in sys.path:
        sys.path.insert(0, str(_HERE.parents[1]))
    from scripts.interpret.serve_policy_attn import _pad_times, memory_key_mask, parse_ablate
    cfgp = policy.config
    tpf = int(cfgp.token_per_image) * int(cfgp.num_views)
    budget = int(cfgp.budget)
    static = np.asarray(obs.static_mask).reshape(-1).astype(bool)
    if motion:
        mem_order_np = np.asarray(obs.mem_order).reshape(-1).astype(np.int64)
        valid = np.concatenate([static, np.asarray(obs.motion_mask).reshape(-1).astype(bool)])[mem_order_np]
    else:
        mem_order_np = np.arange(MEM, dtype=np.int64)
        valid = static
    frames = policy.mem_buffer.get_frame_sampling_indices(int(policy.step_idx), budget, int(cfgp.token_per_image))
    ft = _pad_times(frames, budget // tpf)
    mask_kind = "mask_motion" if motion else "mask_vis"
    mk = np.zeros(P, bool)
    mk[:MEM] = memory_key_mask({mask_kind}, mem_order_np, ft, budget=budget, tokens_per_frame=tpf, valid=valid)
    n_mk = int(mk.sum())
    heads = parse_ablate(f"heads:{args.ablate_heads}")["heads"]
    # ① 遮挡全关 ≡ 原路径：空 mask（全 False）走门控分支、空头元组，动作与概率逐位一致
    x_z, pa_z, pt_z, _, _ = ad.run(key0, obs, noise=noise, mask_keys=np.zeros(P, bool), ablate_heads=())
    x_z, pa_z, pt_z = jax.device_get((x_z, pa_z, pt_z))
    ablate_off_same = int(np.array_equal(x_off, x_z) and np.array_equal(pa_off, pa_z) and np.array_equal(pt_off, pt_z)
                          and np.array_equal(a_mine_np, np.asarray(x_off)))
    del pa_z, pt_z
    # ② mask_keys：被遮列在 probs_action 与 probs_prefix（非被遮且有效的 query 行）上质量为 0，行和仍为 1（f32）
    _, pa_m, _, _, _, pp_m = ad.run(key0, obs, noise=noise, out_f16=False, save_prefix=True, mask_keys=mk)
    pa_m, pp_m = jax.device_get((pa_m, pp_m))
    rows = pv[0] & ~mk
    mass_a = float(pa_m[..., :P][..., mk].max()) if n_mk else 0.0
    pp0 = pp_m[:, 0]                                              # (L,G,P,P)
    mass_p = float(pp0[:, :, rows][..., mk].max()) if n_mk else 0.0
    masked_cols_mass = max(mass_a, mass_p)
    masked_rowsum = max(float(np.abs(pa_m.astype(np.float64).sum(-1) - 1.0).max()),
                        float(np.abs(pp0[:, :, pv[0]].astype(np.float64).sum(-1) - 1.0).max()))
    del pa_m, pp_m, pp0
    # ④ 遮挡后动作确实改变（bf16 生产口径）
    x_mb = np.asarray(jax.device_get(ad.run(key0, obs, noise=noise, mask_keys=mk)[0]))
    x_hb = np.asarray(jax.device_get(ad.run(key0, obs, noise=noise, ablate_heads=heads)[0]))
    d_mask, d_head = _maxd(x_off, x_mb), _maxd(x_off, x_hb)
    print(f"[attn] ablate {mask_kind} n_masked_keys={n_mk} masked_cols_mass={masked_cols_mass:.3g} "
          f"masked_rowsum_err={masked_rowsum:.3g} off_same={ablate_off_same} | actions max_abs vs off: "
          f"mask={d_mask:.4g} heads{list(heads)}={d_head:.4g}", flush=True)

    # ④ f32 语义闸：参数与 embed_dtype 升 f32、matmul_precision=highest，生产 nn.scan 路径 vs 展开版
    t1 = time.perf_counter()
    run_root = ckpt.parent
    params_f32 = _model.restore_params(ckpt / "params", dtype=jnp.float32)
    cfg32 = dataclasses.replace(policy._model.config, dtype="float32")
    strict = (run_root / "history_config.resolved.yaml").is_file()
    model32 = cfg32.load(params_f32, remove_extra_params=not strict)
    del params_f32
    gc.collect()
    ad32 = AttnAdapter(_ModelShim(model32))
    prod32 = nnx_utils.module_jit(model32.sample_actions)
    with jax.default_matmul_precision("highest"):
        b32_prod = np.asarray(jax.block_until_ready(prod32(key0, obs, noise=noise)))
        x32, _, _, _, _ = ad32.run(key0, obs, noise=noise, out_f16=True)
        b32_mine = np.asarray(jax.block_until_ready(x32))
        # ③ 头遮挡 ≡ 把该层该头的 attn_vec_einsum 权重行置零（两路 expert）：encoded 该头贡献为 0 的独立对照
        st_w, n_w = zero_head_out_weights(ad32.gstate, heads)
        x_ab = np.asarray(jax.block_until_ready(ad32.run(key0, obs, noise=noise, ablate_heads=heads)[0]))
        x_wz = np.asarray(jax.block_until_ready(ad32.run(key0, obs, noise=noise, state=st_w)[0]))
        del st_w
    f32_maxabs = float(np.abs(b32_prod.astype(np.float64) - b32_mine.astype(np.float64)).max())
    head_vs_weightzero = _maxd(x_ab, x_wz)
    head_effect32 = _maxd(x_ab, b32_mine)
    head_zero_ok = int(n_w == 2 and head_vs_weightzero <= 1e-5)
    print(f"[attn] f32 prod_vs_unrolled max_abs={f32_maxabs:.3g} strict_load={int(strict)} wall={time.perf_counter() - t1:.0f}s", flush=True)
    print(f"[attn] f32 heads{list(heads)} ablate_vs_weightzero max_abs={head_vs_weightzero:.3g} "
          f"(attn_vec_einsum 叶子 {n_w}) ablate_vs_off max_abs={head_effect32:.4g}", flush=True)

    ablate_ok = (ablate_off_same == 1 and n_mk > 0 and masked_cols_mass <= 1e-7 and masked_rowsum <= 1e-4
                 and head_zero_ok == 1 and d_mask > 0 and d_head > 0)
    ok = (shape_ok and f32_maxabs <= 1e-5 and rowsum_err <= 1e-4 and uniform_ok == 1
          and np.all(np.isfinite(a_mine_np))
          and prefix_rowsum_err <= 1e-4 and prefix_pad_max == 0.0 and prefix_same == 1 and prefix_shape_ok == 1
          and ablate_ok)
    note = f"motion=Zero{motion_dim}(免sidecar,attention机制与motion内容无关)" if motion else "motion=off"
    print(f"ATTN_SELFTEST={'PASS' if ok else 'FAIL'} f32_maxabs={f32_maxabs:.3g} bf16_relfro={bf16_relfro:.3g} "
          f"rowsum_err={rowsum_err:.3g} uniform_ok={uniform_ok} P={P} MEM={MEM} "
          f"shape_ok={int(shape_ok)} rowsum_err_f16={rowsum16:.3g} {note} wall={time.perf_counter() - t0:.0f}s "
          f"prefix_rowsum_err={prefix_rowsum_err:.3g} prefix_same_actions={prefix_same} "
          f"ablate_off_same={ablate_off_same} masked_cols_mass={masked_cols_mass:.3g} head_zero_ok={head_zero_ok} "
          f"ablate_mask={mask_kind} n_masked_keys={n_mk} ablate_mask_maxabs={d_mask:.3g} ablate_head_maxabs={d_head:.3g}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-shapes", action="store_true")
    ap.add_argument("--ckpt", default=None, help="--selftest 的 checkpoint 目录（…/<step>）")
    ap.add_argument("--config", default=None, help="--selftest 的 train config 名")
    ap.add_argument("--n-frames", type=int, default=49, help="selftest 喂入的随机帧数（49 帧 → motion 侧 2 个窗口）")
    ap.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    ap.add_argument("--ablate-heads", default="L3H0,L16H0", help="selftest 头遮挡用的头（L<层>H<头>,...）")
    ap.add_argument("--official-ckpt", default=str(C.V1 / "models/official-mme-vla/perceptual-framesamp-context/79999"))
    ap.add_argument("--official-config", default="mme_vla_suite_b128")
    ap.add_argument("--motion-ckpt", default=str(C.V1 / "assets/vla-o16cc-59999/59999"))
    ap.add_argument("--motion-config", default="mme_vla_suite_orig16_b128_60k")
    args = ap.parse_args()
    if args.dry_shapes:
        if os.environ.get("JAX_PLATFORMS", "") != "cpu":
            C.require_det_flags()        # CPU dry-shapes 豁免确定性开关；非 CPU 仍硬闸
        return dry_shapes(args)
    if args.selftest:
        if not args.ckpt or not args.config:
            ap.error("--selftest 需要 --ckpt 与 --config")
        return selftest(args)
    ap.error("只支持 --selftest / --dry-shapes（适配器由 serve_policy_attn.py import 使用）")
    return 2


if __name__ == "__main__":
    sys.exit(main())
