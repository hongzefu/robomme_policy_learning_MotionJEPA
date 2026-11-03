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
from mv_model_adapter import _capture_attn, _guard_model_source, _make_patched_attention  # noqa: E402

N_LAYERS = 18
DEFAULT_NUM_STEPS = 10
# RoboMME 推理只喂两路图像（robomme_policy.py 去掉了 right_wrist_0_rgb），dry-shapes 的抽象观测与之一致
ROBOMME_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb")


class _ModelShim:
    """只带 `_model` 的替身（dry-shapes 与 f32 自检用），与 MME_VLA_Policy 暴露同一属性。"""

    def __init__(self, model):
        self._model = model


class AttnAdapter:
    def __init__(self, policy, *, require_flags: bool = True):
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

        def block_apply(layers, l, xs, kv, positions, mask3d, adarms):
            p_l = jax.tree.map(lambda x: x[l], layers)
            with _capture_attn(hg, patched) as cap:
                out, kv_out = block.apply({"params": p_l}, xs, kv, positions, mask3d[:, None, :, :],
                                          adarms, None, None, True)
                probs = cap[0]
            return out, kv_out, probs

        @functools.partial(jax.jit, static_argnames=("num_steps", "txt_lo", "out_f16"))
        def f_sample_attn(state, rng, obs, noise, num_steps, txt_lo, out_f16=True):
            """逐字复刻 HistoryPi0.sample_actions（context 分支）的 18 层展开版；返回 (x_0, probs_action, probs_text)。
            probs_action (num_steps, L, b, G, AH, P+AH)；probs_text (L, b, G, TXT, P)，padding 文本行置 0。"""
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
            pos = jnp.cumsum(pm, axis=1) - 1
            P = pt.shape[1]
            h = pt.astype(ed)
            ks, vs, txt = [], [], []
            for l in range(L):
                (h, _), (k, v), probs = block_apply(layers, l, [h, None], None, pos, base, [None, None])
                ks.append(k); vs.append(v)
                txt.append(probs[:, 0, :, txt_lo:P, :])                  # (b,G,TXT,P)
            K = jnp.stack(ks); V = jnp.stack(vs)                         # (L,b,P,1,H)
            txt_valid = pm[:, txt_lo:P].astype(jnp.float32)              # padding 文本 query 行（全遮蔽 → softmax 均匀）置 0
            probs_text = jnp.stack(txt) * txt_valid[None, :, None, :, None]
            spos_base = jnp.sum(pm, axis=-1)[:, None]

            def step(carry):
                x_t, time, i, acc = carry
                st, sm, sar, _, ad = m.embed_suffix(o, x_t, jnp.broadcast_to(time, b))
                sattn = make_attn_mask(sm, sar)
                prefix_attn = einops.repeat(pm, "b p -> b s p", s=st.shape[1])
                full = jnp.concatenate([prefix_attn, sattn], axis=-1)
                spos = spos_base + jnp.cumsum(sm, axis=-1) - 1
                hs = st.astype(ed)
                rows = []
                for l in range(L):
                    (_, hs), _, probs = block_apply(layers, l, [None, hs], (K[l], V[l]), spos, full, [None, ad])
                    rows.append(probs[:, 0])                             # (b,G,AH,P+AH)：不平均，原样保留
                hs, _ = final_norm_1.apply({"params": fn1}, hs, ad)
                v_t = m.action_out_proj(hs[:, -AH:])
                return x_t + dt * v_t, time + dt, i + 1, acc.at[i].set(jnp.stack(rows))

            def cond(carry):
                return carry[1] >= -dt / 2

            acc0 = jnp.zeros((num_steps, L, b, cfgs[0].num_heads, AH, P + AH), jnp.float32)
            x_0, _, n_done, acc = jax.lax.while_loop(cond, step, (noise, 1.0, jnp.int32(0), acc0))
            if out_f16:
                return x_0, acc.astype(jnp.float16), probs_text.astype(jnp.float16), n_done
            return x_0, acc, probs_text, n_done

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
            return {"text": base[:, txt_lo:], "text_valid": pm[:, txt_lo:], "action": full}

        self.f_sample_attn = f_sample_attn
        self.f_masks = f_masks

    def run(self, rng, observation, *, noise=None, num_steps: int = DEFAULT_NUM_STEPS, out_f16: bool = True, state=None):
        """调一次展开版；返回 (x_0, probs_action, probs_text, layout)，probs 仍在 device 上、保留 batch 轴。"""
        lay = self.layout(observation)
        st = self.gstate if state is None else state
        x_0, pa, ptxt, n_done = self.f_sample_attn(st, rng, observation, noise, int(num_steps),
                                                   lay["MEM"] + lay["IMG"], out_f16)
        return x_0, pa, ptxt, n_done, lay

    def make_sample_fn(self):
        """返回可顶掉 policy._sample_actions 的可调用：fn(rng, observation, **kw) -> actions（与生产同形同 dtype）。
        每次调用后 self.last_attn 存本次推理的原始概率（batch=1，去 batch 轴，f16，不平均）。"""
        jax = self.jax

        def fn(rng, observation, **kw):
            extra = set(kw) - {"noise", "num_steps"}
            if extra:
                raise TypeError(f"AttnAdapter sample_fn 不支持的参数 {sorted(extra)}")
            num_steps = int(kw.get("num_steps", DEFAULT_NUM_STEPS))
            if observation.state.shape[0] != 1:
                raise ValueError(f"只支持 batch=1，得到 {observation.state.shape[0]}")
            x_0, pa, ptxt, n_done, lay = self.run(rng, observation, noise=kw.get("noise"), num_steps=num_steps)
            pa_h, ptxt_h, n_h = jax.device_get((pa, ptxt, n_done))
            if int(n_h) != num_steps:
                raise RuntimeError(f"去噪循环实际步数 {int(n_h)} != num_steps {num_steps}")
            self.last_attn = {"probs_action": np.asarray(pa_h[:, :, 0]),       # (num_steps, L, G, AH, P+AH) f16
                              "probs_text": np.asarray(ptxt_h[:, 0]),          # (L, G, TXT, P) f16
                              "P": int(lay["P"]), "MEM": int(lay["MEM"])}
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
        P, MEM = lay["P"], lay["MEM"]
        want_pa = (steps, N_LAYERS, 1, ad.n_heads, ad.AH, P + ad.AH)
        want_pt = (N_LAYERS, 1, ad.n_heads, lay["TXT"], P)
        want_x = (1, ad.AH, ad.AD)
        ok = (tuple(pa.shape) == want_pa and tuple(ptxt.shape) == want_pt and tuple(x_0.shape) == want_x
              and pa.dtype == jnp.float16 and ptxt.dtype == jnp.float16 and x_0.dtype == jnp.float32
              and lay["TXT"] == 64 and ad.AH == 20 and ad.n_heads == 8)
        ok_all &= ok
        la = (steps, N_LAYERS) + want_pa[3:]
        lt = (N_LAYERS,) + want_pt[2:]
        print(f"DRY_SHAPES {tag} ok={int(ok)} config={cfg_used} history={hc_src} "
              f"motion={int(bool(getattr(model.mem_encoder, 'motion_enabled', False)))} "
              f"P={P} MEM={MEM} IMG={lay['IMG']} TXT={lay['TXT']} n_img={lay['n_img']} "
              f"actions={tuple(x_0.shape)}/{x_0.dtype} last_attn.probs_action={la}/float16 "
              f"last_attn.probs_text={lt}/float16 wall={time.perf_counter() - t0:.0f}s", flush=True)
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
    print(f"[attn] rowsum_err_f32={rowsum_err:.3g} rowsum_err_f16={rowsum16:.3g} uniform spread={spread:.3g} "
          f"illegal_max={illegal:.3g} q_einsum_leaves_zeroed={n_zero[0]}", flush=True)

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
    f32_maxabs = float(np.abs(b32_prod.astype(np.float64) - b32_mine.astype(np.float64)).max())
    print(f"[attn] f32 prod_vs_unrolled max_abs={f32_maxabs:.3g} strict_load={int(strict)} wall={time.perf_counter() - t1:.0f}s", flush=True)

    ok = (shape_ok and f32_maxabs <= 1e-5 and rowsum_err <= 1e-4 and uniform_ok == 1
          and np.all(np.isfinite(a_mine_np)))
    note = f"motion=Zero{motion_dim}(免sidecar,attention机制与motion内容无关)" if motion else "motion=off"
    print(f"ATTN_SELFTEST={'PASS' if ok else 'FAIL'} f32_maxabs={f32_maxabs:.3g} bf16_relfro={bf16_relfro:.3g} "
          f"rowsum_err={rowsum_err:.3g} uniform_ok={uniform_ok} P={P} MEM={MEM} "
          f"shape_ok={int(shape_ok)} rowsum_err_f16={rowsum16:.3g} {note} wall={time.perf_counter() - t0:.0f}s")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-shapes", action="store_true")
    ap.add_argument("--ckpt", default=None, help="--selftest 的 checkpoint 目录（…/<step>）")
    ap.add_argument("--config", default=None, help="--selftest 的 train config 名")
    ap.add_argument("--n-frames", type=int, default=49, help="selftest 喂入的随机帧数（49 帧 → motion 侧 2 个窗口）")
    ap.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
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
