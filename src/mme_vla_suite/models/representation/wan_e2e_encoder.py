"""E2E-Enc：可训练 Wan latent motion encoder 的 flax nnx 实现（计划 1117-e2e-enc-train-plan 一节、二节 S1 行）。

规格源只用 MotionJEPA d3213de 的 ``src/motion_jepa/models/wan_latent_encoder.py`` 与
``wan_latent_dit.py`` 的 ``LatentAffineMixin``（vendored 副本：``scripts/training/g0/ref_torch/wan_latent_encoder_d3213de.py``）；
禁止参考 MotionJEPA HEAD（483c23e，那是另一架构）。逐项对应：

- 输入 ``(n, 9, 16384)``，bf16（训练/在线）或 f32（G1 对拍）；16384 = C·H·W 按 ``1024c + 32h + w`` 展平
  （与 ``shared/wan_raw9.flatten_raw9`` 及 d3213de ``zn.flatten(2)`` 同序）。
- 标准化：第一步一律 ``astype(float32)``，在 f32 里做 ``(z - mean) / std``（逐通道 16 个常数），再 cast 到
  ``config.dtype``。常数是**非状态的 fp32 常量**：以 Python float 元组存为静态属性，不是 ``nnx.Variable`` /
  ``nnx.Param``，不进参数树、不进 EMA、不被 bf16 恢复舍入。值逐位等于 P3 checkpoint
  (``checkpoint_epoch_72.pt#encoder``) 的 ``latents_mean`` / ``latents_std`` buffer；与 Wan VAE config 的
  四位小数常数只差舍入级（max|Δmean| = 4.17e-7、max|Δstd| = 3.22e-5，计划 P8）。
- 共享 ``input_proj``: Linear(16384→768, 无 bias)。
- ``layers`` 个 pre-norm 因果块（深度为配置项；正式训练 YAML = 1，G1 第二次对拍 / G3 临时 8）：
  ``x + attn(norm1(x))``、``x + ffn(norm2(x))``，无 layer scale；RMSNorm eps 1e-6；
  注意力 to_qkv Linear(768→2304)、to_out Linear(768→768)、12 头 × 64、因果、无 qk-norm；
  1D RoPE base 10000、位置 0..8、只作用于 q/k、rotate-half ``[x1·cos − x2·sin, x2·cos + x1·sin]``；
  SwiGLU hidden = ``round_up8(int(2·768·4/3))`` = 2048，``down(silu(gate(x)) * up(x))``，全部无 bias；
  dropout 只在 to_out 与 down_proj 之后（计划 D2 定为 0）。
- 读出 ``x[:, -1]`` → final RMSNorm → output_proj Linear(768→768, 无 bias) → ``(n, 768)``。
- RMSNorm 运算顺序同 d3213de：``(x32 * rsqrt(mean(x32²) + eps)).astype(x.dtype) * weight``。
- 精度：参数 ``param_dtype``（f32）、计算 ``dtype``（训练为 bf16）；RoPE 与 softmax 在 f32 里算。
- 显式初始化：所有 Linear 用 ``variance_scaling(1/3, "fan_in", "uniform")``（≡ U(±1/√fan_in)，即 PyTorch
  ``nn.Linear`` 默认），RMSNorm weight = 1；不用 flax 默认的 lecun_normal。
- 构造参数接收独立的 ``nnx.Rngs``（R7：由调用方——S3 的 PerceptualMemory——两级 fold_in 派生后传入）。

参数量：depth=1 为 20,252,928；depth=8 为 69,808,896（= P3 checkpoint 规格）。

另附只依赖 NumPy 的 PyTorch state dict → nnx 键映射（``load_torch_state_dict``），供 G1/G1c/G3 载入
同一份权重：RoPE ``cos_cache``/``sin_cache`` buffer 丢弃（JAX 端现算），``latents_mean``/``latents_std`` 与
本模块常数逐位核对，未映射或多余的键一律 raise。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# ── 常量 ────────────────────────────────────────────────────────────────────────────────────────
E2E_SEQ_LEN = 9
E2E_LATENT_CHANNELS = 16
E2E_LATENT_SIZE = 32
E2E_RAW_DIM = E2E_LATENT_CHANNELS * E2E_LATENT_SIZE * E2E_LATENT_SIZE  # 16384
E2E_WIDTH = 768
E2E_HEADS = 12
E2E_DIM_HEAD = 64
E2E_MLP_RATIO = 4
E2E_ROPE_BASE = 10000.0
E2E_RMS_EPS = 1e-6
E2E_PARAMS_DEPTH1 = 20_252_928
E2E_PARAMS_DEPTH8 = 69_808_896

# P3 checkpoint encoder buffer 的 fp32 位模式（逐位；见模块 docstring）。
_LATENTS_MEAN_BITS = (
    0xBF41D14E, 0xBF357A78, 0xBF694AF5, 0x3DDC28F6, 0xBE32B021, 0x3F771DE7, 0xBE1B573F, 0x3FC6809D,
    0x3ED3A92A, 0xBD926ED0, 0x3F0D3C36, 0xBEB9F55A, 0xBE44D014, 0xBF731F8A, 0x3E802752, 0xBE958E22,
)
_LATENTS_STD_BITS = (
    0x403460AA, 0x3FBA1FF3, 0x4014F5C3, 0x4029F8A1, 0x3F9C1BDA, 0x3FE2AA24, 0x4026BB99, 0x4004C155,
    0x40513261, 0x4009C433, 0x40375FF7, 0x3FC7697D, 0x3FD1B08A, 0x3F9009D5, 0x4034CE70, 0x3FF54003,
)
LATENTS_MEAN = tuple(float(v) for v in np.array(_LATENTS_MEAN_BITS, np.uint32).view(np.float32))
LATENTS_STD = tuple(float(v) for v in np.array(_LATENTS_STD_BITS, np.uint32).view(np.float32))
# Wan2.1 VAE config.json 的四位小数常数（D1 的出处；仅用于核对 P8「舍入级一致」，不参与计算）
VAE_CONFIG_LATENTS_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
VAE_CONFIG_LATENTS_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
)

linear_init = nnx.initializers.variance_scaling(1.0 / 3.0, "fan_in", "uniform")  # ≡ U(±1/√fan_in)


def swiglu_hidden(dim: int, mlp_ratio: int | float) -> int:
    """d3213de ``SwiGLU.__init__``：``int(2·dim·mlp_ratio/3)`` 向上取整到 8 的倍数（768, 4 → 2048）。"""
    hidden = int(2 * dim * mlp_ratio / 3)
    return ((hidden + 7) // 8) * 8


def _to_dtype(d) -> Any:
    if d is None:
        return None
    if isinstance(d, str):
        return jnp.dtype(d)
    return d


@dataclasses.dataclass(frozen=True)
class WanE2EEncoderConfig:
    """encoder 超参；与 YAML ``motion.encoder`` 节同名（``layers/width/heads/mlp_ratio/dropout/norm``）。"""

    layers: int = 1
    width: int = E2E_WIDTH
    heads: int = E2E_HEADS
    mlp_ratio: int | float = E2E_MLP_RATIO
    dropout: float = 0.0
    norm: str = "vae_config"
    dim_head: int = E2E_DIM_HEAD
    motion_dim: int = E2E_WIDTH
    seq_len: int = E2E_SEQ_LEN
    dtype: Any = jnp.bfloat16

    @classmethod
    def from_any(cls, cfg=None, **overrides) -> "WanE2EEncoderConfig":
        """从 dict / OmegaConf DictConfig / 带属性对象 / 本类实例构造；``overrides`` 优先。未知键 raise。"""
        if cfg is None:
            d: dict = {}
        elif isinstance(cfg, cls):
            d = dataclasses.asdict(cfg)
        elif isinstance(cfg, Mapping) or hasattr(cfg, "items"):
            d = {str(k): v for k, v in cfg.items()}
        else:
            d = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cls) if hasattr(cfg, f.name)}
        d.update({k: v for k, v in overrides.items() if v is not None})
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown:
            raise ValueError(f"WanE2EEncoderConfig 未知键：{unknown}")
        out = cls(**d)
        if out.norm != "vae_config":
            raise ValueError(f"encoder.norm 只支持 vae_config，得到 {out.norm!r}")
        if int(out.layers) < 1:
            raise ValueError(f"encoder.layers 必须 ≥1，得到 {out.layers}")
        if int(out.heads) * int(out.dim_head) != int(out.width):
            raise ValueError(f"heads×dim_head={out.heads}×{out.dim_head} != width={out.width}")
        return dataclasses.replace(out, dtype=_to_dtype(out.dtype))


def rope_cos_sin(seq_len: int, dim_head: int, base: float = E2E_ROPE_BASE):
    """d3213de ``RotaryEmbedding1D``：f32 的 cos/sin，形状 (seq_len, dim_head//2)。"""
    freqs = 1.0 / (base ** (jnp.arange(0, dim_head, 2, dtype=jnp.float32) / dim_head))
    angles = jnp.arange(seq_len, dtype=jnp.float32)[:, None] * freqs[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x, cos, sin):
    """x: (n, L, h, d)，在 f32 里做 rotate-half；cos/sin: (L, d/2)。"""
    x = x.astype(jnp.float32)
    x1, x2 = jnp.split(x, 2, axis=-1)
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


class RMSNorm(nnx.Module):
    """d3213de RMSNorm：f32 里归一化 → cast 回输入 dtype → 乘 weight（weight 初值 1、param_dtype）。

    ``stop_grad_denominator`` 只供 G1c 负例使用（给分母 r 加 stop_gradient），训练路径恒为 False。
    """

    def __init__(self, dim: int, *, eps: float = E2E_RMS_EPS, param_dtype=jnp.float32):
        self.weight = nnx.Param(jnp.ones((dim,), dtype=param_dtype))
        self.eps = float(eps)
        self.stop_grad_denominator = False

    def __call__(self, x):
        x32 = x.astype(jnp.float32)
        r = jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + self.eps)
        if self.stop_grad_denominator:
            r = jax.lax.stop_gradient(r)
        return (x32 * r).astype(x.dtype) * self.weight.value


def _linear(din, dout, *, dtype, param_dtype, rngs):
    return nnx.Linear(din, dout, use_bias=False, dtype=dtype, param_dtype=param_dtype,
                      kernel_init=linear_init, precision=None, rngs=rngs)


class CausalSelfAttention(nnx.Module):
    def __init__(self, dim, heads, dim_head, dropout, *, dtype, param_dtype, rngs):
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        inner = self.heads * self.dim_head
        self.dtype = dtype
        self.to_qkv = _linear(dim, 3 * inner, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.to_out = _linear(inner, dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.dropout = nnx.Dropout(float(dropout), rngs=rngs if float(dropout) > 0 else None)

    def __call__(self, x, cos, sin, *, deterministic=None):
        n, L, _ = x.shape
        qkv = self.to_qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(n, L, self.heads, self.dim_head)
        k = k.reshape(n, L, self.heads, self.dim_head)
        v = v.reshape(n, L, self.heads, self.dim_head)
        q = apply_rope(q, cos, sin).astype(self.dtype)
        k = apply_rope(k, cos, sin).astype(self.dtype)
        v = v.astype(self.dtype)
        logits = jnp.einsum("nqhd,nkhd->nhqk", q, k, preferred_element_type=jnp.float32)
        logits = logits * (1.0 / np.sqrt(self.dim_head))
        causal = jnp.tril(jnp.ones((L, L), dtype=bool))
        logits = jnp.where(causal[None, None], logits, jnp.finfo(jnp.float32).min)
        probs = jax.nn.softmax(logits, axis=-1).astype(self.dtype)
        out = jnp.einsum("nhqk,nkhd->nqhd", probs, v).astype(self.dtype)
        out = out.reshape(n, L, self.heads * self.dim_head)
        return self.dropout(self.to_out(out), deterministic=deterministic)


class SwiGLU(nnx.Module):
    def __init__(self, dim, mlp_ratio, dropout, *, dtype, param_dtype, rngs):
        self.hidden = swiglu_hidden(dim, mlp_ratio)
        self.gate_proj = _linear(dim, self.hidden, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.up_proj = _linear(dim, self.hidden, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.down_proj = _linear(self.hidden, dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.dropout = nnx.Dropout(float(dropout), rngs=rngs if float(dropout) > 0 else None)

    def __call__(self, x, *, deterministic=None):
        return self.dropout(self.down_proj(nnx.silu(self.gate_proj(x)) * self.up_proj(x)),
                            deterministic=deterministic)


class EncoderBlock(nnx.Module):
    def __init__(self, dim, heads, dim_head, mlp_ratio, dropout, *, dtype, param_dtype, rngs):
        self.norm1 = RMSNorm(dim, param_dtype=param_dtype)
        self.attn = CausalSelfAttention(dim, heads, dim_head, dropout,
                                        dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.norm2 = RMSNorm(dim, param_dtype=param_dtype)
        self.ffn = SwiGLU(dim, mlp_ratio, dropout, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, x, cos, sin, *, deterministic=None):
        x = x + self.attn(self.norm1(x), cos, sin, deterministic=deterministic)
        x = x + self.ffn(self.norm2(x), deterministic=deterministic)
        return x


class WanLatentMotionEncoderJax(nnx.Module):
    """(n, 9, 16384) bf16/f32 raw Wan latent → (n, 768) motion token（compute dtype）。

    用法::

        enc = WanLatentMotionEncoderJax(mcfg.encoder, rngs=enc_rngs, dtype=jnp.bfloat16)
        tok = enc(latents)            # latents: (n, 9, 16384)

    ``config`` 可为 YAML ``motion.encoder`` 节（dict / DictConfig）、``WanE2EEncoderConfig`` 或 None；
    关键字 ``layers`` / ``dtype`` 等覆盖 config。``dtype`` 缺省取 config 的 ``dtype``（缺省 bf16）。
    ``rngs`` 必须由调用方传入独立的 ``nnx.Rngs``（R7）。
    """

    def __init__(self, config=None, *, rngs: nnx.Rngs, dtype=None, param_dtype=jnp.float32, **overrides):
        cfg = WanE2EEncoderConfig.from_any(config, dtype=dtype, **overrides)
        self.config = cfg
        self.dtype = cfg.dtype
        self.param_dtype = param_dtype
        self.depth = int(cfg.layers)
        self.seq_len = int(cfg.seq_len)
        # 非状态 fp32 常量：Python float 元组（静态属性，不是 nnx.Variable，不进 nnx.state）
        self.latents_mean = LATENTS_MEAN
        self.latents_std = LATENTS_STD
        width = int(cfg.width)
        self.input_proj = _linear(E2E_RAW_DIM, width, dtype=self.dtype, param_dtype=param_dtype, rngs=rngs)
        self.blocks = [
            EncoderBlock(width, int(cfg.heads), int(cfg.dim_head), cfg.mlp_ratio, float(cfg.dropout),
                         dtype=self.dtype, param_dtype=param_dtype, rngs=rngs)
            for _ in range(self.depth)
        ]
        self.final_norm = RMSNorm(width, param_dtype=param_dtype)
        self.output_proj = _linear(width, int(cfg.motion_dim), dtype=self.dtype,
                                   param_dtype=param_dtype, rngs=rngs)

    def normalize(self, latents):
        """升 f32 → 逐通道 ``(z - mean) / std``（f32），返回 f32 (…, 9, 16384)。"""
        z = jnp.asarray(latents).astype(jnp.float32)
        lead = z.shape[:-1]
        z = z.reshape(*lead, E2E_LATENT_CHANNELS, E2E_LATENT_SIZE * E2E_LATENT_SIZE)
        mean = jnp.asarray(np.asarray(self.latents_mean, np.float32))[:, None]
        std = jnp.asarray(np.asarray(self.latents_std, np.float32))[:, None]
        return ((z - mean) / std).reshape(*lead, E2E_RAW_DIM)

    def __call__(self, latents, *, deterministic=None):
        if latents.ndim != 3 or tuple(latents.shape[1:]) != (self.seq_len, E2E_RAW_DIM):
            raise ValueError(f"WanLatentMotionEncoderJax 输入须为 (n,{self.seq_len},{E2E_RAW_DIM})，"
                             f"得到 {tuple(latents.shape)}")
        x = self.normalize(latents).astype(self.dtype)
        x = self.input_proj(x)
        cos, sin = rope_cos_sin(self.seq_len, int(self.config.dim_head))
        for blk in self.blocks:
            x = blk(x, cos, sin, deterministic=deterministic)
        x = x[:, -1]
        return self.output_proj(self.final_norm(x))

    def set_rms_stop_gradient(self, flag: bool) -> None:
        """G1c 负例专用：给所有 RMSNorm 分母加 / 去 stop_gradient。"""
        for blk in self.blocks:
            blk.norm1.stop_grad_denominator = bool(flag)
            blk.norm2.stop_grad_denominator = bool(flag)
        self.final_norm.stop_grad_denominator = bool(flag)


def count_params(model: nnx.Module) -> int:
    return int(sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))))


# ── PyTorch state dict（NumPy 数组）→ nnx 参数 ──────────────────────────────────────────────────────
_BLOCK_LINEARS = {
    "attn.to_qkv.weight": ("attn", "to_qkv"),
    "attn.to_out.weight": ("attn", "to_out"),
    "ffn.gate_proj.weight": ("ffn", "gate_proj"),
    "ffn.up_proj.weight": ("ffn", "up_proj"),
    "ffn.down_proj.weight": ("ffn", "down_proj"),
}
_BLOCK_NORMS = {"norm1.weight": "norm1", "norm2.weight": "norm2"}
_DROPPED_BLOCK_BUFFERS = ("attn.rope.cos_cache", "attn.rope.sin_cache")


def _np(v) -> np.ndarray:
    if hasattr(v, "detach"):  # torch.Tensor（本模块不 import torch）
        v = v.detach().cpu().float().numpy()
    return np.asarray(v)


def load_torch_state_dict(model: WanLatentMotionEncoderJax, state_dict: Mapping[str, Any], *,
                          check_stats: bool = True) -> dict:
    """把 d3213de ``WanLatentMotionEncoder`` 的 state dict 原地写进 ``model``（Linear 权重转置为 (in, out)）。

    - ``blocks.{i}.attn.rope.{cos,sin}_cache`` 丢弃（JAX 现算）；
    - ``latents_mean`` / ``latents_std`` 不进参数，``check_stats`` 时须与本模块常数逐位相等，否则 raise；
    - 任何未映射或缺失的键 raise。返回 ``{"loaded": [...], "dropped": [...]}``。
    """
    sd = {k: _np(v) for k, v in state_dict.items()}
    used, dropped = set(), []

    def take(key, shape):
        if key not in sd:
            raise KeyError(f"state dict 缺键 {key}")
        arr = sd[key].astype(np.float32)
        if tuple(arr.shape) != tuple(shape):
            raise ValueError(f"{key} 形状 {arr.shape} != {shape}")
        used.add(key)
        return arr

    def set_linear(lin: nnx.Linear, key):
        din, dout = lin.kernel.value.shape
        lin.kernel.value = jnp.asarray(take(key, (dout, din)).T, dtype=lin.kernel.value.dtype)

    def set_norm(norm: RMSNorm, key):
        norm.weight.value = jnp.asarray(take(key, norm.weight.value.shape), dtype=norm.weight.value.dtype)

    for stat, ref in (("latents_mean", LATENTS_MEAN), ("latents_std", LATENTS_STD)):
        if stat not in sd:
            raise KeyError(f"state dict 缺键 {stat}")
        used.add(stat)
        if check_stats:
            got = sd[stat].astype(np.float32).reshape(-1)
            want = np.asarray(ref, np.float32)
            if got.shape != want.shape or not np.array_equal(got.view(np.uint32), want.view(np.uint32)):
                raise ValueError(f"{stat} 与模块常数不逐位相等：max|Δ|={np.abs(got - want).max()}")
        dropped.append(stat)

    set_linear(model.input_proj, "input_proj.weight")
    for i, blk in enumerate(model.blocks):
        for suffix, (sub, name) in _BLOCK_LINEARS.items():
            set_linear(getattr(getattr(blk, sub), name), f"blocks.{i}.{suffix}")
        for suffix, name in _BLOCK_NORMS.items():
            set_norm(getattr(blk, name), f"blocks.{i}.{suffix}")
        for suffix in _DROPPED_BLOCK_BUFFERS:
            k = f"blocks.{i}.{suffix}"
            if k in sd:
                used.add(k)
                dropped.append(k)
    set_norm(model.final_norm, "final_norm.weight")
    set_linear(model.output_proj, "output_proj.weight")

    extra = sorted(set(sd) - used)
    if extra:
        raise KeyError(f"state dict 有未映射的多余键：{extra}")
    return {"loaded": sorted(used - set(dropped)), "dropped": dropped}


def param_groups(model: WanLatentMotionEncoderJax) -> dict[str, list[tuple[str, Any]]]:
    """G1c 的四组参数（输入投影、各块、最终归一化、输出投影）→ [(torch 键名, nnx.Param)]。"""
    groups: dict[str, list] = {"input_proj": [("input_proj.weight", model.input_proj.kernel)]}
    for i, blk in enumerate(model.blocks):
        g = []
        for suffix, (sub, name) in _BLOCK_LINEARS.items():
            g.append((f"blocks.{i}.{suffix}", getattr(getattr(blk, sub), name).kernel))
        for suffix, name in _BLOCK_NORMS.items():
            g.append((f"blocks.{i}.{suffix}", getattr(blk, name).weight))
        groups[f"block{i}"] = g
    groups["final_norm"] = [("final_norm.weight", model.final_norm.weight)]
    groups["output_proj"] = [("output_proj.weight", model.output_proj.kernel)]
    return groups


def nnx_grads_to_torch_layout(model: WanLatentMotionEncoderJax, grads_state) -> dict[str, np.ndarray]:
    """把 ``nnx.grad`` 得到的 State 转成 torch 键名 → 数组（Linear 转回 (out, in)）。"""
    flat: dict[str, np.ndarray] = {}

    def lin(g, key):
        flat[key] = np.asarray(g["kernel"].value, np.float32).T

    def norm(g, key):
        flat[key] = np.asarray(g["weight"].value, np.float32)

    lin(grads_state["input_proj"], "input_proj.weight")
    for i in range(model.depth):
        b = grads_state["blocks"][i]
        for suffix, (sub, name) in _BLOCK_LINEARS.items():
            lin(b[sub][name], f"blocks.{i}.{suffix}")
        for suffix, name in _BLOCK_NORMS.items():
            norm(b[name], f"blocks.{i}.{suffix}")
    norm(grads_state["final_norm"], "final_norm.weight")
    lin(grads_state["output_proj"], "output_proj.weight")
    return flat
