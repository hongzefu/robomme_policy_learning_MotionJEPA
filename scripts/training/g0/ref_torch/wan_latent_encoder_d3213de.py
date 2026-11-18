# ---------------------------------------------------------------------------------------------
# VENDORED PyTorch 参考实现（计划 1117-e2e-enc-train-plan 一节 / 二节 S1 行），只读生成、不改原文。
# 来源：MotionJEPA commit d3213de673d6b71b4ff46f6a568c9b79560231a9（禁止参考 HEAD 483c23e，那是另一架构）
#   src/motion_jepa/models/wan_latent_encoder.py  blob 78c04066be27471939e717f6f98b8f0130c926ba
#       sha256 bd8775d64356d224f9ad945dc908b4830041417fb05487e40ee8a5dcdcf6ea44
#   src/motion_jepa/models/wan_latent_dit.py      blob bf67d0a265bb227d22a66d62619bcebe9c843454
#       sha256 44a9370da4c5d64e1ea2dcb46f3dd995e73a2fbe649b26a5d68e8f951e40e309
# 拼接方式：wan_latent_encoder.py 全文逐字保留，唯一改动是把第 17 行
#   `from .wan_latent_dit import LatentAffineMixin`
# 换成 wan_latent_dit.py 第 35-65 行（class LatentAffineMixin 全文，逐字；sha256 e2e517eb53370ee4bc8b58e27c70894a12450c9d1b08b296a0c84f973f58fe8b），
# 因为该类只依赖 torch。本文件末尾「vendored 附加」段是本仓新增的自检工具（strict 加载 ckpt、按 depth 构造），
# 不改原类。训练代码不得 import 本文件；策略仓 pyproject 不加 motion_jepa 包。
# 生成者：S1（2026-11-18 美东）。
# ---------------------------------------------------------------------------------------------
"""Motion encoder for the v7 Wan latent path (see archived Wan plan, section 7).

Each chunk has nine frozen Wan VAE latents: one frame in z0 and four frames
in each of z1..z8. An affine normalization and flattening precede a shared
Linear(16384, 768) projection. This projection has no spatial weight sharing
and is shared across the two VAE encoding modes, a documented limitation.
Llama3-style causal blocks use RMSNorm, 1D RoPE, and SwiGLU. The final tokens
are read out and projected to motion_dim. There is no pixel frontend, state
fusion, VAE gradient, or gradient checkpointing requirement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ==== BEGIN vendored wan_latent_dit.py:35-65 (d3213de) ====
class LatentAffineMixin:
    """Shared channel-wise affine buffers for the encoder and decoder."""

    def _register_latent_stats(self, latent_channels, latents_mean, latents_std):
        shape = (1, latent_channels, 1, 1)
        if latents_mean is None or latents_std is None:
            # NaN placeholders fail the finite check if no statistics were loaded.
            mean = torch.full(shape, float("nan"))
            std = torch.full(shape, float("nan"))
        else:
            assert latents_mean.numel() == latent_channels
            assert latents_std.numel() == latent_channels
            mean = latents_mean.float().reshape(shape).clone()
            std = latents_std.float().reshape(shape).clone()
        # Persist statistics in checkpoints; EMA leaves constant buffers unchanged.
        self.register_buffer("latents_mean", mean, persistent=True)
        self.register_buffer("latents_std", std, persistent=True)

    def _assert_stats_ready(self):
        assert torch.isfinite(self.latents_std).all() and torch.isfinite(self.latents_mean).all(), \
            "affine buffers were not populated by constructor arguments or checkpoint (still NaN)"

    def normalize(self, z):
        """Normalize raw latents, broadcasting over arbitrary leading dimensions."""
        self._assert_stats_ready()
        return (z - self.latents_mean) / self.latents_std

    def denormalize(self, zn):
        """Convert normalized latents to raw latents before evaluation decode."""
        self._assert_stats_ready()
        return zn * self.latents_std + self.latents_mean
# ==== END vendored wan_latent_dit.py:35-65 ====


class RMSNorm(nn.Module):
    """Root-mean-square normalization without mean subtraction."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class RotaryEmbedding1D(nn.Module):
    """One-dimensional rotary embedding for causal sequences."""

    def __init__(self, dim_head, max_seq_len=1024, base=10000.0):
        super().__init__()
        freqs = 1.0 / (base ** (torch.arange(0, dim_head, 2, dtype=torch.float32) / dim_head))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (L, dim_head//2)
        cos_cache = angles.cos()
        sin_cache = angles.sin()
        self.register_buffer("cos_cache", cos_cache)
        self.register_buffer("sin_cache", sin_cache)

    def forward(self, q, k):
        """Apply RoPE to Q/K of shape (B, heads, L, dim_head)."""
        L = q.shape[2]
        cos = self.cos_cache[:L].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:L].unsqueeze(0).unsqueeze(0)

        def _apply(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

        return _apply(q), _apply(k)


class CausalSelfAttention(nn.Module):
    """Multihead causal self-attention with 1D RoPE."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, rope=None):
        super().__init__()
        self.heads = heads
        inner_dim = heads * dim_head

        self.to_qkv = nn.Linear(dim, 3 * inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = rope

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=self.heads), qkv)

        if self.rope is not None:
            q, k = self.rope(q, k)

        # Mixed-precision coupling: fp32 RoPE buffers promote q/k while v
        # remains bf16. The surrounding autocast converts SDPA inputs to bf16.
        # Moving this call outside autocast requires explicit dtype alignment.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.dropout(self.to_out(out))


class SwiGLU(nn.Module):
    """SwiGLU feedforward layer: gate(x) * up(x), then a down projection."""

    def __init__(self, dim, mlp_ratio=4, dropout=0.0):
        super().__init__()
        hidden = int(2 * dim * mlp_ratio / 3)
        hidden = ((hidden + 7) // 8) * 8

        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class EncoderBlock(nn.Module):
    """Pre-normalized block: RMSNorm, causal attention, RMSNorm, SwiGLU."""

    def __init__(self, dim, heads=8, dim_head=64, mlp_ratio=4, dropout=0.0, rope=None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads=heads, dim_head=dim_head,
                                        dropout=dropout, rope=rope)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class WanLatentMotionEncoder(LatentAffineMixin, nn.Module):
    """Map raw latents (B, 9, 16, 32, 32) to (B, M, motion_dim)."""

    def __init__(self, latent_channels=16, latent_size=32, hidden_dim=768,
                 motion_dim=768, num_tokens=1, seq_len=9, depth=8, heads=12,
                 dim_head=64, mlp_ratio=4, dropout=0.05,
                 latents_mean=None, latents_std=None):
        super().__init__()
        assert num_tokens <= seq_len, \
            f"num_tokens={num_tokens} must be <= seq_len={seq_len} for final-token readout"
        self.num_tokens = num_tokens
        self.seq_len = seq_len
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self._register_latent_stats(latent_channels, latents_mean, latents_std)

        flat_dim = latent_channels * latent_size * latent_size     # 16384
        self.input_proj = nn.Linear(flat_dim, hidden_dim, bias=False)

        rope = RotaryEmbedding1D(dim_head, max_seq_len=seq_len)
        self.blocks = nn.ModuleList([
            EncoderBlock(hidden_dim, heads=heads, dim_head=dim_head,
                         mlp_ratio=mlp_ratio, dropout=dropout, rope=rope)
            for _ in range(depth)
        ])
        self.final_norm = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, motion_dim, bias=False)

    def forward(self, latents):
        """
        Args:
            latents: (B, 9, 16, 32, 32) in the raw latent domain.
        Returns:
            motion_tokens: (B, M, motion_dim)
        """
        zn = self.normalize(latents)                               # Input affine normalization
        # Explicitly reject short sequences; slicing the RoPE cache would
        # otherwise silently accept a sequence missing z0.
        assert zn.shape[1] == self.seq_len, \
            f"latent sequence length {zn.shape[1]} != seq_len={self.seq_len} (= data.max_horizon + 1)"
        assert zn.shape[2:] == (self.latent_channels, self.latent_size, self.latent_size), \
            f"latent shape {tuple(zn.shape[2:])} != expected ({self.latent_channels},{self.latent_size},{self.latent_size})"

        x = self.input_proj(zn.flatten(2))                         # (B,9,16384)→(B,9,768)
        for block in self.blocks:
            x = block(x)
        motion_tokens = x[:, -self.num_tokens:]                    # Final-token readout
        return self.output_proj(self.final_norm(motion_tokens))   # (B,M,motion_dim)


# =============================================================================================
# vendored 附加（本仓新增，非 d3213de 原文）：按 depth 构造、strict 加载 checkpoint 自检
# =============================================================================================
VENDORED_FROM = "d3213de673d6b71b4ff46f6a568c9b79560231a9"
P3_CKPT_SHA256 = "0c1986297ccc0ab1913910f33a09ec74ba4c208844d0f5d72dd7ba59e0d9e3ca"
PARAMS_DEPTH1 = 20_252_928
PARAMS_DEPTH8 = 69_808_896


def build_reference(depth, *, dropout=0.0, latents_mean=None, latents_std=None):
    """以 d3213de 默认超参构造参考 encoder：hidden 768、12 头、dim_head 64、mlp_ratio 4、seq_len 9、num_tokens 1。"""
    if latents_mean is not None:
        latents_mean = torch.as_tensor(latents_mean, dtype=torch.float32)
        latents_std = torch.as_tensor(latents_std, dtype=torch.float32)
    return WanLatentMotionEncoder(depth=int(depth), dropout=float(dropout),
                                  latents_mean=latents_mean, latents_std=latents_std)


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def load_ckpt_encoder_state(path):
    """读 P3 checkpoint 的 encoder state dict（EMA 权重）。"""
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if "encoder" not in ck:
        raise KeyError(f"checkpoint 缺 encoder 键：{sorted(ck)}")
    return ck["encoder"]


def strict_load_ckpt(path, depth=8):
    """strict 加载 checkpoint 的 encoder；返回 (model, missing, unexpected)。"""
    model = build_reference(depth, dropout=0.0)
    res = model.load_state_dict(load_ckpt_encoder_state(path), strict=True)
    model.eval()
    return model, list(res.missing_keys), list(res.unexpected_keys)


if __name__ == "__main__":
    import argparse
    import hashlib

    ap = argparse.ArgumentParser(description="vendored d3213de encoder 自检：strict 加载 ckpt")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--depth", type=int, default=8)
    a = ap.parse_args()
    h = hashlib.sha256()
    with open(a.ckpt, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    m, missing, unexpected = strict_load_ckpt(a.ckpt, a.depth)
    print(f"ckpt_sha256={h.hexdigest()} expected={P3_CKPT_SHA256} params={count_params(m)}")
    ok = not missing and not unexpected and h.hexdigest() == P3_CKPT_SHA256
    print(f"VENDORED_STRICT_LOAD={'PASS' if ok else 'FAIL'} depth={a.depth} missing={missing} unexpected={unexpected}")
