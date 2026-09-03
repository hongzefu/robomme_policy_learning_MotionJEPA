import flax.nnx as nnx
import jax.numpy as jnp


import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.mem_encoder import FeatureEncoder
from mme_vla_suite.models.representation.utils import kernel_init


class PerceptualMemory(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs, dtype: at.DTypeLike = jnp.float32):
        self.config = config
        self.dtype = dtype

        self.feature_encoder = FeatureEncoder(
            rngs=rngs,
            dtype=dtype,
            image_input_dim=self.config.memory_feature.img.input_dim,
            pos_input_dim=self.config.memory_feature.pos.input_dim,
            state_input_dim=self.config.memory_feature.state.input_dim,
            pos_output_dim=self.config.memory_feature.pos.hidden_dim,
            state_output_dim=self.config.memory_feature.state.hidden_dim,
            output_dim_for_percep=self.config.memory_token_dim,
            use_pos_emb=self.config.use_pos_emb,
            use_state_emb=self.config.use_state_emb,
        )

        # ── motion memory 运动路（motion-memory-plan.md 2.4 / 2.9）──────────────────────────
        # 唯一原则：模块不是在 __call__ 里跳过，而是在 __init__ 里根本不创建（红线 5）——两个新 kernel 只要存在，
        # train_step 的 param_norm 与 bench 的 n_leaves（177 → 193）立刻变，关闭态就打不中黄金锚点。
        # 且必须建在 feature_encoder 之后：flax nnx 单条 default RNG 流按调用顺序 fold_in，插在前面会改变帧路的初始化值。
        mcfg = getattr(config, "motion", None)                        # 旧 yaml 缺整节 → None
        self.motion_enabled = bool(mcfg is not None and mcfg.get("enabled", False))
        if self.motion_enabled:
            self.motion_budget = int(mcfg.budget)
            self.motion_dim = int(mcfg.dim)
            self.motion_pos_dim = int(mcfg.pos_dim)
            pos_hidden = int(self.config.memory_feature.pos.hidden_dim)
            # 新参数名不得含 img（freeze filter PathRegex(".*img.*") 会误冻结 + 强转 bf16）；路径含 mem_encoder 即入 memory_params
            self.motion_pos_proj = nnx.Linear(                        # 256 → 768，W 256×768 + b 768
                self.motion_pos_dim, pos_hidden, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
            self.motion_encoder_static = nnx.Linear(                  # 1536 → 2048，W 1536×2048 + b 2048
                self.motion_dim + pos_hidden, int(self.config.memory_token_dim),
                rngs=rngs, dtype=dtype, kernel_init=kernel_init)

    def __call__(
        self,
        static_image_emb: at.Float[at.Array, "b l d1"],
        static_pos_emb: at.Float[at.Array, "b l d2"],
        static_state_emb: at.Float[at.Array, "b l d3"],
        motion_emb=None,
        motion_pos=None,
        motion_mask=None,
    ):
        # get memory tokens using feature encoder
        assert static_image_emb.shape[1] == self.config.budget

        hidden_states = self.feature_encoder.encode_perceptual_memory(
            static_image_emb, static_pos_emb, static_state_emb
        )

        if not self.motion_enabled:
            return hidden_states, None, None

        # ── 运动路：motion_pos → motion_pos_proj → silu，与 motion_emb 在最后一维 concat → motion_encoder_static ──
        # padding 行不做特殊处理（motion_emb 该位为 0），屏蔽完全交给 input_mask——与帧路对 padding 帧的处理逐字同构。
        # 返回并列序 (b, 512 + 96, 2048)；按 mem_order 的重排不在这里做（本层不吃 obs），统一放 HistoryPi0.embed_memory。
        if motion_emb is None or motion_pos is None or motion_mask is None:
            raise ValueError("motion.enabled=true 但 motion_emb / motion_pos / motion_mask 有 None（observation 缺键）")
        b = static_image_emb.shape[0]
        if tuple(motion_emb.shape) != (b, self.motion_budget, self.motion_dim):
            raise ValueError(f"motion_emb 形制 {tuple(motion_emb.shape)} != {(b, self.motion_budget, self.motion_dim)}")
        if tuple(motion_pos.shape) != (b, self.motion_budget, self.motion_pos_dim):
            raise ValueError(f"motion_pos 形制 {tuple(motion_pos.shape)} != {(b, self.motion_budget, self.motion_pos_dim)}")
        if tuple(motion_mask.shape) != (b, self.motion_budget):
            raise ValueError(f"motion_mask 形制 {tuple(motion_mask.shape)} != {(b, self.motion_budget)}")
        pos_h = nnx.silu(self.motion_pos_proj(motion_pos))
        motion_tokens = self.motion_encoder_static(jnp.concatenate([motion_emb, pos_h], axis=-1))
        return jnp.concatenate([hidden_states, motion_tokens], axis=1), None, None
