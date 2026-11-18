import inspect

import flax.nnx as nnx
import jax
import jax.numpy as jnp


import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.mem_encoder import FeatureEncoder
from mme_vla_suite.models.representation.utils import kernel_init
from mme_vla_suite.shared.wan_raw9 import motion_token_budget

# ── E2E-Enc（1117-e2e-enc-train-plan.html 一节 percep_mem 行、YAML 行）：训练侧共用的身份常量 ─────────────────
E2E_ENCODER_KIND = "wan_e2e"
E2E_READOUT = "wan_e2e_enc768_s16_v1"
E2E_TABLE_KIND = "wan_raw_bf16_s16_v1"                  # 表 provenance 的制表身份（S2 契约）
E2E_LAYOUT = "motion-raw9bf16-grid16-demopad17-v1"      # 表布局名（S2 契约）
E2E_LATENTS_PER_WINDOW = 9
E2E_RAW_DIM = 16 * 32 * 32                              # 16384
E2E_RAW_ROW_SHAPE = (9, 16, 32, 32)
E2E_POLICY_ENCODER_PARAMS = 20_252_928                  # depth=1 的 encoder 参数量（〇.3；只进 policy_encoder，不与表的 0 混用）
E2E_SPEC_SRC = "d3213de"
E2E_ENCODER_ATTR = "motion_encoder_e2e"                 # 子模块属性名：不含 img（冻结正则）与 llm（lora 正则）
# R7：独立 RNG 流两级派生的常数（单级 fold_in(base, 0xE2E) 会与 default 流第 3630 次取用撞同一 key）
E2E_RNG_FOLD_1 = 0x7FFFFFFF
E2E_RNG_FOLD_2 = 0xE2E


def is_e2e(mcfg) -> bool:
    """motion 节是否为 E2E 分支（encoder_kind == wan_e2e）；mcfg 为 None 或缺键时 False。"""
    return bool(mcfg is not None and mcfg.get("encoder_kind", None) == E2E_ENCODER_KIND)


def e2e_slot_latents(mcfg) -> int:
    """E2E 传输长度 = budget（槽数）× latents_per_window；motion_emb 形状 (b, 该值, raw_dim)。"""
    return int(mcfg.budget) * int(mcfg.latents_per_window)


def e2e_rngs(rngs: nnx.Rngs) -> nnx.Rngs:
    """R7：只读 default 流的 key（不调用 rngs()、不推进 count），两级 fold_in 派生 encoder 专用流。"""
    base = rngs.default.key.value
    return nnx.Rngs(params=jax.random.fold_in(jax.random.fold_in(base, E2E_RNG_FOLD_1), E2E_RNG_FOLD_2))


def build_e2e_encoder(mcfg, rngs: nnx.Rngs, dtype):
    """按 YAML motion.encoder 节构造 S1 的 ``WanLatentMotionEncoderJax``（深度取 ``encoder.layers``，不写死）。

    S1 的构造签名在本文件写作时尚未合入：按参数名把 YAML 值映射过去，签名里不认识的名字不传；
    必需参数缺失时由类自身报错（fail-loud）。
    """
    from mme_vla_suite.models.representation.wan_e2e_encoder import WanLatentMotionEncoderJax

    ecfg = mcfg.encoder
    cand = {
        "depth": int(ecfg.layers), "layers": int(ecfg.layers), "num_layers": int(ecfg.layers),
        "width": int(ecfg.width), "dim": int(ecfg.width), "hidden_dim": int(ecfg.width),
        "heads": int(ecfg.heads), "num_heads": int(ecfg.heads),
        "dropout": float(ecfg.get("dropout", 0.0)),
        "dtype": dtype, "rngs": rngs, "config": ecfg, "cfg": ecfg,
    }
    params = inspect.signature(WanLatentMotionEncoderJax.__init__).parameters
    kwargs = {k: v for k, v in cand.items() if k in params}
    return WanLatentMotionEncoderJax(**kwargs)


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

        # ── motion memory 运动路（0901-motion-memory-plan.md 2.4 / 2.9）──────────────────────────
        # 唯一原则：模块不是在 __call__ 里跳过，而是在 __init__ 里根本不创建（红线 5）——两个新 kernel 只要存在，
        # train_step 的 param_norm 与参数叶数就会改变，关闭态将无法命中原有锚点。
        # 且必须建在 feature_encoder 之后：flax nnx 单条 default RNG 流按调用顺序 fold_in，插在前面会改变帧路的初始化值。
        mcfg = getattr(config, "motion", None)                        # 旧 yaml 缺整节 → None
        self.motion_enabled = bool(mcfg is not None and mcfg.get("enabled", False))
        # E2E 分支（1117 计划 R3）：encoder_kind 未写或为其他值时 motion_e2e=False，encoder 根本不创建
        self.motion_e2e = bool(self.motion_enabled and is_e2e(mcfg))
        if self.motion_enabled:
            self.motion_budget = int(mcfg.budget)                      # 窗数（C-B 契约：motion.budget 恒为窗数）
            # token 序列长度 = budget × tokens_per_window；旧 yaml 无 tokens_per_window 时等于 budget（计划 1112 §1.0 C-B）
            self.motion_tokens = motion_token_budget(mcfg)
            self.motion_dim = int(mcfg.dim)
            # motion_emb 的第 1 维：非 E2E 等于 token 数；E2E 为 budget × latents_per_window（1440，模型内再 reshape）
            self.motion_emb_len = self.motion_tokens
            self.motion_emb_dim = self.motion_dim
            if self.motion_e2e:
                if mcfg.get("tokens_per_window", None) is not None:
                    raise ValueError("E2E 分支不得写 motion.tokens_per_window（槽数 = budget，见 1117 计划一节 YAML 行）")
                self.motion_lpw = int(mcfg.latents_per_window)
                self.motion_emb_len = e2e_slot_latents(mcfg)
                self.motion_emb_dim = int(mcfg.raw_dim)
            self.motion_pos_dim = int(mcfg.pos_dim)
            pos_hidden = int(self.config.memory_feature.pos.hidden_dim)
            # 新参数名不得含 img（freeze filter PathRegex(".*img.*") 会误冻结 + 强转 bf16）；路径含 mem_encoder 即入 memory_params
            self.motion_pos_proj = nnx.Linear(                        # 256 → 768，W 256×768 + b 768
                self.motion_pos_dim, pos_hidden, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
            # (motion.dim + 768) → memory_token_dim：wantp768 为 1536→1024；wan_raw9 为 17152→1024（9 个 latent 共用这一个 W）
            self.motion_encoder_static = nnx.Linear(
                self.motion_dim + pos_hidden, int(self.config.memory_token_dim),
                rngs=rngs, dtype=dtype, kernel_init=kernel_init)
            if self.motion_e2e:
                # R7：独立 RNG 流（只读 default.key，不推进 count），构造位置不影响其余模块初值
                setattr(self, E2E_ENCODER_ATTR, build_e2e_encoder(mcfg, e2e_rngs(rngs), dtype))

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
        # 返回 (b, 帧路 budget + motion_tokens, memory_token_dim)；重排统一由 HistoryPi0.embed_memory 完成。
        if motion_emb is None or motion_pos is None or motion_mask is None:
            raise ValueError("motion.enabled=true 但 motion_emb / motion_pos / motion_mask 有 None（observation 缺键）")
        b = static_image_emb.shape[0]
        if tuple(motion_emb.shape) != (b, self.motion_emb_len, self.motion_emb_dim):
            raise ValueError(f"motion_emb 形制 {tuple(motion_emb.shape)} != {(b, self.motion_emb_len, self.motion_emb_dim)}")
        if tuple(motion_pos.shape) != (b, self.motion_tokens, self.motion_pos_dim):
            raise ValueError(f"motion_pos 形制 {tuple(motion_pos.shape)} != {(b, self.motion_tokens, self.motion_pos_dim)}")
        if tuple(motion_mask.shape) != (b, self.motion_tokens):
            raise ValueError(f"motion_mask 形制 {tuple(motion_mask.shape)} != {(b, self.motion_tokens)}")
        if self.motion_e2e:
            # (b, budget·9, 16384) bf16 → (b·budget, 9, 16384) 过 encoder（内部先升 fp32 标准化）→ (b, budget, 768)
            enc = getattr(self, E2E_ENCODER_ATTR)
            z = motion_emb.reshape(b * self.motion_tokens, self.motion_lpw, self.motion_emb_dim)
            motion_emb = enc(z).reshape(b, self.motion_tokens, self.motion_dim)
        pos_h = nnx.silu(self.motion_pos_proj(motion_pos))
        motion_tokens = self.motion_encoder_static(jnp.concatenate([motion_emb, pos_h], axis=-1))
        return jnp.concatenate([hidden_states, motion_tokens], axis=1), None, None
