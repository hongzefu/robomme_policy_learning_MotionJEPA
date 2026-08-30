"""在线评估 framesamp 专用记忆缓冲 `FrameSampMemory`（commitV4.4 新增）。

替代旧 `shared.mem_buffer.MemoryBuffer` 在在线评估侧的角色，训练/在线自此不再
import mem_buffer（建库域留有冻结副本 `dataset_builder/mem_buffer.py`）。

与旧实现的三处合法差异（均有不可观测性论证，其余逐跳同式）：
① pos 表与池化只算 4x4 一档（frame_sampling 唯一消费档；PosEmb3D 是无 RNG、无参数
   的纯函数，各档独立，只算一档不改变该档的任何字节——表本体 1.01 GiB → 192 MiB）；
② `jax.device_get` 提到循环外一次（旧实现每步对同一 device 数组重复 get，
   取回的字节相同）；
③ 不存 image_pixels 与 8x8/2x2 档位（image_pixels 的唯一消费者是 token_drop 打分
   与死码可视化，装配只读 image_emb_4x4/pos_emb_4x4/state_emb 三键）。

⚠ 禁把 encode 与 pool 包进新的 jax.jit（R18）：融合边界变了，bf16 累加序可能变位。
本实现保持 encode（注入的 vision_enc_fn，本身已 jit）与 pool 分离调用，同旧实现。
"""

import math
from typing import Callable

import einops
import jax
import jax.numpy as jnp
import numpy as np

from openpi.shared import image_tools
from mme_vla_suite.shared.data_utils import pool_tokens_to_size, right_padding_token_emb
from mme_vla_suite.shared.posemb_3d import PosEmb3D
from mme_vla_suite.shared.sampling import even_sampling_indices


class FrameSampMemory:
    def __init__(
        self,
        num_views: int = 1,
        img_emb_dim: int = 2048,
        pos_emb_dim: int = 768,
        state_emb_dim: int = 8,
        max_steps: int = 4096,
        *,
        vision_enc_fn: Callable,
    ):
        if vision_enc_fn is None:
            raise ValueError("FrameSampMemory 必须注入 vision_enc_fn（模型侧编码器）")
        self.num_views = num_views
        self.img_emb_dim = img_emb_dim
        self.pos_emb_dim = pos_emb_dim
        self.state_emb_dim = state_emb_dim
        self.max_steps = max_steps

        self.vision_enc = vision_enc_fn
        # 与旧实现同一 PosEmb3D、同一 arange 输入——4x4 表逐位同旧表 "4x4" 档
        pos_embedder = PosEmb3D(dim=pos_emb_dim)
        ranges = jnp.arange(max_steps)
        self.pos_emb_4x4 = np.array(pos_embedder(ranges, 4))

        self._history_feats = {}

    @property
    def n_steps(self) -> int:
        return len(self._history_feats)

    def clear(self):
        self._history_feats.clear()

    def add_buffer(
        self,
        images,  #: (t v h w 3), np.uint8
        states,  #: (t d), np.float32
        step_idx_list: list[int],
    ):
        t, v, _, _, _ = images.shape
        assert v == self.num_views

        for step_idx in step_idx_list:
            # 有效域上界显式 raise：pos 表只有 max_steps 行，numpy 切片越界会静默
            # 返回空数组、坏数据一路流到装配层——禁止依赖切片行为
            if step_idx >= self.max_steps:
                raise ValueError(
                    f"step_idx {step_idx} 超出记忆有效域 [0, {self.max_steps})"
                )
            if step_idx in self._history_feats:
                raise ValueError(f"step_idx {step_idx} already in buffer")

        # 与旧实现逐字同式的归一化 / resize / 编码链
        image_jnp = jnp.array(
            images.astype(np.float32) / 255.0 * 2.0 - 1.0
        )
        image_jnp = einops.rearrange(image_jnp, "t v h w c -> (t v) h w c")
        image_jnp = image_tools.resize_with_pad(image_jnp, 224, 224)
        image_jnp = einops.rearrange(image_jnp, "(t v) h w c -> t v h w c", t=t, v=v)
        output_emb = self.vision_enc(image_jnp)  # (t, v, 64, 2048)

        pooled_emb_4x4 = pool_tokens_to_size(output_emb, 16)  # (t, v, 16, 2048)
        pooled_host = jax.device_get(pooled_emb_4x4)  # 循环外一次（合法差异②）

        for i, step_idx in enumerate(step_idx_list):
            image_emb_4x4 = pooled_host[i]  # (v, 16, 2048)
            pos_emb_4x4 = self.pos_emb_4x4[
                step_idx*self.num_views : (step_idx+1)*self.num_views]  # (v, 16, 768)

            self._history_feats[step_idx] = {
                "image_emb_4x4": image_emb_4x4,  # bf16
                "pos_emb_4x4": pos_emb_4x4,      # fp32
                "state_emb": states[i],          # fp32
            }

    # ―― 以下装配路径与旧 MemoryBuffer 的 frame_sampling 支路逐字同式 ――

    def get_frame_sampling_indices(self, step_idx, token_budget, token_per_image):
        max_size = token_budget // (token_per_image * self.num_views)
        return even_sampling_indices(step_idx, max_size)

    def _prepare_frame_sampling(self, history_feats, indices_to_load, token_budget, token_per_image):
        spatial_size = str(int(math.sqrt(token_per_image)))
        spatial_key = f"{spatial_size}x{spatial_size}"
        max_size = token_budget // (token_per_image * self.num_views)

        sampled_img_emb = self._load_emb(history_feats, indices_to_load, f"image_emb_{spatial_key}")
        sampled_pos_emb = self._load_emb(history_feats, indices_to_load, f"pos_emb_{spatial_key}")
        sampled_state_emb = self._load_emb(history_feats, indices_to_load, "state_emb")
        mask = np.ones((sampled_img_emb.shape[0]), dtype=np.bool_)

        # we use right padding to the perceptual memory
        # （必须复用 right_padding_token_emb——只换模块、不换数值路径，禁改写预分配版）
        sampled_img_emb, sampled_pos_emb, sampled_state_emb, mask = right_padding_token_emb(
            sampled_img_emb, sampled_pos_emb, sampled_state_emb, mask, max_size
        )

        img_emb = np.reshape(sampled_img_emb, (-1, self.img_emb_dim))
        pos_emb = np.reshape(sampled_pos_emb, (-1, self.pos_emb_dim))
        mask = np.repeat(mask, self.num_views * token_per_image)
        state_emb = np.repeat(sampled_state_emb, self.num_views * token_per_image, axis=0)

        return img_emb, pos_emb, state_emb, mask

    def prepare_frame_sampling(self, step_idx, token_budget, token_per_image, history_feats_gather_fn, *args, **kwargs):
        indices_to_load = self.get_frame_sampling_indices(step_idx, token_budget, token_per_image)
        history_feats = history_feats_gather_fn(indices_to_load, *args, **kwargs)
        return self._prepare_frame_sampling(history_feats, indices_to_load, token_budget, token_per_image)

    @staticmethod
    def _load_emb(history_feats: dict, indices_to_load: list[int], key: str):
        return np.stack(
            [history_feats[idx][key] for idx in indices_to_load],
            axis=0,
        )

    def default_history_feats_gather_fn(self, indices_to_load, *args, **kwargs):
        return {idx: self._history_feats[idx] for idx in indices_to_load}
