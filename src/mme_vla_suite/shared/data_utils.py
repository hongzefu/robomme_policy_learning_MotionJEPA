import numpy as np
import math
import einops
import flax.nnx as nnx


# even_sampling_indices 已于 commitV4.4 原样搬入 shared/sampling.py（只依赖 numpy，
# 解 dataloader worker 的 flax/jax 导入负担），函数体一字未动。


def right_padding_token_emb(
    sampled_img_emb, # (l v p d1)
    sampled_pos_emb, # (l v p d2)
    sampled_state_emb, # (l d3)
    mask, # (l)
    max_size: int):
    if sampled_img_emb.shape[0] < max_size:
        sampled_img_emb = np.concatenate(
            [
                sampled_img_emb,
                np.zeros(
                    (
                        max_size - sampled_img_emb.shape[0],
                        *sampled_img_emb.shape[1:],
                    ),
                    dtype=sampled_img_emb.dtype,
                ),
            ],
            axis=0,
        )
        sampled_pos_emb = np.concatenate(
            [
                sampled_pos_emb,
                np.zeros(
                    (
                        max_size - sampled_pos_emb.shape[0],
                        *sampled_pos_emb.shape[1:],
                    ),
                    dtype=sampled_pos_emb.dtype,
                ),
            ],
            axis=0,
        )
        sampled_state_emb = np.concatenate(
            [
                sampled_state_emb,
                np.zeros(
                    (
                        max_size - sampled_state_emb.shape[0],
                        *sampled_state_emb.shape[1:],
                    ),
                    dtype=sampled_state_emb.dtype,
                ),
            ],
            axis=0,
        )
        mask = np.concatenate(
            [mask, np.zeros((max_size - mask.shape[0]), dtype=np.bool_)], axis=0
        )
    else:
        sampled_img_emb = sampled_img_emb[:max_size]
        sampled_pos_emb = sampled_pos_emb[:max_size]
        sampled_state_emb = sampled_state_emb[:max_size]
        mask = mask[:max_size]
    return sampled_img_emb, sampled_pos_emb, sampled_state_emb, mask




def pool_tokens_to_size(
    tokens, # (b v p d) or (b p d)
    target_size: int = 64,
    pool_type: str = "mean",
):
    if len(tokens.shape) == 4:
        b, v, p, d = tokens.shape
    elif len(tokens.shape) == 3:
        b, p, d = tokens.shape
    else:
        raise ValueError(f"Invalid tokens shape: {tokens.shape}")
    if p == target_size:
        return tokens

    h = w = int(math.sqrt(p))
    if len(tokens.shape) == 4:
        tokens_2d = einops.rearrange(tokens, "b v (h w) d -> (b v) h w d", h=h, w=w)
    else:
        tokens_2d = einops.rearrange(tokens, "b (h w) d -> b h w d", h=h, w=w)

    pool_size = int(math.sqrt(p // target_size))
    if pool_type == "mean":
        pool_func = nnx.avg_pool
    elif pool_type == "max":
        pool_func = nnx.max_pool
    else:
        raise ValueError(f"Invalid pool type: {pool_type}")

    pooled = pool_func(
        tokens_2d, window_shape=(pool_size, pool_size), strides=(pool_size, pool_size)
    )
    if len(tokens.shape) == 4:
        out = einops.rearrange(pooled, "(b v) h w d -> b v (h w) d", b=b, v=v)
    else:
        out = einops.rearrange(pooled, "b h w d -> b (h w) d", b=b)

    return out
