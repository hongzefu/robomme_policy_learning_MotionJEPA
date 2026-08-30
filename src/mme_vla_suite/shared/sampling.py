"""训练/在线共用的选帧函数（commitV4.4 自 shared/data_utils.py 原样搬入）。

单独成模块的原因：本函数是 dataloader worker 导入链上的唯一 shared 依赖，
原先住在 data_utils.py 会连带拉起 flax/jax（pool_tokens_to_size 用 nnx），
搬出后 worker 只需 numpy。函数体一个字符未动（保留路径清单，R4/G3 bitwise 前提）。
"""

import numpy as np


def even_sampling_indices(step_idx: int, token_budget: int) -> list[int]:
    """Generate evenly spaced indices for sampling frames."""
    if step_idx < token_budget:
        return list(range(step_idx+1))
    else:
        return np.linspace(0, step_idx, token_budget, dtype=np.int32).tolist()
