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


# ── motion memory 交错次序（motion-memory-plan.md 3.4 ④′ / 第二部分 2.6）────────────────────
# 训练侧（FrameSampDataset.__getitem__）与在线侧（MME_VLA_Policy._prepare_history）必须 import 同一份本函数
# （R20：两侧各写一份不会报错，只静默让在线看到与训练不同的次序）。本模块 import 面只有 numpy。

MEM_ORDER_SENTINEL = int(np.iinfo(np.int32).max)   # padding 位的「时刻」哨兵：≥ 任何真实帧号，排到尾部


def pad_times(times, budget: int) -> np.ndarray:
    """把 ≤ budget 个真实全域时刻右填充成 (budget,) int64，padding 位记哨兵。"""
    t = np.asarray(times, dtype=np.int64).reshape(-1)
    if t.size > budget:
        raise ValueError(f"真实时刻数 {t.size} > 预算 {budget}（零截断契约被破坏）")
    if t.size and int(t.max()) >= MEM_ORDER_SENTINEL:
        raise ValueError(f"时刻 {int(t.max())} 触到哨兵 {MEM_ORDER_SENTINEL}")
    out = np.full(budget, MEM_ORDER_SENTINEL, dtype=np.int64)
    out[:t.size] = t
    return out


def memory_order(frame_times, tokens_per_frame: int, motion_times) -> np.ndarray:
    """608 个候选位按 (全域时刻, 类型) 稳定排序，返回 0..N-1 的置换（int32）。

    frame_times：(F,) int64，帧路每帧的全域帧号（padding 帧记 MEM_ORDER_SENTINEL），每帧占 tokens_per_frame 个连续位；
    motion_times：(M,) int64，运动路每个起点的全域帧号（padding 记哨兵）。
    键 = 时刻 × 2 + 类型（帧 0、motion 1）：同刻帧在 motion 前、同帧 16 位保持内部次序、两路 padding 一并落尾且帧路 padding 在前
    （拼接顺序 [帧路 | 运动路] + kind="stable" 共同保证）。产出后显式 raise 校验是合法置换——
    jnp.take_along_axis 默认 mode="fill"：float 侧越界填 NaN、bool 侧越界填 True、负索引静默回绕，「界内但非置换」只有这道校验能拦。
    """
    ft = np.asarray(frame_times, dtype=np.int64).reshape(-1)
    mt = np.asarray(motion_times, dtype=np.int64).reshape(-1)
    if tokens_per_frame < 1:
        raise ValueError(f"tokens_per_frame={tokens_per_frame} 非法")
    frame_keys = np.repeat(ft * 2 + 0, tokens_per_frame)
    motion_keys = mt * 2 + 1
    keys = np.concatenate([frame_keys, motion_keys])
    order = np.argsort(keys, kind="stable").astype(np.int32)
    n = keys.shape[0]
    if not np.array_equal(np.sort(order), np.arange(n, dtype=np.int32)):
        raise RuntimeError(f"mem_order 不是 0..{n - 1} 的置换")
    return order
