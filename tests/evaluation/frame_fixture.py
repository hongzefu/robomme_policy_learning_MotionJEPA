"""独立算术真值及轻量记忆夹具；替身编码不证明 VAE、EMA 或位置码公式。"""

from dataclasses import dataclass
import hashlib

import numpy as np


DEMO_LENGTHS = (0, 1, 16, 17, 32, 33, 48, 49, 50, 64, 65, 1808, 1809)
EXEC_QUERIES = (31, 32, 33, 63, 64, 65, 1301)
SENTINEL = 2147483647


@dataclass(frozen=True)
class ExpectedWindow:
    start: int
    frame_ids: tuple[int, ...]
    real: int
    pad: int
    segment: str


def oracle_windows(demo_length, query):
    """直接枚举候选起点和真实帧，不调用线上选窗或离线索引函数。"""
    if demo_length < 0 or query < 0:
        raise ValueError("独立真值只接受非负段长度与执行序号")
    result = []
    for start in range(0, demo_length, 32):
        available = demo_length - start
        if available >= 17:
            real = min(33, available)
            ids = tuple(range(start, start + real)) + (demo_length - 1,) * (33 - real)
            result.append(ExpectedWindow(start, ids, real, 33 - real, "demo"))
    for local_start in range(0, query + 1, 32):
        if local_start + 32 <= query:
            start = demo_length + local_start
            result.append(ExpectedWindow(start, tuple(range(start, start + 33)), 33, 0, "exec"))
    return result


def expected_order(frame_times, motion_times, tokens_per_frame=64):
    """用 Python 元组排序单独定义时刻、类型和原顺序三层规则。"""
    entries = []
    for slot, time in enumerate(frame_times):
        for patch in range(tokens_per_frame):
            index = slot * tokens_per_frame + patch
            entries.append((int(time), 0, index))
    offset = len(frame_times) * tokens_per_frame
    entries.extend((int(time), 1, offset + slot) for slot, time in enumerate(motion_times))
    return np.asarray([item[2] for item in sorted(entries)], dtype=np.int32)


def pixel_color(frame_id):
    """两个字节保留完整帧号，第三字节检测非首像素的搬运错误。"""
    return np.asarray([frame_id % 256, (frame_id // 256) % 256, (frame_id * 17 + 3) % 256], np.uint8)


def raw_frame(frame_id):
    return np.broadcast_to(pixel_color(frame_id), (256, 256, 3))


def packet(frame_ids, *, dtype=np.uint8, frame_size=256):
    colors = np.asarray([pixel_color(index) for index in frame_ids], dtype=dtype)
    images = np.broadcast_to(colors[:, None, None, None, :], (len(frame_ids), 1, frame_size, frame_size, 3))
    states = np.asarray([[index + axis / 16 for axis in range(8)] for index in frame_ids], np.float32)
    return images, states


class RecordingEncoder:
    """逐像素核搬运的替身；只检测原始窗与 4×256 展平后的单 token 装配。"""

    def __init__(self):
        self.calls = []

    def __call__(self, window, start):
        assert window.shape == (33, 256, 256, 3)
        assert window.dtype == np.uint8
        assert window.flags.c_contiguous
        colors = window[:, 0, 0]
        ids = tuple(int(color[0]) + 256 * int(color[1]) for color in colors)
        # 原始像素逐点比较；避免测试框架为数百万元素生成冗余诊断副本。
        assert np.all(window == colors[:, None, None, :]), "窗口内像素与帧号标记不一致"
        np.testing.assert_array_equal(colors, np.asarray([pixel_color(index) for index in ids]))
        self.calls.append((start, ids))
        # 四槽分别保留自己的 256 个通道，避免均值池化也能蒙混过关。
        slots = np.asarray([[start + (slot * 256 + channel) / 2048 for channel in range(256)] for slot in range(4)], np.float32)
        return slots.reshape(1024)


def make_memory():
    """跳过昂贵的真实位置表构造；生产入库、选窗、装配方法保持原样。"""
    from mme_vla_suite.policies.framesamp_memory import FrameSampMemory

    memory = FrameSampMemory.__new__(FrameSampMemory)
    memory.num_views = 1
    memory.token_per_image = 64
    memory.image_key = "image_emb_8x8"
    memory.pos_key = "pos_emb_8x8"
    memory.img_emb_dim = 2
    memory.pos_emb_dim = 256
    memory.state_emb_dim = 8
    memory.max_steps = 4096
    # 这里只验证选取窗口起点的第零 patch，不能据此宣称真实正弦位置码通过。
    rows = np.arange(4096, dtype=np.float32)[:, None, None]
    channels = np.arange(256, dtype=np.float32)[None, None, :] / 256
    memory.pos_emb = np.broadcast_to(rows * 1024 + channels, (4096, 64, 256))
    visual_template = np.stack((np.arange(64) + 1, np.arange(64) + 65), axis=-1).astype(np.dtype("bfloat16"))
    memory.vision_enc = lambda images: np.broadcast_to(visual_template, (len(images), 1, 64, 2))
    memory.motion_enabled = True
    memory.motion_stride = 32
    memory.motion_window = 33
    memory.motion_budget = 96
    memory.motion_frame_size = 256
    memory.motion_pos_dim = 256
    memory.motion_dim = 1024
    memory.demo_min_real = 17
    memory.demo_tail_pad = "repeat_last"
    memory.motion_enc = RecordingEncoder()
    memory._history_feats = {}
    memory._history_feats_motion = {}
    memory._raw_frames = {}
    memory._next_grid_start_demo = 0
    memory._next_grid_start_exec = 0
    memory.exec_start_idx = None
    memory.motion_encode_calls = 0
    memory.motion_encode_s = 0.0
    return memory


def feed_packet(memory, frame_ids, demo_length, **kwargs):
    images, states = packet(frame_ids, **kwargs)
    memory.add_buffer(images, states, list(frame_ids), exec_start_idx=demo_length)


def feed_scheduler(memory, frame_ids, demo_length):
    """直接注入完整 raw 段，隔离调度公式；不冒充 add_buffer 输入校验证据。"""
    memory.exec_start_idx = demo_length
    for frame_id in frame_ids:
        memory._raw_frames[frame_id] = raw_frame(frame_id)
    memory._encode_ready_windows(last_frame=frame_ids[-1])


def memory_snapshot(memory):
    """失败前后比较所有内容状态，不把计时开销当语义状态。"""
    def arrays(mapping):
        return tuple((key, value.dtype.str, value.shape, hashlib.sha256(value.tobytes()).hexdigest()) for key, value in sorted(mapping.items()))

    history = tuple((index, arrays(features)) for index, features in sorted(memory._history_feats.items()))
    return (memory.exec_start_idx, memory._next_grid_start_demo, memory._next_grid_start_exec,
            history, arrays(memory._raw_frames), arrays(memory._history_feats_motion),
            memory.motion_encode_calls, tuple(memory.motion_enc.calls))
