"""frame_swap_exact（示范帧 token 替换、motion 仍用原帧）的 CPU 测试：真实 FrameSampMemory + 假编码器，不导入 robomme。

核对三件事：
- 开启时送入帧记忆（vision 编码器）的被换帧 = 供体图像，其余帧不变；
- motion 编码器收到的窗口帧与不替换时逐字节相同（原始帧缓冲在 motion 编码前被写回原图）；
- 关闭（ablate=none）时 vision / motion 输入与不装记录器的裸策略逐字节相同。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.interpret import serve_policy_attn as SA

jnp = pytest.importorskip("jax.numpy")
from mme_vla_suite.policies.framesamp_memory import FrameSampMemory  # noqa: E402

SIZE, DIM, IMG_DIM = 8, 3, 4
ES = 6                      # 示范段 6 帧（全域帧号 0..5），exec 第零帧为 6
MOTION_CFG = {"stride": 2, "window_frames": 4, "budget": 16, "frame_size": SIZE, "pos_dim": 8, "dim": DIM,
              "demo_min_real_frames": 1, "demo_tail_pad": "repeat_last"}
TARGET, DONOR = "VideoUnmaskSwap:3", "VideoUnmaskSwap:7"
P, MEM, AH = 9, 7, SA.ACTION_HORIZON


def receiver_image(step):
    return np.full((1, SIZE, SIZE, 3), (step * 7) % 200 + 1, np.uint8)


def donor_image(step):
    return np.full((1, SIZE, SIZE, 3), 250 - step, np.uint8)


class Encoders:
    """记录 vision 编码器的输入（已归一化、resize 到 224）与 motion 编码器的窗口帧。"""

    def __init__(self):
        self.vision_inputs, self.motion_windows, self.motion_starts = [], [], []

    def vision(self, x):
        arr = np.asarray(x)
        self.vision_inputs.append(arr.copy())
        t, v = arr.shape[:2]
        value = arr.reshape(t, v, -1).mean(-1)[..., None, None]
        return jnp.asarray(np.broadcast_to(value, (t, v, 256, IMG_DIM)).astype(np.float32))

    def motion(self, frames, start):
        self.motion_windows.append(np.asarray(frames).copy())
        self.motion_starts.append(int(start))
        return np.full(DIM, float(np.asarray(frames).mean()), np.float32)

    def vision_frames(self):
        """按调用顺序拼成 (总帧数, v, 224, 224, 3)，下标即全域帧号。"""
        return np.concatenate(self.vision_inputs, axis=0)


def make_policy(motion=True):
    enc = Encoders()
    config = SimpleNamespace(budget=4, token_per_image=4, num_views=1, motion=SimpleNamespace(budget=16))
    policy = SimpleNamespace(config=config, motion_enabled=motion, metadata={"model_kind": "fake"}, encoders=enc)
    policy._motion_client = enc.motion if motion else None

    def reset():
        policy.mem_buffer = FrameSampMemory(num_views=1, img_emb_dim=IMG_DIM, pos_emb_dim=768, state_emb_dim=8,
                                            max_steps=64, vision_enc_fn=enc.vision, token_per_image=4,
                                            motion_enc_fn=policy._motion_client,
                                            motion_cfg=MOTION_CFG if motion else None)
        policy.step_idx, policy.exec_start_idx = -1, 0

    def add_buffer(obs):
        images, states = obs["images"], obs["state"]
        es = int(obs.get("exec_start_idx", 0)) or policy.exec_start_idx
        steps = list(range(policy.step_idx + 1, policy.step_idx + len(images) + 1))
        if motion:
            policy.mem_buffer.add_buffer(images, states, steps, exec_start_idx=es)
        else:
            policy.mem_buffer.add_buffer(images, states, steps)
        policy.exec_start_idx = es
        policy.step_idx += len(images)

    observation = SimpleNamespace(static_mask=np.ones((1, 4), bool), motion_mask=np.ones((1, 3), bool),
                                  mem_order=np.arange(MEM)[None], tokenized_prompt_mask=np.ones((1, 3), bool),
                                  motion_emb=np.zeros((1, 3, DIM), np.float32),
                                  motion_pos=np.zeros((1, 3, DIM), np.float32))

    def infer(obs):
        if motion:
            policy.mem_buffer._prepare_motion(policy.step_idx)
        return {"actions": policy._sample_actions(None, observation)}

    policy.reset, policy.add_buffer, policy.infer = reset, add_buffer, infer
    policy._sample_actions = lambda rng, obs, **kw: np.zeros((AH, SA.ACTION_DIM), np.float32)
    reset()
    return policy


def make_adapter():
    adapter = SimpleNamespace(last_attn=None)

    def make_sample_fn():
        def sample(rng, obs, **kw):
            adapter.last_attn = {"P": P, "MEM": MEM}
            return np.zeros((AH, SA.ACTION_DIM), np.float32)
        return sample

    adapter.make_sample_fn = make_sample_fn
    return adapter


def write_frame_donor(directory, frame_ids, four_dim=False):
    directory.mkdir(parents=True, exist_ok=True)
    images = np.stack([donor_image(f) for f in frame_ids])
    if four_dim:
        images = images[:, 0]
    np.savez(directory / "VideoUnmaskSwap_ep7.npz", frame_ids=np.asarray(frame_ids, np.int32), images=images)
    return directory


def batches():
    """首包 = 示范 6 帧 + exec 第零帧，之后 exec 每包 2 帧，共 13 帧（全域帧号 0..12）。"""
    yield {"images": np.stack([receiver_image(s) for s in range(ES + 1)]),
           "state": np.zeros((ES + 1, 8), np.float32), "exec_start_idx": ES}
    for first in range(ES + 1, 13, 2):
        yield {"images": np.stack([receiver_image(s) for s in (first, first + 1)]),
               "state": np.zeros((2, 8), np.float32), "exec_start_idx": 0}


def drive(policy, nonce="n0"):
    SA._RPC.update(op="reset", rpc={"episode_nonce": nonce, "episode_key": TARGET})
    policy.reset()
    for obs in batches():
        policy.add_buffer(obs)
        policy.infer({"observation/state": np.zeros(8, np.float32), "prompt": "p"})
    return policy.encoders


def recorded(tmp_path, ablate, motion=True, **kwargs):
    policy = make_policy(motion)
    rec = SA.AttnRecorder(policy, make_adapter(), tmp_path / "attn", ablate=ablate, **kwargs)
    rec.install()
    return rec, policy


def preprocessed(image_batch):
    """与 FrameSampMemory.add_buffer 同式的归一化 + resize（用来算供体图像应有的 vision 输入）。"""
    enc = Encoders()
    memory = FrameSampMemory(num_views=1, img_emb_dim=IMG_DIM, pos_emb_dim=768, state_emb_dim=8, max_steps=64,
                             vision_enc_fn=enc.vision, token_per_image=4)
    memory.add_buffer(image_batch, np.zeros((len(image_batch), 8), np.float32), list(range(len(image_batch))))
    return enc.vision_frames()


# ―― 解析与参数规则 ――

def test_parse_frame_swap_and_combination():
    parsed = SA.parse_ablate("frame_swap_exact:/d")
    assert parsed["frame"] == {"kind": "swap_exact", "dir": "/d"} and parsed["motion"] is None
    combined = SA.parse_ablate("frame_swap_exact:/f+motion_swap_exact:/m+heads:L3H0")
    assert combined["frame"]["dir"] == "/f" and combined["motion"] == {"kind": "swap_exact", "dir": "/m"}
    assert SA.parse_ablate("none")["frame"] is None and SA.parse_ablate("motion_shift")["frame"] is None
    for bad, message in (("frame_swap_exact:", "后须给供体目录"), ("frame_swap_exact", "未知项"),
                         ("frame_swap_exact:/a+frame_swap_exact:/b", "只能出现一次")):
        with pytest.raises(ValueError, match=message):
            SA.parse_ablate(bad)


def test_frame_swap_does_not_save_attn():
    parsed = SA.parse_ablate("frame_swap_exact:/d")
    assert SA.replaces_frames(parsed) and not SA.replaces_motion(parsed)
    assert SA.resolve_save_attn(parsed) is False
    with pytest.raises(SystemExit, match="帧 token 替换实验"):
        SA.resolve_save_attn(parsed, True)
    with pytest.raises(SystemExit, match="互斥"):
        SA.resolve_save_attn(parsed, save_prefix=True)


def test_frame_swap_needs_swap_map(tmp_path):
    with pytest.raises(SystemExit, match="只配 --swap-map"):
        SA.AttnRecorder(make_policy(), None, tmp_path, ablate=f"frame_swap_exact:{tmp_path}")
    with pytest.raises(SystemExit, match="只配 --swap-map"):
        SA.AttnRecorder(make_policy(), None, tmp_path, ablate=f"frame_swap_exact:{tmp_path}", swap_donor=DONOR)
    rec = SA.AttnRecorder(make_policy(), None, tmp_path, ablate=f"frame_swap_exact:{tmp_path}",
                          swap_map={TARGET: DONOR})
    assert rec.donor is None and rec.motion_client is None and rec.save_attn is False
    with pytest.raises(SystemExit, match="同时给出"):     # 既有规则：无替换时不收 --swap-map
        SA.AttnRecorder(make_policy(), None, tmp_path, ablate="none", swap_map={TARGET: DONOR})


def test_load_frame_donor_validation(tmp_path):
    donor = SA.load_frame_donor(write_frame_donor(tmp_path / "a", [1, 2], four_dim=True), DONOR)
    assert donor["images"].shape == (2, 1, SIZE, SIZE, 3) and donor["index"] == {1: 0, 2: 1}
    with pytest.raises(SystemExit, match="不存在"):
        SA.load_frame_donor(tmp_path / "missing", DONOR)
    bad = tmp_path / "b"
    bad.mkdir()
    np.savez(bad / "VideoUnmaskSwap_ep7.npz", frame_ids=np.asarray([1, 1]), images=np.zeros((2, 1, 2, 2, 3), np.uint8))
    with pytest.raises(SystemExit, match="重复"):
        SA.load_frame_donor(bad, DONOR)
    np.savez(bad / "VideoUnmaskSwap_ep7.npz", frame_ids=np.asarray([1]), images=np.zeros((1, 1, 2, 2, 3), np.float32))
    with pytest.raises(SystemExit, match="uint8"):
        SA.load_frame_donor(bad, DONOR)


# ―― 真实记忆：帧输入换、motion 输入不换 ――

def test_frame_swap_replaces_vision_keeps_motion(tmp_path):
    donors = write_frame_donor(tmp_path / "donors", [1, 2, 99])     # 99 不在本局：不替换也不报错
    plain = drive(make_policy())                                   # 裸策略对照
    rec, policy = recorded(tmp_path, f"frame_swap_exact:{donors}", swap_map={TARGET: DONOR})
    swapped = drive(policy)

    base_v, swap_v = plain.vision_frames(), swapped.vision_frames()
    assert base_v.shape == swap_v.shape == (13, 1, 224, 224, 3)
    expected = preprocessed(np.stack([donor_image(1), donor_image(2)]))
    np.testing.assert_array_equal(swap_v[1:3], expected)
    assert not np.array_equal(swap_v[1:3], base_v[1:3])
    keep = [i for i in range(13) if i not in (1, 2)]
    np.testing.assert_array_equal(swap_v[keep], base_v[keep])

    # motion：起点序列与每个窗口的帧逐字节相同（含覆盖帧 1、2 的示范窗口）
    assert swapped.motion_starts == plain.motion_starts and 0 in plain.motion_starts
    assert len(swapped.motion_windows) == len(plain.motion_windows) > 0
    for left, right in zip(swapped.motion_windows, plain.motion_windows):
        np.testing.assert_array_equal(left, right)

    info = json.loads((tmp_path / "attn" / "n0" / "episode.json").read_text(encoding="utf-8"))
    content = info["frame_content"]
    assert content["mode"] == "swap_exact" and content["donor"] == DONOR
    assert content["replace_log"] == [1, 2] and content["n_replaced"] == 2
    assert content["n_motion_restored"] == 2 and content["n_donor_frames"] == 3
    assert info["swap_donor"] == DONOR and info["episode_key"] == TARGET and info["attn_saved"] is False
    assert "motion_content" not in info and not list((tmp_path / "attn" / "n0").glob("*.npz"))


def plain_and_swapped_embs(policy):
    return sorted(policy.mem_buffer._history_feats_motion.items())


def test_frame_swap_motion_embeddings_equal_plain(tmp_path):
    donors = write_frame_donor(tmp_path / "donors", [0, 1, 2, 3, 4, 5])
    plain_policy = make_policy()
    drive(plain_policy)
    rec, policy = recorded(tmp_path, f"frame_swap_exact:{donors}", swap_map={TARGET: DONOR})
    drive(policy)
    left, right = plain_and_swapped_embs(plain_policy), plain_and_swapped_embs(policy)
    assert [s for s, _ in left] == [s for s, _ in right]
    for (_, a), (_, b) in zip(left, right):
        np.testing.assert_array_equal(a, b)
    # 帧记忆里的示范帧特征确实变了
    for step in range(ES):
        assert not np.array_equal(plain_policy.mem_buffer._history_feats[step]["image_emb_2x2"],
                                  policy.mem_buffer._history_feats[step]["image_emb_2x2"])
    assert rec.frame_log == list(range(ES)) and rec.frame_motion_restored == ES


def test_frame_swap_per_episode_and_baseline_model(tmp_path):
    donors = write_frame_donor(tmp_path / "donors", [2])
    rec, policy = recorded(tmp_path, f"frame_swap_exact:{donors}", motion=False, swap_map={TARGET: DONOR})
    enc = drive(policy, "a")
    np.testing.assert_array_equal(enc.vision_frames()[2:3], preprocessed(donor_image(2)[None]))
    assert enc.motion_windows == [] and rec.frame_log == [2] and rec.frame_motion_restored == 0
    drive(policy, "b")                                             # 换局：日志清零后重计
    info = json.loads((tmp_path / "attn" / "b" / "episode.json").read_text(encoding="utf-8"))
    assert info["frame_content"]["replace_log"] == [2]


def test_frame_swap_shape_mismatch_raises(tmp_path):
    donors = tmp_path / "donors"
    donors.mkdir()
    np.savez(donors / "VideoUnmaskSwap_ep7.npz", frame_ids=np.asarray([1]),
             images=np.zeros((1, 1, SIZE + 1, SIZE, 3), np.uint8))
    rec, policy = recorded(tmp_path, f"frame_swap_exact:{donors}", swap_map={TARGET: DONOR})
    SA._RPC.update(op="reset", rpc={"episode_nonce": "x", "episode_key": TARGET})
    policy.reset()
    with pytest.raises(RuntimeError, match="不符"):
        policy.add_buffer(next(batches()))
    assert rec._frame_restore is None


# ―― 关闭时逐字节不变 ――

def test_off_is_byte_identical(tmp_path):
    plain_policy = make_policy()
    plain = drive(plain_policy)
    rec, policy = recorded(tmp_path, "none", save_attn=False)   # 假 adapter 不产注意力数组，走不存路径
    wrapped = drive(policy)
    assert rec.frame_donor is None and "frame_content" not in json.loads(
        (tmp_path / "attn" / "n0" / "episode.json").read_text(encoding="utf-8"))
    for a, b in zip(plain.vision_inputs, wrapped.vision_inputs):
        np.testing.assert_array_equal(a, b)
    assert len(plain.vision_inputs) == len(wrapped.vision_inputs)
    assert plain.motion_starts == wrapped.motion_starts
    for a, b in zip(plain.motion_windows, wrapped.motion_windows):
        np.testing.assert_array_equal(a, b)
    # 关闭时 mem_buffer 的 _encode_ready_windows 未被包装
    assert "_encode_ready_windows" not in vars(policy.mem_buffer)
