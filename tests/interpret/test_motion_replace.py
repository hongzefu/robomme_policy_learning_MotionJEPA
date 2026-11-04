"""S8 单测：motion 内容替换（record / motion_zero / motion_swap）的包装句柄、供体缺窗回退、serve 侧记录与 run_local 命令形状（CPU）。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.interpret import run_local as RL
from scripts.interpret import serve_policy_attn as SA

DIM = 1024
SENTINEL = int(np.iinfo(np.int32).max)


class FakeClient:
    """模仿 MotionEncoderClient：(frames, start_frame) -> (dim,) f32，内容由 seed 与起点决定；带 provenance / close。"""

    def __init__(self, seed=0):
        self.seed, self.calls, self.closed = seed, [], False
        self.provenance = {"kind": "fake", "seed": seed}

    def __call__(self, frames, start_frame):
        self.calls.append(int(start_frame))
        return np.random.default_rng(self.seed * 100000 + int(start_frame)).standard_normal(DIM).astype(np.float32)

    def close(self):
        self.closed = True


def _emb(seed, start):
    return np.random.default_rng(seed * 100000 + start).standard_normal(DIM).astype(np.float32)


# ―――――――――――――――――――――――― 解析 / 回退规则 ――――――――――――――――――――――――

def test_parse_ablate_motion():
    assert SA.parse_ablate("motion_zero")["motion"] == {"kind": "zero"}
    got = SA.parse_ablate("motion_swap:/d/donor")
    assert got["motion"] == {"kind": "swap", "dir": "/d/donor"} and got["masks"] == frozenset() and got["heads"] == ()
    combo = SA.parse_ablate("mask_vis+motion_zero+heads:L3H0")
    assert combo["motion"] == {"kind": "zero"} and combo["masks"] == {"mask_vis"} and combo["heads"] == ((3, 0),)
    assert SA.parse_ablate("mask_vis")["motion"] is None


@pytest.mark.parametrize("bad", ["motion_swap:", "motion_zero+motion_zero", "motion_zero+motion_swap:/d", "motion_one"])
def test_parse_ablate_motion_rejects(bad):
    with pytest.raises(ValueError):
        SA.parse_ablate(bad)


def test_donor_lookup_fallback():
    starts = [0, 16, 32, 48]
    assert SA.donor_lookup(starts, 32) == 2                     # 同起点
    assert SA.donor_lookup(starts, 40) == 2                     # 缺窗：起点 ≤ 40 的最大窗口 32
    assert SA.donor_lookup(starts, 1000) == 3
    assert SA.donor_lookup([16, 32], 5) == 0                    # 没有 ≤ 的：第一个窗口
    assert SA.donor_lookup([32, 16], 20) == 1                   # 按起点值取最大，不按下标
    with pytest.raises(ValueError):
        SA.donor_lookup([], 0)
    assert SA.parse_episode("ButtonUnmaskSwap:04") == ("ButtonUnmaskSwap", 4)
    assert SA.record_name("ButtonUnmaskSwap", 4) == "ButtonUnmaskSwap_ep4.npz"
    with pytest.raises(ValueError):
        SA.parse_episode("ButtonUnmaskSwap")


# ―――――――――――――――――――――――― 包装句柄 ――――――――――――――――――――――――

def test_wrapper_record_then_swap_roundtrip(tmp_path):
    rec = tmp_path / "rec"
    inner = FakeClient(seed=8)
    w = SA.MotionContentClient(inner, mode="none", record_dir=rec)
    assert SA._accepts_start(w) and w.provenance == inner.provenance       # 签名与属性透传
    w.begin_episode("nonceA")
    frames = np.zeros((33, 2, 4, 4, 3), np.uint8)
    for f in (0, 16, 32, 16):                                              # 16 重复：保留首次、计次
        out = w(frames, f)
        np.testing.assert_array_equal(out, _emb(8, f))                     # none：原样返回
    path = w.flush()
    assert path == rec / "nonceA.npz" and w.flush() is None                # 无变化不重写
    with np.load(path) as z:
        assert z["start_frames"].dtype == np.int32 and z["start_frames"].tolist() == [0, 16, 32]
        assert z["embs"].dtype == np.float32 and z["embs"].shape == (3, DIM)
        assert z["counts"].tolist() == [1, 2, 1] and str(z["mode"]) == "none"
        for i, f in enumerate((0, 16, 32)):
            np.testing.assert_array_equal(z["embs"][i], _emb(8, f))
    w.close()
    assert inner.closed

    # 供体文件按 run_local 改名规则就位，再用另一局（seed=4）走 swap 回读
    (rec / "nonceA.npz").rename(rec / SA.record_name("ButtonUnmaskSwap", 8))
    donor = SA.load_donor(rec, "ButtonUnmaskSwap:8")
    target_inner = FakeClient(seed=4)
    rec2 = tmp_path / "rec2"
    s = SA.MotionContentClient(target_inner, mode="swap", donor=donor, record_dir=rec2)
    s.begin_episode("nonceB")
    expect_donor = {0: 0, 16: 16, 32: 32, 48: 32, 8: 0}
    for f, d in expect_donor.items():
        np.testing.assert_array_equal(s(frames, f), _emb(8, d))           # 换成供体同起点 / 回退窗口
    assert target_inner.calls == list(expect_donor)                        # 原 client 照常被调用
    assert s.replaced == 5 and s.replace_log == [[f, d] for f, d in expect_donor.items()]
    s.flush()
    with np.load(rec2 / "nonceB.npz") as z:                                # 记录的是替换前的原编码
        assert z["start_frames"].tolist() == list(expect_donor)
        assert z["donor_start_frames"].tolist() == list(expect_donor.values())
        np.testing.assert_array_equal(z["embs"][3], _emb(4, 48))
        assert str(z["mode"]) == "swap" and str(z["donor"]) == "ButtonUnmaskSwap:8"
    summary = s.summary()
    assert summary["n_replaced"] == 5 and summary["donor"] == "ButtonUnmaskSwap:8" and summary["n_windows"] == 5


def test_wrapper_zero(tmp_path):
    inner = FakeClient(seed=1)
    w = SA.MotionContentClient(inner, mode="zero")
    w.begin_episode("n")
    out = w(np.zeros((33, 1), np.uint8), 16)
    assert out.dtype == np.float32 and out.shape == (DIM,) and not out.any()
    assert inner.calls == [16] and w.replaced == 1 and w.replace_log == [[16, -1]]
    assert w.flush() is None                                               # 未给 record_dir 不落盘
    w.begin_episode("n2")
    assert w.replaced == 0 and w.replace_log == []                         # 每局清零


def test_load_donor_missing(tmp_path):
    with pytest.raises(SystemExit):
        SA.load_donor(tmp_path, "ButtonUnmaskSwap:8")


# ―――――――――――――――――――――――― serve 侧记录（假 policy）――――――――――――――――――――――――

class FakeMem:
    """持 motion_enc；add_buffer 时为起点 0,16,... 且 f+32 ≤ step 的窗口调用编码。"""

    def __init__(self, enc):
        self.motion_enc, self.feats = enc, {}

    def encode(self, step_idx):
        f = 0
        while f + 32 <= step_idx:
            if f not in self.feats:
                self.feats[f] = np.asarray(self.motion_enc(np.zeros((33, 1), np.uint8), f), np.float32)
            f += 16

    def get_frame_sampling_indices(self, step_idx, budget, token_per_image):
        return list(range(0, step_idx + 1, 4))[:32]

    def _prepare_motion(self, step_idx):
        mask = np.zeros(96, bool)
        mask[:len(self.feats)] = True
        times = np.full(96, SENTINEL, np.int64)
        times[:len(self.feats)] = sorted(self.feats)
        return np.zeros((96, DIM), np.float32), np.zeros((96, 4), np.float32), mask, times


class FakePolicy:
    def __init__(self, client):
        self.motion_enabled = True
        self._motion_client = client
        self.config = SimpleNamespace(budget=512, token_per_image=8, num_views=2, motion=SimpleNamespace(budget=96))
        self.metadata = {"model_kind": "stage2_pool4_concat"}
        self.reset()

    def reset(self):
        self.mem_buffer = FakeMem(self._motion_client)
        self.step_idx, self.exec_start_idx = -1, 0

    def add_buffer(self, obs):
        self.step_idx += len(obs["images"])
        self.mem_buffer.encode(self.step_idx)

    def infer(self, obs):
        n_frames = len(self.mem_buffer.get_frame_sampling_indices(self.step_idx, 512, 8))
        static_mask = np.repeat(np.arange(32) < n_frames, 16)[None]
        motion_emb, motion_pos, motion_mask, _ = self.mem_buffer._prepare_motion(self.step_idx)
        observation = SimpleNamespace(static_mask=static_mask, motion_mask=motion_mask[None],
                                      motion_emb=motion_emb[None], motion_pos=motion_pos[None],
                                      mem_order=np.arange(608, dtype=np.int32)[None],
                                      tokenized_prompt_mask=(np.arange(64) < 21)[None])
        return {"actions": self._sample_actions(None, observation), "state": np.zeros(8, np.float32)}


class Adapter:
    P, MEM, last_attn = 1184, 608, None

    def make_sample_fn(self):
        def sample(rng, observation, **kw):
            self.last_attn = {"probs_action": np.zeros((10, 18, 8, 20, self.P + 20), np.float16),
                              "probs_text": np.zeros((18, 8, 64, self.P), np.float16), "P": self.P, "MEM": self.MEM}
            return np.zeros((20, 8), np.float32)
        return sample


def _episode(tmp_path, monkeypatch, client, nonce, **kw):
    policy = FakePolicy(client)
    recorder = SA.AttnRecorder(policy, Adapter(), tmp_path / "attn", save_top_frames=0, **kw)
    recorder.install()
    monkeypatch.setitem(SA._RPC, "op", "reset")
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": nonce})
    policy.reset()
    policy.add_buffer({"images": np.zeros((50, 1), np.uint8)})          # step 49 → 窗口 0、16
    policy.infer({"observation/state": np.zeros(8), "prompt": "p"})
    policy.add_buffer({"images": np.zeros((16, 1), np.uint8)})          # step 65 → 再加 32
    policy.infer({"observation/state": np.zeros(8), "prompt": "p"})
    return policy, recorder, tmp_path / "attn" / nonce


def test_recorder_record_then_swap(tmp_path, monkeypatch):
    rec = tmp_path / "rec"
    donor_client = FakeClient(seed=8)
    policy, recorder, ep = _episode(tmp_path, monkeypatch, donor_client, "donor", record_motion=rec)
    assert isinstance(policy._motion_client, SA.MotionContentClient) and policy.mem_buffer.motion_enc is policy._motion_client
    for f in (0, 16, 32):                                               # none + record：原样
        np.testing.assert_array_equal(policy.mem_buffer.feats[f], _emb(8, f))
    with np.load(ep / "infer_0001.npz") as z:
        assert "n_motion_replaced" not in z.files                       # 只记录不替换：键表不变
    with np.load(rec / "donor.npz") as z:
        assert z["start_frames"].tolist() == [0, 16, 32]
    (rec / "donor.npz").rename(rec / "ButtonUnmaskSwap_ep8.npz")

    target = FakeClient(seed=4)
    policy, recorder, ep = _episode(tmp_path, monkeypatch, target, "target", ablate=f"motion_swap:{rec}",
                                    swap_donor="ButtonUnmaskSwap:8")
    assert target.calls == [0, 16, 32]
    for f in (0, 16, 32):
        np.testing.assert_array_equal(policy.mem_buffer.feats[f], _emb(8, f))
    with np.load(ep / "infer_0000.npz") as z0, np.load(ep / "infer_0001.npz") as z1:
        assert set(z1.files) == (set(SA.NPZ_KEYS) | set(SA.ABLATE_NPZ_KEYS)
                                 | set(SA.MOTION_REPLACE_NPZ_KEYS) | set(SA.MOTION_INPUT_NPZ_KEYS))
        assert int(z0["n_motion_replaced"]) == 2 and int(z1["n_motion_replaced"]) == 3
        assert int(z1["n_masked_keys"]) == 0 and str(z1["ablate"]) == f"motion_swap:{rec}"
        assert z1["motion_mask"].sum() == 3                              # 位置 / mask 不动
    info = json.loads((ep / "episode.json").read_text())
    assert info["motion_content"]["replace_log"] == [[0, 0], [16, 16], [32, 32]]
    assert info["motion_content"]["donor"] == "ButtonUnmaskSwap:8"


def test_recorder_zero(tmp_path, monkeypatch):
    policy, _, ep = _episode(tmp_path, monkeypatch, FakeClient(seed=3), "z", ablate="motion_zero")
    assert all(not v.any() for v in policy.mem_buffer.feats.values()) and len(policy.mem_buffer.feats) == 3
    with np.load(ep / "infer_0001.npz") as z:
        assert int(z["n_motion_replaced"]) == 3


def test_recorder_motion_checks(tmp_path):
    base = FakePolicy(FakeClient())
    base.motion_enabled = False
    with pytest.raises(SystemExit):                                     # 基线不得 motion 替换
        SA.AttnRecorder(base, Adapter(), tmp_path / "a", ablate="motion_zero")
    with pytest.raises(SystemExit):                                     # swap 缺 --swap-donor
        SA.AttnRecorder(FakePolicy(FakeClient()), Adapter(), tmp_path / "a", ablate=f"motion_swap:{tmp_path}")
    with pytest.raises(SystemExit):                                     # 供体文件不存在
        SA.AttnRecorder(FakePolicy(FakeClient()), Adapter(), tmp_path / "a", ablate=f"motion_swap:{tmp_path}",
                        swap_donor="ButtonUnmaskSwap:8")


def test_serve_help_has_record_motion():
    pytest.importorskip("tyro")
    import dataclasses
    try:
        Args = SA._make_args_class()
    except Exception as exc:                                            # noqa: BLE001
        pytest.skip(f"serve_policy 不可导入: {type(exc).__name__}")
    names = {f.name: f.default for f in dataclasses.fields(Args)}
    assert names["record_motion"] is None and names["swap_donor"] is None


# ―――――――――――――――――――――――― run_local 命令形状 / spec ――――――――――――――――――――――――

SWAP_LIST = [4, 8, 11, 12, 17, 18, 20, 26, 34, 37, 38, 40]
SWAP_MAP = {f"ButtonUnmaskSwap:{a}": f"ButtonUnmaskSwap:{b}" for a, b in zip(SWAP_LIST, SWAP_LIST[1:] + SWAP_LIST[:1])}


def _cmd_spec(**kw):
    spec = {"policy_python": "/p/python", "stage2_python": "/s/python", "run_identity_sha256": "a" * 64,
            "checkpoint": "/ck", "policy_config": "cfg", "model_kind": "stage2_pool4_concat", "cuda": "1",
            "attn_out": "/attn"}
    spec.update(kw)
    return spec


def test_swap_map_parse():
    assert SWAP_MAP["ButtonUnmaskSwap:40"] == "ButtonUnmaskSwap:4" and SWAP_MAP["ButtonUnmaskSwap:4"] == "ButtonUnmaskSwap:8"
    text = ",".join(f"{k}={v}" for k, v in SWAP_MAP.items())
    assert RL.parse_swap_map(text) == SWAP_MAP == RL.parse_swap_map(SWAP_MAP)
    with pytest.raises(SystemExit):
        RL.parse_swap_map("ButtonUnmaskSwap:4")


def test_server_command_record_and_swap():
    command = RL.build_server_command(_cmd_spec(record_motion="/rec"), "n" * 32, 9100)
    i = command.index("policy:checkpoint")
    assert command[i - 4:i] == ["--attn-out", "/attn", "--record-motion", "/rec"]
    spec = _cmd_spec(ablate="motion_swap:/rec", swap_map=SWAP_MAP)
    ep = [{"episode_key": "ButtonUnmaskSwap:12"}]
    command = RL.build_server_command(spec, "n" * 32, 9100, ep)
    i = command.index("policy:checkpoint")
    assert command[i - 6:i] == ["--attn-out", "/attn", "--ablate", "motion_swap:/rec", "--swap-donor", "ButtonUnmaskSwap:17"]
    with pytest.raises(ValueError):
        RL.build_server_command(spec, "n" * 32, 9100, ep * 2)            # swap 要求单局块
    command = RL.build_server_command(_cmd_spec(ablate="motion_zero"), "n" * 32, 9100)
    assert command[command.index("--ablate") + 1] == "motion_zero" and "--swap-donor" not in command


def test_load_spec_swap_rules(tmp_path):
    def load(**kw):
        path = tmp_path / "spec.json"
        spec = {k: "x" for k in RL.REQUIRED_KEYS}
        spec.update(model_kind="stage2_pool4_concat", cuda=1)
        spec.update(kw)
        path.write_text(json.dumps(spec))
        return RL.load_spec(path)

    got = load(ablate="motion_swap:/rec", swap_map=",".join(f"{k}={v}" for k, v in SWAP_MAP.items()))
    assert got["swap_map"] == SWAP_MAP
    for bad in (dict(ablate="motion_swap:/rec"), dict(ablate="motion_zero", swap_map=SWAP_MAP),
                dict(ablate="motion_zero", model_kind="history"), dict(record_motion="/rec", model_kind="history")):
        with pytest.raises(SystemExit):
            load(**bad)
    manifest = {"sha256": "f" * 64}
    spec = {"checkpoint": "/ck", "policy_config": "cfg", "model_kind": "stage2_pool4_concat", "model_seed": 42,
            "ablate": "motion_swap:/rec"}
    a = RL.run_identity(dict(spec, swap_map=SWAP_MAP), manifest)
    b = RL.run_identity(dict(spec, swap_map={**SWAP_MAP, "ButtonUnmaskSwap:4": "ButtonUnmaskSwap:11"}), manifest)
    assert a != b
    base = {k: v for k, v in spec.items() if k != "ablate"}
    assert RL.run_identity(dict(base, record_motion="/rec"), manifest) == RL.run_identity(base, manifest)


def test_select_chunks_single():
    manifest = {"episodes": [{"episode_key": k} for k in ("A:1", "A:2", "A:3")],
                "chunks": [{"chunk_id": "c0", "episode_keys": ["A:1", "A:2"]}, {"chunk_id": "c1", "episode_keys": ["A:3"]}]}
    assert [c for c, _ in RL.select_chunks(manifest)] == ["c0", "c1"]
    single = RL.select_chunks(manifest, single=True)
    assert [[e["episode_key"] for e in eps] for _, eps in single] == [["A:1"], ["A:2"], ["A:3"]]
    assert single[0][0] == RL.M.digest(["A:1"])


def test_rename_motion_record(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "abc.npz").write_bytes(b"x")
    episode = {"task_name": "ButtonUnmaskSwap", "episode_id": 4}
    got = RL.rename_motion_record({"record_motion": str(rec)}, "abc", episode)
    assert got == str(rec / "ButtonUnmaskSwap_ep4.npz") and not (rec / "abc.npz").exists()
    assert RL.rename_motion_record({}, "abc", episode) is None
