"""S2 单测：run_local 驱动（假服务端 / 假客户端）、build_manifest，以及 serve 侧 npz 写盘契约（假 adapter + 假 policy）。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation import manifest as M
from scripts.interpret import build_manifest as BM
from scripts.interpret import run_local as RL
from scripts.interpret import serve_policy_attn as SA

REPO = Path(__file__).resolve().parents[2]
SENTINEL = int(np.iinfo(np.int32).max)

# ―――――――――――――――――――――――― 假服务端 / 假客户端 ――――――――――――――――――――――――

FAKE_SERVER = r'''
import http.server, json, os, signal, sys
argv = sys.argv[1:]
port = int(argv[argv.index("--port") + 1])
log_dir = os.environ["FAKE_LOG_DIR"]
with open(os.path.join(log_dir, f"server-{port}.json"), "w") as f:
    json.dump({"argv": argv, "env": {k: os.environ.get(k) for k in
               ("CUDA_VISIBLE_DEVICES", "XLA_FLAGS", "XLA_PYTHON_CLIENT_MEM_FRACTION", "PYTHONPATH")}}, f)
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/healthz" else 404); self.end_headers(); self.wfile.write(b"OK\n")
    def log_message(self, *a): pass
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
print("FAKE_SERVER_READY", port, flush=True)
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
'''

FAKE_CLIENT = r'''
import json, os, sys, uuid
argv = sys.argv[1:]
get = lambda k: argv[argv.index(k) + 1]
grant = json.load(open(get("--episode-manifest")))
out = get("--attempt-output")
os.makedirs(out, exist_ok=False)
log_dir, attn = os.environ["FAKE_LOG_DIR"], os.environ["FAKE_ATTN_OUT"]
n = len([p for p in os.listdir(log_dir) if p.startswith("client-")])
json.dump({"argv": argv, "grant": grant}, open(os.path.join(log_dir, f"client-{n:03d}.json"), "w"))
for e in grant["episodes"]:
    nonce = uuid.uuid4().hex
    ident = dict(e, episode_nonce=nonce, run_identity_sha256=grant["run_identity_sha256"], chunk_id=grant["chunk_id"])
    json.dump(dict(ident, record_type="started"), open(os.path.join(out, e["episode_key"] + ".started.json"), "w"))
    d = os.path.join(attn, nonce); os.makedirs(d)
    json.dump({"episode_nonce": nonce}, open(os.path.join(d, "episode.json"), "w"))
    for i in range(2):
        open(os.path.join(d, f"infer_{i:04d}.npz"), "wb").close()
    outcome = "success" if e["episode_id"] % 2 == 0 else "fail"
    json.dump(dict(ident, record_type="terminal", outcome=outcome, rpc_seq=1 + 2 * 2),
              open(os.path.join(out, e["episode_key"] + ".terminal.json"), "w"))
'''


def _exe(path: Path, body: str) -> str:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _manifest(tmp_path, pairs, chunk_size):
    meta = BM.default_metadata_root()
    if not any(meta.glob("record_dataset_*_metadata.json")):
        pytest.skip("官方元数据不可读")
    manifest = BM.build(pairs, meta, chunk_size)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return manifest, path


def _spec(tmp_path, manifest_path, model_kind):
    return {"repo": str(REPO), "policy_python": _exe(tmp_path / "fake_policy_python", FAKE_SERVER),
            "sim_python": _exe(tmp_path / "fake_sim_python", FAKE_CLIENT), "stage2_python": "/fake/stage2/python",
            "checkpoint": str(tmp_path / "ckpt"), "policy_config": "cfg_x", "model_kind": model_kind,
            "model_seed": 42, "purpose": "validation", "cuda": "1", "attn_out": str(tmp_path / "attn"),
            "run_root": str(tmp_path / "run"), "manifest": str(manifest_path), "xla_mem_fraction": 0.4,
            "startup_timeout_seconds": 30}


# ―――――――――――――――――――――――― 命令形状 ――――――――――――――――――――――――

@pytest.mark.parametrize("model_kind", ["history", "stage2_pool4_concat"])
def test_server_command_shape(model_kind):
    spec = {"policy_python": "/p/python", "stage2_python": "/s/python", "run_identity_sha256": "a" * 64,
            "checkpoint": "/ck", "policy_config": "cfg", "model_kind": model_kind, "cuda": "1", "attn_out": "/attn"}
    command = RL.build_server_command(spec, "n" * 32, 9100)
    assert command[:3] == ["/p/python", "-m", "scripts.interpret.serve_policy_attn"]
    assert "scripts.training.serve_policy" not in command
    i = command.index("policy:checkpoint")
    assert command[i - 2:i] == ["--attn-out", "/attn"]
    assert command[i:] == ["policy:checkpoint", "--policy.config", "cfg", "--policy.dir", "/ck"]
    if model_kind == "history":
        assert "--motion-online-gpu" not in command
    else:
        assert command[command.index("--motion-online-gpu") + 1] == "1"
    assert command[command.index("--server-nonce") + 1] == "n" * 32
    assert command[command.index("--port") + 1] == "9100"


def test_run_identity_and_bad_model_kind(tmp_path):
    manifest, path = _manifest(tmp_path, [("ButtonUnmaskSwap", 4)], 1)
    spec = _spec(tmp_path, path, "history")
    first = RL.run_identity(spec, manifest)
    assert len(first) == 64 and first == RL.run_identity(dict(spec), manifest)
    assert first != RL.run_identity(dict(spec, model_kind="stage2_pool4_concat"), manifest)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dict(spec, model_kind="frame_sampling")))
    with pytest.raises(SystemExit):
        RL.load_spec(bad)


# ―――――――――――――――――――――――― 端到端（假进程）――――――――――――――――――――――――

@pytest.mark.parametrize("model_kind", ["history", "stage2_pool4_concat"])
def test_run_local_end_to_end(tmp_path, monkeypatch, capsys, model_kind):
    log_dir = tmp_path / "fake_logs"
    log_dir.mkdir()
    monkeypatch.setenv("FAKE_LOG_DIR", str(log_dir))
    monkeypatch.setenv("FAKE_ATTN_OUT", str(tmp_path / "attn"))
    pairs = [("ButtonUnmaskSwap", 4), ("ButtonUnmaskSwap", 8), ("RouteStick", 1)]
    manifest, mpath = _manifest(tmp_path, pairs, 5)
    spec = _spec(tmp_path, mpath, model_kind)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    code = RL.main(["--spec", str(spec_path), "--episodes", "RouteStick:1,ButtonUnmaskSwap:4"])
    out = capsys.readouterr().out
    assert code == 0, out
    identity = RL.run_identity(RL.load_spec(spec_path), manifest)

    # chunk 顺序 = --episodes 顺序，每局单独一块
    clients = [json.loads(p.read_text()) for p in sorted(log_dir.glob("client-*.json"))]
    assert [c["grant"]["episodes"][0]["episode_key"] for c in clients] == ["RouteStick:1", "ButtonUnmaskSwap:4"]
    for c in clients:
        grant = c["grant"]
        assert set(grant) == {"chunk_id", "run_identity_sha256", "episodes"}
        assert grant["run_identity_sha256"] == identity
        assert grant["chunk_id"] == M.digest([grant["episodes"][0]["episode_key"]])
        argv = c["argv"]
        assert argv[0].endswith("examples/robomme/eval.py")
        assert argv[argv.index("--chunk") + 1] == grant["chunk_id"]
        assert argv[argv.index("--model-kind") + 1] == model_kind
        assert argv[argv.index("--run-identity-sha256") + 1] == identity

    servers = [json.loads(p.read_text()) for p in log_dir.glob("server-*.json")]
    assert len(servers) == 2
    for s in servers:
        argv = s["argv"]
        assert argv[:2] == ["-m", "scripts.interpret.serve_policy_attn"]
        assert argv[argv.index("policy:checkpoint") - 2:argv.index("policy:checkpoint")] == ["--attn-out", spec["attn_out"]]
        assert ("--motion-online-gpu" in argv) == (model_kind == "stage2_pool4_concat")
        assert s["env"]["CUDA_VISIBLE_DEVICES"] == "1"
        assert s["env"]["XLA_FLAGS"] == RL.XLA_FLAGS
        assert s["env"]["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.4"
        assert s["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(REPO)
    # 两次 server_nonce 各自独立，且与客户端期望一致
    nonces = {s["argv"][s["argv"].index("--server-nonce") + 1] for s in servers}
    assert nonces == {c["argv"][c["argv"].index("--expected-server-nonce") + 1] for c in clients}

    accepted = sorted(p.name for p in (Path(spec["run_root"]) / "accepted").glob("*.terminal.json"))
    assert len(accepted) == 2
    lines = [line for line in out.splitlines() if line.startswith(("EPISODE_RESULT", "RUN_LOCAL="))]
    assert lines == ["EPISODE_RESULT task=RouteStick ep=1 success=0 npz=2",
                     "EPISODE_RESULT task=ButtonUnmaskSwap ep=4 success=1 npz=2",
                     f"RUN_LOCAL=PASS model={model_kind} episodes=2 success=1 npz=4"]
    # episode.json 已并入局身份
    infos = [json.loads(p.read_text()) for p in Path(spec["attn_out"]).glob("*/episode.json")]
    assert sorted(i["episode_key"] for i in infos) == ["ButtonUnmaskSwap:4", "RouteStick:1"]


def test_run_local_server_dies(tmp_path, monkeypatch, capsys):
    log_dir = tmp_path / "fake_logs"
    log_dir.mkdir()
    monkeypatch.setenv("FAKE_LOG_DIR", str(log_dir))
    monkeypatch.setenv("FAKE_ATTN_OUT", str(tmp_path / "attn"))
    _, mpath = _manifest(tmp_path, [("ButtonUnmaskSwap", 4), ("ButtonUnmaskSwap", 8)], 1)
    spec = _spec(tmp_path, mpath, "history")
    spec["policy_python"] = _exe(tmp_path / "dead_python", "import sys; print('boom', flush=True); sys.exit(7)")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    assert RL.main(["--spec", str(spec_path)]) == 1
    out = capsys.readouterr().out
    assert "CHUNK_ERROR index=0" in out and "boom" in out
    assert "CHUNK_BEGIN index=1" not in out          # 故障后不再起后续块
    assert out.strip().splitlines()[-1] == "RUN_LOCAL=FAIL model=history episodes=1 success=0 npz=0"


# ―――――――――――――――――――――――― build_manifest ――――――――――――――――――――――――

def test_build_manifest_cli(tmp_path, capsys):
    if not any(BM.default_metadata_root().glob("record_dataset_*_metadata.json")):
        pytest.skip("官方元数据不可读")
    out = tmp_path / "m.json"
    assert BM.main(["--episodes", "ButtonUnmaskSwap:4", "--out", str(out)]) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    manifest = json.loads(out.read_text())
    assert line == f"MANIFEST=PASS episodes=1 chunks=1 seal={manifest['sha256']}"
    assert manifest["episodes"][0] == {"task_name": "ButtonUnmaskSwap", "episode_id": 4,
                                       "env_seed": manifest["episodes"][0]["env_seed"],
                                       "difficulty": manifest["episodes"][0]["difficulty"],
                                       "episode_key": "ButtonUnmaskSwap:4"}
    M.validate_episodes(manifest, official=False)
    full = tmp_path / "full.json"
    assert BM.main(["--out", str(full)]) == 0
    assert capsys.readouterr().out.strip().startswith("MANIFEST=PASS episodes=20 chunks=4 ")


# ―――――――――――――――――――――――― serve 侧 npz 契约 ――――――――――――――――――――――――

class FakeMem:
    def __init__(self):
        self.motion_calls = 0

    def get_frame_sampling_indices(self, step_idx, budget, token_per_image):
        return list(range(0, step_idx + 1, 4))[:32]

    def _prepare_motion(self, step_idx):
        self.motion_calls += 1
        mask = np.zeros(96, bool)
        mask[:3] = True
        times = np.full(96, SENTINEL, np.int64)
        times[:3] = [0, 16, 32]
        return np.zeros((96, 768), np.float32), np.zeros((96, 4), np.float32), mask, times


class FakePolicy:
    def __init__(self, motion):
        self.motion_enabled = motion
        self.config = SimpleNamespace(budget=512, token_per_image=8, num_views=2, motion=SimpleNamespace(budget=96))
        self.metadata = {"model_kind": "stage2_pool4_concat" if motion else "history"}
        self.reset()

    def reset(self):
        self.mem_buffer = FakeMem()
        self.step_idx, self.exec_start_idx = -1, 0

    def add_buffer(self, obs):
        self.step_idx += len(obs["images"])

    def infer(self, obs):
        n_frames = len(self.mem_buffer.get_frame_sampling_indices(self.step_idx, 512, 8))
        static_mask = np.repeat(np.arange(32) < n_frames, 16)[None]
        prompt_mask = (np.arange(64) < 21)[None]
        if self.motion_enabled:
            _, _, motion_mask, _ = self.mem_buffer._prepare_motion(self.step_idx)
            mem_order = np.random.default_rng(0).permutation(608).astype(np.int32)[None]
            observation = SimpleNamespace(static_mask=static_mask, motion_mask=motion_mask[None], mem_order=mem_order,
                                          tokenized_prompt_mask=prompt_mask)
        else:
            observation = SimpleNamespace(static_mask=static_mask, motion_mask=None, mem_order=None,
                                          tokenized_prompt_mask=prompt_mask)
        actions = self._sample_actions(None, observation)
        return {"actions": actions, "state": np.zeros(8, np.float32), "infer_time_ms": 1.0}


class FakeAdapter:
    def __init__(self, P, MEM, hot_slot):
        self.P, self.MEM, self.hot_slot, self.last_attn = P, MEM, hot_slot, None

    def make_sample_fn(self):
        def sample(rng, observation):
            mem_order = (np.asarray(observation.mem_order)[0] if observation.mem_order is not None
                         else np.arange(self.MEM))
            pa = np.zeros((10, 18, 8, 20, self.P + 20), np.float16)
            pa[..., np.flatnonzero(mem_order // 16 == self.hot_slot)] = 0.04   # 让第 hot_slot 帧最高
            pa[..., np.flatnonzero(mem_order // 16 == 0)] = 0.02
            self.last_attn = {"probs_action": pa, "probs_text": np.zeros((18, 8, 64, self.P), np.float16),
                              "P": self.P, "MEM": self.MEM}
            return np.arange(160, dtype=np.float64).reshape(20, 8)
        return sample


EXPECTED = {  # §1.3：键 → (dtype, shape 生成器)
    "P": (np.int64, lambda P, MEM: ()), "MEM": (np.int64, lambda P, MEM: ()),
    "step_idx": (np.int64, lambda P, MEM: ()), "exec_start_idx": (np.int64, lambda P, MEM: ()),
    "infer_seq": (np.int64, lambda P, MEM: ()), "txt_len": (np.int64, lambda P, MEM: ()),
    "mem_order": (np.int32, lambda P, MEM: (MEM,)), "mem_mask": (np.bool_, lambda P, MEM: (MEM,)),
    "frame_times": (np.int32, lambda P, MEM: (32,)),
    "actions": (np.float32, lambda P, MEM: (20, 8)), "state": (np.float32, lambda P, MEM: (8,)),
    "top_frames": (np.int32, lambda P, MEM: (2,)),
    "probs_action": (np.float16, lambda P, MEM: (10, 18, 8, 20, P + 20)),
    "probs_text": (np.float16, lambda P, MEM: (18, 8, 64, P)),
}


@pytest.mark.parametrize("motion", [False, True])
def test_npz_contract(tmp_path, monkeypatch, motion):
    P, MEM = (1184, 608) if motion else (1088, 512)
    policy = FakePolicy(motion)
    adapter = FakeAdapter(P, MEM, hot_slot=3)
    recorder = SA.AttnRecorder(policy, adapter, tmp_path / "attn", save_top_frames=2)
    recorder.install()
    monkeypatch.setitem(SA._RPC, "op", "reset")
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": "nonce123"})
    policy.reset()
    assert policy.mem_buffer._prepare_motion.__name__ == "prepare_motion" if motion else True
    images = np.random.default_rng(1).integers(0, 255, (40, 2, 8, 8, 3), dtype=np.uint8)
    policy.add_buffer({"images": images, "state": np.zeros((40, 8), np.float32)})
    state = np.arange(8, dtype=np.float64)
    out = policy.infer({"observation/state": state, "prompt": "do it"})
    policy.infer({"observation/state": state, "prompt": "do it"})
    assert out["actions"].shape == (20, 8)
    ep = tmp_path / "attn" / "nonce123"
    assert sorted(p.name for p in ep.glob("infer_*.npz")) == ["infer_0000.npz", "infer_0001.npz"]
    with np.load(ep / "infer_0000.npz") as z:
        assert set(z.files) == set(SA.NPZ_KEYS)
        expected = dict(EXPECTED)
        expected["motion_times"] = (np.int32, lambda P, MEM: (96,) if motion else (0,))
        expected["motion_mask"] = (np.bool_, lambda P, MEM: (96,) if motion else (0,))
        assert set(expected) == set(SA.NPZ_KEYS)
        for key, (dtype, shape) in expected.items():
            assert z[key].dtype == dtype, key
            assert z[key].shape == shape(P, MEM), key
        assert int(z["P"]) == P and int(z["MEM"]) == MEM and int(z["infer_seq"]) == 0
        assert int(z["step_idx"]) == 39 and int(z["txt_len"]) == 21
        frames = list(range(0, 40, 4))
        assert z["frame_times"].tolist() == frames + [-1] * (32 - len(frames))
        static = np.repeat(np.arange(32) < len(frames), 16)
        if motion:
            mem_order = np.random.default_rng(0).permutation(608).astype(np.int32)
            assert np.array_equal(z["mem_order"], mem_order)
            full = np.concatenate([static, np.arange(96) < 3])
            assert np.array_equal(z["mem_mask"], full[mem_order])
            assert z["motion_times"].tolist()[:4] == [0, 16, 32, -1]
            assert z["motion_mask"].sum() == 3
        else:
            assert np.array_equal(z["mem_order"], np.arange(512))
            assert np.array_equal(z["mem_mask"], static)
        assert z["top_frames"].tolist() == [12, 0]       # 第 3 槽 → 帧 12 最高，其次第 0 槽 → 帧 0
        assert np.array_equal(z["state"], state.astype(np.float32))
        assert np.array_equal(z["actions"], np.arange(160, dtype=np.float32).reshape(20, 8))
    with np.load(ep / "infer_0001.npz") as z:
        assert int(z["infer_seq"]) == 1
    assert sorted(p.name for p in (ep / "frames").glob("*.jpg")) == ["0.jpg", "12.jpg"]
    info = json.loads((ep / "episode.json").read_text())
    assert info["episode_nonce"] == "nonce123" and info["infers"] == 2 and info["prompt"] == "do it"
    if motion:
        assert policy.mem_buffer.motion_calls == 2   # 只捕获返回值，没有额外调用 _prepare_motion
    # reset 后包装重新挂在新 mem_buffer 上
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": "nonce456"})
    policy.reset()
    if motion:
        assert policy.mem_buffer._prepare_motion.__name__ == "prepare_motion"
    assert (tmp_path / "attn" / "nonce456" / "episode.json").exists()


def test_npz_top_frames_off(tmp_path, monkeypatch):
    policy = FakePolicy(False)
    recorder = SA.AttnRecorder(policy, FakeAdapter(1088, 512, hot_slot=3), tmp_path / "attn", save_top_frames=0)
    recorder.install()
    monkeypatch.setitem(SA._RPC, "op", "reset")
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": "z"})
    policy.reset()
    policy.add_buffer({"images": np.zeros((5, 2, 4, 4, 3), np.uint8), "state": np.zeros((5, 8), np.float32)})
    policy.infer({"observation/state": np.zeros(8), "prompt": "p"})
    with np.load(tmp_path / "attn" / "z" / "infer_0000.npz") as z:
        assert z["top_frames"].shape == (0,) and z["top_frames"].dtype == np.int32
    assert not (tmp_path / "attn" / "z" / "frames").exists()


# ―――――――――――――――――――――――― save_prefix（S6）――――――――――――――――――――――――

@pytest.mark.parametrize("save_prefix", [None, False, True])
def test_server_command_save_prefix(save_prefix):
    spec = {"policy_python": "/p/python", "stage2_python": "/s/python", "run_identity_sha256": "a" * 64,
            "checkpoint": "/ck", "policy_config": "cfg", "model_kind": "history", "cuda": "1", "attn_out": "/attn"}
    if save_prefix is not None:
        spec["save_prefix"] = save_prefix
    command = RL.build_server_command(spec, "n" * 32, 9100)
    i = command.index("policy:checkpoint")
    assert command[i:] == ["policy:checkpoint", "--policy.config", "cfg", "--policy.dir", "/ck"]
    if save_prefix:
        assert command.count("--save-prefix") == 1
        assert command[i - 1] == "--save-prefix" and command[i - 3:i - 1] == ["--attn-out", "/attn"]
    else:
        assert "--save-prefix" not in command
        assert command[i - 2:i] == ["--attn-out", "/attn"]


class FakePrefixAdapter(FakeAdapter):
    def make_sample_fn(self):
        inner = super().make_sample_fn()

        def sample(rng, observation):
            out = inner(rng, observation)
            pp = np.zeros((18, 8, self.P, self.P), np.float16)
            pp[..., 0] = 1.0
            self.last_attn["probs_prefix"] = pp
            return out
        return sample


@pytest.mark.parametrize("motion", [False, True])
def test_prefix_npz_contract(tmp_path, monkeypatch, motion):
    P, MEM = (1184, 608) if motion else (1088, 512)
    policy = FakePolicy(motion)
    recorder = SA.AttnRecorder(policy, FakePrefixAdapter(P, MEM, hot_slot=3), tmp_path / "attn",
                               save_top_frames=0, save_prefix=True)
    recorder.install()
    monkeypatch.setitem(SA._RPC, "op", "reset")
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": "pfx"})
    policy.reset()
    policy.add_buffer({"images": np.zeros((40, 2, 4, 4, 3), np.uint8), "state": np.zeros((40, 8), np.float32)})
    for _ in range(2):
        policy.infer({"observation/state": np.zeros(8), "prompt": "p"})
    ep = tmp_path / "attn" / "pfx"
    assert sorted(p.name for p in ep.glob("infer_*.npz")) == ["infer_0000.npz", "infer_0001.npz"]
    assert sorted(p.name for p in ep.glob("prefix_*.npz")) == ["prefix_0000.npz", "prefix_0001.npz"]
    assert not list(ep.glob(".*tmp*"))
    with np.load(ep / "infer_0000.npz") as z:
        assert set(z.files) == set(SA.NPZ_KEYS)          # infer npz 16 键不变
    for seq in range(2):
        with np.load(ep / f"prefix_{seq:04d}.npz") as z:
            assert set(z.files) == set(SA.PREFIX_NPZ_KEYS)
            assert z["probs_prefix"].dtype == np.float16 and z["probs_prefix"].shape == (18, 8, P, P)
            for key in ("P", "MEM", "infer_seq", "step_idx"):
                assert z[key].dtype == np.int64 and z[key].shape == (), key
            assert int(z["P"]) == P and int(z["MEM"]) == MEM
            assert int(z["infer_seq"]) == seq and int(z["step_idx"]) == 39
            assert float(z["probs_prefix"][3, 2, 5, 0]) == 1.0


def test_prefix_off_writes_nothing(tmp_path, monkeypatch):
    policy = FakePolicy(False)
    recorder = SA.AttnRecorder(policy, FakePrefixAdapter(1088, 512, hot_slot=3), tmp_path / "attn", save_top_frames=0)
    recorder.install()
    monkeypatch.setitem(SA._RPC, "op", "reset")
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": "off"})
    policy.reset()
    policy.add_buffer({"images": np.zeros((5, 2, 4, 4, 3), np.uint8), "state": np.zeros((5, 8), np.float32)})
    policy.infer({"observation/state": np.zeros(8), "prompt": "p"})
    assert (tmp_path / "attn" / "off" / "infer_0000.npz").exists()
    assert not list((tmp_path / "attn" / "off").glob("prefix_*.npz"))
