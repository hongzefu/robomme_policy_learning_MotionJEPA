"""计划 1116 S-C：示范窗复用 N 录制（F2，--demo-cache-map / spec.demo_cache_map）与每 spec 常驻服务（F5，resident_server）。
不加载真模型：原 client 用记录调用的假编码器，策略用 SimpleNamespace 假体。"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.interpret import build_manifest as BM
from scripts.interpret import fp_swap_prep as F
from scripts.interpret import run_local as RL
from scripts.interpret import serve_policy_attn as SA

FRAMES = np.zeros((33, 1), np.uint8)
DIM = 4
ES = 40                       # 段边界：示范窗起点 0、16、32；执行窗 40、56
DEMO, EXEC = (0, 16, 32), (40, 56)


class Inner:
    """记下每次调用的起点；原编码 = 1000 + 起点。"""

    def __init__(self):
        self.calls = []

    def __call__(self, frames, start=None):
        self.calls.append(start)
        return np.full(DIM, 1000 + (start or 0), np.float32)


def cached(start, ep=3):
    """N 录制里的原编码（与在线不同值，便于区分来源）。"""
    return np.full(DIM, -(ep * 100 + start), np.float32)


def write_record(path, ep=3, starts=DEMO + EXEC):
    np.savez(path, start_frames=np.asarray(starts, np.int32), embs=np.stack([cached(s, ep) for s in starts]),
             counts=np.ones(len(starts), np.int32), mode=np.str_("none"))
    return str(path)


def cache_map(tmp_path, eps=(3,)):
    return {f"RouteStick:{ep}": write_record(tmp_path / f"RouteStick_ep{ep}.npz", ep) for ep in eps}


def client(tmp_path, mode="none", donor=None, per_episode_donor=False, record_dir=None, es=ES, eps=(3,)):
    inner = Inner()
    c = SA.MotionContentClient(inner, mode=mode, donor=donor, per_episode_donor=per_episode_donor,
                               record_dir=record_dir, demo_cache_map=cache_map(tmp_path, eps), exec_start_fn=lambda: es)
    return c, inner


def run(c, starts=DEMO + EXEC):
    return [c(FRAMES, s).copy() for s in starts]


# ―――――――――――――――――――――――― MotionContentClient ――――――――――――――――――――――――

def test_cache_hit_skips_inner_and_exec_still_online(tmp_path):
    c, inner = client(tmp_path)
    c.begin_episode("n1", episode_key="RouteStick:3")
    out = run(c)
    assert inner.calls == list(EXEC)                       # 示范窗一次也不调原 client
    for s, o in zip(DEMO, out[:3]):
        assert np.array_equal(o, cached(s))
    for s, o in zip(EXEC, out[3:]):
        assert np.array_equal(o, np.full(DIM, 1000 + s, np.float32))
    summary = c.summary()
    assert summary["n_cache_hits"] == 3 and summary["n_inner_calls"] == 2
    assert summary["cache_hit_starts"] == list(DEMO) and summary["n_windows"] == 5 and summary["n_replaced"] == 0


def test_donor_and_zero_take_precedence_over_cache(tmp_path):
    donor = {"name": "RouteStick:20000302", "start_frames": np.asarray([16, 40], np.int32),
             "embs": np.stack([np.full(DIM, 7.0, np.float32), np.full(DIM, 8.0, np.float32)])}
    c, inner = client(tmp_path, mode="swap_exact", per_episode_donor=True)
    c.begin_episode("n1", donor=donor, episode_key="RouteStick:3")
    out = run(c)
    assert inner.calls == list(EXEC)
    assert np.array_equal(out[0], cached(0)) and np.array_equal(out[2], cached(32))
    assert np.array_equal(out[1], np.full(DIM, 7.0)) and np.array_equal(out[3], np.full(DIM, 8.0))
    assert np.array_equal(out[4], np.full(DIM, 1056.0))
    assert c.replace_log == [[16, 16], [40, 40]]
    z, zi = client(tmp_path, mode="zero")
    z.begin_episode("n2", episode_key="RouteStick:3")
    assert all(not o.any() for o in run(z)) and zi.calls == list(EXEC)


def test_shift_uses_cached_original(tmp_path):
    c, inner = client(tmp_path, mode="shift")
    c.begin_episode("n1", episode_key="RouteStick:3")
    out = run(c)
    assert np.array_equal(out[1], cached(0)) and np.array_equal(out[3], cached(32))
    assert np.array_equal(out[4], np.full(DIM, 1040.0)) and inner.calls == list(EXEC)


def test_missing_keys_raise(tmp_path):
    c, _ = client(tmp_path)
    with pytest.raises(RuntimeError, match="没有局"):
        c.begin_episode("n1", episode_key="RouteStick:9")
    with pytest.raises(RuntimeError, match="局键"):
        c.begin_episode("n1")
    c.begin_episode("n1", episode_key="RouteStick:3")
    with pytest.raises(RuntimeError, match="不在缓存"):
        c(FRAMES, 8)
    with pytest.raises(RuntimeError, match="起点帧号"):
        c(FRAMES)
    none_es, _ = client(tmp_path, es=None)
    none_es.begin_episode("n1", episode_key="RouteStick:3")
    with pytest.raises(RuntimeError, match="exec_start_idx"):
        none_es(FRAMES, 0)
    with pytest.raises(ValueError, match="不存在"):
        SA.parse_demo_cache_map(json.dumps({"RouteStick:1": str(tmp_path / "nope.npz")}))


def test_no_cache_call_sequence_and_outputs_unchanged(tmp_path):
    """不设缓存：每个窗口都调原 client，返回值、summary 键与录制 npz 键与旧版相同。"""
    inner = Inner()
    c = SA.MotionContentClient(inner, record_dir=tmp_path)
    c.begin_episode("n1")
    out = run(c)
    assert inner.calls == list(DEMO + EXEC)
    assert all(np.array_equal(o, np.full(DIM, 1000 + s, np.float32)) for s, o in zip(DEMO + EXEC, out))
    assert set(c.summary()) == {"mode", "donor", "n_windows", "n_replaced", "replace_log", "record_file"}
    path = c.flush()
    with np.load(path) as z:
        assert set(z.files) == {"start_frames", "embs", "counts", "mode", "episode_nonce", "call_start_frames",
                                "call_embs", "call_seq"}


def test_record_marks_cache_windows(tmp_path):
    rec = tmp_path / "rec"
    c, _ = client(tmp_path, record_dir=rec)
    c.begin_episode("n1", episode_key="RouteStick:3")
    run(c)
    c(FRAMES, 16)                                          # 重复起点照旧计次
    with np.load(c.flush()) as z:
        assert z["cache_hit"].tolist() == [True, True, True, False, False]
        assert z["call_cache_hit"].tolist() == [True, True, True, False, False, True]
        assert str(z["demo_cache_file"]).endswith("RouteStick_ep3.npz")
        assert np.array_equal(z["embs"][0], cached(0))      # 记的仍是「原编码」（此处即 N 的录制值）


def test_episode_switch_resets_state(tmp_path):
    c, inner = client(tmp_path, eps=(3, 4))
    c.begin_episode("a", episode_key="RouteStick:3")
    first = run(c)
    c.begin_episode("b", episode_key="RouteStick:4")
    assert c.n_cache_hits == 0 and c.starts == [] and c.demo_cache_file.endswith("RouteStick_ep4.npz")
    second = run(c)
    assert np.array_equal(first[0], cached(0, 3)) and np.array_equal(second[0], cached(0, 4))
    assert inner.calls == list(EXEC) * 2
    # 载入失败的局不残留上一局缓存
    with pytest.raises(RuntimeError):
        c.begin_episode("c", episode_key="RouteStick:5")
    assert c.demo_cache is None


# ―――――――――――――――――――――――― 记录器（MotionOnlyRecorder）多局 ――――――――――――――――――――――――

def fake_modul_policy():
    config = SimpleNamespace(budget=4, token_per_image=1, num_views=1, motion=SimpleNamespace(budget=8))
    policy = SimpleNamespace(config=config, motion_enabled=True, metadata={"model_kind": "modul_v10_768"},
                             step_idx=0, exec_start_idx=0)
    policy._motion_client = Inner()

    def make_buffer():
        # 与 FrameSampMemory 同：首包 add_buffer 先写 exec_start_idx 再编 motion
        return SimpleNamespace(motion_enc=policy._motion_client, exec_start_idx=None,
                               _prepare_motion=lambda step: None)

    policy.mem_buffer = make_buffer()

    def reset():
        policy.mem_buffer = make_buffer()
        policy.step_idx = 0

    def add_buffer(obs):
        buf = policy.mem_buffer
        if buf.exec_start_idx is None:
            buf.exec_start_idx = ES
            for s in DEMO:
                buf.motion_enc(FRAMES, s)
        for s in EXEC:
            buf.motion_enc(FRAMES, s)
        policy.step_idx += len(obs["images"])

    policy.reset, policy.add_buffer = reset, add_buffer
    policy.infer = lambda obs: {"actions": np.zeros((2, 2), np.float32)}
    return policy


def test_recorder_multi_episode_cache(tmp_path, monkeypatch):
    policy = fake_modul_policy()
    inner = policy._motion_client
    recorder = SA.MotionOnlyRecorder(policy, tmp_path / "attn", demo_cache_map=cache_map(tmp_path, (3, 4)))
    recorder.install()
    for nonce, ep in (("e1", 3), ("e2", 4)):
        monkeypatch.setitem(SA._RPC, "op", "reset")
        monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": nonce, "episode_key": f"RouteStick:{ep}"})
        policy.reset()
        policy.add_buffer({"images": np.zeros((ES + 1, 1, 2, 2, 3), np.uint8)})
        policy.infer({"prompt": "p"})
        info = json.loads((tmp_path / "attn" / nonce / "episode.json").read_text())
        mc = info["motion_content"]
        assert mc["n_cache_hits"] == 3 and mc["n_inner_calls"] == 2
        assert mc["demo_cache_file"].endswith(f"RouteStick_ep{ep}.npz") and mc["mode"] == "none"
    assert inner.calls == list(EXEC) * 2
    # 缺 episode_key 的 reset 直接失败
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": "e3"})
    with pytest.raises(RuntimeError, match="episode_key"):
        policy.reset()


def test_recorder_without_cache_not_installed(tmp_path):
    policy = fake_modul_policy()
    inner = policy._motion_client
    SA.MotionOnlyRecorder(policy, tmp_path / "attn").install()
    assert policy._motion_client is inner                   # 不设任何参数时不包装原 client


def test_parse_demo_cache_map_inline_and_file(tmp_path):
    m = cache_map(tmp_path)
    assert SA.parse_demo_cache_map(json.dumps(m)) == m
    f = tmp_path / "map.json"
    f.write_text(json.dumps({"RouteStick:03": m["RouteStick:3"]}))
    assert SA.parse_demo_cache_map(str(f)) == m


# ―――――――――――――――――――――――― run_local ――――――――――――――――――――――――

def base_spec(tmp_path, **changes):
    meta = tmp_path / "meta"
    meta.mkdir(exist_ok=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": []}))
    values = dict(repo="/repo", policy_python="/policy/python", sim_python="/sim/python",
                  stage2_python="/stage2/python", checkpoint="/ckpt", policy_config="cfg",
                  model_kind="stage2_pool4_concat", cuda="0", attn_out="/out/attn", run_root="/out/run",
                  manifest=str(manifest), xla_mem_fraction=0.5, model_seed=42, run_identity_sha256="a" * 64,
                  profile="plfull", plfull_metadata_dir=str(meta))
    return values | changes


def load(tmp_path, spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return RL.load_spec(path)


def test_spec_demo_cache_rules(tmp_path):
    m = cache_map(tmp_path)
    spec = load(tmp_path, base_spec(tmp_path, demo_cache_map={"RouteStick:03": m["RouteStick:3"]}))
    assert spec["demo_cache_map"] == m
    with pytest.raises(SystemExit, match="plfull"):
        load(tmp_path, base_spec(tmp_path, demo_cache_map=m, profile="official"))
    with pytest.raises(SystemExit, match="只适用于"):
        load(tmp_path, base_spec(tmp_path, demo_cache_map=m, model_kind="history"))
    with pytest.raises(SystemExit, match="非空对象"):
        load(tmp_path, base_spec(tmp_path, demo_cache_map={}))
    with pytest.raises(SystemExit, match="true / false"):
        load(tmp_path, base_spec(tmp_path, resident_server="yes"))
    assert load(tmp_path, base_spec(tmp_path, resident_server=True))["resident_server"] is True


def test_server_command_demo_cache(tmp_path):
    m = cache_map(tmp_path, (3, 4))
    episodes = [{"episode_key": "RouteStick:3"}]
    plain = RL.build_server_command(base_spec(tmp_path), "n" * 32, 9100, episodes)
    with_cache = RL.build_server_command(base_spec(tmp_path, demo_cache_map=m), "n" * 32, 9100, episodes)
    i = with_cache.index("--demo-cache-map")
    assert json.loads(with_cache[i + 1]) == {"RouteStick:3": m["RouteStick:3"]}    # 只传本服务的局
    assert with_cache[:i] + with_cache[i + 2:] == plain                              # 其余命令不变
    assert plain == RL.build_server_command(base_spec(tmp_path, resident_server=True), "n" * 32, 9100, episodes)
    with pytest.raises(ValueError, match="没有局"):
        RL.build_server_command(base_spec(tmp_path, demo_cache_map=m), "n" * 32, 9100, [{"episode_key": "RouteStick:9"}])


def test_identity_demo_cache_and_resident(tmp_path):
    manifest = {"sha256": "b" * 64}
    spec = base_spec(tmp_path)
    before = RL.run_identity(spec, manifest)
    assert RL.run_identity(dict(spec, resident_server=True), manifest) == before
    assert RL.run_identity(dict(spec, demo_cache_map=cache_map(tmp_path)), manifest) != before


def test_resident_runs_all_chunks_on_one_server(tmp_path, monkeypatch):
    """resident_server：所有块共用一个服务，服务端局清单为全部局并集；客户端非 -1 失败不停后续批次。"""
    calls, servers = [], []

    class Proc:
        def __init__(self, command, server):
            self.command, self.server = command, server
            self.returncode = None if server else (3 if len(calls) == 2 else 0)

        def poll(self):
            return self.returncode

    def popen(command, **kwargs):
        p = Proc(command, command[0] == "server")
        calls.append(p)
        return p

    def server_command(spec, nonce, port, episodes=None):
        servers.append([e["episode_key"] for e in episodes or ()])
        return ["server", nonce]

    monkeypatch.setattr(RL, "pick_port", lambda value: 9001)
    monkeypatch.setattr(RL, "build_environment", lambda spec: {})
    monkeypatch.setattr(RL, "wait_ready", lambda *args: None)
    monkeypatch.setattr(RL, "build_server_command", server_command)
    monkeypatch.setattr(RL, "build_client_command", lambda spec, grant, output, nonce, port, drain: ["client", nonce])
    monkeypatch.setattr(RL.RW, "stop_process_tree", lambda process: None)
    chunks = [(f"c{i}", [{"episode_key": f"RouteStick:{2 * i}"}, {"episode_key": f"RouteStick:{2 * i + 1}"}])
              for i in range(3)]
    spec = {"run_root": str(tmp_path), "repo": "/repo", "run_identity_sha256": "a" * 64, "resident_server": True}
    results = list(RL.run_oracle_chunks(spec, chunks, popen=popen))
    assert sum(p.server for p in calls) == 1 and len(servers) == 1
    assert servers[0] == [f"RouteStick:{i}" for i in range(6)]
    assert [r[2] for r in results] == [0, 3, 0]             # 第 2 批客户端失败（非 -1）后仍跑第 3 批
    assert len(list((Path(tmp_path) / "attempts").glob("resident-service-*"))) == 1


def _fake_exe(path, body):
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)
    return str(path)


def test_resident_end_to_end_fake_processes(tmp_path, monkeypatch, capsys):
    from tests.interpret.test_run_local import FAKE_CLIENT, FAKE_SERVER
    meta = BM.default_metadata_root()
    if not any(meta.glob("record_dataset_*_metadata.json")):
        pytest.skip("官方元数据不可读")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setenv("FAKE_LOG_DIR", str(log_dir))
    monkeypatch.setenv("FAKE_ATTN_OUT", str(tmp_path / "attn"))
    pairs = [("RouteStick", 1), ("RouteStick", 2), ("RouteStick", 3)]
    manifest = BM.build(pairs, meta, 1)
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    m = cache_map(tmp_path, (1, 2, 3))
    spec = base_spec(tmp_path, policy_python=_fake_exe(tmp_path / "fp", FAKE_SERVER),
                     sim_python=_fake_exe(tmp_path / "fs", FAKE_CLIENT), repo=str(Path(RL.REPO)),
                     attn_out=str(tmp_path / "attn"), run_root=str(tmp_path / "run"), startup_timeout_seconds=30,
                     demo_cache_map=m, resident_server=True, cuda="1", manifest=str(tmp_path / "m.json"))
    spec.pop("run_identity_sha256")
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    assert RL.main(["--spec", str(path)]) == 0, capsys.readouterr().out
    started = [json.loads(p.read_text()) for p in log_dir.glob("server-*.json")]
    assert len(started) == 1                                 # 3 块只起 1 个服务
    argv = started[0]["argv"]
    assert json.loads(argv[argv.index("--demo-cache-map") + 1]) == m
    assert len(list(log_dir.glob("client-*.json"))) == 3


# ―――――――――――――――――――――――― fp_swap_prep ――――――――――――――――――――――――

def test_fp_builder_speedup_only_on_cache_stages(tmp_path):
    b = F.Builder.__new__(F.Builder)
    b.demo_cache_from, b.resident_server = None, False
    assert F.apply_speedup(b, SimpleNamespace(demo_cache_from=None, resident_server=False)) is b
    assert b.demo_cache_from is None and b.resident_server is False   # 缺省不改
    F.apply_speedup(b, SimpleNamespace(demo_cache_from=str(tmp_path / "rec"), resident_server=True))
    assert b.demo_cache_from == (tmp_path / "rec").resolve() and b.resident_server
    assert F.apply_speedup(b, None) is b
    assert set(F.CACHE_STAGES) == {"samep", "noise"}
