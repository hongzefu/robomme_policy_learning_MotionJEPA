"""plfull 多局交换（服务端按 reset 带来的 episode_key 逐局取供体）与 Wan 时间池化模型接入 run_local 的 CPU 测试。

不起服务端、不起仿真、不依赖 GPU；W3 的 pl_full_prep 生成的 spec 用 dry 方式交给 run_local 校验与命令构造。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation import manifest as M
from scripts.interpret import run_local as local
from scripts.interpret import serve_policy_attn as SA
from scripts.interpret.test_motion_swap_modes import FRAMES, Inner, attn_recorder, drive, original

TASK = "PatternLock"


def key(ep):
    return f"{TASK}:{ep}"


def donor_emb(ep, start):
    return np.full(6, -(ep * 1000 + start + 1), np.float32)


def write_pl_donor(directory, ep, starts=(0,)):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / f"{TASK}_ep{ep}.npz", start_frames=np.asarray(starts, np.int32),
             embs=np.stack([donor_emb(ep, s) for s in starts]), counts=np.ones(len(starts), np.int32))


# ---------------------------------------------------------------- 服务端：解析与逐局供体

def test_parse_swap_map_arg():
    assert SA.parse_swap_map_arg("PatternLock:01=PatternLock:41, PatternLock:2=PatternLock:2") == {
        key(1): key(41), key(2): key(2)}
    for bad in ("", "PatternLock:1", "PatternLock:1=PatternLock:2,PatternLock:1=PatternLock:3", "x=y"):
        with pytest.raises(ValueError):
            SA.parse_swap_map_arg(bad)


def test_client_per_episode_donor():
    client = SA.MotionContentClient(Inner(), mode="swap_exact", per_episode_donor=True)
    with pytest.raises(RuntimeError, match="尚未确定供体"):
        client(FRAMES, 0)
    with pytest.raises(RuntimeError, match="必须给出供体"):
        client.begin_episode("n0")
    for ep in (41, 52):
        donor = {"name": key(ep), "start_frames": np.asarray([0], np.int32), "embs": donor_emb(ep, 0)[None]}
        client.begin_episode(f"n{ep}", donor=donor)
        np.testing.assert_array_equal(client(FRAMES, 0), donor_emb(ep, 0))
        np.testing.assert_array_equal(client(FRAMES, 33), original(33))   # exact：供体无此起点保留原值
        assert client.summary()["donor"] == key(ep) and client.replaced == 1
    with pytest.raises(ValueError):
        SA.MotionContentClient(Inner(), mode="shift", per_episode_donor=True)
    fixed = SA.MotionContentClient(Inner(), mode="none")
    with pytest.raises(ValueError, match="固定供体"):
        fixed.begin_episode("n", donor={"name": "x"})


def multi_recorder(tmp_path, mapping, record=None):
    """在 test_motion_swap_modes 的假策略上换成 --swap-map 构造（不带 --swap-donor）。"""
    donors = tmp_path / "donors"
    for ep in set(int(v.rpartition(":")[2]) for v in mapping.values()):
        write_pl_donor(donors, ep, starts=(0, 16))
    recorder, policy, out = attn_recorder(tmp_path, "none")   # 先拿一份装好的假策略与 adapter
    rec = SA.AttnRecorder(recorder.policy, recorder.adapter, tmp_path / "attn2", ablate=f"motion_swap_exact:{donors}",
                          swap_map=mapping, record_motion=record)
    # attn_recorder 已经 install 过一次：恢复原方法后重新安装
    policy.reset, policy.add_buffer, policy.infer = recorder._orig_reset, recorder._orig_add, recorder._orig_infer
    rec._orig_reset, rec._orig_add, rec._orig_infer = policy.reset, policy.add_buffer, policy.infer
    rec.install()
    return rec, policy, tmp_path / "attn2"


def drive_episode(policy, out, ep, nonce):
    SA._RPC.update(op="reset", rpc={"episode_nonce": nonce, "episode_key": key(ep)})
    policy.reset()
    policy.add_buffer({"images": np.zeros((1, 1, 2, 2, 3), np.uint8), "state": np.zeros((1, 8), np.float32)})
    policy.infer({"observation/state": np.zeros(8, np.float32), "prompt": "p"})
    return json.loads((out / nonce / "episode.json").read_text(encoding="utf-8"))


def test_multi_episode_server_picks_donor_per_episode(tmp_path):
    mapping = {key(0): key(41), key(1): key(40), key(2): key(2)}
    rec, policy, out = multi_recorder(tmp_path, mapping, record=tmp_path / "rec")
    infos = [drive_episode(policy, out, ep, f"nonce{ep}") for ep in (0, 1, 2)]
    for ep, info in zip((0, 1, 2), infos):
        assert info["episode_key"] == key(ep) and info["swap_donor"] == mapping[key(ep)]
        content = info["motion_content"]
        assert content["donor"] == mapping[key(ep)] and content["n_replaced"] == 2
        assert content["replace_log"] == [[0, 0], [16, 16]]
    # 录制：多局块每局各自一份 <nonce>.npz，记录的是原编码，供体名逐局不同
    rec.motion_client.flush()
    for ep in (0, 1, 2):
        with np.load(tmp_path / "rec" / f"nonce{ep}.npz") as z:
            assert str(z["donor"]) == mapping[key(ep)]
            np.testing.assert_array_equal(z["embs"][0], original(0))
    assert sorted(rec._donor_cache) == sorted(set(mapping.values()))


def test_multi_episode_server_rejects_missing_key(tmp_path):
    rec, policy, out = multi_recorder(tmp_path, {key(0): key(41)})
    SA._RPC.update(op="reset", rpc={"episode_nonce": "n-a"})
    with pytest.raises(RuntimeError, match="episode_key"):
        policy.reset()
    SA._RPC.update(op="reset", rpc={"episode_nonce": "n-b", "episode_key": key(5)})
    with pytest.raises(RuntimeError, match="没有局"):
        policy.reset()
    assert not (out / "n-a").exists() and not (out / "n-b").exists()


def test_swap_map_and_swap_donor_exclusive(tmp_path):
    from scripts.interpret.test_motion_swap_modes import fake_policy
    write_pl_donor(tmp_path, 41)
    with pytest.raises(SystemExit, match="互斥"):
        SA.AttnRecorder(fake_policy(), None, tmp_path, ablate=f"motion_swap_exact:{tmp_path}",
                        swap_donor=key(41), swap_map={key(0): key(41)})
    with pytest.raises(SystemExit, match="同时给出"):
        SA.AttnRecorder(fake_policy(), None, tmp_path, ablate="none", swap_map={key(0): key(41)})
    rec = SA.AttnRecorder(fake_policy(), None, tmp_path, ablate=f"motion_swap_exact:{tmp_path}", swap_map={key(0): key(41)})
    assert rec.donor is None and rec.swap_map == {key(0): key(41)}


def test_old_single_donor_path_unchanged(tmp_path):
    """--swap-donor 路径：reset 不带 episode_key 也照常，供体固定。"""
    recorder, policy, out = attn_recorder(tmp_path, "motion_swap_exact:{d}")
    assert recorder.swap_map is None and recorder.donor["name"] == "ButtonUnmaskSwap:8"
    SA._RPC.update(op=None, rpc=None)
    _, directory = drive(policy, out)
    info = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    assert "swap_donor" not in info and info["motion_content"]["donor"] == "ButtonUnmaskSwap:8"


def test_server_args_has_swap_map():
    pytest.importorskip("tyro")
    try:
        Args = SA._make_args_class()
    except Exception as exc:  # serve_policy 依赖 jax 等；缺环境时只核字段名
        pytest.skip(f"serve_policy 不可导入: {exc}")
    names = {f.name for f in __import__("dataclasses").fields(Args)}
    assert {"swap_map", "swap_donor"} <= names


# ---------------------------------------------------------------- run_local：plfull 与 Wan

def pl_manifest(tmp_path, eps, chunk=5):
    rows = [{"task_name": TASK, "episode_id": ep, "env_seed": 900000 + 100 * ep, "difficulty": "easy"} for ep in eps]
    manifest = M.make_manifest(rows, metadata_sha256={"m": "0" * 64}, chunk_size=chunk, official=False)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def pl_spec(tmp_path, eps=(0, 1, 41, 60, 79, 3), **changes):
    meta = tmp_path / "metadata"
    meta.mkdir(exist_ok=True)
    (meta / f"record_dataset_{TASK}_metadata.json").write_text("[]")
    path, _ = pl_manifest(tmp_path, eps)
    values = dict(repo="/repo", policy_python="/policy/python", sim_python="/sim/python",
                  stage2_python="/stage2/python", checkpoint="/ckpt", policy_config="cfg",
                  model_kind="stage2_pool4_concat", cuda="0", attn_out=str(tmp_path / "attn"),
                  run_root=str(tmp_path / "run"), manifest=str(path), xla_mem_fraction=0.5, model_seed=42,
                  profile="plfull", plfull_metadata_dir=str(meta), subgoal_trace=True, save_attn=False,
                  target_contract="/contracts/runtime-mot.json", run_identity_sha256="a" * 64)
    return values | changes


def load(tmp_path, spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return local.load_spec(path)


def test_constants_match_sources():
    from examples.robomme import eval as official
    from mme_vla_suite.shared.wan_timepool import TIMEPOOL_ENCODER_KIND
    assert local.WAN_MODEL_KIND == TIMEPOOL_ENCODER_KIND and local.WAN_MODEL_KIND in local.MODEL_KINDS
    assert local.PLFULL_SEEDS == official.PLFULL_SEEDS


def test_plfull_spec_rules(tmp_path):
    spec = load(tmp_path, pl_spec(tmp_path, model_seed=1))
    assert spec["profile"] == "plfull" and spec["model_seed"] == 1
    for bad in ({"model_seed": 7}, {"plfull_metadata_dir": str(tmp_path / "none")}, {"profile": "plx"}):
        with pytest.raises(SystemExit):
            load(tmp_path, pl_spec(tmp_path, **bad))
    with pytest.raises(SystemExit, match="plfull_metadata_dir"):
        load(tmp_path, pl_spec(tmp_path, profile="official"))


def test_plfull_chunks_not_split_and_server_gets_whole_map(tmp_path):
    donors = "motion_swap_exact:/O/motion/donor_trim"
    mapping = {key(0): key(40), key(1): key(41), key(41): key(1), key(60): key(60), key(79): key(79), key(3): key(43)}
    spec = load(tmp_path, pl_spec(tmp_path, ablate=donors, swap_map=mapping))
    manifest = M.validate_episodes(json.loads(Path(spec["manifest"]).read_text()), official=False)
    chunks = local.select_chunks(manifest, single=bool(spec.get("swap_map")) and not local.is_plfull(spec))
    assert [len(eps) for _, eps in chunks] == [5, 1]
    base = local.build_server_command(pl_spec(tmp_path), "nonce", 9000)
    index = base.index("policy:checkpoint")
    command = local.build_server_command(spec, "nonce", 9000, episodes=chunks[0][1])
    whole = ",".join(f"{k}={v}" for k, v in sorted(mapping.items()))
    assert command == base[:index] + ["--ablate", donors, "--swap-map", whole] + base[index:]
    assert "--swap-donor" not in command and SA.parse_swap_map_arg(whole) == mapping
    with pytest.raises(ValueError, match="没有局"):
        local.build_server_command(spec, "nonce", 9000, episodes=[{"episode_key": key(9)}])


def test_plfull_client_command(tmp_path):
    spec = load(tmp_path, pl_spec(tmp_path))
    grant = tmp_path / "grant.json"
    grant.write_text(json.dumps({"chunk_id": "c"}))
    command = local.build_client_command(spec, grant, tmp_path / "out", "nonce", 9000, tmp_path / "drain")
    assert command[1].endswith("scripts/interpret/subgoal_trace_runner.py")
    i = command.index("--profile")
    assert command[i:i + 4] == ["--profile", "plfull", "--plfull-metadata-dir", spec["plfull_metadata_dir"]]


def test_plfull_identity_and_official_unchanged(tmp_path):
    manifest = {"sha256": "b" * 64}
    plain = pl_spec(tmp_path)
    official = {k: v for k, v in plain.items() if k not in ("profile", "plfull_metadata_dir")}
    core = {"manifest_seal": "b" * 64, "checkpoint": "/ckpt", "policy_config": "cfg",
            "model_kind": "stage2_pool4_concat", "model_seed": 42, "subgoal_trace": True, "save_attn": False}
    assert local.run_identity(official, manifest) == M.digest(core)
    plfull_id = local.run_identity(plain, manifest)
    assert plfull_id != M.digest(core)
    meta = Path(plain["plfull_metadata_dir"]) / f"record_dataset_{TASK}_metadata.json"
    meta.write_text("[1]")
    assert local.run_identity(plain, manifest) != plfull_id


def test_official_swap_map_still_single(tmp_path):
    spec = pl_spec(tmp_path, ablate="motion_swap_exact:/d", swap_map={key(0): key(40)})
    del spec["profile"], spec["plfull_metadata_dir"]
    with pytest.raises(ValueError, match="恰 1 局"):
        local.build_server_command(spec, "nonce", 9000, episodes=[{"episode_key": key(0)}, {"episode_key": key(1)}])
    command = local.build_server_command(spec, "nonce", 9000, episodes=[{"episode_key": key(0)}])
    assert command[command.index("--swap-donor") + 1] == key(40) and "--swap-map" not in command


WAN_EXTRA = dict(model_kind="wan_fixedpool1024", policy_config="mme_vla_suite_orig16_b128_60k",
                 checkpoint="/store/assets/vla-o16wan/x/59999", stage2_python="/store/envs/stage2/bin/python")


@pytest.mark.parametrize("contract", [None, "", "none"])
def test_wan_spec_server_command(tmp_path, contract):
    changes = dict(WAN_EXTRA, ablate="motion_swap_exact:/d", swap_map={key(0): key(40)}, v1_store="/store")
    if contract is None:
        spec = pl_spec(tmp_path, **changes)
        del spec["target_contract"]
    else:
        spec = pl_spec(tmp_path, target_contract=contract, **changes)
    spec = load(tmp_path, spec)
    assert "target_contract" not in spec
    command = local.build_server_command(spec, "nonce", 9000, episodes=[{"episode_key": key(0)}])
    assert command[command.index("--motion-online-gpu") + 1] == "0"
    assert command[command.index("--stage2-python") + 1] == "/store/envs/stage2/bin/python"
    assert "--eval-target-contract" not in command and command[2] == local.ATTN_MODULE
    assert command[-4:] == ["--policy.config", "mme_vla_suite_orig16_b128_60k", "--policy.dir",
                            "/store/assets/vla-o16wan/x/59999"]
    environment = local.build_environment(spec)
    assert environment["MMEVLA_V1_STORE"] == "/store"


def test_wan_rejects_contract_and_needs_v1_store(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="target_contract"):
        load(tmp_path, pl_spec(tmp_path, **WAN_EXTRA))
    spec = load(tmp_path, pl_spec(tmp_path, target_contract="none", **WAN_EXTRA))
    monkeypatch.delenv("MMEVLA_V1_STORE", raising=False)
    with pytest.raises(ValueError, match="MMEVLA_V1_STORE"):
        local.build_environment(spec)
    monkeypatch.setenv("MMEVLA_V1_STORE", "/env/store")
    assert local.build_environment(spec)["MMEVLA_V1_STORE"] == "/env/store"


def test_history_still_drops_online_gpu(tmp_path):
    spec = pl_spec(tmp_path, model_kind="history")
    assert "--motion-online-gpu" not in local.build_server_command(spec, "nonce", 9000)
    assert "--motion-online-gpu" in local.build_server_command(pl_spec(tmp_path), "nonce", 9000)


# ---------------------------------------------------------------- W3 生成的 spec dry 通过 run_local

def fake_replay(seed):
    from scripts.interpret import pl_full_prep as P
    combos = P.all_combos()
    idx = (seed // 100) % 50
    if idx < 40:
        start, d = combos[idx]
        return [start, P.step_cell(start, d)]
    return [0, 4, 8]


@pytest.mark.parametrize("model, args", [
    ("motion", []),
    ("wan", ["--checkpoint", "/store/assets/vla-o16wan/x/59999", "--policy-config", "mme_vla_suite_orig16_b128_60k",
             "--model-kind", "wan_fixedpool1024", "--target-contract", "none",
             "--extra", json.dumps({"v1_store": "/store"})]),
])
def test_w3_specs_dry_through_run_local(tmp_path, monkeypatch, model, args):
    from scripts.interpret import pl_full_prep as P
    monkeypatch.setattr(P, "replay_path", fake_replay)
    root = tmp_path / "O"
    assert P.main(["pool", "--out", str(root)]) == 0
    out = root / model
    assert P.main(["specs", "--out", str(out), "--model", model, "--commit", "a" * 40, "--run-tree", "/tmp/tree", *args]) == 0
    checked = 0
    for path in sorted((out / "specs").glob("*.json")):
        spec = local.load_spec(path)
        manifest = M.validate_episodes(json.loads(Path(spec["manifest"]).read_text()), official=False)
        spec["run_identity_sha256"] = local.run_identity(spec, manifest)
        chunks = local.select_chunks(manifest, single=bool(spec.get("swap_map")) and not local.is_plfull(spec))
        for _, episodes in chunks:
            server = local.build_server_command(spec, "nonce", 9000, episodes)
            if spec.get("swap_map") and local.is_plfull(spec):
                assert "--swap-map" in server
            if model == "wan":
                assert "--motion-online-gpu" in server and "--eval-target-contract" not in server
                assert local.build_environment(spec)["MMEVLA_V1_STORE"] == "/store"
            grant = tmp_path / "grant.json"
            grant.write_text(json.dumps({"chunk_id": "c"}))
            client = local.build_client_command(spec, grant, tmp_path / "o", "nonce", 9000, tmp_path / "d")
            assert ("--profile" in client) == local.is_plfull(spec)
        checked += 1
    assert checked >= 20
