"""计划 1115 §4.1（S1）：modulation 模型的「只换 motion、不装 AttnAdapter」服务模式、run_local 白名单与环境透传、
eval.py plfull 的 max_steps。全部用假策略 / 假 client / monkeypatch，不加载权重、不起 GPU、不起 sidecar。"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.interpret import run_local as local
from scripts.interpret import serve_policy_attn as SA

FRAMES = np.zeros((33, 1), np.uint8)
DIM = 768
ACTIONS = np.arange(SA.ACTION_HORIZON * SA.ACTION_DIM, dtype=np.float32).reshape(SA.ACTION_HORIZON, SA.ACTION_DIM)


def original(start):
    return np.full(DIM, 1000 + start, np.float32)


class Inner:
    def __init__(self, dim=DIM):
        self.dim, self.calls = dim, []

    def __call__(self, frames, start=None):
        self.calls.append(start)
        return np.full(self.dim, 1000 + (start or 0), np.float32)


def write_donor(directory, name="RouteStick_ep3.npz", starts=(0, 16), dim=DIM):
    np.savez(directory / name, start_frames=np.asarray(starts, np.int32),
             embs=np.stack([np.full(dim, -s - 1, np.float32) for s in starts]), counts=np.ones(len(starts), np.int32),
             mode=np.str_("none"))


def fake_modul_policy(model_kind="modul_v10_768", integration="modulation"):
    """假 modulation 策略：infer 走「生产」路径（直接调 mem_buffer 的 motion client，再调 _sample_actions）。"""
    config = SimpleNamespace(budget=4, token_per_image=1, num_views=1, motion=SimpleNamespace(budget=3),
                             integration_type=integration)
    policy = SimpleNamespace(config=config, motion_enabled=True, metadata={"model_kind": model_kind},
                             step_idx=0, exec_start_idx=0)
    policy._motion_client = Inner()
    sampled = []

    def production_sample(rng, observation, **kwargs):
        sampled.append(kwargs)
        return ACTIONS.copy()

    policy._sample_actions = production_sample
    policy.sampled = sampled

    def make_buffer():
        return SimpleNamespace(motion_enc=policy._motion_client,
                               _prepare_motion=lambda step: (None, None, None, np.asarray([0, 16, -1], np.int32)))

    policy.mem_buffer = make_buffer()

    def reset():
        policy.mem_buffer = make_buffer()
        policy.step_idx = 0

    def add_buffer(obs):
        policy.step_idx += len(obs["images"])

    def infer(obs):
        policy.mem_buffer._prepare_motion(policy.step_idx)
        outs = [policy.mem_buffer.motion_enc(FRAMES, start) for start in (0, 16)]
        return {"actions": policy._sample_actions(None, SimpleNamespace()), "motion": outs}

    policy.reset, policy.add_buffer, policy.infer = reset, add_buffer, infer
    return policy


def drive(policy, out, infers=2):
    policy.reset()
    results = []
    for _ in range(infers):
        policy.add_buffer({"images": np.zeros((1, 1, 2, 2, 3), np.uint8), "state": np.zeros((1, 8), np.float32)})
        results.append(policy.infer({"prompt": "p"}))
    directory, = [path for path in Path(out).iterdir() if path.is_dir()]
    return results, directory


# ―― 模式选择 ――

@pytest.mark.parametrize("kind", ["modul_v10_768", "wan_fixedpool768"])
def test_modulation_kinds_select_motion_only(kind):
    assert SA.motion_only_mode(fake_modul_policy(kind, integration=None))


def test_integration_type_modulation_selects_motion_only():
    assert SA.motion_only_mode(fake_modul_policy("history", integration="modulation"))


@pytest.mark.parametrize("kind", ["history", "stage2_pool4_concat", "wan_fixedpool1024"])
def test_context_kinds_keep_attn_mode(kind):
    assert not SA.motion_only_mode(fake_modul_policy(kind, integration="context"))


def test_motion_only_kinds_agree_with_policy_config_and_run_local():
    from mme_vla_suite.policies.policy_config import _motion_model_kind
    assert _motion_model_kind({"encoder_kind": None}, "modulation") == "modul_v10_768"
    assert _motion_model_kind({"encoder_kind": "wan_fixedpool768"}, "modulation") == "wan_fixedpool768"
    assert set(SA.MOTION_ONLY_KINDS) == set(local.MOTION_ONLY_KINDS) == {"modul_v10_768", "wan_fixedpool768"}


# ―― 构造服务：不装 AttnAdapter ――

class _ServePolicyStub:
    """替身 scripts.training.serve_policy：main 只调用（可能被换过的）create_policy 一次。"""

    def __init__(self, policy):
        self.policy, self.created = policy, []

    def create_policy(self, policy_args):
        return self.policy

    def main(self, args):
        self.created.append(self.create_policy(SimpleNamespace(policy=SimpleNamespace(dir="/ckpt"))))


def serve_args(tmp_path, **changes):
    values = dict(attn_out=tmp_path / "attn", save_top_frames=3, save_prefix=False, no_save_attn=False, ablate="none",
                  record_motion=None, swap_donor=None, swap_map=None, query_record=0, query_batch_size=None,
                  oracle_mode="", rollout_record=False, seed=42)
    return SimpleNamespace(**(values | changes))


@pytest.fixture
def serve_stub(monkeypatch):
    import scripts.interpret.attn_adapter as attn_adapter
    import scripts.training as training_pkg

    def make(policy):
        stub = _ServePolicyStub(policy)
        monkeypatch.setitem(sys.modules, "scripts.training.serve_policy", stub)
        monkeypatch.setattr(training_pkg, "serve_policy", stub, raising=False)
        monkeypatch.setattr(SA, "_install_rpc_tracking", lambda **kw: None)
        return stub

    def boom(*args, **kwargs):
        raise AssertionError("只换 motion 模式不得构造 AttnAdapter")

    monkeypatch.setattr(attn_adapter, "AttnAdapter", boom)
    return make


@pytest.mark.parametrize("kind", ["modul_v10_768", "wan_fixedpool768"])
def test_main_motion_only_skips_attn_adapter(tmp_path, serve_stub, kind, capsys):
    policy = fake_modul_policy(kind)
    production = policy._sample_actions
    stub = serve_stub(policy)
    SA.main(serve_args(tmp_path, record_motion=tmp_path / "rec"))
    assert stub.created == [policy]
    assert policy._sample_actions is production          # 采样仍是生产函数
    assert isinstance(policy._motion_client, SA.MotionContentClient)
    assert "MOTION_ONLY_SERVE" in capsys.readouterr().out
    results, directory = drive(policy, tmp_path / "attn")
    np.testing.assert_array_equal(results[0]["actions"], ACTIONS)
    info = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    assert info["serve_mode"] == "motion_only" and info["attn_saved"] is False and info["infers"] == 2
    assert info["model_kind"] == kind
    assert [p.name for p in directory.iterdir()] == ["episode.json"]
    rec = np.load(tmp_path / "rec" / f"{info['episode_nonce']}.npz")
    assert rec["embs"].shape == (2, DIM) and rec["start_frames"].tolist() == [0, 16]


def test_main_context_still_uses_attn_adapter(tmp_path, serve_stub):
    policy = fake_modul_policy("stage2_pool4_concat", integration="context")
    serve_stub(policy)
    with pytest.raises(AssertionError, match="不得构造 AttnAdapter"):
        SA.main(serve_args(tmp_path))


@pytest.mark.parametrize("ablate, message", [
    ("mask_motion", "key 遮挡"), ("mask_vis", "key 遮挡"), ("heads:L3H0", "heads"),
])
def test_main_motion_only_rejects_unsupported(tmp_path, serve_stub, ablate, message):
    serve_stub(fake_modul_policy())
    with pytest.raises(SystemExit, match=message):
        SA.main(serve_args(tmp_path, ablate=ablate))


def test_motion_only_rejects_frame_swap_and_save_prefix(tmp_path):
    with pytest.raises(SystemExit, match="frame_swap_exact"):
        SA.MotionOnlyRecorder(fake_modul_policy(), tmp_path, ablate="frame_swap_exact:/d", swap_map={"A:1": "A:2"})
    with pytest.raises(SystemExit, match="save-prefix"):
        SA.MotionOnlyRecorder(fake_modul_policy(), tmp_path, save_prefix=True)
    with pytest.raises(SystemExit, match="save_attn"):
        SA.MotionOnlyRecorder(fake_modul_policy(), tmp_path, save_attn=True)


# ―― 只换 motion：替换、录制、逐局供体 ――

def test_noop_eq_self_donor_gives_identical_motion(tmp_path):
    """NOOP_EQ 的 CPU 版：以自身录制为供体的 swap_exact 与不干预逐位相同，且每窗都计入替换。"""
    rec_dir = tmp_path / "rec"
    policy = fake_modul_policy()
    recorder = SA.MotionOnlyRecorder(policy, tmp_path / "a", record_motion=rec_dir)
    recorder.install()
    base, directory = drive(policy, tmp_path / "a")
    nonce = json.loads((directory / "episode.json").read_text())["episode_nonce"]
    (rec_dir / f"{nonce}.npz").rename(rec_dir / "RouteStick_ep3.npz")

    policy2 = fake_modul_policy()
    recorder2 = SA.MotionOnlyRecorder(policy2, tmp_path / "b", ablate=f"motion_swap_exact:{rec_dir}",
                                      swap_donor="RouteStick:3")
    recorder2.install()
    swapped, directory2 = drive(policy2, tmp_path / "b")
    for x, y in zip(base, swapped):
        for a, b in zip(x["motion"], y["motion"]):
            np.testing.assert_array_equal(a, b)
    content = json.loads((directory2 / "episode.json").read_text())["motion_content"]
    assert content["mode"] == "swap_exact" and content["n_replaced"] == 4


def test_motion_only_per_episode_donor(tmp_path, monkeypatch):
    write_donor(tmp_path)
    policy = fake_modul_policy("wan_fixedpool768")
    recorder = SA.MotionOnlyRecorder(policy, tmp_path / "out", ablate=f"motion_swap:{tmp_path}",
                                     swap_map={"RouteStick:5": "RouteStick:3"})
    recorder.install()
    monkeypatch.setitem(SA._RPC, "op", "reset")
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": "n1", "episode_key": "RouteStick:5"})
    results, directory = drive(policy, tmp_path / "out", infers=1)
    np.testing.assert_array_equal(results[0]["motion"][1], np.full(DIM, -17, np.float32))
    info = json.loads((directory / "episode.json").read_text())
    assert info["swap_donor"] == "RouteStick:3" and info["episode_key"] == "RouteStick:5"


def test_donor_dim_mismatch_raises_even_without_hit(tmp_path):
    """768 维核对：swap_exact 本窗未命中也不放过维数不符的供体。"""
    donor = {"name": "x", "start_frames": np.asarray([999], np.int32), "embs": np.zeros((1, 1024), np.float32)}
    client = SA.MotionContentClient(Inner(), mode="swap_exact", donor=donor)
    with pytest.raises(RuntimeError, match="形状"):
        client(FRAMES, 0)
    ok = {"name": "y", "start_frames": np.asarray([999], np.int32), "embs": np.zeros((1, DIM), np.float32)}
    np.testing.assert_array_equal(SA.MotionContentClient(Inner(), mode="swap_exact", donor=ok)(FRAMES, 0), original(0))


def test_motion_only_shift_and_zero(tmp_path):
    for ablate in ("motion_shift", "motion_zero"):
        policy = fake_modul_policy()
        SA.MotionOnlyRecorder(policy, tmp_path / ablate, ablate=ablate).install()
        results, directory = drive(policy, tmp_path / ablate, infers=1)
        assert json.loads((directory / "episode.json").read_text())["motion_content"]["mode"] == ablate[len("motion_"):]


# ―― run_local ――

def base_spec(tmp_path, **changes):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": []}))
    values = dict(repo="/repo", policy_python="/policy/python", sim_python="/sim/python",
                  checkpoint="/ckpt", policy_config="mme_vla_suite_b128_80k", model_kind="modul_v10_768", cuda="1",
                  attn_out="/out/attn", run_root="/out/run", manifest=str(manifest), xla_mem_fraction=0.4,
                  model_seed=42, run_identity_sha256="a" * 64, v1_store="/turbo/v1-store")
    return {k: v for k, v in (values | changes).items() if v is not None}


def load(tmp_path, spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return local.load_spec(path)


@pytest.mark.parametrize("kind", ["modul_v10_768", "wan_fixedpool768"])
def test_run_local_accepts_new_kinds_without_stage2(tmp_path, kind):
    spec = load(tmp_path, base_spec(tmp_path, model_kind=kind, record_motion="/rec"))
    assert spec["model_kind"] in local.MODEL_KINDS and kind in local.MOTION_KINDS
    assert local.saves_attn(spec) is False
    command = local.build_server_command(spec, "nonce", 9000)
    assert "--stage2-python" not in command and "--motion-online-gpu" in command
    assert command[command.index("--motion-online-gpu") + 1] == "1"
    assert command[command.index("--record-motion") + 1] == "/rec"


def test_run_local_old_kinds_still_need_stage2(tmp_path):
    with pytest.raises(SystemExit, match="stage2_python"):
        load(tmp_path, base_spec(tmp_path, model_kind="stage2_pool4_concat", v1_store=None))
    spec = load(tmp_path, base_spec(tmp_path, model_kind="stage2_pool4_concat", v1_store=None, stage2_python="/s2/python"))
    command = local.build_server_command(spec, "nonce", 9000)
    assert command[command.index("--stage2-python") + 1] == "/s2/python"


def test_run_local_new_kinds_reject_save_attn_and_target_contract(tmp_path):
    with pytest.raises(SystemExit, match="不存注意力"):
        load(tmp_path, base_spec(tmp_path, save_attn=True))
    with pytest.raises(SystemExit, match="target_contract"):
        load(tmp_path, base_spec(tmp_path, target_contract="/c.json"))
    with pytest.raises(SystemExit, match="save_prefix"):
        load(tmp_path, base_spec(tmp_path, save_prefix=True))


def test_run_local_env_passthrough(tmp_path, monkeypatch):
    monkeypatch.setenv("MMEVLA_MOTION_ONLINE_GPU", "7")      # 外层残留值不得漏进 modulation worker
    monkeypatch.delenv("MMEVLA_MOTION_PROV_RELAX", raising=False)
    spec = load(tmp_path, base_spec(tmp_path, motion_prov_relax="gpu_name,compute_cap,sm_count",
                                    motion_overflow="resample"))
    env = local.build_environment(spec)
    assert env["MMEVLA_V1_STORE"] == "/turbo/v1-store"
    assert env["MMEVLA_MOTION_ONLINE_GPU"] == "1"
    assert env["MMEVLA_MOTION_PROV_RELAX"] == "gpu_name,compute_cap,sm_count"
    assert env["MMEVLA_MOTION_OVERFLOW"] == "resample"
    env = local.build_environment(load(tmp_path, base_spec(tmp_path, model_kind="wan_fixedpool768", motion_online_gpu="0")))
    assert env["MMEVLA_MOTION_ONLINE_GPU"] == "0"


def test_run_local_new_kinds_need_v1_store(tmp_path, monkeypatch):
    monkeypatch.delenv("MMEVLA_V1_STORE", raising=False)
    spec = load(tmp_path, base_spec(tmp_path, v1_store=None))
    with pytest.raises(ValueError, match="MMEVLA_V1_STORE"):
        local.build_environment(spec)


def test_run_local_env_keys_scoped(tmp_path):
    with pytest.raises(SystemExit, match="v1_store"):
        load(tmp_path, base_spec(tmp_path, model_kind="history", stage2_python="/s2"))


def test_run_local_plfull_max_steps(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "record_dataset_RouteStick_metadata.json").write_text("{}")
    plain = load(tmp_path, base_spec(tmp_path, profile="plfull", plfull_metadata_dir=str(meta)))
    spec = load(tmp_path, base_spec(tmp_path, profile="plfull", plfull_metadata_dir=str(meta), max_steps=2000))
    manifest = {"sha256": "b" * 64}
    assert local.run_identity(spec, manifest) != local.run_identity(plain, manifest)
    same = load(tmp_path, base_spec(tmp_path, profile="plfull", plfull_metadata_dir=str(meta), max_steps=1300))
    assert local.run_identity(same, manifest) == local.run_identity(plain, manifest)
    grant = tmp_path / "grant.json"
    grant.write_text(json.dumps({"chunk_id": "c"}))
    command = local.build_client_command(spec, grant, tmp_path / "o", "n", 9000, tmp_path / "d")
    assert command[command.index("--max-steps") + 1] == "2000"
    assert "--max-steps" not in local.build_client_command(plain, grant, tmp_path / "o", "n", 9000, tmp_path / "d")
    with pytest.raises(SystemExit, match="max_steps"):
        load(tmp_path, base_spec(tmp_path, profile="plfull", plfull_metadata_dir=str(meta), max_steps=1500))
    with pytest.raises(SystemExit, match="max_steps"):
        load(tmp_path, base_spec(tmp_path, max_steps=2000))


# ―― eval.py max_steps ――

def test_eval_max_steps_default_and_plfull_2000():
    from examples.robomme import eval as official
    assert official.Args().max_steps == 1300
    assert official.PLFULL_MAX_STEPS == (1300, 2000)
    # 1300 原口径预算判定不变；2000 只查帧数上限
    assert official.validate_budget(400, "plfull", 1300)["conservative_exec_windows"] == 40
    assert official.validate_budget(400, "plfull", 2000)["conservative_exec_windows"] == 2001 // 32
    with pytest.raises(official.ContractError):
        official.validate_budget(2200, "plfull", 2000)


def test_eval_plfull_runner_gets_max_steps(tmp_path):
    from examples.robomme import eval as official
    from tests.evaluation.test_plfull_client import KeyClient, make_args, metadata_dir
    from tests.evaluation.test_testx_client import FakeRunner, Recorder
    FakeRunner.created = []
    previous = official.RolloutRecorder
    official.RolloutRecorder = Recorder
    try:
        meta = metadata_dir(tmp_path)
        args = make_args(tmp_path, [("PatternLock", 1, "easy")], profile="plfull", plfull_metadata_dir=meta,
                         max_steps=2000)
        assert official.evaluate(args, runner_factory=FakeRunner, client_factory=lambda a: KeyClient()) == 0
        assert FakeRunner.created[0].kwargs["max_steps"] == 2000
    finally:
        official.RolloutRecorder = previous


@pytest.mark.parametrize("module", ["scripts.interpret.subgoal_trace_runner", "scripts.interpret.bin_trace_runner"])
def test_trace_runners_accept_max_steps(module):
    import importlib
    import tyro
    runner = importlib.import_module(module)
    assert runner.Args().max_steps == 1300
    args = tyro.cli(runner.Args, args=["--profile", "plfull", "--max-steps", "2000"])
    assert args.max_steps == 2000
