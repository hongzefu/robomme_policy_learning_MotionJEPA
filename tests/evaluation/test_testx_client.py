"""testx 口径客户端与 xhard 路由的 CPU 测试；不创建仿真环境、不启动模型。"""

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from examples.robomme import eval as candidate
from examples.robomme import env_runner as runner_module


IMAGE = np.full((256, 256, 3), 7, np.uint8)
WRIST = np.full((256, 256, 3), 11, np.uint8)
STATE = np.arange(8, dtype=np.float32)


class Recorder:
    def __init__(self, save_dir, task_goal, fps):
        self.save_dir = Path(save_dir)

    def record(self, **frame):
        pass

    def save_video(self, filename):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / filename).write_bytes(b"fixture-video")


class FakeRunner:
    stop_at = 40
    demo = 33
    created = []

    def __init__(self, env_id, video_save_dir=None, **kwargs):
        self.env_id = env_id
        self.kwargs = kwargs
        difficulty = kwargs.get("difficulty", "hard")
        self.env_builder = SimpleNamespace(resolve_episode=lambda episode_id: (episode_id + 100, difficulty))
        # testx 不检查局数；official 仍要求 50。
        self.num_episodes = 110 if "difficulty" in kwargs else 50
        self.count = 0
        self.created.append(self)

    def make_env(self, episode_id):
        self.episode_id = episode_id

    def get_init_obs(self):
        n = self.demo + 1
        return {"images": [IMAGE] * n, "wrist_images": [WRIST] * n, "states": [STATE] * n, "task_goal": "prompt"}

    def step(self, action):
        self.count += 1
        done = self.count == self.stop_at
        return (IMAGE, WRIST, STATE), done, "success" if done else "unknown"

    def close_env(self):
        pass


class FakeClient:
    def __init__(self, model_seed=7, downsample=True):
        self.metadata = {"run_identity_sha256": "a" * 64, "server_nonce": "server", "model_seed": model_seed,
                         "model_kind": "stage2"}
        self.downsample = downsample
        self.infers = 0
        self.closed = False

    def request(self, payload):
        rpc = payload["eval_rpc"]
        if "observation/image" in payload:
            self.infers += 1
            response = {"actions": np.zeros((20, 8), np.float32), "eval_rpc": rpc}
            if self.downsample:
                response["motion_downsample_steps"] = self.infers * 10
            return response
        operation = "reset" if payload.get("reset") else "add_buffer" if payload.get("add_buffer") else "release_attempt"
        return {f"{operation}_finished": True, "eval_rpc": rpc, "quiescent": True}

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(candidate, "RolloutRecorder", Recorder)
    FakeRunner.created = []


def make_args(tmp_path, episodes, **overrides):
    members = [{"episode_key": f"{task}:{ep}", "task_name": task, "episode_id": ep, "env_seed": ep + 100,
                "difficulty": difficulty} for task, ep, difficulty in episodes]
    manifest = tmp_path / "grant.json"
    manifest.write_text(json.dumps({"chunk_id": "chunk", "run_identity_sha256": "a" * 64, "episodes": members}))
    values = dict(episode_manifest=str(manifest), chunk="chunk", attempt_output=str(tmp_path / "client"),
                  run_identity_sha256="a" * 64, expected_server_nonce="server", model_kind="stage2")
    values.update(overrides)
    return candidate.Args(**values)


def test_cli_parses_profile_max_steps_model_seed():
    import tyro
    args = tyro.cli(candidate.Args, args=["--profile", "testx", "--max-steps", "2000", "--model-seed", "7"])
    assert (args.profile, args.max_steps, args.model_seed) == ("testx", 2000, 7)
    assert candidate.Args().profile == "official"


def test_testx_accepts_xhard_seed7_2000_steps(tmp_path):
    episodes = [("BinFill", 50, "xhard12"), ("MoveCube", 109, "xhard20"), ("PickXtimes", 3, "easy")]
    args = make_args(tmp_path, episodes, profile="testx", max_steps=2000, model_seed=7)
    client = FakeClient()
    assert candidate.evaluate(args, runner_factory=FakeRunner, client_factory=lambda a: client) == 0
    calls = [(runner.env_id, runner.kwargs) for runner in FakeRunner.created]
    assert calls == [(task, {"max_steps": 2000, "dataset": "test", "difficulty": difficulty})
                     for task, _, difficulty in episodes]
    output = Path(args.attempt_output)
    first = json.loads((output / "BinFill:50.terminal.json").read_text())
    # stop_at=40 → 3 次推理，取最后一次响应值 30。
    assert first["outcome"] == "success" and first["motion_downsample_steps"] == 30
    assert first["budget"] == {"demo_frames": 33, "demo_windows": 1, "conservative_exec_windows": 2001 // 32}


def test_testx_downsample_defaults_to_zero(tmp_path):
    args = make_args(tmp_path, [("BinFill", 60, "xhard14")], profile="testx", max_steps=2000, model_seed=7)
    client = FakeClient(downsample=False)
    assert candidate.evaluate(args, runner_factory=FakeRunner, client_factory=lambda a: client) == 0
    record = json.loads((Path(args.attempt_output) / "BinFill:60.terminal.json").read_text())
    assert record["motion_downsample_steps"] == 0


def test_official_record_unchanged(tmp_path):
    args = make_args(tmp_path, [("BinFill", 0, "hard")])
    client = FakeClient(model_seed=42)
    assert candidate.evaluate(args, runner_factory=FakeRunner, client_factory=lambda a: client) == 0
    assert FakeRunner.created[0].kwargs == {"max_steps": 1300, "dataset": "test"}
    record = json.loads((Path(args.attempt_output) / "BinFill:0.terminal.json").read_text())
    assert "motion_downsample_steps" not in record


def test_testx_budget():
    assert candidate.validate_budget(1173, "testx", 2000)["conservative_exec_windows"] == 2001 // 32
    with pytest.raises(candidate.ContractError):
        candidate.validate_budget(4096 - 2001, "testx", 2000)
    # testx 不按窗口数拒绝；官方口径不变，同一 D=2000 因窗口数（62+40>96）被拒。
    assert candidate.validate_budget(2000, "testx", 2000)["demo_frames"] == 2000
    with pytest.raises(candidate.ContractError):
        candidate.validate_budget(2000)


@pytest.mark.parametrize("overrides", [{"model_seed": 7}, {"max_steps": 2000}])
def test_official_still_rejects_testx_values(tmp_path, overrides):
    args = make_args(tmp_path, [("BinFill", 0, "hard")], **overrides)
    with pytest.raises(candidate.ContractError):
        candidate.evaluate(args, runner_factory=FakeRunner, client_factory=lambda a: FakeClient())


def test_episode_limits(tmp_path):
    official = make_args(tmp_path, [("BinFill", 50, "hard")])
    with pytest.raises(candidate.ContractError):
        candidate.load_chunk(official)
    testx = make_args(tmp_path, [("BinFill", 110, "xhard12")], profile="testx")
    with pytest.raises(candidate.ContractError):
        candidate.load_chunk(testx)
    ok = make_args(tmp_path, [("BinFill", 109, "xhard12")], profile="testx")
    assert candidate.load_chunk(ok)[0]["episode_id"] == 109


class FakeBuilder:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)

    def resolve_episode(self, episode_id):
        return episode_id + 100, "unused"

    def make_env_for_episode(self, episode_id):
        return SimpleNamespace(unwrapped=SimpleNamespace(difficulty=self.difficulty))


@pytest.fixture
def fake_sim(monkeypatch):
    FakeBuilder.instances = []
    robomme = ModuleType("robomme")
    wrapper = ModuleType("robomme.env_record_wrapper")
    wrapper.BenchmarkEnvBuilder = FakeBuilder
    monkeypatch.setitem(sys.modules, "robomme", robomme)
    monkeypatch.setitem(sys.modules, "robomme.robomme_env", ModuleType("robomme.robomme_env"))
    monkeypatch.setitem(sys.modules, "robomme.env_record_wrapper", wrapper)
    monkeypatch.setitem(sys.modules, "robomme_xhard", ModuleType("robomme_xhard"))
    return FakeBuilder


def test_xhard_route(tmp_path, monkeypatch, fake_sim, capsys):
    monkeypatch.setenv("XHARD_METADATA_DIR", str(tmp_path))
    runner = runner_module.EnvRunner("BinFill", str(tmp_path), max_steps=2000, dataset="test", difficulty="xhard16")
    builder = fake_sim.instances[0]
    assert builder.kwargs["env_id"] == "BinFillXHard"
    assert builder.kwargs["override_metadata_path"] == Path(tmp_path)
    assert builder.kwargs["max_steps"] == 2000
    assert runner.env_id == "BinFill"
    builder.difficulty = "xhard16"
    runner.make_env(55)
    assert "ENV_ROUTE route=xhard env=BinFillXHard task=BinFill episode=55 difficulty=xhard16" in capsys.readouterr().out
    builder.difficulty = "xhard12"
    with pytest.raises(ValueError):
        runner.make_env(55)


def test_base_route(tmp_path, fake_sim, capsys):
    runner = runner_module.EnvRunner("MoveCube", str(tmp_path), max_steps=2000, dataset="test", difficulty="hard")
    builder = fake_sim.instances[0]
    assert builder.kwargs["env_id"] == "MoveCube" and "override_metadata_path" not in builder.kwargs
    assert runner.env_id == "MoveCube"
    builder.difficulty = "hard"
    runner.make_env(3)
    assert "ENV_ROUTE route=base env=MoveCube task=MoveCube episode=3 difficulty=hard" in capsys.readouterr().out


def test_official_route_unchanged(tmp_path, fake_sim, capsys):
    runner_module.EnvRunner("BinFill", str(tmp_path))
    assert fake_sim.instances[0].kwargs == {"env_id": "BinFill", "dataset": "test", "action_space": "joint_angle",
                                            "gui_render": False, "max_steps": 1300}


@pytest.mark.parametrize("value", [None, "missing"])
def test_xhard_requires_metadata_dir(tmp_path, monkeypatch, fake_sim, value):
    if value is None:
        monkeypatch.delenv("XHARD_METADATA_DIR", raising=False)
    else:
        monkeypatch.setenv("XHARD_METADATA_DIR", str(tmp_path / value))
    with pytest.raises(ValueError, match="XHARD_METADATA_DIR"):
        runner_module.EnvRunner("BinFill", str(tmp_path), max_steps=2000, difficulty="xhard12")
