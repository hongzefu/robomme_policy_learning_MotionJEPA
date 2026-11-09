"""plfull 口径（PatternLock 全因子交换自建 80 局清单）客户端与 metadata 覆盖路由的 CPU 测试；不创建仿真、不起模型。"""

import json
from pathlib import Path

import pytest

from examples.robomme import eval as candidate
from examples.robomme import env_runner as runner_module
from tests.evaluation.test_testx_client import FakeClient, FakeRunner, Recorder, fake_sim  # noqa: F401


class KeyClient(FakeClient):
    """记下每次 reset 的 eval_rpc，用来核对 plfull 是否带 episode_key。"""

    def __init__(self, model_seed=42):
        super().__init__(model_seed=model_seed, downsample=False)
        self.resets = []

    def request(self, payload):
        if payload.get("reset"):
            self.resets.append(dict(payload["eval_rpc"]))
        return super().request(payload)


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


def metadata_dir(tmp_path):
    directory = tmp_path / "metadata"
    directory.mkdir(exist_ok=True)
    return str(directory)


def test_cli_parses_plfull():
    import tyro
    args = tyro.cli(candidate.Args, args=["--profile", "plfull", "--plfull-metadata-dir", "/m", "--model-seed", "1"])
    assert (args.profile, args.plfull_metadata_dir, args.model_seed) == ("plfull", "/m", 1)
    assert candidate.Args().plfull_metadata_dir == ""


@pytest.mark.parametrize("seed", [42, 1, 2])
def test_plfull_accepts_ep_0_to_79_and_passes_metadata(tmp_path, seed):
    meta = metadata_dir(tmp_path)
    episodes = [("PatternLock", 0, "easy"), ("PatternLock", 41, "easy"), ("PatternLock", 79, "easy")]
    args = make_args(tmp_path, episodes, profile="plfull", plfull_metadata_dir=meta, model_seed=seed)
    client = KeyClient(model_seed=seed)
    assert candidate.evaluate(args, runner_factory=FakeRunner, client_factory=lambda a: client) == 0
    assert [r.kwargs for r in FakeRunner.created] == [
        {"max_steps": 1300, "dataset": "test", "difficulty": "easy", "metadata_dir": meta}] * 3
    # 每局 reset 的 eval_rpc 带本局 episode_key，服务端据此查供体
    assert [r["episode_key"] for r in client.resets] == ["PatternLock:0", "PatternLock:41", "PatternLock:79"]
    record = json.loads((Path(args.attempt_output) / "PatternLock:79.terminal.json").read_text())
    assert record["outcome"] == "success" and "motion_downsample_steps" not in record


def test_plfull_rejects_ep_80_and_bad_seed_and_missing_dir(tmp_path):
    meta = metadata_dir(tmp_path)
    with pytest.raises(candidate.ContractError):
        candidate.load_chunk(make_args(tmp_path, [("PatternLock", 80, "easy")], profile="plfull"))
    assert candidate.load_chunk(make_args(tmp_path, [("PatternLock", 79, "easy")], profile="plfull"))
    with pytest.raises(candidate.ContractError, match="seed"):
        candidate.evaluate(make_args(tmp_path, [("PatternLock", 1, "easy")], profile="plfull",
                                     plfull_metadata_dir=meta, model_seed=7), runner_factory=FakeRunner)
    with pytest.raises(candidate.ContractError, match="plfull-metadata-dir"):
        candidate.evaluate(make_args(tmp_path, [("PatternLock", 1, "easy")], profile="plfull",
                                     plfull_metadata_dir=str(tmp_path / "nope")), runner_factory=FakeRunner)
    with pytest.raises(candidate.ContractError, match="1300"):
        candidate.evaluate(make_args(tmp_path, [("PatternLock", 1, "easy")], profile="plfull",
                                     plfull_metadata_dir=meta, max_steps=2000), runner_factory=FakeRunner)


def test_plfull_seed_mismatch_is_contract_error(tmp_path):
    """resolve_episode 的 seed／难度核对在 plfull 下照做。"""
    meta = metadata_dir(tmp_path)
    args = make_args(tmp_path, [("PatternLock", 3, "easy")], profile="plfull", plfull_metadata_dir=meta)
    manifest = json.loads(Path(args.episode_manifest).read_text())
    manifest["episodes"][0]["env_seed"] = 999
    Path(args.episode_manifest).write_text(json.dumps(manifest))
    assert candidate.evaluate(args, runner_factory=FakeRunner, client_factory=lambda a: KeyClient()) == 2
    record = json.loads((Path(args.attempt_output) / "PatternLock:3.infra.json").read_text())
    assert record["error_type"] == "ContractError" and "seed" in record["error"]


def test_official_unchanged_and_refuses_plfull_keys(tmp_path):
    with pytest.raises(candidate.ContractError):
        candidate.load_chunk(make_args(tmp_path, [("PatternLock", 50, "easy")]))
    with pytest.raises(candidate.ContractError, match="plfull"):
        candidate.evaluate(make_args(tmp_path, [("PatternLock", 1, "easy")], plfull_metadata_dir=metadata_dir(tmp_path)),
                           runner_factory=FakeRunner)
    client = KeyClient()
    # FakeRunner 在官方口径下 resolve_episode 返回 hard
    args = make_args(tmp_path, [("PatternLock", 1, "hard")])
    assert candidate.evaluate(args, runner_factory=FakeRunner, client_factory=lambda a: client) == 0
    assert FakeRunner.created[0].kwargs == {"max_steps": 1300, "dataset": "test"}
    assert "episode_key" not in client.resets[0]


def test_plfull_metadata_route(tmp_path, fake_sim, capsys):  # noqa: F811
    meta = metadata_dir(tmp_path)
    runner = runner_module.EnvRunner("PatternLock", str(tmp_path), max_steps=1300, dataset="test",
                                     difficulty="easy", metadata_dir=meta)
    builder = fake_sim.instances[0]
    assert builder.kwargs == {"env_id": "PatternLock", "dataset": "test", "action_space": "joint_angle",
                              "gui_render": False, "override_metadata_path": Path(meta), "max_steps": 1300}
    builder.difficulty = "easy"
    runner.make_env(61)
    assert "ENV_ROUTE route=plfull env=PatternLock task=PatternLock episode=61 difficulty=easy" in capsys.readouterr().out
    builder.difficulty = "hard"
    with pytest.raises(ValueError):
        runner.make_env(61)


@pytest.mark.parametrize("difficulty, sub", [(None, None), ("xhard12", None), ("easy", "missing")])
def test_plfull_metadata_route_rejects(tmp_path, fake_sim, difficulty, sub):  # noqa: F811
    meta = metadata_dir(tmp_path) if sub is None else str(tmp_path / sub)
    with pytest.raises(ValueError, match="metadata_dir"):
        runner_module.EnvRunner("PatternLock", str(tmp_path), difficulty=difficulty, metadata_dir=meta)


def test_subgoal_trace_runner_accepts_plfull(tmp_path, monkeypatch):
    """轨迹客户端经官方 evaluate；plfull 局号 0–79 通过它的任务预检，runner 换成 TraceRunner。"""
    from scripts.interpret import subgoal_trace_runner as trace
    seen = {}

    def fake_evaluate(args, runner_factory):
        seen.update(args=args, factory=runner_factory, evaluator=candidate.EpisodeEvaluator)
        return 0

    monkeypatch.setattr(trace.official, "evaluate", fake_evaluate)
    args = make_args(tmp_path, [("PatternLock", 70, "easy")], profile="plfull", plfull_metadata_dir=metadata_dir(tmp_path))
    assert trace.evaluate(args) == 0
    assert seen["factory"] is trace.TraceRunner and seen["evaluator"] is trace.TraceEvaluator
    assert candidate.EpisodeEvaluator is not trace.TraceEvaluator
