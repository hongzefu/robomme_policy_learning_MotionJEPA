"""testx 口径（1200 局 hold-out xhard，seed 7，max_steps 2000，motion 超预算 resample）的 CPU 契约测试。

不拉起仿真与 GPU；官方缺省口径的命令行、环境与响应须与改动前逐字相同。
"""
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation import manifest as manifest_module
from scripts.evaluation.manifest import build_testx_manifest, validate_episodes
from scripts.evaluation.queue_store import Controller, QueueStore, publish, summarize
from scripts.evaluation.resident_worker import client_command, server_command, worker_environment
from mme_vla_suite.policies.framesamp_memory import FrameSampMemory, _even_pick, _motion_quota
from mme_vla_suite.policies.policy import motion_overflow_from_env


TESTX_ROOT = Path("/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/testx")
TESTX_JSONL = TESTX_ROOT / "eval-testx-1200.jsonl"
TESTX_METADATA = TESTX_ROOT / "metadata"
COMMIT = "856bc3a189d4172f3f47dbee4424d585f8d78db3"


# ―― testx 清单转换 ――

def test_testx_manifest_from_real_jsonl(tmp_path):
    manifest, jsonl_sha = build_testx_manifest(TESTX_JSONL, TESTX_METADATA, benchmark_commit=COMMIT)
    assert jsonl_sha.startswith("20d4fbfb8430")
    validate_episodes(manifest, official=False)
    episodes = manifest["episodes"]
    assert len(episodes) == 1200 and len(manifest["chunks"]) == 240
    assert len({e["episode_key"] for e in episodes}) == 1200
    assert manifest["dataset"] == "test" and manifest["chunk_size"] == 5
    assert manifest["benchmark_commit"] == COMMIT
    assert episodes[0] == {"task_name": "BinFill", "episode_id": 0, "env_seed": 540000,
                           "difficulty": "easy", "episode_key": "BinFill:0"}
    counts = Counter(e["difficulty"] for e in episodes)
    assert counts == {"easy": 192, "medium": 192, "hard": 192, "xhard12": 132, "xhard14": 144,
                      "xhard16": 120, "xhard18": 120, "xhard20": 108}
    rows = [json.loads(line) for line in TESTX_JSONL.read_text().splitlines() if line.strip()]
    assert [(e["task_name"], e["episode_id"], e["env_seed"]) for e in episodes] == \
        [(r["task"], r["episode"], r["seed"]) for r in rows]
    metadata = manifest["metadata_sha256"]
    assert list(metadata) == sorted(p.name for p in TESTX_METADATA.glob("*_metadata.json"))
    assert len(metadata) == 14 and all(len(v) == 64 for v in metadata.values())

    # CLI 产物与函数结果一致，且控制器能直接按连续每五局接受这份清单
    out = tmp_path / "episodes.json"
    manifest_module.main(["testx-episodes", "--jsonl", str(TESTX_JSONL), "--benchmark-commit", COMMIT,
                          "--xhard-metadata-dir", str(TESTX_METADATA), "--out", str(out)])
    assert json.loads(out.read_text()) == manifest
    store = QueueStore(tmp_path / "run")
    store.initialize(manifest, {"run_identity_sha256": "b" * 64})
    with Controller(store) as controller:
        assert len(controller.chunks) == 240 and len(controller.episodes) == 1200
        assert controller.summary()["compute_status"] == "READY"


# ―― resident_worker 命令行与环境 ――

BASE_SPEC = {"repo": "/repo", "policy_python": "/policy/python", "stage2_python": "/sidecar/python",
             "sim_python": "/sim/python", "checkpoint": "/model/20000", "policy_config": "config",
             "run_identity_sha256": "a" * 64, "model_kind": "stage2_pool4_concat", "target_contract": "/runtime.json"}


def _grant(tmp_path):
    path = tmp_path / "grant.json"
    publish(path, {"chunk_id": "c" * 64})
    return path


def test_official_default_commands_and_environment_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixed")
    monkeypatch.delenv("XHARD_METADATA_DIR", raising=False)
    monkeypatch.delenv("MMEVLA_MOTION_OVERFLOW", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    grant = _grant(tmp_path)
    assert server_command(BASE_SPEC, "nonce", 1234) == [
        "/policy/python", "-m", "scripts.training.serve_policy", "--port", "1234",
        "--seed", "42", "--motion-online-gpu", "GPU-fixed", "--stage2-python", "/sidecar/python",
        "--run-identity-sha256", "a" * 64, "--server-nonce", "nonce", "--eval-purpose", "validation",
        "--eval-target-contract", "/runtime.json",
        "policy:checkpoint", "--policy.config", "config", "--policy.dir", "/model/20000"]
    assert client_command(BASE_SPEC, grant, tmp_path / "out", "nonce", 1234, tmp_path / "drain") == [
        "/sim/python", "/repo/examples/robomme/eval.py", "--episode-manifest", str(grant),
        "--chunk", "c" * 64, "--attempt-output", str(tmp_path / "out"),
        "--run-identity-sha256", "a" * 64, "--expected-server-nonce", "nonce",
        "--model-kind", "stage2_pool4_concat", "--model-seed", "42", "--host", "127.0.0.1",
        "--port", "1234", "--rpc-timeout-seconds", "600", "--drain-file", str(tmp_path / "drain")]
    paths = os.pathsep.join(["/repo", "/repo/src", "/repo/packages/openpi-client/src",
                             "/repo/third_party/robomme_benchmark/src"])
    assert worker_environment(BASE_SPEC) == dict(os.environ, PYTHONPATH=paths, PYTHONUNBUFFERED="1")
    # 显式写出官方缺省值也不改变命令与环境
    explicit = {**BASE_SPEC, "profile": "official", "model_seed": 42, "max_steps": 1300, "motion_overflow": "raise"}
    assert server_command(explicit, "nonce", 1234) == server_command(BASE_SPEC, "nonce", 1234)
    assert (client_command(explicit, grant, tmp_path / "out", "nonce", 1234, tmp_path / "drain")
            == client_command(BASE_SPEC, grant, tmp_path / "out", "nonce", 1234, tmp_path / "drain"))
    assert worker_environment(explicit) == worker_environment(BASE_SPEC)


def test_testx_commands_and_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixed")
    spec = {**BASE_SPEC, "profile": "testx", "model_seed": 7, "max_steps": 2000,
            "xhard_metadata_dir": "/xhard/metadata", "motion_overflow": "resample"}
    grant = _grant(tmp_path)
    server = server_command(spec, "nonce", 1234)
    assert server[server.index("--seed") + 1] == "7"
    official_client = client_command(BASE_SPEC, grant, tmp_path / "out", "nonce", 1234, tmp_path / "drain")
    client = client_command(spec, grant, tmp_path / "out", "nonce", 1234, tmp_path / "drain")
    assert client[client.index("--model-seed") + 1] == "7"
    assert client[len(official_client):] == ["--profile", "testx", "--max-steps", "2000"]
    environment = worker_environment(spec)
    assert environment["XHARD_METADATA_DIR"] == "/xhard/metadata"
    assert environment["MMEVLA_MOTION_OVERFLOW"] == "resample"


# ―― FrameSampMemory 超预算处理 ――

def _bare_memory(overflow, es, k_demo_frames=None):
    """不建 PosEmb3D 表、不加载模型：只设 _prepare_motion 读到的字段。"""
    memory = object.__new__(FrameSampMemory)
    memory.exec_start_idx = es
    memory.motion_window, memory.motion_stride, memory.motion_budget = 33, 32, 96
    memory.demo_min_real = 17
    memory.motion_dim, memory.motion_pos_dim = 4, 3
    memory.motion_overflow = overflow
    memory.motion_downsample_steps = 0
    memory.pos_emb = np.arange(4096 * 1 * 3, dtype=np.float32).reshape(4096, 1, 3)
    memory._history_feats_motion = {}
    return memory


ES = 1170          # demo 起点数 (1170-17)//32+1 = 37
STEP = ES + 1984   # exec 起点数 (1984-32)//32+1 = 62


def _fill(memory):
    for f in memory.visible_motion_frames(STEP):
        memory._history_feats_motion[f] = np.full(4, f, np.float32)


def test_prepare_motion_raise_default():
    memory = _bare_memory("raise", ES)
    _fill(memory)
    frames = memory.visible_motion_frames(STEP)
    assert len([f for f in frames if f < ES]) == 37 and len([f for f in frames if f >= ES]) == 62
    with pytest.raises(RuntimeError, match="budget"):
        memory._prepare_motion(STEP)
    assert memory.motion_downsample_steps == 0


def test_prepare_motion_resample_keeps_demo_and_thins_exec(capsys):
    memory = _bare_memory("resample", ES)
    _fill(memory)
    frames = memory.visible_motion_frames(STEP)
    demo = [f for f in frames if f < ES]
    exec_ = [f for f in frames if f >= ES]
    emb, pos, mask, times = memory._prepare_motion(STEP)
    kept = [int(t) for t in times]
    assert len(kept) == 96 and mask.all()
    assert all(b > a for a, b in zip(kept, kept[1:]))
    assert kept[:37] == demo
    assert kept[37:] == [exec_[i] for i in _even_pick(62, 59)] and set(kept[37:]) <= set(exec_)
    assert kept[37] == exec_[0] and kept[-1] == exec_[-1]
    np.testing.assert_array_equal(emb[:, 0], np.asarray(kept, np.float32))
    np.testing.assert_array_equal(pos, memory.pos_emb[kept, 0, :3])
    assert memory.motion_downsample_steps == 1
    assert ("MOTION_DOWNSAMPLE step=3154 es=1170 k=99 k_demo=37 k_exec=62 kept_demo=37 kept_exec=59 budget=96"
            in capsys.readouterr().out)
    memory._prepare_motion(STEP)
    assert memory.motion_downsample_steps == 2
    # 未超预算的步不计数
    memory._prepare_motion(ES + 32 * 40)
    assert memory.motion_downsample_steps == 2


def test_motion_quota_and_even_pick():
    assert _motion_quota(37, 62, 96) == (37, 59)
    assert _motion_quota(60, 62, 96) == (48, 48)
    assert _motion_quota(80, 20, 96) == (76, 20)
    assert _even_pick(5, 5) == [0, 1, 2, 3, 4]
    assert _even_pick(62, 59)[0] == 0 and _even_pick(62, 59)[-1] == 61


def test_clear_resets_downsample_counter():
    memory = _bare_memory("resample", ES)
    memory._history_feats, memory._raw_frames = {}, {}
    memory.motion_downsample_steps = 5
    memory.clear()
    assert memory.motion_downsample_steps == 0


# ―― policy 环境变量解析 ――

def test_motion_overflow_from_env(monkeypatch):
    monkeypatch.delenv("MMEVLA_MOTION_OVERFLOW", raising=False)
    assert motion_overflow_from_env() == "raise"
    monkeypatch.setenv("MMEVLA_MOTION_OVERFLOW", "resample")
    assert motion_overflow_from_env() == "resample"
    monkeypatch.setenv("MMEVLA_MOTION_OVERFLOW", "raise")
    assert motion_overflow_from_env() == "raise"
    monkeypatch.setenv("MMEVLA_MOTION_OVERFLOW", "truncate")
    with pytest.raises(ValueError):
        motion_overflow_from_env()


# ―― summarize 降级局计数 ――

def test_summarize_counts_downsampled_episodes(tmp_path):
    store = QueueStore(tmp_path)
    store.initialize()
    episodes = {f"BinFill:{i}": {} for i in range(4)}
    for i, steps in enumerate((0, 3, None, 1)):
        record = {"episode_key": f"BinFill:{i}", "outcome": "success"}
        if steps is not None:
            record["motion_downsample_steps"] = steps
        publish(tmp_path / "accepted" / f"{i}.json", record)
    state = {"assignments": {}, "errors": {}, "retries": {}, "uncertain": []}
    assert summarize(store, state, episodes)["downsampled_episodes"] == 2
