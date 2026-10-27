"""简化队列的三条核心CPU路径，不启动仿真或访问Slurm。"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from scripts.evaluation.manifest import make_manifest
from scripts.evaluation.queue_store import Controller, QueueStore, publish, read_json
from scripts.evaluation.resident_worker import server_command, client_command, worker_environment


def setup(root, count=10):
    store = QueueStore(root)
    episodes = [{"task_name": "BinFill", "episode_id": i, "episode_key": f"BinFill:{i}",
                 "env_seed": 100 + i, "difficulty": "easy"} for i in range(count)]
    store.initialize(make_manifest(episodes, official=False), {"run_identity_sha256": "a" * 64})
    return store


def submit_claim(item):
    root, worker = item
    QueueStore(root).request(f"claim-{worker}", "claim", {"worker_id": worker})


def records(grant, selected, outcome="success", *, exit_code=None):
    directory = Path(grant["attempt_dir"]) / "client"
    for e in selected:
        identity = {**e, "run_identity_sha256": grant["run_identity_sha256"], "chunk_id": grant["chunk_id"],
                    "episode_nonce": f"nonce-{e['episode_key']}"}
        publish(directory / f"{e['episode_key']}.started.json", {**identity, "record_type": "started"})
        publish(directory / f"{e['episode_key']}.terminal.json", {**identity, "record_type": "terminal",
                "outcome": outcome, "actions_executed": 1, "rpc_seq": 1})
    if exit_code is not None:
        done = [e["episode_key"] for e in selected]
        publish(directory / "attempt.json", {"exit_code": exit_code, "completed_episode_keys": done,
                "started_episode_keys": done,
                "unstarted_episode_keys": [e["episode_key"] for e in grant["episodes"] if e["episode_key"] not in done]})


def test_parallel_claims_and_simple_execution_commands(tmp_path, monkeypatch):
    store = setup(tmp_path)
    with Controller(store) as c:
        c.register("one", "10", "0")
        c.register("two", "20", "0")
        with ProcessPoolExecutor(2) as pool:
            list(pool.map(submit_claim, [(str(tmp_path), "one"), (str(tmp_path), "two")]))
        c.process_requests()
        grants = [store.query(f"claim-{name}")["grant"] for name in ("one", "two")]
        assert len({e["episode_key"] for g in grants for e in g["episodes"]}) == 10
        assert all(len(g["episodes"]) == 5 for g in grants)
        assert c.claim("claim-one", "one")["grant"] == grants[0]
        with pytest.raises(ValueError, match="已有一块"):
            c.claim("new-request", "one")
        for grant in grants:
            records(grant, grant["episodes"], "fail", exit_code=0)
            c.finish(grant["attempt_id"], sim_stopped=True)
        assert c.summary()["compute_status"] == "COMPUTE_DONE"
        assert c.summary()["outcomes"]["fail"] == 10
        assert c.summary()["retries"] == {}
    with Controller(store) as c:
        assert c.claim("claim-one", "one")["grant"] == grants[0]
        assert c.summary()["completed"] == 10
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixed")
    monkeypatch.setenv("HF_HOME", "/explicit/cache")
    spec = {"repo": "/repo", "policy_python": "/policy/python", "stage2_python": "/sidecar/python",
            "sim_python": "/sim/python", "checkpoint": "/model/20000", "policy_config": "config",
            "run_identity_sha256": "a" * 64, "model_kind": "stage2_pool4_concat", "target_contract": "/runtime.json"}
    server = server_command(spec, "nonce", 1234)
    assert "--model-binding" not in server and "--checkpoint-step" not in server
    assert server[server.index("--motion-online-gpu") + 1] == "GPU-fixed"
    assert server[server.index("--eval-target-contract") + 1] == "/runtime.json"
    assert worker_environment(spec)["HF_HOME"] == "/explicit/cache"
    path = tmp_path / "grant.json"
    publish(path, grants[0])
    client = client_command(spec, path, tmp_path / "output", "nonce", 1234, tmp_path / "drain")
    assert client[client.index("--chunk") + 1] == grants[0]["chunk_id"]


def test_join_drain_and_return_unstarted(tmp_path):
    store = setup(tmp_path, 5)
    with Controller(store) as c:
        c.register("old", "10", "0")
        grant = c.claim("old-claim", "old")["grant"]
        records(grant, grant["episodes"][:2], "timeout", exit_code=3)
        store.request("drain", "drain", {"worker_id": "old"})
        c.process_requests()
        assert store.query("drain")["status"] == "DRAINING"
        c.finish(grant["attempt_id"], sim_stopped=True)
        assert c.claim("old-after-drain", "old")["status"] == "DRAIN"
        c.register("new", "20", "0")
        remaining = c.claim("new-claim", "new")["grant"]
        assert remaining["episodes"] == grant["episodes"][2:]
        assert c.summary()["retries"] == {} and c.summary()["uncertain"] == []
        records(remaining, remaining["episodes"], exit_code=0)
        c.finish(remaining["attempt_id"], sim_stopped=True)
        assert c.summary()["compute_status"] == "COMPUTE_DONE"
        assert c.summary()["outcomes"] == {"success": 3, "fail": 0, "timeout": 2}


def test_fault_requires_exact_step_then_only_one_retry(tmp_path):
    store = setup(tmp_path, 5)
    verified = False

    def terminal(job, step):
        if not verified:
            raise ValueError("旧step仍在运行")
        return {"job_id": job, "step_id": step, "state": "FAILED", "end": "known"}

    with Controller(store, terminal_lookup=terminal) as c:
        c.register("old", "10", "0")
        grant = c.claim("first", "old")["grant"]
        records(grant, grant["episodes"][:1])
        e = grant["episodes"][1]
        identity = {**e, "record_type": "started", "run_identity_sha256": "a" * 64,
                    "chunk_id": grant["chunk_id"], "episode_nonce": "inflight"}
        publish(Path(grant["attempt_dir"]) / "client" / f"{e['episode_key']}.started.json", identity)
        c.register("waiting", "20", "0")
        assert c.claim("not-stolen", "waiting")["status"] == "WAIT"
        with pytest.raises(ValueError, match="原worker"):
            c.recover("old", "10", "1")
        with pytest.raises(ValueError, match="仍在运行"):
            c.recover("old", "10", "0")
        verified = True
        c.recover("old", "10", "0")
        assert c.summary()["completed"] == 1
        assert c.summary()["retries"] == {e["episode_key"]: 1}
        remaining = c.claim("retry", "waiting")["grant"]
        assert remaining["episodes"] == grant["episodes"][1:]
        records(remaining, remaining["episodes"][1:], exit_code=None)
        publish(Path(remaining["attempt_dir"]) / "client" / f"{e['episode_key']}.started.json",
                {**identity, "chunk_id": remaining["chunk_id"]})
        c.recover("waiting", "20", "0")
        assert c.summary()["compute_status"] == "STOPPED_ERROR"
        assert c.summary()["completed"] == 4
        assert c.summary()["retries"] == {e["episode_key"]: 1}
        assert e["episode_key"] in c.summary()["errors"]
        # 旧attempt恢复后迟到，不能再导入或覆盖已接受记录。
        before = store.accepted()
        c.import_results(grant["attempt_id"])
        assert store.accepted() == before
