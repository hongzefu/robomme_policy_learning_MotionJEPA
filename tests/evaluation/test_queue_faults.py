"""CPU文件队列故障、并发授权、重试与不确定历史的真实测试。"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.evaluation.manifest import digest, make_manifest
from scripts.evaluation.queue_store import Controller, QueueStore, publish, read_json


def setup_queue(tmp_path, count=5, chunks=5, mode="production"):
    manifest = make_manifest([{"task_name": "Task", "episode_id": i, "env_seed": 100 + i,
                               "difficulty": "easy"} for i in range(count)], chunk_size=chunks, official=False)
    store = QueueStore(tmp_path / "run")
    store.initialize(manifest, "a" * 64, official=False, mode=mode)
    return store


def slot(i=0, state="RUNNING"):
    return {"slot_instance": f"slot-{i}", "worker_uuid": f"worker-{i}", "job_id": str(1000 + i),
            "job_incarnation": f"start-{i}", "step_id": "0", "state": state}


def proof(s):
    return {**{k: v for k, v in s.items() if k != "state"}, "terminal": True,
            "rpc_quiescent": True, "sidecar_quiescent": True}


def terminal(store, grant, e, outcome="success"):
    return store.write_attempt(grant, "terminal", {**e, "episode_nonce": "nonce",
                               "outcome": outcome, "actions_executed": 10, "rpc_seq": 3}, e["episode_key"])


@pytest.mark.parametrize("workers", [1, 2, 8])
def test_concurrent_requests_unique_and_fixed_coverage(tmp_path, workers):
    store = setup_queue(tmp_path, 20)
    with Controller(store) as controller:
        for i in range(workers):
            controller.register_slot(slot(i))
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(lambda i: store.request(f"request-{i}", "claim", {"slot_instance": f"slot-{i}"}), range(workers)))
        controller.process_requests()
        grants = [store.query(f"request-{i}") for i in range(workers)]
        grants = [r["grant"] for r in grants if r and r["status"] == "GRANTED"]
        assert len({g["chunk_id"] for g in grants}) == len(grants)
        for g in grants:
            assert controller.claim(g["claim_request_id"], g["slot"]["slot_instance"])["grant"] == g
            for e in g["episodes"]:
                assert controller.accept_terminal(terminal(store, g, e))
            controller.close_assignment(g["attempt_id"], sim_stopped=True)
        n = 100
        while controller.summary()["compute_status"] == "READY":
            n += 1
            g = controller.claim(f"next-{n}", "slot-0")["grant"]
            for e in g["episodes"]:
                assert controller.accept_terminal(terminal(store, g, e))
            controller.close_assignment(g["attempt_id"], sim_stopped=True)
        result = controller.summary()
        assert result["compute_status"] == "COMPUTE_DONE"
        assert result["execution_coverage"] == 20
        assert result["score_status"] == "PASS"
        assert result["artifact_status"] == "BLOCKED_ARTIFACTS"


def test_process_lifetime_lock_and_lost_grant(tmp_path):
    store = setup_queue(tmp_path)
    with Controller(store) as c:
        with pytest.raises(RuntimeError, match="持锁"):
            with Controller(store):
                pass
        c.register_slot(slot())
        g = c.claim("lost-reply", "slot-0")["grant"]
        with pytest.raises(RuntimeError, match="query"):
            c.claim("wrong-new-id", "slot-0")
    with Controller(store) as c:
        assert c.claim("lost-reply", "slot-0")["grant"] == g
        assert c.summary()["compute_status"] == "WAIT"


def test_ownerless_freezes_all_membership_no_ttl(tmp_path):
    store = setup_queue(tmp_path)
    with Controller(store) as c:
        for i, state in enumerate(["RUNNING", "STARTING", "DRAINING", "CANCEL_REQUESTED", "UNKNOWN", "SUBMITTING"]):
            c.register_slot(slot(i, state))
        chunk = c.manifest["chunks"][0]["chunk_id"]
        (store.root / "queue/claims" / f"{chunk}-0").mkdir()
    with Controller(store) as c:
        freeze = c.state["freeze"]
        assert len(freeze["slots"]) == 6
        assert c.claim("wait", "slot-0")["status"] == "WAIT"
        assert c.request_resize(3, c.state["revision"])["status"] == "QUEUED"
        with pytest.raises(ValueError, match="全部"):
            c.recover([proof(slot())])
        bad = [proof(s) for s in freeze["slots"].values()]
        bad[0]["job_incarnation"] = "reused-job-id"
        with pytest.raises(ValueError, match="incarnation"):
            c.recover(bad)
        c.recover([proof(s) for s in freeze["slots"].values()])
        assert c.state["generations"][chunk] == 1
    with Controller(store) as c:
        assert c.state["phase"] == "RUNNING"


def test_partial_crash_accept_before_retry_and_late_fence(tmp_path):
    store = setup_queue(tmp_path)
    with Controller(store) as c:
        c.register_slot(slot())
        g = c.claim("first", "slot-0")["grant"]
        for e in g["episodes"][:2]:
            terminal(store, g, e)
        e = g["episodes"][2]
        store.write_attempt(g, "started", e, "start")
        c.begin_recovery("crash")
        c.recover([proof(slot())])
        assert c.summary()["execution_coverage"] == 2
        assert c.state["retries"] == {e["episode_key"]: 1}
        assert c.summary()["U"] == 1
        c.register_slot(slot(1))
        retry = c.claim("second", "slot-1")["grant"]
        assert len(retry["episodes"]) == 3
        assert [e["env_seed"] for e in retry["episodes"]] == [102, 103, 104]
        for e in retry["episodes"]:
            assert c.accept_terminal(terminal(store, retry, e))
        c.close_assignment(retry["attempt_id"], sim_stopped=True)
        assert not c.accept_terminal(terminal(store, g, g["episodes"][2]))
        assert c.summary()["score_status"] == "QUALIFIED"
        assert c.summary()["U"] == 1
        assert c.summary()["diagnostic_interval"] == [0.8, 1.0]
    with Controller(store) as c:
        assert c.summary()["U"] == 1


def test_retry_exhaustion_invalid_distinct_unknown(tmp_path):
    store = setup_queue(tmp_path, 2, chunks=1)
    with Controller(store) as c:
        for i in range(2):
            c.register_slot(slot(i))
            g = c.claim(f"claim-{i}", f"slot-{i}")["grant"]
            e = g["episodes"][0]
            store.write_attempt(g, "infra", {**e, "retryable": True, "uncertain_terminal": True}, "infra")
            c.begin_recovery("infra")
            c.recover([proof(s) for s in c.state["freeze"]["slots"].values()])
        result = c.summary()
        assert result["score_status"] == "INVALID"
        assert result["U"] == 1 and result["K"] == 2
        c.register_slot(slot(2))
        g = c.claim("last", "slot-2")["grant"]
        for e in g["episodes"]:
            c.accept_terminal(terminal(store, g, e))
        c.close_assignment(g["attempt_id"], sim_stopped=True)
        assert c.summary()["compute_status"] == "STOPPED_ERROR"


def test_canonical_copy_and_artifact_tamper(tmp_path):
    store = setup_queue(tmp_path, 1)
    with Controller(store) as c:
        c.register_slot(slot())
        g = c.claim("one", "slot-0")["grant"]
        e = g["episodes"][0]
        path = terminal(store, g, e)
        assert c.accept_terminal(path)
        accepted_before = c._accepted()
        path.write_text("{}")
        assert c._accepted() == accepted_before
        asset = Path(g["attempt_dir"]) / "video.bin"
        asset.write_bytes(b"video")
        import hashlib
        receipt = store.write_attempt(g, "artifact", {**e, "files": [{"path": str(asset.relative_to(store.root)),
                    "bytes": 5, "sha256": hashlib.sha256(b"video").hexdigest()}], "artifact_error": False}, "art")
        assert c.accept_artifact(receipt)
        assert c.summary()["artifact_coverage"] == 1
        close_error = store.write_attempt(g, "artifact", {**e, "files": [], "artifact_error": True}, "close-error")
        assert not c.accept_artifact(close_error)
        assert c.summary()["artifact_coverage"] == 0
        error_id = digest(read_json(close_error))
        repair = store.write_attempt(g, "artifact", {**read_json(receipt)["payload"], "resolves": [error_id]}, "repair")
        assert c.accept_artifact(repair)
        assert c.summary()["artifact_coverage"] == 1
        asset.write_bytes(b"other")
        assert c.summary()["artifact_coverage"] == 0
        assert c.summary()["score_status"] == "PASS"
        with pytest.raises(ValueError, match="路径注入"):
            store.write_attempt({**g, "attempt_id": "../../accepted"}, "terminal", e, "escape")


def test_recovery_transaction_replay_does_not_consume_retry(tmp_path, monkeypatch):
    store = setup_queue(tmp_path, 1)
    with Controller(store) as c:
        c.register_slot(slot())
        g = c.claim("first", "slot-0")["grant"]
        store.write_attempt(g, "started", g["episodes"][0], "start")
        c.begin_recovery("crash")
        original = c._apply_recovery
        monkeypatch.setattr(c, "_apply_recovery", lambda event: (_ for _ in ()).throw(RuntimeError("power loss")))
        with pytest.raises(RuntimeError, match="power loss"):
            c.recover([proof(slot())])
    with Controller(store) as c:
        c.recover([proof(slot())])
        assert c.state["retries"] == {"Task:0": 1}
        assert c.state["exhausted"] == []
        assert c.state["generations"][g["chunk_id"]] == 1


def test_bootstrap_and_identity_negative(tmp_path):
    store = setup_queue(tmp_path, mode="measurement_bootstrap")
    with Controller(store) as c:
        c.register_slot(slot())
        with pytest.raises(RuntimeError, match="bootstrap"):
            c.claim("no", "slot-0")
    store = setup_queue(tmp_path / "other", 1)
    with Controller(store) as c:
        c.register_slot(slot())
        g = c.claim("yes", "slot-0")["grant"]
        bad = terminal(store, g, {**g["episodes"][0], "env_seed": 999})
        assert not c.accept_terminal(bad)
        assert not c._accepted()


def test_worker_registration_requires_controller_permit(tmp_path):
    store = setup_queue(tmp_path)
    s = slot()
    with Controller(store) as c:
        c.authorize_slot({k: s[k] for k in ("slot_instance", "worker_uuid", "job_id", "job_incarnation")})
        store.request("register", "register_slot", {"slot": s})
        c.process_requests()
        assert store.query("register")["status"] == "REGISTERED"
        store.request("claim", "claim", {"slot_instance": s["slot_instance"]})
        c.process_requests()
        assert store.query("claim")["status"] == "GRANTED"


def _worker_request(pair):
    root, i = pair
    QueueStore(root).request(f"multiprocess-{i}", "claim", {"slot_instance": f"slot-{i}"})


def test_actual_multiprocess_request_publication(tmp_path):
    store = setup_queue(tmp_path, 20)
    with Controller(store) as c:
        for i in range(4):
            c.register_slot(slot(i))
        with ProcessPoolExecutor(4) as pool:
            list(pool.map(_worker_request, [(str(store.root), i) for i in range(4)]))
        c.process_requests()
        grants = [store.query(f"multiprocess-{i}")["grant"] for i in range(4)]
        assert len({g["attempt_id"] for g in grants}) == 4
        assert len({e["episode_key"] for g in grants for e in g["episodes"]}) == 20


def test_owner_published_snapshot_lost_and_accepted_restart(tmp_path, monkeypatch):
    store = setup_queue(tmp_path, 1)
    with Controller(store) as c:
        c.register_slot(slot())
        monkeypatch.setattr(c, "_save", lambda: (_ for _ in ()).throw(RuntimeError("snapshot lost")))
        with pytest.raises(RuntimeError, match="snapshot lost"):
            c.claim("durable-owner", "slot-0")
    with Controller(store) as c:
        g = c.claim("durable-owner", "slot-0")["grant"]
        assert c.accept_terminal(terminal(store, g, g["episodes"][0], "timeout"))
        before = c.summary()
    with Controller(store) as c:
        assert c.summary() == before
        c.close_assignment(g["attempt_id"], sim_stopped=True)
        assert c.summary()["compute_status"] == "COMPUTE_DONE"
        assert c.claim("finished", "slot-0")["status"] == "COMPUTE_DONE"


def test_799_normal_plus_exhausted_error_stops(tmp_path):
    store = setup_queue(tmp_path, 800, chunks=800)
    with Controller(store) as c:
        c.register_slot(slot())
        g = c.claim("original800", "slot-0")["grant"]
        for e in g["episodes"][:-1]:
            assert c.accept_terminal(terminal(store, g, e, "fail"))
        final = g["episodes"][-1]
        store.write_attempt(g, "infra", {**final, "retryable": True, "uncertain_terminal": False}, "error")
        c.begin_recovery("first")
        c.recover([proof(slot())])
        c.register_slot(slot(1))
        g = c.claim("retry800", "slot-1")["grant"]
        assert len(g["episodes"]) == 1
        store.write_attempt(g, "infra", {**final, "retryable": True, "uncertain_terminal": False}, "error")
        c.begin_recovery("second")
        result = c.recover([proof(slot(1))])
        assert result["execution_coverage"] == 799
        assert result["compute_status"] == "STOPPED_ERROR"
        assert result["score_status"] == "INVALID"
        assert result["U"] == 0
        assert result["K"] == 1


def test_late_terminal_during_recovery_then_handback_no_retry(tmp_path):
    store = setup_queue(tmp_path, 2)
    with Controller(store) as c:
        c.register_slot(slot())
        g = c.claim("drain", "slot-0")["grant"]
        c.begin_recovery("normal drain")
        late = terminal(store, g, g["episodes"][0], "fail")
        assert not c.accept_terminal(late)
        store.write_attempt(g, "handback", g["episodes"][1], "not-started")
        c.recover([proof(slot())])
        assert c.summary()["execution_coverage"] == 1
        assert c.state["retries"] == {} and c.summary()["U"] == 0
        c.register_slot(slot(1))
        assert c.claim("remaining", "slot-1")["grant"]["episodes"] == [g["episodes"][1]]


def test_readonly_summary_does_not_write_snapshot(tmp_path):
    store = setup_queue(tmp_path)
    before = (store.root / "queue/state.json").read_bytes()
    with Controller(store, readonly=True) as c:
        c.summary()
    assert (store.root / "queue/state.json").read_bytes() == before


def test_authorize_slot_spool_checks_pool_ledger(tmp_path):
    store = setup_queue(tmp_path)
    s = slot()
    member = {**s, "state": "STARTING", "launch_id": "launch-a", "run_identity_sha256": "a" * 64}
    publish(store.root / "pool/ledger.json", {"state": "ACTIVE", "pool_revision": 3,
                                              "slots": {s["slot_instance"]: member}})
    with Controller(store) as c:
        request_slot = {k: s[k] for k in ("slot_instance", "worker_uuid", "job_id", "job_incarnation")}
        store.request("bad-launch", "authorize_slot", {"slot": request_slot, "launch_id": "wrong"})
        store.request("launch", "authorize_slot", {"slot": request_slot, "launch_id": "launch-a"})
        c.process_requests()
        assert store.query("bad-launch")["status"] == "REJECTED"
        assert store.query("launch") == {"status": "AUTHORIZED", "pool_revision": 3}
        store.request("register", "register_slot", {"slot": s})
        c.process_requests()
        assert store.query("register")["status"] == "REGISTERED"


def test_global_contract_failure_blocks_unstarted_episodes(tmp_path):
    store = setup_queue(tmp_path)
    with Controller(store) as c:
        c.register_slot(slot())
        store.request("fatal", "worker_error", {"slot_instance": "slot-0", "error": "metadata mismatch",
                      "retryable": False, "uncertain_terminal": False})
        c.process_requests()
        assert store.query("fatal")["status"] == "RECOVERING"
        result = c.recover([proof(slot())])
        assert result["compute_status"] == "STOPPED_ERROR"
        assert result["score_status"] == "INVALID"
        assert result["U"] == 0 and result["execution_errors"] == 5


def test_pool_membership_freeze_and_retiring_claim(tmp_path):
    store = setup_queue(tmp_path)
    s = slot()
    member = {**s, "state": "RUNNING", "retiring": True}
    unlaunched = {**slot(1, "UNKNOWN"), "step_id": None}
    publish(store.root / "pool/ledger.json", {"state": "ACTIVE", "pool_revision": 4,
                 "slots": {"slot-0": member, "slot-1": unlaunched}})
    with Controller(store) as c:
        c.register_slot(s)
        assert c.claim("retiring", "slot-0")["status"] == "DRAIN"
        freeze = c.begin_recovery("ownerless")
        assert freeze["pool_revision"] == 4
        assert set(freeze["slots"]) == {"slot-0", "slot-1"}
        with pytest.raises(ValueError, match="全部"):
            c.recover([proof(s)])
        c.recover([proof(s), proof(unlaunched)])
        assert c.state["phase"] == "RUNNING"


def test_empty_artifact_error_is_retained_and_worker_infra_recovers(tmp_path):
    store = setup_queue(tmp_path, 1)
    with Controller(store) as c:
        c.register_slot(slot())
        g = c.claim("claim", "slot-0")["grant"]
        e = g["episodes"][0]
        c.accept_terminal(terminal(store, g, e))
        receipt = store.write_attempt(g, "artifact", {**e, "files": [], "artifact_error": True}, "error")
        assert not c.accept_artifact(receipt)
        assert list((store.root / "artifacts").glob("*.json"))
        store.request("worker-infra", "worker_infra", {"slot_instance": "slot-0", "error": "close failed"})
        c.process_requests()
        assert store.query("worker-infra")["status"] == "RECOVERING"
        result = c.recover([proof(slot())])
        assert result["compute_status"] == "COMPUTE_DONE"
        assert result["score_status"] == "PASS"
        assert result["artifact_status"] == "BLOCKED_ARTIFACTS"


def test_normal_drain_two_terminal_three_handback(tmp_path):
    store = setup_queue(tmp_path)
    with Controller(store) as c:
        c.register_slot(slot())
        g = c.claim("drain", "slot-0")["grant"]
        for e in g["episodes"][:2]:
            c.accept_terminal(terminal(store, g, e))
        for e in g["episodes"][2:]:
            store.write_attempt(g, "handback", {**e, "reason": "drain"}, "handback-" + e["episode_key"])
        c.close_assignment(g["attempt_id"], sim_stopped=True)
        assert c.state["retries"] == {} and c.summary()["U"] == 0
        assert c.state["assignments"][g["attempt_id"]]["closed"]
        c.register_slot(slot(1))
        retry = c.claim("remaining", "slot-1")["grant"]
        assert retry["episodes"] == g["episodes"][2:]
        assert retry["generation"] == 1
    with Controller(store) as c:
        assert c.state["assignments"][g["attempt_id"]]["closed"]
        assert c.summary()["U"] == 0


def test_drain_refuses_started_handback(tmp_path):
    store = setup_queue(tmp_path, 1)
    with Controller(store) as c:
        c.register_slot(slot())
        g = c.claim("started", "slot-0")["grant"]
        e = g["episodes"][0]
        store.write_attempt(g, "started", e, "start")
        store.write_attempt(g, "handback", {**e, "reason": "drain"}, "return")
        with pytest.raises(ValueError, match="已开始"):
            c.close_assignment(g["attempt_id"], sim_stopped=True)


def test_raw_client_receipts_recovered_before_generation(tmp_path):
    store = setup_queue(tmp_path)
    with Controller(store) as c:
        c.register_slot(slot())
        g = c.claim("raw", "slot-0")["grant"]
        client = Path(g["attempt_dir"]) / "client"
        for e in g["episodes"][:3]:
            identity = {**e, "run_identity_sha256": g["run_identity_sha256"],
                        "chunk_id": g["chunk_id"], "episode_nonce": "nonce"}
            publish(client / f"{e['episode_key']}.started.json", {**identity, "record_type": "started"})
            if e["episode_id"] < 2:
                publish(client / f"{e['episode_key']}.terminal.json", {**identity, "record_type": "terminal",
                    "outcome": "success", "actions_executed": 5, "rpc_seq": 2})
        # 模拟G一份原始收据也未来得及包裹便退出resident。
        assert not list((Path(g["attempt_dir"]) / "started").glob("*.json"))
        store.request("step-died", "worker_infra", {"slot_instance": "slot-0", "error": "verified step exit"})
        c.process_requests()
        assert c.summary()["compute_status"] == "WAIT"
        store.request("recover", "recover", {"proofs": [proof(slot())]})
        c.process_requests()
        response = store.query("recover")
        assert response["status"] == "RECOVERED"
        assert response["summary"]["execution_coverage"] == 2
        assert response["summary"]["U"] == 1
        assert c.state["retries"] == {"Task:2": 1}
        assert c.state["generations"][g["chunk_id"]] == 1


def test_raw_client_identity_mismatch_quarantined(tmp_path):
    store = setup_queue(tmp_path, 1)
    with Controller(store) as c:
        c.register_slot(slot())
        g = c.claim("raw", "slot-0")["grant"]
        e = g["episodes"][0]
        payload = {**e, "env_seed": 999, "run_identity_sha256": "a" * 64, "chunk_id": g["chunk_id"],
                   "episode_nonce": "nonce", "record_type": "terminal", "outcome": "success",
                   "actions_executed": 1, "rpc_seq": 1}
        publish(Path(g["attempt_dir"]) / "client" / "Task:0.terminal.json", payload)
        c.begin_recovery("verified death")
        c.recover([proof(slot())])
        assert not c._accepted()
        assert list((store.root / "quarantine").glob("*.json"))


def test_worker_died_before_register_uses_verified_pool_step(tmp_path):
    store = setup_queue(tmp_path)
    member = {**slot(1, "STARTING"), "launch_id": "launch-1"}
    publish(store.root / "pool/ledger.json", {"state": "ACTIVE", "pool_revision": 1,
                                              "slots": {"slot-1": member}})
    with Controller(store) as c:
        store.request("unregistered-step-dead", "worker_infra", {"slot_instance": "slot-1", "error": "step failed"})
        c.process_requests()
        assert set(c.state["freeze"]["slots"]) == {"slot-1"}
        c.recover([proof(member)])
        assert c.state["slots"]["slot-1"]["state"] == "TERMINATED"


def test_controller_cli_enters_recovery_and_publishes_summary(tmp_path):
    store = setup_queue(tmp_path, 1)
    with Controller(store) as c:
        c.register_slot(slot())
        c.claim("abrupt-step-exit", "slot-0")
    proofs = tmp_path / "proofs.json"
    proofs.write_text(json.dumps([proof(slot())]))
    script = Path(__file__).resolve().parents[2] / "scripts/evaluation/official_queue.py"
    result = subprocess.run([sys.executable, str(script), "controller", "--run-root", str(store.root),
                             "--recovery-proofs", str(proofs), "--once"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    summary = read_json(store.root / "queue/summary.json")
    assert summary["compute_status"] == "READY"
    with Controller(store) as c:
        assert c.state["assignments"][next(iter(c.state["assignments"]))]["closed"]


def test_freeze_excludes_retired_pool_slots(tmp_path):
    store = setup_queue(tmp_path)
    retired = slot(1, "RETIRED")
    terminal_hold = slot(2, "TERMINAL")
    publish(store.root / "pool/ledger.json", {"state": "ACTIVE", "pool_revision": 1,
                 "slots": {"slot-1": retired, "slot-2": terminal_hold, "slot-0": slot()}})
    with Controller(store) as c:
        freeze = c.begin_recovery("new fault")
        assert set(freeze["slots"]) == {"slot-0"}


def test_unregistered_starting_step_late_binding_requires_exact_launch(tmp_path):
    store = setup_queue(tmp_path)
    member = {**slot(1, "STARTING"), "step_id": None, "launch_id": "launch-1"}
    publish(store.root / "pool/ledger.json", {"state": "ACTIVE", "pool_revision": 1,
                                              "slots": {"slot-1": member}})
    with Controller(store) as c:
        c.begin_recovery("starting death")
        p = {**proof(member), "step_id": "0", "launch_id": "launch-1", "slurm_job_name": "wrong"}
        with pytest.raises(ValueError, match="正向绑定"):
            c.recover([p])
        p["slurm_job_name"] = "official-resident-launch-1"
        c.recover([p])
        assert c.state["slots"]["slot-1"]["step_id"] == "0"
        assert c.state["slots"]["slot-1"]["step_binding"]["launch_id"] == "launch-1"


def test_validation_smoke_and_shards_keep_separate_scopes(tmp_path):
    results = []
    for name, count, workers in (("smoke", 1, 1), ("shards", 10, 2)):
        store = setup_queue(tmp_path / name, count, mode="validation")
        with Controller(store) as c:
            for i in range(workers):
                c.register_slot(slot(i))
            for i in range(workers):
                grant = c.claim(f"{name}-{i}", f"slot-{i}")["grant"]
                for episode in grant["episodes"]:
                    assert c.accept_terminal(terminal(store, grant, episode, "fail"))
                c.close_assignment(grant["attempt_id"], sim_stopped=True)
            summary = c.summary()
            assert summary["purpose"] == "validation"
            assert summary["execution_coverage"] == count
            assert summary["compute_status"] == "COMPUTE_DONE"
            assert c.claim("no-success-rerun", "slot-0")["status"] == "COMPUTE_DONE"
            results.append(summary["execution_coverage"])
    assert results == [1, 10]
