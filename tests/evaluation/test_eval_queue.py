"""eval_queue 计划/调度纯函数与 export_episodes 的离线测试（不起任何进程、不用 GPU）。"""
import csv
import json
import os
from pathlib import Path

import pytest

from scripts.evaluation import eval_queue as q
from scripts.evaluation import export_episodes as ex
from scripts.evaluation.manifest import make_manifest

REPO = Path(__file__).resolve().parents[2]
V1 = Path(os.environ.get("MMEVLA_V1_STORE") or "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store")
RUN1107 = V1 / "evaluation/ctx-wan-timepool1024-3seed-1107"


def queue():
    return q.load_queue(q.DEFAULT_QUEUE)


def test_plan_has_33_items_in_fixed_order():
    items = q.plan_items(queue())
    names = [it["name"] for it in items]
    assert len(names) == 33
    assert names[:9] == [f"fs1024-{s}-seed{d}" for s in (79999, 70000, 60000) for d in (42, 7, 0)]
    rest = ["motionmem-55000", "motionmem-50000", "wanpool-55000", "wanpool-50000",
            "noxyz-70000", "noxyz-60000", "nostate-70000", "nostate-60000"]
    assert names[9:] == [f"{r}-seed{d}" for d in (42, 7, 0) for r in rest]
    assert len(q.distinct_ckpts(items)) == 11


def test_optional_tail_switch_appends_nine():
    data = queue()
    data["optional_tail_enabled"] = True
    names = [it["name"] for it in q.plan_items(data)]
    assert len(names) == 42
    assert names[33:36] == ["motionmem-59999-seed42", "noxyz-79999-seed42", "nostate-79999-seed42"]


def test_queue_fields_match_runbook():
    data = queue()
    assert (data["per_gpu"], data["mem_fraction"], data["gpus"], data["tmux_prefix"]) == (4, "0.18", list(range(8)), "ev3-")
    assert data["manifest_sha256"].startswith("991c3496")
    kinds = {m: s["model_kind"] for m, s in data["models"].items()}
    assert kinds == {"fs1024": "history", "motionmem": "stage2_pool4_concat", "wanpool": "wan_fixedpool1024",
                     "noxyz": "stage2_pool4_concat", "nostate": "stage2_pool4_concat"}
    for spec in data["models"].values():
        assert (q.YAML_DIR / spec["history_yaml"]).is_file()


def test_expected_kind_matches_policy_config_rule():
    from omegaconf import OmegaConf
    assert q.expected_kind(OmegaConf.create({"budget": 1024})) == "history"
    assert q.expected_kind(OmegaConf.create({"motion": {"enabled": False, "encoder_kind": "stage2_pool4_concat"}})) == "history"
    assert q.expected_kind(OmegaConf.create({"motion": {"enabled": True, "encoder_kind": "stage2_pool4_concat"}})) == "stage2_pool4_concat"
    assert q.expected_kind(OmegaConf.create({"motion": {"enabled": True, "encoder_kind": "wan_fixedpool1024"}})) == "wan_fixedpool1024"


def _episodes(n):
    tasks = ("BinFill", "StopCube", "PickXtimes")
    return [{"task_name": tasks[i % 3], "episode_id": i // 3, "env_seed": i, "difficulty": "easy"} for i in range(n)]


def test_unclaimed_chunks_counts_free_ready_chunks(tmp_path):
    manifest = make_manifest(_episodes(15), official=False)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    keys = [e["episode_key"] for e in manifest["episodes"]]
    chunks = manifest["chunks"]
    assert q.unclaimed_chunks(tmp_path, {}, {"assignments": {}}) == 3
    state = {"assignments": {"a": {"closed": False, "grant": {"worker_id": "w", "chunk_id": chunks[0]["chunk_id"]}}}}
    summary = {"missing": keys[:10], "errors": {}}
    assert q.unclaimed_chunks(tmp_path, summary, state) == 1       # 块0被占、块2已全部accepted
    summary = {"missing": keys[:10], "errors": {k: {} for k in keys[5:10]}}
    assert q.unclaimed_chunks(tmp_path, summary, state) == 0       # 块1全部进入errors不再可领


def test_smoke_manifest_single_episode_is_valid():
    small = make_manifest(_episodes(1), official=False)
    assert len(small["chunks"]) == 1 and small["chunks"][0]["episode_keys"] == ["BinFill:0"]


def test_execution_copy_guard_rejects_primary_checkout(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(q, "REPO", tmp_path)
    with pytest.raises(SystemExit):
        q.require_execution_copy()


@pytest.mark.skipif(not RUN1107.is_dir(), reason="本机无 1107 评估输出")
def test_export_1107_matches_published_counts(tmp_path):
    roots = [RUN1107 / f"seed{s}/run800" for s in (42, 7, 0)]
    rows, infos = ex.export(roots, tmp_path)
    assert [i["success"] for i in infos] == [291, 280, 280]           # 36.38% / 35.00% / 35.00%
    assert [i["model_seed"] for i in infos] == [42, 7, 0]
    assert all(i["accepted"] == 800 and i["model_kind"] == "wan_fixedpool1024" for i in infos)
    with (tmp_path / "episodes.csv").open() as stream:
        table = list(csv.DictReader(stream))
    assert list(table[0]) == list(ex.COLUMNS) and len(table) == 2400
    for seed, want in ((42, 291), (7, 280), (0, 280)):
        mine = [r for r in table if int(r["model_seed"]) == seed]
        assert sum(int(r["success"]) for r in mine) == want
        assert {(r["task"], int(r["episode_id"])) for r in mine} == {(t, i) for t in q_tasks() for i in range(50)}
    with (tmp_path / "per_task.csv").open() as stream:
        tasks = list(csv.DictReader(stream))
    assert len(tasks) == 48 and sum(int(r["success"]) for r in tasks) == 851


def q_tasks():
    from scripts.evaluation.manifest import TASKS
    return TASKS


class FakeTmux:
    """替身：记录 start_session，alive 查字典，不起任何真实会话。"""

    def __init__(self):
        self.live, self.started = {}, []

    def start(self, name, argv, log, env=None):
        log.parent.mkdir(parents=True, exist_ok=True)
        self.live[name] = True
        self.started.append((name, [str(a) for a in argv], {k: str(v) for k, v in (env or {}).items()}))

    def exit(self, run, wid, code=0):
        worker = run.workers[wid]
        self.live[worker["session"]] = False
        worker["log"].write_text(f"EXIT_CODE={code}\n")


def _sched(tmp_path, monkeypatch, n_items=2):
    data = queue()
    root = tmp_path / "train/fs"
    ckpt = root / "100"
    (ckpt / "params").mkdir(parents=True)
    (ckpt / "assets").mkdir()
    (ckpt / "_CHECKPOINT_METADATA").write_text("{}")
    data["models"]["fs1024"]["train_root"] = str(root)
    data["out_root"] = str(tmp_path / "out")
    data["worker_start_gap_seconds"] = 0
    data["items"] = [{"model": "fs1024", "step": 100, "seed": s} for s in (42, 7)][:n_items]
    data["optional_tail_enabled"] = False
    path = q.smoke_record_path(data, "fs1024", 100)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"verdict": "PASS", "eval_code_commit": "c" * 40}))
    tmux = FakeTmux()
    monkeypatch.setattr(q, "start_session", tmux.start)
    monkeypatch.setattr(q, "alive", lambda name: tmux.live.get(name, False))
    monkeypatch.setattr(q, "kill_session", lambda name: tmux.live.__setitem__(name, False))
    monkeypatch.setattr(q, "gpu_used_mib", lambda: {g: 0 for g in range(8)})
    manifest = make_manifest(_episodes(15), official=False)
    sched = q.Scheduler(data, q.plan_items(data), manifest, "c" * 40)
    return sched, tmux, manifest


def _write(run, summary, state):
    (run.root / "queue").mkdir(parents=True, exist_ok=True)
    (run.root / "queue/summary.json").write_text(json.dumps(summary))
    (run.root / "queue/state.json").write_text(json.dumps(state))


def test_scheduler_hands_gpus_to_next_item_and_finishes(tmp_path, monkeypatch, capsys):
    sched, tmux, manifest = _sched(tmp_path, monkeypatch)
    keys = [e["episode_key"] for e in manifest["episodes"]]
    chunk0 = manifest["chunks"][0]["chunk_id"]
    sched.tick()
    a = sched.active[0]
    assert a.label == "fs1024-100-seed42" and a.gpus == set(range(8))
    for _ in range(40):
        sched.tick()
    assert len(a.workers) == 32 and not sched.pending_starts
    worker_cmd = [s for s in tmux.started if "--worker-id" in s[1]][0]
    assert worker_cmd[2]["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.18" and "--local" in worker_cmd[1]
    # 只剩块0未完成且被 g3 上的一个 worker 占着：其余 worker 被 drain、退出，7 张卡交给下一项
    busy = next(w for w, info in a.workers.items() if info["gpu"] == 3)
    state = {"workers": {busy: {"job_id": "local", "step_id": "999999999", "state": "RUNNING"}},
             "assignments": {"x": {"closed": False, "grant": {"worker_id": busy, "chunk_id": chunk0}}},
             "retries": {"BinFill:0": 1}, "errors": {}}
    _write(a, {"compute_status": "WAIT", "missing": keys[:5], "errors": {}, "completed": 10,
               "outcomes": {"success": 4}}, state)
    sched.tick()
    drained = {p.name for p in (a.root / "queue/drain").iterdir()}
    assert len(drained) == 31
    for wid in a.workers:
        if wid != busy:
            tmux.exit(a, wid, 0)
    sched.tick()
    assert a.gpus == {3}
    b = sched.active[1]
    assert b.label == "fs1024-100-seed7" and b.gpus == set(range(8)) - {3}
    # 块0完成：A 结束、写 EVAL_DONE，GPU3 交给 B
    tmux.exit(a, busy, 0)
    state["assignments"]["x"]["closed"] = True
    _write(a, {"compute_status": "COMPUTE_DONE", "missing": [], "errors": {}, "completed": 15,
               "outcomes": {"success": 6}}, state)
    sched.tick()
    sched.tick()
    out = capsys.readouterr().out
    assert "EVAL_DONE item=fs1024-100-seed42 success=6/800 errors=0 retries=1 status=COMPUTE_DONE" in out
    assert "EVAL_START item=fs1024-100-seed7" in out
    assert b.gpus == set(range(8)) and sched.active == [b]


def test_scheduler_blocks_item_without_smoke(tmp_path, monkeypatch, capsys):
    sched, tmux, _ = _sched(tmp_path, monkeypatch, n_items=1)
    q.smoke_record_path(sched.queue, "fs1024", 100).write_text(json.dumps({"verdict": "FAIL", "eval_code_commit": "c" * 40}))
    assert sched.tick() is False
    assert "ITEM_BLOCKED item=fs1024-100-seed42 reason=smoke_fail" in capsys.readouterr().out
    assert sched.blocked == 1 and not tmux.started
