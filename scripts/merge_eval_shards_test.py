"""merge_eval_shards 的 CPU 单测（合成记录，不需要仿真）：uv run pytest scripts/merge_eval_shards_test.py"""
import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("merge_eval_shards", Path(__file__).with_name("merge_eval_shards.py"))
mes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mes)

CAPS = mes.FALLBACK_TIER_MAX_STEPS


def _manifest():
    """2 任务 × (xhard0 2 局 + xhard1 2 局)，两轮各两片、每片 2 局。"""
    rows, i = [], 0
    for task in ("BinFill", "StopCube"):
        for tier, ep0 in (("xhard0", 0), ("xhard1", 12)):
            for j in range(2):
                seed = 1000 + i
                rows.append(dict(task=task, episode=ep0 + j, tier=tier, seed=seed,
                                 candidate=None if tier == "xhard0" else j,
                                 source_episode=3 + 4 * j if tier == "xhard0" else None,
                                 round=1 + (i % 2), shard=(i // 2) % 2))
                i += 1
    return rows


def _binding(tier, **kw):
    if tier == "xhard0":
        b = dict(available=True, mode="export", spec_kind="native-parity/1", spec_sha256=None, value_points=9,
                 injected_mismatch=0, recorded_drift=0, recorded_max_abs=0.0, unused=0)
    else:
        b = dict(available=True, mode="replay", spec_kind="native/1", spec_sha256="abc", value_points=9,
                 injected_mismatch=0, recorded_drift=1, recorded_max_abs=1e-7, unused=0, layout_drift=0)
    b.update(kw)
    return b


def _mme(row, **kw):
    """MME-VLA episodes.jsonl 行（身份在 identity 子字典里）。"""
    rec = dict(task=row["task"], episode=row["episode"], seed=row["seed"],
               identity=dict(episode=row["episode"], tier=row["tier"], seed=row["seed"], candidate=row["candidate"]),
               tier=row["tier"], binding_class="export" if row["tier"] == "xhard0" else "replay",
               max_steps=CAPS[row["tier"]], steps=100, status="success", spec_binding=_binding(row["tier"]))
    rec.update(kw)
    return rec


def _smv(row, **kw):
    """SimpleMemVLA results-r*-shard*of*.jsonl 行（顶层字段、无 max_steps）。"""
    rec = dict(task=row["task"], episode=row["episode"], tier=row["tier"], seed=row["seed"], candidate=row["candidate"],
               source_episode=row["source_episode"], status="fail", steps=CAPS[row["tier"]] + 1,
               spec_binding=_binding(row["tier"]), binding_class="export" if row["tier"] == "xhard0" else "replay")
    rec.update(kw)
    return rec


def _write(tmp, name, recs):
    p = tmp / name
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return str(p)


def _setup(tmp, mutate=None):
    rows = _manifest()
    man = _write(tmp, "ids.jsonl", rows)
    files = {1: [], 2: []}
    for r in (1, 2):
        for s in (0, 1):
            sel = [x for x in rows if x["round"] == r and x["shard"] == s]
            make = _mme if r == 1 else _smv
            recs = [make(x) for x in sel]
            if mutate:
                recs = mutate(r, s, sel, recs)
            files[r].append(_write(tmp, f"r{r}s{s}.jsonl", recs))
    return man, files


def _run(monkeypatch, capsys, man, files):
    argv = ["merge", "--policy", "t", "--identities", man, "--round1", *files[1], "--round2", *files[2]]
    monkeypatch.setattr(sys, "argv", argv)
    rc = mes.main()
    out = capsys.readouterr().out
    lines = {l.split("=", 1)[0]: l for l in out.splitlines() if l.startswith("EVAL_") and "=" in l.split()[0]}
    return rc, out, lines


def test_all_pass(tmp_path, monkeypatch, capsys):
    man, files = _setup(tmp_path)
    rc, out, lines = _run(monkeypatch, capsys, man, files)
    assert rc == 0, out
    for k in ("EVAL_ROUND1", "EVAL_ROUND2", "EVAL_IDENTITY_SET", "EVAL_BINDING", "EVAL_TIER_CAP"):
        assert lines[k].startswith(f"{k}=PASS"), out
    assert "replay=4/4 export=4/4" in lines["EVAL_BINDING"]
    assert "episodes=8/8" in lines["EVAL_IDENTITY_SET"]
    assert "inferred_from_steps=4" in lines["EVAL_TIER_CAP"]
    assert "tier=xhard0 episodes=4 success=2" in out


def test_retry_then_normal_passes(tmp_path, monkeypatch, capsys):
    def mutate(r, s, sel, recs):
        if r == 1 and s == 0:
            recs = [dict(recs[0], status="error")] + recs  # 先 error 后重评，最后一条为准
        return recs
    man, files = _setup(tmp_path, mutate)
    rc, out, lines = _run(monkeypatch, capsys, man, files)
    assert rc == 0, out
    assert "retries=1" in lines["EVAL_ROUND1"]


def test_missing_and_error_left_fail(tmp_path, monkeypatch, capsys):
    def mutate(r, s, sel, recs):
        if r == 1 and s == 0:
            return recs[:1]
        if r == 2 and s == 1:
            return [dict(recs[0], status="error")] + recs[1:]
        return recs
    man, files = _setup(tmp_path, mutate)
    rc, out, lines = _run(monkeypatch, capsys, man, files)
    assert rc == 1
    assert lines["EVAL_ROUND1"].startswith("EVAL_ROUND1=FAIL") and "missing=1" in lines["EVAL_ROUND1"]
    assert "shard_mismatch=2" in lines["EVAL_ROUND1"]  # 缺局的文件对不上任何一片，且那一片没人认领
    assert lines["EVAL_ROUND2"].startswith("EVAL_ROUND2=FAIL") and "error_left=1" in lines["EVAL_ROUND2"]
    assert lines["EVAL_IDENTITY_SET"].startswith("EVAL_IDENTITY_SET=FAIL")


def test_binding_classification_fail(tmp_path, monkeypatch, capsys):
    def mutate(r, s, sel, recs):
        out = []
        for x, rec in zip(sel, recs):
            if r == 1 and x["tier"] == "xhard0" and s == 0:
                rec = dict(rec, spec_binding=_binding("xhard0", spec_kind="native/1"))  # export 却不是 parity 规格
            if r == 2 and x["tier"] == "xhard1" and s == 1:
                rec = dict(rec, spec_binding=_binding("xhard1", injected_mismatch=2))
            out.append(rec)
        return out
    man, files = _setup(tmp_path, mutate)
    rc, out, lines = _run(monkeypatch, capsys, man, files)
    assert rc == 1
    b = lines["EVAL_BINDING"]
    assert b.startswith("EVAL_BINDING=FAIL"), b
    assert "injected_mismatch=4" in b and "export=2/4" in b and "replay=2/4" in b, b
    assert lines["EVAL_ROUND1"].startswith("EVAL_ROUND1=PASS")


def test_binding_class_field_mismatch(tmp_path, monkeypatch, capsys):
    def mutate(r, s, sel, recs):
        return [dict(recs[0], binding_class="replay" if recs[0]["binding_class"] == "export" else "export")] + recs[1:]
    man, files = _setup(tmp_path, mutate)
    rc, out, lines = _run(monkeypatch, capsys, man, files)
    assert rc == 1 and "class_mismatch=4" in lines["EVAL_BINDING"]


def test_tier_cap_fail(tmp_path, monkeypatch, capsys):
    def mutate(r, s, sel, recs):
        if r == 1 and s == 1:
            return [dict(recs[0], max_steps=999)] + recs[1:]
        if r == 2 and s == 0:
            return [dict(recs[0], steps=5000)] + recs[1:]
        return recs
    man, files = _setup(tmp_path, mutate)
    rc, out, lines = _run(monkeypatch, capsys, man, files)
    assert rc == 1 and lines["EVAL_TIER_CAP"].startswith("EVAL_TIER_CAP=FAIL") and "mismatch=2" in lines["EVAL_TIER_CAP"]


def test_round1_only(tmp_path, monkeypatch, capsys):
    man, files = _setup(tmp_path)
    monkeypatch.setattr(sys, "argv", ["merge", "--policy", "t", "--identities", man, "--round1", *files[1]])
    assert mes.main() == 0
    out = capsys.readouterr().out
    assert out.startswith("EVAL_ROUND1=PASS") and "EVAL_BINDING" not in out
