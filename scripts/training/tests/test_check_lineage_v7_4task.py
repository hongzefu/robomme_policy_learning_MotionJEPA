"""check_lineage_v7_4task.py 的 CPU 单测：临时目录造最小假数据，正常 PASS、篡改各环即 FAIL。"""

import copy
import hashlib
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/training"))
import check_lineage_v7_4task as lin  # noqa: E402

TASKS = ("BinFill", "RouteStick", "VideoRepick", "VideoUnmaskSwap")
PER_TASK = 400  # 1600 / 4，与真实口径一致
REV = "604f16da36d6b6d175884df8fb687dc08e0a36eb"
MAN = "d" * 64


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))
    return p


def _row(task, i):
    diff = ("easy", "medium", "hard", "xhard")[i % 4]
    h = hashlib.sha256(f"{task}-{i}".encode()).hexdigest()
    return {"task": task, "difficulty": diff, "episode": i // 4, "seed": 1000 + i, "role": "primary",
            "member": f"record_dataset_{task}_{diff}/hdf5_files/{task}_ep{i // 4}_seed{1000 + i}.h5",
            "h5_sha256": h, "h5_bytes": 100 + i, "timestep_count": 50 + i}


def build(tmp, mutate_scope=None):
    data_root = tmp / "enc-data"
    rows = {t: [_row(t, i) for i in range(PER_TASK)] for t in TASKS}
    entries = []
    for t in TASKS:
        for r in rows[t]:
            for suffix in ("demo", "exec"):  # 每成员两条变体，模拟 2×去重
                entries.append(dict(r, suffix=suffix, name=f"{t}_{suffix}"))
    scope = {"revision": REV, "episodes": 1600, "entries": entries}
    if mutate_scope:
        mutate_scope(scope)
    _dump(data_root / "control/scope.json", scope)
    _dump(data_root / "control/source_pin.json",
          {"revision": REV, "manifest_sha256": MAN, "status": "PASS", "episodes": 1600})
    ckpt_dir = tmp / "ckpt"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "checkpoint_epoch_72.pt"
    ckpt.write_bytes(b"fake-ckpt")
    (ckpt_dir / "config.yaml").write_text(f"data:\n  data_root: {data_root}\n")
    recs = tmp / "records"
    maps_sha = {}
    for t in TASKS:
        maps_sha[t] = _sha(_dump(recs / f"record_dataset_{t}_episode_map.json",
                                 [dict(r, h5_sha256_local=r["h5_sha256"]) for r in rows[t]]))
    files = {f"record_dataset_{t}.h5": hashlib.sha256(t.encode()).hexdigest() for t in TASKS}
    merge = {"source_pin": {"revision": REV, "manifest_sha256": MAN}, "selection": {"tasks": list(TASKS)},
             "maps_sha256": maps_sha, "files": files, "tasks": {t: {"episodes": PER_TASK} for t in TASKS},
             "episodes": 1600, "source_sha256_checked": 1600, "mismatches": 0}
    _dump(recs / "MERGE_DONE.json", merge)
    vla = _dump(tmp / "vla/meta/input_manifest.json",
                {"files": {k: {"sha256": v, "size": 1} for k, v in files.items()}})
    return {"encoder_ckpt": str(ckpt), "expected_ckpt_sha256": _sha(ckpt), "vla_input_manifest": str(vla),
            "merge_done": str(recs / "MERGE_DONE.json"), "episode_maps": str(recs)}


def argv(a, out=None):
    v = ["--encoder-ckpt", a["encoder_ckpt"], "--expected-ckpt-sha256", a["expected_ckpt_sha256"],
         "--vla-input-manifest", a["vla_input_manifest"], "--merge-done", a["merge_done"],
         "--episode-maps", a["episode_maps"]]
    return v + (["--out", str(out)] if out else [])


def test_pass(tmp_path, capsys):
    a = build(tmp_path)
    out = tmp_path / "ev.json"
    assert lin.main(argv(a, out)) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("DATA_LINEAGE=PASS mode=v7-4ring encoder_ckpt=")
    assert "encoder_source_pin=dddddddd" in line and "files=4 members=1600 mismatches=0" in line
    ev = json.loads(out.read_text())
    assert all(ev["rings"][k]["status"] == "PASS" for k in ev["rings"])
    assert ev["rings"]["4_members"]["encoder_entries"] == 3200


def test_seed_tamper_fails_ring4(tmp_path, capsys):
    def bump(scope):
        for e in scope["entries"]:
            if e["task"] == "RouteStick" and e["episode"] == 5 and e["difficulty"] == "easy":
                e["seed"] += 1  # 两条变体同改，保持 entries 内部自洽，只与 episode_map 不等
    a = build(tmp_path, bump)
    assert lin.main(argv(a)) == 3
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("DATA_LINEAGE=FAIL mode=v7-4ring ring=4")


def test_missing_member_fails_ring4(tmp_path, capsys):
    def drop(scope):
        scope["entries"] = [e for e in scope["entries"] if not (e["task"] == "BinFill" and e["episode"] == 0
                                                                and e["difficulty"] == "easy")]
    assert lin.main(argv(build(tmp_path, drop))) == 3
    assert "ring=4" in capsys.readouterr().out


def test_inconsistent_variants_fail_ring4(tmp_path, capsys):
    def split(scope):
        scope["entries"][0]["h5_bytes"] += 1  # 同一成员的两条变体字段不一致
    assert lin.main(argv(build(tmp_path, split))) == 3
    assert "ring=4" in capsys.readouterr().out


@pytest.mark.parametrize("ring,edit", [
    (0, lambda a, t: a.update(expected_ckpt_sha256="0" * 64)),
    (1, lambda a, t: _dump(pathlib.Path(a["vla_input_manifest"]),
                           {"files": {"record_dataset_BinFill.h5": {"sha256": "1" * 64}}})),
    (2, lambda a, t: pathlib.Path(a["episode_maps"], "record_dataset_VideoRepick_episode_map.json").write_text("[]")),
    (3, lambda a, t: _dump(pathlib.Path(t, "enc-data/control/source_pin.json"),
                           {"revision": REV, "manifest_sha256": "e" * 64, "status": "PASS", "episodes": 1600})),
])
def test_other_rings_fail(tmp_path, capsys, ring, edit):
    a = build(tmp_path)
    edit(a, tmp_path)
    assert lin.main(argv(a)) == 3
    assert f"ring={ring}" in capsys.readouterr().out.strip().splitlines()[-1]


def test_relative_path_rejected(tmp_path, capsys):
    a = build(tmp_path)
    a = copy.deepcopy(a)
    a["merge_done"] = "records/MERGE_DONE.json"
    assert lin.main(argv(a)) == 3
    assert "DATA_LINEAGE=FAIL" in capsys.readouterr().out
