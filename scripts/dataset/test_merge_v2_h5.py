"""单 episode 合并的来源、内容、结构与续跑守卫；只使用临时合成数据。"""

import json
from pathlib import Path
import sys
import threading

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_v2_h5 as merge


MIXED8_LEVELS = ("easy", "medium", "hard", "xhard12", "xhard14", "xhard16", "xhard18", "xhard20")


def write_episode(path, episode):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        group = f.create_group(f"episode_{episode}")
        group["setup/task_goal"] = np.asarray(["目标甲", "目标乙"], dtype=h5py.string_dtype())
        for step in range(2):
            prefix = f"timestep_{step}"
            group[f"{prefix}/info/is_video_demo"] = step == 0
            group[f"{prefix}/info/is_completed"] = False
            group[f"{prefix}/info/is_subgoal_boundary"] = False
            group[f"{prefix}/uint8"] = np.asarray([1, 2], dtype=np.uint8)
            group[f"{prefix}/float32"] = np.asarray([1, np.nan], dtype=np.float32)
            group[f"{prefix}/bytes"] = np.bytes_(b"abc")
            group[f"{prefix}/int64"] = np.int64(99)


def build_source(tmp_path, monkeypatch, form):
    """合成源：hf 形制（repo_id + counts，期望由注入的 summary.cells 给出）或 mixed8 local-benchmark 形制。"""
    monkeypatch.setattr(merge, "V1_STORE", tmp_path / "v1-store")
    snapshot, control, extracted = (tmp_path / n for n in ("snapshot", "control", "extracted"))
    for path in (snapshot, control, extracted):
        path.mkdir()
    # mixed8：Alpha 有 8 档中的 5 档（含 xhard），Beta 只有 3 档；逐集平铺在 extracted/ 下。
    layout = {"Alpha": ("easy", "medium", "hard"), "Beta": ("easy", "medium", "hard")} if form == "hf" else \
        {"Alpha": ("xhard20", "easy", "xhard12", "hard", "medium"), "Beta": ("hard", "easy", "medium")}
    rows = []
    for task, levels in layout.items():
        for i, difficulty in enumerate(levels):
            episode = 7 + i * 11
            if form == "hf":
                member = f"record_dataset_{task}_{difficulty}/hdf5_files/{task}_ep{episode}_seed{i}.h5"
            else:
                member = f"{task}_ep{episode}_seed{i}.h5"
            path = extracted / member
            write_episode(path, episode)
            rows.append({"task": task, "difficulty": difficulty, "episode": episode, "seed": i, "role": "primary",
                         "member": member, "h5_sha256": merge.sha256_file(path), "h5_bytes": path.stat().st_size,
                         "timestep_count": 2})
    cells = {task: dict.fromkeys(levels, 1) for task, levels in layout.items()}
    if form == "hf":
        rows.extend([{"role": "spare", "member": "spare/unused.h5"}, {"role": "smoke", "member": "smoke/unused.h5"}])
        manifest = {"repo_id": "测试/合并", "counts": {"primary": 6, "spare": 1, "smoke": 1,
                    "videos": 9, "archives": 8, "files": 17}, "summary": {"cells": cells}, "episodes": rows}
    else:
        manifest = {"archives": [], "revision": "b" * 40, "stray_files": [],
                    "summary": {"cells": cells, "per_task": {t: len(v) for t, v in layout.items()},
                                "partial": False, "missing": []}, "episodes": rows}
    merge.write_json(snapshot / "MANIFEST.json", manifest)
    digest = merge.sha256_file(snapshot / "MANIFEST.json")
    if form == "hf":
        pin = {"repo_id": manifest["repo_id"], "revision": "a" * 40, "manifest_sha256": digest}
    else:
        # 生产只认登记的 mixed8 摘要；测试进程内把合成 MANIFEST 的摘要加入登记集合。
        monkeypatch.setattr(merge, "LOCAL_BENCHMARK_MANIFESTS", frozenset({digest}))
        pin = {"manifest_sha256": digest, "results_sha256": "c" * 64, "revision": "b" * 40,
               "run_parameters_sha256": "d" * 64, "source_kind": "local-benchmark"}
    merge.write_json(control / "source.json", pin)
    out = merge.V1_STORE / "raw-h5/test"

    def run(command, *extra):
        args = [command, "--out", str(out), "--extracted", str(extracted), "--procs", "1"]
        if command == "plan":
            args += ["--snapshot", str(snapshot), "--control", str(control)]
        return merge.main(args + list(extra))

    return run, out, extracted, snapshot, rows


@pytest.fixture
def mini(tmp_path, monkeypatch):
    return build_source(tmp_path, monkeypatch, "hf")


@pytest.fixture
def mixed8(tmp_path, monkeypatch):
    return build_source(tmp_path, monkeypatch, "mixed8")


def built(mini):
    run, out, *_ = mini
    assert run("plan") == 0
    assert run("merge", "--check-source-sha256") == 0
    return run, out


def test_complete_pipeline_and_resume(mini):
    run, out = built(mini)
    assert run("verify", "--level", "sample") == 0
    assert not (out / "MERGE_DONE.json").exists()
    assert run("verify", "--level", "full", "--procs", "2") == 0
    done = merge.read_json(out / "MERGE_DONE.json")
    assert done["episodes"] == done["source_sha256_checked"] == 6
    assert done["role"] == {"primary": 6}
    assert done["timesteps"] == 12
    assert len(done["files"]) == 2
    assert run("merge") == 1
    assert run("merge", "--resume", "--check-source-sha256") == 0
    assert run("plan") == 0


@pytest.mark.parametrize("kind", ["uint8", "float32", "nan", "bool", "bytes", "object", "int64"])
def test_seven_corruptions(mini, kind):
    run, out = built(mini)
    with h5py.File(out / "record_dataset_Alpha.h5", "r+") as f:
        group = f["episode_0"]
        if kind == "object":
            group["setup/task_goal"][1] = "已篡改"
        elif kind == "bool":
            group["timestep_0/info/is_video_demo"][()] = False
        elif kind == "bytes":
            del group["timestep_0/bytes"]
            group["timestep_0/bytes"] = np.bytes_(b"abcd")
        elif kind == "nan":
            group["timestep_0/float32"][0] = np.nan
        elif kind == "float32":
            group["timestep_0/float32"][0] += np.float32(1e-7)
        elif kind == "uint8":
            group["timestep_0/uint8"][0] += 1
        else:
            group["timestep_0/int64"][()] += 1
    assert run("verify", "--level", "full") == 1
    assert not (out / "MERGE_DONE.json").exists()
    assert run("merge", "--resume") == 1


@pytest.mark.parametrize("kind", ["attrs", "soft", "external", "alias", "extra", "dtype", "root_attrs"])
def test_structure_guards(mini, kind):
    run, out = built(mini)
    with h5py.File(out / "record_dataset_Alpha.h5", "r+") as f:
        group = f["episode_0"]
        if kind == "attrs":
            group.attrs["新增"] = 1
        elif kind == "root_attrs":
            f.attrs["新增"] = 1
        elif kind in ("soft", "external", "alias"):
            del group["timestep_0/uint8"]
            if kind == "soft":
                group["timestep_0/uint8"] = h5py.SoftLink("/episode_1/timestep_0/uint8")
            elif kind == "external":
                group["timestep_0/uint8"] = h5py.ExternalLink("other.h5", "/data")
            else:
                group["timestep_0/uint8"] = group["timestep_1/uint8"]
        elif kind == "dtype":
            del group["timestep_0/uint8"]
            group["timestep_0/uint8"] = np.asarray([1, 2], dtype=np.int64)
        else:
            group["extra"] = 1
    assert run("verify") == 1


def test_partial_resume(mini):
    run, out = built(mini)
    path = out / "record_dataset_Alpha.h5"
    tmp = path.with_name(path.name + ".tmp")
    path.rename(tmp)
    with h5py.File(tmp, "r+") as f:
        del f["episode_2"]
    assert run("merge", "--resume", "--check-source-sha256") == 0
    assert not tmp.exists()
    assert run("verify") == 0


def test_map_tamper_and_source_sha(mini):
    run, out, extracted, _, source_rows = mini
    assert run("plan") == 0
    p = merge.map_path(out, "Alpha")
    rows = merge.read_json(p)
    rows[0]["seed"] += 1
    merge.write_json(p, rows)
    assert run("merge") == 1
    rows[0]["seed"] -= 1
    merge.write_json(p, rows)
    src = extracted / source_rows[0]["member"]
    with h5py.File(src, "r+") as f:
        f[f"episode_{source_rows[0]['episode']}/timestep_0/uint8"][0] += 1
    assert run("merge", "--check-source-sha256") == 1


@pytest.mark.parametrize("kind", ["demo", "completed", "boundary", "gap", "wrong_group"])
def test_source_contract(mini, kind):
    run, _, extracted, _, rows = mini
    row = rows[0]
    with h5py.File(extracted / row["member"], "r+") as f:
        group = f[f"episode_{row['episode']}"]
        if kind == "demo":
            group["timestep_0/info/is_video_demo"][()] = False
            group["timestep_1/info/is_video_demo"][()] = True
        elif kind == "completed":
            group["timestep_0/info/is_completed"][()] = True
        elif kind == "boundary":
            del group["timestep_0/info/is_subgoal_boundary"]
        elif kind == "gap":
            group.move("timestep_1", "timestep_2")
        else:
            f.move(group.name, "episode_0")
    assert run("plan") == 1


def test_selection_and_path_guards(mini):
    run, out, _, snapshot, _ = mini
    manifest = merge.read_json(snapshot / "MANIFEST.json")
    assert run("plan", "--per-group", "1") == 0
    manifest["counts"]["spare"] = 2
    with pytest.raises(ValueError, match="role"):
        merge.select_primary(manifest, ["Alpha"])
    manifest["counts"]["spare"] = 1
    manifest["episodes"][0]["difficulty"] = "unknown"
    with pytest.raises(ValueError, match="未知难度"):
        merge.select_primary(manifest, ["Alpha"])
    with pytest.raises(ValueError, match="输出必须"):
        merge.output_path(out.parents[3] / "outside")
    link = out.parent / "linked"
    link.symlink_to(out, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        merge.output_path(link)


def test_force_keeps_provenance(mini):
    run, out = built(mini)
    assert run("verify") == 0
    old = {p.name: p.read_bytes() for p in (merge.map_path(out, "Alpha"), out / "source_pin.json", out / "MERGE_DONE.json")}
    assert run("merge", "--force", "--check-source-sha256") == 0
    assert all((out / name).read_bytes() == content for name, content in old.items())
    assert run("verify") == 0


def test_finalize_sha_parallel_and_failures(tmp_path, monkeypatch):
    import finalize_checks as checks

    names = [f"{i}.h5" for i in range(4)]
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    manifest = {"canonical_order": names}
    reference = {name: {"size": (tmp_path / name).stat().st_size, "sha256": checks.sha256_file(str(tmp_path / name))} for name in names}
    pin = tmp_path / "input.json"
    pin.write_text(json.dumps({"files": reference}))
    original = checks.sha256_file
    barrier = threading.Barrier(4, timeout=10)

    def concurrent_hash(path):
        barrier.wait()
        return original(path)

    monkeypatch.setattr(checks, "sha256_file", concurrent_hash)
    assert checks.check_inputs(manifest, str(tmp_path), str(pin), "sha256") == []
    monkeypatch.setattr(checks, "sha256_file", original)
    (tmp_path / names[0]).write_bytes(b"xxxx")
    (tmp_path / names[1]).unlink()
    (tmp_path / names[2]).write_bytes(b"longer")
    del reference[names[3]]
    pin.write_text(json.dumps({"files": reference}))
    errors = checks.check_inputs(manifest, str(tmp_path), str(pin), "sha256")
    assert len(errors) == 4
    assert any("sha256 不符" in error for error in errors)
    assert any("缺输入 H5" in error for error in errors)
    assert any("字节数不符" in error for error in errors)
    assert any("输入清单里没有" in error for error in errors)


def test_mixed8_local_benchmark_pipeline(mixed8):
    run, out, *_ = mixed8
    assert run("plan") == 0
    plan = merge.read_json(out / "merge_plan.json")
    assert plan["tasks"] == ["Alpha", "Beta"]
    assert plan["source_pin"]["source_kind"] == "local-benchmark" and "repo_id" not in plan["source_pin"]
    order = [row["difficulty"] for row in merge.read_json(merge.map_path(out, "Alpha"))]
    assert order == ["easy", "medium", "hard", "xhard12", "xhard20"]
    assert run("merge", "--check-source-sha256") == 0
    assert run("verify", "--level", "full") == 0
    done = merge.read_json(out / "MERGE_DONE.json")
    assert done["episodes"] == done["source_sha256_checked"] == 8
    assert done["source_pin"]["source_kind"] == "local-benchmark"
    assert run("merge") == 1
    assert run("merge", "--resume") == 0


def test_mixed8_subset_precheck(mixed8, capsys):
    run, out, extracted, _, rows = mixed8
    beta = next(row for row in rows if row["task"] == "Beta")
    (extracted / beta["member"]).unlink()
    assert run("plan") == 1
    assert run("plan", "--tasks", "Beta") == 1
    assert not out.exists() or not (out / "merge_plan.json").exists()
    capsys.readouterr()
    assert run("plan", "--tasks", "Alpha") == 0
    assert "PLAN_OK tasks=1 selected=5 role=primary:5 max_timesteps=2 prechecks=PASS precheck_files=5" in capsys.readouterr().out
    assert set(merge.read_json(out / "task_goal_variants.json")) == {"Alpha"}
    assert not merge.map_path(out, "Beta").exists()
    assert run("merge") == 0
    assert run("verify") == 0
    assert merge.read_json(out / "MERGE_DONE.json")["tasks"] == {"Alpha": {"episodes": 5, "timesteps": 10}}


def test_mixed8_global_count_kept_for_subset(mixed8, monkeypatch):
    run, _, _, snapshot, _ = mixed8
    manifest = merge.read_json(snapshot / "MANIFEST.json")
    manifest["summary"]["cells"]["Beta"]["hard"] = 2
    manifest["summary"]["per_task"]["Beta"] = 4
    with pytest.raises(ValueError, match="summary.cells"):
        merge.select_primary(manifest, ["Alpha"])
    manifest["summary"]["cells"]["Beta"]["hard"] = 1
    manifest["summary"]["per_task"]["Beta"] = 4
    with pytest.raises(ValueError, match="per_task"):
        merge.select_primary(manifest, ["Alpha"])
    manifest["summary"]["per_task"]["Beta"] = 3
    manifest["summary"]["partial"] = True
    with pytest.raises(ValueError, match="不完整"):
        merge.select_primary(manifest, ["Alpha"])
    manifest["summary"]["partial"] = False
    manifest["episodes"][0]["difficulty"] = "xhard"
    with pytest.raises(ValueError, match="summary.cells"):
        merge.select_primary(manifest, ["Alpha"])
    manifest["episodes"][0]["difficulty"] = "xhard13"
    with pytest.raises(ValueError, match="未知难度"):
        merge.select_primary(manifest, ["Alpha"])
    with pytest.raises(ValueError, match="任务列表"):
        merge.select_primary(merge.read_json(snapshot / "MANIFEST.json"), ["Gamma"])


@pytest.mark.parametrize("kind", ["unregistered", "repo_id", "revision", "manifest_revision", "unknown_kind"])
def test_local_benchmark_source_guards(mixed8, monkeypatch, kind):
    run, _, _, snapshot, _ = mixed8
    control = snapshot.parent / "control"
    pin = merge.read_json(control / "source.json")
    if kind == "unregistered":
        monkeypatch.setattr(merge, "LOCAL_BENCHMARK_MANIFESTS", frozenset({"e" * 64}))
    elif kind == "repo_id":
        pin["repo_id"] = "测试/合并"
    elif kind == "revision":
        pin["revision"] = "main"
    elif kind == "unknown_kind":
        pin["source_kind"] = "hf-bucket"
    else:
        manifest = merge.read_json(snapshot / "MANIFEST.json")
        manifest["revision"] = "f" * 40
        merge.write_json(snapshot / "MANIFEST.json", manifest)
        pin["manifest_sha256"] = merge.sha256_file(snapshot / "MANIFEST.json")
        monkeypatch.setattr(merge, "LOCAL_BENCHMARK_MANIFESTS", frozenset({pin["manifest_sha256"]}))
    merge.write_json(control / "source.json", pin)
    assert run("plan") == 1


def test_legacy_manifest_without_cells():
    # 旧 4 任务源：MANIFEST 无 summary.cells，全集仍按固定 4×400 校验。
    rows = [{"task": task, "difficulty": ("easy", "medium", "hard", "xhard")[i % 4], "episode": i, "seed": i,
             "role": "primary", "member": f"{task}/{i}.h5", "h5_sha256": "0" * 64, "h5_bytes": 1, "timestep_count": 3}
            for task in merge.LEGACY_SOURCE_COUNTS for i in range(400)]
    manifest = {"repo_id": "旧/源", "counts": {"primary": 1600, "spare": 0, "smoke": 0}, "episodes": rows}
    selected = merge.select_primary(manifest, ["BinFill"], per_group=2)
    assert [row["difficulty"] for row in selected["BinFill"]] == ["easy"] * 2 + ["medium"] * 2 + ["hard"] * 2 + ["xhard"] * 2
    manifest["episodes"] = rows[:-1]
    manifest["counts"]["primary"] = 1599
    with pytest.raises(ValueError, match="400"):
        merge.select_primary(manifest, ["BinFill"])
