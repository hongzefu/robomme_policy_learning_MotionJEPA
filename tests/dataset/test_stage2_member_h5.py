"""1114 计划 S2 契约②（单集 H5）与契约①（运行项目选择）的 CPU 测试。"""
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

import h5py
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAPS = ROOT / "docs/dataset-build-doc/4task-v2-1600ep-604f16da/records"


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("_t_member_h5", "scripts/dataset/stage2_member_h5.py")
P = load("_t_stage2_project", "scripts/dataset/stage2_project.py")


def frame(member_seed, t):
    return np.full((256, 256, 3), (member_seed * 7 + t) % 251, dtype=np.uint8)


def write_member_h5(path, raw_episode, timesteps, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group(f"episode_{raw_episode}")
        for t in range(timesteps):
            group.create_dataset(f"timestep_{t}/obs/front_rgb", data=frame(seed, t))
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size


def make_maps(tmp_path, entries):
    """entries: [(task, new_idx, difficulty, raw, seed_tag, timesteps)] → 写单集 H5 与 episode_map。"""
    root, maps = tmp_path / "raw-h5", tmp_path / "maps"
    maps.mkdir(parents=True)
    by_task = {}
    for task, new_idx, difficulty, raw, seed_tag, timesteps in entries:
        member = f"record_dataset_{task}_{difficulty}/hdf5_files/{task}_ep{raw}_seed{seed_tag}.h5"
        sha, size = write_member_h5(root / member, raw, timesteps, new_idx)
        by_task.setdefault(task, []).append(dict(task=task, difficulty=difficulty, episode=raw, seed=seed_tag,
                                                 member=member, h5_sha256=sha, h5_bytes=size, new_idx=new_idx,
                                                 src_group=f"episode_{raw}", h5_sha256_local=sha))
    for task, rows in by_task.items():
        (maps / f"record_dataset_{task}_episode_map.json").write_text(json.dumps(rows))
    expected = {task: M.sha256_file(maps / f"record_dataset_{task}_episode_map.json") for task in by_task}
    return root, maps, expected


def test_real_git_map_binfill_new_idx_100_is_raw_102():
    """真实 git 内 map：BinFill new_idx=100 → BinFill_ep102_seed14200.h5 的组 episode_102。"""
    merge = MAPS / "MERGE_DONE.json"
    records = M.load_member_records(MAPS, expected_map_sha256=M.expected_maps_from_merge_done(merge))
    assert len(records) == 1600
    record = records["BinFill", 100]
    assert record.member.endswith("BinFill_ep102_seed14200.h5") and record.group == "episode_102"
    assert record.raw_episode == 102 and record.episode == 100
    assert record.tar_name == record.member.split("/")[0] + ".h5.tar.xz"
    assert sum(r.raw_episode != r.episode for r in records.values()) == 1390


def test_raw_episode_differs_from_episode_reads_correct_group(tmp_path):
    root, maps, expected = make_maps(tmp_path, [("BinFill", 100, "medium", 102, 14200, 5)])
    record = M.load_member_records(maps, expected_map_sha256=expected)["BinFill", 100]
    path = M.ensure_members([record], roots=[root])[record.member]
    frames = M.read_frames(record, path, [0, 3, 3])
    assert frames.shape == (3, 256, 256, 3) and frames.dtype == np.uint8
    assert (frames[1] == frame(100, 3)).all()
    entry = M.source_entry(record, path)
    assert entry["kind"] == "member_h5" and entry["raw_episode"] == 102 and entry["episode"] == 100
    assert M.record_from_source(entry) == record


@pytest.mark.parametrize("fault", ["group", "sha", "bytes", "map_sha"])
def test_contract_faults_fail(tmp_path, fault):
    root, maps, expected = make_maps(tmp_path, [("BinFill", 100, "medium", 102, 14200, 3)])
    if fault == "map_sha":
        expected["BinFill"] = "0" * 64
        with pytest.raises(ValueError, match="map sha"):
            M.load_member_records(maps, expected_map_sha256=expected)
        return
    record = M.load_member_records(maps, expected_map_sha256=expected)["BinFill", 100]
    path = root / record.member
    if fault == "group":
        # 组名错配：文件里只有 episode_100（统一编号），不是 raw_episode 组 episode_102。
        path.unlink()
        write_member_h5(path, 100, 3, 100)
        rows = json.loads((maps / "record_dataset_BinFill_episode_map.json").read_text())
        rows[0].update(h5_sha256_local=M.sha256_file(path), h5_bytes=path.stat().st_size)
        (maps / "record_dataset_BinFill_episode_map.json").write_text(json.dumps(rows))
        record = M.load_member_records(maps)["BinFill", 100]
        with pytest.raises(ValueError, match="组集合"):
            M.read_frames(record, M.verify_member_file(record, path), [0])
        return
    if fault == "sha":
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with pytest.raises(ValueError, match="sha256"):
            M.verify_member_file(record, path)
    else:
        with open(path, "ab") as stream:
            stream.write(b"x")
        with pytest.raises(ValueError, match="字节数"):
            M.verify_member_file(record, path)


def test_cross_difficulty_same_basename_resolved_by_full_member(tmp_path):
    root, maps, expected = make_maps(tmp_path, [("RouteStick", 5, "easy", 7, 900, 3),
                                                ("RouteStick", 105, "hard", 7, 900, 4)])
    records = M.load_member_records(maps, expected_map_sha256=expected)
    a, b = records["RouteStick", 5], records["RouteStick", 105]
    assert pathlib.PurePosixPath(a.member).name == pathlib.PurePosixPath(b.member).name and a.member != b.member
    paths = M.ensure_members([a, b], roots=[root])
    assert paths[a.member] != paths[b.member]
    assert (M.read_frames(b, paths[b.member], [3])[0] == frame(105, 3)).all()
    # 只按完整路径找：把 a 挪成同 basename 的别处文件也不会被误认。
    moved = root / "elsewhere" / pathlib.PurePosixPath(a.member).name
    moved.parent.mkdir()
    (root / a.member).rename(moved)
    assert M.find_member(a, [root]) is None


def test_missing_members_extracted_from_tar_once(tmp_path):
    src, maps, expected = make_maps(tmp_path, [("BinFill", 1, "easy", 1, 4100, 2), ("BinFill", 2, "easy", 2, 4200, 2)])
    records = M.load_member_records(maps, expected_map_sha256=expected)
    tar_root = tmp_path / "tar"
    tar_root.mkdir()
    subprocess.run(["tar", "-cJf", str(tar_root / "record_dataset_BinFill_easy.h5.tar.xz"), "-C", str(src),
                    "record_dataset_BinFill_easy"], check=True)
    empty, extract = tmp_path / "raw-empty", tmp_path / "raw-h5-new"
    empty.mkdir()
    paths = M.ensure_members(list(records.values()), roots=[empty], tar_root=tar_root, extract_root=extract)
    assert sorted(paths) == sorted(r.member for r in records.values())
    assert all(str(p).startswith(str(extract)) for p in paths.values())
    with pytest.raises(FileExistsError):
        M.extract_members(list(records.values()), tar_root, extract)
    with pytest.raises(FileNotFoundError, match="tar"):
        M.ensure_members(list(records.values()), roots=[empty])


def test_inventory_lists_extra_members(tmp_path):
    root, maps, expected = make_maps(tmp_path, [("BinFill", 1, "easy", 1, 4100, 2), ("BinFill", 2, "easy", 2, 4200, 2),
                                                ("BinFill", 3, "hard", 3, 4300, 2)])
    records = M.load_member_records(maps, expected_map_sha256=expected)
    stray = root / "record_dataset_BinFill_hard/hdf5_files/BinFill_ep9_seed1.h5"
    write_member_h5(stray, 9, 1, 9)
    g4 = {records["BinFill", 1].member}
    result = M.inventory(root, records, g4, jobs=2)
    assert result["files"] == 4 and result["sha_ok"] == 3 and result["g4prep_found"] == 1
    extra = {row["member"]: row["verdict"] for row in result["extra"]}
    assert extra[str(stray.relative_to(root))] == "只读不用" and len(extra) == 3
    assert extra[records["BinFill", 2].member] == "核过可复用"


# ─────────────────────────────── 契约①：运行项目选择

def test_project_registry_and_inference():
    old, new = P.get_project("wan_stage2"), P.get_project("wan_stage2_rgbstate")
    assert old.root.name == "wan_stage2" and new.root.name == "wan_stage2_rgbstate"
    assert old.venv.name == "wan-stage2" and new.venv.name == "wan-stage2-rgbstate"
    assert old.sidecar_dir == new.sidecar_dir == ROOT / "scripts/dataset/wan_stage2"
    assert P.resolve_project_name(arch=P.RGBSTATE_ARCH) == "wan_stage2_rgbstate"
    assert P.resolve_project_name(arch=P.THREE_DECODER_ARCH) == "wan_stage2"
    assert P.resolve_project_name("wan_stage2_rgbstate", arch=P.RGBSTATE_ARCH) == "wan_stage2_rgbstate"
    with pytest.raises(ValueError, match="不符"):
        P.resolve_project_name("wan_stage2", arch=P.RGBSTATE_ARCH)        # 新 store 配旧项目
    with pytest.raises(ValueError, match="不符"):
        P.resolve_project_name("wan_stage2_rgbstate", arch=P.THREE_DECODER_ARCH)
    with pytest.raises(ValueError, match="必须显式"):
        P.resolve_project_name(None, require_explicit=True)
    with pytest.raises(ValueError, match="未知"):
        P.resolve_project_name("wan_stage3")
    with pytest.raises(ValueError, match="推断"):
        P.resolve_project_name(arch="wan-latent-v10-st")


def test_both_projects_pin_and_lock_are_self_consistent():
    for name in P.PROJECT_NAMES:
        project = P.get_project(name)
        pin = P.check_project_files(project, M.sha256_file)
        assert set(pin) == {"mj_repo_commit", "export_source_sha256", "encoder_source_sha256",
                            "vae_adapter_sha256", "dependency_lock_sha256"}
        assert P.project_of_source_pin(project.source_pin) == name
    new = P.get_project("wan_stage2_rgbstate").read_pin()
    old = P.get_project("wan_stage2").read_pin()
    assert new["mj_repo_commit"] != old["mj_repo_commit"]
    assert "rgbstate" in (P.get_project("wan_stage2_rgbstate").root / "pyproject.toml").read_text()
    with pytest.raises(ValueError, match="不属于"):
        P.project_of_source_pin(MAPS / "MERGE_DONE.json")


@pytest.mark.parametrize("script,extra", [
    ("scripts/dataset/check_stage2_validation_inputs.py",
     ["--manifest", "m", "--latents", "l", "--out", "o", "--episode-maps", "e"]),
    ("scripts/dataset/check_motion_xpath.py",
     ["--data-root", "d", "--latents", "l", "--episode-maps", "e", "--merge-done", "m", "--ckpt", "c",
      "--expected-sha256", "a" * 64, "--out", "o"]),
])
def test_entries_without_store_or_ckpt_require_explicit_project(script, extra, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / script), *extra], capture_output=True, text=True,
                            timeout=60, cwd=tmp_path)
    assert result.returncode == 2 and "--stage2-project" in result.stderr
