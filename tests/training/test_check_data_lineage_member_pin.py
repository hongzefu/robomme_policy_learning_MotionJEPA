"""check_data_lineage.py member-pin：四环链、契约③公式、--out 证据与负例（1114 计划 S1）。"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/training/check_data_lineage.py"
SPEC = importlib.util.spec_from_file_location("check_data_lineage_member_pin", SCRIPT)
lineage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lineage)

# 计划文首外部依赖锚点：git 内 4 份 episode_map 的 1600 成员按契约③得 2cb9026f…；
# 保留字面键 h5_sha256_local 得 01027855…（错误口径）。MJ 侧（S2）独立实现须得同值，对拍常量即此值。
MAPS = REPO / "docs/dataset-build-doc/4task-v2-1600ep-604f16da/records"
PIN_1600 = "2cb9026fc874964941ddb8c1cf1c0425a3b8ac94d3bd8e2082779f4e95f5c8bd"
PIN_1600_LITERAL_KEY = "01027855e0980c08da617a927e9f6bb5e6a114a720014b498381c3315dad5ed3"
INPUT_MANIFEST_SHA = "655e64b4fd033aea1ff79a1e5ad47f886a148592496759ed2a667fe167c57d86"


def map_rows():
    rows = []
    for path in sorted(MAPS.glob("record_dataset_*_episode_map.json")):
        rows += json.loads(path.read_text())
    return rows


def test_real_maps_member_pin_matches_anchor_and_literal_key_differs():
    rows = map_rows()
    assert len(rows) == 1600
    records = lineage.canonical_member_records(rows)
    assert set(records[0]) == {"member", "h5_sha256", "h5_bytes"}
    assert lineage.member_pin_sha256(records) == PIN_1600
    assert lineage._literal_key_pin(records) == PIN_1600_LITERAL_KEY != PIN_1600
    # 与 MJ 侧对拍：独立按公式逐字写一遍，必须同值（不 import MJ）。
    ordered = sorted(({"member": r["member"], "h5_sha256": r["h5_sha256_local"], "h5_bytes": r["h5_bytes"]}
                      for r in rows), key=lambda r: r["member"])
    payload = json.dumps(ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == PIN_1600
    assert lineage.sha256_file(MAPS / "input_manifest.json") == INPUT_MANIFEST_SHA


def test_member_pin_order_independent_and_rejects_duplicates():
    records = lineage.canonical_member_records(map_rows()[:5])
    assert lineage.member_pin_sha256(records) == lineage.member_pin_sha256(list(reversed(records)))
    with pytest.raises(ValueError, match="重复"):
        lineage.member_pin_sha256(records + records[:1])


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


@pytest.fixture
def chain(tmp_path):
    """两任务各 2 集的合成四环链；结构与真实 MERGE_DONE／episode_map／source_manifest 一致。"""
    tasks = ["TaskA", "TaskB"]
    maps, merged_files, maps_sha, counts, all_rows = tmp_path / "records", {}, {}, {}, []
    for t, task in enumerate(tasks):
        rows = []
        for e in range(2):
            sha = hashlib.sha256(f"{task}{e}".encode()).hexdigest()
            rows.append({"task": task, "episode": e, "member": f"record_dataset_{task}_easy/hdf5_files/{task}_ep{e}.h5",
                         "h5_sha256": sha, "h5_sha256_local": sha, "h5_bytes": 100 + e, "new_idx": e,
                         "src_group": f"episode_{e}"})
        path = maps / f"record_dataset_{task}_episode_map.json"
        dump(path, rows)
        maps_sha[task] = lineage.sha256_file(path)
        counts[task] = {"episodes": 2, "timesteps": 10}
        merged_files[f"record_dataset_{task}.h5"] = str(t + 1) * 64
        all_rows += rows
    member_records = lineage.canonical_member_records(all_rows)
    pin = lineage.member_pin_sha256(member_records)
    source_pin = {"repo_id": "owner/data", "revision": "a" * 40, "manifest_sha256": "b" * 64}
    dump(maps / "MERGE_DONE.json", {"source_pin": source_pin, "selection": {"tasks": tasks}, "maps_sha256": maps_sha,
                                    "files": merged_files, "tasks": counts, "episodes": 4,
                                    "source_sha256_checked": 4, "mismatches": 0})
    dump(tmp_path / "input_manifest.json", {"raw_dir": "/x", "count": 2, "files": {
        name: {"size": 1000, "sha256": sha} for name, sha in merged_files.items()}})
    root = tmp_path / "encoder-data"
    dump(root / "control/build_spec.json", {"identity": {"source_pin": {
        "manifest_sha256": "b" * 64, "member_pin_sha256": pin, "scope_sha256": "c" * 64}}})
    dump(root / "control/source_manifest.json", {
        "repo_id": "owner/data", "revision": "a" * 40, "manifest_sha256": "b" * 64, "scope_sha256": "c" * 64,
        "member_pin_sha256": pin, "files": [{"member": r["member"], "h5_sha256": r["h5_sha256_local"],
                                             "h5_bytes": r["h5_bytes"], "task": r["task"]} for r in all_rows]})
    ckpt = tmp_path / "reg/checkpoint_epoch_144.pt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"ckpt fixture")
    (ckpt.parent / "config.yaml").write_text(yaml.safe_dump({"data": {"data_root": str(root)}}))
    return {"tmp": tmp_path, "maps": maps, "ckpt": ckpt, "root": root, "pin": pin,
            "input": tmp_path / "input_manifest.json", "merge": maps / "MERGE_DONE.json"}


def cli(c, out=None):
    argv = ["member-pin", "--encoder-ckpt", str(c["ckpt"]), "--expected-ckpt-sha256",
            lineage.sha256_file(c["ckpt"]), "--vla-input-manifest", str(c["input"]),
            "--merge-done", str(c["merge"]), "--episode-maps", str(c["maps"])]
    return argv + (["--out", str(out)] if out else [])


def edit(path, func):
    value = json.loads(path.read_text())
    func(value)
    dump(path, value)


def test_pass_writes_evidence_consistent_with_line(chain, capsys):
    out = chain["tmp"] / "log/lineage.json"
    assert lineage.main(cli(chain, out)) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("DATA_LINEAGE=PASS mode=member-pin ")
    assert f"member_pin={chain['pin']} members=4 h5_files=2 maps=2" in line and line.endswith("mismatches=0")
    evidence = json.loads(out.read_text())
    assert evidence["schema"] == "data-lineage-member-pin-v1" and evidence["status"] == "PASS"
    assert evidence["line"] == line and evidence["member_pin_sha256"] == chain["pin"]
    assert evidence["member_pin_literal_key_sha256"] != chain["pin"]
    assert {k: v["status"] for k, v in evidence["rings"].items()} == {
        "1_merged_h5": "PASS", "2_episode_maps": "PASS", "3_source_manifest": "PASS", "4_members": "PASS"}
    assert evidence["vla_input_manifest"]["sha256"] == lineage.sha256_file(chain["input"])
    assert evidence["encoder"]["checkpoint_sha256"] == lineage.sha256_file(chain["ckpt"])


def _input_sha(c):
    edit(c["input"], lambda v: v["files"]["record_dataset_TaskA.h5"].update(sha256="f" * 64))


def _input_extra(c):
    edit(c["input"], lambda v: (v["files"].update({"record_dataset_X.h5": {"size": 1, "sha256": "e" * 64}}),
                                v.update(count=3)))


def _swap_map(c):
    # 换一份 episode_map：内容合法但不是合并记录登记的那份（行序不同 → sha 不同）。
    path = c["maps"] / "record_dataset_TaskA_episode_map.json"
    dump(path, list(reversed(json.loads(path.read_text()))))


def _map_member_sha(c):
    edit(c["maps"] / "record_dataset_TaskB_episode_map.json", lambda v: v[0].update(h5_sha256_local="0" * 64))


def _map_missing(c):
    (c["maps"] / "record_dataset_TaskB_episode_map.json").unlink()


def _merge_manifest(c):
    edit(c["merge"], lambda v: v["source_pin"].update(manifest_sha256="d" * 64))


def _source_revision(c):
    edit(c["root"] / "control/source_manifest.json", lambda v: v.update(revision="9" * 40))


def _encoder_member_sha(c):
    edit(c["root"] / "control/source_manifest.json", lambda v: v["files"][0].update(h5_sha256="0" * 64))


def _encoder_member_missing(c):
    edit(c["root"] / "control/source_manifest.json", lambda v: v["files"].pop())


def _encoder_pin(c):
    edit(c["root"] / "control/build_spec.json",
         lambda v: v["identity"]["source_pin"].update(member_pin_sha256=PIN_1600_LITERAL_KEY))


def _encoder_pin_literal(c):
    # encoder 侧登记值若按字面键口径算出，也必须判不等。
    rows = []
    for path in sorted(c["maps"].glob("record_dataset_*_episode_map.json")):
        rows += json.loads(path.read_text())
    literal = lineage._literal_key_pin(lineage.canonical_member_records(rows))
    for rel, keys in (("control/build_spec.json", ("identity", "source_pin")), ("control/source_manifest.json", ())):
        def setter(v, keys=keys):
            for key in keys:
                v = v[key]
            v["member_pin_sha256"] = literal
        edit(c["root"] / rel, setter)


def _ckpt(c):
    c["ckpt"].write_bytes(b"other checkpoint")


FAULTS = {"input_sha": (_input_sha, 1), "input_extra": (_input_extra, 1), "swap_map": (_swap_map, 2),
          "map_member_sha": (_map_member_sha, 2), "map_missing": (_map_missing, None),
          "merge_manifest": (_merge_manifest, 3), "source_revision": (_source_revision, 3),
          "encoder_member_sha": (_encoder_member_sha, 4), "encoder_member_missing": (_encoder_member_missing, 4),
          "encoder_pin": (_encoder_pin, 4), "encoder_pin_literal": (_encoder_pin_literal, 4)}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_each_ring_fault_fails_with_evidence(chain, fault, capsys):
    func, ring = FAULTS[fault]
    argv = cli(chain, chain["tmp"] / "lineage.json")  # 先取期望 sha，再施加故障
    func(chain)
    assert lineage.main(argv) == 3
    err = capsys.readouterr().err
    assert "DATA_LINEAGE=FAIL mode=member-pin" in err and "DATA_LINEAGE=PASS" not in err
    evidence = json.loads((chain["tmp"] / "lineage.json").read_text())
    assert evidence["status"] == "FAIL" and evidence["line"] == err.strip().splitlines()[-1]
    assert evidence["failed_ring"] == ring


def test_checkpoint_sha_mismatch_is_ring0(chain, capsys):
    argv = cli(chain)
    _ckpt(chain)
    assert lineage.main(argv) == 3
    assert "ring=0" in capsys.readouterr().err


def test_cli_subprocess_exit_codes(chain):
    ok = subprocess.run([sys.executable, str(SCRIPT), *cli(chain)], capture_output=True, text=True, timeout=60)
    assert ok.returncode == 0 and "DATA_LINEAGE=PASS mode=member-pin" in ok.stdout
    missing = subprocess.run([sys.executable, str(SCRIPT), *cli(chain)[:-2]], capture_output=True, text=True,
                             timeout=60)
    assert missing.returncode == 3 and "DATA_LINEAGE=FAIL" in missing.stderr

