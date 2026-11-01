"""证据重定根 MMEVLA_STAGE2_EVIDENCE_ROOT（1031 计划 K11）：未设时路径不变、设后只换目录、SHA 不符即 FAIL。

重定根消费者清单（grep 全仓 evidence 路径读取处后逐个核对）：
- scripts/training/train.py::write_stage2_motion_snapshot：G0 父目录读 online_restore_inputs.*，逐角色读证据 → 本文件覆盖；
- scripts/training/check_orig16_stage2_launch.py::check_dataset 内 validate_stage2_evidence 与 --verify-evidence-root
  子命令 → 本文件与 test_orig16_gl_contract.py 覆盖；
- scripts/training/check_stage2_checkpoint.py 与 motion_client 在 checkpoint 根内按快照相对路径读，不经 AWS 绝对路径，无需重定根。
"""
import json
import pathlib
import shutil
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/training"))
from test_stage2_snapshot import snapshot, train  # noqa: E402,F401  复用合成快照夹具
import test_check_orig16_stage2_launch as base  # noqa: E402
from mme_vla_suite.datastore import motion_store as ms  # noqa: E402

launch = base.launch
ENV = "MMEVLA_STAGE2_EVIDENCE_ROOT"


def test_unset_path_unchanged(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    for value in ("/scratch/hongze/x/v1-store/validation/orig16-stage2/G0.json", "motion_evidence/G2.json"):
        assert train.stage2_evidence_path(value) == pathlib.Path(value)


def test_set_only_replaces_parent(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path / "rebased"))
    got = train.stage2_evidence_path("/scratch/hongze/x/v1-store/validation/orig16-stage2/SOURCE_PIN.json")
    assert got == tmp_path / "rebased/SOURCE_PIN.json"


def move_evidence(tmp_path, provenance):
    """把快照夹具的证据与在线输入整体挪到新目录（平铺），原位置不再存在。"""
    rebased = tmp_path / "turbo/validation/orig16-stage2"
    rebased.mkdir(parents=True)
    for item in provenance["stage2_compat"]["evidence"].values():
        shutil.move(item["path"], rebased / pathlib.Path(item["path"]).name)
    for name in ("online_restore_inputs.json", "online_restore_inputs.npz"):
        shutil.move(tmp_path / name, rebased / name)
    return rebased


def test_snapshot_reads_rebased_root(snapshot, monkeypatch, tmp_path):
    root, registered, provenance = snapshot
    before = json.dumps(provenance, sort_keys=True)
    rebased = move_evidence(tmp_path, provenance)
    monkeypatch.setenv(ENV, str(rebased))
    ready = train.write_stage2_motion_snapshot(root, provenance, registered_root=registered)
    assert json.dumps(provenance, sort_keys=True) == before, "调用方 provenance 不得被改写"
    saved = json.loads((root / "motion_provenance.json").read_text())
    for label, item in saved["stage2_compat"]["evidence"].items():
        assert item["path"] == f"motion_evidence/{label}.json"
        name = pathlib.Path(provenance["stage2_compat"]["evidence"][label]["path"]).name
        assert (root / item["path"]).read_bytes() == (rebased / name).read_bytes()
        assert item["sha256"] == provenance["stage2_compat"]["evidence"][label]["sha256"]
    for name in ("online_restore_inputs.json", "online_restore_inputs.npz"):
        assert (root / name).read_bytes() == (rebased / name).read_bytes()
    assert all(ms.sha256_file(root / name) == sha for name, sha in ready["files"].items())


def test_snapshot_without_env_keeps_original_paths(snapshot, monkeypatch, tmp_path):
    root, registered, provenance = snapshot
    move_evidence(tmp_path, provenance)
    monkeypatch.delenv(ENV, raising=False)
    with pytest.raises(OSError):
        train.write_stage2_motion_snapshot(root, provenance, registered_root=registered)
    assert not (root / "motion_snapshot.ready.json").exists()


@pytest.mark.parametrize("target", ["G2", "online_restore_inputs.npz"])
def test_snapshot_rebased_sha_mismatch_fails(snapshot, monkeypatch, tmp_path, target):
    root, registered, provenance = snapshot
    rebased = move_evidence(tmp_path, provenance)
    name = target if "." in target else pathlib.Path(provenance["stage2_compat"]["evidence"][target]["path"]).name
    (rebased / name).write_bytes((rebased / name).read_bytes() + b" ")
    monkeypatch.setenv(ENV, str(rebased))
    with pytest.raises(ValueError):
        train.write_stage2_motion_snapshot(root, provenance, registered_root=registered)
    assert not (root / "motion_snapshot.ready.json").exists()


def test_checker_rebase_and_verify(snapshot, tmp_path):
    _, _, provenance = snapshot
    rebased = move_evidence(tmp_path, provenance)
    copy = launch.rebase_provenance(provenance, rebased)
    assert copy is not provenance
    for label, item in copy["stage2_compat"]["evidence"].items():
        old = provenance["stage2_compat"]["evidence"][label]
        assert pathlib.Path(item["path"]) == rebased / pathlib.Path(old["path"]).name
        assert item["sha256"] == old["sha256"] and pathlib.Path(old["path"]).parent == tmp_path
    files, bad = launch.verify_evidence_root(rebased, provenance)
    assert (files, bad) == (7, [])
    (rebased / "G3.json").write_bytes(b"{}")
    (rebased / "online_restore_inputs.npz").unlink()
    assert launch.verify_evidence_root(rebased, provenance)[1] == ["G3", "online_restore_inputs.npz"]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """真实打包的缓存证据小库：motion 表 provenance 里证据是绝对路径。"""
    fx = base.make_fixture(tmp_path_factory.mktemp("o16-rebase"))
    motion = fx.parent / "motion-stage2-1024"
    provenance = ms.MotionMeta.load(motion).provenance
    rebased = fx.root / "v1-store/validation/orig16-stage2"
    rebased.mkdir(parents=True)
    for item in provenance["stage2_compat"]["evidence"].values():
        shutil.copyfile(item["path"], rebased / pathlib.Path(item["path"]).name)
    npz = rebased / "online_restore_inputs.npz"
    npz.write_bytes(b"synthetic")
    (rebased / "online_restore_inputs.json").write_text(json.dumps(
        {"payload": {"path": npz.name, "sha256": ms.sha256_file(npz)}}))
    return motion, rebased


def test_verify_evidence_root_cli_pass(store, capsys):
    motion, rebased = store
    assert launch.main(["--verify-evidence-root", str(rebased), "--store", str(motion)]) == 0
    assert "EVIDENCE_REBASE=PASS files=7 sha_mismatch=0" in capsys.readouterr().out


def test_verify_evidence_root_cli_sha_mismatch(store, capsys):
    motion, rebased = store
    g0 = ms.MotionMeta.load(motion).provenance["stage2_compat"]["evidence"]["G0"]["path"]
    path = rebased / pathlib.Path(g0).name
    before = path.read_bytes()
    path.write_bytes(before + b" ")
    try:
        assert launch.main(["--verify-evidence-root", str(rebased), "--store", str(motion)]) == 1
        assert "EVIDENCE_REBASE=FAIL files=7 sha_mismatch=1" in capsys.readouterr().out
    finally:
        path.write_bytes(before)
