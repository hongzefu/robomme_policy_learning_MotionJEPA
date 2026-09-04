#!/usr/bin/env python3
"""资产锁的守卫回归测试。

沿用 ``scripts/dataset/test_guards.py`` 的哲学：每条用例刻意制造一次失败，断言守卫必须亮红灯。
篡改类用例全部在 ``tmp_path`` 里造假资产树——**绝不动** ``v1-store/models/**``（三个子目录本机是
指向 turbo 只读归档的 symlink）、``v1-store/external/motionjepa/**``（本机唯一副本）、
``v1-store/cache/hf/hub/**``（重下要 508 MB），也绝不「先改真资产再改回来」。

跑法：
  UV_LINK_MODE=copy JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \\
    uv run pytest scripts/assets/test_assets_lock.py -q
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "dataset" / "wan"))

import assets_lock as al  # noqa: E402

LOCK = al.load_lock()


# ── lock 自身 ─────────────────────────────────────────────────────────────────


def test_lock_self_hash_ok() -> None:
    assert LOCK["sha256"] == al.manifest_sha256(LOCK)


def test_lock_self_hash_detects_edit(tmp_path: pathlib.Path) -> None:
    """改 lock 里任意一个值而不改顶层 sha256 → load_lock 必须 fail-loud。"""
    payload = json.loads(al.LOCK_PATH.read_text(encoding="utf-8"))
    payload["assets"]["siglip_params"]["bytes"] += 1
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="自哈希不符"):
        al.load_lock(p)


def test_manifest_sha256_three_implementations_agree() -> None:
    """assets_lock / wan_common / datastore.manifest 三处口径必须逐字同值。"""
    import wan_common as wc

    from mme_vla_suite.datastore import manifest as ms

    payload = {"b": [1, 2, {"z": "中文"}], "a": "x", "sha256": "ignored"}
    got = al.manifest_sha256(payload)
    assert got == wc.manifest_sha256(payload) == ms.manifest_sha256(payload)


def test_headtail_digest_matches_framesamp_store(tmp_path: pathlib.Path) -> None:
    """cheap 档摘要口径必须与 framesamp_store 的同名函数逐字节同值（含 >2 MiB 的非全覆盖分支）。"""
    from mme_vla_suite.datastore.framesamp_store import headtail_digest as ref

    for size in (1024, 3 << 20):
        p = tmp_path / f"f{size}"
        p.write_bytes(bytes(range(256)) * (size // 256))
        assert al.headtail_digest(p) == ref(p)


def test_lock_revisions_are_commit_sha() -> None:
    """HF 来源的 revision 必须是 40 hex commit sha —— 禁 main/master（private repo 自己能 push，会漂）。"""
    for name, entry in LOCK["assets"].items():
        src = entry["source"]
        if src["type"].startswith("hf"):
            assert re.fullmatch(r"[0-9a-f]{40}", src["revision"]), f"{name} revision 非 40 hex"


def test_lock_dest_is_relative_inside_v1_store() -> None:
    for name, entry in LOCK["assets"].items():
        dest = entry["dest"]
        assert not dest.startswith("/"), f"{name} dest 是绝对路径"
        assert dest.startswith("v1-store/"), f"{name} dest 不在 v1-store/ 内（AGENTS.md 第 14 条）"
        assert ".." not in pathlib.PurePosixPath(dest).parts, f"{name} dest 含 .."


def test_lock_matches_source_pin() -> None:
    """lock 记的 MotionJEPA 推理侧 commit 必须与 SOURCE_PIN.json 同值（同一事实只有一处真值）。"""
    pin = json.loads((_REPO_ROOT / "scripts/dataset/wan/SOURCE_PIN.json").read_text(encoding="utf-8"))
    assert LOCK["assets"]["motionjepa_ckpt"]["related"]["mj_repo_commit"] == pin["mj_repo_commit"]


def test_lock_vae_secondary_matches_infer_module() -> None:
    """lock 的 VAE state_dict 指纹必须等于复制件里的 VAE_STATE_SHA256_EXPECTED。

    刻意用正则从源码文本抠常量而**不** import：该模块的 PINNED 要 torch 2.9.0+cu128，
    主 venv 是 2.7.1，import 会被 check_versions 挡下。文件完整性由 SOURCE_PIN 哨兵保证。
    """
    text = (_REPO_ROOT / "scripts/dataset/wan/wan_motion_infer.py").read_text(encoding="utf-8")
    m = re.search(r'VAE_STATE_SHA256_EXPECTED\s*=\s*"([0-9a-f]{64})"', text)
    assert m, "未能在复制件里找到 VAE_STATE_SHA256_EXPECTED"
    secondary = LOCK["assets"]["wan_vae"]["secondary"][0]
    assert secondary["kind"] == "torch_state_dict_sha256"
    assert secondary["value"] == m.group(1)


def test_expected_sha256_unknown_asset_raises() -> None:
    """未知名必须 raise 而不是返回 None —— 否则 `args.x or None` 的静默跳过老路会复活。"""
    with pytest.raises(SystemExit, match="没有资产"):
        al.expected_sha256("nonexistent_asset", LOCK)


def test_expected_sha256_rejects_dir_kind() -> None:
    with pytest.raises(SystemExit, match="不是单文件型"):
        al.expected_sha256("pi05_base", LOCK)


# ── 篡改检测（全部在 tmp_path 造假树）─────────────────────────────────────────


def _fake_tree(tmp_path: pathlib.Path) -> tuple[pathlib.Path, dict]:
    """按真实相对路径造两条资产：一个 3 MiB 文件（跨过 headtail 全覆盖阈值）与一个小文件。"""
    specs = {
        "siglip_params": ("v1-store/models/pi05_vision_encoder/siglip_params.pkl",
                          bytes(range(256)) * ((3 << 20) // 256)),
        "motionjepa_config": ("v1-store/external/motionjepa/wan-v8-filter10-72ep-a/config.yaml",
                              b"motion:\n  dim: 768\n"),
    }
    assets = {}
    for name, (rel, data) in specs.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        size, ht, _ = al.headtail_digest(p)
        assets[name] = {
            "kind": "file", "dest": rel, "bytes": size, "headtail": ht,
            "sha256": hashlib.sha256(data).hexdigest(), "needs_token": False,
            "source": {"type": "hf_file", "repo_id": "x/y", "revision": "0" * 40, "filename": rel},
        }
    lock = {"schema": "assets-lock-v1", "assets": assets}
    lock["sha256"] = al.manifest_sha256(lock)
    return tmp_path, lock


def test_clean_tree_passes(tmp_path: pathlib.Path) -> None:
    root, lock = _fake_tree(tmp_path)
    assert al.verify(lock, root, level="full") == []
    assert al.verify(lock, root, level="cheap") == []


@pytest.mark.parametrize("mut", ["append_byte", "flip_first_byte", "truncate"])
def test_tamper_is_caught_in_both_levels(tmp_path: pathlib.Path, mut: str) -> None:
    root, lock = _fake_tree(tmp_path)
    p = root / "v1-store/models/pi05_vision_encoder/siglip_params.pkl"
    b = bytearray(p.read_bytes())
    if mut == "append_byte":
        b += b"\x00"
    elif mut == "flip_first_byte":
        b[0] ^= 0xFF
    else:
        b = b[:-1]
    p.write_bytes(bytes(b))
    for level in ("cheap", "full"):
        fails = al.verify(lock, root, level=level)
        assert fails, f"{mut} 在 {level} 档未被抓到"
        assert any("siglip_params" in f for f in fails), fails


def test_flip_midfile_is_cheap_blind_and_full_catches(tmp_path: pathlib.Path) -> None:
    """**cheap 档盲区的显式契约**：保持长度的中段字节篡改，cheap 必然放行、full 必须抓到。

    这条不是普通用例，它把「cheap 挡不住什么」钉成契约，防后人误以为 cheap 挡一切。
    """
    root, lock = _fake_tree(tmp_path)
    p = root / "v1-store/models/pi05_vision_encoder/siglip_params.pkl"
    b = bytearray(p.read_bytes())
    b[1_500_000] ^= 0xFF               # 大小不变，首尾各 1 MiB 也不变
    p.write_bytes(bytes(b))
    assert al.verify(lock, root, level="cheap") == []
    fails = al.verify(lock, root, level="full")
    assert fails
    assert "sha256" in fails[0]


def test_missing_file_is_caught(tmp_path: pathlib.Path) -> None:
    root, lock = _fake_tree(tmp_path)
    (root / "v1-store/models/pi05_vision_encoder/siglip_params.pkl").unlink()
    fails = al.verify(lock, root, level="cheap")
    assert fails
    assert "文件缺失" in fails[0]


def test_broken_symlink_is_caught(tmp_path: pathlib.Path) -> None:
    root, lock = _fake_tree(tmp_path)
    p = root / "v1-store/models/pi05_vision_encoder/siglip_params.pkl"
    p.unlink()
    p.symlink_to(root / "nonexistent-target")
    fails = al.verify(lock, root, level="cheap")
    assert fails
    assert "文件缺失" in fails[0]


def test_lock_missing_entry_is_not_a_pass(tmp_path: pathlib.Path) -> None:
    """lock 少一条资产时，对该名字的校验必须报错，不得静默 PASS（漏检比误杀更危险）。"""
    root, lock = _fake_tree(tmp_path)
    del lock["assets"]["siglip_params"]
    with pytest.raises(SystemExit, match="没有资产"):
        al.verify(lock, root, level="cheap", names=["siglip_params"])


def test_verify_rejects_bad_level(tmp_path: pathlib.Path) -> None:
    root, lock = _fake_tree(tmp_path)
    with pytest.raises(SystemExit, match="level 非法"):
        al.verify(lock, root, level="loose")


# ── 与真实落盘资产的一条轻量互证（缺文件时 skip，异地也能跑）──────────────────


@pytest.mark.skipif(
    not (al.REPO_ROOT / LOCK["assets"]["motionjepa_config"]["dest"]).is_file(),
    reason="本机没有 MotionJEPA config.yaml（异地未取资产）")
def test_real_small_asset_matches_lock() -> None:
    assert al.verify(LOCK, al.REPO_ROOT, level="full", names=["motionjepa_config"]) == []


@pytest.mark.skipif(
    not (al.REPO_ROOT / "v1-store/datasets/4task-motion-40ep/motion/meta/store_meta.json").is_file(),
    reason="本机没有 40ep motion 库")
def test_lock_cross_checks_motion_store_provenance() -> None:
    """lock ↔ motion store provenance 交叉互证。

    后者是 2026-09-03 建 40 ep 库时由 wan 子链路**独立**写出的，不是抄 lock 的，
    所以两者一致等于两条互不相干的记录互证。
    """
    meta = json.loads((al.REPO_ROOT / "v1-store/datasets/4task-motion-40ep/motion/meta/store_meta.json")
                      .read_text(encoding="utf-8"))
    prov = meta["provenance"]
    assert prov["encoder"]["checkpoint_sha256"] == al.expected_sha256("motionjepa_ckpt", LOCK)
    assert prov["vae"]["vae_state_sha256"] == LOCK["assets"]["wan_vae"]["secondary"][0]["value"]


def test_env_does_not_override_home() -> None:
    """AGENTS.md 第 14 条：禁止覆盖 HOME（覆盖会让 ssh 找不到 ~/.ssh 与 ControlMaster socket）。"""
    assert os.environ.get("HOME"), "HOME 未设置"
