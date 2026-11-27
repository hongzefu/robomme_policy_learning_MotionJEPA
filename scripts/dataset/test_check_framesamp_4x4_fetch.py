#!/usr/bin/env python3
"""check_framesamp_4x4_fetch.py 的回归测试：合成小表正例全 PASS，篡改后对应项 FAIL。

跑法（禁止裸 python）：
  uv run python -m pytest scripts/dataset/test_check_framesamp_4x4_fetch.py -q
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

import ml_dtypes
import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from mme_vla_suite.datastore import framesamp_store as fs  # noqa: E402
from mme_vla_suite.datastore.manifest import manifest_sha256  # noqa: E402

_spec = importlib.util.spec_from_file_location("check_fs4", _HERE / "check_framesamp_4x4_fetch.py")
chk = importlib.util.module_from_spec(_spec)
sys.modules["check_fs4"] = chk   # 进程池按模块名 pickle worker 函数
_spec.loader.exec_module(chk)

EP_LEN = [25, 15]          # 2 个 episode，40 行；num_pos_rows = 25
PROBES = [(0, 3), (1, 14)]


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write_store(root: pathlib.Path, layout: str, img: np.ndarray, pos: np.ndarray,
                 stt: np.ndarray, part_bounds: list[int], man_sha: str, t_of: np.ndarray) -> None:
    spec = fs.SPECS[layout]
    (root / spec.image_part_dir).mkdir(parents=True)
    (root / "meta").mkdir()
    parts = []
    for i in range(len(part_bounds) - 1):
        a, b = part_bounds[i], part_bounds[i + 1]
        rel = f"{spec.image_part_dir}/part_{i:03d}.bf16.bin"
        (root / rel).write_bytes(img[a:b].tobytes())
        _, ht, full = fs.headtail_digest(root / rel)
        parts.append({"index": i, "path": rel, "start_row": a, "num_rows": b - a,
                      "bytes": (b - a) * spec.image_row_bytes, "sha256": _sha(root / rel),
                      "head_tail_digest": ht, "full_covered": full})
    (root / spec.pos_table_relpath).write_bytes(pos.tobytes())
    (root / fs.STATE_TABLE_RELPATH).write_bytes(stt.tobytes())
    dig = bytearray()
    for r in range(len(img)):
        h = hashlib.blake2b(digest_size=16)
        h.update(img[r].tobytes())
        h.update(pos[t_of[r]].tobytes())
        h.update(stt[r].tobytes())
        dig += h.digest()
    (root / fs.ROW_DIGESTS_RELPATH).write_bytes(bytes(dig))
    meta = {
        "schema": 1, "layout": layout, "status": "verified", "byte_order": "little",
        "array_order": "C", "bf16_encoding": fs.BF16_ENCODING,
        "tables": {
            spec.image_key: {"row_shape": list(spec.image_row_shape), "dtype": "bfloat16",
                             "row_bytes": spec.image_row_bytes, "num_rows": len(img),
                             "part_dir": spec.image_part_dir},
            spec.pos_key: {"row_shape": list(spec.pos_row_shape), "dtype": "float32",
                           "row_bytes": spec.pos_row_bytes, "num_rows": len(pos),
                           "relpath": spec.pos_table_relpath,
                           "sha256": _sha(root / spec.pos_table_relpath)},
            fs.STATE_KEY: {"row_shape": [8], "dtype": "float32", "row_bytes": 32,
                           "num_rows": len(img), "relpath": fs.STATE_TABLE_RELPATH},
        },
        "num_rows": len(img), "num_exec_samples": 0, "num_pos_rows": len(pos),
        "manifest_sha256": man_sha, "manifest_path": "x", "source_dataset_root": "x",
        "manifest_scope": "full", "parts": parts,
        "row_digests": {"relpath": fs.ROW_DIGESTS_RELPATH, "algo": "blake2b-128",
                        "covered_rows": len(img), "coverage": fs.ROW_DIGEST_COVERAGE,
                        "byte_count": len(dig), "sha256": hashlib.sha256(bytes(dig)).hexdigest()},
    }
    (root / fs.META_RELPATH).write_text(json.dumps(meta, ensure_ascii=False))


@pytest.fixture()
def world(tmp_path):
    rng = np.random.default_rng(0)
    n = sum(EP_LEN)
    eps, off = [], 0
    for g, nt in enumerate(EP_LEN):
        eps.append({"global_episode_idx": g, "num_timesteps": nt, "exec_start_idx": 0,
                    "exec_samples": nt, "exec_sample_offset": off, "total_sample_offset": off})
        off += nt
    man = {"version": 1, "totals": {"episodes": len(EP_LEN), "timesteps": n, "exec_samples": n},
           "episodes": eps}
    man["sha256"] = manifest_sha256(man)
    (tmp_path / "episode_manifest.json").write_text(json.dumps(man))
    t_of = np.concatenate([np.arange(nt) for nt in EP_LEN])

    img4 = rng.standard_normal((n, 16, 2048)).astype(ml_dtypes.bfloat16)
    # 8×8 由 4×4 反推：每个 4×4 token 复制到对应 2×2 块（行主序），池化后与 4×4 完全一致
    g4 = img4.reshape(n, 4, 4, 2048)
    img8 = np.repeat(np.repeat(g4, 2, axis=1), 2, axis=2).reshape(n, 64, 2048)
    pos4 = rng.standard_normal((max(EP_LEN), 16, 768)).astype(np.float32)
    pos8 = rng.standard_normal((max(EP_LEN), 64, 768)).astype(np.float32)
    stt = rng.standard_normal((n, 8)).astype(np.float32)

    s4, s8 = tmp_path / "fs4", tmp_path / "fs8"
    _write_store(s4, "framesamp-4x4-v1", img4, pos4, stt, [0, 25, 40], man["sha256"], t_of)
    _write_store(s8, "framesamp-8x8-v1", img8, pos8, stt, [0, 40], man["sha256"], t_of)

    feats = tmp_path / "features"
    for g, t in PROBES:
        row = eps[g]["total_sample_offset"] + t
        d = {"image_emb_4x4": img4[row][None], "pos_emb_4x4": pos4[t][None],
             "image_emb_8x8": img8[row][None], "pos_emb_8x8": pos8[t][None],
             "state_emb": stt[row].copy()}
        (feats / f"episode_{g}").mkdir(parents=True)
        np.save(feats / f"episode_{g}" / f"token_emb_{t}.npy", d, allow_pickle=True)

    plain = [{"rel": "framesamp/" + p.relative_to(s4).as_posix(), "sha256": _sha(p),
              "bytes": p.stat().st_size} for p in sorted(s4.rglob("*")) if p.is_file()]
    plain.append({"rel": "other/x.txt", "sha256": "0" * 64, "bytes": 1})   # 非 framesamp 项须被忽略
    (tmp_path / "upload_manifest.json").write_text(json.dumps({"plain_files": plain}))

    argv = ["--store", str(s4), "--ref8", str(s8), "--probes", str(feats),
            "--manifest-json", str(tmp_path / "upload_manifest.json"),
            "--episode-manifest", str(tmp_path / "episode_manifest.json"),
            "--procs", "2", "--expect-files", str(len(plain) - 1), "--expect-rows", str(n),
            "--expect-manifest-sha", man["sha256"], "--expect-probes", str(len(PROBES))]
    return {"argv": argv, "s4": s4, "s8": s8, "n": n, "files": len(plain) - 1}


def _run(capsys, argv):
    rc = chk.main(argv)
    out = capsys.readouterr().out
    return rc, out


def _flip(path: pathlib.Path, offset: int, mask: int = 0x01) -> None:
    b = bytearray(path.read_bytes())
    b[offset] ^= mask
    path.write_bytes(bytes(b))


def test_all_pass(world, capsys):
    rc, out = _run(capsys, world["argv"])
    assert rc == 0, out
    n = world["n"]
    assert f"FS4X4_FETCH=PASS files={world['files']} " in out
    assert "mismatches=0" in out
    assert f"FS4X4_ROWDIGEST=PASS rows={n} mismatches=0" in out
    assert f"FS4X4_XGRID=PASS rows={n} min_cos=" in out
    assert "FS4X4_PROBE=PASS frames=2 mismatches=0" in out
    assert "FS4X4_META=PASS" in out
    assert out.strip().splitlines()[-1] == "FS4X4_CHECK=PASS failed=none"


def test_tamper_image_row(world, capsys):
    # 第 30 行（part_001 内）某个 bf16 的最低尾数位：摘要必须抓到，xgrid 仍过
    part = world["s4"] / "image_emb_4x4/part_001.bf16.bin"
    _flip(part, (30 - 25) * fs.IMAGE_ROW_BYTES + 100)
    rc, out = _run(capsys, world["argv"] + ["--only", "rowdigest,xgrid"])
    assert rc != 0
    assert "FS4X4_ROWDIGEST=FAIL" in out and "mismatches=1" in out
    assert "[30]" in out
    assert "FS4X4_XGRID=PASS" in out
    assert "FS4X4_CHECK=FAIL failed=rowdigest" in out


def test_tamper_pos_row(world, capsys):
    # pos 帧号 t=3：两个 episode 各有一行 t=3 → 摘要 2 行不符；探针 (0,3) 也不符
    pos = world["s4"] / "pos_emb_4x4.f32.bin"
    _flip(pos, 3 * fs.POS_ROW_BYTES + 7, 0x40)
    rc, out = _run(capsys, world["argv"] + ["--only", "rowdigest,probe"])
    assert rc != 0
    assert "FS4X4_ROWDIGEST=FAIL" in out and "mismatches=2" in out
    assert "FS4X4_PROBE=FAIL frames=2 mismatches=1" in out
    assert "FS4X4_CHECK=FAIL failed=rowdigest,probe" in out


def test_tamper_file_byte(world, capsys):
    _flip(world["s4"] / "state_emb.f32.bin", 5)
    rc, out = _run(capsys, world["argv"] + ["--only", "fetch"])
    assert rc != 0
    assert "FS4X4_FETCH=FAIL" in out and "mismatches=1" in out
    assert "state_emb.f32.bin" in out


def test_missing_file(world, capsys):
    (world["s4"] / "meta/pack_progress.jsonl").write_text("x")   # 清单外多余文件
    (world["s4"] / "image_emb_4x4/part_000.bf16.bin").unlink()
    rc, out = _run(capsys, world["argv"])
    assert rc != 0
    assert "FS4X4_FETCH=FAIL" in out and "mismatches=2" in out
    assert "FS4X4_ROWDIGEST=FAIL" in out


def test_tamper_xgrid(world, capsys):
    # 8×8 第 7 行整行置零外的扰动：把 token 0 改成随机值 → 该行池化余弦明显 < 阈值
    part8 = world["s8"] / "image_emb_8x8/part_000.bf16.bin"
    b = bytearray(part8.read_bytes())
    off = 7 * 64 * 2048 * 2
    noise = np.random.default_rng(1).standard_normal(2048).astype(ml_dtypes.bfloat16).tobytes()
    b[off:off + len(noise)] = noise
    part8.write_bytes(bytes(b))
    rc, out = _run(capsys, world["argv"] + ["--only", "xgrid"])
    assert rc != 0
    assert "FS4X4_XGRID=FAIL" in out and "bad_rows=1" in out
    assert "(7," in out


def test_meta_status(world, capsys):
    p = world["s4"] / fs.META_RELPATH
    m = json.loads(p.read_text())
    m["status"] = "packed"
    p.write_text(json.dumps(m))
    rc, out = _run(capsys, world["argv"] + ["--only", "meta"])
    assert rc != 0
    assert "FS4X4_META=FAIL" in out and "status=packed" in out
