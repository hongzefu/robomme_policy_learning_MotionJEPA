"""restore_mixed8_vla_latents 的 CPU 回归：合成小 tar + 假 hf 命令 + 合成 metadata，不访问真实 HF。

全部夹具只写 tmp_path；正例与 D4/D8 反例（错归档 SHA、链接/越界/重复/名单外成员、错 arc_prefix、少一段、
同步篡改 bin+sidecar+单段 metadata、错总 metadata SHA、错 VAE state、错 manifest、重试上限、断点续跑）。
"""

import dataclasses
import hashlib
import io
import json
import os
import pathlib
import sys
import tarfile

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for relative in ("scripts/dataset", "scripts/dataset/wan"):
    sys.path.insert(0, str(ROOT / relative))

import restore_mixed8_vla_latents as R  # noqa: E402
import wan_common as wc  # noqa: E402

BUCKET = "Test/fake-bucket"
VAE_STATE = "ab" * 32


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_imports_worktree_code():
    assert pathlib.Path(R.__file__).resolve().parent == ROOT / "scripts/dataset"
    assert pathlib.Path(wc.__file__).resolve().parent == ROOT / "scripts/dataset/wan"


# ── 夹具 ──────────────────────────────────────────────────────────────────────


def make_manifest():
    eps = [dict(global_episode_idx=0, h5_file="record_dataset_T.h5", raw_ep_idx=0, num_timesteps=50, exec_start_idx=20),
           dict(global_episode_idx=1, h5_file="record_dataset_T.h5", raw_ep_idx=1, num_timesteps=100, exec_start_idx=40)]
    m = {"version": 1, "raw_dir": "/scratch/hongze/测试/raw", "episodes": eps}
    m["sha256"] = wc.manifest_sha256(m)
    return m


def vae_info(state=VAE_STATE):
    return {"vae_id": "Wan-AI/test", "vae_state_sha256": state, "vae_dtype": "float32", "latent_mode": "mode",
            "batch": 1, "tf32": "off", "amp": "off", "torch": "2.9", "cuda": "12", "cudnn": 9,
            "diffusers": "0.35", "module_sha256": "cd" * 32, "encoder_src_sha256": "ef" * 32}


def make_latents(manifest, *, vae_state=VAE_STATE, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    files = {}
    segs = {}
    for it in wc.list_segments(manifest):
        key, ng = it["key"], it["num_grid"]
        rows = []
        for m in range(ng):
            off = 16 * m
            real = min(33, it["seg_len"] - off)
            rows.append({"m": m, "seg_offset": off, "start_global_frame": it["seg_start"] + off,
                         "input_shape": [33, 256, 256, 3], "input_dtype": "uint8",
                         "input_frames_sha256": sha(f"{key}{m}".encode()), "real_frames": real,
                         "pad_frames": 33 - real,
                         "pad_source_frame": it["seg_start"] + it["seg_len"] - 1 if real < 33 else None})
        blob = rng.integers(0, 255, ng * wc.CHUNK_BYTES, dtype=np.uint8).tobytes()
        s = sha(blob)
        files[f"wan-latents/{key}.bin"] = blob
        files[f"wan-latents/{key}.bin.sha256"] = f"{s}  {key}.bin\n".encode()
        worker = {"hostname": "h", "gpu_uuid": "u", "worker": "gpu0", "pid": 1, "git_commit": "g", "mj_commit": "mj"}
        meta = {"schema": 3, "segment": key, "g": it["g"], "segment_kind": it["segment"], "seg_len": it["seg_len"],
                "seg_start_global": it["seg_start"], "num_grid": ng, "min_real": it["min_real"], "rows": rows,
                "sha256": s, "bytes": len(blob), "worker": worker, "vae": vae_info(vae_state)}
        files[f"wan-latents/{key}.metadata.json"] = json.dumps(meta).encode()
        segs[key] = {"num_grid": ng, "seg_len": it["seg_len"], "seg_start_global": it["seg_start"],
                     "segment_kind": it["segment"], "g": it["g"], "rows": rows, "sha256": s, "bytes": len(blob),
                     "worker": "gpu0", "min_real": it["min_real"]}
    total = {"schema": 3, "stage": "wan", "grid_stride": 16, "window_frames": 33, "demo_min_real_frames": 17,
             "exec_min_real_frames": 33, "demo_tail_pad": "repeat_last", "raw_dir": manifest["raw_dir"],
             "manifest_sha256": manifest["sha256"], "segments": segs,
             "workers": [{"hostname": "h", "git_commit": "g", "mj_commit": "mj"}],
             "totals": {"segments": len(segs), "windows": sum(s["num_grid"] for s in segs.values()),
                        "bytes": sum(s["bytes"] for s in segs.values())}}
    files["wan-latents/metadata.json"] = json.dumps(total, indent=1).encode()
    return files, total


def make_oracle(total, n_shards=2, *, rows_checked=None, drop_pad=False):
    segs = sorted(total["segments"])
    sampled = []
    for k in segs:
        for r in total["segments"][k]["rows"]:
            if r["pad_frames"] > 0 or r["m"] == 0:
                sampled.append({"segment": k, "m": r["m"]})
    if drop_pad:
        pads = [w for w in sampled if any(r["m"] == w["m"] and r["pad_frames"] > 0
                                          for r in total["segments"][w["segment"]]["rows"])]
        sampled.remove(pads[0])
    windows = sum(total["segments"][k]["num_grid"] for k in segs)
    tested = sha(json.dumps(total, indent=1).encode())
    common = {"manifest_sha256": total["manifest_sha256"], "raw_dir": total["raw_dir"],
              "sample_spec": "padded:all,rest:0.1,seed:0", "demo_min_real_frames": 17, "exec_min_real_frames": 33,
              "tested_metadata_sha256": tested, "vae_provenance": vae_info(), "orig_module_sha256": "11" * 32}
    files = {}
    for i in range(n_shards):
        mine = segs[i::n_shards]
        sw = [w for w in sampled if w["segment"] in mine]
        rep = dict(common, segments={k: {} for k in mine}, frame_mismatches=0, metadata_mismatches=0,
                   metadata_rows_checked=sum(total["segments"][k]["num_grid"] for k in mine),
                   windows=len(sw), windows_encoded=len(sw), shard={"idx": i, "num": n_shards})
        files[f"oracle/wan-mj/vae_report.shard{i}of{n_shards}.json"] = json.dumps(rep).encode()
        files[f"oracle/wan-mj/sampled_windows.shard{i}of{n_shards}.json"] = json.dumps(
            {"sample_spec": common["sample_spec"], "windows": sw}).encode()
    agg = dict(common, segments={k: {} for k in segs}, frame_mismatches=0, metadata_mismatches=0,
               metadata_rows_checked=windows if rows_checked is None else rows_checked,
               windows=len(sampled), windows_encoded=len(sampled), aggregated_from_shards=n_shards)
    files["oracle/wan-mj/vae_report.json"] = json.dumps(agg).encode()
    files["oracle/wan-mj/sampled_windows.json"] = json.dumps({"sample_spec": common["sample_spec"],
                                                             "windows": sampled}).encode()
    files["oracle/wan-mj/encoder_report.json"] = b'{"old": "encoder"}'
    files["oracle/wan-mj/motion_token.f32.bin"] = b"\x00" * 64
    files["oracle/wan-mj/T_ep0_demo.bin"] = b"\x01" * 32
    return files, len(sampled), sum(1 for w in sampled if any(
        r["m"] == w["m"] and r["pad_frames"] > 0 for r in total["segments"][w["segment"]]["rows"]))


def write_tar(path: pathlib.Path, entries):
    """entries: [(name, bytes | None, type, linkname)]，与 pack_and_hash 一样零化元数据。"""
    with tarfile.open(path, "w") as tf:
        for name, data, typ, link in entries:
            ti = tarfile.TarInfo(name=name)
            ti.type = typ
            ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = 0, 0o644, 0, 0, "", ""
            if typ in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                ti.linkname = link
                tf.addfile(ti)
            elif typ in (tarfile.REGTYPE, tarfile.AREGTYPE):
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
            else:
                tf.addfile(ti)


FAKE_HF = r"""#!/bin/bash
# 假 hf：buckets cp hf://buckets/<bucket>/<rel> <dst>
[ "$1" = buckets ] && [ "$2" = cp ] || exit 9
rel="${3#hf://buckets/$FAKE_BUCKET/}"
echo "$rel" >> "$FAKE_LOG"
n=$(wc -l < "$FAKE_LOG")
if [ -n "$FAKE_FAIL_FIRST" ] && [ "$n" -le "$FAKE_FAIL_FIRST" ]; then exit 1; fi
if [ -n "$FAKE_FAIL_REL" ] && [ "$rel" = "$FAKE_FAIL_REL" ]; then exit 1; fi
cp "$FAKE_ROOT/$rel" "$4"
"""


class Fix:
    def __init__(self, tmp: pathlib.Path, monkeypatch, *, latent_hook=None, tar_hook=None, plan_hook=None,
                 oracle_kw=None, vae_state=VAE_STATE, drop_segment_files=None):
        self.tmp = tmp
        manifest = make_manifest()
        files, total = make_latents(manifest, vae_state=vae_state)
        if latent_hook:
            latent_hook(files)
        ofiles, n_numeric, n_pad = make_oracle(total, **(oracle_kw or {}))
        if drop_segment_files:
            for suf in (".bin", ".bin.sha256", ".metadata.json"):
                del files[f"wan-latents/{drop_segment_files}{suf}"]
        names = sorted(files)
        half = len(names) // 2
        groups = [names[:half], names[half:]]
        bucket = tmp / "bucket"
        export = tmp / "export"
        (export / "logs/parts").mkdir(parents=True)
        sums, progress, lat_plan = [], [], []
        for i, g in enumerate(groups):
            rel = f"wan-latents_tars/wan-latents-{i:05d}.tar"
            entries = [(n, files[n], tarfile.REGTYPE, "") for n in g]
            plan_names = [n[len("wan-latents/"):] for n in g]
            if tar_hook:
                entries = tar_hook("latent", i, entries)
            p = bucket / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            write_tar(p, entries)
            shard = {"names": plan_names, "bytes": sum(len(files[n]) for n in g), "rel": rel,
                     "src_dir": "/x", "arc_prefix": "wan-latents", "kind": "flat_shard"}
            if plan_hook:
                plan_hook(shard)
            lat_plan.append(shard)
            (export / "logs/parts" / (pathlib.Path(rel).name + ".sha256")).write_text(
                "".join(f"{sha(files[n])}  {n}\n" for n in g))
            tsha = sha(p.read_bytes())
            sums.append(f"{tsha}  {rel}")
            progress.append({"rel": rel, "kind": "flat_shard", "sha256": tsha, "bytes": p.stat().st_size,
                             "members": len(g), "member_bytes": shard["bytes"]})
        orel = "oracle_tars/oracle-wan-mj-00035.tar"
        onames = sorted(ofiles)
        op = bucket / orel
        op.parent.mkdir(parents=True, exist_ok=True)
        oentries = [(n, ofiles[n], tarfile.REGTYPE, "") for n in onames]
        if tar_hook:
            oentries = tar_hook("oracle", 0, oentries)
        write_tar(op, oentries)
        osha = sha(op.read_bytes())
        sums.append(f"{osha}  {orel}")
        progress.append({"rel": orel, "kind": "flat_shard", "sha256": osha, "bytes": op.stat().st_size,
                         "members": len(onames), "member_bytes": sum(len(ofiles[n]) for n in onames)})
        (export / "logs/parts" / "oracle-wan-mj-00035.tar.sha256").write_text(
            "".join(f"{sha(ofiles[n])}  {n}\n" for n in onames))
        (export / "SHA256SUMS.post.txt").write_text("\n".join(sums) + "\n")
        (export / "logs/pack_progress.jsonl").write_text("".join(json.dumps(x) + "\n" for x in progress))
        (export / "logs/wan-latents_plan.json").write_text(json.dumps({"shards": lat_plan}))
        (export / "logs/oracle-wan-mj_plan.json").write_text(json.dumps({"shards": [
            {"names": [n[len("oracle/"):] for n in onames], "bytes": sum(len(ofiles[n]) for n in onames),
             "rel": orel, "src_dir": "/x", "arc_prefix": "oracle", "kind": "flat_shard"}]}))
        parent = tmp / "parent"
        (parent / "meta").mkdir(parents=True)
        ep = json.dumps(manifest).encode()
        inp = b'{"input": 1}'
        (parent / "meta/episode_manifest.json").write_bytes(ep)
        (parent / "meta/input_manifest.json").write_bytes(inp)
        norm = tmp / "norm_stats.json"
        norm.write_bytes(b'{"norm": 1}')
        windows = total["totals"]["windows"]
        padded = sum(1 for s in total["segments"].values() for r in s["rows"] if r["pad_frames"] > 0)
        self.pins = R.Pins(
            bucket=BUCKET, latent_tars=2, latent_tar_bytes=sum(x["bytes"] for x in progress[:2]),
            latent_members=len(files), segments=len(total["segments"]), oracle_sha256=osha,
            oracle_bytes=op.stat().st_size, total_metadata_sha256=sha(files["wan-latents/metadata.json"]),
            vae_state_sha256=VAE_STATE, episode_manifest_sha256=sha(ep), input_manifest_sha256=sha(inp),
            manifest_canonical_sha256=manifest["sha256"], norm_sha256=sha(b'{"norm": 1}'), windows=windows,
            numeric_windows=n_numeric, padded_windows=padded, oracle_shards=2)
        self.files, self.ofiles, self.total = files, ofiles, total
        self.export, self.parent, self.norm, self.bucket = export, parent, norm, bucket
        self.out = tmp / "restore" / "out-a"
        self.log = tmp / "fakehf.log"
        hf = tmp / "fakehf.sh"
        hf.write_text(FAKE_HF)
        self.hf = hf
        monkeypatch.setenv("FAKE_BUCKET", BUCKET)
        monkeypatch.setenv("FAKE_ROOT", str(bucket))
        monkeypatch.setenv("FAKE_LOG", str(self.log))
        monkeypatch.delenv("FAKE_FAIL_FIRST", raising=False)
        monkeypatch.delenv("FAKE_FAIL_REL", raising=False)

    def argv(self, *extra):
        return ["--export-dir", str(self.export), "--parent-lib", str(self.parent), "--norm-stats", str(self.norm),
                "--out-dir", str(self.out), "--hf-cmd", f"bash {self.hf}", "--retry-sleep", "0",
                "--min-free-gib", "0", "--allow-no-token", *extra]

    def run(self, *extra, pins=None):
        args = R.build_parser().parse_args(self.argv(*extra))
        return R.run(args, pins or self.pins)

    def downloads(self):
        return self.log.read_text().split() if self.log.exists() else []


def expect_fail(fix, reason, *extra, pins=None):
    with pytest.raises(R.RestoreError) as ei:
        fix.run(*extra, pins=pins)
    assert ei.value.reason == reason, str(ei.value)
    return ei.value


# ── 正例 ──────────────────────────────────────────────────────────────────────


def test_positive_restore_and_receipt(tmp_path, monkeypatch, capsys):
    fix = Fix(tmp_path, monkeypatch)
    assert R.main(fix.argv(), pins=fix.pins) == 0
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith("VAE_INPUT_READY=PASS objects=3 ")
    rc = json.loads((fix.out / R.RECEIPT).read_text())
    assert rc["schema_version"] == R.SCHEMA_VERSION and rc["status"] == "PASS"
    assert rc["new_encoder_identity"] is None
    assert rc["counts"]["segments"] == fix.pins.segments and rc["counts"]["windows"] == fix.pins.windows
    assert rc["counts"]["padded_windows"] == fix.pins.padded_windows
    assert rc["latents"]["total_metadata_sha256"] == fix.pins.total_metadata_sha256
    assert rc["vae"]["vae_state_sha256"] == VAE_STATE
    assert rc["oracle"]["numeric_padded_windows"] == fix.pins.padded_windows
    assert "不是全量数值对拍" in rc["oracle"]["coverage"]
    assert [o["kind"] for o in rc["objects"]] == ["latent", "latent", "oracle"]
    # 文件逐字节等于源，oracle 只提白名单，旧 encoder 报告不落盘
    for n, b in fix.files.items():
        assert (fix.out / n).read_bytes() == b
    got_oracle = sorted(p.name for p in (fix.out / "oracle/wan-mj").iterdir())
    assert got_oracle == sorted(pathlib.Path(n).name for n in fix.pins.oracle_whitelist())
    assert not (fix.out / "oracle/wan-mj/encoder_report.json").exists()
    lst = (fix.out / R.FILES_LIST).read_text().splitlines()
    assert len(lst) == len(fix.files) + len(fix.pins.oracle_whitelist())
    assert not list((tmp_path / "restore").glob("out-a.staging/*.tar"))
    # 已封口不改写
    expect_fail(fix, "receipt_exists")


# ── 归档 / tar 反例（D8）──────────────────────────────────────────────────────


def test_wrong_archive_sha(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch)
    bad = "0" * 64
    rel = "wan-latents_tars/wan-latents-00000.tar"
    s = (fix.export / "SHA256SUMS.post.txt").read_text().splitlines()
    s = [f"{bad}  {rel}" if l.endswith(rel) else l for l in s]
    (fix.export / "SHA256SUMS.post.txt").write_text("\n".join(s) + "\n")
    pp = [json.loads(l) for l in (fix.export / "logs/pack_progress.jsonl").read_text().splitlines()]
    for p in pp:
        if p["rel"] == rel:
            p["sha256"] = bad
    (fix.export / "logs/pack_progress.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pp))
    expect_fail(fix, "download_retries_exhausted")
    assert len(fix.downloads()) == 3
    assert not (fix.out / "wan-latents").exists()


def test_sums_vs_progress_disagree(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch)
    rel = "wan-latents_tars/wan-latents-00001.tar"
    s = [f"{'1' * 64}  {rel}" if l.endswith(rel) else l
         for l in (fix.export / "SHA256SUMS.post.txt").read_text().splitlines()]
    (fix.export / "SHA256SUMS.post.txt").write_text("\n".join(s) + "\n")
    expect_fail(fix, "progress_sha_mismatch")


def _inject(kind_want, idx_want, extra):
    def hook(kind, idx, entries):
        if kind == kind_want and idx == idx_want:
            return entries + extra(entries)
        return entries
    return hook


@pytest.mark.parametrize("extra,reason", [
    (lambda e: [("wan-latents/evil", None, tarfile.SYMTYPE, "/etc/passwd")], "tar_link_member"),
    (lambda e: [("wan-latents/evil", None, tarfile.LNKTYPE, e[0][0])], "tar_link_member"),
    (lambda e: [("wan-latents/../evil", b"x", tarfile.REGTYPE, "")], "tar_path_traversal"),
    (lambda e: [("/wan-latents/evil", b"x", tarfile.REGTYPE, "")], "tar_absolute_path"),
    (lambda e: [(e[0][0], e[0][1], tarfile.REGTYPE, "")], "tar_duplicate_member"),
    (lambda e: [("wan-latents/extra.bin", b"x", tarfile.REGTYPE, "")], "tar_unexpected_member"),
    (lambda e: [("other/x.bin", b"x", tarfile.REGTYPE, "")], "tar_arc_prefix"),
    (lambda e: [("wan-latents/fifo", None, tarfile.FIFOTYPE, "")], "tar_non_regular"),
    (lambda e: [("wan-latents/dev", None, tarfile.CHRTYPE, "")], "tar_non_regular"),
    (lambda e: [("wan-latents/sub", None, tarfile.DIRTYPE, "")], "tar_non_regular"),
])
def test_tar_member_rejections(tmp_path, monkeypatch, extra, reason):
    fix = Fix(tmp_path, monkeypatch, tar_hook=_inject("latent", 1, extra))
    expect_fail(fix, reason)
    # 扫描在写任何文件之前失败：第 0 包已完成，第 1 包一个文件都没落盘
    bad_names = [n for n in fix.files if n.startswith("wan-latents/") and
                 n[len("wan-latents/"):] in json.loads((fix.export / "logs/wan-latents_plan.json").read_text())
                 ["shards"][1]["names"]]
    assert not any((fix.out / n).exists() for n in bad_names)


def test_oracle_tar_unexpected_member(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch,
              tar_hook=_inject("oracle", 0, lambda e: [("oracle/wan-mj/injected.json", b"{}", tarfile.REGTYPE, "")]))
    expect_fail(fix, "tar_unexpected_member")


def test_tar_missing_member(tmp_path, monkeypatch):
    def hook(kind, idx, entries):
        return entries[1:] if (kind, idx) == ("latent", 0) else entries
    fix = Fix(tmp_path, monkeypatch, tar_hook=hook)
    expect_fail(fix, "tar_missing_member")


def test_plan_wrong_arc_prefix(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch, plan_hook=lambda s: s.update(arc_prefix="wan-latent"))
    expect_fail(fix, "plan_arc_prefix")


def test_dest_parent_symlink(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch)
    real = tmp_path / "elsewhere"
    real.mkdir()
    (tmp_path / "restore").mkdir()
    os.symlink(real, tmp_path / "restore" / "out-a")
    expect_fail(fix, "out_dir_symlink")
    assert not any(real.iterdir())


# ── 段 / metadata / 来源反例（D1/D4/D5）────────────────────────────────────────


def test_missing_segment(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch, drop_segment_files="T_ep1_exec")
    expect_fail(fix, "segment_set_mismatch")


def test_tampered_bin_with_synced_sidecar_and_segment_meta(tmp_path, monkeypatch):
    """D4：bin 被改且 sidecar / 单段 metadata 同步重写；tar 与 parts 也随之更新，仍被固定总 metadata 拦下。"""
    def hook(files):
        key = "T_ep1_demo"
        b = bytearray(files[f"wan-latents/{key}.bin"])
        b[100] ^= 0xFF
        b = bytes(b)
        s = sha(b)
        files[f"wan-latents/{key}.bin"] = b
        files[f"wan-latents/{key}.bin.sha256"] = f"{s}  {key}.bin\n".encode()
        m = json.loads(files[f"wan-latents/{key}.metadata.json"])
        m["sha256"] = s
        files[f"wan-latents/{key}.metadata.json"] = json.dumps(m).encode()
    fix = Fix(tmp_path, monkeypatch, latent_hook=hook)
    expect_fail(fix, "bin_vs_total_metadata")


def test_wrong_total_metadata_pin(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch)
    expect_fail(fix, "total_metadata_pin", pins=dataclasses.replace(fix.pins, total_metadata_sha256="2" * 64))


def test_extracted_total_metadata_tampered_before_closure(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_FAIL_REL", "oracle_tars/oracle-wan-mj-00035.tar")
    expect_fail(fix, "download_retries_exhausted", "--retries", "0")
    (fix.out / "wan-latents/metadata.json").write_bytes(b"{}")
    monkeypatch.delenv("FAKE_FAIL_REL")
    expect_fail(fix, "member_sha_mismatch")


def test_wrong_vae_state(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch, vae_state="cd" * 32)
    expect_fail(fix, "vae_state_or_flags")


def test_wrong_manifest(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch)
    expect_fail(fix, "episode_manifest_sha", pins=dataclasses.replace(fix.pins, episode_manifest_sha256="3" * 64))
    fix2_out = fix.out
    assert not (fix2_out / R.RECEIPT).exists()


def test_wrong_input_manifest_and_norm(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch)
    (fix.parent / "meta/input_manifest.json").write_bytes(b"{}")
    expect_fail(fix, "input_manifest_sha")
    (fix.parent / "meta/input_manifest.json").write_bytes(b'{"input": 1}')
    fix.norm.write_bytes(b"{}")
    expect_fail(fix, "norm_sha")


def test_oracle_rows_checked_short(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch, oracle_kw={"rows_checked": 1})
    expect_fail(fix, "oracle_report_field")


def test_oracle_padded_coverage(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch, oracle_kw={"drop_pad": True})
    expect_fail(fix, "oracle_padded_coverage")


def test_padded_count_pin(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch)
    expect_fail(fix, "window_counts", pins=dataclasses.replace(fix.pins, padded_windows=fix.pins.padded_windows + 1))


# ── 下载重试 / 续跑 / 目录守卫 ─────────────────────────────────────────────────


def test_retry_limit_exhausted(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_FAIL_FIRST", "100")
    expect_fail(fix, "download_retries_exhausted")
    assert len(fix.downloads()) == 3


def test_retry_then_success(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_FAIL_FIRST", "2")
    assert fix.run() == 0
    rc = json.loads((fix.out / R.RECEIPT).read_text())
    assert rc["objects"][0]["attempts"] == 3 and rc["objects"][1]["attempts"] == 1


def test_resume_skips_done_objects(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_FAIL_REL", "wan-latents_tars/wan-latents-00001.tar")
    expect_fail(fix, "download_retries_exhausted")
    st = json.loads((fix.out / R.STATE).read_text())
    assert list(st["objects"]) == ["wan-latents_tars/wan-latents-00000.tar"]
    monkeypatch.delenv("FAKE_FAIL_REL")
    fix.log.unlink()
    assert fix.run() == 0
    assert "wan-latents_tars/wan-latents-00000.tar" not in fix.downloads()


def test_resume_reuses_verified_staging(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_FAIL_REL", "oracle_tars/oracle-wan-mj-00035.tar")
    expect_fail(fix, "download_retries_exhausted", "--keep-staging")
    monkeypatch.delenv("FAKE_FAIL_REL")
    # 人为删掉 state 中第 1 包记录：staging 里 SHA 正确的 tar 被复用而不重新下载
    st = json.loads((fix.out / R.STATE).read_text())
    del st["objects"]["wan-latents_tars/wan-latents-00001.tar"]
    (fix.out / R.STATE).write_text(json.dumps(st))
    fix.log.unlink()
    assert fix.run() == 0
    assert fix.downloads() == ["oracle_tars/oracle-wan-mj-00035.tar"]


def test_out_dir_not_empty_without_state(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch)
    fix.out.mkdir(parents=True)
    (fix.out / "foreign.txt").write_text("x")
    expect_fail(fix, "out_dir_not_empty")
    assert sorted(os.listdir(fix.out)) == ["foreign.txt"]   # 拒绝前不写锁/state


def test_state_from_other_inputs_rejected(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch)
    fix.out.mkdir(parents=True)
    (fix.out / R.STATE).write_text(json.dumps({"task": R.TASK, "inputs_digest": "x", "objects": {}}))
    expect_fail(fix, "state_mismatch")


def test_extra_file_in_restore_dir_rejected(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_FAIL_REL", "oracle_tars/oracle-wan-mj-00035.tar")
    expect_fail(fix, "download_retries_exhausted", "--retries", "0")
    monkeypatch.delenv("FAKE_FAIL_REL")
    (fix.out / "wan-latents/stray.bin").write_bytes(b"x")
    expect_fail(fix, "file_set_mismatch")


def test_hardlinked_output_rejected(tmp_path, monkeypatch):
    fix = Fix(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_FAIL_REL", "oracle_tars/oracle-wan-mj-00035.tar")
    expect_fail(fix, "download_retries_exhausted", "--retries", "0")
    monkeypatch.delenv("FAKE_FAIL_REL")
    os.link(fix.out / "wan-latents/metadata.json", tmp_path / "outside-link")
    expect_fail(fix, "hardlinked_file")


def test_dry_run_and_fail_line(tmp_path, monkeypatch, capsys):
    fix = Fix(tmp_path, monkeypatch)
    assert R.main(fix.argv("--dry-run"), pins=fix.pins) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1].startswith("RESTORE_DRYRUN=PASS objects=3 latent_tars=2 ")
    assert not fix.out.exists() and not fix.downloads()
    bad = dataclasses.replace(fix.pins, latent_tar_bytes=1)
    assert R.main(fix.argv("--dry-run"), pins=bad) == 1
    assert capsys.readouterr().out.strip().splitlines()[-1] == "RESTORE_DRYRUN=FAIL reason=latent_tar_bytes"
    assert R.main(fix.argv(), pins=dataclasses.replace(fix.pins, oracle_sha256="4" * 64)) == 1
    assert capsys.readouterr().out.strip().splitlines()[-1] == "VAE_INPUT_READY=FAIL reason=oracle_pin_mismatch"
