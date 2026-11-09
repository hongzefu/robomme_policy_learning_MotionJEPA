"""export_wan_timepool 的 CPU 小夹具：偶数窗、m//2、补尾、exec fill、SHA 不符与不覆盖。"""
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dataset"))
import export_wan_timepool as ex
import pack_motion_store as pack
from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.datastore.manifest import manifest_sha256
from mme_vla_suite.shared import wan_timepool as tp

H5_NAME = "record_dataset_T.h5"
# (demo_len, exec_len)：ep0 demo 尾窗在偶数 m=4 补帧；ep1 demo 尾窗在偶数 m=2 补帧；ep2 无 demo。
SPECS = [(83, 65), (50, 40), (0, 33)]
REPO = "HongzeFu/synthetic"
REVISION = "d" * 40


def dump(path, value):
    pathlib.Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def load(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


class Tree:
    """一套完整的计划、原收据、恢复收据、恢复 latent、manifest 与源 H5。"""


def fill_route(segment, m, ng, pad):
    """真实计划里 demo 补尾窗与 completed 之后的 exec 窗走 fill；这里同样让末窗走 fill。"""
    return "fill" if pad or (segment == "exec" and m == ng - 1) else "copy"


def latent_for(seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(tp.TIMEPOOL_INPUT_SHAPE, dtype=np.float32)


def build_tree(root: pathlib.Path) -> Tree:
    t = Tree()
    t.root = root
    t.meta_dir = root / "meta"
    t.meta_dir.mkdir(parents=True)
    t.raw = root / "h5"
    t.raw.mkdir()
    (t.raw / H5_NAME).write_bytes(b"synthetic-h5")
    h5_item = dict(name=H5_NAME, sha256=ms.sha256_file(t.raw / H5_NAME), bytes=(t.raw / H5_NAME).stat().st_size)
    t.input_path = t.meta_dir / "input_manifest.json"
    dump(t.input_path, dict(raw_dir=str(t.raw), count=1,
                            files={H5_NAME: dict(size=h5_item["bytes"], sha256=h5_item["sha256"])}))
    rows, offset, total = [], 0, 0
    for g, (demo, exe) in enumerate(SPECS):
        rows.append(dict(global_episode_idx=g, h5_file=H5_NAME, raw_ep_idx=g, num_timesteps=demo + exe,
                         exec_start_idx=demo, exec_samples=exe, exec_sample_offset=offset,
                         total_sample_offset=total, shard_idx=0))
        offset += exe
        total += demo + exe
    manifest = dict(version=1, raw_dir=str(t.raw), canonical_order=[H5_NAME], num_shards=1,
                    shard_load_timesteps=[total], episodes=rows,
                    totals=dict(episodes=len(rows), timesteps=total, exec_samples=offset))
    manifest["sha256"] = manifest_sha256(manifest)
    t.manifest = manifest
    t.manifest_path = t.meta_dir / "episode_manifest.json"
    dump(t.manifest_path, manifest)

    t.latents = root / "recovery" / "wan-latents"
    t.latents.mkdir(parents=True)
    segments, plan_rows, receipt_rows, files, published, seed = [], [], [], [], [], 0
    for g, (demo, exe) in enumerate(SPECS):
        for segment, length, start in (("demo", demo, 0), ("exec", exe, demo)):
            min_real = 17 if segment == "demo" else 33
            ng = ms.seg_num_grid(length, min_real, 16)
            if ng == 0:
                continue
            key = ms.segment_key(H5_NAME, g, segment)
            segments.append(dict(key=key, g=g, h5_file=H5_NAME, raw_ep_idx=g, segment=segment, seg_start=start,
                                 seg_len=length, num_grid=ng, num_chunks=ms.seg_num_chunks(length, min_real),
                                 min_real=min_real))
            published.append(dict(path=f"wan_chunk_latents/{key}.bin", sha256="b" * 64,
                                  bytes=(ng * 16 + 1) * ms.WAN_CHUNK_BYTES))
            blob = bytearray()
            for m in range(ng):
                seed += 1
                real = min(33, length - 16 * m)
                pad = 33 - real
                first = start + 16 * m
                route = fill_route(segment, m, ng, pad)
                row = dict(key=key, m=m, route=route, h5_file=H5_NAME, raw_ep_idx=g, segment_kind=segment,
                           start_global_frame=first, seg_offset=16 * m, real_frames=real, pad_frames=pad,
                           pad_source_frame=first + real - 1 if pad else None,
                           frame_ids=list(range(first, first + real)) + [first + real - 1] * pad,
                           source_file=f"wan_chunk_latents/{key}.bin" if route == "copy" else None,
                           source_row=m * 16 if route == "copy" else None,
                           source_offset=m * 16 * ms.WAN_CHUNK_BYTES if route == "copy" else None,
                           source_layout="group_first" if route == "copy" else None,
                           output_file=f"{key}.bin", output_offset=m * ms.WAN_CHUNK_BYTES)
                data = latent_for(seed).tobytes()
                blob += data
                plan_rows.append(row)
                receipt_rows.append({**row, "input_shape": [33, 256, 256, 3], "input_dtype": "uint8",
                                     "input_frames_sha256": ms.sha256_bytes(f"frames-{key}-{m}".encode()),
                                     "latent_sha256": ms.sha256_bytes(data)})
            (t.latents / f"{key}.bin").write_bytes(bytes(blob))
            files.append(dict(path=f"{key}.bin", sha256=ms.sha256_bytes(bytes(blob)), bytes=len(blob)))
    dump(t.latents / "metadata.json", dict(schema=3, stage="wan", grid_stride=16))
    grid32 = sum(1 for row in plan_rows if row["m"] % 2 == 0)
    counts = dict(windows=len(plan_rows), copy=sum(r["route"] == "copy" for r in plan_rows),
                  fill=sum(r["route"] == "fill" for r in plan_rows))
    plan = dict(schema="orig16-wan-cache-plan-v1", kind=ms.STAGE2_CACHE_ORIGIN, repo=REPO, revision=REVISION,
                manifest_path=str(t.manifest_path), manifest_sha256=manifest["sha256"],
                manifest_file_sha256=ms.sha256_file(t.manifest_path), input_manifest_path=str(t.input_path),
                input_manifest_sha256=ms.sha256_file(t.input_path), source_h5_files=[h5_item], segments=segments,
                windows=plan_rows, counts={**counts, "grid32": grid32, "segments": len(segments)})
    plan["sha256"] = ms.sha256_bytes(ms.canonical_json_bytes(plan))
    t.plan_path = root / "wan-cache-plan.json"
    dump(t.plan_path, plan)

    def receipt(latent_root, metadata_sha):
        return dict(schema=ms.WAN_CACHE_IMPORT_SCHEMA, kind=ms.STAGE2_CACHE_ORIGIN, status="PASS",
                    published=dict(repo=REPO, revision=REVISION, sha256sums_sha256="c" * 64, files=published),
                    source_h5_files=[h5_item], manifest_path=str(t.manifest_path),
                    manifest_sha256=manifest["sha256"], manifest_file_sha256=plan["manifest_file_sha256"],
                    input_manifest_path=str(t.input_path), input_manifest_sha256=plan["input_manifest_sha256"],
                    importer_sha256="e" * 64, plan_sha256=plan["sha256"],
                    latents=dict(root=str(latent_root), metadata_sha256=metadata_sha, files=files),
                    vae=dict(vae_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", vae_state_sha256="f" * 64, latent_mode="mode"),
                    preprocess=dict(frame_size=256, input_dtype="uint8", input_layout="THWC", window_frames=33,
                                    augmentation="none"),
                    numeric_modes=dict(copy=dict(cudnn_conv_tf32=True),
                                       fill=dict(flags=dict(cudnn_allow_tf32=False),
                                                 adapter_sha256=ms.sha256_file(ROOT / "scripts/dataset/wan/wan_motion_infer.py"))),
                    windows=receipt_rows, counts=counts)

    t.receipt_path = root / "wan-cache-import.json"
    dump(t.receipt_path, receipt("/已删除的旧/wan-latents", "1" * 64))
    t.restored_path = root / "recovery" / "wan-cache-import.restored.json"
    dump(t.restored_path, receipt(t.latents, ms.sha256_file(t.latents / "metadata.json")))
    t.out = root / "recovery" / "wan-timepool-export"
    even = [r for r in receipt_rows if r["m"] % 2 == 0]
    t.expected = dict(source_windows=len(receipt_rows), windows=len(even),
                      copy=sum(r["route"] == "copy" for r in even), fill=sum(r["route"] == "fill" for r in even),
                      demo_tail=sum(r["pad_frames"] > 0 for r in even),
                      exec_fill=sum(r["segment_kind"] == "exec" and r["route"] == "fill" for r in even))
    t.segments = segments
    return t


def run(t, **kwargs):
    return ex.run_export(t.plan_path, t.receipt_path, t.restored_path, t.latents, t.out,
                         expected=kwargs.pop("expected", t.expected), **kwargs)


@pytest.fixture
def tree(tmp_path):
    return build_tree(tmp_path)


def test_even_windows_renumber_and_pool_bytes(tree):
    record = run(tree)
    assert record["status"] == "PASS" and record["counts"]["windows"] == tree.expected["windows"]
    assert tree.expected["demo_tail"] == 2 and tree.expected["exec_fill"] >= 1 and tree.expected["copy"] >= 1
    window_map = load(tree.out / pack.TIMEPOOL_EXPORT_MAP)
    receipt = {(w["key"], w["m"]): w for w in load(tree.restored_path)["windows"]}
    for window in window_map["windows"]:
        assert window["source_m"] == window["m"] * 2
        source = receipt[window["key"], window["source_m"]]
        assert window["route"] == source["route"] and window["start_frame"] == source["start_global_frame"]
        assert window["pad_frames"] == source["pad_frames"] and window["raw_latent_sha256"] == source["latent_sha256"]
        assert window["input_frames_sha256"] == source["input_frames_sha256"]
    # 重新独立切片池化，逐段逐行比字节。
    for item in tree.segments:
        key, ng = item["key"], item["num_grid"]
        latent = np.fromfile(tree.latents / f"{key}.bin", dtype="<f4").reshape(ng, *tp.TIMEPOOL_INPUT_SHAPE)
        want = b"".join(tp.pool_raw_wan_latent(np.array(latent[m], dtype=np.float32)).tobytes() for m in range(0, ng, 2))
        got = (tree.out / f"{key}.f32.bin").read_bytes()
        assert got == want and len(got) == ((ng + 1) // 2) * 4096
        meta = load(tree.out / f"{key}.metadata.json")
        assert meta["source_grid"] == list(range(0, ng, 2)) and meta["layout"] == tp.TIMEPOOL_LAYOUT
        assert (tree.out / f"{key}.f32.bin.sha256").read_text().split()[0] == ms.sha256_bytes(got)


def test_window_map_order_and_counts_match_motion_index(tree):
    run(tree)
    window_map = load(tree.out / pack.TIMEPOOL_EXPORT_MAP)
    entries = ms.build_index_entries(tree.manifest, ms.TIMEPOOL_LAYOUT_SPEC)
    ms.validate_timepool_window_map(window_map, {"routes": {"counts": window_map["counts"]}}, entries)
    assert [w["row"] for w in window_map["windows"]] == list(range(len(window_map["windows"])))
    assert ms.index_totals(entries)["rows"] == len(window_map["windows"])
    tails = [w for w in window_map["windows"] if w["pad_frames"]]
    assert {(w["key"], w["source_m"], w["real_frames"]) for w in tails} == {("T_ep0_demo", 4, 19), ("T_ep1_demo", 2, 18)}
    assert all(w["segment"] == "demo" and w["route"] == "fill" for w in tails)
    assert any(w["segment"] == "exec" and w["route"] == "fill" for w in window_map["windows"])


def test_inputs_never_written_and_record_binds_their_sha(tree):
    before = {p: ms.sha256_file(p) for p in (tree.plan_path, tree.receipt_path, tree.restored_path)}
    record = run(tree)
    assert before == {p: ms.sha256_file(p) for p in before}
    assert record["inputs"]["import_receipt"]["file_sha256"] == before[tree.receipt_path]
    assert record["inputs"]["restored_receipt"]["file_sha256"] == before[tree.restored_path]
    assert record["inputs"]["plan"]["canonical_sha256"] == load(tree.plan_path)["sha256"]
    assert record["encoder_ckpt"] is None and record["pool"]["source_sha256"] == tp.pool_source_sha256()


def test_refuses_existing_output_even_when_empty(tree):
    tree.out.mkdir()
    with pytest.raises(FileExistsError):
        run(tree)
    assert list(tree.out.iterdir()) == []


@pytest.mark.parametrize("field", ["input_frames_sha256", "latent_sha256"])
@pytest.mark.parametrize("target", ["restored", "original"])
def test_receipt_sha_mismatch_fails_without_output(tree, field, target):
    path = tree.restored_path if target == "restored" else tree.receipt_path
    receipt = load(path)
    receipt["windows"][1][field] = "0" * 64     # 奇数窗也必须核
    dump(path, receipt)
    with pytest.raises(ValueError, match=field):
        run(tree)
    assert not tree.out.exists() and not list(tree.out.parent.glob("*.partial-*"))


def test_latent_bytes_mismatch_fails_and_cleans_partial(tree):
    key = tree.segments[-1]["key"]
    path = tree.latents / f"{key}.bin"
    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="latent 字节不符"):
        run(tree)
    assert not tree.out.exists() and not list(tree.out.parent.glob("*.partial-*"))


def test_odd_window_bytes_are_checked_too(tree):
    item = next(s for s in tree.segments if s["num_grid"] > 1)
    path = tree.latents / f"{item['key']}.bin"
    data = bytearray(path.read_bytes())
    data[ms.WAN_CHUNK_BYTES] ^= 0xFF            # m=1
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match=f"{item['key']}:1"):
        run(tree)


def test_missing_window_and_route_change_fail(tree):
    receipt = load(tree.restored_path)
    receipt["windows"].pop(0)
    dump(tree.restored_path, receipt)
    with pytest.raises(ValueError):
        run(tree)


def test_nonfinite_even_window_rejected_by_shared_pool(tree):
    item = tree.segments[0]
    path = tree.latents / f"{item['key']}.bin"
    latent = np.fromfile(path, dtype="<f4")
    latent[0] = np.nan
    path.write_bytes(latent.tobytes())
    block = latent[: ms.WAN_CHUNK_BYTES // 4].tobytes()
    for receipt_path in (tree.receipt_path, tree.restored_path):
        receipt = load(receipt_path)
        for window in receipt["windows"]:
            if (window["key"], window["m"]) == (item["key"], 0):
                window["latent_sha256"] = ms.sha256_bytes(block)
        for file in receipt["latents"]["files"]:
            if file["path"] == f"{item['key']}.bin":
                file["sha256"] = ms.sha256_file(path)
        dump(receipt_path, receipt)
    with pytest.raises(ValueError, match="非有限"):
        run(tree)


def test_plan_tamper_detected(tree):
    plan = load(tree.plan_path)
    plan["windows"][0]["route"] = "fill"
    dump(tree.plan_path, plan)
    with pytest.raises(ValueError, match="canonical"):
        run(tree)


def test_real_expected_counts_are_the_default_and_reject_small_fixture(tree):
    assert ex.REAL_COUNTS == dict(source_windows=45228, windows=23253, copy=21814, fill=1439, demo_tail=386,
                                  exec_fill=1053)
    with pytest.raises(ValueError, match="窗口计数"):
        ex.run_export(tree.plan_path, tree.receipt_path, tree.restored_path, tree.latents, tree.out)
    assert not tree.out.exists()


def cli_args(tree, subsample="2"):
    args = ["--plan", str(tree.plan_path), "--receipt", str(tree.receipt_path),
            "--restored-receipt", str(tree.restored_path), "--latents", str(tree.latents), "--out", str(tree.out),
            "--grid-subsample", subsample]
    for name, value in tree.expected.items():
        args += [f"--expected-{name.replace('_', '-')}", str(value)]
    return args


@pytest.mark.parametrize("subsample", ["1", "3", "4"])
def test_cli_rejects_other_subsample(tree, subsample):
    with pytest.raises(SystemExit):
        ex.main(cli_args(tree, subsample))
    assert not tree.out.exists()


def test_cli_prints_pass_line(tree, capsys):
    ex.main(cli_args(tree))
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith(f"EXPORT_WAN_TIMEPOOL=PASS source_windows={tree.expected['source_windows']} ")
    assert f"demo_tail={tree.expected['demo_tail']}" in line and f"exec_fill={tree.expected['exec_fill']}" in line


def test_cli_prints_fail_line(tree, capsys):
    tree.out.mkdir()
    with pytest.raises(FileExistsError):
        ex.main(cli_args(tree))
    assert "EXPORT_WAN_TIMEPOOL=FAIL" in capsys.readouterr().out
