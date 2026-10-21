"""S5-B（C2）CPU 反例与正例：历史 VAE 分组 provenance、raw8×256 行合同、compare/verify 顺序、驱动守卫。

全部夹具只写 tmp_path；不加载模型、不占 GPU。真实 GPU 链路由主会话在 S5-B 运行。
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for rel in ("scripts/dataset", "scripts/dataset/wan"):
    sys.path.insert(0, str(ROOT / rel))

from mme_vla_suite.datastore import motion_store as ms  # noqa: E402
from mme_vla_suite.datastore.manifest import manifest_sha256  # noqa: E402
import build_mixed8_v10_motion as bm  # noqa: E402
import compare_wan  # noqa: E402
import pack_motion_store as pms  # noqa: E402

PIN = json.loads((ROOT / "scripts/dataset/wan/SOURCE_PIN.json").read_text(encoding="utf-8"))
HIST_MODULE = "1" * 64            # 历史适配件 SHA（与当前 SOURCE_PIN 不同）
HIST_ENC_SRC = "2" * 64
HIST_GIT, HIST_MJ = "6ebeab5" + "0" * 33, "9" * 40
NEW_GIT = "c" * 40
VAE_STATE = "9980d252230c265cc2869466a74f85f5ee45b01ea9521bbb31159f90b75fe6d0"
CKPT_NAME = "checkpoint_epoch_60.pt"
SPECS = [(70, 33), (60, 10)]      # ep0: demo 2 窗（1 补帧）+ exec 1 窗；ep1: demo 不足 17 帧、exec 2 窗


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def manifest_for(specs, raw_dir):
    episodes, offset, total_offset = [], 0, 0
    for g, (total, es) in enumerate(specs):
        episodes.append(dict(global_episode_idx=g, h5_file="record_dataset_T.h5", raw_ep_idx=g,
                             num_timesteps=total, exec_start_idx=es, exec_samples=total - es,
                             exec_sample_offset=offset, total_sample_offset=total_offset, shard_idx=0))
        offset += total - es
        total_offset += total
    out = dict(version=1, raw_dir=str(raw_dir), canonical_order=["record_dataset_T.h5"], num_shards=1,
               totals=dict(episodes=len(episodes), timesteps=total_offset, exec_samples=offset),
               shard_load_timesteps=[total_offset], episodes=episodes)
    out["sha256"] = manifest_sha256(out)
    return out


def vae_info():
    return dict(vae_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", vae_state_sha256=VAE_STATE, vae_dtype="float32",
                latent_mode="mode", batch=1, tf32="off", amp="off", torch="2.9.0+cu128", cuda="12.8", cudnn=91002,
                diffusers="0.39.0", cublas_pkg="x", cudnn_pkg="y", module_sha256=HIST_MODULE,
                encoder_src_sha256=HIST_ENC_SRC, flags={"f": 1}, env={}, gpu_name="A100", compute_cap="8.0",
                sm_count=108, driver="570")


def worker(git, mj, wid, pid):
    return dict(hostname="h", gpu_uuid=f"GPU-{wid}", worker=wid, pid=pid, gpu_name="A100", compute_cap="8.0",
                driver_version="570", torch="2.9.0+cu128", cudnn_version=91002, git_commit=git, mj_commit=mj)


@dataclasses.dataclass
class Fx:
    root: pathlib.Path
    manifest_path: pathlib.Path
    manifest: dict
    restore: pathlib.Path
    latents: pathlib.Path
    receipt: pathlib.Path
    tokens: pathlib.Path
    enc_dir: pathlib.Path
    ckpt_sha: str
    pins: pms.HistoricalPins
    keys: list
    rows: dict


def write_tokens(fx: Fx, *, mutate=None, row_shape=(8, 256)) -> dict:
    """按段写 raw8 token（数值确定），返回 {key: bytes}。mutate(key, meta) 可改元数据做反例。"""
    rng = np.random.default_rng(7)
    out = {}
    fx.tokens.mkdir(exist_ok=True)
    state = {"w": sha(b"w")}
    for key in fx.keys:
        ng = fx.rows[key]
        lat = (fx.latents / f"{key}.bin").read_bytes()
        toks = rng.standard_normal((ng,) + tuple(row_shape)).astype(np.float32)
        blob = toks.tobytes()
        (fx.tokens / f"{key}.f32.bin").write_bytes(blob)
        (fx.tokens / f"{key}.f32.bin.sha256").write_text(f"{sha(blob)}  {key}.f32.bin\n")
        enc = dict(run_dir=str(fx.enc_dir), checkpoint=CKPT_NAME, checkpoint_sha256=fx.ckpt_sha, checkpoint_epoch=60,
                   config_sha256=ms.sha256_file(fx.enc_dir / "config.yaml"), arch="wan-latent-v10-st",
                   state_key="encoder", precision="bf16", amp="bf16", tf32="off", batch=1,
                   vae_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", motion_dims=list(row_shape), torch="2.9.0+cu128",
                   cuda="12.8", cudnn=91002, diffusers="0.39.0", cublas_pkg="x", cudnn_pkg="y",
                   module_sha256=PIN["source_sha256"], encoder_src_sha256=PIN["encoder_source_sha256"], env={},
                   gpu_name="A100", compute_cap="8.0", sm_count=108, driver="570")
        meta = dict(schema=1, segment=key, num_grid=ng, token_dim=int(np.prod(row_shape)),
                    token_bytes=int(np.prod(row_shape)) * 4, layout=ms.LAYOUT, row_shape=list(row_shape),
                    dtype="float32", input_latent_sha256=sha(lat),
                    input_block_sha256=[sha(lat[i * pms.CHUNK_BYTES:(i + 1) * pms.CHUNK_BYTES]) for i in range(ng)],
                    sha256=sha(blob), bytes=len(blob), encoder=enc, encoder_flags={"f": 1},
                    encoder_state_sha256=state, worker=worker(NEW_GIT, PIN["mj_repo_commit"], "gpu0", 100))
        if mutate:
            mutate(key, meta)
        (fx.tokens / f"{key}.metadata.json").write_text(json.dumps(meta))
        out[key] = blob
    return out


@pytest.fixture
def fx(tmp_path) -> Fx:
    raw_dir = tmp_path / "已删除的历史raw"          # 从不创建、从不打开
    parent_meta = tmp_path / "parent" / "meta"
    parent_meta.mkdir(parents=True)
    manifest = manifest_for(SPECS, raw_dir)
    manifest_path = parent_meta / "episode_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    (parent_meta / "input_manifest.json").write_text(json.dumps({"raw_dir": str(raw_dir)}))
    restore = tmp_path / "restore"
    latents = restore / "wan-latents"
    latents.mkdir(parents=True)
    spec = ms.LAYOUT_SPECS[ms.LAYOUT]
    entries = ms.build_index_entries(manifest, spec)
    keys, rows, segs = [], {}, {}
    rng = np.random.default_rng(1)
    for e, seg, key, _rb, ng in pms.iter_rows_in_order(entries):
        keys.append(key)
        rows[key] = ng
        s = getattr(e, seg)
        blob = rng.standard_normal(ng * pms.CHUNK_BYTES // 4).astype(np.float32).tobytes()
        (latents / f"{key}.bin").write_bytes(blob)
        (latents / f"{key}.bin.sha256").write_text(f"{sha(blob)}  {key}.bin\n")
        start = 0 if seg == "demo" else e.exec_start_idx
        r = []
        for m in range(ng):
            real = min(33, s.seg_len - 16 * m)
            r.append(dict(m=m, seg_offset=16 * m, start_global_frame=start + 16 * m, real_frames=real,
                          pad_frames=33 - real, pad_source_frame=(start + s.seg_len - 1) if real < 33 else None))
        lm = dict(schema=3, segment=key, num_grid=ng, seg_len=s.seg_len, seg_start_global=start, segment_kind=seg,
                  g=e.g, rows=r, sha256=sha(blob), bytes=len(blob), min_real=spec.min_real(seg), vae=vae_info(),
                  worker=worker(HIST_GIT, HIST_MJ, f"old{len(keys)}", 10 + len(keys)))
        (latents / f"{key}.metadata.json").write_text(json.dumps(lm))
        segs[key] = {k: lm[k] for k in ("num_grid", "seg_len", "seg_start_global", "segment_kind", "g", "rows",
                                        "sha256", "bytes")}
    total = dict(schema=3, raw_dir=str(raw_dir), manifest_sha256=manifest["sha256"], segments=segs,
                 **spec.fields(), grid_stride=16, window_frames=33)
    (latents / "metadata.json").write_text(json.dumps(total))
    total_sha = ms.sha256_file(latents / "metadata.json")
    oracle = restore / "oracle" / "wan-mj"
    oracle.mkdir(parents=True)
    report = dict(raw_dir=str(raw_dir), manifest_sha256=manifest["sha256"], tested_metadata_sha256=total_sha,
                  frame_mismatches=0, metadata_mismatches=0, demo_min_real_frames=17, exec_min_real_frames=33)
    (oracle / "vae_report.json").write_text(json.dumps(report))
    files = restore / "restore_files.sha256"
    files.write_text(f"{total_sha}  wan-latents/metadata.json\n")
    pins = pms.HistoricalPins(total_metadata_sha256=total_sha, vae_state_sha256=VAE_STATE,
                              episode_manifest_sha256=ms.sha256_file(manifest_path),
                              input_manifest_sha256=ms.sha256_file(parent_meta / "input_manifest.json"),
                              manifest_canonical_sha256=manifest["sha256"])
    ident = {k: vae_info()[k] for k in pms.VAE_IDENTITY_KEYS}
    receipt = dict(schema_version=pms.RECEIPT_SCHEMA, status="PASS", out_dir=str(restore),
                   layout={"latents": "wan-latents", "oracle": "oracle/wan-mj"},
                   latents=dict(total_metadata_sha256=total_sha, files_list="restore_files.sha256",
                                files_list_sha256=ms.sha256_file(files)),
                   vae=dict(vae_state_sha256=VAE_STATE, identity=ident, historical_module_sha256=HIST_MODULE,
                            historical_encoder_src_sha256=HIST_ENC_SRC, worker_count=len(keys),
                            worker_git_commits=[HIST_GIT], worker_mj_commits=[HIST_MJ]),
                   parent=dict(episode_manifest_sha256=pins.episode_manifest_sha256,
                               input_manifest_sha256=pins.input_manifest_sha256,
                               manifest_canonical_sha256=manifest["sha256"]),
                   oracle=dict(vae_report_sha256=ms.sha256_file(oracle / "vae_report.json"), coverage="测试"),
                   new_encoder_identity=None)
    receipt_path = restore / "restore_receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    enc_dir = tmp_path / "MotionJEPA/runs/final-e60"
    enc_dir.mkdir(parents=True)
    (enc_dir / CKPT_NAME).write_bytes(b"synthetic-final-e60")
    (enc_dir / "config.yaml").write_text("motion: {num_tokens: 8, dim: 256}\ntraining: {precision: bf16}\n")
    out = Fx(tmp_path, manifest_path, manifest, restore, latents, receipt_path, tmp_path / "out/motion-tokens",
             enc_dir, sha(b"synthetic-final-e60"), pins, keys, rows)
    out.tokens.parent.mkdir(parents=True)
    return out


def compat(fx: Fx) -> dict:
    return pms.load_vae_compat(fx.receipt, fx.latents, manifest_path=fx.manifest_path,
                               oracle_report=fx.restore / "oracle/wan-mj/vae_report.json", pins=fx.pins)


def gather(fx: Fx, vae_compat, expected=None):
    entries = ms.build_index_entries(fx.manifest, ms.LAYOUT_SPECS[ms.LAYOUT])
    return pms.gather_provenance(fx.latents, fx.tokens, entries, fx.manifest, "i" * 64, encoder_run_dir=fx.enc_dir,
                                 raw_dir=pathlib.Path(fx.manifest["raw_dir"]),
                                 input_manifest_path=fx.manifest_path.parent / "input_manifest.json",
                                 vae_compat=vae_compat,
                                 expected_ckpt_sha256=fx.ckpt_sha if expected is None else expected)


def edit_json(path: pathlib.Path, fn) -> None:
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj))


# ── 1. 分组 provenance 正例与历史身份反例 ──────────────────────────────────────


def test_grouped_provenance_positive(fx):
    write_tokens(fx)
    prov = gather(fx, compat(fx))
    assert prov["vae"]["module_sha256"] == HIST_MODULE and prov["encoder"]["module_sha256"] == PIN["source_sha256"]
    assert {w["git_commit"] for w in prov["workers_vae"]} == {HIST_GIT}
    assert {w["git_commit"] for w in prov["workers_encoder"]} == {NEW_GIT}
    assert prov["vae_compat"]["identity"]["vae_state_sha256"] == VAE_STATE
    assert prov["encoder"]["epoch"] == 60 and prov["encoder_state_digest"]
    # 不传 receipt（旧单 PIN 口径）时，历史 VAE 源码必然被拒——没有通用放宽
    with pytest.raises(ValueError):
        gather(fx, None)


@pytest.mark.parametrize("field,value", [("module_sha256", PIN["source_sha256"]),     # 伪造成“当前”源码
                                         ("encoder_src_sha256", PIN["encoder_source_sha256"]),
                                         ("vae_state_sha256", "0" * 64), ("torch", "2.8.0")])
def test_forged_historical_vae_identity_rejected(fx, field, value):
    write_tokens(fx)
    for key in fx.keys:                          # 全部段一起改（组内仍唯一），只能靠 receipt 历史值拦下
        edit_json(fx.latents / f"{key}.metadata.json", lambda m: m["vae"].__setitem__(field, value))
    with pytest.raises(ValueError, match="receipt|不唯一"):
        gather(fx, compat(fx))


def test_vae_worker_commit_forged_rejected(fx):
    write_tokens(fx)
    for key in fx.keys:
        edit_json(fx.latents / f"{key}.metadata.json", lambda m: m["worker"].__setitem__("git_commit", NEW_GIT))
    with pytest.raises(ValueError, match="receipt"):
        gather(fx, compat(fx))


def test_tampered_latent_with_synced_sidecar_rejected(fx):
    """D4：改 bin 并同步改 sidecar / 段 metadata / token 绑定，仍被固定 SHA 的总 metadata 拦下。"""
    key = fx.keys[0]
    p = fx.latents / f"{key}.bin"
    b = bytearray(p.read_bytes()); b[0] ^= 1; p.write_bytes(bytes(b))
    p.with_name(p.name + ".sha256").write_text(f"{sha(bytes(b))}  {key}.bin\n")
    edit_json(fx.latents / f"{key}.metadata.json", lambda m: m.__setitem__("sha256", sha(bytes(b))))
    write_tokens(fx)
    with pytest.raises(ValueError, match="总 metadata"):
        gather(fx, compat(fx))


# ── 2. 错 receipt ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r.__setitem__("schema_version", "mixed8-vla-latent-restore/v0"), "schema"),
    (lambda r: r.__setitem__("status", "FAIL"), "schema"),
    (lambda r: r.__setitem__("new_encoder_identity", {"sha": "x"}), "新 encoder"),
    (lambda r: r.__setitem__("out_dir", "/elsewhere"), "恢复目录"),
    (lambda r: r["latents"].__setitem__("total_metadata_sha256", "0" * 64), "总 metadata"),
    (lambda r: r["latents"].__setitem__("files_list_sha256", "0" * 64), "文件清单"),
    (lambda r: r["vae"]["identity"].pop("module_sha256"), "身份字段"),
    (lambda r: r["vae"].__setitem__("historical_module_sha256", PIN["source_sha256"]), "不自洽"),
    (lambda r: r["vae"].__setitem__("worker_git_commits", [HIST_GIT, NEW_GIT]), "唯一"),
    (lambda r: r["parent"].__setitem__("episode_manifest_sha256", "0" * 64), "父库"),
    (lambda r: r["oracle"].__setitem__("vae_report_sha256", "0" * 64), "oracle"),
])
def test_wrong_receipt_rejected(fx, mutate, match):
    edit_json(fx.receipt, mutate)
    with pytest.raises(ValueError, match=match):
        compat(fx)


def test_receipt_for_other_latents_dir_rejected(fx, tmp_path):
    other = tmp_path / "other/wan-latents"
    other.mkdir(parents=True)
    with pytest.raises(ValueError, match="恢复目录"):
        pms.load_vae_compat(fx.receipt, other, pins=fx.pins)


def test_only_forged_compat_switch_rejected(fx):
    """仅在段 metadata / receipt 里写“兼容”字样不起作用：无完整封口 receipt 即按单 PIN 口径拒绝。"""
    write_tokens(fx)
    for key in fx.keys:
        edit_json(fx.latents / f"{key}.metadata.json", lambda m: m.update(vae_compat=True, compat="historical"))
    with pytest.raises(ValueError, match="SOURCE_PIN"):
        gather(fx, None)
    edit_json(fx.receipt, lambda r: r.update(vae={"compat": True}))
    with pytest.raises(ValueError):
        compat(fx)
    with pytest.raises(ValueError, match="显式"):
        gather(fx, {"mode": "historical-restore-receipt"}, expected="")


# ── 3. encoder 组反例 ───────────────────────────────────────────────────────────


def test_old_encoder_tokens_rejected(fx):
    old = lambda k, m: m["encoder"].update(arch="wan-latent-v9", motion_dims=[768],  # noqa: E731
                                           checkpoint="checkpoint_epoch_72.pt", checkpoint_epoch=72)
    write_tokens(fx, mutate=old)
    with pytest.raises(ValueError):
        gather(fx, compat(fx))


def test_same_shape_different_ckpt_sha_rejected(fx):
    write_tokens(fx)
    with pytest.raises(ValueError, match="显式最终 ckpt"):
        gather(fx, compat(fx), expected="e" * 64)
    other = "f" * 64
    write_tokens(fx, mutate=lambda k, m: m["encoder"].__setitem__("checkpoint_sha256", other))
    with pytest.raises(ValueError):
        gather(fx, compat(fx), expected=other)        # 目录里的 ckpt 字节不是该 SHA


@pytest.mark.parametrize("field,value", [("module_sha256", HIST_MODULE), ("encoder_src_sha256", "3" * 64)])
def test_wrong_encoder_source_sha_rejected(fx, field, value):
    write_tokens(fx, mutate=lambda k, m: m["encoder"].__setitem__(field, value))
    with pytest.raises(ValueError, match="SOURCE_PIN"):
        gather(fx, compat(fx))


def test_encoder_worker_mj_commit_rejected(fx):
    write_tokens(fx, mutate=lambda k, m: m["worker"].__setitem__("mj_commit", "a" * 40))
    with pytest.raises(ValueError, match="SOURCE_PIN"):
        gather(fx, compat(fx))


def test_encoder_state_mixed_rejected(fx):
    write_tokens(fx, mutate=lambda k, m: m.__setitem__("encoder_state_sha256", {"w": k}))
    with pytest.raises(ValueError, match="encoder_state_sha256"):
        gather(fx, compat(fx))


# ── 4. 行大小合同 ───────────────────────────────────────────────────────────────


def test_row_bytes_8192_positive_and_3072_negative(fx):
    blobs = write_tokens(fx)
    key = fx.keys[0]
    data, _ = pms.read_segment_tokens(fx.tokens, key, fx.rows[key])
    assert data == blobs[key] and len(data) == fx.rows[key] * 8192
    write_tokens(fx, row_shape=(768,))
    with pytest.raises(ValueError, match="8192|字节数"):
        pms.read_segment_tokens(fx.tokens, key, fx.rows[key])
    assert compare_wan.TOKEN_BYTES == 8192 and compare_wan.TOKEN_DIM == 2048


# ── 5. 完整 pack → compare → verify 顺序 ─────────────────────────────────────────


def pack_args(fx: Fx, out: pathlib.Path, **kw):
    base = dict(manifest=str(fx.manifest_path), tokens=str(fx.tokens), latents=str(fx.latents), out=str(out),
                encoder_run_dir=str(fx.enc_dir), raw_dir=fx.manifest["raw_dir"],
                oracle_report=str(fx.restore / "oracle/wan-mj/vae_report.json"), restore_receipt=str(fx.receipt),
                expected_ckpt_sha256=fx.ckpt_sha, resume=False, force_break_lock=False)
    base.update(kw)
    return SimpleNamespace(**base)


def write_oracle(fx: Fx, out: pathlib.Path, blobs: dict, *, order=None, flip=None, contract=None):
    """合成 oracle 参考：默认按清单序；order 可给出“按 GPU 分片拼接”的错误行序。"""
    out.mkdir(parents=True, exist_ok=True)
    keys = order or fx.keys
    table, row_map = bytearray(), []
    for key in keys:
        for m in range(fx.rows[key]):
            row = bytearray(blobs[key][m * 8192:(m + 1) * 8192])
            if flip == (key, m):
                row[0] ^= 1
            table += row
            row_map.append(dict(row=len(row_map), segment=key, m=m, g=0, seg="x"))
    (out / "motion_token.f32.bin").write_bytes(bytes(table))
    tok_meta = json.loads((fx.tokens / f"{fx.keys[0]}.metadata.json").read_text())
    prov = dict(tok_meta["encoder"], flags=tok_meta["encoder_flags"], module_sha256=PIN["upstream_source_sha256"])
    rep = dict(rows=len(row_map), table_sha256=sha(bytes(table)), row_map=row_map, encoder_provenance=prov,
               encoder_state_sha256=tok_meta["encoder_state_sha256"], orig_module_sha256=PIN["upstream_source_sha256"],
               token_contract=contract or compare_wan.TOKEN_CONTRACT, segments_filter=None,
               source_identity=dict(upstream_module_sha256=PIN["upstream_source_sha256"],
                                    adapter_module_sha256=PIN["source_sha256"], mj_install_commit=PIN["mj_repo_commit"]))
    (out / "encoder_report.json").write_text(json.dumps(rep))


@pytest.fixture
def packed(fx, monkeypatch):
    monkeypatch.setattr(pms, "load_vae_compat", functools.partial(pms.load_vae_compat, pins=fx.pins))
    blobs = write_tokens(fx)
    store = fx.root / "out/motion"
    pms.cmd_pack(pack_args(fx, store))
    return fx, store, blobs


def test_pack_row_order_from_manifest_only(packed):
    fx, store, blobs = packed
    meta = json.loads((store / ms.META_RELPATH).read_text())
    assert meta["status"] == "packed" and meta["tables"]["motion_token"]["row_bytes"] == 8192
    assert (store / ms.MOTION_TABLE_RELPATH).read_bytes() == b"".join(blobs[k] for k in fx.keys)
    assert meta["provenance"]["vae_compat"]["receipt_sha256"] == ms.sha256_file(fx.receipt)
    # 历史 latent metadata 原样不动（兼容记录只旁挂在新 store）
    assert "vae_compat" not in json.loads((fx.latents / f"{fx.keys[0]}.metadata.json").read_text())


def test_compare_before_pack_rejected(fx, tmp_path):
    with pytest.raises(SystemExit, match="pack"):
        compare_wan.cmd_tokens(SimpleNamespace(store=tmp_path / "nostore", oracle=tmp_path / "noracle"))


def test_compare_pass_then_verify_reaches_verified(packed):
    fx, store, blobs = packed
    orc = fx.root / "out/oracle/wan-mj"
    write_oracle(fx, orc, blobs)
    compare_wan.cmd_tokens(SimpleNamespace(store=store, oracle=orc))
    rec = json.loads((store / pms.BITEXACT_RELPATH).read_text())
    assert rec["status"] == "PASS" and rec["compared"] == sum(fx.rows.values())
    pms.cmd_verify(SimpleNamespace(store=str(store), manifest=None, tokens=None, resume=True, force_break_lock=False))
    assert json.loads((store / ms.META_RELPATH).read_text())["status"] == "verified"


def test_compare_failure_never_reaches_verified(packed, capsys):
    fx, store, blobs = packed
    orc = fx.root / "out/oracle/wan-mj"
    write_oracle(fx, orc, blobs, flip=(fx.keys[-1], 0))
    with pytest.raises(SystemExit):
        compare_wan.cmd_tokens(SimpleNamespace(store=store, oracle=orc))
    assert "ENCODER_BITEXACT=FAIL" in capsys.readouterr().out
    assert not (store / pms.BITEXACT_RELPATH).exists()
    with pytest.raises(RuntimeError, match="compare"):
        pms.cmd_verify(SimpleNamespace(store=str(store), manifest=None, tokens=None, resume=True, force_break_lock=False))
    assert json.loads((store / ms.META_RELPATH).read_text())["status"] == "packed"


def test_gpu_shard_concatenation_order_rejected(packed, capsys):
    fx, store, blobs = packed
    orc = fx.root / "out/oracle/wan-mj"
    gpu_order = fx.keys[1::2] + fx.keys[0::2]        # 两片按 GPU 编号直接拼接
    assert gpu_order != fx.keys
    write_oracle(fx, orc, blobs, order=gpu_order)
    with pytest.raises(SystemExit):
        compare_wan.cmd_tokens(SimpleNamespace(store=store, oracle=orc))
    assert "order_ok=0" in capsys.readouterr().out


def test_compare_rejects_768_contract_and_source_mixup(packed):
    fx, store, blobs = packed
    orc = fx.root / "out/oracle/wan-mj"
    old = dict(compare_wan.TOKEN_CONTRACT, upstream_shape=[768], row_elems=768, row_bytes=3072, row_shape=[768])
    write_oracle(fx, orc, blobs, contract=old)
    with pytest.raises(SystemExit, match="合同"):
        compare_wan.cmd_tokens(SimpleNamespace(store=store, oracle=orc))
    write_oracle(fx, orc, blobs)
    edit_json(orc / "encoder_report.json", lambda r: r.__setitem__("orig_module_sha256", PIN["source_sha256"]))
    with pytest.raises(SystemExit):
        compare_wan.cmd_tokens(SimpleNamespace(store=store, oracle=orc))


def test_stale_bitexact_record_rejected_after_repack(packed):
    fx, store, blobs = packed
    orc = fx.root / "out/oracle/wan-mj"
    write_oracle(fx, orc, blobs)
    compare_wan.cmd_tokens(SimpleNamespace(store=store, oracle=orc))
    edit_json(store / pms.BITEXACT_RELPATH, lambda r: r.__setitem__("store_table_sha256", "0" * 64))
    with pytest.raises(RuntimeError, match="不符"):
        pms.cmd_verify(SimpleNamespace(store=str(store), manifest=None, tokens=None, resume=True, force_break_lock=False))


def test_pack_precheck_happens_before_table_written(fx, monkeypatch):
    monkeypatch.setattr(pms, "load_vae_compat", functools.partial(pms.load_vae_compat, pins=fx.pins))
    write_tokens(fx)
    for key in fx.keys:
        edit_json(fx.latents / f"{key}.metadata.json", lambda m: m["vae"].__setitem__("module_sha256", "5" * 64))
    store = fx.root / "out/motion"
    with pytest.raises(ValueError, match="receipt"):
        pms.cmd_pack(pack_args(fx, store))
    assert not (store / ms.MOTION_TABLE_RELPATH).exists() and not (store / "motion_token.f32.bin.tmp").exists()


def test_a7_uses_layout_row_bytes(packed, capsys):
    fx, store, _ = packed
    import motion_checks
    motion_checks.cmd_a7(SimpleNamespace(manifest=str(fx.manifest_path), latents=str(fx.latents),
                                         tokens=str(fx.tokens), motion=str(store)))
    out = capsys.readouterr().out
    assert "A7_BYTES=PASS" in out and "row_bytes=8192" in out


# ── 6. oracle 聚合合同（独立解释器，oracle 禁止与 mme_vla_suite 同进程）────────────


def test_oracle_aggregate_rejects_smoke_filter_and_old_contract(fx, tmp_path):
    out = tmp_path / "oracle"
    out.mkdir()
    code = r"""
import json, sys, types, pathlib
sys.path.insert(0, 'scripts/dataset/wan')
import oracle_driver as o
out = pathlib.Path(sys.argv[1]); manifest = sys.argv[2]; case = sys.argv[3]
m = o.load_manifest(manifest)
segs = o.expected_segments(m, demo_min_real=17, exec_min_real=33)
for i in range(2):
    part = segs[i::2]; rows = []; blob = b''
    for s in part:
        for k in range(len(s['starts'])):
            rows.append({'row': len(rows), 'segment': s['key'], 'm': k, 'g': s['g'], 'seg': s['seg']}); blob += bytes(o.TOKEN_BYTES)
    (out / f'motion_token.shard{i}of2.f32.bin').write_bytes(blob)
    rep = {'rows': len(rows), 'table_sha256': o.sha_file(out / f'motion_token.shard{i}of2.f32.bin'), 'row_map': rows,
           'encoder_provenance': {}, 'encoder_state_sha256': {}, 'elapsed_s': 1, 'orig_module_sha256': 'u',
           'shard': {'idx': i, 'num': 2}, 'token_contract': dict(o.TOKEN_CONTRACT), 'source_identity': {},
           'segments_filter': None, 'manifest_sha256': m['sha256']}
    if case == 'filter' and i == 1: rep['segments_filter'] = [part[0]['key']]
    if case == 'old': rep['token_contract'] = dict(o.TOKEN_CONTRACT, row_bytes=3072)
    (out / f'encoder_report.shard{i}of2.json').write_text(json.dumps(rep))
args = types.SimpleNamespace(manifest=manifest, out=str(out), num_shards=2, kind='encoder', demo_min_real=17, exec_min_real=33)
try:
    o.cmd_aggregate(args); r = json.loads((out / 'encoder_report.json').read_text())
    print('AGG', r['rows'], r['token_contract']['row_bytes'], [x['segment'] for x in r['row_map']] == [s['key'] for s in segs for _ in s['starts']])
except SystemExit as e:
    print('REJECT', e)
"""
    results = {}
    for case in ("ok", "filter", "old"):
        p = subprocess.run([sys.executable, "-c", code, str(out), str(fx.manifest_path), case], cwd=ROOT,
                           capture_output=True, text=True, check=True)
        results[case] = p.stdout.strip().splitlines()[-1]
    assert results["ok"] == f"AGG {sum(fx.rows.values())} 8192 True"       # 片按取模分，聚合按清单序复原
    assert results["filter"].startswith("REJECT") and "冒烟" in results["filter"]
    assert results["old"].startswith("REJECT") and "合同" in results["old"]


# ── 7. 驱动守卫（假 Runner，不起子进程）──────────────────────────────────────────


class FakeRunner:
    def __init__(self, script):
        self.script, self.calls = script, []

    def run(self, tag, cmd, env, log):
        self.calls.append(tag)
        return self.script(tag, cmd)

    def run_parallel(self, jobs):
        return [self.run(t, c, e, lg) for t, c, e, lg in jobs]


def driver_args(fx: Fx, out: pathlib.Path, **kw):
    a = dict(restore_receipt=str(fx.receipt), encoder_ckpt=str(fx.enc_dir / CKPT_NAME), encoder_sha256=fx.ckpt_sha,
             out_store=str(out), gpus="0-1", max_segments=None, resume=False, parent_lib=str(fx.manifest_path.parents[1]),
             mj_repo=str(fx.root / "MotionJEPA"), require_free_mib=1)
    a.update(kw)
    return SimpleNamespace(**a)


def ok_identity(tag, cmd, epoch=60):
    if tag == "ckpt-identity":
        return 0, [f"CKPT_IDENTITY epoch={epoch} arch=wan-latent-v10-st mj_commit={PIN['mj_repo_commit']}"]
    raise AssertionError(f"意外调用 {tag}")


def make_build(fx, out, runner, **kw):
    expect = dict(bm.EXPECT, segments=len(fx.keys))
    return bm.Build(driver_args(fx, out, **kw), runner=runner, pins=fx.pins, expect=expect)


def test_driver_preflight_positive_and_byte_copies(fx):
    out = fx.root / "newstore"
    b = make_build(fx, out, FakeRunner(ok_identity))
    b.preflight()
    for n in ("episode_manifest.json", "input_manifest.json"):
        assert (out / "meta" / n).read_bytes() == (fx.manifest_path.parent / n).read_bytes()
        assert (out / "meta" / n).stat().st_nlink == 1
    assert json.loads((out / "build/state.json").read_text())["stages"]["preflight"]["status"] == "PASS"
    assert not (out / "framesamp-8x8").exists() and not (out / "source").exists()   # 父库直接引用，不复制


@pytest.mark.parametrize("bad", ["epoch59", "name", "sha", "relative"])
def test_driver_rejects_wrong_ckpt_identity(fx, bad):
    out = fx.root / "newstore"
    kw = {}
    script = ok_identity
    if bad == "epoch59":
        script = functools.partial(ok_identity, epoch=59)
    elif bad == "name":
        other = fx.enc_dir / "checkpoint_final.pt"
        other.write_bytes((fx.enc_dir / CKPT_NAME).read_bytes())
        kw["encoder_ckpt"] = str(other)
    elif bad == "sha":
        kw["encoder_sha256"] = "e" * 64
    else:
        kw["encoder_ckpt"] = "runs/final-e60/" + CKPT_NAME
    with pytest.raises(bm.BuildError, match="preflight"):
        make_build(fx, out, FakeRunner(script), **kw).preflight()


def test_driver_requires_fresh_output_and_same_identity_resume(fx):
    out = fx.root / "newstore"
    out.mkdir()
    (out / "stale.f32.bin").write_bytes(b"old token")
    with pytest.raises(bm.BuildError, match="全新"):
        make_build(fx, out, FakeRunner(ok_identity)).preflight()
    out2 = fx.root / "newstore2"
    make_build(fx, out2, FakeRunner(ok_identity)).preflight()
    make_build(fx, out2, FakeRunner(ok_identity), resume=True).preflight()      # 同身份续跑接受
    other = fx.enc_dir / "config.yaml"
    other.write_text(other.read_text() + "# changed\n")
    with pytest.raises(bm.BuildError, match="身份"):
        make_build(fx, out2, FakeRunner(ok_identity), resume=True).preflight()


def test_driver_first_run_requires_initial_complete_zero(fx):
    def script(tag, cmd):
        if tag == "encode":
            return 0, ["STAGE_START stage=encode initial_complete=2 items=3",
                       "STAGE_WORKSET=PASS stage=encode processed=1 initial_complete=2 duplicates=0 missing=0",
                       "STAGE_DONE stage=encode workers=2 items=3 skipped=0 initial_complete=2 elapsed=1s"]
        return ok_identity(tag, cmd)
    b = make_build(fx, fx.root / "newstore", FakeRunner(script))
    b.preflight()
    with pytest.raises(bm.BuildError, match="initial_complete=0"):
        b.encode()


def test_driver_compare_refused_before_pack_and_failure_stops_before_verify(fx, monkeypatch):
    out = fx.root / "newstore"
    seen = []

    def script(tag, cmd):
        seen.append(tag)
        if tag == "compare":
            return 1, ["ENCODER_BITEXACT=FAIL compared=5 mismatches=1 order_ok=1"]
        return ok_identity(tag, cmd)
    b = make_build(fx, out, FakeRunner(script))
    b.preflight()
    with pytest.raises(bm.BuildError, match="pack"):
        b.compare()                                   # 尚无 packed store
    (b.store / "meta").mkdir(parents=True)
    (b.store / ms.META_RELPATH).write_text(json.dumps({"status": "packed"}))
    for stage in ("encode", "oracle", "pack"):
        b.mark(stage, "PASS", "测试")
    b.args.resume = True                              # 同身份续跑，从 compare 接着走
    monkeypatch.setattr(b, "checks", lambda: seen.append("checks"))
    monkeypatch.setattr(b, "verify", lambda: seen.append("verify"))
    with pytest.raises(bm.BuildError, match="compare"):
        b.run()
    assert "verify" not in seen and "checks" not in seen


def test_driver_row_order_is_manifest_order(fx):
    keys = bm.ordered_segment_keys(fx.manifest)
    assert keys == fx.keys == ["T_ep0_demo", "T_ep0_exec", "T_ep1_exec"]
    assert bm.parse_gpus("0-7") == list(range(8)) and bm.parse_gpus("3,1") == [3, 1]
    with pytest.raises(SystemExit):
        bm.parse_gpus("0,0")


def test_driver_has_no_skip_switches():
    text = (ROOT / "scripts/dataset/build_mixed8_v10_motion.py").read_text(encoding="utf-8")
    opts = [a.option_strings for a in bm.build_parser()._actions]
    flat = {o for group in opts for o in group}
    assert not any("skip" in o.lower() or "allow" in o.lower() or "force" in o.lower() for o in flat)
    assert "SKIP" not in text.split("def build_parser")[1]
