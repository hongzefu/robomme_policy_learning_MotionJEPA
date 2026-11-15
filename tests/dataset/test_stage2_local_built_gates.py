"""1114 计划 S2：4task local-built 的 G0 生产入口、G2/G3 生产端与消费端、MOTION_XPATH 对齐（CPU）。"""
import copy
import importlib.util
import json
import pathlib
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dataset"))
sys.path.insert(0, str(ROOT / "scripts/dataset/wan_stage2"))
sys.path.insert(0, str(ROOT / "tests/dataset"))
import check_stage2_validation_inputs as g0
import local_built_fixture as LB
import validate_stage2 as V
from mme_vla_suite.datastore import motion_store as ms


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


X = load("_t_xpath", "scripts/dataset/check_motion_xpath.py")
NUMERICS = dict(dtype="float32", autocast=False, matmul_allow_tf32=False, cudnn_allow_tf32=False, batch=1,
                torch="2.9.0+cu128", cuda="12.8", cudnn=91002, numpy="2.4.4",
                gpu_name="NVIDIA A100-SXM4-80GB", driver="595.71.05")


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return LB.build(tmp_path_factory.mktemp("lib"))


@pytest.fixture(scope="module")
def g0_record(lib, tmp_path_factory):
    out = tmp_path_factory.mktemp("g0") / "windows.json"
    g0.main(["--manifest", str(lib.manifest_path), "--latents", str(lib.latents), "--out", str(out),
             "--stage2-project", "wan_stage2_rgbstate", "--episode-maps", str(lib.maps),
             "--merge-done", str(lib.merge_done), "--member-h5-root", str(lib.raw)])
    return out


def test_local_g0_produces_4x16_and_passes(lib, g0_record, capsys):
    record = json.loads(g0_record.read_text())
    assert record["kind"] == ms.STAGE2_LOCAL_ORIGIN and record["strategy"] == "task4x16-endpoints-pad8-seed0-v1"
    counts = record["counts"]
    assert counts["windows"] == 64 and counts["tasks"] == 4 and counts["demo_tail"] >= 8
    assert ms.validate_local_g0(record) == counts["demo_tail"]
    assert record["latent_metadata_sha256"] == lib.total_sha
    window = next(w for w in record["windows"] if w["task"] == "RouteStick")
    source = window["sources"][0]
    assert source["kind"] == "member_h5" and source["raw_episode"] == window["episode"] + 2
    assert window["key"] == f"RouteStick_ep{window['episode']}_{window['segment_kind']}"
    # 消费端（sidecar 环境的 validate_stage2）同一校验、按契约②读回原帧并核 33 帧 sha。
    samples = V.load_samples(g0_record)
    frames = V.read_window_frames(samples["windows"][0])
    assert frames.shape == (33, 256, 256, 3)
    assert len(V.read_latent(samples["windows"][0])) == 9


def test_local_g0_rejects_frame_sha_mismatch_and_extracts_missing(lib, tmp_path):
    meta_path = lib.latents / "BinFill_ep0_exec.metadata.json"
    original = meta_path.read_text()
    try:
        meta = json.loads(original)
        for row in meta["rows"]:
            row["input_frames_sha256"] = "0" * 64
        meta_path.write_text(json.dumps(meta))
        total = json.loads((lib.latents / "metadata.json").read_text())
        saved = copy.deepcopy(total)
        total["segments"]["BinFill_ep0_exec"]["rows"] = meta["rows"]
        (lib.latents / "metadata.json").write_text(json.dumps(total))
        with pytest.raises(ValueError, match="原帧 SHA"):
            g0.build_local_inputs(lib.manifest_path, lib.latents, episode_maps=lib.maps, merge_done=lib.merge_done,
                                  member_roots=[lib.raw])
    finally:
        meta_path.write_text(original)
        (lib.latents / "metadata.json").write_text(json.dumps(saved))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="tar"):
        g0.build_local_inputs(lib.manifest_path, lib.latents, episode_maps=lib.maps, merge_done=lib.merge_done,
                              member_roots=[empty])


def test_16task_g0_with_4task_fails(lib, g0_record):
    with pytest.raises(ValueError, match="任务数"):
        g0.select_windows(lib.manifest, lib.latents)          # 旧 16 任务×4 窗口径配 4task 库
    record = json.loads(g0_record.read_text())
    old_style = dict(record, kind=None, strategy=None, counts=dict(record["counts"], tasks=16))
    with pytest.raises(ValueError):
        ms.validate_local_g0(old_style)
    sixteen = copy.deepcopy(record)
    for i, window in enumerate(sixteen["windows"]):
        window["task"] = f"T{i // 4}"
        window["key"] = f"T{i // 4}_ep{window['episode']}_{window['segment_kind']}"
    sixteen["counts"]["tasks"] = 16
    with pytest.raises(ValueError):
        ms.validate_local_g0(sixteen)


# ─────────────────────────────── G2：生产端 gate_g2_local 与消费端同一口径

def g2_fixture(g0_path, noise):
    samples = V.load_samples(g0_path)
    import torch

    class Adapter:
        VAE_STATE_SHA256_EXPECTED = "9" * 64
        __file__ = str(ROOT / "scripts/dataset/wan/wan_motion_infer.py")

        def load_vae(self, vae_id, device, expected_state_sha256=None):
            vae = SimpleNamespace(config=SimpleNamespace(latents_std=[1.0] * 16))
            return vae, dict(vae_id=vae_id, vae_state_sha256=expected_state_sha256, vae_dtype="float32",
                             latent_mode="mode", batch=1, tf32="off", amp="off")

        def pin_numerics(self):
            pass

        def encode_chunk(self, vae, frames, device):
            # 在线路径输出 = 旧库 fp32 latent（按帧 sha 找回所在窗）+ 给定误差
            window = next(w for w in samples["windows"] if V.sha_bytes(frames.tobytes()) == w["input_frames_sha256"])
            return torch.from_numpy(V.read_latent(window) + noise)

    info = dict(encoder=dict(arch=ms.STAGE2_RGBSTATE_ARCH, checkpoint_sha256="c" * 64), numerics=NUMERICS)
    args = SimpleNamespace(device="cpu", stage2_project="wan_stage2_rgbstate")
    return samples, args, (torch, None, Adapter(), None, info)


def g2_record(g0_path, samples, result):
    return dict(schema="stage2-validation-gate-v1", gate="G2", status="PASS", producer=V.PRODUCER,
                checkpoint_sha256="c" * 64, samples_sha256=V.sha_file(g0_path), **result)


def test_g2_producer_and_consumer_agree(g0_record):
    samples, args, runtime = g2_fixture(g0_record, noise=1e-5)
    result = V.gate_g2(args, samples, runtime)
    assert result["kind"] == ms.STAGE2_LOCAL_ORIGIN and result["comparison"] == ms.LOCAL_G2_COMPARISON
    assert result["metrics"]["windows"] == 64 and result["metrics"]["worst_max"] <= ms.STAGE2_G2_MAX_ABS
    record = g2_record(g0_record, samples, result)
    pin = V.project_pin(args)
    status = ms.validate_stage2_g2_evidence(
        record, checkpoint_sha256="c" * 64, samples_sha256=V.sha_file(g0_record), source_pin=pin,
        numerics=NUMERICS, online_vae=result["online_vae"], demo_tail=samples["counts"]["demo_tail"], local=True)
    assert status == "PASS"
    # 消费端：超门槛的证据、非 local 证据、非 PASS 一律拒。
    for mutate in (lambda r: r["metrics"].update(worst_max=0.02), lambda r: r.pop("kind"),
                   lambda r: r.update(comparison="fp16-full33"), lambda r: r.update(status="WAIVED")):
        bad = copy.deepcopy(record)
        mutate(bad)
        with pytest.raises(ValueError):
            ms.validate_stage2_g2_evidence(
                bad, checkpoint_sha256="c" * 64, samples_sha256=V.sha_file(g0_record), source_pin=pin,
                numerics=NUMERICS, online_vae=result["online_vae"], local=True)


def test_g2_online_error_over_threshold_fails(g0_record):
    samples, args, runtime = g2_fixture(g0_record, noise=0.05)
    with pytest.raises(ValueError, match="G2 超阈值"):
        V.gate_g2(args, samples, runtime)


# ─────────────────────────────── G3：rgb-state 掩码分派

class FakeEncoder:
    def __init__(self, torch):
        self.torch = torch

    def __call__(self, latent, mask):
        return latent.reshape(1, -1)[:, :1024].reshape(1, 4, 256).clone()


def g3_setup(monkeypatch, tmp_path, arch):
    import torch
    rng = np.random.default_rng(3)
    keys = [f"W{i:03d}" for i in range(40)]
    common = np.ones((256, 256), dtype=bool)

    class Reader:
        def __init__(self, root, profile=None):
            self.profile = profile
            self.windows = {k: dict(split="val") for k in keys}

        def read_pair(self, key, aug):
            pair = dict(window=dict(key=key, split="val"), params=dict(aug_id=0), common_valid=common,
                        rgb_raw=rng.standard_normal((9, 16, 32, 32)).astype(np.float16))
            if self.profile is None:
                pair["xyz_valid"] = np.ones((33, 256, 256), dtype=bool)
            return pair

    data_module = SimpleNamespace(RGBSTATE_PROFILE="rgbstate-4task", __file__=__file__,
                                  derive_rgb_geometry_masks=lambda c: dict(geometry_valid=np.ones((32, 32), bool)),
                                  derive_geometry_masks=lambda c, x: dict(geometry_valid=np.ones((32, 32), bool)))
    reader_module = SimpleNamespace(PairedWanCacheReader=Reader, __file__=__file__)
    monkeypatch.setattr(V, "load_pinned_mj_source",
                        lambda root, rel, name, pin: reader_module if "reader" in rel else data_module)
    monkeypatch.setattr(V, "assert_ema_state", lambda enc, payload, leaves=65: dict(
        leaves=leaves, state_key="model", exact=1, live_negative_rejected=1))
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"arch": arch}, ckpt)
    encoder = FakeEncoder(torch)
    memory = SimpleNamespace(encode_windows=lambda enc, latent: enc(latent, None).reshape(1, 1024))
    info = dict(encoder=dict(arch=arch), encoder_state_sha256={"sha256": "1" * 64}, numerics=NUMERICS,
                source_pin={})
    project = "wan_stage2_rgbstate" if arch == ms.STAGE2_RGBSTATE_ARCH else "wan_stage2"
    args = SimpleNamespace(ckpt=str(ckpt), mj_root=str(tmp_path), rgb_root=str(tmp_path), device="cpu",
                           cache_import_receipt=None, stage2_project=project)
    return args, (torch, memory, None, encoder, info)


def test_g3_rgbstate_without_xyz_valid_passes(monkeypatch, tmp_path, g0_record):
    args, runtime = g3_setup(monkeypatch, tmp_path, ms.STAGE2_RGBSTATE_ARCH)
    samples = V.load_samples(g0_record)
    result = V.gate_g3(args, samples, runtime)
    assert result["metrics"]["windows"] == 32 and result["metrics"]["geometry_all_true"] == 1
    assert result["kind"] == ms.STAGE2_LOCAL_ORIGIN and result["mask_profile"] == "rgb-state"
    assert result["metrics"]["leaves"] == ms.stage2_expected_leaves(ms.STAGE2_RGBSTATE_ARCH) == 65


def test_g3_three_decoder_path_unchanged(monkeypatch, tmp_path, g0_record):
    args, runtime = g3_setup(monkeypatch, tmp_path, ms.STAGE2_ARCH)
    samples = V.load_samples(g0_record)
    with pytest.raises(ValueError, match="rgb-state"):
        V.gate_g3(args, samples, runtime)                  # local-built 只能配 rgb-state
    samples = dict(samples, kind=None)
    result = V.gate_g3(args, samples, runtime)             # 旧三路仍用双流 xyz_valid 掩码
    assert "kind" not in result and result["metrics"]["max_abs"] == 0


def test_validate_main_infers_project_and_rejects_mismatch(monkeypatch, tmp_path, g0_record):
    import torch
    ckpt = tmp_path / "rgb.pt"
    torch.save({"arch": ms.STAGE2_RGBSTATE_ARCH}, ckpt)
    out = tmp_path / "G2.json"
    with pytest.raises(ValueError, match="不符"):
        V.main(["--gate", "G2", "--samples", str(g0_record), "--ckpt", str(ckpt), "--expected-sha256",
                V.sha_file(ckpt), "--out", str(out), "--stage2-project", "wan_stage2"])
    assert json.loads(out.read_text())["status"] == "FAIL"


# ─────────────────────────────── MOTION_XPATH 对齐（C04）

def xpath_inputs(lib):
    records = X.member_module().load_member_records(lib.maps)
    windows, episodes = [], []
    cells = {("BinFill", "exec"): 12, ("RouteStick", "demo"): 10, ("RouteStick", "exec"): 10,
             ("VideoRepick", "demo"): 10, ("VideoRepick", "exec"): 10, ("VideoUnmaskSwap", "demo"): 10,
             ("VideoUnmaskSwap", "exec"): 10}
    for (task, ep), record in records.items():
        episodes.append(dict(task=task, episode=ep, raw_episode=record.raw_episode, member=record.member))
    wid = 0
    for (task, kind), n in cells.items():
        for i in range(n):
            ep, block = i % LB.EPISODES, 0 if i < LB.EPISODES else 2
            start = block * 16
            meta = json.loads((lib.latents / f"{task}_ep{ep}_{kind}.metadata.json").read_text())
            row = meta["rows"][block]
            ids = list(range(row["start_global_frame"], row["start_global_frame"] + row["real_frames"]))
            ids += [row["pad_source_frame"]] * row["pad_frames"]
            if i >= 2 * LB.EPISODES:
                start = 16                   # 第 2 块（非偶）也合法对齐
                row = meta["rows"][1]
                ids = list(range(row["start_global_frame"], row["start_global_frame"] + row["real_frames"]))
                ids += [row["pad_source_frame"]] * row["pad_frames"]
            windows.append(dict(window_key=f"{task}/ep{ep:03d}/{kind}/s{start:04d}/w{wid}", task=task, episode=ep,
                                kind=kind, start=start, source_frame_ids=ids, real_frames=row["real_frames"],
                                pad_frames=row["pad_frames"], is_tail=row["pad_frames"] > 0, split="val",
                                row_count=1, parity_aug_id=-1))
            wid += 1
    # 过滤负例：train、多行、非 16 倍起点。
    windows.append(dict(windows[0], window_key="train", split="train"))
    windows.append(dict(windows[0], window_key="multi", row_count=16))
    windows.append(dict(windows[0], window_key="odd", start=8))
    return windows, episodes, records


def test_xpath_filter_and_stratified_64(lib):
    windows, episodes, records = xpath_inputs(lib)
    common = X.common_windows(windows)
    assert len(common) == 72
    sample = X.stratified_sample(common, seed=0)
    assert len(sample) == 64
    cells = {}
    for w in sample:
        cells[(w["task"], w["kind"])] = cells.get((w["task"], w["kind"]), 0) + 1
    first = sorted(cells)[0]
    assert cells[first] == 10 and all(v == 9 for k, v in cells.items() if k != first)
    assert [w["window_key"] for w in X.stratified_sample(common, seed=0)] == [w["window_key"] for w in sample]
    plan = X.plan_alignment(windows, episodes, records, lib.latents)
    assert plan["common"] == 72 and plan["sampled"] == 64 and plan["excluded"] == 0
    assert len(plan["aligned"]) == 64


@pytest.mark.parametrize("fault,reason", [("map", "member_map_mismatch"), ("tail_pad", "real_pad_mismatch"),
                                          ("cross_segment", "frame_ids_mismatch"), ("frames", "frame_ids_mismatch"),
                                          ("no_cache", "no_cache")])
def test_xpath_negatives_excluded_by_reason(lib, fault, reason):
    windows, episodes, records = xpath_inputs(lib)
    sample_keys = {w["window_key"] for w in X.stratified_sample(X.common_windows(windows), seed=0)}
    target = next(w for w in windows if w["window_key"] in sample_keys and w["task"] == "RouteStick"
                  and w["kind"] == "demo" and w["start"] == 0)
    if fault == "map":
        # 1113 索引把 raw_episode 当成统一编号（raw_episode≠episode 映射错）
        row = next(e for e in episodes if (e["task"], e["episode"]) == (target["task"], target["episode"]))
        row["raw_episode"] = row["episode"]
    elif fault == "tail_pad":
        target.update(real_frames=17, pad_frames=16)
    elif fault == "cross_segment":
        target["source_frame_ids"] = list(range(LB.ES - 16, LB.ES + 17))     # 跨 demo/exec 段
    elif fault == "frames":
        target["source_frame_ids"] = [f + 1 for f in target["source_frame_ids"]]
    else:
        target["start"] = 48                                                 # 超出段内块数
    plan = X.plan_alignment(windows, episodes, records, lib.latents)
    assert plan["excluded_by_reason"][reason] >= 1 and plan["sampled"] == 64
    assert plan["excluded"] == sum(plan["excluded_by_reason"].values())
    assert len(plan["aligned"]) == 64 - plan["excluded"]                     # 排除后不补抽


def test_xpath_compare_vectors():
    b = np.linspace(-1, 1, 1024, dtype=np.float32)
    stats = X.compare_vectors(b + 0.01, b)
    assert abs(stats["max_abs"] - 0.01) < 1e-6 and stats["mean_abs"] > 0 and stats["max_rel"] > 0
    with pytest.raises(ValueError):
        X.compare_vectors(np.zeros(1023), np.zeros(1023))
