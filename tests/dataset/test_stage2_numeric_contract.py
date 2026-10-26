"""数值闸门的 CPU 合同测试；不运行真实 VAE、GPU 或正式全表。"""

from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "scripts/dataset/wan_stage2"
sys.path.insert(0, str(PROJECT))
import validate_stage2 as V
import compare_stage2 as C

spec = importlib.util.spec_from_file_location("_test_stage2_online_numeric", ROOT / "scripts/training/check_stage2_online.py")
O = importlib.util.module_from_spec(spec)
spec.loader.exec_module(O)


def window(tmp_path, kind="demo"):
    real = 17 if kind == "demo" else 33
    pixels = np.arange(real, dtype=np.uint8)[:, None, None, None] * np.ones((real, 256, 256, 3), np.uint8)
    sources, indices = [], []
    if kind == "demo":
        path = tmp_path / "video_demo.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("frames", data=pixels, compression="gzip")
        sources.append(dict(path=str(path), kind="demo_h5", sha256=V.sha_file(path)))
        indices = [dict(source=0, index=i) for i in range(real)] + [dict(source=0, index=16)] * 16
    else:
        for i in range(real):
            path = tmp_path / f"{i}.pkl"
            path.write_bytes(pickle.dumps(dict(image=pixels[i])))
            sources.append(dict(path=str(path), kind="exec_pkl", sha256=V.sha_file(path)))
            indices.append(dict(source=i, index=0))
    padded = np.concatenate([pixels, np.repeat(pixels[-1:], 33 - real, axis=0)])
    latent = np.zeros(V.LATENT_SHAPE, np.float32)
    path = tmp_path / "latent.bin"
    path.write_bytes(latent.tobytes())
    return dict(key="Task_ep0_" + kind, task="Task", episode=0, segment_kind=kind, source_grid=0,
                real_frames=real, pad_frames=33-real, pad_source_frame=16 if kind == "demo" else None,
                frame_ids=list(range(real)) + [real - 1] * (33-real), sources=sources,
                source_frame_indices=indices, input_frames_sha256=V.sha_bytes(padded.tobytes()),
                latent_path=str(path), latent_block_sha256=V.sha_file(path)), padded


@pytest.mark.parametrize("kind", ["demo", "exec"])
def test_g0_rebuilds_exact_pixels_and_rejects_source_change(tmp_path, kind):
    item, pixels = window(tmp_path, kind)
    assert np.array_equal(V.read_window_frames(item), pixels)
    assert V.read_latent(item).shape == V.LATENT_SHAPE
    Path(item["sources"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA"):
        V.read_window_frames(item)


@pytest.mark.parametrize("field,value", [("source_grid", 1), ("pad_source_frame", 0), ("real_frames", 16)])
def test_window_rejects_wrong_grid_padding(tmp_path, field, value):
    item, _ = window(tmp_path)
    item[field] = value
    with pytest.raises(ValueError):
        V.validate_window(item)


def test_pixel_hash_catches_wrong_source_mapping(tmp_path):
    item, _ = window(tmp_path)
    item["source_frame_indices"][0]["index"] = 1
    with pytest.raises(ValueError, match="像素 SHA"):
        V.read_window_frames(item)


def test_g0_requires_all_64_distinct_windows_and_fixed_task_counts(tmp_path):
    template, _ = window(tmp_path)
    windows = []
    for task in range(16):
        for i in range(4):
            item = copy.deepcopy(template)
            item.update(task=f"Task{task}", key=f"Task{task}_ep{i}_demo", episode=i)
            windows.append(item)
    data = dict(schema="stage2-validation-inputs-v1", gate="G0", status="PASS", seed=0,
                windows=windows, counts=dict(windows=64, tasks=16, demo_tail=64, sha_mismatches=0),
                manifest_sha256="a" * 64, latent_metadata_sha256="b" * 64)
    path = tmp_path / "windows.json"
    V.atomic_json(path, data)
    assert len(V.load_samples(path)["windows"]) == 64
    data["windows"][-1] = data["windows"][0]
    V.atomic_json(path, data)
    with pytest.raises(ValueError):
        V.load_samples(path)


def test_normalization_is_per_channel_and_rejects_nan_or_bad_std():
    expected = np.zeros(V.LATENT_SHAPE, np.float32)
    std = np.arange(1, 17, dtype=np.float64)
    actual = np.broadcast_to(std[None, :, None, None] * 0.001, V.LATENT_SHAPE)
    mse, maximum = V.normalized_error(actual, expected, std)
    assert mse == pytest.approx(1e-6)
    assert maximum == pytest.approx(0.001)
    bad = expected.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        V.normalized_error(bad, expected, std)
    with pytest.raises(ValueError):
        V.normalized_error(expected, expected, np.zeros(16))


def comparison_fixture(tmp_path):
    latents = np.stack([np.full(V.LATENT_SHAPE, i, np.float32) for i in range(3)])
    path = tmp_path / "Task_ep0_demo.bin"
    path.write_bytes(latents.tobytes())
    digest = V.sha_file(path)
    V.atomic_json(path.with_suffix(".metadata.json"), dict(num_grid=3, bytes=3*V.CHUNK_BYTES, sha256=digest))
    table = np.stack([np.full(1024, i, np.float32) for i in (0, 2)])
    calls = []

    def encode(value):
        calls.append(value[0, 0, 0, 0])
        return np.full(1024, value[0, 0, 0, 0], np.float32)

    return dict(table=table, segments=[("Task_ep0_demo", 0, 2)], encode=encode, latent_root=tmp_path,
                root_metadata=dict(segments={"Task_ep0_demo": dict(sha256=digest)})), calls


def test_full_compare_recomputes_every_even_source_grid(tmp_path):
    arguments, calls = comparison_fixture(tmp_path)
    compared, mapping = C.compare_rows(**arguments)
    assert compared == 2 and mapping == {"Task_ep0_demo": [0, 2]}
    assert calls == [0, 2]


def test_compare_detects_changed_unselected_odd_source_block(tmp_path):
    arguments, _ = comparison_fixture(tmp_path)
    original = arguments["encode"]

    def change_odd_block(value):
        if value[0,0,0,0] == 0:
            with (tmp_path / "Task_ep0_demo.bin").open("r+b") as stream:
                stream.seek(V.CHUNK_BYTES)
                stream.write(np.float32(9).tobytes())
        return original(value)

    arguments["encode"] = change_odd_block
    with pytest.raises(ValueError, match="期间"):
        C.compare_rows(**arguments)


@pytest.mark.parametrize("damage", ["one_bit", "signed_zero", "nan", "truncated", "source_sha", "gap", "missing_row"])
def test_full_compare_refuses_incomplete_or_nonexact_table(tmp_path, damage):
    arguments, _ = comparison_fixture(tmp_path)
    if damage == "one_bit":
        arguments["table"].view(np.uint32)[1, 0] ^= 1
    elif damage == "signed_zero":
        arguments["table"][0, 0] = -0.0
    elif damage == "nan":
        arguments["table"][0, 0] = np.nan
    elif damage == "truncated":
        (tmp_path / "Task_ep0_demo.bin").write_bytes(b"short")
    elif damage == "source_sha":
        arguments["root_metadata"]["segments"]["Task_ep0_demo"]["sha256"] = "0" * 64
    elif damage == "gap":
        arguments["segments"] = [("Task_ep0_demo", 1, 2)]
    else:
        arguments["table"] = np.concatenate([arguments["table"], np.zeros((1, 1024), np.float32)])
    with pytest.raises(ValueError):
        C.compare_rows(**arguments)


def test_failed_certification_revokes_old_pass(tmp_path, monkeypatch):
    from mme_vla_suite.datastore import motion_store as ms
    (tmp_path / "meta").mkdir()
    record = tmp_path / "meta/encoder_bitexact.json"
    V.atomic_json(record, dict(schema=C.SCHEMA, status="PASS"))
    monkeypatch.setattr(ms.MotionMeta, "load", lambda root: (_ for _ in ()).throw(ValueError("bad meta")))
    with pytest.raises(ValueError, match="bad meta"):
        C.certify(SimpleNamespace(store=tmp_path, scope="full", batch=1))
    assert not record.exists()
    assert not (tmp_path / "meta/encoder_bitexact.lock").exists()


def test_failed_g2_replaces_previous_pass_with_fail(tmp_path):
    path = tmp_path / "G2.json"
    V.atomic_json(path, dict(schema="stage2-validation-gate-v1", gate="G2", status="PASS"))
    with pytest.raises(FileNotFoundError):
        V.main(["--gate", "G2", "--samples", str(tmp_path / "missing.json"), "--ckpt", str(tmp_path / "missing.pt"),
                "--expected-sha256", "a" * 64, "--out", str(path)])
    assert V.read_json(path)["status"] == "FAIL"


def test_gpu_handshake_checks_uuid_not_just_environment(monkeypatch):
    monkeypatch.setattr(O.subprocess, "check_output", lambda *a, **kw: "0, GPU-abc, NVIDIA A100-SXM4-80GB\n")
    assert O.verify_gpu(dict(cuda_visible_devices="0", gpu_uuid="GPU-abc"), "0")["index"] == "0"
    with pytest.raises(ValueError):
        O.verify_gpu(dict(cuda_visible_devices="0", gpu_uuid="GPU-other"), "0")


def test_stub_never_satisfies_real_handshake():
    client = SimpleNamespace(alive=True, stub=True, provenance={})
    with pytest.raises(ValueError, match="stub"):
        O.require_handshake(client, stub=False)


def test_actual_memory_state_machine_with_frame_ids():
    from mme_vla_suite.policies import motion_protocol as protocol

    class Client:
        alive = True

        def __call__(self, pixels, start):
            assert protocol.stub_decode(pixels)[0] == start
            return np.full(1024, start, np.float32)

    config = dict(stride=32, window_frames=33, budget=96, frame_size=256, pos_dim=256, dim=1024,
                  demo_min_real_frames=17, demo_tail_pad="repeat_last")
    result = O.check_state_machine(Client(), config)
    assert result["frame_ids_mismatch"] == result["state_machine_mismatch"] == 0
    assert result["stub_windows"] == 6 and result["reset"] == 1


def test_torch_ema_and_real_geometry_in_isolated_cpu_environment():
    """主测试环境不import torch；65叶原张量与正式geometry函数在MJ环境运行。"""
    python = Path(os.environ.get("STAGE2_TEST_TORCH_PYTHON", "/scratch/hongze/MotionJEPA/.venv/bin/python"))
    if not python.is_file():
        pytest.skip("需要显式 STAGE2_TEST_TORCH_PYTHON 才运行隔离torch CPU合同")
    code = r'''
import copy, importlib.util, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, sys.argv[1])
import validate_stage2 as V
class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(65):
            self.register_buffer('leaf'+str(i), torch.tensor([float(i)], dtype=torch.float32))
encoder = Encoder()
ema = {'encoder.'+k:v.clone() for k,v in encoder.state_dict().items()}
live = copy.deepcopy(ema)
live['encoder.leaf0'][0] = 1
assert V.assert_ema_state(encoder, dict(model=ema,model_live=live))['live_negative_rejected'] == 1
for payload in (dict(model=live,model_live=ema), dict(model=ema,model_live=ema)):
    try: V.assert_ema_state(encoder,payload)
    except ValueError: pass
    else: raise AssertionError('错误EMA或无效负控被接受')
from motion_jepa.data.dataset_paired_wan import derive_geometry_masks
common=np.ones((256,256),bool); xyz=np.ones((33,256,256),bool)
mask=derive_geometry_masks(common,xyz)['geometry_valid']
assert mask.shape == (32,32) and mask.all()
common[15,23]=False
mask=derive_geometry_masks(common,xyz)['geometry_valid']
assert not mask[1,2] and mask.sum()==1023
print('EMA_GEOMETRY_CPU=PASS leaves=65 live_negative_rejected=1')
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH="/scratch/hongze/worktrees/vla-s2-memexport-20261026/src")
    result = subprocess.run([str(python), "-c", code, str(PROJECT)], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EMA_GEOMETRY_CPU=PASS" in result.stdout


def test_portable_restore_uses_real_state_rows_and_rejects_drift(tmp_path, monkeypatch):
    from mme_vla_suite.datastore import framesamp_store as fs
    from mme_vla_suite.datastore import manifest as manifest_module
    dataset = tmp_path / "dataset"
    frame_root = dataset / "framesamp-8x8"
    source = dataset / "source"
    (dataset / "meta").mkdir(parents=True)
    (frame_root / "meta").mkdir(parents=True)
    (source / "data").mkdir(parents=True)
    manifest_path = dataset / "meta/episode_manifest.json"
    manifest_path.write_text("{}")
    (frame_root / fs.META_RELPATH).write_text("{}")
    episode = dict(h5_file="record_dataset_Task.h5", raw_ep_idx=0, global_episode_idx=0,
                   total_sample_offset=0, exec_sample_offset=0, exec_start_idx=17, num_timesteps=82)
    monkeypatch.setattr(manifest_module, "load_manifest", lambda path: dict(episodes=[episode]))
    states = np.arange(82 * 8, dtype=np.float32).reshape(82, 8)
    state_path = frame_root / fs.STATE_TABLE_RELPATH
    state_path.write_bytes(states.tobytes())
    frame_meta = SimpleNamespace(status="verified", manifest_sha256="a" * 64, manifest_scope="full", num_rows=82,
                                 source_dataset_root=str(source), raw=dict(tables={fs.STATE_KEY:
                                 dict(sha256=V.sha_file(state_path), row_shape=[8], dtype="float32")}))
    monkeypatch.setattr(fs.StoreMeta, "load", lambda root: frame_meta)
    item, _ = window(tmp_path)
    for i in range(65):
        payload = dict(epis_idx=np.array([0], np.int32), step_idx=np.array([17+i], np.int32),
                       exec_start_idx=np.array([17], np.int32), is_demo=np.array([False], bool),
                       image=np.full((256,256,3),i,np.uint8), wrist_image=np.full((256,256,3),7,np.uint8),
                       state=states[17+i], prompt="执行夹具任务")
        (source / "data" / f"{i}.pkl").write_bytes(pickle.dumps(payload))
    tokenizer = tmp_path / "models/big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_bytes(b"synthetic-test-tokenizer")
    meta = SimpleNamespace(manifest_path=str(manifest_path), manifest_sha256="a" * 64)
    out = O.prepare_restore_inputs(meta, dict(windows=[item]), tmp_path, tmp_path)
    record = V.read_json(out["path"])
    assert record["exec_start_idx"] == 17 and record["exec_batch_lengths"] == [33,32]
    with np.load(out["payload"]["path"], allow_pickle=False) as payload:
        assert np.array_equal(payload["states"], states)
        assert payload["images"].shape == (82,256,256,3)
    bad = source / "data/0.pkl"
    payload = pickle.loads(bad.read_bytes())
    payload["state"] = np.zeros(8,np.float32)
    bad.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="state"):
        O.prepare_restore_inputs(meta, dict(windows=[item]), tmp_path, tmp_path)
