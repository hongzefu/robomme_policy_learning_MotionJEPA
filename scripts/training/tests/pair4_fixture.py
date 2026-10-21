"""四对训练冒烟的真实 33 帧监督视图；不截断父样本的未来动作。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MAIN_REPO = Path("/scratch/hongze/robomme_policy_learning_MotionJEPA")
PARENT = MAIN_REPO / "v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f"
PARENT_MANIFEST_SHA = "d6a3ddb2b579b5cf50cd66fbefedd36478766e2b4cc07833a7efae29769742ad"
PARENT_H5 = Path("/scratch/hongze/dataset-local-a100-16task-mixed8-200/data-raw/BinFill_ep0/video_exec.h5")
RUN_NAME = "wan-mixed8-200-v10-inseq256q8-b352x8-trend-20261020-a"
ORIGINAL_RUN = Path("/scratch/hongze/MotionJEPA/runs") / RUN_NAME
REGISTERED_RUN = MAIN_REPO / "v1-store/external/motionjepa" / RUN_NAME
CHECKPOINT = "checkpoint_epoch_6.pt"
CHECKPOINT_SHA = "146efd425f7923428d630dfb7ee2a8f010f19b207d36519dba49454aaa373d30"
LAYOUT = "motion-v10-8x256-grid16-demopad17-v1"
SEGMENT = "BinFill_ep0_exec"
H5_NAME = "record_dataset_BinFill.h5"
FRAME_COUNT = 33
INPUT_FRAMES_SHA = "ffbea3422d40c2fd8120ee9a379d70b818734e73ccda7d327cecbade2a67ac7a"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path: Path, payload: dict) -> None:
    """完整落盘后原子发布，目标已存在即拒绝覆盖历史证据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def checked_root(root: str | Path) -> Path:
    path = Path(root).absolute()
    if not path.is_relative_to(Path("/scratch/hongze")):
        raise ValueError(f"冒烟产物必须留在 /scratch/hongze：{path}")
    for p in (path, *path.parents):
        if p.is_symlink():
            raise ValueError(f"冒烟输出路径不能穿过符号链接：{p}")
    return path


def import_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _copy_exact(source: Path, target: Path) -> str:
    expected = sha256_file(source)
    if target.exists():
        if not target.is_file() or target.is_symlink() or sha256_file(target) != expected:
            raise ValueError(f"已有注册文件与来源不同，拒绝覆盖：{target}")
        return expected
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    if sha256_file(target) != expected:
        raise ValueError(f"复制后 SHA 不同：{target}")
    return expected


def register_encoder() -> dict:
    """批准的 final 原样注册为 epoch6；不使用原 run 内另一份 epoch6。"""
    source = ORIGINAL_RUN / "checkpoint_final.pt"
    if sha256_file(source) != CHECKPOINT_SHA:
        raise ValueError("原始 final checkpoint 不是批准的 SHA256")
    import torch

    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("epoch") != 6 or payload.get("arch") != "wan-latent-v10-st" or "encoder" not in payload:
        raise ValueError("批准 checkpoint 必须包含 epoch6、v10 架构及 encoder EMA")
    del payload
    destination = checked_root(REGISTERED_RUN)
    allowed = {CHECKPOINT, "config.yaml", "pair4_registration.json"}
    if destination.exists() and {p.name for p in destination.iterdir()} - allowed:
        raise ValueError(f"注册目录存在未知产物，拒绝混用：{destination}")
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = _copy_exact(source, destination / CHECKPOINT)
    if checkpoint_sha != CHECKPOINT_SHA:
        raise ValueError("复制过程中原始checkpoint发生变化")
    config_sha = _copy_exact(ORIGINAL_RUN / "config.yaml", destination / "config.yaml")
    record = {
        "purpose": "paired4_training_smoke", "original_checkpoint": str(source),
        "registered_checkpoint": str(destination / CHECKPOINT),
        "original_config": str(ORIGINAL_RUN / "config.yaml"),
        "registered_config": str(destination / "config.yaml"),
        "checkpoint_sha256": checkpoint_sha, "config_sha256": config_sha,
        "checkpoint_epoch": 6, "arch": "wan-latent-v10-st", "state_key": "encoder",
        "source_run": f"{RUN_NAME}/{CHECKPOINT}#encoder",
    }
    record_path = destination / "pair4_registration.json"
    if record_path.exists():
        if read_json(record_path) != record:
            raise ValueError("已有注册记录与本次来源合同不同")
    else:
        write_new_json(record_path, record)
    return record


def _load_parent_samples() -> tuple[np.ndarray, list[bytes], list[dict]]:
    import h5py

    with h5py.File(PARENT_H5, "r") as f:
        frames = f["frames"][:FRAME_COUNT]
    if frames.shape != (33, 256, 256, 3) or frames.dtype != np.uint8:
        raise ValueError("父 H5 前33帧形制不是 uint8 (33,256,256,3)")
    if array_sha(frames) != INPUT_FRAMES_SHA:
        raise ValueError("父 H5 的33帧不是主会话已独立核验的RGB字节")
    blobs, records = [], []
    for t, frame in enumerate(frames):
        path = PARENT / "source/data" / f"{t}.pkl"
        blob = path.read_bytes()
        sample = pickle.loads(blob)
        image = sample["image"]
        if image.shape != frame.shape or image.dtype != frame.dtype or image.tobytes() != frame.tobytes():
            raise ValueError(f"H5 与父 PKL 图像逐位不同：t={t}")
        if int(np.asarray(sample["epis_idx"]).item()) != 0 or int(np.asarray(sample["step_idx"]).item()) != t:
            raise ValueError(f"父 PKL 索引不同：t={t}")
        if np.asarray(sample["actions"]).shape != (20, 8):
            raise ValueError(f"父 PKL 的20步动作不完整：t={t}")
        blobs.append(blob)
        records.append({"t": t, "parent_pkl": str(path), "parent_pkl_sha256": hashlib.sha256(blob).hexdigest(),
                        "image_sha256": array_sha(image), "actions_sha256": array_sha(sample["actions"]),
                        "actions_shape": list(sample["actions"].shape), "actions_dtype": str(sample["actions"].dtype)})
    return frames, blobs, records


def _restore_frame_features(target: Path) -> dict:
    from mme_vla_suite.datastore import framesamp_store as fs

    parent_store = PARENT / "framesamp-8x8"
    meta = fs.StoreMeta.load(parent_store)
    fs.require_no_pack_lock(parent_store)
    if meta.status != "verified" or meta.spec.layout != "framesamp-8x8-v1" or meta.manifest_sha256 != PARENT_MANIFEST_SHA:
        raise ValueError("父帧表状态、布局或清单身份不符")
    digest_meta = meta.raw["row_digests"]
    digest_path = parent_store / digest_meta["relpath"]
    if sha256_file(digest_path) != digest_meta["sha256"]:
        raise ValueError("父帧表逐行摘要文件 SHA 不同")
    with digest_path.open("rb") as f:
        expected_digests = f.read(FRAME_COUNT * fs.ROW_DIGEST_BYTES)
    target.mkdir(parents=True)
    hashes = []
    # 仅取前33行；正常 fast 检查仍执行，固定抽父 data/0.pkl 以免依赖已清理的散特征。
    with fs.FrameSampStore(parent_store, verify_level="fast", fast_sample=(0, 0)) as store:
        for t in range(FRAME_COUNT):
            image = store.read_image_rows([t])
            pos = store.pos_rows([t])
            state = store.state_rows([t])[0]
            raw = image.tobytes() + pos.tobytes() + state.tobytes()
            digest = hashlib.blake2b(raw, digest_size=fs.ROW_DIGEST_BYTES).digest()
            if digest != expected_digests[t * fs.ROW_DIGEST_BYTES:(t + 1) * fs.ROW_DIGEST_BYTES]:
                raise ValueError(f"父帧表第{t}行三键摘要不符")
            path = target / f"token_emb_{t}.npy"
            with path.open("xb") as f:
                np.save(f, {"image_emb_8x8": image, "pos_emb_8x8": pos, "state_emb": state})
            hashes.append({"t": t, "npy_sha256": sha256_file(path), "parent_row_blake2b": digest.hex()})
    return {"parent_store": str(parent_store), "parent_meta_sha256": sha256_file(parent_store / fs.META_RELPATH),
            "parent_row_digests_sha256": digest_meta["sha256"], "rows": hashes}


def prepare_fixture(root: str | Path) -> dict:
    """创建真实输入、监督视图和通过正常全行校验的帧表。"""
    sys.path.insert(0, str(REPO / "src"))
    import h5py

    from mme_vla_suite.datastore import framesamp_store as fs
    from mme_vla_suite.datastore.manifest import load_manifest
    from mme_vla_suite.datastore.manifest import manifest_sha256

    root = checked_root(root)
    fixture = root / "fixture"
    if fixture.exists():
        raise FileExistsError(f"夹具目录已存在；保留现场，不覆盖或自动续跑：{fixture}")
    parent = load_manifest(PARENT / "meta/episode_manifest.json")
    ep = parent["episodes"][0]
    if parent["sha256"] != PARENT_MANIFEST_SHA or any(ep[k] != v for k, v in {
        "global_episode_idx": 0, "h5_file": H5_NAME, "raw_ep_idx": 0, "exec_start_idx": 0,
        "exec_sample_offset": 0, "total_sample_offset": 0,
    }.items()) or ep["num_timesteps"] < 52:
        raise ValueError("父 manifest 或 g0 身份不符，无法构造保留20步动作的前缀")
    frames, pkl_blobs, frame_records = _load_parent_samples()
    registration = register_encoder()
    fixture.mkdir(parents=True)
    raw_dir, source = fixture / "raw", fixture / "source"
    raw_dir.mkdir()
    (source / "data").mkdir(parents=True)
    staged_h5 = raw_dir / H5_NAME
    with h5py.File(staged_h5, "x") as f:
        f.attrs["purpose"] = "smoke_observation_prefix_not_task_terminal"
        for t, frame in enumerate(frames):
            f.create_dataset(f"episode_0/timestep_{t}/obs/front_rgb", data=frame, dtype=np.uint8)
    for t, blob in enumerate(pkl_blobs):
        (source / "data" / f"{t}.pkl").write_bytes(blob)
    manifest = {
        "version": 1, "raw_dir": str(raw_dir), "canonical_order": [H5_NAME], "num_shards": 1,
        "totals": {"episodes": 1, "timesteps": 33, "exec_samples": 33}, "shard_load_timesteps": [33],
        "episodes": [{**ep, "num_timesteps": 33, "exec_samples": 33, "shard_idx": 0}],
        "derivation": {"purpose": "smoke", "parent_manifest_sha256": PARENT_MANIFEST_SHA,
                       "parent_g": 0, "frame_range": [0, 33], "new_task_terminal": False},
    }
    manifest["sha256"] = manifest_sha256(manifest)
    manifest_path = fixture / "meta/episode_manifest.json"
    write_new_json(manifest_path, manifest)
    inputs = {"schema": 1, "purpose": "smoke", "raw_dir": str(raw_dir), "manifest_sha256": manifest["sha256"],
              "parent_manifest_sha256": PARENT_MANIFEST_SHA, "parent_h5": str(PARENT_H5),
              "parent_h5_sha256": sha256_file(PARENT_H5), "staged_h5": str(staged_h5),
              "staged_h5_sha256": sha256_file(staged_h5), "frame_range": [0, 33],
              "input_frames_sha256": array_sha(frames), "frames": frame_records}
    write_new_json(fixture / "meta/input_manifest.json", inputs)
    features = _restore_frame_features(source / "features/episode_0")
    pack = import_script("pair4_frame_pack", REPO / "scripts/dataset/pack_framesamp_store.py")
    frame_store = fixture / "framesamp-8x8"
    pack.cmd_pack(argparse.Namespace(manifest=str(manifest_path), source=str(source), out=str(frame_store),
                                    layout="framesamp-8x8-v1", subset_prefix=None, reader="decode", procs=1,
                                    resume=False, force_break_lock=False))
    pack.cmd_verify(argparse.Namespace(store=str(frame_store), manifest=str(manifest_path), source=str(source),
                                      sample=None, seed=42, procs=1, resume=False, force_break_lock=False))
    meta = fs.StoreMeta.load(frame_store)
    if meta.status != "verified" or meta.manifest_scope != "full" or meta.raw["verify"]["scanned"] != 33:
        raise ValueError("正常帧表全行验证未完成")
    record = {
        "schema": 1, "purpose": "smoke", "root": str(root), "parent_dataset": str(PARENT),
        "parent_manifest_sha256": PARENT_MANIFEST_SHA, "parent_g": 0, "frame_range": [0, 33],
        "supervision": "原PKL字节；观测前缀不改变原任务终止点或未来20步动作", "sample_idx": 32,
        "manifest_sha256": manifest["sha256"], "manifest_file_sha256": sha256_file(manifest_path),
        "input_manifest_sha256": sha256_file(fixture / "meta/input_manifest.json"),
        "frames_meta_sha256": sha256_file(frame_store / fs.META_RELPATH),
        "registration": registration, "features": features, "actions_idx32": frame_records[32],
        "paths": {"fixture": str(fixture), "manifest": str(manifest_path), "raw_dir": str(raw_dir),
                  "source": str(source), "frames": str(frame_store), "latents": str(fixture / "wan-latents"),
                  "raw8": str(root / "raw8"), "oracle": str(fixture / "oracle"),
                  "motion": str(fixture / "motion"), "encoder_run_dir": str(REGISTERED_RUN)},
    }
    write_new_json(fixture / "fixture_manifest.json", record)
    return {"status": "PASS", "gates": {
        "H5_PKL": {"status": "PASS", "frames": 33, "mismatches": 0},
        "FRAME_FIXTURE": {"status": "PASS", "scanned": 33, "manifest_scope": "full"},
        "REGISTERED_SOURCE": {"status": "PASS", **registration}},
        "paths": record["paths"], "fixture_manifest_path": str(fixture / "fixture_manifest.json"),
        "manifest_sha256": manifest["sha256"], "frame_count": 33, "window_count": 1}


def load_fixture(root: str | Path) -> dict:
    """每个后续阶段重新核封口输入；不把上一阶段的内存配置当真值。"""
    root = checked_root(root)
    fixture = root / "fixture"
    record = read_json(fixture / "fixture_manifest.json")
    if record["root"] != str(root) or record["purpose"] != "smoke" or record["parent_manifest_sha256"] != PARENT_MANIFEST_SHA:
        raise ValueError("夹具根或父来源合同不同")
    for rel, key in (("meta/episode_manifest.json", "manifest_file_sha256"),
                     ("meta/input_manifest.json", "input_manifest_sha256"),
                     ("framesamp-8x8/meta/store_meta.json", "frames_meta_sha256")):
        if sha256_file(fixture / rel) != record[key]:
            raise ValueError(f"已封口夹具文件改变：{rel}")
    inputs = read_json(fixture / "meta/input_manifest.json")
    if sha256_file(Path(inputs["staged_h5"])) != inputs["staged_h5_sha256"]:
        raise ValueError("staged H5 在准备后发生变化")
    for frame in inputs["frames"]:
        if sha256_file(fixture / "source/data" / f"{frame['t']}.pkl") != frame["parent_pkl_sha256"]:
            raise ValueError(f"夹具PKL不再与父原字节一致：t={frame['t']}")
    registration = record["registration"]
    if Path(registration["registered_checkpoint"]).parent != REGISTERED_RUN:
        raise ValueError("注册encoder目录不是固定external副本")
    if sha256_file(Path(registration["registered_checkpoint"])) != CHECKPOINT_SHA:
        raise ValueError("已注册checkpoint不再是批准的字节")
    if sha256_file(Path(registration["registered_config"])) != registration["config_sha256"]:
        raise ValueError("注册config在准备后改变")
    return record


def finalize_fixture(root: str | Path) -> dict:
    """CPU复核独立单窗证据后，调用原生 motion pack/verify。"""
    sys.path.insert(0, str(REPO / "src"))
    from mme_vla_suite.datastore import motion_store as ms

    root = checked_root(root)
    record = load_fixture(root)
    paths = {k: Path(v) for k, v in record["paths"].items()}
    source_report = read_json(root / "reports/source-smoke.json")
    if source_report["status"] != "PASS" or source_report["manifest_sha256"] != record["manifest_sha256"]:
        raise ValueError("单窗源提取没有通过或清单身份不同")
    for relative, expected in source_report["artifacts_sha256"].items():
        path = root / relative
        if not path.resolve().is_relative_to(root) or sha256_file(path) != expected:
            raise ValueError(f"单窗产物SHA不同：{relative}")
    oracle = read_json(paths["oracle"] / "vae_report.json")
    if oracle.get("scope") != "smoke_one_window" or oracle.get("checked_windows") != 1:
        raise ValueError("oracle不是已实际校验的一窗报告")
    if any(oracle.get(k) != 0 for k in ("frame_mismatches", "metadata_mismatches", "latent_mismatches", "encoder_mismatches")):
        raise ValueError("独立oracle的帧、元数据或数值对拍未通过")
    pack = import_script("pair4_motion_pack", REPO / "scripts/dataset/pack_motion_store.py")
    pack.check_raw_sources(paths["manifest"], paths["latents"], paths["raw_dir"], paths["oracle"] / "vae_report.json")
    if paths["motion"].exists():
        raise FileExistsError("motion夹具已存在；不覆盖正常pack/verify现场")
    pack.cmd_pack(argparse.Namespace(manifest=str(paths["manifest"]), tokens=str(paths["raw8"]),
                                    latents=str(paths["latents"]), out=str(paths["motion"]),
                                    raw_dir=str(paths["raw_dir"]), encoder_run_dir=str(REGISTERED_RUN),
                                    oracle_report=str(paths["oracle"] / "vae_report.json"), layout=LAYOUT,
                                    resume=False, force_break_lock=False))
    pack.cmd_verify(argparse.Namespace(store=str(paths["motion"]), manifest=str(paths["manifest"]),
                                      tokens=str(paths["raw8"]), resume=False, force_break_lock=False))
    meta = ms.MotionMeta.load(paths["motion"])
    with ms.MotionStore(paths["motion"], meta=meta, manifest_path=str(paths["manifest"]), verify_level="full") as store:
        rows = store.rows(np.array([0], dtype=np.int64))
    blob = (paths["raw8"] / f"{SEGMENT}.f32.bin").read_bytes()
    if rows.shape != (1, 8, 256) or rows.dtype != np.float32 or rows.tobytes() != blob or len(blob) != 8192:
        raise ValueError("motion表回读不是原始8×256逐位同值")
    if (meta.status != "verified" or meta.num_rows != 1 or meta.raw["layout"] != LAYOUT
            or meta.raw.get("manifest_scope") != "full"):
        raise ValueError("motion表状态、行数或布局不符合单窗合同")
    return {"status": "PASS", "gates": {
        "RAW_CONTRACT": {"status": "PASS", "shape": [1, 8, 256], "dtype": "float32", "bytes": 8192,
                         "sha256": hashlib.sha256(blob).hexdigest(), "scanned": meta.raw["verify"]["scanned"]},
        "SOURCE_FRESH": {"status": "PASS", "checked_windows": 1, "oracle": oracle}},
        "paths": record["paths"], "manifest_sha256": record["manifest_sha256"], "frame_count": 33,
        "window_count": 1, "fixture_manifest_path": str(root / "fixture/fixture_manifest.json")}
