#!/usr/bin/env python3
"""阶段二 G2/G3 数值闸门；torch 仅在锁定的子环境内延迟导入。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LATENT_SHAPE = (9, 16, 32, 32)
CHUNK_BYTES = 589824
LAYOUT = "motion-stage2-1x1024-grid32-demopad17-v1"
PRODUCER = "stage2-numeric-validator-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    """仅发布完整 JSON；异常时清掉本次临时文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def revoke_report(path, *, gate=None, schema="stage2-validation-gate-v1"):
    """重验开始即撤销本工具的旧成功记录；拒绝覆盖其它用途的报告。"""
    path = Path(path)
    if path.exists():
        old = read_json(path)
        if old.get("schema") != schema or (gate is not None and old.get("gate") != gate):
            raise ValueError(f"报告路径已用于不同合同：{path}")
        path.unlink()


def require_sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} 必须为完整 SHA256")
    return value


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_samples(path):
    data = read_json(path)
    if (data.get("schema") != "stage2-validation-inputs-v1" or data.get("gate") != "G0"
            or data.get("status") != "PASS" or data.get("seed") != 0):
        raise ValueError("必须提供 G0 成功的固定 seed=0 样本清单")
    windows = data.get("windows", [])
    tasks = Counter(w.get("task") for w in windows)
    tails = sum(w.get("segment_kind") == "demo" and w.get("pad_frames", 0) > 0 for w in windows)
    expected = dict(windows=64, tasks=16, demo_tail=tails, sha_mismatches=0)
    if (len(windows) != 64 or len(tasks) != 16 or set(tasks.values()) != {4} or tails < 16
            or data.get("counts") != expected
            or len({(w.get("key"), w.get("source_grid")) for w in windows}) != 64):
        raise ValueError("G0 必须是每任务4窗、16任务、至少16个demo尾窗的64个不同窗口")
    for field in ("manifest_sha256", "latent_metadata_sha256"):
        require_sha(data.get(field), field)
    for window in windows:
        validate_window(window)
    return data


def validate_window(window):
    kind, grid = window.get("segment_kind"), window.get("source_grid")
    real, pad = window.get("real_frames"), window.get("pad_frames")
    ids = window.get("frame_ids")
    if (kind not in ("demo", "exec") or type(grid) is not int or grid < 0 or grid % 2
            or type(real) is not int or type(pad) is not int or real + pad != 33
            or real < (17 if kind == "demo" else 33) or pad < 0
            or not isinstance(ids, list) or len(ids) != 33
            or any(type(x) is not int or x < 0 for x in ids)
            or ids[:real] != list(range(ids[0], ids[0] + real))
            or ids[real:] != [ids[real - 1]] * pad):
        raise ValueError("窗口偶数格、真实帧与尾部重复合同错误")
    if window.get("pad_source_frame") != (ids[real - 1] if pad else None):
        raise ValueError("pad_source_frame 与真实末帧不同")
    for field in ("input_frames_sha256", "latent_block_sha256"):
        require_sha(window.get(field), field)
    sources, indices = window.get("sources"), window.get("source_frame_indices")
    if not isinstance(sources, list) or not sources or not isinstance(indices, list) or len(indices) != 33:
        raise ValueError("缺少完整的原帧来源映射")
    for item in indices:
        if (type(item.get("source")) is not int or not 0 <= item["source"] < len(sources)
                or type(item.get("index")) is not int or item["index"] < 0):
            raise ValueError("原帧来源索引越界")


def read_window_frames(window, hash_cache=None):
    """按 G0 显式映射读取原像素；exec 的图像键固定为 image。"""
    validate_window(window)
    cache = {} if hash_cache is None else hash_cache
    images = {}
    for number, source in enumerate(window["sources"]):
        path = Path(source["path"])
        expected = require_sha(source.get("sha256"), str(path))
        stat = path.stat()
        stamp = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if stamp not in cache:
            cache[stamp] = sha_file(path)
        if cache[stamp] != expected:
            raise ValueError(f"原帧文件 SHA 改变：{path}")
        wanted = {x["index"] for x in window["source_frame_indices"] if x["source"] == number}
        if source["kind"] == "demo_h5" and window["segment_kind"] == "demo":
            import h5py
            with h5py.File(path, "r") as handle:
                frames = handle["frames"]
                if frames.dtype != np.uint8 or frames.shape[1:] != (256, 256, 3):
                    raise ValueError("demo H5 必须包含 uint8 RGB 原帧")
                for index in wanted:
                    images[number, index] = frames[index]
        elif source["kind"] == "exec_pkl" and window["segment_kind"] == "exec":
            if wanted != {0}:
                raise ValueError("exec pkl 的帧索引只能为0")
            # G0 已认证本机数据来源；反序列化前再次核对文件摘要。
            payload = pickle.loads(path.read_bytes())
            images[number, 0] = payload["image"]
        else:
            raise ValueError("原帧文件类型与段类型不符")
    frames = np.ascontiguousarray(np.stack([images[x["source"], x["index"]]
                                           for x in window["source_frame_indices"]]))
    if frames.dtype != np.uint8 or frames.shape != (33, 256, 256, 3):
        raise ValueError("原帧必须为 uint8[33,256,256,3]")
    if sha_bytes(frames.tobytes()) != window["input_frames_sha256"]:
        raise ValueError("重建的33帧像素 SHA 与 G0 不同")
    return frames


def read_latent(window):
    with open(window["latent_path"], "rb") as stream:
        stream.seek(window["source_grid"] * CHUNK_BYTES)
        block = stream.read(CHUNK_BYTES)
    if len(block) != CHUNK_BYTES or sha_bytes(block) != window["latent_block_sha256"]:
        raise ValueError("恢复 latent 块长度或 SHA 不符")
    values = np.frombuffer(block, dtype="<f4").reshape(LATENT_SHAPE).copy()
    if not np.isfinite(values).all():
        raise ValueError("恢复 latent 包含非有限值")
    return values


def normalized_error(actual, expected, std):
    """逐通道使用 VAE latents_std 归一化，不用相对输出误差替代原阈值。"""
    a, b = np.asarray(actual), np.asarray(expected)
    scale = np.asarray(std, dtype=np.float64).reshape(-1)
    if (a.shape != LATENT_SHAPE or b.shape != LATENT_SHAPE or scale.shape != (16,)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.isfinite(scale).all() or not (scale > 0).all()):
        raise ValueError("归一化比较要求完整有限 latent 与16个正标准差")
    delta = (a.astype(np.float64) - b.astype(np.float64)) / scale[None, :, None, None]
    return float(np.mean(delta ** 2)), float(np.max(np.abs(delta)))


def assert_ema_state(encoder, payload):
    """逐项比较原张量；负控实际把 model_live 送入同一比较器并要求拒绝。"""
    import torch
    current = encoder.state_dict()

    def compare(raw):
        state = {k.removeprefix("encoder."): v for k, v in raw.items() if k.startswith("encoder.")}
        if len(current) != 65 or set(state) != set(current):
            raise ValueError("EMA encoder 必须精确含65个状态项")
        for name, value in current.items():
            target = state[name]
            if (value.dtype != torch.float32 or target.dtype != value.dtype or target.shape != value.shape
                    or not torch.isfinite(value).all() or not torch.isfinite(target).all()
                    or not torch.equal(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                       target.detach().cpu().contiguous().reshape(-1).view(torch.uint8))):
                raise ValueError(f"encoder 原张量不同：{name}")

    compare(payload["model"])
    live = payload.get("model_live")
    if not isinstance(live, dict):
        raise ValueError("负控需要 checkpoint 的真实 model_live")
    if {k for k in live if k.startswith("encoder.")} != {"encoder." + k for k in current}:
        raise ValueError("负控 model_live 的 encoder 键不完整")
    try:
        compare(live)
    except ValueError:
        return dict(leaves=65, state_key="model", exact=1, live_negative_rejected=1)
    raise ValueError("model_live 负控没有被原张量检查拒绝")


def load_pinned_mj_source(root, relative, name, pin):
    """wheel 不收录上游 scripts；只读 checkout 且按锁定 Git blob 核查文件。"""
    root = Path(root).resolve()
    path = root / relative
    expected = subprocess.check_output(["git", "-C", str(root), "show", f"{pin['mj_repo_commit']}:{relative}"])
    if sha_file(path) != sha_bytes(expected):
        raise ValueError(f"MJ 验证源码与 S1 提交不同：{path}")
    return load_module(path, name)


def setup_runtime(args):
    import torch
    from motion_jepa.export import stage2_memory as memory
    adapter = load_module(HERE.parent / "wan/wan_motion_infer.py", "_stage2_numeric_vae")
    adapter.check_env()
    adapter.pin_numerics()
    adapter.check_versions(strict=True, with_diffusers=True)
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("正式数值闸门只能使用已分配的 A100 CUDA 设备")
    if "A100" not in torch.cuda.get_device_name(args.device):
        raise RuntimeError("数值闸门要求同型号 A100")
    torch.manual_seed(0)
    encoder, info = memory.load_ema_encoder(args.ckpt, device=args.device,
                                           expected_sha256=args.expected_sha256)
    pin = read_json(HERE / "SOURCE_PIN.json")
    if (any(pin.get(k) != v for k, v in info["source_pin"].items())
            or pin.get("vae_adapter_sha256") != sha_file(HERE.parent / "wan/wan_motion_infer.py")
            or pin.get("dependency_lock_sha256") != sha_file(HERE / "uv.lock")):
        raise ValueError("实际模块、VAE适配件或锁文件与 SOURCE_PIN 不同")
    return torch, memory, adapter, encoder, info


def gate_g2(args, samples, runtime):
    torch, _, adapter, _, info = runtime
    full = load_pinned_mj_source(args.mj_root, "scripts/dataset-build/wan_fp16_full33.py",
                                 "_stage2_full33", info["source_pin"])
    pin = read_json(HERE / "SOURCE_PIN.json")
    online_model, online_info = adapter.load_vae("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", args.device,
                                                expected_state_sha256=adapter.VAE_STATE_SHA256_EXPECTED)
    online_vae = {key: online_info[key] for key in ("vae_id", "vae_state_sha256", "vae_dtype", "latent_mode", "batch", "tf32", "amp")}
    online_vae.update(adapter_sha256=pin["vae_adapter_sha256"], numerics=info["numerics"],
                      preprocess=dict(frame_size=256, input_dtype="uint8", input_layout="THWC", window_frames=33))
    del online_model
    # 确认整窗模块指定的本地snapshot也来自同份原FP32权重，不能只信目录名。
    full_weight_model = full.load_wan(args.vae_snapshot or full.SNAPSHOT, "cpu", torch.float32)
    full_weight_sha = adapter.vae_state_sha256(full_weight_model)
    del full_weight_model
    if full_weight_sha != online_vae["vae_state_sha256"]:
        raise ValueError("整窗FP16路径的源VAE权重与在线适配件不同")
    # 整窗缓存使用 FP16、CL3D、cudnn TF32/benchmark 开；与在线 FP32 路径分开记录。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    vae = full.load_wan(args.vae_snapshot or full.SNAPSHOT, args.device, torch.float16)
    encode = full.make_encoder(vae, mode=None if args.compile_mode == "eager" else args.compile_mode,
                               batch=1, device=args.device)
    std = full.latents_std(vae).cpu().numpy()
    rows, cache = [], {}
    with torch.no_grad():
        for window in samples["windows"]:
            pixels = read_window_frames(window, cache)
            x = torch.from_numpy(pixels).permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1.0
            actual = full.encode_b2(encode, full.to_encoder_input(x, args.device))[0].cpu().numpy()
            mse, maximum = normalized_error(actual, read_latent(window), std)
            if mse > 1e-6 or maximum > 0.01:
                raise ValueError(f"G2 超阈值：{window['key']}:{window['source_grid']} mse={mse} max={maximum}")
            rows.append(dict(key=window["key"], source_grid=window["source_grid"], mse=mse, max_abs=maximum))
    metrics = dict(windows=len(rows), demo_tail=samples["counts"]["demo_tail"],
                   worst_mse=max(x["mse"] for x in rows), worst_max=max(x["max_abs"] for x in rows))
    return dict(metrics=metrics, windows=rows, full33_source_sha256=sha_file(full.__file__),
                mj_root=str(Path(args.mj_root).resolve()),
                full33_numerics=dict(dtype="float16", memory_format="channels_last_3d", batch=1,
                                     compile_mode=args.compile_mode, cudnn_allow_tf32=True, cudnn_benchmark=True),
                online_adapter_sha256=sha_file(adapter.__file__), encoder=info["encoder"], source_pin=pin,
                online_vae=online_vae, numerics=info["numerics"], full33_vae_state_sha256=full_weight_sha)


def gate_g3(args, samples, runtime):
    torch, memory, _, encoder, info = runtime
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ema = assert_ema_state(encoder, payload)
    reader_module = load_pinned_mj_source(args.mj_root, "scripts/dataset-build/rgb_gtxyz_reader.py",
                                         "_motion_jepa_paired_cache_reader", info["source_pin"])
    data_module = load_pinned_mj_source(args.mj_root, "src/motion_jepa/data/dataset_paired_wan.py",
                                       "_stage2_paired_data", info["source_pin"])
    reader = reader_module.PairedWanCacheReader(args.rgb_root)
    candidates = sorted(key for key in reader.windows if reader.windows[key]["split"] == "val")
    if len(candidates) < 32:
        raise ValueError("val 窗口不足32")
    selected = np.random.default_rng(0).choice(candidates, 32, replace=False).tolist()
    identities = []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        for key in selected:
            pair = reader.read_pair(key, 0)
            if pair["window"]["split"] != "val" or pair["params"]["aug_id"] != 0:
                raise ValueError("G3 reader 返回了非 val aug0 窗口")
            masks = data_module.derive_geometry_masks(pair["common_valid"], pair["xyz_valid"])
            geometry = np.asarray(masks["geometry_valid"])
            if geometry.shape != (32, 32) or geometry.dtype != np.bool_ or not geometry.all():
                raise ValueError("val aug0 的派生 geometry_valid 必须是全真 bool[32,32]")
            raw = np.asarray(pair["rgb_raw"])
            if raw.shape != LATENT_SHAPE or raw.dtype != np.float16 or not np.isfinite(raw).all():
                raise ValueError("训练缓存 rgb_raw 形制或有限性错误")
            latent = torch.from_numpy(raw.astype(np.float32)).unsqueeze(0).to(args.device)
            mask = torch.from_numpy(geometry.copy()).unsqueeze(0).to(args.device)
            # 训练直接调用路径与公开导出路径独立；输出按来源槽行优先展平。
            training = encoder(latent, mask).reshape(1, 1024)
            exported = memory.encode_windows(encoder, latent)
            if (not torch.isfinite(training).all() or not torch.isfinite(exported).all()
                    or not torch.equal(training.view(torch.uint8), exported.view(torch.uint8))):
                raise ValueError(f"G3 两条调用路径未逐位相等：{key}")
            identities.append(dict(window=pair["window"], aug_id=0, latent_sha256=sha_bytes(raw.tobytes()),
                                   geometry_sha256=sha_bytes(geometry.tobytes())))
    return dict(metrics=dict(windows=32, seed=0, max_abs=0, geometry_all_true=1, **ema),
                sample_identity=identities, sample_identity_sha256=sha_bytes(canonical(identities)),
                mj_root=str(Path(args.mj_root).resolve()),
                encoder=info["encoder"], encoder_state_sha256=info["encoder_state_sha256"],
                source_pin=read_json(HERE / "SOURCE_PIN.json"), numerics=info["numerics"],
                reader_source_sha256=sha_file(reader_module.__file__),
                geometry_source_sha256=sha_file(data_module.__file__))


def assert_encoder_identity(info, provenance):
    for group in ("encoder", "encoder_state_sha256", "numerics"):
        actual = {k: v for k, v in info[group].items() if k != "run_dir"}
        wanted = {k: v for k, v in provenance[group].items() if k != "run_dir"}
        if actual != wanted:
            raise ValueError(f"实际 {group} 与 store 来源不同")
    if any(provenance["source_pin"].get(k) != v for k, v in info["source_pin"].items()):
        raise ValueError("实际 S1 源码与 store 不同")


def reference_g4(args, samples, runtime):
    """锁子环境中的生产 sidecar latent 入口与离线编码/表行逐位比较。"""
    torch, memory, _, encoder, info = runtime
    from mme_vla_suite.datastore import motion_store as ms
    meta = ms.MotionMeta.load(args.store)
    if meta.raw["layout"] != LAYOUT:
        raise ValueError("G4 仅支持 stage2 表")
    assert_encoder_identity(info, meta.provenance)
    ms.run_full_checks(meta)
    sidecar = load_module(HERE / "stage2_sidecar.py", "_stage2_numeric_sidecar")
    table = np.memmap(meta.root / ms.MOTION_TABLE_RELPATH, mode="r", dtype="<f4", shape=(meta.num_rows, 1024))
    lookup = {}
    for entry in meta.entries:
        for segment in ("demo", "exec"):
            lookup[ms.segment_key(entry.h5_file, entry.raw_ep_idx, segment)] = getattr(entry, segment)
    outputs = []
    with torch.no_grad():
        for window in samples["windows"]:
            latent = torch.from_numpy(read_latent(window)).to(args.device)
            offline = memory.encode_windows(encoder, latent.unsqueeze(0))[0].cpu().numpy()
            online = np.asarray(sidecar.encode_latent(encoder, latent))
            part = lookup[window["key"]]
            index = window["source_grid"] // 2
            if index >= part.num_grid:
                raise ValueError("G0 窗口不在 packed 索引内")
            packed = table[part.row_base + index]
            if any(x.dtype != np.float32 or x.shape != (1024,) or not np.isfinite(x).all()
                   for x in (offline, online, packed)):
                raise ValueError("同latent比较输出形制错误")
            if offline.tobytes() != online.tobytes() or offline.tobytes() != packed.tobytes():
                raise ValueError("生产 sidecar、离线编码与 packed 表未逐位相同")
            outputs.append(offline.copy())
    output = Path(args.reference_output)
    np.save(output, np.stack(outputs), allow_pickle=False)
    return dict(metrics=dict(windows=64, encoder_max_abs=0), reference_sha256=sha_file(output),
                encoder=info["encoder"], numerics=info["numerics"], source_pin=info["source_pin"],
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                gpu_uuid=str(torch.cuda.get_device_properties(args.device).uuid),
                python_executable=sys.executable)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, choices=("G2", "G3", "G4-reference"))
    parser.add_argument("--samples", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rgb-root")
    parser.add_argument("--mj-root", default=os.environ.get("MOTIONJEPA_SOURCE_ROOT", "/scratch/hongze/MotionJEPA"))
    parser.add_argument("--vae-snapshot")
    parser.add_argument("--compile-mode", choices=("default", "max-autotune", "eager"), default="default")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--store")
    parser.add_argument("--reference-output")
    args = parser.parse_args(argv)
    revoke_report(args.out, gate=args.gate)
    try:
        require_sha(args.expected_sha256, "expected-sha256")
        if sha_file(args.ckpt) != args.expected_sha256:
            raise ValueError("checkpoint SHA 与显式参数不符")
        samples_sha = sha_file(args.samples)
        samples = load_samples(args.samples)
        if args.gate == "G3" and not args.rgb_root:
            raise ValueError("G3 必须指定 --rgb-root")
        if args.gate == "G4-reference" and (not args.store or not args.reference_output):
            raise ValueError("同latent子检查必须指定 --store/--reference-output")
        runtime = setup_runtime(args)
        result = {"G2": gate_g2, "G3": gate_g3, "G4-reference": reference_g4}[args.gate](args, samples, runtime)
        if sha_file(args.samples) != samples_sha or sha_file(args.ckpt) != args.expected_sha256:
            raise ValueError("验证期间样本或 checkpoint 改变")
        record = dict(schema="stage2-validation-gate-v1", gate=args.gate, status="PASS", producer=PRODUCER,
                      checkpoint_sha256=args.expected_sha256, samples_sha256=samples_sha, **result)
        atomic_json(args.out, record)
        m = record["metrics"]
        if args.gate == "G2":
            print("WAN_EQUIV=PASS " + " ".join(f"{k}={v}" for k, v in m.items()))
        elif args.gate == "G3":
            print("EMA_SOURCE=PASS leaves=65 state_key=model exact=1 live_negative_rejected=1")
            print(f"EXPORT_PARITY=PASS windows=32 seed=0 max_abs=0 geometry_all_true=1 ckpt_sha256={args.expected_sha256}")
        else:
            print("STAGE2_LATENT_PARITY=PASS windows=64 encoder_max_abs=0")
        return 0
    except Exception as exc:
        atomic_json(args.out, dict(schema="stage2-validation-gate-v1", gate=args.gate, status="FAIL",
                                   producer=PRODUCER, error=str(exc)))
        raise


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    raise SystemExit(main())
