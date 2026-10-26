#!/usr/bin/env python3
"""G4：真实客户端身份与64窗数值对拍，另用完整 stub IPC 检查窗口状态机。"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO / "scripts/dataset/wan_stage2"
sys.path.insert(0, str(PROJECT))
import validate_stage2 as V


def verify_gpu(provenance, physical_gpu):
    """按 UUID 核查物理卡，不能只相信 CUDA_VISIBLE_DEVICES 字符串。"""
    physical_gpu = str(physical_gpu)
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                                    text=True).splitlines()
    inventory = [tuple(x.strip() for x in row.split(",")) for row in rows]
    chosen = [row for row in inventory if physical_gpu in row[:2]]
    if len(chosen) != 1 or "A100" not in chosen[0][2]:
        raise ValueError("online-gpu 必须明确指向一张物理 A100")
    if (provenance.get("cuda_visible_devices") != physical_gpu
            or str(provenance.get("gpu_uuid", "")).removeprefix("GPU-") != chosen[0][1].removeprefix("GPU-")):
        raise ValueError("sidecar 实际物理 GPU 与显式分配不符")
    return dict(index=chosen[0][0], uuid=chosen[0][1], name=chosen[0][2])


def require_handshake(client, *, stub):
    expected = dict(encoder_kind="stage2_pool4_concat", dtype="float32", shape=[1024],
                    response_bytes=4096, stub=stub)
    if not client.alive or client.stub is not stub:
        raise ValueError("客户端未存活或 stub 模式不符")
    for key, value in expected.items():
        if type(client.provenance.get(key)) is not type(value) or client.provenance[key] != value:
            raise ValueError(f"真实握手缺少或错配 {key}")


def reference_subprocess(args, meta, v1, directory):
    """主 JAX 进程不导入 torch；参考重算只在独立锁环境执行。"""
    enc = meta.provenance["encoder"]
    checkpoint = v1 / "external/motionjepa" / enc["run_name"] / enc["checkpoint_name"]
    output, report = Path(directory) / "references.npy", Path(directory) / "reference.json"
    env = dict(os.environ)
    env.update(UV_PROJECT_ENVIRONMENT=str(v1 / "venvs/wan-stage2"), UV_LINK_MODE="copy",
               CUDA_VISIBLE_DEVICES=str(args.online_gpu), HF_HOME=str(v1 / "cache/hf"), HF_HUB_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    argv = ["uv", "run", "--project", str(PROJECT), "--no-sync", "python", str(PROJECT / "validate_stage2.py"),
            "--gate", "G4-reference", "--samples", str(Path(args.samples).resolve()),
            "--store", str(meta.root.resolve()), "--ckpt", str(checkpoint),
            "--expected-sha256", enc["checkpoint_sha256"], "--reference-output", str(output), "--out", str(report)]
    subprocess.run(argv, env=env, cwd=REPO, check=True)
    record = V.read_json(report)
    if (record.get("status") != "PASS" or record.get("gate") != "G4-reference"
            or record.get("checkpoint_sha256") != enc["checkpoint_sha256"]
            or record.get("samples_sha256") != V.sha_file(args.samples)
            or record.get("reference_sha256") != V.sha_file(output)
            or record.get("metrics") != dict(windows=64, encoder_max_abs=0)):
        raise ValueError("锁子环境同latent参考未完整通过")
    verify_gpu(record, args.online_gpu)
    reference = np.load(output, allow_pickle=False)
    if reference.shape != (64, 1024) or reference.dtype != np.float32 or not np.isfinite(reference).all():
        raise ValueError("子进程参考形制错误")
    return reference, record


def check_state_machine(client, motion_cfg, *, token_per_image=64):
    """原 FrameSampMemory + pack_buffer 跑 demo、三批exec、清空重开；逐窗反推帧号。"""
    import jax.numpy as jnp
    from mme_vla_suite.policies.framesamp_memory import FrameSampMemory
    from mme_vla_suite.policies import motion_protocol as protocol
    utils = V.load_module(REPO / "examples/robomme/utils.py", "_stage2_online_pack")
    seen = []

    def encode(frames, start_frame):
        seen.append((start_frame, protocol.stub_decode(frames)))
        result = client(frames, start_frame)
        if result.shape != (1024,) or result.dtype != np.float32 or not np.all(result == start_frame):
            raise ValueError("stub 回包没有保留完整起点/1024维")
        return result

    memory = FrameSampMemory(num_views=1, img_emb_dim=4, pos_emb_dim=768, state_emb_dim=8,
                             max_steps=256, token_per_image=token_per_image, motion_enc_fn=encode,
                             motion_cfg=motion_cfg,
                             vision_enc_fn=lambda images: jnp.zeros((*images.shape[:2], 256, 4), jnp.float32))

    def add(lo, hi, es):
        packed = utils.pack_buffer([protocol.stub_frame(i) for i in range(lo, hi)],
                                   [np.zeros(8, np.float32) for _ in range(lo, hi)], es)
        memory.add_buffer(packed["images"], packed["state"], list(range(lo, hi)), exec_start_idx=es)

    def check(last, es):
        expected = list(range(0, max(0, es - 16), 32)) + list(range(es, max(es, last - 31), 32))
        if list(memory._history_feats_motion) != expected or memory.visible_motion_frames(last) != expected:
            raise ValueError("在线状态机起点与独立边界公式不同")
        emb, _, mask, times = memory._prepare_motion(last)
        if (mask.sum() != len(expected) or not np.all(mask[:len(expected)]) or mask[len(expected):].any()
                or not np.array_equal(times[:len(expected)], expected)
                or not np.array_equal(emb[:len(expected), 0], expected) or emb[len(expected):].any()):
            raise ValueError("在线可见窗、补位 mask 或时间顺序错误")
        for start, ids in seen:
            real = min(33, es - start) if start < es else 33
            want = list(range(start, start + real)) + [es - 1] * (33 - real)
            if ids != want:
                raise ValueError("stub 反推33帧发现跨段、漏帧或错补位")

    # 49帧demo包含17帧补尾；连续exec的相邻窗共用第32帧。
    add(0, 49, 49)
    check(48, 49)
    for lo, hi in ((49, 82), (82, 114), (114, 146)):
        add(lo, hi, 49)
        check(hi - 1, 49)
    if set(memory._raw_frames) != {145}:
        raise ValueError("连续exec没有精确保留下一窗的共享边界帧")
    before_reset = len(seen)
    memory.clear()
    if memory.exec_start_idx is not None or memory.n_steps or memory._raw_frames or memory._history_feats_motion:
        raise ValueError("reset 后仍残留上一episode数据")
    if memory._next_grid_start_demo or memory._next_grid_start_exec:
        raise ValueError("reset 未重置窗口游标")
    seen.clear()
    add(0, 33, 0)
    check(32, 0)
    if not client.alive:
        raise ValueError("reset 不应销毁客户端")
    return dict(frame_ids_mismatch=0, state_machine_mismatch=0, stub_windows=before_reset + len(seen),
                demo_tail=1, exec_batches=3, reset=1)


def prepare_restore_inputs(meta, samples, v1, output_root):
    """保存真实demo与连续65帧exec；demo state从已核SHA的帧表读取，禁止合成。"""
    import h5py
    from mme_vla_suite.datastore import framesamp_store as fs
    from mme_vla_suite.datastore import motion_store as ms
    from mme_vla_suite.datastore.manifest import load_manifest
    manifest_path = Path(meta.manifest_path)
    manifest = load_manifest(manifest_path)
    frame_root = manifest_path.parent.parent / "framesamp-8x8"
    frame_meta = fs.StoreMeta.load(frame_root)
    if (frame_meta.status != "verified" or frame_meta.manifest_sha256 != meta.manifest_sha256
            or frame_meta.manifest_scope != "full"):
        raise ValueError("便携样本的状态源必须是同manifest已verified的完整帧表")
    state_path = frame_root / fs.STATE_TABLE_RELPATH
    state_sha = V.sha_file(state_path)
    state_info = frame_meta.raw["tables"][fs.STATE_KEY]
    if (state_sha != state_info["sha256"] or state_path.stat().st_size != frame_meta.num_rows * 32
            or state_info["row_shape"] != [8] or state_info["dtype"] != "float32"):
        raise ValueError("原始状态表 SHA 或形制错误")
    episodes = {ms.segment_key(ep["h5_file"], ep["raw_ep_idx"], "demo"): ep for ep in manifest["episodes"]}
    candidates = []
    for window in samples["windows"]:
        if window["segment_kind"] != "demo":
            continue
        ep = episodes[window["key"]]
        es = ep["exec_start_idx"]
        if 17 <= es < 4031 and ep["num_timesteps"] - es >= 65:
            candidates.append((es, window["key"], ep, window))
    if not candidates:
        raise ValueError("G0 中找不到可提供完整demo和连续65帧exec的episode")
    es, key, episode, window = min(candidates, key=lambda item: item[:2])
    demo_source = window["sources"][0]
    demo_path = Path(demo_source["path"])
    if len(window["sources"]) != 1 or demo_source["kind"] != "demo_h5" or V.sha_file(demo_path) != demo_source["sha256"]:
        raise ValueError("便携demo原图必须来自G0已绑定的完整H5")
    with h5py.File(demo_path, "r") as handle:
        demo = handle["frames"][:]
    if demo.shape != (es, 256, 256, 3) or demo.dtype != np.uint8:
        raise ValueError("demo整段形制与manifest不符")
    states_table = np.memmap(state_path, mode="r", dtype="<f4", shape=(frame_meta.num_rows, 8))
    start = episode["total_sample_offset"]
    states = np.array(states_table[start:start + es + 65], dtype=np.float32, copy=True)
    if states.shape != (es + 65, 8) or not np.isfinite(states).all():
        raise ValueError("便携样本状态不足或非有限")
    sources = [dict(path=str(demo_path), sha256=demo_source["sha256"], role="demo_rgb"),
               dict(path=str(state_path), sha256=state_sha, role="demo_and_exec_state", row_start=start, rows=es + 65),
               dict(path=str(frame_root / fs.META_RELPATH), sha256=V.sha_file(frame_root / fs.META_RELPATH), role="state_metadata"),
               dict(path=str(manifest_path), sha256=V.sha_file(manifest_path), role="episode_mapping")]
    exec_frames, prompt, last = [], None, None
    for index in range(65):
        path = Path(frame_meta.source_dataset_root) / "data" / f"{episode['exec_sample_offset'] + index}.pkl"
        raw = path.read_bytes()
        sample = pickle.loads(raw)
        for field, value in (("epis_idx", episode["global_episode_idx"]), ("step_idx", es + index), ("exec_start_idx", es)):
            field_value = sample.get(field)
            if (not isinstance(field_value, np.ndarray) or field_value.dtype != np.int32
                    or field_value.shape != (1,) or field_value.item() != value):
                raise ValueError(f"连续exec的 {field} 不同源：{path}")
        if (sample["is_demo"].shape != (1,) or sample["is_demo"].dtype != np.bool_ or sample["is_demo"].item()
                or sample["image"].shape != (256, 256, 3) or sample["image"].dtype != np.uint8
                or sample["state"].shape != (8,) or sample["state"].dtype != np.float32
                or sample["state"].tobytes() != states[es + index].tobytes()):
            raise ValueError("连续exec原帧/state与已核SHA帧表不同")
        if not isinstance(sample.get("prompt"), str) or not sample["prompt"].strip():
            raise ValueError("exec缺真实任务prompt")
        if prompt is not None and prompt != sample["prompt"]:
            raise ValueError("连续exec任务prompt改变")
        prompt, last = sample["prompt"], sample
        exec_frames.append(sample["image"])
        sources.append(dict(path=str(path), sha256=V.sha_bytes(raw), role="exec_rgb_state_wrist_prompt"))
    wrist = last["wrist_image"]
    if wrist.shape != (256, 256, 3) or wrist.dtype != np.uint8:
        raise ValueError("最后exec缺真实wrist图像")
    arrays = dict(images=np.ascontiguousarray(np.concatenate([demo, np.stack(exec_frames)])),
                  states=states, wrist_image=np.ascontiguousarray(wrist))
    for source_record in sources:
        if V.sha_file(source_record["path"]) != source_record["sha256"]:
            raise ValueError("构建便携样本期间来源文件改变")
    tokenizer = v1 / "models/big_vision/paligemma_tokenizer.model"
    tokenizer_sha = V.sha_file(tokenizer)
    output_root = Path(output_root)
    payload_path = output_root / "online_restore_inputs.npz"
    fd, temporary = tempfile.mkstemp(prefix=".online_restore_inputs.", suffix=".npz", dir=output_root)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, payload_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    record = dict(schema="stage2-online-restore-inputs-v1", payload=dict(path=payload_path.name, sha256=V.sha_file(payload_path)),
                  exec_start_idx=es, exec_batch_lengths=[33, 32], prompt=prompt, sources=sources,
                  demo_state_origin=dict(path="framesamp-8x8/state_emb.f32.bin", row_start=start),
                  episode=dict(key=key, global_episode_idx=episode["global_episode_idx"], manifest_sha256=meta.manifest_sha256),
                  arrays={name: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=V.sha_bytes(value.tobytes()))
                          for name, value in arrays.items()},
                  tokenizer=dict(uri="gs://big_vision/paligemma_tokenizer.model", sha256=tokenizer_sha))
    record_path = output_root / "online_restore_inputs.json"
    V.atomic_json(record_path, record)
    return dict(path=str(record_path.resolve()), sha256=V.sha_file(record_path),
                payload=dict(path=str(payload_path.resolve()), sha256=record["payload"]["sha256"]))


def run(args):
    from mme_vla_suite.datastore import motion_store as ms
    from mme_vla_suite.policies.motion_client import MotionEncoderClient
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.online_gpu):
        raise ValueError("G4 父进程也须只暴露显式指定的同一张物理GPU")
    meta = ms.MotionMeta.load(args.store)
    ms.validate_stage2_provenance(meta.provenance)
    ms.run_fast_checks(meta)
    ms.run_full_checks(meta)
    if meta.raw["layout"] != V.LAYOUT or meta.status != "verified":
        raise ValueError("G4 必须消费已全表认证并 verified 的 stage2 表")
    samples_sha, history_sha = V.sha_file(args.samples), V.sha_file(args.history)
    samples = V.load_samples(args.samples)
    g0 = meta.provenance["stage2_compat"]["evidence"]["G0"]
    if g0["sha256"] != samples_sha or samples["manifest_sha256"] != meta.manifest_sha256:
        raise ValueError("G4 样本不是当前 store 固定的 G0")
    config = copy.deepcopy(yaml.safe_load(Path(args.history).read_text()))
    motion = config["motion"]
    expected = dict(enabled=True, encoder_kind="stage2_pool4_concat", dim=1024, budget=96, stride=32,
                    window_frames=33, demo_min_real_frames=17, demo_tail_pad="repeat_last")
    if any(motion.get(k) != value for k, value in expected.items()):
        raise ValueError("history motion 不满足 stage2 固定合同")
    enc = meta.provenance["encoder"]
    source_run = f"{enc['run_name']}/{enc['checkpoint_name']}#model"
    if motion.get("source_run") != source_run or motion.get("checkpoint_sha256") != enc["checkpoint_sha256"]:
        raise ValueError("history 的 encoder 身份与 store 不同")
    motion["online_gpu"] = args.online_gpu
    v1 = Path(args.v1_store or os.environ.get("MMEVLA_V1_STORE") or REPO / "v1-store").resolve()
    configured_store = Path(motion["store_path"])
    if not configured_store.is_absolute():
        configured_store = (v1 / Path(*configured_store.parts[1:]) if configured_store.parts[0] == "v1-store"
                            else REPO / configured_store)
    if configured_store.resolve() != meta.root.resolve():
        raise ValueError("history 的 motion.store_path 与本次验证表不同")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".g4-reference-", dir=Path(args.out).parent) as directory:
        reference, reference_record = reference_subprocess(args, meta, v1, directory)
    common = dict(encoder_kind="stage2_pool4_concat", online_gpu=motion["online_gpu"], source_run=source_run,
                  expected_ckpt_sha256=enc["checkpoint_sha256"], store_provenance=meta.provenance,
                  provenance_root=meta.root, v1_store=v1)
    real = MotionEncoderClient(**common, stub=False)
    try:
        require_handshake(real, stub=False)
        gpu = verify_gpu(real.provenance, args.online_gpu)
        cache, rows = {}, []
        for i, window in enumerate(samples["windows"]):
            result = real(V.read_window_frames(window, cache), window["frame_ids"][0])
            if result.shape != (1024,) or result.dtype != np.float32 or not np.isfinite(result).all():
                raise ValueError("真实客户端输出形制或有限性错误")
            delta = float(np.max(np.abs(result.astype(np.float64) - reference[i].astype(np.float64))))
            if delta > 1e-5:
                raise ValueError(f"端到端超过1e-5：{window['key']}:{window['source_grid']} max={delta}")
            rows.append(dict(key=window["key"], source_grid=window["source_grid"], max_abs=delta))
        handshake = copy.deepcopy(real.provenance)
    finally:
        real.close()
    stub = MotionEncoderClient(**common, stub=True)
    try:
        require_handshake(stub, stub=True)
        machine = check_state_machine(stub, motion, token_per_image=config["token_per_image"])
    finally:
        stub.close()
    if V.sha_file(args.samples) != samples_sha or V.sha_file(args.history) != history_sha:
        raise ValueError("G4 期间样本或 history YAML 改变")
    ms.run_full_checks(meta)
    if V.sha_file(meta.root / ms.INDEX_RELPATH) != meta.motion_index_sha256:
        raise ValueError("G4 期间 packed 索引改变")
    portable = prepare_restore_inputs(meta, samples, v1, Path(args.out).parent)
    return dict(schema="stage2-validation-gate-v1", gate="G4", status="PASS", producer=V.PRODUCER,
                checkpoint_sha256=enc["checkpoint_sha256"], samples_sha256=samples_sha, history_sha256=history_sha,
                metrics=dict(windows=64, handshake=1, response_bytes=4096, encoder_max_abs=0,
                             e2e_max_abs=max(row["max_abs"] for row in rows), **machine),
                windows=rows, handshake=handshake, physical_gpu=gpu, same_latent=reference_record,
                online_restore_inputs=portable)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "samples", "history", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--online-gpu", required=True)
    parser.add_argument("--v1-store")
    args = parser.parse_args(argv)
    V.revoke_report(args.out, gate="G4")
    try:
        record = run(args)
        V.atomic_json(args.out, record)
    except Exception as exc:
        V.atomic_json(args.out, dict(schema="stage2-validation-gate-v1", gate="G4", status="FAIL",
                                     producer=V.PRODUCER, error=str(exc)))
        raise
    print("ONLINE_OFFLINE_PARITY=PASS " + " ".join(f"{k}={v}" for k, v in record["metrics"].items()))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    raise SystemExit(main())
