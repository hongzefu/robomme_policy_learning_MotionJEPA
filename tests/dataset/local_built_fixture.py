"""1114 计划 S2 测试夹具：合成 4task 旧库（episode 清单、wan-latents grid16、单集 H5、episode_map）。

形状与真实 4task-v2-1600ep-604f16da 同构（统一编号 raw_ep_idx=new_idx，单集 H5 组名用 raw_episode），
体量缩到 4 任务×4 episode、每 episode 120 帧（demo 49 帧＋exec 71 帧）。
"""
import hashlib
import json
import pathlib
from types import SimpleNamespace

import h5py
import numpy as np

from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.datastore.manifest import manifest_sha256

TASKS = ("BinFill", "RouteStick", "VideoRepick", "VideoUnmaskSwap")
EPISODES, T, ES = 4, 120, 49
CHUNK = 589824


def pixel(task_index, episode, t):
    return np.full((256, 256, 3), (task_index * 31 + episode * 7 + t) % 251, dtype=np.uint8)


def _dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def build(root: pathlib.Path):
    root = pathlib.Path(root)
    latents, raw, maps = root / "wan-latents", root / "raw-h5", root / "maps"
    for p in (latents, raw, maps):
        p.mkdir(parents=True)
    episodes, offset, total_t, segments = [], 0, 0, {}
    map_rows = {task: [] for task in TASKS}
    g = 0
    for ti, task in enumerate(TASKS):
        for ep in range(EPISODES):
            raw_ep = ep + 2 * (ti % 2)            # 一半任务 raw_episode ≠ 统一编号
            member = f"record_dataset_{task}_easy/hdf5_files/{task}_ep{raw_ep}_seed{1000 + ep}.h5"
            path = raw / member
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as handle:
                group = handle.create_group(f"episode_{raw_ep}")
                for t in range(T):
                    group.create_dataset(f"timestep_{t}/obs/front_rgb", data=pixel(ti, ep, t),
                                         compression="gzip")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            map_rows[task].append(dict(task=task, difficulty="easy", episode=raw_ep, seed=1000 + ep, member=member,
                                       h5_sha256=sha, h5_bytes=path.stat().st_size, new_idx=ep,
                                       src_group=f"episode_{raw_ep}", h5_sha256_local=sha))
            h5_file = f"record_dataset_{task}.h5"
            episodes.append(dict(global_episode_idx=g, h5_file=h5_file, raw_ep_idx=ep, num_timesteps=T,
                                 exec_start_idx=ES, exec_samples=T - ES, exec_sample_offset=offset,
                                 total_sample_offset=total_t, shard_idx=0))
            for kind, start, length, minimum in (("demo", 0, ES, 17), ("exec", ES, T - ES, 33)):
                count = (length - minimum) // 16 + 1
                key = f"{task}_ep{ep}_{kind}"
                rows, blob = [], bytearray()
                for m in range(count):
                    real = min(33, length - m * 16)
                    s0 = start + m * 16
                    ids = list(range(s0, s0 + real)) + [s0 + real - 1] * (33 - real)
                    pixels = np.ascontiguousarray(np.stack([pixel(ti, ep, i) for i in ids]))
                    rows.append(dict(m=m, seg_offset=m * 16, start_global_frame=s0, input_shape=[33, 256, 256, 3],
                                     input_dtype="uint8",
                                     input_frames_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                                     real_frames=real, pad_frames=33 - real,
                                     pad_source_frame=s0 + real - 1 if real < 33 else None))
                    blob += (np.full(CHUNK // 4, g * 100 + m + (0.5 if kind == "exec" else 0), "<f4")).tobytes()
                (latents / f"{key}.bin").write_bytes(bytes(blob))
                digest = hashlib.sha256(bytes(blob)).hexdigest()
                (latents / f"{key}.bin.sha256").write_text(f"{digest}  {key}.bin\n")
                vae = dict(vae_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", vae_state_sha256="9" * 64, vae_dtype="float32",
                           latent_mode="mode", batch=1, tf32="off", amp="off", torch="2.9.0", cuda="12.8",
                           cudnn=91002, diffusers="0.39.0", module_sha256="1" * 64, encoder_src_sha256="2" * 64)
                meta = dict(schema=3, segment=key, g=g, h5_file=h5_file, raw_ep_idx=ep, segment_kind=kind,
                            seg_start_global=start, seg_len=length, num_grid=count,
                            num_chunks=length - minimum + 1, min_real=minimum, grid_stride=16, window_frames=33,
                            grid_origin="segment_start", window_direction="forward", truncation_policy="none",
                            frame_size=256, latent_shape=[9, 16, 32, 32], latent_dtype="float32", chunk_bytes=CHUNK,
                            sha256=digest, bytes=len(blob), rows=rows, vae=vae,
                            worker=dict(hostname="fixture", gpu_uuid="GPU-0", worker="gpu0", pid=1))
                _dump(latents / f"{key}.metadata.json", meta)
                segments[key] = dict(num_grid=count, seg_len=length, seg_start_global=start, segment_kind=kind,
                                     g=g, sha256=digest, bytes=len(blob), min_real=minimum, rows=rows)
            offset += T - ES
            total_t += T
            g += 1
    manifest = dict(version=1, raw_dir="/scratch/hongze/合成原数据", canonical_order=[f"record_dataset_{t}.h5" for t in TASKS],
                    num_shards=1, shard_load_timesteps=[total_t], episodes=episodes,
                    totals=dict(episodes=len(episodes), timesteps=total_t, exec_samples=offset))
    manifest["sha256"] = manifest_sha256(manifest)
    manifest_path = root / "meta/episode_manifest.json"
    _dump(manifest_path, manifest)
    _dump(root / "meta/input_manifest.json", dict(files={f"record_dataset_{t}.h5": dict(size=1, sha256="a" * 64)
                                                         for t in TASKS}))
    total = dict(schema=3, stage="wan", grid_stride=16, window_frames=33, frame_size=256, demo_min_real_frames=17,
                 exec_min_real_frames=33, demo_tail_pad="repeat_last", grid_origin="segment_start",
                 window_direction="forward", truncation_policy="none", manifest_sha256=manifest["sha256"],
                 segments=segments)
    _dump(latents / "metadata.json", total)
    for task, rows in map_rows.items():
        (maps / f"record_dataset_{task}_episode_map.json").write_text(json.dumps(rows))
    merge_done = root / "MERGE_DONE.json"
    _dump(merge_done, dict(maps_sha256={task: ms.sha256_file(maps / f"record_dataset_{task}_episode_map.json")
                                        for task in TASKS}))
    return SimpleNamespace(root=root, latents=latents, raw=raw, maps=maps, merge_done=merge_done, manifest=manifest,
                           manifest_path=manifest_path, total_sha=ms.sha256_file(latents / "metadata.json"))
