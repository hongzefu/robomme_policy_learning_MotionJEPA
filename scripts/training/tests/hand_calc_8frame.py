"""以手写公式核对 8 帧采样、段边界、可见运动窗口与稳定交错。"""

import argparse
import json
import pathlib
from types import SimpleNamespace

import numpy as np

from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.training.dataloader import _create_framesamp_dataset


def expected_frames(step):
    return list(range(step + 1)) if step < 8 else [i * step // 7 for i in range(8)]


def expected_motion(ep, step):
    es = ep["exec_start_idx"]
    return [s for s in range(0, es, 16) if s + 32 < es] + [s for s in range(es, step + 1, 16) if s + 32 <= step]


def expected_order(frames, motions):
    sentinel = 2**31 - 1
    ft = frames + [sentinel] * (8 - len(frames))
    mt = motions + [sentinel] * (96 - len(motions))
    keys = [(t, 0, i * 64 + j) for i, t in enumerate(ft) for j in range(64)]
    keys += [(t, 1, 512 + i) for i, t in enumerate(mt)]
    return np.asarray([i for _, _, i in sorted(keys)], dtype=np.int32)


def main():
    import os
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=pathlib.Path, required=True)
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--motion", type=pathlib.Path, required=True)
    args = ap.parse_args()
    os.environ["MMEVLA_MOTION_STORE"] = str(args.motion.resolve())
    manifest = json.loads(args.manifest.read_text())
    entries = {e["g"]: e for e in json.loads((args.motion / "meta/motion_index.json").read_text())["entries"]}
    meta = json.loads((args.store / "meta/store_meta.json").read_text())
    source = pathlib.Path(meta["source_dataset_root"])
    stats = SimpleNamespace(q01=np.zeros(8, np.float64), q99=np.ones(8, np.float64))
    dc = SimpleNamespace(norm_stats={"state": stats}, use_quantile_norm=True)
    eps = []
    for es in (0, 66, 114):
        eps.append(next(ep for ep in manifest["episodes"] if ep["exec_start_idx"] == es))
    for suffix in ("", "-motion"):
        hc = get_history_config(f"perceptual-framesamp-context-8frame-8x8{suffix}.yaml")
        ds = _create_framesamp_dataset(str(args.store), dc, hc, 20)
        n = 0
        try:
            for ep in eps:
                es, g = ep["exec_start_idx"], ep["global_episode_idx"]
                for step in sorted({es + t for t in (0, 6, 7, 8, 9, 31, 32, 33)} | {ep["num_timesteps"] - 1}):
                    if step >= ep["num_timesteps"]:
                        continue
                    idx = ep["exec_sample_offset"] + step - es
                    item = ds[idx]
                    frames = expected_frames(step)
                    if ds._max_frames != 8 or ds._tokens_per_frame != 64:
                        raise ValueError("Dataset 未使用 8×64")
                    image = item["static_image_emb"].reshape(8, 64, 2048)
                    pos = item["static_pos_emb"].reshape(8, 64, 768)
                    for i, frame in enumerate(frames):
                        raw = np.load(source / "features" / f"episode_{g}" / f"token_emb_{frame}.npy", allow_pickle=True).item()
                        if image[i].tobytes() != raw["image_emb_8x8"][0].tobytes() or pos[i].tobytes() != raw["pos_emb_8x8"][0].tobytes():
                            raise ValueError(f"手算帧索引与交付不符: g={g} step={step} frame={frame}")
                    mask = np.arange(512) < len(frames) * 64
                    if item["static_mask"].tobytes() != mask.tobytes() or np.any(image[len(frames):]) or np.any(pos[len(frames):]):
                        raise ValueError("手算补零或 mask 不符")
                    if suffix:
                        motions = expected_motion(ep, step)
                        ent = entries[g]
                        rows = [(ent["demo"]["row_base"] + f // 16) if f < es else
                                (ent["exec"]["row_base"] + (f - es) // 16) for f in motions]
                        # 直接从 bin 按手算行号读取，避免与候选共享 MotionStore.rows。
                        table_path = args.motion / "motion_token.f32.bin"
                        with table_path.open("rb") as f:
                            for i, row in enumerate(rows):
                                f.seek(row * 768 * 4)
                                if item["motion_emb"][i].tobytes() != f.read(768 * 4):
                                    raise ValueError("手算 motion 行不符")
                        if item["motion_mask"].tobytes() != (np.arange(96) < len(motions)).tobytes():
                            raise ValueError("手算 motion 窗数不符")
                        if item["mem_order"].tobytes() != expected_order(frames, motions).tobytes():
                            raise ValueError("手算 mem_order 不符")
                    elif any(item[k] is not None for k in ("motion_emb", "motion_pos", "motion_mask", "mem_order")):
                        raise ValueError("关闭态出现 motion 交付")
                    n += 1
        finally:
            ds.close()
        print(f"HAND_CALC_8FRAME=PASS samples={n} mismatches=0 profile={'m8' if suffix else 'c8'}")


if __name__ == "__main__":
    main()
