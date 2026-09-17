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


def expected_motion(ep, step, demo_min_real=33):
    es = ep["exec_start_idx"]
    return [s for s in range(0, es, 16) if s + demo_min_real <= es] + [s for s in range(es, step + 1, 16) if s + 32 <= step]


def expected_order(frames, motions, budget=96):
    sentinel = 2**31 - 1
    ft = frames + [sentinel] * (8 - len(frames))
    mt = motions + [sentinel] * (budget - len(motions))
    keys = [(t, 0, i * 64 + j) for i, t in enumerate(ft) for j in range(64)]
    keys += [(t, 1, 512 + i) for i, t in enumerate(mt)]
    return np.asarray([i for _, _, i in sorted(keys)], dtype=np.int32)


def select_cases(manifest, minimum, *, v2_coverage=False):
    """独立选择全部冷启动、首 exec 窗、高 k 样本及补帧残差层。"""
    cases = set()
    residues = {}
    cold = first = high = 0
    kmax = 0
    for ep in manifest["episodes"]:
        es, total, g = ep["exec_start_idx"], ep["num_timesteps"], ep["global_episode_idx"]
        cases.add((g, es)); cold += 1
        if es+32 < total:
            cases.add((g, es+32)); first += 1
        residues.setdefault((es-minimum) % 16, []).append((g, es))
        d = len(range(0, max(0, es-minimum+1), 16))
        for delta in range(total-es):
            k = d + (0 if delta < 32 else (delta-32)//16+1)
            kmax = max(kmax, k)
            if k >= 140:
                cases.add((g, es+delta)); high += 1
    for group in residues.values():
        cases.update(group[:2])
    coverage = dict(cold=cold, first_exec=first, kmax=kmax, kge140=high,
                    pad_r=len(residues), pad_r_min=min(map(len, residues.values())))
    if v2_coverage and ((cold, first, kmax, high, len(residues)) != (1600,1600,141,32,16)
                        or coverage["pad_r_min"] < 2):
        raise ValueError(f"V4 覆盖集合不符: {coverage}")
    return sorted(cases), coverage


def main():
    import os
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=pathlib.Path, required=True)
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--motion", type=pathlib.Path, required=True)
    ap.add_argument("--yaml", default="perceptual-framesamp-modul-8frame-8x8-motion.yaml")
    ap.add_argument("--v2-coverage", action="store_true")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    os.environ["MMEVLA_MOTION_STORE"] = str(args.motion.resolve())
    manifest = json.loads(args.manifest.read_text())
    entries = {e["g"]: e for e in json.loads((args.motion / "meta/motion_index.json").read_text())["entries"]}
    meta = json.loads((args.store / "meta/store_meta.json").read_text())
    source = pathlib.Path(meta["source_dataset_root"])
    stats = SimpleNamespace(q01=np.zeros(8, np.float64), q99=np.ones(8, np.float64))
    dc = SimpleNamespace(norm_stats={"state": stats}, use_quantile_norm=True)
    hc = get_history_config(args.yaml)
    enabled = bool(hc.get("motion", {}).get("enabled", False))
    minimum = int(hc.motion.get("demo_min_real_frames", 33)) if enabled else 33
    budget = int(hc.motion.budget) if enabled else 0
    if args.v2_coverage and (not enabled or minimum != 17 or budget != 160):
        raise ValueError("V4 正式覆盖必须使用开启态、17 帧与 budget=160")
    cases, coverage = select_cases(manifest, minimum, v2_coverage=args.v2_coverage)
    ds = _create_framesamp_dataset(str(args.store), dc, hc, 20)
    pos_meta = meta["tables"]["pos_emb_8x8"]
    n = 0
    try:
        with (args.motion / "motion_token.f32.bin").open("rb") as tokens, (args.store / pos_meta["relpath"]).open("rb") as positions:
            for g, step in cases:
                ep = manifest["episodes"][g]
                es = ep["exec_start_idx"]
                idx = ep["exec_sample_offset"] + step-es
                item = ds[idx]
                frames = expected_frames(step)
                if ds._max_frames != 8 or ds._tokens_per_frame != 64:
                    raise ValueError("Dataset 未使用 8×64")
                image = item["static_image_emb"].reshape(8,64,2048)
                pos = item["static_pos_emb"].reshape(8,64,768)
                for i, frame in enumerate(frames):
                    raw = np.load(source / "features" / f"episode_{g}" / f"token_emb_{frame}.npy", allow_pickle=True).item()
                    if image[i].dtype != raw["image_emb_8x8"].dtype or image[i].tobytes() != raw["image_emb_8x8"][0].tobytes():
                        raise ValueError(f"手算帧内容不符: g={g} step={step} frame={frame}")
                    if pos[i].dtype != np.float32 or pos[i].tobytes() != raw["pos_emb_8x8"][0].tobytes():
                        raise ValueError("手算帧位置编码不符")
                mask = np.arange(512) < len(frames)*64
                if item["static_mask"].tobytes() != mask.tobytes() or np.any(image[len(frames):]) or np.any(pos[len(frames):]):
                    raise ValueError("手算帧补零或 mask 不符")
                if enabled:
                    motions = expected_motion(ep, step, minimum)
                    ent = entries[g]
                    rows = [(ent["demo"]["row_base"] + f//16) if f < es else
                            (ent["exec"]["row_base"] + (f-es)//16) for f in motions]
                    if item["motion_emb"].shape != (budget,768) or item["motion_emb"].dtype != np.float32:
                        raise ValueError("motion_emb 形状或 dtype 不符")
                    if item["motion_pos"].shape != (budget,256) or item["motion_pos"].dtype != np.float32:
                        raise ValueError("motion_pos 形状或 dtype 不符")
                    for i, (row, frame) in enumerate(zip(rows, motions, strict=True)):
                        tokens.seek(row*768*4)
                        if item["motion_emb"][i].tobytes() != tokens.read(768*4):
                            raise ValueError(f"手算 motion 行不符: g={g} step={step} row={row}")
                        positions.seek(frame*int(pos_meta["row_bytes"]))
                        if item["motion_pos"][i].tobytes() != positions.read(256*4):
                            raise ValueError("motion_pos 未取起点帧第一格的时间码")
                    count = len(motions)
                    if item["motion_mask"].dtype != np.bool_ or item["motion_mask"].tobytes() != (np.arange(budget)<count).tobytes():
                        raise ValueError("手算 motion mask 不符")
                    if any(np.ascontiguousarray(item[k][count:]).view(np.uint8).any() for k in ("motion_emb","motion_pos")):
                        raise ValueError("motion 右填充区不是全零位串")
                    if item["mem_order"].dtype != np.int32 or item["mem_order"].tobytes() != expected_order(frames,motions,budget).tobytes():
                        raise ValueError("手算 mem_order 不符")
                elif any(item[k] is not None for k in ("motion_emb","motion_pos","motion_mask","mem_order")):
                    raise ValueError("关闭态出现 motion 交付")
                n += 1
                if n % 200 == 0:
                    print(f"HAND_CALC_PROGRESS samples={n}/{len(cases)}", flush=True)
    finally:
        ds.close()
    print(f"HAND_CALC_8FRAME=PASS samples={n} mismatches=0 yaml={args.yaml}", flush=True)
    if args.v2_coverage:
        print("V4_COVER=PASS cold=1600 first_exec=1600 kmax=141 kge140=32 pad_r=16/16", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as f:
            json.dump({"yaml": args.yaml, "samples": n, "coverage": coverage, "cases": cases}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
