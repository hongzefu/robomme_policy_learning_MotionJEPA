#!/usr/bin/env python3
"""split（4×256）与concat（1×1024）训练输入逐样本对拍；只读，不写任何文件。

两侧都走现有训练读取路径（dataloader._motion_gates + FrameSampDataset.__getitem__），
在同一批样本上断言：窗口行号/起点一致；split每窗4槽对应concat 1槽的时间码与有效mask；
split.reshape(96,1024)与concat逐位相等；mem_order按窗口折叠后一致；帧路及其余键逐位相等。
末行 NOSPLIT_INPUT=PASS|FAIL samples=<n> windows=<n> mismatches=<n>。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LIBRARY = "v1-store/datasets/16task-orig100-1600ep-9e109419"
YAML_DIR = "src/mme_vla_suite/models/config/robomme"
MOTION_KEYS = ("motion_emb", "motion_pos", "motion_mask", "mem_order")
SPLIT_TOKENS, WINDOWS, DIM = 4, 96, 1024


def history_name(fusion, readout):
    return f"perceptual-framesamp-{fusion}-32frame-4x4-motion-stage2-{readout}-orig16.yaml"


def pick_samples(manifest, concat_ds):
    """按清单自动挑选：demo（demo窗口最多的执行首步）、exec（中段）、短尾（最短集末步）、最大窗口（可见窗口最多的末步）。"""
    from mme_vla_suite.datastore import motion_store as ms
    episodes = manifest["episodes"]

    def index(ep, step):
        return ep["exec_sample_offset"] + step - ep["exec_start_idx"]

    def visible(ep, step):
        return len(ms.visible_motion_rows(concat_ds._motion_entries[ep["global_episode_idx"]], step)[0])

    first = episodes[0]
    # demo样本取执行首步可见demo窗口最多的集，保证覆盖demo段窗口而非空窗口。
    demo = max(episodes, key=lambda ep: (visible(ep, ep["exec_start_idx"]), -ep["global_episode_idx"]))
    shortest = min(episodes, key=lambda ep: (ep["num_timesteps"] - ep["exec_start_idx"], ep["global_episode_idx"]))
    largest = max(episodes, key=lambda ep: (visible(ep, ep["num_timesteps"] - 1), -ep["global_episode_idx"]))
    mid = (first["exec_start_idx"] + first["num_timesteps"] - 1) // 2
    picks = [("demo", index(demo, demo["exec_start_idx"])), ("exec", index(first, mid)),
             ("short_tail", index(shortest, shortest["num_timesteps"] - 1)),
             ("max_window", index(largest, largest["num_timesteps"] - 1))]
    seen, result = set(), []
    for kind, idx in picks:
        if idx not in seen:
            seen.add(idx)
            result.append((kind, idx))
    return result


def collapse_order(order, frame_tokens):
    """split的mem_order把同窗4槽折叠为1槽，得到与concat可比的顺序。"""
    out = []
    for value in np.asarray(order).tolist():
        item = value if value < frame_tokens else frame_tokens + (value - frame_tokens) // SPLIT_TOKENS
        if not out or item != out[-1] or item < frame_tokens:
            out.append(item)
    return np.asarray(out)


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    return type(a) is type(b) and a == b


def compare_sample(split_ds, concat_ds, idx):
    """返回 (窗口数, 不符项列表)；不抛断言，逐项记账。"""
    from mme_vla_suite.datastore import motion_store as ms
    bad = []
    g, step = int(concat_ds._epis_of[idx]), int(concat_ds._step_of[idx])
    if (int(split_ds._epis_of[idx]), int(split_ds._step_of[idx])) != (g, step):
        bad.append("sample_identity")
    rows_s, starts_s = ms.visible_motion_rows(split_ds._motion_entries[g], step)
    rows_c, starts_c = ms.visible_motion_rows(concat_ds._motion_entries[g], step)
    k = len(rows_c)
    if not (np.array_equal(rows_s, rows_c) and np.array_equal(starts_s, starts_c)):
        bad.append("window_rows_or_starts")
    s, c = split_ds[idx], concat_ds[idx]
    if c["motion_emb"].shape != (WINDOWS, DIM) or s["motion_emb"].shape != (WINDOWS * SPLIT_TOKENS, DIM // SPLIT_TOKENS):
        bad.append("motion_shape")
    elif s["motion_emb"].reshape(WINDOWS, 1, DIM).tobytes() != c["motion_emb"].reshape(WINDOWS, 1, DIM).tobytes():
        bad.append("motion_bits")
    elif k and concat_ds._ensure_motion_store().rows(rows_c).tobytes() != c["motion_emb"][:k].tobytes():
        bad.append("concat_raw_rows")
    if not same(s["motion_mask"], np.repeat(c["motion_mask"], SPLIT_TOKENS)) or int(c["motion_mask"].sum()) != k:
        bad.append("motion_mask")
    if not same(s["motion_pos"], np.repeat(c["motion_pos"], SPLIT_TOKENS, axis=0)):
        bad.append("motion_timecode")
    frame_tokens = len(c["static_mask"])
    if len(s["mem_order"]) != frame_tokens + WINDOWS * SPLIT_TOKENS or len(c["mem_order"]) != frame_tokens + WINDOWS \
            or not np.array_equal(collapse_order(s["mem_order"], frame_tokens), c["mem_order"]):
        bad.append("mem_order")
    if set(s) != set(c):
        bad.append("keys")
    for key in sorted(set(s) & set(c) - set(MOTION_KEYS)):
        if not same(s[key], c[key]):
            bad.append(f"frame_or_data:{key}")
    return k, bad


def compare(split_ds, concat_ds, samples, *, out=print):
    windows = mismatches = 0
    for kind, idx in samples:
        k, bad = compare_sample(split_ds, concat_ds, idx)
        windows += k
        mismatches += len(bad)
        out(f"NOSPLIT_SAMPLE kind={kind} idx={idx} windows={k} {'ok=1' if not bad else 'FAIL=' + ','.join(bad)}")
    return windows, mismatches


def build_datasets(library, repo, fusion):
    """按训练入口同一闸门与构造参数取两侧数据集；motion根与源根绑定到给定库。"""
    from omegaconf import OmegaConf
    from openpi.shared import normalize
    from mme_vla_suite.datastore import StoreMeta
    from mme_vla_suite.training import dataloader
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset
    from types import SimpleNamespace
    library = Path(library).resolve()
    frames = library / "framesamp-4x4"
    norm_dir = Path(repo) / "v1-store/train-assets/mme_vla_suite" / library.name / "robomme"
    data_config = SimpleNamespace(norm_stats=normalize.load(norm_dir), use_quantile_norm=False)
    result = []
    for readout in ("4x256", "1x1024"):
        history = OmegaConf.load(Path(repo) / YAML_DIR / history_name(fusion, readout))
        history.motion.store_path = str(library / "motion-stage2-1024")
        motion_root = dataloader._motion_gates(history, StoreMeta.load(frames))
        result.append(FrameSampDataset(str(frames), data_config, history, 20, source_root=str(library / "source"),
                                       manifest_path=str(library / "meta/episode_manifest.json"), motion_root=motion_root))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default=str(REPO / LIBRARY))
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--fusion", choices=("ctx", "modul"), default="ctx")
    parser.add_argument("--max-samples", type=int, default=0, help="只取自动挑选的前N个样本（0为全部）")
    args = parser.parse_args(argv)
    from mme_vla_suite.datastore import load_manifest
    split_ds, concat_ds = build_datasets(args.library, args.repo, args.fusion)
    try:
        samples = pick_samples(load_manifest(Path(args.library) / "meta/episode_manifest.json"), concat_ds)
        if args.max_samples:
            samples = samples[:args.max_samples]
        windows, mismatches = compare(split_ds, concat_ds, samples)
    finally:
        split_ds.close()
        concat_ds.close()
    ok = mismatches == 0 and bool(samples)
    print(f"NOSPLIT_INPUT={'PASS' if ok else 'FAIL'} samples={len(samples)} windows={windows} mismatches={mismatches}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
