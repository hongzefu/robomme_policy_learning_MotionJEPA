"""A1：context 1024 两档打包库读取 vs MemoryBuffer（读 token_emb_*.npy）逐样本等价（1106 计划）。

- td：token_dropping (1024,64,1)，framesamp-8x8，比 MemoryBuffer.prepare_token_dropping（kept_indices 读 JSON）；
- fs：frame_sampling (1024,16,1)，framesamp-4x4，比 MemoryBuffer.prepare_frame_sampling；
- reg512：既有 (512,16,1) frame_sampling 回归，framesamp-4x4，motion 关闭。

比较口径：image 转 float32 后比数值，pos 比字节，state 比「补零后归一化」，mask 比字节。
抽样固定 seed，按任务均分覆盖全部 16 任务，每任务约 1/4 取超预算样本。
库未 verified（或仍持 pack.lock）直接判 FAIL，不比较；--allow-unverified 只供开发期试跑，末行不会是 PASS。
末行：LOADER_EQ=PASS td=<n> fs=<n> mismatch=0，否则 LOADER_EQ=FAIL ...

只读。用 venv 解释器运行；worktree 内另设 PYTHONPATH=<worktree>/src:<worktree>/packages/openpi-client/src。
"""

from __future__ import annotations

import argparse
from collections import Counter
import functools
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ["JAX_PLATFORMS"] = "cpu"  # 不占 GPU；spawn 子进程继承
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

LIB = "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets/16task-orig100-1600ep-9e109419"
ASSETS = "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-assets/mme_vla_suite/16task-orig100-1600ep-9e109419"
YAMLS = ROOT / "src/mme_vla_suite/models/config/robomme"
ACTION_HORIZON = 20
MODES = {
    "td": dict(yaml="perceptual-tokendrop-context-1024.yaml", store="framesamp-8x8", budget=1024, tokens=64),
    "fs": dict(yaml="perceptual-framesamp-context-1024.yaml", store="framesamp-4x4", budget=1024, tokens=16),
    "reg512": dict(yaml="perceptual-framesamp-context.yaml", store="framesamp-4x4", budget=512, tokens=16),
}


def read_kept(source_root, g):
    with open(f"{source_root}/features/episode_{g}/kept_indices.json", encoding="utf-8") as f:
        return json.load(f)


def over_budget_from(mode, source_root, g):
    """该集最小的超预算 step：此后可见帧数／token 数大于预算；None 表示整集都不超。"""
    cfg = MODES[mode]
    if mode != "td":
        return cfg["budget"] // cfg["tokens"]
    frames = sorted(item[0] for item in read_kept(source_root, g))
    return frames[cfg["budget"]] if len(frames) > cfg["budget"] else None


def plan_samples(mode, n, manifest, source_root, seed):
    """n 个不重复样本：按 h5_file 均分，每任务约 1/4 取超预算（池子够时），其余在该任务均匀抽。"""
    by_task = {}
    for ep in manifest["episodes"]:
        k = int(ep["exec_samples"])
        idx = int(ep["exec_sample_offset"]) + np.arange(k)
        steps = int(ep["exec_start_idx"]) + np.arange(k)
        start = over_budget_from(mode, source_root, ep["global_episode_idx"])
        over = idx[steps >= start] if start is not None else idx[:0]
        all_list, over_list = by_task.setdefault(ep["h5_file"], ([], []))
        all_list.append(idx)
        over_list.append(over)
    rng = np.random.default_rng(seed)
    tasks = sorted(by_task)
    chosen = []
    for i, task in enumerate(tasks):
        quota = n // len(tasks) + (i < n % len(tasks))
        everything = np.concatenate(by_task[task][0])
        over = np.concatenate(by_task[task][1])
        take_over = min(len(over), quota // 4)
        a = rng.choice(over, take_over, replace=False)
        b = rng.choice(np.setdiff1d(everything, a), quota - take_over, replace=False)
        chosen.extend(zip(np.concatenate([a, b]).tolist(), [task] * quota))
    return chosen


# ―― worker ――
_ds_cache = {}


def _dataset(mode, lib, assets):
    if mode not in _ds_cache:
        from omegaconf import OmegaConf
        from openpi.shared import normalize

        from mme_vla_suite.training.framesamp_dataset import FrameSampDataset

        cfg = MODES[mode]
        history = OmegaConf.load(YAMLS / cfg["yaml"])
        if "motion" in history:
            history.motion.enabled = False
        # 与 config.py 同：非 PI0 模型 use_quantile_norm=True
        data_config = SimpleNamespace(norm_stats=normalize.load(Path(assets) / "robomme"), use_quantile_norm=True)
        _ds_cache[mode] = FrameSampDataset(f"{lib}/{cfg['store']}", data_config, history, ACTION_HORIZON)
    return _ds_cache[mode]


def _gather(indices, source_root, g):
    return {t: np.load(f"{source_root}/features/episode_{g}/token_emb_{t}.npy", allow_pickle=True).item()
            for t in indices}


@functools.lru_cache(maxsize=64)
def _kept_cached(source_root, g):
    return tuple(tuple(item) for item in read_kept(source_root, g))


def compare_sample(job):
    """返回 (mode, task, idx, over_budget, [差异描述])；异常算差异，不吞掉。"""
    mode, idx, task, lib, assets = job
    try:
        from mme_vla_suite.dataset_builder.mem_buffer import MemoryBuffer

        ds = _dataset(mode, lib, assets)
        cfg = MODES[mode]
        g, step = int(ds._epis_of[idx]), int(ds._step_of[idx])
        src = ds._source_root
        buf = MemoryBuffer()
        if mode == "td":
            kept = _kept_cached(src, g)
            over = sum(item[0] <= step for item in kept) > cfg["budget"]
            ref = buf.prepare_token_dropping(step, cfg["budget"], _gather, kept, src, g)
        else:
            over = step + 1 > cfg["budget"] // cfg["tokens"]
            ref = buf.prepare_frame_sampling(step, cfg["budget"], cfg["tokens"], _gather, src, g)
        ref_img, ref_pos, ref_stt, ref_mask = ref
        got = ds[idx]
        diffs = []
        img = got["static_image_emb"]
        if img.dtype.name != "bfloat16":
            diffs.append(f"image_dtype={img.dtype}")
        if img.shape != ref_img.shape or not np.array_equal(img.astype(np.float32), ref_img.astype(np.float32)):
            diffs.append("image")
        for name, a, b in (("pos", got["static_pos_emb"], ref_pos),
                           ("state", got["static_state_emb"], ds._normalize_state(ref_stt)),
                           ("mask", got["static_mask"], ref_mask)):
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                diffs.append(name)
        if len(got["actions"]) > ACTION_HORIZON or "simple_subgoal_online" in got or got["motion_emb"] is not None:
            diffs.append("other_keys")
        if not got["static_mask"].any():
            diffs.append("empty_mask")
        return mode, task, idx, bool(over), diffs
    except Exception as exc:  # noqa: BLE001 — 计入 mismatch，末行据此判 FAIL
        return mode, task, idx, False, [f"exception {type(exc).__name__}: {exc}"]


def store_ready(root):
    from mme_vla_suite.datastore import StoreMeta, require_no_pack_lock, require_verified

    try:
        require_verified(StoreMeta.load(root))
        require_no_pack_lock(root)
    except Exception as exc:  # noqa: BLE001 — 任何未就绪都如实上报
        return f"{type(exc).__name__}: {exc}"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lib", default=LIB)
    parser.add_argument("--assets-dir", default=ASSETS)
    parser.add_argument("--td-n", type=int, default=1000)
    parser.add_argument("--fs-n", type=int, default=1000)
    parser.add_argument("--reg-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--allow-unverified", action="store_true", help="开发期试跑未 verified 的库；末行不会是 PASS")
    args = parser.parse_args()

    import mme_vla_suite

    origin = Path(mme_vla_suite.__file__).resolve()
    print(f"mme_vla_suite.__file__={origin}", flush=True)
    if ROOT.resolve() not in origin.parents:
        print(f"LOADER_EQ=FAIL reason=wrong_checkout expected_under={ROOT}", flush=True)
        return 2

    from mme_vla_suite.datastore import StoreMeta, load_manifest

    unverified = []
    for store in sorted({cfg["store"] for cfg in MODES.values()}):
        reason = store_ready(f"{args.lib}/{store}")
        if reason:
            print(f"STORE_NOT_READY store={store} {reason}", flush=True)
            unverified.append(store)
    if unverified and not args.allow_unverified:
        print(f"LOADER_EQ=FAIL reason=store_not_ready stores={','.join(unverified)} td=0 fs=0", flush=True)
        return 2

    meta = StoreMeta.load(f"{args.lib}/framesamp-4x4")
    manifest = load_manifest(meta.manifest_path)
    source_root = meta.source_dataset_root
    counts = {"td": args.td_n, "fs": args.fs_n, "reg512": args.reg_n}
    jobs = []
    for offset, (mode, n) in enumerate(counts.items()):
        jobs += [(mode, idx, task, args.lib, args.assets_dir)
                 for idx, task in plan_samples(mode, n, manifest, source_root, args.seed + offset)]

    done, over, tasks, bad = Counter(), Counter(), {m: set() for m in counts}, Counter()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for mode, task, idx, is_over, diffs in pool.imap_unordered(compare_sample, jobs, chunksize=4):
            done[mode] += 1
            over[mode] += is_over
            tasks[mode].add(task)
            if diffs:
                bad[mode] += 1
                if sum(bad.values()) <= 20:
                    print(f"MISMATCH mode={mode} idx={idx} task={task} {diffs}", flush=True)

    ok = True
    for mode, n in counts.items():
        covered = len(tasks[mode]) == 16 and over[mode] > 0 and done[mode] == n
        ok = ok and covered
        print(f"MODE {mode} n={done[mode]} tasks={len(tasks[mode])} over_budget={over[mode]} "
              f"mismatch={bad[mode]} coverage={'ok' if covered else 'FAIL'}", flush=True)
    mismatch = sum(bad.values())
    print(f"REG512={'PASS' if not bad['reg512'] and done['reg512'] == args.reg_n else 'FAIL'} "
          f"n={done['reg512']} mismatch={bad['reg512']}", flush=True)
    verdict = "PASS" if ok and mismatch == 0 else "FAIL"
    if verdict == "PASS" and unverified:
        verdict = "UNVERIFIED_STORE"
    print(f"LOADER_EQ={verdict} td={done['td']} fs={done['fs']} mismatch={mismatch}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
