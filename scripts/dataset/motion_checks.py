#!/usr/bin/env python3
"""S1 附加检查（主 venv、numpy + h5py；motion-memory-plan.md 第二部分四节表二 A5 / A6 / A7 / A9（集合半） / A10）。

  a5      原始帧同源：40 ep 库每帧 `front_rgb` 与 4env400ep 同 (task, raw_ep_idx) 逐帧 sha256 相等（13,756 帧）；
          另与 MotionJEPA v7 data-raw `video_exec.h5` / `video_demo.h5` 的 `frames` 逐帧比（截尾处以内）。
  a6      清单一致：新清单 40 条 vs 旧 400 ep 清单同身份条目五字段相同；framesamp meta / motion meta / motion index
          三处 manifest sha 相同；index 原始字节 sha 命中 store_meta.motion_index_sha256。
  a7      字节数账：每段 wan-latents `.bin == num_grid × 589,824`；motion 表 == rows × 3,072；motion-tokens 每段 == num_grid × 3,072。
  a9set   索引映射（集合半）：随机 500 个 (g, t)，`visible_motion_rows` 解出的起点集合 == 独立实现（直接遍历 wan-latents 目录
          的 metadata + 清单现算）解出的集合；row 与 (段, m) 反查一致。
  a10     行数账：表行数 == 772 = exec 658 + demo 114；逐段 num_grid == len(range(0, max(0, L−32), 16))；row_base 连续、totals 覆盖整表。

判定行：A5_FRAMES=PASS … / A6_MANIFEST=PASS / A7_BYTES=PASS / A9_INDEXSET=PASS / A10_ROWS=PASS。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mme_vla_suite.datastore import StoreMeta, load_manifest  # noqa: E402
from mme_vla_suite.datastore import motion_store as ms  # noqa: E402

CHUNK_BYTES = 9 * 16 * 32 * 32 * 4


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def cmd_a5(args):
    import h5py
    manifest = load_manifest(args.manifest)
    new_root = pathlib.Path(args.raw_dir)
    old_root = pathlib.Path(args.raw_dir_400ep)
    mj_raw = pathlib.Path(args.mj_data_raw)
    frames_cmp = frames_bad = 0
    mj_cmp = mj_bad = mj_skipped_eps = 0
    for ep in manifest["episodes"]:
        h5n, j, T, es = ep["h5_file"], int(ep["raw_ep_idx"]), int(ep["num_timesteps"]), int(ep["exec_start_idx"])
        task = ms.task_of_h5(h5n)
        with h5py.File(new_root / h5n, "r") as fn, h5py.File(old_root / h5n, "r") as fo:
            gn, go = fn[f"episode_{j}"], fo[f"episode_{j}"]
            To = sum(1 for k in go.keys() if k.startswith("timestep_"))
            if To != T:
                frames_bad += 1
                print(f"  ✗ {task} ep{j} 旧库 timesteps={To} != 新 {T}")
                continue
            new_frames = []
            for t in range(T):
                a = gn[f"timestep_{t}/obs/front_rgb"][()]
                b = go[f"timestep_{t}/obs/front_rgb"][()]
                frames_cmp += 1
                if not np.array_equal(a, b):
                    frames_bad += 1
                new_frames.append(a)
        # MotionJEPA v7 data-raw：exec 段 video_exec.h5、demo 段 video_demo.h5（仅 Video* 有）
        d = mj_raw / f"{task}_ep{j}"
        if not d.is_dir():
            mj_skipped_eps += 1
            continue
        for seg, start, L in (("exec", es, T - es), ("demo", 0, es)):
            p = d / f"video_{seg}.h5"
            if not p.is_file():
                if L > 0 and seg == "demo":
                    mj_skipped_eps += 1
                continue
            with h5py.File(p, "r") as f:
                fr = f["frames"]
                n = min(int(fr.shape[0]), L)               # v7 exec 段相对全长截尾，只比截尾处以内
                for i in range(n):
                    mj_cmp += 1
                    if not np.array_equal(fr[i], new_frames[start + i]):
                        mj_bad += 1
    ok = frames_bad == 0 and mj_bad == 0
    print(f"A5_FRAMES={'PASS' if ok else 'FAIL'} compared_400ep={frames_cmp} mismatches_400ep={frames_bad} "
          f"compared_mj_v7={mj_cmp} mismatches_mj_v7={mj_bad} mj_skipped={mj_skipped_eps}")
    if not ok:
        raise SystemExit(1)


def cmd_a6(args):
    new = load_manifest(args.manifest)
    old = load_manifest(args.manifest_400ep)
    ident = {(e["h5_file"], e["raw_ep_idx"]): e for e in old["episodes"]}
    bad = 0
    for e in new["episodes"]:
        o = ident.get((e["h5_file"], e["raw_ep_idx"]))
        if o is None:
            bad += 1
            continue
        for f in ("h5_file", "raw_ep_idx", "num_timesteps", "exec_start_idx", "exec_samples"):
            if e[f] != o[f]:
                bad += 1
                print(f"  ✗ g={e['global_episode_idx']} {f}: {e[f]} vs {o[f]}")
    fmeta = StoreMeta.load(args.framesamp)
    mmeta = ms.MotionMeta.load(args.motion)
    index_raw = json.loads((pathlib.Path(args.motion) / ms.INDEX_RELPATH).read_text(encoding="utf-8"))
    same_sha = fmeta.manifest_sha256 == mmeta.manifest_sha256 == index_raw["manifest_sha256"] == new["sha256"]
    ms.check_index_against_manifest(list(mmeta.entries), new)
    ok = bad == 0 and same_sha
    print(f"A6_MANIFEST={'PASS' if ok else 'FAIL'} episodes=40 field_mismatches={bad} manifest_sha_same={int(same_sha)} "
          f"motion_index_sha256={mmeta.motion_index_sha256[:16]}…")
    if not ok:
        raise SystemExit(1)


def cmd_a7(args):
    manifest = load_manifest(args.manifest)
    entries = ms.build_index_entries(manifest)
    lat = pathlib.Path(args.latents)
    tok = pathlib.Path(args.tokens)
    bad, n_seg = 0, 0
    for e in entries:
        for seg in ms.SEGMENTS:
            s = getattr(e, seg)
            if s.num_grid == 0:
                continue
            key = ms.segment_key(e.h5_file, e.raw_ep_idx, seg)
            n_seg += 1
            if (lat / f"{key}.bin").stat().st_size != s.num_grid * CHUNK_BYTES:
                bad += 1
                print(f"  ✗ {key}.bin 字节数 != {s.num_grid} × {CHUNK_BYTES}")
            if (tok / f"{key}.f32.bin").stat().st_size != s.num_grid * ms.MOTION_ROW_BYTES:
                bad += 1
                print(f"  ✗ {key}.f32.bin 字节数 != {s.num_grid} × {ms.MOTION_ROW_BYTES}")
    meta = ms.MotionMeta.load(args.motion)
    table_ok = (pathlib.Path(args.motion) / ms.MOTION_TABLE_RELPATH).stat().st_size == meta.num_rows * ms.MOTION_ROW_BYTES
    ok = bad == 0 and table_ok
    print(f"A7_BYTES={'PASS' if ok else 'FAIL'} segments={n_seg} mismatches={bad} table_rows={meta.num_rows} table_ok={int(table_ok)}")
    if not ok:
        raise SystemExit(1)


def _independent_visible(manifest: dict, lat_meta: dict, g: int, t: int) -> tuple[list[int], list[tuple[str, int]]]:
    """独立实现：从 wan-latents/metadata.json 的段清单（起点全域帧号）现算可见集合，不用 motion_store 的公式。"""
    ep = manifest["episodes"][g]
    task = ep["h5_file"][len("record_dataset_"):-3]
    frames, rows = [], []
    for seg in ("demo", "exec"):
        key = f"{task}_ep{ep['raw_ep_idx']}_{seg}"
        sm = lat_meta["segments"].get(key)
        if sm is None:
            continue
        for r in sm["rows"]:
            f = int(r["start_global_frame"])
            if f + 32 <= t:
                frames.append(f)
                rows.append((key, int(r["m"])))
    order = sorted(range(len(frames)), key=lambda i: frames[i])
    return [frames[i] for i in order], [rows[i] for i in order]


def cmd_a9set(args):
    manifest = load_manifest(args.manifest)
    lat_meta = json.loads((pathlib.Path(args.latents) / "metadata.json").read_text(encoding="utf-8"))
    meta = ms.MotionMeta.load(args.motion)
    rng = random.Random(args.seed)
    bad = 0
    samples = []
    for _ in range(args.n):
        e = rng.choice(meta.entries)
        t = rng.randrange(e.exec_start_idx, e.num_timesteps)
        samples.append((e, t))
    for e, t in samples:
        rows, frames = ms.visible_motion_rows(e, t)
        f_ind, seg_ind = _independent_visible(manifest, lat_meta, e.g, t)
        if frames.tolist() != f_ind:
            bad += 1
            print(f"  ✗ g={e.g} t={t} 起点集合 {frames.tolist()} != 独立 {f_ind}")
            continue
        # row → (段, m) 反查一致
        for r, (key, m) in zip(rows.tolist(), seg_ind, strict=True):
            seg = key.rsplit("_", 1)[1]
            s = getattr(e, seg)
            if s.row_base is None or s.row_base + m != r:
                bad += 1
                print(f"  ✗ g={e.g} t={t} row {r} != {key} row_base {s.row_base} + m {m}")
                break
    ok = bad == 0
    print(f"A9_INDEXSET={'PASS' if ok else 'FAIL'} samples={len(samples)} mismatches={bad}")
    if not ok:
        raise SystemExit(1)


def cmd_a10(args):
    meta = ms.MotionMeta.load(args.motion)
    totals = ms.index_totals(list(meta.entries))
    bad = 0
    cursor = 0
    for e in meta.entries:
        lens = ms.segment_lengths(e.num_timesteps, e.exec_start_idx)
        for seg in ms.SEGMENTS:
            s = getattr(e, seg)
            want = len(range(0, max(0, lens[seg] - 32), 16))
            if s.num_grid != want:
                bad += 1
            if s.num_grid and s.row_base != cursor:
                bad += 1
            cursor += s.num_grid
    ok = bad == 0 and meta.num_rows == totals["rows"] == cursor == args.expect_rows \
        and totals["exec_rows"] == args.expect_exec and totals["demo_rows"] == args.expect_demo
    print(f"A10_ROWS={'PASS' if ok else 'FAIL'} rows={meta.num_rows} exec={totals['exec_rows']} demo={totals['demo_rows']} "
          f"formula_or_rowbase_mismatches={bad} expect={args.expect_rows}={args.expect_exec}+{args.expect_demo}")
    if not ok:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("a5")
    p.add_argument("--manifest", required=True)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--raw-dir-400ep", default="/data/hongzefu/robomme_data_h5_v2_4env400ep")
    p.add_argument("--mj-data-raw", default="/data/hongzefu/motionjepa-v7/data-raw")
    p.set_defaults(func=cmd_a5)
    p = sub.add_parser("a6")
    p.add_argument("--manifest", required=True)
    p.add_argument("--manifest-400ep", required=True)
    p.add_argument("--framesamp", required=True)
    p.add_argument("--motion", required=True)
    p.set_defaults(func=cmd_a6)
    p = sub.add_parser("a7")
    p.add_argument("--manifest", required=True)
    p.add_argument("--latents", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--motion", required=True)
    p.set_defaults(func=cmd_a7)
    p = sub.add_parser("a9set")
    p.add_argument("--manifest", required=True)
    p.add_argument("--latents", required=True)
    p.add_argument("--motion", required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260903)
    p.set_defaults(func=cmd_a9set)
    p = sub.add_parser("a10")
    p.add_argument("--motion", required=True)
    p.add_argument("--expect-rows", type=int, default=772)
    p.add_argument("--expect-exec", type=int, default=658)
    p.add_argument("--expect-demo", type=int, default=114)
    p.set_defaults(func=cmd_a10)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
