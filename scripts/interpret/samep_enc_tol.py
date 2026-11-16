#!/usr/bin/env python3
"""闸门 ENC_TOL（steer-speedup-1116 计划第一部分 3 节、第二部分 3 节）：新编码 vs 旧编码逐窗比较。

两组 npz 都按 `start_frames` / `embs` 读（新 = samep_offline_encode 的契约 C2 输出；旧 = 上一轮在线录制，
如 steer-4task-1115 的 `record/<task>_ep<id>.npz`、`vrec_record/sNNN/<task>_ep<id>.npz`，`--record-motion` 格式；
同一文件内重复起点只取第一次，与 sp_swap_prep.load_record 相同）。按起点帧号对齐，所有对齐窗口里按 `--seed`
不放回抽样至多 `--max-windows`（默认 300），每窗算余弦与最大绝对差：

    ENC_TOL=PASS|FAIL windows=<n> min_cos=<x> maxabs=<y> files=<配对文件数> aligned=<对齐总数> ...

判据：windows > 0、min_cos ≥ `--min-cos`（默认 0.999）且 maxabs ≤ `--max-abs`（默认 0.1）；维度不符的文件计 dim_mismatch 即 FAIL。

配对方式二选一：
    --new DIR --old DIR        两目录下同名 *.npz 配对（旧目录缺同名文件计 missing_old，不判 FAIL）
    --pairs PAIRS.json         [[new, old], ...] 或 [{"new":…, "old":…}, ...]（新旧文件名不同时用，如变体键 ↔ sNNN/epK）
只依赖 NumPy，主 venv 或 wan venv 皆可。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

MIN_COS, MAX_ABS, MAX_WINDOWS = 0.999, 0.1, 300


def load_embs(path) -> dict[int, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        starts = np.asarray(z["start_frames"], np.int64).reshape(-1)
        embs = np.asarray(z["embs"], np.float32)
    if embs.ndim != 2 or embs.shape[0] != starts.size:
        raise ValueError(f"{path} start_frames / embs 不对齐")
    out = {}
    for i, s in enumerate(starts.tolist()):
        out.setdefault(int(s), embs[i])
    return out


def pairs_from_dirs(new_dir, old_dir) -> tuple[list[tuple[pathlib.Path, pathlib.Path]], int]:
    new_dir, old_dir = pathlib.Path(new_dir), pathlib.Path(old_dir)
    pairs, missing = [], 0
    for p in sorted(new_dir.glob("*.npz")):
        if p.name.startswith("."):
            continue
        q = old_dir / p.name
        if q.is_file():
            pairs.append((p, q))
        else:
            missing += 1
    return pairs, missing


def pairs_from_json(path) -> tuple[list[tuple[pathlib.Path, pathlib.Path]], int]:
    items = json.loads(pathlib.Path(path).read_text())
    pairs, missing = [], 0
    for it in items:
        new, old = (it["new"], it["old"]) if isinstance(it, dict) else (it[0], it[1])
        new, old = pathlib.Path(new), pathlib.Path(old)
        if new.is_file() and old.is_file():
            pairs.append((new, old))
        else:
            missing += 1
    return pairs, missing


def window_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        cos = 1.0
    elif na == 0 or nb == 0:
        cos = 0.0
    else:
        cos = float(np.dot(a, b) / (na * nb))
    return cos, float(np.abs(a - b).max())


def evaluate(pairs, *, missing: int = 0, max_windows: int = MAX_WINDOWS, min_cos: float = MIN_COS,
             max_abs: float = MAX_ABS, seed: int = 0) -> dict:
    aligned, dim_bad, no_overlap = [], 0, 0
    for new, old in pairs:
        a, b = load_embs(new), load_embs(old)
        common = sorted(set(a) & set(b))
        if not common:
            no_overlap += 1
            continue
        if next(iter(a.values())).shape != next(iter(b.values())).shape:
            dim_bad += 1
            continue
        aligned += [(str(new), s, a[s], b[s]) for s in common]
    n_aligned = len(aligned)
    if n_aligned > max_windows:
        idx = np.sort(np.random.default_rng(seed).choice(n_aligned, size=max_windows, replace=False))
        sample = [aligned[i] for i in idx]
    else:
        sample = aligned
    cos_all, abs_all, worst = [], [], None
    for name, s, x, y in sample:
        c, m = window_metrics(x, y)
        cos_all.append(c)
        abs_all.append(m)
        if worst is None or c < worst[2]:
            worst = (name, s, c)
    n = len(sample)
    mc = min(cos_all) if cos_all else float("nan")
    ma = max(abs_all) if abs_all else float("nan")
    ok = n > 0 and dim_bad == 0 and mc >= min_cos and ma <= max_abs
    line = (f"ENC_TOL={'PASS' if ok else 'FAIL'} windows={n} min_cos={mc:.6f} maxabs={ma:.4g} "
            f"files={len(pairs)} aligned={n_aligned} missing_old={missing} no_overlap={no_overlap} "
            f"dim_mismatch={dim_bad} thr_cos={min_cos} thr_abs={max_abs}")
    return {"ok": ok, "line": line, "windows": n, "min_cos": mc, "maxabs": ma, "aligned": n_aligned,
            "worst": None if worst is None else {"file": worst[0], "start_frame": int(worst[1]), "cos": worst[2]},
            "mean_cos": float(np.mean(cos_all)) if cos_all else float("nan")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ENC_TOL：新旧 motion 编码按起点对齐逐窗比较")
    ap.add_argument("--new")
    ap.add_argument("--old")
    ap.add_argument("--pairs")
    ap.add_argument("--max-windows", type=int, default=MAX_WINDOWS)
    ap.add_argument("--min-cos", type=float, default=MIN_COS)
    ap.add_argument("--max-abs", type=float, default=MAX_ABS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="把判定行与明细写成 json（同名 .txt 写判定行）")
    args = ap.parse_args(argv)
    if bool(args.pairs) == bool(args.new and args.old):
        ap.error("须二选一：--new DIR --old DIR，或 --pairs PAIRS.json")
    pairs, missing = pairs_from_json(args.pairs) if args.pairs else pairs_from_dirs(args.new, args.old)
    r = evaluate(pairs, missing=missing, max_windows=args.max_windows, min_cos=args.min_cos,
                 max_abs=args.max_abs, seed=args.seed)
    print(r["line"], flush=True)
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, ensure_ascii=False, indent=1))
        out.with_suffix(".txt").write_text(r["line"] + "\n")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
