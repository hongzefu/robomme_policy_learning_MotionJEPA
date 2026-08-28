#!/usr/bin/env python3
"""样本/batch 内容等价对拍（v2 计划 C.2，第一块之二）+ G6b。

对拍层是 **transform 之后**：两侧各实例化 `transform_dataset` 之后的 Dataset
（与 `create_data_loader` 同一变换栈构造，不起 worker），legacy 侧 RoboMMEDataset、
packed 侧 FrameSampDataset，对定点 idx 列表逐样本对拍全键；再取 C.1 dump 的真实
index 序列前 N 个 batch 过 `_collate_fn` 对拍（覆盖到进 device_put 前的最后一层）。

- **定点集**（由清单精确构造，~8,200）：step_idx∈{0,1,2,29,30,31,32,33} 各 200 +
  每 episode 首样本 1,600 + 固定 seed 均匀随机 5,000（去重后为实际数）。
- **判据（单一模式）**：全部键 shape/dtype/原始位串（等价于 view(uintN)）逐位零容差
  ——两侧交付 dtype 相同，无需 astype 折算。判定行
  `COMPARE_BATCH=PASS samples=… batches=… mismatches=0`。
- **G6b**：在全量打包库上构造 FrameSampDataset，复验 G6a 同两条（len==395,289、
  Video*/VideoSwap 各一 episode 首样本 `_step_of == exec_start_idx`）。
- **失配诊断**：importlib 复用 `scripts/data-preprocess-GL/compare_datasets.py` 的
  `metrics()` 统计口径资产（不改其本体；目录名含连字符不可作包 import）。
- **位型容器**：首个对拍 batch 逐键落 `<out>/exemplar/`（每键一个 `.bin` 原始字节
  + 旁置 JSON 记 shape/逻辑 dtype/字节序/键名；禁 npy/npz——丢 bf16 类型），写盘后
  立即读回断言逐位相同（round-trip 守卫）；样本失配时同容器落双侧现场。

运行（>5 min，tmux + Monitor；先跑 dump_index_seq.py 产出 --idx-file）：
  UV_LINK_MODE=copy uv run python scripts/data-pack-framesamp/compare_batches.py \
    --out v1-store/bench/framesamp-cmp/<run_name> --idx-file <out>/idx_seq_w4.json
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # transforms 为 NumPy/PIL，CPU 即可
os.environ.setdefault("OPENPI_DATA_HOME", str(_REPO_ROOT / "v1-store" / "models"))

import numpy as np

sys.path.insert(0, str(_REPO_ROOT / "src"))

from mme_vla_suite.datastore import StoreMeta, build_exec_lookup, load_manifest  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent


def _load_compare_assets():
    """importlib 载入 compare_datasets.py 的统计口径资产（metrics 等，失配诊断用）。"""
    p = _REPO_ROOT / "scripts" / "data-preprocess-GL" / "compare_datasets.py"
    spec = importlib.util.spec_from_file_location("compare_datasets_assets", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compare_datasets_assets"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── 位型容器（C.2：禁 npy/npz，逐键 .bin + JSON 旁置，round-trip 守卫）──────────


def _flatten(d: dict, prefix: str = "") -> dict:
    """把嵌套 dict 展平成 path 键（transform 后 image/image_mask 为按相机名分键的
    dict——np.asarray(dict) 会得到 object 0-d 数组、tobytes 是内存指针，必须先展平）。"""
    out = {}
    for k, v in d.items():
        path = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, path))
        else:
            out[path] = v
    return out


def dump_bitwise(out_dir: pathlib.Path, name: str, sample: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {}
    for key, val in _flatten(sample).items():
        safe = key.replace("/", "__")
        if val is None:
            sidecar[key] = {"kind": "none"}
            continue
        if isinstance(val, (str, bytes)):
            data = val.encode() if isinstance(val, str) else val
            (out_dir / f"{name}.{safe}.bin").write_bytes(data)
            sidecar[key] = {"kind": "str" if isinstance(val, str) else "bytes",
                            "byte_count": len(data)}
            continue
        arr = np.asarray(val)
        blob = arr.tobytes()
        p = out_dir / f"{name}.{safe}.bin"
        p.write_bytes(blob)
        sidecar[key] = {"kind": "ndarray", "dtype": str(arr.dtype),
                        "shape": list(arr.shape), "byte_order": "little",
                        "order": "C", "byte_count": len(blob)}
        back = np.frombuffer(p.read_bytes(), dtype=np.uint8)
        if back.tobytes() != blob:
            raise RuntimeError(f"位型容器 round-trip 失败: {p}")
    with (out_dir / f"{name}.sidecar.json").open("w") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=1)


# ── 逐键比较（零容差 bitwise）──────────────────────────────────────────────────


def cmp_value(key: str, a, b) -> str | None:
    """返回 None=一致；否则失配原因（含首个失配元素 hex）。"""
    if a is None or b is None:
        return None if (a is None and b is None) else f"{key}: None 侧别 {a is None}/{b is None}"
    if isinstance(a, (str, bytes)) or isinstance(b, (str, bytes)):
        return None if a == b else f"{key}: 字符串不等 {a!r:.60} != {b!r:.60}"
    aa, bb = np.asarray(a), np.asarray(b)
    if aa.dtype == object or bb.dtype == object:
        # 结构应已被 _flatten 展开；落到这里说明遇到未知容器——fail loud 不做指针比较
        return f"{key}: object dtype（未展开的容器 {type(a).__name__}/{type(b).__name__}）"
    if str(aa.dtype) != str(bb.dtype):
        return f"{key}: dtype {aa.dtype} != {bb.dtype}"
    if aa.shape != bb.shape:
        return f"{key}: shape {aa.shape} != {bb.shape}"
    ra, rb = aa.tobytes(), bb.tobytes()
    if ra == rb:
        return None
    ba = np.frombuffer(ra, np.uint8)
    bbb = np.frombuffer(rb, np.uint8)
    first = int(np.nonzero(ba != bbb)[0][0])
    return (f"{key}: 位串失配 首字节偏移 {first}/{len(ra)} "
            f"{ba[first]:02x}!={bbb[first]:02x}")


def cmp_sample(idx, a: dict, b: dict, mismatches: list) -> None:
    fa, fb = _flatten(a), _flatten(b)
    ka, kb = set(fa.keys()), set(fb.keys())
    if ka != kb:
        mismatches.append((idx, "<keys>", f"键集不等: 仅A={sorted(ka - kb)} 仅B={sorted(kb - ka)}"))
        return
    for key in sorted(ka):
        reason = cmp_value(key, fa[key], fb[key])
        if reason is not None:
            mismatches.append((idx, key, reason))


# ── 主流程 ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy-root", default=str(_REPO_ROOT / "v1-store/datasets/4task-gl"))
    ap.add_argument("--packed-root",
                    default=str(_REPO_ROOT / "v1-store/datasets/4task-gl-framesamp"))
    ap.add_argument("--manifest", default=str(_REPO_ROOT / "v1-store/episode_manifest.json"))
    ap.add_argument("--out", required=True, help="记录目录（不得已存在）")
    ap.add_argument("--idx-file", required=True,
                    help="dump_index_seq.py 产出的 idx_seq_w*.json（真实序列）")
    ap.add_argument("--n-step-group", type=int, default=200)
    ap.add_argument("--n-random", type=int, default=5000)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--first-every", type=int, default=1,
                    help="episode 首样本组取每第 K 个 episode（默认 1=全部 1600；"
                         "仅冒烟用大值，正式 S5 判据必须 1）")
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    if out.exists():
        raise SystemExit(f"记录目录已存在, 禁止覆盖: {out}")
    out.mkdir(parents=True)

    t0 = time.perf_counter()
    assets = _load_compare_assets()   # metrics()——失配诊断口径
    manifest = load_manifest(args.manifest)
    meta = StoreMeta.load(args.packed_root)
    if meta.manifest_scope != "full":
        raise SystemExit("S5 判据禁止 subset 迷你库（A.1）")
    epis_of, step_of, _ = build_exec_lookup(manifest)
    n = len(epis_of)

    # ―― 与 create_data_loader 同一变换栈构造（不起 worker）――
    import mme_vla_suite.training.config as _tcfg
    from mme_vla_suite.models.config.utils import get_history_config
    from mme_vla_suite.training.dataset import RoboMMEDataset
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset
    from openpi.training.data_loader import _collate_fn, transform_dataset

    cfg = _tcfg.get_config("mme_vla_suite")
    cfg = dataclasses.replace(cfg, exp_name="framesamp-cmp",
                              assets_base_dir=str(_REPO_ROOT / "v1-store/train-assets"))
    model_cfg = dataclasses.replace(cfg.model, use_history=True,
                                    history_config="perceptual-framesamp-context.yaml")
    data_config = cfg.data.create(cfg.assets_dirs, model_cfg)
    hc = get_history_config("perceptual-framesamp-context.yaml")
    action_horizon = model_cfg.action_horizon

    legacy_ds = RoboMMEDataset(args.legacy_root, data_config, hc, action_horizon)
    packed_ds = FrameSampDataset(args.packed_root, data_config=data_config,
                                 history_config=hc, action_horizon=action_horizon,
                                 manifest_path=args.manifest)

    # ―― G6b：全量库上复验 G6a 同两条 ――
    if len(packed_ds) != 395289:
        raise SystemExit(f"G6B=FAIL len={len(packed_ds)} != 395289")
    for h5 in ("record_dataset_VideoUnmask.h5", "record_dataset_VideoUnmaskSwap.h5"):
        ep = next(e for e in manifest["episodes"] if e["h5_file"] == h5)
        got = int(packed_ds._step_of[ep["exec_sample_offset"]])
        if got != ep["exec_start_idx"]:
            raise SystemExit(f"G6B=FAIL {h5} 首样本 step {got} != {ep['exec_start_idx']}")
    print(f"G6B=PASS len=395289 video_first_steps=ok", flush=True)

    tl = transform_dataset(legacy_ds, data_config)
    tp = transform_dataset(packed_ds, data_config)

    # ―― 定点集（C.2）――
    rng = np.random.default_rng(args.seed)
    picked: list[int] = []
    for s in (0, 1, 2, 29, 30, 31, 32, 33):
        pool = np.nonzero(step_of == s)[0]
        take = min(args.n_step_group, len(pool))
        picked.extend(int(i) for i in rng.choice(pool, size=take, replace=False))
    picked.extend(int(e["exec_sample_offset"])
                  for e in manifest["episodes"][::args.first_every])
    picked.extend(int(i) for i in rng.choice(n, size=min(args.n_random, n), replace=False))
    picked = list(dict.fromkeys(picked))   # 去重保序
    n_first = len(manifest["episodes"][::args.first_every])
    print(f"[cmp] 定点集 {len(picked)} 个样本（step 组 8×≤{args.n_step_group} + "
          f"episode 首样本 {n_first} + 随机 {args.n_random}，去重后）", flush=True)

    mismatches: list[tuple] = []
    exemplar_dumped = False
    for k, idx in enumerate(picked):
        a = tl[idx]
        b = tp[idx]
        before = len(mismatches)
        cmp_sample(idx, a, b, mismatches)
        if len(mismatches) > before:
            dump_bitwise(out / "mismatch" / f"idx_{idx}", "legacy", a)
            dump_bitwise(out / "mismatch" / f"idx_{idx}", "packed", b)
            for m in mismatches[before:]:
                print(f"[cmp] 失配 idx={m[0]} key={m[1]}: {m[2]}", flush=True)
            if len(mismatches) >= 20:
                break
        elif not exemplar_dumped:
            dump_bitwise(out / "exemplar", "sample", a)   # 位型容器 + round-trip 守卫
            exemplar_dumped = True
        if (k + 1) % 500 == 0:
            print(f"[cmp] 样本进度 {k + 1}/{len(picked)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    n_samples = k + 1 if picked else 0

    # ―― batch 级：C.1 真实序列前 N 个 batch 过 _collate_fn ――
    n_batches = 0
    if not mismatches:
        seq = json.load(open(args.idx_file))
        bsz = int(seq["batch"])
        indices = seq["indices"]
        avail = len(indices) // bsz
        n_batches = min(args.batches, avail)
        if n_batches < args.batches:
            print(f"[cmp] ⚠ idx-file 只够 {avail} 个 batch（要求 {args.batches}），"
                  f"按 {n_batches} 跑", flush=True)
        for bi in range(n_batches):
            chunk = indices[bi * bsz:(bi + 1) * bsz]
            ba = _collate_fn([tl[i] for i in chunk])
            bb = _collate_fn([tp[i] for i in chunk])
            before = len(mismatches)
            cmp_sample(f"batch{bi}", ba, bb, mismatches)
            if len(mismatches) > before:
                for m in mismatches[before:]:
                    print(f"[cmp] batch 失配 {m[0]} key={m[1]}: {m[2]}", flush=True)
                break
            if (bi + 1) % 50 == 0:
                print(f"[cmp] batch 进度 {bi + 1}/{n_batches} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

    result = {
        "schema": 1, "samples": n_samples, "batches": n_batches,
        "mismatches": len(mismatches),
        "picked_total": len(picked), "seed": args.seed,
        "idx_file": args.idx_file,
        "legacy_root": args.legacy_root, "packed_root": args.packed_root,
        "manifest_sha256": manifest["sha256"],
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "mismatch_head": [list(m) for m in mismatches[:20]],
    }
    with (out / "compare_result.json").open("w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    if mismatches:
        # 失配诊断补充：对首个失配样本输出 metrics() 统计口径（compare_datasets 资产）
        idx0 = mismatches[0][0]
        if isinstance(idx0, int):
            a, b = _flatten(tl[idx0]), _flatten(tp[idx0])
            for key in sorted(set(a) & set(b)):
                if isinstance(a.get(key), np.ndarray) and isinstance(b.get(key), np.ndarray) \
                        and a[key].shape == b[key].shape and a[key].dtype == b[key].dtype:
                    m = assets.metrics(np.asarray(a[key], np.float64).ravel()
                                       if a[key].dtype.kind in "fV" else a[key].ravel(),
                                       np.asarray(b[key], np.float64).ravel()
                                       if b[key].dtype.kind in "fV" else b[key].ravel())
                    print(f"[diag] idx={idx0} {key}: {m}")
        print(f"COMPARE_BATCH=FAIL samples={n_samples} batches={n_batches} "
              f"mismatches={len(mismatches)}")
        raise SystemExit(1)
    print(f"COMPARE_BATCH=PASS samples={n_samples} batches={n_batches} mismatches=0")


if __name__ == "__main__":
    main()
