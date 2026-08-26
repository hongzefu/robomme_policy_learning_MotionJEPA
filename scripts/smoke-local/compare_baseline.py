#!/usr/bin/env python3
"""基线对拍工具：两份 records 目录的逐步标量 hex / state_digest / batch_digest 对比。

对应 v1-gradient-baseline.md T2 与六节（量化判据权威版本）。与基线同 commit 固化，
防一年后工具漂移、口径变了。

输出（全部为机器可判定行）：
- `SCALARS …`：五标量逐步 hex 列 diff（bitwise 主判据）；
- `REL key=… median=… p95=… max=…`：逐标量 rel 分布（量化判据的原料；FAIL 档的
  两轮 rel 即六节的 null 噪声底）;
- `STATE_DIGEST … / BATCH_DIGEST …`：TrainState 摘要与输入摘要逐行 diff；
- `DET_CHECK=PASS|FAIL tier=… steps=… scalar_hex_diff=… state_digest_diff=…`
  （P2 判定行：两轮逐步标量 hex + 全部 state_digest diff 为空）；
- 给 `--null-pair A B` 时另输出六节量化判据：
  `QUANT_EQUIV=PASS|FAIL scalars=5 null=<pair> margin=2.0`
  （rel 各统计档 ≤ null 相应档 × 2，带绝对下限守卫；包络逐步判定同口径）。

产物完整性：某侧存在 BASELINE_MANIFEST.json 时先逐条 sha256 复验（固化产物防腐），
不符 fail-loud；在跑 run 的 records（无 manifest）跳过。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import sys

_SCALAR_KEYS = ("loss", "grad_norm", "llm_grad_norm", "mem_enc_norm", "param_norm")
_ABS_FLOOR = {"loss": 1e-6, "grad_norm": 1e-5, "llm_grad_norm": 1e-5,
              "mem_enc_norm": 1e-5, "param_norm": 1e-5}
_MARGIN = 2.0


def _sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_manifest(d: pathlib.Path) -> None:
    man_path = d / "BASELINE_MANIFEST.json"
    if not man_path.exists():
        return
    man = json.load(open(man_path))
    for rel, row in man["files"].items():
        p = d / rel
        if not p.exists() or _sha256_file(p) != row["sha256"]:
            raise SystemExit(f"BAD 产物腐烂: {d}/{rel} 与 BASELINE_MANIFEST.json 不符")
    print(f"OK manifest 复验通过: {man_path}（{len(man['files'])} 条）")


def _load_metrics(d: pathlib.Path) -> dict[int, dict]:
    rows = [json.loads(l) for l in open(d / "metrics.jsonl")]
    return {r["step"]: r for r in rows if r.get("loss") is not None}


def _load_jsonl_by_step(path: pathlib.Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    return {r["step"]: r for r in (json.loads(l) for l in open(path))}


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-8)


def compare_scalars(A: dict[int, dict], B: dict[int, dict]):
    steps = sorted(set(A) & set(B))
    if set(A) != set(B):
        print(f"WARN 步集合不同: A 独有 {sorted(set(A)-set(B))[:5]}…, "
              f"B 独有 {sorted(set(B)-set(A))[:5]}…")
    mismatch_steps, first_mismatch = 0, None
    rel_series: dict[str, list[tuple[int, float]]] = {k: [] for k in _SCALAR_KEYS}
    for s in steps:
        step_bad = False
        for k in _SCALAR_KEYS:
            va, vb = A[s].get(k), B[s].get(k)
            if va is None or vb is None:
                continue
            if va["hex"] != vb["hex"]:
                step_bad = True
            rel_series[k].append((s, _rel(va["dec"], vb["dec"])))
        if step_bad:
            mismatch_steps += 1
            if first_mismatch is None:
                first_mismatch = s
    print(f"SCALARS steps={len(steps)} keys={len(_SCALAR_KEYS)} "
          f"hex_mismatch_steps={mismatch_steps} first_mismatch_step={first_mismatch}")
    stats = {}
    for k, series in rel_series.items():
        vals = [v for _, v in series]
        if not vals:
            continue
        sv = sorted(vals)
        stats[k] = {"median": statistics.median(vals),
                    "p95": sv[min(len(sv) - 1, int(len(sv) * 0.95))],
                    "max": sv[-1]}
        print(f"REL key={k} median={stats[k]['median']:.3e} "
              f"p95={stats[k]['p95']:.3e} max={stats[k]['max']:.3e}")
    return mismatch_steps, stats, rel_series


def compare_digests(A: dict[int, dict], B: dict[int, dict], field: str, label: str) -> int:
    steps = sorted(set(A) & set(B))
    bad = [s for s in steps if A[s][field] != B[s][field]]
    detail = ""
    if bad:
        s = bad[0]
        pk_field = "per_leaf" if field == "state_digest" else "per_key"
        pa, pb = A[s].get(pk_field, {}), B[s].get(pk_field, {})
        diff_keys = [k for k in pa if pa[k] != pb.get(k)]
        detail = f" first_bad_step={s} bad_keys={len(diff_keys)} 首个: {diff_keys[:1]}"
    print(f"{label} rows={len(steps)} mismatch={len(bad)}{detail}")
    return len(bad)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--tier", default="adhoc", help="判定行 tier 标签（如 d0/d1/d2/d2cold/g0）")
    ap.add_argument("--null-pair", nargs=2, metavar=("NULL_A", "NULL_B"),
                    help="六节量化判据的 null 对（两份 records 目录）")
    args = ap.parse_args()
    A_dir, B_dir = pathlib.Path(args.dir_a), pathlib.Path(args.dir_b)

    _verify_manifest(A_dir)
    _verify_manifest(B_dir)

    A, B = _load_metrics(A_dir), _load_metrics(B_dir)
    scalar_diff, rel_stats, rel_series = compare_scalars(A, B)

    sd = compare_digests(_load_jsonl_by_step(A_dir / "param_checksums.jsonl"),
                         _load_jsonl_by_step(B_dir / "param_checksums.jsonl"),
                         "state_digest", "STATE_DIGEST")
    bd = compare_digests(_load_jsonl_by_step(A_dir / "batch_digests.jsonl"),
                         _load_jsonl_by_step(B_dir / "batch_digests.jsonl"),
                         "batch_digest", "BATCH_DIGEST")

    verdict = "PASS" if scalar_diff == 0 and sd == 0 and bd == 0 else "FAIL"
    print(f"DET_CHECK={verdict} tier={args.tier} steps={len(sorted(set(A) & set(B)))} "
          f"scalar_hex_diff={scalar_diff} state_digest_diff={sd} batch_digest_diff={bd}")

    if args.null_pair:
        NA, NB = (_load_metrics(pathlib.Path(p)) for p in args.null_pair)
        _, null_stats, null_series = compare_scalars(NA, NB)
        quant_ok = True
        for k, st in rel_stats.items():
            ns = null_stats.get(k, {"median": 0.0, "p95": 0.0, "max": 0.0})
            floor = _ABS_FLOOR[k]
            for tier_name in ("median", "p95", "max"):
                thresh = max(ns[tier_name] * _MARGIN, floor)
                if st[tier_name] > thresh:
                    quant_ok = False
                    print(f"QUANT_FAIL key={k} 档={tier_name} "
                          f"rel={st[tier_name]:.3e} > 阈值 {thresh:.3e}"
                          f"（null×{_MARGIN} 与下限 {floor:g} 取大）")
        # 包络判据：逐步 rel ≤ 全 null 序列上包络 × 2（带下限守卫）
        for k, series in rel_series.items():
            null_env = max((v for _, v in null_series.get(k, [])), default=0.0)
            floor = _ABS_FLOOR[k]
            thresh = max(null_env * _MARGIN, floor)
            bad = [(s, v) for s, v in series if v > thresh]
            if bad:
                quant_ok = False
                print(f"QUANT_FAIL key={k} 档=envelope 超包络步数={len(bad)} "
                      f"首个 step={bad[0][0]} rel={bad[0][1]:.3e} > {thresh:.3e}")
        null_name = f"{pathlib.Path(args.null_pair[0]).name}~{pathlib.Path(args.null_pair[1]).name}"
        print(f"QUANT_EQUIV={'PASS' if quant_ok else 'FAIL'} "
              f"scalars={len(rel_stats)} null={null_name} margin={_MARGIN}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
