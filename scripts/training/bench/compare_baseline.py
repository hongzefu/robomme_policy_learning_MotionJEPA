#!/usr/bin/env python3
"""基线对拍工具：两份 records 目录的逐步标量 hex / state_digest / batch_digest 对比。

对应 v1-gradient-baseline.md T2 与六节（量化判据权威版本）。与基线同 commit 固化，
防一年后工具漂移、口径变了。

输出（全部为机器可判定行）：
- `SCALARS …`：五标量逐步 hex 列 diff（bitwise 主判据）；
- `REL key=… median=… p95=… max=…`：逐标量 rel 分布（量化判据的原料；FAIL 档的
  两轮 rel 即六节的 null 噪声底）;
- `STATE_DIGEST … / BATCH_DIGEST …`：TrainState 摘要与输入摘要逐行 diff；
- `BATCH_DIGEST_CANONICAL … / CANON_CHECK=PASS|FAIL`（P1b）：canonical 数值口径
  （浮点升 f32 后按位哈希，抹平 dtype 差异）的输入摘要 diff——跨 dtype 对拍
  （G1 vs G0）的输入侧判据；两侧都有 schema 2 记录才输出，否则打 WARN 跳过；
- `INDEX_SEQ=PASS|FAIL|SKIP`（P1b）：两侧 index_sequence.json 前缀对比（取共同
  步数×batch 个）——「同一批样本、同一顺序」的独立证据；
- `DET_CHECK=PASS|FAIL tier=… steps=… scalar_hex_diff=… state_digest_diff=…`
  （P2 判定行：两轮逐步标量 hex + 全部 state_digest diff 为空）；
- `STATE_NUMERIC …`（P1b，六节数值裁决）：state_digest 失配时的逐叶数值统计
  （max-abs / max-rel / L2 相对差 / cosine，params/opt_state/EMA 全覆盖）——需要
  两侧的 TrainState 数组落盘（bench 的 STATE_DUMP_STEPS 产物，经
  `--state-arrays-a/--state-arrays-b` 传入）；拿不出数组时输出
  `STATE_NUMERIC=INCONCLUSIVE`，且量化判据不得输出 PASS（只能 INCONCLUSIVE）；
- 给 `--null-pair A B` 时另输出六节量化判据：
  `QUANT_EQUIV=PASS|INCONCLUSIVE|FAIL scalars=5 null=<pair> margin=2.0`
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


def compare_digests(A: dict[int, dict], B: dict[int, dict], field: str, label: str,
                    per_field: str | None = None) -> list[int]:
    steps = sorted(set(A) & set(B))
    bad = [s for s in steps if A[s][field] != B[s][field]]
    detail = ""
    if bad:
        s = bad[0]
        pk_field = per_field or ("per_leaf" if field == "state_digest" else "per_key")
        pa, pb = A[s].get(pk_field, {}), B[s].get(pk_field, {})
        diff_keys = [k for k in pa if pa[k] != pb.get(k)]
        detail = f" first_bad_step={s} bad_keys={len(diff_keys)} 首个: {diff_keys[:1]}"
    print(f"{label} rows={len(steps)} mismatch={len(bad)}{detail}")
    return bad


def compare_index_seq(A_dir: pathlib.Path, B_dir: pathlib.Path, n_steps: int) -> None:
    """index 序列前缀对比（P1b）：取共同步数的样本 index，逐个比。SKIP=某侧无产物。"""
    pa, pb = A_dir / "index_sequence.json", B_dir / "index_sequence.json"
    if not (pa.exists() and pb.exists()):
        print(f"INDEX_SEQ=SKIP （{'A' if not pa.exists() else 'B'} 侧无 index_sequence.json"
              f"——P1b 前的旧记录没有该产物）")
        return
    sa, sb = json.load(open(pa)), json.load(open(pb))
    # batch size 从两侧摘要行拿不到时退化为整序列前缀 min 长度对比
    n = min(sa["n"], sb["n"])
    ia, ib = sa["indices"][:n], sb["indices"][:n]
    if ia == ib:
        print(f"INDEX_SEQ=PASS n={n}（共同前缀逐个一致, steps≈{n_steps}）")
    else:
        first = next(i for i in range(n) if ia[i] != ib[i])
        print(f"INDEX_SEQ=FAIL n={n} first_diff_pos={first} a={ia[first]} b={ib[first]}")


def _load_state_dump(dump_dir: pathlib.Path, step: int):
    """读 bench 落盘的 TrainState 数组（state_step_<N>.json/.bin，裸字节容器）。

    返回 {leaf_key: np.ndarray}；bf16 等 ml_dtypes 类型按名字还原。产物缺失返回 None。
    """
    import numpy as np
    meta_p = dump_dir / f"state_step_{step}.json"
    bin_p = dump_dir / f"state_step_{step}.bin"
    if not (meta_p.exists() and bin_p.exists()):
        return None
    meta = json.load(open(meta_p))

    def resolve_dtype(name: str):
        try:
            return np.dtype(name)
        except TypeError:
            import ml_dtypes
            return np.dtype(getattr(ml_dtypes, name))

    out = {}
    with bin_p.open("rb") as f:
        for key, m in meta["leaves"].items():
            f.seek(m["offset"])
            buf = f.read(m["nbytes"])
            if hashlib.sha256(str(m["dtype"]).encode()
                              + str(tuple(m["shape"])).encode() + buf).hexdigest() != m["sha256"]:
                raise SystemExit(f"BAD TrainState 落盘产物腐烂: {bin_p} 叶 {key}")
            out[key] = np.frombuffer(buf, dtype=resolve_dtype(m["dtype"])).reshape(m["shape"])
    return out


def leaf_numeric_stats(bad_steps: list[int], A_ck: dict[int, dict], B_ck: dict[int, dict],
                       arrays_a: pathlib.Path | None,
                       arrays_b: pathlib.Path | None) -> bool:
    """六节数值裁决（P1b）：state_digest 失配步的逐叶 max-abs/max-rel/L2/cosine。

    返回是否拿到了数值统计（False = INCONCLUSIVE，量化判据不得判 PASS——参数明显
    不同但范数恰好接近时五标量统计会漏判）。只算 sha 失配的叶子（sha 一致的叶子
    数值必然逐位相同，不必算）。
    """
    import numpy as np
    if not bad_steps:
        return True
    if arrays_a is None or arrays_b is None:
        print("STATE_NUMERIC=INCONCLUSIVE reason=未提供 --state-arrays-a/-b"
              "（digest 失配但无数组可算逐叶统计）")
        return False
    got_any = False
    for s in bad_steps:
        ta, tb = _load_state_dump(arrays_a, s), _load_state_dump(arrays_b, s)
        if ta is None or tb is None:
            print(f"STATE_NUMERIC=INCONCLUSIVE step={s} reason=某侧无该步 TrainState 落盘")
            continue
        pa = A_ck[s].get("per_leaf", {})
        pb = B_ck[s].get("per_leaf", {})
        diff_keys = [k for k in pa if pa[k] != pb.get(k)]
        rows = []
        for k in diff_keys:
            if k not in ta or k not in tb:
                raise SystemExit(f"BAD 摘要叶 {k} 不在落盘数组中（产物口径漂移）")
            a = ta[k].astype(np.float64).ravel()
            b = tb[k].astype(np.float64).ravel()
            if a.shape != b.shape:
                rows.append((k, {"shape_mismatch": [list(ta[k].shape), list(tb[k].shape)]}))
                continue
            d = np.abs(a - b)
            denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-8)
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            cos = float(a @ b / max(na * nb, 1e-16))
            rows.append((k, {
                "max_abs": float(d.max()),
                "max_rel": float((d / denom).max()),
                "l2_rel": float(np.linalg.norm(a - b) / max(na, nb, 1e-8)),
                "cosine": cos,
            }))
        got_any = True
        rows.sort(key=lambda kv: kv[1].get("max_rel", float("inf")), reverse=True)
        worst = rows[0][1] if rows else {}
        print(f"STATE_NUMERIC step={s} leaves_diff={len(rows)} "
              f"max_abs={worst.get('max_abs', 0):.3e} max_rel={worst.get('max_rel', 0):.3e} "
              f"l2_rel={worst.get('l2_rel', 0):.3e} cosine_min="
              f"{min((r[1].get('cosine', 1.0) for r in rows), default=1.0):.6f}")
        for k, st in rows[:20]:
            print(f"LEAF step={s} key={k} " + " ".join(
                f"{n}={v:.3e}" if isinstance(v, float) else f"{n}={v}"
                for n, v in st.items()))
        if len(rows) > 20:
            print(f"LEAF … 其余 {len(rows) - 20} 叶略（按 max_rel 降序，只列最差 20）")
    return got_any


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--tier", default="adhoc", help="判定行 tier 标签（如 d0/d1/d2/d2cold/g0）")
    ap.add_argument("--null-pair", nargs=2, metavar=("NULL_A", "NULL_B"),
                    help="六节量化判据的 null 对（两份 records 目录）")
    ap.add_argument("--state-arrays-a", type=pathlib.Path, default=None,
                    help="A 侧 TrainState 数组落盘目录（bench 的 state_dump/，P1b 数值裁决）")
    ap.add_argument("--state-arrays-b", type=pathlib.Path, default=None,
                    help="B 侧 TrainState 数组落盘目录")
    args = ap.parse_args()
    A_dir, B_dir = pathlib.Path(args.dir_a), pathlib.Path(args.dir_b)

    _verify_manifest(A_dir)
    _verify_manifest(B_dir)

    A, B = _load_metrics(A_dir), _load_metrics(B_dir)
    scalar_diff, rel_stats, rel_series = compare_scalars(A, B)

    A_ck = _load_jsonl_by_step(A_dir / "param_checksums.jsonl")
    B_ck = _load_jsonl_by_step(B_dir / "param_checksums.jsonl")
    sd_bad = compare_digests(A_ck, B_ck, "state_digest", "STATE_DIGEST")
    A_dg = _load_jsonl_by_step(A_dir / "batch_digests.jsonl")
    B_dg = _load_jsonl_by_step(B_dir / "batch_digests.jsonl")
    bd = len(compare_digests(A_dg, B_dg, "batch_digest", "BATCH_DIGEST"))
    sd = len(sd_bad)

    # canonical 数值口径（P1b schema 2）：两侧都有 canonical 字段的步才可比
    A_cn = {s: r for s, r in A_dg.items() if "batch_digest_canonical" in r}
    B_cn = {s: r for s, r in B_dg.items() if "batch_digest_canonical" in r}
    common_cn = set(A_cn) & set(B_cn)
    if common_cn:
        cn = len(compare_digests(A_cn, B_cn, "batch_digest_canonical",
                                 "BATCH_DIGEST_CANONICAL", per_field="per_key_canonical"))
        print(f"CANON_CHECK={'PASS' if cn == 0 else 'FAIL'} steps={len(common_cn)}")
    else:
        print("WARN canonical 摘要不可比：某侧无 schema 2 记录（P1b 前的旧产物只有 raw）")
    compare_index_seq(A_dir, B_dir, n_steps=len(set(A) & set(B)))

    # 六节数值裁决：digest 失配时必须拿逐叶数值统计，拿不出即 INCONCLUSIVE
    numeric_ok = leaf_numeric_stats(sd_bad, A_ck, B_ck,
                                    args.state_arrays_a, args.state_arrays_b)

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
        # 六节数值裁决：state_digest 失配又拿不出逐叶数值统计时，五标量统计不足以
        # 判 PASS（范数恰好接近会漏判）——只能 INCONCLUSIVE；FAIL 照旧 FAIL
        quant_verdict = "PASS" if quant_ok else "FAIL"
        if quant_ok and sd > 0 and not numeric_ok:
            quant_verdict = "INCONCLUSIVE"
        print(f"QUANT_EQUIV={quant_verdict} "
              f"scalars={len(rel_stats)} null={null_name} margin={_MARGIN}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
