#!/usr/bin/env python3
"""dtype 统一修复第一块判定：修复前后两份 dump 的离线对拍。

**判据四条**（全部零容差，任一不过即 FAIL）：

1. 全键 shape 相同；
2. 数值 canonical 一致——canonical 摘要是「升 f32 后按位视图哈希」，相等 ⟺ 计划
   判据 2 的「`astype(f32)` 后 `view(uint32)` 逐位相同」；
3. dtype 变化逐键清单与预期完全一致（见 `_EXPECTED_DTYPE`）：memory 三键按短/满长
   分档给出确切的修复前→修复后 dtype，`static_state_emb` 恒 f64 不变，memory 之外
   全部键 dtype 与 **raw** 摘要都必须完全相同（它们与本修复无关，任何变化都是越界）；
4. batch 级 memory 三键 dtype 恒定，不随 batch 组成（含短样本 / 全短 / 全满长 / 随机）
   摆动——这正是「collate 整批提升」行为消失的直接证据。

**外加归一化前纯函数位型测试**：绕开 `_normalize_state` 直接验 `right_padding_token_emb`
本身。`static_state_emb` 交付键经 f64 的 norm stats 恒为 f64、修复前后不变，第三处
`np.zeros` 的修复在交付键上根本不可观测——这个纯函数测试是它唯一的有效证据。判定
并入总判定行。

用法（在修复后 clean HEAD 上跑；cwd 仓库根）：

    JAX_PLATFORMS=cpu UV_LINK_MODE=copy uv run scripts/training/tests/compare_dtype_fix.py \\
        <修复前 dump 目录> <修复后 dump 目录> [--report out.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _common as C  # noqa: E402
import numpy as np  # noqa: E402

# 预期 dtype 变化清单：键 → (短样本 (前, 后), 满长 (前, 后))
_EXPECTED_DTYPE = {
    "static_image_emb": (("float64", "bfloat16"), ("bfloat16", "bfloat16")),
    "static_pos_emb": (("float64", "float32"), ("float32", "float32")),
    # state 交付键经 _normalize_state（norm stats 为 f64）恒 f64，修复前后不变
    "static_state_emb": (("float64", "float64"), ("float64", "float64")),
    "static_mask": (("bool", "bool"), ("bool", "bool")),
}


def _load_jsonl(p: pathlib.Path) -> list[dict]:
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _first_elem_diff(a_dir: pathlib.Path, b_dir: pathlib.Path, key: str) -> str:
    """失配定位：给出首个 f32 位视图不同的元素位置与两侧 hex。"""
    try:
        a = C.load_array(a_dir, key).astype(np.float32)
        b = C.load_array(b_dir, key).astype(np.float32)
    except Exception as e:
        return f"（无数组本体可定位: {e}）"
    if a.shape != b.shape:
        return f"（形状不同: {a.shape} vs {b.shape}）"
    ai = a.reshape(-1).view(np.uint32)
    bi = b.reshape(-1).view(np.uint32)
    bad = np.flatnonzero(ai != bi)
    if bad.size == 0:
        return "（数组本体逐位相同，摘要失配说明产物腐烂）"
    i = int(bad[0])
    pos = np.unravel_index(i, a.shape)
    return (f"首个失配元素 {tuple(int(x) for x in pos)}: "
            f"A=0x{int(ai[i]):08x} B=0x{int(bi[i]):08x} "
            f"(A={float(a.reshape(-1)[i])!r} B={float(b.reshape(-1)[i])!r}) "
            f"共 {bad.size} 个元素不同")


def compare_samples(a_root: pathlib.Path, b_root: pathlib.Path, fails: list[str]) -> int:
    A = {r["index"]: r for r in _load_jsonl(a_root / "samples" / "summary.jsonl")}
    B = {r["index"]: r for r in _load_jsonl(b_root / "samples" / "summary.jsonl")}
    if set(A) != set(B):
        fails.append(f"样本集合不同: 仅 A {sorted(set(A) - set(B))[:5]} 仅 B {sorted(set(B) - set(A))[:5]}")
        return 0
    for idx in sorted(A):
        ra, rb = A[idx], B[idx]
        if (ra["epis_idx"], ra["step_idx"]) != (rb["epis_idx"], rb["step_idx"]):
            fails.append(f"样本 {idx} 身份不同: A=({ra['epis_idx']},{ra['step_idx']}) B=({rb['epis_idx']},{rb['step_idx']})")
            continue
        short = ra["is_short"]
        ka, kb = ra["keys"], rb["keys"]
        if set(ka) != set(kb):
            fails.append(f"样本 {idx} 键集合不同: 仅 A {sorted(set(ka) - set(kb))} 仅 B {sorted(set(kb) - set(ka))}")
            continue
        for key in sorted(ka):
            da, db = ka[key], kb[key]
            if da["kind"] != db["kind"]:
                fails.append(f"样本 {idx} 键 {key} 叶子类别不同: {da['kind']} vs {db['kind']}")
                continue
            if da["kind"] == "none":
                continue
            if da["kind"] == "str":
                if da["value"] != db["value"]:
                    fails.append(f"样本 {idx} 键 {key} 字符串不同")
                continue
            # 判据 1：shape
            if da["shape"] != db["shape"]:
                fails.append(f"样本 {idx} 键 {key} shape 不同: {da['shape']} vs {db['shape']}")
                continue
            # 判据 2：数值（canonical 零容差）
            if da["canon"] != db["canon"]:
                loc = _first_elem_diff(a_root / "samples" / "arrays" / str(idx),
                                       b_root / "samples" / "arrays" / str(idx), key)
                fails.append(f"样本 {idx} 键 {key} 数值失配（canonical 不同）；{loc}")
                continue
            # 判据 3：dtype 清单
            bname = C.base_name(key)
            if bname in _EXPECTED_DTYPE:
                want_a, want_b = _EXPECTED_DTYPE[bname][0 if short else 1]
                if (da["dtype"], db["dtype"]) != (want_a, want_b):
                    fails.append(
                        f"样本 {idx}（{'短' if short else '满长'}）键 {key} dtype 与预期不符: "
                        f"实测 {da['dtype']}→{db['dtype']}，预期 {want_a}→{want_b}")
            elif da["dtype"] != db["dtype"]:
                fails.append(f"样本 {idx} 非 memory 键 {key} dtype 变了: {da['dtype']}→{db['dtype']}（越界）")
            elif da["raw"] != db["raw"]:
                fails.append(f"样本 {idx} 非 memory 键 {key} raw 摘要不同（应逐字节不变）")
    return len(A)


def compare_batches(a_root: pathlib.Path, b_root: pathlib.Path, fails: list[str]) -> int:
    A = {r["batch_id"]: r for r in _load_jsonl(a_root / "batches" / "summary.jsonl")}
    B = {r["batch_id"]: r for r in _load_jsonl(b_root / "batches" / "summary.jsonl")}
    if set(A) != set(B):
        fails.append("batch 集合不同")
        return 0
    post_dtype: dict[str, set[str]] = {}
    for bid in sorted(A):
        ra, rb = A[bid], B[bid]
        if ra["indices"] != rb["indices"]:
            fails.append(f"batch {bid} 样本组成不同")
            continue
        ka, kb = ra["keys"], rb["keys"]
        if set(ka) != set(kb):
            fails.append(f"batch {bid} 键集合不同: 仅 A {sorted(set(ka) - set(kb))} 仅 B {sorted(set(kb) - set(ka))}")
            continue
        for key in sorted(ka):
            da, db = ka[key], kb[key]
            if da["kind"] != db["kind"]:
                fails.append(f"batch {bid} 键 {key} 叶子类别不同")
                continue
            if da["kind"] == "none":
                continue
            if da["kind"] == "str":
                if da["value"] != db["value"]:
                    fails.append(f"batch {bid} 键 {key} 字符串不同")
                continue
            if da["shape"] != db["shape"]:
                fails.append(f"batch {bid} 键 {key} shape 不同: {da['shape']} vs {db['shape']}")
                continue
            if da["canon"] != db["canon"]:
                fails.append(f"batch {bid}（{ra['kind']}）键 {key} 数值失配（canonical 不同）")
                continue
            if C.base_name(key) not in _EXPECTED_DTYPE and da["dtype"] != db["dtype"]:
                fails.append(f"batch {bid} 非 memory 键 {key} dtype 变了: {da['dtype']}→{db['dtype']}（越界）")
            post_dtype.setdefault(key, set()).add(db["dtype"])
    # 判据 4：修复后 memory 键 dtype 不随 batch 组成摆动
    for key, seen in ((k, v) for k, v in post_dtype.items() if C.base_name(k) in _EXPECTED_DTYPE):
        if len(seen) > 1:
            fails.append(f"判据 4 不过: 修复后键 {key} 的 batch dtype 仍随组成摆动: {sorted(seen)}")
    return len(A)


def pure_function_check(fails: list[str]) -> None:
    """归一化前纯函数位型测试：直接验 `right_padding_token_emb` 本身。

    对 `static_state_emb` 这是唯一有效验证——交付键恒 f64 掩盖了第三处修复。
    对 modulation / expert 变体与在线评估路径（同一函数、同一修复、不在本计划验收
    范围）充当函数级证据。
    """
    import ml_dtypes

    from mme_vla_suite.shared.data_utils import right_padding_token_emb

    max_size = 32
    rng = np.random.default_rng(20260827)
    for t in (1, 2, 3, 30, 31, 32, 33):
        img = rng.standard_normal((t, 1, 16, 8)).astype(ml_dtypes.bfloat16)
        pos = rng.standard_normal((t, 1, 16, 4)).astype(np.float32)
        st = rng.standard_normal((t, 8)).astype(np.float32)
        mask = np.ones((t,), dtype=np.bool_)
        oi, op, os_, om = right_padding_token_emb(img, pos, st, mask, max_size)
        tag = f"t={t}"
        for name, out, src in (("img", oi, img), ("pos", op, pos), ("state", os_, st)):
            if str(out.dtype) != str(src.dtype):
                fails.append(f"纯函数测试 {tag}: {name} 输出 dtype {out.dtype} != 输入 {src.dtype}")
            keep = min(t, max_size)
            if out[:keep].tobytes() != np.ascontiguousarray(src[:keep]).tobytes():
                fails.append(f"纯函数测试 {tag}: {name} 非填充区被改动")
            if t < max_size:
                pad = np.asarray(out[keep:]).astype(np.float64)
                if pad.size and np.any(pad != 0):
                    fails.append(f"纯函数测试 {tag}: {name} 填充区非零")
        if str(om.dtype) != "bool":
            fails.append(f"纯函数测试 {tag}: mask dtype {om.dtype} != bool")
        if om.shape[0] != max_size or bool(om[:min(t, max_size)].all()) is False:
            fails.append(f"纯函数测试 {tag}: mask 有效区不全为 True")
        if t < max_size and bool(om[t:].any()):
            fails.append(f"纯函数测试 {tag}: mask 填充区非 False")



def _leaf_numeric(a: np.ndarray, b: np.ndarray) -> dict:
    """逐叶数值统计（f64 精度），口径与 compare_baseline.py 的数值裁决一致。"""
    x = a.astype(np.float64).reshape(-1)
    y = b.astype(np.float64).reshape(-1)
    d = np.abs(x - y)
    denom = np.maximum(np.maximum(np.abs(x), np.abs(y)), 1e-8)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return {
        "max_abs": float(d.max()) if d.size else 0.0,
        "max_rel": float((d / denom).max()) if d.size else 0.0,
        "l2_rel": float(np.linalg.norm(x - y) / max(nx, ny, 1e-30)),
        "cosine": float(np.dot(x, y) / (nx * ny)) if nx > 0 and ny > 0 else 1.0,
    }


def compare_grads(a_dir: pathlib.Path, b_dir: pathlib.Path, fails: list[str]) -> tuple[int, dict]:
    """单步定点梯度对拍。

    `allfull`（整批满长）是阴性对照：两侧本就同为 bf16 交付，任何差异都与 dtype 无关、
    只能是改动越界，因此它的失配单独标记为越界告警。
    """
    A = json.loads((a_dir / "grad_summary.json").read_text(encoding="utf-8"))
    B = json.loads((b_dir / "grad_summary.json").read_text(encoding="utf-8"))
    detail: dict = {}
    kinds = sorted(set(A["results"]) & set(B["results"]))
    if set(A["results"]) != set(B["results"]):
        fails.append("两侧梯度 batch 种类不同")
    for kind in kinds:
        ra, rb = A["results"][kind], B["results"][kind]
        if ra["indices"] != rb["indices"]:
            fails.append(f"梯度 batch {kind} 样本组成不同")
            continue
        # 输入侧先自证：两侧 batch 的 canonical 必须一致（数值同、只是 dtype 不同）
        for key, da in ra["batch_keys"].items():
            db = rb["batch_keys"].get(key)
            if db is None or da.get("kind") != db.get("kind"):
                fails.append(f"梯度 batch {kind} 输入键 {key} 两侧结构不同")
            elif da.get("kind") == "array" and da["canon"] != db["canon"]:
                fails.append(f"梯度 batch {kind} 输入键 {key} 数值失配（canonical 不同）")
        bad = sorted(k for k in set(ra["per_leaf"]) & set(rb["per_leaf"])
                     if ra["per_leaf"][k] != rb["per_leaf"][k])
        only = sorted(set(ra["per_leaf"]) ^ set(rb["per_leaf"]))[:5]
        if only:
            fails.append(f"梯度 batch {kind} 叶子集合不同: {only}")
        entry = {"n_leaves": len(ra["per_leaf"]), "n_mismatch": len(bad),
                 "loss_a": ra["loss_hex"], "loss_b": rb["loss_hex"],
                 "loss_bitwise": ra["loss_hex"] == rb["loss_hex"]}
        if bad:
            if kind == "allfull":
                fails.append(
                    f"⚠ 阴性对照 allfull 有 {len(bad)} 个梯度叶子失配——两侧交付本就同为 "
                    f"bf16，这与 dtype 无关，指向改动越界，必须立刻停下排查（首个: {bad[0]}）")
            else:
                fails.append(f"梯度 batch {kind} 有 {len(bad)} 个叶子 bitwise 失配（首个: {bad[0]}）")
            aa, ab = A.get("arrays_dir"), B.get("arrays_dir")
            if aa and ab:
                nums = {}
                for k in bad[:20]:
                    try:
                        x = C.load_array(pathlib.Path(aa) / kind, k)
                        y = C.load_array(pathlib.Path(ab) / kind, k)
                    except Exception as e:
                        nums[k] = {"error": str(e)}
                        continue
                    nums[k] = _leaf_numeric(x, y)
                entry["numeric"] = nums
            else:
                entry["numeric"] = "INCONCLUSIVE（无数组本体，拿不出逐叶数值裁决）"
        if ra["loss_hex"] != rb["loss_hex"]:
            fails.append(f"梯度 batch {kind} 单步 loss 不 bitwise: {ra['loss_hex']} vs {rb['loss_hex']}")
        detail[kind] = entry
    return len(kinds), detail


def main() -> None:
    ap = argparse.ArgumentParser(description="dtype 修复第一块 / 单步梯度对拍")
    ap.add_argument("dir_a", type=pathlib.Path, nargs="?", help="修复前 dump 目录")
    ap.add_argument("dir_b", type=pathlib.Path, nargs="?", help="修复后 dump 目录")
    ap.add_argument("--grad-a", type=pathlib.Path, default=None, help="修复前梯度 records 目录")
    ap.add_argument("--grad-b", type=pathlib.Path, default=None, help="修复后梯度 records 目录")
    ap.add_argument("--no-pure-check", action="store_true", help="跳过纯函数位型测试（只在修复前侧跑时用）")
    ap.add_argument("--report", type=pathlib.Path, default=None)
    args = ap.parse_args()

    fails: list[str] = []
    grad_fails: list[str] = []
    n_samples = n_batches = n_kinds = 0
    grad_detail: dict = {}

    if bool(args.dir_a) != bool(args.dir_b):
        raise SystemExit("dump 目录必须成对给出")
    if bool(args.grad_a) != bool(args.grad_b):
        raise SystemExit("梯度 records 目录必须成对给出")
    if not args.dir_a and not args.grad_a:
        raise SystemExit("至少要给一对 dump 目录或一对梯度 records 目录")

    if args.dir_a:
        pa = json.loads((args.dir_a / "fixture_plan.json").read_text(encoding="utf-8"))
        pb = json.loads((args.dir_b / "fixture_plan.json").read_text(encoding="utf-8"))
        plan_bad = [f for f in ("seed", "limit", "groups", "batches", "manifest_sha256")
                    if pa.get(f) != pb.get(f)]
        if plan_bad:
            for f in plan_bad:
                print(f"  两侧 fixture_plan 的 {f} 不同——定点集不可比，先查 dump 口径")
            print(f"COMPARE_DTYPE=FAIL samples=0 batches=0 mismatches={len(plan_bad)}")
            raise SystemExit(1)
        n_samples = compare_samples(args.dir_a, args.dir_b, fails)
        n_batches = compare_batches(args.dir_a, args.dir_b, fails)
        if not args.no_pure_check:
            pure_function_check(fails)

    if args.grad_a:
        n_kinds, grad_detail = compare_grads(args.grad_a, args.grad_b, grad_fails)

    lines = []
    if args.dir_a:
        v = "PASS" if not fails else "FAIL"
        lines.append(f"COMPARE_DTYPE={v} samples={n_samples} batches={n_batches} mismatches={len(fails)}")
    if args.grad_a:
        v = "PASS" if not grad_fails else "FAIL"
        lines.append(f"COMPARE_GRAD={v} kinds={n_kinds} mismatches={len(grad_fails)}")

    for m in (fails + grad_fails)[:30]:
        print("  " + m)
    if len(fails + grad_fails) > 30:
        print(f"  …另有 {len(fails + grad_fails) - 30} 条失配未列出")
    for line in lines:
        print(line)

    if args.report:
        args.report.write_text(json.dumps({
            "lines": lines,
            "n_samples": n_samples, "n_batches": n_batches, "n_grad_kinds": n_kinds,
            "mismatches": fails, "grad_mismatches": grad_fails,
            "grad_detail": grad_detail,
            "dir_a": str(args.dir_a) if args.dir_a else None,
            "dir_b": str(args.dir_b) if args.dir_b else None,
            "grad_a": str(args.grad_a) if args.grad_a else None,
            "grad_b": str(args.grad_b) if args.grad_b else None,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    raise SystemExit(0 if not (fails or grad_fails) else 1)


if __name__ == "__main__":
    main()
