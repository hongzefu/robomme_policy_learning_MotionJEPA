"""替换实验的配对统计：每个条件与 E0 按局配对，只比 motion 模型自己。

输入是各条件 first_bin_windows 写出的 windows.json，以及 attn/*/episode.json 里的 motion_content。
读数：官方成功率、第一个 bin 选对率，分任务报告差值、局级重采样区间（2000 次，种子 42）、配对符号检验。
替换检查：E0 替换数必须为 0；E1、E2、E3 每局必须替换过。回退指 replace_log 里 start != donor_start，只对 E1、E2 有意义。

用法：python analyze_swap_exp.py --exp-root <exp 目录> --conds e0 e1 e3 e2 --out <json> [--reference <一阶段 windows.json>]
"""
from __future__ import annotations

import argparse
from math import comb
import json
from pathlib import Path

import numpy as np

TASKS = ("ButtonUnmaskSwap", "VideoUnmaskSwap")
REPLACING = {"e1", "e2", "e3"}
FALLBACK_CONDS = {"e1", "e2"}


def load_windows(path):
    return {entry["episode_key"]: entry for entry in json.loads(Path(path).read_text(encoding="utf-8"))["episodes"]}


def load_content(root):
    """按 episode_nonce 读 motion_content。"""
    content = {}
    for path in Path(root).glob("g[01]/attn/*/episode.json"):
        meta = json.loads(path.read_text(encoding="utf-8"))
        content[path.parent.name] = meta.get("motion_content") or {}
    return content


def sign_test(losses, gains):
    n = losses + gains
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(losses, gains) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(reference, condition, seed=42, draws=2000):
    reference, condition = np.asarray(reference, float), np.asarray(condition, float)
    diff = condition - reference
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, diff.size, size=(draws, diff.size))
    means = diff[picks].mean(axis=1)
    losses, gains = int((diff < 0).sum()), int((diff > 0).sum())
    return {"episodes": int(diff.size), "reference": float(reference.mean()), "condition": float(condition.mean()),
            "diff": float(diff.mean()), "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            "losses": losses, "gains": gains, "sign_p": sign_test(losses, gains)}


def readouts(base, other, keys):
    out = {}
    for name, value in (("success", lambda e: float(e["outcome"] == "success")),
                        ("first_bin_correct", lambda e: float(e["label"] == "correct"))):
        out[name] = paired([value(base[k]) for k in keys], [value(other[k]) for k in keys])
    return out


def fallback_keys(windows, content):
    keys = set()
    for key, entry in windows.items():
        log = content.get(entry["episode_nonce"], {}).get("replace_log") or []
        if any(start != donor for start, donor in log):
            keys.add(key)
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-root", type=Path, required=True)
    parser.add_argument("--conds", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="一阶段原采样路径的 windows.json，只作路径差异参照")
    args = parser.parse_args()
    if args.conds[0] != "e0":
        raise SystemExit("第一个条件必须是 e0")
    windows = {c: load_windows(args.exp_root / c / "windows" / "windows.json") for c in args.conds}
    contents = {c: load_content(args.exp_root / c) for c in args.conds}
    base = windows["e0"]
    result = {"bootstrap": {"draws": 2000, "seed": 42, "unit": "局"}, "conds": {}}
    if args.reference:
        first = load_windows(args.reference)
        keys = sorted(set(first) & set(base))
        result["path_reference"] = {t: readouts(first, base, [k for k in keys if k.startswith(t + ":")]) for t in TASKS}
    for cond in args.conds:
        win, content = windows[cond], contents[cond]
        counts = [content.get(e["episode_nonce"], {}).get("n_replaced", 0) or 0 for e in win.values()]
        errors = sum(e["outcome"] not in ("success", "fail", "timeout") for e in win.values())
        replaced = int(sum(counts))
        ok = len(win) == 100 and errors == 0
        if cond == "e0":
            ok = ok and replaced == 0
        elif cond in REPLACING:
            ok = ok and all(n > 0 for n in counts)
        block = {"episodes": len(win), "infra_errors": errors, "replaced": replaced,
                 "labels": {t: {name: sum(e["label"] == name for k, e in win.items() if k.startswith(t + ":"))
                                for name in ("correct", "wrong", "no_pick")} for t in TASKS},
                 "success": {t: sum(e["outcome"] == "success" for k, e in win.items() if k.startswith(t + ":")) for t in TASKS}}
        if cond != "e0":
            keys = sorted(set(win) & set(base))
            block["vs_e0"] = {t: readouts(base, win, [k for k in keys if k.startswith(t + ":")]) for t in TASKS}
            if cond in FALLBACK_CONDS:
                fallback = fallback_keys(win, content)
                block["fallback_episodes"] = {t: sum(k.startswith(t + ":") for k in fallback) for t in TASKS}
                block["vs_e0_no_fallback"] = {t: readouts(base, win, [k for k in keys if k.startswith(t + ":") and k not in fallback])
                                              for t in TASKS}
        result["conds"][cond] = block
        print(f"SWAP_EXP={'PASS' if ok else 'FAIL'} cond={cond} episodes={len(win)} infra_errors={errors} replaced={replaced}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for cond, block in result["conds"].items():
        for task in TASKS:
            row = block.get("vs_e0", {}).get(task)
            if row:
                s, c = row["success"], row["first_bin_correct"]
                print(f"{cond} {task} success {s['reference']:.2f}->{s['condition']:.2f} diff={s['diff']:+.2f} ci={s['ci95']} -{s['losses']}/+{s['gains']} p={s['sign_p']:.3f}; "
                      f"first_bin {c['reference']:.2f}->{c['condition']:.2f} diff={c['diff']:+.2f} ci={c['ci95']} -{c['losses']}/+{c['gains']} p={c['sign_p']:.3f}")


if __name__ == "__main__":
    main()
