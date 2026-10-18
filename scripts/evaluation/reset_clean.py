"""reset 干净对拍（1017-test-xhard5-12per T23、4.3 第 11 项）：同一服务进程内同一局评两次，逐步比对动作。

用法：``reset_clean.py <首评 steps 目录> <reset-check steps 目录>``。
对第二个目录里的每个 ``<Task>_ep<n>.jsonl``，在第一个目录找同名文件，逐行比对 ``action`` 向量（demo 前缀帧的
action 为 null，两边必须同为 null），也核 ``state``；行数不同或任一步不等即 FAIL。
期望逐位相等；不等时先看是 GPU 非确定性还是残留，FAIL 不放宽（计划 4.2「局间 reset」行）。

判定行：``RESET_CLEAN=PASS|FAIL pairs=<n> steps=<n> max_abs_diff=<x> first_diff_step=<k|none>``。
"""
import json
from pathlib import Path
import sys


def load(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compare(first_dir: Path, second_dir: Path) -> dict:
    pairs, steps, max_diff, first_diff = 0, 0, 0.0, None
    seconds = sorted(second_dir.glob("*.jsonl"))
    if not seconds:
        raise SystemExit(f"RESET_CLEAN=FAIL pairs=0 reason=no-reset-check-logs dir={second_dir}")
    for b_path in seconds:
        a_path = first_dir / b_path.name
        if not a_path.is_file():
            raise SystemExit(f"RESET_CLEAN=FAIL pairs={pairs} reason=missing-first {a_path}")
        a, b = load(a_path), load(b_path)
        pairs += 1
        if len(a) != len(b) and first_diff is None:
            first_diff = f"{b_path.stem}:len{len(a)}!={len(b)}"
        for ra, rb in zip(a, b):
            steps += 1
            diff = 0.0
            for key in ("action", "state"):
                va, vb = ra.get(key), rb.get(key)
                if (va is None) != (vb is None) or (va is not None and len(va) != len(vb)):
                    diff = float("inf")
                    break
                if va is not None:
                    diff = max([diff] + [abs(x - y) for x, y in zip(va, vb)])
            max_diff = max(max_diff, diff)
            if diff > 0 and first_diff is None:
                first_diff = f"{b_path.stem}:{ra.get('step')}"
    verdict = "PASS" if first_diff is None and max_diff == 0 else "FAIL"
    line = (f"RESET_CLEAN={verdict} pairs={pairs} steps={steps} max_abs_diff={max_diff:g} "
            f"first_diff_step={first_diff or 'none'}")
    print(line, flush=True)
    return {"verdict": verdict, "pairs": pairs, "steps": steps, "max_abs_diff": max_diff, "first_diff_step": first_diff}


if __name__ == "__main__":
    first, second = map(Path, sys.argv[1:3])
    result = compare(first, second)
    sys.exit(0 if result["verdict"] == "PASS" else 1)
