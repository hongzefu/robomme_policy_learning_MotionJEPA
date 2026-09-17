"""核验确定性对拍的完整覆盖；不允许公共前缀或缺失记录冒充完整通过。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify(args) -> str:
    require(args.steps > 0 and args.batch_size > 0, "步数与 batch_size 必须为正")
    expected = [int(s) for s in args.digest_steps.split(",")]
    require(expected == sorted(set(expected)), "摘要步必须递增且无重复")
    require(all(0 <= s < args.steps for s in expected), "摘要步越界")
    sides = {}
    for spec in args.sides:
        name, directory = spec.split("=", 1)
        require(bool(name) and name not in sides, f"重复或空侧名：{name}")
        sides[name] = Path(directory)
    require(len(sides) >= 2, "至少需要两个运行侧")
    require(len(set(p.resolve() for p in sides.values())) == len(sides), "运行侧目录重复")
    require(len(args.compare_logs) == len(sides) - 1, "比较日志数量必须等于侧数减一")
    indices = None
    n = args.steps * args.batch_size
    for name, directory in sides.items():
        metrics = rows(directory / "metrics.jsonl")
        require([r["step"] for r in metrics] == list(range(args.steps)), f"{name} 标量步不完整或重复")
        for filename in ("batch_digests.jsonl", "param_checksums.jsonl"):
            require([r["step"] for r in rows(directory / filename)] == expected,
                    f"{name}/{filename} 摘要步不完整或重复")
        sequence = json.loads((directory / "index_sequence.json").read_text())["indices"]
        require(len(sequence) >= n, f"{name} 索引不足 {n}")
        if indices is None:
            indices = sequence[:n]
        require(sequence[:n] == indices, f"{name} 前 {n} 个索引不一致")
    required = (
        f"SCALARS steps={args.steps} keys=5 hex_mismatch_steps=0",
        "INDEX_SEQ=PASS", f"BATCH_DIGEST rows={len(expected)} mismatch=0",
        f"STATE_DIGEST rows={len(expected)} mismatch=0",
        f"CANON_CHECK=PASS steps={len(expected)}", "DET_CHECK=PASS",
    )
    for filename in args.compare_logs:
        lines = Path(filename).read_text().splitlines()
        require(not any(re.search(r"(?:=FAIL|=SKIP)", line) for line in lines), f"{filename} 含失败或跳过")
        for token in required:
            require(any(line == token or line.startswith(token + " ") for line in lines),
                    f"{filename} 缺少判定：{token}")
    return (f"{args.summary_prefix}=PASS scalars_steps={args.steps} index_n={n} "
            f"batch_digest_rows={len(expected)} state_digest_rows={len(expected)} "
            f"sides={len(sides)} pairs={len(args.compare_logs)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sides", nargs="+", required=True)
    parser.add_argument("--compare-logs", nargs="+", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--digest-steps", required=True)
    parser.add_argument("--summary-prefix", default="COLLATE_SHM_GUARD")
    args = parser.parse_args()
    try:
        print(verify(args), flush=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"{args.summary_prefix}=FAIL reason={exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
