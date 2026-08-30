#!/usr/bin/env python3
"""metrics.jsonl → scalars_hex.tsv 规范投影（v5.0）。

**为什么单列一份**：历届（G0b/G1/G2/G3）的 `scalars_hex.tsv` 是收官时手工投影的；
v5.0 的 C4 上游对拍要求 A/B 两侧各自投影后对同一 sha256 锚点，「口径不一致则 sha256
必然不同」，故投影实现必须收敛为唯一一份、两处（C3 收官与 C4 两侧）共用。

口径（与 G0b/G3 固化产物逐字节一致，由 `--selftest` 用 G3 留档实证）：
- 表头恰六列：`step\\tloss.hex\\tgrad_norm.hex\\tllm_grad_norm.hex\\tmem_enc_norm.hex\\tparam_norm.hex`；
- 每步一行、按 step 升序、五键缺一即 fail-loud（不静默跳过）；
- 末尾单换行。

用法：
    uv run scripts/training/tests/project_scalars.py <metrics.jsonl> <scalars_hex.tsv>
    uv run scripts/training/tests/project_scalars.py --selftest   # 用 G3 固化产物自证
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

_KEYS = ["loss", "grad_norm", "llm_grad_norm", "mem_enc_norm", "param_norm"]
_HEADER = "step\t" + "\t".join(f"{k}.hex" for k in _KEYS)


def project(metrics_path: pathlib.Path) -> str:
    rows: dict[int, dict] = {}
    with metrics_path.open() as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            r = json.loads(stripped)
            step = r.get("step")
            if step is None:
                continue
            rows[int(step)] = r    # 同 step 后写覆盖先写（正常逐步记录不会发生）
    if not rows:
        raise SystemExit(f"错误: {metrics_path} 无任何带 step 的记录行")
    out = [_HEADER]
    for step in sorted(rows):
        r = rows[step]
        cells = [str(step)]
        for k in _KEYS:
            v = r.get(k)
            if not isinstance(v, dict) or "hex" not in v:
                raise SystemExit(f"错误: step {step} 缺标量 {k}（fail-loud，不静默跳过）")
            cells.append(v["hex"])
        out.append("\t".join(cells))
    return "\n".join(out) + "\n"


def _selftest() -> int:
    repo = pathlib.Path(__file__).resolve().parents[3]
    rec = repo / "docs/training-doc/v1-postclean-g3/records"
    got = project(rec / "metrics.jsonl")
    expect = (rec / "scalars_hex.tsv").read_bytes()
    ok = got.encode() == expect
    print(f"PROJECT_SELFTEST={'PASS' if ok else 'FAIL'} "
          f"sha256={hashlib.sha256(got.encode()).hexdigest()}")
    return 0 if ok else 1


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        return _selftest()
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    text = project(src)
    dst.write_text(text)
    print(f"PROJECTED rows={text.count(chr(10)) - 1} "
          f"sha256={hashlib.sha256(text.encode()).hexdigest()} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
