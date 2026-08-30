#!/usr/bin/env python3
"""G0 对拍 fail-closed 总闸（v5.0，v5.0-train-entry-restructure-plan.md 第九节）。

**为什么需要**：`compare_baseline.py` 有三处 fail-open——缺 scalar key 即 `continue`、
index 只比最短公共前缀、canonical/INDEX_SEQ 不进总 verdict。历史上靠人工判读补位，
而 AGENTS 16「中位数假象」的教训正是人眼判读吃的亏。本闸把判读收敛为唯一一行
`G0_EQ=PASS`，任何一项缺失或失配都 FAIL 并报出具体原因（fail-closed）。

**只做「解析 + 补检」，不重新解析 records 计算任何比对结果**——避免与
`compare_baseline.py` 成为两把尺子：

- A 类（解析 compare_baseline.py stdout）：四分项判定行逐行正则提取，任一行缺失即
  FAIL；raw `BATCH_DIGEST mismatch` 必须恰为 4 且 `bad_keys=2`、首键为
  `static_image_emb`（stdout 只报首个失配键；配合 `BATCH_DIGEST_CANONICAL mismatch=0`
  ——canonical 口径抹平 dtype 后零失配，说明 raw 失配全部是 dtype 统一的已知预期——
  与 `bad_keys=2` 间接锁定键集合为 {static_image_emb, static_pos_emb}）。
- B 类（独立补检，compare_baseline.py 根本不看的六项）：`scalars_hex.tsv` 表头恰六列
  + 行数恰 1001 + 末尾单换行 + sha256 命中锚点（表头六列 + 每步一行都在，就堵住了
  「缺 key 即 continue」那处 fail-open）；`batch_digests.jsonl` 首行 `n_keys=12`；
  本侧 `index_sequence.json` 的 `n >= 8072`；preflight 输出含 `BASELINE_ENV=PASS`。

用法：
    uv run scripts/training/tests/g0_gate.py \
      --compare-out <compare_baseline.py 的 stdout 落盘文件> \
      --run-dir <本侧 records 目录（含 batch_digests.jsonl / index_sequence.json）> \
      --scalars <scalars_hex.tsv> \
      --expect-sha256 <锚点 sha256> \
      --env-out <check_baseline_env.py check 的输出落盘文件>

唯一成功判定行：`G0_EQ=PASS`；失败时逐条打印 `G0_GATE_FAIL reason=...` 后输出
`G0_EQ=FAIL reasons=N` 并以非零退出。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

_EXPECT_HEADER = "step\tloss.hex\tgrad_norm.hex\tllm_grad_norm.hex\tmem_enc_norm.hex\tparam_norm.hex"
_EXPECT_LINES = 1001          # 表头 1 行 + 步 0..999 各 1 行
_EXPECT_RAW_MISMATCH = 4      # dtype 统一已知预期失配（须与 G2/G3 逐字吻合）
_EXPECT_RAW_BAD_KEYS = 2      # {static_image_emb, static_pos_emb}
_EXPECT_INDEX_N_MIN = 8072


def _check_compare_out(text: str, fails: list[str]) -> None:
    """A 类：四分项 + raw BATCH_DIGEST 逐行正则提取，任一行缺失即 FAIL。"""
    m = re.search(r"^SCALARS steps=(\d+) keys=(\d+) hex_mismatch_steps=(\d+)", text, re.MULTILINE)
    if not m:
        fails.append("compare-out 缺 SCALARS 判定行")
    elif (int(m[1]), int(m[2]), int(m[3])) != (1000, 5, 0):
        fails.append(f"SCALARS 失配: steps={m[1]} keys={m[2]} hex_mismatch_steps={m[3]}"
                     "（期望 1000/5/0）")

    m = re.search(r"^STATE_DIGEST rows=(\d+) mismatch=(\d+)", text, re.MULTILINE)
    if not m:
        fails.append("compare-out 缺 STATE_DIGEST 判定行")
    elif (int(m[1]), int(m[2])) != (12, 0):
        fails.append(f"STATE_DIGEST 失配: rows={m[1]} mismatch={m[2]}（期望 12/0）")

    m = re.search(r"^BATCH_DIGEST_CANONICAL rows=(\d+) mismatch=(\d+)", text, re.MULTILINE)
    if not m:
        fails.append("compare-out 缺 BATCH_DIGEST_CANONICAL 判定行")
    elif (int(m[1]), int(m[2])) != (14, 0):
        fails.append(f"BATCH_DIGEST_CANONICAL 失配: rows={m[1]} mismatch={m[2]}（期望 14/0）")

    m = re.search(r"^CANON_CHECK=(\w+) steps=(\d+)", text, re.MULTILINE)
    if not m:
        fails.append("compare-out 缺 CANON_CHECK 判定行")
    elif m[1] != "PASS" or int(m[2]) != 14:
        fails.append(f"CANON_CHECK 失配: {m[1]} steps={m[2]}（期望 PASS/14）")

    m = re.search(r"^INDEX_SEQ=(\w+) n=(\d+)", text, re.MULTILINE)
    if not m:
        fails.append("compare-out 缺 INDEX_SEQ 判定行")
    elif m[1] != "PASS" or int(m[2]) < _EXPECT_INDEX_N_MIN:
        fails.append(f"INDEX_SEQ 失配: {m[1]} n={m[2]}（期望 PASS 且 n>={_EXPECT_INDEX_N_MIN}）")

    # raw BATCH_DIGEST：^ 锚定避免误配 BATCH_DIGEST_CANONICAL
    m = re.search(r"^BATCH_DIGEST rows=(\d+) mismatch=(\d+)(.*)$", text, re.MULTILINE)
    if not m:
        fails.append("compare-out 缺 raw BATCH_DIGEST 判定行")
    else:
        rows, mismatch, detail = int(m[1]), int(m[2]), m[3]
        if rows != 14 or mismatch != _EXPECT_RAW_MISMATCH:
            fails.append(f"raw BATCH_DIGEST 失配: rows={rows} mismatch={mismatch}"
                         f"（期望 14/{_EXPECT_RAW_MISMATCH}，多一个少一个都 FAIL）")
        else:
            bk = re.search(r"bad_keys=(\d+)", detail)
            if not bk or int(bk[1]) != _EXPECT_RAW_BAD_KEYS:
                fails.append(f"raw BATCH_DIGEST bad_keys 失配: {detail!r}"
                             f"（期望 bad_keys={_EXPECT_RAW_BAD_KEYS}）")
            if "static_image_emb" not in detail:
                fails.append(f"raw BATCH_DIGEST 首个失配键不是 static_image_emb: {detail!r}")


def _check_scalars(path: pathlib.Path, expect_sha256: str, fails: list[str]) -> None:
    """B 类之一：表头恰六列 + 行数恰 1001 + 末尾单换行 + sha256 命中锚点。"""
    if not path.exists():
        fails.append(f"scalars 文件不存在: {path}")
        return
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != expect_sha256:
        fails.append(f"scalars sha256 失配: got={got} expect={expect_sha256}")
    text = raw.decode()
    if not text.endswith("\n") or text.endswith("\n\n"):
        fails.append("scalars 末尾不是单换行")
    lines = text.splitlines()
    if len(lines) != _EXPECT_LINES:
        fails.append(f"scalars 行数 {len(lines)} != {_EXPECT_LINES}")
    if not lines or lines[0] != _EXPECT_HEADER:
        fails.append(f"scalars 表头失配: {lines[0] if lines else '<空文件>'!r}")


def _check_run_dir(run_dir: pathlib.Path, fails: list[str]) -> None:
    """B 类之二：batch_digests.jsonl 首行 n_keys=12；index_sequence.json n>=8072。"""
    bd = run_dir / "batch_digests.jsonl"
    if not bd.exists():
        fails.append(f"缺 {bd}")
    else:
        try:
            first = json.loads(bd.open().readline())
            if first.get("n_keys") != 12:
                fails.append(f"batch_digests 首行 n_keys={first.get('n_keys')} != 12")
        except (json.JSONDecodeError, OSError) as e:
            fails.append(f"batch_digests 首行解析失败: {e}")
    iq = run_dir / "index_sequence.json"
    if not iq.exists():
        fails.append(f"缺 {iq}")
    else:
        try:
            n = json.loads(iq.read_text()).get("n")
            if not isinstance(n, int) or n < _EXPECT_INDEX_N_MIN:
                fails.append(f"index_sequence n={n} < {_EXPECT_INDEX_N_MIN}")
        except (json.JSONDecodeError, OSError) as e:
            fails.append(f"index_sequence 解析失败: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-out", required=True, type=pathlib.Path)
    ap.add_argument("--run-dir", required=True, type=pathlib.Path)
    ap.add_argument("--scalars", required=True, type=pathlib.Path)
    ap.add_argument("--expect-sha256", required=True)
    ap.add_argument("--env-out", required=True, type=pathlib.Path)
    args = ap.parse_args()

    fails: list[str] = []

    if not args.compare_out.exists():
        fails.append(f"compare-out 文件不存在: {args.compare_out}")
    else:
        _check_compare_out(args.compare_out.read_text(), fails)

    _check_scalars(args.scalars, args.expect_sha256, fails)
    _check_run_dir(args.run_dir, fails)

    if not args.env_out.exists():
        fails.append(f"env-out 文件不存在: {args.env_out}")
    elif "BASELINE_ENV=PASS" not in args.env_out.read_text():
        fails.append("env-out 中无 BASELINE_ENV=PASS")

    if fails:
        for r in fails:
            print(f"G0_GATE_FAIL reason={r}")
        print(f"G0_EQ=FAIL reasons={len(fails)}")
        return 1
    print("G0_EQ=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
