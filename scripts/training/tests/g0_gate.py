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
import subprocess
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]   # <repo>/scripts/training/tests/g0_gate.py → 仓库根

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


# ══════════════════════════════════════════════════════════════════════════════
# --profile t2：新库严格 A/B 闸（motion-memory-plan.md 5.2 / 2.8）——reference（S2_BASE，冻结 manifest）vs candidate
# 直接读两侧 records，不经 compare_baseline.py（fail-closed：缺文件 / 缺 step / 缺键 / 空交集 / 任一计数不符均 FAIL）
# ══════════════════════════════════════════════════════════════════════════════

_T2_ARGV_VALUE_SKIP = {"--exp-name", "--checkpoint-base-dir"}     # run / output 路径白名单：只允许这两项不同


def _t2_normalized_argv(argv: list[str]) -> list[str]:
    out = []
    skip = False
    for i, a in enumerate(argv):
        if i == 0:
            continue                                  # 脚本路径（两侧目录可不同）
        if skip:
            skip = False
            continue
        if a in _T2_ARGV_VALUE_SKIP:
            skip = True
            continue
        out.append(a)
    return out


def _t2_metrics(path: pathlib.Path) -> dict[int, dict[str, str]]:
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[int(r["step"])] = {k: v["hex"] for k, v in r.items() if isinstance(v, dict) and "hex" in v}
    return rows


def _t2_jsonl(path: pathlib.Path) -> dict[int, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[int(r["step"])] = r
    return rows


def _t2_log_ok(log: pathlib.Path, fails: list[str], tag: str) -> None:
    if not log.exists():
        fails.append(f"{tag} 日志不存在: {log}")
        return
    lines = [ln for ln in log.read_text(errors="replace").splitlines() if ln.startswith("EXIT_CODE=")]
    if len(lines) != 1 or lines[0] != "EXIT_CODE=0":
        fails.append(f"{tag} 日志 EXIT_CODE 行不唯一或非 0: {lines}")


def _t2_config_diff(ref_yaml_bytes: bytes, cand_yaml_path: pathlib.Path, fails: list[str]) -> None:
    """reference 源 YAML（git show S2_BASE:<path> 恢复的原始字节）vs candidate 当前 YAML：解析后深比较，差异白名单只有新增规范 motion 节且 enabled=false。"""
    from omegaconf import OmegaConf
    a = OmegaConf.to_container(OmegaConf.create(ref_yaml_bytes.decode("utf-8")), resolve=True)
    b = OmegaConf.to_container(OmegaConf.load(cand_yaml_path), resolve=True)
    if "motion" in a:
        fails.append("reference 源 YAML 已含 motion 节（应为 S2 改码前的 closed 文件）")
    m = b.pop("motion", None)
    if a != b:
        fails.append(f"candidate YAML 除 motion 节外与 reference 不同: {sorted(set(a) ^ set(b))}")
    need = {"enabled", "dim", "budget", "stride", "window_frames", "window_direction", "grid_origin", "store_path",
            "source_run", "pos_dim", "frame_size", "online_gpu"}
    if not isinstance(m, dict) or m.get("enabled") is not False or set(m) != need:
        fails.append(f"candidate motion 节不是规范的 enabled:false 全键节: {m}")


def _gate_t2(args) -> int:
    import subprocess
    fails: list[str] = []
    ref = json.loads(pathlib.Path(args.reference_manifest).read_text(encoding="utf-8"))
    A = pathlib.Path(ref["records_dir"]) if args.run_dir_a is None else args.run_dir_a
    B = args.run_dir_b
    steps, batch = int(args.steps), int(args.batch_size)
    if int(ref.get("steps", -1)) != steps or int(ref.get("batch_size", -1)) != batch:
        fails.append(f"reference manifest 的 steps/batch {ref.get('steps')}/{ref.get('batch_size')} != 本次声明 {steps}/{batch}")
    record_steps = set(int(x) for x in ref["record_steps"])          # TrainState 摘要步集
    digest_steps = set(int(x) for x in ref["digest_steps"])          # 输入摘要步集
    # 1) 两侧日志唯一 EXIT_CODE=0
    _t2_log_ok(pathlib.Path(ref["log_path"]) if args.log_a is None else args.log_a, fails, "reference")
    _t2_log_ok(args.log_b, fails, "candidate")
    # 2) 环境指纹相同（两侧 env.json 的 fingerprint 深比较）
    try:
        fa = json.loads((A / "env.json").read_text())["fingerprint"]
        fb = json.loads((B / "env.json").read_text())["fingerprint"]
        if fa != fb:
            diff = [k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)]
            fails.append(f"环境指纹不同: {diff}")
    except (OSError, KeyError, json.JSONDecodeError) as e:
        fails.append(f"env.json fingerprint 缺失: {e}")
    # 3) 规范化 argv 除 run / output 路径与 commit 外逐项相同（run_meta.json 的真实 argv）
    try:
        ra = json.loads((A / "run_meta.json").read_text())["argv"]
        rb = json.loads((B / "run_meta.json").read_text())["argv"]
        na, nb = _t2_normalized_argv(ra), _t2_normalized_argv(rb)
        if na != nb:
            fails.append(f"规范化 argv 不同: {[x for x in na if x not in nb]} vs {[x for x in nb if x not in na]}")
    except (OSError, KeyError, json.JSONDecodeError) as e:
        fails.append(f"run_meta.json 缺失: {e}")
    # 4) 配置：先核 reference 源 YAML sha，再与 candidate 解析比较（只允许新增 motion:false 节）
    try:
        raw = subprocess.run(["git", "-C", str(_REPO_ROOT), "show", f"{ref['S2_BASE']}:{ref['yaml_path']}"],
                             capture_output=True, check=True).stdout
        if hashlib.sha256(raw).hexdigest() != ref["yaml_sha256"]:
            fails.append("git show S2_BASE:<yaml> 的 sha256 与 manifest 记录不符")
        else:
            _t2_config_diff(raw, _REPO_ROOT / ref["yaml_path"], fails)
    except subprocess.CalledProcessError as e:
        fails.append(f"git show 失败: {e}")
    # 5) scalars：step 集与 scalar 键全集相同、逐步 hex 逐位；scalars_hex.tsv 表头 + steps 行、sha256 相等
    try:
        ma, mb = _t2_metrics(A / "metrics.jsonl"), _t2_metrics(B / "metrics.jsonl")
        if set(ma) != set(range(steps)) or set(mb) != set(range(steps)):
            fails.append(f"metrics step 集不是 0..{steps - 1}: A={len(ma)} B={len(mb)}")
        else:
            keys = set(ma[0])
            if any(set(ma[s]) != keys or set(mb[s]) != keys for s in range(steps)) or len(keys) != 5:
                fails.append("scalar 键全集不一致或不为 5 键")
            bad = [s for s in range(steps) if ma[s] != mb[s]]
            if bad:
                fails.append(f"scalars hex 失配步 {len(bad)}（首个 {bad[0]}）")
        sa, sb = (A / "scalars_hex.tsv").read_bytes(), (B / "scalars_hex.tsv").read_bytes()
        for tag, raw in (("A", sa), ("B", sb)):
            lines = raw.decode().splitlines()
            if len(lines) != steps + 1 or lines[0] != _EXPECT_HEADER:
                fails.append(f"{tag} scalars_hex.tsv 行数/表头不符（{len(lines)} 行）")
        if hashlib.sha256(sa).hexdigest() != hashlib.sha256(sb).hexdigest():
            fails.append("scalars_hex.tsv sha256 不等")
        if ref.get("scalars_sha256") and hashlib.sha256(sa).hexdigest() != ref["scalars_sha256"]:
            fails.append("reference scalars_hex.tsv sha256 与 manifest 记录不符（reference 腐烂）")
    except OSError as e:
        fails.append(f"scalars 缺失: {e}")
    # 6) TrainState 摘要：行数 == 声明步集，逐步 state_digest 逐位，n_leaves == 177
    try:
        pa, pb = _t2_jsonl(A / "param_checksums.jsonl"), _t2_jsonl(B / "param_checksums.jsonl")
        if set(pa) != record_steps or set(pb) != record_steps:
            fails.append(f"param_checksums 步集 != 声明 {sorted(record_steps)}: A={sorted(pa)} B={sorted(pb)}")
        else:
            for s in sorted(record_steps):
                if pa[s]["state_digest"] != pb[s]["state_digest"]:
                    fails.append(f"STATE_DIGEST step {s} 不等"); break
                if pa[s]["n_leaves"] != 177 or pb[s]["n_leaves"] != 177:
                    fails.append(f"n_leaves != 177 at step {s}: {pa[s]['n_leaves']}/{pb[s]['n_leaves']}"); break
    except OSError as e:
        fails.append(f"param_checksums 缺失: {e}")
    # 7) 输入摘要：步集 == 声明，逐步逐键 raw sha 逐位（同代码同 dtype，raw 应 0 失配），n_keys == 12
    try:
        ba, bb = _t2_jsonl(A / "batch_digests.jsonl"), _t2_jsonl(B / "batch_digests.jsonl")
        if set(ba) != digest_steps or set(bb) != digest_steps:
            fails.append(f"batch_digests 步集 != 声明 {sorted(digest_steps)}: A={sorted(ba)} B={sorted(bb)}")
        else:
            for s in sorted(digest_steps):
                if ba[s].get("n_keys") != 12 or bb[s].get("n_keys") != 12:
                    fails.append(f"n_keys != 12 at step {s}"); break
                if ba[s].get("per_key") != bb[s].get("per_key") or ba[s].get("sample_indices") != bb[s].get("sample_indices"):
                    fails.append(f"BATCH_DIGEST step {s} raw 不等"); break
    except OSError as e:
        fails.append(f"batch_digests 缺失: {e}")
    # 8) index 序列：两侧 n ≥ steps×batch 且前缀逐项相同
    try:
        ia = json.loads((A / "index_sequence.json").read_text()); ib = json.loads((B / "index_sequence.json").read_text())
        need = steps * batch
        la, lb = ia.get("indices") or ia.get("sequence") or [], ib.get("indices") or ib.get("sequence") or []
        if len(la) < need or len(lb) < need:
            fails.append(f"index 序列不足 {need}: A={len(la)} B={len(lb)}")
        elif la[:need] != lb[:need]:
            fails.append("index 序列前缀不同")
    except OSError as e:
        fails.append(f"index_sequence 缺失: {e}")
    if args.env_out is not None:
        if not args.env_out.exists() or "BASELINE_ENV=PASS" not in args.env_out.read_text():
            fails.append("env-out 中无 BASELINE_ENV=PASS")
    if fails:
        for r in fails:
            print(f"T2_GATE_FAIL reason={r}")
        print(f"T2_EQ=FAIL reasons={len(fails)}")
        return 1
    print(f"T2_EQ=PASS steps={steps} batch={batch} record_steps={sorted(record_steps)} digest_steps={sorted(digest_steps)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["t1", "t2"], default="t1", help="t1：既有 1000 步黄金闸（默认，旧调用不变）；t2：新库严格 A/B 闸")
    ap.add_argument("--compare-out", type=pathlib.Path)
    ap.add_argument("--run-dir", type=pathlib.Path)
    ap.add_argument("--scalars", type=pathlib.Path)
    ap.add_argument("--expect-sha256")
    ap.add_argument("--env-out", type=pathlib.Path)
    # t2
    ap.add_argument("--reference-manifest", type=pathlib.Path)
    ap.add_argument("--run-dir-a", type=pathlib.Path, default=None)
    ap.add_argument("--run-dir-b", type=pathlib.Path)
    ap.add_argument("--log-a", type=pathlib.Path, default=None)
    ap.add_argument("--log-b", type=pathlib.Path)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    if args.profile == "t2":
        for k in ("reference_manifest", "run_dir_b", "log_b"):
            if getattr(args, k) is None:
                ap.error(f"--profile t2 需要 --{k.replace('_', '-')}")
        return _gate_t2(args)
    for k in ("compare_out", "run_dir", "scalars", "expect_sha256", "env_out"):
        if getattr(args, k) is None:
            ap.error(f"--profile t1 需要 --{k.replace('_', '-')}")

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
