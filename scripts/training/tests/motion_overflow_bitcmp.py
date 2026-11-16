"""motion 超预算降级的两道起跑前闸（0922-binfill-demo-prefix-plan.md 验证节）。

**① MOTION_DEFAULT_BITEXACT**——默认口径（``overflow="raise"``）下新旧代码逐位等价。
取基线 commit 的 ``framesamp_memory.py`` 原文动态加载成第二个模块，与当前工作区版本
在同一批输入上各跑一遍 ``_prepare_motion``，四个返回值逐叶 sha256 比对。
注入确定性伪 token，因为 ``_prepare_motion`` 的输出只取决于「挑哪些起点」与查表，
不依赖编码器数值——**所以这一条只验装配层，不验 SigLIP 数值**（那是 enc_chunk_cmp.py 的事）。
两边都不走 ``__init__``：用 ``object.__new__`` 直接设属性，从而不需要真模型、纯 CPU 秒级跑完。

**② MOTION_HEADROOM**——已有 550 条旧结果沿用是安全的，即降级分支在那批集上不可达。
判据 ``k_demo(es_max) + k_exec(2000) <= budget``。

``es_max`` **不能用 ``grep ... | tail -1`` 求**——那只是「已有日志里的最大值」，日志缺失就会
给出偏小的假上界。这里改为「覆盖核对 + 实测最大值」两步：先断言每个 run 的
``exec_start_idx`` 行数 + error 集数 == 计划集数（证明没有集被漏掉），
再取全部非 BinFill 集的实测最大值。缺口不等于 error 集即判 INCOMPLETE 停下来。

用法（server 环境，纯 CPU）：
    uv run scripts/training/tests/motion_overflow_bitcmp.py [--base-commit 2cf4747]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MODULE_REL = "src/mme_vla_suite/policies/framesamp_memory.py"

# 与 checkpoint 桶里 history_config.resolved.yaml 的 motion 段一致
MOTION = dict(stride=16, window_frames=33, budget=160, demo_min_real=17, dim=768, pos_dim=256)
MAX_STEPS = 2000
EXEC_HORIZON = 16
#: 覆盖三段：无 demo（旧 BinFill）、旧 11 组实测区间、补 demo 后的新区间（含两道硬闸边界）
ES_CASES = [0, 1, 16, 17, 33, 100, 254, 264, 318, 502, 508,
            560, 576, 592, 593, 608, 638, 642, 800, 1152, 1281, 1296]


def load_module(name: str, source: str):
    """把一份源码加载成独立模块（不经 sys.path，避免与工作区版本互相覆盖）。"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(source)
        path = fh.name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_memory(cls, es: int, overflow: str | None):
    """不走 __init__ 构造一个只够跑 _prepare_motion 的实例。

    伪 token 用 ``f`` 本身派生，保证「取到哪个起点」的任何差异都会改变输出；
    pos 表用 arange 派生，同理。两边用同一份数组对象，排除构造差异。
    """
    mem = object.__new__(cls)
    mem.exec_start_idx = es
    mem.motion_stride = MOTION["stride"]
    mem.motion_window = MOTION["window_frames"]
    mem.motion_budget = MOTION["budget"]
    mem.motion_dim = MOTION["dim"]
    mem.motion_pos_dim = MOTION["pos_dim"]
    mem.demo_min_real = MOTION["demo_min_real"]
    rows = es + MAX_STEPS + 64
    mem.pos_emb = (np.arange(rows * 1 * MOTION["dim"], dtype=np.float32)
                   .reshape(rows, 1, MOTION["dim"]) * np.float32(1e-4))
    # 起点有两套：demo 段是 0,16,32,…（从全域 0 起），exec 段是 es,es+16,…（从 es 起）。
    # es 不是 16 的倍数时这两套并不重合——只填前一套会让 _prepare_motion 抛「起点应已编码但缓冲中没有」，
    # 那种 raise 与「超预算 raise」都是 RuntimeError，会把真实差异掩盖掉。
    feats = {}
    for f in list(range(0, rows, MOTION["stride"])) + list(range(es, rows, MOTION["stride"])):
        feats[f] = (np.arange(MOTION["dim"], dtype=np.float32) + np.float32(f)) * np.float32(1e-3)
    mem._history_feats_motion = feats
    if overflow is not None:            # 新版才有的字段
        mem.motion_overflow = overflow
        mem.motion_downsample_steps = 0
        mem.motion_downsample_max_k = 0
        mem.motion_last_k = 0
    return mem


def digest(result) -> str:
    emb, pos, mask, times = result
    h = hashlib.sha256()
    for arr in (emb, pos, mask, times):
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def run_case(cls, es: int, overflow: str | None) -> list[str]:
    """跑满 126 次推理，逐次记一个摘要；raise 记成标签而不是让测试崩掉。"""
    out = []
    mem = make_memory(cls, es, overflow)
    for j in range(1, 127):
        step_idx = es + EXEC_HORIZON * (j - 1)
        try:
            out.append(digest(mem._prepare_motion(step_idx)))
        except RuntimeError as exc:
            # 带上消息关键词：「超预算」与「起点未编码」都是 RuntimeError，只记类型会把两者混为一谈
            kind = "budget" if "motion.budget" in str(exc) else (
                "missing" if "应已编码" in str(exc) else "other")
            out.append(f"RAISE:{kind}")
    return out


def check_bitexact(base_commit: str) -> None:
    base_src = subprocess.check_output(
        ["git", "-C", str(REPO), "show", f"{base_commit}:{MODULE_REL}"], text=True)
    head_src = (REPO / MODULE_REL).read_text(encoding="utf-8")
    old = load_module("_framesamp_memory_base", base_src)
    new = load_module("_framesamp_memory_head", head_src)

    total, mismatched = 0, []
    kinds: dict[str, int] = {}
    for es in ES_CASES:
        old_rows = run_case(old.FrameSampMemory, es, None)
        new_rows = run_case(new.FrameSampMemory, es, "raise")
        for j, (a, b) in enumerate(zip(old_rows, new_rows), start=1):
            total += 1
            kinds[a if a.startswith("RAISE") else "digest"] = \
                kinds.get(a if a.startswith("RAISE") else "digest", 0) + 1
            if a != b:
                mismatched.append((es, j, a[:16], b[:16]))
    if mismatched:
        for es, j, a, b in mismatched[:10]:
            print(f"  MISMATCH es={es} infer={j} base={a} head={b}", flush=True)
        raise SystemExit(f"MOTION_DEFAULT_BITEXACT=FAIL 不一致 {len(mismatched)}/{total}")
    # 伪 token 若填漏了起点，两边会一起抛「起点未编码」而依旧 PASS——这道断言堵住那种假绿
    assert kinds.get("RAISE:missing", 0) == 0, f"伪 token 填漏了起点：{kinds}"
    assert kinds.get("RAISE:other", 0) == 0, f"出现预期外的 RuntimeError：{kinds}"
    assert kinds.get("RAISE:budget", 0) > 0, f"没有一组走到超预算分支，这道闸没验到东西：{kinds}"
    print(f"MOTION_DEFAULT_BITEXACT=PASS base={base_commit} cases={len(ES_CASES)} "
          f"comparisons={total} breakdown={json.dumps(kinds, ensure_ascii=False)}", flush=True)

    # 顺带证明 resample 确实改变了行为（否则上面的 PASS 可能只是因为分支根本没进去）
    changed = 0
    for es in ES_CASES:
        rows_raise = run_case(new.FrameSampMemory, es, "raise")
        rows_resample = run_case(new.FrameSampMemory, es, "resample")
        changed += sum(1 for a, b in zip(rows_raise, rows_resample) if a != b)
        for a, b in zip(rows_raise, rows_resample):
            if a.startswith("RAISE"):
                assert not b.startswith("RAISE"), "resample 下仍抛异常，降级没生效"
            else:
                assert a == b, "未超预算时 resample 与 raise 必须逐位相同"
    print(f"MOTION_RESAMPLE_ACTIVE=PASS changed_infers={changed}"
          f"（这些正是 raise 口径下会整集判 error 的推理）", flush=True)


def k_demo_closed(es: int) -> int:
    if es < MOTION["demo_min_real"]:
        return 0
    return (es - MOTION["demo_min_real"]) // MOTION["stride"] + 1


def k_exec_closed(t_rel: int) -> int:
    if t_rel < MOTION["window_frames"] - 1:
        return 0
    return (t_rel - (MOTION["window_frames"] - 1)) // MOTION["stride"] + 1


def check_headroom(runs: list[str], expect_total: int) -> None:
    verdicts = []
    for run in runs:
        run_dir = REPO / "v1-store/evaluation" / run
        if not run_dir.is_dir():
            raise SystemExit(f"MOTION_HEADROOM=INCOMPLETE 找不到 run 目录 {run_dir}")
        values = []
        for log in sorted(run_dir.glob("s*/client.log")):
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("exec_start_idx: "):
                    values.append(int(line.split()[1]))
        errors_path = run_dir / "merged/errors.json"
        if not errors_path.is_file():
            raise SystemExit(f"MOTION_HEADROOM=INCOMPLETE 缺 {errors_path}")
        n_errors = len(json.loads(errors_path.read_text(encoding="utf-8")))
        # 覆盖核对：有 es 的集 + error 集必须正好等于计划集数，缺口只允许来自 error 集
        if len(values) + n_errors != expect_total:
            raise SystemExit(
                f"MOTION_HEADROOM=INCOMPLETE {run}: exec_start_idx 行数 {len(values)} + "
                f"error 集 {n_errors} != 计划 {expect_total}；日志有缺口，不能用它求上界")
        non_binfill = [v for v in values if v > 0]     # BinFill 那 150 条在旧 run 里恒为 0
        verdicts.append({"run": run, "es_rows": len(values), "errors": n_errors,
                         "zeros": len(values) - len(non_binfill),
                         "es_max": max(non_binfill), "es_min": min(non_binfill)})
    es_max = max(v["es_max"] for v in verdicts)
    k_max = k_demo_closed(es_max) + k_exec_closed(MAX_STEPS)
    budget = MOTION["budget"]
    print("MOTION_HEADROOM=" + ("PASS" if k_max <= budget else "FAIL") +
          f" es_max={es_max} k_demo={k_demo_closed(es_max)} k_exec={k_exec_closed(MAX_STEPS)} "
          f"k_max={k_max} budget={budget} " + json.dumps(verdicts, ensure_ascii=False), flush=True)
    if k_max > budget:
        raise SystemExit(f"旧 run 的 k_max={k_max} > budget={budget}，降级分支在旧结果上可达，不能沿用")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-commit", default="2cf4747",
                        help="对拍基线（默认 2cf4747 = 本轮改动前最后一个 commit）")
    parser.add_argument("--runs", nargs="*", default=[
        "primary700-nomotion50k-gl", "primary700-motion50k-gl", "primary700-modul32frame-50k-gl"])
    parser.add_argument("--expect-total", type=int, default=700)
    parser.add_argument("--skip-headroom", action="store_true")
    args = parser.parse_args()
    check_bitexact(args.base_commit)
    if not args.skip_headroom:
        check_headroom(args.runs, args.expect_total)


if __name__ == "__main__":
    main()
