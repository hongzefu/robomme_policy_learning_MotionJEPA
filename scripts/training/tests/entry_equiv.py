#!/usr/bin/env python3
"""C4 上游 main 对拍 harness（v5.0，v5.0-train-entry-restructure-plan.md 第十节）。

两个子命令：

**run**：在目标侧环境里启动训练入口并装三处只读补丁 + 一道来源断言。
    uv run scripts/training/tests/entry_equiv.py run \\
      --entry <入口py绝对路径> --record-dir <dir> --expect-root <目录> \\
      --expect-steps 1000 -- <入口 argv 尾巴...>

  1. wandb 代理：`sys.modules["wandb"]` 预载，显式 `log` 逐步记五标量 `float.hex()`
     落 `metrics_all.jsonl` 后转发**调用时查找**的真 `wandb.log`，其余属性 `__getattr__`
     委托真模块（规避「`wandb.init()` 重赋值模块级 `log` 盖掉补丁」的 2026-08-24 坑）。
  2. `save_state` 摘要器：`openpi.training.checkpoints.save_state` 模块属性替换
     （train.py 的 `_checkpoints.save_state(...)` 属调用时属性查找，补丁必中），
     **不落真权重**。双口径：strict = sha256(key+dtype+shape+bytes)（A/B 互比；裸字节
     口径下 dtype 不同、shape 转置、树中位置不同均会误通过）；g0-compat = 照抄
     `g0/bench_train_steps.py::_leaf_sha256` 的 dtype+shape+bytes 口径（供与 G0b 固化
     `param_checksums.jsonl` 交叉比，口径由 `test_padding_dtype.py` 锁死不得单方面改）。
     另记 `treedef_sha` / `keyset_sha`（堵「结构变了但每个叶子字节恰好对上」）。
  3. resolved-config digest：`tx`/`ema_decay` 标了 `pytree_node=False` 不是 leaf，
     逐叶遍历遍历不到，故收尾用与入口相同的 argv 重调 `cli()`（config 是 argv 的纯
     函数）取 resolved config，逐顶层字段 repr 落 `resolved_config.json`。
  4. 来源污染防护：起跑前断言 `PYTHONPATH` 未设置 + `find_spec` 的 origin 在
     `--expect-root` 下；`initialize_checkpoint_dir` 包装里（main 已开始、任何计算
     之前）再对全部已加载项目模块断言 `__file__` 归属；收尾把模块 `__file__`+sha256
     与 cwd git HEAD 写 `provenance.json`。三道防住「A 侧 import 到 B 侧代码 →
     两侧跑同一份代码必然 PASS」这一最阴的误通过。

  harness 只 import 标准库 + wandb + numpy/jax + openpi.training.checkpoints，
  不 import 本仓库其他训练代码（resolved-config 经 sys.modules 取入口自己加载的
  config 模块，harness 不直接 import 它）。

**judge**：读两侧 record 目录，输出 4.5 的判定行（纯标准库 + 同目录 project_scalars）。
    uv run scripts/training/tests/entry_equiv.py judge \\
      --a-dir <A 侧 record> --b-dir <B 侧 record> --expect-sha256 <锚点>
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import runpy
import subprocess
import sys
import time

_PROJECT_TOPLEVELS = ("openpi", "mme_vla_suite")
_SCALAR_KEYS = ["loss", "grad_norm", "llm_grad_norm", "mem_enc_norm", "param_norm"]
_CFG_WHITELIST = {"exp_name", "dataset_path", "checkpoint_base_dir", "overwrite"}
_CFG_SPOTLIGHT = ["ema_decay", "optimizer", "lr_schedule"]  # 单独列出、不埋进总比对


# ────────────────────────── run 侧 ──────────────────────────

class _WandbRecorderProxy:
    def __init__(self, real, path: pathlib.Path):
        self._real = real
        self._path = path
        self.rows = 0

    def log(self, data, step=None, **kwargs):
        row: dict = {"step": int(step) if step is not None else None,
                     "wall_time": time.time()}
        n_scalar = 0
        for k, v in data.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            row[k] = {"dec": fv, "hex": fv.hex()}
            n_scalar += 1
        if n_scalar:
            with self._path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            self.rows += 1
        # 属性在调用时才查找真 wandb.log（wandb.init() 重赋值也拦不住转发）
        return self._real.log(data, step=step, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _assert_provenance(expect_root: pathlib.Path) -> None:
    for name, mod in list(sys.modules.items()):
        if name.split(".")[0] not in _PROJECT_TOPLEVELS:
            continue
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = pathlib.Path(f).resolve()
        if not p.is_relative_to(expect_root):
            raise SystemExit(f"ENTRY_PROVENANCE 违例: 模块 {name} 来自 {p}，"
                             f"不在 --expect-root {expect_root} 下（来源污染，立即停）")


def _module_provenance(expect_root: pathlib.Path, entry: pathlib.Path) -> dict:
    mods = {}
    for name, mod in sorted(sys.modules.items()):
        if name.split(".")[0] not in _PROJECT_TOPLEVELS:
            continue
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = pathlib.Path(f).resolve()
        mods[name] = {"file": str(p),
                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.getcwd(),
                          capture_output=True, text=True, check=False).stdout.strip()
    return {"expect_root": str(expect_root),
            "entry": {"file": str(entry),
                      "sha256": hashlib.sha256(entry.read_bytes()).hexdigest()},
            "cwd": os.getcwd(), "git_head_of_cwd": head, "modules": mods}


def _split_segments(all_path: pathlib.Path, record_dir: pathlib.Path) -> tuple[int, int]:
    """按 step 回绕切分 tentative / 正式两段；>2 段即 fail-loud。"""
    segments: list[list[str]] = [[]]
    prev = None
    with all_path.open() as f:
        for line in f:
            step = json.loads(line).get("step")
            if prev is not None and step is not None and step < prev:
                segments.append([])
            if step is not None:
                prev = step
            segments[-1].append(line)
    if len(segments) > 2:
        raise SystemExit(f"metrics 段数 {len(segments)} > 2（step 回绕多于一次），无法切分")
    main_seg = segments[-1]
    tent_seg = segments[0] if len(segments) == 2 else []
    (record_dir / "metrics.jsonl").write_text("".join(main_seg))
    if tent_seg:
        (record_dir / "metrics_tentative.jsonl").write_text("".join(tent_seg))
    return len(tent_seg), len(main_seg)


def _dump_resolved_config(record_dir: pathlib.Path) -> None:
    cfgmod = sys.modules.get("mme_vla_suite.training.config")
    if cfgmod is None:
        (record_dir / "resolved_config.json").write_text(
            json.dumps({"error": "config 模块未加载（入口未跑到 import）"}))
        return
    cfg = cfgmod.cli()   # config 是 argv 的纯函数；sys.argv 仍为入口 argv
    fields = {f.name: repr(getattr(cfg, f.name)) for f in dataclasses.fields(cfg)}
    body = {k: v for k, v in fields.items() if k not in _CFG_WHITELIST}
    sha = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    (record_dir / "resolved_config.json").write_text(json.dumps(
        {"fields": fields, "whitelist": sorted(_CFG_WHITELIST),
         "resolved_cfg_sha_ex_whitelist": sha}, indent=2, ensure_ascii=False))


def _run(args, tail: list[str]) -> int:
    entry = pathlib.Path(args.entry).resolve()
    record_dir = pathlib.Path(args.record_dir).resolve()
    expect_root = pathlib.Path(args.expect_root).resolve()

    if "PYTHONPATH" in os.environ or "PYTHONHOME" in os.environ:
        raise SystemExit("来源污染防护: 必须以 env -u PYTHONPATH -u PYTHONHOME 启动")
    if not entry.is_relative_to(expect_root):
        raise SystemExit(f"入口 {entry} 不在 --expect-root {expect_root} 下")
    import importlib.util
    for top in _PROJECT_TOPLEVELS:
        spec = importlib.util.find_spec(top)
        if spec is None or not spec.origin:
            raise SystemExit(f"find_spec({top!r}) 失败，环境不完整")
        if not pathlib.Path(spec.origin).resolve().is_relative_to(expect_root):
            raise SystemExit(f"来源污染: {top} 解析到 {spec.origin}，"
                             f"不在 {expect_root} 下")

    record_dir.mkdir(parents=True, exist_ok=True)
    all_path = record_dir / "metrics_all.jsonl"
    if all_path.exists():
        raise FileExistsError(f"记录已存在，拒绝覆盖: {all_path}")

    # 补丁 1：wandb 代理（必须在入口 import wandb 之前进 sys.modules）
    import wandb as _real_wandb
    proxy = _WandbRecorderProxy(_real_wandb, all_path)
    sys.modules["wandb"] = proxy   # type: ignore[assignment]

    # 补丁 2 + 断言点：save_state 摘要器与 initialize_checkpoint_dir 包装
    import jax  # noqa: I001 —— 延迟到断言通过后才 import 项目环境
    import numpy as np
    import openpi.training.checkpoints as _ckpt
    state_path = record_dir / "state_digests.jsonl"
    _orig_init_dir = _ckpt.initialize_checkpoint_dir

    def _wrapped_init_dir(*a, **kw):
        _assert_provenance(expect_root)   # main 已开始、任何计算之前
        return _orig_init_dir(*a, **kw)

    def _summarize_state(checkpoint_manager, state, data_loader, step):
        trees = {"params": state.params, "opt_state": state.opt_state,
                 "step": state.step}
        if state.ema_params is not None:
            trees["ema_params"] = state.ema_params
        per_strict: dict[str, str] = {}
        per_g0: dict[str, str] = {}
        for tree_name, tree in trees.items():
            flat, _ = jax.tree_util.tree_flatten_with_path(tree)
            for path, leaf in flat:
                if leaf is None:
                    continue
                key = tree_name + jax.tree_util.keystr(path)
                arr = np.asarray(jax.device_get(leaf))
                hs = hashlib.sha256()
                hs.update(key.encode())
                hs.update(str(arr.dtype).encode())
                hs.update(str(arr.shape).encode())
                hs.update(arr.tobytes())
                per_strict[key] = hs.hexdigest()
                hg = hashlib.sha256()   # 照抄 g0 _leaf_sha256 口径
                hg.update(str(arr.dtype).encode())
                hg.update(str(arr.shape).encode())
                hg.update(arr.tobytes())
                per_g0[key] = hg.hexdigest()
        gs, gg = hashlib.sha256(), hashlib.sha256()
        for key in sorted(per_strict):
            gs.update(f"{key}:{per_strict[key]}\n".encode())
            gg.update(f"{key}:{per_g0[key]}\n".encode())
        row = {"step": int(step),
               "strict_global": gs.hexdigest(),
               "g0_global": gg.hexdigest(),
               "treedef_sha": hashlib.sha256(
                   repr(jax.tree_util.tree_structure(trees)).encode()).hexdigest(),
               "keyset_sha": hashlib.sha256(
                   "\n".join(sorted(per_strict)).encode()).hexdigest(),
               "n_leaves": len(per_strict),
               "per_leaf_strict": per_strict, "per_leaf_g0": per_g0}
        with state_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[entry_equiv] step {step}: TrainState 摘要已记（{len(per_strict)} 叶，"
              f"不落真权重）", flush=True)

    _ckpt.save_state = _summarize_state
    _ckpt.initialize_checkpoint_dir = _wrapped_init_dir

    sys.path.insert(0, str(entry.parent))
    sys.argv = [entry.name, *tail]
    (record_dir / "harness_meta.json").write_text(json.dumps(
        {"entry": str(entry), "argv_tail": tail, "expect_root": str(expect_root),
         "expect_steps": args.expect_steps}, indent=2, ensure_ascii=False))

    rc = 0
    try:
        runpy.run_path(str(entry), run_name="__main__")
    except SystemExit as e:
        rc = int(e.code or 0)
    except BaseException as e:  # 落档后原样上报
        print(f"[entry_equiv] 入口异常: {type(e).__name__}: {e}", flush=True)
        rc = 1
    finally:
        _assert_provenance(expect_root)
        (record_dir / "provenance.json").write_text(json.dumps(
            _module_provenance(expect_root, entry), indent=2, ensure_ascii=False))
        _dump_resolved_config(record_dir)

    if rc == 0:
        if proxy.rows == 0:
            raise SystemExit("wandb 代理零记录——补丁未生效或训练没跑，fail-loud")
        tent, main_rows = _split_segments(all_path, record_dir)
        print(f"SEGMENTS tentative_rows={tent} main_rows={main_rows}")
        if main_rows != args.expect_steps:
            raise SystemExit(f"正式段行数 {main_rows} != 期望步数 {args.expect_steps}，"
                             "fail-loud")
        print("ENTRY_RUN=OK")
    return rc


# ────────────────────────── judge 侧 ──────────────────────────

def _load_metrics(path: pathlib.Path) -> dict[int, dict]:
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("step") is not None:
                out[int(r["step"])] = r
    return out


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in path.open() if x.strip()]


def _judge(args) -> int:
    import project_scalars  # 同目录；与 C3 收官共用同一份投影实现
    a_dir, b_dir = pathlib.Path(args.a_dir), pathlib.Path(args.b_dir)
    fails: list[str] = []

    # A 侧分段留档（非判据）
    tent = a_dir / "metrics_tentative.jsonl"
    tent_rows = sum(1 for _ in tent.open()) if tent.exists() else 0
    a_metrics = _load_metrics(a_dir / "metrics.jsonl")
    b_metrics = _load_metrics(b_dir / "metrics.jsonl")
    print(f"A_SIDE_SEGMENTS tentative_rows={tent_rows} main_rows={len(a_metrics)}")

    # ① ENTRY_SCALARS：逐步五键 hex 互比
    common = sorted(set(a_metrics) & set(b_metrics))
    mism = 0
    for s in common:
        for k in _SCALAR_KEYS:
            va, vb = a_metrics[s].get(k), b_metrics[s].get(k)
            if va is None or vb is None or va["hex"] != vb["hex"]:
                mism += 1
                break
    if len(a_metrics) != len(b_metrics) or len(common) != len(a_metrics):
        fails.append(f"两侧步集合不同: A={len(a_metrics)} B={len(b_metrics)} 共同={len(common)}")
    print(f"ENTRY_SCALARS steps={len(common)} keys={len(_SCALAR_KEYS)} hex_mismatch={mism}")
    if mism or len(common) != 1000:
        fails.append(f"ENTRY_SCALARS 失配: steps={len(common)} mismatch={mism}")

    # ② ENTRY_STATE_DIGEST：strict/treedef/keyset 三域互比
    a_st = {r["step"]: r for r in _load_jsonl(a_dir / "state_digests.jsonl")}
    b_st = {r["step"]: r for r in _load_jsonl(b_dir / "state_digests.jsonl")}
    st_common = sorted(set(a_st) & set(b_st))
    st_mism = sum(1 for s in st_common if any(
        a_st[s][k] != b_st[s][k] for k in ("strict_global", "treedef_sha", "keyset_sha")))
    if set(a_st) != set(b_st):
        fails.append(f"状态摘要步集合不同: A={sorted(a_st)} B={sorted(b_st)}")
    print(f"ENTRY_STATE_DIGEST rows={len(st_common)} mismatch={st_mism}")
    if st_mism or not st_common:
        fails.append(f"ENTRY_STATE_DIGEST 失配: rows={len(st_common)} mismatch={st_mism}")

    # ③ ENTRY_RESOLVED_CFG：逐顶层字段 repr，排除白名单四项
    a_cfg = json.loads((a_dir / "resolved_config.json").read_text())
    b_cfg = json.loads((b_dir / "resolved_config.json").read_text())
    if "fields" not in a_cfg or "fields" not in b_cfg:
        fails.append("resolved_config 缺 fields（某侧入口未跑到 import config）")
        cfg_mism = -1
    else:
        fa, fb = a_cfg["fields"], b_cfg["fields"]
        keys = (set(fa) | set(fb)) - _CFG_WHITELIST
        bad = sorted(k for k in keys if fa.get(k) != fb.get(k))
        cfg_mism = len(bad)
        for k in bad:
            print(f"CFG_DIFF field={k} a={fa.get(k)!r} b={fb.get(k)!r}")
        for k in _CFG_SPOTLIGHT:   # 单独列出对比、不埋进总哈希
            tag = "SAME" if fa.get(k) == fb.get(k) else "DIFF"
            print(f"CFG_SPOTLIGHT field={k} {tag}")
    print(f"ENTRY_RESOLVED_CFG mismatch={cfg_mism} whitelist={len(_CFG_WHITELIST)}")
    if cfg_mism != 0:
        fails.append(f"ENTRY_RESOLVED_CFG mismatch={cfg_mism}")

    # ④ ENTRY_PROVENANCE：两侧各自归位且根不同
    prov_ok = True
    roots = []
    for side, d in (("A", a_dir), ("B", b_dir)):
        p = json.loads((d / "provenance.json").read_text())
        root = pathlib.Path(p["expect_root"])
        roots.append(root)
        for name, m in p["modules"].items():
            if not pathlib.Path(m["file"]).is_relative_to(root):
                print(f"PROVENANCE_BAD side={side} module={name} file={m['file']}")
                prov_ok = False
    if len(roots) == 2 and roots[0] == roots[1]:
        print("PROVENANCE_BAD 两侧 expect_root 相同——A/B 跑的是同一份代码")
        prov_ok = False
    print(f"ENTRY_PROVENANCE={'PASS' if prov_ok else 'FAIL'}")
    if not prov_ok:
        fails.append("ENTRY_PROVENANCE=FAIL")

    # ⑤ 两侧各自投影对锚点
    shas = {}
    for side, d in (("A", a_dir), ("B", b_dir)):
        text = project_scalars.project(d / "metrics.jsonl")
        (d / "scalars_hex.tsv").write_text(text)
        shas[side] = hashlib.sha256(text.encode()).hexdigest()
        hit = shas[side] == args.expect_sha256
        print(f"{side}_SCALARS_SHA256 {shas[side]} anchor_hit={hit}")
        if not hit:
            fails.append(f"{side} 侧 scalars sha256 未命中锚点")

    verdict = "PASS" if not fails else "FAIL"
    for r in fails:
        print(f"ENTRY_EQ_FAIL reason={r}")
    print(f"ENTRY_EQ={verdict}")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--entry", required=True)
    p_run.add_argument("--record-dir", required=True)
    p_run.add_argument("--expect-root", required=True)
    p_run.add_argument("--expect-steps", type=int, default=1000)
    p_judge = sub.add_parser("judge")
    p_judge.add_argument("--a-dir", required=True)
    p_judge.add_argument("--b-dir", required=True)
    p_judge.add_argument("--expect-sha256", required=True)

    argv = sys.argv[1:]
    tail: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, tail = argv[:i], argv[i + 1:]
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return _run(args, tail)
    return _judge(args)


if __name__ == "__main__":
    sys.exit(main())
