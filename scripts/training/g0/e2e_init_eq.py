#!/usr/bin/env python3
"""G9 初值相等闸（计划 1117-e2e-enc-train-plan.html 第二部分「三 闸门总表」G9 行，S3 写集）。

JAX_PLATFORMS=cpu 下用当前代码、同一初始化入口（训练条目 ``mme_vla_suite_b128_80k`` 的 ``model.create``，与
train.py ``init_train_state`` 同一函数），分别以 e2e YAML 与 MotionMEM YAML 建模型：

- INIT_EQ：全树逐叶比（含 action_in_proj / time_mlp_* / action_out_proj / memory expert / MemoryAttention 等全部叶），
  encoder 子树除外；另断言加入 encoder 前后 default RNG 流的 count 相等（R7，不等计入 mismatches）。
- R3_OFF_EQ：关闭态回归——对不写 encoder_kind=wan_e2e 的 YAML（MotionMEM、wantp768、wanraw9s32、无 motion 的 8x8），
  当前代码与基线代码（默认 dd39f3d，``git archive`` 取 src 到临时目录、子进程加载）建出的模型参数叶数、
  逐叶字节 sha256 与 param_norm（train_step 同口径：ndim>1 的 kernel，排除 bias/scale/pos_embedding/input_embedding）逐位相同。

本闸只证明当前代码下两种 YAML 初值相同，不证明与历史 80k 正式训练初值相同（计划七节）。
判定行：``INIT_EQ=PASS leaves_compared=<n> mismatches=0``、``R3_OFF_EQ=PASS yamls=<k> leaves=<n> mismatches=0``。

用法（worktree 内 CPU）：
    JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/e2e_init_eq.py [--variant real|small] [--base dd39f3d]
S1 的 encoder 未合入时须显式 ``--allow-stub``（用 test_e2e_dataset.py 的替身，只验 R7 接线），判定行带 ``encoder=stub``。
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO = pathlib.Path(__file__).resolve().parents[3]
YAML_DIR = REPO / "src/mme_vla_suite/models/config/robomme"
E2E_YAML = "perceptual-framesamp-modul-8frame-8x8-e2eenc.yaml"
MOTION_YAML = "perceptual-framesamp-modul-8frame-8x8-motion.yaml"
CLOSED_YAMLS = (MOTION_YAML, "perceptual-framesamp-modul-8frame-8x8-wantp768.yaml",
                "perceptual-framesamp-modul-8frame-8x8-wanraw9s32.yaml", "perceptual-framesamp-modul-8frame-8x8.yaml")
SMALL = {"paligemma_variant": "gemma_150m", "action_expert_variant": "gemma_300m"}


def _model_config(yaml_path: pathlib.Path, variant: str):
    from omegaconf import OmegaConf
    from mme_vla_suite.training import config as _config

    cfg = _config.get_config("mme_vla_suite_b128_80k")
    over = {"history_config": OmegaConf.load(yaml_path), "use_history": True}
    if variant == "small":
        over.update(SMALL)
    return dataclasses.replace(cfg.model, **over)


def summarize(yaml_path: pathlib.Path, variant: str, seed: int, *, track_rng: bool = False) -> dict:
    """建模型（真实初始化，非 eval_shape），返回 {leaf_path: sha256}、param_norm 十六进制与 RNG 计数记录。"""
    import jax
    import numpy as np
    import optax
    import flax.nnx as nnx
    from flax.traverse_util import flatten_dict
    import openpi.shared.nnx_utils as nnx_utils

    rng_log = {}
    if track_rng:
        from mme_vla_suite.models.representation import percep_mem as pm
        orig_rngs, orig_build = pm.e2e_rngs, pm.build_e2e_encoder

        def e2e_rngs(rngs):
            rng_log["outer"] = rngs
            rng_log["count_before"] = int(rngs.default.count.value)
            return orig_rngs(rngs)

        def build(mcfg, rngs, dtype):
            enc = orig_build(mcfg, rngs, dtype)
            rng_log["count_after"] = int(rng_log["outer"].default.count.value)
            return enc

        pm.e2e_rngs, pm.build_e2e_encoder = e2e_rngs, build
    try:
        mc = _model_config(yaml_path, variant)
        model = mc.create(jax.random.key(seed))
    finally:
        if track_rng:
            pm.e2e_rngs, pm.build_e2e_encoder = orig_rngs, orig_build
    state = nnx.state(model)
    leaves = {}
    for k, v in flatten_dict(state.to_pure_dict()).items():
        if v is None:
            continue
        a = np.asarray(jax.device_get(v))
        leaves["/".join(map(str, k))] = f"{a.dtype}:{a.shape}:" + hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    kernel_params = nnx.state(model, nnx.All(
        nnx.Param, nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
        lambda _, x: x.value is not None and x.value.ndim > 1))
    pn = float(optax.global_norm(kernel_params))
    del model, state
    out = {"leaves": leaves, "param_norm_hex": pn.hex(), "param_norm": pn}
    if track_rng:
        out["rng"] = {k: v for k, v in rng_log.items() if k != "outer"}
    return out


def _child(args) -> int:
    """基线子进程：只算关闭态 YAML 的摘要，写 JSON。"""
    res = {name: summarize(pathlib.Path(args.yaml_dir) / name, args.variant, args.seed) for name in args.child_yamls}
    pathlib.Path(args.child_out).write_text(json.dumps(res))
    return 0


def _baseline(base: str, variant: str, seed: int, yamls: list[str]) -> dict:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="s3-g9-base-"))
    tar = subprocess.run(["git", "-C", str(REPO), "archive", base, "src"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=tar, check=True)
    out = tmp / "base.json"
    env = dict(os.environ, PYTHONPATH=f"{tmp / 'src'}:{REPO / 'packages/openpi-client/src'}", JAX_PLATFORMS="cpu")
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--child-out", str(out), "--variant", variant,
           "--seed", str(seed), "--yaml-dir", str(YAML_DIR), "--child-yamls", *yamls]
    r = subprocess.run(cmd, env=env, cwd=str(tmp))
    if r.returncode != 0:
        raise SystemExit(f"基线子进程失败 rc={r.returncode}")
    probe = subprocess.run([sys.executable, "-c", "import mme_vla_suite;print(mme_vla_suite.__file__)"], env=env,
                           cwd=str(tmp), capture_output=True, text=True).stdout.strip()
    if not probe.startswith(str(tmp)):
        raise SystemExit(f"基线子进程 import 到了非基线代码: {probe}")
    return json.loads(out.read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("real", "small"), default="real")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default="dd39f3d")
    ap.add_argument("--allow-stub", action="store_true")
    ap.add_argument("--skip-r3", action="store_true")
    ap.add_argument("--child-out")
    ap.add_argument("--child-yamls", nargs="*")
    ap.add_argument("--yaml-dir", default=str(YAML_DIR))
    args = ap.parse_args(argv)
    if args.child_out:
        return _child(args)

    sys.path.insert(0, str(REPO / "src"))
    import mme_vla_suite
    if not mme_vla_suite.__file__.startswith(str(REPO)):
        raise SystemExit(f"mme_vla_suite 不在本仓库: {mme_vla_suite.__file__}")
    sys.path.insert(0, str(REPO / "scripts/training/tests"))
    import test_e2e_dataset as T          # 只取替身安装函数
    real = T.S1_REAL
    if not real and not args.allow_stub:
        raise SystemExit("S1 wan_e2e_encoder 未合入；只验接线时显式加 --allow-stub")
    from mme_vla_suite.models.representation import percep_mem as pm

    e2e = summarize(YAML_DIR / E2E_YAML, args.variant, args.seed, track_rng=True)
    mot = summarize(YAML_DIR / MOTION_YAML, args.variant, args.seed)
    enc_pre = f"mem_encoder/{pm.E2E_ENCODER_ATTR}/"
    e_rest = {k: v for k, v in e2e["leaves"].items() if not k.startswith(enc_pre)}
    n_enc = len(e2e["leaves"]) - len(e_rest)
    mism = sorted(k for k in set(e_rest) | set(mot["leaves"]) if e_rest.get(k) != mot["leaves"].get(k))
    rng = e2e.get("rng", {})
    rng_eq = "count_before" in rng and rng.get("count_before") == rng.get("count_after")
    n_mis = len(mism) + (0 if rng_eq else 1) + (0 if n_enc > 0 else 1)
    for k in mism[:20]:
        print(f"INIT_EQ_MISMATCH leaf={k}")
    tag = "" if real else " encoder=stub"
    print(f"INIT_EQ={'PASS' if n_mis == 0 else 'FAIL'} leaves_compared={len(set(e_rest) | set(mot['leaves']))} "
          f"mismatches={n_mis} encoder_leaves={n_enc} default_rng_count_before={rng.get('count_before')} "
          f"after={rng.get('count_after')} variant={args.variant}{tag}", flush=True)
    rc = 0 if n_mis == 0 else 1

    if not args.skip_r3:
        names = list(CLOSED_YAMLS)
        cur = {MOTION_YAML: mot}
        for n in names:
            if n not in cur:
                cur[n] = summarize(YAML_DIR / n, args.variant, args.seed)
        base = _baseline(args.base, args.variant, args.seed, names)
        r3_mis, r3_leaves = 0, 0
        for n in names:
            a, b = cur[n], base[n]
            bad = sorted(k for k in set(a["leaves"]) | set(b["leaves"]) if a["leaves"].get(k) != b["leaves"].get(k))
            pn_ok = a["param_norm_hex"] == b["param_norm_hex"]
            r3_leaves += len(b["leaves"])
            r3_mis += len(bad) + (0 if pn_ok else 1)
            print(f"R3_OFF_YAML yaml={n} leaves={len(a['leaves'])}/{len(b['leaves'])} mismatches={len(bad)} "
                  f"param_norm={a['param_norm_hex']} base={b['param_norm_hex']}", flush=True)
            for k in bad[:10]:
                print(f"R3_OFF_MISMATCH yaml={n} leaf={k}")
        print(f"R3_OFF_EQ={'PASS' if r3_mis == 0 else 'FAIL'} yamls={len(names)} leaves={r3_leaves} "
              f"mismatches={r3_mis} base={args.base} variant={args.variant}", flush=True)
        rc = rc or (0 if r3_mis == 0 else 1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
