#!/usr/bin/env python3
"""官方 / 旧口径 checkpoint 参数树严格核对（留档 docs/training-doc/eval-official-framesamp-context/）。

背景：run 根只有 history_config.txt 的 checkpoint 走 create_trained_policy 的旧兼容分支——model.load(params) 默认
remove_extra_params=True，checkpoint 多出的参数会被静默 intersect 掉，加载成功不等于参数树一致。本脚本只读、CPU 上
（JAX_PLATFORMS=cpu）把 checkpoint params 与 HEAD 按 history_config.txt 所指 yaml 建出的模型逐路径比 missing / extra / shape，
并调用 policy_config._assert_param_tree_exact 复核。必须在仓库根运行（get_history_config 按 cwd 相对路径读 yaml）。
判定行：PARAM_TREE_EXACT=PASS|FAIL config=<c> history_config=<hc> yaml_sha256=<16位> n_model=<a> n_ckpt=<b> missing=<m> extra=<e> shape_mismatch=<s>
用法：JAX_PLATFORMS=cpu UV_LINK_MODE=copy uv run --no-sync python scripts/training/legacy-eval/check_ckpt_param_tree.py \\
        --ckpt-dir v1-store/models/official-mme-vla/perceptual-framesamp-context/79999 --out <records>/param_tree.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _paths_of(tree) -> dict:
    import jax
    return {"/".join(str(getattr(k, "key", getattr(k, "name", k))) for k in kp): v
            for kp, v in jax.tree_util.tree_flatten_with_path(tree)[0]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True, help="checkpoint 目录（含 params/；父目录须有 history_config.txt）")
    ap.add_argument("--config", default="mme_vla_suite", help="训练配置条目名（与 serve_policy --policy.config 一致）")
    ap.add_argument("--out", required=True, help="判定 json 输出路径")
    args = ap.parse_args()
    os.chdir(_REPO_ROOT)
    print(f"cwd={os.getcwd()} JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')}")

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    import openpi.models.model as _model
    import mme_vla_suite.training.config as _config
    from mme_vla_suite.policies import policy_config

    ckpt = pathlib.Path(args.ckpt_dir).resolve()
    run_root = ckpt.parent
    hc_path = run_root / "history_config.txt"
    if not hc_path.is_file():
        print(f"错误: 缺 {hc_path}", file=sys.stderr); return 2
    hc = hc_path.read_text()  # 与 create_trained_policy 同口径：原文不 strip
    print(f"history_config.txt = {hc!r}")
    yaml_path = pathlib.Path("src/mme_vla_suite/models/config/robomme") / hc
    yaml_sha = hashlib.sha256(yaml_path.read_bytes()).hexdigest() if yaml_path.is_file() else None
    print(f"yaml={yaml_path} exists={yaml_path.is_file()} sha256={yaml_sha}")

    train_config = _config.get_config(args.config)
    train_config = dataclasses.replace(
        train_config, model=dataclasses.replace(train_config.model, history_config=hc, use_history=True))
    t0 = time.time()
    params = _model.restore_params(ckpt / "params", restore_type=np.ndarray, dtype=jnp.bfloat16)
    print(f"restore_params 用时 {time.time() - t0:.0f}s")
    model = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    m = _paths_of(nnx.state(model, nnx.Param).to_pure_dict())
    c = _paths_of(params)
    missing = sorted(m.keys() - c.keys())
    extra = sorted(c.keys() - m.keys())
    shape_mismatch = sorted(k for k in (m.keys() & c.keys()) if tuple(m[k].shape) != tuple(c[k].shape))
    assert_ok, assert_err = True, None
    try:
        policy_config._assert_param_tree_exact(model, params)
    except Exception as e:  # noqa: BLE001
        assert_ok, assert_err = False, str(e)[:500]
    ok = not missing and not extra and not shape_mismatch and assert_ok
    line = (f"PARAM_TREE_EXACT={'PASS' if ok else 'FAIL'} config={args.config} history_config={hc.strip()} "
            f"yaml_sha256={(yaml_sha or 'NA')[:16]} n_model={len(m)} n_ckpt={len(c)} missing={len(missing)} extra={len(extra)} "
            f"shape_mismatch={len(shape_mismatch)}")
    print(line)
    for name, lst in (("missing", missing), ("extra", extra), ("shape_mismatch", shape_mismatch)):
        for k in lst[:20]:
            print(f"  {name}: {k}" + (f" model={tuple(m[k].shape)} ckpt={tuple(c[k].shape)}" if name == "shape_mismatch" else ""))
    if not assert_ok:
        print(f"  _assert_param_tree_exact: {assert_err}")
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"line": line, "ckpt_dir": str(ckpt), "config": args.config, "history_config_repr": repr(hc),
                               "yaml": str(yaml_path), "yaml_sha256": yaml_sha, "n_model": len(m), "n_ckpt": len(c),
                               "missing": missing, "extra": extra, "shape_mismatch": shape_mismatch,
                               "assert_param_tree_exact": assert_ok, "assert_error": assert_err}, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
