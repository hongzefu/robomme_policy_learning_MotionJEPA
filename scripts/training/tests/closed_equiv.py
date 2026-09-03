#!/usr/bin/env python3
"""关闭态轻量等价对拍 A13 / A15 / A17（motion-memory-plan.md 第二部分五节）：S2_BASE 代码 vs S2 新代码，motion.enabled=false。

两侧各跑一次 `dump`（PYTHONPATH 分别指向两棵源码树），再 `compare` 两份 json：
  dump     A15：固定 idx 集合的 `FrameSampDataset.__getitem__` 全键 raw sha256 / dtype / shape（None 键记 "None"）；
           collate 后 batch 全键（np.stack）raw sha；四键存在且为 None（新代码侧）；
           A17：`JAX_PLATFORMS=cpu`、`nnx.Rngs(0)` 现场 init（gemma dummy 变体）后 `embed_memory` 四返回与 `embed_prefix` 四返回的
           `.view(uint8)` sha256（`ar_mask` / `na_mask` 是 Python list → np.asarray）；
           A14 旁证：`mem_encoder.feature_encoder.*` 四叶 sha256；n_leaves（params 叶数）。
  compare  两份 json 逐键相等 → CLOSED_EQUIV=PASS；A13 静态检查（mem_encoder.py 零改动、even_sampling_indices 函数体零改动、
           sampling.py import 面只有 numpy、四字段声明顺序、新参数名不含 img）在 compare 里对当前源码树做。

用法：
    PYTHONPATH=<S2_BASE 树>/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= uv run scripts/training/tests/closed_equiv.py dump --out a.json
    PYTHONPATH=<新代码树>/src   JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= uv run scripts/training/tests/closed_equiv.py dump --out b.json
    uv run scripts/training/tests/closed_equiv.py compare --a a.json --b b.json --base-commit <S2_BASE>
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import types

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_V1 = pathlib.Path(os.environ.get("MMEVLA_V1_STORE", str(_REPO_ROOT / "v1-store")))
IDX = [0, 1, 2, 5, 30, 31, 100, 1000, 5000, 9000, 11529]
CLOSED_YAML = "perceptual-framesamp-context.yaml"


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).view(np.uint8).tobytes()).hexdigest()


def _desc(v) -> dict | str:
    if v is None:
        return "None"
    if isinstance(v, (bytes, str)):
        return {"kind": type(v).__name__, "sha256": hashlib.sha256(str(v).encode()).hexdigest()}
    a = np.asarray(v)
    return {"kind": "array", "dtype": str(a.dtype), "shape": list(a.shape), "sha256": _sha(a)}


def cmd_dump(args):
    import mme_vla_suite
    src_tree = pathlib.Path(mme_vla_suite.__file__).resolve().parents[2]
    import omegaconf
    from mme_vla_suite.training.dataloader import _create_framesamp_dataset
    lib = pathlib.Path(args.lib)
    ns = json.load(open(_V1 / "train-assets/mme_vla_suite/robomme/norm_stats.json"))["norm_stats"]["state"]
    st = types.SimpleNamespace(q01=np.array(ns["q01"]), q99=np.array(ns["q99"]), mean=np.array(ns["mean"]), std=np.array(ns["std"]))
    dc = types.SimpleNamespace(norm_stats={"state": st}, use_quantile_norm=True)
    hc = omegaconf.OmegaConf.load(src_tree / "src/mme_vla_suite/models/config/robomme" / CLOSED_YAML)
    ds = _create_framesamp_dataset(str(lib / "framesamp"), dc, hc, 20)
    out = {"src_tree": str(src_tree), "samples": {}, "batch": {}, "model": {}}
    samples = []
    for i in IDX:
        d = ds[i]
        out["samples"][str(i)] = {k: _desc(v) for k, v in sorted(d.items())}
        samples.append(d)
    keys = sorted(samples[0])
    out["sample_keys"] = keys
    for k in keys:
        vals = [s[k] for s in samples[:8]]
        if all(v is None for v in vals):
            out["batch"][k] = "None"
        elif all(isinstance(v, np.ndarray) for v in vals):
            out["batch"][k] = _desc(np.stack(vals))
        else:
            out["batch"][k] = {"kind": "other", "sha256": hashlib.sha256(repr(vals).encode()).hexdigest()}
    # A17：dummy 模型 embed_memory / embed_prefix
    import jax, jax.numpy as jnp
    from flax import nnx
    import openpi.shared.array_typing as at
    from openpi.models import gemma as _gemma
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    hc_m = omegaconf.OmegaConf.load(src_tree / "src/mme_vla_suite/models/config/robomme" / CLOSED_YAML)
    hc_m.memory_token_dim = int(_gemma.get_config("dummy").width)
    cfg = HistoryPi0Config(use_history=True, history_config=hc_m, dtype="bfloat16", action_horizon=20,
                           paligemma_variant="dummy", action_expert_variant="dummy")
    model = cfg.create(jax.random.key(0))
    B = 4
    def stack(k, dt=None):
        a = np.stack([np.asarray(s[k]) for s in samples[:B]]); return jnp.asarray(a if dt is None else a.astype(dt))
    obs = HistAugObservation(images={"base_0_rgb": jnp.zeros((B, 224, 224, 3), jnp.float32), "left_wrist_0_rgb": jnp.zeros((B, 224, 224, 3), jnp.float32)},
                             image_masks={"base_0_rgb": jnp.ones((B,), jnp.bool_), "left_wrist_0_rgb": jnp.ones((B,), jnp.bool_)},
                             state=jnp.zeros((B, 32), jnp.float32), tokenized_prompt=jnp.zeros((B, 64), jnp.int32),
                             tokenized_prompt_mask=jnp.zeros((B, 64), jnp.bool_).at[:, :8].set(True), token_ar_mask=None, token_loss_mask=None,
                             static_image_emb=stack("static_image_emb", np.float32), static_mask=stack("static_mask"),
                             static_pos_emb=stack("static_pos_emb"), static_state_emb=stack("static_state_emb", np.float32))
    with at.disable_typechecking():
        mem = model.embed_memory(obs)
        pre = model.embed_prefix(obs)
    out["model"]["embed_memory"] = [_desc(np.asarray(x)) for x in mem]
    out["model"]["embed_prefix"] = [_desc(np.asarray(x)) for x in pre]
    params = nnx.state(model, nnx.Param).to_pure_dict()
    leaves = {jax.tree_util.keystr(kp): _sha(v) for kp, v in jax.tree_util.tree_flatten_with_path(params)[0]}
    out["model"]["n_leaves_params"] = len(leaves)
    out["model"]["feature_encoder_leaves"] = {k: v for k, v in leaves.items() if "feature_encoder" in k}
    out["model"]["all_leaves_sha256"] = hashlib.sha256(json.dumps(leaves, sort_keys=True).encode()).hexdigest()
    ds.close()
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"dumped {args.out} src_tree={src_tree} sample_keys={keys}")


def cmd_compare(args):
    a = json.loads(pathlib.Path(args.a).read_text()); b = json.loads(pathlib.Path(args.b).read_text())
    fails = []
    # A15：样本全键与 batch 全键逐键相同；键集合：新代码侧多出四个 None 键，其余相同
    extra = {"motion_emb", "motion_pos", "motion_mask", "mem_order"}
    ka, kb = set(a["sample_keys"]), set(b["sample_keys"])
    if kb - ka != extra or ka - kb:
        fails.append(f"键集合差异不是恰好四个 motion 键: +{sorted(kb - ka)} -{sorted(ka - kb)}")
    for i in a["samples"]:
        for k, v in a["samples"][i].items():
            if b["samples"].get(i, {}).get(k) != v:
                fails.append(f"样本 {i} 键 {k} 不同"); break
        for k in extra & set(b["samples"].get(i, {})):
            if b["samples"][i][k] != "None":
                fails.append(f"样本 {i} 关闭态键 {k} 不是 None")
    for k, v in a["batch"].items():
        if b["batch"].get(k) != v:
            fails.append(f"batch 键 {k} 不同")
    # A17 / A14 旁证
    for name in ("embed_memory", "embed_prefix"):
        if a["model"][name] != b["model"][name]:
            fails.append(f"{name} 四返回不逐位")
    if a["model"]["n_leaves_params"] != b["model"]["n_leaves_params"]:
        fails.append(f"params 叶数 {a['model']['n_leaves_params']} != {b['model']['n_leaves_params']}")
    if a["model"]["feature_encoder_leaves"] != b["model"]["feature_encoder_leaves"]:
        fails.append("feature_encoder 叶子初始化不同")
    if a["model"]["all_leaves_sha256"] != b["model"]["all_leaves_sha256"]:
        fails.append("全部参数叶初始化不同")
    # A13：静态检查（对当前树，与 --base-commit 比）
    if args.base_commit:
        def show(rel):
            return subprocess.run(["git", "-C", str(_REPO_ROOT), "show", f"{args.base_commit}:{rel}"], capture_output=True, check=True).stdout.decode()
        rel = "src/mme_vla_suite/models/representation/mem_encoder.py"
        if show(rel) != (_REPO_ROOT / rel).read_text():
            fails.append("A13: mem_encoder.py 有改动")
        rel = "src/mme_vla_suite/shared/sampling.py"
        old_t, new_t = ast.parse(show(rel)), ast.parse((_REPO_ROOT / rel).read_text())
        def fn_src(tree, text):
            for n in ast.walk(tree):
                if isinstance(n, ast.FunctionDef) and n.name == "even_sampling_indices":
                    return ast.get_source_segment(text, n)
        if fn_src(old_t, show(rel)) != fn_src(new_t, (_REPO_ROOT / rel).read_text()):
            fails.append("A13: even_sampling_indices 函数体有改动")
        imports = {n.names[0].name.split(".")[0] if isinstance(n, ast.Import) else n.module.split(".")[0]
                   for n in ast.walk(new_t) if isinstance(n, (ast.Import, ast.ImportFrom))}
        if imports != {"numpy"}:
            fails.append(f"A13: sampling.py import 面 {sorted(imports)} != ['numpy']")
        obs_src = (_REPO_ROOT / "src/mme_vla_suite/models/integration/history_observation.py").read_text()
        order = [obs_src.index(k + ":") for k in ("static_image_emb", "static_mask", "static_pos_emb", "static_state_emb", "motion_emb", "motion_pos", "motion_mask", "mem_order")]
        if order != sorted(order):
            fails.append("A13: 四个 motion 字段未按序追加在 static_* 之后")
        pm = (_REPO_ROOT / "src/mme_vla_suite/models/representation/percep_mem.py").read_text()
        if "img" in "".join(ln for ln in pm.splitlines() if "self.motion_" in ln and "=" in ln and "nnx.Linear" in ln):
            fails.append("A13: 新参数名含 img")
    for f in fails:
        print("  ✗", f)
    print(f"CLOSED_EQUIV={'PASS' if not fails else 'FAIL'} samples={len(a['samples'])} keys={len(a['sample_keys'])}")
    if fails:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump"); p.add_argument("--out", required=True); p.add_argument("--lib", default=str(_V1 / "datasets/4task-motion-40ep")); p.set_defaults(func=cmd_dump)
    p = sub.add_parser("compare"); p.add_argument("--a", required=True); p.add_argument("--b", required=True); p.add_argument("--base-commit", default=""); p.set_defaults(func=cmd_compare)
    args = ap.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
