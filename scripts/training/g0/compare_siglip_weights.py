#!/usr/bin/env python3
"""D-A1b：离线 `siglip_params.pkl` 与 `pi05_base` / 训练 checkpoint 的 `PaliGemma.img` 子树同源核对（纯 CPU，不占 GPU）。

背景（审计报告 train-infer-consistency-b12f1a0.md D-A1）：训练建库的记忆帧 SigLIP 特征由离线 f32 权重
`$OPENPI_DATA_HOME/pi05_vision_encoder/siglip_params.pkl`（`SigLipTokenizer`）算出；在线推理由 checkpoint 内 bf16 的
`PaliGemma.img` 算出。仓库里没有任何断言两份权重同源。本脚本回答三件事：

  PARAM_SAME_ENC=PASS|FAIL n_leaves=23 max_abs=<x>        pkl 23 叶 vs pi05_base img 子树（原 dtype）逐叶 max|Δ|；== 0 才 PASS
  CKPT_IMG_FROZEN=PASS|FAIL mismatched_leaves=<n>          base.astype(bf16) vs 训练 checkpoint img 子树逐叶 leaf_sha256（img 冻结 + bf16，EMA 不应改它）
  PKL_VS_CKPT max_abs=<x> max_rel=<x> rel_le_2^-8=<bool>   pkl(f32) vs ckpt(bf16→f32) 总差（描述性；纯 bf16 舍入的相对误差上界 2^-9）

键对齐：pkl 顶层即 flax 参数名（Transformer / embedding / pos_embedding / head），checkpoint 键去掉 `PaliGemma/img` 前缀后一一对应，
`Transformer/encoderblock/*` 因 scan=True 带前导维 27。

用法：
  JAX_PLATFORMS=cpu UV_LINK_MODE=copy uv run --no-sync python scripts/training/g0/compare_siglip_weights.py \
    --ckpt v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999/params --out <records>/weights.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "training" / "tests"))
_V1 = pathlib.Path(os.environ.get("MMEVLA_V1_STORE", str(_REPO_ROOT / "v1-store")))

import _common as C  # noqa: E402


def _flatten(tree, prefix=()):
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(_flatten(v, prefix + (str(k),)))
    else:
        out["/".join(prefix)] = tree
    return out


def _restore_img(params_path: pathlib.Path):
    from openpi.models import model as _model
    t0 = time.perf_counter()
    params = _model.restore_params(params_path, restore_type=np.ndarray, dtype=None)
    img = params["PaliGemma"]["img"]
    return _flatten(img), time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", default=str(_V1 / "models/pi05_vision_encoder/siglip_params.pkl"))
    ap.add_argument("--base", default=str(_V1 / "models/openpi-assets/checkpoints/pi05_base/params"))
    ap.add_argument("--ckpt", required=True, help="训练 checkpoint 的 params 目录（如 .../39999/params）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax.numpy as jnp

    t0 = time.perf_counter()
    with open(args.pkl, "rb") as f:
        pkl = _flatten(pickle.load(f))
    print(f"[w] pkl 叶 {len(pkl)} 载入 {time.perf_counter() - t0:.1f}s dtypes={sorted({str(v.dtype) for v in pkl.values()})}")
    base, s_base = _restore_img(pathlib.Path(args.base))
    print(f"[w] pi05_base img 叶 {len(base)} 载入 {s_base:.1f}s dtypes={sorted({str(v.dtype) for v in base.values()})}")
    ckpt, s_ckpt = _restore_img(pathlib.Path(args.ckpt))
    print(f"[w] ckpt img 叶 {len(ckpt)} 载入 {s_ckpt:.1f}s dtypes={sorted({str(v.dtype) for v in ckpt.values()})}")

    keys = sorted(pkl)
    if set(keys) != set(base) or set(keys) != set(ckpt):
        print(f"KEY_MISMATCH pkl-base={sorted(set(keys) ^ set(base))} pkl-ckpt={sorted(set(keys) ^ set(ckpt))}")
        return 2

    leaves = []
    same_max = 0.0
    frozen_mismatch = 0
    pk_max_abs = 0.0
    pk_max_rel = 0.0
    for k in keys:
        a = np.asarray(pkl[k]); b = np.asarray(base[k]); c = np.asarray(ckpt[k])
        if a.shape != b.shape or a.shape != c.shape:
            print(f"SHAPE_MISMATCH {k} pkl={a.shape} base={b.shape} ckpt={c.shape}")
            return 2
        a32 = a.astype(np.float32); b32 = b.astype(np.float32); c32 = c.astype(np.float32)
        d_ab = float(np.max(np.abs(a32 - b32)))
        same_max = max(same_max, d_ab)
        b_bf16 = b32.astype(jnp.bfloat16)          # 与 train.py init 的 astype(jnp.bfloat16) 同一转换
        frozen_ok = C.leaf_sha256(np.asarray(b_bf16)) == C.leaf_sha256(c) if str(c.dtype) == "bfloat16" \
            else np.array_equal(np.asarray(b_bf16).astype(np.float32), c32)
        frozen_mismatch += 0 if frozen_ok else 1
        d_ac = np.abs(a32 - c32)
        rel = d_ac / (np.abs(a32) + 1e-12)
        pk_max_abs = max(pk_max_abs, float(d_ac.max()))
        pk_max_rel = max(pk_max_rel, float(rel[np.abs(a32) > 1e-6].max()) if np.any(np.abs(a32) > 1e-6) else 0.0)
        leaves.append({"key": k, "shape": list(a.shape), "pkl_dtype": str(a.dtype), "base_dtype": str(b.dtype), "ckpt_dtype": str(c.dtype),
                       "max_abs_pkl_base": d_ab, "pkl_base_bitexact": bool(C.leaf_sha256(a) == C.leaf_sha256(b)),
                       "frozen_bf16_bitexact": bool(frozen_ok),
                       "max_abs_pkl_ckpt": float(d_ac.max()), "mean_abs_pkl_ckpt": float(d_ac.mean()),
                       "max_abs_value": float(np.abs(a32).max())})

    same = same_max == 0.0
    lines = [
        f"PARAM_SAME_ENC={'PASS' if same else 'FAIL'} n_leaves={len(keys)} max_abs={same_max:.3e}",
        f"CKPT_IMG_FROZEN={'PASS' if frozen_mismatch == 0 else 'FAIL'} mismatched_leaves={frozen_mismatch}",
        f"PKL_VS_CKPT max_abs={pk_max_abs:.3e} max_rel={pk_max_rel:.3e} rel_le_2^-8={pk_max_rel <= 2.0 ** -8}",
    ]
    for k in ("embedding/kernel", "embedding/bias", "pos_embedding"):
        L = next(x for x in leaves if x["key"] == k)
        lines.append(f"LEAF {k}: pkl_base_max_abs={L['max_abs_pkl_base']:.3e} frozen_bf16={L['frozen_bf16_bitexact']} "
                     f"pkl_ckpt_max_abs={L['max_abs_pkl_ckpt']:.3e} mean_abs={L['mean_abs_pkl_ckpt']:.3e} |max|={L['max_abs_value']:.3e}")
    print("\n".join(lines))
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"lines": lines, "leaves": leaves, "argv": sys.argv,
                               "pkl": args.pkl, "base": args.base, "ckpt": args.ckpt}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0 if (same and frozen_mismatch == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
