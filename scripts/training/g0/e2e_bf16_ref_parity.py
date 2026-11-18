"""G1c bf16 参考对拍（CPU 或 1 卡）：JAX bf16 encoder 对独立 PyTorch autocast bf16 参考（vendored d3213de，depth=1）。

计划 1117-e2e-enc-train-plan 一节 ``e2e_bf16_ref_parity.py`` 行、三节 G1c 行。

- 同一批 bf16 输入：``raw9_f32_to_bf16``（S2 契约，``mme_vla_suite.shared.raw_bf16``）作用于
  ``e2e_windows_v1.json`` 窗口的源 fp32 latent；
- 同一份权重：depth=1，与 G1 ① 相同的固定种子随机初始化（``torch.manual_seed(seed)`` 后构造 vendored 类），
  不载入 8 层 checkpoint；JAX 侧经 ``load_torch_state_dict`` 载入；
- 三条路径：JAX（参数 f32、计算 bf16、输入 bf16）、PyTorch autocast bf16（输入 bf16）、PyTorch fp32（输入为同一
  bf16 值升 f32）；前向 + 反向，loss = Σ out·G（G 为固定种子的随机上游梯度），按组比参数 VJP：
  输入投影 / 块 0 / 最终归一化 / 输出投影；
- 容差只由参考侧定：ε_ref_fwd = rel_rmse(PT bf16, PT fp32)，ε_ref_grad[g] = rel_L2(PT bf16 组梯度, PT fp32 组梯度)；
  limit_fwd = 3·ε_ref_fwd，limit_grad[g] = 3·ε_ref_grad[g]；fwd_rel = rel_rmse(JAX bf16, PT bf16)，
  grad_rel[g] = rel_L2(JAX bf16 组梯度, PT bf16 组梯度)；全部 ≤ 限值才 PASS；
- 两个内置负例必须判 FAIL：bf16 分支输出 ×2；bf16 RMSNorm 分母 r 加 stop_gradient。

判定行：
``E2E_BF16_REF=PASS windows=<n> manifest_sha=<s> fwd_rel=<x> fwd_limit=<y> grad_rel_max=<x> grad_limit_min=<y> neg_x2=FAIL_AS_EXPECTED neg_stopgrad=FAIL_AS_EXPECTED``
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import port_e2e_encoder_parity as P  # noqa: E402  (同时把 ref_torch 与 src 放进 sys.path)
import e2e_windows_v1 as W  # noqa: E402

COEF = 3.0
GROUP_ORDER = ("input_proj", "block0", "final_norm", "output_proj")


def group_of(key: str) -> str:
    if key.startswith("input_proj."):
        return "input_proj"
    if key.startswith("blocks."):
        return "block" + key.split(".")[1]
    if key.startswith("final_norm."):
        return "final_norm"
    if key.startswith("output_proj."):
        return "output_proj"
    raise KeyError(key)


def group_concat(grads: dict) -> dict:
    out: dict[str, list] = {}
    for k in sorted(grads):
        out.setdefault(group_of(k), []).append(np.asarray(grads[k], np.float64).ravel())
    return {g: np.concatenate(v) for g, v in out.items()}


def judge(y_c, g_c, y_b, g_b, eps_fwd, eps_grad):
    """候选 (y_c, g_c) 对 PT bf16 参考 (y_b, g_b)；返回 (ok, fwd_rel, grad_rel{g})。"""
    fwd_rel = P.rel_rmse(y_c, y_b)
    gc, gb = group_concat(g_c), group_concat(g_b)
    grad_rel = {g: P.rel_l2(gc[g], gb[g]) for g in gb}
    ok = bool(np.isfinite(fwd_rel) and fwd_rel <= COEF * eps_fwd)
    for g, v in grad_rel.items():
        ok = ok and bool(np.isfinite(v) and v <= COEF * eps_grad[g])
    return ok, fwd_rel, grad_rel


def run(args) -> int:
    import jax
    import jax.numpy as jnp
    import torch

    jax.config.update("jax_default_matmul_precision", "highest")
    torch.set_num_threads(max(1, min(32, os.cpu_count() or 1)))
    t0 = time.time()
    f32_to_bf16 = P.import_raw_bf16()
    wins, msha, x_src = P.load_inputs(args)
    n = len(wins)
    if f32_to_bf16 is None:
        print(f"E2E_BF16_REF=FAIL reason=raw_bf16_missing windows={n} manifest_sha={msha}")
        return 1
    x_bf = f32_to_bf16(x_src)                       # ml_dtypes.bfloat16 (n,9,16384)
    x_bf_as_f32 = np.asarray(x_bf, np.float32)      # 同一 bf16 值，升 f32（精确）
    cot = np.random.default_rng(args.cot_seed).standard_normal((n, 768)).astype(np.float32)
    xb32 = P.batches(x_bf_as_f32, args.batch)
    cb = P.batches(cot, args.batch)

    tmodel, info = P.build_torch_reference(1, "seed", args.seed, None)
    y32, g32 = P.torch_forward_backward(tmodel, xb32, cb, n_total=n)
    yb, gb = P.torch_forward_backward(tmodel, xb32, cb, n_total=n, autocast_bf16=True, input_bf16=True)
    eps_fwd = P.rel_rmse(yb, y32)
    gb_c, g32_c = group_concat(gb), group_concat(g32)
    eps_grad = {g: P.rel_l2(gb_c[g], g32_c[g]) for g in gb_c}
    if sorted(eps_grad) != sorted(GROUP_ORDER):
        raise RuntimeError(f"参数分组 {sorted(eps_grad)} != {GROUP_ORDER}")

    jm = P.build_jax_from_torch(tmodel, 1, jnp.bfloat16, args.seed)
    xbb = [np.asarray(b) for b in P.batches(x_bf, args.batch)]   # bf16 输入
    yj, gj = P.jax_forward_backward(jm, xbb, cb, n_total=n)
    ok, fwd_rel, grad_rel = judge(yj, gj, yb, gb, eps_fwd, eps_grad)

    # 负例 1：bf16 分支输出 ×2
    y2, g2 = P.jax_forward_backward(jm, xbb, cb, n_total=n, out_scale=2.0)
    ok_x2, fwd_x2, grad_x2 = judge(y2, g2, yb, gb, eps_fwd, eps_grad)
    # 负例 2：bf16 RMSNorm 分母 r 加 stop_gradient
    jm.set_rms_stop_gradient(True)
    try:
        ys, gs = P.jax_forward_backward(jm, xbb, cb, n_total=n)
    finally:
        jm.set_rms_stop_gradient(False)
    ok_sg, fwd_sg, grad_sg = judge(ys, gs, yb, gb, eps_fwd, eps_grad)

    neg_x2 = "FAIL_AS_EXPECTED" if not ok_x2 else "NOT_DETECTED"
    neg_sg = "FAIL_AS_EXPECTED" if not ok_sg else "NOT_DETECTED"
    fwd_limit = COEF * eps_fwd
    grad_limit = {g: COEF * v for g, v in eps_grad.items()}
    for g in GROUP_ORDER:
        print(f"E2E_BF16_REF_GROUP group={g} grad_rel={grad_rel[g]:.4e} grad_limit={grad_limit[g]:.4e} "
              f"eps_ref_grad={eps_grad[g]:.4e} neg_x2={grad_x2[g]:.3e} neg_stopgrad={grad_sg[g]:.3e}")
    print(f"E2E_BF16_REF_DETAIL {info} eps_ref_fwd={eps_fwd:.4e} neg_x2_fwd={fwd_x2:.3e} "
          f"neg_stopgrad_fwd={fwd_sg:.3e} cot_seed={args.cot_seed} elapsed_s={time.time() - t0:.1f}")
    passed = ok and not ok_x2 and not ok_sg
    print(f"E2E_BF16_REF={'PASS' if passed else 'FAIL'} windows={n} manifest_sha={msha} fwd_rel={fwd_rel:.4e} "
          f"fwd_limit={fwd_limit:.4e} grad_rel_max={max(grad_rel.values()):.4e} "
          f"grad_limit_min={min(grad_limit.values()):.4e} neg_x2={neg_x2} neg_stopgrad={neg_sg}", flush=True)
    return 0 if passed else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="G1c bf16 参考对拍（depth=1）")
    ap.add_argument("--seed", type=int, default=0, help="权重种子（与 G1 ① 相同）")
    ap.add_argument("--cot-seed", type=int, default=1234, help="固定随机上游梯度种子")
    ap.add_argument("--latents", default=None)
    ap.add_argument("--windows", type=pathlib.Path, default=W.DEFAULT_MANIFEST)
    ap.add_argument("--max-windows", type=int, default=None, help="仅 CPU 冒烟用")
    ap.add_argument("--batch", type=int, default=128)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
