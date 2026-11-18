"""G1 移植对拍（两次）与 G1b bf16 噪声底：JAX ``WanLatentMotionEncoderJax`` 对 vendored d3213de PyTorch 参考。

计划 1117-e2e-enc-train-plan 一节 ``port_e2e_encoder_parity.py`` 行、三节 G1／G1b 行。

① 正式训练配置 depth=1（固定种子随机初始化，不载入 8 层 checkpoint）::

    JAX_DEFAULT_MATMUL_PRECISION=highest uv run scripts/training/g0/port_e2e_encoder_parity.py \
        --depth 1 --init seed --seed 0 --latents <P1 的 wan-latents 根> --windows <e2e_windows_v1.json>

② depth=8 载入 P3 预训练 checkpoint（证明多层实现正确、供 G3 数据通路验证）::

    ... port_e2e_encoder_parity.py --depth 8 --init ckpt --ckpt <P3 的 checkpoint_epoch_72.pt> ...

PyTorch 参照只 import ``scripts/training/g0/ref_torch/wan_latent_encoder_d3213de.py``。两侧同一份权重
（PyTorch state dict 经 ``wan_e2e_encoder.load_torch_state_dict`` 键映射进 nnx；RoPE cos/sin buffer 丢弃、
JAX 现算；未映射或多余的键即 raise → FAIL）。fp32 + ``jax.default_matmul_precision("highest")``，输入为
源 fp32 latent（只验移植）。前向 + 反向：loss = mean_over_windows(‖out‖²)，对参数求梯度逐叶比。

误差口径：
- ``fwd_rel_rmse`` = sqrt(mean_j (‖y_jax,j − y_pt,j‖₂ / ‖y_pt,j‖₂)²)（逐窗 rel_L2 再对窗取均方根，与 G3 同口径）；
- ``grad_rel_rmse`` = max_over_leaves ‖g_jax − g_pt‖₂ / ‖g_pt‖₂（每个参数张量的 rmse(Δ)/rms(ref)，取最差叶）。

判定行（计划三节）：
``E2E_PORT_PARITY=PASS windows=<n> manifest_sha=<s> depth=<1|8> init=<seed|ckpt> params=<N> fwd_rel_rmse<=1e-4 grad_rel_rmse<=1e-3``
前一行 ``E2E_PORT_PARITY_DETAIL`` 给实测值。depth=1 init=seed 时另打 G1b：
``E2E_BF16_FLOOR=RECORDED f=<x> windows=<n> manifest_sha=<s>``（训练精度路径：输入 = raw9_f32_to_bf16(源 latent)
→ 升 fp32 标准化 → bf16 计算，对 fp32 JAX 在源 fp32 latent 上的输出取 rel_rmse；只记录、不判）。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ref_torch"))
if str(REPO / "src") not in sys.path:
    sys.path.insert(1, str(REPO / "src"))

import e2e_windows_v1 as W  # noqa: E402

FWD_TOL = 1e-4
GRAD_TOL = 1e-3
P3_CKPT = pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/external/motionjepa/"
                       "wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt")


# ── 误差口径 ──────────────────────────────────────────────────────────────────────────────────────
def rel_rmse(x: np.ndarray, ref: np.ndarray) -> float:
    """逐窗 rel_L2 再对窗取均方根；x/ref: (n, d)。"""
    x = np.asarray(x, np.float64).reshape(len(ref), -1)
    ref = np.asarray(ref, np.float64).reshape(len(ref), -1)
    per = np.linalg.norm(x - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
    return float(np.sqrt(np.mean(per ** 2)))


def rel_l2(x, ref) -> float:
    x = np.asarray(x, np.float64).ravel()
    ref = np.asarray(ref, np.float64).ravel()
    return float(np.linalg.norm(x - ref) / max(np.linalg.norm(ref), 1e-30))


# ── 两侧模型 ──────────────────────────────────────────────────────────────────────────────────────
def build_torch_reference(depth: int, init: str, seed: int, ckpt: pathlib.Path | None):
    import torch
    import wan_latent_encoder_d3213de as ref
    from mme_vla_suite.models.representation import wan_e2e_encoder as E

    if init == "seed":
        torch.manual_seed(int(seed))
        model = ref.build_reference(depth, dropout=0.0, latents_mean=E.LATENTS_MEAN, latents_std=E.LATENTS_STD)
        info = f"seed={seed}"
    elif init == "ckpt":
        if ckpt is None:
            raise SystemExit("--init ckpt 需要 --ckpt")
        sha = W.file_sha256(ckpt)
        if sha != ref.P3_CKPT_SHA256:
            raise RuntimeError(f"ckpt sha256 {sha} != P3 {ref.P3_CKPT_SHA256}")
        model, missing, unexpected = ref.strict_load_ckpt(ckpt, depth)
        if missing or unexpected:
            raise RuntimeError(f"strict 加载失败 missing={missing} unexpected={unexpected}")
        info = f"ckpt_sha256={sha[:12]} missing=[] unexpected=[]"
    else:
        raise SystemExit(f"未知 --init {init}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return model, info


def build_jax_from_torch(tmodel, depth: int, dtype, seed: int = 0):
    import jax.numpy as jnp  # noqa: F401
    from flax import nnx
    from mme_vla_suite.models.representation import wan_e2e_encoder as E

    jm = E.WanLatentMotionEncoderJax(layers=depth, rngs=nnx.Rngs(int(seed)), dtype=dtype)
    E.load_torch_state_dict(jm, tmodel.state_dict())
    return jm


def torch_forward_backward(tmodel, x_batches, cot_batches=None, *, n_total: int, autocast_bf16=False,
                           input_bf16=False):
    """逐批前向 + 反向，返回 (out (n,768) f32, {参数名: grad})。

    cot_batches 为 None 时 loss = Σ‖out‖² / n_total；否则 loss = Σ out·cot（固定上游梯度）。
    """
    import torch

    tmodel.zero_grad(set_to_none=True)
    outs = []
    for bi, xb in enumerate(x_batches):
        x = torch.from_numpy(np.ascontiguousarray(xb, np.float32)).view(len(xb), 9, 16, 32, 32)
        if input_bf16:
            x = x.to(torch.bfloat16)
        if autocast_bf16:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y = tmodel(x)[:, 0]
        else:
            y = tmodel(x)[:, 0]
        yf = y.float()
        if cot_batches is None:
            loss = (yf ** 2).sum() / n_total
        else:
            loss = (yf * torch.from_numpy(cot_batches[bi])).sum()
        loss.backward()
        outs.append(yf.detach().numpy().copy())
    grads = {k: p.grad.detach().numpy().astype(np.float32).copy() for k, p in tmodel.named_parameters()}
    return np.concatenate(outs), grads


def jax_forward_backward(jmodel, x_batches, cot_batches=None, *, n_total: int, out_scale: float = 1.0):
    """JAX 侧同口径前向 + 反向；返回 (out f32, {torch 键名: grad})。out_scale 仅供 G1c 负例 ×2。"""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from mme_vla_suite.models.representation import wan_e2e_encoder as E

    graphdef, params, rest = nnx.split(jmodel, nnx.Param, ...)

    def fwd(p, x):
        m = nnx.merge(graphdef, p, rest)
        return (m(x).astype(jnp.float32) * out_scale)

    def loss_fn(p, x, c):
        y = fwd(p, x)
        if c is None:
            return jnp.sum(y ** 2) / n_total, y
        return jnp.sum(y * c), y

    vg = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    acc = None
    outs = []
    for bi, xb in enumerate(x_batches):
        c = None if cot_batches is None else jnp.asarray(cot_batches[bi])
        (_, y), g = vg(params, jnp.asarray(xb), c)
        acc = g if acc is None else jax.tree_util.tree_map(jnp.add, acc, g)
        outs.append(np.asarray(y, np.float32))
    return np.concatenate(outs), E.nnx_grads_to_torch_layout(jmodel, acc)


def batches(arr, bs):
    return [arr[i:i + bs] for i in range(0, len(arr), bs)]


def import_raw_bf16():
    """S2 契约：``mme_vla_suite.shared.raw_bf16.raw9_f32_to_bf16``；缺失返回 None（调用方打 FAIL）。"""
    try:
        from mme_vla_suite.shared.raw_bf16 import raw9_f32_to_bf16
    except ImportError:
        return None
    return raw9_f32_to_bf16


def load_inputs(args):
    manifest, msha = W.load_manifest(args.windows)
    wins = W.select_windows(manifest, args.max_windows)
    latents = pathlib.Path(args.latents) if args.latents else \
        pathlib.Path(manifest["dataset_root"]) / manifest["latents_relpath"]
    x = W.read_window_latents(latents, wins)
    return wins, msha, x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="G1 移植对拍 / G1b bf16 噪声底")
    ap.add_argument("--depth", type=int, choices=(1, 8), required=True)
    ap.add_argument("--init", choices=("seed", "ckpt"), default=None,
                    help="缺省：depth=1 → seed，depth=8 → ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=pathlib.Path, default=None)
    ap.add_argument("--latents", default=None, help="P1 wan-latents 根（缺省取清单里的 dataset_root）")
    ap.add_argument("--windows", type=pathlib.Path, default=W.DEFAULT_MANIFEST)
    ap.add_argument("--max-windows", type=int, default=None, help="仅 CPU 冒烟用：取前 n 窗（四任务轮转）")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--no-bf16-floor", action="store_true", help="depth=1 init=seed 时跳过 G1b")
    args = ap.parse_args(argv)
    init = args.init or ("seed" if args.depth == 1 else "ckpt")
    if init == "ckpt" and args.ckpt is None:
        args.ckpt = P3_CKPT

    import jax
    import jax.numpy as jnp
    import torch
    from mme_vla_suite.models.representation import wan_e2e_encoder as E

    jax.config.update("jax_default_matmul_precision", "highest")
    torch.set_num_threads(max(1, min(32, os.cpu_count() or 1)))
    t0 = time.time()
    wins, msha, x = load_inputs(args)
    n = len(wins)
    tmodel, info = build_torch_reference(args.depth, init, args.seed, args.ckpt)
    jmodel = build_jax_from_torch(tmodel, args.depth, jnp.float32, args.seed)
    want = E.E2E_PARAMS_DEPTH1 if args.depth == 1 else E.E2E_PARAMS_DEPTH8
    p_t = int(sum(p.numel() for p in tmodel.parameters()))
    p_j = E.count_params(jmodel)

    xb = batches(x, args.batch)
    y_t, g_t = torch_forward_backward(tmodel, xb, n_total=n)
    y_j, g_j = jax_forward_backward(jmodel, xb, n_total=n)
    fwd = rel_rmse(y_j, y_t)
    keys_t, keys_j = sorted(g_t), sorted(g_j)
    if keys_t != keys_j:
        raise RuntimeError(f"梯度键不一致：只在 torch {sorted(set(keys_t) - set(keys_j))}，"
                           f"只在 jax {sorted(set(keys_j) - set(keys_t))}")
    per = {k: rel_l2(g_j[k], g_t[k]) for k in keys_t}
    worst = max(per, key=per.get)
    grad = per[worst]
    ok = (p_t == want and p_j == want and fwd <= FWD_TOL and grad <= GRAD_TOL and np.isfinite(fwd)
          and np.isfinite(grad))
    print(f"E2E_PORT_PARITY_DETAIL depth={args.depth} init={init} {info} params_torch={p_t} params_jax={p_j} "
          f"fwd_rel_rmse={fwd:.3e} grad_rel_rmse={grad:.3e} worst_leaf={worst} leaves={len(per)} "
          f"windows={n} elapsed_s={time.time() - t0:.1f}")
    status = "PASS" if ok else "FAIL"
    print(f"E2E_PORT_PARITY={status} windows={n} manifest_sha={msha} depth={args.depth} init={init} "
          f"params={p_j} fwd_rel_rmse<=1e-4 grad_rel_rmse<=1e-3", flush=True)
    rc = 0 if ok else 1

    if args.depth == 1 and init == "seed" and not args.no_bf16_floor:
        f32_to_bf16 = import_raw_bf16()
        if f32_to_bf16 is None:
            print(f"E2E_BF16_FLOOR=FAIL reason=raw_bf16_missing windows={n} manifest_sha={msha}")
            return 1
        jb = build_jax_from_torch(tmodel, 1, jnp.bfloat16, args.seed)
        outs = []
        from flax import nnx
        gd, prm, rest = nnx.split(jb, nnx.Param, ...)
        f = jax.jit(lambda p, xx: nnx.merge(gd, p, rest)(xx).astype(jnp.float32))
        for b in xb:
            outs.append(np.asarray(f(prm, jnp.asarray(f32_to_bf16(b)))))
        floor = rel_rmse(np.concatenate(outs), y_j)
        print(f"E2E_BF16_FLOOR=RECORDED f={floor:.4e} windows={n} manifest_sha={msha}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
