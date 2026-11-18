#!/usr/bin/env python3
"""G3 记忆对齐（只验数据通路）：E2E 数据通路 vs MotionMEM 离线 768 表（计划 1117 第二部分「三 闸门总表」G3 行，S3 写集）。

验的是新表、读表、标准化、读出、mask/order 与 MotionMEM 离线表对齐，不是正式 encoder：脚本内临时把 encoder 配成
depth=8、载入 P3 预训练权重（冻结：只前向、不求梯度、不建优化器；不改 YAML，正式训练仍 layers: 1）。

输入：
  --ckpt     P3 checkpoint_epoch_72.pt（PyTorch 参照与 JAX 两侧同一份权重，取 ckpt["encoder"]）
  --table768 MotionMEM 离线 768 表根（float32，由 wan_motion_infer.py 在 autocast(bf16) 下生成，自带 bf16 噪声）
  --table16  s16 bf16 新表根（S2 导出，布局 motion-raw9bf16-grid16-demopad17-v1）
  --latents  P1 源 latent 根（<段>.bin，fp32 (n,9,16,32,32)，与 768 表 / 新表同一行序）
  --windows  固定窗口清单（缺省为提交的 e2e_windows_v1.json，S1 load_manifest 校验 sha；与 G1/G1b/G1c 同一批）
  --framesamp 帧库根（mask/order 比对用；缺省取 table768 的上级目录下 framesamp-8x8）

误差口径：逐窗 rel_L2 = ‖x−y‖₂/‖y‖₂，再对窗取均方根（rel_rmse）。
(a) JAX fp32 + matmul precision highest 对 PyTorch fp32（vendored d3213de 类，autocast 之外），两侧输入都是源 fp32 latent：
    ``E2E_MEM_FP32=PASS rel_rmse=<x> limit=1e-3``
(b) ε = rel_rmse(PyTorch fp32 现算, 离线表)；ε_ref_fwd = rel_rmse(PyTorch bf16 autocast, PyTorch fp32)（与 G1c 同口径，
    但在 depth=8 预训练权重上现算）；JAX 训练精度路径（读 s16 bf16 表 → 升 fp32 标准化 → bf16 计算）对离线表 rel_rmse
    ≤ ε + 3·ε_ref_fwd（线性相加，系数 3 写死，R4 不事后放宽）；floor_diag = JAX bf16 路径对 JAX fp32 路径（只作诊断）；
    mask / order 逐位相等：
    ``E2E_MEM_VS_MOTIONMEM=PASS windows=<n> manifest_sha=<s> eps=<ε> eps_ref_fwd=<x> floor_diag=<f> rel_rmse_bf16=<x>
      limit=<eps+3*eps_ref_fwd> mask_mismatch=0 order_mismatch=0``

依赖：S1 的 ``wan_e2e_encoder``（``WanLatentMotionEncoderJax``、``load_torch_state_dict``）、vendored
``ref_torch/wan_latent_encoder_d3213de.py``（``strict_load_ckpt``）、``e2e_windows_v1.py``（``load_manifest`` /
``select_windows`` / ``read_window_latents``，行号键 ``table_row``）；S2 的 bf16 布局读表。
``--max-windows`` 只供 CPU 冒烟（按任务轮转取前 n 窗），正式 G3 不加。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
G0 = pathlib.Path(__file__).resolve().parent
REF_TORCH = G0 / "ref_torch" / "wan_latent_encoder_d3213de.py"
WINDOWS_PY = G0 / "e2e_windows_v1.py"
E2E_YAML = REPO / "src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-e2eenc.yaml"
MOTION_YAML = REPO / "src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion.yaml"
G3_DEPTH = 8
FP32_LIMIT = 1e-3
REF_COEF = 3.0
ROW_SHAPE = (9, 16, 32, 32)


def _load_by_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rel_rmse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, np.float64).reshape(len(x), -1)
    y = np.asarray(y, np.float64).reshape(len(y), -1)
    r = np.linalg.norm(x - y, axis=1) / np.maximum(np.linalg.norm(y, axis=1), 1e-30)
    return float(np.sqrt(np.mean(r ** 2)))


def torch_ref(ckpt: pathlib.Path, device: str):
    """vendored d3213de 类 depth=8 strict 加载 P3 ckpt["encoder"]（冻结：只前向、不求梯度）。"""
    import torch

    sys.path.insert(0, str(REF_TORCH.parent))
    ref = _load_by_path("wan_latent_encoder_d3213de", REF_TORCH)
    model, missing, unexpected = ref.strict_load_ckpt(ckpt, G3_DEPTH)
    if missing or unexpected:
        raise SystemExit(f"vendored 类 strict 加载失败 missing={missing} unexpected={unexpected}")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def torch_forward(model, x: np.ndarray, device: str, autocast: bool, chunk: int = 128) -> np.ndarray:
    """x: (n,9,16384) f32 源 latent → (n,768) f32；autocast=True 为 PyTorch bf16 autocast 参考（ε_ref_fwd）。"""
    import torch

    outs = []
    with torch.no_grad():
        for i in range(0, len(x), chunk):
            z = torch.from_numpy(np.ascontiguousarray(x[i:i + chunk], np.float32)).view(-1, *ROW_SHAPE).to(device)
            if autocast:
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                    y = model(z)[:, 0]
            else:
                y = model(z)[:, 0]
            outs.append(y.float().cpu().numpy())
    return np.concatenate(outs)


def jax_encoder(tmodel, dtype):
    """depth=8 的 JAX encoder（脚本内临时覆盖 layers，不改 YAML），经 S1 键映射载入同一份 ckpt 权重。"""
    import jax
    import flax.nnx as nnx
    from omegaconf import OmegaConf
    from mme_vla_suite.models.representation import percep_mem as pm
    from mme_vla_suite.models.representation import wan_e2e_encoder as E

    mcfg = OmegaConf.load(E2E_YAML).motion
    enc = pm.build_e2e_encoder(mcfg, nnx.Rngs(params=jax.random.key(0)), dtype, layers=G3_DEPTH)
    E.load_torch_state_dict(enc, tmodel.state_dict())
    return enc


def jax_forward(enc, x, chunk: int = 128) -> np.ndarray:
    """x: (n,9,16384) bf16 或 f32 → (n,768) f32。"""
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx

    gd, st = nnx.split(enc)

    @jax.jit
    def f(st, z):
        return nnx.merge(gd, st)(z).astype(jnp.float32)

    outs = []
    for i in range(0, len(x), chunk):
        z = x[i:i + chunk]
        outs.append(np.asarray(f(st, z.reshape(len(z), ROW_SHAPE[0], -1))).reshape(len(z), -1))
    return np.concatenate(outs)


def mask_order_check(framesamp: pathlib.Path, t768: pathlib.Path, t16: pathlib.Path, wins: list,
                     max_eps: int) -> tuple[int, int, int]:
    """真实数据集两条路径：每个涉及的 episode 取末步样本，比 motion_mask / mem_order / motion_pos 逐位，
    并核 e2e motion_emb 前缀 = 新表可见窗行原字节。返回 (mask_mismatch, order_mismatch, samples)。"""
    import dataclasses
    from omegaconf import OmegaConf
    from mme_vla_suite.training import config as _config
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset

    cfg = _config.get_config("mme_vla_suite_b128_80k")
    v1 = REPO / "v1-store" if (REPO / "v1-store").exists() else pathlib.Path(
        os.environ.get("MMEVLA_V1_STORE", "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store"))
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, assets=dataclasses.replace(
        cfg.data.assets, assets_dir=str(v1 / "train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"))))
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    ds_e = FrameSampDataset(str(framesamp), dc, OmegaConf.load(E2E_YAML), 20, motion_root=str(t16))
    ds_m = FrameSampDataset(str(framesamp), dc, OmegaConf.load(MOTION_YAML), 20, motion_root=str(t768))
    gs = sorted({int(w["g"]) for w in wins})[:max_eps]
    mm = om = 0
    for g in gs:
        cand = np.nonzero(ds_e._epis_of == g)[0]
        idx = int(cand[np.argmax(ds_e._step_of[cand])])
        a, b = ds_e[idx], ds_m[idx]
        mm += int(a["motion_mask"].tobytes() != b["motion_mask"].tobytes()
                  or a["motion_pos"].tobytes() != b["motion_pos"].tobytes())
        om += int(a["mem_order"].tobytes() != b["mem_order"].tobytes())
        k = int(a["motion_mask"].sum())
        from mme_vla_suite.datastore import motion_store as ms
        rows_m, _ = ms.visible_motion_rows(ds_e._motion_entries[g], int(ds_e._step_of[idx]))
        tab = ds_e._ensure_motion_store()
        buf = np.empty((k,) + ROW_SHAPE, a["motion_emb"].dtype)
        tab.rows_into(rows_m, buf)
        mm += int(a["motion_emb"][:9 * k].tobytes() != buf.tobytes())
    return mm, om, len(gs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=pathlib.Path)
    ap.add_argument("--table768", required=True, type=pathlib.Path)
    ap.add_argument("--table16", required=True, type=pathlib.Path)
    ap.add_argument("--latents", required=True, type=pathlib.Path)
    ap.add_argument("--windows", type=pathlib.Path, default=None, help="缺省为提交的 e2e_windows_v1.json")
    ap.add_argument("--framesamp", type=pathlib.Path)
    ap.add_argument("--max-windows", type=int, default=None, help="只供 CPU 冒烟；正式 G3 不加")
    ap.add_argument("--mask-episodes", type=int, default=64)
    ap.add_argument("--torch-device", default="cpu")
    args = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp
    import ml_dtypes
    from mme_vla_suite.datastore import motion_store as ms

    W = _load_by_path("e2e_windows_v1_for_g3", WINDOWS_PY)
    manifest, man_sha = W.load_manifest(args.windows) if args.windows else W.load_manifest()
    wins = W.select_windows(manifest, args.max_windows)
    rows = np.asarray([int(w["table_row"]) for w in wins], np.int64)
    m768 = ms.MotionMeta.load(args.table768)
    m16 = ms.MotionMeta.load(args.table16)
    if m16.raw.get("layout") != ms.LAYOUT_RAW9BF16:
        raise SystemExit(f"--table16 布局不符: {m16.raw.get('layout')}")
    src = W.read_window_latents(args.latents, wins)                     # (n,9,16384) f32
    s768 = ms.MotionStore(args.table768, meta=m768)
    off = np.asarray(s768.rows(rows), np.float32)
    s16 = ms.MotionStore(args.table16, meta=m16)
    b16 = np.empty((len(rows),) + ROW_SHAPE, np.dtype(ml_dtypes.bfloat16))
    s16.rows_into(rows, b16)

    tmodel = torch_ref(args.ckpt, args.torch_device)
    t32 = torch_forward(tmodel, src, args.torch_device, autocast=False)
    tbf = torch_forward(tmodel, src, args.torch_device, autocast=True)
    with jax.default_matmul_precision("highest"):
        j32 = jax_forward(jax_encoder(tmodel, jnp.float32), src)
    jbf = jax_forward(jax_encoder(tmodel, jnp.bfloat16), b16)

    fp32_rel = rel_rmse(j32, t32)
    eps = rel_rmse(t32, off)
    eps_ref = rel_rmse(tbf, t32)
    floor = rel_rmse(jbf, j32)
    rel_bf16 = rel_rmse(jbf, off)
    limit = eps + REF_COEF * eps_ref
    fw = args.framesamp or (args.table768.parent / "framesamp-8x8")
    mm, om, n_s = mask_order_check(fw, args.table768, args.table16, wins, args.mask_episodes)

    ok_a = fp32_rel <= FP32_LIMIT
    print(f"E2E_MEM_FP32={'PASS' if ok_a else 'FAIL'} rel_rmse={fp32_rel:.6g} limit=1e-3", flush=True)
    ok_b = rel_bf16 <= limit and mm == 0 and om == 0
    smoke = "" if args.max_windows is None else f" smoke_max_windows={args.max_windows}"
    print(f"E2E_MEM_VS_MOTIONMEM={'PASS' if ok_b else 'FAIL'} windows={len(rows)} manifest_sha={man_sha} "
          f"eps={eps:.6g} eps_ref_fwd={eps_ref:.6g} floor_diag={floor:.6g} rel_rmse_bf16={rel_bf16:.6g} "
          f"limit={limit:.6g} mask_mismatch={mm} order_mismatch={om} depth={G3_DEPTH} mask_samples={n_s}{smoke}",
          flush=True)
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
