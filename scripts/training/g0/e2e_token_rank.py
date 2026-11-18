"""E2E encoder 训练诊断（1117-e2e-enc-train-plan.html 第二部分一节 e2e_token_rank.py 行、RB-F2）：只读。

在固定窗口清单 ``e2e_windows_v1.json``（S1，读入时校验提交 sha256）上，用 checkpoint 的 **EMA 参数**（本仓 ckpt 只存 EMA）
跑策略内 encoder，输入取 s16 bf16 新表（与 768 表同一行序，行号 = 清单 ``table_row``），产出 ``diagnostics/<step>.json``：

  - 有效秩：中心化输出的奇异值归一化为分布 p 后 ``exp(−Σ p·log p)``（与 train.py ``_e2e_effective_rank`` 同定义）；
  - 逐维方差（均值 / 最小 / 最大）与输出 RMS；
  - 与步 0 输出的线性 CKA（同一窗口配对；步 0 张量取 train.py 写的 ``diagnostics/0.npy``，缺则记 null）；
  - 与 MotionMEM 离线 768 token（``motion/`` 表同行）的线性 CKA，及「本步输出 RMS / MotionMEM token RMS」；
  - 对任务的线性 probe（多类逻辑回归，纯 numpy 全批梯度下降），按 episode 划分训练 / 验证（同一 episode 的窗不跨两侧）。

用法（CPU 或一张空闲卡，按三节 GPU 空闲判据重查；主检出根目录）::

  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/e2e_token_rank.py \\
      --run-root v1-store/train-runs/mme_vla_suite_b128_80k/<run> --step 5000

缺省：--yaml 取 run 快照 ``history_config.resolved.yaml``，--table16 / --table768 取 4task 库的 motion-wan-raw9-s16 / motion，
--out 为 ``<run-root>/diagnostics/<step>.json``（已存在即拒）。末行打印
``E2E_TOKEN_RANK=RECORDED step=<s> windows=<n> manifest_sha=<16> effective_rank=<x> cka_step0=<x> cka_motionmem=<x> probe_val_acc=<x>``
（只记录、不判，R10）。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np

ENC_ATTR = "motion_encoder_e2e"          # percep_mem.E2E_ENCODER_ATTR（S3 契约）
REPO = pathlib.Path(__file__).resolve().parents[3]
DS_DEFAULT = pathlib.Path("v1-store/datasets/4task-v2-1600ep-604f16da")


# ─────────────────────────── 统计量（纯 numpy，单测直接调用） ───────────────────────────


def effective_rank(x: np.ndarray) -> float:
    xc = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(xc.astype(np.float64), compute_uv=False)
    p = s / max(s.sum(), 1e-30)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """线性 CKA（Kornblith 2019）：‖Yᶜᵀ Xᶜ‖²_F / (‖Xᶜᵀ Xᶜ‖_F ‖Yᶜᵀ Yᶜ‖_F)，行配对。"""
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"CKA 行数不配对：{x.shape[0]} vs {y.shape[0]}")
    xc = (x - x.mean(0)).astype(np.float64)
    yc = (y - y.mean(0)).astype(np.float64)
    num = np.linalg.norm(yc.T @ xc) ** 2
    den = np.linalg.norm(xc.T @ xc) * np.linalg.norm(yc.T @ yc)
    return float(num / den) if den > 0 else float("nan")


def episode_split(groups: list, val_frac: float = 0.25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """按 episode（组键）划分：同一组的样本只落一侧。返回 (train_idx, val_idx)。"""
    keys = sorted(set(groups))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    n_val = max(1, int(round(len(keys) * val_frac)))
    val_keys = {keys[i] for i in order[:n_val]}
    g = np.asarray([k in val_keys for k in groups])
    return np.nonzero(~g)[0], np.nonzero(g)[0]


def logistic_probe(x: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray,
                   iters: int = 500, lr: float = 0.5, l2: float = 1e-3) -> dict:
    """多类逻辑回归（标准化特征、全批梯度下降、L2）。返回训练 / 验证准确率。"""
    classes = sorted(set(y.tolist()))
    yi = np.asarray([classes.index(v) for v in y])
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    z = (x - mu) / sd
    w = np.zeros((z.shape[1], len(classes)))
    b = np.zeros(len(classes))
    onehot = np.eye(len(classes))[yi[tr]]
    for _ in range(iters):
        logits = z[tr] @ w + b
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits); p /= p.sum(1, keepdims=True)
        g = (p - onehot) / len(tr)
        w -= lr * (z[tr].T @ g + l2 * w)
        b -= lr * g.sum(0)
    acc = lambda idx: float(((z[idx] @ w + b).argmax(1) == yi[idx]).mean()) if len(idx) else float("nan")
    return {"classes": classes, "train_acc": acc(tr), "val_acc": acc(va), "n_train": int(len(tr)), "n_val": int(len(va))}


def summarize(out: np.ndarray, groups: list, labels: list, step0: np.ndarray | None, mm: np.ndarray | None) -> dict:
    var = out.var(axis=0)
    rms = float(np.sqrt((out.astype(np.float64) ** 2).mean()))
    tr, va = episode_split(groups)
    res = {
        "windows": int(out.shape[0]), "out_shape": list(out.shape),
        "out_sha256": hashlib.sha256(np.ascontiguousarray(out).tobytes()).hexdigest(),
        "effective_rank": effective_rank(out), "rms": rms,
        "dim_var_mean": float(var.mean()), "dim_var_min": float(var.min()), "dim_var_max": float(var.max()),
        "finite": bool(np.isfinite(out).all()),
        "cka_step0": linear_cka(out, step0) if step0 is not None else None,
        "cka_motionmem": linear_cka(out, mm) if mm is not None else None,
        "rms_ratio_vs_motionmem": (rms / float(np.sqrt((mm.astype(np.float64) ** 2).mean()))) if mm is not None else None,
        "probe_task": logistic_probe(out, np.asarray(labels), tr, va),
        "probe_split": "by_episode(task,episode) val_frac=0.25 seed=0",
    }
    if step0 is not None:
        r0 = float(np.sqrt((step0.astype(np.float64) ** 2).mean()))
        res["rms_ratio_step0_vs_motionmem"] = (r0 / float(np.sqrt((mm.astype(np.float64) ** 2).mean()))) if mm is not None else None
    return res


# ─────────────────────────── checkpoint → encoder ───────────────────────────


def _norm_key(k) -> str:
    return str(getattr(k, "key", getattr(k, "name", getattr(k, "idx", k))))


def encoder_subtree(params) -> dict[tuple, np.ndarray]:
    """从 restore_params 的 pure dict 里取 encoder 子树：{路径（encoder 属性之后，字符串元组）: 数组}。"""
    import jax
    out = {}
    for kp, x in jax.tree_util.tree_flatten_with_path(params)[0]:
        keys = [_norm_key(k) for k in kp]
        if ENC_ATTR in keys:
            out[tuple(keys[keys.index(ENC_ATTR) + 1:])] = x
    if not out:
        raise SystemExit(f"checkpoint 里找不到 encoder 子树（{ENC_ATTR}）")
    return out


def load_encoder(enc_cfg, sub: dict[tuple, np.ndarray]):
    """按 YAML encoder 节建 WanLatentMotionEncoderJax，把 checkpoint 子树逐叶写入（路径、形状、叶数必须全对上）。"""
    import flax.nnx as nnx
    import jax.numpy as jnp
    from mme_vla_suite.models.representation.wan_e2e_encoder import WanLatentMotionEncoderJax

    enc = WanLatentMotionEncoderJax(enc_cfg, rngs=nnx.Rngs(params=0), dtype=jnp.bfloat16)
    st = nnx.state(enc, nnx.Param)
    pure = st.to_pure_dict()
    used = set()

    def fill(node, prefix):
        if isinstance(node, dict):
            return {k: fill(v, prefix + (str(k),)) for k, v in node.items()}
        if node is None:                 # 无 bias 的线性层等：模型侧空叶，checkpoint 里本就没有
            return None
        if prefix not in sub:
            raise SystemExit(f"checkpoint encoder 子树缺叶 {'/'.join(prefix)}")
        val = np.asarray(sub[prefix])
        if tuple(val.shape) != tuple(node.shape):
            raise SystemExit(f"encoder 叶 {'/'.join(prefix)} 形状 {val.shape} != 模型 {node.shape}")
        used.add(prefix)
        return jnp.asarray(val, dtype=node.dtype)

    new = fill(pure, ())
    extra = sorted(set(sub) - used)
    if extra:
        raise SystemExit(f"checkpoint encoder 子树多出叶 {['/'.join(p) for p in extra[:4]]}")
    st.replace_by_pure_dict(new)
    nnx.update(enc, st)
    return enc


def encode(enc, rows_bf16: np.ndarray, chunk: int = 128) -> np.ndarray:
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    gd, st = nnx.split(enc)

    @jax.jit
    def f(s, z):
        return nnx.merge(gd, s)(z).astype(jnp.float32)

    outs = []
    for i in range(0, len(rows_bf16), chunk):
        z = rows_bf16[i:i + chunk]
        outs.append(np.asarray(f(st, z.reshape(len(z), 9, -1))))
    return np.concatenate(outs, axis=0)


def read_rows(root: pathlib.Path, rows: np.ndarray) -> np.ndarray:
    from mme_vla_suite.datastore import motion_store as ms
    meta = ms.MotionMeta.load(root)
    store = ms.MotionStore(root, meta=meta)
    try:
        buf = np.empty((len(rows),) + tuple(meta.spec.row_shape), dtype=meta.spec.np_dtype)
        store.rows_into(rows, buf)
    finally:
        store.close()
    return buf


def load_windows(path: pathlib.Path | None):
    spec = importlib.util.spec_from_file_location("e2e_windows_v1_diag", str(REPO / "scripts/training/g0/e2e_windows_v1.py"))
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    manifest, sha = w.load_manifest(path) if path else w.load_manifest()
    return manifest["windows"], sha


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True, type=pathlib.Path)
    ap.add_argument("--step", required=True, type=int)
    ap.add_argument("--yaml", type=pathlib.Path, default=None, help="缺省 <run-root>/history_config.resolved.yaml")
    ap.add_argument("--table16", type=pathlib.Path, default=DS_DEFAULT / "motion-wan-raw9-s16")
    ap.add_argument("--table768", type=pathlib.Path, default=DS_DEFAULT / "motion")
    ap.add_argument("--windows", type=pathlib.Path, default=None, help="缺省 S1 的 e2e_windows_v1.json（校验提交 sha256）")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    import jax.numpy as jnp
    import openpi.models.model as _model
    from omegaconf import OmegaConf

    out_path = args.out or (args.run_root / "diagnostics" / f"{args.step}.json")
    if out_path.exists():
        print(f"拒绝覆盖已存在的诊断 {out_path}", file=sys.stderr)
        return 2
    ycfg = OmegaConf.load(str(args.yaml or (args.run_root / "history_config.resolved.yaml")))
    enc_cfg = OmegaConf.to_container(ycfg.motion.encoder, resolve=True)
    windows, man_sha = load_windows(args.windows)
    if args.max_windows:
        windows = windows[:args.max_windows]
    rows = np.asarray([int(w["table_row"]) for w in windows], dtype=np.int64)
    groups = [(w["task"], int(w["episode"])) for w in windows]
    labels = [w["task"] for w in windows]

    params = _model.restore_params(args.run_root / str(args.step) / "params", dtype=jnp.float32)
    enc = load_encoder(enc_cfg, encoder_subtree(params))
    out = encode(enc, read_rows(args.table16, rows))
    mm = read_rows(args.table768, rows).reshape(len(rows), -1).astype(np.float32) if args.table768.exists() else None
    s0p = args.run_root / "diagnostics" / "0.npy"
    step0 = np.load(s0p)[:len(rows)] if s0p.is_file() else None

    res = {"step": args.step, "params": "ema(checkpoint)", "run_root": str(args.run_root), "manifest_sha256": man_sha,
           "table16": str(args.table16), "table768": str(args.table768), "step0": str(s0p) if step0 is not None else None,
           **summarize(out, groups, labels, step0, mm)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    fmt = lambda v: "null" if v is None else f"{v:.4f}"
    print(f"E2E_TOKEN_RANK=RECORDED step={args.step} windows={len(rows)} manifest_sha={man_sha[:16]} "
          f"effective_rank={res['effective_rank']:.3f} cka_step0={fmt(res['cka_step0'])} cka_motionmem={fmt(res['cka_motionmem'])} "
          f"probe_val_acc={res['probe_task']['val_acc']:.4f} out={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
