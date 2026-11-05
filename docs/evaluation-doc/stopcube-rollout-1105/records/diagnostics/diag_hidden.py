"""一次性诊断（不进仓库、只读输入、不写产物）：隐状态受 bf16 影响多大。

判据（对抗验证建议）：相对 L2 误差最大值 ≤2%，且至少比不同推理点之间的相对距离小一个量级。
"""
import dataclasses, gc, glob, sys
from pathlib import Path
import numpy as np

label, root, ckpt, cfg = sys.argv[1:5]
ckpt = Path(ckpt)
sys.path.insert(0, ".")
from scripts.interpret.attn_adapter import _ZeroMotion, _MV_DIR, AttnAdapter, _ModelShim
sys.path.insert(0, str(_MV_DIR))
from mv_model_adapter import build_policy
from mme_vla_suite.training import config
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from openpi.models import model as _model
from scripts.interpret.offline_query import unpack_observation
from scripts.interpret.rollout_offline import build_groups
import jax, jax.numpy as jnp

policy = build_policy(config.get_config(cfg), ckpt.resolve(), seed=42, motion_factory=lambda **k: _ZeroMotion())
adapter = AttnAdapter(policy)
paths = sorted(glob.glob(f"{root}/{label}/attn/*/rollout_*.npz"))
picks = [paths[0], paths[len(paths) // 2], paths[-1]]
tpf = int(policy.config.token_per_image) * int(policy.config.num_views)
GROUPS = ("历史视觉", "motion", "当前视觉", "文字")


def hidden(ad, path):
    with np.load(path, allow_pickle=False) as z:
        rec = {k: z[k] for k in z.files}
    obs = jax.tree.map(jnp.asarray, unpack_observation(rec, HistAugObservation))
    _, masks = build_groups(obs, ad.layout(obs), rec, tpf)
    out = ad.run(jnp.asarray(rec["sample_rng"], dtype=jnp.uint32), obs, noise=jnp.asarray(rec["noise"]),
                 out_f16=False, capture_hidden=True, group_masks=masks)
    h = jax.device_get(out[-1])
    return {k: np.asarray(h[k], np.float64) for k in ("hidden_prefix", "hidden_action", "group_tokens")}


bf = [hidden(adapter, p) for p in picks]
params = _model.restore_params(ckpt / "params", dtype=jnp.float32)
strict = (ckpt.parent / "history_config.resolved.yaml").is_file()
model32 = dataclasses.replace(policy._model.config, dtype="float32").load(params, remove_extra_params=not strict)
del params
gc.collect()
ad32 = AttnAdapter(_ModelShim(model32))
with jax.default_matmul_precision("highest"):
    fl = [hidden(ad32, p) for p in picks]


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def report(name, get, slices):
    errs, dists = [], []
    for s in slices:
        errs += [rel(get(bf[i])[s], get(fl[i])[s]) for i in range(len(picks))]
        dists += [rel(get(fl[i])[s], get(fl[j])[s]) for i in range(len(picks)) for j in range(i)]
    print(f"{name}: bf16对f32相对误差 中位={np.median(errs):.4f} 最大={np.max(errs):.4f} | "
          f"点间相对距离 中位={np.median(dists):.4f} 最小={np.min(dists):.4f}", flush=True)
    return float(np.max(errs)), float(np.median(dists))


results = {}
results["动作第17层"] = report("动作第17层", lambda h: h["hidden_action"], [17])
results["动作全部层"] = report("动作全部层", lambda h: h["hidden_action"], list(range(18)))
for g, name in enumerate(GROUPS):
    if all(h["group_tokens"][g] > 0 for h in bf):
        results[name] = report(f"前缀{name}", lambda h: h["hidden_prefix"][:, g], list(range(19)))
worst = max(v[0] for v in results.values())
ratio = min(v[1] / max(v[0], 1e-12) for v in results.values())
ok = worst <= 0.02 and ratio >= 10
print(f"HIDDEN_PRECISION={'PASS' if ok else 'FAIL'} model={label} points={len(picks)} worst_rel={worst:.4f} min_dist_over_err={ratio:.1f}", flush=True)
print("DIAG_DONE", flush=True)
