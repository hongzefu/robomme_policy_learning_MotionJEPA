"""一次性诊断（不进仓库、只读输入、不写产物）：全部推理点的开关捕获差异分布，以及单头差落在哪。"""
import glob, sys
from pathlib import Path
import numpy as np

label, root, ckpt, cfg = sys.argv[1:5]
sys.path.insert(0, ".")
from scripts.interpret.attn_adapter import _ZeroMotion, _MV_DIR, AttnAdapter
sys.path.insert(0, str(_MV_DIR))
from mv_model_adapter import build_policy
from mme_vla_suite.training import config
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from scripts.interpret.offline_query import unpack_observation
from scripts.interpret.rollout_offline import build_groups, PARITY_GROUPS
from scripts.interpret.analysis.offline_mechanism import aggregate_attention
import jax, jax.numpy as jnp

policy = build_policy(config.get_config(cfg), Path(ckpt).resolve(), seed=42, motion_factory=lambda **k: _ZeroMotion())
adapter = AttnAdapter(policy)
tpf = int(policy.config.token_per_image) * int(policy.config.num_views)
paths = sorted(glob.glob(f"{root}/{label}/attn/*/rollout_*.npz"))
rows = []
for p in paths:
    with np.load(p, allow_pickle=False) as z:
        rec = {k: z[k] for k in z.files}
    obs = jax.tree.map(jnp.asarray, unpack_observation(rec, HistAugObservation))
    groups, masks = build_groups(obs, adapter.layout(obs), rec, tpf)
    rng = jnp.asarray(rec["sample_rng"], dtype=jnp.uint32)
    noise = jnp.asarray(rec["noise"])
    on = adapter.run(rng, obs, noise=noise, out_f16=False, capture_hidden=True, group_masks=masks)
    off = adapter.run(rng, obs, noise=noise, out_f16=False)
    x_on, pa_on = jax.device_get(on[:2])
    x_off, pa_off = jax.device_get(off[:2])
    five = {n: groups[n] for n in PARITY_GROUPS}
    a_on, a_off = aggregate_attention(pa_on, five), aggregate_attention(pa_off, five)
    best = (0.0, None)
    totals, mrel = [], None
    for n in PARITY_GROUPS:
        if not a_off[n]["tokens"]:
            continue
        x, y = np.array(a_on[n]["mass"]), np.array(a_off[n]["mass"])
        d = np.abs(x - y)
        i = np.unravel_index(int(d.argmax()), d.shape)
        if d.max() > best[0]:
            best = (float(d.max()), (n, int(i[0]), int(i[1]), float(y[i]), float(x[i])))
        totals.append(abs(x.mean() - y.mean()))
        if n == "motion" and y.mean() > 0:
            mrel = abs(x.mean() - y.mean()) / y.mean()
    rows.append((Path(p).name, float(np.abs(np.asarray(x_on, np.float64) - np.asarray(x_off, np.float64)).max()),
                 best[0], best[1], max(totals), mrel))
    print(rows[-1], flush=True)
head = np.array([r[2] for r in rows]); x0 = np.array([r[1] for r in rows]); tot = np.array([r[4] for r in rows])
mr = [r[5] for r in rows if r[5] is not None]
print(f"SUMMARY model={label} points={len(rows)} x0_max={x0.max():.5f} head中位={np.median(head):.5f} head_p90={np.percentile(head, 90):.5f} "
      f"head_max={head.max():.5f} 超0.005={int((head > 0.005).sum())} total_max={tot.max():.6f} motion_rel_max={max(mr) if mr else None}", flush=True)
print("DIAG_DONE", flush=True)
