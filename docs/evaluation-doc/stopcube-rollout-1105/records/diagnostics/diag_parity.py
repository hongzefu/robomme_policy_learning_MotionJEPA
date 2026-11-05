"""一次性诊断（不进仓库、只读输入、不写产物）：隐状态捕获开关差异的来源。"""
import dataclasses, gc, glob, json, sys
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
from openpi.shared import nnx_utils
from scripts.interpret.offline_query import unpack_observation, replay_query
from scripts.interpret.rollout_offline import build_groups
from scripts.interpret.analysis.offline_mechanism import aggregate_attention
import jax, jax.numpy as jnp

policy = build_policy(config.get_config(cfg), ckpt.resolve(), seed=42, motion_factory=lambda **k: _ZeroMotion())
adapter = AttnAdapter(policy)
print("embed_dtype", adapter.embed_dtype, flush=True)
paths = sorted(glob.glob(f"{root}/{label}/attn/*/rollout_*.npz"))
picks = [paths[0], paths[len(paths) // 2], paths[-1]]

def d(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))

def go(ad, path, f32=False):
    with np.load(path, allow_pickle=False) as z:
        rec = {k: z[k] for k in z.files}
    obs = jax.tree.map(jnp.asarray, unpack_observation(rec, HistAugObservation))
    tpf = int(policy.config.token_per_image) * int(policy.config.num_views)
    groups, masks = build_groups(obs, ad.layout(obs), rec, tpf)
    rng = jnp.asarray(rec["sample_rng"], dtype=jnp.uint32)
    noise = jnp.asarray(rec["noise"])
    off1 = jax.device_get(ad.run(rng, obs, noise=noise, out_f16=False)[:3])
    off2 = jax.device_get(ad.run(rng, obs, noise=noise, out_f16=False)[:3])
    on = ad.run(rng, obs, noise=noise, out_f16=False, capture_hidden=True, group_masks=masks)
    on3 = jax.device_get(on[:3])
    print(f"  off/off x0={d(off1[0], off2[0]):.3g} probs={d(off1[1], off2[1]):.3g}", flush=True)
    print(f"  on/off  x0={d(on3[0], off1[0]):.3g} probs_action={d(on3[1], off1[1]):.3g} probs_text={d(on3[2], off1[2]):.3g}", flush=True)
    print(f"  相对原路径动作 off={d(off1[0], rec['actions_normalized']):.3g} on={d(on3[0], rec['actions_normalized']):.3g}", flush=True)
    a_off, a_on = aggregate_attention(off1[1], groups), aggregate_attention(on3[1], groups)
    for g in ("motion", "历史视觉", "当前视觉", "文字", "其余"):
        if a_off[g]["tokens"]:
            x, y = np.array(a_off[g]["mass"]), np.array(a_on[g]["mass"])
            print(f"    {g}: 总质量 off={x.mean():.5f} on={y.mean():.5f} 单头最大差={np.abs(x - y).max():.4f}", flush=True)
    return rec, obs

for i, p in enumerate(picks):
    print(f"[bf16] {Path(p).name}", flush=True)
    r = replay_query(policy, p)
    print(f"  原路径重放 same_path={r['same_path_maxabs']:.3g} status={r['status']}", flush=True)
    go(adapter, p)

print("[f32 语义] 参数与 embed_dtype 升 f32、matmul_precision=highest", flush=True)
params = _model.restore_params(ckpt / "params", dtype=jnp.float32)
cfg32 = dataclasses.replace(policy._model.config, dtype="float32")
strict = (ckpt.parent / "history_config.resolved.yaml").is_file()
model32 = cfg32.load(params, remove_extra_params=not strict)
del params
gc.collect()
ad32 = AttnAdapter(_ModelShim(model32))
with jax.default_matmul_precision("highest"):
    for p in picks:
        print(f"[f32] {Path(p).name}", flush=True)
        go(ad32, p, True)
print("DIAG_DONE", flush=True)
