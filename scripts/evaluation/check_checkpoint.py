"""不加载大数组，先核对冻结配置、归一化与完整参数形状。
按 run 根目录名（checkpoint 的父目录名）查 RUNS 表，决定允许的 step 集合、motion 开关、config 名与两个 sha256 的期望值；
不在表里的 run 直接拒绝——评估只认这张表里登记过的权重。"""
import dataclasses
import hashlib
from pathlib import Path
import sys

from flax import nnx
from flax.traverse_util import flatten_dict
import jax
import orbax.checkpoint as ocp
from mme_vla_suite.training.config import get_config
from mme_vla_suite.policies.policy_config import _load_resolved_snapshot
import mme_vla_suite.policies.policy_config as policy_source
import openpi.models.model as model_source

repo = Path(__file__).resolve().parents[2]
assert Path(policy_source.__file__).resolve() == repo / 'src/mme_vla_suite/policies/policy_config.py'
assert Path(model_source.__file__).resolve() == repo / 'src/openpi/models/model.py'
print('SERVER_SOURCE', sys.executable, policy_source.__file__, model_source.__file__, flush=True)

# 登记过的权重：键 = run 根目录名（download.sh 的落点目录名）
RUNS = {
    # bucket HongzeFu/robomme-vla-modul-60k-v1，run v2-1600ep-m8x8-modul-b128-60k，motion 关闭，61 叶；
    # 59999 = primary700-gl 基线，50000 = 与 motion 50000 同步数对照（2026-09-18 加入；norm_stats 两步同一份 sha）
    'robomme-vla-modul-60k-v1': dict(
        steps=('59999', '50000'), motion=False, config='mme_vla_suite',
        resolved_sha256='804b25382af668d67c8f8ea2d1cca414aee9a184ec737dcad704bdff253a2f92',
        norm_stats_sha256='856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173',
    ),
    # bucket HongzeFu/robomme-vla-modul-motion-80k-v1，run v2-1600ep-m8x8-modul-motion-b128-80k，motion 开启，65 叶；
    # resolved sha 与 bucket 根 history_config.resolved.sha256 及 motion_provenance.json.resolved_sha256 三方一致（2026-09-18 核）；
    # norm_stats 与 60k 同一份（开 motion 不重算归一化）
    'robomme-vla-modul-motion-80k-v1': dict(
        steps=('50000',), motion=True, config='mme_vla_suite_b128_80k',
        resolved_sha256='9650f225fe42b802262f93b71139cebbda934dc5ccfaa849fa4dadaf30d73110',
        norm_stats_sha256='856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173',
        motion_budget=160,
        motion_source_run='wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt#encoder',
    ),
}

checkpoint = Path(sys.argv[1]).resolve()
run = RUNS.get(checkpoint.parent.name)
assert run is not None, f'未登记的 run 根目录 {checkpoint.parent.name}，已登记: {sorted(RUNS)}'
assert checkpoint.name in run['steps'], (checkpoint, run['steps'])
cfg, motion_enabled = _load_resolved_snapshot(checkpoint.parent)
assert motion_enabled == run['motion'], (motion_enabled, run['motion'])
assert cfg.budget == 512 and cfg.token_per_image == 64 and cfg.integration_type == 'modulation'
assert cfg.perceptual_memory.type == 'frame_sampling'
assert hashlib.sha256((checkpoint.parent / 'history_config.resolved.yaml').read_bytes()).hexdigest() == run['resolved_sha256']
assert hashlib.sha256((checkpoint / 'assets/robomme/norm_stats.json').read_bytes()).hexdigest() == run['norm_stats_sha256']
if run['motion']:
    # budget 是模型语义参数（决定 mem_len=512+budget 与 RoPE 位置号），参数树形状对它失明，必须显式钉死
    assert int(cfg.motion.budget) == run['motion_budget'], cfg.motion.budget
    assert str(cfg.motion.source_run) == run['motion_source_run'], cfg.motion.source_run
    assert int(cfg.motion.stride) == 16 and int(cfg.motion.window_frames) == 33 and int(cfg.motion.frame_size) == 256
    print(f'BUDGET_CONSISTENT=PASS budget={int(cfg.motion.budget)} source_run={cfg.motion.source_run}', flush=True)
model_config = dataclasses.replace(get_config(run['config']).model, history_config=cfg, use_history=True)
model = nnx.eval_shape(model_config.create, jax.random.key(0))
expected = flatten_dict(nnx.state(model, nnx.Param).to_pure_dict())
with ocp.PyTreeCheckpointer() as reader:
    metadata = reader.metadata(checkpoint / 'params')
assert set(metadata.tree) == {'params'}
actual = flatten_dict(metadata.tree['params'])
if all(key[-1] == 'value' for key in actual):
    actual = {key[:-1]: value for key, value in actual.items()}
assert set(actual) == set(expected), (set(expected) - set(actual), set(actual) - set(expected))
for key, value in expected.items():
    assert value.shape == actual[key].shape, (key, value.shape, actual[key].shape)
motion_leaves = sorted('/'.join(map(str, k)) for k in actual if 'motion_pos_proj' in '/'.join(map(str, k)) or 'motion_encoder_static' in '/'.join(map(str, k)))
assert bool(motion_leaves) == run['motion'], motion_leaves
print(f'PARAM_TREE_EXACT=PASS n_model={len(expected)} n_ckpt={len(actual)} missing=0 extra=0 shape_mismatch=0 motion_leaves={len(motion_leaves)}', flush=True)
print(f"HISTORY_CONFIG=PASS NORM_STATS=PASS {'MOTION_ENABLED=PASS' if run['motion'] else 'MOTION_DISABLED=PASS'} config={run['config']} step={checkpoint.name}", flush=True)
