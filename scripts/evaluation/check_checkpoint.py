"""不加载大数组，先核对冻结配置、归一化与完整参数形状。"""
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

checkpoint = Path(sys.argv[1]).resolve()
assert checkpoint.name == '59999', checkpoint
cfg, motion_enabled = _load_resolved_snapshot(checkpoint.parent)
assert not motion_enabled
assert cfg.budget == 512 and cfg.token_per_image == 64 and cfg.integration_type == 'modulation'
assert cfg.perceptual_memory.type == 'frame_sampling'
assert hashlib.sha256((checkpoint.parent / 'history_config.resolved.yaml').read_bytes()).hexdigest() == '804b25382af668d67c8f8ea2d1cca414aee9a184ec737dcad704bdff253a2f92'
assert hashlib.sha256((checkpoint / 'assets/robomme/norm_stats.json').read_bytes()).hexdigest() == '856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173'
model_config = dataclasses.replace(get_config('mme_vla_suite').model, history_config=cfg, use_history=True)
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
print(f'PARAM_TREE_EXACT=PASS n_model={len(expected)} n_ckpt={len(actual)} missing=0 extra=0 shape_mismatch=0', flush=True)
print('HISTORY_CONFIG=PASS NORM_STATS=PASS MOTION_DISABLED=PASS', flush=True)
