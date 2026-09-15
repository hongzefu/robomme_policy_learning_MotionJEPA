"""守卫轻量验收：使用真实新库，逐键比较原始字节。"""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.training.config import get_config
from mme_vla_suite.training.dataloader import _create_framesamp_dataset

start = time.monotonic()
root = Path.cwd()
dsroot = root / "v1-store/datasets/4task-v2-1600ep-604f16da"
os.environ["MMEVLA_FRAMESAMP_SOURCE"] = str(dsroot / "source")
os.environ["MMEVLA_FRAMESAMP_MANIFEST"] = str(dsroot / "meta/episode_manifest.json")
config = get_config("mme_vla_suite_b128_60k")
dc = config.data.create(config.assets_dirs, config.model)

def dataset(name, layout="framesamp-8x8", mismatch=False):
    hc = get_history_config(name)
    if mismatch:
        hc = OmegaConf.merge(hc, {"memory_token_dim": 2048})
    return _create_framesamp_dataset(str(dsroot / layout), dc, hc, 20)

mod = dataset("perceptual-framesamp-modul-8frame-8x8.yaml")
print("GUARD_MODUL_8X8=PASS", flush=True)
plain = dataset("perceptual-framesamp-modul.yaml", "framesamp")
plain.close()
print("GUARD_MODUL_4X4=PASS", flush=True)
for name, mismatch, label in [
    ("perceptual-framesamp-expert.yaml", False, "EXPERT"),
    ("perceptual-framesamp-modul-8frame-8x8.yaml", True, "MISMATCH"),
]:
    try:
        dataset(name, mismatch=mismatch)
    except ValueError as e:
        assert "形制断言失败" in str(e) and "integration_type, memory_token_dim" in str(e), str(e)
        print(f"GUARD_{label}_REJECT=PASS reason={e}", flush=True)
    else:
        raise AssertionError(f"{label} 未被拒绝")
context = dataset("perceptual-framesamp-context-8frame-8x8.yaml")
manifest = json.loads((dsroot / "meta/episode_manifest.json").read_text())
assert len(mod) == len(context) == 605611
indices = set(map(int, np.random.default_rng(20260915).choice(605611, 256)))
indices.update((0, 605610))
for ep in manifest["episodes"]:
    offset = ep["exec_sample_offset"]
    indices.update(i for i in (offset - 1, offset, offset + ep["exec_samples"] - 1) if 0 <= i < len(mod))

def summary(value):
    if isinstance(value, dict):
        return {k: summary(v) for k, v in sorted(value.items())}
    if isinstance(value, np.ndarray):
        assert not value.dtype.hasobject
        return {"dtype": str(value.dtype), "shape": list(value.shape), "bytes": value.nbytes,
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [summary(v) for v in value]]
    if isinstance(value, np.generic):
        return summary(np.asarray(value))
    return {"type": type(value).__name__, "value": value}

aggregate = hashlib.sha256()
example = None
try:
    for n, idx in enumerate(sorted(indices), 1):
        left, right = summary(mod[idx]), summary(context[idx])
        assert left == right, f"样本 {idx} 内容不等"
        aggregate.update(json.dumps([idx, left], sort_keys=True).encode())
        if example is None:
            example = left
        if n % 500 == 0:
            print(f"DS_PROGRESS samples={n}/{len(indices)} seconds={time.monotonic()-start:.1f}", flush=True)
finally:
    mod.close()
    context.close()
print("DS_SAMPLE_META=" + json.dumps(example, ensure_ascii=False, sort_keys=True), flush=True)
print(f"DS_EQUIV=PASS samples={len(indices)} keys={len(example)} mismatches=0 sha256={aggregate.hexdigest()} seconds={time.monotonic()-start:.3f}", flush=True)
