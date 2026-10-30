"""split/concat输入对拍工具：用小夹具库走真实FrameSampDataset，正例零失配、注入故障即记失配。"""
import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from mme_vla_suite.datastore.motion_adapter import STAGE2_SPLIT_READOUT

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/training"))
from test_framesamp_motion_stage2 import dataset, stage2_fx  # noqa: E402, F401

spec = importlib.util.spec_from_file_location("check_nosplit_input", ROOT / "scripts/training/check_nosplit_input.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


@pytest.fixture
def pair(dataset):
    concat = dataset.create(copy.deepcopy(dataset.config))
    cfg = copy.deepcopy(dataset.config)
    cfg.motion.update(readout=STAGE2_SPLIT_READOUT, num_tokens=4, dim=256, budget=384)
    split = dataset.create(cfg)
    yield split, concat, dataset.fixture.fx.manifest
    split.close()
    concat.close()


def test_pick_samples_covers_four_kinds(pair):
    _, concat, manifest = pair
    samples = tool.pick_samples(manifest, concat)
    assert {kind for kind, _ in samples} <= {"demo", "exec", "short_tail", "max_window"}
    assert samples[0][0] == "demo" and len({idx for _, idx in samples}) == len(samples)
    assert all(0 <= idx < len(concat) for _, idx in samples)


def test_split_and_concat_inputs_match(pair):
    split, concat, manifest = pair
    samples = tool.pick_samples(manifest, concat)
    lines = []
    windows, mismatches = tool.compare(split, concat, samples, out=lines.append)
    assert mismatches == 0, lines
    assert windows > 0 and all(line.endswith("ok=1") for line in lines)
    for _, idx in samples:
        assert split[idx]["motion_emb"].shape == (384, 256) and concat[idx]["motion_emb"].shape == (96, 1024)


@pytest.mark.parametrize("fault,expect", [("motion_bits", "motion_bits"), ("mask", "motion_mask"),
                                          ("timecode", "motion_timecode"), ("frame", "frame_or_data:static_image_emb"),
                                          ("order", "mem_order")])
def test_injected_faults_are_counted(pair, monkeypatch, fault, expect):
    split, concat, manifest = pair
    samples = tool.pick_samples(manifest, concat)
    original = type(split).__getitem__

    def corrupt(self, idx):
        data = original(self, idx)
        if fault == "motion_bits":
            data["motion_emb"] = data["motion_emb"].copy(); data["motion_emb"].view(np.uint32)[0, 0] ^= 1
        elif fault == "mask":
            data["motion_mask"] = data["motion_mask"].copy(); data["motion_mask"][-1] = True
        elif fault == "timecode":
            data["motion_pos"] = data["motion_pos"].copy(); data["motion_pos"][-1, 0] += 1
        elif fault == "frame":
            data["static_image_emb"] = data["static_image_emb"].copy(); data["static_image_emb"][0, 0] += 1
        else:
            data["mem_order"] = data["mem_order"][::-1].copy()
        return data
    monkeypatch.setattr(type(split), "__getitem__", lambda self, idx: corrupt(self, idx) if self is split else original(self, idx))
    lines = []
    _, mismatches = tool.compare(split, concat, samples, out=lines.append)
    assert mismatches >= len(samples)
    assert all(expect in line for line in lines), lines
