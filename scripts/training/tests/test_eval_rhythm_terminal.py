"""用真实 EpisodeState/pack_buffer 控制流检查终止边界，不加载模型。"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import eval_rhythm_gates as gates


class RecordingPolicy:
    def __init__(self):
        self.step_idx = -1

    def add_buffer(self, batch):
        assert len(batch["images"]) == len(batch["state"])
        self.step_idx += len(batch["images"])


class RecordingMemory:
    def add_buffer(self, images, states, indices, *, exec_start_idx):
        assert len(images) == len(states) == len(indices)


@pytest.mark.parametrize("steps,calls", [
    (15,1),(16,1),(17,2),(31,2),(32,2),(33,3),
    (1295,81),(1296,81),(1297,82),(1300,82),(1301,82),(1400,82),
])
def test_terminal_and_count_guard_keep_identical_infer_points(steps, calls):
    es, limit = 17, 1300
    total = es + steps + 1
    actual = []
    info = gates.eval_drive(RecordingPolicy(), es, total, actual.append, max_steps=limit)
    reference = []
    gates.G._drive(RecordingMemory(), es, gates.reference_frame_count(es,total,limit), reference.append)
    assert actual == reference == gates.expected_taus(es,total,limit)
    assert len(actual) == calls
    assert actual[-1] < es + min(steps,limit+1)
    assert info["termination"] == ("env_done" if steps<=limit else "count_guard")


@pytest.mark.parametrize("steps", [16,32])
def test_uncut_reference_exposes_the_old_extra_infer(steps):
    es = 17
    actual, uncut = [], []
    gates.eval_drive(RecordingPolicy(), es, es+steps+1, actual.append)
    gates.G._drive(RecordingMemory(), es, es+steps+1, uncut.append)
    assert uncut[:-1] == actual and uncut[-1] == es+steps
