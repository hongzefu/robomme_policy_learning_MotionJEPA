"""xhard0_manifest 纯函数的 CPU 单测（不需要仿真）。"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import xhard0_manifest as xm  # noqa: E402

ORDER = ["BinFill", "StopCube", "PickXtimes"]
ROWS = [
    dict(task="PickXtimes", source_episode=7, seed=70, shard=0),
    dict(task="BinFill", source_episode=47, seed=470, shard=0),
    dict(task="BinFill", source_episode=3, seed=30, shard=0),
    dict(task="StopCube", source_episode=11, seed=110, shard=1),
]


def _write(tmp, rows):
    p = tmp / "m.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_load_rows(tmp_path):
    g = xm.load_rows(_write(tmp_path, ROWS), 0, ORDER)
    assert list(g) == ["BinFill", "PickXtimes"]
    assert [r["source_episode"] for r in g["BinFill"]] == [3, 47]
    assert list(xm.load_rows(_write(tmp_path, ROWS), 1, ORDER)) == ["StopCube"]
    with pytest.raises(ValueError):
        xm.load_rows(_write(tmp_path, ROWS + [ROWS[1]]), 0, ORDER)
    with pytest.raises(ValueError):
        xm.load_rows(_write(tmp_path, ROWS + [dict(ROWS[1], task="Nope")]), 0, ORDER)


def test_check_seed():
    xm.check_seed(ROWS[0], 70, "hard")
    for seed, diff in ((71, "hard"), (70, "easy"), (None, "hard")):
        with pytest.raises(AssertionError):
            xm.check_seed(ROWS[0], seed, diff)


def test_done_keys_last_record_wins(tmp_path):
    p = tmp_path / "episodes.jsonl"
    recs = [dict(task="BinFill", source_episode=3, status="error"), dict(task="BinFill", source_episode=3, status="fail"),
            dict(task="BinFill", source_episode=47, status="success"), dict(task="BinFill", source_episode=47, status="error")]
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    done, rows = xm.done_keys(p)
    assert done == {("BinFill", 3)} and len(rows) == 4
    assert xm.done_keys(tmp_path / "none.jsonl") == (set(), [])


def test_video_name():
    assert xm.video_name("BinFill", 3, 30) == "BinFill_xhard0_3_30.mp4"
