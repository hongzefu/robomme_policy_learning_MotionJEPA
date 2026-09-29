"""identity_manifest 纯函数的 CPU 单测（不需要仿真）：uv run pytest examples/robomme/identity_manifest_test.py"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import identity_manifest as im  # noqa: E402

ORDER = ["BinFill", "StopCube", "PickXtimes"]


def _rows():
    return [
        dict(task="PickXtimes", episode=3, tier="xhard0", seed=11, candidate=None, source_episode=7, round=1, shard=0),
        dict(task="BinFill", episode=40, tier="xhard2", seed=22, candidate=5, source_episode=None, round=1, shard=0),
        dict(task="BinFill", episode=2, tier="xhard0", seed=33, candidate=None, source_episode=3, round=1, shard=0),
        dict(task="StopCube", episode=20, tier="xhard4", seed=44, candidate=1, source_episode=None, round=1, shard=0),
        dict(task="BinFill", episode=5, tier="xhard0", seed=55, candidate=None, source_episode=15, round=1, shard=1),
        dict(task="BinFill", episode=9, tier="xhard0", seed=66, candidate=None, source_episode=31, round=2, shard=0),
    ]


def test_select_filters_and_orders(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in _rows()) + "\n")
    rows = im.load_manifest(p)
    assert len(rows) == 6
    sel = im.select_rows(rows, 1, 0, ORDER)
    assert [(r["task"], r["episode"]) for r in sel] == [
        ("BinFill", 2), ("BinFill", 40), ("StopCube", 20), ("PickXtimes", 3)]
    g = im.group_by_task(sel)
    assert list(g) == ["BinFill", "StopCube", "PickXtimes"]
    assert [r["episode"] for r in g["BinFill"]] == [2, 40]
    assert [(r["task"], r["episode"]) for r in im.select_rows(rows, 2, 0, ORDER)] == [("BinFill", 9)]
    assert im.select_rows(rows, 2, 5, ORDER) == []
    # 不给 task_order 时按任务名排序
    assert [r["task"] for r in im.select_rows(rows, 1, 0)] == ["BinFill", "BinFill", "PickXtimes", "StopCube"]


def test_select_rejects_duplicates_and_unknown_tasks():
    rows = _rows()
    with pytest.raises(ValueError):
        im.select_rows(rows + [dict(rows[1])], 1, 0, ORDER)
    with pytest.raises(ValueError):
        im.select_rows(rows + [dict(rows[1], task="Nope")], 1, 0, ORDER)


def test_check_identity():
    row = _rows()[0]
    im.check_identity(row, {"tier": "xhard0", "seed": 11, "candidate": None})
    with pytest.raises(AssertionError):
        im.check_identity(row, {"tier": "xhard0", "seed": 12})
    with pytest.raises(AssertionError):
        im.check_identity(row, {"tier": "xhard1", "seed": 11})


def test_binding_class_and_video_name():
    assert im.binding_class("xhard0") == "export"
    assert all(im.binding_class(f"xhard{i}") == "replay" for i in range(1, 5))
    assert im.video_name("BinFill", "xhard2", 40, 22) == "BinFill_xhard2_40_22.mp4"
