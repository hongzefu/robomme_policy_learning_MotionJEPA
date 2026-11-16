"""FIX-R：读数兼容常驻服务（resident_server）的 attempts 嵌套目录。

普通模式  <run_root>/attempts/<NNN-chunk-nonce>/client/traces/<nonce>/...
常驻模式  <run_root>/attempts/resident-service-<id>/<NNN-chunk>/client/traces/<nonce>/...
两种布局下各读数函数应找到同一份轨迹、回执与尝试计数（不开仿真、不用 GPU）。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.interpret import first_bin_windows as FBW
from scripts.interpret import fp_swap_prep as FP
from scripts.interpret import rollout_offline as RO
from scripts.interpret import sp_swap_prep as S
from scripts.interpret.analysis import first_bin_attention as FBA
from scripts.interpret.analysis import pl_swap_readout as PLR
from scripts.interpret.analysis import sp_swap_readout as SPR

KEY, NONCE = "RouteStick:0", "abcdef0123456789"


def _make(root: Path, resident: bool) -> Path:
    run_root = root / "run"
    attempt = (run_root / "attempts" / "resident-service-72538f5f" / "000-5ad0771fa979" if resident
               else run_root / "attempts" / "000-5ad0771fa979-abcdef01")
    (attempt.parent).mkdir(parents=True, exist_ok=True)
    if resident:
        (attempt.parent / "server-command.json").write_text("{}")
        (attempt.parent / "server.log").write_text("")
    tdir = attempt / "client" / "traces" / NONCE
    (tdir / "meta").mkdir(parents=True)
    (attempt / "commands.json").write_text(json.dumps({"episode_keys": [KEY]}))
    (attempt / "client" / f"{KEY}.started.json").write_text(json.dumps({"episode_key": KEY, "episode_nonce": NONCE}))
    row = {"event": "step", "episode_key": KEY, "infer_seq": 0, "row_in_chunk": 0, "tcp_after": [0.1, 0.2, 0.3]}
    (tdir / "step_trace.jsonl").write_text(json.dumps(row) + "\n")
    (tdir / "meta" / "hold_receipt.json").write_text(json.dumps({"prompt": "route the stick", "m": 1}))
    (run_root / "accepted").mkdir(parents=True)
    (run_root / "accepted" / f"{KEY}.{NONCE[:8]}.terminal.json").write_text(
        json.dumps({"episode_key": KEY, "episode_nonce": NONCE, "outcome": "success"}))
    return run_root


@pytest.fixture
def layouts(tmp_path):
    return _make(tmp_path / "normal", False), _make(tmp_path / "resident", True)


def _rel(path, run_root):
    return Path(path).relative_to(run_root).parts[-4:]   # client/traces/<nonce>/... 尾部


def test_find_run_both_layouts(layouts, tmp_path):
    for mod in (S, FP):
        got = [mod.find_run(r, tmp_path / "attn", KEY) for r in layouts]
        for r, out in zip(layouts, got):
            assert out["trace"] is not None and Path(out["trace"]).is_file()
            assert out["receipt"] == {"prompt": "route the stick", "m": 1}
        assert _rel(got[0]["trace"], layouts[0]) == _rel(got[1]["trace"], layouts[1])


def test_trace_helpers_both_layouts(layouts):
    for r in layouts:
        assert FBW.trace_path(r, NONCE).name == "step_trace.jsonl"
        assert FBA.receipt_prompt(r, NONCE) == "route the stick"
        assert RO.load_trace(r, NONCE)


def test_attempts_for_both_layouts(layouts):
    for r in layouts:
        assert PLR.attempts_for(r, KEY) == 1
        assert SPR.attempts_for(r, KEY) == 1


def test_resident_service_root_files_not_counted(layouts):
    # 常驻服务根下只有 server-command.json / server.log，不应被当成尝试目录。
    resident = layouts[1]
    assert not list(resident.glob("attempts/*/commands.json"))
    assert PLR.attempts_for(resident, KEY) == 1
