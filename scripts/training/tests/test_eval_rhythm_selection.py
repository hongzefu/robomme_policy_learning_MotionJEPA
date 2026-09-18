"""核对已批准的真实代表名单，以及完整预算扫描不会随回放选择缩减。"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import eval_rhythm_gates as gates

ROOT = pathlib.Path(__file__).resolve().parents[3]
LIB = ROOT / "v1-store/datasets/4task-v2-1600ep-604f16da"
APPROVED_GIDS = {401,1201,400,1405,500,810,895,905,809,877,912,866,808,835,811,862,1505,
                 700,1194,168,179,392,335,311,297,313,399,292,299,268,330,317,273,355,367}


def test_real_selection_matches_approved_list():
    cases = gates.load_real_cases(LIB)
    selected = gates.representative_replay_cases(cases, minimum=17, stride=16)
    assert len(cases) == 411 and len(selected) == 35
    assert {int(c[2].split("-",1)[0][1:]) for c in selected} == APPROVED_GIDS
    assert {c[2].split("-",1)[1] for c in selected} == {"BinFill","RouteStick","VideoRepick","VideoUnmaskSwap"}
    assert {((c[0]-17)%16) for c in selected} == set(range(16))
    assert (selected[0][0], selected[-1][0]) == (100,1152)
    assert gates.representative_replay_cases(list(reversed(cases)), minimum=17, stride=16) == selected


@pytest.mark.parametrize("mode,replays", [("all",411),("representative",35)])
def test_main_keeps_full_budget_scope(monkeypatch, mode, replays):
    seen = {}
    def replay(name):
        def run(cases, rec):
            seen[name] = len(cases)
            return name+"=PASS", True
        return run
    def boundary(real_es, rec):
        seen["budget_lengths"] = len(real_es)
        return "ES_BOUNDARY=PASS", True
    def long(rec, *, max_es):
        seen["longest"] = max_es
        return "TAU_LONG=PASS", True
    monkeypatch.setattr(gates, "gate_rhythm", replay("rhythm"))
    monkeypatch.setattr(gates, "gate_termination", replay("termination"))
    monkeypatch.setattr(gates, "gate_es_boundary", boundary)
    monkeypatch.setattr(gates, "gate_tau_long", long)
    monkeypatch.setattr(sys, "argv", ["eval_rhythm_gates.py","--lib",str(LIB),"--replay-cases",mode,
                                     "--expect-replay-cases",str(replays),"--expect-real-es","411"])
    assert gates.main() == 0
    assert seen == {"rhythm":replays,"termination":replays,"budget_lengths":411,"longest":1152}


@pytest.mark.parametrize("flag,value", [("--expect-replay-cases","34"),("--expect-real-es","410")])
def test_external_coverage_count_rejects_mismatch(monkeypatch, flag, value):
    monkeypatch.setattr(sys, "argv", ["eval_rhythm_gates.py","--lib",str(LIB),
                                     "--replay-cases","representative",flag,value])
    with pytest.raises(ValueError, match="外部期望"):
        gates.main()
