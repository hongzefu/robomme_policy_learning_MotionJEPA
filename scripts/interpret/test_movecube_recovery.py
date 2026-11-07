"""movecube_recovery 的合成数据测试：统计已知值、四条判定行的 PASS 与 FAIL、CLI 全产物。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.interpret.analysis import movecube_recovery as mr


ORIGIN = 10
EPISODES = {"MoveCube:1": "A", "MoveCube:2": "A", "MoveCube:3": "B"}
KS = (0, 48)
CUBE, PEG = [0.1, 0.0, 0.02], [0.4, 0.0, 0.02]


def table_actions(index):
    actions = np.zeros((80, 8))
    actions[:, 0] = np.arange(80) * 0.01 + index
    actions[:, 7] = 1.0
    return actions


def write_table(root):
    path = Path(root) / "force-table.json"
    episodes = {key: {"actions": table_actions(i).tolist(), "first_env_step": ORIGIN,
                      "env_seed": 7, "way": "grasp_putdown" if direction == "A" else "peg_push", "direction": direction}
                for i, (key, direction) in enumerate(EPISODES.items())}
    path.write_text(json.dumps({"schema_version": 1, "force_steps": 80, "episodes": episodes}))
    return path


def write_run(root, model, k, key, *, grasp=1, outcome="success", steps=None, reached=None, release=None, tamper=None,
              relative_trace=False):
    """写一局回执与轨迹。tamper(rows) 可在落盘前篡改轨迹行。"""
    index = list(EPISODES).index(key)
    direction = EPISODES[key]
    steps = k + 20 if steps is None else steps
    reached = steps >= k if reached is None else reached
    rows = []
    for i in range(steps):
        forced = i < k
        action = table_actions(index)[i] if forced else np.full(8, 9.0)
        # 策略段 TCP 线性走向正确物体，末步正好水平距离 0.05；强推段停在强推目标旁。
        correct, wrong = (CUBE, PEG) if direction == "A" else (PEG, CUBE)
        tcp = [wrong[0] + 0.01, 0.0, 0.2] if forced else [correct[0] + 0.05 + (steps - 1 - i) * 0.01, 0.0, 0.2]
        # 与 S1 runner 同形：env_step 是步前值，env_step_after 与位置字段是步后值。
        rows.append({"event": "step", "episode_key": key, "action_index": i, "env_step": ORIGIN + i, "env_step_after": ORIGIN + i + 1,
                     "infer_seq": i // 16, "row_in_chunk": i % 16, "chunk_start_env_step": ORIGIN + i - i % 16,
                     "control_source": "force" if forced else "policy", "command_action": np.full(8, 9.0).tolist(),
                     "executed_action": action.tolist(), "env_status": "ongoing", "tcp": tcp, "cube": CUBE, "peg_tail": PEG,
                     "peg_head": PEG})
    if tamper:
        tamper(rows)
    shard = Path(root) / model / f"k{k}" / "s0"
    attempt = shard / "attempts" / "force-service-0" / "0"
    nonce = attempt / "client" / "traces" / f"{key.replace(':', '_')}"
    (nonce / "meta").mkdir(parents=True)
    trace = nonce / "step_trace.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = {"episode_key": key, "episode_id": int(key.split(":")[1]), "grasp_success": grasp,
               "grasp_step": ORIGIN + steps if grasp else None, "grasped_object": "cube" if grasp else None, "outcome": outcome,
               "env_status": "ongoing", "force_k": k,
               "way": "grasp_putdown" if direction == "A" else "peg_push", "direction": direction, "origin_env_step": ORIGIN,
               "reached_release": reached, "actual_release_step": (ORIGIN + k if reached else None) if release is None else release,
               "actions_executed": steps, "trace_path": "../step_trace.jsonl" if relative_trace else str(trace)}
    (nonce / "meta" / "force_receipt.json").write_text(json.dumps(receipt))
    server = {"server": ["uv", "run", "python", "-m", "scripts.training.serve_policy", "--port", "1"]}
    (attempt / "commands.json").write_text(json.dumps(server))
    (shard / "attempts" / "force-service-0" / "server-command.json").write_text(json.dumps(server))


def write_all(root, overrides=None):
    """全部格子都写成抓取成功；overrides 以 (model, k, key) 为键给 write_run 传参。"""
    for model in mr.MODELS:
        for k in KS:
            for key in EPISODES:
                write_run(root, model, k, key, **(overrides or {}).get((model, k, key), {}))
    return write_table(root)


def run(root, table, episodes=3):
    return mr.analyze(root, table, episodes, KS)


def line(report, name):
    return next(text for text in mr.verdict_lines(report) if text.startswith(name + "="))


def group(report, direction, model, k):
    return next(g for g in report["groups"] if (g["direction"], g["model"], g["k"]) == (direction, model, k))


def test_classify():
    ok = {"way": "grasp_putdown", "direction": "A", "origin_env_step": ORIGIN}
    assert mr.classify({**ok, "grasp_success": 1, "outcome": "success"}) == "grasp"
    assert mr.classify({**ok, "grasp_success": 0, "outcome": "fail"}) == "fail"
    assert mr.classify({**ok, "grasp_success": 0, "outcome": "timeout"}) == "timeout"
    assert mr.classify({**ok, "grasp_success": 0, "outcome": None}) == "error"
    assert mr.classify({**ok, "grasp_success": 1, "outcome": None}) == "error"
    # runner 的 success 只应伴随 grasp_success=1；单独出现是契约破坏，不能读成成功。
    assert mr.classify({**ok, "grasp_success": 0, "outcome": "success"}) == "error"
    for name in ok:
        assert mr.classify({**ok, name: None, "grasp_success": 0, "outcome": "fail"}) == "error"


def test_wilson_known_values():
    assert mr.wilson(8, 10) == pytest.approx((0.4902, 0.9433), abs=1e-4)
    assert mr.wilson(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert mr.wilson(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)
    assert mr.wilson(0, 0) == (None, None)


def test_mcnemar_known_values():
    assert mr.mcnemar_exact(0, 0) == 1.0
    assert mr.mcnemar_exact(5, 5) == 1.0
    assert mr.mcnemar_exact(6, 0) == pytest.approx(2 / 64)
    assert mr.mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert mr.mcnemar_exact(8, 2) == pytest.approx(0.109375)


def test_all_pass_and_statistics(tmp_path):
    table = write_all(tmp_path, {
        ("ctx", 48, "MoveCube:1"): {"grasp": 0, "outcome": "timeout"},
        ("ctx", 48, "MoveCube:2"): {"grasp": 0, "outcome": "fail"},
        ("ctx", 0, "MoveCube:2"): {"grasp": 0, "outcome": "fail"},
    })
    report = run(tmp_path, table)
    assert line(report, "FORCE_EXEC") == "FORCE_EXEC=PASS runs=12 mismatches=0"
    assert line(report, "FORCE_RELEASE") == "FORCE_RELEASE=PASS reached_release=12 mismatches=0"
    assert line(report, "NO_ATTN") == "NO_ATTN=PASS npz=0"
    assert line(report, "MFR_RUN") == "MFR_RUN=PASS completed=12 missing=0 errors=0"

    ctx48 = group(report, "A", "ctx", 48)
    assert (ctx48["grasp_success"], ctx48["fail"], ctx48["timeout"], ctx48["total"]) == (0, 1, 1, 2)
    # k=0 只有 MoveCube:1 抓起，恢复率分母只算这一局。
    assert (ctx48["recovery_grasp"], ctx48["recovery_baseline_total"]) == (0, 1)
    mot48 = group(report, "A", "mot", 48)
    assert (mot48["grasp_rate"], mot48["recovery_baseline_total"]) == (1.0, 2)
    assert (mot48["wilson_low"], mot48["wilson_high"]) == pytest.approx(mr.wilson(2, 2))
    # 放手到抓起：grasp_step 是步后 elapsed（origin+68），actual_release_step 是步前（origin+48），差 20。
    assert mot48["median_release_to_grasp"] == 20
    # 放手后 TCP 末步距 cube 0.05；到 peg_tail 最近在放手第一步：x=0.1+0.05+0.19=0.34，距 0.4 为 0.06。
    assert mot48["median_min_dist_correct"] == pytest.approx(0.05)
    assert mot48["median_min_dist_forced"] == pytest.approx(0.06)
    # 方向 B 的正确物体是 peg_tail，距离要跟着换。
    assert group(report, "B", "mot", 48)["median_min_dist_correct"] == pytest.approx(0.05)

    pair = next(p for p in report["mcnemar"] if (p["direction"], p["k"]) == ("A", 48))
    assert (pair["pairs"], pair["only_mot"], pair["only_ctx"], pair["both"]) == (2, 2, 0, 0)
    assert pair["mcnemar_p"] == pytest.approx(0.5)
    k0 = next(p for p in report["mcnemar"] if (p["direction"], p["k"]) == ("A", 0))
    assert (k0["pairs"], k0["only_mot"]) == (2, 1)
    assert "MFR_RESULT direction=A model=ctx k=48 grasp_success=0/2" in mr.verdict_lines(report)


def test_relative_trace_path(tmp_path):
    table = write_all(tmp_path, {("mot", 48, "MoveCube:1"): {"relative_trace": True}})
    assert "PASS" in line(run(tmp_path, table), "FORCE_EXEC")


def test_force_exec_fail_on_one_wrong_action(tmp_path):
    def tamper(rows):
        rows[7]["executed_action"][0] += 0.001
    table = write_all(tmp_path, {("mot", 48, "MoveCube:1"): {"tamper": tamper}})
    report = run(tmp_path, table)
    # 表不符 1 处，两模型序列不一致再记 1 处。
    assert line(report, "FORCE_EXEC") == "FORCE_EXEC=FAIL runs=12 mismatches=2"
    assert "PASS" in line(report, "MFR_RUN")


def test_force_exec_within_tolerance(tmp_path):
    def tamper(rows):
        rows[7]["executed_action"][0] += 5e-7
    table = write_all(tmp_path, {("mot", 48, "MoveCube:1"): {"tamper": tamper}})
    assert "PASS" in line(run(tmp_path, table), "FORCE_EXEC")


def test_force_exec_fail_when_k0_has_forced(tmp_path):
    def tamper(rows):
        rows[0]["control_source"] = "force"
    table = write_all(tmp_path, {("mot", 0, "MoveCube:1"): {"tamper": tamper}})
    assert "FORCE_EXEC=FAIL" in line(run(tmp_path, table), "FORCE_EXEC")


def test_force_exec_fail_on_short_forcing(tmp_path):
    """强推少一步（第 47 步已被策略接管）既不满足 min(k, 实际步数)，也让两模型序列不一致。"""
    def tamper(rows):
        rows[47]["control_source"] = "policy"
    table = write_all(tmp_path, {("mot", 48, "MoveCube:1"): {"tamper": tamper}})
    report = run(tmp_path, table)
    assert "FORCE_EXEC=FAIL" in line(report, "FORCE_EXEC")


def test_force_exec_fail_when_models_differ_in_forced_length(tmp_path):
    """一个模型在强推途中结束：单局都满足 min(k, 实际步数)，但同局同 k 两模型序列不一致。"""
    table = write_all(tmp_path, {("ctx", 48, "MoveCube:1"): {"steps": 20, "reached": False, "grasp": 0, "outcome": "fail"}})
    assert line(run(tmp_path, table), "FORCE_EXEC") == "FORCE_EXEC=FAIL runs=12 mismatches=1"


def test_force_release_fail_on_wrong_release_step(tmp_path):
    table = write_all(tmp_path, {("mot", 48, "MoveCube:1"): {"release": ORIGIN + 49}})
    report = run(tmp_path, table)
    assert line(report, "FORCE_RELEASE") == "FORCE_RELEASE=FAIL reached_release=12 mismatches=1"
    assert "PASS" in line(report, "FORCE_EXEC")


def test_force_release_applies_to_k0(tmp_path):
    table = write_all(tmp_path, {("ctx", 0, "MoveCube:3"): {"release": ORIGIN + 1}})
    assert "FORCE_RELEASE=FAIL" in line(run(tmp_path, table), "FORCE_RELEASE")


def test_force_release_fail_on_forced_after_release(tmp_path):
    def tamper(rows):
        rows[50]["control_source"] = "force"
    table = write_all(tmp_path, {("mot", 48, "MoveCube:1"): {"tamper": tamper}})
    assert "FORCE_RELEASE=FAIL" in line(run(tmp_path, table), "FORCE_RELEASE")


def test_unreached_release_counts_in_denominator_only(tmp_path):
    """两模型都在强推途中判失败：不进放手核对、不算 mismatch，但进分母记失败，也不产生距离样本。"""
    early = {"steps": 20, "reached": False, "grasp": 0, "outcome": "fail", "release": None}
    table = write_all(tmp_path, {("mot", 48, "MoveCube:1"): early, ("ctx", 48, "MoveCube:1"): early})
    report = run(tmp_path, table)
    assert line(report, "FORCE_RELEASE") == "FORCE_RELEASE=PASS reached_release=10 mismatches=0"
    assert line(report, "FORCE_EXEC") == "FORCE_EXEC=PASS runs=12 mismatches=0"
    assert line(report, "MFR_RUN") == "MFR_RUN=PASS completed=12 missing=0 errors=0"
    mot48 = group(report, "A", "mot", 48)
    assert (mot48["grasp_success"], mot48["fail"], mot48["total"], mot48["n_dist"]) == (1, 1, 2, 1)


def test_no_attn_fail_on_npz(tmp_path):
    table = write_all(tmp_path)
    (tmp_path / "mot" / "k0" / "s0" / "x.npz").write_bytes(b"")
    assert line(run(tmp_path, table), "NO_ATTN") == "NO_ATTN=FAIL npz=1"


def test_no_attn_fail_on_attn_dir(tmp_path):
    table = write_all(tmp_path)
    (tmp_path / "ctx" / "k48" / "s0" / "attn").mkdir()
    assert line(run(tmp_path, table), "NO_ATTN") == "NO_ATTN=FAIL npz=0"


@pytest.mark.parametrize("server", [
    ["python", "-m", "scripts.interpret.serve_policy_attn"],
    ["python", "-m", "scripts.training.serve_policy", "--attn-out", "x"],
    ["python", "-m", "other.server"],
])
def test_no_attn_fail_on_server_command(tmp_path, server):
    table = write_all(tmp_path)
    next((tmp_path / "mot" / "k0" / "s0").rglob("commands.json")).write_text(json.dumps({"server": server}))
    assert line(run(tmp_path, table), "NO_ATTN") == "NO_ATTN=FAIL npz=0"


def test_no_attn_fail_on_server_command_file(tmp_path):
    table = write_all(tmp_path)
    next((tmp_path / "ctx" / "k48" / "s0").rglob("server-command.json")).write_text(
        json.dumps({"server": ["python", "-m", "scripts.interpret.serve_policy_attn"]}))
    assert line(run(tmp_path, table), "NO_ATTN") == "NO_ATTN=FAIL npz=0"


def test_no_attn_fail_without_commands(tmp_path):
    table = write_all(tmp_path)
    for path in [*tmp_path.rglob("commands.json"), *tmp_path.rglob("server-command.json")]:
        path.unlink()
    assert "NO_ATTN=FAIL" in line(run(tmp_path, table), "NO_ATTN")


def test_mfr_run_fail_on_missing_episode(tmp_path):
    table = write_all(tmp_path)
    for path in (tmp_path / "ctx" / "k48" / "s0").rglob("MoveCube_3/meta/force_receipt.json"):
        path.unlink()
    report = run(tmp_path, table)
    assert line(report, "MFR_RUN") == "MFR_RUN=FAIL completed=11 missing=1 errors=0"
    assert report["verdicts"]["MFR_RUN"]["missing_cells"] == [["ctx", 48, "MoveCube:3"]]


def test_mfr_run_error_outcome_excluded_from_denominator(tmp_path):
    table = write_all(tmp_path, {("mot", 48, "MoveCube:3"): {"grasp": 0, "outcome": None}})
    report = run(tmp_path, table)
    assert line(report, "MFR_RUN") == "MFR_RUN=FAIL completed=11 missing=0 errors=1"
    b48 = group(report, "B", "mot", 48)
    assert (b48["total"], b48["errors"]) == (0, 1)
    assert "MFR_RESULT direction=B model=mot k=48 grasp_success=0/0" in mr.verdict_lines(report)


def test_identity_failure_counts_as_error_without_exec_mismatch(tmp_path):
    """身份核对失败：way、direction、origin 为 null，轨迹为空。记 errors，不报异常、不算动作 mismatch。"""
    table = write_all(tmp_path)
    path = next((tmp_path / "ctx" / "k48" / "s0").rglob("MoveCube_1/meta/force_receipt.json"))
    receipt = json.loads(path.read_text())
    receipt.update(way=None, direction=None, origin_env_step=None, reached_release=False, actual_release_step=None,
                   grasp_success=0, grasp_step=None, grasped_object=None, outcome=None)
    path.write_text(json.dumps(receipt))
    Path(receipt["trace_path"]).unlink()
    report = run(tmp_path, table)
    assert line(report, "MFR_RUN") == "MFR_RUN=FAIL completed=11 missing=0 errors=1"
    assert line(report, "FORCE_EXEC") == "FORCE_EXEC=PASS runs=12 mismatches=0"
    assert line(report, "FORCE_RELEASE") == "FORCE_RELEASE=PASS reached_release=11 mismatches=0"
    a48 = group(report, "A", "ctx", 48)
    assert (a48["total"], a48["errors"]) == (1, 1)


def test_mfr_run_fail_on_duplicate_receipt(tmp_path):
    table = write_all(tmp_path)
    extra = tmp_path / "mot" / "k0" / "s1" / "client" / "traces" / "dup" / "meta"
    extra.mkdir(parents=True)
    source = next((tmp_path / "mot" / "k0" / "s0").rglob("force_receipt.json"))
    (extra / "force_receipt.json").write_text(source.read_text())
    assert line(run(tmp_path, table), "MFR_RUN") == "MFR_RUN=FAIL completed=12 missing=0 errors=1"


def test_smoke_scale_uses_observed_episodes(tmp_path):
    """冒烟：表里 3 局只跑其中 2 局，规模按 --expected-episodes 算，不把其余局记缺失。"""
    table = write_table(tmp_path)
    for model in mr.MODELS:
        for k in KS:
            for key in ("MoveCube:1", "MoveCube:3"):
                write_run(tmp_path, model, k, key)
    report = run(tmp_path, table, episodes=2)
    assert line(report, "MFR_RUN") == "MFR_RUN=PASS completed=8 missing=0 errors=0"
    assert report["episodes"] == ["MoveCube:1", "MoveCube:3"]


def test_smoke_shortfall_is_missing(tmp_path):
    """期望 2 局却只见 1 局时，差额进 missing。"""
    table = write_table(tmp_path)
    for model in mr.MODELS:
        for k in KS:
            write_run(tmp_path, model, k, "MoveCube:1")
    assert line(run(tmp_path, table, episodes=2), "MFR_RUN") == "MFR_RUN=FAIL completed=4 missing=4 errors=0"


def test_cli_writes_all_outputs(tmp_path, capsys):
    root = tmp_path / "root"
    root.mkdir()
    table = write_all(root, {("ctx", 48, "MoveCube:1"): {"grasp": 0, "outcome": "timeout"}})
    out = tmp_path / "out"
    code = mr.main(["--root", str(root), "--force-table", str(table), "--out", str(out), "--expected-episodes", "3", "--expected-k", "0,48"])
    assert code == 0
    for name in ("summary.json", "summary.txt", "per_episode.csv", "recovery_curve_A.png", "recovery_curve_A.pdf",
                 "recovery_curve_B.png", "recovery_curve_B.pdf"):
        assert (out / name).stat().st_size > 0, name
    printed = capsys.readouterr().out
    text = (out / "summary.txt").read_text()
    assert printed == text
    tail = text.strip().splitlines()
    assert tail[-12:-8] == [line for line in tail if line.split("=")[0] in ("FORCE_EXEC", "FORCE_RELEASE", "NO_ATTN", "MFR_RUN")]
    assert tail[-1].startswith("MFR_RESULT ")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdicts"]["MFR_RUN"]["pass"] is True
    assert len((out / "per_episode.csv").read_text().strip().splitlines()) == 13


def test_cli_returns_1_on_fail_but_still_writes(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    table = write_all(root)
    (root / "x.npz").write_bytes(b"")
    out = tmp_path / "out"
    assert mr.main(["--root", str(root), "--force-table", str(table), "--out", str(out), "--expected-episodes", "3", "--expected-k", "0,48"]) == 1
    assert "NO_ATTN=FAIL npz=1" in (out / "summary.txt").read_text()


def test_missing_direction_still_writes_curve(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    table = write_table(root)
    for model in mr.MODELS:
        for k in KS:
            write_run(root, model, k, "MoveCube:1")
    out = tmp_path / "out"
    mr.main(["--root", str(root), "--force-table", str(table), "--out", str(out), "--expected-episodes", "1", "--expected-k", "0,48"])
    assert (out / "recovery_curve_B.png").stat().st_size > 0


def test_missing_font_warns_without_error(tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    root.mkdir()
    table = write_all(root)
    monkeypatch.setattr(mr, "CJK_FAMILIES", ("不存在的字体",))
    out = tmp_path / "out"
    assert mr.main(["--root", str(root), "--force-table", str(table), "--out", str(out), "--expected-episodes", "3", "--expected-k", "0,48"]) == 0
    assert "未找到中文字体" in capsys.readouterr().err
    assert "未找到中文字体" in (out / "summary.txt").read_text()
    assert (out / "recovery_curve_A.png").stat().st_size > 0
