"""同一局供体读数（计划 1115 §3、§4.3）的合成数据单测：只用 CPU、不读任何真实数据。"""
from __future__ import annotations

from scripts.interpret.analysis import fp_swap_readout as FR
from scripts.interpret.analysis import sp_swap_readout as SR

# 路线 0→2→4→6（路线格共线），每步绕向 cw / ccw / cw
E = {"task": "RouteStick", "cells": [0, 2, 4, 6], "swings": ["cw", "ccw", "cw"], "theta": 0.0, "m": 3,
     "difficulty": "hard"}
KEY = "RouteStick:7"
POOL = {KEY: E, "RouteStick:8": dict(E), "RouteStick:9": dict(E)}


def st(i, start, reached, completed, swing):
    return {"i": i, "start": start, "reached": reached, "completed": completed, "swing": swing, "how": "jump"}


N_STEPS = [st(1, 0, 2, True, "cw"), st(2, 2, 4, True, "ccw"), st(3, 4, 6, True, "cw")]


def run(group, steps, key=KEY, j=None, seed=42, **kw):
    return {"task": "RouteStick", "key": key, "group": group, "j": j, "seed": seed, "status": "ok", "steps": steps,
            "noise_rep": kw.pop("noise_rep", None), **kw}


# ---------------------------------------------------------------- 绕向两口径与按钮
def test_phys_swing_same_side_equals_label_other_side_flips():
    assert SR.phys_swing(st(2, 2, 4, True, "ccw"), 2, 4) == "ccw"
    assert SR.phys_swing(st(2, 2, 0, False, "ccw"), 2, 4) == "cw"
    assert SR.phys_swing(st(2, 2, None, False, None), 2, 4) is None


def test_step_view_s_swing_label_and_phys_follow():
    # 只翻绕向：去原按钮 4、绕向 cw（原 ccw）
    donor = SR.donor_meta({"donor_side": 1, "donor_swing": "clockwise"}, E, 2)
    assert donor == (4, "cw", "cw")
    steps = N_STEPS[:1] + [st(2, 2, 4, True, "cw")]
    v = SR.samep_step_view(steps, E, 2, donor)
    assert v["swing_follow"] and v["swing_follow_phys"] and v["button_follow"] and not v["own"]
    vn = SR.samep_step_view(N_STEPS, E, 2, donor)
    assert vn["own"] and not vn["swing_follow"] and vn["button_follow"]


def test_step_view_s_side_includes_failed_step_and_phys_arc_fallback():
    # 只翻左右：去另一侧按钮 0、标签绕向不变（ccw）→ 物理弧侧缺省推为 cw；当场失败（completed=False）也读
    donor = SR.donor_meta({"donor_side": -1, "donor_swing": "ccw"}, E, 2)
    assert donor == (0, "ccw", "cw")
    steps = N_STEPS[:1] + [st(2, 2, 0, False, "ccw")]
    v = SR.samep_step_view(steps, E, 2, donor)
    assert v["button_follow"] and v["swing_follow"] and v["swing_follow_phys"] and not v["completed"]
    # 显式给出的物理弧侧优先于缺省推导
    donor2 = SR.donor_meta({"donor_target": 0, "donor_swing": "ccw", "variant_physical_arc": "ccw"}, E, 2)
    assert donor2 == (0, "ccw", "ccw")
    assert not SR.samep_step_view(steps, E, 2, donor2)["swing_follow_phys"]


def test_unreached_counts_as_non_follow_in_denominator_and_n_not_reached_excluded():
    runs = [run("N", N_STEPS),
            run("S_swing", [st(1, 0, 6, False, "cw")], j=2, donor_side=1, donor_swing="cw"),
            run("N", [st(1, 0, 2, False, "cw")], key="RouteStick:8"),
            run("S_swing", N_STEPS, key="RouteStick:8", j=2, donor_side=1, donor_swing="cw")]
    inter, noise, invalid, not_reached = SR.samep_rows(runs, POOL)
    assert len(inter) == 1 and inter[0]["unreached"] and not inter[0]["swing_follow"]
    assert all(inter[0][f] for f in SR.SAMEP_FLIPS)
    assert not_reached == 1 and sum(invalid.values()) == 0 and noise == []


# ---------------------------------------------------------------- 噪声、区间与有效应
def test_noise_flip_rate_by_j():
    runs = [run("N", N_STEPS),
            run("N", N_STEPS[:1] + [st(2, 2, 4, True, "cw")], seed=1),          # j=2 绕向翻转
            run("N", N_STEPS, seed=2),
            run("S_swing", N_STEPS, j=2, donor_side=1, donor_swing="cw"),
            run("Z", N_STEPS, j=1)]
    inter, noise, invalid, _ = SR.samep_rows(runs, POOL)
    assert len(noise) == 4 and sorted((r["j"], r["flip_swing"]) for r in noise) == [(1, False), (1, False),
                                                                                       (2, False), (2, True)]
    fr = SR.flip_rates(noise)
    assert fr["j1"]["flip_any"] == 0.0 and fr["j2+"]["flip_swing"] == 0.5 and fr["j2+"]["flip_button"] == 0.0


def test_bootstrap_reproducible_and_effect_rule():
    rows = [{"recv": f"R:{i % 5}", "x": i % 3 == 0} for i in range(30)]
    a, b = SR.rate_ci(rows, "x"), SR.rate_ci(rows, "x")
    assert a == b and a["n"] == 30 and a["recv"] == 5 and a["ci"][0] <= a["rate"] <= a["ci"][1]
    assert SR.effect_flag([0.5, 0.9], [0.2, 0.4]) is True
    assert SR.effect_flag([0.3, 0.9], [0.2, 0.4]) is False
    assert SR.effect_flag(None, [0.1]) is None and SR.effect_flag([0.5, 0.6], [None, None]) is None


def test_summary_lines_groups_and_primary():
    runs = [run("N", N_STEPS, key=k) for k in POOL]
    runs += [run("N", N_STEPS, key=k, seed=1) for k in POOL]
    for k in POOL:
        runs.append(run("S_swing", N_STEPS[:1] + [st(2, 2, 4, True, "cw")], key=k, j=2, donor_side=1, donor_swing="cw"))
        runs.append(run("S_side", N_STEPS[:1] + [st(2, 2, 0, False, "ccw")], key=k, j=2, donor_side=-1,
                        donor_swing="ccw"))
        runs.append(run("S_both", N_STEPS[:1] + [st(2, 2, 0, False, "cw")], key=k, j=2, donor_side=-1,
                        donor_swing="cw"))
        runs.append(run("Z", N_STEPS, key=k, j=2))
    s = SR.samep_summary(*SR.samep_rows(runs, POOL), model="A", expect_groups=4)
    assert s["lines"][0].startswith("NOISE_BASE=PASS model=A pairs=3 flip_rate=0.000")
    assert s["lines"][1] == "RS_SAMEP=PASS model=A groups=4 invalid=0"
    sw = s["groups"]["S_swing"]["tiers"]["j2+"]
    assert sw["reads"]["swing_follow"]["rate"] == 1.0 and sw["effect"]["swing_follow"]["effect"] is True
    assert sw["effect"]["swing_follow"]["primary"]
    both = s["groups"]["S_both"]["tiers"]["j2+"]
    # S_both：去另一侧 + 标签翻成 cw → 物理弧侧 = ccw = 供体弧侧（缺省推导）
    assert both["reads"]["button_follow"]["rate"] == 1.0 and both["reads"]["swing_follow_phys"]["rate"] == 1.0
    assert s["groups"]["S_both"]["primary"] == ["button_follow", "swing_follow_phys"]
    s7 = SR.samep_summary(*SR.samep_rows(runs, POOL), model="A", expect_groups=7)
    assert s7["lines"][1].startswith("RS_SAMEP=FAIL")


def test_invalid_counted():
    runs = [run("N", N_STEPS), dict(run("S_swing", N_STEPS, j=1, donor_side=1, donor_swing="ccw"), status="invalid"),
            run("S_side", N_STEPS, j=1)]                                    # 缺供体元数据
    inter, noise, invalid, _ = SR.samep_rows(runs, POOL)
    assert invalid["intervention"] == 1 and invalid["donor_meta"] == 1 and inter == []


# ---------------------------------------------------------------- VideoRepick
SCHED = [[0, 1, 0, 50], [1, 2, 50, 100]]


def test_pred_k_from_slot_simulation():
    assert FR.pred_k_from(3, 0, SCHED, 1, 1) == 0 and FR.pred_k_from(3, 0, SCHED, 2, 2) == 0
    assert FR.pred_k_from(3, 0, SCHED, 2, 1) == 2
    assert FR.pred_k_from(3, 0, SCHED, 1, 2) == 2


def vr(group, picked, key="VideoRepick:1", seed=42, status="ok", category=None, **kw):
    return {"task": "VideoRepick", "key": key, "group": group, "seed": seed, "status": status, "picked": picked,
            "category": category or ("correct" if picked == 0 else "other"),
            "rf": {"target_index": 0, "schedule": SCHED}, "labels": {"n_objects": 3, "blind_k": {1: 2, 2: 0}},
            "noise_rep": kw.pop("noise_rep", None), **kw}


def test_vr_categories_nondiscrim_and_noise():
    runs = [vr("N", 0), vr("N", 0, seed=1), vr("N", 1, seed=2),
            vr("T", 2, k=2, k_src=1),                       # pred_2←1 = 2 → pred
            vr("T", 0, k=1, k_src=2, pred_index=0),         # pred = 目标 → 无区分力，剔除
            vr("Z", 2, k=1),                                # blind_1 = 2 → swap_blind_k
            vr("Z", 0, k=2),                                # blind_2 = 目标 → 无区分力
            vr("N", 1, key="VideoRepick:2"), vr("T", 2, key="VideoRepick:2", k=2, k_src=1)]  # 原 N 没拿对 → 不进分母
    inter, noise, invalid, chosen = FR.vr_rows(runs)
    assert chosen == 1 and len(noise) == 2 and sum(r["flip"] for r in noise) == 1
    s = FR.vr_summary(inter, noise, invalid, chosen, "A")
    t, z = s["groups"]["T"], s["groups"]["Z"]
    assert t["nondiscrim"] == 1 and t["reads"]["pred"]["n"] == 1 and t["reads"]["pred"]["rate"] == 1.0
    assert z["nondiscrim"] == 1 and z["reads"]["swap_blind_k"]["rate"] == 1.0
    assert s["lines"] == ["NOISE_BASE=PASS model=A task=VideoRepick pairs=2 flip_rate=0.500",
                          "VR_SAMEP=PASS model=A invalid=0"]
    assert FR.ratio_ci(inter, "flip") == FR.ratio_ci(inter, "flip")


def test_vr_invalid():
    runs = [vr("N", 0), vr("T", None, k=2, k_src=1, status="invalid")]
    s = FR.vr_summary(*FR.vr_rows(runs), "B")
    assert s["lines"][1] == "VR_SAMEP=FAIL model=B invalid=1"
