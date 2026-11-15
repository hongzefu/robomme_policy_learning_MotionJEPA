"""1114 计划 S4-09：scripts/evaluation/check_checkpoint.py 的 RUNS 键、stage 路径取键与 stage2 concat 分支（CPU）。

只验键与分支结构，不验 resolved_sha256 值（S3 合入后由主会话复算补入，占位期间 main 必须拒绝）：
* 两条正式、两条 -attempt02、两条 v16-smoke-* 短测 run 都能查到，正式与 attempt02 只许 50000、短测只许 19；
* 真实形状的 stage 目录 …/exports/hf-ckpt-<run>/stage/50000 命中对应 RUNS 键，未登记 run 与非 hf-ckpt- 上级被拒；
* context 两条不被 modulation 断言拒，modulation 快照配 context run 被拒；
* stage2 concat 分支：rgb-state arch、local-built-latents 来源、1×1024 readout、stride 32、budget 96 通过，篡改任一项被拒；
* c0826be 已有条目逐字不变。
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[2]
CFGDIR = REPO / "src/mme_vla_suite/models/config/robomme"
FORMAL = {"modul": "v2-1600ep-m8x8-modul-motion-s2rgbstate-concat1x1024-b128-50k",
          "ctx": "v2-1600ep-m8x8-ctx-motion-s2rgbstate-concat1x1024-b128-50k"}


@pytest.fixture(scope="module")
def cc():
    spec = importlib.util.spec_from_file_location("check_checkpoint_v16", REPO / "scripts/evaluation/check_checkpoint.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runs_entries(cc):
    for fusion, itype in (("modul", "modulation"), ("ctx", "context")):
        names = {FORMAL[fusion]: ("50000",), f"{FORMAL[fusion]}-attempt02": ("50000",), f"v16-smoke-{fusion}-s2rgbstate-concat1x1024": ("19",)}
        for name, steps in names.items():
            run = cc.RUNS[name]
            assert run["steps"] == steps and run["integration_type"] == itype
            assert run["config"] == "mme_vla_suite_4task_b128_50k" and run["budget"] == 512 and run["motion"] is True
            assert run["motion_budget"] == 96 and run["motion_kind"] == "stage2_pool4_concat"
            assert run["readout"] == "stage2_pool4_concat_v1"
            assert run["motion_arch"] == "wan-latent-stage2-blockwise-pool4-rgb-state"
            assert run["motion_source"] == "local-built-latents"
            assert run["norm_stats_sha256"].startswith("856c75ea")
            assert run["resolved_sha256"] == cc.S2RGBSTATE_PENDING          # S3 合入后由主会话补入


def test_old_entries_unchanged(cc):
    src = subprocess.check_output(["git", "show", "c0826be:scripts/evaluation/check_checkpoint.py"], cwd=REPO, text=True)
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "RUNS")
    ns = {"dict": dict, "WANTP768_RESOLVED_SHA256": cc.WANTP768_RESOLVED_SHA256, "WANTP768_KIND": cc.WANTP768_KIND,
          "WANTP768_READOUT": cc.WANTP768_READOUT, "WANRAW9_RESOLVED_SHA256": cc.WANRAW9_RESOLVED_SHA256,
          "WANRAW9_KIND": cc.WANRAW9_KIND, "WANRAW9_READOUT": cc.WANRAW9_READOUT}
    old = eval(compile(ast.Expression(node.value), "old", "eval"), ns)
    for key, value in old.items():
        assert cc.RUNS[key] == value, key
    assert set(cc.RUNS) - set(old) == {FORMAL["modul"], FORMAL["ctx"], f"{FORMAL['modul']}-attempt02",
                                       f"{FORMAL['ctx']}-attempt02", "v16-smoke-modul-s2rgbstate-concat1x1024", "v16-smoke-ctx-s2rgbstate-concat1x1024"}


def test_stage_path_key(cc, tmp_path):
    exports = tmp_path / "v1-store/exports"
    for fusion in ("modul", "ctx"):
        for name in (FORMAL[fusion], f"{FORMAL[fusion]}-attempt02"):
            ck = exports / f"hf-ckpt-{name}" / "stage" / "50000"
            assert cc.run_key_of(ck) == name
            assert cc.lookup_run(ck)[0] == name
        smoke = tmp_path / "train-runs/mme_vla_suite_4task_b128_50k" / f"v16-smoke-{fusion}-s2rgbstate-concat1x1024" / "19"
        assert cc.lookup_run(smoke)[0] == f"v16-smoke-{fusion}-s2rgbstate-concat1x1024"            # 短测走 parent.name 一支
    old = tmp_path / "x/robomme-vla-modul-60k-v1/59999"
    assert cc.lookup_run(old)[0] == "robomme-vla-modul-60k-v1"             # 旧取键规则不变
    for bad in (exports / "hf-ckpt-not-registered/stage/50000",           # 未登记 run
                exports / f"{FORMAL['modul']}/stage/50000",                # 上级缺 hf-ckpt- 前缀
                exports / "hf-ckpt-/stage/50000",
                exports / f"hf-ckpt-{FORMAL['modul']}/stage/19",           # 正式 run 只许 50000
                tmp_path / "v16-smoke-ctx-s2rgbstate-concat1x1024/50000"):                         # 短测只许 19
        with pytest.raises(AssertionError):
            cc.lookup_run(bad)


def _cfg(fusion: str):
    from omegaconf import OmegaConf
    return OmegaConf.load(CFGDIR / f"perceptual-framesamp-{fusion}-8frame-8x8-motion-stage2-1x1024-orig16.yaml")


def test_history_integration_by_run(cc):
    mod_cfg, ctx_cfg = _cfg("modul"), _cfg("ctx")
    cc.assert_history(mod_cfg, cc.RUNS[FORMAL["modul"]])
    cc.assert_history(ctx_cfg, cc.RUNS[FORMAL["ctx"]])                     # context 不被 modulation 断言拒
    cc.assert_history(ctx_cfg, cc.RUNS[f"{FORMAL['ctx']}-attempt02"])
    with pytest.raises(AssertionError):
        cc.assert_history(mod_cfg, cc.RUNS[FORMAL["ctx"]])
    with pytest.raises(AssertionError):
        cc.assert_history(ctx_cfg, cc.RUNS[FORMAL["modul"]])
    with pytest.raises(AssertionError):                                    # 旧条目缺省仍是 modulation
        cc.assert_history(ctx_cfg, cc.RUNS["robomme-vla-modul-60k-v1"])
    bad = mod_cfg.copy()
    bad.token_per_image = 16
    with pytest.raises(AssertionError):
        cc.assert_history(bad, cc.RUNS[FORMAL["modul"]])


PROV = {"encoder": {"arch": "wan-latent-stage2-blockwise-pool4-rgb-state", "output_shape": [1, 1024]},
        "stage2_compat": {"kind": "local-built-latents"}}


def test_stage2_concat_branch(cc, capsys):
    run = cc.RUNS[FORMAL["ctx"]]
    cfg = _cfg("ctx")
    cc.assert_motion(cfg, run, PROV)
    assert "BUDGET_CONSISTENT=PASS budget=96 motion_kind=stage2_pool4_concat" in capsys.readouterr().out
    bad_provs = [
        {**PROV, "encoder": {**PROV["encoder"], "arch": "wan-latent-stage2-blockwise-pool4-three-decoder"}},
        {**PROV, "encoder": {**PROV["encoder"], "output_shape": [4, 256]}},
        {**PROV, "stage2_compat": {"kind": "historical-restore-receipt"}},
        {"encoder": PROV["encoder"]},
    ]
    for prov in bad_provs:
        with pytest.raises(AssertionError):
            cc.assert_motion(cfg, run, prov)
    for key, value in (("budget", 160), ("stride", 16), ("readout", "stage2_pool4_split_v1"), ("num_tokens", 4),
                       ("encoder_kind", "wan_raw9"), ("source_shape", [4, 256])):
        bad = cfg.copy()
        bad.motion[key] = value
        with pytest.raises(AssertionError):
            cc.assert_motion(bad, run, PROV)


def test_main_refuses_placeholder_sha(cc, tmp_path):
    ck = tmp_path / f"v1-store/exports/hf-ckpt-{FORMAL['modul']}/stage/50000"
    ck.mkdir(parents=True)
    with pytest.raises(AssertionError, match="占位"):
        cc.main(["check_checkpoint.py", str(ck)])
