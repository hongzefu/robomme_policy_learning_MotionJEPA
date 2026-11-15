"""1114 计划 S4a（裁决 5、S4-06、S4-08、A21）：local_692_pool.sh 的分片退出分类与 DRY_RUN 逐片断言。

* cmd_scan：rc ∈ {124, 137, 143, 130} 的假分片，progress 完整、含预标 error 或新 error，一律 verdict=infra 且不写
  retry1-plan.json；rc≠0 无 error 仍 infra；rc=0 只含预标 error 为 ok，rc≠0 只含预标 error 为 expected；
  rc≠0 含新 error 才 retry；progress 不完整为 infra。
* cmd_integrate：重跑片 rc 用同一判定，命中即 RETRY_INFRA 非零退出。
* DRY_RUN：假仓库（假 env.sh / 假 robomme，真 local_692_pool.sh、run_shard.sh、make_shard_plan.py）里对合成
  stage/50000 列出 32 片命令，每卡 4 片，逐片经 run_shard.sh 的 DRY_RUN 打出最终 server argv：
  ``-m scripts.training.serve_policy``、自己的 --motion_online_gpu、config 为调用方的值、客户端入口 eval_692.py；
  不传 POLICY_CONFIG 时仍是 mme_vla_suite_b128_80k；漏传 EXPECT_MOTION_BUDGET / EXPECT_MOTION_STRIDE 必须报错。
全部 CPU、只用标准库与 bash，不起 GPU、不起服务。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "scripts/evaluation"
POOL = EVAL / "local_692_pool.sh"
PRESEED_JSON = REPO / "docs/training-doc/finalckpt-nomotion59999-gl/records/preseeded-errors.json"
POLICY, CKPT, SEED = "pol", "50000", "7"
PRE_KEY = "RouteStick/easy/151"           # 预标 error
NEW_KEY = "RouteStick/easy/152"           # 新 error
OK_KEY = "RouteStick/easy/153"


def pyhelp(tmp: Path) -> Path:
    """抽出 local_692_pool.sh 内嵌的 PYHELP（只用标准库），落成独立脚本。"""
    text = POOL.read_text(encoding="utf-8")
    m = re.search(r"^read -r -d '' PYHELP <<'PY'\n(.*?)^PY\n", text, re.M | re.S)
    assert m
    path = tmp / "pyhelp.py"
    path.write_text(m.group(1), encoding="utf-8")
    return path


def make_shard(root: Path, rc: int, values: dict, planned=(PRE_KEY, NEW_KEY, OK_KEY)) -> tuple[Path, Path]:
    eval_root, plan_dir = root / "eval", root / "plans"
    (eval_root / "logs").mkdir(parents=True)
    plan_dir.mkdir()
    (eval_root / "preseeded-errors.json").write_text(json.dumps({"keys": [PRE_KEY]}))
    (eval_root / "logs" / "s0.rc").write_text(f"{rc}\n")
    groups = {}
    for key in planned:
        g, e = key.rsplit("/", 1)
        groups.setdefault(g, []).append(int(e))
    (plan_dir / "shard0.json").write_text(json.dumps({"shard_tag": "s0", "candidates": "/c.jsonl",
                                                      "identity_sha256": "ab" * 32, "groups": groups}))
    res = eval_root / "s0" / POLICY / f"ckpt{CKPT}" / f"seed{SEED}"
    res.mkdir(parents=True)
    prog = {}
    for key, v in values.items():
        g, e = key.rsplit("/", 1)
        prog.setdefault(g, {})[e] = v
    (res / "progress.json").write_text(json.dumps(prog))
    return eval_root, plan_dir


def scan(tmp: Path, rc: int, values: dict, **kw):
    eval_root, plan_dir = make_shard(tmp, rc, values, **kw)
    out = subprocess.run([sys.executable, str(pyhelp(tmp)), "scan", str(eval_root), str(plan_dir),
                          POLICY, CKPT, SEED, "1"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    verdict = re.search(r"verdict=(\w+)", out.stdout).group(1)
    return verdict, (eval_root / "s0" / "retry1-plan.json").exists()


FULL_PRE = {PRE_KEY: "error", NEW_KEY: False, OK_KEY: True}
FULL_NEW = {PRE_KEY: "error", NEW_KEY: "error", OK_KEY: True}
FULL_CLEAN = {PRE_KEY: True, NEW_KEY: False, OK_KEY: True}


@pytest.mark.parametrize("rc", [124, 137, 143, 130])
@pytest.mark.parametrize("values", [FULL_PRE, FULL_NEW, FULL_CLEAN], ids=["preseeded", "new_error", "no_error"])
def test_signal_exit_is_infra(tmp_path, rc, values):
    assert scan(tmp_path, rc, values) == ("infra", False)


@pytest.mark.parametrize("rc,values,want", [
    (1, FULL_CLEAN, ("infra", False)),          # rc≠0 无 error 仍 infra
    (0, FULL_PRE, ("ok", False)),               # rc=0 只含预标 error
    (1, FULL_PRE, ("expected", False)),         # rc≠0 只含预标 error
    (1, FULL_NEW, ("retry", True)),             # rc≠0 含新 error：只重跑新 error
    (0, FULL_CLEAN, ("ok", False)),
    (0, {PRE_KEY: "error", OK_KEY: True}, ("infra", False)),   # 不完整
])
def test_scan_other_verdicts(tmp_path, rc, values, want):
    assert scan(tmp_path, rc, values) == want


@pytest.mark.parametrize("retry_rc,value,want", [
    (137, "error", 1), (124, "error", 1), (143, False, 1), (130, False, 1),   # 信号退出：有无 error 都是 infra
    (2, False, 1),                                                          # rc≠0 且无 error：infra
    (1, "error", 0), (0, False, 0),                                         # 仍是新 error / 正常跑完：并回
])
def test_integrate_retry_rc_same_rule(tmp_path, retry_rc, value, want):
    eval_root, _ = make_shard(tmp_path, 1, FULL_NEW)
    shard = eval_root / "s0"
    g, e = NEW_KEY.rsplit("/", 1)
    (shard / "retry1-plan.json").write_text(json.dumps({"groups": {g: [int(e)]}}))
    (eval_root / "logs" / "s0-retry1.rc").write_text(f"{retry_rc}\n")
    retry_root = tmp_path / "retry"
    rres = retry_root / "s0" / POLICY / f"ckpt{CKPT}" / f"seed{SEED}"
    rres.mkdir(parents=True)
    (rres / "progress.json").write_text(json.dumps({g: {e: value}}))
    if value != "error":                                                   # 非 error 局须恰有一条重跑视频
        (rres / "videos").mkdir()
        (rres / "videos" / f"RouteStick_ep{e}_False_goal_easy.mp4").write_bytes(b"")
    (shard / POLICY / f"ckpt{CKPT}" / f"seed{SEED}" / "videos").mkdir()
    out = subprocess.run([sys.executable, str(pyhelp(tmp_path)), "integrate", str(eval_root), str(retry_root),
                          POLICY, CKPT, SEED, "1"], capture_output=True, text=True, timeout=60)
    assert out.returncode == want, out.stdout + out.stderr
    if want:
        assert f"RETRY_INFRA shard=s0 rc={retry_rc}" in out.stdout
        assert not (shard / "attempt0").exists()                          # 命中即退出，不动最终目录
    else:
        assert "RETRY_INTEGRATE=PASS retried=1" in out.stdout


# ════════════════════════════ DRY_RUN（假仓库）════════════════════════════
GROUP_RANGES = {
    ("BinFill", "easy"): 155, ("BinFill", "medium"): 153, ("BinFill", "hard"): 153,
    ("RouteStick", "easy"): 150, ("RouteStick", "medium"): 150, ("RouteStick", "hard"): 150,
    ("RouteStick", "xhard"): 150,
    ("VideoRepick", "easy"): 150, ("VideoRepick", "medium"): 150, ("VideoRepick", "hard"): 150,
    ("VideoRepick", "xhard"): 152,
    ("VideoUnmaskSwap", "easy"): 150, ("VideoUnmaskSwap", "medium"): 150, ("VideoUnmaskSwap", "xhard"): 130,
}
FAKE_ROBOMME = f'''
GROUP_RANGES = {GROUP_RANGES!r}
def candidate_key(row):
    return (row["task"], row["difficulty"], int(row["episode"]))
def load_candidates(path, repo_root=None):
    rows = [dict(task=t, difficulty=d, episode=e, split="test", role="primary")
            for (t, d), start in GROUP_RANGES.items() for e in range(start, start + 50)]
    return {{"identity_sha256": "{'3b' * 32}", "sampling_config": {{}}}}, rows
'''
FAKE_ENV_SH = '''#!/usr/bin/env bash
EVAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EVAL_REPO
CLIENT_UV=(env "PYTHONPATH=@SRC@" "@PY@")
SERVER_UV=(env "UV_PROJECT_ENVIRONMENT=$EVAL_REPO/v1-store/envs/policy-eval-server" uv run --project "$EVAL_REPO" --frozen --no-sync python)
'''
FORMAL = "v2-1600ep-m8x8-modul-motion-s2rgbstate-concat1x1024-b128-50k"


@pytest.fixture
def fake_repo(tmp_path):
    src = tmp_path / "fake/src/robomme"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "injection_candidates.py").write_text(FAKE_ROBOMME)
    ev = tmp_path / "repo/scripts/evaluation"
    ev.mkdir(parents=True)
    for name in ("local_692_pool.sh", "run_shard.sh", "make_shard_plan.py"):
        shutil.copy(EVAL / name, ev / name)
    (ev / "env.sh").write_text(FAKE_ENV_SH.replace("@SRC@", str(tmp_path / "fake/src")).replace("@PY@", sys.executable))
    rec = tmp_path / "repo/docs/training-doc/finalckpt-nomotion59999-gl/records"
    rec.mkdir(parents=True)
    shutil.copy(PRESEED_JSON, rec / "preseeded-errors.json")
    store = tmp_path / "store"
    store.mkdir()
    index = {f"BinFill/{d}/{e}": {"path": f"BinFill/{d}/ep{e}.h5", "D": 300, "k_demo": 18, "demo_status": "ok"}
             for (t, d), start in GROUP_RANGES.items() if t == "BinFill" for e in range(start, start + 50)}
    (store / "index.json").write_text(json.dumps(index))
    (store / "E0.txt").write_text("DEMO_PREFIX_SRC=PASS code_same=1 benchmark=b4e97f2\n")
    # 合成 stage：真实形状 …/exports/hf-ckpt-<正式 run>/stage/50000
    ckpt = tmp_path / f"v1-store/exports/hf-ckpt-{FORMAL}/stage/50000"
    (ckpt / "params").mkdir(parents=True)
    (ckpt / "assets/robomme").mkdir(parents=True)
    (ckpt / "assets/robomme/norm_stats.json").write_text("{}")
    env = {k: v for k, v in os.environ.items()
           if k not in ("POLICY_CONFIG", "EXPECT_MOTION_BUDGET", "EXPECT_MOTION_STRIDE", "STAGE2_PYTHON",
                        "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "DRY_RUN")}
    env.update(STORE=str(store), DRY_RUN="1", RUN_NAME="local692-s2rgbstate-modul-s7", POLICY="pol-s2")
    return {"repo": tmp_path / "repo", "ckpt": ckpt, "env": env}


def run_pool(fr, **extra):
    env = dict(fr["env"], **extra)
    return subprocess.run(["bash", str(fr["repo"] / "scripts/evaluation/local_692_pool.sh"), str(fr["ckpt"]), "7"],
                          env=env, capture_output=True, text=True, timeout=240)


def test_dry_run_32_shards_new_config(fake_repo):
    out = run_pool(fake_repo, POLICY_CONFIG="mme_vla_suite_4task_b128_50k",
                   EXPECT_MOTION_BUDGET="96", EXPECT_MOTION_STRIDE="32")
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    s = out.stdout
    assert "EXPECT_MOTION_BUDGET=96 EXPECT_MOTION_STRIDE=32" in s
    assert "POLICY_CONFIG=mme_vla_suite_4task_b128_50k" in s
    cmds = re.findall(r"^SHARD_CMD shard=s(\d+) gpu=(\d) client=(\S+) server: (.+)$", s, re.M)
    assert len(cmds) == 32
    per_gpu = {}
    for k, g, client, argv in cmds:
        k, g = int(k), int(g)
        assert g == k % 8 and client == "examples/robomme/eval_692.py"
        per_gpu[g] = per_gpu.get(g, 0) + 1
        toks = argv.split()
        assert toks[toks.index("-m") + 1] == "scripts.training.serve_policy"
        assert f"--motion_online_gpu={g}" in toks
        assert "--policy.config=mme_vla_suite_4task_b128_50k" in toks
        assert f"--policy.dir={fake_repo['ckpt']}" in toks
        assert f"--port={18011 + k * 7}" in toks
        assert not any(t.startswith("--stage2_python") for t in toks)
    assert per_gpu == {g: 4 for g in range(8)}
    assert "DRY_SHARDS=PASS shards=32 per_gpu=4 max_on_gpu=4 config=mme_vla_suite_4task_b128_50k" in s
    assert "PRESEED=PASS episodes=9 placed=9 duplicates=0" in s
    assert not (fake_repo["repo"] / "v1-store").exists()                 # DRY_RUN 不写评估目录


def test_dry_run_default_config_and_stage2_python(fake_repo):
    s2py = "/abs/v1-store/venvs/wan-stage2-rgbstate/bin/python"
    out = run_pool(fake_repo, EXPECT_MOTION_BUDGET="96", EXPECT_MOTION_STRIDE="32", STAGE2_PYTHON=s2py)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    cmds = re.findall(r"^SHARD_CMD .* server: (.+)$", out.stdout, re.M)
    assert len(cmds) == 32
    assert all("--policy.config=mme_vla_suite_b128_80k" in c.split() for c in cmds)
    assert all(f"--stage2_python={s2py}" in c.split() for c in cmds)


@pytest.mark.parametrize("missing", ["EXPECT_MOTION_BUDGET", "EXPECT_MOTION_STRIDE"])
def test_dry_run_requires_motion_expectations(fake_repo, missing):
    extra = {"POLICY_CONFIG": "mme_vla_suite_4task_b128_50k", "EXPECT_MOTION_BUDGET": "96",
             "EXPECT_MOTION_STRIDE": "32"}
    del extra[missing]
    out = run_pool(fake_repo, **extra)
    assert out.returncode != 0
    assert f"必须显式传 {missing}" in out.stdout + out.stderr
    assert "SHARD_CMD" not in out.stdout


def test_run_shard_dry_rejects_bad_ckpt_id(fake_repo, tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    env = dict(fake_repo["env"], CUDA_VISIBLE_DEVICES="5", SHARD_INDEX="13", EXPECT_CKPT_ID="59999")
    out = subprocess.run(["bash", str(fake_repo["repo"] / "scripts/evaluation/run_shard.sh"), "r", str(fake_repo["ckpt"]),
                          str(plan), "7", "s13"], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 2 and "本轮只验收 step 59999" in out.stdout
    out = subprocess.run(["bash", str(fake_repo["repo"] / "scripts/evaluation/run_shard.sh"), "r", str(fake_repo["ckpt"]),
                          str(plan), "7", "s13"], env=dict(env, EXPECT_CKPT_ID="50000", STAGE2_PYTHON="python"),
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 2 and "STAGE2_PYTHON 必须是" in out.stdout
