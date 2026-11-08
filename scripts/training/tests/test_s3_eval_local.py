"""1108 计划 S3（评估侧）单测：只用主 venv 能跑的部分，CPU、无 GPU、不下载、不起评估。

覆盖（对应 1108-4task-modul-wantp768-50k-plan.html 第二部分「二」S3 行的验收夹具）：
* C6：make_shard_plan.py 带 --demo-store 时 BinFill 三组 50/35/50（15 条失败全来自 hard）10 片通过、
  同组跨片最多差 1、全集覆盖无重复；旧版在同输入上必失败；无 --demo-store 时新旧两版输出与计划文件逐字节相同。
* C5：DRY_RUN=1 local_692_pool.sh 在前缀成功 142 / 143 两种库上都 PRESEED=PASS episodes=9 placed=9 duplicates=0，
  所在分片按本次计划文件独立复算一致，且 143 时确有身份换了分片（轮转整体错位）。
* C7：在假仓库里端到端跑 local_692_pool.sh（假 run_shard.sh，真 make_shard_plan / check_shard / merge_shards）：
  一局新 error 重跑成功、一局重跑仍 error、一局普通失败不重试；首次产物进 attempt0/、重跑进 retry1/，
  check_shard.py 原判据按最终目录通过，分片首次非零退出被记录而不终止驱动，EVAL692 统计正确。
* C4：run_shard.sh::check_local_gpu 抽出单测：默认 LOCAL_GPU_SHARE=1 与 ba52d79 旧函数输出、退出码逐字相同；
  =2 时登记进程树放行、非登记进程被拒、第 3 片被拒、本片未登记被拒；驱动 slot_reserve 第 3 片被拒。
* finalize：pregen_binfill_demo.py 不传参、传 0.05 与 ba52d79 旧版行为相同；6% 失败在 0.10 下通过；>10% 退出；
  diff 只落在 cmd_finalize 与 main。
* check_checkpoint：旧条目逐字不变；resolved_sha256 复算口径对既有两条 run 逐位复现；新两条条目字段；
  assert_motion 的 encoder / 池化分支与篡改反例。
运行：UV_PROJECT_ENVIRONMENT=<主 .venv> PYTHONPATH=<worktree>/src uv run --no-sync python -m pytest \
      scripts/training/tests/test_s3_eval_local.py -q
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap
import types

import pytest

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / "scripts/evaluation"
BASE = "ba52d79"
PRESEED_JSON = REPO / "docs/training-doc/finalckpt-nomotion59999-gl/records/preseeded-errors.json"
OLD_INDEX = REPO / "docs/training-doc/binfilldemo-nomotion50k-gl/records/demo-prefix-index.json"
OLD_MANIFEST = REPO / "docs/training-doc/binfilldemo-nomotion50k-gl/records/demo-prefix-manifest.json"
IDENTITY = "3b4de03a0b4639181f9c68086aea11e9b51ffcb65bbed701a44f573e74e0beb7"

# 合成候选库：14 组 × 50，组与 episode 区间覆盖历史 9 局预标身份与旧前缀库的 BinFill 区间
GROUP_RANGES = {
    ("BinFill", "easy"): 155, ("BinFill", "medium"): 153, ("BinFill", "hard"): 153,
    ("RouteStick", "easy"): 150, ("RouteStick", "medium"): 150, ("RouteStick", "hard"): 150,
    ("RouteStick", "xhard"): 150,
    ("VideoRepick", "easy"): 150, ("VideoRepick", "medium"): 150, ("VideoRepick", "hard"): 150,
    ("VideoRepick", "xhard"): 152,
    ("VideoUnmaskSwap", "easy"): 150, ("VideoUnmaskSwap", "medium"): 150, ("VideoUnmaskSwap", "xhard"): 130,
}

FAKE_ROBOMME = f'''
"""测试用假 robomme：只提供 make_shard_plan.py 用到的 load_candidates / candidate_key。"""
GROUP_RANGES = {GROUP_RANGES!r}
IDENTITY = {IDENTITY!r}

def candidate_key(row):
    return (row["task"], row["difficulty"], int(row["episode"]))

def load_candidates(path, repo_root=None):
    rows = [dict(task=t, difficulty=d, episode=e, split="test", role="primary")
            for (t, d), start in GROUP_RANGES.items() for e in range(start, start + 50)]
    return {{"identity_sha256": IDENTITY, "sampling_config": {{}}}}, rows
'''


def git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True)


def write_fake_robomme(root: Path) -> Path:
    pkg = root / "fake/src/robomme"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "injection_candidates.py").write_text(FAKE_ROBOMME)
    return root / "fake/src"


def binfill_index(n_ok: int | None = None, drop_hard: int = 0) -> dict:
    """旧前缀库（142 条 ok）为基；n_ok=143 时把旧失败清单里第一条补成 ok；drop_hard 时 hard 只留 50-drop_hard 条。"""
    if drop_hard:
        index = {}
        for (t, d), start in GROUP_RANGES.items():
            if t != "BinFill":
                continue
            eps = range(start, start + 50 - (drop_hard if d == "hard" else 0))
            for e in eps:
                index[f"{t}/{d}/{e}"] = {"path": f"{t}/{d}/ep{e}.h5", "D": 300, "k_demo": 18, "demo_status": "ok"}
        return index
    index = json.loads(OLD_INDEX.read_text(encoding="utf-8"))
    if n_ok == 143:
        key = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))["failed_keys"][0]
        task, diff, ep = key.split("/")
        index[key] = {"path": f"{task}/{diff}/ep{ep}.h5", "D": 300, "k_demo": 18, "demo_status": "ok"}
    return index


# ════════════════════════════ C6：make_shard_plan.py ════════════════════════════
def _run_plan(script: Path, out: Path, src: Path, store: Path | None):
    cmd = [sys.executable, str(script), "shards", str(out), "--shards", "10"]
    if store is not None:
        cmd += ["--demo-store", str(store)]
    env = dict(os.environ, PYTHONPATH=str(src))
    return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)


def test_c6_binfill_50_35_50_cross_shard_balance(tmp_path):
    src = write_fake_robomme(tmp_path)
    store = tmp_path / "store"
    store.mkdir()
    (store / "index.json").write_text(json.dumps(binfill_index(drop_hard=15)))
    new = _run_plan(EVAL / "make_shard_plan.py", tmp_path / "new", src, store)
    assert new.returncode == 0, new.stderr
    assert "PLAN_OK shards=10 total=685 " in new.stdout
    shards = [json.loads((tmp_path / "new" / f"shard{i}.json").read_text()) for i in range(10)]
    keys = [(g, e) for s in shards for g, eps in s["groups"].items() for e in eps]
    assert len(keys) == len(set(keys)) == 685                                   # 无重复、全集覆盖
    for group in {g for s in shards for g in s["groups"]}:
        counts = [len(s["groups"].get(group, [])) for s in shards]
        assert max(counts) - min(counts) <= 1, (group, counts)                  # 同组跨片最多差 1
    # 旧版「片内各组最多差 1」在同一输入上必失败（这正是替换的原因）
    old_script = tmp_path / "old_make_shard_plan.py"
    old_script.write_text(git("show", f"{BASE}:scripts/evaluation/make_shard_plan.py"))
    old = _run_plan(old_script, tmp_path / "old", src, store)
    assert old.returncode != 0 and "组分布不均" in old.stderr


def test_c6_default_path_byte_identical(tmp_path):
    src = write_fake_robomme(tmp_path)
    old_script = tmp_path / "old_make_shard_plan.py"
    old_script.write_text(git("show", f"{BASE}:scripts/evaluation/make_shard_plan.py"))
    new = _run_plan(EVAL / "make_shard_plan.py", tmp_path / "out_new", src, None)
    old = _run_plan(old_script, tmp_path / "out_old", src, None)
    assert new.returncode == old.returncode == 0
    assert new.stdout == old.stdout and "PLAN_OK shards=10 total=700 per_shard=70" in new.stdout
    for name in sorted(p.name for p in (tmp_path / "out_old").iterdir()):
        assert (tmp_path / "out_new" / name).read_bytes() == (tmp_path / "out_old" / name).read_bytes(), name


# ════════════════════════════ 假仓库（C5 / C7 / demo 驱动）════════════════════════════
FAKE_ENV_SH = '''#!/usr/bin/env bash
# 测试用假 env.sh：CLIENT_UV / SERVER_UV 指向主 venv 解释器 + 假 robomme
EVAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EVAL_REPO
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.7}"
CLIENT_UV=(env "PYTHONPATH=@SRC@" "@PY@")
SERVER_UV=("${CLIENT_UV[@]}")
'''

FAKE_RUN_SHARD_SH = '''#!/usr/bin/env bash
# 测试用假 run_shard.sh：exec 保持 PID 不变（驱动登记的就是这个 PID）
source "$(dirname "$0")/env.sh"
exec "${CLIENT_UV[@]}" "$EVAL_REPO/fake_run_shard.py" "$@"
'''

FAKE_RUN_SHARD_PY = r'''
"""假 run_shard.sh：核占位登记与环境，按场景写 progress / 视频 / motion_stats / client.log，按场景退出码退出。"""
import json, os, shutil, sys, time
from pathlib import Path
run, ckpt, plan_path, seed, tag = sys.argv[1:6]
repo = Path(os.environ["EVAL_REPO"])
scen = json.loads(Path(os.environ["FAKE_SCENARIO"]).read_text())
ledger = repo / "ledger.jsonl"
slot_file = Path(os.environ["LOCAL_GPU_SLOT_FILE"])
assert os.environ["LOCAL_GPU_SHARE"] == "2" and os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.45"
for name, want in (("MMEVLA_MOTION_OVERFLOW", "resample"), ("EXPECT_MOTION_STATS", "1"), ("EXPECT_DEMO_INJECT", "1"),
                   ("CHUNK_EPISODES", "20"), ("EPISODE_WALL_S", "2400"), ("ALLOW_RESUME", "1"),
                   ("MMEVLA_MOTION_PROV_RELAX", "gpu_name,compute_cap,sm_count")):
    assert os.environ[name] == want, (name, os.environ.get(name))
lines = []
for _ in range(100):                      # 驱动在后台起本进程后才写 PID，最多等 5 s
    lines = [l.split() for l in slot_file.read_text().splitlines() if l.strip()]
    if [tag, str(os.getpid())] in lines:
        break
    time.sleep(0.05)
own = [tag, str(os.getpid())] in lines
with open(ledger, "a") as f:
    f.write(json.dumps({"run": run, "tag": tag, "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                        "registered": len(lines), "own": own}) + "\n")
assert own and len(lines) <= 2, lines
policy = os.environ["POLICY"]
run_dir = repo / "v1-store/evaluation" / run / tag
root = run_dir / policy / f"ckpt{Path(ckpt).name}" / f"seed{seed}"
root.mkdir(parents=True, exist_ok=True)
plan = json.loads(Path(plan_path).read_text())
pp = root / "progress.json"
prog = json.loads(pp.read_text()) if pp.exists() else {}
table = scen["retry" if run.endswith("-retry1") else "first"]
stats, log = {}, []
for group, eps in plan["groups"].items():
    task, diff = group.split("/")
    prog.setdefault(group, {})
    for e in eps:
        if str(e) in prog[group]:
            continue
        key = f"{group}/{e}"
        value = table.get(key, True)
        prog[group][str(e)] = value
        if value != "error":
            (root / "videos").mkdir(exist_ok=True)
            shutil.copy(scen["video"], root / "videos" / f"{task}_ep{e}_fake_{diff}.mp4")
            stats[key] = {"exec_start_idx": 0, "motion_infers": 1, "motion_k_max": 0, "motion_downsample_steps": 0,
                          "motion_budget": 160, "motion_overflow": "resample"}
            if task == "BinFill":
                log.append(f"DEMO_PREFIX_INJECTED key={key} D=300")
pp.write_text(json.dumps(prog))
if stats:
    (root / "motion_stats.json").write_text(json.dumps(stats))
(root / "plan.json").write_text(json.dumps(plan))
with open(run_dir / "client.log", "a") as f:
    f.write("\n".join(log) + "\n")
nonzero = (not run.endswith("-retry1")) and any(
    f"{g}/{e}" in scen["rc_on_keys"] for g, eps in plan["groups"].items() for e in eps)
sys.exit(1 if nonzero else 0)
'''


def make_fake_repo(tmp: Path, index: dict, scenario: dict | None = None) -> dict:
    src = write_fake_robomme(tmp)
    ev = tmp / "repo/scripts/evaluation"
    ev.mkdir(parents=True)
    for name in ("local_692_pool.sh", "local_demo_pool.sh", "make_shard_plan.py", "check_shard.py", "merge_shards.py"):
        shutil.copy(EVAL / name, ev / name)
    (ev / "env.sh").write_text(FAKE_ENV_SH.replace("@SRC@", str(src)).replace("@PY@", sys.executable))
    (ev / "run_shard.sh").write_text(FAKE_RUN_SHARD_SH)
    (tmp / "repo/fake_run_shard.py").write_text(FAKE_RUN_SHARD_PY)
    rec = tmp / "repo/docs/training-doc/finalckpt-nomotion59999-gl/records"
    rec.mkdir(parents=True)
    shutil.copy(PRESEED_JSON, rec / "preseeded-errors.json")
    store = tmp / "store"
    store.mkdir()
    (store / "index.json").write_text(json.dumps(index))
    (store / "E0.txt").write_text("DEMO_PREFIX_SRC=PASS code_same=1 benchmark=b4e97f2 shards_ok=16/16 ok=142 "
                                  "failed=8 diff_vs_old=0\n")
    ckpt = tmp / "ckpt/v2-1600ep-m8x8-modul-wantp768-b128-50k/50000"
    (ckpt / "params").mkdir(parents=True)
    env = dict(os.environ, STORE=str(store))
    if scenario is not None:
        import imageio.v2 as imageio
        import numpy as np
        video = tmp / "fake.mp4"
        w = imageio.get_writer(video, fps=10)
        for i in range(3):
            w.append_data(np.full((16, 16, 3), 40 * i, np.uint8))
        w.close()
        scenario = dict(scenario, video=str(video))
        (tmp / "scenario.json").write_text(json.dumps(scenario))
        env["FAKE_SCENARIO"] = str(tmp / "scenario.json")
    return {"repo": tmp / "repo", "ckpt": ckpt, "env": env, "store": store}


def _preseed_keys():
    data = json.loads(PRESEED_JSON.read_text())
    return sorted({f"{g}/{e}" for run in data.values() for groups in run["by_shard"].values()
                   for g, eps in groups.items() for e in eps})


def _dry_placements(tmp: Path, index: dict):
    fr = make_fake_repo(tmp, index)
    out = subprocess.run(["bash", str(fr["repo"] / "scripts/evaluation/local_692_pool.sh"), str(fr["ckpt"]), "7"],
                         env=dict(fr["env"], DRY_RUN="1"), capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PRESEED=PASS episodes=9 placed=9 duplicates=0" in out.stdout
    plan_dir = Path(re.search(r"^DRY_PLAN_DIR=(.+)$", out.stdout, re.M).group(1))
    placed = dict(re.findall(r"^PRESEED_PLACE key=(\S+) shards=(\S+)$", out.stdout, re.M))
    # 按本次计划文件独立复算：每个身份恰好在一个分片
    for key in _preseed_keys():
        group, ep = key.rsplit("/", 1)
        hits = [f"s{k}" for k in range(10)
                if int(ep) in json.loads((plan_dir / f"shard{k}.json").read_text())["groups"].get(group, [])]
        assert hits == [placed[key]], (key, hits, placed.get(key))
    assert not (fr["repo"] / "v1-store").exists()          # DRY_RUN 不写评估目录
    return out.stdout, placed


def test_c5_preseed_relocation_142_vs_143(tmp_path):
    out142, p142 = _dry_placements(tmp_path / "a", binfill_index())
    out143, p143 = _dry_placements(tmp_path / "b", binfill_index(143))
    assert "PLAN_OK shards=10 total=692 per_shard=69 disjoint=True groups=14" in out142
    assert "PLAN_OK shards=10 total=693 " in out143
    assert len(p142) == len(p143) == 9
    assert any(p142[k] != p143[k] for k in p142), "前缀成功 +1 后轮转错位，至少一局应换分片"
    for k in range(10):
        assert f"SHARD_MAP shard=s{k} gpu={k % 8} port_base={18011 + k * 7} " in out142


def test_c7_retry_flow_end_to_end(tmp_path):
    new_ok, new_err, plain_fail = "BinFill/easy/160", "RouteStick/hard/170", "RouteStick/easy/151"
    scenario = {"first": {new_ok: "error", new_err: "error", plain_fail: False},
                "retry": {new_ok: True, new_err: "error"},
                "rc_on_keys": [new_ok]}
    fr = make_fake_repo(tmp_path, binfill_index(), scenario)
    out = subprocess.run(["bash", str(fr["repo"] / "scripts/evaluation/local_692_pool.sh"), str(fr["ckpt"]), "7"],
                         env=fr["env"], capture_output=True, text=True, timeout=280)
    assert out.returncode == 0, out.stdout[-4000:] + out.stderr[-4000:]
    s = out.stdout
    root = fr["repo"] / "v1-store/evaluation/local692-wantp768-50k"
    owner = {}
    for k in range(10):
        plan = json.loads((root / "plans" / f"shard{k}.json").read_text())
        for g, eps in plan["groups"].items():
            for e in eps:
                owner[f"{g}/{e}"] = f"s{k}"
    tag_ok, tag_err = owner[new_ok], owner[new_err]
    assert f"SHARD_EXIT_CODE=1 shard={tag_ok}" in s                     # 首次非零退出被记录
    assert "verdict=infra" not in s
    assert s.count("CHECK_SHARD_EXIT_CODE=0 ") == 10                     # check_shard 原判据按最终目录通过
    assert ("EVAL692=PASS episodes=692 success=681 rate=0.9841 preseeded_errors=9 new_errors=1 retried=2 "
            "expect_motion_stats=1") in s
    retries = json.loads((root / "retries.json").read_text())
    assert sorted(r["key"] for r in retries) == sorted([new_ok, new_err])  # 普通失败局不重试
    pol = "perceptual-framesamp-modul-8frame-8x8-wantp768/ckpt50000/seed7"
    group, ep = new_ok.rsplit("/", 1)
    a0 = json.loads((root / tag_ok / "attempt0/progress.json").read_text())
    assert a0[group][ep] == "error"
    final = json.loads((root / tag_ok / pol / "progress.json").read_text())
    assert final[group][ep] is True
    assert (root / tag_ok / "retry1" / pol / "progress.json").is_file()
    assert (root / tag_err / "attempt0").is_dir() and (root / tag_err / "retry1").is_dir()
    vids = list((root / tag_ok / pol / "videos").glob(f"BinFill_ep{ep}_*_easy.mp4"))
    assert len(vids) == 1
    assert not (fr["repo"] / "v1-store/evaluation/local692-wantp768-50k-retry1").exists()
    ledger = [json.loads(l) for l in (fr["repo"] / "ledger.jsonl").read_text().splitlines()]
    assert len(ledger) == 12 and all(r["own"] and r["registered"] <= 2 for r in ledger)
    for p in (root / "gpu-slots").glob("gpu?"):
        assert p.read_text().strip() == "", p                            # 片结束全部释放


def test_demo_pool_dry_run_maps(tmp_path):
    fr = make_fake_repo(tmp_path, binfill_index())
    script = str(fr["repo"] / "scripts/evaluation/local_demo_pool.sh")
    out = subprocess.run(["bash", script], env=dict(fr["env"], DRY_RUN="1", SHARDS="24"),
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    maps = re.findall(r"^DEMO_SHARD_MAP shard=(\d+)/24 gpu=(\d)$", out.stdout, re.M)
    assert [(int(k), int(g)) for k, g in maps] == [(k, k % 8) for k in range(24)]
    assert "--max-fail-rate 0.10" in out.stdout
    probe = subprocess.run(["bash", script, "v1-store/demo-prefix/probe-w768"],
                           env=dict(fr["env"], DRY_RUN="1", PROBE="1", PROBE_GPU="3", PROBE_SHARDS="2"),
                           capture_output=True, text=True, timeout=60)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert re.findall(r"^DEMO_SHARD_MAP shard=(\d+)/16 gpu=(\d)$", probe.stdout, re.M) == [("0", "3"), ("1", "3")]
    assert "finalize" not in probe.stdout
    bad = subprocess.run(["bash", script], env=dict(fr["env"], DRY_RUN="1", SHARDS="15"),
                         capture_output=True, text=True, timeout=60)
    assert bad.returncode == 2


# ════════════════════════════ C4：check_local_gpu / slot_reserve ════════════════════════════
def _extract_func(text: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{[^\n]*\n.*?^\}}\n", text, re.M | re.S)
    assert m, name
    return m.group(0)


def _run_func(tmp: Path, func: str, env: dict, call: str = "check_local_gpu"):
    script = tmp / "case.sh"
    script.write_text("set -euo pipefail\n" + func + f"\n{call}\n")
    return subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=30)


@pytest.fixture
def gpu_env(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "nvidia-smi").write_text('#!/usr/bin/env bash\nprintf "%s" "${FAKE_PIDS:-}"\n')
    (bindir / "nvidia-smi").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("SLURM_JOB_ID", "LOCAL_GPU_SHARE")}
    env.update(PATH=f"{bindir}:{os.environ['PATH']}", CUDA_VISIBLE_DEVICES="3", SHARD_TAG="s3")
    return env


def test_c4_check_local_gpu(tmp_path, gpu_env):
    new_func = _extract_func((EVAL / "run_shard.sh").read_text(), "check_local_gpu")
    old_func = _extract_func(git("show", f"{BASE}:scripts/evaluation/run_shard.sh"), "check_local_gpu")
    root = subprocess.Popen(["bash", "-c", "sleep 30 & wait"])
    stranger = subprocess.Popen(["sleep", "30"])
    try:
        child = None
        for _ in range(100):
            kids = Path(f"/proc/{root.pid}/task/{root.pid}/children").read_text().split()
            if kids:
                child = kids[0]
                break
            import time
            time.sleep(0.05)
        assert child
        # 默认（未设 LOCAL_GPU_SHARE）：与旧函数输出、退出码逐字相同
        for pids in ("", child):
            env = dict(gpu_env, FAKE_PIDS=pids)
            n, o = _run_func(tmp_path, new_func, env), _run_func(tmp_path, old_func, env)
            assert (n.returncode, n.stdout) == (o.returncode, o.stdout)
            assert n.returncode == (0 if not pids else 2)
        assert "LOCAL_GPU_IDLE=PASS device=3" in _run_func(tmp_path, new_func, dict(gpu_env, FAKE_PIDS="")).stdout
        slot = tmp_path / "gpu3"
        share = dict(gpu_env, LOCAL_GPU_SHARE="2", LOCAL_GPU_SLOT_FILE=str(slot))
        slot.write_text(f"s3 pending\ns7 {root.pid}\n")
        ok = _run_func(tmp_path, new_func, dict(share, FAKE_PIDS=child))           # 登记进程树的子孙：放行
        assert ok.returncode == 0 and "LOCAL_GPU_SHARE=PASS device=3 share=2 registered=2 gpu_procs=1" in ok.stdout
        bad = _run_func(tmp_path, new_func, dict(share, FAKE_PIDS=f"{child}\n{stranger.pid}"))  # 非本轮进程：拒绝
        assert bad.returncode == 2 and "不属于已登记分片的进程树" in bad.stdout
        slot.write_text(f"s3 pending\ns7 {root.pid}\ns9 123\n")                    # 第 3 片：拒绝
        third = _run_func(tmp_path, new_func, dict(share, FAKE_PIDS=""))
        assert third.returncode == 2 and "> 2" in third.stdout
        slot.write_text(f"s7 {root.pid}\n")                                        # 本片未登记：拒绝
        assert _run_func(tmp_path, new_func, dict(share, FAKE_PIDS="")).returncode == 2
    finally:
        root.kill()
        stranger.kill()


def test_c4_slot_reserve_third_refused(tmp_path):
    text = (EVAL / "local_692_pool.sh").read_text()
    func = _extract_func(text, "slot_reserve")
    env = dict(os.environ)
    script = tmp_path / "slots.sh"
    script.write_text("set -uo pipefail\n" + f'SLOT_DIR="{tmp_path}"\n' + func + textwrap.dedent("""
        slot_reserve 0 s0; echo "rc=$?"
        slot_reserve 0 s8; echo "rc=$?"
        slot_reserve 0 s16; echo "rc=$?"
    """))
    out = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=30)
    assert re.findall(r"^rc=(\d+)$", out.stdout, re.M) == ["0", "0", "3"]
    assert "SLOT_REFUSED gpu=0 tag=s16 registered=2" in out.stdout
    assert (tmp_path / "gpu0").read_text() == "s0 pending\ns8 pending\n"


# ════════════════════════════ finalize --max-fail-rate ════════════════════════════
def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.subprocess = types.SimpleNamespace(check_output=lambda *a, **k: "deadbeef\n")
    return mod


def _finalize(mod, tmp: Path, n_failed: int, extra: list[str], tag: str):
    out = tmp / f"{tag}-f{n_failed}"
    rows = [dict(task="BinFill", difficulty=d, episode=e, split="test", role="primary")
            for d, start in (("easy", 155), ("hard", 153), ("medium", 153)) for e in range(start, start + 50)]
    failed = {f"BinFill/{r['difficulty']}/{r['episode']}" for r in rows[50:50 + n_failed]}
    (out / "logs").mkdir(parents=True)
    for shard in range(2):
        recs = []
        for r in rows[shard::2]:
            key = f"BinFill/{r['difficulty']}/{r['episode']}"
            path = f"BinFill/{r['difficulty']}/ep{r['episode']}.h5"
            recs.append(dict(task="BinFill", difficulty=r["difficulty"], episode=r["episode"], path=path,
                             D=300, k_demo=18, bytes=10, used_rrt=False, attempts=1,
                             demo_status="failed" if key in failed else "ok"))
            if key not in failed:
                (out / path).parent.mkdir(parents=True, exist_ok=True)
                (out / path).write_bytes(b"")
        (out / "logs" / f"shard{shard}.json").write_text(json.dumps({"identity_sha256": IDENTITY, "episodes": recs}))
    mod.load_binfill_rows = lambda candidates: ({"identity_sha256": IDENTITY}, rows)
    argv = ["pregen_binfill_demo.py", "finalize", "--out", str(out), "--shards", "2",
            "--candidates", "/nonexistent/candidates.jsonl", *extra]
    old_argv, sys.argv = sys.argv, argv
    import contextlib
    import io
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            try:
                mod.main()
                code = None
            except SystemExit as e:
                code = e.code
    finally:
        sys.argv = old_argv
    files = {n: (out / n).read_bytes() for n in ("index.json", "manifest.json") if (out / n).exists()}
    text = buf.getvalue().replace(str(out), "<OUT>")
    files = {n: b.replace(str(out).encode(), b"<OUT>") for n, b in files.items()}
    return code, text, files


def test_finalize_max_fail_rate(tmp_path):
    old_path = tmp_path / "old_pregen.py"
    old_path.write_text(git("show", f"{BASE}:scripts/evaluation/pregen_binfill_demo.py"))
    new = _load_module(EVAL / "pregen_binfill_demo.py", "pregen_new")
    old = _load_module(old_path, "pregen_old")
    for n in (6, 8):                                  # 4%：通过；5.3%：> 5% 退出——三种调用逐字相同
        a = _finalize(new, tmp_path, n, [], "default")
        b = _finalize(new, tmp_path, n, ["--max-fail-rate", "0.05"], "explicit")
        c = _finalize(old, tmp_path, n, [], "old")
        assert a == b == c, n
    assert _finalize(new, tmp_path, 8, [], "d2")[0] == "失败率 5.3% > 5%，按计划停下来找用户裁决"
    assert _finalize(new, tmp_path, 9, ["--max-fail-rate", "0.10"], "r10")[0] == 0        # 6% 在 0.10 下通过
    assert _finalize(new, tmp_path, 15, ["--max-fail-rate", "0.10"], "r10")[0] == 0       # 恰 10% 通过
    code = _finalize(new, tmp_path, 16, ["--max-fail-rate", "0.10"], "r10")[0]
    assert code == "失败率 10.7% > 10%，按计划停下来找用户裁决"                              # > 10% 退出


def test_pregen_diff_only_finalize_and_argparse():
    hunks = [l for l in git("diff", "-U0", BASE, "--", "scripts/evaluation/pregen_binfill_demo.py").splitlines()
             if l.startswith("@@")]
    assert hunks and all(h.endswith("@@ def cmd_finalize(args):") or h.endswith("@@ def main():") for h in hunks), hunks
    assert git("diff", BASE, "--", "scripts/evaluation/run_demo_shard.sh", "examples/robomme/demo_prefix.py") == ""


# ════════════════════════════ check_checkpoint.py ════════════════════════════
@pytest.fixture(scope="module")
def cc():
    return _load_module(EVAL / "check_checkpoint.py", "check_checkpoint_s3")


def _old_runs():
    tree = ast.parse(git("show", f"{BASE}:scripts/evaluation/check_checkpoint.py"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "RUNS")
    return eval(compile(ast.Expression(node.value), "old", "eval"), {"dict": dict})


def test_check_checkpoint_old_entries_and_digest_rule(cc):
    old = _old_runs()
    for key, value in old.items():
        assert cc.RUNS[key] == value, key                                   # 旧条目逐字不变
    cfgdir = REPO / "src/mme_vla_suite/models/config/robomme"
    # 复算口径对既有两条 run 逐位复现白名单里的 resolved_sha256（同 init_history_config 口径）
    assert cc.resolved_sha256_of_yaml(cfgdir / "perceptual-framesamp-modul-8frame-8x8-motion.yaml") == \
        old["robomme-vla-modul-motion-80k-v1"]["resolved_sha256"]
    assert cc.resolved_sha256_of_yaml(cfgdir / "perceptual-framesamp-modul-8frame-8x8.yaml") == \
        old["robomme-vla-modul-60k-v1"]["resolved_sha256"]


def test_check_checkpoint_new_entries(cc):
    formal, smoke = cc.RUNS["v2-1600ep-m8x8-modul-wantp768-b128-50k"], cc.RUNS["w768-smoke20"]
    for run in (formal, smoke):
        assert run["motion"] is True and run["config"] == "mme_vla_suite_b128_80k" and run["budget"] == 512
        assert run["motion_budget"] == 160 and run["motion_kind"] == "wan_fixedpool768"
        assert run["readout"] == "wan_raw_timepool8_adaptive6x8_v1" and "motion_source_run" not in run
        assert run["norm_stats_sha256"].startswith("856c75ea") and run["norm_stats_sha256"].endswith("d173")
    assert formal["steps"] == ("50000",) and smoke["steps"] == ("19",)
    assert formal["resolved_sha256"] == smoke["resolved_sha256"] == cc.WANTP768_RESOLVED_SHA256
    yaml = REPO / cc.WANTP768_YAML
    if not yaml.is_file():
        pytest.skip("S2 定稿 YAML 尚未并入本分支：WANTP768_RESOLVED_SHA256 待填")
    assert cc.WANTP768_RESOLVED_SHA256 == cc.resolved_sha256_of_yaml(yaml)


def test_check_checkpoint_assert_motion_branches(cc, capsys):
    from omegaconf import OmegaConf
    cfgdir = REPO / "src/mme_vla_suite/models/config/robomme"
    enc_cfg = OmegaConf.load(cfgdir / "perceptual-framesamp-modul-8frame-8x8-motion.yaml")
    cc.assert_motion(enc_cfg, cc.RUNS["robomme-vla-modul-motion-80k-v1"], {})
    assert "BUDGET_CONSISTENT=PASS budget=160 source_run=wan-full1600" in capsys.readouterr().out
    pool_cfg = OmegaConf.load(cfgdir / "perceptual-framesamp-modul-8frame-8x8-motion.yaml")
    del pool_cfg.motion["source_run"]
    pool_cfg.motion.encoder_kind = "wan_fixedpool768"
    pool_cfg.motion.readout = "wan_raw_timepool8_adaptive6x8_v1"
    prov = {"encoder_kind": "wan_fixedpool768", "readout": "wan_raw_timepool8_adaptive6x8_v1",
            "encoder": None, "vae": {"sha256": "x"}, "pool_source_sha256": "ab" * 32}
    run = cc.RUNS["v2-1600ep-m8x8-modul-wantp768-b128-50k"]
    cc.assert_motion(pool_cfg, run, prov)                                    # 池化分支不要求 source_run
    assert "motion_kind=wan_fixedpool768" in capsys.readouterr().out
    bad_cfg = pool_cfg.copy()
    bad_cfg.motion.readout = "wan_raw_timepool8_8x8_v0"
    for c, p in ((bad_cfg, prov), (pool_cfg, dict(prov, encoder="x.pt")), (pool_cfg, dict(prov, readout="other")),
                 (pool_cfg, dict(prov, pool_source_sha256=None))):
        with pytest.raises(AssertionError):
            cc.assert_motion(c, run, p)
    with pytest.raises(Exception):                                           # 旧 encoder 条目遇池化快照仍要 source_run
        cc.assert_motion(pool_cfg, cc.RUNS["robomme-vla-modul-motion-80k-v1"], prov)
