"""共享任务池（pool_make.py / pool_runner.sh / gl_hold_pool.sh）的 CPU 测试：用假的 run_local（bash 脚本）代替真实评估。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from scripts.interpret import pool_make as PM

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts/interpret/pool_runner.sh"
HOLD = REPO / "scripts/interpret/gl_hold_pool.sh"
POOL_MAKE = REPO / "scripts/interpret/pool_make.py"

# 假 run_local：记录 "<spec 名> <cuda> <frac> <pid>" 到计数文件，按 spec 里的 fake_sleep 睡、按 fake_rc 退出
FAKE = r"""#!/usr/bin/env bash
[ "$1" = "--spec" ] || { echo "bad args $*"; exit 64; }
read -r name rc slp cuda frac < <(python3 -c '
import json,sys; s=json.load(open(sys.argv[1]))
print(s["fake_name"], s.get("fake_rc",0), s.get("fake_sleep",0), s.get("cuda"), s.get("xla_mem_fraction"))' "$2")
echo "$name $cuda $frac $$" >> "$FAKE_COUNTER"
sleep "$slp"
echo "fake run $name rc=$rc"
exit "$rc"
"""


def write_specs(root: Path, stage: str, names, *, site_suffix=True, rc=None, sleep=0.0, j_in=None, diff=None):
    """写 GL 与 aspen 两套 spec，返回 (gl_glob, aspen_glob)。"""
    gl, asp = root / stage / "gl", root / stage / "aspen"
    gl.mkdir(parents=True), asp.mkdir(parents=True)
    for name in names:
        base = {"fake_name": name, "fake_rc": (rc or {}).get(name, 0), "fake_sleep": sleep,
                "cuda": "0", "xla_mem_fraction": 0.27}
        if j_in and name in j_in:
            base["j"] = j_in[name]
        if diff and name in diff:
            man = root / stage / f"{name}.manifest.json"
            man.write_text(json.dumps({"episodes": [{"difficulty": d} for d in diff[name]]}))
            base["manifest"] = str(man)
        g = gl / (f"{name}.gl.json" if site_suffix else f"{name}.json")
        a = asp / (f"{name}.aspen.json" if site_suffix else f"{name}.json")
        g.write_text(json.dumps({**base, "site": "gl"}))
        a.write_text(json.dumps({**base, "site": "aspen"}))
    return str(gl / "*.json"), str(asp / "*.json")


def make_pool(out: Path, stage_specs, pool_name=None, cost_json=None):
    argv = ["make", "--out", str(out)]
    for st in stage_specs:
        argv += ["--stage-spec", *st]
    if pool_name:
        argv += ["--pool-name", pool_name]
    if cost_json:
        argv += ["--cost-json", str(cost_json)]
    r = subprocess.run([sys.executable, str(POOL_MAKE), *argv], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    assert re.search(r"^POOL_MAKE=PASS ", r.stdout, re.M), r.stdout
    return r.stdout


@pytest.fixture
def fake(tmp_path):
    f = tmp_path / "fake_run_local.sh"
    f.write_text(FAKE)
    counter = tmp_path / "counter.txt"
    counter.write_text("")
    return f, counter


def run_runner(out, stages, site, fake_path, counter, extra=(), env_extra=None, wait=True):
    cmd = ["bash", str(RUNNER), "--pool-root", str(out), "--site", site, "--run-cmd", f"bash {fake_path}"]
    for st in stages:
        cmd += ["--stage", st]
    cmd += list(extra)
    env = {**os.environ, "FAKE_COUNTER": str(counter), **(env_extra or {})}
    if wait:
        return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)


# ---------------------------------------------------------------- 静态
@pytest.mark.parametrize("script", [RUNNER, HOLD])
def test_bash_syntax(script):
    r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_inner_never_calls_srun():
    text = HOLD.read_text()
    m = re.search(r"^# BEGIN_INNER[^\n]*\n(.*?)^# END_INNER", text, re.S | re.M)
    assert m, "找不到 BEGIN_INNER/END_INNER 段"
    inner = m.group(1)
    assert "run_inner()" in inner
    for line in inner.splitlines():
        code = line.split("#", 1)[0].strip()
        if "srun" not in code:
            continue
        # 只允许把 srun 覆盖成报错函数并导出，不允许任何调用
        assert code.startswith("srun()") or code == "export -f srun", line
    # 外层只有 gl 分支起一次 srun，且带 R1 要求的全部参数
    calls = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#") and re.search(r"\bsrun --jobid", ln)]
    assert len(calls) == 1 and "--overlap --exact --ntasks=1" in calls[0] and "--gpu_cmode=shared" in calls[0]


# ---------------------------------------------------------------- 成本排序
def test_cost_order_and_layout(tmp_path):
    names = ["a-j1-x", "b-j6-x", "c-j3-x", "d"]
    diff = {"a-j1-x": ["easy"], "b-j6-x": ["easy"], "c-j3-x": ["xhard", "hard"], "d": ["medium"]}
    g, a = write_specs(tmp_path / "src", "samep", names, diff=diff)
    out = tmp_path / "O"
    make_pool(out, [("samep", g, a)])
    pool = out / "pool" / "samep"
    todo = sorted(p.name for p in (pool / "todo").iterdir())
    # c: (150+120)*(1+0.75)=472.5 > b: 60*2.5=150 > d: 90 > a: 75
    assert todo == ["000000-c-j3-x.aspen.json", "000000-c-j3-x.gl.json", "000001-b-j6-x.aspen.json",
                    "000001-b-j6-x.gl.json", "000002-d.aspen.json", "000002-d.gl.json",
                    "000003-a-j1-x.aspen.json", "000003-a-j1-x.gl.json"]
    man = json.loads((pool / "manifest.json").read_text())
    assert [e["cost_seconds"] for e in man["entries"]] == [472.5, 150.0, 90.0, 75.0]
    assert json.loads((pool / "todo/000000-c-j3-x.gl.json").read_text())["site"] == "gl"
    assert json.loads((pool / "todo/000000-c-j3-x.aspen.json").read_text())["site"] == "aspen"


def test_cost_json_and_multi_stage_merge(tmp_path):
    g1, a1 = write_specs(tmp_path / "src", "noise", ["n1", "n2"])
    g2, _ = write_specs(tmp_path / "src", "samep", ["s1-j2-q"], site_suffix=False)
    cost = tmp_path / "cost.json"
    cost.write_text(json.dumps({"stage_mult": {"noise": 3.0}, "spec_seconds": {"n2": 1}}))
    out = tmp_path / "O"
    stdout = make_pool(out, [("noise", g1, a1), ("samep", g2, "-")], pool_name="mix", cost_json=cost)
    assert "stage=noise,samep specs=3 gl=3 aspen=2" in stdout
    man = json.loads((out / "pool/mix/manifest.json").read_text())
    # n1: 90*3=270 > s1: 90*1.5=135 > n2: 1
    assert [(e["name"], e["stage"]) for e in man["entries"]] == [("n1", "noise"), ("s1-j2-q", "samep"), ("n2", "noise")]
    assert not list((out / "pool/mix/todo").glob("000001-*.aspen.json"))   # 只有 GL 版


def test_pool_make_refuses_overwrite_and_duplicates(tmp_path):
    g, a = write_specs(tmp_path / "src", "n", ["x"])
    out = tmp_path / "O"
    make_pool(out, [("n", g, a)])
    with pytest.raises(SystemExit, match="拒绝覆盖"):
        PM.main(["make", "--out", str(out), "--stage-spec", "n", g, a])
    with pytest.raises(SystemExit, match="池内键必须唯一"):
        PM.main(["make", "--out", str(tmp_path / "O2"), "--stage-spec", "n", g, a, "--stage-spec", "m", g, "-"])


def test_wide_glob_skips_other_site_suffix(tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    for n in ("x", "y"):
        (d / f"{n}.gl.json").write_text(json.dumps({"site": "gl"}))
        (d / f"{n}.json").write_text(json.dumps({"site": "aspen"}))
    out = tmp_path / "O"
    stdout = make_pool(out, [("m", str(d / "*.gl.json"), str(d / "*.json"))])
    assert "specs=2 gl=2 aspen=2" in stdout
    for p in (out / "pool/m/todo").iterdir():
        assert json.loads(p.read_text())["site"] == p.name.split(".")[-2]


def test_j_parsing():
    assert PM.spec_j({}, "samep-g1b-j3-Sside-024") == 3
    assert PM.spec_j({}, "vrec-s022") == 0
    assert PM.spec_j({"j": 5}, "x-j3-y") == 5
    assert PM.spec_name("/a/000-x.gl.json") == "000-x" and PM.spec_name("/a/y.json") == "y"


# ---------------------------------------------------------------- 领取与执行
def test_concurrent_claims_each_spec_exactly_once(tmp_path, fake):
    fake_path, counter = fake
    names = [f"s{i:02d}" for i in range(40)]
    rc = {"s07": 3, "s21": 1}
    g, a = write_specs(tmp_path / "src", "st", names, rc=rc, sleep=0.05)
    out = tmp_path / "O"
    make_pool(out, [("st", g, a)])
    procs = [run_runner(out, ["st"], "gl" if i % 2 else "aspen", fake_path, counter,
                        extra=["--channel", f"ch{i}"], wait=False) for i in range(8)]
    outs = [p.communicate(timeout=120)[0] for p in procs]
    lines = counter.read_text().split()
    ran = [lines[i] for i in range(0, len(lines), 4)]
    assert sorted(ran) == sorted(names), "每个 spec 恰好执行一次"
    pool = out / "pool/st"
    assert not list((pool / "todo").iterdir())
    done = {p.name.split("-", 1)[1][: -len(".result")] for p in (pool / "done").glob("*.result")}
    fail = {p.name.split("-", 1)[1][: -len(".result")] for p in (pool / "fail").glob("*.result")}
    assert fail == set(rc) and done == set(names) - set(rc) and not (done & fail)
    # 失败 spec 的两套 json 都进 fail/，结果文件带 rc 与日志
    for name, code in rc.items():
        res = next((pool / "fail").glob(f"*-{name}.result")).read_text()
        assert f"rc={code}" in res
        log = Path(re.search(r"^log=(.*)$", res, re.M).group(1))
        assert log.is_file() and f"fake run {name}" in log.read_text()
        assert len(list((pool / "fail").glob(f"*-{name}.*.json"))) == 2
    assert len(list((pool / "claim").iterdir())) == 40
    # 每个通道写 EXIT_CODE；有失败的通道 EXIT_CODE=1
    total_end = 0
    for o in outs:
        assert re.search(r"^EXIT_CODE=[01]$", o, re.M), o
        total_end += len(re.findall(r"^POOL_SPEC_END ", o, re.M))
        assert len(re.findall(r"^POOL_SPEC_START ", o, re.M)) == len(re.findall(r"^POOL_SPEC_END ", o, re.M))
    assert total_end == 40
    st = PM.pool_status(pool)
    assert st["todo"] == 0 and st["running"] == 0 and st["done"] == 38 and st["fail"] == 2


def test_claim_order_and_stage_sequence_and_overrides(tmp_path, fake):
    fake_path, counter = fake
    g1, a1 = write_specs(tmp_path / "src", "first", ["f-j1-a", "f-j5-b"])
    g2, a2 = write_specs(tmp_path / "src", "second", ["z"])
    out = tmp_path / "O"
    make_pool(out, [("first", g1, a1)])
    make_pool(out, [("second", g2, a2)])
    log = tmp_path / "runner.log"
    r = run_runner(out, ["first", "second"], "aspen", fake_path, counter,
                   extra=["--gpu", "1", "--mem-frac", "0.31", "--log", str(log)])
    assert r.returncode == 0, r.stdout + r.stderr
    rows = [ln.split() for ln in counter.read_text().splitlines()]
    # 先长后短（j5 先于 j1），前一个池领空才领下一个；cuda / 显存比例被覆盖
    assert [x[0] for x in rows] == ["f-j5-b", "f-j1-a", "z"]
    assert all(x[1] == "1" and x[2] == "0.31" for x in rows)
    assert log.read_text().rstrip().endswith("EXIT_CODE=0")
    # 原 spec 未被修改（覆盖写在 claim/<键>/ 下）
    done = json.loads(next((out / "pool/first/done").glob("*-f-j5-b.aspen.json")).read_text())
    assert done["cuda"] == "0"
    assert json.loads(next((out / "pool/first/claim").glob("*-f-j5-b/spec.aspen.json")).read_text())["cuda"] == "1"


def test_site_only_specs_and_stop_file(tmp_path, fake):
    fake_path, counter = fake
    g, _ = write_specs(tmp_path / "src", "st", ["only-gl"])
    out = tmp_path / "O"
    make_pool(out, [("st", g, "-")])
    r = run_runner(out, ["st"], "aspen", fake_path, counter)
    assert r.returncode == 0 and "ran=0" in r.stdout and counter.read_text() == ""
    (out / "pool/st/STOP").write_text("")
    r = run_runner(out, ["st"], "gl", fake_path, counter)
    assert "POOL_STOP_SEEN" in r.stdout and counter.read_text() == ""
    (out / "pool/st/STOP").unlink()
    r = run_runner(out, ["st"], "gl", fake_path, counter)
    assert r.returncode == 0 and counter.read_text().split()[0] == "only-gl"


def test_env_file_failure_reports_exit_code(tmp_path, fake):
    fake_path, counter = fake
    g, a = write_specs(tmp_path / "src", "st", ["x"])
    out = tmp_path / "O"
    make_pool(out, [("st", g, a)])
    env = tmp_path / "env-bad.sh"
    env.write_text('echo "ENV_CHECK=FAIL reason=head"; return 2\n')
    r = run_runner(out, ["st"], "gl", fake_path, counter, extra=["--env-file", str(env)])
    assert r.returncode == 2 and "EXIT_CODE=2" in r.stdout and counter.read_text() == ""
    assert not list((out / "pool/st/claim").iterdir())


def test_env_file_sets_py_and_frac(tmp_path, fake):
    """不给 --run-cmd 时用 env 文件设置的 $py 跑 -m scripts.interpret.run_local；这里让 $py 指向假解释器。"""
    _, counter = fake
    g, a = write_specs(tmp_path / "src", "st", ["x"])
    out = tmp_path / "O"
    make_pool(out, [("st", g, a)])
    fakepy = tmp_path / "fakepy.sh"
    fakepy.write_text('#!/usr/bin/env bash\necho "$PWD|$*|$XLA_PYTHON_CLIENT_MEM_FRACTION|$CUDA_VISIBLE_DEVICES" >> "$FAKE_COUNTER"\n')
    fakepy.chmod(0o755)
    env = tmp_path / "env-ok.sh"
    env.write_text(f'gpu=$1; frac=${{2:-0.27}}\nexport CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_MEM_FRACTION=$frac\n'
                   f'py={fakepy}\ncd {tmp_path}\n')
    cmd = ["bash", str(RUNNER), "--pool-root", str(out), "--site", "gl", "--stage", "st", "--env-file", str(env),
           "--gpu", "0", "--mem-frac", "0.22", "--log", "rel.log"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path / "src",
                       env={**os.environ, "FAKE_COUNTER": str(counter), "POOL_JSON_PY": sys.executable})
    assert r.returncode == 0, r.stdout + r.stderr
    cwd, args, frac, cuda = counter.read_text().strip().split("|")
    assert cwd == str(tmp_path) and args.startswith("-m scripts.interpret.run_local --spec ") and frac == "0.22" and cuda == "0"
    assert (tmp_path / "src/rel.log").read_text().rstrip().endswith("EXIT_CODE=0")   # 相对 --log 按启动目录解析


# ---------------------------------------------------------------- 启动器
def test_hold_pool_inner_runs_channels(tmp_path, fake):
    fake_path, counter = fake
    names = [f"s{i}" for i in range(9)]
    g, a = write_specs(tmp_path / "src", "st", names, rc={"s4": 2}, sleep=0.05)
    out = tmp_path / "O"
    make_pool(out, [("st", g, a)])
    logs = tmp_path / "logs"
    cmd = ["bash", str(HOLD), "aspen", "--gpu", "1", "--channels", "3", "--mem-frac", "0.3", "--pool-root", str(out),
           "--stage", "st", "--log-dir", str(logs), "--stagger", "0", "--name", "t-hold", "--no-tmux",
           "--run-cmd", f"bash {fake_path}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                       env={**os.environ, "FAKE_COUNTER": str(counter), "POOL_JSON_PY": sys.executable})
    assert r.returncode == 0, r.stdout + r.stderr
    session = (logs / "t-hold.log").read_text()
    assert "POOL_HOLD_DONE name=t-hold channels=3 failed=1" in session
    assert session.rstrip().endswith("EXIT_CODE=1")
    assert len(re.findall(r"^POOL_CHANNEL_END ", session, re.M)) == 3
    ran = [ln.split()[0] for ln in counter.read_text().splitlines()]
    assert sorted(ran) == sorted(names)
    assert {ln.split()[1] for ln in counter.read_text().splitlines()} == {"1"}
    for i in range(3):
        assert re.search(r"^EXIT_CODE=[01]$", (logs / f"t-hold-c{i}.log").read_text(), re.M)


def test_hold_pool_gl_dry_run_builds_single_step(tmp_path):
    cmd = ["bash", str(HOLD), "gl", "--hold", "12345", "--channels", "3", "--mem-frac", "0.27", "--pool-root",
           str(tmp_path / "O"), "--stage", "a", "--stage", "b", "--env-file", str(tmp_path / "env-gl.sh"),
           "--log-dir", str(tmp_path / "logs"), "--dry-run"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("DRY_RUN cmd="))
    assert line.count("srun ") == 1
    assert "srun --jobid=12345 --overlap --exact --ntasks=1 --cpus-per-task=6 --gpu_cmode=shared bash " in line
    assert " inner --site gl --gpu 0 --channels 3 " in line and "--stage a --stage b" in line
    assert "name=s5t-pool-12345" in r.stdout


def test_hold_pool_rejects_bad_channels(tmp_path):
    r = subprocess.run(["bash", str(HOLD), "gl", "--hold", "1", "--channels", "5", "--mem-frac", "0.2",
                        "--pool-root", str(tmp_path), "--stage", "a", "--log-dir", str(tmp_path), "--dry-run"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "1–4" in r.stderr
