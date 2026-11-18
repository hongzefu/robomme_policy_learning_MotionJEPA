"""F4：同卡多片并发起跑时 run_shard.sh 进程归属检查误判退出（G5b a02 g5b-0 exit 2）的回归单测。CPU、无 GPU。

根因两处：
* run_shard.sh::check_local_gpu 拿 nvidia-smi 列出的 PID 沿 /proc/<pid>/status 回溯 PPid；同卡别片的短命 GPU 进程
  （render_probe 等）在「查询之后、回溯之前」退出时 /proc 已无此 PID，回溯得到空串即被当成外来进程拒绝。
  修法：回溯不到且该 PID 已不在 /proc（进程已退出、不再占卡）时跳过；仍存活的照旧拒绝。
* local_692_pool_e2e.sh::run_slot 先后台起 run_shard.sh 再把 pending 改写成 PID，中间有「已起跑仍是 pending」窗口。
  修法：子 shell 先登记自身 $BASHPID 再 exec，run_shard.sh 起跑时登记已在。
这里用假 nvidia-smi + 真子进程构造同卡 4 片并发的时序，证明：不再误判、真正外来存活进程仍被拒、旧函数在同输入上必误判。
运行：UV_PROJECT_ENVIRONMENT=<主 .venv> uv run --no-sync python -m pytest scripts/training/tests/test_f4_gpu_slot_race.py -q
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import textwrap
import time

import pytest

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / "scripts/evaluation"
OLD = "ca98ad2"   # F4 修复前基线


def _extract_func(text: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{[^\n]*\n.*?^\}}\n", text, re.M | re.S)
    assert m, name
    return m.group(0)


def _old_text(path: str) -> str:
    return subprocess.check_output(["git", "show", f"{OLD}:{path}"], cwd=REPO, text=True)


@pytest.fixture
def fake_smi(tmp_path):
    """假 nvidia-smi：打印 $FAKE_GPU_PIDS_FILE 的内容（每行一个 PID），模拟驱动列表滞后于 /proc。"""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "nvidia-smi").write_text('#!/usr/bin/env bash\ncat "${FAKE_GPU_PIDS_FILE:-/dev/null}"\n')
    (bindir / "nvidia-smi").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("SLURM_JOB_ID",)}
    env.update(PATH=f"{bindir}:{os.environ['PATH']}", CUDA_VISIBLE_DEVICES="0", LOCAL_GPU_SHARE="4")
    return env


def _spawn_root():
    """模拟一个已登记的 run_shard.sh：bash 根 + 一个子进程（代表其 server / render_probe）。返回 (根, 子 PID)。"""
    root = subprocess.Popen(["bash", "-c", "sleep 60 & wait"])
    for _ in range(200):
        kids = Path(f"/proc/{root.pid}/task/{root.pid}/children").read_text().split()
        if kids:
            return root, int(kids[0])
        time.sleep(0.01)
    raise AssertionError("子进程未起")


def _dead_pid() -> int:
    p = subprocess.Popen(["true"])
    p.wait()
    assert not Path(f"/proc/{p.pid}").exists()
    return p.pid


def _check(tmp: Path, func: str, env: dict, tag: str):
    script = tmp / f"check-{tag}.sh"
    script.write_text("set -euo pipefail\n" + func + "\ncheck_local_gpu\n")
    return subprocess.run(["bash", str(script)], env=dict(env, SHARD_TAG=tag), capture_output=True, text=True,
                          timeout=30)


def test_check_local_gpu_vanished_pid_vs_stranger(tmp_path, fake_smi):
    new_func = _extract_func((EVAL / "run_shard.sh").read_text(), "check_local_gpu")
    old_func = _extract_func(_old_text("scripts/evaluation/run_shard.sh"), "check_local_gpu")
    roots = [_spawn_root() for _ in range(4)]
    stranger = subprocess.Popen(["sleep", "60"])
    try:
        slot = tmp_path / "gpu0"
        slot.write_text("".join(f"g5b-{i} {r.pid}\n" for i, (r, _) in enumerate(roots)))
        pids = tmp_path / "pids"
        env = dict(fake_smi, LOCAL_GPU_SLOT_FILE=str(slot), FAKE_GPU_PIDS_FILE=str(pids))
        live_kids = [str(k) for _, k in roots[1:]]
        dead = _dead_pid()
        # ① 别片存活子进程 + 刚退出的别片 GPU 进程（G5b a02 g5b-0 的时序）：新函数放行，旧函数误判
        pids.write_text("\n".join(live_kids + [str(dead)]) + "\n")
        n, o = _check(tmp_path, new_func, env, "g5b-0"), _check(tmp_path, old_func, env, "g5b-0")
        assert n.returncode == 0, n.stdout + n.stderr
        assert "LOCAL_GPU_SHARE=PASS device=0 share=4 registered=4 gpu_procs=4" in n.stdout
        assert o.returncode == 2 and f"计算进程 {dead} 不属于已登记分片的进程树" in o.stdout
        # ② 真正外来的存活进程：新函数照旧拒绝（无论是否夹着已退出 PID）
        for extra in ([str(stranger.pid)], [str(dead), str(stranger.pid)]):
            pids.write_text("\n".join(live_kids + extra) + "\n")
            bad = _check(tmp_path, new_func, env, "g5b-0")
            assert bad.returncode == 2 and f"计算进程 {stranger.pid} 不属于已登记分片的进程树" in bad.stdout
        # ③ 本片未登记 / 超额登记：判据不变
        slot.write_text("".join(f"g5b-{i} {r.pid}\n" for i, (r, _) in enumerate(roots[1:], 1)))
        pids.write_text("")
        assert "未在占位文件登记" in _check(tmp_path, new_func, env, "g5b-0").stdout
        slot.write_text("".join(f"g5b-{i} {r.pid}\n" for i, (r, _) in enumerate(roots)) + "g5b-4 1\n")
        assert "> 4" in _check(tmp_path, new_func, env, "g5b-0").stdout
    finally:
        for r, _ in roots:
            r.kill()
        stranger.kill()


SHARD_SIM = r'''
# 假分片：反复「起短命 GPU 子进程 → 登记进驱动列表 → 子进程退出」，同时做归属检查；驱动列表里退出的 PID 不删
# （模拟 nvidia-smi 滞后），任何一次检查失败即退出 2。
set -euo pipefail
@FUNC@
for i in $(seq 1 @ITERS@); do
    sleep 0.02 & k=$!
    ( flock 9; echo "$k" >> "$FAKE_GPU_PIDS_FILE" ) 9>"$FAKE_GPU_PIDS_FILE.lock"
    check_local_gpu > /dev/null || { echo "SIM_FAIL tag=$SHARD_TAG iter=$i"; exit 2; }
    wait "$k" || true
done
echo "SIM_OK tag=$SHARD_TAG"
'''


def _run_sim(tmp: Path, func: str, env: dict, iters: int = 15) -> list[subprocess.CompletedProcess]:
    """同卡 4 片同时起：每片自身就是登记根（$$），并发做 iters 次检查。"""
    slot = tmp / "gpu0"
    pids = tmp / "pids"
    pids.write_text("")
    script = tmp / "sim.sh"
    script.write_text(SHARD_SIM.replace("@FUNC@", func).replace("@ITERS@", str(iters)))
    procs = []
    lines = []
    for i in range(4):   # 先登记全部 4 片再放行（等价于 run_slot 的先登记后起跑）
        p = subprocess.Popen(["bash", "-c", 'read -r _; exec bash "$0"', str(script)], stdin=subprocess.PIPE,
                             env=dict(env, SHARD_TAG=f"g5b-{i}", LOCAL_GPU_SLOT_FILE=str(slot),
                                      FAKE_GPU_PIDS_FILE=str(pids)),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append(p)
        lines.append(f"g5b-{i} {p.pid}\n")
    slot.write_text("".join(lines))
    for p in procs:
        p.stdin.write("go\n")
        p.stdin.close()
    out = []
    for p in procs:
        stdout = p.stdout.read()
        p.wait(timeout=120)
        out.append(subprocess.CompletedProcess(p.args, p.returncode, stdout, ""))
    return out


def test_same_gpu_4_shards_concurrent_no_false_reject(tmp_path, fake_smi):
    new_func = _extract_func((EVAL / "run_shard.sh").read_text(), "check_local_gpu")
    (tmp_path / "new").mkdir()
    res = _run_sim(tmp_path / "new", new_func, fake_smi)
    assert all(r.returncode == 0 and "SIM_OK" in r.stdout for r in res), [r.stdout for r in res]
    # 对照：旧函数在同样时序下必然误判（驱动列表里留有已退出 PID）
    old_func = _extract_func(_old_text("scripts/evaluation/run_shard.sh"), "check_local_gpu")
    (tmp_path / "old").mkdir()
    old = _run_sim(tmp_path / "old", old_func, fake_smi)
    assert any(r.returncode == 2 and "SIM_FAIL" in r.stdout for r in old), [r.stdout for r in old]


def test_run_slot_registers_pid_before_run_shard_starts(tmp_path):
    """local_692_pool_e2e.sh::run_slot：同卡 4 片同时起，每个 run_shard.sh 第一行就能在占位文件里看到「<tag> <$$>」。"""
    text = (EVAL / "local_692_pool_e2e.sh").read_text()
    funcs = "".join(_extract_func(text, n) for n in ("slot_reserve", "slot_set_pid", "slot_release", "run_slot"))
    ev = tmp_path / "scripts/evaluation"
    ev.mkdir(parents=True)
    # 假 run_shard.sh：起跑第一件事读占位文件，记录自身 $$ 与是否已登记、有无 pending，然后停留片刻让 4 片重叠
    (ev / "run_shard.sh").write_text(textwrap.dedent('''\
        #!/usr/bin/env bash
        own=0; grep -qx "$5 $$" "$LOCAL_GPU_SLOT_FILE" && own=1
        pend=$(grep -c " pending$" "$LOCAL_GPU_SLOT_FILE" || true)
        echo "FAKE_SHARD tag=$5 pid=$$ own=$own pending_seen=$pend gpu=$CUDA_VISIBLE_DEVICES"
        sleep 0.3
    '''))
    logs = tmp_path / "logs"
    logs.mkdir()
    driver = tmp_path / "driver.sh"
    driver.write_text("set -uo pipefail\n" + f'cd "{tmp_path}"\nSLOT_DIR="{tmp_path}/slots"; mkdir -p "$SLOT_DIR"\n'
                      'PER_GPU=4; CKPT=/nonexistent/19; SEED=7\n' + funcs + textwrap.dedent(f'''
        for I in 0 1 2 3; do
            run_slot 0 "g5b-$I" "$I" plan.json run "{logs}/g5b-$I.log" "{logs}/g5b-$I.rc" &
        done
        wait
        run_slot 0 g5b-4 4 plan.json run "{logs}/g5b-4.log" "{logs}/g5b-4.rc"
    '''))
    out = subprocess.run(["bash", str(driver)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    reg = dict(re.findall(r"^SLOT_REGISTERED gpu=0 tag=(\S+) pid=(\d+) ", out.stdout, re.M))
    assert sorted(reg) == [f"g5b-{i}" for i in range(5)]
    for i in range(5):
        log = (logs / f"g5b-{i}.log").read_text()
        m = re.search(r"FAKE_SHARD tag=(\S+) pid=(\d+) own=(\d) pending_seen=(\d+) gpu=0", log)
        assert m, log
        assert m.group(2) == reg[f"g5b-{i}"]     # $! == run_shard.sh 的 $$（exec 不换 PID）
        assert m.group(3) == "1"                 # 起跑时本片已按真实 PID 登记
        assert (logs / f"g5b-{i}.rc").read_text().strip() == "0"
    assert (tmp_path / "slots/gpu0").read_text().strip() == ""   # 全部释放
