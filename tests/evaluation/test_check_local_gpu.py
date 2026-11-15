"""1114 计划 S4-04 / S4-05：run_shard.sh::check_local_gpu 进程归属的三个场景（CPU，假 nvidia-smi）。

摘自 c0826be scripts/training/tests/test_s3_eval_local.py 的 gpu_env、_extract_func 与 test_c4_check_local_gpu 三段，
去掉对 BASE 旧函数的逐字对照，只验 LOCAL_GPU_SHARE>=2 分支的进程归属：
* PPid 读空不判外来：短命进程在读 /proc 前已退出（nvidia-smi 首次仍列出、重采样后消失），重采样后放行；
* 孤儿被收养到 PID 1 且属本仓库评估（environ 带同一 EVAL_REPO）不判外来；
* 真外来进程（链终止于 PID 1、environ 与 cwd 都不属本仓库）3 次采样后仍被拒。
另保留登记进程树放行、第 3 片被拒、本片未登记被拒三条原有行为。
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import time

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN_SHARD = REPO / "scripts/evaluation/run_shard.sh"


def _extract_func(text: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{[^\n]*\n.*?^\}}\n", text, re.M | re.S)
    assert m, name
    return m.group(0)


def _run_func(tmp: Path, func: str, env: dict, call: str = "check_local_gpu"):
    script = tmp / "case.sh"
    script.write_text("set -euo pipefail\n" + func + f"\n{call}\n")
    return subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=60)


@pytest.fixture
def gpu_env(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    # 假 nvidia-smi：FAKE_PIDS_FIRST 只在首次调用时输出（模拟短命进程），其后输出 FAKE_PIDS
    (bindir / "nvidia-smi").write_text(
        '#!/usr/bin/env bash\n'
        'c="$FAKE_COUNTER"; n=$(cat "$c" 2>/dev/null || echo 0); echo $((n + 1)) > "$c"\n'
        'if [[ "$n" == 0 && -n "${FAKE_PIDS_FIRST:-}" ]]; then printf "%s" "$FAKE_PIDS_FIRST"; else printf "%s" "${FAKE_PIDS:-}"; fi\n')
    (bindir / "nvidia-smi").chmod(0o755)
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("SLURM_JOB_ID", "LOCAL_GPU_SHARE", "EVAL_REPO")}
    env.update(PATH=f"{bindir}:{os.environ['PATH']}", CUDA_VISIBLE_DEVICES="3", SHARD_TAG="s3",
               EVAL_REPO=str(repo), FAKE_COUNTER=str(tmp_path / "counter"))
    return env


@pytest.fixture
def func():
    return _extract_func(RUN_SHARD.read_text(encoding="utf-8"), "check_local_gpu")


def _share(tmp_path, gpu_env, slot_text, **extra):
    slot = tmp_path / "gpu3"
    slot.write_text(slot_text)
    (tmp_path / "counter").unlink(missing_ok=True)
    return dict(gpu_env, LOCAL_GPU_SHARE="4", LOCAL_GPU_SLOT_FILE=str(slot), **extra)


def _orphan(env: dict, cwd: Path) -> int:
    """双 fork 造孤儿（父进程立即退出，被收养到 PID 1 或子收割者），返回孤儿 PID。"""
    out = subprocess.run(["bash", "-c", "sleep 30 >/dev/null 2>&1 & echo $!"], env=env, cwd=cwd,
                         capture_output=True, text=True, timeout=10)
    return int(out.stdout.strip())


def test_registered_tree_and_slot_rules(tmp_path, gpu_env, func):
    root = subprocess.Popen(["bash", "-c", "sleep 30 & wait"])
    try:
        child = None
        for _ in range(100):
            kids = Path(f"/proc/{root.pid}/task/{root.pid}/children").read_text().split()
            if kids:
                child = kids[0]
                break
            time.sleep(0.05)
        assert child
        ok = _run_func(tmp_path, func, _share(tmp_path, gpu_env, f"s3 pending\ns7 {root.pid}\n", FAKE_PIDS=child))
        assert ok.returncode == 0 and "LOCAL_GPU_SHARE=PASS device=3 share=4 registered=2 gpu_procs=1" in ok.stdout
        assert "LOCAL_GPU_FOREIGN_SAMPLE" not in ok.stdout                     # 登记树内：首采即放行
        many = "".join(f"s{k} {k}\n" for k in (3, 11, 19, 27, 35))
        third = _run_func(tmp_path, func, _share(tmp_path, gpu_env, many))
        assert third.returncode == 2 and "> 4" in third.stdout                 # 超出每卡片数
        unreg = _run_func(tmp_path, func, _share(tmp_path, gpu_env, f"s7 {root.pid}\n"))
        assert unreg.returncode == 2 and "未在占位文件登记" in unreg.stdout
        assert "LOCAL_GPU_IDLE=PASS device=3" in _run_func(tmp_path, func, dict(gpu_env)).stdout   # 默认 =1 分支不变
    finally:
        root.kill()


def test_ppid_unreadable_not_foreign(tmp_path, gpu_env, func):
    gone = subprocess.Popen(["true"])
    gone.wait()                                                               # 已退出：/proc/<pid>/status 读空
    env = _share(tmp_path, gpu_env, "s3 pending\n", FAKE_PIDS_FIRST=str(gone.pid), FAKE_PIDS="")
    out = _run_func(tmp_path, func, env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"LOCAL_GPU_FOREIGN_SAMPLE attempt=1/3 pids={gone.pid}" in out.stdout   # 首采判不了 → 重采样
    assert "LOCAL_GPU_SHARE=PASS" in out.stdout and "attempt=2/3" not in out.stdout


def test_orphan_of_own_eval_not_foreign(tmp_path, gpu_env, func):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    pid = _orphan(dict(gpu_env), elsewhere)                                   # environ 带同一 EVAL_REPO
    try:
        out = _run_func(tmp_path, func, _share(tmp_path, gpu_env, "s3 pending\n", FAKE_PIDS=str(pid)))
        assert out.returncode == 0, out.stdout + out.stderr
        assert f"LOCAL_GPU_OWN_ORPHAN pid={pid} via=environ" in out.stdout
        assert "LOCAL_GPU_FOREIGN_SAMPLE" not in out.stdout
        # environ 不带 EVAL_REPO、但 cwd 在本仓库内：同样不判外来
        env2 = {k: v for k, v in gpu_env.items() if k != "EVAL_REPO"}
        sub = Path(gpu_env["EVAL_REPO"]) / "sub"
        sub.mkdir()
        pid2 = _orphan(env2, sub)
        try:
            out2 = _run_func(tmp_path, func, _share(tmp_path, gpu_env, "s3 pending\n", FAKE_PIDS=str(pid2)))
            assert out2.returncode == 0 and f"LOCAL_GPU_OWN_ORPHAN pid={pid2} via=cwd" in out2.stdout
        finally:
            os.kill(pid2, 9)
    finally:
        os.kill(pid, 9)


def test_true_foreign_rejected_after_three_samples(tmp_path, gpu_env, func):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    env = {k: v for k, v in gpu_env.items() if k != "EVAL_REPO"}
    env["EVAL_REPO"] = str(tmp_path / "other-repo")                          # 别的仓库的评估
    pid = _orphan(env, elsewhere)
    stranger = subprocess.Popen(["sleep", "30"], cwd=elsewhere,
                                env={k: v for k, v in gpu_env.items() if k != "EVAL_REPO"})
    try:
        out = _run_func(tmp_path, func, _share(tmp_path, gpu_env, "s3 pending\n", FAKE_PIDS=f"{pid}\n{stranger.pid}"))
        assert out.returncode == 2, out.stdout + out.stderr
        for k in (1, 2, 3):
            assert f"LOCAL_GPU_FOREIGN_SAMPLE attempt={k}/3" in out.stdout
        assert "3 次采样均判外来，停止起跑" in out.stdout
        assert "LOCAL_GPU_SHARE=PASS" not in out.stdout
    finally:
        os.kill(pid, 9)
        stranger.kill()
