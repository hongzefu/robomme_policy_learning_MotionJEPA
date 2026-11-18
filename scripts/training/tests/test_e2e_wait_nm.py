"""S5a 触发器单测（1117-e2e-enc-train-plan.html 第二部分〇.1.1「验收用例（S5 单测）」、二节 S5 行）。

只跑 CPU、秒级；nvidia-smi 与 tmux 用 PATH 前置的假脚本模拟（状态放在临时目录的文件里），
测试绝不起真 tmux、不碰 GPU，也不写 v1-store（锁目录用 --lock-dir 指向临时目录）。

A. NM_DONE 判定夹具：全满足 → NM_DONE=PASS；末行 SHARD_EXIT_CODE=0 干扰行、会话仍在 → 不 PASS（等待）；
   NM_PIPELINE=FAIL、末行 EXIT_CODE 非 0 → NM_DONE=FAIL 且不起任何会话；--dry-run 只打印 TRIG_DRYRUN、不起会话不建锁。
B. 〇.1.1 四个用例：
   ① NM 日志写完 EXIT_CODE=0 后不再增长，之后只改变模拟的 tmux、GPU 与前置状态，触发器仍在定时重查内
     恰好启动一次目标闸门（TRIG_LAUNCH 恰 1 行、锁文件 1 个）；
   ② 启动后杀掉触发器再重启，不重复启动（只多 TRIG_SKIP，TRIG_LAUNCH 仍为 1）；
   ③ 模拟 GPU memory.used 之和 >0 时只打印 TRIG_WAIT … missing=gpu_busy、不启动；
   ④ G3 会话仍在时 G4 不启动。
末尾 test_zz_verdict 在四个用例全部通过时打印
  E2E_TRIG_SELFTEST=PASS cases=4 launches_exact_once=PASS restart_no_dup=PASS（pytest -s 可见）。
"""
from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/training/e2e_wait_nm.sh"
INTERVAL = "0.2"
CASES: dict[str, bool] = {}

FAKE_TMUX = r"""#!/usr/bin/env bash
# 假 tmux：会话 = $FAKE_DIR/sessions/<名> 文件；只支持 has-session / new-session，其余子命令记入 forbidden.log
d="$FAKE_DIR"; mkdir -p "$d/sessions"
case "$1" in
  has-session)
    t="${3#=}"; [ -e "$d/sessions/$t" ]; exit $? ;;
  new-session)
    shift; s=""; cmd=""
    while [ $# -gt 0 ]; do case "$1" in -d) shift ;; -s) s="$2"; shift 2 ;; *) cmd="$1"; shift ;; esac; done
    [ -e "$d/sessions/$s" ] && { echo "duplicate session: $s" >&2; exit 1; }
    : > "$d/sessions/$s"; printf '%s\t%s\n' "$s" "$cmd" >> "$d/launches.log"; exit 0 ;;
  *) echo "$*" >> "$d/forbidden.log"; exit 1 ;;
esac
"""

FAKE_SMI = r"""#!/usr/bin/env bash
# 假 nvidia-smi：memory.used 读 $FAKE_DIR/gpu（每行一卡）；compute-apps 只打表头
for a in "$@"; do
  case "$a" in
    --query-gpu=memory.used) cat "$FAKE_DIR/gpu"; exit 0 ;;
    --query-compute-apps=*) echo "pid, process_name, used_gpu_memory [MiB]"; exit 0 ;;
  esac
done
exit 1
"""

GOOD_LOG = """\
TRAIN_EXIT_CODE=0
SHARD_EXIT_CODE=0
EVAL692=PASS episodes=692 success=290 rate=0.4191 preseeded_errors=0
NM_PIPELINE=PASS
EXIT_CODE=0
"""


class Env:
    def __init__(self, tmp: pathlib.Path):
        self.tmp = tmp
        self.fake = tmp / "fake"
        self.bin = tmp / "bin"
        self.locks = tmp / "locks"
        self.logs = tmp / "gatelogs"
        self.ready = tmp / "ready"
        for p in (self.fake / "sessions", self.bin, self.locks, self.logs, self.ready):
            p.mkdir(parents=True, exist_ok=True)
        (self.bin / "tmux").write_text(FAKE_TMUX)
        (self.bin / "nvidia-smi").write_text(FAKE_SMI)
        for f in ("tmux", "nvidia-smi"):
            (self.bin / f).chmod(0o755)
        self.nm_log = tmp / "nm-pipe2.log"
        self.gates = tmp / "gates.sh"
        self.out = tmp / "trig.out"
        self.set_gpu(0)
        self.write_gates(["G3", "G4"])

    # ---- 模拟状态 ----
    def set_gpu(self, used: int):
        (self.fake / "gpu").write_text("".join(f"{used if i == 0 else 0}\n" for i in range(8)))

    def session(self, name: str, alive: bool):
        p = self.fake / "sessions" / name
        if alive:
            p.touch()
        elif p.exists():
            p.unlink()

    def launches(self) -> list[str]:
        p = self.fake / "launches.log"
        return p.read_text().splitlines() if p.exists() else []

    def lock_files(self) -> list[pathlib.Path]:
        return sorted(self.locks.glob("e2e-trig.*.launched"))

    def write_gates(self, names):
        lines = []
        for g in names:
            lc = g.lower()
            lines.append(
                f"gate_def {g} a01 e2e-{lc}-a01 {self.logs}/e2e-{lc}-a01.log "
                f"'[ -e {self.ready}/{g} ] || {{ echo prereq_{g}; exit 1; }}' "
                f"'echo run {g}' "
                f"'trig_verdict_all \"$GATE_LOG\" {g}_VERDICT'"
            )
        self.gates.write_text("\n".join(lines) + "\n")

    def gate_pass(self, g: str):
        (self.logs / f"e2e-{g.lower()}-a01.log").write_text(f"{g}_VERDICT=PASS x=1\nEXIT_CODE=0\n")

    # ---- 运行 ----
    def env(self):
        e = dict(os.environ)
        e["PATH"] = f"{self.bin}:{e['PATH']}"
        e["FAKE_DIR"] = str(self.fake)
        return e

    def args(self, *extra):
        return ["bash", str(SCRIPT), "--nm-log", str(self.nm_log), "--nm-tmux", "nm-pipe2",
                "--gates-file", str(self.gates), "--lock-dir", str(self.locks),
                "--interval", INTERVAL, *extra]

    def run(self, *extra, timeout=30):
        return subprocess.run(self.args(*extra), env=self.env(), capture_output=True, text=True, timeout=timeout)

    def start(self):
        self.fh = open(self.out, "a")
        self.proc = subprocess.Popen(self.args(), env=self.env(), stdout=self.fh, stderr=subprocess.STDOUT,
                                     text=True, start_new_session=True)
        return self.proc

    def stop(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGKILL)  # 只杀本测试起的触发器进程组
        self.proc.wait(timeout=10)
        self.fh.close()

    def text(self) -> str:
        return self.out.read_text() if self.out.exists() else ""

    def wait_for(self, pred, timeout=15.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if pred():
                return True
            time.sleep(0.05)
        return False

    def rechecks(self) -> int:
        return self.text().count("TRIG_RECHECK ")

    def let_rounds(self, n=4):
        base = self.rechecks()
        assert self.wait_for(lambda: self.rechecks() >= base + n), self.text()


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def count(text: str, prefix: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.startswith(prefix))


# ---------------- A. NM_DONE 判定夹具 ----------------

def test_nm_done_pass_only_when_all_met(env):
    env.nm_log.write_text(GOOD_LOG)
    r = env.run("--dry-run")
    assert r.returncode == 0, r.stderr
    assert "NM_DONE=PASS" in r.stdout and "GPU_FREE=PASS gpus=8" in r.stdout
    assert env.launches() == [] and env.lock_files() == []
    assert count(r.stdout, "TRIG_DRYRUN gate=") == 2 and "tmux=e2e-g3-a01" in r.stdout
    assert "TRIG_DRYRUN_END gates=2" in r.stdout and "launched=0" in r.stdout


def test_nm_shard_exit_code_last_line_not_pass(env):
    env.nm_log.write_text(GOOD_LOG.replace("EXIT_CODE=0\n", "") + "SHARD_EXIT_CODE=0\n")
    r = env.run("--dry-run")
    assert "NM_DONE=PASS" not in r.stdout and "NM_DONE=FAIL" not in r.stdout
    assert "exit_code_last_line" in r.stdout
    r = env.run("--max-rounds", "2")
    assert "NM_DONE=PASS" not in r.stdout and env.launches() == []


def test_nm_pipeline_fail(env):
    env.nm_log.write_text(GOOD_LOG.replace("NM_PIPELINE=PASS", "NM_PIPELINE=FAIL stage=eval"))
    r = env.run()
    assert r.returncode == 3, r.stdout
    assert "NM_DONE=FAIL reason=NM_PIPELINE=FAIL" in r.stdout and "NM_DONE=PASS" not in r.stdout
    assert env.launches() == [] and env.lock_files() == []


def test_nm_last_exit_nonzero(env):
    env.nm_log.write_text(GOOD_LOG.replace("\nEXIT_CODE=0\n", "\nEXIT_CODE=1\n"))
    r = env.run()
    assert r.returncode == 3, r.stdout
    assert "NM_DONE=FAIL reason=last_line_EXIT_CODE=1" in r.stdout
    assert env.launches() == []


def test_nm_session_alive_not_pass(env):
    env.nm_log.write_text(GOOD_LOG)
    env.session("nm-pipe2", True)
    r = env.run("--max-rounds", "2")
    assert "NM_DONE=PASS" not in r.stdout and "tmux_nm-pipe2_alive" in r.stdout
    assert env.launches() == []


def test_never_calls_kill(env):
    src = SCRIPT.read_text()
    for bad in ("tmux kill-session", "tmux kill-server", "tmux -L", "tmux -S", "pkill", "killall"):
        assert bad not in src, bad


# ---------------- B. 〇.1.1 四个用例 ----------------

def test_case1_launch_exactly_once_after_log_stops(env):
    env.nm_log.write_text(GOOD_LOG)        # 日志已写完，此后不再增长
    env.session("nm-pipe2", True)
    env.set_gpu(5000)
    env.start()
    try:
        env.let_rounds(3)
        assert "NM_DONE=PASS" not in env.text()
        env.session("nm-pipe2", False)     # 只改模拟状态
        env.set_gpu(0)
        assert env.wait_for(lambda: "NM_DONE=PASS" in env.text()), env.text()
        assert "GPU_FREE=PASS gpus=8" in env.text()
        env.let_rounds(3)
        assert "TRIG_WAIT gate=G3 missing=prereq_G3" in env.text()
        assert env.launches() == []
        (env.ready / "G3").touch()         # 前置就绪
        assert env.wait_for(lambda: "TRIG_LAUNCH gate=G3" in env.text()), env.text()
        env.let_rounds(5)
    finally:
        env.stop()
    t = env.text()
    assert count(t, "TRIG_LAUNCH ") == 1
    assert "TRIG_LAUNCH gate=G3 tmux=e2e-g3-a01 session=e2e-g3-a01 attempt=a01" in t
    assert len(env.launches()) == 1 and len(env.lock_files()) == 1
    lk = env.lock_files()[0].read_text()
    assert "attempt=a01" in lk and "tmux=e2e-g3-a01" in lk and "launched_et=" in lk
    assert not (env.fake / "forbidden.log").exists()
    CASES["launch_once"] = True


def test_case2_restart_no_duplicate(env):
    env.nm_log.write_text(GOOD_LOG)
    (env.ready / "G3").touch()
    env.start()
    try:
        assert env.wait_for(lambda: "TRIG_LAUNCH gate=G3" in env.text()), env.text()
    finally:
        env.stop()                          # 杀掉触发器
    env.start()                             # 原样重启
    try:
        assert env.wait_for(lambda: "TRIG_SKIP gate=G3 locked=a01" in env.text()), env.text()
        env.let_rounds(4)
        # 锁在、会话（模拟）仍在：只 TRIG_SKIP，不重启
    finally:
        env.stop()
    t = env.text()
    assert count(t, "TRIG_LAUNCH ") == 1 and len(env.launches()) == 1 and len(env.lock_files()) == 1
    CASES["restart_no_dup"] = True


def test_case3_gpu_busy_no_launch(env):
    env.nm_log.write_text(GOOD_LOG)
    env.start()
    try:
        assert env.wait_for(lambda: "NM_DONE=PASS" in env.text()), env.text()
        env.set_gpu(1234)
        (env.ready / "G3").touch()
        assert env.wait_for(lambda: "TRIG_WAIT gate=G3 missing=gpu_busy" in env.text()), env.text()
        env.let_rounds(4)
        assert env.launches() == [] and env.lock_files() == []
        assert "TRIG_LAUNCH" not in env.text()
    finally:
        env.stop()
    CASES["gpu_busy"] = True


def test_case4_g4_waits_for_g3_session(env):
    env.nm_log.write_text(GOOD_LOG)
    (env.ready / "G3").touch()
    (env.ready / "G4").touch()
    env.start()
    try:
        assert env.wait_for(lambda: "TRIG_LAUNCH gate=G3" in env.text()), env.text()
        env.gate_pass("G3")                 # G3 已判 PASS，但会话（模拟）仍在
        assert env.wait_for(lambda: "missing=session_alive:e2e-g3-a01" in env.text()), env.text()
        env.let_rounds(4)
        assert "TRIG_LAUNCH gate=G4" not in env.text()
        assert [l.split("\t")[0] for l in env.launches()] == ["e2e-g3-a01"]
        env.session("e2e-g3-a01", False)    # G3 退出、卡释放
        assert env.wait_for(lambda: "TRIG_LAUNCH gate=G4" in env.text()), env.text()
        env.gate_pass("G4")
        env.session("e2e-g4-a01", False)
        assert env.wait_for(lambda: env.proc.poll() is not None), env.text()
    finally:
        env.stop()
    t = env.text()
    assert "TRIG_DONE gates=2 all=PASS" in t and env.proc.returncode == 0
    assert count(t, "TRIG_LAUNCH ") == 2
    CASES["g4_waits_g3"] = True


def test_gate_fail_stops(env):
    env.nm_log.write_text(GOOD_LOG)
    (env.logs / "e2e-g3-a01.log").write_text("G3_VERDICT=FAIL reason=x\n")
    r = env.run()
    assert r.returncode == 4 and "TRIG_STOP gate=G3 verdict=FAIL" in r.stdout
    assert env.launches() == []


def test_zz_verdict():
    need = ("launch_once", "restart_no_dup", "gpu_busy", "g4_waits_g3")
    assert all(CASES.get(k) for k in need), CASES
    print("E2E_TRIG_SELFTEST=PASS cases=4 launches_exact_once=PASS restart_no_dup=PASS")
