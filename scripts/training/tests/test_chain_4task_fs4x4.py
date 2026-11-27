"""S4 单测：1127 4×4 触发链 chain_4task_fs4x4.sh 与增量上传 run_fs4x4_ckpt_export.sh
（1127-4task-modul-fs4x4-512-1024-60k-plan.md 第二部分一节 S4 行验收；只跑 CPU、不碰 GPU、不写 v1-store、不连 HF）。

覆盖：
  ① 链 --dry-run 步骤顺序 preflight→g4_512→g4_1024→g5→train_512→train_1024→eval_pairs→summary，
     评估在两个训练之后；每个 GPU 步骤前都有 gpu_wait，summary 前没有；runner 命令与评估命令按接口拼写；
  ② --from 续跑：之前的步骤产物齐全则 SKIP verified 且不再打印其命令；缺 ckpt / 缺标记 / EXIT_CODE≠0 即 FAIL；
     非 dry-run 下不允许 FS4X4_CHAIN_STORE；未知 --from 与非 40 位 sha 被拒；
  ③ 上传包装器：RUN_NAME 白名单与 bucket 映射、--dry-run 不调 hf；假 hf 夹具下两份 step + run 根文件全流程
     打印 HF_VERIFY=PASS objects=<n> bytes=<n> mismatches=0 与 INC_END、源 ckpt 不被删；读回篡改即 FAIL；私有桶即拒。
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import textwrap

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
CHAIN = REPO / "scripts/training/chain_4task_fs4x4.sh"
EXPORT = REPO / "scripts/dataset/hf_export/run_fs4x4_ckpt_export.sh"
BETA = "a" * 40
STEPS = ["preflight", "g4_512", "g4_1024", "g5", "train_512", "train_1024", "eval_pairs", "summary"]
FORMAL = {"512": "v2-1600ep-m32x4x4-modul-b128-50k", "1024": "v2-1600ep-m64x4x4-modul-b128-50k"}
SMOKE = {"512": "fs4a-smoke20", "1024": "fs4b-smoke20"}
CKPTS = [str(5000 * i) for i in range(1, 11)]
CLEAN = ("FS4X4_CHAIN_STORE", "FS4X4_EXPORT_FAKE_HF", "FS4X4_EXPORT_STEPS", "SRC", "EXPORT_ROOT", "TRAIN_LOG", "POLL_S")


def run(cmd, env_extra=None, timeout=120):
    env = {k: v for k, v in os.environ.items() if k not in CLEAN}
    env.update(env_extra or {})
    return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)


def step_events(out: str):
    return re.findall(r"^FS4X4_STEP=(\S+) (START|PASS|FAIL|SKIP)", out, flags=re.M)


# ---------------- ① 链 dry-run 顺序 ----------------
def test_chain_dry_run_order(tmp_path):
    r = run(["bash", str(CHAIN), BETA, str(tmp_path / "evalwt"), "--dry-run"], {"FS4X4_CHAIN_STORE": str(tmp_path / "store")})
    assert r.returncode == 0, r.stdout + r.stderr
    starts = [s for s, k in step_events(r.stdout) if k == "START"]
    assert starts == STEPS
    assert [s for s, k in step_events(r.stdout) if k == "PASS"] == STEPS
    lines = r.stdout.splitlines()
    idx = {s: next(i for i, l in enumerate(lines) if l.startswith(f"FS4X4_STEP={s} START")) for s in STEPS}
    # 评估在两个训练都 PASS 之后
    pass_b = next(i for i, l in enumerate(lines) if l.startswith("FS4X4_STEP=train_1024 PASS"))
    assert idx["train_512"] < idx["train_1024"] < pass_b < idx["eval_pairs"] < idx["summary"]
    # 每个 GPU 步骤 START 的前一行（或前两行内）是 gpu_wait；summary 前没有
    for s in ["g4_512", "g4_1024", "g5", "train_512", "train_1024", "eval_pairs"]:
        assert lines[idx[s] - 1].startswith(f"DRY gpu_wait before={s}"), s
    assert not any(l.startswith("DRY gpu_wait before=summary") for l in lines)
    assert sum(l.startswith("DRY gpu_wait") for l in lines) == 6
    cmds = [l for l in lines if l.startswith("DRY cmd: ")]
    assert len(cmds) == 7
    assert cmds[0].endswith(f"run_4task_fs4x4_50k.sh {BETA} --budget 512 --smoke20")
    assert cmds[1].endswith(f"run_4task_fs4x4_50k.sh {BETA} --budget 1024 --smoke20")
    assert "fs4x4_eval_pairs.sh --g6b" in cmds[2] and cmds[2].endswith("fs4a-smoke20/19 512")
    assert cmds[3].endswith(f"--budget 512") and cmds[4].endswith(f"--budget 1024")
    assert cmds[5].endswith("bash scripts/evaluation/fs4x4_eval_pairs.sh")
    assert "fs4x4_3seed_summary.py --out" in cmds[6] and cmds[6].endswith("jobs/fs4x4-1127/fs4x4_3seed.json")
    assert "FS4X4_CHAIN=DRY_DONE" in r.stdout
    assert not (tmp_path / "store").exists(), "dry-run 不得写文件"


def test_chain_rejects(tmp_path):
    assert run(["bash", str(CHAIN), "abc", "/tmp/x", "--dry-run"]).returncode == 2
    assert run(["bash", str(CHAIN), BETA, "/tmp/x", "--dry-run", "--from", "nope"]).returncode == 2
    r = run(["bash", str(CHAIN), BETA, "/tmp/x"], {"FS4X4_CHAIN_STORE": str(tmp_path)})
    assert r.returncode == 2 and "只允许与 --dry-run" in r.stderr


# ---------------- ② --from 续跑 ----------------
def make_run(store: pathlib.Path, run_name: str, ckpts, exit_code=0):
    root = store / "train-runs/mme_vla_suite_b128_80k" / run_name
    for c in ckpts:
        (root / c).mkdir(parents=True, exist_ok=True)
        (root / c / "_CHECKPOINT_METADATA").write_text("{}")
    logs = store / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{run_name}.log").write_text(f"FS4X4_TRAIN=PASS run={run_name}\nEXIT_CODE={exit_code}\n")


def marker(store: pathlib.Path, step: str, line: str = ""):
    st = store / "jobs/fs4x4-1127/chain-state"
    st.mkdir(parents=True, exist_ok=True)
    (st / f"{step}.pass").write_text(f"step={step} beta={BETA}\n{line}\n")


def full_fixture_until(store, last):
    order = STEPS[1:]
    for s in order[: order.index(last) + 1]:
        if s.startswith("g4_"):
            make_run(store, SMOKE[s[3:]], ["19"]); marker(store, s)
        elif s.startswith("train_"):
            make_run(store, FORMAL[s[6:]], CKPTS); marker(store, s)
        elif s == "g5":
            marker(store, s, "FS4X4_G5=PASS episodes=1")
        elif s == "eval_pairs":
            marker(store, s, "FS4X4_EVAL_PAIRS=PASS done=6")


def test_chain_from_eval_pairs_skips_verified(tmp_path):
    store = tmp_path / "store"
    full_fixture_until(store, "train_1024")
    r = run(["bash", str(CHAIN), BETA, "/tmp/x", "--dry-run", "--from", "eval_pairs"], {"FS4X4_CHAIN_STORE": str(store)})
    assert r.returncode == 0, r.stdout
    ev = step_events(r.stdout)
    assert [s for s, k in ev if k == "SKIP"] == ["g4_512", "g4_1024", "g5", "train_512", "train_1024"]
    assert [s for s, k in ev if k == "START"] == ["preflight", "eval_pairs", "summary"]
    assert "run_4task_fs4x4_50k.sh" not in r.stdout


def test_chain_from_train_1024(tmp_path):
    store = tmp_path / "store"
    full_fixture_until(store, "train_512")
    r = run(["bash", str(CHAIN), BETA, "/tmp/x", "--dry-run", "--from", "train_1024"], {"FS4X4_CHAIN_STORE": str(store)})
    assert r.returncode == 0, r.stdout
    assert [s for s, k in step_events(r.stdout) if k == "START"] == ["preflight", "train_1024", "eval_pairs", "summary"]
    assert "--budget 512" not in r.stdout


@pytest.mark.parametrize("breakage", ["missing_ckpt", "missing_marker", "bad_exit", "tmp_dir"])
def test_chain_from_fails_on_bad_artifact(tmp_path, breakage):
    store = tmp_path / "store"
    full_fixture_until(store, "train_1024")
    a = store / "train-runs/mme_vla_suite_b128_80k" / FORMAL["512"]
    if breakage == "missing_ckpt":
        (a / "50000/_CHECKPOINT_METADATA").unlink()
    elif breakage == "missing_marker":
        (store / "jobs/fs4x4-1127/chain-state/train_512.pass").unlink()
    elif breakage == "bad_exit":
        make_run(store, FORMAL["512"], CKPTS, exit_code=1)
    else:
        (a / "45000.orbax-checkpoint-tmp-123").mkdir()
    r = run(["bash", str(CHAIN), BETA, "/tmp/x", "--dry-run", "--from", "eval_pairs"], {"FS4X4_CHAIN_STORE": str(store)})
    assert r.returncode == 1, r.stdout
    assert "FS4X4_CHAIN=FAIL step=train_512" in r.stdout
    assert "FS4X4_STEP=eval_pairs START" not in r.stdout


# ---------------- ③ 上传包装器 ----------------
FAKE_HF = r"""#!/usr/bin/env bash
# 假 hf：hf://buckets/<ns>/<name>/<path> 映射到 $FAKE_BUCKETS/<ns>/<name>/<path>
set -euo pipefail
B="${FAKE_BUCKETS:?}"
map() { local p="$1"; case "$p" in hf://buckets/*) echo "$B/${p#hf://buckets/}" ;; *) echo "$p" ;; esac; }
echo "$*" >> "$B/../hf_calls.log"
case "$1 ${2:-}" in
  "auth whoami") echo "user=HongzeFu orgs=x" ;;
  "buckets create") mkdir -p "$B/$3"; [ -f "$B/$3.private" ] || true ;;
  "buckets info")
    d="$B/$3"
    n=$(find "$d" -type f | wc -l); s=$(find "$d" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
    priv=false; [ -f "$d.private" ] && priv=true
    printf '{"id": "%s", "private": %s, "size": %s, "total_files": %s}\n' "$3" "$priv" "$s" "$n" ;;
  "buckets cp") src=$(map "$3"); dst=$(map "$4"); mkdir -p "$(dirname "$dst")"; cp "$src" "$dst"
    [ -n "${FAKE_CORRUPT:-}" ] && [[ "$3" == hf://* ]] && [[ "$dst" == *"$FAKE_CORRUPT"* ]] && echo x >> "$dst"; true ;;
  sync*) src=$(map "$2"); dst=$(map "$3"); mkdir -p "$dst"; cp -a "$src/." "$dst/"
    if [ -n "${FAKE_CORRUPT:-}" ] && [[ "$2" == hf://* ]]; then
      f=$(find "$dst" -type f -path "*$FAKE_CORRUPT*" | head -n 1); [ -z "$f" ] || echo x >> "$f"; fi ;;
  *) echo "fake hf: 未支持 $*" >&2; exit 3 ;;
esac
"""


def export_fixture(tmp_path, steps=("5000", "10000")):
    fake = tmp_path / "bin/hf"; fake.parent.mkdir()
    fake.write_text(FAKE_HF); fake.chmod(0o755)
    buckets = tmp_path / "hfroot/buckets"; buckets.mkdir(parents=True)
    src = tmp_path / "src"
    for s in steps:
        (src / s / "params").mkdir(parents=True)
        (src / s / "_CHECKPOINT_METADATA").write_text("{}")
        (src / s / "params/shard0").write_bytes(os.urandom(4096))
    (src / "assets/robomme").mkdir(parents=True)
    (src / "assets/robomme/norm_stats.json").write_text("{}")
    (src / "history_config.json").write_text('{"budget": 512}')
    log = tmp_path / "train.log"; log.write_text("step 1\n")
    env = {"FS4X4_EXPORT_FAKE_HF": str(fake), "FAKE_BUCKETS": str(buckets), "SRC": str(src),
           "EXPORT_ROOT": str(tmp_path / "export"), "TRAIN_LOG": str(log),
           "FS4X4_EXPORT_STEPS": " ".join(steps), "POLL_S": "1"}
    return env, src, buckets


def test_export_rejects_and_dry_run(tmp_path):
    assert run(["bash", str(EXPORT), "fs4a-smoke20"]).returncode == 2
    assert run(["bash", str(EXPORT)]).returncode == 2
    env, _, buckets = export_fixture(tmp_path)
    for name, bucket in [(FORMAL["512"], "robomme-vla-4task-modul-m32x4x4-50k-v1"),
                         (FORMAL["1024"], "robomme-vla-4task-modul-m64x4x4-50k-v1")]:
        r = run(["bash", str(EXPORT), name, "--dry-run"], env)
        assert r.returncode == 0, r.stdout + r.stderr
        assert f"bucket=HongzeFu/{bucket}" in r.stdout and "FS4X4_EXPORT_DRY=DONE" in r.stdout
    assert not (tmp_path / "hfroot/hf_calls.log").exists(), "dry-run 不得调 hf"
    assert not (tmp_path / "export").exists(), "dry-run 不得写文件"


def test_export_fake_full_flow(tmp_path):
    env, src, buckets = export_fixture(tmp_path)
    r = run(["bash", str(EXPORT), FORMAL["512"]], env)
    assert r.returncode == 0, r.stdout + r.stderr
    m = re.search(r"^HF_VERIFY=PASS objects=(\d+) bytes=(\d+) mismatches=0$", r.stdout, flags=re.M)
    assert m, r.stdout
    nfiles = sum(1 for p in src.rglob("*") if p.is_file())
    nbytes = sum(p.stat().st_size for p in src.rglob("*") if p.is_file())
    assert int(m.group(1)) == nfiles and int(m.group(2)) == nbytes
    assert r.stdout.rstrip().splitlines()[-1].startswith("INC_END run=" + FORMAL["512"])
    assert (src / "10000/params/shard0").is_file(), "源 ckpt 不得被删"
    bk = buckets / "HongzeFu/robomme-vla-4task-modul-m32x4x4-50k-v1"
    assert (bk / "5000/params/shard0").read_bytes() == (src / "5000/params/shard0").read_bytes()
    calls = (tmp_path / "hfroot/hf_calls.log").read_text()
    assert "buckets create HongzeFu/robomme-vla-4task-modul-m32x4x4-50k-v1 --exist-ok" in calls
    assert "--private" not in calls
    # 续传：再跑一次全部跳过且仍 PASS
    r2 = run(["bash", str(EXPORT), FORMAL["512"]], env)
    assert r2.returncode == 0 and "step 5000 已校验，跳过" in r2.stdout and "HF_VERIFY=PASS" in r2.stdout


def test_export_readback_mismatch_fails(tmp_path):
    env, _, _ = export_fixture(tmp_path)
    env["FAKE_CORRUPT"] = "10000/params"
    r = run(["bash", str(EXPORT), FORMAL["512"]], env)
    assert r.returncode != 0
    assert "HF_VERIFY=FAIL sub=10000" in r.stdout and "HF_VERIFY=PASS" not in r.stdout


def test_export_private_bucket_refused(tmp_path):
    env, _, buckets = export_fixture(tmp_path)
    (buckets / "HongzeFu").mkdir()
    (buckets / "HongzeFu/robomme-vla-4task-modul-m64x4x4-50k-v1.private").write_text("")
    r = run(["bash", str(EXPORT), FORMAL["1024"]], env)
    assert r.returncode != 0 and "private=true" in r.stdout
