"""S4 运行脚本侧单测（1112-4task-modul-wanraw9-s32-50k-plan.html 第二部分 1.2 test_raw9_runner_chain.py 行、闸 G2）。

只跑 CPU、秒级：
  ① runner 四种 --dry-run（正式 / --g0-only / --smoke20 / --bench300）打印的 TRAIN_ARGS 与 v11 runner --dry-run 逐 token 比，
     差异只允许落在 history YAML、run 名、步数三处（表路径与诊断开关不在 TRAIN_ARGS 里：表路径走 YAML store_path，
     诊断开关是环境变量，单独核 DIAG_SWITCH 行）；
  ② --g0-only 在「表缺失」夹具（R9_G0_MOTION 指向不存在的路径）下真实只读跑一遍：输出 DATA_LINEAGE_PRE=PASS table=pending，
     不得出现 DATA_LINEAGE=PASS；R9_G0_MOTION 在非 --g0-only 模式下被拒；（需要主检出 v1-store 数据与 .venv，缺则 skip）
  ③ 链脚本 CHAIN_DRY=1 + 三种假日志夹具（缺 CKPT_LOAD、EXIT_CODE=1、正常）：只在正常时打印 eval 命令，且含写死的
     POLICY / RUN_NAME / EXPECT_MOTION_STRIDE=32 / EXPECT_MOTION_BUDGET=80 与 …/50000 7；
  ④ local_692_pool.sh：相对基线 e59cde3 只改了 POLICY 一行；DRY_RUN=1 POLICY=<新值> 的 EVAL_ENV 行打印新值，
     不设 POLICY 时 EVAL_ENV 行与基线脚本逐字相同。
末尾 test_zz_verdict 打印 S4_RUNNER_UNIT=PASS（pytest -s 可见；全部用例通过才会执行到它）。
"""
from __future__ import annotations

import os
import pathlib
import re
import shlex
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts/training/run_4task_wanraw9s32_50k.sh"
V11_RUNNER = REPO / "scripts/training/run_4task_wantp768_50k.sh"
CHAIN = REPO / "scripts/training/chain_wanraw9s32_train_eval.sh"
POOL = REPO / "scripts/evaluation/local_692_pool.sh"
EXPORT = REPO / "scripts/dataset/hf_export/run_wanraw9s32_ckpt_export.sh"
BASE = "e59cde3"
MAIN = pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA")
NEW_POLICY = "perceptual-framesamp-modul-8frame-8x8-wanraw9s32"
FORMAL_RUN = "v2-1600ep-m8x8-modul-wanraw9s32-b128-50k"


def run(cmd, env_extra=None, cwd=REPO, check=False):
    env = {k: v for k, v in os.environ.items() if k not in ("R9_G0_MOTION", "CHAIN_DRY", "CHAIN_RUNNER_LOG", "POLICY", "RUN_NAME", "DRY_RUN")}
    env.update(env_extra or {})
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
    if check:
        assert p.returncode == 0, p.stdout + p.stderr
    return p


def train_args(out: str) -> list[str]:
    line = next(x for x in out.splitlines() if x.startswith("TRAIN_ARGS="))
    return shlex.split(line[len("TRAIN_ARGS="):])


def test_bash_syntax():
    for f in (RUNNER, CHAIN, EXPORT, POOL):
        run(["bash", "-n", str(f)], check=True)


@pytest.mark.parametrize("flags,run_name,steps", [
    ([], FORMAL_RUN, "50001"),
    (["--g0-only"], FORMAL_RUN, "50001"),
    (["--smoke20"], "r9-smoke20", "20"),
    (["--bench300"], "r9-bench300", "300"),
])
def test_runner_dry_train_args_vs_v11(flags, run_name, steps):
    v11 = train_args(run(["bash", str(V11_RUNNER), "X", "--dry-run"], check=True).stdout)
    out = run(["bash", str(RUNNER), "X", *flags, "--dry-run"], check=True).stdout
    new = train_args(out)
    assert len(new) == len(v11)
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(v11, new)) if a != b]
    allowed = {
        v11.index("--exp-name") + 1: run_name,
        v11.index("--model.history-config") + 1: "perceptual-framesamp-modul-8frame-8x8-wanraw9s32.yaml",
        len(v11) - 1: f"--num_train_steps={steps}",
    }
    for i, _a, b in diffs:
        assert i in allowed and allowed[i] == b, (i, _a, b)
    # 未改动的位置逐字相同（数据集、assets、ckpt 根、TrainConfig 名）
    assert all(v11[i] == new[i] for i in range(len(v11)) if i not in allowed)
    if flags == ["--g0-only"]:
        assert "G0_ONLY=1" in out and "DIAG_SWITCH" not in out
    else:
        assert re.search(r"^DIAG_SWITCH TRAIN_TIMING_STEPS=300 TRAIN_RECORD_DIR=\S+ follower=STEP_TIME\+DIAG gpu_sample_s=15 ", out, re.M)
    assert "motion-wan-raw9-s32" in out


def test_runner_rejects_bad_mode_and_names():
    assert run(["bash", str(RUNNER), "X", "--smoke20", "--bench300", "--dry-run"]).returncode == 2
    assert run(["bash", str(RUNNER), "X", "--smoke20", "--run-name", "foo", "--dry-run"]).returncode == 2
    assert run(["bash", str(RUNNER), "X", "--bench300", "--run-name", "r9-smoke20", "--dry-run"]).returncode == 2
    assert run(["bash", str(RUNNER), "X", "--bench300", "--run-name", "r9-memprobe", "--dry-run"]).returncode == 0
    p = run(["bash", str(RUNNER), "X", "--smoke20", "--dry-run"], {"R9_G0_MOTION": "/tmp/x"})
    assert p.returncode == 2 and "R9_G0_MOTION" in p.stderr


_HAVE_MAIN_DATA = (MAIN / ".venv/bin/python").is_file() and (
    MAIN / "v1-store/datasets/4task-v2-1600ep-604f16da/meta/input_manifest.json").is_file()


@pytest.mark.skipif(not _HAVE_MAIN_DATA, reason="需要主检出 .venv 与 v1-store 4task 库（只读）")
def test_g0_only_pending_fixture(tmp_path):
    missing = tmp_path / "motion-wan-raw9-s32"   # 表缺失夹具：不建这个目录
    dry = run(["bash", str(RUNNER), "X", "--g0-only", "--dry-run"], {"R9_G0_MOTION": str(missing)}, check=True).stdout
    assert "TABLE=pending" in dry and "checks=pre" in dry
    p = run(["bash", str(RUNNER), "X", "--g0-only"], {"R9_G0_MOTION": str(missing)})
    out = p.stdout
    assert p.returncode == 0, out + p.stderr
    assert not re.search(r"^DATA_LINEAGE=PASS", out, re.M)
    m = re.search(r"^DATA_LINEAGE_PRE=PASS table=pending encoder=wanvae-[0-9a-f]{8}\+raw9s32 "
                  r"encoder_source_pin=([0-9a-f]{64}) vla_input_pin=([0-9a-f]{64}) files=4 mismatches=0 ", out, re.M)
    assert m and m.group(1) == m.group(2), out
    assert "G0_RING=PENDING ring=4 table=pending" in out
    assert "G0_RING=FAIL" not in out
    assert not missing.exists()


FIXTURES = {
    "no_ckpt_load": "DATA_LINEAGE=PASS encoder=wanvae-9980d252+raw9s32 files=4 mismatches=0\n"
                    "TRAIN_FINAL_RECORD=PASS step=50000 ckpts=10\nEND_UTC=x\nEXIT_CODE=0\n",
    "exit1": "DATA_LINEAGE=PASS encoder=wanvae-9980d252+raw9s32 files=4 mismatches=0\n"
             "CKPT_LOAD=PASS step=50000 leaves=1 resolved_sha256=x\n"
             "TRAIN_FINAL_RECORD=PASS step=50000 ckpts=10\nEND_UTC=x\nEXIT_CODE=1\n",
    "normal": "DATA_LINEAGE=PASS encoder=wanvae-9980d252+raw9s32 files=4 mismatches=0\n"
              "CKPT_LOAD=PASS step=50000 leaves=1 resolved_sha256=x\n"
              "TRAIN_FINAL_RECORD=PASS step=50000 ckpts=10\nEND_UTC=x\nEXIT_CODE=0\n",
}


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_chain_dry_fixtures(tmp_path, name):
    log = tmp_path / f"{name}.runner.log"
    log.write_text(FIXTURES[name])
    p = run(["bash", str(CHAIN), "a" * 40], {"CHAIN_DRY": "1", "CHAIN_RUNNER_LOG": str(log)})
    out = p.stdout
    if name == "normal":
        assert p.returncode == 0, out + p.stderr
        line = next(x for x in out.splitlines() if x.startswith("CHAIN_DRY eval_cmd="))
        for tok in (f"POLICY={NEW_POLICY}", "RUN_NAME=local692-wanraw9s32-50k", "EXPECT_MOTION_STRIDE=32",
                    "EXPECT_MOTION_BUDGET=80", "STORE=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/demo-prefix/binfill-3b4de03a0b46",
                    f"scripts/evaluation/local_692_pool.sh /scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-runs/mme_vla_suite_b128_80k/{FORMAL_RUN}/50000 7"):
            assert tok in line, (tok, line)
        assert "tmux new-session -d -s r9-eval" in out and "CHAIN=DRY_EVAL session=r9-eval" in out
        assert "CHAIN=STOP" not in out
    else:
        assert p.returncode != 0
        assert "eval_cmd=" not in out and "tmux new-session" not in out
        reason = {"no_ckpt_load": "ckpt_load", "exit1": "train_exit"}[name]
        assert re.search(rf"^CHAIN=STOP reason={reason} ", out, re.M), out


def test_chain_rejects_fixture_log_in_real_mode(tmp_path):
    p = run(["bash", str(CHAIN), "a" * 40], {"CHAIN_RUNNER_LOG": str(tmp_path / "x.log")})
    assert p.returncode == 2 and "CHAIN=STOP reason=bad_env" in p.stdout


def _eval_env_line(script: pathlib.Path, env_extra, tmp_path) -> str:
    # STORE 指向没有 E0.txt 的临时目录：脚本在打印 EVAL_ENV 后于 E0 闸退出，不建任何目录、不跑 uv
    store = tmp_path / "store"
    store.mkdir(exist_ok=True)
    p = run(["bash", str(script), "/abs/ckpt/50000", "7"], {"DRY_RUN": "1", "STORE": str(store), **env_extra})
    assert "E0_GATE=FAIL" in p.stdout, p.stdout + p.stderr
    return next(x for x in p.stdout.splitlines() if x.startswith("EVAL_ENV "))


def test_local692_policy_override(tmp_path):
    base_txt = run(["git", "show", f"{BASE}:scripts/evaluation/local_692_pool.sh"], check=True).stdout
    cur_txt = POOL.read_text()
    changed = [(a, b) for a, b in zip(base_txt.splitlines(), cur_txt.splitlines()) if a != b]
    assert len(base_txt.splitlines()) == len(cur_txt.splitlines()) and len(changed) == 1
    assert changed[0][0].replace("POLICY=perceptual-framesamp-modul-8frame-8x8-wantp768",
                                 'POLICY="${POLICY:-perceptual-framesamp-modul-8frame-8x8-wantp768}"') == changed[0][1]
    # 基线脚本放进临时仓库骨架（同目录带 env.sh），打印出的 EVAL_ENV 不含路径，可与当前脚本逐字比
    skel = tmp_path / "skel/scripts/evaluation"
    skel.mkdir(parents=True)
    (skel / "local_692_pool.sh").write_text(base_txt)
    shutil.copy(REPO / "scripts/evaluation/env.sh", skel / "env.sh")
    base_line = _eval_env_line(skel / "local_692_pool.sh", {}, tmp_path)
    assert _eval_env_line(POOL, {}, tmp_path) == base_line
    assert "POLICY=perceptual-framesamp-modul-8frame-8x8-wantp768 " in base_line
    new_line = _eval_env_line(POOL, {"POLICY": NEW_POLICY}, tmp_path)
    assert f"POLICY={NEW_POLICY} " in new_line
    assert new_line == base_line.replace("perceptual-framesamp-modul-8frame-8x8-wantp768", NEW_POLICY)


def test_export_script_only_names_changed():
    base = (REPO / "scripts/dataset/hf_export/run_wantp768_ckpt_export.sh").read_text()
    new = EXPORT.read_text()
    code = lambda t: [x for x in t.splitlines() if not x.lstrip().startswith("#")]
    diffs = [(a, b) for a, b in zip(code(base), code(new)) if a != b]
    assert len(code(base)) == len(code(new))
    assert diffs == [
        ('RUN_NAME="v2-1600ep-m8x8-modul-wantp768-b128-50k"', f'RUN_NAME="{FORMAL_RUN}"'),
        ('BUCKET_ID="HongzeFu/robomme-vla-4task-modul-wantp768-50k-v1"', 'BUCKET_ID="HongzeFu/robomme-4task-modul-wanraw9-s32-50k-v1"'),
        ('TRAIN_LOG="${TRAIN_LOG:-$REPO/v1-store/logs/w768-train.log}"', 'TRAIN_LOG="${TRAIN_LOG:-$REPO/v1-store/logs/r9-train.log}"'),
    ]
    assert new.rstrip().endswith('echo "HF_VERIFY=PASS ckpts=$NDONE"') and "STEPS=(5000 10000 15000 20000 25000 30000 35000 40000 45000 50000)" in new


def test_v11_scripts_unchanged():
    for rel in ("scripts/training/run_4task_wantp768_50k.sh", "scripts/dataset/hf_export/run_wantp768_ckpt_export.sh"):
        assert run(["git", "diff", "--quiet", BASE, "--", rel]).returncode == 0, rel


def test_zz_verdict():
    print("S4_RUNNER_UNIT=PASS")
