"""S5 runner、链脚本、诊断脚本单测（1117-e2e-enc-train-plan.html 第二部分二节 S5 行验收；只跑 CPU、不碰 GPU、不写 v1-store）。

覆盖：
  ① runner 五种 --dry-run 的 TRAIN_ARGS 与 Wan-9 runner 只差三项超参类开关（--exp-name / --model.history-config / --num_train_steps），
     路径类相同；--ckptprobe 恰多 --save-interval 50 --keep-period 50，其余模式没有（R2、R8）；
  ② 三种探测模式只接受各自的 -aNN 名（e2e-bench-a01 通过；e2e-bench、e2e-bench-a1、bench 模式下的 e2e-shmprobe-a01 被拒），
     同名日志 / run 目录已存在即拒；冒烟缺省名 e2e-smoke20（check_checkpoint RUNS 同名）；
  ③ --g0-only 真实只读跑（需主检出 .venv 与 v1-store，缺则 skip）：YAML 夹具配对 → DATA_LINEAGE=PASS；错配 YAML → FAIL；
  ④ 链脚本 CHAIN_DRY 夹具：已有占位日志时不停下（顺延 e2e-eval.r1.log）、POLICY 不是缺省 wantp768、16/160、五个评估吞吐值；
  ⑤ local_692_pool_e2e.sh DRY_RUN（临时仓库骨架 + 主检出评估环境，缺则 skip）：打印五值；32 片与 24 片计划都过 EVAL692_PLAN_IDENTITY；
     EVAL_TIMEOUT 不足即拒；缩时夹具（run_shard.sh 原样分块循环 + 假客户端）验证「每局合法、累计超旧块时限」不丢回合；
  ⑥ check_eval692_identity.py 负例（少一条、多一条、等数量换一条）全部 FAIL，32 片与 24 片计划并集为同一全集；
  ⑦ e2e_g6_judge.py 假日志夹具分阶段落证据并聚合：BENCH300 在 2.0/2.5/3.5 三档 PASS／ACCEPTED_OVER_LIMIT…user_preauth_3.0s_42h／ASK，
     GPU_UTIL 在 90/80 两档 PASS／RECORDED，SHM_PEAK、GPU_PEAK、SMOKE_CKPT，以及 prefetch 不符、证据不覆盖、摘要不同不聚合；
  ⑦b（用户 2026-11-18 裁定：G6 ①③ 合成一次 --benchckpt，② 移到正式训练前 2000 步）runner --benchckpt 白名单正反例、
     TRAIN_ARGS 恰多 --save-interval 250 --keep-period 250、期望 ckpt 250/299；judge benchckpt 各档（BENCH300 三档、GPU_UTIL
     RECORDED、DEVICE_PEAK 缺行与超 72 FAIL、缺 CKPT250_READBACK FAIL、窗口不全 FAIL、prefetch 阶段 FAIL 不被吞），
     aggregate --plan benchckpt 打印 SHM_PEAK=DEFERRED；⑦c shm-formal PASS（0）/FAIL（2）/未到判定点（3）与 metrics 末尾半行，
     shmmon.sh --formal-rec 只判一次，judge 出错（1 / 127）不置已判、打 SHMMON_ERROR（限频）并下轮重判；
  ⑧ shmmon.sh 超线打印 SHM_ALERT；⑨ e2e_token_rank.py 统计量与 checkpoint 子树载入往返；
  ⑩ e2e_trig_gates.sh 被触发器 --dry-run 解析（G6 只剩单个 gate G6-benchckpt，判定只读 G6 行）。
末尾 test_zz_verdict 打印 S5_RUNNER_UNIT=PASS（pytest -s 可见；全部用例通过才会执行到它）。
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
MAIN = pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA")
RUNNER = REPO / "scripts/training/run_4task_e2eenc_50k.sh"
WAN9 = REPO / "scripts/training/run_4task_wanraw9s32_50k.sh"
CHAIN = REPO / "scripts/training/chain_e2eenc_train_eval.sh"
POOL = REPO / "scripts/evaluation/local_692_pool_e2e.sh"
IDENT = REPO / "scripts/evaluation/check_eval692_identity.py"
JUDGE = REPO / "scripts/training/g0/e2e_g6_judge.py"
SHMMON = REPO / "scripts/training/shmmon.sh"
GATES = REPO / "scripts/training/e2e_trig_gates.sh"
TRIG = REPO / "scripts/training/e2e_wait_nm.sh"
YAML_NAME = "perceptual-framesamp-modul-8frame-8x8-e2eenc.yaml"
FORMAL = "v2-1600ep-m8x8-modul-e2eenc-s16-b128-50k"
POLICY = "perceptual-framesamp-modul-8frame-8x8-e2eenc"
CLEAN_ENV = ("E2E_G0_MOTION", "E2E_G0_YAML", "E2E_DRY_STATE_DIR", "E2E_PREFETCH_FACTOR", "R9_G0_MOTION", "R9_TIMING_STEPS",
             "CHAIN_DRY", "CHAIN_RUNNER_LOG", "CHAIN_LOG_DIR", "POLICY", "RUN_NAME", "DRY_RUN", "EVAL_SHARDS", "EVAL_PER_GPU",
             "EVAL_MEM_FRACTION", "CHUNK_EPISODES", "EVAL_TIMEOUT", "STORE", "PYTHONPATH")


def run(cmd, env_extra=None, cwd=REPO, check=False, timeout=180):
    env = {k: v for k, v in os.environ.items() if k not in CLEAN_ENV}
    env.update(env_extra or {})
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    if check:
        assert p.returncode == 0, p.stdout + p.stderr
    return p


def train_args(out: str) -> list[str]:
    line = next(x for x in out.splitlines() if x.startswith("TRAIN_ARGS="))
    return shlex.split(line[len("TRAIN_ARGS="):])


def test_bash_syntax():
    for f in (RUNNER, CHAIN, POOL, SHMMON, GATES):
        run(["bash", "-n", str(f)], check=True)


# ── ① / ② runner ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("flags,wan9_flags,run_name,steps,extra", [
    ([], [], FORMAL, "50001", []),
    (["--g0-only"], ["--g0-only"], FORMAL, "50001", []),
    (["--smoke20"], ["--smoke20"], "e2e-smoke20", "20", []),
    (["--bench300", "--run-name", "e2e-bench-a01"], ["--bench300"], "e2e-bench-a01", "300", []),
    (["--shmprobe", "--run-name", "e2e-shmprobe-a03"], ["--bench300"], "e2e-shmprobe-a03", "2000", []),
    (["--ckptprobe", "--run-name", "e2e-ckptprobe-a01"], ["--bench300"], "e2e-ckptprobe-a01", "80",
     ["--save-interval", "50", "--keep-period", "50"]),
    (["--benchckpt", "--run-name", "e2e-benchckpt-a01"], ["--bench300"], "e2e-benchckpt-a01", "300",
     ["--save-interval", "250", "--keep-period", "250"]),
])
def test_runner_train_args_vs_wan9(tmp_path, flags, wan9_flags, run_name, steps, extra):
    w9 = train_args(run(["bash", str(WAN9), "X", *wan9_flags, "--dry-run"], check=True).stdout)
    out = run(["bash", str(RUNNER), "X", *flags, "--dry-run"], {"E2E_DRY_STATE_DIR": str(tmp_path)}, check=True).stdout
    new = train_args(out)
    assert new[len(w9):] == extra and len(new) == len(w9) + len(extra)
    allowed = {w9.index("--exp-name") + 1: run_name,
               w9.index("--model.history-config") + 1: YAML_NAME,
               len(w9) - 1: f"--num_train_steps={steps}"}
    for i, (a, b) in enumerate(zip(w9, new)):
        if a != b:
            assert i in allowed and allowed[i] == b, (i, a, b)
        else:
            assert i not in allowed or allowed[i] == b
    assert "motion-wan-raw9-s16" in out and f"STEPS={steps} " in out
    if flags[:1] == ["--g0-only"]:
        assert "G0_ONLY=1" in out and "DIAG_SWITCH" not in out
    else:
        assert re.search(r"^DIAG_SWITCH TRAIN_TIMING_STEPS=300 TRAIN_TIMING_PROFILE=0 TRAIN_RECORD_DIR=\S+ follower=STEP_TIME\+DIAG gpu_sample_s=15 ", out, re.M)
    assert re.search(r"^PATHCFG=prefetch=2 compact=off remat=off PATHCFG_SHA=[0-9a-f]{16}$", out, re.M)
    if "--shmprobe" in flags:
        assert "shmmon=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/logs/e2e-shmmon-a03.log" in out
    if "--benchckpt" in flags:
        assert "EXPECT_CKPTS=250 299 ATTEMPT=a01" in out and " shmmon=off " in out, out
        assert "g6_evidence=/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/bench/g6/g6-benchckpt-a01.json" in out
        assert re.search(r"^BENCHCKPT bench_window=steps_50_240 readback_step=250 expect_ckpts=250,299 "
                         r"judge=aggregate_plan_benchckpt shm=deferred_formal_first2000$", out, re.M), out
    yaml = REPO / "src/mme_vla_suite/models/config/robomme" / YAML_NAME
    if yaml.is_file():          # S3 合入后：HC_SHA 必须等于现文件字节摘要
        assert f"HC_SHA={hashlib.sha256(yaml.read_bytes()).hexdigest()} " in out


def test_runner_probe_name_whitelist(tmp_path):
    st = {"E2E_DRY_STATE_DIR": str(tmp_path)}
    ok = run(["bash", str(RUNNER), "X", "--bench300", "--run-name", "e2e-bench-a01", "--dry-run"], st)
    assert ok.returncode == 0 and "ATTEMPT=a01" in ok.stdout
    for flags in (["--bench300", "--run-name", "e2e-bench"], ["--bench300", "--run-name", "e2e-bench-a1"],
                  ["--bench300", "--run-name", "e2e-shmprobe-a01"], ["--shmprobe", "--run-name", "e2e-ckptprobe-a01"],
                  ["--bench300"], ["--ckptprobe", "--run-name", "r9-bench300"],
                  ["--smoke20", "--run-name", "e2e-bench-a01"], ["--run-name", "e2e-bench-a01"],
                  ["--smoke20", "--bench300"]):
        p = run(["bash", str(RUNNER), "X", *flags, "--dry-run"], st)
        assert p.returncode == 2, (flags, p.stdout, p.stderr)
    # 同名日志 / run 目录已存在即拒
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/e2e-bench-a02.runner.log").write_text("old\n")
    p = run(["bash", str(RUNNER), "X", "--bench300", "--run-name", "e2e-bench-a02", "--dry-run"], st)
    assert p.returncode == 2 and "拒绝覆盖本轮日志" in p.stderr
    (tmp_path / "train-runs/e2e-shmprobe-a02").mkdir(parents=True)
    p = run(["bash", str(RUNNER), "X", "--shmprobe", "--run-name", "e2e-shmprobe-a02", "--dry-run"], st)
    assert p.returncode == 2 and "拒绝复用" in p.stderr
    # 冒烟缺省名与 check_checkpoint RUNS 一致
    out = run(["bash", str(RUNNER), "X", "--smoke20", "--dry-run"], st, check=True).stdout
    assert "RUN=e2e-smoke20 STEPS=20 EXPECT_CKPTS=19 " in out
    # 夹具变量只在各自模式生效
    assert run(["bash", str(RUNNER), "X", "--smoke20"], st).returncode == 2
    p = run(["bash", str(RUNNER), "X", "--smoke20", "--dry-run"], {**st, "E2E_G0_YAML": "/tmp/x"})
    assert p.returncode == 2 and "E2E_G0_YAML" in p.stderr


def test_runner_benchckpt_name_whitelist(tmp_path):
    st = {"E2E_DRY_STATE_DIR": str(tmp_path)}
    for name in ("e2e-benchckpt-a01", "e2e-benchckpt-a99"):
        p = run(["bash", str(RUNNER), "X", "--benchckpt", "--run-name", name, "--dry-run"], st)
        assert p.returncode == 0 and f"RUN={name} STEPS=300 " in p.stdout, (name, p.stdout, p.stderr)
    for flags in (["--benchckpt"], ["--benchckpt", "--run-name", "e2e-benchckpt"],
                  ["--benchckpt", "--run-name", "e2e-benchckpt-a1"], ["--benchckpt", "--run-name", "e2e-benchckpt-a001"],
                  ["--benchckpt", "--run-name", "e2e-bench-a01"], ["--benchckpt", "--run-name", "e2e-ckptprobe-a01"],
                  ["--bench300", "--run-name", "e2e-benchckpt-a01"], ["--ckptprobe", "--run-name", "e2e-benchckpt-a01"],
                  ["--benchckpt", "--bench300", "--run-name", "e2e-benchckpt-a01"]):
        p = run(["bash", str(RUNNER), "X", *flags, "--dry-run"], st)
        assert p.returncode == 2, (flags, p.stdout, p.stderr)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/e2e-benchckpt-a02.runner.log").write_text("old\n")
    p = run(["bash", str(RUNNER), "X", "--benchckpt", "--run-name", "e2e-benchckpt-a02", "--dry-run"], st)
    assert p.returncode == 2 and "拒绝覆盖本轮日志" in p.stderr
    (tmp_path / "train-runs/e2e-benchckpt-a03").mkdir(parents=True)
    p = run(["bash", str(RUNNER), "X", "--benchckpt", "--run-name", "e2e-benchckpt-a03", "--dry-run"], st)
    assert p.returncode == 2 and "拒绝复用" in p.stderr


def test_runner_prefetch_pathcfg(tmp_path):
    st = {"E2E_DRY_STATE_DIR": str(tmp_path)}
    a = run(["bash", str(RUNNER), "X", "--bench300", "--run-name", "e2e-bench-a01", "--dry-run"], st, check=True).stdout
    b = run(["bash", str(RUNNER), "X", "--bench300", "--run-name", "e2e-bench-a01", "--dry-run"],
            {**st, "E2E_PREFETCH_FACTOR": "1"}, check=True).stdout
    sha = lambda o: re.search(r"PATHCFG_SHA=(\w+)", o).group(1)
    assert "PATHCFG=prefetch=1 " in b and sha(a) != sha(b)
    assert train_args(a) == train_args(b)          # prefetch 走环境变量，不进 TRAIN_ARGS（R2）
    assert run(["bash", str(RUNNER), "X", "--dry-run"], {**st, "E2E_PREFETCH_FACTOR": "0"}).returncode == 2


_HAVE_MAIN = (MAIN / ".venv/bin/python").is_file() and (
    MAIN / "v1-store/datasets/4task-v2-1600ep-604f16da/motion-wan-raw9-s16/meta/store_meta.json").is_file()


def _yaml_fixture(tmp_path, kind="wan_e2e"):
    y = tmp_path / YAML_NAME
    y.write_text("motion:\n  enabled: true\n  stride: 16\n  budget: 160\n"
                 "  store_path: v1-store/datasets/4task-v2-1600ep-604f16da/motion-wan-raw9-s16\n"
                 f"  encoder_kind: {kind}\n  readout: wan_e2e_enc768_s16_v1\n  raw_dim: 16384\n  latents_per_window: 9\n")
    return y


@pytest.mark.skipif(not _HAVE_MAIN, reason="需要主检出 .venv 与 v1-store s16 新表（只读）")
def test_g0_only_real(tmp_path):
    p = run(["bash", str(RUNNER), "X", "--g0-only"], {"E2E_G0_YAML": str(_yaml_fixture(tmp_path))}, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    m = re.search(r"^DATA_LINEAGE=PASS encoder=wanvae-[0-9a-f]{8}\+e2e-s16 encoder_source_pin=([0-9a-f]{64}) "
                  r"vla_input_pin=([0-9a-f]{64}) files=4 mismatches=0 ", p.stdout, re.M)
    assert m and m.group(1) == m.group(2), p.stdout
    assert "G0_RING=FAIL" not in p.stdout and "MOTION_STORE_META=PASS" in p.stdout
    bad = tmp_path / "bad"; bad.mkdir()
    p = run(["bash", str(RUNNER), "X", "--g0-only"], {"E2E_G0_YAML": str(_yaml_fixture(bad, "wan_raw9"))}, timeout=600)
    assert p.returncode != 0 and re.search(r"^DATA_LINEAGE=FAIL ", p.stdout, re.M), p.stdout


# ── ④ 链脚本 ─────────────────────────────────────────────────────────────────

FIX = {
    "no_ckpt_load": "TRAIN_FINAL_RECORD=PASS step=50000 ckpts=10\nEND_UTC=x\nEXIT_CODE=0\n",
    "exit1": "CKPT_LOAD=PASS step=50000 leaves=1\nTRAIN_FINAL_RECORD=PASS step=50000 ckpts=10\nEND_UTC=x\nEXIT_CODE=1\n",
    "normal": "DATA_LINEAGE=PASS x\nCKPT_LOAD=PASS step=50000 leaves=98\nTRAIN_FINAL_RECORD=PASS step=50000 ckpts=10\nEND_UTC=x\nEXIT_CODE=0\n",
}


@pytest.mark.parametrize("name", sorted(FIX))
def test_chain_dry_fixtures(tmp_path, name):
    log = tmp_path / f"{name}.runner.log"
    log.write_text(FIX[name])
    logs = tmp_path / "logs"; logs.mkdir()
    p = run(["bash", str(CHAIN), "a" * 40], {"CHAIN_DRY": "1", "CHAIN_RUNNER_LOG": str(log), "CHAIN_LOG_DIR": str(logs)})
    out = p.stdout
    if name != "normal":
        assert p.returncode != 0 and "eval_cmd=" not in out
        assert re.search(rf"^CHAIN=STOP reason={ {'no_ckpt_load': 'ckpt_load', 'exit1': 'train_exit'}[name]} ", out, re.M), out
        return
    assert p.returncode == 0, out + p.stderr
    line = next(x for x in out.splitlines() if x.startswith("CHAIN_DRY eval_cmd="))
    for tok in (f"POLICY={POLICY}", "RUN_NAME=local692-e2eenc-s16-50k", "EXPECT_MOTION_STRIDE=16", "EXPECT_MOTION_BUDGET=160",
                "CHUNK_EPISODES=5", "EVAL_TIMEOUT=14400", "EVAL_SHARDS=32", "EVAL_PER_GPU=4", "EVAL_MEM_FRACTION=0.2",
                f"scripts/evaluation/local_692_pool_e2e.sh {MAIN}/v1-store/train-runs/mme_vla_suite_b128_80k/{FORMAL}/50000 7"):
        assert tok in line, (tok, line)
    assert "wantp768" not in line and "wanraw9s32" not in line
    assert f"CHAIN=DRY_EVAL session=e2e-eval log={logs}/e2e-eval.log" in out


def test_chain_placeholder_log_does_not_stop(tmp_path):
    log = tmp_path / "r.log"; log.write_text(FIX["normal"])
    logs = tmp_path / "logs"; logs.mkdir()
    (logs / "e2e-eval.log").write_text("PLACEHOLDER\n")
    (logs / "e2e-eval.r1.log").write_text("PLACEHOLDER\n")
    p = run(["bash", str(CHAIN), "a" * 40], {"CHAIN_DRY": "1", "CHAIN_RUNNER_LOG": str(log), "CHAIN_LOG_DIR": str(logs),
                                             "EVAL_SHARDS": "24", "EVAL_PER_GPU": "3"})
    assert p.returncode == 0 and "CHAIN=STOP" not in p.stdout, p.stdout
    assert f"CHAIN_EVAL_LOG existing={logs}/e2e-eval.log using={logs}/e2e-eval.r2.log" in p.stdout
    assert f"CHAIN=DRY_EVAL session=e2e-eval log={logs}/e2e-eval.r2.log" in p.stdout
    assert "EVAL_SHARDS=24 EVAL_PER_GPU=3 EVAL_MEM_FRACTION=0.2" in p.stdout
    assert (logs / "e2e-eval.log").read_text() == "PLACEHOLDER\n"
    p = run(["bash", str(CHAIN), "a" * 40], {"CHAIN_LOG_DIR": str(logs)})
    assert p.returncode == 2 and "reason=bad_env" in p.stdout


# ── ⑤ 评估脚本 ───────────────────────────────────────────────────────────────

CLIENT_ENV = MAIN / "v1-store/envs/policy-eval-client"
DEMO_STORE = MAIN / "v1-store/demo-prefix/binfill-3b4de03a0b46"
_HAVE_EVAL = CLIENT_ENV.is_dir() and (DEMO_STORE / "E0.txt").is_file() and (MAIN / "third_party/robomme_benchmark").is_dir()


def _skeleton(tmp_path) -> pathlib.Path:
    """临时仓库骨架：评估脚本 + 身份清单 + 指向主检出评估环境的软链（只读用），uv 缓存落在临时目录。"""
    sk = tmp_path / "sk"
    ev = sk / "scripts/evaluation"
    ev.mkdir(parents=True)
    for f in ("local_692_pool_e2e.sh", "env.sh", "make_shard_plan.py", "check_eval692_identity.py", "eval692_identity_v1.json"):
        shutil.copy(REPO / "scripts/evaluation" / f, ev / f)
    (ev / "client").symlink_to(REPO / "scripts/evaluation/client")
    (sk / "v1-store/envs").mkdir(parents=True)
    (sk / "v1-store/envs/policy-eval-client").symlink_to(CLIENT_ENV)
    (sk / "third_party").symlink_to(MAIN / "third_party")
    (sk / "docs").symlink_to(REPO / "docs")              # 预标清单 docs/training-doc/finalckpt-nomotion59999-gl（只读）
    return sk


def _pool_dry(sk, extra=None, args=("/abs/ckpt/50000", "7")):
    return run(["bash", str(sk / "scripts/evaluation/local_692_pool_e2e.sh"), *args],
               {"DRY_RUN": "1", "STORE": str(DEMO_STORE), **(extra or {})}, cwd=sk, timeout=300)


@pytest.mark.skipif(not _HAVE_EVAL, reason="需要主检出评估客户端环境与 demo 前缀库（只读）")
@pytest.mark.parametrize("shards,per_gpu", [("32", "4"), ("24", "3")])
def test_pool_dry_params_and_plan_identity(tmp_path, shards, per_gpu):
    sk = _skeleton(tmp_path)
    p = _pool_dry(sk, {"EVAL_SHARDS": shards, "EVAL_PER_GPU": per_gpu} if shards != "32" else {})
    out = p.stdout
    assert p.returncode == 0, out + p.stderr
    assert re.search(rf"^EVAL_PARAMS CHUNK_EPISODES=5 EVAL_TIMEOUT=14400 EVAL_SHARDS={shards} EVAL_PER_GPU={per_gpu} "
                     rf"EVAL_MEM_FRACTION=0\.2 EPISODE_WALL_S=2400 EXPECT_MOTION_STRIDE=16 EXPECT_MOTION_BUDGET=160 POLICY={POLICY}$", out, re.M), out
    assert re.search(rf"^EVAL692_PLAN_IDENTITY=PASS manifest_sha=[0-9a-f]{{16}} shards={shards} union=692 per_group=14/14$", out, re.M), out
    assert "PRESEED=PASS episodes=9 placed=9 duplicates=0" in out
    assert not (sk / "v1-store/evaluation").exists()


def test_pool_rejects_short_timeout(tmp_path):
    store = tmp_path / "store"; store.mkdir()
    p = run(["bash", str(POOL), "/abs/ckpt/50000", "7"], {"DRY_RUN": "1", "STORE": str(store), "EVAL_TIMEOUT": "12000"})
    assert p.returncode == 2 and "EVAL_TIMEOUT=12000 <" in p.stdout
    p = run(["bash", str(POOL), "/abs/ckpt/50000", "7"], {"DRY_RUN": "1", "STORE": str(store), "CHUNK_EPISODES": "4",
                                                       "EVAL_TIMEOUT": "12000"})
    assert "E0_GATE=FAIL" in p.stdout and "EVAL_PARAMS CHUNK_EPISODES=4 EVAL_TIMEOUT=12000 " in p.stdout


FAKE_CLIENT = r"""
import json, sys, time, pathlib
a = dict(x.split("=", 1) for x in sys.argv[1:] if x.startswith("--args."))
prog = pathlib.Path(a["--args.save_dir"]) / "progress.json"
plan = json.loads(pathlib.Path(a["--args.episode_plan"]).read_text())
eps = [f"{g}/{e}" for g, es in plan["groups"].items() for e in es]
done = json.loads(prog.read_text()) if prog.exists() else {}
n = 0
for k in eps:
    if k in done:
        continue
    if n >= int(a["--args.max_new_episodes"]):
        break
    time.sleep(float(a["--args.episode_wall_s"]) * 0.8)       # 每局合法：0.8 × 单局墙钟
    done[k] = True; n += 1
    prog.write_text(json.dumps(done))
"""


def _chunk_loop(tmp_path, chunk: int, timeout_s: float, wall: float, n_eps: int) -> int:
    """原样取 run_shard.sh 的分块循环（CHUNKS= 到 done），用假客户端跑缩时夹具，返回完成局数。"""
    src = (REPO / "scripts/evaluation/run_shard.sh").read_text().splitlines()
    i0 = next(i for i, l in enumerate(src) if l.startswith("CHUNKS=$(("))
    i1 = next(i for i in range(i0, len(src)) if src[i] == "done")
    plan = tmp_path / f"plan{chunk}.json"
    plan.write_text(json.dumps({"groups": {"T/easy": list(range(n_eps))}}))
    run_dir = tmp_path / f"run{chunk}"; run_dir.mkdir()
    (tmp_path / "fake_client.py").write_text(FAKE_CLIENT)
    header = (f'PLAN_EPISODES={n_eps}; CHUNK_EPISODES={chunk}; EVAL_TIMEOUT={timeout_s}; EPISODE_WALL_S={wall}; TLS_TUNABLE=x\n'
              f'CLIENT_UV=({shlex.quote(sys.executable)} {shlex.quote(str(tmp_path / "fake_client.py"))})\n'
              f'PORT=1; SEED=7; POLICY=p; CKPT_ID=1; MAX_STEPS=1; RUN_DIR={shlex.quote(str(run_dir))}; PLAN={shlex.quote(str(plan))}; SHARD_TAG=s\n'
              'DEMO_PREFIX_STORE=\n')
    script = header + "\n".join(src[i0:i1 + 1]).replace("examples/robomme/eval.py", "") + "\n"
    subprocess.run(["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    prog = run_dir / "progress.json"
    return len(json.loads(prog.read_text())) if prog.exists() else 0


def test_chunk_timing_fixture_no_lost_episodes(tmp_path):
    # 缩时 1:6000：EPISODE_WALL_S 2400→0.4 s、EVAL_TIMEOUT 14400→2.4 s；每局 0.32 s（合法），10 局累计 3.2 s 超过旧块时限
    wall, timeout_s, n = 0.4, 2.4, 10
    assert _chunk_loop(tmp_path, 5, timeout_s, wall, n) == n             # E2E：CHUNK_EPISODES=5 → 每块 1.6 s < 2.4 s，不丢
    assert _chunk_loop(tmp_path, 20, timeout_s, wall, n) < n             # 旧口径 20：一块 3.2 s 被块时限误杀，丢回合


# ── ⑥ 692 身份闸 ─────────────────────────────────────────────────────────────


def _manifest():
    return json.loads((REPO / "scripts/evaluation/eval692_identity_v1.json").read_text())


def _write_plans(d: pathlib.Path, keys: list[str], shards: int):
    d.mkdir(parents=True, exist_ok=True)
    for k in range(shards):
        groups: dict = {}
        for key in keys[k::shards]:
            g, e = key.rsplit("/", 1)
            groups.setdefault(g, []).append(int(e))
        (d / f"shard{k}.json").write_text(json.dumps({"groups": groups}))


def _ident(*args):
    return run([sys.executable, "-I", str(IDENT), *args])


def test_identity_manifest_and_sources():
    m = _manifest()
    assert m["total"] == 692 and len(m["groups"]) == 14 and len(set(m["episodes"])) == 692
    p = _ident("verify-sources")
    assert p.returncode == 0 and p.stdout.startswith("CROSSCHECK=PASS "), p.stdout + p.stderr


def test_identity_plan_and_result_negatives(tmp_path):
    keys = list(_manifest()["episodes"])
    for shards in (32, 24):
        _write_plans(tmp_path / f"ok{shards}", keys, shards)
        p = _ident("plan", "--plan-dir", str(tmp_path / f"ok{shards}"), "--shards", str(shards))
        assert p.returncode == 0 and f"shards={shards} union=692 per_group=14/14" in p.stdout, p.stdout
    fake = "BinFill/easy/999999"
    cases = {"less": keys[:-1], "more": keys + [fake], "swap": keys[:-1] + [fake]}
    for name, ks in cases.items():
        _write_plans(tmp_path / name, ks, 32)
        p = _ident("plan", "--plan-dir", str(tmp_path / name), "--shards", "32")
        assert p.returncode == 1 and p.stdout.startswith("EVAL692_PLAN_IDENTITY=FAIL "), (name, p.stdout)
        prog: dict = {}
        for key in ks:
            g, e = key.rsplit("/", 1)
            prog.setdefault(g, {})[e] = True
        (tmp_path / f"{name}.json").write_text(json.dumps(prog))
        p = _ident("result", "--progress", str(tmp_path / f"{name}.json"))
        assert p.returncode == 1 and p.stdout.startswith("EVAL692_RESULT_IDENTITY=FAIL "), (name, p.stdout)
    # 相交（同一局落两片）也 FAIL
    _write_plans(tmp_path / "dup", keys, 32)
    s0 = json.loads((tmp_path / "dup/shard0.json").read_text())
    s1 = json.loads((tmp_path / "dup/shard1.json").read_text())
    g = next(iter(s0["groups"])); s1["groups"].setdefault(g, []).append(s0["groups"][g][0])
    (tmp_path / "dup/shard1.json").write_text(json.dumps(s1))
    assert _ident("plan", "--plan-dir", str(tmp_path / "dup"), "--shards", "32").returncode == 1
    prog = {}
    for key in keys:
        g, e = key.rsplit("/", 1)
        prog.setdefault(g, {})[e] = False
    (tmp_path / "okres.json").write_text(json.dumps(prog))
    p = _ident("result", "--progress", str(tmp_path / "okres.json"))
    assert p.returncode == 0 and re.match(r"EVAL692_RESULT_IDENTITY=PASS manifest_sha=[0-9a-f]{16} episodes=692$", p.stdout.strip())


@pytest.mark.skipif(not _HAVE_EVAL, reason="需要主检出评估客户端环境（make_shard_plan.py 读 robomme 候选）")
def test_identity_real_32_and_24_same_universe(tmp_path):
    unions = {}
    for shards in (32, 24):
        d = tmp_path / f"p{shards}"
        p = run(["env", f"UV_PROJECT_ENVIRONMENT={CLIENT_ENV}", f"UV_CACHE_DIR={tmp_path / 'uvc'}", "uv", "run",
                 "--project", str(REPO / "scripts/evaluation/client"), "--frozen", "--no-sync", "python",
                 str(REPO / "scripts/evaluation/make_shard_plan.py"), "shards", str(d), "--shards", str(shards),
                 "--demo-store", str(DEMO_STORE)], cwd=MAIN, timeout=300)
        assert p.returncode == 0 and "PLAN_OK " in p.stdout, p.stdout + p.stderr
        q = _ident("plan", "--plan-dir", str(d), "--shards", str(shards))
        assert q.returncode == 0 and "union=692 per_group=14/14" in q.stdout, q.stdout
        unions[shards] = {f"{g}/{e}" for k in range(shards)
                          for g, es in json.loads((d / f"shard{k}.json").read_text())["groups"].items() for e in es}
    assert unions[32] == unions[24] == set(_manifest()["episodes"])


# ── ⑦ G6 judge ───────────────────────────────────────────────────────────────


def _ts(t: float) -> str:
    return datetime.datetime.fromtimestamp(t).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]


def _g6_fixture(tmp_path, mode, attempt, s_per_step=2.0, util=90.0, steps=300, prefetch="2", shm=(200.0, 201.0),
                peak_bytes=60 * 1024 ** 3, ckpt_after=29):
    rec = tmp_path / f"rec-{mode}-{attempt}"; rec.mkdir(exist_ok=True)
    t0 = 1_800_000_000.0
    n_timing = min(steps, 300)
    rows, t = [], t0
    for s in range(n_timing):
        dt = 30.0 if s < 20 else s_per_step
        rows.append({"step": s, "host_step_s": dt, "phases_s": {"data_next": 0.01}, "completed": True,
                     "wall_start": t, "wall_end": t + dt})
        t += dt
    (rec / "step_timing.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    t_end = t0 + 20 * 30 + (steps - 20) * s_per_step
    with (rec / "gpu_util_15s_full.csv").open("w") as f:
        tt = t0
        while tt <= t_end:
            for g in range(8):
                u = 0.0 if tt < t0 + 600 else util
                f.write(f"{_ts(tt)}, {g}, {u}, 70000\n")
            tt += 15
    metrics = [{"step": s, "wall_time": t0 + 600 + (s - 20) * s_per_step if s >= 20 else t0 + s * 30} for s in range(0, steps, 100)]
    (rec / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics) + "\n")
    log = tmp_path / f"{mode}-{attempt}.runner.log"
    lines = [f"DATALOADER_PREFETCH_FACTOR={prefetch} env=unset num_workers=16", f"TRAIN_START_EPOCH={int(t0)}"]
    if peak_bytes is not None:
        lines += [f"DEVICE_PEAK device={d} peak_bytes_in_use={peak_bytes - d}" for d in range(8)]
    if mode == "ckptprobe":
        lines.append("CKPT50_READBACK step=50 path=x bytes=123456 leaves=98 enc_leaves=33 wan9_leaves=65 finite=1")
    lines.append(f"TRAIN_DONE_EPOCH={int(t_end)}")
    log.write_text("\n".join(lines) + "\n")
    shm_log = None
    if mode == "shmprobe":
        shm_log = tmp_path / f"shmmon-{attempt}.log"
        split = t0 + 600 + (steps - 1500 - 20) * s_per_step
        out, tt = [], t0
        while tt <= t_end:
            g = shm[0] if tt < split else shm[1]
            out.append(f"SHM_SAMPLE ts=00:00:00 shm_used_gb={g} epoch={int(tt)} gpu_util=na gpu_util_mean=na")
            tt += 15
        shm_log.write_text("\n".join(out) + "\nEXIT_CODE=0\n")
    if mode == "ckptprobe":
        rows = [{"step": s, "host_step_s": 1.0, "phases_s": {}, "completed": True, "wall_start": t0 + s, "wall_end": t0 + s + 1}
                for s in range(51 + ckpt_after)]
        (rec / "step_timing.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rec, log, shm_log


def _stage(tmp_path, mode, attempt, ev, pathcfg_sha="aaaa", steps=300, **kw):
    rec, log, shm = _g6_fixture(tmp_path, mode, attempt, steps=steps, **kw)
    args = [sys.executable, "-I", str(JUDGE), "stage", "--mode", mode, "--attempt", attempt, "--run", f"e2e-{mode}-{attempt}",
            "--pathcfg", "prefetch=2 compact=off remat=off", "--pathcfg-sha", pathcfg_sha, "--prefetch-expect", "2",
            "--run-log", str(log), "--rec", str(rec), "--evidence-dir", str(ev), "--steps", str(steps), "--skip", "20",
            "--now-et", "2026-11-18T12:00:00-0500"]
    if shm:
        args += ["--shmmon-log", str(shm)]
    if mode == "ckptprobe":
        run_root = tmp_path / f"root-{attempt}"; (run_root / "50/params").mkdir(parents=True, exist_ok=True)
        args += ["--run-root", str(run_root), "--wan9-leaves", "65"]
    return run(args)


def _agg(ev, sha="aaaa"):
    return run([sys.executable, "-I", str(JUDGE), "aggregate", "--pathcfg-sha", sha, "--evidence-dir", str(ev)])


@pytest.mark.parametrize("x,word", [(2.0, "PASS"), (2.5, "ACCEPTED_OVER_LIMIT"), (3.5, "ASK")])
def test_g6_bench300_three_tiers(tmp_path, x, word):
    ev = tmp_path / "ev"
    p = _stage(tmp_path, "bench", "a01", ev, s_per_step=x)
    line = next(l for l in p.stdout.splitlines() if l.startswith("BENCH300="))
    if word == "PASS":
        assert line.startswith("BENCH300=PASS s_per_step=2.0000 limit=2.27"), line
    elif word == "ACCEPTED_OVER_LIMIT":
        assert line.startswith("BENCH300=ACCEPTED_OVER_LIMIT measured=2.5000 limit=2.27 "
                               "accepted_by=user_preauth_3.0s_42h at=2026-11-18T12:00:00-0500"), line
    else:
        assert line.startswith("BENCH300=ASK measured=3.5000 limit=2.27"), line
    assert (ev / "g6-bench-a01.json").is_file()
    assert re.search(rf"^G6_STAGE={'ASK' if word == 'ASK' else 'PASS'} mode=bench attempt=a01 ", p.stdout, re.M), p.stdout


@pytest.mark.parametrize("util,word", [(90.0, "PASS"), (80.0, "RECORDED")])
def test_g6_full_aggregate(tmp_path, util, word):
    ev = tmp_path / "ev"
    assert _stage(tmp_path, "bench", "a01", ev, util=util).returncode == 0
    assert _stage(tmp_path, "shmprobe", "a01", ev, util=util, steps=2000).returncode == 0
    assert _stage(tmp_path, "ckptprobe", "a01", ev, steps=80).returncode == 0
    p = _agg(ev)
    out = p.stdout
    assert p.returncode == 0, out
    assert re.search(rf"^GPU_UTIL={word} util_mean={util:.2f} limit=85 ", out, re.M), out
    assert re.search(r"^BENCH300=PASS s_per_step=2\.0000 limit=2\.27 ", out, re.M)
    assert re.search(r"^SHM_PEAK=PASS peak_gb=201\.0 limit=330 growth_last1500=0\.50% ", out, re.M), out
    assert re.search(r"^GPU_PEAK=PASS peak_gb=60\.00 limit=72$", out, re.M), out
    assert re.search(r"^SMOKE_CKPT=PASS step=50 updates_after=29 bytes=123456 leaves=98 enc_leaves=33 ", out, re.M), out
    assert re.search(r"^G6=PASS pathcfg_sha=aaaa attempts=bench:a01,shmprobe:a01,ckptprobe:a01$", out, re.M), out


def test_g6_failures_and_rules(tmp_path):
    ev = tmp_path / "ev"
    # prefetch 读回不符 → 该阶段 FAIL
    p = _stage(tmp_path, "bench", "a01", ev, prefetch="1")
    assert p.returncode == 1 and re.search(r"^G6_STAGE=FAIL .*prefetch_factor=1 expect=2", p.stdout, re.M), p.stdout
    # 证据不覆盖
    p = _stage(tmp_path, "bench", "a01", ev)
    assert p.returncode == 2 and "拒绝覆盖" in p.stderr
    # 新 attempt 通过 → 聚合取最新通过的 a02
    assert _stage(tmp_path, "bench", "a02", ev).returncode == 0
    # shm 增长 >2% / 超 330 → FAIL；无 DEVICE_PEAK → GPU_PEAK FAIL
    assert _stage(tmp_path, "shmprobe", "a01", ev, steps=2000, shm=(200.0, 210.0)).returncode == 1
    assert _stage(tmp_path, "shmprobe", "a02", ev, steps=2000, shm=(200.0, 340.0)).returncode == 1
    assert _stage(tmp_path, "shmprobe", "a03", ev, steps=2000, peak_bytes=None).returncode == 0
    # step 50 之后只更新 10 次 → SMOKE_CKPT FAIL
    p = _stage(tmp_path, "ckptprobe", "a01", ev, steps=80, ckpt_after=10, peak_bytes=None)
    assert p.returncode == 1 and "SMOKE_CKPT=FAIL" in p.stdout
    assert _stage(tmp_path, "ckptprobe", "a02", ev, steps=80, peak_bytes=None).returncode == 0
    out = _agg(ev).stdout
    assert "BENCH300=PASS" in out and "attempt=a02" in out
    assert re.search(r"^GPU_PEAK=PASS peak_gb=60\.00", out, re.M)       # bench a02 带 DEVICE_PEAK
    assert re.search(r"^G6=PASS .*bench:a02,shmprobe:a03,ckptprobe:a02$", out, re.M), out
    # 路径配置摘要不同的尝试不参与聚合
    p = _agg(ev, "bbbb")
    assert p.returncode == 1 and "BENCH300=FAIL reason=no_bench_evidence" in p.stdout and re.search(r"^G6=FAIL", p.stdout, re.M)
    # 全部无 DEVICE_PEAK → GPU_PEAK=FAIL
    ev2 = tmp_path / "ev2"
    for m, st in (("bench", 300), ("shmprobe", 2000), ("ckptprobe", 80)):
        (tmp_path / m).mkdir()
        _stage(tmp_path / m, m, "a09", ev2, steps=st, peak_bytes=None)
    out = _agg(ev2).stdout
    assert "GPU_PEAK=FAIL reason=no_peak_bytes_evidence" in out and re.search(r"^G6=FAIL", out, re.M)


def test_g6_stage_fail_not_swallowed_by_items(tmp_path):
    # bench-a01 阶段因 prefetch 读回不符 FAIL（分项 s/步本身 PASS），shm/ckpt a01 PASS → 聚合必须 G6=FAIL
    ev = tmp_path / "ev"
    assert _stage(tmp_path, "bench", "a01", ev, prefetch="1").returncode == 1
    b = json.loads((ev / "g6-bench-a01.json").read_text())
    assert b["bench_status"] == "PASS" and b["stage_status"] == "FAIL"
    assert _stage(tmp_path, "shmprobe", "a01", ev, steps=2000).returncode == 0
    assert _stage(tmp_path, "ckptprobe", "a01", ev, steps=80).returncode == 0
    p = _agg(ev)
    assert p.returncode == 1, p.stdout
    assert re.search(r"^BENCH300=FAIL s_per_step=2\.0000 limit=2\.27 .*stage_status=FAIL stage_reasons=.*prefetch_factor=1 expect=2", p.stdout, re.M), p.stdout
    assert re.search(r"^G6=FAIL ", p.stdout, re.M) and not re.search(r"^BENCH300=PASS", p.stdout, re.M)


# ── ⑦b G6 benchckpt（一次 300 步 run 出四项，2026-11-18 用户裁定）────────────────


def _bc_stage(tmp_path, attempt, ev, s_per_step=2.0, util=90.0, prefetch="2", peak_bytes=60 * 1024 ** 3, readback=True,
              n_timing=300, pathcfg_sha="aaaa"):
    """benchckpt 假证据：第 0–19 步编译 30 s，第 20–49 步 5 s、第 241–299 步 9 s（都在窗口外，混进来均值就会变），
    第 50–240 步 = s_per_step；第 250 步另含 40 s checkpoint 相；GPU 采样窗口外为 0、窗口内为 util。"""
    d = tmp_path / f"bc-{attempt}"; d.mkdir()
    rec = d / "rec"; rec.mkdir()
    t0 = 1_800_000_000.0
    rows, t, win = [], t0, [None, None]
    for s in range(n_timing):
        dt = 30.0 if s < 20 else 5.0 if s < 50 else s_per_step if s <= 240 else 9.0
        ph = {"data_next": 0.01}
        if s == 250:
            ph["checkpoint"] = 40.0; dt += 40.0
        rows.append({"step": s, "host_step_s": dt, "phases_s": ph, "completed": True, "wall_start": t, "wall_end": t + dt})
        if s == 50: win[0] = t
        if s == 240: win[1] = t + dt
        t += dt
    (rec / "step_timing.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    win[1] = t if win[1] is None else win[1]
    with (rec / "gpu_util_15s_full.csv").open("w") as f:
        tt = t0
        while tt <= t:
            for g in range(8):
                f.write(f"{_ts(tt)}, {g}, {util if win[0] <= tt <= win[1] else 0.0}, 70000\n")
            tt += 15
    log = d / "runner.log"
    lines = [f"DATALOADER_PREFETCH_FACTOR={prefetch} env=unset num_workers=16", f"TRAIN_START_EPOCH={int(t0)}"]
    if peak_bytes is not None:
        lines += [f"DEVICE_PEAK device={g} peak_bytes_in_use={peak_bytes - g}" for g in range(8)]
    if readback:
        lines.append("CKPT250_READBACK step=250 path=x bytes=123456 leaves=98 enc_leaves=33 wan9_leaves=65 finite=1")
    lines.append(f"TRAIN_DONE_EPOCH={int(t)}")
    log.write_text("\n".join(lines) + "\n")
    root = d / "root"; (root / "250/params").mkdir(parents=True)
    return run([sys.executable, "-I", str(JUDGE), "stage", "--mode", "benchckpt", "--attempt", attempt,
                "--run", f"e2e-benchckpt-{attempt}", "--pathcfg", "prefetch=2 compact=off remat=off", "--pathcfg-sha", pathcfg_sha,
                "--prefetch-expect", "2", "--run-log", str(log), "--rec", str(rec), "--evidence-dir", str(ev), "--steps", "300",
                "--skip", "20", "--win-lo", "50", "--win-hi", "240", "--ckpt-step", "250", "--run-root", str(root),
                "--wan9-leaves", "65", "--now-et", "2026-11-18T12:00:00-0500"])


def _bc_agg(ev, sha="aaaa"):
    return run([sys.executable, "-I", str(JUDGE), "aggregate", "--plan", "benchckpt", "--pathcfg-sha", sha,
                "--evidence-dir", str(ev)])


DEFERRED = "SHM_PEAK=DEFERRED reason=user_2026-11-18_formal_first2000"


@pytest.mark.parametrize("x,word,g6", [(2.0, "PASS", "PASS"), (2.5, "ACCEPTED_OVER_LIMIT", "PASS"), (3.5, "ASK", "ASK")])
def test_g6_benchckpt_bench_tiers(tmp_path, x, word, g6):
    ev = tmp_path / "ev"
    p = _bc_stage(tmp_path, "a01", ev, s_per_step=x)
    assert p.returncode == 0, p.stdout + p.stderr
    assert re.search(rf"^G6_STAGE={g6} mode=benchckpt attempt=a01 .*util_mean=90\.00 ", p.stdout, re.M), p.stdout
    out = _bc_agg(ev).stdout
    b = next(l for l in out.splitlines() if l.startswith("BENCH300="))
    head = {"PASS": f"BENCH300=PASS s_per_step={x:.4f} limit=2.27",
            "ACCEPTED_OVER_LIMIT": f"BENCH300=ACCEPTED_OVER_LIMIT measured={x:.4f} limit=2.27 "
                                   "accepted_by=user_preauth_3.0s_42h at=2026-11-18T12:00:00-0500",
            "ASK": f"BENCH300=ASK measured={x:.4f} limit=2.27"}[word]
    assert b.startswith(head) and " window=steps_50_240 window_steps=191/191" in b and " steps=300/300" in b, b
    assert re.search(r"^GPU_UTIL=PASS util_mean=90\.00 limit=85 samples=\d+ window=steps_50_240 ", out, re.M), out
    assert re.search(r"^GPU_PEAK=PASS peak_gb=60\.00 limit=72 attempt=a01$", out, re.M), out
    assert re.search(r"^SMOKE_CKPT=PASS step=250 updates_after=49 bytes=123456 leaves=98 enc_leaves=33 attempt=a01$", out, re.M), out
    assert DEFERRED in out.splitlines()
    assert re.search(rf"^G6={g6} pathcfg_sha=aaaa plan=benchckpt attempts=benchckpt:a01$", out, re.M), out
    assert out.splitlines()[-1].startswith("G6=") and "SHM_PEAK=PASS" not in out and "SHM_PEAK=FAIL" not in out


def test_g6_benchckpt_util_recorded_and_item_fails(tmp_path):
    ev = tmp_path / "ev"
    assert _bc_stage(tmp_path, "a01", ev, util=80.0).returncode == 0
    out = _bc_agg(ev).stdout
    assert re.search(r"^GPU_UTIL=RECORDED util_mean=80\.00 limit=85 ", out, re.M) and re.search(r"^G6=PASS ", out, re.M), out
    # DEVICE_PEAK 缺行 → GPU_PEAK=FAIL、G6=FAIL；其余分项不被连坐（base_status 只含非分项原因）
    ev2 = tmp_path / "ev2"
    p = _bc_stage(tmp_path, "a02", ev2, peak_bytes=None)
    assert p.returncode == 1 and re.search(r"^G6_STAGE=FAIL .*GPU_PEAK=FAIL", p.stdout, re.M), p.stdout
    p = _bc_agg(ev2)
    assert p.returncode == 1
    assert re.search(r"^GPU_PEAK=FAIL reason=no_peak_bytes_evidence limit=72 attempt=a02$", p.stdout, re.M), p.stdout
    assert re.search(r"^BENCH300=PASS ", p.stdout, re.M) and re.search(r"^SMOKE_CKPT=PASS ", p.stdout, re.M)
    assert DEFERRED in p.stdout and re.search(r"^G6=FAIL .*plan=benchckpt", p.stdout, re.M)
    # 峰值超 72 GiB → FAIL；缺 CKPT250_READBACK → SMOKE_CKPT=FAIL
    ev3 = tmp_path / "ev3"
    assert _bc_stage(tmp_path, "a03", ev3, peak_bytes=75 * 1024 ** 3).returncode == 1
    assert re.search(r"^GPU_PEAK=FAIL peak_gb=75\.00 limit=72", _bc_agg(ev3).stdout, re.M)
    ev4 = tmp_path / "ev4"
    assert _bc_stage(tmp_path, "a04", ev4, readback=False).returncode == 1
    out = _bc_agg(ev4).stdout
    assert re.search(r"^SMOKE_CKPT=FAIL step=250 updates_after=49 readback=missing reasons=.*缺 CKPT250_READBACK 行", out, re.M), out
    assert re.search(r"^G6=FAIL ", out, re.M)
    # 测速窗口不全（timing 只到第 200 步）→ BENCH300=FAIL，且 250 之后没有更新 → SMOKE_CKPT=FAIL
    ev5 = tmp_path / "ev5"
    assert _bc_stage(tmp_path, "a05", ev5, n_timing=200).returncode == 1
    out = _bc_agg(ev5).stdout
    assert re.search(r"^BENCH300=FAIL reason=incomplete_or_missing", out, re.M) and "updates_after=0" in out, out


def test_g6_benchckpt_stage_fail_not_swallowed(tmp_path):
    ev = tmp_path / "ev"
    p = _bc_stage(tmp_path, "a01", ev, prefetch="1")
    assert p.returncode == 1 and re.search(r"^G6_STAGE=FAIL .*prefetch_factor=1 expect=2", p.stdout, re.M), p.stdout
    e = json.loads((ev / "g6-benchckpt-a01.json").read_text())
    assert e["bench_status"] == "PASS" and e["base_status"] == "FAIL" and e["stage_status"] == "FAIL"
    p = _bc_agg(ev)
    assert p.returncode == 1, p.stdout
    for key in ("BENCH300", "GPU_UTIL", "GPU_PEAK", "SMOKE_CKPT"):
        assert re.search(rf"^{key}=FAIL .*stage_status=FAIL stage_reasons=.*prefetch_factor=1 expect=2", p.stdout, re.M), (key, p.stdout)
    assert not re.search(r"^(BENCH300|GPU_UTIL|GPU_PEAK|SMOKE_CKPT)=PASS", p.stdout, re.M)
    assert re.search(r"^G6=FAIL ", p.stdout, re.M)
    # 新尝试通过 → 取最新通过的 a02；摘要不同 / 无证据 → 四项 FAIL + DEFERRED
    assert _bc_stage(tmp_path, "a02", ev).returncode == 0
    assert re.search(r"^G6=PASS .*attempts=benchckpt:a02$", _bc_agg(ev).stdout, re.M)
    p = _bc_agg(ev, "bbbb")
    assert p.returncode == 1 and p.stdout.count("reason=no_benchckpt_evidence pathcfg_sha=bbbb") == 4, p.stdout
    assert DEFERRED in p.stdout and re.search(r"^G6=FAIL pathcfg_sha=bbbb plan=benchckpt attempts=benchckpt:none$", p.stdout, re.M)
    # 旧口径（无 --plan）不读 benchckpt 证据
    assert "BENCH300=FAIL reason=no_bench_evidence" in _agg(ev).stdout


# ── ⑦c 正式训练前 2000 步的 SHM 判定（shm-formal 与 shmmon.sh --formal-rec）──────────


def _formal_fixture(d: pathlib.Path, t0: float, last_step=2000, before=200.0, after=201.0, pre_start=999.0, samples=True):
    rec = d / "rec"; rec.mkdir(parents=True)
    (rec / "metrics.jsonl").write_text("\n".join(json.dumps({"step": s, "wall_time": t0 + 10 + s * 2.0, "loss": {"dec": 1.0}})
                                                 for s in range(0, last_step + 1, 100)) + "\n")
    log = d / "runner.log"; log.write_text(f"TRAIN_START_EPOCH={int(t0)}\n")
    smp = d / "samples.log"
    if samples:
        split, end = t0 + 10 + 500 * 2.0, t0 + 10 + 2000 * 2.0
        out, tt = [], t0 - 300
        while tt <= end + 300:
            g = pre_start if tt < t0 else before if tt < split else after
            out.append(f"SHM_SAMPLE ts=00:00:00 shm_used_gb={g} epoch={int(tt)} gpu_util=na gpu_util_mean=na")
            tt += 15
        smp.write_text("\n".join(out) + "\n")
    return rec, log, smp


def _shm_formal(rec, log, smp):
    return run([sys.executable, "-I", str(JUDGE), "shm-formal", "--rec", str(rec), "--shmmon-samples", str(smp),
                "--run-log", str(log)])


def test_shm_formal_pass_fail_pending(tmp_path):
    t0 = 1_800_000_000.0
    rec, log, smp = _formal_fixture(tmp_path / "pass", t0)          # 训练开始前的 999 GB 采样不计入窗口
    p = _shm_formal(rec, log, smp)
    assert p.returncode == 0, p.stdout + p.stderr
    assert re.search(r"^SHM_PEAK=PASS peak_gb=201\.0 limit=330 growth_last1500=0\.50% window=formal_steps_0_2000 samples=\d+ "
                     r"split_step=500 t_start_src=TRAIN_START_EPOCH$", p.stdout, re.M), p.stdout
    assert "SHM_ALERT" not in p.stdout
    rec, log, smp = _formal_fixture(tmp_path / "grow", t0, after=210.0)
    p = _shm_formal(rec, log, smp)
    assert p.returncode == 2 and re.search(r"^SHM_PEAK=FAIL peak_gb=210\.0 limit=330 growth_last1500=5\.00% "
                                           r"window=formal_steps_0_2000 ", p.stdout, re.M), p.stdout
    assert re.search(r"^SHM_ALERT reason=formal_first2000 .*RB-D2.*E2E_PREFETCH_FACTOR=1 从头重起.*本脚本不停训", p.stdout, re.M)
    rec, log, smp = _formal_fixture(tmp_path / "over", t0, before=335.0, after=335.0)
    p = _shm_formal(rec, log, smp)
    assert p.returncode == 2 and re.search(r"^SHM_PEAK=FAIL peak_gb=335\.0 limit=330 growth_last1500=0\.00% ", p.stdout, re.M)
    rec, log, smp = _formal_fixture(tmp_path / "pend", t0, last_step=1900)
    p = _shm_formal(rec, log, smp)
    assert p.returncode == 3 and p.stdout == "", p.stdout
    # metrics.jsonl 末尾半行（训练进程正在写）→ 不崩、跳过该行仍能判
    rec, log, smp = _formal_fixture(tmp_path / "half", t0)
    with (rec / "metrics.jsonl").open("a") as f:
        f.write('{"step": 2100, "wall_ti')
    p = _shm_formal(rec, log, smp)
    assert p.returncode == 0 and re.search(r"^SHM_PEAK=PASS peak_gb=201\.0 ", p.stdout, re.M), p.stdout + p.stderr
    # 半行之前还没到第 2000 步 → 仍是未到判定点（3），不报错
    rec, log, smp = _formal_fixture(tmp_path / "halfpend", t0, last_step=1900)
    with (rec / "metrics.jsonl").open("a") as f:
        f.write('{"step": 20')
    p = _shm_formal(rec, log, smp)
    assert p.returncode == 3 and p.stdout == "", p.stdout + p.stderr


def test_shmmon_formal_judges_once(tmp_path):
    now = datetime.datetime.now().timestamp()
    env = {"SHMMON_DF_CMD": "echo 200000000000", "SHMMON_SMI_CMD": "false", "SHMMON_PY": sys.executable}
    # 已过第 2000 步：只有窗口后段的采样（训练开始前没有 shmmon 采样）→ 判 FAIL（无前段证据）一次，之后不再判
    rec, log, _ = _formal_fixture(tmp_path / "a", now - 4100, samples=False)
    smp = tmp_path / "a/samples.log"
    rec_end = json.loads((rec / "metrics.jsonl").read_text().splitlines()[-1])
    rows = [json.loads(x) for x in (rec / "metrics.jsonl").read_text().splitlines()]
    rows[-1]["wall_time"] = now + 3600          # 第 2000 步在「现在」之后：当前采样都落在第 500–2000 步之间
    (rec / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert rec_end["step"] == 2000
    args = ["bash", str(SHMMON), "--limit-gb", "293", "--interval", "0", "--max-samples", "3", "--formal-rec", str(rec),
            "--formal-run-log", str(log), "--samples-log", str(smp)]
    p = run(args, env, timeout=60)
    out = p.stdout
    assert p.returncode == 0 and out.rstrip().endswith("EXIT_CODE=0"), out + p.stderr
    assert out.count("SHM_FORMAL_JUDGED rc=2") == 1 and out.count("SHM_PEAK=FAIL reason=no_evidence") == 1, out
    assert "SHMMON_ERROR" not in out
    assert out.count("SHM_ALERT reason=formal_first2000") == 1 and "SHM_ALERT used_gb" not in out
    assert len([l for l in smp.read_text().splitlines() if l.startswith("SHM_SAMPLE ")]) == 3
    # samples 文件已存在即拒；--samples-log 不能单独给
    assert run(args, env, timeout=30).returncode == 2
    assert run(["bash", str(SHMMON), "--samples-log", str(tmp_path / "x.log"), "--max-samples", "1"], env).returncode == 2
    # 未到第 2000 步：只采样、不判
    rec2, log2, _ = _formal_fixture(tmp_path / "b", now - 100, last_step=1000, samples=False)
    p = run(["bash", str(SHMMON), "--limit-gb", "293", "--interval", "0", "--max-samples", "2", "--formal-rec", str(rec2),
             "--formal-run-log", str(log2), "--samples-log", str(tmp_path / "b/samples.log")], env, timeout=60)
    assert p.returncode == 0 and "SHM_FORMAL_JUDGED" not in p.stdout and "SHM_PEAK" not in p.stdout, p.stdout
    # 告警线 293：用量 300 GB 时每次采样都打 SHM_ALERT used_gb
    p = run(["bash", str(SHMMON), "--limit-gb", "293", "--interval", "0", "--max-samples", "1"],
            {**env, "SHMMON_DF_CMD": "echo 300000000000"}, timeout=30)
    assert "SHM_ALERT used_gb=300.00 limit_gb=293" in p.stdout


def test_shmmon_formal_judge_error_retries(tmp_path):
    """judge 退出 1 / 127（出错）→ 不置 JUDGED、打 SHMMON_ERROR，下一次采样重判；同码连续出现限频。"""
    now = datetime.datetime.now().timestamp()
    rec, log, _ = _formal_fixture(tmp_path / "f", now - 4100, samples=False)
    rows = [json.loads(x) for x in (rec / "metrics.jsonl").read_text().splitlines()]
    rows[-1]["wall_time"] = now + 3600
    (rec / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    base = {"SHMMON_DF_CMD": "echo 200000000000", "SHMMON_SMI_CMD": "false"}
    for code in (1, 127):
        # 假解释器：前两次调用以 code 退出，之后转交真解释器
        fake = tmp_path / f"py{code}"
        cnt = tmp_path / f"cnt{code}"
        fake.write_text(f"#!/usr/bin/env bash\nn=$(cat {cnt} 2>/dev/null || echo 0); echo $((n+1)) > {cnt}\n"
                        f"[ \"$n\" -ge 2 ] || {{ echo boom >&2; exit {code}; }}\nexec {shlex.quote(sys.executable)} \"$@\"\n")
        fake.chmod(0o755)
        smp = tmp_path / f"s{code}.log"
        p = run(["bash", str(SHMMON), "--limit-gb", "293", "--interval", "0", "--max-samples", "4", "--formal-rec", str(rec),
                 "--formal-run-log", str(log), "--samples-log", str(smp)], {**base, "SHMMON_PY": str(fake)}, timeout=60)
        out = p.stdout
        assert p.returncode == 0, out + p.stderr
        assert out.count(f"SHMMON_ERROR judge_rc={code} repeat=1 ") == 1 and "repeat=2" not in out, out   # 限频：连续两次只打一行
        assert out.count("SHM_FORMAL_JUDGED rc=2") == 1 and out.count("SHM_PEAK=FAIL") == 1, out          # 第 3 次采样后重判成功
        assert out.index("SHMMON_ERROR") < out.index("SHM_FORMAL_JUDGED")
        assert cnt.read_text().strip() == "3", cnt.read_text()                                             # 判过后不再调用
    # 解释器根本不存在（127）：一直出错、从不置 JUDGED
    p = run(["bash", str(SHMMON), "--limit-gb", "293", "--interval", "0", "--max-samples", "3", "--formal-rec", str(rec),
             "--formal-run-log", str(log), "--samples-log", str(tmp_path / "s-miss.log")],
            {**base, "SHMMON_PY": str(tmp_path / "no-such-python")}, timeout=60)
    assert p.returncode == 0 and "SHM_FORMAL_JUDGED" not in p.stdout, p.stdout
    assert p.stdout.count("SHMMON_ERROR judge_rc=127 ") == 1, p.stdout


# ── ⑧ shmmon ────────────────────────────────────────────────────────────────


def test_shmmon_alert(tmp_path):
    env = {"SHMMON_DF_CMD": "echo 340000000000", "SHMMON_SMI_CMD": "printf '90\\n80\\n'"}
    p = run(["bash", str(SHMMON), "--limit-gb", "330", "--interval", "0", "--max-samples", "2"], env, timeout=30)
    out = p.stdout
    assert p.returncode == 0 and out.rstrip().endswith("EXIT_CODE=0"), out
    assert re.search(r"^SHM_SAMPLE ts=\d\d:\d\d:\d\d shm_used_gb=340\.00 epoch=\d+ gpu_util=90,80 gpu_util_mean=85\.0$", out, re.M), out
    assert out.count("SHM_ALERT used_gb=340.00 limit_gb=330") == 2
    p = run(["bash", str(SHMMON), "--limit-gb", "330", "--interval", "0", "--max-samples", "1"],
            {"SHMMON_DF_CMD": "echo 200000000000", "SHMMON_SMI_CMD": "false"}, timeout=30)
    assert "SHM_ALERT" not in p.stdout and "gpu_util=na" in p.stdout


# ── ⑨ token rank ────────────────────────────────────────────────────────────


def _token_rank_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("e2e_token_rank_t", str(REPO / "scripts/training/g0/e2e_token_rank.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_token_rank_metrics():
    T = _token_rank_mod()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 16))
    assert abs(T.linear_cka(x, x) - 1) < 1e-9
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    assert abs(T.linear_cka(x, x @ q * 3.0) - 1) < 1e-9            # 正交变换与缩放不变
    assert T.linear_cka(x, rng.normal(size=(400, 16))) < 0.2
    assert 14 < T.effective_rank(x) <= 16 and T.effective_rank(np.outer(rng.normal(size=400), rng.normal(size=16))) < 1.5
    groups = [(f"t{i % 4}", i // 8) for i in range(400)]
    tr, va = T.episode_split(groups)
    assert not ({groups[i] for i in tr} & {groups[i] for i in va}) and len(tr) + len(va) == 400
    labels = [g[0] for g in groups]
    feat = np.eye(4)[[int(l[1]) for l in labels]] * 3 + rng.normal(size=(400, 4)) * 0.1
    r = T.logistic_probe(feat, np.asarray(labels), tr, va)
    assert r["val_acc"] > 0.95 and r["n_val"] == len(va)
    s = T.summarize(x.astype(np.float32), groups, labels, x.astype(np.float32), x.astype(np.float32) * 2)
    assert abs(s["cka_step0"] - 1) < 1e-6 and abs(s["rms_ratio_vs_motionmem"] - 0.5) < 1e-6


def test_token_rank_ckpt_subtree_roundtrip():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import flax.nnx as nnx
    import jax.numpy as jnp
    from mme_vla_suite.models.representation.wan_e2e_encoder import WanLatentMotionEncoderJax
    T = _token_rank_mod()
    cfg = {"layers": 1, "width": 768, "heads": 12, "mlp_ratio": 4, "dropout": 0.0, "norm": "vae_config"}
    src = WanLatentMotionEncoderJax(cfg, rngs=nnx.Rngs(params=7), dtype=jnp.bfloat16)
    pure = nnx.state(src, nnx.Param).to_pure_dict()

    def strkeys(d):        # orbax 读回的 pure dict：列表下标变成字符串键
        return {str(k): strkeys(v) for k, v in d.items()} if isinstance(d, dict) else (None if d is None else np.asarray(d))
    params = {"PaliGemma": {"x": np.zeros(3)}, "mem_encoder": {T.ENC_ATTR: strkeys(pure)}}
    enc = T.load_encoder(cfg, T.encoder_subtree(params))
    z = np.random.default_rng(1).normal(size=(2, 9, 16384)).astype(np.float32)
    a = np.asarray(src(jnp.asarray(z)).astype(jnp.float32))
    b = T.encode(enc, z.astype(jnp.bfloat16).reshape(2, 9, 16, 32, 32))
    ref = np.asarray(src(jnp.asarray(z.astype(jnp.bfloat16))).astype(jnp.float32))
    rel = float(np.linalg.norm(b - ref) / np.linalg.norm(ref))
    fresh = WanLatentMotionEncoderJax(cfg, rngs=nnx.Rngs(params=0), dtype=jnp.bfloat16)
    rel_fresh = float(np.linalg.norm(T.encode(fresh, z.astype(jnp.bfloat16).reshape(2, 9, 16, 32, 32)) - ref) / np.linalg.norm(ref))
    assert rel < 1e-2 < 0.5 < rel_fresh and a.shape == (2, 768), (rel, rel_fresh)   # 载入后与源一致（bf16 舍入内），未载入则完全不同
    with pytest.raises(SystemExit):
        bad = strkeys(pure); bad["extra"] = np.zeros(1)
        T.load_encoder(cfg, T.encoder_subtree({"mem_encoder": {T.ENC_ATTR: bad}}))


# ── ⑩ 闸门表 ─────────────────────────────────────────────────────────────────

FAKE_SMI = "#!/usr/bin/env bash\nfor a in \"$@\"; do case \"$a\" in --query-gpu=memory.used) printf '0\\n0\\n0\\n0\\n0\\n0\\n0\\n0\\n'; exit 0;; esac; done; exit 1\n"
FAKE_TMUX = "#!/usr/bin/env bash\ncase \"$1\" in has-session) exit 1;; *) echo \"$*\" >> \"$FAKE_DIR/forbidden.log\"; exit 1;; esac\n"


def test_gates_file_parsed_by_trigger(tmp_path):
    bin_ = tmp_path / "bin"; bin_.mkdir()
    (bin_ / "nvidia-smi").write_text(FAKE_SMI); (bin_ / "tmux").write_text(FAKE_TMUX)
    for f in ("nvidia-smi", "tmux"):
        (bin_ / f).chmod(0o755)
    nm = tmp_path / "nm.log"
    nm.write_text("EVAL692=PASS episodes=692 success=290\nNM_PIPELINE=PASS\nEXIT_CODE=0\n")
    locks = tmp_path / "locks"; locks.mkdir()
    p = run(["bash", str(TRIG), "--nm-log", str(nm), "--nm-tmux", "nm-pipe2", "--gates-file", str(GATES),
             "--lock-dir", str(locks), "--dry-run"],
            {"PATH": f"{bin_}:{os.environ['PATH']}", "FAKE_DIR": str(tmp_path)}, timeout=120)
    out = p.stdout
    assert p.returncode == 0, out + p.stderr
    assert "NM_DONE=PASS" in out and "GPU_FREE=PASS gpus=8" in out
    gates = re.findall(r"^TRIG_DRYRUN gate=(\S+) tmux=(\S+) attempt=(\S+) log=(\S+) ", out, re.M)
    assert [g[0] for g in gates] == ["G1", "G3", "G4", "G6-benchckpt", "G5", "G5b"], out
    for g, s, a, log in gates[1:]:
        assert s.endswith("-a01") and a == "a01" and log == f"{locks}/{s}.log", (g, s, log)
    assert gates[3][1] == "e2e-benchckpt-a01" and gates[3][3] == f"{locks}/e2e-benchckpt-a01.log"
    assert re.search(r"--benchckpt\\? --run-name\\? e2e-benchckpt-a01", out), out
    assert not re.search(r"--bench300|--shmprobe|--ckptprobe|e2e-shmprobe|e2e-ckptprobe", out), out
    assert "--smoke20" in out and "--g5b" in out and "--acceptance" in out
    # G6 gate 只读 aggregate --plan benchckpt 的 G6 行
    src = GATES.read_text()
    blk = src[src.index("gate_def G6-benchckpt "):]
    blk = blk[:blk.index("\n\n")]
    verdict = re.findall(r"'([^']*)'\s*$", blk)[-1]
    assert verdict == 'trig_verdict_all "$GATE_LOG" G6', blk
    trig_src = TRIG.read_text()
    fn = trig_src[trig_src.index("trig_verdict_all() {"):]
    fn = fn[:fn.index("\n}\n") + 3]
    for body, want in (("G6_STAGE=PASS x\nBENCH300=PASS x\nSHM_PEAK=DEFERRED reason=user_2026-11-18_formal_first2000\n"
                        "G6=PASS pathcfg_sha=a plan=benchckpt\n", "PASS"),
                       ("BENCH300=ASK x\nG6=ASK pathcfg_sha=a plan=benchckpt\n", "ASK"),
                       ("GPU_PEAK=FAIL x\nG6=FAIL pathcfg_sha=a plan=benchckpt\n", "FAIL"),
                       ("BENCH300=PASS x\n", "PENDING")):
        log = tmp_path / "g6.log"; log.write_text(body)
        v = run(["bash", "-c", fn + f'\nGATE_LOG={shlex.quote(str(log))}\n' + verdict], check=True).stdout.strip()
        assert v == want, (body, v)
    assert "launched=0" in out and not list(locks.iterdir()) and not (tmp_path / "forbidden.log").exists()


def test_zz_verdict():
    print("S5_RUNNER_UNIT=PASS")
