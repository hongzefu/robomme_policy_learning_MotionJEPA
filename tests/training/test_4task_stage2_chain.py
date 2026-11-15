"""1114 4task 正式串行链 M→E1→C→E2（S3）：DRY 命令串、必填变量、分段收据 sha 接续与篡改拒绝。

真实跑链脚本本体；训练单条入口、合同、S5 导出、交付检查、692 评估与 tmux 都由夹具替身代替（X06：不依赖 S4/S5 文件已合入）。
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
CHAIN = "scripts/training/run_4task_concat_modul_eval_ctx_eval_50k.sh"
RM = "v2-1600ep-m8x8-modul-motion-s2rgbstate-concat1x1024-b128-50k"
RC = "v2-1600ep-m8x8-ctx-motion-s2rgbstate-concat1x1024-b128-50k"
EM, EC = "local692-s2rgbstate-modul-concat1024-50k-s7", "local692-s2rgbstate-ctx-concat1024-50k-s7"
MOD_H = "perceptual-framesamp-modul-8frame-8x8-motion-stage2-1x1024-4task-rgbstate.yaml"
CTX_H = "perceptual-framesamp-ctx-8frame-8x8-motion-stage2-1x1024-4task-rgbstate.yaml"
H0, H1, H2 = "0" * 40, "1" * 40, "2" * 40


def write(path, text, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)


def chain_tree(tmp_path):
    root = tmp_path / "repo"
    write(root / CHAIN, (REPO / CHAIN).read_text(), 0o755)
    py = root / ".venv/bin/python"
    py.parent.mkdir(parents=True)
    py.symlink_to(sys.executable)
    (root / "v1-store/logs/v16-4trs-vla").mkdir(parents=True)
    # 训练单条入口替身：写日志（RUN=/TRAIN_HEAD=/EXIT_CODE=0）、final record、10 份 ckpt 与增量上传的 INC_END。
    write(root / "scripts/training/run_orig16_stage2_60k.sh", r'''#!/usr/bin/env bash
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
echo "train $RUN_NAME $TRAIN_HEAD $HISTORY ${ATTEMPT_MANIFEST-unset}" >> "$ROOT/calls"
[[ ! -e "$ROOT/fail-train-$RUN_NAME" ]] || exit 9
LOG="$ROOT/v1-store/logs/v16-4trs-vla/$RUN_NAME.log"
printf 'RUN=%s\nSTEPS=50001\nTRAIN_HEAD=%s\nStep 50000: loss=0.25, grad_norm=0.5\nEXIT_CODE=0\n' "$RUN_NAME" "$TRAIN_HEAD" > "$LOG"
mkdir -p "$FINAL_RECORD_DIR"
printf '{"run_name": "%s", "head": "%s"}\n' "$RUN_NAME" "$TRAIN_HEAD" > "$FINAL_RECORD_DIR/final.json"
for s in 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000; do
  mkdir -p "$ROOT/v1-store/train-runs/mme_vla_suite_4task_b128_50k/$RUN_NAME/$s"
done
short=m; [[ $RUN_NAME == *-ctx-* ]] && short=c
printf 'INC_END run=%s done=10\n' "$RUN_NAME" >> "$ROOT/v1-store/logs/v16-4trs-vla/up-$short.log"
''', 0o755)
    # S1 合同替身：只实现 --check-final-record（run、head、steps 三项）。
    write(root / "scripts/training/check_4task_stage2_launch.py", r'''import argparse, json, sys
p = argparse.ArgumentParser()
p.add_argument("--check-final-record"); p.add_argument("--run"); p.add_argument("--steps", type=int)
p.add_argument("--train-head")
a = p.parse_args()
try:
    r = json.load(open(a.check_final_record))
    ok = r["run_name"] == a.run and r["head"] == a.train_head and a.steps == 50001
except OSError:
    ok = False
print(f"TRAIN_FINAL_RECORD={'PASS' if ok else 'FAIL'} run={a.run} tail_steps=0")
sys.exit(0 if ok else 1)
''')
    # 交付检查替身：核参数形状后出判定行。
    write(root / "scripts/training/check_stage2_checkpoint.py", r'''import os, sys
a = sys.argv[1:]
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
assert a[a.index("--delivery-step") + 1] == "50000" and a[a.index("--online-gpu") + 1] == "1"
ckpt, manifest = a[a.index("--checkpoint") + 1], a[a.index("--expected-manifest") + 1]
assert ckpt.endswith("/stage/50000") and manifest.endswith("/stage/manifests/step-50000.json")
assert os.path.isdir(ckpt) and os.path.isfile(manifest)
print("CKPT_DELIVERY_READY=PASS step=50000 root_complete=1 sha256_mismatch=0 strict_restore=1 online_infer=1")
''')
    for short, run in (("modul", RM), ("ctx", RC)):
        write(root / f"scripts/dataset/hf_export/run_4task_{short}8x8_concat_50k_ckpt_export.sh", f'''#!/usr/bin/env bash
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
echo "export {short} $O16_MODE" >> "$ROOT/calls"
STAGE="$ROOT/v1-store/exports/hf-ckpt-{run}/stage"
if [[ $O16_MODE == stage ]]; then
  mkdir -p "$STAGE/50000/params" "$STAGE/manifests"; echo '{{}}' > "$STAGE/manifests/step-50000.json"
  echo STAGE=PASS; echo EXIT_CODE=0
fi
''', 0o755)
    write(root / "scripts/evaluation/local_692_pool.sh", r'''#!/usr/bin/env bash
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
echo "eval $RUN_NAME $POLICY $POLICY_CONFIG $EXPECT_MOTION_BUDGET $EXPECT_MOTION_STRIDE $1 $2" >> "$ROOT/calls"
[[ -d "$1/params" ]] || exit 5
[[ ! -e "$ROOT/fail-eval-$RUN_NAME" ]] || { echo "EVAL692=FAIL reason=infra"; exit 4; }
echo "EVAL692=PASS episodes=692 success=300 rate=0.4335"
''', 0o755)
    # tmux 替身：没有会话；new-session 只登记。
    write(tmp_path / "bin/tmux", f'''#!/usr/bin/env bash
case "$1" in
  has-session) exit 1 ;;
  new-session) printf '%s\\n' "$*" >> {root}/tmux-calls; exit 0 ;;
esac
exit 64
''', 0o755)
    return root


def run_chain(tmp_path, root, **extra):
    env = {k: v for k, v in os.environ.items()
           if k not in ("MAX_STEPS", "DRY_RUN", "DRY", "START_STAGE", "ATTEMPT_MANIFEST_MODUL", "ATTEMPT_MANIFEST_CTX")}
    env.update(PATH=f"{tmp_path / 'bin'}:{env['PATH']}", FINAL_ROOT=str(root / "v1-store/validation/4task"),
               CHAIN_POLL_S="0")
    env.update(extra)
    env = {k: v for k, v in env.items() if v is not None}
    return subprocess.run(["bash", str(root / CHAIN)], env=env, capture_output=True, text=True)


def snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns if p.is_file() else 0) for p in root.rglob("*")}


def test_dry_prints_four_stages_without_side_effects(tmp_path):
    root = chain_tree(tmp_path)
    before = snapshot(root)
    result = run_chain(tmp_path, root, DRY="1", TRAIN_HEAD=H0)
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert snapshot(root) == before and not (root / "tmux-calls").exists()
    lines = [line for line in out.splitlines() if line.startswith("CHAIN_DRY ")]
    for stage in ("M", "C"):
        run, hist, short = (RM, MOD_H, "m") if stage == "M" else (RC, CTX_H, "c")
        mine = [line for line in lines if f"stage={stage} " in line]
        train = next(line for line in mine if "train_cmd=" in line)
        assert f"HISTORY={hist}" in train and f"RUN_NAME={run}" in train and f"TRAIN_HEAD={H0}" in train
        assert "run_orig16_stage2_60k.sh" in train
        final = next(line for line in mine if "final_record_cmd=" in line)
        assert "--steps 50001" in final and f"--run {run}" in final
        assert any("ckpt_steps=5000,10000,15000,20000,25000,30000,35000,40000,45000,50000" in line for line in mine)
        stage_cmd = next(line for line in mine if "stage_cmd=" in line)
        assert "O16_MODE=stage" in stage_cmd and f"log={root}/v1-store/logs/v16-4trs-vla/stage-{short}.log" in stage_cmd
        upload = next(line for line in mine if "upload_tmux=" in line)
        assert f"new-session -d -s v16-up-{short}" in upload and "O16_MODE=upload" in upload
        assert f"up-{short}-final.log" in upload
        delivery = next(line for line in mine if "delivery_cmd=" in line)
        assert "--mode delivery --delivery-step 50000" in delivery
        assert f"--checkpoint {root}/v1-store/exports/hf-ckpt-{run}/stage/50000" in delivery
        assert f"--expected-manifest {root}/v1-store/exports/hf-ckpt-{run}/stage/manifests/step-50000.json" in delivery
        assert f"--out {root}/v1-store/logs/v16-4trs-vla/delivery-{short}.json" in delivery
    evals = [line for line in lines if "eval_cmd=" in line]
    assert [line.split()[1] for line in evals] == ["stage=E1", "stage=E2"]
    names = []
    for line, run, hist in zip(evals, (RM, RC), (MOD_H, CTX_H)):
        assert "POLICY_CONFIG=mme_vla_suite_4task_b128_50k" in line
        assert "EXPECT_MOTION_BUDGET=96 EXPECT_MOTION_STRIDE=32" in line and f"POLICY={hist[:-5]}" in line
        assert line.split(" log=")[0].strip().endswith(f"{root}/v1-store/exports/hf-ckpt-{run}/stage/50000 7")
        names.append(line.split("RUN_NAME=")[1].split()[0])
    assert names == [EM, EC] and len(set(names)) == 2
    assert "CHAIN=DRY start_stage=M" in out


@pytest.mark.parametrize("missing", ["TRAIN_HEAD", "FINAL_ROOT"])
def test_dry_requires_train_head_and_final_root(tmp_path, missing):
    root = chain_tree(tmp_path)
    result = run_chain(tmp_path, root, **{"DRY": "1", "TRAIN_HEAD": H0, missing: None})
    assert result.returncode != 0 and "CHAIN_DRY" not in result.stdout
    assert f"必须指定{missing}" in result.stderr


@pytest.mark.parametrize("extra", [dict(TRAIN_HEAD="abc"), dict(TRAIN_HEAD=H0, START_STAGE="X"),
                                   dict(TRAIN_HEAD=H0, MAX_STEPS="20"), dict(TRAIN_HEAD=H0, DRY_RUN="1"),
                                   dict(TRAIN_HEAD=H0, FINAL_ROOT="relative/path")])
def test_bad_arguments_rejected(tmp_path, extra):
    root = chain_tree(tmp_path)
    result = run_chain(tmp_path, root, DRY="1", **extra)
    assert result.returncode == 2 and "CHAIN_DRY" not in result.stdout


def receipts(root):
    return {p.stem: json.loads(p.read_text()) for p in (root / "v1-store/validation/4task/receipts").glob("*.json")}


def test_full_chain_with_stubs_writes_per_stage_receipts(tmp_path):
    root = chain_tree(tmp_path)
    result = run_chain(tmp_path, root, TRAIN_HEAD=H0)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CHAIN=PASS runs=2 evals=2" in result.stdout
    calls = (root / "calls").read_text().splitlines()
    assert [" ".join(c.split()[:3 if c.startswith("export") else 2]) for c in calls] == [
        f"train {RM}", "export modul stage", f"eval {EM}", f"train {RC}", "export ctx stage", f"eval {EC}"]
    tmux = (root / "tmux-calls").read_text().splitlines()
    assert len(tmux) == 2 and "-s v16-up-m" in tmux[0] and "-s v16-up-c" in tmux[1]
    assert all("O16_MODE=upload" in line for line in tmux)
    got = receipts(root)
    assert set(got) == {f"{RM}-train", f"{RM}-delivery", f"{EM}-eval", f"{RC}-train", f"{RC}-delivery", f"{EC}-eval"}
    assert all(r["head"] == H0 for r in got.values())
    logs = root / "v1-store/logs/v16-4trs-vla"
    assert (logs / "delivery-m.log").read_text().splitlines()[:2] == [f"DELIVERY_HEAD={H0}", f"RUN={RM}"]
    assert (logs / f"{EM}.log").read_text().splitlines()[0] == f"EVAL_HEAD={H0}"
    assert sum(line.startswith("EVAL692=") for line in (logs / f"{EC}.log").read_text().splitlines()) == 1


def resolve_pending(root):
    final = root / "v1-store/validation/4task"
    for item in (final / "pending").glob("*.json"):
        (final / "decisions").mkdir(exist_ok=True)
        (final / "decisions" / item.name).write_text('{"note": "用户裁决（夹具）"}')


@pytest.fixture
def multi_head(tmp_path):
    """M、E1 在 H0 成功；C 在 H0 失败、修复后 H1 接续成功；E2 在 H1 失败。"""
    root = chain_tree(tmp_path)
    (root / f"fail-train-{RC}").touch()
    first = run_chain(tmp_path, root, TRAIN_HEAD=H0)
    assert first.returncode == 9 and f"WAIT_USER_DECISION run={RC} stage=train_exit" in first.stdout
    (root / f"fail-train-{RC}").unlink()
    resolve_pending(root)
    (root / f"fail-eval-{EC}").touch()
    second = run_chain(tmp_path, root, TRAIN_HEAD=H1, START_STAGE="C")
    assert second.returncode == 3, second.stdout + second.stderr
    assert f"CHAIN_RESUME=PASS stage=train run={RM} head={H0}" in second.stdout
    assert f"CHAIN_RESUME=PASS stage=eval run={RM} eval={EM} head={H0}" in second.stdout
    assert f"WAIT_USER_DECISION run={RC} stage=eval692" in second.stdout
    resolve_pending(root)
    got = receipts(root)
    assert got[f"{RM}-train"]["head"] == got[f"{EM}-eval"]["head"] == H0
    assert got[f"{RC}-train"]["head"] == got[f"{RC}-delivery"]["head"] == H1
    return tmp_path, root


@pytest.mark.parametrize("start", ["C", "E2"])
def test_resume_checks_each_stage_by_its_own_sha(multi_head, start):
    tmp_path, root = multi_head
    result = run_chain(tmp_path, root, DRY="1", TRAIN_HEAD=H2, START_STAGE=start)
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert f"CHAIN_RESUME=PASS stage=train run={RM} head={H0}" in out
    assert f"CHAIN_RESUME=PASS stage=delivery run={RM} head={H0}" in out
    assert f"CHAIN_RESUME=PASS stage=eval run={RM} eval={EM} head={H0}" in out
    if start == "E2":
        assert f"CHAIN_RESUME=PASS stage=train run={RC} head={H1}" in out
        assert f"CHAIN_RESUME=PASS stage=delivery run={RC} head={H1}" in out
        assert "stage=C train_cmd=" not in out
    else:
        assert f"stage=C train_cmd=" in out and f"TRAIN_HEAD={H2}" in out
    assert "stage=E2 eval_cmd=" in out and f"CHAIN=DRY start_stage={start}" in out


@pytest.mark.parametrize("target,field,value", [
    (f"{RM}-train", "head", H1),          # 收据 sha 与日志自记 TRAIN_HEAD 不符
    (f"{EM}-eval", "head", H2),           # 评估收据 sha 被改成当前 HEAD
    (f"{RC}-delivery", "head", H0),
    (f"{RM}-delivery", "log_sha256", "f" * 64),
])
def test_tampered_receipt_is_rejected(multi_head, target, field, value):
    tmp_path, root = multi_head
    path = root / "v1-store/validation/4task/receipts" / f"{target}.json"
    body = json.loads(path.read_text())
    body[field] = value
    path.write_text(json.dumps(body))
    result = run_chain(tmp_path, root, DRY="1", TRAIN_HEAD=H2, START_STAGE="E2")
    assert result.returncode == 3 and "WAIT_USER_DECISION" in result.stdout
    assert "stage=E2 eval_cmd=" not in result.stdout


def test_tampered_log_head_line_is_rejected(multi_head):
    tmp_path, root = multi_head
    log = root / "v1-store/logs/v16-4trs-vla" / f"{EM}.log"
    log.write_text(log.read_text().replace(f"EVAL_HEAD={H0}", f"EVAL_HEAD={H2}"))
    result = run_chain(tmp_path, root, DRY="1", TRAIN_HEAD=H2, START_STAGE="C")
    assert result.returncode == 3 and f"WAIT_USER_DECISION run={RM} stage=resume_eval_receipt" in result.stdout


def test_resume_without_receipts_waits_for_user(tmp_path):
    root = chain_tree(tmp_path)
    result = run_chain(tmp_path, root, DRY="1", TRAIN_HEAD=H0, START_STAGE="E1")
    assert result.returncode == 3 and "stage=resume_train_receipt" in result.stdout
    assert not (root / "v1-store/validation/4task/pending").exists()     # DRY 不写待决文件
