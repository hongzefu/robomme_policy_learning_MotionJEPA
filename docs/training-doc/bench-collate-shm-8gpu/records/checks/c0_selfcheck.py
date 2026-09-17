"""C0 的真实配置入口与守卫自检，训练前截停，不初始化模型。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

root = Path.cwd()
sys.path.insert(0, str(root / "scripts/training/g0"))
import bench_train_steps as bench
import guard_finish_check as guard


class Checked(Exception):
    pass


def stop_before_training():
    raise Checked()


bench._record_dir = stop_before_training
sys.argv = ["bench_train_steps.py", "mme_vla_suite_b128_60k", "--exp-name", "cs-c0-selfcheck",
            "--num-train-steps", "1", "--batch-size", "2", "--num-workers", "0",
            "--fsdp-devices", "1", "--log-interval", "1", "--no-wandb-enabled",
            "--model.history-config", "perceptual-framesamp-modul-8frame-8x8.yaml"]
try:
    bench.main()
except Checked:
    print("C0_ENTRY=PASS real_cli=1 model_started=0", flush=True)
else:
    raise AssertionError("入口未按预期截停")
sys.argv[-1] = "not-allowed.yaml"
try:
    bench.main()
except ValueError as exc:
    assert "history_config" in str(exc)
else:
    raise AssertionError("白名单未拒绝未知配置")

with tempfile.TemporaryDirectory(prefix="cs-c0-") as directory:
    tmp = Path(directory)
    sides = []
    for side in ("a1", "a2", "b"):
        d = tmp / side
        d.mkdir()
        sides.append(f"{side}={d}")
        for name, steps in (("metrics.jsonl", [0, 1, 2]), ("batch_digests.jsonl", [0, 2]),
                            ("param_checksums.jsonl", [0, 2])):
            (d / name).write_text("".join(json.dumps({"step": n}) + "\n" for n in steps))
        (d / "index_sequence.json").write_text(json.dumps({"indices": list(range(8))}))
    logs = []
    for name in ("aa", "ab"):
        p = tmp / f"{name}.log"
        p.write_text("SCALARS steps=3 keys=5 hex_mismatch_steps=0\nINDEX_SEQ=PASS n=6\n"
                     "BATCH_DIGEST rows=2 mismatch=0\nSTATE_DIGEST rows=2 mismatch=0\n"
                     "CANON_CHECK=PASS steps=2\nDET_CHECK=PASS tier=test\n")
        logs.append(str(p))
    args = SimpleNamespace(sides=sides, compare_logs=logs, steps=3, batch_size=2,
                           digest_steps="0,2", summary_prefix="TEST_GUARD")
    assert "=PASS" in guard.verify(args)
    bad_cases = 0
    for p, bad in ((tmp / "b/metrics.jsonl", '{"step":0}\n'),
                   (tmp / "b/batch_digests.jsonl", '{"step":0}\n'),
                   (tmp / "b/param_checksums.jsonl", '{"step":0}\n{"step":2}\n{"step":2}\n'),
                   (tmp / "b/index_sequence.json", '{"indices":[0,1,2]}'),
                   (tmp / "b/index_sequence.json", '{"indices":[1,0,2,3,4,5]}'),
                   (tmp / "ab.log", 'DET_CHECK=PASS\n'),
                   (tmp / "ab.log", Path(logs[1]).read_text() + 'DET_CHECK=FAIL\n')):
        original = p.read_text()
        p.write_text(bad)
        try:
            guard.verify(args)
        except ValueError:
            bad_cases += 1
        else:
            raise AssertionError(f"守卫漏报：{p}")
        finally:
            p.write_text(original)
    print(f"C0_GUARD=PASS positive=1 negative={bad_cases}", flush=True)
