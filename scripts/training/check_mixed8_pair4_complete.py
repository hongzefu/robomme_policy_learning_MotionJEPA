#!/usr/bin/env python
"""四对 modulation 正式 80k（S7）与 S6 短测的完成检查；不沿用旧 61 叶 check_modul_completion.py。

核对：final-record 三件套（start/final/checkpoint_wait_done）同 run_uuid/head；
``final.json`` 的 ``state_step``=步数、``loop_step``=``checkpoint_step``=步数−1；
预期 checkpoint 目录集合（正式为 5000…75000 与 79999 共 16 个）完整且无残留临时目录；
每个目录 ``_CHECKPOINT_METADATA.commit_timestamp_nsecs`` 有效、数据与 norm 资产齐全；
最终目录的 65 叶 EMA 以 ``restore_params(..., dtype=None)`` 原 dtype 严格回读，
逐叶 dtype/shape/bytes/SHA/finite 与同 run_uuid 的 ``final.json.ema_leaves`` 相同，
且键集/shape/dtype 等于已验四对冒烟的 65 叶清单。

末行：S7 模式 ``POLICY_DONE=PASS|FAIL``；S6 模式 ``POLICY_8GPU_RECORD=PASS|FAIL``。
本检查只证明权重严格回读，不证明 optimizer 可续训或在线四对接口可用。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check_mixed8_pair4_launch import (  # noqa: E402 -- 同目录工具
    Contract, MOTION_PROJ, Recorder, reference_leaves, sha256_file)
from final_record import summarize_tree  # noqa: E402

FORMAL_STEPS = 80000
SAVE_INTERVAL = 5000


def load(path):
    return json.loads(Path(path).read_text())


def expected_steps_dirs(steps, interval=SAVE_INTERVAL):
    return sorted(set(range(interval, steps, interval)) | {steps - 1})


def check_records(rec, start, final, wait, *, run_root, steps, run=None, head=None):
    ids = (start.get("run_uuid"), final.get("run_uuid"), wait.get("run_uuid"))
    try:
        uuid.UUID(str(ids[0]))
        valid = True
    except ValueError:
        valid = False
    rec("RECORD_RUN_UUID", valid and len(set(ids)) == 1, f"start/final/wait={ids}")
    heads = (start.get("head"), final.get("head"), wait.get("head"))
    rec("RECORD_HEAD", len(set(heads)) == 1 and re.fullmatch(r"[0-9a-f]{40}", str(heads[0])) is not None
        and (head is None or heads[0] == head), f"start/final/wait={heads} 期望={head}")
    if run is not None:
        rec("RECORD_RUN_NAME", start.get("run_name") == final.get("run_name") == run,
            f"start={start.get('run_name')} final={final.get('run_name')} 期望={run}")
    same = all(start.get(k) == final.get(k) for k in
               ("run_name", "checkpoint_dir", "budget", "num_train_steps", "save_interval", "keep_period", "complete"))
    rec("RECORD_IDENTITY", same and Path(str(start.get("checkpoint_dir"))).resolve() == Path(run_root).resolve(),
        f"start与final身份一致={same} checkpoint_dir={start.get('checkpoint_dir')} 被验={run_root}")
    rec("RECORD_WAIT_DONE", wait.get("wait_until_finished") is True, f"wait_until_finished={wait.get('wait_until_finished')}")
    rec("RECORD_STEPS", start.get("num_train_steps") == steps and final.get("state_step") == steps
        and final.get("loop_step") == final.get("checkpoint_step") == steps - 1,
        f"num_train_steps={start.get('num_train_steps')} state_step={final.get('state_step')} "
        f"loop_step={final.get('loop_step')} checkpoint_step={final.get('checkpoint_step')} 期望={steps}/{steps - 1}")
    rec("RECORD_SAVE_INTERVAL", start.get("save_interval") == start.get("keep_period") == SAVE_INTERVAL,
        f"save/keep={start.get('save_interval')}/{start.get('keep_period')}")


def check_checkpoint_dirs(rec, run_root, steps, contract):
    run_root = Path(run_root)
    entries = list(run_root.iterdir())
    names = sorted(int(p.name) for p in entries if p.is_dir() and p.name.isdecimal())
    want = expected_steps_dirs(steps)
    rec("CKPT_DIR_SET", names == want, f"期望{len(want)}个={want[:3]}…{want[-2:]} 实际{len(names)}个 缺={sorted(set(want) - set(names))} 多={sorted(set(names) - set(want))}")
    tmp = [p.name for p in entries if "orbax-checkpoint-tmp" in p.name or p.name.endswith(".tmp")]
    rec("CKPT_NO_TMP", not tmp, f"残留临时目录={tmp}")
    bad = []
    for name in names:
        path = run_root / str(name)
        try:
            meta = load(path / "_CHECKPOINT_METADATA")
            committed = int(meta["commit_timestamp_nsecs"]) > int(meta["init_timestamp_nsecs"]) > 0
        except (OSError, ValueError, KeyError, TypeError):
            committed = False
        norm = path / "assets/robomme/norm_stats.json"
        files = ((path / "params/manifest.ocdbt").is_file() and (path / "params/_METADATA").is_file()
                 and norm.is_file() and sha256_file(norm) == contract.norm_sha256)
        if not (committed and files):
            bad.append(name)
    rec("CKPT_COMMITTED", not bad, f"未提交完成或数据/norm不全的目录={bad}")


def check_final_leaves(rec, final, reference):
    leaves = final.get("ema_leaves", {})
    rec("FINAL_EMA_65", len(leaves) == 65 and all(row.get("finite") is True for row in leaves.values()),
        f"final.json ema_leaves={len(leaves)} 全有限={all(row.get('finite') is True for row in leaves.values())}")
    shapes = {k: (row.get("shape"), row.get("dtype")) for k, row in leaves.items()}
    rec("FINAL_EMA_TREE", shapes == {k: (s, d) for k, (s, d) in reference.items()},
        f"键/shape/dtype 等于四对冒烟65叶 {MOTION_PROJ}={shapes.get(MOTION_PROJ)}")


def check_restored(rec, final, restored_summary):
    rec("RESTORE_65_FINITE", len(restored_summary) == 65 and all(r["finite"] for r in restored_summary.values()),
        f"回读叶数={len(restored_summary)}")
    diff = sorted(k for k in set(restored_summary) | set(final.get("ema_leaves", {}))
                  if restored_summary.get(k) != final["ema_leaves"].get(k))
    rec("RESTORE_EQUALS_FINAL", not diff, f"原dtype回读与final.json逐叶dtype/shape/bytes/SHA/finite不同={diff[:5]}")


def restore_summary(checkpoint):
    import numpy as np
    from openpi.models.model import restore_params
    return summarize_tree(restore_params(Path(checkpoint) / "params", restore_type=np.ndarray, dtype=None))


def check_pid_exited(rec, start):
    pid = start.get("pid")
    try:
        os.kill(int(pid), 0)
        alive = True
    except (ProcessLookupError, TypeError, ValueError):
        alive = False
    except PermissionError:
        alive = True
    rec("TRAIN_PID_EXITED", not alive, f"训练pid={pid} 仍存活={alive}")


def run(args, contract=Contract(), restore=restore_summary):
    rec = Recorder()
    records, run_root = Path(args.records), Path(args.run_root)
    docs = rec.guard("RECORD_FILES", lambda: tuple(load(records / n) for n in
                                                   ("start.json", "final.json", "checkpoint_wait_done.json")))
    if docs is not None:
        start, final, wait = docs
        rec.guard("RECORD", check_records, rec, start, final, wait, run_root=run_root, steps=args.steps,
                  run=args.run, head=args.head)
        if not args.skip_pid:
            check_pid_exited(rec, start)
    rec.guard("CKPT", check_checkpoint_dirs, rec, run_root, args.steps, contract)
    reference = rec.guard("REFERENCE", reference_leaves, args.repo, contract)
    if docs is not None and reference is not None:
        rec.guard("FINAL_EMA", check_final_leaves, rec, docs[1], reference)
        checkpoint = run_root / str(args.steps - 1)
        summary = rec.guard("RESTORE", restore, checkpoint) if checkpoint.is_dir() else None
        if summary is None:
            rec("RESTORE_65_FINITE", False, f"最终目录不可回读 {checkpoint}")
        else:
            rec.guard("RESTORE", check_restored, rec, docs[1], summary)
    failed = rec.failed
    verdict = "FAIL" if failed else "PASS"
    uid = docs[0].get("run_uuid") if docs else None
    if args.mode == "s7":
        print(f"POLICY_DONE={verdict} steps={args.steps} checkpoints={len(expected_steps_dirs(args.steps))} "
              f"run_uuid={uid} failed={','.join(failed) or 0}", flush=True)
    else:
        print(f"POLICY_8GPU_RECORD={verdict} steps={args.steps} final_dir={args.steps - 1} "
              f"run_uuid={uid} failed={','.join(failed) or 0}", flush=True)
    return rec


def main(raw=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="TRAIN_FINAL_RECORD_DIR")
    parser.add_argument("--run-root", required=True, help="v1-store/train-runs/mme_vla_suite_b128_80k/<run>")
    parser.add_argument("--mode", choices=("s6", "s7"), default="s7")
    parser.add_argument("--steps", type=int, default=None, help="S7 固定80000；S6 填实际短测步数")
    parser.add_argument("--run")
    parser.add_argument("--head")
    parser.add_argument("--repo", default=str(HERE.parents[1]), help="含冒烟65叶参考文件的仓库根")
    parser.add_argument("--skip-pid", action="store_true", help="仅供离线测试")
    args = parser.parse_args(raw)
    if args.steps is None:
        args.steps = FORMAL_STEPS if args.mode == "s7" else None
    if args.steps is None or (args.mode == "s7" and args.steps != FORMAL_STEPS) or not 0 < args.steps <= FORMAL_STEPS:
        parser.error("S7 步数固定80000；S6 必须显式给出 1..79999 的实际短测步数")
    if args.mode == "s6" and args.steps == FORMAL_STEPS:
        parser.error("S6 短测步数不能等于正式80000")
    return 1 if run(args).failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
