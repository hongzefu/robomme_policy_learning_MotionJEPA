"""可选的最终保存现场记录；不改变训练输入、计算、checkpoint或W&B日志。"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import uuid

SCALARS = {"loss", "grad_norm", "llm_grad_norm", "param_norm", "mem_enc_norm"}


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def begin(config, wandb):
    root = Path(os.environ["TRAIN_FINAL_RECORD_DIR"]).resolve()
    repo = Path(__file__).resolve().parents[2]
    if not root.is_relative_to(repo / "v1-store"):
        raise ValueError("最终记录必须落本仓库v1-store内")
    root.mkdir(parents=True, exist_ok=True)
    from mme_vla_suite.models.config.utils import get_history_config
    from config_record import complete_record
    history = get_history_config(config.model.history_config)
    identity = {"record_version": 1, "run_uuid": str(uuid.uuid4()), "run_name": config.exp_name,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "budget": int(history.budget), "checkpoint_dir": str(config.checkpoint_dir),
        "num_train_steps": config.num_train_steps, "log_interval": config.log_interval,
        "save_interval": config.save_interval, "keep_period": config.keep_period,
        "wandb_run_id": getattr(wandb.run, "id", None), "pid": os.getpid(),
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "complete": complete_record(config)}
    write_json(root / "start.json", identity)
    print(f"FINAL_RECORD_START run={config.exp_name} uuid={identity['run_uuid']} pid={os.getpid()}", flush=True)
    return root, identity


def array_record(array):
    import numpy as np
    value = np.asarray(array)
    digest = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()
    return {"dtype": str(value.dtype), "shape": list(value.shape), "bytes": value.nbytes,
            "sha256": digest, "finite": bool(np.isfinite(value).all())}


def summarize_tree(tree):
    import jax
    from flax import traverse_util
    if hasattr(tree, "to_pure_dict"):
        tree = tree.to_pure_dict()
    return {"/".join(path): array_record(jax.device_get(value))
            for path, value in sorted(traverse_util.flatten_dict(tree).items())}


def finish(handle, state, infos, loop_step):
    import jax
    root, identity = handle
    if state.ema_params is None:
        raise ValueError("本实验要求EMA最终权重")
    tail = []
    first = loop_step + 1 - len(infos)
    for index, info in enumerate(infos, first):
        if set(info) != SCALARS:
            raise ValueError("尾窗必须含完整五标量")
        values = {key: float(value) for key, value in jax.device_get(info).items()}
        tail.append({"step": index, "scalars": {
            key: {"dec": value if math.isfinite(value) else None, "hex": value.hex(),
                  "finite": math.isfinite(value)} for key, value in values.items()}})
    means = {key: (sum(row["scalars"][key]["dec"] for row in tail) / len(tail)
                   if tail and all(row["scalars"][key]["finite"] for row in tail) else None)
             for key in sorted(SCALARS)}
    record = {**identity, "loop_step": loop_step, "state_step": int(state.step),
              "checkpoint_step": loop_step, "ema_leaves": summarize_tree(state.ema_params),
              "tail": tail, "tail_means": means,
              "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(root / "final.json", record)
    print(f"FINAL_RECORD_SAVED run={identity['run_name']} state_step={record['state_step']} tail={len(tail)}", flush=True)


def committed(handle):
    root, identity = handle
    write_json(root / "checkpoint_wait_done.json", {"run_uuid": identity["run_uuid"],
        "head": identity["head"], "wait_until_finished": True,
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()})
