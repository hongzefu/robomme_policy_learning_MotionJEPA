"""验证本run真实末步保存、尾窗和固定动作；成功记录是顺序训练的交接依据。"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/training"))
from final_record import SCALARS, summarize_tree, write_json
from config_record import file_record


def require(ok, why):
    if not ok:
        raise ValueError(why)


def load(path):
    return json.loads(Path(path).read_text())


def validate_records(start, final, wait, metrics, exits, names, *, run, head, budget, steps, interval):
    expected = {"run_name": run, "head": head, "budget": budget, "num_train_steps": steps, "log_interval": interval}
    require(all(start.get(k) == final.get(k) == value for k, value in expected.items()), "运行身份或配置不同")
    require(start["complete"] == final["complete"], "保存前后完整配置不同")
    uuid.UUID(start["run_uuid"])
    require(start["run_uuid"] == final["run_uuid"] == wait["run_uuid"], "运行UUID不同")
    require(start["wandb_run_id"] == final["wandb_run_id"], "W&B运行身份不同")
    require(wait["head"] == head and wait["wait_until_finished"] is True, "没有异步保存完成记录")
    require(exits == ["EXIT_CODE=0"], "必须有唯一成功终态")
    expected_names = sorted(set(range(5000, steps, 5000)) | {steps-1})
    require(names == expected_names, "checkpoint目录集合不完整或有多余目录")
    require(final["state_step"] == steps and final["loop_step"] == final["checkpoint_step"] == steps-1, "最终实际更新次数不同")
    require([row["step"] for row in metrics] == list(range(0, steps, interval)), "正常日志步集合不完整")
    for row in metrics:
        require(set(row)-{"step", "wall_time"} == SCALARS, "日志不是完整五标量")
        for key in SCALARS:
            value = row[key]
            require(math.isfinite(value["dec"]) and float(value["dec"]).hex() == value["hex"], "日志标量非有限或摘要不同")
    first = (steps-1)//interval*interval+1
    require([row["step"] for row in final["tail"]] == list(range(first, steps)), "最终尾窗不完整")
    for row in final["tail"]:
        require(set(row["scalars"]) == SCALARS, "尾窗缺标量")
        for value in row["scalars"].values():
            require(value["finite"] is True and value["dec"] is not None and math.isfinite(value["dec"])
                    and float(value["dec"]).hex() == value["hex"], "尾窗存在非有限值或内容错配")
    for key in SCALARS:
        want = sum(row["scalars"][key]["dec"] for row in final["tail"])/len(final["tail"]) if final["tail"] else None
        require(final["tail_means"][key] == want, "尾窗均值不符")
    require(len(final["ema_leaves"]) == 61 and all(row["finite"] is True for row in final["ema_leaves"].values()), "保存现场EMA非有限或叶数错误")


def validate_arrays(final, restored):
    observed = summarize_tree(restored)
    require(len(observed) == 61 and all(row["finite"] for row in observed.values()), "真实checkpoint含NaN/Inf或叶集错误")
    require(observed == final["ema_leaves"], "真实checkpoint数组不同于本run保存现场EMA")
    return observed


def action_check(checkpoint, budget):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from mme_vla_suite.training.config import get_config
    from mme_vla_suite.policies.policy_config import create_trained_policy, _load_resolved_snapshot
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    from openpi.training.data_loader import transform_dataset, _collate_fn
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset
    import openpi.shared.array_typing as at

    history, motion = _load_resolved_snapshot(checkpoint.parent)
    require(history.budget == budget and not motion, "真实加载快照预算或motion不符")
    root = ROOT / "v1-store"
    dataset = root / "datasets/4task-v2-1600ep-604f16da"
    assets = root / "train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
    config = get_config("mme_vla_suite_b128_80k")
    config = dataclasses.replace(config, model=dataclasses.replace(config.model,history_config=history),
        data=dataclasses.replace(config.data,assets=dataclasses.replace(config.data.assets,assets_dir=str(assets),asset_id="robomme")))
    policy = create_trained_policy(config, checkpoint, seed=42)
    data = config.data.create(config.assets_dirs, config.model)
    samples = FrameSampDataset(dataset/"framesamp-8x8",data,history,20,
                              source_root=dataset/"source",manifest_path=dataset/"meta/episode_manifest.json")
    try:
        batch = _collate_fn([transform_dataset(samples,data)[0]])
        obs = HistAugObservation.from_dict(jax.tree.map(jnp.asarray,batch))
    finally:
        samples.close()
    noise = jnp.asarray(np.random.default_rng(42).normal(size=(1,20,32)).astype(np.float32))
    @nnx.jit
    def actions(model, observation):
        with at.disable_typechecking():
            return model.sample_actions(jax.random.key(9),observation,noise=noise,num_steps=10)
    output = np.asarray(actions(policy._model,obs))
    require(output.shape == (1,20,32) and np.isfinite(output).all(), "真实动作输出非有限或形状错误")
    return {"sample_index":0,"noise_seed":42,"action_seed":9,"num_steps":10,
            "shape":list(output.shape),"sha256":hashlib.sha256(output.tobytes()).hexdigest(),"finite":True}


def check(args):
    import numpy as np
    from openpi.models.model import restore_params
    records = Path(args.records).resolve()
    run_root = Path(args.run_root).resolve()
    start, final, wait = (load(records/name) for name in ("start.json","final.json","checkpoint_wait_done.json"))
    require(Path(start["checkpoint_dir"]).resolve() == Path(final["checkpoint_dir"]).resolve() == run_root, "记录的输出根不是被验收run")
    metrics_path = Path(args.metrics)
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    log = Path(args.log)
    log_lines = log.read_text().splitlines()
    exits = [line for line in log_lines if line.startswith("EXIT_CODE=")]
    for prefix in ("GPU_SAMPLER_PID=", "TRAIN_WRAPPER_PID="):
        pids = [int(line.removeprefix(prefix)) for line in log_lines if line.startswith(prefix)]
        require(len(pids) == 1, "缺唯一的训练或GPU采样PID")
        try:
            os.kill(pids[0],0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError(f"本轮进程仍存在，禁止交接: {pids[0]}")
    require(any(line.startswith("GPU_SAMPLER_STOP ") for line in log_lines), "GPU采样未正常收尾")
    try:
        os.kill(start["pid"],0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("保存现场记录的真实训练PID尚未退出")
    names = sorted(int(path.name) for path in run_root.iterdir() if path.is_dir() and path.name.isdecimal())
    require(not any("orbax-checkpoint-tmp" in p.name for p in run_root.iterdir()), "残留未完成checkpoint")
    validate_records(start,final,wait,metrics,exits,names,run=args.run,head=args.head,budget=args.budget,steps=args.steps,interval=args.log_interval)
    for name in names:
        path = run_root/str(name)
        metadata = load(path/"_CHECKPOINT_METADATA")
        require(metadata["commit_timestamp_nsecs"] > metadata["init_timestamp_nsecs"] > 0, "checkpoint没有提交完成")
        require((path/"params/manifest.ocdbt").is_file() and (path/"params/_METADATA").is_file()
                and (path/"assets/robomme/norm_stats.json").is_file(), "checkpoint数据或资产不全")
    checkpoint = run_root/str(args.steps-1)
    restored = restore_params(checkpoint/"params",restore_type=np.ndarray,dtype=None)
    validate_arrays(final,restored)
    del restored
    actions = action_check(checkpoint,args.budget)
    files = [records/name for name in ("start.json","final.json","checkpoint_wait_done.json")]
    files += [metrics_path,log,run_root/"motion_provenance.json",run_root/"history_config.resolved.yaml",run_root/"history_config.resolved.sha256"]
    files += sorted(p for p in checkpoint.rglob("*") if p.is_file())
    result = {"status":"PASS","run_name":args.run,"head":args.head,"budget":args.budget,"steps":args.steps,
              "run_uuid":start["run_uuid"],"run_root":str(run_root),"checkpoints":names,"actions":actions,
              "files":[file_record(path) for path in files]}
    write_json(args.out,result)
    print(f"RUN_COMPLETED=PASS run={args.run} budget={args.budget} steps={args.steps} checkpoints={len(names)} final={args.steps-1}",flush=True)


def verify_handoff(path, head):
    result = load(path)
    require((result.get("status"),result.get("run_name"),result.get("head"),result.get("budget"),result.get("steps")) ==
            ("PASS","v2-1600ep-m64x8x8-modul-b128-80k",head,4096,80000), "交接记录不是本轮4096完整训练")
    require(result["checkpoints"] == [*range(5000,80000,5000),79999] and result["actions"]["finite"], "交接验收不完整")
    run = "v2-1600ep-m64x8x8-modul-b128-80k"
    expected_root = ROOT/"v1-store/train-runs/mme_vla_suite_b128_80k"/run
    require(Path(result["run_root"]).resolve() == expected_root, "交接输出根不是本轮4096")
    records = ROOT/"v1-store/bench/modul-budget-sweep/runs"/run/"final"
    start, final = load(records/"start.json"),load(records/"final.json")
    require(all(record["run_name"] == run and record["head"] == head and record["budget"] == 4096
                and record["run_uuid"] == result["run_uuid"] for record in (start,final)), "交接实际记录身份不符")
    require(final["state_step"] == 80000, "交接实际记录没有完成80000步")
    require(result["files"] and len({row["path"] for row in result["files"]}) == len(result["files"]), "交接文件清单为空或重复")
    for row in result["files"]:
        require(file_record(row["path"]) == row, "交接后产物已改变")
    require({str(records/name) for name in ("start.json","final.json","checkpoint_wait_done.json")} <= {row["path"] for row in result["files"]}, "交接缺保存现场文件摘要")
    require(str(Path(result["run_root"])/"79999/params/_METADATA") in {r["path"] for r in result["files"]}, "交接缺最终权重摘要")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-handoff")
    parser.add_argument("--head",required=True)
    for key in ("records","run-root","metrics","log","run","out"):
        parser.add_argument("--"+key)
    parser.add_argument("--budget",type=int,choices=(1024,2048,4096))
    parser.add_argument("--steps",type=int,default=80000)
    parser.add_argument("--log-interval",type=int,default=100)
    args = parser.parse_args()
    try:
        if args.verify_handoff:
            verify_handoff(args.verify_handoff,args.head)
            print("HANDOFF_4096=PASS",flush=True)
        else:
            require(all(getattr(args,k.replace("-","_")) for k in ("records","run-root","metrics","log","run","out","budget")), "缺验收参数")
            check(args)
        return 0
    except (ValueError,TypeError,KeyError,OSError) as error:
        print(f"RUN_COMPLETED=FAIL reason={error}",flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
