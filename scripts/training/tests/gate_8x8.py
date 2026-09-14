"""四个 profile 的 1000 步逐位验收；任何缺失记录都必须失败。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pathlib
import subprocess

from project_scalars import _HEADER, _KEYS, project

STATE_STEPS = (0, 2, 3, 25, 50, 100, 200, 400, 600, 800, 1000)
BATCH_STEPS = (0, 1, 2, 24, 49, 99, 199, 399, 599, 799, 999)
BASE_KEYS = {"['actions']", "['image']['base_0_rgb']", "['image']['left_wrist_0_rgb']",
             "['image_mask']['base_0_rgb']", "['image_mask']['left_wrist_0_rgb']", "['state']",
             "['static_image_emb']", "['static_mask']", "['static_pos_emb']", "['static_state_emb']",
             "['tokenized_prompt']", "['tokenized_prompt_mask']"}
MOTION_KEYS = {f"['{key}']" for key in ("motion_emb", "motion_pos", "motion_mask", "mem_order")}
NORM_SHA = "750a8e9bd6e1e5a3cf5c294864c44564153309ef92492eb083fa361096d470d2"
ORIGIN_PATHS = {"mme_vla_suite": "src/mme_vla_suite/__init__.py", "train": "scripts/training/train.py",
                "framesamp_dataset": "src/mme_vla_suite/training/framesamp_dataset.py",
                "framesamp_store": "src/mme_vla_suite/datastore/framesamp_store.py"}


def require(cond, message):
    if not cond:
        raise ValueError(message)


def keyed(rows, field):
    require(all(field in r for r in rows), f"缺 {field}")
    out = {r[field]: r for r in rows}
    require(len(out) == len(rows), f"{field} 重复")
    return out


def argv_options(argv):
    out = {}
    values = iter(argv[1:])
    for value in values:
        if not value.startswith("--"):
            require("config" not in out, "多余位置参数")
            out["config"] = value
        elif "=" in value:
            key, val = value.split("=", 1)
            require(key not in out, f"重复参数 {key}")
            out[key] = val
        elif value in ("--model.use-history", "--no-wandb-enabled"):
            require(value not in out, f"重复参数 {value}")
            out[value] = True
        else:
            require(value not in out, f"重复参数 {value}")
            out[value] = next(values)
    return out


def validate_run(run, profile, role, ref_root, cand_root):
    motion = profile.startswith("m")
    meta = run["meta"]
    require(meta["start_status"] == "", "起跑工作区不干净")
    expected_root = ref_root if role != "b" else cand_root
    require(set(meta["import_origins"]) == {"mme_vla_suite", "train", "framesamp_dataset", "framesamp_store"}, "导入来源记录不完整")
    for key, value in meta["import_origins"].items():
        require(pathlib.Path(value).resolve() == (expected_root / ORIGIN_PATHS[key]).resolve(), f"IMPORT_ORIGIN: {role}/{key}={value}")
    expected_yaml = "perceptual-framesamp-context" + ("-8frame-8x8" if profile.endswith("8") else "") + ("-motion" if motion else "") + ".yaml"
    require(meta["history_config"] == expected_yaml, "history_config 不符")
    require(meta["batch_size"] == 8 and meta["epoch_samples"] == 101066, "batch 或 400ep 口径不符")
    require(meta["bench_checksum_enabled"] and meta["bench_batch_digests_enabled"], "记录器未启用")
    args = argv_options(meta["argv"])
    for key, val in {"--num-train-steps": "1000", "--batch-size": "8", "--num-workers": "4",
                     "--seed": "42", "--fsdp-devices": "2", "--log-interval": "1", "--save-interval": "1",
                     "--model.history-config": expected_yaml, "--model.use-history": True, "--no-wandb-enabled": True}.items():
        require(args.get(key) == val, f"启动覆盖参数不符: {key}")
    expected_impl = "refnpy" if profile.endswith("8") and role != "b" else "packed"
    require(meta["dataset_impl"] == expected_impl, "参考/候选 Dataset 选择错误")
    require(meta["environment"]["CUDA_VISIBLE_DEVICES"] == ("6,7" if motion else "4,5"), "GPU 绑定不符")
    fp = run["env"]["fingerprint"]
    require(fp["assets"]["norm_stats_sha256"] == NORM_SHA, "norm_stats 文件不符")
    require(meta.get("norm_stats_file_sha256") == NORM_SHA, "实际资产文件不是指定的 norm_stats")
    require(meta.get("norm_stats_actual") == meta.get("norm_stats_expected") == fp["assets"].get("norm_stats_arrays")
            and isinstance(meta.get("norm_stats_actual"), dict) and len(meta["norm_stats_actual"]) == 8,
            "实际加载的 norm_stats 数组与指定文件解析结果不同")
    scalars = keyed(run["metrics"], "step")
    require(set(scalars) == set(range(1000)), "标量步集不完整")
    for step, row in scalars.items():
        for key in _KEYS:
            require(isinstance(row.get(key), dict) and "hex" in row[key], f"缺标量 {step}/{key}")
            value = float.fromhex(row[key]["hex"])
            require(math.isfinite(value), f"非有限标量 {step}/{key}")
            require(float(row[key]["dec"]).hex() == row[key]["hex"], f"标量十进制与 hex 不同 {step}/{key}")
    require(float.fromhex(scalars[999]["loss"]["hex"]) < float.fromhex(scalars[0]["loss"]["hex"]), "loss 未降低")
    require(scalars[999]["param_norm"]["hex"] != scalars[0]["param_norm"]["hex"], "参数范数未变化")
    require(all(float.fromhex(r["mem_enc_norm"]["hex"]) > 0 for r in scalars.values()), "记忆梯度不活跃")
    states = keyed(run["states"], "state_step")
    require(set(states) == set(STATE_STEPS), "TrainState 摘要步集不完整")
    leaf_count = 193 if motion else 177
    for step, row in states.items():
        require(row["phase"] == ("init" if step == 0 else "post_update"), "状态阶段不符")
        require(row["loop_step"] == (None if step == 0 else step - 1), "loop_step 与 state_step 错位")
        require(row["n_leaves"] == len(row["per_leaf"]) == leaf_count, "全树叶子数不符")
        require(set(row["per_leaf_finite"]) == set(row["per_leaf"]) and all(v is True for v in row["per_leaf_finite"].values()), "全树有限性失败")
        require(any(k.startswith("params") for k in row["per_leaf"]), "缺参数子树")
    require({k: v for k, v in states[0]["per_leaf"].items() if k.startswith("params")} !=
            {k: v for k, v in states[1000]["per_leaf"].items() if k.startswith("params")}, "参数没有更新")
    batches = keyed(run["batches"], "step")
    require(set(batches) == set(BATCH_STEPS), "batch 摘要步集不完整")
    indices = run["indices"]["indices"]
    require(run["indices"]["n"] == len(indices) and len(indices) >= 8072, "index 序列不足 8072")
    for step, row in batches.items():
        keys = row["per_key"]
        require(set(keys) == BASE_KEYS | MOTION_KEYS, "batch 键集不符，包含 None 的键也不能省略")
        require(all(isinstance(keys[k], str) for k in BASE_KEYS), "基础数组摘要缺失")
        require(all(isinstance(keys[k], str) if motion else keys[k] is None for k in MOTION_KEYS), "motion 开关态与 None 键不符")
        require(row["sample_indices"] == indices[step * 8:(step + 1) * 8], "batch 与 index 序列错位")
    if role == "b" and profile.endswith("8"):
        ck = meta.get("final_checkpoint")
        require(ck is not None and ck["checkpoint_id"] == 999 and ck["state_step"] == 1000
                and ck["param_kind"] == "ema" and ck["n_leaves_params"] == sum(k.startswith("params") for k in states[1000]["per_leaf"]), "最终 checkpoint 来源或步数不符")


def compare_runs(a, b, profile):
    for field, align, keys in (("metrics", "step", _KEYS), ("states", "state_step", ("per_leaf",)),
                                ("batches", "step", ("per_key", "sample_indices"))):
        ar, br = keyed(a[field], align), keyed(b[field], align)
        require(set(ar) == set(br), f"{field} 步集不同")
        for step in ar:
            for key in keys:
                va, vb = ar[step][key], br[step][key]
                if field == "metrics":
                    va, vb = va["hex"], vb["hex"]
                require(va == vb, f"{field} 不同: step={step} key={key}")
    require(a["indices"]["indices"][:8000] == b["indices"]["indices"][:8000], "前 8000 个 index 不同")
    aa, ba = argv_options(a["meta"]["argv"]), argv_options(b["meta"]["argv"])
    allowed = {"--exp-name", "--checkpoint-base-dir"}
    if profile.endswith("8"):
        allowed.add("--dataset-path")
    require({k: v for k, v in aa.items() if k not in allowed} == {k: v for k, v in ba.items() if k not in allowed}, "启动参数存在非白名单差异")
    af, bf = copy.deepcopy(a["env"]["fingerprint"]), copy.deepcopy(b["env"]["fingerprint"])
    if profile.endswith("8"):
        af["dataset"].pop("store_meta_sha256")
        bf["dataset"].pop("store_meta_sha256")
    require(af == bf, "环境指纹不同")


def load_run(root, env, log):
    def rows(name):
        return [json.loads(s) for s in (root / name).read_text().splitlines() if s.strip()]
    require(log.read_text().rstrip().endswith("EXIT_CODE=0"), f"日志未成功退出: {log}")
    manifest = json.loads((root / "BASELINE_MANIFEST.json").read_text())
    for rel, entry in manifest["files"].items():
        path = root / rel
        require(path.resolve().is_relative_to(root.resolve()), "产物清单路径越界")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"产物摘要不符: {rel}")
    projected = project(root / "metrics.jsonl")
    require((root / "scalars_hex.tsv").read_text() == projected and projected.splitlines()[0] == _HEADER, "标量投影不符")
    return {"meta": json.loads((root / "run_meta.json").read_text()), "env": json.loads(env.read_text()),
            "metrics": rows("metrics.jsonl"), "states": rows("param_checksums.jsonl"),
            "batches": rows("batch_digests.jsonl"), "indices": json.loads((root / "index_sequence.json").read_text())}


def self_test():
    # 每个负例只改变一个条件；必须到对应检查才失败，不能依靠一份本已损坏的样本。
    ref, cand = pathlib.Path("/tmp/t8-self/ref"), pathlib.Path("/tmp/t8-self/cand")
    yaml = "perceptual-framesamp-context-8frame-8x8.yaml"
    opts = {"--num-train-steps": "1000", "--batch-size": "8", "--num-workers": "4", "--seed": "42",
            "--fsdp-devices": "2", "--log-interval": "1", "--save-interval": "1", "--model.history-config": yaml}
    argv = ["bench.py", "mme_vla_suite"] + [x for kv in opts.items() for x in kv] + ["--model.use-history", "--no-wandb-enabled"]
    metrics = [{"step": i, **{k: {"hex": float(2000 - i if k in ("loss", "param_norm") else 1).hex(),
                                        "dec": float(2000 - i if k in ("loss", "param_norm") else 1)} for k in _KEYS}} for i in range(1000)]
    states = [{"state_step": s, "loop_step": None if s == 0 else s - 1, "phase": "init" if s == 0 else "post_update",
               "n_leaves": 177, "per_leaf": {f"{'params' if i < 55 else 'other'}{i}": f"{s}:{i}" for i in range(177)},
               "per_leaf_finite": {f"{'params' if i < 55 else 'other'}{i}": True for i in range(177)}} for s in STATE_STEPS]
    keys = {k: "hash" for k in BASE_KEYS} | {k: None for k in MOTION_KEYS}
    run = {"meta": {"start_status": "", "import_origins": {k: str(cand / p) for k, p in ORIGIN_PATHS.items()},
                     "history_config": yaml, "batch_size": 8, "epoch_samples": 101066,
                     "bench_checksum_enabled": True, "bench_batch_digests_enabled": True,
                     "argv": argv, "dataset_impl": "packed", "environment": {"CUDA_VISIBLE_DEVICES": "4,5"},
                     "final_checkpoint": {"checkpoint_id": 999, "state_step": 1000, "param_kind": "ema", "n_leaves_params": 55}},
           "env": {"fingerprint": {"assets": {"norm_stats_sha256": NORM_SHA}}}, "metrics": metrics, "states": states,
           "batches": [{"step": s, "per_key": keys, "sample_indices": list(range(s * 8, (s + 1) * 8))} for s in BATCH_STEPS],
           "indices": {"n": 8072, "indices": list(range(8072))}}
    norms = {f"{name}.{key}": "digest" for name in ("state", "actions") for key in ("mean", "std", "q01", "q99")}
    run["env"]["fingerprint"]["assets"]["norm_stats_arrays"] = norms
    run["meta"].update(norm_stats_file_sha256=NORM_SHA, norm_stats_actual=norms, norm_stats_expected=norms)
    validate_run(run, "c8", "b", ref, cand)
    def wrong_origin(x):
        x["meta"]["import_origins"]["framesamp_store"] = str(ref / "wrong.py")
    def missing_none(x):
        x["batches"][0]["per_key"].pop("['motion_emb']")
    def missing_state(x):
        x["states"].pop(1)
    def wrong_norm(x):
        x["env"]["fingerprint"]["assets"]["norm_stats_sha256"] = "wrong"
    def nan_scalar(x):
        x["metrics"][0]["loss"] = {"dec": float("nan"), "hex": "nan"}
    def stale_checkpoint(x):
        x["meta"]["final_checkpoint"]["state_step"] = 100
    for mutate in (wrong_origin, missing_none, missing_state, wrong_norm, nan_scalar, stale_checkpoint):
        changed = copy.deepcopy(run)
        mutate(changed)
        try:
            validate_run(changed, "c8", "b", ref, cand)
        except ValueError as exc:
            print(f"GATE_SELFTEST_REJECT=PASS case={mutate.__name__} reason={exc}")
        else:
            raise ValueError(f"负例未拒绝: {mutate.__name__}")
    print("GATE_SELFTEST=PASS cases=6")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--profile", choices=("c32", "m32", "c8", "m8"))
    for role in ("a1", "a2", "b"):
        for kind in ("run", "log", "env"):
            ap.add_argument(f"--{kind}-{role}", type=pathlib.Path)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    require(args.profile is not None and args.steps == 1000 and args.batch_size == 8, "验收固定 1000 步、batch 8")
    repo = pathlib.Path(__file__).resolve().parents[3]
    ref_root = repo / "v1-store/worktrees/ref-8x8"
    runs = []
    for role in ("a1", "a2", "b"):
        paths = [getattr(args, f"{kind}_{role}") for kind in ("run", "env", "log")]
        require(all(paths), f"{role} 的 run/env/log 均必填")
        run = load_run(*paths)
        validate_run(run, args.profile, role, ref_root, repo)
        meta = run["meta"]
        if role != "b":
            require(meta["start_head"] == meta["reference_commit"], "A 侧起跑版本不是 REF")
        else:
            changed = subprocess.check_output(["git", "diff", "--name-only", meta["candidate_commit"], meta["start_head"]], cwd=repo, text=True).splitlines()
            require(all(p.startswith("docs/") for p in changed), "CAND 到起跑 HEAD 含非文档修改")
        runs.append(run)
    compare_runs(runs[0], runs[1], args.profile)
    print(f"BASELINE_REPEAT_EXACT=PASS profile={args.profile}")
    compare_runs(runs[0], runs[2], args.profile)
    for name in ("TRAIN_1000_EXACT", "FINITE_ALIVE", "IMPORT_ORIGIN"):
        print(f"{name}=PASS profile={args.profile}")
    print(f"FINAL_STATE_EXACT=PASS profile={args.profile} state_steps=11")
    print(f"NORM_STATS_ACTUAL=PASS sha={NORM_SHA}")
    print(f"GATE_8X8=PASS profile={args.profile} steps=1000 batch=8 state_steps={list(STATE_STEPS)} digest_steps={list(BATCH_STEPS)}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, StopIteration) as exc:
        print(f"GATE_8X8=FAIL reason={exc}")
        raise SystemExit(1) from exc
