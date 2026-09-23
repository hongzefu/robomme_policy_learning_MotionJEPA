"""按明确步集合核验真实训练记录，不以两侧交集冒充完整证据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[3]

SCALARS = {"loss", "grad_norm", "llm_grad_norm", "param_norm", "mem_enc_norm"}
NORM_SHA = "856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173"
MEM_LEAVES = (
    "PaliGemma/llm/layers/mem_attn/q_einsum_mem/w",
    "PaliGemma/llm/layers/mem_attn/kv_einsum_mem/w",
    "PaliGemma/llm/layers/mem_attn/out_einsum_mem/w",
    "PaliGemma/llm/layers/mem_attn/mem_rms_norm/scale",
    "PaliGemma/llm/layers/mem_rms_norm_ffn/Dense_0/kernel",
    "PaliGemma/llm/layers/mem_rms_norm_ffn/Dense_0/bias",
    "mem_encoder/feature_encoder/pos_proj/kernel",
    "mem_encoder/feature_encoder/pos_proj/bias",
    "mem_encoder/feature_encoder/encoder_static/kernel",
    "mem_encoder/feature_encoder/encoder_static/bias",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def flag(argv, key):
    matches = [argv[i + 1] if x == key else x.split("=", 1)[1]
               for i, x in enumerate(argv) if x == key or x.startswith(key + "=")]
    require(len(matches) == 1, f"参数必须明确且唯一: {key}")
    return matches[0]


def state_records(root, expected):
    states = rows(root / "param_checksums.jsonl")
    require([r["state_step"] for r in states] == expected, "TrainState 更新步集合缺失、重复或乱序")
    keys = set(states[0]["per_leaf"])
    require(keys and any(k.startswith("opt_state") for k in keys), "没有完整优化器叶集合")
    params={leaf_path(k) for k in keys if k.startswith("params")}
    ema={leaf_path(k) for k in keys if k.startswith("ema_params")}
    trainable={k for k in params if "img" not in k}
    mu={leaf_path(k) for k in keys if k.startswith("opt_state") and ".mu" in k}
    nu={leaf_path(k) for k in keys if k.startswith("opt_state") and ".nu" in k}
    counters={k for k in keys if not k.startswith(("params","ema_params")) and ".mu" not in k and ".nu" not in k}
    require(len(keys)==201 and len(params)==61 and params==ema and len(trainable)==38 and mu==nu==trainable
            and counters=={"step","opt_state[1][0].count","opt_state[1][2].count"},
            "无motion modulation预期61参数/61EMA/38mu/38nu/3计数叶不完整")
    for row in states:
        step = row["state_step"]
        require(row["phase"] == ("init" if step == 0 else "post_update"), "状态 phase 与更新步数不符")
        require(row["loop_step"] == (None if step == 0 else step - 1), "循环编号与更新编号不符")
        require(set(row["per_leaf"]) == set(row["per_leaf_finite"]) == keys, "逐叶记录集合发生变化")
        require(row["n_leaves"] == len(keys), "状态叶数量不符")
        require(all(v is True for v in row["per_leaf_finite"].values()), "状态存在非有限叶")
        digest = hashlib.sha256()
        for key in sorted(keys):
            digest.update(f"{key}:{row['per_leaf'][key]}\n".encode())
        require(digest.hexdigest() == row["state_digest"], "完整状态摘要与逐叶摘要不一致")
    return states


def validate(root, steps, batch_size, expected_devices="4,5", expected_workers=4, expected_fsdp=2, budget=2048):
    devices = expected_devices.split(",")
    require(len(devices) == len(set(devices)) == expected_fsdp and all(x in tuple(map(str, range(8))) for x in devices),
            "必须显式指定与FSDP一致的不重复物理GPU编号")
    root = Path(root)
    meta = load(root / "run_meta.json")
    require(meta["start_status"] == meta["source_status"] == "", "起跑工具或源码工作区不干净")
    require(all(re.fullmatch(r"[0-9a-f]{40}",meta[key]) for key in ("start_head","source_head")), "源码或取证器锚点不完整")
    if meta["start_head"] != meta["source_head"]:
        require(meta["reference_commit"] == meta["source_head"] and meta["candidate_commit"] == meta["start_head"], "旧代码对照未明确绑定两侧提交")
    source = Path(meta["source_root"]).resolve()
    require(set(meta["import_origins"]) == {"train", "framesamp_dataset", "framesamp_store", "mme_vla_suite"}, "导入来源不完整")
    require(all(Path(p).resolve().is_relative_to(source) for p in meta["import_origins"].values()), "导入源码越出记录仓库")
    require(meta["norm_stats_file_sha256"] == NORM_SHA, "norm_stats 文件不是已定资产")
    require(meta["norm_stats_actual"] == meta["norm_stats_expected"] and len(meta["norm_stats_actual"]) == 8,
            "真实 loader 的归一化数组与磁盘资产不同")
    if "actual_config" in meta:
        complete = meta["actual_config"]
        require(complete["config_record_version"] == 2, "需要完整实际解析配置")
        actual = complete["train_config"]["fields"]
        runtime = load(root / "runtime.json")
        loader = meta["loader"]
        require(actual["num_train_steps"] == steps, "实际步数不符")
        require(actual["batch_size"] == runtime["batch_size"] == loader["batch_size"] == batch_size == meta["batch_size"], "实际batch不符")
        require(actual["num_workers"] == runtime["num_workers"] == loader["workers"] == expected_workers, "实际worker不符")
        require(actual["seed"] == runtime["seed"] == 42, "实际种子不符")
        require(actual["fsdp_devices"] == runtime["fsdp_devices"] == runtime["mesh"]["fsdp"] == runtime["device_count"] == expected_fsdp, "实际mesh不符")
        require(runtime["cuda_visible_devices"] == expected_devices and len(set(runtime["devices"])) == expected_fsdp, "真实设备列表不符")
        require(complete["history_values"]["budget"] == budget and type(complete["history_values"]["budget"]) is int, "实际预算不符")
        require(loader["prefetch_factor"] == 2 and loader["persistent_workers"] and not loader["pin_memory"], "真实loader预取方式不符")
    else:
        # 保留历史两卡显式CLI记录的回归，新的八卡证据不得走此兼容分支。
        require(expected_fsdp == 2 and expected_workers == 4 and budget == 2048, "新档缺完整配置记录")
        require(int(flag(meta["argv"], "--num-train-steps")) == steps, "实际步数不符")
        require(int(flag(meta["argv"], "--batch-size")) == batch_size == meta["batch_size"], "实际batch不符")
        require(int(flag(meta["argv"], "--num-workers")) == expected_workers, "实际worker不符")
        require(int(flag(meta["argv"], "--seed")) == 42, "种子不符")
        require(int(flag(meta["argv"], "--fsdp-devices")) == expected_fsdp, "验证mesh不符")
    workers = expected_workers
    require(meta["environment"]["CUDA_VISIBLE_DEVICES"] == expected_devices, "验证使用的物理GPU不同")
    require(meta["environment"]["XLA_FLAGS"] == "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0", "确定性环境不同")
    require(meta["bench_checksum_enabled"] and meta["bench_batch_digests_enabled"] and meta["bench_dump_idx_enabled"],
            "有记录器未启用")
    require(len(meta["history_config_resolved_sha256"]) == 64, "缺解析后配置摘要")
    metrics = rows(root / "metrics.jsonl")
    require([r["step"] for r in metrics] == list(range(steps)), "五标量步集合不完整")
    for row in metrics:
        require(set(row) - {"step", "wall_time"} == SCALARS, "标量集合不是完整五键")
        for name in SCALARS:
            v = row[name]
            require(math.isfinite(v["dec"]) and float(v["dec"]).hex() == v["hex"], f"标量非有限或hex不符: {name}")
        require(row["mem_enc_norm"]["dec"] > 0, "记忆梯度范数没有严格为正")
    expected_states = [0, 1] if steps == 1 else [0, *range(2, steps + 1)]
    states = state_records(root, expected_states)
    batches = rows(root / "batch_digests.jsonl")
    require([r["step"] for r in batches] == list(range(steps)), "输入摘要步集合不完整")
    indices = load(root / "index_sequence.json")
    idx_rows = rows(root / "idx_seq.jsonl")
    expected_batches = steps + 1 + workers * 2
    require(indices["n"] == len(indices["indices"]) == expected_batches * batch_size, "索引记录含预取后的数量不符")
    require(hashlib.sha256(json.dumps(indices["indices"]).encode()).hexdigest() == indices["indices_sha256"], "索引摘要不符")
    require([r["batch"] for r in idx_rows] == list(range(expected_batches)), "batch_sampler 索引记录不完整")
    require([i for r in idx_rows for i in r["indices"]] == indices["indices"], "两个索引记录器不一致")
    require([i for r in batches for i in r["sample_indices"]] == indices["indices"][:steps * batch_size], "真实训练输入与前段索引不一致")
    for row in batches:
        for key in ("motion_emb", "motion_pos", "motion_mask", "mem_order"):
            require(row["per_key"][f"['{key}']"] is None, "关闭态输入出现 motion")
    if meta["history_config"] == f"perceptual-framesamp-modul-{budget//64}frame-8x8.yaml":
        expected = hashlib.sha256(b"bool"+str((batch_size,budget)).encode()+bytes([1])*(batch_size*budget)).hexdigest()
        require(all(row["per_key"]["['static_mask']"] == expected for row in batches),"真实训练batch并非全满历史")
        if "actual_config" in meta:
            require(all(row["static_mask_counts"] == [budget]*batch_size for row in batches), "逐样本有效token数不符")
        print(f"MASK_FULL=PASS batches={steps} mask_sum={batch_size*budget}")
    return dict(meta=meta, metrics=metrics, states=states, batches=batches, indices=indices,
                idx_rows=idx_rows, root=str(root))


def with_step1(root, supplement, steps, batch_size, expected_devices="4,5", expected_workers=4, expected_fsdp=2, budget=2048):
    main = validate(root, steps, batch_size, expected_devices, expected_workers, expected_fsdp, budget)
    one = validate(supplement, 1, batch_size, expected_devices, expected_workers, expected_fsdp, budget)
    for key in ("source_head", "history_config_resolved_sha256", "norm_stats_actual", "import_origins"):
        require(main["meta"][key] == one["meta"][key], f"补跑与主 run 不同: {key}")
    require(main["states"][0]["per_leaf"] == one["states"][0]["per_leaf"], "补跑初态不同")
    require(main["batches"][0]["per_key"] == one["batches"][0]["per_key"], "补跑首个输入不同")
    require(all(main["metrics"][0][k] == one["metrics"][0][k] for k in SCALARS), "补跑首步标量不同")
    main["complete_states"] = [main["states"][0], one["states"][1], *main["states"][1:]]
    require([r["state_step"] for r in main["complete_states"]] == list(range(steps + 1)), "补跑后状态仍不完整")
    return main


def leaf_path(key):
    return "/".join(re.findall(r"\['([^']+)'\]", key))


def leaves_updated(root, names):
    states = rows(Path(root) / "param_checksums.jsonl")
    require(states[0]["phase"] == "init" and states[0]["state_step"] == 0, "缺真实初态")
    require(states[-1]["phase"] == "post_update", "缺更新后状态")
    require(len(names) == len(set(names)) == 10 and set(names) == set(MEM_LEAVES), "显式记忆叶清单必须完整十叶")
    for name in names:
        ps = [k for k in states[0]["per_leaf"] if k.startswith("params") and leaf_path(k) == name]
        ns = [k for k in states[0]["per_leaf"] if k.startswith("opt_state") and ".nu" in k and leaf_path(k).endswith(name)]
        require(len(ps) == len(ns) == 1, f"参数或Adam nu叶缺失/重复: {name}: {ps}, {ns}")
        for key in ps + ns:
            require(all(row["per_leaf_finite"].get(key) is True for row in states), f"非有限叶: {key}")
            require(states[0]["per_leaf"][key] != states[-1]["per_leaf"][key], f"窗口内未更新: {key}")
    print(f"MODUL_LEAVES_UPDATED=PASS leaves=10 nu_changed=10 params_changed=10 finite=10 first=0 last={states[-1]['state_step']}")
    return {"leaves": names, "first": 0, "last": states[-1]["state_step"]}


def negative_tests(root,steps,batch_size,expected_devices="4,5",expected_workers=4,expected_fsdp=2,budget=2048):
    """复制本次真实记录做缺项、非有限值和错配反例，不碰原记录。"""
    root=Path(root)
    validate(root,steps,batch_size,expected_devices,expected_workers,expected_fsdp,budget)
    tmp_root=ROOT/"v1-store/tmp"
    faults=("missing_state","wrong_phase","nonfinite","missing_leaf","consistent_missing_leaf","wrong_index","missing_scalar")
    with tempfile.TemporaryDirectory(prefix="m2048-record-negative-",dir=tmp_root) as temporary:
        for fault in faults:
            out=Path(temporary)/fault;out.mkdir()
            for name in ("run_meta.json","metrics.jsonl","param_checksums.jsonl","batch_digests.jsonl","index_sequence.json","idx_seq.jsonl"):
                shutil.copyfile(root/name,out/name)
            if (root/"runtime.json").is_file():
                shutil.copyfile(root/"runtime.json",out/"runtime.json")
            if fault in ("missing_state","wrong_phase","nonfinite","missing_leaf","consistent_missing_leaf"):
                path=out/"param_checksums.jsonl";records=rows(path)
                if fault=="missing_state":records.pop(1)
                if fault=="wrong_phase":records[-1]["phase"]="init"
                if fault=="nonfinite":records[-1]["per_leaf_finite"][next(iter(records[-1]["per_leaf"]))]=False
                if fault=="missing_leaf":records[-1]["per_leaf"].pop(next(iter(records[-1]["per_leaf"])))
                if fault=="consistent_missing_leaf":
                    key=next(iter(records[0]["per_leaf"]))
                    for record in records:
                        record["per_leaf"].pop(key);record["per_leaf_finite"].pop(key);record["n_leaves"]-=1
                        record["state_digest"]=hashlib.sha256("".join(f"{k}:{v}\n" for k,v in sorted(record["per_leaf"].items())).encode()).hexdigest()
                path.write_text("".join(json.dumps(r)+"\n" for r in records))
            elif fault=="wrong_index":
                path=out/"index_sequence.json";record=load(path);record["indices"][0]+=1
                path.write_text(json.dumps(record))
            else:
                path=out/"metrics.jsonl";records=rows(path);records[0].pop("mem_enc_norm")
                path.write_text("".join(json.dumps(r)+"\n" for r in records))
            try:validate(out,steps,batch_size,expected_devices,expected_workers,expected_fsdp,budget)
            except (ValueError,KeyError):pass
            else:raise ValueError(f"记录检查器未拒绝反例: {fault}")
    print(f"TRAIN_RECORD_NEGATIVE=PASS cases={len(faults)} rejected={len(faults)}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("records-a", "records-b", "step1-a", "step1-b", "records", "expect-leaves", "out"):
        p.add_argument("--" + name)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--expected-devices",default="4,5",help="两侧主run与补跑共同使用的物理GPU对，默认保持旧档4,5")
    p.add_argument("--expected-workers",type=int,default=4)
    p.add_argument("--expected-fsdp",type=int,default=2)
    p.add_argument("--budget",type=int,choices=(1024,2048,4096),default=2048)
    p.add_argument("--negative-tests",action="store_true")
    a = p.parse_args()
    shape = (a.expected_devices,a.expected_workers,a.expected_fsdp,a.budget)
    if a.records:
        require(a.expect_leaves is not None, "单侧模式必须显式 --expect-leaves")
        validate(a.records,a.steps,a.batch_size,*shape)
        names = list(MEM_LEAVES) if a.expect_leaves == "modulation10" else load(a.expect_leaves)
        result = leaves_updated(a.records, names)
        if a.negative_tests:negative_tests(a.records,a.steps,a.batch_size,*shape)
    else:
        require(all((a.records_a, a.records_b, a.step1_a, a.step1_b)), "双侧比较必须提供两份主记录与两份补跑")
        left = with_step1(a.records_a, a.step1_a, a.steps, a.batch_size,*shape)
        right = with_step1(a.records_b, a.step1_b, a.steps, a.batch_size,*shape)
        if a.negative_tests:negative_tests(a.records_a,a.steps,a.batch_size,*shape)
        for key in ("history_config_resolved_sha256", "norm_stats_actual", "batch_size", "environment"):
            if key != "environment":
                require(left["meta"][key] == right["meta"][key], f"两侧实际配置不一致: {key}")
        for la, rb in zip(left["complete_states"], right["complete_states"], strict=True):
            require(la["per_leaf"] == rb["per_leaf"], f"状态逐叶不同: state_step={la['state_step']}")
        for la, rb in zip(left["metrics"], right["metrics"], strict=True):
            require(all(la[k] == rb[k] for k in SCALARS), f"五标量不同: step={la['step']}")
        for la, rb in zip(left["batches"], right["batches"], strict=True):
            require(all(la[k] == rb[k] for k in ("per_key", "per_key_canonical", "sample_indices")), "输入逐键不同")
        require(left["indices"]["indices"][:a.steps*a.batch_size] == right["indices"]["indices"][:a.steps*a.batch_size], "实际训练索引不同")
        n = len(left["states"][0]["per_leaf"])
        result = {"steps": a.steps, "leaves": n, "mismatches": 0}
        print(f"TRAIN_RECORDS=PASS state_steps=0,2..{a.steps} step1_from_rerun=1 index_train={a.steps*a.batch_size} "
              f"idx_rows={len(left['idx_rows'])} index_n={left['indices']['n']} leaves={n} finite=1 mismatches=0")
    if a.out:
        with Path(a.out).open("x") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
