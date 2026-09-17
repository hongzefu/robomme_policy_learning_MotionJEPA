"""motion 本轮 100 步对拍和 20 步 smoke 的完整性验收，不复用历史关闭态硬编码。"""

import argparse
import hashlib
import json
import math
import pathlib
import re
import subprocess

ROOT=pathlib.Path(__file__).resolve().parents[3]
SCALARS=("loss","grad_norm","llm_grad_norm","mem_enc_norm","param_norm")
MOTION_KEYS=tuple(f"['{name}']" for name in ("motion_emb","motion_pos","motion_mask","mem_order"))
NORM_SHA="856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173"


def json_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metric_rows(path, steps):
    rows=json_rows(path)
    if [row["step"] for row in rows] != list(range(steps)):
        raise ValueError("逐步指标缺失、重复或顺序错误")
    for row in rows:
        for key in SCALARS:
            value=row[key]
            if not math.isfinite(value["dec"]) or float.fromhex(value["hex"]) != value["dec"]:
                raise ValueError(f"指标非有限或 hex 回读不符: step={row['step']} key={key}")
    return rows


def trajectory(path, head):
    env=json.loads((path/"env.json").read_text())
    if env["git_head"] != head or env["git_dirty"] is not False:
        raise ValueError("轨迹源码身份不符或起跑不干净")
    metrics=metric_rows(path/"metrics.jsonl",100)
    states=json_rows(path/"param_checksums.jsonl")
    batches=json_rows(path/"batch_digests.jsonl")
    if [r["step"] for r in states] != [0,25,50,75,99] or [r["step"] for r in batches] != [0,1,2,25,50,75,99]:
        raise ValueError("状态或输入摘要步集合不完整")
    for row in states:
        if not row["per_leaf"] or set(row["per_leaf"]) != set(row["per_leaf_finite"]) or not all(row["per_leaf_finite"].values()):
            raise ValueError("状态叶缺失或非有限")
        digest=hashlib.sha256("".join(f"{k}:{row['per_leaf'][k]}\n" for k in sorted(row["per_leaf"])).encode()).hexdigest()
        if digest != row["state_digest"]:
            raise ValueError("完整状态摘要与逐叶摘要不一致")
        if row["state_step"] != (0 if row["step"]==0 else row["step"]+1):
            raise ValueError("状态摘要不是约定的初始化或更新后状态")
    seq=json.loads((path/"index_sequence.json").read_text())
    if seq["n"] != len(seq["indices"]) or seq["n"] < 800:
        raise ValueError("训练索引序列不足 800 或自报计数不符")
    return metrics,states,batches,seq


def check_pair(args):
    if args.kind == "closed" and args.base_head == args.candidate_head:
        raise ValueError("关闭态改前、改后不能来自同一个提交")
    a=trajectory(args.base,args.base_head); b=trajectory(args.candidate,args.candidate_head)
    for x,y in zip(a[0],b[0],strict=True):
        if any(x[k]["hex"] != y[k]["hex"] for k in SCALARS): raise ValueError("逐步标量不同")
    if any(x["per_leaf"] != y["per_leaf"] for x,y in zip(a[1],b[1],strict=True)):
        raise ValueError("完整 TrainState 逐叶摘要不同")
    for x,y in zip(a[2],b[2],strict=True):
        for key in ("batch_digest","batch_digest_canonical","per_key","per_key_canonical"):
            if x[key] != y[key]: raise ValueError(f"输入 {key} 不同")
        for key in MOTION_KEYS:
            if key not in x["per_key"] or (x["per_key"][key] is None) != (args.kind=="closed"):
                raise ValueError("motion 四键状态与所验档位不符")
    if a[3]["indices"][:800] != b[3]["indices"][:800]:
        raise ValueError("前 800 个训练索引不同")
    subprocess.run(["uv","run","--no-sync","python",str(ROOT/"scripts/training/g0/compare_baseline.py"),
                    str(args.base),str(args.candidate),"--tier",args.kind],check=True,cwd=ROOT)
    print(f"INDEX_TRAIN=PASS n=800 recorded_n={min(a[3]['n'],b[3]['n'])}")
    if args.kind == "aa":
        first,last=b[1][0]["per_leaf"],b[1][-1]["per_leaf"]
        motion=[key for key in first if key.startswith("params") and "motion" in key]
        if len(motion)!=4 or any(first[key]==last[key] for key in motion):
            raise ValueError("四条 motion 参数叶并未全部更新")
        print("MOTION_PARAMS_UPDATED=PASS n=4 first_step=0 last_step=99")
    label="AA_100" if args.kind=="aa" else "GUARD_GRAD_100"
    print(f"{label}=PASS scalars_steps=100 index_n=800 batch_digest_rows=7 state_digest_rows=5")


def check_smoke(args):
    from omegaconf import OmegaConf
    metric_rows(args.records/"metrics.jsonl",20)
    runtime=json.loads((args.records/"runtime.json").read_text())
    if (runtime["device_count"],runtime["fsdp_devices"],runtime["batch_size"],runtime["num_workers"]) != (8,8,128,16) or runtime["mesh"] != {"batch":1,"fsdp":8}:
        raise ValueError("smoke 的真实设备、mesh、batch 或 worker 不符")
    log=args.log.read_text(errors="replace").replace("\r","\n")
    if not log.rstrip().endswith("EXIT_CODE=0") or "PREFLIGHT=PASS n=30" not in log:
        raise ValueError("smoke 退出码或 preflight 未通过")
    if len(re.findall(r"^CHECK_.*=PASS ",log,re.M)) != 30:
        raise ValueError("preflight 不是完整 30 项")
    snapshot=OmegaConf.load(args.run_root/"history_config.resolved.yaml")
    repo_yaml=OmegaConf.load(ROOT/"src/mme_vla_suite/models/config/robomme"/args.yaml)
    if snapshot.motion.enabled is not True or snapshot.integration_type != "modulation" or snapshot.token_per_image != 64:
        raise ValueError("smoke 未使用 modulation 8×8 开启态")
    if not (int(snapshot.motion.budget)==int(repo_yaml.motion.budget)==160):
        raise ValueError("跨源 motion budget 不一致")
    prov=json.loads((args.run_root/"motion_provenance.json").read_text())
    if prov["motion_enabled"] is not True:
        raise ValueError("训练没有留存开启态 provenance")
    tree=json.loads(args.tree_report.read_text())
    if tree["n_model"] != 65 or tree["n_ckpt"] != 65 or tree["missing"] or tree["extra"] or tree["shape_mismatch"] or tree["assert_param_tree_exact"] is not True:
        raise ValueError("checkpoint 参数树不精确匹配 65 叶")
    deployed=json.loads(args.provenance_report.read_text())
    if deployed["pass"] is not True or len(deployed["lines"]) != 5 or any("=FAIL" in line for line in deployed["lines"]):
        raise ValueError("生产口径配置和 checkpoint 加载未通过全部检查")
    metadata=(args.run_root/"19/params/_METADATA").read_text()
    expected=["q_einsum_mem', 'w'","kv_einsum_mem', 'w'","out_einsum_mem', 'w'","mem_rms_norm', 'scale'",
              "mem_rms_norm_ffn', 'Dense_0', 'kernel'","mem_rms_norm_ffn', 'Dense_0', 'bias'",
              "motion_pos_proj', 'kernel'","motion_pos_proj', 'bias'","motion_encoder_static', 'kernel'","motion_encoder_static', 'bias'"]
    if not all(key in metadata for key in expected): raise ValueError("六条 modulation 与四条 motion 叶不齐备")
    norm=args.run_root/"19/assets/robomme/norm_stats.json"
    if hashlib.sha256(norm.read_bytes()).hexdigest()!=NORM_SHA: raise ValueError("归一化统计被改变")
    print("SMOKE20=PASS steps=20 finite=1 exit_code=0\nMEM_PARAMS=PASS n=10")
    print(tree["line"])
    print("BUDGET_CONSISTENT=PASS snapshot=160 repo_yaml=160 expected=160")
    print(f"NORM_STATS=PASS sha256={NORM_SHA}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="kind",required=True)
    for name in ("closed","aa"):
        p=sub.add_parser(name)
        p.add_argument("--base",type=pathlib.Path,required=True)
        p.add_argument("--candidate",type=pathlib.Path,required=True)
        p.add_argument("--base-head",required=True)
        p.add_argument("--candidate-head",required=True)
    p=sub.add_parser("smoke")
    for name in ("records","run-root","log","tree-report","provenance-report"):
        p.add_argument("--"+name,type=pathlib.Path,required=True)
    p.add_argument("--yaml",default="perceptual-framesamp-modul-8frame-8x8-motion.yaml")
    args=parser.parse_args()
    check_smoke(args) if args.kind=="smoke" else check_pair(args)


if __name__ == "__main__":
    main()
