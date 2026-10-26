#!/usr/bin/env bash
# 新stage2入口专用公共体。Python内嵌以保持计划中的文件写集，旧驱动不引用本文件。
set -euo pipefail
: "${STAGE2_MODE:?仅允许新stage2包装入口调用}"
[ "$#" = 0 ] || { echo "本入口不接受额外参数" >&2; exit 2; }
STAGE2_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export STAGE2_REPO STAGE2_MODE
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
cd "$STAGE2_REPO"
uv run --no-sync python - <<'STAGE2_PY'
"""stage2根快照、依赖和完整checkpoint的封闭清单导出。"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time

REPO = Path(os.environ.get("STAGE2_REPO", Path.cwd())).resolve()
sys.path.insert(0, str(REPO / "scripts/dataset/hf_export"))
import ckpt_upload_common as io
import scan_hygiene as hygiene

RUN = "mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k"
ENC_RUN = "stage2-full144-1025-a"
SCALE_RUN = "stage2-scale-20261025-d"
ARCH = "wan-latent-stage2-blockwise-pool4-three-decoder"
EXPORT_NAME = "hf-ckpt-mixed8-200-vla-stage2-1024-80k"
EXPECT_MOTION = True
VLA_BUCKET = "HongzeFu/robomme-vla-mixed8-200-modul-motion-stage2-1024-80k-v1"
MJ_BUCKET = "HongzeFu/motionjepa-stage2-pool4-3decoder-1025-v1"
LAYOUT = "motion-stage2-1x1024-grid32-demopad17-v1"
DATASET = "16task-mixed8-200-3200ep-e3c7564f"
HISTORY = "perceptual-framesamp-modul-8frame-8x8-motion-stage2-mixed8.yaml"
STEPS = tuple(range(5000, 80000, 5000)) + (79999,)
ROOT_FILES = frozenset({"history_config.txt", "history_config.resolved.yaml",
    "history_config.resolved.sha256", "motion_provenance.json", "encoder_dependencies.json",
    "online_restore_inputs.json", "online_restore_inputs.npz"})
READY = "motion_snapshot.ready.json"
EVIDENCE_FILES = {key: f"motion_evidence/{key}.json"
                  for key in ("G0", "G2", "G3", "restore_receipt", "SOURCE_PIN")}
GROUPS = ("encoder", "encoder_state_sha256", "source_pin", "numerics", "stage2_compat", "vae")
SHA = re.compile(r"[0-9a-f]{64}\Z")
SCHEMA = "stage2-delivery-manifest-v1"
TOKENIZER_URI = "gs://big_vision/paligemma_tokenizer.model"
TOKENIZER_SHA = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
MAX_STEP_BYTES = 16 * 1024**3
DISK_MARGIN = 40 * 1024**3


def need(ok, message):
    if not ok:
        raise ValueError(message)


def relative(name):
    """发布路径只接受规范POSIX相对路径，禁止目录穿越和隐含别名。"""
    need(isinstance(name, str) and bool(name) and "\\" not in name
         and all(ord(c) >= 32 and ord(c) != 127 for c in name), "发布路径非法")
    path = PurePosixPath(name)
    need(not path.is_absolute() and all(p not in ("", ".", "..") for p in name.split("/"))
         and str(path) == name, f"发布路径越界: {name}")
    return name


def no_symlink(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        need(not part.is_symlink(), f"拒绝符号链接: {part}")
    return path


def member(root, name):
    path = no_symlink(Path(root) / relative(name))
    need(path.is_file() and path.resolve().is_relative_to(Path(root).resolve()), f"源文件不存在或越界: {name}")
    return path


def file_info(path):
    no_symlink(path)
    before = io.stat_identity(path)
    digest, size = io.sha256_file(path)
    need(io.stat_identity(path) == before, f"计算SHA期间源文件变化: {Path(path).name}")
    return {"sha256": digest, "bytes": size}


def json_file(path):
    return json.loads(Path(path).read_text())


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def manifest(files, **identity):
    return {"schema": SCHEMA, **identity, "files": dict(sorted(files.items()))}


def check_encoder(encoder, sha=None):
    expected = {"run_name": ENC_RUN, "checkpoint_name": "checkpoint_epoch_105.pt", "epoch": 105,
        "global_step": 7770, "arch": ARCH, "state_key": "model", "weight_type": "ema",
        "source_times": [0, 3, 5, 8], "output_shape": [1, 1024]}
    need(isinstance(encoder, dict) and all(encoder.get(k) == v for k, v in expected.items()), "encoder不是ep105/model/EMA")
    for key in ("checkpoint_sha256", "config_sha256", "fingerprint_sha256"):
        need(isinstance(encoder.get(key), str) and SHA.fullmatch(encoder[key]), f"encoder缺少{key}")
    if sha is not None:
        need(encoder["checkpoint_sha256"] == sha, "encoder SHA与yaml不一致")


def check_history(history):
    motion = history.get("motion", {})
    fixed = {"enabled": True, "source_shape": [1, 1024], "readout": "stage2_pool4_concat_v1",
        "num_tokens": 1, "dim": 1024, "budget": 96, "stride": 32, "window_frames": 33,
        "demo_min_real_frames": 17, "demo_tail_pad": "repeat_last", "pos_dim": 256,
        "window_direction": "forward", "grid_origin": "segment_start", "frame_size": 256,
        "store_path": f"v1-store/datasets/{DATASET}/motion-stage2-1024",
        "source_run": f"{ENC_RUN}/checkpoint_epoch_105.pt#model", "encoder_kind": "stage2_pool4_concat",
        "online_gpu": 4}
    need({k: v for k, v in motion.items() if k != "checkpoint_sha256"} == fixed, "yaml不是固定stage2配置")
    sha = motion.get("checkpoint_sha256", "")
    need(isinstance(sha, str) and SHA.fullmatch(sha), "yaml仍含占位SHA")
    return sha


def load_yaml(path):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def check_online_inputs(record, payload):
    """发布真实恢复输入；每个数组的实际字节都必须能从交付NPZ得到。"""
    import numpy as np
    es = record.get("exec_start_idx")
    need(type(es) is int and es >= 17 and record.get("exec_batch_lengths") == [33, 32], "在线恢复窗口合同不符")
    need(isinstance(record.get("prompt"), str) and record["prompt"].strip(), "恢复prompt缺失")
    need(isinstance(record.get("sources"), list) and record["sources"], "恢复输入来源缺失")
    for source in record["sources"]:
        need(isinstance(source.get("path"), str) and bool(source["path"])
             and isinstance(source.get("sha256"), str) and SHA.fullmatch(source["sha256"]), "恢复输入来源SHA非法")
    need(record.get("tokenizer") == {"uri": TOKENIZER_URI, "sha256": TOKENIZER_SHA}, "tokenizer取得URI或SHA不符")
    expected = {"images": ([es + 65, 256, 256, 3], "uint8"),
                "states": ([es + 65, 8], "float32"), "wrist_image": ([256, 256, 3], "uint8")}
    arrays = record.get("arrays", {})
    need(set(arrays) == set(expected), "在线恢复数组清单不符")
    with np.load(payload, allow_pickle=False) as archive:
        need(set(archive.files) == set(expected), "恢复NPZ数组键不符")
        for key, (shape, dtype) in expected.items():
            value = archive[key]
            need(list(value.shape) == shape and str(value.dtype) == dtype and value.flags.c_contiguous,
                 f"恢复数组shape/dtype/连续性不符: {key}")
            need(np.isfinite(value).all(), f"恢复数组含非有限值: {key}")
            need(arrays[key] == {"shape": shape, "dtype": dtype,
                                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()},
                 f"恢复数组描述或SHA不符: {key}")


def snapshot_sources(run, registered):
    """ready列举的根文件、引用证据和两份注册依赖共同构成可交付根。"""
    run, registered = no_symlink(run), no_symlink(registered)
    need(run.name == RUN, "源run身份不符")
    ready = json_file(member(run, READY))
    need(ready.get("schema") == "stage2-motion-snapshot-v1", "根快照schema不符")
    ready_files = ready.get("files")
    need(isinstance(ready_files, dict) and set(ready_files) == ROOT_FILES | set(EVIDENCE_FILES.values()),
         "ready文件集合必须精确等于12个已知根文件及证据文件")
    sources = {}
    for name, expected in ready_files.items():
        relative(name)
        need(name in ROOT_FILES or name.startswith("motion_evidence/"), f"ready未知来源文件: {name}")
        need(isinstance(expected, str) and SHA.fullmatch(expected), "ready文件SHA非法")
        path = member(run, name)
        need(file_info(path)["sha256"] == expected, f"根快照SHA不符: {name}")
        sources[name] = path
    online = json_file(sources["online_restore_inputs.json"])
    need(online.get("schema") == "stage2-online-restore-inputs-v1", "在线恢复素材schema不符")
    payload = online.get("payload", {})
    need(payload.get("path") == "online_restore_inputs.npz"
         and payload.get("sha256") == ready_files["online_restore_inputs.npz"], "在线恢复NPZ未闭合SHA")
    check_online_inputs(online, sources["online_restore_inputs.npz"])
    need((run / "history_config.txt").read_text().strip() == HISTORY, "history标签不符")
    history = load_yaml(sources["history_config.resolved.yaml"])
    sha = check_history(history)
    resolved_sha = file_info(sources["history_config.resolved.yaml"])["sha256"]
    need(sources["history_config.resolved.sha256"].read_text().split()[0] == resolved_sha, "resolved SHA不符")
    prov = json_file(sources["motion_provenance.json"])
    need(prov.get("motion_enabled") is EXPECT_MOTION and prov.get("resolved_sha256") == resolved_sha,
         "motion来源关闭或resolved SHA不符")
    need(all(isinstance(prov.get(k), dict) and prov[k] for k in GROUPS), "motion来源六组不完整")
    check_encoder(prov["encoder"], sha)
    evidence = prov["stage2_compat"].get("evidence", {})
    need(set(evidence) == set(EVIDENCE_FILES), "来源验证证据集合不符")
    referenced = set()
    for key, ref in evidence.items():
        name = relative(ref["path"])
        need(name == EVIDENCE_FILES[key] and ready_files.get(name) == ref["sha256"],
             f"证据未绑定固定路径及ready摘要: {key}")
        referenced.add(name)
        if key in ("G0", "G2", "G3"):
            gate = json_file(sources[name])
            if key == "G2":
                from mme_vla_suite.datastore.motion_store import validate_stage2_g2_evidence
                validate_stage2_g2_evidence(
                    gate, checkpoint_sha256=sha, samples_sha256=evidence["G0"]["sha256"],
                    source_pin=prov["source_pin"], numerics=prov["numerics"],
                    online_vae=prov["stage2_compat"]["online_vae"],
                    demo_tail=json_file(sources[EVIDENCE_FILES["G0"]]).get("counts", {}).get("demo_tail"))
            else:
                need(gate.get("gate") == key and gate.get("status") == "PASS", f"证据未通过: {key}")
            if key != "G0":
                need(gate.get("checkpoint_sha256") == sha, f"证据encoder不符: {key}")
    # 引用、固定发布路径与内容三者必须同时匹配，不能另并入一份未引用的pin。
    need(json_file(sources[EVIDENCE_FILES["SOURCE_PIN"]]) == prov["source_pin"], "SOURCE_PIN内容与来源身份不符")
    need(referenced == set(EVIDENCE_FILES.values()), "ready包含未知来源证据")
    deps = json_file(sources["encoder_dependencies.json"])
    need(deps.get("schema") == "stage2-encoder-dependencies-v1", "encoder依赖schema不符")
    for key in ("encoder", "source_pin", "vae"):
        need(deps.get(key) == prov[key], f"依赖身份与来源不符: {key}")
    dep_files = deps.get("files", [])
    need(len(dep_files) == 2 and {f.get("role") for f in dep_files} == {"encoder_checkpoint", "encoder_config"},
         "依赖必须恰为encoder checkpoint/config")
    for item in dep_files:
        filename = "checkpoint_epoch_105.pt" if item["role"] == "encoder_checkpoint" else "config.yaml"
        expected_path = f"encoder_dependencies/{ENC_RUN}/{filename}"
        need(relative(item["relative_path"]) == expected_path, "依赖发布路径不符")
        path = member(registered, filename)
        need(Path(item["source_path"]).absolute() == path, "未知依赖源路径")
        expected_sha = prov["encoder"]["checkpoint_sha256" if item["role"] == "encoder_checkpoint" else "config_sha256"]
        need(item["sha256"] == expected_sha and file_info(path) == {"sha256": item["sha256"], "bytes": item["bytes"]},
             "注册依赖SHA或字节数不符")
        need(expected_path not in sources, "依赖与根路径冲突")
        sources[expected_path] = path
    sources[READY] = member(run, READY)
    g0_sha = evidence["G0"]["sha256"]
    for gate in ("G2", "G3"):
        need(json_file(sources[evidence[gate]["path"]]).get("samples_sha256") == g0_sha,
             f"{gate}没有绑定同一G0样本")
    need(prov["stage2_compat"].get("restore_receipt_sha256") == evidence["restore_receipt"]["sha256"],
         "恢复receipt SHA未闭合")
    return sources, prov, history


def validate_store(repo, prov, history):
    """真实SMOKE也要求已verified全表及完整S2来源validator，不接受fixture开关。"""
    from mme_vla_suite.datastore.motion_store import MotionMeta, validate_stage2_provenance
    validate_stage2_provenance(prov)
    need(history == load_yaml(repo / "src/mme_vla_suite/models/config/robomme" / HISTORY),
         "run实际history与正式stage2配置不符")
    root = repo / history["motion"]["store_path"]
    need(no_symlink(root).is_dir(), "stage2库不存在")
    meta_path = member(root, "meta/store_meta.json")
    meta = json_file(meta_path)
    loaded = MotionMeta.load(root)
    need(meta.get("status") == "verified" and meta.get("layout") == LAYOUT and meta.get("num_rows") == 65419,
         "motion库不是已verified的65419行stage2库")
    need(prov.get("motion_root") == str(root) and prov.get("motion_store_meta_sha256") == file_info(meta_path)["sha256"],
         "快照没有绑定当前motion库")
    table = meta["tables"]["motion_token"]
    need(table.get("row_shape") == [1, 1024] and table.get("row_bytes") == 4096 and table.get("dtype") == "float32",
         "stage2表形状不符")
    need(file_info(member(root, table["relpath"])) == {"sha256": loaded.table_sha256, "bytes": 267956224},
         "stage2全表SHA不符")
    for key in GROUPS:
        # 快照证据路径已变为run相对路径，其余来源不可变化。
        if key != "stage2_compat":
            need(prov[key] == meta["provenance"][key], f"快照来源漂移: {key}")
    def portable_compat(value):
        value = json.loads(json.dumps(value))
        for ref in value["evidence"].values():
            ref.pop("path", None)
        return value
    need(portable_compat(prov["stage2_compat"]) == portable_compat(meta["provenance"]["stage2_compat"]),
         "根快照兼容身份与数据表不符")
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage2_pack_contract", repo / "scripts/dataset/pack_motion_store.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.check_bitexact_record(root, meta)


def step_sources(run, step):
    need(step in STEPS, "非计划step")
    root = no_symlink(Path(run) / str(step))
    need(root.is_dir(), f"step缺失: {step}")
    metadata = json_file(member(root, "_CHECKPOINT_METADATA"))
    commit = metadata.get("commit_timestamp_nsecs")
    need(type(commit) is int and commit > 0, f"Orbax尚未提交: {step}")
    need((root / "params").is_dir() and (root / "params/_METADATA").is_file(), "checkpoint缺完整params元数据")
    sources = {}
    for path in sorted(root.rglob("*")):
        no_symlink(path)
        need("tmp" not in path.name.lower(), f"checkpoint含临时文件: {path.name}")
        if path.is_file():
            sources[f"{step}/{path.relative_to(root).as_posix()}"] = path
    need(len(sources) >= 3 and sum(p.stat().st_size for p in sources.values()) <= MAX_STEP_BYTES,
         "checkpoint为空或超过单step上界")
    need(json_file(root / "_CHECKPOINT_METADATA") == metadata, "checkpoint提交身份发生变化")
    return sources


def encoder_sources(mj_root):
    """实际两源checkpoint集合精确核验，禁止目录重名和伪造final。"""
    all_sources, checkpoints = {}, []
    for run, names in ((ENC_RUN, [f"checkpoint_epoch_{e}.pt" for e in range(5, 106, 5)]),
                       (SCALE_RUN, ["checkpoint_epoch_4.pt", "checkpoint_epoch_8.pt", "checkpoint_final.pt"])):
        root = no_symlink(Path(mj_root) / "runs" / run)
        need(root.is_dir(), f"缺MJ源run: {run}")
        actual = {p.name for p in root.glob("checkpoint*.pt")}
        need(actual == set(names), f"MJ checkpoint集合不符: {run}")
        for name in [*names, "config.yaml", "train_metrics_epoch.jsonl"]:
            path = member(root, name)
            all_sources[f"{run}/{name}"] = path
            if name.endswith(".pt"):
                checkpoints.append({"path": str(path), "run_name": run, "name": name})
        if run == ENC_RUN:
            ablation = root / "motion_ablation"
            need(ablation.is_dir(), "阶段III缺motion_ablation")
            entries = [no_symlink(p) for p in sorted(ablation.rglob("*"))]
            files = [p for p in entries if p.is_file()]
            need(files, "motion_ablation为空")
            for path in files:
                rel = path.relative_to(root).as_posix()
                all_sources[f"{run}/{rel}"] = member(root, rel)
    return all_sources, checkpoints


MJ_SCHEMA_CODE = r'''
import gc, hashlib, json, pathlib, sys, torch, yaml
from motion_jepa.models import factory
from motion_jepa.models.stage2_model import load_stage2_checkpoint, fingerprint_sha256


def require(ok, detail):
    # 生产合同不能使用assert；PYTHONOPTIMIZE=1/2也必须执行全部校验。
    if not ok:
        raise ValueError(detail)


items = json.loads(sys.stdin.read())
torch.set_num_threads(1)
results = {}
for item in items:
    p = pathlib.Path(item["path"])
    c = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    require(c["arch"] == "wan-latent-stage2-blockwise-pool4-three-decoder", "checkpoint架构不符")
    require(c["weight_type"] == "ema" and c["config"]["run_name"] == item["run_name"], "EMA或run身份不符")
    require(c["fingerprint_sha256"] == fingerprint_sha256(c["fingerprint"]), "checkpoint指纹摘要不符")
    require(yaml.safe_load((p.parent / "config.yaml").read_text()) == c["config"], "checkpoint与config文件不符")
    epoch = int(item["name"].split("_")[-1][:-3]) if item["name"] != "checkpoint_final.pt" else 8
    require(c["epoch"] == epoch and type(c["global_step"]) is int and c["global_step"] > 0, "epoch或global_step不符")
    # schema重建仅用已存档affine；不下载VAE，不改变待检权重的dtype/shape。
    mean, std = c["model"]["encoder.latents_mean"], c["model"]["encoder.latents_std"]
    require(tuple(mean.shape) == tuple(std.shape) == (1, 16, 1, 1) and (std > 0).all(), "latent affine结构或std不符")
    for prefix in ("rgb_decoder.", "xyz_decoder."):
        require(torch.equal(c["model"][prefix + "latents_mean"], mean), "decoder与encoder的latent mean不符")
        require(torch.equal(c["model"][prefix + "latents_std"], std), "decoder与encoder的latent std不符")
    factory.load_wan_latent_stats = lambda vae_id: (mean.flatten(), std.flatten())
    model = factory.build_stage2_model_from_cfg(c["config"], data_metadata=c["fingerprint"]["data_metadata"])
    require(len(c["model"]) == 185 and sum(k.startswith("encoder.") for k in c["model"]) == 65, "模型状态项数不符")
    load_stage2_checkpoint(model, c, weights="main")
    for key in ("model", "model_live"):
        if key not in c:
            continue
        state = c[key]
        expected = model.state_dict()
        require(set(state) == set(expected), f"{key}状态键集合不符")
        for name, tensor in state.items():
            require(tensor.dtype == expected[name].dtype and tensor.shape == expected[name].shape,
                    f"{key}状态shape或dtype不符: {name}")
            require(torch.isfinite(tensor).all(), f"{key}状态含非有限值: {name}")
    require(all(any(k.startswith(prefix) for k in c["model"]) for prefix in
                ("encoder.", "rgb_decoder.", "xyz_decoder.", "state_decoder.")), "模型分支不完整")
    if "optimizer" in c:
        require(isinstance(c["optimizer"], dict) and {"state", "param_groups"} <= c["optimizer"].keys(),
                "optimizer状态结构不符")
    results[item["run_name"] + "/" + item["name"]] = {
        "epoch": c["epoch"], "global_step": c["global_step"], "arch": c["arch"],
        "weight_type": c["weight_type"], "model_leaves": len(c["model"]),
        "encoder_leaves": 65, "fingerprint_sha256": c["fingerprint_sha256"],
        "has_model_live": "model_live" in c, "has_optimizer": "optimizer" in c}
    del model, c
    gc.collect()
print("STAGE2_SCHEMA_JSON=" + json.dumps(results, sort_keys=True))
'''


def checkpoint_schemas(mj_root, checkpoints):
    env = dict(os.environ)
    env.update(UV_PROJECT_ENVIRONMENT=str(mj_root / ".venv"), PYTHONPATH=str(mj_root / "src"), CUDA_VISIBLE_DEVICES="",
               HF_HOME=str(REPO / "v1-store/cache/hf"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    proc = subprocess.run(["uv", "run", "--no-sync", "python", "-c", MJ_SCHEMA_CODE], cwd=mj_root,
                          env=env, input=json.dumps(checkpoints), capture_output=True, text=True, timeout=300)
    need(proc.returncode == 0, "MJ真实模型schema校验失败（未上传；请在源环境复核模型/config/fingerprint）")
    lines = [s for s in proc.stdout.splitlines() if s.startswith("STAGE2_SCHEMA_JSON=")]
    need(len(lines) == 1, "MJ schema输出缺失")
    return json.loads(lines[0].split("=", 1)[1])


def scan(files, allowlist):
    """仅扫描准确上传集合；疑似凭据不输出原文，二进制pt另扫zip中的pickle文本。"""
    entries = json_file(allowlist)["entries"]
    compiled = [(p, s, re.compile(rx), d) for p, s, rx, d in hygiene.PATTERNS]
    secret_rx = [rx for _, severity, rx, _ in compiled if severity == "secret"]
    for name, path in files.items():
        if hygiene.is_text(path):
            for hit in hygiene.scan_file(path, name, compiled):
                need(hit["severity"] != "secret", f"公开前发现疑似凭据: {name}")
                need(any(e.get("decision") == "accept" and e["pattern"] == hit["pattern"]
                         and fnmatch.fnmatch(name, e["path_glob"]) for e in entries),
                     f"公开前未裁决命中: {name} {hit['pattern']}")
        elif path.suffix == ".npz":
            import numpy as np
            with np.load(path, allow_pickle=False) as arrays:
                for key in arrays.files:
                    array = arrays[key]
                    if array.dtype.kind in ("U", "S"):
                        need(not any(rx.search(str(text)) for text in array.flat for rx in secret_rx),
                             f"NPZ文本包含疑似凭据: {name}")
        elif path.suffix == ".pt":
            import zipfile
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if info.filename.endswith(".pkl"):
                        need(info.file_size <= 32 * 1024**2, "checkpoint pickle异常大")
                        text = archive.read(info).decode("utf-8", errors="replace")
                        need(not any(rx.search(text) for rx in secret_rx), f"checkpoint元数据含疑似凭据: {name}")
    print(f"HYGIENE=PASS files={len(files)}", flush=True)


def freeze_sources(sources, stage):
    """逐文件实体复制；已存在文件只能与当前源精确相同，不覆盖不同内容。"""
    result = {}
    required = sum(p.stat().st_size for name, p in sources.items() if not (stage / name).exists())
    need(shutil.disk_usage(stage).free >= required + DISK_MARGIN, "stage空间余量不足")
    for name, source in sorted(sources.items()):
        relative(name)
        expected = file_info(source)
        target = no_symlink(stage / name)
        if target.exists():
            need(file_info(target) == expected, f"已有stage内容不符: {name}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            actual = io.copy_with_hash(source, target)
            need(actual == (expected["sha256"], expected["bytes"]), f"复制内容不符: {name}")
        need(file_info(source) == expected, f"复制期间源变化: {name}")
        result[name] = expected
    return result


def write_generated(stage, name, body):
    path = no_symlink(stage / relative(name))
    need(not path.exists() or json_file(path) == body, f"已存在生成清单发生变化: {name}")
    if not path.exists():
        io.atomic_write_json(path, body)
    return path


class AnonymousBucket(io.HFBucket):
    def _env(self):
        env = super()._env()
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN_PATH"):
            env.pop(name, None)
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        env["HF_HOME"] = str(self._cache_root / "anonymous-home")
        need(not (Path(env["HF_HOME"]) / "token").exists(), "匿名缓存目录出现token文件")
        return env


class Publisher:
    def __init__(self, root, bucket_id, identity):
        self.root, self.bucket_id, self.identity = root, bucket_id, identity
        token = io.load_hf_token(REPO / "v1-store/secrets/hf.env")
        self.bucket = io.HFBucket(bucket_id, token, cache_root=root / "cache")
        self.anon = AnonymousBucket(bucket_id, "", cache_root=root / "cache")
        self.marker_root = no_symlink(root / "logs/uploaded")
        self.marker_root.mkdir(parents=True, exist_ok=True)
        self.verify = no_symlink(root / "verify")
        self.verify.mkdir(exist_ok=True)
        self.ensure_bucket()

    def ensure_bucket(self):
        owner_path = no_symlink(self.root / "owner.json")
        owner = json_file(owner_path) if owner_path.exists() else None
        need(owner is None or owner == self.identity, "导出归属冲突")
        entry = self.bucket.entry()
        if owner is None:
            need(entry is None, "新bucket已存在且没有本任务归属记录，拒绝接管")
            io.atomic_write_json(owner_path, self.identity)
        if entry is None:
            rc, _, _ = self.bucket._run(["buckets", "create", self.bucket_id], 600)
            need(rc == 0, "公开bucket创建失败")
            entry = self.bucket.entry()
        need(entry is not None and entry.get("private") is False, "bucket必须公开")

    def readback(self, name, expected):
        # 单文件回读后即删除，磁盘峰值有界；每次必须无鉴权。
        local = no_symlink(self.verify / "readback.bin")
        need(shutil.disk_usage(self.verify).free >= expected["bytes"] + DISK_MARGIN, "匿名回读空间不足")
        if local.exists():
            local.unlink()
        result = self.anon.download(name, local)
        need(result.ok, f"匿名回读失败: {name}")
        need(file_info(local) == expected, f"匿名SHA或字节数不符: {name}")
        local.unlink()

    def transfer(self, key, stage, files):
        marker = no_symlink(self.marker_root / f"{key}.done")
        expected = {"schema": "stage2-uploaded-v1", **self.identity, "object": key,
                    "files_sha256": hashlib.sha256(canon(files)).hexdigest()}
        if marker.exists():
            need(json_file(marker) == expected, f"共享上传标记身份冲突: {key}")
        else:
            for name, info in sorted(files.items()):
                need(file_info(member(stage, name)) == info, f"stage漂移: {name}")
                need(self.bucket.upload(stage / name, name).ok, f"上传失败: {name}")
                self.readback(name, info)
            # .done仅代表已传输和匿名回读；不表示严格恢复或G12完成。
            io.atomic_write_json(marker, expected)
        return expected

    def verify_all(self, stage, files):
        remote = self.anon.list_files()
        need(remote == {n: m["bytes"] for n, m in files.items()}, "远端文件集合或字节数不符")
        for name, info in sorted(files.items()):
            self.readback(name, info)
            need(file_info(member(stage, name)) == info, f"最终stage漂移: {name}")
        need(self.anon.entry().get("private") is False, "匿名公开性复核失败")


def smoke_gate(root, mode, digest, smoke):
    """传输前要求同一模式、同一清单已通过独立本地SMOKE。"""
    path = no_symlink(root / f"smoke-{mode}.json")
    expected = {"schema": "stage2-export-smoke-v1", "mode": mode, "sha256": digest}
    if smoke:
        return path, expected
    need(path.is_file() and json_file(path) == expected, "请先对当前相同清单运行SMOKE=1")
    return path, expected


def train_finished(log):
    if not log.is_file():
        return False
    lines = log.read_text(errors="replace").splitlines()
    if any(s.startswith("RUN=") for s in lines):
        need(next(s for s in lines if s.startswith("RUN=")) == f"RUN={RUN}", "训练日志run不符")
    if lines and lines[-1].startswith("EXIT_CODE="):
        need(lines[-1] == "EXIT_CODE=0", "训练退出失败；停止上传等待裁决")
        return True
    return False


def strict_final(run, log):
    need(train_finished(log), "最终导出要求训练EXIT_CODE=0")
    need(any(s == f"RUN={RUN}" for s in log.read_text().splitlines()), "训练日志缺run身份")
    numeric = {p.name for p in run.iterdir() if p.name.isdecimal()}
    need(numeric == {str(s) for s in STEPS}, "最终16步集合不符")
    need(not any("tmp" in p.name.lower() for p in run.iterdir()), "run根仍有临时保存目录")


def run_vla(mode, smoke, root):
    run = REPO / "v1-store/train-runs/mme_vla_suite_b128_80k" / RUN
    registered = REPO / "v1-store/external/motionjepa" / ENC_RUN
    log = REPO / "v1-store/logs" / f"{RUN}.log"
    stage = root / "stage"
    allow = REPO / "scripts/dataset/hf_export/hygiene_allowlist_stage2_vla.json"
    if mode == "final":
        strict_final(run, log)
    while not (run / READY).is_file():
        need(mode == "incremental" and not smoke and not train_finished(log), "缺原子根快照ready")
        print("ROOT_WAIT=1", flush=True)
        time.sleep(30)
    sources, prov, history = snapshot_sources(run, registered)
    validate_store(REPO, prov, history)
    readme = REPO / "scripts/dataset/hf_export/stage2_vla_bucket_README.md"
    sources["README.md"] = readme
    scan(sources, allow)
    root_files = freeze_sources(sources, stage)
    root_manifest = write_generated(stage, "manifests/root.json", manifest(root_files, run_name=RUN, bucket=VLA_BUCKET))
    root_files["manifests/root.json"] = file_info(root_manifest)
    all_files = dict(root_files)
    smoke_digest = file_info(root_manifest)["sha256"]
    if mode == "final":
        # 最终SMOKE须绑定全部16步内容；不能拿训练期的根快照SMOKE代替。
        preflight = {name: file_info(path) for step in STEPS for name, path in step_sources(run, step).items()}
        smoke_digest = hashlib.sha256(canon({"root": smoke_digest, "steps": preflight,
                                            "log": file_info(log)})).hexdigest()
    if mode == "incremental":
        # ready在训练启动后才出现；根源校验、扫描和实体stage全部成功后才签发本地预检。
        smoke_path, smoke_record = smoke_gate(root, mode, smoke_digest, True)
        io.atomic_write_json(smoke_path, smoke_record)
        print(f"ROOT_LOCAL_SMOKE=PASS files={len(root_files)} sha256={smoke_digest}", flush=True)
    else:
        smoke_path, smoke_record = smoke_gate(root, mode, smoke_digest, smoke)
    publisher = None if smoke else Publisher(root, VLA_BUCKET,
        {"kind": "stage2-vla", "run_name": RUN, "bucket": VLA_BUCKET,
         "root_manifest_sha256": file_info(root_manifest)["sha256"]})
    if publisher:
        publisher.transfer("root", stage, root_files)
        # 每次增量/最终启动都重新匿名核根快照，不能信任旧done代替当前可读性。
        for name, info in root_files.items():
            publisher.readback(name, info)
        print(f"ROOT_UPLOAD=PASS files={len(root_files)}", flush=True)
    processed = set()
    while True:
        for step in STEPS:
            if step in processed:
                continue
            if not (run / str(step)).exists():
                continue
            commit_path = run / str(step) / "_CHECKPOINT_METADATA"
            if mode == "incremental" and (not commit_path.is_file() or not json_file(commit_path).get("commit_timestamp_nsecs")):
                continue
            step_src = step_sources(run, step)
            scan(step_src, allow)
            files = freeze_sources(step_src, stage)
            if mode == "final":
                need(all(preflight[name] == info for name, info in files.items()), "最终预检后step内容发生变化")
            step_manifest = write_generated(stage, f"manifests/step-{step}.json",
                manifest({**root_files, **files}, run_name=RUN, bucket=VLA_BUCKET, step=step))
            files[f"manifests/step-{step}.json"] = file_info(step_manifest)
            if publisher:
                publisher.transfer(str(step), stage, files)
                print(f"INC_UPLOADED step={step}", flush=True)
            processed.add(step)
            all_files.update(files)
            sources.update(step_src)
        if mode == "final":
            need(processed == set(STEPS), "最终导出缺step")
            # 日志仅在训练结束后固定，不能污染训练期间不可变根快照身份。
            extra = {"_run-meta/train.log": log}
            scan(extra, allow)
            files = freeze_sources(extra, stage)
            all_files.update(files)
            sources.update(extra)
            final_manifest = write_generated(stage, "delivery_manifest.json",
                manifest(all_files, run_name=RUN, bucket=VLA_BUCKET, steps=list(STEPS)))
            all_files["delivery_manifest.json"] = file_info(final_manifest)
            if publisher:
                publisher.transfer("final", stage, {**files, "delivery_manifest.json": all_files["delivery_manifest.json"]})
                publisher.verify_all(stage, all_files)
            break
        if smoke or train_finished(log):
            break
        print(f"INC_WAIT done={len(processed)} expected=16", flush=True)
        time.sleep(30)
    for name, source in sources.items():
        need(file_info(source) == all_files[name], f"源内容改变: {name}")
    if smoke:
        io.atomic_write_json(smoke_path, smoke_record)
        print(f"SMOKE=PASS kind=vla mode={mode} steps={len(processed)} root_files={len(root_files)}")
    elif mode == "final":
        print(f"CKPT_UPLOAD=PASS steps=16 root_files={len(root_files)} sha256_mismatch=0 public=1")
    else:
        need(processed == set(STEPS), "训练结束但增量未齐16步；最终驱动补传并验收")
        print(f"INC_END run={RUN} done={len(processed)}")


def run_encoder(smoke, root):
    mj_root = Path("/scratch/hongze/MotionJEPA")
    sources, checkpoints = encoder_sources(mj_root)
    baseline = {name: file_info(path) for name, path in sources.items()}
    schemas = checkpoint_schemas(mj_root, checkpoints)
    need(all(file_info(sources[name]) == info for name, info in baseline.items()), "模型schema检查期间MJ源文件改变")
    stage = root / "stage"
    allow = REPO / "scripts/dataset/hf_export/hygiene_allowlist_stage2_pool4.json"
    sources["README.md"] = REPO / "scripts/dataset/hf_export/stage2_pool4_bucket_README.md"
    scan(sources, allow)
    files = freeze_sources(sources, stage)
    need(all(files[name] == info for name, info in baseline.items()), "MJ复制前源文件改变")
    for run in (ENC_RUN, SCALE_RUN):
        path = write_generated(stage, f"manifests/{run}.json", manifest(
            {k: v for k, v in files.items() if k.startswith(run + "/")}, source_run=run,
            checkpoints={k: v for k, v in schemas.items() if k.startswith(run + "/")}))
        files[f"manifests/{run}.json"] = file_info(path)
    path = write_generated(stage, "delivery_manifest.json", manifest(files, bucket=MJ_BUCKET, source_runs=[ENC_RUN, SCALE_RUN]))
    files["delivery_manifest.json"] = file_info(path)
    scan({name: stage / name for name in files}, allow)
    smoke_path, smoke_record = smoke_gate(root, "encoder", file_info(path)["sha256"], smoke)
    if not smoke:
        publisher = Publisher(root, MJ_BUCKET, {"kind": "stage2-pool4", "bucket": MJ_BUCKET,
            "manifest_sha256": file_info(path)["sha256"]})
        publisher.transfer("encoder", stage, files)
        publisher.verify_all(stage, files)
    for name, source in sources.items():
        need(file_info(source) == files[name], f"MJ源文件发生变化: {name}")
    if smoke:
        io.atomic_write_json(smoke_path, smoke_record)
        print(f"SMOKE=PASS kind=encoder source_runs=2 checkpoints={len(checkpoints)} files={len(files)}")
    else:
        print(f"MJ_CKPT_UPLOAD=PASS source_runs=2 files={len(files)} bytes={sum(v['bytes'] for v in files.values())} sha256_mismatch=0 public=1")


def main():
    mode = os.environ["STAGE2_MODE"]
    need(mode in ("final", "incremental", "encoder"), "未知导出模式")
    smoke = os.environ.get("SMOKE", "0")
    need(smoke in ("0", "1"), "SMOKE只允许0或1")
    name = "hf-ckpt-stage2-pool4-3decoder-1025" if mode == "encoder" else EXPORT_NAME
    root = no_symlink(REPO / "v1-store/exports" / name)
    need(REPO == Path("/scratch/hongze/robomme_policy_learning_MotionJEPA"), "实际导出仅允许主检出；worktree仅运行CPU合同测试")
    root.mkdir(parents=True, exist_ok=True)
    no_symlink(root / "stage").mkdir(exist_ok=True)
    with io.SingleInstanceLock(no_symlink(root / "upload.lock")):
        if mode == "encoder":
            run_encoder(smoke == "1", root)
        else:
            run_vla(mode, smoke == "1", root)
    print("EXIT_CODE=0", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # 不输出外部命令stderr、checkpoint值或凭据，只报本地已脱敏异常类别与消息。
        print(f"STAGE2_EXPORT=FAIL error={type(exc).__name__} detail={exc}", flush=True)
        print("EXIT_CODE=1", flush=True)
        raise SystemExit(1)
STAGE2_PY
