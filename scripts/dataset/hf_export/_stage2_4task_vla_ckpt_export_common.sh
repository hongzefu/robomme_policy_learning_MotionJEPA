#!/usr/bin/env bash
# 4task（4task-v2-1600ep-604f16da）stage2 rgb-state concat 1×1024 两条50k的公开导出公共体（1114计划 S5）。
# 只由 run_4task_stage2_ckpt_inc.sh 与 run_4task_{modul,ctx}8x8_concat_50k_ckpt_export.sh 包装入口调用。
# O16_MODE：
#   incremental  每份ckpt落盘即冻结并上传、匿名回读，持 hf-ckpt-<run>/upload.lock，只出 INC_END（不算最终验收）
#   stage        训练成功后同步调用：strict_final、冻结源、写 stage/50000（含encoder依赖实体）、
#                manifests/step-50000.json 与 delivery_manifest.json；不联网、不读凭据、不建 upload.lock
#   upload       复用 stage 做上传与匿名读回（持 hf-ckpt-<run>/upload.lock），出 HF_VERIFY=PASS 与 CKPT_UPLOAD=PASS steps=10；
#                stage 缺失或与 delivery_manifest.json 不符即拒，不重建
#   eval         692评估结果传同一bucket的 eval/ 前缀；独立导出根 hf-eval-<run>/、独立 upload.lock、独立清单与读回，
#                出 EVAL_HF_VERIFY=PASS
#   stage-delete 三条同时满足才删 hf-ckpt-<run>/stage：HF_VERIFY=PASS、CKPT_DELIVERY_READY=PASS step=50000、EVAL692 已出1行
set -euo pipefail
: "${O16_MODE:?仅允许4task包装入口调用}"
[ "$#" = 0 ] || { echo "本入口不接受额外参数" >&2; exit 2; }
O16_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export O16_REPO O16_MODE RUN BUCKET
[ -z "${ATTEMPT_MANIFEST:-}" ] || export ATTEMPT_MANIFEST
[ -z "${EVAL_RUN_NAME:-}" ] || export EVAL_RUN_NAME
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
cd "$O16_REPO"
uv run --no-sync python - <<'O16_PY'
"""4task两条50k checkpoint与评估结果的公开导出；复用现有HF CLI封装与逐对象匿名回读。"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time

REPO = Path(os.environ.get("O16_REPO", Path.cwd())).resolve()
EXPORT = REPO / "scripts/dataset/hf_export"
sys.path.insert(0, str(EXPORT))
import ckpt_upload_common as io
import scan_hygiene as hygiene

DATASET = "4task-v2-1600ep-604f16da"
MOTION_SUB = "motion-stage2-1024-rgbstate-ep144"
# 运行名 → (bucket, 融合标签, 融合方式, memory_token_dim)；计划§4两条，attempt02经ATTEMPT_MANIFEST登记。
RUNS = {
    "v2-1600ep-m8x8-modul-motion-s2rgbstate-concat1x1024-b128-50k": (
        "HongzeFu/robomme-vla-4task-m8x8-modul-s2rgbstate-concat1x1024-50k-v1", "modul", "modulation", 1024),
    "v2-1600ep-m8x8-ctx-motion-s2rgbstate-concat1x1024-b128-50k": (
        "HongzeFu/robomme-vla-4task-m8x8-ctx-s2rgbstate-concat1x1024-50k-v1", "ctx", "context", 2048),
}
SHORT = {"modul": "m", "ctx": "c"}
RUN = os.environ.get("RUN", "")
ATTEMPT = None
TRAIN_ROOT = "v1-store/train-runs/mme_vla_suite_4task_b128_50k"
LOG_DIR = "v1-store/logs/v16-4trs-vla"
ENC_RUN = "stage2-4task-rgbstate-144ep-1113-a"
ENC_CKPT = "checkpoint_epoch_144.pt"
ARCH = "wan-latent-stage2-blockwise-pool4-rgb-state"
KIND = "local-built-latents"
LAYOUT = "motion-stage2-1x1024-grid32-demopad17-v1"
# 训练步数名为 NUM_TRAIN_STEPS=50001；交付目录名 CKPT_STEPS 固定 5000…50000 共10份（T06、A17）。
CKPT_STEPS = tuple(range(5000, 50001, 5000))
DELIVERY_STEP = 50000
ROOT_FILES = frozenset({"history_config.txt", "history_config.resolved.yaml",
    "history_config.resolved.sha256", "motion_provenance.json", "encoder_dependencies.json",
    "online_restore_inputs.json", "online_restore_inputs.npz"})
READY = "motion_snapshot.ready.json"
GROUPS = ("encoder", "encoder_state_sha256", "source_pin", "numerics", "stage2_compat", "vae")
SHA = re.compile(r"[0-9a-f]{64}\Z")
SCHEMA = "stage2-delivery-manifest-v1"
EVAL_SCHEMA = "stage2-eval-manifest-v1"
EVAL_PREFIX = "eval/"
EVAL_SKIP_DIRS = ("gpu-slots",)
TOKENIZER_URI = "gs://big_vision/paligemma_tokenizer.model"
TOKENIZER_SHA = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
MAX_STEP_BYTES = 16 * 1024**3
DISK_MARGIN = 40 * 1024**3
ALLOWLIST = "hygiene_allowlist_4task_stage2_vla.json"


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
    expected = {"run_name": ENC_RUN, "checkpoint_name": ENC_CKPT, "epoch": 144, "arch": ARCH,
                "state_key": "model", "weight_type": "ema", "output_shape": [1, 1024]}
    need(isinstance(encoder, dict) and all(encoder.get(k) == v for k, v in expected.items()),
         "encoder不是ep144 rgb-state/model/EMA")
    need(type(encoder.get("global_step")) is int and encoder["global_step"] > 0, "encoder缺少global_step")
    for key in ("checkpoint_sha256", "config_sha256", "fingerprint_sha256"):
        need(isinstance(encoder.get(key), str) and SHA.fullmatch(encoder[key]), f"encoder缺少{key}")
    if sha is not None:
        need(encoder["checkpoint_sha256"] == sha, "encoder SHA与yaml不一致")


def load_yaml(path):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def evidence_roles(provenance):
    """证据角色以S2的motion_store为唯一来源（local-built-latents的角色集合由S2定义）。"""
    from mme_vla_suite.datastore import motion_store as ms
    return tuple(ms.stage2_evidence_roles(provenance))


def check_online_inputs(record, payload):
    """发布真实恢复输入；每个数组的实际字节都必须能从交付NPZ得到。"""
    import numpy as np
    es = record.get("exec_start_idx")
    need(type(es) is int and es >= 17 and record.get("exec_batch_lengths") in ([1, 16, 16, 16, 16], [33, 32]),
         "在线恢复窗口合同不符")
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


def base_run():
    """attempt按其base_run校验历史与读法；未登记attempt时即RUN本身。"""
    return ATTEMPT["base_run"] if ATTEMPT else RUN


def bucket_id():
    return ATTEMPT["bucket"] if ATTEMPT else RUNS[RUN][0]


def role(run=None):
    return RUNS[run or base_run()][1]


def history_name(run):
    need(run in RUNS, "RUN不在两条已批准训练中")
    return f"perceptual-framesamp-{RUNS[run][1]}-8frame-8x8-motion-stage2-1x1024-4task-rgbstate.yaml"


def readme_path(run):
    return EXPORT / f"4task_{RUNS[run][1]}8x8_concat_50k_bucket_README.md"


def eval_run_name():
    """评估目录名：默认 seed 7；EVAL_RUN_NAME 只许同融合的 local692-s2rgbstate-<融合>-concat1024-50k-s<seed>。"""
    name = os.environ.get("EVAL_RUN_NAME") or f"local692-s2rgbstate-{role()}-concat1024-50k-s7"
    need(re.fullmatch(rf"local692-s2rgbstate-{role()}-concat1024-50k-s\d+", name) is not None,
         f"EVAL_RUN_NAME与本run融合方式不符: {name}")
    return name


def check_attempt(path):
    """attempt清单须与RUN/BUCKET、base及决定记录同时一致；任一不符即失败，不回退固定名。"""
    record = json_file(path)
    need(isinstance(record, dict) and set(record) == {"base_run", "run", "history", "train_head", "decision", "bucket"},
         "attempt清单字段不符")
    base, run = record["base_run"], record["run"]
    need(base in RUNS, "attempt base_run不是两条已批准训练之一")
    match = re.fullmatch(re.escape(base) + r"-attempt(\d{2})", str(run))
    need(match is not None and int(match.group(1)) >= 2, "attempt run必须为base_run加-attemptNN（NN≥02）")
    need(run == RUN and record["bucket"] == os.environ.get("BUCKET"), "RUN/BUCKET与attempt清单不符")
    need(record["bucket"] == RUNS[base][0] + f"-attempt{match.group(1)}", "attempt bucket必须为base bucket加相同后缀")
    need(record["history"] == history_name(base), "attempt history与base融合不符")
    decision = Path(record["decision"])
    need(decision.is_file() and not decision.is_symlink(), "attempt决定记录不存在")
    allowed = json_file(decision).get("allowed_runs")
    need(isinstance(allowed, list) and run in allowed, "决定记录未登记该attempt")
    return record


def check_history(history):
    base = base_run()
    need(history == load_yaml(REPO / "src/mme_vla_suite/models/config/robomme" / history_name(base)),
         "run快照与对应正式yaml不同")
    motion = history.get("motion", {})
    need(motion.get("store_path") == f"v1-store/datasets/{DATASET}/{MOTION_SUB}", "motion库与固定库名不符")
    need(motion.get("readout") == "stage2_pool4_concat_v1" and motion.get("num_tokens") == 1
         and motion.get("dim") == 1024 and motion.get("budget") == 96, "concat训练必须为1×1024、96窗读法")
    sha = motion.get("checkpoint_sha256", "")
    need(isinstance(sha, str) and SHA.fullmatch(sha), "yaml含占位SHA")
    return sha


def snapshot_sources(run, registered):
    """ready列举的根文件、引用证据和两份注册依赖共同构成可交付根。"""
    run, registered = no_symlink(run), no_symlink(registered)
    need(run.name == RUN, "源run身份不符")
    ready = json_file(member(run, READY))
    need(ready.get("schema") == "stage2-motion-snapshot-v1", "根快照schema不符")
    ready_files = ready.get("files")
    need(isinstance(ready_files, dict), "ready文件清单缺失")
    evidence_names = {name for name in ready_files if name.startswith("motion_evidence/")}
    need(evidence_names and set(ready_files) == ROOT_FILES | evidence_names,
         "ready文件集合必须精确等于7个已知根文件及证据文件")
    sources = {}
    for name, expected in ready_files.items():
        relative(name)
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
    need((run / "history_config.txt").read_text().strip() == history_name(base_run()), "history标签不符")
    history = load_yaml(sources["history_config.resolved.yaml"])
    sha = check_history(history)
    resolved_sha = file_info(sources["history_config.resolved.yaml"])["sha256"]
    need(sources["history_config.resolved.sha256"].read_text().split()[0] == resolved_sha, "resolved SHA不符")
    prov = json_file(sources["motion_provenance.json"])
    need(prov.get("motion_enabled") is True and prov.get("resolved_sha256") == resolved_sha,
         "motion来源关闭或resolved SHA不符")
    need(all(isinstance(prov.get(k), dict) and prov[k] for k in GROUPS), "motion来源六组不完整")
    check_encoder(prov["encoder"], sha)
    need(prov["stage2_compat"].get("kind") == KIND, "来源kind不是local-built-latents")
    roles = evidence_roles(prov)
    expected_paths = {r: f"motion_evidence/{r}.json" for r in roles}
    evidence = prov["stage2_compat"].get("evidence", {})
    need(set(evidence) == set(roles) and evidence_names == set(expected_paths.values()), "来源验证证据集合不符")
    for key, ref in evidence.items():
        name = relative(ref["path"])
        need(name == expected_paths[key] and ready_files.get(name) == ref["sha256"],
             f"证据未绑定固定路径及ready摘要: {key}")
        if key in ("G0", "G2", "G3"):
            gate = json_file(sources[name])
            need(gate.get("gate") == key and gate.get("status") == "PASS", f"证据未通过: {key}")
            if key == "G3":
                need(gate.get("checkpoint_sha256") == sha, f"证据encoder不符: {key}")
    need(json_file(sources[expected_paths["SOURCE_PIN"]]) == prov["source_pin"], "SOURCE_PIN内容与来源身份不符")
    deps = json_file(sources["encoder_dependencies.json"])
    need(deps.get("schema") == "stage2-encoder-dependencies-v1", "encoder依赖schema不符")
    for key in ("encoder", "source_pin", "vae"):
        need(deps.get(key) == prov[key], f"依赖身份与来源不符: {key}")
    dep_files = deps.get("files", [])
    need(len(dep_files) == 2 and {f.get("role") for f in dep_files} == {"encoder_checkpoint", "encoder_config"},
         "依赖必须恰为encoder checkpoint/config")
    for item in dep_files:
        filename = ENC_CKPT if item["role"] == "encoder_checkpoint" else "config.yaml"
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
    print(f"ENCODER_CHECK=PASS run={ENC_RUN} epoch=144 arch={ARCH} sha={sha[:8]}", flush=True)
    return sources, prov, history


def validate_store(repo, prov, history):
    """消费S2已验证的来源合同和实际表身份；不另维护行数、norm或manifest固定常量。"""
    from mme_vla_suite.datastore.motion_store import MotionMeta, validate_stage2_provenance
    validate_stage2_provenance(prov)
    need(prov["stage2_compat"].get("kind") == KIND, "来源kind不是local-built-latents")
    root = repo / history["motion"]["store_path"]
    loaded = MotionMeta.load(root)
    meta_path = member(root, "meta/store_meta.json")
    meta = json_file(meta_path)
    need(meta.get("status") == "verified" and meta.get("layout") == LAYOUT, "motion表未验证或layout错误")
    need(prov.get("motion_root") == str(root) and prov.get("motion_store_meta_sha256") == file_info(meta_path)["sha256"],
         "根快照未绑定实际motion表")
    table = meta["tables"]["motion_token"]
    need(table.get("row_shape") == [1, 1024] and table.get("row_bytes") == 4096 and table.get("dtype") == "float32",
         "motion表形状不符")
    need(file_info(member(root, table["relpath"])) == {"sha256": loaded.table_sha256, "bytes": meta["num_rows"] * 4096},
         "motion全表SHA或字节数不符")
    for key in GROUPS:
        if key != "stage2_compat":
            need(prov[key] == meta["provenance"][key], f"快照来源漂移: {key}")

    def portable(value):
        value = json.loads(json.dumps(value))
        for ref in value["evidence"].values():
            ref.pop("path", None)
        return value
    need(portable(prov["stage2_compat"]) == portable(meta["provenance"]["stage2_compat"]), "快照与表的证据身份不同")


def step_sources(run, step):
    need(step in CKPT_STEPS, "非计划step")
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


def scan(files, allowlist):
    """仅扫描准确上传集合；疑似凭据不输出原文，二进制pt另扫zip中的pickle文本。"""
    entries = json_file(allowlist)["entries"]
    compiled = [(p, s, re.compile(rx), d) for p, s, rx, d in hygiene.PATTERNS]
    secret_rx = [rx for _, severity, rx, _ in compiled if severity == "secret"]
    for name, path in files.items():
        if hygiene.is_text(path):
            for hit in hygiene.scan_file(path, name, compiled):
                need(hit["severity"] != "secret", f"公开前发现疑似凭据: {name}")
                need(any(e.get("decision") == "accept" and e.get("pattern") == hit["pattern"]
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
    stage.mkdir(parents=True, exist_ok=True)
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


def make_buckets(bucket, root):
    """(带鉴权写端, 匿名读端)；离线测试替换本函数为本地假bucket。"""
    token = io.load_hf_token(REPO / "v1-store/secrets/hf.env")
    return (io.HFBucket(bucket, token, cache_root=root / "cache"),
            AnonymousBucket(bucket, "", cache_root=root / "cache"))


class Publisher:
    """create=True 只给ckpt范围（bucket归属记录在hf-ckpt-<run>/owner.json）；eval范围只接已归属的公开bucket。"""

    def __init__(self, root, bucket, identity, *, create=True):
        self.root, self.bucket_id, self.identity = root, bucket, identity
        self.bucket, self.anon = make_buckets(bucket, root)
        self.marker_root = no_symlink(root / "logs/uploaded")
        self.marker_root.mkdir(parents=True, exist_ok=True)
        self.verify = no_symlink(root / "verify")
        self.verify.mkdir(exist_ok=True)
        self.ensure_bucket(create)

    def ensure_bucket(self, create=True):
        owner_path = no_symlink(self.root / "owner.json")
        owner = json_file(owner_path) if owner_path.exists() else None
        need(owner is None or owner == self.identity, "导出归属冲突")
        entry = self.bucket.entry()
        need(entry is not None or create, "eval上传要求ckpt增量上传已建好该bucket")
        if owner is None:
            need(entry is None or not create, "新bucket已存在且没有本任务归属记录，拒绝接管")
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
            # .done仅代表已传输和匿名回读；不表示严格恢复完成。
            io.atomic_write_json(marker, expected)
        return expected

    def verify_scope(self, stage, files, *, scope):
        """ckpt范围比对 eval/ 以外的全部远端对象；eval范围只比对 eval/ 前缀。两者分清单、分读回。"""
        remote = self.anon.list_files()
        if scope == "ckpt":
            remote = {n: b for n, b in remote.items() if not n.startswith(EVAL_PREFIX)}
            need(not any(n.startswith(EVAL_PREFIX) for n in files), "ckpt清单不得含eval/对象")
        else:
            remote = {n: b for n, b in remote.items() if n.startswith(EVAL_PREFIX)}
            need(all(n.startswith(EVAL_PREFIX) for n in files), "eval清单只能含eval/对象")
        need(remote == {n: m["bytes"] for n, m in files.items()}, f"{scope}远端文件集合或字节数不符")
        for name, info in sorted(files.items()):
            self.readback(name, info)
            need(file_info(member(stage, name)) == info, f"最终stage漂移: {name}")
        need(self.anon.entry().get("private") is False, "匿名公开性复核失败")


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
    need(numeric == {str(s) for s in CKPT_STEPS}, "最终10步集合不符（CKPT_STEPS=5000…50000）")
    need(not any("tmp" in p.name.lower() for p in run.iterdir()), "run根仍有临时保存目录")


def add_manifest(stage, files, name, **identity):
    path = write_generated(stage, name, manifest(files, **identity))
    return {**files, name: file_info(path)}


def paths():
    run = REPO / TRAIN_ROOT / RUN
    registered = REPO / "v1-store/external/motionjepa" / ENC_RUN
    log = REPO / LOG_DIR / f"{RUN}.log"
    return run, registered, log


def identity_for(root_manifest_sha):
    return {"kind": "4task-stage2-vla", "run_name": RUN, "bucket": bucket_id(),
            "root_manifest_sha256": root_manifest_sha}


def stage_root_files(stage, run, registered):
    """根快照、encoder依赖与README冻结进stage并写 manifests/root.json；返回(root_files, sources)。"""
    sources, prov, history = snapshot_sources(run, registered)
    validate_store(REPO, prov, history)
    sources["README.md"] = readme_path(base_run())
    scan(sources, EXPORT / ALLOWLIST)
    files = freeze_sources(sources, stage)
    root_files = add_manifest(stage, files, "manifests/root.json", run_name=RUN, bucket=bucket_id())
    return root_files, sources


def stage_step(stage, run, step, root_files):
    step_src = step_sources(run, step)
    scan(step_src, EXPORT / ALLOWLIST)
    files = freeze_sources(step_src, stage)
    # 每份step清单包含根快照与encoder依赖，异地单步恢复不依赖其他step。
    path = write_generated(stage, f"manifests/step-{step}.json",
                           manifest({**root_files, **files}, run_name=RUN, bucket=bucket_id(), step=step))
    files[f"manifests/step-{step}.json"] = file_info(path)
    return files, step_src


def run_incremental(smoke, root):
    run, registered, log = paths()
    stage = root / "stage"
    print(f"CKPT_STEPS_PLAN={','.join(map(str, CKPT_STEPS))} count={len(CKPT_STEPS)} run={RUN}", flush=True)
    while not (run / READY).is_file():
        need(not smoke and not train_finished(log), "缺原子根快照ready")
        print("ROOT_WAIT=1", flush=True)
        time.sleep(30)
    root_files, sources = stage_root_files(stage, run, registered)
    publisher = None
    if not smoke:
        publisher = Publisher(root, bucket_id(), identity_for(root_files["manifests/root.json"]["sha256"]))
        publisher.transfer("root", stage, root_files)
        print(f"ROOT_UPLOAD=PASS files={len(root_files)}", flush=True)
    all_files, processed = dict(root_files), set()
    while True:
        for step in CKPT_STEPS:
            if step in processed:
                continue
            metadata = run / str(step) / "_CHECKPOINT_METADATA"
            if not metadata.is_file() or not json_file(metadata).get("commit_timestamp_nsecs"):
                continue
            files, step_src = stage_step(stage, run, step, root_files)
            if publisher:
                publisher.transfer(str(step), stage, files)
                print(f"INC_UPLOADED step={step}", flush=True)
            all_files.update(files)
            sources.update(step_src)
            processed.add(step)
        if smoke or processed == set(CKPT_STEPS) or train_finished(log):
            break
        print(f"INC_WAIT done={len(processed)} expected={len(CKPT_STEPS)}", flush=True)
        time.sleep(30)
    for name, path in sources.items():
        need(file_info(path) == all_files[name], f"源文件变化: {name}")
    if smoke:
        print(f"SMOKE=PASS kind=checkpoint mode=incremental steps={len(processed)} network=0", flush=True)
        return
    need(processed == set(CKPT_STEPS), f"训练结束但增量未齐{len(CKPT_STEPS)}步")
    # 增量只出 INC_END，不冒充最终验收；HF_VERIFY 只由 upload 模式给出。
    print(f"INC_END run={RUN} done={len(CKPT_STEPS)}", flush=True)


def run_stage(root):
    """训练成功后同步运行：不联网、不读凭据、不持 upload.lock。"""
    run, registered, log = paths()
    stage = root / "stage"
    strict_final(run, log)
    need((run / READY).is_file(), "缺原子根快照ready")
    root_files, sources = stage_root_files(stage, run, registered)
    all_files = dict(root_files)
    for step in CKPT_STEPS:
        files, step_src = stage_step(stage, run, step, root_files)
        all_files.update(files)
        sources.update(step_src)
    extra = {"_run-meta/train.log": log}
    scan(extra, EXPORT / ALLOWLIST)
    all_files.update(freeze_sources(extra, stage))
    sources.update(extra)
    for name, path in sources.items():
        need(file_info(path) == all_files[name], f"源文件变化: {name}")
    path = write_generated(stage, "delivery_manifest.json",
                           manifest(all_files, run_name=RUN, bucket=bucket_id(), steps=list(CKPT_STEPS)))
    delivery = file_info(path)
    step_manifest = file_info(stage / f"manifests/step-{DELIVERY_STEP}.json")
    print(f"STAGE_DIR={stage}", flush=True)
    print(f"STAGE=PASS run={RUN} steps={len(CKPT_STEPS)} objects={len(all_files) + 1} "
          f"bytes={sum(v['bytes'] for v in all_files.values()) + delivery['bytes']} "
          f"step_manifest=manifests/step-{DELIVERY_STEP}.json:{step_manifest['sha256'][:8]} "
          f"delivery_manifest={delivery['sha256'][:8]} network=0", flush=True)


def load_stage(stage):
    """upload 只复用已生成的 stage：缺失或与 delivery_manifest.json 任一处不符即拒，不重建。"""
    need(stage.is_dir(), f"stage不存在，须先跑 O16_MODE=stage: {stage}")
    delivery_path = stage / "delivery_manifest.json"
    need(delivery_path.is_file(), "stage缺delivery_manifest.json，须先跑 O16_MODE=stage")
    delivery = json_file(member(stage, "delivery_manifest.json"))
    need(delivery.get("schema") == SCHEMA and delivery.get("run_name") == RUN and delivery.get("bucket") == bucket_id()
         and delivery.get("steps") == list(CKPT_STEPS), "delivery_manifest身份或步集合不符")
    files = delivery.get("files")
    need(isinstance(files, dict) and files, "delivery_manifest文件清单为空")
    actual = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
    need(actual - {"delivery_manifest.json"} == set(files), "stage实际文件集合与delivery_manifest不符")
    for name, info in sorted(files.items()):
        need(file_info(member(stage, name)) == info, f"stage与delivery_manifest不符: {name}")
    root_body = json_file(stage / "manifests/root.json")
    need(root_body.get("schema") == SCHEMA and root_body.get("run_name") == RUN and root_body.get("bucket") == bucket_id(),
         "root清单身份不符")
    root_files = {**root_body["files"], "manifests/root.json": files["manifests/root.json"]}
    groups = {"root": root_files}
    for step in CKPT_STEPS:
        name = f"manifests/step-{step}.json"
        need(name in files, f"stage缺 {name}")
        body = json_file(stage / name)
        need(body.get("step") == step and body.get("run_name") == RUN and body.get("bucket") == bucket_id(),
             f"{name}身份不符")
        need(all(body["files"].get(k) == v for k, v in root_files.items()), f"{name}未包含同一根快照")
        own = {k: v for k, v in body["files"].items() if k not in root_files}
        need(own and all(k.startswith(f"{step}/") for k in own), f"{name}含非本步对象")
        groups[str(step)] = {**own, name: files[name]}
    final = {"_run-meta/train.log": files["_run-meta/train.log"], "delivery_manifest.json": file_info(delivery_path)}
    groups["final"] = final
    union = {}
    for group in groups.values():
        union.update(group)
    need(set(union) == set(files) | {"delivery_manifest.json"}, "各上传组的并集与delivery_manifest不符")
    return groups, union


def run_upload(root):
    stage = root / "stage"
    groups, all_files = load_stage(stage)
    print(f"STAGE_REUSE=PASS objects={len(all_files)}", flush=True)
    publisher = Publisher(root, bucket_id(), identity_for(all_files["manifests/root.json"]["sha256"]))
    for key, files in groups.items():
        publisher.transfer(key, stage, files)
    publisher.verify_scope(stage, all_files, scope="ckpt")
    print(f"HF_VERIFY=PASS bucket={bucket_id()} scope=ckpt objects={len(all_files)} "
          f"bytes={sum(v['bytes'] for v in all_files.values())} mismatches=0", flush=True)
    print(f"CKPT_UPLOAD=PASS steps={len(CKPT_STEPS)}", flush=True)


def eval_line(path):
    lines = [s for s in Path(path).read_text(errors="replace").splitlines() if s.startswith("EVAL692=PASS")]
    need(len(lines) == 1, f"EVAL692=PASS 行数必须恰为1: {path}")
    return lines[0]


def run_eval(root):
    """eval/<评估名>/ 下放整份评估目录（除flock占位）与 eval_manifest.json；独立导出根、锁、标记与读回。"""
    name = eval_run_name()
    source = no_symlink(REPO / "v1-store/evaluation" / name)
    need(source.is_dir(), f"评估目录不存在: {source}")
    line = eval_line(member(source, "EVAL692.txt"))
    ckpt_owner = no_symlink(REPO / "v1-store/exports" / f"hf-ckpt-{RUN}" / "owner.json")
    need(ckpt_owner.is_file(), "eval上传要求ckpt增量上传已建立bucket归属")
    owner = json_file(ckpt_owner)
    need(owner.get("run_name") == RUN and owner.get("bucket") == bucket_id(), "ckpt归属与本run/bucket不符")
    prefix = f"{EVAL_PREFIX}{name}"
    sources = {}
    for path in sorted(source.rglob("*")):
        no_symlink(path)
        rel = path.relative_to(source)
        if rel.parts[0] in EVAL_SKIP_DIRS or not path.is_file():
            continue
        sources[f"{prefix}/{rel.as_posix()}"] = path
    need(sources, "评估目录为空")
    scan(sources, EXPORT / ALLOWLIST)
    stage = root / "stage"
    files = freeze_sources(sources, stage)
    body = {"schema": EVAL_SCHEMA, "run_name": RUN, "bucket": bucket_id(), "eval_name": name,
            "eval692": line, "files": dict(sorted(files.items()))}
    path = write_generated(stage, f"{prefix}/eval_manifest.json", body)
    files[f"{prefix}/eval_manifest.json"] = file_info(path)
    for key, item in sources.items():
        need(file_info(item) == files[key], f"评估源文件变化: {key}")
    publisher = Publisher(root, bucket_id(), {"kind": "4task-stage2-vla-eval", "run_name": RUN, "bucket": bucket_id()},
                          create=False)
    publisher.transfer(name, stage, files)
    # 已登记的全部评估（如以后补跑的其他seed）共同构成 eval/ 范围的精确集合。
    everything = {}
    for manifest_path in sorted((stage / EVAL_PREFIX.rstrip("/")).glob("*/eval_manifest.json")):
        record = json_file(manifest_path)
        need(record.get("schema") == EVAL_SCHEMA and record.get("run_name") == RUN and record.get("bucket") == bucket_id(),
             f"eval清单身份不符: {manifest_path}")
        everything.update(record["files"])
        everything[manifest_path.relative_to(stage).as_posix()] = file_info(manifest_path)
    publisher.verify_scope(stage, everything, scope="eval")
    print(f"EVAL_HF_VERIFY=PASS bucket={bucket_id()} scope=eval evals={len(list((stage / 'eval').iterdir()))} "
          f"objects={len(everything)} bytes={sum(v['bytes'] for v in everything.values())} mismatches=0", flush=True)


def run_stage_delete(root):
    """三条同时满足才删 stage：HF_VERIFY=PASS（ckpt范围）、CKPT_DELIVERY_READY=PASS step=50000、EVAL692 已出1行。"""
    stage = root / "stage"
    log_dir = REPO / LOG_DIR
    short = SHORT[role()]
    upload = log_dir / f"up-{short}-final.log"
    delivery = log_dir / f"delivery-{short}.log"
    evaluation = log_dir / f"{eval_run_name()}.log"
    reasons = []
    up_lines = upload.read_text(errors="replace").splitlines() if upload.is_file() else []
    if not (any(s.startswith(f"HF_VERIFY=PASS bucket={bucket_id()} scope=ckpt ") for s in up_lines)
            and f"CKPT_UPLOAD=PASS steps={len(CKPT_STEPS)}" in up_lines):
        reasons.append("hf_verify")
    if not (delivery.is_file()
            and f"CKPT_DELIVERY_READY=PASS step={DELIVERY_STEP}" in delivery.read_text(errors="replace").splitlines()):
        reasons.append("delivery")
    try:
        eval_line(evaluation)
    except (ValueError, OSError):
        reasons.append("eval692")
    if reasons:
        print(f"STAGE_DELETE=SKIP run={RUN} reason={','.join(reasons)} stage={stage}", flush=True)
        return
    need(stage.is_dir(), f"stage不存在: {stage}")
    no_symlink(stage)
    shutil.rmtree(stage)
    print(f"STAGE_DELETE=PASS run={RUN} stage={stage} free_bytes={shutil.disk_usage(root).free}", flush=True)


MODES = ("incremental", "stage", "upload", "eval", "stage-delete")


def main():
    global ATTEMPT
    mode = os.environ["O16_MODE"]
    need(mode in MODES, "未知导出模式")
    smoke = os.environ.get("SMOKE", "0")
    need(smoke in ("0", "1"), "SMOKE只允许0或1")
    smoke = smoke == "1"
    need(not smoke or mode == "incremental", "SMOKE只用于incremental（stage本身离线）")
    need(smoke or "O16_EXPORT_ROOT" not in os.environ, "路径覆盖只允许离线SMOKE")
    if os.environ.get("ATTEMPT_MANIFEST"):
        ATTEMPT = check_attempt(os.environ["ATTEMPT_MANIFEST"])
        print(f"ATTEMPT=PASS run={RUN} base={ATTEMPT['base_run']} bucket={ATTEMPT['bucket']}", flush=True)
    else:
        need(RUN in RUNS and os.environ.get("BUCKET") == RUNS[RUN][0], "RUN与BUCKET必须是计划固定配对")
    name = f"hf-eval-{RUN}" if mode == "eval" else f"hf-ckpt-{RUN}"
    root = no_symlink(Path(os.environ.get("O16_EXPORT_ROOT", REPO / "v1-store/exports" / name)))
    if mode == "stage-delete":
        with io.SingleInstanceLock(no_symlink(root / "upload.lock")):
            run_stage_delete(root)
    elif mode == "stage":
        # 不建 upload.lock、不读凭据；导出根由冻结步骤按需创建（strict_final 失败时不留空目录）。
        run_stage(root)
    else:
        root.mkdir(parents=True, exist_ok=True)
        with io.SingleInstanceLock(no_symlink(root / "upload.lock")):
            if mode == "incremental":
                run_incremental(smoke, root)
            elif mode == "upload":
                run_upload(root)
            else:
                run_eval(root)
    print("EXIT_CODE=0", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"4TASK_EXPORT=FAIL mode={os.environ.get('O16_MODE')} error={type(exc).__name__} detail={exc}", flush=True)
        print("EXIT_CODE=1", flush=True)
        raise SystemExit(1)
O16_PY
