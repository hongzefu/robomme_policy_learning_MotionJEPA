#!/usr/bin/env bash
# 官方16×100公开导出公共体；旧导出入口不引用本文件。
set -euo pipefail
: "${O16_MODE:?仅允许官方16×100包装入口调用}"
[ "$#" = 0 ] || { echo "本入口不接受额外参数" >&2; exit 2; }
O16_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export O16_REPO O16_MODE RUN BUCKET
export UV_CACHE_DIR=/scratch/hongze/.cache/uv CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
cd "$O16_REPO"
uv run --no-sync python - <<'O16_PY'
"""公开训练库和两条60k checkpoint；复用现有打包、HF CLI与逐对象匿名回读。"""
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
import tarfile
import time

REPO = Path(os.environ.get("O16_REPO", Path.cwd())).resolve()
EXPORT = REPO / "scripts/dataset/hf_export"
sys.path.insert(0, str(EXPORT))
import ckpt_upload_common as io
import scan_hygiene as hygiene

# 建库后仅回填库名；哈希和行数由实际已验证产物进入发布清单，不复制训练契约常量。
DATASET = "16task-orig100-1600ep-__FILL_AFTER_BUILD__"
TRAINSET_BUCKET = "HongzeFu/robomme-16task-orig100-1600ep-trainset-20261028-v1"
RUNS = {
    "orig100-ctx32x4x4-motion-s2split4x256-b128-60k": (
        "HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2split-60k-v1", "ctx", "context", 2048),
    "orig100-m32x4x4-modul-motion-s2split4x256-b128-60k": (
        "HongzeFu/robomme-vla-orig100-m32x4x4-modul-motion-s2split-60k-v1", "modul", "modulation", 1024),
}
RUN = os.environ.get("RUN", "")
ENC_RUN = "stage2-full144-1025-a"
ARCH = "wan-latent-stage2-blockwise-pool4-three-decoder"
LAYOUT = "motion-stage2-1x1024-grid32-demopad17-v1"
STEPS = tuple(range(5000, 60000, 5000)) + (59999,)
ROOT_FILES = frozenset({"history_config.txt", "history_config.resolved.yaml",
    "history_config.resolved.sha256", "motion_provenance.json", "encoder_dependencies.json",
    "online_restore_inputs.json", "online_restore_inputs.npz"})
READY = "motion_snapshot.ready.json"
EVIDENCE_FILES = {role: f"motion_evidence/{role}.json"
                  for role in ("G0", "G2", "G3", "wan_cache_import", "SOURCE_PIN")}
GROUPS = ("encoder", "encoder_state_sha256", "source_pin", "numerics", "stage2_compat", "vae")
SHA = re.compile(r"[0-9a-f]{64}\Z")
SCHEMA = "orig16-stage2-delivery-manifest-v1"
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

def load_yaml(path):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)

def check_online_inputs(record, payload):
    """发布真实恢复输入；每个数组的实际字节都必须能从交付NPZ得到。"""
    import numpy as np
    es = record.get("exec_start_idx")
    # 新入口先提交D+1首包，再每次推进不超过16帧；旧素材的两批描述继续可读。
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

def history_name(run):
    need(run in RUNS, "RUN不在两条已批准训练中")
    return f"perceptual-framesamp-{RUNS[run][1]}-32frame-4x4-motion-stage2-4x256-orig16.yaml"


def check_history(history):
    need(history == load_yaml(REPO / "src/mme_vla_suite/models/config/robomme" / history_name(RUN)),
         "run快照与对应正式yaml不同")
    motion = history.get("motion", {})
    need(motion.get("store_path") == f"v1-store/datasets/{DATASET}/motion-stage2-1024",
         "motion库与固定库名不符")
    need(motion.get("readout") == "stage2_pool4_split_v1" and motion.get("num_tokens") == 4
         and motion.get("dim") == 256, "正式训练必须为4×256读法")
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
    need((run / "history_config.txt").read_text().strip() == history_name(RUN), "history标签不符")
    history = load_yaml(sources["history_config.resolved.yaml"])
    sha = check_history(history)
    resolved_sha = file_info(sources["history_config.resolved.yaml"])["sha256"]
    need(sources["history_config.resolved.sha256"].read_text().split()[0] == resolved_sha, "resolved SHA不符")
    prov = json_file(sources["motion_provenance.json"])
    need(prov.get("motion_enabled") is True and prov.get("resolved_sha256") == resolved_sha,
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
    need(prov["stage2_compat"].get("cache_import_receipt_sha256") == evidence["wan_cache_import"]["sha256"],
         "缓存导入收据SHA未闭合")
    return sources, prov, history

def validate_store(repo, prov, history):
    """消费S2已验证的来源合同和实际表身份；不另维护行数、norm或manifest固定常量。"""
    from mme_vla_suite.datastore.motion_store import MotionMeta, validate_stage2_provenance
    validate_stage2_provenance(prov)
    need(prov["stage2_compat"].get("kind") == "published-cache-with-local-fill", "来源kind不是本轮缓存合同")
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
    need(numeric == {str(s) for s in STEPS}, "最终12步集合不符")
    need(not any("tmp" in p.name.lower() for p in run.iterdir()), "run根仍有临时保存目录")

def add_manifest(stage, files, name, **identity):
    path = write_generated(stage, name, manifest(files, **identity))
    return {**files, name: file_info(path)}


def run_checkpoints(mode, smoke, root):
    run = REPO / "v1-store/train-runs/mme_vla_suite_orig16_b128_60k" / RUN
    registered = REPO / "v1-store/external/motionjepa" / ENC_RUN
    log = REPO / "v1-store/logs" / f"{RUN}.log"
    bucket_id = RUNS[RUN][0]
    stage = root / "stage"
    allow = EXPORT / "hygiene_allowlist_orig16_stage2_vla.json"
    if mode == "final":
        strict_final(run, log)
    while not (run / READY).is_file():
        need(mode == "incremental" and not smoke and not train_finished(log), "缺原子根快照ready")
        print("ROOT_WAIT=1", flush=True)
        time.sleep(30)
    sources, prov, history = snapshot_sources(run, registered)
    validate_store(REPO, prov, history)
    sources["README.md"] = EXPORT / f"orig16_{RUNS[RUN][1]}32x4x4_split_60k_bucket_README.md"
    scan(sources, allow)
    files = freeze_sources(sources, stage)
    root_files = add_manifest(stage, files, "manifests/root.json", run_name=RUN, bucket=bucket_id)
    identity = {"kind": "orig16-stage2-vla", "run_name": RUN, "bucket": bucket_id,
                "root_manifest_sha256": root_files["manifests/root.json"]["sha256"]}
    publisher = None if smoke else Publisher(root, bucket_id, identity)
    if publisher:
        publisher.transfer("root", stage, root_files)
        for name, info in root_files.items():
            publisher.readback(name, info)
        print(f"ROOT_UPLOAD=PASS files={len(root_files)}", flush=True)
    all_files, processed = dict(root_files), set()
    while True:
        for step in STEPS:
            if step in processed:
                continue
            metadata = run / str(step) / "_CHECKPOINT_METADATA"
            if mode == "incremental" and (not metadata.is_file() or not json_file(metadata).get("commit_timestamp_nsecs")):
                continue
            step_src = step_sources(run, step)
            scan(step_src, allow)
            files = freeze_sources(step_src, stage)
            # 每份step清单包含根快照与encoder依赖，异地单步恢复不依赖其他step。
            path = write_generated(stage, f"manifests/step-{step}.json",
                                   manifest({**root_files, **files}, run_name=RUN, bucket=bucket_id, step=step))
            files[f"manifests/step-{step}.json"] = file_info(path)
            if publisher:
                publisher.transfer(str(step), stage, files)
                print(f"INC_UPLOADED step={step}", flush=True)
            all_files.update(files)
            sources.update(step_src)
            processed.add(step)
        if mode == "final":
            extra = {"_run-meta/train.log": log}
            scan(extra, allow)
            files = freeze_sources(extra, stage)
            all_files.update(files)
            sources.update(extra)
            path = write_generated(stage, "delivery_manifest.json",
                                   manifest(all_files, run_name=RUN, bucket=bucket_id, steps=list(STEPS)))
            all_files["delivery_manifest.json"] = file_info(path)
            if publisher:
                publisher.transfer("final", stage, {**files, "delivery_manifest.json": all_files["delivery_manifest.json"]})
                publisher.verify_all(stage, all_files)
            break
        if smoke or processed == set(STEPS) or train_finished(log):
            break
        print(f"INC_WAIT done={len(processed)} expected=12", flush=True)
        time.sleep(30)
    for name, path in sources.items():
        need(file_info(path) == all_files[name], f"源文件变化: {name}")
    if smoke:
        print(f"SMOKE=PASS kind=checkpoint mode={mode} steps={len(processed)}")
    elif mode == "final":
        print(f"HF_VERIFY=PASS bucket={bucket_id} objects={len(all_files)} bytes={sum(v['bytes'] for v in all_files.values())} mismatches=0")
        print("CKPT_UPLOAD=PASS steps=12")
    else:
        need(processed == set(STEPS), "训练结束但增量未齐12步")
        print(f"INC_END run={RUN} done=12")


def packed_trainset_files(stage):
    """以既有打包清单核交付对象，不能给续跑时同尺寸损坏的文件重新签摘要。"""
    packed = json_file(member(stage, "manifest/upload_manifest.json"))
    files = {}
    for group, checksum in (("shards", "sha256-shards.txt"), ("plain_files", "sha256-plain.txt")):
        hashes = {}
        for entry in packed[group]:
            name = relative(entry["rel"])
            need(name not in files, f"打包清单存在重复对象: {name}")
            files[name] = {"sha256": entry["sha256"], "bytes": entry["bytes"]}
            hashes[name] = entry["sha256"]
        lines = [line.split("  ", 1) for line in member(stage, f"checksums/{checksum}").read_text().splitlines()]
        listed = {name: digest for digest, name in lines}
        need(len(lines) == len(listed) and listed == hashes, f"打包校验和与清单不符: {checksum}")
    controls = {"manifest/upload_manifest.json", "checksums/sha256-shards.txt", "checksums/sha256-plain.txt",
                "checksums/sha256-source-files.txt", "README.md"}
    need(not controls.intersection(files), "打包对象与清单文件路径冲突")
    actual = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
    need(actual - {"delivery_manifest.json"} == set(files) | controls, "打包交付对象集合不符")
    for name, expected in files.items():
        need(file_info(member(stage, name)) == expected, f"打包清单SHA或字节数不符: {name}")
    need(sum(info["bytes"] for info in files.values()) == packed["total_bytes"], "打包清单总字节数不符")
    files.update({name: file_info(member(stage, name)) for name in controls})
    print(f"PACK_MANIFEST_VERIFY=PASS files={len(files)}", flush=True)
    return files


def run_trainset(smoke, root):
    lib = Path(os.environ.get("O16_LIB", REPO / "v1-store/datasets" / DATASET))
    norm = Path(os.environ.get("O16_NORM_STATS", REPO / "v1-store/train-assets/mme_vla_suite" / DATASET / "robomme/norm_stats.json"))
    allow = EXPORT / "hygiene_allowlist_orig100_trainset.json"
    stage = root / "stage"
    # 公开前扫描源文本，tar内元数据也在打包前扫描；原始数据字节不作脱敏改写。
    sources = {}
    for sub in ("meta", "source", "framesamp-4x4", "framesamp-8x8", "wan-latents", "motion-tokens-stage2", "motion-stage2-1024"):
        base = no_symlink(lib / sub)
        need(base.is_dir(), f"缺交付目录: {sub}")
        sources.update({p.relative_to(lib).as_posix(): p for p in base.rglob("*") if p.is_file() and hygiene.is_text(p)})
    sources["assets/robomme/norm_stats.json"] = norm
    mo = json_file(member(lib, "motion-stage2-1024/meta/store_meta.json"))
    compat = mo["provenance"]["stage2_compat"]
    need(compat.get("kind") == "published-cache-with-local-fill" and set(compat.get("evidence", {})) == set(EVIDENCE_FILES),
         "训练库缓存来源证据集合不符")
    for role, ref in compat["evidence"].items():
        path = Path(ref["path"])
        if not path.is_absolute():
            path = lib / "motion-stage2-1024" / path
        need(file_info(path)["sha256"] == ref["sha256"], f"训练库证据SHA不符: {role}")
        sources[f"evidence/{role}.json"] = path
    scan(sources, allow)
    packed = ("source/data", "source/features", "wan-latents", "motion-tokens-stage2")
    estimated = sum(p.stat().st_size + 1024 for sub in packed for p in (lib / sub).rglob("*") if p.is_file())
    have = sum(p.stat().st_size for p in stage.rglob("*.tar"))
    required = max(0, estimated - have) + 100 * 1024**3 + DISK_MARGIN
    available = shutil.disk_usage(root).free
    print(f"DF_CHECK avail={available} need={required}", flush=True)
    need(smoke or available >= required, "打包和匿名回读空间不足")
    command = ["uv", "run", "--no-sync", "python", str(EXPORT / "pack_and_hash.py"),
               "--layout", "orig100-stage2-v1", "--lib", str(lib), "--stage-root", str(root),
               "--bucket", f"hf://buckets/{TRAINSET_BUCKET}", "--norm-stats", str(norm), "--workers", "8"]
    if smoke:
        command += ["--limit-shards", "2"]
    subprocess.run(command, check=True, cwd=REPO)
    readme = EXPORT / "orig100_trainset_bucket_README.md"
    freeze_sources({"README.md": readme}, stage)
    if smoke:
        checked = 0
        for archive in stage.rglob("*.tar"):
            with tarfile.open(archive) as tar:
                for item in tar:
                    if item.isfile():
                        need(tar.extractfile(item).read() == (lib / "source" / item.name).read_bytes(),
                             f"SMOKE源字节不同: {item.name}")
                        checked += 1
        need(checked > 0, "SMOKE没有tar成员")
        print(f"SMOKE=PASS kind=trainset members={checked} network=0")
        return
    files = packed_trainset_files(stage)
    files = add_manifest(stage, files, "delivery_manifest.json", bucket=TRAINSET_BUCKET, dataset=DATASET)
    scan({name: stage / name for name in files}, allow)
    publisher = Publisher(root, TRAINSET_BUCKET, {"kind": "orig100-trainset", "bucket": TRAINSET_BUCKET,
                          "manifest_sha256": files["delivery_manifest.json"]["sha256"]})
    publisher.transfer("trainset", stage, files)
    publisher.verify_all(stage, files)
    print(f"HF_VERIFY=PASS bucket={TRAINSET_BUCKET} objects={len(files)} bytes={sum(v['bytes'] for v in files.values())} mismatches=0")


def main():
    mode = os.environ["O16_MODE"]
    need(mode in ("trainset", "incremental", "final"), "未知导出模式")
    smoke = os.environ.get("SMOKE", "0")
    need(smoke in ("0", "1"), "SMOKE只允许0或1")
    smoke = smoke == "1"
    overrides = [key for key in ("O16_LIB", "O16_NORM_STATS", "O16_EXPORT_ROOT") if key in os.environ]
    need(smoke or not overrides, "路径覆盖只允许离线SMOKE")
    need("__FILL_AFTER_BUILD__" not in DATASET or (smoke and mode == "trainset" and "O16_LIB" in os.environ),
         "建库后须回填DATASET，当前仍含__FILL_AFTER_BUILD__")
    if mode != "trainset":
        need(RUN in RUNS and os.environ.get("BUCKET") == RUNS[RUN][0], "RUN与BUCKET必须是计划固定配对")
    name = "hf-orig100-trainset" if mode == "trainset" else f"hf-ckpt-{RUN}"
    root = no_symlink(Path(os.environ.get("O16_EXPORT_ROOT", REPO / "v1-store/exports" / name)))
    root.mkdir(parents=True, exist_ok=True)
    no_symlink(root / "stage").mkdir(exist_ok=True)
    with io.SingleInstanceLock(no_symlink(root / "upload.lock")):
        if mode == "trainset":
            run_trainset(smoke, root)
        else:
            run_checkpoints(mode, smoke, root)
    print("EXIT_CODE=0", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ORIG16_EXPORT=FAIL error={type(exc).__name__} detail={exc}", flush=True)
        print("EXIT_CODE=1", flush=True)
        raise SystemExit(1)
O16_PY
