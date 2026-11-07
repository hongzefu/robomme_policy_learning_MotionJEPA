#!/usr/bin/env python3
"""Wan 池化 60k 的 12 份 checkpoint 公开备份；增量与最终入口共用本文件。

行为沿用 `_stage2_orig16_vla_ckpt_export_common.sh`（打包、卫生扫描、HF CLI、匿名读回、判定行），
差异只有三处：身份校验换成 timepool 快照；增量按序等待；所有 v1-store 路径从显式 --store-root 派生，
代码从本文件所在源码树加载，因此可以从独立固定源码树运行。对 train-runs 只读，不做任何本地清理。

    run_orig16_ctx_timepool_60k_ckpt_inc.sh    --store-root <主检出> [--train-tmux <训练会话名>]
    run_orig16_ctx_timepool_60k_ckpt_export.sh --store-root <主检出>
"""
from __future__ import annotations

import argparse
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

REPO = Path(__file__).resolve().parents[3]
EXPORT = REPO / "scripts/dataset/hf_export"
# 主检出 venv 的 editable 安装指向主检出 src；放到最前面才能加载本源码树。
for _entry in (str(EXPORT), str(REPO / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
import ckpt_upload_common as io  # noqa: E402
import scan_hygiene as hygiene  # noqa: E402

RUN = "orig100-ctx32x4x4-wan-timepool8x8-b128-60k"
BUCKET = "HongzeFu/robomme-vla-orig100-ctx32x4x4-wan-timepool8x8-60k-v1"
DATASET = "16task-orig100-1600ep-9e109419"
READOUT = "wan_raw_timepool8_spatial8x8_v1"
ENCODER_KIND = "wan_fixedpool1024"
LAYOUT = "motion-wan-timepool1024-grid32-demopad17-v1"
HISTORY_NAME = "perceptual-framesamp-ctx-32frame-4x4-wan-timepool1024-orig16.yaml"
STEPS = tuple(range(5000, 60000, 5000)) + (59999,)
TRAIN_PARENT = "v1-store/train-runs/mme_vla_suite_orig16_b128_60k"
MOTION_REL = f"v1-store/datasets/{DATASET}/motion-wan-timepool1024"
FRAMESAMP_REL = f"v1-store/datasets/{DATASET}/framesamp-4x4"
READY = "motion_snapshot.ready.json"
EVIDENCE_FILES = frozenset(f"motion_evidence/{role}.json" for role in
                           ("DATA_LINEAGE", "WAN_RESTORE_EQ", "WINDOW_EQ", "POOL_CONTRACT", "POOL_TABLE_EQ"))
ROOT_FILES = frozenset({"history_config.txt", "history_config.resolved.yaml", "history_config.resolved.sha256",
                        "motion_provenance.json", "timepool_window_map.json", "online_restore_inputs.json",
                        "online_restore_inputs.npz"}) | EVIDENCE_FILES
README = EXPORT / "orig16_ctx_timepool_60k_bucket_README.md"
ALLOWLIST = EXPORT / "hygiene_allowlist_orig16_timepool_vla.json"
SHA = re.compile(r"[0-9a-f]{64}\Z")
MANIFEST_SCHEMA = "orig16-timepool-delivery-manifest-v1"
UPLOADED_SCHEMA = "orig16-timepool-uploaded-v1"
TOKENIZER_URI = "gs://big_vision/paligemma_tokenizer.model"
TOKENIZER_SHA = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
MAX_STEP_BYTES = 16 * 1024**3
DISK_MARGIN = 40 * 1024**3
POLL_SECONDS = 30


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
    return {"schema": MANIFEST_SCHEMA, **identity, "files": dict(sorted(files.items()))}


def load_yaml(path):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


class Paths:
    """全部 v1-store 路径从 --store-root 派生；smoke 覆盖只在 main 里放行。"""

    def __init__(self, store_root, *, export_root=None, run_root=None):
        self.store = no_symlink(os.path.abspath(store_root))
        need((self.store / "v1-store").is_dir(), f"--store-root 下没有 v1-store: {self.store}")
        self.source_override = run_root is not None
        self.run = no_symlink(os.path.abspath(run_root)) if run_root else self.store / TRAIN_PARENT / RUN
        self.log = self.store / "v1-store/logs" / f"{RUN}.log"
        self.export = no_symlink(os.path.abspath(export_root)) if export_root else (
            self.store / "v1-store/exports" / f"hf-ckpt-{RUN}")
        self.token_file = self.store / "v1-store/secrets/hf.env"


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


def check_history(history):
    need(history == load_yaml(REPO / "src/mme_vla_suite/models/config/robomme" / HISTORY_NAME),
         "run快照与源码树正式yaml不同")
    motion = history.get("motion", {})
    need(motion.get("store_path") == MOTION_REL, "motion库与固定库名不符")
    need(motion.get("readout") == READOUT and motion.get("encoder_kind") == ENCODER_KIND
         and motion.get("num_tokens") == 1 and motion.get("dim") == 1024 and motion.get("budget") == 96,
         "history不是wan池化1×1024、96窗读法")


def snapshot_sources(paths):
    """ready 列举的 12 个根文件；SHA 闭合与来源合同由 verify_timepool_snapshot 判定。"""
    from mme_vla_suite.policies.motion_client import verify_timepool_snapshot
    run = no_symlink(paths.run)
    need(paths.source_override or run.name == RUN, "源run身份不符")
    prov = verify_timepool_snapshot(run)
    ready_files = json_file(member(run, READY))["files"]
    need(set(ready_files) == ROOT_FILES, "ready文件集合必须精确等于12个已知根文件")
    sources = {name: member(run, name) for name in sorted(ready_files)}
    check_online_inputs(json_file(sources["online_restore_inputs.json"]), sources["online_restore_inputs.npz"])
    need(sources["history_config.txt"].read_text().strip() == HISTORY_NAME, "history标签不符")
    history = load_yaml(sources["history_config.resolved.yaml"])
    check_history(history)
    timepool = prov["timepool"]
    need(prov["history_config_name"] == HISTORY_NAME, "来源history名不符")
    need(timepool["readout"] == READOUT and timepool["encoder_kind"] == ENCODER_KIND and timepool["layout"] == LAYOUT,
         "来源readout/encoder_kind/layout不符")
    sources[READY] = member(run, READY)
    return sources, prov


def validate_store(paths, prov):
    """快照必须绑定 --store-root 下实际的 motion 表与同数据集帧表。"""
    from mme_vla_suite.datastore.motion_store import MotionMeta, validate_timepool_provenance
    root = paths.store / MOTION_REL
    need(prov["motion_root"] == str(root), "根快照motion_root不是store-root下固定库")
    need(prov["framesamp_root"] == str(paths.store / FRAMESAMP_REL), "根快照framesamp_root不是同数据集帧表")
    loaded = MotionMeta.load(root)
    meta_path = member(root, "meta/store_meta.json")
    meta = json_file(meta_path)
    need(meta.get("status") == "verified" and meta.get("layout") == LAYOUT, "motion表未验证或layout错误")
    need(prov["motion_store_meta_sha256"] == file_info(meta_path)["sha256"], "根快照未绑定实际motion表")
    table = meta["tables"]["motion_token"]
    need(table.get("row_shape") == [1, 1024] and table.get("row_bytes") == 4096 and table.get("dtype") == "float32",
         "motion表形状不符")
    need(file_info(member(root, table["relpath"])) == {"sha256": loaded.table_sha256, "bytes": meta["num_rows"] * 4096},
         "motion全表SHA或字节数不符")
    need(prov["timepool"] == meta["provenance"], "快照来源与表内来源不同")
    validate_timepool_provenance(prov["timepool"], status="verified", manifest_sha256=meta["manifest_sha256"],
                                 table_sha256=loaded.table_sha256, index_sha256=meta["motion_index_sha256"],
                                 num_rows=meta["num_rows"])


def committed(run, step):
    """Orbax 已提交判据沿旧实现：_CHECKPOINT_METADATA 带正的 commit_timestamp_nsecs；未写完按未提交。"""
    metadata = Path(run) / str(step) / "_CHECKPOINT_METADATA"
    if not metadata.is_file():
        return False
    try:
        commit = json_file(metadata).get("commit_timestamp_nsecs")
    except ValueError:
        return False
    return type(commit) is int and commit > 0


def step_sources(run, step, steps):
    need(step in steps, "非计划step")
    root = no_symlink(Path(run) / str(step))
    need(root.is_dir(), f"step缺失: {step}")
    metadata = json_file(member(root, "_CHECKPOINT_METADATA"))
    need(committed(run, step), f"Orbax尚未提交: {step}")
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
    """仅扫描准确上传集合；疑似凭据不输出原文。"""
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
    def __init__(self, root, token_file, identity):
        self.root, self.identity = root, identity
        token = io.load_hf_token(token_file)
        self.bucket = io.HFBucket(BUCKET, token, cache_root=root / "cache")
        self.anon = AnonymousBucket(BUCKET, "", cache_root=root / "cache")
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
            rc, _, _ = self.bucket._run(["buckets", "create", BUCKET], 600)
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
        expected = {"schema": UPLOADED_SCHEMA, **self.identity, "object": key,
                    "files_sha256": hashlib.sha256(canon(files)).hexdigest()}
        if marker.exists():
            need(json_file(marker) == expected, f"共享上传标记身份冲突: {key}")
        else:
            for name, info in sorted(files.items()):
                need(file_info(member(stage, name)) == info, f"stage漂移: {name}")
                need(self.bucket.upload(stage / name, name).ok, f"上传失败: {name}")
                self.readback(name, info)
            # .done 只代表已传输和匿名回读，不表示严格恢复完成。
            io.atomic_write_json(marker, expected)
        return expected

    def verify_all(self, stage, files):
        remote = self.anon.list_files()
        need(remote == {n: m["bytes"] for n, m in files.items()}, "远端文件集合或字节数不符")
        for name, info in sorted(files.items()):
            self.readback(name, info)
            need(file_info(member(stage, name)) == info, f"最终stage漂移: {name}")
        need(self.anon.entry().get("private") is False, "匿名公开性复核失败")


def train_exit(log):
    """训练日志的 EXIT_CODE 尾行；还没写时返回 None。只读。"""
    if not log.is_file():
        return None
    lines = log.read_text(errors="replace").splitlines()
    run_line = next((s for s in lines if s.startswith("RUN=")), None)
    need(run_line in (None, f"RUN={RUN}"), "训练日志run不符")
    last = next((s for s in reversed(lines) if s.strip()), "")
    match = re.fullmatch(r"EXIT_CODE=(-?\d+)", last)
    return int(match.group(1)) if match else None


def session_alive(session):
    return subprocess.run(["tmux", "has-session", "-t", f"={session}"], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, timeout=30).returncode == 0


def train_ended(log, session):
    """未结束返回 None，否则返回结束原因。强杀会话不写 EXIT_CODE，靠 --train-tmux 识别。"""
    code = train_exit(log)
    if code is None and session and not session_alive(session):
        code = train_exit(log)  # 尾行写完会话才退出：重读一次，排除两次查询之间正常结束
        if code is None:
            return "session_gone_without_exit_code"
    return None if code is None else f"EXIT_CODE={code}"


def strict_final(paths, steps):
    code = train_exit(paths.log)
    need(code == 0, f"最终导出要求训练EXIT_CODE=0，实际{code}")
    need(any(s == f"RUN={RUN}" for s in paths.log.read_text(errors="replace").splitlines()), "训练日志缺run身份")
    numeric = {p.name for p in paths.run.iterdir() if p.name.isdecimal()}
    need(numeric == {str(s) for s in steps}, "最终12步集合不符")
    need(not any("tmp" in p.name.lower() for p in paths.run.iterdir()), "run根仍有临时保存目录")


def add_manifest(stage, files, name, **identity):
    path = write_generated(stage, name, manifest(files, **identity))
    return {**files, name: file_info(path)}


def run_checkpoints(mode, smoke, paths, steps, session):
    run, log, root = paths.run, paths.log, paths.export
    stage = root / "stage"
    if mode == "final":
        strict_final(paths, steps)
    while not (run / READY).is_file():
        need(mode == "incremental" and not smoke and train_ended(log, session) is None, "缺原子根快照ready")
        print("ROOT_WAIT=1", flush=True)
        time.sleep(POLL_SECONDS)
    sources, prov = snapshot_sources(paths)
    validate_store(paths, prov)
    print("SNAPSHOT=PASS files=13", flush=True)
    sources["README.md"] = README
    scan(sources, ALLOWLIST)
    files = freeze_sources(sources, stage)
    root_files = add_manifest(stage, files, "manifests/root.json", run_name=RUN, bucket=BUCKET)
    identity = {"kind": "orig16-wan-timepool-vla", "run_name": RUN, "bucket": BUCKET,
                "root_manifest_sha256": root_files["manifests/root.json"]["sha256"]}
    publisher = None if smoke else Publisher(root, paths.token_file, identity)
    if publisher:
        publisher.transfer("root", stage, root_files)
        for name, info in root_files.items():
            publisher.readback(name, info)
        print(f"ROOT_UPLOAD=PASS files={len(root_files)}", flush=True)
    all_files, processed = dict(root_files), set()
    while True:
        # 先取结束状态再扫步：训练结束前提交的最后一步一定会被这一轮扫到，不会误判缺步。
        ended = train_ended(log, session) if mode == "incremental" and not smoke else None
        for step in steps:
            if step in processed:
                continue
            if mode == "incremental" and not committed(run, step):
                break  # 按序：前一步没提交就不看后面的
            step_src = step_sources(run, step, steps)
            scan(step_src, ALLOWLIST)
            files = freeze_sources(step_src, stage)
            print(f"STEP_PACKED step={step} files={len(files)} bytes={sum(v['bytes'] for v in files.values())}", flush=True)
            # 每份 step 清单包含根快照，异地单步恢复不依赖其他 step。
            path = write_generated(stage, f"manifests/step-{step}.json",
                                   manifest({**root_files, **files}, run_name=RUN, bucket=BUCKET, step=step))
            files[f"manifests/step-{step}.json"] = file_info(path)
            if publisher:
                publisher.transfer(str(step), stage, files)
                print(f"INC_UPLOADED step={step}", flush=True)
            all_files.update(files)
            sources.update(step_src)
            processed.add(step)
        if mode == "final":
            extra = {"_run-meta/train.log": log}
            scan(extra, ALLOWLIST)
            files = freeze_sources(extra, stage)
            all_files.update(files)
            sources.update(extra)
            path = write_generated(stage, "delivery_manifest.json",
                                   manifest(all_files, run_name=RUN, bucket=BUCKET, steps=list(steps)))
            all_files["delivery_manifest.json"] = file_info(path)
            if publisher:
                publisher.transfer("final", stage, {**files, "delivery_manifest.json": all_files["delivery_manifest.json"]})
                publisher.verify_all(stage, all_files)
            break
        if smoke or processed == set(steps):
            break
        missing = [s for s in steps if s not in processed]
        need(ended is None, f"训练已结束({ended})但增量未齐12步 missing={missing}")
        print(f"INC_WAIT done={len(processed)} expected={len(steps)} next={missing[0]}", flush=True)
        time.sleep(POLL_SECONDS)
    for name, path in sources.items():
        need(file_info(path) == all_files[name], f"源文件变化: {name}")
    if smoke:
        print(f"SMOKE=PASS kind=checkpoint mode={mode} steps={len(processed)} stage={stage} network=0 credentials=0")
    elif mode == "final":
        print(f"HF_VERIFY=PASS bucket={BUCKET} objects={len(all_files)} bytes={sum(v['bytes'] for v in all_files.values())} mismatches=0")
        print(f"CKPT_UPLOAD=PASS steps={len(steps)}")
    else:
        print(f"INC_END run={RUN} done={len(steps)}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Wan 池化 60k checkpoint 公开备份")
    parser.add_argument("--mode", required=True, choices=("incremental", "final"))
    parser.add_argument("--store-root", default=os.environ.get("O16_STORE_ROOT"),
                        help="持有 v1-store 的主检出；也可用环境变量 O16_STORE_ROOT")
    parser.add_argument("--train-tmux", help="训练 tmux 会话全名；增量用它识别被强杀、未写 EXIT_CODE 的训练")
    parser.add_argument("--export-root", help="仅 SMOKE=1：导出工作目录，SMOKE 必须显式给")
    parser.add_argument("--smoke-run-root", help="仅 SMOKE=1：改检查另一个 run 根")
    parser.add_argument("--smoke-steps", help="仅 SMOKE=1：逗号分隔的步数，默认计划 12 步")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    need(args.store_root, "必须给 --store-root 或 O16_STORE_ROOT")
    smoke = os.environ.get("SMOKE", "0")
    need(smoke in ("0", "1"), "SMOKE只允许0或1")
    smoke = smoke == "1"
    overrides = (args.export_root, args.smoke_run_root, args.smoke_steps)
    need(smoke or not any(overrides), "路径与步数覆盖只允许离线SMOKE")
    need(not smoke or args.export_root, "SMOKE必须显式给 --export-root，不写正式导出目录")
    # 本工具全程不用 GPU；下面才开始加载 mme_vla_suite。
    os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    import mme_vla_suite
    code_root = Path(mme_vla_suite.__file__).resolve()
    need(code_root.is_relative_to(REPO), f"mme_vla_suite 未从本源码树加载: {code_root}")
    paths = Paths(args.store_root, export_root=args.export_root, run_root=args.smoke_run_root)
    steps = tuple(int(s) for s in args.smoke_steps.split(",")) if args.smoke_steps else STEPS
    print(f"CODE_ROOT={REPO} STORE_ROOT={paths.store} RUN={RUN} BUCKET={BUCKET} MODE={args.mode} SMOKE={int(smoke)}", flush=True)
    paths.export.mkdir(parents=True, exist_ok=True)
    no_symlink(paths.export / "stage").mkdir(exist_ok=True)
    with io.SingleInstanceLock(no_symlink(paths.export / "upload.lock")):
        run_checkpoints(args.mode, smoke, paths, steps, args.train_tmux)
    print("EXIT_CODE=0", flush=True)


def cli(argv=None):
    try:
        main(argv)
    except Exception as exc:
        print(f"TIMEPOOL_EXPORT=FAIL error={type(exc).__name__} detail={exc}", flush=True)
        print(f"WAIT_USER_DECISION run={RUN} stage=ckpt-backup bucket={BUCKET}", flush=True)
        print("EXIT_CODE=1", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
