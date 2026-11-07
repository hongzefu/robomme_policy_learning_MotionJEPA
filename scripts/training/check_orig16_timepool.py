#!/usr/bin/env python3
"""原版 16 任务 Wan 时间优先池化实验的 G0–G9 验收器（1106 计划第 7、8、10 节，用户裁决：全部判定行集中于本脚本）。

每条子命令把 JSON 记录写进 ``--out-dir``（文件名 ``<ROLE>.json``），stdout 打判定行，通过退 0，失败给出具体字段与索引、退 1。
已有同名记录不覆盖：先改名为 ``<ROLE>.attempt-<美东时刻>.json`` 留证，再写新记录。

顶层只依赖标准库与 NumPy：``online`` 与 ``_sidecar-tokens`` 要在 ``v1-store/venvs/wan-stage2`` 解释器里按路径运行。
需要 JAX / OmegaConf / mme_vla_suite 的部分都在函数内延迟导入。

参考计算的独立性：G2 的映射从计划、收据和 manifest 重算，不调用 ``validate_timepool_window_map``；G3/G4 的参考公式写在本文件
（``independent_pool_f32`` / ``reference_pool_f64``），不 import ``wan_timepool``（G3 只把它当被验对象按路径加载）；
TRAIN_CONFIG_EQ 的差分器不依赖 ``check_orig16_timepool_launch.py``。
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
YAML_DIR = SRC / "mme_vla_suite/models/config/robomme"
POOL_SOURCE = SRC / "mme_vla_suite/shared/wan_timepool.py"
SIDECAR_SOURCE = REPO / "scripts/dataset/wan_timepool/timepool_sidecar.py"
STAGE2_CHECK_SOURCE = REPO / "scripts/training/check_stage2_checkpoint.py"
RECORD_SCHEMA = "orig16-timepool-gate-record-v1"
CONFIG_NAME = "mme_vla_suite_orig16_b128_60k"
EASTERN = ZoneInfo("America/New_York")

CHUNK_BYTES = 9 * 16 * 32 * 32 * 4
ROW_BYTES = 4096
LATENT_SHAPE = (9, 16, 32, 32)
MOTION_TABLE = "motion_token.f32.bin"
STORE_META = "meta/store_meta.json"
STORE_INDEX = "meta/motion_index.json"
STORE_WINDOW_MAP = "meta/timepool_window_map.json"
KIND = "wan_fixedpool1024"
READOUT = "wan_raw_timepool8_spatial8x8_v1"
LAYOUT = "motion-wan-timepool1024-grid32-demopad17-v1"
POOL_CONTRACT_LITERAL = {"version": "wan-timepool-pool-v1", "raw": True, "drop_first": True, "temporal_mean8": True,
                         "spatial_block4": True, "flatten_CHW": True, "float32": True,
                         "reduction_order": "T>block_W>block_H", "backend": "numpy-cpu"}
# G3 误差界（事先写定，不看结果调整）：最长运算链 7 次时间加法 + /8 + 3 次块内 W + 3 次块内 H + /16 = 15 步，
# 每步相对舍入至多 u=2^-24，所以 |F-P| ≤ ((1+u)^15-1)·S + 2^-149，S=(1/128)Σ|Z|，覆盖 128 个被加元素。
FP32_UNIT_ROUNDOFF = 2.0 ** -24
FP32_POOL_DEPTH = 15
FP32_ABS_FLOOR = 2.0 ** -149
BOUND_GAMMA = (1.0 + FP32_UNIT_ROUNDOFF) ** FP32_POOL_DEPTH - 1.0
TRAIN_SCALARS = {"loss", "grad_norm", "llm_grad_norm", "param_norm", "mem_enc_norm"}
NONFINITE_LOG = re.compile(
    rb"^\r*Step [0-9]+:.*(loss|grad_norm|llm_grad_norm|param_norm|mem_enc_norm)=[-+]?(nan|inf)(,|\s|$)"
    rb"|RESOURCE_EXHAUSTED:|CUDA out of memory|OutOfMemoryError|TRAIN_NUMERICS=FAIL", re.IGNORECASE)

# 真实 orig16 全库的固定期望（计划第 1、4、6、8、10 节）。CLI 不提供改写入口；单测用 monkeypatch 换成夹具值。
EXPECT = dict(
    files=16, source_windows=45228, source_copy=42404, source_fill=2824,
    windows=23253, copy=21814, fill=1439, demo_tail=386, exec_fill=1053, max_visible=44, bytes=95244288,
    episodes=1600, exec_samples=476857, framesamp_layout="framesamp-4x4-v1",
    episode_manifest_file_sha256="93d7c5a0c1f34fd955de5f8c37370369e54457019f75e56aea8908cba894d999",
    episode_manifest_canonical_sha256="337c4fcde163974039b5c0f2ad1f46ebaacd272719f72818503e23c1360447e2",
    input_manifest_sha256="5d1bfed54377c22e68bc1883ebd35fdf77372129ac1565dd6a78601c1d9965f9",
    plan_canonical_sha256="b56fc7dbf224ea3e21bbe68b143a0b9e849d5c870b1b6cc4571efda4888dfa19",
    norm_sha256="c5d45b4bdb481ca46d18aeb32446bbedec34de435eef31d3212a179ee7868bcd",
    published_repo="HongzeFu/robomme-original-16x100-wan21-front-33f-dense-v1",
    published_revision="384d5db410be713a49357bf3f55ca4f4d2f559a0",
    total=3363663120, trainable=2948859424, frozen=414803696, leaves=59, trainable_leaves=36,
    mem_encoder=10229248, motion_path=3869440,
    online_tasks=16, online_per_task=4, online_windows=64,
    checkpoint_steps=tuple(range(5000, 60000, 5000)) + (59999,), final_state_step=60000, final_checkpoint_step=59999,
)

GATE_ROLES = {
    "lineage": ("DATA_LINEAGE",), "restore": ("WAN_RESTORE_EQ",), "window": ("WINDOW_EQ",),
    "pool-contract": ("POOL_CONTRACT",), "pool-table": ("POOL_TABLE_EQ",), "vla-param": ("VLA_PARAM_EQ",),
    "vla-init": ("VLA_INIT_EQ",), "train-config": ("TRAIN_CONFIG_EQ",), "non-motion": ("NON_MOTION_EQ",),
    "pool-shared": ("POOL_SHARED_EQ",), "online": ("ONLINE_WINDOW_EQ", "ONLINE_NUMERICS"), "save-load": ("SAVE_LOAD",),
    "final": ("TRAIN_FINAL_RECORD", "CKPT_INVENTORY", "RESTORE_POOL"),
}
BIND_ROLES = ("DATA_LINEAGE", "WAN_RESTORE_EQ", "WINDOW_EQ", "POOL_CONTRACT", "POOL_TABLE_EQ")


# ── 通用小件 ──────────────────────────────────────────────────────────────────


def now_et() -> str:
    return datetime.datetime.now(EASTERN).isoformat(timespec="seconds")


def sha_bytes(data) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def same(a, b) -> bool:
    """JSON 语义严格相等：1 与 True、1 与 1.0 不算相同。"""
    return canonical(a) == canonical(b)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_body_sha(payload: dict) -> str:
    return sha_bytes(canonical({k: v for k, v in payload.items() if k != "sha256"}))


def load_by_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_src() -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def short(value, limit=160):
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


class Findings:
    """累积失败，保留前 200 条细节（字段 + 索引 + 实得 + 期望），总数单独计。"""

    def __init__(self):
        self.items, self.total = [], 0

    def add(self, field, index=None, got=None, want=None):
        self.total += 1
        if len(self.items) < 200:
            self.items.append(dict(field=field, index=index, got=short(got), want=short(want)))

    def eq(self, field, got, want, index=None) -> bool:
        if same(got, want):
            return True
        self.add(field, index, got, want)
        return False

    def __bool__(self):
        return self.total > 0

    def fail_line(self, role) -> str:
        first = self.items[0]
        return (f"{role}=FAIL field={first['field']} index={first['index']} got={first['got']} "
                f"want={first['want']} total={self.total}")


@dataclasses.dataclass
class RoleResult:
    role: str
    pass_line: str
    findings: Findings
    metrics: dict


class Context:
    def __init__(self, args):
        self.args, self.inputs, self.outputs = args, {}, {}

    def input(self, label, path, sha=None):
        self.inputs[label] = dict(path=str(path), sha256=sha if sha is not None else sha_file(path))

    def output(self, label, path, sha=None):
        self.outputs[label] = dict(path=str(path), sha256=sha if sha is not None else sha_file(path))


def code_identity() -> dict:
    run = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=False)
    return dict(head=run("rev-parse", "HEAD").stdout.strip(), dirty=bool(run("status", "--porcelain").stdout.strip()),
                script_sha256=sha_file(__file__), repo=str(REPO))


def environment() -> dict:
    keys = ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT", "MMEVLA_V1_STORE",
            "OPENPI_DATA_HOME", "HF_HOME", "HF_HUB_OFFLINE")
    return dict(python=sys.version, executable=sys.executable, prefix=sys.prefix, numpy=np.__version__,
                hostname=socket.gethostname(), platform=platform.platform(),
                env={key: os.environ.get(key) for key in keys})


def store_binding(store) -> dict:
    """现场重算表与索引 SHA，并要求与 store_meta 记录一致；这两个值就是 pack verify 核对的绑定字段。"""
    store = Path(store)
    meta = read_json(store / STORE_META)
    table_sha, index_sha = sha_file(store / MOTION_TABLE), sha_file(store / STORE_INDEX)
    if table_sha != meta["tables"]["motion_token"]["sha256"]:
        raise ValueError(f"表现场 SHA {table_sha} 与 store_meta 不符")
    if index_sha != meta["motion_index_sha256"]:
        raise ValueError(f"索引现场 SHA {index_sha} 与 store_meta 不符")
    return dict(store_table_sha256=table_sha, motion_index_sha256=index_sha)


def write_record(out_dir, role, record) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{role}.json"
    if path.exists():
        stamp = datetime.datetime.now(EASTERN).strftime("%Y%m%dT%H%M%S%f")
        os.rename(path, out_dir / f"{role}.attempt-{stamp}.json")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)
    return path


def run_gate(args, gate, body) -> int:
    """统一入口：写记录、打判定行。任何异常都转成带字段的 FAIL，不让判定行缺失。"""
    started, ctx = now_et(), Context(args)
    identity = code_identity()
    try:
        results = body(args, ctx)
    except Exception as exc:  # noqa: BLE001  任何异常都必须落成 FAIL 记录
        results = []
        for role in GATE_ROLES[gate]:
            findings = Findings()
            findings.add("exception", type(exc).__name__, str(exc), None)
            results.append(RoleResult(role, "", findings, {}))
    binding = {}
    store = getattr(args, "store", None)
    if store:
        try:
            binding = store_binding(store)
        except Exception as exc:  # noqa: BLE001
            for result in results:
                result.findings.add("store_binding", str(store), str(exc), None)
    expect_head = getattr(args, "expect_head", None)
    if expect_head and expect_head != identity["head"]:
        for result in results:
            result.findings.add("code_head", None, identity["head"], expect_head)
    code = 0
    for result in results:
        passed = not result.findings
        line = result.pass_line if passed else result.findings.fail_line(result.role)
        record = dict(schema=RECORD_SCHEMA, gate=gate, role=result.role, status="PASS" if passed else "FAIL",
                      verdict=line, code=identity, inputs=ctx.inputs, outputs=ctx.outputs,
                      command=[sys.executable, *sys.argv], environment=environment(), started_at_et=started,
                      finished_at_et=now_et(), exit_code=0 if passed else 1, metrics=result.metrics,
                      findings_total=result.findings.total, findings=result.findings.items, **binding)
        write_record(args.out_dir, result.role, record)
        print(line, flush=True)
        code = code or (0 if passed else 1)
    return code


def hash_many(paths: dict, workers: int) -> dict:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return dict(zip(paths, pool.map(sha_file, paths.values()), strict=True))


def task_of_h5(name: str) -> str:
    match = re.fullmatch(r"record_dataset_(.+)\.h5", name)
    if match is None:
        raise ValueError(f"H5 文件名不合法: {name}")
    return match.group(1)


def num_grid(seg_len: int, min_real: int, stride: int) -> int:
    return (seg_len - min_real) // stride + 1 if seg_len >= min_real else 0


def segment_spans(episode) -> list[tuple[str, int, int, int]]:
    """(段, 全域起点, 段长, 最少真帧)；demo 尾窗至少 17 真帧，exec 只取完整 33 帧。"""
    demo = episode["exec_start_idx"]
    return [("demo", 0, demo, 17), ("exec", demo, episode["num_timesteps"] - demo, 33)]


# ── G0 同源 ───────────────────────────────────────────────────────────────────


def compare_h5_sets(findings, label, listed: dict, live: dict) -> None:
    if set(listed) != set(live):
        findings.add(f"{label}.file_set", None, sorted(set(listed) ^ set(live)), "完全相同的 H5 集合")
    for name in sorted(set(listed) & set(live)):
        findings.eq(f"{label}.bytes", listed[name][0], live[name][0], name)
        findings.eq(f"{label}.sha256", listed[name][1], live[name][1], name)


def gate_lineage(args, ctx):
    findings = Findings()
    raw_dir = Path(args.raw_dir)
    inputs = read_json(args.input_manifest)
    plan, receipt = read_json(args.plan), read_json(args.receipt)
    manifest = read_json(args.episode_manifest)
    names = sorted(inputs["files"])
    findings.eq("input_manifest.count", inputs.get("count"), EXPECT["files"])
    findings.eq("files", len(names), EXPECT["files"])
    present = {name: raw_dir / name for name in names if (raw_dir / name).is_file()}
    for name in names:
        if name not in present:
            findings.add("h5_missing", name, None, str(raw_dir / name))
    digests = hash_many(present, args.workers)
    live = {name: (path.stat().st_size, digests[name]) for name, path in present.items()}
    compare_h5_sets(findings, "input_manifest", {k: (v["size"], v["sha256"]) for k, v in inputs["files"].items()}, live)
    for label, payload in (("plan", plan), ("receipt", receipt)):
        rows = payload["source_h5_files"]
        table = {item["name"]: (item["bytes"], item["sha256"]) for item in rows}
        findings.eq(f"{label}.source_h5_unique", len(table), len(rows))
        compare_h5_sets(findings, label, table, live)

    input_sha, manifest_file_sha = sha_file(args.input_manifest), sha_file(args.episode_manifest)
    canonical_sha = canonical_body_sha(manifest)
    findings.eq("input_manifest_sha256", input_sha, EXPECT["input_manifest_sha256"])
    findings.eq("episode_manifest_file_sha256", manifest_file_sha, EXPECT["episode_manifest_file_sha256"])
    findings.eq("episode_manifest_canonical_sha256", canonical_sha, EXPECT["episode_manifest_canonical_sha256"])
    findings.eq("episode_manifest.sha256", manifest.get("sha256"), canonical_sha)
    plan_sha = canonical_body_sha(plan)
    findings.eq("plan.sha256", plan.get("sha256"), plan_sha)
    findings.eq("plan_canonical_sha256", plan_sha, EXPECT["plan_canonical_sha256"])
    findings.eq("receipt.plan_sha256", receipt.get("plan_sha256"), plan_sha)
    for label, payload in (("plan", plan), ("receipt", receipt)):
        findings.eq(f"{label}.manifest_sha256", payload.get("manifest_sha256"), canonical_sha)
        findings.eq(f"{label}.manifest_file_sha256", payload.get("manifest_file_sha256"), manifest_file_sha)
        findings.eq(f"{label}.input_manifest_sha256", payload.get("input_manifest_sha256"), input_sha)
    episodes = manifest["episodes"]
    findings.eq("episodes", len(episodes), EXPECT["episodes"])
    findings.eq("exec_samples", manifest["totals"]["exec_samples"], EXPECT["exec_samples"])
    for episode in episodes:
        if episode["h5_file"] not in inputs["files"]:
            findings.add("episode.h5_file", episode["global_episode_idx"], episode["h5_file"], "input_manifest 内文件")

    frame_meta_path = Path(args.framesamp_store) / "meta/store_meta.json"
    frame_meta = read_json(frame_meta_path)
    for key, want in (("layout", EXPECT["framesamp_layout"]), ("status", "verified"), ("manifest_scope", "full"),
                      ("manifest_sha256", canonical_sha), ("num_exec_samples", EXPECT["exec_samples"])):
        findings.eq(f"framesamp.{key}", frame_meta.get(key), want)
    norm_sha = sha_file(args.norm_stats)
    findings.eq("norm_sha256", norm_sha, EXPECT["norm_sha256"])
    norm = read_json(args.norm_stats).get("norm_stats", {})
    for key in ("state", "actions"):
        for stat in ("mean", "std", "q01", "q99"):
            values = norm.get(key, {}).get(stat)
            if not isinstance(values, list) or not values or not all(math.isfinite(v) for v in values):
                findings.add("norm_stats", f"{key}.{stat}", values, "有限数列表")

    if getattr(args, "store", None):
        provenance = read_json(Path(args.store) / STORE_META)["provenance"]
        findings.eq("store.encoder_kind", provenance.get("encoder_kind"), KIND)
        findings.eq("store.encoder_ckpt", provenance.get("encoder_ckpt"), None)
        compare_h5_sets(findings, "store.source_h5",
                        {i["name"]: (i["bytes"], i["sha256"]) for i in provenance["source_h5"]}, live)
        findings.eq("store.input_manifest_sha256", provenance["manifests"]["input_manifest_sha256"], input_sha)
    for label, path in (("input_manifest", args.input_manifest), ("episode_manifest", args.episode_manifest),
                        ("plan", args.plan), ("receipt", args.receipt), ("framesamp_meta", frame_meta_path),
                        ("norm_stats", args.norm_stats)):
        ctx.input(label, path)
    for name, (size, digest) in sorted(live.items()):
        ctx.inputs[f"h5/{name}"] = dict(path=str(raw_dir / name), sha256=digest, bytes=size)
    line = (f"DATA_LINEAGE=PASS encoder_kind={KIND} encoder_ckpt=none files={len(names)} mismatches={findings.total}")
    metrics = dict(encoder_kind=KIND, encoder_ckpt=None, parameter_count=0, files=len(names), mismatches=findings.total,
                   canonical_manifest_sha256=canonical_sha)
    return [RoleResult("DATA_LINEAGE", line, findings, metrics)]


# ── G1 恢复对拍 ───────────────────────────────────────────────────────────────


def window_table(findings, label, receipt) -> dict:
    table = {}
    for row in receipt["windows"]:
        ident = (row["key"], row["m"])
        if ident in table:
            findings.add(f"{label}.duplicate_window", f"{ident[0]}:{ident[1]}", None, None)
        table[ident] = row
    return table


def hash_segment_chunks(path, rows: list, sample_every: int) -> list:
    """顺序读整段：按收据 offset 切块独立重算 SHA。返回 (窗 m, 现场 SHA)。"""
    results = []
    by_offset = {row["output_offset"]: row for row in rows}
    with Path(path).open("rb") as stream:
        offset = 0
        while True:
            blob = stream.read(CHUNK_BYTES)
            if not blob:
                break
            row = by_offset.get(offset)
            if row is not None and row["m"] % sample_every == 0:
                results.append((row["m"], sha_bytes(blob) if len(blob) == CHUNK_BYTES else "short-read"))
            offset += len(blob)
    return results


def gate_restore(args, ctx):
    findings = Findings()
    plan, original, restored = read_json(args.plan), read_json(args.receipt), read_json(args.restored_receipt)
    latents = Path(args.latents)
    plan_sha = canonical_body_sha(plan)
    findings.eq("plan.sha256", plan.get("sha256"), plan_sha)
    findings.eq("plan_canonical_sha256", plan_sha, EXPECT["plan_canonical_sha256"])
    for label, receipt in (("receipt", original), ("restored", restored)):
        findings.eq(f"{label}.status", receipt.get("status"), "PASS")
        findings.eq(f"{label}.plan_sha256", receipt.get("plan_sha256"), plan_sha)
        findings.eq(f"{label}.published.repo", receipt["published"]["repo"], EXPECT["published_repo"])
        findings.eq(f"{label}.published.revision", receipt["published"]["revision"], EXPECT["published_revision"])
    for key in ("source_h5_files", "manifest_sha256", "manifest_file_sha256", "input_manifest_sha256", "vae", "preprocess"):
        findings.eq(f"restored.{key}", restored.get(key), original.get(key))
    findings.eq("restored.published.files", restored["published"]["files"], original["published"]["files"])
    findings.eq("restored.latents.metadata_sha256", sha_file(latents / "metadata.json"),
                restored["latents"]["metadata_sha256"])
    findings.eq("restored.latents.root", str(Path(restored["latents"]["root"]).resolve()), str(latents.resolve()))
    if args.published_root:
        published = {item["path"]: Path(args.published_root) / item["path"] for item in restored["published"]["files"]}
        digests = hash_many({k: v for k, v in published.items() if v.is_file()}, args.workers)
        for item in restored["published"]["files"]:
            findings.eq("published.sha256", digests.get(item["path"]), item["sha256"], item["path"])

    planned = {(row["key"], row["m"]): row for row in plan["windows"]}
    old, new = window_table(findings, "receipt", original), window_table(findings, "restored", restored)
    for label, table in (("receipt", old), ("restored", new)):
        if set(table) != set(planned):
            diff = sorted(set(table) ^ set(planned))
            findings.add(f"{label}.window_set", f"{diff[0][0]}:{diff[0][1]}", len(table), len(planned))
    input_mismatches = latent_mismatches = 0
    failed = []
    counts = dict(copy=0, fill=0)
    for ident in sorted(set(old) & set(new) & set(planned)):
        a, b, index = old[ident], new[ident], f"{ident[0]}:{ident[1]}"
        counts[b["route"]] = counts.get(b["route"], 0) + 1
        for key, want in planned[ident].items():
            findings.eq(f"restored.{key}", b.get(key), want, index)
        if a["input_frames_sha256"] != b["input_frames_sha256"]:
            input_mismatches += 1
            findings.add("input_frames_sha256", index, b["input_frames_sha256"], a["input_frames_sha256"])
        if a["latent_sha256"] != b["latent_sha256"]:
            latent_mismatches += 1
            failed.append(dict(window=index, route=b["route"], restored=b["latent_sha256"], original=a["latent_sha256"]))
            findings.add("latent_sha256", index, b["latent_sha256"], a["latent_sha256"])
    findings.eq("windows", len(new), EXPECT["source_windows"])
    findings.eq("copy", counts["copy"], EXPECT["source_copy"])
    findings.eq("fill", counts["fill"], EXPECT["source_fill"])

    # 收据只说它算过；这里对落盘字节独立重算，默认全核。
    every = 1 if args.latent_check == "full" else args.sample_every
    by_file = {}
    for row in new.values():
        by_file.setdefault(row["output_file"], []).append(row)
    declared = {item["path"]: item for item in restored["latents"]["files"]}
    actual = {path.name for path in latents.glob("*.bin")}
    if actual != set(by_file):
        findings.add("latent_file_set", None, sorted(actual ^ set(by_file))[:5], "与收据逐段相同")
    names = sorted(actual & set(by_file))

    def check_file(name):
        path = latents / name
        return name, path.stat().st_size, sha_file(path), hash_segment_chunks(path, by_file[name], every)

    live_hashed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for name, size, digest, chunks in pool.map(check_file, names):
            findings.eq("latent_file.bytes", size, len(by_file[name]) * CHUNK_BYTES, name)
            findings.eq("latent_file.sha256", digest, declared.get(name, {}).get("sha256"), name)
            table = {row["m"]: row for row in by_file[name]}
            for m, got in chunks:
                live_hashed += 1
                if got != table[m]["latent_sha256"]:
                    latent_mismatches += 1
                    index = f"{name}:{m}"
                    failed.append(dict(window=index, route=table[m]["route"], live=got, receipt=table[m]["latent_sha256"]))
                    findings.add("latent_bytes", index, got, table[m]["latent_sha256"])
    for label, path in (("plan", args.plan), ("receipt", args.receipt), ("restored_receipt", args.restored_receipt),
                        ("latents_metadata", latents / "metadata.json")):
        ctx.input(label, path)
    line = (f"WAN_RESTORE_EQ=PASS windows={len(new)} input_mismatches={input_mismatches} "
            f"latent_mismatches={latent_mismatches}")
    metrics = dict(windows=len(new), copy=counts["copy"], fill=counts["fill"], input_mismatches=input_mismatches,
                   latent_mismatches=latent_mismatches, latent_check=args.latent_check, sample_every=every,
                   latent_windows_rehashed=live_hashed, numeric_modes=restored["numeric_modes"],
                   numeric_modes_equal_original=same(restored["numeric_modes"], original["numeric_modes"]),
                   failed_windows=failed[:200])
    return [RoleResult("WAN_RESTORE_EQ", line, findings, metrics)]


# ── G2 窗口等同 ───────────────────────────────────────────────────────────────


def expected_window_rows(findings, plan, receipt, manifest):
    """只从 manifest 段长、计划段表和原收据重算 grid32 全部映射行（不含 pooled_sha256）；同时算计数。"""
    plan_segments = {item["key"]: item for item in plan["segments"]}
    planned = {(row["key"], row["m"]): row for row in plan["windows"]}
    table = {(row["key"], row["m"]): row for row in receipt["windows"]}
    rows, seen_keys, seen_source = [], set(), 0
    counts = dict(copy=0, fill=0, demo_tail=0, exec_fill=0)
    max_visible = 0
    episodes = sorted(manifest["episodes"], key=lambda e: e["global_episode_idx"])
    if [e["global_episode_idx"] for e in episodes] != list(range(len(episodes))):
        findings.add("manifest.global_episode_idx", None, [e["global_episode_idx"] for e in episodes][:5], "连续编号")
    for episode in episodes:
        task = task_of_h5(episode["h5_file"])
        visible = 0
        for segment, start, length, min_real in segment_spans(episode):
            key = f"{task}_ep{episode['raw_ep_idx']}_{segment}"
            target_grid = num_grid(length, min_real, 32)
            visible += target_grid
            source_grid = num_grid(length, min_real, 16)
            item = plan_segments.get(key)
            if source_grid == 0:
                if item is not None:
                    findings.add("plan.segment_extra", key, item.get("num_grid"), None)
                continue
            if item is None:
                findings.add("plan.segment_missing", key, None, source_grid)
                continue
            seen_keys.add(key)
            for name, want in (("seg_start", start), ("seg_len", length), ("num_grid", source_grid),
                               ("min_real", min_real), ("segment", segment), ("g", episode["global_episode_idx"]),
                               ("h5_file", episode["h5_file"]), ("raw_ep_idx", episode["raw_ep_idx"])):
                findings.eq(f"plan.segment.{name}", item.get(name), want, key)
            for source_m in range(source_grid):
                seen_source += 1
                source = table.get((key, source_m))
                index = f"{key}:{source_m}"
                if source is None:
                    findings.add("receipt.window_missing", index, None, "存在")
                    continue
                first = start + 16 * source_m
                real = min(33, length - 16 * source_m)
                pad = 33 - real
                last = first + real - 1
                frames = list(range(first, first + real)) + [last] * pad
                for name, want in (("start_global_frame", first), ("seg_offset", 16 * source_m), ("real_frames", real),
                                   ("pad_frames", pad), ("pad_source_frame", last if pad else None),
                                   ("frame_ids", frames), ("segment_kind", segment), ("h5_file", episode["h5_file"]),
                                   ("raw_ep_idx", episode["raw_ep_idx"])):
                    findings.eq(f"receipt.{name}", source.get(name), want, index)
                plan_row = planned.get((key, source_m))
                if plan_row is None or plan_row.get("route") != source.get("route"):
                    findings.add("plan.route", index, plan_row and plan_row.get("route"), source.get("route"))
                if source_m % 2:
                    continue
                if last >= start + length:
                    findings.add("segment_boundary", index, last, start + length - 1)
                route = source["route"]
                counts[route] = counts.get(route, 0) + 1
                counts["demo_tail"] += int(pad > 0)
                counts["exec_fill"] += int(segment == "exec" and route == "fill")
                if pad and segment != "demo":
                    findings.add("exec_pad", index, pad, 0)
                rows.append(dict(row=len(rows), key=key, segment=segment, m=source_m // 2, source_m=source_m, route=route,
                                 start_frame=first, real_frames=real, pad_frames=pad,
                                 input_frames_sha256=source["input_frames_sha256"],
                                 raw_latent_sha256=source["latent_sha256"]))
            if num_grid(length, min_real, 32) != (source_grid + 1) // 2:
                findings.add("grid32_vs_even", key, (source_grid + 1) // 2, num_grid(length, min_real, 32))
        max_visible = max(max_visible, visible)
    if set(plan_segments) != seen_keys:
        findings.add("plan.segment_set", None, sorted(set(plan_segments) ^ seen_keys)[:5], "与 manifest 重算相同")
    if seen_source != len(table):
        findings.add("receipt.window_count", None, len(table), seen_source)
    return rows, counts, max_visible


def compare_window_map(findings, window_map, want_rows, counts) -> None:
    for key, want in (("schema", "wan-timepool-window-map-v1"), ("layout", LAYOUT), ("source_grid_stride", 16),
                      ("target_grid_stride", 32),
                      ("counts", dict(windows=len(want_rows), copy=counts["copy"], fill=counts["fill"]))):
        findings.eq(f"window_map.{key}", window_map.get(key), want)
    rows = window_map["windows"]
    findings.eq("window_map.rows", len(rows), len(want_rows))
    for index, want in enumerate(want_rows):
        if index >= len(rows):
            findings.add("window_map.missing_row", index, None, want["key"])
            continue
        got = rows[index]
        for name, value in want.items():
            findings.eq(f"window_map.{name}", got.get(name), value, index)
        extra = set(got) - set(want) - {"pooled_sha256"}
        if extra:
            findings.add("window_map.extra_field", index, sorted(extra), None)
    for index in range(len(want_rows), len(rows)):
        findings.add("window_map.extra_row", index, rows[index].get("key"), None)


def check_index(findings, index_payload, manifest) -> None:
    """按 manifest 重算 grid32 row_base / num_grid / seg_len；demo 与 exec 边界在这里钉死。"""
    entries = index_payload["entries"]
    episodes = sorted(manifest["episodes"], key=lambda e: e["global_episode_idx"])
    findings.eq("index.entries", len(entries), len(episodes))
    row = 0
    for entry, episode in zip(entries, episodes, strict=False):
        g = episode["global_episode_idx"]
        for name in ("h5_file", "raw_ep_idx", "num_timesteps", "exec_start_idx"):
            findings.eq(f"index.{name}", entry.get(name), episode[name], g)
        findings.eq("index.g", entry.get("g"), g, g)
        for segment, _, length, min_real in segment_spans(episode):
            ng = num_grid(length, min_real, 32)
            info = entry[segment]
            findings.eq("index.num_grid", info.get("num_grid"), ng, f"{g}:{segment}")
            findings.eq("index.seg_len", info.get("seg_len"), length, f"{g}:{segment}")
            findings.eq("index.row_base", info.get("row_base"), row if ng else None, f"{g}:{segment}")
            row += ng
    findings.eq("index.layout", index_payload.get("layout"), LAYOUT)


def load_window_map(args):
    if getattr(args, "store", None):
        path = Path(args.store) / STORE_WINDOW_MAP
    elif args.window_map:
        path = Path(args.window_map)
    else:
        raise ValueError("必须给 --store 或 --window-map")
    return path, read_json(path)


def gate_window(args, ctx):
    findings = Findings()
    plan, receipt, manifest = read_json(args.plan), read_json(args.receipt), read_json(args.episode_manifest)
    map_path, window_map = load_window_map(args)
    want_rows, counts, max_visible = expected_window_rows(findings, plan, receipt, manifest)
    compare_window_map(findings, window_map, want_rows, counts)
    if getattr(args, "store", None):
        check_index(findings, read_json(Path(args.store) / STORE_INDEX), manifest)
    for name, got in (("windows", len(want_rows)), ("copy", counts["copy"]), ("fill", counts["fill"]),
                      ("demo_tail", counts["demo_tail"]), ("exec_fill", counts["exec_fill"]),
                      ("max_visible", max_visible)):
        findings.eq(name, got, EXPECT[name])
    for label, path in (("plan", args.plan), ("receipt", args.receipt), ("episode_manifest", args.episode_manifest),
                        ("window_map", map_path)):
        ctx.input(label, path)
    line = (f"WINDOW_EQ=PASS windows={len(want_rows)} copy={counts['copy']} fill={counts['fill']} "
            f"demo_tail={counts['demo_tail']} exec_fill={counts['exec_fill']} max_visible={max_visible}")
    return [RoleResult("WINDOW_EQ", line, findings, dict(windows=len(want_rows), max_visible=max_visible, **counts))]


# ── 独立池化参考（不 import wan_timepool）──────────────────────────────────────


def independent_pool_f32(z: np.ndarray) -> np.ndarray:
    """计划数值合同的显式实现：每一步都是 float32 逐元素加法，次序 T(1→8)、块内 W(0→3)、块内 H(0→3)。"""
    single = np.float32
    temporal = np.array(z[1], dtype=single, copy=True)
    for t in range(2, 9):
        temporal = np.add(temporal, z[t], dtype=single)
    temporal = np.divide(temporal, single(8), dtype=single)
    out = np.empty(1024, dtype=single)
    channels = np.arange(16)
    for i in range(8):
        for j in range(8):
            over_h = None
            for a in range(4):
                over_w = temporal[:, 4 * i + a, 4 * j].copy()
                for b in range(1, 4):
                    over_w = np.add(over_w, temporal[:, 4 * i + a, 4 * j + b], dtype=single)
                over_h = over_w if over_h is None else np.add(over_h, over_w, dtype=single)
            out[64 * channels + 8 * i + j] = np.divide(over_h, single(16), dtype=single)
    return out


def reference_pool_f64(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FP64 数学参考：返回 (F, S)，S 是误差界用的 (1/128)Σ|Z|。"""
    wide = np.asarray(z, dtype=np.float64)
    drop = wide[1:9]
    def blocks(values):
        return values.reshape(8, 16, 8, 4, 8, 4).sum(axis=(0, 3, 5)) / 128.0
    return blocks(drop).reshape(1024), blocks(np.abs(drop)).reshape(1024)


def pool_contract_inputs() -> dict:
    rng = np.random.default_rng(1106)
    shape = LATENT_SHAPE
    cases = {f"normal_{seed}": np.random.default_rng(seed).standard_normal(shape, dtype=np.float32) for seed in range(4)}
    wide = rng.standard_normal(shape, dtype=np.float32) * np.float32(10.0) ** rng.integers(-6, 7, shape).astype(np.float32)
    cases["wide_dynamic_range"] = wide.astype(np.float32)
    cases["constant"] = np.full(shape, 0.3, dtype=np.float32)
    return cases


def check_pool_contract(pool_fn) -> tuple[Findings, dict]:
    """被验对象 pool_fn 只作输入输出黑盒；全部期望由闭式编码或独立实现给出。"""
    findings = Findings()
    metrics = dict(max_error_over_bound=0.0, cases=0)

    def run(label, z):
        before = z.tobytes()
        out = pool_fn(z)
        metrics["cases"] += 1
        findings.eq("input_unchanged", z.tobytes() == before, True, label)
        findings.eq("shape", list(out.shape), [1, 1024], label)
        findings.eq("dtype", str(out.dtype), "float32", label)
        findings.eq("byteorder_native", out.dtype.isnative, True, label)
        findings.eq("c_contiguous", bool(out.flags["C_CONTIGUOUS"]), True, label)
        findings.eq("output_bytes", len(out.tobytes()), ROW_BYTES, label)
        if not np.isfinite(out).all():
            findings.add("nonfinite", label, int((~np.isfinite(out)).sum()), 0)
        return out.reshape(1024)

    for label, z in pool_contract_inputs().items():
        got = run(label, z)
        reference, magnitude = reference_pool_f64(z)
        bound = BOUND_GAMMA * magnitude + FP32_ABS_FLOOR
        error = np.abs(got.astype(np.float64) - reference)
        ratio = float(np.max(error / bound))
        metrics["max_error_over_bound"] = max(metrics["max_error_over_bound"], ratio)
        if ratio > 1.0:
            worst = int(np.argmax(error / bound))
            findings.add("fp64_bound", f"{label}[{worst}]", float(error[worst]), float(bound[worst]))
        fixed = independent_pool_f32(z)
        if got.tobytes() != fixed.tobytes():
            worst = int(np.argmax(got != fixed))
            findings.add("reduction_order_bits", f"{label}[{worst}]", float(got[worst]), float(fixed[worst]))

    # 首 latent 哨兵：极值只放在 T=0，输出必须与全零首帧逐位相同。
    base = np.random.default_rng(7).standard_normal(LATENT_SHAPE, dtype=np.float32)
    sentinel = base.copy()
    sentinel[0] = 1e30
    zeroed = base.copy()
    zeroed[0] = 0
    if run("first_latent_sentinel", sentinel).tobytes() != run("first_latent_zero", zeroed).tobytes():
        findings.add("drop_first", "sentinel", "输出随 T=0 变化", "与 T=0 无关")

    # 时间编码：Z[t]=t，均值必须恰为 (1+…+8)/8=4.5；首帧未丢会得到 4.0。
    ramp = np.broadcast_to(np.arange(9, dtype=np.float32)[:, None, None, None], LATENT_SHAPE).copy()
    got = run("temporal_ramp", ramp)
    if not np.all(got == np.float32(4.5)):
        findings.add("temporal_mean8", "ramp", float(got[0]), 4.5)

    # 通道/坐标编码：每块内同值，输出位置 64c+8i+j 必须恰为 c·1000+i·10+j。抓 H/W 颠倒、通道混合、展平次序、额外归一化。
    c_axis = np.arange(16)[None, :, None, None]
    h_axis = (np.arange(32) // 4)[None, None, :, None]
    w_axis = (np.arange(32) // 4)[None, None, None, :]
    encoded = np.broadcast_to((c_axis * 1000 + h_axis * 10 + w_axis).astype(np.float32), LATENT_SHAPE).copy()
    got = run("channel_coord_encoding", encoded)
    want = np.array([c * 1000 + i * 10 + j for c in range(16) for i in range(8) for j in range(8)], dtype=np.float32)
    if got.tobytes() != want.tobytes():
        worst = int(np.argmax(got != want))
        findings.add("flatten_CHW", worst, float(got[worst]), float(want[worst]))

    # 脉冲：单元素 128 只影响其所在块，且恰为 1；非重叠 4×4 块的边界由此钉死。
    for t, c, h, w in ((1, 0, 0, 0), (8, 15, 31, 31), (3, 7, 5, 22), (5, 2, 28, 3)):
        impulse = np.zeros(LATENT_SHAPE, dtype=np.float32)
        impulse[t, c, h, w] = 128
        got = run(f"impulse_{t}_{c}_{h}_{w}", impulse)
        want = np.zeros(1024, dtype=np.float32)
        want[64 * c + 8 * (h // 4) + w // 4] = 1
        if got.tobytes() != want.tobytes():
            findings.add("spatial_block4", f"{t},{c},{h},{w}", np.flatnonzero(got).tolist()[:4],
                         int(np.flatnonzero(want)[0]))

    # 拒绝：错形状、错 dtype、错字节序、非有限、非 ndarray 都必须报错，且不做隐式转换。
    bad = {"wrong_shape": np.zeros((8, 16, 32, 32), np.float32), "float64": np.zeros(LATENT_SHAPE, np.float64),
           "big_endian": np.zeros(LATENT_SHAPE, ">f4"), "list": [[0.0]]}
    with_nan = np.zeros(LATENT_SHAPE, np.float32)
    with_nan[2, 3, 4, 5] = np.nan
    with_inf = np.zeros(LATENT_SHAPE, np.float32)
    with_inf[0, 0, 0, 0] = np.inf
    bad.update(nan=with_nan, inf_in_dropped_latent=with_inf)
    for label, value in bad.items():
        try:
            pool_fn(value)
        except (TypeError, ValueError):
            continue
        findings.add("accepts_invalid_input", label, "未报错", "TypeError/ValueError")
    return findings, metrics


def check_pool_source_zero_params(source: Path, findings: Findings) -> dict:
    """源码级零参数：只允许标准库与 NumPy，不含任何深度框架或可训练层关键字。"""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    extra = imports - {"__future__", "hashlib", "pathlib", "types", "numpy"}
    if extra:
        findings.add("pool_imports", None, sorted(extra), "仅 __future__/hashlib/pathlib/types/numpy")
    return dict(imports=sorted(imports))


def gate_pool_contract(args, ctx):
    source = Path(args.pool_source)
    module = load_by_path(source, "_o16_pool_under_test")
    findings, metrics = check_pool_contract(module.pool_raw_wan_latent)
    metrics.update(check_pool_source_zero_params(source, findings))
    findings.eq("parameter_count", module.TIMEPOOL_PARAMETER_COUNT, 0)
    findings.eq("contract", dict(module.POOL_CONTRACT), POOL_CONTRACT_LITERAL)
    findings.eq("readout", module.TIMEPOOL_READOUT, READOUT)
    findings.eq("layout", module.TIMEPOOL_LAYOUT, LAYOUT)
    findings.eq("encoder_kind", module.TIMEPOOL_ENCODER_KIND, KIND)
    source_sha = sha_file(source)
    if getattr(args, "store", None):
        findings.eq("store.pool.source_sha256", read_json(Path(args.store) / STORE_META)["provenance"]["pool"]["source_sha256"],
                    source_sha)
    ctx.input("pool_source", source, source_sha)
    metrics.update(bound_gamma=BOUND_GAMMA, abs_floor=FP32_ABS_FLOOR, depth=FP32_POOL_DEPTH)
    return [RoleResult("POOL_CONTRACT", "POOL_CONTRACT=PASS shape=1x1024 dtype=float32 params=0 raw=1 drop_first=1",
                       findings, metrics)]


# ── G4 全表 ───────────────────────────────────────────────────────────────────


def check_store_identity(findings, store, args, table_sha, window_map_sha) -> None:
    meta = read_json(Path(store) / STORE_META)
    prov = meta["provenance"]
    findings.eq("store.schema", meta.get("schema"), 1)
    findings.eq("store.layout", meta.get("layout"), LAYOUT)
    findings.eq("store.table.row_shape", meta["tables"]["motion_token"]["row_shape"], [1, 1024])
    findings.eq("store.table.dtype", meta["tables"]["motion_token"]["dtype"], "float32")
    findings.eq("store.table.sha256", meta["tables"]["motion_token"]["sha256"], table_sha)
    for key, want in (("schema", "wan-timepool-provenance-v1"), ("layout", LAYOUT), ("readout", READOUT),
                      ("encoder_kind", KIND), ("encoder_ckpt", None), ("parameter_count", 0),
                      ("output_shape", [1, 1024])):
        findings.eq(f"provenance.{key}", prov.get(key), want)
    findings.eq("provenance.pool.contract", prov["pool"]["contract"], POOL_CONTRACT_LITERAL)
    findings.eq("provenance.pool.source_sha256", prov["pool"]["source_sha256"], sha_file(POOL_SOURCE))
    findings.eq("provenance.output.motion_table_sha256", prov["output"]["motion_table_sha256"], table_sha)
    findings.eq("provenance.output.motion_index_sha256", prov["output"]["motion_index_sha256"],
                sha_file(Path(store) / STORE_INDEX))
    findings.eq("provenance.window_map.sha256", prov["window_map"]["sha256"], window_map_sha)
    for key, path in (("plan_file_sha256", args.plan), ("import_receipt_file_sha256", args.receipt),
                      ("restored_receipt_file_sha256", args.restored_receipt)):
        findings.eq(f"provenance.wan_cache.{key}", prov["wan_cache"][key], sha_file(path))


def gate_pool_table(args, ctx):
    findings = Findings()
    store, latents = Path(args.store), Path(args.latents)
    plan, receipt, manifest = read_json(args.plan), read_json(args.receipt), read_json(args.episode_manifest)
    map_path, window_map = load_window_map(args)
    want_rows, counts, _ = expected_window_rows(findings, plan, receipt, manifest)
    compare_window_map(findings, window_map, want_rows, counts)
    check_index(findings, read_json(store / STORE_INDEX), manifest)
    table_path = store / MOTION_TABLE
    size = table_path.stat().st_size
    findings.eq("table.bytes", size, EXPECT["bytes"])
    findings.eq("table.rows", size // ROW_BYTES, EXPECT["windows"])
    rows = window_map["windows"]
    mismatches = nonfinite = 0
    with table_path.open("rb") as table:
        handles = {}
        try:
            for index, window in enumerate(rows):
                stored = table.read(ROW_BYTES)
                if len(stored) != ROW_BYTES:
                    findings.add("table.short_row", index, len(stored), ROW_BYTES)
                    mismatches += 1
                    continue
                values = np.frombuffer(stored, dtype="<f4")
                bad = int((~np.isfinite(values)).sum())
                if bad:
                    nonfinite += bad
                    findings.add("table.nonfinite", index, bad, 0)
                key = window["key"]
                handle = handles.get(key)
                if handle is None:
                    handle = handles[key] = (latents / f"{key}.bin").open("rb")
                handle.seek(window["source_m"] * CHUNK_BYTES)
                blob = handle.read(CHUNK_BYTES)
                if sha_bytes(blob) != window["raw_latent_sha256"]:
                    findings.add("raw_latent_sha256", f"{index}:{key}:{window['source_m']}", sha_bytes(blob),
                                 window["raw_latent_sha256"])
                    mismatches += 1
                    continue
                raw = np.frombuffer(blob, dtype="<f4").reshape(LATENT_SHAPE)
                want = independent_pool_f32(raw).astype("<f4").tobytes()
                if stored != want:
                    mismatches += 1
                    delta = np.flatnonzero(values != np.frombuffer(want, dtype="<f4"))
                    position = int(delta[0]) if len(delta) else -1
                    findings.add("table.row_bytes", f"{index}:{key}:{window['m']}[{position}]", sha_bytes(stored), sha_bytes(want))
                if sha_bytes(stored) != window["pooled_sha256"]:
                    findings.add("window_map.pooled_sha256", index, sha_bytes(stored), window["pooled_sha256"])
                if index % 8 == 0 and len(handles) > 64:
                    for key_open, stream in list(handles.items()):
                        if key_open != key:
                            stream.close()
                            del handles[key_open]
        finally:
            for stream in handles.values():
                stream.close()
        if table.read(1):
            findings.add("table.extra_bytes", None, size, len(rows) * ROW_BYTES)
    table_sha = sha_file(table_path)
    window_map_sha = sha_file(map_path)
    check_store_identity(findings, store, args, table_sha, window_map_sha)
    for label, path in (("plan", args.plan), ("receipt", args.receipt), ("restored_receipt", args.restored_receipt),
                        ("episode_manifest", args.episode_manifest), ("window_map", map_path),
                        ("store_meta", store / STORE_META), ("store_index", store / STORE_INDEX)):
        ctx.input(label, path)
    ctx.output("motion_table", table_path, table_sha)
    line = f"POOL_TABLE_EQ=PASS rows={len(rows)} mismatches={mismatches} nonfinite={nonfinite} bytes={size}"
    return [RoleResult("POOL_TABLE_EQ", line, findings, dict(rows=len(rows), mismatches=mismatches, nonfinite=nonfinite, bytes=size))]


# ── G5 参数恒等 ───────────────────────────────────────────────────────────────


def resolve_yaml(name) -> Path:
    path = Path(name)
    return path if path.is_absolute() else YAML_DIR / name


def build_param_inventory(yaml_path: Path) -> dict:
    """按 train.py::init_train_state 同口径建模型（eval_shape，不占内存）：冻结叶转 bf16，冻结标志从真实过滤器求。"""
    ensure_src()
    import jax
    import jax.numpy as jnp
    from flax import nnx, traverse_util
    from mme_vla_suite.training import config as config_lib
    from openpi.shared import nnx_utils

    base = config_lib.get_config(CONFIG_NAME)
    model_config = dataclasses.replace(base.model, history_config=str(yaml_path), use_history=True)
    model = nnx.eval_shape(model_config.create, jax.random.key(base.seed))
    state = nnx.state(model)

    def paths(tree):
        return {"/".join(map(str, key)) for key in traverse_util.flatten_dict(tree.to_pure_dict())}

    frozen = paths(state.filter(base.freeze_filter))
    trainable = paths(state.filter(base.trainable_filter))
    converted = nnx_utils.state_map(state, base.freeze_filter,
                                    lambda item: item.replace(jax.ShapeDtypeStruct(item.value.shape, jnp.bfloat16)))
    inventory = {}
    for key, leaf in traverse_util.flatten_dict(converted.to_pure_dict()).items():
        name = "/".join(map(str, key))
        if (name in frozen) == (name in trainable):
            raise ValueError(f"叶 {name} 不是恰好属于冻结或可训练之一")
        inventory[name] = dict(shape=list(leaf.shape), dtype=str(leaf.dtype), frozen=name in frozen,
                               numel=int(np.prod(leaf.shape, dtype=np.int64)))
    return inventory


def inventory_totals(inventory: dict) -> dict:
    motion = ("mem_encoder/motion_pos_proj/", "mem_encoder/motion_encoder_static/")
    return dict(total=sum(v["numel"] for v in inventory.values()),
                trainable=sum(v["numel"] for v in inventory.values() if not v["frozen"]),
                frozen=sum(v["numel"] for v in inventory.values() if v["frozen"]),
                leaves=len(inventory), trainable_leaves=sum(not v["frozen"] for v in inventory.values()),
                mem_encoder=sum(v["numel"] for k, v in inventory.items() if k.startswith("mem_encoder/")),
                motion_path=sum(v["numel"] for k, v in inventory.items() if k.startswith(motion)))


def gate_vla_param(args, ctx):
    findings = Findings()
    old_path, new_path = resolve_yaml(args.old_yaml), resolve_yaml(args.new_yaml)
    old, new = build_param_inventory(old_path), build_param_inventory(new_path)
    for name in sorted(set(old) | set(new)):
        if name not in new:
            findings.add("leaf_missing_in_new", name, None, old[name])
        elif name not in old:
            findings.add("leaf_extra_in_new", name, new[name], None)
        else:
            for field in ("shape", "dtype", "frozen", "numel"):
                findings.eq(f"leaf.{field}", new[name][field], old[name][field], name)
    totals_old, totals_new = inventory_totals(old), inventory_totals(new)
    for key in totals_new:
        findings.eq(f"new.{key}", totals_new[key], EXPECT[key])
        findings.eq(f"old.{key}", totals_old[key], EXPECT[key])
    ctx.input("old_yaml", old_path)
    ctx.input("new_yaml", new_path)
    line = (f"VLA_PARAM_EQ=PASS total={totals_new['total']} trainable={totals_new['trainable']} "
            f"frozen={totals_new['frozen']} leaves={totals_new['leaves']} trainable_leaves={totals_new['trainable_leaves']}")
    metrics = dict(totals=totals_new, old_totals=totals_old, inventory_sha256=sha_bytes(canonical(new)), inventory=new)
    return [RoleResult("VLA_PARAM_EQ", line, findings, metrics)]


def build_param_digests(yaml_path: Path, seed: int) -> dict:
    """同种子真实初始化（CPU，约 13 GB 峰值），逐叶写 dtype/shape/SHA 后立刻释放。"""
    ensure_src()
    import jax
    from flax import nnx, traverse_util
    from mme_vla_suite.training import config as config_lib

    base = config_lib.get_config(CONFIG_NAME)
    model_config = dataclasses.replace(base.model, history_config=str(yaml_path), use_history=True)
    model = model_config.create(jax.random.key(seed))
    digests = {}
    for key, leaf in traverse_util.flatten_dict(nnx.state(model).to_pure_dict()).items():
        array = np.asarray(leaf)
        digests["/".join(map(str, key))] = dict(shape=list(array.shape), dtype=str(array.dtype),
                                                sha256=sha_bytes(np.ascontiguousarray(array).tobytes()))
    return digests


def gate_vla_init(args, ctx):
    findings = Findings()
    old_path, new_path = resolve_yaml(args.old_yaml), resolve_yaml(args.new_yaml)
    old, new = build_param_digests(old_path, args.seed), build_param_digests(new_path, args.seed)
    findings.eq("leaves", len(new), len(old))
    for name in sorted(set(old) | set(new)):
        findings.eq("leaf_digest", new.get(name), old.get(name), name)
    ctx.input("old_yaml", old_path)
    ctx.input("new_yaml", new_path)
    line = f"VLA_INIT_EQ=PASS leaves={len(new)} mismatches={findings.total} seed={args.seed}"
    return [RoleResult("VLA_INIT_EQ", line, findings, dict(seed=args.seed, leaves=len(new), digest=sha_bytes(canonical(new))))]


# ── G5 配置差分 ───────────────────────────────────────────────────────────────

_MISSING = "<缺失>"


def diff_records(old, new, path=()) -> list:
    """本文件独立实现的递归差分，返回 (点分路径, 旧, 新)；键增减与类型变化都算差异。"""
    if type(old) is not type(new):
        return [(".".join(path), old, new)]
    if isinstance(old, dict):
        out = []
        for key in sorted(set(old) | set(new)):
            if key not in old:
                out.append((".".join((*path, key)), _MISSING, new[key]))
            elif key not in new:
                out.append((".".join((*path, key)), old[key], _MISSING))
            else:
                out.extend(diff_records(old[key], new[key], (*path, key)))
        return out
    if isinstance(old, list):
        if len(old) != len(new):
            return [(".".join(path), old, new)]
        out = []
        for index, (a, b) in enumerate(zip(old, new, strict=True)):
            out.extend(diff_records(a, b, (*path, str(index))))
        return out
    return [] if old == new else [(".".join(path), old, new)]


# 计划第 8 节：仅允许新 run 名、history YAML、motion readout/kind/store/provenance 及其来源记录变化。
ALLOWED_CONFIG_DIFF = frozenset({
    "train_config.fields.exp_name", "train_config.fields.model.fields.history_config", "derived.checkpoint_dir",
    "history_values.motion.readout", "history_values.motion.encoder_kind", "history_values.motion.store_path",
    "history_values.motion.source_run", "history_values.motion.checkpoint_sha256"})
SMOKE_EXTRA_DIFF = frozenset({"train_config.fields.num_train_steps"})


def check_config_diff(findings, old_record, new_record, *, steps) -> list:
    allowed = ALLOWED_CONFIG_DIFF | (SMOKE_EXTRA_DIFF if steps == 20 else frozenset())
    differences = diff_records(old_record, new_record)
    names = {item[0] for item in differences}
    for name, old, new in differences:
        if name not in allowed:
            findings.add("unauthorized_diff", name, new, old)
    for name in sorted(allowed - names):
        findings.add("expected_diff_absent", name, "两侧相同", "必须不同")
    motion = new_record["history_values"]["motion"]
    for key, want in (("readout", READOUT), ("encoder_kind", KIND), ("online_gpu", 1), ("source_shape", [1, 1024]),
                      ("num_tokens", 1), ("dim", 1024), ("budget", 96)):
        findings.eq(f"new.motion.{key}", motion.get(key), want)
    for key in ("source_run", "checkpoint_sha256"):
        if key in motion:
            findings.add(f"new.motion.{key}", None, motion[key], "不得存在")
    fields = new_record["train_config"]["fields"]
    schedule = fields["lr_schedule"]["fields"]
    for name, want in (("batch_size", 128), ("fsdp_devices", 8), ("num_workers", 16), ("seed", 42), ("ema_decay", 0.999),
                       ("log_interval", 100), ("save_interval", 5000), ("keep_period", 5000),
                       ("num_train_steps", steps)):
        findings.eq(f"new.{name}", fields.get(name), want)
    for name, want in (("warmup_steps", 5000), ("peak_lr", 5e-5), ("decay_steps", 60000), ("decay_lr", 5e-5)):
        findings.eq(f"new.lr.{name}", schedule.get(name), want)
    return differences


def runner_argv(store_root: Path, run: str, history: str, steps: int, library: str) -> list:
    """与 run_orig16_*_60k.sh 的 ARGS 同形，在此独立写一份，避免依赖子任务 D 的实现。"""
    v1 = store_root / "v1-store"
    argv = [CONFIG_NAME, "--exp-name", run, "--dataset-path", str(v1 / f"datasets/{library}/framesamp-4x4"),
            "--assets-base-dir", str(v1 / "train-assets"),
            "--data.assets.assets-dir", str(v1 / f"train-assets/mme_vla_suite/{library}"),
            "--data.assets.asset-id", "robomme", "--checkpoint-base-dir", str(v1 / "train-runs"),
            "--weight-loader.params-path", str(v1 / "models/openpi-assets/checkpoints/pi05_base/params"),
            "--model.use-history", "--model.history-config", history]
    if steps == 20:
        argv += ["--num-train-steps", "20"]
    if not (v1 / "secrets/wandb.env").is_file():
        argv += ["--no-wandb-enabled"]
    return argv


def build_complete_record(argv: list, store_root: Path) -> dict:
    """走真实 tyro CLI 与 complete_record；须在仓库根 cwd（get_history_config 用相对路径）。"""
    os.environ["OPENPI_DATA_HOME"] = str(store_root / "v1-store/models")
    ensure_src()
    sys.path.insert(0, str(REPO / "scripts/training"))
    import config_record
    from mme_vla_suite.training import config as config_lib
    previous_argv, previous_cwd = sys.argv, os.getcwd()
    try:
        os.chdir(REPO)
        sys.argv = ["train.py", *argv]
        return json.loads(json.dumps(config_record.complete_record(config_lib.cli()), allow_nan=False))
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)


def gate_train_config(args, ctx):
    findings = Findings()
    store_root = Path(args.store_root)
    old_record = read_json(args.old_start_record)["complete"]
    new_record = build_complete_record(runner_argv(store_root, args.new_run, args.new_yaml, args.steps, args.library), store_root)
    differences = check_config_diff(findings, old_record, new_record, steps=args.steps)
    ctx.input("old_start_record", args.old_start_record)
    ctx.input("new_yaml", resolve_yaml(args.new_yaml))
    return [RoleResult("TRAIN_CONFIG_EQ", "TRAIN_CONFIG_EQ=PASS", findings,
                       dict(differences=[dict(path=a, old=short(b), new=short(c)) for a, b, c in differences],
                            allowed=sorted(ALLOWED_CONFIG_DIFF), steps=args.steps,
                            new_record_sha256=sha_bytes(canonical(new_record))))]


# ── G5 固定样本 ───────────────────────────────────────────────────────────────


def diff_samples(a, b, path, skip, findings) -> None:
    """除 skip 外逐位比较；数组比 dtype/shape/字节，容器比结构，标量比类型和值。"""
    if path in skip:
        return
    if type(a) is not type(b):
        findings.add("sample.type", path, type(b).__name__, type(a).__name__)
    elif isinstance(a, dict):
        if set(a) != set(b):
            findings.add("sample.keys", path, sorted(map(str, set(a) ^ set(b))), "两侧相同")
        for key in sorted(set(a) & set(b), key=str):
            diff_samples(a[key], b[key], f"{path}.{key}" if path else str(key), skip, findings)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            findings.add("sample.length", path, len(b), len(a))
        for index, (x, y) in enumerate(zip(a, b, strict=False)):
            diff_samples(x, y, f"{path}[{index}]", skip, findings)
    elif isinstance(a, np.ndarray) or isinstance(a, np.generic):
        x, y = np.asarray(a), np.asarray(b)
        if x.dtype != y.dtype or x.shape != y.shape or np.ascontiguousarray(x).tobytes() != np.ascontiguousarray(y).tobytes():
            findings.add("sample.array", path, f"{y.dtype}{y.shape}:{sha_bytes(np.ascontiguousarray(y).tobytes())[:16]}",
                         f"{x.dtype}{x.shape}:{sha_bytes(np.ascontiguousarray(x).tobytes())[:16]}")
    elif hasattr(a, "__array__"):
        diff_samples(np.asarray(a), np.asarray(b), path, skip, findings)
    elif a != b:
        findings.add("sample.value", path, short(b), short(a))


def open_sample_source(yaml_path: Path, store_root: Path, library: str, motion_root: Path):
    """返回 (len, get(idx)->{"raw":真实 __getitem__, "transformed":真实归一化后})。"""
    ensure_src()
    from omegaconf import OmegaConf
    from mme_vla_suite.training import config as config_lib
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset
    from openpi.training.data_loader import transform_dataset

    history = OmegaConf.load(yaml_path)
    v1 = store_root / "v1-store"
    base = config_lib.get_config(CONFIG_NAME)
    base = dataclasses.replace(base, assets_base_dir=str(v1 / "train-assets"), data=dataclasses.replace(
        base.data, assets=dataclasses.replace(base.data.assets, assets_dir=str(v1 / f"train-assets/mme_vla_suite/{library}"),
                                              asset_id="robomme")))
    model = dataclasses.replace(base.model, history_config=history, use_history=True)
    data_config = base.data.create(base.assets_dirs, model)
    library_root = v1 / "datasets" / library
    dataset = FrameSampDataset(str(library_root / "framesamp-4x4"), data_config, history, model.action_horizon,
                               source_root=str(library_root / "source"),
                               manifest_path=str(library_root / "meta/episode_manifest.json"), motion_root=str(motion_root))
    transformed = transform_dataset(dataset, data_config, skip_norm_stats=False)
    return len(dataset), lambda idx: dict(raw=dataset[idx], transformed=transformed[idx])


def sample_indices(total: int, count: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    picks = {0, total - 1, *(int(x) for x in rng.integers(0, total, size=max(0, count - 2)))}
    return sorted(picks)


def gate_non_motion(args, ctx):
    findings = Findings()
    store_root = Path(args.store_root)
    old_yaml, new_yaml = resolve_yaml(args.old_yaml), resolve_yaml(args.new_yaml)
    library_root = store_root / "v1-store/datasets" / args.library
    old_total, old_get = open_sample_source(old_yaml, store_root, args.library, Path(args.old_store or library_root / "motion-stage2-1024"))
    new_total, new_get = open_sample_source(new_yaml, store_root, args.library, Path(args.new_store or library_root / "motion-wan-timepool1024"))
    findings.eq("dataset_len", new_total, old_total)
    indices = sample_indices(old_total, args.samples, args.seed)
    changed = checked_with_windows = 0
    visible = {}
    skip = {"raw.motion_emb", "transformed.motion_emb"}
    for idx in indices:
        local = Findings()
        old, new = old_get(idx), new_get(idx)
        diff_samples(old, new, "", skip, local)
        for item in local.items:
            findings.add(item["field"], f"idx={idx}:{item['index']}", item["got"], item["want"])
        findings.total += max(0, local.total - len(local.items))
        emb_old, emb_new = np.asarray(old["raw"]["motion_emb"]), np.asarray(new["raw"]["motion_emb"])
        findings.eq("motion_emb.dtype", str(emb_new.dtype), str(emb_old.dtype), idx)
        findings.eq("motion_emb.shape", list(emb_new.shape), list(emb_old.shape), idx)
        k = int(np.asarray(new["raw"]["motion_mask"]).sum())
        visible[k] = visible.get(k, 0) + 1
        for label, emb in (("old", emb_old), ("new", emb_new)):
            if np.any(emb[k:]):
                findings.add(f"motion_padding.{label}", idx, int(np.count_nonzero(emb[k:])), 0)
        if not np.isfinite(emb_new).all():
            findings.add("motion_emb.nonfinite", idx, int((~np.isfinite(emb_new)).sum()), 0)
        if k:
            checked_with_windows += 1
            changed += int(emb_old[:k].tobytes() != emb_new[:k].tobytes())
    # 全相同说明新旧没换到不同的特征表，“仅 motion 内容变化”的前提不成立。
    if checked_with_windows and changed != checked_with_windows:
        findings.add("motion_content_unchanged", None, changed, checked_with_windows)
    ctx.input("old_yaml", old_yaml)
    ctx.input("new_yaml", new_yaml)
    line = f"NON_MOTION_EQ=PASS mismatches={findings.total}"
    metrics = dict(samples=len(indices), seed=args.seed, indices_sha256=sha_bytes(canonical(indices)),
                   motion_content_changed=changed, samples_with_windows=checked_with_windows,
                   visible_histogram={str(k): v for k, v in sorted(visible.items())})
    return [RoleResult("NON_MOTION_EQ", line, findings, metrics)]


# ── G7 窗口选取（G6 也用）───────────────────────────────────────────────────────

ONLINE_CATEGORIES = ("demo_complete", "demo_tail", "exec_copy", "exec_fill")


def window_category(window) -> str:
    if window["segment"] == "demo":
        return "demo_tail" if window["pad_frames"] else "demo_complete"
    return "exec_fill" if window["route"] == "fill" else "exec_copy"


def task_of_key(key: str) -> str:
    match = re.fullmatch(r"(.+)_ep[0-9]+_(demo|exec)", key)
    if match is None:
        raise ValueError(f"段键不合法: {key}")
    return match.group(1)


def select_named_windows(windows: list, per_task: int) -> tuple[list, dict]:
    """每任务按类别取中位行；某类不存在时明确记入 missing，并从现有类别顺次补足 per_task。"""
    by_task = {}
    for window in windows:
        by_task.setdefault(task_of_key(window["key"]), {}).setdefault(window_category(window), []).append(window)
    selected, coverage = [], {}
    for task in sorted(by_task):
        pools = by_task[task]
        chosen, missing = [], []
        for category in ONLINE_CATEGORIES:
            rows = pools.get(category)
            if rows:
                chosen.append((category, rows[len(rows) // 2]))
            else:
                missing.append(category)
        used = {w["row"] for _, w in chosen}
        for category in ONLINE_CATEGORIES:
            rows = [w for w in pools.get(category, []) if w["row"] not in used]
            step = max(1, len(rows) // max(1, per_task))
            for window in rows[::step]:
                if len(chosen) >= per_task:
                    break
                chosen.append((category, window))
                used.add(window["row"])
        chosen = chosen[:per_task]
        coverage[task] = dict(rows={c: [w["row"] for k, w in chosen if k == c] for c in ONLINE_CATEGORIES}, missing=missing,
                              name=[f"{w['key']}:{w['m']}" for _, w in chosen])
        selected.extend(window for _, window in chosen)
    return sorted(selected, key=lambda w: w["row"]), coverage


class H5FrameSource:
    """直接读原始 H5（独立于 orig16_wan_cache 的读帧实现），帧序与补尾按映射行重建。"""

    def __init__(self, raw_dir):
        self.raw_dir, self.handles = Path(raw_dir), {}

    def read(self, window, episode):
        import h5py
        name = episode["h5_file"]
        handle = self.handles.get(name)
        if handle is None:
            handle = self.handles[name] = h5py.File(self.raw_dir / name, "r")
        first, real, pad = window["start_frame"], window["real_frames"], window["pad_frames"]
        group = handle[f"episode_{episode['raw_ep_idx']}"]
        frames = np.stack([group[f"timestep_{i}/obs/front_rgb"][()] for i in range(first, first + real)])
        if pad:
            frames = np.concatenate((frames, np.repeat(frames[-1:], pad, axis=0)))
        return np.ascontiguousarray(frames)

    def close(self):
        for handle in self.handles.values():
            handle.close()


def episode_by_key(manifest) -> dict:
    table = {}
    for episode in manifest["episodes"]:
        task = task_of_h5(episode["h5_file"])
        for segment in ("demo", "exec"):
            table[f"{task}_ep{episode['raw_ep_idx']}_{segment}"] = episode
    return table


def read_restored_latent(latents, window) -> np.ndarray:
    with (Path(latents) / f"{window['key']}.bin").open("rb") as stream:
        stream.seek(window["source_m"] * CHUNK_BYTES)
        blob = stream.read(CHUNK_BYTES)
    if sha_bytes(blob) != window["raw_latent_sha256"]:
        raise ValueError(f"恢复 latent 字节不符: {window['key']}:{window['source_m']}")
    return np.frombuffer(blob, dtype="<f4").reshape(LATENT_SHAPE).astype(np.float32)


def read_table_row(store, row) -> np.ndarray:
    with (Path(store) / MOTION_TABLE).open("rb") as stream:
        stream.seek(row * ROW_BYTES)
        return np.frombuffer(stream.read(ROW_BYTES), dtype="<f4").astype(np.float32)


# ── G6 同 latent ──────────────────────────────────────────────────────────────


def pool_shared_samples(args) -> list:
    samples = []
    for seed in range(8):
        samples.append((f"synthetic_{seed}", np.random.default_rng(seed).standard_normal(LATENT_SHAPE, dtype=np.float32), None))
    rng = np.random.default_rng(100)
    wide = rng.standard_normal(LATENT_SHAPE, dtype=np.float32) * np.float32(10.0) ** rng.integers(-6, 7, LATENT_SHAPE).astype(np.float32)
    samples.append(("wide_dynamic_range", wide.astype(np.float32), None))
    samples.append(("constant", np.full(LATENT_SHAPE, 0.3, dtype=np.float32), None))
    if args.store and args.latents:
        _, window_map = load_window_map(args)
        windows, _ = select_named_windows(window_map["windows"], EXPECT["online_per_task"])
        for window in windows:
            samples.append((f"real_{window['key']}:{window['m']}", read_restored_latent(args.latents, window), window))
    return samples


def gate_pool_shared(args, ctx):
    findings = Findings()
    offline_module = load_by_path(args.pool_source, "_o16_offline_pool")
    sidecar = load_by_path(args.sidecar_source, "_o16_sidecar_for_shared")
    findings.eq("pool_source_sha256", sha_file(sidecar.POOL_PATH), sha_file(args.pool_source))
    samples = pool_shared_samples(args)
    max_abs, compared = 0.0, 0
    offline_all = []
    for label, latent, window in samples:
        offline = offline_module.pool_raw_wan_latent(latent)
        online = sidecar.token_from_latent(latent)
        offline_all.append(offline.reshape(1024))
        compared += 1
        delta = float(np.max(np.abs(offline.reshape(1024).astype(np.float64) - online.astype(np.float64))))
        max_abs = max(max_abs, delta)
        if offline.tobytes() != online.tobytes():
            findings.add("bitwise", label, sha_bytes(online.tobytes()), sha_bytes(offline.tobytes()))
        if str(online.dtype) != "float32" or online.shape != (1024,):
            findings.add("sidecar_shape", label, f"{online.dtype}{online.shape}", "float32(1024,)")
        if window is not None and read_table_row(args.store, window["row"]).tobytes() != online.tobytes():
            findings.add("table_row", f"{label}:row{window['row']}", "表字节", "sidecar 输出")
    if args.sidecar_python:
        # 真实在线 sidecar 跑在 wan-stage2 环境的 NumPy 上；这里跨解释器再比一次同一批 latent。
        with tempfile.TemporaryDirectory(prefix="o16-pool-shared-") as directory:
            source = Path(directory) / "latents.npy"
            target = Path(directory) / "tokens.npy"
            np.save(source, np.stack([latent for _, latent, _ in samples]), allow_pickle=False)
            env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
            subprocess.run([str(args.sidecar_python), str(Path(__file__).resolve()), "_sidecar-tokens",
                            "--input", str(source), "--output", str(target)], check=True, env=env)
            tokens = np.load(target, allow_pickle=False)
        for (label, _, _), offline, token in zip(samples, offline_all, tokens, strict=True):
            delta = float(np.max(np.abs(offline.astype(np.float64) - token.astype(np.float64))))
            max_abs = max(max_abs, delta)
            if offline.tobytes() != token.tobytes():
                findings.add("bitwise_cross_env", label, sha_bytes(token.tobytes()), sha_bytes(offline.tobytes()))
    ctx.input("pool_source", args.pool_source)
    ctx.input("sidecar_source", args.sidecar_source)
    bitwise = int(not findings)
    line = f"POOL_SHARED_EQ=PASS max_abs={0 if max_abs == 0 else max_abs} bitwise_equal={bitwise}"
    return [RoleResult("POOL_SHARED_EQ", line, findings,
                       dict(samples=compared, max_abs=max_abs, bitwise_equal=bitwise, cross_env=bool(args.sidecar_python)))]


# ── G7 真实在线 ───────────────────────────────────────────────────────────────


class RealOnlineBackend:
    """真实路径：本解释器（wan-stage2）内 load_online_stack 直编，另走 MotionEncoderClient 的 sidecar IPC。"""

    def __init__(self, args, provenance):
        ensure_src()
        self.sidecar = load_by_path(SIDECAR_SOURCE, "_o16_sidecar_online")
        stack = self.sidecar.load_online_stack(provenance["wan"]["vae_state_sha256"])
        self.adapter, self.vae, self.vinfo, self.device = stack
        from mme_vla_suite.policies.motion_client import MotionEncoderClient
        self.client = MotionEncoderClient(online_gpu=args.online_gpu, encoder_kind=KIND, store_provenance=provenance,
                                          v1_store=args.v1_store, stage2_python=sys.executable)

    def encode(self, window):
        return self.sidecar.encode_raw_latent(self.adapter, self.vae, window, self.device)

    def token(self, latent):
        return self.sidecar.token_from_latent(latent)

    def ipc(self, window, start):
        return self.client(window, start)

    def numerics(self):
        return dict(flags=self.adapter.pin_numerics(), handshake=self.client.provenance)

    def close(self):
        self.client.close()


def error_stats(pairs: list) -> dict:
    """pairs: [(在线, 参考)]。分路线记录最大误差、相对 L2、余弦与逐元素绝对误差分位数。"""
    if not pairs:
        return dict(windows=0)
    max_abs, rel, cos, errors = 0.0, [], [], []
    for online, reference in pairs:
        a, b = online.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
        delta = a - b
        errors.append(np.abs(delta))
        max_abs = max(max_abs, float(np.max(np.abs(delta))))
        norm_b = float(np.linalg.norm(b))
        rel.append(float(np.linalg.norm(delta)) / norm_b if norm_b else float(np.linalg.norm(delta)))
        denom = norm_b * float(np.linalg.norm(a))
        cos.append(float(np.dot(a, b)) / denom if denom else 1.0)
    errors = np.concatenate(errors)
    quantiles = {f"p{q}": float(np.quantile(errors, q / 100)) for q in (50, 90, 99, 99.9)}
    return dict(windows=len(pairs), max_abs=max_abs, rel_l2_max=max(rel), rel_l2_mean=sum(rel) / len(rel),
                cosine_min=min(cos), cosine_mean=sum(cos) / len(cos), abs_quantiles=quantiles)


def gate_online(args, ctx):
    findings, numerics = Findings(), Findings()
    store = Path(args.store)
    meta = read_json(store / STORE_META)
    provenance = meta["provenance"]
    findings.eq("store.status", meta.get("status"), "verified")
    findings.eq("store.encoder_kind", provenance.get("encoder_kind"), KIND)
    window_map = read_json(store / STORE_WINDOW_MAP)
    restored = read_json(args.restored_receipt)
    episodes = episode_by_key(read_json(args.episode_manifest))
    selected, coverage = select_named_windows(window_map["windows"], EXPECT["online_per_task"])
    findings.eq("tasks", len(coverage), EXPECT["online_tasks"])
    findings.eq("windows", len(selected), EXPECT["online_windows"])
    source = args.frame_source_factory(args.raw_dir)
    backend = args.backend_factory(args, provenance)
    input_mismatches = cross_window_state = ipc_mismatches = 0
    pairs = dict(copy=dict(latent=[], pool=[]), fill=dict(latent=[], pool=[]))
    rows = []
    try:
        frames = {}
        for window in selected:
            index = f"{window['key']}:{window['m']}"
            frame = source.read(window, episodes[window["key"]])
            frames[window["row"]] = frame
            if frame.shape != (33, 256, 256, 3) or frame.dtype != np.uint8:
                findings.add("frames.shape", index, f"{frame.dtype}{frame.shape}", "uint8(33,256,256,3)")
            got = sha_bytes(np.ascontiguousarray(frame).tobytes())
            if got != window["input_frames_sha256"]:
                input_mismatches += 1
                findings.add("input_frames_sha256", index, got, window["input_frames_sha256"])
        for position, window in enumerate(selected):
            index = f"{window['key']}:{window['m']}"
            other = selected[(position + 1) % len(selected)]
            frame, frame_b = frames[window["row"]], frames[other["row"]]
            # A/B/A：同窗两次编码之间插入另一窗。若有跨窗缓存残留，两次 A 不逐位相同。
            first = backend.encode(frame)
            backend.encode(frame_b)
            second = backend.encode(frame)
            if np.asarray(first).tobytes() != np.asarray(second).tobytes():
                cross_window_state += 1
                findings.add("cross_window_state", index, sha_bytes(np.asarray(second).tobytes()), sha_bytes(np.asarray(first).tobytes()))
            if np.asarray(first).dtype != np.float32 or tuple(np.asarray(first).shape) != LATENT_SHAPE:
                findings.add("latent.shape", index, f"{np.asarray(first).dtype}{np.asarray(first).shape}", "float32(9,16,32,32)")
                continue
            if not np.isfinite(first).all():
                findings.add("latent.nonfinite", index, int((~np.isfinite(first)).sum()), 0)
                continue
            token = backend.token(first)
            # 真实 IPC 同样 A/B/A，并要求与同卡直编的池化逐位相同；同代码、同钉数值下差异属未解释差异。
            ipc_first = backend.ipc(frame, window["start_frame"])
            backend.ipc(frame_b, other["start_frame"])
            ipc_second = backend.ipc(frame, window["start_frame"])
            if ipc_first.tobytes() != ipc_second.tobytes():
                cross_window_state += 1
                findings.add("cross_window_state.ipc", index, sha_bytes(ipc_second.tobytes()), sha_bytes(ipc_first.tobytes()))
            if ipc_first.tobytes() != token.tobytes():
                ipc_mismatches += 1
                findings.add("ipc_vs_direct", index, sha_bytes(ipc_first.tobytes()), sha_bytes(token.tobytes()))
            stored_latent = read_restored_latent(args.latents, window)
            stored_token = read_table_row(store, window["row"])
            pairs[window["route"]]["latent"].append((np.asarray(first), stored_latent))
            pairs[window["route"]]["pool"].append((token, stored_token))
            rows.append(dict(row=window["row"], key=window["key"], m=window["m"], route=window["route"],
                             category=window_category(window), start_frame=window["start_frame"],
                             input_frames_sha256=window["input_frames_sha256"],
                             latent_max_abs=float(np.max(np.abs(np.asarray(first, np.float64) - stored_latent.astype(np.float64)))),
                             pool_max_abs=float(np.max(np.abs(token.astype(np.float64) - stored_token.astype(np.float64))))))
        online_info = backend.numerics()
    finally:
        backend.close()
        source.close()
    modes = restored["numeric_modes"]
    copy_tf32 = modes["copy"].get("cudnn_conv_tf32")
    fill_tf32 = modes["fill"].get("flags", {}).get("cudnn_allow_tf32")
    online_tf32 = online_info["flags"].get("cudnn_allow_tf32")
    numerics.eq("copy_tf32", copy_tf32, True)
    numerics.eq("fill_tf32", fill_tf32, False)
    numerics.eq("online_tf32", online_tf32, False)
    stats = {route: {level: error_stats(items) for level, items in levels.items()} for route, levels in pairs.items()}
    for label, path in (("restored_receipt", args.restored_receipt), ("episode_manifest", args.episode_manifest),
                        ("window_map", store / STORE_WINDOW_MAP)):
        ctx.input(label, path)
    missing = {task: item["missing"] for task, item in coverage.items() if item["missing"]}
    metrics = dict(windows=len(selected), input_mismatches=input_mismatches, cross_window_state=cross_window_state,
                   ipc_mismatches=ipc_mismatches, coverage=coverage, missing_categories=missing, rows=rows)
    numeric_metrics = dict(copy_tf32=copy_tf32, fill_tf32=fill_tf32, online_tf32=online_tf32, routes=stats,
                           numeric_modes=modes, online=online_info, missing_categories=missing, rows=rows,
                           note="旧 copy 的 TF32=true、fill 与在线为 false；实数误差如实记录，不宣称逐位")
    window_line = (f"ONLINE_WINDOW_EQ=PASS windows={len(selected)} input_mismatches={input_mismatches} "
                   f"cross_window_state={cross_window_state}")
    numeric_line = (f"ONLINE_NUMERICS=RECORDED copy_tf32={str(copy_tf32).lower()} "
                    f"fill_online_tf32={str(bool(fill_tf32) or bool(online_tf32)).lower()}")
    return [RoleResult("ONLINE_WINDOW_EQ", window_line, findings, metrics),
            RoleResult("ONLINE_NUMERICS", numeric_line, numerics, numeric_metrics)]


# ── G8 / G9 恢复 ──────────────────────────────────────────────────────────────


def check_pre_save_reference(findings, reference) -> None:
    """沿用 write_stage2_pre_save_reference 的路径与 schema（假设见报告）。"""
    for key, want in (("schema", "stage2-pre-save-reference-v1"), ("step", 19), ("state_step", 20), ("seed", 42),
                      ("weight_type", "ema")):
        findings.eq(f"reference.{key}", reference.get(key), want)
    inference = reference.get("inference", {})
    for key, want in (("parameter_dtype", "bfloat16"), ("num_steps", 10), ("rng", "jax.random.split(jax.random.key(seed))[1]")):
        findings.eq(f"reference.inference.{key}", inference.get(key), want)
    findings.eq("reference.inference.sharding.kind", inference.get("sharding", {}).get("kind"), "SingleDeviceSharding")
    findings.eq("reference.inference.sharding.batch_size", inference.get("sharding", {}).get("batch_size"), 1)


def check_snapshot_identity(findings, run_root: Path) -> dict:
    """真实 verify_timepool_snapshot 验全部 SHA；另外独立确认快照与解析配置都没有旧 encoder 依赖。"""
    ensure_src()
    from mme_vla_suite.policies.motion_client import verify_timepool_snapshot
    provenance = verify_timepool_snapshot(run_root)
    findings.eq("snapshot.encoder", provenance.get("encoder"), None)
    findings.eq("snapshot.vae", provenance.get("vae"), None)
    timepool = provenance["timepool"]
    findings.eq("snapshot.encoder_kind", timepool.get("encoder_kind"), KIND)
    findings.eq("snapshot.encoder_ckpt", timepool.get("encoder_ckpt"), None)
    findings.eq("snapshot.parameter_count", timepool.get("parameter_count"), 0)
    if (run_root / "encoder_dependencies.json").exists():
        findings.add("snapshot.encoder_dependencies", str(run_root), "存在", "不得存在")
    text = (run_root / "history_config.resolved.yaml").read_text(encoding="utf-8")
    for forbidden in ("source_run", "checkpoint_sha256", "stage2_pool4", "external/motionjepa", "checkpoint_epoch"):
        if forbidden in text:
            findings.add("snapshot.yaml_encoder_reference", forbidden, "存在", "不得存在")
    return provenance


ENCODER_PATH_MARKERS = ("external/motionjepa", "stage2_sidecar", "checkpoint_epoch", "stage2-full144")


def worker_encoder_references(argv, pid) -> list:
    """sidecar 的 argv 与它打开的文件里都不得出现旧 encoder 路径。"""
    hits = [item for item in argv if any(marker in str(item) for marker in ENCODER_PATH_MARKERS)]
    fd_root = Path(f"/proc/{pid}/fd")
    if fd_root.is_dir():
        for entry in fd_root.iterdir():
            try:
                target = os.readlink(entry)
            except OSError:
                continue
            if any(marker in target for marker in ENCODER_PATH_MARKERS):
                hits.append(target)
    return hits


def restore_and_infer(args, checkpoint: Path, reference, reference_path) -> dict:
    """真实恢复（GPU）：严格参数树、EMA 字节、固定输入随机种子推理、真实 pack_buffer 在线。方法沿 check_stage2_checkpoint。"""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or args.online_gpu != 1:
        raise ValueError("恢复闸门要求父进程 CUDA_VISIBLE_DEVICES=0，显式 --online-gpu 1")
    if os.environ.get("JAX_PLATFORMS") == "cpu":
        raise ValueError("真实恢复闸门必须用 GPU")
    for name in ("MMEVLA_V1_STORE", "OPENPI_DATA_HOME"):
        value = os.environ.get(name)
        if not value or not Path(value).is_absolute() or not Path(value).is_dir():
            raise ValueError(f"恢复前必须显式设置现存绝对目录 {name}")
    ensure_src()
    helper = load_by_path(STAGE2_CHECK_SOURCE, "_o16_stage2_check_helper")
    root = checkpoint.parent
    online_record, online_arrays = helper.load_online_inputs(root)
    report = {}
    import jax
    report["compilation_cache"] = helper.configure_compilation_cache(jax.config)
    import jax.numpy as jnp
    from flax import nnx
    from mme_vla_suite.datastore.motion_adapter import motion_policy_spec
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    from mme_vla_suite.policies import policy_config
    from mme_vla_suite.training import config as config_lib
    from openpi.models import model as model_lib
    from openpi.shared import nnx_utils

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError(f"父进程必须只看到一个 GPU: {devices}")
    if reference is not None:
        saved = reference["inference"]
        jax.config.update("jax_default_matmul_precision", saved["jax_default_matmul_precision"])
        device = saved.get("device", {})
        if (device.get("platform") != "gpu" or device.get("id") != 0 or device.get("device_kind") != devices[0].device_kind
                or saved["sharding"].get("parameter_devices") != [str(devices[0])]
                or saved["sharding"].get("input_devices") != [str(devices[0])]):
            raise ValueError("保存前参考与恢复的单卡设备型号或首卡身份不一致")
    inventory = helper.gpu_inventory()
    if 0 not in inventory or 1 not in inventory:
        raise ValueError("缺物理 GPU0/1")
    train_config = config_lib.get_config(CONFIG_NAME)
    train_config = dataclasses.replace(train_config, data=dataclasses.replace(train_config.data,
        assets=dataclasses.replace(train_config.data.assets, assets_dir=str(checkpoint / "assets"))))
    resolved, enabled = policy_config._load_resolved_snapshot(root)
    if not enabled or resolved.motion.get("encoder_kind") != KIND or resolved.motion.get("readout") != READOUT:
        raise ValueError("恢复快照不是已开启的固定池化")
    spec = motion_policy_spec(resolved.motion)
    report["memory"] = dict(tokens=512 + spec.budget, readout=spec.readout, integration=resolved.integration_type,
                            dim=resolved.memory_token_dim)
    if (spec.window_budget != 96 or resolved.budget != 512 or resolved.token_per_image != 16 or resolved.num_views != 1
            or resolved.integration_type != "context" or resolved.memory_token_dim != 2048):
        raise ValueError("恢复快照的帧、motion 或融合合同不符")
    model_config = dataclasses.replace(train_config.model, history_config=resolved, use_history=True)
    train_config = dataclasses.replace(train_config, model=model_config)
    shape_model = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    expected = nnx_utils.state_map(nnx.state(shape_model), train_config.freeze_filter,
        lambda item: item.replace(jax.ShapeDtypeStruct(item.value.shape, jnp.bfloat16))).to_pure_dict()
    raw_params = model_lib.restore_params(checkpoint / "params", restore_type=np.ndarray, dtype=None)
    report["strict_restore"] = helper.verify_parameter_tree(expected, raw_params)
    summary, fp32 = helper.parameter_summary(raw_params)
    report["parameters"] = dict(sha256=summary["sha256"], leaves=len(summary["leaves"]))
    if reference is not None:
        helper.verify_summary(reference.get("parameters"), summary, "完整 EMA")
        helper.verify_summary(reference.get("fp32_ema"), fp32, "原始 fp32 EMA")
        if not fp32["leaves"]:
            raise ValueError("保存前参考没有原始 fp32 EMA 叶")
        report["ema_mismatch"] = 0
    del raw_params, shape_model, expected, summary
    policy = None
    try:
        policy = policy_config.create_trained_policy(train_config, checkpoint, seed=42, sample_kwargs={"num_steps": 10},
                                                     motion_online_gpu=args.online_gpu)
        client = policy._motion_client
        report["worker_encoder_references"] = worker_encoder_references(client.argv, client.worker_pid)
        report["sidecar_argv"] = [str(item) for item in client.argv]
        state = jax.tree.map(lambda value: jax.device_put(value, devices[0]), nnx.state(policy._model))
        leaves = jax.tree.leaves(state)
        if any(str(value.dtype) != "bfloat16" for value in leaves):
            raise ValueError("恢复 policy 的实际参数精度不是统一 bfloat16")
        if any(not isinstance(value.sharding, jax.sharding.SingleDeviceSharding) or value.devices() != {devices[0]}
               for value in leaves):
            raise ValueError("恢复 policy 参数不是实际单卡 SingleDeviceSharding")
        del leaves
        nnx.update(policy._model, state)
        policy._sample_actions = nnx_utils.module_jit(policy._model.sample_actions)
        policy._vision_encode = nnx_utils.module_jit(policy._model.vision_encode)
        del state
        if reference is not None:
            reference_root = Path(reference_path).resolve().parent
            inputs = helper.load_reference_tree(reference_root, reference["inputs"])
            wanted = helper.load_reference_tree(reference_root, reference["outputs"])
            observation = HistAugObservation.from_dict(jax.tree.map(lambda value: jax.device_put(value, devices[0]), inputs))
            output = policy._sample_actions(jax.random.split(jax.random.key(reference["seed"]))[1], observation, num_steps=10)
            jax.block_until_ready(output)
            report["reference_output_max_abs"] = helper.compare_outputs(wanted, {"actions": np.asarray(output)})
        report["online"] = helper.run_online(policy, online_record, online_arrays, args.online_gpu, inventory)
    finally:
        if policy is not None and policy._motion_client is not None:
            policy._motion_client.close()
    return report


def check_restore_report(findings, report, *, with_reference) -> None:
    strict = report.get("strict_restore", {})
    for key in ("missing", "extra", "shape_mismatch", "dtype_mismatch"):
        findings.eq(f"strict_restore.{key}", strict.get(key), 0)
    findings.eq("strict_restore.leaves", strict.get("leaves"), EXPECT["leaves"])
    if with_reference:
        findings.eq("ema_mismatch", report.get("ema_mismatch"), 0)
        output_abs = report.get("reference_output_max_abs")
        if output_abs != 0:
            findings.add("reference_output_max_abs", None, output_abs, 0)
    findings.eq("online_infer", report.get("online", {}).get("online_infer"), 1)
    if report.get("worker_encoder_references"):
        findings.add("worker_encoder_references", None, report["worker_encoder_references"], [])
    provenance = report.get("online", {}).get("sidecar_provenance", {})
    findings.eq("sidecar.encoder_kind", provenance.get("encoder_kind"), KIND)
    for key in provenance:
        if key != "encoder_kind" and re.search(r"(^|_)(encoder|ckpt|checkpoint)(_|$)", key):
            findings.add("sidecar.encoder_field", key, provenance[key], None)


def gate_save_load(args, ctx):
    findings = Findings()
    checkpoint = Path(args.checkpoint).absolute()
    findings.eq("checkpoint_step", checkpoint.name, "19")
    root = checkpoint.parent
    reference = read_json(args.reference)
    check_pre_save_reference(findings, reference)
    check_snapshot_identity(findings, root)
    report = restore_and_infer(args, checkpoint, reference, args.reference)
    check_restore_report(findings, report, with_reference=True)
    ctx.input("reference", args.reference)
    ctx.input("snapshot_ready", root / "motion_snapshot.ready.json")
    line = f"SAVE_LOAD=PASS checkpoint_step={checkpoint.name} state_step={reference.get('state_step')}"
    return [RoleResult("SAVE_LOAD", line, findings, dict(checkpoint=str(checkpoint), report=report))]


def check_final_record(findings, start, final, wait_done, *, run, train_head) -> None:
    steps, last = EXPECT["final_state_step"], EXPECT["final_checkpoint_step"]
    for label, record in (("start", start), ("final", final)):
        findings.eq(f"{label}.run_name", record.get("run_name"), run)
        findings.eq(f"{label}.head", record.get("head"), train_head)
        findings.eq(f"{label}.num_train_steps", record.get("num_train_steps"), steps)
    findings.eq("final.run_uuid", final.get("run_uuid"), start.get("run_uuid"))
    findings.eq("wait_done.run_uuid", wait_done.get("run_uuid"), start.get("run_uuid"))
    findings.eq("wait_done.head", wait_done.get("head"), train_head)
    findings.eq("wait_done.wait_until_finished", wait_done.get("wait_until_finished"), True)
    findings.eq("state_step", final.get("state_step"), steps)
    findings.eq("checkpoint_step", final.get("checkpoint_step"), last)
    findings.eq("loop_step", final.get("loop_step"), last)
    interval = final.get("log_interval")
    findings.eq("log_interval", interval, 100)
    first = (steps - 1) // 100 * 100 + 1
    tail = final.get("tail") or []
    findings.eq("tail.steps", [row.get("step") for row in tail], list(range(first, steps)))
    for row in tail:
        values = row.get("scalars", {})
        findings.eq("tail.scalars", sorted(values), sorted(TRAIN_SCALARS), row.get("step"))
        for name, item in values.items():
            if item.get("finite") is not True:
                findings.add("tail.nonfinite", f"{row.get('step')}:{name}", item, "finite")
    for name, value in (final.get("tail_means") or {}).items():
        if value is None or not math.isfinite(value):
            findings.add("tail_means.nonfinite", name, value, "有限")
    leaves = final.get("ema_leaves") or {}
    findings.eq("ema_leaves", len(leaves), EXPECT["leaves"])
    for name, item in leaves.items():
        if item.get("finite") is not True:
            findings.add("ema.nonfinite", name, item, "finite")


def check_log_exit(findings, log_path) -> int:
    nonempty_last, hits = None, 0
    with Path(log_path).open("rb") as stream:
        for line in stream:
            if line.strip():
                nonempty_last = line.strip()
            if NONFINITE_LOG.search(line):
                hits += 1
                findings.add("log.nonfinite_or_oom", line[:120].decode("utf-8", "replace"), None, None)
    findings.eq("log.last_line", (nonempty_last or b"").decode("utf-8", "replace"), "EXIT_CODE=0")
    return hits


def check_checkpoint_inventory(findings, run_root: Path) -> list:
    want = list(EXPECT["checkpoint_steps"])
    numeric = sorted(int(path.name) for path in run_root.iterdir() if path.is_dir() and path.name.isdigit())
    findings.eq("checkpoint_steps", numeric, want)
    for path in run_root.iterdir():
        if "orbax-checkpoint-tmp" in path.name:
            findings.add("uncommitted_checkpoint", path.name, "存在", "不得存在")
    for step in want:
        directory = run_root / str(step)
        for name in ("_CHECKPOINT_METADATA", "params/_METADATA", "params/manifest.ocdbt", "train_state"):
            if not (directory / name).exists():
                findings.add("checkpoint_entry_missing", f"{step}/{name}", None, "存在")
    return numeric


def gate_final(args, ctx):
    record_findings, inventory_findings, restore_findings = Findings(), Findings(), Findings()
    record_dir, run_root = Path(args.final_record_dir), Path(args.run_root)
    start = read_json(record_dir / "start.json")
    final = read_json(record_dir / "final.json")
    wait_done = read_json(record_dir / "checkpoint_wait_done.json")
    check_final_record(record_findings, start, final, wait_done, run=args.run, train_head=args.train_head)
    log_hits = check_log_exit(record_findings, args.log)
    steps = check_checkpoint_inventory(inventory_findings, run_root)
    last = run_root / str(EXPECT["final_checkpoint_step"])
    report = {}
    if not args.skip_restore:
        check_snapshot_identity(restore_findings, run_root)
        report = restore_and_infer(args, last, None, None)
        check_restore_report(restore_findings, report, with_reference=False)
    else:
        restore_findings.add("restore_skipped", str(last), "--skip-restore", "真实恢复")
    for label, path in (("final_record", record_dir / "final.json"), ("start_record", record_dir / "start.json"),
                        ("log", args.log)):
        ctx.input(label, path)
    return [RoleResult("TRAIN_FINAL_RECORD",
                       f"TRAIN_FINAL_RECORD=PASS state_step={final.get('state_step')} checkpoint_step={final.get('checkpoint_step')}",
                       record_findings, dict(log_exit_code=0, log_nonfinite_or_oom=log_hits, tail=len(final.get("tail") or []))),
            RoleResult("CKPT_INVENTORY", f"CKPT_INVENTORY=PASS steps={len(steps)}", inventory_findings, dict(steps=steps)),
            RoleResult("RESTORE_POOL", "RESTORE_POOL=PASS", restore_findings, dict(checkpoint=str(last), report=report))]


# ── 绑定与内部件 ──────────────────────────────────────────────────────────────


def cmd_bind(args) -> int:
    """按原证据为先期记录追加表/索引绑定：只新增字段，verdict 与其余内容一字不改；重绑到另一张表直接拒绝。"""
    binding = store_binding(args.store)
    roles = args.roles or list(BIND_ROLES)
    bound, skipped, failed = [], [], []
    for role in roles:
        path = Path(args.out_dir) / f"{role}.json"
        if not path.is_file():
            failed.append(f"{role}:缺记录")
            continue
        record = read_json(path)
        if record.get("status") != "PASS" or not str(record.get("verdict", "")).startswith(f"{role}=PASS"):
            failed.append(f"{role}:结论不是 PASS")
            continue
        present = {key: record.get(key) for key in binding if key in record}
        if present == binding:
            skipped.append(role)
            continue
        if present:
            failed.append(f"{role}:已绑定另一张表")
            continue
        original = sha_file(path)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps({**record, **binding, "bound_at": now_et(), "pre_bind_record_sha256": original},
                                        ensure_ascii=False, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        bound.append(role)
    if failed:
        print(f"GATE_BIND=FAIL field=record index={failed[0]} total={len(failed)}", flush=True)
        return 1
    print(f"GATE_BIND=PASS bound={len(bound)} already={len(skipped)} store_table_sha256={binding['store_table_sha256']} "
          f"motion_index_sha256={binding['motion_index_sha256']}", flush=True)
    return 0


def cmd_sidecar_tokens(args) -> int:
    """内部件：在 sidecar 解释器里把一批 latent 走 token_from_latent。"""
    sidecar = load_by_path(SIDECAR_SOURCE, "_o16_sidecar_tokens")
    latents = np.load(args.input, allow_pickle=False)
    np.save(args.output, np.stack([sidecar.token_from_latent(item) for item in latents]), allow_pickle=False)
    return 0


GATES = {"lineage": gate_lineage, "restore": gate_restore, "window": gate_window, "pool-contract": gate_pool_contract,
         "pool-table": gate_pool_table, "vla-param": gate_vla_param, "vla-init": gate_vla_init,
         "train-config": gate_train_config, "non-motion": gate_non_motion, "pool-shared": gate_pool_shared,
         "online": gate_online, "save-load": gate_save_load, "final": gate_final}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="cmd", required=True)

    def gate(name):
        sub = subs.add_parser(name)
        sub.add_argument("--out-dir", required=True, help="验证记录根，每个 ROLE 一份 JSON")
        sub.add_argument("--store", help="新池化库根；给定时记录写 store_table_sha256/motion_index_sha256 绑定")
        sub.add_argument("--expect-head", help="固定代码 HEAD，不符判 FAIL")
        return sub

    def flags(sub, *names):
        options = {"plan": "原 wan-cache-plan.json", "receipt": "原 wan-cache-import.json",
                   "restored-receipt": "wan-cache-import.restored.json", "latents": "恢复 latent 根",
                   "episode-manifest": "episode_manifest.json"}
        for name in names:
            sub.add_argument(f"--{name}", required=True, help=options.get(name))

    sub = gate("lineage")
    flags(sub, "input-manifest", "plan", "receipt", "episode-manifest", "raw-dir", "framesamp-store", "norm-stats")
    sub.add_argument("--workers", type=int, default=16)
    sub = gate("restore")
    flags(sub, "plan", "receipt", "restored-receipt", "latents")
    sub.add_argument("--published-root", help="给定时逐个重算发布文件 SHA")
    sub.add_argument("--latent-check", choices=("full", "sample"), default="full")
    sub.add_argument("--sample-every", type=int, default=16)
    sub.add_argument("--workers", type=int, default=16)
    sub = gate("window")
    flags(sub, "plan", "receipt", "episode-manifest")
    sub.add_argument("--window-map", help="未给 --store 时的映射文件（导出根 window_map.json）")
    sub = gate("pool-contract")
    sub.add_argument("--pool-source", default=str(POOL_SOURCE))
    sub = gate("pool-table")
    flags(sub, "plan", "receipt", "restored-receipt", "latents", "episode-manifest")
    sub = gate("vla-param")
    flags(sub, "old-yaml", "new-yaml")
    sub = gate("vla-init")
    flags(sub, "old-yaml", "new-yaml")
    sub.add_argument("--seed", type=int, default=42)
    sub = gate("train-config")
    flags(sub, "old-start-record", "new-yaml", "new-run", "store-root")
    sub.add_argument("--steps", type=int, choices=(20, 60000), default=60000)
    sub.add_argument("--library", default="16task-orig100-1600ep-9e109419")
    sub = gate("non-motion")
    flags(sub, "old-yaml", "new-yaml", "store-root")
    sub.add_argument("--old-store")
    sub.add_argument("--new-store")
    sub.add_argument("--samples", type=int, default=256)
    sub.add_argument("--seed", type=int, default=42)
    sub.add_argument("--library", default="16task-orig100-1600ep-9e109419")
    sub = gate("pool-shared")
    sub.add_argument("--pool-source", default=str(POOL_SOURCE))
    sub.add_argument("--sidecar-source", default=str(SIDECAR_SOURCE))
    sub.add_argument("--sidecar-python", help="wan-stage2 解释器；给定时跨环境再比一次")
    sub.add_argument("--latents")
    sub.add_argument("--window-map")
    sub = gate("online")
    flags(sub, "latents", "restored-receipt", "episode-manifest", "raw-dir", "v1-store")
    sub.add_argument("--online-gpu", required=True, type=int)
    sub = gate("save-load")
    flags(sub, "checkpoint", "reference")
    sub.add_argument("--online-gpu", required=True, type=int)
    sub = gate("final")
    flags(sub, "final-record-dir", "run-root", "log", "run", "train-head")
    sub.add_argument("--online-gpu", required=True, type=int)
    sub.add_argument("--skip-restore", action="store_true", help="只核记录与清单；RESTORE_POOL 判 FAIL，不能当完成")
    sub = subs.add_parser("bind")
    sub.add_argument("--out-dir", required=True)
    sub.add_argument("--store", required=True)
    sub.add_argument("--roles", nargs="*")
    sub = subs.add_parser("_sidecar-tokens")
    sub.add_argument("--input", required=True)
    sub.add_argument("--output", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "bind":
        return cmd_bind(args)
    if args.cmd == "_sidecar-tokens":
        return cmd_sidecar_tokens(args)
    if args.cmd == "online":
        args.frame_source_factory = H5FrameSource
        args.backend_factory = RealOnlineBackend
    return run_gate(args, args.cmd, GATES[args.cmd])


if __name__ == "__main__":
    raise SystemExit(main())
