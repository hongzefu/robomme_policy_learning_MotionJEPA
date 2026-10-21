#!/usr/bin/env python3
"""S5-A：从 HF bucket 选择性恢复历史 mixed8-200 VLA 的 Wan VAE latent 缓存并封口（``VAE_INPUT_READY``）。

对应 MotionJEPA 计划 ``1020-v10-encoder-to-pair4-modulation-8gpu-plan.md`` 第二部分「S5-A」与对抗检查
D1/D2/D4/D5/D8。本脚本只恢复**历史已验收**的 VAE 输入，不涉及任何新 encoder 身份、不生成 motion 表。

恢复对象（固定 37 个归档）
  bucket ``HongzeFu/robomme-16task-mixed8-200-trainset-20261016-v1``：
  * ``wan-latents_tars/wan-latents-00000.tar`` … ``-00035.tar``：36 包合计 75,811,553,280 B，
    成员 14986 个 = 4995 段 × (``.bin`` / ``.bin.sha256`` / ``.metadata.json``) + 总 ``metadata.json``；
  * ``oracle_tars/oracle-wan-mj-00035.tar``：1,412,976,640 B，只提取白名单 VAE 报告与采样记录
    （``vae_report.json``、``vae_report.shard<i>of8.json``、``sampled_windows.json``、
    ``sampled_windows.shard<i>of8.json``），旧 encoder 报告 / 表 / oracle bin 一律不提取。

外部期望值（全部来自打包当时留下的本地记录，不下载后现算自证）
  ``<export-dir>/SHA256SUMS.post.txt``          每个 tar 的 SHA256；
  ``<export-dir>/logs/pack_progress.jsonl``     每个 tar 的字节数、成员数、成员字节和（与上条交叉核对）；
  ``<export-dir>/logs/wan-latents_plan.json``、``oracle-wan-mj_plan.json``  成员名单与 ``arc_prefix``；
  ``<export-dir>/logs/parts/<tar>.sha256``      打包时逐成员 SHA256（成员名单必须与 plan 逐项同序一致）。
  另有脚本内钉死常量（``Pins``）：oracle 包 SHA/字节、36 包字节和、成员/段/窗数、总 metadata SHA、
  原 VAE state SHA、父库 manifest 两文件 SHA、规范化 manifest SHA、norm SHA。两方任一不一致即失败。

逐包流程（单流顺序执行）
  1. 下载到 staging：``ionice -c3 nice -n19 <hf> buckets cp hf://buckets/<bucket>/<rel> <tmp>``；
     首试 + ``--retries`` 次重试（默认 2）；每次下载后核字节数与 SHA，不符视为一次失败。
  2. 第一遍只读扫描 tar 头：只接受 REGTYPE/AREGTYPE 普通文件；拒绝软链接、硬链接、目录、设备、FIFO、
     绝对路径、``..``、空段、非规范路径、反斜杠、重复成员、名单外成员、缺成员、``arc_prefix`` 不符
     （latent 成员必须以 ``wan-latents/`` 开头，oracle 成员必须以 ``oracle/wan-mj/`` 开头）。
     扫描全部通过后才写任何文件。
  3. 第二遍按名单解包到 ``--out-dir``：目标的每一级父目录从 out-dir 起逐级 ``lstat``（不 resolve），
     拒绝链接或非目录；文件先写同目录 ``O_EXCL|O_NOFOLLOW`` 临时文件，边写边算 SHA，与 parts 记录一致后
     fsync + ``os.replace``；目标已存在但不是普通文件即失败。
  4. 在 ``restore_state.json`` 记该包完成，删除 staging tar（``--keep-staging`` 保留）。

断点续跑
  ``restore_state.json`` 中已完成的包不再下载；staging 里残留的未完成包若 SHA 正确直接复用。
  无论是否续跑，最后的封口阶段都**全量重算**每个已解包文件的 SHA（不设快速档），所以中断/重启不会把
  未核实的文件带进 receipt。out-dir 非空且没有本任务 ``restore_state.json``（任务名与输入摘要一致）时拒绝；
  已存在 ``restore_receipt.json`` 时拒绝改写（旧恢复记录不改写）。

封口校验（全部通过才写 receipt）
  * 文件集合：``wan-latents/`` 恰为 14986 个普通文件、``oracle/wan-mj/`` 恰为白名单，均 ``st_nlink==1``，
    无多余文件/目录/临时文件；每个文件 SHA = parts 记录。
  * 总 metadata 文件 SHA = ``bda20fd1…1061``；schema 3、``manifest_sha256`` = 规范化 manifest SHA、
    段数 4995、窗数 128342、补帧窗 1800；段集合及每段 num_grid/seg_len/起点/类型与父库 manifest 独立重算一致。
  * 闭环链：固定 tar SHA → 总 metadata（固定 SHA）记录的每段 ``sha256``/``bytes`` → 当前 ``.bin`` 实算
    SHA/字节（且 = num_grid × 589,824）→ ``.bin.sha256`` sidecar 内容 → 单段 metadata 的 sha/bytes/rows
    与总 metadata 逐项相同。只篡改 bin 并同步改 sidecar/单段 metadata 仍会在总 metadata 一环被拦下。
  * 原 VAE：每段 ``vae.vae_state_sha256`` = ``9980d252…e6d0``，batch=1、fp32、tf32/amp=off、latent_mode=mode，
    且身份字段跨段唯一；oracle ``vae_provenance.vae_state_sha256`` 也须相同。
  * 父库：``episode_manifest.json`` / ``input_manifest.json`` 文件 SHA、规范化 manifest SHA（记录值且现算）、
    norm 文件 SHA 均等于钉死值。
  * oracle 报告：``metadata_rows_checked``=128342、``windows``=``windows_encoded``=14480、mismatch 全 0、
    ``tested_metadata_sha256`` = 总 metadata SHA、``manifest_sha256`` = 规范化 SHA、``aggregated_from_shards``=8；
    8 份分片报告段集合不相交且并集 = 4995 段、行数/窗数之和与聚合一致；``sampled_windows.json`` 14480 个
    互不重复且都在总 metadata 中，其中补帧窗 1800 个 = 全部补帧窗。覆盖口径固定写为「全 128342 窗输入帧 /
    metadata 核对；14480 窗数值对拍（含全部 1800 个补帧窗）」，**不是全量数值对拍**。

restore_receipt.json（schema_version ``mixed8-vla-latent-restore/v1``，C2 消费；原子写）
  {
   "schema_version", "task", "status": "PASS",
   "bucket", "out_dir", "layout": {"latents": "wan-latents", "oracle": "oracle/wan-mj"},
   "objects": [{"rel", "kind": "latent"|"oracle", "sha256", "bytes", "members", "member_bytes",
                "extracted", "attempts"}...]  (37 项，36 latent 在前、oracle 在后),
   "counts": {"objects", "latent_tar_bytes", "latent_members", "oracle_members_extracted",
              "segments", "windows", "padded_windows"},
   "latents": {"total_metadata_sha256", "segment_bins_bytes", "files_list": "restore_files.sha256",
               "files_list_sha256"},
   "vae": {"vae_state_sha256", "identity": {VAE_IDENTITY_KEYS…}, "historical_module_sha256",
           "historical_encoder_src_sha256", "worker_count", "worker_git_commits", "worker_mj_commits"},
   "parent": {"lib", "episode_manifest_sha256", "input_manifest_sha256", "manifest_canonical_sha256",
              "raw_dir_recorded", "norm_stats", "norm_sha256"},
   "oracle": {"vae_report_sha256", "metadata_rows_checked", "numeric_windows", "numeric_padded_windows",
              "frame_mismatches", "metadata_mismatches", "aggregated_from_shards", "sample_spec",
              "tested_metadata_sha256", "orig_module_sha256", "coverage"},
   "expectation_sources": {<文件>: sha256…},
   "tool": {"script", "script_sha256", "wan_common_sha256", "python"},
   "excluded": [...], "new_encoder_identity": null,
   "started_at_utc", "finished_at_utc"
  }
  ``restore_files.sha256`` 为 ``sha256sum`` 格式的全部已解包文件清单（相对 out-dir）。

末行判定：``VAE_INPUT_READY=PASS objects=37 members=14986 segments=4995 windows=128342 padded=1800
oracle_rows=128342 oracle_numeric=14480 total_meta=<前 16> vae_state=<前 16> receipt=<前 16>``；
失败为 ``VAE_INPUT_READY=FAIL reason=<…>``（退出码 1）。``--dry-run`` 只解析本地期望值与父库，不下载、
不写盘，末行 ``RESTORE_DRYRUN=PASS objects=37 …``。

运行（主会话放 detached tmux；凭据只 source 不打印）：
  set -a; . v1-store/secrets/hf.env; set +a
  ionice -c3 nice -n19 uv run --no-project python scripts/dataset/restore_mixed8_vla_latents.py \\
      --out-dir <恢复目录>
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import posixpath
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import time

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "wan"))
import wan_common as wc  # noqa: E402  只用 manifest_sha256 / list_segments / 网格常量

POLICY_ROOT = pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA")
DEFAULT_EXPORT = POLICY_ROOT / "v1-store/exports/hf-mixed8-200-trainset"
DEFAULT_PARENT = POLICY_ROOT / "v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f"
DEFAULT_NORM = POLICY_ROOT / "v1-store/train-assets/mme_vla_suite/16task-mixed8-200-3200ep-e3c7564f/robomme/norm_stats.json"
DEFAULT_OUT = POLICY_ROOT / "v1-store/restore/mixed8-200-vla-latents-20261021-a"
DEFAULT_HF_CMD = 'uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf'

SCHEMA_VERSION = "mixed8-vla-latent-restore/v1"
TASK = "mixed8-200-vla-latents-restore-s5a"
STATE = "restore_state.json"
RECEIPT = "restore_receipt.json"
FILES_LIST = "restore_files.sha256"
LOCK = ".restore.lock"
CHUNK = 8 * 1024 * 1024
HEX = set("0123456789abcdef")
VAE_IDENTITY_KEYS = ("vae_id", "vae_state_sha256", "vae_dtype", "latent_mode", "batch", "tf32", "amp",
                     "torch", "cuda", "cudnn", "diffusers", "module_sha256", "encoder_src_sha256")


@dataclasses.dataclass(frozen=True)
class Pins:
    """钉死的外部期望值（计划第二部分 0.1 与 S5-A）。CLI 不提供覆盖入口，测试可传入合成值。"""
    bucket: str = "HongzeFu/robomme-16task-mixed8-200-trainset-20261016-v1"
    latent_tars: int = 36
    latent_tar_bytes: int = 75_811_553_280
    latent_members: int = 14986
    segments: int = 4995
    oracle_rel: str = "oracle_tars/oracle-wan-mj-00035.tar"
    oracle_sha256: str = "3b755bd279163164f8313391523a7fc1bbb3dc399dd2725d0504b1504b0215bd"
    oracle_bytes: int = 1_412_976_640
    total_metadata_sha256: str = "bda20fd101f57271a9b18db295679256cbf12d1d5c52e1fab80b936f7d9f1061"
    vae_state_sha256: str = "9980d252230c265cc2869466a74f85f5ee45b01ea9521bbb31159f90b75fe6d0"
    episode_manifest_sha256: str = "c6a063dfbfc6da404dd8f1d3ddc38e9ba6e25abca81417fe742872401e7f97df"
    input_manifest_sha256: str = "d7dd651edb8cb9652fa647a73a2d0d3c4be64d6876e1d7a186a5e8c75b02aafb"
    manifest_canonical_sha256: str = "d6a3ddb2b579b5cf50cd66fbefedd36478766e2b4cc07833a7efae29769742ad"
    norm_sha256: str = "36ee9acdea4c8383b50c7b07b45a740efb3fa073f6e60455977d5a6fb3204394"
    windows: int = 128342
    numeric_windows: int = 14480
    padded_windows: int = 1800
    oracle_shards: int = 8
    latent_prefix: str = "wan-latents"
    oracle_prefix: str = "oracle"
    oracle_sub: str = "wan-mj"

    def latent_rel(self, i: int) -> str:
        return f"wan-latents_tars/wan-latents-{i:05d}.tar"

    def oracle_whitelist(self) -> list[str]:
        base = f"{self.oracle_prefix}/{self.oracle_sub}"
        names = [f"{base}/vae_report.json", f"{base}/sampled_windows.json"]
        for i in range(self.oracle_shards):
            names += [f"{base}/vae_report.shard{i}of{self.oracle_shards}.json",
                      f"{base}/sampled_windows.shard{i}of{self.oracle_shards}.json"]
        return names


class RestoreError(RuntimeError):
    """任一校验失败；reason 进入 FAIL 判定行。"""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path: pathlib.Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=False) + "\n")


def is_sha(s) -> bool:
    return isinstance(s, str) and len(s) == 64 and set(s) <= HEX


# ══ 期望值 ═══════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class Obj:
    rel: str
    kind: str              # latent | oracle
    sha256: str
    bytes: int
    members: list[str]     # 完整 tar 成员名（含前缀），与 plan 同序
    member_sha: dict       # 成员名 → 打包时 sha256
    member_bytes: int
    extract: list[str]     # 需落盘的成员名


@dataclasses.dataclass
class Expect:
    objects: list[Obj]
    sources: dict          # 期望值来源文件 → sha256

    def digest(self) -> str:
        body = [(o.rel, o.sha256, o.bytes, len(o.members), sorted(o.extract)) for o in self.objects]
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    def all_extract(self) -> dict:
        return {n: o.member_sha[n] for o in self.objects for n in o.extract}


def _parse_sums(path: pathlib.Path) -> dict:
    out = {}
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not is_sha(parts[0]):
            raise RestoreError("sums_format", f"{path.name}:{ln}")
        name = parts[1].strip().lstrip("*")
        if name in out:
            raise RestoreError("sums_duplicate", name)
        out[name] = parts[0]
    return out


def _parse_parts(path: pathlib.Path) -> list[tuple[str, str]]:
    rows = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not is_sha(parts[0]):
            raise RestoreError("parts_format", f"{path.name}:{ln}")
        rows.append((parts[1].strip(), parts[0]))
    return rows


def _progress(path: pathlib.Path) -> dict:
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rel = d.get("rel")
        if not rel:
            continue
        keep = {k: d.get(k) for k in ("sha256", "bytes", "members", "member_bytes")}
        if rel in out and out[rel] != keep:
            raise RestoreError("progress_inconsistent", rel)
        out[rel] = keep
    return out


def load_expectations(export_dir: pathlib.Path, pins: Pins) -> Expect:
    logs = export_dir / "logs"
    src_files = {"SHA256SUMS.post.txt": export_dir / "SHA256SUMS.post.txt",
                 "logs/wan-latents_plan.json": logs / "wan-latents_plan.json",
                 "logs/oracle-wan-mj_plan.json": logs / "oracle-wan-mj_plan.json",
                 "logs/pack_progress.jsonl": logs / "pack_progress.jsonl"}
    for k, p in src_files.items():
        if not p.is_file():
            raise RestoreError("missing_expectation_source", k)
    sums = _parse_sums(src_files["SHA256SUMS.post.txt"])
    prog = _progress(src_files["logs/pack_progress.jsonl"])
    lat_plan = json.loads(src_files["logs/wan-latents_plan.json"].read_text(encoding="utf-8"))["shards"]
    ora_plan = json.loads(src_files["logs/oracle-wan-mj_plan.json"].read_text(encoding="utf-8"))["shards"]
    if len(lat_plan) != pins.latent_tars:
        raise RestoreError("plan_shard_count", f"{len(lat_plan)} != {pins.latent_tars}")
    ora = [s for s in ora_plan if s.get("rel") == pins.oracle_rel]
    if len(ora) != 1:
        raise RestoreError("oracle_plan_missing", pins.oracle_rel)
    sources = {k: sha256_file(p) for k, p in src_files.items()}
    objects: list[Obj] = []
    whitelist = pins.oracle_whitelist()
    shards = [(i, s, "latent") for i, s in enumerate(lat_plan)] + [(None, ora[0], "oracle")]
    for i, shard, kind in shards:
        rel = shard.get("rel")
        if kind == "latent" and rel != pins.latent_rel(i):
            raise RestoreError("plan_rel", f"{rel} != {pins.latent_rel(i)}")
        want_prefix = pins.latent_prefix if kind == "latent" else pins.oracle_prefix
        if shard.get("arc_prefix") != want_prefix or shard.get("kind") != "flat_shard":
            raise RestoreError("plan_arc_prefix", f"{rel}: {shard.get('arc_prefix')!r}")
        members = [f"{want_prefix}/{n}" for n in shard["names"]]
        if len(set(members)) != len(members):
            raise RestoreError("plan_duplicate_member", rel)
        if rel not in sums:
            raise RestoreError("sums_missing", rel)
        p = prog.get(rel)
        if p is None or p["sha256"] != sums[rel]:
            raise RestoreError("progress_sha_mismatch", rel)
        if p["members"] != len(members) or p["member_bytes"] != shard["bytes"] or not isinstance(p["bytes"], int):
            raise RestoreError("progress_plan_mismatch", rel)
        part_path = logs / "parts" / (pathlib.PurePosixPath(rel).name + ".sha256")
        if not part_path.is_file():
            raise RestoreError("parts_missing", rel)
        rows = _parse_parts(part_path)
        if [n for n, _ in rows] != members:
            raise RestoreError("parts_plan_mismatch", rel)
        sources[f"logs/parts/{part_path.name}"] = sha256_file(part_path)
        member_sha = dict((n, s) for n, s in rows)
        if kind == "oracle":
            if sums[rel] != pins.oracle_sha256 or p["bytes"] != pins.oracle_bytes:
                raise RestoreError("oracle_pin_mismatch", rel)
            if any(not m.startswith(f"{pins.oracle_prefix}/{pins.oracle_sub}/") for m in members):
                raise RestoreError("plan_arc_prefix", f"{rel} 含非 {pins.oracle_sub} 成员")
            missing = [w for w in whitelist if w not in member_sha]
            if missing:
                raise RestoreError("oracle_whitelist_missing", missing[0])
            extract = list(whitelist)
        else:
            extract = list(members)
        objects.append(Obj(rel, kind, sums[rel], int(p["bytes"]), members, member_sha, int(shard["bytes"]), extract))
    lat = [o for o in objects if o.kind == "latent"]
    if sum(o.bytes for o in lat) != pins.latent_tar_bytes:
        raise RestoreError("latent_tar_bytes", f"{sum(o.bytes for o in lat)} != {pins.latent_tar_bytes}")
    all_lat = [m for o in lat for m in o.members]
    if len(all_lat) != pins.latent_members or len(set(all_lat)) != len(all_lat):
        raise RestoreError("latent_member_count", f"{len(all_lat)} != {pins.latent_members}")
    meta_name = f"{pins.latent_prefix}/metadata.json"
    holders = [o for o in lat if meta_name in o.member_sha]
    if len(holders) != 1 or holders[0].member_sha[meta_name] != pins.total_metadata_sha256:
        raise RestoreError("total_metadata_pin", "parts 记录的总 metadata SHA 与钉死值不符")
    return Expect(objects, sources)


# ══ tar 安全检查与解包 ══════════════════════════════════════════════════════


def check_member_name(name: str, prefix: str) -> None:
    if not name or "\x00" in name or "\\" in name:
        raise RestoreError("tar_bad_name", repr(name))
    if name.startswith("/") or posixpath.isabs(name):
        raise RestoreError("tar_absolute_path", name)
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts) or posixpath.normpath(name) != name:
        raise RestoreError("tar_path_traversal", name)
    if not name.startswith(prefix + "/"):
        raise RestoreError("tar_arc_prefix", name)


def scan_tar(path: pathlib.Path, obj: Obj, pins: Pins) -> None:
    """第一遍：只读扫描全部头，任一异常即失败，此时尚未写任何文件。"""
    prefix = pins.latent_prefix if obj.kind == "latent" else f"{pins.oracle_prefix}/{pins.oracle_sub}"
    expected = set(obj.members)
    seen: set[str] = set()
    with tarfile.open(path, mode="r:") as tf:
        for ti in tf:
            name = ti.name
            if ti.issym() or ti.islnk():
                raise RestoreError("tar_link_member", name)
            if ti.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or ti.linkname:
                raise RestoreError("tar_non_regular", f"{name} type={ti.type!r}")
            check_member_name(name, prefix)
            if name in seen:
                raise RestoreError("tar_duplicate_member", name)
            if name not in expected:
                raise RestoreError("tar_unexpected_member", name)
            seen.add(name)
    if seen != expected:
        raise RestoreError("tar_missing_member", sorted(expected - seen)[0])


def check_root(out_dir: pathlib.Path) -> None:
    """out-dir 自身及各级祖先都不能是链接（逐级 lstat，不 resolve）。"""
    cur = pathlib.Path("/")
    for part in out_dir.parts[1:]:
        cur = cur / part
        try:
            st = os.lstat(cur)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(st.st_mode):
            raise RestoreError("out_dir_symlink", str(cur))
        if not stat.S_ISDIR(st.st_mode):
            raise RestoreError("out_dir_not_dir", str(cur))


def safe_parent(out_dir: pathlib.Path, rel: str) -> pathlib.Path:
    """从 out-dir 起逐级 lstat 目标父目录（未 resolve 路径），缺失则建，遇链接/非目录即失败。"""
    cur = out_dir
    st = os.lstat(cur)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise RestoreError("out_dir_symlink", str(cur))
    for part in rel.split("/")[:-1]:
        cur = cur / part
        try:
            st = os.lstat(cur)
        except FileNotFoundError:
            os.mkdir(cur, 0o755)
            st = os.lstat(cur)
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            raise RestoreError("dest_parent_link", str(cur))
    return cur


def extract_tar(path: pathlib.Path, obj: Obj, out_dir: pathlib.Path) -> int:
    want = set(obj.extract)
    n = 0
    with tarfile.open(path, mode="r:") as tf:
        for ti in tf:
            if ti.name not in want:
                continue
            parent = safe_parent(out_dir, ti.name)
            dest = parent / ti.name.rsplit("/", 1)[-1]
            try:
                st = os.lstat(dest)
                if not stat.S_ISREG(st.st_mode):
                    raise RestoreError("dest_not_regular", str(dest))
            except FileNotFoundError:
                pass
            tmp = dest.with_name(dest.name + f".tmp.{os.getpid()}")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
            h = hashlib.sha256()
            size = 0
            try:
                src = tf.extractfile(ti)
                with os.fdopen(fd, "wb") as fo:
                    while True:
                        b = src.read(CHUNK)
                        if not b:
                            break
                        h.update(b)
                        fo.write(b)
                        size += len(b)
                    fo.flush()
                    os.fsync(fo.fileno())
                if size != ti.size or h.hexdigest() != obj.member_sha[ti.name]:
                    raise RestoreError("member_sha_mismatch", ti.name)
                os.replace(tmp, dest)
            finally:
                if os.path.lexists(tmp):
                    os.unlink(tmp)
            n += 1
    if n != len(want):
        raise RestoreError("extract_count", f"{obj.rel}: {n} != {len(want)}")
    dfd = os.open(out_dir, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return n


# ══ 下载 ════════════════════════════════════════════════════════════════════


def verify_file(path: pathlib.Path, sha: str, size: int) -> bool:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISREG(st.st_mode) and st.st_size == size and sha256_file(path) == sha


def download(obj: Obj, staging: pathlib.Path, hf_cmd: list[str], bucket: str, retries: int,
             sleep_s: float, low_prio: bool) -> tuple[pathlib.Path, int]:
    dst = staging / pathlib.PurePosixPath(obj.rel).name
    if os.path.lexists(dst):
        if verify_file(dst, obj.sha256, obj.bytes):
            log(f"staging 已有且 SHA 正确，复用 {dst.name}")
            return dst, 0
        os.unlink(dst)
    prio = ["ionice", "-c3", "nice", "-n19"] if low_prio and shutil.which("ionice") and shutil.which("nice") else []
    attempts = 1 + retries
    for attempt in range(1, attempts + 1):
        tmp = dst.with_name(dst.name + ".part")
        if os.path.lexists(tmp):
            os.unlink(tmp)
        cmd = prio + hf_cmd + ["buckets", "cp", f"hf://buckets/{bucket}/{obj.rel}", str(tmp)]
        t0 = time.perf_counter()
        rc = subprocess.run(cmd).returncode
        ok = rc == 0 and verify_file(tmp, obj.sha256, obj.bytes)
        if ok:
            os.replace(tmp, dst)
            log(f"下载完成 {obj.rel} bytes={obj.bytes} attempt={attempt} {time.perf_counter() - t0:.0f}s")
            return dst, attempt
        log(f"DOWNLOAD_FAIL object={obj.rel} attempt={attempt}/{attempts} rc={rc}")
        if os.path.lexists(tmp):
            os.unlink(tmp)
        if attempt < attempts:
            time.sleep(sleep_s)
    raise RestoreError("download_retries_exhausted", f"{obj.rel} 首试 + 重试 {retries} 次")


# ══ 封口校验 ═══════════════════════════════════════════════════════════════


def _list_regular(d: pathlib.Path) -> dict:
    out = {}
    try:
        st = os.lstat(d)
    except FileNotFoundError:
        raise RestoreError("missing_dir", str(d))
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise RestoreError("dest_parent_link", str(d))
    with os.scandir(d) as it:
        for e in it:
            st = os.lstat(e.path)
            if not stat.S_ISREG(st.st_mode):
                raise RestoreError("unexpected_entry", e.path)
            if st.st_nlink != 1:
                raise RestoreError("hardlinked_file", e.path)
            out[e.name] = st.st_size
    return out


def verify_closure(out_dir: pathlib.Path, expect: Expect, pins: Pins, parent_lib: pathlib.Path,
                   norm_path: pathlib.Path) -> dict:
    lat_dir = out_dir / pins.latent_prefix
    ora_dir = out_dir / pins.oracle_prefix / pins.oracle_sub
    # 1. 顶层与目录集合
    allowed_top = {pins.latent_prefix, pins.oracle_prefix, STATE, LOCK}
    for e in os.scandir(out_dir):
        if e.name not in allowed_top:
            raise RestoreError("unexpected_entry", e.path)
    with os.scandir(out_dir / pins.oracle_prefix) as it:
        names = [e.name for e in it]
    if names != [pins.oracle_sub]:
        raise RestoreError("unexpected_entry", f"{pins.oracle_prefix}/ 下: {sorted(names)}")
    want = expect.all_extract()
    lat_files = _list_regular(lat_dir)
    ora_files = _list_regular(ora_dir)
    got = {f"{pins.latent_prefix}/{n}" for n in lat_files} | {f"{pins.oracle_prefix}/{pins.oracle_sub}/{n}" for n in ora_files}
    if got != set(want):
        miss, extra = sorted(set(want) - got), sorted(got - set(want))
        raise RestoreError("file_set_mismatch", f"缺 {miss[:3]} 多 {extra[:3]}")
    # 2. 全量重算每个文件 SHA，与打包时 parts 记录比
    actual: dict[str, str] = {}
    t0 = time.perf_counter()
    for i, name in enumerate(sorted(want)):
        actual[name] = sha256_file(out_dir / name)
        if actual[name] != want[name]:
            raise RestoreError("member_sha_mismatch", name)
        if (i + 1) % 2000 == 0:
            log(f"封口重算 SHA {i + 1}/{len(want)} ({time.perf_counter() - t0:.0f}s)")
    # 3. 总 metadata 固定 SHA
    total_meta_path = lat_dir / "metadata.json"
    if actual[f"{pins.latent_prefix}/metadata.json"] != pins.total_metadata_sha256:
        raise RestoreError("total_metadata_sha", actual[f"{pins.latent_prefix}/metadata.json"])
    total = json.loads(total_meta_path.read_text(encoding="utf-8"))
    # 4. 父库 manifest / norm
    ep_path, in_path = parent_lib / "meta/episode_manifest.json", parent_lib / "meta/input_manifest.json"
    for p, pin, reason in ((ep_path, pins.episode_manifest_sha256, "episode_manifest_sha"),
                           (in_path, pins.input_manifest_sha256, "input_manifest_sha"),
                           (norm_path, pins.norm_sha256, "norm_sha")):
        if not p.is_file() or sha256_file(p) != pin:
            raise RestoreError(reason, str(p))
    manifest = json.loads(ep_path.read_text(encoding="utf-8"))
    canon = wc.manifest_sha256(manifest)
    if manifest.get("sha256") != canon or canon != pins.manifest_canonical_sha256:
        raise RestoreError("manifest_canonical_sha", canon)
    if total.get("schema") != 3 or total.get("manifest_sha256") != pins.manifest_canonical_sha256:
        raise RestoreError("total_metadata_contract", "schema/manifest_sha256")
    contract = {"grid_stride": wc.GRID_STRIDE, "window_frames": wc.WINDOW_FRAMES,
                "demo_min_real_frames": wc.DEMO_MIN_REAL_FRAMES, "exec_min_real_frames": wc.EXEC_MIN_REAL_FRAMES}
    if any(total.get(k) != v for k, v in contract.items()):
        raise RestoreError("total_metadata_contract", "窗口契约")
    # 5. 段集合与 manifest 独立重算
    expected_segs = {it["key"]: it for it in wc.list_segments(manifest)}
    segs = total["segments"]
    if len(segs) != pins.segments or set(segs) != set(expected_segs):
        raise RestoreError("segment_set_mismatch", f"{len(segs)} 段 vs 清单 {len(expected_segs)}")
    bin_names = {n[:-4] for n in lat_files if n.endswith(".bin")}
    if bin_names != set(segs):
        raise RestoreError("segment_set_mismatch", "bin 文件集合与总 metadata 段集合不同")
    windows = padded = bin_bytes = 0
    vae_ids: dict[str, dict] = {}
    rows_index: dict[str, dict] = {}
    for key, seg in segs.items():
        it = expected_segs[key]
        ng = int(seg["num_grid"])
        if (ng != it["num_grid"] or int(seg["seg_len"]) != it["seg_len"]
                or int(seg["seg_start_global"]) != it["seg_start"] or seg["segment_kind"] != it["segment"]):
            raise RestoreError("segment_contract", key)
        # 闭环：总 metadata 记录的 sha/bytes → 当前 bin 实算
        bname = f"{pins.latent_prefix}/{key}.bin"
        if (int(seg["bytes"]) != ng * wc.CHUNK_BYTES or lat_files[f"{key}.bin"] != int(seg["bytes"])
                or actual[bname] != seg["sha256"]):
            raise RestoreError("bin_vs_total_metadata", key)
        sidecar = (lat_dir / f"{key}.bin.sha256").read_text(encoding="utf-8")
        if sidecar != f"{seg['sha256']}  {key}.bin\n":
            raise RestoreError("sidecar_mismatch", key)
        sm = json.loads((lat_dir / f"{key}.metadata.json").read_text(encoding="utf-8"))
        if (sm.get("schema") != 3 or sm.get("segment") != key or sm.get("sha256") != seg["sha256"]
                or sm.get("bytes") != seg["bytes"] or sm.get("num_grid") != seg["num_grid"]
                or sm.get("rows") != seg["rows"]):
            raise RestoreError("segment_metadata_mismatch", key)
        rows = seg["rows"]
        if len(rows) != ng or [int(r["m"]) for r in rows] != list(range(ng)):
            raise RestoreError("segment_rows", key)
        v = sm.get("vae") or {}
        if (v.get("vae_state_sha256") != pins.vae_state_sha256 or v.get("batch") != 1 or v.get("vae_dtype") != "float32"
                or v.get("tf32") != "off" or v.get("amp") != "off" or v.get("latent_mode") != "mode"):
            raise RestoreError("vae_state_or_flags", key)
        ident = {k: v.get(k) for k in VAE_IDENTITY_KEYS}
        vae_ids[json.dumps(ident, sort_keys=True, default=str)] = ident
        windows += ng
        bin_bytes += int(seg["bytes"])
        rows_index[key] = {int(r["m"]): int(r.get("pad_frames") or 0) for r in rows}
        padded += sum(1 for r in rows if int(r.get("pad_frames") or 0) > 0)
    if len(vae_ids) != 1:
        raise RestoreError("vae_identity_not_unique", f"{len(vae_ids)} 种")
    if windows != pins.windows or padded != pins.padded_windows:
        raise RestoreError("window_counts", f"windows={windows} padded={padded}")
    tot = total.get("totals") or {}
    if tot.get("segments") != pins.segments or tot.get("windows") != windows or tot.get("bytes") != bin_bytes:
        raise RestoreError("total_metadata_totals", json.dumps(tot))
    # 6. oracle 报告（聚合 + 分片 + 抽样）
    oracle = verify_oracle(ora_dir, pins, set(segs), rows_index, actual)
    workers = total.get("workers") or []
    ident = next(iter(vae_ids.values()))
    return {
        "actual": actual, "segments": len(segs), "windows": windows, "padded": padded, "bin_bytes": bin_bytes,
        "vae": {"vae_state_sha256": ident["vae_state_sha256"], "identity": ident,
                "historical_module_sha256": ident.get("module_sha256"),
                "historical_encoder_src_sha256": ident.get("encoder_src_sha256"),
                "worker_count": len(workers),
                "worker_git_commits": sorted({str(w.get("git_commit")) for w in workers}),
                "worker_mj_commits": sorted({str(w.get("mj_commit")) for w in workers})},
        "parent": {"lib": str(parent_lib), "episode_manifest_sha256": pins.episode_manifest_sha256,
                   "input_manifest_sha256": pins.input_manifest_sha256,
                   "manifest_canonical_sha256": canon, "raw_dir_recorded": total.get("raw_dir"),
                   "norm_stats": str(norm_path), "norm_sha256": pins.norm_sha256},
        "oracle": oracle,
    }


def verify_oracle(ora_dir: pathlib.Path, pins: Pins, seg_keys: set, rows_index: dict, actual: dict) -> dict:
    rep = json.loads((ora_dir / "vae_report.json").read_text(encoding="utf-8"))
    n = pins.oracle_shards
    checks = {"manifest_sha256": pins.manifest_canonical_sha256,
              "tested_metadata_sha256": pins.total_metadata_sha256,
              "metadata_rows_checked": pins.windows, "windows": pins.numeric_windows,
              "windows_encoded": pins.numeric_windows, "frame_mismatches": 0, "metadata_mismatches": 0,
              "aggregated_from_shards": n,
              "demo_min_real_frames": wc.DEMO_MIN_REAL_FRAMES, "exec_min_real_frames": wc.EXEC_MIN_REAL_FRAMES}
    for k, v in checks.items():
        if rep.get(k) != v:
            raise RestoreError("oracle_report_field", f"{k}={rep.get(k)!r} != {v!r}")
    if (rep.get("vae_provenance") or {}).get("vae_state_sha256") != pins.vae_state_sha256:
        raise RestoreError("oracle_vae_state", "vae_provenance.vae_state_sha256")
    if set(rep.get("segments") or {}) != seg_keys:
        raise RestoreError("oracle_segment_set", "聚合报告段集合 != 总 metadata")
    union: set = set()
    rows_sum = win_sum = 0
    sampled_shard: list = []
    for i in range(n):
        sr = json.loads((ora_dir / f"vae_report.shard{i}of{n}.json").read_text(encoding="utf-8"))
        if (sr.get("shard") or {}) != {"idx": i, "num": n}:
            raise RestoreError("oracle_shard_identity", f"shard {i}")
        for k in ("manifest_sha256", "tested_metadata_sha256", "sample_spec"):
            if sr.get(k) != rep.get(k):
                raise RestoreError("oracle_shard_field", f"shard {i} {k}")
        if sr.get("frame_mismatches") != 0 or sr.get("metadata_mismatches") != 0:
            raise RestoreError("oracle_shard_mismatch", f"shard {i}")
        ks = set(sr.get("segments") or {})
        if ks & union:
            raise RestoreError("oracle_shard_overlap", f"shard {i}")
        union |= ks
        rows_sum += int(sr["metadata_rows_checked"])
        win_sum += int(sr["windows"])
        sw = json.loads((ora_dir / f"sampled_windows.shard{i}of{n}.json").read_text(encoding="utf-8"))
        if sw.get("sample_spec") != rep.get("sample_spec"):
            raise RestoreError("oracle_sample_spec", f"shard {i}")
        sampled_shard += [(w["segment"], int(w["m"])) for w in sw["windows"]]
    if union != seg_keys or rows_sum != pins.windows or win_sum != pins.numeric_windows:
        raise RestoreError("oracle_shard_totals", f"segs={len(union)} rows={rows_sum} windows={win_sum}")
    sw = json.loads((ora_dir / "sampled_windows.json").read_text(encoding="utf-8"))
    sampled = [(w["segment"], int(w["m"])) for w in sw["windows"]]
    if sw.get("sample_spec") != rep.get("sample_spec") or sorted(sampled) != sorted(sampled_shard):
        raise RestoreError("oracle_sampled_mismatch", "聚合抽样表与分片并集不同")
    if len(sampled) != pins.numeric_windows or len(set(sampled)) != len(sampled):
        raise RestoreError("oracle_sampled_count", str(len(sampled)))
    pad_hit = 0
    for key, m in sampled:
        if key not in rows_index or m not in rows_index[key]:
            raise RestoreError("oracle_sampled_unknown", f"{key}:{m}")
        pad_hit += rows_index[key][m] > 0
    if pad_hit != pins.padded_windows:
        raise RestoreError("oracle_padded_coverage", f"{pad_hit} != {pins.padded_windows}")
    return {"vae_report_sha256": actual[f"{pins.oracle_prefix}/{pins.oracle_sub}/vae_report.json"],
            "metadata_rows_checked": rep["metadata_rows_checked"], "numeric_windows": rep["windows"],
            "numeric_padded_windows": pad_hit, "frame_mismatches": 0, "metadata_mismatches": 0,
            "aggregated_from_shards": n, "sample_spec": rep.get("sample_spec"),
            "tested_metadata_sha256": rep["tested_metadata_sha256"],
            "orig_module_sha256": rep.get("orig_module_sha256"),
            "coverage": (f"全 {pins.windows} 窗输入帧/metadata 核对；{pins.numeric_windows} 窗数值对拍"
                         f"（含全部 {pins.padded_windows} 个补帧窗）；不是全量数值对拍")}


# ══ 主流程 ═════════════════════════════════════════════════════════════════


def precheck_out_dir(out_dir: pathlib.Path, expect: Expect) -> dict | None:
    """只读预检：已封口拒绝；非空时必须带本任务 state（任务名与输入摘要一致）。返回已有 state 或 None。"""
    check_root(out_dir)
    if not os.path.lexists(out_dir):
        return None
    if os.path.lexists(out_dir / RECEIPT):
        raise RestoreError("receipt_exists", "已封口的恢复目录拒绝改写")
    st_path = out_dir / STATE
    if os.path.lexists(st_path):
        if not stat.S_ISREG(os.lstat(st_path).st_mode):
            raise RestoreError("state_mismatch", "state 不是普通文件")
        state = json.loads(st_path.read_text(encoding="utf-8"))
        if state.get("task") != TASK or state.get("inputs_digest") != expect.digest():
            raise RestoreError("state_mismatch", "恢复目录属于其他任务或输入")
        return state
    if [e for e in os.listdir(out_dir) if e != LOCK]:
        raise RestoreError("out_dir_not_empty", f"{out_dir} 非空且无本任务 {STATE}")
    return None


def prepare_out_dir(out_dir: pathlib.Path, expect: Expect) -> dict:
    state = precheck_out_dir(out_dir, expect)
    if state is not None:
        return state
    state = {"schema_version": SCHEMA_VERSION, "task": TASK, "inputs_digest": expect.digest(),
             "started_at_utc": utc_now(), "objects": {}}
    write_json(out_dir / STATE, state)
    return state


def run(args, pins: Pins = Pins()) -> int:
    started = utc_now()
    expect = load_expectations(pathlib.Path(args.export_dir), pins)
    objs = expect.objects
    lat = [o for o in objs if o.kind == "latent"]
    if args.dry_run:
        for o in objs:
            print(f"  {o.kind:6s} {o.rel}  bytes={o.bytes}  sha256={o.sha256}  members={len(o.members)}  "
                  f"extract={len(o.extract)}")
        parent = pathlib.Path(args.parent_lib)
        for p, pin, reason in ((parent / "meta/episode_manifest.json", pins.episode_manifest_sha256, "episode_manifest_sha"),
                               (parent / "meta/input_manifest.json", pins.input_manifest_sha256, "input_manifest_sha"),
                               (pathlib.Path(args.norm_stats), pins.norm_sha256, "norm_sha")):
            if not p.is_file() or sha256_file(p) != pin:
                raise RestoreError(reason, str(p))
        manifest = json.loads((parent / "meta/episode_manifest.json").read_text(encoding="utf-8"))
        canon = wc.manifest_sha256(manifest)
        if canon != pins.manifest_canonical_sha256 or manifest.get("sha256") != canon:
            raise RestoreError("manifest_canonical_sha", canon)
        nseg = len(wc.list_segments(manifest))
        if nseg != pins.segments:
            raise RestoreError("segment_set_mismatch", f"清单重算 {nseg} 段")
        print(f"latent_tar_bytes={sum(o.bytes for o in lat)} oracle_bytes={objs[-1].bytes} "
              f"total_bytes={sum(o.bytes for o in objs)} latent_members={sum(len(o.members) for o in lat)} "
              f"manifest_segments={nseg} manifest_canonical={canon[:16]}")
        print(f"RESTORE_DRYRUN=PASS objects={len(objs)} latent_tars={len(lat)} "
              f"latent_tar_bytes={sum(o.bytes for o in lat)} members={sum(len(o.members) for o in lat)} "
              f"oracle_extract={len(objs[-1].extract)} segments={nseg}", flush=True)
        return 0
    if not args.allow_no_token and not os.environ.get("HF_TOKEN"):
        raise RestoreError("hf_token_missing", "请先 source v1-store/secrets/hf.env")
    out_dir = pathlib.Path(os.path.abspath(args.out_dir))
    staging = pathlib.Path(os.path.abspath(args.staging_dir)) if args.staging_dir else out_dir.with_name(out_dir.name + ".staging")
    precheck_out_dir(out_dir, expect)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(out_dir / LOCK, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o644)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RestoreError("concurrent_run", str(out_dir))
        return _run_locked(args, pins, expect, out_dir, staging, started)
    finally:
        os.close(lock_fd)


def _run_locked(args, pins: Pins, expect: Expect, out_dir: pathlib.Path, staging: pathlib.Path, started: str) -> int:
    objs = expect.objects
    lat = [o for o in objs if o.kind == "latent"]
    state = prepare_out_dir(out_dir, expect)
    check_root(staging)
    staging.mkdir(parents=True, exist_ok=True)
    hf_cmd = shlex.split(args.hf_cmd)
    for k, o in enumerate(objs, 1):
        done = state["objects"].get(o.rel)
        if done and done.get("status") == "extracted" and done.get("sha256") == o.sha256:
            log(f"[{k}/{len(objs)}] 跳过已完成 {o.rel}（封口阶段全量重核）")
            continue
        free = shutil.disk_usage(staging).free
        if free < o.bytes + args.min_free_gib * 2**30:
            raise RestoreError("disk_space", f"staging 剩余 {free} B 不足")
        log(f"[{k}/{len(objs)}] 下载 {o.rel} bytes={o.bytes}")
        tar_path, attempts = download(o, staging, hf_cmd, pins.bucket, args.retries, args.retry_sleep, not args.no_ionice)
        scan_tar(tar_path, o, pins)
        n = extract_tar(tar_path, o, out_dir)
        state["objects"][o.rel] = {"status": "extracted", "sha256": o.sha256, "bytes": o.bytes,
                                   "extracted": n, "attempts": attempts, "finished_at_utc": utc_now()}
        write_json(out_dir / STATE, state)
        if not args.keep_staging:
            os.unlink(tar_path)
        log(f"[{k}/{len(objs)}] 已解包 {n} 个成员 {o.rel}")
    log("全部对象已解包，开始封口全量校验")
    res = verify_closure(out_dir, expect, pins, pathlib.Path(args.parent_lib), pathlib.Path(args.norm_stats))
    lines = "".join(f"{res['actual'][n]}  {n}\n" for n in sorted(res["actual"]))
    receipt_files = out_dir / FILES_LIST
    atomic_write_text(receipt_files, lines)
    script = pathlib.Path(__file__).resolve()
    receipt = {
        "schema_version": SCHEMA_VERSION, "task": TASK, "status": "PASS",
        "bucket": pins.bucket, "out_dir": str(out_dir),
        "layout": {"latents": pins.latent_prefix, "oracle": f"{pins.oracle_prefix}/{pins.oracle_sub}"},
        "objects": [{"rel": o.rel, "kind": o.kind, "sha256": o.sha256, "bytes": o.bytes,
                     "members": len(o.members), "member_bytes": o.member_bytes, "extracted": len(o.extract),
                     "attempts": state["objects"][o.rel].get("attempts")} for o in objs],
        "counts": {"objects": len(objs), "latent_tar_bytes": sum(o.bytes for o in lat),
                   "latent_members": sum(len(o.members) for o in lat),
                   "oracle_members_extracted": len(objs[-1].extract),
                   "segments": res["segments"], "windows": res["windows"], "padded_windows": res["padded"]},
        "latents": {"total_metadata_sha256": pins.total_metadata_sha256, "segment_bins_bytes": res["bin_bytes"],
                    "files_list": FILES_LIST, "files_list_sha256": sha256_file(receipt_files)},
        "vae": res["vae"], "parent": res["parent"], "oracle": res["oracle"],
        "expectation_sources": {"export_dir": str(pathlib.Path(args.export_dir).resolve()), **expect.sources},
        "tool": {"script": str(script), "script_sha256": sha256_file(script),
                 "wan_common_sha256": sha256_file(pathlib.Path(wc.__file__).resolve()),
                 "python": sys.version.split()[0]},
        "excluded": ["旧 motion token / encoder_report / oracle bin 不提取", "不含任何新 encoder 身份",
                     "未从 RGB 重新做 VAE；复用的是原已验收并归档的 latent"],
        "new_encoder_identity": None,
        "started_at_utc": state.get("started_at_utc", started), "finished_at_utc": utc_now(),
    }
    write_json(out_dir / RECEIPT, receipt)
    rsha = sha256_file(out_dir / RECEIPT)
    print(f"VAE_INPUT_READY=PASS objects={len(objs)} members={receipt['counts']['latent_members']} "
          f"segments={res['segments']} windows={res['windows']} padded={res['padded']} "
          f"oracle_rows={res['oracle']['metadata_rows_checked']} oracle_numeric={res['oracle']['numeric_windows']} "
          f"total_meta={pins.total_metadata_sha256[:16]} vae_state={res['vae']['vae_state_sha256'][:16]} "
          f"receipt={rsha[:16]}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", default=str(DEFAULT_EXPORT), help="打包期留下的期望值目录（只读）")
    ap.add_argument("--parent-lib", default=str(DEFAULT_PARENT), help="VLA 父库根（只读）")
    ap.add_argument("--norm-stats", default=str(DEFAULT_NORM), help="父库 norm_stats.json（只读）")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="新恢复目录（空或本任务续跑）")
    ap.add_argument("--staging-dir", default=None, help="下载暂存目录（默认 <out-dir>.staging）")
    ap.add_argument("--hf-cmd", default=DEFAULT_HF_CMD, help="hf CLI 前缀（测试可换假命令）")
    ap.add_argument("--retries", type=int, default=2, help="首试之外的重试次数")
    ap.add_argument("--retry-sleep", type=float, default=30.0)
    ap.add_argument("--min-free-gib", type=float, default=4.0, help="下载前 staging 至少再留多少 GiB")
    ap.add_argument("--keep-staging", action="store_true", help="解包后保留 staging tar")
    ap.add_argument("--no-ionice", action="store_true", help="下载不加 ionice -c3 nice -n19")
    ap.add_argument("--allow-no-token", action="store_true", help="测试用：不要求 HF_TOKEN")
    ap.add_argument("--dry-run", action="store_true", help="只解析期望值与父库，不下载不写盘")
    return ap


def main(argv=None, pins: Pins = Pins()) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args, pins)
    except RestoreError as e:
        print(f"错误: {e}", flush=True)
        tag = "RESTORE_DRYRUN" if args.dry_run else "VAE_INPUT_READY"
        print(f"{tag}=FAIL reason={e.reason}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
