#!/usr/bin/env python3
"""四对 modulation VLA（mixed8 pair4，80k 步）Orbax 检查点增量上传器（1020 计划「上传支线」VLA 部分）。

三种模式：

watch     S7 训练期间运行。每轮（间隔 >=30 s）扫描 run 目录下的预期 step 目录
          ``{save_interval, 2·save_interval, … < steps-1} ∪ {steps-1}``（正式 5000…75000 + 79999 共 16 个）：
            1. 只接受目录名恰为整数且在预期集合里的最终 step 目录；``<step>.orbax-checkpoint-tmp-*``
               等临时目录一律不传；``_CHECKPOINT_METADATA.commit_timestamp_nsecs`` 必须是正整数；
            2. 需要 final-record 的 ``start.json``：run_name、checkpoint_dir 与本 run 一致，取其 run_uuid
               绑定全部 marker；最后一份（steps-1）另需同 run_uuid/head 且 ``wait_until_finished=true`` 的
               ``checkpoint_wait_done.json``；
            3. 成员清单快照：第一遍逐成员（相对路径、bytes、mtime、SHA256），第二遍只重新 stat；
               两遍不一致视为写入未稳定，不计错误、下一轮重扫（用户裁决 F1 情况①）；
               Orbax 目录不复制（单份约 12 GB），上传直接读源文件，上传后回读 SHA 与清单比对兜底；
            4. 逐成员上传到 ``<step>/<相对路径>``（保留完整 Orbax 相对路径）→ 逐成员鉴权回读比
               SHA/bytes → 源目录全量重哈希核对未变 → 上传并回读远端完成清单
               ``_upload_manifests/<step>.json`` → 原子写 verified marker。
          已快照或已 verified 的 step，成员集合或字节再变即 ``VLA_UPLOAD_CONFLICT``，上传器非零退出
          （F1 情况②），不影响训练。每对象训练期最多首试＋2 次重试，失败留在队列（F2）。
          训练 tmux 退出后再做一次完整补扫，然后退出。
finalize  训练与 watch 都退出后运行。每个未完成 step 再给一轮首试＋2 次重试（F2）；上传 final-record 三件套、
          run 根目录平文件、实际 history YAML、norm_stats.json、motion store meta、来源摘要
          ``source_summary.json`` 与覆盖全部上传对象的 ``SHA256SUMS``；远端核完整对象集；末行
          ``VLA_UPLOAD=PASS|FAIL objects=<n> verified=<n> during_training=<true|false>
          first_verified_at=<UTC> train_end_at=<UTC>``（F3：全部 verified 即 PASS，during_training 如实）。
scan      只读：枚举 run 目录下的 step 目录，打印提交状态、成员清单与清单摘要；不联网、不读凭据、不写任何文件。

本进程只用标准库，不导入 JAX/torch、不初始化 CUDA；HF 子进程由 ckpt_upload_common.HFBucket 负责
（``ionice -c3 nice -n19``、``CUDA_VISIBLE_DEVICES=''``、只在子进程解除 HF_HUB_OFFLINE、token 只进子进程环境）。

退出码：0 正常（watch 全部 verified／达到 --max-rounds；finalize PASS；scan 完成）；1 finalize FAIL；
2 停止事件（冲突、身份错误、归属不符、bucket 非私有等）；3 watch 结束时仍有待传对象；64 参数错误。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat as _stat
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ckpt_upload_common as common  # noqa: E402
from ckpt_upload_common import UploaderStop, log  # noqa: E402

SCHEMA = "mixed8-pair4-vla-ckpt-upload/1"
KIND = "vla"
COMMIT_FILE = "_CHECKPOINT_METADATA"
MANIFEST_PREFIX = "_upload_manifests"
META_PREFIX = "_run-meta"
SUMS_NAME = "SHA256SUMS"
WAIT_DONE = "checkpoint_wait_done.json"
FINAL_RECORD_FILES = ("start.json", "final.json", WAIT_DONE)
ATTEMPTS_PER_PHASE = 3  # 首试＋2 次重试
EXIT_OK, EXIT_FAIL, EXIT_STOP, EXIT_PENDING, EXIT_USAGE = 0, 1, 2, 3, 64
STEP_RE = re.compile(r"^[0-9]+$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


# ----------------------------------------------------------------------------------------------
# step 目录：提交判定、成员清单
# ----------------------------------------------------------------------------------------------

def expected_steps(steps: int, save_interval: int) -> list:
    """训练循环 0 基 step：``step % save_interval == 0 and step > 0`` 或 ``step == steps-1`` 时保存。"""
    out = {s for s in range(save_interval, steps - 1, save_interval)}
    out.add(steps - 1)
    return sorted(out)


def commit_status(step_dir: Path):
    """返回 (ok, reason, commit_ns)。只认 _CHECKPOINT_METADATA 里有效的 commit_timestamp_nsecs。"""
    meta = step_dir / COMMIT_FILE
    if meta.is_symlink() or not meta.is_file():
        return False, "commit_metadata_missing", None
    try:
        data = json.loads(meta.read_text())
    except (ValueError, OSError):
        return False, "commit_metadata_unreadable", None
    if not isinstance(data, dict):
        return False, "commit_metadata_not_object", None
    ts = data.get("commit_timestamp_nsecs")
    if isinstance(ts, bool) or not isinstance(ts, int) or ts <= 0:
        return False, "commit_timestamp_invalid", None
    init = data.get("init_timestamp_nsecs")
    if isinstance(init, int) and not isinstance(init, bool) and init > ts:
        return False, "commit_before_init", None
    return True, "ok", ts


def enumerate_members(step_dir: Path, with_hash: bool):
    """枚举 step 目录全部成员。返回 (members, empty_dirs)；members: 相对路径 -> {bytes, mtime_ns[, sha256]}。

    符号链接或非常规文件属于身份错误（Orbax 正常不产生），抛 UploaderStop。
    """
    members, empty_dirs = {}, []
    for dirpath, dirnames, filenames in os.walk(step_dir, followlinks=False):
        base = Path(dirpath)
        for d in dirnames:
            p = base / d
            if p.is_symlink():
                raise UploaderStop("VLA_UPLOAD_IDENTITY", f"{p} 是符号链接")
        if not dirnames and not filenames and base != step_dir:
            empty_dirs.append(base.relative_to(step_dir).as_posix())
        for f in filenames:
            p = base / f
            st = os.lstat(p)
            if not _stat.S_ISREG(st.st_mode):
                raise UploaderStop("VLA_UPLOAD_IDENTITY", f"{p} 不是常规文件")
            rel = p.relative_to(step_dir).as_posix()
            entry = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
            if with_hash:
                sha, size = common.sha256_file(p)
                entry["sha256"] = sha
                entry["hashed_bytes"] = size
            members[rel] = entry
    return members, sorted(empty_dirs)


def manifest_digest(members: dict) -> str:
    rows = [[rel, members[rel]["bytes"], members[rel]["sha256"]] for rel in sorted(members)]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def snapshot_step(step_dir: Path):
    """成员清单快照（F1 情况①）：两遍枚举不一致返回 (None, 原因)。"""
    ok, reason, ts = commit_status(step_dir)
    if not ok:
        return None, reason
    first, empty = enumerate_members(step_dir, with_hash=True)
    second, empty2 = enumerate_members(step_dir, with_hash=False)
    ok2, _, ts2 = commit_status(step_dir)
    if set(first) != set(second) or empty != empty2 or not ok2 or ts2 != ts:
        return None, "member_set_changed"
    for rel, a in first.items():
        b = second[rel]
        if a["bytes"] != b["bytes"] or a["mtime_ns"] != b["mtime_ns"] or a["hashed_bytes"] != a["bytes"]:
            return None, "member_stat_changed"
    if COMMIT_FILE not in first:
        return None, "commit_metadata_missing"
    members = {rel: {"bytes": m["bytes"], "mtime_ns": m["mtime_ns"], "sha256": m["sha256"]}
               for rel, m in first.items()}
    return {"members": members, "empty_dirs": empty, "commit_timestamp_nsecs": ts,
            "digest": manifest_digest(members), "file_count": len(members),
            "total_bytes": sum(m["bytes"] for m in members.values())}, "ok"


# ----------------------------------------------------------------------------------------------
# 上传器主体
# ----------------------------------------------------------------------------------------------

def step_key(step: int) -> str:
    return f"step-{step}"


def meta_key(remote_path: str) -> str:
    return "meta-" + remote_path.replace("/", "__")


class Uploader:
    def __init__(self, args, mode):
        self.a = args
        self.mode = mode
        self.run_dir = Path(args.run_dir)
        self.final_record_dir = Path(args.final_record_dir)
        self.export_root = Path(args.export_root)
        self.run_name = args.run_name or self.run_dir.name
        self.readback_dir = self.export_root / "readback"
        self.manifest_dir = self.export_root / "manifests"
        self.meta_dir = self.export_root / "meta"
        self.state_path = self.export_root / "state.json"
        self.ledger = common.ObjectLedger(self.export_root, schema=SCHEMA, run_name=self.run_name,
                                          bucket_id=args.bucket, kind=KIND)
        self.bucket = None
        self.bucket_ready = False
        self.seen = set()
        self.remote_files = None
        self.state = None
        self.start = None  # 已核身份的 start.json

    # --- 通用 ----------------------------------------------------------------------------------
    def steps(self):
        return expected_steps(self.a.steps, self.a.save_interval)

    def last_step(self):
        return self.a.steps - 1

    def train_alive(self) -> bool:
        rc = subprocess.run(["tmux", "has-session", "-t", f"={self.a.train_tmux}"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        return rc == 0

    def get_bucket(self):
        if self.bucket is None:
            token = common.load_hf_token(self.a.hf_env)
            self.bucket = common.HFBucket(self.a.bucket, token, cache_root=self.export_root / "cache")
        return self.bucket

    def ensure_bucket(self):
        if not self.bucket_ready:
            common.ensure_private_bucket(self.get_bucket(), self.ledger)
            self.bucket_ready = True

    def remote_listing(self, refresh=False):
        if self.remote_files is None or refresh:
            self.ensure_bucket()
            self.remote_files = self.bucket.list_files()
        return self.remote_files

    def sleep_retry(self):
        if self.a.retry_sleep > 0:
            time.sleep(self.a.retry_sleep)

    # --- 运行身份（final-record） --------------------------------------------------------------
    def load_start(self):
        """start.json 存在则核身份并返回；不存在返回 None。身份不符停上传器。"""
        path = self.final_record_dir / "start.json"
        if not path.is_file():
            if self.start is not None:
                raise UploaderStop("VLA_UPLOAD_IDENTITY", "start.json 已核过但现在消失")
            return None
        try:
            data = json.loads(path.read_text())
        except ValueError:
            return None  # 写入中（final_record 用 open('x') 直写），下一轮再读
        problems = []
        uuid = data.get("run_uuid")
        if not isinstance(uuid, str) or not uuid:
            problems.append("run_uuid 缺失")
        if data.get("run_name") != self.run_name:
            problems.append(f"run_name={data.get('run_name')} 期望 {self.run_name}")
        cdir = data.get("checkpoint_dir")
        if not isinstance(cdir, str) or Path(cdir).resolve() != self.run_dir.resolve():
            problems.append(f"checkpoint_dir={cdir} 不是 {self.run_dir}")
        if data.get("num_train_steps") not in (None, self.a.steps):
            problems.append(f"num_train_steps={data.get('num_train_steps')} 期望 {self.a.steps}")
        if data.get("save_interval") not in (None, self.a.save_interval):
            problems.append(f"save_interval={data.get('save_interval')} 期望 {self.a.save_interval}")
        if problems:
            raise UploaderStop("VLA_UPLOAD_IDENTITY", "start.json: " + "; ".join(problems))
        if self.start is not None and self.start["run_uuid"] != uuid:
            raise UploaderStop("VLA_UPLOAD_IDENTITY", "start.json 的 run_uuid 在运行中改变")
        if self.start is None:
            self.bind_uuid(uuid)
            log("VLA_RUN_IDENTITY", run=self.run_name, run_uuid=uuid, head=data.get("head"))
        self.start = data
        return data

    def bind_uuid(self, uuid):
        if self.state.get("run_uuid") not in (None, uuid):
            raise UploaderStop("UPLOAD_MARKER_MISMATCH",
                               f"state.json run_uuid={self.state.get('run_uuid')} 与 start.json {uuid} 不符")
        for name in self.ledger.names():
            m = self.ledger.load(name)
            if m and m.get("run_uuid") != uuid:
                raise UploaderStop("UPLOAD_MARKER_MISMATCH", f"marker {name} run_uuid={m.get('run_uuid')} 不是 {uuid}")
        if self.state.get("run_uuid") is None:
            self.state["run_uuid"] = uuid
            common.atomic_write_json(self.state_path, self.state)

    def wait_done_status(self):
        """返回 ok / missing / unreadable；身份不符停上传器。"""
        path = self.final_record_dir / WAIT_DONE
        if not path.is_file():
            return "missing"
        try:
            data = json.loads(path.read_text())
        except ValueError:
            return "unreadable"
        if data.get("run_uuid") != self.start["run_uuid"] or data.get("head") != self.start.get("head"):
            raise UploaderStop("VLA_UPLOAD_IDENTITY",
                               f"{WAIT_DONE} run_uuid/head={data.get('run_uuid')}/{data.get('head')} 与 start.json 不符")
        if data.get("wait_until_finished") is not True:
            return "not_finished"
        return "ok"

    def load_marker(self, name):
        m = self.ledger.load(name)
        if m is not None and self.start is not None and m.get("run_uuid") != self.start["run_uuid"]:
            raise UploaderStop("UPLOAD_MARKER_MISMATCH", f"marker {name} run_uuid={m.get('run_uuid')} 与 start.json 不符")
        return m

    def load_state(self):
        try:
            st = common.read_json(self.state_path)
        except ValueError:
            raise UploaderStop("UPLOAD_MARKER_MISMATCH", "state.json 不是合法 JSON")
        expected = self.ledger.owner_expected()
        if st is None:
            st = dict(expected, run_uuid=None, train_end_at=None, train_end_note=None)
            common.atomic_write_json(self.state_path, st)
        elif any(st.get(k) != v for k, v in expected.items()):
            raise UploaderStop("UPLOAD_MARKER_MISMATCH", "state.json 属于其他 run/bucket")
        self.state = st

    def note_train_end(self, note):
        if self.state.get("train_end_at"):
            return
        self.state["train_end_at"] = common.utc_now()
        self.state["train_end_note"] = note
        common.atomic_write_json(self.state_path, self.state)
        log("VLA_TRAIN_END_OBSERVED", at=self.state["train_end_at"], note=note)

    # --- 源核对 --------------------------------------------------------------------------------
    def check_source(self, step, marker, force_hash):
        """已有 marker 的 step：成员集合与字节必须仍是 marker 绑定的清单（F1 情况②）。"""
        step_dir = self.run_dir / str(step)
        name = step_key(step)
        if not step_dir.is_dir() or step_dir.is_symlink():
            raise UploaderStop("VLA_UPLOAD_CONFLICT", f"{name} 已快照/已上传但源目录消失")
        now, empty = enumerate_members(step_dir, with_hash=False)
        bound = marker["members"]
        if set(now) != set(bound) or empty != marker.get("empty_dirs", []):
            added = sorted(set(now) - set(bound))[:3]
            removed = sorted(set(bound) - set(now))[:3]
            raise UploaderStop("VLA_UPLOAD_CONFLICT",
                               f"{name} 状态={marker['status']} 后成员集合变化 added={added} removed={removed}")
        touched = {}
        for rel, cur in now.items():
            ref = bound[rel]
            same = cur["bytes"] == ref["bytes"] and cur["mtime_ns"] == ref["mtime_ns"]
            if force_hash or not same:
                sha, size = common.sha256_file(step_dir / rel)
                if sha != ref["sha256"] or size != ref["bytes"]:
                    raise UploaderStop("VLA_UPLOAD_CONFLICT",
                                       f"{name} 状态={marker['status']} 后成员 {rel} 字节变化 "
                                       f"marker={ref['sha256'][:16]} now={sha[:16]} bytes={size}")
                if not same:
                    touched[rel] = dict(ref, mtime_ns=cur["mtime_ns"])
        if touched:
            members = dict(bound)
            members.update(touched)
            marker = self.ledger.save(name, dict(marker, members=members))
            log("VLA_SOURCE_TOUCHED", object=name, files=len(touched), note="mtime变化但字节相同")
        return marker

    # --- 单 step 状态机 -------------------------------------------------------------------------
    def process(self, step, phase):
        """返回 absent / waiting / unstable / verified / pending。"""
        name = step_key(step)
        step_dir = self.run_dir / str(step)
        marker = self.load_marker(name)
        first = name not in self.seen
        if marker is None:
            if not step_dir.is_dir():
                return "absent"
            if step_dir.is_symlink():
                raise UploaderStop("VLA_UPLOAD_IDENTITY", f"{step_dir} 是符号链接")
            if self.load_start() is None:
                log("VLA_WAIT", object=name, reason="start_json_missing")
                return "waiting"
            if step == self.last_step():
                ws = self.wait_done_status()
                if ws != "ok":
                    log("VLA_WAIT", object=name, reason=f"wait_done_{ws}")
                    return "waiting"
            snap, reason = snapshot_step(step_dir)
            if snap is None:
                log("VLA_WAIT", object=name, reason=reason)
                return "unstable"
            marker = self.ledger.save(name, dict(snap, status="snapshot", step=step, run_uuid=self.start["run_uuid"],
                                                 remote_prefix=str(step), uploaded_members=[],
                                                 snapshot_at=common.utc_now(),
                                                 attempts={"training": 0, "finalize": 0}))
            log("VLA_SNAPSHOT", object=name, files=snap["file_count"], bytes=snap["total_bytes"],
                digest=snap["digest"], commit_ns=snap["commit_timestamp_nsecs"])
            self.seen.add(name)
        else:
            # 重启时对未 verified 的对象做全量重哈希；已 verified 的做 stat 级核对＋远端核对（避免每次重启重读 190 GB）。
            marker = self.check_source(step, marker, force_hash=first and marker["status"] != "verified")
            if step == self.last_step() and marker["status"] != "verified" and self.wait_done_status() != "ok":
                raise UploaderStop("VLA_UPLOAD_IDENTITY", f"{name} 已快照但 {WAIT_DONE} 不再有效")
            self.seen.add(name)

        if marker["status"] == "verified":
            if not first:
                return "verified"
            try:
                remote = self.remote_listing()
            except common.RemoteError as exc:
                self.seen.discard(name)
                log("VLA_REMOTE_ERROR", object=name, step="list", detail=str(exc))
                return "verified"
            missing, wrong = self.remote_diff(marker, remote)
            if wrong:
                raise UploaderStop("VLA_UPLOAD_CONFLICT", f"{name} 远端 bytes 与 marker 不符: {wrong[:3]}")
            if not missing:
                return "verified"
            keep = [r for r in marker.get("uploaded_members", []) if f"{step}/{r}" not in missing]
            marker = self.ledger.save(name, dict(marker, status="snapshot", uploaded_members=keep))
            log("VLA_REMOTE_MISSING", object=name, missing=len(missing), action="requeue")
        return self.transfer(step, marker, phase)

    def remote_diff(self, marker, remote):
        step = marker["step"]
        expected = {f"{step}/{rel}": m["bytes"] for rel, m in marker["members"].items()}
        mpath = f"{MANIFEST_PREFIX}/{step}.json"
        if marker.get("manifest_sha256"):
            expected[mpath] = marker["manifest_bytes"]
        missing = sorted(p for p in expected if p not in remote)
        wrong = sorted(p for p in expected if p in remote and remote[p] != expected[p])
        return missing, wrong

    def write_manifest(self, marker) -> Path:
        """远端完成清单；内容确定（无时间戳），重试与重启得到相同字节。"""
        body = {"schema": SCHEMA, "run_name": self.run_name, "run_uuid": marker["run_uuid"],
                "bucket_id": self.a.bucket, "step": marker["step"], "digest": marker["digest"],
                "commit_timestamp_nsecs": marker["commit_timestamp_nsecs"], "file_count": marker["file_count"],
                "total_bytes": marker["total_bytes"], "empty_dirs": marker.get("empty_dirs", []),
                "members": [{"path": rel, "bytes": m["bytes"], "sha256": m["sha256"]}
                            for rel, m in sorted(marker["members"].items())]}
        path = self.manifest_dir / f"{marker['step']}.json"
        common.atomic_write_json(path, body)
        return path

    def transfer(self, step, marker, phase):
        name = step_key(step)
        attempts = dict(marker.get("attempts") or {"training": 0, "finalize": 0})
        if attempts.get(phase, 0) >= ATTEMPTS_PER_PHASE:
            return "pending"
        step_dir = self.run_dir / str(step)
        while attempts.get(phase, 0) < ATTEMPTS_PER_PHASE:
            attempts[phase] = attempts.get(phase, 0) + 1
            marker = self.ledger.save(name, dict(marker, attempts=attempts))
            n = attempts[phase]

            def fail(step_name, detail, readback=False):
                nonlocal marker
                marker = self.ledger.save(name, dict(marker, last_error=f"{step_name} {detail}"[:600]))
                log("VLA_READBACK_FAIL" if readback else "VLA_UPLOAD_RETRY", object=name, phase=phase,
                    attempt=n, step=step_name, detail=detail)
                if n < ATTEMPTS_PER_PHASE:
                    self.sleep_retry()

            try:
                self.ensure_bucket()
            except common.RemoteError as exc:
                fail("bucket", str(exc))
                continue
            marker = self.check_source(step, marker, force_hash=False)
            uploaded = list(marker.get("uploaded_members", []))
            ok = True
            for rel in sorted(marker["members"]):
                if rel in uploaded:
                    continue
                res = self.bucket.upload(step_dir / rel, f"{step}/{rel}")
                if not res.ok:
                    fail("upload", f"{rel} rc={res.rc} {res.detail}")
                    ok = False
                    break
                uploaded.append(rel)
                marker = self.ledger.save(name, dict(marker, uploaded_members=uploaded))
            if not ok:
                continue
            if marker["status"] == "snapshot":
                marker = self.ledger.save(name, dict(marker, status="uploaded", uploaded_at=common.utc_now(),
                                                     last_error=None))
                log("VLA_UPLOADED", object=name, files=marker["file_count"], bytes=marker["total_bytes"])
            # 逐成员鉴权回读
            bad, transient = [], None
            for rel, m in sorted(marker["members"].items()):
                good, detail = self.readback(f"{step}/{rel}", m["sha256"], m["bytes"])
                if not good:
                    if detail.startswith("mismatch"):
                        bad.append(rel)
                    else:
                        transient = f"{rel} {detail}"
                        break
            if bad or transient:
                keep = [r for r in uploaded if r not in bad]
                status = "snapshot" if bad else "uploaded"
                marker = self.ledger.save(name, dict(marker, status=status, uploaded_members=keep))
                fail("readback", f"mismatch={bad[:3]}" if bad else transient, readback=True)
                continue
            # 上传后源再核对（全量重哈希）：读回一致且源未变，才承认这份清单
            marker = self.check_source(step, marker, force_hash=True)
            mpath = self.write_manifest(marker)
            msha, mbytes = common.sha256_file(mpath)
            res = self.bucket.upload(mpath, f"{MANIFEST_PREFIX}/{step}.json")
            if not res.ok:
                fail("manifest_upload", f"rc={res.rc} {res.detail}")
                continue
            good, detail = self.readback(f"{MANIFEST_PREFIX}/{step}.json", msha, mbytes)
            if not good:
                fail("manifest_readback", detail, readback=True)
                continue
            alive = self.train_alive()
            marker = self.ledger.save(name, dict(marker, status="verified", verified_at=common.utc_now(),
                                                 verified_during_training=alive, manifest_sha256=msha,
                                                 manifest_bytes=mbytes, last_error=None))
            if self.remote_files is not None:
                for rel, m in marker["members"].items():
                    self.remote_files[f"{step}/{rel}"] = m["bytes"]
                self.remote_files[f"{MANIFEST_PREFIX}/{step}.json"] = mbytes
            log("VLA_VERIFIED", object=name, files=marker["file_count"], bytes=marker["total_bytes"],
                digest=marker["digest"], during_training=alive, attempt=n, phase=phase)
            return "verified"
        log("VLA_UPLOAD_GIVEUP", object=name, phase=phase, attempts=ATTEMPTS_PER_PHASE,
            last_error=marker.get("last_error", ""))
        return "pending"

    def readback(self, remote_path, sha256, nbytes):
        """鉴权下载到独立回读目录，比 SHA/bytes；回读文件随即删除。"""
        target = self.readback_dir / f"{remote_path.replace('/', '__')}.{os.getpid()}"
        target.unlink(missing_ok=True)
        try:
            res = self.bucket.download(remote_path, target)
            if not res.ok:
                return False, f"download rc={res.rc} {res.detail}"
            if not target.is_file():
                return False, "download 未产生文件"
            sha, size = common.sha256_file(target)
            if sha != sha256 or size != nbytes:
                return False, f"mismatch sha={sha[:16]} bytes={size}"
            return True, "ok"
        finally:
            target.unlink(missing_ok=True)

    # --- 元数据（finalize） -------------------------------------------------------------------
    def snapshot_meta(self, remote_path, src: Path) -> Path:
        """元数据独立复制成不可变快照；已有快照与源不一致即冲突。"""
        snap = self.meta_dir / remote_path
        if snap.is_file():
            if common.sha256_file(snap)[0] != common.sha256_file(src)[0]:
                raise UploaderStop("VLA_UPLOAD_CONFLICT", f"元数据 {remote_path} 快照与源 {src} 不一致")
            return snap
        common.copy_with_hash(src, snap)
        return snap

    def upload_meta(self, remote_path, local: Path):
        name = meta_key(remote_path)
        sha, size = common.sha256_file(local)
        marker = self.load_marker(name)
        if marker is not None and marker["sha256"] != sha:
            if marker["status"] == "verified":
                raise UploaderStop("VLA_UPLOAD_CONFLICT", f"{remote_path} 已 verified 但内容变化")
            marker = None
        if marker is not None and marker["status"] == "verified":
            if self.remote_listing().get(remote_path) == size:
                return "verified"
            marker = self.ledger.save(name, dict(marker, status="snapshot"))
        if marker is None:
            marker = self.ledger.save(name, {"status": "snapshot", "kind_detail": "meta", "remote_path": remote_path,
                                             "sha256": sha, "bytes": size, "run_uuid": self.start["run_uuid"],
                                             "attempts": {"training": 0, "finalize": 0}})
        attempts = dict(marker.get("attempts") or {})
        while attempts.get("finalize", 0) < ATTEMPTS_PER_PHASE:
            attempts["finalize"] = attempts.get("finalize", 0) + 1
            n = attempts["finalize"]
            marker = self.ledger.save(name, dict(marker, attempts=attempts))
            try:
                self.ensure_bucket()
            except common.RemoteError as exc:
                log("VLA_UPLOAD_RETRY", object=remote_path, phase="finalize", attempt=n, step="bucket", detail=str(exc))
                self.sleep_retry()
                continue
            if marker["status"] == "snapshot":
                res = self.bucket.upload(local, remote_path)
                if not res.ok:
                    log("VLA_UPLOAD_RETRY", object=remote_path, phase="finalize", attempt=n, step="upload", rc=res.rc)
                    self.sleep_retry()
                    continue
                marker = self.ledger.save(name, dict(marker, status="uploaded", uploaded_at=common.utc_now()))
            ok, detail = self.readback(remote_path, sha, size)
            if not ok:
                status = "snapshot" if detail.startswith("mismatch") else "uploaded"
                marker = self.ledger.save(name, dict(marker, status=status, last_error=f"readback {detail}"))
                log("VLA_READBACK_FAIL", object=remote_path, phase="finalize", attempt=n, detail=detail)
                self.sleep_retry()
                continue
            self.ledger.save(name, dict(marker, status="verified", verified_at=common.utc_now()))
            if self.remote_files is not None:
                self.remote_files[remote_path] = size
            log("VLA_VERIFIED", object=remote_path, sha256=sha, bytes=size, phase="finalize")
            return "verified"
        log("VLA_UPLOAD_GIVEUP", object=remote_path, phase="finalize", attempts=ATTEMPTS_PER_PHASE)
        return "pending"

    def check_final_record(self):
        """final-record 三件套同 run_uuid；final.json 记 state_step=steps、loop_step=checkpoint_step=steps-1。"""
        problems = []
        for fname in FINAL_RECORD_FILES:
            if not (self.final_record_dir / fname).is_file():
                problems.append(f"{fname} 缺失")
        if problems:
            return problems
        final = json.loads((self.final_record_dir / "final.json").read_text())
        if final.get("run_uuid") != self.start["run_uuid"]:
            raise UploaderStop("VLA_UPLOAD_IDENTITY", f"final.json run_uuid={final.get('run_uuid')} 与 start.json 不符")
        for key in ("run_name", "head", "checkpoint_dir"):
            if final.get(key) != self.start.get(key):
                problems.append(f"final.json {key} 与 start.json 不符")
        if final.get("state_step") != self.a.steps:
            problems.append(f"final.json state_step={final.get('state_step')} 期望 {self.a.steps}")
        if final.get("loop_step") != self.last_step() or final.get("checkpoint_step") != self.last_step():
            problems.append(f"final.json loop/checkpoint_step={final.get('loop_step')}/{final.get('checkpoint_step')}")
        if self.wait_done_status() != "ok":
            problems.append(f"{WAIT_DONE} 无效")
        return problems

    def source_summary(self, files: dict) -> dict:
        def ident(path):
            sha, size = common.sha256_file(path)
            return {"path": str(path), "sha256": sha, "bytes": size}
        summary = {"schema": SCHEMA, "run_name": self.run_name, "run_uuid": self.start["run_uuid"],
                   "head": self.start.get("head"), "bucket_id": self.a.bucket, "num_train_steps": self.a.steps,
                   "save_interval": self.a.save_interval, "expected_steps": self.steps(),
                   "encoder_ckpt_sha256": self.a.encoder_sha256,
                   "history_yaml": ident(Path(self.a.history_yaml)),
                   "norm_stats": ident(Path(self.a.norm_stats)),
                   "motion_store_meta": ident(Path(self.a.motion_store_meta)),
                   "step_manifests": {}}
        for step in self.steps():
            m = self.load_marker(step_key(step))
            summary["step_manifests"][str(step)] = {"digest": m["digest"], "file_count": m["file_count"],
                                                    "total_bytes": m["total_bytes"]}
        try:
            meta = json.loads(Path(self.a.motion_store_meta).read_text())
            if isinstance(meta, dict):
                summary["motion_store_meta_keys"] = sorted(meta)[:200]
        except (ValueError, OSError):
            pass
        summary["run_meta_files"] = sorted(files)
        return summary

    # --- 入口 ---------------------------------------------------------------------------------
    def startup(self):
        for d in (self.export_root.parent, self.export_root):
            if d.is_symlink():
                raise UploaderStop("EXPORT_ROOT_SYMLINK", str(d))
        self.export_root.mkdir(parents=True, exist_ok=True)
        self.readback_dir.mkdir(exist_ok=True)
        self.ledger.check_all()
        self.load_state()
        self.load_start()
        if self.start is None and self.ledger.names():
            raise UploaderStop("UPLOAD_MARKER_MISMATCH", "已有 marker 但 start.json 不存在")
        log("VLA_UPLOAD_START", mode=self.mode, run=self.run_name, bucket=self.a.bucket, run_dir=self.run_dir,
            final_record_dir=self.final_record_dir, export_root=self.export_root, objects=len(self.steps()),
            steps=",".join(str(s) for s in self.steps()), run_uuid=(self.start or {}).get("run_uuid", "pending"),
            python=sys.executable)

    def summary(self):
        counts = {"verified": 0, "pending": 0}
        for step in self.steps():
            m = self.load_marker(step_key(step))
            counts["verified" if m and m["status"] == "verified" else "pending"] += 1
        return counts

    def report_unexpected(self):
        if not self.run_dir.is_dir():
            return
        expected = {str(s) for s in self.steps()}
        for p in sorted(self.run_dir.iterdir()):
            if p.is_dir() and p.name not in expected and (STEP_RE.match(p.name.split(".")[0] or "")):
                if p.name not in self._reported:
                    self._reported.add(p.name)
                    log("VLA_SKIP_DIR", name=p.name,
                        reason="orbax_tmp" if "tmp" in p.name else "not_expected_step")

    def watch(self):
        self._reported = set()
        rounds = 0
        while True:
            rounds += 1
            alive = self.train_alive()
            if not alive:
                self.note_train_end("tmux_not_running")
            self.report_unexpected()
            self.load_start()  # 每轮复核运行身份（run_uuid 漂移即停）
            tally = {}
            for step in self.steps():
                status = self.process(step, "training")
                tally[status] = tally.get(status, 0) + 1
            log("VLA_ROUND", round=rounds, train_alive=alive, **tally)
            if not alive:
                c = self.summary()
                log("VLA_WATCH_END", verified=c["verified"], pending=c["pending"])
                return EXIT_OK if c["pending"] == 0 else EXIT_PENDING
            if self.a.max_rounds and rounds >= self.a.max_rounds:
                log("VLA_WATCH_MAX_ROUNDS", rounds=rounds)
                return EXIT_OK
            time.sleep(self.a.poll_seconds)

    def finalize(self):
        if self.train_alive():
            raise UploaderStop("TRAIN_STILL_RUNNING", f"tmux {self.a.train_tmux} 仍在运行，拒绝 finalize")
        self.note_train_end("finalize_found_tmux_absent")
        self._reported = set()
        self.report_unexpected()
        statuses = {s: self.process(s, "finalize") for s in self.steps()}
        all_steps_verified = all(v == "verified" for v in statuses.values())
        meta_ok = remote_ok = False
        if all_steps_verified:
            problems = self.check_final_record()
            for p in problems:
                log("VLA_META_INVALID", detail=p)
            meta_sources = {}
            if not problems:
                for fname in FINAL_RECORD_FILES:
                    meta_sources[f"{META_PREFIX}/final-record/{fname}"] = self.final_record_dir / fname
                for p in sorted(self.run_dir.iterdir()):
                    if p.is_file() and not p.is_symlink():
                        meta_sources[f"{META_PREFIX}/run-root/{p.name}"] = p
                meta_sources[f"{META_PREFIX}/history/{Path(self.a.history_yaml).name}"] = Path(self.a.history_yaml)
                meta_sources[f"{META_PREFIX}/norm_stats.json"] = Path(self.a.norm_stats)
                meta_sources[f"{META_PREFIX}/motion_store/{Path(self.a.motion_store_meta).name}"] = \
                    Path(self.a.motion_store_meta)
                snaps = {rp: self.snapshot_meta(rp, src) for rp, src in meta_sources.items()}
                summary_path = self.meta_dir / META_PREFIX / "source_summary.json"
                body = self.source_summary(snaps)
                if summary_path.is_file() and json.loads(summary_path.read_text()) != body:
                    raise UploaderStop("VLA_UPLOAD_CONFLICT", "source_summary.json 与上次生成内容不一致")
                common.atomic_write_json(summary_path, body)
                snaps[f"{META_PREFIX}/source_summary.json"] = summary_path
                # SHA256SUMS 覆盖全部上传对象（step 成员、完成清单、元数据）
                lines = []
                for step in self.steps():
                    m = self.load_marker(step_key(step))
                    for rel, mm in sorted(m["members"].items()):
                        lines.append((f"{step}/{rel}", mm["sha256"]))
                    lines.append((f"{MANIFEST_PREFIX}/{step}.json", m["manifest_sha256"]))
                for rp, local in sorted(snaps.items()):
                    lines.append((rp, common.sha256_file(local)[0]))
                sums = self.meta_dir / SUMS_NAME
                common.atomic_write_text(sums, "".join(f"{sha}  {p}\n" for p, sha in sorted(lines)))
                log("VLA_SHA256SUMS", lines=len(lines))
                snaps[SUMS_NAME] = sums
                meta_ok = all(self.upload_meta(rp, local) == "verified" for rp, local in sorted(snaps.items()))
                remote = None
                if meta_ok:
                    try:
                        remote = self.remote_listing(refresh=True)
                    except common.RemoteError as exc:
                        log("VLA_REMOTE_ERROR", step="final_list", detail=str(exc))
                if remote is not None:
                    expected = {}
                    for step in self.steps():
                        m = self.load_marker(step_key(step))
                        for rel, mm in m["members"].items():
                            expected[f"{step}/{rel}"] = mm["bytes"]
                        expected[f"{MANIFEST_PREFIX}/{step}.json"] = m["manifest_bytes"]
                    for rp, local in snaps.items():
                        expected[rp] = os.path.getsize(local)
                    missing = sorted(p for p in expected if p not in remote)
                    wrong = sorted(p for p in expected if p in remote and remote[p] != expected[p])
                    extra = sorted(p for p in remote if p not in expected)
                    remote_ok = not (missing or wrong or extra)
                    log("VLA_REMOTE_SET", expected=len(expected), remote=len(remote),
                        missing=",".join(missing[:5]) or "-", wrong=",".join(wrong[:5]) or "-",
                        extra=",".join(extra[:5]) or "-")
        verified = sum(1 for s in self.steps() if (self.load_marker(step_key(s)) or {}).get("status") == "verified")
        firsts = [m for m in (self.load_marker(step_key(s)) for s in self.steps()) if m and m.get("verified_at")]
        first = min(firsts, key=lambda m: m["verified_at"]) if firsts else None
        during = bool(first and first.get("verified_during_training"))
        passed = all_steps_verified and meta_ok and remote_ok
        if passed and not during:
            log("VLA_UPLOAD_NOTE", note="训练中上传未达成", first_object=first["object"] if first else "-")
        print(f"VLA_UPLOAD={'PASS' if passed else 'FAIL'} objects={len(self.steps())} verified={verified} "
              f"during_training={'true' if during else 'false'} "
              f"first_verified_at={first['verified_at'] if first else 'none'} "
              f"train_end_at={self.state.get('train_end_at') or 'none'}", flush=True)
        return EXIT_OK if passed else EXIT_FAIL


# ----------------------------------------------------------------------------------------------
# scan：只读清单
# ----------------------------------------------------------------------------------------------

def scan_main(a):
    run_dir = Path(a.run_dir)
    if not run_dir.is_dir():
        log("VLA_SCAN_USAGE", error=f"{run_dir} 不存在")
        return EXIT_USAGE
    found = []
    for p in sorted(run_dir.iterdir(), key=lambda q: q.name):
        if not p.is_dir():
            continue
        if not STEP_RE.match(p.name) or p.is_symlink():
            log("VLA_SCAN_SKIP", name=p.name, reason="not_final_step_dir")
            continue
        step = int(p.name)
        ok, reason, ts = commit_status(p)
        if not ok:
            log("VLA_SCAN_STEP", step=step, committed=False, reason=reason)
            continue
        snap, why = snapshot_step(p)
        if snap is None:
            log("VLA_SCAN_STEP", step=step, committed=True, stable=False, reason=why)
            continue
        found.append(step)
        log("VLA_SCAN_STEP", step=step, committed=True, stable=True, commit_ns=ts, files=snap["file_count"],
            bytes=snap["total_bytes"], digest=snap["digest"], empty_dirs=len(snap["empty_dirs"]))
        for rel, m in sorted(snap["members"].items()):
            print(f"VLA_SCAN_MEMBER step={step} bytes={m['bytes']} sha256={m['sha256']} path={rel}", flush=True)
    missing = []
    if a.steps:
        missing = [s for s in expected_steps(a.steps, a.save_interval) if s not in found]
        log("VLA_SCAN_EXPECTED", expected=len(expected_steps(a.steps, a.save_interval)),
            missing=",".join(map(str, missing)) or "-")
    print(f"VLA_SCAN=DONE steps={len(found)} missing={len(missing)}", flush=True)
    return EXIT_OK


# ----------------------------------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------------------------------

DEFAULT_HF_ENV = "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/secrets/hf.env"


def parse_args(argv):
    p = argparse.ArgumentParser(description="mixed8 pair4 VLA Orbax 检查点增量上传（watch / finalize / scan）")
    sub = p.add_subparsers(dest="mode", required=True)
    for mode in ("watch", "finalize"):
        s = sub.add_parser(mode)
        s.add_argument("--run-dir", required=True, help="训练 checkpoint 目录绝对路径（由训练器创建）")
        s.add_argument("--final-record-dir", required=True, help="TRAIN_FINAL_RECORD_DIR 绝对路径")
        s.add_argument("--bucket", required=True, help="HF bucket id")
        s.add_argument("--export-root", required=True, help="marker/清单/回读/日志目录绝对路径")
        s.add_argument("--train-tmux", required=True, help="训练 tmux 会话名（精确匹配）")
        s.add_argument("--run-name", default=None, help="默认取 run 目录名；须等于 start.json run_name")
        s.add_argument("--steps", type=int, default=80000)
        s.add_argument("--save-interval", type=int, default=5000)
        s.add_argument("--hf-env", default=DEFAULT_HF_ENV)
        s.add_argument("--retry-sleep", type=float, default=60.0, help="同一对象两次尝试之间的等待秒数")
        if mode == "watch":
            s.add_argument("--poll-seconds", type=float, default=60.0, help="扫描间隔，必须 >=30")
            s.add_argument("--max-rounds", type=int, default=0, help="0=不限；仅运维/测试用")
        else:
            s.add_argument("--history-yaml", required=True, help="实际 history YAML 绝对路径")
            s.add_argument("--norm-stats", required=True, help="norm_stats.json 绝对路径")
            s.add_argument("--motion-store-meta", required=True, help="motion store 元数据文件绝对路径")
            s.add_argument("--encoder-sha256", required=True, help="encoder final ckpt 的 SHA256（64 位小写十六进制）")
    s = sub.add_parser("scan")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--steps", type=int, default=0, help=">0 时报告预期 step 缺失情况")
    s.add_argument("--save-interval", type=int, default=5000)
    a = p.parse_args(argv)
    errors = []
    if a.mode == "scan":
        if not os.path.isabs(a.run_dir):
            errors.append("--run-dir 必须是绝对路径")
        return a, errors
    keys = ["run_dir", "final_record_dir", "export_root", "hf_env"]
    if a.mode == "finalize":
        keys += ["history_yaml", "norm_stats", "motion_store_meta"]
    for key in keys:
        if not os.path.isabs(getattr(a, key)):
            errors.append(f"--{key.replace('_', '-')} 必须是绝对路径")
    if a.steps < 2 or a.save_interval < 1:
        errors.append("--steps 必须 >=2，--save-interval 必须为正")
    if a.mode == "watch" and (not math.isfinite(a.poll_seconds) or a.poll_seconds < 30):
        errors.append("--poll-seconds 必须 >=30")
    if a.retry_sleep < 0:
        errors.append("--retry-sleep 不能为负")
    if a.mode == "finalize":
        if not SHA_RE.match(a.encoder_sha256 or ""):
            errors.append("--encoder-sha256 必须是 64 位小写十六进制")
        for key in ("history_yaml", "norm_stats", "motion_store_meta"):
            if os.path.isabs(getattr(a, key)) and not Path(getattr(a, key)).is_file():
                errors.append(f"--{key.replace('_', '-')} 文件不存在")
    run_dir, export_root = Path(a.run_dir).resolve(), Path(a.export_root).resolve()
    if export_root == run_dir or run_dir in export_root.parents:
        errors.append("--export-root 不能位于 run 目录内")
    if not run_dir.is_dir():
        errors.append("--run-dir 不存在（必须由训练器创建，不得预建）")
    return a, errors


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    a, errors = parse_args(argv)
    if errors:
        for e in errors:
            log("VLA_UPLOAD_USAGE", error=e)
        return EXIT_USAGE
    if a.mode == "scan":
        try:
            return scan_main(a)
        except UploaderStop as stop:
            log(stop.event, detail=stop.detail)
            return EXIT_STOP
    up = Uploader(a, a.mode)
    try:
        with common.SingleInstanceLock(Path(a.export_root) / "uploader.lock"):
            up.startup()
            return up.watch() if a.mode == "watch" else up.finalize()
    except UploaderStop as stop:
        log(stop.event, detail=stop.detail)
        log("VLA_UPLOAD_STOPPED", reason=stop.event)
        return EXIT_STOP
    except common.RemoteError as exc:
        log("VLA_REMOTE_ERROR", detail=str(exc))
        return EXIT_STOP


if __name__ == "__main__":
    sys.exit(main())
