#!/usr/bin/env python3
"""MotionJEPA v10 encoder 检查点增量上传器（1020 计划「上传支线」encoder 部分）。

两种模式：

watch     训练期间运行。每轮（间隔 >=30 s）扫描 run 目录里的 ``checkpoint_epoch_1.pt`` …
          ``checkpoint_epoch_<E>.pt`` 与 ``checkpoint_final.pt``，逐份：
            1. 独立复制快照（不建硬链接）；复制前后核源 SHA/size/mtime 一致，否则视为写入未稳定，
               不计错误、下一轮重扫（用户裁决 F1 情况①）；
            2. CPU 子进程 ``torch.load(map_location='cpu', weights_only=False)``，并用 MotionJEPA
               factory 构建 encoder/decoder，对 EMA 与 live 两套权重 ``strict=True`` 加载；核 epoch、
               arch、global_step==steps_per_epoch×epoch、optimizer 存在、全部张量有限；
               加载失败视为未稳定，合同不符视为错误身份并停上传器；
            3. 上传快照 → 鉴权回读到独立目录比 SHA/bytes → 原子写 verified marker。
          已做快照或已 verified 的对象，源字节再变即 ``ENC_UPLOAD_CONFLICT``，上传器非零退出
          （F1 情况②），不影响训练。每对象训练期最多首试＋2 次重试，失败留在队列（F2）。
          训练 tmux 退出后再做一次完整补扫（含 final），然后退出。
finalize  训练结束后运行。每个未完成对象再给一轮首试＋2 次重试（F2）；上传冻结 config 快照与
          ``SHA256SUMS``（E+1 行）；远端核完整对象集；末行
          ``ENC_UPLOAD=PASS|FAIL objects=<E+1> verified=<n> during_training=<true|false>
          first_verified_at=<UTC> train_end_at=<UTC>``。全部 verified 即 PASS，
          during_training 按首份 verified 时训练是否仍在运行如实标注（F3）。

退出码：0 正常（watch 全部 verified／达到 --max-rounds；finalize PASS）；1 finalize FAIL；
2 停止事件（冲突、身份错误、归属不符、bucket 非私有等）；3 watch 结束时仍有待传对象；64 参数错误。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ckpt_upload_common as common  # noqa: E402
from ckpt_upload_common import UploaderStop, log  # noqa: E402

SCHEMA = "mj-v10-encoder-ckpt-upload/1"
KIND = "encoder"
EXPECTED_ARCH = "wan-latent-v10-st"
FINAL_NAME = "checkpoint_final.pt"
CONFIG_NAME = "config.yaml"
SUMS_NAME = "SHA256SUMS"
ATTEMPTS_PER_PHASE = 3  # 首试＋2 次重试
EXIT_OK, EXIT_FAIL, EXIT_STOP, EXIT_PENDING, EXIT_USAGE = 0, 1, 2, 3, 64
VALIDATE_LOAD_ERROR, VALIDATE_CONTRACT_ERROR = 10, 11

# train.py 的 save_now 分支（逐轮 checkpoint_epoch_N.pt）实际键集；没有 checkpoint_id 等字段。
EPOCH_KEYS = ("epoch", "epoch_completed", "global_step", "samples_seen_global", "arch", "config",
              "val_loss", "val_sigreg_loss", "val_latent_loss", "ema_only", "has_live_weights",
              "encoder", "wan_decoder", "encoder_live", "wan_decoder_live", "optimizer")
# make_checkpoint + atomic_save_checkpoint（checkpoint_final.pt）的完整字段。
FINAL_KEYS = ("arch", "config", "epoch", "epoch_completed", "global_step", "samples_seen_global",
              "checkpoint_id", "ema_only", "has_live_weights", "has_ema_weights",
              "final_eval_complete", "encoder", "wan_decoder", "encoder_live", "wan_decoder_live",
              "optimizer")


# ----------------------------------------------------------------------------------------------
# CPU 校验子进程（``python -I mj_v10_ckpt_upload.py _validate ...``）
# ----------------------------------------------------------------------------------------------

def _emit(result: dict, code: int):
    print("VALIDATE_RESULT " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    sys.exit(code)


def validate_main(argv):
    p = argparse.ArgumentParser(prog="_validate")
    p.add_argument("--mj-src", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--kind", choices=("epoch", "final"), required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--steps-per-epoch", type=int, required=True)
    p.add_argument("--run-name", required=True)
    a = p.parse_args(argv)
    mj_src = Path(a.mj_src).resolve()
    sys.path.insert(0, str(mj_src))
    import torch
    torch.set_num_threads(2)
    from motion_jepa.models import factory
    module_file = Path(factory.__file__).resolve()
    if mj_src not in module_file.parents:
        _emit({"error": f"factory 来自 {module_file}，不在 --mj-src 之下"}, VALIDATE_CONTRACT_ERROR)
    try:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:  # 半文件、损坏、读失败
        _emit({"error": f"torch.load 失败: {type(exc).__name__}: {exc}"[:400]}, VALIDATE_LOAD_ERROR)
    problems = []

    def need(cond, msg):
        if not cond:
            problems.append(msg)

    need(isinstance(ck, dict), "顶层不是 dict")
    if problems:
        _emit({"error": "; ".join(problems)}, VALIDATE_CONTRACT_ERROR)
    keys = EPOCH_KEYS if a.kind == "epoch" else FINAL_KEYS
    missing = [k for k in keys if k not in ck]
    need(not missing, f"缺键 {missing}")
    expected_step = a.steps_per_epoch * a.epoch
    need(ck.get("arch") == EXPECTED_ARCH and factory.ARCH_TAG == EXPECTED_ARCH,
         f"arch={ck.get('arch')} factory={factory.ARCH_TAG}")
    need(ck.get("epoch") == a.epoch, f"epoch={ck.get('epoch')} 期望 {a.epoch}")
    need(ck.get("epoch_completed") is True, f"epoch_completed={ck.get('epoch_completed')}")
    need(ck.get("global_step") == expected_step, f"global_step={ck.get('global_step')} 期望 {expected_step}")
    need(isinstance(ck.get("samples_seen_global"), int) and ck.get("samples_seen_global") > 0,
         "samples_seen_global 非正整数")
    need(ck.get("ema_only") is False, f"ema_only={ck.get('ema_only')}")
    need(ck.get("has_live_weights") is True, f"has_live_weights={ck.get('has_live_weights')}")
    cfg = ck.get("config")
    try:
        cfg_run = cfg["run_name"]
    except Exception:
        cfg_run = None
    need(cfg_run == a.run_name, f"config.run_name={cfg_run} 期望 {a.run_name}")
    opt = ck.get("optimizer")
    need(isinstance(opt, dict) and "state" in opt and "param_groups" in opt, "optimizer 缺失或结构不对")
    if a.kind == "final":
        need(ck.get("has_ema_weights") is True, f"has_ema_weights={ck.get('has_ema_weights')}")
        need(isinstance(ck.get("final_eval_complete"), dict), "final_eval_complete 不是字典")
        need(ck.get("checkpoint_id") == f"{a.run_name}:step:{expected_step}",
             f"checkpoint_id={ck.get('checkpoint_id')}")

    def walk(obj, path, out):
        if torch.is_tensor(obj):
            if (obj.is_floating_point() or obj.is_complex()) and obj.numel() and not torch.isfinite(obj).all():
                out.append(path)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}.{k}", out)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]", out)

    nonfinite = []
    for key in ("encoder", "wan_decoder", "encoder_live", "wan_decoder_live", "optimizer"):
        walk(ck.get(key), key, nonfinite)
    need(not nonfinite, f"非有限张量 {nonfinite[:5]}")
    if not problems:
        # 统计量 buffer 是持久 buffer，strict load 会从 checkpoint 填回；不读 VAE config、不联网。
        factory.load_wan_latent_stats = lambda vae_id: (None, None)
        try:
            enc = factory.build_encoder_from_cfg(cfg, activation_checkpoint=False)
            dec = factory.build_decoder_from_cfg(cfg)
            for enc_key, dec_key in (("encoder", "wan_decoder"), ("encoder_live", "wan_decoder_live")):
                enc.load_state_dict(ck[enc_key], strict=True)
                dec.load_state_dict(ck[dec_key], strict=True)
        except Exception as exc:
            problems.append(f"strict load 失败: {type(exc).__name__}: {exc}"[:400])
    if problems:
        _emit({"error": "; ".join(problems)}, VALIDATE_CONTRACT_ERROR)
    _emit({"ok": True, "epoch": ck["epoch"], "global_step": ck["global_step"],
           "motion_jepa_file": str(module_file)}, 0)


def probe_main(argv):
    p = argparse.ArgumentParser(prog="_probe")
    p.add_argument("--mj-src", required=True)
    a = p.parse_args(argv)
    sys.path.insert(0, str(Path(a.mj_src).resolve()))
    import motion_jepa
    from motion_jepa.models import factory
    print("PROBE " + json.dumps({"motion_jepa": str(Path(motion_jepa.__file__).resolve()),
                                 "factory": str(Path(factory.__file__).resolve()),
                                 "arch": factory.ARCH_TAG}), flush=True)


# ----------------------------------------------------------------------------------------------
# 上传器主体
# ----------------------------------------------------------------------------------------------

class Uploader:
    def __init__(self, args, mode):
        self.a = args
        self.mode = mode
        self.run_dir = Path(args.run_dir)
        self.export_root = Path(args.export_root)
        self.run_name = args.run_name or self.run_dir.name
        self.snap_dir = self.export_root / "snapshots"
        self.readback_dir = self.export_root / "readback"
        self.state_path = self.export_root / "state.json"
        self.ledger = common.ObjectLedger(self.export_root, schema=SCHEMA, run_name=self.run_name,
                                          bucket_id=args.bucket, kind=KIND)
        self.bucket = None
        self.bucket_ready = False
        self.seen = set()          # 本进程里已做过完整源核对（重新哈希/重新校验）的对象
        self.remote_files = None   # 本进程首次需要时列一次远端
        self.state = None

    # --- 通用 ----------------------------------------------------------------------------------
    def objects(self):
        out = [(f"checkpoint_epoch_{n}.pt", "epoch", n) for n in range(1, self.a.epochs + 1)]
        out.append((FINAL_NAME, "final", self.a.epochs))
        return out

    def train_alive(self) -> bool:
        rc = subprocess.run(["tmux", "has-session", "-t", f"={self.a.train_tmux}"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        return rc == 0

    def load_state(self):
        try:
            st = common.read_json(self.state_path)
        except ValueError:
            raise UploaderStop("UPLOAD_MARKER_MISMATCH", "state.json 不是合法 JSON")
        expected = self.ledger.owner_expected()
        if st is None:
            st = dict(expected, train_end_at=None, train_end_note=None)
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
        log("ENC_TRAIN_END_OBSERVED", at=self.state["train_end_at"], note=note)

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

    # --- 源与快照 ------------------------------------------------------------------------------
    def check_source(self, name, marker, force_hash):
        """已有 marker 的对象：源必须仍是 marker 绑定的字节，否则身份冲突（F1 情况②）。"""
        src = self.run_dir / name
        if not src.is_file():
            raise UploaderStop("ENC_UPLOAD_CONFLICT", f"{name} 已快照/已上传但源文件消失")
        ident = common.stat_identity(src)
        same_stat = ident["size"] == marker["source_size"] and ident["mtime_ns"] == marker["source_mtime_ns"]
        if force_hash or not same_stat:
            sha, size = common.sha256_file(src)
            if sha != marker["sha256"] or size != marker["bytes"]:
                raise UploaderStop("ENC_UPLOAD_CONFLICT",
                                   f"{name} 状态={marker['status']} 后源字节变化 marker={marker['sha256'][:16]} "
                                   f"now={sha[:16]} bytes={size}")
            if not same_stat:
                marker = self.ledger.save(name, dict(marker, source_size=ident["size"],
                                                     source_mtime_ns=ident["mtime_ns"]))
                log("ENC_SOURCE_TOUCHED", object=name, note="mtime变化但字节相同")
        return marker

    def take_snapshot(self, name):
        """F1 情况①：复制前后源不一致则返回 None（不计错误，下轮重扫）。"""
        src = self.run_dir / name
        tmp = self.snap_dir / f"{name}.partial"
        pre = common.stat_identity(src)
        copy_sha, copy_bytes = common.copy_with_hash(src, tmp)
        post = common.stat_identity(src)
        src_sha, src_bytes = common.sha256_file(src)
        post2 = common.stat_identity(src)
        stable = (pre == post == post2 and copy_sha == src_sha and copy_bytes == src_bytes == pre["size"])
        if not stable:
            tmp.unlink(missing_ok=True)
            log("ENC_WAIT", object=name, reason="write_unstable")
            return None
        final = self.snap_dir / name
        os.replace(tmp, final)
        return {"sha256": copy_sha, "bytes": copy_bytes, "source_size": pre["size"],
                "source_mtime_ns": pre["mtime_ns"], "snapshot": str(final)}

    def validate(self, path, kind, epoch):
        argv = list(common.IO_PREFIX) + [sys.executable, "-I", str(Path(__file__).resolve()), "_validate",
                                         "--mj-src", self.a.mj_src, "--ckpt", str(path), "--kind", kind,
                                         "--epoch", str(epoch), "--steps-per-epoch", str(self.a.steps_per_epoch),
                                         "--run-name", self.run_name]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2",
                   PYTHONUNBUFFERED="1")
        proc = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        result = {}
        for line in proc.stdout.splitlines():
            if line.startswith("VALIDATE_RESULT "):
                result = json.loads(line[len("VALIDATE_RESULT "):])
        if proc.returncode not in (0, VALIDATE_LOAD_ERROR, VALIDATE_CONTRACT_ERROR):
            # 子进程崩溃（如 OOM）按未稳定处理，下一轮重试；输出尾部写入日志供排查。
            tail = " | ".join(proc.stdout.strip().splitlines()[-3:])[:400]
            return VALIDATE_LOAD_ERROR, {"error": f"校验子进程 rc={proc.returncode} {tail}"}
        return proc.returncode, result

    def ensure_snapshot_bytes(self, name, marker):
        """上传前快照必须存在且 SHA 与 marker 一致；快照丢失时从（已核身份的）源重建。"""
        snap = self.snap_dir / name
        if snap.is_file():
            sha, size = common.sha256_file(snap)
            if sha == marker["sha256"] and size == marker["bytes"]:
                return snap
            snap.unlink()
        rebuilt = self.take_snapshot(name)
        if rebuilt is None or rebuilt["sha256"] != marker["sha256"]:
            raise UploaderStop("ENC_UPLOAD_CONFLICT", f"{name} 重建快照与 marker 不一致")
        return snap

    # --- 单对象状态机 ---------------------------------------------------------------------------
    def process(self, name, kind, epoch, phase):
        """返回 absent / unstable / verified / pending。"""
        src = self.run_dir / name
        marker = self.ledger.load(name)
        first = name not in self.seen
        if marker is None:
            if not src.is_file():
                return "absent"
            snap = self.take_snapshot(name)
            if snap is None:
                return "unstable"
            rc, res = self.validate(snap["snapshot"], kind, epoch)
            if rc == VALIDATE_LOAD_ERROR:
                Path(snap["snapshot"]).unlink(missing_ok=True)
                log("ENC_WAIT", object=name, reason="load_failed", detail=res.get("error", ""))
                return "unstable"
            if rc != 0:
                raise UploaderStop("ENC_UPLOAD_IDENTITY", f"{name} 合同不符: {res.get('error', '')}")
            marker = self.ledger.save(name, dict(snap, status="snapshot", kind=kind, epoch=epoch,
                                                 global_step=res["global_step"], validated_at=common.utc_now(),
                                                 attempts={"training": 0, "finalize": 0}))
            log("ENC_SNAPSHOT", object=name, sha256=marker["sha256"], bytes=marker["bytes"],
                global_step=marker["global_step"])
            self.seen.add(name)
        else:
            marker = self.check_source(name, marker, force_hash=first)
            if first and marker["status"] != "verified":
                # 重启补扫走完整检查：快照字节 + CPU strict-load 重新校验。
                snap = self.ensure_snapshot_bytes(name, marker)
                rc, res = self.validate(snap, kind, epoch)
                if rc != 0:
                    raise UploaderStop("ENC_UPLOAD_IDENTITY", f"{name} 重启复核失败: {res.get('error', '')}")
            self.seen.add(name)

        if marker["status"] == "verified":
            if first:
                try:
                    remote = self.remote_listing()
                except common.RemoteError as exc:
                    self.seen.discard(name)  # 下轮重新做完整核对
                    log("ENC_REMOTE_ERROR", object=name, step="list", detail=str(exc))
                    return "verified"
                if name not in remote:
                    marker = self.ledger.save(name, dict(marker, status="snapshot"))
                    log("ENC_REMOTE_MISSING", object=name, action="requeue")
                elif remote[name] != marker["bytes"]:
                    raise UploaderStop("ENC_UPLOAD_CONFLICT",
                                       f"{name} 远端 bytes={remote[name]} 与 marker {marker['bytes']} 不符")
                else:
                    return "verified"
            else:
                return "verified"
        return self.transfer(name, marker, phase)

    def transfer(self, name, marker, phase):
        attempts = dict(marker.get("attempts") or {"training": 0, "finalize": 0})
        if attempts.get(phase, 0) >= ATTEMPTS_PER_PHASE:
            return "pending"
        while attempts.get(phase, 0) < ATTEMPTS_PER_PHASE:
            attempts[phase] = attempts.get(phase, 0) + 1
            marker = self.ledger.save(name, dict(marker, attempts=attempts))
            n = attempts[phase]
            try:
                self.ensure_bucket()
            except common.RemoteError as exc:
                # 网络/CLI 查询失败计为本对象一次失败尝试，不停上传器、不影响训练。
                marker = self.ledger.save(name, dict(marker, last_error=f"remote {exc}"))
                log("ENC_UPLOAD_RETRY", object=name, phase=phase, attempt=n, step="bucket", detail=str(exc))
                if n < ATTEMPTS_PER_PHASE:
                    self.sleep_retry()
                continue
            marker = self.check_source(name, marker, force_hash=True)
            snap = self.ensure_snapshot_bytes(name, marker)
            if marker["status"] == "snapshot":
                res = self.bucket.upload(snap, name)
                if not res.ok:
                    marker = self.ledger.save(name, dict(marker, last_error=f"upload rc={res.rc} {res.detail}"))
                    log("ENC_UPLOAD_RETRY", object=name, phase=phase, attempt=n, step="upload", rc=res.rc)
                    if n < ATTEMPTS_PER_PHASE:
                        self.sleep_retry()
                    continue
                marker = self.check_source(name, marker, force_hash=True)
                marker = self.ledger.save(name, dict(marker, status="uploaded", uploaded_at=common.utc_now(),
                                                     last_error=None))
                log("ENC_UPLOADED", object=name, bytes=marker["bytes"], sha256=marker["sha256"])
            ok, detail = self.readback(name, marker)
            if not ok:
                status = "snapshot" if detail.startswith("mismatch") else "uploaded"
                marker = self.ledger.save(name, dict(marker, status=status, last_error=f"readback {detail}"))
                log("ENC_READBACK_FAIL", object=name, phase=phase, attempt=n, detail=detail)
                if n < ATTEMPTS_PER_PHASE:
                    self.sleep_retry()
                continue
            alive = self.train_alive()
            marker = self.ledger.save(name, dict(marker, status="verified", verified_at=common.utc_now(),
                                                 verified_during_training=alive, last_error=None))
            Path(snap).unlink(missing_ok=True)
            if self.remote_files is not None:
                self.remote_files[name] = marker["bytes"]
            log("ENC_VERIFIED", object=name, sha256=marker["sha256"], bytes=marker["bytes"],
                during_training=alive, attempt=n, phase=phase)
            return "verified"
        log("ENC_UPLOAD_GIVEUP", object=name, phase=phase, attempts=ATTEMPTS_PER_PHASE,
            last_error=marker.get("last_error", ""))
        return "pending"

    def readback(self, name, marker):
        """鉴权下载到独立回读目录，比 SHA/bytes；回读文件随即删除。"""
        target = self.readback_dir / f"{name}.{os.getpid()}"
        target.unlink(missing_ok=True)
        try:
            res = self.bucket.download(name, target)
            if not res.ok:
                return False, f"download rc={res.rc} {res.detail}"
            if not target.is_file():
                return False, "download 未产生文件"
            sha, size = common.sha256_file(target)
            if sha != marker["sha256"] or size != marker["bytes"]:
                return False, f"mismatch sha={sha[:16]} bytes={size}"
            return True, "ok"
        finally:
            target.unlink(missing_ok=True)

    # --- 元数据（finalize） -------------------------------------------------------------------
    def upload_meta(self, name, local_bytes_path):
        """config 快照 / SHA256SUMS：同样快照 → 上传 → 鉴权回读 → verified marker。"""
        sha, size = common.sha256_file(local_bytes_path)
        marker = self.ledger.load(name)
        if marker is not None and marker["sha256"] != sha:
            if marker["status"] == "verified":
                raise UploaderStop("ENC_UPLOAD_CONFLICT", f"{name} 已 verified 但内容变化")
            marker = None
        if marker is not None and marker["status"] == "verified":
            remote = self.remote_listing()
            if remote.get(name) == size:
                return "verified"
            marker = self.ledger.save(name, dict(marker, status="snapshot"))
        if marker is None:
            marker = self.ledger.save(name, {"status": "snapshot", "kind": "meta", "sha256": sha, "bytes": size,
                                             "snapshot": str(local_bytes_path), "source_size": None,
                                             "source_mtime_ns": None,
                                             "attempts": {"training": 0, "finalize": 0}})
        attempts = dict(marker.get("attempts") or {})
        while attempts.get("finalize", 0) < ATTEMPTS_PER_PHASE:
            attempts["finalize"] = attempts.get("finalize", 0) + 1
            marker = self.ledger.save(name, dict(marker, attempts=attempts))
            try:
                self.ensure_bucket()
            except common.RemoteError as exc:
                log("ENC_UPLOAD_RETRY", object=name, phase="finalize", step="bucket", detail=str(exc))
                self.sleep_retry()
                continue
            if marker["status"] == "snapshot":
                res = self.bucket.upload(local_bytes_path, name)
                if not res.ok:
                    log("ENC_UPLOAD_RETRY", object=name, phase="finalize", attempt=attempts["finalize"], rc=res.rc)
                    self.sleep_retry()
                    continue
                marker = self.ledger.save(name, dict(marker, status="uploaded", uploaded_at=common.utc_now()))
            ok, detail = self.readback(name, marker)
            if not ok:
                status = "snapshot" if detail.startswith("mismatch") else "uploaded"
                marker = self.ledger.save(name, dict(marker, status=status, last_error=f"readback {detail}"))
                log("ENC_READBACK_FAIL", object=name, phase="finalize", detail=detail)
                self.sleep_retry()
                continue
            self.ledger.save(name, dict(marker, status="verified", verified_at=common.utc_now()))
            if self.remote_files is not None:
                self.remote_files[name] = size
            log("ENC_VERIFIED", object=name, sha256=sha, bytes=size, phase="finalize")
            return "verified"
        log("ENC_UPLOAD_GIVEUP", object=name, phase="finalize", attempts=ATTEMPTS_PER_PHASE)
        return "pending"

    # --- 入口 ---------------------------------------------------------------------------------
    def startup(self):
        for d in (self.export_root.parent, self.export_root):
            if d.is_symlink():
                raise UploaderStop("EXPORT_ROOT_SYMLINK", str(d))
        self.export_root.mkdir(parents=True, exist_ok=True)
        self.snap_dir.mkdir(exist_ok=True)
        self.readback_dir.mkdir(exist_ok=True)
        self.ledger.check_all()
        self.load_state()
        proc = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "_probe",
                               "--mj-src", self.a.mj_src],
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1"),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        probe = None
        for line in proc.stdout.splitlines():
            if line.startswith("PROBE "):
                probe = json.loads(line[len("PROBE "):])
        if proc.returncode != 0 or probe is None:
            raise UploaderStop("MJ_SRC_IMPORT_FAIL", proc.stdout[-400:])
        mj_src = Path(self.a.mj_src).resolve()
        if mj_src not in Path(probe["motion_jepa"]).parents or probe["arch"] != EXPECTED_ARCH:
            raise UploaderStop("MJ_SRC_MISMATCH", json.dumps(probe))
        log("ENC_UPLOAD_START", mode=self.mode, run=self.run_name, bucket=self.a.bucket,
            run_dir=self.run_dir, export_root=self.export_root, objects=self.a.epochs + 1,
            steps_per_epoch=self.a.steps_per_epoch, motion_jepa_file=probe["motion_jepa"],
            arch=probe["arch"], python=sys.executable)

    def summary(self):
        counts = {"verified": 0, "pending": 0}
        for name, _, _ in self.objects():
            m = self.ledger.load(name)
            counts["verified" if m and m["status"] == "verified" else "pending"] += 1
        return counts

    def watch(self):
        rounds = 0
        while True:
            rounds += 1
            alive = self.train_alive()
            if not alive:
                self.note_train_end("tmux_not_running")
            tally = {}
            for name, kind, epoch in self.objects():
                status = self.process(name, kind, epoch, "training")
                tally[status] = tally.get(status, 0) + 1
            log("ENC_ROUND", round=rounds, train_alive=alive, **tally)
            if not alive:
                c = self.summary()
                log("ENC_WATCH_END", verified=c["verified"], pending=c["pending"])
                return EXIT_OK if c["pending"] == 0 else EXIT_PENDING
            if self.a.max_rounds and rounds >= self.a.max_rounds:
                log("ENC_WATCH_MAX_ROUNDS", rounds=rounds)
                return EXIT_OK
            time.sleep(self.a.poll_seconds)

    def finalize(self):
        if self.train_alive():
            raise UploaderStop("TRAIN_STILL_RUNNING", f"tmux {self.a.train_tmux} 仍在运行，拒绝 finalize")
        self.note_train_end("finalize_found_tmux_absent")
        statuses = {}
        for name, kind, epoch in self.objects():
            statuses[name] = self.process(name, kind, epoch, "finalize")
        ckpt_names = [name for name, _, _ in self.objects()]
        all_ckpt_verified = all(statuses[n] == "verified" for n in ckpt_names)
        meta_ok = False
        remote_ok = False
        if all_ckpt_verified:
            meta_dir = self.export_root / "meta"
            meta_dir.mkdir(exist_ok=True)
            cfg_src = self.run_dir / CONFIG_NAME
            cfg_ok = cfg_src.is_file()
            if cfg_ok:
                cfg_snap = meta_dir / CONFIG_NAME
                if not cfg_snap.is_file():
                    common.copy_with_hash(cfg_src, cfg_snap)
                elif common.sha256_file(cfg_snap)[0] != common.sha256_file(cfg_src)[0]:
                    raise UploaderStop("ENC_UPLOAD_CONFLICT", "冻结 config 快照与 run 目录 config.yaml 不一致")
            else:
                log("ENC_META_MISSING", object=CONFIG_NAME)
            lines = []
            for name in ckpt_names:
                m = self.ledger.load(name)
                lines.append(f"{m['sha256']}  {name}\n")
            sums = meta_dir / SUMS_NAME
            common.atomic_write_text(sums, "".join(lines))
            log("ENC_SHA256SUMS", lines=len(lines))
            meta_ok = cfg_ok and all(self.upload_meta(n, meta_dir / n) == "verified" for n in (CONFIG_NAME, SUMS_NAME))
            remote = None
            if meta_ok:
                try:
                    remote = self.remote_listing(refresh=True)
                except common.RemoteError as exc:
                    log("ENC_REMOTE_ERROR", step="final_list", detail=str(exc))
            if remote is not None:
                expected = {n: self.ledger.load(n)["bytes"] for n in ckpt_names + [CONFIG_NAME, SUMS_NAME]}
                missing = sorted(n for n in expected if n not in remote)
                wrong = sorted(n for n in expected if n in remote and remote[n] != expected[n])
                extra = sorted(n for n in remote if n not in expected)
                remote_ok = not (missing or wrong or extra)
                log("ENC_REMOTE_SET", expected=len(expected), remote=len(remote), missing=",".join(missing) or "-",
                    wrong=",".join(wrong) or "-", extra=",".join(extra) or "-")
        verified = sum(1 for n in ckpt_names if (self.ledger.load(n) or {}).get("status") == "verified")
        firsts = [m for m in (self.ledger.load(n) for n in ckpt_names) if m and m.get("verified_at")]
        first = min(firsts, key=lambda m: m["verified_at"]) if firsts else None
        during = bool(first and first.get("verified_during_training"))
        passed = all_ckpt_verified and meta_ok and remote_ok
        if passed and not during:
            log("ENC_UPLOAD_NOTE", note="训练中上传未达成", first_object=first["object"] if first else "-")
        print(f"ENC_UPLOAD={'PASS' if passed else 'FAIL'} objects={len(ckpt_names)} verified={verified} "
              f"during_training={'true' if during else 'false'} "
              f"first_verified_at={first['verified_at'] if first else 'none'} "
              f"train_end_at={self.state.get('train_end_at') or 'none'}", flush=True)
        return EXIT_OK if passed else EXIT_FAIL


def parse_args(argv):
    p = argparse.ArgumentParser(description="MotionJEPA v10 encoder 检查点增量上传（watch / finalize）")
    sub = p.add_subparsers(dest="mode", required=True)
    for mode in ("watch", "finalize"):
        s = sub.add_parser(mode)
        s.add_argument("--run-dir", required=True, help="训练 run 目录绝对路径（由训练器创建）")
        s.add_argument("--bucket", required=True, help="HF bucket id，如 HongzeFu/motionjepa-v10-...")
        s.add_argument("--export-root", required=True, help="marker/快照/回读/日志目录绝对路径")
        s.add_argument("--train-tmux", required=True, help="训练 tmux 会话名（精确匹配）")
        s.add_argument("--mj-src", required=True, help="MotionJEPA 只读执行副本的 src 绝对路径")
        s.add_argument("--run-name", default=None, help="默认取 run 目录名；须等于 ckpt config.run_name")
        s.add_argument("--epochs", type=int, default=60)
        s.add_argument("--steps-per-epoch", type=int, default=314)
        s.add_argument("--hf-env", default="/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/secrets/hf.env")
        s.add_argument("--retry-sleep", type=float, default=60.0, help="同一对象两次尝试之间的等待秒数")
        if mode == "watch":
            s.add_argument("--poll-seconds", type=float, default=60.0, help="扫描间隔，必须 >=30")
            s.add_argument("--max-rounds", type=int, default=0, help="0=不限；仅运维/测试用")
    a = p.parse_args(argv)
    errors = []
    for key in ("run_dir", "export_root", "mj_src", "hf_env"):
        if not os.path.isabs(getattr(a, key)):
            errors.append(f"--{key.replace('_', '-')} 必须是绝对路径")
    if a.epochs < 1 or a.steps_per_epoch < 1:
        errors.append("--epochs/--steps-per-epoch 必须为正")
    if a.mode == "watch" and (not math.isfinite(a.poll_seconds) or a.poll_seconds < 30):
        errors.append("--poll-seconds 必须 >=30")
    if a.retry_sleep < 0:
        errors.append("--retry-sleep 不能为负")
    run_dir, export_root = Path(a.run_dir).resolve(), Path(a.export_root).resolve()
    if export_root == run_dir or run_dir in export_root.parents:
        errors.append("--export-root 不能位于 run 目录内")
    if not run_dir.is_dir():
        errors.append("--run-dir 不存在（必须由训练器创建，不得预建）")
    return a, errors


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_validate":
        return validate_main(argv[1:])
    if argv and argv[0] == "_probe":
        return probe_main(argv[1:])
    a, errors = parse_args(argv)
    if errors:
        for e in errors:
            log("ENC_UPLOAD_USAGE", error=e)
        return EXIT_USAGE
    up = Uploader(a, a.mode)
    try:
        with common.SingleInstanceLock(Path(a.export_root) / "uploader.lock"):
            up.startup()
            return up.watch() if a.mode == "watch" else up.finalize()
    except UploaderStop as stop:
        log(stop.event, detail=stop.detail)
        log("ENC_UPLOAD_STOPPED", reason=stop.event)
        return EXIT_STOP
    except common.RemoteError as exc:
        # 远端查询失败（建 bucket/列目录）：不改训练，非零退出交主会话处置。
        log("ENC_REMOTE_ERROR", detail=str(exc))
        return EXIT_STOP


if __name__ == "__main__":
    sys.exit(main())
