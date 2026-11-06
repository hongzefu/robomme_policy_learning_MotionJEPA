#!/usr/bin/env python3
"""context 1024 档 checkpoint 增量上传：Orbax 提交一份传一份，逐文件匿名回读核 sha256。

bucket 内对象沿用既有 VLA 导出的 ``<step>/<checkpoint 内相对路径>``，下载后即标准 step 目录。
重启安全：每份传完写 ``$STORE/exports/<run>-inc/logs/uploaded/<step>.done``，已完成的不再传。

用法：ctx1024_ckpt_inc.py --run <run> --bucket <id> [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import ckpt_upload_common as io

STORE = Path("/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store")
STEPS = [*range(10000, 80000, 10000), 79999]
POLL_SECONDS = 60
RETRIES = 3
RETRY_SECONDS = 60


class UploadFail(Exception):
    pass


class AnonymousBucket(io.HFBucket):
    """回读不带凭据，同时证明 bucket 确实公开。"""

    def _env(self):
        env = super()._env()
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN_PATH"):
            env.pop(name, None)
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        env["HF_HOME"] = str(self._cache_root / "anonymous-home")
        return env


def committed(step_dir: Path) -> bool:
    try:
        meta = io.read_json(step_dir / "_CHECKPOINT_METADATA")
    except ValueError:
        return False
    return isinstance(meta, dict) and bool(meta.get("commit_timestamp_nsecs"))


def step_files(run_dir: Path, step: int) -> dict[str, Path]:
    root = run_dir / str(step)
    return {f"{step}/{p.relative_to(root).as_posix()}": p for p in sorted(root.rglob("*")) if p.is_file()}


def ensure_public_bucket(bucket: io.HFBucket) -> None:
    entry = bucket.entry()
    if entry is None:
        rc, _, err = bucket._run(["buckets", "create", bucket.bucket_id], 600)
        if rc != 0:
            raise UploadFail(f"bucket 创建失败 rc={rc} {bucket._clean(err)}")
        entry = bucket.entry()
    if entry is None or entry.get("private") is not False:
        raise UploadFail(f"bucket 必须公开: {bucket.bucket_id}")


def transfer_file(bucket, anon, name, source: Path, local: Path) -> dict:
    """上传并匿名回读一个文件；传输失败与 sha256／字节数不符都重试。"""
    sha, size = io.sha256_file(source)
    for attempt in range(RETRIES + 1):
        if attempt:
            time.sleep(RETRY_SECONDS)
        result = bucket.upload(source, name)
        if not result.ok:
            detail = f"upload rc={result.rc} {result.detail}"
            continue
        result = anon.download(name, local)
        if not result.ok:
            detail = f"readback rc={result.rc} {result.detail}"
            continue
        actual = io.sha256_file(local)
        local.unlink()
        if actual == (sha, size):
            return {"sha256": sha, "bytes": size}
        detail = "readback sha256 或字节数不符"
    raise UploadFail(f"object={name} attempts={RETRIES + 1} {detail}")


def upload_step(bucket, anon, run_dir: Path, step: int, verify_dir: Path) -> dict:
    return {name: transfer_file(bucket, anon, name, source, verify_dir / "readback.bin")
            for name, source in step_files(run_dir, step).items()}


def verify_remote(anon, bucket_id: str, sizes: dict[str, int]) -> int:
    """逐文件 sha256 已在上传时核过，这里只核最终文件集合与字节数。"""
    remote = anon.list_files()
    mismatches = sum(remote.get(name) != size for name, size in sizes.items())
    mismatches += sum(name not in sizes for name in remote)
    entry = anon.entry()
    if entry is None or entry.get("private") is not False:
        raise UploadFail(f"匿名复核 bucket 非公开: {bucket_id}")
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--dry-run", action="store_true", help="只列已提交 step 的文件，不读凭据、不联网")
    args = parser.parse_args()
    run_dir = args.store / "train-runs/mme_vla_suite" / args.run

    if args.dry_run:
        for step in STEPS:
            if not committed(run_dir / str(step)):
                print(f"DRYRUN_NOT_COMMITTED step={step}")
                continue
            for name, path in step_files(run_dir, step).items():
                print(f"DRYRUN_FILE {name} bytes={path.stat().st_size}")
        return 0

    export = args.store / "exports" / f"{args.run}-inc"
    markers = export / "logs/uploaded"
    verify_dir = export / "verify"
    markers.mkdir(parents=True, exist_ok=True)
    verify_dir.mkdir(parents=True, exist_ok=True)
    step = "-"
    try:
        token = io.load_hf_token(args.store / "secrets/hf.env")
        bucket = io.HFBucket(args.bucket, token, cache_root=export / "cache")
        anon = AnonymousBucket(args.bucket, "", cache_root=export / "cache")
        ensure_public_bucket(bucket)
        print(f"INC_START run={args.run} bucket={args.bucket}", flush=True)

        sizes: dict[str, int] = {}
        pending = list(STEPS)
        while pending:
            for step in list(pending):
                marker = markers / f"{step}.done"
                record = io.read_json(marker)
                if record is None:
                    if not committed(run_dir / str(step)):
                        continue
                    files = upload_step(bucket, anon, run_dir, step, verify_dir)
                    record = {"run": args.run, "bucket": args.bucket, "step": step, "files": files}
                    io.atomic_write_json(marker, record)
                    print(f"INC_UPLOADED step={step}", flush=True)
                elif record.get("run") != args.run or record.get("bucket") != args.bucket:
                    raise UploadFail(f"上传标记属于其他 run/bucket: {marker}")
                sizes.update({name: info["bytes"] for name, info in record["files"].items()})
                pending.remove(step)
            step = "-"
            if pending:
                time.sleep(POLL_SECONDS)

        mismatches = verify_remote(anon, args.bucket, sizes)
    except (UploadFail, io.UploaderStop, io.RemoteError, OSError, ValueError) as exc:
        print(f"UPLOAD_FAIL step={step} {exc}", flush=True)
        return 1
    status = "PASS" if mismatches == 0 else "FAIL"
    print(f"HF_VERIFY={status} bucket={args.bucket} steps={len(STEPS)} mismatches={mismatches}", flush=True)
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
