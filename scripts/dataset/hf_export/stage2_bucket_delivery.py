"""消融交付的上传与匿名流式校验；两者分别在 Aspen 与 GL CPU 执行。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen

import ckpt_upload_common as io


RUNS = {
    f"HongzeFu/robomme-vla-gl-orig100-ctx-s2split-{tag}-b64-80k-v1":
    f"gl-orig100-ctx32x4x4-{tag}-motion-s2split4x256-b64-80k"
    for tag in ("noxyz", "nostate")
}
TOKEN_KEYS = ("HF_TOKEN", "HF_TOKEN_PATH", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")
CHUNK_BYTES = 8 * 1024**2


def need(ok, message):
    if not ok:
        raise ValueError(message)


def plain_path(path):
    path = Path(path).absolute()
    need(not any(p.is_symlink() for p in (path, *path.parents)), "拒绝符号链接")
    return path


def file_info(path):
    digest, size = hashlib.sha256(), 0
    with plain_path(path).open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def load_manifest(path, expected_sha, bucket):
    path = plain_path(path)
    need(path.name == "delivery_manifest.json", "必须使用最终交付清单")
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == expected_sha, "清单SHA不符")
    data = json.loads(raw)
    need(data.get("schema") == "stage2-delivery-manifest-v1", "清单schema不符")
    need(bucket in RUNS and data.get("bucket") == bucket
         and data.get("run_name") == RUNS[bucket], "bucket或run身份不符")
    files = data["files"]
    need(isinstance(files, dict) and bool(files), "清单对象为空")
    for name, info in files.items():
        relative = PurePosixPath(name)
        need(bool(name) and not relative.is_absolute() and "\\" not in name
             and str(relative) == name and all(p not in ("", ".", "..") for p in name.split("/"))
             and all(ord(c) >= 32 and ord(c) != 127 for c in name), "对象路径非法")
        need(isinstance(info, dict) and re.fullmatch(r"[0-9a-f]{64}", str(info.get("sha256")))
             and type(info.get("bytes")) is int and info["bytes"] >= 0, "对象指纹非法")
    need(path.name not in files, "清单不应包含自身")
    return data, {**files, path.name: {"sha256": expected_sha, "bytes": len(raw)}}


def upload(manifest, expected_sha, bucket_id, state, hf_cli):
    data, files = load_manifest(manifest, expected_sha, bucket_id)
    stage = Path(manifest).absolute().parent
    need(not state.is_relative_to(stage), "上传账本不能写入stage")
    # 上传前完整核对冻结副本，不从远端尺寸推断SHA身份。
    for name, expected in files.items():
        need(file_info(stage / name) == expected, f"stage指纹不符: {name}")
    token = os.environ.get("HF_TOKEN", "")
    need(bool(token), "上传要求当前环境提供HF_TOKEN")
    identity = {"schema": "stage2-bucket-upload-v1", "run_name": data["run_name"],
                "bucket": bucket_id, "manifest_sha256": expected_sha}
    owner_path = plain_path(state / "owner.json")
    owner = json.loads(owner_path.read_text()) if owner_path.exists() else None
    need(owner is None or owner == identity, "上传归属冲突")
    bucket = io.HFBucket(bucket_id, token, cache_root=state / "cache", cli=(str(hf_cli),))
    entry = bucket.entry()
    if owner is None:
        need(entry is None, "bucket已存在且无本任务归属记录，拒绝接管")
        io.atomic_write_json(owner_path, identity)
    if entry is None:
        rc, _, _ = bucket._run(["buckets", "create", bucket_id], 600)
        need(rc == 0, "公开bucket创建失败")
        entry = bucket.entry()
    need(entry is not None and entry.get("private") is False, "bucket必须公开")
    for name in sorted(files):
        need(bucket.upload(stage / name, name).ok, f"上传失败: {name}")
    io.atomic_write_json(plain_path(state / "upload_receipt.json"),
                         {**identity, "objects": len(files), "status": "uploaded"})
    print(f"UPLOAD_DONE=PASS objects={len(files)}", flush=True)


def anonymous_files(bucket, hf_cli, home):
    env = dict(os.environ)
    for key in (*TOKEN_KEYS, "HF_HUB_OFFLINE"):
        env.pop(key, None)
    env.update(HF_HOME=str(home), HF_HUB_DISABLE_IMPLICIT_TOKEN="1", CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([str(hf_cli), "buckets", "list", bucket, "-R", "--format", "json"],
                          env=env, capture_output=True, text=True, timeout=600)
    # 不回显子进程输出，避免错误文本意外泄漏上游环境。
    need(proc.returncode == 0, "匿名远端列表请求失败")
    data = json.loads(proc.stdout)
    need(isinstance(data, list), "远端列表不是数组")
    files = {}
    for item in data:
        if item.get("type", "file") == "file":
            need(item["path"] not in files, "远端列表对象重复")
            files[item["path"]] = int(item["size"])
    return files


def resolve_url(bucket, name):
    return f"https://huggingface.co/buckets/{bucket}/resolve/{quote(name, safe='/')}"


def stream_info(bucket, name):
    digest, size = hashlib.sha256(), 0
    # urllib 不读取HF缓存或环境token；请求不携带Authorization。
    with urlopen(Request(resolve_url(bucket, name)), timeout=120) as response:
        while chunk := response.read(CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def verify(manifest, expected_sha, bucket, state, hf_cli, smoke=False):
    report = {"bucket": bucket, "manifest_sha256": expected_sha, "smoke": smoke,
              "objects": 0, "sha_match": 0, "size_equal": 0,
              "missing": [], "extra": [], "size_mismatch": [], "checked": {}}
    verdict = "HF_VERIFY_SMOKE" if smoke else "HF_VERIFY"
    try:
        _, files = load_manifest(manifest, expected_sha, bucket)
        report["objects"] = len(files)
        if smoke:
            need("motion_snapshot.ready.json" in files, "缺根快照小JSON对象")
            selected = {"motion_snapshot.ready.json": files["motion_snapshot.ready.json"]}
        else:
            with tempfile.TemporaryDirectory(prefix="anonymous-", dir=state) as home:
                remote = anonymous_files(bucket, hf_cli, Path(home))
            report["missing"] = sorted(files.keys() - remote.keys())
            report["extra"] = sorted(remote.keys() - files.keys())
            report["size_mismatch"] = sorted(n for n in files.keys() & remote.keys()
                                               if files[n]["bytes"] != remote[n])
            need(not any(report[k] for k in ("missing", "extra", "size_mismatch")),
                 "远端对象集合或尺寸不符")
            selected = files
        for name, expected in sorted(selected.items()):
            actual = stream_info(bucket, name)
            report["checked"][name] = actual
            report["sha_match"] += actual["sha256"] == expected["sha256"]
            report["size_equal"] += actual["bytes"] == expected["bytes"]
            need(actual == expected, f"匿名流式指纹不符: {name}")
        report["status"] = "pass"
    except Exception as exc:
        report.update(status="fail", error_type=type(exc).__name__)
        raise
    finally:
        io.atomic_write_json(plain_path(state / ("verify_smoke_report.json" if smoke else "verify_report.json")), report)
        print(f"{verdict}={report['status'].upper()} objects={len(report['checked'])} "
              f"sha_match={report['sha_match']} size_equal={report['size_equal']} "
              f"missing={len(report['missing'])} extra={len(report['extra'])}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("upload", "verify"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--bucket", choices=tuple(RUNS), required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--hf-cli", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        need(args.hf_cli.is_absolute() and args.hf_cli.is_file()
             and os.access(args.hf_cli, os.X_OK), "hf-cli必须是已安装的绝对可执行路径")
        need(not args.smoke or args.command == "verify", "smoke仅用于匿名校验")
        state = plain_path(args.state_dir)
        need(not state.is_relative_to(args.manifest.absolute().parent), "状态目录不能写入stage")
        state.mkdir(parents=True, exist_ok=True)
        if args.command == "upload":
            upload(args.manifest, args.expected_manifest_sha256, args.bucket, state, args.hf_cli)
        else:
            verify(args.manifest, args.expected_manifest_sha256, args.bucket, state, args.hf_cli, args.smoke)
        return 0
    except Exception as exc:
        # 失败类型足以定位报告；不输出可能含凭据的第三方异常文本。
        print(f"DELIVERY=FAIL error_type={type(exc).__name__}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
