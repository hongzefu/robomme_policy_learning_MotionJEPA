#!/usr/bin/env python3
"""按 validate_stage2.py::gate_g3 的原样抽样生成并下载 G3 的 32 窗子集。

先取 build_spec 与其四份索引并核 SHA；只读所选片的 DONE.json 得到精确字节数，
不下载 latent。fetch 按 bucket 的 SHA256SUMS 核全部所需对象，支持有限重试与断点续传。
必须显式指定 --root（正式落点为 $P/v1-store/external/rgb-gtxyz-g3-subset）。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pyarrow.parquet as pq


BUCKET = "HongzeFu/robomme-16task-rgb-gtxyz-stride2-16views-fp16-v1"
BASE_URL = f"https://huggingface.co/buckets/{BUCKET}/resolve/"
SPEC = "control/build_spec.json"
SUMS = "SHA256SUMS"
INDEX = ("index/episodes.parquet", "index/source_valid_episodes.json",
         "index/state_episodes.json", "index/windows.parquet")
SHARD_FILES = ("rgb.raw.f16.bin", "gtxyz.raw.f16.bin", "common_geometry.bits",
               "rows.parquet", "params.parquet")
CHUNK = 8 * 1024 * 1024


def relative(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or str(path) != name):
        raise ValueError(f"非法对象路径：{name!r}")
    return path


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def read_remote(base_url, name):
    with urllib.request.urlopen(base_url + urllib.parse.quote(str(relative(name)), safe="/"), timeout=60) as response:
        return response.read()


def read_sums(raw):
    sums = {}
    for line in raw.decode("utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip(" *")
        relative(name)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or name in sums:
            raise ValueError("SHA256SUMS 格式错误或对象重复")
        sums[name] = digest
    return sums


def checked(raw, digest, name):
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"SHA256 不符：{name}")
    return raw


def save(root, name, raw):
    target = root / str(relative(name))
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(raw)
    os.replace(part, target)


def select_windows(windows):
    # 与 validate_stage2.py::gate_g3 的候选排序及抽样表达式保持逐字相同。
    reader = argparse.Namespace(windows=windows)
    candidates = sorted(key for key in reader.windows if reader.windows[key]["split"] == "val")
    if len(candidates) < 32:
        raise ValueError("val 窗口不足32")
    selected = np.random.default_rng(0).choice(candidates, 32, replace=False).tolist()
    return selected


def cmd_plan(args):
    root = args.root
    sums_raw = read_remote(args.base_url, SUMS)
    sums = read_sums(sums_raw)
    spec_raw = checked(read_remote(args.base_url, SPEC), sums[SPEC], SPEC)
    spec = json.loads(spec_raw)
    if spec["schema_version"] != 2 or spec["rows_per_shard"] != 4096 or set(spec["index_files"]) != set(INDEX):
        raise ValueError("G3 要求 schema 2、4096 行分片及四份索引")
    objects = {SUMS: dict(bytes=len(sums_raw), sha256=hashlib.sha256(sums_raw).hexdigest()),
               SPEC: dict(bytes=len(spec_raw), sha256=sums[SPEC])}
    save(root, SUMS, sums_raw)
    save(root, SPEC, spec_raw)
    for name in INDEX:
        meta = spec["index_files"][name]
        if meta["sha256"] != sums[name]:
            raise ValueError(f"build_spec 与 SHA256SUMS 不一致：{name}")
        download(root, args.base_url, dict(path=name, **meta), args.retries, args.timeout)
        objects[name] = meta
    table = pq.ParquetFile(root / "index/windows.parquet").read(
        columns=["window_key", "task", "episode", "split", "row_start"], use_threads=False)
    windows = {row["window_key"]: row for row in table.to_pylist()}
    selected = select_windows(windows)
    shards = sorted({int(windows[key]["row_start"]) // 4096 for key in selected})
    source = json.loads((root / "index/source_valid_episodes.json").read_bytes())
    episodes = {(entry["task"], entry["episode"]): entry for entry in source["episodes"]}
    for key in selected:
        window = windows[key]
        entry = episodes[window["task"], window["episode"]]
        name = f"masks/source_valid/{window['task']}/ep{window['episode']:03d}.bits"
        if entry["file"] != name or entry["sha256"] != sums[name]:
            raise ValueError(f"源掩码索引与 SHA256SUMS 不一致：{name}")
        objects[name] = dict(bytes=entry["bytes"], sha256=sums[name])
    for sid in shards:
        prefix = f"shards/{sid:06d}/"
        name = prefix + "DONE.json"
        raw = checked(read_remote(args.base_url, name), sums[name], name)
        done = json.loads(raw)
        objects[name] = dict(bytes=len(raw), sha256=sums[name])
        save(root, name, raw)
        for filename in SHARD_FILES:
            name = prefix + filename
            meta = done["files"][filename]
            if meta["sha256"] != sums[name]:
                raise ValueError(f"DONE 与 SHA256SUMS 不一致：{name}")
            objects[name] = dict(bytes=meta["bytes"], sha256=sums[name])
    plan = dict(schema="g3-subset-v1", bucket=BUCKET, base_url=args.base_url,
                windows=selected, shards=shards,
                objects=[dict(path=name, **objects[name]) for name in sorted(objects)])
    plan["total_bytes"] = sum(obj["bytes"] for obj in plan["objects"])
    output = args.plan or root / "g3-subset-plan.json"
    save(output.parent, output.name, (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode())
    print(f"G3_SUBSET_PLAN=PASS windows=32 shards={len(shards)} bytes={plan['total_bytes']}", flush=True)
    return 0


def download(root, base_url, obj, retries, timeout):
    """只在核 SHA 和长度通过后改名，网络失败保留 .part 供下次续传。"""
    target = root / str(relative(obj["path"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == obj["bytes"] and sha_file(target) == obj["sha256"]:
        return
    part = target.with_name(target.name + ".part")
    for attempt in range(retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            if have > obj["bytes"]:
                part.unlink()
                have = 0
            if have < obj["bytes"] or not part.exists():
                request = urllib.request.Request(base_url + urllib.parse.quote(obj["path"], safe="/"))
                if have:
                    request.add_header("Range", f"bytes={have}-")
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    mode = "ab" if have and response.status == 206 else "wb"
                    with part.open(mode) as handle:
                        for block in iter(lambda: response.read(CHUNK), b""):
                            handle.write(block)
            if part.stat().st_size != obj["bytes"] or sha_file(part) != obj["sha256"]:
                part.unlink()
                raise ValueError(f"SHA256 或字节数不符：{obj['path']}")
            os.replace(part, target)
            return
        except (OSError, urllib.error.URLError, ValueError) as exc:
            if attempt == retries:
                raise
            print(f"G3_SUBSET_RETRY path={obj['path']} attempt={attempt + 1} reason={exc}", flush=True)
            time.sleep(min(30, 2 ** attempt))


def cmd_fetch(args):
    plan = json.loads((args.plan or args.root / "g3-subset-plan.json").read_bytes())
    objects = plan["objects"]
    if len({obj["path"] for obj in objects}) != len(objects):
        raise ValueError("计划包含重复对象，不能并发写同一文件")
    sums_obj = next(obj for obj in objects if obj["path"] == SUMS)
    download(args.root, plan["base_url"], sums_obj, args.retries, args.timeout)
    sums = read_sums((args.root / SUMS).read_bytes())
    # 全部身份先核完，再启动下载；SUMS 已单线程核过，不重复派发。
    for obj in objects:
        if obj["path"] != SUMS and sums[obj["path"]] != obj["sha256"]:
            raise ValueError(f"计划与 bucket SHA256SUMS 不一致：{obj['path']}")
    total = sum(obj["bytes"] for obj in objects)
    completed, completed_bytes = 1, sums_obj["bytes"]
    print(f"G3_FETCH_PROGRESS objects={completed}/{len(objects)} bytes={completed_bytes}/{total}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(download, args.root, plan["base_url"], obj, args.retries, args.timeout): obj
                   for obj in objects if obj["path"] != SUMS}
        for future in as_completed(pending):
            future.result()
            completed += 1
            completed_bytes += pending[future]["bytes"]
            print(f"G3_FETCH_PROGRESS objects={completed}/{len(objects)} bytes={completed_bytes}/{total}", flush=True)
    print(f"G3_SUBSET=PASS shards={len(plan['shards'])} bytes={total} sha_mismatch=0", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "fetch"))
    parser.add_argument("--root", required=True, type=Path, help="子集落点，必须显式指定")
    parser.add_argument("--plan", type=Path, help="清单路径，默认 <root>/g3-subset-plan.json")
    parser.add_argument("--base-url", default=BASE_URL, help="plan 使用的匿名对象地址前缀")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--workers", type=int, default=1, help="fetch 的对象并发数，默认单线程")
    args = parser.parse_args(argv)
    if args.retries < 0 or args.timeout <= 0 or args.workers < 1:
        parser.error("重试次数须非负、超时与并发数须为正")
    args.base_url = args.base_url.rstrip("/") + "/"
    # Aspen 的 IPv6 不可达；只改变本进程的解析，与既有 orig100 拉取工具一致。
    original = socket.getaddrinfo
    socket.getaddrinfo = lambda host, port, family=0, *a, **kw: original(host, port, socket.AF_INET, *a, **kw)
    try:
        return {"plan": cmd_plan, "fetch": cmd_fetch}[args.command](args)
    except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
        print(f"G3_SUBSET{'_PLAN' if args.command == 'plan' else ''}=FAIL reason={exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        socket.getaddrinfo = original


if __name__ == "__main__":
    raise SystemExit(main())
