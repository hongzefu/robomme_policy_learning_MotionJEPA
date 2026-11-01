#!/usr/bin/env python3
"""从 HF bucket 匿名拉取 orig100 训练库的 K7 子集、核 SHA、解包并复核（1031 GL 复现计划 S3）。

只依赖标准库，可在 `uvx --python 3.11 --from 'huggingface_hub[cli]==1.30.0' python <本脚本>` 下运行；
不读取任何凭据，全程匿名 HTTP（bucket 对外可读，`resolve/<path>` 支持 HTTP Range）。

四个子命令：

  plan    读 delivery_manifest.json（--manifest 本地文件或 --manifest-url，默认 bucket 根），按
          --include 前缀过滤出对象清单，写 plan.json（含原始 manifest 文本，fetch 时一并落库），
          打印 `PLAN objects=<n> bytes=<n>`。
  fetch   按 plan.json 逐对象下载到 <LIB>/<原相对路径>：先写 <目标>.part，断点续传用 Range；
          下载时流式算 sha256，完成后核 sha256 + 字节数，不符删 .part 重试（有限次）；通过后
          原子改名并在 <LIB>/.fetch-state/ok/ 写标记，重跑时已核过的对象直接跳过。
          末行 `TRAINSET_FETCH=PASS|FAIL objects=<n> bytes=<n> sha_mismatch=<n>`。
  unpack  把 source/data_tars/*.tar、source/features_tars/*.tar 流式解到 <LIB>/source/，按
          checksums/sha256-source-files.txt（路径相对 source/，如 `data/0.pkl`）逐成员核 sha256；
          成员数与 manifest/upload_manifest.json 记录一致且零不符才写解包标记并删除该 tar。
          拒绝绝对路径、`..`、符号链接、设备等非常规成员。按 tar 并行（--workers）。
          末行 `TRAINSET_UNPACK=PASS|FAIL members=<n> mismatch=<n>`。
  verify  复核：plan 内平文件逐个重算 sha256；每个 tar 都有解包标记；source/data、source/features
          实际文件数等于各 tar 记录成员数之和。--deep 时另按 sha256-source-files.txt 重算全部成员。
          末行 `TRAINSET_VERIFY=PASS|FAIL plain=<n> plain_mismatch=<n> tars_unpacked=<n>/<n> source_files=<n> expected=<n>`。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import socket
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

BUCKET = "HongzeFu/robomme-16task-orig100-1600ep-trainset-20261028-v1"
DEFAULT_BASE_URL = "https://huggingface.co/buckets/" + BUCKET + "/resolve/"
# 正式 K7 子集（计划 K7 与 S3 行）；delivery_manifest.json 本身由 plan 内嵌、fetch 落库。
DEFAULT_INCLUDE = (
    "meta/", "source/data_tars/", "source/features_tars/", "source/meta/", "framesamp-4x4/",
    "motion-stage2-1024/", "assets/", "checksums/", "evidence/", "manifest/",
)
TAR_DIRS = ("source/data_tars/", "source/features_tars/")
SOURCE_SUMS = "checksums/sha256-source-files.txt"
UPLOAD_MANIFEST = "manifest/upload_manifest.json"
STATE_DIR = ".fetch-state"
CHUNK = 8 * 1024 * 1024

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def force_ipv4() -> None:
    """只让本进程解析 IPv4（aspen 实测 IPv6 不可达）；不改主机网络。"""
    orig = socket.getaddrinfo

    def only_v4(host, port, family=0, *args, **kwargs):
        return orig(host, port, socket.AF_INET, *args, **kwargs)

    socket.getaddrinfo = only_v4


def safe_rel(name: str) -> PurePosixPath:
    """校验相对路径：拒绝空、绝对路径、`..`、反斜杠与非规范写法。"""
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or str(path) != name.rstrip("/") or name.startswith("./")):
        raise ValueError("非法路径: " + repr(name))
    return path


def under(root: Path, rel: str) -> Path:
    return root.joinpath(*safe_rel(rel).parts)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def object_url(base_url: str, rel: str) -> str:
    return base_url + urllib.parse.quote(rel, safe="/")


def http_get(url: str, start: int = 0, timeout: float = 60.0):
    req = urllib.request.Request(url, headers={"User-Agent": "fetch_orig100_trainset/1"})
    if start:
        req.add_header("Range", "bytes=%d-" % start)
    return urllib.request.urlopen(req, timeout=timeout)


def read_url(url: str) -> bytes:
    with http_get(url) as resp:
        return resp.read()


# ---------------------------------------------------------------- plan

def cmd_plan(args) -> int:
    if args.manifest:
        raw = Path(args.manifest).read_bytes()
    else:
        raw = read_url(args.manifest_url or object_url(args.base_url, "delivery_manifest.json"))
    manifest = json.loads(raw)
    files = manifest["files"]
    include = tuple(args.include)
    objects = []
    for rel in sorted(files):
        if not rel.startswith(include):
            continue
        safe_rel(rel)
        meta = files[rel]
        objects.append({"path": rel, "bytes": int(meta["bytes"]), "sha256": meta["sha256"]})
    total = sum(o["bytes"] for o in objects)
    plan = {
        "schema": "orig100-trainset-fetch-plan-v1",
        "bucket": manifest.get("bucket"),
        "dataset": manifest.get("dataset"),
        "base_url": args.base_url,
        "include": list(include),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_raw": raw.decode("utf-8"),
        "objects": objects,
        "total_bytes": total,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    by_top: dict[str, list[int]] = {}
    for o in objects:
        top = "/".join(o["path"].split("/")[:2]) if o["path"].startswith("source/") else o["path"].split("/")[0]
        agg = by_top.setdefault(top, [0, 0])
        agg[0] += 1
        agg[1] += o["bytes"]
    for top in sorted(by_top):
        log("PLAN_PART %s objects=%d bytes=%d" % (top, by_top[top][0], by_top[top][1]))
    unmatched = [p for p in include if not any(o["path"].startswith(p) for o in objects)]
    if unmatched:
        log("PLAN_WARN 以下前缀没有命中任何对象: " + " ".join(unmatched))
    log("PLAN objects=%d bytes=%d" % (len(objects), total))
    return 0


# ---------------------------------------------------------------- fetch

def ok_marker(lib: Path, rel: str) -> Path:
    return under(lib / STATE_DIR / "ok", rel + ".ok")


def unpack_marker(lib: Path, tar_rel: str) -> Path:
    return under(lib / STATE_DIR / "unpacked", tar_rel + ".done")


def is_source_tar(rel: str) -> bool:
    return rel.startswith(TAR_DIRS) and rel.endswith(".tar")


def already_done(lib: Path, obj: dict) -> bool:
    marker = ok_marker(lib, obj["path"])
    if not marker.is_file() or marker.read_text().strip() != obj["sha256"]:
        return False
    target = under(lib, obj["path"])
    if target.is_file() and target.stat().st_size == obj["bytes"]:
        return True
    # tar 已解包并删除：不再重下
    return is_source_tar(obj["path"]) and unpack_marker(lib, obj["path"]).is_file()


def download_one(lib: Path, base_url: str, obj: dict, retries: int, timeout: float) -> tuple[str, str]:
    """返回 (状态, 说明)；状态 ∈ ok / skip / mismatch / error。"""
    rel, want_sha, want_bytes = obj["path"], obj["sha256"], obj["bytes"]
    if already_done(lib, obj):
        return "skip", ""
    target = under(lib, rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    # 目标已存在（标记丢失）：直接核，不重下
    if target.is_file() and target.stat().st_size == want_bytes and sha256_file(target) == want_sha:
        write_marker(ok_marker(lib, rel), want_sha)
        return "ok", "existing"
    last = ""
    saw_mismatch = False
    for attempt in range(retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            if have > want_bytes:
                part.unlink()
                have = 0
            h = hashlib.sha256()
            if have:
                with part.open("rb") as f:
                    for block in iter(lambda: f.read(CHUNK), b""):
                        h.update(block)
            if have < want_bytes or want_bytes == 0:
                with http_get(object_url(base_url, rel), start=have, timeout=timeout) as resp:
                    status = getattr(resp, "status", 200)
                    if have and status != 206:
                        # 服务端不认 Range：从头来
                        have = 0
                        h = hashlib.sha256()
                        mode = "wb"
                    else:
                        mode = "ab" if have else "wb"
                    with part.open(mode) as out:
                        for block in iter(lambda: resp.read(CHUNK), b""):
                            out.write(block)
                            h.update(block)
            size = part.stat().st_size
            got = h.hexdigest()
            if size == want_bytes and got == want_sha:
                os.replace(part, target)
                write_marker(ok_marker(lib, rel), want_sha)
                return "ok", "attempt=%d" % (attempt + 1)
            saw_mismatch = True
            last = "size=%d/%d sha=%s" % (size, want_bytes, got[:12])
            log("FETCH_MISMATCH %s %s attempt=%d" % (rel, last, attempt + 1))
            part.unlink(missing_ok=True)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = "%s: %s" % (type(exc).__name__, exc)
            log("FETCH_RETRY %s attempt=%d %s" % (rel, attempt + 1, last))
            time.sleep(min(30, 2 ** attempt))
    return ("mismatch" if saw_mismatch else "error"), last


def write_marker(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text + "\n")
    os.replace(tmp, path)


def cmd_fetch(args) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    lib = Path(args.out)
    lib.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url or plan["base_url"]
    state = lib / STATE_DIR
    state.mkdir(parents=True, exist_ok=True)
    write_marker(state / "plan.json", json.dumps(plan, ensure_ascii=False))
    raw = plan["manifest_raw"].encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != plan["manifest_sha256"]:
        log("TRAINSET_FETCH=FAIL objects=0 bytes=0 sha_mismatch=0 reason=plan_manifest_corrupt")
        return 1
    (lib / "delivery_manifest.json").write_bytes(raw)
    objects = plan["objects"]
    counts = {"ok": 0, "skip": 0, "mismatch": 0, "error": 0}
    done_bytes = 0
    failed = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(download_one, lib, base_url, o, args.retries, args.timeout): o for o in objects}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            o = futs[fut]
            try:
                status, note = fut.result()
            except Exception as exc:  # 路径非法等：记失败，不中断其他对象
                status, note = "error", "%s: %s" % (type(exc).__name__, exc)
            counts[status] += 1
            if status in ("ok", "skip"):
                done_bytes += o["bytes"]
            else:
                failed.append((o["path"], status, note))
            if status != "skip" or i == len(objects):
                log("FETCH_PROGRESS %d/%d %s %s bytes_done=%d elapsed=%.0fs"
                    % (i, len(objects), status, o["path"], done_bytes, time.time() - t0))
    for path, status, note in failed:
        log("FETCH_FAILED %s %s %s" % (path, status, note))
    good = counts["ok"] + counts["skip"]
    verdict = "PASS" if good == len(objects) else "FAIL"
    log("TRAINSET_FETCH=%s objects=%d bytes=%d sha_mismatch=%d errors=%d skipped=%d"
        % (verdict, good, done_bytes, counts["mismatch"], counts["error"], counts["skip"]))
    return 0 if verdict == "PASS" else 1


# ---------------------------------------------------------------- unpack

def load_source_sums(lib: Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    with under(lib, SOURCE_SUMS).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            digest, _, rel = line.partition("  ")
            if len(digest) != 64 or not rel:
                raise ValueError("校验行格式不对: " + line[:120])
            sums[rel] = digest
    return sums


def load_expected_members(lib: Path) -> dict[str, int]:
    path = under(lib, UPLOAD_MANIFEST)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {s["rel"]: int(s["members"]) for s in data.get("shards", []) if "members" in s}


def unpack_one(lib: Path, tar_rel: str, sums: dict[str, str], expected: int | None) -> tuple[int, int, str]:
    """解一个 tar；返回 (成员数, 不符数, 说明)。全部对上才写标记并删 tar。"""
    marker = unpack_marker(lib, tar_rel)
    tar_path = under(lib, tar_rel)
    if marker.is_file():
        n = int(json.loads(marker.read_text())["members"])
        if tar_path.exists():
            tar_path.unlink()  # 上次写完标记后没来得及删
        return n, 0, "skip"
    src_root = lib / "source"
    members = 0
    mismatch = 0
    notes = []
    with tarfile.open(tar_path, mode="r|") as tf:
        for m in tf:
            if m.isdir():
                continue
            try:
                rel = str(safe_rel(m.name))
            except ValueError:
                mismatch += 1
                notes.append("unsafe:" + m.name)
                continue
            if not m.isfile():
                mismatch += 1
                notes.append("nonfile:" + m.name)
                continue
            members += 1
            want = sums.get(rel)
            dest = under(src_root, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".unpack-tmp")
            h = hashlib.sha256()
            src = tf.extractfile(m)
            with tmp.open("wb") as out:
                for block in iter(lambda: src.read(CHUNK), b""):
                    out.write(block)
                    h.update(block)
            if want is None or h.hexdigest() != want:
                mismatch += 1
                notes.append(("nosum:" if want is None else "sha:") + rel)
                tmp.unlink()
                continue
            os.replace(tmp, dest)
    if expected is not None and members != expected:
        mismatch += 1
        notes.append("members=%d expected=%d" % (members, expected))
    if mismatch == 0:
        write_marker(marker, json.dumps({"members": members}))
        tar_path.unlink()
    return members, mismatch, ";".join(notes[:5])


def list_tars(lib: Path, plan: dict | None) -> list[str]:
    if plan is not None:
        return [o["path"] for o in plan["objects"] if is_source_tar(o["path"])]
    rels = set()
    for d in TAR_DIRS:
        root = under(lib, d.rstrip("/"))
        if root.is_dir():
            rels.update(d + p.name for p in root.glob("*.tar"))
        mroot = lib / STATE_DIR / "unpacked" / d.rstrip("/")
        if mroot.is_dir():
            rels.update(d + p.name[: -len(".done")] for p in mroot.glob("*.tar.done"))
    return sorted(rels)


def read_lib_plan(lib: Path) -> dict | None:
    p = lib / STATE_DIR / "plan.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def cmd_unpack(args) -> int:
    lib = Path(args.lib)
    plan = read_lib_plan(lib)
    tars = list_tars(lib, plan)
    sums = load_source_sums(lib)
    expected = load_expected_members(lib)
    total_members = 0
    total_mismatch = 0
    missing = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for t in tars:
            if not under(lib, t).is_file() and not unpack_marker(lib, t).is_file():
                missing += 1
                log("UNPACK_MISSING %s" % t)
                continue
            futs[pool.submit(unpack_one, lib, t, sums, expected.get(t))] = t
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            t = futs[fut]
            try:
                n, bad, note = fut.result()
            except Exception as exc:
                n, bad, note = 0, 1, "%s: %s" % (type(exc).__name__, exc)
            total_members += n
            total_mismatch += bad
            log("UNPACK_PROGRESS %d/%d %s members=%d mismatch=%d %s" % (i, len(futs), t, n, bad, note))
    verdict = "PASS" if total_mismatch == 0 and missing == 0 and tars else "FAIL"
    log("TRAINSET_UNPACK=%s members=%d mismatch=%d tars=%d missing_tars=%d"
        % (verdict, total_members, total_mismatch, len(tars), missing))
    return 0 if verdict == "PASS" else 1


# ---------------------------------------------------------------- verify

def cmd_verify(args) -> int:
    lib = Path(args.lib)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8")) if args.plan else read_lib_plan(lib)
    if plan is None:
        log("TRAINSET_VERIFY=FAIL reason=no_plan")
        return 1
    plain = [o for o in plan["objects"] if not is_source_tar(o["path"])]
    tars = [o["path"] for o in plan["objects"] if is_source_tar(o["path"])]
    plain_bad = 0
    for o in plain:
        p = under(lib, o["path"])
        if not p.is_file() or p.stat().st_size != o["bytes"] or sha256_file(p) != o["sha256"]:
            plain_bad += 1
            log("VERIFY_PLAIN_BAD %s" % o["path"])
    unpacked = sum(1 for t in tars if unpack_marker(lib, t).is_file())
    expected_map = load_expected_members(lib)
    expected = sum(expected_map.get(t, 0) for t in tars)
    actual = 0
    for sub in ("data", "features"):
        root = lib / "source" / sub
        if root.is_dir():
            for _dirpath, _dirs, names in os.walk(root):
                actual += sum(1 for n in names if not n.endswith(".unpack-tmp"))
    deep_bad = 0
    if args.deep:
        sums = load_source_sums(lib)
        for rel, digest in sums.items():
            if not rel.startswith(("data/", "features/")):
                continue
            p = under(lib / "source", rel)
            if not p.is_file() or sha256_file(p) != digest:
                deep_bad += 1
                if deep_bad <= 20:
                    log("VERIFY_DEEP_BAD %s" % rel)
    ok = plain_bad == 0 and unpacked == len(tars) and actual == expected and deep_bad == 0
    log("TRAINSET_VERIFY=%s plain=%d plain_mismatch=%d tars_unpacked=%d/%d source_files=%d expected=%d%s"
        % ("PASS" if ok else "FAIL", len(plain), plain_bad, unpacked, len(tars), actual, expected,
           (" deep_mismatch=%d" % deep_bad) if args.deep else ""))
    return 0 if ok else 1


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ipv4", action="store_true", help="只用 IPv4 解析（本机 IPv6 不可达时用）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--manifest", help="本地 delivery_manifest.json")
    src.add_argument("--manifest-url", help="manifest URL（默认 bucket 根 delivery_manifest.json）")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--include", nargs="+", default=list(DEFAULT_INCLUDE))
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("fetch")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True, help="库根 <LIB>")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--base-url", default=None, help="覆盖 plan 内的 base_url（测试用）")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("unpack")
    p.add_argument("--lib", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_unpack)

    p = sub.add_parser("verify")
    p.add_argument("--lib", required=True)
    p.add_argument("--plan", default=None, help="默认 <LIB>/.fetch-state/plan.json")
    p.add_argument("--deep", action="store_true", help="按 sha256-source-files.txt 重算全部成员")
    p.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    if args.ipv4:
        force_ipv4()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
