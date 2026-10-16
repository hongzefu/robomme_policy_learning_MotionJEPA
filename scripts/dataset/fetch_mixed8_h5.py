#!/usr/bin/env python3
"""下载并解包 mixed8 16 任务 × 200 的逐集 H5 源；只准备 merge 的输入，不调用 merge。

子命令：
  fetch    从 HF bucket 拉 ``record_dataset_<Task>_<difficulty>.h5.tar.xz``（共 100 个，16 个视频包不拉），
           对照 bucket 的 ``SHA256SUMS`` 逐包校验，包级并发解包到平铺的 ``<src>/extracted/`` 后删包；
           同时从本地 benchmark 产物拷贝 ``<src>/snapshot/MANIFEST.json`` 与 ``<src>/control/source.json``。
           断点续传：已解包且逐文件字节数核对通过的包直接跳过，已下载且 sha256 通过的包不重下。
           每个任务输出一行 ``FETCH_VERIFY=PASS task=<T> archives=<n> episodes=<m>``（失败为 FAIL 行）。
  cleanup  自己核对 merge verify 日志里有 ``MERGE_VERIFY=PASS level=full … episodes=3200 … mismatches=0``，
           有才删除 ``<src>/extracted/``，否则拒绝删除并以非 0 退出。
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

DEFAULT_HF = "/scratch/hongze/MotionJEPA/.venv/bin/hf"
DEFAULT_BUCKET = "HongzeFu/robomme-16task-mixed8-200-h5-20261014-v1"
DEFAULT_BENCHMARK = "/scratch/hongze/robomme_benchmark_newtask-v3-MotionJepa1006/artifacts/mixed8/full200"
MANIFEST_SHA256 = "e3c7564f787781910f741d57b487d45565defbe8d3c2c17af9233c651093de11"
EXPECTED_ARCHIVES = 100
EXPECTED_EPISODES = 3200
ARCHIVE_RE = re.compile(r"record_dataset_(?P<task>[A-Za-z0-9]+)_(?P<difficulty>easy|medium|hard|xhard\d+)\.h5\.tar\.xz")
SUMS_RE = re.compile(r"(?P<sha>[0-9a-f]{64}) [ *](?P<name>\S+)")
PRINT_LOCK = threading.Lock()


def log(message):
    with PRINT_LOCK:
        print(message, flush=True)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def plain_dir(path):
    """返回不含符号链接的绝对路径；下载/解包/删除都只在真实目录里进行。"""
    path = Path(os.path.abspath(path))
    require(not any(p.is_symlink() for p in (path, *path.parents)), f"路径包含符号链接: {path}")
    return path


@contextmanager
def src_lock(src):
    src.mkdir(parents=True, exist_ok=True)
    with (src / ".fetch.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"源目录正在被另一个 fetch/cleanup 使用: {src}") from error
        yield


def copy_verified(source, target, expected_sha=None):
    """原子拷贝；目标已存在时只接受逐字节相同（续跑），不覆盖。"""
    require(source.is_file(), f"缺文件: {source}")
    if target.exists():
        require(target.is_file() and not target.is_symlink(), f"目标不是普通文件: {target}")
        require(sha256_file(target) == sha256_file(source), f"已有文件与来源不同，拒绝覆盖: {target}")
    else:
        tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
    got = sha256_file(target)
    require(expected_sha is None or got == expected_sha, f"sha256 不符: {target} got={got} want={expected_sha}")
    return got


def prepare_snapshot(src, benchmark):
    """拷贝 snapshot/MANIFEST.json 与 control/source.json，并核对 MANIFEST 的固定 sha256。"""
    (src / "snapshot").mkdir(exist_ok=True)
    (src / "control").mkdir(exist_ok=True)
    copy_verified(benchmark / "snapshot" / "MANIFEST.json", src / "snapshot" / "MANIFEST.json", MANIFEST_SHA256)
    copy_verified(benchmark / "control" / "source.json", src / "control" / "source.json")
    source = read_json(src / "control" / "source.json")
    require(source.get("manifest_sha256") == MANIFEST_SHA256, "control/source.json 的 manifest_sha256 不符")
    require(source.get("source_kind") == "local-benchmark", "control/source.json 的 source_kind 不是 local-benchmark")
    log(f"SNAPSHOT_OK manifest_sha256={MANIFEST_SHA256}")
    return read_json(src / "snapshot" / "MANIFEST.json")


def hf_env():
    env = dict(os.environ)
    env.pop("HF_HUB_OFFLINE", None)
    return env


def hf_cp(hf, bucket, name, dest):
    """单文件下载；hf 的报错原文透传给调用方。"""
    result = subprocess.run([hf, "buckets", "cp", f"hf://buckets/{bucket}/{name}", str(dest)],
                            env=hf_env(), capture_output=True, text=True)
    if result.returncode != 0 or not Path(dest).is_file():
        tail = (result.stdout + result.stderr).strip().splitlines()[-5:]
        raise RuntimeError(f"hf buckets cp 失败 rc={result.returncode}: {name}: {' | '.join(tail)}")


def parse_sums(text):
    sums = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        match = SUMS_RE.fullmatch(line.strip())
        require(match is not None, f"SHA256SUMS 行格式非法: {line!r}")
        name = match["name"].removeprefix("./")
        require(name not in sums, f"SHA256SUMS 重复条目: {name}")
        sums[name] = match["sha"]
    return sums


def fetch_sums(args, src):
    """下载 bucket 的 SHA256SUMS 与 MANIFEST.json；续跑时要求与已落盘的 SHA256SUMS 逐字节相同。"""
    meta = src / "bucket"
    meta.mkdir(exist_ok=True)
    fresh = meta / f"SHA256SUMS.{os.getpid()}.part"
    fresh.unlink(missing_ok=True)
    hf_cp(args.hf, args.bucket, "SHA256SUMS", fresh)
    final = meta / "SHA256SUMS"
    if final.exists():
        require(final.read_bytes() == fresh.read_bytes(), "bucket SHA256SUMS 与上次下载不同，拒绝续跑")
        fresh.unlink()
    else:
        os.replace(fresh, final)
    remote_manifest = meta / "MANIFEST.json"
    if not remote_manifest.exists():
        part = meta / f"MANIFEST.json.{os.getpid()}.part"
        part.unlink(missing_ok=True)
        hf_cp(args.hf, args.bucket, "MANIFEST.json", part)
        os.replace(part, remote_manifest)
    require(sha256_file(remote_manifest) == MANIFEST_SHA256, "bucket MANIFEST.json sha256 与本地 snapshot 不符")
    return parse_sums(final.read_text())


def archive_table(sums, manifest):
    """SHA256SUMS 里的 H5 包与 MANIFEST summary.cells 一一对应：包 = (task, difficulty) 一格。"""
    archives = {}
    for name, sha in sums.items():
        match = ARCHIVE_RE.fullmatch(name)
        if match is None:
            # 另有 16 个 videos_<Task>.tar.xz 视频包，按口径不拉。
            continue
        archives[name] = {"name": name, "sha256": sha, "task": match["task"], "difficulty": match["difficulty"]}
    require(len(archives) == EXPECTED_ARCHIVES, f"SHA256SUMS 中 record_dataset_*.h5.tar.xz 应为 {EXPECTED_ARCHIVES} 个，实为 {len(archives)}")
    cells = manifest["summary"]["cells"]
    require({(a["task"], a["difficulty"]) for a in archives.values()} == {(t, d) for t, v in cells.items() for d in v},
            "SHA256SUMS 的包与 MANIFEST summary.cells 不一一对应")
    members = defaultdict(dict)
    for row in manifest["episodes"]:
        if row["role"] == "primary":
            members[(row["task"], row["difficulty"])][row["member"]] = row["h5_bytes"]
    for archive in archives.values():
        key = (archive["task"], archive["difficulty"])
        archive["members"] = members[key]
        require(len(archive["members"]) == cells[key[0]][key[1]], f"包 {archive['name']} 的 MANIFEST 集数与 cells 不符")
    return archives


def extracted_ok(archive, extracted):
    for member, size in archive["members"].items():
        path = extracted / member
        if not (path.is_file() and not path.is_symlink() and path.stat().st_size == size):
            return False
    return True


def flat_member(name):
    require(name == Path(name).name and name.endswith(".h5") and not name.startswith("."), f"包内路径不是平铺 .h5: {name!r}")
    return name


def unpack(archive, path, extracted, staging):
    """解包到 staging 子目录，核对成员集合与字节数，再逐个移入平铺 extracted/。"""
    listing = subprocess.run(["tar", "-tJf", str(path)], capture_output=True, text=True)
    require(listing.returncode == 0, f"tar 列目录失败: {listing.stderr.strip()[-300:]}")
    names = [flat_member(n.removeprefix("./")) for n in listing.stdout.splitlines() if n not in ("", "./")]
    require(len(names) == len(set(names)) and set(names) == set(archive["members"]),
            f"包内成员与 MANIFEST 不符: 包内 {len(names)} 个，MANIFEST {len(archive['members'])} 个")
    work = staging / archive["name"]
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    result = subprocess.run(["tar", "-xJf", str(path), "-C", str(work), "--no-same-owner", "--no-same-permissions"],
                            capture_output=True, text=True)
    require(result.returncode == 0, f"tar 解包失败: {result.stderr.strip()[-300:]}")
    entries = list(work.iterdir())
    require({p.name for p in entries} == set(archive["members"]), "解包结果与成员清单不符")
    for entry in entries:
        require(entry.is_file() and not entry.is_symlink(), f"包内存在非普通文件: {entry.name}")
        require(entry.stat().st_size == archive["members"][entry.name], f"解包件字节数与 MANIFEST 不符: {entry.name}")
    for entry in entries:
        os.replace(entry, extracted / entry.name)
    work.rmdir()


def process_archive(archive, args, src):
    """单包：已解包核对通过就跳过；否则下载（已校验的包不重下）→ sha256 → 解包 → 删包。失败最多重试 retries 次。"""
    extracted, store, staging, state = (src / n for n in ("extracted", "archives", "staging", "state"))
    marker = state / f"{archive['name']}.json"
    expect = {"name": archive["name"], "sha256": archive["sha256"], "episodes": len(archive["members"])}
    if marker.exists() and read_json(marker) == expect and extracted_ok(archive, extracted):
        log(f"FETCH_ARCHIVE=SKIP archive={archive['name']} episodes={expect['episodes']}")
        return expect["episodes"]
    path = store / archive["name"]
    error = None
    for attempt in range(1, args.retries + 2):
        started = time.monotonic()
        try:
            if path.exists() and sha256_file(path) != archive["sha256"]:
                path.unlink()
            if not path.exists():
                part = store / f"{archive['name']}.part"
                part.unlink(missing_ok=True)
                hf_cp(args.hf, args.bucket, archive["name"], part)
                got = sha256_file(part)
                if got != archive["sha256"]:
                    part.unlink()
                    raise ValueError(f"包 sha256 与 SHA256SUMS 不符: got={got}")
                os.replace(part, path)
            unpack(archive, path, extracted, staging)
            write_json(marker, expect)
            path.unlink()
            log(f"FETCH_ARCHIVE=PASS archive={archive['name']} episodes={expect['episodes']} attempt={attempt} "
                f"secs={time.monotonic() - started:.0f}")
            return expect["episodes"]
        except (ValueError, RuntimeError, OSError) as exc:
            error = exc
            log(f"FETCH_ARCHIVE=RETRY archive={archive['name']} attempt={attempt} error={exc}")
    raise RuntimeError(f"{archive['name']} 重试 {args.retries} 次仍失败: {error}")


def cmd_fetch(args):
    os.environ.pop("HF_HUB_OFFLINE", None)  # paths.sh 会 export HF_HUB_OFFLINE=1；本进程及子进程一律移除
    require(Path(args.hf).is_file() and os.access(args.hf, os.X_OK), f"hf 不可执行: {args.hf}")
    require(1 <= args.jobs <= 8, "--jobs 必须在 1..8")
    require(args.retries >= 0, "--retries 不能为负")
    src = plain_dir(args.src)
    with src_lock(src):
        manifest = prepare_snapshot(src, plain_dir(args.benchmark))
        archives = archive_table(fetch_sums(args, src), manifest)
        all_tasks = sorted({a["task"] for a in archives.values()})
        tasks = all_tasks if args.tasks is None else args.tasks.split(",")
        require(tasks and len(tasks) == len(set(tasks)) and set(tasks) <= set(all_tasks), f"任务列表非法: {args.tasks}")
        if set(tasks) == set(all_tasks):
            require(sum(len(a["members"]) for a in archives.values()) == EXPECTED_EPISODES, "全量 primary 集数不是 3200")
        for name in ("extracted", "archives", "staging", "state"):
            (src / name).mkdir(exist_ok=True)
        chosen = sorted((a for a in archives.values() if a["task"] in tasks), key=lambda a: a["name"])
        expected = Counter(row["task"] for row in manifest["episodes"] if row["role"] == "primary")
        log(f"FETCH_START tasks={len(tasks)} archives={len(chosen)} episodes={sum(expected[t] for t in tasks)} jobs={args.jobs}")
        pending = Counter(a["task"] for a in chosen)
        done = Counter()
        failed = {}
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(process_archive, a, args, src): a for a in chosen}
            for future in as_completed(futures):
                archive = futures[future]
                task = archive["task"]
                try:
                    done[task] += future.result()
                except (ValueError, RuntimeError, OSError) as exc:
                    failed.setdefault(task, []).append(archive["name"])
                    log(f"FETCH_ARCHIVE=FAIL archive={archive['name']} error={exc}")
                pending[task] -= 1
                if pending[task] == 0:
                    n = sum(a["task"] == task for a in chosen)
                    if task in failed or done[task] != expected[task]:
                        log(f"FETCH_VERIFY=FAIL task={task} archives={n} episodes={done[task]} expected={expected[task]} "
                            f"failed={','.join(sorted(failed.get(task, []))) or '-'}")
                    else:
                        log(f"FETCH_VERIFY=PASS task={task} archives={n} episodes={done[task]}")
        staging = src / "staging"
        if staging.is_dir() and not any(staging.iterdir()):
            staging.rmdir()
        bad = sorted(name for names in failed.values() for name in names)
        require(not bad, f"{len(bad)} 个包失败: {','.join(bad)}")
        require(all(done[t] == expected[t] for t in tasks), "解包集数与 MANIFEST 不符")
        log(f"FETCH_DONE tasks={len(tasks)} archives={len(chosen)} episodes={sum(done.values())}")


VERIFY_RE = re.compile(r"MERGE_VERIFY=PASS level=full tasks=(?P<tasks>\d+) episodes=(?P<episodes>\d+) "
                       r"timesteps=(?P<timesteps>\d+) role=primary:(?P<primary>\d+) mismatches=(?P<mismatches>\d+)")


def check_verify_log(text, expect_episodes):
    """日志里所有 MERGE_VERIFY 行中最后一行必须是 full 级 PASS，且 episodes/primary 等于期望、mismatches=0。"""
    lines = [line.strip() for line in text.splitlines() if "MERGE_VERIFY=" in line]
    require(lines, "verify 日志里没有 MERGE_VERIFY 行")
    require(not any("MERGE_VERIFY=FAIL" in line for line in lines), "verify 日志含 MERGE_VERIFY=FAIL")
    match = VERIFY_RE.search(lines[-1])
    require(match is not None, f"最后一行 MERGE_VERIFY 不是 full 级 PASS: {lines[-1]}")
    require(int(match["episodes"]) == int(match["primary"]) == expect_episodes,
            f"episodes 不是 {expect_episodes}: {lines[-1]}")
    require(int(match["mismatches"]) == 0, f"mismatches 非 0: {lines[-1]}")
    return match


def cmd_cleanup(args):
    src = plain_dir(args.src)
    log_path = Path(args.verify_log)
    require(log_path.is_file(), f"verify 日志不存在: {log_path}")
    match = check_verify_log(log_path.read_text(errors="replace"), args.expect_episodes)
    target = src / "extracted"
    require(target.is_dir() and not target.is_symlink(), f"extracted 不是真实目录: {target}")
    with src_lock(src):
        manifest_path = src / "snapshot" / "MANIFEST.json"
        if manifest_path.is_file():
            # 与 MANIFEST 的 timestep_count 之和对拍（G2：2,145,548）。
            primary = [r for r in read_json(manifest_path)["episodes"] if r["role"] == "primary"]
            require(int(match["timesteps"]) == sum(r["timestep_count"] for r in primary),
                    f"verify timesteps 与 MANIFEST timestep_count 之和不符: {match['timesteps']}")
        files = sum(1 for _ in target.iterdir())
        shutil.rmtree(target)
    log(f"CLEANUP=PASS removed={target} files={files} verify_episodes={match['episodes']} timesteps={match['timesteps']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="下载、校验、解包 mixed8 H5 包，并准备 snapshot/control")
    fetch.set_defaults(callback=cmd_fetch)
    fetch.add_argument("--hf", default=DEFAULT_HF, help=f"hf CLI 绝对路径（默认 {DEFAULT_HF}）")
    fetch.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"H5 源 bucket id（默认 {DEFAULT_BUCKET}）")
    fetch.add_argument("--src", type=Path, required=True, help="下载与解包落点；产出 extracted/、snapshot/、control/")
    fetch.add_argument("--tasks", help="逗号分隔任务子集；缺省为 SHA256SUMS 里的全部 16 个任务")
    fetch.add_argument("--jobs", type=int, default=8, help="包级并发（本机 xz 5.2.5 无多线程解压），1..8，默认 8")
    fetch.add_argument("--retries", type=int, default=3, help="每包失败后的最多重试次数（默认 3，即最多 4 次尝试）")
    fetch.add_argument("--benchmark", type=Path, default=Path(DEFAULT_BENCHMARK),
                       help="本地 benchmark 产物目录，内含 snapshot/MANIFEST.json 与 control/source.json")
    cleanup = commands.add_parser("cleanup", help="核对 merge verify 日志的 PASS 标记后删除 <src>/extracted/")
    cleanup.set_defaults(callback=cmd_cleanup)
    cleanup.add_argument("--src", type=Path, required=True)
    cleanup.add_argument("--verify-log", type=Path, required=True, help="merge_v2_h5.py verify --level full 的 tee 日志")
    cleanup.add_argument("--expect-episodes", type=int, default=EXPECTED_EPISODES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        args.callback(args)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"{args.command.upper()}=FAIL {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
