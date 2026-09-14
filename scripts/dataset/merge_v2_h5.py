#!/usr/bin/env python3
"""将有来源清单的单 episode H5 无损合并；仅 full 校验可以发布完成标记。"""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
V1_STORE = REPO / "v1-store"
SOURCE_COUNTS = dict.fromkeys(("BinFill", "RouteStick", "VideoRepick", "VideoUnmaskSwap"), 400)
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2, "xhard": 3}
SORT_VERSION = "difficulty-episode-v1"
ROW_KEYS = ("task", "difficulty", "episode", "seed", "role", "member", "h5_sha256", "h5_bytes", "timestep_count")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    require(not path.is_symlink(), f"拒绝写入符号链接: {path}")
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def real_path(path):
    path = Path(os.path.abspath(path))
    require(not any(p.is_symlink() for p in (path, *path.parents)), f"路径包含符号链接: {path}")
    return path


def output_path(path):
    path = real_path(path)
    root = real_path(V1_STORE)
    require(path != root and path.is_relative_to(root), f"输出必须在 {root} 内: {path}")
    require(not path.exists() or path.is_dir(), f"输出不是目录: {path}")
    return path


def source_path(extracted, member):
    rel = Path(member)
    require(not rel.is_absolute() and ".." not in rel.parts, f"非法源路径: {member}")
    root = real_path(extracted)
    path = real_path(root / rel)
    require(path.is_relative_to(root) and path.is_file(), f"源文件不存在: {path}")
    return path


@contextmanager
def output_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    lock = real_path(out / ".merge.lock")
    with lock.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"输出目录正在被另一个合并命令使用: {out}") from error
        yield


def load_manifest(snapshot, control):
    source = read_json(Path(control) / "source.json")
    manifest_path = Path(snapshot) / "MANIFEST.json"
    require(sha256_file(manifest_path) == source["manifest_sha256"], "MANIFEST sha256 与 source.json 不符")
    manifest = read_json(manifest_path)
    require(manifest["repo_id"] == source["repo_id"], "MANIFEST repo_id 不符")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", source["revision"])), "revision 必须是完整提交")
    return manifest, {k: source[k] for k in ("repo_id", "revision", "manifest_sha256")}


def select_primary(manifest, tasks, per_group=None):
    episodes = manifest["episodes"]
    roles = Counter(row["role"] for row in episodes)
    require(set(roles) <= {"primary", "spare", "smoke"}, "未知 role")
    # counts 同时含 videos/archives/files；role 只与三个对应计数字段比较。
    require({role: roles[role] for role in ("primary", "spare", "smoke")}
            == {role: manifest["counts"][role] for role in ("primary", "spare", "smoke")},
            f"role 分布不符: {dict(roles)}")
    primary = [row for row in episodes if row["role"] == "primary"]
    require(dict(Counter(row["task"] for row in primary)) == SOURCE_COUNTS, "源全集必须每任务 400 条，共 1600 条 primary")
    require(len(primary) == manifest["counts"]["primary"], "源全集 primary 总数不符")
    require(len(tasks) == len(set(tasks)) and set(tasks) <= set(SOURCE_COUNTS) and tasks, "任务列表非法")
    require(per_group is None or per_group > 0, "per-group 必须大于零")
    require(len({row["member"] for row in episodes}) == len(episodes), "源 member 重复或跨 role 重叠")
    require(len({(r["task"], r["difficulty"], r["episode"]) for r in primary}) == len(primary), "primary 三元身份重复")
    for row in primary:
        require(row["difficulty"] in DIFFICULTY_ORDER, f"未知难度 {row['difficulty']}")
        require(isinstance(row["episode"], int) and row["episode"] >= 0, "原 episode 编号非法")
        require(0 < row["timestep_count"] < 4096, f"步数超出 MemoryBuffer 契约: {row['member']}")
        require(bool(re.fullmatch(r"[0-9a-f]{64}", row["h5_sha256"])), "源摘要格式非法")
    excluded = {r["member"] for r in episodes if r["role"] != "primary"}
    selected = {}
    for task in tasks:
        counts = Counter()
        rows = []
        for row in sorted((r for r in primary if r["task"] == task), key=lambda r: (DIFFICULTY_ORDER[r["difficulty"]], r["episode"])):
            difficulty = row["difficulty"]
            if per_group is not None and counts[difficulty] >= per_group:
                continue
            counts[difficulty] += 1
            require(row["member"] not in excluded, "选取集合混入非 primary")
            rows.append({**{k: row[k] for k in ROW_KEYS}, "new_idx": len(rows), "src_group": f"episode_{row['episode']}"})
        selected[task] = rows
    return selected


def timestep_names(group, count):
    names = {k for k in group if k.startswith("timestep_")}
    require(names == {f"timestep_{i}" for i in range(count)}, f"timestep 索引不连续或数量不符: {group.name}")


def precheck_source(job):
    row, extracted = job
    path = source_path(extracted, row["member"])
    require(path.stat().st_size == row["h5_bytes"], f"源字节数不符: {path}")
    with h5py.File(path, "r") as source:
        require(list(source) == [row["src_group"]], f"源顶层组不符: {path}")
        group = source[row["src_group"]]
        timestep_names(group, row["timestep_count"])
        exec_seen = False
        for i in range(row["timestep_count"]):
            step = group[f"timestep_{i}"]
            demo = bool(step["info/is_video_demo"][()])
            require(not (exec_seen and demo), f"demo 不是严格前缀: {path} step={i}")
            exec_seen |= not demo
            require(i != 0 or not bool(step["info/is_completed"][()]), f"首帧 is_completed 为真: {path}")
            require("info/is_subgoal_boundary" in step, f"缺 is_subgoal_boundary: {path} step={i}")
        goals = np.asarray(group["setup/task_goal"][()]).reshape(-1)
        return row["task"], [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in goals]


def parallel_map(function, jobs, procs):
    if procs == 1:
        return [function(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=procs) as pool:
        return list(pool.map(function, jobs))


def map_path(out, task):
    return out / f"record_dataset_{task}_episode_map.json"


def base_rows(rows):
    return [{k: v for k, v in row.items() if k != "h5_sha256_local"} for row in rows]


def cmd_plan(args):
    out = output_path(args.out)
    manifest, pin = load_manifest(args.snapshot, args.control)
    tasks = args.tasks.split(",")
    selected = select_primary(manifest, tasks, args.per_group)
    all_rows = select_primary(manifest, list(SOURCE_COUNTS))
    plan = {"source_pin": pin, "tasks": tasks, "per_group": args.per_group, "sort_version": SORT_VERSION,
            "snapshot": str(real_path(args.snapshot)), "control": str(real_path(args.control)),
            "extracted": str(real_path(args.extracted))}
    with output_lock(out):
        existing = out / "merge_plan.json"
        if existing.exists():
            require(read_json(existing) == plan, "输出目录已有不同选取身份；请用全新目录")
            for task, rows in selected.items():
                require(base_rows(read_json(map_path(out, task))) == rows, "已有 map 与源清单不符")
        else:
            require({p.name for p in out.iterdir()} == {".merge.lock"}, "新计划拒绝使用非空输出目录")
        goals = defaultdict(set)
        jobs = [(row, args.extracted) for rows in all_rows.values() for row in rows]
        for task, variants in parallel_map(precheck_source, jobs, args.procs):
            goals[task].update(variants)
        if not existing.exists():
            for task, rows in selected.items():
                write_json(map_path(out, task), rows)
            write_json(out / "source_pin.json", pin)
            write_json(out / "merge_plan.json", plan)
        write_json(out / "task_goal_variants.json", {task: sorted(values) for task, values in goals.items()})
    count = sum(map(len, selected.values()))
    maximum = max(row["timestep_count"] for rows in all_rows.values() for row in rows)
    print(f"PLAN_OK tasks={len(tasks)} selected={count} role=primary:{count} max_timesteps={maximum} prechecks=PASS", flush=True)


def load_plan(out, extracted):
    plan = read_json(out / "merge_plan.json")
    manifest, pin = load_manifest(plan["snapshot"], plan["control"])
    require(pin == plan["source_pin"] == read_json(out / "source_pin.json"), "source_pin 身份不符")
    require(str(real_path(extracted)) == plan["extracted"], "extracted 与原计划不符")
    require(plan["sort_version"] == SORT_VERSION, "排序版本不符")
    expected = select_primary(manifest, plan["tasks"], plan["per_group"])
    maps = {}
    for task, rows in expected.items():
        maps[task] = read_json(map_path(out, task))
        require(base_rows(maps[task]) == rows, f"map 身份或内容不符: {task}")
        for row in maps[task]:
            require(row.get("h5_sha256_local", row["h5_sha256"]) == row["h5_sha256"], "本地源摘要与清单不符")
    return plan, maps


def ds_equal(a, b):
    arr = np.asarray(a)
    if arr.dtype.kind == "f":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def compare_tree(source, target, context, seen_source=None, seen_target=None):
    """按链接遍历，拒绝 attrs、软/外链及硬链接别名，逐对象核对形状、dtype 与值。"""
    if seen_source is None:
        seen_source, seen_target = set(), set()
    require(not source.attrs and not target.attrs, f"attrs 非空: {context}")
    require(isinstance(source, h5py.Group) == isinstance(target, h5py.Group), f"对象类型不符: {context}")
    for obj, seen in ((source, seen_source), (target, seen_target)):
        address = h5py.h5o.get_info(obj.id).addr
        require(address not in seen, f"硬链接别名或循环: {context}")
        seen.add(address)
    if isinstance(source, h5py.Dataset):
        require(source.shape == target.shape and source.dtype == target.dtype, f"shape/dtype 不符: {context}")
        require(source.chunks == target.chunks and source.compression == target.compression
                and source.compression_opts == target.compression_opts
                and source.shuffle == target.shuffle and source.fletcher32 == target.fletcher32
                and source.scaleoffset == target.scaleoffset and source.maxshape == target.maxshape,
                f"dataset 布局不符: {context}")
        require(ds_equal(source[()], target[()]), f"dataset 值不符: {context}")
        return
    require(set(source) == set(target), f"名字集合不符: {context}")
    for name in source:
        require(isinstance(source.get(name, getlink=True), h5py.HardLink)
                and isinstance(target.get(name, getlink=True), h5py.HardLink), f"非 HardLink: {context}/{name}")
        compare_tree(source[name], target[name], f"{context}/{name}", seen_source, seen_target)


def compare_episode(source, target, row, level):
    require(list(source) == [row["src_group"]], "源顶层组不符")
    require(not source.attrs and not target.file.attrs, "文件根 attrs 非空")
    require(isinstance(source.get(row["src_group"], getlink=True), h5py.HardLink), "源顶层非 HardLink")
    group = source[row["src_group"]]
    timestep_names(group, row["timestep_count"])
    timestep_names(target, row["timestep_count"])
    if level == "full":
        compare_tree(group, target, row["member"])
    else:
        require(not group.attrs and not target.attrs and set(group) == set(target), "sample 顶层结构不符")
        for name in ("setup", "timestep_0", f"timestep_{row['timestep_count'] - 1}"):
            require(isinstance(group.get(name, getlink=True), h5py.HardLink)
                    and isinstance(target.get(name, getlink=True), h5py.HardLink), "sample 非 HardLink")
            compare_tree(group[name], target[name], f"{row['member']}/{name}")


def merge_task(job):
    task, rows, plan, out, extracted, check_sha, resume, force = job
    out = Path(out)
    final = real_path(out / f"record_dataset_{task}.h5")
    tmp = real_path(out / f"{final.name}.tmp")
    state_path = out / f"merge_state_{task}.json"
    identity = digest_json({"plan": plan, "rows": base_rows(rows)})
    if force:
        require(out.is_dir() and output_path(out) == out and (out / "source_pin.json").is_file(), "force 路径守卫失败")
        require(read_json(out / "source_pin.json") == plan["source_pin"], "force 来源身份不符")
        for path in (tmp, final):
            if path.exists():
                require(path.is_file(), f"force 目标不是文件: {path}")
                path.unlink()
    if final.exists() or tmp.exists():
        require(resume, f"已有合并文件，须显式 --resume 或 --force: {task}")
        require(state_path.exists() and read_json(state_path)["identity"] == identity, "续跑身份与原 map 不符")
        require(not (final.exists() and tmp.exists()), "正式文件和临时文件同时存在，拒绝推断归属")
    else:
        write_json(state_path, {"identity": identity})
    path = final if final.exists() else tmp
    existing = set()
    if path.exists():
        with h5py.File(path, "r") as target:
            existing = set(target)
            require(not target.attrs, "已有合并文件根 attrs 非空")
            require(existing <= {f"episode_{r['new_idx']}" for r in rows}, "已有文件存在多余组")
        print(f"MERGE_RESUME task={task} 已有={len(existing)}", flush=True)
    for row in rows:
        started = time.monotonic()
        src = source_path(extracted, row["member"])
        require(src.stat().st_size == row["h5_bytes"], f"源字节数改变: {src}")
        if check_sha:
            got = sha256_file(src)
            require(got == row["h5_sha256"], f"源 sha256 不符: {src}")
            row["h5_sha256_local"] = got
        name = f"episode_{row['new_idx']}"
        with h5py.File(src, "r") as source:
            require(list(source) == [row["src_group"]], f"源组名改变: {src}")
            if name in existing:
                with h5py.File(path, "r") as target:
                    require(isinstance(target.get(name, getlink=True), h5py.HardLink), "续跑顶层非 HardLink")
                    compare_episode(source, target[name], row, "full")
            else:
                require(path != final, "正式文件不完整，拒绝追加")
                # 每条独立开关文件；中断后已有组仍须 full 自校验，不能仅凭组名跳过。
                with h5py.File(tmp, "a") as target:
                    source.copy(row["src_group"], target, name=name)
        if check_sha:
            write_json(map_path(out, task), rows)
        print(f"MERGE_EP task={task} idx={row['new_idx']} member={row['member']} secs={time.monotonic() - started:.3f}", flush=True)
    if path == tmp:
        os.replace(tmp, final)
    print(f"MERGE_TASK=PASS task={task} episodes={len(rows)}", flush=True)


def cmd_merge(args):
    out = output_path(args.out)
    require(out.is_dir(), "请先执行 plan")
    with output_lock(out):
        plan, maps = load_plan(out, args.extracted)
        jobs = [(task, rows, plan, out, args.extracted, args.check_source_sha256,
                 args.resume, args.force) for task, rows in maps.items()]
        parallel_map(merge_task, jobs, min(args.procs, len(jobs)))


def verify_shard(job):
    task, rows, extracted, out, level = job
    errors = []
    path = Path(out) / f"record_dataset_{task}.h5"
    with h5py.File(path, "r") as target:
        for row in rows:
            try:
                src = source_path(extracted, row["member"])
                require(src.stat().st_size == row["h5_bytes"], "源字节数不符")
                require(row["role"] == "primary", "map 混入非 primary")
                with h5py.File(src, "r") as source:
                    compare_episode(source, target[f"episode_{row['new_idx']}"], row, level)
            except (ValueError, KeyError, OSError, TypeError) as error:
                errors.append(f"{task}/{row['new_idx']}: {error}")
    print(f"VERIFY_SHARD task={task} lo={rows[0]['new_idx']} hi={rows[-1]['new_idx'] + 1} mismatches={len(errors)}", flush=True)
    return errors


def cmd_verify(args):
    out = output_path(args.out)
    require(out.is_dir(), "请先执行 plan 和 merge")
    with output_lock(out):
        plan, maps = load_plan(out, args.extracted)
        before = {}
        for task, rows in maps.items():
            path = real_path(out / f"record_dataset_{task}.h5")
            before[task] = (path.stat().st_size, path.stat().st_mtime_ns)
            with h5py.File(path, "r") as target:
                require(set(target) == {f"episode_{r['new_idx']}" for r in rows}, f"合并文件 episode 集合不符: {task}")
                require(not target.attrs, "合并文件根 attrs 非空")
                require(all(isinstance(target.get(n, getlink=True), h5py.HardLink) for n in target), "顶层非 HardLink")
        jobs = []
        total = sum(map(len, maps.values()))
        width = max(1, math.ceil(total / args.procs))
        for task, rows in maps.items():
            jobs.extend((task, rows[i:i + width], args.extracted, out, args.level) for i in range(0, len(rows), width))
        errors = [error for group in parallel_map(verify_shard, jobs, args.procs) for error in group]
        require(not errors, f"mismatches={len(errors)}\n" + "\n".join(errors[:20]))
        paths = [out / f"record_dataset_{task}.h5" for task in maps]
        hashes = parallel_map(sha256_file, paths, min(4, args.procs)) if args.level == "full" else []
        for task in maps:
            path = out / f"record_dataset_{task}.h5"
            require(before[task] == (path.stat().st_size, path.stat().st_mtime_ns), "校验期间输出文件改变")
        timesteps = sum(row["timestep_count"] for rows in maps.values() for row in rows)
        result = {"level": args.level, "source_pin": plan["source_pin"], "selection": {k: plan[k] for k in ("tasks", "per_group", "sort_version")},
                  "maps_sha256": {task: sha256_file(map_path(out, task)) for task in maps},
                  "files": {path.name: digest for path, digest in zip(paths, hashes, strict=True)} if args.level == "full" else {},
                  "tasks": {task: {"episodes": len(rows), "timesteps": sum(r["timestep_count"] for r in rows)} for task, rows in maps.items()},
                  "role": {"primary": total}, "episodes": total, "timesteps": timesteps,
                  "source_sha256_checked": sum("h5_sha256_local" in row for rows in maps.values() for row in rows),
                  "mismatches": 0, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}
        write_json(out / ("MERGE_DONE.json" if args.level == "full" else "MERGE_SAMPLE_DONE.json"), result)
    print(f"MERGE_VERIFY=PASS level={args.level} tasks={len(maps)} episodes={total} timesteps={timesteps} role=primary:{total} mismatches=0", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, callback, procs in (("plan", cmd_plan, 48), ("merge", cmd_merge, 4), ("verify", cmd_verify, 48)):
        sub = commands.add_parser(name)
        sub.set_defaults(callback=callback)
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument("--extracted", type=Path, required=True)
        sub.add_argument("--procs", type=int, default=procs)
        if name == "plan":
            sub.add_argument("--snapshot", type=Path, required=True)
            sub.add_argument("--control", type=Path, required=True)
            sub.add_argument("--tasks", default=",".join(SOURCE_COUNTS))
            sub.add_argument("--per-group", type=int)
        elif name == "merge":
            sub.add_argument("--check-source-sha256", action="store_true")
            mode = sub.add_mutually_exclusive_group()
            mode.add_argument("--resume", action="store_true")
            mode.add_argument("--force", action="store_true")
        else:
            sub.add_argument("--level", choices=("sample", "full"), default="full")
    args = parser.parse_args(argv)
    try:
        require(0 < args.procs <= 96, "procs 必须在 1..96")
        args.callback(args)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f"{'MERGE_VERIFY' if args.command == 'verify' else args.command.upper()}=FAIL {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
