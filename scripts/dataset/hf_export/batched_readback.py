#!/usr/bin/env python3
"""分批回读验收：下载一批 → 逐文件 sha256 对上传前清单 → tar 成员抽样 → 删除该批 → 下一批。

由 run_mixed8_200_trainset_export.sh 的阶段 7/8 调用（用户 2026-10-16「a. 分批回读（推荐） 同意」，
计划口径 38）。取代模板 run_v2_1600ep_dataset_export.sh 阶段 7 的整桶 `hf sync` 与阶段 8 的整树
sha256：新库回读总量 1.3–1.7 TB，整库一次回读会在上沿顶穿磁盘余量；分批后任何时刻回读目录
不超过一批（单批上限默认 100 GiB = 107,374,182,400 B）。

两个子命令：

  plan  读阶段 5 已落盘的上传装箱计划（plan_upload_batches.py 的输出），把**相邻**的上传批合并到
        单批预算以内，写 readback_plan.json（已存在则读盘复用，保证续跑分组不漂移）。
        最大单对象 > 预算时报错退出——那种对象无论如何放不进一批。
  run   按 readback_plan.json 逐批执行。每批完成写 <state>/<id>.post.txt 与 <id>.done.json，
        中断后重跑跳过已完成批；回读目录在每批开始前必须为空（残留视为上次中断的半批，整体清掉）。
        全部批次完成后把各批 post 行合并、用与上传前清单相同的 `sort -k 2` 排序，与
        SHA256SUMS.pre.txt 逐行比对，完全相同才打印 SHA256_PRE_POST_DIFF=0。

判定行（run）：
  READBACK_BATCH_OK id=<id> files=<n> bytes=<n> du=<n> left_files=0   每批一行
  READBACK_PEAK_BYTES=<n>     每批下完、删前 `du -sb` 回读目录的最大值；n > 上限即非 0 退出
  SHA256_PRE_POST_DIFF=0      仅在全部批次汇总后 post 与 pre 逐行相同时打印，恰一行
  READBACK=PASS batches=<n>

下载源两种：`--source hf`（正式：`hf sync <bucket>/<sub> <verify>/<sub> --include ...`，根下平文件
走 `hf buckets cp`，均断点续传：首试 + 重试 --retries 次，计划口径 27 取 3）；`--source local`（单测：从一个本地目录拷贝，
模拟 bucket，不触网）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK = 8 * 1024 * 1024
DEFAULT_MAX_BYTES = 107_374_182_400  # 100 GiB
# du -sb 还会计入目录项自身的字节数（ext4 每个目录 4096 B）；装箱预算留出这块余量，
# 避免一批文件字节正好贴着上限时被目录开销顶过上限而误判失败。
DEFAULT_SLACK = 1 * 1024**3
DEFAULT_HF_CMD = 'uvx --python 3.11 --from "huggingface_hub[cli]==1.30.0" hf'
ROOT_SUB = ""  # 根下平文件的 sub 标记


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_sums(path: Path) -> dict[str, str]:
    """sha256sum 格式清单 → {相对路径: hex}。路径重复视为清单损坏。"""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        digest, sep, rel = line.partition("  ")
        if not sep or len(digest) != 64:
            raise SystemExit(f"错误: 清单 {path} 行格式异常: {line[:120]}")
        if rel in out:
            raise SystemExit(f"错误: 清单 {path} 路径重复: {rel}")
        out[rel] = digest
    return out


def du_sb(path: Path) -> int:
    r = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True, check=True)
    return int(r.stdout.split()[0])


def list_files(root: Path) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            out.append(str((Path(dirpath) / fn).relative_to(root)))
    return out


def clear_dir(root: Path) -> None:
    """清空回读目录的内容（保留目录本身）。"""
    for child in list(root.iterdir()):
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def guard_verify_dir(verify: Path) -> None:
    """回读目录会被反复整体清空，路径写错代价不可逆：只认名为 verify 且不是 / 或家目录的目录。"""
    v = verify.resolve()
    if v.name != "verify" or v == Path("/") or v == Path.home():
        raise SystemExit(f"错误: 回读目录必须是某个导出根下的 verify/，拒绝清空 {v}")


# ---------------------------------------------------------------------------- plan
def build_readback_plan(upload_plan: dict, stage: Path, max_bytes: int, slack: int) -> dict:
    budget = max_bytes - slack
    if budget <= 0:
        raise SystemExit(f"错误: 单批上限 {max_bytes} 不大于目录余量 {slack}")
    # 单元：每个上传批是一个单元（保持上传时的 --include 分组）；根下平文件逐个成单元。
    units: list[dict] = []
    largest = ("", 0)
    for b in upload_plan["batches"]:
        files = []
        for inner in b["files"]:
            sz = (stage / b["sub"] / inner).stat().st_size
            files.append({"rel": f"{b['sub']}/{inner}", "inner": inner, "bytes": sz})
            if sz > largest[1]:
                largest = (f"{b['sub']}/{inner}", sz)
        units.append({"sub": b["sub"], "upload_id": b["id"], "files": files,
                      "bytes": sum(f["bytes"] for f in files)})
    for rf in upload_plan["root_files"]:
        sz = (stage / rf).stat().st_size
        if sz > largest[1]:
            largest = (rf, sz)
        units.append({"sub": ROOT_SUB, "upload_id": "_root", "files": [
            {"rel": rf, "inner": rf, "bytes": sz}], "bytes": sz})
    if largest[1] > budget:
        raise SystemExit(
            f"错误: 最大单对象 {largest[0]} 为 {largest[1]} B，超过单批预算 {budget} B"
            f"（上限 {max_bytes} − 目录余量 {slack}）；READBACK_BATCH_MAX_BYTES 必须不小于最大单对象"
        )
    # 单元本身超预算（只可能是多文件上传批 > 预算，例如测试里把上限设得很小）时拆成逐文件单元
    flat: list[dict] = []
    for u in units:
        if u["bytes"] <= budget:
            flat.append(u)
        else:
            for f in u["files"]:
                flat.append({"sub": u["sub"], "upload_id": u["upload_id"], "files": [f],
                             "bytes": f["bytes"]})
    # 相邻贪心合并（保持上传计划顺序，不重排）
    batches: list[dict] = []
    cur: list[dict] = []
    cur_bytes = 0
    for u in flat:
        if cur and cur_bytes + u["bytes"] > budget:
            batches.append({"units": cur, "bytes": cur_bytes})
            cur, cur_bytes = [], 0
        cur.append(u)
        cur_bytes += u["bytes"]
    if cur:
        batches.append({"units": cur, "bytes": cur_bytes})
    for i, b in enumerate(batches):
        b["id"] = f"rb-{i:04d}"
        b["files"] = [f["rel"] for u in b["units"] for f in u["files"]]
    total_files = sum(len(b["files"]) for b in batches)
    if total_files != upload_plan["total_files"]:
        raise SystemExit(f"错误: 回读计划文件数 {total_files} != 上传计划 {upload_plan['total_files']}")
    return {
        "max_bytes": max_bytes,
        "slack": slack,
        "budget": budget,
        "largest_object": {"rel": largest[0], "bytes": largest[1]},
        "upload_plan_sha256": upload_plan.get("plan_sha256"),
        "total_files": total_files,
        "total_bytes": sum(b["bytes"] for b in batches),
        "batches": batches,
    }


def cmd_plan(args) -> int:
    upload_plan = json.loads(args.upload_plan.read_text())
    if args.out.exists():
        rp = json.loads(args.out.read_text())
        if rp["max_bytes"] != args.max_bytes or rp["slack"] != args.slack:
            raise SystemExit(
                f"错误: 已落盘回读计划的上限/余量（{rp['max_bytes']}/{rp['slack']}）与本次参数"
                f"（{args.max_bytes}/{args.slack}）不同；要改上限须先删 {args.out} 与 state 目录"
            )
        if rp.get("upload_plan_sha256") != upload_plan.get("plan_sha256"):
            raise SystemExit("错误: 已落盘回读计划对应的上传计划 sha 与当前上传计划不同")
        log("回读计划已存在，读盘复用（不重算分组）")
    else:
        rp = build_readback_plan(upload_plan, args.stage, args.max_bytes, args.slack)
        tmp = args.out.with_name(args.out.name + ".tmp")
        tmp.write_text(json.dumps(rp, ensure_ascii=False, indent=1))
        os.replace(tmp, args.out)
    mx = max(b["bytes"] for b in rp["batches"])
    print(f"READBACK_PLAN batches={len(rp['batches'])} files={rp['total_files']} "
          f"bytes={rp['total_bytes']} max_batch_bytes={mx} budget={rp['budget']} "
          f"max_bytes={rp['max_bytes']} largest_object={rp['largest_object']['bytes']}", flush=True)
    return 0


# ---------------------------------------------------------------------------- run
class Downloader:
    def __init__(self, args):
        self.kind = args.source
        self.bucket = (args.bucket or "").rstrip("/")
        self.local_root = args.local_root
        self.hf_cmd = shlex.split(args.hf_cmd)
        self.retries = args.retries
        self.sleep = args.retry_sleep
        if self.kind == "hf" and not self.bucket:
            raise SystemExit("错误: --source hf 需要 --bucket")
        if self.kind == "local" and (self.local_root is None or not self.local_root.is_dir()):
            raise SystemExit("错误: --source local 需要存在的 --local-root")

    def _retry(self, what: str, fn) -> None:
        attempts = self.retries + 1          # 首试 + 重试 retries 次
        for attempt in range(1, attempts + 1):
            try:
                if fn():
                    return
            except OSError as e:  # 本地拷贝的失败也按可重试处理
                log(f"  {what} 第 {attempt}/{attempts} 次尝试失败: {e}")
            else:
                log(f"  {what} 第 {attempt}/{attempts} 次尝试中断")
            if attempt < attempts:
                time.sleep(self.sleep)
        print(f"READBACK_FAIL object={what} attempts={attempts}", flush=True)
        raise SystemExit(f"错误: {what} 首试 + 重试 {self.retries} 次仍失败")

    def fetch_unit(self, unit: dict, verify: Path) -> None:
        sub, inners = unit["sub"], [f["inner"] for f in unit["files"]]
        if self.kind == "local":
            def go():
                for inner in inners:
                    rel = f"{sub}/{inner}" if sub else inner
                    dst = verify / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(self.local_root / rel, dst)
                return True
            self._retry(f"本地拷贝 [{unit['upload_id']}]", go)
            return
        if sub == ROOT_SUB:
            for inner in inners:
                def go(inner=inner):
                    return subprocess.run(self.hf_cmd + [
                        "buckets", "cp", f"{self.bucket}/{inner}", str(verify / inner)]).returncode == 0
                self._retry(f"cp {inner}", go)
            return
        (verify / sub).mkdir(parents=True, exist_ok=True)
        inc: list[str] = []
        for inner in inners:
            inc += ["--include", inner]

        def go():
            return subprocess.run(self.hf_cmd + [
                "sync", f"{self.bucket}/{sub}", str(verify / sub)] + inc).returncode == 0
        self._retry(f"sync [{sub} {unit['upload_id']}]", go)


def pick_tars(tars: list[str], n: int, seed: int) -> list[str]:
    """按父目录分组每组抽一个，再随机补到 n 个（同 verify_tar_members.pick_tars 的口径）。"""
    if n <= 0 or not tars:
        return []
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for t in sorted(tars):
        groups.setdefault(str(Path(t).parent), []).append(t)
    picked = [rng.choice(groups[k]) for k in sorted(groups)]
    rng.shuffle(picked)
    picked = picked[:n]
    rest = [t for t in sorted(tars) if t not in picked]
    rng.shuffle(rest)
    while len(picked) < n and rest:
        picked.append(rest.pop())
    return sorted(picked)


def check_tar_members(tar_path: Path, members: dict[str, str]) -> tuple[int, list[str]]:
    bad: list[str] = []
    n = 0
    with tarfile.open(tar_path, "r") as tf:
        for ti in tf:
            if not ti.isfile():
                continue
            fo = tf.extractfile(ti)
            if fo is None:
                bad.append(f"{ti.name} 无法读取成员")
                continue
            h = hashlib.sha256()
            while True:
                b = fo.read(CHUNK)
                if not b:
                    break
                h.update(b)
            want = members.get(ti.name)
            if want is None:
                bad.append(f"{ti.name} 不在成员清单里")
            elif want != h.hexdigest():
                bad.append(f"{ti.name} sha256 不符")
            n += 1
    return n, bad


def run_batches(args, downloader=None, before_batch=None) -> int:
    """可在进程内调用（单测用 before_batch 钩子观察每批开始前回读目录的状态）。"""
    rp = json.loads(args.readback_plan.read_text())
    if rp["max_bytes"] != args.max_bytes:
        raise SystemExit(f"错误: 回读计划上限 {rp['max_bytes']} 与本次 --max-bytes {args.max_bytes} 不同")
    pre_path: Path = args.pre
    pre = load_sums(pre_path)
    pre_self_sha = sha256_file(pre_path)
    members = load_sums(args.members_checksums) if args.members_checksums else None
    verify: Path = args.verify
    state: Path = args.state
    guard_verify_dir(verify)
    verify.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    dl = downloader or Downloader(args)

    # 覆盖性：回读计划的文件集必须 = pre 清单 ∪ {清单自身}，一个不多一个不少
    plan_files = [f for b in rp["batches"] for f in b["files"]]
    if len(plan_files) != len(set(plan_files)):
        raise SystemExit("错误: 回读计划里有重复文件")
    plan_set = {f for f in plan_files if not Path(f).name.startswith("SHA256SUMS.")}
    if plan_set != set(pre):
        miss = sorted(set(pre) - plan_set)[:5]
        extra = sorted(plan_set - set(pre))[:5]
        raise SystemExit(f"错误: 回读计划与 pre 清单文件集不符。计划缺 {miss}，计划多 {extra}")

    leftovers = list_files(verify)
    if leftovers:
        log(f"回读目录有 {len(leftovers)} 个残留文件（上次中断的半批），整体清掉后从未完成批续跑")
        clear_dir(verify)

    peak = 0
    nb = len(rp["batches"])
    for i, b in enumerate(rp["batches"]):
        done = state / f"{b['id']}.done.json"
        post_part = state / f"{b['id']}.post.txt"
        if done.exists() and post_part.exists():
            d = json.loads(done.read_text())
            peak = max(peak, d["du"])
            log(f"[{i+1}/{nb}] {b['id']} 已完成（du={d['du']}），跳过")
            continue
        if list_files(verify):
            raise SystemExit(f"错误: 批 {b['id']} 开始前回读目录非空")
        if before_batch is not None:
            before_batch(b, verify)
        log(f"[{i+1}/{nb}] {b['id']} 下载 {len(b['files'])} 件 / {b['bytes']} B")
        for u in b["units"]:
            dl.fetch_unit(u, verify)

        du = du_sb(verify)
        peak = max(peak, du)
        if du > args.max_bytes:
            clear_dir(verify)
            print(f"READBACK_PEAK_BYTES={peak}", flush=True)
            print(f"错误: 批 {b['id']} 回读目录 du={du} 超过单批上限 {args.max_bytes}（已删该批）",
                  flush=True)
            return 1

        got_files = set(list_files(verify))
        want_files = set(b["files"])
        if got_files != want_files:
            print(f"错误: 批 {b['id']} 回读文件集不符。缺 {sorted(want_files - got_files)[:5]}，"
                  f"多 {sorted(got_files - want_files)[:5]}", flush=True)
            return 1

        with ThreadPoolExecutor(max_workers=args.nproc) as pool:
            digests = dict(zip(b["files"], pool.map(lambda r: sha256_file(verify / r), b["files"])))
        bad = []
        lines = []
        for rel in b["files"]:
            if Path(rel).name.startswith("SHA256SUMS."):
                # 清单自身不在清单里：直接与本地 stage 那份逐字节同哈希
                if digests[rel] != pre_self_sha:
                    bad.append(f"{rel} 与本地清单 sha256 不同")
                continue
            if digests[rel] != pre[rel]:
                bad.append(f"{rel} pre={pre[rel]} post={digests[rel]}")
            lines.append(f"{digests[rel]}  {rel}\n")
        if bad:
            print(f"错误: 批 {b['id']} 有 {len(bad)} 个文件 sha256 与上传前清单不符（回读件保留待查）:",
                  flush=True)
            for x in bad[:20]:
                print(f"    {x}", flush=True)
            return 1

        n_members = 0
        tars = [r for r in b["files"] if r.endswith(".tar")]
        sampled = pick_tars(tars, args.sample_tars_per_batch, args.seed + i) if members is not None else []
        for t in sampled:
            n, mbad = check_tar_members(verify / t, members)
            n_members += n
            if mbad:
                print(f"MEMBERS_SPOT=FAIL batch={b['id']} tar={t} mismatches={len(mbad)}", flush=True)
                for x in mbad[:20]:
                    print(f"    {x}", flush=True)
                return 1

        tmp = post_part.with_name(post_part.name + ".tmp")
        tmp.write_text("".join(lines))
        os.replace(tmp, post_part)
        done.write_text(json.dumps({"id": b["id"], "files": len(b["files"]), "bytes": b["bytes"],
                                    "du": du, "tars_sampled": sampled, "members": n_members}))
        clear_dir(verify)
        left = len(list_files(verify))
        if left:
            raise SystemExit(f"错误: 批 {b['id']} 删除后回读目录仍有 {left} 个文件")
        print(f"READBACK_BATCH_OK id={b['id']} files={len(b['files'])} bytes={b['bytes']} du={du} "
              f"tars_sampled={len(sampled)} members={n_members} left_files=0", flush=True)

    # ---- 汇总：post 与 pre 逐行比对 ----
    print(f"READBACK_PEAK_BYTES={peak}", flush=True)
    if peak > args.max_bytes:
        print(f"错误: 回读峰值 {peak} 超过单批上限 {args.max_bytes}", flush=True)
        return 1
    unsorted = args.post_out.with_name(args.post_out.name + ".unsorted")
    with open(unsorted, "w") as fo:
        for b in rp["batches"]:
            fo.write((state / f"{b['id']}.post.txt").read_text())
    # 与上传前清单同一条命令排序（hash_tree 的 `sort -k 2`，继承同一 locale），保证同集合必同序
    with open(args.post_out, "w") as fo:
        subprocess.run(["sort", "-k", "2", str(unsorted)], stdout=fo, check=True)
    unsorted.unlink()
    pre_lines = pre_path.read_text().splitlines()
    post_lines = args.post_out.read_text().splitlines()
    if pre_lines != post_lines:
        import difflib
        print("错误: 上传前后 sha256 清单逐行不一致：", flush=True)
        for x in list(difflib.unified_diff(pre_lines, post_lines, "pre", "post", lineterm=""))[:40]:
            print(f"    {x}", flush=True)
        return 1
    print("SHA256_PRE_POST_DIFF=0", flush=True)
    print(f"READBACK=PASS batches={nb} lines={len(post_lines)} peak={peak} max={args.max_bytes}",
          flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("plan", help="由上传装箱计划合并相邻批，落盘回读计划")
    p.add_argument("--upload-plan", type=Path, required=True)
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    p.add_argument("--slack", type=int, default=DEFAULT_SLACK)

    r = sp.add_parser("run", help="按回读计划逐批下载、比对、删除")
    r.add_argument("--readback-plan", type=Path, required=True)
    r.add_argument("--pre", type=Path, required=True, help="stage 侧 SHA256SUMS.pre.txt")
    r.add_argument("--verify", type=Path, required=True, help="回读目录（每批清空）")
    r.add_argument("--state", type=Path, required=True, help="每批 post 分片与完成标记目录")
    r.add_argument("--post-out", type=Path, required=True, help="汇总后的 post 清单")
    r.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    r.add_argument("--members-checksums", type=Path, default=None,
                   help="tar 成员清单（sha256-source-files.txt）；给出才做成员抽样")
    r.add_argument("--sample-tars-per-batch", type=int, default=1)
    r.add_argument("--seed", type=int, default=20261016)
    r.add_argument("--nproc", type=int, default=16)
    r.add_argument("--source", choices=("hf", "local"), required=True)
    r.add_argument("--bucket", default=None, help="hf://buckets/<id>")
    r.add_argument("--local-root", type=Path, default=None, help="仅 --source local：模拟 bucket 的目录")
    r.add_argument("--hf-cmd", default=DEFAULT_HF_CMD)
    r.add_argument("--retries", type=int, default=3, help="首试之后的重试次数（口径 27：3）")
    r.add_argument("--retry-sleep", type=float, default=60.0)

    args = ap.parse_args(argv)
    if args.cmd == "plan":
        return cmd_plan(args)
    return run_batches(args)


if __name__ == "__main__":
    sys.exit(main())
