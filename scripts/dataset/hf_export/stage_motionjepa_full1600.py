#!/usr/bin/env python3
"""把 MotionJEPA 本机建库产物组装成待上传的 stage 树（硬链接，不复制字节）。

**为什么硬链接而不是复制**：最终产物 479 GB，复制一份要多占 479 GB 磁盘并多读写一遍。
硬链接同 inode、零拷贝，代价是「stage 里改一个字节等于改原件」——所以本脚本只建链接、
绝不写入 stage 里的任何已有文件，driver 的收尾阶段还会对源侧重算一遍 sha256 反证原件未动。

**stage 里为什么不能出现符号链接**：上传端遍历不跟随符号链接目录，其内容会被静默整体漏传
（400ep 那轮踩过，driver 里保留了显式断言）。硬链接在 os.walk 眼里就是普通文件，没有这个问题。

范围（用户 2026-09-14 拍板「只传最终产物 + 溯源」）：
  - dataset-token/ 下 PUBLISHED.json 清单列的 2804 个文件，逐条按清单自带的 sha256 核对；
  - PUBLISHED.json 自身（清单不含自己）；
  - control/ 全部 2985 个溯源文件，其中 content_hashes/ 的 2800 个小文件打成一个 tar
    （2800 个 31 KB 均值的对象直传会退化成逐文件 HTTP 往返，且它们是逐 bin 的 sha 边车、
    没有单取需求；其余 185 个含 source_pin / model_pin / environment 等，原样直传保持可读）。
不传：data-raw/、reference*/、smoke28/、logs/，以及 dataset-token 下 5618 个清单外的边车
（*.sha256 / *.complete.json / numeric_mode/ / performance/）——边车的 P7 命中（构建机内部
主机名 ip-*.compute.internal）共 8400 处，正是靠「不在发布清单里」这一条自然挡在门外。

幂等：已存在且 inode 与源相同的链接跳过；tar 已存在且成员数吻合则跳过重打。
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

CONTROL_TAR_DIRS = ("content_hashes",)   # 打 tar 的 control 子目录


def link_one(src: Path, dst: Path) -> str:
    """硬链接 src → dst。返回 'linked' / 'skip'。

    src 若是符号链接则先解引用再链——stage 里绝不能出现符号链接，`hf sync` 的本地遍历是
    `os.walk(followlinks=False)`，符号链接目录的内容会被静默整体漏传（400ep 那轮踩过）。
    """
    real = Path(os.path.realpath(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.stat().st_ino == real.stat().st_ino:
            return "skip"
        dst.unlink()          # 残留的异源文件（上一轮中断）：删链接不动源文件
    os.link(real, dst)
    return "linked"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="dataset-local-a100-4task-full1600 根")
    ap.add_argument("--stage", type=Path, required=True)
    args = ap.parse_args()

    token = args.src / "dataset-token"
    control = args.src / "control"
    stage = args.stage
    for p in (token, control):
        if not p.is_dir():
            sys.exit(f"错误: 源目录不存在 {p}")

    published = json.loads((token / "PUBLISHED.json").read_text())
    files = published["files"]           # {相对路径: sha256}
    stage.mkdir(parents=True, exist_ok=True)

    # ---- 1. 发布清单 2804 个 ----
    linked = skipped = 0
    for rel in files:
        src = token / rel
        if not src.is_file():
            sys.exit(f"错误: 清单文件缺失 {src}")
        if link_one(src, stage / rel) == "linked":
            linked += 1
        else:
            skipped += 1
    # PUBLISHED.json 自身：清单不含自己，但下游要靠它核对，必须一起发
    link_one(token / "PUBLISHED.json", stage / "PUBLISHED.json")
    print(f"STAGE_PUBLISHED files={len(files)} linked={linked} skipped={skipped}")

    # ---- 2. control/ 直传件 ----
    c_linked = c_skip = 0
    for dirpath, dirnames, filenames in os.walk(control):
        rel_dir = Path(dirpath).relative_to(control)
        top = rel_dir.parts[0] if rel_dir.parts else ""
        if top in CONTROL_TAR_DIRS:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in sorted(dirnames) if d != "__pycache__"]
        for name in sorted(filenames):
            src = Path(dirpath) / name
            if name.endswith((".lock", ".pyc")) or name == "run.lock":
                continue          # 运行期锁文件不是溯源内容
            if link_one(src, stage / "control" / rel_dir / name) == "linked":
                c_linked += 1
            else:
                c_skip += 1
    print(f"STAGE_CONTROL_PLAIN linked={c_linked} skipped={c_skip}")

    # ---- 3. control 子目录打 tar ----
    for sub in CONTROL_TAR_DIRS:
        srcdir = control / sub
        if not srcdir.is_dir():
            print(f"  跳过（不存在）: control/{sub}")
            continue
        members = sorted(p for p in srcdir.rglob("*") if p.is_file())
        out = stage / "control" / f"{sub}.tar"
        if out.exists():
            with tarfile.open(out, "r") as tf:
                have = sum(1 for ti in tf if ti.isfile())
            if have == len(members):
                print(f"STAGE_CONTROL_TAR {sub} members={have} (已存在，跳过)")
                continue
            out.unlink()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tar.tmp")
        # 固定 mtime/uid/gid/mode：让同样的输入在任何机器上打出逐位相同的 tar，
        # 否则重跑会生成新字节、Xet 认成新对象，白传一遍。
        with tarfile.open(tmp, "w", format=tarfile.GNU_FORMAT) as tf:
            for m in members:
                ti = tf.gettarinfo(str(m), arcname=str(m.relative_to(srcdir)))
                ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname, ti.mode = 0, 0, 0, "", "", 0o644
                with open(m, "rb") as fh:
                    tf.addfile(ti, fh)
        tmp.rename(out)
        print(f"STAGE_CONTROL_TAR {sub} members={len(members)} bytes={out.stat().st_size}")

    # ---- 4. 断言：stage 内不得有符号链接 ----
    syms = [str(p) for p in stage.rglob("*") if p.is_symlink()]
    if syms:
        sys.exit(f"错误: stage 内出现符号链接 {syms[:5]}")

    n = sum(1 for p in stage.rglob("*") if p.is_file())
    b = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    print(f"STAGE_DONE files={n} bytes={b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
