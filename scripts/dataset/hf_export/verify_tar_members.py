#!/usr/bin/env python3
"""tar 成员级抽样复算：从回读目录里随机抽几个 tar，流式逐成员算 sha256 对清单。

为什么抽样而不是全解：driver 的 SHA256SUMS.pre/post 两遍独立重算已经逐字节覆盖了这些
tar 的全部内容，成员级只是换一个视角看同一批字节。全量解包要再落地 122 GB、多一遍
open/read，收益为零。抽样要防的是另一类失效——打包侧把成员名写错、或清单与 tar 内容
对不上，这种错误在任何一个分片上都会暴露，抽几个就够。

流式解包（tarfile.extractfile 逐块读），**不落地任何文件**。

判定行：MEMBERS_SPOT=PASS tars=<n> members=<n> mismatches=0
"""

import argparse
import hashlib
import random
import sys
import tarfile
from pathlib import Path

CHUNK = 8 * 1024 * 1024


def load_checksums(path: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            digest, _, name = line.partition("  ")
            m[name] = digest
    return m


def pick_tars(root: Path, n: int, seed: int) -> list[Path]:
    """按顶层类别分组后每组抽一个，凑不够 n 个再从剩余里随机补。

    分组是刻意的：data / features / wan-latents / oracle 四类 tar 的成员命名规则各不相同，
    只在同一类里抽会漏掉另外三类的命名错误。
    """
    tars = sorted(root.rglob("*.tar"))
    if not tars:
        return []
    groups: dict[str, list[Path]] = {}
    for t in tars:
        groups.setdefault(t.parent.name, []).append(t)
    rng = random.Random(seed)
    picked: list[Path] = []
    for key in sorted(groups):
        picked.append(rng.choice(groups[key]))
    rest = [t for t in tars if t not in picked]
    rng.shuffle(rest)
    while len(picked) < n and rest:
        picked.append(rest.pop())
    return sorted(picked[:n]) if len(picked) > n else sorted(picked)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-root", type=Path, required=True)
    ap.add_argument("--checksums", type=Path, required=True,
                    help="sha256-source-files.txt（tar 成员名 → sha256）")
    ap.add_argument("--sample-tars", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()

    if not args.checksums.is_file():
        print(f"错误: 清单不存在 {args.checksums}")
        return 1
    sums = load_checksums(args.checksums)
    print(f"  清单载入 {len(sums)} 行")

    tars = pick_tars(args.verify_root, args.sample_tars, args.seed)
    if not tars:
        print(f"错误: {args.verify_root} 下没有 .tar")
        return 1

    total_members = 0
    mismatches: list[str] = []
    for t in tars:
        n_here = 0
        with tarfile.open(t, "r") as tf:
            for ti in tf:
                if not ti.isfile():
                    continue
                fo = tf.extractfile(ti)
                if fo is None:
                    mismatches.append(f"{t.name}:{ti.name} 无法读取成员")
                    continue
                h = hashlib.sha256()
                while True:
                    b = fo.read(CHUNK)
                    if not b:
                        break
                    h.update(b)
                want = sums.get(ti.name)
                if want is None:
                    mismatches.append(f"{t.name}:{ti.name} 不在清单里")
                elif want != h.hexdigest():
                    mismatches.append(f"{t.name}:{ti.name} sha256 不符")
                n_here += 1
        total_members += n_here
        print(f"  {t.relative_to(args.verify_root)}: {n_here} 个成员已复算")

    if mismatches:
        print(f"MEMBERS_SPOT=FAIL tars={len(tars)} members={total_members} "
              f"mismatches={len(mismatches)}")
        for m in mismatches[:20]:
            print(f"    {m}")
        return 1
    print(f"MEMBERS_SPOT=PASS tars={len(tars)} members={total_members} mismatches=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
