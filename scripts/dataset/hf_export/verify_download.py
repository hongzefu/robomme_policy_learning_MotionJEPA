#!/usr/bin/env python3
"""回读校验：对 bucket 下载回来的 verify/ 副本做字节级 sha256 闭环。

三步：
  A. stage/ 与 verify/ 的相对文件集合完全一致；
  B. 清单/说明类小文件逐字节一致（checksums/、manifest/、README.md）；
  C. 用系统 sha256sum -c 对 verify/ 逐文件重算哈希，对照打包时生成的
     sha256-shards.txt + sha256-packed.txt（清单切块并行跑，加速大体量校验）。
全部通过打 RESULT=PASS（退出码 0），否则 RESULT=FAIL（退出码 1）。
"""

import argparse
import filecmp
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_ROOT = Path("/data/hongzefu/hf-export-robomme-vla-motionjepa-v1")


def rel_files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def run_sha256sum_chunk(chunk_file: Path, cwd: Path) -> tuple[int, str]:
    p = subprocess.run(
        ["sha256sum", "-c", "--quiet", str(chunk_file)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return p.returncode, (p.stdout + p.stderr).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    stage, verify = args.export_root / "stage", args.export_root / "verify"
    fail = False

    # A. 文件集合一致
    s_set, v_set = rel_files(stage), rel_files(verify)
    if s_set != v_set:
        fail = True
        for m in sorted(s_set - v_set)[:10]:
            print(f"缺失于 verify/: {m}")
        for m in sorted(v_set - s_set)[:10]:
            print(f"多出于 verify/: {m}")
        print(f"文件集合不一致：stage {len(s_set)} vs verify {len(v_set)}")
    else:
        print(f"文件集合一致：{len(s_set)} 个文件")

    # B. 清单与说明文件逐字节一致
    for rel in sorted(s_set & v_set):
        if rel.startswith(("checksums/", "manifest/")) or rel == "README.md":
            if not filecmp.cmp(stage / rel, verify / rel, shallow=False):
                print(f"清单文件内容不一致: {rel}")
                fail = True

    # C. sha256sum -c（清单切块并行）
    lines: list[str] = []
    for name in ("sha256-shards.txt", "sha256-packed.txt"):
        lines += (stage / "checksums" / name).read_text().splitlines()
    lines = [l for l in lines if l.strip()]
    nw = max(1, min(args.workers, len(lines)))
    with tempfile.TemporaryDirectory() as td:
        chunks = []
        for i in range(nw):
            cf = Path(td) / f"chunk_{i:02d}.txt"
            cf.write_text("\n".join(lines[i::nw]) + "\n")
            chunks.append(cf)
        with ThreadPoolExecutor(max_workers=nw) as pool:
            results = list(pool.map(lambda c: run_sha256sum_chunk(c, verify), chunks))
    n_bad = 0
    for rc, out in results:
        if rc != 0:
            n_bad += 1
            fail = True
            print(out)
    print(f"sha256sum -c：{len(lines)} 行清单，{nw} 路并行，失败块 {n_bad} 个")

    print("RESULT=FAIL" if fail else "RESULT=PASS")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
