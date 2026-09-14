#!/usr/bin/env python3
"""把 stage 目录装箱成上传批次计划，落盘成 JSON 供 driver 逐批执行。

**为什么按字节而不是按个数分批**：2026-09-08 实测的失败模式是服务端 `new_upload_commit`
随**批内字节量**增长而超时——1.62 GB × 10 个 = 16.2 GB 过，2.0 GB × 10 个 = 20 GB 挂，
2.0 GB × 5 个 = 10 GB 稳。而本轮的 `wan_chunk_latents/*.bin` 是重尾分布（最小 33.7 MiB、
中位 132 MiB、最大 612 MiB），固定「每批 N 个」会让批内字节量在 0.9–15 GB 之间浮动，最坏
一批直接顶到已知会挂的量级。所以字节预算是硬约束，件数上限只是对「单次 commit 能带多少
条目」这个未知维度的保守兜底。

**为什么计划要落盘**：源目录任何变动都会让贪心装箱的分组漂移，续跑时已传的批和新分组对不上，
`--include` 模式也会失配。首跑写盘、重跑读盘，分组就是稳定的。

用法：
  uv run scripts/dataset/hf_export/plan_upload_batches.py \
      --stage <stage 根> --out <plan.json> [--max-bytes N] [--max-files N]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# 8 GiB：在实测稳过的 10 GB 之下再留 20% 余量
DEFAULT_MAX_BYTES = 8 * 1024**3
# 对象个数这一维在 400ep 里没有 >10 的证据（最大批就是 10 个对象），取一个保守上限兜底
DEFAULT_MAX_FILES = 256


def pack(items: list[tuple[str, int]], max_bytes: int, max_files: int) -> list[list[str]]:
    """按给定顺序贪心装箱。保持输入顺序（不重排），便于人读日志时定位进度。

    单个文件超预算时独占一批——它无论如何都得自己走一次 commit。
    """
    batches: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for name, size in items:
        if cur and (cur_bytes + size > max_bytes or len(cur) >= max_files):
            batches.append(cur)
            cur, cur_bytes = [], 0
        cur.append(name)
        cur_bytes += size
    if cur:
        batches.append(cur)
    return batches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    args = ap.parse_args()

    stage = args.stage.resolve()
    if not stage.is_dir():
        sys.exit(f"错误: stage 不存在 {stage}")

    # 按 bucket 内前缀（= stage 下的一级目录）分组；stage 根下的平文件单独走 `hf buckets cp`
    # （`hf sync` 的源必须是目录）
    groups: dict[str, list[tuple[str, int]]] = {}
    root_files: list[str] = []
    for p in sorted(stage.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(stage)
        if len(rel.parts) == 1:
            root_files.append(str(rel))
            continue
        sub = rel.parts[0]
        inner = str(Path(*rel.parts[1:]))
        groups.setdefault(sub, []).append((inner, p.stat().st_size))

    batches = []
    for sub in sorted(groups):
        for i, files in enumerate(pack(groups[sub], args.max_bytes, args.max_files)):
            nbytes = sum(dict(groups[sub])[f] for f in files)
            batches.append({
                "id": f"{sub}-{i:04d}",
                "sub": sub,
                "files": files,
                "count": len(files),
                "bytes": nbytes,
            })

    total_files = sum(b["count"] for b in batches) + len(root_files)
    total_bytes = (sum(b["bytes"] for b in batches)
                   + sum((stage / f).stat().st_size for f in root_files))
    plan = {
        "stage": str(stage),
        "max_bytes": args.max_bytes,
        "max_files": args.max_files,
        "batches": batches,
        "root_files": sorted(root_files),
        "total_files": total_files,
        "total_bytes": total_bytes,
    }
    body = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True)
    plan["plan_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))

    mx_b = max((b["bytes"] for b in batches), default=0)
    mx_f = max((b["count"] for b in batches), default=0)
    print(f"PLAN_BATCHES={len(batches)} PLAN_FILES={total_files} PLAN_BYTES={total_bytes} "
          f"ROOT_FILES={len(root_files)} MAX_BATCH_BYTES={mx_b} MAX_BATCH_FILES={mx_f} "
          f"PLAN_SHA={plan['plan_sha256'][:16]}")
    if mx_b > args.max_bytes:
        print(f"  注意: 有单文件超预算独占一批（{mx_b} > {args.max_bytes}）")
    for sub in sorted(groups):
        n = sum(1 for b in batches if b["sub"] == sub)
        print(f"  {sub}: {len(groups[sub])} 件 → {n} 批")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
