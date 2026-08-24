#!/usr/bin/env python3
"""把采样文件汇总成档位判据需要的几个数。

本机扫档与集群探针**共用这一份**，保证两边的指标定义逐字相同——否则「本地筛出的档位
到集群上还成不成立」这个问题就无从谈起了。

指标口径三条：
  · GPU 平均/中位利用率：**跳过前 warmup 个采样点**。开头那段是 SigLIP 权重加载
    （1.66 GB pickle → jnp）加 XLA 编译，GPU 基本闲着；算进均值会把所有档位一起
    压低、掩盖档位之间的真实差异。
  · 内存看 cgroup 的 **anon + shmem**，不看 MaxRSS。greatlakes.md 已实证 MaxRSS 会精确
    贴住 --mem 申请上限（47.71/48、15.77/16），那是页缓存填满配额的读数，不是工作集；
    据此推断「必须要这么多内存」正是当年被证伪的错误。
  · file 页缓存单列出来，用来解释 cg_current 为什么总是贴着上限。
"""

from __future__ import annotations

import argparse
import json
import statistics

FIELDS = ("gpu_util", "gpu_mib", "cg_current", "cg_anon", "cg_file", "cg_shmem")


def _gib(b: float) -> float:
    return b / 1073741824


def summarize(path: str, warmup: int = 6) -> dict:
    util: list[float] = []
    peak: dict[str, float] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("sample "):
                continue
            kv = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
            for k in FIELDS:
                try:
                    v = float(kv.get(k, 0))
                except ValueError:
                    continue
                peak[k] = max(peak.get(k, 0.0), v)
                if k == "gpu_util":
                    util.append(v)
    warm = util[warmup:] or util
    return {
        "samples": len(util),
        "gpu_util_mean": round(statistics.mean(warm), 2) if warm else 0.0,
        "gpu_util_median": round(statistics.median(warm), 2) if warm else 0.0,
        "gpu_util_max": round(max(util), 2) if util else 0.0,
        "gpu_mib_peak": round(peak.get("gpu_mib", 0.0)),
        "cg_current_peak_gib": round(_gib(peak.get("cg_current", 0.0)), 3),
        "cg_anon_peak_gib": round(_gib(peak.get("cg_anon", 0.0)), 3),
        "cg_shmem_peak_gib": round(_gib(peak.get("cg_shmem", 0.0)), 3),
        "cg_file_peak_gib": round(_gib(peak.get("cg_file", 0.0)), 3),
        "cg_workingset_peak_gib": round(
            _gib(peak.get("cg_anon", 0.0) + peak.get("cg_shmem", 0.0)), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("samples")
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--tier", default="")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    s = summarize(args.samples, args.warmup)
    if args.json:
        print(json.dumps(s))
    else:
        kv = " ".join(f"{k}={v}" for k, v in s.items())
        print(f"TIER_SUMMARY tier={args.tier} {kv}")


if __name__ == "__main__":
    main()
