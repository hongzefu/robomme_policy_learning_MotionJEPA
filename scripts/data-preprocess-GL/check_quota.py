#!/usr/bin/env python3
"""校验本次提交是否装得下 chaijy2 的剩余配额（GPU / CPU / MEM 三维）。

单独成文件而不是内联在 step1_submit.sh 里：内联 python 在 bash 里要处理三层嵌套引号，
本轮已经在这里踩过两次（一次是「管道喂 stdin + heredoc 传源码」互相顶掉导致
sys.stdin 读到空串、三维配额全部「未解析出」；一次是 -c 单引号里的转义被当字面量）。
pre-flight 是拦住 8 个 job 集体炸掉的最后一道闸，不该建在这种脆弱写法上。

输入取自集群两条只读命令的输出拼接（中间用 --- 分隔）：
    sacctmgr -nP show assoc account=chaijy2 format=GrpTRES   → 配额上限（account 层，
                                                                user 层没有 GrpTRES）
    squeue -A chaijy2 -t RUNNING -h -O tres-alloc             → 组内当前 RUNNING 占用
"""

from __future__ import annotations

import argparse
import re
import sys

_UNIT = {"": 1 / 1024, "K": 1 / 1048576, "M": 1 / 1024, "G": 1.0, "T": 1024.0}
_TRES = re.compile(r"(cpu|mem|gres/gpu)=([0-9.]+)([KMGT]?)")
_DIMS = (("gres/gpu", "GPU"), ("cpu", "CPU"), ("mem", "MEM(G)"))


def to_gib(value: str, unit: str) -> float:
    """squeue/sacctmgr 的 mem 带单位后缀，无后缀按 MB 记（Slurm 的默认）。"""
    return float(value) * _UNIT.get(unit, 1.0)


def parse(text: str) -> tuple[dict[str, float], dict[str, float]]:
    head, _, body = text.partition("---")
    limits: dict[str, float] = {}
    for key, val, unit in _TRES.findall(head):
        limits[key] = to_gib(val, unit) if key == "mem" else float(val)
    used = {"cpu": 0.0, "mem": 0.0, "gres/gpu": 0.0}
    for line in body.splitlines():
        for key, val, unit in _TRES.findall(line):
            used[key] += to_gib(val, unit) if key == "mem" else float(val)
    return limits, used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quota_text", required=True)
    ap.add_argument("--need_gpu", type=int, required=True)
    ap.add_argument("--need_cpu", type=int, required=True)
    ap.add_argument("--need_mem_gb", type=int, required=True)
    args = ap.parse_args()

    limits, used = parse(args.quota_text)
    need = {"gres/gpu": args.need_gpu, "cpu": args.need_cpu, "mem": args.need_mem_gb}
    ok = True
    for key, label in _DIMS:
        if key not in limits:
            print(f"  · {label} 配额未解析出，跳过该维校验")
            continue
        left = limits[key] - used[key]
        mark = "✓" if left >= need[key] else "✗"
        if left < need[key]:
            ok = False
        print(f"  {mark} {label}: 配额 {limits[key]:.0f} 已用 {used[key]:.0f} "
              f"余 {left:.0f}，本次需 {need[key]}（占配额 {need[key] / limits[key]:.0%}）")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
