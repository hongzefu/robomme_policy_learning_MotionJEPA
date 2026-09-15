#!/usr/bin/env python3
"""严格比较 modulation 的固定输入、初态、loss 和全部可训练叶梯度。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def compare(a, b):
    """拒绝缺失与空判；先校验输入和初态，再比梯度。"""
    for name in ("schema", "seed", "fsdp_devices", "xla_flags", "environment"):
        if a[name] != b[name]:
            raise ValueError(f"运行口径不同：{name}")
    if a["schema"] != "modul-fixed-grad-v1":
        raise ValueError("摘要 schema 不支持")
    if a["source"]["history_config_sha256"] != b["source"]["history_config_sha256"]:
        raise ValueError("history YAML 不同源")
    initial = a["initial_params"]
    if not initial or initial != b["initial_params"]:
        raise ValueError("两侧初态参数树或逐叶摘要不同")
    print(f"INIT_EQ=PASS leaves={len(initial)} mismatches=0")
    kinds = {"mixed1", "allshort", "allfull"}
    if set(a["results"]) != kinds or set(b["results"]) != kinds:
        raise ValueError("必须覆盖三类 batch")
    leaves = set()
    for kind in sorted(kinds):
        x, y = a["results"][kind], b["results"][kind]
        for name in ("batch_id", "indices", "batch_keys", "loss_hex", "per_leaf", "n_leaves"):
            if x[name] != y[name]:
                raise ValueError(f"{kind} 不等：{name}")
        for row in (x, y):
            if row["finite"] is not True or not math.isfinite(float.fromhex(row["loss_hex"])):
                raise ValueError(f"{kind} 含非有限数")
            if not row["per_leaf"] or row["n_leaves"] != len(row["per_leaf"]):
                raise ValueError(f"{kind} 梯度叶数不符或为空")
            if len(row["indices"]) != 8 or len(row["batch_keys"]) != 12:
                raise ValueError(f"{kind} 输入范围不符")
        leaves.add(x["n_leaves"])
    if len(leaves) != 1:
        raise ValueError("三类 batch 的梯度叶数不一致")
    print(f"GRAD_EQ=PASS kinds=3 leaves={leaves.pop()} mismatches=0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args()
    try:
        compare(json.loads(args.a.read_text()), json.loads(args.b.read_text()))
    except (ValueError, KeyError, TypeError) as error:
        print(f"GRAD_EQ=FAIL reason={error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
