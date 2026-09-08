#!/usr/bin/env python3
"""公开前体检：扫描将要上传到公开 HF bucket 的文件，拦截内部路径 / 凭据 / 未公开信息。

**只读工具，没有 --fix。** 这是刻意的：本库的 store_meta / episode_manifest 等文件的
整文件 sha256 是训练 provenance 门与建库契约链的锚点（`scripts/training/g0/
check_config_provenance.py` 的 motion_store_meta_sha256、各 store_meta 的
manifest_sha256），改一个字节就会让公开出去的库再也过不了自己的门。发现问题的正确处置
是「排除该路径」或「在 README 里披露」，绝不是改数据字节。

判定：
  exit 0  全部命中都在 allowlist 里有显式裁决，且数量与 expected_count 吻合
  exit 1  有未裁决命中，或已裁决条目的命中数与 expected_count 不符（源库变了）
  exit 2  命中疑似凭据（P5），硬停

用法：
  uv run scripts/dataset/hf_export/scan_hygiene.py \
    --root v1-store/datasets/4task-motion-400ep \
    --extra v1-store/train-assets/mme_vla_suite/robomme-400ep/robomme/norm_stats.json \
    --allowlist scripts/dataset/hf_export/hygiene_allowlist.json \
    --report v1-store/exports/hf-dataset-4task-motion-400ep/logs/hygiene.json
"""

import argparse
import fnmatch
import json
import os
import re
import sys
from pathlib import Path

# 逐行匹配的模式表。severity: secret > gate > notice
# secret 命中即 exit 2；gate / notice 命中都必须在 allowlist 里有裁决，否则 exit 1。
# notice 与 gate 的区别只在报告里的醒目程度，闸门强度相同——「必须有人明确说过 OK」。
PATTERNS: list[tuple[str, str, str, str]] = [
    # (id, severity, 正则, 说明)
    ("P5", "secret",
     r"hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}"
     r"|BEGIN [A-Z ]*PRIVATE KEY|WANDB_API_KEY\s*[:=]\s*\S",
     "疑似凭据（HF token / AWS key / OpenAI key / 私钥 / wandb key）"),
    ("P1", "gate", r"/nfs/turbo", "集群 NFS 绝对路径"),
    ("P2", "gate", r"coe-chaijy", "实验室 NFS 组织层级"),
    ("P3", "gate", r"/data/hongzefu", "环境 A 本机绝对路径"),
    ("P4", "gate", r"(?i)slurm[_-]?jobs?\W{0,3}[:=]\W{0,3}[\"']?\d+", "Slurm job id"),
    ("P6", "gate", r"github\.com[:/][A-Za-z0-9_.-]+/", "GitHub 仓库地址"),
    # 前瞻否定里带 `_`：仓库名 robomme_policy_learning_MotionJEPA 由 P9 覆盖，
    # P10 只抓独立的 MotionJEPA 引用（如 mj_repo_path 的 .../MotionJEPA）
    ("P10", "gate", r"(?<![A-Za-z_])MotionJEPA(?![A-Za-z])", "非公开仓库名（独立引用）"),
    ("P11", "gate", r"400\s*ep\s*录制|私有.{0,12}录制", "未公开数据集（环境 A 私有录制版）"),
    ("P7", "notice", r"ip-\d+-\d+-\d+-\d+|compute\.internal", "构建机内部主机名"),
    ("P8", "notice", r"GPU-[0-9a-f]{8}-", "GPU UUID"),
    ("P9", "notice", r"/scratch/hongze|robomme_policy_learning_MotionJEPA",
     "构建机绝对路径 / 仓库名"),
]

TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".md", ".yaml", ".yml", ".csv", ".sh", ".py"}
# 已知二进制的快速路径：本库有 22.4 万个 .pkl/.npy，逐个嗅探 8 KiB 等于多读 1.8 GB
# 并做 22 万次 open，而它们不可能是文本。缺了这条黑名单扫描要多花几分钟。
BINARY_SUFFIXES = {".pkl", ".npy", ".npz", ".bin", ".tar", ".pt", ".pth", ".h5", ".safetensors",
                   ".xz", ".gz", ".zip", ".png", ".jpg", ".jpeg", ".mp4", ".so"}
SNIFF = 8192


def is_text(path: Path) -> bool:
    """扩展名黑/白名单优先；其余按前 8 KiB 嗅探（有 NUL 字节即判二进制）。"""
    suf = path.suffix.lower()
    if suf in BINARY_SUFFIXES:
        return False
    if suf in TEXT_SUFFIXES:
        return True
    try:
        with open(path, "rb") as f:
            head = f.read(SNIFF)
    except OSError:
        return False
    if not head or b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def iter_files(root: Path, excludes: list[str]) -> tuple[list[Path], list[str]]:
    """返回 (要扫描的文件, 被 exclude 规则剔除的相对路径)。"""
    keep: list[Path] = []
    skipped: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            rel = str(p.relative_to(root))
            if any(fnmatch.fnmatch(rel, g) for g in excludes):
                skipped.append(rel)
                continue
            keep.append(p)
    return keep, skipped


def scan_file(path: Path, rel: str, compiled: list[tuple[str, str, re.Pattern, str]]) -> list[dict]:
    hits = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                for pid, sev, rx, desc in compiled:
                    if rx.search(line):
                        hits.append({
                            "pattern": pid, "severity": sev, "desc": desc,
                            "path": rel, "line": lineno,
                            # 只留 120 字符上下文，且 secret 命中不回显原文
                            "excerpt": "" if sev == "secret" else line.strip()[:120],
                        })
    except OSError as e:
        hits.append({"pattern": "IO", "severity": "gate", "desc": f"读取失败: {e}",
                     "path": rel, "line": 0, "excerpt": ""})
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="要扫描的目录（将要上传的内容）")
    ap.add_argument("--extra", type=Path, action="append", default=[],
                    help="额外单独扫描的文件（如 train-assets 下的 norm_stats.json），可多次")
    ap.add_argument("--allowlist", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--scope", choices=("source", "stage"), default="source",
                    help="source=扫源库（expected_count 严格校验）；"
                         "stage=扫打包后的 stage（那里小文件目录已变成 tar，"
                         "源库侧的计数不再适用，只校验有无未裁决命中）")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"错误: --root 不是目录 {root}", flush=True)
        return 1

    allow = json.loads(args.allowlist.read_text()) if args.allowlist.exists() else {"entries": []}
    entries = allow.get("entries", [])
    excludes = [e["path_glob"] for e in entries if e.get("decision") == "exclude"]

    compiled = [(pid, sev, re.compile(rx), desc) for pid, sev, rx, desc in PATTERNS]

    files, skipped = iter_files(root, excludes)
    hits: list[dict] = []
    n_scanned = 0
    for p in files:
        if not is_text(p):
            continue
        n_scanned += 1
        hits.extend(scan_file(p, str(p.relative_to(root)), compiled))
    for extra in args.extra:
        ep = extra.resolve()
        if not ep.is_file():
            print(f"错误: --extra 不存在 {ep}", flush=True)
            return 1
        if is_text(ep):
            n_scanned += 1
            hits.extend(scan_file(ep, f"<extra>/{ep.name}", compiled))

    # ---- 对照 allowlist ----
    accepts = [e for e in entries if e.get("decision") == "accept"]
    matched_counts: dict[int, int] = {i: 0 for i in range(len(accepts))}
    ungated: list[dict] = []
    secrets: list[dict] = []
    for h in hits:
        if h["severity"] == "secret":
            secrets.append(h)
            continue
        for i, e in enumerate(accepts):
            if e["pattern"] == h["pattern"] and fnmatch.fnmatch(h["path"], e["path_glob"]):
                matched_counts[i] += 1
                break
        else:
            ungated.append(h)

    # expected_count 只在条目声明的 scope 上校验。理由：stage 里 wan-latents/ 等小文件目录
    # 已经变成 tar（二进制、不扫描），源库侧那些计数在 stage 上必然为 0——拿同一份数字去卡
    # 两边只会制造误停。两侧共同的闸门是「不得有未裁决命中」，那一条对两个 scope 都生效。
    count_mismatch = []
    for i, e in enumerate(accepts):
        if e.get("scope", "both") not in (args.scope, "both"):
            continue
        got = matched_counts[i]
        exp = e.get("expected_count")
        if exp is not None and got != exp:
            count_mismatch.append({"pattern": e["pattern"], "path_glob": e["path_glob"],
                                   "expected": exp, "got": got})

    # 按 pattern × 顶层目录聚合：这是裁决 allowlist 时唯一好用的视图
    by_pattern_dir: dict[str, int] = {}
    for h in hits:
        top = h["path"].split("/")[0] if "/" in h["path"] else h["path"]
        by_pattern_dir[f'{h["pattern"]}|{top}'] = by_pattern_dir.get(f'{h["pattern"]}|{top}', 0) + 1

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "root": str(root),
        "scope": args.scope,
        "files_scanned": n_scanned,
        "files_excluded": len(skipped),
        "excluded_globs": excludes,
        "total_hits": len(hits),
        "ungated": ungated[:200],
        "ungated_total": len(ungated),
        "count_mismatch": count_mismatch,
        "by_pattern": {pid: sum(1 for h in hits if h["pattern"] == pid)
                       for pid, _, _, _ in PATTERNS},
        "by_pattern_dir": by_pattern_dir,
        "accept_entries": [{**e, "got": matched_counts[i]} for i, e in enumerate(accepts)],
    }, ensure_ascii=False, indent=1))

    print(f"  扫描 {n_scanned} 个文本文件，排除 {len(skipped)} 个（{excludes}）")
    for pid, _, _, desc in PATTERNS:
        n = sum(1 for h in hits if h["pattern"] == pid)
        if n:
            print(f"  {pid} {desc}: {n} 处命中")

    if secrets:
        print(f"HYGIENE=FAIL 疑似凭据命中 {len(secrets)} 处，硬停（不回显原文）")
        for h in secrets[:10]:
            print(f"    {h['path']}:{h['line']}")
        return 2
    if count_mismatch:
        print("HYGIENE=FAIL allowlist 命中数与实测不符（源库可能已变，需重新裁决）")
        for m in count_mismatch:
            print(f"    {m['pattern']} {m['path_glob']}: 期望 {m['expected']} 实得 {m['got']}")
        return 1
    if ungated:
        print(f"HYGIENE=FAIL 有 {len(ungated)} 处命中未在 allowlist 中裁决，停止并交用户")
        for h in ungated[:20]:
            print(f"    [{h['pattern']}] {h['path']}:{h['line']}  {h['excerpt']}")
        return 1

    print(f"HYGIENE=PASS scanned={n_scanned} hits={len(hits)} ungated=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
