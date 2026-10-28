#!/usr/bin/env python3
"""强制核验 encoder、MJ 缓存和 VLA 的源 H5 集合完全同源。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 键重复: {key}")
        result[key] = value
    return result


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"清单顶层必须为对象: {path}")
    return value


def require_sha256(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} 缺少合法 SHA256")
    return value


def normalize_files(records, *, label):
    """规范化为文件名→字节数和摘要；路径、mtime 与整体 pin 不参与同源比较。"""
    result = {}
    for name, size, sha in records:
        if not isinstance(name, str) or not name or "\\" in name:
            raise ValueError(f"{label} 文件名无效")
        name = Path(name).name
        if not name.endswith(".h5") or name in result:
            raise ValueError(f"{label} H5 文件名无效或重复: {name}")
        if type(size) is not int or size <= 0:
            raise ValueError(f"{label} {name} 字节数必须为正整数")
        result[name] = {"bytes": size, "sha256": require_sha256(sha, f"{label} {name}")}
    if not result:
        raise ValueError(f"{label} 源 H5 集合为空")
    return result


def input_files(manifest):
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("VLA files 必须为文件名字典")
    if "count" in manifest and (type(manifest["count"]) is not int or manifest["count"] != len(files)):
        raise ValueError("VLA count 与文件数不符")
    return normalize_files(((name, item["size"], item["sha256"]) for name, item in files.items()), label="VLA")


def check_lineage(encoder_ckpt, expected_ckpt_sha256, vla_input_manifest, mj_source_manifest):
    """独立核算 checkpoint SHA 后，逐文件比对三份来源清单，不加载模型。"""
    checkpoint = Path(encoder_ckpt).resolve()
    expected = require_sha256(expected_ckpt_sha256, "expected_ckpt_sha256")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != expected:
        raise ValueError(f"encoder checkpoint SHA256 不符: expected={expected} actual={checkpoint_sha}")
    config_path = checkpoint.parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = config["data"]["data_root"]
    if not isinstance(root, str) or not root:
        raise ValueError("config.yaml::data.data_root 必须为非空路径")
    root = Path(root)
    if not root.is_absolute():
        root = config_path.parent / root
    build_spec_path = root / "control/build_spec.json"
    source_pin = read_json(build_spec_path)["identity"]["source_pin"]
    source_records = source_pin["files"]
    if not isinstance(source_records, list):
        raise ValueError("encoder source_pin.files 必须为列表")
    encoder = normalize_files(((x["path"], x["bytes"], x["sha256"]) for x in source_records), label="encoder")
    vla = input_files(read_json(vla_input_manifest))
    mj_records = read_json(mj_source_manifest)["files"]
    if not isinstance(mj_records, dict):
        raise ValueError("MJ files 必须为任务字典")
    mj = normalize_files(((x["h5"]["file"], x["h5"]["size"], x["h5"]["sha256"])
                          for x in mj_records.values()), label="MJ")
    names = set(encoder) | set(vla) | set(mj)
    mismatches = [name for name in sorted(names) if not (encoder.get(name) == vla.get(name) == mj.get(name))]
    if mismatches:
        raise ValueError(f"三源 H5 集合不同 mismatches={len(mismatches)} files={','.join(mismatches)}")
    # 两种 pin 的载荷本来不同，只各自记录，绝不拿其相等代替逐文件比对。
    encoder_pin = require_sha256(source_pin.get("sha256"), "encoder_source_pin")
    return {"status": "PASS", "encoder_ckpt": checkpoint_sha, "encoder_source_pin": encoder_pin,
            "vla_input_pin": sha256_file(vla_input_manifest), "mj_source_pin": sha256_file(mj_source_manifest),
            "files": len(names), "mismatches": 0}


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(3, f"DATA_LINEAGE=FAIL error={message}\n")


def main(argv=None):
    parser = _Parser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    lineage = sub.add_parser("lineage")
    for name in ("encoder-ckpt", "expected-ckpt-sha256", "vla-input-manifest", "mj-source-manifest"):
        lineage.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    try:
        result = check_lineage(args.encoder_ckpt, args.expected_ckpt_sha256,
                               args.vla_input_manifest, args.mj_source_manifest)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError) as exc:
        print(f"DATA_LINEAGE=FAIL error={exc}", file=sys.stderr)
        return 3
    print(f"DATA_LINEAGE=PASS encoder_ckpt={result['encoder_ckpt'][:8]} "
          f"encoder_source_pin={result['encoder_source_pin']} vla_input_pin={result['vla_input_pin']} "
          f"files={result['files']} mismatches=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
