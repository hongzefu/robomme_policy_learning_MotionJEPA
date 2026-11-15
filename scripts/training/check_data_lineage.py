#!/usr/bin/env python3
"""强制核验 encoder、MJ 缓存和 VLA 的源 H5 集合完全同源。

子命令 lineage：16task 三源逐文件比对；member-pin：4task 四环链（合并 H5、episode_map、源清单、1600 成员与 member pin）。
"""
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


# ── member-pin：4task 四环链（1114 计划 §2、B 契约③） ─────────────────────────
MEMBER_PIN_SCHEMA = "data-lineage-member-pin-v1"
MAP_NAME = "record_dataset_{task}_episode_map.json"


class LineageRingError(ValueError):
    """某一环比对不等；ring 取 1–4，0 表示输入身份（checkpoint、文件读取）本身不合格。"""

    def __init__(self, ring, message):
        super().__init__(message)
        self.ring = ring


def canonical_member_records(rows, *, sha_key="h5_sha256_local", out_key="h5_sha256"):
    """map 行 → 契约③规范记录 {member, h5_sha256, h5_bytes}；sha 取 map 的 h5_sha256_local。"""
    records = []
    for row in rows:
        member, size = row["member"], row["h5_bytes"]
        if not isinstance(member, str) or not member or type(size) is not int or size <= 0:
            raise ValueError(f"成员记录非法: {row}")
        records.append({"member": member, out_key: require_sha256(row[sha_key], f"member {member}"), "h5_bytes": size})
    return records


def member_pin_sha256(records):
    """按 member 排序，json.dumps(sort_keys, 紧凑分隔, ensure_ascii=False) 的 UTF-8 字节取 sha256。"""
    ordered = sorted(records, key=lambda record: record["member"])
    members = [record["member"] for record in ordered]
    if len(set(members)) != len(members):
        raise ValueError("member 重复")
    payload = json.dumps(ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json_any(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)


def _members_by_name(records, label):
    result = {}
    for record in records:
        if record["member"] in result:
            raise LineageRingError(4, f"{label} member 重复: {record['member']}")
        result[record["member"]] = (record["h5_sha256"], record["h5_bytes"])
    return result


def check_member_pin(encoder_ckpt, expected_ckpt_sha256, vla_input_manifest, merge_done, episode_maps,
                     evidence=None):
    """四环链逐项比对，任一环不等抛 LineageRingError；evidence 字典随过程填入已比对的各环。"""
    evidence = {} if evidence is None else evidence
    rings = evidence.setdefault("rings", {})
    # 输入身份：checkpoint 独立复算 sha，encoder 侧清单由 ckpt 同目录 config.yaml 的 data_root 定位。
    checkpoint = Path(encoder_ckpt).resolve()
    expected = require_sha256(expected_ckpt_sha256, "expected_ckpt_sha256")
    checkpoint_sha = sha256_file(checkpoint)
    evidence["encoder"] = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha}
    if checkpoint_sha != expected:
        raise LineageRingError(0, f"encoder checkpoint SHA256 不符: expected={expected} actual={checkpoint_sha}")
    config_path = checkpoint.parent / "config.yaml"
    root = yaml.safe_load(config_path.read_text(encoding="utf-8"))["data"]["data_root"]
    if not isinstance(root, str) or not root:
        raise LineageRingError(0, "config.yaml::data.data_root 必须为非空路径")
    root = Path(root) if Path(root).is_absolute() else config_path.parent / root
    build_spec_path, source_path = root / "control/build_spec.json", root / "control/source_manifest.json"
    pin = read_json(build_spec_path)["identity"]["source_pin"]
    source = read_json(source_path)
    evidence["encoder"].update(config_sha256=sha256_file(config_path), data_root=str(root),
                               build_spec=str(build_spec_path), build_spec_sha256=sha256_file(build_spec_path),
                               source_manifest=str(source_path), source_manifest_sha256=sha256_file(source_path),
                               member_pin_sha256=pin.get("member_pin_sha256"))
    vla = read_json(vla_input_manifest)
    merge = read_json(merge_done)
    evidence["vla_input_manifest"] = {"path": str(Path(vla_input_manifest).resolve()),
                                      "sha256": sha256_file(vla_input_manifest)}
    evidence["merge_done"] = {"path": str(Path(merge_done).resolve()), "sha256": sha256_file(merge_done)}

    # ① 合并 H5：VLA input_manifest 的逐文件 sha 与 MERGE_DONE.files 恰好相等。
    vla_files = input_files(vla)
    merged = merge.get("files")
    if not isinstance(merged, dict) or not merged:
        raise LineageRingError(1, "MERGE_DONE.files 必须为非空字典")
    bad = sorted(name for name in set(vla_files) | set(merged)
                 if name not in vla_files or vla_files[name]["sha256"] != merged.get(name))
    rings["1_merged_h5"] = {"status": "FAIL" if bad else "PASS", "checked": len(merged), "mismatches": bad}
    if bad:
        raise LineageRingError(1, f"合并 H5 与合并记录不等 files={','.join(bad)}")

    # ② episode_map：每份文件 sha 等于 MERGE_DONE.maps_sha256，行数与任务等于合并记录。
    maps_sha = merge.get("maps_sha256")
    tasks = merge.get("selection", {}).get("tasks")
    counts = merge.get("tasks")
    if (not isinstance(maps_sha, dict) or not isinstance(tasks, list) or not isinstance(counts, dict)
            or set(maps_sha) != set(tasks) or set(counts) != set(tasks) or len(tasks) != len(set(tasks))):
        raise LineageRingError(2, "MERGE_DONE 的 maps_sha256、selection.tasks、tasks 任务集合不一致")
    map_records, map_evidence, bad = [], {}, []
    for task in sorted(tasks):
        path = Path(episode_maps) / MAP_NAME.format(task=task)
        digest = sha256_file(path)
        rows = _read_json_any(path)
        ok = (digest == maps_sha[task] and isinstance(rows, list) and len(rows) == counts[task].get("episodes")
              and all(isinstance(row, dict) and row.get("task") == task for row in rows))
        map_evidence[task] = {"path": str(path.resolve()), "sha256": digest,
                              "rows": len(rows) if isinstance(rows, list) else None}
        if not ok:
            bad.append(task)
            continue
        map_records += canonical_member_records(rows)
    episodes = merge.get("episodes")
    if not bad and (len(map_records) != episodes or merge.get("mismatches") != 0
                    or merge.get("source_sha256_checked") != episodes):
        bad.append("episodes")
    evidence["episode_maps"] = map_evidence
    rings["2_episode_maps"] = {"status": "FAIL" if bad else "PASS", "checked": len(tasks), "mismatches": bad}
    if bad:
        raise LineageRingError(2, f"episode_map 与合并记录不等 tasks={','.join(bad)}")

    # ③ 源清单：合并记录的 bucket 清单 sha、repo、revision 与 encoder 侧相等。
    merge_pin = merge.get("source_pin", {})
    pairs = {"manifest_sha256": (merge_pin.get("manifest_sha256"), pin.get("manifest_sha256"),
                                 source.get("manifest_sha256")),
             "repo_id": (merge_pin.get("repo_id"), source.get("repo_id")),
             "revision": (merge_pin.get("revision"), source.get("revision")),
             "scope_sha256": (pin.get("scope_sha256"), source.get("scope_sha256"))}
    bad = sorted(key for key, values in pairs.items() if values[0] in (None, "") or len(set(values)) != 1)
    require_sha256(merge_pin.get("manifest_sha256"), "MERGE_DONE.source_pin.manifest_sha256")
    rings["3_source_manifest"] = {"status": "FAIL" if bad else "PASS",
                                  "manifest_sha256": merge_pin.get("manifest_sha256"), "mismatches": bad}
    if bad:
        raise LineageRingError(3, f"合并记录源清单与 encoder 侧不等 keys={','.join(bad)}")

    # ④ 成员：1600 条逐条相等，并按契约③复算 member pin 与 encoder 登记值相等。
    encoder_records = canonical_member_records(source.get("files") or [], sha_key="h5_sha256")
    vla_members = _members_by_name(map_records, "episode_map")
    encoder_members = _members_by_name(encoder_records, "encoder")
    bad = sorted(name for name in set(vla_members) | set(encoder_members)
                 if vla_members.get(name) != encoder_members.get(name))
    member_pin = member_pin_sha256(map_records)
    literal = _literal_key_pin(map_records)
    registered = (pin.get("member_pin_sha256"), source.get("member_pin_sha256"))
    pin_ok = (len(set(registered)) == 1 and member_pin == registered[0]
              and member_pin_sha256(encoder_records) == member_pin and literal != member_pin)
    rings["4_members"] = {"status": "PASS" if not bad and pin_ok else "FAIL", "members": len(vla_members),
                          "mismatches": bad[:20], "mismatch_count": len(bad), "member_pin_sha256": member_pin,
                          "member_pin_literal_key_sha256": literal, "encoder_member_pin_sha256": registered[0]}
    if bad:
        raise LineageRingError(4, f"成员与 encoder 侧不等 mismatches={len(bad)} first={bad[0]}")
    if not pin_ok:
        raise LineageRingError(4, f"member pin 不等 recomputed={member_pin} registered={','.join(map(str, registered))}")
    return {"status": "PASS", "encoder_ckpt": checkpoint_sha, "member_pin": member_pin,
            "member_pin_literal_key": literal, "members": len(vla_members), "h5_files": len(merged),
            "maps": len(tasks), "vla_input_pin": evidence["vla_input_manifest"]["sha256"], "mismatches": 0}


def _literal_key_pin(records):
    """反例口径：保留 map 的字面键 h5_sha256_local；只用于记录并断言与规范 pin 不等。"""
    return member_pin_sha256([{"member": r["member"], "h5_sha256_local": r["h5_sha256"], "h5_bytes": r["h5_bytes"]}
                              for r in records])


def member_pin_line(result):
    return (f"DATA_LINEAGE=PASS mode=member-pin encoder_ckpt={result['encoder_ckpt'][:8]} "
            f"member_pin={result['member_pin']} members={result['members']} h5_files={result['h5_files']} "
            f"maps={result['maps']} vla_input_pin={result['vla_input_pin']} mismatches=0")


def write_evidence(path, evidence):
    """原子写 JSON 证据：先写同目录临时文件再 rename。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_member_pin(args):
    evidence = {"schema": MEMBER_PIN_SCHEMA, "mode": "member-pin", "status": "FAIL"}
    try:
        result = check_member_pin(args.encoder_ckpt, args.expected_ckpt_sha256, args.vla_input_manifest,
                                  args.merge_done, args.episode_maps, evidence)
        line, code = member_pin_line(result), 0
        evidence.update(status="PASS", member_pin_sha256=result["member_pin"],
                        member_pin_literal_key_sha256=result["member_pin_literal_key"], members=result["members"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError) as exc:
        ring = getattr(exc, "ring", None)
        line, code = f"DATA_LINEAGE=FAIL mode=member-pin ring={'?' if ring is None else ring} error={exc}", 3
        evidence.update(failed_ring=ring, error=str(exc))
    evidence["line"] = line
    if args.out:
        write_evidence(args.out, evidence)
    print(line, file=sys.stdout if code == 0 else sys.stderr)
    return code


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(3, f"DATA_LINEAGE=FAIL error={message}\n")


def main(argv=None):
    parser = _Parser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    lineage = sub.add_parser("lineage")
    for name in ("encoder-ckpt", "expected-ckpt-sha256", "vla-input-manifest", "mj-source-manifest"):
        lineage.add_argument(f"--{name}", required=True)
    member_pin = sub.add_parser("member-pin")
    for name in ("encoder-ckpt", "expected-ckpt-sha256", "vla-input-manifest", "merge-done", "episode-maps"):
        member_pin.add_argument(f"--{name}", required=True)
    member_pin.add_argument("--out", help="写 JSON 证据（PASS 与 FAIL 都写），内容与判定行一致")
    args = parser.parse_args(argv)
    if args.command == "member-pin":
        return run_member_pin(args)
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
