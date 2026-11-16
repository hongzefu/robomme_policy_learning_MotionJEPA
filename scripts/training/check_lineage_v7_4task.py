#!/usr/bin/env python3
"""4task 旧 v7 encoder（wan-full1600-filter2-b176x4-72ep-a）与 VLA 原库的四环血缘闸门（1115 计划 B2、§5 第 1 道、R1）。

v16 的 ``check_data_lineage.py member-pin`` 要求 encoder 数据根有 ``control/build_spec.json`` 与
``control/source_manifest.json``，9 月的旧数据根 ``dataset-local-a100-4task-full1600`` 没有这两份，
只有 ``control/source_pin.json`` 与 ``control/scope.json``，所以本脚本按同样的四环结构写一个适配版：

- 环 0：encoder ckpt sha256 复算 == 期望值；由 ckpt 同目录 ``config.yaml`` 的 ``data.data_root`` 定位 encoder 数据根。
- 环 1：VLA ``meta/input_manifest.json`` 的逐文件 sha256 与 ``MERGE_DONE.json`` 的 ``files`` 恰好相等（4 个合并 H5）。
- 环 2：4 份 ``record_dataset_<任务>_episode_map.json`` 的文件 sha256 == ``MERGE_DONE.maps_sha256``，
  任务集合与行数等于合并记录，每行 task 字段与文件名一致。
- 环 3：``MERGE_DONE.source_pin.{revision, manifest_sha256}`` == encoder 侧 ``control/source_pin.json`` 同名键，
  revision 还须 == ``control/scope.json.revision``；``source_pin.status == PASS``、``episodes == 1600``。
- 环 4：episode_map 的 1600 个成员与 encoder 侧 ``scope.json.entries``（2800 条 demo/exec 变体）按
  (task, difficulty, episode, seed) 去重后的 1600 集逐条相等；比较字段为 MEMBER_FIELDS 全部字段。
  同一成员在 entries 里的多条变体字段不一致、episode_map 内部 h5_sha256 与 h5_sha256_local 不一致、
  任一边多或少，均 FAIL。

输入路径一律用绝对路径参数（R6：不按脚本位置推仓库根）。末行判定行：
``DATA_LINEAGE=PASS mode=v7-4ring encoder_ckpt=<sha8> encoder_source_pin=<manifest_sha8> vla_input_pin=<sha8> files=4 members=1600 mismatches=0``
FAIL 时 ``DATA_LINEAGE=FAIL mode=v7-4ring ring=<n> error=<原因>``，退出码 3。``--out`` 写 JSON 证据。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

MAP_NAME = "record_dataset_{task}_episode_map.json"
MEMBER_KEY = ("task", "difficulty", "episode", "seed")
MEMBER_FIELDS = ("task", "difficulty", "episode", "seed", "member", "h5_sha256", "h5_bytes", "timestep_count", "role")
EXPECTED_EPISODES = 1600


class LineageRingError(Exception):
    def __init__(self, ring, message):
        super().__init__(message)
        self.ring = ring


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path, ring):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LineageRingError(ring, f"读取 JSON 失败 {path}: {exc}") from exc


def require_sha256(value, label, ring):
    if not (isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)):
        raise LineageRingError(ring, f"{label} 不是 64 位小写十六进制 sha256: {value!r}")
    return value


def _require_abs(path, label):
    p = Path(path)
    if not p.is_absolute():
        raise LineageRingError("args", f"{label} 必须是绝对路径: {path}")
    return p


def member_tuple(row, label, ring):
    missing = [k for k in MEMBER_FIELDS if k not in row]
    if missing:
        raise LineageRingError(ring, f"{label} 成员缺字段 {missing}: {dict((k, row.get(k)) for k in MEMBER_KEY)}")
    return tuple(row[k] for k in MEMBER_FIELDS)


def check_v7_lineage(encoder_ckpt, expected_ckpt_sha256, vla_input_manifest, merge_done, episode_maps, evidence=None):
    """四环逐项比对，任一环不等抛 LineageRingError；evidence 随过程填入已比对的各环。"""
    evidence = {} if evidence is None else evidence
    rings = evidence.setdefault("rings", {})

    # ── 环 0：encoder ckpt 身份 + 数据根定位 ──
    ckpt = _require_abs(encoder_ckpt, "--encoder-ckpt")
    expected = require_sha256(expected_ckpt_sha256, "--expected-ckpt-sha256", 0)
    if not ckpt.is_file():
        raise LineageRingError(0, f"encoder ckpt 不存在: {ckpt}")
    ckpt_sha = sha256_file(ckpt)
    evidence["encoder"] = {"checkpoint": str(ckpt), "checkpoint_sha256": ckpt_sha}
    if ckpt_sha != expected:
        raise LineageRingError(0, f"encoder ckpt sha256 不符 expected={expected} actual={ckpt_sha}")
    config_path = ckpt.parent / "config.yaml"
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        root = cfg["data"]["data_root"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise LineageRingError(0, f"读不到 {config_path}::data.data_root: {exc}") from exc
    if not isinstance(root, str) or not root:
        raise LineageRingError(0, "config.yaml::data.data_root 必须为非空路径")
    root = Path(root) if Path(root).is_absolute() else (config_path.parent / root)
    pin_path, scope_path = root / "control/source_pin.json", root / "control/scope.json"
    evidence["encoder"].update(config=str(config_path), config_sha256=sha256_file(config_path), data_root=str(root))
    rings["0_encoder_ckpt"] = {"status": "PASS", "checkpoint_sha256": ckpt_sha, "data_root": str(root)}

    vla_path = _require_abs(vla_input_manifest, "--vla-input-manifest")
    merge_path = _require_abs(merge_done, "--merge-done")
    maps_dir = _require_abs(episode_maps, "--episode-maps")
    vla = read_json(vla_path, 1)
    merge = read_json(merge_path, 1)
    evidence["vla_input_manifest"] = {"path": str(vla_path), "sha256": sha256_file(vla_path)}
    evidence["merge_done"] = {"path": str(merge_path), "sha256": sha256_file(merge_path)}

    # ── 环 1：合并 H5 ──
    vla_files = vla.get("files")
    merged = merge.get("files")
    if not isinstance(vla_files, dict) or not vla_files:
        raise LineageRingError(1, "input_manifest.files 必须为非空字典")
    if not isinstance(merged, dict) or not merged:
        raise LineageRingError(1, "MERGE_DONE.files 必须为非空字典")
    vla_sha = {k: (v.get("sha256") if isinstance(v, dict) else None) for k, v in vla_files.items()}
    for k, v in vla_sha.items():
        require_sha256(v, f"input_manifest.files[{k}].sha256", 1)
    bad = sorted(n for n in set(vla_sha) | set(merged) if vla_sha.get(n) is None or vla_sha.get(n) != merged.get(n))
    rings["1_merged_h5"] = {"status": "FAIL" if bad else "PASS", "checked": len(merged), "mismatches": bad}
    if bad:
        raise LineageRingError(1, f"合并 H5 与 MERGE_DONE.files 不等 files={','.join(bad)}")

    # ── 环 2：episode_map 文件 sha、任务集合、行数 ──
    maps_sha = merge.get("maps_sha256")
    tasks = (merge.get("selection") or {}).get("tasks")
    counts = merge.get("tasks")
    if (not isinstance(maps_sha, dict) or not isinstance(tasks, list) or not isinstance(counts, dict)
            or len(tasks) != len(set(tasks)) or set(maps_sha) != set(tasks) or set(counts) != set(tasks)):
        raise LineageRingError(2, "MERGE_DONE 的 maps_sha256、selection.tasks、tasks 任务集合不一致")
    map_rows, map_ev, bad = [], {}, []
    for task in sorted(tasks):
        path = maps_dir / MAP_NAME.format(task=task)
        if not path.is_file():
            bad.append(f"{task}:缺文件")
            continue
        digest = sha256_file(path)
        rows = read_json(path, 2)
        ok = (digest == maps_sha[task] and isinstance(rows, list)
              and len(rows) == (counts[task] or {}).get("episodes")
              and all(isinstance(r, dict) and r.get("task") == task for r in rows))
        map_ev[task] = {"path": str(path), "sha256": digest, "rows": len(rows) if isinstance(rows, list) else None}
        if not ok:
            bad.append(task)
            continue
        map_rows += rows
    episodes = merge.get("episodes")
    if not bad and (episodes != EXPECTED_EPISODES or len(map_rows) != episodes or merge.get("mismatches") != 0
                    or merge.get("source_sha256_checked") != episodes):
        bad.append(f"episodes={episodes} rows={len(map_rows)} mismatches={merge.get('mismatches')} "
                   f"checked={merge.get('source_sha256_checked')}")
    evidence["episode_maps"] = map_ev
    rings["2_episode_maps"] = {"status": "FAIL" if bad else "PASS", "checked": len(tasks), "mismatches": bad}
    if bad:
        raise LineageRingError(2, f"episode_map 与合并记录不等 {';'.join(bad)}")

    # ── 环 3：源 pin ──
    pin = read_json(pin_path, 3)
    scope = read_json(scope_path, 3)
    evidence["encoder"].update(source_pin=str(pin_path), source_pin_sha256=sha256_file(pin_path),
                               scope=str(scope_path), scope_sha256=sha256_file(scope_path))
    mp = merge.get("source_pin") or {}
    require_sha256(mp.get("manifest_sha256"), "MERGE_DONE.source_pin.manifest_sha256", 3)
    checks = {
        "manifest_sha256": (mp.get("manifest_sha256"), pin.get("manifest_sha256")),
        "revision": (mp.get("revision"), pin.get("revision"), scope.get("revision")),
        "source_pin.status": (pin.get("status"), "PASS"),
        "source_pin.episodes": (pin.get("episodes"), EXPECTED_EPISODES),
    }
    bad = sorted(k for k, vals in checks.items() if vals[0] in (None, "") or len(set(vals)) != 1)
    rings["3_source_pin"] = {"status": "FAIL" if bad else "PASS", "manifest_sha256": mp.get("manifest_sha256"),
                             "revision": mp.get("revision"), "mismatches": bad,
                             "values": {k: list(v) for k, v in checks.items()}}
    if bad:
        raise LineageRingError(3, f"源 pin 不等 keys={','.join(bad)}")

    # ── 环 4：成员逐条相等 ──
    vla_members = {}
    for r in map_rows:
        t = member_tuple(r, "episode_map", 4)
        if "h5_sha256_local" in r and r["h5_sha256_local"] != r["h5_sha256"]:
            raise LineageRingError(4, f"episode_map h5_sha256_local != h5_sha256: {t[:4]}")
        if t[:4] in vla_members:
            raise LineageRingError(4, f"episode_map 成员重复: {t[:4]}")
        vla_members[t[:4]] = t
    entries = scope.get("entries")
    if not isinstance(entries, list) or not entries:
        raise LineageRingError(4, "scope.json.entries 必须为非空列表")
    enc_members = {}
    for e in entries:
        t = member_tuple(e, "scope.entries", 4)
        prev = enc_members.setdefault(t[:4], t)
        if prev != t:
            raise LineageRingError(4, f"scope.entries 同一成员的变体字段不一致: {t[:4]}")
    bad = sorted((k for k in set(vla_members) | set(enc_members) if vla_members.get(k) != enc_members.get(k)),
                 key=lambda k: tuple(str(x) for x in k))
    ok = not bad and len(vla_members) == EXPECTED_EPISODES and len(enc_members) == EXPECTED_EPISODES \
        and scope.get("episodes") == EXPECTED_EPISODES
    rings["4_members"] = {"status": "PASS" if ok else "FAIL", "vla_members": len(vla_members),
                          "encoder_entries": len(entries), "encoder_members": len(enc_members),
                          "mismatch_count": len(bad), "mismatches": [list(k) for k in bad[:20]],
                          "match_key": list(MEMBER_KEY), "compared_fields": list(MEMBER_FIELDS)}
    if bad:
        k = bad[0]
        raise LineageRingError(4, f"成员不等 mismatches={len(bad)} first={list(k)} "
                                  f"vla={vla_members.get(k)} encoder={enc_members.get(k)}")
    if not ok:
        raise LineageRingError(4, f"成员数不符 vla={len(vla_members)} encoder={len(enc_members)} "
                                  f"scope.episodes={scope.get('episodes')}")
    return {"status": "PASS", "encoder_ckpt": ckpt_sha, "encoder_source_pin": mp["manifest_sha256"],
            "vla_input_pin": evidence["vla_input_manifest"]["sha256"], "files": len(merged),
            "members": len(vla_members), "mismatches": 0}


def pass_line(r):
    return (f"DATA_LINEAGE=PASS mode=v7-4ring encoder_ckpt={r['encoder_ckpt'][:8]} "
            f"encoder_source_pin={r['encoder_source_pin'][:8]} vla_input_pin={r['vla_input_pin'][:8]} "
            f"files={r['files']} members={r['members']} mismatches={r['mismatches']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encoder-ckpt", required=True, help="encoder checkpoint 绝对路径（同目录须有 config.yaml）")
    ap.add_argument("--expected-ckpt-sha256", required=True)
    ap.add_argument("--vla-input-manifest", required=True, help="VLA 库 meta/input_manifest.json 绝对路径")
    ap.add_argument("--merge-done", required=True, help="records/MERGE_DONE.json 绝对路径")
    ap.add_argument("--episode-maps", required=True, help="四份 episode_map 所在目录绝对路径")
    ap.add_argument("--out", help="可选：JSON 证据输出路径")
    a = ap.parse_args(argv)
    evidence = {"mode": "v7-4ring"}
    try:
        r = check_v7_lineage(a.encoder_ckpt, a.expected_ckpt_sha256, a.vla_input_manifest,
                             a.merge_done, a.episode_maps, evidence)
        line, code = pass_line(r), 0
        evidence["result"] = r
    except LineageRingError as exc:
        line, code = f"DATA_LINEAGE=FAIL mode=v7-4ring ring={exc.ring} error={exc}", 3
        evidence["result"] = {"status": "FAIL", "ring": exc.ring, "error": str(exc)}
    evidence["verdict"] = line
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(line, flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
