"""逐项比较完整 fixture；不接受 canonical 相同但 raw 不同。"""

import argparse
import json
import pathlib

import _common as C


def compare(left, right):
    for root in (left, right):
        manifest = json.loads((root / "DUMP_MANIFEST.json").read_text())
        for rel, entry in manifest["entries"].items():
            if C.sha256_file(root / rel) != entry["sha256"]:
                raise ValueError(f"取证文件已改变: {root / rel}")
    plans = [json.loads((r / "fixture_plan.json").read_text()) for r in (left, right)]
    if plans[0] != plans[1] or plans[0]["limit"] != 0 or len(plans[0]["batches"]) != 200:
        raise ValueError("fixture 计划不同或不是完整取证")
    plan = plans[0]
    identities = [json.loads((r / "identity.json").read_text()) for r in (left, right)]
    if identities[0] != identities[1]:
        raise ValueError("全清单身份或采样索引不同")
    counts = {}
    for layer in ("samples", "batches"):
        a, b = [[json.loads(line) for line in (root / layer / "summary.jsonl").read_text().splitlines()]
                for root in (left, right)]
        expected = sum(map(len, plan["groups"].values())) if layer == "samples" else 200
        if len(a) != expected or len(b) != expected:
            raise ValueError(f"{layer} 记录不完整: {len(a)}, {len(b)}, expected={expected}")
        for i, (ra, rb) in enumerate(zip(a, b, strict=True)):
            if set(ra) != set(rb):
                raise ValueError(f"{layer}[{i}] 结构不同")
            for k in ra:
                if k in ("keys", "transformed_keys"):
                    if set(ra[k]) != set(rb[k]):
                        raise ValueError(f"{layer}[{i}]/{k} 键集不同（含 None）")
                    for key in ra[k]:
                        da, db = ra[k][key], rb[k][key]
                        if da != db:
                            raise ValueError(f"{layer}[{i}]/{k}/{key} 类型、形状或字节不同: {da} != {db}")
                elif ra[k] != rb[k]:
                    raise ValueError(f"{layer}[{i}]/{k} 身份不同")
        counts[layer] = expected
    ident = identities[0]
    print(f"SOURCE_IDENTITY=PASS episodes={ident['episodes']} samples={ident['samples']}")
    print(f"FRAME_INDEX_EXACT=PASS steps={ident['steps']} max_frames={ident['max_frames']}")
    print(f"SAMPLE_RAW_EXACT=PASS samples={counts['samples']} per_step={plan['per_step']} mismatches=0")
    print("BATCH_RAW_EXACT=PASS batches=200 mismatches=0")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("left", type=pathlib.Path)
    ap.add_argument("right", type=pathlib.Path)
    args = ap.parse_args()
    compare(args.left, args.right)
