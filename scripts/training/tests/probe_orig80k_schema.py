"""使用既有40集库做三样本CPU取证；不建库、不训练、不替代正式INPUT_EQ。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import check_orig80k_inputs as inputs

ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "v1-store/datasets/4task-motion-40ep"
MANIFEST_SHA = "d7cfb137b6ba01c42894e2d6d421a8c3f87dc1afeef1ac650563609bd7501d05"
INDICES = (0, 31, 32)


def run(args):
    out = Path(args.out)
    inputs.require(out.is_absolute() and out == out.resolve()
                   and out.is_relative_to(ROOT / "v1-store") and not out.exists(), "输出必须为本轮全新实体路径")
    root = Path(args.expect_root).resolve()
    before = inputs.snapshot(root, args.expect_head, args.forbid_root)
    manifest_path = LIB / "meta/episode_manifest.json"
    manifest = inputs.read_json(manifest_path)
    inputs.require(manifest.get("sha256") == MANIFEST_SHA, "既有40集库manifest身份改变")
    inputs.validate_manifest(manifest)
    meta_path = LIB / "framesamp/meta/store_meta.json"
    meta = inputs.read_json(meta_path)
    inputs.require((meta.get("status"), meta.get("layout"), meta.get("manifest_sha256"))
                   == ("verified", "framesamp-4x4-v1", MANIFEST_SHA), "既有packed库未验证或不同源")
    episode = manifest["episodes"][0]
    inputs.require(episode["h5_file"] == "record_dataset_ButtonUnmask.h5"
                   and episode["raw_ep_idx"] == episode["global_episode_idx"] == episode["exec_start_idx"] == 0,
                   "三样本物理身份与已核查记录不同")
    side_args = argparse.Namespace(**vars(args), dataset_path=LIB / ("source" if args.side == "upstream" else "framesamp"),
        source=LIB / "source", manifest=manifest_path,
        assets_dir=ROOT / "v1-store/train-assets/mme_vla_suite")
    factory, _, _ = inputs.load_side(side_args)
    loader = factory()
    transformed = loader._data_loader.torch_loader.dataset  # noqa: SLF001 -- 读取真实侧工厂的取证点
    raw = transformed._dataset  # noqa: SLF001 -- 同一样本变换前后，不替换Dataset实现
    observations = []
    for index in INDICES:
        raw_value = raw[index]
        inputs.require(int(raw_value["epis_idx"].item()) == 0 and int(raw_value["step_idx"].item()) == index,
                       "真实样本身份不符")
        transformed_value = transformed[index]
        observations.append({"index": index, "task": "ButtonUnmask", "raw_episode": 0, "step": index,
                             "raw": inputs.comparison_tree(raw_value), "raw_all": inputs.describe_tree(raw_value),
                             "transformed": inputs.comparison_tree(transformed_value),
                             "transformed_all": inputs.describe_tree(transformed_value)})
    if hasattr(raw, "close"):
        raw.close()
    after = inputs.snapshot(root, args.expect_head, args.forbid_root)
    record = {"schema": inputs.SCHEMA, "kind": "three_sample_tool_diagnostic", "side": args.side,
              "head": args.expect_head, "library": str(LIB), "indices": list(INDICES),
              "scope": "既有40集库三样本raw/transforms字段取证；非正式两库INPUT_EQ，无训练",
              "manifest": inputs.file_record(manifest_path), "packed_meta": inputs.file_record(meta_path),
              "norm_stats": inputs.file_record(side_args.assets_dir / "robomme/norm_stats.json"),
              "provenance_start": before, "provenance_end": after, "observations": observations,
              "finished_unix": time.time()}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"SCHEMA_PROBE_COLLECT=PASS side={args.side} samples=3 out={out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("upstream", "current"), required=True)
    for name in ("expect-root", "expect-head", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--forbid-root", action="append", default=[])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
