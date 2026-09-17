"""两张历史 motion 表与冻结 checkpoint 快照的生产 policy 构造回归。"""

import argparse
import hashlib
import json
import pathlib

from mme_vla_suite.datastore import motion_store as ms


def check_layouts():
    for name, rows in (("4task-motion-40ep",772),("4task-motion-400ep",6832)):
        meta=ms.MotionMeta.load(pathlib.Path("v1-store/datasets")/name/"motion")
        if meta.num_rows != rows or meta.spec != ms.LayoutSpec(33,33,"none"):
            raise ValueError(f"历史布局回归失败: {name}")
    print("LEGACY_LAYOUT=PASS tables=2 rows=772/6832 spec=(33,33,none)",flush=True)


def check_policy(checkpoint):
    from mme_vla_suite.training.config import get_config
    from mme_vla_suite.policies.policy_config import create_trained_policy
    root=checkpoint.parent
    paths=[root/n for n in ("history_config.resolved.yaml","history_config.resolved.sha256","motion_provenance.json","history_config.txt")]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    policy=create_trained_policy(get_config("mme_vla_suite_b128"),checkpoint,motion_stub=True)
    try:
        if (policy._motion_cfg["demo_min_real_frames"],policy._motion_cfg["demo_tail_pad"]) != (33,"none"):
            raise ValueError("历史快照未走 33/none 兼容分支")
        if not policy._motion_client.stub or not policy._motion_client.alive:
            raise ValueError("历史 policy 未完成 stub sidecar 握手")
    finally:
        policy._motion_client.close()
    if policy._motion_client._proc.returncode != 0:
        raise ValueError("stub sidecar 未正常退出")
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if before != after:
        raise ValueError("回归期间历史快照发生变化")
    print("LEGACY_POLICY=PASS spec=(33,none) stub=1 snapshot_unchanged=1",flush=True)
    return before


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate",choices=("all","layout","policy"),default="all")
    parser.add_argument("--checkpoint",type=pathlib.Path,default=pathlib.Path("v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/5000"))
    parser.add_argument("--out",type=pathlib.Path)
    args=parser.parse_args()
    record={"gate":args.gate}
    if args.gate in ("all","layout"): check_layouts()
    if args.gate in ("all","policy"): record["snapshot_sha256"]=check_policy(args.checkpoint)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x") as f: json.dump(record,f,ensure_ascii=False,indent=2)


if __name__ == "__main__":
    main()
