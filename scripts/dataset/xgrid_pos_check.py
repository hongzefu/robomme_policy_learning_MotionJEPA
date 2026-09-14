"""不经打包规格表的源 npy 抽样，以及两档位置表的完整时间码对拍。"""

import argparse
import json
import pathlib

import numpy as np

from mme_vla_suite.datastore.framesamp_store import FrameSampStore, StoreMeta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store-4x4", type=pathlib.Path, required=True)
    ap.add_argument("--store-8x8", type=pathlib.Path, required=True)
    ap.add_argument("--source", type=pathlib.Path, required=True)
    ap.add_argument("--image-spot", type=int, default=512)
    args = ap.parse_args()
    m4, m8 = StoreMeta.load(args.store_4x4), StoreMeta.load(args.store_8x8)
    if (m4.raw["layout"], m8.raw["layout"]) != ("framesamp-4x4-v1", "framesamp-8x8-v1"):
        raise ValueError("两库布局与角色不符")
    if m4.manifest_sha256 != m8.manifest_sha256 or m4.num_rows != m8.num_rows:
        raise ValueError("两库不同源")
    manifest = json.loads(pathlib.Path(m8.manifest_path).read_text())
    # 独立写明键、形状与文件名，防止规格表的对称错误造成假通过。
    p4 = np.fromfile(args.store_4x4 / "pos_emb_4x4.f32.bin", dtype=np.float32).reshape(-1, 16, 768)
    p8 = np.fromfile(args.store_8x8 / "pos_emb_8x8.f32.bin", dtype=np.float32).reshape(-1, 64, 768)
    if p4[:, 0, :256].tobytes() != p8[:, 0, :256].tobytes():
        raise ValueError("MOTION_POS_XGRID=FAIL 表时间码不同")
    pairs = [(ep["global_episode_idx"], t, ep["total_sample_offset"] + t)
             for ep in manifest["episodes"] for t in range(ep["num_timesteps"])]
    if not 64 <= args.image_spot <= len(pairs):
        raise ValueError("image 抽样数须在 64..num_rows 内")
    indices = np.random.default_rng(42).choice(len(pairs), args.image_spot, replace=False)
    store = FrameSampStore(args.store_8x8)
    try:
        for n, i in enumerate(indices):
            g, t, row = pairs[int(i)]
            raw = np.load(args.source / "features" / f"episode_{g}" / f"token_emb_{t}.npy", allow_pickle=True).item()
            got = store.read_image_rows(np.asarray([row]))[0]
            want = raw["image_emb_8x8"][0]
            if (got.shape, str(got.dtype), got.tobytes()) != (want.shape, str(want.dtype), want.tobytes()):
                raise ValueError(f"IMAGE_NPY_SPOT=FAIL episode={g} t={t}")
            if p8[t].tobytes() != raw["pos_emb_8x8"][0].tobytes():
                raise ValueError(f"源 pos 与表不同: episode={g} t={t}")
            if n < 64 and raw["pos_emb_4x4"][0, 0, :256].tobytes() != raw["pos_emb_8x8"][0, 0, :256].tobytes():
                raise ValueError(f"MOTION_POS_XGRID=FAIL 源时间码 episode={g} t={t}")
    finally:
        store.close()
    print(f"IMAGE_NPY_SPOT=PASS frames={args.image_spot} mismatches=0")
    print(f"MOTION_POS_XGRID=PASS t={len(p8)} npy=64 mismatches=0")


if __name__ == "__main__":
    main()
