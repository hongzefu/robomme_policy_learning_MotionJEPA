"""同一真实 SigLIP 编码器下，REF 与候选的 4×4 在线记忆逐位回归。"""

import argparse
import importlib.util
import json
import pathlib
import pickle
import subprocess

import numpy as np

import _common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref-commit", required=True)
    ap.add_argument("--frames-from", type=pathlib.Path, required=True)
    ap.add_argument("--n-frames", type=int, default=256)
    ap.add_argument("--out", type=pathlib.Path, default=C.REPO_ROOT / "v1-store/reports/t8-infer-regress-4x4")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "framesamp_memory_ref.py"
    src = subprocess.check_output(["git", "show", f"{args.ref_commit}:src/mme_vla_suite/policies/framesamp_memory.py"], cwd=C.REPO_ROOT)
    if path.exists() and path.read_bytes() != src:
        raise ValueError("参考源码载体已存在且内容不符")
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("framesamp_memory_ref", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    import jax
    from mme_vla_suite.policies import framesamp_memory as candidate
    from mme_vla_suite.dataset_builder.siglip_tokenizer import SigLipTokenizer
    from mme_vla_suite.models.config.utils import get_history_config
    if pathlib.Path(candidate.__file__).resolve() == path.resolve():
        raise ValueError("参考与候选导入同一源码")
    origin = importlib.util.find_spec("mme_vla_suite").origin
    if not pathlib.Path(origin).resolve().is_relative_to(C.REPO_ROOT):
        raise ValueError("候选导入不在主树")
    if jax.default_backend() != "gpu":
        raise ValueError("位置表回归必须使用 GPU")
    manifest = json.loads((args.frames_from.parent / "meta/episode_manifest.json").read_text())
    ep = next(ep for ep in manifest["episodes"] if ep["exec_start_idx"] == 0 and ep["num_timesteps"] >= args.n_frames)
    cfg = dict(get_history_config("perceptual-framesamp-context-motion.yaml").motion)
    # motion 使用现有离线真值 token，验证两侧的窗口调度和装配；真 sidecar 另跑 772 窗回归。
    motion_root = args.frames_from.parent / "motion"
    entry = json.loads((motion_root / "meta/motion_index.json").read_text())["entries"][ep["global_episode_idx"]]
    motion = np.fromfile(motion_root / "motion_token.f32.bin", dtype=np.float32).reshape(-1, 768)
    def motion_fn(frames, start):
        if frames.shape != (33, 256, 256, 3) or start % 16:
            raise ValueError("在线窗口形制或起点不符")
        return motion[entry["exec"]["row_base"] + start // 16]
    enc = jax.jit(SigLipTokenizer().__call__)
    kwargs = dict(vision_enc_fn=enc, motion_enc_fn=motion_fn, motion_cfg=cfg)
    a = reference.FrameSampMemory(**kwargs)
    b = candidate.FrameSampMemory(token_per_image=16, **kwargs)
    rows = []
    for step in range(args.n_frames):
        idx = ep["exec_sample_offset"] + step
        with (args.frames_from / "data" / f"{idx}.pkl").open("rb") as f:
            data = pickle.load(f)
        image, state = data["image"][None, None], data["state"][None]
        for mem in (a, b):
            mem.add_buffer(image, state, [step], exec_start_idx=0)
        outputs = [mem.prepare_frame_sampling(step, 512, 16, mem.default_history_feats_gather_fn) for mem in (a, b)]
        frames_sha = [[C.leaf_sha256(np.asarray(x)) for x in out] for out in outputs]
        motion_sha = [[C.leaf_sha256(np.asarray(x)) for x in mem._prepare_motion(step)] for mem in (a, b)]
        if frames_sha[0] != frames_sha[1] or motion_sha[0] != motion_sha[1]:
            raise ValueError(f"ONLINE_MEM_REGRESS=FAIL step={step}")
        rows.append({"step": step, "frames": frames_sha[0], "motion": motion_sha[0]})
    (args.out / "online_mem_regress.json").write_text(json.dumps({"ref_commit": args.ref_commit,
        "origins": [str(path), candidate.__file__], "rows": rows}, indent=2))
    print(f"ONLINE_MEM_REGRESS=PASS frames={args.n_frames} keys=4 mismatches=0")


if __name__ == "__main__":
    main()
