#!/usr/bin/env python3
"""ONLINE_MEM 三方 A/B 对拍量具（闸门 N4，commitV4.4；计划五节 a）。

三方（同一进程、同一个 jitted SigLIP callable 注入三侧，消除编译非确定性）：
  A = 旧 `shared.mem_buffer.MemoryBuffer` —— 从 `git show $COPY_BASE:` 提取到
      scratch 目录的四文件冻结快照（主流程，二节第 12 条：commit 后在 clean HEAD
      上跑，不需要脏工作区共存）；
  B = 建库域冻结副本 `dataset_builder.mem_buffer.MemoryBuffer`（IMPORT_ISOLATION
      c-2 白名单唯一豁免项——这正是「副本与原件行为逐位相同」的唯一 bitwise 证据）；
  C = 新 `policies.framesamp_memory.FrameSampMemory`。

三层判据（leaf_sha256 = sha256(dtype‖shape‖tobytes)，bf16 位型安全；禁 allclose/==）：
  POS_TABLE=PASS            4x4 pos 表全表三方逐位
  ENC_LAYER=PASS steps=13 keys=3 mismatch=0   （image_emb_4x4/pos_emb_4x4/state_emb；
                            A/B 多出的 8x8/2x2/image_pixels 为白名单删除、不比）
  ASSEMBLY=PASS steps=13 mismatch=0           （prepare_frame_sampling 四元组）
另有 OOB_PROBE（4096 越界探针，三方都必须响亮失败——numpy 切片越界静默返回空数组，
禁止依赖）与合成极值帧不炸检查（非判据）。

用法（1 卡 ~10GB）：
  CUDA_VISIBLE_DEVICES=0 OPENPI_DATA_HOME=<REPO_ROOT>/v1-store/models \\
  UV_LINK_MODE=copy uv run scripts/smoke-local/compare_online_memory.py \\
    --h5 /data/hongzefu/robomme_data_h5_v2_4env400ep/record_dataset_ButtonUnmask.h5 \\
    --out v1-store/bench/online-mem/<TAG>
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}")
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "dtype-unify"))
import _common as C  # noqa: E402  leaf_sha256 所在（V4.6 迁移时同步改此插入）

# A 侧提取锚点：钉死为 V4.0 的父提交，不随 HEAD 走（计划〇节常量）
COPY_BASE = "732fae3b13e2ff5f485d7014473b99ed577de387"

# 步位网格（覆盖 even_sampling 全分支）：0/1/2 起步、15/16 中段、
# 30（恰 1 行填充）、31（恰填满零填充）、32（首进 linspace 含重复索引）、
# 33/34、100/291、585
STEP_GRID = (0, 1, 2, 15, 16, 30, 31, 32, 33, 34, 100, 291, 585)
ENC_KEYS = ("image_emb_4x4", "pos_emb_4x4", "state_emb")
MAX_STEPS = 4096


def extract_legacy_shared(scratch: pathlib.Path) -> None:
    """git show 提取 A 侧四文件并改包名为 legacy_shared（四文件必须一起提取，
    只提 mem_buffer 会让 A 侧绑到 B/C 的新实现、退化为自比）。"""
    pkg = scratch / "legacy_shared"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for f in ("data_utils", "posemb_3d", "siglip_tokenizer", "mem_buffer"):
        src = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "show",
             f"{COPY_BASE}:src/mme_vla_suite/shared/{f}.py"],
            capture_output=True, text=True, check=True,
        ).stdout
        src = src.replace("from mme_vla_suite.shared.", "from legacy_shared.")
        (pkg / f"{f}.py").write_text(src)
    sys.path.insert(0, str(scratch))


def load_real_frames(h5_path: str, needed: list[int]):
    """从真实 h5 取帧/状态（(256,256,3) u8——必须真实 256 帧才走 resize_with_pad 跳）。

    对拍只要求三方喂进完全相同的字节，不要求帧构成连贯 episode：多 episode 顺序
    平铺凑足 len(needed) 帧，逐一映射到 needed 里的 step。"""
    import h5py
    frames, states = [], []
    with h5py.File(h5_path, "r") as data:
        ep_ids = sorted(int(k.split("_")[1]) for k in data.keys() if k.startswith("episode_"))
        for ep in ep_ids:
            g = data[f"episode_{ep}"]
            ts_ids = sorted(int(k.split("_")[-1]) for k in g.keys() if k.startswith("timestep_"))
            for t in ts_ids:
                ts = g[f"timestep_{t}"]
                img = ts["obs"]["front_rgb"][()]
                joint = ts["obs"]["joint_state"][()]
                grip = ts["obs"]["gripper_state"][()]
                frames.append(img)
                states.append(np.concatenate([joint, grip[:1]], axis=0, dtype=np.float32))
                if len(frames) >= len(needed):
                    return frames, states
    raise SystemExit(f"错误: h5 帧数不足 {len(needed)}（实际 {len(frames)}）")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", default="/data/hongzefu/robomme_data_h5_v2_4env400ep/record_dataset_ButtonUnmask.h5")
    ap.add_argument("--out", required=True, help="判定行与逐步摘要落盘目录")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENPI_DATA_HOME", "").startswith("/"):
        # siglip_tokenizer 读 OPENPI_DATA_HOME 时不做 expanduser，必须给绝对路径
        raise SystemExit("错误: OPENPI_DATA_HOME 必须显式给绝对路径（不做 expanduser）")

    extract_legacy_shared(out / "ascratch")

    import jax
    from legacy_shared.mem_buffer import MemoryBuffer as LegacyMemoryBuffer
    from mme_vla_suite.dataset_builder.mem_buffer import MemoryBuffer as BuilderMemoryBuffer
    from mme_vla_suite.policies.framesamp_memory import FrameSampMemory
    from mme_vla_suite.shared.sampling import even_sampling_indices

    # 进程内只构造一次 jitted SigLIP，同一 callable 注入三方（建库域副本的
    # SigLipTokenizer 与 legacy 提取件逐字节相同——COPY_DIFF 已证）
    from mme_vla_suite.dataset_builder.siglip_tokenizer import SigLipTokenizer
    enc = jax.jit(SigLipTokenizer().__call__)

    dims = dict(num_views=1, img_emb_dim=2048, pos_emb_dim=768, state_emb_dim=8)
    a = LegacyMemoryBuffer(**dims, prepare_buffer=True, vision_enc_fn=enc)
    b = BuilderMemoryBuffer(**dims, prepare_buffer=True, vision_enc_fn=enc)
    c = FrameSampMemory(**dims, vision_enc_fn=enc)

    lines = []

    def emit(line: str) -> None:
        print(line, flush=True)
        lines.append(line)

    # ── POS_TABLE：4x4 pos 表全表三方逐位 ───────────────────────────────────
    sha_a = C.leaf_sha256(a.pos_emb_dict["4x4"])
    sha_b = C.leaf_sha256(b.pos_emb_dict["4x4"])
    sha_c = C.leaf_sha256(c.pos_emb_4x4)
    pos_ok = sha_a == sha_b == sha_c
    emit(f"POS_TABLE={'PASS' if pos_ok else 'FAIL'} rows={MAX_STEPS} a==b={sha_a==sha_b} a==c={sha_a==sha_c}")

    # ── 喂帧：并集覆盖每个网格步的 even_sampling_indices(step, 32) ─────────
    needed = sorted({i for s in STEP_GRID for i in even_sampling_indices(s, 32)})
    frames, states = load_real_frames(args.h5, needed)
    for k, step in enumerate(needed):
        img = frames[k][None, None, ...]   # (1,1,h,w,3) u8
        st = states[k][None, ...]          # (1,8) f32
        a.add_buffer(img, st, [step])
        b.add_buffer(img, st, [step])
        c.add_buffer(img, st, [step])
    emit(f"FED steps={len(needed)} grid={len(STEP_GRID)}")

    # ── ENC_LAYER：逐网格步逐键三方逐位 ────────────────────────────────────
    enc_mismatch = 0
    per_step = {}
    for s in STEP_GRID:
        row = {}
        for key in ENC_KEYS:
            ka = C.leaf_sha256(np.asarray(a._history_feats[s][key]))
            kb = C.leaf_sha256(np.asarray(b._history_feats[s][key]))
            kc = C.leaf_sha256(np.asarray(c._history_feats[s][key]))
            ok = ka == kb == kc
            row[key] = {"ok": ok, "a": ka[:16], "b": kb[:16], "c": kc[:16]}
            if not ok:
                enc_mismatch += 1
        per_step[s] = row
    emit(f"ENC_LAYER={'PASS' if enc_mismatch == 0 else 'FAIL'} steps={len(STEP_GRID)} keys={len(ENC_KEYS)} mismatch={enc_mismatch}")

    # ── ASSEMBLY：逐网格步 prepare_frame_sampling 四元组三方逐位 ───────────
    asm_mismatch = 0
    asm = {}
    for s in STEP_GRID:
        outs = []
        for side in (a, b, c):
            four = side.prepare_frame_sampling(s, 512, 16, side.default_history_feats_gather_fn)
            outs.append(tuple(C.leaf_sha256(np.asarray(x)) for x in four))
        ok = outs[0] == outs[1] == outs[2]
        asm[s] = {"ok": ok, "a": [h[:16] for h in outs[0]], "c": [h[:16] for h in outs[2]]}
        if not ok:
            asm_mismatch += 1
    emit(f"ASSEMBLY={'PASS' if asm_mismatch == 0 else 'FAIL'} steps={len(STEP_GRID)} mismatch={asm_mismatch}")

    # ── OOB_PROBE：step 4096 三方都必须响亮失败（禁依赖 numpy 切片行为）────
    probe_img = frames[0][None, None, ...]
    probe_st = states[0][None, ...]
    results = {}
    for name, side in (("a", a), ("b", b)):
        # 旧实现不 raise：切片静默返回空 pos——由探针自己判「存了坏数据即失败」
        side.add_buffer(probe_img, probe_st, [MAX_STEPS])
        stored = side._history_feats[MAX_STEPS]["pos_emb_4x4"]
        results[name] = "empty-slice-detected" if stored.shape[0] == 0 else "SILENT-BAD-DATA"
    try:
        c.add_buffer(probe_img, probe_st, [MAX_STEPS])
        results["c"] = "NO-RAISE"
    except ValueError:
        results["c"] = "raise"
    oob_ok = results["a"] == "empty-slice-detected" and results["b"] == "empty-slice-detected" and results["c"] == "raise"
    emit(f"OOB_PROBE={'PASS' if oob_ok else 'FAIL'} a={results['a']} b={results['b']} c={results['c']}")

    # ── 合成极值帧不炸（非判据）────────────────────────────────────────────
    extreme = np.full_like(frames[0], 255)[None, None, ...]
    c2 = FrameSampMemory(**dims, vision_enc_fn=enc)
    c2.add_buffer(extreme, probe_st, [0])
    _ = c2.prepare_frame_sampling(0, 512, 16, c2.default_history_feats_gather_fn)
    emit("EXTREME_FRAME=OK (非判据)")

    verdict = pos_ok and enc_mismatch == 0 and asm_mismatch == 0 and oob_ok
    emit(f"ONLINE_MEM={'PASS' if verdict else 'FAIL'} base={COPY_BASE[:8]}")

    (out / "verdict.txt").write_text("\n".join(lines) + "\n")
    (out / "per_step.json").write_text(json.dumps(
        {"enc_layer": {str(k): v for k, v in per_step.items()},
         "assembly": {str(k): v for k, v in asm.items()},
         "needed_steps": needed}, ensure_ascii=False, indent=2))
    if not verdict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
