#!/usr/bin/env python3
"""P5：真编码器在线链 vs 离线 motion 表（0901-motion-memory-plan.md 四节表一 P5；run `motion-p5-online`）。

驱动：从录制 h5 按 `episode_manifest.json` 逐 episode 读全部 `front_rgb` (256,256,3) uint8 帧，沿 eval.py 真实节奏喂
`FrameSampMemory.add_buffer`（首批 pre_traj = demo [0, es) + exec 首帧、之后每批 16 帧、`exec_start_idx` 下传），运动路编码走真
`motion_sidecar.py`（wan 子 venv、fp32 / 关 TF32 / B=1 / 33 帧一次喂，独占 `--gpu` 那张卡）。40 条 episode 全跑。

五条判据 + provenance + 三笔耗时：
  ONLINE_ENC_BITEXACT=PASS compared=772 mismatches=0   每窗在线 (768,) 与离线表对应行 `np.array_equal`，772 行全覆盖
  ONLINE_START_SET=PASS steps=N                        每个推理时刻在线起点集合 == `motion_store.visible_motion_rows`
  ONLINE_POS=PASS                                      `motion_pos` 行 == `FrameSampStore.pos_rows([f])[0, 0, :256]` 逐位
  ONLINE_ORDER=PASS steps=N                            `mem_order`（含 motion_emb/pos/mask 四键）== 训练侧 `FrameSampDataset.__getitem__` 逐位，且为合法置换
  PROVENANCE=PASS                                      sidecar 握手 provenance 与 store_meta.provenance 逐键相等（客户端构造时已 raise 兜底）
  耗时：每窗（客户端夹 send/recv）、首批 demo（首次 add_buffer 挂钟）、每次推理前固定开销（后续每批 add_buffer 挂钟，含帧路 SigLIP 编码）
`--stub` 档：帧换成编号合成帧、验起点集合 / pos / order；错误 stub token 仍须拒绝。
CPU 的非装配档必须显式传 --gpu-posemb-cache：GPU 预生成的生产 PosEmb3D 输出已经与
整张离线表逐位核对；这里复核来源与字节后，只在验证对象中注入该组件输出。

用法（主树）：
  UV_LINK_MODE=copy uv run --no-sync python scripts/training/g0/compare_online_motion.py --gpu 1 --out v1-store/reports/motion/p5_online.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import subprocess
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "training" / "tests"))
_V1 = pathlib.Path(os.environ.get("MMEVLA_V1_STORE", str(_REPO_ROOT / "v1-store")))

from mme_vla_suite.datastore import motion_store as ms          # noqa: E402
from mme_vla_suite.datastore.framesamp_store import StoreMeta   # noqa: E402
from mme_vla_suite.policies.framesamp_memory import FrameSampMemory  # noqa: E402
from mme_vla_suite.policies.motion_client import MotionEncoderClient  # noqa: E402
from mme_vla_suite.shared.sampling import even_sampling_indices  # noqa: E402

HIST_BUDGET, TOKEN_PER_IMAGE, NUM_VIEWS, POS_DIM = 512, 16, 1, 256


def load_episode(raw_dir: pathlib.Path, h5_file: str, raw_ep_idx: int, T: int, *, frames_only=False):
    import h5py
    frames, states = [], []
    with h5py.File(raw_dir / h5_file, "r") as f:
        g = f[f"episode_{raw_ep_idx}"]
        ts_ids = sorted(int(k.split("_")[-1]) for k in g.keys() if k.startswith("timestep_"))
        if len(ts_ids) != T or ts_ids != list(range(T)):
            raise SystemExit(f"错误: {h5_file} episode_{raw_ep_idx} timesteps {len(ts_ids)} != 清单 {T}")
        for t in ts_ids:
            ts = g[f"timestep_{t}"]
            img = ts["obs"]["front_rgb"][()]
            if img.shape != (256, 256, 3) or img.dtype != np.uint8:
                raise SystemExit(f"错误: 帧形制 {img.shape} {img.dtype}")
            frames.append(img)
            if not frames_only:
                joint = ts["obs"]["joint_state"][()]
                grip = ts["obs"]["gripper_state"][()]
                states.append(np.concatenate([joint, grip[:1]], axis=0, dtype=np.float32))
    return np.stack(frames)[:, None], None if frames_only else np.stack(states)


class _Cfg:
    budget = HIST_BUDGET; token_per_image = TOKEN_PER_IMAGE; num_views = NUM_VIEWS


def _bare_policy(mem, motion_cfg, norm):
    from mme_vla_suite.policies.policy import MME_VLA_Policy
    pol = MME_VLA_Policy.__new__(MME_VLA_Policy)
    pol.config = _Cfg(); pol.mem_buffer = mem; pol.motion_enabled = True; pol._motion_cfg = motion_cfg
    pol.step_idx = -1; pol.exec_start_idx = 0; pol.use_quantiles = False; pol.state_norm_stats = norm
    return pol


def bytes_equal(a, b) -> bool:
    """逐位比较同时约束 dtype 与 shape，区分符号零。"""
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def select_real_episodes(manifest: dict) -> set[int]:
    """独立按清单选择最大 k、大 es、全部余数层与每个任务的整集。"""
    eps = manifest["episodes"]
    def maximum(ep):
        es, total = int(ep["exec_start_idx"]), int(ep["num_timesteps"])
        return len(range(0, max(0, es-16), 16)) + len(range(0, max(0, total-es-32), 16))
    selected = {int(max(eps, key=lambda e: (maximum(e), -int(e["global_episode_idx"])))["global_episode_idx"])}
    selected.update(int(e["global_episode_idx"]) for e in [e for e in eps if int(e["exec_start_idx"]) >= 1000][:3])
    residues, tasks = {}, {}
    for e in eps:
        g = int(e["global_episode_idx"])
        residues.setdefault((int(e["exec_start_idx"])-17) % 16, g)
        tasks.setdefault(e["h5_file"], g)
    selected.update(residues.values()); selected.update(tasks.values())
    return selected


def row_metadata(meta, latent_meta: dict):
    """把离线逐窗元数据映射到全局行，保留独立输入 SHA 与补帧集合。"""
    result = {}
    for entry in meta.entries:
        for kind in ("demo", "exec"):
            segment = getattr(entry, kind)
            if not segment.num_grid:
                continue
            key = ms.segment_key(entry.h5_file, entry.raw_ep_idx, kind)
            rows = latent_meta["segments"][key]["rows"]
            if len(rows) != segment.num_grid or {r["m"] for r in rows} != set(range(segment.num_grid)):
                raise ValueError(f"离线 metadata 的窗口集合不完整: {key}")
            for row in rows:
                index = segment.row_base + int(row["m"])
                result[index] = row
    if len(result) != meta.num_rows:
        raise ValueError("离线逐窗元数据未覆盖整张 motion 表")
    return result


def report_passed(report: dict) -> bool:
    """已发现的失配在任何模式都阻断；没有 stub 短路。"""
    if report["mismatch_total"] != 0 or report["start_ok"] is not True:
        return False
    if len(report["rows_seen"]) != report["expected_rows"]:
        return False
    if report["mode"] != "assembly" and (report["pos_ok"] is not True or report["order_ok"] is not True):
        return False
    if report["mode"] in ("real", "real_strata") and report["compared_real"] != report["expected_real_rows"]:
        return False
    return True


def aggregate(argv) -> int:
    ap = argparse.ArgumentParser(description="在线分片的结构化验收，不解析日志文本")
    ap.add_argument("--reports", nargs="+", required=True)
    ap.add_argument("--lib", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--expect-episodes", type=int, default=1600)
    ap.add_argument("--expect-rows", type=int, default=71316)
    ap.add_argument("--assembly-report", type=pathlib.Path)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"拒绝覆盖汇总: {args.out}")
    manifest = ms.load_manifest(args.lib / "meta/episode_manifest.json")
    meta = ms.MotionMeta.load(args.lib / "motion")
    reports = [json.loads(pathlib.Path(p).read_text()) for p in args.reports]
    required = {"mode","shard_idx","num_shards","episode_ids","rows_seen","real_rows","padded_covered",
                "start_ok","pos_ok","order_ok","n_compared","n_steps","assembly_checked","compared_real",
                "expected_rows","expected_real_rows","mismatch_total","manifest_sha256","yaml_sha256",
                "metadata_sha256","source_head","source_status","selected_full_episodes","lib","position_cache"}
    if len(reports) != args.num_shards or any(required-set(r) for r in reports):
        raise ValueError("分片数量不符或缺少结构化字段")
    if {r["shard_idx"] for r in reports} != set(range(args.num_shards)):
        raise ValueError("分片索引集合不完整")
    for key in ("mode","num_shards","manifest_sha256","yaml_sha256","metadata_sha256","source_head","selected_full_episodes","lib","position_cache"):
        if len({json.dumps(r[key], sort_keys=True) for r in reports}) != 1:
            raise ValueError(f"各片 {key} 不一致")
    first = reports[0]
    if (first["num_shards"] != args.num_shards or first["manifest_sha256"] != manifest["sha256"]
            or pathlib.Path(first["lib"]).resolve() != args.lib.resolve() or any(r["source_status"] for r in reports)):
        raise ValueError("分片来源、源码状态或清单身份不符")
    if len(manifest["episodes"]) != args.expect_episodes or meta.num_rows != args.expect_rows:
        raise ValueError("数据规模与外部期望值不同")
    all_eps, all_rows, real_rows, padded = set(), set(), set(), set()
    for r in reports:
        if not report_passed(r):
            raise ValueError(f"分片 {r['shard_idx']} 判定失败")
        eps, rows = set(r["episode_ids"]), set(r["rows_seen"])
        expected_eps = {e.g for e in meta.entries if e.g % args.num_shards == r["shard_idx"]}
        expected_rows = {s.row_base+m for e in meta.entries if e.g in expected_eps
                         for s in (e.demo,e.exec) for m in range(s.num_grid)}
        if (eps != expected_eps or rows != expected_rows or len(rows) != len(r["rows_seen"])
                or all_eps & eps or all_rows & rows):
            raise ValueError("分片 episode 或窗口存在重复、遗漏或切分口径错误")
        all_eps |= eps; all_rows |= rows
        real_rows.update(r["real_rows"]); padded.update(r["padded_covered"])
    if len(all_eps) != args.expect_episodes or all_rows != set(range(args.expect_rows)):
        raise ValueError("分片并集不覆盖全库")
    latent_path = args.lib / "wan-latents/metadata.json"
    if ms.sha256_file(latent_path) != first["metadata_sha256"]:
        raise ValueError("离线 metadata 在各档验证之间变化")
    row_meta = row_metadata(meta, json.loads(latent_path.read_text()))
    expected_pad = {i for i,r in row_meta.items() if int(r.get("pad_frames",0)) > 0}
    if padded != expected_pad:
        raise ValueError("在线补帧集合与离线 pad_frames 集合不同")
    assembly_checked = sum(r["assembly_checked"] for r in reports)
    mode = first["mode"]
    if mode == "assembly":
        if assembly_checked != args.expect_rows:
            raise ValueError("33 帧装配未覆盖全库")
        print(f"ONLINE_INPUT_SHA=PASS windows={len(all_rows)} padded={len(padded)} mismatches=0")
    elif mode == "real_strata":
        selected = select_real_episodes(manifest)
        if set(first["selected_full_episodes"]) != selected:
            raise ValueError("必含整集选择与独立重算不同")
        expected_real = expected_pad | {s.row_base+m for e in meta.entries if e.g in selected
                                       for s in (e.demo,e.exec) for m in range(s.num_grid)}
        if real_rows != expected_real or sum(r["compared_real"] for r in reports) != len(real_rows):
            raise ValueError("真编码器未覆盖全部补帧窗与必含整集")
        chosen = [e for e in manifest["episodes"] if e["global_episode_idx"] in selected]
        big = sum(e["exec_start_idx"] >= 1000 for e in chosen)
        residues = {(e["exec_start_idx"]-17) % 16 for e in chosen}
        tasks = {e["h5_file"] for e in manifest["episodes"]}
        per_task_min = min(sum(e["h5_file"] == task for e in chosen) for task in tasks)
        if 367 not in selected or big < 3 or len(residues) != 16 or per_task_min < 1 or len(padded) != 1600:
            raise ValueError("真编码器分层覆盖不满足计划")
        if args.assembly_report is None:
            raise ValueError("真编码器汇总必须引用已通过的装配汇总")
        assembly = json.loads(args.assembly_report.read_text())
        if assembly.get("passed") is not True or assembly.get("mode") != "assembly":
            raise ValueError("装配汇总未通过")
        for key in ("manifest_sha256","yaml_sha256","metadata_sha256","source_head"):
            if assembly[key] != first[key]:
                raise ValueError(f"装配档与真编码档 {key} 不同")
        assembly_checked = assembly["assembly_checked"]
        if assembly_checked != args.expect_rows:
            raise ValueError("装配引用覆盖不足")
        print(f"ONLINE_STRATA=PASS padded={len(padded)}/1600 kmax_ep=367 es_ge1000={big} pad_r=16/16 per_task_min={per_task_min}")
    print(f"SHARD_SET=PASS shards={args.num_shards} eps={len(all_eps)} disjoint=1")
    if mode != "assembly":
        print("ONLINE_START_SET=PASS\nONLINE_POS=PASS\nONLINE_ORDER=PASS")
    print(f"P5_ONLINE=PASS episodes={len(all_eps)} windows={len(all_rows)} assembly_checked={assembly_checked} "
          f"compared_real={len(real_rows)} padded_covered={len(padded)} stub={mode == 'stub'}")
    result = {k:first[k] for k in ("mode","manifest_sha256","yaml_sha256","metadata_sha256","source_head","selected_full_episodes","position_cache")}
    result.update(passed=True, episodes=len(all_eps), windows=len(all_rows), assembly_checked=assembly_checked,
                  compared_real=len(real_rows), padded_covered=len(padded), shard_reports=args.reports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "aggregate":
        return aggregate(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", default=str(_V1 / "datasets/4task-motion-40ep"))
    ap.add_argument("--yaml", default="perceptual-framesamp-context-motion.yaml")
    ap.add_argument("--store-subdir", choices=("framesamp","framesamp-8x8"), default="framesamp")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--gpu-posemb-cache", type=pathlib.Path)
    ap.add_argument("--episodes", type=int, default=0)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--stub", action="store_true")
    group.add_argument("--assembly-hash", type=pathlib.Path)
    group.add_argument("--real-strata", action="store_true")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default=str(_V1 / "reports/motion/p5_online.json"))
    ap.add_argument("--inject-stub-token-error", action="store_true")
    ap.add_argument("--inject-signed-zero-error", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.shard_idx < args.num_shards or args.episodes < 0:
        raise ValueError("分片或 episode 参数不合法")
    if (args.inject_stub_token_error or args.inject_signed_zero_error) and not args.stub:
        raise ValueError("反向注入只允许 stub 验证")
    out = pathlib.Path(args.out)
    if out.exists():
        raise FileExistsError(f"拒绝覆盖在线报告: {out}")

    from motion_gates_model import _load_yaml, _make_dataset
    from mme_vla_suite.training.dataloader import _motion_gates
    from mme_vla_suite.policies import motion_protocol as protocol
    lib = pathlib.Path(args.lib).resolve()
    hc = _load_yaml(args.yaml)
    _Cfg.budget, _Cfg.token_per_image, _Cfg.num_views = int(hc.budget), int(hc.token_per_image), int(hc.num_views)
    mcfg = hc.motion
    if ("demo_min_real_frames" in mcfg) != ("demo_tail_pad" in mcfg):
        raise ValueError("demo 补帧配置必须成对出现")
    motion_cfg = {k:mcfg[k] for k in ("stride","window_frames","budget","frame_size","pos_dim","dim","window_direction","grid_origin")}
    motion_cfg.update(demo_min_real_frames=mcfg.get("demo_min_real_frames",33), demo_tail_pad=mcfg.get("demo_tail_pad","none"))
    os.environ.pop("MMEVLA_MOTION_STORE", None)
    resolved = _motion_gates(hc, StoreMeta.load(lib / args.store_subdir))
    if pathlib.Path(resolved).resolve() != lib / "motion":
        raise ValueError("YAML store_path 与本轮目标 motion 表不同")
    print(f"YAML_STORE_PATH=PASS path={resolved}", flush=True)
    ds = _make_dataset(lib, args.yaml, store_subdir=args.store_subdir)
    meta = ds._motion_meta
    manifest = ms.load_manifest(lib / "meta/episode_manifest.json")
    latent_dir = args.assembly_hash or lib / "wan-latents"
    metadata_path = latent_dir / "metadata.json"
    row_meta = row_metadata(meta, json.loads(metadata_path.read_text()))
    entries = meta.entries
    eps = [ep for ep in manifest["episodes"][:args.episodes or None] if ep["global_episode_idx"] % args.num_shards == args.shard_idx]
    episode_ids = {int(e["global_episode_idx"]) for e in eps}
    expected_rows = {s.row_base+m for e in entries if e.g in episode_ids for s in (e.demo,e.exec) for m in range(s.num_grid)}
    expected_pad = {i for i in expected_rows if int(row_meta[i].get("pad_frames",0)) > 0}
    selected = select_real_episodes(manifest) if args.real_strata else set()
    expected_real = expected_pad | {s.row_base+m for e in entries if e.g in selected & episode_ids
                                   for s in (e.demo,e.exec) for m in range(s.num_grid)}
    if not args.real_strata:
        expected_real = expected_rows
    mode = "assembly" if args.assembly_hash else ("stub" if args.stub else ("real_strata" if args.real_strata else "real"))
    import jax
    position_table = position_cache = None
    if mode != "assembly":
        if args.gpu_posemb_cache:
            from verified_gpu_posemb import load_verified_table
            position_table, position_cache = load_verified_table(
                args.gpu_posemb_cache, lib / args.store_subdir, int(hc.token_per_image))
            print(f"GPU_POSEMB_CACHE=PASS sha256={position_cache['binary_sha256']}", flush=True)
        elif jax.default_backend() == "cpu":
            raise ValueError("CPU 在线验证必须传 --gpu-posemb-cache，不能直接用 CPU PosEmb3D 冒充 GPU 逐位结果")
    source_head = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    source_status = subprocess.check_output(["git","status","--porcelain"], text=True)
    yaml_path = _REPO_ROOT / "src/mme_vla_suite/models/config/robomme" / args.yaml
    seen, real_rows, padded = set(), set(), set()
    mismatches, mismatch_total = [], 0
    start_ok = True
    pos_ok = order_ok = None if mode == "assembly" else True
    steps = assembly_checked = 0

    def mark(kind, **detail):
        nonlocal mismatch_total
        mismatch_total += 1
        if len(mismatches) < 200:
            mismatches.append({"kind":kind, **detail})

    client = None
    store = None if mode in ("assembly","stub") else ds._ensure_motion_store()
    frame_store = None if mode == "assembly" else ds._ensure_store()
    if mode != "assembly":
        client = MotionEncoderClient(online_gpu=args.gpu, stub=args.stub, source_run=str(mcfg.source_run),
                                     store_provenance={"vae":meta.provenance.get("vae"),"encoder":meta.provenance.get("encoder")},
                                     expected_ckpt_sha256=None if args.stub else meta.provenance["encoder"]["checkpoint_sha256"])
    import jax.numpy as jnp
    def vision_enc(x):
        return jnp.zeros((*x.shape[:2],64,2048), jnp.bfloat16)

    try:
        for number, ep in enumerate(eps):
            g, es, total = int(ep["global_episode_idx"]), int(ep["exec_start_idx"]), int(ep["num_timesteps"])
            entry = entries[g]
            def encode(window, start):
                nonlocal assembly_checked
                segment = entry.demo if start < es else entry.exec
                offset = start if start < es else start-es
                if offset % 16 or segment.row_base is None or not 0 <= offset//16 < segment.num_grid:
                    raise ValueError("在线生成了离线集合以外的起点")
                row = segment.row_base + offset//16
                if row in seen:
                    raise ValueError("同一个窗口被重复编码")
                seen.add(row)
                if start < es and start+33 > es:
                    padded.add(row)
                if mode == "assembly":
                    assembly_checked += 1
                    if hashlib.sha256(np.ascontiguousarray(window).tobytes()).hexdigest() != row_meta[row]["input_frames_sha256"]:
                        mark("input_sha",g=g,start=start,row=row)
                    return np.zeros(768,np.float32)
                if args.stub:
                    token = client(window,start)
                    if args.inject_stub_token_error and start == 0:
                        token = token + np.float32(1)
                    if args.inject_signed_zero_error and start == 0:
                        token = token.copy(); token[0] = np.float32(-0.0)
                    if not bytes_equal(token,np.full(768,float(start),np.float32)):
                        mark("stub_token",g=g,start=start)
                    return token
                off = store.rows(np.asarray([row],np.int64))[0]
                if mode == "real" or row in expected_real:
                    token = client(window,start)
                    real_rows.add(row)
                    if not bytes_equal(token,off):
                        mark("token",g=g,start=start,row=row)
                    return token
                return off.copy()

            if mode == "stub":
                frames = np.stack([protocol.stub_frame(t) for t in range(total)])[:,None]
                states = np.zeros((total,8),np.float32)
            else:
                frames, states = load_episode(pathlib.Path(manifest["raw_dir"]),ep["h5_file"],int(ep["raw_ep_idx"]),total,
                                               frames_only=mode == "assembly")
            if mode == "assembly":
                # 此档只验证生产装配方法；时间调度、pos 与排序由独立 stub 档逐决策时刻覆盖。
                mem = FrameSampMemory.__new__(FrameSampMemory)
                mem.exec_start_idx = es
                mem.motion_window, mem.motion_stride, mem.motion_dim = 33,16,768
                mem.demo_min_real, mem.demo_tail_pad = motion_cfg["demo_min_real_frames"],motion_cfg["demo_tail_pad"]
                mem._raw_frames = {t:frames[t,0] for t in range(total)}
                mem._history_feats_motion = {}
                mem._next_grid_start_demo = mem._next_grid_start_exec = 0
                mem.motion_encode_calls = 0; mem.motion_encode_s = 0.0
                mem.motion_enc = encode
                mem._encode_ready_windows(total-1)
            else:
                mem = FrameSampMemory(token_per_image=int(hc.token_per_image),vision_enc_fn=vision_enc,
                                      max_steps=1 if position_table is not None else 4096,
                                      motion_enc_fn=encode,motion_cfg=motion_cfg)
                if position_table is not None:
                    # 只替换已独立验证的组件输出；避免每集在 CPU 再算一张不会使用的位置表。
                    mem.pos_emb = position_table
                    mem.max_steps = position_table.shape[0]
                pol = _bare_policy(mem,motion_cfg,ds.state_norm_stats)
                def feed(lo, hi):
                    pol.add_buffer({"images":frames[lo:hi],"state":states[lo:hi],"exec_start_idx":es if lo == 0 else 0})
                def check(t):
                    nonlocal steps,start_ok,pos_ok,order_ok
                    steps += 1
                    _, expected_frames = ms.visible_motion_rows(entry,t)
                    if mem.visible_motion_frames(t) != expected_frames.tolist() or sorted(mem._history_feats_motion) != expected_frames.tolist():
                        start_ok = False; mark("start_set",g=g,t=t)
                    item = pol._prepare_history({})
                    count = len(expected_frames)
                    expected_pos = frame_store.pos_rows(expected_frames)[:,0,:POS_DIM] if count else np.zeros((0,POS_DIM),np.float32)
                    if not bytes_equal(item["motion_pos"][:count],expected_pos) or np.ascontiguousarray(item["motion_pos"][count:]).view(np.uint8).any():
                        pos_ok = False; mark("pos",g=g,t=t)
                    sample = ds[int(ep["exec_sample_offset"])+t-es]
                    keys = ("motion_pos","motion_mask","mem_order") if args.stub else ("motion_emb","motion_pos","motion_mask","mem_order")
                    for key in keys:
                        if not bytes_equal(item[key],sample[key]):
                            order_ok = False; mark(key,g=g,t=t)
                    order = np.asarray(item["mem_order"])
                    if order.dtype != np.int32 or not np.array_equal(np.sort(order),np.arange(len(order))):
                        order_ok = False; mark("permutation",g=g,t=t)
                feed(0,es+1); check(es)
                t = es
                while t+16 < total:
                    feed(t+1,t+17); t += 16; check(t)
            if (number+1) % 20 == 0 or number+1 == len(eps):
                print(f"ONLINE_PROGRESS shard={args.shard_idx} episodes={number+1}/{len(eps)} rows={len(seen)} real={len(real_rows)} mismatches={mismatch_total}",flush=True)
            del mem, frames, states
    finally:
        if client is not None:
            client.close()
        ds.close()
    if seen != expected_rows:
        start_ok = False; mark("row_set")
    if padded != expected_pad:
        mark("padded_set")
    if client is not None and client.n_calls != (len(seen) if args.stub else len(real_rows)):
        mark("client_calls")
    if subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip() != source_head or subprocess.check_output(["git","status","--porcelain"],text=True) != source_status:
        mark("source_changed")
    report = dict(schema=2,mode=mode,lib=str(lib),shard_idx=args.shard_idx,num_shards=args.num_shards,
                  episode_ids=sorted(episode_ids),rows_seen=sorted(seen),real_rows=sorted(real_rows),
                  padded_covered=sorted(padded),expected_rows=len(expected_rows),expected_real_rows=len(expected_real),
                  start_ok=start_ok,pos_ok=pos_ok,order_ok=order_ok,n_compared=len(seen),n_steps=steps,
                  assembly_checked=assembly_checked,compared_real=len(real_rows),mismatch_total=mismatch_total,
                  mismatches=mismatches,manifest_sha256=manifest["sha256"],yaml_sha256=ms.sha256_file(yaml_path),
                  metadata_sha256=ms.sha256_file(metadata_path),source_head=source_head,source_status=source_status,
                  selected_full_episodes=sorted(selected),position_cache=position_cache,
                  provenance=None if client is None else client.provenance,argv=sys.argv)
    passed = report_passed(report)
    report["passed"] = passed
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x") as f:
        json.dump(report,f,ensure_ascii=False,indent=1,default=str)
    if mode == "assembly":
        print(f"ONLINE_INPUT_SHA={'PASS' if passed else 'FAIL'} windows={assembly_checked} padded={len(padded)} mismatches={mismatch_total}")
    else:
        print(f"ONLINE_START_SET={'PASS' if start_ok else 'FAIL'}\nONLINE_POS={'PASS' if pos_ok else 'FAIL'}\nONLINE_ORDER={'PASS' if order_ok else 'FAIL'}")
        print(f"ONLINE_ENC_BITEXACT={'SKIP(stub)' if args.stub else ('PASS' if passed else 'FAIL')} compared_real={len(real_rows)} mismatches={mismatch_total}")
        print("PROVENANCE=SKIP(stub)" if args.stub else "PROVENANCE=PASS")
    print(f"P5_ONLINE={'PASS' if passed else 'FAIL'} episodes={len(eps)} windows={len(seen)} assembly_checked={assembly_checked} compared_real={len(real_rows)} padded_covered={len(padded)} stub={args.stub}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
