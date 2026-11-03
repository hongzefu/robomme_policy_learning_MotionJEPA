"""attention 汇总与分流（计划 1103-interpret-motion-attention-plan.html 第二部分 §2.3、§1.6）。

输入：每局一个目录 <runs>/<episode_nonce>/，内含 infer_<seq:04d>.npz（键见计划 §1.3，单一真源）
与 episode.json（S2 写）；可选 sim_eef_pose.json（read_sim_eef_pose.py 写）。

做三件事：
1. 把原始 probs_action（10 去噪步 × 18 层 × 8 头 × 20 动作 query × (P+20) key）按命令行选定的
   步／层／头平均成一个 key 向量 a[s]，再按 §2.3 拆成五流 + padding：
   帧流（mem_order[s] < 512，帧 f = mem_order[s] // 16，帧号 frame_times[f]）、
   motion 流（mem_order[s] ≥ 512，窗 w = mem_order[s] − 512，起点 motion_times[w]）、
   当前图像（base 256 + wrist 256）、文本、动作；mask 为 False 的列记为 pad。
   断言 pad 质量 ≤ 1e-6、五流质量之和 = 1 ± 1e-4。
2. --candidates：读 sim_eef_pose.json，列出「动作块内夹爪由开到闭、闭合后块内 TCP z 上升 ≥ 3 cm」
   的全部推理，供主会话看视频后定唯一的「揭开」推理。
3. --key-infer：对选定推理算 VIS_SHARE／SWAP_SHARE 与各自的均匀期望，打印 KEY_STEP= 行。

输出 <out>/<局目录名>/streams.json 与 <out>/summary.json，末行 ATTN_AGG=PASS|FAIL。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# §2.3 常量：记忆区里帧 token 占前 512 个原始序号（32 帧 × 16），其后是 motion 槽。
FRAME_TOKENS = 512
TOKENS_PER_FRAME = 16
IMAGE_BASE = 256
IMAGE_WRIST = 256
ACTION_TOKENS = 20
MOTION_WINDOW = 33
# §1.6 窗口常量：方块裸露窗口 [0,32)；第 j 次交换 [64+50(j−1), 64+50j)。
VISIBLE = (0, 32)
SWAP_START = 64
SWAP_LEN = 50
# 判据阈值（计划 §五 ATTN_AGG 行），不得放宽。
PAD_TOL = 1e-6
SUM_TOL = 1e-4
# 「揭开」候选：闭合后块内 TCP z 上升阈值（米）。
LIFT_THRESH = 0.03


# ---------------------------------------------------------------- 聚合

def _parse_index(spec: str | None, n: int) -> np.ndarray:
    """把 "all"／"3"／"0,2,5"／"10-17" 解析成索引数组。"""
    if spec is None or spec == "all":
        return np.arange(n)
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    idx = np.asarray(out, dtype=np.int64)
    if idx.size == 0 or idx.min() < 0 or idx.max() >= n:
        raise ValueError(f"索引 {spec!r} 越界（范围 0..{n - 1}）")
    return idx


def aggregate(npz, *, query: str = "action", steps: str = "all",
              layers: str | None = None, heads: str | None = None) -> np.ndarray:
    """按 K4 把原始概率平均成 key 向量 a[s]。

    query=action：probs_action (10,18,8,20,P+20) 在选定步／层／头与全部 20 个动作 query 上平均，长度 P+20；
    query=text：probs_text (18,8,64,P) 在选定层／头与前 txt_len 个有效文本 query 上平均，长度 P。
    """
    if query == "action":
        x = np.asarray(npz["probs_action"])
        n_steps = x.shape[0]
        if steps == "last":
            si = np.array([n_steps - 1])
        elif steps == "all":
            si = np.arange(n_steps)
        else:
            si = _parse_index(steps, n_steps)
        x = x[si][:, _parse_index(layers, x.shape[1])][:, :, _parse_index(heads, x.shape[2])]
        return x.astype(np.float64).mean(axis=(0, 1, 2, 3))
    if query == "text":
        x = np.asarray(npz["probs_text"])
        txt_len = int(npz["txt_len"])
        x = x[_parse_index(layers, x.shape[0])][:, _parse_index(heads, x.shape[1])][:, :, :txt_len]
        return x.astype(np.float64).mean(axis=(0, 1, 2))
    raise ValueError(f"未知 query {query!r}")


def _mem_layout(npz) -> tuple[int, int, np.ndarray, np.ndarray]:
    """取 P、MEM、mem_order、mem_mask；基线无 mem_order 时恒等映射、mask 全真。"""
    P = int(npz["P"])
    MEM = int(npz["MEM"])
    keys = set(npz.keys()) if hasattr(npz, "keys") else set(npz.files)
    mem_order = np.asarray(npz["mem_order"], dtype=np.int64) if "mem_order" in keys else np.arange(MEM)
    mem_mask = np.asarray(npz["mem_mask"], dtype=bool) if "mem_mask" in keys else np.ones(MEM, bool)
    if mem_order.shape != (MEM,) or mem_mask.shape != (MEM,):
        raise ValueError(f"mem_order／mem_mask 形状应为 ({MEM},)，得到 {mem_order.shape}／{mem_mask.shape}")
    if P != MEM + IMAGE_BASE + IMAGE_WRIST + 64:
        raise ValueError(f"P={P} 与 MEM={MEM}+512+64 不符")
    return P, MEM, mem_order, mem_mask


def split_streams(npz, *, query: str = "action", steps: str = "all",
                  layers: str | None = None, heads: str | None = None) -> dict:
    """严格按 §2.3 分流，返回各流质量；断言 pad ≤ 1e-6、五流和 = 1 ± 1e-4。"""
    a = aggregate(npz, query=query, steps=steps, layers=layers, heads=heads)
    P, MEM, mem_order, mem_mask = _mem_layout(npz)
    expect_len = P + ACTION_TOKENS if query == "action" else P
    if a.shape != (expect_len,):
        raise ValueError(f"聚合向量长度 {a.shape} 与期望 {expect_len} 不符")

    frame_times = np.asarray(npz["frame_times"], dtype=np.int64)
    motion_times = np.asarray(npz["motion_times"], dtype=np.int64).reshape(-1)
    motion_mask = np.asarray(npz["motion_mask"], dtype=bool).reshape(-1)
    n_frames = FRAME_TOKENS // TOKENS_PER_FRAME
    A_frame = np.zeros(n_frames)
    A_motion = np.zeros(motion_times.shape[0])

    mem_a = a[:MEM]
    valid = mem_mask
    is_frame = mem_order < FRAME_TOKENS
    # 帧流：f = mem_order[s] // 16，同帧 16 个 token 求和。
    fs = mem_order[valid & is_frame] // TOKENS_PER_FRAME
    np.add.at(A_frame, fs, mem_a[valid & is_frame])
    if np.any(frame_times[np.unique(fs)] < 0):
        raise ValueError("有效帧 token 指向 frame_times 为 −1 的帧")
    # motion 流：w = mem_order[s] − 512。
    ws = mem_order[valid & ~is_frame] - FRAME_TOKENS
    if ws.size:
        if ws.max() >= A_motion.shape[0]:
            raise ValueError(f"motion 窗号 {ws.max()} 超出 motion_times 长度 {A_motion.shape[0]}")
        if not np.all(motion_mask[np.unique(ws)]):
            raise ValueError("有效 motion token 指向 motion_mask 为 False 的窗")
        np.add.at(A_motion, ws, mem_a[valid & ~is_frame])
    pad = float(mem_a[~valid].sum())

    img0 = MEM
    image_base = float(a[img0:img0 + IMAGE_BASE].sum())
    image_wrist = float(a[img0 + IMAGE_BASE:img0 + IMAGE_BASE + IMAGE_WRIST].sum())
    t0 = img0 + IMAGE_BASE + IMAGE_WRIST
    txt_len = int(npz["txt_len"])
    text = float(a[t0:t0 + txt_len].sum())
    pad += float(a[t0 + txt_len:P].sum())  # 文本 padding 位
    action = float(a[P:].sum()) if query == "action" else 0.0

    frame_mass = float(A_frame.sum())
    motion_mass = float(A_motion.sum())
    total = frame_mass + motion_mass + image_base + image_wrist + text + action + pad
    sum_err = abs(total - 1.0)
    assert pad <= PAD_TOL, f"pad 质量 {pad:.3e} > {PAD_TOL}"
    assert sum_err <= SUM_TOL, f"五流和误差 {sum_err:.3e} > {SUM_TOL}"

    frames = [{"f": int(f), "t": int(frame_times[f]), "mass": float(A_frame[f])}
              for f in range(n_frames) if frame_times[f] >= 0]
    motion = [{"w": int(w), "t": int(motion_times[w]), "mass": float(A_motion[w])}
              for w in range(A_motion.shape[0]) if motion_mask[w]]
    return {
        "P": P, "MEM": MEM, "has_motion": bool(MEM > FRAME_TOKENS),
        "step_idx": int(npz["step_idx"]), "infer_seq": int(npz["infer_seq"]),
        "exec_start_idx": int(npz["exec_start_idx"]),
        "frame": frames, "motion": motion,
        "mass": {"frame": frame_mass, "motion": motion_mass, "image_base": image_base,
                 "image_wrist": image_wrist, "text": text, "action": action, "pad": pad},
        "pad": pad, "sum_err": sum_err,
    }


# ---------------------------------------------------------------- 关键推理

def swap_interval(k: int) -> tuple[int, int]:
    """交换区间并集 [64, 64+50k)。"""
    return SWAP_START, SWAP_START + SWAP_LEN * int(k)


def key_step_shares(streams: dict, k: int | None) -> dict:
    """VIS_SHARE／SWAP_SHARE 与均匀期望（§1.6）。基线或 motion 流为空时 swap 记 None。"""
    fr = streams["frame"]
    tot_f = sum(x["mass"] for x in fr)
    in_vis = [x for x in fr if VISIBLE[0] <= x["t"] < VISIBLE[1]]
    vis_share = sum(x["mass"] for x in in_vis) / tot_f if tot_f > 0 else float("nan")
    vis_uniform = len(in_vis) / len(fr) if fr else float("nan")
    swap_share = swap_uniform = None
    mo = streams["motion"]
    if streams["has_motion"] and mo:
        if k is None:
            raise ValueError("motion 模型算 SWAP_SHARE 需要交换次数 k")
        lo, hi = swap_interval(k)
        hit = [x for x in mo if x["t"] < hi and x["t"] + MOTION_WINDOW > lo]
        tot_m = sum(x["mass"] for x in mo)
        swap_share = sum(x["mass"] for x in hit) / tot_m if tot_m > 0 else float("nan")
        swap_uniform = len(hit) / len(mo)
    return {"vis_share": vis_share, "vis_uniform": vis_uniform,
            "swap_share": swap_share, "swap_uniform": swap_uniform}


def find_candidates(npzs: dict[int, dict], pose: dict) -> list[dict]:
    """动作块内夹爪（actions[:,7]，1 开 −1 闭）由开到闭、闭合后块内 TCP z 上升 ≥ 3 cm 的推理。"""
    out = []
    infers = pose["infers"]
    for seq in sorted(npzs):
        rec = infers.get(str(seq))
        if rec is None:
            continue
        g = np.asarray(npzs[seq]["actions"], dtype=np.float64)[:, 7]
        z = np.asarray(rec["action_z"], dtype=np.float64)
        closes = [i for i in range(1, len(g)) if g[i - 1] > 0 and g[i] <= 0]
        for i in closes:
            rise = float(z[i:].max() - z[i])
            if rise >= LIFT_THRESH:
                out.append({"infer_seq": seq, "step_idx": int(npzs[seq]["step_idx"]), "close_row": i,
                            "rise_m": rise, "state_z": rec["state_z"], "action_z": [float(v) for v in z]})
                break
    return out


def _main_checkout() -> Path:
    """worktree 里 third_party 可能为空：返回主检出根（只读使用）。"""
    here = Path(__file__).resolve().parents[2]
    try:
        common = subprocess.run(["git", "-C", str(here), "rev-parse", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()
        common_p = Path(common)
        if not common_p.is_absolute():
            common_p = here / common_p
        return common_p.resolve().parent
    except (subprocess.CalledProcessError, FileNotFoundError):
        return here


def compute_swap_times(task: str, episode_id: int) -> int:
    """照抄 ButtonUnmaskSwap.__init__ 的 swap_times 逻辑：由元数据 seed／difficulty 与 torch 生成器决定。"""
    import ast
    import torch

    rel = Path("third_party/robomme_benchmark/src/robomme")
    roots = [Path(__file__).resolve().parents[2], _main_checkout()]
    root = next(r for r in roots if (r / rel / "robomme_env" / f"{task}.py").is_file())
    meta = json.loads((root / rel / "env_metadata/test" / f"record_dataset_{task}_metadata.json").read_text())
    rec = next(r for r in meta["records"] if int(r["episode"]) == int(episode_id))
    seed, difficulty = int(rec["seed"]), str(rec["difficulty"])
    # 从源码读 config_<difficulty> 的 swap_min／swap_max，不手抄数值。
    tree = ast.parse((root / rel / "robomme_env" / f"{task}.py").read_text())
    cfg = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == f"config_{difficulty}" for t in node.targets):
            cfg = ast.literal_eval(node.value)
    if cfg is None:
        raise ValueError(f"{task}.py 里找不到 config_{difficulty}")
    g = torch.Generator()
    g.manual_seed(seed)
    return int(torch.randint(cfg["swap_min"], cfg["swap_max"] + 1, (1,), generator=g).item())


# ---------------------------------------------------------------- CLI

def _episode_dirs(runs: Path) -> list[Path]:
    if any(runs.glob("infer_*.npz")):
        return [runs]
    return sorted(d for d in runs.iterdir() if d.is_dir() and any(d.glob("infer_*.npz")))


def _load_npzs(d: Path) -> dict[int, dict]:
    out = {}
    for p in sorted(d.glob("infer_*.npz")):
        with np.load(p) as z:
            rec = {k: z[k] for k in z.files}
        out[int(rec["infer_seq"])] = rec
    return out


def _episode_meta(d: Path, args) -> dict:
    meta = {}
    if (d / "episode.json").is_file():
        meta = json.loads((d / "episode.json").read_text())
    task = args.task or meta.get("task_name") or meta.get("task") or "ButtonUnmaskSwap"
    ep = args.episode if args.episode is not None else meta.get("episode_id")
    model = args.model or meta.get("model_kind") or meta.get("model") or d.name
    return {"task": task, "episode_id": ep, "model": model, "nonce": d.name}


def _key_infer_for(d: Path, specs: list[str]) -> int | None:
    """--key-infer 可给 "<k>"（全部局）或 "<局目录名>=<k>"（逐局）。"""
    val = None
    for s in specs:
        if "=" in s:
            name, k = s.split("=", 1)
            if name == d.name:
                return int(k)
        else:
            val = int(s)
    return val


def _fmt(v) -> str:
    return "na" if v is None else f"{v:.4f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="局目录，或其上级（下面可有多个局目录）")
    ap.add_argument("--out", type=Path, default=None, help="records 目录，默认 <runs>/records")
    ap.add_argument("--steps", default="all", help="去噪步：last|all|索引表")
    ap.add_argument("--layers", default="all", help="层：all|0-17|0,3,5")
    ap.add_argument("--heads", default="all", help="头：all|0-7|0,3")
    ap.add_argument("--query", choices=["action", "text"], default="action")
    ap.add_argument("--candidates", action="store_true", help="读 <局目录>/sim_eef_pose.json 列「揭开」候选")
    ap.add_argument("--key-infer", action="append", default=[], help="<k> 或 <局目录名>=<k>，可重复")
    ap.add_argument("--swap-times", type=int, default=None, help="交换次数 k；缺省读 sim_eef_pose.json，再缺省按元数据算")
    ap.add_argument("--task", default=None)
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)

    out = args.out or (args.runs / "records")
    dirs = _episode_dirs(args.runs)
    if not dirs:
        print(f"找不到含 infer_*.npz 的局目录：{args.runs}", flush=True)
        print("ATTN_AGG=FAIL episodes=0 pad_mass=nan stream_sum_err=nan", flush=True)
        return 1

    summary = {"query": args.query, "steps": args.steps, "layers": args.layers, "heads": args.heads,
               "episodes": [], "key_steps": []}
    max_pad = 0.0
    max_err = 0.0
    ok = True
    for d in dirs:
        meta = _episode_meta(d, args)
        npzs = _load_npzs(d)
        pose = json.loads((d / "sim_eef_pose.json").read_text()) if (d / "sim_eef_pose.json").is_file() else None
        streams = {}
        try:
            for seq, z in npzs.items():
                streams[seq] = split_streams(z, query=args.query, steps=args.steps,
                                             layers=args.layers, heads=args.heads)
        except AssertionError as e:
            print(f"分流断言失败 局={d.name} infer={seq}: {e}", flush=True)
            ok = False
            continue
        ep_pad = max(s["pad"] for s in streams.values())
        ep_err = max(s["sum_err"] for s in streams.values())
        max_pad, max_err = max(max_pad, ep_pad), max(max_err, ep_err)

        (out / d.name).mkdir(parents=True, exist_ok=True)
        (out / d.name / "streams.json").write_text(json.dumps(
            {**meta, "infers": {str(k): v for k, v in sorted(streams.items())}}, ensure_ascii=False))
        ep_rec = {**meta, "infers": len(streams), "pad_mass": ep_pad, "stream_sum_err": ep_err,
                  "mean_mass": {k: float(np.mean([s["mass"][k] for s in streams.values()]))
                                for k in next(iter(streams.values()))["mass"]}}

        if args.candidates:
            if pose is None:
                print(f"局 {d.name} 缺 sim_eef_pose.json，先跑 read_sim_eef_pose.py", flush=True)
                ok = False
            else:
                cands = find_candidates(npzs, pose)
                for c in cands:
                    zs = ",".join(f"{v:.3f}" for v in c["action_z"])
                    print(f"CANDIDATE nonce={d.name} infer={c['infer_seq']} step_idx={c['step_idx']} "
                          f"close_row={c['close_row']} rise_m={c['rise_m']:.3f} state_z={c['state_z']:.3f} "
                          f"action_z=[{zs}]", flush=True)
                print(f"KEY_CANDIDATES task={meta['task']} ep={meta['episode_id']} model={meta['model']} "
                      f"infers={[c['infer_seq'] for c in cands]}", flush=True)
                ep_rec["candidates"] = cands

        k_inf = _key_infer_for(d, args.key_infer)
        if k_inf is not None:
            if k_inf not in streams:
                print(f"局 {d.name} 没有 infer_seq={k_inf}", flush=True)
                ok = False
            else:
                st = streams[k_inf]
                k = args.swap_times
                if k is None and pose is not None and pose.get("swap_times") is not None:
                    k = int(pose["swap_times"])
                if k is None and st["has_motion"]:
                    k = compute_swap_times(meta["task"], int(meta["episode_id"]))
                sh = key_step_shares(st, k)
                print(f"KEY_STEP=task={meta['task']} ep={meta['episode_id']} model={meta['model']} "
                      f"infer={k_inf} vis_share={_fmt(sh['vis_share'])} vis_uniform={_fmt(sh['vis_uniform'])} "
                      f"swap_share={_fmt(sh['swap_share'])} swap_uniform={_fmt(sh['swap_uniform'])}", flush=True)
                summary["key_steps"].append({**meta, "infer_seq": k_inf, "step_idx": st["step_idx"],
                                             "swap_times": k, "has_motion": st["has_motion"],
                                             "frame": st["frame"], "motion": st["motion"],
                                             "mass": st["mass"], **sh})
        summary["episodes"].append(ep_rec)

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    ok = ok and max_pad <= PAD_TOL and max_err <= SUM_TOL and len(summary["episodes"]) == len(dirs)
    print(f"ATTN_AGG={'PASS' if ok else 'FAIL'} episodes={len(summary['episodes'])} "
          f"pad_mass={max_pad:.3e} stream_sum_err={max_err:.3e}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
