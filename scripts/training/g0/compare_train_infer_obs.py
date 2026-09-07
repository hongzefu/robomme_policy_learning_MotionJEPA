#!/usr/bin/env python3
"""L1–L5：训练侧数据表 vs 推理侧在线现算，逐层对拍（TIC 组 B/C；1 张 GPU，可选 sidecar 独占第二张卡）。

背景：训练时记忆（608 个 token）从离线表按行读；评估时机器人边跑边现算。本脚本按真实评估节奏
（首批整段 demo [0, es]、之后每批 16 帧）把原始 h5 帧喂进生产口径 policy，在每个决策时刻把推理端
现拼的东西抓出来，与训练端 `FrameSampDataset` / `transform_dataset` 同一时刻的样本逐层比：

  L1 输入键     原始 obs vs 建库 pkl；`_prepare_history` 八键 vs 训练 dataset；（sidecar 档）现算运动 token vs 表
  L2 预处理后   两侧各自过完整 transforms 链后的全部键（含 prompt token）
  L3 模型内部   `embed_prefix` 的 4 个叶、attn_mask、positions、每层 KV；记忆并列序↔重排的逆置换还原
  L4 整段 vs 缓存  同一权重同一输入同一 x_t/time 下，`compute_loss` 的整段前向 与 `sample_actions` 的前缀缓存+单步
  L5 最终动作   同一噪声下 10 步去噪动作逐位；确定性重跑；增广影响；checkpoint dtype 观察

三臂（沿用 `compare_siglip_replay.py` 口径，把「训练 f32 离线表 vs 推理 bf16 checkpoint」这一已知差异隔离掉）：
  S = 帧特征灌训练库真值行（**全部阻断判据都在 S 臂**），运动路按档位取表或真 sidecar
  A = 离线 f32 `SigLipTokenizer` 成批现算（只出帧级观察行）
  B = 在线 bf16 `policy._vision_encode` 现算（生产推理真身；只出帧级与动作级观察行）

L4 的两条判定行说明（2026-09-07 实测，阈值未改）：
  `VT_FULL_VS_CACHED` 的阈值 `rel_fro ≤ 1e-3` / `ulp_p99 ≤ 4` 由计划事先定死（假设「18 层每层 ≤1 ULP」）。
  生产 bf16 档实测 rel_fro ≈ 3.0e-3、ulp_p99 22–31、max_abs 0.0156（vt_rms ≈ 1.0），且前缀 KV 自身
  rel_fro 就已是 0.36–0.60%、`prefix_kv_bitexact=0/N`——差异在 `llm([prefix, suffix], …)` 与
  `llm([prefix, None], …)` 的前缀 pass 就已产生，与 suffix 段无关；结构三项（mask 子块 / suffix 行 /
  positions）全等，故不是语义错误。两次独立跑数字还不同（2.83e-3 / 3.04e-3），含 XLA autotune 的
  非确定性。**「阈值是否按实测重定」属放宽判据，须用户裁决，本脚本不自行改阈值、不改阻断性质。**
  `VT_FULL_VS_CACHED_F32` 是为该裁决提供依据的**观察行**：把参数树与 `embed_dtype` 一起升到 f32
  后在同一 obs / x_t / time 上重跑两路。f32 下差异若掉到 1e-5 量级，则 bf16 档的 3e-3 就是
  kernel 选择与归约序的数值来源；若 f32 下仍是 3e-3 量级，那是真问题。

判定行分「阻断」与「观察」两类；任一阻断 FAIL 非零退出，但**所有已算出的判定行先全部打印**。
`OBS_PROMPT` 是唯一的「FAIL 即停后续层」项——两侧 prompt token 不等时 L3–L5 没有比较意义，
但 L1 / L2 仍跑完全部 episode 后再打印。

用法（主进程 jax 必须在 GPU 上——CPU 上 PosEmb3D 4x4 表不逐位，P5 留档记过）：
  CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 UV_LINK_MODE=copy uv run --no-sync python \
    scripts/training/g0/compare_train_infer_obs.py --motion store --out v1-store/reports/tic-dev/obsmodel.json
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import gc
import json
import os
import pathlib
import pickle
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "training" / "tests"))
_V1 = pathlib.Path(os.environ.get("MMEVLA_V1_STORE", str(_REPO_ROOT / "v1-store")))
os.environ.setdefault("OPENPI_DATA_HOME", str(_V1 / "models"))   # SigLipTokenizer 要求绝对路径（不 expanduser）

import _common as C  # noqa: E402

KEYS8 = ("static_image_emb", "static_pos_emb", "static_state_emb", "static_mask",
         "motion_emb", "motion_pos", "motion_mask", "mem_order")
FRAME_ARMS = ("S", "A", "B")
MEM_PAIRS = (("S", "B"), ("S", "A"), ("A", "B"))
DEFAULT_EPISODES = "ButtonUnmaskSwap:3,VideoUnmask:3,VideoUnmaskSwap:5,VideoUnmaskSwap:31,VideoUnmaskSwap:3"
# L4 数值阈值（计划 2.4 事先定死，超阈值即 FAIL，不现场放宽）
THR_REL_FRO = 1e-3
THR_ULP_P99 = 4.0
MAX_STEPS = 1300              # examples/robomme/eval.py::Args.max_steps（与 eval_rhythm_gates.py 同值）


# ─────────────────────────── 通用小工具（派生自 compare_siglip_replay.py） ───────────────────────────


def load_episode_full(raw_dir: pathlib.Path, h5_file: str, raw_ep_idx: int, T: int, n_read: int):
    """读 h5 一集的前 n_read 帧（front_rgb / wrist_rgb / joint+gripper）与 setup/task_goal 原文。

    仍先核对整集 timestep 数与清单一致（n_read 只裁读取量，不放宽一致性校验）。
    """
    import h5py
    frames, wrists, states = [], [], []
    with h5py.File(raw_dir / h5_file, "r") as f:
        g = f[f"episode_{raw_ep_idx}"]
        ts_ids = sorted(int(k.split("_")[-1]) for k in g.keys() if k.startswith("timestep_"))
        if len(ts_ids) != T or ts_ids != list(range(T)):
            raise SystemExit(f"错误: {h5_file} episode_{raw_ep_idx} timesteps {len(ts_ids)} != 清单 {T}")
        tg = g["setup"]["task_goal"][()]
        task_goal = (tg[0] if getattr(tg, "shape", ()) else tg)
        task_goal = task_goal.decode() if isinstance(task_goal, bytes) else str(task_goal)
        for t in ts_ids[:n_read]:
            ts = g[f"timestep_{t}"]
            img = ts["obs"]["front_rgb"][()]
            wr = ts["obs"]["wrist_rgb"][()]
            if img.shape != (256, 256, 3) or img.dtype != np.uint8 or wr.shape != (256, 256, 3) or wr.dtype != np.uint8:
                raise SystemExit(f"错误: 帧形制 front={img.shape} {img.dtype} wrist={wr.shape} {wr.dtype}")
            frames.append(img)
            wrists.append(wr)
            joint = ts["obs"]["joint_state"][()]
            grip = ts["obs"]["gripper_state"][()]
            states.append(np.concatenate([joint, grip[:1]], axis=0, dtype=np.float32))
    return np.stack(frames)[:, None], np.stack(wrists), np.stack(states), task_goal


def _bytes_equal(a, b) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a.view(np.uint8), b.view(np.uint8))


def _bf16_ulp(x32: np.ndarray) -> np.ndarray:
    """bf16 在 |x| 处的 ULP：2^(floor(log2|x|) - 7)。"""
    ax = np.abs(x32).astype(np.float64)
    e = np.floor(np.log2(np.where(ax > 0, ax, 1.0)))
    return np.ldexp(1.0, (e - 7).astype(np.int32))


class DiffAcc:
    """两组数组的数值差累计。ULP 以两侧绝对值较大者为基（零元素不计入 ULP 分位）。"""

    def __init__(self):
        self.n = 0
        self.max_abs = 0.0
        self.sum_abs = 0.0
        self.cnt = 0
        self.sum_d2 = 0.0
        self.sum_a2 = 0.0
        self.nonzero = 0
        self.cos: list[float] = []
        self.ulp: list[np.ndarray] = []

    def add(self, a, b):
        a32 = np.asarray(a).astype(np.float32)
        b32 = np.asarray(b).astype(np.float32)
        d = b32 - a32
        self.n += 1
        self.max_abs = max(self.max_abs, float(np.abs(d).max()))
        self.sum_abs += float(np.abs(d).sum())
        self.cnt += d.size
        self.nonzero += int((d != 0).sum())
        self.sum_d2 += float((d.astype(np.float64) ** 2).sum())
        self.sum_a2 += float((a32.astype(np.float64) ** 2).sum())
        na = np.linalg.norm(a32, axis=-1)
        nb = np.linalg.norm(b32, axis=-1)
        self.cos.extend(((a32 * b32).sum(-1) / (na * nb + 1e-12)).ravel().tolist())
        base = np.maximum(np.abs(a32), np.abs(b32))
        sel = base > 0
        if np.any(sel):
            self.ulp.append((np.abs(d)[sel] / _bf16_ulp(base[sel])).ravel())

    def summary(self) -> dict:
        ulp = np.concatenate(self.ulp) if self.ulp else np.zeros(1)
        return {"n": self.n, "max_abs": self.max_abs, "mean_abs": self.sum_abs / max(self.cnt, 1),
                "rel_fro": float(np.sqrt(self.sum_d2 / max(self.sum_a2, 1e-30))),
                "cos_min": float(min(self.cos)) if self.cos else None,
                "cos_mean": float(np.mean(self.cos)) if self.cos else None,
                "ulp_p50": float(np.percentile(ulp, 50)), "ulp_p99": float(np.percentile(ulp, 99)),
                "ulp_max": float(ulp.max()), "frac_nonzero": self.nonzero / max(self.cnt, 1)}

    def line(self, tag: str, unit: str = "frames") -> str:
        s = self.summary()
        return (f"{tag} {unit}={s['n']} max_abs={s['max_abs']:.4g} mean_abs={s['mean_abs']:.4g} rel_fro={s['rel_fro']:.4g} "
                f"cos_min={s['cos_min']:.6f} cos_mean={s['cos_mean']:.6f} ulp_p50={s['ulp_p50']:.3g} "
                f"ulp_p99={s['ulp_p99']:.3g} ulp_max={s['ulp_max']:.3g} frac_nonzero={s['frac_nonzero']:.3f}")


def _mean_str(xs, fmt: str = ".4g") -> str:
    """空序列（如 --noise-seeds 只给一个 seed 时的 NOISE_S 参照）输出 n/a，不输出 nan。"""
    return format(float(np.mean(xs)), fmt) if len(xs) else "n/a"


def _ratio_str(num, den, fmt: str = ".4g") -> str:
    if not len(num) or not len(den) or float(np.mean(den)) == 0.0:
        return "n/a"
    return format(float(np.mean(num)) / float(np.mean(den)), fmt)


def _rms(x) -> float:
    x = np.asarray(x, np.float64)
    return float(np.sqrt(np.mean(x ** 2)))


def _flatten(d, prefix: str = "") -> dict:
    """把嵌套 dict 展平成 'a/b' → 值；None 值保留（键集比较需要）。"""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "/"))
        else:
            out[key] = v
    return out


class Verdict:
    """判定行收集器：阻断行计入 blocking 分母，观察行只记录。"""

    def __init__(self):
        self.rows: list[dict] = []

    def block(self, ok: bool, text: str) -> None:
        self.rows.append({"kind": "block", "ok": bool(ok), "text": text})

    def observe(self, text: str) -> None:
        self.rows.append({"kind": "observe", "ok": None, "text": text})

    @property
    def block_total(self) -> int:
        return sum(1 for r in self.rows if r["kind"] == "block")

    @property
    def block_pass(self) -> int:
        return sum(1 for r in self.rows if r["kind"] == "block" and r["ok"])

    @property
    def n_observe(self) -> int:
        return sum(1 for r in self.rows if r["kind"] == "observe")

    @property
    def all_ok(self) -> bool:
        return self.block_total == self.block_pass

    def texts(self) -> list[str]:
        return [r["text"] for r in self.rows]


# ─────────────────────────── 决策时刻清单（独立生成，与 dataset 逐点核对） ───────────────────────────


def resolve_episodes(manifest: dict, spec_str: str) -> tuple[list[dict], str]:
    """解析 --episodes；任一项缺失 / (h5,raw) 重复 / es 取值重复时，改按库 manifest 每个 es 取 T 最长的一集。"""
    eps = manifest["episodes"]
    picked: list[dict] = []
    reason = ""
    seen_pair: set[tuple[str, int]] = set()
    seen_es: set[int] = set()
    for spec in spec_str.split(","):
        if ":" not in spec:
            reason = f"条目 {spec!r} 不是 <task>:<raw_ep_idx> 形式"
            break
        task, raw = spec.rsplit(":", 1)
        h5 = f"record_dataset_{task}.h5"
        cand = [e for e in eps if e["h5_file"] == h5 and int(e["raw_ep_idx"]) == int(raw)]
        if len(cand) != 1:
            reason = f"条目 {spec!r} 在库 manifest 中命中 {len(cand)} 条"
            break
        e = cand[0]
        if (h5, int(raw)) in seen_pair:
            reason = f"条目 {spec!r} 重复"
            break
        if int(e["exec_start_idx"]) in seen_es:
            reason = f"条目 {spec!r} 的 exec_start_idx={e['exec_start_idx']} 与前面条目重复"
            break
        seen_pair.add((h5, int(raw)))
        seen_es.add(int(e["exec_start_idx"]))
        picked.append(e)
    if reason:
        best: dict[int, dict] = {}
        for e in eps:
            es = int(e["exec_start_idx"])
            if es not in best or int(e["num_timesteps"]) > int(best[es]["num_timesteps"]):
                best[es] = e
        picked = [best[es] for es in sorted(best)]
        return picked, f"manifest(es 取值各取 T 最长；回退原因：{reason})"
    return picked, "args"


def expected_taus(es: int, t_env: int, max_steps: int = MAX_STEPS) -> list[int]:
    """决策时刻清单（与 `scripts/training/tests/eval_rhythm_gates.py::expected_taus` 同一公式，不 import 它）。

    不 import 的原因：那个模块顶层做 `os.environ.setdefault("JAX_PLATFORMS", "cpu")`，
    在本脚本里会把主进程 jax 打到 CPU 上、直接撞 GPU 硬闸。

    真实控制流是 `examples/robomme/eval.py::EpisodeEvaluator.eval_each_episode`：推理发生在循环体开头
    `count = 16j` 处，循环体开头出现过的 count 值是 0..min(C, max_steps+1)−1，其中 C = t_env − 1 − es
    是环境步数（count = C 那一次在体尾 break，不再进入下一轮体首）。
    注意与 `motion_gates_online._drive` 的 `t + 16 < T` 不同：后者给 ⌊C/16⌋+1 个点，
    在 C 恰为 16 的倍数时会多出一个训练表里不存在的时刻（本轮五集 C%16 = 10/12/11/4/1，不触发）。
    """
    c_env = t_env - 1 - es
    last_count = min(c_env, max_steps + 1) - 1
    return [es + 16 * j for j in range(last_count // 16 + 1)] if last_count >= 0 else []


def build_points(picked: list[dict], max_points: int) -> list[dict]:
    """按真实评估节奏推每集的决策时刻，并留 C % 16 诊断字段。"""
    out = []
    for e in picked:
        es = int(e["exec_start_idx"])
        T = int(e["num_timesteps"])
        taus = expected_taus(es, T)
        c_env = T - 1 - es
        if max_points > 0:
            taus = taus[:max_points]
        out.append({"h5_file": e["h5_file"], "raw_ep_idx": int(e["raw_ep_idx"]),
                    "g": int(e["global_episode_idx"]), "es": es, "T": T, "c_env": c_env,
                    "c_mod16": c_env % 16, "row_base": int(e["total_sample_offset"]), "taus": taus})
    return out


# ─────────────────────────── 主流程 ───────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", default=str(_V1 / "datasets/4task-motion-400ep"), help="对拍用的库根（含 framesamp/ 与 motion/）")
    ap.add_argument("--ckpt", default=str(_V1 / "train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999"),
                    help="checkpoint 目录（含 params/ 与 assets/）")
    ap.add_argument("--train-config", default="mme_vla_suite_b128", help="生产训练配置条目（两侧同用）")
    ap.add_argument("--episodes", default=DEFAULT_EPISODES, help="逗号分隔的 <task>:<raw_ep_idx>，需覆盖 5 个 es 取值")
    ap.add_argument("--motion", choices=("store", "sidecar"), default="store",
                    help="运动路来源：store=从 MotionStore 查表；sidecar=S 臂用真 MotionEncoderClient 现算")
    ap.add_argument("--motion-gpu", default="1", help="sidecar 子进程的 CUDA_VISIBLE_DEVICES（物理卡号）")
    ap.add_argument("--noise-seeds", default="0,1,2")
    ap.add_argument("--max-points", type=int, default=0, help="开发用：每集只跑前 N 个决策时刻（0 = 全部）")
    ap.add_argument("--f32-diag-points-per-episode", type=int, default=1,
                    help="VT_FULL_VS_CACHED_F32 观察行：每集取前 N 个决策点做 f32 重跑（0 = 关闭）")
    ap.add_argument("--out", default=str(_V1 / "reports/tic/compare_train_infer_obs.json"), help="输出 json 路径")
    args = ap.parse_args()

    os.chdir(_REPO_ROOT)      # train_config.data.assets.assets_dir 是仓库相对路径，必须从仓库根解析
    lib = pathlib.Path(args.lib).resolve()
    ckpt_dir = pathlib.Path(args.ckpt).resolve()
    run_root = ckpt_dir.parent
    noise_seeds = [int(s) for s in args.noise_seeds.split(",")]
    os.environ["MMEVLA_MOTION_STORE"] = str(lib / "motion")     # 快照 store_path 记的是 40ep，必须显式覆盖

    import einops
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import omegaconf
    from openpi.models import model as _model
    from openpi.training.data_loader import transform_dataset
    from mme_vla_suite.training import config as _config
    from mme_vla_suite.policies import policy_config as _policy_config
    from mme_vla_suite.datastore import motion_store as ms
    from mme_vla_suite.datastore.framesamp_store import StoreMeta
    from mme_vla_suite.training.dataloader import _create_framesamp_dataset, _motion_gates
    from mme_vla_suite.dataset_builder.siglip_tokenizer import SigLipTokenizer
    from mme_vla_suite.models.integration.history_observation import HistAugObservation, preprocess_observation
    from mme_vla_suite.models.integration.history_pi0 import make_attn_mask

    if jax.default_backend() != "gpu":
        raise SystemExit(f"错误: 主进程 jax 后端 {jax.default_backend()} != gpu（PosEmb3D 4x4 表 CPU/GPU 不逐位，P5 留档）")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    V = Verdict()
    rec: dict = {"argv": sys.argv, "lib": str(lib), "ckpt": str(ckpt_dir), "train_config": args.train_config,
                 "motion": args.motion, "noise_seeds": noise_seeds, "max_points": args.max_points}

    # ── 生产口径 policy（B 臂编码器 + transforms + norm_stats 全部来自 checkpoint）──────────────────
    t0 = time.perf_counter()
    train_config = _config.get_config(args.train_config)
    policy = _policy_config.create_trained_policy(train_config, ckpt_dir, motion_stub=True)
    stub_client = policy._motion_client
    stub_client.close()                                     # 只借它过构造，随后按档位换成查表闭包或真 sidecar
    enc_B = policy._vision_encode
    model = policy._model
    print(f"[tic] policy 构造 {time.perf_counter() - t0:.1f}s config={args.train_config} ckpt={ckpt_dir} "
          f"use_quantiles={policy.use_quantiles} action_horizon={model.action_horizon} action_dim={model.action_dim}")

    # ── 训练侧真口径 dataset：data_config 由生产配置 create，raw / transformed 两层都建 ────────────
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    hc = omegaconf.OmegaConf.load(run_root / "history_config.resolved.yaml")
    ds_raw = _create_framesamp_dataset(str(lib / "framesamp"), data_config, hc, int(model.action_horizon))
    ds_tf = transform_dataset(ds_raw, data_config)
    fmeta = StoreMeta.load(str(lib / "framesamp"))
    motion_root = _motion_gates(hc, fmeta)
    mmeta = ms.MotionMeta.load(motion_root)
    mstore = ms.MotionStore(motion_root, meta=mmeta)
    fstore = ds_raw._ensure_store()
    manifest = json.load(open(lib / "meta" / "episode_manifest.json", encoding="utf-8"))
    raw_dir = pathlib.Path(manifest["raw_dir"])
    source_root = pathlib.Path(ds_raw._source_root)
    ns_actions_std = float(np.mean(np.asarray(data_config.norm_stats["actions"].std)))

    # ── A 臂（离线 f32 SigLIP，成批喂）──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    tokA = SigLipTokenizer()
    enc_A = jax.jit(tokA.__call__)
    print(f"[tic] A 臂构造 {time.perf_counter() - t0:.1f}s "
          f"dtypes={sorted({str(x.dtype) for x in jax.tree.leaves(nnx.state(tokA.img))})}")

    # ── 决策时刻清单：脚本独立生成，随后与 FrameSampDataset 的 (g,t) 存在性逐点核对 ───────────────
    picked, ep_source = resolve_episodes(manifest, args.episodes)
    eps = build_points(picked, args.max_points)
    n_points = sum(len(e["taus"]) for e in eps)
    entries = ds_raw._motion_entries
    n_windows = 0
    for e in eps:
        rows_last, _ = ms.visible_motion_rows(entries[e["g"]], e["taus"][-1])
        n_windows += int(len(rows_last))
        for t in e["taus"]:
            hit = np.flatnonzero((ds_raw._epis_of == e["g"]) & (ds_raw._step_of == t))
            if len(hit) != 1:
                raise SystemExit(f"错误: 决策时刻 (g={e['g']}, t={t}) 在 FrameSampDataset 中命中 {len(hit)} 条样本")
    print(f"POINTS episodes={len(eps)} points={n_points} windows={n_windows} "
          f"c_mod16={[e['c_mod16'] for e in eps]} max_steps={MAX_STEPS}")
    print(f"EPISODES source={ep_source} " + " ".join(
        f"{e['h5_file'].replace('record_dataset_', '').replace('.h5', '')}:{e['raw_ep_idx']}"
        f"(g={e['g']},es={e['es']},T={e['T']},pts={len(e['taus'])})" for e in eps))
    rec["points"] = {"episodes": len(eps), "points": n_points, "windows": n_windows, "source": ep_source,
                     "episodes_detail": eps}

    # ── 模型侧 jit 函数（冻结一次 state，与 nnx_utils.module_jit 同法，避免 nnx.jit 的额外开销）──────
    gdef, gstate = nnx.split(model)

    def _obs_of(inputs_np: dict):
        return HistAugObservation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs_np))

    @jax.jit
    def f_prefix(state, obs):
        """L3：preprocess(train=False) → embed_prefix → attn_mask / positions → 前缀 pass 取 KV；另出并列序记忆。"""
        m = nnx.merge(gdef, state)
        o = preprocess_observation(None, obs, train=False)
        tok, mask, ar, na = m.embed_prefix(o)
        attn = make_attn_mask(mask, ar, na)
        pos = jnp.cumsum(mask, axis=1) - 1
        _, kv = m.PaliGemma.llm([tok, None], mask=attn, positions=pos)
        par_tok, _, _ = m.mem_encoder(o.static_image_emb, o.static_pos_emb, o.static_state_emb,
                                      motion_emb=o.motion_emb, motion_pos=o.motion_pos, motion_mask=o.motion_mask)
        par_mask = jnp.concatenate([o.static_mask, o.motion_mask], axis=1)
        mem_tok, mem_mask, _, _ = m.embed_memory(o)
        return {"tokens": tok, "mask": mask, "ar": ar, "na": na, "attn": attn, "pos": pos,
                "k": kv[0], "v": kv[1], "par_tok": par_tok, "par_mask": par_mask,
                "mem_tok": mem_tok, "mem_mask": mem_mask,
                "txt_len": jnp.int32(o.tokenized_prompt.shape[1])}

    def _mk_l4(gdef_):
        """按给定 graphdef 生成 L4 的两路 jit 函数（bf16 生产权重与 f32 诊断权重各一套）。"""

        @jax.jit
        def f_full(state, obs, x_t, tstep):
            """L4 训练路径：照抄 compute_loss 主体（prefix+suffix 一次前向）。"""
            m = nnx.merge(gdef_, state)
            o = preprocess_observation(None, obs, train=False)
            pt, pm, par_, pna = m.embed_prefix(o)
            st, sm, sar, sna, ad = m.embed_suffix(o, x_t, tstep)
            input_mask = jnp.concatenate([pm, sm], axis=1)
            ar_mask = jnp.concatenate([par_, sar], axis=0)
            na_mask = jnp.concatenate([pna, sna], axis=0)
            attn = make_attn_mask(input_mask, ar_mask, na_mask)
            pos = jnp.cumsum(input_mask, axis=1) - 1
            (_, so), kv = m.PaliGemma.llm([pt, st], mask=attn, positions=pos, adarms_cond=[None, ad])
            v_t = m.action_out_proj(so[:, -m.action_horizon:])
            return {"v_t": v_t, "attn": attn, "pos": pos, "prefix_len": jnp.int32(pm.shape[1]),
                    "pmask": pm, "k": kv[0], "v": kv[1]}

        @jax.jit
        def f_cached(state, obs, x_t, tstep):
            """L4 推理路径：照抄 sample_actions 的前缀 pass + 一次 step（不走 while_loop）。"""
            m = nnx.merge(gdef_, state)
            o = preprocess_observation(None, obs, train=False)
            pt, pm, par_, pna = m.embed_prefix(o)
            pattn = make_attn_mask(pm, par_, pna)
            ppos = jnp.cumsum(pm, axis=1) - 1
            _, kv = m.PaliGemma.llm([pt, None], mask=pattn, positions=ppos)
            st, sm, sar, _, ad = m.embed_suffix(o, x_t, tstep)
            sattn = make_attn_mask(sm, sar)
            prefix_attn = einops.repeat(pm, "b p -> b s p", s=st.shape[1])
            full = jnp.concatenate([prefix_attn, sattn], axis=-1)
            spos = jnp.sum(pm, axis=-1)[:, None] + jnp.cumsum(sm, axis=-1) - 1
            (_, so), _ = m.PaliGemma.llm([None, st], mask=full, positions=spos, kv_cache=kv, adarms_cond=[None, ad])
            v_t = m.action_out_proj(so[:, -m.action_horizon:])
            return {"v_t": v_t, "prefix_attn": pattn, "full_attn": full, "spos": spos,
                    "pmask": pm, "k": kv[0], "v": kv[1]}

        return f_full, f_cached

    f_full, f_cached = _mk_l4(gdef)

    @jax.jit
    def f_aug(obs, rng):
        """L5 增广观察：同一 obs 走 preprocess(train=True)（图像增广）。记忆键应原样透传。"""
        return preprocess_observation(rng, obs, train=True)

    # ── L2 静态部分：两侧 transforms 链结构与 RepackTransform 丢弃键 ────────────────────────────────
    train_chain = [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs,
                   "Normalize", *data_config.model_transforms.inputs]
    train_names = [t if isinstance(t, str) else type(t).__name__ for t in train_chain]
    infer_names = [type(t).__name__ for t in policy._input_transform.transforms]
    sm_ = difflib.SequenceMatcher(a=train_names, b=infer_names, autojunk=False)
    train_only, infer_only = [], []
    for tag, i1, i2, j1, j2 in sm_.get_opcodes():
        if tag in ("delete", "replace"):
            train_only += train_names[i1:i2]
        if tag in ("insert", "replace"):
            infer_only += infer_names[j1:j2]
    repack_struct = data_config.repack_transforms.inputs[0].structure
    repack_sources = set(repack_struct.values())
    # tokenize 变换与「是否做 lower」按实测取，不写死类名（生产链用的是 mme_vla_suite 自带的
    # TokenizePromptWithState + PaligemmaTokenizer，与 openpi 同名类不是同一个）
    _tok_infer = [t for t in policy._input_transform.transforms if hasattr(t, "tokenizer")]
    _tok_train = [t for t in data_config.model_transforms.inputs if hasattr(t, "tokenizer")]
    if len(_tok_infer) != 1 or len(_tok_train) != 1:
        raise SystemExit(f"错误: tokenize 变换数 infer={len(_tok_infer)} train={len(_tok_train)}，期望各 1 个")
    if type(_tok_infer[0]) is not type(_tok_train[0]) or type(_tok_infer[0].tokenizer) is not type(_tok_train[0].tokenizer):
        raise SystemExit("错误: 两侧 tokenize 变换/分词器类型不同")
    _probe_up = _tok_infer[0].tokenizer.tokenize("Press The Button")[0]
    _probe_lo = _tok_infer[0].tokenizer.tokenize("press the button")[0]
    _does_lower = bool(np.array_equal(np.asarray(_probe_up), np.asarray(_probe_lo)))
    tok_desc = (f"{type(_tok_infer[0]).__name__}/{type(_tok_infer[0].tokenizer).__name__}"
                f"(strip,underscore,newline;lower={'yes' if _does_lower else 'no'};"
                f"discrete_state_input={getattr(_tok_infer[0], 'discrete_state_input', None)})")

    # ── 运动路句柄：store 档三臂都查表；sidecar 档 S 臂用真 MotionEncoderClient，A/B 仍查表（省 3 倍编码）──
    client = None
    if args.motion == "sidecar":
        from mme_vla_suite.policies.motion_client import MotionEncoderClient
        t0 = time.perf_counter()
        client = MotionEncoderClient(online_gpu=args.motion_gpu,
                                     store_provenance={"vae": mmeta.provenance.get("vae"),
                                                       "encoder": mmeta.provenance.get("encoder")},
                                     expected_ckpt_sha256=mmeta.provenance["encoder"]["checkpoint_sha256"])
        print(f"[tic] sidecar 就绪 {time.perf_counter() - t0:.1f}s gpu={args.motion_gpu} "
              f"warmup={client.provenance.get('warmup_s')}")

    AH, AD = int(model.action_horizon), int(model.action_dim)
    NOISE = {s: jax.random.normal(jax.random.key(s), (1, AH, AD), dtype=jnp.float32) for s in noise_seeds}

    # ── 累加器 ─────────────────────────────────────────────────────────────────────────────────
    raw_mis = {"image": 0, "wrist_image": 0, "state": 0, "prompt_text": 0}
    trainset_key_mis = {k: 0 for k in KEYS8}
    obs_key_mis: dict[str, int] = {}
    obs_train_only: set[str] = set()
    obs_infer_only: set[str] = set()
    prompt_tok_mis = 0
    prompt_text_eq = 0
    prompt_samples: list[dict] = []
    noop_inject_ok = True
    repack_dropped: set[str] = set()
    repack_reach_model = 0
    frame_acc = {p: DiffAcc() for p in MEM_PAIRS}
    motion_sidecar_cmp = [0, 0]                 # [compared, mismatches]
    prefix_shape_ok = True
    prefix_shape_info: dict = {}
    prefix_leaf_mis = 0
    pos_mis = 0
    attn_mis = 0
    invperm_tok_mis = 0
    invperm_mask_mis = 0
    invperm_bad = 0
    kv_k_mis = 0
    kv_v_mis = 0
    kv_layers = 0
    scale_ratios: list[float] = []
    l4_struct = {"prefix_mask_block": 0, "suffix_rows": 0, "suffix_positions": 0,
                 "prefix_kv_bitexact": 0, "prefix_kv_pad": 0, "n": 0}
    l4_vt = DiffAcc()
    l4_vt_rms: list[float] = []
    f32_cache: list[dict] = []          # VT_FULL_VS_CACHED_F32 用：每集前 N 个点的 (obs, x_t, time)
    l4_kv = {"n": 0, "max_abs": 0.0, "sum_d2": 0.0, "sum_a2": 0.0, "ulp_p99": 0.0, "ulp_max": 0.0}
    act_ti_mis = 0
    act_det_ok = True
    act_sb = {"norm": [], "unnorm": []}
    act_sb_max = 0.0
    noise_ref = {"norm": [], "unnorm": []}
    aug_mem_unchanged = True
    aug_img_changed = 0
    aug_points = 0
    aug_act_rms: list[float] = []
    per_point: list[dict] = []
    per_episode: list[dict] = []
    prompt_fail = False
    last_infer_inputs: dict | None = None
    n_done = 0
    ckpt_dtype_line = None
    f32_line = None

    try:
        for ep in eps:
            g, es, T, row_base = ep["g"], ep["es"], ep["T"], ep["row_base"]
            taus = ep["taus"]
            entry = entries[g]
            rows_all, f_all = ms.visible_motion_rows(entry, T - 1)
            f2row = {int(f): int(r) for r, f in zip(rows_all.tolist(), f_all.tolist())}

            def motion_lookup(window, start_frame, _f2row=f2row):
                return np.asarray(mstore.rows(np.asarray([_f2row[int(start_frame)]], dtype=np.int64))[0],
                                  dtype=np.float32)

            n_read = min(T, taus[-1] + 1)
            frames, wrists, states, task_goal = load_episode_full(raw_dir, ep["h5_file"], ep["raw_ep_idx"], T, n_read)

            def make_mem(enc, mfn):
                policy._vision_encode = enc
                policy._motion_client = mfn
                policy._prepare_mem_buffer()
                return policy.mem_buffer

            mems = {"S": make_mem(enc_A, client if client is not None else motion_lookup),
                    "A": make_mem(enc_A, motion_lookup),
                    "B": make_mem(enc_B, motion_lookup)}       # B 最后建，policy._vision_encode 停在 enc_B
            if policy._vision_encode is not enc_B:
                raise RuntimeError("policy._vision_encode 未停在 B 臂编码器")
            policy.mem_buffer = mems["B"]
            policy.step_idx = -1
            policy.exec_start_idx = 0
            tb = time.perf_counter()

            def feed(lo, hi):
                # check() 内会把 policy.mem_buffer 临时切到 S / B 臂，喂帧前强制回到 B 臂：
                # policy.add_buffer 写的是 self.mem_buffer，停在 S 臂会让这一批被写两遍（step 已在缓冲 → raise）
                policy.mem_buffer = mems["B"]
                prev = policy.step_idx
                policy.add_buffer({"images": frames[lo:hi], "state": states[lo:hi],
                                   "exec_start_idx": es if lo == 0 else 0})      # B 臂，成批
                sl = list(range(prev + 1, prev + 1 + (hi - lo)))
                esx = policy.exec_start_idx
                for name in ("S", "A"):
                    mems[name].add_buffer(frames[lo:hi], states[lo:hi], sl, exec_start_idx=esx)
                rows = fstore.read_image_rows(np.asarray([row_base + s for s in sl], dtype=np.int64))
                for i, s in enumerate(sl):                                       # S 臂帧特征改写为训练库真值行
                    mems["S"]._history_feats[s]["image_emb_4x4"] = np.ascontiguousarray(rows[i][None])
                return sl

            def check(t: int, new_steps: list[int], is_last: bool, ep_pt_idx: int):
                nonlocal prompt_tok_mis, prompt_text_eq, noop_inject_ok, repack_reach_model
                nonlocal prefix_shape_ok, prefix_leaf_mis, pos_mis, attn_mis
                nonlocal invperm_tok_mis, invperm_mask_mis, invperm_bad, kv_k_mis, kv_v_mis, kv_layers
                nonlocal act_ti_mis, act_det_ok, act_sb_max, aug_mem_unchanged, aug_img_changed, aug_points
                nonlocal prompt_fail, last_infer_inputs, n_done
                if policy.step_idx != t:
                    raise RuntimeError(f"policy.step_idx {policy.step_idx} != 决策时刻 {t}")
                r: dict = {"g": g, "t": t, "new_frames": len(new_steps)}
                hit = np.flatnonzero((ds_raw._epis_of == g) & (ds_raw._step_of == t))
                if len(hit) != 1:
                    raise RuntimeError(f"训练样本定位失败 g={g} t={t}: {hit}")
                idx = int(hit[0])

                # ── L1a 原始 obs vs 建库 pkl ────────────────────────────────────────────────
                with open(source_root / "data" / f"{idx}.pkl", "rb") as fh:
                    pkl = pickle.load(fh)
                if not _bytes_equal(pkl["image"], frames[t, 0]):
                    raw_mis["image"] += 1
                if not _bytes_equal(pkl["wrist_image"], wrists[t]):
                    raw_mis["wrist_image"] += 1
                if not _bytes_equal(pkl["state"], states[t]):
                    raw_mis["state"] += 1
                if pkl["prompt"] != task_goal:
                    raw_mis["prompt_text"] += 1

                # ── L1b 帧级三臂数值差（观察）+ S 臂运动路 vs 表（sidecar 档阻断）─────────────
                for s in new_steps:
                    e_arm = {n: np.asarray(mems[n]._history_feats[s]["image_emb_4x4"])[0] for n in FRAME_ARMS}
                    for p in MEM_PAIRS:
                        frame_acc[p].add(e_arm[p[0]], e_arm[p[1]])
                if client is not None:
                    for f in mems["S"].visible_motion_frames(t):
                        if f in checked_motion:
                            continue
                        checked_motion.add(f)
                        on = np.asarray(mems["S"]._history_feats_motion[f], np.float32)
                        off = np.asarray(mstore.rows(np.asarray([f2row[int(f)]], np.int64))[0], np.float32)
                        motion_sidecar_cmp[0] += 1
                        if not _bytes_equal(on, off):
                            motion_sidecar_cmp[1] += 1

                # ── L1c 装配八键 vs 训练 dataset（S 臂，阻断）───────────────────────────────
                element = {"observation/image": frames[t, 0], "observation/wrist_image": wrists[t],
                           "observation/state": states[t], "prompt": task_goal}
                policy.mem_buffer = mems["S"]
                assembled_S = policy._prepare_history(dict(element))
                sample_raw = ds_raw[idx]
                key_ok = {}
                for k in KEYS8:
                    ok = _bytes_equal(assembled_S[k], np.asarray(sample_raw[k]))
                    key_ok[k] = ok
                    if not ok:
                        trainset_key_mis[k] += 1
                r["trainset_key_ok_S"] = key_ok

                # ── L2 预处理后逐键 ────────────────────────────────────────────────────────
                infer_inputs = policy._input_transform(dict(assembled_S))
                train_inputs = ds_tf[idx]
                fi = _flatten(infer_inputs)
                ft = _flatten(train_inputs)
                for k in sorted(set(ft) - set(fi)):
                    obs_train_only.add(k)
                for k in sorted(set(fi) - set(ft)):
                    obs_infer_only.add(k)
                sha_pair = {}
                for k in sorted(set(ft) & set(fi)):
                    a, b = ft[k], fi[k]
                    if a is None and b is None:
                        continue
                    if a is None or b is None:
                        obs_key_mis[k] = obs_key_mis.get(k, 0) + 1
                        continue
                    sa = C.leaf_sha256(np.asarray(a))
                    sb = C.leaf_sha256(np.asarray(b))
                    sha_pair[k] = [sa[:16], sb[:16]]
                    if sa != sb:
                        obs_key_mis[k] = obs_key_mis.get(k, 0) + 1
                r["obs_key_sha16"] = sha_pair
                # RepackTransform 丢弃的键不得抵达模型
                dropped = set(pkl.keys()) - repack_sources
                repack_dropped.update(dropped)
                repack_reach_model += len(dropped & set(ft))
                # InjectDefaultPrompt(None) 在推理链上是 noop 的实证
                inj = policy._input_transform.transforms[0]
                before = dict(assembled_S)
                after = inj(dict(assembled_S))
                if set(before) != set(after) or any(
                        not (before[k] is None and after[k] is None) and not _bytes_equal(np.asarray(before[k]), np.asarray(after[k]))
                        for k in before if not isinstance(before[k], str)):
                    noop_inject_ok = False

                # ── L2 prompt 专项 ─────────────────────────────────────────────────────────
                tok_eq = _bytes_equal(ft["tokenized_prompt"], fi["tokenized_prompt"]) and \
                    _bytes_equal(ft["tokenized_prompt_mask"], fi["tokenized_prompt_mask"])
                if pkl["prompt"] == task_goal:
                    prompt_text_eq += 1
                if not tok_eq:
                    prompt_tok_mis += 1
                    prompt_fail = True
                    if len(prompt_samples) < 4:
                        prompt_samples.append({"g": g, "t": t, "train_text": pkl["prompt"], "infer_text": task_goal,
                                               "train_tok": np.asarray(ft["tokenized_prompt"]).tolist(),
                                               "infer_tok": np.asarray(fi["tokenized_prompt"]).tolist()})
                elif not prompt_samples:
                    prompt_samples.append({"g": g, "t": t, "train_text": pkl["prompt"], "infer_text": task_goal,
                                           "equal": True})
                n_done += 1
                per_point.append(r)
                if prompt_fail:
                    print(f"[tic] t={t:4d} L1/L2 完成（OBS_PROMPT 已 FAIL，跳过 L3–L5）")
                    return

                # ── L3 模型内部（两侧 obs 各走一次 preprocess(train=False) + embed_prefix + 前缀 pass）──
                obs_T = _obs_of(train_inputs)
                obs_I = _obs_of(infer_inputs)
                oT = jax.tree.map(lambda x: np.asarray(x), jax.block_until_ready(f_prefix(gstate, obs_T)))
                oI = jax.tree.map(lambda x: np.asarray(x), jax.block_until_ready(f_prefix(gstate, obs_I)))
                mem_len = int(oI["mem_tok"].shape[1])
                total_len = int(oI["tokens"].shape[1])
                txt_len = int(oI["txt_len"])
                img_len = total_len - mem_len - txt_len
                n_img = len(infer_inputs["image"])
                want_mem = int(hc.budget) + int(hc.motion.budget)
                shape_ok = (mem_len == want_mem and txt_len == int(model.config.max_token_len)
                            and img_len > 0 and img_len % n_img == 0)
                if not shape_ok:
                    prefix_shape_ok = False
                prefix_shape_info.update({"len": total_len, "mem": mem_len, "img": img_len, "txt": txt_len,
                                          "img_per_view": img_len // n_img if n_img else None, "views": n_img,
                                          "want_mem": want_mem})
                for k in ("tokens", "mask", "ar", "na"):
                    if not _bytes_equal(oT[k], oI[k]):
                        prefix_leaf_mis += 1
                if not _bytes_equal(oT["pos"], oI["pos"]):
                    pos_mis += 1
                if not _bytes_equal(oT["attn"], oI["attn"]):
                    attn_mis += 1
                prefix_shape_info["attn_shape"] = tuple(int(x) for x in oI["attn"].shape)
                if not _bytes_equal(oT["k"], oI["k"]):
                    kv_k_mis += 1
                if not _bytes_equal(oT["v"], oI["v"]):
                    kv_v_mis += 1
                kv_layers = int(oI["k"].shape[0])

                # MEM_INVPERM：并列序 →(mem_order)→ 重排序，逆置换应原样还原
                order = np.asarray(assembled_S["mem_order"], np.int64)
                if order.shape != (mem_len,) or not np.array_equal(np.sort(order), np.arange(mem_len)):
                    invperm_bad += 1
                else:
                    par_tok = oI["par_tok"][0]
                    par_mask = oI["par_mask"][0]
                    if not _bytes_equal(par_tok[order], oI["mem_tok"][0]):
                        invperm_tok_mis += 1
                    if not _bytes_equal(par_mask[order], oI["mem_mask"][0]):
                        invperm_mask_mis += 1
                    inv = np.argsort(order)
                    if not _bytes_equal(oI["mem_tok"][0][inv], par_tok) or not _bytes_equal(oI["mem_mask"][0][inv], par_mask):
                        invperm_tok_mis += 1

                # MEM_SCALE_OBS（A20 口径：取数点在 take_along_axis 之前的并列序上）
                par_tok = oI["par_tok"][0].astype(np.float32)
                sm_mask = np.asarray(assembled_S["static_mask"], bool)
                mo_mask = np.asarray(assembled_S["motion_mask"], bool)
                nf = int(sm_mask.sum())
                nm = int(mo_mask.sum())
                if nf and nm:
                    fr = float(np.mean(np.linalg.norm(par_tok[:len(sm_mask)][sm_mask], axis=-1)))
                    mr = float(np.mean(np.linalg.norm(par_tok[len(sm_mask):][mo_mask], axis=-1)))
                    scale_ratios.append(mr / (fr + 1e-30))

                # ── L4 整段前向 vs 缓存分步（同一权重、同一 S 臂 obs、同一 x_t / time）───────────
                acts_true = jnp.asarray(np.asarray(train_inputs["actions"], np.float32))[np.newaxis, ...]
                tt = 0.5
                x_t = tt * NOISE[noise_seeds[0]] + (1.0 - tt) * acts_true
                tstep = jnp.asarray([tt], jnp.float32)
                fu = jax.tree.map(lambda x: np.asarray(x), jax.block_until_ready(f_full(gstate, obs_I, x_t, tstep)))
                ca = jax.tree.map(lambda x: np.asarray(x), jax.block_until_ready(f_cached(gstate, obs_I, x_t, tstep)))
                P = int(fu["prefix_len"])
                l4_struct["n"] += 1
                l4_struct["prefix_mask_block"] += int(_bytes_equal(fu["attn"][:, :P, :P], ca["prefix_attn"]))
                l4_struct["suffix_rows"] += int(_bytes_equal(fu["attn"][:, P:, :], ca["full_attn"]))
                l4_struct["suffix_positions"] += int(_bytes_equal(fu["pos"][:, P:], ca["spos"]))
                pval = np.asarray(fu["pmask"][0], bool)          # 只比 prefix_mask=True 的 KV 位
                l4_struct["prefix_kv_pad"] += int((~pval).sum())
                l4_struct["prefix_kv_bitexact"] += int(_bytes_equal(fu["k"][:, :, :P][:, :, pval], ca["k"][:, :, pval])
                                                       and _bytes_equal(fu["v"][:, :, :P][:, :, pval], ca["v"][:, :, pval]))
                l4_vt.add(fu["v_t"], ca["v_t"])
                l4_vt_rms.append(_rms(np.asarray(fu["v_t"], np.float32)))
                _acc_num(l4_kv, fu["k"][:, :, :P][:, :, pval], ca["k"][:, :, pval])
                _acc_num(l4_kv, fu["v"][:, :, :P][:, :, pval], ca["v"][:, :, pval])
                r["l4"] = {"prefix_len": P, "vt_max_abs": float(np.abs(np.asarray(fu["v_t"], np.float32)
                                                                      - np.asarray(ca["v_t"], np.float32)).max())}
                if ep_pt_idx < int(args.f32_diag_points_per_episode):
                    # f32 诊断留样：主循环里不建第二份权重（省显存），跑完再统一重算
                    f32_cache.append({"g": g, "t": t, "inputs": jax.tree.map(lambda x: np.asarray(x), infer_inputs),
                                      "x_t": np.asarray(x_t), "tstep": np.asarray(tstep)})

                # ── L5 最终动作 ────────────────────────────────────────────────────────────
                policy.mem_buffer = mems["B"]
                infer_inputs_B = policy._input_transform(policy._prepare_history(dict(element)))
                obs_B = _obs_of(infer_inputs_B)
                acts = {"T": {}, "I": {}, "B": {}}
                for sd in noise_seeds:
                    for nm_, ob in (("T", obs_T), ("I", obs_I), ("B", obs_B)):
                        a = policy._sample_actions(jax.random.key(0), ob, noise=NOISE[sd], **policy._sample_kwargs)
                        acts[nm_][sd] = np.asarray(jax.block_until_ready(a))[0]
                    if not _bytes_equal(acts["T"][sd], acts["I"][sd]):
                        act_ti_mis += 1
                a2 = np.asarray(jax.block_until_ready(policy._sample_actions(
                    jax.random.key(0), obs_I, noise=NOISE[noise_seeds[0]], **policy._sample_kwargs)))[0]
                if not _bytes_equal(a2, acts["I"][noise_seeds[0]]):
                    act_det_ok = False
                un = {}
                for nm_ in ("I", "B"):
                    un[nm_] = {sd: policy._output_transform({"state": np.asarray(obs_I.state[0]), "actions": acts[nm_][sd]})["actions"]
                               for sd in noise_seeds}
                dn = [acts["B"][sd] - acts["I"][sd] for sd in noise_seeds]
                du = [un["B"][sd] - un["I"][sd] for sd in noise_seeds]
                act_sb["norm"].extend(_rms(x) for x in dn)
                act_sb["unnorm"].extend(_rms(x) for x in du)
                act_sb_max = max(act_sb_max, max(float(np.abs(x).max()) for x in dn))
                pairs = [(i, j) for i in range(len(noise_seeds)) for j in range(i + 1, len(noise_seeds))]
                noise_ref["norm"].extend(_rms(acts["I"][noise_seeds[j]] - acts["I"][noise_seeds[i]]) for i, j in pairs)
                noise_ref["unnorm"].extend(_rms(un["I"][noise_seeds[j]] - un["I"][noise_seeds[i]]) for i, j in pairs)
                r["act"] = {"S_vs_B_rms_norm": [_rms(x) for x in dn], "S_vs_B_rms_unnorm": [_rms(x) for x in du]}

                # AUG_EFFECT_OBS：preprocess(train=True) 的增广只应动图像、记忆键必须原样透传
                aug = jax.block_until_ready(f_aug(obs_I, jax.random.key(0)))
                aug_points += 1
                for k in KEYS8:
                    a = getattr(aug, k)
                    b = getattr(obs_I, k)
                    if a is None or b is None or not _bytes_equal(np.asarray(a), np.asarray(b)):
                        aug_mem_unchanged = False
                if not _bytes_equal(np.asarray(aug.images["base_0_rgb"]), np.asarray(obs_I.images["base_0_rgb"])):
                    aug_img_changed += 1
                if is_last:
                    aa = np.asarray(jax.block_until_ready(policy._sample_actions(
                        jax.random.key(0), aug, noise=NOISE[noise_seeds[0]], **policy._sample_kwargs)))[0]
                    aug_act_rms.append(_rms(aa - acts["I"][noise_seeds[0]]))
                last_infer_inputs = {k: (np.asarray(v) if v is not None else None)
                                     for k, v in _flatten(infer_inputs).items()}
                print(f"[tic] g={g} t={t:4d} 新帧 {len(new_steps):3d} | 帧 rel_fro S-B={frame_acc[('S','B')].summary()['rel_fro']:.3g} "
                      f"| 八键 ok={all(key_ok.values())} obs_mis={len(obs_key_mis)} tok_eq={tok_eq} "
                      f"| L4 vt_max={r['l4']['vt_max_abs']:.3g} | act T==I {act_ti_mis == 0}")

            checked_motion: set[int] = set()
            sl = feed(0, es + 1)
            check(es, sl, len(taus) == 1, 0)
            for j, tau in enumerate(taus[1:], start=1):
                sl = feed(tau - 15, tau + 1)
                check(tau, sl, j == len(taus) - 1, j)
            per_episode.append({"g": g, "h5_file": ep["h5_file"], "raw_ep_idx": ep["raw_ep_idx"], "es": es, "T": T,
                                "points": len(taus), "motion_windows": len(checked_motion) if client is not None else None,
                                "wall_s": time.perf_counter() - tb})
            print(f"[tic] episode g={g} 完成 points={len(taus)} 用时 {per_episode[-1]['wall_s']:.1f}s")

        # ── VT_FULL_VS_CACHED_F32（观察）：参数树与 embed_dtype 一起升 f32 后在同一输入上重跑两路 ──
        # 目的是给「VT_FULL_VS_CACHED 阈值是否按实测重定」提供依据：f32 下差异掉到 1e-5 量级
        # ⇒ bf16 档的 3e-3 是 kernel 选择与归约序的数值来源；仍是 3e-3 量级 ⇒ 是真问题。
        # 注意必须同时把 embed_dtype 升 f32——只换参数树不够，`history_gemma.Module.__call__` 会把
        # embedded 一律 astype(self.embed_dtype)，激活留在 bf16 就测不出权重精度以外的东西。
        if f32_cache and not prompt_fail:
            t0 = time.perf_counter()
            params_f32 = _model.restore_params(ckpt_dir / "params", dtype=jnp.float32)
            tc_f32 = dataclasses.replace(train_config, model=dataclasses.replace(
                train_config.model, history_config=hc, use_history=True, dtype="float32"))
            model_f32 = tc_f32.model.load(params_f32, remove_extra_params=False)
            gdef32, gstate32 = nnx.split(model_f32)
            f_full32, f_cached32 = _mk_l4(gdef32)
            acc_vt32 = DiffAcc()
            acc_kv32 = {"n": 0, "max_abs": 0.0, "sum_d2": 0.0, "sum_a2": 0.0, "ulp_p99": 0.0, "ulp_max": 0.0}
            rel_parts: list[np.ndarray] = []
            vt_rms32: list[float] = []
            # A100 上 XLA 对 f32 matmul 默认走 TF32（10 bit 尾数），要坐实「真 f32 归约」必须显式 highest；
            # 在 with 内首次调用 jit 即按 highest 追踪编译。
            with jax.default_matmul_precision("highest"):
              for it in f32_cache:
                ob32 = _obs_of(it["inputs"])
                xt32 = jnp.asarray(it["x_t"])
                ts32 = jnp.asarray(it["tstep"])
                fu32 = jax.tree.map(lambda x: np.asarray(x), jax.block_until_ready(f_full32(gstate32, ob32, xt32, ts32)))
                ca32 = jax.tree.map(lambda x: np.asarray(x), jax.block_until_ready(f_cached32(gstate32, ob32, xt32, ts32)))
                P32 = int(fu32["prefix_len"])
                acc_vt32.add(fu32["v_t"], ca32["v_t"])
                vt_rms32.append(_rms(np.asarray(fu32["v_t"], np.float32)))
                pval32 = np.asarray(fu32["pmask"][0], bool)
                _acc_num(acc_kv32, fu32["k"][:, :, :P32][:, :, pval32], ca32["k"][:, :, pval32])
                _acc_num(acc_kv32, fu32["v"][:, :, :P32][:, :, pval32], ca32["v"][:, :, pval32])
                av = np.asarray(fu32["v_t"], np.float32)
                bv = np.asarray(ca32["v_t"], np.float32)
                base32 = np.maximum(np.abs(av), np.abs(bv))
                sel32 = base32 > 0
                if np.any(sel32):
                    rel_parts.append(np.abs(bv - av)[sel32] / base32[sel32])
            rr = np.concatenate(rel_parts) if rel_parts else np.zeros(1)
            s32 = acc_vt32.summary()
            kv_rel32 = float(np.sqrt(acc_kv32["sum_d2"] / max(acc_kv32["sum_a2"], 1e-30)))
            f32_line = (f"VT_FULL_VS_CACHED_F32 points={len(f32_cache)} rel_fro={s32['rel_fro']:.4g} "
                        f"max_abs={s32['max_abs']:.4g} vt_rms={_mean_str(vt_rms32)} "
                        f"max_rel_p99={float(np.percentile(rr, 99)):.4g} max_rel_max={float(rr.max()):.4g} "
                        f"prefix_kv_rel_fro={kv_rel32:.4g} prefix_kv_max_abs={acc_kv32['max_abs']:.4g} "
                        f"dtype=f32(params=restore_params(dtype=float32),embed_dtype=float32,matmul_precision=highest) "
                        f"bf16_ref_rel_fro={l4_vt.summary()['rel_fro']:.4g} "
                        f"points_per_episode={args.f32_diag_points_per_episode} "
                        f"wall_s={time.perf_counter() - t0:.1f}")
            rec["l4_f32"] = {"vt": s32, "prefix_kv": acc_kv32, "vt_rms": vt_rms32,
                             "max_rel_p99": float(np.percentile(rr, 99)), "max_rel_max": float(rr.max()),
                             "samples": [{"g": it["g"], "t": it["t"]} for it in f32_cache]}
            del params_f32, model_f32, gstate32, gdef32, f_full32, f_cached32
            gc.collect()                    # 三份权重同卡（bf16 生产 + f32 诊断 + 随后的 ACT_CKPT_DTYPE）时先放掉 f32 缓冲

        # ── ACT_CKPT_DTYPE（仅 store 档；盘上原精度 vs 生产 bf16 加载，观察）────────────────────
        if args.motion == "store" and last_infer_inputs is not None and not prompt_fail:
            t0 = time.perf_counter()
            params_raw = _model.restore_params(ckpt_dir / "params", dtype=None)
            tc_raw = dataclasses.replace(train_config, model=dataclasses.replace(
                train_config.model, history_config=hc, use_history=True))       # 与 create_trained_policy 逐字同式
            model_raw = tc_raw.model.load(params_raw, remove_extra_params=False)
            import openpi.shared.nnx_utils as _nnx_utils
            sample_raw_fn = _nnx_utils.module_jit(model_raw.sample_actions)
            # last_infer_inputs 是展平后的键，重建嵌套结构再造 obs
            nested: dict = {}
            for k, v in last_infer_inputs.items():
                cur = nested
                parts = k.split("/")
                for p in parts[:-1]:
                    cur = cur.setdefault(p, {})
                cur[parts[-1]] = v
            obs_last = _obs_of(nested)
            a_bf16 = np.asarray(jax.block_until_ready(policy._sample_actions(
                jax.random.key(0), obs_last, noise=NOISE[noise_seeds[0]], **policy._sample_kwargs)))[0]
            a_raw = np.asarray(jax.block_until_ready(sample_raw_fn(
                jax.random.key(0), obs_last, noise=NOISE[noise_seeds[0]], **policy._sample_kwargs)))[0]
            dtypes = sorted({str(np.asarray(x).dtype) for x in jax.tree.leaves(params_raw)})
            ckpt_dtype_line = (f"ACT_CKPT_DTYPE points=1 ckpt_dtypes={dtypes} rms_norm={_rms(a_raw - a_bf16):.4g} "
                               f"max_abs_norm={float(np.abs(a_raw - a_bf16).max()):.4g} "
                               f"noise_rms_norm={_mean_str(noise_ref['norm'])} "
                               f"ratio={_ratio_str([_rms(a_raw - a_bf16)], noise_ref['norm'])} "
                               f"act_std_mean={ns_actions_std:.4g} wall_s={time.perf_counter() - t0:.1f}")
            del params_raw, model_raw, sample_raw_fn
    finally:
        if client is not None:
            client.close()
        mstore.close()
        ds_raw.close()

    # ── 判定行 ────────────────────────────────────────────────────────────────────────────────
    raw_block_mis = {k: v for k, v in raw_mis.items() if v and k != "prompt_text"}
    V.block(not raw_block_mis,
            f"RAW_OBS_VS_PKL={'PASS' if not raw_block_mis else 'FAIL'} points={n_done} "
            f"keys=[image,wrist_image,state,prompt_text] mismatches={raw_block_mis or 'none'} "
            f"prompt_text_mismatch={raw_mis['prompt_text']}")
    key_mis = {k: v for k, v in trainset_key_mis.items() if v}
    V.block(not key_mis, f"MEM_S_VS_TRAINSET={'PASS' if not key_mis else 'FAIL'} points={n_done} "
                         f"keys={len(KEYS8)} key_mismatches={key_mis or 'none'}")
    for p in MEM_PAIRS:
        V.observe(frame_acc[p].line(f"MEM_{p[0]}_VS_{p[1]}"))
    if client is not None:
        ok = motion_sidecar_cmp[1] == 0 and motion_sidecar_cmp[0] > 0
        V.block(ok, f"MOTION_S_VS_SIDECAR={'PASS' if ok else 'FAIL'} windows={motion_sidecar_cmp[0]} "
                    f"mismatches={motion_sidecar_cmp[1]} provenance_keys_equal=1 lib={lib.name} "
                    f"policy=create_trained_policy motion_client=MotionEncoderClient(online_gpu={args.motion_gpu})")
    obs_ok = (not obs_key_mis) and obs_train_only <= {"actions"} and not obs_infer_only
    V.block(obs_ok, f"OBS_T_VS_I={'PASS' if obs_ok else 'FAIL'} points={n_done} keys={len(obs_key_mis) if obs_key_mis else 'all_equal'} "
                    f"mismatches={obs_key_mis or 'none'} train_only={sorted(obs_train_only)} infer_only={sorted(obs_infer_only)}")
    p_ok = prompt_tok_mis == 0 and n_done > 0
    if prompt_samples:
        _ps = prompt_samples[0]
        _pt = (f"OBS_PROMPT={'PASS' if p_ok else 'FAIL'} points={n_done} text_equal={prompt_text_eq} "
               f"tok_mismatches={prompt_tok_mis} train_text={_ps['train_text']!r} infer_text={_ps['infer_text']!r} "
               f"tokenizer={tok_desc}")
    else:
        _pt = "OBS_PROMPT=FAIL points=0 tok_mismatches=0 train_text=None infer_text=None"
    V.block(p_ok, _pt)
    tc_ok = train_only == ["RepackTransform"] and infer_only == ["InjectDefaultPrompt"] and noop_inject_ok and repack_reach_model == 0
    V.block(tc_ok, f"TRANSFORMS_CHAIN={'PASS' if tc_ok else 'FAIL'} train_only={train_only} infer_only={infer_only} "
                   f"noop_proofs=2 inject_default_prompt_noop={int(noop_inject_ok)} "
                   f"repack_dropped_keys={sorted(repack_dropped)} repack_dropped_reach_model={repack_reach_model}")
    if prompt_fail:
        for name in ("PREFIX_SHAPE", "PREFIX_T_VS_I", "PREFIX_POSITIONS", "MEM_INVPERM", "KV_T_VS_I",
                     "FULL_VS_CACHED_STRUCT", "VT_FULL_VS_CACHED", "ACT_T_VS_I"):
            V.block(False, f"{name}=SKIP(OBS_PROMPT_FAIL)")
        V.observe("MEM_SCALE_OBS=SKIP(OBS_PROMPT_FAIL)")
        V.observe("VT_FULL_VS_CACHED_F32=SKIP(OBS_PROMPT_FAIL)")
        V.observe("ACT_S_VS_B=SKIP(OBS_PROMPT_FAIL)")
        V.observe("AUG_EFFECT_OBS=SKIP(OBS_PROMPT_FAIL)")
    else:
        i = prefix_shape_info
        V.block(prefix_shape_ok, f"PREFIX_SHAPE={'PASS' if prefix_shape_ok else 'FAIL'} len={i.get('len')} "
                                 f"mem={i.get('mem')} img={i.get('img')} txt={i.get('txt')} "
                                 f"img_per_view={i.get('img_per_view')} views={i.get('views')} want_mem={i.get('want_mem')}")
        V.block(prefix_leaf_mis == 0, f"PREFIX_T_VS_I={'PASS' if prefix_leaf_mis == 0 else 'FAIL'} points={n_done} "
                                      f"leaves=4 mismatches={prefix_leaf_mis or 'none'}")
        V.block(pos_mis == 0 and attn_mis == 0,
                f"PREFIX_POSITIONS={'PASS' if pos_mis == 0 and attn_mis == 0 else 'FAIL'} positions_mismatches={pos_mis} "
                f"attn_mask_mismatches={attn_mis} attn_mask_shape={i.get('attn_shape')} na_branch=1")
        ip_ok = invperm_tok_mis == 0 and invperm_mask_mis == 0 and invperm_bad == 0
        V.block(ip_ok, f"MEM_INVPERM={'PASS' if ip_ok else 'FAIL'} token_mismatches={invperm_tok_mis} "
                       f"mask_mismatches={invperm_mask_mis} perm_valid={n_done - invperm_bad}/{n_done}")
        kv_ok = kv_k_mis == 0 and kv_v_mis == 0
        V.block(kv_ok, f"KV_T_VS_I={'PASS' if kv_ok else 'FAIL'} layers={kv_layers} k_mismatches={kv_k_mis} v_mismatches={kv_v_mis}")
        sr = np.asarray(scale_ratios) if scale_ratios else np.zeros(0)
        in_band = int(np.sum((sr >= 0.3) & (sr <= 3.0))) if sr.size else 0
        V.observe(f"MEM_SCALE_OBS points={sr.size} ratio_mean={float(sr.mean()) if sr.size else float('nan'):.4g} "
                  f"min={float(sr.min()) if sr.size else float('nan'):.4g} max={float(sr.max()) if sr.size else float('nan'):.4g} "
                  f"a20_band=[0.3,3.0] in_band={in_band}/{sr.size}")
        n4 = l4_struct["n"]
        st_ok = (l4_struct["prefix_mask_block"] == n4 and l4_struct["suffix_rows"] == n4
                 and l4_struct["suffix_positions"] == n4 and n4 > 0)
        V.block(st_ok, f"FULL_VS_CACHED_STRUCT={'PASS' if st_ok else 'FAIL'} points={n4} "
                       f"prefix_mask_block_equal={l4_struct['prefix_mask_block']}/{n4} "
                       f"suffix_rows_equal={l4_struct['suffix_rows']}/{n4} "
                       f"suffix_positions_equal={l4_struct['suffix_positions']}/{n4} "
                       f"prefix_kv_bitexact_valid={l4_struct['prefix_kv_bitexact']}/{n4} "
                       f"prefix_kv_pad_positions_excluded={l4_struct['prefix_kv_pad']}")
        s4 = l4_vt.summary()
        vt_ok = n4 > 0 and s4["rel_fro"] <= THR_REL_FRO and s4["ulp_p99"] <= THR_ULP_P99
        V.block(vt_ok, f"VT_FULL_VS_CACHED={'PASS' if vt_ok else 'FAIL'} points={n4} rel_fro={s4['rel_fro']:.4g} "
                       f"max_abs={s4['max_abs']:.4g} vt_rms={_mean_str(l4_vt_rms)} "
                       f"ulp_p99={s4['ulp_p99']:.3g} ulp_max={s4['ulp_max']:.3g} "
                       f"thr_rel_fro={THR_REL_FRO} thr_ulp_p99={THR_ULP_P99} | prefix_kv {_num_line(l4_kv)} "
                       f"(prefix_kv 只统计 prefix_mask=True 的位) "
                       f"| note=threshold_pending_user_review(阈值为计划事先定死，实测超出属待裁决项，"
                       f"本脚本不自行放宽；结构三项全等，差异自前缀 pass 起，见 VT_FULL_VS_CACHED_F32)")
        if f32_line is not None:
            V.observe(f32_line)
        a_ok = act_ti_mis == 0 and act_det_ok
        V.block(a_ok, f"ACT_T_VS_I={'PASS' if a_ok else 'FAIL'} points={n_done} seeds={len(noise_seeds)} "
                      f"mismatches={act_ti_mis} determinism_rerun={'PASS' if act_det_ok else 'FAIL'}")
        V.observe(f"ACT_S_VS_B points={n_done} rms_norm={_mean_str(act_sb['norm'])} max_abs_norm={act_sb_max:.4g} "
                  f"rms_unnorm={_mean_str(act_sb['unnorm'])} | NOISE_S seeds={len(noise_seeds)} "
                  f"rms_norm={_mean_str(noise_ref['norm'])} rms_unnorm={_mean_str(noise_ref['unnorm'])} "
                  f"| ratio_norm={_ratio_str(act_sb['norm'], noise_ref['norm'])} "
                  f"| act_std_mean={ns_actions_std:.4g} "
                  f"rms_unnorm/act_std={_mean_str(act_sb['unnorm']) if not act_sb['unnorm'] else format(float(np.mean(act_sb['unnorm'])) / ns_actions_std, '.4g')}")
        V.observe(f"AUG_EFFECT_OBS points={aug_points} img_changed={aug_img_changed}/{aug_points} "
                  f"mem_tokens_unchanged={int(aug_mem_unchanged)} "
                  f"act_rms_norm={np.mean(aug_act_rms) if aug_act_rms else float('nan'):.4g}")
        if not aug_mem_unchanged:      # 子判据硬：增广动了记忆键即视作阻断失败
            V.block(False, "AUG_EFFECT_OBS_SUBGATE=FAIL mem_tokens_unchanged=0")
        if ckpt_dtype_line is not None:
            V.observe(ckpt_dtype_line)

    tag = "TIC_OBS_MODEL" if args.motion == "store" else "TIC_SIDECAR"
    extra = f"motion=store" if args.motion == "store" else f"motion=sidecar windows={motion_sidecar_cmp[0]}"
    final = (f"{tag}={'PASS' if V.all_ok else 'FAIL'} episodes={len(eps)} points={n_done} {extra} "
             f"blocking={V.block_pass}/{V.block_total} observe={V.n_observe}")
    lines = V.texts() + [final]
    print("\n".join(lines))
    rec.update({"lines": lines, "per_episode": per_episode, "per_point": per_point,
                "prompt_samples": prompt_samples, "prefix_shape": prefix_shape_info,
                "frame_diff": {f"{p[0]}_vs_{p[1]}": frame_acc[p].summary() for p in MEM_PAIRS},
                "l4_struct": l4_struct, "l4_vt": l4_vt.summary(), "l4_vt_rms": l4_vt_rms, "l4_prefix_kv": l4_kv,
                "obs_key_mismatches": obs_key_mis, "obs_train_only": sorted(obs_train_only),
                "obs_infer_only": sorted(obs_infer_only), "repack_dropped_keys": sorted(repack_dropped),
                "transforms": {"train": train_names, "infer": infer_names,
                               "train_only": train_only, "infer_only": infer_only},
                "scale_ratios": scale_ratios, "act_S_vs_B": act_sb, "noise_ref": noise_ref,
                "aug": {"points": aug_points, "img_changed": aug_img_changed,
                        "mem_unchanged": aug_mem_unchanged, "act_rms": aug_act_rms},
                "blocking": [V.block_pass, V.block_total], "observe": V.n_observe})
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    (out_path.parent / (out_path.stem + ".summary.txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if V.all_ok else 1


def _acc_num(acc: dict, a, b) -> None:
    """大数组数值差累计（不算 cos，避免 KV 这类 5M 元素张量把内存打满）。"""
    a32 = np.asarray(a).astype(np.float32)
    b32 = np.asarray(b).astype(np.float32)
    d = b32 - a32
    acc["n"] += 1
    acc["max_abs"] = max(acc["max_abs"], float(np.abs(d).max()))
    acc["sum_d2"] += float((d.astype(np.float64) ** 2).sum())
    acc["sum_a2"] += float((a32.astype(np.float64) ** 2).sum())
    base = np.maximum(np.abs(a32), np.abs(b32))
    sel = base > 0
    if np.any(sel):
        u = (np.abs(d)[sel] / _bf16_ulp(base[sel])).ravel()
        acc["ulp_p99"] = max(acc["ulp_p99"], float(np.percentile(u, 99)))
        acc["ulp_max"] = max(acc["ulp_max"], float(u.max()))


def _num_line(acc: dict) -> str:
    rel = float(np.sqrt(acc["sum_d2"] / max(acc["sum_a2"], 1e-30)))
    return (f"tensors={acc['n']} rel_fro={rel:.4g} max_abs={acc['max_abs']:.4g} "
            f"ulp_p99_max={acc['ulp_p99']:.3g} ulp_max={acc['ulp_max']:.3g}")


if __name__ == "__main__":
    sys.exit(main())
