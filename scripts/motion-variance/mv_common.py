#!/usr/bin/env python3
"""motion-variance 公共层：常量、路径、哈希口径、判定行收集器、源码护栏、episode 计划、motion 列算法。

**为什么单独一层**：本目录的 adapter / serve / open_loop / summarize 四个入口共用同一套判定行格式、
哈希口径与「第 n 次 reset 应是哪一集」的遍历序；把它们收在一处，防止四份实现各自漂移。
哈希口径 `leaf_sha256` 与 `scripts/training/tests/_common.py` 逐字相同（raw = sha256(dtype‖shape‖bytes)）。

**三条纪律（计划第二部分 §2）**：
- `guard_source` 先做空白规范化再匹配 needle（Codex 审计 3：`kv_cache=kv_cache,` 与 `adarms_cond=[…]` 在源码里分两行）。
- `episode_plan` 是 eval 客户端 / serve 端 / 批验收三方**唯一**的遍历序来源（审计 2：stride 8 下 w0/w1 各 28 集、其余 24 集）。
- `motion_col_mask_np` 用与 `HistoryPi0.embed_memory` 同向的 take_along_axis（`is_motion_par[mem_order]`），不用 argsort。
"""

from __future__ import annotations

import hashlib
import inspect
import os
import pathlib
import re

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if not (REPO_ROOT / "pyproject.toml").is_file():
    raise SystemExit(f"错误: 仓库根解析失败 {REPO_ROOT}（缺 pyproject.toml）")
V1 = pathlib.Path(os.environ.get("MMEVLA_V1_STORE", str(REPO_ROOT / "v1-store")))

# ── 被评对象与库（环境 B 实体路径，计划第二部分 §2）────────────────────────────────────────
CKPT_MOTION = V1 / "train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999"
CKPT_OFFICIAL = V1 / "models/official-mme-vla/perceptual-framesamp-context/79999"
LIB = V1 / "datasets/4task-motion-400ep"
REPORT_ROOT = V1 / "reports/motion-variance"
TRAIN_CONFIG = "mme_vla_suite_b128"

TASKS = ("ButtonUnmask", "VideoUnmask", "ButtonUnmaskSwap", "VideoUnmaskSwap")
SEEDS = (42, 7, 2024, 17)
SPLITS = ("test", "val")
CONDS = ("official", "normal", "mask", "swap")
EPISODES_PER_TASK = 50            # benchmark test / val 每任务集数（env_metadata/<split>/）
MAX_STEPS = 1300                  # robomme/eval.py::Args.max_steps

# ── 模型形制（awsprod40k-b128-motion 快照：budget 512 / token_per_image 16 / motion.budget 96 / 文本 64）──
MEM_LEN, FRAME_SLOTS, MOTION_SLOTS = 608, 512, 96
IMG_LEN, TXT_LEN = 512, 64
PREFIX_LEN = MEM_LEN + IMG_LEN + TXT_LEN        # 1184
N_LAYERS, N_QHEAD, N_KVHEAD, HEAD_DIM = 18, 8, 1, 256
ACT_H = 20
MOTION_DIM, MOTION_POS_DIM = 768, 256
GRID_STRIDE, WINDOW_FRAMES = 16, 33

# ── 统计常量 ──────────────────────────────────────────────────────────────────────────────
ROPE_PP = 2.0        # 实际等价界限 ±2pp（用户 2026-09-08 拍板）
T975_DF3 = 3.182     # t 分布 0.975 分位、自由度 3（4 seed 配对差）；NIST/SEMATECH e-Handbook prc/section3/prc312
BOOT_N = 10000
BOOT_SEED = 20260908

XLA_DET_FLAGS = ("--xla_gpu_deterministic_ops=true", "--xla_gpu_autotune_level=0")

KEYS8 = ("static_image_emb", "static_pos_emb", "static_state_emb", "static_mask",
         "motion_emb", "motion_pos", "motion_mask", "mem_order")


# ── 哈希 / 数值口径（逐字抄 scripts/training/tests/_common.py 与 g0/compare_train_infer_obs.py）────
def leaf_sha256(arr) -> str:
    a = np.asarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def bytes_equal(a, b) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a.view(np.uint8), b.view(np.uint8))


def rms(x) -> float:
    x = np.asarray(x, np.float64)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def bf16_ulp(x32: np.ndarray) -> np.ndarray:
    """bf16 在 |x| 处的 ULP：2^(floor(log2|x|) - 7)。"""
    ax = np.abs(x32).astype(np.float64)
    e = np.floor(np.log2(np.where(ax > 0, ax, 1.0)))
    return np.ldexp(1.0, (e - 7).astype(np.int32))


def mask_bits(arr) -> str:
    a = np.asarray(arr)
    if a.dtype != np.bool_:
        raise RuntimeError(f"mask dtype {a.dtype} != bool_")
    return "".join("1" if bool(x) else "0" for x in a.reshape(-1))


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
        self.ulp: list[np.ndarray] = []

    def add(self, a, b):
        a32 = np.asarray(a).astype(np.float32)
        b32 = np.asarray(b).astype(np.float32)
        d = b32 - a32
        self.n += 1
        self.max_abs = max(self.max_abs, float(np.abs(d).max()) if d.size else 0.0)
        self.sum_abs += float(np.abs(d).sum())
        self.cnt += d.size
        self.nonzero += int((d != 0).sum())
        self.sum_d2 += float((d.astype(np.float64) ** 2).sum())
        self.sum_a2 += float((a32.astype(np.float64) ** 2).sum())
        base = np.maximum(np.abs(a32), np.abs(b32))
        sel = base > 0
        if np.any(sel):
            self.ulp.append((np.abs(d)[sel] / bf16_ulp(base[sel])).ravel())

    def summary(self) -> dict:
        ulp = np.concatenate(self.ulp) if self.ulp else np.zeros(1)
        return {"n": self.n, "max_abs": self.max_abs, "mean_abs": self.sum_abs / max(self.cnt, 1),
                "rel_fro": float(np.sqrt(self.sum_d2 / max(self.sum_a2, 1e-30))),
                "ulp_p50": float(np.percentile(ulp, 50)), "ulp_p99": float(np.percentile(ulp, 99)),
                "ulp_max": float(ulp.max()), "frac_nonzero": self.nonzero / max(self.cnt, 1)}

    def line(self, tag: str) -> str:
        s = self.summary()
        return (f"{tag} n={s['n']} max_abs={s['max_abs']:.4g} rel_fro={s['rel_fro']:.4g} "
                f"ulp_p50={s['ulp_p50']:.3g} ulp_p99={s['ulp_p99']:.3g} ulp_max={s['ulp_max']:.3g} "
                f"frac_nonzero={s['frac_nonzero']:.3f}")


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

    def final(self, tag: str, extra: str = "") -> str:
        return (f"{tag}={'PASS' if self.all_ok else 'FAIL'} {extra}".rstrip()
                + f" blocking={self.block_pass}/{self.block_total} observe={self.n_observe}")


# ── 运行前提硬闸 ──────────────────────────────────────────────────────────────────────────
def require_gpu() -> str:
    import jax
    backend = jax.default_backend()
    if backend != "gpu":
        raise SystemExit(f"错误: 主进程 jax 后端 {backend} != gpu（PosEmb3D 4x4 表 CPU/GPU 不逐位，P5 留档）")
    return backend


def require_det_flags() -> str:
    flags = os.environ.get("XLA_FLAGS", "")
    missing = [f for f in XLA_DET_FLAGS if f not in flags]
    if missing:
        raise SystemExit(f"错误: XLA_FLAGS 缺确定性开关 {missing}（当前 {flags!r}）；"
                         f"须 XLA_FLAGS='{' '.join(XLA_DET_FLAGS)}'")
    return flags


def require_motion_store(lib: pathlib.Path = LIB) -> pathlib.Path:
    """run 快照 store_path 记的是 40ep 库，必须显式覆盖到 400ep 库，否则 check_same_source 会 raise。"""
    root = pathlib.Path(lib) / "motion"
    if not (root / "meta" / "store_meta.json").is_file():
        raise SystemExit(f"错误: motion 库不存在或缺 meta: {root}")
    os.environ["MMEVLA_MOTION_STORE"] = str(root)
    return root


# ── 源码指纹护栏（范式 scripts/training/tests/single_step_grad.py::_guard_train_step_source）────
def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def guard_source(obj, needles: tuple[str, ...], what: str) -> None:
    """`inspect.getsource(obj)` 经空白规范化后必须包含每一个 needle（needle 同样规范化），缺任一即拒跑。"""
    src = _norm_ws(inspect.getsource(obj))
    missing = [n for n in needles if _norm_ws(n) not in src]
    if missing:
        raise SystemExit(f"错误: 源码护栏 {what} 失败——以下预期片段不在当前源码中（生产代码已漂移，本工具的复刻不再可信）:\n"
                         + "\n".join(f"  - {m}" for m in missing))


# ── episode 计划（三方唯一来源）────────────────────────────────────────────────────────────
def episode_plan(tasks, ep_start: int, ep_stride: int, n_per_task: int = EPISODES_PER_TASK,
                 ep_count: int = 0) -> list[tuple[str, int]]:
    """复刻 robomme/eval.py::evaluate 的遍历序：任务在外层、集号 range(start, N', stride) 在内层，
    N' = min(N, start + count) 当 count > 0，否则 N。stride 8 下 w0/w1 各 28 集、w2..w7 各 24 集。"""
    if isinstance(tasks, str):
        tasks = [t for t in tasks.split(",") if t]
    n_eff = min(n_per_task, ep_start + ep_count) if ep_count > 0 else n_per_task
    return [(t, e) for t in tasks for e in range(ep_start, n_eff, ep_stride)]


def expected_taus(es: int, t_env: int, max_steps: int = MAX_STEPS) -> list[int]:
    """决策时刻清单（与 scripts/training/g0/compare_train_infer_obs.py::expected_taus 同一公式）。"""
    c_env = t_env - 1 - es
    last_count = min(c_env, max_steps + 1) - 1
    return [es + 16 * j for j in range(last_count // 16 + 1)] if last_count >= 0 else []


def visible_motion_frames(es: int, step_idx: int) -> list[int]:
    """与 FrameSampMemory.visible_motion_frames 同式（demo 段 s+32≤es-1；exec 段 u+32≤t-es；全域 f=es+u）。"""
    out = []
    s = 0
    while s + (WINDOW_FRAMES - 1) <= es - 1:
        out.append(s)
        s += GRID_STRIDE
    u = 0
    while u + (WINDOW_FRAMES - 1) <= step_idx - es:
        out.append(es + u)
        u += GRID_STRIDE
    return out


# ── motion 列（与 embed_memory 的 take_along_axis 同向）────────────────────────────────────
IS_MOTION_PAR = np.concatenate([np.zeros(FRAME_SLOTS, bool), np.ones(MOTION_SLOTS, bool)])


def motion_col_mask_np(mem_order) -> np.ndarray:
    """并列序里 motion 是第 512..607 位；重排序列第 i 位的内容是并列序第 mem_order[i] 位，
    故重排序列第 i 位是 motion ⟺ IS_MOTION_PAR[mem_order[i]]。返回 bool[608]。"""
    order = np.asarray(mem_order, np.int64).reshape(-1)
    if order.shape != (MEM_LEN,) or not np.array_equal(np.sort(order), np.arange(MEM_LEN)):
        raise RuntimeError(f"mem_order 不是 0..{MEM_LEN - 1} 的置换（shape={order.shape}）")
    out = IS_MOTION_PAR[order]
    if int(out.sum()) != MOTION_SLOTS:
        raise RuntimeError(f"motion 列数 {int(out.sum())} != {MOTION_SLOTS}")
    return out


def task_of_h5(h5_file: str) -> str:
    return h5_file.replace("record_dataset_", "").replace(".h5", "")


if __name__ == "__main__":
    a = episode_plan(TASKS, 0, 8)
    b = episode_plan(TASKS, 2, 8)
    c = episode_plan(TASKS, 0, 16)
    d = episode_plan(TASKS, 2, 16)
    print(f"episode_plan stride8 w0={len(a)} w2={len(b)} | stride16 w0={len(c)} w2={len(d)} | first={a[:3]}")
    order = np.random.default_rng(0).permutation(MEM_LEN).astype(np.int32)
    m = motion_col_mask_np(order)
    assert m.sum() == MOTION_SLOTS
    print(f"MV_COMMON_SELFTEST=PASS plan_w0={len(a)} plan_w2={len(b)} motion_cols={int(m.sum())}")
