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

L4 判定口径（用户 2026-09-14 已裁决）：
  `VT_FULL_VS_CACHED` 保留 `rel_fro ≤ 1e-3` / `ulp_p99 ≤ 4` 的报告阈值，作为 bf16 观察项。
  `VT_FULL_VS_CACHED_F32` 升为阻断项：参数与 embed_dtype 一起升 f32，matmul_precision=highest，
  要求有效前缀 KV 逐位相同且输出 rel_fro ≤ 1e-6；缺失 f32 重跑证据直接失败。
  正式 8×8 验收显式传 `--f32-diag-points-per-episode 3`，两条数值路径使用同一 obs / x_t / time。

判定行分「阻断」与「观察」两类；任一阻断 FAIL 非零退出，但**所有已算出的判定行先全部打印**。
`OBS_PROMPT` 是唯一的「FAIL 即停后续层」项——两侧 prompt token 不等时 L3–L5 没有比较意义，
但 L1 / L2 仍跑完全部 episode 后再打印。

用法（主进程 jax 必须在 GPU 上——CPU 上 PosEmb3D 4x4 表不逐位，P5 留档记过）：
  CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 UV_LINK_MODE=copy uv run --no-sync python \
    scripts/training/g0/compare_train_infer_obs.py --motion store --out v1-store/reports/tic-dev/obsmodel.json

`--motion-subdir`（计划 1112-4task-modul-wanraw9-s32-50k-plan.html 第二部分 1.1 第 24 行，默认 `motion` 旧行为逐字不变）：
运动表所在子目录，如 `motion-wan-raw9-s32`。表为 wan_raw9（快照 `motion.encoder_kind == "wan_raw9"`）时 store 档查表闭包返回
`wan_raw9.flatten_raw9(mstore.rows([r])[0])`（(9,16384)，与在线 `_history_feats_motion[f]` 同形），sidecar 档按 raw9 kind
构造 `MotionEncoderClient`；记忆长度闸的 motion 部分按 token 数 budget × tokens_per_window 计（与 `wan_raw9.motion_token_budget` 同式；
旧 YAML 缺省 1，即原值）。

`--h5-root` / `--episode-map-dir`（闸 G5；本机无合并后的整任务 H5 时用）：经 `record_dataset_<task>_episode_map.json`
把 (h5_file, raw_ep_idx=new_idx) 定位到单 episode H5（`<h5-root>/<member>`，顶层组 `src_group`，即合并 H5 的 `episode_<new_idx>`），
核 timestep_count / h5_bytes / h5_sha256 后读帧；不给时合并 H5 读法不变。给 --h5-root 而不给 --episodes 时，从本机已有的单 episode H5
按任务名轮转取 5 个互异 es（每个任务取其未占用 es 中最小者，同 (任务, es) 取 T 最长——DEFAULT_EPISODES「各 es 取 T 最长」口径），
本机覆盖不足时按同一规则用全库清单补齐并打印缺失的 `H5_EPISODE {...}` 行（含 tar_object/member/h5_sha256/h5_bytes），非零退出。
`--dry-run`：只打印选中 episodes、`H5_COVERAGE=` 与 `DRY_RUN episodes=<n> complete=yes|no episodes_arg=<...>`，不建 policy、不用 GPU。

E2E-Enc 分支（计划 1117-e2e-enc-train-plan.html 第二部分「一」本文件行、三节 G5）：快照 `motion.encoder_kind == "wan_e2e"` 时
（`--motion sidecar --motion-subdir motion-wan-raw9-s16`，stride 16、budget 160）决策时刻上限 `MAX_STEPS_E2E = 2000`（旧口径仍 1300）；
store 档查表闭包返回表里 bf16 (9,16,32,32) 行的 (9,16384) 视图（不做 dtype 转换）；sidecar 档构造 e2e 客户端
（收 fp32 后立即调 `raw_bf16.raw9_f32_to_bf16`），比对对象是客户端转换后的 bf16 (9,16384) 槽对训练集 bf16 行（逐字节），
并逐窗核客户端用的转换函数就是 `raw_bf16.raw9_f32_to_bf16`（`E2E_BF16_CONVERT` 阻断行）；策略内 encoder 在 bf16 restore 下的输出
由 L3 的 `PREFIX_T_VS_I`（modulation 含 mem_tok）覆盖；另记 `E2E_ENC_TRAIN_VS_ONLINE=RECORDED`（训练精度路径 f32 参数批量 vs
在线 bf16 参数逐窗的 encoder token rel_rmse，只记不判）。

`--acceptance`（G5 验收模式）：锁定覆盖清单，必须显式给 --episodes、--ckpt、--lib、--store-subdir、--train-config、--norm-stats、
--h5-root、--episode-map-dir、--history-yaml、--motion、--motion-subdir、--motion-gpu、--out（不靠默认值），--max-points 只能为 0；
关掉 `resolve_episodes` 的静默回退与 `plan_episodes` 的自动补样——指定样本缺失、(h5,raw) 重复或 exec_start_idx 重复即
`TIC_ACCEPTANCE=FAIL` 非零退出、不换样；覆盖条件（`acceptance_coverage`）：四任务各 ≥1 局、BinFill 局有补尾窗（padded demo 窗）、
max_visible_k ≥ 120、含全库 c_env 最长的一局（训练集最长局 c_env < 2000，真实长局只要求覆盖到其最大 k）。
通过时汇总行为 `TIC_SIDECAR=PASS blocking=<n>/<n> tasks=4/4 tail_pad_windows>=1 max_visible_k=<k> …`。

`--perf-report <jsonl>…`（G5b 汇总入口）：读 `MMEVLA_PERF_LOG` 让 policy 服务进程逐次写的 JSONL（每片一份）与
`--perf-gpu-csv`（`nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -lms 1000 -i <卡>` 的采样），打印
`E2E_ONLINE_PERF=PASS p95_ms=<x> shards_per_gpu=<n> peak_gb=<x>`（另记 add_buffer、策略推理、单局墙钟三项耗时、请求字节与降采样次数，不判）。
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
MAX_STEPS_E2E = 2000          # E2E-Enc 评估 max_steps（计划 1117 P7：seed 7、max_steps 2000）；只在 e2e 快照下生效
E2E_KIND = "wan_e2e"
E2E_SUBDIR = "motion-wan-raw9-s16"
ACCEPT_MIN_K = 120            # G5 覆盖条件：至少一局 max_visible_k ≥ 120
ACCEPT_REQUIRED_FLAGS = ("--episodes", "--ckpt", "--lib", "--store-subdir", "--train-config", "--norm-stats", "--h5-root",
                         "--episode-map-dir", "--history-yaml", "--motion", "--motion-subdir", "--motion-gpu", "--out")


class AcceptanceError(SystemExit):
    """--acceptance 下的覆盖 / 选样失败：打印 TIC_ACCEPTANCE=FAIL 后非零退出，不回退、不换样。"""

    def __init__(self, reason: str):
        print(f"TIC_ACCEPTANCE=FAIL reason={reason}", flush=True)
        super().__init__(1)


def max_steps_for(motion_cfg) -> int:
    """e2e 快照（motion.encoder_kind == wan_e2e）用 2000，其余沿用 1300。"""
    if motion_cfg is not None and motion_cfg.get("encoder_kind", None) == E2E_KIND:
        return MAX_STEPS_E2E
    return MAX_STEPS


# ─────────────────────────── 通用小工具（派生自 compare_siglip_replay.py） ───────────────────────────


def load_episode_full(raw_dir: pathlib.Path, h5_file: str, raw_ep_idx: int, T: int, n_read: int,
                      single: dict | None = None):
    """读 h5 一集的前 n_read 帧（front_rgb / wrist_rgb / joint+gripper）与 setup/task_goal 原文。

    仍先核对整集 timestep 数与清单一致（n_read 只裁读取量，不放宽一致性校验）。
    ``single``（给出 --h5-root 时）为 ``plan_episodes`` 已核过 h5_bytes / h5_sha256 的单 episode H5：
    ``{"path": 绝对路径, "src_group": 顶层组名}``；该组即 merge_v2_h5.py 拷进合并 H5 的 ``episode_<new_idx>``
    （``source.copy(src_group, target, name=f"episode_{new_idx}")``），组内结构逐项相同。不给时按合并 H5 读。
    """
    import h5py
    frames, wrists, states = [], [], []
    path, group = (raw_dir / h5_file, f"episode_{raw_ep_idx}") if single is None else (single["path"], single["src_group"])
    with h5py.File(path, "r") as f:
        if single is not None and list(f) != [group]:
            raise SystemExit(f"错误: 单 episode H5 顶层组 {list(f)} != [{group!r}]: {path}")
        g = f[group]
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


def resolve_episodes(manifest: dict, spec_str: str, strict: bool = False) -> tuple[list[dict], str]:
    """解析 --episodes；任一项缺失 / (h5,raw) 重复 / es 取值重复时，改按库 manifest 每个 es 取 T 最长的一集。
    ``strict=True``（--acceptance）：上述任一情形直接 AcceptanceError，不回退。"""
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
    if reason and strict:
        raise AcceptanceError(f"指定样本不可用（{reason}），验收模式禁止回退换样")
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


def build_points(picked: list[dict], max_points: int, max_steps: int = MAX_STEPS) -> list[dict]:
    """按真实评估节奏推每集的决策时刻，并留 C % 16 诊断字段。``max_steps`` 为评估步数上限（e2e 为 2000）。"""
    out = []
    for e in picked:
        es = int(e["exec_start_idx"])
        T = int(e["num_timesteps"])
        taus = expected_taus(es, T, max_steps)
        c_env = T - 1 - es
        if max_points > 0:
            taus = taus[:max_points]
        out.append({"h5_file": e["h5_file"], "raw_ep_idx": int(e["raw_ep_idx"]),
                    "g": int(e["global_episode_idx"]), "es": es, "T": T, "c_env": c_env,
                    "c_mod16": c_env % 16, "row_base": int(e["total_sample_offset"]), "taus": taus})
    return out


# ─────────────────────────── 单 episode H5（--h5-root / --episode-map-dir） ───────────────────────────
# 读法与 compare_online_raw9.py 同：map 行 new_idx == 清单 raw_ep_idx，member 相对 --h5-root，核 h5_bytes / h5_sha256。

N_ES = 5                      # 需覆盖的 es 取值个数（DEFAULT_EPISODES 口径：5 个互异 es，各取 T 最长一集）
MAP_ROW_KEYS = ("task", "difficulty", "episode", "seed", "role", "member", "h5_sha256", "h5_bytes",
                "timestep_count", "new_idx", "src_group")


def _task_of(h5_file: str) -> str:
    return h5_file.removeprefix("record_dataset_").removesuffix(".h5")


def load_map_rows(map_dir: pathlib.Path, task: str) -> dict[int, dict]:
    """读 merge_v2_h5.py 写出的 record_dataset_<task>_episode_map.json，按 new_idx 建索引（只收 primary 行）。"""
    path = map_dir / f"record_dataset_{task}_episode_map.json"       # 与 merge_v2_h5.py::map_path 同名
    if not path.is_file():
        raise SystemExit(f"错误: 找不到 episode map: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"错误: episode map 不是行列表: {path}")
    out: dict[int, dict] = {}
    for r in rows:
        missing = [k for k in MAP_ROW_KEYS if k not in r]
        if missing:
            raise SystemExit(f"错误: episode map 行缺键 {missing}: {path}")
        if r["role"] != "primary" or r["task"] != task:
            continue
        if int(r["new_idx"]) in out:
            raise SystemExit(f"错误: episode map new_idx 重复: task={task} new_idx={r['new_idx']}")
        out[int(r["new_idx"])] = r
    return out


def tar_object(row: dict) -> str:
    """h5 bucket 里 primary episode 所在的 tar.xz 对象（与 compare_online_raw9.py::tar_object 同式）。"""
    return f"record_dataset_{row['task']}_{row['difficulty']}.h5.tar.xz"


def _sha256_file(path: pathlib.Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _local_ok(h5_root: pathlib.Path, row: dict) -> bool:
    """本机是否有该单 episode H5（存在且字节数与 map 相等；sha256 在选定后再核）。"""
    p = h5_root / row["member"]
    return p.is_file() and p.stat().st_size == int(row["h5_bytes"])


def _round_robin(cands: list[dict], n: int, taken_es: set[int]) -> list[dict]:
    """按任务名轮转，每轮每个任务取其未占用 es 中最小的一个（同 (任务, es) 取 T 最长、再取 g 最小），直到 n 个。"""
    best: dict[tuple[str, int], dict] = {}
    for e in cands:
        k = (_task_of(e["h5_file"]), int(e["exec_start_idx"]))
        cur = best.get(k)
        if cur is None or (int(e["num_timesteps"]), -int(e["global_episode_idx"])) > (
                int(cur["num_timesteps"]), -int(cur["global_episode_idx"])):
            best[k] = e
    per_task: dict[str, list[dict]] = {}
    for (task, es), e in sorted(best.items()):
        per_task.setdefault(task, []).append(e)
    out, taken = [], set(taken_es)
    while len(out) < n:
        progressed = False
        for task in sorted(per_task):
            if len(out) >= n:
                break
            queue = per_task[task]
            while queue and int(queue[0]["exec_start_idx"]) in taken:
                queue.pop(0)
            if queue:
                e = queue.pop(0)
                taken.add(int(e["exec_start_idx"]))
                out.append(e)
                progressed = True
        if not progressed:
            break
    return out


def episodes_arg(picked: list[dict]) -> str:
    return ",".join(f"{_task_of(e['h5_file'])}:{int(e['raw_ep_idx'])}" for e in picked)


def plan_episodes(manifest: dict, episodes: str | None, h5_root: pathlib.Path | None,
                  map_dir: pathlib.Path | None, verify_sha: bool = True, strict: bool = False) -> dict:
    """定下要跑的 episodes 与各自的帧来源。

    不给 h5_root：``resolve_episodes(manifest, episodes or DEFAULT_EPISODES)``，合并 H5 读法不变。
    给 h5_root：显式 episodes 照用；否则在本机已有的单 episode H5 中按 ``_round_robin`` 选 N_ES 个互异 es，
    不足时用全库清单按同一规则补齐，补上的即缺失项。选中项逐个核 map timestep_count、h5_bytes、h5_sha256。
    返回 {picked, source, single: {(h5_file, raw): {"path","src_group"}}, missing: [map 行+清单字段], arg}。
    """
    if strict and not episodes:
        raise AcceptanceError("验收模式必须显式给 --episodes（锁定覆盖清单，禁止自动选样）")
    if h5_root is None:
        picked, source = resolve_episodes(manifest, episodes or DEFAULT_EPISODES, strict=strict)
        return {"picked": picked, "source": source, "single": None, "missing": [], "arg": episodes_arg(picked)}
    if map_dir is None:
        raise SystemExit("错误: 给出 --h5-root 时必须同时给 --episode-map-dir")
    tasks = sorted({_task_of(e["h5_file"]) for e in manifest["episodes"]})
    maps = {t: load_map_rows(map_dir, t) for t in tasks}

    def row_of(e: dict) -> dict:
        r = maps[_task_of(e["h5_file"])].get(int(e["raw_ep_idx"]))
        if r is None:
            raise SystemExit(f"错误: episode map 缺行 task={_task_of(e['h5_file'])} new_idx={e['raw_ep_idx']}")
        return r

    if episodes:
        picked, source = resolve_episodes(manifest, episodes, strict=strict)
    else:
        local = [e for e in manifest["episodes"]
                 if int(e["raw_ep_idx"]) in maps[_task_of(e["h5_file"])] and _local_ok(h5_root, row_of(e))]
        picked = _round_robin(local, N_ES, set())
        n_local = len(picked)
        if n_local < N_ES:
            picked += _round_robin([e for e in manifest["episodes"] if int(e["raw_ep_idx"]) in maps[_task_of(e["h5_file"])]],
                                   N_ES - n_local, {int(e["exec_start_idx"]) for e in picked})
        picked = sorted(picked, key=lambda e: int(e["exec_start_idx"]))
        source = (f"auto-h5(本机单 episode H5 {len(local)} 集，按任务轮转取 {N_ES} 个互异 es、同 es 取 T 最长；"
                  f"本机覆盖 {n_local}/{N_ES})")
    single: dict = {}
    missing: list[dict] = []
    for e in picked:
        r = row_of(e)
        if int(r["timestep_count"]) != int(e["num_timesteps"]):
            raise SystemExit(f"错误: map timestep_count {r['timestep_count']} != 清单 {e['num_timesteps']}: {r['member']}")
        if not _local_ok(h5_root, r):
            missing.append({"task": r["task"], "raw_ep_idx": int(e["raw_ep_idx"]), "g": int(e["global_episode_idx"]),
                            "es": int(e["exec_start_idx"]), "T": int(e["num_timesteps"]),
                            **{k: r[k] for k in ("difficulty", "episode", "member", "h5_sha256", "h5_bytes")},
                            "tar_object": tar_object(r)})
            continue
        path = h5_root / r["member"]
        if verify_sha:
            got = _sha256_file(path)
            if got != r["h5_sha256"]:
                raise SystemExit(f"错误: H5 sha256 不符: {path} {got} != {r['h5_sha256']}")
        single[(e["h5_file"], int(e["raw_ep_idx"]))] = {"path": path, "src_group": r["src_group"], "member": r["member"]}
    return {"picked": picked, "source": source, "single": single, "missing": missing, "arg": episodes_arg(picked)}


# ─────────────────────────── --acceptance 覆盖条件（G5） ───────────────────────────


def visible_windows(es: int, t: int, stride: int, window: int, demo_min_real: int) -> list[int]:
    """t 时刻合法 motion 起点（全域帧号，升序）——与 FrameSampMemory.visible_motion_frames / motion_store.visible_motion_rows 同式。"""
    out = list(range(0, max(es - (demo_min_real - 1), 0), stride))
    u = 0
    while u + (window - 1) <= t - es:
        out.append(es + u)
        u += stride
    return out


def acceptance_coverage(manifest: dict, picked: list[dict], motion_cfg, max_steps: int, max_points: int = 0) -> dict:
    """G5 覆盖条件（锁定清单逐项核，不补样）：
    ① 四任务各 ≥ 1 局（任务全集取自库 manifest）；② BinFill 局有补尾窗（demo 段 real < window 的起点，padded demo 窗）；
    ③ max_visible_k ≥ ACCEPT_MIN_K（各局最后一个决策时刻的可见窗数，k 随 t 单调）；④ 含全库 c_env 最长的一局
    （训练集最长局 c_env < 2000，「2000 步长局」只要求覆盖到真实最长局的最大 k）；⑤ exec_start_idx 互不相同；⑥ max_points == 0。
    返回 dict（含 ok、reasons、各计数）。"""
    stride, window = int(motion_cfg["stride"]), int(motion_cfg["window_frames"])
    dmin = int(motion_cfg.get("demo_min_real_frames", window))
    budget = int(motion_cfg["budget"])
    tasks_all = sorted({_task_of(e["h5_file"]) for e in manifest["episodes"]})
    tasks = sorted({_task_of(e["h5_file"]) for e in picked})
    es_list = [int(e["exec_start_idx"]) for e in picked]
    c_max = max(int(e["num_timesteps"]) - 1 - int(e["exec_start_idx"]) for e in manifest["episodes"])
    per = []
    for e in picked:
        es, T = int(e["exec_start_idx"]), int(e["num_timesteps"])
        taus = expected_taus(es, T, max_steps)
        last = taus[-1] if taus else es
        frames = visible_windows(es, last, stride, window, dmin)
        pad = sum(1 for f in frames if f < es and es - f < window)
        per.append({"task": _task_of(e["h5_file"]), "raw_ep_idx": int(e["raw_ep_idx"]), "es": es, "T": T,
                    "c_env": T - 1 - es, "last_tau": last, "k": len(frames), "tail_pad_windows": pad,
                    "k_over_budget": len(frames) > budget})
    tail_pad = sum(r["tail_pad_windows"] for r in per)
    binfill_pad = sum(r["tail_pad_windows"] for r in per if r["task"] == "BinFill")
    max_k = max((r["k"] for r in per), default=0)
    long_c = max((r["c_env"] for r in per), default=-1)
    reasons = []
    if tasks != tasks_all:
        reasons.append(f"tasks={len(tasks)}/{len(tasks_all)} 缺 {sorted(set(tasks_all) - set(tasks))}")
    if binfill_pad < 1:
        reasons.append("BinFill 补尾窗局缺失（BinFill 局 padded demo 窗数为 0）")
    if max_k < ACCEPT_MIN_K:
        reasons.append(f"max_visible_k={max_k} < {ACCEPT_MIN_K}")
    if long_c != c_max:
        reasons.append(f"未含全库最长局（清单最长 c_env={long_c}，全库 {c_max}）")
    if len(set(es_list)) != len(es_list):
        reasons.append(f"exec_start_idx 重复 {sorted(es_list)}")
    if max_points != 0:
        reasons.append(f"--max-points={max_points}（验收模式只能为 0）")
    return {"ok": not reasons, "reasons": reasons, "tasks": len(tasks), "tasks_total": len(tasks_all),
            "tail_pad_windows": tail_pad, "binfill_tail_pad_windows": binfill_pad, "max_visible_k": max_k,
            "long_c_env": long_c, "manifest_max_c_env": c_max, "es_unique": len(set(es_list)), "episodes": len(picked),
            "max_steps": max_steps, "per_episode": per}


def coverage_tag(cov: dict) -> str:
    """汇总行里的覆盖段：满足时写计划三节的字面 `tasks=4/4 tail_pad_windows>=1 max_visible_k=<k>`。"""
    pad = "tail_pad_windows>=1" if cov["binfill_tail_pad_windows"] >= 1 else f"tail_pad_windows={cov['tail_pad_windows']}"
    return f"tasks={cov['tasks']}/{cov['tasks_total']} {pad} max_visible_k={cov['max_visible_k']}"


def coverage_line(cov: dict) -> str:
    return (f"TIC_ACCEPTANCE_COVERAGE={'PASS' if cov['ok'] else 'FAIL'} {coverage_tag(cov)} "
            f"tail_pad_windows_n={cov['tail_pad_windows']} binfill_tail_pad_windows={cov['binfill_tail_pad_windows']} "
            f"long_c_env={cov['long_c_env']}/{cov['manifest_max_c_env']} es_unique={cov['es_unique']}/{cov['episodes']} "
            f"max_steps={cov['max_steps']} min_k={ACCEPT_MIN_K}" + ("" if cov["ok"] else f" reasons={cov['reasons']}"))


def explicit_flags(argv: list[str]) -> set[str]:
    """argv 里显式出现的长选项名（含 --flag=value 写法）。"""
    return {a.split("=", 1)[0] for a in argv if a.startswith("--")}


def check_acceptance_args(argv: list[str], args) -> None:
    """--acceptance：入口必须显式给齐 G4 checkpoint、库、framesamp-8x8 表、训练配置、norm stats、H5 映射、history YAML 等，不靠默认值。"""
    given = explicit_flags(argv)
    missing = [f for f in ACCEPT_REQUIRED_FLAGS if f not in given]
    if missing:
        raise AcceptanceError(f"验收模式缺显式参数 {missing}")
    if args.motion != "sidecar":
        raise AcceptanceError(f"验收模式必须 --motion sidecar（得到 {args.motion}）")
    if args.store_subdir != "framesamp-8x8":
        raise AcceptanceError(f"验收模式必须 --store-subdir framesamp-8x8（得到 {args.store_subdir}）")
    if args.max_points != 0:
        raise AcceptanceError(f"验收模式 --max-points 只能为 0（得到 {args.max_points}）")


def yaml_resolved_sha256(path) -> str:
    """与 train.py::init_history_config / check_checkpoint.resolved_sha256_of_yaml 同口径。"""
    import hashlib
    from omegaconf import OmegaConf
    return hashlib.sha256(OmegaConf.to_yaml(OmegaConf.load(str(path)), resolve=True).encode("utf-8")).hexdigest()


# ─────────────────────────── --perf-report（G5b 汇总） ───────────────────────────


def _pct(xs, q) -> float:
    return float(np.percentile(np.asarray(xs, np.float64), q)) if len(xs) else float("nan")


def perf_report(logs: list[str], gpu_csv: str | None, shards_per_gpu: int) -> tuple[bool, str, dict]:
    """汇总 MMEVLA_PERF_LOG 的 JSONL（每片一份）与 nvidia-smi 采样，给出 E2E_ONLINE_PERF 行。
    PASS 条件：日志份数 == shards_per_gpu、每份至少 1 次策略推理、无 error 事件；给了 GPU 采样且峰值 < 卡总显存
    （未超显存）。p95_ms 为全部片的单次策略推理耗时 p95；peak_gb 为该卡 memory.used 峰值（GiB，1024³）。"""
    infer_ms, add_ms, walls, req_bytes, ds_steps, ds_k, max_step = [], [], [], 0, 0, 0, 0
    per_log, reasons = [], []
    proc_peak = 0
    for path in logs:
        recs = [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        inf = [r for r in recs if r.get("event") == "infer"]
        eps = [r for r in recs if r.get("event") == "episode"]
        errs = [r for r in recs if r.get("event") == "error"]
        if not inf:
            reasons.append(f"{path} 无策略推理记录")
        if errs:
            reasons.append(f"{path} 有 error 事件 {len(errs)}")
        infer_ms += [float(r["infer_ms"]) for r in inf]
        done = {int(r["episode_idx"]) for r in eps}
        for r in eps:
            add_ms += [float(x) for x in r["add_buffer_ms"]]
            walls.append(float(r["wall_s"]))
            req_bytes += int(r.get("request_bytes") or 0)
            ds_steps += int(r.get("motion_downsample_steps") or 0)
            ds_k = max(ds_k, int(r.get("motion_downsample_max_k") or 0))
            max_step = max(max_step, int(r.get("last_step_idx") or 0))
            if r.get("peak_bytes_in_use"):
                proc_peak = max(proc_peak, int(r["peak_bytes_in_use"]))
        # 服务进程被外部结束、最后一局没有 episode 行时，用该局最后一条 infer 行补算墙钟与字节
        open_eps = {}
        for r in inf:
            if int(r["episode_idx"]) not in done:
                open_eps[int(r["episode_idx"])] = r
            if r.get("peak_bytes_in_use"):
                proc_peak = max(proc_peak, int(r["peak_bytes_in_use"]))
            max_step = max(max_step, int(r.get("step_idx") or 0))
        for r in open_eps.values():
            walls.append(float(r["t_rel_s"]))
            req_bytes += int(r.get("request_bytes") or 0)
        per_log.append({"path": str(path), "infer": len(inf), "episodes": len(eps), "open_episodes": len(open_eps)})
    if len(logs) != shards_per_gpu:
        reasons.append(f"日志份数 {len(logs)} != shards_per_gpu {shards_per_gpu}")
    peak_gb = total_gb = float("nan")
    if gpu_csv:
        used, total = [], []
        for line in pathlib.Path(gpu_csv).read_text(encoding="utf-8").splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2 and parts[0].replace(".", "", 1).isdigit():
                used.append(float(parts[0]))
                total.append(float(parts[1]))
        if not used:
            reasons.append(f"GPU 采样文件无有效行: {gpu_csv}")
        else:
            peak_gb, total_gb = max(used) / 1024.0, max(total) / 1024.0
            if max(used) >= max(total):
                reasons.append(f"该卡显存打满 peak={max(used):.0f}MiB total={max(total):.0f}MiB")
    else:
        reasons.append("缺 --perf-gpu-csv（该卡峰值显存无证据）")
    ok = not reasons
    line = (f"E2E_ONLINE_PERF={'PASS' if ok else 'FAIL'} p95_ms={_pct(infer_ms, 95):.1f} shards_per_gpu={shards_per_gpu} "
            f"peak_gb={peak_gb:.2f} | infer_p50_ms={_pct(infer_ms, 50):.1f} infer_n={len(infer_ms)} "
            f"add_buffer_p50_ms={_pct(add_ms, 50):.1f} add_buffer_p95_ms={_pct(add_ms, 95):.1f} "
            f"add_buffer_total_s={sum(add_ms) / 1000.0:.1f} episode_wall_s={[round(w, 1) for w in walls]} "
            f"request_mb={req_bytes / 1e6:.1f} motion_downsample_steps={ds_steps} motion_downsample_max_k={ds_k} "
            f"max_step_idx={max_step} proc_peak_gb={proc_peak / 1024 ** 3:.2f} card_total_gb={total_gb:.2f}"
            + ("" if ok else f" reasons={reasons}"))
    return ok, line, {"per_log": per_log, "reasons": reasons, "infer_ms": infer_ms, "add_buffer_ms": add_ms,
                      "episode_wall_s": walls, "request_bytes": req_bytes, "peak_gb": peak_gb, "card_total_gb": total_gb}


def report_plan(plan: dict, raw_dir: pathlib.Path) -> bool:
    """打印选中 episodes、缺失的 H5_EPISODE 行与齐全判定；返回是否齐全。"""
    single = plan["single"]
    for e in plan["picked"]:
        key = (e["h5_file"], int(e["raw_ep_idx"]))
        rec = {"task": _task_of(e["h5_file"]), "raw_ep_idx": key[1], "g": int(e["global_episode_idx"]),
               "es": int(e["exec_start_idx"]), "T": int(e["num_timesteps"])}
        if single is None:
            rec.update(source="merged-h5", path=str(raw_dir / e["h5_file"]), present=(raw_dir / e["h5_file"]).is_file())
        else:
            hit = single.get(key)
            rec.update(source="single-h5", member=hit["member"] if hit else None, present=hit is not None)
        print("EPISODE " + json.dumps(rec, ensure_ascii=False))
    for m in plan["missing"]:
        print("H5_EPISODE " + json.dumps(m, ensure_ascii=False))
    if single is None:
        complete = all((raw_dir / e["h5_file"]).is_file() for e in plan["picked"])
    else:
        complete = not plan["missing"]
    print(f"H5_COVERAGE={'COMPLETE' if complete else 'MISSING'} mode={'merged' if single is None else 'single'} "
          f"episodes={len(plan['picked'])} es={sorted(int(e['exec_start_idx']) for e in plan['picked'])} "
          f"missing={len(plan['missing'])} episodes_arg={plan['arg']}")
    print(f"EPISODES_SOURCE {plan['source']}")
    return complete


# ─────────────────────────── L3 / L4 模型侧 jit 函数（按 integration_type 与 history_pi0 同构） ───────────────────────────
# context：记忆在 embed_prefix 里，前缀 pass 用 na_mask（sample_actions 的 context 分支）——旧写法逐字不变。
# modulation：记忆不进前缀，由 embed_memory 产出 mem_seq / mem_mask，只给 action expert（第 2 个流）：
#   compute_loss  → llm([prefix, suffix], adarms_cond=[None, ad], mem_seq=[None, mem_seq], mem_mask=[None, mem_mask])，attn 带 na_mask；
#   sample_actions → 前缀 pass make_attn_mask(prefix_mask, prefix_ar_mask)（不带 na）、llm([prefix, None])，
#                    去噪步 llm([None, suffix], kv_cache, adarms_cond=[None, ad], mem_seq=[None, mem_seq], mem_mask=[None, mem_mask])。
# 两侧（训练 obs / 推理 obs）各自走 preprocess(train=False) → embed_memory，不共享记忆张量。


def make_l3_fn(gdef, integration_type: str, motion_enabled: bool):
    """L3：preprocess(train=False) → embed_prefix → attn_mask / positions → 前缀 pass 取 KV；另出并列序记忆。"""
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from mme_vla_suite.models.integration.history_observation import preprocess_observation
    from mme_vla_suite.models.integration.history_pi0 import make_attn_mask

    @jax.jit
    def f_prefix(state, obs):
        m = nnx.merge(gdef, state)
        o = preprocess_observation(None, obs, train=False)
        tok, mask, ar, na = m.embed_prefix(o)
        if integration_type == "modulation":
            attn = make_attn_mask(mask, ar)                  # sample_actions modulation 分支：前缀不带 na
        else:
            attn = make_attn_mask(mask, ar, na)
        pos = jnp.cumsum(mask, axis=1) - 1
        _, kv = m.PaliGemma.llm([tok, None], mask=attn, positions=pos)
        par_tok, _, _ = m.mem_encoder(o.static_image_emb, o.static_pos_emb, o.static_state_emb,
                                      motion_emb=o.motion_emb, motion_pos=o.motion_pos, motion_mask=o.motion_mask)
        par_mask = jnp.concatenate([o.static_mask, o.motion_mask], axis=1) if motion_enabled else o.static_mask
        mem_tok, mem_mask, _, _ = m.embed_memory(o)
        return {"tokens": tok, "mask": mask, "ar": ar, "na": na, "attn": attn, "pos": pos,
                "k": kv[0], "v": kv[1], "par_tok": par_tok, "par_mask": par_mask,
                "mem_tok": mem_tok, "mem_mask": mem_mask,
                "txt_len": jnp.int32(o.tokenized_prompt.shape[1])}

    return f_prefix


def make_l4_fns(gdef_, integration_type: str):
    """按给定 graphdef 生成 L4 的两路 jit 函数（bf16 生产权重与 f32 诊断权重各一套）。"""
    import einops
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from mme_vla_suite.models.integration.history_observation import preprocess_observation
    from mme_vla_suite.models.integration.history_pi0 import make_attn_mask
    modulation = integration_type == "modulation"

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
        if modulation:
            mem_seq, mem_mask, _, _ = m.embed_memory(o)
            (_, so), kv = m.PaliGemma.llm([pt, st], mask=attn, positions=pos, adarms_cond=[None, ad],
                                          mem_seq=[None, mem_seq], mem_mask=[None, mem_mask])
        else:
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
        if modulation:
            pattn = make_attn_mask(pm, par_)
        else:
            pattn = make_attn_mask(pm, par_, pna)
        ppos = jnp.cumsum(pm, axis=1) - 1
        _, kv = m.PaliGemma.llm([pt, None], mask=pattn, positions=ppos)
        if modulation:
            mem_seq, mem_mask, _, _ = m.embed_memory(o)
        st, sm, sar, _, ad = m.embed_suffix(o, x_t, tstep)
        sattn = make_attn_mask(sm, sar)
        prefix_attn = einops.repeat(pm, "b p -> b s p", s=st.shape[1])
        full = jnp.concatenate([prefix_attn, sattn], axis=-1)
        spos = jnp.sum(pm, axis=-1)[:, None] + jnp.cumsum(sm, axis=-1) - 1
        if modulation:
            (_, so), _ = m.PaliGemma.llm([None, st], mask=full, positions=spos, kv_cache=kv, adarms_cond=[None, ad],
                                         mem_seq=[None, mem_seq], mem_mask=[None, mem_mask])
        else:
            (_, so), _ = m.PaliGemma.llm([None, st], mask=full, positions=spos, kv_cache=kv, adarms_cond=[None, ad])
        v_t = m.action_out_proj(so[:, -m.action_horizon:])
        return {"v_t": v_t, "prefix_attn": pattn, "full_attn": full, "spos": spos,
                "pmask": pm, "k": kv[0], "v": kv[1]}

    return f_full, f_cached


# ─────────────────────────── 主流程 ───────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", default=str(_V1 / "datasets/4task-motion-400ep"), help="对拍用的库根（含 framesamp/ 与 motion/）")
    ap.add_argument("--ckpt", default=str(_V1 / "train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999"),
                    help="checkpoint 目录（含 params/ 与 assets/）")
    ap.add_argument("--train-config", default="mme_vla_suite", help="生产训练配置条目（两侧同用）")
    ap.add_argument("--store-subdir", choices=("framesamp", "framesamp-8x8"), default="framesamp")
    ap.add_argument("--motion-subdir", default="motion", help="运动表子目录（默认 motion；wan_raw9 为 motion-wan-raw9-s32）")
    ap.add_argument("--norm-stats", type=pathlib.Path, help="训练启动时显式指定的 norm_stats 文件")
    ap.add_argument("--episodes", default=None,
                    help="逗号分隔的 <task>:<raw_ep_idx>，需覆盖 5 个 es 取值；不给时：无 --h5-root 用 DEFAULT_EPISODES，"
                         "有 --h5-root 从本机已有单 episode H5 自动选")
    ap.add_argument("--h5-root", default=None, help="单 episode H5 根（map 行 member 相对它）；不给则读清单 raw_dir 下的合并 H5")
    ap.add_argument("--episode-map-dir", default=None, help="record_dataset_<task>_episode_map.json 所在目录（--h5-root 必需）")
    ap.add_argument("--dry-run", action="store_true", help="只打印选中的 episodes 与 H5 是否齐全，不建 policy、不用 GPU")
    ap.add_argument("--motion", choices=("store", "sidecar"), default="store",
                    help="运动路来源：store=从 MotionStore 查表；sidecar=S 臂用真 MotionEncoderClient 现算")
    ap.add_argument("--motion-gpu", default=None, help="sidecar 子进程的 CUDA_VISIBLE_DEVICES（物理卡号，必须显式指定）")
    ap.add_argument("--noise-seeds", default="0,1,2")
    ap.add_argument("--max-points", type=int, default=0, help="开发用：每集只跑前 N 个决策时刻（0 = 全部）")
    ap.add_argument("--f32-diag-points-per-episode", type=int, default=1,
                    help="每集取前 N 个决策点做 f32 阻断验证；正式验收传 3，0 会因缺证据失败")
    ap.add_argument("--out", default=str(_V1 / "reports/tic/compare_train_infer_obs.json"), help="输出 json 路径")
    ap.add_argument("--acceptance", action="store_true",
                    help="G5 验收模式：锁定覆盖清单、全部入口显式给出、禁止回退与自动补样（见模块说明）")
    ap.add_argument("--history-yaml", default=None,
                    help="history YAML（如 …-8x8-e2eenc.yaml）；给出时其 resolved sha256 必须等于 run 快照（验收模式必需）")
    ap.add_argument("--perf-report", nargs="+", default=None,
                    help="G5b：汇总各片 MMEVLA_PERF_LOG JSONL，打印 E2E_ONLINE_PERF 行后退出（不建 policy）")
    ap.add_argument("--perf-gpu-csv", default=None, help="G5b：该卡 nvidia-smi memory.used,memory.total 采样（csv,noheader,nounits）")
    ap.add_argument("--shards-per-gpu", type=int, default=4, help="G5b：同卡并发片数（4；三片退路为 3）")
    args = ap.parse_args()

    if args.perf_report:
        ok, line, detail = perf_report(args.perf_report, args.perf_gpu_csv, int(args.shards_per_gpu))
        print(line, flush=True)
        if "--out" in explicit_flags(sys.argv):
            pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(args.out).write_text(json.dumps({"line": line, **detail}, ensure_ascii=False, indent=1) + "\n",
                                             encoding="utf-8")
        return 0 if ok else 1
    if args.acceptance:
        check_acceptance_args(sys.argv, args)

    os.chdir(_REPO_ROOT)      # train_config.data.assets.assets_dir 是仓库相对路径，必须从仓库根解析
    lib = pathlib.Path(args.lib).resolve()
    ckpt_dir = pathlib.Path(args.ckpt).resolve()
    run_root = ckpt_dir.parent
    noise_seeds = [int(s) for s in args.noise_seeds.split(",")]
    os.environ["MMEVLA_MOTION_STORE"] = str(lib / args.motion_subdir)     # 快照 store_path 记的是 40ep，必须显式覆盖
    h5_root = pathlib.Path(args.h5_root).resolve() if args.h5_root else None
    map_dir = pathlib.Path(args.episode_map_dir).resolve() if args.episode_map_dir else None
    manifest0 = json.load(open(lib / "meta" / "episode_manifest.json", encoding="utf-8"))
    plan = plan_episodes(manifest0, args.episodes, h5_root, map_dir, strict=args.acceptance)
    coverage = None
    if args.history_yaml:
        import omegaconf as _oc
        _hy = _oc.OmegaConf.load(args.history_yaml)
        _hy_sha = yaml_resolved_sha256(args.history_yaml)
        _snap = run_root / "history_config.resolved.sha256"
        if _snap.is_file() and _snap.read_text().split()[0] != _hy_sha:
            msg = f"--history-yaml resolved sha256 {_hy_sha[:16]}… != run 快照 {_snap.read_text().split()[0][:16]}…"
            if args.acceptance:
                raise AcceptanceError(msg)
            raise SystemExit("错误: " + msg)
        if args.acceptance and not _snap.is_file():
            raise AcceptanceError(f"run 缺 history_config.resolved.sha256: {run_root}")
        print(f"HISTORY_YAML sha256={_hy_sha[:16]} yaml={args.history_yaml} snapshot_equal=1")
        if args.acceptance:
            _mc = _hy.get("motion", None)
            if _mc is None or not _mc.get("enabled", False):
                raise AcceptanceError("验收模式的 history YAML 必须开启 motion")
            coverage = acceptance_coverage(manifest0, plan["picked"], _mc, max_steps_for(_mc), args.max_points)
            print(coverage_line(coverage), flush=True)
            if not coverage["ok"]:
                raise AcceptanceError("覆盖条件不满足: " + "; ".join(coverage["reasons"]))
    if args.dry_run or h5_root is not None:
        complete = report_plan(plan, pathlib.Path(manifest0["raw_dir"]))
        if args.dry_run:
            print(f"DRY_RUN episodes={len(plan['picked'])} complete={'yes' if complete else 'no'} episodes_arg={plan['arg']}")
            return 0 if complete else 1
        if not complete:
            print("错误: 单 episode H5 不齐，见上方 H5_EPISODE 行（补下后重跑）")
            return 1

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
    if h5_root is not None:
        rec["h5_source"] = {"h5_root": str(h5_root), "episode_map_dir": str(map_dir), "episodes_arg": plan["arg"],
                            "members": [v["member"] for v in plan["single"].values()]}

    # ── 生产口径 policy（B 臂编码器 + transforms + norm_stats 全部来自 checkpoint）──────────────────
    t0 = time.perf_counter()
    train_config = _config.get_config(args.train_config)
    if args.norm_stats:
        train_config = dataclasses.replace(train_config, data=dataclasses.replace(
            train_config.data, assets=dataclasses.replace(train_config.data.assets,
                assets_dir=str(args.norm_stats.resolve().parent.parent), asset_id=args.norm_stats.parent.name)))
    # 构造期 stub 禁止看到任何 GPU，真 sidecar 在后文显式指定物理卡号。
    from unittest import mock
    from mme_vla_suite.policies import motion_client as mc
    real_client = mc.MotionEncoderClient
    def cpu_stub(**kw):
        return real_client(**(kw | {"online_gpu": ""}))
    with mock.patch.object(mc, "MotionEncoderClient", cpu_stub):
        policy = _policy_config.create_trained_policy(train_config, ckpt_dir, motion_stub=True)
    motion_enabled = bool(policy.motion_enabled)
    memory_keys = KEYS8 if motion_enabled else KEYS8[:4]
    stub_client = policy._motion_client
    if stub_client is not None:
        stub_client.close()                                     # 只借它过构造，随后按档位换成查表闭包或真 sidecar
    enc_B = policy._vision_encode
    model = policy._model
    print(f"[tic] policy 构造 {time.perf_counter() - t0:.1f}s config={args.train_config} ckpt={ckpt_dir} "
          f"use_quantiles={policy.use_quantiles} action_horizon={model.action_horizon} action_dim={model.action_dim}")

    # ── 训练侧真口径 dataset：data_config 由生产配置 create，raw / transformed 两层都建 ────────────
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    hc = omegaconf.OmegaConf.load(run_root / "history_config.resolved.yaml")
    ds_raw = _create_framesamp_dataset(str(lib / args.store_subdir), data_config, hc, int(model.action_horizon))
    ds_tf = transform_dataset(ds_raw, data_config)
    fmeta = StoreMeta.load(str(lib / args.store_subdir))
    motion_root = _motion_gates(hc, fmeta)
    mmeta = ms.MotionMeta.load(motion_root) if motion_enabled else None
    mstore = ms.MotionStore(motion_root, meta=mmeta) if motion_enabled else None
    # wan_raw9 表行是 (9,16,32,32)，在线缓冲存的是展平后的 (9,16384)；其余 kind 行即 token，原样返回
    raw9_kind = motion_enabled and hc.motion.get("encoder_kind", None) == "wan_raw9"
    e2e_kind = motion_enabled and hc.motion.get("encoder_kind", None) == E2E_KIND
    if raw9_kind:
        from mme_vla_suite.shared import wan_raw9 as _wan_raw9
    max_steps = max_steps_for(hc.motion if motion_enabled else None)
    if args.acceptance and not e2e_kind:
        raise AcceptanceError(f"验收模式只用于 e2e 快照（motion.encoder_kind={hc.motion.get('encoder_kind', None) if motion_enabled else None}）")

    def _motion_row(r: int) -> np.ndarray:
        row = mstore.rows(np.asarray([int(r)], dtype=np.int64))[0]
        if e2e_kind:
            return e2e_row_view(row)
        return _wan_raw9.flatten_raw9(row) if raw9_kind else row
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
    picked, ep_source = plan["picked"], plan["source"]
    eps = build_points(picked, args.max_points, max_steps)
    n_points = sum(len(e["taus"]) for e in eps)
    entries = ds_raw._motion_entries
    n_windows = 0
    k_actual_max = 0
    for e in eps:
        if motion_enabled:
            rows_last, _ = ms.visible_motion_rows(entries[e["g"]], e["taus"][-1])
            n_windows += int(len(rows_last))
            k_actual_max = max(k_actual_max, int(len(rows_last)))
        for t in e["taus"]:
            hit = np.flatnonzero((ds_raw._epis_of == e["g"]) & (ds_raw._step_of == t))
            if len(hit) != 1:
                raise SystemExit(f"错误: 决策时刻 (g={e['g']}, t={t}) 在 FrameSampDataset 中命中 {len(hit)} 条样本")
    print(f"POINTS episodes={len(eps)} points={n_points} windows={n_windows} "
          f"c_mod16={[e['c_mod16'] for e in eps]} max_steps={max_steps}")
    print(f"EPISODES source={ep_source} " + " ".join(
        f"{e['h5_file'].replace('record_dataset_', '').replace('.h5', '')}:{e['raw_ep_idx']}"
        f"(g={e['g']},es={e['es']},T={e['T']},pts={len(e['taus'])})" for e in eps))
    rec["points"] = {"episodes": len(eps), "points": n_points, "windows": n_windows, "source": ep_source,
                     "episodes_detail": eps}

    # ── 模型侧 jit 函数（冻结一次 state，与 nnx_utils.module_jit 同法，避免 nnx.jit 的额外开销）──────
    gdef, gstate = nnx.split(model)

    def _obs_of(inputs_np: dict):
        return HistAugObservation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs_np))

    integration_type = str(model.integration_type)
    f_prefix = make_l3_fn(gdef, integration_type, motion_enabled)

    def _mk_l4(gdef_):
        return make_l4_fns(gdef_, integration_type)

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
    if motion_enabled and args.motion == "sidecar":
        if args.motion_gpu is None:
            raise ValueError("sidecar 模式必须显式给 --motion-gpu")
        from mme_vla_suite.policies.motion_client import MotionEncoderClient
        t0 = time.perf_counter()
        if e2e_kind:
            # e2e：握手比 run 快照的 kind / readout 与 run motion_provenance.json 的 raw9_source_sha256；vae 段比表 provenance（缺则用 run 的）
            run_prov = json.loads((run_root / "motion_provenance.json").read_text(encoding="utf-8"))
            client = MotionEncoderClient(online_gpu=args.motion_gpu,
                                         store_provenance={"vae": mmeta.provenance.get("vae") or run_prov.get("vae")},
                                         encoder_kind=E2E_KIND, readout=str(hc.motion.readout),
                                         expected_raw9_sha256=run_prov["raw9_source_sha256"], keep_last_raw_f32=True)
            client = E2EConvertCheck(client)
        elif raw9_kind:
            client = MotionEncoderClient(online_gpu=args.motion_gpu, store_provenance={"vae": mmeta.provenance.get("vae")},
                                         encoder_kind="wan_raw9", readout=str(hc.motion.readout),
                                         expected_raw9_sha256=mmeta.provenance["raw9_source_sha256"])
        else:
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
    trainset_key_mis = {k: 0 for k in memory_keys}
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
    e2e_windows: list[np.ndarray] = []           # E2E_ENC_TRAIN_VS_ONLINE 用的 S 臂 bf16 槽（sidecar 档）
    e2e_rec_line = None

    try:
        for ep in eps:
            g, es, T, row_base = ep["g"], ep["es"], ep["T"], ep["row_base"]
            taus = ep["taus"]
            if motion_enabled:
                entry = entries[g]
                rows_all, f_all = ms.visible_motion_rows(entry, T - 1)
                f2row = {int(f): int(r) for r, f in zip(rows_all.tolist(), f_all.tolist())}
            else:
                f2row = {}

            def motion_lookup(window, start_frame, _f2row=f2row):
                if e2e_kind:                    # e2e：表里的 bf16 字节原样给出，不做 dtype 转换（R9）
                    return _motion_row(_f2row[int(start_frame)])
                return np.asarray(_motion_row(_f2row[int(start_frame)]), dtype=np.float32)

            n_read = min(T, taus[-1] + 1)
            frames, wrists, states, task_goal = load_episode_full(raw_dir, ep["h5_file"], ep["raw_ep_idx"], T, n_read,
                                                                  single=None if plan["single"] is None else
                                                                  plan["single"][(ep["h5_file"], ep["raw_ep_idx"])])

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
                    mems["S"]._history_feats[s][fmeta.spec.image_key] = np.ascontiguousarray(rows[i][None])
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
                    e_arm = {n: np.asarray(mems[n]._history_feats[s][fmeta.spec.image_key])[0] for n in FRAME_ARMS}
                    for p in MEM_PAIRS:
                        frame_acc[p].add(e_arm[p[0]], e_arm[p[1]])
                if client is not None:
                    for f in mems["S"].visible_motion_frames(t):
                        if f in checked_motion:
                            continue
                        checked_motion.add(f)
                        if e2e_kind:            # e2e：bf16 对 bf16 逐字节（dtype 也须相同）
                            on = np.asarray(mems["S"]._history_feats_motion[f])
                            off = np.asarray(_motion_row(f2row[int(f)]))
                            if len(e2e_windows) < E2E_REC_WINDOWS:
                                e2e_windows.append(np.array(on))
                        else:
                            on = np.asarray(mems["S"]._history_feats_motion[f], np.float32)
                            off = np.asarray(_motion_row(f2row[int(f)]), np.float32)
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
                for k in memory_keys:
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
                # 只有 context 把记忆拼进前缀；modulation 前缀 = 图像 + 文本，记忆走 mem_seq（见 make_l3_fn 注释）
                img_len = total_len - (mem_len if integration_type == "context" else 0) - txt_len
                n_img = len(infer_inputs["image"])
                want_mem = int(hc.budget) + (int(hc.motion.budget) * int(hc.motion.get("tokens_per_window", 1))
                                             if motion_enabled else 0)
                shape_ok = (mem_len == want_mem and txt_len == int(model.config.max_token_len)
                            and img_len > 0 and img_len % n_img == 0)
                if not shape_ok:
                    prefix_shape_ok = False
                prefix_shape_info.update({"len": total_len, "mem": mem_len, "img": img_len, "txt": txt_len,
                                          "img_per_view": img_len // n_img if n_img else None, "views": n_img,
                                          "want_mem": want_mem})
                # modulation 的记忆不在前缀叶里，另把 embed_memory 两返回（即 llm 的 mem_seq / mem_mask）并入同一计数
                for k in ("tokens", "mask", "ar", "na") + (("mem_tok", "mem_mask") if integration_type == "modulation" else ()):
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
                order = np.asarray(assembled_S["mem_order"], np.int64) if motion_enabled else np.arange(mem_len, dtype=np.int64)
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
                mo_mask = np.asarray(assembled_S["motion_mask"], bool) if motion_enabled else np.zeros(0, dtype=bool)
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
                for k in memory_keys:
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

        # ── f32 阻断验证：参数树与 embed_dtype 一起升 f32 后，在同一输入上重跑两路 ──
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
            f32_ok = s32["rel_fro"] <= 1e-6 and acc_kv32["max_abs"] == 0 and len(f32_cache) > 0
            f32_line = (f"VT_FULL_VS_CACHED_F32={'PASS' if f32_ok else 'FAIL'} points={len(f32_cache)} rel_fro={s32['rel_fro']:.4g} "
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

        # ── E2E_ENC_TRAIN_VS_ONLINE（只记录、不判）：同一批窗口，训练精度路径（盘上 f32 参数、批量）vs 在线路径（bf16 restore、逐窗）
        if e2e_kind and e2e_windows and not prompt_fail:
            t0 = time.perf_counter()
            try:
                params_tr = _model.restore_params(ckpt_dir / "params", dtype=None)
                tc_tr = dataclasses.replace(train_config, model=dataclasses.replace(
                    train_config.model, history_config=hc, use_history=True))
                model_tr = tc_tr.model.load(params_tr, remove_extra_params=False)
                rel, n_win = e2e_encoder_train_vs_online(model_tr, model, e2e_windows)
                e2e_rec_line = (f"E2E_ENC_TRAIN_VS_ONLINE=RECORDED rel_rmse={rel:.4g} windows={n_win} "
                                f"train=f32_params_batched online=bf16_restore_per_window wall_s={time.perf_counter() - t0:.1f}")
                del params_tr, model_tr
                gc.collect()
            except Exception as exc:  # 只记录项：测不出时如实写出原因，不影响阻断判定
                e2e_rec_line = f"E2E_ENC_TRAIN_VS_ONLINE=RECORDED rel_rmse=nan windows={len(e2e_windows)} error={type(exc).__name__}:{exc}"

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
        if mstore is not None:
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
                         f"keys={len(memory_keys)} key_mismatches={key_mis or 'none'}")
    for p in MEM_PAIRS:
        V.observe(frame_acc[p].line(f"MEM_{p[0]}_VS_{p[1]}"))
    if client is not None:
        ok = motion_sidecar_cmp[1] == 0 and motion_sidecar_cmp[0] > 0
        V.block(ok, f"MOTION_S_VS_SIDECAR={'PASS' if ok else 'FAIL'} windows={motion_sidecar_cmp[0]} "
                    f"mismatches={motion_sidecar_cmp[1]} provenance_keys_equal=1 lib={lib.name} "
                    f"policy=create_trained_policy motion_client=MotionEncoderClient(online_gpu={args.motion_gpu})")
    if isinstance(client, E2EConvertCheck):
        cv_ok = client.checked > 0 and client.mismatches == 0 and client.fn_ok
        V.block(cv_ok, f"E2E_BF16_CONVERT={'PASS' if cv_ok else 'FAIL'} windows={client.checked} mismatches={client.mismatches} "
                       f"fn_is_raw9_f32_to_bf16={int(client.fn_ok)} fn=mme_vla_suite.shared.raw_bf16.raw9_f32_to_bf16")
    if coverage is not None:
        k_ok = k_actual_max == coverage["max_visible_k"]
        V.block(coverage["ok"] and k_ok,
                coverage_line(coverage).replace("TIC_ACCEPTANCE_COVERAGE=PASS", "TIC_ACCEPTANCE_COVERAGE=" + ("PASS" if k_ok else "FAIL"))
                + f" k_closed_form_vs_index={'equal' if k_ok else str(coverage['max_visible_k']) + '!=' + str(k_actual_max)}")
    if e2e_rec_line is not None:
        V.observe(e2e_rec_line)
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
                     "FULL_VS_CACHED_STRUCT", "VT_FULL_VS_CACHED_F32", "ACT_T_VS_I"):
            V.block(False, f"{name}=SKIP(OBS_PROMPT_FAIL)")
        V.observe("MEM_SCALE_OBS=SKIP(OBS_PROMPT_FAIL)")
        V.observe("VT_FULL_VS_CACHED=SKIP(OBS_PROMPT_FAIL)")
        V.observe("ACT_S_VS_B=SKIP(OBS_PROMPT_FAIL)")
        V.observe("AUG_EFFECT_OBS=SKIP(OBS_PROMPT_FAIL)")
    else:
        i = prefix_shape_info
        V.block(prefix_shape_ok, f"PREFIX_SHAPE={'PASS' if prefix_shape_ok else 'FAIL'} len={i.get('len')} "
                                 f"mem={i.get('mem')} img={i.get('img')} txt={i.get('txt')} "
                                 f"img_per_view={i.get('img_per_view')} views={i.get('views')} want_mem={i.get('want_mem')}")
        V.block(prefix_leaf_mis == 0, f"PREFIX_T_VS_I={'PASS' if prefix_leaf_mis == 0 else 'FAIL'} points={n_done} "
                                      f"leaves={6 if integration_type == 'modulation' else 4} mismatches={prefix_leaf_mis or 'none'}")
        V.block(pos_mis == 0 and attn_mis == 0,
                f"PREFIX_POSITIONS={'PASS' if pos_mis == 0 and attn_mis == 0 else 'FAIL'} positions_mismatches={pos_mis} "
                f"attn_mask_mismatches={attn_mis} attn_mask_shape={i.get('attn_shape')} na_branch={0 if integration_type == 'modulation' else 1}")
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
        V.observe(f"VT_FULL_VS_CACHED={'PASS' if vt_ok else 'FAIL'} points={n4} rel_fro={s4['rel_fro']:.4g} "
                       f"max_abs={s4['max_abs']:.4g} vt_rms={_mean_str(l4_vt_rms)} "
                       f"ulp_p99={s4['ulp_p99']:.3g} ulp_max={s4['ulp_max']:.3g} "
                       f"thr_rel_fro={THR_REL_FRO} thr_ulp_p99={THR_ULP_P99} | prefix_kv {_num_line(l4_kv)} "
                       f"(prefix_kv 只统计 prefix_mask=True 的位) "
                       f"| note=用户已裁决为bf16观察项；阻断条件见VT_FULL_VS_CACHED_F32")
        if f32_line is not None:
            V.block(f32_ok, f32_line)
        else:
            V.block(False, "VT_FULL_VS_CACHED_F32=FAIL 缺f32重跑证据")
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
    if coverage is not None:
        # 验收模式汇总行（计划三节 G5）：TIC_SIDECAR=PASS blocking=<n>/<n> tasks=4/4 tail_pad_windows>=1 max_visible_k=<k>
        final = (f"{tag}={'PASS' if V.all_ok else 'FAIL'} blocking={V.block_pass}/{V.block_total} {coverage_tag(coverage)} "
                 f"episodes={len(eps)} points={n_done} {extra} observe={V.n_observe} max_steps={max_steps} mode=acceptance")
    else:
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


E2E_REC_WINDOWS = 32          # E2E_ENC_TRAIN_VS_ONLINE 取的窗数上限


def e2e_row_view(row: np.ndarray) -> np.ndarray:
    """e2e 表行（bf16 (9,16,32,32)）→ (9,16384) 视图：零拷贝、不改 dtype（R9：数据集与比对侧只搬运 bf16 字节）。"""
    if row.dtype.name != "bfloat16" or tuple(row.shape) != (9, 16, 32, 32):
        raise RuntimeError(f"e2e 表行形制 {row.shape} {row.dtype} != (9,16,32,32) bfloat16")
    out = np.ascontiguousarray(row).reshape(9, 16384)
    return out


class E2EConvertCheck:
    """包住 e2e MotionEncoderClient：每窗核客户端返回的 bf16 与 raw9_f32_to_bf16(同窗 fp32) 逐字节相等，
    并核客户端持有的转换函数对象就是 mme_vla_suite.shared.raw_bf16.raw9_f32_to_bf16（R9 唯一转换点）。"""

    def __init__(self, client):
        from mme_vla_suite.shared import raw_bf16
        self._client = client
        self._fn = raw_bf16.raw9_f32_to_bf16
        self.fn_ok = getattr(client, "_bf16_fn", None) is raw_bf16.raw9_f32_to_bf16
        self.checked = 0
        self.mismatches = 0

    def __call__(self, frames, start_frame=0):
        out = self._client(frames, start_frame)
        ref = self._fn(self._client.last_raw_f32)
        self.checked += 1
        if not _bytes_equal(out, ref):
            self.mismatches += 1
        return out

    def __getattr__(self, name):
        return getattr(self._client, name)

    def close(self):
        self._client.close()


def _find_e2e_encoder(model):
    """在 nnx 模型里按类名找 E2E encoder 子模块（S1 的 WanLatentMotionEncoderJax；属性名由 S3 定，不写死路径）。"""
    hits = [(path, m) for path, m in model.iter_modules() if type(m).__name__ == "WanLatentMotionEncoderJax"]
    if len(hits) != 1:
        raise RuntimeError(f"模型里 WanLatentMotionEncoderJax 子模块数 {len(hits)} != 1")
    return hits[0][1]


def e2e_encoder_train_vs_online(model_train, model_online, windows: list[np.ndarray]) -> tuple[float, int]:
    """同一批 bf16 (9,16384) 窗：训练精度路径（f32 参数、一次批量 (n,9,16384)）vs 在线路径（bf16 参数、逐窗 (1,9,16384)）
    的 encoder 输出 rel_rmse = ||on − tr|| / ||tr||（只记录）。"""
    import jax
    import jax.numpy as jnp
    enc_tr = _find_e2e_encoder(model_train)
    enc_on = _find_e2e_encoder(model_online)
    x = np.stack(windows)
    tr = np.asarray(jax.block_until_ready(enc_tr(jnp.asarray(x))), np.float32)
    on = np.concatenate([np.asarray(jax.block_until_ready(enc_on(jnp.asarray(x[i:i + 1]))), np.float32)
                         for i in range(len(x))], axis=0)
    rel = float(np.sqrt(np.sum((on.astype(np.float64) - tr) ** 2) / max(np.sum(tr.astype(np.float64) ** 2), 1e-30)))
    return rel, len(x)


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
