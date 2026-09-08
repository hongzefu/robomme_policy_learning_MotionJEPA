#!/usr/bin/env python3
"""donor bank：swap 条件的「同任务、别的 episode」motion 内容来源（计划第二部分 §4，用户拍板契约）。

来源（本轮只实现 library）：训练集 motion 库 v1-store/datasets/4task-motion-400ep/motion 全量 6832 行 × 768。

查表契约 `lookup(split, task, recv_ep, seg, m)`：
  1. 主 donor（hash(split|task|recv_ep) 固定选、四个 policy seed 共用）有该 (seg, m) → how="exact"；
  2. 否则沿同任务兜底链（按 exec 段窗数降序、并列按 g 升序）找第一个有 (seg, m) 的 donor → how="fallback"；
  3. 全库都没有该偏移 → 在兜底链第一个 donor 的该段内 m % n 循环 → how="cycle"；
  4. 该任务全库无此段（Button 类无 demo 段）而接收方却有 → how="cross_seg"（应为 0）。
阻断只有 self_loops==0（接收方在 test/val，与训练集 donor 天然无自环；开环用 recv_g 排除自身）与 cross_seg==0；
exact/fallback/cycle 比例按实际推理次数实测报并分层（不设 wrap 阈值——Codex 审计 1）。

落盘：<bank>/donor_bank.npz（tokens f32[R,768]）+ <bank>/donor_bank.json（segments / primary / fallback_chain / provenance / sha256）。
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import mv_common as C  # noqa: E402

SEGS = ("demo", "exec")
OPEN_LOOP_SEEDS = (0, 1, 2, 3, 4)


def _pick(key: str, n: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % n


class DonorBank:
    def __init__(self, bank_dir: pathlib.Path):
        self.dir = pathlib.Path(bank_dir)
        self.meta = json.loads((self.dir / "donor_bank.json").read_text(encoding="utf-8"))
        self.tokens = np.load(self.dir / "donor_bank.npz")["tokens"]
        if self.tokens.shape != tuple(self.meta["tokens_shape"]):
            raise SystemExit(f"错误: bank tokens 形状 {self.tokens.shape} != json {self.meta['tokens_shape']}")
        sha = hashlib.sha256(np.ascontiguousarray(self.tokens).tobytes()).hexdigest()
        if sha != self.meta["sha256"]:
            raise SystemExit(f"错误: bank tokens sha256 不符（产物腐烂）")
        self.segments: dict[str, dict] = self.meta["segments"]
        self.primary: dict[str, int] = self.meta["primary"]
        self.chain: dict[str, list[int]] = self.meta["fallback_chain"]
        self.es_of: dict[str, int] = self.meta["es_of"]
        self.sha256 = sha

    @property
    def source(self) -> str:
        return self.meta["source"]

    def seg(self, task: str, g: int, seg: str) -> dict | None:
        return self.segments.get(f"{task}|{g}|{seg}")

    def primary_of(self, split: str, task: str, recv_ep: int) -> int:
        key = f"{split}|{task}|{recv_ep}"
        if key not in self.primary:
            raise KeyError(f"bank 没有接收方 {key} 的主 donor（建库时 splits/episodes 不含它）")
        return int(self.primary[key])

    def lookup(self, split: str, task: str, recv_ep: int, seg: str, m: int):
        """返回 (tok f32[768], how, donor_g)。split 可为 'ol<seed>'（开环）。"""
        g0 = self.primary_of(split, task, recv_ep)
        s = self.seg(task, g0, seg)
        if s is not None and m < s["n"]:
            return self.tokens[s["row_base"] + m], "exact", g0
        for g in self.chain[task]:
            s = self.seg(task, g, seg)
            if s is not None and m < s["n"]:
                return self.tokens[s["row_base"] + m], "fallback", int(g)
        for g in self.chain[task]:
            s = self.seg(task, g, seg)
            if s is not None and s["n"] > 0:
                return self.tokens[s["row_base"] + (m % s["n"])], "cycle", int(g)
        # 该任务全库无此段
        other = "exec" if seg == "demo" else "demo"
        for g in [g0] + self.chain[task]:
            s = self.seg(task, g, other)
            if s is not None and s["n"] > 0:
                return self.tokens[s["row_base"] + (m % s["n"])], "cross_seg", int(g)
        raise RuntimeError(f"bank 里任务 {task} 没有任何窗口")

    def coverage_estimate(self, splits=("test", "val")) -> dict:
        """按接收方最坏情况 t = es + MAX_STEPS 估各档比例（es 用主 donor 的 es 作代理；观察量）。"""
        cnt = {h: 0 for h in ("exact", "fallback", "cycle", "cross_seg")}
        by_task: dict[str, dict] = {}
        for split in splits:
            for task in C.TASKS:
                bt = by_task.setdefault(task, {h: 0 for h in cnt})
                for ep in range(C.EPISODES_PER_TASK):
                    g0 = self.primary_of(split, task, ep)
                    es = int(self.es_of[f"{task}|{g0}"])
                    for f in C.visible_motion_frames(es, es + C.MAX_STEPS):
                        seg, m = ("demo", f // C.GRID_STRIDE) if f < es else ("exec", (f - es) // C.GRID_STRIDE)
                        _, how, _ = self.lookup(split, task, ep, seg, m)
                        cnt[how] += 1
                        bt[how] += 1
        return {"total": cnt, "by_task": by_task}


def build_library_bank(lib: pathlib.Path, out_dir: pathlib.Path) -> dict:
    """从训练集 motion 库建 bank（全量 tokens 拷出 + 段表 + 主 donor 映射 + 兜底链）。"""
    from mme_vla_suite.datastore import motion_store as ms
    motion_root = C.require_motion_store(lib)
    mmeta = ms.MotionMeta.load(motion_root)
    store = ms.MotionStore(motion_root, meta=mmeta)
    try:
        tokens = np.ascontiguousarray(store.rows(np.arange(store.num_rows, dtype=np.int64)), dtype=np.float32)
        entries = list(store.entries)
    finally:
        store.close()
    segments: dict[str, dict] = {}
    es_of: dict[str, int] = {}
    pool: dict[str, list[int]] = {t: [] for t in C.TASKS}
    for e in entries:
        task = C.task_of_h5(e.h5_file)
        if task not in pool:
            continue
        pool[task].append(int(e.g))
        es_of[f"{task}|{e.g}"] = int(e.exec_start_idx)
        for seg in SEGS:
            si = getattr(e, seg)
            if si.row_base is not None and si.num_grid > 0:
                segments[f"{task}|{e.g}|{seg}"] = {"row_base": int(si.row_base), "n": int(si.num_grid)}
    for t, p in pool.items():
        if len(p) != 100:
            raise SystemExit(f"错误: 任务 {t} 在库里 {len(p)} 集 ≠ 100")
    chain = {t: sorted(p, key=lambda g: (-(segments.get(f"{t}|{g}|exec", {"n": 0})["n"]), g)) for t, p in pool.items()}
    primary: dict[str, int] = {}
    self_loops = 0
    for split in C.SPLITS:
        for t in C.TASKS:
            for ep in range(C.EPISODES_PER_TASK):
                primary[f"{split}|{t}|{ep}"] = pool[t][_pick(f"{split}|{t}|{ep}", 100)]
    for sd in OPEN_LOOP_SEEDS:
        for t in C.TASKS:
            for recv_g in pool[t]:
                cand = [g for g in pool[t] if g != recv_g]
                g = cand[_pick(f"ol{sd}|{t}|{recv_g}", len(cand))]
                if g == recv_g:
                    self_loops += 1
                primary[f"ol{sd}|{t}|{recv_g}"] = g
    sha = hashlib.sha256(tokens.tobytes()).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "donor_bank.npz", tokens=tokens)
    meta = {"source": "library", "schema": 1,
            "provenance": {"motion_root": str(motion_root), "motion_index_sha256": mmeta.motion_index_sha256,
                           "manifest_sha256": mmeta.manifest_sha256, "num_rows": int(mmeta.num_rows)},
            "tokens_shape": list(tokens.shape), "sha256": sha, "segments": segments, "es_of": es_of,
            "primary": primary, "fallback_chain": chain, "pool_size": {t: len(p) for t, p in pool.items()},
            "open_loop_seeds": list(OPEN_LOOP_SEEDS), "self_loops": self_loops,
            "exec_grid_max": {t: max(segments.get(f"{t}|{g}|exec", {"n": 0})["n"] for g in p) for t, p in pool.items()}}
    (out_dir / "donor_bank.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return meta
