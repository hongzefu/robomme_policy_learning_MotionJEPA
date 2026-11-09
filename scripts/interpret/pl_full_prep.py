#!/usr/bin/env python3
"""PatternLock 单步 motion token 交换「全因子全量」的准备脚本（计划 1109-patternlock-swap-full-factorial-wan-plan.html
第一部分第二、四节；第二部分〇′裁决、一节 pl_full_prep 行、四节 runbook）。

三个子命令（只用 CPU，不起模型与仿真）：
  pool    --out <O>
          新扫 env_seed：900000 起、步长 100、到 1299900，跳过 val 清单 1150000–1154900 的 50 个（共 3950 个），用
          pl_swap_prep.replay_path 复算 easy 路径，只留单步（路径长 2）局；40 种（起点格，方向）组合按 起点格升序、
          再按方向 (dr,dc) 升序排出，每种按 env_seed 升序取前 2 局：第 k 种的第 1 局为接收集 R 局号 k，第 2 局为
          供体集 S 局号 40+k。写：
            <O>/pool.json                                     扫描统计与 80 局清单
            <O>/metadata/record_dataset_PatternLock_metadata.json   load_episode_metadata 可读的自定义元数据
            <O>/manifest.json                                 80 局总清单（1106 manifest 同 schema，每块 5 局）
            <O>/pairs.json                                    A/B/C/D 配对表与 seed 复核抽样
          需在 official-sim 解释器里运行（replay_path 要 torch 与 robomme）。末两行判定：
            PL_POOL=PASS|FAIL eps=80 combos=40 overlap=<与 train/test/val env_seed 交集数>
            PL_PAIRS=PASS|FAIL A=40 B=160 C=168 D=1232 D_sel=400 seedcheck=60
  specs   --out <O>/<M> --model motion|wan --commit <sha>
          生成该模型的 repro（官方 PatternLock test 50 局，seed 42，official 口径）、smoke（N 2 局 → 裁供体 → 交换 2 局
          + 置零 1 局，全在 w00 顺序跑）、multieq（12 个交换局：每块 1 局 与 每块多局 两种跑法）、n（R+S 80 局，录 motion）
          四个阶段的 spec、逐 spec 清单、4 个 worker 队列、env.sh、launch_stage.sh、plan.json。
  donors  --out <O>/<M> [--wan-gate] [--smoke]
          读 n 阶段终局与 <O>/<M>/record/*.npz：写 donor_trim/（S 40 局，只留 start_frames==0 那一行）、zero/ 与
          random/（R 40 局；随机 = 同形状、同 L2 范数、随机方向，numpy 种子 = 局号），打印
            DONOR_TRIM=PASS|FAIL donors=<n> zeros=<n> randoms=<n> rows=<最大行数> maxabs=<最大偏差> norm_maxrel=<最大相对范数差>
            DONOR_DISTINCT=PASS|FAIL donors=40 identical_pairs=<n>
          再按 n 阶段结果过滤（接收局原样失败的交换不跑），生成 swap 阶段（Z、Rnd、A、B、C、D，seed 42）与 seeds 阶段
          （seed 复核对，seed 1、2）的 spec 与队列，并把这两阶段并入 plan.json。--wan-gate：R 集原样成功 < 12 时只生成
          Z、Rnd（〇′ F5），plan.json 记 gate。--smoke：只处理 smoke 阶段录制（<O>/<M>/smoke/），不碰 plan.json。

接口约定（主代理 plfull-contract.md）：局号 0–39 为 R、40–79 为 S；spec 新键 profile="plfull"、plfull_metadata_dir；
plfull 下一个 spec 可含多局，swap_map 逐局给供体；Z 用 motion_swap_exact:<O>/<M>/zero 自映射，Rnd 用 random 自映射，
A/B/C/D 用 donor_trim。repro 不写 profile（缺省 official，旧行为不变）。
运行目录 <O>/<M>/runs/<spec 名>/{attn,run}：一个 spec 一个 run_root，同一 run_root 内接收局唯一，不同 worker 不共用。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import uuid

import numpy as np

from scripts.evaluation import manifest as M
from scripts.interpret.pl_swap_prep import (BENCH_SRC, DIFFICULTY, GRID, MAIN, STORE, SPEC_TEMPLATE, TASK, XLA_MEM_FRACTION,
                                            button_xy, cell_rc, direction, episode_key, record_name, replay_path)

# ---------------------------------------------------------------- 常量
SCAN_START, SCAN_STOP, SCAN_STEP = 900000, 1299900, 100
VAL_RANGE = (1150000, 1154900)
N_COMBOS = 40
R_OFFSET = 40                       # S 局号 = 40 + 组合号
D_IN, D_OUT = 300, 100              # D 分层抽样：供体方向从接收起点出发仍在盘内 / 出盘
PAIR_SEED = 0                       # D 抽样与 seed 复核抽样的 numpy 种子
SEEDCHECK_PER_GROUP = 15
SEEDCHECK_SEEDS = (1, 2)
MAIN_SEED = 42
SWAP_GROUPS = ("Z", "Rnd", "A", "B", "C", "D")
PAIR_GROUPS = ("A", "B", "C", "D")
EXPECTED_PAIRS = {"A": 40, "B": 160, "C": 168, "D": 1232}
WAN_GATE_MIN = 12
SPEC_MAX = 20                       # 每个 spec（一次 run_local）最多局数
CHUNK = 5                           # 清单每块局数（eval.py 每块 1–5 局）
WORKERS = ("w00", "w01", "w10", "w11")
WORKER_CUDA = {"w00": "0", "w01": "0", "w10": "1", "w11": "1"}
ASSIGN_ORDER = ("w00", "w10", "w01", "w11")   # 轮转时两卡交替
STAGES = ("repro", "smoke", "multieq", "n", "swap", "seeds")
SMOKE_R, SMOKE_S = 0, 41            # 组合 0 与组合 1 同起点格、不同方向（B 型）
MULTIEQ_PER_GROUP = 3
RUN_TREE = STORE / "worktrees/pl-full-run"
LOG_DIR = STORE / "logs/pl-swap-full-1109"
META_NAME = f"record_dataset_{TASK}_metadata.json"
OFFICIAL_META = BENCH_SRC / "robomme/env_metadata"

# 模型表：Wan 项由主代理按 W2 交回填（或用 specs 的 --checkpoint/--policy-config/--model-kind/--target-contract/--extra 覆盖）
MODELS = {
    "motion": {"checkpoint": str(STORE / "assets/vla-o16cc-59999/59999"),
               "policy_config": "mme_vla_suite_orig16_b128_60k",
               "model_kind": "stage2_pool4_concat",
               "target_contract": SPEC_TEMPLATE["target_contract"]},
    "wan": {"checkpoint": None, "policy_config": None, "model_kind": None, "target_contract": None},
}
ABLATE_DIR = {"Z": "zero", "Rnd": "random", "A": "donor_trim", "B": "donor_trim", "C": "donor_trim", "D": "donor_trim"}


# ---------------------------------------------------------------- 几何与组合
def all_combos():
    """40 种合法 (起点格, (dr,dc))，按起点格升序、再按方向升序。"""
    out = []
    for cell in range(GRID * GRID):
        r, c = cell_rc(cell)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr, dc) != (0, 0) and 0 <= r + dr < GRID and 0 <= c + dc < GRID:
                    out.append((cell, (dr, dc)))
    return out


def step_cell(start, d):
    """从 start 走方向 d 到达的格；出盘返回 None。"""
    r, c = cell_rc(start)
    r2, c2 = r + d[0], c + d[1]
    if 0 <= r2 < GRID and 0 <= c2 < GRID:
        return r2 * GRID + c2
    return None


def combo_of_path(path):
    return (int(path[0]), direction(path))


# ---------------------------------------------------------------- pool
def scan_seeds():
    return [s for s in range(SCAN_START, SCAN_STOP + 1, SCAN_STEP) if not VAL_RANGE[0] <= s <= VAL_RANGE[1]]


def split_metadata_seeds(dataset):
    data = json.loads((OFFICIAL_META / dataset / META_NAME).read_text())
    return {int(r["seed"]) for r in data["records"]}


def choose_pool(single_step):
    """single_step: [(env_seed, path)]（任意顺序，路径长 2）。返回 (episodes, per_combo_count)；
    episodes 为 80 局 dict，组合不足 2 局时 SystemExit。"""
    combos = all_combos()
    index = {combo: k for k, combo in enumerate(combos)}
    by_combo = {combo: [] for combo in combos}
    for seed, path in sorted(single_step):
        if len(path) != 2:
            raise ValueError(f"env_seed {seed} 不是单步: {path}")
        by_combo[combo_of_path(path)].append((int(seed), [int(x) for x in path]))
    short = [f"{c[0]}:{c[1]}" for c, v in by_combo.items() if len(v) < 2]
    if short:
        raise SystemExit(f"这些组合不足 2 局: {short}")
    episodes = []
    for role, offset, pick in (("R", 0, 0), ("S", R_OFFSET, 1)):
        for combo in combos:
            seed, path = by_combo[combo][pick]
            k = index[combo]
            episodes.append({"episode_id": offset + k, "role": role, "combo": k, "env_seed": seed, "path": path,
                             "start": combo[0], "dir": list(combo[1]), "target": path[1]})
    episodes.sort(key=lambda e: e["episode_id"])
    return episodes, {f"{c[0]}:{c[1][0]},{c[1][1]}": len(v) for c, v in by_combo.items()}


def metadata_payload(episodes):
    records = [{"task": TASK, "episode": e["episode_id"], "seed": e["env_seed"], "difficulty": DIFFICULTY}
               for e in episodes]
    return {"env_id": TASK, "record_count": len(records), "records": records}


def manifest_for(episodes, meta_sha, chunk=CHUNK):
    """与 1106 manifest 同 schema；episodes 依给定顺序，每块 ≤ chunk 局。"""
    rows = [{"task_name": TASK, "episode_id": int(e["episode_id"]), "env_seed": int(e["env_seed"]),
             "difficulty": DIFFICULTY} for e in episodes]
    return M.make_manifest(rows, metadata_sha256={META_NAME: meta_sha}, chunk_size=chunk, official=False)


def build_pairs(combos=None):
    """返回 pairs dict（局号用 R=k、S=40+j）。combos 为 40 种组合（缺省 all_combos）。"""
    combos = combos or all_combos()
    rows = {g: [] for g in PAIR_GROUPS}
    d_in, d_out = [], []
    for k, (rs, rd) in enumerate(combos):
        for j, (ds, dd) in enumerate(combos):
            pair = {"recv": k, "donor": R_OFFSET + j, "recv_combo": k, "donor_combo": j,
                    "recv_start": rs, "recv_dir": list(rd), "donor_start": ds, "donor_dir": list(dd),
                    "own": step_cell(rs, rd), "donor_abs": step_cell(ds, dd), "donor_dir_cell": step_cell(rs, dd)}
            same_start, same_dir = rs == ds, rd == dd
            if same_start and same_dir:
                rows["A"].append(pair)
            elif same_start:
                rows["B"].append(pair)
            elif same_dir:
                rows["C"].append(pair)
            else:
                pair["stratum"] = "in" if pair["donor_dir_cell"] is not None else "out"
                rows["D"].append(pair)
                (d_in if pair["stratum"] == "in" else d_out).append(pair)
    if len(d_in) < D_IN or len(d_out) < D_OUT:
        raise SystemExit(f"D 分层不够抽：in={len(d_in)} out={len(d_out)}")
    rng = np.random.default_rng(PAIR_SEED)
    pick_in = sorted(rng.choice(len(d_in), size=D_IN, replace=False).tolist())
    pick_out = sorted(rng.choice(len(d_out), size=D_OUT, replace=False).tolist())
    d_sel = sorted([d_in[i] for i in pick_in] + [d_out[i] for i in pick_out], key=lambda p: (p["recv"], p["donor"]))
    selected = {"A": rows["A"], "B": rows["B"], "C": rows["C"], "D": d_sel}
    seedcheck = {}
    for group in PAIR_GROUPS:
        rng_g = np.random.default_rng(PAIR_SEED)
        pool = selected[group]
        idx = sorted(rng_g.choice(len(pool), size=SEEDCHECK_PER_GROUP, replace=False).tolist())
        seedcheck[group] = [pool[i] for i in idx]
    return {"combos": [{"combo": k, "start": s, "dir": list(d), "target": step_cell(s, d)} for k, (s, d) in enumerate(combos)],
            "counts": {g: len(rows[g]) for g in PAIR_GROUPS}, "d_strata": {"in": len(d_in), "out": len(d_out)},
            "all": rows, "selected": selected, "seedcheck": seedcheck,
            "seedcheck_seeds": list(SEEDCHECK_SEEDS), "rng_seed": PAIR_SEED}


def pairs_line(pairs):
    c = pairs["counts"]
    sel = pairs["selected"]
    seedcheck = sum(len(v) for v in pairs["seedcheck"].values())
    ok = (all(c[g] == EXPECTED_PAIRS[g] for g in PAIR_GROUPS) and len(sel["D"]) == D_IN + D_OUT
          and sum(1 for p in sel["D"] if p["stratum"] == "in") == D_IN
          and seedcheck == SEEDCHECK_PER_GROUP * len(PAIR_GROUPS))
    return (f"PL_PAIRS={'PASS' if ok else 'FAIL'} A={c['A']} B={c['B']} C={c['C']} D={c['D']} "
            f"D_sel={len(sel['D'])} seedcheck={seedcheck}"), ok


def cmd_pool(args):
    out = Path(args.out).resolve()
    seeds = scan_seeds()
    single, lengths = [], {}
    for i, seed in enumerate(seeds):
        path = replay_path(seed)
        lengths[len(path)] = lengths.get(len(path), 0) + 1
        if len(path) == 2:
            single.append((seed, path))
        if (i + 1) % 500 == 0:
            print(f"SCAN {i + 1}/{len(seeds)} single={len(single)}", flush=True)
    episodes, per_combo = choose_pool(single)
    split_seeds = {ds: split_metadata_seeds(ds) for ds in ("train", "test", "val")}
    chosen = {e["env_seed"] for e in episodes}
    overlap = {ds: sorted(chosen & s) for ds, s in split_seeds.items()}
    n_overlap = sum(len(v) for v in overlap.values())
    # 复核：80 局逐个再复算一次
    recheck = sum(1 for e in episodes if replay_path(e["env_seed"]) != e["path"])
    combos_covered = len({e["combo"] for e in episodes if e["role"] == "R"})
    meta = metadata_payload(episodes)
    meta_bytes = (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode()
    meta_sha = hashlib.sha256(meta_bytes).hexdigest()
    _write_once(out / "metadata" / META_NAME, meta_bytes, args.overwrite)
    manifest = manifest_for(episodes, meta_sha)
    _write_once(out / "manifest.json", _json_bytes(manifest), args.overwrite)
    pool = {"task": TASK, "difficulty": DIFFICULTY, "scan": {"start": SCAN_START, "stop": SCAN_STOP, "step": SCAN_STEP,
            "skip": list(VAL_RANGE), "scanned": len(seeds), "single_step": len(single),
            "length_hist": {str(k): v for k, v in sorted(lengths.items())}},
            "per_combo_single_step": per_combo, "overlap": overlap, "recheck_mismatch": recheck,
            "metadata_sha256": meta_sha, "manifest_seal": manifest["sha256"], "episodes": episodes}
    _write_once(out / "pool.json", _json_bytes(pool), args.overwrite)
    pairs = build_pairs()
    _write_once(out / "pairs.json", _json_bytes(pairs), args.overwrite)
    print(f"SCAN_DONE scanned={len(seeds)} single_step={len(single)} min_per_combo={min(per_combo.values())}", flush=True)
    ok_pool = len(episodes) == 2 * N_COMBOS and combos_covered == N_COMBOS and n_overlap == 0 and recheck == 0
    print(f"PL_POOL={'PASS' if ok_pool else 'FAIL'} eps={len(episodes)} combos={combos_covered} overlap={n_overlap}", flush=True)
    line, ok_pairs = pairs_line(pairs)
    print(line, flush=True)
    return 0 if ok_pool and ok_pairs else 1


# ---------------------------------------------------------------- specs 公共
def model_entry(name, overrides=None):
    entry = dict(MODELS[name])
    entry.update({k: v for k, v in (overrides or {}).items() if v is not None})
    missing = [k for k in ("checkpoint", "policy_config", "model_kind", "target_contract") if not entry.get(k)]
    if missing:
        raise SystemExit(f"模型 {name} 缺 {missing}：Wan 项须由主代理按 W2 交回给出（--checkpoint 等参数）")
    return entry


def base_spec(model, run_tree):
    spec = dict(SPEC_TEMPLATE)
    spec.update({k: v for k, v in model.items() if k != "extra"})
    spec.update(model.get("extra") or {})
    spec["repo"] = str(run_tree)
    return spec


class Builder:
    """收集某模型某几个阶段的 spec、清单与 plan 行；write() 落盘。"""

    def __init__(self, out, root, model_name, model, run_tree, frac, meta_dir, meta_sha):
        self.out, self.root, self.model_name, self.model = Path(out), Path(root), model_name, model
        self.run_tree, self.frac, self.meta_dir, self.meta_sha = Path(run_tree), float(frac), Path(meta_dir), meta_sha
        self.specs = {}       # 名 → (spec, manifest, stage, worker)
        self.rows = []        # plan 行
        self.queue = {}       # (stage, worker) → [行文本]

    def add_spec(self, name, stage, worker, episodes, *, seed=MAIN_SEED, profile="plfull", ablate="none",
                 swap_map=None, record=None, manifest=None, chunk=CHUNK, record_dir=None, rows=()):
        if name in self.specs:
            raise ValueError(f"spec 重名 {name}")
        run_dir = self.out / "runs" / name
        spec = base_spec(self.model, self.run_tree)
        spec.update(model_seed=int(seed), cuda=WORKER_CUDA[worker], attn_out=str(run_dir / "attn"),
                    run_root=str(run_dir / "run"), xla_mem_fraction=self.frac,
                    manifest=str(self.out / "manifests" / f"{name}.json"), ablate=ablate)
        if profile == "plfull":
            spec.update(profile="plfull", plfull_metadata_dir=str(self.meta_dir))
        if swap_map:
            spec["swap_map"] = dict(swap_map)
        if record:
            spec["record_motion"] = str(record)
        if manifest is None:
            manifest = manifest_for(episodes, self.meta_sha, chunk=chunk)
        self.specs[name] = (spec, manifest, stage, worker)
        self.queue.setdefault((stage, worker), []).append(f"run_spec {stage} {name} {shlex.quote(str(self.out / 'specs' / f'{name}.json'))}\n")
        for row in rows:
            row = dict(row)
            row.update(stage=stage, worker=worker, cuda=WORKER_CUDA[worker], seed=int(seed), spec=name,
                       spec_path=str(self.out / "specs" / f"{name}.json"), attn_out=spec["attn_out"],
                       run_root=spec["run_root"], record_dir=str(record_dir or self.out / "record"))
            self.rows.append(row)
        return spec

    def add_cmd(self, stage, worker, text):
        self.queue.setdefault((stage, worker), []).append(f"run_cmd {stage} {shlex.quote(text)}\n")

    def write(self, stages):
        for name, (spec, manifest, stage, worker) in self.specs.items():
            _write_json(self.out / "manifests" / f"{name}.json", manifest)
            _write_json(self.out / "specs" / f"{name}.json", spec)
        written = []
        (self.out / "queues").mkdir(parents=True, exist_ok=True)
        for stage in stages:
            for worker in WORKERS:
                lines = self.queue.get((stage, worker), [])
                n_spec = sum(1 for x in lines if x.startswith("run_spec "))
                path = self.out / "queues" / f"{stage}-{worker}.sh"
                text = [QUEUE_HEAD.format(stage=stage, worker=worker, cuda=WORKER_CUDA[worker], n=n_spec,
                                          env=shlex.quote(str(self.out / "env.sh")), frac=self.frac)]
                text += lines
                text.append(QUEUE_TAIL.format(stage=stage, worker=worker, n=n_spec))
                path.write_text("".join(text))
                path.chmod(0o755)
                written.append(path)
        return written


def run_row(group, recv, donor, pool_by_id, **extra):
    """一局的 plan 行（不含 stage/worker/spec 等，由 Builder 补）。"""
    re_ = pool_by_id[recv]
    row = {"group": group, "ep": int(recv), "donor": None if donor is None else int(donor),
           "path": list(re_["path"]), "donor_path": None}
    if donor is not None and donor != recv:
        row["donor_path"] = list(pool_by_id[donor]["path"])
    row.update(extra)
    return row


def pack(runs, cap=SPEC_MAX):
    """runs: [(key, recv, ...)]，按序装箱：每箱同 key、接收局唯一、≤ cap。返回 [(key, [run,...])]。"""
    bins = []
    for run in runs:
        key, recv = run[0], run[1]
        for bkey, members, recvs in bins:
            if bkey == key and len(members) < cap and recv not in recvs:
                members.append(run)
                recvs.add(recv)
                break
        else:
            bins.append((key, [run], {recv}))
    return [(key, members) for key, members, _ in bins]


def round_robin(items):
    return [(ASSIGN_ORDER[i % len(ASSIGN_ORDER)], item) for i, item in enumerate(items)]


# ---------------------------------------------------------------- specs 子命令
def multieq_pairs(pairs):
    """每组（A/B/C/D 已选配对）按表顺序各取 3 对，12 对接收局两两不同。"""
    chosen, used = [], set()
    for group in PAIR_GROUPS:
        n = 0
        for p in pairs["selected"][group]:
            if p["recv"] in used:
                continue
            chosen.append((group, p))
            used.add(p["recv"])
            n += 1
            if n == MULTIEQ_PER_GROUP:
                break
    return chosen


def repro_manifest():
    """官方 PatternLock test 50 局，按 1106 manifest 的方式（build_manifest.build）取局，每块 5 局。"""
    from scripts.interpret import build_manifest as BM
    return BM.build([(TASK, i) for i in range(50)], OFFICIAL_META / "test", 5)


def build_stage_specs(b, pool_by_id, pairs, stages):
    """repro / smoke / multieq / n 四阶段。"""
    out = b.out
    if "repro" in stages:
        full = repro_manifest()
        by_key = {e["episode_key"]: e for e in full["episodes"]}
        keys = [e["episode_key"] for e in full["episodes"]]
        for w_i, worker in enumerate(ASSIGN_ORDER):
            mine = keys[w_i::len(ASSIGN_ORDER)]
            rows_ = [{"task_name": TASK, "episode_id": by_key[k]["episode_id"], "env_seed": by_key[k]["env_seed"],
                      "difficulty": by_key[k]["difficulty"]} for k in mine]
            man = M.make_manifest(rows_, metadata_sha256=full["metadata_sha256"], chunk_size=CHUNK, official=False)
            plan_rows = [{"group": "repro", "ep": int(by_key[k]["episode_id"]), "donor": None, "path": None,
                          "donor_path": None, "env_seed": int(by_key[k]["env_seed"])} for k in mine]
            b.add_spec(f"repro-{worker}", "repro", worker, None, profile="official", manifest=man, rows=plan_rows)
    if "smoke" in stages:
        smoke = out / "smoke"
        eps = [pool_by_id[SMOKE_R], pool_by_id[SMOKE_S]]
        b.add_spec("smoke-n", "smoke", "w00", eps, record=smoke / "record", record_dir=smoke / "record",
                   rows=[run_row("N", e["episode_id"], None, pool_by_id) for e in eps])
        b.add_cmd("smoke", "w00", f"\"$py\" -m scripts.interpret.pl_full_prep donors --out {shlex.quote(str(out))} --smoke")
        b.add_spec("smoke-swap", "smoke", "w00", eps, ablate=f"motion_swap_exact:{smoke / 'donor_trim'}",
                   swap_map={episode_key(SMOKE_R): episode_key(SMOKE_S), episode_key(SMOKE_S): episode_key(SMOKE_R)},
                   record_dir=smoke / "record",
                   rows=[run_row("B", SMOKE_R, SMOKE_S, pool_by_id), run_row("B", SMOKE_S, SMOKE_R, pool_by_id)])
        b.add_spec("smoke-zero", "smoke", "w00", [pool_by_id[SMOKE_R]], ablate=f"motion_swap_exact:{smoke / 'zero'}",
                   swap_map={episode_key(SMOKE_R): episode_key(SMOKE_R)}, record_dir=smoke / "record",
                   rows=[run_row("Z", SMOKE_R, SMOKE_R, pool_by_id)])
    if "multieq" in stages:
        chosen = multieq_pairs(pairs)
        ablate = f"motion_swap_exact:{out / 'donor_trim'}"
        single_workers = ("w00", "w01", "w11")
        for i, (group, p) in enumerate(chosen):
            worker = single_workers[i % len(single_workers)]
            b.add_spec(f"multieq-single-{i:02d}", "multieq", worker, [pool_by_id[p["recv"]]], ablate=ablate,
                       swap_map={episode_key(p["recv"]): episode_key(p["donor"])}, chunk=1,
                       rows=[run_row(group, p["recv"], p["donor"], pool_by_id, variant="single", pair_index=i)])
        b.add_spec("multieq-multi", "multieq", "w10", [pool_by_id[p["recv"]] for _, p in chosen], ablate=ablate,
                   swap_map={episode_key(p["recv"]): episode_key(p["donor"]) for _, p in chosen},
                   rows=[run_row(g, p["recv"], p["donor"], pool_by_id, variant="multi", pair_index=i)
                         for i, (g, p) in enumerate(chosen)])
    if "n" in stages:
        ids = sorted(pool_by_id)
        for w_i, worker in enumerate(ASSIGN_ORDER):
            mine = [pool_by_id[i] for i in ids[w_i::len(ASSIGN_ORDER)]]
            b.add_spec(f"n-{worker}", "n", worker, mine, record=out / "record",
                       rows=[run_row("N", e["episode_id"], None, pool_by_id) for e in mine])


def load_pool(pool_root):
    pool = _read_json(Path(pool_root) / "pool.json")
    pairs = _read_json(Path(pool_root) / "pairs.json")
    pool_by_id = {int(e["episode_id"]): e for e in pool["episodes"]}
    if sorted(pool_by_id) != list(range(2 * N_COMBOS)):
        raise SystemExit("pool.json 局号不是 0–79")
    return pool, pairs, pool_by_id


def make_builder(out, model_name, model, args, pool):
    pool_root = out.parent
    meta_dir = pool_root / "metadata"
    meta_sha = hashlib.sha256((meta_dir / META_NAME).read_bytes()).hexdigest()
    if meta_sha != pool["metadata_sha256"]:
        raise SystemExit("metadata 与 pool.json 记录的 sha256 不符")
    return Builder(out, pool_root, model_name, model, args.run_tree, args.frac, meta_dir, meta_sha)


def cmd_specs(args):
    out = Path(args.out).resolve()
    if not args.commit or len(args.commit) < 7:
        raise SystemExit("--commit 须给运行工作树的 commit sha")
    if out.name != args.model:
        raise SystemExit(f"--out 末级目录须为模型名 {args.model}（约定 <O>/<M>）")
    extra = json.loads(args.extra) if args.extra else None
    model = model_entry(args.model, {"checkpoint": args.checkpoint, "policy_config": args.policy_config,
                                     "model_kind": args.model_kind, "target_contract": args.target_contract})
    if extra:
        model["extra"] = extra
    pool, pairs, pool_by_id = load_pool(out.parent)
    plan_path = out / "plan.json"
    if plan_path.exists() and not args.overwrite:
        raise SystemExit(f"{plan_path} 已存在；确需重写 spec 加 --overwrite（不删任何运行目录）")
    out.mkdir(parents=True, exist_ok=True)
    b = make_builder(out, args.model, model, args, pool)
    stages = ("repro", "smoke", "multieq", "n")
    build_stage_specs(b, pool_by_id, pairs, stages)
    env_text = ENV_TEMPLATE.format(frac=args.frac, run_tree=args.run_tree, commit=args.commit, store=STORE,
                                   bench=BENCH_SRC, out=out, py=SPEC_TEMPLATE["policy_python"],
                                   sim=SPEC_TEMPLATE["sim_python"])
    (out / "env.sh").write_text(env_text)
    launch = out / "launch_stage.sh"
    launch.write_text(LAUNCH_TEMPLATE.format(out=out, model=args.model, logs=LOG_DIR, workers=" ".join(WORKERS),
                                             stages="|".join(STAGES)))
    launch.chmod(0o755)
    queues = b.write(STAGES)   # swap/seeds 先写空队列，donors 再覆盖
    plan = {"model": args.model, "commit": args.commit, "run_tree": str(args.run_tree), "out": str(out),
            "pool_root": str(out.parent), "record_dir": str(out / "record"), "model_entry": model,
            "manifest_seal": pool["manifest_seal"], "frac": float(args.frac), "gate": None, "runs": b.rows}
    _write_json(plan_path, plan)
    counts = {st: sum(1 for r in b.rows if r["stage"] == st) for st in stages}
    print("PLF_SPECS model=" + args.model + " " + " ".join(f"{k}={v}" for k, v in counts.items())
          + f" specs={len(b.specs)} queues={len(queues)}", flush=True)
    return 0


# ---------------------------------------------------------------- donors
def latest_terminal(run_root, key):
    paths = sorted(Path(run_root).glob(f"accepted/{key}.*.terminal.json"), key=lambda p: p.stat().st_mtime)
    return _read_json(paths[-1]) if paths else None


def n_results(plan):
    """n 阶段每局 → success（1/0）或 None（无终局）。"""
    out = {}
    for row in plan["runs"]:
        if row["stage"] != "n":
            continue
        term = latest_terminal(row["run_root"], episode_key(row["ep"]))
        out[int(row["ep"])] = None if term is None else int(term.get("outcome") == "success")
    return out


def _savez_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def start0_row(path):
    """录制 npz 中 start_frames==0 的 (emb[1,D], count[1])；不满足返回 (None, 原因)。"""
    if not Path(path).is_file():
        return None, "missing_record"
    with np.load(path) as z:
        starts = np.asarray(z["start_frames"], np.int32).reshape(-1)
        embs = np.asarray(z["embs"], np.float32)
        counts = np.asarray(z["counts"], np.int32).reshape(-1) if "counts" in z.files else np.ones_like(starts)
    idx = np.flatnonzero(starts == 0)
    if idx.size != 1 or embs.ndim != 2 or embs.shape[0] != starts.size:
        return None, f"start0_rows={idx.size}"
    i = int(idx[0])
    count = counts[i:i + 1].astype(np.int32) if counts.size == starts.size else np.ones(1, np.int32)
    return (embs[i:i + 1].copy(), count), None


def random_like(row, ep):
    """同形状、同 L2 范数、随机方向；numpy 种子 = 局号。"""
    rng = np.random.default_rng(int(ep))
    v = rng.standard_normal(row.shape).astype(np.float64)
    norm = float(np.linalg.norm(row.astype(np.float64)))
    v *= norm / float(np.linalg.norm(v))
    return v.astype(np.float32)


def make_donor_files(base, donor_eps, recv_eps):
    """base 下 record/ → donor_trim/（donor_eps）、zero/ 与 random/（recv_eps）。返回 (ok, lines, problems, rows)。"""
    base = Path(base)
    problems, rows = [], {}
    stats = {"donors": 0, "zeros": 0, "randoms": 0, "rows": 0, "maxabs": 0.0, "norm_maxrel": 0.0}
    for ep in sorted(set(donor_eps) | set(recv_eps)):
        got, why = start0_row(base / "record" / record_name(ep))
        if got is None:
            problems.append(f"ep{ep}:{why}")
            continue
        row, count = got
        rows[ep] = row
        targets = []
        if ep in donor_eps:
            targets.append(("donor", base / "donor_trim" / record_name(ep), row))
        if ep in recv_eps:
            targets.append(("zero", base / "zero" / record_name(ep), np.zeros_like(row)))
            targets.append(("random", base / "random" / record_name(ep), random_like(row, ep)))
        for kind, path, emb in targets:
            _savez_atomic(path, start_frames=np.zeros(1, np.int32), embs=emb.astype(np.float32), counts=count)
            with np.load(path) as back:
                b_starts, b_embs = np.asarray(back["start_frames"]), np.asarray(back["embs"], np.float32)
            stats["rows"] = max(stats["rows"], int(b_starts.size))
            if b_starts.size != 1 or int(b_starts[0]) != 0 or b_embs.shape != row.shape:
                problems.append(f"ep{ep}:{kind}_shape")
            if kind == "random":
                n0, n1 = float(np.linalg.norm(row.astype(np.float64))), float(np.linalg.norm(b_embs.astype(np.float64)))
                rel = abs(n1 - n0) / n0 if n0 > 0 else float("inf")
                stats["norm_maxrel"] = max(stats["norm_maxrel"], rel)
                if not rel < 1e-5:
                    problems.append(f"ep{ep}:random_norm_rel={rel:g}")
            else:
                dev = float(np.max(np.abs(b_embs - (row if kind == "donor" else 0.0))))
                stats["maxabs"] = max(stats["maxabs"], dev)
                if dev != 0.0:
                    problems.append(f"ep{ep}:{kind}_maxabs={dev}")
            stats[{"donor": "donors", "zero": "zeros", "random": "randoms"}[kind]] += 1
    ok = not problems
    trim = (f"DONOR_TRIM={'PASS' if ok else 'FAIL'} donors={stats['donors']} zeros={stats['zeros']} "
            f"randoms={stats['randoms']} rows={stats['rows']} maxabs={stats['maxabs']:g} norm_maxrel={stats['norm_maxrel']:.2e}")
    return ok, trim, problems, rows


def distinct_line(rows, donor_eps):
    have = [ep for ep in sorted(donor_eps) if ep in rows]
    identical = []
    for i, a in enumerate(have):
        for b_ in have[i + 1:]:
            if float(np.max(np.abs(rows[a] - rows[b_]))) == 0.0:
                identical.append((a, b_))
    ok = not identical and len(have) == len(donor_eps)
    return f"DONOR_DISTINCT={'PASS' if ok else 'FAIL'} donors={len(have)} identical_pairs={len(identical)}", ok, identical


def swap_runs(pairs, success, gate_few):
    """返回 ([(group, recv, donor, seed)], skipped)：接收局原样成功才进。"""
    runs, skipped = [], []
    recv_ok = lambda r: success.get(r) == 1
    for group in ("Z", "Rnd"):
        for k in range(N_COMBOS):
            (runs if recv_ok(k) else skipped).append((group, k, k, MAIN_SEED))
    if not gate_few:
        for group in PAIR_GROUPS:
            for p in pairs["selected"][group]:
                (runs if recv_ok(p["recv"]) else skipped).append((group, p["recv"], p["donor"], MAIN_SEED))
        for seed in SEEDCHECK_SEEDS:
            for group in PAIR_GROUPS:
                for p in pairs["seedcheck"][group]:
                    (runs if recv_ok(p["recv"]) else skipped).append((group, p["recv"], p["donor"], seed))
    return runs, skipped


def build_swap_specs(b, pool_by_id, runs):
    """swap（seed 42）与 seeds（seed 1、2）两阶段：轮转分 worker，再按 (seed, 供体目录) 装箱。"""
    for stage, members in (("swap", [r for r in runs if r[3] == MAIN_SEED]),
                           ("seeds", [r for r in runs if r[3] != MAIN_SEED])):
        per_worker = {}
        for worker, run in round_robin(members):
            per_worker.setdefault(worker, []).append(run)
        for worker in WORKERS:
            items = [((run[3], ABLATE_DIR[run[0]]), run[1], run) for run in per_worker.get(worker, [])]
            for i, ((seed, kind), bin_) in enumerate(pack(items)):
                name = f"{stage}-{worker}-{kind}-s{seed}-{i:02d}"
                recvs = [item[1] for item in bin_]
                b.add_spec(name, stage, worker, [pool_by_id[r] for r in recvs], seed=seed,
                           ablate=f"motion_swap_exact:{b.out / kind}",
                           swap_map={episode_key(item[2][1]): episode_key(item[2][2]) for item in bin_},
                           rows=[run_row(item[2][0], item[2][1], item[2][2], pool_by_id) for item in bin_])


def cmd_donors(args):
    out = Path(args.out).resolve()
    if args.smoke:
        ok, trim, problems, rows = make_donor_files(out / "smoke", {SMOKE_R, SMOKE_S}, {SMOKE_R})
        for problem in problems:
            print(f"DONOR_PROBLEM {problem}", flush=True)
        print(trim, flush=True)
        return 0 if ok else 1
    plan = _read_json(out / "plan.json")
    pool, pairs, pool_by_id = load_pool(out.parent)
    success = n_results(plan)
    donor_eps = set(range(R_OFFSET, 2 * N_COMBOS))
    recv_eps = set(range(N_COMBOS))
    ok, trim, problems, rows = make_donor_files(out, donor_eps, recv_eps)
    dline, dok, identical = distinct_line(rows, donor_eps)
    for problem in problems:
        print(f"DONOR_PROBLEM {problem}", flush=True)
    for a, b_ in identical:
        print(f"DONOR_IDENTICAL {a} {b_}", flush=True)
    print(trim, flush=True)
    print(dline, flush=True)
    missing_n = sorted(ep for ep in range(2 * N_COMBOS) if success.get(ep) is None)
    r_success = sum(1 for k in range(N_COMBOS) if success.get(k) == 1)
    s_success = sum(1 for k in donor_eps if success.get(k) == 1)
    gate_few = bool(args.wan_gate) and r_success < WAN_GATE_MIN
    print(f"N_RESULT r_success={r_success}/40 s_success={s_success}/40 missing={len(missing_n)} "
          f"gate={'few' if gate_few else ('ok' if args.wan_gate else 'off')}", flush=True)
    if not (ok and dok):
        print("DONORS_SPECS=SKIP reason=donor_files", flush=True)
        return 1
    runs, skipped = swap_runs(pairs, success, gate_few)
    model = plan["model_entry"]
    b = make_builder(out, plan["model"], model, argparse.Namespace(run_tree=plan["run_tree"], frac=plan.get("frac", XLA_MEM_FRACTION)), pool)
    build_swap_specs(b, pool_by_id, runs)
    b.write(("swap", "seeds"))
    plan["runs"] = [r for r in plan["runs"] if r["stage"] not in ("swap", "seeds")] + b.rows
    plan["gate"] = {"wan_gate": bool(args.wan_gate), "r_success": r_success, "few": gate_few, "min": WAN_GATE_MIN,
                    "groups": ["N", "Z", "Rnd"] if gate_few else ["N"] + list(SWAP_GROUPS)}
    plan["skipped_recv_failed"] = [list(x) for x in skipped]
    plan["n_missing"] = missing_n
    _write_json(out / "plan.json", plan)
    counts = {}
    for row in b.rows:
        counts[f"{row['stage']}:{row['group']}"] = counts.get(f"{row['stage']}:{row['group']}", 0) + 1
    print("PLF_SWAP_SPECS " + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
          + f" total={len(b.rows)} specs={len(b.specs)} skipped={len(skipped)}", flush=True)
    return 0


# ---------------------------------------------------------------- 模板
ENV_TEMPLATE = """#!/usr/bin/env bash
# 由 scripts/interpret/pl_full_prep.py specs 生成。用法：source env.sh <GPU> [显存比例，默认 {frac:.2f}]
# 环境变量照抄先导轮 env.sh；运行工作树固定 {run_tree}，HEAD 必须为 {commit} 且工作区 clean。
set -o pipefail
gpu=$1; frac=${{2:-{frac:.2f}}}
export CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_MEM_FRACTION=$frac
export PYTHONUNBUFFERED=1 UV_CACHE_DIR={store}/cache/uv XDG_CACHE_HOME={store}/cache HF_HOME={store}/cache/hf TORCH_HOME={store}/cache/torch JAX_COMPILATION_CACHE_DIR={store}/cache/jax MMEVLA_V1_STORE={store} OPENPI_DATA_HOME={store}/models HF_HUB_OFFLINE=1
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
T={run_tree}
export PYTHONPATH=$T:$T/src:{bench}
cd "$T" || return 2
test "$(git rev-parse HEAD)" = {commit} || {{ echo "ENV_CHECK=FAIL reason=head"; return 2; }}
test -z "$(git status --porcelain)" || {{ echo "ENV_CHECK=FAIL reason=dirty"; return 2; }}
O={out}
py={py}
sim={sim}
echo "ENV_CHECK=PASS head={commit} gpu=$gpu frac=$frac"
"""

QUEUE_HEAD = """#!/usr/bin/env bash
# 由 scripts/interpret/pl_full_prep.py 生成：阶段 {stage}、worker {worker}（GPU {cuda}），{n} 个 spec。
# 某 spec rc≠0 记失败并继续下一个；run_cmd 失败即退出。末行 PLF_QUEUE，有失败时以非零退出。
source {env} {cuda} {frac:.2f} || exit $?
fail=0
run_spec() {{
  local stage=$1 name=$2 spec=$3
  "$py" -m scripts.interpret.run_local --spec "$spec"
  local rc=$?
  echo "PLF_SPEC stage=$stage spec=$name rc=$rc"
  if [ "$rc" -ne 0 ]; then fail=$((fail+1)); fi
}}
run_cmd() {{
  local stage=$1 cmd=$2
  eval "$cmd"
  local rc=$?
  echo "PLF_CMD stage=$stage rc=$rc"
  if [ "$rc" -ne 0 ]; then echo "PLF_QUEUE stage=$stage worker={worker} specs={n} failed=cmd"; exit "$rc"; fi
}}
"""

QUEUE_TAIL = """echo "PLF_QUEUE stage={stage} worker={worker} specs={n} failed=$fail"
[ "$fail" -eq 0 ]
"""

LAUNCH_TEMPLATE = """#!/usr/bin/env bash
# 由 scripts/interpret/pl_full_prep.py specs 生成。用法：bash launch_stage.sh <{stages}>
# 为该阶段有 spec 的 worker 各起一个 detached tmux（plf-{model}-<阶段>-<worker>），日志 {logs}/plf-{model}-<阶段>-<worker>.log。
# 顺序：repro → smoke → n → （pl_full_prep donors）→ multieq → swap → seeds；multieq 与 swap/seeds 要先有 donor_trim。
O={out}
st=$1
case "$st" in {stages}) ;; *) echo "未知阶段 $st"; exit 2;; esac
mkdir -p {logs}
for w in {workers}; do
  q=$O/queues/$st-$w.sh
  [ -f "$q" ] && grep -q '^run_spec ' "$q" || continue
  name=plf-{model}-$st-$w; log={logs}/$name.log
  if tmux has-session -t "=$name" 2>/dev/null; then echo "SKIP_EXISTS $name"; continue; fi
  if [ -e "$log" ]; then echo "SKIP_LOG_EXISTS $log（重跑前先把旧日志改名留档）"; continue; fi
  tmux new-session -d -s "$name" "set -o pipefail; PYTHONUNBUFFERED=1 bash $q 2>&1 | tee $log; echo \\"EXIT_CODE=\\$?\\" >> $log"
  echo "STARTED $name $log"
done
"""


# ---------------------------------------------------------------- 公共
def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def _read_json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp.write_bytes(_json_bytes(value))
    os.replace(tmp, path)


def _write_once(path, data, overwrite=False):
    """已存在且内容不同则拒绝覆盖（两模型共用文件不得被悄悄改），--overwrite 例外。"""
    path = Path(path)
    if path.exists() and path.read_bytes() != data and not overwrite:
        raise SystemExit(f"{path} 已存在且内容不同，拒绝覆盖")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp.write_bytes(data)
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--out", required=True)
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("specs")
    p.add_argument("--out", required=True, help="<O>/<M>")
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--commit", required=True, help="运行工作树应处的 commit（env.sh 核对）")
    p.add_argument("--run-tree", default=str(RUN_TREE))
    p.add_argument("--frac", type=float, default=XLA_MEM_FRACTION)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--policy-config", default=None)
    p.add_argument("--model-kind", default=None)
    p.add_argument("--target-contract", default=None)
    p.add_argument("--extra", default=None, help="并入 spec 的额外键（JSON 对象，如 Wan 的 sidecar 键）")
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("donors")
    p.add_argument("--out", required=True, help="<O>/<M>")
    p.add_argument("--wan-gate", action="store_true", help="R 集原样成功 < 12 时只生成 Z、Rnd")
    p.add_argument("--smoke", action="store_true", help="只处理 smoke 阶段录制")
    args = parser.parse_args(argv)
    return {"pool": cmd_pool, "specs": cmd_specs, "donors": cmd_donors}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
