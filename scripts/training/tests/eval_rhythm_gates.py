#!/usr/bin/env python3
"""TIC 节奏层闸门：把「评估到底在哪些时刻做推理、最多做多少次、运动 token 会不会撞预算」从推算变实测。

对应 TIC 计划第二部分 2.3。档位与 `motion_gates_online.py` 相同——CPU、零特征 vision（`_dummy_vision_enc`）、
stub 运动编码（`_stub_enc_local`）、合成帧（`motion_protocol.stub_frame`），不加载任何模型、不起 sidecar、不碰 GPU。

与 `motion_gates_online.py` 的唯一区别是**驱动方式**：那里用简化的 `_drive`（`while t + 16 < T`），
这里把 `examples/robomme/utils.py` 的 `EpisodeState` / `pack_buffer` 原对象挂进来，按
`examples/robomme/eval.py::EpisodeEvaluator.eval_each_episode` 逐语句照抄真实控制流：

    init_episode  → image_buffer 收下 demo 段整段 [0, es]，exec_start_idx = len(buffer) - 1 = es
    while True:
        action_plan 空 → add_buffer(pack_buffer(image_buffer, state_buffer, exec_start_idx)) + infer
                        → action_plan 收 16 个动作 → clear_buffers()（buffer 清空、exec_start_idx 归 0）
        popleft 一个动作 → env.step → count += 1
        count > max_steps(1300) → 记 timeout 并 break（**观测不入 buffer**）
        add_observation → stop_flag → break

由此得到的决策时刻清单由本脚本**独立生成**（`τ_0 = es`，之后 `τ_j = es + 16j`），不写死；
两种终止分别记 `termination=count_guard`（外层 1300 步兜底）与 `termination=env_done`（环境主动截断）。
环境侧建模：episode 共 T_env 帧（帧号 0..T_env−1，其中 0..es 是 demo 段），环境在观测到最后一帧 T_env−1 时
返回 `stop_flag=True`，即环境步数 `C = T_env − 1 − es`。

四条判定行（全部阻断）：

1. `RHYTHM_EQ` —— 同进程内以 `motion_gates_online.py::_drive` 为对照臂，逐点比 τ、es、
   合法运动起点数 k、`mem_order` 的 `_common.leaf_sha256`。证明真实 eval 控制流与既有闸门用的简化驱动同节奏。
2. `EVAL_TERMINATION` —— `max_steps=1300` 兜底 case 的推理次数、最晚决策时刻、首批是否计入、
   末尾未满 16 帧的残批是否被丢弃；另一组用环境截断验证 `termination=env_done` 且推理次数 = ⌊(T−1−es)/16⌋+1。
3. `TAU_LONG` —— 两条撑满 1300 步的 case（es=0 与真实最大 es=216），在最晚决策时刻实测 k、
   是否 raise、`mem_order` 是否合法置换，并给出相对 budget=96 的余量。
4. `ES_BOUNDARY` —— 扫 demo 段长度 es ∈ [200, 320) 找第一个撞 budget 的值，与公式独立预测互校；
   给出真实四任务的 es 取值集合与其最大 k 及余量。

撞上限的后果链（留档须写清，本脚本只负责坐实前两环）：`FrameSampMemory._prepare_motion` raise
→ policy server 侧异常经 websocket 传回 → `examples/robomme/eval.py::evaluate` 的 `except Exception`
把该集记成 `"error"` → 续评时 `setup_log_dict` 从 progress.json 读回后该集被重评、error 条目被覆盖
→ 汇总 `sum(log_dict[task].values())` 遇到字符串 `"error"` 抛 TypeError，又被外层 `except` 吞掉。
即：撞上限不会报警，只会让该集悄悄消失在成功率分母之外。

用法：
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= UV_LINK_MODE=copy uv run --no-sync python \
    scripts/training/tests/eval_rhythm_gates.py [--gate all|rhythm|term|taulong|esbound] [--out <records>/rhythm.json]
唯一成功行：`TIC_RHYTHM=PASS`；任一判定 FAIL 即打 `TIC_RHYTHM=FAIL` 并以非零码退出。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import traceback

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OPENPI_DATA_HOME", str(_REPO_ROOT / "v1-store" / "models"))
sys.path.insert(0, str(_HERE))                                   # motion_gates_online / _common
sys.path.insert(0, str(_REPO_ROOT / "examples" / "robomme"))     # utils.py（EpisodeState / pack_buffer）

import _common as C                                              # noqa: E402
import motion_gates_online as G                                  # noqa: E402
from utils import EpisodeState, pack_buffer                      # noqa: E402

from mme_vla_suite.datastore import motion_store as ms           # noqa: E402
from mme_vla_suite.policies import motion_protocol as P          # noqa: E402
from mme_vla_suite.policies.framesamp_memory import FrameSampMemory  # noqa: E402

MAX_STEPS = 1300              # examples/robomme/eval.py::Args.max_steps
OBS_HORIZON = 16              # examples/robomme/eval.py::Args.obs_horizon（utils.check_args 硬钉 16）
BUDGET = G.MOTION_CFG["budget"]                                  # 96
STATE_DIM = 8
_WRIST = np.zeros((256, 256, 3), np.uint8)                       # wrist 不进 pack_buffer，只喂 add_observation
_ZERO_STATE = np.zeros(STATE_DIM, np.float32)
_ZERO_ACTION = np.zeros(STATE_DIM, np.float32)

# 真实四任务库（v1-store/datasets/4task-motion-400ep/meta/episode_manifest.json）的 demo 段长度取值
# 与各取值下最长 episode 的总帧数，用作 RHYTHM_EQ / EVAL_TERMINATION 的真实样本。
DEF_LIB = "v1-store/datasets/4task-motion-400ep"


def _new_mem() -> FrameSampMemory:
    return FrameSampMemory(vision_enc_fn=G._dummy_vision_enc, motion_enc_fn=G._stub_enc_local,
                           motion_cfg=G.MOTION_CFG)


def _reset(pol, mem) -> None:
    """把 policy + memory 恢复到 `MME_VLA_Policy.reset()` 后的状态（复用实例、避免重建 200 MiB pos 表）。"""
    mem.clear()
    mem.motion_encode_calls = 0
    mem.motion_encode_s = 0.0
    pol.step_idx = -1
    pol.exec_start_idx = 0


# ── 真实 eval 控制流 ──────────────────────────────────────────────────────────

def eval_drive(pol, es: int, t_env: int, per_step, max_steps: int = MAX_STEPS) -> dict:
    """按 `eval.py::EpisodeEvaluator.eval_each_episode` 逐语句照抄；per_step(τ) 在每次 infer 前调用一次。

    per_step 抛出的异常按真实链路的语义原样上抛（policy server 侧 raise → 该集记 error）。
    """
    st = EpisodeState()
    # ── init_episode：pre_traj 是 demo 段整段（帧 0..es），共 es+1 帧 ──
    for i in range(es + 1):
        st.image_buffer.append(P.stub_frame(i))
        st.wrist_image_buffer.append(_WRIST)
        st.state_buffer.append(_ZERO_STATE.copy())
    st.exec_start_idx = len(st.image_buffer) - 1

    taus: list[int] = []
    termination = None
    while True:
        if not st.action_plan:
            # get_action_chunk：先 add_buffer（use_history=True），再 infer 当前观测
            pol.add_buffer(pack_buffer(st.image_buffer, st.state_buffer, st.exec_start_idx))
            taus.append(pol.step_idx)
            per_step(pol.step_idx)
            st.action_plan.extend([_ZERO_ACTION] * OBS_HORIZON)
            st.clear_buffers()

        st.action_plan.popleft()
        # env_runner.step(action)：环境在观测到最后一帧 t_env-1 时给 stop_flag
        st.count += 1
        frame_idx = es + st.count
        stop_flag = frame_idx >= t_env - 1

        if st.count > max_steps:
            termination = "count_guard"       # success_flag = "timeout"，本步观测不入 buffer
            break

        st.add_observation(P.stub_frame(frame_idx), _WRIST, _ZERO_STATE.copy())
        if stop_flag:
            termination = "env_done"
            break

    return {"taus": taus, "termination": termination, "count": st.count,
            "residual_frames": len(st.image_buffer), "es": es, "t_env": t_env}


def expected_taus(es: int, t_env: int, max_steps: int = MAX_STEPS) -> list[int]:
    """脚本独立生成的时刻清单：τ_0 = es，之后 τ_j = es + 16j，直到环境截断或 count 兜底。

    推理发生在循环体开头 `count = 16j` 处；循环体开头出现过的 count 值是 0..min(C, max_steps+1)−1，
    其中 C = t_env − 1 − es 是环境步数（count = C 那一次在体尾 break，不再进入下一轮体首）。
    """
    c_env = t_env - 1 - es
    last_count = min(c_env, max_steps + 1) - 1
    return [es + 16 * j for j in range(last_count // 16 + 1)] if last_count >= 0 else []


def _per_step_collect(pol, entry, out: list):
    """收集每个决策时刻的 (τ, es, k, mem_order sha)；k 同时与训练侧 motion_store 公式交叉核对。

    `es` 一律取 **memory 侧** `FrameSampMemory.exec_start_idx`——它是两臂都真实写入的段边界
    （对照臂 `motion_gates_online._drive` 只驱动 memory、不经 `MME_VLA_Policy.add_buffer`，
    policy 侧的 `exec_start_idx` 在那条臂上从来不被写）。policy 侧的值另记 `es_pol`，
    在真实 eval 臂上额外核对两侧一致。
    """
    def fn(t: int):
        inputs = pol._prepare_history({})
        _, ref_frames = ms.visible_motion_rows(entry, t)
        got_frames = pol.mem_buffer.visible_motion_frames(t)
        out.append({"tau": t, "es": pol.mem_buffer.exec_start_idx, "es_pol": pol.exec_start_idx,
                    "k": len(got_frames), "k_ref": int(len(ref_frames)),
                    "order_sha": C.leaf_sha256(np.asarray(inputs["mem_order"]))})
    return fn


# ── 判定 1：RHYTHM_EQ ────────────────────────────────────────────────────────

def gate_rhythm(cases, rec: dict) -> tuple[str, bool]:
    mem_a, mem_b = _new_mem(), _new_mem()
    pol_a, pol_b = G._bare_policy(mem_a), G._bare_policy(mem_b)
    tau_mis = es_mis = k_mis = sha_mis = 0
    n_points = 0
    details: list[dict] = []
    for es, t_env, tag in cases:
        entry = G._entry(es, t_env)

        # 主臂：eval.py 真实控制流
        _reset(pol_a, mem_a)
        rows_a: list[dict] = []
        info = eval_drive(pol_a, es, t_env, _per_step_collect(pol_a, entry, rows_a))

        # 对照臂：motion_gates_online._drive（while t + 16 < T）
        _reset(pol_b, mem_b)
        rows_b: list[dict] = []

        def per_step_b(t: int, _pol=pol_b, _entry=entry, _rows=rows_b):
            # _drive 只驱动 memory：policy 的步号与段边界手动同步（步号同 gate_p3 的做法）
            _pol.step_idx = t
            _pol.exec_start_idx = _pol.mem_buffer.exec_start_idx
            _per_step_collect(_pol, _entry, _rows)(t)

        G._drive(mem_b, es, t_env, per_step_b)

        # 本脚本独立生成的清单（第三方核对，不取自任一臂）
        want = expected_taus(es, t_env)
        n = min(len(rows_a), len(rows_b))
        if len(rows_a) != len(rows_b):
            tau_mis += abs(len(rows_a) - len(rows_b))
        if [r["tau"] for r in rows_a] != want:
            tau_mis += 1
        for i in range(n):
            a, b = rows_a[i], rows_b[i]
            tau_mis += int(a["tau"] != b["tau"])
            # 段边界三处必须一致：真实 eval 臂 memory 侧、对照臂 memory 侧、以及真实 eval 臂 policy 侧
            es_mis += int(a["es"] != b["es"] or a["es"] != es or a["es_pol"] != a["es"])
            k_mis += int(a["k"] != b["k"] or a["k"] != a["k_ref"] or b["k"] != b["k_ref"])
            sha_mis += int(a["order_sha"] != b["order_sha"])
        n_points += len(rows_a)
        details.append({"tag": tag, "es": es, "t_env": t_env, "points_eval": len(rows_a),
                        "points_drive": len(rows_b), "points_expected": len(want),
                        "tau_max": rows_a[-1]["tau"] if rows_a else None,
                        "k_max": max((r["k"] for r in rows_a), default=0),
                        "termination": info["termination"],
                        "c_env": t_env - 1 - es, "c_env_mod16": (t_env - 1 - es) % 16})

    ok = tau_mis == es_mis == k_mis == sha_mis == 0
    rec["rhythm"] = {"episodes": len(cases), "points": n_points, "details": details,
                     "tau_mismatches": tau_mis, "es_mismatches": es_mis,
                     "k_mismatches": k_mis, "order_sha_mismatches": sha_mis}
    for d in details:
        print(f"  {d['tag']}: es={d['es']} T={d['t_env']} 点数 eval/_drive/独立生成 = "
              f"{d['points_eval']}/{d['points_drive']}/{d['points_expected']} "
              f"τ_max={d['tau_max']} k_max={d['k_max']} 终止={d['termination']} "
              f"C={d['c_env']} C%16={d['c_env_mod16']}")
    line = (f"RHYTHM_EQ={'PASS' if ok else 'FAIL'} episodes={len(cases)} points={n_points} "
            f"tau_mismatches={tau_mis} es_mismatches={es_mis} k_mismatches={k_mis} "
            f"order_sha_mismatches={sha_mis} driver=examples/robomme/utils.py:EpisodeState")
    return line, ok


# ── 判定 2：EVAL_TERMINATION ─────────────────────────────────────────────────

def gate_termination(cases, rec: dict) -> tuple[str, bool]:
    mem = _new_mem()
    pol = G._bare_policy(mem)
    fails: list[str] = []

    # case A：环境永不截断，1300 步兜底生效
    es_a = 0
    t_big = es_a + MAX_STEPS + 400            # 远大于 1300 步所需，stop_flag 永不触发
    _reset(pol, mem)
    entry_a = G._entry(es_a, t_big)
    rows_a: list[dict] = []
    info_a = eval_drive(pol, es_a, t_big, _per_step_collect(pol, entry_a, rows_a))
    taus_a = info_a["taus"]
    want_a = expected_taus(es_a, t_big)
    infer_calls = len(taus_a)
    tau_max = taus_a[-1] if taus_a else None
    first_included = int(bool(taus_a) and taus_a[0] == es_a)
    dropped = int(info_a["residual_frames"] > 0)
    if info_a["termination"] != "count_guard":
        fails.append(f"case A 终止方式 {info_a['termination']} != count_guard")
    if taus_a != want_a:
        fails.append(f"case A 时刻清单与独立生成不符: {len(taus_a)} vs {len(want_a)}")
    if tau_max != es_a + 1296:
        fails.append(f"case A τ_max={tau_max} != es+1296={es_a + 1296}")
    if not first_included:
        fails.append("case A 首批（τ_0 = es）未计入推理时刻")
    if not dropped:
        fails.append("case A 末尾残批未被丢弃（buffer 为空，与 eval.py 语义不符）")

    # case B：环境主动截断（真实四任务 episode，T < 1300）
    rows_b: list[dict] = []
    for es, t_env, tag in cases:
        _reset(pol, mem)
        entry = G._entry(es, t_env)
        sink: list[dict] = []
        info = eval_drive(pol, es, t_env, _per_step_collect(pol, entry, sink))
        got = len(info["taus"])
        want_formula = (t_env - 1 - es) // 16 + 1          # 计划公式 ⌊(T−1−es)/16⌋+1
        want_sim = len(expected_taus(es, t_env))
        rows_b.append({"tag": tag, "es": es, "t_env": t_env, "termination": info["termination"],
                       "infer_calls": got, "formula": want_formula, "simulated": want_sim,
                       "count": info["count"], "residual_frames": info["residual_frames"]})
        if info["termination"] != "env_done":
            fails.append(f"case B {tag} 终止方式 {info['termination']} != env_done")
        if got != want_formula:
            fails.append(f"case B {tag} infer 次数 {got} != ⌊(T−1−es)/16⌋+1 = {want_formula}")
        if got != want_sim:
            fails.append(f"case B {tag} infer 次数 {got} != 独立生成 {want_sim}")
        print(f"  env_done {tag}: es={es} T={t_env} C={t_env - 1 - es} infer={got} "
              f"（公式 {want_formula} / 独立生成 {want_sim}）终止={info['termination']}")

    ok = not fails
    rec["termination"] = {"case_count_guard": {"es": es_a, "t_env": t_big, "infer_calls": infer_calls,
                                               "tau_max": tau_max, "count": info_a["count"],
                                               "residual_frames": info_a["residual_frames"],
                                               "termination": info_a["termination"]},
                          "cases_env_done": rows_b, "fails": fails}
    line = (f"EVAL_TERMINATION={'PASS' if ok else 'FAIL'} max_steps={MAX_STEPS} infer_calls={infer_calls} "
            f"first_batch_included={first_included} tau_max=es+{tau_max - es_a} "
            f"last_partial_batch_dropped={dropped} residual_frames={info_a['residual_frames']} "
            f"termination={info_a['termination']} env_done_cases={len(rows_b)}")
    if fails:
        line += "\n" + "\n".join(f"  {f}" for f in fails)
    return line, ok


# ── 判定 3：TAU_LONG ────────────────────────────────────────────────────────

def gate_tau_long(rec: dict) -> tuple[str, bool]:
    mem = _new_mem()
    pol = G._bare_policy(mem)
    fails: list[str] = []
    rows: list[dict] = []
    for es in (0, 216):                                    # 0 = Button 系；216 = 真实四任务最大 demo 段
        t_big = es + MAX_STEPS + 400
        _reset(pol, mem)
        entry = G._entry(es, t_big)
        sink: list[dict] = []
        raised = ""
        try:
            info = eval_drive(pol, es, t_big, _per_step_collect(pol, entry, sink))
        except RuntimeError as e:                          # budget 越界即在此抛出
            raised = str(e)
            info = {"taus": [r["tau"] for r in sink], "termination": "raise"}
        tau_last = sink[-1]["tau"] if sink else None
        k_last = sink[-1]["k"] if sink else None
        order_legal = 1
        if not raised:
            inputs = pol._prepare_history({})
            order = np.asarray(inputs["mem_order"])
            order_legal = int(order.dtype == np.int32 and np.array_equal(np.sort(order), np.arange(order.size)))
            if not order_legal:
                fails.append(f"es={es} 的 mem_order 不是合法置换（dtype={order.dtype} size={order.size}）")
        else:
            fails.append(f"es={es} 在 τ={tau_last} 意外 raise: {raised[:160]}")
        k_ref = int(len(ms.visible_motion_rows(entry, tau_last)[0])) if tau_last is not None else None
        if k_last != k_ref:
            fails.append(f"es={es} 在线 k={k_last} != 训练侧公式 k={k_ref}")
        if tau_last != es + 1296:
            fails.append(f"es={es} 最晚决策时刻 {tau_last} != es+1296")
        rows.append({"es": es, "tau": tau_last, "k": k_last, "k_ref": k_ref,
                     "raise": int(bool(raised)), "order_legal": order_legal,
                     "headroom": (BUDGET - k_last) if k_last is not None else None})
    headroom_min = min((r["headroom"] for r in rows if r["headroom"] is not None), default=None)
    ok = not fails
    rec["tau_long"] = {"cases": rows, "budget": BUDGET, "headroom_min": headroom_min, "fails": fails}
    line = (f"TAU_LONG={'PASS' if ok else 'FAIL'} cases={len(rows)} "
            + " | ".join(f"es={r['es']} tau={r['tau']} k={r['k']} raise={r['raise']} "
                         f"order_legal={r['order_legal']}" for r in rows)
            + f" budget={BUDGET} headroom_min={headroom_min}")
    if fails:
        line += "\n" + "\n".join(f"  {f}" for f in fails)
    return line, ok


# ── 判定 4：ES_BOUNDARY ─────────────────────────────────────────────────────

def _k_and_raise(mem: FrameSampMemory, es: int, tau: int) -> tuple[int, bool]:
    """轻量档：只走真实的 `visible_motion_frames` + `_prepare_motion`（budget 闸），不跑编码。

    `_history_feats_motion` 直接按合法起点填桩值——`_prepare_motion` 只在缺键时另行 raise，
    填齐后剩下的唯一 raise 来源就是 `k > motion.budget` 这一条，正是本关要找的。
    """
    mem.exec_start_idx = es
    mem._history_feats_motion.clear()
    frames = mem.visible_motion_frames(tau)
    for f in frames:
        mem._history_feats_motion[f] = np.full(G.MOTION_CFG["dim"], float(f), np.float32)
    try:
        mem._prepare_motion(tau)
        return len(frames), False
    except RuntimeError as e:
        if "> motion.budget" not in str(e):
            raise
        return len(frames), True


def gate_es_boundary(real_es: list[int], rec: dict) -> tuple[str, bool]:
    lo, hi = 200, 320
    mem = _new_mem()
    fails: list[str] = []

    scan: list[dict] = []
    first_raise = None
    for es in range(lo, hi):
        k, raised = _k_and_raise(mem, es, es + 1296)       # 最晚决策时刻
        scan.append({"es": es, "k": k, "raise": int(raised)})
        if raised and first_raise is None:
            first_raise = es

    # 独立预测：demo 段可见窗数 ⌊(es−33)/16⌋+1（es ≥ 33）＋ exec 段在 τ=es+1296 处的 80 个 > budget
    def predict_k(es: int, tau: int) -> int:
        demo = ((es - 33) // 16 + 1) if es >= 33 else 0
        ex = ((tau - es - 32) // 16 + 1) if tau - es >= 32 else 0
        return demo + ex
    predicted = next((es for es in range(lo, hi) if predict_k(es, es + 1296) > BUDGET), None)
    if first_raise != predicted:
        fails.append(f"实测首个撞 budget 的 es={first_raise} != 公式预测 {predicted}")
    k288 = next(r["k"] for r in scan if r["es"] == 288)
    k289 = next(r["k"] for r in scan if r["es"] == 289)
    if k288 > BUDGET or k289 <= BUDGET:
        fails.append(f"边界两侧不符：k_at_288={k288}（应 ≤ {BUDGET}）k_at_289={k289}（应 > {BUDGET}）")

    # 边界两侧各跑一遍完整 eval 控制流（真编码、真装配），确认 288 不 raise、289 在最晚时刻 raise
    mem_full = _new_mem()
    pol = G._bare_policy(mem_full)
    full: list[dict] = []
    for es in (288, 289):
        _reset(pol, mem_full)
        t_big = es + MAX_STEPS + 400
        entry = G._entry(es, t_big)
        sink: list[dict] = []
        raised_at = None
        msg = ""
        try:
            eval_drive(pol, es, t_big, _per_step_collect(pol, entry, sink))
        except RuntimeError as e:
            msg = str(e)
            raised_at = (sink[-1]["tau"] + 16) if sink else es
            if "> motion.budget" not in msg:
                fails.append(f"es={es} 完整驱动 raise 但不是 budget 闸: {msg[:160]}")
        full.append({"es": es, "raise": int(raised_at is not None), "raise_at_tau": raised_at,
                     "points": len(sink), "k_last": sink[-1]["k"] if sink else None, "msg": msg[:200]})
    if full[0]["raise"]:
        fails.append(f"es=288 完整驱动竟 raise: {full[0]['msg']}")
    if not full[1]["raise"]:
        fails.append("es=289 完整驱动未 raise（budget 闸没拦住）")
    elif full[1]["raise_at_tau"] != 289 + 1296:
        fails.append(f"es=289 raise 发生在 τ={full[1]['raise_at_tau']}，预期最晚决策时刻 {289 + 1296}")

    # 真实四任务的 es 取值与其在最晚决策时刻的 k
    real = [{"es": es, "k": _k_and_raise(mem, es, es + 1296)[0]} for es in sorted(real_es)]
    real_max_k = max(r["k"] for r in real)
    margin = BUDGET - real_max_k
    if real_max_k > BUDGET:
        fails.append(f"真实 es 取值的最大 k={real_max_k} 已超 budget {BUDGET}")

    ok = not fails
    rec["es_boundary"] = {"scanned": [lo, hi], "first_raise_es": first_raise, "predicted": predicted,
                          "k_at_288": k288, "k_at_289": k289, "full_drive": full,
                          "real": real, "real_max_k": real_max_k, "margin_windows": margin,
                          "budget": BUDGET, "fails": fails}
    print(f"  完整驱动：es=288 raise={full[0]['raise']} 点数={full[0]['points']} k_last={full[0]['k_last']} | "
          f"es=289 raise={full[1]['raise']} @τ={full[1]['raise_at_tau']} 点数={full[1]['points']}")
    print(f"  真实 es → 最晚决策时刻 k: " + " ".join(f"{r['es']}:{r['k']}" for r in real))
    line = (f"ES_BOUNDARY={'PASS' if ok else 'FAIL'} scanned=[{lo},{hi}) first_raise_es={first_raise} "
            f"predicted={predicted} k_at_288={k288} k_at_289={k289} "
            f"real_es_values=[{','.join(str(r['es']) for r in real)}] real_max_k={real_max_k} "
            f"margin_windows={margin}")
    if fails:
        line += "\n" + "\n".join(f"  {f}" for f in fails)
    return line, ok


# ── 真实样本：每个 demo 段长度取最长的一条 episode ────────────────────────────

def load_real_cases(lib: pathlib.Path) -> list[tuple[int, int, str]]:
    from mme_vla_suite.datastore.manifest import load_manifest
    manifest = load_manifest(lib / "meta" / "episode_manifest.json")
    best: dict[int, dict] = {}
    for ep in manifest["episodes"]:
        es = int(ep["exec_start_idx"])
        if es not in best or int(ep["num_timesteps"]) > int(best[es]["num_timesteps"]):
            best[es] = ep
    out = []
    for es in sorted(best):
        ep = best[es]
        out.append((es, int(ep["num_timesteps"]),
                    f"g{ep['global_episode_idx']}-{ms.task_of_h5(ep['h5_file'])}"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="TIC 节奏层闸门（CPU + stub）")
    ap.add_argument("--gate", default="all", choices=["all", "rhythm", "term", "taulong", "esbound"])
    ap.add_argument("--lib", default=DEF_LIB, help="取真实 es / T 取值的库根")
    ap.add_argument("--out", default="", help="判定明细 JSON 落点（可选）")
    args = ap.parse_args()

    lib = pathlib.Path(args.lib)
    if not lib.is_absolute():
        lib = _REPO_ROOT / lib
    cases = load_real_cases(lib)
    real_es = [c[0] for c in cases]
    print(f"真实样本（每个 demo 段长度取最长一条）: " + " ".join(f"es={a} T={b} {c}" for a, b, c in cases))

    rec: dict = {"schema": "tic-rhythm-v1", "lib": str(lib),
                 "real_cases": [{"es": a, "t_env": b, "tag": c} for a, b, c in cases]}
    lines: list[str] = []
    results: list[bool] = []

    def run(name: str, fn):
        print(f"== {name} ==", flush=True)
        try:
            line, ok = fn()
        except Exception:
            traceback.print_exc()
            line, ok = f"{name}=FAIL（异常，见上方 traceback）", False
        print(line, flush=True)
        lines.append(line)
        results.append(ok)

    sel = ["rhythm", "term", "taulong", "esbound"] if args.gate == "all" else [args.gate]
    if "rhythm" in sel:
        run("RHYTHM_EQ", lambda: gate_rhythm(cases, rec))
    if "term" in sel:
        run("EVAL_TERMINATION", lambda: gate_termination(cases, rec))
    if "taulong" in sel:
        run("TAU_LONG", lambda: gate_tau_long(rec))
    if "esbound" in sel:
        run("ES_BOUNDARY", lambda: gate_es_boundary(real_es, rec))

    ok = all(results)
    rec["lines"] = lines
    rec["pass"] = ok
    if args.out:
        out = pathlib.Path(args.out)
        if not out.is_absolute():
            out = _REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"明细写入 {out}")
    print(f"TIC_RHYTHM={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
