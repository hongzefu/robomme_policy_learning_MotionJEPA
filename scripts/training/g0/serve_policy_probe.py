#!/usr/bin/env python3
"""L6 闭环探针版 policy server（TIC 计划 2.5，审计第 3 条）：仿 `scripts/training/serve_policy.py` 起同一个
`MME_VLA_Policy`，另在**实例属性**上遮蔽三处，把每次 infer 的在线记忆装配现场抓成 jsonl。

**主线零改动**：不改 `src/mme_vla_suite/policies/*`，也不改 `examples/robomme/*`。遮蔽全部落在
`create_trained_policy` 返回的那个实例上：

- `policy.reset`      —— 包一层，先调原 `MME_VLA_Policy.reset`，再 `episode_seq += 1` 并把 `infer_seq` 归零。
                         注意 `__init__` 内部那次 `reset()` 发生在遮蔽之前，故第一集的 `episode_seq` 为 1。
- `policy._prepare_history` —— 包一层，先调原方法拿装配结果，再按当刻 `policy.step_idx` / `policy.exec_start_idx`
                         采集帧路 / 运动路的**完整数组**（不是只记 sha），暂存进 pending。
- `policy._input_transform` —— 本身就是实例属性（`_transforms.compose(...)`），包一层拿 tokenize 之后的
                         `tokenized_prompt` / `tokenized_prompt_mask` / `state`，与 pending 合并后落一行 jsonl。

`infer()` 内固定是 `_prepare_history` → `_input_transform` 的顺序（`policies/policy.py::MME_VLA_Policy.infer`），
所以在第二处 flush 一行、且每行必然配对；pending 未被消费即说明链路变了，显式 raise。

**为什么记完整数组**：汇总器 `summarize_eval_probe.py` 要在不 import `shared/sampling` 的前提下**独立重算**
次序表与 k 公式并逐元素比对（EVAL_K_FORMULA / EVAL_ORDER_LEGAL）。只记 sha 无法核排列合法性、长度与 dtype。

启动时打一行 `PROBE_ENV`，给出 jax 后端与两张位置编码表的指纹：
  - `pos_table_sha`  = 在线 `policy.mem_buffer.pos_emb_4x4` 前 `pos_rows` 行（`FrameSampMemory.__init__` 里
                       `PosEmb3D(dim)(arange(4096), 4)` 现算，GPU 上算与 CPU 上算不逐位，见 P5 留档）；
  - `store_pos_sha`  = 训练库 `FrameSampStore.pos_rows(arange(pos_rows))`（离线建库那张表）。
  `pos_rows` 取库侧 `store_meta.num_pos_rows`（400ep 库为 586 = 最长 episode 帧数），在线表 4096 行只能比这个前缀；
  在线全表 sha 另记 `pos_table_full_sha` / `pos_table_full_rows` 供留档。

用法（正式；micromamba robomme 侧的 eval.py 照常连这个端口）：
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
    uv run --no-sync python scripts/training/g0/serve_policy_probe.py \
      --ckpt v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999 \
      --lib v1-store/datasets/4task-motion-400ep --port 8041 --seed 42 \
      --probe-out v1-store/reports/tic/eval_probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import socket
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if not (_REPO_ROOT / "pyproject.toml").is_file():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "training" / "tests"))
_V1 = pathlib.Path(os.environ.get("MMEVLA_V1_STORE", str(_REPO_ROOT / "v1-store")))
os.environ.setdefault("OPENPI_DATA_HOME", str(_V1 / "models"))

import _common as C  # noqa: E402  （leaf_sha256 与 compare_* 系列同一份哈希口径）


def _mask_bits(arr) -> str:
    """把布尔数组压成 '0'/'1' 串（长度即元素数）；非布尔即 raise。"""
    a = np.asarray(arr)
    if a.dtype != np.bool_:
        raise RuntimeError(f"mask dtype {a.dtype} != bool_（交付形制变了）")
    return "".join("1" if bool(x) else "0" for x in a.reshape(-1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint 目录（含 params/ 与 assets/，如 .../39999）")
    ap.add_argument("--lib", default=str(_V1 / "datasets/4task-motion-400ep"),
                    help="训练库根（取 framesamp/ 的 pos 小表算 store_pos_sha）")
    ap.add_argument("--port", type=int, default=8041)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe-out", required=True, help="jsonl 输出路径（每次 infer 一行）")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--config", default="mme_vla_suite_b128", help="训练配置条目（生产口径 mme_vla_suite_b128）")
    ap.add_argument("--motion-gpu", type=int, default=None,
                    help="仅开发自检 / 资源冲突时用：把 motion sidecar 落到指定物理 GPU。"
                         "缺省即生产路径——由 create_trained_policy 按 run 快照的 motion.online_gpu 起 sidecar")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="仅开发自检用：放行 jax CPU 后端（在线 pos 表在 CPU 上与库表不逐位，P5 留档）")
    args = ap.parse_args()

    import jax

    from mme_vla_suite.datastore.framesamp_store import FrameSampStore, StoreMeta
    from mme_vla_suite.policies import policy_config as _policy_config
    from mme_vla_suite.serving import websocket_policy_server
    from mme_vla_suite.training import config as _config

    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_cpu:
        raise SystemExit(f"错误: 主进程 jax 后端 {backend} != gpu（PosEmb3D 4x4 表 CPU/GPU 不逐位，P5 留档）")

    ckpt_dir = pathlib.Path(args.ckpt)
    lib = pathlib.Path(args.lib)
    probe_out = pathlib.Path(args.probe_out)
    probe_out.parent.mkdir(parents=True, exist_ok=True)
    if probe_out.exists():
        raise SystemExit(f"错误: probe 输出已存在，拒绝追加混行: {probe_out}")

    train_config = _config.get_config(args.config)
    if args.motion_gpu is None:
        # 生产路径：与 scripts/training/serve_policy.py 逐字同式，sidecar 落在 run 快照的 motion.online_gpu
        policy = _policy_config.create_trained_policy(train_config, ckpt_dir, seed=args.seed)
    else:
        # 换卡路径（沿用 compare_siglip_replay.py 的既有做法）：先用 stub 过构造（stub 子进程随即关闭），
        # 再按同一份 run provenance 自建真 sidecar 客户端落到 --motion-gpu，换进去后重建 mem_buffer。
        from mme_vla_suite.policies.motion_client import MotionEncoderClient
        policy = _policy_config.create_trained_policy(train_config, ckpt_dir, seed=args.seed, motion_stub=True)
        if not policy.motion_enabled:
            raise SystemExit("错误: --motion-gpu 只对 motion 开启态的 run 有意义")
        policy._motion_client.close()
        prov = json.loads((ckpt_dir.parent / "motion_provenance.json").read_text(encoding="utf-8"))
        store_prov = {"vae": prov.get("vae"), "encoder": prov.get("encoder")}
        if store_prov["vae"] is None or store_prov["encoder"] is None:
            raise SystemExit(f"错误: run 缺 motion_provenance.json 的 vae / encoder 字段: {ckpt_dir.parent}")
        policy._motion_client = MotionEncoderClient(
            online_gpu=args.motion_gpu, stub=False, store_provenance=store_prov,
            expected_ckpt_sha256=(store_prov["encoder"] or {}).get("checkpoint_sha256"))
        policy.reset()          # 重建 FrameSampMemory，注入换卡后的编码句柄

    # ── 两张 pos 表指纹（在线现算 vs 训练库离线表）────────────────────────────────
    fmeta = StoreMeta.load(lib / "framesamp")
    store = FrameSampStore(lib / "framesamp", meta=fmeta)
    try:
        n_pos = int(fmeta.num_pos_rows)
        pos_online = np.ascontiguousarray(np.asarray(policy.mem_buffer.pos_emb_4x4))
        if pos_online.shape[0] < n_pos:
            raise SystemExit(f"错误: 在线 pos 表只有 {pos_online.shape[0]} 行 < 库 {n_pos} 行")
        pos_store = np.ascontiguousarray(store.pos_rows(np.arange(n_pos, dtype=np.int64)))
        pos_table_sha = C.leaf_sha256(np.ascontiguousarray(pos_online[:n_pos]))
        store_pos_sha = C.leaf_sha256(pos_store)
        pos_full_sha = C.leaf_sha256(pos_online)
        pos_full_rows = int(pos_online.shape[0])
    finally:
        store.close()

    cfg = policy.config
    print(
        "PROBE_ENV"
        f" backend={backend}"
        f" devices={','.join(str(d) for d in jax.devices())}"
        f" pos_rows={n_pos}"
        f" pos_table_sha={pos_table_sha}"
        f" store_pos_sha={store_pos_sha}"
        f" pos_table_full_sha={pos_full_sha}"
        f" pos_table_full_rows={pos_full_rows}"
        f" motion_enabled={int(bool(policy.motion_enabled))}"
        f" motion_gpu_override={'none' if args.motion_gpu is None else args.motion_gpu}"
        f" budget={int(cfg.budget)}"
        f" token_per_image={int(cfg.token_per_image)}"
        f" num_views={int(cfg.num_views)}"
        f" motion_budget={int(cfg.motion.budget) if policy.motion_enabled else 0}"
        f" max_token_len={int(policy._model.max_token_len)}"
        f" discrete_state_input={int(bool(train_config.model.discrete_state_input))}"
        f" state_dim={int(np.asarray(policy.state_norm_stats.mean).reshape(-1).shape[0])}"
        f" config={args.config}"
        f" ckpt={ckpt_dir}"
        f" lib={lib}"
        f" probe_out={probe_out}"
        f" seed={args.seed}",
        flush=True,
    )

    # ── 实例属性遮蔽（主线 src/ 零改动）──────────────────────────────────────────
    fh = probe_out.open("w", encoding="utf-8")
    st = {"episode_seq": 0, "infer_seq": 0, "pending": None}
    _orig_reset = policy.reset
    _orig_prepare_history = policy._prepare_history
    _orig_input_transform = policy._input_transform

    def probe_reset() -> None:
        _orig_reset()
        st["episode_seq"] += 1
        st["infer_seq"] = 0
        if st["pending"] is not None:
            raise RuntimeError("reset 时仍有未落盘的 pending 记录（_prepare_history 与 _input_transform 未配对）")

    def probe_prepare_history(inputs: dict) -> dict:
        out = _orig_prepare_history(inputs)
        if st["pending"] is not None:
            raise RuntimeError("上一条 pending 未被 _input_transform 消费（调用链变了）")
        t = int(policy.step_idx)
        es = int(policy.exec_start_idx)
        mb = policy.mem_buffer
        frames_sampled = [int(x) for x in mb.get_frame_sampling_indices(t, int(cfg.budget), int(cfg.token_per_image))]
        motion_frames = [int(x) for x in mb.visible_motion_frames(t)] if policy.motion_enabled else []
        mem_order = np.asarray(out["mem_order"]) if policy.motion_enabled else np.zeros(0, np.int32)
        st["infer_seq"] += 1
        st["pending"] = {
            "episode_seq": st["episode_seq"],
            "infer_seq": st["infer_seq"],
            "t": t,
            "es": es,
            "frames_sampled": frames_sampled,
            "motion_frames": motion_frames,
            "k": len(motion_frames),
            "static_mask": _mask_bits(out["static_mask"]),
            "motion_mask": _mask_bits(out["motion_mask"]) if policy.motion_enabled else "",
            "mem_order": [int(x) for x in mem_order.reshape(-1)],
            "mem_order_dtype": str(mem_order.dtype),
            "mem_order_len": int(mem_order.size),
            "motion_emb_sha": C.leaf_sha256(np.asarray(out["motion_emb"])) if policy.motion_enabled else "",
            "motion_pos_sha": C.leaf_sha256(np.asarray(out["motion_pos"])) if policy.motion_enabled else "",
            "static_image_emb_sha": C.leaf_sha256(np.asarray(out["static_image_emb"])),
            "static_pos_emb_sha": C.leaf_sha256(np.asarray(out["static_pos_emb"])),
            "static_state_emb_sha": C.leaf_sha256(np.asarray(out["static_state_emb"])),
            "prompt_text": inputs.get("prompt"),
        }
        return out

    def probe_input_transform(inputs: dict) -> dict:
        out = _orig_input_transform(inputs)
        rec = st["pending"]
        st["pending"] = None
        if rec is None:
            raise RuntimeError("_input_transform 被调用时没有 pending 记录（调用链变了）")
        tok = np.asarray(out["tokenized_prompt"]).reshape(-1)
        tok_mask = np.asarray(out["tokenized_prompt_mask"]).reshape(-1)
        rec["tok_ids"] = [int(x) for x in tok]
        rec["tok_mask"] = [int(bool(x)) for x in tok_mask]
        # state_tok：归一化后的 state。model_transforms 顺序是 TokenizePromptWithState → PadStatesAndActions，
        # 故 tokenizer 看到的是**未右填零之前**的前 state_dim 维；这里记完整数组（尾部是 pad 的 0），
        # 汇总器按需取前缀。本 run 的 `discrete_state_input=False`，tokenizer 根本不消费 state，
        # 但仍记录，好让汇总器在完全独立的前提下重算 tokenize(prompt_text, state) 并与 tok_ids 逐位比
        # （计划字段之外的补充，只增不减）。
        rec["state_tok"] = [float(x) for x in np.asarray(out["state"]).reshape(-1)]
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        return out

    policy.reset = probe_reset
    policy._prepare_history = probe_prepare_history
    policy._input_transform = probe_input_transform

    hostname = socket.gethostname()
    logging.info("Creating probe server (host: %s, ip: %s)", hostname, socket.gethostbyname(hostname))
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=policy.metadata,
    )
    try:
        server.serve_forever()
    finally:
        fh.close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    sys.exit(main())
