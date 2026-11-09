"""context 2048（32 帧 × 8×8，无 motion）mixed8-200 训练前后核对（docs/root-archive/1018-mixed8-2048-context-modul-plan.md 第二部分 §3 G2–G5）。

照 check_32frame_modul.py 的写法，YAML 换成 perceptual-framesamp-context-32frame-8x8.yaml，数据换 mixed8 库。四个子命令，
各自末行打印判定行：

- short-hist（G2，CPU，只读清单）：mixed8 执行样本里 step<31（不满 32 帧）的数量 → SHORT_HIST=<n>/<总执行样本数>
- online（G3，CPU）：FrameSampMemory + policy 侧 _prepare_history 按 budget 装配 5 步 → ONLINE_ASM_CTX2048=PASS steps=5 mem_len=2048
- smoke-summary（G4）：训练日志 + nvidia-smi csv 汇总 → SMOKE20=PASS|FAIL oom=… loss_finite=… peak_mib=… steady_s_per_step=…
- ckpt（G5）：train-runs/mme_vla_suite_b128_80k/<run>/ 最新 step 经 create_trained_policy 加载 →
  CKPT_LOAD_CTX2048=PASS leaves_equal=1 mem_leaves_changed=<n> actions_match=1 mem_len=2048

路径一律按仓库根 ROOT 解析；须从仓库根以 uv run 运行（get_history_config 按相对 cwd 找 yaml）。
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
YAML = "perceptual-framesamp-context-32frame-8x8.yaml"
NAME = "16task-mixed8-200-3200ep-e3c7564f"
DS = ROOT / "v1-store/datasets" / NAME
ASSETS = ROOT / "v1-store/train-assets/mme_vla_suite" / NAME
CONFIG = "mme_vla_suite_b128_80k"
BUDGET = 2048
TOKENS = 64
COUNT = BUDGET // TOKENS
KEYS = ("static_image_emb", "static_pos_emb", "static_state_emb", "static_mask")
NONE_KEYS = ("motion_emb", "motion_pos", "motion_mask", "mem_order")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def exact(a, b):
    if a is None or b is None:
        return a is b
    aa, bb = np.asarray(a), np.asarray(b)
    return aa.dtype == bb.dtype and aa.shape == bb.shape and aa.tobytes() == bb.tobytes()


def sample_indices_oracle(step, count):
    """独立保留既有 float64 步长、正数整数截断和末端固定（同 check_32frame_modul.py）。"""
    if step < count:
        return list(range(step + 1))
    spacing = float(step) / (count - 1)
    result = [int(float(index) * spacing) for index in range(count)]
    result[-1] = int(step)
    return result


def config():
    from mme_vla_suite.models.config.utils import get_history_config
    hc = get_history_config(YAML)
    require(type(hc.budget) is int and (hc.budget, hc.token_per_image, hc.num_views) == (BUDGET, TOKENS, 1),
            "context-32frame yaml 形制不是 (2048,64,1)")
    require(str(hc.integration_type) == "context" and int(hc.memory_token_dim) == 2048, "yaml 不是 context/2048")
    mcfg = hc.get("motion", None)
    require(not (mcfg is not None and mcfg.get("enabled", False)), "yaml motion 未关闭")
    return hc


def data_config(a):
    from openpi.shared.normalize import deserialize_json
    return SimpleNamespace(norm_stats=deserialize_json(Path(a.norm_stats).read_text()), use_quantile_norm=True)


def dataset(a):
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset
    return FrameSampDataset(a.store, data_config(a), config(), 20, source_root=a.source, manifest_path=a.manifest)


# ── G2：短历史统计（只读清单，不加载 embedding）─────────────────────────────────


def short_hist(a):
    from mme_vla_suite.datastore.framesamp_store import StoreMeta, build_exec_lookup
    from mme_vla_suite.datastore.manifest import load_manifest
    manifest = load_manifest(a.manifest)
    _, step_of, _ = build_exec_lookup(manifest)
    total = int(step_of.shape[0])
    meta = StoreMeta.load(Path(a.store))
    require(total == int(meta.num_exec_samples), f"清单执行样本数 {total} != 库 meta {meta.num_exec_samples}")
    short = int(np.count_nonzero(step_of < COUNT - 1))
    print(f"SHORT_HIST_DETAIL min_step={int(step_of.min())} max_step={int(step_of.max())} "
          f"episodes={len(manifest['episodes'])} max_frames={COUNT}")
    print(f"SHORT_HIST={short}/{total}")


# ── G3：context 2048 在线装配（CPU）──────────────────────────────────────────────


def online(a):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import openpi.shared.array_typing as at
    from mme_vla_suite.policies.framesamp_memory import FrameSampMemory
    from mme_vla_suite.policies.policy import MME_VLA_Policy
    from mme_vla_suite.models.representation.percep_mem import PerceptualMemory
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0
    hc = config()

    def encoder(x):
        # 帧号随像素编码进桩特征，避免全零/全一桩漏掉错选帧或重排（同 check_32frame_modul.py online）。
        values = jnp.rint((x[:, :, 0, 0, 0] + 1) * 127.5)
        return jnp.broadcast_to(values[..., None, None], (*x.shape[:2], TOKENS, 2048)).astype(jnp.bfloat16)

    ns = SimpleNamespace(q01=np.linspace(-1.0, -0.5, 8), q99=np.linspace(0.5, 1.0, 8), mean=np.zeros(8), std=np.ones(8))
    # policy 侧真实 add_buffer / _prepare_history，self 用最小替身（不建大模型）
    fake = SimpleNamespace(config=hc, motion_enabled=False, use_quantiles=True, state_norm_stats=ns,
                           step_idx=-1, exec_start_idx=0)
    fake._normalize_state = lambda s: MME_VLA_Policy._normalize_state(fake, s)
    fake.mem_buffer = FrameSampMemory(
        num_views=int(hc.num_views), img_emb_dim=int(hc.memory_feature.img.input_dim),
        pos_emb_dim=int(hc.memory_feature.pos.input_dim), state_emb_dim=int(hc.memory_feature.state.input_dim),
        vision_enc_fn=encoder, token_per_image=int(hc.token_per_image))
    mem = PerceptualMemory(hc, nnx.Rngs(42))
    model_fake = SimpleNamespace(mem_encoder=mem, history_config=hc)
    steps = (0, 15, 30, 31, 177)
    mem_lens = set()
    for t in steps:
        previous = fake.step_idx + 1
        frame_ids = np.arange(previous, t + 1)
        images = np.broadcast_to(frame_ids[:, None, None, None, None], (len(frame_ids), 1, 224, 224, 3)).astype(np.uint8)
        states = np.broadcast_to(frame_ids[:, None], (len(frame_ids), 8)).astype(np.float32)
        MME_VLA_Policy.add_buffer(fake, {"images": images, "state": states})
        require(fake.step_idx == t, "policy add_buffer 步号不符")
        inputs = MME_VLA_Policy._prepare_history(fake, {})
        selected = sample_indices_oracle(t, COUNT)
        require(list(fake.mem_buffer.get_frame_sampling_indices(t, BUDGET, TOKENS)) == selected, "在线选帧与独立 float64 规则不同")
        img, pos, state, mask = (inputs[k] for k in KEYS)
        for x, shape in zip((img, pos, state, mask), ((BUDGET, 2048), (BUDGET, 768), (BUDGET, 8), (BUDGET,)), strict=True):
            require(x.shape == shape, f"在线装配 shape 错误: {x.shape} != {shape}")
        require(str(img.dtype) == "bfloat16" and str(pos.dtype) == "float32" and str(mask.dtype) == "bool", "在线装配 dtype 错误")
        require(np.isfinite(np.asarray(state, np.float64)).all(), "归一化后 state 非有限")
        require(int(mask.sum()) == TOKENS * min(t + 1, COUNT), "在线 mask 计数错误")
        require(np.all(img[mask].astype(np.float32) == np.repeat(selected, TOKENS)[:, None]), "桩特征的帧身份错误")
        require(exact(pos[mask], fake.mem_buffer.pos_emb[selected].reshape(-1, 768)), "在线位置表选帧或展开错误")
        require(all(not np.any(np.asarray(x)[~mask]) for x in (img, pos)), "在线补零尾部错误")
        require(all(k not in inputs for k in NONE_KEYS), "无 motion 档不应装配 motion 键")
        obs = SimpleNamespace(static_image_emb=jnp.asarray(img)[None], static_pos_emb=jnp.asarray(pos)[None],
                              static_state_emb=jnp.asarray(state, jnp.float32)[None], static_mask=jnp.asarray(mask)[None],
                              motion_emb=None, motion_pos=None, motion_mask=None, mem_order=None)
        with at.disable_typechecking():
            tokens, tmask, _, _ = HistoryPi0.embed_memory(model_fake, obs)
        require(tokens.shape == (1, BUDGET, int(hc.memory_token_dim)) and tmask.shape == (1, BUDGET), f"记忆 token 形状错误: {tokens.shape}")
        require(bool(jnp.isfinite(tokens.astype(jnp.float32)).all()), "记忆 token 非有限")
        mem_lens.add(int(tokens.shape[1]))
        print(f"ONLINE_STEP t={t} frames={min(t + 1, COUNT)} mask_sum={int(mask.sum())} tokens={tuple(tokens.shape)}", flush=True)
    require(mem_lens == {BUDGET}, f"mem_len 不是 {BUDGET}: {mem_lens}")
    del fake, mem
    gc.collect()
    jax.clear_caches()
    print(f"ONLINE_ASM_CTX2048=PASS steps={len(steps)} mem_len={BUDGET}")


# ── G4：20 步短测汇总 ──────────────────────────────────────────────────────────

_TQDM = re.compile(r"(\d+)/(\d+) \[(?:(\d+):)?(\d+):(\d+)<")
_RATE = re.compile(r"([\d.]+)(s/it|it/s)")
_STEP = re.compile(r"^Step (\d+): (.*)$")


def _num(text):
    return float(text.strip().split()[0])


def smoke_summary(a):
    raw = Path(a.log).read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    lines = raw.splitlines()
    oom = int(bool(re.search(r"RESOURCE_EXHAUSTED|out of memory", raw, re.IGNORECASE)))
    codes = [ln.split("=", 1)[1].strip() for ln in lines if ln.startswith("EXIT_CODE=")]
    exit_code = codes[-1] if codes else "<缺>"
    losses = []
    for ln in lines:
        m = _STEP.match(ln.strip())
        if m:
            for kv in m.group(2).split(","):
                k, _, v = kv.strip().partition("=")
                if k == "loss":
                    losses.append(float(v))
    loss_finite = int(bool(losses) and all(math.isfinite(x) for x in losses))
    # 稳态步时：tqdm 已完成步数与 elapsed，取 n≥5（避开首步编译）到末条的差分；不足时退回末条 rate
    progress = {}
    for m in _TQDM.finditer(raw):
        n = int(m.group(1))
        elapsed = int(m.group(3) or 0) * 3600 + int(m.group(4)) * 60 + int(m.group(5))
        progress[n] = elapsed
    warm = [n for n in sorted(progress) if n >= 5]
    steady = float("nan")
    if len(warm) >= 2 and warm[-1] > warm[0]:
        steady = (progress[warm[-1]] - progress[warm[0]]) / (warm[-1] - warm[0])
    else:
        rates = _RATE.findall(raw)
        if rates:
            value, unit = rates[-1]
            steady = float(value) if unit == "s/it" else 1.0 / float(value)
    gpus = [int(x) for x in a.gpus.split(",")]
    peaks = {g: None for g in gpus}
    header = None
    for ln in Path(a.nvml).read_text(encoding="utf-8", errors="replace").splitlines():
        cells = [c.strip() for c in ln.split(",")]
        if cells and cells[0] == "index":
            header = cells
            continue
        if header is None or len(cells) != len(header) or not cells[0].isdigit():
            continue
        g = int(cells[0])
        if g in peaks:
            used = _num(cells[header.index(next(h for h in header if h.startswith("memory.used")))])
            peaks[g] = used if peaks[g] is None else max(peaks[g], used)
    print("SMOKE20_DETAIL " + " ".join(f"gpu{g}_peak_mib={'<无采样>' if v is None else int(v)}" for g, v in peaks.items())
          + f" loss_lines={len(losses)} exit_code={exit_code} tqdm_points={len(progress)}")
    have_all = all(v is not None for v in peaks.values())
    peak = int(max(v for v in peaks.values() if v is not None)) if any(v is not None for v in peaks.values()) else "nan"
    ok = oom == 0 and loss_finite == 1 and exit_code == "0" and have_all
    steady_text = "nan" if math.isnan(steady) else f"{steady:.3f}"
    print(f"SMOKE20={'PASS' if ok else 'FAIL'} oom={oom} loss_finite={loss_finite} peak_mib={peak} steady_s_per_step={steady_text}")


# ── G5：100 步 checkpoint 加载核对 ─────────────────────────────────────────────


def _flat(tree):
    from flax import traverse_util
    return {"/".join(str(p) for p in k): v for k, v in traverse_util.flatten_dict(tree).items()}


def ckpt(a):
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    from flax import nnx, traverse_util
    import openpi.shared.array_typing as at
    from openpi.models.model import restore_params
    from openpi.training.data_loader import transform_dataset, _collate_fn
    from mme_vla_suite.training.config import get_config
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    from mme_vla_suite.policies.policy_config import create_trained_policy, _load_resolved_snapshot
    run_root = ROOT / "v1-store/train-runs" / CONFIG / a.run
    steps = sorted(int(p.name) for p in run_root.iterdir() if p.is_dir() and p.name.isdigit())
    require(steps, f"{run_root} 下没有 step 目录")
    ck = run_root / str(steps[-1])
    print(f"CKPT_DIR={ck} steps={steps}", flush=True)
    hc, motion = _load_resolved_snapshot(run_root)
    require((hc.budget, hc.token_per_image, hc.num_views, str(hc.integration_type), int(hc.memory_token_dim), motion)
            == (BUDGET, TOKENS, 1, "context", 2048, False), "run 快照形制不是 context 2048 无 motion")
    tc = get_config(CONFIG)
    tc = dataclasses.replace(tc, model=dataclasses.replace(tc.model, history_config=hc, use_history=True),
                             data=dataclasses.replace(tc.data, assets=dataclasses.replace(
                                 tc.data.assets, assets_dir=str(Path(a.norm_stats).parent.parent), asset_id="robomme")))
    saved = _flat(restore_params(ck / "params", restore_type=np.ndarray))
    policy = create_trained_policy(tc, ck, seed=42, norm_stats=data_config(a).norm_stats)
    loaded = _flat(nnx.state(policy._model, nnx.Param).to_pure_dict())
    require(set(loaded) == set(saved), f"加载叶集合与保存不同: missing={sorted(set(saved)-set(loaded))[:5]} extra={sorted(set(loaded)-set(saved))[:5]}")
    leaves_equal = all(exact(np.asarray(jax.device_get(loaded[k])), saved[k].astype(ml_dtypes.bfloat16)) for k in saved)
    require(leaves_equal, "加载叶与保存（bf16）不一致")
    print(f"CKPT_LEAVES leaves={len(saved)} equal=1", flush=True)

    # 初态：复现 train.py 的 rng 链（key(seed) → split 取 init_rng → init 内 split 取 model_rng），
    # jit 内只输出记忆叶，其余参数初始化被 XLA 消掉；记忆叶不在 pi05_base 里，初态即 create 的值。
    _, init_rng = jax.random.split(jax.random.key(tc.seed))
    _, model_rng = jax.random.split(init_rng)
    mem_keys = sorted(k for k in saved if "mem" in k)
    require(mem_keys, "checkpoint 里没有记忆相关叶")

    @jax.jit
    def init_mem(rng):
        flat = _flat(nnx.state(tc.model.create(rng), nnx.Param).to_pure_dict())
        return {k: flat[k] for k in mem_keys}
    initial = jax.device_get(init_mem(model_rng))
    changed, changed_bf16 = 0, 0
    for k in mem_keys:
        init_arr = np.asarray(initial[k])
        require(init_arr.shape == saved[k].shape, f"记忆叶初态形状不同: {k}")
        changed += int(not exact(init_arr.astype(saved[k].dtype), saved[k]))
        changed_bf16 += int(not exact(init_arr.astype(ml_dtypes.bfloat16), saved[k].astype(ml_dtypes.bfloat16)))
    print(f"CKPT_VS_INIT mem_leaves={len(mem_keys)} changed_saved_dtype={changed} changed_bf16={changed_bf16} "
          f"saved_dtypes={sorted({str(saved[k].dtype) for k in mem_keys})}", flush=True)
    require(changed == len(mem_keys), f"记忆叶未全部区别于初态: {changed}/{len(mem_keys)}")
    del initial
    gc.collect()

    ds = dataset(a)
    try:
        transformed = transform_dataset(ds, tc.data.create(tc.assets_dirs, tc.model))
        batch = _collate_fn([transformed[0]])
        obs = HistAugObservation.from_dict(jax.tree.map(jnp.asarray, batch))
    finally:
        ds.close()
    reference = tc.model.load(traverse_util.unflatten_dict(
        {tuple(k.split("/")): jnp.asarray(v.astype(ml_dtypes.bfloat16)) for k, v in saved.items()}), remove_extra_params=False)
    del saved
    gc.collect()
    noise = jnp.asarray(np.random.default_rng(42).normal(size=(1, 20, 32)).astype(np.float32))

    @nnx.jit
    def actions(model, observation):
        with at.disable_typechecking():
            return model.sample_actions(jax.random.key(9), observation, noise=noise, num_steps=10)
    with at.disable_typechecking():
        tokens, tmask, _, _ = policy._model.embed_memory(obs)
    mem_len = int(tokens.shape[1])
    require(mem_len == BUDGET and tmask.shape == (1, BUDGET), f"mem_len={mem_len} != {BUDGET}")
    left = np.asarray(actions(reference, obs))
    again = np.asarray(actions(reference, obs))
    right = np.asarray(actions(policy._model, obs))
    require(left.shape == right.shape == (1, 20, 32) and np.isfinite(left).all() and np.isfinite(right).all(), "动作 shape 或有限性错误")
    rms = float(np.sqrt(np.mean((left.astype(np.float64) - right) ** 2)))
    aa = float(np.sqrt(np.mean((left.astype(np.float64) - again) ** 2)))
    print(f"CKPT_ACTIONS rms_diff={rms} aa_rms={aa} threshold=6.8e-5 num_steps=10", flush=True)
    require(rms <= 6.8e-5, "固定 noise 动作偏差超过阈值")
    print(f"CKPT_LOAD_CTX2048=PASS leaves_equal=1 mem_leaves_changed={changed} actions_match=1 mem_len={mem_len}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--store", default=str(DS / "framesamp-8x8"))
    common.add_argument("--source", default=str(DS / "source"))
    common.add_argument("--manifest", default=str(DS / "meta/episode_manifest.json"))
    common.add_argument("--norm-stats", default=str(ASSETS / "robomme/norm_stats.json"))
    sub.add_parser("short-hist", parents=[common], help="G2：统计 step<31 的执行样本数（CPU，只读清单）")
    sub.add_parser("online", parents=[common], help="G3：context 2048 在线装配 5 步（CPU）")
    s = sub.add_parser("smoke-summary", help="G4：20 步短测日志与 NVML csv 汇总")
    s.add_argument("--log", required=True)
    s.add_argument("--nvml", required=True)
    s.add_argument("--gpus", required=True, help="逗号分隔的 GPU 编号，如 0,1,2,3")
    c = sub.add_parser("ckpt", parents=[common], help="G5：最新 step checkpoint 经 create_trained_policy 加载核对")
    c.add_argument("--run", required=True)
    a = p.parse_args()
    {"short-hist": short_hist, "online": online, "smoke-summary": smoke_summary, "ckpt": ckpt}[a.command](a)


if __name__ == "__main__":
    main()
