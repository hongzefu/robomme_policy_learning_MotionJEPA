"""完整询问输入的无损存取与原采样路径离线重算。"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np


def query_noise(seed, episode_key, query_seq, count, horizon, action_dim):
    """每个预测独立派生；不含模型身份，不读取或推进策略随机状态。"""
    samples = []
    for index in range(count):
        identity = json.dumps([int(seed), str(episode_key), int(query_seq), index], separators=(",", ":"))
        entropy = np.frombuffer(hashlib.sha256(identity.encode()).digest(), dtype="<u4")
        samples.append(np.random.default_rng(np.random.SeedSequence(entropy)).standard_normal(
            (horizon, action_dim), dtype=np.float32))
    return np.stack(samples)


def pack_observation(observation):
    """按字段递归保存全部叶子和空字段；另存结构，避免 pickle。"""
    arrays = {}

    def visit(value, path):
        if value is None:
            return {"kind": "none"}
        if dataclasses.is_dataclass(value):
            return {"kind": "dict", "items": {field.name: visit(getattr(value, field.name), path + [field.name])
                                               for field in dataclasses.fields(value)}}
        if isinstance(value, dict):
            return {"kind": "dict", "items": {key: visit(item, path + [key]) for key, item in value.items()}}
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"观测叶子不允许 object dtype: {path}")
        if (array.dtype.kind in "fc" or str(array.dtype) == "bfloat16") and not np.isfinite(array).all():
            raise ValueError(f"观测叶子包含非有限值: {path}")
        key = "obs/" + "/".join(path)
        arrays[key] = array
        return {"kind": "array", "key": key, "dtype": str(array.dtype), "shape": list(array.shape)}

    schema = visit(observation, [])
    arrays["obs_schema"] = np.str_(json.dumps(schema, ensure_ascii=False))
    return arrays


def unpack_observation(record, factory=None):
    """恢复原字段、形状和 dtype；bfloat16 的 npy 描述符为 V2，用结构表恢复。"""
    schema = json.loads(str(np.asarray(record["obs_schema"]).item()))

    def visit(node):
        if node["kind"] == "none":
            return None
        if node["kind"] == "dict":
            return {key: visit(item) for key, item in node["items"].items()}
        value = np.asarray(record[node["key"]])
        if node["dtype"] == "bfloat16":
            import ml_dtypes
            value = value.view(ml_dtypes.bfloat16)
        if str(value.dtype) != node["dtype"] or list(value.shape) != node["shape"]:
            raise ValueError(f"观测叶子形状或 dtype 损坏: {node['key']}")
        return value

    values = visit(schema)
    if factory is None:
        return values
    # HistAugObservation 的扩展字段要求 JAX 数组；先转设备叶子，再构造，不绕过类型检查。
    import jax
    import jax.numpy as jnp
    return factory(**jax.tree.map(jnp.asarray, values))


def repeat_observation(observation, count):
    """批量复制全部非空叶子；不只复制 state 或 motion。"""
    import jax
    import jax.numpy as jnp

    def repeat(value):
        if value.shape[0] != 1:
            raise ValueError(f"询问输入必须 batch=1，得到 {value.shape}")
        return jnp.repeat(jnp.asarray(value), count, axis=0)

    return jax.tree.map(repeat, observation)


def sample_predictions(policy, observation, noise, *, sample_rng, batch_size, sample_kwargs=None):
    """固定批量分块调用原采样器，并逐预测调用原二维输出变换。"""
    import jax
    import jax.numpy as jnp

    noise = np.asarray(noise, dtype=np.float32)
    count = len(noise)
    if batch_size <= 0 or count % batch_size:
        raise ValueError("预测总数必须能整除固定采样批量")
    kwargs = dict(sample_kwargs if sample_kwargs is not None else policy._sample_kwargs)
    kwargs.pop("noise", None)
    normalized = []
    for start in range(0, count, batch_size):
        batch_obs = repeat_observation(observation, batch_size)
        values = policy._sample_actions(sample_rng, batch_obs, noise=jnp.asarray(noise[start:start + batch_size]), **kwargs)
        normalized.append(np.asarray(jax.block_until_ready(values), dtype=np.float32))
    normalized = np.concatenate(normalized)
    if normalized.shape != noise.shape:
        raise ValueError(f"采样输出形状 {normalized.shape} 与噪声 {noise.shape} 不符")
    actions = []
    state = np.asarray(observation.state[0])
    for value in normalized:
        transformed = policy._output_transform({"state": state.copy(), "actions": value.copy()})
        actions.append(np.asarray(transformed["actions"], dtype=np.float32))
    actions = np.stack(actions)
    if actions.shape != (count, noise.shape[1], 8):
        raise ValueError(f"输出变换后动作形状不符: {actions.shape}")
    if not np.isfinite(normalized).all() or not np.isfinite(actions).all():
        raise ValueError("询问预测包含非有限值")
    return normalized, actions


def replay_query(policy, path, *, adapter=None):
    """原路径必须按存盘批量复现；注意力展开差异只观察，不放宽原路径阈值。"""
    import jax
    import jax.numpy as jnp
    from mme_vla_suite.models.integration.history_observation import HistAugObservation

    with np.load(path, allow_pickle=False) as archive:
        record = {key: archive[key] for key in archive.files}
    observation = unpack_observation(record, HistAugObservation)
    observation = jax.tree.map(jnp.asarray, observation)
    rng = jnp.asarray(record["sample_rng"], dtype=jnp.uint32)
    kwargs = json.loads(str(record["sample_kwargs"]))
    normalized, actions = sample_predictions(policy, observation, record["noise"], sample_rng=rng,
        batch_size=int(record["sample_batch_size"]), sample_kwargs=kwargs)
    difference = float(np.maximum(np.max(np.abs(normalized - record["actions_normalized"])),
                                  np.max(np.abs(actions - record["actions"]))))
    result = {"query": str(path), "same_path_maxabs": difference, "tol": 1e-6,
              "predictions": len(actions), "attention": "not_run", "attn_path_maxabs": None}
    if adapter is not None:
        maximum = 0.0
        for index, noise in enumerate(record["noise"]):
            output = adapter.run(rng, observation, noise=jnp.asarray(noise[None]),
                                 num_steps=int(kwargs.get("num_steps", 10)))
            value = np.asarray(jax.block_until_ready(output[0]))[0]
            maximum = max(maximum, float(np.max(np.abs(value - record["actions_normalized"][index]))))
        result.update(attention="ok", attn_path_maxabs=maximum)
    result["status"] = "PASS" if np.isfinite(difference) and difference <= 1e-6 else "FAIL"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, nargs="+", required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    # 复用已用于适配器自检的检查点恢复函数；离线没有编码请求，不启动 motion sidecar。
    from scripts.interpret.attn_adapter import _ZeroMotion, _MV_DIR
    import sys
    sys.path.insert(0, str(_MV_DIR))
    from mv_model_adapter import build_policy
    from mme_vla_suite.training import config

    policy = build_policy(config.get_config(args.config), args.ckpt.resolve(), seed=args.seed,
                          motion_factory=lambda **kwargs: _ZeroMotion())
    adapter = None
    if args.attention:
        from scripts.interpret.attn_adapter import AttnAdapter
        adapter = AttnAdapter(policy)
    results = [replay_query(policy, path, adapter=adapter) for path in args.query]
    report = {"checkpoint": str(args.ckpt.resolve()), "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in results:
        print(f"OFFLINE_REPLAY={item['status']} points=1 same_path_maxabs={item['same_path_maxabs']:.9g} "
              f"tol=1e-6 attn_path_maxabs={item['attn_path_maxabs']} attention={item['attention']}", flush=True)
    return 0 if all(item["status"] == "PASS" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
