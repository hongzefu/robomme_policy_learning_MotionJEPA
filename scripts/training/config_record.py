"""完整配置的稳定记录；未知对象拒绝序列化，禁止用地址或字段摘录充数。"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
from pathlib import Path
import re


def typename(value):
    cls = type(value)
    return cls.__module__ + "." + cls.__qualname__


def file_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def encode(value):
    """递归保留全部字段、容器类型、dataclass类型和数组原始字节摘要。"""
    import numpy as np
    from omegaconf import OmegaConf

    if isinstance(value, enum.Enum):
        return {"type": typename(value), "value": encode(value.value)}
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("配置不能包含非有限数值")
        return value
    if isinstance(value, Path):
        return {"type": "pathlib.Path", "value": str(value)}
    if isinstance(value, re.Pattern):
        return {"type": "re.Pattern", "pattern": value.pattern, "flags": value.flags}
    if OmegaConf.is_config(value):
        return encode(OmegaConf.to_container(value, resolve=True))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"type": typename(value), "fields": {
            field.name: encode(getattr(value, field.name)) for field in dataclasses.fields(value)}}
    if isinstance(value, dict):
        if not all(type(key) is str for key in value):
            raise TypeError("配置字典只能使用字符串键")
        return {key: encode(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return {"type": typename(value), "items": [encode(item) for item in value]}
    if isinstance(value, np.ndarray):
        return {"type": "numpy.ndarray", "dtype": str(value.dtype), "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return {"type": typename(value), "value": encode(value.item())}
    if typename(value) in ("openpi.models.tokenizer.PaligemmaTokenizer", "mme_vla_suite.training.config.PaligemmaTokenizer"):
        if set(vars(value)) != {"_max_len", "_tokenizer"}:
            raise TypeError("tokenizer新增或缺失字段，须更新完整配置记录协议")
        return {"type": typename(value), "fields": {"_max_len": value._max_len,
            "_tokenizer": {"type": typename(value._tokenizer),
                "model_sha256": hashlib.sha256(value._tokenizer.serialized_model_proto()).hexdigest()}}}
    raise TypeError(f"未登记的配置对象类型: {typename(value)}")


def complete_record(config):
    from mme_vla_suite.models.config.utils import get_history_config
    from omegaconf import OmegaConf

    history = get_history_config(config.model.history_config) if config.model.history_config else None
    resolved_model = dataclasses.replace(config.model, history_config=history)
    data = config.data.create(config.assets_dirs, resolved_model)
    assets = {}
    norm = Path(config.data.assets.assets_dir or config.assets_dirs) / data.asset_id / "norm_stats.json"
    assets["norm_stats"] = file_record(norm)
    params = Path(config.weight_loader.params_path)
    # 大权重由资产锁及启动前内容校验约束；本记录额外绑定原生恢复索引，避免把路径当摘要。
    for name in ("_METADATA", "manifest.ocdbt", "commit_success.txt"):
        if (params / name).is_file():
            assets["initialization/" + name] = file_record(params / name)
    import os
    model_root = Path(os.environ["OPENPI_DATA_HOME"])
    assets["tokenizer"] = file_record(model_root / "big_vision/paligemma_tokenizer.model")
    lock = Path(__file__).resolve().parents[1] / "assets/ASSETS_LOCK.json"
    if lock.is_file():
        assets["assets_lock"] = file_record(lock)
    return {"config_record_version": 2, "train_config": encode(config),
            "resolved_data": encode(data), "assets": assets,
            "history_values": OmegaConf.to_container(history, resolve=True) if history is not None else None,
            "derived": {"checkpoint_dir": str(config.checkpoint_dir), "assets_dirs": str(config.assets_dirs)}}


def compare_records(reference, candidate, allowed=()):
    """只放行精确叶路径的值差异；新增、缺失字段和类型变化始终拒绝。"""
    if reference.get("config_record_version") != 2 or candidate.get("config_record_version") != 2:
        raise ValueError("需要完整配置记录版本2")
    allowed = set(allowed)
    differences = []

    def visit(left, right, path):
        if type(left) is not type(right):
            raise ValueError(f"配置类型改变: {'.'.join(path)}")
        if isinstance(left, dict):
            if left.keys() != right.keys():
                raise ValueError(f"配置字段新增或缺失: {'.'.join(path)}")
            for key in left:
                visit(left[key], right[key], (*path, key))
        elif isinstance(left, list):
            if len(left) != len(right):
                raise ValueError(f"配置序列长度改变: {'.'.join(path)}")
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                visit(a, b, (*path, str(index)))
        elif left != right:
            field = ".".join(path)
            if field not in allowed:
                raise ValueError(f"配置出现未授权差异: {field}: {left!r} -> {right!r}")
            differences.append(field)

    visit(reference, candidate, ())
    return differences


IDENTITY_FIELDS = ("train_config.fields.exp_name", "derived.checkpoint_dir")
PERF_FIELDS = (*IDENTITY_FIELDS, "train_config.fields.num_train_steps", "train_config.fields.wandb_enabled")
BUDGET_FIELDS = (*PERF_FIELDS, "train_config.fields.model.fields.history_config", "history_values.budget",
                 "train_config.fields.log_interval")


def main():
    """可在固定旧提交的PYTHONPATH上执行同一序列化器，不修改旧档案。"""
    from mme_vla_suite.training.config import cli
    print("COMPLETE_CONFIG_JSON=" + json.dumps(complete_record(cli()), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
