#!/usr/bin/env python3
"""在两侧独立环境取证真实输入，再按原始 dtype、shape 与字节严格判定。

collect 不导入本工具所在仓库的辅助模块；项目模块只来自 --expect-root。
索引探针与内容取证都委托目标侧创建的真实 TorchDataLoader / sampler。
探针只把该实例的数据集改成等长索引返回器；内容侧只筛选原 BatchSampler
产出的批次，并核对两遍的完整 sampler 序列、generator 状态和 drop_last。
筛选不改变单个样本及 collate；原始短历史 dtype 不转换、不补缺失 motion 键。
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time

SCHEMA = 1
BATCH_SIZE = 64
WORKERS = 4
SEED = 42
PREFIX_BATCHES = 100
HISTORY = "perceptual-framesamp-modul.yaml"
PROJECTS = ("mme_vla_suite", "openpi", "openpi_client")
LEGACY_NONE_KEYS = ("recur_image_emb", "recur_pos_emb", "recur_state_emb", "recur_mask")
REQUIRED_RAW = {"image", "wrist_image", "state", "actions", "prompt", "static_image_emb",
                "static_pos_emb", "static_state_emb", "static_mask"}
REQUIRED_TRANSFORMED = {"image", "image_mask", "state", "actions", "tokenized_prompt", "tokenized_prompt_mask",
                        "static_image_emb", "static_pos_emb", "static_state_emb", "static_mask"}
PACKAGES = ("numpy", "torch", "jax", "jaxlib", "flax", "ml-dtypes", "optax", "orbax-checkpoint",
            "sentencepiece", "omegaconf", "tyro")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha_file(path)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"JSON 键重复: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"JSON 含非有限常量: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs,
                      parse_constant=_reject_constant)


def read_jsonl(path):
    return [json.loads(line, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
            for line in Path(path).read_text(encoding="utf-8").splitlines()]


def manifest_content_sha(manifest):
    # 上游没有 datastore.manifest；此处独立校验格式，不能 import 当前仓库来污染 A。
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    encoded = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_manifest(manifest):
    require(manifest.get("sha256") == manifest_content_sha(manifest), "episode manifest 摘要不符")
    require(type(manifest.get("version")) is int and manifest["version"] == 1, "manifest version 不符")
    seen = set()
    offset = total = 0
    episodes = manifest["episodes"]
    require(isinstance(episodes, list) and episodes, "manifest episodes 为空或类型不符")
    for index, episode in enumerate(episodes):
        identity = (episode["h5_file"], episode["raw_ep_idx"])
        require(identity not in seen, f"episode 物理身份重复: {identity}")
        seen.add(identity)
        for field in ("raw_ep_idx", "global_episode_idx", "num_timesteps", "exec_start_idx", "exec_samples",
                      "exec_sample_offset", "total_sample_offset"):
            require(type(episode[field]) is int and episode[field] >= 0, f"manifest 整数字段不符: {field}")
        require(episode["num_timesteps"] > episode["exec_start_idx"], f"episode 无执行样本: {identity}")
        require(episode["exec_samples"] == episode["num_timesteps"] - episode["exec_start_idx"], "执行区间不符")
        require((episode["global_episode_idx"], episode["exec_sample_offset"], episode["total_sample_offset"])
                == (index, offset, total), "manifest 编号或偏移不连续")
        offset += episode["exec_samples"]
        total += episode["num_timesteps"]
    expected = {"episodes": len(episodes), "exec_samples": offset, "timesteps": total}
    require(manifest["totals"] == expected, "manifest 总数不符")
    return expected


def sample_plan(manifest):
    """每集的执行首尾、交界与历史 31/32/33 帧边界；不把 demo 帧冒充执行样本。"""
    validate_manifest(manifest)
    rows = []
    for episode in manifest["episodes"]:
        start, end = episode["exec_start_idx"], episode["num_timesteps"]
        steps = sorted(step for step in {start, start + 1, end - 1, 30, 31, 32} if start <= step < end)
        rows.extend({"h5_file": episode["h5_file"], "raw_ep_idx": episode["raw_ep_idx"],
                     "epis_idx": episode["global_episode_idx"], "step_idx": step,
                     "index": episode["exec_sample_offset"] + step - start} for step in steps)
    return rows


def full_identity(source, manifest):
    """只解码每个 pkl 的身份，不重读每个样本的整段历史特征。"""
    import numpy as np

    totals = validate_manifest(manifest)
    data = Path(source) / "data"
    names = {entry.name for entry in data.iterdir() if entry.is_file()}
    require(names == {f"{i}.pkl" for i in range(totals["exec_samples"])}, "source pkl 文件集合不完整")
    digest = hashlib.sha256()
    count = 0
    for episode in manifest["episodes"]:
        for step in range(episode["exec_start_idx"], episode["num_timesteps"]):
            index = episode["exec_sample_offset"] + step - episode["exec_start_idx"]
            with (data / f"{index}.pkl").open("rb") as stream:
                value = pickle.load(stream)
                require(not stream.read(1), f"pkl 尾部多余数据: {index}")
            for key, expected in (("epis_idx", episode["global_episode_idx"]), ("step_idx", step),
                                  ("exec_start_idx", episode["exec_start_idx"])):
                array = value[key]
                require(isinstance(array, np.ndarray) and array.dtype == np.dtype("int32") and array.shape == (1,)
                        and int(array[0]) == expected, f"全量 pkl 身份不符: index={index} key={key}")
            identity = [index, episode["h5_file"], episode["raw_ep_idx"], step]
            digest.update(json.dumps(identity, separators=(",", ":")).encode() + b"\n")
            count += 1
            if count % 10000 == 0:
                print(f"INPUT_IDENTITY_PROGRESS samples={count}/{totals['exec_samples']}", flush=True)
    require(count == totals["exec_samples"], "全量身份扫描不足")
    return {**totals, "identity_sha256": digest.hexdigest(), "manifest_sha256": manifest["sha256"]}


def describe_tree(value):
    """记录结构和 raw 位模式；有限值诊断使用副本视图，不修改任何训练输入。"""
    import numpy as np

    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), "输入字典必须使用字符串键")
        return {"kind": "dict", "items": {key: describe_tree(value[key]) for key in sorted(value)}}
    if value is None:
        return {"kind": "none"}
    if isinstance(value, tuple | list):
        return {"kind": type(value).__name__, "items": [describe_tree(item) for item in value]}
    if isinstance(value, str):
        return {"kind": "string", "sha256": hashlib.sha256(value.encode()).hexdigest(), "bytes": len(value.encode())}
    array = np.asarray(value)
    require(not array.dtype.hasobject, "不接受不能定义原始位模式的 object 数组")
    finite = True
    if array.dtype.kind in "fc" or str(array.dtype) == "bfloat16":
        finite = bool(np.isfinite(array.astype(np.complex128 if array.dtype.kind == "c" else np.float64)).all())
    blob = array.tobytes(order="C")
    return {"kind": "array", "dtype": str(array.dtype), "dtype_str": array.dtype.str, "shape": list(array.shape),
            "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(), "finite": finite}


def comparison_tree(value):
    require(isinstance(value, dict), "样本或 batch 顶层必须为字典")
    for key in LEGACY_NONE_KEYS:
        require(key not in value or value[key] is None, f"废弃 recurrent 字段并未关闭: {key}")
    return describe_tree({key: item for key, item in value.items() if key not in LEGACY_NONE_KEYS})


def require_finite_tree(tree, label):
    require(isinstance(tree, dict) and tree.get("kind") in ("array", "dict", "none", "string", "list", "tuple"),
            f"输入摘要结构无效: {label}")
    if tree["kind"] == "array":
        require(tree.get("finite") is True, f"输入存在非有限值: {label}")
    elif tree["kind"] == "dict":
        for key, child in tree["items"].items():
            require_finite_tree(child, f"{label}.{key}")
    elif tree["kind"] in ("list", "tuple"):
        for index, child in enumerate(tree["items"]):
            require_finite_tree(child, f"{label}[{index}]")


def validate_projection(all_fields, compared, required, label):
    require(all_fields.get("kind") == "dict" and set(all_fields["items"]) >= required, f"模型输入键缺失: {label}")
    for key in LEGACY_NONE_KEYS:
        require(key not in all_fields["items"] or all_fields["items"][key] == {"kind": "none"},
                f"recurrent 废弃键包含实际内容: {label}.{key}")
    expected = {"kind": "dict", "items": {key: value for key, value in all_fields["items"].items()
                                         if key not in LEGACY_NONE_KEYS}}
    require(compared == expected, f"比较投影丢弃或改变了模型输入: {label}")
    require_finite_tree(compared, label)


def first_difference(a, b, label):
    if type(a) is not type(b):
        return f"{label}: 类型不同"
    if isinstance(a, dict):
        if set(a) != set(b):
            return f"{label}: 字段集合不同，A独有={sorted(set(a)-set(b))}，B独有={sorted(set(b)-set(a))}"
        for key in sorted(a):
            difference = first_difference(a[key], b[key], f"{label}.{key}")
            if difference:
                return difference
    elif isinstance(a, list):
        if len(a) != len(b):
            return f"{label}: 长度不同"
        for index, (left, right) in enumerate(zip(a, b, strict=True)):
            difference = first_difference(left, right, f"{label}[{index}]")
            if difference:
                return difference
    elif a != b:
        return f"{label}: A={a!r}, B={b!r}"
    return ""


def module_records(root, forbidden=()):
    root = Path(root).resolve()
    forbidden = [Path(path).resolve() for path in forbidden]
    records = {}
    for name, module in sorted(sys.modules.items()):
        if name.split(".")[0] not in PROJECTS or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(root) and not any(path.is_relative_to(item) for item in forbidden),
                f"项目模块来源污染: {name} -> {path}")
        records[name] = file_record(path)
    return records


def git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def snapshot(root, head, forbidden):
    root = Path(root).resolve()
    require(git(root, "rev-parse", "--show-toplevel") == str(root), "Git 根不符")
    actual = git(root, "rev-parse", "HEAD")
    status = git(root, "status", "--porcelain")
    require(actual == head and not status, "输入取证必须从指定 clean HEAD 启动并保持 clean")
    return {"root": str(root), "head": actual, "porcelain": status, "modules": module_records(root, forbidden)}


class DatasetProbe:
    """委托实际 dataset；索引遍单独声明不解码，内容遍保持返回值原样。"""

    def __init__(self, dataset, *, index_only, worker_dir=None, root=None, forbidden=()):
        self.dataset = dataset
        self.index_only = index_only
        self.worker_dir = worker_dir
        self.root = root
        self.forbidden = forbidden
        self.recorded_pid = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        result = int(index) if self.index_only else self.dataset[index]
        if self.worker_dir and self.recorded_pid != os.getpid():
            import torch

            worker = torch.utils.data.get_worker_info()
            identifier = worker.id if worker else -1
            require(Path(sys.prefix).resolve() == (Path(self.root) / ".venv").resolve(), "worker 解释器不属于目标侧")
            record = {"worker": identifier, "pid": os.getpid(), "prefix": sys.prefix,
                      "modules": module_records(self.root, self.forbidden)}
            write_json(Path(self.worker_dir) / f"worker-{identifier}.json", record)
            self.recorded_pid = os.getpid()
        return result


def index_collate(items):
    return [int(item) for item in items]


class SamplerRecorder:
    """原 sampler 的透明迭代包装，连 drop_last 丢弃的尾样本也全部记录。"""

    def __init__(self, sampler):
        self.sampler = sampler
        self.epochs = []

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        indices = []
        self.epochs.append(indices)
        for index in self.sampler:
            indices.append(int(index))
            yield index


class SelectedBatches:
    """完整消费原 BatchSampler，仅转发已登记位置；不另造乱序算法。"""

    def __init__(self, sampler, selections):
        self.sampler = sampler
        self.selections = selections
        self.epoch = 0
        self.forwarded = []

    def __len__(self):
        return len(self.selections[min(self.epoch, len(self.selections) - 1)])

    def __iter__(self):
        require(self.epoch < len(self.selections), "批次取证意外进入第三个 epoch")
        selected = self.selections[self.epoch]
        self.epoch += 1
        forwarded = []
        self.forwarded.append(forwarded)
        for position, batch in enumerate(self.sampler):
            if position in selected:
                forwarded.append((position, [int(index) for index in batch]))
                yield batch


def batch_selections(size, batch_size=None, prefix=None):
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    prefix = PREFIX_BATCHES if prefix is None else prefix
    batches = size // batch_size
    require(batches >= max(prefix, 2), "数据不足以覆盖前 100 批和 epoch 交界")
    return [sorted(set(range(prefix)) | {batches - 2, batches - 1}), [0, 1]]


def generator_sha(generator):
    return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()


def capture_epochs(loader, *, index_only, selections=None, callback=None, worker_dir=None, root=None, forbidden=()):
    """两遍均让 PyTorch 真实 iterator 处理 base_seed、持久 worker 和 epoch 重建。"""
    import torch

    size, batch_size = len(loader.dataset), loader.batch_size
    require(loader.drop_last and loader.num_workers >= 0, "loader 必须保留 drop_last=True")
    require(isinstance(loader.sampler, torch.utils.data.RandomSampler), "正式 loader 未使用 RandomSampler")
    require(not loader.sampler.replacement and loader.sampler.num_samples == size, "sampler 不是无放回完整 epoch")
    require(loader.sampler.generator is loader.generator, "sampler 与 loader 的 generator 已不共享")
    require(getattr(loader, "in_order", True), "loader 必须按批次顺序交付")
    recorder = SamplerRecorder(loader.batch_sampler.sampler)
    loader.batch_sampler.sampler = recorder
    proxy = DatasetProbe(loader.dataset, index_only=index_only, worker_dir=worker_dir, root=root, forbidden=forbidden)
    # DataLoader 禁止普通 setattr 改 dataset；此处仅改取证实例并在记录中显式披露。
    object.__setattr__(loader, "dataset", proxy)
    selected_sampler = None
    if index_only:
        loader.collate_fn = index_collate
    else:
        require(selections is not None and callback is not None, "内容取证缺少批次计划或摘要回调")
        selected_sampler = SelectedBatches(loader.batch_sampler, selections)
        object.__setattr__(loader, "batch_sampler", selected_sampler)
    records = []
    iterator = None
    try:
        for epoch in range(2):
            before = generator_sha(loader.generator)
            iterator = iter(loader)
            base_seed = int(iterator._base_seed)  # noqa: SLF001 — 记录真实 PyTorch 种子，不能自行模拟消耗 RNG。
            observed = []
            received = 0
            for batch in iterator:
                if index_only:
                    observed.extend(batch)
                else:
                    position = selections[epoch][received]
                    callback(epoch, position, batch)
                received += 1
            require(len(recorder.epochs) == epoch + 1, "sampler epoch 数量不符")
            indices = recorder.epochs[epoch]
            require(len(indices) == size and sorted(indices) == list(range(size)), "sampler 不是完整无重复排列")
            expected_batches = size // batch_size
            yielded = indices[:expected_batches * batch_size]
            if index_only:
                require(received == expected_batches and observed == yielded, "真实索引交付与 sampler 顺序不符")
            else:
                expected = [(position, indices[position * batch_size:(position + 1) * batch_size])
                            for position in selections[epoch]]
                require(received == len(expected) and selected_sampler.forwarded[epoch] == expected,
                        "筛选包装交付的批次与原 sampler 不符")
            records.append({"epoch": epoch, "indices": indices, "drop_last_tail": indices[len(yielded):],
                            "logical_batches": expected_batches, "generator_before": before,
                            "generator_after": generator_sha(loader.generator), "base_seed": base_seed})
    finally:
        if iterator is not None and loader.num_workers:
            iterator._shutdown_workers()  # noqa: SLF001 — 精确结束本次取证创建的持久 worker。
    return records


def validate_sampler_records(records, size, batch_size):
    require(len(records) == 2, "sampler 必须有两个完整 epoch")
    for epoch, record in enumerate(records):
        indices = record["indices"]
        require(record["epoch"] == epoch and len(indices) == size and sorted(indices) == list(range(size)),
                "sampler 序列重复、缺失或乱序 epoch")
        batches = size // batch_size
        require(record["logical_batches"] == batches and record["drop_last_tail"] == indices[batches * batch_size:],
                "sampler drop_last 或换轮边界不符")
        for key in ("generator_before", "generator_after"):
            require(isinstance(record[key], str) and len(record[key]) == 64, "缺 generator 状态摘要")
        if epoch:
            require(records[epoch - 1]["generator_after"] == record["generator_before"], "epoch 间 generator 状态断裂")


def _implementation(value):
    return f"{value.__module__}.{value.__qualname__}"


def load_side(args):
    """只依赖解释器自己的 editable 安装，不向 sys.path 注入任何项目 src。"""
    require(not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME"), "须清除 PYTHONPATH/PYTHONHOME")
    root = Path(args.expect_root).resolve(strict=True)
    require(Path.cwd().resolve() == root, "collect 必须从目标侧根目录运行")
    require(Path(sys.prefix).resolve() == (root / ".venv").resolve(), "collect 未使用该侧独立 uv 环境")
    require(os.environ.get("JAX_PLATFORMS") == "cpu", "非训练输入取证必须显式 JAX_PLATFORMS=cpu")
    for name in PROJECTS:
        importlib.import_module(name)
    module_records(root, args.forbid_root)
    configs = importlib.import_module("mme_vla_suite.training.config")
    dataloader = importlib.import_module("mme_vla_suite.training.dataloader")
    shared_loader = importlib.import_module("openpi.training.data_loader")
    get_history = importlib.import_module("mme_vla_suite.models.config.utils").get_history_config
    import jax

    require(not jax.config.x64_enabled, "本轮输入取证要求 JAX x64 关闭")
    config = configs.get_config("mme_vla_suite")
    require((config.batch_size, config.num_workers, config.seed, config.fsdp_devices, config.model.action_horizon)
            == (BATCH_SIZE, WORKERS, SEED, 4, 20), "原版训练配置的输入参数已改变")
    assets = dataclasses.replace(config.data.assets, assets_dir=str(args.assets_dir), asset_id="robomme")
    model = dataclasses.replace(config.model, use_history=True, history_config=HISTORY)
    config = dataclasses.replace(config, model=model, dataset_path=str(args.dataset_path),
                                 data=dataclasses.replace(config.data, assets=assets))
    history = get_history(HISTORY)
    require((history.budget, history.token_per_image, history.num_views, history.representation_type,
             history.integration_type, history.perceptual_memory.type)
            == (512, 16, 1, "perceptual", "modulation", "frame_sampling"), "history 不是原版 modul 4×4")
    motion = getattr(history, "motion", None)
    require(motion is None or not motion.get("enabled", False), "本轮 motion 必须关闭")
    if args.side == "current":
        os.environ["MMEVLA_FRAMESAMP_SOURCE"] = str(args.source)
        os.environ["MMEVLA_FRAMESAMP_MANIFEST"] = str(args.manifest)
    else:
        require(args.dataset_path == args.source, "上游必须消费 source")
        os.environ.pop("MMEVLA_FRAMESAMP_SOURCE", None)
        os.environ.pop("MMEVLA_FRAMESAMP_MANIFEST", None)
    data_config = config.data.create(config.assets_dirs, config.model)
    require(data_config.norm_stats is not None, "真实 data_config 没有加载 norm_stats")

    def factory():
        return dataloader.create_data_loader(str(args.dataset_path), data_config, history_config=history,
                                            action_horizon=config.model.action_horizon, batch_size=BATCH_SIZE,
                                            num_workers=WORKERS, seed=SEED, shuffle=True)

    return factory, getattr(shared_loader, "_from_shared_torch", lambda batch: batch), config


def common_files(args):
    root, source = Path(args.expect_root), Path(args.source)
    model_root = Path(os.environ["OPENPI_DATA_HOME"]).resolve(strict=True)
    return {
        "manifest": file_record(args.manifest), "input_manifest": file_record(args.input_manifest),
        "source_stats": file_record(source / "meta/stats.json"),
        "source_provenance": file_record(source / "meta/provenance.json"),
        "norm_stats": file_record(Path(args.assets_dir) / "robomme/norm_stats.json"),
        "tokenizer": file_record(model_root / "big_vision/paligemma_tokenizer.model"),
        "history": file_record(root / "src/mme_vla_suite/models/config/robomme" / HISTORY),
    }


def write_record_manifest(out):
    records = {str(path.relative_to(out)): {"sha256": sha_file(path), "bytes": path.stat().st_size}
               for path in sorted(out.rglob("*")) if path.is_file() and path.name != "record_manifest.json"}
    write_json(out / "record_manifest.json", {"schema": SCHEMA, "files": records})


def verify_record_manifest(out):
    out = Path(out)
    manifest = read_json(out / "record_manifest.json")
    require(manifest["schema"] == SCHEMA, "记录清单 schema 不符")
    files = manifest["files"]
    current = {str(path.relative_to(out)) for path in out.rglob("*") if path.is_file() and path.name != "record_manifest.json"}
    require(set(files) == current, "取证文件缺失或多出未登记文件")
    for name, expected in files.items():
        path = (out / name).resolve(strict=True)
        require(path.is_relative_to(out.resolve()), "取证文件穿出记录根")
        require(file_record(path)["sha256"] == expected["sha256"] and path.stat().st_size == expected["bytes"],
                f"取证文件已变化: {name}")


def collect(args):
    require(not Path(args.out).is_symlink(), "拒绝向已有输出链接写入")
    require(all(Path(path).is_absolute() for path in args.forbid_root), "--forbid-root 必须为绝对路径")
    args.forbid_root = [str(Path(path).resolve()) for path in args.forbid_root]
    for key in ("expect_root", "dataset_path", "source", "manifest", "input_manifest", "assets_dir", "out"):
        value = Path(getattr(args, key))
        require(value.is_absolute(), f"--{key.replace('_', '-')} 必须为绝对路径")
        setattr(args, key, value.resolve())
    owner_store = Path(__file__).resolve().parents[3] / "v1-store"
    require(args.out.is_relative_to(owner_store) and not args.out.exists(), "取证输出必须为本轮 v1-store 下全新目录")
    snapshot(args.expect_root, args.expect_head, args.forbid_root)
    factory, unwrap, config = load_side(args)
    before = snapshot(args.expect_root, args.expect_head, args.forbid_root)
    files = common_files(args)
    manifest = read_json(args.manifest)
    totals = validate_manifest(manifest)
    selected = batch_selections(totals["exec_samples"])
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "provenance_start.json", before)
    write_json(args.out / "episode_manifest.json", manifest)
    try:
        identity = full_identity(args.source, manifest)
        write_json(args.out / "identity.json", identity)
        plan = sample_plan(manifest)
        write_json(args.out / "sample_plan.json", plan)
        probe_side = factory()
        probe_loader = probe_side._data_loader.torch_loader  # noqa: SLF001 — 取目标侧实际 loader，避免重写工厂。
        require(len(probe_loader.dataset) == totals["exec_samples"], "真实 Dataset 长度与 manifest 不符")
        transformed = probe_loader.dataset
        raw = transformed._dataset  # noqa: SLF001 — 同一真实 transforms 包装前后的取证点。
        expected_dataset = "RoboMMEDataset" if args.side == "upstream" else "FrameSampDataset"
        require(type(raw).__name__ == expected_dataset, "没有构造目标侧实际 Dataset")
        implementations = {"dataset": _implementation(type(raw)), "transformed": _implementation(type(transformed)),
                           "collate": _implementation(probe_loader.collate_fn), "sampler": _implementation(type(probe_loader.sampler))}
        with (args.out / "samples.jsonl").open("x", encoding="utf-8") as stream:
            for position, item in enumerate(plan, 1):
                raw_value, transformed_value = raw[item["index"]], transformed[item["index"]]
                require(set(raw_value) >= REQUIRED_RAW and set(transformed_value) >= REQUIRED_TRANSFORMED,
                        f"模型输入键缺失: {item}")
                require(int(raw_value["epis_idx"].item()) == item["epis_idx"]
                        and int(raw_value["step_idx"].item()) == item["step_idx"], "真实 Dataset 样本身份错位")
                row = {"identity": item, "raw_all": describe_tree(raw_value), "raw": comparison_tree(raw_value),
                       "transformed_all": describe_tree(transformed_value), "transformed": comparison_tree(transformed_value)}
                require_finite_tree(row["raw"], str(item))
                require_finite_tree(row["transformed"], str(item))
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                if position % 100 == 0:
                    print(f"INPUT_SAMPLE_PROGRESS samples={position}/{len(plan)}", flush=True)
        worker_dirs = {name: args.out / "workers" / name for name in ("index", "content")}
        for directory in worker_dirs.values():
            directory.mkdir(parents=True)
        probe = capture_epochs(probe_loader, index_only=True, worker_dir=worker_dirs["index"],
                               root=args.expect_root, forbidden=args.forbid_root)
        if hasattr(raw, "close"):
            raw.close()
        actual_side = factory()
        actual_loader = actual_side._data_loader.torch_loader  # noqa: SLF001 — 仍由目标侧工厂构造。
        with (args.out / "batches.jsonl").open("x", encoding="utf-8") as stream:
            def callback(epoch, position, batch):
                batch = unwrap(batch)
                require(set(batch) >= REQUIRED_TRANSFORMED, "真实 collate 模型输入键缺失")
                inputs = comparison_tree(batch)
                require_finite_tree(inputs, f"epoch={epoch}/batch={position}")
                indices = probe[epoch]["indices"][position * BATCH_SIZE:(position + 1) * BATCH_SIZE]
                row = {"epoch": epoch, "batch": position, "indices": indices,
                       "inputs_all": describe_tree(batch), "inputs": inputs}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                print(f"INPUT_BATCH epoch={epoch} batch={position} samples={len(indices)}", flush=True)
            actual = capture_epochs(actual_loader, index_only=False, selections=selected, callback=callback,
                                    worker_dir=worker_dirs["content"], root=args.expect_root, forbidden=args.forbid_root)
        require(actual == probe, "索引探针与真实内容 loader 的 sampler / generator 状态不同")
        validate_sampler_records(probe, totals["exec_samples"], BATCH_SIZE)
        write_json(args.out / "sampler.json", {"index_probe": probe, "real_content": actual, "selected": selected})
        after = snapshot(args.expect_root, args.expect_head, args.forbid_root)
        require(all(after["modules"].get(key) == value for key, value in before["modules"].items()), "运行期间模块字节变化")
        require(common_files(args) == files, "运行期间数据或资产摘要变化")
        write_json(args.out / "provenance_end.json", after)
        extra = {}
        if args.side == "current":
            packed = read_json(args.dataset_path / "meta/store_meta.json")
            require(packed["status"] == "verified" and packed["layout"] == "framesamp-4x4-v1"
                    and packed["manifest_sha256"] == manifest["sha256"], "packed 元数据未绑定已验证输入")
            extra["packed_meta"] = file_record(args.dataset_path / "meta/store_meta.json")
        metadata = {"schema": SCHEMA, "status": "COMPLETE", "side": args.side, "root": str(args.expect_root),
                    "head": args.expect_head, "forbid_roots": args.forbid_root, "prefix": sys.prefix,
                    "python": platform.python_version(), "python_build": sys.version,
                    "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
                    "harness": file_record(__file__), "common_files": files, "side_files": extra,
                    "dataset_path": str(args.dataset_path), "source": str(args.source), "implementations": implementations,
                    "contract": {"batch_size": BATCH_SIZE, "workers": WORKERS, "seed": SEED, "epochs": 2,
                                 "prefix_batches": PREFIX_BATCHES, "drop_last": True, "history": HISTORY,
                                 "action_horizon": config.model.action_horizon, "endpoint": "collate_before_jax",
                                 "excluded_legacy_none_keys": list(LEGACY_NONE_KEYS)},
                    "index_probe_overrides": ["等长 dataset 返回索引", "索引 collate"],
                    "content_overrides": ["委托原 BatchSampler 选择已登记批次", "worker 模块来源记录"],
                    "finished_unix": time.time()}
        write_json(args.out / "meta.json", metadata)
        write_record_manifest(args.out)
        validate_record(args.out, args.expect_head)
        print(f"INPUT_COLLECT=PASS side={args.side} samples={len(plan)} batches={sum(map(len, selected))}", flush=True)
    except BaseException as exc:
        write_json(args.out / "failure.json", {"error": str(exc), "type": type(exc).__name__})
        raise


def validate_record(out, expected_head):
    out = Path(out)
    verify_record_manifest(out)
    metadata = read_json(out / "meta.json")
    require(metadata["schema"] == SCHEMA and metadata["status"] == "COMPLETE", "取证未完整结束")
    require(metadata["head"] == expected_head and len(expected_head) == 40, "取证 HEAD 与完整提交锚不符")
    root = Path(metadata["root"]).resolve()
    require(Path(metadata["prefix"]).resolve() == (root / ".venv").resolve(), "记录未使用独立解释器")
    contract = metadata["contract"]
    require(set(metadata["common_files"]) == {"manifest", "input_manifest", "source_stats", "source_provenance",
                                              "norm_stats", "tokenizer", "history"}, "数据或资产来源记录不完整")
    expected_dataset = ("mme_vla_suite.training.dataset.RoboMMEDataset" if metadata["side"] == "upstream"
                        else "mme_vla_suite.training.framesamp_dataset.FrameSampDataset")
    require(metadata["implementations"]["dataset"] == expected_dataset, "未记录目标侧真实 Dataset")
    expected = {"batch_size": BATCH_SIZE, "workers": WORKERS, "seed": SEED, "epochs": 2,
                "prefix_batches": PREFIX_BATCHES, "drop_last": True, "history": HISTORY, "action_horizon": 20,
                "endpoint": "collate_before_jax", "excluded_legacy_none_keys": list(LEGACY_NONE_KEYS)}
    require(contract == expected, "输入取证契约被缩小或改变")
    for phase in ("start", "end"):
        provenance = read_json(out / f"provenance_{phase}.json")
        require(provenance["root"] == str(root) and provenance["head"] == expected_head
                and not provenance["porcelain"], "启动/结束版本或工作区状态无效")
        require(provenance["modules"], "缺项目模块来源证据")
        for name, record in provenance["modules"].items():
            path = Path(record["path"]).resolve()
            require(path.is_relative_to(root) and not any(path.is_relative_to(Path(item)) for item in metadata["forbid_roots"]),
                    f"来源污染: {name}")
    start = read_json(out / "provenance_start.json")["modules"]
    end = read_json(out / "provenance_end.json")["modules"]
    require(all(end.get(key) == value for key, value in start.items()), "取证期间模块变化")
    for phase in ("index", "content"):
        records = [read_json(path) for path in sorted((out / "workers" / phase).glob("*.json"))]
        require(sorted(record["worker"] for record in records) == list(range(WORKERS)), "worker 来源记录缺失或重复")
        for record in records:
            require(Path(record["prefix"]).resolve() == (root / ".venv").resolve() and record["modules"], "worker 环境无效")
            for module in record["modules"].values():
                path = Path(module["path"]).resolve()
                require(path.is_relative_to(root)
                        and not any(path.is_relative_to(Path(item)) for item in metadata["forbid_roots"]), "worker 来源污染")
    manifest = read_json(out / "episode_manifest.json")
    totals = validate_manifest(manifest)
    identity = read_json(out / "identity.json")
    require(all(identity[key] == value for key, value in totals.items())
            and identity["manifest_sha256"] == manifest["sha256"] and len(identity["identity_sha256"]) == 64,
            "全量身份记录不完整")
    plan = sample_plan(manifest)
    require(read_json(out / "sample_plan.json") == plan, "定点样本计划不完整")
    samples = read_jsonl(out / "samples.jsonl")
    require([row["identity"] for row in samples] == plan, "定点样本记录缺失、重复或乱序")
    for row in samples:
        validate_projection(row["raw_all"], row["raw"], REQUIRED_RAW, str(row["identity"]) + "/raw")
        validate_projection(row["transformed_all"], row["transformed"], REQUIRED_TRANSFORMED,
                            str(row["identity"]) + "/transformed")
    sampler = read_json(out / "sampler.json")
    require(sampler["index_probe"] == sampler["real_content"], "真实内容采样与索引探针不一致")
    validate_sampler_records(sampler["index_probe"], totals["exec_samples"], BATCH_SIZE)
    selections = batch_selections(totals["exec_samples"])
    require(sampler["selected"] == selections, "批次计划未覆盖规定窗口")
    batches = read_jsonl(out / "batches.jsonl")
    positions = [(epoch, position) for epoch, selected in enumerate(selections) for position in selected]
    require([(row["epoch"], row["batch"]) for row in batches] == positions, "真实批次记录缺失、重复或乱序")
    for row in batches:
        indices = sampler["index_probe"][row["epoch"]]["indices"]
        require(row["indices"] == indices[row["batch"] * BATCH_SIZE:(row["batch"] + 1) * BATCH_SIZE], "batch 身份不符")
        validate_projection(row["inputs_all"], row["inputs"], REQUIRED_TRANSFORMED,
                            f"epoch={row['epoch']}/batch={row['batch']}")
    return metadata, identity, sampler, samples, batches


def judge(args):
    a = validate_record(args.a_dir, args.expect_head_a)
    b = validate_record(args.b_dir, args.expect_head_b)
    ma, mb = a[0], b[0]
    require((ma["side"], mb["side"]) == ("upstream", "current") and ma["root"] != mb["root"], "A/B 必须是独立上游和当前根")
    # A worktree 可能嵌套于 B/v1-store；即使调用者漏传 forbid-root，也不能让 B 混入 A 的局部模块。
    upstream_root = Path(ma["root"]).resolve()
    b_out = Path(args.b_dir)
    provenance_paths = [b_out / "provenance_start.json", b_out / "provenance_end.json",
                        *sorted((b_out / "workers").glob("*/*.json"))]
    for path in provenance_paths:
        for name, record in read_json(path)["modules"].items():
            require(not Path(record["path"]).resolve().is_relative_to(upstream_root),
                    f"B 项目模块混入 A worktree: {path.name} {name} -> {record['path']}")
    for key in ("contract", "python", "python_build", "packages", "source"):
        require(ma[key] == mb[key], f"两侧环境或输入契约不同: {key}")
    require(ma["harness"]["sha256"] == mb["harness"]["sha256"], "两侧不是同一取证工具")
    for key, record in ma["common_files"].items():
        other = mb["common_files"].get(key)
        require(other is not None and record["sha256"] == other["sha256"] and record["bytes"] == other["bytes"],
                f"数据或资产指纹不同: {key}")
        if key != "history":
            require(record["path"] == other["path"], f"两侧未消费同一数据/资产: {key}")
    require(set(ma["common_files"]) == set(mb["common_files"]), "来源指纹键集合不同")
    require(a[1] == b[1], "全量样本身份不同")
    require(a[2] == b[2], "两个 epoch 的 sampler 或 generator 序列不同")
    for rows_a, rows_b, fields in ((a[3], b[3], ("raw", "transformed")), (a[4], b[4], ("inputs",))):
        require(len(rows_a) == len(rows_b), "输入记录数量不同")
        for left, right in zip(rows_a, rows_b, strict=True):
            label = str(left.get("identity", {"epoch": left.get("epoch"), "batch": left.get("batch")}))
            for field in fields:
                difference = first_difference(left[field], right[field], f"{label}/{field}")
                require(not difference, difference)
    print(f"INPUT_EQ=PASS episodes={a[1]['episodes']} samples={a[1]['exec_samples']} "
          f"boundary_samples={len(a[3])} batches={len(a[4])} epochs=2", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--side", choices=("upstream", "current"), required=True)
    for name in ("expect-root", "expect-head", "dataset-path", "source", "manifest", "input-manifest", "assets-dir", "out"):
        collect_parser.add_argument("--" + name, required=True)
    collect_parser.add_argument("--forbid-root", action="append", default=[])
    judge_parser = commands.add_parser("judge")
    for name in ("a-dir", "b-dir", "expect-head-a", "expect-head-b"):
        judge_parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        (collect if args.command == "collect" else judge)(args)
    except (OSError, ValueError, KeyError, TypeError, EOFError, RuntimeError, ImportError, pickle.UnpicklingError,
            subprocess.CalledProcessError) as exc:
        print(f"INPUT_{'COLLECT' if args.command == 'collect' else 'EQ'}=FAIL {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
