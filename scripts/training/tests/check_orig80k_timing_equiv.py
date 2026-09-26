"""同一只读取证层下，比较20步真实保存的测速包装off/on；不用于吞吐结论。"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "scripts/training/train.py"
SCHEMA = 1
STEPS = 20
OWNED = ("metadata.json", "fetch_inputs.jsonl", "model_inputs.jsonl", "full_state.json")
SPEED_FILES = ("speed_start.json", "speed_run.json", "step_timing.jsonl", "host_samples.jsonl", "disk_samples.jsonl")
TREE_FIELDS = {"params", "ema_params", "opt_state", "step"}
WRAPPER_EXITS = ("RUNNER_EXIT_CODE", "COMPLETION_EXIT_CODE", "COMPLETION_TEE_EXIT_CODE",
                 "WRAPPER_COMMAND_EXIT", "WRAPPER_TEE_EXIT", "WRAPPER_FOOTER_PRINTF_EXIT",
                 "WRAPPER_FOOTER_TEE_EXIT", "WRAPPER_EXIT_CODE")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "JSON包含重复键")
        result[key] = value
    return result


def parse(text):
    def invalid(value):
        raise ValueError("JSON包含非有限常量: " + value)
    return json.loads(text, object_pairs_hook=_unique, parse_constant=invalid)


def load(path):
    return parse(Path(path).read_text())


def rows(path):
    return [parse(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(canonical(value).decode() + "\n")


def record_file(path):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve() and path.is_file(), "记录文件必须是绝对实体路径")
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def training_imports():
    path = str(ROOT / "scripts/training")
    if path not in sys.path:
        sys.path.insert(0, path)
    import config_record
    import entry_equiv
    import final_record
    return config_record, final_record, entry_equiv


def tree_record(tree):
    """逐叶device_get并立即摘要，不聚集完整TrainState的主机副本，不改dtype。"""
    import jax
    import numpy as np
    _, final_record, _ = training_imports()
    flattened, definition = jax.tree_util.tree_flatten_with_path(tree, is_leaf=lambda value: value is None)
    leaves = {}
    for path, value in flattened:
        key = jax.tree_util.keystr(path)
        require(key not in leaves, "pytree叶路径重复")
        if value is None:
            leaves[key] = {"kind": "none"}
            continue
        array = np.asarray(jax.device_get(value))
        require(not array.dtype.hasobject, "不支持object叶")
        descriptor = final_record.array_record(array)
        require(descriptor["finite"], "输入或完整状态含非有限数值")
        leaves[key] = {"kind": "array", "value_type": type(value).__module__ + "." + type(value).__qualname__,
                       "array": descriptor}
    require(leaves and any(item["kind"] == "array" for item in leaves.values()), "数值树为空")
    result = {"treedef": repr(definition), "leaves": leaves}
    result["sha256"] = digest(result)
    return result


def validate_tree(tree):
    import ml_dtypes  # noqa: F401 -- 注册bfloat16等NumPy dtype，不作转换。
    import numpy as np
    require(set(tree) == {"treedef", "leaves", "sha256"} and tree["treedef"] and tree["leaves"], "树记录schema错误")
    require(tree["sha256"] == digest({key: tree[key] for key in ("treedef", "leaves")}), "树结构/叶摘要被改动")
    arrays = 0
    for item in tree["leaves"].values():
        if item == {"kind": "none"}:
            continue
        require(set(item) == {"kind", "value_type", "array"} and item["kind"] == "array" and item["value_type"],
                "树叶类型错误")
        array = item["array"]
        require(set(array) == {"dtype", "shape", "bytes", "sha256", "finite"}, "数组记录字段错误")
        dtype = np.dtype(array["dtype"])
        require(not dtype.hasobject and isinstance(array["shape"], list)
                and all(type(n) is int and n >= 0 for n in array["shape"]), "数组dtype/shape错误")
        require(type(array["bytes"]) is int and array["bytes"] == math.prod(array["shape"]) * dtype.itemsize
                and array["finite"] is True and re.fullmatch(r"[0-9a-f]{64}", array["sha256"]),
                "数组字节数、有限性或SHA错误")
        arrays += 1
    require(arrays > 0, "缺数值叶")


def full_state_record(state):
    """参数转纯字典便于与真实EMA恢复记录对齐；原pytree定义另行保留。"""
    import jax
    require(state.ema_params is not None, "本验证要求完整EMA")
    original = {name: getattr(state, name) for name in TREE_FIELDS}
    trees = {name: value.to_pure_dict() if name in {"params", "ema_params"}
             and hasattr(value, "to_pure_dict") else value for name, value in original.items()}
    groups = {name: tree_record(value) for name, value in trees.items()}
    ema_leaves, ema_leaf_paths = {}, {}
    flattened, _ = jax.tree_util.tree_flatten_with_path(trees["ema_params"])
    for path, _ in flattened:
        require(path and all(isinstance(item, jax.tree_util.DictKey) and isinstance(item.key, str) for item in path),
                "EMA纯字典包含不支持的路径")
        key = "/".join(item.key for item in path)
        require(key not in ema_leaves, "EMA恢复路径重复")
        ema_leaf_paths[key] = jax.tree_util.keystr(path)
        ema_leaves[key] = groups["ema_params"]["leaves"][ema_leaf_paths[key]]["array"]
    return {"groups": groups, "original_treedefs": {name: repr(jax.tree_util.tree_structure(value))
                                                    for name, value in original.items()},
            "ema_leaves": ema_leaves, "ema_leaf_paths": ema_leaf_paths,
            "state_step": int(jax.device_get(state.step))}


class Observer:
    """两侧完全相同的只读观察层；所有原对象、jit配置和保存返回值原样传递。"""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.fetched = []
        self.used = []
        self.iterations = 0
        self.jit_matches = 0
        self.completed_calls = 0
        self.save_calls = 0
        self.save_returned = False
        self.config = None
        self.identity = None
        self.fingerprint_start = None
        self.loader_contract = None
        self.fetch_stream = self.model_stream = None
        self.overhead = {"fetch_readback_hash_s": 0.0, "model_readback_hash_s": 0.0, "state_readback_hash_s": 0.0}

    def fetched_batch(self, batch):
        require(len(self.fetched) < STEPS + 1, "取批超过21次")
        started = time.perf_counter()
        tree = tree_record(batch)
        elapsed = time.perf_counter() - started
        row = {"fetch_index": len(self.fetched), "extra_unused": len(self.fetched) == STEPS,
               "tree": tree, "readback_hash_s": elapsed}
        self.fetched.append(tree)
        self.fetch_stream.write(canonical(row).decode() + "\n")
        self.fetch_stream.flush()
        self.overhead["fetch_readback_hash_s"] += elapsed

    def compile(self, original, function, *args, **kwargs):
        target = function.func if isinstance(function, functools.partial) else function
        candidate = isinstance(function, functools.partial) and getattr(target, "__name__", None) == "train_step"
        underlying = inspect.unwrap(target) if candidate else None
        code = getattr(underlying, "__code__", None)
        matched = candidate and code is not None and Path(code.co_filename).resolve() == ENTRY
        compiled = original(function, *args, **kwargs)
        if not matched:
            return compiled
        self.jit_matches += 1
        require(self.jit_matches == 1 and kwargs.get("donate_argnums") == (1,), "模型jit边界不符合固定入口")

        @functools.wraps(compiled)
        def observed(rng, state, batch):
            import jax
            step = len(self.used)
            require(step < STEPS and len(self.fetched) == step + 1, "模型调用和实际取批没有一一对应")
            started = time.perf_counter()
            tree = tree_record(batch)
            require(tree == self.fetched[step], "模型实参不同于对应DataLoader返回值")
            rng_record = {"dtype": str(rng.dtype), "implementation": str(jax.random.key_impl(rng)),
                          "key_data": tree_record(jax.random.key_data(rng))}
            elapsed = time.perf_counter() - started
            self.used.append(tree)
            self.model_stream.write(canonical({"step": step, "fetch_index": step, "tree": tree,
                                               "rng": rng_record, "readback_hash_s": elapsed}).decode() + "\n")
            self.model_stream.flush()
            self.overhead["model_readback_hash_s"] += elapsed
            result = compiled(rng, state, batch)
            self.completed_calls += 1
            return result
        return observed

    def save(self, original, manager, state, loader, step):
        require(step == 19 and self.save_calls == 0 and self.completed_calls == 20
                and len(self.fetched) == 21 and self.identity is not None, "真实保存没有发生在预期末步")
        self.save_calls += 1
        started = time.perf_counter()
        result = full_state_record(state)
        require(result["state_step"] == 20, "最终state.step不是20")
        self.overhead["state_readback_hash_s"] += time.perf_counter() - started
        write_json(self.directory / "full_state.json", {"schema": SCHEMA, "loop_step": step,
            "head": self.identity["head"], "run_name": self.identity["run_name"],
            "run_uuid": self.identity["run_uuid"], **result})
        returned = original(manager, state, loader, step)
        self.save_returned = True
        return returned

    @contextlib.contextmanager
    def installed(self):
        import jax

        from mme_vla_suite.training import dataloader
        from openpi.training import checkpoints
        _, final_record, entry_equiv = training_imports()
        original_iter = dataloader.DataLoaderImpl.__iter__
        original_jit, original_save, original_begin = jax.jit, checkpoints.save_state, final_record.begin
        require(Path(inspect.getsourcefile(original_save)).resolve() == Path(checkpoints.__file__).resolve()
                and original_save.__name__ == "save_state", "save_state已被其他观测层替换")
        self.original_save = {"file": str(Path(inspect.getsourcefile(original_save)).resolve()),
                              "module": original_save.__module__, "name": original_save.__name__}

        def observed_iter(loader):
            self.iterations += 1
            require(self.iterations == 1, "DataLoader被重复迭代")
            torch_loader = loader._data_loader.torch_loader  # noqa: SLF001 -- 只读核对真实最终loader，不改变交付对象。
            self.loader_contract = {"batch_size": torch_loader.batch_size, "workers": torch_loader.num_workers,
                                    "prefetch_factor": torch_loader.prefetch_factor,
                                    "persistent_workers": torch_loader.persistent_workers,
                                    "drop_last": torch_loader.drop_last}
            require(torch_loader.batch_size == 64 and torch_loader.num_workers == 4,
                    "实际最终loader不是b64/workers4")
            for batch in original_iter(loader):
                self.fetched_batch(batch)
                yield batch

        def observed_begin(config, wandb):
            require(self.config is None, "训练入口重复初始化")
            handle = original_begin(config, wandb)
            self.config, self.identity = config, handle[1]
            self.fingerprint_start = entry_equiv._runtime_fingerprint(config, ROOT)  # noqa: SLF001 -- 复用既有完整环境与资产指纹口径。
            return handle

        try:
            with (self.directory / "fetch_inputs.jsonl").open("x") as self.fetch_stream, \
                    (self.directory / "model_inputs.jsonl").open("x") as self.model_stream:
                dataloader.DataLoaderImpl.__iter__ = observed_iter
                jax.jit = functools.partial(self.compile, original_jit)
                checkpoints.save_state = functools.partial(self.save, original_save)
                final_record.begin = observed_begin
                yield
        finally:
            dataloader.DataLoaderImpl.__iter__ = original_iter
            jax.jit, checkpoints.save_state, final_record.begin = original_jit, original_save, original_begin

    def check_complete(self):
        require(self.iterations == self.jit_matches == self.save_calls == 1 and self.save_returned
                and self.completed_calls == len(self.used) == STEPS and len(self.fetched) == STEPS + 1,
                "取批、模型调用或真实保存次数不完整")


def provenance(entry_equiv, head):
    forbidden = [ROOT / "v1-store/worktrees/orig-ecf086c"]
    entry_equiv._assert_provenance(ROOT, forbidden)  # noqa: SLF001 -- 沿用既有模块根隔离守卫。
    result = entry_equiv._module_provenance(ROOT, ENTRY)  # noqa: SLF001 -- 复用真实模块来源与文件摘要记录。
    entry_equiv._assert_clean_snapshot(result, head)  # noqa: SLF001 -- 与既有对拍保持相同HEAD和clean判据。
    require(Path(result["cwd"]) == ROOT and result["modules"], "运行根或模块证据为空")
    return result


def toolchain():
    names = ("scripts/training/tests/check_orig80k_timing_equiv.py", "scripts/training/tests/check_orig80k_speed.py",
             "scripts/training/tests/entry_equiv.py", "scripts/training/tests/check_orig80k_completion.py",
             "scripts/training/prod/run_orig80k.sh", "scripts/training/config_record.py",
             "scripts/training/final_record.py", "scripts/training/train.py", "pyproject.toml", "uv.lock")
    return {name: record_file(ROOT / name) for name in names}


def execute_entry(timing, argv, records):
    """两侧仅在此处分流；共同观察层在调用者中安装。"""
    require(timing in {"off", "on"}, "未知测速包装模式")
    import check_orig80k_speed as speed
    previous = sys.argv
    try:
        if timing == "on":
            return speed.run(SimpleNamespace(records=str(records), mode="smoke", train_args=argv))
        sys.argv = [str(ENTRY), *argv]
        return runpy.run_path(str(ENTRY), run_name="__main__")
    finally:
        sys.argv = previous


def run(args):
    require(args.timing in {"off", "on"} and os.environ.get("ORIG80K_MODE") == "smoke"
            and os.environ.get("ORIG80K_SMOKE_EQ_MODE") == args.timing, "仅允许显式smoke off/on入口")
    require(Path.cwd() == ROOT and Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(), "必须用主仓独立uv环境")
    require(not os.environ.get("XLA_FLAGS") and os.environ.get("TRAIN_TIMING_STEPS", "0") == "0",
            "入口前不得继承确定性档或计时覆盖")
    records = Path(args.records)
    require(records.is_absolute() and records == records.resolve(), "记录根必须为绝对实体路径")
    require(os.environ.get("TRAIN_RECORD_DIR") == str(records)
            and os.environ.get("TRAIN_FINAL_RECORD_DIR") == str(records / "final"), "共同证据根未绑定runner")
    config_record, _, entry_equiv = training_imports()
    import check_orig80k_speed  # noqa: F401 -- 两侧在来源快照前加载同一观测工具。
    from orig80k_contract import validate_argv
    argv = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    values = validate_argv(argv, "smoke")
    name = values["--exp-name"]
    require(records == ROOT / "v1-store/bench/orig80k" / name, "取证根不是本run的runner记录根")
    require(not any((records / path).exists() for path in SPEED_FILES), "存在旧测速记录")
    head = os.environ.get("MMEVLA_EXPECTED_TRAIN_HEAD", "")
    require(re.fullmatch(r"[0-9a-f]{40}", head), "缺少完整TRAIN_HEAD")
    directory = records / "timing_eq"
    directory.mkdir()  # 新子目录独占创建，不复用旧20步记录。
    observer = Observer(directory)
    previous_argv = sys.argv
    metadata = {"schema": SCHEMA, "status": "FAIL", "timing": args.timing, "head": head, "run_name": name,
                "records": str(records), "pid": os.getpid(), "argv": argv, "toolchain": toolchain(),
                "start_wall": time.time(), "observations": "两侧共同逐批device_get/CPU哈希及完整末步state读回；非吞吐测量"}
    try:
        with observer.installed():
            metadata["provenance_start"] = provenance(entry_equiv, head)
            execute_entry(args.timing, argv, records)
            observer.check_complete()
            metadata["provenance_end"] = provenance(entry_equiv, head)
            metadata["fingerprint"] = observer.fingerprint_start
            require(entry_equiv._runtime_fingerprint(observer.config, ROOT) == metadata["fingerprint"],  # noqa: SLF001 -- 用相同量具复核起止指纹。
                    "起止环境/输入/资产指纹变化")
            metadata["complete"] = observer.identity["complete"]
            require(config_record.complete_record(observer.config) == metadata["complete"], "完整配置发生变化")
            metadata["run_uuid"] = observer.identity["run_uuid"]
            require(toolchain() == metadata["toolchain"], "运行期间工具或依赖文件变化")
            metadata["status"] = "PASS"
    finally:
        sys.argv = previous_argv
        metadata.update(end_wall=time.time(), fetch_count=len(observer.fetched), model_input_count=len(observer.used),
                        completed_calls=observer.completed_calls, loader_iterations=observer.iterations,
                        jit_matches=observer.jit_matches, save_calls=observer.save_calls,
                        original_save_returned=observer.save_returned, original_save=getattr(observer, "original_save", None),
                        loader=observer.loader_contract, observation_seconds=observer.overhead)
        write_json(directory / "metadata.json", metadata)
        write_json(directory / "manifest.json", {"schema": SCHEMA, "files": {
            file: record_file(directory / file) for file in OWNED if (directory / file).is_file()}})
    print(f"TIMING_EQ_RUN=PASS timing={args.timing} fetched=21 used=20 state_step=20 real_save=1", flush=True)


def validate_inputs(fetches, models):
    require(all(set(row) == {"fetch_index", "extra_unused", "tree", "readback_hash_s"}
                and type(row["fetch_index"]) is int and type(row["extra_unused"]) is bool for row in fetches),
            "取批记录schema错误")
    require(all(set(row) == {"step", "fetch_index", "tree", "rng", "readback_hash_s"}
                and type(row["step"]) is int and type(row["fetch_index"]) is int for row in models),
            "模型实参记录schema错误")
    require([row["fetch_index"] for row in fetches] == list(range(21))
            and [row["extra_unused"] for row in fetches] == [False] * 20 + [True], "21次取批范围错误")
    require([row["step"] for row in models] == [row["fetch_index"] for row in models] == list(range(20)),
            "20个真实模型输入范围错误")
    for row in fetches + models:
        validate_tree(row["tree"])
        require(math.isfinite(row["readback_hash_s"]) and row["readback_hash_s"] >= 0, "读回耗时无效")
    for index, row in enumerate(models):
        require(row["tree"] == fetches[index]["tree"], "实际模型输入与该次取批不同")
        require(set(row["rng"]) == {"dtype", "implementation", "key_data"}
                and row["rng"]["dtype"] and row["rng"]["implementation"], "缺实际模型RNG")
        validate_tree(row["rng"]["key_data"])


def validate_state(state, identity):
    require(set(state) == {"schema", "loop_step", "state_step", "run_name", "run_uuid", "head", "groups",
                           "original_treedefs", "ema_leaves", "ema_leaf_paths"}, "完整状态schema错误")
    require(state["schema"] == SCHEMA and state["loop_step"] == 19 and state["state_step"] == 20,
            "完整状态的步骤错误")
    for key in ("run_name", "run_uuid", "head"):
        require(state[key] == identity[key], "完整状态身份不同")
    require(set(state["groups"]) == set(state["original_treedefs"]) == TREE_FIELDS,
            "完整状态缺params/EMA/optimizer/step")
    for name in TREE_FIELDS:
        validate_tree(state["groups"][name])
        require(state["original_treedefs"][name], "缺原状态结构")
    mapping, leaves = state["ema_leaf_paths"], state["groups"]["ema_params"]["leaves"]
    require(mapping and set(mapping) == set(state["ema_leaves"])
            and len(set(mapping.values())) == len(mapping) and set(mapping.values()) == set(leaves),
            "完整状态缺EMA恢复映射或映射未覆盖所有叶")
    require(all(leaves[path]["kind"] == "array" and leaves[path]["array"] == state["ema_leaves"][key]
                for key, path in mapping.items()), "完整状态EMA组与恢复映射分离")


def validate_fingerprint(fingerprint):
    fields = {"version", "python", "dependencies", "uv_lock_sha256", "devices", "environment", "jax",
              "history", "assets", "source", "dataset"}
    require(set(fingerprint) == fields and fingerprint["version"] == 1
            and all(fingerprint[key] for key in fields - {"version"}), "缺完整运行指纹")
    devices = fingerprint["devices"]
    require(len(devices) == 4 and len({item["uuid"] for item in devices}) == 4
            and all(set(item) == {"index", "uuid", "name", "driver"} and all(item.values()) for item in devices),
            "指纹没有四个不同物理GPU")
    require(fingerprint["jax"]["enable_x64"] is False and not fingerprint["environment"]["XLA_FLAGS"],
            "运行没有使用生产x64/确定性环境口径")
    require(fingerprint["uv_lock_sha256"] == record_file(ROOT / "uv.lock")["sha256"], "依赖锁指纹不同")
    require(record_file(Path(fingerprint["history"]["path"])) == fingerprint["history"], "history内容不同")
    require({"episode_manifest", "stats", "input_manifest.json"} <= set(fingerprint["source"]["records"]),
            "缺数据来源pin/清单/统计")
    files = list(fingerprint["source"]["records"].values())
    assets = fingerprint["assets"]
    require(set(assets) == {"norm_stats", "initialization", "tokenizer"} and assets["initialization"], "缺真实资产指纹")
    files += [assets["norm_stats"], assets["tokenizer"], *assets["initialization"].values()]
    require(fingerprint["dataset"]["store_meta"] is not None, "本验证必须使用packed真实训练路径")
    files.append(fingerprint["dataset"]["store_meta"])
    require(all(record_file(Path(item["path"])) == item for item in files), "数据/资产指纹所绑定内容变化")


def validate_provenance(metadata, head):
    _, _, entry_equiv = training_imports()
    snapshots = [metadata["provenance_start"], metadata["provenance_end"]]
    for snapshot in snapshots:
        entry_equiv._assert_clean_snapshot(snapshot, head)  # noqa: SLF001 -- 判定器复用原始快照的HEAD和clean校验。
        require(snapshot["expect_root"] == snapshot["git_root"] == snapshot["cwd"] == str(ROOT)
                and snapshot["entry"]["file"] == str(ENTRY) and snapshot["modules"], "来源根/入口不符")
        for item in [snapshot["entry"], *snapshot["modules"].values()]:
            path = Path(item["file"])
            require(path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / "v1-store"), "项目模块混入产物/上游根")
            require(hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], "当前项目模块与记录不同")
    require(snapshots[0]["entry"] == snapshots[1]["entry"] and all(
        snapshots[1]["modules"].get(name) == value for name, value in snapshots[0]["modules"].items()),
        "起止模块或入口变化")


def validate_wrapper(path, identity, completion):
    lines = Path(path).read_text().splitlines()
    def field(name):
        matches = [line[len(name) + 1:] for line in lines if line.startswith(name + "=")]
        require(len(matches) == 1, "外层回执缺失或重复: " + name)
        return matches[0]
    for key in WRAPPER_EXITS:
        require(field(key) == "0", "外层或子命令失败: " + key)
    for key, value in {"WRAPPER_RUN": identity["run_name"], "WRAPPER_RUN_UUID": identity["run_uuid"],
                       "WRAPPER_TRAIN_HEAD": identity["head"],
                       "WRAPPER_COMPLETION_SHA256": record_file(completion)["sha256"]}.items():
        require(field(key) == value, "外层回执身份或完成结果不符")
    command = Path(field("WRAPPER_COMMAND_FILE"))
    require(command.is_relative_to(ROOT / "v1-store"), "外层命令文件越界")
    command_record = record_file(command)
    require(field("WRAPPER_COMMAND_SHA256_EXPECTED") == field("WRAPPER_COMMAND_SHA256_ACTUAL")
            == command_record["sha256"], "外层命令文件摘要不同")
    return {"log": record_file(path), "command": command_record}


def validate_completion(records, completion_path, identity, full_state):
    import check_orig80k_completion as completion_tool
    result = load(completion_path)
    run, head = identity["run_name"], identity["head"]
    checkpoint = ROOT / "v1-store/train-runs/mme_vla_suite" / run / "19"
    driver = ROOT / "v1-store/logs" / (run + ".driver.log")
    expected = [records / name for name in ("launch.json", "runtime.json", "metrics.jsonl", "gpu.csv", "gpu.csv.err")]
    expected += [records / "final" / name for name in ("start.json", "final.json", "checkpoint_wait_done.json")]
    expected += [driver, checkpoint / "_CHECKPOINT_METADATA", checkpoint / "params/_METADATA"]
    actual = result.get("files", [])
    require(len(actual) == len(expected) and {row["path"] for row in actual} == {str(path) for path in expected},
            "真实完成器的完整文件证据集合缺失/重复/越界")
    require(all(record_file(Path(row["path"])) == row for row in actual), "完成器所绑定文件发生变化")
    require(result.get("status") == "PASS" and result.get("mode") == "smoke" and result.get("state_step") == 20
            and result.get("final") == 19 and result.get("checkpoints") == [19], "未通过本次真实恢复")
    for key in ("run_name", "run_uuid", "head"):
        require(result.get(key) == identity[key], "恢复结果身份不符")
    start, final, wait = [load(records / "final" / name)
                          for name in ("start.json", "final.json", "checkpoint_wait_done.json")]
    metrics = rows(records / "metrics.jsonl")
    lines = driver.read_text().splitlines()
    completion_tool.validate_records(start, final, wait, metrics,
        [line for line in lines if line.startswith("EXIT_CODE=")], [19], mode="smoke", run=run, head=head)
    for key in ("TRAIN_PIPE_EXIT", "TEE_EXIT", "FOOTER_PRINTF_EXIT", "FOOTER_TEE_EXIT"):
        require([line for line in lines if line.startswith(key + "=")] == [key + "=0"], "driver终态不完整")
    launch, runtime = load(records / "launch.json"), load(records / "runtime.json")
    require(launch["head"] == head and launch["run_name"] == run and launch["mode"] == "smoke"
            and launch["actual"]["complete"] == start["complete"] == identity["complete"]
            and launch["actual"]["jax_enable_x64"] is False, "启动完整配置与真实配置不符")
    require((runtime["exp_name"], runtime["config_name"], runtime["seed"], runtime["device_count"],
             runtime["batch_size"], runtime["num_workers"], runtime["fsdp_devices"]) ==
            (run, "mme_vla_suite", 42, 4, 64, 4, 4), "实际runtime与本次身份/四卡配置不符")
    gpu_ids = [int(item) for item in runtime["cuda_visible_devices"].split(",")]
    require(completion_tool.validate_gpu_sampling(records, lines, gpu_ids, start, wait) == result["gpu_sampling"],
            "恢复结果GPU覆盖记录不符")
    require(final["ema_leaves"] == result["restored_leaves"] == full_state["ema_leaves"],
            "真实恢复、末步EMA与完整状态摘要不一致")
    scalars = [{"step": 0, "scalars": {key: metrics[0][key] for key in completion_tool.SCALARS}}]
    scalars += final["tail"]
    require([item["step"] for item in scalars] == list(range(20)), "五标量没有完整20步")
    return result, [{"step": item["step"], "hex": {key: value["hex"] for key, value in item["scalars"].items()}}
                    for item in scalars]


def validate_speed(records, timing, head):
    if timing == "off":
        require(not any((records / name).exists() for name in SPEED_FILES), "off侧实际启用了测速包装")
        return None
    import check_orig80k_speed as speed
    meta = load(records / "speed_run.json")
    require(meta.get("head") == head and meta.get("mode") == "smoke" and meta.get("steps") == 20
            and meta.get("success") and meta.get("entry_calls") == 1 and meta.get("sampler_stopped")
            and not meta.get("sampling_error") and meta.get("profiler") is False, "on侧不是成功的真实smoke测速包装")
    require(meta["wrapper_sha256"] == record_file(Path(speed.__file__).resolve())["sha256"]
            and meta["loader"]["batch_size"] == 64 and meta["loader"]["workers"] == 4, "测速来源或loader错误")
    require(len(meta["save_calls"]) == 1 and meta["save_calls"][0]["step"] == 19
            and meta["save_calls"][0]["state_step"] == 20, "测速没有转发末步真实保存")
    timing_rows = rows(records / "step_timing.jsonl")
    require([row["step"] for row in timing_rows] == list(range(20))
            and all(row.get("completed") for row in timing_rows), "20步包装计时不完整")
    require([(row["step"], row["sync"]["location"]) for row in timing_rows if "sync" in row]
            == [(19, "before_final_save")], "20步测速同步边界错误")
    return {"files": [record_file(records / name) for name in SPEED_FILES],
            "disk": speed.summarize_disk(records, meta)}


def read_side(records, completion_path, wrapper_path, timing, head):
    records = Path(records)
    require(records.is_absolute() and records == records.resolve(), "记录根必须为实体绝对路径")
    directory = records / "timing_eq"
    manifest = load(directory / "manifest.json")
    require(manifest["schema"] == SCHEMA and set(manifest["files"]) == set(OWNED), "共同取证文件集合不完整")
    require({path.name for path in directory.iterdir()} == {*OWNED, "manifest.json"}, "共同取证目录含缺失/额外文件")
    for name in OWNED:
        require(record_file(directory / name) == manifest["files"][name], "共同取证文件摘要不同")
    meta, state = load(directory / "metadata.json"), load(directory / "full_state.json")
    require(meta["schema"] == SCHEMA and meta["status"] == "PASS" and meta["timing"] == timing
            and meta["head"] == head and meta["records"] == str(records), "共同取证身份/状态不符")
    require(records == ROOT / "v1-store/bench/orig80k" / meta["run_name"], "记录路径未绑定本run")
    require((meta["fetch_count"], meta["model_input_count"], meta["completed_calls"], meta["loader_iterations"],
             meta["jit_matches"], meta["save_calls"], meta["original_save_returned"]) == (21, 20, 20, 1, 1, 1, True),
            "共同取证没有覆盖真实输入与原始保存")
    require(meta["original_save_returned"] is True and all(type(meta[name]) is int for name in
            ("fetch_count", "model_input_count", "completed_calls", "loader_iterations", "jit_matches", "save_calls")),
            "取证计数类型错误")
    require(meta["toolchain"] == toolchain(), "取证/训练/测速/完成器或依赖文件与当前版本不同")
    require(meta["original_save"] == {"file": str(ROOT / "src/openpi/training/checkpoints.py"),
                                       "module": "openpi.training.checkpoints", "name": "save_state"},
            "没有绑定生产原始save_state")
    validate_provenance(meta, head)
    validate_fingerprint(meta["fingerprint"])
    require(set(meta["loader"]) == {"batch_size", "workers", "prefetch_factor", "persistent_workers", "drop_last"}
            and meta["loader"]["batch_size"] == 64 and meta["loader"]["workers"] == 4,
            "缺实际最终loader形制")
    require(set(meta["observation_seconds"]) == {"fetch_readback_hash_s", "model_readback_hash_s", "state_readback_hash_s"}
            and all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                    for value in meta["observation_seconds"].values()), "共同只读取证耗时错误")
    fetches, models = rows(directory / "fetch_inputs.jsonl"), rows(directory / "model_inputs.jsonl")
    validate_inputs(fetches, models)
    validate_state(state, meta)
    restored, scalars = validate_completion(records, completion_path, meta, state)
    wrapper = validate_wrapper(wrapper_path, meta, completion_path)
    speed = validate_speed(records, timing, head)
    return {"metadata": meta, "state": state, "fetches": fetches, "models": models, "scalars": scalars,
            "bindings": {"manifest": record_file(directory / "manifest.json"),
                         "completion": record_file(completion_path), "completion_files": restored["files"],
                         "wrapper": wrapper, "speed": speed}}


def compare_sides(off, on):
    config_record, _, _ = training_imports()
    left, right = off["metadata"], on["metadata"]
    require(left["head"] == right["head"] and left["run_name"] != right["run_name"]
            and left["run_uuid"] != right["run_uuid"] and left["records"] != right["records"], "对照必须是同HEAD的独立新run")
    require(left["fingerprint"] == right["fingerprint"] and left["toolchain"] == right["toolchain"],
            "依赖/设备/种子/数据/资产/有效环境或工具不同")
    require(left["loader"] == right["loader"], "两侧实际loader形制不同")
    require(left["provenance_end"] == right["provenance_end"], "两侧项目模块来源不同")
    changes = config_record.compare_records(left["complete"], right["complete"], config_record.IDENTITY_FIELDS)
    require(left["original_save"] == right["original_save"], "两侧原始保存函数不同")
    require([row["tree"] for row in off["fetches"]] == [row["tree"] for row in on["fetches"]], "21次取批逐位不一致")
    require([(row["tree"], row["rng"]) for row in off["models"]] ==
            [(row["tree"], row["rng"]) for row in on["models"]], "20次模型实参与RNG逐位不一致")
    require(off["scalars"] == on["scalars"], "20步五标量不是逐位一致")
    for key in ("groups", "original_treedefs", "ema_leaves", "ema_leaf_paths", "state_step", "loop_step"):
        require(off["state"][key] == on["state"][key], "最终完整参数/EMA/优化器/step或结构不同: " + key)
    return changes


def judge(args):
    require(re.fullmatch(r"[0-9a-f]{40}", args.head), "judge需要完整HEAD")
    off = read_side(args.off_records, args.off_completion, args.off_wrapper_log, "off", args.head)
    on = read_side(args.on_records, args.on_completion, args.on_wrapper_log, "on", args.head)
    changes = compare_sides(off, on)
    output = Path(args.out)
    require(output.is_absolute() and output == output.resolve() and output.is_relative_to(ROOT / "v1-store")
            and not any(output.is_relative_to(Path(side["metadata"]["records"])) for side in (off, on)),
            "独立judge结果不能写入任一run的取证目录")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"schema": SCHEMA, "status": "PASS", "head": args.head, "used_batches": 20,
        "fetched_batches": 21, "extra_unused_batches": 1, "scalar_steps": 20, "state_step": 20,
        "loop_step": 19, "config_identity_differences": changes,
        "off": off["bindings"], "on": on["bindings"],
        "observation_seconds": {key: side["metadata"]["observation_seconds"] for key, side in (("off", off), ("on", on))},
        "scope": "同一额外只读取证层下20步包装off/on逐位相等，含各自真实EMA恢复；不是吞吐或无观测运行证明"})
    print("TIMING_EQ=PASS fetched=21 used=20 scalar_steps=20 full_state=bitwise real_restore=both", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--timing", required=True, choices=("off", "on"))
    execute.add_argument("--records", required=True)
    execute.add_argument("train_args", nargs=argparse.REMAINDER)
    compare = commands.add_parser("judge")
    for key in ("off-records", "on-records", "off-completion", "on-completion", "off-wrapper-log", "on-wrapper-log", "head", "out"):
        compare.add_argument("--" + key, required=True)
    args = parser.parse_args()
    try:
        run(args) if args.command == "run" else judge(args)
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        print(f"TIMING_EQ_{'RUN' if args.command == 'run' else 'JUDGE'}=FAIL reason={error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
