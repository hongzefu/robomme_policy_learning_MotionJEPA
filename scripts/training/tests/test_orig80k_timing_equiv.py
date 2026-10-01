"""测速包装20步对照的CPU测试；含显式档位、记录闭合与原对象转发。"""

from __future__ import annotations

import contextlib
import copy
import functools
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_orig80k_timing_equiv as eq


def synthetic_state():
    return SimpleNamespace(params={"weight": np.asarray([1., -0.], dtype=np.float32)},
                           ema_params={"layer": {"weight": np.asarray([1., 2.], dtype=np.float32)}},
                           opt_state=({"mu": np.asarray([3., 4.], dtype=np.float32)}, np.asarray(20, dtype=np.int32)),
                           step=np.asarray(20, dtype=np.int32))


def identity(name="off"):
    return {"head": "a" * 40, "run_name": name, "run_uuid": name + "-uuid"}


def state_record(name="off"):
    return {"schema": eq.SCHEMA, "loop_step": 19, **identity(name), **eq.full_state_record(synthetic_state())}


def test_tree_bytes_dtype_shape_and_none_are_strict_and_do_not_mutate():
    source = np.asarray([-0., 1.], dtype=np.float32)
    before = source.tobytes()
    record = eq.tree_record({"x": source, "absent": None})
    eq.validate_tree(record)
    assert source.tobytes() == before
    assert record != eq.tree_record({"x": np.asarray([0., 1.], dtype=np.float32), "absent": None})
    assert record != eq.tree_record({"x": source.astype(np.float64), "absent": None})
    assert record != eq.tree_record({"x": source.reshape(1, 2), "absent": None})
    assert record != eq.tree_record({"x": source})
    assert eq.tree_record(np.asarray([2**53], dtype=np.int64)) != eq.tree_record(np.asarray([2**53 + 1], dtype=np.int64))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_identical_nonfinite_values_never_produce_pass(value):
    with pytest.raises(ValueError, match="非有限"):
        eq.tree_record(np.asarray([value]))


def test_malformed_bytes_and_missing_optimizer_rejected():
    record = eq.tree_record(np.ones(2, dtype=np.float32))
    next(iter(record["leaves"].values()))["array"]["bytes"] += 1
    record["sha256"] = eq.digest({key: record[key] for key in ("treedef", "leaves")})
    with pytest.raises(ValueError, match="字节数"):
        eq.validate_tree(record)
    state = state_record()
    del state["groups"]["opt_state"]
    with pytest.raises(ValueError, match="optimizer"):
        eq.validate_state(state, identity())


def test_full_state_includes_optimizer_step_and_binds_ema_mapping():
    left = state_record()
    eq.validate_state(left, identity())
    changed = synthetic_state()
    changed.opt_state[0]["mu"][0] += 1
    assert eq.full_state_record(changed)["groups"]["opt_state"] != left["groups"]["opt_state"]
    # 从JSON读回的对象没有内存别名，模拟修改独立的恢复映射。
    detached = json.loads(json.dumps(state_record()))
    detached["ema_leaves"]["layer/weight"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="恢复映射分离"):
        eq.validate_state(detached, identity())


def input_rows():
    tree = eq.tree_record(np.asarray([1, 2], dtype=np.int32))
    fetched = [{"fetch_index": i, "extra_unused": i == 20, "tree": copy.deepcopy(tree), "readback_hash_s": 0.}
               for i in range(21)]
    used = [{"step": i, "fetch_index": i, "tree": copy.deepcopy(tree), "readback_hash_s": 0.,
             "rng": {"dtype": "key<fry>", "implementation": "threefry2x32", "key_data": copy.deepcopy(tree)}}
            for i in range(20)]
    return fetched, used


@pytest.mark.parametrize("fault", ["missing_extra", "extra_used", "wrong_model_batch", "duplicate_step", "missing_rng"])
def test_input_range_and_actual_model_boundary_faults(fault):
    fetched, used = input_rows()
    eq.validate_inputs(fetched, used)
    if fault == "missing_extra":
        fetched.pop()
    elif fault == "extra_used":
        used.append(copy.deepcopy(used[-1]))
    elif fault == "wrong_model_batch":
        used[3]["tree"] = eq.tree_record(np.asarray([99, 2], dtype=np.int32))
    elif fault == "duplicate_step":
        used[4]["step"] = 3
    else:
        del used[0]["rng"]["key_data"]
    with pytest.raises((ValueError, KeyError)):
        eq.validate_inputs(fetched, used)


def test_narrow_jit_hook_preserves_constructor_arguments_and_actual_objects(tmp_path):
    import jax

    from openpi.shared import array_typing as at
    namespace = {}
    exec(compile("def train_step(config, rng, state, batch):\n    return state, batch\n", str(eq.ENTRY), "exec"), namespace)
    target = functools.partial(at.typecheck(namespace["train_step"]), object())
    observer = eq.Observer(tmp_path)
    observer.fetch_stream, observer.model_stream = io.StringIO(), io.StringIO()
    constructor = []
    calls = []
    expected = object()
    def original(function, *args, **kwargs):
        constructor.append((function, args, kwargs))
        def compiled(rng, state, batch):
            calls.append((rng, state, batch))
            return expected
        return compiled
    in_sharding, out_sharding = object(), object()
    compiled = observer.compile(original, target, in_shardings=in_sharding,
                                out_shardings=out_sharding, donate_argnums=(1,))
    rng, state, batch = jax.random.key(42), object(), (np.ones(2, dtype=np.float32),)
    observer.fetched_batch(batch)
    assert compiled(rng, state, batch) is expected
    assert constructor[0][0] is target
    assert constructor[0][2] == {"in_shardings": in_sharding, "out_shardings": out_sharding, "donate_argnums": (1,)}
    assert calls[0][0] is rng
    assert calls[0][1] is state
    assert calls[0][2] is batch
    assert observer.jit_matches == observer.completed_calls == 1
    def unrelated(value):
        return value

    plain = observer.compile(original, unrelated)
    assert plain is not compiled
    assert observer.jit_matches == 1
    other = {}
    exec(compile("def train_step(config, rng, state, batch):\n    return state, batch\n", "/other/train.py", "exec"), other)
    observer.compile(original, functools.partial(at.typecheck(other["train_step"]), object()))
    assert observer.jit_matches == 1


def test_observer_restores_every_hook_after_failure(tmp_path):
    import jax

    from mme_vla_suite.training import dataloader
    from openpi.training import checkpoints
    _, final_record, _ = eq.training_imports()
    originals = (jax.jit, dataloader.DataLoaderImpl.__iter__, checkpoints.save_state, final_record.begin)
    observer = eq.Observer(tmp_path)
    def fail_inside_observer():
        with observer.installed():
            assert jax.jit is not originals[0]
            assert checkpoints.save_state is not originals[2]
            raise RuntimeError("fixture")

    with pytest.raises(RuntimeError, match="fixture"):
        fail_inside_observer()
    assert (jax.jit, dataloader.DataLoaderImpl.__iter__, checkpoints.save_state, final_record.begin) == originals


@pytest.mark.parametrize("timing", ["off", "on"])
def test_entry_dispatch_keeps_original_argv_and_only_on_uses_smoke_speed(tmp_path, monkeypatch, timing):
    calls = []
    returned = object()
    def speed_run(args):
        calls.append(("on", args))
        return returned
    def direct_run(path, *, run_name):
        calls.append(("off", path, run_name, list(sys.argv)))
        return returned
    monkeypatch.setitem(sys.modules, "check_orig80k_speed", SimpleNamespace(run=speed_run))
    monkeypatch.setattr(eq.runpy, "run_path", direct_run)
    argv, previous = ["mme_vla_suite", "--num-train-steps", "20"], sys.argv
    assert eq.execute_entry(timing, argv, tmp_path) is returned
    assert sys.argv is previous
    assert len(calls) == 1
    assert calls[0][0] == timing
    if timing == "on":
        assert calls[0][1].mode == "smoke"
        assert calls[0][1].train_args is argv
    else:
        assert calls[0][1:] == (str(eq.ENTRY), "__main__", [str(eq.ENTRY), *argv])


@pytest.mark.parametrize("timing", ["off", "on"])
def test_shared_state_observer_forwards_real_save_once_and_preserves_result(tmp_path, timing):
    observer = eq.Observer(tmp_path)
    observer.identity = identity(timing)
    observer.completed_calls = 20
    observer.fetched = [None] * 21
    state, manager, loader, returned = synthetic_state(), object(), object(), object()
    called = []
    before = state.opt_state[0]["mu"].tobytes()
    def save_state(*args):
        called.append(args)
        (tmp_path / "real_save_marker").write_text("真实保存替身被调用")
        return returned
    common = functools.partial(observer.save, save_state)
    if timing == "on":
        def speed_save(*args):
            return common(*args)
        call = speed_save
    else:
        call = common
    assert call(manager, state, loader, 19) is returned
    assert len(called) == 1
    assert all(a is b for a, b in zip(called[0][:3], (manager, state, loader), strict=True))
    assert (tmp_path / "real_save_marker").is_file()
    assert observer.save_returned
    assert state.opt_state[0]["mu"].tobytes() == before
    eq.validate_state(eq.load(tmp_path / "full_state.json"), identity(timing))
    with pytest.raises(ValueError, match="真实保存"):
        call(manager, state, loader, 19)
    assert len(called) == 1


def side(name):
    fetched, used = input_rows()
    return {"metadata": {**identity(name), "records": "/records/" + name, "timing": name,
            "timing_eq_profile": {"name": "normal", "xla_flags": ""},
            "fingerprint": {"same": "environment", "environment": {"XLA_FLAGS": None}},
            "toolchain": {"same": "files"},
            "loader": {"batch_size": 64, "workers": 4}, "provenance_end": {"same": "modules"},
            "original_save": {"same": "save_state"},
            "complete": {"config_record_version": 2, "train_config": {"fields": {"exp_name": name, "seed": 42}},
                         "derived": {"checkpoint_dir": "/runs/" + name}}},
            "state": state_record(name), "fetches": fetched, "models": used,
            "scalars": [{"step": step, "hex": {"loss": "0x1.0p+0"}} for step in range(20)]}


@pytest.mark.parametrize("fault", ["head", "config", "fingerprint", "modules", "input", "scalar", "optimizer", "same_run"])
def test_pair_comparison_rejects_all_nonidentity_changes(fault):
    off, on = side("off"), side("on")
    assert eq.compare_sides(off, on)
    if fault == "head":
        on["metadata"]["head"] = "b" * 40
    elif fault == "config":
        on["metadata"]["complete"]["train_config"]["fields"]["seed"] = 43
    elif fault == "fingerprint":
        on["metadata"]["fingerprint"]["same"] = "different"
    elif fault == "modules":
        on["metadata"]["provenance_end"]["same"] = "different"
    elif fault == "input":
        on["models"][0]["tree"] = eq.tree_record(np.ones(3))
    elif fault == "scalar":
        on["scalars"][19]["hex"]["loss"] = "0x1.0000000000001p+0"
    elif fault == "optimizer":
        on["state"]["groups"]["opt_state"] = eq.tree_record(np.ones(3))
    else:
        on["metadata"]["run_name"] = "off"
    messages = {"head": "同HEAD", "same_run": "独立新run", "config": "配置出现未授权差异",
                "fingerprint": "依赖/设备/种子/数据/资产/有效环境或工具不同", "modules": "两侧项目模块来源不同",
                "input": "20次模型实参与RNG逐位不一致", "scalar": "20步五标量不是逐位一致",
                "optimizer": "最终完整参数/EMA/优化器/step或结构不同"}
    with pytest.raises(ValueError, match=messages[fault]):
        eq.compare_sides(off, on)


def wrapper_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(eq, "ROOT", tmp_path)
    folder = tmp_path / "v1-store"
    folder.mkdir()
    command, completion, log = (folder / name for name in ("command.txt", "completion.json", "wrapper.log"))
    command.write_text("候选命令记录，不执行\n")
    completion.write_text("{}\n")
    sha = eq.record_file(command)["sha256"]
    values = dict.fromkeys(eq.WRAPPER_EXITS, "0")
    values.update(WRAPPER_RUN="off", WRAPPER_RUN_UUID="off-uuid", WRAPPER_TRAIN_HEAD="a" * 40,
                  WRAPPER_COMMAND_FILE=str(command), WRAPPER_COMMAND_SHA256_EXPECTED=sha,
                  WRAPPER_COMMAND_SHA256_ACTUAL=sha, WRAPPER_COMPLETION_SHA256=eq.record_file(completion)["sha256"])
    return command, completion, log, values


def test_wrapper_uses_independent_names_and_binds_real_command(tmp_path, monkeypatch):
    command, completion, log, values = wrapper_fixture(tmp_path, monkeypatch)
    text = "EXIT_CODE=0\nEXIT_CODE=0\nFOOTER_TEE_EXIT=0\n"
    log.write_text(text + "".join(f"{key}={value}\n" for key, value in values.items()))
    assert eq.validate_wrapper(log, identity(), completion)
    command.write_text("改动\n")
    with pytest.raises(ValueError, match="命令文件摘要"):
        eq.validate_wrapper(log, identity(), completion)


@pytest.mark.parametrize("fault", ["RUNNER_EXIT_CODE", "COMPLETION_EXIT_CODE", "WRAPPER_FOOTER_TEE_EXIT", "duplicate", "missing"])
def test_wrapper_failure_or_ambiguous_record_rejected(tmp_path, monkeypatch, fault):
    _, completion, log, values = wrapper_fixture(tmp_path, monkeypatch)
    if fault == "missing":
        values.pop("WRAPPER_EXIT_CODE")
    elif fault != "duplicate":
        values[fault] = "29"
    log.write_text("".join(f"{key}={value}\n" for key, value in values.items())
                   + ("WRAPPER_EXIT_CODE=0\n" if fault == "duplicate" else ""))
    message = "外层回执缺失或重复" if fault in {"duplicate", "missing"} else "外层或子命令失败"
    with pytest.raises(ValueError, match=message):
        eq.validate_wrapper(log, identity(), completion)


def test_completion_pass_without_real_file_membership_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(eq, "ROOT", tmp_path)
    result = tmp_path / "completion.json"
    result.write_text(json.dumps({"status": "PASS", "files": []}))
    with pytest.raises(ValueError, match="文件证据集合"):
        eq.validate_completion(tmp_path, result, identity(), {})


def profile(name="normal"):
    flags = eq.profile_contract().TIMING_EQ_DETERMINISTIC_FLAGS if name == "deterministic100" else ""
    return {"name": name, "xla_flags": flags}


def profile_metadata(name="normal", timing="off"):
    value = profile(name)
    return {**identity(timing), "timing": timing, "timing_eq_profile": value,
            "fingerprint": {"environment": {"XLA_FLAGS": value["xla_flags"] or None}}}


def profile_launch(metadata):
    value = metadata["timing_eq_profile"]
    return {"mode": "smoke", "timing_eq_profile": copy.deepcopy(value), "environment": {
        "XLA_FLAGS": metadata["fingerprint"]["environment"]["XLA_FLAGS"],
        "ORIG80K_SMOKE_EQ_MODE": metadata["timing"],
        "ORIG80K_TIMING_EQ_PROFILE": value["name"] if value["name"] == "deterministic100" else None}}


@pytest.mark.parametrize("name", ["normal", "deterministic100"])
@pytest.mark.parametrize("timing", ["off", "on"])
def test_record_profile_uses_recorded_environment_only(monkeypatch, name, timing):
    metadata = profile_metadata(name, timing)
    launch = profile_launch(metadata)
    monkeypatch.setenv("XLA_FLAGS", "不属于被判定运行的当前环境")
    monkeypatch.setenv("ORIG80K_TIMING_EQ_PROFILE", "bad")
    monkeypatch.setenv("ORIG80K_SMOKE_EQ_MODE", "bad")
    assert eq.validate_metadata_profile(metadata) == profile(name)
    assert eq.validate_launch_profile(launch, metadata) == profile(name)


@pytest.mark.parametrize("fault", ["missing_metadata", "none_metadata", "extra_metadata", "missing_flags",
    "wrong_flags", "missing_launch", "missing_selector", "wrong_selector", "wrong_timing", "mixed_profiles"])
def test_profile_records_reject_missing_tampered_and_mixed_fields(fault):
    metadata = profile_metadata("deterministic100")
    launch = profile_launch(metadata)
    if fault == "missing_metadata":
        del metadata["timing_eq_profile"]
    elif fault == "none_metadata":
        metadata["timing_eq_profile"] = None
    elif fault == "extra_metadata":
        metadata["timing_eq_profile"]["unused"] = True
    elif fault == "missing_flags":
        del metadata["fingerprint"]["environment"]["XLA_FLAGS"]
    elif fault == "wrong_flags":
        metadata["fingerprint"]["environment"]["XLA_FLAGS"] += " --extra-flag"
    elif fault == "missing_launch":
        del launch["timing_eq_profile"]
    elif fault == "missing_selector":
        del launch["environment"]["ORIG80K_TIMING_EQ_PROFILE"]
    elif fault == "wrong_selector":
        launch["environment"]["ORIG80K_TIMING_EQ_PROFILE"] = None
    elif fault == "wrong_timing":
        launch["environment"]["ORIG80K_SMOKE_EQ_MODE"] = "on"
    else:
        launch = profile_launch(profile_metadata("normal"))
    with pytest.raises((ValueError, KeyError)):
        eq.validate_launch_profile(launch, metadata)


def test_pair_profile_difference_is_not_a_config_identity_exception():
    off, on = side("off"), side("on")
    on["metadata"]["timing_eq_profile"] = profile("deterministic100")
    on["metadata"]["fingerprint"]["environment"]["XLA_FLAGS"] = profile("deterministic100")["xla_flags"]
    with pytest.raises(ValueError, match="两侧计时档位不同"):
        eq.compare_sides(off, on)
    config_record, _, _ = eq.training_imports()
    assert config_record.IDENTITY_FIELDS == ("train_config.fields.exp_name", "derived.checkpoint_dir")


def fingerprint_fixture(tmp_path, monkeypatch, name):
    monkeypatch.setattr(eq, "ROOT", tmp_path)
    files = {}
    for filename in ("uv.lock", "history.yaml", "norm.json", "tokenizer", "init", "manifest", "stats", "input", "store"):
        path = tmp_path / filename
        path.write_text("本测试的小型固定资产\n")
        files[filename] = eq.record_file(path)
    return {"version": 1, "python": {"version": "fixture"}, "dependencies": {"fixture": "1"},
            "uv_lock_sha256": files["uv.lock"]["sha256"],
            "devices": [{"index": str(i), "uuid": f"GPU-{i}", "name": "fixture", "driver": "fixture"}
                        for i in range(4)],
            "environment": {"XLA_FLAGS": profile(name)["xla_flags"] or None}, "jax": {"enable_x64": False},
            "history": files["history.yaml"], "assets": {"norm_stats": files["norm.json"],
                "tokenizer": files["tokenizer"], "initialization": {"metadata": files["init"]}},
            "source": {"records": {"episode_manifest": files["manifest"], "stats": files["stats"],
                                   "input_manifest.json": files["input"]}},
            "dataset": {"store_meta": files["store"]}}


@pytest.mark.parametrize("name", ["normal", "deterministic100"])
def test_fingerprint_closes_profile_to_actual_flags_and_keeps_x64_false(tmp_path, monkeypatch, name):
    fingerprint = fingerprint_fixture(tmp_path, monkeypatch, name)
    monkeypatch.setenv("XLA_FLAGS", "仅污染判定环境")
    eq.validate_fingerprint(fingerprint, profile(name), timing="on")
    altered = copy.deepcopy(fingerprint)
    altered["environment"]["XLA_FLAGS"] = "" if name == "deterministic100" else profile("deterministic100")["xla_flags"]
    with pytest.raises(ValueError, match="实际XLA_FLAGS"):
        eq.validate_fingerprint(altered, profile(name), timing="on")
    fingerprint["jax"]["enable_x64"] = True
    with pytest.raises(ValueError, match="x64"):
        eq.validate_fingerprint(fingerprint, profile(name), timing="on")


@pytest.mark.parametrize("fault", ["missing", "none", "legacy_schema"])
def test_read_side_rejects_missing_profile_even_with_rehashed_manifest(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(eq, "ROOT", tmp_path)
    records = tmp_path / "v1-store/bench/orig80k/off"
    directory = records / "timing_eq"
    directory.mkdir(parents=True)
    metadata = {**profile_metadata(), "schema": eq.SCHEMA, "status": "PASS", "records": str(records)}
    if fault == "missing":
        del metadata["timing_eq_profile"]
    elif fault == "none":
        metadata["timing_eq_profile"] = None
    else:
        metadata["schema"] = 1
    for name in eq.OWNED:
        (directory / name).write_text(json.dumps(metadata if name == "metadata.json" else {}))
    eq.write_json(directory / "manifest.json", {"schema": eq.SCHEMA, "files": {
        name: eq.record_file(directory / name) for name in eq.OWNED}})
    with pytest.raises((ValueError, KeyError)):
        eq.read_side(records, records / "completion.json", records / "wrapper.log", "off", "a" * 40)


def speed_fixture(tmp_path, monkeypatch, name):
    import check_orig80k_speed as speed
    monkeypatch.setattr(speed, "summarize_disk", lambda *_: {"fixture": "保留原磁盘验收接口"})
    flags = profile(name)["xla_flags"] or None
    shared = {"schema": 2, "head": "a" * 40, "mode": "smoke", "timing_eq_profile": profile(name), "xla_flags": flags}
    meta = {**shared, "steps": 20, "success": True, "entry_calls": 1, "sampler_stopped": True,
            "sampling_error": None, "profiler": False,
            "wrapper_sha256": eq.record_file(Path(speed.__file__).resolve())["sha256"],
            "loader": {"batch_size": 64, "workers": 4}, "save_calls": [{"step": 19, "state_step": 20}]}
    for filename in eq.SPEED_FILES:
        (tmp_path / filename).write_text("{}\n")
    (tmp_path / "speed_start.json").write_text(json.dumps(shared))
    (tmp_path / "speed_run.json").write_text(json.dumps(meta))
    (tmp_path / "step_timing.jsonl").write_text("".join(json.dumps({"step": i, "completed": True,
        **({"sync": {"location": "before_final_save"}} if i == 19 else {})}) + "\n" for i in range(20)))
    return flags


@pytest.mark.parametrize("name", ["normal", "deterministic100"])
def test_speed_start_and_run_profile_use_records_not_judge_environment(tmp_path, monkeypatch, name):
    flags = speed_fixture(tmp_path, monkeypatch, name)
    monkeypatch.setenv("XLA_FLAGS", "判定侧任意环境")
    assert eq.validate_speed(tmp_path, "on", "a" * 40, profile(name), xla_flags=flags)["files"]


@pytest.mark.parametrize("filename", ["speed_start.json", "speed_run.json"])
@pytest.mark.parametrize("fault", ["missing", "none", "wrong_flags", "mixed", "legacy_schema"])
def test_speed_profile_cannot_be_missing_or_disagree_between_records(tmp_path, monkeypatch, filename, fault):
    flags = speed_fixture(tmp_path, monkeypatch, "deterministic100")
    path = tmp_path / filename
    value = eq.load(path)
    if fault == "missing":
        del value["timing_eq_profile"]
    elif fault == "none":
        value["timing_eq_profile"] = None
    elif fault == "wrong_flags":
        value["xla_flags"] = ""
    elif fault == "mixed":
        value.update(timing_eq_profile=profile(), xla_flags=None)
    else:
        value["schema"] = 1
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, KeyError)):
        eq.validate_speed(tmp_path, "on", "a" * 40, profile("deterministic100"), xla_flags=flags)


@pytest.mark.parametrize("name", ["normal", "deterministic100"])
def test_off_profile_still_forbids_speed_artifacts(tmp_path, name):
    flags = profile(name)["xla_flags"] or None
    assert eq.validate_speed(tmp_path, "off", "a" * 40, profile(name), xla_flags=flags) is None
    (tmp_path / "speed_start.json").write_text("{}\n")
    with pytest.raises(ValueError, match="off侧"):
        eq.validate_speed(tmp_path, "off", "a" * 40, profile(name), xla_flags=flags)


def completion_profile_fixture(tmp_path, monkeypatch, name, timing):
    import check_orig80k_completion as completion_tool
    monkeypatch.setattr(eq, "ROOT", tmp_path)
    # 底层恢复/采样已有独立测试；本夹具验证真实文件成员链中的档位绑定，不加载权重。
    monkeypatch.setattr(completion_tool, "validate_records", lambda *_, **__: None)
    monkeypatch.setattr(completion_tool, "validate_gpu_sampling", lambda *_: {"fixture": True})
    metadata = profile_metadata(name, timing)
    metadata["complete"] = {"fixture": "同一完整配置"}
    run = metadata["run_name"]
    records = tmp_path / "v1-store/bench/orig80k" / run
    checkpoint = tmp_path / "v1-store/train-runs/mme_vla_suite" / run / "19"
    (records / "final").mkdir(parents=True)
    (checkpoint / "params").mkdir(parents=True)
    driver = tmp_path / "v1-store/logs" / (run + ".driver.log")
    driver.parent.mkdir(parents=True)
    driver.write_text("\n".join(key + "=0" for key in
        ("EXIT_CODE", "TRAIN_PIPE_EXIT", "TEE_EXIT", "FOOTER_PRINTF_EXIT", "FOOTER_TEE_EXIT")) + "\n")
    launch = {**profile_launch(metadata), "run_name": run, "head": metadata["head"],
              "actual": {"complete": metadata["complete"], "jax_enable_x64": False}}
    launch["environment"]["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    scalars = {key: {"dec": 1., "hex": 1.0.hex(), "finite": True} for key in completion_tool.SCALARS}
    state = eq.full_state_record(synthetic_state())
    contents = {"launch.json": launch,
        "runtime.json": {"exp_name": run, "config_name": "mme_vla_suite", "seed": 42, "device_count": 4,
                         "batch_size": 64, "num_workers": 4, "fsdp_devices": 4, "cuda_visible_devices": "0,1,2,3"},
        "metrics.jsonl": {"step": 0, **scalars}, "gpu.csv": {}, "gpu.csv.err": {},
        "final/start.json": {"complete": metadata["complete"]},
        "final/final.json": {"ema_leaves": state["ema_leaves"],
                             "tail": [{"step": i, "scalars": scalars} for i in range(1, 20)]},
        "final/checkpoint_wait_done.json": {}}
    for relative, value in contents.items():
        (records / relative).write_text(json.dumps(value) + "\n")
    for path in (checkpoint / "_CHECKPOINT_METADATA", checkpoint / "params/_METADATA"):
        path.write_text("{}\n")
    evidence = [records / name for name in contents] + [driver, checkpoint / "_CHECKPOINT_METADATA",
                                                       checkpoint / "params/_METADATA"]
    completion = records / "completion.json"
    result = {**identity(run), "status": "PASS", "mode": "smoke", "state_step": 20,
              "final": 19, "checkpoints": [19], "gpu_sampling": {"fixture": True},
              "restored_leaves": state["ema_leaves"], "files": [eq.record_file(path) for path in evidence]}
    completion.write_text(json.dumps(result))
    return records, completion, metadata, state, evidence


@pytest.mark.parametrize("name", ["normal", "deterministic100"])
@pytest.mark.parametrize("timing", ["off", "on"])
def test_completion_keeps_real_membership_and_accepts_matching_record_profile(tmp_path, monkeypatch, name, timing):
    records, completion, metadata, state, _ = completion_profile_fixture(tmp_path, monkeypatch, name, timing)
    monkeypatch.setenv("XLA_FLAGS", "判定进程不作为历史运行证据")
    _, scalars = eq.validate_completion(records, completion, metadata, state)
    assert len(scalars) == 20


@pytest.mark.parametrize("fault", ["missing", "none", "mixed", "selector"])
def test_completion_profile_rejects_self_consistent_rehash_of_wrong_launch(tmp_path, monkeypatch, fault):
    records, completion, metadata, state, evidence = completion_profile_fixture(
        tmp_path, monkeypatch, "deterministic100", "on")
    path = records / "launch.json"
    launch = eq.load(path)
    if fault == "missing":
        del launch["timing_eq_profile"]
    elif fault == "none":
        launch["timing_eq_profile"] = None
    elif fault == "mixed":
        launch["timing_eq_profile"] = profile()
        launch["environment"].update(XLA_FLAGS=None, ORIG80K_TIMING_EQ_PROFILE=None)
    else:
        launch["environment"]["ORIG80K_TIMING_EQ_PROFILE"] = "normal"
    path.write_text(json.dumps(launch))
    receipt = eq.load(completion)
    receipt["files"] = [eq.record_file(item) for item in evidence]
    completion.write_text(json.dumps(receipt))
    with pytest.raises((ValueError, KeyError)):
        eq.validate_completion(records, completion, metadata, state)


def run_environment(tmp_path, monkeypatch, name, timing):
    monkeypatch.setattr(eq, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / ".venv"))
    monkeypatch.setenv("ORIG80K_MODE", "smoke")
    monkeypatch.setenv("ORIG80K_SMOKE_EQ_MODE", timing)
    monkeypatch.setenv("TRAIN_TIMING_STEPS", "0")
    monkeypatch.delenv("ORIG80K_TIMING_EQ_PROFILE", raising=False)
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    if name == "deterministic100":
        monkeypatch.setenv("ORIG80K_TIMING_EQ_PROFILE", name)
        monkeypatch.setenv("XLA_FLAGS", profile(name)["xla_flags"])
    records = tmp_path / "v1-store/bench/orig80k/fixture"
    monkeypatch.setenv("TRAIN_RECORD_DIR", str(records))
    monkeypatch.setenv("TRAIN_FINAL_RECORD_DIR", str(records / "final"))
    monkeypatch.setenv("MMEVLA_EXPECTED_TRAIN_HEAD", "a" * 40)
    return SimpleNamespace(timing=timing, records=str(records), train_args=["mme_vla_suite", "--num-train-steps", "20"])


@pytest.mark.parametrize(("selector", "flags"), [("", ""), ("normal", ""), ("bad", ""),
    (None, "--xla_gpu_deterministic_ops=true"), ("deterministic100", ""),
    ("deterministic100", "--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true")])
def test_run_rejects_unregistered_profile_before_creating_observer(tmp_path, monkeypatch, selector, flags):
    args = run_environment(tmp_path, monkeypatch, "normal", "off")
    if selector is not None:
        monkeypatch.setenv("ORIG80K_TIMING_EQ_PROFILE", selector)
    monkeypatch.setenv("XLA_FLAGS", flags)
    with pytest.raises(ValueError, match="ORIG80K_TIMING_EQ_PROFILE|实际XLA_FLAGS"):
        eq.run(args)
    assert not Path(args.records).exists()


@pytest.mark.parametrize("name", ["normal", "deterministic100"])
@pytest.mark.parametrize("timing", ["off", "on"])
@pytest.mark.parametrize("change_profile", [False, True])
def test_run_records_profile_preserves_dispatch_and_rejects_drift(tmp_path, monkeypatch, name, timing, change_profile):
    args = run_environment(tmp_path, monkeypatch, name, timing)
    records = Path(args.records)
    records.mkdir(parents=True)
    complete = {"fixture": "同一完整配置"}
    fingerprint = {"environment": {"XLA_FLAGS": os.environ.get("XLA_FLAGS")}}
    calls = []
    class FakeObserver:
        def __init__(self, directory):
            self.config = object()
            self.identity = {**identity("fixture"), "complete": complete}
            self.fingerprint_start = fingerprint
            self.fetched, self.used = [None] * 21, [None] * 20
            self.completed_calls, self.iterations, self.jit_matches, self.save_calls = 20, 1, 1, 1
            self.save_returned = True
            self.loader_contract = {"batch_size": 64, "workers": 4}
            self.overhead = {}
        def installed(self):
            return contextlib.nullcontext()
        def check_complete(self):
            calls.append("checked")
    monkeypatch.setattr(eq, "Observer", FakeObserver)
    monkeypatch.setattr(eq, "training_imports", lambda: (
        SimpleNamespace(complete_record=lambda _: complete), None,
        SimpleNamespace(_runtime_fingerprint=lambda *_: fingerprint)))
    monkeypatch.setattr(eq, "provenance", lambda *_: {"fixture": "相同来源"})
    monkeypatch.setattr(eq, "toolchain", lambda: {"fixture": "相同工具"})
    monkeypatch.setattr(eq.profile_contract(), "validate_argv", lambda *_: {"--exp-name": "fixture"})
    def dispatch(mode, argv, directory):
        calls.append((mode, argv, directory))
        if change_profile:
            if name == "normal":
                monkeypatch.setenv("ORIG80K_TIMING_EQ_PROFILE", "deterministic100")
            else:
                monkeypatch.delenv("ORIG80K_TIMING_EQ_PROFILE")
    monkeypatch.setattr(eq, "execute_entry", dispatch)
    if change_profile:
        with pytest.raises(ValueError, match="实际XLA_FLAGS"):
            eq.run(args)
    else:
        eq.run(args)
    metadata = eq.load(records / "timing_eq/metadata.json")
    assert metadata["schema"] == eq.SCHEMA == 2
    assert metadata["status"] == ("FAIL" if change_profile else "PASS")
    assert metadata["timing_eq_profile"] == profile(name)
    assert calls == [(timing, args.train_args, records), "checked"]
    assert calls[0][1] is args.train_args
    assert eq.load(records / "timing_eq/manifest.json")["schema"] == 2


def runner_text():
    return (Path(__file__).resolve().parents[3] / "scripts/training/prod/run_orig80k.sh").read_text()


@pytest.mark.parametrize(("mode", "value", "expected"), [("prod", None, 0), ("perf", None, 0), ("smoke", None, 0),
    ("smoke", "off", 0), ("smoke", "on", 0), ("smoke", "", 2), ("smoke", "bad", 2),
    ("prod", "off", 2), ("perf", "on", 2), ("prod", "", 2)])
def test_runner_explicit_smoke_only_guard(mode, value, expected):
    prefix = runner_text().split('LOG="$MAIN/v1-store/logs/$RUN.driver.log"')[0]
    env = dict(os.environ, TRAIN_HEAD="a" * 40, HISTORY_CONFIG_SHA256="b" * 64, NORM_STATS_SHA256="c" * 64)
    env.pop("ORIG80K_SMOKE_EQ_MODE", None)
    if value is not None:
        env["ORIG80K_SMOKE_EQ_MODE"] = value
    result = subprocess.run(["bash", "-c", prefix, "--", mode, "fixture", "0,1,2,3", "/lib", "/assets"],
                            env=env, capture_output=True, text=True, check=False)
    assert result.returncode == expected


@pytest.mark.parametrize(("body_rc", "fail_tee", "expected"), [(0, 0, 0), (7, 0, 7), (0, 1, 29), (0, 2, 29)])
def test_runner_footer_cannot_log_success_before_tee_failure(tmp_path, body_rc, fail_tee, expected):
    footer = 'body 2>&1 | tee "$LOG"' + runner_text().split('body 2>&1 | tee "$LOG"', 1)[1]
    prefix = r"""
set -o pipefail
body() { printf 'BODY_DONE\n'; return "$BODY_RC"; }
tee() {
  n=0
  if [ -f "$COUNT" ]; then n=$(< "$COUNT"); fi
  n=$((n+1))
  printf '%s' "$n" > "$COUNT"
  /usr/bin/tee "$@"
  code=$?
  if [ "$n" = "$FAIL_TEE" ]; then return 29; fi
  return "$code"
}
"""
    log = tmp_path / "driver.log"
    env = dict(os.environ, LOG=str(log), COUNT=str(tmp_path / "count"), BODY_RC=str(body_rc), FAIL_TEE=str(fail_tee))
    result = subprocess.run(["bash", "-c", prefix + footer], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == expected
    exits = [line for line in log.read_text().splitlines() if line.startswith("EXIT_CODE=")]
    assert exits == [f"EXIT_CODE={expected}"]
    assert ("FOOTER_TEE_EXIT=29" in log.read_text()) == (fail_tee == 2)
