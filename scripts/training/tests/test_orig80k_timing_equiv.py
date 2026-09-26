"""测速包装20步对照的候选CPU测试；冻结期只准备，解冻应用后执行。"""

from __future__ import annotations

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
    return {"schema": 1, "loop_step": 19, **identity(name), **eq.full_state_record(synthetic_state())}


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
    return {"metadata": {**identity(name), "records": "/records/" + name,
            "fingerprint": {"same": "environment"}, "toolchain": {"same": "files"},
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
