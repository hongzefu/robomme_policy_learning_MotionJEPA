"""四对冒烟专用完整模型阶段：原初始化、一步训练、原保存及严格权重回读。"""
from __future__ import annotations

import dataclasses
import functools
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

from check_pair4_contract import ENCODER_SHA
from check_pair4_contract import REPO
from check_pair4_contract import code_identity
from check_pair4_contract import gate
from check_pair4_contract import require
from check_pair4_contract import saved_artifact_manifest
from check_pair4_contract import scope_checks
from check_pair4_contract import sha
from check_pair4_contract import write_json
import numpy as np

BASE_CONFIG = "mme_vla_suite_b128_80k"
RUN_NAME = "paired4-epoch6-original-smoke-20261020-a"
NORM_REL = "v1-store/train-assets/mme_vla_suite/16task-mixed8-200-3200ep-e3c7564f/robomme/norm_stats.json"


class ResourceProbe:
    """记录JAX分配器真实峰值与宿主峰值，不轮询驱动或其他进程。"""
    def __init__(self):
        self.start = time.perf_counter()

    def finish(self):
        result = {"elapsed_seconds": time.perf_counter() - self.start,
                  "host_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                  "gpu_peak_scope": "JAX分配器的peak_bytes_in_use；不是驱动或全进程显存峰值"}
        try:
            jax = sys.modules.get("jax")
            if jax is None:
                return result
            result["devices"] = [{"kind": device.device_kind, "platform": device.platform,
                "id": device.id, "memory_stats": device.memory_stats()} for device in jax.local_devices()]
        except Exception as error:
            result["jax_memory_error"] = str(error)
        return result


def _flatten(tree):
    import jax
    return {jax.tree_util.keystr(path): value for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]}


def _array_record(array):
    value = np.asarray(array)
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def _compare_trees(reference, candidate):
    import jax
    left, right = _flatten(reference), _flatten(candidate)
    require(left.keys() == right.keys(), f"参数路径不完整：缺{sorted(left.keys() - right.keys())}，多{sorted(right.keys() - left.keys())}")
    records = {}
    for path, value in left.items():
        a, b = np.asarray(jax.device_get(value)), np.asarray(jax.device_get(right[path]))
        require(a.shape == b.shape and a.dtype == b.dtype, f"参数形状或dtype改变：{path}")
        require(a.tobytes() == b.tobytes(), f"参数未逐位回读：{path}")
        records[path] = _array_record(a)
    return records


def _read_contract(run_root, fixture_root):
    """从落盘记录重建完整模型配置；逐项核验来源，不依赖内存配置。"""
    import config_record
    from omegaconf import OmegaConf
    from pair4_fixture import load_fixture

    from mme_vla_suite.datastore import StoreMeta
    from mme_vla_suite.datastore import motion_store as ms
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
    from mme_vla_suite.training.dataloader import _motion_gates

    load_fixture(fixture_root.parent)

    record_path = run_root / "config_record.json"
    bundle = json.loads(record_path.read_text())
    record = bundle["complete_record"]
    contract = bundle["execution"]
    require(record["config_record_version"] == 2 and contract["base_config"] == BASE_CONFIG, "完整配置版本或基础配置不符")
    raw = (run_root / "history_config.resolved.yaml").read_bytes()
    resolved_sha = hashlib.sha256(raw).hexdigest()
    require((run_root / "history_config.resolved.sha256").read_text().strip() == resolved_sha, "resolved YAML摘要不符")
    require(contract["resolved_sha256"] == resolved_sha, "完整记录未绑定实际resolved YAML")
    history = OmegaConf.create(raw.decode())
    actual_values = OmegaConf.to_container(history, resolve=True)
    # 原complete_record保留源YAML；执行快照只把store_path解析为该夹具绝对路径。
    source_values = json.loads(json.dumps(record["history_values"]))
    source_values["motion"]["store_path"] = str(fixture_root / "motion")
    require(source_values == actual_values, "源配置与执行快照除store绝对路径外存在差异")
    expected_motion = {"source_shape": [8, 256], "readout": "adjacent_pair_v1", "num_tokens": 4,
                       "dim": 512, "budget": 640, "pos_dim": 256, "checkpoint_sha256": ENCODER_SHA}
    for name, value in expected_motion.items():
        require(actual_values["motion"].get(name) == value, f"回读motion.{name}合同不符")
    require(actual_values["budget"] == 512 and actual_values["memory_token_dim"] == 1024
            and actual_values["integration_type"] == "modulation", "回读frame/调制合同不符")
    provenance = json.loads((run_root / "motion_provenance.json").read_text())
    require(sha(run_root / "motion_provenance.json") == contract["motion_provenance_sha256"], "provenance文件被替换")
    require(provenance["motion_enabled"] is True and provenance["resolved_sha256"] == resolved_sha, "provenance启用态或配置摘要不符")
    label = (run_root / "history_config.txt").read_text()
    require(label == provenance["history_config_name"] == record["train_config"]["fields"]["model"]["fields"]["history_config"], "history来源标签不符")
    frames = fixture_root / "framesamp-8x8"
    motion = fixture_root / "motion"
    require(Path(provenance["framesamp_root"]).resolve() == frames and Path(provenance["motion_root"]).resolve() == motion, "来源目录不符")
    fm, mm = StoreMeta.load(frames), ms.MotionMeta.load(motion)
    require(fm.manifest_sha256 == mm.manifest_sha256 == provenance["framesamp_manifest_sha256"] == provenance["motion_manifest_sha256"], "frame/motion清单身份不一致")
    checks = {"framesamp_store_meta_sha256": frames / "meta/store_meta.json",
              "motion_store_meta_sha256": motion / "meta/store_meta.json",
              "motion_index_sha256": motion / ms.INDEX_RELPATH,
              "motion_table_sha256": motion / ms.MOTION_TABLE_RELPATH}
    for name, path in checks.items():
        require(sha(path) == provenance[name], f"来源{ name }摘要不符")
    require(provenance["vae"] == mm.provenance["vae"] and provenance["encoder"] == mm.provenance["encoder"], "VAE或encoder身份不符")
    require(provenance["encoder"]["checkpoint_sha256"] == ENCODER_SHA, "encoder不是批准的epoch6")
    registered = (REPO / "v1-store/external/motionjepa" / provenance["encoder"]["run_name"]
                  / provenance["encoder"]["checkpoint_name"])
    require(sha(registered) == ENCODER_SHA, "注册encoder权重实际字节不符")
    for name, path in (("latents_metadata_sha256", Path(mm.provenance["latents_root"]) / "metadata.json"),
                       ("input_manifest_sha256", fixture_root / "meta/input_manifest.json")):
        require(sha(path) == mm.provenance[name], f"来源{name}实际字节不符")
    require(Path(_motion_gates(history, fm)).resolve() == motion, "原loader来源闸未返回当前表")
    require(sha(fixture_root / "fixture_manifest.json") == contract["fixture_manifest_sha256"], "派生输入来源记录被替换")
    for asset in record["assets"].values():
        require(sha(asset["path"]) == asset["sha256"] and Path(asset["path"]).stat().st_size == asset["bytes"], "基础模型/归一化/tokenizer资产身份不符")
    model_record = record["train_config"]["fields"]["model"]
    require(model_record["type"] == "mme_vla_suite.models.integration.history_pi0.HistoryPi0Config", "模型类型不符")
    fields = dict(model_record["fields"])
    require(set(fields) == {f.name for f in dataclasses.fields(HistoryPi0Config)}, "完整模型字段存在缺失或额外字段")
    expected_model = {"paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m", "pi05": True,
                      "action_horizon": 20, "use_history": True, "discrete_state_input": False, "max_token_len": 64}
    for name, value in expected_model.items():
        require(fields[name] == value, f"基础模型字段{name}被改变")
    require(all(value is None or type(value) in (str, int, float, bool) for value in fields.values()), "模型字段含非标量，拒绝隐式反序列化")
    fields["history_config"] = history
    model_config = HistoryPi0Config(**fields)
    # 重新读配置定义只用于比对原optimizer/LR；重建模型来自上述落盘字段。
    from mme_vla_suite.training.config import get_config
    base = get_config(BASE_CONFIG)
    train_fields = record["train_config"]["fields"]
    require(model_record == config_record.encode(dataclasses.replace(base.model, history_config=label)),
            "完整模型字段与记录的基础配置不符")
    for name in ("optimizer", "lr_schedule", "freeze_filter", "ema_decay"):
        require(train_fields[name] == config_record.encode(getattr(base, name)), f"原训练{name}被改变")
    require(train_fields["batch_size"] == 1 and train_fields["num_workers"] == 0
            and train_fields["seed"] == 42 and train_fields["fsdp_devices"] == 1, "实际训练固定参数不符")
    return model_config, {"config_record_sha256": sha(record_path), "resolved_sha256": resolved_sha,
                          "provenance_sha256": sha(run_root / "motion_provenance.json"),
                          "manifest_sha256": fm.manifest_sha256, "source_run": actual_values["motion"]["source_run"]}


def _batch_check(observation, actions, fixture_root):
    """独立从raw表和帧表核固定idx32；不借用loader的配对或排序实现。"""
    from mme_vla_suite.datastore import FrameSampStore
    from mme_vla_suite.datastore import StoreMeta
    from mme_vla_suite.datastore import motion_store as ms
    from mme_vla_suite.shared.sampling import MEM_ORDER_SENTINEL
    mm = ms.MotionMeta.load(fixture_root / "motion")
    raw = np.fromfile(fixture_root / "motion" / ms.MOTION_TABLE_RELPATH, dtype="<f4").reshape(1, 8, 256)
    emb, pos, mask, order = (np.asarray(value) for value in
        (observation.motion_emb, observation.motion_pos, observation.motion_mask, observation.mem_order))
    require(emb.shape == (1, 640, 512) and pos.shape == (1, 640, 256), "idx32 motion形状不符")
    require(mask.shape == (1, 640) and mask.sum() == 4 and mask[0, :4].all() and not mask[0, 4:].any(), "idx32必须恰有一窗四个有效pair")
    for pair in range(4):
        require(emb[0, pair, :256].tobytes() == raw[0, 2 * pair].tobytes(), "前半raw槽未逐位保留")
        require(emb[0, pair, 256:].tobytes() == raw[0, 2 * pair + 1].tobytes(), "后半raw槽未逐位保留")
    require(np.all(emb[0, 4:] == 0) and np.all(pos[0, 4:] == 0), "idx32 motion填充非零")
    fm = StoreMeta.load(fixture_root / "framesamp-8x8")
    store = FrameSampStore(fixture_root / "framesamp-8x8", meta=fm, verify_level="full")
    time_pe = store.pos_rows(np.array([0]))[0, 0, :256]
    for pair in range(4):
        require(pos[0, pair].tobytes() == time_pe.tobytes(), "窗口PE未逐位repeat4")
    if hasattr(store, "close"):
        store.close()
    # idx32按原even_sampling采8帧；这里直接独立构造固定期待身份。
    frame_times = [0, 4, 9, 13, 18, 22, 27, 32]
    identities = [(frame_times[i // 64], 0, i) for i in range(512)]
    identities += [(0 if i < 4 else MEM_ORDER_SENTINEL, 1, 512 + i) for i in range(640)]
    expected = np.array([row[2] for row in sorted(identities)], np.int32)
    np.testing.assert_array_equal(order, expected[None])
    require(np.asarray(observation.static_image_emb).shape == (1, 512, 2048), "idx32 frame容量不符")
    require(np.asarray(actions).shape == (1, 20, 32), "原模型动作输入应为B1×20×32")
    require(mm.num_rows == 1, "真实motion夹具必须恰好一行")
    return {"sample_idx": 32, "batch_size": 1, "valid_motion_tokens": 4,
            "frame_times": frame_times, "motion_window_starts": [0],
            "arrays": {name: _array_record(value) for name, value in
                {"motion_emb": emb, "motion_pos": pos, "motion_mask": mask, "mem_order": order,
                 "static_image_emb": observation.static_image_emb, "actions": actions}.items()}}


def _make_diagnostics(config, model_def, train_rng):
    """和原训练步骤同损失、同随机键，只回传梯度摘要。"""
    from flax import nnx
    import jax
    import jax.numpy as jnp
    import optax

    @jax.jit
    def diagnostics(params, obs, act, step):
        model = nnx.merge(model_def, params)
        model.train()
        key = jax.random.fold_in(train_rng, step)
        def loss_fn(m, o, a, emb, pos):
            changed = dataclasses.replace(o, motion_emb=emb, motion_pos=pos)
            return jnp.mean(m.compute_loss(key, changed, a, train=True))
        loss, (grads, emb_grad, pos_grad) = nnx.value_and_grad(loss_fn,
            argnums=(nnx.DiffState(0, config.trainable_filter), 3, 4))(model, obs, act, obs.motion_emb, obs.motion_pos)
        gradient_arrays = jax.tree_util.tree_leaves(grads)
        valid = obs.motion_mask[..., None]
        kernel = grads.mem_encoder.motion_encoder_static.kernel.value
        return {"loss": loss, "all_parameter_gradients_finite": jnp.all(jnp.stack([jnp.isfinite(value).all() for value in gradient_arrays])),
            "motion_input_gradient_finite": jnp.isfinite(emb_grad).all() & jnp.isfinite(pos_grad).all(),
            "valid_motion_emb_grad_norm": jnp.linalg.norm(jnp.where(valid, emb_grad, 0)),
            "valid_motion_pos_grad_norm": jnp.linalg.norm(jnp.where(valid, pos_grad, 0)),
            "padding_motion_emb_grad_norm": jnp.linalg.norm(jnp.where(valid, 0, emb_grad)),
            "projection_content_grad_norm": jnp.linalg.norm(kernel[:512]),
            "projection_position_grad_norm": jnp.linalg.norm(kernel[512:]),
            "position_projection_grad_norm": optax.global_norm(grads.mem_encoder.motion_pos_proj)}
    return diagnostics


def model_suite(root, history_name):
    probe = ResourceProbe()
    report = {"status": "NOT_RUN", "gates": {"REAL_FORWARD": {"status": "NOT_RUN"}, "SAVE_LOAD": {"status": "NOT_RUN"}},
              "stage": "preflight", "root": str(root), "code_sha256": code_identity(), "timings": {}}
    manager = None
    def persist(stage):
        report["stage"] = stage
        report["resources"] = probe.finish()
        write_json(root / "reports/model.json", report)
        print(f"PAIR4_MODEL_STAGE={stage}", flush=True)
    try:
        require(Path.cwd().resolve() == REPO, "模型阶段须从当前仓库根执行，避免源YAML路径错位")
        models_root = REPO / "v1-store/models"
        if os.environ.get("OPENPI_DATA_HOME"):
            require(Path(os.environ["OPENPI_DATA_HOME"]).resolve() == models_root, "OPENPI_DATA_HOME必须指向本仓现有models资产")
        os.environ["OPENPI_DATA_HOME"] = str(models_root)
        require((models_root / "openpi-assets/checkpoints/pi05_base/params/commit_success.txt").is_file(), "本地pi05_base资产缺失")
        require((REPO / NORM_REL).is_file(), "mixed8专用归一化统计缺失")
        fixture_root = root / "fixture"
        motion_override = os.environ.get("MMEVLA_MOTION_STORE")
        require(not motion_override or Path(motion_override).resolve() == fixture_root / "motion", "MMEVLA_MOTION_STORE指向其他数据")
        report["input_report_sha256"] = {}
        for name in ("cpu", "prepare-fixture", "source-smoke", "finalize-fixture"):
            path = root / "reports" / f"{name}.json"
            previous = json.loads(path.read_text())
            require(previous.get("status") == "PASS", f"前置{name}尚未通过")
            if name != "source-smoke":
                require(previous.get("code_sha256") == report["code_sha256"], f"前置{name}不是当前代码版本")
            report["input_report_sha256"][str(path.relative_to(root))] = sha(path)
        scope_checks()
        sys.path.insert(0, str(REPO / "scripts/training"))
        import config_record
        from flax import nnx
        import jax
        import train

        from mme_vla_suite.models.config.utils import get_history_config
        from mme_vla_suite.training.config import AssetsConfig
        from mme_vla_suite.training.config import get_config
        from mme_vla_suite.training.dataloader import create_data_loader
        from openpi.models import model as model_lib
        from openpi.training import checkpoints
        from openpi.training import sharding

        require(len(jax.local_devices()) == 1 and jax.local_devices()[0].platform == "gpu", "完整模型阶段必须只可见一张GPU")
        base = get_config(BASE_CONFIG)
        config = dataclasses.replace(base, batch_size=1, num_workers=0, fsdp_devices=1,
            seed=42, exp_name=RUN_NAME, checkpoint_base_dir=str(root / "policy"), wandb_enabled=False,
            model=dataclasses.replace(base.model, history_config=history_name),
            data=dataclasses.replace(base.data, assets=AssetsConfig(assets_dir=str((REPO / NORM_REL).parent.parent), asset_id="robomme")))
        require((config.model.paligemma_variant, config.model.action_expert_variant, config.model.pi05,
                 config.model.action_horizon) == ("gemma_2b", "gemma_300m", True, 20), "禁止降低完整模型")
        require(config.optimizer == base.optimizer and config.lr_schedule == base.lr_schedule, "禁止修改optimizer/LR")
        history = get_history_config(history_name)
        history.motion.store_path = str(fixture_root / "motion")
        manager, resuming = checkpoints.initialize_checkpoint_dir(config.checkpoint_dir, keep_period=config.keep_period,
                                                                  overwrite=False, resume=False)
        require(not resuming, "本次必须从基础模型新建训练状态")
        train.init_history_config(config, history, framesamp_root=str(fixture_root / "framesamp-8x8"))
        run_root = config.checkpoint_dir
        complete = config_record.complete_record(config)
        bundle = {"complete_record": complete, "execution": {"base_config": BASE_CONFIG, "updates": 1,
            "fixture_manifest_sha256": sha(fixture_root / "fixture_manifest.json"),
            "resolved_sha256": sha(run_root / "history_config.resolved.yaml"),
            "motion_provenance_sha256": sha(run_root / "motion_provenance.json")}}
        write_json(run_root / "config_record.json", bundle)
        config = dataclasses.replace(config, model=dataclasses.replace(config.model, history_config=history))
        disk_config, contract = _read_contract(run_root, fixture_root)
        data = config.data.create(config.assets_dirs, config.model)
        # 原loader的全部33批依序经过转换与归一化；不直接索引构造模型输入。
        loader = create_data_loader(str(fixture_root / "framesamp-8x8"), data, history,
            action_horizon=20, batch_size=1, shuffle=False, num_batches=33, num_workers=0, seed=42, skip_norm_stats=False)
        selected = None
        for index, batch in enumerate(loader):
            if index == 32:
                selected = batch
                break
        require(selected is not None, "真实loader未提供第33批")
        observation, actions = selected
        report["batch"] = _batch_check(observation, actions, fixture_root)
        report["configuration"] = {"base_config": BASE_CONFIG, "run_root": str(run_root), "initial_lr": float(config.lr_schedule.create()(0)),
                                   "norm_stats": complete["assets"]["norm_stats"], "contract": contract}
        persist("initialize")
        start = time.perf_counter()
        train_rng, init_rng = jax.random.split(jax.random.key(42))
        mesh = sharding.make_mesh(1)
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        with sharding.set_mesh(mesh):
            state, state_sharding = train.init_train_state(config, init_rng, mesh, resume=False)
            jax.block_until_ready(state)
        report["timings"]["initialize"] = time.perf_counter() - start
        before = {path: np.asarray(jax.device_get(value)).copy() for path, value in _flatten(state.params.to_pure_dict()).items() if "motion_" in path}
        require(before and all(value.dtype == np.float32 for value in before.values()), "motion参数必须为可训练fp32")
        model_def = state.model_def

        diagnostics = _make_diagnostics(config, model_def, train_rng)

        persist("gradient_diagnostics")
        start = time.perf_counter()
        with sharding.set_mesh(mesh):
            diagnostics_result = jax.device_get(diagnostics(state.params, observation, actions, state.step))
        diagnostic = {name: np.asarray(value).item() for name, value in diagnostics_result.items()}
        require(diagnostic["all_parameter_gradients_finite"] and diagnostic["motion_input_gradient_finite"], "出现非有限梯度")
        require(all(np.isfinite(value) for value in diagnostic.values()), "梯度诊断出现非有限值")
        for name in ("valid_motion_emb_grad_norm", "valid_motion_pos_grad_norm", "projection_content_grad_norm",
                     "projection_position_grad_norm", "position_projection_grad_norm"):
            require(diagnostic[name] > 0, f"有效motion或投影梯度为零：{name}")
        report["gradient_diagnostics"] = diagnostic
        report["timings"]["gradient_diagnostics"] = time.perf_counter() - start
        diagnostics.clear_cache()
        del diagnostics
        gc.collect()
        persist("original_train_step")
        start = time.perf_counter()
        step_fn = jax.jit(functools.partial(train.train_step, config),
            in_shardings=(replicated, state_sharding, data_sharding), out_shardings=(state_sharding, replicated), donate_argnums=(1,))
        with sharding.set_mesh(mesh):
            state, metrics = step_fn(train_rng, state, selected)
            jax.block_until_ready((state, metrics))
        metric_values = {name: np.asarray(value).item() for name, value in jax.device_get(metrics).items()}
        require(all(np.isfinite(value) for value in metric_values.values()), "原train_step损失或梯度非有限")
        require(int(state.step) == 1 and metric_values["grad_norm"] > 0, "原train_step未完成一次有效更新")
        after = _flatten(state.params.to_pure_dict())
        updates = {path: int(np.count_nonzero(value.view(np.uint32) != np.asarray(jax.device_get(after[path])).view(np.uint32))) for path, value in before.items()}
        del after
        require(sum(updates.values()) > 0, "没有fp32 motion参数真实改变")
        report["timings"]["original_train_step"] = time.perf_counter() - start
        report["gates"]["REAL_FORWARD"] = gate({"metrics": metric_values, "gradients": diagnostic,
            "updated_fp32_motion_values": updates, "step": 1, "full_model": True, "memory_tokens": 1152})
        step_fn.clear_cache()
        del step_fn, before
        gc.collect()
        persist("save_weights")
        start = time.perf_counter()
        exported = state.ema_params if state.ema_params is not None else state.params
        export_kind = "ema" if state.ema_params is not None else "live"
        # 先捕获实际导出分支的逐叶摘要；保存器和本检查不改变任何dtype。
        export_pure = exported.to_pure_dict()
        reference_records = {path: _array_record(jax.device_get(value)) for path, value in _flatten(export_pure).items()}
        write_json(run_root / "exported_parameter_reference.json", {"kind": export_kind, "leaves": reference_records})
        checkpoints.save_state(manager, state, loader, int(state.step))
        manager.wait_until_finished()
        step_dir = run_root / str(int(state.step))
        report["timings"]["save_weights"] = time.perf_counter() - start
        # 显式释放optimizer及训练live状态，回读阶段只保留真正导出的参数。
        del state
        gc.collect()
        persist("strict_restore")
        start = time.perf_counter()
        disk_config, contract = _read_contract(run_root, fixture_root)
        saved = model_lib.restore_params(step_dir / "params", restore_type=np.ndarray, dtype=None)
        compared = _compare_trees(export_pure, saved)
        require(compared == reference_records, "实际导出分支的参数摘要不符")
        loaded = disk_config.load(saved, remove_extra_params=False)
        loaded.eval()
        _compare_trees(saved, nnx.state(loaded).to_pure_dict())
        # 两侧均用同一落盘配置重建；参考模型只装实际导出的同dtype参数。
        reference = disk_config.load(export_pure, remove_extra_params=False)
        reference.eval()
        reference_def, reference_state = nnx.split(reference)
        loaded_def, loaded_state = nnx.split(loaded)
        require(reference_def == loaded_def, "参考与回读模型结构不一致")
        del reference, loaded, export_pure, exported, saved
        gc.collect()
        fixed_rng = jax.random.fold_in(train_rng, 20261020)
        @jax.jit
        def fixed_loss(params, obs, act):
            model = nnx.merge(reference_def, params)
            return model.compute_loss(fixed_rng, obs, act, train=False)
        with sharding.set_mesh(mesh):
            reference_loss = np.asarray(jax.device_get(fixed_loss(reference_state, observation, actions)))
            loaded_loss = np.asarray(jax.device_get(fixed_loss(loaded_state, observation, actions)))
        require(np.isfinite(reference_loss).all() and np.isfinite(loaded_loss).all(), "回读损失出现非有限值")
        np.testing.assert_array_equal(loaded_loss, reference_loss)
        maximum = float(np.max(np.abs(loaded_loss - reference_loss)))
        require(maximum == 0.0, "固定输入损失最大差异非零")
        report["timings"]["strict_restore_and_loss"] = time.perf_counter() - start
        report["gates"]["SAVE_LOAD"] = gate({"export_kind": export_kind, "save_step": 1, "parameter_leaves": len(compared),
            "parameter_reference_sha256": sha(run_root / "exported_parameter_reference.json"), "contract": contract,
            "loss_max_abs": maximum, "reference_loss": _array_record(reference_loss), "loaded_loss": _array_record(loaded_loss),
            "backend": jax.default_backend(), "fixed_rng": np.asarray(jax.random.key_data(fixed_rng)).tolist(),
            "optimizer_restored": False})
        report["saved_artifacts"] = saved_artifact_manifest(run_root)
        report["status"] = "PASS"
        persist("complete")
    except (Exception, SystemExit) as error:
        report["status"] = "FAIL"
        report["failure"] = {"stage": report["stage"], "type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()}
        for value in report["gates"].values():
            if value["status"] == "NOT_RUN":
                value["reason"] = f"在{report['stage']}阶段终止"
        traceback.print_exc()
    finally:
        if manager is not None:
            try:
                manager.wait_until_finished()
                manager.close()
            except Exception as error:
                report["status"] = "FAIL"
                report["checkpoint_close_error"] = str(error)
        report["resources"] = probe.finish()
        write_json(root / "reports/model.json", report)
    return report
