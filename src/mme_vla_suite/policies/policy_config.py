import logging
import pathlib
from typing import Any

import jax.numpy as jnp

import dataclasses

import openpi.models.model as _model
from openpi.training import checkpoints as _checkpoints
import openpi.transforms as transforms

import mme_vla_suite.policies.policy as _policy
import mme_vla_suite.training.config as _config



def _params_have_motion(params) -> bool:
    import jax
    paths = ["/".join(str(getattr(k, "key", getattr(k, "name", k))) for k in kp)
             for kp, _ in jax.tree_util.tree_flatten_with_path(params)[0]]
    return any("motion_pos_proj" in p or "motion_encoder_static" in p for p in paths)


def _load_resolved_snapshot(run_root: pathlib.Path):
    """读 run 内 history_config.resolved.yaml，核 sha256 与 motion_provenance.json；返回 (DictConfig, motion_enabled)。"""
    import hashlib
    import json
    from omegaconf import OmegaConf
    raw = (run_root / "history_config.resolved.yaml").read_bytes()
    want = (run_root / "history_config.resolved.sha256").read_text().split()[0]
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        raise ValueError(f"history_config.resolved.yaml sha256 {got[:16]}… != 记录 {want[:16]}…: {run_root}")
    prov_path = run_root / "motion_provenance.json"
    if not prov_path.is_file():
        raise ValueError(f"缺 motion_provenance.json: {run_root}")
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    if prov.get("resolved_sha256") != got:
        raise ValueError("motion_provenance.json 的 resolved_sha256 与快照不符")
    cfg = OmegaConf.create(raw.decode("utf-8"))
    mcfg = cfg.get("motion", None)
    enabled = bool(mcfg is not None and mcfg.get("enabled", False))
    if bool(prov.get("motion_enabled")) != enabled:
        raise ValueError("motion_provenance.json 的 motion_enabled 与快照不符")
    return cfg, enabled


def _assert_param_tree_exact(model, params) -> None:
    """checkpoint 参数树与模型定义逐路径相等：missing / extra 任一非空即 raise（motion checkpoint 严格恢复）。"""
    import jax
    from flax import nnx

    def paths_of(tree):
        return {"/".join(str(getattr(k, "key", getattr(k, "name", k))) for k in kp)
                for kp, _ in jax.tree_util.tree_flatten_with_path(tree)[0]}
    model_paths = paths_of(nnx.state(model, nnx.Param).to_pure_dict())
    ckpt_paths = paths_of(params)
    missing = sorted(model_paths - ckpt_paths)
    extra = sorted(ckpt_paths - model_paths)
    if missing or extra:
        raise ValueError(f"严格恢复失败：模型有而 checkpoint 缺 {missing[:5]}…（{len(missing)}），checkpoint 多出 {extra[:5]}…（{len(extra)}）")


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    seed: int = 42,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.MME_VLA_Policy:
    
    repack_transforms = repack_transforms or transforms.Group()
    
    logging.info(f"Checking history config")
    run_root = pathlib.Path(checkpoint_dir).parent
    history_config = None
    history_config_path = run_root / "history_config.txt"
    if history_config_path.exists():
        with open(history_config_path, "r") as f:
            history_config = f.read()

    # ── run 内配置快照（motion-memory-plan.md 2.1，红线 16）──────────────────────────────
    # 带 history_config.resolved.yaml 的新 run：只从快照恢复（先核 sha 与 motion_provenance.json），
    # 并要求 checkpoint 参数树与模型定义 missing / extra 集合均为空（禁止 remove_extra_params 静默裁掉）。
    # 旧的不含快照的非 motion checkpoint 保留 history_config.txt 兼容路径；任何带 motion 参数的 checkpoint
    # 缺快照 / provenance，或快照声称关闭但 checkpoint 带 motion 参数，均拒绝加载。
    resolved_path = run_root / "history_config.resolved.yaml"
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    ckpt_has_motion = _params_have_motion(params)
    if resolved_path.exists():
        resolved_history_config, snapshot_enabled = _load_resolved_snapshot(run_root)
        if snapshot_enabled != ckpt_has_motion:
            raise ValueError(
                f"run 快照 motion.enabled={snapshot_enabled} 与 checkpoint 参数树含 motion 参数={ckpt_has_motion} 不符: {run_root}")
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, history_config=resolved_history_config, use_history=True))
        strict = True
    else:
        if ckpt_has_motion:
            raise ValueError(f"checkpoint 含 motion 参数但 run 缺 history_config.resolved.yaml / motion_provenance.json: {run_root}")
        if train_config.model.history_config != history_config:
            print(f" == You are using {train_config.model.history_config}, changing to {history_config} ==")
            train_config = dataclasses.replace(
                train_config,
                model=dataclasses.replace(train_config.model, history_config=history_config, use_history=history_config is not None)
            )
        strict = False

    logging.info("Loading model...")
    if strict:
        model = train_config.model.load(params, remove_extra_params=False)
        _assert_param_tree_exact(model, params)
    else:
        model = train_config.model.load(params)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    print("Training config: ", train_config)
    print("Data config: ", data_config)

    return _policy.MME_VLA_Policy(
        model,
        seed=seed,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        norm_stats=norm_stats,
        use_quantiles=data_config.use_quantile_norm
    )
