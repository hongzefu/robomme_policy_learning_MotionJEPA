import dataclasses
import enum
import logging
import socket
import os
import uuid
from typing import Literal

import tyro

from mme_vla_suite.policies import policy as _policy
from mme_vla_suite.policies import policy_config as _policy_config
from mme_vla_suite.serving import websocket_policy_server
from mme_vla_suite.training import config as _config

from pathlib import Path


class EnvMode(enum.Enum):
    """Supported environments."""

    HISTORY_BENCH = "history_bench"

@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str
    
    def __post_init__(self):
        self.dir = Path(self.dir)


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.HISTORY_BENCH

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    seed: int = 42
    # 设备与解释器覆盖只作用于恢复后的内存配置。
    motion_online_gpu: str | None = None
    stage2_python: str | None = None
    eval_target_contract: Path | None = None
    run_identity_sha256: str | None = None
    server_nonce: str | None = None
    eval_purpose: Literal["production", "validation"] = "production"
    checkpoint_step: int | None = None
    worker_uuid: str | None = None
    model_binding: Path | None = None

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.HISTORY_BENCH: Checkpoint(
        config="mme_vla_suite",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    )
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None, seed: int = 42,
                          motion_online_gpu=None, stage2_python=None) -> _policy.MME_VLA_Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt,
            seed=seed, motion_online_gpu=motion_online_gpu, stage2_python=stage2_python,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.MME_VLA_Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt,
                seed=args.seed,
                motion_online_gpu=args.motion_online_gpu, stage2_python=args.stage2_python,
                eval_target_contract=args.eval_target_contract,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt, seed=args.seed,
                                         motion_online_gpu=args.motion_online_gpu, stage2_python=args.stage2_python)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = dict(policy.metadata)
    policy_metadata.update(model_seed=policy._seed, server_pid=os.getpid(),
                           cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                           slurm_job_id=os.environ.get("SLURM_JOB_ID"), slurm_step_id=os.environ.get("SLURM_STEP_ID"))
    if args.run_identity_sha256 is not None:
        policy_metadata.update(run_identity_sha256=args.run_identity_sha256,
                               server_nonce=args.server_nonce or uuid.uuid4().hex,
                               eval_purpose=args.eval_purpose, checkpoint_step=args.checkpoint_step,
                               worker_uuid=args.worker_uuid)
    client = getattr(policy, "_motion_client", None)
    if client is not None:
        policy_metadata["motion_runtime"] = client.provenance

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    try:
        server.serve_forever()
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
