from collections.abc import Sequence
import time
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0, _motion_enabled
from mme_vla_suite.policies.framesamp_memory import FrameSampMemory
from mme_vla_suite.shared.sampling import memory_order, pad_times   # 与训练侧 FrameSampDataset 同一份排序函数（R20）

class MME_VLA_Policy:
    def __init__(
        self,
        model: HistoryPi0,
        *,
        seed: int = 42,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        norm_stats: dict[str, _transforms.NormStats] | None = None,
        use_quantiles: bool = False,
        motion_enc_fn: Any | None = None,
    ):
        self._model = model
        self._seed = seed
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._vision_encode = nnx_utils.module_jit(model.vision_encode)
        
        
        self.config = model.history_config
        self.mem_buffer = None
        
        self.state_norm_stats = norm_stats['state']
        self.use_quantiles = use_quantiles

        # ── motion memory（motion-memory-plan.md 第二部分三节）：编码句柄（sidecar 客户端 / stub）建在 policy 层、跨 episode 常驻；
        #    FrameSampMemory 每 episode 随 reset() 销毁重建，只注入引用
        mcfg = getattr(self.config, "motion", None)
        self.motion_enabled = _motion_enabled(self.config)      # 与模型侧同一判定式
        self._motion_client = motion_enc_fn
        if self.motion_enabled and self._motion_client is None:
            raise ValueError("motion.enabled=true 的模型必须注入 motion_enc_fn（MotionEncoderClient 或 stub）")
        if self.motion_enabled:
            self._motion_cfg = {k: mcfg[k] for k in ("stride", "window_frames", "budget", "frame_size", "pos_dim", "dim",
                                                     "window_direction", "grid_origin")}
        
        self.reset()
        
    
    def _prepare_mem_buffer(self):
        # commitV4.4：framesamp 是唯一在线记忆路径（recurrent/symbolic/token_dropping
        # 与 pi05_baseline 的 config is None 分支均已删除，见 git 历史）
        self.mem_buffer = FrameSampMemory(
            num_views=self.config.num_views,
            img_emb_dim=self.config.memory_feature.img.input_dim,
            pos_emb_dim=self.config.memory_feature.pos.input_dim,
            state_emb_dim=self.config.memory_feature.state.input_dim,
            vision_enc_fn=self._vision_encode,
            motion_enc_fn=self._motion_client if self.motion_enabled else None,
            motion_cfg=self._motion_cfg if self.motion_enabled else None,
        )

    @override
    def infer(self, obs: dict) -> dict:
        if self.mem_buffer.n_steps == 0:
            raise RuntimeError("history feats is empty, add buffer first")

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._prepare_history(inputs)
        inputs = self._input_transform(inputs)
        observation = HistAugObservation.from_dict(
            jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        )
        self._rng, sample_rng = jax.random.split(self._rng)
    
        start_time = time.monotonic()
        outputs = {
            "state": observation.state,
            "actions": self._sample_actions(sample_rng, observation, **self._sample_kwargs),
        }
        # 计时段内同步（S3：infer_time_ms 原先只夹派发，jit 异步；同步后才可引用为端到端延迟）
        jax.block_until_ready(outputs["actions"])
        model_time = time.monotonic() - start_time
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)      
        outputs = self._output_transform(outputs)
        outputs["infer_time_ms"] = model_time * 1000
        
        return outputs
    
    @override
    def reset(self) -> None:
        del self.mem_buffer
        self._prepare_mem_buffer()
        self.step_idx = -1  
        self.exec_start_idx = 0
        self._rng = jax.random.key(self._seed)
            
    
    def add_buffer(self, obs: dict) -> None:
        if self.mem_buffer is None:
            return
        images = obs["images"]
        states = obs["state"]
        es_in = int(obs.get("exec_start_idx", 0))
        # 段边界状态机（S3 P4）：首批接受真实 es；后续客户端 clear_buffers 后传回的 0 按协议解释为「沿用已保存值」；
        # 后续非零值只有等于已保存值才合法，不同则 raise（Button 首批 / 后续均为 0；Video 首批 66 / 114 / 168 / 216…）
        if es_in > 0:
            if self.step_idx >= 0 and es_in != self.exec_start_idx:
                raise ValueError(f"exec_start_idx 在 episode 中途变化: 已保存 {self.exec_start_idx} → 本批 {es_in}")
            self.exec_start_idx = es_in
        
        step_idx_list = list(range(self.step_idx+1, self.step_idx + len(images) + 1))
        if self.motion_enabled:
            self.mem_buffer.add_buffer(images, states, step_idx_list, exec_start_idx=self.exec_start_idx)
        else:
            self.mem_buffer.add_buffer(images, states, step_idx_list)
        self.step_idx += len(images)

    def _normalize_state(self, state):
        if self.use_quantiles:
            return (state - self.state_norm_stats.q01) / (self.state_norm_stats.q99 - self.state_norm_stats.q01 + 1e-6) * 2.0 - 1.0
        else:
            return (state - self.state_norm_stats.mean) / (self.state_norm_stats.std + 1e-6)

    def _prepare_history(self, inputs: dict) -> dict:
        history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn
        token_budget = self.config.budget
        token_per_image = self.config.token_per_image
        static_image_emb, static_pos_emb, static_state_emb, static_mask = \
            self.mem_buffer.prepare_frame_sampling(
                self.step_idx, token_budget, token_per_image, history_feats_gather_fn)

        inputs["static_image_emb"] = static_image_emb
        inputs["static_pos_emb"] = static_pos_emb
        inputs["static_state_emb"] = self._normalize_state(static_state_emb)
        inputs["static_mask"] = static_mask

        if self.motion_enabled:
            # 运动路装配 + 与训练侧同一份 memory_order 交错（帧路帧号再调一次纯函数 get_frame_sampling_indices，与内部那次同值）
            motion_emb, motion_pos, motion_mask, motion_times = self.mem_buffer._prepare_motion(self.step_idx)
            frames = self.mem_buffer.get_frame_sampling_indices(self.step_idx, token_budget, token_per_image)
            max_frames = token_budget // (token_per_image * self.config.num_views)
            frame_times = pad_times(frames, max_frames)
            inputs["motion_emb"] = motion_emb
            inputs["motion_pos"] = motion_pos
            inputs["motion_mask"] = motion_mask
            inputs["mem_order"] = memory_order(frame_times, token_per_image * self.config.num_views, motion_times)

        return inputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata