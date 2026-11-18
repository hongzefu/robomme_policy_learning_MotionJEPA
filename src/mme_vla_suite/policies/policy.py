from collections.abc import Sequence
import os
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

        # ── motion memory（0901-motion-memory-plan.md 第二部分三节）：编码句柄（sidecar 客户端 / stub）建在 policy 层、跨 episode 常驻；
        #    FrameSampMemory 每 episode 随 reset() 销毁重建，只注入引用
        mcfg = getattr(self.config, "motion", None)
        self.motion_enabled = _motion_enabled(self.config)      # 与模型侧同一判定式
        self._motion_client = motion_enc_fn
        if self.motion_enabled and self._motion_client is None:
            raise ValueError("motion.enabled=true 的模型必须注入 motion_enc_fn（MotionEncoderClient 或 stub）")
        if self.motion_enabled:
            if ("demo_min_real_frames" in mcfg) != ("demo_tail_pad" in mcfg):
                raise ValueError("demo_min_real_frames 与 demo_tail_pad 必须同时提供或同时缺省")
            self._motion_cfg = {k: mcfg[k] for k in ("stride", "window_frames", "budget", "frame_size", "pos_dim", "dim",
                                                     "window_direction", "grid_origin")}
            self._motion_cfg.update(demo_min_real_frames=mcfg.get("demo_min_real_frames", 33),
                                    demo_tail_pad=mcfg.get("demo_tail_pad", "none"))
            # 每窗 token 数（计划 1112 第二部分 1.1 第 21 行）：wan_raw9 为 9，旧 YAML 缺省 1；下游只在 >1 时读取
            self._motion_cfg["tokens_per_window"] = int(mcfg.get("tokens_per_window", 1))
            # E2E-Enc 键（计划 1117 第二部分「一」policy.py 行）：只在快照写了 encoder_kind 时拷，旧 run 的 _motion_cfg 逐键不变。
            # FrameSampMemory 按 encoder_kind == "wan_e2e" 走 e2e 装配（每槽 latents_per_window×raw_dim bf16）。
            if mcfg.get("encoder_kind", None) is not None:
                self._motion_cfg["encoder_kind"] = str(mcfg["encoder_kind"])
                for key in ("readout", "latents_per_window", "raw_dim"):
                    if mcfg.get(key, None) is not None:
                        self._motion_cfg[key] = mcfg[key] if key == "readout" else int(mcfg[key])
            # 超预算口径只能走环境变量：评测侧 motion 配置来自 checkpoint 桶里的 history_config.resolved.yaml，
            # 其 sha256 被 check_checkpoint.py::RUNS 与 motion_provenance.json 三方钉死，改任何一份都到不了评测路径或直接炸闸。
            # 优先级 = 环境变量 > resolved 快照可选键 > "raise"（默认，与补 demo 之前一字不差）。
            overflow = (os.environ.get("MMEVLA_MOTION_OVERFLOW", "").strip()
                        or str(mcfg.get("overflow", "raise")))
            if overflow not in ("raise", "resample"):
                raise ValueError(
                    f"MMEVLA_MOTION_OVERFLOW / motion.overflow 只能是 raise / resample，得到 {overflow!r}")
            self._motion_cfg["overflow"] = overflow
            print(f"MOTION_OVERFLOW={overflow} budget={self._motion_cfg['budget']} "
                  f"stride={self._motion_cfg['stride']} window_frames={self._motion_cfg['window_frames']}", flush=True)
        
        # G5b 在线性能测量（计划 1117 三节 G5b）：设 MMEVLA_PERF_LOG=<jsonl 路径> 时逐事件追加记录（add_buffer 耗时、
        # 单次策略推理耗时、单局墙钟、请求字节、本进程 XLA 峰值显存、降采样次数），由
        # compare_train_infer_obs.py --perf-report 汇总成 E2E_ONLINE_PERF 行；不设时零开销、行为不变。
        self._perf_path = os.environ.get("MMEVLA_PERF_LOG", "").strip() or None
        self._perf_ep: dict | None = None
        self._perf_episode_idx = 0
        if self._perf_path:
            import atexit
            atexit.register(self._perf_close_episode)
        self.reset()
        
    
    def _perf_write(self, rec: dict) -> None:
        import json
        with open(self._perf_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def _perf_peak_bytes() -> int | None:
        try:
            stats = jax.local_devices()[0].memory_stats() or {}
            return int(stats.get("peak_bytes_in_use")) if "peak_bytes_in_use" in stats else None
        except Exception:
            return None

    def _perf_close_episode(self) -> None:
        ep = self._perf_ep
        self._perf_ep = None
        if not self._perf_path or ep is None or ep["n_add"] == 0:
            return
        client = self._motion_client
        mem = self.mem_buffer
        rec = {"event": "episode", "pid": os.getpid(), "episode_idx": self._perf_episode_idx,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "wall_s": ep["t_last"] - ep["t_first"], "n_add_buffer": ep["n_add"], "add_buffer_ms": ep["add_ms"],
               "n_infer": len(ep["infer_ms"]), "infer_ms": ep["infer_ms"], "last_step_idx": int(self.step_idx),
               "request_bytes": (getattr(client, "bytes_sent", 0) or 0) - ep["bytes_sent0"] if client is not None else 0,
               "response_bytes": (getattr(client, "bytes_recv", 0) or 0) - ep["bytes_recv0"] if client is not None else 0,
               "motion_encode_calls": int(getattr(mem, "motion_encode_calls", 0)),
               "motion_downsample_steps": int(getattr(mem, "motion_downsample_steps", 0)),
               "motion_downsample_max_k": int(getattr(mem, "motion_downsample_max_k", 0)),
               "peak_bytes_in_use": self._perf_peak_bytes()}
        self._perf_write(rec)
        self._perf_episode_idx += 1

    def _perf_touch(self) -> dict:
        if self._perf_ep is None:
            client = self._motion_client
            now = time.perf_counter()
            self._perf_ep = {"t_first": now, "t_last": now, "n_add": 0, "add_ms": [], "infer_ms": [],
                             "bytes_sent0": getattr(client, "bytes_sent", 0) or 0 if client is not None else 0,
                             "bytes_recv0": getattr(client, "bytes_recv", 0) or 0 if client is not None else 0}
        return self._perf_ep

    def _prepare_mem_buffer(self):
        # commitV4.4：framesamp 是唯一在线记忆路径（recurrent/symbolic/token_dropping
        # 与 pi05_baseline 的 config is None 分支均已删除，见 git 历史）
        self.mem_buffer = FrameSampMemory(
            num_views=self.config.num_views,
            img_emb_dim=self.config.memory_feature.img.input_dim,
            pos_emb_dim=self.config.memory_feature.pos.input_dim,
            state_emb_dim=self.config.memory_feature.state.input_dim,
            vision_enc_fn=self._vision_encode,
            token_per_image=int(self.config.token_per_image),
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
        if self._perf_path:
            ep = self._perf_touch()
            ep["infer_ms"].append(model_time * 1000)
            ep["t_last"] = time.perf_counter()
            # 逐次落一行：服务进程被外部结束时最后一局没有 episode 行，汇总端用 infer 行补算
            self._perf_write({"event": "infer", "pid": os.getpid(), "episode_idx": self._perf_episode_idx,
                              "step_idx": int(self.step_idx), "infer_ms": model_time * 1000,
                              "t_rel_s": ep["t_last"] - ep["t_first"],
                              "add_ms_sum": float(sum(ep["add_ms"])), "n_add_buffer": ep["n_add"],
                              "request_bytes": ((getattr(self._motion_client, "bytes_sent", 0) or 0) - ep["bytes_sent0"])
                              if self._motion_client is not None else 0,
                              "motion_k": int(getattr(self.mem_buffer, "motion_last_k", 0)),
                              "peak_bytes_in_use": self._perf_peak_bytes()})
        if self.motion_enabled:
            # 随 infer_time_ms 一并回传（已有先例）：客户端累计成 motion_stats.json，
            # 供 check_shard.py 的 MOTION_WINDOWS 闸用闭式公式独立复核
            outputs["motion_k"] = int(self.mem_buffer.motion_last_k)
            outputs["motion_downsample_steps"] = int(self.mem_buffer.motion_downsample_steps)
            # budget 与 overflow 一并回传，验收端才不用硬编码假设这两个值是什么
            outputs["motion_budget"] = int(self.mem_buffer.motion_budget)
            outputs["motion_overflow"] = str(self.mem_buffer.motion_overflow)

        return outputs
    
    @override
    def reset(self) -> None:
        if getattr(self, "_perf_path", None):
            self._perf_close_episode()
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
        ep = self._perf_touch() if self._perf_path else None      # 先建本局记录（含客户端字节基线），再计时
        t_add = time.perf_counter()
        if self.motion_enabled:
            self.mem_buffer.add_buffer(images, states, step_idx_list, exec_start_idx=self.exec_start_idx)
        else:
            self.mem_buffer.add_buffer(images, states, step_idx_list)
        self.step_idx += len(images)
        if ep is not None:
            if ep["n_add"] == 0:
                ep["t_first"] = t_add
            ep["n_add"] += 1
            ep["add_ms"].append((time.perf_counter() - t_add) * 1000)
            ep["t_last"] = time.perf_counter()

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
