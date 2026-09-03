"""在线评估 framesamp 专用记忆缓冲 `FrameSampMemory`（commitV4.4 新增；v2-motionmem S3 加运动路）。

替代旧 `shared.mem_buffer.MemoryBuffer` 在在线评估侧的角色，训练/在线自此不再
import mem_buffer（建库域留有冻结副本 `dataset_builder/mem_buffer.py`）。

与旧实现的三处合法差异（均有不可观测性论证，其余逐跳同式）：
① pos 表与池化只算 4x4 一档（frame_sampling 唯一消费档；PosEmb3D 是无 RNG、无参数
   的纯函数，各档独立，只算一档不改变该档的任何字节——表本体 1.01 GiB → 192 MiB）；
② `jax.device_get` 提到循环外一次（旧实现每步对同一 device 数组重复 get，
   取回的字节相同）；
③ 不存 image_pixels 与 8x8/2x2 档位（image_pixels 的唯一消费者是 token_drop 打分
   与死码可视化，装配只读 image_emb_4x4/pos_emb_4x4/state_emb 三键）。

⚠ 禁把 encode 与 pool 包进新的 jax.jit（R18）：融合边界变了，bf16 累加序可能变位。
本实现保持 encode（注入的 vision_enc_fn，本身已 jit）与 pool 分离调用，同旧实现。

── 运动路（motion-memory-plan.md 第二部分三节，S3）──────────────────────────────
帧路 `_prepare_frame_sampling` 一字不动（「只换模块、不换数值路径」）。另加：
- 注入 `motion_enc_fn`（同 `vision_enc_fn` 范式；模型本体 / sidecar 句柄建在 `MME_VLA_Policy.__init__`，本类每 episode 随 `reset()` 销毁重建，不持模型）；
- 256 域原始帧缓冲 `_raw_frames`（现有 add_buffer 缩到 224 后就丢了原图，Wan VAE 要 256 域），入库前 raise 校验 `(frame_size, frame_size, 3)`；
- 段边界 `exec_start_idx` 由 policy 显式下传；demo / exec 各持 `next_grid_start`（段内绝对位置，初值 0，编完一窗 `+= stride`），
  每次 add_buffer 后用 **while** 循环把所有已合法起点编完（demo 判据 `next+32 ≤ es−1`、exec 判据 `next+32 ≤ 本批末帧段内帧号`），
  存 `_history_feats_motion[f]`（键 = 全域起点帧号）；编完一窗后删除 `< next_grid_start` 的原始帧；
- `_prepare_motion(step_idx)`：按训练侧同一公式取全部合法起点（>budget 立即报错、不裁剪），右填充 + mask，
  `motion_pos = pos_emb_4x4[f, 0, :pos_dim]`（与训练侧 `store.pos_rows` 同表同切片），并返回每行全域时刻（padding 记哨兵）供交错排序。
"""

import math
from typing import Callable

import einops
import jax
import jax.numpy as jnp
import numpy as np

from openpi.shared import image_tools
from mme_vla_suite.shared.data_utils import pool_tokens_to_size, right_padding_token_emb
from mme_vla_suite.shared.posemb_3d import PosEmb3D
from mme_vla_suite.shared.sampling import even_sampling_indices, pad_times


class FrameSampMemory:
    def __init__(
        self,
        num_views: int = 1,
        img_emb_dim: int = 2048,
        pos_emb_dim: int = 768,
        state_emb_dim: int = 8,
        max_steps: int = 4096,
        *,
        vision_enc_fn: Callable,
        motion_enc_fn: Callable | None = None,
        motion_cfg: dict | None = None,
    ):
        if vision_enc_fn is None:
            raise ValueError("FrameSampMemory 必须注入 vision_enc_fn（模型侧编码器）")
        self.num_views = num_views
        self.img_emb_dim = img_emb_dim
        self.pos_emb_dim = pos_emb_dim
        self.state_emb_dim = state_emb_dim
        self.max_steps = max_steps

        self.vision_enc = vision_enc_fn
        # 与旧实现同一 PosEmb3D、同一 arange 输入——4x4 表逐位同旧表 "4x4" 档
        pos_embedder = PosEmb3D(dim=pos_emb_dim)
        ranges = jnp.arange(max_steps)
        self.pos_emb_4x4 = np.array(pos_embedder(ranges, 4))

        self._history_feats = {}

        # ―― 运动路 ――
        self.motion_enabled = motion_enc_fn is not None
        if self.motion_enabled:
            if motion_cfg is None:
                raise ValueError("注入 motion_enc_fn 时必须给 motion_cfg")
            self.motion_enc = motion_enc_fn
            self.motion_stride = int(motion_cfg["stride"])
            self.motion_window = int(motion_cfg["window_frames"])
            self.motion_budget = int(motion_cfg["budget"])
            self.motion_frame_size = int(motion_cfg["frame_size"])
            self.motion_pos_dim = int(motion_cfg["pos_dim"])
            self.motion_dim = int(motion_cfg["dim"])
            if str(motion_cfg.get("window_direction", "forward")) != "forward" or str(motion_cfg.get("grid_origin", "segment_start")) != "segment_start":
                raise ValueError("在线运动路只实现 forward + segment_start 口径")
        self._history_feats_motion: dict[int, np.ndarray] = {}
        self._raw_frames: dict[int, np.ndarray] = {}
        self._next_grid_start_demo = 0
        self._next_grid_start_exec = 0
        self.exec_start_idx: int | None = None
        self.motion_encode_calls = 0
        self.motion_encode_s = 0.0

    @property
    def n_steps(self) -> int:
        return len(self._history_feats)

    def clear(self):
        self._history_feats.clear()
        self._history_feats_motion.clear()
        self._raw_frames.clear()
        self._next_grid_start_demo = 0
        self._next_grid_start_exec = 0
        self.exec_start_idx = None

    def add_buffer(
        self,
        images,  #: (t v h w 3), np.uint8
        states,  #: (t d), np.float32
        step_idx_list: list[int],
        exec_start_idx: int | None = None,
    ):
        t, v, _, _, _ = images.shape
        assert v == self.num_views

        for step_idx in step_idx_list:
            # 有效域上界显式 raise：pos 表只有 max_steps 行，numpy 切片越界会静默
            # 返回空数组、坏数据一路流到装配层——禁止依赖切片行为
            if step_idx >= self.max_steps:
                raise ValueError(
                    f"step_idx {step_idx} 超出记忆有效域 [0, {self.max_steps})"
                )
            if step_idx in self._history_feats:
                raise ValueError(f"step_idx {step_idx} already in buffer")

        if self.motion_enabled:
            # 运动路入库前校验：段边界必须下传；原始帧必须是 256 域（喂 224 / 512 网络照样能算、结果却与训练不同，属静默错配）
            # 先全部校验、后写状态（校验失败的批不得留下 exec_start_idx / 原始帧等半截状态）
            if exec_start_idx is None:
                raise ValueError("motion 开启时 add_buffer 必须显式传 exec_start_idx（段边界下传）")
            if self.exec_start_idx is not None and int(exec_start_idx) != self.exec_start_idx:
                raise ValueError(f"exec_start_idx 变化: 已保存 {self.exec_start_idx} → 本批 {exec_start_idx}")
            if tuple(images.shape[-3:]) != (self.motion_frame_size, self.motion_frame_size, 3) or images.dtype != np.uint8:
                raise ValueError(
                    f"原始帧尺寸 {tuple(images.shape[-3:])} {images.dtype} != ({self.motion_frame_size}, {self.motion_frame_size}, 3) uint8")
            for step_idx in step_idx_list:
                if step_idx in self._raw_frames:
                    raise ValueError(f"step_idx {step_idx} 已在原始帧缓冲")
            if self.exec_start_idx is None:
                self.exec_start_idx = int(exec_start_idx)
            for i, step_idx in enumerate(step_idx_list):
                self._raw_frames[step_idx] = np.ascontiguousarray(images[i, 0])

        # 与旧实现逐字同式的归一化 / resize / 编码链
        image_jnp = jnp.array(
            images.astype(np.float32) / 255.0 * 2.0 - 1.0
        )
        image_jnp = einops.rearrange(image_jnp, "t v h w c -> (t v) h w c")
        image_jnp = image_tools.resize_with_pad(image_jnp, 224, 224)
        image_jnp = einops.rearrange(image_jnp, "(t v) h w c -> t v h w c", t=t, v=v)
        output_emb = self.vision_enc(image_jnp)  # (t, v, 64, 2048)

        pooled_emb_4x4 = pool_tokens_to_size(output_emb, 16)  # (t, v, 16, 2048)
        pooled_host = jax.device_get(pooled_emb_4x4)  # 循环外一次（合法差异②）

        for i, step_idx in enumerate(step_idx_list):
            image_emb_4x4 = pooled_host[i]  # (v, 16, 2048)
            pos_emb_4x4 = self.pos_emb_4x4[
                step_idx*self.num_views : (step_idx+1)*self.num_views]  # (v, 16, 768)

            self._history_feats[step_idx] = {
                "image_emb_4x4": image_emb_4x4,  # bf16
                "pos_emb_4x4": pos_emb_4x4,      # fp32
                "state_emb": states[i],          # fp32
            }

        if self.motion_enabled:
            # 判据用**本批最后一帧的全域帧号**（禁止读上一批末帧：exec 段会整整晚 16 帧、首批更是 −1）
            self._encode_ready_windows(last_frame=int(step_idx_list[-1]))

    # ―― 运动路：增量编码（while 循环补齐全部已合法起点）――
    def _encode_window(self, f: int) -> None:
        import time
        W = self.motion_window
        frames = [self._raw_frames.get(f + j) for j in range(W)]
        if any(x is None for x in frames):
            missing = [f + j for j, x in enumerate(frames) if x is None]
            raise RuntimeError(f"起点 {f} 的 33 帧不齐（缺 {missing[:4]}…），原始帧缓冲被过早清理？")
        window = np.ascontiguousarray(np.stack(frames))
        t0 = time.perf_counter()
        tok = np.asarray(self.motion_enc(window, f) if _accepts_start(self.motion_enc) else self.motion_enc(window), dtype=np.float32)
        self.motion_encode_s += time.perf_counter() - t0
        self.motion_encode_calls += 1
        if tok.shape != (self.motion_dim,):
            raise RuntimeError(f"motion_enc_fn 返回形制 {tok.shape} != ({self.motion_dim},)")
        if f in self._history_feats_motion:
            raise RuntimeError(f"起点 {f} 已编过")
        self._history_feats_motion[f] = tok

    def _encode_ready_windows(self, last_frame: int) -> None:
        es = self.exec_start_idx
        W = self.motion_window
        # demo 段：整段已见，判据与 t 无关（首批一次跑完）
        while self._next_grid_start_demo + (W - 1) <= es - 1:
            s = self._next_grid_start_demo
            self._encode_window(s)                                   # 全域帧号 f = s
            self._next_grid_start_demo += self.motion_stride
        # exec 段：判据用本批末帧的段内帧号
        while self._next_grid_start_exec + (W - 1) <= last_frame - es:
            u = self._next_grid_start_exec
            self._encode_window(es + u)                              # 全域帧号 f = es + u
            self._next_grid_start_exec += self.motion_stride
        # 只保留下一个起点之后的原始帧（demo 段编完后其帧不再需要；exec 段保留 ≥ es + next_exec 的帧）
        keep_from = es + self._next_grid_start_exec
        if self._next_grid_start_demo + (W - 1) <= es - 1:           # demo 未编完（不应发生：首批整段到货）
            keep_from = min(keep_from, self._next_grid_start_demo)
        for k in [k for k in self._raw_frames if k < keep_from and not (k < es and self._next_grid_start_demo + (W - 1) <= es - 1)]:
            del self._raw_frames[k]

    def visible_motion_frames(self, step_idx: int) -> list[int]:
        """当前帧 step_idx 下的合法起点全域帧号（升序）——与训练侧 motion_store.visible_motion_rows 同式。"""
        es = self.exec_start_idx
        W = self.motion_window
        out = []
        s = 0
        while s + (W - 1) <= es - 1:
            out.append(s)
            s += self.motion_stride
        u = 0
        while u + (W - 1) <= step_idx - es:
            out.append(es + u)
            u += self.motion_stride
        return out

    def _prepare_motion(self, step_idx: int):
        """运动路装配：全部合法起点（>budget 报错、不裁剪）→ 右填充 + mask；motion_pos = pos_emb_4x4[f, 0, :pos_dim]。
        返回 (motion_emb (B,768) f32, motion_pos (B,pos_dim) f32, motion_mask (B,) bool, times (B,) int64)。"""
        if self.exec_start_idx is None:
            raise RuntimeError("尚未 add_buffer（exec_start_idx 未知）")
        frames = self.visible_motion_frames(step_idx)
        k = len(frames)
        B = self.motion_budget
        if k > B:
            raise RuntimeError(f"step {step_idx} 合法 motion 起点数 {k} > motion.budget {B}（零截断契约，禁止裁剪）")
        emb = np.zeros((B, self.motion_dim), np.float32)
        pos = np.zeros((B, self.motion_pos_dim), np.float32)
        for i, f in enumerate(frames):
            if f not in self._history_feats_motion:
                raise RuntimeError(f"起点 {f} 应已编码但缓冲中没有（增量编码落后）")
            emb[i] = self._history_feats_motion[f]
            pos[i] = self.pos_emb_4x4[f, 0, : self.motion_pos_dim]
        mask = np.zeros(B, np.bool_)
        mask[:k] = True
        return emb, pos, mask, pad_times(frames, B)

    # ―― 以下装配路径与旧 MemoryBuffer 的 frame_sampling 支路逐字同式 ――

    def get_frame_sampling_indices(self, step_idx, token_budget, token_per_image):
        max_size = token_budget // (token_per_image * self.num_views)
        return even_sampling_indices(step_idx, max_size)

    def _prepare_frame_sampling(self, history_feats, indices_to_load, token_budget, token_per_image):
        spatial_size = str(int(math.sqrt(token_per_image)))
        spatial_key = f"{spatial_size}x{spatial_size}"
        max_size = token_budget // (token_per_image * self.num_views)

        sampled_img_emb = self._load_emb(history_feats, indices_to_load, f"image_emb_{spatial_key}")
        sampled_pos_emb = self._load_emb(history_feats, indices_to_load, f"pos_emb_{spatial_key}")
        sampled_state_emb = self._load_emb(history_feats, indices_to_load, "state_emb")
        mask = np.ones((sampled_img_emb.shape[0]), dtype=np.bool_)

        # we use right padding to the perceptual memory
        # （必须复用 right_padding_token_emb——只换模块、不换数值路径，禁改写预分配版）
        sampled_img_emb, sampled_pos_emb, sampled_state_emb, mask = right_padding_token_emb(
            sampled_img_emb, sampled_pos_emb, sampled_state_emb, mask, max_size
        )

        img_emb = np.reshape(sampled_img_emb, (-1, self.img_emb_dim))
        pos_emb = np.reshape(sampled_pos_emb, (-1, self.pos_emb_dim))
        mask = np.repeat(mask, self.num_views * token_per_image)
        state_emb = np.repeat(sampled_state_emb, self.num_views * token_per_image, axis=0)

        return img_emb, pos_emb, state_emb, mask

    def prepare_frame_sampling(self, step_idx, token_budget, token_per_image, history_feats_gather_fn, *args, **kwargs):
        indices_to_load = self.get_frame_sampling_indices(step_idx, token_budget, token_per_image)
        history_feats = history_feats_gather_fn(indices_to_load, *args, **kwargs)
        return self._prepare_frame_sampling(history_feats, indices_to_load, token_budget, token_per_image)

    @staticmethod
    def _load_emb(history_feats: dict, indices_to_load: list[int], key: str):
        return np.stack(
            [history_feats[idx][key] for idx in indices_to_load],
            axis=0,
        )

    def default_history_feats_gather_fn(self, indices_to_load, *args, **kwargs):
        return {idx: self._history_feats[idx] for idx in indices_to_load}


def _accepts_start(fn) -> bool:
    """motion_enc_fn 若接受第二个位置参数（起点帧号，MotionEncoderClient 用于 stub 校验）则传入。"""
    import inspect
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(params) >= 2
