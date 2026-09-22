"""在线评估 framesamp 专用记忆缓冲 `FrameSampMemory`（commitV4.4 新增；v2-motionmem S3 加运动路）。

替代旧 `shared.mem_buffer.MemoryBuffer` 在在线评估侧的角色，训练/在线自此不再
import mem_buffer（建库域留有冻结副本 `dataset_builder/mem_buffer.py`）。

与旧实现的三处合法差异（均有不可观测性论证，其余逐跳同式）：
① pos 表与池化只算配置指定的一档（PosEmb3D 是无 RNG、无参数的纯函数；
   4×4 表为 192 MiB，8×8 表为 768 MiB，各档独立计算，不改变原档数值）；
② `jax.device_get` 提到循环外一次（旧实现每步对同一 device 数组重复 get，
   取回的字节相同）；
③ 不存 image_pixels 或配置之外的网格，装配只读当前网格的 image/pos 与 state 三键。

⚠ 禁把 encode 与 pool 包进新的 jax.jit（R18）：融合边界变了，bf16 累加序可能变位。
本实现保持 encode（注入的 vision_enc_fn，本身已 jit）与 pool 分离调用，同旧实现。

── 运动路（0901-motion-memory-plan.md 第二部分三节，S3）──────────────────────────────
帧路 `_prepare_frame_sampling` 一字不动（「只换模块、不换数值路径」）。另加：
- 注入 `motion_enc_fn`（同 `vision_enc_fn` 范式；模型本体 / sidecar 句柄建在 `MME_VLA_Policy.__init__`，本类每 episode 随 `reset()` 销毁重建，不持模型）；
- 256 域原始帧缓冲 `_raw_frames`（现有 add_buffer 缩到 224 后就丢了原图，Wan VAE 要 256 域），入库前 raise 校验 `(frame_size, frame_size, 3)`；
- 段边界 `exec_start_idx` 由 policy 显式下传；demo / exec 各持 `next_grid_start`（段内绝对位置，初值 0，编完一窗 `+= stride`），
  每次 add_buffer 后用 **while** 编完合法起点（demo 按最少真实帧契约，exec 始终要求 33 帧），
  新 demo 契约把不足 33 帧的尾窗用第 `es−1` 帧补齐，绝不借用 exec 首帧；
  存 `_history_feats_motion[f]`（键 = 全域起点帧号）；编完一窗后删除 `< next_grid_start` 的原始帧；
- `_prepare_motion(step_idx)`：按训练侧同一公式取全部合法起点（>budget 立即报错、不裁剪），右填充 + mask，
  `motion_pos = pos_emb[f, 0, :pos_dim]`（与训练侧 `store.pos_rows` 同表同切片），并返回每行全域时刻（padding 记哨兵）供交错排序。
"""

import math
import os
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
        token_per_image: int,
        motion_enc_fn: Callable | None = None,
        motion_cfg: dict | None = None,
    ):
        if vision_enc_fn is None:
            raise ValueError("FrameSampMemory 必须注入 vision_enc_fn（模型侧编码器）")
        self.num_views = num_views
        grid = math.isqrt(token_per_image)
        if grid * grid != token_per_image or grid not in (2, 4, 8):
            raise ValueError(f"不支持的 token_per_image={token_per_image}，须为 4、16 或 64")
        self.token_per_image = token_per_image
        self.image_key = f"image_emb_{grid}x{grid}"
        self.pos_key = f"pos_emb_{grid}x{grid}"
        self.img_emb_dim = img_emb_dim
        self.pos_emb_dim = pos_emb_dim
        self.state_emb_dim = state_emb_dim
        self.max_steps = max_steps

        self.vision_enc = vision_enc_fn
        # 同一 PosEmb3D、同一 arange 输入，只按配置选择空间网格。
        pos_embedder = PosEmb3D(dim=pos_emb_dim)
        ranges = jnp.arange(max_steps)
        self.pos_emb = np.array(pos_embedder(ranges, grid))

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
            has_min = "demo_min_real_frames" in motion_cfg
            has_pad = "demo_tail_pad" in motion_cfg
            if has_min != has_pad:
                raise ValueError("demo_min_real_frames 与 demo_tail_pad 必须同时提供或同时缺省")
            self.demo_min_real = motion_cfg.get("demo_min_real_frames", 33)
            self.demo_tail_pad = motion_cfg.get("demo_tail_pad", "none")
            if type(self.demo_min_real) is not int or not (
                (self.demo_min_real == self.motion_window and self.demo_tail_pad == "none")
                or (1 <= self.demo_min_real <= self.motion_window and self.demo_tail_pad == "repeat_last")
            ):
                raise ValueError(f"在线 demo 窗口契约不合法: {self.demo_min_real}, {self.demo_tail_pad!r}")
            if str(motion_cfg.get("window_direction", "forward")) != "forward" or str(motion_cfg.get("grid_origin", "segment_start")) != "segment_start":
                raise ValueError("在线运动路只实现 forward + segment_start 口径")
            # 超预算口径（0922-binfill-demo-prefix-plan.md C 节）：默认 "raise"，与补 demo 之前一字不差；
            # "resample" 只在 k > budget 时生效，做「demo 段全保 + exec 段内等距降采样」。开关由 policy 层
            # 从环境变量 MMEVLA_MOTION_OVERFLOW 读入后写进 motion_cfg——不能进 resolved 快照，那份 sha 被三方钉死。
            self.motion_overflow = str(motion_cfg.get("overflow", "raise"))
            if self.motion_overflow not in ("raise", "resample"):
                raise ValueError(f"motion.overflow 只能是 raise / resample，得到 {self.motion_overflow!r}")
        self._history_feats_motion: dict[int, np.ndarray] = {}
        self._raw_frames: dict[int, np.ndarray] = {}
        self._next_grid_start_demo = 0
        self._next_grid_start_exec = 0
        self.exec_start_idx: int | None = None
        self.motion_encode_calls = 0
        self.motion_encode_s = 0.0
        # 降级统计：由 policy.infer 回传客户端，落 motion_stats.json 供 check_shard.py 的 MOTION_WINDOWS 闸验收
        self.motion_downsample_steps = 0
        self.motion_downsample_max_k = 0
        self.motion_last_k = 0

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
        self.motion_downsample_steps = 0
        self.motion_downsample_max_k = 0
        self.motion_last_k = 0

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
        # MMEVLA_ENC_CHUNK（默认 0 = 关闭）：BinFill 补 demo 前缀后首批最多 1282 帧一次性进 SigLIP，
        # (t,1,224,224,3) float32 输入与 bf16 输出都随 t 线性涨（1282 帧约 771 MiB / 1.3 GiB），显存风险未测。
        # 设正整数即按该尺寸切 batch 维分批；切 batch 可能让 XLA 换 kernel、改变 bf16 累加序，
        # 因此默认关闭（chunk >= t 时走单次循环，与关闭时逐字等价），开启前须先跑 enc_chunk_cmp.py 测出差异量级。
        chunk = int(os.environ.get("MMEVLA_ENC_CHUNK", "0") or 0)
        if chunk < 0:
            raise ValueError(f"MMEVLA_ENC_CHUNK 必须为非负整数，得到 {chunk}")
        if chunk == 0 or chunk >= t:
            chunk = t
        pooled_parts = []
        for b0 in range(0, t, chunk):
            b1 = min(b0 + chunk, t)
            image_jnp = jnp.array(
                images[b0:b1].astype(np.float32) / 255.0 * 2.0 - 1.0
            )
            image_jnp = einops.rearrange(image_jnp, "t v h w c -> (t v) h w c")
            image_jnp = image_tools.resize_with_pad(image_jnp, 224, 224)
            image_jnp = einops.rearrange(image_jnp, "(t v) h w c -> t v h w c", t=b1 - b0, v=v)
            output_emb = self.vision_enc(image_jnp)  # 真实 SigLIP 输出 (t,v,256,2048)

            pooled_emb = pool_tokens_to_size(output_emb, self.token_per_image)
            pooled_parts.append(jax.device_get(pooled_emb))  # 每批循环外一次（合法差异②；不分批时就是原来的一次）
        pooled_host = pooled_parts[0] if len(pooled_parts) == 1 else np.concatenate(pooled_parts, axis=0)

        for i, step_idx in enumerate(step_idx_list):
            image_emb = pooled_host[i]  # (v,token_per_image,2048)
            pos_emb = self.pos_emb[
                step_idx*self.num_views : (step_idx+1)*self.num_views]

            self._history_feats[step_idx] = {
                self.image_key: image_emb,        # bf16
                self.pos_key: pos_emb,            # fp32
                "state_emb": states[i],          # fp32
            }

        if self.motion_enabled:
            # 判据用**本批最后一帧的全域帧号**（禁止读上一批末帧：exec 段会整整晚 16 帧、首批更是 −1）
            self._encode_ready_windows(last_frame=int(step_idx_list[-1]))

    # ―― 运动路：增量编码（while 循环补齐全部已合法起点）――
    def _encode_window(self, f: int) -> None:
        import time
        W = self.motion_window
        es = self.exec_start_idx
        real = min(W, es - f) if f < es else W
        if real < (self.demo_min_real if f < es else W):
            raise RuntimeError(f"起点 {f} 的真实帧数 {real} 不满足段契约")
        if real < W and self.demo_tail_pad != "repeat_last":
            raise RuntimeError("当前 demo 契约不允许尾部补帧")
        frame_ids = [f + j for j in range(real)] + [es - 1] * (W - real)
        frames = [self._raw_frames.get(i) for i in frame_ids]
        if any(x is None for x in frames):
            missing = [i for i, x in zip(frame_ids, frames) if x is None]
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
        while self._next_grid_start_demo + (self.demo_min_real - 1) <= es - 1:
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
        demo_unfinished = self._next_grid_start_demo + (self.demo_min_real - 1) <= es - 1
        if demo_unfinished:                                      # demo 未编完（首批应整段到货）
            keep_from = min(keep_from, self._next_grid_start_demo)
        for k in [k for k in self._raw_frames if k < keep_from and not (k < es and demo_unfinished)]:
            del self._raw_frames[k]

    def visible_motion_frames(self, step_idx: int) -> list[int]:
        """当前帧 step_idx 下的合法起点全域帧号（升序）——与训练侧 motion_store.visible_motion_rows 同式。"""
        es = self.exec_start_idx
        W = self.motion_window
        out = []
        s = 0
        while s + (self.demo_min_real - 1) <= es - 1:
            out.append(s)
            s += self.motion_stride
        u = 0
        while u + (W - 1) <= step_idx - es:
            out.append(es + u)
            u += self.motion_stride
        return out

    def _prepare_motion(self, step_idx: int):
        """运动路装配：全部合法起点（>budget 报错、不裁剪）→ 右填充 + mask；motion_pos = pos_emb[f, 0, :pos_dim]。
        返回 (motion_emb (B,768) f32, motion_pos (B,pos_dim) f32, motion_mask (B,) bool, times (B,) int64)。"""
        if self.exec_start_idx is None:
            raise RuntimeError("尚未 add_buffer（exec_start_idx 未知）")
        frames = self.visible_motion_frames(step_idx)
        k = len(frames)
        B = self.motion_budget
        self.motion_last_k = k
        if k > B:
            if self.motion_overflow != "resample":
                raise RuntimeError(f"step {step_idx} 合法 motion 起点数 {k} > motion.budget {B}（零截断契约，禁止裁剪）")
            # 降级（overflow=resample）：超额只可能来自 exec 段——评测 max_steps=2000 让 exec 窗恒 124，
            # 是训练 exec 上限 70 的 1.77 倍；demo 段与训练同分布，故 demo 全保、超额由 exec 段内等距抽稀承担。
            # 被丢的窗仍留在 _history_feats_motion 里，下一步重算取点时可能又被选中，直接取用、不重编码。
            es = self.exec_start_idx
            demo = [f for f in frames if f < es]
            exec_ = [f for f in frames if f >= es]
            keep_demo, keep_exec = _motion_quota(len(demo), len(exec_), B)
            frames = ([demo[i] for i in _even_pick(len(demo), keep_demo)]
                      + [exec_[i] for i in _even_pick(len(exec_), keep_exec)])
            # 显式核，不靠注释：降级后必须恰好填满预算且全域帧号严格递增（mem_order 依赖升序）
            if len(frames) != B or any(b <= a for a, b in zip(frames, frames[1:])):
                raise RuntimeError(
                    f"降级结果非法：step {step_idx} 得到 {len(frames)} 个起点（应为 {B}）或次序非严格递增")
            self.motion_downsample_steps += 1
            self.motion_downsample_max_k = max(self.motion_downsample_max_k, k)
            print(f"MOTION_DOWNSAMPLE step={step_idx} es={es} k={k} k_demo={len(demo)} k_exec={len(exec_)} "
                  f"kept_demo={keep_demo} kept_exec={keep_exec} budget={B}", flush=True)
            k = len(frames)
        emb = np.zeros((B, self.motion_dim), np.float32)
        pos = np.zeros((B, self.motion_pos_dim), np.float32)
        for i, f in enumerate(frames):
            if f not in self._history_feats_motion:
                raise RuntimeError(f"起点 {f} 应已编码但缓冲中没有（增量编码落后）")
            emb[i] = self._history_feats_motion[f]
            pos[i] = self.pos_emb[f, 0, : self.motion_pos_dim]
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

        # 视觉记忆使用右侧补零。
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


def _even_pick(n: int, keep: int) -> list[int]:
    """在 ``[0, n)`` 上等距取 ``keep`` 个下标（升序、无重复）。

    ``keep == n`` 时恒等返回、绝不走 linspace（浮点取整在边界上可能与 range 不同）。
    取点用 ``np.linspace(...).astype(np.int64)`` **截断**而非 round，与帧路
    ``sampling.even_sampling_indices`` 同一算子。调用方保证 ``keep < n`` 时步长
    ``(n-1)/(keep-1) > 1``，故截断不会撞出重复；仍显式核一遍——一旦少一个，
    ``mask[:k]`` 与 ``pad_times`` 会静默给出「非满」形态，没有别的地方能发现。
    """
    if keep == n:
        return list(range(n))
    if not (0 < keep < n):
        raise RuntimeError(f"_even_pick 非法参数 n={n} keep={keep}")
    out = [int(i) for i in np.linspace(0, n - 1, keep, dtype=np.int64)]
    if len(out) != keep or any(b <= a for a, b in zip(out, out[1:])):
        raise RuntimeError(f"_even_pick 产生重复或非递增下标 n={n} keep={keep}")
    return out


def _motion_quota(k_demo: int, k_exec: int, budget: int) -> tuple[int, int]:
    """超预算时两段各保留几个窗。调用方保证 ``k_demo + k_exec > budget``。

    demo 段优先全保，但不得超过 ``budget // 2``——这条保护防 demo 变长时把 exec 挤成 0。
    预生成侧另有 ``D <= 1281`` 的硬闸把 ``k_demo`` 钉在 80 以内，所以正常样本里这条保护不会真的丢窗
    （训练实测 demo 窗上限 71）。exec 段吃不满余额时把余额退回 demo 段，保证合计恰好等于 budget。
    """
    keep_demo = min(k_demo, budget // 2)
    keep_exec = min(k_exec, budget - keep_demo)
    keep_demo = budget - keep_exec
    if keep_demo > k_demo or keep_exec > k_exec or keep_demo + keep_exec != budget:
        raise RuntimeError(
            f"_motion_quota 配额非法：k_demo={k_demo} k_exec={k_exec} budget={budget} "
            f"→ keep_demo={keep_demo} keep_exec={keep_exec}")
    return keep_demo, keep_exec


def _accepts_start(fn) -> bool:
    """motion_enc_fn 若接受第二个位置参数（起点帧号，MotionEncoderClient 用于 stub 校验）则传入。"""
    import inspect
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(params) >= 2
