"""FrameSampDataset：packed 特征库上的装配层（v2 计划 B.3）。

（本文件注释中的「旧路径」均指已删除的 legacy 数据链 RoboMMEDataset +
散 npy + MemoryBuffer——commitV4.1 删除，见 git 历史；对齐结论在删除前逐字核对固化。）

单一路径、无分支、只服务 `perceptual-framesamp-context` 一种 run：
① 清单查表得 (g, step)（O(1) 数组，含 exec_start_idx 换算，不读目录）；
② pickle.load 源库 data/{idx}.pkl（与旧路径同源同字节）＋ pkl 内身份互校
   （不符显式 raise——行号错位的最后一道闸，禁 assert，R6）；
③ even_sampling_indices 复用同一函数（R4），选帧逐位不变；
④ FrameSampStore gather：常驻 fd 游程合并 pread + 进程内小表——0 open、0 线程池、
   0 pickle；
⑤ _pad 预分配、填充区清零，交付 dtype 逐键与旧路径一致（image bf16 / pos f32 /
   stt f32）；
⑥ 拼装与 _normalize_state 与旧路径逐字同式（norm stats q01/q99 为 f64，输出恒 f64）。

spawn 生命周期契约（B.2）：__init__（主进程）只做读 meta + fail-loud 静态校验
（fast 档）+ 清单派生查表数组，不打开任何 part fd、不建任何 mmap；__getstate__
剔除 store 句柄；每进程首次 __getitem__ 懒构造 FrameSampStore 并记 owner pid，
pid 失配先显式 close() 旧句柄再重建；close() + atexit 兜底。
"""

from __future__ import annotations

import atexit
import logging
import os
import pathlib
import pickle

import numpy as np
from omegaconf import DictConfig

from openpi.training import config as _config
from openpi.training.data_loader import Dataset

from mme_vla_suite.datastore import (
    FrameSampStore,
    StoreMeta,
    build_exec_lookup,
    load_manifest,
    run_fast_checks,
    run_full_checks,
)
from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.shared.sampling import MEM_ORDER_SENTINEL, even_sampling_indices, memory_order, pad_times

logger = logging.getLogger(__name__)

# 与旧 RoboMMEDataset.__getitem__（commitV4.1 已删，见 git 历史）尾部补空键列表逐字一致（recur_* / subgoal 等
# 下游 transforms 会索引的键）
_NONE_KEYS = (
    "static_image_emb",
    "static_pos_emb",
    "static_state_emb",
    "static_mask",
    "prompt",
    # motion memory 四键（motion-memory-plan.md 2.6；关闭态恒 None，与 compute_norm_stats._NONE_KEYS 同 commit 同步）
    "motion_emb",
    "motion_pos",
    "motion_mask",
    "mem_order",
)


class FrameSampDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        data_config: _config.DataConfig,
        history_config: DictConfig | None,
        action_horizon: int,
        *,
        source_root: str | None = None,
        manifest_path: str | None = None,
        verify_level: str = "fast",
        motion_root: str | None = None,
    ):
        # ―― 形制断言即文档（必须能挡住同形的 modul 配置，G13；显式 raise 不用 assert）――
        hc = history_config

        def _req(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(f"FrameSampDataset 形制断言失败: {msg}")

        _req(hc is not None, "history_config 不能为 None")
        _req(hc.representation_type == "perceptual",
             f"representation_type={hc.representation_type!r} != 'perceptual'")
        _req(hc.perceptual_memory.type == "frame_sampling",
             f"perceptual_memory.type={hc.perceptual_memory.type!r} != 'frame_sampling'")
        _req(hc.integration_type == "context",
             f"integration_type={hc.integration_type!r} != 'context'")
        _req(int(hc.memory_token_dim) == 2048,
             f"memory_token_dim={hc.memory_token_dim} != 2048")
        _req((int(hc.budget), int(hc.token_per_image), int(hc.num_views)) == (512, 16, 1),
             f"(budget,token_per_image,num_views)=({hc.budget},{hc.token_per_image},"
             f"{hc.num_views}) != (512,16,1)")
        _req(int(hc.memory_feature.img.input_dim) == 2048,
             f"memory_feature.img.input_dim={hc.memory_feature.img.input_dim} != 2048")
        _req(int(hc.memory_feature.pos.input_dim) == 768,
             f"memory_feature.pos.input_dim={hc.memory_feature.pos.input_dim} != 768")
        _req(hc.use_state_emb is False, f"use_state_emb={hc.use_state_emb!r} 必须为 False")

        # ── motion memory（motion-memory-plan.md 2.1 / 2.6）：关闭态只判 enabled、不判子键（旧 yaml 缺整节照跑）──
        mcfg = getattr(hc, "motion", None)
        self._motion_enabled = bool(mcfg is not None and mcfg.get("enabled", False))
        if self._motion_enabled:
            _req(int(mcfg.dim) == ms.MOTION_ROW_SHAPE[0], f"motion.dim={mcfg.dim} != {ms.MOTION_ROW_SHAPE[0]}")
            _req(int(mcfg.budget) == 96, f"motion.budget={mcfg.budget} != 96")
            _req(int(mcfg.pos_dim) == int(hc.memory_feature.pos.input_dim) // 3,
                 f"motion.pos_dim={mcfg.pos_dim} != pos.input_dim // 3 = {int(hc.memory_feature.pos.input_dim) // 3}")
            _req(int(mcfg.stride) >= 1, f"motion.stride={mcfg.stride} < 1")
            _req(int(mcfg.stride) == ms.GRID_STRIDE, f"motion.stride={mcfg.stride} != motion store GRID_STRIDE {ms.GRID_STRIDE}")
            _req(int(mcfg.window_frames) == ms.WINDOW_FRAMES, f"motion.window_frames={mcfg.window_frames} != {ms.WINDOW_FRAMES}")
            _req(str(mcfg.window_direction) == ms.WINDOW_DIRECTION, f"motion.window_direction={mcfg.window_direction!r}")
            _req(str(mcfg.grid_origin) == ms.GRID_ORIGIN, f"motion.grid_origin={mcfg.grid_origin!r}")
            _req(int(mcfg.frame_size) == ms.FRAME_SIZE, f"motion.frame_size={mcfg.frame_size} != {ms.FRAME_SIZE}")
            _req(motion_root is not None, "motion.enabled=true 但未给 motion_root")

        if verify_level not in ("fast", "full"):
            raise ValueError(f"verify_level 非法: {verify_level!r}（∈ fast|full）")

        # ―― 主进程静态校验（fast 档；不开任何 fd/mmap，B.2）――
        self._root = pathlib.Path(dataset_path)
        self._meta = StoreMeta.load(self._root)
        self._manifest_path = str(manifest_path or self._meta.manifest_path)
        self._source_root = str(source_root or self._meta.source_dataset_root)
        run_fast_checks(self._meta, manifest_path=self._manifest_path,
                        source_root=self._source_root)
        if verify_level == "full":
            # full 档只允许独立 preflight 场景（禁性能 allocation，B.2）；worker 侧恒 fast
            run_full_checks(self._meta)

        # ―― 清单派生查表数组（唯一身份来源，R6）――
        manifest = load_manifest(self._manifest_path)
        num_eps = len(self._meta.subset_episodes) \
            if self._meta.manifest_scope == "subset" else None
        self._epis_of, self._step_of, self._row_base = \
            build_exec_lookup(manifest, num_episodes=num_eps)
        if len(self._epis_of) != self._meta.num_exec_samples:
            raise ValueError(
                f"清单派生样本数 {len(self._epis_of)} != meta.num_exec_samples "
                f"{self._meta.num_exec_samples}")

        self.action_horizon = action_horizon
        self.state_norm_stats = data_config.norm_stats["state"]
        self.use_quantiles = data_config.use_quantile_norm
        self._max_frames = int(hc.budget) // (int(hc.token_per_image) * int(hc.num_views))
        self._tokens_per_frame = int(hc.token_per_image) * int(hc.num_views)

        self._store: FrameSampStore | None = None
        self._atexit_registered = False

        # ── motion store（主进程只读 meta 与静态校验；整表 np.fromfile 在 worker 内懒构造）──
        self._mstore = None
        self._motion_root = None
        self._motion_meta = None
        self._motion_entries = None
        if self._motion_enabled:
            self._motion_root = pathlib.Path(motion_root)
            self._motion_meta = ms.MotionMeta.load(self._motion_root)
            ms.run_fast_checks(self._motion_meta, manifest_path=self._manifest_path)
            # 双 store 同源硬闸：同一份清单 + 逐 episode 身份互校（R23）
            ms.check_same_source(self._meta.manifest_sha256, self._motion_meta, manifest)
            if num_eps is not None:
                raise ValueError("motion memory 不支持 subset 迷你库（motion_index 覆盖全清单）")
            self._motion_entries = self._motion_meta.entries
            self._motion_budget = int(mcfg.budget)
            self._motion_pos_dim = int(mcfg.pos_dim)
            # 零截断契约的预检：每 episode 的最大合法起点数 ≤ 预算，超过即报 episode 身份（禁止静默裁剪）
            for e in self._motion_entries:
                mx = ms.max_visible_count(e)
                if mx > self._motion_budget:
                    raise ValueError(
                        f"episode g={e.g} ({e.h5_file}#{e.raw_ep_idx}) 最大合法起点数 {mx} > motion.budget "
                        f"{self._motion_budget}——零截断契约要求按全集重新定标，不做最近 N 裁剪")

    # ―― spawn 生命周期（B.2）――
    def __getstate__(self):
        d = dict(self.__dict__)
        d["_store"] = None              # 不携带任何 fd/小表进 spawn worker
        d["_mstore"] = None             # motion 整表同样不跨进程携带
        d["_atexit_registered"] = False
        return d

    def _ensure_motion_store(self):
        pid = os.getpid()
        s = self._mstore
        if s is not None and s.owner_pid == pid:
            return s
        if s is not None:
            s.close()
            self._mstore = None
        s = ms.MotionStore(self._motion_root, meta=self._motion_meta, manifest_path=self._manifest_path)
        self._mstore = s
        if not self._atexit_registered:
            atexit.register(self.close)
            self._atexit_registered = True
        return s

    def _ensure_store(self) -> FrameSampStore:
        pid = os.getpid()
        s = self._store
        if s is not None and s.owner_pid == pid:
            return s                    # 同 pid 复用（含 w0 重复建 loader 场景）
        if s is not None:
            s.close()                   # pid 失配：先显式关旧句柄再替换引用（B.2 ①）
            self._store = None
        s = FrameSampStore(self._root, meta=self._meta,
                           manifest_path=self._manifest_path,
                           source_root=self._source_root)
        self._store = s
        if not self._atexit_registered:
            atexit.register(self.close)  # 兜底（B.2 ②）
            self._atexit_registered = True
        return s

    def close(self) -> None:
        s = self._store
        self._store = None
        if s is not None:
            s.close()
        m = self._mstore
        self._mstore = None
        if m is not None:
            m.close()

    def __len__(self) -> int:
        return len(self._epis_of)

    # ―― 装配（与旧路径逐字对齐处已注明）――
    def _normalize_state(self, state):
        # 与旧 RoboMMEDataset._normalize_state（commitV4.1 已删）逐字同式（q01/q99 为 f64，输出恒 f64）
        if self.use_quantiles:
            return (state - self.state_norm_stats.q01) / (
                self.state_norm_stats.q99 - self.state_norm_stats.q01 + 1e-6) * 2.0 - 1.0
        else:
            return (state - self.state_norm_stats.mean) / (self.state_norm_stats.std + 1e-6)

    def _pad(self, img, pos, stt, n: int):
        """单一实现（B.3）：按最终形状一次性预分配，填充区清零，全程零 concatenate。

        交付 dtype 与旧路径 right_padding_token_emb 逐键一致（image bf16 / pos f32 /
        stt f32——填充零的 bf16/f32 位型同为全零字节）；n = 实际帧数，目标长度
        _max_frames(32) 是内部常量。
        """
        m = self._max_frames
        if n > m:
            raise ValueError(f"实际帧数 {n} > 目标长度 {m}（even_sampling 契约被破坏）")
        out_img = np.empty((m,) + img.shape[1:], dtype=img.dtype)
        out_pos = np.empty((m,) + pos.shape[1:], dtype=pos.dtype)
        out_stt = np.empty((m,) + stt.shape[1:], dtype=stt.dtype)
        out_img[:n] = img
        out_pos[:n] = pos
        out_stt[:n] = stt
        out_img[n:] = 0
        out_pos[n:] = 0
        out_stt[n:] = 0
        mask = np.zeros(m, dtype=np.bool_)
        mask[:n] = True
        return out_img, out_pos, out_stt, mask

    def _pad_motion(self, emb, pos, frames, k: int):
        """运动路右填充（另写、不复用 _pad：目标长度是 motion.budget、签名是 emb/pos 两键并附带每行全域时刻）。

        只负责 padding、绝不裁剪：k > 预算即 raise（__init__ 已按 index 预检，此处是防御性 overflow 闸）。
        返回 (emb (B,768) f32, pos (B,256) f32, mask (B,) bool, times (B,) int64 padding 记哨兵)。
        """
        B = self._motion_budget
        if k > B:
            raise ValueError(f"合法 motion 起点数 {k} > motion.budget {B}（零截断契约被破坏）")
        out_emb = np.zeros((B,) + emb.shape[1:], dtype=np.float32)
        out_pos = np.zeros((B,) + pos.shape[1:], dtype=np.float32)
        out_emb[:k] = emb
        out_pos[:k] = pos
        mask = np.zeros(B, dtype=np.bool_)
        mask[:k] = True
        times = pad_times(frames, B)
        return out_emb, out_pos, mask, times

    def __getitem__(self, idx):
        idx = int(idx)
        if not 0 <= idx < len(self._epis_of):
            raise IndexError(f"样本 idx 越界: {idx} ∉ [0, {len(self._epis_of)})")
        store = self._ensure_store()
        g = int(self._epis_of[idx])
        step = int(self._step_of[idx])

        with open(os.path.join(self._source_root, "data", f"{idx}.pkl"), "rb") as f:
            data = pickle.load(f)
        pkl_g = int(data["epis_idx"].item())
        pkl_step = int(data["step_idx"].item())
        if pkl_g != g or pkl_step != step:
            # 行号错位的最后一道闸；显式 raise（PYTHONOPTIMIZE=1 会剥离 assert，R6）
            raise RuntimeError(
                f"身份互校失败: data/{idx}.pkl 记 (epis={pkl_g}, step={pkl_step}) != "
                f"清单推导 ({g}, {step})——行号错位或清单/源库不配套")

        data["actions"] = data["actions"][: self.action_horizon]   # 与旧路径逐字相同
        data.pop("simple_subgoal_online")
        data.pop("grounded_subgoal_online")

        frames = even_sampling_indices(step, self._max_frames)     # 同一函数 import（R4）
        frames_arr = np.asarray(frames, dtype=np.int64)
        rows = self._row_base[g] + frames_arr
        img = store.read_image_rows(rows)          # (n,16,2048) bf16——0 open、0 线程池
        pos = store.pos_rows(frames_arr)           # (n,16,768) f32，进程内小表
        stt = store.state_rows(rows)               # (n,8) f32
        n = len(frames)
        img, pos, stt, mask = self._pad(img, pos, stt, n)

        # 与旧路径 _prepare_frame_sampling 的 reshape/repeat 逐字对齐：
        # (32,16,2048)→(512,2048) 与旧 (32,1,16,2048)→(512,2048) 字节相同（C-order）
        data["static_image_emb"] = img.reshape(-1, img.shape[-1])
        data["static_pos_emb"] = pos.reshape(-1, pos.shape[-1])
        data["static_state_emb"] = self._normalize_state(
            np.repeat(stt, self._tokens_per_frame, axis=0))
        data["static_mask"] = np.repeat(mask, self._tokens_per_frame)

        if self._motion_enabled:
            # ①′ 起点集合（段内绝对网格、前视 33 帧、尾端 ≤ 当前帧；三侧同式 visible_motion_rows）
            entry = self._motion_entries[g]
            rows_m, f_m = ms.visible_motion_rows(entry, step)
            k = int(len(rows_m))
            if k > self._motion_budget:
                raise ValueError(
                    f"样本 idx={idx} (g={g}, t={step}) 合法 motion 起点数 {k} > motion.budget {self._motion_budget}")
            mstore = self._ensure_motion_store()
            # ②′ 查表：motion 行 (k,768) f32；起点帧 pos 行第 0 行前 pos_dim 维（时间码，纯切片）
            memb = mstore.rows(rows_m) if k else np.zeros((0, ms.MOTION_ROW_SHAPE[0]), np.float32)
            mpos = (np.ascontiguousarray(store.pos_rows(f_m)[:, 0, : self._motion_pos_dim])
                    if k else np.zeros((0, self._motion_pos_dim), np.float32))
            # ③′ 右填充 + ④′ 两路按 (全域时刻, 类型) 交错
            memb, mpos, mmask, mtimes = self._pad_motion(memb, mpos, f_m, k)
            ftimes = pad_times(frames_arr, self._max_frames)
            data["motion_emb"] = memb
            data["motion_pos"] = mpos
            data["motion_mask"] = mmask
            data["mem_order"] = memory_order(ftimes, self._tokens_per_frame, mtimes)

        for key in _NONE_KEYS:                     # 与旧路径尾部补空键逐字一致
            if key not in data:
                data[key] = None
        return data
