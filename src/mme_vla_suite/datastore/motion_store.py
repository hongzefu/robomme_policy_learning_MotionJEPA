"""motion 离线表：格式常量、motion_index 契约、MotionMeta、只读 MotionStore、起点查表公式。

对应 0901-motion-memory-plan.md 第二部分一节 1.1（离线 motion 表格式契约）。体例照 ``framesamp_store.py``：
本文件是格式层唯一实现——打包工具（scripts/dataset/pack_motion_store.py）、FrameSampDataset（S2）、
在线侧（S3）与对拍工具一律从这里 import，绝不复制；``framesamp_store.py`` 一字不动（两套索引公式不同，
帧路按 ``row_of()`` 逐帧、运动路按段内网格，混放会互相污染）。

写入 ``motion-v10-8x256-grid16-demopad17-v1``，同时保留两个历史 768 布局的读取：

    <motion_root>/
    ├── meta/store_meta.json          唯一契约，两阶段写：pack→"packed"、verify→"verified"
    ├── meta/motion_index.json        段基址表（唯一身份来源），store_meta 记其 sha256、加载时现场重算
    ├── meta/row_digests.blake2b.bin  逐行 blake2b-128（verify 产出）
    ├── meta/pack_progress.jsonl      断点续跑记录
    └── motion_token.f32.bin          (rows, *row_shape) f32 裸字节；行序 = 清单 canonical_order 逐 episode，
                                      每 episode 先 demo 段后 exec 段，段内按网格序 0,16,32,… 升序

起点钉在段内网格 ``0,16,32,…``，demo / exec 各自起算。历史布局两段均要求 33 个真实帧；
新布局 demo 至少 17 个真实帧，不足 33 时重复本段最后一帧，exec 始终要求 33 帧。
窗口契约随 ``IndexEntry.spec`` 与 ``MotionMeta.spec`` 传递，禁止用最新布局替代旧表契约。

读取实现：每进程按该布局的 ``prod(row_shape)`` 整表 ``np.fromfile`` 读入，
仍照抄 FrameSampStore 的三条纪律——记录 ``owner_pid``、``__reduce__`` 直接 raise 禁 pickle、跨进程懒构造
（懒构造由 FrameSampDataset 负责）。本模块不 import 任何 training/model 模块（单向依赖）。

沿用 framesamp 的禁 ``.npy`` 容器定论：一律裸 ``.bin`` + meta 声明 dtype。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import random
import re

import numpy as np

from mme_vla_suite.datastore.manifest import load_manifest

logger = logging.getLogger(__name__)

# ── 布局常量 ──────────────────────────────────────────────────────────────────
LAYOUT = "motion-v10-8x256-grid16-demopad17-v1"
STAGE2_LAYOUT = "motion-stage2-1x1024-grid32-demopad17-v1"
STAGE2_ARCH = "wan-latent-stage2-blockwise-pool4-three-decoder"
STAGE2_GROUPS = ("encoder", "encoder_state_sha256", "source_pin", "numerics", "stage2_compat", "vae")
STAGE2_CACHE_ORIGIN = "published-cache-with-local-fill"
WAN_CACHE_IMPORT_SCHEMA = "orig16-wan-cache-import-v1"
WAN_CHUNK_BYTES = 9 * 16 * 32 * 32 * 4
SAME_INPUT_NOT_REQUIRED = {"status": "NOT_REQUIRED", "reason": "user_decision_same_input"}
STAGE2_SAMPLE_STRATEGY = "task64-endpoints-pad8-seed0-v1"
STAGE2_SAMPLE_AUTHORIZATION = "同意按照你说过的更省时方案 做\nencoder只抽机小部分"
STAGE2_G2_WAIVER_CHECKPOINT_SHA256 = "27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035"
STAGE2_G2_WAIVER_SAMPLES_SHA256 = "4cb05b6b3097b22a8594576def1505957399a924d4189591980292e3bd320d30"
STAGE2_G2_WAIVER_FAILURE_SHA256 = "88cc88fb69827f00763b5499e3aa3d39bdb411db3e186660eed596716d6dce72"
META_SCHEMA = 1
INDEX_SCHEMA = 1

MOTION_KEY = "motion_token"
MOTION_ROW_SHAPE = (8, 256)
MOTION_DTYPE = np.float32
MOTION_ROW_BYTES = int(np.prod(MOTION_ROW_SHAPE)) * np.dtype(MOTION_DTYPE).itemsize

WINDOW_FRAMES = 33                      # 与 MotionJEPA 的 WINDOW = 4*K+1 同值，verify 时核对
GRID_STRIDE = 16                        # 段内绝对网格步长；加载时须 == yaml motion.stride
GRID_ORIGIN = "segment_start"           # 网格锚点：每段各自从段起点起算，两段互不延续
WINDOW_DIRECTION = "forward"            # 前视：窗口 = [起点, 起点+32]
TRUNCATION_POLICY = "none"              # exec 段不截尾：num_chunks = max(0, 段帧数 − 32)
FRAME_SIZE = 256                        # 原始帧边长 h == w；离线抽取输入与在线 add_buffer 入库校验同用
LAYOUT_GRID_SUFFIX = f"grid{GRID_STRIDE}"

BYTE_ORDER = "little"
ARRAY_ORDER = "C"

META_RELPATH = "meta/store_meta.json"
INDEX_RELPATH = "meta/motion_index.json"
LOCK_RELPATH = "meta/pack.lock"
PROGRESS_RELPATH = "meta/pack_progress.jsonl"
ROW_DIGESTS_RELPATH = "meta/row_digests.blake2b.bin"
MOTION_TABLE_RELPATH = "motion_token.f32.bin"

ROW_DIGEST_BYTES = 16                   # blake2b-128
ROW_DIGEST_COVERAGE = "motion_token 行原始位串（store 侧字节）"

SEGMENTS = ("demo", "exec")             # 行序内 episode 内的段顺序

@dataclasses.dataclass(frozen=True)
class LayoutSpec:
    """布局自身的窗口契约；随索引一起传递，可哈希且可被 worker pickle。"""

    demo_min_real: int
    exec_min_real: int
    demo_tail_pad: str
    row_shape: tuple[int, ...] = (768,)
    grid_stride: int = 16

    @property
    def row_bytes(self) -> int:
        return int(np.prod(self.row_shape)) * np.dtype(MOTION_DTYPE).itemsize

    def min_real(self, segment: str) -> int:
        if segment not in SEGMENTS:
            raise ValueError(f"未知段类型 {segment!r}")
        return self.demo_min_real if segment == "demo" else self.exec_min_real

    def fields(self) -> dict:
        return {"demo_min_real_frames": self.demo_min_real,
                "exec_min_real_frames": self.exec_min_real,
                "demo_tail_pad": self.demo_tail_pad}


LAYOUT_SPECS = {
    "motion-768-grid16-v1": LayoutSpec(33, 33, "none"),
    "motion-768-grid16-demopad17-v1": LayoutSpec(17, 33, "repeat_last"),
    "motion-v10-8x256-grid16-demopad17-v1": LayoutSpec(17, 33, "repeat_last", (8, 256)),
    STAGE2_LAYOUT: LayoutSpec(17, 33, "repeat_last", (1, 1024), grid_stride=32),
}
if LAYOUT not in LAYOUT_SPECS or any(
    f"-grid{spec.grid_stride}-" not in name or not name.endswith("-v1") for name, spec in LAYOUT_SPECS.items()
):
    raise RuntimeError(f"motion 布局名与 GRID_STRIDE={GRID_STRIDE} 不符")

DEMO_MIN_REAL_FRAMES = LAYOUT_SPECS[LAYOUT].demo_min_real
EXEC_MIN_REAL_FRAMES = LAYOUT_SPECS[LAYOUT].exec_min_real
DEMO_TAIL_PAD = LAYOUT_SPECS[LAYOUT].demo_tail_pad


def layout_spec(raw: dict, ctx: str) -> LayoutSpec:
    """新布局要求三键齐备；历史布局允许三键全缺，部分缺失或错值一律拒绝。"""
    name = raw.get("layout")
    if name not in LAYOUT_SPECS:
        raise ValueError(f"{ctx} 未知 layout={name!r}")
    spec = LAYOUT_SPECS[name]
    fields = spec.fields()
    present = {key for key in fields if key in raw}
    if not present and name == "motion-768-grid16-v1":
        return spec
    if present != set(fields):
        raise ValueError(f"{ctx} 布局契约缺字段 {sorted(set(fields) - present)}")
    if any(type(raw[key]) is not type(value) or raw[key] != value for key, value in fields.items()):
        raise ValueError(f"{ctx} 三键与布局 {name} 不符，期望 {fields}")
    return spec


# ── 网格公式（写读共用；训练侧 / 在线侧 / oracle 三方同式）───────────────────────


def seg_num_chunks(seg_len: int, min_real: int = 33) -> int:
    """至少剩余 min_real 个真实帧的起点数；exec 恒取 33。"""
    if not 1 <= min_real <= WINDOW_FRAMES:
        raise ValueError(f"min_real={min_real} 不在 [1,{WINDOW_FRAMES}]")
    return max(0, int(seg_len) - (min_real - 1))


def seg_num_grid(seg_len: int, min_real: int = 33, stride: int = GRID_STRIDE) -> int:
    """段内网格起点数；默认步长16保留旧调用口径。"""
    if type(stride) is not int or stride <= 0:
        raise ValueError("网格步长必须为正整数")
    return len(range(0, seg_num_chunks(seg_len, min_real), stride))


def segment_lengths(num_timesteps: int, exec_start_idx: int) -> dict[str, int]:
    """demo 段帧数 = exec_start_idx；exec 段帧数 = num_timesteps − exec_start_idx。"""
    es = int(exec_start_idx)
    nt = int(num_timesteps)
    if not 0 <= es <= nt:
        raise ValueError(f"exec_start_idx={es} 越界（num_timesteps={nt}）")
    return {"demo": es, "exec": nt - es}


def segment_grid_starts(seg_len: int, min_real: int = 33, stride: int = GRID_STRIDE) -> list[int]:
    """按指定步长返回段内起点列表；默认步长16。"""
    if type(stride) is not int or stride <= 0:
        raise ValueError("网格步长必须为正整数")
    return list(range(0, seg_num_chunks(seg_len, min_real), stride))


def visible_motion_rows(entry: "IndexEntry", t: int) -> tuple[np.ndarray, np.ndarray]:
    """给定 episode 的 index 条目与当前样本全域帧号 t，返回 (rows, frames)：

    - demo 段：``s = 16m``、合法条件 ``s + spec.demo_min_real − 1 ≤ es − 1``（整段已见、与 t 无关）；
    - exec 段：``u = 16m``、合法条件 ``u + 32 ≤ t − es``，全域起点 ``f = es + u``；
    合并后按 f 升序。rows 是 motion 表全局行号（int64），frames 是全域起点帧号（int64）。
    预算（motion.budget）上限检查不在这里做——由调用方按配置 raise，本函数只负责集合本身。
    """
    es = int(entry.exec_start_idx)
    t = int(t)
    if not es <= t < entry.num_timesteps:
        raise ValueError(
            f"t={t} 不在 exec 段 [{es}, {entry.num_timesteps}) 内（g={entry.g}）")
    rows: list[int] = []
    frames: list[int] = []
    if entry.demo.row_base is not None:
        for m in range(entry.demo.num_grid):
            s = entry.spec.grid_stride * m
            if s + (entry.spec.demo_min_real - 1) <= es - 1:
                rows.append(entry.demo.row_base + m)
                frames.append(s)
    if entry.exec.row_base is not None:
        for m in range(entry.exec.num_grid):
            u = entry.spec.grid_stride * m
            if u + (WINDOW_FRAMES - 1) <= t - es:
                rows.append(entry.exec.row_base + m)
                frames.append(es + u)
    r = np.asarray(rows, dtype=np.int64)
    f = np.asarray(frames, dtype=np.int64)
    if len(f) > 1 and not np.all(np.diff(f) > 0):
        order = np.argsort(f, kind="stable")
        r, f = r[order], f[order]
    return r, f


def max_visible_count(entry: "IndexEntry", stride: int | None = None) -> int:
    """该 episode 任一 exec 样本的合法起点数上界（t = 最后一帧时取到）。"""
    if stride is not None and stride != entry.spec.grid_stride:
        raise ValueError("max_visible_count 的显式步长与索引布局不同")
    r, _ = visible_motion_rows(entry, entry.num_timesteps - 1)
    return int(len(r))


# ── 通用小件 ──────────────────────────────────────────────────────────────────


def sha256_file(p: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_json_bytes(value) -> bytes:
    """跨进程身份的唯一 JSON 编码；拒绝非有限数。"""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def require_sha256(value, ctx: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{ctx} 必须为 64 位小写 SHA256")
    return value


def stage2_uses_cache(provenance):
    """只对明确声明的新来源合同放宽跨精度输出比较。"""
    return provenance.get("stage2_compat", {}).get("kind") == STAGE2_CACHE_ORIGIN


def stage2_evidence_roles(provenance):
    return ("G0", "G2", "G3", "wan_cache_import" if stage2_uses_cache(provenance) else "restore_receipt", "SOURCE_PIN")


def validate_wan_cache_import_receipt(receipt, *, manifest_sha256=None, input_manifest_sha256=None):
    """校验发布缓存导入的可携带合同；文件内容由生产与打包入口现场核验。"""
    if (not isinstance(receipt, dict) or receipt.get("schema") != WAN_CACHE_IMPORT_SCHEMA
            or receipt.get("kind") != STAGE2_CACHE_ORIGIN or receipt.get("status") != "PASS"):
        raise ValueError("Wan 缓存收据 schema、kind 或状态不符")
    for key in ("manifest_sha256", "manifest_file_sha256", "input_manifest_sha256", "importer_sha256"):
        require_sha256(receipt.get(key), f"cache.{key}")
    for key, expected in (("manifest_sha256", manifest_sha256), ("input_manifest_sha256", input_manifest_sha256)):
        if expected is not None and receipt[key] != expected:
            raise ValueError(f"缓存收据的 {key} 与当前输入不同")

    def files(items, label, name_key="path"):
        result = {}
        if not isinstance(items, list) or not items:
            raise ValueError(f"{label} 文件清单为空")
        for item in items:
            name = item.get(name_key) if isinstance(item, dict) else None
            if (not isinstance(name, str) or not name or pathlib.PurePosixPath(name).is_absolute()
                    or ".." in pathlib.PurePosixPath(name).parts or name in result
                    or type(item.get("bytes")) is not int or item["bytes"] <= 0):
                raise ValueError(f"{label} 文件路径、字节或唯一性不符")
            require_sha256(item.get("sha256"), f"{label}.{name}")
            result[name] = item
        return result

    published = receipt.get("published", {})
    if (not isinstance(published.get("repo"), str) or not published["repo"]
            or re.fullmatch(r"[0-9a-f]{40}", str(published.get("revision"))) is None):
        raise ValueError("缓存发布 repo/revision 不完整")
    require_sha256(published.get("sha256sums_sha256"), "published.SHA256SUMS")
    source_files = files(published.get("files"), "published")
    h5_files = files(receipt.get("source_h5_files"), "source_h5", "name")
    if any(pathlib.PurePosixPath(name).name != name or not name.endswith(".h5") for name in h5_files):
        raise ValueError("源 H5 清单必须以原文件名标识")
    latents = receipt.get("latents", {})
    if not isinstance(latents.get("root"), str) or not latents["root"]:
        raise ValueError("缓存收据缺目标 latent 根")
    require_sha256(latents.get("metadata_sha256"), "latents.metadata_sha256")
    output_files = files(latents.get("files"), "latents")
    vae = receipt.get("vae", {})
    if vae.get("vae_id") != "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" or vae.get("latent_mode") != "mode":
        raise ValueError("缓存 VAE 模型或 latent_mode 不符")
    require_sha256(vae.get("vae_state_sha256"), "cache.vae_state_sha256")
    if receipt.get("preprocess") != dict(frame_size=256, input_dtype="uint8", input_layout="THWC",
                                          window_frames=33, augmentation="none"):
        raise ValueError("缓存预处理或增强合同不符")
    modes = receipt.get("numeric_modes", {})
    if set(modes) != {"copy", "fill"} or any(not isinstance(v, dict) or not v for v in modes.values()):
        raise ValueError("缓存必须分别记录 copy/fill 实际数值环境")
    windows = receipt.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("缓存收据缺逐窗映射")
    mapping, outputs, counts = {}, {}, dict(windows=len(windows), copy=0, fill=0)
    for window in windows:
        if not isinstance(window, dict):
            raise ValueError("缓存窗口必须为对象")
        key, m, route = window.get("key"), window.get("m"), window.get("route")
        if (not isinstance(key, str) or re.fullmatch(r".+_ep[0-9]+_(demo|exec)", key) is None
                or type(m) is not int or m < 0 or (key, m) in mapping or route not in ("copy", "fill")):
            raise ValueError("缓存窗口键、网格或路线缺失/重复")
        real, pad, frames = window.get("real_frames"), window.get("pad_frames"), window.get("frame_ids")
        if (type(real) is not int or type(pad) is not int or real + pad != 33 or pad < 0
                or real < (17 if key.endswith("_demo") else 33) or not isinstance(frames, list)
                or len(frames) != 33 or any(type(f) is not int or f < 0 for f in frames)
                or frames != list(range(frames[0], frames[0] + real)) + [frames[real - 1]] * pad
                or window.get("pad_source_frame") != (frames[real - 1] if pad else None)):
            raise ValueError("缓存窗口帧号、补帧或真实帧数不符")
        for name in ("input_frames_sha256", "latent_sha256"):
            require_sha256(window.get(name), f"cache.window.{name}")
        output, offset = window.get("output_file"), window.get("output_offset")
        if (output not in output_files or output != f"{key}.bin" or type(offset) is not int
                or offset != m * WAN_CHUNK_BYTES or offset + WAN_CHUNK_BYTES > output_files[output]["bytes"]):
            raise ValueError("缓存目标文件、offset 或边界不符")
        if route == "copy":
            source, row, offset = window.get("source_file"), window.get("source_row"), window.get("source_offset")
            if (source not in source_files or type(row) is not int or row < 0 or type(offset) is not int
                    or offset != row * WAN_CHUNK_BYTES or offset + WAN_CHUNK_BYTES > source_files[source]["bytes"]
                    or pathlib.PurePosixPath(source).name != f"{key}.bin" or row != m * 16
                    or window.get("source_layout") not in ("group_first", "channel_first")):
                raise ValueError("缓存源 row/offset/轴序或文件边界不符")
        elif any(window.get(name) is not None for name in ("source_file", "source_row", "source_offset", "source_layout")):
            raise ValueError("fill 窗口不得伪造发布缓存来源")
        if "h5_file" in window and window["h5_file"] not in h5_files:
            raise ValueError("窗口源 H5 不在收据清单")
        if key.endswith("_demo") and frames[0] != m * 16:
            raise ValueError("demo 缓存窗口起点与 grid16 不符")
        mapping[key, m] = window
        outputs.setdefault(output, []).append(m)
        counts[route] += 1
    if receipt.get("counts") != counts:
        raise ValueError("缓存 copy/fill 计数与逐窗映射不符")
    if {name for name in output_files if name.endswith(".bin")} != set(outputs):
        raise ValueError("缓存最终文件缺映射或含多余段")
    for name, rows in outputs.items():
        if sorted(rows) != list(range(len(rows))) or output_files[name]["bytes"] != len(rows) * WAN_CHUNK_BYTES:
            raise ValueError("缓存逐段映射有缺窗、重窗或额外字节")
    return mapping


def validate_stage2_g2_evidence(record, *, checkpoint_sha256, samples_sha256,
                               source_pin, numerics, online_vae, demo_tail=None,
                               cache_import_receipt=None, cache_import_receipt_sha256=None) -> str:
    """只接受严格数值PASS，或用户针对固定三份身份授权的数值例外；来源检查始终执行。"""
    if cache_import_receipt is not None:
        receipt = cache_import_receipt
        validate_wan_cache_import_receipt(receipt)
        require_sha256(cache_import_receipt_sha256, "G2.cache_import_receipt_sha256")
        if (record.get("schema") != "stage2-validation-gate-v1" or record.get("gate") != "G2"
                or record.get("status") != "PASS" or record.get("kind") != STAGE2_CACHE_ORIGIN
                or record.get("samples_sha256") != samples_sha256
                or record.get("cache_import_receipt_sha256") != cache_import_receipt_sha256
                or record.get("output_numerics") != SAME_INPUT_NOT_REQUIRED):
            raise ValueError("缓存 G2 的证据绑定或数值 NOT_REQUIRED 声明不符")
        for key, expected in (("manifest_sha256", receipt["manifest_sha256"]),
                              ("latent_metadata_sha256", receipt["latents"]["metadata_sha256"]),
                              ("vae", receipt["vae"]), ("preprocess", receipt["preprocess"]),
                              ("numeric_modes", receipt["numeric_modes"])):
            if record.get(key) != expected:
                raise ValueError(f"缓存 G2 的 {key} 不同源")
        if online_vae.get("vae_state_sha256") != receipt["vae"]["vae_state_sha256"]:
            raise ValueError("缓存与在线 VAE 权重不同源")
        metrics = record.get("metrics", {})
        fixed = dict(windows=64, sha_mismatches=0, frame_mismatches=0, shape_mismatches=0, nonfinite=0)
        if (any(type(metrics.get(k)) is not int or metrics[k] != v for k, v in fixed.items())
                or type(metrics.get("demo_tail")) is not int or metrics["demo_tail"] < 16
                or (demo_tail is not None and metrics["demo_tail"] != demo_tail)
                or any(type(metrics.get(k)) is not int or metrics[k] < 1 for k in ("copy", "demo_fill", "exec_fill"))):
            raise ValueError("缓存 G2 输入、形状、有限性或路线覆盖未通过")
        return "PASS"
    require_sha256(checkpoint_sha256, "G2.checkpoint_sha256")
    require_sha256(samples_sha256, "G2.samples_sha256")
    if (not isinstance(record, dict) or record.get("schema") != "stage2-validation-gate-v1"
            or record.get("gate") != "G2" or record.get("checkpoint_sha256") != checkpoint_sha256
            or record.get("samples_sha256") != samples_sha256):
        raise ValueError("G2 证据未绑定同一checkpoint与G0清单")
    pin_keys = {"mj_repo_commit", "export_source_sha256", "encoder_source_sha256",
                "vae_adapter_sha256", "dependency_lock_sha256"}
    if (not isinstance(source_pin, dict) or set(source_pin) != pin_keys
            or not isinstance(source_pin.get("mj_repo_commit"), str)
            or re.fullmatch(r"[0-9a-f]{40}", source_pin["mj_repo_commit"]) is None):
        raise ValueError("G2 来源必须为完整五键SOURCE_PIN")
    for key in pin_keys - {"mj_repo_commit"}:
        require_sha256(source_pin[key], f"G2.source_pin.{key}")
    if not isinstance(numerics, dict):
        raise ValueError("G2 数值环境必须为对象")
    for key, value in {"dtype": "float32", "autocast": False, "matmul_allow_tf32": False,
                       "cudnn_allow_tf32": False, "batch": 1}.items():
        if type(numerics.get(key)) is not type(value) or numerics[key] != value:
            raise ValueError(f"G2 实际encoder数值合同不符: {key}")
    if (any(not isinstance(numerics.get(k), str) or not numerics[k]
            for k in ("torch", "cuda", "numpy", "gpu_name", "driver"))
            or type(numerics.get("cudnn")) is not int or numerics["cudnn"] <= 0):
        raise ValueError("G2 实际encoder环境身份不完整")
    if not isinstance(online_vae, dict):
        raise ValueError("G2 在线VAE身份必须为对象")
    vae_fixed = {"vae_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "vae_dtype": "float32",
                 "latent_mode": "mode", "batch": 1, "tf32": "off", "amp": "off"}
    if any(type(online_vae.get(k)) is not type(v) or online_vae[k] != v for k, v in vae_fixed.items()):
        raise ValueError("G2 在线VAE固定模型或数值合同不符")
    require_sha256(online_vae.get("vae_state_sha256"), "G2.online_vae.vae_state_sha256")
    preprocess = dict(frame_size=256, input_dtype="uint8", input_layout="THWC", window_frames=33)
    if (online_vae.get("adapter_sha256") != source_pin["vae_adapter_sha256"]
            or canonical_json_bytes(online_vae.get("preprocess")) != canonical_json_bytes(preprocess)
            or canonical_json_bytes(online_vae.get("numerics")) != canonical_json_bytes(numerics)):
        raise ValueError("G2 在线VAE未绑定当前适配件、预处理或数值环境")
    for key, expected in (("source_pin", source_pin), ("numerics", numerics), ("online_vae", online_vae)):
        if canonical_json_bytes(record.get(key)) != canonical_json_bytes(expected):
            raise ValueError(f"G2 证据的{key}与实际来源不同")
    status = record.get("status")
    if status == "PASS":
        if "exception" in record or record.get("producer") == "stage2-g2-user-exception-v1":
            raise ValueError("G2 用户例外不能改标为PASS")
        metrics = record.get("metrics")
        if (not isinstance(metrics, dict) or type(metrics.get("windows")) is not int or metrics["windows"] != 64
                or type(metrics.get("demo_tail")) is not int or metrics["demo_tail"] < 16
                or (demo_tail is not None and metrics["demo_tail"] != demo_tail)
                or type(metrics.get("worst_mse")) not in (int, float) or not 0 <= metrics["worst_mse"] <= 1e-6
                or type(metrics.get("worst_max")) not in (int, float) or not 0 <= metrics["worst_max"] <= 0.01):
            raise ValueError("G2 数值证据不完整或超过原阈值")
        return "PASS"
    if status != "WAIVED" or record.get("producer") != "stage2-g2-user-exception-v1":
        raise ValueError("G2 未通过且不属于用户明确授权的例外")
    allowed = {"schema", "gate", "status", "producer", "checkpoint_sha256", "samples_sha256",
               "source_pin", "numerics", "online_vae", "exception", "metrics"}
    if set(record) - allowed or record.get("metrics") is not None:
        raise ValueError("G2 例外不得携带伪造的数值完成指标")
    if checkpoint_sha256 != STAGE2_G2_WAIVER_CHECKPOINT_SHA256 or samples_sha256 != STAGE2_G2_WAIVER_SAMPLES_SHA256:
        raise ValueError("G2 例外只能用于本次固定checkpoint和G0清单")
    exception = record.get("exception")
    fixed = dict(schema="stage2-g2-user-exception-v1", authorized_by="user", user_quote="可以忽略 继续做",
                 authorized_at="2026-10-26T19:21:45Z", scope="g2_cross_precision_numerics_only",
                 numeric_validation_complete=False, original_failure_sha256=STAGE2_G2_WAIVER_FAILURE_SHA256)
    if (not isinstance(exception, dict) or set(exception) != set(fixed) | {"original_failure_json"}
            or any(canonical_json_bytes(exception.get(k)) != canonical_json_bytes(v) for k, v in fixed.items())):
        raise ValueError("G2 用户原话、授权范围或固定失败身份不符")
    original = exception.get("original_failure_json")
    if not isinstance(original, str) or sha256_bytes(original.encode("utf-8")) != STAGE2_G2_WAIVER_FAILURE_SHA256:
        raise ValueError("G2 原始失败UTF8字节不保真")
    try:
        failure = json.loads(original)
    except json.JSONDecodeError as exc:
        raise ValueError("G2 原始失败不是完整JSON") from exc
    if not isinstance(failure, dict) or failure.get("gate") != "G2" or failure.get("status") != "FAIL":
        raise ValueError("G2 例外只能引用原始G2失败")
    return "WAIVED"


def validate_stage2_provenance(provenance: dict) -> None:
    """严格检查 stage2 六组来源；只验证身份，不依赖建库机器的绝对路径。"""
    if not isinstance(provenance, dict):
        raise ValueError("stage2 provenance 必须为对象")
    for group in STAGE2_GROUPS:
        if not isinstance(provenance.get(group), dict) or not provenance[group]:
            raise ValueError(f"stage2 provenance 缺完整 {group}")
    enc = provenance["encoder"]
    fields = ("run_name", "checkpoint_name", "epoch", "global_step", "state_key", "weight_type",
              "arch", "checkpoint_sha256", "config_sha256", "fingerprint_sha256", "source_times", "output_shape")
    if any(k not in enc or enc[k] is None for k in fields):
        raise ValueError("stage2 encoder 必填字段不全")
    fixed = {"state_key": "model", "weight_type": "ema", "arch": STAGE2_ARCH,
             "source_times": [0, 3, 5, 8], "output_shape": [1, 1024]}
    if any(canonical_json_bytes(enc[k]) != canonical_json_bytes(v) for k, v in fixed.items()):
        raise ValueError("stage2 encoder 架构、EMA 键或输出槽序不符")
    if (not isinstance(enc["run_name"], str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", enc["run_name"]) is None
            or type(enc["epoch"]) is not int or enc["epoch"] < 0
            or type(enc["global_step"]) is not int or enc["global_step"] < 0
            or enc["checkpoint_name"] != f"checkpoint_epoch_{enc['epoch']}.pt"):
        raise ValueError("stage2 encoder 轮次或名称不自洽")
    for k in ("checkpoint_sha256", "config_sha256", "fingerprint_sha256"):
        require_sha256(enc[k], f"encoder.{k}")
    state = provenance["encoder_state_sha256"]
    leaves = state.get("leaves")
    if state.get("algorithm") != "named-tensors-v1" or not isinstance(leaves, dict) or len(leaves) != 65:
        raise ValueError("stage2 encoder 状态摘要必须含 named-tensors-v1 的完整 65 项")
    for name, leaf in leaves.items():
        if (not isinstance(name, str) or not name or name.startswith("encoder.")
                or not isinstance(leaf, dict) or leaf.get("dtype") != "float32"
                or not isinstance(leaf.get("shape"), list)
                or any(type(d) is not int or d < 0 for d in leaf["shape"])):
            raise ValueError(f"stage2 encoder 状态项不合法: {name}")
        require_sha256(leaf.get("sha256"), f"encoder_state.{name}")
    if state.get("sha256") != sha256_bytes(canonical_json_bytes(leaves)):
        raise ValueError("stage2 encoder 状态清单摘要不自洽")
    pin = provenance["source_pin"]
    if not isinstance(pin.get("mj_repo_commit"), str) or re.fullmatch(r"[0-9a-f]{40}", pin["mj_repo_commit"]) is None:
        raise ValueError("stage2 source_pin 缺 40 位 MJ 提交")
    for k in ("export_source_sha256", "encoder_source_sha256", "vae_adapter_sha256", "dependency_lock_sha256"):
        require_sha256(pin.get(k), f"source_pin.{k}")
    numerics = provenance["numerics"]
    for k, value in {"dtype": "float32", "autocast": False, "matmul_allow_tf32": False,
                     "cudnn_allow_tf32": False, "batch": 1}.items():
        if type(numerics.get(k)) is not type(value) or numerics[k] != value:
            raise ValueError(f"stage2 numerics.{k} 不符")
    for k in ("torch", "cuda", "cudnn", "numpy", "gpu_name", "driver"):
        if numerics.get(k) in (None, ""):
            raise ValueError(f"stage2 numerics 缺 {k}")
    if type(numerics["cudnn"]) is not int or numerics["cudnn"] <= 0:
        raise ValueError("stage2 numerics.cudnn 必须为正整数版本")
    compat = provenance["stage2_compat"]
    if compat.get("schema") != "stage2-compat-v1":
        raise ValueError("stage2_compat schema 不符")
    if stage2_uses_cache(provenance):
        _validate_stage2_cache_provenance(provenance)
        if "encoder_validation" in compat:
            validate_stage2_encoder_validation(compat["encoder_validation"], provenance)
        canonical_json_bytes(provenance)
        return
    if compat.get("kind") not in (None, "historical-restore-receipt"):
        raise ValueError("stage2_compat 未知来源 kind")
    require_sha256(compat.get("restore_receipt_sha256"), "stage2_compat.restore_receipt_sha256")
    historical, online = compat.get("historical_vae"), compat.get("online_vae")
    if not isinstance(historical, dict) or not isinstance(online, dict) or not historical or not online:
        raise ValueError("stage2_compat 缺历史或在线 VAE 身份")
    vae_fixed = {"vae_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "vae_dtype": "float32",
                 "latent_mode": "mode", "batch": 1, "tf32": "off", "amp": "off"}
    if any(type(historical.get(k)) is not type(v) or historical[k] != v for k, v in vae_fixed.items()):
        raise ValueError("stage2_compat 历史 VAE 的固定模型、精度或数值合同不符")
    for k in ("vae_id", "vae_state_sha256", "vae_dtype", "latent_mode", "batch", "tf32", "amp"):
        if (k not in historical or canonical_json_bytes(historical[k]) != canonical_json_bytes(provenance["vae"].get(k))
                or canonical_json_bytes(historical[k]) != canonical_json_bytes(online.get(k))):
            raise ValueError(f"stage2_compat 的历史、在线 VAE {k} 不一致")
    require_sha256(historical["vae_state_sha256"], "vae.vae_state_sha256")
    for k in ("torch", "cuda", "cudnn", "diffusers", "module_sha256", "encoder_src_sha256"):
        if historical.get(k) in (None, "") or historical[k] != provenance["vae"].get(k):
            raise ValueError(f"stage2_compat 历史 VAE 缺少或不匹配 {k}")
    for k in ("module_sha256", "encoder_src_sha256"):
        require_sha256(historical[k], f"historical_vae.{k}")
    if online.get("adapter_sha256") != pin["vae_adapter_sha256"]:
        raise ValueError("stage2_compat 在线 VAE 适配件未绑定 source_pin")
    if online.get("preprocess") != {"frame_size": 256, "input_dtype": "uint8", "input_layout": "THWC", "window_frames": 33}:
        raise ValueError("stage2_compat 在线 VAE 预处理不符")
    if canonical_json_bytes(online.get("numerics")) != canonical_json_bytes(numerics):
        raise ValueError("stage2_compat 在线 VAE 的锁定环境与 encoder 数值环境不一致")
    evidence = compat.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("stage2_compat 缺 evidence")
    for gate in ("G0", "G2", "G3", "restore_receipt", "SOURCE_PIN"):
        record = evidence.get(gate)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not record["path"]:
            raise ValueError(f"stage2_compat 缺 {gate} 文件引用")
        require_sha256(record.get("sha256"), f"evidence.{gate}")
    if evidence["restore_receipt"]["sha256"] != compat["restore_receipt_sha256"]:
        raise ValueError("stage2_compat receipt 引用与身份不同")
    if "encoder_validation" in compat:
        validate_stage2_encoder_validation(compat["encoder_validation"], provenance)
    canonical_json_bytes(provenance)


def _validate_stage2_cache_provenance(provenance):
    """新缓存合同只区分数值环境，模型、权重与输入规范仍唯一。"""
    compat, vae, pin = provenance["stage2_compat"], provenance["vae"], provenance["source_pin"]
    online = compat.get("online_vae", {})
    if (vae.get("vae_id") != "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" or vae.get("latent_mode") != "mode"
            or any(online.get(key) != vae.get(key) for key in ("vae_id", "vae_state_sha256", "latent_mode"))):
        raise ValueError("缓存与在线 VAE 模型/权重/取值方式不同")
    require_sha256(vae.get("vae_state_sha256"), "cache.vae_state_sha256")
    fixed = dict(vae_dtype="float32", batch=1, tf32="off", amp="off")
    if (any(type(online.get(k)) is not type(v) or online[k] != v for k, v in fixed.items())
            or online.get("adapter_sha256") != pin["vae_adapter_sha256"]
            or online.get("preprocess") != dict(frame_size=256, input_dtype="uint8", input_layout="THWC", window_frames=33)
            or online.get("numerics") != provenance["numerics"]):
        raise ValueError("缓存合同的在线 VAE 精度、适配件或预处理不符")
    modes = compat.get("numeric_modes", {})
    if (set(modes) != {"copy", "fill"} or any(not isinstance(v, dict) or not v for v in modes.values())
            or compat.get("output_numerics") != SAME_INPUT_NOT_REQUIRED):
        raise ValueError("缓存合同缺 copy/fill 数值记录或明确 NOT_REQUIRED")
    require_sha256(compat.get("cache_import_receipt_sha256"), "cache_import_receipt_sha256")
    evidence = compat.get("evidence", {})
    if set(evidence) != set(stage2_evidence_roles(provenance)):
        raise ValueError("缓存合同证据集合不完整或混入历史收据")
    for label, item in evidence.items():
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"]:
            raise ValueError(f"缓存合同缺 {label} 路径")
        require_sha256(item.get("sha256"), f"cache.evidence.{label}")
    if evidence["wan_cache_import"]["sha256"] != compat["cache_import_receipt_sha256"]:
        raise ValueError("缓存收据引用与合同身份不符")


def validate_stage2_cache_evidence(provenance, records):
    """共用 G0/G2/G3 消费校验，不把跨精度输出差异伪标为 PASS。"""
    compat = provenance["stage2_compat"]
    receipt, g0, g3 = records["wan_cache_import"], records["G0"], records["G3"]
    mapping = validate_wan_cache_import_receipt(receipt, manifest_sha256=provenance.get("manifest_sha256"),
                                               input_manifest_sha256=provenance.get("input_manifest_sha256"))
    if (records["SOURCE_PIN"] != provenance["source_pin"] or receipt["vae"] != provenance["vae"]
            or receipt["numeric_modes"] != compat["numeric_modes"]):
        raise ValueError("缓存证据的权重、源码或数值记录与来源合同不同")
    counts, windows = g0.get("counts", {}), g0.get("windows", [])
    if (g0.get("schema") != "stage2-validation-inputs-v1" or g0.get("gate") != "G0"
            or g0.get("status") != "PASS" or g0.get("seed") != 0 or len(windows) != 64
            or counts.get("windows") != 64 or counts.get("tasks") != 16 or counts.get("sha_mismatches") != 0
            or g0.get("cache_import_receipt_sha256") != compat["cache_import_receipt_sha256"]
            or g0.get("manifest_sha256") != receipt["manifest_sha256"]
            or g0.get("latent_metadata_sha256") != receipt["latents"]["metadata_sha256"]):
        raise ValueError("缓存 G0 证据不完整或绑定不同输入")
    tasks, seen, routes, tails = {}, set(), set(), 0
    h5 = {f["name"]: f for f in receipt["source_h5_files"]}
    for window in windows:
        identity = (window.get("key"), window.get("source_grid"))
        row = mapping.get(identity)
        if row is None or identity in seen or type(identity[1]) is not int or identity[1] % 2:
            raise ValueError("缓存 G0 窗口重复、缺映射或不是偶数格")
        seen.add(identity)
        for key in ("frame_ids", "real_frames", "pad_frames", "pad_source_frame", "input_frames_sha256", "route"):
            if window.get(key) != row.get(key):
                raise ValueError(f"缓存 G0 窗口与导入收据不同: {key}")
        if window.get("latent_block_sha256") != row["latent_sha256"]:
            raise ValueError("缓存 G0 latent 块 SHA 不同")
        task, episode, segment = window.get("task"), window.get("episode"), window.get("segment_kind")
        if (not isinstance(task, str) or type(episode) is not int or episode < 0 or segment not in SEGMENTS
                or identity[0] != f"{task}_ep{episode}_{segment}"):
            raise ValueError("缓存 G0 的任务、episode 或段身份不符")
        sources = window.get("sources", [])
        if len(sources) == 1 and sources[0].get("kind") == "raw_h5":
            source = sources[0]
            expected_indices = [dict(source=0, index=f) for f in row["frame_ids"]]
        elif segment == "exec" and len(sources) == 33 and all(s.get("kind") == "exec_pkl" for s in sources):
            source = window.get("raw_h5", {})
            expected_indices = [dict(source=i, index=0) for i in range(33)]
            for item in sources:
                require_sha256(item.get("sha256"), "G0.exec_pkl.sha256")
                if not isinstance(item.get("path"), str) or not item["path"]:
                    raise ValueError("缓存 G0 exec_pkl 路径缺失")
        else:
            raise ValueError("缓存 G0 必须绑定原始 H5 与实际执行样本")
        if source.get("kind") != "raw_h5" or source.get("episode") != episode:
            raise ValueError("缓存 G0 原 H5 episode 不同")
        source_name = pathlib.Path(source.get("path", "")).name
        if (source_name not in h5 or source_name != f"record_dataset_{task}.h5"
                or source.get("sha256") != h5[source_name]["sha256"]
                or ("h5_file" in row and source_name != row["h5_file"])):
            raise ValueError("缓存 G0 源 H5 SHA 与收据不同")
        if window.get("source_frame_indices") != expected_indices:
            raise ValueError("缓存 G0 源帧映射与输入不符")
        tasks[task] = tasks.get(task, 0) + 1
        routes.add("copy" if row["route"] == "copy" else f"{segment}_fill")
        tails += int(row["pad_frames"] > 0)
    if (len(tasks) != 16 or set(tasks.values()) != {4} or tails < 16 or counts.get("demo_tail") != tails
            or routes != {"copy", "demo_fill", "exec_fill"}):
        raise ValueError("缓存 G0 每任务四窗、尾窗或三种路线覆盖不完整")
    validate_stage2_g2_evidence(records["G2"], checkpoint_sha256=provenance["encoder"]["checkpoint_sha256"],
        samples_sha256=compat["evidence"]["G0"]["sha256"], source_pin=provenance["source_pin"],
        numerics=provenance["numerics"], online_vae=compat["online_vae"], demo_tail=tails,
        cache_import_receipt=receipt, cache_import_receipt_sha256=compat["cache_import_receipt_sha256"])
    metrics = g3.get("metrics", {})
    fixed = dict(windows=32, seed=0, geometry_all_true=1, leaves=65, state_key="model", exact=1,
                 live_negative_rejected=1, shape_mismatches=0, nonfinite=0)
    if (g3.get("schema") != "stage2-validation-gate-v1" or g3.get("gate") != "G3" or g3.get("status") != "PASS"
            or g3.get("kind") != STAGE2_CACHE_ORIGIN or g3.get("output_numerics") != SAME_INPUT_NOT_REQUIRED
            or g3.get("cache_import_receipt_sha256") != compat["cache_import_receipt_sha256"]
            or g3.get("samples_sha256") != compat["evidence"]["G0"]["sha256"]
            or g3.get("checkpoint_sha256") != provenance["encoder"]["checkpoint_sha256"]
            or any(metrics.get(k) != v for k, v in fixed.items())
            or any(g3.get(k) != provenance[k] for k in ("source_pin", "numerics", "encoder_state_sha256"))):
        raise ValueError("缓存 G3 的 EMA、输入绑定、形状或有限性证据不完整")


def validate_stage2_encoder_validation(summary, provenance, *, total_rows=None,
                                       table_sha256=None, index_sha256=None):
    """校验随快照携带的认证原文与覆盖范围；交付环境不依赖原机器索引。"""
    if not isinstance(summary, dict) or not isinstance(summary.get("record_json"), str):
        raise ValueError("encoder认证摘要缺完整UTF8原文")
    if sha256_bytes(summary["record_json"].encode("utf-8")) != require_sha256(summary.get("record_sha256"), "encoder认证原文SHA"):
        raise ValueError("encoder认证原文SHA不符")
    try:
        record = json.loads(summary["record_json"])
    except json.JSONDecodeError as exc:
        raise ValueError("encoder认证原文不可解析") from exc
    if stage2_uses_cache(provenance):
        rows = summary.get("total_rows")
        fixed = dict(schema="stage2-packed-integrity-v1", scope="full", layout=STAGE2_LAYOUT, status="PASS",
                     compared=rows, packed_compared=rows, mismatches=0, nonfinite=0,
                     output_numerics=SAME_INPUT_NOT_REQUIRED,
                     checkpoint_sha256=provenance["encoder"]["checkpoint_sha256"],
                     cache_import_receipt_sha256=provenance["stage2_compat"]["cache_import_receipt_sha256"],
                     source_grid_sha256=provenance["source_grid_sha256"],
                     source_metadata_sha256=provenance["latents_metadata_sha256"])
        if (type(rows) is not int or rows <= 0 or (total_rows is not None and rows != total_rows)
                or summary.get("scope") != "full" or summary.get("encoder_compared") != 0
                or summary.get("packed_compared") != rows or any(record.get(k) != v for k, v in fixed.items())):
            raise ValueError("缓存完整性认证缺全表 memcmp/finite 或明确数值 NOT_REQUIRED")
        for key, expected in (("store_table_sha256", table_sha256 or provenance.get("motion_table_sha256")),
                              ("motion_index_sha256", index_sha256 or provenance.get("motion_index_sha256"))):
            require_sha256(record.get(key), f"cache.integrity.{key}")
            if expected is not None and record[key] != expected:
                raise ValueError(f"缓存完整性认证与当前表不同: {key}")
        return record
    scope = summary.get("scope")
    rows = summary.get("total_rows")
    if (not isinstance(record, dict) or scope not in ("full", "sampled") or record.get("scope") != scope
            or record.get("status") != "PASS" or record.get("layout") != STAGE2_LAYOUT
            or type(rows) is not int or rows <= 0 or (total_rows is not None and rows != total_rows)
            or type(summary.get("packed_compared")) is not int or summary["packed_compared"] != rows
            or type(record.get("mismatches")) is not int or record["mismatches"] != 0):
        raise ValueError("encoder认证范围、总行数或全表CPU覆盖不符")
    expected = {"checkpoint_sha256": provenance["encoder"]["checkpoint_sha256"]}
    bindings = {"store_table_sha256": table_sha256 or provenance.get("motion_table_sha256"),
                "motion_index_sha256": index_sha256 or provenance.get("motion_index_sha256"),
                "source_grid_sha256": provenance.get("source_grid_sha256"),
                "source_metadata_sha256": provenance.get("latents_metadata_sha256")}
    expected.update({key: value for key, value in bindings.items() if value is not None})
    for key in ("store_table_sha256", "motion_index_sha256", "checkpoint_sha256", "source_grid_sha256"):
        require_sha256(record.get(key), f"encoder认证.{key}")
    for key, value in expected.items():
        if record.get(key) != value and (scope == "sampled" or key != "source_metadata_sha256" or key in record):
            raise ValueError(f"encoder认证与来源绑定不同: {key}")
    count = 1024 if scope == "sampled" else rows
    if (type(summary.get("encoder_compared")) is not int or summary["encoder_compared"] != count
            or type(record.get("compared")) is not int or record["compared"] != count):
        raise ValueError("encoder认证比较数不符合声明范围")
    if scope == "full":
        if (record.get("schema") != "stage2-encoder-bitexact-v1"
                or any(key in summary or key in record for key in ("sample_indices", "sample_indices_sha256", "authorization_quote"))):
            raise ValueError("全量encoder认证不能混入抽样声明")
    else:
        if (record.get("schema") != "stage2-encoder-sampled-v1"
                or type(record.get("total_rows")) is not int or record["total_rows"] != rows
                or type(record.get("packed_compared")) is not int or record["packed_compared"] != rows
                or type(record.get("packed_mismatches")) is not int or record["packed_mismatches"] != 0
                or record.get("strategy") != STAGE2_SAMPLE_STRATEGY
                or type(record.get("seed")) is not int or record["seed"] != 0
                or summary.get("authorization_quote") != STAGE2_SAMPLE_AUTHORIZATION
                or record.get("authorization_quote") != STAGE2_SAMPLE_AUTHORIZATION):
            raise ValueError("抽样encoder认证缺精确范围、全表CPU覆盖或用户授权")
        indices, counts = record.get("sample_indices"), record.get("task_counts")
        if (not isinstance(indices, list) or len(indices) != 1024
                or any(type(i) is not int or not 0 <= i < rows for i in indices)
                or indices != sorted(set(indices)) or not isinstance(counts, dict) or len(counts) != 16
                or any(not isinstance(k, str) or not k or type(v) is not int or v != 64 for k, v in counts.items())):
            raise ValueError("抽样encoder认证缺1024个唯一升序行或每任务64窗")
        digest = sha256_bytes(canonical_json_bytes(indices))
        if record.get("sample_indices_sha256") != digest or summary.get("sample_indices_sha256") != digest:
            raise ValueError("抽样encoder认证样本集合SHA不符")
        if (any(type(record.get(k)) is not int or record[k] < 0 for k in ("padded_count", "demo_count", "exec_count"))
                or record["demo_count"] + record["exec_count"] != 1024 or record["padded_count"] > record["demo_count"]
                or type(record.get("segments")) is not int or record["segments"] <= 0):
            raise ValueError("抽样encoder认证分类覆盖或段数不符")
        for key in ("source_metadata_sha256", "source_tokens_manifest_sha256"):
            require_sha256(record.get(key), f"encoder认证.{key}")
    for key in ("encoder_state_sha256", "numerics"):
        if (scope == "sampled" or key in record) and canonical_json_bytes(record.get(key)) != canonical_json_bytes(provenance[key]):
            raise ValueError(f"encoder认证的{key}与来源不同")
    return record


def task_of_h5(h5_file: str) -> str:
    """``record_dataset_<Task>.h5`` → ``<Task>``。"""
    name = os.path.basename(h5_file)
    if not (name.startswith("record_dataset_") and name.endswith(".h5")):
        raise ValueError(f"h5 文件名不符合 record_dataset_<Task>.h5: {h5_file}")
    return name[len("record_dataset_"):-len(".h5")]


def segment_key(h5_file: str, raw_ep_idx: int, segment: str) -> str:
    """段工作项键：``<Task>_ep<j>_<exec|demo>``（与 MotionJEPA 抽取器同构）。"""
    if segment not in SEGMENTS:
        raise ValueError(f"segment 非法: {segment!r}")
    return f"{task_of_h5(h5_file)}_ep{int(raw_ep_idx)}_{segment}"


# ── motion_index.json ─────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SegmentInfo:
    row_base: int | None        # 该段首行在 motion 表中的全局行号；num_grid == 0 时为 None
    num_grid: int
    num_chunks: int
    seg_len: int


@dataclasses.dataclass(frozen=True)
class IndexEntry:
    g: int
    h5_file: str
    raw_ep_idx: int
    num_timesteps: int
    exec_start_idx: int
    demo: SegmentInfo
    exec: SegmentInfo
    spec: LayoutSpec


def stage2_encoder_sample_plan(entries):
    """任务名排序、同一seed0随机源的分层1024窗；pad8是优先配额，随机余量仍可选更多pad。"""
    grouped, cursor = {}, 0
    for entry in entries:
        if entry.spec != LAYOUT_SPECS[STAGE2_LAYOUT]:
            raise ValueError("encoder抽样只能消费stage2索引")
        pools = grouped.setdefault(task_of_h5(entry.h5_file), {"demo": [], "exec": [], "padded": [], "plain_demo": []})
        for kind in SEGMENTS:
            segment = getattr(entry, kind)
            if not segment.num_grid:
                continue
            if segment.row_base != cursor:
                raise ValueError("encoder抽样索引必须连续覆盖全表")
            for j in range(segment.num_grid):
                row = cursor + j
                pools[kind].append(row)
                if kind == "demo":
                    pools["padded" if segment.seg_len - j * entry.spec.grid_stride < WINDOW_FRAMES else "plain_demo"].append(row)
            cursor += segment.num_grid
    if len(grouped) != 16:
        raise ValueError("encoder抽样必须恰有16个任务")
    rng, selected_all, task_counts = random.Random(0), [], {}
    padded_count = demo_count = exec_count = 0
    for task, pools in sorted(grouped.items()):
        all_rows = sorted(pools["demo"] + pools["exec"])
        if len(all_rows) < 64:
            raise ValueError(f"encoder抽样任务{task}不足64行")
        selected = set()
        for kind in SEGMENTS:
            if pools[kind]:
                selected.update((pools[kind][0], pools[kind][-1]))
        for name, quota in (("padded", 8), ("exec", 16), ("plain_demo", 16)):
            available = sorted(set(pools[name]) - selected)
            needed = min(quota, len(pools[name])) - len(selected.intersection(pools[name]))
            if needed > 0:
                selected.update(rng.sample(available, needed))
        selected.update(rng.sample(sorted(set(all_rows) - selected), 64 - len(selected)))
        task_counts[task] = len(selected)
        selected_all.extend(selected)
        padded_count += len(selected.intersection(pools["padded"]))
        demo_count += len(selected.intersection(pools["demo"]))
        exec_count += len(selected.intersection(pools["exec"]))
    indices = sorted(selected_all)
    if len(indices) != 1024 or len(set(indices)) != 1024:
        raise ValueError("encoder抽样未产生1024个唯一行")
    return dict(strategy=STAGE2_SAMPLE_STRATEGY, seed=0, sample_indices=indices,
                sample_indices_sha256=sha256_bytes(canonical_json_bytes(indices)), task_counts=task_counts,
                padded_count=padded_count, demo_count=demo_count, exec_count=exec_count)


def build_index_entries(manifest: dict, spec: LayoutSpec) -> list[IndexEntry]:
    """按清单 canonical 序（episodes 列表序 = global_episode_idx 序）算出每 episode 的段基址表。

    行序契约：逐 episode，每 episode 先 demo 后 exec，段内网格升序。这是 motion 表唯一的行序定义，
    打包器写、MotionMeta 校验、oracle 重算三方都从这里派生。
    """
    entries: list[IndexEntry] = []
    if spec not in LAYOUT_SPECS.values():
        raise ValueError(f"未支持的窗口契约 {spec}")
    cursor = 0
    for g, ep in enumerate(manifest["episodes"]):
        if int(ep["global_episode_idx"]) != g:
            raise ValueError(f"清单 episodes 序错乱: 第 {g} 项 global_episode_idx={ep['global_episode_idx']}")
        lens = segment_lengths(ep["num_timesteps"], ep["exec_start_idx"])
        segs: dict[str, SegmentInfo] = {}
        for seg in SEGMENTS:
            L = lens[seg]
            min_real = spec.min_real(seg)
            ng = seg_num_grid(L, min_real, spec.grid_stride)
            segs[seg] = SegmentInfo(row_base=(cursor if ng > 0 else None), num_grid=ng,
                                    num_chunks=seg_num_chunks(L, min_real), seg_len=L)
            cursor += ng
        entries.append(IndexEntry(
            g=g, h5_file=str(ep["h5_file"]), raw_ep_idx=int(ep["raw_ep_idx"]),
            num_timesteps=int(ep["num_timesteps"]), exec_start_idx=int(ep["exec_start_idx"]),
            demo=segs["demo"], exec=segs["exec"], spec=spec))
    return entries


def index_totals(entries: list[IndexEntry]) -> dict[str, int]:
    exec_rows = sum(e.exec.num_grid for e in entries)
    demo_rows = sum(e.demo.num_grid for e in entries)
    return {"rows": exec_rows + demo_rows, "exec_rows": exec_rows, "demo_rows": demo_rows}


def index_payload(manifest: dict, entries: list[IndexEntry], *, spec: LayoutSpec, layout: str,
                  mj_repo_commit: str) -> dict:
    """motion_index.json 的完整内容（不含任何运行时字段，可重算比对）。"""
    if LAYOUT_SPECS.get(layout) != spec or any(e.spec != spec for e in entries):
        raise ValueError("writer 的布局名、窗口契约与索引条目不一致")
    def seg(s: SegmentInfo) -> dict:
        return {"row_base": s.row_base, "num_grid": s.num_grid, "num_chunks": s.num_chunks,
                "seg_len": s.seg_len}
    return {
        "schema": INDEX_SCHEMA,
        "layout": layout,
        **spec.fields(),
        "grid_stride": spec.grid_stride,
        "window_frames": WINDOW_FRAMES,
        "grid_origin": GRID_ORIGIN,
        "window_direction": WINDOW_DIRECTION,
        "truncation_policy": TRUNCATION_POLICY,
        "row_order": "canonical_order episodes; per episode demo then exec; grid ascending",
        "entries": [
            {"g": e.g, "h5_file": e.h5_file, "raw_ep_idx": e.raw_ep_idx,
             "num_timesteps": e.num_timesteps, "exec_start_idx": e.exec_start_idx,
             "demo": seg(e.demo), "exec": seg(e.exec)}
            for e in entries],
        "totals": index_totals(entries),
        "manifest_sha256": manifest["sha256"],
        "mj_repo_commit": mj_repo_commit,
    }


def parse_index(raw: dict, ctx: str = "motion_index.json") -> list[IndexEntry]:
    """解析并逐项校验 motion_index.json：常量三方同值、row_base 连续、num_grid 符合公式、totals 自洽。"""
    def need(k: str):
        if k not in raw:
            raise ValueError(f"{ctx} 缺字段 {k!r}")
        return raw[k]
    if int(need("schema")) != INDEX_SCHEMA:
        raise ValueError(f"{ctx} schema={raw['schema']} != {INDEX_SCHEMA}")
    spec = layout_spec(raw, ctx)
    for key, want in (("grid_stride", spec.grid_stride), ("window_frames", WINDOW_FRAMES),
                      ("grid_origin", GRID_ORIGIN), ("window_direction", WINDOW_DIRECTION),
                      ("truncation_policy", TRUNCATION_POLICY)):
        if need(key) != want or (raw["layout"] == STAGE2_LAYOUT and type(raw[key]) is not type(want)):
            raise ValueError(f"{ctx} {key}={raw[key]!r} != 格式常量 {want!r}")
    entries: list[IndexEntry] = []
    cursor = 0
    for g, e in enumerate(need("entries")):
        if int(e["g"]) != g:
            raise ValueError(f"{ctx} entries 序错乱: 第 {g} 项 g={e['g']}")
        lens = segment_lengths(e["num_timesteps"], e["exec_start_idx"])
        segs: dict[str, SegmentInfo] = {}
        for seg in SEGMENTS:
            s = e[seg]
            L = lens[seg]
            if int(s["seg_len"]) != L:
                raise ValueError(f"{ctx} g={g} {seg}.seg_len={s['seg_len']} != {L}")
            min_real = spec.min_real(seg)
            if int(s["num_chunks"]) != seg_num_chunks(L, min_real) or int(s["num_grid"]) != seg_num_grid(L, min_real, spec.grid_stride):
                raise ValueError(
                    f"{ctx} g={g} {seg} num_chunks/num_grid={s['num_chunks']}/{s['num_grid']} "
                    f"!= 公式 {seg_num_chunks(L, min_real)}/{seg_num_grid(L, min_real, spec.grid_stride)}")
            ng = int(s["num_grid"])
            if ng == 0:
                if s["row_base"] is not None:
                    raise ValueError(f"{ctx} g={g} {seg} num_grid=0 但 row_base={s['row_base']}")
            else:
                if int(s["row_base"]) != cursor:
                    raise ValueError(
                        f"{ctx} g={g} {seg} row_base={s['row_base']} != 累计 {cursor}（不连续）")
            segs[seg] = SegmentInfo(row_base=(cursor if ng > 0 else None), num_grid=ng,
                                    num_chunks=int(s["num_chunks"]), seg_len=L)
            cursor += ng
        entries.append(IndexEntry(
            g=g, h5_file=str(e["h5_file"]), raw_ep_idx=int(e["raw_ep_idx"]),
            num_timesteps=int(e["num_timesteps"]), exec_start_idx=int(e["exec_start_idx"]),
            demo=segs["demo"], exec=segs["exec"], spec=spec))
    totals = need("totals")
    calc = index_totals(entries)
    if {k: int(totals[k]) for k in calc} != calc:
        raise ValueError(f"{ctx} totals={totals} != 现算 {calc}")
    if cursor != calc["rows"]:
        raise ValueError(f"{ctx} 行游标 {cursor} != totals.rows {calc['rows']}")
    return entries


def check_index_against_manifest(entries: list[IndexEntry], manifest: dict) -> None:
    """逐 episode 身份互校：g / h5_file / raw_ep_idx / num_timesteps / exec_start_idx 五字段。"""
    eps = manifest["episodes"]
    if len(eps) != len(entries):
        raise ValueError(f"motion index 条目数 {len(entries)} != 清单 episode 数 {len(eps)}")
    for e, ep in zip(entries, eps):
        ident = (int(ep["global_episode_idx"]), str(ep["h5_file"]), int(ep["raw_ep_idx"]),
                 int(ep["num_timesteps"]), int(ep["exec_start_idx"]))
        mine = (e.g, e.h5_file, e.raw_ep_idx, e.num_timesteps, e.exec_start_idx)
        if ident != mine:
            raise ValueError(f"motion index 与清单身份不符 g={e.g}: index={mine} manifest={ident}")


# ── MotionMeta ────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class MotionMeta:
    """store_meta.json 的结构化视图 + 逐项 fail-loud 校验（含 motion_index.json 现场 sha256）。"""

    root: pathlib.Path
    raw: dict
    status: str
    num_rows: int
    manifest_sha256: str
    manifest_path: str
    motion_index_sha256: str
    table_sha256: str | None
    entries: tuple[IndexEntry, ...]
    provenance: dict
    spec: LayoutSpec

    @property
    def row_shape(self) -> tuple[int, ...]:
        return self.spec.row_shape

    @property
    def row_bytes(self) -> int:
        return self.spec.row_bytes

    @classmethod
    def load(cls, store_root: str | pathlib.Path) -> "MotionMeta":
        root = pathlib.Path(store_root)
        meta_path = root / META_RELPATH
        if not meta_path.is_file():
            raise FileNotFoundError(f"motion 库 meta 缺失: {meta_path}")
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"store_meta.json 损坏不可解析: {meta_path}: {e}") from e

        def need(key: str):
            if key not in raw:
                raise ValueError(f"store_meta.json 缺字段 {key!r}: {meta_path}")
            return raw[key]

        if int(need("schema")) != META_SCHEMA:
            raise ValueError(f"schema={raw['schema']} != {META_SCHEMA}")
        spec = layout_spec(raw, str(meta_path))
        if raw["layout"] in (LAYOUT, STAGE2_LAYOUT) and need("manifest_scope") != "full":
            raise ValueError("raw8 motion 表必须完整覆盖绑定清单（manifest_scope=full）")
        if need("byte_order") != BYTE_ORDER or need("array_order") != ARRAY_ORDER:
            raise ValueError("byte_order/array_order 与格式常量不符")
        for key, want in (("grid_stride", spec.grid_stride), ("window_frames", WINDOW_FRAMES),
                          ("grid_origin", GRID_ORIGIN), ("window_direction", WINDOW_DIRECTION),
                          ("truncation_policy", TRUNCATION_POLICY), ("frame_size", FRAME_SIZE)):
            if need(key) != want or (raw["layout"] == STAGE2_LAYOUT and type(raw[key]) is not type(want)):
                raise ValueError(f"store_meta {key}={raw[key]!r} != 格式常量 {want!r}")
        status = need("status")
        if status not in ("packed", "verified"):
            raise ValueError(f"status 非法: {status!r}")
        t = need("tables").get(MOTION_KEY)
        if t is None:
            raise ValueError(f"tables 缺 {MOTION_KEY}")
        if any(type(d) is not int for d in t["row_shape"]) or tuple(t["row_shape"]) != spec.row_shape or t["dtype"] != "float32" \
                or int(t["row_bytes"]) != spec.row_bytes or t["relpath"] != MOTION_TABLE_RELPATH:
            raise ValueError(f"tables[{MOTION_KEY}] 形制与格式常量不符: {t}")
        num_rows = int(need("num_rows"))
        if num_rows < 0 or int(t["num_rows"]) != num_rows or int(t["byte_count"]) != num_rows * spec.row_bytes:
            raise ValueError("tables.num_rows / byte_count 与 num_rows 不符")

        # motion_index.json：现场重算 sha256，不等立即拒绝（R23 / 红线：唯一身份来源）
        idx_path = root / INDEX_RELPATH
        if not idx_path.is_file():
            raise FileNotFoundError(f"motion_index.json 缺失: {idx_path}")
        idx_bytes = idx_path.read_bytes()
        got = sha256_bytes(idx_bytes)
        if got != need("motion_index_sha256"):
            raise ValueError(
                f"motion_index.json sha256 不符（被改动？）: 现算 {got[:16]}… != "
                f"store_meta 记 {raw['motion_index_sha256'][:16]}…")
        index_raw = json.loads(idx_bytes.decode("utf-8"))
        entries = parse_index(index_raw, ctx=str(idx_path))
        if index_raw["layout"] != raw["layout"] or layout_spec(index_raw, str(idx_path)) != spec:
            raise ValueError("motion_index 与 store_meta 的布局或窗口契约不同")
        totals = index_totals(entries)
        if totals["rows"] != num_rows:
            raise ValueError(f"motion_index totals.rows={totals['rows']} != num_rows {num_rows}")
        if index_raw.get("manifest_sha256") != need("manifest_sha256"):
            raise ValueError("motion_index.json 的 manifest_sha256 与 store_meta 不符")
        provenance = need("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("provenance 须为对象")
        if raw["layout"] == STAGE2_LAYOUT:
            validate_stage2_provenance(provenance)
            if status == "verified" and stage2_uses_cache(provenance) and "encoder_validation" not in provenance["stage2_compat"]:
                raise ValueError("已 verified 的缓存表缺完整性认证")
            if "encoder_validation" in provenance["stage2_compat"]:
                validate_stage2_encoder_validation(provenance["stage2_compat"]["encoder_validation"], provenance,
                                                   total_rows=num_rows, table_sha256=t.get("sha256"),
                                                   index_sha256=raw["motion_index_sha256"])
        return cls(root=root, raw=raw, status=status, num_rows=num_rows,
                   manifest_sha256=need("manifest_sha256"), manifest_path=need("manifest_path"),
                   motion_index_sha256=raw["motion_index_sha256"], table_sha256=t.get("sha256"),
                   entries=tuple(entries), provenance=provenance, spec=spec)


def run_fast_checks(meta: MotionMeta, *, manifest_path: str | None = None) -> None:
    """fast 档：清单 sha256 现场重算 + index 与清单逐 episode 身份互校 + 表文件大小。"""
    manifest = load_manifest(manifest_path or meta.manifest_path)
    if manifest["sha256"] != meta.manifest_sha256:
        raise ValueError(
            f"清单指纹不符: motion store_meta 记 {meta.manifest_sha256[:16]}… != "
            f"现场 {manifest['sha256'][:16]}…")
    check_index_against_manifest(list(meta.entries), manifest)
    table = meta.root / MOTION_TABLE_RELPATH
    if not table.is_file():
        raise FileNotFoundError(f"motion 表缺失: {table}")
    st = table.stat().st_size
    if st != meta.num_rows * meta.row_bytes:
        raise ValueError(f"motion 表大小不符: {st} != {meta.num_rows} × {meta.row_bytes}")


def run_full_checks(meta: MotionMeta) -> None:
    """full 档：整表 sha256 对 meta 比对。"""
    if meta.table_sha256 is None:
        raise ValueError("store_meta.tables.motion_token 缺 sha256")
    got = sha256_file(meta.root / MOTION_TABLE_RELPATH)
    if got != meta.table_sha256:
        raise ValueError(f"motion 表 sha256 不符（full 档）: {got[:16]}…")


def require_verified(meta: MotionMeta) -> None:
    """status != "verified" 即 raise；显式设 MMEVLA_MOTION_ALLOW_UNVERIFIED=1 可放行但必打 WARNING。"""
    if meta.status == "verified":
        return
    if os.environ.get("MMEVLA_MOTION_ALLOW_UNVERIFIED") == "1":
        logger.warning("motion 库 status=%r 未通过 verify，因 MMEVLA_MOTION_ALLOW_UNVERIFIED=1 放行"
                       "——仅限迷你库/开发期: %s", meta.status, meta.root)
        return
    raise RuntimeError(
        f"motion 库未通过 verify（status={meta.status!r}）: {meta.root}；"
        f"先跑 pack_motion_store.py verify，或（仅开发期）设 MMEVLA_MOTION_ALLOW_UNVERIFIED=1 放行")


def require_no_pack_lock(store_root: str | pathlib.Path) -> None:
    lock = pathlib.Path(store_root) / LOCK_RELPATH
    if lock.exists():
        raise RuntimeError(f"motion 库存在 pack.lock（打包或 verify 进行中/异常残留）: {lock}")


def check_same_source(frame_manifest_sha256: str, motion_meta: MotionMeta,
                      manifest: dict) -> None:
    """双 store 同源硬闸：framesamp 与 motion 两库必须绑同一份清单，index 逐 episode 身份相同。"""
    if frame_manifest_sha256 != motion_meta.manifest_sha256:
        raise ValueError(
            f"framesamp 库与 motion 库绑定的清单不同: {frame_manifest_sha256[:16]}… != "
            f"{motion_meta.manifest_sha256[:16]}…（两个各自 verified 的库被串配）")
    if manifest["sha256"] != motion_meta.manifest_sha256:
        raise ValueError("现场清单与 motion 库 manifest_sha256 不符")
    check_index_against_manifest(list(motion_meta.entries), manifest)


# ── MotionStore（只读，整表进程内）───────────────────────────────────────────


class MotionStore:
    """整表常驻进程内的只读访问层：构造即 ``np.fromfile`` 整表读入并记录 ``_owner_pid``；
    禁止 pickle 跨进程携带（由 FrameSampDataset 在 worker 内按 pid 懒构造）。"""

    def __init__(self, store_root: str | pathlib.Path, *, meta: MotionMeta | None = None,
                 manifest_path: str | None = None, verify_level: str = "fast"):
        if verify_level not in ("fast", "full"):
            raise ValueError(f"verify_level 非法: {verify_level!r}（∈ fast|full）")
        self._root = pathlib.Path(store_root)
        self._meta = meta if meta is not None else MotionMeta.load(self._root)
        run_fast_checks(self._meta, manifest_path=manifest_path)
        if verify_level == "full":
            run_full_checks(self._meta)
        table = np.fromfile(self._root / MOTION_TABLE_RELPATH, dtype=MOTION_DTYPE)
        if table.size != self._meta.num_rows * int(np.prod(self._meta.row_shape)):
            raise ValueError("motion 表元素数与 num_rows 不符")
        table.shape = (self._meta.num_rows,) + self._meta.row_shape
        self._table = table
        self._owner_pid = os.getpid()

    @property
    def meta(self) -> MotionMeta:
        return self._meta

    @property
    def owner_pid(self) -> int:
        return self._owner_pid

    @property
    def num_rows(self) -> int:
        return self._meta.num_rows

    @property
    def table(self) -> np.ndarray:
        return self._table

    @property
    def entries(self) -> tuple[IndexEntry, ...]:
        return self._meta.entries

    def __reduce__(self):
        raise TypeError(
            "MotionStore 持有进程内整表，禁止 pickle 跨进程携带；"
            "由 FrameSampDataset 在 worker 内按 pid 懒构造")

    def rows(self, rows) -> np.ndarray:
        """按全局行号取 (n, *row_shape) f32（进程内副本切片）。"""
        r = np.asarray(rows, dtype=np.int64)
        if len(r) and (int(r.min()) < 0 or int(r.max()) >= self._meta.num_rows):
            raise IndexError(f"motion 行号越界: [{r.min()}, {r.max()}] ∉ [0, {self._meta.num_rows})")
        return self._table[r]

    def close(self) -> None:
        self._table = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
