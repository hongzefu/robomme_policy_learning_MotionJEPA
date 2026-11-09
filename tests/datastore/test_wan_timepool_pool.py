"""零参数池化数值合同（计划 G3）：独立参考、哨兵、坐标编码、归约次序与负例。"""
import ast
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from mme_vla_suite.shared import wan_timepool as tp
from mme_vla_suite.shared.wan_timepool import pool_raw_wan_latent as pool

SOURCE = pathlib.Path(tp.__file__)
SHAPE = (9, 16, 32, 32)

# 事先写定的误差界，不随观察结果调整。每个输出元素经过至多 13 次加法（T 7 次、块内 W 3 次、
# 块内 H 3 次），除以 8 和 16 在非下溢时精确，顺序求和的一阶界为 γ_n·Σ|x|，γ_n = n·u/(1-n·u)。
# 记 M 为该块 128 个输入的 |Z| 均值，则 |F32 - F真值| ≤ γ13(u=2^-24)·M；FP64 参考自身的误差
# 用同一公式 γ13(u=2^-53)·M 计入；两次除法下溢的舍入合计不超过 2^-149，取 2^-148 作绝对项。
GAMMA13_F32 = 13 * 2.0**-24 / (1 - 13 * 2.0**-24)
GAMMA13_F64 = 13 * 2.0**-53 / (1 - 13 * 2.0**-53)
ABSOLUTE_TERM = 2.0**-148


def reference_f64(z):
    """独立公式：FP64 下先把 T 轴切片取均值，再对块取均值，展平索引逐元素写入。"""
    wide = z.astype(np.float64)
    out = np.zeros(1024, dtype=np.float64)
    magnitude = np.zeros(1024, dtype=np.float64)
    for c in range(16):
        for i in range(8):
            for j in range(8):
                block = wide[1:, c, 4 * i:4 * i + 4, 4 * j:4 * j + 4]
                out[64 * c + 8 * i + j] = block.mean()
                magnitude[64 * c + 8 * i + j] = np.abs(block).mean()
    return out, magnitude


def assert_within_bound(z):
    got = pool(z)
    reference, magnitude = reference_f64(z)
    limit = (GAMMA13_F32 + GAMMA13_F64) * magnitude + ABSOLUTE_TERM
    error = np.abs(got[0].astype(np.float64) - reference)
    assert np.all(error <= limit), f"最大超界 {(error - limit).max()}"


def scalar_fixed_order(z):
    """纯标量逐元素按 T→块内 W→块内 H 累加，用来钉死向量实现的归约次序。"""
    out = np.empty(1024, dtype=np.float32)
    for c in range(16):
        temporal = [[None] * 32 for _ in range(32)]
        for h in range(32):
            for w in range(32):
                acc = z[1, c, h, w]
                for t in range(2, 9):
                    acc = acc + z[t, c, h, w]
                temporal[h][w] = acc / np.float32(8)
        for i in range(8):
            for j in range(8):
                rows = []
                for a in range(4):
                    acc = temporal[4 * i + a][4 * j]
                    for b in range(1, 4):
                        acc = acc + temporal[4 * i + a][4 * j + b]
                    rows.append(acc)
                acc = rows[0]
                for a in range(1, 4):
                    acc = acc + rows[a]
                out[64 * c + 8 * i + j] = acc / np.float32(16)
    return out.reshape(1, 1024)


def coordinate_input():
    """Z = 1024c + 32h + w，所有中间和小于 2^24，浮点运算精确。"""
    c, h, w = np.meshgrid(np.arange(16), np.arange(32), np.arange(32), indexing="ij")
    value = (1024 * c + 32 * h + w).astype(np.float32)
    return np.broadcast_to(value, SHAPE).copy()


def coordinate_expected():
    expected = np.empty(1024, dtype=np.float32)
    for c in range(16):
        for i in range(8):
            for j in range(8):
                expected[64 * c + 8 * i + j] = 1024 * c + 32 * (4 * i + 1.5) + (4 * j + 1.5)
    return expected


@pytest.mark.parametrize("seed,scale", [(0, 1.0), (1, 1e-3), (2, 1e6), (3, 1e-30)])
def test_fp64_reference_within_prewritten_bound(seed, scale):
    z = (np.random.default_rng(seed).normal(size=SHAPE) * scale).astype(np.float32)
    assert_within_bound(z)


def test_fp64_reference_within_bound_on_mixed_magnitude():
    rng = np.random.default_rng(4)
    z = (rng.normal(size=SHAPE) * 10.0 ** rng.integers(-6, 7, size=SHAPE)).astype(np.float32)
    assert_within_bound(z)


def test_bound_detects_first_latent_not_dropped():
    z = np.random.default_rng(5).normal(size=SHAPE).astype(np.float32)
    wrong = z.astype(np.float64).mean(axis=0).reshape(16, 8, 4, 8, 4).mean(axis=(2, 4)).reshape(-1)
    reference, magnitude = reference_f64(z)
    assert np.any(np.abs(wrong - reference) > (GAMMA13_F32 + GAMMA13_F64) * magnitude + ABSOLUTE_TERM)


def test_reduction_order_bit_exact_and_order_is_observable():
    rng = np.random.default_rng(6)
    z = (rng.normal(size=SHAPE) * 10.0 ** rng.integers(-3, 4, size=SHAPE)).astype(np.float32)
    got = pool(z)
    assert got.tobytes() == scalar_fixed_order(z).tobytes()
    # 反向对照：先块内 H 再块内 W 必须在这份数据上产生不同位串，否则上面的断言抓不住次序错误。
    temporal = z[1].copy()
    for t in range(2, 9):
        temporal += z[t]
    blocks = (temporal / np.float32(8)).reshape(16, 8, 4, 8, 4)
    first_h = blocks[:, :, 0].copy()
    for a in range(1, 4):
        first_h += blocks[:, :, a]
    swapped = first_h[..., 0].copy()
    for b in range(1, 4):
        swapped += first_h[..., b]
    assert (swapped / np.float32(16)).reshape(1, 1024).tobytes() != got.tobytes()


@pytest.mark.parametrize("value", [np.finfo(np.float32).max, -np.finfo(np.float32).max, 1e30])
def test_first_latent_sentinel_does_not_change_output(value):
    z = np.random.default_rng(7).normal(size=SHAPE).astype(np.float32)
    baseline = pool(z).tobytes()
    z[0] = np.float32(value)
    assert pool(z).tobytes() == baseline


def test_coordinate_encoding_catches_axis_and_flatten_errors():
    z = coordinate_input()
    expected = coordinate_expected()
    assert pool(z)[0].tobytes() == expected.tobytes()
    grid = expected.reshape(16, 8, 8)
    wrong = {
        "H/W 颠倒": grid.transpose(0, 2, 1).reshape(-1),
        "Fortran 展平": grid.reshape(-1, order="F"),
        "通道放到最后": grid.transpose(1, 2, 0).reshape(-1),
        "通道混合": np.broadcast_to(grid.mean(axis=0), (16, 8, 8)).reshape(-1),
    }
    for name, value in wrong.items():
        assert value.tobytes() != expected.tobytes(), name


@pytest.mark.parametrize("c,i,j", [(0, 0, 0), (15, 7, 7), (3, 1, 6), (9, 6, 1)])
def test_single_block_has_exact_flat_index(c, i, j):
    z = np.zeros(SHAPE, dtype=np.float32)
    z[1:, c, 4 * i:4 * i + 4, 4 * j:4 * j + 4] = 2.5
    out = pool(z)[0]
    assert out[64 * c + 8 * i + j] == 2.5
    assert np.count_nonzero(out) == 1


@pytest.mark.parametrize("value", [0.0, 1.0, -0.75, 1024.5])
def test_constant_input_is_preserved_without_normalization(value):
    out = pool(np.full(SHAPE, value, dtype=np.float32))
    assert np.array_equal(out, np.full((1, 1024), value, dtype=np.float32))


@pytest.mark.parametrize("t,c,h,w", [(1, 0, 0, 0), (8, 15, 31, 31), (4, 5, 6, 27), (2, 11, 30, 1)])
def test_impulse_weight_and_location(t, c, h, w):
    z = np.zeros(SHAPE, dtype=np.float32)
    z[t, c, h, w] = 128.0
    out = pool(z)[0]
    assert out[64 * c + 8 * (h // 4) + w // 4] == 1.0
    assert np.count_nonzero(out) == 1


def test_impulse_on_first_latent_is_dropped():
    z = np.zeros(SHAPE, dtype=np.float32)
    z[0, 3, 4, 5] = 128.0
    assert np.count_nonzero(pool(z)) == 0


def test_linear_scaling_means_no_affine_or_norm():
    z = np.random.default_rng(8).normal(size=SHAPE).astype(np.float32)
    assert pool(z * np.float32(2)).tobytes() == (pool(z) * np.float32(2)).tobytes()


def test_output_contract_and_input_untouched():
    z = np.random.default_rng(9).normal(size=SHAPE).astype(np.float32)
    before = z.tobytes()
    out = pool(z)
    assert out.shape == (1, 1024) and out.dtype == np.dtype(np.float32)
    assert out.flags.c_contiguous and len(out.tobytes()) == 4096 == tp.TIMEPOOL_OUTPUT_BYTES
    assert z.tobytes() == before
    out[:] = 0
    assert z.tobytes() == before


def test_memory_layout_of_input_does_not_change_bits():
    z = np.random.default_rng(10).normal(size=SHAPE).astype(np.float32)
    assert pool(np.asfortranarray(z)).tobytes() == pool(z).tobytes()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("position", [(0, 0, 0, 0), (8, 15, 31, 31)])
def test_nonfinite_rejected_including_dropped_latent(bad, position):
    z = np.zeros(SHAPE, dtype=np.float32)
    z[position] = bad
    with pytest.raises(ValueError, match="非有限"):
        pool(z)


def test_accumulation_overflow_rejected():
    with pytest.raises(ValueError, match="溢出"):
        pool(np.full(SHAPE, np.finfo(np.float32).max, dtype=np.float32))


@pytest.mark.parametrize("dtype", [np.float64, np.float16, np.int32, np.uint8, ">f4"])
def test_wrong_dtype_rejected_without_conversion(dtype):
    with pytest.raises(TypeError):
        pool(np.zeros(SHAPE, dtype=dtype))


@pytest.mark.parametrize("shape", [(8, 16, 32, 32), (9, 32, 32, 16), (16, 9, 32, 32), (9, 16, 32), (1, 9, 16, 32, 32)])
def test_wrong_shape_rejected(shape):
    with pytest.raises(ValueError, match="形状"):
        pool(np.zeros(shape, dtype=np.float32))


def test_non_array_rejected():
    with pytest.raises(TypeError):
        pool(np.zeros(SHAPE, dtype=np.float32).tolist())


def test_constants_and_contract():
    assert tp.TIMEPOOL_READOUT == "wan_raw_timepool8_spatial8x8_v1"
    assert tp.TIMEPOOL_LAYOUT == "motion-wan-timepool1024-grid32-demopad17-v1"
    assert tp.TIMEPOOL_ENCODER_KIND == "wan_fixedpool1024"
    assert tp.TIMEPOOL_SHAPE == (1, 1024) and tp.TIMEPOOL_PARAMETER_COUNT == 0
    contract = dict(tp.POOL_CONTRACT)
    json.dumps(contract)
    for key in ("raw", "drop_first", "temporal_mean8", "spatial_block4", "flatten_CHW", "float32"):
        assert contract[key] is True
    assert contract["version"] == tp.POOL_VERSION
    with pytest.raises(TypeError):
        tp.POOL_CONTRACT["raw"] = False


def test_source_sha256_matches_file_bytes():
    assert tp.pool_source_sha256() == hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def test_imports_are_stdlib_and_numpy_only():
    imported = set()
    for node in ast.walk(ast.parse(SOURCE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported == {"__future__", "hashlib", "pathlib", "types", "numpy"}


def test_loadable_by_path_without_package_torch_or_jax():
    code = (
        "import importlib.util,sys\n"
        "s=importlib.util.spec_from_file_location('tp_alone',sys.argv[1]);m=importlib.util.module_from_spec(s)\n"
        "s.loader.exec_module(m)\n"
        "import numpy as np\n"
        "assert m.pool_raw_wan_latent(np.zeros((9,16,32,32),np.float32)).shape==(1,1024)\n"
        "assert not {'torch','jax','mme_vla_suite'}&set(sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", code, str(SOURCE)], check=True, cwd="/")
