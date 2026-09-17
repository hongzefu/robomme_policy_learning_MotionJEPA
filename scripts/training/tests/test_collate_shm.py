"""验证真实 spawn 队列、bf16 全位模式及无 worker 时的共享视图。"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import ml_dtypes
import numpy as np
import pytest
import torch

from openpi.training.data_loader import _collate_fn, _collate_fn_shm, _from_shared_torch, _to_shared_torch


def samples():
    return [{"bf16": (np.arange(6).reshape(2, 3) + i).astype(ml_dtypes.bfloat16),
             "f32": np.array([i, -0.0], dtype=np.float32),
             "nested": {"f64": np.array([i / 3], dtype=np.float64),
                        "u8": np.array([i, 255], dtype=np.uint8)},
             "i64": np.array([i, -i], dtype=np.int64),
             "bool": np.array([i % 2 == 0], dtype=bool), "none": None} for i in range(8)]


def assert_bytes(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        got = actual[key]
        if isinstance(value, dict):
            assert_bytes(got, value)
        elif value is None:
            assert got is None
        else:
            assert got.dtype == value.dtype and got.shape == value.shape
            assert got.tobytes() == value.tobytes()


@pytest.mark.parametrize("workers", [0, 2])
def test_queue_bytes(workers):
    items = samples()
    loader = torch.utils.data.DataLoader(items, batch_size=4, num_workers=workers,
                                        multiprocessing_context="spawn" if workers else None,
                                        collate_fn=_collate_fn_shm)
    for i, batch in enumerate(loader):
        if workers:
            assert batch["bf16"].is_shared() and batch["f32"].is_shared()
        assert_bytes(_from_shared_torch(batch), _collate_fn(items[4 * i:4 * (i + 1)]))


def test_bfloat16_all_bit_patterns():
    original = np.arange(65536, dtype=np.uint16).view(ml_dtypes.bfloat16)
    tensor = _to_shared_torch({"x": original, "none": None})
    restored = _from_shared_torch(tensor)
    assert_bytes(restored, {"x": original, "none": None})
    assert np.shares_memory(original, restored["x"])


def test_numpy_view_retains_tensor_storage():
    batch = _from_shared_torch(_collate_fn_shm(samples()))
    assert_bytes(batch, _collate_fn(samples()))
    batch["f32"][0, 0] = 17
    assert batch["f32"][0, 0] == 17
