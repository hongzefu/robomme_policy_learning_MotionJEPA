"""取证优化必须与原C序字节协议一致，包含顺序、dtype、非有限标记及写盘字节。"""

import hashlib
from pathlib import Path
import sys

import jax
import ml_dtypes
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"g0"))
from state_checksum import leaf_sha256, collect_leaves


def old_digest(array):
    digest=hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("dtype",[np.float32,np.float64,ml_dtypes.bfloat16,np.int32,np.int64,np.bool_])
def test_scalar_empty_and_strided_protocol(dtype):
    array=np.arange(240).reshape(4,6,10).astype(dtype)
    for value in (array,array.transpose(2,0,1),array[:,::2,::-1],array.reshape(-1)[:0],np.asarray(7,dtype=dtype)):
        assert leaf_sha256(value)==old_digest(value)


@pytest.mark.parametrize("workers",[1,2,8])
def test_actual_jax_leaves_serial_parallel_and_dump_bytes(workers):
    jobs=[(str(i),jax.device_put((np.arange(240,dtype=np.float32)+i).reshape(4,6,10).transpose(2,0,1))) for i in range(19)]
    jobs += [("nonfinite",jax.device_put(np.array([np.nan,np.inf],np.float32))),
             ("scalar",jax.device_put(np.asarray(1,np.int32))),
             ("bf16",jax.device_put(np.arange(30,dtype=np.float32).astype(ml_dtypes.bfloat16)))]
    output=list(collect_leaves(jobs,workers=workers,keep_bytes=True))
    assert [key for key,_ in output]==[key for key,_ in jobs]
    for (_,leaf),(_,observed) in zip(jobs,output,strict=True):
        value=np.asarray(jax.device_get(leaf))
        assert observed["sha256"]==old_digest(value)
        assert observed["dtype"]==str(value.dtype) and observed["shape"]==list(value.shape)
        assert observed["finite"]==bool(np.isfinite(value).all())
        assert observed["nbytes"]==len(value.tobytes())
        assert observed["data"].tobytes()==value.tobytes()


@pytest.mark.parametrize("workers",[0,9,True,1.5])
def test_invalid_worker_count(workers):
    with pytest.raises(ValueError):list(collect_leaves([],workers=workers))
