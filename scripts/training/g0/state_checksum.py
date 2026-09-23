"""有界并行读取训练状态；SHA协议及主线程的逐叶写盘顺序保持不变。"""

from concurrent.futures import ThreadPoolExecutor
import hashlib

import numpy as np


def byte_view(array):
    """保持原dtype的C序字节；标量和bfloat16也通过uint8视图交给buffer协议。"""
    return np.ascontiguousarray(array).reshape(-1).view(np.uint8)


def leaf_sha256(array):
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(byte_view(array))
    return digest.hexdigest()


def collect_one(leaf, keep_bytes):
    import jax
    array = np.asarray(jax.device_get(leaf))
    data = byte_view(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(data)
    return {"sha256":digest.hexdigest(),"finite":bool(np.isfinite(array).all()),
            "dtype":str(array.dtype),"shape":list(array.shape),"nbytes":array.nbytes,
            "data":data if keep_bytes else None}


def collect_leaves(jobs, *, workers=1, keep_bytes=False):
    """每次最多在途workers片叶；跨组不持有结果，返回顺序与原树遍历完全一致。"""
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("状态取证worker必须是1–8的整数")
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix="state-checksum") as executor:
        for start in range(0,len(jobs),workers):
            group = jobs[start:start+workers]
            # 官方device_get同样先发起异步读取；有界分组避免一次排队整个状态树。
            for _,leaf in group:
                if hasattr(leaf,"copy_to_host_async"):
                    leaf.copy_to_host_async()
            futures = [executor.submit(collect_one,leaf,keep_bytes) for _,leaf in group]
            for (key,_),future in zip(group,futures,strict=True):
                yield key,future.result()
