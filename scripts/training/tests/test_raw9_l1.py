#!/usr/bin/env python3
"""S5-L1 等价单测（计划 1112-4task-modul-wanraw9-s32-50k-plan.html §3.4、分配表 S5 行 L1）。

L1 改了三处数据管线实现：
  ① ``FrameSampDataset.getitem_into``：raw9 的 motion_emb 直接写进调用方给的 (T,16384) 槽位；
  ② ``TransformedDataset.__getitems__``：raw9 时整批 motion_emb 预分配在共享内存里、逐样本直接填，
     其余键照旧 collate，交给 collate_fn 的是已拼好的批（``_PreCollated``）；
  ③ ``_collate_fn_shm``：其余键 np.stack 直接写进共享内存 storage，不再二次拷进共享内存。

本测试把基线 BASE=e5d4354 的 ``data_loader.py`` 与 ``framesamp_dataset.py`` 原文用 ``git show`` 取出、以独立模块名加载，
构造「修复前」整链；与当前 worktree 的「修复后」整链在真实库上用同一组 idx 出批，逐键比 dtype / shape / 字节：
  - raw9（wanraw9s32.yaml）：主进程直调两批 + 真实 TorchDataLoader（2 worker、shuffle、同 seed）两批
    + 完整 ``TorchDataLoader.__iter__``（含新增后台取批线程与 make_array，0 worker）两批；
  - v11（wantp768.yaml）：同上（v11 不走 ②，只经 ③，仍须逐字节相等）。
末行 ``L1_EQUIV=PASS|FAIL batches=<n> keys=<n> mismatch=<n>``。

运行：
  JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/scratch/hongze/robomme_policy_learning_MotionJEPA/.venv \\
    PYTHONPATH=<worktree>/src uv run --no-sync python scripts/training/tests/test_raw9_l1.py
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
WT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT / "src"))
BASE = "e5d435486df08fd3849807cbfb3719286b8e11c3"
MAIN = pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA")
LIB = MAIN / "v1-store/datasets/4task-v2-1600ep-604f16da"
YAML_DIR = WT / "src/mme_vla_suite/models/config/robomme"
CASES = (
    ("raw9", "perceptual-framesamp-modul-8frame-8x8-wanraw9s32.yaml", LIB / "motion-wan-raw9-s32"),
    ("v11", "perceptual-framesamp-modul-8frame-8x8-wantp768.yaml", LIB / "motion-wan-pool768"),
)
BATCH = 8

import numpy as np  # noqa: E402

_OLD_DIR = pathlib.Path(tempfile.mkdtemp(prefix="s5l1-old-"))


def _load_old(relpath: str, modname: str):
    """把 BASE 版源码原文落到临时目录，以独立模块名加载（不污染 sys.modules 里的新模块）。"""
    src = subprocess.run(["git", "-C", str(WT), "show", f"{BASE}:{relpath}"], check=True,
                         capture_output=True, text=True).stdout
    f = _OLD_DIR / f"{modname}.py"
    f.write_text(src)
    spec = importlib.util.spec_from_file_location(modname, f)
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m          # spawn worker 反序列化时需要按名找到（见 _old_sys_path）
    spec.loader.exec_module(m)
    return m


def _data_config():
    from mme_vla_suite.training import config as _config
    tc = _config.get_config("mme_vla_suite_b128_80k")
    assets = MAIN / "v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
    tc = dataclasses.replace(tc, data=dataclasses.replace(
        tc.data, assets=dataclasses.replace(tc.data.assets, assets_dir=str(assets), asset_id="robomme")))
    return tc.data.create(tc.assets_dirs, tc.model), int(tc.model.action_horizon)


def _datasets(yaml_name, motion_root, old_dl, old_fs):
    import omegaconf
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset
    from openpi.training import data_loader as new_dl

    dc, ah = _data_config()
    hc = omegaconf.OmegaConf.load(YAML_DIR / yaml_name)
    kw = dict(data_config=dc, history_config=hc, action_horizon=ah, motion_root=str(motion_root))
    new_raw = FrameSampDataset(str(LIB / "framesamp-8x8"), **kw)
    old_raw = old_fs.FrameSampDataset(str(LIB / "framesamp-8x8"), **kw)
    return new_dl.transform_dataset(new_raw, dc), old_dl.transform_dataset(old_raw, dc)


def _flat(tree, prefix=""):
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}/"))
        else:
            out[prefix + k] = v
    return out


def _cmp(new, old, tag):
    a, b = _flat(new), _flat(old)
    bad = []
    if a.keys() != b.keys():
        return len(a), [f"{tag}: 键集合不同 {sorted(set(a) ^ set(b))}"]
    for k in a:
        x, y = a[k], b[k]
        if x is None or y is None:
            if not (x is None and y is None):
                bad.append(f"{tag}:{k} None 不一致")
            continue
        x, y = np.asarray(x), np.asarray(y)
        if x.dtype != y.dtype or x.shape != y.shape or np.ascontiguousarray(x).tobytes() != np.ascontiguousarray(y).tobytes():
            bad.append(f"{tag}:{k} {x.dtype}{x.shape} vs {y.dtype}{y.shape}")
    return len(a), bad


def main() -> int:
    import jax
    import torch
    from openpi.training import data_loader as new_dl

    old_dl = _load_old("src/openpi/training/data_loader.py", "s5l1_old_data_loader")
    old_fs = _load_old("src/mme_vla_suite/training/framesamp_dataset.py", "s5l1_old_framesamp_dataset")
    sys.path.insert(0, str(_OLD_DIR))         # spawn worker 继承父进程 sys.path，可按模块名 import 旧模块
    print(f"[env] new data_loader={new_dl.__file__} old={old_dl.__file__}", flush=True)

    n_batches = n_keys = 0
    mism: list[str] = []
    keep = []
    for name, yaml_name, motion_root in CASES:
        new_ds, old_ds = _datasets(yaml_name, motion_root, old_dl, old_fs)
        rng = np.random.default_rng(1112)
        # ── 主进程直调：新链 __getitems__ → _collate_fn_shm；旧链逐样本 → 旧 _collate_fn_shm ──
        for bi in range(2):
            idxs = rng.choice(len(new_ds), size=BATCH, replace=False).tolist()
            if bi == 1:
                idxs[0] = 0                          # 带一个 episode 起点样本（k 小、motion 尾部大段为零）
            new_b = new_dl._from_shared_torch(new_dl._collate_fn_shm(new_ds.__getitems__(idxs)))
            old_b = old_dl._from_shared_torch(old_dl._collate_fn_shm([old_ds[i] for i in idxs]))
            k, bad = _cmp(new_b, old_b, f"{name}/direct{bi}")
            n_batches += 1; n_keys += k; mism += bad
        # ── 真实 TorchDataLoader（spawn 2 worker、shuffle、同 seed）──
        loaders = []
        for mod, ds in ((new_dl, new_ds), (old_dl, old_ds)):
            loaders.append(mod.TorchDataLoader(ds, local_batch_size=BATCH, shuffle=True, num_workers=2, seed=7,
                                               framework="jax"))
        it_new, it_old = (iter(l.torch_loader) for l in loaders)
        for bi in range(2):
            new_b = new_dl._from_shared_torch(next(it_new))
            old_b = old_dl._from_shared_torch(next(it_old))
            if name == "raw9":
                assert isinstance(new_b["motion_emb"], np.ndarray)
            k, bad = _cmp(new_b, old_b, f"{name}/loader{bi}")
            n_batches += 1; n_keys += k; mism += bad
        keep.append((it_new, it_old, loaders))
        # ── 完整 TorchDataLoader.__iter__（新：后台取批线程 + 预建页表 + make_array；0 worker、shuffle、同 seed）──
        full = [mod.TorchDataLoader(ds, local_batch_size=BATCH, shuffle=True, num_workers=0, seed=11, framework="jax")
                for mod, ds in ((new_dl, new_ds), (old_dl, old_ds))]
        g_new, g_old = (iter(l) for l in full)
        for bi in range(2):
            new_b = jax.tree.map(np.asarray, next(g_new))
            old_b = jax.tree.map(np.asarray, next(g_old))
            k, bad = _cmp(new_b, old_b, f"{name}/iter{bi}")
            n_batches += 1; n_keys += k; mism += bad
        g_new.close(); g_old.close()   # 不在用例间销毁 iterator（spawn worker 退出偶发久挂，实测与本改动无关）
        print(f"[{name}] 累计 batches={n_batches} mismatch={len(mism)}", flush=True)

    for m in mism[:20]:
        print("MISMATCH", m)
    ok = not mism and n_batches == 12
    print(f"L1_EQUIV={'PASS' if ok else 'FAIL'} batches={n_batches} keys={n_keys} mismatch={len(mism)}")
    return 0 if ok else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)        # 直接退出：persistent spawn worker 在解释器退出阶段偶发挂住数分钟
