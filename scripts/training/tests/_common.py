"""dtype 统一修复验证工具的公共层：位型容器、摘要口径、定点样本集构造。

**为什么单独一层**：本目录三个入口脚本（dump / 单步梯度 / 对拍）共用同一套
落盘格式与哈希口径，而对拍脚本必须能在不拉起 jax 训练栈的前提下跑（它只读文件）。
把容器与哈希收在这里，入口脚本各自只 import 自己真正需要的重活。

**哈希口径与 `scripts/training/g0/bench_train_steps.py` 逐字相同**（raw =
`sha256(dtype‖shape‖bytes)`，canonical = 浮点键升 f32 后 `sha256("f32"‖shape‖bytes)`）。
两份实现的一致性由 `test_padding_dtype.py::test_hash_kouging_matches_bench` 用
importlib 加载 bench 模块现场比对锁死，防止口径漂移。

**位型容器为什么不用 npy/npz**：`np.save` 会把 `ml_dtypes.bfloat16` 写成 `V2` void
类型，`np.load` 读回即丢逻辑类型——用它存 fixture 会让对拍读到错误对象。改为每键
一个 `.bin`（原始字节，C-order）+ 旁置 JSON 记 shape / 逻辑 dtype / 字节序 / 键名，
读回按 JSON 以 `np.frombuffer` + `view` 重建，并在写盘后立即读回做 round-trip 守卫。
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
import re

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if not (REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {REPO_ROOT}（缺 pyproject.toml）")

# 本 run 的 memory 交付键：dtype 修复的全部作用面都在这四个键上
MEMORY_KEYS = ("static_image_emb", "static_pos_emb", "static_state_emb", "static_mask",
               "motion_emb", "motion_pos", "motion_mask", "mem_order")
# perceptual 模式下恒为 None 的四个 recurrent 键（jax pytree 视其为空节点）
RECUR_KEYS = ("recur_image_emb", "recur_pos_emb", "recur_state_emb", "recur_mask")

CONTAINER_SCHEMA = 1


# --------------------------------------------------------------------------
# 哈希口径（与 bench_train_steps.py 逐字相同）
# --------------------------------------------------------------------------
def leaf_sha256(arr: np.ndarray) -> str:
    """raw 物理口径：dtype 参与哈希域。用于「应逐字节不变」的键。"""
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def canonical_sha256(arr: np.ndarray) -> str:
    """canonical 数值口径：浮点键升 f32 后哈希，dtype 不入域。

    kind 'V' 覆盖 ml_dtypes.bfloat16 这类自定义浮点。bf16→f32→f64 均为精确升位，
    故 canonical 相等 ⟺ 「astype(f32) 后逐位相同」——正是本计划判据 2 的机器形式。
    """
    h = hashlib.sha256()
    if arr.dtype.kind in "fV":
        a32 = arr.astype(np.float32)
        h.update(b"f32")
        h.update(str(a32.shape).encode())
        h.update(a32.tobytes())
    else:
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_dtype(name: str) -> np.dtype:
    """按逻辑 dtype 名还原 numpy dtype；bfloat16 等自定义类型回落 ml_dtypes。

    写法与 `compare_baseline.py::_load_state_dump` 一致。
    """
    try:
        return np.dtype(name)
    except TypeError:
        import ml_dtypes

        return np.dtype(getattr(ml_dtypes, name))


# --------------------------------------------------------------------------
# 位型容器
# --------------------------------------------------------------------------
def base_name(keystr: str) -> str:
    """从 pytree keystr（形如 `['image']['base_0_rgb']`）取末段名字，用于按键名匹配。"""
    m = re.findall(r"\['([^']*)'\]", keystr)
    return m[-1] if m else keystr


def safe_name(key: str) -> str:
    """把 pytree keystr 变成可用作文件名的形式。"""
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", key).strip("_") or "leaf"


def describe_tree(tree) -> dict:
    """把一个 pytree 展开成 {keystr: 摘要条目}。

    keystr 口径与 `bench_train_steps.py` 的 `batch_digests` 完全一致
    （`jax.tree_util.keystr(path)`），因此两边产物可以直接互相印证。嵌套子树
    （如 `image` 下的多相机）会被展平到各自叶子，不会整块塞进一个条目。
    """
    import jax

    flat, _ = jax.tree_util.tree_flatten_with_path(tree, is_leaf=lambda x: x is None)
    return {jax.tree_util.keystr(path): describe_leaf(jax.tree_util.keystr(path), leaf)
            for path, leaf in flat}


def describe_leaf(key: str, value) -> dict:
    """把一个交付叶子描述成可 JSON 化的摘要条目（不含数组本体）。

    三类叶子分别处理：数组走双口径哈希；None（本 run 的四个 recur_* 键）记
    kind='none'；字符串（prompt / *_subgoal）记原文用等值断言——`<U` 数组的字节
    表示随长度变化，位型比对没有意义。
    """
    if value is None:
        return {"kind": "none"}
    if isinstance(value, str | bytes):
        return {"kind": "str", "value": value if isinstance(value, str) else value.decode()}
    arr = np.asarray(value)
    if arr.dtype.kind in "USO":
        return {"kind": "str", "value": arr.item() if arr.ndim == 0 else arr.tolist()}
    return {
        "kind": "array",
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "raw": leaf_sha256(arr),
        "canon": canonical_sha256(arr),
    }


def save_array(out_dir: pathlib.Path, key: str, arr: np.ndarray) -> dict:
    """把一个数组写成位型容器（.bin + 旁置 .json），并立即读回做 round-trip 守卫。

    守卫失败即 fail-loud——半套 fixture 比没有 fixture 更危险（对拍会读到错误对象
    却照常给出判定）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_name(key)
    bin_p = out_dir / f"{safe}.bin"
    meta_p = out_dir / f"{safe}.json"
    data = np.ascontiguousarray(arr).tobytes()
    meta = {
        "schema": CONTAINER_SCHEMA,
        "key": key,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "byteorder": "little" if arr.dtype.byteorder in ("<", "=", "|") else "big",
        "nbytes": len(data),
        "raw": leaf_sha256(arr),
        "canon": canonical_sha256(arr),
    }
    bin_p.write_bytes(data)
    meta_p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    back = load_array(out_dir, key)
    if str(back.dtype) != str(arr.dtype) or back.shape != arr.shape:
        raise SystemExit(
            f"round-trip 守卫失败（类型/形状未还原）: {key} "
            f"写入 {arr.dtype}{arr.shape} 读回 {back.dtype}{back.shape}"
        )
    if back.tobytes() != data:
        raise SystemExit(f"round-trip 守卫失败（字节不一致）: {key}")
    return meta


def load_array(out_dir: pathlib.Path, key: str) -> np.ndarray:
    safe = safe_name(key)
    meta = json.loads((out_dir / f"{safe}.json").read_text(encoding="utf-8"))
    buf = (out_dir / f"{safe}.bin").read_bytes()
    if len(buf) != meta["nbytes"]:
        raise SystemExit(f"位型容器字节数不符（产物腐烂）: {out_dir / safe}.bin")
    dt = resolve_dtype(meta["dtype"])
    return np.frombuffer(buf, dtype=dt).reshape(tuple(meta["shape"]))


# --------------------------------------------------------------------------
# 定点样本集构造（由 episode_manifest.json 精确算出，不依赖 shuffle）
# --------------------------------------------------------------------------
def fixture_steps(max_frames: int = 32):
    """覆盖补零、采样分支切换和首个可见运动窗口。"""
    if max_frames not in (8, 32):
        raise ValueError(f"不支持的帧预算: {max_frames}")
    return (0, 1, 2, max_frames - 3, max_frames - 2), (max_frames - 1, max_frames, max_frames + 1), (33, 34, 35)


def fixture_origin_mode(manifest: dict) -> str:
    """含零起点时保留历史绝对时刻；全带 demo 时按各集执行起点取偏移。"""
    return "absolute" if any(ep["exec_start_idx"] == 0 for ep in manifest["episodes"]) else "per_episode_offset"


def fixture_candidates(manifest: dict, step: int) -> list[int]:
    """候选身份独立按清单换算，偏移模式不假装覆盖补零边界。"""
    relative = fixture_origin_mode(manifest) == "per_episode_offset"
    candidates = []
    for ep in manifest["episodes"]:
        t = step + ep["exec_start_idx"] if relative else step
        if ep["exec_start_idx"] <= t < ep["num_timesteps"]:
            candidates.append(index_of(ep, t))
    return candidates


def fixture_per_step(manifest: dict, max_frames: int = 32) -> int:
    """旧清单保持原配额；偏移清单取各档最小候选数，上限 200。"""
    if fixture_origin_mode(manifest) == "absolute":
        n = min(200, sum(ep["exec_start_idx"] == 0 for ep in manifest["episodes"]))
    else:
        steps = {s for group in fixture_steps(max_frames) for s in group}
        n = min(200, *(len(fixture_candidates(manifest, s)) for s in steps))
    if n == 0:
        raise ValueError("清单至少一个定点档没有合法执行样本，拒绝不完整取证")
    return n


N_RANDOM = 1000
FIXTURE_SEED = 20260827


def load_manifest(manifest_path: pathlib.Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def index_of(ep: dict, step_idx: int) -> int:
    """manifest 正向公式：dataset_index = exec_sample_offset + (step_idx - exec_start_idx)。

    只有 exec_start_idx <= step_idx < num_timesteps 的帧才有对应 pkl（video demo
    段不写 execution 样本）。
    """
    return ep["exec_sample_offset"] + (step_idx - ep["exec_start_idx"])


def build_fixture_indices(manifest: dict, max_frames: int = 32) -> dict:
    """构造可复现定点集；相对偏移组名不代表真实短历史，模式须记入计划。"""
    total = manifest["totals"]["exec_samples"]
    groups: dict[str, list[int]] = {}
    per_step = fixture_per_step(manifest, max_frames)
    for step_idx in dict.fromkeys(s for group in fixture_steps(max_frames) for s in group):
        cand = fixture_candidates(manifest, step_idx)
        if len(cand) < per_step:
            raise SystemExit(f"step_idx={step_idx} 的候选只有 {len(cand)} 个，不足 {per_step}")
        groups[f"step{step_idx}"] = sorted(cand)[:per_step]

    rng = random.Random(FIXTURE_SEED)
    fixed = {i for g in groups.values() for i in g}
    if total - len(fixed) < N_RANDOM:
        raise ValueError("清单剩余样本不足随机档 1000 个，拒绝不完整取证")
    rand: list[int] = []
    seen = set(fixed)
    while len(rand) < N_RANDOM:
        i = rng.randrange(total)
        if i in seen:
            continue
        seen.add(i)
        rand.append(i)
    groups["random"] = rand
    return groups


def resolve_index(manifest: dict, idx: int) -> tuple[int, int]:
    """反向：index → (epis_idx, step_idx)，供 dump 工具做同源自校验。"""
    for ep in manifest["episodes"]:
        off = ep["exec_sample_offset"]
        if off <= idx < off + ep["exec_samples"]:
            return ep["global_episode_idx"], ep["exec_start_idx"] + (idx - off)
    raise SystemExit(f"index {idx} 不在任何 episode 区间内（manifest 与数据集不同源？）")


# --------------------------------------------------------------------------
# 定点 batch 组成（200 个，覆盖四种组成形态）
# --------------------------------------------------------------------------
BATCH_SIZE = 8
BATCH_PLAN = (
    ("mixed1", 50),   # 1 个短样本 + 7 个满长——最典型的「被整批抬 f64」场景
    ("allshort", 50), # 全短样本——差异密度最大
    ("allfull", 50),  # 全满长——阴性对照，两侧本就同为 bf16
    ("random", 50),   # 固定 seed 随机混合
)


def build_fixture_batches(groups: dict, max_frames: int = 32) -> list[dict]:
    """按 BATCH_PLAN 组出 200 个定点 batch（每个 8 个样本 index）。"""
    short_pool = [i for k, g in groups.items() if k.startswith("step") and int(k[4:]) <= max_frames - 2 for i in g]
    full_pool = [i for k, g in groups.items() if k.startswith("step") and int(k[4:]) >= max_frames - 1 for i in g]
    rand_pool = groups["random"]
    rng = random.Random(FIXTURE_SEED + 1)
    out: list[dict] = []
    bid = 0
    for kind, n in BATCH_PLAN:
        for _ in range(n):
            if kind == "mixed1":
                idxs = [rng.choice(short_pool)] + [rng.choice(full_pool) for _ in range(BATCH_SIZE - 1)]
            elif kind == "allshort":
                idxs = [rng.choice(short_pool) for _ in range(BATCH_SIZE)]
            elif kind == "allfull":
                idxs = [rng.choice(full_pool) for _ in range(BATCH_SIZE)]
            else:
                idxs = [rng.choice(rand_pool) for _ in range(BATCH_SIZE)]
            out.append({"batch_id": bid, "kind": kind, "indices": idxs})
            bid += 1
    return out


# --------------------------------------------------------------------------
# 产物清单
# --------------------------------------------------------------------------
def write_manifest(root: pathlib.Path, extra: dict | None = None) -> pathlib.Path:
    """为 dump 产物写 sha256 清单（防产物腐烂与工具漂移）。

    只登记摘要类文本产物；数组容器（可达数十 GB）逐个 sha256 已在各自旁置 JSON 里，
    此处只记文件数与总字节。
    """
    entries = {}
    for p in sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.json")):
        if p.name == "DUMP_MANIFEST.json" or p.parent.name == "arrays" or "arrays/" in str(p.relative_to(root)):
            continue
        rel = str(p.relative_to(root))
        entries[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    arrays = list(root.rglob("arrays/**/*.bin"))
    doc = {
        "schema": "dtype-unify-dump-v1",
        "entries": entries,
        "arrays": {"files": len(arrays), "bytes": sum(p.stat().st_size for p in arrays)},
    }
    if extra:
        doc.update(extra)
    out = root / "DUMP_MANIFEST.json"
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
