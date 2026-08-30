"""dtype 统一修复验证工具的公共层：位型容器、摘要口径、定点样本集构造。

**为什么单独一层**：本目录三个入口脚本（dump / 单步梯度 / 对拍）共用同一套
落盘格式与哈希口径，而对拍脚本必须能在不拉起 jax 训练栈的前提下跑（它只读文件）。
把容器与哈希收在这里，入口脚本各自只 import 自己真正需要的重活。

**哈希口径与 `scripts/training/bench/bench_train_steps.py` 逐字相同**（raw =
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
MEMORY_KEYS = ("static_image_emb", "static_pos_emb", "static_state_emb", "static_mask")
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

    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
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
SHORT_STEPS = (0, 1, 2, 29, 30)      # padding 分支（step_idx <= 30）
FULL_STEPS = (31, 32, 33)            # 满长切片分支
PER_STEP = 200
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


def build_fixture_indices(manifest: dict) -> dict:
    """构造 ~2600 个定点样本 index。

    短样本档（step_idx <= 30）只能取自 exec_start_idx == 0 的 800 个 Button 系
    episode——Video 系 exec_start_idx 最小 66，其样本 step_idx 恒 >= 66、永远走满长
    分支。这是数据事实而非取样偏置，留档须声明；满长档与随机 1000 自然覆盖两系。
    """
    eps = manifest["episodes"]
    total = manifest["totals"]["exec_samples"]
    groups: dict[str, list[int]] = {}
    for step_idx in (*SHORT_STEPS, *FULL_STEPS):
        cand = [
            index_of(ep, step_idx)
            for ep in eps
            if ep["exec_start_idx"] <= step_idx < ep["num_timesteps"]
        ]
        if len(cand) < PER_STEP:
            raise SystemExit(f"step_idx={step_idx} 的候选只有 {len(cand)} 个，不足 {PER_STEP}")
        groups[f"step{step_idx}"] = sorted(cand)[:PER_STEP]

    rng = random.Random(FIXTURE_SEED)
    fixed = {i for g in groups.values() for i in g}
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


def build_fixture_batches(groups: dict) -> list[dict]:
    """按 BATCH_PLAN 组出 200 个定点 batch（每个 8 个样本 index）。"""
    short_pool = [i for k, g in groups.items() if k.startswith("step") and int(k[4:]) <= 30 for i in g]
    full_pool = [i for k, g in groups.items() if k.startswith("step") and int(k[4:]) >= 31 for i in g]
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
