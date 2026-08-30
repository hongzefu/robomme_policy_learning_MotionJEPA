#!/usr/bin/env python3
"""分层一致性比对：把「分片改造」与「跨架构」两个变量分开各个击破。

直接拿集群产物和本地产物对拍是说不清问题的——差异既可能来自我把串行 builder 改成了
8 分片，也可能来自 A40(sm_86) 与本机 RTX 6000 Ada(sm_89) 是不同架构。所以分层：

  第一层 bitexact   同机同架构、同一批 episode，计算逻辑未改动的 builder vs 分片实现。
                    硬件变量为零 ⇒ 要求**逐字节相同**，零容差。过了这层，「分片」
                    被彻底排除，且分片实现取得「本地真值」资格（后续跨架构对拍
                    就不再受未改动 builder「只能取前缀」的限制）。

  第二层 crossarch  跨架构，按「是否过 GPU」逐 key 分类，而不是一刀切阈值：
                      kept_indices  —— numpy 像素差算的，没碰 GPU        → 逐位
                      data/*.pkl    —— 直接从 H5 读出来的                 → 逐位
                      state_emb     —— 就是那个 state 数组                → 逐位
                      pos_emb_*     —— 走 JAX 在 GPU 上算（秩一外积+sin/cos，
                                       无归约累加）                        → 实测判定归属
                      image_emb_*   —— SigLIP bf16 GPU 前向               → 量化等价
                    image_emb 是 bf16，尾数只有 8 位、1 ULP ≈ 0.4% 相对误差，
                    给绝对阈值毫无意义，故改报三个量：位完全相同占比 / 最大 ULP 差 /
                    逐 token 余弦。

  第三层 downstream 训练实际怎么用它：同一批 (episode, step) 走 prepare_frame_sampling，
                    **选帧索引与 mask 必须逐位相同**（只依赖 step_idx 与 budget，
                    不依赖任何 GPU 数值），img_emb 按第二层口径。这道保证数值噪声
                    没有改变 dataloader 的任何离散决策。

episode 一律按**物理身份 (h5_file, raw_ep_idx)** 匹配，绝不按目录名——两个库的
`features/episode_{g}/` 编号体系不同（对照库只到 11 或 39，集群库是 0–1599）。

报告里 `errors` 与 `fails` 对同一失败**双报是刻意的**，因为两者的截断行为不同：
  · `errors` 是逐条明细，供人定位，**有上限**（累计 >100 条即提前中止，报告只留前 100 条）；
  · `fails`  是聚合判据，供机器判死与跨次回归对比，条数恒定、**不会被截断**。
只留明细会在错误多时丢失判据，只留聚合则定位不到具体是哪个 episode。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import re
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from scan_manifest import load_manifest  # noqa: E402

EXACT_KEYS = ("state_emb",)
GPU_KEYS = ("image_emb_8x8", "image_emb_4x4", "image_emb_2x2")
UNDECIDED_KEYS = ("pos_emb_8x8", "pos_emb_4x4", "pos_emb_2x2")


# ── episode 身份映射 ──────────────────────────────────────────────────────────
def identity_index(manifest: dict) -> dict[tuple[str, int], dict]:
    return {(e["h5_file"], e["raw_ep_idx"]): e for e in manifest["episodes"]}


def map_untouched(manifest: dict, raw_dir: str, max_episodes: int, log_path: str) -> dict[int, dict]:
    """恢复「未改动 builder」的 global_episode_idx → 物理身份 映射。

    未改动的 builder 用 os.listdir 遍历 H5，顺序不确定，所以它的 episode_{g} 编号
    无法先验推断。这里用**两个独立来源交叉验证**，一致才放行、不一致直接 fail loud：
      ① 对同一个未被改动过的原始目录重新 os.listdir，复现它当时的遍历顺序；
      ② 解析 builder 自己打印的 `Episode {g}: timesteps=..., task_goal='...'`，
         用 timesteps 序列反查。四任务的 episode 长度分布互不相同，映射唯一。
    """
    listdir_order = [f for f in os.listdir(raw_dir) if f.endswith(".h5")]
    by_file: dict[str, list[dict]] = {}
    for e in manifest["episodes"]:
        by_file.setdefault(e["h5_file"], []).append(e)

    predicted: list[dict] = []
    for name in listdir_order:
        eps = sorted(by_file[name], key=lambda e: e["raw_ep_idx"])[:max_episodes]
        predicted.extend(eps)

    logged = [
        (int(m.group(1)), int(m.group(2)))
        for m in re.finditer(r"Episode (\d+): timesteps=(\d+)",
                             pathlib.Path(log_path).read_text(encoding="utf-8", errors="replace"))
    ]
    if len(logged) != len(predicted):
        raise SystemExit(
            f"映射交叉验证失败：日志里 {len(logged)} 个 Episode 行，"
            f"listdir 推断出 {len(predicted)} 个。拒绝猜测。")
    for (g, steps), ep in zip(sorted(logged), predicted, strict=True):
        if steps != ep["num_timesteps"]:
            raise SystemExit(
                f"映射交叉验证失败：g={g} 日志 timesteps={steps}，"
                f"listdir 推断为 {ep['h5_file']}#{ep['raw_ep_idx']} "
                f"({ep['num_timesteps']})。拒绝猜测。")
    print(f"  ✓ 映射交叉验证通过：listdir 顺序 {listdir_order} 与日志 {len(logged)} 行一致")
    return {g: ep for (g, _), ep in zip(sorted(logged), predicted, strict=True)}


def lib_index(lib: str, manifest: dict, untouched: dict | None) -> dict[tuple[str, int], dict]:
    """库内 物理身份 → {local_g, exec_offset, ep} 。

    分片实现产出的库直接沿用清单编号；未改动 builder 的库用上面恢复的映射，
    并按同一顺序重算它自己的 exec_sample 偏移（它是从 0 开始独立累加的）。
    """
    out: dict[tuple[str, int], dict] = {}
    if untouched is None:
        for e in manifest["episodes"]:
            out[(e["h5_file"], e["raw_ep_idx"])] = {
                "local_g": e["global_episode_idx"],
                "exec_offset": e["exec_sample_offset"],
                "ep": e,
            }
        return out
    running = 0
    for g in sorted(untouched):
        ep = untouched[g]
        out[(ep["h5_file"], ep["raw_ep_idx"])] = {"local_g": g, "exec_offset": running, "ep": ep}
        running += ep["exec_samples"]
    return out


# ── 数值指标 ──────────────────────────────────────────────────────────────────
def _raw_bits(a: np.ndarray) -> np.ndarray:
    """按元素宽度取位模式（不是按字节）——bf16 每元素 2 字节，按字节统计会把
    「一个元素差了 1 ULP」摊成「两个字节里错一个」，位相同占比就虚高了。"""
    utype = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize]
    return np.ascontiguousarray(a).view(utype)


def _ordered_int(a: np.ndarray) -> np.ndarray:
    """把浮点位模式映射成单调整数，好让 ULP 差就是整数差（IEEE 符号-幅值序转补码序）。"""
    raw = _raw_bits(a).astype(np.int64)
    sign_bit = 1 << (a.dtype.itemsize * 8 - 1)
    return np.where(raw >= sign_bit, sign_bit - raw, raw)


def _mantissa_bits(dt: np.dtype) -> int:
    """尾数位数：bf16 7 位、fp16 10 位、fp32 23 位、fp64 52 位。用于算某数值处的局部 ULP。"""
    if "bfloat" in str(dt):
        return 7
    return {2: 10, 4: 23, 8: 52}.get(dt.itemsize, 23)


def grid_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """误差地板：平均绝对误差 ÷ 非零元素的中位幅值，**无量纲、与容器 dtype 无关**。

    为什么不按 dtype 的 ULP 算（这版是踩过两次坑后改的）：
      · 下游打包张量按 budget=512 右侧补零，早期 step 绝大多数行是 padding，
        用全体中位幅值当参考尺度会让分母落到零上、比值爆掉；
      · 更麻烦的是 `right_padding_token_emb` 在需要补零时会把 bf16 **上抬成 float64**
        （实测：step<31 时 dtype=float64，满帧 step≥31 才是 bfloat16）。按容器 dtype 取
        ULP，就会用 float64 的 2^-52 去量 bf16 粒度的数据，比值被放大 2^45≈3.5e13
        （实测报出过 5.2e13，纯属该缺陷）。
    所以参考尺度改用**数据自身的幅值**：重排累加顺序造成的是固定绝对地板，
    该比值应是小常数；乘性/结构性错误会让它随分布漂移。

    `int_ulp_frac` 只在两侧都确为 bf16 时才有意义，其余情况返回 1.0（不参与判定）。
    """
    af, bf = a.astype(np.float64), b.astype(np.float64)
    d = np.abs(af - bf)
    out = {"err_floor_rel": 0.0, "int_ulp_frac": 1.0, "mean_abs": 0.0, "nonzero_frac": 1.0,
           "has_nonfinite": False}
    if not af.size:
        return out
    # NaN/Inf 单独走布尔通道：下面全部用 d.mean()，任一元素为 NaN 就让 err_floor_rel
    # 整体变 NaN，而 NaN 与阈值的任何比较恒为 False —— 判据会静默失效。
    # 转 float64 后再 isfinite，避开 ml_dtypes 的 bf16 ufunc 支持问题（转换本来就要做）。
    out["has_nonfinite"] = bool(not (np.isfinite(af).all() and np.isfinite(bf).all()))
    out["mean_abs"] = float(d.mean())
    mag = np.abs(af)
    nzmag = mag[mag > 1e-30]
    out["nonzero_frac"] = float(nzmag.size / mag.size)
    if not nzmag.size:
        return out
    med = float(np.median(nzmag))
    out["err_floor_rel"] = float(d.mean() / med)
    if "bfloat" in str(a.dtype) and "bfloat" in str(b.dtype):
        nz = d > 0
        if nz.any():
            local = 2.0 ** (np.floor(np.log2(np.maximum(np.abs(af[nz]), 1e-300))) - 7)
            k = d[nz] / local
            out["int_ulp_frac"] = float(np.mean(np.abs(k - np.round(k)) < 1e-6))
    return out


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """报四类量：逐位相同性、bf16 网格归属、误差地板、逐 token 余弦。

    ⚠ `max_ulp` 仍计算并输出，但**不再作为判据**：它用单调整数映射度量距离，
    在跨符号与含零（padding）时给出无意义的巨值——实测第三层报出过 9.27e18（≈2^63），
    那只是 padding 零与负值在该映射下的距离。判据改用 grid_metrics 的两项。

    ⚠ NaN/Inf 必须走独立的 `has_nonfinite` 通道，不能指望任何数值判据兜住它：
      · 余弦：含 NaN 的行范数是 NaN，`NaN > 0` 恒 False ⇒ 该行落进 `~ok` 分支，
        `cos` 保留初值 1.0（被当成完美一致），min/p5 余弦都不会变小；
      · 误差地板：d.mean() 变 NaN，而 NaN 与阈值比较恒 False；
      · 连 `bitwise_equal` 都会**假通过** —— 两侧产出同一个 NaN 位模式时 ra==rb 全 True。
    也就是说 GPU 真出数值故障时，三道判据会同时失明。
    """
    ra, rb = _raw_bits(a), _raw_bits(b)
    same_bits = float(np.mean(ra == rb)) if ra.size else 1.0
    bitwise_equal = bool(np.array_equal(ra, rb))
    try:
        ulp = int(np.max(np.abs(_ordered_int(a) - _ordered_int(b)))) if a.size else 0
    except Exception:
        ulp = -1
    af = a.astype(np.float64).reshape(-1, a.shape[-1]) if a.ndim >= 2 else a.astype(np.float64).reshape(1, -1)
    bf = b.astype(np.float64).reshape(af.shape)
    finite_row = (np.isfinite(af) & np.isfinite(bf)).all(axis=-1)
    has_nonfinite = bool(not finite_row.all()) if finite_row.size else False
    na, nb = np.linalg.norm(af, axis=-1), np.linalg.norm(bf, axis=-1)
    # 掩码里显式带上 finite_row，把「零范数 padding」与「NaN 污染」分成两回事：
    # 前者保留 cos=1.0 是对的（补零行本就该视为一致），后者的 1.0 只是占位，
    # 真正的判定由 has_nonfinite 承担（见 verdict）。
    ok = finite_row & (na > 0) & (nb > 0)
    cos = np.ones(af.shape[0])
    if ok.any():
        cos[ok] = np.sum(af[ok] * bf[ok], axis=-1) / (na[ok] * nb[ok])
    return {
        "bitwise_equal": bitwise_equal,
        "same_bit_frac": same_bits,
        "max_ulp": ulp,
        "min_cosine": float(np.min(cos)) if cos.size else 1.0,
        "max_abs_diff": float(np.max(np.abs(af - bf))) if af.size else 0.0,
        "p5_cosine": float(np.quantile(cos, 0.05)) if cos.size else 1.0,
        **grid_metrics(a, b),
        # 必须写在 **grid_metrics 之后：后写的键覆盖先展开的键，
        # 两者本应一致，这样写能保证行级判定与标量标志永不打架。
        "has_nonfinite": has_nonfinite,
    }


class Agg:
    """跨大量元素累积指标——逐文件报太啰嗦，只留全局最坏值。"""

    def __init__(self) -> None:
        self.n = 0
        self.bitwise_equal = True
        self.min_same_bit_frac = 1.0
        self.max_ulp = 0
        self.min_cosine = 1.0
        self.min_p5_cosine = 1.0
        self.max_err_floor_rel = 0.0
        self.min_int_ulp_frac = 1.0
        self.max_abs_diff = 0.0
        self.has_nonfinite = False
        self.n_nonfinite = 0

    def add(self, m: dict) -> None:
        self.n += 1
        if m.get("has_nonfinite", False):
            self.has_nonfinite = True
            self.n_nonfinite += 1
            # 非有限项一律**不折叠**进下面任何字段，三个理由：
            #   · Python 内建 min/max 对 NaN 的结果取决于入参顺序
            #     （max(0.0, nan)==0.0 而 max(nan, 0.0)==nan），会把污染静默吞掉；
            #   · NaN 落进 report 会让 json.dumps 写出裸 NaN token，不是合法 JSON，
            #     谁用 jq 读这份报告都会直接炸；
            #   · bitwise_equal 对同位 NaN 是假通过，折叠进来只会污染这个字段。
            # 数值字段的语义因此固定为「仅在有限项上的最坏值」，判定走布尔通道。
            return
        self.bitwise_equal &= m["bitwise_equal"]
        self.min_same_bit_frac = min(self.min_same_bit_frac, m["same_bit_frac"])
        self.max_ulp = max(self.max_ulp, m["max_ulp"])
        self.min_cosine = min(self.min_cosine, m["min_cosine"])
        self.min_p5_cosine = min(self.min_p5_cosine, m.get("p5_cosine", 1.0))
        self.max_err_floor_rel = max(self.max_err_floor_rel, m.get("err_floor_rel", 0.0))
        self.min_int_ulp_frac = min(self.min_int_ulp_frac, m.get("int_ulp_frac", 1.0))
        self.max_abs_diff = max(self.max_abs_diff, m["max_abs_diff"])

    def as_dict(self) -> dict:
        return {
            "compared": self.n,
            "bitwise_equal": self.bitwise_equal,
            "min_same_bit_frac": round(self.min_same_bit_frac, 6),
            "max_ulp": self.max_ulp,
            "min_cosine": round(self.min_cosine, 9),
            "p5_cosine": round(self.min_p5_cosine, 9),
            "err_floor_rel": round(self.max_err_floor_rel, 5),
            "int_ulp_frac": round(self.min_int_ulp_frac, 4),
            "max_abs_diff": self.max_abs_diff,
            # 上面的数值都只统计有限项；这两个字段才是 NaN/Inf 的唯一去处
            "has_nonfinite": self.has_nonfinite,
            "nonfinite_items": self.n_nonfinite,
        }


# ── 逐 episode 比对 ──────────────────────────────────────────────────────────
def pick_steps(n: int, k: int) -> list[int]:
    if k <= 0 or k >= n:
        return list(range(n))
    idx = {0, n - 1}
    idx.update(round(i * (n - 1) / (k - 1)) for i in range(k))
    return sorted(idx)


def compare_episode(a_lib: str, b_lib: str, a: dict, b: dict, steps: list[int],
                    aggs: dict[str, Agg], errs: list[str]) -> None:
    ga, gb = a["local_g"], b["local_g"]
    ep = a["ep"]
    tag = f"{ep['h5_file']}#{ep['raw_ep_idx']}"

    ka = pathlib.Path(a_lib, "features", f"episode_{ga}", "kept_indices.json")
    kb = pathlib.Path(b_lib, "features", f"episode_{gb}", "kept_indices.json")
    # 无论成败都 add 真实结果：只在成功时 add 会让失败的 episode 不进聚合（n 不增），
    # verdict 里 `agg.n == 0: continue` 就把整项判据跳过了。
    kept_same = ka.read_bytes() == kb.read_bytes()
    if not kept_same:
        errs.append(f"{tag}: kept_indices.json 不逐位相同（它是纯 numpy 像素差算的，"
                    f"没碰 GPU，出现差异一定是 bug 不是硬件噪声）")
    aggs["kept_indices"].add({"bitwise_equal": kept_same,
                              "same_bit_frac": 1.0 if kept_same else 0.0,
                              "max_ulp": 0, "min_cosine": 1.0, "max_abs_diff": 0.0})

    for s in steps:
        fa = np.load(pathlib.Path(a_lib, "features", f"episode_{ga}", f"token_emb_{s}.npy"),
                     allow_pickle=True).item()
        fb = np.load(pathlib.Path(b_lib, "features", f"episode_{gb}", f"token_emb_{s}.npy"),
                     allow_pickle=True).item()
        if set(fa) != set(fb):
            errs.append(f"{tag} step{s}: token_emb key 集合不同 {set(fa)} vs {set(fb)}")
            continue
        for k in sorted(fa):
            x, y = np.asarray(fa[k]), np.asarray(fb[k])
            if x.shape != y.shape or x.dtype != y.dtype:
                errs.append(f"{tag} step{s} {k}: 形制不同 {x.shape}/{x.dtype} vs {y.shape}/{y.dtype}")
                continue
            aggs.setdefault(k, Agg()).add(metrics(x, y))

    # pkl 按 (episode 身份, 该 episode 内第 j 个执行步) 定位，不按文件名
    n_exec = ep["exec_samples"]
    for j in sorted({0, n_exec - 1} | set(pick_steps(n_exec, min(8, n_exec)))):
        if j < 0 or j >= n_exec:
            continue
        pa = pathlib.Path(a_lib, "data", f"{a['exec_offset'] + j}.pkl")
        pb = pathlib.Path(b_lib, "data", f"{b['exec_offset'] + j}.pkl")
        da, db = pickle.loads(pa.read_bytes()), pickle.loads(pb.read_bytes())
        pkl_ok = True
        if set(da) != set(db):
            errs.append(f"{tag} exec{j}: pkl key 集合不同")
            aggs["pkl"].add({"bitwise_equal": False, "same_bit_frac": 0.0,
                             "max_ulp": 0, "min_cosine": 1.0, "max_abs_diff": 0.0})
            continue
        for k in sorted(da):
            va, vb = da[k], db[k]
            if k == "epis_idx":
                # epis_idx 是**身份标签**不是内容：它等于该库自己的 global_episode_idx，
                # 而两个库的编号体系本就不同（未改动 builder 走 os.listdir 顺序，
                # 分片实现走清单规范序）。所以这里不比「两库是否相等」，而是分别校验
                # 「每个库标的是不是它自己那个目录号」——RoboMMEDataset.__getitem__
                # 正是拿它去找 features/episode_{epis_idx}/，标错了训练就读错 episode。
                if int(np.asarray(va).reshape(-1)[0]) != ga:
                    errs.append(f"{tag} exec{j}: A 库 epis_idx={va} ≠ 其目录号 {ga}")
                    pkl_ok = False
                if int(np.asarray(vb).reshape(-1)[0]) != gb:
                    errs.append(f"{tag} exec{j}: B 库 epis_idx={vb} ≠ 其目录号 {gb}")
                    pkl_ok = False
                continue
            if isinstance(va, np.ndarray):
                if not np.array_equal(va, vb):
                    errs.append(f"{tag} exec{j} {k}: pkl 数组不逐位相同（直接来自 H5，"
                                f"不该有任何差异）")
                    pkl_ok = False
            elif va != vb:
                errs.append(f"{tag} exec{j} {k}: pkl 标量/字符串不同: {va!r} vs {vb!r}")
                pkl_ok = False
        aggs["pkl"].add({"bitwise_equal": pkl_ok, "same_bit_frac": 1.0 if pkl_ok else 0.0,
                         "max_ulp": 0, "min_cosine": 1.0, "max_abs_diff": 0.0})


def gather_fn_factory(lib: str, local_g: int):
    def gather(indices, *_args, **_kwargs):
        return {
            i: np.load(pathlib.Path(lib, "features", f"episode_{local_g}", f"token_emb_{i}.npy"),
                       allow_pickle=True).item()
            for i in indices
        }
    return gather


def compare_downstream(a_lib: str, b_lib: str, a: dict, b: dict, steps: list[int],
                       cfg: dict, aggs: dict[str, Agg], errs: list[str]) -> None:
    """第三层：同一批 (episode, step) 走 prepare_frame_sampling，比选帧索引 / mask / img_emb。
    prepare_buffer=False ⇒ 不载 SigLIP、不碰 GPU，纯打包路径。"""
    from mme_vla_suite.dataset_builder.mem_buffer import MemoryBuffer

    buf = MemoryBuffer(
        num_views=cfg["num_views"],
        img_emb_dim=cfg["img_emb_dim"],
        pos_emb_dim=cfg["pos_emb_dim"],
        state_emb_dim=cfg["state_emb_dim"],
    )
    tag = f"{a['ep']['h5_file']}#{a['ep']['raw_ep_idx']}"
    for s in steps:
        # 选帧索引只依赖 (step_idx, budget, token_per_image)，与库无关——这是**设计事实**，
        # 不是需要对拍的假设：get_frame_sampling_indices 是纯函数，两个库喂的是同一组
        # 入参、共用这**一次**计算。曾经这里挂过一个 ds_indices 聚合项、恒填
        # bitwise_equal=True，那是装饰性的假对拍（改成「各算一次再比」只会更糟：
        # 比的是同一个纯函数对同一组入参是否给出同一结果，测的是 numpy 不是数据集）。
        # 已删除，换成下面 ds_frames_present 这项真检查：两个库各自的帧文件是否都在。
        idx = buf.get_frame_sampling_indices(s, cfg["budget"], cfg["token_per_image"])
        frames_ok = True
        for lib, ent in ((a_lib, a), (b_lib, b)):
            missing = [i for i in idx
                       if not pathlib.Path(lib, "features", f"episode_{ent['local_g']}",
                                           f"token_emb_{i}.npy").is_file()]
            if missing:
                errs.append(f"{tag} step{s}: {lib} 缺帧 {missing[:5]}（选帧索引 {list(idx)[:8]}…）")
                frames_ok = False
        aggs.setdefault("ds_frames_present", Agg()).add(
            {"bitwise_equal": frames_ok, "same_bit_frac": 1.0 if frames_ok else 0.0,
             "max_ulp": 0, "min_cosine": 1.0, "max_abs_diff": 0.0})
        if not frames_ok:
            # 必须 continue：不跳过的话下面 prepare_frame_sampling → gather 会 np.load
            # 一个不存在的文件，FileNotFoundError 直接把进程带 traceback 打死——
            # 上面那条自己写的缺帧诊断、以及此前累积的整份 report 全都看不到。
            continue
        ra = buf.prepare_frame_sampling(s, cfg["budget"], cfg["token_per_image"],
                                        gather_fn_factory(a_lib, a["local_g"]))
        rb = buf.prepare_frame_sampling(s, cfg["budget"], cfg["token_per_image"],
                                        gather_fn_factory(b_lib, b["local_g"]))
        img_a, pos_a, st_a, mask_a = ra
        img_b, pos_b, st_b, mask_b = rb
        if not np.array_equal(mask_a, mask_b):
            errs.append(f"{tag} step{s}: mask 不逐位相同——数值差异改变了 dataloader 的离散决策")
        if not np.array_equal(st_a, st_b):
            errs.append(f"{tag} step{s}: state_emb 不逐位相同")
        aggs.setdefault("ds_img_emb", Agg()).add(metrics(img_a, img_b))
        aggs.setdefault("ds_pos_emb", Agg()).add(metrics(pos_a, pos_b))
        aggs.setdefault("ds_mask", Agg()).add(
            {"bitwise_equal": bool(np.array_equal(mask_a, mask_b)), "same_bit_frac": 1.0,
             "max_ulp": 0, "min_cosine": 1.0, "max_abs_diff": 0.0})


# ── 判据 ──────────────────────────────────────────────────────────────────────
def verdict(mode: str, aggs: dict[str, Agg], args: argparse.Namespace) -> list[str]:
    fails: list[str] = []
    # pos_emb_*（UNDECIDED_KEYS）无条件进零容差，不再只在 bitexact 下才判。
    # 它是「image_emb 的跨架构差异成因是归约累加顺序、不是 bug」这条论证的**对照组**：
    # 同走 GPU/JAX，但秩一外积无归约 ⇒ 跨架构逐位相同（2026-08-23 实测判入零容差桶）。
    # 一旦它不再逐位相同，那条归因论证本身就塌了，必须判死让人重新审，
    # 而不是像从前那样只把数值打印出来、任何差异都不影响 PASS/FAIL。
    # 三种 mode 都安全：下面的循环对 `agg is None or agg.n == 0` 已有 continue。
    zero_tolerance = ["kept_indices", "pkl", *EXACT_KEYS, *UNDECIDED_KEYS]
    if mode == "bitexact":
        zero_tolerance += list(GPU_KEYS)
    equiv_keys: tuple[str, ...] = () if mode == "bitexact" else (*GPU_KEYS, "ds_img_emb")
    ds_zero: tuple[str, ...] = () if mode == "bitexact" else (
        "ds_mask", "ds_pos_emb", "ds_frames_present")

    # NaN/Inf 零容差通道：对所有参与判定的 key 一律判死。理由见 metrics 的 docstring——
    # 余弦、误差地板、bitwise_equal 三道判据在 NaN 面前会同时失明，只能独立成判据。
    for k in (*zero_tolerance, *equiv_keys, *ds_zero):
        agg = aggs.get(k)
        if agg is None or agg.n == 0:
            continue
        if agg.has_nonfinite:
            fails.append(f"[零容差] {k} 的输入含 NaN/Inf（{agg.n_nonfinite}/{agg.n} 项）"
                         f"——数值判据对非有限值一律失效，故独立判死")

    for k in zero_tolerance:
        agg = aggs.get(k)
        if agg is None or agg.n == 0:
            continue
        if not agg.bitwise_equal:
            # kept_indices / pkl 的聚合是手工合成的（max_ulp 恒 0、same_bit_frac 恒 1），
            # 直接打印这两个数会让失败读起来像「明明一样」，故按项数报并指向逐条明细。
            fails.append(f"[零容差] {k} 未逐位相同（{agg.n} 项中至少 1 项不同："
                         f"max_ulp={agg.max_ulp} same_bit_frac={agg.min_same_bit_frac:.6f}；"
                         f"逐条明细见 report 的 errors）")
    if mode != "bitexact":
        # ⚠ 判据经 2026-08-23 实测后重新推导。原先的「位相同占比 ≥0.95 / 最大 ULP 差 ≤1」
        # 前提是「fp32 网络、差异只来自最后一层舍入」，而 SigLIP So400m 的 dtype_mm
        # 是 bfloat16、27 层全程 bf16 计算，输出差异天然在 bf16 粒度（0.4%）量级——
        # 拿 fp32 的尺子量 bf16 的网络量不出有意义的结论。`max_ulp` 还在跨符号/含零时
        # 给出无意义巨值（实测 9.27e18）。两者改为「只报不判」，判据换成下面两条：
        #   · 余弦   —— 方向是否保持（下游是线性投影，方向才是要紧的）
        #   · 误差地板 —— 平均绝对误差 ÷ 中位幅值处的 ULP。重排累加顺序造成的是
        #     「固定绝对地板」，该比值应是小常数；乘性错误会让它随分布漂移。
        for k in equiv_keys:
            agg = aggs.get(k)
            if agg is None or agg.n == 0:
                continue
            if agg.min_cosine < args.min_cosine:
                fails.append(f"[等价] {k} 最小余弦 {agg.min_cosine:.9f} < 阈值 {args.min_cosine}")
            if agg.min_p5_cosine < args.min_p5_cosine:
                fails.append(f"[等价] {k} p5 余弦 {agg.min_p5_cosine:.9f} "
                             f"< 阈值 {args.min_p5_cosine}")
            if agg.max_err_floor_rel > args.max_err_floor_rel:
                fails.append(f"[等价] {k} 误差地板 {agg.max_err_floor_rel:.4f} "
                             f"（平均绝对误差÷非零中位幅值）> 阈值 {args.max_err_floor_rel}"
                             f"——说明误差不只是逐层舍入累积")
        for k in ds_zero:
            agg = aggs.get(k)
            if agg and agg.n and not agg.bitwise_equal:
                if k == "ds_frames_present":
                    fails.append(f"[零容差] {k}：有 step 的选帧目标文件在某一侧缺失"
                                 f"（{agg.n} 项中至少 1 项，逐条明细见 report 的 errors）")
                else:
                    fails.append(f"[零容差] {k} 不逐位相同"
                                 f"——数值差异改变了 dataloader 的离散决策")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["bitexact", "crossarch", "downstream"], required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--a_lib", required=True, help="参照系库")
    ap.add_argument("--b_lib", required=True, help="被测库")
    ap.add_argument("--a_untouched_log", default="",
                    help="a_lib 由未改动 builder 产出时，给它的构建日志以恢复 episode 映射")
    ap.add_argument("--a_max_episodes", type=int, default=0, help="未改动 builder 的 --max_episodes")
    ap.add_argument("--raw_dir", default="", help="恢复 listdir 顺序用")
    ap.add_argument("--subset", default="", help="限定比对的 episode（sample 产物）")
    ap.add_argument("--steps_per_episode", type=int, default=0, help="0=全部 step")
    # ⚠ 曾有 --min_same_bit_frac / --max_ulp 两个参数，2026-08-23 判据重推后它们已不参与
    # 判定，却仍被接受、还被 step2_verify.sh 一直传着——使用者以为设了阈值，其实完全无效。
    # 2026-08-24 删除：传旧参数现在会被 argparse 明确报错，比静默忽略正确得多。
    # 现行三条判据（2026-08-23 实测后重新推导，括号内是当时的观测值与裕度）
    ap.add_argument("--min_cosine", type=float, default=1 - 1e-3)       # 观测 ≥0.99991，裕度 110×
    ap.add_argument("--min_p5_cosine", type=float, default=1 - 1e-4)    # 观测 ≥0.99998，裕度 4.8×
    ap.add_argument("--max_err_floor_rel", type=float, default=0.05)    # 观测 ≈0.016，裕度 3×
    ap.add_argument("--report", default="", help="把结果 JSON 落盘")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    untouched = None
    if args.a_untouched_log:
        if not args.raw_dir or not args.a_max_episodes:
            raise SystemExit("给了 --a_untouched_log 就必须同时给 --raw_dir 与 --a_max_episodes")
        print("=== episode 映射恢复（未改动 builder 侧）===")
        untouched = map_untouched(manifest, args.raw_dir, args.a_max_episodes, args.a_untouched_log)

    a_idx = lib_index(args.a_lib, manifest, untouched)
    b_idx = lib_index(args.b_lib, manifest, None)
    keys = sorted(set(a_idx) & set(b_idx))
    if args.subset:
        want = set(json.loads(pathlib.Path(args.subset).read_text())["global_episode_idx"])
        ident = identity_index(manifest)
        keys = [k for k in keys if ident[k]["global_episode_idx"] in want]
    if not keys:
        raise SystemExit("两个库没有共同的 episode，无法比对")

    print(f"=== 比对 mode={args.mode}  共同 episode={len(keys)} ===")
    print(f"  A(参照)={args.a_lib}\n  B(被测)={args.b_lib}")

    aggs: dict[str, Agg] = {"kept_indices": Agg(), "pkl": Agg()}
    errs: list[str] = []
    cfg = {"budget": 512, "token_per_image": 16, "num_views": 1,
           "img_emb_dim": 2048, "pos_emb_dim": 768, "state_emb_dim": 8}

    n = 0
    aborted_early = False
    for n, key in enumerate(keys, 1):
        a, b = a_idx[key], b_idx[key]
        steps = pick_steps(a["ep"]["num_timesteps"], args.steps_per_episode)
        if args.mode == "downstream":
            compare_downstream(args.a_lib, args.b_lib, a, b,
                               pick_steps(a["ep"]["num_timesteps"], min(8, a["ep"]["num_timesteps"])),
                               cfg, aggs, errs)
        else:
            compare_episode(args.a_lib, args.b_lib, a, b, steps, aggs, errs)
        if n % 5 == 0 or n == len(keys):
            print(f"  ...{n}/{len(keys)} episode", flush=True)
        if len(errs) > 100:
            errs.append("错误过多，提前中止")
            aborted_early = True
            break

    print("\n=== 逐 key 结果 ===")
    for k in sorted(aggs):
        if aggs[k].n:
            print(f"  {k:18s} {aggs[k].as_dict()}")

    fails = verdict(args.mode, aggs, args)
    if args.mode != "bitexact":
        print("\n=== pos_emb_* 桶归属实测判定 ===")
        for k in UNDECIDED_KEYS:
            agg = aggs.get(k)
            if agg and agg.n:
                if agg.bitwise_equal:
                    print(f"  {k}: 零容差桶（跨架构逐位相同）")
                else:
                    print(f"  {k}: ⚠ 跨架构不再逐位相同"
                          f"（max_ulp={agg.max_ulp}, min_cos={agg.min_cosine:.9f}）"
                          f"——已按零容差判 FAIL。")
                    print(f"     这同时推翻了「{k} 无归约 ⇒ 跨架构逐位相同」这条对照论证，"
                          f"image_emb 差异的归因需人工重新判定。")

    # episodes 记**真实比对数**而不是 len(keys)：提前中止时后者是假的。
    report = {"mode": args.mode, "episodes": n, "episodes_selected": len(keys),
              "aborted_early": aborted_early,
              "metrics": {k: v.as_dict() for k, v in aggs.items() if v.n},
              "has_nonfinite_any": any(v.has_nonfinite for v in aggs.values() if v.n),
              "errors": errs[:100], "fails": fails,
              "thresholds": {"min_cosine": args.min_cosine,
                             "min_p5_cosine": args.min_p5_cosine,
                             "max_err_floor_rel": args.max_err_floor_rel}}
    if args.report:
        pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\n报告 -> {args.report}")

    if errs or fails:
        print("\n=== 失败明细 ===")
        for e in (errs[:40] + fails):
            print(f"  ✗ {e}")
        print(f"COMPARE_RESULT={args.mode} FAIL")
        raise SystemExit(1)
    print(f"COMPARE_RESULT={args.mode} PASS")


if __name__ == "__main__":
    main()
