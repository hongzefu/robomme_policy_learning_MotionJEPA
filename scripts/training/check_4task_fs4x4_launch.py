"""G3 启动合同：4task modulation 帧记忆每帧 4×4（1127 计划第四节第 5 条、第五节 G3）。

用法：
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/check_4task_fs4x4_launch.py --budget 512|1024 \
      [--store <framesamp-4x4 根>] [--skip-dataset] [--runner <runner 路径>] [--repo <仓库根>]

判定行（stdout 最后一行）：
  LAUNCH_CONTRACT=PASS budget=<b> frames=<n>
  LAUNCH_CONTRACT=FAIL reason=<第一条失败>[;<其余失败>…]

检查项：
  ① YAML：A（32frame-4x4）相对 perceptual-framesamp-modul-8frame-8x8.yaml 只差 token_per_image（16 对 64）与注释；
     B（64frame-4x4）相对 A 只差 budget（1024 对 512）与注释；本 budget 的 yaml：budget 一致、
     budget // token_per_image ∈ {32, 64} 且整除、motion 节缺省（关闭态）；
  ② runner（--dry-run，不写任何东西）：钉死的 HC_SHA 等于 yaml 文件字节 sha256；TRAIN_ARGS 与期望列表逐项相等
     （训练配置 mme_vla_suite_b128_80k、--num_train_steps=50001、--dataset-path <DS>/framesamp-4x4、
     --model.history-config <本 yaml>，不得有任何额外覆盖）；冒烟 dry-run 只差 run 名与 20 步；
  ③ 训练配置 mme_vla_suite_b128_80k：batch 128、warmup 5000、peak = decay = 5e-5、EMA 0.999、save_interval 5000、FSDP 8；
  ④ 帧表 meta：layout framesamp-4x4-v1、status verified、num_rows 1192918、manifest_sha256 = runner 的 MANIFEST_ID；
  ⑤ runner 打印的 MMEVLA_FRAMESAMP_SOURCE（目录）与 MMEVLA_FRAMESAMP_MANIFEST（文件）存在；
  ⑥ 未给 --skip-dataset：CPU 上实例化 FrameSampDataset 取 1 个样本，核帧数 32/64、每帧 16 token。
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import shlex
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_DEFAULT = pathlib.Path(__file__).resolve().parents[2]
YAML_REL = pathlib.Path("src/mme_vla_suite/models/config/robomme")
BASE_8X8 = "perceptual-framesamp-modul-8frame-8x8.yaml"
YAML_A = "perceptual-framesamp-modul-32frame-4x4.yaml"
YAML_B = "perceptual-framesamp-modul-64frame-4x4.yaml"
YAML_OF = {512: YAML_A, 1024: YAML_B}
FRAMES_OF = {512: 32, 1024: 64}
RUN_OF = {512: "v2-1600ep-m32x4x4-modul-b128-50k", 1024: "v2-1600ep-m64x4x4-modul-b128-50k"}
SMOKE_RUN_OF = {512: "fs4a-smoke20", 1024: "fs4b-smoke20"}
TOKEN_PER_IMAGE = 16
TRAIN_CONFIG = "mme_vla_suite_b128_80k"
STEPS = 50001
SMOKE_STEPS = 20
LAYOUT = "framesamp-4x4-v1"
NUM_ROWS = 1192918
MAIN = pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA")
V1 = MAIN / "v1-store"
DS = V1 / "datasets/4task-v2-1600ep-604f16da"
ASSETS_DIR = V1 / "train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
MANIFEST_ID = "4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918"
FORBIDDEN_ARG_WORDS = ("motion", "e2e", "wan", "raw9", "pool768", "8x8")


class Contract:
    def __init__(self) -> None:
        self.fails: list[str] = []

    def need(self, ok: bool, reason: str) -> bool:
        if not ok:
            self.fails.append(reason)
            print(f"CHECK=FAIL {reason}", flush=True)
        return ok


def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _body_lines(p: pathlib.Path) -> list[str]:
    """去掉整行注释与空行后的 yaml 行（「只差注释」的行级判据）。"""
    return [ln for ln in p.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def _yaml_dict(p: pathlib.Path) -> dict:
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(p), resolve=True)


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _diff_keys(a: dict, b: dict) -> dict:
    fa, fb = _flat(a), _flat(b)
    return {k: (fa.get(k, "<缺>"), fb.get(k, "<缺>")) for k in sorted(set(fa) | set(fb)) if fa.get(k, "<缺>") != fb.get(k, "<缺>")}


def _line_diff_only(a: pathlib.Path, b: pathlib.Path, key: str) -> bool:
    la, lb = _body_lines(a), _body_lines(b)
    if len(la) != len(lb):
        return False
    return all(x == y or (x.startswith(f"{key}:") and y.startswith(f"{key}:")) for x, y in zip(la, lb))


def check_yamls(c: Contract, yaml_dir: pathlib.Path, budget: int) -> pathlib.Path | None:
    base, a, b = yaml_dir / BASE_8X8, yaml_dir / YAML_A, yaml_dir / YAML_B
    for p in (base, a, b):
        if not c.need(p.is_file(), f"yaml_missing:{p.name}"):
            return None
    da, db, dbase = _yaml_dict(a), _yaml_dict(b), _yaml_dict(base)
    d_a = _diff_keys(da, dbase)
    c.need(d_a == {"token_per_image": (16, 64)}, f"yaml_A_vs_8x8_diff={d_a}")
    c.need(_line_diff_only(a, base, "token_per_image"), "yaml_A_vs_8x8_line_diff_not_only_token_per_image")
    d_b = _diff_keys(db, da)
    c.need(set(d_b) == {"budget"}, f"yaml_B_vs_A_diff={d_b}")
    c.need(da.get("budget") == 512, f"yaml_A_budget={da.get('budget')}")
    c.need(_line_diff_only(b, a, "budget"), "yaml_B_vs_A_line_diff_not_only_budget")
    sel = yaml_dir / YAML_OF[budget]
    ds = da if budget == 512 else db
    yb, tpi, nv = ds.get("budget"), ds.get("token_per_image"), ds.get("num_views")
    c.need(yb == budget, f"yaml_budget={yb}!={budget}")
    c.need(tpi == TOKEN_PER_IMAGE and nv == 1, f"token_per_image/num_views={tpi}/{nv}")
    ok_div = isinstance(yb, int) and isinstance(tpi, int) and tpi > 0 and yb % tpi == 0
    c.need(ok_div and yb // tpi in (32, 64) and yb // tpi == FRAMES_OF[budget],
           f"budget_not_divisible_or_frames:{yb}//{tpi}")
    c.need("motion" not in ds, "yaml_has_motion_section")
    c.need(ds.get("integration_type") == "modulation" and ds.get("memory_token_dim") == 1024,
           f"integration={ds.get('integration_type')}/{ds.get('memory_token_dim')}")
    return sel


def run_dry(runner: pathlib.Path, budget: int, smoke: bool) -> dict:
    cmd = ["bash", str(runner), "0" * 40, "--budget", str(budget), "--dry-run"] + (["--smoke20"] if smoke else [])
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    out = {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    for ln in p.stdout.splitlines():
        if ln.startswith("TRAIN_ARGS="):
            out["TRAIN_ARGS"] = shlex.split(ln[len("TRAIN_ARGS="):])
            continue
        for tok in ln.split(" "):
            if "=" in tok:
                k, v = tok.split("=", 1)
                out.setdefault(k, v)
    return out


def expected_train_args(budget: int, smoke: bool) -> list[str]:
    run = SMOKE_RUN_OF[budget] if smoke else RUN_OF[budget]
    return [
        TRAIN_CONFIG,
        "--exp-name", run,
        "--assets-base-dir", str(V1 / "train-assets"),
        "--data.assets.assets-dir", str(ASSETS_DIR),
        "--data.assets.asset-id", "robomme",
        "--checkpoint-base-dir", str(V1 / "train-runs"),
        "--dataset-path", str(DS / "framesamp-4x4"),
        "--model.history-config", YAML_OF[budget],
        f"--num_train_steps={SMOKE_STEPS if smoke else STEPS}",
    ]


def check_runner(c: Contract, runner: pathlib.Path, budget: int, yaml_path: pathlib.Path | None) -> dict:
    if not c.need(runner.is_file(), f"runner_missing:{runner}"):
        return {}
    formal = run_dry(runner, budget, smoke=False)
    if not c.need(formal["rc"] == 0 and "TRAIN_ARGS" in formal, f"runner_dry_run_rc={formal['rc']}"):
        return formal
    ta = formal["TRAIN_ARGS"]
    c.need(f"--num_train_steps={STEPS}" in ta and formal.get("STEPS") == str(STEPS),
           f"steps_not_{STEPS}:{[x for x in ta if 'num_train_steps' in x]}")
    c.need(bool(ta) and ta[0] == TRAIN_CONFIG, f"train_config={ta[0] if ta else None}")
    c.need(ta == expected_train_args(budget, smoke=False), f"train_args_mismatch:{ta}")
    # 仓库根名 robomme_policy_learning_MotionJEPA 本身含 motion，先剥掉主检出前缀再查
    bad = [x for x in ta if any(w in x.removeprefix(str(MAIN)).lower() for w in FORBIDDEN_ARG_WORDS)]
    c.need(not bad, f"train_args_forbidden:{bad}")
    want_ck = " ".join(str(s) for s in range(5000, 50001, 5000))
    got_ck = formal["stdout"].split("EXPECT_CKPTS=", 1)[-1].splitlines()[0] if "EXPECT_CKPTS=" in formal["stdout"] else ""
    c.need(got_ck == want_ck, f"expect_ckpts={got_ck!r}")
    c.need(formal.get("RUN") == RUN_OF[budget], f"run_name={formal.get('RUN')}")
    if yaml_path is not None and yaml_path.is_file():
        c.need(formal.get("HC_SHA") == sha256_file(yaml_path), f"yaml_sha_pin_mismatch:{formal.get('HC_SHA')}")
    c.need(formal.get("MANIFEST_ID") == MANIFEST_ID, f"manifest_id_pin={formal.get('MANIFEST_ID')}")
    smoke = run_dry(runner, budget, smoke=True)
    c.need(smoke["rc"] == 0 and smoke.get("TRAIN_ARGS") == expected_train_args(budget, smoke=True)
           and smoke.get("RUN") == SMOKE_RUN_OF[budget] and smoke.get("EXPECT_CKPTS") == "19",
           f"smoke_dry_run_mismatch:{smoke.get('TRAIN_ARGS')}")
    return formal


def get_train_config(name: str):
    from mme_vla_suite.training import config as _config
    return _config.get_config(name)


def check_train_config(c: Contract, cfg) -> None:
    ls = cfg.lr_schedule
    got = dict(name=cfg.name, batch=cfg.batch_size, warmup=getattr(ls, "warmup_steps", None),
               peak=getattr(ls, "peak_lr", None), decay=getattr(ls, "decay_lr", None),
               ema=cfg.ema_decay, save=cfg.save_interval, fsdp=cfg.fsdp_devices)
    want = dict(name=TRAIN_CONFIG, batch=128, warmup=5000, peak=5e-5, decay=5e-5, ema=0.999, save=5000, fsdp=8)
    c.need(got == want, f"train_config_fields={got}")


def check_store_meta(c: Contract, store: pathlib.Path) -> None:
    p = store / "meta/store_meta.json"
    if not c.need(p.is_file(), f"store_meta_missing:{p}"):
        return
    m = json.loads(p.read_text())
    got = dict(layout=m.get("layout"), status=m.get("status"), num_rows=m.get("num_rows"),
               manifest_sha256=m.get("manifest_sha256"))
    want = dict(layout=LAYOUT, status="verified", num_rows=NUM_ROWS, manifest_sha256=MANIFEST_ID)
    c.need(got == want, f"store_meta={got}")


def check_dataset(c: Contract, store: pathlib.Path, yaml_path: pathlib.Path, budget: int,
                  source: str, manifest: str, assets_dir: pathlib.Path = ASSETS_DIR) -> None:
    from omegaconf import OmegaConf
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset

    cfg = get_train_config(TRAIN_CONFIG)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(
        cfg.data, assets=dataclasses.replace(cfg.data.assets, assets_dir=str(assets_dir))))
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    hc = OmegaConf.load(yaml_path)
    ds = FrameSampDataset(str(store), dc, hc, cfg.model.action_horizon, source_root=source, manifest_path=manifest)
    frames = FRAMES_OF[budget]
    c.need(ds._max_frames == frames and ds._tokens_per_frame == TOKEN_PER_IMAGE,
           f"dataset_frames/tokens={ds._max_frames}/{ds._tokens_per_frame}")
    s = ds[len(ds) // 2]
    img, mask = s["static_image_emb"], s["static_mask"]
    c.need(tuple(img.shape) == (budget, 2048) and tuple(s["static_pos_emb"].shape) == (budget, 768)
           and tuple(mask.shape) == (budget,), f"sample_shapes={img.shape}/{s['static_pos_emb'].shape}/{mask.shape}")
    nreal = int(mask.sum())
    c.need(nreal > 0 and nreal % TOKEN_PER_IMAGE == 0 and nreal // TOKEN_PER_IMAGE <= frames,
           f"sample_mask_true={nreal}")
    c.need(s.get("motion_emb") is None, "sample_has_motion_emb")
    print(f"DATASET_SAMPLE=PASS idx={len(ds) // 2} frames={frames} tokens_per_frame={TOKEN_PER_IMAGE} "
          f"real_frames={nreal // TOKEN_PER_IMAGE} len={len(ds)} module={sys.modules[FrameSampDataset.__module__].__file__}",
          flush=True)


def main(argv=None, *, train_config_getter=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budget", type=int, required=True, choices=(512, 1024))
    ap.add_argument("--store", default=None, help="framesamp-4x4 根（缺省 runner 的 --dataset-path）")
    ap.add_argument("--skip-dataset", action="store_true", help="不在 CPU 上实例化 FrameSampDataset")
    ap.add_argument("--repo", default=str(REPO_DEFAULT), help="读 yaml 的仓库根（缺省本脚本所在仓库）")
    ap.add_argument("--runner", default=None, help="runner 路径（缺省 <repo>/scripts/training/run_4task_fs4x4_50k.sh）")
    a = ap.parse_args(argv)
    getter = train_config_getter or get_train_config
    c = Contract()
    repo = pathlib.Path(a.repo)
    runner = pathlib.Path(a.runner) if a.runner else repo / "scripts/training/run_4task_fs4x4_50k.sh"
    try:
        yaml_path = check_yamls(c, repo / YAML_REL, a.budget)
        dry = check_runner(c, runner, a.budget, yaml_path)
        check_train_config(c, getter(TRAIN_CONFIG))
        store = pathlib.Path(a.store) if a.store else DS / "framesamp-4x4"
        check_store_meta(c, store)
        source, manifest = dry.get("MMEVLA_FRAMESAMP_SOURCE", ""), dry.get("MMEVLA_FRAMESAMP_MANIFEST", "")
        c.need(source == str(DS / "source") and pathlib.Path(source).is_dir(), f"framesamp_source={source or '<缺>'}")
        c.need(manifest == str(DS / "meta/episode_manifest.json") and pathlib.Path(manifest).is_file(),
               f"framesamp_manifest={manifest or '<缺>'}")
        if not a.skip_dataset and not c.fails:
            check_dataset(c, store, yaml_path, a.budget, source, manifest)
    except Exception as e:  # noqa: BLE001 — 合同脚本一律落判定行
        c.fails.append(f"exception:{type(e).__name__}:{e}".replace("\n", " "))
    if c.fails:
        print(f"LAUNCH_CONTRACT=FAIL reason={';'.join(c.fails)}", flush=True)
        return 1
    print(f"LAUNCH_CONTRACT=PASS budget={a.budget} frames={FRAMES_OF[a.budget]}"
          f"{' dataset=skipped' if a.skip_dataset else ''}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
