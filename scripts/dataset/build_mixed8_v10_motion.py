#!/usr/bin/env python3
"""S5-B 驱动：用 S1 最终 v10 encoder 对历史 VLA latent 重算 motion 表并验收（计划
``1020-v10-encoder-to-pair4-modulation-8gpu-plan.md`` 第二部分「S5-B」，对抗检查 D3/D6/D7/D9/R2）。

只编排现成入口，不改模型、窗口公式、分片调度与数值通路：

  preflight  核 S5-A ``restore_receipt.json``（pack_motion_store.load_vae_compat）、注册 ckpt 原字节 SHA、
             冻结 config、Wan 环境 SOURCE_PIN 与 ckpt 内部 epoch/arch；新库须全新（首跑）；逐字节复制父库
             ``meta/{episode_manifest,input_manifest}.json``；历史 VAE 组在编码前预检（check_vae_group）
  encode     ``run_local.py --stage encode``（每卡一 worker、现有 LPT/claim 调度、B1 原数值通路），
             latent 读恢复目录，token 写 ``<out>/motion-tokens``；首跑要求 ``initial_complete=0``
  oracle     ``oracle_driver.py encoder`` 每卡一片（上游原件，独立参考）→ ``aggregate --kind encoder``
  pack       ``pack_motion_store.py pack --restore-receipt … --expected-ckpt-sha256 …``（status=packed，
             行序只由 manifest + iter_rows_in_order 决定）
  compare    ``compare_wan.py tokens``：新表 vs 参考逐位（必须在 pack 之后；写 meta/encoder_bitexact.json）
  checks     ``motion_checks.py`` a6set / a7 / a9set / a10 / a11（packed 表上运行）
  verify     ``pack_motion_store.py verify``（兼容模式要求 compare 记录，status=verified）
  publish    结构总量复核并写 ``<out>/build/motion_build_record.json``，末行 ``MOTION_BUILD=PASS …``

frames / source / norm 不复制、不硬链：完成记录直接引用父库路径与 SHA。历史 ``raw_dir`` 只作路径比较，从不打开。
每阶段写 ``<out>/build/state.json`` 与判定行 ``MOTION_STAGE=<名> PASS|FAIL …``；``--resume`` 只接受
``build/identity.json`` 完全相同的续跑（同 receipt / ckpt / config / SOURCE_PIN / 清单），已 PASS 阶段跳过。

冒烟（``--max-segments N``）：只取清单序前 N 段，单卡 ``encode_motion.py --segments`` + 单片 oracle +
``compare_wan.py segtokens``，不 pack、不发布，末行 ``MOTION_BUILD_SMOKE=PASS …``；输出目录须与正式库不同。

用法（主 venv，主检出根；GPU 段放 detached tmux）：
  uv run --frozen --no-sync python scripts/dataset/build_mixed8_v10_motion.py \\
      --restore-receipt v1-store/restore/mixed8-200-vla-latents-20261021-a/restore_receipt.json \\
      --encoder-ckpt /scratch/hongze/MotionJEPA/runs/<注册目录>/checkpoint_epoch_60.pt --encoder-sha256 <sha> \\
      --out-store v1-store/datasets/16task-mixed8-200-v10e60-pair4-20261020-a --gpus 0-7
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "wan"))

import pack_motion_store as pms  # noqa: E402
from mme_vla_suite.datastore import motion_store as ms  # noqa: E402
from mme_vla_suite.datastore.manifest import load_manifest  # noqa: E402

V1_STORE = _REPO_ROOT / "v1-store"
WAN_DIR = _HERE / "wan"
DEFAULT_PARENT = V1_STORE / "datasets" / "16task-mixed8-200-3200ep-e3c7564f"
DEFAULT_MJ = pathlib.Path(os.environ.get("MJ_REPO", "/scratch/hongze/MotionJEPA"))
EXPECTED_EPOCH = 60
EXPECTED_ARCH = "wan-latent-v10-st"
ROW_SHAPE = (8, 256)
ROW_BYTES = 8 * 256 * 4                    # 8,192 B/行（D6）
EXPECT = {"rows": 128342, "exec": 81611, "demo": 46731, "episodes": 3200, "segments": 4995,
          "padded": 1800, "eligible": 1800, "bytes": 128342 * ROW_BYTES}
assert EXPECT["bytes"] == 1_051_377_664 and EXPECT["rows"] == EXPECT["exec"] + EXPECT["demo"]
FULL_STAGES = ("preflight", "encode", "oracle", "pack", "compare", "checks", "verify", "publish")
SMOKE_STAGES = ("preflight", "encode", "oracle", "compare")
BUILD_DIR = "build"


class BuildError(RuntimeError):
    def __init__(self, stage: str, reason: str):
        super().__init__(f"{stage}: {reason}")
        self.stage, self.reason = stage, reason


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256_file(p: pathlib.Path) -> str:
    return ms.sha256_file(p)


def write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def copy_exact(src: pathlib.Path, dst: pathlib.Path) -> str:
    """逐字节复制（不硬链、不软链），复制后两侧 SHA 必须相同。"""
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or not dst.is_file() or sha256_file(dst) != sha256_file(src):
            raise BuildError("preflight", f"{dst} 已存在且不是 {src} 的原字节副本")
        return sha256_file(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    a, b = sha256_file(src), sha256_file(dst)
    if a != b or os.stat(dst).st_nlink != 1:
        raise BuildError("preflight", f"{dst} 复制后 SHA/链接数不符")
    return b


def parse_gpus(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    if not out or len(set(out)) != len(out):
        raise SystemExit(f"--gpus 非法: {text!r}")
    return out


def ordered_segment_keys(manifest: dict) -> list[str]:
    """唯一行序来源：manifest + iter_rows_in_order（episode 序、demo→exec、段内网格递增）。"""
    entries = ms.build_index_entries(manifest, ms.LAYOUT_SPECS[ms.LAYOUT])
    return [key for _e, _seg, key, _rb, _ng in pms.iter_rows_in_order(entries)]


# ══ 子进程执行（测试可替换）══════════════════════════════════════════════════════


class Runner:
    """运行子进程：stdout 逐行带前缀转发并写日志；返回 (rc, lines)。并行版每个 job 一线程。"""

    def run(self, tag: str, cmd: list[str], env: dict, log: pathlib.Path) -> tuple[int, list[str]]:
        return self.run_parallel([(tag, cmd, env, log)])[0]

    def run_parallel(self, jobs: list[tuple[str, list[str], dict, pathlib.Path]]) -> list[tuple[int, list[str]]]:
        results: list = [None] * len(jobs)

        def one(i, tag, cmd, env, log):
            log.parent.mkdir(parents=True, exist_ok=True)
            lines: list[str] = []
            with open(log, "a", encoding="utf-8") as lf:
                lf.write(f"### {now_utc()} cmd={' '.join(cmd)}\n")
                print(f"[{tag}] 起: {' '.join(cmd)}", flush=True)
                p = subprocess.Popen(cmd, env=env, cwd=str(_REPO_ROOT), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in p.stdout:
                    lines.append(line.rstrip("\n"))
                    lf.write(line)
                    lf.flush()
                    sys.stdout.write(f"[{tag}] {line}")
                    sys.stdout.flush()
                p.wait()
                lf.write(f"### rc={p.returncode}\n")
            results[i] = (p.returncode, lines)

        threads = [threading.Thread(target=one, args=(i, *job), daemon=True) for i, job in enumerate(jobs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results


def main_env() -> dict:
    env = dict(os.environ)
    env.update({"PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"})
    return env


def wan_env(gpu: int | None) -> dict:
    env = dict(os.environ)
    env.update({"PYTHONUNBUFFERED": "1", "UV_PROJECT_ENVIRONMENT": str(V1_STORE / "venvs" / "wan"),
                "HF_HUB_OFFLINE": "1", "HF_HOME": str(V1_STORE / "cache" / "hf"),
                "XDG_CACHE_HOME": str(V1_STORE / "cache" / "xdg"), "UV_LINK_MODE": "copy",
                "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu)})
    return env


WAN_PY = ["uv", "run", "--project", str(WAN_DIR), "--frozen", "--no-sync", "python"]


def find_line(lines: list[str], prefix: str) -> str | None:
    hits = [ln for ln in lines if ln.startswith(prefix)]
    return hits[-1] if hits else None


def require_line(stage: str, rc: int, lines: list[str], prefix: str, pass_token: str = "PASS") -> str:
    line = find_line(lines, prefix)
    if rc != 0 or line is None or not line.startswith(f"{prefix}{pass_token}"):
        raise BuildError(stage, f"rc={rc} 判定行 {line!r}（要求 {prefix}{pass_token}）")
    return line


# ══ 构建 ═════════════════════════════════════════════════════════════════════


class Build:
    def __init__(self, args, runner: Runner | None = None, pins: pms.HistoricalPins = pms.HistoricalPins(),
                 expect: dict | None = None):
        """``pins`` / ``expect`` 只供 CPU 测试夹具注入合成值；CLI 不提供任何覆盖或跳过开关。"""
        self.args = args
        self.expect = dict(EXPECT if expect is None else expect)
        self.runner = runner or Runner()
        self.pins = pins
        self.smoke = args.max_segments is not None
        self.out = pathlib.Path(args.out_store).resolve()
        self.receipt = pathlib.Path(args.restore_receipt).resolve()
        self.latents = self.receipt.parent / "wan-latents"
        self.ckpt = pathlib.Path(args.encoder_ckpt)
        self.parent = pathlib.Path(args.parent_lib).resolve()
        self.gpus = parse_gpus(args.gpus)
        self.mj = pathlib.Path(args.mj_repo)
        self.meta_dir = self.out / "meta"
        self.manifest_path = self.meta_dir / "episode_manifest.json"
        self.tokens = self.out / "motion-tokens"
        self.oracle = self.out / "oracle" / "wan-mj"
        self.store = self.out / "motion"
        self.build = self.out / BUILD_DIR
        self.logs = self.out / "logs"
        self.stages = SMOKE_STAGES if self.smoke else FULL_STAGES
        self.state: dict = {}
        self.lines: dict[str, str] = {}

    # ── 状态 ──
    def stage_done(self, name: str) -> bool:
        return self.state.get("stages", {}).get(name, {}).get("status") == "PASS"

    def mark(self, name: str, status: str, detail: str) -> None:
        self.state.setdefault("stages", {})[name] = {"status": status, "detail": detail, "at_utc": now_utc()}
        write_json(self.build / "state.json", self.state)
        print(f"MOTION_STAGE={name} {status} {detail}", flush=True)

    # ── identity ──
    def identity(self) -> dict:
        cfg = self.ckpt.parent / "config.yaml"
        return {"schema": "mixed8-v10-motion-build/v1", "kind": "smoke" if self.smoke else "full",
                "max_segments": self.args.max_segments,
                "restore_receipt": str(self.receipt), "restore_receipt_sha256": sha256_file(self.receipt),
                "latents_root": str(self.latents),
                "encoder_ckpt": str(self.ckpt), "encoder_sha256": self.args.encoder_sha256,
                "encoder_run_dir": str(self.ckpt.parent), "encoder_config_sha256": sha256_file(cfg),
                "expected_epoch": EXPECTED_EPOCH,
                "source_pin_sha256": sha256_file(WAN_DIR / "SOURCE_PIN.json"),
                "parent_lib": str(self.parent),
                "parent_episode_manifest_sha256": sha256_file(self.parent / "meta" / "episode_manifest.json"),
                "parent_input_manifest_sha256": sha256_file(self.parent / "meta" / "input_manifest.json"),
                "mj_repo": str(self.mj.resolve())}

    # ── preflight ──
    def preflight(self) -> None:
        a = self.args
        if not self.ckpt.is_absolute():
            raise BuildError("preflight", "--encoder-ckpt 必须是注册副本的绝对路径")
        if self.ckpt.name != f"checkpoint_epoch_{EXPECTED_EPOCH}.pt" or not self.ckpt.is_file():
            raise BuildError("preflight", f"encoder ckpt 必须是已注册的 checkpoint_epoch_{EXPECTED_EPOCH}.pt: {self.ckpt}")
        if not re.fullmatch(r"[0-9a-f]{64}", a.encoder_sha256 or ""):
            raise BuildError("preflight", "--encoder-sha256 必须是 64 位小写十六进制")
        if sha256_file(self.ckpt) != a.encoder_sha256:
            raise BuildError("preflight", "注册 ckpt 原字节 SHA 与 --encoder-sha256 不同")
        if not (self.ckpt.parent / "config.yaml").is_file():
            raise BuildError("preflight", "注册目录缺冻结 config.yaml")
        if self.out == self.parent or self.parent in self.out.parents or self.out in self.parent.parents:
            raise BuildError("preflight", "新库不得与父库重叠")
        ident = self.identity()
        if self.out.is_symlink():
            raise BuildError("preflight", "输出根是符号链接")
        existing = self.out.exists() and any(self.out.iterdir())
        if existing:
            old = self.build / "identity.json"
            if not a.resume:
                raise BuildError("preflight", f"输出目录非空（首跑必须全新）: {self.out}；续跑须 --resume")
            if not old.is_file() or json.loads(old.read_text(encoding="utf-8")) != ident:
                raise BuildError("preflight", "续跑身份与已有 build/identity.json 不同，拒绝接受缓存")
            self.state = json.loads((self.build / "state.json").read_text(encoding="utf-8"))
        else:
            if a.resume:
                raise BuildError("preflight", "--resume 但输出目录不存在或为空")
            self.build.mkdir(parents=True, exist_ok=True)
            write_json(self.build / "identity.json", ident)
            self.state = {"identity_sha256": hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest(),
                          "started_at_utc": now_utc(), "resume_count": 0, "stages": {}}
            write_json(self.build / "state.json", self.state)
        if existing:
            self.state["resume_count"] = int(self.state.get("resume_count", 0)) + 1
            write_json(self.build / "state.json", self.state)
        if self.stage_done("preflight"):
            return
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        for name in ("episode_manifest.json", "input_manifest.json"):
            copy_exact(self.parent / "meta" / name, self.meta_dir / name)
        compat = pms.load_vae_compat(self.receipt, self.latents, manifest_path=self.manifest_path,
                                     oracle_report=self.latents.parent / "oracle" / "wan-mj" / "vae_report.json",
                                     pins=self.pins)
        manifest = load_manifest(self.manifest_path)
        keys = ordered_segment_keys(manifest)
        if len(keys) != self.expect["segments"]:
            raise BuildError("preflight", f"清单段数 {len(keys)} != {self.expect['segments']}")
        pms.check_vae_group(self.latents, keys, compat)
        write_json(self.build / "vae_compat.json", compat)
        # Wan 环境：SOURCE_PIN（安装提交 + 适配件 SHA + encoder/factory 源码）与 ckpt 内部 epoch/arch
        code = ("import sys,pathlib,torch; sys.path.insert(0,sys.argv[1]); import wan_common as wc; "
                "pin=wc.load_source_pin(pathlib.Path(sys.argv[1])); "
                "ck=torch.load(sys.argv[2],map_location='cpu',weights_only=False); "
                "print(f\"CKPT_IDENTITY epoch={ck.get('epoch')} arch={ck.get('arch')} mj_commit={pin['mj_repo_commit']}\")")
        rc, lines = self.runner.run("ckpt-identity", [*WAN_PY, "-c", code, str(WAN_DIR), str(self.ckpt)],
                                    wan_env(None), self.logs / "preflight-ckpt.log")
        line = find_line(lines, "CKPT_IDENTITY ")
        want = f"CKPT_IDENTITY epoch={EXPECTED_EPOCH} arch={EXPECTED_ARCH} mj_commit="
        pin = json.loads((WAN_DIR / "SOURCE_PIN.json").read_text(encoding="utf-8"))
        if rc != 0 or line != want + pin["mj_repo_commit"]:
            raise BuildError("preflight", f"ckpt 内部身份或 Wan 环境 SOURCE_PIN 不符: rc={rc} {line!r}")
        self.mark("preflight", "PASS", f"segments={len(keys)} receipt={compat['receipt_sha256'][:16]} "
                  f"ckpt={self.args.encoder_sha256[:16]} epoch={EXPECTED_EPOCH}")

    # ── encode ──
    def smoke_keys(self) -> list[str]:
        keys = ordered_segment_keys(load_manifest(self.manifest_path))
        n = int(self.args.max_segments)
        if not 1 <= n <= len(keys):
            raise BuildError("encode", f"--max-segments={n} 越界")
        return keys[:n]

    def encode(self) -> None:
        a = self.args
        raw_dir = load_manifest(self.manifest_path)["raw_dir"]       # 历史路径，只比较不打开
        if self.smoke:
            gpu = self.gpus[0]
            cmd = [*WAN_PY, str(WAN_DIR / "encode_motion.py"), "--manifest", str(self.manifest_path),
                   "--latents", str(self.latents), "--out", str(self.tokens), "--worker", f"smoke-gpu{gpu}",
                   "--encoder-run-dir", str(self.ckpt.parent), "--checkpoint", self.ckpt.name,
                   "--expected-ckpt-sha256", a.encoder_sha256, "--segments", *self.smoke_keys()]
            rc, lines = self.runner.run(f"encode-gpu{gpu}", cmd, wan_env(gpu), self.logs / "encode-smoke.log")
            line = find_line(lines, "WORKER_DONE stage=encode")
            if rc != 0 or line is None:
                raise BuildError("encode", f"rc={rc} {line!r}")
            self.check_token_epochs(self.smoke_keys())
            self.mark("encode", "PASS", line)
            return
        cmd = [sys.executable, str(_HERE / "run_local.py"), "--stage", "encode", "--lib", str(self.out),
               "--latents", str(self.latents), "--gpus", ",".join(map(str, self.gpus)), "--raw-dir", raw_dir,
               "--encoder-run-dir", str(self.ckpt.parent), "--checkpoint", self.ckpt.name,
               "--expected-ckpt-sha256", a.encoder_sha256, "--require-free-mib", str(a.require_free_mib)]
        rc, lines = self.runner.run("encode", cmd, main_env_with_gpus(self.gpus), self.logs / "encode.log")
        start = find_line(lines, "STAGE_START stage=encode")
        work = find_line(lines, "STAGE_WORKSET=")
        done = find_line(lines, "STAGE_DONE stage=encode")
        if rc != 0 or start is None or work is None or done is None or not work.startswith("STAGE_WORKSET=PASS"):
            raise BuildError("encode", f"rc={rc} {work!r} {done!r}")
        first_run = int(self.state.get("resume_count", 0)) == 0
        if first_run and "initial_complete=0" not in start:
            raise BuildError("encode", f"首跑要求 initial_complete=0: {start!r}")
        if f"items={self.expect['segments']}" not in done:
            raise BuildError("encode", f"段数不符: {done!r}")
        self.check_token_epochs(ordered_segment_keys(load_manifest(self.manifest_path)))
        self.mark("encode", "PASS", f"{work} | {done}")

    def check_token_epochs(self, keys: list[str]) -> None:
        for key in keys:
            enc = json.loads((self.tokens / f"{key}.metadata.json").read_text(encoding="utf-8"))["encoder"]
            if (enc.get("checkpoint_epoch"), enc.get("checkpoint_sha256"), enc.get("arch"),
                    enc.get("motion_dims")) != (EXPECTED_EPOCH, self.args.encoder_sha256, EXPECTED_ARCH, list(ROW_SHAPE)):
                raise BuildError("encode", f"{key} token 记录的 encoder 身份不是最终 e{EXPECTED_EPOCH}/raw8")

    # ── oracle ──
    def oracle_cmd(self, shard: int, num: int, extra: list[str]) -> list[str]:
        return [*WAN_PY, str(WAN_DIR / "oracle_driver.py"), "--mj-repo", str(self.mj), "encoder",
                "--manifest", str(self.manifest_path), "--latents", str(self.latents), "--out", str(self.oracle),
                "--encoder-run-dir", str(self.ckpt.parent), "--checkpoint", self.ckpt.name,
                "--expected-ckpt-sha256", self.args.encoder_sha256, "--shard-idx", str(shard),
                "--num-shards", str(num), "--demo-min-real", "17", "--exec-min-real", "33", *extra]

    def run_oracle(self) -> None:
        if self.smoke:
            gpu = self.gpus[0]
            rc, lines = self.runner.run(f"oracle-gpu{gpu}", self.oracle_cmd(0, 1, ["--segments", *self.smoke_keys()]),
                                        wan_env(gpu), self.logs / "oracle-smoke.log")
            line = find_line(lines, "ORACLE_ENCODER=DONE")
            if rc != 0 or line is None:
                raise BuildError("oracle", f"rc={rc} {line!r}")
            self.mark("oracle", "PASS", line)
            return
        n = len(self.gpus)
        jobs = [(f"oracle-gpu{g}", self.oracle_cmd(i, n, []), wan_env(g), self.logs / f"oracle-encoder-{i}.log")
                for i, g in enumerate(self.gpus)]
        res = self.runner.run_parallel(jobs)
        bad = [(i, rc) for i, (rc, lines) in enumerate(res)
               if rc != 0 or find_line(lines, f"ORACLE_ENCODER=DONE shard={i}/{n}") is None]
        if bad:
            raise BuildError("oracle", f"分片失败 {bad}")
        cmd = [*WAN_PY, str(WAN_DIR / "oracle_driver.py"), "--mj-repo", str(self.mj), "aggregate",
               "--manifest", str(self.manifest_path), "--out", str(self.oracle), "--num-shards", str(n),
               "--kind", "encoder", "--demo-min-real", "17", "--exec-min-real", "33"]
        rc, lines = self.runner.run("oracle-aggregate", cmd, wan_env(None), self.logs / "oracle-aggregate.log")
        line = find_line(lines, "ORACLE_ENCODER=DONE")
        agg = find_line(lines, "ORACLE_AGGREGATE=DONE")
        if rc != 0 or line is None or agg is None or f"rows={self.expect['rows']}" not in line:
            raise BuildError("oracle", f"aggregate rc={rc} {line!r}")
        self.mark("oracle", "PASS", f"{line} | ORACLE_SHARDS=PASS ok={n}/{n}")

    # ── pack / compare / checks / verify ──
    def pack(self) -> None:
        raw_dir = load_manifest(self.manifest_path)["raw_dir"]
        cmd = [sys.executable, str(_HERE / "pack_motion_store.py"), "pack", "--manifest", str(self.manifest_path),
               "--tokens", str(self.tokens), "--latents", str(self.latents), "--out", str(self.store),
               "--encoder-run-dir", str(self.ckpt.parent), "--raw-dir", raw_dir,
               "--oracle-report", str(self.latents.parent / "oracle" / "wan-mj" / "vae_report.json"),
               "--restore-receipt", str(self.receipt), "--expected-ckpt-sha256", self.args.encoder_sha256]
        if (self.store / ms.META_RELPATH).exists():
            cmd.append("--resume")
        rc, lines = self.runner.run("pack", cmd, main_env(), self.logs / "pack.log")
        line = require_line("pack", rc, lines, "PACK_MOTION_DONE=", "1")
        meta = json.loads((self.store / ms.META_RELPATH).read_text(encoding="utf-8"))
        if meta.get("status") != "packed":
            raise BuildError("pack", f"pack 后 status={meta.get('status')!r}")
        self.mark("pack", "PASS", line)

    def compare(self) -> None:
        if self.smoke:
            cmd = [sys.executable, str(WAN_DIR / "compare_wan.py"), "segtokens", "--tokens", str(self.tokens),
                   "--oracle", str(self.oracle)]
            rc, lines = self.runner.run("compare", cmd, main_env(), self.logs / "compare-smoke.log")
            self.mark("compare", "PASS", require_line("compare", rc, lines, "ENCODER_SEG_BITEXACT="))
            return
        meta = self.store / ms.META_RELPATH
        if not meta.is_file() or json.loads(meta.read_text(encoding="utf-8")).get("status") != "packed":
            raise BuildError("compare", "compare 必须在 pack（status=packed）之后、verify 之前运行")
        cmd = [sys.executable, str(WAN_DIR / "compare_wan.py"), "tokens", "--store", str(self.store),
               "--oracle", str(self.oracle)]
        rc, lines = self.runner.run("compare", cmd, main_env(), self.logs / "compare.log")
        line = require_line("compare", rc, lines, "ENCODER_BITEXACT=")
        if f"compared={self.expect['rows']} mismatches=0" not in line:
            raise BuildError("compare", f"逐位比较未覆盖全表: {line!r}")
        if not (self.store / pms.BITEXACT_RELPATH).is_file():
            raise BuildError("compare", "compare 未写 meta/encoder_bitexact.json")
        self.mark("compare", "PASS", line)

    def checks(self) -> None:
        mc = [sys.executable, str(_HERE / "motion_checks.py")]
        M, L, S, T = str(self.manifest_path), str(self.latents), str(self.store), str(self.tokens)
        jobs = [
            ("A6_SAMESOURCE=", [*mc, "a6set", "--manifest", M, "--framesamp", str(self.parent / "framesamp-8x8"),
                                "--motion", S]),
            ("A7_BYTES=", [*mc, "a7", "--manifest", M, "--latents", L, "--tokens", T, "--motion", S]),
            ("A9_INDEXSET=", [*mc, "a9set", "--manifest", M, "--latents", L, "--motion", S, "--n", "500", "--cold-all"]),
            ("A10_ROWS=", [*mc, "a10", "--manifest", M, "--motion", S, "--demo-min-real", "17",
                           "--expect-rows", str(self.expect["rows"]), "--expect-exec", str(self.expect["exec"]),
                           "--expect-demo", str(self.expect["demo"]), "--expect-episodes", str(self.expect["episodes"])]),
            ("A11_PAD=", [*mc, "a11", "--manifest", M, "--latents", L, "--demo-min-real", "17",
                          "--expect-rows", str(self.expect["rows"]), "--expect-padded", str(self.expect["padded"]),
                          "--expect-episodes", str(self.expect["eligible"])]),
        ]
        got = []
        for prefix, cmd in jobs:
            name = prefix.rstrip("=").lower()
            rc, lines = self.runner.run(name, cmd, main_env(), self.logs / f"check-{name}.log")
            got.append(require_line("checks", rc, lines, prefix))
        a7 = got[1]
        if f"row_bytes={ROW_BYTES}" not in a7:
            raise BuildError("checks", f"A7 未按 {ROW_BYTES} B/行核对: {a7!r}")
        self.lines.update({p.rstrip("="): ln for (p, _), ln in zip(jobs, got)})
        self.mark("checks", "PASS", " | ".join(got))

    def verify(self) -> None:
        cmd = [sys.executable, str(_HERE / "pack_motion_store.py"), "verify", "--store", str(self.store), "--resume"]
        rc, lines = self.runner.run("verify", cmd, main_env(), self.logs / "verify.log")
        line = require_line("verify", rc, lines, "VERIFY_MOTION=")
        if line != f"VERIFY_MOTION=PASS scanned={self.expect['rows']} mismatches=0":
            raise BuildError("verify", line)
        self.mark("verify", "PASS", line)

    def publish(self) -> str:
        meta = ms.MotionMeta.load(self.store)
        ms.require_verified(meta)
        raw = meta.raw
        table = raw["tables"][ms.MOTION_KEY]
        totals = raw["totals"]
        got = {"rows": meta.num_rows, "exec": totals["exec_rows"], "demo": totals["demo_rows"],
               "bytes": table["byte_count"]}
        if any(got[k] != self.expect[k] for k in got) or tuple(table["row_shape"]) != ROW_SHAPE or table["row_bytes"] != ROW_BYTES:
            raise BuildError("publish", f"总量或行合同不符 {got} {table['row_shape']} {table['row_bytes']}")
        prov = raw["provenance"]
        enc = prov["encoder"]
        if enc["checkpoint_sha256"] != self.args.encoder_sha256 or int(enc["epoch"]) != EXPECTED_EPOCH:
            raise BuildError("publish", "store provenance 的 encoder 身份不是本次最终 ckpt")
        bitexact = pms.check_bitexact_record(self.store, raw)
        norm = self.parent.parent.parent / "train-assets" / "mme_vla_suite" / self.parent.name / "robomme" / "norm_stats.json"
        record = {
            "schema": "mixed8-v10-motion-build-record/v1", "status": "PASS", "finished_at_utc": now_utc(),
            "identity": json.loads((self.build / "identity.json").read_text(encoding="utf-8")),
            "store": str(self.store), "store_meta_sha256": sha256_file(self.store / ms.META_RELPATH),
            "layout": raw["layout"], "rows": got["rows"], "exec": got["exec"], "demo": got["demo"],
            "bytes": got["bytes"], "row_shape": list(ROW_SHAPE), "row_bytes": ROW_BYTES,
            "table_sha256": table["sha256"], "motion_index_sha256": raw["motion_index_sha256"],
            "source_run": f"{enc['run_name']}/{enc['checkpoint_name']}#{enc['state_key']}",
            "encoder_checkpoint_sha256": enc["checkpoint_sha256"], "encoder_epoch": enc["epoch"],
            "vae_compat": prov.get("vae_compat"), "bitexact": bitexact,
            "parent_references": {  # 直接引用，不复制、不硬链
                "lib": str(self.parent), "framesamp_8x8": str(self.parent / "framesamp-8x8"),
                "source": str(self.parent / "source"), "norm_stats": str(norm),
                "norm_sha256": sha256_file(norm) if norm.is_file() else None},
            "copied_meta": {n: sha256_file(self.meta_dir / n) for n in ("episode_manifest.json", "input_manifest.json")},
            "stage_lines": {k: v.get("detail") for k, v in self.state.get("stages", {}).items()},
        }
        write_json(self.build / "motion_build_record.json", record)
        line = (f"MOTION_BUILD=PASS rows={got['rows']} exec={got['exec']} demo={got['demo']} bytes={got['bytes']} "
                f"row_bytes={ROW_BYTES} ckpt={enc['checkpoint_sha256'][:16]} epoch={enc['epoch']} "
                f"table={table['sha256'][:16]} store={self.store}")
        self.mark("publish", "PASS", f"record={sha256_file(self.build / 'motion_build_record.json')[:16]}")
        return line

    def run(self) -> str:
        self.preflight()
        steps = {"encode": self.encode, "oracle": self.run_oracle, "pack": self.pack, "compare": self.compare,
                 "checks": self.checks, "verify": self.verify}
        for name in self.stages[1:]:
            if name == "publish":
                return self.publish()
            if self.stage_done(name):
                print(f"MOTION_STAGE={name} SKIP 已 PASS（同身份续跑）", flush=True)
                continue
            steps[name]()
        n = len(self.smoke_keys())
        rows = sum(len(json.loads((self.tokens / f"{k}.metadata.json").read_text())["input_block_sha256"])
                   for k in self.smoke_keys())
        return (f"MOTION_BUILD_SMOKE=PASS segments={n} rows={rows} bytes={rows * ROW_BYTES} row_bytes={ROW_BYTES} "
                f"ckpt={self.args.encoder_sha256[:16]} epoch={EXPECTED_EPOCH}")


def main_env_with_gpus(gpus: list[int]) -> dict:
    env = main_env()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))     # run_local 再为每个 worker 单独设置
    return env


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--restore-receipt", required=True, help="S5-A restore_receipt.json（只读消费）")
    ap.add_argument("--encoder-ckpt", required=True, help="注册副本绝对路径 …/checkpoint_epoch_60.pt（只读）")
    ap.add_argument("--encoder-sha256", required=True)
    ap.add_argument("--out-store", required=True, help="新库根（首跑必须不存在或为空）")
    ap.add_argument("--gpus", default="0-7")
    ap.add_argument("--max-segments", type=int, default=None, help="仅冒烟：清单序前 N 段，不 pack、不发布")
    ap.add_argument("--resume", action="store_true", help="只接受 build/identity.json 完全相同的续跑")
    ap.add_argument("--parent-lib", default=str(DEFAULT_PARENT))
    ap.add_argument("--mj-repo", default=str(DEFAULT_MJ))
    ap.add_argument("--require-free-mib", type=int, default=20000)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    build = Build(args)
    try:
        line = build.run()
    except Exception as exc:  # noqa: BLE001 — 任何失败都落 FAIL 判定行与阶段状态
        stage = getattr(exc, "stage", "unknown")
        if build.build.is_dir() and build.state:
            build.mark(stage, "FAIL", str(exc)[:500])
        tag = "MOTION_BUILD_SMOKE" if build.smoke else "MOTION_BUILD"
        print(f"{tag}=FAIL stage={stage} reason={exc}", flush=True)
        return 1
    print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
