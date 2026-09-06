#!/usr/bin/env bash
# 官方 MME-VLA perceptual-framesamp-context 权重下载：HF 合集仓 Yinpei/mme_vla_suite（单模型仓只发布了 modul），钉 commit 5db4d53d…，
# 文件 perceptual-framesamp-context/{79999.zip,history_config.txt} 落 v1-store/models/official-mme-vla/perceptual-framesamp-context/，
# zip 按 HF LFS sha256 与钉死值三方核对；history_config.txt 须为 'perceptual-framesamp-context.yaml' 无换行（create_trained_policy 原文当文件名）。
# 解压另行用 scripts/training/unzip_ckpt.py（zip 内路径剥到 79999/ 下）。日志 v1-store/logs/evoffctx-download.log，结束 EXIT_CODE=。
set -o pipefail
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
cd "$MAIN"
export HF_HOME=$MAIN/v1-store/cache/hf UV_LINK_MODE=copy PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE
DST=$MAIN/v1-store/models/official-mme-vla
LOG=$MAIN/v1-store/logs/evoffctx-download.log
{
echo "DL_START=$(date '+%F %T') dst=$DST"
uv run --no-sync python - "$DST" <<'PYEOF'
import sys, pathlib, hashlib, time
from huggingface_hub import hf_hub_download, HfApi
repo, rev = "Yinpei/mme_vla_suite", "5db4d53ddb98c7f80cab08792dd53d985d712ab1"
SUB = "perceptual-framesamp-context"
PINNED = {f"{SUB}/79999.zip": "387d4bd500a29588d81c2675a83be351f84383b93bea4161b8f7fadbf91a8ae8"}
dst = pathlib.Path(sys.argv[1]); dst.mkdir(parents=True, exist_ok=True)
info = HfApi().model_info(repo, revision=rev, files_metadata=True)
lfs = {s.rfilename: (s.lfs.sha256 if s.lfs else None, s.size) for s in info.siblings}
print("repo", repo, "revision", info.sha, {k: v[1] for k, v in lfs.items() if k.startswith(SUB)}, flush=True)
assert info.sha == rev, info.sha
for name in (f"{SUB}/history_config.txt", f"{SUB}/79999.zip"):
    t0 = time.time()
    p = hf_hub_download(repo, name, revision=rev, local_dir=str(dst))
    sz = pathlib.Path(p).stat().st_size
    print(f"got {name} size={sz} expected={lfs[name][1]} in {time.time()-t0:.0f}s", flush=True)
    if sz != lfs[name][1]:
        print("SIZE_MISMATCH"); sys.exit(4)
    if lfs[name][0]:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        pinned = PINNED.get(name)
        ok = h.hexdigest() == lfs[name][0] == pinned
        print(f"SHA256 {name} local={h.hexdigest()} hf_lfs={lfs[name][0]} pinned={pinned} match={ok}", flush=True)
        if not ok:
            sys.exit(3)
hc = (dst / SUB / "history_config.txt").read_text()
print("history_config.txt =", repr(hc))
if hc != "perceptual-framesamp-context.yaml":
    print("HISTORY_CONFIG_MISMATCH"); sys.exit(5)
print("DOWNLOAD=PASS")
PYEOF
EC=$?
echo "DL_END=$(date '+%F %T')"
echo "EXIT_CODE=$EC"
} 2>&1 | grep -v dev-dependencies | tee -a "$LOG"
