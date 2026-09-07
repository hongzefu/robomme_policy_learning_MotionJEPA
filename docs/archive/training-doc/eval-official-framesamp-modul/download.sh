#!/usr/bin/env bash
# 官方 MME-VLA perceptual-framesamp-modul 权重下载：HF 公开仓 Yinpei/perceptual-framesamp-modul，钉 commit c0f565dd…，
# 落 v1-store/models/official-mme-vla/perceptual-framesamp-modul/{79999.zip,history_config.txt,README.md}，zip 按 HF LFS sha256 核对。
# 解压另行用 scripts/training/unzip_ckpt.py（zip 内路径剥到 79999/ 下）。日志 v1-store/logs/evoff-download.log，结束 EXIT_CODE=。
set -o pipefail
MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA
cd "$MAIN"
export HF_HOME=$MAIN/v1-store/cache/hf UV_LINK_MODE=copy PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE
DST=$MAIN/v1-store/models/official-mme-vla/perceptual-framesamp-modul
LOG=$MAIN/v1-store/logs/evoff-download.log
{
echo "DL_START=$(date '+%F %T') dst=$DST"
uv run --no-sync python - "$DST" <<'PYEOF'
import sys, pathlib, hashlib, time
from huggingface_hub import hf_hub_download, HfApi
repo, rev = "Yinpei/perceptual-framesamp-modul", "c0f565dd40082f32863b3ee9db99de5ed5d3d4e0"
dst = pathlib.Path(sys.argv[1]); dst.mkdir(parents=True, exist_ok=True)
info = HfApi().model_info(repo, revision=rev, files_metadata=True)
lfs = {s.rfilename: (s.lfs.sha256 if s.lfs else None, s.size) for s in info.siblings}
print("repo", repo, "revision", info.sha, {k: v[1] for k, v in lfs.items()}, flush=True)
for name in ("history_config.txt", "README.md", "79999.zip"):
    t0 = time.time()
    p = hf_hub_download(repo, name, revision=rev, local_dir=str(dst))
    sz = pathlib.Path(p).stat().st_size
    print(f"got {name} size={sz} expected={lfs[name][1]} in {time.time()-t0:.0f}s", flush=True)
    if lfs[name][0]:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        ok = h.hexdigest() == lfs[name][0]
        print(f"SHA256 {name} local={h.hexdigest()} hf_lfs={lfs[name][0]} match={ok}", flush=True)
        if not ok:
            sys.exit(3)
print("history_config.txt =", (dst / "history_config.txt").read_text().strip())
print("DOWNLOAD=PASS")
PYEOF
EC=$?
echo "DL_END=$(date '+%F %T')"
echo "EXIT_CODE=$EC"
} 2>&1 | grep -v dev-dependencies | tee -a "$LOG"
