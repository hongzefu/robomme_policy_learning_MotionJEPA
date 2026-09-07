#!/usr/bin/env bash
# 官方 MME-VLA perceptual-framesamp-modul 权重下载（本机 / 环境 A 口径）。
# 源同 docs/archive/training-doc/eval-official-framesamp-modul/download.sh：HF 公开仓 Yinpei/perceptual-framesamp-modul，
# 钉 commit c0f565dd…；与那份的唯一差别是 MAIN 由环境 B 的 /scratch/hongze 改为本机 /data/hongzefu，并显式传 HF_TOKEN
# （HF_HOME 改指 v1-store/cache/hf 后默认 token 不再自动可见，仓库虽公开但统一口径带上）。
# 落 v1-store/models/official-mme-vla/perceptual-framesamp-modul/{79999.zip,history_config.txt,README.md}，zip 按 HF LFS sha256 核对。
# 解压另行用 scripts/training/unzip_ckpt.py。日志 v1-store/logs/evhard-dl-modul.log，结束写 EXIT_CODE=。
set -o pipefail
MAIN=/data/hongzefu/robomme_policy_learning_MotionJEPA
cd "$MAIN"
export HF_HOME=$MAIN/v1-store/cache/hf UV_LINK_MODE=copy PYTHONUNBUFFERED=1
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
unset HF_HUB_OFFLINE
DST=$MAIN/v1-store/models/official-mme-vla/perceptual-framesamp-modul
LOG=$MAIN/v1-store/logs/evhard-dl-modul.log
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
