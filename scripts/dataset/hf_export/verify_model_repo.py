#!/usr/bin/env python3
"""L1 验收：不下载文件体，只用**内容寻址元数据**证明 HF 上的字节 == 本地字节。

为什么不用 `usedStorage` / 网页显示的体积：上次 bucket 导出实测该统计接口异步滞后
（480 GB 传完后一度显示 312 GB / 200 文件），而且它是字节总量，对**等长篡改**完全失明，
失败时也不告诉你是哪个文件坏了。

内容寻址的两把尺子（都由 Hub 侧按内容定义，不是我们自报的字段）：
  - LFS 对象：``lfs.sha256`` 就是文件内容的 sha256；
  - 非 LFS 文件：``blob_id`` 是 git 的 ``sha1("blob <len>\\0" + content)``。
诚实边界：``lfs.sha256`` 是 commit 时客户端随 payload 提交的，能证明「Hub 记录的就是我这份
文件的 sha」（挡传错、挡漏传），不能证明「Hub 存的字节确实哈希成这个值」——后者由 driver
阶段 6 的回读闭环（``sha256sum -c``）兜底。

用法：
  uv run --no-sync python scripts/dataset/hf_export/verify_model_repo.py \\
      --repo-id HongzeFu/MotionJEPA --src v1-store/external/motionjepa/wan-v8-filter10-72ep-a \\
      --stage v1-store/exports/hf-model-motionjepa-encoder-v1/stage --json <out.json>
判定行：``HF_META=PASS files=N lfs_mismatch=0 blob_mismatch=0`` 与 ``HF_PRIVATE=True commit=<40 hex>``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

from huggingface_hub import HfApi

CHUNK = 8 * 1024 * 1024


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(p: pathlib.Path) -> str:
    """与 `git hash-object` 同口径：sha1(b"blob <len>\\0" + content)。"""
    data = p.read_bytes()
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--src", required=True, help="run 目录（本机权重原件）")
    ap.add_argument("--stage", required=True, help="上传用的 stage 目录，作为本地真值")
    ap.add_argument("--json", default=None, help="把远端元数据快照写到该路径")
    args = ap.parse_args()

    stage = pathlib.Path(args.stage)
    api = HfApi()
    info = api.repo_info(args.repo_id, repo_type="model")
    tree = {
        f.path: f
        for f in api.list_repo_tree(args.repo_id, repo_type="model", recursive=True)
        if getattr(f, "size", None) is not None
    }

    lfs_bad: list[str] = []
    blob_bad: list[str] = []
    snapshot: dict[str, dict] = {}
    for path, f in sorted(tree.items()):
        lfs_sha = f.lfs.sha256 if getattr(f, "lfs", None) is not None else None
        snapshot[path] = {"size": f.size, "blob_id": f.blob_id, "lfs_sha256": lfs_sha}
        local = stage / path
        if not local.is_file():
            # .gitattributes 是 Hub 建仓时自带的，本地 stage 里没有，不参与比对
            snapshot[path]["local"] = "absent-in-stage"
            continue
        size_ok = f.size == local.stat().st_size
        if lfs_sha is not None:
            want = sha256_file(local)
            snapshot[path]["local_sha256"] = want
            if not size_ok or lfs_sha != want:
                lfs_bad.append(path)
        else:
            want = git_blob_sha1(local)
            snapshot[path]["local_blob_id"] = want
            if not size_ok or f.blob_id != want:
                blob_bad.append(path)

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"repo_id": args.repo_id, "private": info.private, "commit": info.sha, "files": snapshot},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    ok = not lfs_bad and not blob_bad
    if lfs_bad:
        print(f"  LFS 不符: {lfs_bad}")
    if blob_bad:
        print(f"  blob 不符: {blob_bad}")
    print(f"HF_META={'PASS' if ok else 'FAIL'} files={len(tree)} "
          f"lfs_mismatch={len(lfs_bad)} blob_mismatch={len(blob_bad)}")
    print(f"HF_PRIVATE={info.private} commit={info.sha}")
    if info.private is not True:
        print("错误: repo 不是 private，立即用 `hf repos settings <repo> --private` 改回")
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
