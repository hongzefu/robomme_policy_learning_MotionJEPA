#!/usr/bin/env python3
"""把 dataset repo 上那些 **bucket 结构上装不下** 的元信息原样抓进留档目录。

repo 删除之后，下面这些一个字都拿不回来，而 bucket 只是个对象存储、没有对应概念：

  revision.json      revision 全量元数据（含每个 blob 的 oid/size/lfs）
  tree.json          递归文件树（含 xetHash 与每文件 lastCommit）
  commits.json       提交历史（id / title / date / author）
  paths_info.json    全部路径的 paths-info（xetHash 是服务端复制与事后对拍的依据）
  discussions.json   讨论区与 PR（删库会一并删掉）
  croissant.json     Croissant 元数据（dataset viewer 的机器可读描述）
  README.raw.md      README 原文，**含 YAML front matter**（license / task_categories /
                     tags / pretty_name）——bucket 不渲染 dataset card，这段机器可读的
                     license 声明事实上会消失，必须留底
  .gitattributes     LFS 追踪规则
  summary.json       下载数 / likes / 创建时间 / sha 等一次性快照

用法：
  uv run scripts/dataset/hf_export/archive_h5v2_metadata.py \
      --repo HongzeFu/robomme-4task-h5-20260912-v2 --revision <sha> --out <records 目录>
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://huggingface.co/api"


def _req(url: str, payload: dict | None = None):
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        sys.exit("错误: 环境里没有 HF_TOKEN")
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Authorization": f"Bearer {tok}"}
    if data:
        headers["Content-Type"] = "application/json"
    return json.load(urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers)))


def _save(out: Path, name: str, obj) -> None:
    p = out / name
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2))
    print(f"  saved {name}  {p.stat().st_size} B")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    R, REV = args.repo, args.revision

    info = _req(f"{API}/datasets/{R}/revision/{REV}?blobs=true")
    _save(out, "revision.json", info)

    paths = sorted(s["rfilename"] for s in info["siblings"])
    pinfo = []
    for i in range(0, len(paths), 200):
        pinfo += _req(f"{API}/datasets/{R}/paths-info/{REV}",
                      {"paths": paths[i:i + 200], "expand": True})
    _save(out, "paths_info.json", pinfo)

    # tree 分页取全
    tree, cursor = [], None
    while True:
        # expand=true 时 limit 上限是 100（配 limit=1000 直接 HTTP 400，实测）
        url = f"{API}/datasets/{R}/tree/{REV}?recursive=true&expand=true&limit=100"
        if cursor:
            url += f"&cursor={cursor}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"})
        with urllib.request.urlopen(req) as r:
            tree += json.load(r)
            link = r.headers.get("Link", "")
        if 'rel="next"' not in link:
            break
        cursor = link.split("cursor=")[1].split(">")[0].split("&")[0]
    _save(out, "tree.json", tree)

    for name, url in (("commits.json", f"{API}/datasets/{R}/commits/main"),
                      ("discussions.json", f"{API}/datasets/{R}/discussions"),
                      ("croissant.json", f"{API}/datasets/{R}/croissant")):
        try:
            _save(out, name, _req(url))
        except urllib.error.HTTPError as e:
            _save(out, name, {"_error": f"HTTP {e.code}", "_url": url})

    for fname, saveas in (("README.md", "README.raw.md"), (".gitattributes", "gitattributes.raw.txt")):
        try:
            u = f"https://huggingface.co/datasets/{R}/raw/{REV}/{fname}"
            req = urllib.request.Request(u, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"})
            _save(out, saveas, urllib.request.urlopen(req).read().decode())
        except urllib.error.HTTPError as e:
            print(f"  警告: 取 {fname} 失败 HTTP {e.code}")

    ds = _req(f"{API}/datasets/{R}")
    _save(out, "summary.json", {
        "id": ds.get("id"), "sha": ds.get("sha"), "private": ds.get("private"),
        "downloads": ds.get("downloads"), "likes": ds.get("likes"),
        "createdAt": ds.get("createdAt"), "lastModified": ds.get("lastModified"),
        "tags": ds.get("tags"), "cardData": ds.get("cardData"),
        "usedStorage": ds.get("usedStorage"), "siblings_count": len(info["siblings"]),
        "_note": "删除 dataset repo 前的一次性快照；bucket 没有 card/revision/discussions 概念。",
    })
    xet = sum(1 for e in pinfo if e.get("xetHash"))
    print(f"SRC_SNAPSHOT_SAVED files={len(paths)} sha={REV[:12]} xet={xet} "
          f"commits={len(_req(f'{API}/datasets/{R}/commits/main'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
