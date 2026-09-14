#!/usr/bin/env python3
"""把 dataset repo 的内容搬进 bucket，优先走**服务端按 xet hash 复制**（零字节传输）。

**为什么不是「下载再上传」**：`batch_bucket_files(copy=[...])` 让服务端按内容哈希直接把对象
挂到目标 bucket 上，文档原文 "This is a server-side operation — no data is downloaded or
re-uploaded"。本轮实测这个 repo 的 2057 个文件里 **1978 个（126.59 GB）带 xetHash**，只有
79 个普通 git blob（22.12 MB，meta/ 的 74 个 + MANIFEST.json / SHA256SUMS / README.md /
tarxz_h5.py / .gitattributes）没有，需要客户端中转。也就是说 99.98% 的字节根本不用过本地。

附带的好处是**逐位同一性由内容寻址本身保证**：复制后两侧 xetHash 相等，就是「同一份内容」的
直接证据，而不是「我下下来又传上去、但愿中间没坏」。这一层零成本，但**不替代全量回读**——
它证明的是 Hub 的元数据认为两边同 hash，不证明取回来的字节是对的。两层都要做。

约束：`hf cp` 的帮助文本明写 "Remote-to-remote copies only work within the same storage
region"。源 repo 的 region tag 是 us，新建 bucket 落在哪个区不由我们控制，所以 `--smoke`
必须先拿**一个真实文件**验证复制能成、且能原样取回，再走全量。

子命令：
  probe  取 repo 全部文件的 xetHash，写清单 JSON（可反复跑，纯读）
  smoke  在临时 bucket 上验证服务端复制可用 + 长文件名可落盘，跑完删掉临时 bucket
  copy   按清单分批复制到目标 bucket（幂等：每批一个 .done 标记）
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

API = "https://huggingface.co/api"
COPY_BATCH = 200          # copy 不搬字节，负担只与条目数相关
PLAIN_BATCH = 40


def _tok() -> str:
    t = os.environ.get("HF_TOKEN")
    if not t:
        sys.exit("错误: 环境里没有 HF_TOKEN")
    return t


def _post(url: str, payload: dict) -> list:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {_tok()}", "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def _get(url: str):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_tok()}"})
    return json.load(urllib.request.urlopen(req))


def cmd_probe(args) -> int:
    info = _get(f"{API}/datasets/{args.repo}/revision/{args.revision}?blobs=true")
    sizes = {s["rfilename"]: s.get("size", 0) for s in info["siblings"]}
    paths = sorted(sizes)
    xet, plain = {}, []
    for i in range(0, len(paths), 200):
        for e in _post(f"{API}/datasets/{args.repo}/paths-info/{args.revision}",
                       {"paths": paths[i:i + 200], "expand": True}):
            if e.get("type") != "file":
                continue
            (xet.__setitem__(e["path"], e["xetHash"]) if e.get("xetHash")
             else plain.append(e["path"]))
    out = {"repo": args.repo, "revision": args.revision, "sizes": sizes,
           "xet": xet, "plain": sorted(plain)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"PROBE_FILES={len(paths)} XET={len(xet)} PLAIN={len(plain)} "
          f"XET_BYTES={sum(sizes[p] for p in xet)} PLAIN_BYTES={sum(sizes[p] for p in plain)}")
    # 最长 basename——smoke 必须拿它做，否则验不出 NAME_MAX 边界
    worst = max(paths, key=lambda p: len(p.split("/")[-1].encode()))
    print(f"LONGEST_BASENAME={len(worst.split('/')[-1].encode())} PATH={worst}")
    return 0


def cmd_smoke(args) -> int:
    from huggingface_hub import HfApi
    api = HfApi()
    man = json.loads(args.manifest.read_text())
    # 刻意挑 basename 最长的那个：xfs 的 NAME_MAX 是 255，这个 repo 里最长的达 247 字节，
    # 只剩 8 字节余量。若下载端给临时文件加后缀，就会在回读阶段 ENAMETOOLONG——那时已经
    # 复制完 126 GB 了。这个 smoke 的全部意义就是把那个失败提前到第一分钟。
    worst = max(man["xet"], key=lambda p: len(p.split("/")[-1].encode()))
    blen = len(worst.split("/")[-1].encode())
    print(f"SMOKE_TARGET bytes={man['sizes'][worst]} basename_bytes={blen}\n  {worst}")

    tmp_bucket = args.tmp_bucket
    created = False
    try:
        try:
            api.create_bucket(tmp_bucket)
            created = True
            print(f"  临时 bucket 已建: {tmp_bucket}")
        except Exception as e:
            print(f"  临时 bucket 创建返回: {type(e).__name__}: {e}")
        api.batch_bucket_files(tmp_bucket,
                               copy=[("dataset", man["repo"], man["xet"][worst], worst)])
        print("SERVERSIDE_COPY=OK")

        dest = args.workdir / "smoke"
        dest.mkdir(parents=True, exist_ok=True)
        local = dest / Path(worst).name
        api.download_bucket_files(tmp_bucket, [(worst, str(local))])
        got = hashlib.sha256(local.read_bytes()).hexdigest()
        size_ok = local.stat().st_size == man["sizes"][worst]
        print(f"LONGNAME_ROUNDTRIP={'OK' if size_ok else 'FAIL'} "
              f"bytes={local.stat().st_size} sha256={got[:16]}…")
        if not size_ok:
            return 1
        local.unlink()
    finally:
        if created:
            try:
                api.delete_bucket(tmp_bucket)
                print(f"  临时 bucket 已删: {tmp_bucket}")
            except Exception as e:
                print(f"  警告: 临时 bucket 删除失败，请手工清理 {tmp_bucket}: {e}")
    return 0


def cmd_copy(args) -> int:
    from huggingface_hub import HfApi
    api = HfApi()
    man = json.loads(args.manifest.read_text())
    done_dir = args.workdir / "copied"
    done_dir.mkdir(parents=True, exist_ok=True)

    # 按目录分组再按条目数切批：目录是天然的断言单位，失败只需重放一个目录
    groups: dict[str, list[str]] = defaultdict(list)
    for p in sorted(man["xet"]):
        groups["/".join(p.split("/")[:-1]) or "<root>"].append(p)

    batches: list[tuple[str, list[str]]] = []
    for g in sorted(groups):
        for i in range(0, len(groups[g]), COPY_BATCH):
            batches.append((f"{g.replace('/', '_')}-{i // COPY_BATCH:02d}",
                            groups[g][i:i + COPY_BATCH]))
    print(f"COPY_BATCHES={len(batches)} XET_FILES={sum(len(b[1]) for b in batches)}")

    for k, (bid, files) in enumerate(batches, 1):
        mark = done_dir / f"{bid}.done"
        if mark.exists():
            print(f"  [{k}/{len(batches)}] {bid} 已完成，跳过")
            continue
        ok = False
        for attempt in range(1, 9):
            try:
                api.batch_bucket_files(
                    args.bucket,
                    copy=[("dataset", man["repo"], man["xet"][p], p) for p in files])
                ok = True
                break
            except Exception as e:
                print(f"  {bid} 第 {attempt} 次失败: {type(e).__name__}: {str(e)[:160]}")
                import time
                time.sleep(30)
        if not ok:
            sys.exit(f"错误: 批 {bid} 八次仍失败")
        mark.write_text("")
        print(f"  COPY_OK [{k}/{len(batches)}] {bid} files={len(files)}")

    # 79 个普通 git blob（22 MB）：没有 xetHash，只能下载再上传
    plain = [p for p in man["plain"] if p != ".gitattributes"]   # LFS 追踪规则，bucket 无意义
    if plain:
        from huggingface_hub import hf_hub_download
        dl = args.workdir / "plain"
        dl.mkdir(parents=True, exist_ok=True)
        add = []
        for p in plain:
            # cache 模式（不用 local_dir）：blob 以 etag 命名，避开长文件名边车问题
            f = hf_hub_download(man["repo"], p, repo_type="dataset",
                                revision=man["revision"], cache_dir=str(dl))
            add.append((os.path.realpath(f), p))
        for i in range(0, len(add), PLAIN_BATCH):
            chunk = add[i:i + PLAIN_BATCH]
            api.batch_bucket_files(args.bucket, add=chunk)
            print(f"  PLAIN_OK batch={i // PLAIN_BATCH} files={len(chunk)}")
    print(f"COPY_DONE xet={len(man['xet'])} plain={len(plain)} "
          f"total={len(man['xet']) + len(plain)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("smoke")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--tmp-bucket", required=True)
    p.add_argument("--workdir", type=Path, required=True)
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("copy")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--workdir", type=Path, required=True)
    p.set_defaults(func=cmd_copy)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
