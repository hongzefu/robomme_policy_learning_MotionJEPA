#!/usr/bin/env python3
"""CPU 离线契约验收：伪造小权重和 HF 命令，所有产物只写 --out。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from omegaconf import OmegaConf


RUN = "wan-mixed8-200-st512q16-b88x8-60ep-a"
BUCKET = "HongzeFu/motionjepa-wan-mixed8-200-st512q16-60ep-v1"
EXPORT = "hf-mixed8-200-v9st-encoder-ckpt"
FINAL = "run_mixed8_200_v9st_encoder_ckpt_export.sh"
INC = "mj_v9st_ckpt_inc.sh"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run(argv, env, log, expected=0):
    result = subprocess.run(argv, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=180)
    log.write_text(result.stdout)
    if expected == 0:
        assert result.returncode == 0, f"{log}: rc={result.returncode}\n{result.stdout[-4000:]}"
    else:
        assert result.returncode != 0, f"{log}: 应失败却成功"
    return result.stdout


# 假命令只读写审计目录；不导入网络库，也不执行真正的 HF/HTTP 客户端。
FAKE = r'''import fnmatch, json, os, pathlib, shutil, sys
p = pathlib.Path
root = p(os.environ["FAKE_ROOT"])
state_path, bucket = root/"state.json", root/"bucket"
state = json.loads(state_path.read_text())
name = p(sys.argv[0]).name
args = sys.argv[1:]
def event(kind, **more):
    with (root/"events.jsonl").open("a") as f:
        f.write(json.dumps(dict(kind=kind, **more))+"\n")
def save(): state_path.write_text(json.dumps(state))
def local(value):
    prefix="hf://buckets/"+os.environ["TEST_BUCKET"]
    return bucket/value[len(prefix):].lstrip("/") if value.startswith(prefix) else p(value)
def unsigned():
    home=p(os.environ.get("HF_HOME","/"))
    return ("HF_TOKEN" not in os.environ and os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN")=="1"
            and home.is_relative_to(root.parent) and "tmp" in home.parts)
if name=="curl":
    event("list")
    if os.environ.get("FAKE_API_FAIL"): sys.exit(7)
    files=[x for x in bucket.rglob("*") if x.is_file()]
    print(json.dumps([dict(id=os.environ["TEST_BUCKET"], totalFiles=len(files), size=sum(x.stat().st_size for x in files), private=state["private"])] if state["exists"] else []))
elif name=="tmux": sys.exit(1)
elif name=="sleep": event("sleep", args=args)
elif name=="uvx":
    args=args[args.index("hf")+1:]
    if args[:2]==["auth","whoami"]: print("HongzeFu")
    elif args[:2]==["buckets","create"]:
        event("create"); state["exists"]=True; state["private"]=False; bucket.mkdir(exist_ok=True); save()
    elif args[:2]==["buckets","settings"]:
        assert "--public" in args; event("public"); state["private"]=False; save()
    elif args[:2]==["buckets","cp"]:
        src,dst=local(args[2]),local(args[3]); dst.parent.mkdir(parents=True,exist_ok=True)
        if args[2].startswith("hf://"):
            assert unsigned(); event("anonymous_ckpt")
            state["readbacks"]=state.get("readbacks",0)+1; save()
            if state["readbacks"]<=int(os.environ.get("FAKE_READBACK_FAILURES","0")): sys.exit(1)
        else: event("upload", name=src.name)
        shutil.copy2(src,dst)
        if args[2].startswith("hf://") and os.environ.get("FAKE_READBACK_BAD_SHA"):
            dst.write_bytes(b"corrupt")
    elif args[0]=="sync":
        src,dst=local(args[1]),local(args[2]); includes=[args[i+1] for i,a in enumerate(args) if a=="--include"]
        download=args[1].startswith("hf://")
        if download and includes:
            assert unsigned(); event("anonymous_readme")
        else: event("download" if download else "upload_batch")
        for file in src.rglob("*"):
            if not file.is_file(): continue
            rel=str(file.relative_to(src))
            if includes and not any(fnmatch.fnmatch(rel,x) for x in includes): continue
            target=dst/rel; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(file,target)
        if download and not includes and os.environ.get("FAKE_CORRUPT"):
            (dst/"checkpoint_epoch_4.pt").write_bytes(b"读回篡改")
    else: raise AssertionError(args)
else: raise AssertionError(name)
'''.replace('b"读回篡改"', '"读回篡改".encode()')


def missing_marker_cases(out):
    """只核已有 done 丢失身份标记的启动门，不执行上传或重复全套回归。"""
    out.mkdir(parents=True, exist_ok=True)
    repo, source, commands = out/"repo", out/"source", out/"bin"
    source.mkdir(); commands.mkdir()
    target = repo/"scripts/dataset/hf_export"/INC
    target.parent.mkdir(parents=True)
    original = (Path(__file__).resolve().parent/INC).read_text()
    needle = 'SRC="/scratch/hongze/MotionJEPA/runs/$RUN_NAME"'
    assert original.count(needle) == 1
    target.write_text(original.replace(needle, f'SRC="{source}"'))
    secrets = repo/"v1-store/secrets/hf.env"
    secrets.parent.mkdir(parents=True)
    secrets.write_text('printf read >> "$TEST_CREDENTIAL_READ"\nHF_TOKEN=offline_test_fixture\n')
    for name, tail in (("uvx", "echo HongzeFu"), ("tmux", "exit 1"), ("curl", "exit 99")):
        command = commands/name
        command.write_text('#!/usr/bin/env bash\nprintf call >> "$TEST_HF_EVENTS"\n' + tail + '\n')
        command.chmod(0o755)
    env = os.environ.copy()
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "SRC_OVERRIDE", "DRYRUN_ALIVE"):
        env.pop(key, None)
    env.update(PATH=str(commands)+os.pathsep+env["PATH"],
               TEST_HF_EVENTS=str(out/"hf-events"), TEST_CREDENTIAL_READ=str(out/"credential-read"))
    for label, epochs in (("all-done", range(2, 61, 2)), ("partial-done", (2,))):
        logs = repo/"v1-store/exports"/label/"logs"
        (logs/"uploaded").mkdir(parents=True)
        for epoch in epochs:
            (logs/"uploaded"/f"ckpt-epoch-{epoch}.done").write_text("a"*64 + "\n")
        (logs/"first-readback.done").write_text("a"*64 + "\n")
        text = run(["bash", str(target), RUN, BUCKET+"-changed", label, "fake-train"],
                   env, out/f"{label}.log", expected=1)
        assert "UPLOAD_MARKER_MISMATCH" in text and "INC_END" not in text
        assert not (logs/"uploaded.marker").exists()
        assert not (out/"hf-events").exists() and not (out/"credential-read").exists()
    dump(out/"result.json", dict(status="PASS", cases=2, hf_calls=0, credential_reads=0,
                                checks=["all-done+missing-marker", "partial-done+missing-marker"]))
    print(f"MISSING_MARKER_SELFTEST=PASS cases=2 out={out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--missing-marker-only", action="store_true", help="只跑已有 done 丢失 marker 的两个启动反例")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.missing_marker_only:
        missing_marker_cases(out)
        return
    here = Path(__file__).resolve().parent
    mj = Path("/scratch/hongze/MotionJEPA")
    source = out/"source"
    source.mkdir()
    for epoch in range(2, 61, 2):
        torch.save(dict(epoch=epoch, arch="wan-latent-v9-st", ema_only=True,
                        has_live_weights=False, encoder={"weight":torch.ones(1)}),
                   source/f"checkpoint_epoch_{epoch}.pt")
    # 使用计划锚点的真实默认配置，按训练器逻辑合成冻结路径，不导入训练器。
    defaults = subprocess.check_output(["git", "-C", str(mj), "show", "2fbf980:configs/default.yaml"], text=True)
    mixed8 = subprocess.check_output(["git", "-C", str(mj), "show", "2fbf980:configs/mixed8_16task_200.yaml"], text=True)
    cfg = OmegaConf.merge(OmegaConf.create(defaults), OmegaConf.create(mixed8))
    cfg.run_name = RUN
    cfg.data.data_root = "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/datasets"
    dataset = Path(cfg.data.data_root)/str(cfg.data.dataset)
    for key, value in {"_dataset_dir":dataset, "_chunk_motion_path":dataset/"chunk_motion.npz",
                       "_latent_dir":dataset/str(cfg.wan.latent_dir), "_arm_mask_dir":dataset/str(cfg.data.arm_mask_dir)}.items():
        cfg.data[key] = str(value)
    (source/"config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    rows = [dict(epoch=e, loss_total=.2, **{"loss_raw/latent":.1,"loss_raw/sigreg":.1,
                  "motion/std_batch":.05,"motion/cos_cross_chunk":.8}) for e in range(1,61)]
    (source/"train_metrics_epoch.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    for name in ("sig_loss_epoch1.csv", "gpu_samples.csv"): (source/name).write_text("value\n1\n")
    train_log = out/"train.log"
    train_log.write_text("Epoch 1/60: 10%|xx| [00:01]MOTION_MON std=1\rMOTION_MON std=2\n" +
                         "".join(f"Epoch {e}/60 total=.2\n" for e in range(1,61)) + "TRAIN_DONE\nEXIT_CODE=0\n")
    compare=out/"compare"
    for arm in ("new","old","ref"):
        for task in range(16):
            for abl in ("normal","zero","cross","neighbor"):
                file=compare/"videos"/arm/f"Task_with_underscore_{task}_ep199_{arm}_{abl}.mp4"
                file.parent.mkdir(parents=True,exist_ok=True); file.write_bytes("\x00CPU假视频".encode())
        dump(compare/arm/"result.json",dict(arm=arm, checkpoint=f"/scratch/hongze/MotionJEPA/runs/{RUN}/checkpoint_epoch_60.pt"))
    dump(compare/"summary.json",dict(status="PASS")); dump(compare/"windows.json", [])
    eval_log=out/"eval.log"
    eval_text="FOURARM=FAIL arm=new\nFOURARM_ROBUST=FAIL arm=new\nRGB_COMPARE=PASS\nRGB_ABLATION_VIDEOS=PASS arms=3 holdouts=16 ablations=4 videos=192\nEXIT_CODE=0\n"
    eval_log.write_text(eval_text)
    mid=out/"midprobe"
    line="MIDPROBE epoch=10 windows=16 enc_std=0.1 enc_cos=0.2 normal=0.1 zero_d=0 cross_d=0 nbr_d=0 mean_d=0 drift=0 tol=0 pos=0/0/0 robust=FAIL"
    for epoch,status in ((10,"ok"),(20,"fail"),(30,"skip"),(50,"running"),(60,"ok")):
        dump(mid/f"e{epoch}.status.json",dict(epoch=epoch,status=status,terminal=status!="running"))
        dump(mid/f"e{epoch}.json",dict(epoch=epoch,status="PASS",robust=False,midprobe_line=line if epoch==10 else ""))
    dump(mid/"e40.json",dict(epoch=40,status="PASS"))
    # 隔离仓库有自己生成的假凭据文件；绝不访问真实 secrets。
    repo=out/"isolated-repo"; dst=repo/"scripts/dataset/hf_export"; dst.mkdir(parents=True)
    for name in (FINAL,INC,"scan_hygiene.py","plan_upload_batches.py",
                 "hygiene_allowlist_mixed8_200_v9st_encoder_ckpt.json","mixed8_200_v9st_encoder_ckpt_bucket_README.md"):
        shutil.copy2(here/name,dst/name)
    (repo/"pyproject.toml").write_text('[project]\nname="v9st-export-offline-test"\nversion="0.0.0"\nrequires-python=">=3.11"\n')
    secrets=repo/"v1-store/secrets"; secrets.mkdir(parents=True)
    (secrets/"hf.env").write_text("HF_TOKEN=offline_test_fixture\n")
    fake=out/"fake-hf"; fake.mkdir(); dump(fake/"state.json",dict(exists=False,private=True))
    fake_bin=out/"bin"; fake_bin.mkdir()
    for name in ("uvx","curl","tmux","sleep"):
        cmd=fake_bin/name; cmd.write_text(f"#!{sys.executable}\n"+FAKE); cmd.chmod(0o755)
    env=os.environ.copy()
    for key in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN","SRC_OVERRIDE","DRYRUN_ALIVE"): env.pop(key,None)
    env.update(PATH=str(fake_bin)+os.pathsep+env["PATH"], FAKE_ROOT=str(fake),TEST_BUCKET=BUCKET,
               UV_OFFLINE="1",UV_PROJECT_ENVIRONMENT=str(mj/".venv"),UV_CACHE_DIR="/scratch/hongze/.cache/uv",
               M8X_SRC=str(source),M8X_TRAIN_LOG=str(train_log),M8X_MJ=str(mj),M8X_COMPARE=str(compare),
               M8X_MIDPROBE=str(mid),M8X_EVAL_LOG=str(eval_log))
    export=repo/"v1-store/exports"/EXPORT
    def final(label, *, changes=None, expected=0):
        use=env|dict(M8X_EXPORT_ROOT=str(export))|(changes or {})
        return run(["bash",str(dst/FINAL)],use,out/f"{label}.log",expected)
    def events():
        path=fake/"events.jsonl"
        return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    smoke=final("smoke_partial",changes=dict(SMOKE="1"))
    assert "LOCAL_FILES=235" in smoke and not events()
    stage=export/"stage"
    assert (stage/"train.log").read_text().count("MOTION_MON")==2
    assert not (stage/"eval-fourarm/reports/midprobe-e20.json").exists()
    assert "robust=FAIL" in (stage/"README.md").read_text()
    hygiene=json.loads((export/"logs/hygiene-stage.json").read_text())
    dump(out/"hygiene-evidence.json",hygiene)
    for amount in (0,5):
        other=out/f"midprobe-{amount}"; other.mkdir()
        if amount:
            for e in (10,20,30,40,50):
                dump(other/f"e{e}.status.json",dict(epoch=e,status="ok",terminal=True))
                dump(other/f"e{e}.json",dict(epoch=e,status="PASS",robust=False))
        text=final(f"smoke_k{amount}",changes=dict(SMOKE="1",M8X_MIDPROBE=str(other)))
        assert f"LOCAL_FILES={234+amount}" in text
    eval_log.write_text(eval_text.replace("RGB_COMPARE=PASS","RGB_COMPARE=FAIL"))
    assert "缺评估判定行" in final("reject_rgb",changes=dict(SMOKE="1"),expected=1)
    eval_log.write_text(eval_text)
    dry=out/"dry-source"; dry.mkdir()
    for e in (2,4,6): shutil.copy2(source/f"checkpoint_epoch_{e}.pt",dry/f"checkpoint_epoch_{e}.pt")
    torch.save(dict(epoch=6,arch="bad-arch",ema_only=True),dry/"checkpoint_epoch_6.pt")
    for e in (2,6): os.utime(dry/f"checkpoint_epoch_{e}.pt",(time.time()-700,)*2)
    text=run(["bash",str(dst/INC),"--dry-run",RUN,BUCKET,EXPORT,"fake-train"],
             env|dict(SRC_OVERRIDE=str(dry),DRYRUN_ALIVE="1"),out/"inc_dry.log")
    assert "INC_DRYRUN=PASS epoch=2 round=2" in text and "INC_CKPT_INVALID epoch=6" in text
    assert "INC_DRYRUN=PASS epoch=4" not in text and "INC_DRYRUN_END rounds=2" in text
    run(["bash",str(dst/INC),RUN,BUCKET,EXPORT,"fake-train"],env|dict(SRC_OVERRIDE=str(dry)),out/"inc_reject_override.log",1)
    assert not [e for e in events() if e["kind"] != "sleep"]
    # 正式增量只在测试副本替换固定输入根，所有算法/命令/凭据路径与原脚本完全一致。
    inc_source=out/"inc-source"; inc_source.mkdir(); shutil.copy2(source/"checkpoint_epoch_2.pt",inc_source/"checkpoint_epoch_2.pt")
    text=(dst/INC).read_text(); needle='SRC="/scratch/hongze/MotionJEPA/runs/$RUN_NAME"'
    assert text.count(needle)==1
    (dst/INC).write_text(text.replace(needle,f'SRC="{inc_source}"'))
    dump(fake/"state.json",dict(exists=True,private=True)); (fake/"bucket").mkdir()
    (fake/"bucket/foreign.bin").write_bytes(b"foreign")
    for label,argv in (("final_nonempty",["bash",str(dst/FINAL)]),("inc_nonempty",["bash",str(dst/INC),RUN,BUCKET,EXPORT,"fake-train"])):
        before=len(events())
        text=run(argv,env|dict(M8X_EXPORT_ROOT=str(export)),out/f"{label}.log",1)
        assert "无本轮上传标记" in text
        assert not any(e["kind"] in ("public","create","upload") for e in events()[before:])
    marker=export/"logs/uploaded.marker"
    marker_identity=dict(schema_version=1,run_name=RUN,bucket_id=BUCKET)
    for label,identity in (
        ("wrong_bucket",marker_identity|dict(bucket_id=BUCKET+"-other")),
        ("wrong_run",marker_identity|dict(run_name=RUN+"-other")),
        ("empty",None),
        ("malformed",{}),
    ):
        if identity is None: marker.write_text("")
        else: dump(marker,identity)
        for driver,argv in (("final",["bash",str(dst/FINAL)]),("inc",["bash",str(dst/INC),RUN,BUCKET,EXPORT,"fake-train"])):
            before=len(events())
            text=run(argv,env|dict(M8X_EXPORT_ROOT=str(export)),out/f"{driver}_marker_{label}.log",1)
            assert "UPLOAD_MARKER_MISMATCH" in text
            assert not any(e["kind"] in ("public","create","upload","upload_batch") for e in events()[before:])
    marker.unlink()
    (fake/"bucket/foreign.bin").unlink(); dump(fake/"state.json",dict(exists=False,private=True))
    text=run(["bash",str(dst/INC),RUN,BUCKET,EXPORT,"fake-train"],env,out/"inc_real_fake_hf.log",1)
    assert "FIRST_READBACK=PASS epoch=2" in text and "INC_END_INCOMPLETE" in text
    assert json.loads(marker.read_text())==marker_identity
    done=export/"logs/uploaded/ckpt-epoch-2.done"
    assert done.read_text().strip()==sha(source/"checkpoint_epoch_2.pt")
    # 已有 done 却丢失清单时，重启应从标记恢复完整清单。
    (export/"logs/inc-sha256.txt").unlink()
    run(["bash",str(dst/INC),RUN,BUCKET,EXPORT,"fake-train"],env,out/"inc_rebuild_manifest.log",1)
    assert "checkpoint_epoch_2.pt" in (export/"logs/inc-sha256.txt").read_text()
    before=len(events())
    text=final("full_fake_hf")
    assert "INC_SHA_MATCH epoch=2" in text and "SRC_UNCHANGED=OK" in text
    assert "RESULT=PASS" in text and "PUBLIC_ANON_READ=OK" in text
    later=events()[before:]
    assert next(i for i,e in enumerate(later) if e["kind"]=="list") < next(i for i,e in enumerate(later) if e["kind"]=="public")
    assert not any(e.get("name")=="checkpoint_epoch_2.pt" for e in later)
    assert len([p for p in (fake/"bucket").rglob("*") if p.is_file()])==235
    # 全部 done 已存在时也必须核身份，不能跳过 ensure_bucket 后误报 INC_END。
    dump(marker,marker_identity|dict(bucket_id=BUCKET+"-other"))
    before=len(events())
    text=run(["bash",str(dst/INC),RUN,BUCKET,EXPORT,"fake-train"],env,out/"inc_marker_all_done.log",1)
    assert "UPLOAD_MARKER_MISMATCH" in text and len(events())==before
    dump(marker,marker_identity)
    old_status=(mid/"e20.status.json").read_text()
    dump(mid/"e20.status.json",dict(epoch=20,status="ok",terminal=True))
    before=len(events())
    text=final("reject_changed_eval",expected=1)
    assert "EVAL_STAGE_CHANGED" in text and len(events())==before
    (mid/"e20.status.json").write_text(old_status)
    done.write_text("0"*64+"\n")
    text=final("reject_done_sha",expected=1); assert "INC_SHA_MISMATCH epoch=2" in text
    done.write_text(sha(source/"checkpoint_epoch_2.pt")+"\n")
    text=final("reject_corrupt_readback",changes=dict(FAKE_CORRUPT="1"),expected=1)
    assert "上传前后 sha256 不一致" in text and "RESULT=PASS" not in text
    # API 错误必须停止，不能当成不存在而创建或设公开。
    before=len(events())
    text=final("reject_api_failure",changes=dict(FAKE_API_FAIL="1"),expected=1)
    assert "无法查询 bucket" in text
    assert not any(e["kind"] in ("public","create") for e in events()[before:])
    for label,extra,passed in (
        ("retry_success",dict(FAKE_READBACK_FAILURES="3"),True),
        ("retry_exhausted",dict(FAKE_READBACK_FAILURES="4"),False),
        ("bad_first_sha",dict(FAKE_READBACK_BAD_SHA="1"),False),
    ):
        retry_export=repo/"v1-store/exports"/label
        (retry_export/"logs").mkdir(parents=True)
        dump(retry_export/"logs/uploaded.marker",marker_identity)
        state=json.loads((fake/"state.json").read_text()); state["readbacks"]=0; dump(fake/"state.json",state)
        text=run(["bash",str(dst/INC),RUN,BUCKET,label,"fake-train"],env|extra,out/f"inc_{label}.log",1)
        assert ("FIRST_READBACK=PASS" in text)==passed
        assert (retry_export/"logs/uploaded/ckpt-epoch-2.done").exists()==passed
        if label!="bad_first_sha": assert json.loads((fake/"state.json").read_text())["readbacks"]==4
    missing_marker_cases(out/"missing-marker")
    report=dict(status="PASS",synthetic_video=True,network=False,real_credentials=False,
                checks=["日志阶段1与9一致，保留两条MOTION_MON","k=0/1/5文件数234/235/239", "四臂FAIL允许，RGB FAIL拒绝",
                        "失败/缺失/未终态及e60不冒充成功probe", "增量两轮旧e2通过/新e4跳过/坏arch e6拒绝", "正式模式拒绝干跑覆盖",
                        "两个驱动非空无marker均不设public", "两个驱动均拒绝其他bucket的有效marker", "两个驱动均拒绝其他run的有效marker",
                        "两个驱动均拒绝空或无效marker", "全部done存在时仍核验marker身份", "e2匿名读回SHA与最终逐件done衔接", "末尾全部源映射含任务下划线",
                        "错误done SHA拒绝", "读回篡改拒绝且不提前RESULT=PASS", "API失败不创建或公开",
                        "增量中断后从done恢复SHA清单", "评估产物变化在联网前拒绝复用旧计划",
                        "匿名读取使用隔离HF_HOME", "首读回三失败后成功/四失败耗尽/坏SHA均按契约落标记",
                        "全部done且缺marker时启动拒绝", "部分done且缺marker时启动拒绝"])
    dump(out/"result.json",report)
    print(f"EXPORT_SELFTEST=PASS checks={len(report['checks'])} out={out}")


if __name__=="__main__":
    main()
