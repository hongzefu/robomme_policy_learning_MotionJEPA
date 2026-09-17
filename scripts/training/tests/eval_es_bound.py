"""经真实 EnvRunner.reset 采集 test 集 demo 长度；任务分进程，聚合后才判预算。"""

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import sys
import subprocess
import time

ROOT=pathlib.Path(__file__).resolve().parents[3]
TASKS=("BinFill","RouteStick","VideoRepick","VideoUnmaskSwap")


def predicted_windows(es, max_steps=1300):
    demo=len(range(0,max(0,int(es)-16),16))
    delta=16*(int(max_steps)//16)
    execution=0 if delta<32 else (delta-32)//16+1
    return demo+execution


def reset_task(args):
    sys.path.insert(0,str(ROOT/"examples/robomme"))
    from env_runner import EnvRunner
    import robomme
    dependency=ROOT/"third_party/robomme_benchmark"
    if not pathlib.Path(robomme.__file__).resolve().is_relative_to(dependency/"src"):
        raise ValueError("仿真环境没有导入本仓库钉版的 RoboMME")
    dependency_head=subprocess.check_output(["git","-C",str(dependency),"rev-parse","HEAD"],text=True).strip()
    if subprocess.check_output(["git","-C",str(dependency),"status","--porcelain"],text=True):
        raise ValueError("RoboMME 依赖工作区不干净")
    if args.out.exists(): raise FileExistsError(f"拒绝覆盖 reset 报告: {args.out}")
    runner=EnvRunner(args.task,str(args.out.parent),max_steps=args.max_steps)
    metadata_path=pathlib.Path(runner.env_builder._resolve_metadata_path())
    metadata_sha=hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    count=runner.num_episodes
    if count!=50: raise ValueError(f"{args.task} 的 test 集数 {count} != 50")
    indices=sorted(ep for task,ep in runner.env_builder.metadata_index if task==args.task)
    if indices!=list(range(count)): raise ValueError("test episode 编号与真实 eval 的 range 口径不同")
    selected=indices[:args.limit or None]
    rows=[]
    for ep in selected:
        start=time.perf_counter()
        try:
            runner.make_env(ep)
            observation=runner.get_init_obs()
            n=len(observation["images"])
            if n<1 or len(observation["wrist_images"])!=n or len(observation["states"])!=n:
                raise ValueError("reset 返回的前轨迹图像与状态长度不一致")
            es=n-1
            rows.append({"episode":ep,"es":es,"k_eval":predicted_windows(es,args.max_steps),
                         "difficulty":str(runner.difficulty),"goal":runner.task_goal,
                         "render_device":str(getattr(runner.env.unwrapped,"_render_device","unknown")),
                         "seconds":time.perf_counter()-start})
            del observation
            print(f"EVAL_RESET task={args.task} episode={ep} es={es} k={rows[-1]['k_eval']}",flush=True)
        finally:
            runner.close_env()
    if hashlib.sha256(metadata_path.read_bytes()).hexdigest()!=metadata_sha:
        raise ValueError("test metadata 在 reset 期间变化")
    versions={}
    for package in ("sapien","mani_skill","robomme","numpy","torch"):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]="unknown"
    result={"task":args.task,"complete":len(rows)==count,"episodes":rows,"budget":args.budget,
            "max_steps":args.max_steps,"metadata_path":str(metadata_path),"metadata_sha256":metadata_sha,
            "python":sys.executable,"packages":versions,
            "robomme_commit":dependency_head,
            "env_runner_sha256":hashlib.sha256((ROOT/"examples/robomme/env_runner.py").read_bytes()).hexdigest()}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x") as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(f"EVAL_RESET_DONE task={args.task} episodes={len(rows)} complete={int(result['complete'])}",flush=True)


def aggregate(args):
    if args.out.exists(): raise FileExistsError(f"拒绝覆盖预算报告: {args.out}")
    reports=[json.loads(p.read_text()) for p in args.reports]
    if len(reports)!=4 or {r["task"] for r in reports}!=set(TASKS):
        raise ValueError("评估预检必须覆盖全部四任务，且没有重复报告")
    if any(r["complete"] is not True or len(r["episodes"])!=50 for r in reports):
        raise ValueError("评估预检没有完成全部 200 次 reset")
    if any([row["episode"] for row in r["episodes"]] != list(range(50)) for r in reports):
        raise ValueError("评估预检的 episode 集合缺失、重复或乱序")
    for key in ("python","packages","env_runner_sha256","robomme_commit"):
        if len({json.dumps(r[key],sort_keys=True) for r in reports}) != 1:
            raise ValueError(f"四任务 {key} 不一致")
    if any(r["budget"]!=args.budget or r["max_steps"]!=args.max_steps for r in reports):
        raise ValueError("各任务使用的预算或评估步数不一致")
    rows=[row for r in reports for row in r["episodes"]]
    maximum=max(row["es"] for row in rows)
    max_windows=max(predicted_windows(row["es"],args.max_steps) for row in rows)
    headroom=args.budget-max_windows
    result={"tasks":4,"episodes":200,"es_max":maximum,"k_eval_max":max_windows,
            "budget":args.budget,"max_steps":args.max_steps,"headroom":headroom,
            "passed":headroom>=0,"reports":[str(p) for p in args.reports]}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x") as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(f"EVAL_ES_BOUND={'PASS' if headroom>=0 else 'FAIL'} tasks=4 episodes=200 es_max={maximum} budget={args.budget} headroom={headroom}",flush=True)
    if headroom<0: raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    for name in ("reset","aggregate"):
        p=commands.add_parser(name)
        p.add_argument("--out",type=pathlib.Path,required=True)
        p.add_argument("--budget",type=int,default=160)
        p.add_argument("--max-steps",type=int,default=1300)
        if name=="reset":
            p.add_argument("--task",choices=TASKS,required=True)
            p.add_argument("--limit",type=int,default=0,help="仅供短测；部分结果不能通过 aggregate")
        else: p.add_argument("--reports",type=pathlib.Path,nargs="+",required=True)
    args=parser.parse_args()
    if args.budget <= 0 or args.max_steps <= 0 or (args.command=="reset" and args.limit<0):
        raise ValueError("预算、步数或短测数量非法")
    reset_task(args) if args.command=="reset" else aggregate(args)


if __name__ == "__main__":
    main()
