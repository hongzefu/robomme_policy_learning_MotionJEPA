"""从固定提交执行验证与4096→1024队列；任一阶段失败立即停止并保留现场。"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import os
from pathlib import Path
import shlex
import shutil
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"scripts/training"))
from modul_launch_contract import BUDGET_YAMLS, PROD_RUNS, CONFIG, MANIFEST_SHA, STORE_SHA, NORM_SHA, parse_config, validate_config, validate_data, resources, runtime_environment, json_sha, sha
from final_record import write_json

BETA_2048 = "55647ff33c8ddb9ec324fdbcee8bd1491456725b"
USER_QUOTE = "/scratch/hongze/robomme_policy_learning_MotionJEPA/0922-4096-1024-8gpu-training-plan.md 开始做 有问题立刻问用户 起泡后给出预计的时间"
RECORDER_DECISION = "可以优化取证方式 但要保证改前后都是用的一种取证方式 这个是用户最终决策"


def require(ok, message):
    if not ok:
        raise ValueError(message)


class Workflow:
    def __init__(self, batch, head, before_head, baseline_records=None):
        require(batch and all(c.isalnum() or c in "-_" for c in batch), "批次名非法")
        self.head, self.before_head, self.batch = head,before_head,batch
        self.baseline_records = baseline_records
        self.store = ROOT/"v1-store"
        self.root = self.store/"bench/modul-budget-sweep"/batch
        self.root.mkdir(parents=True,exist_ok=False)
        self.ds = self.store/"datasets/4task-v2-1600ep-604f16da"
        self.asset = self.store/"train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
        self.env = dict(os.environ)
        self.env.update(PYTHONUNBUFFERED="1",UV_CACHE_DIR=str(self.store/"cache/uv"),
            OPENPI_DATA_HOME=str(self.store/"models"),CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
            XLA_PYTHON_CLIENT_MEM_FRACTION="0.95",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",TZ="UTC",
            MMEVLA_FRAMESAMP_SOURCE=str(self.ds/"source"),MMEVLA_FRAMESAMP_MANIFEST=str(self.ds/"meta/episode_manifest.json"),
            HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",CUDA_CACHE_PATH=str(self.store/"cache/cuda"),
            WANDB_DATA_DIR=str(self.store/"cache/wandb-data"),XDG_DATA_HOME=str(self.store/"cache/xdg-data"))
        for key in ("PYTHONPATH","JAX_PLATFORMS","XLA_FLAGS","MMEVLA_MOTION_STORE","MMEVLA_FRAMESAMP_ALLOW_SUBSET"):
            self.env.pop(key,None)
        for key in list(self.env):
            if key.startswith("BENCH_") or key.startswith("TRAIN_"):
                self.env.pop(key,None)
        # 当前调度器只做CPU配置核验；数据来源环境与子进程显式一致。
        for key in ("MMEVLA_FRAMESAMP_SOURCE","MMEVLA_FRAMESAMP_MANIFEST","OPENPI_DATA_HOME"):
            os.environ[key] = self.env[key]
        self.clean()
        write_json(self.root/"start.json",{"head":head,"before_head":before_head,"pid":os.getpid(),
            "batch":batch,"user_quote":USER_QUOTE,"recorder_decision":RECORDER_DECISION,
            "baseline_records":str(baseline_records) if baseline_records else None,
            "started_at":self.now(),"environment":runtime_environment(ROOT),
            "tmux_session":os.environ.get("MODUL_TMUX_SESSION"),"environment_overrides":{k:self.env[k] for k in sorted(self.env) if k in (
                "CUDA_VISIBLE_DEVICES","XLA_PYTHON_CLIENT_MEM_FRACTION","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","TZ","OPENPI_DATA_HOME","MMEVLA_FRAMESAMP_SOURCE","MMEVLA_FRAMESAMP_MANIFEST","UV_CACHE_DIR")}})

    @staticmethod
    def now():
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def clean(self):
        require(subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()==self.head,"HEAD已变化")
        require(not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True),"工作区不干净")

    def event(self, **row):
        with (self.root/"events.jsonl").open("a") as stream:
            stream.write(json.dumps({"time":self.now(),**row},ensure_ascii=False)+"\n")

    def execute(self, name, command, changes=None, cwd=ROOT):
        self.clean()
        log = self.root/(name+".log")
        require(not log.exists(),"拒绝覆盖阶段日志")
        env = dict(self.env)
        env.update(changes or {})
        env["STAGE_LOG"] = str(log)
        self.event(stage=name,status="START",command=command,cwd=str(cwd),environment_overrides=changes or {})
        print(f"SWEEP_STAGE_START name={name} utc={self.now()}",flush=True)
        wrapper = 'set -o pipefail; "$@" 2>&1 | tee "$STAGE_LOG"; rc=$?; printf "EXIT_CODE=%s\\n" "$rc" | tee -a "$STAGE_LOG"; end=$?; if [ "$end" -ne 0 ]; then exit "$end"; fi; exit "$rc"'
        result = subprocess.run(["bash","-c",wrapper,"stage",*command],cwd=cwd,env=env)
        self.event(stage=name,status="PASS" if result.returncode==0 else "FAIL",exit_code=result.returncode)
        print(f"SWEEP_STAGE_END name={name} exit_code={result.returncode} utc={self.now()}",flush=True)
        require(result.returncode==0,f"阶段失败: {name}；停止队列，不自动重试")
        return log

    @staticmethod
    def python(script, *args):
        return ["uv","run","--project",str(ROOT),"--no-sync","python",str(ROOT/script),*map(str,args)]

    def train_args(self, budget, run, impl="packed"):
        return [CONFIG,"--exp-name",run,"--assets-base-dir",str(self.store/"train-assets"),
            "--data.assets.assets-dir",str(self.asset),"--data.assets.asset-id","robomme",
            "--checkpoint-base-dir",str(self.store/"train-runs"),"--dataset-path",str(self.ds/("source" if impl=="refnpy" else "framesamp-8x8")),
            "--model.history-config",BUDGET_YAMLS[budget]]

    def snapshot(self, name, head):
        path = self.root/name
        subprocess.run(["git","worktree","add","--detach",str(path),head],cwd=ROOT,check=True)
        require(not subprocess.check_output(["git","status","--porcelain"],cwd=path,text=True),"源码快照不干净")
        return path

    def baseline(self):
        old = self.snapshot("source-before",self.before_head)
        beta = self.snapshot("source-2048-beta",BETA_2048)
        log = self.execute("baseline-config",self.python("scripts/training/config_record.py",*self.train_args(2048,PROD_RUNS[2048])),
            {"PYTHONPATH":str(beta/"src"),"JAX_PLATFORMS":"cpu","CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1"},cwd=beta)
        lines = [line.removeprefix("COMPLETE_CONFIG_JSON=") for line in log.read_text().splitlines() if line.startswith("COMPLETE_CONFIG_JSON=")]
        require(len(lines)==1,"基线完整配置不唯一")
        self.baseline_path = self.root/"baseline-2048.json"
        write_json(self.baseline_path,{"source_head":BETA_2048,"complete":json.loads(lines[0])})
        original = self.bench_pair(2048,"before",source=old)
        after = self.bench_pair(2048,"after")
        recorded = self.bench_pair(2048,"after-final",final=True)
        self.compare("regression-2048",2048,original,after)
        self.compare("regression-final-record",2048,after,recorded)

    def reuse_baseline(self, source):
        """只复用完整对照组；代码、取证器、依赖、硬件、数据和精度均须仍可比。"""
        source = Path(source).resolve()
        require(source != self.root.resolve() and source.parent == self.root.parent.resolve(), "基线必须是本仓库其他批次")
        old = json.loads((source/"start.json").read_text())
        require(old["batch"] == source.name and old["before_head"] == self.before_head, "基线来源或改前版本不符")
        require(re.fullmatch(r"[0-9a-f]{40}", old["head"]) is not None, "基线提交锚点不完整")
        changed = subprocess.check_output(["git","diff","--name-only",old["head"],self.head],cwd=ROOT,text=True).splitlines()
        # 此清单只含验收/调度；取证、训练、配置和依赖发生任何变化都不能复用。
        allowed = {"scripts/training/tests/check_32frame_modul.py", "scripts/training/tests/check_modul_train_records.py",
            "scripts/training/tests/test_modul_train_records.py", "scripts/training/tests/test_modul_sampling_oracle.py",
            "scripts/training/tests/test_modul_sweep.py", "scripts/training/prod/modul_budget_workflow.py",
            "scripts/training/prod/run_modul4096_then1024.sh", "0922-4096-1024-8gpu-training-plan.md"}
        require(all(p in allowed or p.startswith("docs/training-doc/") for p in changed), "基线后有训练/取证/依赖或未列明代码变化，必须重跑")
        current = runtime_environment(ROOT)
        require(current == old["environment"], "基线环境指纹不符，必须重跑")
        data = {str(self.ds/"meta/episode_manifest.json"):MANIFEST_SHA,
                str(self.ds/"framesamp-8x8/meta/store_meta.json"):STORE_SHA,
                str(self.asset/"robomme/norm_stats.json"):NORM_SHA}
        require(all(sha(path) == expected for path,expected in data.items()), "基线数据摘要已变化")
        events = [json.loads(line) for line in (source/"events.jsonl").read_text().splitlines()]
        pairs = {label:tuple(source/f"{old['batch']}-m2048-{label}-{steps}" for steps in (20,1))
                 for label in ("before","after","after-final")}
        stages = ["baseline-config", "regression-2048", "regression-final-record"]
        stages += [path.name for pair in pairs.values() for path in pair]
        for stage in stages:
            outcomes = [r for r in events if r.get("stage") == stage and r.get("status") in ("PASS","FAIL")]
            require(len(outcomes) == 1 and outcomes[0]["status"] == "PASS" and outcomes[0]["exit_code"] == 0,
                    f"基线阶段没有唯一成功终态: {stage}")
            exits = [line for line in (source/(stage+".log")).read_text().splitlines() if line.startswith("EXIT_CODE=")]
            require(exits == ["EXIT_CODE=0"], f"基线阶段退出码不完整: {stage}")
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("reuse_record_checker",ROOT/"scripts/training/tests/check_modul_train_records.py")
        checker = module_from_spec(spec);spec.loader.exec_module(checker)
        for label,pair in pairs.items():
            for path in pair:
                meta = checker.load(path/"run_meta.json")
                expected_source = self.before_head if label == "before" else old["head"]
                require(meta["source_head"] == expected_source and meta["tool_head"] == old["head"], "基线记录版本与批次不符")
                current_meta = dict(meta,tool_head=self.head,checksum_workers=8,bench_checksum_enabled=True,
                    bench_batch_digests_enabled=True,bench_dump_idx_enabled=True,digest_interval_effective=1,
                    extra_digest_steps=[],state_dump_steps=[])
                require(checker.recorder_identity(meta) == checker.recorder_identity(current_meta), "当前取证方式与基线不相同")
        self.baseline_path = self.root/"baseline-2048.json"
        shutil.copyfile(source/"baseline-2048.json",self.baseline_path)
        # 新验收器重新读全部原记录，不能仅相信旧PASS标签。
        self.compare("regression-2048",2048,pairs["before"],pairs["after"])
        self.compare("regression-final-record",2048,pairs["after"],pairs["after-final"])
        files = [source/"start.json",source/"events.jsonl",source/"baseline-2048.json"]
        files += [path/name for pair in pairs.values() for path in pair for name in
                  ("run_meta.json","runtime.json","metrics.jsonl","param_checksums.jsonl","batch_digests.jsonl","index_sequence.json","idx_seq.jsonl")]
        write_json(self.root/"baseline-reuse.json",{"source_batch":old["batch"],"source_head":old["head"],
            "before_head":self.before_head,"current_head":self.head,"changed_paths":changed,"environment":current,
            "data_sha256":data,"record_sha256":{str(p):sha(p) for p in files},"recorder_decision":RECORDER_DECISION,
            "result":"PASS","meaning":"整组原始记录以相同取证器重验；没有混用不同测法，也没有重新执行训练"})
        print(f"BASELINE_REUSE=PASS batch={old['batch']} complete_groups=3 recorder_identical=1",flush=True)

    def bench(self, budget, label, steps, impl="packed", source=None, final=False, save=False):
        run = f"{self.batch}-m{budget}-{label}-{steps}"
        rec = self.root/run
        require(not rec.exists(),"取证目录已存在")
        env = {"BENCH_RECORD_DIR":str(rec),"TRAIN_RECORD_DIR":str(rec),"BENCH_DUMP_IDX":"1",
            "BENCH_DATASET_IMPL":impl,"BENCH_REF_SOURCE":str(self.ds/"source"),
            "BENCH_REF_MANIFEST":str(self.ds/"meta/episode_manifest.json"),"BENCH_CHECKSUM":"1",
            "BENCH_CHECKSUM_WORKERS":"8",
            "BENCH_BATCH_DIGESTS":"1","MMEVLA_JAX_CACHE_DIR":str(self.store/"cache/jax"/run),
            "WANDB_MODE":"disabled","XLA_FLAGS":"--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0",
            "BENCH_REF_COMMIT":self.before_head,"BENCH_CAND_COMMIT":self.head}
        if source:
            env.update(BENCH_SOURCE_ROOT=str(source),PYTHONPATH=str(source/"src"),PYTHONDONTWRITEBYTECODE="1")
        if final:
            env["TRAIN_FINAL_RECORD_DIR"] = str(rec/"final")
        if save:
            env.update(BENCH_SAVE_FINAL_CKPT="1",BENCH_FINAL_STEP=str(steps-1),
                       BENCH_STATE_DUMP_STEPS=f"0,{steps-1}",BENCH_STATE_DUMP_DIR=str(rec/"arrays"),
                       BENCH_DIGEST_INTERVAL=str(steps-1))
        self.execute(run,self.python("scripts/training/g0/bench_train_steps.py",*self.train_args(budget,run,impl),
            "--num-train-steps",steps,"--log-interval",1,"--save-interval",1,"--no-wandb-enabled"),env,cwd=source or ROOT)
        return rec

    def bench_pair(self,budget,label,impl="packed",source=None,final=False):
        return (self.bench(budget,label,20,impl,source,final),self.bench(budget,label,1,impl,source,final))

    def compare(self,name,budget,left,right):
        self.execute(name,self.python("scripts/training/tests/check_modul_train_records.py", "--records-a",left[0],
            "--step1-a",left[1],"--records-b",right[0],"--step1-b",right[1],"--batch-size",128,
            "--expected-devices","0,1,2,3,4,5,6,7","--expected-workers",16,"--expected-fsdp",8,
            "--budget",budget,"--negative-tests","--out",self.root/(name+".json")),{"JAX_PLATFORMS":"cpu","CUDA_VISIBLE_DEVICES":""})

    def production_runner(self,mode,budget,run,approval="",report="",handoff=""):
        return ["bash","scripts/training/prod/run_modul_budget.sh",mode,str(budget),run,self.head,
            sha(ROOT/"src/mme_vla_suite/models/config/robomme"/BUDGET_YAMLS[budget]),str(approval),str(report),str(self.baseline_path),str(handoff)]

    def run_records(self,run):
        return self.store/"bench/modul-budget-sweep/runs"/run

    def complete(self,budget,run,steps=80000,interval=100):
        rec = self.run_records(run)
        out = self.root/(run+".completed.json")
        self.execute(run+"-completion",self.python("scripts/training/tests/check_modul_completion.py",
            "--records",rec/"final","--metrics",rec/"metrics.jsonl","--log",self.store/"logs"/(run+".driver.log"),
            "--run-root",self.store/"train-runs"/CONFIG/run,"--run",run,"--head",self.head,
            "--budget",budget,"--steps",steps,"--log-interval",interval,"--out",out),
            {"MMEVLA_JAX_CACHE_DIR":str(self.store/"cache/jax"/(run+"-completion"))})
        resources(ROOT,"smoke")
        return out

    def verify_budget(self,budget):
        for command in ("yaml","frames","guards","assembly","pad","online","collate","oracle","init"):
            name = f"m{budget}-{command}"
            env = {"JAX_PLATFORMS":"cpu","CUDA_VISIBLE_DEVICES":""} if command!="init" else {}
            env["MMEVLA_JAX_CACHE_DIR"] = str(self.store/"cache/jax"/(self.batch+"-"+name))
            self.execute(name,self.python("scripts/training/tests/check_32frame_modul.py",command,"--budget",budget,
                "--out",self.root/(name+".json")),env)
        ref = self.bench_pair(budget,"refnpy",impl="refnpy")
        packed = self.bench_pair(budget,"packed")
        self.compare(f"m{budget}-refnpy-packed",budget,ref,packed)
        input_report = json.loads((self.root/f"m{budget}-assembly.json").read_text())["result"]
        actual_indices = json.loads((packed[0]/"index_sequence.json").read_text())["indices"][:2560]
        require(actual_indices == [i for row in input_report["batch_indices"] for i in row],"轻量对拍索引与实际前20个batch不同")
        hundred = self.bench(budget,"save",100,save=True)
        run = hundred.name
        checkpoint = self.store/"train-runs"/CONFIG/run/"999"
        self.execute(f"m{budget}-provenance",self.python("scripts/training/g0/check_config_provenance.py",
            "--ckpt",checkpoint,"--lib",self.ds,"--train-config",CONFIG,"--store-subdir","framesamp-8x8",
            "--norm-stats",self.asset/"robomme/norm_stats.json","--out",self.root/f"m{budget}-provenance.json"),
            {"JAX_PLATFORMS":"cpu","CUDA_VISIBLE_DEVICES":""})
        self.execute(f"m{budget}-reload",self.python("scripts/training/tests/check_32frame_modul.py","ckpt","--budget",budget,
            "--records",hundred,"--init-records",packed[0],"--state-dump-dir",hundred/"arrays","--ckpt",checkpoint,
            "--out",self.root/f"m{budget}-reload.json"),{"MMEVLA_JAX_CACHE_DIR":str(self.store/"cache/jax"/(run+"-reload"))})
        smoke = f"{self.batch}-m{budget}-capacity"
        self.execute(smoke,self.production_runner("smoke",budget,smoke))
        self.complete(budget,smoke,20,1)
        perf = f"{self.batch}-m{budget}-perf"
        self.execute(perf,self.production_runner("perf",budget,perf))
        report = self.root/f"m{budget}-speed.json"
        self.execute(perf+"-report",self.python("scripts/training/tests/check_modul_speed.py","report","--records",self.run_records(perf),
            "--gpu",str(self.run_records(perf))+".gpu.csv","--log",self.store/"logs"/(perf+".driver.log"),"--out",report))
        return report

    def approve(self,budget,report):
        from types import SimpleNamespace
        run = PROD_RUNS[budget]
        runner = ROOT/"scripts/training/prod/run_modul_budget.sh"
        args = SimpleNamespace(repo=str(ROOT),launch_mode="prod",expected_run_name=run,history_config=BUDGET_YAMLS[budget],
            run_root=str(self.store/"train-runs"/CONFIG/run),assets_dir=str(self.asset),asset_id="robomme",
            dataset_path=str(self.ds/"framesamp-8x8"),config_baseline=str(self.baseline_path),norm_stats_sha256=NORM_SHA)
        actual = validate_config(parse_config(self.train_args(budget,run),ROOT),args)
        data = validate_data(args)
        approval = self.root/f"m{budget}-approval.json"
        write_json(approval,{"train_head":self.head,"runner_sha256":sha(runner),"report_sha256":sha(report),
            "config_baseline_sha256":sha(self.baseline_path),
            "run_name":run,"config_sha256":json_sha(actual),"yaml_sha256":sha(ROOT/"src/mme_vla_suite/models/config/robomme"/BUDGET_YAMLS[budget]),
            "norm_stats_sha256":NORM_SHA,"manifest_file_sha256":data["manifest_file_sha256"],"store_meta_sha256":data["store_meta_sha256"],
            "run_root":args.run_root,"approved":True,"user_quote":USER_QUOTE,"approved_at":self.now(),
            "approval_time_kind":"本轮既有授权与实际报告完成绑定的时间，不代表用户再次发言",
            "scope":"执行0922计划；4096验收后1024；八卡b128/w16/FSDP8，各80000步，不自动改参"})
        return approval

    def archive_all(self):
        """两档进程均结束后才回写预建档案，避免影响中途clean HEAD检查。"""
        self.clean()
        resources(ROOT,"smoke")
        docs = ROOT/"docs/training-doc"
        changed = []
        def copy(source,destination):
            require(source.is_file(),f"归档源缺失: {source}")
            require(not destination.exists(),f"拒绝覆盖归档记录: {destination}")
            destination.parent.mkdir(parents=True,exist_ok=True)
            if source.suffix == ".log":
                lines = source.read_text().replace("\r","\n").splitlines()
                cleaned = [line.rstrip() for line in lines if not re.search(r"%\|",line)]
                require([l for l in lines if l.startswith("EXIT_CODE=")] ==
                        [l for l in cleaned if l.startswith("EXIT_CODE=")],"清洗丢失退出码")
                destination.write_text("\n".join(cleaned)+"\n")
            elif destination.suffix == ".gz":
                with source.open("rb") as src,gzip.open(destination,"xb") as dst:
                    shutil.copyfileobj(src,dst)
            else:
                shutil.copyfile(source,destination)
            changed.append(destination)
        events = [json.loads(line) for line in (self.root/"events.jsonl").read_text().splitlines()]
        starts = {row["stage"]:row for row in events if row.get("status")=="START"}
        ends = {row["stage"]:row for row in events if row.get("status")=="PASS"}
        require(starts.keys()==ends.keys(),"存在未完成阶段，禁止生成全部成功档案")
        group = docs/self.batch
        for name in ("start.json","events.jsonl","baseline-2048.json"):
            copy(self.root/name,group/"records"/name)
        if (self.root/"baseline-reuse.json").is_file():
            copy(self.root/"baseline-reuse.json",group/"records/baseline-reuse.json")
        for stage in starts:
            copy(self.root/(stage+".log"),group/"records"/(stage+".summary.log"))
            result = self.root/(stage+".json")
            if result.is_file():copy(result,group/"records"/result.name)
        names = [name for name in starts if name.startswith(self.batch+"-m") and name.endswith(("-20","-1","-100","-capacity","-perf"))]
        names += [PROD_RUNS[4096],PROD_RUNS[1024]]
        for name in names:
            target = docs/name
            require((target/"launch.md").is_file(),f"缺预建档案: {name}")
            record_root = self.run_records(name) if name in PROD_RUNS.values() or name.endswith(("-capacity","-perf")) else self.root/name
            for filename in ("run_meta.json","runtime.json","metrics.jsonl","param_checksums.jsonl","batch_digests.jsonl",
                             "index_sequence.json","idx_seq.jsonl","final_checkpoint.json","speed_start.json","speed_run.json",
                             "host_samples.jsonl","step_timing.jsonl"):
                source = record_root/filename
                if source.is_file():copy(source,target/"records"/filename)
            if (record_root/"final").is_dir():
                for filename in ("start.json","final.json","checkpoint_wait_done.json"):
                    copy(record_root/"final"/filename,target/"records/final"/filename)
            gpu = Path(str(record_root)+".gpu.csv")
            if gpu.is_file():copy(gpu,target/"records/gpu.csv.gz")
            driver = self.store/"logs"/(name+".driver.log")
            copy(driver if driver.is_file() else self.root/(name+".log"),target/"records/run.summary.log")
            complete = self.root/(name+".completed.json")
            if complete.is_file():copy(complete,target/"records/completed.json")
            actual = {"head":self.head,"start":starts[name],"end":ends[name],"user_quote":USER_QUOTE,
                      "recorder_decision":RECORDER_DECISION}
            write_json(target/"records/launch.actual.json",actual);changed.append(target/"records/launch.actual.json")
            meta = json.loads((record_root/"run_meta.json").read_text())
            source_head = meta.get("source_head",self.head)
            text = f"# {name} 实际结果\n\n本阶段正常完成，退出码0。起跑/取证器提交为 `{self.head}`，训练源码为 `{source_head}`；启动UTC为{starts[name]['time']}，结束UTC为{ends[name]['time']}。\n\n"
            text += f"用户原话：「{USER_QUOTE}」。配置、命令、全部阶段覆盖与硬件见[实际记录](records/launch.actual.json)及[总档案](../{self.batch}/launch.md)。\n\n"
            text += f"取证最终决定：「{RECORDER_DECISION}」。主跑、补跑及对照两侧共同核对取证源码与设置。\n\n"
            text += "本轮使用AWS八卡A100、全局batch128、数据worker16、FSDP8。状态取证worker8仅作用于验证记录器，不是数据worker或训练并行度。权重及原数组保留在v1-store，不复制到Git。\n\n"
            metric_rows = [json.loads(line) for line in (record_root/"metrics.jsonl").read_text().splitlines()]
            text += f"本阶段实际记录{len(metric_rows)}条指标，记录步范围{metric_rows[0]['step']}至{metric_rows[-1]['step']}。日志与指标见records。本轮两侧主跑和补跑共同满足20+1步与201叶逐位判据；100步保存、容量与测速分别按总档案约定验收。训练loss用于过程检查，不作为策略成功率结论。\n"
            if name in PROD_RUNS.values():
                result = json.loads(complete.read_text())
                text += f"\n正式80000步完成，{len(result['checkpoints'])}份checkpoint齐全；最终79999原dtype权重与现场EMA摘要一致、末99步及固定noise的10步动作有限。末条常规日志为79900，不代表79999单步loss。GPU原始500ms采样无损压缩为gpu.csv.gz。\n"
            (target/"result.md").write_text(text);changed.append(target/"result.md")
            with (target/"launch.md").open("a") as stream:
                stream.write(f"\n## 实际起跑与完成\n\n已从 `{self.head}` 正常完成；实际命令、UTC、环境和退出码见[launch.actual.json](records/launch.actual.json)，结论见[result.md](result.md)。上文未启动标记为起跑前记录。\n")
            changed.append(target/"launch.md")
        for budget in (4096,1024):
            target = docs/f"{self.batch}-m{budget}-input"
            commands = ("yaml","frames","guards","assembly","pad","online","collate","oracle","init")
            for command in commands:
                name = f"m{budget}-{command}"
                copy(self.root/(name+".json"),target/"records"/(command+".json"))
            (target/"result.md").write_text(f"# {budget}输入及语义验收\n\n所有九项验收通过；提交 `{self.head}`。有界真实样本、索引、字节、padding、在线缓存、RoPE和初始化的实测记录见records；没有宣称全库特征重新扫描。用户原话及全部命令见[总档案](../{self.batch}/launch.md)。\n")
            changed.append(target/"result.md")
            for name in (f"m{budget}-speed.json",f"m{budget}-approval.json",f"m{budget}-provenance.json",f"m{budget}-reload.json",f"m{budget}-refnpy-packed.json"):
                destination = group/"records"/name
                if not destination.exists():copy(self.root/name,destination)
        (group/"result.md").write_text(f"# 顺序训练完成\n\n4096与1024均完成80000步和最终验收，先4096后1024，全部阶段退出0。起跑锚点 `{self.head}`；旧源码对照 `{self.before_head}`。\n\n用户原话：「{USER_QUOTE}」。全部展开命令、环境、UTC及阶段退出见records/events.jsonl，两档独立测速与初始化/稳态/保存的ETA见records/m4096-speed.json、records/m1024-speed.json。正式结果在两个run子档案。\n\n取证优化保持原dtype/C序字节/SHA、全部201叶及20+1步判据；没有减少验证或改变正式超参。原a批次的中断和615.913秒初态摘要保留。策略rollout与外部权重导出未执行。\n")
        with (group/"result.md").open("a") as stream:
            stream.write(f"\n用户取证最终决定：「{RECORDER_DECISION}」。同组取证方式的检查记录与完整比较结果一同归档。\n")
        changed.append(group/"result.md")
        index = docs/"README.md"
        rows=[]
        for line in index.read_text().splitlines():
            if line.startswith("| ") and any(f"`{name}/`" in line for name in [self.batch,*names,f"{self.batch}-m4096-input",f"{self.batch}-m1024-input"]):
                cells=line.split("|");cells[-2]=" 已完成，通过对应验收 ";line="|".join(cells)
            rows.append(line)
        index.write_text("\n".join(rows)+"\n");changed.append(index)
        relative = sorted({str(path.relative_to(ROOT)) for path in changed})
        subprocess.run(["git","diff","--check"],cwd=ROOT,check=True)
        logs = [path for path in relative if path.endswith(".summary.log")]
        subprocess.run(["git","add","--",*[path for path in relative if path not in logs]],cwd=ROOT,check=True)
        if logs:
            # 仓库全局忽略*.log；只对本函数明确生成的清洗日志逐路径放行。
            subprocess.run(["git","add","-f","--",*logs],cwd=ROOT,check=True)
        staged = set(subprocess.check_output(["git","diff","--cached","--name-only"],cwd=ROOT,text=True).splitlines())
        require(staged == set(relative),"暂存范围含非本轮文件")
        subprocess.run(["git","diff","--cached","--check"],cwd=ROOT,check=True)
        subject = subprocess.check_output(["git","show","-s","--format=%s",self.head],cwd=ROOT,text=True).strip()
        match = re.match(r"commitV(\d+\.\d+)Beta:",subject)
        require(match is not None,"起跑提交不是Beta锚点")
        body = f"commitV{match.group(1)}: 完成4096与1024八卡80k顺序训练归档\n\n用户原话：{USER_QUOTE}\n\n按0922计划，完成2048改前/改后20+1步及最终记录开关回归，两个新预算输入与NPY/packed完整状态对拍、100步保存加载、20步容量、1000步独立测速均通过；两档各80000步、16份checkpoint、尾窗与最终动作验收通过。所有阶段命令、覆盖、SHA、实测日志和指标见本批次档案。仅回写本轮文档与运行记录，未改训练源码、未复制权重或配置脚本。初始慢取证中断与有界并行优化已留档；本次沿用全部逐位判据。当前训练及归档完成，策略rollout和外部导出未开展。\n"
        body += f"\n取证最终用户决定：{RECORDER_DECISION}\n各组主跑、补跑与对照两侧使用相同取证器源码和设置；不同方法的记录不拼接为等价结论。\n"
        message=self.root/"completion-commit.txt";message.write_text(body)
        subprocess.run(["git","commit","-F",str(message)],cwd=ROOT,check=True)
        subprocess.run(["git","push"],cwd=ROOT,check=True)
        print("SWEEP_ARCHIVE=PASS commit="+subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),flush=True)

    def run(self):
        resources(ROOT,"smoke")
        if getattr(self,"baseline_records",None):
            self.reuse_baseline(self.baseline_records)
        else:
            self.baseline()
        handoff = ""
        for budget in (4096,1024):
            if budget == 1024:
                self.execute("handoff-4096",self.python("scripts/training/tests/check_modul_completion.py","--verify-handoff",handoff,"--head",self.head),
                    {"JAX_PLATFORMS":"cpu","CUDA_VISIBLE_DEVICES":""})
            resources(ROOT,"smoke")
            report = self.verify_budget(budget)
            approval = self.approve(budget,report)
            self.execute(PROD_RUNS[budget],self.production_runner("prod",budget,PROD_RUNS[budget],approval,report,handoff))
            handoff = self.complete(budget,PROD_RUNS[budget])
        self.archive_all()
        self.event(status="ALL_COMPLETED")
        print("SWEEP_COMPLETED=PASS budgets=4096,1024",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch",required=True)
    parser.add_argument("--head",required=True)
    parser.add_argument("--before-head",required=True)
    parser.add_argument("--baseline-records",type=Path,help="完整2048对照组；仅环境、代码和取证器均匹配时复用")
    args = parser.parse_args()
    Workflow(args.batch,args.head,args.before_head,args.baseline_records).run()


if __name__ == "__main__":
    main()
