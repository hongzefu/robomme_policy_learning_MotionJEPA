"""记忆预算启动契约：完整CPU配置、资源、测速与本次授权共同约束入口。"""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

YAML="perceptual-framesamp-modul-32frame-8x8.yaml"
PROD_RUN="v2-1600ep-m32x8x8-modul-b128-80k"
CONFIG="mme_vla_suite_b128_80k"
MANIFEST_SHA="df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482"
NORM_SHA="856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173"
STORE_SHA="f7677e69e5c473ab2962a5ac05a5909348c2f96736152b7217b77f0d2eb4231a"
BUDGET_YAMLS = {budget: f"perceptual-framesamp-modul-{budget//64}frame-8x8.yaml"
                for budget in (1024, 2048, 4096)}
PROD_RUNS = {budget: f"v2-1600ep-m{budget//64}x8x8-modul-b128-80k" for budget in BUDGET_YAMLS}


def runtime_environment(repo):
    """只读记录当前依赖、设备、存储及计算环境，不读取任何凭据。"""
    return {"packages":{name:importlib.metadata.version(name) for name in ("torch","jax","jaxlib","numpy","ml_dtypes","flax","optax")},
        "uv_lock_sha256":sha(Path(repo)/"uv.lock"),"sys_prefix":sys.prefix,
        "storage":subprocess.check_output(["findmnt","-T",str(repo),"-no","SOURCE,FSTYPE,TARGET"],text=True).strip(),
        "gpus":subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"],text=True).splitlines(),
        "environment":{key:os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES","XLA_FLAGS","XLA_PYTHON_CLIENT_MEM_FRACTION",
            "JAX_ENABLE_X64","JAX_DEFAULT_MATMUL_PRECISION","XLA_PYTHON_CLIENT_PREALLOCATE","MMEVLA_JAX_CACHE_DIR","CUDA_CACHE_PATH")}}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda:f.read(1<<20),b""):h.update(data)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()


def validate_argv(argv,mode):
    require(argv and argv[0] == CONFIG,"具名配置必须为mme_vla_suite_b128_80k")
    path_flags={"--exp-name","--assets-base-dir","--data.assets.assets-dir","--data.assets.asset-id",
                "--checkpoint-base-dir","--dataset-path","--model.history-config","--weight-loader.params-path"}
    allowed=path_flags|{"--model.use-history"}
    booleans={"--model.use-history","--no-wandb-enabled"}
    if mode in ("smoke","perf"):allowed|={"--num-train-steps","--no-wandb-enabled"}
    if mode == "smoke":allowed|={"--log-interval"}
    values={}
    i=1
    while i<len(argv):
        raw=argv[i];key=raw.split("=",1)[0]
        require(key in allowed,f"{mode}禁止该覆盖或多余位置参数: {key}")
        require(key not in values,f"重复参数: {key}")
        if "=" in raw:
            require(key not in booleans,"布尔旗标不接受含糊等号值")
            value=raw.split("=",1)[1]
        elif key in booleans:value=True
        else:
            i+=1
            require(i<len(argv) and not argv[i].startswith("--"),f"参数缺值: {key}")
            value=argv[i]
        values[key]=value;i+=1
    if mode in ("smoke","perf"):
        require(values.get("--num-train-steps") == ("20" if mode=="smoke" else "1000"),"smoke/perf步数不符")
        require(values.get("--no-wandb-enabled") is True,"smoke/perf必须关闭W&B")
    if "--log-interval" in values:require(values["--log-interval"] == "1","smoke逐步日志必须为1")
    return values


PARSE_SCRIPT = '''
import dataclasses,json,sys
sys.path.insert(0, __RECORD_MODULE_DIR__)
from config_record import complete_record
from mme_vla_suite.training.config import cli,get_config
from mme_vla_suite.models.config.utils import get_history_config
from omegaconf import OmegaConf
def serialize(c):
    return {"name":c.name,"exp_name":c.exp_name,"batch_size":c.batch_size,
        "num_train_steps":c.num_train_steps,"num_workers":c.num_workers,"fsdp_devices":c.fsdp_devices,
        "seed":c.seed,"lr_schedule":dataclasses.asdict(c.lr_schedule),
        "optimizer":dataclasses.asdict(c.optimizer),"ema_decay":c.ema_decay,
        "model":dataclasses.asdict(c.model),"freeze_filter":repr(c.freeze_filter),
        "pytorch_training_precision":c.pytorch_training_precision,
        "weight_loader_path":str(c.weight_loader.params_path),
        "checkpoint_dir":str(c.checkpoint_dir),"assets_base_dir":str(c.assets_base_dir),
        "assets_dir":str(c.data.assets.assets_dir),"asset_id":c.data.assets.asset_id,
        "dataset_path":str(c.dataset_path),"log_interval":c.log_interval,
        "save_interval":c.save_interval,"keep_period":c.keep_period,
        "wandb_enabled":c.wandb_enabled,"overwrite":c.overwrite,"resume":c.resume,
        "history_values":OmegaConf.to_container(get_history_config(c.model.history_config)) if c.model.history_config else None,
        "complete":complete_record(dataclasses.replace(c,model=dataclasses.replace(c.model,history_config=actual.model.history_config)))}
actual=cli()
default=get_config("mme_vla_suite_b128_80k")
default=dataclasses.replace(default,exp_name=actual.exp_name)
history_expected=OmegaConf.to_container(get_history_config("perceptual-framesamp-modul-8frame-8x8.yaml"))
history_expected["budget"]=OmegaConf.to_container(get_history_config(actual.model.history_config))["budget"]
print("LAUNCH_PARSE_JSON="+json.dumps({"actual":serialize(actual),"default":serialize(default),"history_expected":history_expected},sort_keys=True))
'''.replace("__RECORD_MODULE_DIR__", repr(str(Path(__file__).resolve().parent)))


def parse_config(argv,repo):
    env=dict(os.environ,JAX_PLATFORMS="cpu",CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    for key,suffix in (("UV_CACHE_DIR","uv"),("XDG_CACHE_HOME","xdg"),("HF_HOME","hf"),("MMEVLA_JAX_CACHE_DIR","jax")):
        env[key]=str(Path(repo)/"v1-store/cache"/suffix)
    env.setdefault("OPENPI_DATA_HOME", str(Path(repo)/"v1-store/models"))
    result=subprocess.run([sys.executable,"-c",PARSE_SCRIPT,*argv],cwd=repo,env=env,text=True,capture_output=True)
    require(result.returncode == 0,f"真实CLI解析失败: {result.stdout}\n{result.stderr}")
    lines=[x.removeprefix("LAUNCH_PARSE_JSON=") for x in result.stdout.splitlines() if x.startswith("LAUNCH_PARSE_JSON=")]
    require(len(lines)==1,"真实配置解析结果不唯一")
    return json.loads(lines[0])


def validate_config(parsed,a):
    actual,default=parsed["actual"],parsed["default"]
    target=80000 if a.launch_mode=="prod" else (20 if a.launch_mode=="smoke" else 1000)
    require(actual["name"]==CONFIG and actual["exp_name"]==a.expected_run_name,"实际配置名或run名不符")
    require((actual["batch_size"],actual["num_train_steps"],actual["num_workers"],actual["fsdp_devices"],actual["seed"]) ==
            (128,target,16,8,42),"生产形制b128/w16/fsdp8/seed42或步数不符")
    for key in ("lr_schedule","optimizer","ema_decay","freeze_filter","pytorch_training_precision","save_interval","keep_period"):
        require(actual[key]==default[key],f"实际配置偏离具名默认: {key}")
    require(actual["lr_schedule"] == {"warmup_steps":5000,"peak_lr":5e-5,"decay_steps":80000,"decay_lr":5e-5},"实际学习率口径不符")
    require(actual["ema_decay"]==.999 and actual["optimizer"]["clip_gradient_norm"]==1.,"EMA或裁剪不符")
    history=actual["history_values"]
    budget = history["budget"]
    require(type(budget) is int and budget in BUDGET_YAMLS, "实际记忆预算不受支持")
    expected_model={**default["model"],"use_history":True,"history_config":BUDGET_YAMLS[budget]}
    require(actual["model"]==expected_model and a.history_config==BUDGET_YAMLS[budget],"实际模型、精度或history配置不同")
    require(history==parsed["history_expected"] and all(type(history[k]) is int for k in ("budget","token_per_image","num_views")),"实际YAML内容或类型不是已验证预算档")
    require(not actual["overwrite"] and not actual["resume"],"禁止覆盖或续训")
    if a.launch_mode=="prod":
        require(actual["wandb_enabled"]==default["wandb_enabled"] and actual["log_interval"]==default["log_interval"],"正式日志或W&B偏离默认")
        require(a.expected_run_name==PROD_RUNS[budget],"正式run名不是本预算已定名称")
    else:
        require(a.expected_run_name not in PROD_RUNS.values() and not actual["wandb_enabled"],"验证运行使用正式名或开启W&B")
        if a.launch_mode=="perf":require(actual["log_interval"]==100,"测速必须保持log100")
    repo=Path(a.repo).resolve();v1=repo/"v1-store"
    ds=v1/"datasets/4task-v2-1600ep-604f16da"
    expected_run=v1/"train-runs"/CONFIG/a.expected_run_name
    require(Path(actual["checkpoint_dir"]).resolve()==Path(a.run_root).resolve()==expected_run,"实际输出根错配")
    require(Path(actual["weight_loader_path"]).resolve()==v1/"models/openpi-assets/checkpoints/pi05_base/params","初始化权重路径不符")
    require(Path(actual["assets_base_dir"]).resolve()==v1/"train-assets","资产根不符")
    require(Path(actual["assets_dir"]).resolve()==Path(a.assets_dir).resolve()==v1/"train-assets/mme_vla_suite/4task-v2-1600ep-604f16da","归一化资产路径不符")
    require(actual["asset_id"]==a.asset_id=="robomme","资产id不符")
    require(Path(actual["dataset_path"]).resolve()==Path(a.dataset_path).resolve()==ds/"framesamp-8x8","实际数据路径不符")
    from config_record import compare_records, BUDGET_FIELDS
    # 从当前具名默认完整比对，只有已单独校验的启动路径和运行身份可以变化。
    compare_records(default["complete"], actual["complete"], (*BUDGET_FIELDS,
        "train_config.fields.assets_base_dir", "train_config.fields.checkpoint_base_dir",
        "train_config.fields.dataset_path", "train_config.fields.data.fields.assets.fields.assets_dir",
        "derived.assets_dirs"))
    baseline_path = getattr(a, "config_baseline", None)
    if budget != 2048:
        require(baseline_path, "新预算必须绑定2048 Beta补建的完整配置基线")
    if baseline_path:
        baseline = json.loads(Path(baseline_path).read_text())
        require(baseline.get("source_head") == "55647ff33c8ddb9ec324fdbcee8bd1491456725b", "完整配置基线不是2048 Beta")
        compare_records(baseline["complete"], actual["complete"], BUDGET_FIELDS)
    return actual


def validate_data(a):
    ds=Path(a.dataset_path)
    parent=ds.parent
    require(not ds.is_symlink() and ds.is_dir(),"输入库必须为实体目录")
    require(Path(os.environ.get("MMEVLA_FRAMESAMP_SOURCE","")).resolve()==parent/"source","source与输入库不同源")
    manifest=Path(os.environ.get("MMEVLA_FRAMESAMP_MANIFEST",""))
    require(manifest.resolve()==parent/"meta/episode_manifest.json" and sha(manifest)==MANIFEST_SHA,"数据清单路径或文件摘要不同")
    document=json.loads(manifest.read_text())
    canonical=json_sha({k:v for k,v in document.items() if k!="sha256"})
    require(canonical==document["sha256"],"清单规范化指纹不符")
    meta=json.loads((ds/"meta/store_meta.json").read_text())
    require(sha(ds/"meta/store_meta.json") == STORE_SHA,"packed元数据不再与2048基线同源")
    require((meta["layout"],meta["status"],meta["manifest_scope"],meta["num_rows"],meta["num_exec_samples"]) ==
            ("framesamp-8x8-v1","verified","full",1192918,605611),"数据布局、验证状态或覆盖规模不符")
    require(meta["manifest_sha256"]==canonical,"packed与清单指纹不符")
    require(a.norm_stats_sha256==NORM_SHA,"norm_stats期望不是已定资产")
    return {"manifest_file_sha256":MANIFEST_SHA,"manifest_canonical_sha256":canonical,"store_meta_sha256":sha(ds/"meta/store_meta.json")}


def validate_approval(a,actual,data):
    require(a.approval_record and a.runner and a.report,"正式模式缺同意记录、runner或报告路径")
    approval=json.loads(Path(a.approval_record).read_text())
    expected={"train_head":a.train_head,"runner_sha256":sha(a.runner),"report_sha256":sha(a.report),
              "run_name":a.expected_run_name,"config_sha256":json_sha(actual),"yaml_sha256":a.history_config_sha256,
              "norm_stats_sha256":a.norm_stats_sha256,"manifest_file_sha256":data["manifest_file_sha256"],
              "store_meta_sha256":data["store_meta_sha256"],"run_root":str(Path(a.run_root).resolve())}
    if getattr(a,"config_baseline",None):
        expected["config_baseline_sha256"] = sha(a.config_baseline)
    require(all(approval.get(k)==v for k,v in expected.items()),"同意记录未绑定本次代码、配置、报告、runner或数据")
    require(approval.get("user_quote","").strip() and approval.get("approved") is True,"缺用户明确同意原话")
    approved_at=datetime.datetime.fromisoformat(approval["approved_at"])
    require(approved_at.tzinfo is not None and approved_at.timestamp()<=time_now(),"同意记录时间无效")
    report=json.loads(Path(a.report).read_text())
    require(report.get("status")=="READY","测速报告尚不完整")
    from config_record import compare_records, PERF_FIELDS
    perf_actual = report["launch"]["actual"]
    require(perf_actual["num_train_steps"] == 1000 and not perf_actual["wandb_enabled"], "报告不是独立1000步测速")
    require(perf_actual["history_values"] == actual["history_values"], "测速报告属于另一记忆预算")
    require(report["launch"]["data"]["yaml_sha256"] == a.history_config_sha256, "测速YAML摘要与正式配置不同")
    compare_records(perf_actual["complete"], actual["complete"], PERF_FIELDS)
    current=runtime_environment(a.repo)
    environment=report.get("environment",{})
    for key in ("packages","uv_lock_sha256","sys_prefix","storage","gpus"):
        require(current[key]==environment.get(key),f"当前环境与测速不同: {key}")
    for key in ("CUDA_VISIBLE_DEVICES","XLA_FLAGS","XLA_PYTHON_CLIENT_MEM_FRACTION","JAX_ENABLE_X64","JAX_DEFAULT_MATMUL_PRECISION","XLA_PYTHON_CLIENT_PREALLOCATE"):
        require(current["environment"][key]==environment.get("environment",{}).get(key),f"当前计算环境与测速不同: {key}")
    perf_head=report.get("perf_head","")
    require(len(perf_head)==40 and all(c in "0123456789abcdef" for c in perf_head),"报告缺完整PERF_HEAD")
    if perf_head != a.train_head:
        diff=subprocess.run(["git","diff","--exit-code",perf_head,a.train_head,"--","src","scripts","packages","pyproject.toml","uv.lock"],
                            cwd=a.repo,text=True,capture_output=True)
        require(diff.returncode==0,"PERF_HEAD到TRAIN_HEAD的训练、启动代码或依赖已有变化，须重新验证测速")
    return expected


def time_now():
    return datetime.datetime.now(datetime.timezone.utc).timestamp()


def resources(repo,mode):
    require(shutil.disk_usage(repo).free >= 400*10**9,"scratch余量不足400GB")
    query=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True,capture_output=True)
    require(query.returncode==0,"GPU查询失败")
    rows=[tuple(int(x.strip()) for x in line.split(",")) for line in query.stdout.splitlines()]
    require(sorted(rows)==[(i,0) for i in range(8)],f"八卡并非全部空闲: {rows}")
    sessions=subprocess.run(["tmux","list-sessions","-F","#{session_name}"],text=True,capture_output=True)
    require(sessions.returncode==0 or "no server running" in sessions.stderr,"tmux会话查询失败")
    names=sessions.stdout.splitlines()
    if mode == "prod":
        require(not any(n.startswith(("m2048-perf-","m2048-perf-gpu-")) for n in names),"本轮测速会话尚未退出")
    return names


def check_contract(a,argv,check):
    try:
        require(a.expected_run_name and a.run_root,"启动契约必须明确run名与实际输出根")
        validate_argv(argv,a.launch_mode)
        parsed=parse_config(argv,a.repo)
        actual=validate_config(parsed,a)
        data=validate_data(a)
        sys.path.insert(0,str(Path(a.repo)/"scripts/assets"))
        import assets_lock
        assets_lock.require(["pi05_base","paligemma_tokenizer"],level="full")
        print("INITIAL_ASSETS=PASS level=full assets=pi05_base,paligemma_tokenizer",flush=True)
        data.update(yaml_sha256=a.history_config_sha256,norm_stats_file_sha256=a.norm_stats_sha256)
        require(not os.environ.get("XLA_FLAGS"),"必须unset验证阶段XLA_FLAGS")
        require(os.environ.get("CUDA_VISIBLE_DEVICES")=="0,1,2,3,4,5,6,7","生产形制必须明确使用物理GPU0–7")
        if a.launch_mode=="prod":
            require(os.environ.get("TRAIN_TIMING_STEPS","0")=="0","正式训练禁止未测的profiler")
            approved=validate_approval(a,actual,data)
            print(f"USER_APPROVAL=PASS report_sha={approved['report_sha256']} runner_sha={approved['runner_sha256']} train_head={a.train_head} run={a.expected_run_name}",flush=True)
        names=resources(a.repo,a.launch_mode)
        print("GPU_IDLE=PASS gpus=0,1,2,3,4,5,6,7 used_mib=0 sessions="+json.dumps(names),flush=True)
        check("LAUNCH_CONFIG","实际配置绑定",True,"已定生产形制",json_sha(actual))
        print(f"LAUNCH_CONFIG=PASS mode={a.launch_mode} steps={actual['num_train_steps']} batch=128 workers=16 fsdp=8",flush=True)
        print("LAUNCH_RESOLVED="+json.dumps({"actual":actual,"data":data},sort_keys=True),flush=True)
    except (ValueError,OSError,KeyError,TypeError,subprocess.SubprocessError) as error:
        check("LAUNCH_CONFIG","实际配置绑定",False,"配置、数据、资源及同意全部通过",str(error))
