"""2048无motion档的配置、输入、逐位置功能与保存加载验收。"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
YAML = "perceptual-framesamp-modul-32frame-8x8.yaml"
OLD_YAML = "perceptual-framesamp-modul-8frame-8x8.yaml"
KEYS = ("static_image_emb", "static_pos_emb", "static_state_emb", "static_mask")
NONE_KEYS = ("motion_emb", "motion_pos", "motion_mask", "mem_order")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def exact(a, b):
    if a is None or b is None:
        return a is b
    aa, bb = np.asarray(a), np.asarray(b)
    return aa.dtype == bb.dtype and aa.shape == bb.shape and aa.tobytes() == bb.tobytes()


def config(a):
    from mme_vla_suite.models.config.utils import get_history_config
    return get_history_config(a.history_config)


def data_config(a):
    from openpi.shared.normalize import deserialize_json
    return SimpleNamespace(norm_stats=deserialize_json(Path(a.norm_stats).read_text()), use_quantile_norm=True)


def dataset(a):
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset
    return FrameSampDataset(a.store, data_config(a), config(a), 20, source_root=a.source, manifest_path=a.manifest)


def frames(a):
    from mme_vla_suite.shared.sampling import even_sampling_indices
    expected = {t:list(range(t+1)) for t in (0,1,7,8,15,30,31)}
    expected.update({
        32: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32],
        33: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,33],
        63: [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,63],
        64: [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,64],
        1151: [0,37,74,111,148,185,222,259,297,334,371,408,445,482,519,556,594,631,668,705,742,779,816,853,891,928,965,1002,1039,1076,1113,1151],
    })
    for t, want in expected.items():
        require(list(even_sampling_indices(t,32)) == want, f"独立选帧失败: t={t}")
    print("FRAME_SELECT=PASS cases=12 max_frames=32 mismatches=0")
    return expected


def yaml_check(a):
    from omegaconf import OmegaConf
    from mme_vla_suite.models.config.utils import get_history_config
    old,new = [OmegaConf.to_container(get_history_config(x)) for x in (OLD_YAML,a.history_config)]
    diff = sorted(k for k in set(old)|set(new) if old.get(k) != new.get(k))
    require(diff == ["budget"] and old["budget"] == 512 and new["budget"] == 2048
            and type(new["budget"]) is int and "motion" not in new, "YAML差异不止budget或新值类型错误")
    print("YAML_DIFF=PASS keys_diff=['budget'] old=512 new=2048 budget_type=int motion_enabled=0")
    return {"keys_diff":diff}


def guards(a):
    import jax.numpy as jnp
    from flax import nnx
    from omegaconf import OmegaConf
    import openpi.shared.array_typing as at
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset
    from mme_vla_suite.models.config.utils import get_history_config
    from mme_vla_suite.models.representation.percep_mem import PerceptualMemory
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0, HistoryPi0Config
    from ref_npy_dataset import RefNpyFrameSampDataset
    hc = config(a)
    dc = data_config(a)
    def construct(h):
        packed = FrameSampDataset(a.store,dc,h,20,source_root=a.source,manifest_path=a.manifest)
        ref = RefNpyFrameSampDataset(a.source,a.manifest,dc,h,20)
        require(packed._max_frames == ref._max_frames == 32, "新档帧预算错误")
        packed.close(); ref.close()
    construct(hc)
    explicit = OmegaConf.create(OmegaConf.to_container(hc)); explicit.motion={"enabled":False}
    construct(explicit)
    bad = [{"budget":2048.5},{"budget":"2048"},{"num_views":True},
           {"integration_type":"context","memory_token_dim":2048},{"motion":{"enabled":True}},
           {"token_per_image":16},{"num_views":2}]
    for change in bad:
        h = OmegaConf.create({**OmegaConf.to_container(hc),**change})
        for cls in (FrameSampDataset,RefNpyFrameSampDataset):
            try:
                if cls is FrameSampDataset:
                    cls(a.store,dc,h,20,source_root=a.source,manifest_path=a.manifest)
                else:
                    cls(a.source,a.manifest,dc,h,20)
            except ValueError:
                pass
            else:
                raise ValueError(f"未拒绝坏配置: {change} {cls}")
    for y,sub in ((OLD_YAML,"framesamp-8x8"),("perceptual-framesamp-modul.yaml","framesamp")):
        ds = FrameSampDataset(str(Path(a.store).parent/sub),dc,get_history_config(y),20,source_root=a.source,manifest_path=a.manifest)
        ds.close()
    mem = PerceptualMemory(hc,nnx.Rngs(42))
    arrays = [jnp.ones((1,2048,d),jnp.float32) for d in (2048,768,8)]
    # 不建立大模型；直接调用真实接口，记忆编码器是真实实例。
    fake = SimpleNamespace(mem_encoder=mem,history_config=hc)
    obs = SimpleNamespace(**dict(zip(KEYS[:3],arrays)),static_mask=jnp.ones((1,2048),bool),
                          motion_emb=None,motion_pos=None,motion_mask=None)
    with at.disable_typechecking():
        tokens,mask,_,_ = HistoryPi0.embed_memory(fake,obs)
        require(tokens.shape[:2] == mask.shape == (1,2048), "正确输入未得到2048记忆")
        for i in range(3):
            for wrong in (None,arrays[i][:,:512],jnp.ones((2,2048,arrays[i].shape[-1]))):
                changed = list(arrays); changed[i]=wrong
                try:
                    mem(*changed)
                except ValueError:
                    pass
                else:
                    raise ValueError(f"static输入错误未被拒绝: {KEYS[i]}")
        for wrong in (None,jnp.ones((1,2048),jnp.float32),jnp.ones((1,1),bool),jnp.ones((1,512),bool)):
            broken = SimpleNamespace(**{**vars(obs),"static_mask":wrong})
            try:
                HistoryPi0.embed_memory(fake,broken)
            except ValueError:
                pass
            else:
                raise ValueError("static_mask坏形制未被拒绝")
    spec,_ = HistoryPi0Config(use_history=True,history_config=hc,pi05=True).inputs_spec(batch_size=2)
    require(all(getattr(spec,k).shape[:2] == (2,2048) for k in KEYS), "inputs_spec预算错误")
    if not a.skip_opt:
        child = [sys.executable,"-O",str(Path(__file__).resolve()),"guards","--skip-opt"]
        for key in ("store","source","manifest","norm_stats","history_config"):
            child += ["--"+key.replace("_","-"),str(getattr(a,key))]
        subprocess.run(child,check=True)
    print("GUARD_2048=PASS accept=1 reject_type=3 reject_combo=4 legacy_accept=2 ref_same=1")
    print(f"STATIC_SHAPE=PASS keys=4 bad_mask_b1_rejected=1 optimized_rejected={int(not a.skip_opt or not __debug__)}")
    return {"type_reject":3,"combo_reject":4,"optimized":not __debug__}


def assembly(a):
    from ref_npy_dataset import RefNpyFrameSampDataset
    import _common as C
    manifest = C.load_manifest(Path(a.manifest))
    ds,ref = dataset(a),RefNpyFrameSampDataset(a.source,a.manifest,data_config(a),config(a),20)
    selected, descriptors = [], None
    try:
        for task in range(4):
            for ep in manifest["episodes"][task*400:task*400+50]:
                for t in (ep["exec_start_idx"],ep["num_timesteps"]-1):
                    i = C.index_of(ep,t)
                    left,right = ds[i],ref[i]
                    require(set(left) == set(right), f"样本键集不同: {i}")
                    for key in left:
                        require(exact(left[key],right[key]),f"源NPY对packed不同: {i}/{key}")
                    require(all(left[k] is None for k in NONE_KEYS), "无motion档四键并非None")
                    require(left["static_mask"].sum() == 2048, "真实样本不是满32帧")
                    descriptors = {k:{"shape":list(left[k].shape),"dtype":str(left[k].dtype),"bytes":left[k].nbytes} for k in KEYS}
                    selected.append(i)
    finally:
        ds.close(); ref.close()
    require(len(selected) == len(set(selected)) == 400, "四任务有界取样数量或唯一性不符")
    print("REF_VS_PACKED=PASS samples=400 keys=8 none_keys=4 tasks=4 seams=200 tails=200 mismatches=0 mask_sum_all=2048")
    return {"indices":selected,"descriptors":descriptors,"proof":"独立特征读取、padding与身份换算；共享采样、归一化和下游模型"}


def pad(a):
    from mme_vla_suite.shared.data_utils import right_padding_token_emb
    ds = dataset(a)
    try:
        store = ds._ensure_store()
        for n in range(1,33):
            f = np.arange(n,dtype=np.int64)
            img,pos,state = store.read_image_rows(ds._row_base[0]+f),store.pos_rows(f),store.state_rows(ds._row_base[0]+f)
            out = ds._pad(img,pos,state,n)
            reference = right_padding_token_emb(img,pos,state,np.ones(n,bool),32)
            require(all(exact(x,y) for x,y in zip(out,reference,strict=True)), f"两种padding不等: n={n}")
            for x,dt in zip(out,("bfloat16","float32","float32","bool"),strict=True):
                require(str(x.dtype) == dt and x.shape[0] == 32, "padding形状或dtype不同")
                require(not np.any(x[n:]), "补零尾部非零")
            require(out[3].sum() == n and out[3][:n].all(), "padding mask错误")
        f = np.asarray(frames(a)[1151],dtype=np.int64)
        full = ds._pad(store.read_image_rows(ds._row_base[0]+f),store.pos_rows(f),store.state_rows(ds._row_base[0]+f),32)
        require(full[3].all(), "真实长帧号装配mask错误")
    finally:
        ds.close()
    print("PAD_SYNTH=PASS n_cases=32 full=1 zero_tail=1 mask_ok=1 dtype_ok=1 online_pad_bitexact=1")
    return {"n_cases":32,"full":1}


def online(a):
    import jax.numpy as jnp
    from mme_vla_suite.policies.framesamp_memory import FrameSampMemory
    hc=config(a)
    require((hc.budget,hc.token_per_image,hc.num_views)==(2048,64,1),"在线检查必须使用2048/64/1配置")
    summaries=[]
    for tokens in (256,64):
        def encoder(x):
            # 帧号随像素编码进桩特征，避免全零/全一桩漏掉错选帧或重排。
            values=jnp.rint((x[:,:,0,0,0]+1)*127.5)
            return jnp.broadcast_to(values[...,None,None],(*x.shape[:2],tokens,2048)).astype(jnp.bfloat16)
        memory = FrameSampMemory(vision_enc_fn=encoder,token_per_image=int(hc.token_per_image),num_views=int(hc.num_views))
        previous = 0
        for t in (0,7,31,32,40):
            n=t+1-previous
            frame_ids=np.arange(previous,t+1)
            images=np.broadcast_to(frame_ids[:,None,None,None,None],(n,1,224,224,3)).astype(np.uint8)
            states=np.broadcast_to(frame_ids[:,None],(n,8)).astype(np.float32)
            memory.add_buffer(images,states,list(range(previous,t+1)))
            previous=t+1
            selected=list(range(t+1)) if t<32 else [t*i//31 for i in range(32)]
            require(list(memory.get_frame_sampling_indices(t,2048,64))==selected,"在线选帧与独立整数公式不同")
            result = memory.prepare_frame_sampling(t,int(hc.budget),int(hc.token_per_image),memory.default_history_feats_gather_fn)
            for x,shape,dt in zip(result,((2048,2048),(2048,768),(2048,8),(2048,)),("bfloat16","float32","float32","bool"),strict=True):
                require(x.shape == shape and str(x.dtype) == dt,"在线装配shape或dtype错误")
            require(result[3].sum() == 64*min(t+1,32),"在线mask计数错误")
            require(np.all(result[0][result[3]].astype(np.float32)==np.repeat(selected,64)[:,None]),"桩特征的帧身份或池化错误")
            require(np.all(result[2][result[3]]==np.repeat(selected,64)[:,None]),"在线状态的帧身份错误")
            require(exact(result[1][result[3]],memory.pos_emb[selected].reshape(-1,768)),"在线位置表选帧或展开错误")
            require(all(not np.any(x[~result[3]]) for x in result[:3]),"在线补零尾部错误")
        summaries.append({"encoder_tokens":tokens,"pos_table_bytes":memory.pos_emb.nbytes})
        del memory
        gc.collect()
    print("ONLINE_ASM=PASS steps=5 shapes=(2048,2048)/(2048,768)/(2048,8)/(2048,) dtype_stage=pre_normalize dtype_ok=1 mask_sums=64,512,2048,2048,2048 pooling_branches=2")
    return summaries


def _collate_child(samples,queue,consumed):
    """spawn子进程实际经过torch队列共享内存交付，等待父进程完成读取。"""
    from openpi.training.data_loader import _collate_fn_shm
    try:
        queue.put((True,_collate_fn_shm(samples)))
        if not consumed.wait(120):
            raise RuntimeError("父进程未确认共享内存读取")
    except Exception as error:
        queue.put((False,repr(error)))


def collate(a):
    import jax.numpy as jnp
    import ml_dtypes
    import multiprocessing
    from openpi.training.data_loader import _from_shared_torch
    ns=data_config(a).norm_stats["state"]
    ds=dataset(a)
    try:
        store=ds._ensure_store()
        for counts in ((1,8,31),(1,32,8),(32,32,32)):
            samples=[]
            for n in counts:
                f=np.arange(n,dtype=np.int64)
                img,pos,state,mask=ds._pad(store.read_image_rows(ds._row_base[0]+f),store.pos_rows(f),store.state_rows(ds._row_base[0]+f),n)
                state=np.repeat(state,64,axis=0)
                state=(state-ns.q01)/(ns.q99-ns.q01+1e-6)*2.-1.
                samples.append(dict(static_image_emb=img.reshape(2048,2048),static_pos_emb=pos.reshape(2048,768),
                                    static_state_emb=state,static_mask=np.repeat(mask,64),**dict.fromkeys(NONE_KEYS)))
            ctx=multiprocessing.get_context("spawn")
            queue,consumed=ctx.Queue(),ctx.Event()
            worker=ctx.Process(target=_collate_child,args=(samples,queue,consumed))
            worker.start()
            try:
                ok,shared=queue.get(timeout=120)
                require(ok,f"合成collate子进程失败: {shared}")
                require(all(shared[k].is_shared() for k in KEYS),"合成batch未经过真实共享内存")
                batch=_from_shared_torch(shared)
                for k,dt in zip(KEYS,(ml_dtypes.bfloat16,np.float32,np.float64,np.bool_),strict=True):
                    expected=np.stack([s[k] for s in samples])
                    require(batch[k].dtype == dt and exact(batch[k],expected),f"共享内存dtype或内容变动: {k}")
                    wanted=expected.astype(np.float32) if k == "static_state_emb" else expected
                    require(exact(np.asarray(jnp.asarray(batch[k])),wanted),f"JAX交付错误: {k}")
                require(all(batch[k] is None for k in NONE_KEYS),"共享内存丢失None键")
                del shared,batch,samples
            finally:
                consumed.set();worker.join(10)
                if worker.is_alive():worker.terminate();worker.join()
                queue.close();queue.join_thread()
            require(worker.exitcode==0,"合成collate子进程非零退出")
    finally:
        ds.close()
    print("JAX_DELIVERY=PASS keys=8 f64_to_f32=1 mismatches=0")
    print("COLLATE_MIXED=PASS kinds=3 dtype_contract=1 mismatches=0")
    return {"kinds":3}


def init(a):
    import jax
    from mme_vla_suite.training.config import get_config
    from mme_vla_suite.models.config.utils import get_history_config
    from openpi.training import sharding
    sys.path.insert(0,str(ROOT/"scripts/training"))
    import train
    from _common import leaf_sha256
    mesh=sharding.make_mesh(1)
    _,rng=jax.random.split(jax.random.key(42))
    outputs=[]
    for name in (OLD_YAML,a.history_config):
        c=get_config("mme_vla_suite")
        c=dataclasses.replace(c,model=dataclasses.replace(c.model,use_history=True,history_config=get_history_config(name)))
        state,_=train.init_train_state(c,rng,mesh,resume=False)
        jax.block_until_ready(state)
        result={}
        for path,value in jax.tree_util.tree_flatten_with_path(state.params.to_pure_dict())[0]:
            arr=np.asarray(jax.device_get(value))
            require(np.isfinite(arr).all(),"初态非有限")
            result[jax.tree_util.keystr(path)]={"shape":list(arr.shape),"dtype":str(arr.dtype),"sha256":leaf_sha256(arr)}
        outputs.append(result)
        del state
        gc.collect()
    require(outputs[0] == outputs[1] and len(outputs[0]) == 61,"512与2048初态不相同")
    print("INIT_SAME_512_2048=PASS leaves=61 mismatches=0 shape_mismatch=0")
    return outputs[0]


def memory_oracle(a):
    import jax
    import jax.numpy as jnp
    from mme_vla_suite.models.integration import history_gemma as hg
    rng=np.random.default_rng(42)
    x=rng.normal(size=(1,20,1024)).astype(np.float32)
    mem=rng.normal(size=(1,2048,1024)).astype(np.float32)
    module=hg.MemoryAttention()
    params=module.init(jax.random.key(42),jnp.asarray(x),jnp.asarray(mem),jnp.ones((1,2048),bool))
    p=jax.tree.map(np.asarray,params["params"])
    require(set(p) == {"mem_rms_norm","q_einsum_mem","kv_einsum_mem","out_einsum_mem"},"MemoryAttention参数树变化")
    print("MEMATTN_PARAMS="+json.dumps({k:{kk:list(vv.shape) for kk,vv in v.items()} for k,v in p.items()}))

    def oracle(length,n,compact=False):
        def rms(z):
            var=np.mean(np.square(z),axis=-1,keepdims=True,dtype=np.float32)
            return z*(np.float32(1)/np.sqrt(var+np.float32(1e-6)))*(np.float32(1)+p["mem_rms_norm"]["scale"])
        def rope(z,positions):
            exponents=np.arange(128,dtype=np.float32)*np.float32(2/256)
            # 独立高精度求幂后舍入到f32；numpy直接f32幂在本机有1 ULP误差，
            # 位置2048会把频率误差放大。后续RoPE各阶段仍为f32，验收容差不变。
            scales=np.power(10000.,exponents.astype(np.float64)).astype(np.float32)
            theta=np.asarray(positions,np.float32)[None,:,None,None]/scales[None,None,None,:]
            sin,cos=np.sin(theta),np.cos(theta)
            first,second=np.split(z,2,axis=-1)
            return np.concatenate((first*cos-second*sin,second*cos+first*sin),axis=-1)
        q=np.einsum("btd,ndh->btnh",rms(x),p["q_einsum_mem"]["w"],optimize=True)
        kv=np.einsum("bsd,akdh->abskh",rms(mem[:,:length]),p["kv_einsum_mem"]["w"],optimize=True)
        q=rope(q,np.arange(64 if compact else length,(64 if compact else length)+20))*np.float32(1/16)
        k=rope(kv[0],np.arange(length));v=kv[1]
        logits=np.einsum("btgh,bsh->bgts",q,k[:,:,0],optimize=True).astype(np.float32)
        logits[:,:,:,n:]=np.float32(-2.3819763e38)
        ex=np.exp(logits-np.max(logits,axis=-1,keepdims=True))
        probs=ex/np.sum(ex,axis=-1,keepdims=True,dtype=np.float32)
        encoded=np.einsum("bgts,bsh->btgh",probs,v[:,:,0],optimize=True)
        return np.einsum("btgh,ghd->btd",encoded,p["out_einsum_mem"]["w"],optimize=True)

    def apply(length,n,garbage=False,compact=False):
        memory=mem[:,:length].copy()
        if garbage:
            memory[:,n:]=1000
        mask=np.arange(length)[None,:]<n
        original=hg._apply_rope
        if compact:
            def compact_rope(z,*,positions):
                if z.shape[1] == 20:
                    positions=jnp.arange(64,84,dtype=jnp.int32)[None,:]
                return original(z,positions=positions)
            hg._apply_rope=compact_rope
        try:
            return np.asarray(module.apply(params,jnp.asarray(x),jnp.asarray(memory),jnp.asarray(mask)))
        finally:
            hg._apply_rope=original

    errors={}
    for length in (512,2048):
        for n in (1,64,length):
            actual,want=apply(length,n),oracle(length,n)
            error=float(np.max(np.abs(actual-want)))
            tolerance=1e-6+1e-5*float(np.max(np.abs(want)))
            require(error <= tolerance,f"独立oracle不符: length={length} n={n} error={error} tolerance={tolerance}")
            require(exact(actual,apply(length,n,garbage=True)),"mask外垃圾改变输出")
            errors[f"{length}/{n}"]={"max_abs":error,"tolerance":tolerance}
    require(np.allclose(apply(512,1),apply(2048,1),atol=1e-6,rtol=1e-5),"单key退化输出随长度改变")
    oa,ob=oracle(512,64),oracle(2048,64)
    length_delta=float(np.max(np.abs(oa-ob)))
    tol=1e-6+1e-5*max(float(np.max(np.abs(oa))),float(np.max(np.abs(ob))))
    require(length_delta > 10*tol,"非退化fixture对RoPE长度不敏感")
    for length in (512,2048):
        actual,want=apply(length,64,compact=True),oracle(length,64,compact=True)
        require(np.max(np.abs(actual-want)) <= 1e-6+1e-5*np.max(np.abs(want)),"统一query位置与oracle不同")
    ca,cb=apply(512,64,compact=True),apply(2048,64,compact=True)
    require(np.max(np.abs(ca-cb)) <= 1e-6+1e-5*np.max(np.abs(cb)),"统一query位置后长度差异未消失")
    print("MEMATTN_ORACLE=PASS atol=1e-6 rtol=1e-5 singleton_length_invariant=1 garbage_bitexact=1 nondegenerate_rope_sensitive=1 compact_q_match=1")
    return {"errors":errors,"length_delta":length_delta}


def func(a):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import openpi.shared.array_typing as at
    from mme_vla_suite.training.config import get_config
    from openpi.models.model import restore_params
    from compatible_pi05_weights import merge_compatible
    from motion_gates_model import _obs_from_samples
    from _common import leaf_sha256
    sys.path.insert(0,str(ROOT/"scripts/assets"))
    import assets_lock
    lock=assets_lock.load_lock()
    assets_lock.require(["pi05_base"],level="full",lock=lock)
    c=dataclasses.replace(get_config("mme_vla_suite_b128_80k").model,history_config=config(a),
                          paligemma_variant="gemma_150m",action_expert_variant="gemma_300m")
    model=c.create(jax.random.key(42))
    initial=nnx.state(model,nnx.Param)
    path=ROOT/"v1-store/models/openpi-assets/checkpoints/pi05_base/params"
    merged,pretrained=merge_compatible(initial.to_pure_dict(),restore_params(path,restore_type=np.ndarray))
    pretrained.update(path=str(path.resolve()),asset=assets_lock.asset("pi05_base",lock))
    initial.replace_by_pure_dict(jax.tree.map(jnp.asarray,merged));nnx.update(model,initial)
    del initial,merged
    random=np.random.default_rng(42)
    sample={k:random.normal(0,.25,size=(2048,d)).astype(np.float32) for k,d in zip(KEYS[:3],(2048,768,8))}
    sample["static_mask"]=np.ones(2048,bool)
    full=_obs_from_samples([sample],motion=False)
    graph,trainable,frozen=nnx.split(model,nnx.All(nnx.Param,nnx.Not(c.get_freeze_filter())),...)
    rng=jax.random.key(7)

    def loss_fn(tp,fp,o):
        m=nnx.merge(graph,tp,fp)
        actions=jnp.asarray(np.random.default_rng(8).normal(size=(o.state.shape[0],20,32)).astype(np.float32))
        with at.disable_typechecking():
            return jnp.mean(m.compute_loss(rng,o,actions,train=False))
    loss_only=jax.jit(loss_fn)
    loss_grads=jax.jit(jax.value_and_grad(loss_fn,argnums=0))

    def digest(o):
        loss,gs=loss_grads(trainable,frozen,o)
        require(np.isfinite(float(loss)),"功能检验loss非有限")
        shas={}
        for p,v in jax.tree_util.tree_flatten_with_path(gs.to_pure_dict())[0]:
            arr=np.asarray(jax.device_get(v))
            require(np.isfinite(arr).all(),"功能检验梯度非有限")
            shas[jax.tree_util.keystr(p)]=leaf_sha256(arr)
        return float(loss),shas

    probes=[digest(full) for _ in range(3)]
    require(all(x[0].hex()==probes[0][0].hex() and x[1]==probes[0][1] for x in probes),"A/A非确定性：禁止排除梯度叶以凑通过")
    # 帧带扰动使用loss_only，因此其A/A基线也必须走同一个已编译函数。
    loss_probes=[float(loss_only(trainable,frozen,full)) for _ in range(3)]
    require(all(np.isfinite(v) and v.hex()==loss_probes[0].hex() for v in loss_probes),"loss_only A/A不确定或非有限")
    noise_floor=max(abs(v-loss_probes[0]) for v in loss_probes)
    jit_path_delta=abs(loss_probes[0]-probes[0][0])
    print(f"DETERMINISM=PASS probes=3 gradient_leaves={len(probes[0][1])} excluded=0",flush=True)
    print(f"LOSS_PATH_BASELINE=PASS loss_only_probes=3 jit_path_delta={jit_path_delta} band_reference=loss_only",flush=True)

    # 权重显式传入JIT，避免闭包把整棵参数树编译为巨大的设备常量。
    def input_function(tp,fp,o,drop=False):
        def f(si,sp):
            if drop:
                keep=(jnp.arange(2048)%64 == 0)[None,:,None]
                si,sp=jnp.where(keep,si,0),jnp.where(keep,sp,0)
            return loss_fn(tp,fp,dataclasses.replace(o,static_image_emb=si,static_pos_emb=sp))
        return jax.grad(f,argnums=(0,1))(o.static_image_emb,o.static_pos_emb)
    input_normal=jax.jit(input_function)
    input_drop=jax.jit(lambda tp,fp,o:input_function(tp,fp,o,True))

    def token_table(o,gradients):
        tables=[]
        valid=np.asarray(o.static_mask)
        for grad in gradients:
            arr=np.asarray(grad,dtype=np.float32)
            require(np.isfinite(arr).all(),"输入梯度非有限")
            norm=np.linalg.norm(arr.astype(np.float64),axis=-1)
            tables.append(norm)
            if np.any(~valid):
                require(np.all(arr[~valid] == 0),"无效位置梯度泄漏")
        return tables
    full_norms=token_table(full,input_normal(trainable,frozen,full))
    require(all(np.all(t>0) for t in full_norms),"TOKEN_GRAD有位置未参与")
    print("TOKEN_GRAD=PASS tokens=2048 image_nonzero=2048 pos_nonzero=2048",flush=True)
    dropped=token_table(full,input_drop(trainable,frozen,full))
    require(all(np.count_nonzero(t) == 32 and np.all(t[:,::64]>0) for t in dropped),"故障注入没有隔离每帧63位置")
    require(not all(np.all(t>0) for t in dropped),"逐位置门未拒绝删token故障")
    print("TOKEN_DROP_NEGATIVE=PASS rejected=1 kept_per_band=1",flush=True)
    deltas=[]
    for band in range(32):
        changed=dataclasses.replace(full,static_image_emb=full.static_image_emb.at[:,band*64:(band+1)*64].add(.25))
        value=float(loss_only(trainable,frozen,changed))
        require(np.isfinite(value),"帧带扰动loss非有限")
        deltas.append(abs(value-loss_probes[0]))
    require(all(d>noise_floor for d in deltas),"帧带扰动未全部超过A/A噪声")
    print(f"FRAME_BAND_PERTURB=PASS bands=32 above_noise=32 noise_floor={noise_floor} min_delta={min(deltas)}",flush=True)

    samples=[]
    for n in (1,8,31):
        samples.append({**sample,"static_mask":np.arange(2048)<n*64})
    short=_obs_from_samples(samples,motion=False)
    short_norms=token_table(short,input_normal(trainable,frozen,short))
    valid=np.asarray(short.static_mask)
    require(valid.any() and (~valid).any() and all(np.all(t[valid]>0) for t in short_norms),"短历史有效位置梯度为零")
    dirty={}
    for key in KEYS[:3]:
        arr=np.array(getattr(short,key));arr[~valid]=random.normal(0,1000,arr[~valid].shape)
        dirty[key]=jnp.asarray(arr)
    garbage=dataclasses.replace(short,**dirty)
    left,right=digest(short),digest(garbage)
    require(left[0].hex()==right[0].hex() and left[1]==right[1] and len(left[1])>0,"mask外垃圾改变loss或任一可训练梯度")
    @jax.jit
    def act(tp,fp,o):
        m=nnx.merge(graph,tp,fp)
        noise=jnp.asarray(np.random.default_rng(9).normal(size=(o.state.shape[0],20,32)).astype(np.float32))
        with at.disable_typechecking():
            return m.sample_actions(jax.random.key(3),o,noise=noise,num_steps=10)
    la,ra=np.asarray(act(trainable,frozen,short)),np.asarray(act(trainable,frozen,garbage))
    require(np.isfinite(la).all() and exact(la,ra),"mask外垃圾改变固定noise动作")
    changed=np.array(short.static_image_emb);changed[valid]=random.normal(0,1000,changed[valid].shape)
    short_baseline=float(loss_only(trainable,frozen,short))
    negative=float(loss_only(trainable,frozen,dataclasses.replace(short,static_image_emb=jnp.asarray(changed))))
    require(np.isfinite(short_baseline) and np.isfinite(negative) and negative != short_baseline,
            "有效位置垃圾负对照没有有限的loss变化")
    print(f"MASK_SYNTH=PASS n_set=1,8,31 loss_bitexact=1 actions_bitexact=1 grad_sha_same={len(left[1])}/{len(left[1])} masked_grad_zero=1 valid_tokens_nonzero=1 negative_control_changed=1",flush=True)
    return {"full_token_norms":[t.tolist() for t in full_norms],"short_token_norms":[t.tolist() for t in short_norms],
            "band_deltas":deltas,"noise_floor":noise_floor,"loss":left[0],"gradient_shas":left[1],"pretrained":pretrained,
            "loss_only_probes":loss_probes,"loss_grad_baseline":probes[0][0],"jit_path_delta":jit_path_delta,
            "valid_garbage_baseline":short_baseline,"valid_garbage_loss":negative}


def ckpt(a):
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    import re
    from flax import nnx,traverse_util,linen
    from _common import leaf_sha256,resolve_dtype
    from check_modul_train_records import MEM_LEAVES,SCALARS
    from mme_vla_suite.training.config import get_config
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    from mme_vla_suite.models.integration import history_gemma as hg
    from mme_vla_suite.policies.policy_config import create_trained_policy,_load_resolved_snapshot
    from openpi.models.model import restore_params
    from openpi.training.data_loader import transform_dataset,_collate_fn
    import openpi.shared.array_typing as at
    rec=Path(a.records)
    states=[json.loads(x) for x in (rec/"param_checksums.jsonl").read_text().splitlines()]
    require(states[0]["state_step"] == 0 and states[-1]["state_step"] == 100,"100步初末状态不符")
    require(all(all(v is True for v in r["per_leaf_finite"].values()) and set(r["per_leaf"]) == set(states[0]["per_leaf"]) for r in states),"100步状态摘要非有限或叶集合改变")
    metrics=[json.loads(x) for x in (rec/"metrics.jsonl").read_text().splitlines()]
    require([r["step"] for r in metrics] == list(range(100)),"100步标量不完整")
    require(all(set(r)-{"step","wall_time"} == SCALARS and all(np.isfinite(r[k]["dec"]) for k in SCALARS)
                and r["mem_enc_norm"]["dec"] > 0 for r in metrics),"100步标量非有限或记忆梯度为零")
    print("TRAIN100=PASS steps=100 finite_keys=5/5",flush=True)
    baseline=json.loads((Path(a.init_records)/"param_checksums.jsonl").read_text().splitlines()[0])
    require(baseline["per_leaf"] == states[0]["per_leaf"],"100步初态与20步初态不同")
    print("INIT_MATCH_100=PASS mismatches=0",flush=True)
    snapshots=[]
    for loop,record,wanted_step in ((0,states[0],0),(99,states[-1],100)):
        base=Path(a.state_dump_dir)/f"state_step_{loop}"
        metadata=json.loads(base.with_suffix(".json").read_text())
        require(set(metadata["leaves"]) == set(record["per_leaf"]),"原数组叶集与现场摘要不同")
        require("step" in metadata["leaves"],"原数组缺实际state.step")
        require(base.with_suffix(".bin").stat().st_size == metadata["total_bytes"],"数组文件字节数不符")
        ema={}
        for key,desc in metadata["leaves"].items():
            arr=np.memmap(base.with_suffix(".bin"),mode="r",dtype=resolve_dtype(desc["dtype"]),
                          offset=desc["offset"],shape=tuple(desc["shape"]))
            require(arr.nbytes == desc["nbytes"] and leaf_sha256(arr) == desc["sha256"] == record["per_leaf"][key],f"原数组内容或摘要不同: {key}")
            require(np.isfinite(arr).all(),f"原数组非有限: {key}")
            if key == "step":
                require(int(arr) == wanted_step,"原数组中的实际state.step不符")
            if key.startswith("ema_params"):
                parts=tuple(re.findall(r"\['([^']+)'\]",key))
                if parts[-1:] == ("value",):parts=parts[:-1]
                ema[parts]=np.asarray(arr).copy()
        require(len(ema) == 61,"EMA原数组参数树不是61叶")
        snapshots.append(ema)
    print("SAVE_ARRAYS=PASS init_state=0 final_state=100 raw_sha_match=1",flush=True)
    initial,final=snapshots
    ck=Path(a.ckpt)
    loaded=traverse_util.flatten_dict(restore_params(ck/"params",dtype=jnp.bfloat16,restore_type=np.ndarray))
    require(set(loaded) == set(final),"checkpoint与保存现场叶集合不同")
    converted={k:v.astype(ml_dtypes.bfloat16) for k,v in final.items()}
    require(all(exact(loaded[k],converted[k]) for k in final),"bf16加载叶与现场EMA不同")
    print("CKPT_LEAF_BF16=PASS mismatches=0",flush=True)
    changes={}
    for name in MEM_LEAVES:
        key=tuple(name.split("/"))
        require(key in initial and key in final,"缺记忆叶")
        changed_f32=not exact(initial[key],final[key])
        changed_bf16=not exact(initial[key].astype(ml_dtypes.bfloat16),converted[key])
        require(changed_f32 and changed_bf16,f"记忆叶在加载精度未区别于初态: {name}")
        difference=final[key].astype(np.float64)-initial[key].astype(np.float64)
        norm=float(np.linalg.norm(initial[key].astype(np.float64)))
        changes[name]={"changed_f32":changed_f32,"changed_bf16":changed_bf16,
                       "delta_l2":float(np.linalg.norm(difference)),"relative_l2":float(np.linalg.norm(difference))/norm if norm else None}
    print("CKPT_VS_INIT=PASS mem_leaves_changed=10/10",flush=True)
    hc,motion=_load_resolved_snapshot(ck.parent)
    require((hc.budget,hc.token_per_image,hc.memory_token_dim,motion) == (2048,64,1024,False),"加载配置形制错误")
    tc=get_config("mme_vla_suite")
    tc=dataclasses.replace(tc,model=dataclasses.replace(tc.model,history_config=hc,use_history=True),
                          data=dataclasses.replace(tc.data,assets=dataclasses.replace(tc.data.assets,
                              assets_dir=str(Path(a.norm_stats).parent.parent),asset_id="robomme")))
    reference=tc.model.load(traverse_util.unflatten_dict(jax.tree.map(jnp.asarray,converted)),remove_extra_params=False)
    policy=create_trained_policy(tc,ck,seed=42,norm_stats=data_config(a).norm_stats)
    ds=dataset(a)
    try:
        transformed=transform_dataset(ds,tc.data.create(tc.assets_dirs,tc.model))
        batch=_collate_fn([transformed[0]])
        obs=HistAugObservation.from_dict(jax.tree.map(jnp.asarray,batch))
    finally:
        ds.close()
    noise=jnp.asarray(np.random.default_rng(42).normal(size=(1,20,32)).astype(np.float32))
    seen=[]
    def capture(next_fn,args,kwargs,context):
        if isinstance(context.module,hg.MemoryAttention) and context.method_name=="__call__":
            memory=args[1] if len(args)>1 else kwargs["mem_seq"]
            seen.append(int(memory.shape[1]))
        return next_fn(*args,**kwargs)
    with linen.intercept_methods(capture):
        @nnx.jit
        def actions(model,observation):
            with at.disable_typechecking():
                return model.sample_actions(jax.random.key(9),observation,noise=noise,num_steps=10)
        left=np.asarray(actions(reference,obs))
        again=np.asarray(actions(reference,obs))
        right=np.asarray(actions(policy._model,obs))
    require(left.shape == right.shape == (1,20,32) and np.isfinite(left).all() and np.isfinite(right).all(),"checkpoint动作shape或有限性错误")
    rms=float(np.sqrt(np.mean((left.astype(np.float64)-right)**2)))
    aa=float(np.sqrt(np.mean((left.astype(np.float64)-again)**2)))
    require(rms <= 6.8e-5,"checkpoint动作偏差超过预设阈值")
    require(seen and set(seen) == {2048},f"实际MemoryAttention长度不是2048: {seen}")
    print(f"CKPT_ACTIONS=PASS rms_diff={rms} threshold=6.8e-5 num_steps=10 aa_rms={aa}",flush=True)
    print("CKPT_SHAPE=PASS budget=2048 mem_len=2048 motion_enabled=0",flush=True)
    return {"changes":changes,"actions_rms":rms,"aa_rms":aa,"memory_lengths":seen}


def capacity(a):
    import jax
    import jax.numpy as jnp
    from flax import nnx,traverse_util
    from openpi.models.model import restore_params
    from mme_vla_suite.training.config import get_config
    from mme_vla_suite.policies.policy_config import _load_resolved_snapshot
    from check_modul_train_records import MEM_LEAVES,SCALARS
    rec=Path(a.records)
    run=json.loads((rec/"speed_run.json").read_text())
    require(run.get("success") and run["mode"]=="smoke" and run["sampler_stopped"] and not run.get("sampling_error"),"容量任务或采样未完成")
    require(Path(a.log).read_text().splitlines()[-1]=="EXIT_CODE=0","容量任务没有成功退出")
    metrics=[json.loads(x) for x in (rec/"metrics.jsonl").read_text().splitlines()]
    require([r["step"] for r in metrics]==list(range(20)),"容量任务20步标量不完整")
    require(all(set(r)-{"step","wall_time"}==SCALARS and all(np.isfinite(r[k]["dec"]) for k in SCALARS) for r in metrics),"容量标量非有限")
    require(len(run["save_calls"])==1 and run["save_calls"][0]["state_step"]==20,"未真实保存第20次更新")
    checkpoint=Path(run["save_calls"][0]["checkpoint_root"])/"19"
    snapshot,enabled=_load_resolved_snapshot(checkpoint.parent)
    require((snapshot.budget,snapshot.token_per_image,snapshot.memory_token_dim,enabled)==(2048,64,1024,False),"容量checkpoint配置错误")
    tc=get_config("mme_vla_suite_b128_80k")
    tc=dataclasses.replace(tc,model=dataclasses.replace(tc.model,history_config=snapshot,use_history=True))
    restored=restore_params(checkpoint/"params",dtype=jnp.bfloat16)
    model=tc.model.load(restored,remove_extra_params=False)
    left=traverse_util.flatten_dict(nnx.state(model,nnx.Param).to_pure_dict(),sep="/")
    right=traverse_util.flatten_dict(restored,sep="/")
    require(set(left)==set(right) and len(left)==61,"容量参数树不是完整61叶")
    require(all(left[k].shape==right[k].shape and np.isfinite(np.asarray(right[k])).all() for k in left),"容量参数形状或数值错误")
    require(set(MEM_LEAVES)=={k for k in left if "mem_encoder" in k or "mem_attn" in k or "mem_rms_norm_ffn" in k},"容量参数缺少十个记忆叶")
    host=[json.loads(x) for x in (rec/"host_samples.jsonl").read_text().splitlines()]
    ratio=max(r["shm_used"]/r["shm_capacity"] for r in host)
    require(ratio<=.70,"共享内存超过70%")
    print("SMOKE20=PASS steps=20 finite=1 exit_code=0")
    print("PARAM_TREE_EXACT=PASS n_model=61 n_ckpt=61 missing=0 extra=0 shape_mismatch=0")
    print("MEM_PARAMS=PASS n=10")
    print(f"SHM_PEAK=PASS ratio={ratio}")
    return {"checkpoint":str(checkpoint),"shm_peak_ratio":ratio,"elapsed_s":run["end_wall"]-run["start_wall"]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("yaml","guards","frames","assembly","pad","online","collate","init","oracle","func","ckpt","capacity"))
    ds=ROOT/"v1-store/datasets/4task-v2-1600ep-604f16da"
    p.add_argument("--store",default=str(ds/"framesamp-8x8"))
    p.add_argument("--source",default=str(ds/"source"))
    p.add_argument("--manifest",default=str(ds/"meta/episode_manifest.json"))
    p.add_argument("--norm-stats",default=str(ROOT/"v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da/robomme/norm_stats.json"))
    p.add_argument("--history-config",default=YAML)
    p.add_argument("--out")
    p.add_argument("--skip-opt",action="store_true")
    for name in ("records","init-records","state-dump-dir","ckpt","log"):
        p.add_argument("--"+name)
    a=p.parse_args()
    if a.out:
        require(not Path(a.out).exists(),"拒绝覆盖验收报告")
    commands={"yaml":yaml_check,"guards":guards,"frames":frames,"assembly":assembly,"pad":pad,
              "online":online,"collate":collate,"init":init,"oracle":memory_oracle,"func":func,"ckpt":ckpt,"capacity":capacity}
    started=time.time()
    result=commands[a.command](a)
    if a.out:
        with Path(a.out).open("x") as f:
            json.dump({"command":a.command,"seconds":time.time()-started,"result":result},f,ensure_ascii=False,indent=2)


if __name__ == "__main__":
    main()
