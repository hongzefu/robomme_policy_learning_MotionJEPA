"""motion 接入的学习率、初态、RoPE 长度效应及真实 policy 配置前置检查。"""

import argparse
import dataclasses
import hashlib
import json
import pathlib
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"scripts/training"))


def check_lr():
    import jax
    import jax.numpy as jnp
    from mme_vla_suite.training.config import get_config
    old,new=get_config("mme_vla_suite_b128_60k"),get_config("mme_vla_suite_b128_80k")
    steps=jnp.arange(80000)
    a=np.asarray(jax.jit(jax.vmap(old.lr_schedule.create()))(steps))
    b=np.asarray(jax.jit(jax.vmap(new.lr_schedule.create()))(steps))
    if a.dtype != b.dtype or a.tobytes() != b.tobytes():
        raise ValueError("60k 与 80k 学习率并非逐步相同")
    if (new.batch_size,new.num_train_steps,new.fsdp_devices,new.num_workers) != (128,80000,8,16):
        raise ValueError("80k 具名配置没有承载约定的四项改动")
    print("GUARD_LR=PASS steps=80000 mismatches=0",flush=True)
    return {"steps":80000,"sha256":hashlib.sha256(a.tobytes()).hexdigest()}


def check_init():
    import jax
    from mme_vla_suite.training.config import get_config
    from mme_vla_suite.models.config.utils import get_history_config
    from openpi.training import sharding
    import train
    start=time.perf_counter()
    mesh=sharding.make_mesh(1)
    _,init_rng=jax.random.split(jax.random.key(42))
    results=[]
    for name,yaml in (("mme_vla_suite_b128_60k","perceptual-framesamp-modul-8frame-8x8.yaml"),
                      ("mme_vla_suite_b128_80k","perceptual-framesamp-modul-8frame-8x8-motion.yaml")):
        config=get_config(name)
        config=dataclasses.replace(config,model=dataclasses.replace(config.model,history_config=get_history_config(yaml)))
        state,_=train.init_train_state(config,init_rng,mesh,resume=False)
        jax.block_until_ready(state)
        leaves={}
        for path,value in jax.tree_util.tree_flatten_with_path(state.params.to_pure_dict())[0]:
            array=np.asarray(jax.device_get(value))
            if not np.isfinite(array).all(): raise ValueError("初态含非有限值")
            leaves[jax.tree_util.keystr(path)]={"shape":list(array.shape),"dtype":str(array.dtype),
                                              "sha256":hashlib.sha256(array.tobytes()).hexdigest()}
        results.append(leaves)
        del state
        print(f"INIT_COMMON_SIDE config={name} leaves={len(leaves)} elapsed={time.perf_counter()-start:.1f}s",flush=True)
    old,new=results
    shared=set(old)&set(new)
    bad=[k for k in shared if old[k] != new[k]]
    extra=set(new)-set(old)
    if bad or set(old)-set(new) or len(extra) != 4 or any("motion" not in k for k in extra):
        raise ValueError(f"公共初态或新增叶不符: common={bad}, extra={sorted(extra)}")
    modulation=[k for k in shared if "mem_attn" in k or "mem_rms_norm_ffn" in k]
    if len(modulation) != 6: raise ValueError("未覆盖六条 modulation 初态叶")
    print("INIT_COMMON=PASS common_mismatches=0 open_only=4",flush=True)
    return {"closed":old,"open":new,"modulation_leaves":modulation,"seconds":time.perf_counter()-start}


def check_rope():
    import jax
    import jax.numpy as jnp
    from openpi.models.gemma import _apply_rope
    rng=np.random.default_rng(42)
    q=jnp.asarray(rng.normal(size=(1,20,4,256)).astype(np.float32))
    keys=rng.normal(size=(1,672,1,256)).astype(np.float32)
    values=rng.normal(size=keys.shape).astype(np.float32)
    def run(length,real,garbage=False,positioned=True,compact=False):
        k,v=keys[:,:length].copy(),values[:,:length].copy()
        if garbage:
            k[:,real:]=1e3; v[:,real:]=-1e3
        k,v=jnp.asarray(k),jnp.asarray(v)
        query=q
        if positioned:
            offset=real if compact else length
            query=_apply_rope(query,positions=jnp.arange(offset,offset+20)[None])
            k=_apply_rope(k,positions=jnp.arange(length)[None])
        logits=jnp.einsum("bthd,bskd->bhts",query/16,k,preferred_element_type=jnp.float32)
        probability=jax.nn.softmax(jnp.where(jnp.arange(length)<real,logits,-2.3819763e38),axis=-1)
        encoded=jnp.einsum("bhts,bskd->bthd",probability,v)
        return np.asarray(probability[...,:real]),np.asarray(encoded)
    p144,y144=run(656,600); p160,y160=run(672,600)
    p512,_=run(512,400); p672,_=run(672,400)
    garbage_p,garbage_y=run(672,600,garbage=True)
    if garbage_p.tobytes()!=p160.tobytes() or garbage_y.tobytes()!=y160.tobytes():
        raise ValueError("同长度下 padding 内容影响了概率或输出")
    tv=lambda a,b: float(np.mean(np.abs(a-b).sum(axis=-1)/2))
    result=dict(seed=42,dtype="float32",heads=4,head_dim=256,real_keys_budget_pair=600,
                real_keys_baseline_pair=400,tv_144_160=tv(p144,p160),
                relL2_144_160=float(np.linalg.norm(y144-y160)/np.linalg.norm(y144)),
                tv_512_672=tv(p512,p672),pad_content_diff=0)
    a,_=run(656,600,positioned=False); b,_=run(672,600,positioned=False)
    result["tv_without_rope"]=tv(a,b)
    a,_=run(656,600,compact=True); b,_=run(672,600,compact=True)
    result["tv_without_padding_positions"]=tv(a,b)
    if not result["tv_144_160"] > 0 or not result["tv_512_672"] > 0:
        raise ValueError("长度变化没有产生预期的位置效应")
    print("ROPE_LEN_EFFECT=PASS "+" ".join(f"{k}={result[k]}" for k in
          ("tv_144_160","relL2_144_160","tv_512_672","pad_content_diff")),flush=True)
    return result


def check_policy_config():
    import jax.numpy as jnp
    from flax import nnx
    from mme_vla_suite.models.config.utils import get_history_config
    from mme_vla_suite.policies.policy import MME_VLA_Policy
    from mme_vla_suite.policies import motion_protocol as protocol
    hc=get_history_config("perceptual-framesamp-modul-8frame-8x8-motion.yaml")
    class StubModel(nnx.Module):
        def __init__(self): self.history_config=hc
        def sample_actions(self,rng,observation): return jnp.zeros((1,20,32))
        def vision_encode(self,x): return jnp.zeros((*x.shape[:2],64,2048),jnp.bfloat16)
    def motion(window,start):
        es=policy.exec_start_idx
        real=min(33,es-start) if start<es else 33
        expected=list(range(start,start+real))+[es-1]*(33-real)
        if protocol.stub_decode(window)!=expected: raise ValueError("真实 policy 装配的补帧内容不符")
        return np.full(768,float(start),np.float32)
    policy=MME_VLA_Policy(StubModel(),norm_stats={"state":SimpleNamespace(mean=np.zeros(8),std=np.ones(8))},motion_enc_fn=motion)
    keys={"stride","window_frames","budget","frame_size","pos_dim","dim","window_direction","grid_origin","demo_min_real_frames","demo_tail_pad"}
    if set(policy._motion_cfg)!=keys or (policy._motion_cfg["budget"],policy._motion_cfg["demo_min_real_frames"],policy._motion_cfg["demo_tail_pad"])!=(160,17,"repeat_last"):
        raise ValueError("真实 policy 构造没有逐键透传 motion 配置")
    for es,want in ((17,1),(114,7)):
        previous=policy.mem_buffer; policy.reset()
        if policy.mem_buffer is previous: raise ValueError("reset 没有重建 memory")
        policy.add_buffer({"images":np.stack([protocol.stub_frame(i) for i in range(es+1)])[:,None],
                           "state":np.zeros((es+1,8),np.float32),"exec_start_idx":es})
        if policy.mem_buffer.motion_encode_calls!=want: raise ValueError("真实 policy 补帧窗计数不符")
    print("ONLINE_POLICY_CONFIG=PASS keys=10 budget=160 reset_episodes=2",flush=True)
    return {"keys":sorted(keys),"budget":160,"episodes":[17,114]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate",choices=("basic","lr","init","rope","policy"),default="basic")
    parser.add_argument("--out",type=pathlib.Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError(f"拒绝覆盖检查报告: {args.out}")
    checks={"lr":check_lr,"init":check_init,"rope":check_rope,"policy":check_policy_config}
    names=("lr","rope","policy") if args.gate=="basic" else (args.gate,)
    results={name:checks[name]() for name in names}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x") as f: json.dump(results,f,ensure_ascii=False,indent=2)


if __name__ == "__main__":
    main()
