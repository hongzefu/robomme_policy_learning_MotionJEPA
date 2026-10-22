"""仅验证计划数学与只读metadata；不导入项目，不运行模型，不改输入资产。"""
import argparse
import copy
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

ROOT = Path('/scratch/hongze/robomme_policy_learning_MotionJEPA')
MANIFEST = ROOT/'v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/meta/episode_manifest.json'
OLD_INDEX = ROOT/'v1-store/datasets/16task-mixed8-200-v10e60-pair4-20261020-a/motion/meta/motion_index.json'
RESTORE = ROOT/'v1-store/restore/mixed8-200-vla-latents-20261021-a'
LATENTS = RESTORE/'wan-latents'
CHUNK = 9*16*32*32*4
RULE = 'grid32_plus_last16_demo_v1'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def regular(L, kind, stride):
    if type(L) is not int or L < 0:
        raise ValueError('invalid length')
    return list(range(0, max(0, L-(16 if kind == 'demo' else 32)), stride))

def selected(L, kind):
    a = regular(L, kind, 32)
    if kind == 'demo' and L >= 17:
        a = sorted(set(a + [16*((L-17)//16)]))
    return a

def oracle(L, kind):
    # 独立从旧grid16序列筛常规偶数行，并追加原demo最后行。
    old = regular(L, kind, 16)
    return [s for i,s in enumerate(old) if i%2 == 0 or (kind == 'demo' and i == len(old)-1)]

def record(L, kind, s, seg_start):
    r = min(33, L-s)
    return {'s':s, 'source_chunk':s//16, 'source_byte_offset':s//16*CHUNK,
            'global_start':seg_start+s, 'real_frames':r, 'pad_frames':33-r,
            'pad_source_frame':seg_start+L-1 if r < 33 else None}

def valid_records(L, kind, rows, seg_start=0, schema=2, selection=RULE):
    if type(schema) is not int or schema != 2 or selection != RULE:
        raise ValueError('schema/selection')
    if kind not in ('demo','exec'):
        raise ValueError('segment kind')
    ss=selected(L,kind)
    if len(rows) != len(ss):raise ValueError('count')
    for i,(r,s) in enumerate(zip(rows,ss)):
        for key in ('s','source_chunk','source_byte_offset','global_start','real_frames','pad_frames'):
            if type(r.get(key)) is not int or r[key] < 0:
                raise ValueError('integer field '+key)
        if r.get('pad_source_frame') is not None and type(r['pad_source_frame']) is not int:
            raise ValueError('pad source type')
        expected=record(L,kind,s,seg_start)
        if r != expected:raise ValueError('row identity '+str(i))
        if kind=='exec' and (r['real_frames']!=33 or r['pad_frames']):raise ValueError('exec pad')
    return True

def frames(L, s, seg_start=0):
    n=min(33,L-s)
    return list(range(seg_start+s,seg_start+s+n))+[seg_start+L-1]*(33-n)

def valid_header(L, kind, header, expected_base):
    ss=selected(L,kind)
    if type(header.get('num_grid')) is not int or header['num_grid'] != len(ss):
        raise ValueError('num_grid')
    if header.get('starts') != ss or any(type(s) is not int for s in header['starts']):
        raise ValueError('starts')
    rb=header.get('row_base')
    if ss:
        if type(rb) is not int or rb != expected_base or rb < 0:raise ValueError('row_base')
    elif rb is not None:raise ValueError('empty row_base')
    return True

def require(condition, message):
    if not condition:raise AssertionError(message)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    t0=time.monotonic();results={};inputs={str(p.relative_to(ROOT)):sha(p) for p in (MANIFEST,OLD_INDEX,RESTORE/'restore_receipt.json')}
    require(inputs[str(MANIFEST.relative_to(ROOT))]=='c6a063dfbfc6da404dd8f1d3ddc38e9ba6e25abca81417fe742872401e7f97df','manifest identity')
    require(inputs[str(OLD_INDEX.relative_to(ROOT))]=='d9ec1feed44a7ca3a7a8d7adf5fea5abd2eaf0b6e889558bf2065786d152f67c','index identity')
    m=json.loads(MANIFEST.read_text());idx=json.loads(OLD_INDEX.read_text());rc=json.loads((RESTORE/'restore_receipt.json').read_text())
    require(rc['status']=='PASS','receipt status');require(len(m['episodes'])==len(idx['entries'])==3200,'episodes')

    # 任意L的代数边界另在报告证明；此处枚举8193种长度，覆盖所有模32余数。
    examples={};old_miss=0
    for L in range(8193):
        for kind in ('demo','exec'):
            ss=selected(L,kind);require(ss==oracle(L,kind),('oracle',L,kind))
            require(ss==sorted(set(ss)),('sorted unique',L,kind))
            require(all(s in regular(L,kind,16) for s in ss),('old subset',L,kind))
            if kind=='exec':require(all(s+33<=L for s in ss),('full exec',L))
        ds=selected(L,'demo')
        if L<17:require(not ds,('short demo',L));continue
        tail=16*((L-17)//16);require(tail in ds and 17<=L-tail<=32,('tail',L))
        covered=-1
        for s in ds:
            require(s<=covered+1,('coverage gap',L,s));covered=max(covered,min(L-1,s+32))
            fr=frames(L,s);require(len(fr)==33 and max(fr)<L and min(fr)>=0,('frame boundary',L,s))
            r=min(33,L-s);require(fr[r:]==[L-1]*(33-r),('repeat last',L,s))
        require(covered==L-1,('final coverage',L))
        if L in (17,32,33,34,48,49,64,65,96,97):examples[str(L)]={'starts':ds,'last_real':L-ds[-1],'last_pad':33-(L-ds[-1])}
    results['length_exhaustion']={'lengths':8193,'kinds':2,'cases':16386,'pass':True,'examples':examples}

    totals=Counter();metadata_aggregate=hashlib.sha256();identity_a=hashlib.sha256();identity_b=hashlib.sha256()
    max_visible=0;max_owner=None;events=0;new_base=0;old_base=0;missing_example=None
    for e,ie in zip(m['episodes'],idx['entries']):
        g=int(e['global_episode_idx']);es=int(e['exec_start_idx']);nt=int(e['num_timesteps'])
        require((g,e['h5_file'],e['raw_ep_idx'],nt,es)==(ie['g'],ie['h5_file'],ie['raw_ep_idx'],ie['num_timesteps'],ie['exec_start_idx']),('parent/index identity',g))
        lengths={'demo':es,'exec':nt-es};target_by_kind={}
        for kind in ('demo','exec'):
            L=lengths[kind];origin=0 if kind=='demo' else es;old=regular(L,kind,16);ss=selected(L,kind);target_by_kind[kind]=ss
            require(ie[kind]['num_grid']==len(old) and ie[kind]['seg_len']==L,('old count',g,kind))
            require(ie[kind]['row_base']==(old_base if old else None),('old base',g,kind));old_base+=len(old)
            require(ss==oracle(L,kind),('real oracle',g,kind));totals['old_'+kind]+=len(old);totals['new_'+kind]+=len(ss)
            valid_header(L,kind,{'starts':ss,'num_grid':len(ss),'row_base':new_base if ss else None},new_base)
            if not old:
                totals['empty_segments']+=1
                require(not ss,('empty source',g,kind));continue
            totals['segments']+=1
            task=e['h5_file'][len('record_dataset_'):-len('.h5')];key=f'{task}_ep{e["raw_ep_idx"]}_{kind}'
            mp=LATENTS/f'{key}.metadata.json';raw=mp.read_bytes();meta=json.loads(raw)
            metadata_aggregate.update(key.encode()+b'\0'+hashlib.sha256(raw).digest())
            expected={'schema':3,'g':g,'h5_file':e['h5_file'],'raw_ep_idx':e['raw_ep_idx'],'segment_kind':kind,
                      'seg_start_global':origin,'seg_len':L,'num_grid':len(old),'grid_stride':16,'window_frames':33,
                      'latent_shape':[9,16,32,32],'latent_dtype':'float32','chunk_bytes':CHUNK}
            for k,v in expected.items():require(meta.get(k)==v,('metadata',key,k,meta.get(k),v))
            require(len(meta['rows'])==len(old),('metadata rows',key));require(meta['bytes']==len(old)*CHUNK,('metadata bytes',key))
            bp=LATENTS/f'{key}.bin';require(bp.stat().st_size==len(old)*CHUNK,('bin size',key))
            require((LATENTS/f'{key}.bin.sha256').read_text().split()[0]==meta['sha256'],('sidecar hash',key))
            for j,s0 in enumerate(old):
                row=meta['rows'][j];r=record(L,kind,s0,origin)
                for k,v in {'m':j,'seg_offset':s0,'start_global_frame':r['global_start'],'real_frames':r['real_frames'],'pad_frames':r['pad_frames'],'pad_source_frame':r['pad_source_frame'],'input_shape':[33,256,256,3],'input_dtype':'uint8'}.items():
                    require(row.get(k)==v,('source row',key,j,k))
                digest=row.get('input_frames_sha256','');require(len(digest)==64 and all(c in '0123456789abcdef' for c in digest),('source frame digest',key,j))
                totals['source_rows_checked']+=1
            if kind=='demo':
                regular32=regular(L,kind,32);gap=max(0,L-(regular32[-1]+33))
                totals['demo_segments']+=1;totals['additional_tail']+=len(ss)-len(regular32)
                totals['strict_grid32_padded']+=sum(s+33>L for s in regular32)
                totals['strict_grid32_missing_tail_segments']+=bool(gap)
                totals['strict_grid32_exact_end_no_pad']+=not gap and min(33,L-regular32[-1])==33
                if gap and missing_example is None:missing_example={'g':g,'L':L,'strict_last_start':regular32[-1],'gap':gap,'retained_tail':ss[-1]}
            for j,s0 in enumerate(ss):
                row=meta['rows'][s0//16];r=record(L,kind,s0,origin)
                require(row['seg_offset']==s0 and row['pad_frames']==r['pad_frames'] and row['pad_source_frame']==r['pad_source_frame'],('selection mapping',key,s0))
                item=[g,kind,s0,r['global_start'],s0//16,r['real_frames'],r['pad_frames'],r['pad_source_frame'],row['input_frames_sha256'],new_base+j]
                payload=(json.dumps(item,separators=(',',':'))+'\n').encode();identity_a.update(payload);identity_b.update(payload)
                totals['new_padded']+=bool(r['pad_frames'])
            new_base+=len(ss)
        ds=target_by_kind['demo'];xs=target_by_kind['exec'];n=len(ds)+len(xs)
        if n>max_visible:max_visible=n;max_owner=g
        require(n<=80,('capacity',g,n))
        check_t={es,nt-1}
        for u in xs:check_t.update((es+u+31,es+u+32,es+u+33))
        for t in sorted(v for v in check_t if es<=v<nt):
            visible=[origin for origin in xs if origin+32<=t-es]
            count=max(0,min(len(xs),(t-es-32)//32+1))
            require(len(visible)==count,('exec causal',g,t))
            require(all(es+u+32<=t for u in visible),('future read',g,t))
            require(all(s+min(33,es-s)-1<=es-1<=t for s in ds),('demo causal',g,t))
            k=len(ds)+len(visible);require(k<=80,('capacity at t',g,t))
            for q in (8,9):
                mask=[True]*(k*q)+[False]*((80-k)*q)
                require(len(mask)==80*q and sum(mask)==k*q,('group mask',q,k))
            events+=1
    require(totals['new_demo']==24791 and totals['new_exec']==41679 and new_base==66470,'new totals')
    require(totals['old_demo']==46731 and totals['old_exec']==81611 and old_base==128342,'old totals')
    require(totals['new_padded']==1800 and totals['additional_tail']==1051,'tail totals')
    require(totals['segments']==4995 and max_visible==62,'segments/capacity')
    require(identity_a.hexdigest()==identity_b.hexdigest(),'A/B windows')
    results['full_metadata']={'pass':True,**dict(totals),'rows':new_base,'max_visible':max_visible,'max_owner':max_owner,
                              'causal_boundary_events':events,'selected_identity_sha256':identity_a.hexdigest(),
                              'source_metadata_aggregate_sha256':metadata_aggregate.hexdigest(),'missing_tail_example':missing_example,
                              'latent_payloads_rehashed':False,'source_bin_size_only':True}

    rejected=[]
    def expect_reject(name,fn):
        try:fn()
        except (ValueError,AssertionError,IndexError):rejected.append(name)
        else:raise AssertionError('mutation survived: '+name)
    base=[record(65,'demo',s,0) for s in selected(65,'demo')]
    for name,rows in [('missing_tail',base[:-1]),('duplicate_tail',base+[base[-1]]),('reverse_order',list(reversed(base))),
                      ('extra_interior16',base[:1]+[record(65,'demo',16,0)]+base[1:])]:
        expect_reject(name,lambda rows=rows:valid_records(65,'demo',rows))
    for field,wrong in [('s',64),('source_chunk',4),('source_byte_offset',2*CHUNK),('global_start',64),
                        ('real_frames',33),('pad_frames',0),('pad_source_frame',65)]:
        rows=copy.deepcopy(base);rows[-1][field]=wrong
        expect_reject('wrong_'+field,lambda rows=rows:valid_records(65,'demo',rows))
    expect_reject('schema1_new_layout',lambda:valid_records(65,'demo',base,schema=1))
    expect_reject('wrong_selection_kind',lambda:valid_records(65,'demo',base,selection='grid32_only'))
    expect_reject('short_demo_force_window',lambda:valid_records(16,'demo',[record(16,'demo',0,0)]))
    ex=[record(40,'exec',0,238),record(40,'exec',32,238)]
    expect_reject('exec_tail_padding',lambda:valid_records(40,'exec',ex,seg_start=238))
    expect_reject('target_ordinal_times32',lambda:valid_records(65,'demo',[record(65,'demo',32*j,0) for j in range(3)]))
    expect_reject('global_frame_mod32',lambda:valid_records(100,'exec',[record(100,'exec',u,238) for u in selected(100,'exec') if (238+u)%32==0],seg_start=238))
    expect_reject('demo_nominal_endpoint_visibility',lambda:require(len([s for s in selected(33,'demo') if s+32<=33])==len(selected(33,'demo')),'demo wrongly delayed'))
    expect_reject('repeat_exec_frame',lambda:require(frames(65,48)==list(range(48,65))+[65]*16,'cross segment'))
    expect_reject('repeat_penultimate_demo',lambda:require(frames(65,48)==list(range(48,65))+[63]*16,'wrong repeat source'))
    expect_reject('mask_real_tail_window',lambda:require([False]*9==[True]*9,'real window masked'))
    expect_reject('mask_padding_latent_slots',lambda:require([True]*5+[False]*4==[True]*9,'partial slots masked'))
    expect_reject('capacity81_truncate',lambda:require(81<=80,'capacity exceed'))
    old_local=regular(65,'demo',16);global_base=1
    wrong=[u for j,u in enumerate(old_local) if (global_base+j)%2==0]
    expect_reject('global_table_even_rows',lambda:require(wrong==selected(65,'demo'),'global rows'))
    expect_reject('sequential_source_rows',lambda:require(list(range(3))==[s//16 for s in selected(65,'demo')],'wrong source sequence'))
    for field,bad in [('s',False),('s',0.0),('s','0'),('source_chunk',False),('source_byte_offset',0.0),
                      ('global_start',-1),('real_frames',33.0),('pad_frames',False)]:
        rows=copy.deepcopy(base);rows[0][field]=bad
        expect_reject('type_'+field+'_'+type(bad).__name__,lambda rows=rows:valid_records(65,'demo',rows))
    expect_reject('empty_segment_base_zero',lambda:valid_header(16,'demo',{'starts':[],'num_grid':0,'row_base':0},0))
    expect_reject('nonempty_segment_base_null',lambda:valid_header(17,'demo',{'starts':[0],'num_grid':1,'row_base':None},0))
    expect_reject('nonempty_segment_base_bool',lambda:valid_header(17,'demo',{'starts':[0],'num_grid':1,'row_base':False},0))
    expect_reject('starts_bool_zero',lambda:valid_header(17,'demo',{'starts':[False],'num_grid':1,'row_base':0},0))
    expect_reject('num_grid_bool_one',lambda:valid_header(17,'demo',{'starts':[0],'num_grid':True,'row_base':0},0))
    results['mutations']={'pass':True,'rejected':len(rejected),'names':rejected,'scope':'独立规格探针拒绝，非生产解析器已实现'}
    # 带非方形轴的展平索引，避免32×32掩盖转置；此处只验证索引规格。
    C,H,W=3,5,7;arr=[10000*c+100*h+w for c in range(C) for h in range(H) for w in range(W)]
    for c in range(C):
        for h in range(H):
            for w in range(W):require(arr[(c*H+h)*W+w]==10000*c+100*h+w,'CHW flatten')
    results['dimensions']={'pass':True,'flatten_non_square_cases':C*H*W,'a_row_bytes':8*512*4,'b_row_bytes':CHUNK,
      'a_table_bytes':new_base*8*512*4,'b_table_bytes':new_base*CHUNK,'window_budget':80,'a_memory':80*8+512,'b_memory':80*9+512,
      'wan_projection_params':16384*512+512,'memory_total_a':89974272,'memory_total_b':98363392,
      'b_batch128_fp32_bytes':128*80*9*16384*4,'b_per_gpu4_fp32_bytes':128*80*9*16384*4//4,
      'samples_per_arm':128*60000,'steps_per_arm':60000}
    require(all(sha(ROOT/k)==v for k,v in inputs.items()),'inputs changed during probe')
    out={'scope':'计划规格与只读metadata；不导入生产代码、不做模型/训练/VAE数值验证',
         'inputs':inputs,'checks':results,'elapsed_seconds':round(time.monotonic()-t0,3),'status':'PASS'}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('PLAN_CONTRACT_PROBE=PASS length_cases=16386 source_segments=4995 source_rows=128342 selected_rows=66470 padded=1800 max_visible=62 mutations='+str(len(rejected))+' output='+str(path))

if __name__=='__main__':main()
