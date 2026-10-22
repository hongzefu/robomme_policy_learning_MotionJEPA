"""只读核对1800个旧demo尾窗的实际latent字节，不运行VAE/encoder。"""
import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT=Path('/scratch/hongze/robomme_policy_learning_MotionJEPA')
PARENT=ROOT/'v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f/meta/episode_manifest.json'
LATENTS=ROOT/'v1-store/restore/mixed8-200-vla-latents-20261021-a/wan-latents'
REFERENCES=ROOT/'v1-store/datasets/16task-mixed8-200-v10e60-pair4-20261020-a/motion-tokens'
CHUNK=9*16*32*32*4

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    t0=time.monotonic();raw=PARENT.read_bytes();m=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()=='c6a063dfbfc6da404dd8f1d3ddc38e9ba6e25abca81417fe742872401e7f97df'
    n=0;odd=0;total=0;dig=hashlib.sha256();refs=hashlib.sha256()
    for ep in m['episodes']:
        L=int(ep['exec_start_idx'])
        if L<17:continue
        task=ep['h5_file'][len('record_dataset_'):-3];key=f'{task}_ep{ep["raw_ep_idx"]}_demo'
        metadata=json.loads((LATENTS/f'{key}.metadata.json').read_text())
        rb=(REFERENCES/f'{key}.metadata.json').read_bytes();reference=json.loads(rb)
        refs.update(key.encode()+b'\0'+hashlib.sha256(rb).digest())
        s=16*((L-17)//16);source_chunk=s//16
        assert 17<=L-s<=32 and source_chunk==metadata['num_grid']-1
        row=metadata['rows'][source_chunk]
        assert row['seg_offset']==s and row['real_frames']==L-s and row['pad_frames']==33-(L-s)
        assert row['pad_source_frame']==L-1
        assert reference['input_latent_sha256']==metadata['sha256']
        assert Path(reference['input_latent_bin']).resolve()==(LATENTS/f'{key}.bin').resolve()
        assert reference['num_grid']==metadata['num_grid']==len(reference['input_block_sha256'])
        with (LATENTS/f'{key}.bin').open('rb') as f:
            f.seek(source_chunk*CHUNK);buf=f.read(CHUNK)
        assert len(buf)==CHUNK
        actual=hashlib.sha256(buf).hexdigest()
        assert actual==reference['input_block_sha256'][source_chunk],(key,source_chunk)
        dig.update((key+':'+str(s)+':'+actual+'\n').encode())
        n+=1;odd+=s%32!=0;total+=len(buf)
    assert n==1800 and odd==1051
    out={'status':'PASS','scope':'实际旧demo尾窗latent与既有encoder输入块SHA核对；未重算RGB/VAE数值',
         'windows':n,'additional_off_grid32_tails':odd,'bytes_read':total,
         'tail_payload_identity_sha256':dig.hexdigest(),'reference_metadata_aggregate_sha256':refs.hexdigest(),
         'full_source_bins_rehashed':False,'elapsed_seconds':round(time.monotonic()-t0,3)}
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('TAIL_PAYLOAD=PASS windows=1800 additional=1051 bytes_read='+str(total)+' output='+str(p))

if __name__=='__main__':main()
