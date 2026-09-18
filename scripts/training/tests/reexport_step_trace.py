"""在独立CPU进程中完整重导出同一次原始XPlane，并绑定输入与输出SHA。"""

import argparse
import hashlib
import json
import os
import pathlib
import sys


def sha256_file(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1<<20),b''): digest.update(chunk)
    return digest.hexdigest()


def write_binding(source,output,event_limit,exporter):
    traces=list(output.rglob('*.trace.json.gz'))
    if len(traces)!=1: raise ValueError('重导出结果不是单一trace')
    record={'schema':1,'source_xplane':str(source.resolve()),
            'source_xplane_sha256':sha256_file(source),'trace_sha256':sha256_file(traces[0]),
            'trace_viewer_event_limit':event_limit,'exporter':exporter}
    with (output/'trace_reexport.json').open('x') as file:
        json.dump(record,file,ensure_ascii=False,indent=2)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=pathlib.Path,required=True)
    parser.add_argument('--out-dir',type=pathlib.Path,required=True)
    args=parser.parse_args()
    sources=list((args.records/'step_trace').rglob('*.xplane.pb'))
    if len(sources)!=1: raise ValueError('原始XPlane缺失或混入多轮')
    if args.out_dir.exists(): raise FileExistsError(args.out_dir)
    os.environ['JAX_PLATFORMS']='cpu'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from jax._src.lib import xla_client
    sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
    from step_timing import complete_trace_export
    options=xla_client.profiler.ProfileOptions()
    options.host_tracer_level=0; options.python_tracer_level=0; options.enable_hlo_proto=False
    session=xla_client.profiler.ProfilerSession(options)
    session.stop()
    raw=sources[0].read_bytes()
    source_sha=hashlib.sha256(raw).hexdigest()
    with complete_trace_export() as limit: session.export(raw,str(args.out_dir))
    record=write_binding(sources[0],args.out_dir,limit,'jaxlib.ProfilerSession.export')
    if record['source_xplane_sha256']!=source_sha: raise ValueError('导出期间原始XPlane发生变化')
    print('TRACE_REEXPORT=PASS '+json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
