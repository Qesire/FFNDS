"""Verify immutable calibration payloads with bounded I/O read-ahead on the GPU host."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import platform
import time


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections',type=Path,nargs=2,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--start-block',type=int,default=0)
    args=ap.parse_args()
    if platform.node().split('.')[0]!='worker':raise RuntimeError('Read-ahead must warm the actual GPU host file cache')
    if args.start_block not in range(40):ap.error('start-block must be0..39')
    entries=[];bindings=[]
    for path in args.collections:
        report=json.loads(path.read_text())
        if report['status']!='PASS_CALIBRATION24_SHARD_FULL160_OPERANDS_AND_NOOP':raise RuntimeError('Need both complete24-input shards')
        bindings.append({'path':str(path.resolve()),'sha256':sha(path)})
        for entry in report['completed']:
            if sha(entry['input_report'])!=entry['input_report_sha256']:raise RuntimeError('Input report changed')
            case=json.loads(Path(entry['input_report']).read_text())
            entries.extend({**scope,'index':case['index']} for scope in case['scopes'])
    if len(entries)!=3840 or len({e['path'] for e in entries})!=3840:raise RuntimeError('Need3840 distinct original operands')
    entries.sort(key=lambda e:((e['block']-args.start_block)%40,e['step'],e['index']))
    def verify(entry):
        if Path(entry['path']).stat().st_size!=entry['bytes'] or sha(entry['path'])!=entry['sha256']:
            raise RuntimeError('Original calibration payload changed')
        return entry['bytes']
    started=time.perf_counter();total=0
    # Four I/O workers, no tensor reconstruction or GPU work. Every original
    # file remains in its HF cache path; root disk and learned values unchanged.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(as_completed([pool.submit(verify,e) for e in entries]),1):
            total+=result.result()
            if i%192==0:print(f'VERIFIED_AND_READ_AHEAD {i}/3840 payloads',flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    report={'status':'PASS_CALIBRATION24_ALL_PAYLOAD_SHA_AND_HOST_READ_AHEAD','collections':bindings,
        'files':len(entries),'bytes':total,'source_sha256':sha(Path(__file__)),'host':platform.node(),
        'IO_workers':4,'GPU_work':False,'extra_disk_copy_bytes':0,'elapsed_seconds':time.perf_counter()-started,
        'quality_or_speed_acceptance':False}
    temp=args.out/'RUN.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.out/'RUN.json')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
