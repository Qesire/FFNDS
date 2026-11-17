"""Offline timing provenance audit. Does not run or schedule any videos."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def bound(spec):
    path=Path(spec['path']).resolve()
    if sha(path)!=spec['sha256']: raise ValueError('changed timing evidence')
    return json.loads(path.read_text(encoding='utf8'))


def analyze(recipe):
    rows=[];seen=set()
    for entry in recipe['receipts']:
        receipt=bound(entry['receipt']); key=(receipt['source'],receipt['arm'])
        if key in seen: raise ValueError('duplicate timing cohort slot')
        seen.add(key)
        for name in ('wall_seconds','ffn_stream_elapsed_ms','ffn_host_call_seconds','peak_gpu_reserved','peak_gpu_allocated','peak_host_rss_bytes'):
            n=receipt[name]
            if not isinstance(n,(int,float)) or isinstance(n,bool) or not math.isfinite(n) or n<0:
                raise ValueError('invalid timing/resource value: '+name)
        if receipt['peak_gpu_allocated']>receipt['peak_gpu_reserved']: raise ValueError('invalid GPU peak values')
        if len(receipt['dispatch'])!=4 or len(receipt['sites'])!=160: raise ValueError('incomplete trajectory receipt')
        protocol=entry.get('protocol')
        if protocol is not None:
            protocol=bound(protocol)
            if (protocol.get('receipt_sha256')!=entry['receipt']['sha256'] or not isinstance(protocol.get('protocol_id'),str)
                    or protocol.get('observer_instrumented') not in (True,False)
                    or not isinstance(protocol.get('comparison_contract'),dict)
                    or set(protocol['comparison_contract']) != {'hardware','torch','cuda','schedule_sha256','decode_sha256','measurement'}):
                raise ValueError('timing protocol receipt mismatch')
        rows.append({'source':key[0],'arm':key[1],'receipt_sha256':entry['receipt']['sha256'],
                     'wall_seconds':receipt['wall_seconds'], 'ffn_stream_elapsed_ms':receipt['ffn_stream_elapsed_ms'],
                     'ffn_host_call_seconds':receipt['ffn_host_call_seconds'],
                     'peak_gpu_reserved':receipt['peak_gpu_reserved'], 'peak_gpu_allocated':receipt['peak_gpu_allocated'],
                     'peak_host_rss_bytes':receipt['peak_host_rss_bytes'],
                     'host_rss_scope':'process_lifetime_high_water_mark',
                     'ffn_stream_semantics':'CUDA event interval includes stream idle between recorded events; not kernel-only',
                     'timing_protocol':None if protocol is None else protocol['protocol_id'],
                     'observer_instrumented':None if protocol is None else protocol['observer_instrumented'],
                     'comparison_contract':None if protocol is None else protocol['comparison_contract'],
                     'wall_timing_scope':receipt.get('wall_timing_scope','UNSPECIFIED'),
                     'receipt_plain_uninstrumented':receipt.get('timing_is_plain_uninstrumented') is True})
    # Absence of explicit matching plain protocol is an evidence gap, not equality.
    comparisons=[]
    for row in rows:
        if row['arm']=='Dense': continue
        dense=next((x for x in rows if x['source']==row['source'] and x['arm']=='Dense'),None)
        eligible=(dense is not None and row['timing_protocol'] is not None
                  and row['timing_protocol']==dense['timing_protocol'] and row['comparison_contract']==dense['comparison_contract']
                  and row['observer_instrumented'] is False and dense['observer_instrumented'] is False
                  and row['receipt_plain_uninstrumented'] and dense['receipt_plain_uninstrumented']
                  and dense['wall_seconds']>0)
        comparisons.append({'source':row['source'],'arm':row['arm'], 'paired_plain_protocol_verified':eligible,
                            'wall_ratio_to_dense':row['wall_seconds']/dense['wall_seconds'] if eligible else None})
    return {'status':'AUDIT_ONLY', 'attempts_added':0, 'rows':rows, 'paired_comparisons':comparisons}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('recipe');parser.add_argument('recipe_sha256');parser.add_argument('output');a=parser.parse_args()
    result=analyze(bound({'path':a.recipe,'sha256':a.recipe_sha256}))
    with Path(a.output).open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
