"""Replay audited native row coordinates only; never read operand payloads."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import struct

ROOT=Path(__file__).resolve().parents[1]
HF=Path('/home/xiang/.cache/huggingface')
CORPUS=ROOT/'research/20261029/ORIGINAL_BF16_FULL_DEPTH_CORPUS_RESULTS_20261029.json'


def need(ok,message):
    if not ok: raise ValueError(message)


def bound_bytes(root,b):
    path=(root/b['path']).resolve();need(path.is_relative_to(root.resolve()),'binding escaped root')
    value=path.read_bytes();need(len(value)==b['bytes'] and hashlib.sha256(value).hexdigest()==b['sha256'],'binding mismatch')
    return value


def row_ids(hf,entry,case):
    meta=json.loads(bound_bytes(hf,entry['metadata']));p=entry['payload'];path=(hf/p['path']).resolve()
    need(path.is_relative_to(hf.resolve()) and path.stat().st_size==p['bytes'],'packet path/size differs')
    # Unbuffered reads prevent read-ahead into neighboring operand payloads.
    with path.open('rb', buffering=0) as f:
        prefix=f.read(8);need(len(prefix)==8,'short header prefix');n=struct.unpack('<Q',prefix)[0]
        need(0<n<=1<<20,'header size');header=json.loads(f.read(n));row=header['rows']
        a,b=row['data_offsets'];need(row['dtype']=='I64' and row['shape']==[256] and b-a==2048,'row layout')
        f.seek(8+n+a);raw=f.read(2048)
    need(len(raw)==2048 and hashlib.sha256(raw).hexdigest()==entry['tensor_metadata']['rows']['raw_tensor_sha256'], 'actual rows SHA mismatch')
    rows=list(struct.unpack('<256q',raw));s,b=entry['step'],entry['block'];ordinal=case['native_source_ordinal']
    expected=sorted(random.Random(42+(ordinal+1)*1000003+1009*s+37*b).sample(range(31668),512)[:256])
    need(rows==expected==meta['selected_rows'] and meta['grid']==[21,29,52] and meta['step']==s and meta['block']==b
        and meta['native_source_ordinal']==ordinal and meta['source_id']==case['source_id']
        and meta['tensor_metadata']['rows']==entry['tensor_metadata']['rows'], 'row/metadata provenance mismatch')
    return rows


def stats(rows):
    spatial=Counter(r%1508 for r in rows);selected=set(rows)
    pairs=sum(n*(n-1)//2 for n in spatial.values())
    adjacent=sum((r+1508) in selected for r in rows if r//1508<20)
    full=sum(n==21 for n in spatial.values())
    tiles={str(m):sum(all(t*m+j in selected for j in range(min(m,31668-t*m))) and min(m,31668-t*m)==m
        for t in {r//m for r in rows}) for m in (16,128)}
    return {'distinct_xy':len(spatial),'same_xy_cross_tau_pairs':pairs,'adjacent_tau_pairs':adjacent,
        'complete_21_tau_xy':full,'full_aligned_tiles':tiles}


def distribution(values):
    values=list(values);return {'min':min(values),'max':max(values),'mean':sum(values)/len(values),'sum':sum(values)}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--hf-root',type=Path,default=HF);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    hf=args.hf_root.resolve();proof=json.loads(CORPUS.read_text());need(proof['status']=='COMPLETE_ASSET_AUDIT_ACTUAL_CPU_EXIT0','audit not complete')
    manifest=json.loads(bound_bytes(hf,proof['manifest']));need(manifest['totals']['packets']==1280,'unexpected audit geometry')
    rows={};records=[]
    for case in manifest['cases']:
        for entry in case['packets']:
            key=(case['native_source_ordinal'],entry['step'],entry['block']);need(key not in rows,'duplicate packet')
            value=row_ids(hf,entry,case);rows[key]=set(value)
            records.append({'source':key[0],'role':case['role'],'step':key[1],'block':key[2],**stats(value)})
    need(len(rows)==1280,'incomplete packet set')
    comparisons={}
    for label,pairs in {
        'same_source_step_different_block':((rows[(o,s,a)],rows[(o,s,b)]) for o in sorted({x[0] for x in rows}) for s in range(4) for a,b in itertools.combinations(range(40),2)),
        'same_source_block_different_step':((rows[(o,a,b)],rows[(o,c,b)]) for o in sorted({x[0] for x in rows}) for b in range(40) for a,c in itertools.combinations(range(4),2)),
        'different_source_same_step_block':((rows[(a,s,b)],rows[(c,s,b)]) for a,c in itertools.combinations(sorted({x[0] for x in rows}),2) for s in range(4) for b in range(40))}.items():
        n=identical=total_overlap=0;lo=256;hi=0
        for x,y in pairs:
            n+=1;identical+=x==y;overlap=len(x&y);total_overlap+=overlap;lo=min(lo,overlap);hi=max(hi,overlap)
        comparisons[label]={'comparisons':n,'identical_row_sets':identical,'intersection_rows_mean':total_overlap/n,'intersection_rows_min':lo,'intersection_rows_max':hi}
    result={'status':'OBS_ROW_COORDINATES_ONLY_NO_MODEL_OR_OPERAND_PAYLOAD','manifest':proof['manifest'],'grid_tau_y_x':[21,29,52],
        'flattening':'native row = tau*(29*52)+y*52+x','sources':8,'source_state':'TEACHER_INPUT','packets':len(records),
        'rows_bytes_read':1280*2048,'packet_operand_payload_bytes_read':0,
        'per_packet_summary':{k:distribution(r[k] for r in records) for k in ('distinct_xy','same_xy_cross_tau_pairs','adjacent_tau_pairs','complete_21_tau_xy')},
        'full_M16_tiles':distribution(r['full_aligned_tiles']['16'] for r in records),'full_M128_tiles':distribution(r['full_aligned_tiles']['128'] for r in records),
        'row_set_comparisons':comparisons,'packets_by_source_step_block':records}
    need(not args.out.exists(),'fresh output required');args.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
