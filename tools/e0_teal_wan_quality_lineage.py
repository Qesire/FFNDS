"""Shared strict lineage for future TEAL Wan video quality scopes; no model execution."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_teal_quality_video_parent',ROOT/'tools/e0_teal_wan_all160_video_v3.py')
video=importlib.util.module_from_spec(spec);spec.loader.exec_module(video)
native=video.native
PARENT='config/e0_teal_wan_all160_video_v3_part{}_20261029_pending_review_v1.json'
ORDINALS=(1040,979,456,923);MODES=('dense','threshold25','threshold50')

def need(v,msg):
    if not v:raise ValueError(msg)

def parent_data(hf,part):
    need(type(part) is int and part in range(4),'Invalid shard')
    path=ROOT/PARENT.format(part);plan=native.read(path);out=Path(hf)/plan['output_HF_relative']
    return {'protocol':native.binding(path,ROOT),'RUN':native.binding(out/'RUN.json',hf),
            'exit':native.binding(out.parent/'EXIT_RECEIPT.json',hf)}

def closed_rows(proof,hf,part):
    hf=Path(hf).resolve();need(proof['protocol']['path']==PARENT.format(part),'Wrong native parent protocol')
    p=native.read(native.bound(ROOT,proof['protocol']));r=native.read(native.bound(hf,proof['RUN']))
    closed=video.verify_closed(SimpleNamespace(hf_root=hf,protocol=ROOT/proof['protocol']['path'],
         run=hf/proof['RUN']['path'],exit=hf/proof['exit']['path']))
    need(closed['status']=='PASS_CLOSED_TEAL_VIDEO_V3_QUALITY_DIAGNOSTIC' and p['part']==part
      and p['case']['native_source_ordinal']==ORDINALS[part] and len(r['phases'])==5
      and [x['run_id'] for x in r['phases']]==p['run_ids'] and [x['mode'] for x in r['phases'][2:]]==list(MODES)
      and r['same_hardware_old_new_all10_and_video_neutral'] is True,'Actual closed parent phases differ')
    case=p['case'];rows=[];dense=r['phases'][2]
    for x in r['phases'][2:]:
        need(all(video.vm.same_state(x['states'][name],dense['states'][name]) for name in ('raw_noise_FP32','initial_noise')), 'Same-source actual noise differs')
        gen=native.read(native.bound(hf,x['generation_receipt']))
        need(x['source_id']==case['source_id'] and x['image_sha256']==case['image']['sha256']
          and x['prompt_sha256']==case['prompt_sha256'] and x['seed']==42 and x['all160_FFN_covered'] is True
          and gen['run_id']==x['run_id'] and gen['mode']==x['mode'] and gen['video']==x['video']
          and gen['states']==x['states'] and len(x['states'])==10,'Source/generation mismatch')
        path=Path(x['video']['path']).resolve();need(path.is_relative_to(hf),'Video escaped HF')
        binding={'path':str(path.relative_to(hf)),'bytes':x['video']['bytes'],'sha256':x['video']['sha256']}
        native.bound(hf,binding)
        rows.append({'ordinal':ORDINALS[part],'mode':x['mode'],'case':case,'generation':x['generation_receipt'],'video':binding,'states':x['states']})
    paths=[(hf/x['video']['path']).resolve() for x in rows]
    need(len(rows)==len(set(paths))==len({(p.stat().st_dev,p.stat().st_ino) for p in paths})==3,
      'Distinct real clips required')
    return rows

def source_equal_vectors(source_rows,metric_keys):
    """Pure reduction only; caller must first independently verify four closed runs."""
    import math
    rows=list(source_rows);need(len(rows)==4 and {r['ordinal'] for r in rows}==set(ORDINALS),'Four distinct sources required')
    need(all(set(r['scores'])==set(MODES) and all(set(r['scores'][mode])==set(metric_keys) for mode in MODES) for r in rows),'Metric vectors differ')
    need(all(math.isfinite(float(r['scores'][mode][k])) for r in rows for mode in MODES for k in metric_keys),'Nonfinite metric')
    return {'source_ordinals':list(ORDINALS),'per_source':[next(r for r in rows if r['ordinal']==o) for o in ORDINALS],
      'source_equal_means':{mode:{k:sum(float(r['scores'][mode][k]) for r in rows)/4 for k in metric_keys} for mode in MODES},
      'source_equal_candidate_minus_dense':{mode:{k:sum(float(r['scores'][mode][k])-float(r['scores']['dense'][k]) for r in rows)/4 for k in metric_keys} for mode in MODES[1:]},
      'independent_sources':4,'seeds_per_source':1,'official_five_independent_samples':False}


def raft_source_equal(source_rows):
    """Summarize four already-verified source observations, preserving zero baselines."""
    rows=list(source_rows);need(len(rows)==4 and {r['ordinal'] for r in rows}==set(ORDINALS),'Four distinct source observations required')
    need(all(set(r['means'])==set(MODES) for r in rows),'RAFT arm coverage differs')
    result=source_equal_vectors([{'ordinal':r['ordinal'],'scores':{m:{'top5_px_per_2frame_pair':r['means'][m]} for m in MODES}} for r in rows],['top5_px_per_2frame_pair'])
    ratios={}
    for mode in MODES[1:]:
        vals=[r['means'][mode]/r['means']['dense'] if r['means']['dense']>0 else None for r in rows]
        ratios[mode]=sum(vals)/4 if all(x is not None for x in vals) else None
    result['mean_source_ratios']=ratios
    result['scope']='full-frame top5percent flow; camera/background/water motion may dominate; not subject motion'
    return result
