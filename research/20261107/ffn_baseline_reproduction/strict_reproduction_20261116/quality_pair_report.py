"""Paired source/cluster summaries for custom seed42 VBenchI2V measurements.

Input cells are real per-video metric receipts. Missing/invalid cells remain
missing. The motion ratio floor is frozen in the input before scores exist.
"""
import json
import math
from pathlib import Path

ARMS=('Dense','Q1r2','ICC','Chipmunk Author Safe','Global-P64')
DIMS=('subject_consistency','background_consistency','motion_smoothness','dynamic_degree',
      'aesthetic_quality','imaging_quality','i2v_subject','i2v_background','camera_motion','temporal_flickering')
DEV=('vbi_1013','vbi_1040','vbi_0886','vbi_0947','vbi_1109','vbi_1084','vbi_0950','vbi_0176','vbi_0800','vbi_0392','vbi_0056')


def finite_number(v):
    return type(v) in (int,float) and math.isfinite(v)


def aggregate(spec, cells):
    if tuple(spec['sources']) != DEV or tuple(spec['arms']) != ARMS:
        raise ValueError('fixed DEV11 and five-arm identity required')
    floor=spec['motion_dense_floor']
    if not finite_number(floor) or floor<0: raise ValueError('predeclared motion denominator floor required')
    groups=spec['source_clusters']
    if set(groups)!=set(DEV) or any(not isinstance(v,str) or not v for v in groups.values()):
        raise ValueError('every source needs a verified cluster ID')
    per_video={}
    for row in cells:
        key=(row['source'],row['arm'],row['dimension'])
        if key in per_video or key[0] not in DEV or key[1] not in ARMS or key[2] not in DIMS:
            raise ValueError('duplicate or outside fixed comparison')
        if row['status'] not in ('COMPLETE','MISSING','NOT_APPLICABLE','FAILED'):
            raise ValueError('invalid result status')
        if row['status']=='COMPLETE':
            if row['dimension']=='camera_motion' and spec.get('explicit_camera_instruction_by_source',{}).get(row['source']) != row.get('camera_instruction'):
                raise ValueError('camera result lacks fixed source applicability')
            if row['dimension']=='camera_motion' and row.get('camera_instruction') is None: raise ValueError('camera instruction absent')
            if row.get('source_cluster_id') != groups[row['source']]: raise ValueError('metric source cluster differs from frozen mapping')
            if not finite_number(row['value']): raise ValueError('invalid per-video score')
            if not row.get('scope_sha256') or not row.get('video_sha256') or not row.get('receipt_sha256'):
                raise ValueError('complete score must bind scope/video/receipt')
            if row['dimension']=='dynamic_degree':
                if not finite_number(row.get('continuous_motion')) or row['continuous_motion']<0:
                    raise ValueError('continuous RAFT motion required')
                windows=row.get('motion_windows')
                if not isinstance(windows,list) or len(windows)!=5 or any(not finite_number(x) or x<0 for x in windows):
                    raise ValueError('five continuous motion windows required')
        elif row.get('value') is not None:
            raise ValueError('missing/inapplicable/failed cannot carry an observed score')
        per_video[key]=row
    pairs=[]
    for dimension in DIMS:
        for arm in ARMS[1:]:
            for source in DEV:
                d=per_video.get((source,'Dense',dimension)); x=per_video.get((source,arm,dimension))
                default_status='NOT_APPLICABLE' if dimension=='camera_motion' and source not in spec.get('explicit_camera_instruction_by_source',{}) else 'MISSING'
                pair={'source':source,'cluster':groups[source],'arm':arm,'dimension':dimension,
                      'dense_status':d['status'] if d else default_status,
                      'arm_status':x['status'] if x else default_status, 'dense':None,'value':None,'delta':None}
                if d and x and d['status']=='COMPLETE' and x['status']=='COMPLETE':
                    if dimension=='camera_motion' and (not d.get('camera_instruction') or d.get('camera_instruction')!=x.get('camera_instruction')):
                        raise ValueError('camera instruction differs or absent')
                    if not d.get('metric_batch_identity') or d['metric_batch_identity']!=x.get('metric_batch_identity'):
                        raise ValueError('paired videos were not scored in the same fixed metric batch')
                    if not d.get('metric_protocol_sha256') or d['metric_protocol_sha256']!=x.get('metric_protocol_sha256'):
                        raise ValueError('metric protocol differs within pair')
                    if d.get('input_identity')!=x.get('input_identity') or not d.get('input_identity'):
                        raise ValueError('pair input identity mismatch or absent')
                    if d.get('camera_stratum') != x.get('camera_stratum'): raise ValueError('camera stratum differs within pair')
                    pair.update(dense=d['value'],value=x['value'],delta=x['value']-d['value'],
                                camera_stratum=d.get('camera_stratum','CAMERA_UNSPECIFIED'))
                    if dimension=='dynamic_degree':
                        base=d['continuous_motion']; val=x['continuous_motion']
                        pair.update(dense_motion=base,motion=val,motion_delta=val-base,
                                    motion_ratio=val/base if base>floor else None,
                                    dense_motion_windows=d['motion_windows'],motion_windows=x['motion_windows'])
                pairs.append(pair)
    summaries=[]
    for dim in DIMS:
        for arm in ARMS[1:]:
            valid=[p for p in pairs if p['dimension']==dim and p['arm']==arm and p['delta'] is not None]
            by_cluster={}
            for p in valid: by_cluster.setdefault(p['cluster'],[]).append(p['delta'])
            cluster_means={k:math.fsum(v)/len(v) for k,v in by_cluster.items()}
            vals=list(cluster_means.values()); mean=math.fsum(vals)/len(vals) if vals else None
            loo={k:math.fsum(v for name,v in cluster_means.items() if name!=k)/(len(vals)-1)
                 for k in cluster_means} if len(vals)>1 else {}
            strata={}
            if dim=='temporal_flickering':
                for category in ('CAMERA_STATIC_EXPLICIT','CAMERA_DYNAMIC_INSTRUCTION','CAMERA_UNSPECIFIED'):
                    subset=[p for p in valid if p['camera_stratum']==category]; cluster={}
                    for p in subset:cluster.setdefault(p['cluster'],[]).append(p['delta'])
                    means=[math.fsum(v)/len(v) for v in cluster.values()]
                    strata[category]={'sources':len(subset),'clusters':len(means),
                                      'cluster_equal_mean_delta':math.fsum(means)/len(means) if means else None}
            all_pairs=[p for p in pairs if p['dimension']==dim and p['arm']==arm]
            not_applicable=sum('NOT_APPLICABLE' in (p['dense_status'],p['arm_status']) for p in all_pairs)
            failed=sum('FAILED' in (p['dense_status'],p['arm_status']) for p in all_pairs)
            missing=sum('MISSING' in (p['dense_status'],p['arm_status']) for p in all_pairs)
            summaries.append({'dimension':dim,'arm':arm,'valid_source_count':len(valid),
                              'valid_cluster_count':len(vals),'cluster_equal_mean_delta':mean,
                              'cluster_mean_range':[min(vals),max(vals)] if vals else None,
                              'source_delta_range':[min(p['delta'] for p in valid),max(p['delta'] for p in valid)] if valid else None,
                              'leave_one_cluster_out':loo,'missing_source_count':missing,
                              'not_applicable_source_count':not_applicable,'failed_source_count':failed,
                              'camera_strata':strata if dim=='temporal_flickering' else None})
    return {'status':'PAIRED_SUMMARY_FROM_AVAILABLE_CELLS','source_count':len(DEV),
            'distinct_source_clusters':len(set(groups.values())),'motion_dense_floor':floor,
            'source_pairs':pairs,'summaries':summaries,
            'interpretation':'exposed DEV development comparison; no predeclared noninferiority margin',
            'units':'raw per-video metric differences; continuous motion in pixels per two-frame pair'}

def load_verified_runs(runs):
    """Verify worker result file and successful EXIT receipt, then attach identity."""
    import hashlib
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        return h.hexdigest()
    cells=[]
    for run in runs:
        result_path,exit_path=Path(run['per_video_path']),Path(run['exit_path'])
        if sha(result_path)!=run['per_video_sha256'] or sha(exit_path)!=run['exit_sha256']:
            raise ValueError('metric output or EXIT identity mismatch')
        receipt=json.loads(exit_path.read_text(encoding='utf8'))
        if receipt['exit_code']!=0 or receipt['scope_sha256']!=run['scope_sha256']:
            raise ValueError('metric run did not exit successfully under bound scope')
        if receipt['result']['status']!='COMPLETE' or receipt['result']['dimension'] not in DIMS:
            raise ValueError('metric receipt is not a complete supported dimension')
        if receipt['result']['per_video_sha256']!=run['per_video_sha256']:
            raise ValueError('EXIT receipt does not bind per-video scores')
        batch=json.loads(result_path.read_text(encoding='utf8'))
        if not isinstance(batch,list) or len(batch)!=receipt['result']['video_count']:
            raise ValueError('metric batch length differs from receipt')
        for row in batch:
            if row['scope_sha256']!=run['scope_sha256'] or row['dimension']!=receipt['result']['dimension']:
                raise ValueError('metric row identity differs from receipt')
            cells.append({**row,'receipt_sha256':run['exit_sha256']})
    return cells
if __name__=='__main__':
    import argparse
    import hashlib
    p=argparse.ArgumentParser();p.add_argument('spec');p.add_argument('spec_sha256');p.add_argument('runs');p.add_argument('output');a=p.parse_args()
    if hashlib.sha256(Path(a.spec).read_bytes()).hexdigest()!=a.spec_sha256: raise ValueError('frozen summary spec identity mismatch')
    spec=json.loads(Path(a.spec).read_text(encoding='utf8'));runs=json.loads(Path(a.runs).read_text(encoding='utf8'));cells=load_verified_runs(runs)
    result=aggregate(spec,cells)
    with Path(a.output).open('x',encoding='utf8') as f: json.dump(result,f,indent=2,allow_nan=False);f.write('\n')







