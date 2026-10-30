#!/usr/bin/env python3
"""Aggregate only actual closed input-video quality metadata; never decode video."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
HF = Path('/home/xiang/.cache/huggingface')
KINDS = ('quality', 'raft', 'visual')

def read(path):
    b = path.read_bytes()
    if len(b) > 16 * 1024 * 1024:
        raise ValueError('metadata size limit')
    return json.loads(b), {'bytes':len(b), 'sha256':hashlib.sha256(b).hexdigest()}

def reference(path):
    obj, ident = read(path)
    return obj, {'path':str(path.relative_to(HF)), **ident}

def verify_ref(ref):
    obj, actual = reference(HF / ref['path'])
    if actual != ref:
        raise ValueError('closed receipt binding differs')
    return obj

def verify_exit(exit, runref, protocol_sha, scope):
    if not (exit.get('id') == scope and exit.get('process_exited') is True and exit.get('child_returncode') == 0 and exit.get('timeout') is False and exit.get('automatic_retry') is False and exit.get('run_sha256') == runref['sha256'] and exit.get('protocol_sha256') == protocol_sha):
        raise ValueError('actual EXIT identity or status differs')

def scope(kind, part):
    version = '_v2' if kind != 'visual' or part >= 2 else ''
    path = ROOT / f'config/e0_chipmunk_input_video_{kind}{version}_20261030_part{part}_pending_review_v1.json'
    p, pid = read(path)
    private = 'chipmunk_input_video_quality_v2_20261030' if kind != 'visual' or part >= 2 else 'chipmunk_input_video_quality_20261030'
    proofpath = ROOT / f'.runtime/{private}/{kind}/part{part}/CLOSED_VERIFICATION_V1.json'
    proof, proofid = read(proofpath)
    expected_status = 'PASS_ACTUAL_EXIT_CHIPMUNK_INPUT_VIDEO_QUALITY_CLOSED'
    if proof.get('status') != expected_status or proof.get('parent_id') != p['id']:
        raise ValueError('actual independent closure required')
    output = HF / p['output_HF_relative']
    run, rr = reference(output / 'RUN.json')
    ex, er = reference(output.parent / 'EXIT_RECEIPT.json')
    verify_exit(ex,rr,pid['sha256'],p['id'])
    if run.get('id') != p['id'] or run.get('protocol_sha256') != pid['sha256'] or run.get('job_spec_sha256') != ex.get('spec_sha256') or run.get('automatic_retry') is not False:
        raise ValueError('producer RUN identity differs')
    vr, ve = verify_ref(proof['verification_RUN']), verify_ref(proof['verification_exit'])
    verify_exit(ve,proof['verification_RUN'],pid['sha256'],p['id']+'_VERIFY')
    if vr.get('status') != 'PASS_VERIFICATION_EXIT_PENDING' or vr.get('job_spec_sha256') != ve.get('spec_sha256') or vr.get('automatic_retry') is not False or vr.get('predebited_bytes') != vr.get('completed_bytes') or vr.get('inflight') is not None:
        raise ValueError('verifier incomplete')
    validated = vr['result']
    if validated.get('RUN') != rr or validated.get('exit') != er:
        raise ValueError('verifier producer binding differs')
    public={'scope_id':p['id'],'protocol':{'path':str(path.relative_to(ROOT)),**pid},'closure_proof_sha256':proofid['sha256'],'RUN':rr,'EXIT':er,'VERIFY_RUN':proof['verification_RUN'],'VERIFY_EXIT':proof['verification_exit'],'verify_hash_bytes':vr['completed_bytes']}
    return p,run,public

def historical_failures():
    failures=[]
    for i in range(2):
        p,_=read(ROOT / f'config/e0_chipmunk_input_video_quality_20261030_part{i}_pending_review_v1.json')
        output=HF / p['output_HF_relative']
        run,rr=reference(output/'RUN.json'); ex,er=reference(output.parent/'EXIT_RECEIPT.json')
        if run.get('status') != 'FAILED_GPU_MONITOR_PRESERVED_NO_RETRY' or run.get('error') != 'Assigned GPU monitor failed: TimeoutExpired' or ex.get('child_returncode') == 0 or ex.get('run_sha256') != rr['sha256']:
            raise ValueError('V1 failure identity differs')
        failures.append({'scope_id':p['id'],'RUN':rr,'EXIT':er,'status':run['status'],'error':run['error'],'invoked':run['invoked']})
    return failures

def historical_visual_failures():
    failures=[]
    for i in (2,3):
        p,pid=read(ROOT/f'config/e0_chipmunk_input_video_visual_20261030_part{i}_pending_review_v1.json')
        out=HF/p['output_HF_relative']; ex,er=reference(out.parent/'EXIT_RECEIPT.json')
        if not (ex.get('id')==p['id'] and ex.get('protocol_sha256')==pid['sha256'] and ex.get('process_exited') is True and ex.get('child_returncode')==1 and ex.get('run_sha256') is None and ex.get('automatic_retry') is False and not (out/'RUN.json').exists()):
            raise ValueError('old visual taskset failure identity differs')
        failures.append({'scope_id':p['id'],'EXIT':er,'RUN':'ABSENT','context':'taskset failed before worker startup; zero decoded frames'})
    return failures

def main():
    # Read all twelve proofs before making any output file.
    all_scopes={(kind,i):scope(kind,i) for kind in KINDS for i in range(4)}
    rows=[]
    for i in range(4):
        q=all_scopes['quality',i][1]
        r=all_scopes['raft',i][1]
        sources=q['source_summary']['per_source']
        if len(sources)!=1: raise ValueError('expected one source')
        src=sources[0]; scores=src['scores']; modes=tuple(scores)
        if len(modes)!=3 or 'none' not in modes or len(next(iter(scores.values())))!=8: raise ValueError('eight-axis or arm coverage')
        obs=r['RAFT_observations']
        if len(obs)!=3 or {x['mode'] for x in obs}!=set(modes) or any(x['ordinal']!=src['ordinal'] for x in obs): raise ValueError('RAFT coverage')
        raft={x['mode']:x['summary'] for x in obs}
        for d in scores.values():
            if any(not math.isfinite(float(v)) for v in d.values()): raise ValueError('nonfinite score')
        rows.append({'ordinal':src['ordinal'],'scores':scores,'candidate_minus_dense':src['candidate_minus_same_fresh_Dense'],'continuous_raft':raft})
    if len({x['ordinal'] for x in rows})!=4: raise ValueError('source coverage')
    modes=tuple(rows[0]['scores']); axes=tuple(rows[0]['scores']['none'])
    if any(set(x['scores'])!=set(modes) or any(set(x['scores'][m])!=set(axes) for m in modes) for x in rows): raise ValueError('inconsistent score axes')
    means={axis:{m:statistics.fmean(x['scores'][m][axis] for x in rows) for m in modes} for axis in axes}
    diffs={axis:{m:statistics.fmean(x['scores'][m][axis]-x['scores']['none'][axis] for x in rows) for m in modes if m!='none'} for axis in axes}
    key='mean_top5percent_flow_px_per_2frame_pair'
    for x in rows:
        for m in modes:
            val=x['continuous_raft'][m][key]
            if not math.isfinite(val): raise ValueError('nonfinite RAFT')
        dense=x['continuous_raft']['none'][key]
        x['raft_candidate_minus_dense']={m:x['continuous_raft'][m][key]-dense for m in modes if m!='none'}
        x['raft_candidate_over_dense']={m:x['continuous_raft'][m][key]/dense if dense>0 else None for m in modes if m!='none'}
    data={'scope':'Four exposed sources, one seed each; three paired videos per source, no extra independent N from reused video or neutral controls.','sources':rows,'source_equal_eight_axes':means,'source_equal_paired_differences':diffs,'source_equal_raft_means':{m:statistics.fmean(x['continuous_raft'][m][key] for x in rows) for m in modes},'source_equal_raft_paired_differences':{m:statistics.fmean(x['raft_candidate_minus_dense'][m] for x in rows) for m in modes if m!='none'},'source_equal_mean_of_source_ratios':{m:(statistics.fmean(x['raft_candidate_over_dense'][m] for x in rows) if all(x['raft_candidate_over_dense'][m] is not None for x in rows) else None) for m in modes if m!='none'},'scopes':[all_scopes[k,i][2] for k in KINDS for i in range(4)],'raft_semantics':'40 stride-2 pairs per video, 20 iterations each; five eight-pair windows. Continuous top-5% flow is distinct from official binary dynamic degree.','visual_review':'Independent AI visual review to be linked only when actually completed.','v1_monitor_failures':historical_failures(),'v1_visual_taskset_failures':historical_visual_failures(),'quality_acceptance':False}
    target=ROOT/'research/20261030/CHIPMUNK_INPUT_VIDEO_QUALITY_RESULTS_20261030.json'
    target.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    lines=['# Input video: actual quality results','',data['scope'],'','| Source | Arm | Continuous RAFT mean |','|---|---|---:|']
    for x in rows:
        for m in modes: lines.append(f"| {x['ordinal']} | {m} | {x['continuous_raft'][m][key]:.6f} |")
    lines += ['',data['raft_semantics'],'','Eight per-source axes, paired differences, five-window observations, source-equal summaries and all twelve actual closure bindings are in the JSON. Independent AI visual review remains separate.','']
    target.with_suffix('.md').write_text('\n'.join(lines))
    print(target)

if __name__=='__main__': main()
