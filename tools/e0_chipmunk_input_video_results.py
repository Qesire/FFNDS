#!/usr/bin/env python3
"""Summarize four independently closed input-video runs using small JSON only."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
HF = Path('/home/xiang/.cache/huggingface')
MODES = ('none', 'h_cache_dense', 'x_cache_dense', 'high_h_readonly_p64', 'high_x_recompute_p64')

def load(path):
    data = path.read_bytes()
    if len(data) > 4 * 1024 * 1024:
        raise ValueError(f'not small metadata: {path}')
    return json.loads(data), {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def bound(ref):
    path = HF / ref['path']
    obj, identity = load(path)
    if identity != {k: ref[k] for k in ('bytes', 'sha256')}:
        raise ValueError('receipt identity mismatch')
    return obj

def receipt(path):
    obj, identity = load(path)
    return obj, {'path': str(path.relative_to(HF)), **identity}

def exit_ok(e, run_ref, protocol_sha):
    if not (e.get('process_exited') and e.get('child_returncode') == 0 and not e.get('timeout') and e.get('run_sha256') == run_ref['sha256'] and e.get('protocol_sha256') == protocol_sha):
        raise ValueError('actual EXIT does not close RUN and protocol')

def one(i):
    cfgpath = ROOT / f'config/e0_chipmunk_input_video_20261030_part{i}_pending_review_v1.json'
    cfg, cfgid = load(cfgpath)
    proof, proofid = load(ROOT / f'.runtime/chipmunk_input_video_20261030/part{i}/CLOSED_VERIFICATION_FINAL.json')
    if proof.get('status') != 'PASS_ACTUAL_EXIT_CLOSED_CHIPMUNK_INPUT_VIDEO' or proof.get('id') != cfg['id']:
        raise ValueError('parent not closed')
    runpath = HF / cfg['output_HF_relative'] / 'RUN.json'
    run, rr = receipt(runpath)
    ex, er = receipt(runpath.parent.parent / 'EXIT_RECEIPT.json')
    exit_ok(ex, rr, cfgid['sha256'])
    if run['id'] != cfg['id'] or run['protocol_sha256'] != cfgid['sha256'] or run['status'] != 'COMPLETE_CHIPMUNK_INPUT_VIDEO_EXIT_PENDING':
        raise ValueError('invalid producer RUN')
    vr, ve = bound(proof['verification_RUN']), bound(proof['verification_exit'])
    exit_ok(ve, proof['verification_RUN'], cfgid['sha256'])
    if vr.get('status') != 'PASS_CLOSED_CHIPMUNK_INPUT_VIDEO' or vr.get('artifact_count') != 57 or vr.get('RUN') != rr or vr.get('exit') != er:
        raise ValueError('invalid independent verification')
    cr, ce = bound(run['CPU_preflight']['RUN']), bound(run['CPU_preflight']['exit'])
    exit_ok(ce, run['CPU_preflight']['RUN'], cfgid['sha256'])
    if [p['mode'] for p in run['phases']] != list(MODES) or any(p['actual_full_FFN_sites'] != 160 for p in run['phases']):
        raise ValueError('phase coverage')
    if not run['cache_dense_matches_current_none'] or not all(c['equal'] for c in run['prefix_byte_controls']) or not all(run['neutral_video_checks'].values()):
        raise ValueError('byte gates')
    samples = run['cache_boundary_memory']
    if len(samples) != 160 or len(run['mask_observations']) != 80:
        raise ValueError('observer coverage')
    phases=[]
    for p in run['phases']:
        q = {k:v for k,v in p.items() if k not in ('states','video','native_transfer_observations')}
        selected = [s for s in samples if s['mode'] == p['mode']]
        if p['mode'] != 'none':
            if len(selected) != 40 or len({s['block'] for s in selected}) != 40:
                raise ValueError('cache boundary coverage')
            q['cache_boundary_samples'] = selected
            for key in ('allocated_bytes','reserved_bytes'):
                m=max(selected,key=lambda s:s['memory'][key]); q['sampled_max_live_'+key]={'value':m['memory'][key], 'block':m['block'], 'cache_bytes':m['cache_bytes']}
        phases.append(q)
    return {'source_ordinal':run['source_ordinal'],'seed':run['seed'],'scope_id':run['id'], 'protocol':{'path':str(cfgpath.relative_to(ROOT)),**cfgid}, 'closure_proof_sha256':proofid['sha256'], 'CPU':run['CPU_preflight'], 'GPU':{'RUN':rr,'exit':er}, 'VERIFY':{'RUN':proof['verification_RUN'],'exit':proof['verification_exit'],'artifact_count':vr['artifact_count'],'predebited_bytes':vr['predebited_bytes'],'completed_bytes':vr['completed_bytes']}, 'phases':phases,'prefix_byte_controls':run['prefix_byte_controls'],'neutral_video_checks':run['neutral_video_checks'],'mask_observations':run['mask_observations'],'operation_ledger':run['operation_ledger'],'memory_observations':run['memory_observations'],'peak_GPU_allocated_bytes':run['peak_GPU_allocated_bytes'],'peak_GPU_reserved_bytes':run['peak_GPU_reserved_bytes'],'peak_RSS_bytes':run['peak_RSS_bytes'],'wall_seconds':run['wall_seconds'],'raw_model_setup_seconds':run['raw_model_setup_seconds']}

def main():
    rows=[one(i) for i in range(4)]
    means={mode:statistics.mean(next(p['FFN_boundary_CUDA_seconds'] for p in r['phases'] if p['mode']==mode) for r in rows) for mode in MODES}
    result={'status':'FOUR_ACTUAL_CLOSED_METADATA_SUMMARY','sources':rows,'source_equal_FFN_seconds':means,'limitations':['One seed per source; no quality acceptance is implied.','Cache-boundary live maxima are discrete observations, not continuous arm peaks; CUDA allocator peaks are cumulative and never reset.','Requested transfer bytes are not measured physical bus bytes.','Prior captures and other-scope timings may use different hardware or ordering and are not a fair system comparison.']}
    out=ROOT/'research/20261030/CHIPMUNK_INPUT_VIDEO_NATIVE_RESULTS_20261030.json'
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    md=['# Input-state native video: actual closed results','', 'Four sources, seed 42, five phases each. This is native execution evidence; quality assessment is separate.','', '| Source | NONE FFN s | H candidate FFN s | X candidate FFN s | H live allocated GiB | X live allocated GiB |','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        p={v['mode']:v for v in r['phases']}; md.append(f"| {r['source_ordinal']} | {p['none']['FFN_boundary_CUDA_seconds']:.4f} | {p['high_h_readonly_p64']['FFN_boundary_CUDA_seconds']:.4f} | {p['high_x_recompute_p64']['FFN_boundary_CUDA_seconds']:.4f} | {p['high_h_readonly_p64']['sampled_max_live_allocated_bytes']['value']/2**30:.3f} | {p['high_x_recompute_p64']['sampled_max_live_allocated_bytes']['value']/2**30:.3f} |")
    md += ['', 'Source-equal FFN seconds: '+', '.join(f'{k} {v:.4f}' for k,v in means.items())+'.', '', 'Each phase covers 160 FFN sites; candidate phases use 40 sparse sites and X additionally recomputes old W1 at 40 sites. Dense-cache byte controls, per-arm mask identities, transfer requests, actual CPU/GPU/VERIFY receipts and detailed timings are in the JSON.', '', *result['limitations'], '']
    (out.with_suffix('.md')).write_text('\n'.join(md))
    print(out)

if __name__ == '__main__': main()
