"""Independent CPU audit of the real Wan mass-centered low OUT-D receipt."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, geometric_mean

import numpy as np


def payload(path):
    b = path.read_bytes()
    return gzip.decompress(b) if path.suffix == '.gz' else b


def read(path):
    return json.loads(payload(path))


def geometry(g, background=False):
    g = np.asarray(g, dtype=np.float64)
    r, t = float(g[0, 0]), float(g[1, 1])
    d2 = r + t - 2 * float(g[0, 1])
    rd = float(g[0, 1]) - r
    bg_cross = 0.
    if background:
        r += float(g[2, 2] + 2 * g[0, 2])
        t += float(g[2, 2] + 2 * g[1, 2])
        bg_cross = float(2 * (g[1, 2] - g[0, 2]))
        rd += bg_cross / 2
    denominator = math.sqrt(max(r * d2, 0))
    assert math.isclose(2 * rd + d2, t - r, rel_tol=1e-10, abs_tol=1e-9)
    return dict(raw_energy=r, centered_energy=t, ratio=t/r, energy_delta=t-r,
                update_energy=d2, reference_dot_update=rd,
                update_cosine=rd/denominator if denominator else None,
                benefit_cosine_threshold=-0.5*math.sqrt(d2/r),
                sign_margin=-(t-r)/(2*denominator) if denominator else None,
                background_cross_delta=bg_cross)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--historical-quality', type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('mass_numeric_verified.json'))
    args = parser.parse_args()
    src = args.source
    rawpath = src/'MASS_CENTERED_201500.json.gz'
    data, archived = read(rawpath), read(src/'MASS_CENTERED_SUMMARY.json')
    assert data['status'] == 'complete'
    assert data['job'] == '201500'
    assert data['plan_commit'] == 'ff8fd9dcbed3812b1da7f589f3d2e817f48c26c9'
    assert hashlib.sha256(payload(rawpath)).hexdigest() == '979e68efc18c1f999e0370c7698e435880190598aa79add6d23567d7eea4ff20'
    assert hashlib.sha256((src/'attention_mass_centered_low.py').read_bytes()).hexdigest() == data['source_sha256']
    assert set(data['fixed']) == {'vbi_0392', 'vbi_1013'}
    assert data['scope'] == {'anchor_step': 2, 'hit_step': 3, 'audit_layers': [0,13,26,39],
        'frozen_high': True, 'sources': ['vbi_0392','vbi_1013'], 'timing_repeats':2, 'threshold':1e-12, 'probes':64}

    checked = []
    numeric_compared = []
    def gram(g, width, name):
        a = np.asarray(g, dtype=np.float64)
        assert a.shape == (width,width) and np.isfinite(a).all()
        assert np.array_equal(a, a.T), name
        eigen = np.linalg.eigvalsh(a)
        assert eigen[0] >= -1e-10*max(float(a.diagonal().max()),1e-10), name
        checked.append(dict(name=name,width=width,min_eigenvalue=float(eigen[0]),
            max_eigenvalue=float(eigen[-1]),asymmetry=float(np.max(np.abs(a-a.T)))))
        return a

    def compare(a,b,name=''):
        if isinstance(b,dict):
            assert a.keys()==b.keys(), name
            for k in b:compare(a[k],b[k],name+'/'+k)
        elif isinstance(b,list):
            assert len(a)==len(b),name
            for i,(x,y) in enumerate(zip(a,b,strict=True)):compare(x,y,name+'/'+str(i))
        elif isinstance(b,(int,float)) and not isinstance(b,bool):
            assert math.isclose(a,b,rel_tol=2e-12,abs_tol=1e-10),(name,a,b)
            numeric_compared.append(dict(name=name,error=abs(a-b)))
        else:assert a==b,(name,a,b)

    hist = read(args.historical_quality) if args.historical_quality else None
    hist_c = {row['source']:row['record'] for row in hist['renders'] if row['arm']=='C'} if hist else None
    traces = {}
    def check_trace(record,name,sid):
        keys = ['initial','flows','latents','video','experts']
        assert record['source']==sid
        assert record['video']['frames']==81 and record['video']['width']==832 and record['video']['height']==464
        if hist_c:
            matches={key:record[key]==hist_c[sid][key] for key in keys}
            assert all(matches.values()),(name,matches)
            traces[name]=matches
        assert record['counters']['tile_rows']==8192
        assert record['counters']['anchors']==record['counters']['hits']==80

    fixed, derived, background_flips = {}, {}, []
    for sid,track in data['fixed'].items():
        check_trace(track['track']['record'],sid+'/fixed',sid)
        assert list(map(int,track['sites'])) == [0,13,26,39]
        fixed[sid], derived[sid] = {}, {}
        for layer,row in track['sites'].items():
            label=sid+'/'+layer
            tail=row['tail']
            assert row['identical_final_attention_probes'] and tail['shared_probe_ffn_input_exact']
            assert tail['reused_live_ffn_probe_teacher'] and tail['dense_native_ffn_extra_held_rows']==448
            assert tail['axes']==[448,5120] and tail['shared_probe_vs_live_relative_l2']==0
            bg=gram(row['held_base_error_gram'],2,label+'/base')
            stage={k:gram(g,2,label+'/'+k) for k,g in tail['stage_gram_raw_transport'].items()}
            assert len(stage)==10
            backgrounds={}
            for k,g in tail['stage_gram_raw_transport_background'].items():
                a=gram(g,3,label+'/'+k+'/background')
                assert np.allclose(a[:2,:2],stage[k],rtol=1e-10,atol=1e-12)
                backgrounds[k]=geometry(a,True)
                if (a[1,1]-a[0,0])*backgrounds[k]['energy_delta']<0:
                    background_flips.append(dict(source=sid,layer=layer,stage=k,
                        fixed_wide_delta=float(a[1,1]-a[0,0]),**backgrounds[k]))
            st={k:geometry(g) for k,g in stage.items()}
            basegeo=geometry(bg)
            d=backgrounds['block_exit']
            raww=stage['block_exit'][0,0]
            f=dict(anchor_raw_relative_l2=row['anchor_raw_relative_l2'],anchor_relative_l2=row['anchor_relative_l2'],
                inactive_anchor_channels=row['inactive_anchor_channels'],inactive_current_channels=row['inactive_current_channels'],
                negative_rho_rows=row['negative_rho_rows'],rho_min=row['rho_min'],
                probe_to_live_relative_l2=tail['shared_probe_vs_live_relative_l2'],
                replay_to_live_relative_l2=tail['live_relative_l2'],
                difference_identity_relative_l2=row['difference_identity_relative_l2'])
            for key,g in [('base',bg),('attention',stage['attention_head']),('wide_exit',stage['block_exit'])]:
                f[key]=float(g[1,1]/g[0,0]);f[key+'_raw_energy']=float(g[0,0])
            f.update(dense_exit=d['ratio'],dense_delta=d['energy_delta'],dense_cross_delta=d['background_cross_delta'])
            fixed[sid][layer]=f
            att=st['attention_head']
            derived[sid][layer]=dict(base=basegeo,stages=st,backgrounds=backgrounds,
                fit_over_base_energy={arm:float(stage['attention_head'][i,i]/bg[i,i]) for i,arm in enumerate(['raw','centered'])},
                fitted_update_energy_over_base_update=att['update_energy']/basegeo['update_energy'],
                finite_tail_relative_squared_gain=st['block_exit']['ratio']/att['ratio'])

    timing={'repeat_trace_matches':{},'sources':{}}
    pair_details=[]
    for sid in data['fixed']:
        pairs=[p for p in data['pairs'] if p['source']==sid]
        assert [p['repeat'] for p in pairs]==[0,1]
        assert pairs[0]['order']==pairs[1]['order'][::-1]
        timing['repeat_trace_matches'][sid]={}
        timing['sources'][sid]={}
        for pair in pairs:
            assert pair['prefix_exact']
            raw,centered=(pair['arms'][k]['record'] for k in ['raw','centered'])
            assert raw['initial']==centered['initial']
            assert all(raw[k][:3]==centered[k][:3] for k in ['flows','latents'])
            assert raw['flows'][3]!=centered['flows'][3] and raw['latents'][3]!=centered['latents'][3]
            check_trace(raw,sid+'/pair'+str(pair['repeat']),sid)
            pair_details.append(dict(source=sid,repeat=pair['repeat'],order=pair['order'],
                exact_initial_and_first_three_states=True,
                cycle_delta_s=(sum(centered['step_ms'][2:])-sum(raw['step_ms'][2:]))/1000,
                wall_delta_s=centered['wall']-raw['wall'],
                gpu_delta_s=(sum(centered['step_ms'])-sum(raw['step_ms']))/1000))
        for arm in ['raw','centered']:
            records=[p['arms'][arm]['record'] for p in pairs]
            repeat_match=all(records[0][k]==records[1][k] for k in ['initial','latents','flows'])
            repeat_match=repeat_match and records[0]['video']['decoded_rgb24_sha256']==records[1]['video']['decoded_rgb24_sha256']
            assert repeat_match
            timing['repeat_trace_matches'][sid][arm]=repeat_match
            step=[mean(float(row['step_ms'][i])/1000 for row in records) for i in range(4)]
            timing['sources'][sid][arm]=dict(anchor=step[2],hit=step[3],cycle=sum(step[2:]),
                gpu=sum(step),wall=mean(row['wall'] for row in records),
                peak_gib=max(row['peak_allocated']/2**30 for row in records))
    for key in ['anchor','hit','cycle','gpu','wall']:
        timing[key]={arm:mean(v[arm][key] for v in timing['sources'].values()) for arm in ['raw','centered']}
        timing[key]['ratio']=geometric_mean(v['centered'][key]/v['raw'][key] for v in timing['sources'].values())
    comparable=dict(job=data['job'],plan_commit=data['plan_commit'],
        receipt_sha256=hashlib.sha256(payload(rawpath)).hexdigest(),fixed=fixed,timing=timing)
    reference={k:v for k,v in archived.items() if k!='motion'}
    compare(comparable,reference)
    for arm,w in data['warmup'].items():
        if arm=='raw':check_trace(w['record'],'warmup/raw','vbi_0392')

    allrows=[(sid,layer,fixed[sid][layer],derived[sid][layer]) for sid in fixed for layer in fixed[sid]]
    ag=dict(source_clusters=2,nested_sites=8,gram_count=len(checked),
        stage_grams=80,base_grams=8,background_grams=24,
        minimum_gram_eigenvalue=min(x['min_eigenvalue'] for x in checked),
        summary_numeric_fields_compared=len(numeric_compared),
        summary_max_absolute_error=max(x['error'] for x in numeric_compared),
        all_attention_improved=all(f['attention']<1 for _,_,f,_ in allrows),
        improving_wide_exit_sites=sum(f['wide_exit']<1 for _,_,f,_ in allrows),
        improving_dense_exit_sites=sum(f['dense_exit']<1 for _,_,f,_ in allrows),
        all_anchor_reconstruction_improved=all(f['anchor_relative_l2']<f['anchor_raw_relative_l2'] for _,_,f,_ in allrows),
        max_actual_raw_replay_live_relative_l2=max(f['replay_to_live_relative_l2'] for _,_,f,_ in allrows),
        max_probe_live_relative_l2=max(f['probe_to_live_relative_l2'] for _,_,f,_ in allrows),
        max_update_identity_relative_l2=max(f['difference_identity_relative_l2'] for _,_,f,_ in allrows),
        total_inactive_anchor_and_current_channels=sum(f['inactive_anchor_channels']+f['inactive_current_channels'] for _,_,f,_ in allrows),
        minimum_rho=min(f['rho_min'] for _,_,f,_ in allrows),
        negative_rho_rows_total=sum(f['negative_rho_rows'] for _,_,f,_ in allrows),
        fitted_update_energy_over_base_update_range=[
            min(d['fitted_update_energy_over_base_update'] for _,_,_,d in allrows),
            max(d['fitted_update_energy_over_base_update'] for _,_,_,d in allrows)],
        minimum_cycle_saving_s=min(-p['cycle_delta_s'] for p in pair_details),
        maximum_cycle_saving_s=max(-p['cycle_delta_s'] for p in pair_details),
        wall_delta_range_s=[min(p['wall_delta_s'] for p in pair_details),max(p['wall_delta_s'] for p in pair_details)])
    result=dict(status='REAL_RECEIPT_RECOMPUTED',evidence_commit='ec8a39b8a6c8c6a6cb25badfe11e8b6a38b09f63',
        scope='CPU arithmetic on archived actual Wan outputs; no new model execution or video decode',
        receipt_sha256=comparable['receipt_sha256'],source_sha256=data['source_sha256'],
        aggregates=ag,recomputed_summary_without_video_motion=comparable,
        video_motion_status='Archived proxy values not recomputed; original video bytes were not supplied to this audit',
        source_trace_matches=traces,historical_quality_verified=bool(hist_c),
        pair_details=pair_details,derived=derived,background_sign_flips=background_flips,gram_checks=checked,
        fitted_update_ratio_scope='H448 difference energy after recovery and final BF16 writeback divided by H448 FP32 base difference energy; recovery also depends on P64 labels. Not an isolated real-arithmetic recovery norm.',
        actual_reference_semantics={
            'raw':'Frozen C on its own native-high/OUT-ABS/native-low/OUT-D trajectory',
            'centered':'Current measured C block input, same original low features/P64/calibration/regularizer, recoded centered historical slot',
            'native':'All-key H448 attention, same actual P64 teacher; observed through shared fitted wide8192 tail',
            'dense':'Same current C block input, all-key H448 attention plus native H448 FFN',
            'live_check':'raw replay versus actual live C block; LowTail.live_arm is raw',
            'previous_201276':'Different base (selected + groups), different fixed A trajectory and different fitted features; effect sizes are not paired method comparisons'})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(ag,indent=2))


if __name__=='__main__':
    main()
