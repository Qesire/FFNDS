"""Read-only independent arithmetic audit of the five-source screening."""
from pathlib import Path
from statistics import mean, median, geometric_mean
import argparse, csv, hashlib, json, math

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir',type=Path,default=Path(__file__).resolve().parent.parent)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
BASE=args.source_dir.resolve()
P=json.loads((BASE/'PILOT_RESULTS.json').read_text())
V=json.loads((BASE/'VBENCH_SUMMARY.json').read_text())['scores']
CSV=list(csv.DictReader((BASE/'SOURCE_METRICS.csv').open()))
S=P['sources']; F='00_frozen'; rows={}
checks=[]

def check(a,b,label):
    if not math.isclose(a,b,rel_tol=2e-12,abs_tol=2e-12):
        raise AssertionError((label,a,b))
    checks.append(label)

for profile,data in P['profiles'].items():
    q={x['source_id']:x for x in data['quality']}
    assert set(q)==set(S)
    row={'quality':{},'exact_trace':sum(x['exact_frozen_trace'] for x in q.values())}
    for metric in V:
        delta={s:100*(V[metric][profile]['rows'][s]-V[metric][F]['rows'][s]) for s in S}
        check(mean(V[metric][profile]['rows'].values()),V[metric][profile]['score'],f'{profile}/{metric}/score')
        row['quality'][metric]={'mean_delta_pp':mean(delta.values()),'median_delta_pp':median(delta.values()),'min_delta_pp':min(delta.values()),'max_delta_pp':max(delta.values()),'negative_sources':sum(v<0 for v in delta.values()),'source_delta_pp':delta}
    for key in ('final_latent_relative_l2','rgb_rmse_0to1','temporal_delta_rgb_mae_0to1'):
        row[key]={'mean':mean(x[key] for x in q.values()),'min':min(x[key] for x in q.values()),'max':max(x[key] for x in q.values())}
    row['adjacent_rgb_change_pct']={s:100*(q[s]['mean_adjacent_rgb_mae_0to1']/next(x for x in P['profiles'][F]['quality'] if x['source_id']==s)['mean_adjacent_rgb_mae_0to1']-1) for s in S}
    if data['timing']:
        t=data['timing']; a=t['sources']
        wall=geometric_mean(x['reference_wall_s']/x['candidate_wall_s'] for x in a)
        gpu=geometric_mean(sum(x['reference_steps_ms'])/sum(x['candidate_steps_ms']) for x in a)
        check(wall,t['wall_speedup'],f'{profile}/wall_speedup')
        check(gpu,t['gpu_speedup'],f'{profile}/gpu_speedup')
        cand=mean(x['candidate_wall_s'] for x in a); ref=mean(x['reference_wall_s'] for x in a)
        check(cand,t['candidate_mean_seconds'],f'{profile}/candidate_mean')
        check(ref,t['reference_mean_seconds'],f'{profile}/reference_mean')
        peak_c=max(x['candidate_peak_gib'] for x in a);peak_r=max(x['reference_peak_gib'] for x in a)
        check(peak_c,t['peak_candidate_gib'],f'{profile}/peak_c')
        check(peak_r,t['peak_reference_gib'],f'{profile}/peak_r')
        cs=[mean(x['candidate_steps_ms'][i] for x in a)/1000 for i in range(4)]
        rs=[mean(x['reference_steps_ms'][i] for x in a)/1000 for i in range(4)]
        delta=[c-r for c,r in zip(cs,rs)]
        row['time']={'wall_speedup':wall,'gpu_speedup':gpu,'time_increase_pct':100*(1/wall-1),'candidate_s':cand,'reference_s':ref,'added_wall_s':cand-ref,'wall_speedup_range':[min(x['wall_speedup'] for x in a),max(x['wall_speedup'] for x in a)],'faster_sources':sum(x['wall_speedup']>1 for x in a),'candidate_steps_s':cs,'reference_steps_s':rs,'step_added_s':delta,'gpu_added_s':sum(delta),'non_gpu_window_delta_s':(cand-ref)-sum(delta),'low_anchor_share_of_added_gpu':delta[2]/sum(delta) if sum(delta) else None}
        row['memory']={'candidate_peak_gib':peak_c,'reference_peak_gib':peak_r,'saved_gib':peak_r-peak_c,'reduction_pct':100*(1-peak_c/peak_r)}
    rows[profile]=row
for x in CSV:
    profile=x['profile']; s=x['source_id']
    for metric in V:check(float(x[metric]),V[metric][profile]['rows'][s],f'csv/{profile}/{s}/{metric}')
asset_reuse={}
for kind,single in [('ffn_factor','07_ffn_factor'),('attention_basis','08_attention_joint')]:
    a=P['profiles'][single]['asset_sha256'][kind];b=P['profiles']['10_combined_state']['asset_sha256'][kind]
    asset_reuse[kind]={'same_sites':set(a)==set(b),'same_hashes':a==b,'sites':len(a)}
interaction={}
for metric in V:
    val=lambda name:rows[name]['quality'][metric]['mean_delta_pp']
    interaction[metric]=val('10_combined_state')-val('09_combined_graph')-val('07_ffn_factor')-val('08_attention_joint')
result={'source_head':'5f83e1c590b6cb93b81dbe88a21101ca1b9260a6','source_count':len(S),'profiles':rows,'numeric_consistency_checks_passed':len(checks),'asset_reuse':asset_reuse,'descriptive_nonadditivity_delta_pp':interaction,'inference_scope':'fixed five-source stress/median screening; one timed request per profile per source with matched same-GPU reference; descriptive only, no population confidence claims'}
result['input_sha256']={name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in ('PILOT_RESULTS.json','VBENCH_SUMMARY.json','SOURCE_METRICS.csv')}
out=args.out.resolve();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('arithmetic checks',len(checks),'asset reuse',asset_reuse)
for name,r in rows.items():
    if 'time' not in r:continue
    t=r['time'];m=r['memory'];v=r['quality']
    print(name, f"wall {t['wall_speedup']:.6f}, GPU {t['gpu_speedup']:.6f}, latency {t['time_increase_pct']:+.2f}%, memory {m['saved_gib']:.3f} GiB/{m['reduction_pct']:.2f}%", 'step excess s',','.join(f'{x:+.4f}' for x in t['step_added_s']),f"wall excess {t['added_wall_s']:+.4f}, imaging {v['imaging_quality']['mean_delta_pp']:+.4f} ({v['imaging_quality']['negative_sources']}/5 decline)")
print('nonadditivity mean pp',interaction)
