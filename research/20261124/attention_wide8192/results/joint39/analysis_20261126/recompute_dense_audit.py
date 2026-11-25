"""Independently recompute the frozen C / native Dense evidence, source clustered.

No model, video decoder or GPU is executed. Timing comes from the raw generation
receipt and log timestamps; quality per-source numbers come from the published
summary because the native scoring JSON files are archived by hash only here.
"""
from pathlib import Path
from datetime import datetime
from itertools import product
import argparse,hashlib,json,math
import numpy as np

HERE=Path(__file__).resolve().parent
SRC=HERE.parent/'c_dense_baseline'
HISTORICAL=HERE.parent/'C_CONFIRM_RESULT_196504.json'
HEAD='253d2b72569289469ef191086e81643c90c37cfb'
def read(name): return json.loads((SRC/name).read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def geo(x): return float(np.exp(np.mean(np.log(x))))
def trace(a,b):
    return all(a[k]==b[k] for k in ('initial','flows','latents')) and a['video']['decoded_rgb24_sha256']==b['video']['decoded_rgb24_sha256']

def main():
    identities=json.loads((HERE/'source_identity.json').read_text())
    for item in identities:
        body=((HERE/'source'/item['filename']) if item['filename']=='resident_historical.py' else (SRC/item['filename'])).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()==item['git_blob_sha']
    protocol=read('C_DENSE_PROTOCOL.archive.json')
    rec=read('C_DENSE_RECEIPT_199260.json')
    timestamps=read('C_DENSE_SAVE_TIMESTAMPS_199260.json')
    summary=read('C_DENSE_SUMMARY.json')
    assert sha(SRC/'C_DENSE_RECEIPT_199260.json')==summary['receipt_sha256']
    assert rec['status']=='complete' and rec['job']==summary['job']==protocol['generation_job']
    assert rec['method_sha256']==protocol['method_sha256']
    assert rec['manifest_sha256']==protocol['source_manifest_sha256']
    assert rec['protocol_sha256']==protocol['frozen_protocol_sha256']
    assert summary['quality']['metric_scope_sha256']==protocol['metric_scope_sha256']
    sources=protocol['source_ids']; n=len(sources)
    assert n==6 and len(set(sources))==6 and len(rec['pairs'])==2*n
    index=np.array(list(product(range(n),repeat=n)),dtype=np.int16)
    data={s:{} for s in sources}; original={}; repeats={a:0 for a in ('dense','C')}
    rawpairs=[]
    for pair in rec['pairs']:
        sid,rep=pair['source_id'],pair['repeat']; assert rep in (0,1)
        expected=['dense','C'] if (sources.index(sid)+rep)%2==0 else ['C','dense']
        assert pair['order']==expected
        assert pair['arms']['dense']['record']['initial']==pair['arms']['C']['record']['initial']
        assert all(pair['arms']['dense']['record'][k][0]==pair['arms']['C']['record'][k][0] for k in ('flows','latents'))
        obs={}
        for arm in ('dense','C'):
            row=pair['arms'][arm]['record']
            assert row['experts']==[0,0,1,1]
            assert [row['video'][k] for k in ('width','height','frames','fps')]==[832,464,81,16]
            if arm=='C':
                c=row['counters']; a=pair['arms'][arm]['attention']
                assert (c['variant'],c['tile_rows'],c['anchors'],c['hits'])==('wide8192',8192,80,80)
                assert (a['attention_sites'],a['anchors'],a['hits'],a['native_probe_queries'])==(160,80,80,5120)
            else: assert 'counters' not in row and not pair['arms'][arm]['attention']
            st=timestamps[row['group']+'/'+sid+'.mp4']
            measured=(datetime.fromisoformat(st['end'])-datetime.fromisoformat(st['start'])).total_seconds()
            assert abs(measured-st['seconds'])<1e-12
            item=dict(wall=row['wall'],gpu=sum(row['step_ms'])/1000,save=measured,
                      without_save=row['wall']-measured,peak_gib=row['peak_allocated']/2**30,
                      reserved_gib=row['peak_reserved']/2**30,steps=[t/1000 for t in row['step_ms']])
            obs[arm]=item
            data[sid].setdefault(arm,[]).append(item)
            if rep==0: original[(sid,arm)]=row
            else:
                exact=trace(original[(sid,arm)],row)
                assert exact and pair['repeat_trace_exact'][arm]
                repeats[arm]+=int(exact)
        for metric in ('wall','gpu'):
            assert math.isclose(pair[metric+'_ratio'],obs['C'][metric]/obs['dense'][metric],abs_tol=1e-12)
        rawpairs.append(dict(source=sid,repeat=rep,order=expected,arms=obs,
                             ratios={m:obs['C'][m]/obs['dense'][m] for m in ('wall','gpu','without_save')}))
    assert len({p['cluster_id'] for p in rec['pairs']})==6
    per_source=[]
    for sid in sources:
        armstats={a:{m:float(np.mean([v[m] for v in data[sid][a]])) for m in ('wall','gpu','save','without_save')} for a in ('dense','C')}
        per_source.append(dict(source=sid,arms=armstats,ratios={m:armstats['C'][m]/armstats['dense'][m] for m in ('wall','gpu','without_save')}))
    timing={}
    for m in ('wall','gpu','without_save'):
        ratios=np.array([s['ratios'][m] for s in per_source]); boots=np.exp(np.log(ratios[index]).mean(1))
        timing[m]=dict(ratio=geo(ratios),speedup=1/geo(ratios),reduction_percent=100*(1-geo(ratios)),
                       exact_source_bootstrap_95=np.quantile(boots,[.025,.975]).tolist(),
                       archived_bootstrap_95=summary['comparisons'][m]['bootstrap_95'],
                       dense_mean=float(np.mean([s['arms']['dense'][m] for s in per_source])),
                       C_mean=float(np.mean([s['arms']['C'][m] for s in per_source])),
                       faster_sources=int((ratios<1).sum()),faster_pairs=sum(p['ratios'][m]<1 for p in rawpairs),
                       min_source_ratio=float(ratios.min()),max_source_ratio=float(ratios.max()))
        timing[m]['mean_saved_seconds']=timing[m]['dense_mean']-timing[m]['C_mean']
        assert math.isclose(timing[m]['ratio'],summary['comparisons'][m]['ratio'],rel_tol=1e-12)
        for a,key in (('dense','dense_mean'),('C','C_mean')):
            assert math.isclose(timing[m][key],summary['comparisons'][m][a],rel_tol=1e-12)
    steps={a:np.mean([p['arms'][a]['steps'] for p in rawpairs],axis=0) for a in ('dense','C')}
    savings=steps['dense']-steps['C']
    resource=dict(dense_step_mean_s=steps['dense'].tolist(),C_step_mean_s=steps['C'].tolist(),
                  saved_by_step_s=savings.tolist(),anchor_extra_s=float(-savings[[0,2]].sum()),
                  hit_saved_s=float(savings[[1,3]].sum()),net_gpu_saved_s=float(savings.sum()),
                  mean_save_dense_s=float(np.mean([s['arms']['dense']['save'] for s in per_source])),
                  mean_save_C_s=float(np.mean([s['arms']['C']['save'] for s in per_source])),
                  cold_init_seconds=rec['cold_init_seconds'],setup=rec['setup'])
    resource['non_gpu_extra_C_minus_dense_s']=timing['gpu']['mean_saved_seconds']-timing['wall']['mean_saved_seconds']
    resource['save_extra_C_minus_dense_s']=resource['mean_save_C_s']-resource['mean_save_dense_s']
    resource['remaining_non_gpu_extra_s']=resource['non_gpu_extra_C_minus_dense_s']-resource['save_extra_C_minus_dense_s']
    resource['peaks_in_shared_process']={a:{m:dict(min=min(p['arms'][a][m] for p in rawpairs),max=max(p['arms'][a][m] for p in rawpairs),mean=float(np.mean([p['arms'][a][m] for p in rawpairs]))) for m in ('peak_gib','reserved_gib')} for a in ('dense','C')}
    resource['warmup_allocated_gib']={a:rec['warmup'][a]['record']['peak_allocated']/2**30 for a in ('dense','C')}
    order={}
    for first in ('dense','C'):
        chosen=[p for p in rawpairs if p['order'][0]==first]
        order[first+'_first']={m:dict(pairs=len(chosen),ratio=geo([p['ratios'][m] for p in chosen])) for m in ('wall','gpu','without_save')}
    quality={}
    for name,q in summary['quality']['dimensions'].items():
        d=np.array([q['by_source']['dense'][s] for s in sources]);c=np.array([q['by_source']['C'][s] for s in sources]);diff=c-d
        assert math.isclose(float(d.mean()),q['dense'],abs_tol=1e-12)
        assert math.isclose(float(c.mean()),q['C'],abs_tol=1e-12)
        assert math.isclose(float(diff.mean()),q['delta'],abs_tol=1e-12)
        quality[name]=dict(dense=float(d.mean()),C=float(c.mean()),delta_pp=float(diff.mean()*100),
                           exact_source_bootstrap_95_pp=(100*np.quantile(diff[index].mean(1),[.025,.975])).tolist(),
                           archived_bootstrap_95_pp=[100*x for x in q['bootstrap_95']],
                           by_source_delta_pp={s:float(x*100) for s,x in zip(sources,diff)},
                           worst_source=sources[int(diff.argmin())],worst_delta_pp=float(diff.min()*100),
                           better_sources=int((diff>0).sum()),
                           leave_one_source_out_delta_pp={s:float(np.delete(diff,i).mean()*100) for i,s in enumerate(sources)})
    historic_path=HISTORICAL
    hist={}
    if historic_path.exists():
        old=json.loads(historic_path.read_text());hist['sha256_matches_protocol']=sha(historic_path)==protocol['prior_confirmation_sha256']
        ref={p['source_id']:p['arms']['C']['record'] for p in old['pairs']}
        hist['C_trace_matches']=sum(trace(p['arms']['C']['record'],ref[p['source_id']]) for p in rec['pairs'])
        assert hist['C_trace_matches']==12
    result=dict(evidence_head=HEAD,evidence_identity={'generation_job':rec['job'],'scoring_job':summary['quality']['job'],'source_units':6,'timing_repeats_per_source':2,'formal_pairs':12,'video_dimensions':[832,464,81,16],'seed':42},
                timing=timing,per_source=per_source,pairs=rawpairs,resources=resource,order=order,
                quality=quality,repeat_trace_exact=repeats,historical_replay=hist,
                quality_scope='Recomputed from six-source published scores; scorer code and hashes inspected. Native score JSONs and videos are not in this checkout, so VBench itself was not rerun.',
                bootstrap_scope='Exact enumeration of 6^6 source resamples (46656); source means retain the two timing repeats. These descriptive intervals are not equivalence tests.',
                source_git_blobs_verified=len(identities),
                source_sha256={item['filename']:sha((HERE/'source'/item['filename']) if item['filename']=='resident_historical.py' else (SRC/item['filename'])) for item in identities})
    (HERE/'dense_audit_verified.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('evidence_identity','timing','resources','order','repeat_trace_exact','historical_replay')},ensure_ascii=False,indent=2))
    print(json.dumps({'quality':quality},ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SRC)
    parser.add_argument('--historical',type=Path,default=HISTORICAL)
    args=parser.parse_args()
    SRC,HISTORICAL=args.source,args.historical
    main()
