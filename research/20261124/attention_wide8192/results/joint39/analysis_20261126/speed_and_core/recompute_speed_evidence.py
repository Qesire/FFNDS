"""Recompute measured speedups and their denominators, without model execution."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, geometric_mean


def read(path):
    data = path.read_bytes()
    if path.suffix == '.gz':
        data = gzip.decompress(data)
    return json.loads(data)


def speed(reference, candidate, ratio=None):
    r = candidate / reference if ratio is None else ratio
    return dict(reference_seconds=reference, candidate_seconds=candidate,
                candidate_over_reference=r, percent_time_saved=100 * (1-r),
                speedup_reference_over_candidate=1/r,
                arithmetic_mean_seconds_saved=reference-candidate)


def aggregate_paired(receipt, ref, candidate, source_key):
    by_source = {}
    pair_detail = []
    def metrics(record):
        steps = [x/1000 for x in record['step_ms']]
        assert len(steps) == 4 and record['experts'] == [0,0,1,1]
        return dict(wall=record['wall'], gpu=sum(steps),
                    step0=steps[0],step1=steps[1],step2=steps[2],step3=steps[3],
                    anchors=steps[0]+steps[2],hits=steps[1]+steps[3],
                    low_cycle=steps[2]+steps[3],outside_gpu_events=record['wall']-sum(steps))
    for sid in sorted({p[source_key] for p in receipt['pairs']}):
        pairs = [p for p in receipt['pairs'] if p[source_key] == sid]
        by_source[sid] = {}
        assert len(pairs) == 2 and pairs[0]['order'] == pairs[1]['order'][::-1]
        for arm in (ref,candidate):
            rows=[metrics(p['arms'][arm]['record']) for p in pairs]
            by_source[sid][arm]={k:mean(r[k] for r in rows) for k in rows[0]}
            for pair in pairs:
                record=pair['arms'][arm]['record']
                assert len(record['flows']) == len(record['latents']) == 4
                if 'counters' in record:
                    counts=record['counters']
                    assert counts['sites']==160 and counts['anchors']==counts['hits']==80
                    assert counts['tile_rows']==8192 and counts['variant']=='wide8192'
        for p in pairs:
            a,b=(metrics(p['arms'][arm]['record']) for arm in (ref,candidate))
            pair_detail.append(dict(source=sid,repeat=p['repeat'],order=p['order'],
                difference_candidate_minus_reference={k:b[k]-a[k] for k in a}))
    result={}
    for k in by_source[next(iter(by_source))][ref]:
        a=mean(s[ref][k] for s in by_source.values())
        b=mean(s[candidate][k] for s in by_source.values())
        ratio=geometric_mean(s[candidate][k]/s[ref][k] for s in by_source.values())
        result[k]=speed(a,b,ratio)
    return dict(job=receipt['job'],source_clusters=len(by_source),pairs=len(pair_detail),
                aggregation='Within-source arithmetic mean of two repeats, then source-equal geometric mean of ratios; absolute seconds are arithmetic means.',
                sources=by_source,aggregate=result,pairs_detail=pair_detail)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dense-c',type=Path,required=True)
    p.add_argument('--mass',type=Path,required=True)
    p.add_argument('--ffn-only',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path(__file__).with_name('SPEED_EVIDENCE_VERIFIED.json'))
    args=p.parse_args()
    d,m,f=(read(getattr(args,k)) for k in ('dense_c','mass','ffn_only'))
    assert d['job']=='199260' and m['job']=='201500' and f['job']=='185220'
    dc=aggregate_paired(d,'dense','C','source_id')
    mc=aggregate_paired(m,'raw','centered','source')
    assert math.isclose(dc['aggregate']['wall']['candidate_over_reference'],0.8647788577367816,abs_tol=1e-14)
    assert math.isclose(dc['aggregate']['gpu']['candidate_over_reference'],0.8148484741229921,abs_tol=1e-14)
    assert math.isclose(mc['aggregate']['wall']['candidate_over_reference'],0.993972368122908,abs_tol=1e-14)
    assert math.isclose(mc['aggregate']['gpu']['candidate_over_reference'],0.9927054700237485,abs_tol=1e-14)
    fmetrics={}
    for k in f['aggregate']['arms_s']['dense']:
        a,b=(mean(pair['arms_s'][arm][k] for pair in f['pairs']) for arm in ('dense','wide8192'))
        assert math.isclose(a,f['aggregate']['arms_s']['dense'][k],abs_tol=1e-12)
        assert math.isclose(b,f['aggregate']['arms_s']['wide8192'][k],abs_tol=1e-12)
        fmetrics[k]=speed(a,b)
        assert math.isclose(fmetrics[k]['speedup_reference_over_candidate'],f['aggregate']['speedup'][k],abs_tol=1e-12)
    ffn=dict(job=f['job'],source_clusters=f['source_count'],pairs=f['formal_pairs'],
             aggregation='Ratio of arithmetic mean seconds over six balanced matched pairs, as archived.',aggregate=fmetrics)
    da=dc['aggregate']
    phase=dict(anchor_extra_seconds=da['anchors']['candidate_seconds']-da['anchors']['reference_seconds'],
               hit_saved_seconds=da['hits']['reference_seconds']-da['hits']['candidate_seconds'],
               net_GPU_saved_seconds=da['gpu']['reference_seconds']-da['gpu']['candidate_seconds'],
               outside_events_extra_seconds=da['outside_gpu_events']['candidate_seconds']-da['outside_gpu_events']['reference_seconds'],
               net_wall_saved_seconds=da['wall']['reference_seconds']-da['wall']['candidate_seconds'])
    assert math.isclose(phase['hit_saved_seconds']-phase['anchor_extra_seconds'],phase['net_GPU_saved_seconds'],abs_tol=1e-12)
    assert math.isclose(phase['net_GPU_saved_seconds']-phase['outside_events_extra_seconds'],phase['net_wall_saved_seconds'],abs_tol=1e-12)
    overlaps=sorted(set(dc['sources']) & set(mc['sources']))
    assert not overlaps
    projections={k:dict(candidate_over_reference=da[k]['candidate_over_reference']*mc['aggregate'][k]['candidate_over_reference'],
                        speedup=1/(da[k]['candidate_over_reference']*mc['aggregate'][k]['candidate_over_reference'])) for k in ('wall','gpu')}
    fc=fmetrics
    bounds=dict(
        scope='Conditional measured-share ceilings: set specified work to zero while all other measured costs stay fixed; not attainable performance predictions.',
        FFN_share_of_Dense_fourstep_GPU=fc['ffn_with_post']['reference_seconds']/fc['denoise']['reference_seconds'],
        maximum_GPU_fraction_saved_if_hit_FFN_free_no_anchor_extra=fc['hit_steps']['reference_seconds']/fc['denoise']['reference_seconds'],
        maximum_GPU_speedup_if_hit_FFN_free_no_anchor_extra=fc['denoise']['reference_seconds']/(fc['denoise']['reference_seconds']-fc['hit_steps']['reference_seconds']),
        C_GPU_floor_if_both_entire_hit_steps_free=da['anchors']['candidate_seconds'],
        Dense_over_C_GPU_ceiling_if_both_entire_hit_steps_free=da['gpu']['reference_seconds']/da['anchors']['candidate_seconds'],
        C_wall_floor_if_entire_hit_steps_free_others_fixed=da['wall']['candidate_seconds']-da['hits']['candidate_seconds'],
        Dense_over_C_wall_ceiling_if_entire_hit_steps_free_others_fixed=da['wall']['reference_seconds']/(da['wall']['candidate_seconds']-da['hits']['candidate_seconds']))
    identities={k:dict(path=str(getattr(args,k)),sha256=hashlib.sha256(getattr(args,k).read_bytes()).hexdigest()) for k in ('dense_c','mass','ffn_only')}
    setup_seconds=sum(item['seconds'] for item in d['setup'])
    break_even=setup_seconds/phase['net_wall_saved_seconds']
    out=dict(status='RECOMPUTED_FROM_MATCHED_RECEIPTS',source_identity=identities,
             model_scope='All three experiments use the distilled four-step HHLL Wan path; Dense means native attention and FFN on the same distilled model, not the original many-step teacher.',
             ffn_only=ffn,frozen_C_vs_Dense=dc,centered_vs_frozen_C=mc,
             frozen_C_phase_accounting=phase,conditional_measured_share_bounds=bounds,
             conditional_preparation_amortization=dict(
                 assumption='Treat all archived 22.4499 s wide setup as a candidate-only one-off cost; preserve the observed arithmetic mean warm saving per clip. Excludes unseparated compilation and shared cold initialization.',
                 preparation_seconds=setup_seconds,warm_mean_wall_saved_per_clip=phase['net_wall_saved_seconds'],
                 break_even_clip_count=break_even,minimum_integer_clips=math.ceil(break_even),
                 scope='Conditional accounting example, not a measured single-request cold-start speedup.'),
             preparation_scope=dict(
                 formal_wall='Warm prepared process through result save; excludes model load, port preparation and warm-up.',
                 job199260=dict(cold_initialization_seconds=d['cold_init_seconds'],wide_setup=d['setup'],
                    warmup_wall_seconds={k:v['record']['wall'] for k,v in d['warmup'].items()},
                    warmup_C_new_packed_sites=d['warmup']['C']['record']['counters']['new_packed_sites']),
                 job201500=dict(cold_initialization_recorded=False,port_setup_recorded=False,
                    first_compilation_separately_recorded=False,
                    warmup_wall_seconds={k:v['record']['wall'] for k,v in m['warmup'].items()},
                    warmup_new_packed_sites={k:v['record']['counters']['new_packed_sites'] for k,v in m['warmup'].items()})),
             cross_batch_bridge=dict(source_intersection=overlaps,status='NOT_A_MATCHED_MEASUREMENT',
                                     conditional_projection_only=projections,
                                     reason='Different source sets and jobs; no common same-protocol Dense/student/centered panel.'),
             distillation_composition=dict(identity='T_teacher / T_combined = (T_teacher / T_student) * (T_student / T_combined)',
                  valid_scope='The same source cohort, timing definition, actual common student configuration, hardware and quality target. No probabilistic independence assumption is needed for the exact ratio identity.',
                  teacher_matched_wall_measurement_available_in_reviewed_receipts=False,
                  combined_original_teacher_speedup_measured=False,
                  NFE_caveat='Count actual conditional/unconditional evaluations and distinguish CFG removal, per-evaluation kernel differences and fixed end-to-end overhead.'),
             new_model_executions=0,new_GPU_jobs=0,new_video_decodes=0)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'ffn_only':fmetrics,'frozen_C':dc['aggregate'],'centered':mc['aggregate'],
                      'phase_accounting':phase,'conditional_bounds':bounds,'cross_batch':out['cross_batch_bridge']},indent=2))


if __name__=='__main__':main()
