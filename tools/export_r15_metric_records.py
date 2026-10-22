"""Map frozen R14/R14N0 measurements to metric records; no model evaluation."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def put(path,data):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes((json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
def export():
    cat={m['id']:m for m in read('research/metric_system/catalog.json')['metrics']}
    result=read('research/20261023/metric_evidence_recalculation_R15.json')
    aggregation_sha=hashlib.sha256((ROOT/'tools/audit_r15_metric_evidence.py').read_bytes()).hexdigest()
    observations=[]
    for source in result['per_source']:
        for arm,stats in source['statistics'].items():
            # The posthoc common mask is a saved-scalar reference, not an executed mask.
            if arm=='posthoc_common':continue
            for key,mid in [('SSE','post_writeback_sse'),('MSE','post_writeback_mse')]:
                m=cat[mid]
                observations.append(dict(metric_id=mid,observation_id=f"source{source['source_index']}/{arm}",level=m['level'],unit=m['unit'],value=stats[key],source=f"R14N0 source {source['source_index']}",sample_unit='64 four-token stripes / 256 rows / 5120 channels',spatial_grain='registered complete horizontal four-token stripes',temporal_grain='step2 block20',population='two previously exposed development sources',aggregation='sum squared error' if key=='SSE' else 'SSE / (64*4*5120)',comparison_reference='same-state native dense D-Q writeback; arm separately identified',execution_mode='saved_native_same_state',source_exposure='development',implementation_sha=aggregation_sha,evidence_locator=f"research/20261023/metric_evidence_recalculation_R15.json#per_source/{source['source_index']}/{arm}",original_native_result_sha256=result['source_sha256']['R14N0_native_results.json']))
    put('research/metric_system/records/R15_R14N0_local_mapping.json',dict(record_id='R15_MAPPING_OF_FROZEN_R14N0',method_id='R14N0_COMPLETE_STRIPE_NATIVE_VALUE',evaluation_status='MEASURED',claim='local_quality_diagnostic',observation_origin='Reaggregation of frozen native R14N0 data; no new experiment',observations=observations))
    backend=read('research/20261022/R14_three_direction_backend_results.json')
    ms=backend['median_ms'];m=cat['synthetic_backend_speedup']
    x=dict(metric_id=m['id'],level=m['level'],unit=m['unit'],value=ms['dense']/ms['three_fused_dynamic'],source='R14H1 same-run random resident tensors',sample_unit='registered full-shape synthetic FFN-like chain',spatial_grain='M31668 D5120 H13824',temporal_grain='repeated backend timing; no video steps',population='7 registered arms on one A100',aggregation='ratio of same-run median dense/dynamic latency',comparison_reference='same-run synthetic dense',execution_mode='synthetic',source_exposure='synthetic_not_quality_data',implementation_sha=backend['source_sha256']['tools/probe_three_direction_backend.py'],evidence_locator='research/20261022/R14_three_direction_backend_results.json#median_ms',raw_candidate_ms=ms['three_fused_dynamic'],raw_baseline_ms=ms['dense'])
    put('research/metric_system/records/R15_R14_backend_mapping.json',dict(record_id='R15_MAPPING_OF_FROZEN_R14_BACKEND',method_id=backend['id'],evaluation_status='MEASURED',claim='synthetic_backend_diagnostic',observation_origin='Frozen measured R14 backend timing',observations=[x]))
    return len(observations)+1
if __name__=='__main__':print(json.dumps({'mapped_historical_observations':export(),'new_runs':0}))
