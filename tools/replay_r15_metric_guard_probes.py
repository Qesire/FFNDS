"""Replay five explicitly synthetic adversarial metadata cases against the guard."""
import json
from pathlib import Path
from check_metric_contract import catalog,validate_record

ROOT=Path(__file__).resolve().parents[1]
def replay():
    cat=catalog();m={v['id']:v for v in cat['metrics']}
    def obs(key,value=1.0,mode='saved_native_same_state'):
        x=m[key]
        return dict(metric_id=key,level=x['level'],unit=x['unit'],value=value,source='TEST_ONLY',sample_unit='fixture',spatial_grain='fixture',temporal_grain='fixture',population='development',aggregation='fixture',comparison_reference='fixture',execution_mode=mode,source_exposure='development',implementation_sha='not-a-hash',evidence_locator='does-not-exist')
    original_five=['vbench_i2v_subject','vbench_i2v_background','vbench_i2v_motion_smoothness','vbench_i2v_aesthetic_quality','vbench_i2v_imaging_quality']
    tests={
      'synthetic_labeled_native_speed':dict(record_id='test',evaluation_status='MEASURED',claim='native_speedup',observations=[obs('complete_ffn_latency_ms',1,'synthetic')]),
      'failed_video_accepted_as_smoke':dict(record_id='test',evaluation_status='MEASURED',claim='video_noncollapse_smoke',full_student_free_running=True,paired_baseline_full_video=True,observations=[obs('free_rollout_completed',False),obs('visual_collapse_review',{'verdict':'COLLAPSE'})]),
      'nonfinite_local_SSE':dict(record_id='test',evaluation_status='MEASURED',claim='local_quality_diagnostic',observations=[obs('post_writeback_sse',float('nan'))]),
      'development_claimed_independent':dict(record_id='test',evaluation_status='MEASURED',claim='independent_video_quality',full_student_free_running=True,paired_baseline_full_video=True,observations=[obs(k) for k in original_five]),
      'pareto_without_complete_quality_or_rollout':dict(record_id='test',evaluation_status='MEASURED',claim='video_speed_quality_pareto',full_student_free_running=True,paired_baseline_full_video=True,matched_prior_art_or_strong_fixed=True,observations=[obs(k) for k in ['visual_collapse_review','complete_ffn_latency_ms','generation_gpu_time_s','fvd_dataset']]),
    }
    out={name:{'accepted':not validate_record(r,cat),'errors':validate_record(r,cat)} for name,r in tests.items()}
    assert all(not v['accepted'] for v in out.values())
    target=ROOT/'research/20261023/metric_guard_after_R15.json'
    target.write_bytes((json.dumps(dict(scope='Synthetic metadata cases only; unit tests also isolate each failure with valid surrounding metadata',probes=out),indent=2)+'\n').encode())
    return {'rejected':len(out),'total':len(out)}
if __name__=='__main__':print(json.dumps(replay()))
