"""Adversarial synthetic metadata fixtures; none represents generated videos."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check_metric_contract import catalog, validate_record
from metric_evidence_validation import OFFICIAL_I2V
import test_candidate_rollout_gate as rollout_tests

class MetricEvidenceValueTests(unittest.TestCase):
    def setUp(self):
        self.cat=catalog();self.metrics={m['id']:m for m in self.cat['metrics']}

    def obs(self,mid,value=1.0):
        m=self.metrics[mid]
        return dict(metric_id=mid,level=m['level'],unit=m['unit'],value=value,
            source='SYNTHETIC_TEST_ONLY',sample_unit='fixture',spatial_grain='fixture',
            temporal_grain='fixture',population='fixture',aggregation='fixture',
            comparison_reference='fixture',execution_mode=m.get('allowed_execution_modes',['saved_native_same_state'])[0],
            source_exposure='development',implementation_sha='a'*64,evidence_locator='SYNTHETIC_TEST_ONLY')

    def record(self,claim,entries):
        return dict(record_id='SYNTHETIC_TEST_ONLY',method_id='SYNTHETIC_FIXTURE_NOT_ACTUAL_VIDEO',
                    evaluation_status='MEASURED',claim=claim,observations=entries)

    def test_nonfinite_and_wrong_scalar_types(self):
        for value in [float('nan'),float('inf'),float('-inf'),True,'1',{},-1]:
            with self.subTest(value=value):
                self.assertTrue(validate_record(self.record('local_quality_diagnostic',[self.obs('post_writeback_sse',value)]),self.cat))

    def test_per_source_observations_and_duplicate_identity(self):
        x=self.obs('post_writeback_sse');x['observation_id']='source0'
        y=copy.deepcopy(x);y['observation_id']='source10'
        r=self.record('local_quality_diagnostic',[x,y])
        self.assertEqual(validate_record(r,self.cat),[])
        y['observation_id']='source0'
        self.assertIn('duplicate observation identity post_writeback_sse',validate_record(r,self.cat))

    def test_nmse_needs_explicit_positive_energy(self):
        x=self.obs('post_writeback_nmse')
        r=self.record('local_quality_diagnostic',[x])
        self.assertTrue(validate_record(r,self.cat))
        x.update(denominator=10,denominator_definition='sum squared same-state native D-Q post tensor')
        self.assertEqual(validate_record(r,self.cat),[])

    def speed_record(self):
        x=self.obs('complete_ffn_latency_ms',1)
        x.update(baseline_value=2,repetitions=7,timing_boundary='all intended FFN calls including routing and writeback',deployment_id='fixture')
        r=self.record('native_speedup',[x])
        r['benchmark_context']=dict(deployment_id='fixture',checkpoint_identity='fixture',precision='fixture',scheduler='fixture',actual_timesteps=[4,3,2,1],frames=81,resolution=[480,832],hardware='fixture')
        return r

    def test_native_timing_and_relabelled_synthetic(self):
        r=self.speed_record();self.assertEqual(validate_record(r,self.cat),[])
        r['observations'][0]['execution_mode']='synthetic'
        self.assertIn('complete_ffn_latency_ms execution_mode incompatible with metric',validate_record(r,self.cat))

    def test_speedup_requires_improvement_and_matching_deployment(self):
        r=self.speed_record();r['observations'][0]['value']=3
        self.assertIn('native_speedup requires faster measured complete FFN',validate_record(r,self.cat))
        r['observations'][0]['deployment_id']='other'
        self.assertTrue(any('deployment identity' in e for e in validate_record(r,self.cat)))

    def video_record(self,root):
        rollout=rollout_tests.CandidateRolloutGateTests().sample()
        report=root/'research/candidate_rollout/README.md';report.parent.mkdir(parents=True)
        report.write_text('SYNTHETIC review fixture only.',encoding='utf-8')
        (root/'rollout.json').write_text(json.dumps(rollout),encoding='utf-8')
        r=self.record('video_noncollapse_smoke',[self.obs('free_rollout_completed',True),self.obs('visual_collapse_review',{'verdict':'NO_CATASTROPHIC_COLLAPSE'})])
        r.update(full_student_free_running=True,paired_baseline_full_video=True,candidate_rollout_record='rollout.json')
        return r

    def test_successful_smoke_metadata_and_failed_values(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.video_record(root)
            self.assertEqual(validate_record(r,self.cat,root),[])
            r['observations'][0]['value']=False
            r['observations'][1]['value']['verdict']='COLLAPSE'
            errors=validate_record(r,self.cat,root)
            self.assertIn('video claim contains incomplete rollout',errors)
            self.assertIn('video claim contains failed or inconclusive review',errors)

    def test_video_booleans_do_not_replace_linked_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.video_record(root);r.pop('candidate_rollout_record')
            self.assertTrue(any('candidate_rollout_record' in e for e in validate_record(r,self.cat,root)))

    def test_partial_official_suite_is_a_valid_scoped_diagnostic(self):
        r=self.record('scoped_official_video_diagnostic',[self.obs('vbench_i2v_dynamic_degree')])
        self.assertEqual(validate_record(r,self.cat),[])

    def test_full_independent_metadata_then_development_exposure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.video_record(root);r['claim']='independent_video_quality'
            r.update(source_overlap_audit_passed=True,official_suite_revision='a'*40)
            r['benchmark_context']=self.speed_record()['benchmark_context']
            for mid in sorted(OFFICIAL_I2V):
                x=self.obs(mid);x.update(source_exposure='fresh_final',source_manifest_sha256='b'*64,baseline_value=1)
                r['observations'].append(x)
            for x in r['observations']:x['deployment_id']='fixture'
            self.assertEqual(validate_record(r,self.cat,root),[])
            r['observations'][-1]['source_exposure']='development'
            self.assertTrue(any('fresh_final' in e for e in validate_record(r,self.cat,root)))
            r['observations'].pop()
            self.assertIn('full independent claim requires all nine official I2V dimensions',validate_record(r,self.cat,root))

    def test_pareto_requires_complete_rollout_quality_and_review(self):
        r=self.record('video_speed_quality_pareto',[self.obs('visual_collapse_review',{'verdict':'NO_CATASTROPHIC_COLLAPSE'}),self.obs('fvd_dataset')])
        r.update(full_student_free_running=True,paired_baseline_full_video=True,matched_prior_art_or_strong_fixed=True)
        errors=validate_record(r,self.cat)
        self.assertIn('claim missing required metrics',errors)
        self.assertTrue(any('pareto_review_record' in e for e in errors))

    def test_failed_rollout_can_be_recorded_as_diagnostic(self):
        r=self.record('video_rollout_diagnostic',[self.obs('free_rollout_completed',False),self.obs('visual_collapse_review',{'verdict':'COLLAPSE'})])
        self.assertEqual(validate_record(r,self.cat),[])

    def test_unbiased_jedi_estimate_can_be_negative(self):
        x=self.obs('jedi_dataset',-0.001);x['estimator']='unbiased_u_statistic'
        r=self.record('scoped_official_video_diagnostic',[x])
        self.assertEqual(validate_record(r,self.cat),[])
        x['estimator']='biased_v_statistic'
        self.assertIn('jedi_dataset value must be nonnegative',validate_record(r,self.cat))

if __name__=='__main__':unittest.main()
