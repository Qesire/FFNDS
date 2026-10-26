import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from e0_full_depth_trace_contract import four_arm_geometry,pair_geometry,scheduler_geometry,step_pre_geometry,FullDepthTrace


class PairedTraceTest(unittest.TestCase):
    def setUp(self):
        self.zero=np.zeros((3,2));self.one=np.ones((3,2))

    def test_equal_large_components_can_cancel_exactly(self):
        g=four_arm_geometry(self.zero,self.one,self.one,self.zero,chunk_rows=2)
        self.assertEqual(g['student_operator_SSE'],6)
        self.assertEqual(g['prefix_response_SSE'],6)
        self.assertEqual(g['operator_prefix_dot'],-6)
        self.assertEqual(g['total_SSE'],0)
        self.assertEqual(g['operator_prefix_cosine'],-1)

    def test_signed_direction_distinguishes_repair_from_growth(self):
        a=scheduler_geometry(self.zero,self.one,self.zero,2*self.one,self.zero,self.zero,sigma=.5,sigma_next=0)
        b=scheduler_geometry(self.zero,self.one,self.zero,-2*self.one,self.zero,2*self.one,sigma=.5,sigma_next=0)
        self.assertEqual(a['actual_energy_change'],-6)
        self.assertEqual(b['actual_energy_change'],18)
        self.assertEqual(a['ideal_update_defect_SSE'],0)

    def test_rounding_correction_is_recorded_not_assumed_zero(self):
        g=scheduler_geometry(self.zero,self.one,self.zero,2*self.one,self.zero,.01*self.one,sigma=.5,sigma_next=0)
        self.assertGreater(g['ideal_update_defect_SSE'],0)

    def test_actual_four_dimensional_entry_cast_can_remove_a_difference(self):
        teacher=np.full((2,3,2,2),1.0001,dtype=np.float32)
        student=np.full_like(teacher,1.0002)
        entry=step_pre_geometry(teacher,student,teacher.astype(np.float16),student.astype(np.float16))
        self.assertGreater(entry['delta_before_SSE'],0)
        self.assertEqual(entry['delta_after_SSE'],0)
        flow=np.zeros_like(teacher)
        g=scheduler_geometry(teacher.astype(np.float16),student.astype(np.float16),flow,flow,
                             teacher.astype(np.float16),student.astype(np.float16),sigma=.75,sigma_next=.25)
        self.assertEqual(g['shape'],[2,3,2,2])
        self.assertEqual(g['h'],.5)
        self.assertEqual(g['ideal_update_defect_SSE'],0)

    def test_native_state_panels_cannot_be_called_full_latents(self):
        with self.assertRaises(ValueError):
            FullDepthTrace((3,2),0,42,dict(protocol_sha256='fixture',native_source_sha256='fixture',
                                         model_digest='fixture',bank_RUN_sha256='fixture'),state_shape=(3,2))

    def test_shape_and_nonfinite_are_rejected(self):
        with self.assertRaises(ValueError):pair_geometry(self.zero,np.zeros((2,2)))
        x=self.one.copy();x[0,0]=float('nan')
        with self.assertRaises(ValueError):four_arm_geometry(self.zero,x,self.one,self.zero)

    def test_invalid_reduction_chunk_and_nonfinite_metadata_are_rejected(self):
        for chunk in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):pair_geometry(self.zero,self.one,chunk_rows=chunk)
        p=self.payload()
        p['numerical_check']['teacher_native_vs_continuous']['difference_SSE']=float('nan')
        with self.assertRaises(ValueError):self.collector().add_site(0,0,'high_noise_model',**p)
        g=scheduler_geometry(self.zero,self.one,self.zero,2*self.one,self.zero,self.zero,sigma=.5,sigma_next=0)
        g['delta_flow_dot']=float('inf')
        with self.assertRaises(ValueError):
            self.collector().add_step(0,g,entry_cast=step_pre_geometry(self.zero,self.one,self.zero,self.one),boundary_hashes=self.hashes(),
                                     pre_dtype_teacher='float64',post_dtype_teacher='float64',
                                     pre_dtype_student='float64',post_dtype_student='float64')

    def collector(self):
        return FullDepthTrace((3,2),0,42,dict(protocol_sha256='fixture',native_source_sha256='fixture',
                                            model_digest='fixture',bank_RUN_sha256='fixture'),state_shape=(3,2),synthetic=True)

    def hashes(self):
        return {f'{branch}_{boundary}':'fixture' for branch in ['teacher','student']
                for boundary in ['before_step_pre','after_step_pre','after_step_post']}

    def payload(self):
        return dict(teacher_q1=self.zero,student_q1=self.one,
                    ffn_outputs=dict(dense_teacher=self.zero,approx_teacher=self.one,
                                     dense_student=self.one,approx_student=self.zero),
                    native_writebacks=dict(dense_teacher=self.zero,approx_teacher=self.one,
                                           dense_student=self.one,approx_student=self.zero),
                    teacher_block=self.zero,student_block=self.one,
                    numerical_check=dict(status='MEASURED_SAME_INPUT',coefficient_provenance_sha256='fixture',
                                         row_ids=[0],teacher_native_vs_continuous={'difference_SSE':0},
                                         student_native_vs_continuous={'difference_SSE':0}))

    def test_two_layers_cannot_finish_full_depth_contract(self):
        c=self.collector()
        c.add_site(0,0,'high_noise_model',**self.payload())
        c.add_site(0,30,'high_noise_model',**self.payload())
        with self.assertRaises(ValueError):c.finish()
        with self.assertRaises(ValueError):c.add_site(0,0,'high_noise_model',**self.payload())

    def test_writeback_and_numeric_check_cannot_be_inferred_from_ffn(self):
        c=self.collector();p=self.payload();p['native_writebacks']={}
        with self.assertRaises(ValueError):c.add_site(0,0,'high_noise_model',**p)
        p=self.payload();p['numerical_check']={'status':'NOT_RUN'}
        with self.assertRaises(ValueError):c.add_site(0,0,'high_noise_model',**p)

    def test_complete_fixture_stays_synthetic(self):
        c=self.collector()
        for s in range(4):
            for b in range(40):c.add_site(s,b,'high_noise_model' if s<2 else 'low_noise_model',**self.payload())
            g=scheduler_geometry(self.zero,self.one,self.zero,2*self.one,self.zero,self.zero,sigma=.5,sigma_next=0)
            c.add_step(s,g,entry_cast=step_pre_geometry(self.zero,self.one,self.zero,self.one),boundary_hashes=self.hashes(),
                       pre_dtype_teacher='float64',post_dtype_teacher='float64',
                       pre_dtype_student='float64',post_dtype_student='float64')
        r=c.finish();self.assertEqual(len(r['sites']),160)
        self.assertEqual(r['status'],'SYNTHETIC_SOFTWARE_VALIDATION_ONLY')
        self.assertFalse(r['quality_accepted'])
        c.steps[2]['boundary_hashes']['teacher_before_step_pre']='different-state'
        with self.assertRaises(ValueError):c.finish()


if __name__=='__main__':unittest.main()
