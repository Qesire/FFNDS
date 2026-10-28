"""Portable synthetic controls for new full21tau acquisition, never a model run."""
import copy
import hashlib
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from tools import e0_raw_bf16_full_trajectory_collect as m


def source_cohort():
    strata = ['subject']*5+['background']*2+['camera']*3
    rows = []
    for i, stratum in enumerate(strata):
        prompt = 'synthetic condition '+str(i)
        rows.append({'source_id': 'test/source'+str(i), 'native_source_ordinal': 100+i, 'cohort_index': i,
                     'part': i % 2, 'role': 'NEW_DEVELOPMENT', 'collector_role': 'observation',
                     'scene_stratum': stratum, 'dimension': 'synthetic_'+stratum,
                     'image_type': 'synthetic', 'original_prompt': prompt,
                     'original_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                     'archive_member': 'data/'+str(i)+'.jpg',
                     'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                     'image': {'root': 'hf', 'path': 'tiny/'+str(i)+'.png', 'bytes': 4, 'sha256': format(i, '064x')},
                     'history': 'SYNTHETIC_CONTROL_NOT_REAL_SOURCE', 'exposure_history': {'prior_evaluation': False}})
    return {'schema': m.COHORT_SCHEMA, 'status': 'PROVISIONAL_NEEDS_REVIEW', 'sources': rows}


class TrajectoryControls(unittest.TestCase):
    def test_actual_coordinates_complete_trajectories_and_horizontal_neighbors(self):
        rows = m.anchor_rows()
        self.assertEqual(len(rows), 168)
        self.assertEqual(rows, sorted(set(rows)))
        geometry = m.row_geometry(rows)
        for anchor, (y, x) in enumerate(m.ANCHORS):
            self.assertEqual([row//1508 for row in rows if row % 1508 == y*52+x], list(range(21)))
            self.assertEqual([rows[t*8+anchor] for t in range(21)], [t*1508+y*52+x for t in range(21)])
        for a, b in m.NEIGHBORS:
            self.assertEqual(m.ANCHORS[a][0], m.ANCHORS[b][0])
            self.assertEqual(m.ANCHORS[b][1]-m.ANCHORS[a][1], 1)
        self.assertEqual(geometry['row_taus'], [i//8 for i in range(168)])
        self.assertEqual(geometry['row_anchor_ids'], [i % 8 for i in range(168)])
        for bad in (list(reversed(rows)), rows[:-1], rows[:-1]+[rows[-1]+1]):
            with self.assertRaises(ValueError):
                m.row_geometry(bad)
        draws = m.drawings()
        self.assertTrue(all(draws[str(s)][str(b)] == rows for s in range(4) for b in range(40)))

    def test_actual_packet_shape_and_wrong_tau_or_dtype_rejected(self):
        meta = {k: {'shape': v, 'dtype': 'int64' if k == 'rows' else 'bfloat16', 'raw_tensor_sha256': 'a'*64}
                for k, v in m.ROW_SHAPES.items()}
        m.check_packet_metadata(meta, m.anchor_rows())
        bad = copy.deepcopy(meta)
        bad['h']['shape'][0] = 256
        with self.assertRaises(ValueError):
            m.check_packet_metadata(bad, m.anchor_rows())
        bad = copy.deepcopy(meta)
        bad['gate']['dtype'] = 'float32'
        with self.assertRaises(ValueError):
            m.check_packet_metadata(bad, m.anchor_rows())
        with self.assertRaises(ValueError):
            m.check_packet_metadata(meta, m.anchor_rows()[:-1]+[m.anchor_rows()[-1]+1])

    def test_asymmetric_fixed_actual_work_and_tensor_bytes(self):
        self.assertEqual(m.PACKET_BYTES, 11558208)
        for part, count, flows, loads, neutral in ((0, 6, 24, 972, 10), (1, 5, 20, 810, 0)):
            limits = m.limits(part)
            self.assertEqual(limits['native_full_model_flows'], flows)
            self.assertEqual(limits['native_FFN_calls'], count*160)
            self.assertEqual(limits['packet_safetensor_writes'], count*160)
            self.assertEqual(limits['original_checkpoint_block_buffer_loads'], loads)
            self.assertEqual(limits['packet_tensor_D2H'], count*160*9)
            self.assertEqual(limits['packet_tensor_D2H_bytes'], count*160*11558208)
            self.assertEqual(limits['row_index_upload_bytes'], count*160*168*8)
            self.assertEqual(limits['neutral_state_byte_checks'], neutral)
            self.assertEqual(limits['video_generations'], 5)
            self.assertEqual(limits['core_CPU_reports'], 5)
            self.assertEqual(limits['T5_Default_MM_calls'], count*168)
            self.assertEqual(limits['raw_Default_MM_loads'], 812)
            self.assertEqual(limits['decompositions'], 0)
            self.assertLess(m.byte_budget(part)['total_bound_before_actual_frozen_source_snapshot_bytes'], 16 << 30)
        with self.assertRaises(ValueError):
            m.limits(True)

    def test_pure_import_and_private_geometry_do_not_modify_frozen_public_modules(self):
        script = """import sys
sys.path.insert(0,'tools')
import e0_original_bf16_full_depth_collect as original
import e0_original_bf16_full_depth_collect_v2 as v2
before=(original.ROWS,original.PACKET_BYTES,original.limits(),dict(v2.RESOURCES))
import e0_raw_bf16_full_trajectory_collect as new
assert before==(original.ROWS,original.PACKET_BYTES,original.limits(),dict(v2.RESOURCES))
assert original.ROWS==256 and new.legacy.ROWS==168
assert not {'numpy','torch'} & set(sys.modules)
"""
        result = subprocess.run([sys.executable, '-c', script], cwd=m.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_bank_preserves_roles_review_gate_and_ten_image_clusters(self):
        value = source_cohort()
        m.validate_cohort(value, m.ROOT, frozen=False)
        with self.assertRaises(ValueError):
            m.validate_cohort(value, m.ROOT, frozen=True)
        value.update(status='REVIEWED_READY_FOR_RAW_DENSE_ACQUISITION', source_audit={'test': True},
                     source_review={'reviewer_kind': 'ROOT_AGENT', 'status': 'REVIEWED_NEW_DEVELOPMENT'})
        m.validate_cohort(value, m.ROOT, frozen=True)
        for mutate in (lambda x: x['sources'][0].update(consumer20_index=0),
                       lambda x: x['sources'][1]['image'].update(sha256=x['sources'][0]['image']['sha256']),
                       lambda x: x['sources'][-1].update(scene_stratum='subject'),
                       lambda x: x['sources'][0].update(native_source_ordinal=894),
                       lambda x: x['source_review'].update(status='REJECTED'),
                       lambda x: x['source_review'].pop('status')):
            bad = copy.deepcopy(value)
            mutate(bad)
            with self.assertRaises(ValueError):
                m.validate_cohort(bad, m.ROOT, frozen=True)

    def test_before_call_debit_and_failed_call_is_not_observed_or_retried(self):
        report, events = {}, []
        original_part = m.CURRENT_PART
        m.CURRENT_PART = 0
        try:
            ledger = m.legacy.Ledger(report, ['tiny'], events.append)
            seen = []
            def fail():
                seen.append(True)
                raise RuntimeError('fixed synthetic failure')
            with self.assertRaises(RuntimeError):
                ledger.call('tiny', {'video_generations': 1}, fail)
            self.assertEqual(seen, [True])
            self.assertEqual(report['predebited'], {'video_generations': 1})
            self.assertEqual(report['observed'], {})
            self.assertEqual(events[0]['kind'], 'reserved_before_call')
            with self.assertRaises(ValueError):
                ledger.call('unregistered', {'video_generations': 1}, lambda: None)
        finally:
            m.CURRENT_PART = original_part

    def test_closed_v2_source_binding_rejects_v1_wrong_prompt_and_bool_exit(self):
        cohort = source_cohort()
        proof = {'protocol': {'path': 'source-protocol', 'sha256': '1'*64},
                 'RUN': {'path': 'source-RUN', 'sha256': '2'*64}, 'exit': {'path': 'source-exit', 'sha256': '3'*64}}
        cohort.update(status='REVIEWED_READY_FOR_RAW_DENSE_ACQUISITION', source_audit=proof,
                      source_review={'reviewer_kind': 'ROOT_AGENT', 'status': 'REVIEWED_NEW_DEVELOPMENT', 'document': {'path': 'review'}})
        cases = [{**copy.deepcopy(s), 'role': 'observation', 'evidence_role': s['role'], 'image_sha256': s['image']['sha256']}
                 for s in cohort['sources'] if s['part'] == 0]
        plan = {'part': 0, 'cases': cases, 'source_cohort': {'path': 'cohort'}, 'source_acquisition_proof': proof}
        protocol = {'id': m.SOURCE_ID}
        run = {'id': m.SOURCE_ID, 'protocol_sha256': '1'*64, 'model_calls': 0, 'GPU_calls': 0,
               'status': 'COMPLETE_PROVISIONAL_SOURCE_BYTES_PENDING_HUMAN_AND_HISTORY_REVIEW', 'sources': copy.deepcopy(cohort['sources'])}
        exited = {'id': m.SOURCE_ID, 'protocol_sha256': '1'*64, 'run_sha256': '2'*64,
                  'process_exited': True, 'child_returncode': 0, 'timeout': False, 'automatic_retry': False}
        values = {'cohort': cohort, 'source-protocol': protocol, 'source-RUN': run, 'source-exit': exited}
        def execute(mapping):
            seen = []
            def bound(_root, b):
                seen.append(b['path'])
                return b['path']
            with patch.object(m.native, 'bound', side_effect=bound), patch.object(m.native, 'read', side_effect=lambda p: mapping[p]):
                result = m.verify_source_cohort(plan, m.ROOT, verify_images=True)
            self.assertEqual(result['sources'], cohort['sources'])
            self.assertTrue(all(s['image']['path'] in seen for s in cohort['sources']))
        execute(values)
        for mutate in (lambda x: [x[k].update(id='E0_RAW_BF16_FRESH10_SOURCE_ARCHIVE_CPU_20261029')
                                 for k in ('source-protocol', 'source-RUN', 'source-exit')],
                       lambda x: x['source-RUN']['sources'][0].update(prompt='changed after actual extraction'),
                       lambda x: x['source-exit'].update(child_returncode=False)):
            bad = copy.deepcopy(values)
            mutate(bad)
            with self.assertRaises(ValueError):
                execute(bad)


if __name__ == '__main__':
    unittest.main()
