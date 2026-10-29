"""Official-result composition controls; no metric models or saved experiments."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
from tools import e0_raw_bf16_temporal_video20_quality as m


class QualityControls(unittest.TestCase):
    def test_new_counts_exclude_twenty_previously_scored_Dense_clips(self):
        self.assertEqual(m.LIMITS['new_video_dimension_results'], 40*6+20*2)
        self.assertEqual(m.LIMITS['RAFT_forwards'], 40*40)
        self.assertEqual(m.LIMITS['RAFT_iterations'], 40*40*20)
        self.assertEqual(m.COSTS['imaging_quality']['MUSIQ'], 40*81)
        self.assertEqual(m.COSTS['i2v_subject']['DINO'], 20*82)
        self.assertEqual(m.LIMITS['video_generations'], 0)

    def fixture(self):
        hf = Path('/hf')
        videos = [{'source': i, 'mode': mode, 'video': {'path': f'{i}/{mode}.mp4'},
                   'i2v_dimension': 'i2v_subject' if i < 10 else 'i2v_background'}
                  for i in range(20) for mode in m.MODES]
        plan = {'videos': videos, 'new_videos': [v for v in videos if v['mode'] != 'dense']}
        dimensions, previous = [], []
        for d in m.DIMENSIONS:
            selected = m.rows_for_dimension(videos, d)
            rows = [{'video_path': str(hf/v['video']['path']),
                     'video_results': True if d == 'dynamic_degree' else (1 if v['mode'] == 'dense' else 2)}
                    for v in selected]
            dimensions.append({'dimension': d, 'result': [123, [r for r in rows if '/dense.mp4' not in r['video_path']]]})
            # Old unrelated arms are intentionally very bad and must be ignored.
            dense = [r for r in rows if '/dense.mp4' in r['video_path']]
            unrelated = [{'video_path': f'/hf/{i}/shared.mp4', 'video_results': False if d == 'dynamic_degree' else -999} for i in range(len(dense))]
            previous.append({'dimension': d, 'result': [456, dense+unrelated]})
        flows = [{'video_path': str(hf/v['video']['path'])} for v in videos if v['mode'] != 'dense']
        prior_flows = [{'video_path': f'/hf/{i}/dense.mp4'} for i in range(20)]
        prior_flows += [{'video_path': f'/hf/{i}/shared.mp4'} for i in range(20)]
        return plan, dimensions, flows, {'completed_dimensions': previous, 'RAFT_observations': prior_flows}

    def test_reuses_only_exact_Dense_rows_and_never_invents_official_aggregate(self):
        plan, dimensions, flows, old = self.fixture()
        with patch.object(m, 'dense_quality', return_value=({}, old)):
            combined, raft = m.combine_per_video_evidence(plan, Path('/hf'), dimensions, flows)
        self.assertEqual(len(combined), 8)
        self.assertEqual(len(raft), 60)
        self.assertEqual(len(flows), 40)
        for entry in combined:
            self.assertIsNone(entry['result'][0])
            self.assertFalse(any('/shared.mp4' in r['video_path'] for r in entry['result'][-1]))
            self.assertEqual(len(entry['result'][-1]), 60 if entry['dimension'] in m.COMMON else 30)

    def test_missing_or_duplicate_new_per_video_result_is_rejected(self):
        plan, dimensions, flows, old = self.fixture()
        for altered in ('missing', 'duplicate'):
            changed = copy.deepcopy(dimensions)
            if altered == 'missing': changed[0]['result'][-1].pop()
            else: changed[0]['result'][-1].append(copy.deepcopy(changed[0]['result'][-1][0]))
            with patch.object(m, 'dense_quality', return_value=({}, old)):
                with self.assertRaises(ValueError): m.combine_per_video_evidence(plan, Path('/hf'), changed, flows)


if __name__ == '__main__': unittest.main()
