import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from run_e0_fresh_official_directional import validate_cohort,PAIRS


class FreshSourceContract(unittest.TestCase):
    def setUp(self):
        self.audit=json.loads((ROOT/'research/method_discussion/discoveries/E0/independent-phase2-official-four-input-audit-20261026_v11.json').read_text())
        self.samples=[dict(s,source_id=s['image_sha256']) for s in self.audit['selected_sources']]

    def test_four_sources_with_two_crossed_seed_repeats_are_six_pairs(self):
        validate_cohort(self.samples,self.audit)
        self.assertEqual(len(PAIRS),6);self.assertEqual(len({x[0] for x in PAIRS}),4)
        self.assertEqual([s for i,s in PAIRS if i==908],[42,54])
        self.assertEqual([s for i,s in PAIRS if i==425],[42,54])

    def test_reused_project_image_is_refused_even_if_prompt_is_new(self):
        self.audit['prior_image_SHA'].append(self.samples[0]['source_id'])
        with self.assertRaises(ValueError):validate_cohort(self.samples,self.audit)

    def test_changed_camera_instruction_and_duplicate_source_are_refused(self):
        bad=copy.deepcopy(self.samples);bad[2]['camera_instruction']='camera zooms in'
        with self.assertRaises(ValueError):validate_cohort(bad,self.audit)
        bad=copy.deepcopy(self.samples);bad[3]['source_id']=bad[2]['source_id']
        with self.assertRaises(ValueError):validate_cohort(bad,self.audit)


if __name__=='__main__':unittest.main()
