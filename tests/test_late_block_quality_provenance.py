import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from compare_late_block_quality import validate_generation_pair


class LateBlockQualityProvenanceTests(unittest.TestCase):
    def fixtures(self):
        dense = {'arm':'D-Q','packed_policy_sha256':None,'assets':{'model':'same'},'inputs_sha256':'inputs',
                 'input_recovery_sha256':'recovery','versions':{'torch':'same'},'seed_semantics':'reset each request',
                 'native_sources':{'native.py':'same'},'sources':{'core.py':'same','old_wrapper.py':'old'}}
        candidate = copy.deepcopy(dense)
        candidate.update(arm='D-Q-PACKED-TRAINED-REFERENCE',packed_policy_sha256='active-policy',
                         identity_control_proof_sha256='four-controls',cohort='disjoint20',runtime_sources={'rank.py':'rank512'})
        candidate['sources'] = {'core.py':'same','new_wrapper.py':'new'}
        return dense,candidate,copy.deepcopy(dense)

    def check(self,dense,candidate,trusted):
        validate_generation_pair(dense,candidate,trusted,'active-policy','four-controls','disjoint20',{'rank.py':'rank512'},['core.py'])

    def test_truthful_wrapper_difference_preserves_native_reference(self):
        self.check(*self.fixtures())

    def test_wrapper_permission_does_not_allow_native_or_input_changes(self):
        for change in ['native','core','input','precision','seed','proof','cohort','policy']:
            dense,candidate,trusted = self.fixtures()
            if change == 'native': candidate['native_sources']['native.py'] = 'changed'
            if change == 'core': candidate['sources']['core.py'] = 'changed'
            if change == 'input': candidate['inputs_sha256'] = 'other-images'
            if change == 'precision': candidate['versions']['torch'] = 'other-version'
            if change == 'seed': candidate['seed_semantics'] = 'continued stream'
            if change == 'proof': candidate['identity_control_proof_sha256'] = 'other-proof'
            if change == 'cohort': candidate['cohort'] = 'formal20'
            if change == 'policy': candidate['packed_policy_sha256'] = 'other-policy'
            with self.subTest(change=change),self.assertRaises(RuntimeError):
                self.check(dense,candidate,trusted)

    def test_reused_baseline_cannot_hide_an_untrusted_native_file(self):
        dense,candidate,trusted = self.fixtures()
        dense['native_sources']['untrusted.py'] = 'not-in-current-pipeline'
        with self.assertRaises(RuntimeError):
            self.check(dense,candidate,trusted)


if __name__ == '__main__':
    unittest.main()
