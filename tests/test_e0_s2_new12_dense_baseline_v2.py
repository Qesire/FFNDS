import copy
import unittest
from tools import e0_s2_new12_dense_baseline_v2 as m


class CurrentAuthorityControls(unittest.TestCase):
    def state(self):
        return {'user_continuation_authorization_20261027': {'new_scope_registration_allowed': True},
            'current_session_resource_constraint_20261028': {'mode': 'DYNAMIC_MAX_ACTUALLY_CALLABLE', 'max_simultaneously_assigned_idle_GPUs': None},
            'experiment_authority': {'current_user_authorized_scopes': [{'assigned_GPU_slots': 1}, {'assigned_GPU_slots': 1}]}}

    def test_current_dynamic_authority_and_actual_two_slots(self):
        m.resource_authority(self.state())

    def test_extra_assigned_slot_is_rejected_for_this_pair(self):
        a=self.state();a['experiment_authority']['current_user_authorized_scopes'].append({'assigned_GPU_slots': 1})
        with self.assertRaises(ValueError): m.resource_authority(a)

    def test_inactive_continuation_rejected(self):
        a=self.state();a['user_continuation_authorization_20261027']['new_scope_registration_allowed']=False
        with self.assertRaises(ValueError): m.resource_authority(a)

    def test_versioned_identity_and_same_physical_budget(self):
        self.assertTrue(m.scope_id(1).endswith('PART1_V2'))
        self.assertIn('baseline_v2_',m.output(1))
        self.assertEqual(m.core.limits(0)['video_generations'],7)
        self.assertEqual(m.core.limits(0)['historical_state_byte_checks'],10)


if __name__ == '__main__': unittest.main()
