import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('producer', ROOT / 'tools/prepare_local_transformers_mirror.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_authorization_unique_and_revocation(self):
        plan = {'id': m.ID, 'assets_finalized': True, 'limits': m.LIMITS, 'transformers_version': '5.6.2', 'automatic_retry': False}
        scope = {'id': m.ID, 'protocol_sha256': 'abc', 'execution_status': 'AUTHORIZED_NOT_STARTED'}
        active = {'user_continuation_authorization_20261027': {'id': 'USER_RESUME_OFFLINE_IFC_20261027', 'status': 'ACTIVE_CURRENT_USER_DIRECTED_SESSION'}, 'experiment_authority': {'current_user_authorized_scopes': [scope]}}
        m.authorize(plan, 'abc', active)
        active['experiment_authority']['current_user_authorized_scopes'].append(scope)
        with self.assertRaises(ValueError): m.authorize(plan, 'abc', active)
        active['experiment_authority']['current_user_authorized_scopes'].pop()
        active['user_continuation_authorization_20261027']['status'] = 'PAUSED'
        with self.assertRaises(ValueError): m.authorize(plan, 'abc', active)

if __name__ == '__main__': unittest.main()
