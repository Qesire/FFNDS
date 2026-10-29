import unittest
from tools import e0_chipmunk_p4_causal_proxy as p4

class AuthorizationTests(unittest.TestCase):
    def test_draft_rejects_before_active_read(self):
        with self.assertRaisesRegex(ValueError,'draft not executable'):
            p4.authorize({'id':p4.ID,'assets_finalized':False},'0'*64)
    def test_wrong_scope_rejects(self):
        with self.assertRaises(ValueError):p4.authorize({'id':'OTHER','assets_finalized':True},'0'*64)
if __name__=='__main__':unittest.main()
