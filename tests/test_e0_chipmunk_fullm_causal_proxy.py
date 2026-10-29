import unittest
from tools import e0_chipmunk_fullm_causal_proxy as m
class Boundaries(unittest.TestCase):
    def test_unfinalized_cannot_run(self):
        with self.assertRaisesRegex(ValueError,'unfinalized draft'):
            m.authorize({'assets_finalized':False,'id':'E0_CHIPMUNK_FULLM_CAUSAL_PROXY_20261030_PART0'},'0'*64)
if __name__=='__main__':unittest.main()
