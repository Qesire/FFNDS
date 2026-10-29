import pathlib,sys,unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_fullm_packed_native as d
class EntryTests(unittest.TestCase):
    def test_draft_never_authorizes(self):
        with self.assertRaisesRegex(ValueError,'unfinalized'):
            d.authorize({'id':'E0_CHIPMUNK_FULLM_PACKED_NATIVE_20261030_PART0','assets_finalized':False},'x')
    def test_cpu_receipt_rejects_cuda_initialized(self):
        class A: hf_root=pathlib.Path('/tmp')
        p={'id':'E0_CHIPMUNK_FULLM_PACKED_NATIVE_20261030_PART0'}
        scope={'cpu_preflight':{'RUN':{'sha256':'run'},'EXIT':{},'job_spec_sha256':'job'},'review':{'path':'/dev/null'}}
        run={'id':p['id']+'_CPU_PREFLIGHT','status':'PASS_ISOLATED_ENTRY_CPU_PREFLIGHT','protocol_sha256':'digest','cuda_initialized':True,'gpu_job_sha256':'gpu','job_spec_sha256':'job'}
        ex={'id':p['id']+'_CPU_PREFLIGHT','protocol_sha256':'digest','child_returncode':0,'process_exited':True,'run_sha256':'run','spec_sha256':'job'}
        with patch.object(d,'bound_hf',side_effect=[run,ex]),patch.object(d.json,'loads',return_value={'cpu_job_spec_sha256':'job','job_spec_sha256':'gpu'}):
            with self.assertRaisesRegex(ValueError,'CPU preflight not closed'):d.verify_cpu_receipts(A(),p,'digest',scope)
if __name__=='__main__':unittest.main()
