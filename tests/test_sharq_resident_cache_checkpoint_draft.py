import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
from sharq_resident_cache_checkpoint_draft import cache_checkpoint
class Tensor:
 def __init__(self,p):self.p=p;self.shape=(1,);self.dtype='bf16';self.device='cuda'
 def numel(self):return 1
 def element_size(self):return 2
 def data_ptr(self):return self.p
 def stride(self):return (1,)
class T(unittest.TestCase):
 def test_process_identity_and_no_content_claim(self):
  c={(e,b,s):tuple(Tensor(1) for _ in range(4)) for e in (0,1) for b in range(40) for s in ('ffn.0','ffn.2')};a=cache_checkpoint(c);self.assertEqual(a['entries'],160);self.assertEqual(a['tensor_content_identity'],'NOT_PROVED_BY_METADATA');self.assertEqual(a,cache_checkpoint(c));next(iter(c.values()))[0].p=2;self.assertNotEqual(a,cache_checkpoint(c));c.pop(next(iter(c)))
  with self.assertRaises(ValueError):cache_checkpoint(c)
if __name__=='__main__':unittest.main()
