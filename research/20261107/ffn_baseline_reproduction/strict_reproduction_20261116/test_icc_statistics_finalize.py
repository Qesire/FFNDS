"""Small temporary-data engineering check, no research result is produced."""
import json
import tempfile
import unittest
from pathlib import Path
import torch
from icc_four_boundary_collector import FourBoundaryCollector
from icc_statistics_finalize import finalize, sha

class AggregateTest(unittest.TestCase):
    def test_source_equal_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); dirs = {}; ids = [f's{i}' for i in range(11)]
            for i, source in enumerate(ids):
                c = FourBoundaryCollector(torch,root/source,source,'high','CA',tokens=3,d=2,m=3,layers=1,identity={'test':i})
                c.begin_site('CA',0)
                for name, width in (('u',2),('z',3),('h',3),('y',2)):
                    c.observe(name,torch.arange(1,width+1,dtype=torch.bfloat16).repeat(3,1)*(i+1))
                c.finish_site(); c.finish(); dirs[source]=str(root/source)
            source_file = Path(__file__).with_name('icc_four_boundary_collector.py')
            receipt = root/'temporary_fake_oracle.json'
            receipt.write_text(json.dumps({'status':'PASS','collector_sha256':sha(source_file)}))
            result = finalize(dirs,expert='high',family='CA',expected_sources=ids,output=root/'scales',
                              oracle_receipt=receipt,collector_file=source_file,layers=1,d=2,m=3,tokens=3)
            self.assertEqual(len(result['source_statistics']),11)
            weights = torch.load(root/'scales/scales_00.pt',weights_only=True)
            self.assertTrue(torch.isfinite(weights['u']['capped']).all().item())
            self.assertGreaterEqual(weights['u']['capped'].min().item(),weights['u']['capped'].max().item()/1000 - 1e-7)

if __name__ == '__main__': unittest.main()
