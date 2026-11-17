import importlib.util,json,tempfile,unittest
from pathlib import Path
p=Path(__file__).with_name('wan_timing_provenance_audit.py');s=importlib.util.spec_from_file_location('timing_audit_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class TimingTest(unittest.TestCase):
 def test_missing_or_mismatched_protocol_never_promotes_ratios(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   def write(name,value):
    path=root/name;path.write_text(json.dumps(value));return {'path':str(path),'sha256':m.sha(path)}
   def receipt(arm,wall):
    return {'source':'vbi_1013','arm':arm,'wall_seconds':wall,'ffn_stream_elapsed_ms':10,'ffn_host_call_seconds':2,'peak_gpu_reserved':10,'peak_gpu_allocated':9,'peak_host_rss_bytes':11,'dispatch':[1]*4,'sites':[1]*160,'timing_is_plain_uninstrumented':True}
   a=write('dense.json',receipt('Dense',10)); b=write('chip.json',receipt('Chip',5))
   result=m.analyze({'receipts':[{'receipt':a},{'receipt':b}]})
   self.assertIsNone(result['paired_comparisons'][0]['wall_ratio_to_dense'])
   self.assertEqual(result['rows'][0]['host_rss_scope'],'process_lifetime_high_water_mark')
   contract={'hardware':'A100','torch':'2.10','cuda':'12','schedule_sha256':'a'*64,'decode_sha256':'b'*64,'measurement':'whole-call'}
   pa=write('a.json',{'receipt_sha256':a['sha256'],'protocol_id':'p1','observer_instrumented':False,'comparison_contract':contract})
   pb=write('b.json',{'receipt_sha256':b['sha256'],'protocol_id':'p1','observer_instrumented':False,'comparison_contract':contract})
   result=m.analyze({'receipts':[{'receipt':a,'protocol':pa},{'receipt':b,'protocol':pb}]})
   self.assertEqual(result['paired_comparisons'][0]['wall_ratio_to_dense'],.5)
   (root/'chip.json').write_text('{}')
   with self.assertRaises(ValueError):m.analyze({'receipts':[{'receipt':b}]})

if __name__=='__main__':unittest.main()
