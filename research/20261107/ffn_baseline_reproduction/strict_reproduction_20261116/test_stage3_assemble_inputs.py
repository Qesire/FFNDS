import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from stage3_assemble_inputs import DEV,NEUTRALS,SITES,build,sha
class AssemblyTest(unittest.TestCase):
 def test_published_manifest_schema(self):
  root=Path(__file__).resolve().parents[4]
  actual=root/'research/experiment_reviews/FFNDS_STRICT_STAGE1_DENSE_FULL_U_V2/MANIFEST.json'
  d=json.loads(actual.read_text(encoding='utf8'))
  rows=[x for x in d['files'] if x['kind'] in ('full_u','panel')]
  self.assertEqual(len(rows),48)
  self.assertTrue(all('path' in x and 'file' not in x for x in rows))
  for x in rows:
   if x['kind']=='panel':
    self.assertTrue(all(k in x['event'] for k in ('source','step','layer','expert','image_sha256','prompt_sha256','source_cluster_id','checkpoint_sha256','token_id_sha256_le_int64')))
 def test_files_and_hash(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t); bank=r/'bank'; bank.mkdir(); bm=r/'bank.json'; bm.write_text('{}'); token='a'*64
   b={'bank':{'bank_root':str(bank),'manifest_path':str(bm),'manifest_sha256':sha(bm)},'panel':{'per_source':{s:{'token_id_sha256_le_int64':token} for s in DEV}}}; bp=r/'b.json'; bp.write_text(json.dumps(b))
   for s in DEV:
    if s in NEUTRALS: continue
    d=bank/s; d.mkdir()
    for e,l in SITES:
     for step in ((0,1) if e=='high' else (2,3)):
      name=f'{s}__seed42__s{step:02d}__b{l:02d}.pt'; f=d/name; f.write_bytes(name.encode())
      Path(str(f)+'.json').write_text(json.dumps({'source':s,'seed':42,'step':step,'layer':l,'expert':e,'file':name,'token_id_sha256_le_int64':token,'file_sha256':sha(f),'image_sha256':'image','prompt_sha256':'prompt','source_cluster_id':s,'checkpoint_sha256':e}))
   cap=r/'capture';cap.mkdir();files=[]
   for s in NEUTRALS:
    for e,l in SITES:
     for step in ((0,1) if e=='high' else (2,3)):
      for k in ('full_u','panel'):
       name=f'{s}_{step}_{l}_{k}.pt'; f=cap/name; f.write_bytes(name.encode()); row={'kind':k,'path':name,'sha256':sha(f)}
       if k=='full_u': row.update(source=s,step=step,layer=l)
       else: row.update(event={'source':s,'step':step,'layer':l,'expert':e,'token_id_sha256_le_int64':token,'image_sha256':'image','prompt_sha256':'prompt','source_cluster_id':s,'checkpoint_sha256':e},u_panel_readback_equal=True)
       files.append(row)
   mf=cap/'MANIFEST.json';mf.write_text(json.dumps({'schema':'FFNDS_FULL_U_STAGE1_V1','files':files})); c={'initial_noise_equal':True,'decoded_equal':True,'flows_equal':[True]*4,'latents_equal':[True]*4,'time_sigma_equal':[True]*4}; sf=r/'SUMMARY.json';sf.write_text(json.dumps({'status':'COMPLETE','d4_completed':4,'per_source':{s:c for s in NEUTRALS}}))
   a=SimpleNamespace(mode='dev',bindings=str(bp),bindings_sha256=sha(bp),capture_manifest=str(mf),capture_manifest_sha256=sha(mf),capture_summary=str(sf),capture_summary_sha256=sha(sf))
   result=build(a);self.assertEqual(len(result['input_pairs']),66);self.assertEqual(sum(x['anchor']['kind']=='stage1_full_u' for x in result['input_pairs']),12)
   (cap/files[0]['path']).write_bytes(b'tampered')
   with self.assertRaises(ValueError):build(a)
if __name__=='__main__':unittest.main()



