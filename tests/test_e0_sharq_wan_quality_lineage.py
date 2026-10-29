import sys,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_sharq_wan_quality_lineage as m
class Tests(unittest.TestCase):
 def test_rejects_invalid_part_before_asset_reads(self):
  for value in (-1,2,True,None):
   with self.assertRaises(ValueError):m.parent_data(Path('/tmp'),value)
 def test_rejects_unclosed_parent(self):
  with patch.object(m.owner,'validate',side_effect=ValueError('not closed')):
   with self.assertRaises(ValueError):m.parent_data(Path('/tmp'),0)
 def test_selects_first_bf16_and_two_candidates_only(self):
  with tempfile.TemporaryDirectory() as tmp:
   hf=Path(tmp);phases=[]
   for i,mode in enumerate(('bf16','bf16_cache','dense32','sharq')):
    f=hf/(str(i)+'.mp4');f.write_bytes(b'x');phases.append({'mode':mode,'video':{'path':str(f),'bytes':1,'sha256':str(i)*64},'states':{},'run_id':str(i)})
   proof={'protocol':{'sha256':'p'},'RUN':{'sha256':'r'},'exit':{'sha256':'e'},'verification_RUN':{},'verification_exit':{}}
   parent={'id':'parent','case':{'prompt':'x'}};run={'phases':phases,'current_phase':{'status':'EVIDENCE_COMPLETE'},'current_state':None,'operation_ledger':{side:{'sharq_stream_bridge_calls':800,'sharq_bridge_event_records':1600,'sharq_bridge_event_waits':1600} for side in ('predebited','observed')}};verification={'RUN':proof['RUN'],'exit':proof['exit'],'parent_scope_id':'parent','completed_bytes':3,'predebited_bytes':3,'inflight':'NONE'};vals=iter((parent,run,{},verification))
   with patch.object(m,'parent_data',return_value=proof),patch.object(m.native,'bound',return_value='unused'),patch.object(m.native,'read',side_effect=lambda _:next(vals)),patch.object(m.owner.evidence,'verify_record'):
    rows=m.closed_rows(proof,hf,0)
   self.assertEqual([r['run_id'] for r in rows],['0','2','3']);self.assertEqual([r['mode'] for r in rows],list(m.MODES))
 def test_partial_phase_or_incomplete_bridge_cannot_be_scored(self):
  import copy
  with tempfile.TemporaryDirectory() as tmp:
   hf=Path(tmp);phases=[]
   for i,mode in enumerate(('bf16','bf16_cache','dense32','sharq')):
    f=hf/(str(i)+'.mp4');f.write_bytes(b'x');phases.append({'mode':mode,'video':{'path':str(f),'bytes':1,'sha256':str(i)*64},'states':{},'run_id':str(i)})
   proof={'protocol':{'sha256':'p'},'RUN':{'sha256':'r'},'exit':{'sha256':'e'},'verification_RUN':{},'verification_exit':{}}
   parent={'id':'parent','case':{}};ledger={side:{'sharq_stream_bridge_calls':800,'sharq_bridge_event_records':1600,'sharq_bridge_event_waits':1600} for side in ('predebited','observed')};run={'phases':phases,'current_phase':{'status':'EVIDENCE_COMPLETE'},'current_state':None,'operation_ledger':ledger};verification={'RUN':proof['RUN'],'exit':proof['exit'],'parent_scope_id':'parent','completed_bytes':3,'predebited_bytes':3,'inflight':'NONE'}
   for failure in ('partial_phase','inflight_state','bridge_short'):
    bad=copy.deepcopy(run)
    if failure=='partial_phase':bad['current_phase']['status']='STARTED'
    if failure=='inflight_state':bad['current_state']={'status':'STARTED'}
    if failure=='bridge_short':bad['operation_ledger']['observed']['sharq_stream_bridge_calls']=799
    vals=iter((parent,bad,{},verification))
    with patch.object(m,'parent_data',return_value=proof),patch.object(m.native,'bound',return_value='unused'),patch.object(m.native,'read',side_effect=lambda _:next(vals)),patch.object(m.owner.evidence,'verify_record'):
     with self.assertRaises(ValueError):m.closed_rows(proof,hf,0)
 def test_fixed_modes_and_sources(self):
  self.assertEqual(m.MODES,('dense','dense32','sharq'));self.assertEqual(m.ORDINALS,(1040,979))
if __name__=='__main__':unittest.main()
