import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('test_teal_quality_lineage',ROOT/'tools/e0_teal_wan_quality_lineage.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
    def test_source_arm_design(self):
        self.assertEqual(m.ORDINALS,(1040,979,456,923));self.assertEqual(m.MODES,('dense','threshold25','threshold50'))
    def test_parent_exit_required(self):
        with patch.object(m.native,'read',side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):m.parent_data(Path('/tmp'),0)
    def test_source_equal_four_cluster_vectors(self):
        rows=[{'ordinal':o,'scores':{k:{'axis':float(i)} for k in m.MODES}} for i,o in enumerate(m.ORDINALS)]
        value=m.source_equal_vectors(rows,['axis']);self.assertEqual(value['source_equal_means']['dense']['axis'],1.5)
        with self.assertRaises(ValueError):m.source_equal_vectors(rows[:-1],['axis'])
    def test_raft_ratios_respect_zero_dense(self):
        rows=[{'ordinal':o,'means':{'dense':2,'threshold25':1,'threshold50':0}} for o in m.ORDINALS]
        self.assertEqual(m.raft_source_equal(rows)['mean_source_ratios']['threshold25'],0.5)
        rows[0]['means']['dense']=0
        self.assertIsNone(m.raft_source_equal(rows)['mean_source_ratios']['threshold25'])
    def test_wrong_parent_rejected(self):
        with self.assertRaises(ValueError):m.closed_rows({'protocol':{'path':'fake'}},Path('/tmp'),0)
if __name__=='__main__':unittest.main()

class FixtureTests(unittest.TestCase):
 def test_real_small_parent_clips_order_noises_and_inode(self):
  import json,tempfile
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'repo';hf=Path(tmp)/'hf';root.mkdir();hf.mkdir();case={'native_source_ordinal':1040,'source_id':'src','image':{'sha256':'a'*64},'prompt_sha256':'b'*64}
   parent={'part':0,'case':case,'run_ids':['n0','n1','d','a','b']};pp=root/'parent.json';pp.write_text(json.dumps(parent));pb=m.native.binding(pp,root)
   noise={'sha256':'c'*64,'raw_tensor_sha256':'d'*64,'bytes':1,'shape':[1],'dtype':'bfloat16'};states={k:dict(noise) for k in ['raw_noise_FP32','initial_noise']+[f'flow{i}' for i in range(8)]}
   phases=[{'run_id':'n0'},{'run_id':'n1'}]
   for rid,mode in zip(('d','a','b'),m.MODES):
    v=hf/(rid+'.mp4');v.write_bytes(rid.encode());clip={'path':str(v),'bytes':v.stat().st_size,'sha256':m.native.sha(v)}
    phase={'run_id':rid,'mode':mode,'source_id':'src','image_sha256':'a'*64,'prompt_sha256':'b'*64,'seed':42,'all160_FFN_covered':True,'states':states,'video':clip}
    gen=hf/(rid+'.json');gen.write_text(json.dumps(phase));phase['generation_receipt']=m.native.binding(gen,hf);phases.append(phase)
   run={'phases':phases,'same_hardware_old_new_all10_and_video_neutral':True};rp=hf/'RUN.json';rp.write_text(json.dumps(run));ep=hf/'EXIT.json';ep.write_text('{}')
   proof={'protocol':pb,'RUN':m.native.binding(rp,hf),'exit':m.native.binding(ep,hf)}
   with patch.object(m,'ROOT',root),patch.object(m,'PARENT','parent.json'),patch.object(m.video,'verify_closed',return_value={'status':'PASS_CLOSED_TEAL_VIDEO_V3_QUALITY_DIAGNOSTIC'}):
    self.assertEqual(len(m.closed_rows(proof,hf,0)),3)
    (hf/'a.mp4').unlink();(hf/'a.mp4').hardlink_to(hf/'d.mp4')
    # Update generation and run clip bindings so rejection is alias-specific.
    phases[3]['video']=dict(phases[2]['video'],path=str(hf/'a.mp4'));g=hf/'a.json';g.write_text(json.dumps({k:v for k,v in phases[3].items() if k!='generation_receipt'}));phases[3]['generation_receipt']=m.native.binding(g,hf)
    rp.write_text(json.dumps(run));proof['RUN']=m.native.binding(rp,hf)
    with self.assertRaises(ValueError):m.closed_rows(proof,hf,0)
