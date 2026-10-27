import importlib.util, unittest
from pathlib import Path
import numpy as np
import cv2
p=Path(__file__).resolve().parents[1]/'tools/ub1_disjoint20_visual_media.py';s=importlib.util.spec_from_file_location('media',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class MediaTests(unittest.TestCase):
 def fixture(self):
  arm={'source_id':'s','seed':42,'prompt_sha256':'p','frames':81,'fps':16,'size':[832,464]};return {'index':0,'seed':42,'source_id':'s','prompt_sha256':'p','arms':{'dense':dict(arm),'candidate':dict(arm),'neutral':dict(arm,initial_noise_status='NOT_SAVED_HISTORICALLY_NO_BYTE_PAIRING_CLAIM')}}
 def test_three_columns_and_shapes(self):
  frames=[np.full((464,832,3),x,dtype=np.uint8) for x in (10,60,110)];out=m.compose(frames,256,cv2,np);self.assertEqual(out.shape,(143,768,3));self.assertEqual([int(out[-1,x,0]) for x in (20,276,532)],[10,60,110])
  with self.assertRaises(ValueError):m.compose(frames[:2],256,cv2,np)
 def test_provenance_and_geometry_rejections(self):
  source=self.fixture();self.assertTrue(m.validate_source(source));source['arms']['candidate']['seed']=43
  with self.assertRaises(ValueError):m.validate_source(source)
 def test_valid_receipt_for_another_source_is_rejected(self):
  source=self.fixture();source['arms']['dense'].update(path='dense.mp4',sha256='video',initial_noise={'path':'noise','sha256':'noise_sha'})
  receipt={'status':'PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE','source_index':0,'source_id':'s','image_sha256':'s','prompt_sha256':'p','seed':42,'mode':'dense','initial_noise':source['arms']['dense']['initial_noise'],'video':{'sha256':'video','path':'/tmp/hf/dense.mp4'},'FFN_boundary_timings':[{'step':s,'block':b} for s in range(4) for b in range(40)],'native_counter_delta':{'linear_forward_calls':1600,'activation_qdq_calls':1600}}
  self.assertTrue(m.validate_generation(source,'dense',receipt,Path('/tmp/hf')))
  receipt['source_index']=1
  with self.assertRaises(ValueError):m.validate_generation(source,'dense',receipt,Path('/tmp/hf'))
  source=self.fixture();source['arms']['neutral']['initial_noise_status']='PAIRED'
  with self.assertRaises(ValueError):m.validate_source(source)
  source=self.fixture();source['arms']['dense']['frames']=80
  with self.assertRaises(ValueError):m.validate_source(source)
if __name__=='__main__':unittest.main()
