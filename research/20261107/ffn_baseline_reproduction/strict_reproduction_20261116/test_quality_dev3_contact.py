"""Synthetic contact sheet geometry/frame-selection engineering test."""
import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image
from quality_dev3_contact import create,sha,DEV3,FRAMES
from quality_pair_report import ARMS
class ContactTest(unittest.TestCase):
 def test_frames_and_missing_cells(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); exe=root/'ffmpeg';exe.write_bytes(b'synthetic');records=[]
   for i,source in enumerate(DEV3):
    video=root/f'v{i}';video.write_bytes(f'v{i}'.encode());records.append({'source':source,'arm':'Dense','video_path':str(video),'video_sha256':sha(video),'generation_proof_sha256':'a'*64})
   scope={'sources':DEV3,'arms':ARMS,'frame_indices':FRAMES,'ffmpeg_path':str(exe),'ffmpeg_sha256':sha(exe),'records':records,'output':str(root/'out')}
   data=bytes(832*464*3*5)
   def fake(command,**kwargs):
    self.assertIn('select=eq(n\\,0)+eq(n\\,20)+eq(n\\,40)+eq(n\\,60)+eq(n\\,80)',command)
    return SimpleNamespace(stdout=data,stderr=b'')
   with patch('quality_dev3_contact.subprocess.run',fake):
    out=create(scope,authorize=lambda x:None,verify_generation=lambda r,s:{'status':'PASS','proof_sha256':'a'*64},validate_video=lambda p,s:None)
   self.assertEqual(len(out['sheets']),3);self.assertEqual(len(out['sheets'][0]['missing_arms']),4)
   with Image.open(root/'out'/out['sheets'][0]['file']) as image:self.assertEqual(image.size,(1165,735))
if __name__=='__main__':unittest.main()


