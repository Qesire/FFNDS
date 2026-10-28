"""Small CPU-only regression for repairing the actual paired CORE seam."""
import copy
from pathlib import Path
import tempfile
import unittest

from tools import raw_bf16_video20_prefix_core_repair as repair
from tools import ffnds_video_metrics as metrics


class PrefixCoreRepairControls(unittest.TestCase):
    def test_actual_81_frame_paired_core_and_missing_reference(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest('OpenCV and NumPy required for real CPU codec control')
        cv2.setNumThreads(1)
        with tempfile.TemporaryDirectory() as td:
            candidate,dense=(Path(td)/'candidate.mp4',Path(td)/'dense.mp4')
            for path,offset in ((candidate,4),(dense,0)):
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),16,(32,32))
                if not writer.isOpened():self.skipTest('MP4 encoder unavailable')
                for t in range(81):
                    frame=np.full((32,32,3),60+offset,dtype=np.uint8)
                    x=2+t%20;cv2.rectangle(frame,(x,9),(x+6,18),(210,210,210),-1);writer.write(frame)
                writer.release()
            case={'image':{'sha256':'image'},'prompt_sha256':'prompt'}
            plan={'models':[]};old={'raw_model_setup_seconds':.5}
            dense_meta={'mode':'dense','image_sha256':'image','prompt_sha256':'prompt','seed':42}
            gen={'mode':'shared','protocol_sha256':'protocol','generation_wall_seconds_including_encoders_decode_observers':1.,
                'FFN_boundary_CUDA_seconds':.2,'peak_cuda_allocated_bytes':1024}
            ctx=repair.context(plan,old,case,gen,{'sha256':'candidateGEN'},dense_meta,{'sha256':'denseGEN'})
            report=metrics.evaluate(candidate,dense,ctx)
            self.assertEqual(metrics.validate(report),[])
            self.assertEqual(report['metrics']['artifact']['decoded_frames'],81)
            self.assertEqual(report['reference_identity']['run_sha256'],'denseGEN')
            self.assertEqual(report['reference_identity']['source_id'],'image')
            bad=copy.deepcopy(report);bad['reference_identity']=None
            with self.assertRaises(AttributeError):metrics.validate(bad)
            missing=copy.deepcopy(report);missing['reference_identity'].pop('run_sha256')
            self.assertIn('reference run receipt identity absent',metrics.validate(missing))
            wrong=copy.deepcopy(report);wrong['reference_identity']['input_prompt_digest']='different'
            self.assertIn('reference does not match source/seed/prompt: input_prompt_digest',metrics.validate(wrong))


if __name__=='__main__':unittest.main()
