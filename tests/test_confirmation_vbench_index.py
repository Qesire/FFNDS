"""A mixed-category cohort must preserve real types and score every source."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from confirmation_vbench_index import build_full_info,validate_video_name_coverage,DIMS
from vbench_input_index import build_full_info as original_index


class ConfirmationIndexTests(unittest.TestCase):
    def fixtures(self):
        samples=[];historical=[]
        for dimension,oldkind,newkinds in [('i2v_subject','animal',['food','transportation']),('i2v_background','architecture',['indoor','scenery'])]:
            for i in range(10):
                prompt=f'new {dimension} {i}'
                samples.append({'source_group':f'{dimension}:{100+i}','dimension':dimension,'prompt':prompt,
                    'image_name':prompt+'.jpg','image_type':newkinds[i%2]})
                historical.append({'prompt_en':f'old {dimension} {i}','dimension':[dimension,*DIMS],
                    'image_type':oldkind,'image_name':f'old{i}.jpg'})
        return samples,historical

    def test_mixed_types_preserved_with_full_three_metric_filename_coverage(self):
        samples,historical=self.fixtures()
        with self.assertRaisesRegex(ValueError,'Source type differs'):
            original_index(samples,historical)
        result=build_full_info(samples,historical)
        self.assertEqual([v['image_type'] for v in result],[v['image_type'] for v in samples])
        names=[f'{v["prompt"]}-{i}.mp4' for v in samples for i in range(5)]
        self.assertEqual(validate_video_name_coverage(result,names),{dim:100 for dim in DIMS})
        with self.assertRaises(ValueError): validate_video_name_coverage(result,names[:-1])

    def test_missing_metric_tag_and_camera_source_are_rejected(self):
        samples,historical=self.fixtures()
        samples[0]['dimension']='i2v_camera'
        with self.assertRaises(ValueError):build_full_info(samples,historical)
        samples,historical=self.fixtures()
        for row in historical[10:]:row['dimension'].remove('motion_smoothness')
        with self.assertRaisesRegex(ValueError,'all three'):build_full_info(samples,historical)


if __name__=='__main__':unittest.main()
