import copy
import json
import unittest
from pathlib import Path
from tools.raw_bf16_full_depth_sources import make_cohort,validate_sources

def assets_available():
    root=Path(__file__).resolve().parents[1]
    hf=Path('/home/xiang/.cache/huggingface')
    banks=[root/'.runtime/inputs'/name/'canonical.json' for name in ('disjoint20','formal20','diverse_confirmation20')]
    if not hf.is_dir() or not all(p.is_file() for p in banks):
        return False
    try:
        rows=json.loads(banks[0].read_text())
        return all(Path(row['image_path']).is_file() for row in rows)
    except (OSError, ValueError, KeyError, TypeError):
        return False


@unittest.skipUnless(assets_available(), 'real source banks and HF images are not available')
class CohortTest(unittest.TestCase):
    def test_selection_and_mutations(self):
        hf=Path('/home/xiang/.cache/huggingface');c=make_cohort(hf)
        self.assertTrue(validate_sources(c,hf))
        self.assertEqual([x['source_group'] for x in c['sources'][2:]],['i2v_subject:25','i2v_subject:27','i2v_subject:24','i2v_background:36','i2v_background:30','i2v_background:38'])
        for mutate in ('ordinal','prompt','duplicate','pool','part'):
            x=copy.deepcopy(c)
            if mutate=='ordinal': x['sources'][2]['native_source_ordinal']+=1
            elif mutate=='prompt': x['sources'][2]['prompt']='altered'
            elif mutate=='duplicate': x['sources'][2]['image']['sha256']=x['sources'][0]['image']['sha256']
            elif mutate=='pool': x['ordered_candidate_pools']['i2v_subject'].reverse()
            else: x['sources'][2]['part']=1
            with self.assertRaises(ValueError):validate_sources(x,hf)

if __name__=='__main__':unittest.main()
