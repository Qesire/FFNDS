import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import raw_bf16_video20_sources as source

class CohortTest(unittest.TestCase):
    def test_fixed_cohort_and_rejections(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);hf=root/'hf';hf.mkdir();consumer=[];formal=[]
            def row(ordinal,dim):
                data=f'image-{ordinal}'.encode();p=hf/f'{ordinal}.jpg';p.write_bytes(data);prompt=f'prompt {ordinal}'
                return {'source_index':ordinal,'dimension':dim,'source_group':f'{dim}:{ordinal}','prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'image_path':str(p),'input_provenance':{'image_sha256':hashlib.sha256(data).hexdigest()}}
            for i in range(894,904):consumer.append(row(i,'i2v_subject'))
            for i in range(240,313,8):consumer.append(row(i,'i2v_background'))
            for i in (882,885,888):formal.append(row(i,'i2v_subject'))
            for i in (96,128,144):formal.append(row(i,'i2v_background'))
            confirmation=[row(700,'i2v_background')]
            banks={'consumer20':'consumer.json','formal20':'formal.json','confirmation20':'confirmation.json'}
            for k,rows in zip(banks,(consumer,formal,confirmation)):(root/banks[k]).write_text(json.dumps(rows))
            with patch.object(source,'ROOT',root),patch.object(source,'BANKS',banks):
                c=source.make_cohort(hf);self.assertTrue(source.validate_cohort(c,hf,True))
                self.assertEqual([x['stratum'] for x in c['sources']].count('camera'),3)
                self.assertEqual([sum(x['part']==i for x in c['sources']) for i in (0,1)],[10,10])
                for mode in ('prompt','duplicate','part','ordinal','camera'):
                    x=copy.deepcopy(c)
                    if mode=='prompt':x['sources'][0]['prompt']='changed'
                    elif mode=='duplicate':x['sources'][1]['image']['sha256']=x['sources'][0]['image']['sha256']
                    elif mode=='part':x['sources'][0]['part']=1
                    elif mode=='ordinal':x['sources'][0]['native_source_ordinal']=240
                    else:x['sources'][-1]['camera_instruction']='other'
                    with self.assertRaises(ValueError):source.validate_cohort(x,hf)
                (hf/'894.jpg').write_bytes(b'changed')
                with self.assertRaises(ValueError):source.validate_cohort(c,hf,True)

if __name__=='__main__':unittest.main()
