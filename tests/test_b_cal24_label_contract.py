import copy
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import b_cal24_label_contract as c


def sha(b): return hashlib.sha256(b).hexdigest()


def fixture(root):
    inputs, sources = [], []
    (root/'operand').write_bytes(b'x')
    for i in range(24):
        image = ('image%d' % i).encode(); (root/f'image{i}').write_bytes(image)
        prompt = f'prompt{i}'; group = f'group{i}'
        source = {'index': i, 'source_id': sha(image), 'source_group': group, 'dimension': 'subject',
                  'seed':42+i, 'semantic_source_id': f'source{i}'}
        scopes = [{'step':step, 'block':block, 'expert':'high_noise_model' if step < 2 else 'low_noise_model',
                   'grid':[21,29,52], 'rows':256, 'selected_rows':list(range(256)), 'sampling_seed': 1,
                   'path':'operand','sha256':sha(b'x'),'bytes':1} for step in range(4) for block in range(40)]
        trajectory = [{'step':step,'sha256':sha(bytes([step]))} for step in range(4)]
        item = {'status':'PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP','index':i,
                'input':{'calibration_seed':42+i,'source_group':group,'dimension':'subject','prompt':prompt,
                         'img_paths':[str(root/f'image{i}')], 'input_provenance':{'image_sha256':sha(image),'prompt_sha256':sha(prompt.encode())}},
                'scopes':scopes, 'trajectories':{'plain':trajectory,'observed':trajectory}}
        raw=json.dumps(item).encode(); (root/f'input{i}.json').write_bytes(raw)
        inputs.append({'index':i,'path':f'input{i}.json','bytes':len(raw),'sha256':sha(raw)}); sources.append(source)
    return {'inputs':inputs,'sources':sources,'source_split':{'training':list(range(16)),'diagnostic_heldout':list(range(16,24))}}


def write_header(path, case, change=None):
    meta={k:str(case[k]) for k in c.META_KEYS}; meta['rows_sha256']=c.rows_sha256(case['rows'])
    for k in c.EVIDENCE_SHA_KEYS: meta[k]='a'*64
    meta['actual_post_sha256']=case['historical_post_sha256']
    for k in c.EVIDENCE_BOOL_KEYS: meta[k]='true'
    meta.update(head_weight_stride='[1,5120]',head_bias_stride='[1]',head_modulation_stride='[10240,5120,1]',head_shift_stride='[10240,5120,1]',head_scale_stride='[10240,5120,1]')
    header={'__metadata__':meta}; pos=0
    for name,(dtype,shape) in c.PACK_SCHEMA.items():
        count=1
        for n in shape: count*=n
        size=count*(8 if dtype=='I64' else 2); header[name]={'dtype':dtype,'shape':shape,'data_offsets':[pos,pos+size]};pos+=size
    if change: change(header)
    raw=json.dumps(header).encode()
    with path.open('wb') as f:
        f.write(struct.pack('<Q',len(raw)));f.write(raw);f.truncate(8+len(raw)+pos)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.protocol=fixture(self.root);self.cases=c.build_cases(self.protocol,self.root)

    def test_fixture_and_split(self):
        self.assertEqual(len(self.cases),96);self.assertEqual([sum(x['part']==i for x in self.cases) for i in range(4)],[24]*4)
        self.assertEqual(sum(x['role']=='TRAIN' for x in self.cases),64)
        self.assertEqual(self.cases[0]['rows'],list(range(256)))

    def test_duplicate_source_rejected(self):
        p=copy.deepcopy(self.protocol);p['sources'][1]['source_id']=p['sources'][0]['source_id']
        with self.assertRaises(ValueError): c.build_cases(p,self.root)

    def test_wrong_seed_rejected(self):
        p=copy.deepcopy(self.protocol);p['sources'][0]['seed']=43
        with self.assertRaises(ValueError): c.build_cases(p,self.root)

    def test_case_expert_and_rows_rejected(self):
        for key,value in [('expert','low_noise_model'),('rows',[0]*256)]:
            cases=copy.deepcopy(self.cases);cases[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): c.Ledger(cases,0)

    def test_part_boundary_and_duplicate_case(self):
        ledger=c.Ledger(self.cases,0)
        with self.assertRaises(ValueError): ledger.mark_source(6)
        ledger.mark_source(0);ledger.mark_case(self.cases[0])
        with self.assertRaises(ValueError): ledger.mark_case(self.cases[0])
        with self.assertRaises(ValueError): ledger.mark_case(self.cases[24])

    def test_predebit_then_observe(self):
        ledger=c.Ledger(self.cases,0)
        with self.assertRaises(ValueError): ledger.observe('dense_flows')
        ledger.predebit('dense_flows');ledger.observe('dense_flows')
        with self.assertRaises(ValueError): ledger.observe('dense_flows')

    def test_overbudget_rejected(self):
        ledger=c.Ledger(self.cases,0)
        with self.assertRaises(ValueError): ledger.predebit('dense_flows',25)
        with self.assertRaises(ValueError): ledger.predebit('decoder_calls')

    def test_exact_ledger_completion(self):
        ledger=c.Ledger(self.cases,0)
        for i in range(6):ledger.mark_source(i)
        for case in self.cases[:24]:ledger.mark_case(case)
        for key,value in ledger.limits.items():ledger.predebit(key,value);ledger.observe(key,value)
        self.assertEqual(ledger.finish()['case_count'],24)

    def test_header_region_and_gate_shape(self):
        p=self.root/'test.safetensors';write_header(p,self.cases[0]);c.validate_label_header(p,self.cases[0])
        write_header(p,self.cases[0],lambda h:h['gate'].update(shape=[5120]))
        with self.assertRaises(ValueError):c.validate_label_header(p,self.cases[0])
        write_header(p,self.cases[0],lambda h:h['gate']['data_offsets'].__setitem__(0,1))
        with self.assertRaises(ValueError):c.validate_label_header(p,self.cases[0])

    def test_header_seed_expert_rows_and_evidence(self):
        p=self.root/'test.safetensors'
        for key,value in [('seed','43'),('expert','low_noise_model'),('rows_sha256','0'*64),('original_y_byte_equal','false'),('actual_post_sha256','0'*64),('head_weight_raw_sha256','x'),('head_weight_stride','[1]')]:
            def mutate(h):h['__metadata__'][key]=value
            write_header(p,self.cases[0],mutate)
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate_label_header(p,self.cases[0])


if __name__=='__main__':unittest.main()
