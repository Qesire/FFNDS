"""Small stdlib controls: no real factor payload, GPU or model."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import e0_raw_bf16_full160_video20 as m


class VideoControls(unittest.TestCase):
    def test_parts_order_and_exact_operation_counts(self):
        self.assertNotEqual(m.scope_id(0),m.scope_id(1))
        with self.assertRaises(ValueError): m.scope_id(True)
        for part in (0,1):
            cases=[{'index':i,'native_source_ordinal':100+i} for i in range(part,20,2)]
            orders=[m.phase_order(c) for c in cases]
            self.assertEqual(sorted(sum([list(o) for o in orders],[])),sorted(list(m.MODES)*10))
            self.assertEqual(sorted(orders.count(o) for o in set(orders)),[3,3,4])
            self.assertEqual(len({m.run_id(c,a) for c in cases for a in m.MODES}),30)
            limits=m.limits(part)
            for key,value in {'video_generations':30,'native_full_model_flows':120,'native_FFN_calls':4800,
                              'BF16_factor_GEMM':6400,'original_checkpoint_block_buffer_loads':4860,
                              'matched_triad_state_byte_checks':40,'historical_state_byte_checks':20 if part==0 else 0}.items():
                self.assertEqual(limits[key],value)

    def test_coordinates_and_invalid_family(self):
        self.assertEqual(m.bank_coordinates({'expert':1,'block':39,'family':'shared'}),[(1,2,39),(1,3,39)])
        self.assertEqual(m.bank_coordinates({'expert':0,'block':13,'family':'step1'}),[(0,1,13)])
        for row in ({'expert':True,'block':0,'family':'shared'}, {'expert':0,'block':40,'family':'shared'}, {'expert':0,'block':0,'family':'step2'}):
            with self.assertRaises(ValueError):m.bank_coordinates(row)

    def test_triad_equal_values_no_alias_and_wrong_state(self):
        state={'sha256':'s','bytes':4,'raw_tensor_sha256':'r','shape':[2],'dtype':'BF16'}
        phases=[{'mode':mode,'native_source_ordinal':123,'seed':42,'completed_native_flow_steps':4,
                 'states':{'raw_noise_FP32':copy.deepcopy(state),'initial_noise':copy.deepcopy(state)}} for mode in m.MODES]
        self.assertTrue(m.verify_triad(phases,{'native_source_ordinal':123})['raw_FP32_and_initial_BF16_same_bytes_all_arms'])
        phases[1]['states']['initial_noise']['raw_tensor_sha256']='other'
        with self.assertRaises(ValueError):m.verify_triad(phases,{'native_source_ordinal':123})
        phases[1]['states']['initial_noise']['raw_tensor_sha256']='r';phases[2]['mode']='shared'
        with self.assertRaises(ValueError):m.verify_triad(phases,{'native_source_ordinal':123})

    def test_factor_header_and_raw_hash(self):
        shapes={'left.weight':[1,2],'right.weight':[2,1],'right.bias':[2]}
        raw={k:bytes([i+1])*4 for i,k in enumerate(shapes)};header={};meta={};offset=0
        for k in shapes:
            header[k]={'dtype':'BF16','shape':shapes[k],'data_offsets':[offset,offset+4]};offset+=4
            meta[k]={'shape':shapes[k],'dtype':'bfloat16','raw_tensor_sha256':hashlib.sha256(raw[k]).hexdigest()}
        def file_data(h,body):
            encoded=json.dumps(h,separators=(',',':')).encode();return struct.pack('<Q',len(encoded))+encoded+body
        with tempfile.TemporaryDirectory() as d, patch.object(m,'SHAPES',shapes),patch.object(m,'FACTOR_BYTES',12):
            path=Path(d)/'tiny';body=b''.join(raw.values());path.write_bytes(file_data(header,body))
            result=m.factor_header(path,meta,payload=True);self.assertEqual(result['actual_payload_read_bytes'],12)
            self.assertEqual(result['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            path.write_bytes(file_data(header,body[:-1]+b'!'))
            with self.assertRaises(ValueError):m.factor_header(path,meta,payload=True)
            bad=copy.deepcopy(header);bad['right.weight']['data_offsets']=[5,9];path.write_bytes(file_data(bad,body))
            with self.assertRaises(ValueError):m.factor_header(path,meta,payload=True)

    def test_ledger_predebit_failed_work(self):
        report={};events=[]
        with patch.object(m.native,'RUN_IDS',{'run'}),patch.object(m.native,'LIMITS',{'probe':1}):
            ledger=m.Ledger(report,events.append)
            with self.assertRaises(RuntimeError):ledger.call('run',{'probe':1},lambda: (_ for _ in ()).throw(RuntimeError('failure')))
            self.assertEqual(report['predebited'],{'probe':1});self.assertEqual(report['observed'],{})
            with self.assertRaises(ValueError):ledger.call('run',{'probe':1},lambda:None)
            self.assertEqual(events[0]['kind'],'reserved_before_call')

    def test_stdlib_import_no_numeric_modules(self):
        code="import sys;sys.path.insert(0,'tools');import e0_raw_bf16_full160_video20;assert not {'torch','numpy'} & set(sys.modules)"
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


if __name__=='__main__':unittest.main()
