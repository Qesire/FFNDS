import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
import e0_teal_fullm_row_group_support as m

class TestSupport(unittest.TestCase):
    def test_bf16_scalar_mask_matches_torch(self):
        import torch
        source=torch.tensor([[-1.,-.125,-.0625,-0.,0.,.0625,.125,1.]],dtype=torch.bfloat16)
        raw=source.view(torch.int16).numpy().view(np.uint16)
        thresholds=[.0626,.126]
        for got,threshold in zip(m.build_keep(raw,thresholds,np),thresholds):
            np.testing.assert_array_equal(got,(source.abs()>threshold).numpy())
    def test_natural_shuffle_union_and_tail_are_real(self):
        keep=np.zeros((36,64),dtype=np.bool_)
        keep[:16,0]=True;keep[16:32,16]=True;keep[32:,32]=True
        natural=m.group_stats(keep,None,16,np,out_dim=1)
        perm=np.array([i for pair in zip(range(16),range(16,32)) for i in pair]+list(range(32,36)))
        shuffled=m.group_stats(keep,perm,16,np,out_dim=1)
        self.assertEqual(natural['union_columns'],[1,1,1]);self.assertEqual(shuffled['union_columns'],[2,2,1])
        self.assertEqual(natural['tail_rows'],4);self.assertEqual(natural['logical_union_macs'],36)
        self.assertEqual(shuffled['logical_union_macs'],68);self.assertEqual(natural['empty_K16_tiles'],[3,3,3])
        self.assertEqual(natural['empty_K64_tiles'],[0,0,0])
    def test_all_zero_tiles_and_m32_tail(self):
        result=m.group_stats(np.zeros((36,64),dtype=np.bool_),None,32,np,out_dim=1)
        self.assertEqual(result['tail_rows'],4);self.assertEqual(result['logical_union_macs'],0)
        self.assertEqual(result['full_groups']['K16_empty_fraction'],1.)
    def test_reject_nonfinite_and_unbound(self):
        with self.assertRaises(ValueError):m.build_keep(np.array([[0x7f80]],dtype=np.uint16),[.1,.2],np)
        with self.assertRaisesRegex(ValueError,'NOT_BOUND'):m.validate_plan({'id':m.ID,'assets_finalized':False})
    def test_header_rejects_overlaps_and_wrong_shape(self):
        import tempfile,json,struct
        from unittest.mock import patch
        header={'h':{'dtype':'BF16','shape':[2,64],'data_offsets':[0,256]},
                'gate':{'dtype':'BF16','shape':[2],'data_offsets':[256,260]}}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'h.safetensors'
            def write():
                data=json.dumps(header).encode();path.write_bytes(struct.pack('<Q',len(data))+data+bytes(260))
            write()
            with patch.object(m,'M',2), patch.object(m,'K',64):
                with path.open('rb') as stream:
                    actual,base=m.safetensor_header(stream);self.assertEqual(actual['gate']['shape'],[2])
                    self.assertEqual(m.read_tensor_sha(stream,base,256),__import__('hashlib').sha256(bytes(256)).hexdigest())
                header['gate']['data_offsets']=[254,258];write()
                with path.open('rb') as stream, self.assertRaises(ValueError):m.safetensor_header(stream)
    def test_budget_uses_full_real_population(self):
        plan={'repo_sources':[], 'read_budget':{'file_sha_bytes':123,'H_tensor_sha_bytes':8*m.M*m.K*2,
              'H_analysis_input_bytes':8*m.M*m.K*2,'gate_tensor_sha_bytes':160}}
        budget=m.limits(plan);self.assertEqual(budget['row_groups'],95040)
        self.assertEqual(budget['threshold_keep_matrices'],16)
        self.assertEqual(m.M%16,4);self.assertEqual(m.M%32,20)
    def test_parent_requires_actual_exit_and_complete_counts(self):
        import tempfile,json,hashlib
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); parent=root/'parent'; runfile=root/'run'; exitfile=root/'exit'
            parent.write_text(json.dumps({'id':m.PRODUCER_ID,'cases':[{'native_source_ordinal':s} for s in m.SOURCES],
                'limits':{'capture_packets':8,'historical_video_byte_checks':2}}))
            digest=hashlib.sha256(parent.read_bytes()).hexdigest(); assets=[{'id':i} for i in range(8)]
            phases=[{'native_source_ordinal':source,'all160_FFN_covered':True,'completed_native_flow_steps':4,
                     'matched_olddev_all_ten_states':True,'seed':42,'mode':'dense'} for source in m.SOURCES]
            run={'id':m.PRODUCER_ID,'protocol_sha256':digest,'status':'COMPLETE_FULLM_OBSERVATIONAL_CAPTURE',
                 'captures':assets,'phases':phases,'assigned_GPU_resources':{'violations':[]},
                 'operation_ledger':{'predebited':{'capture_packets':8,'historical_video_byte_checks':2},
                                     'observed':{'capture_packets':8,'historical_video_byte_checks':2},'forbidden_attempts':[]}}
            runfile.write_text(json.dumps(run)); runsha=hashlib.sha256(runfile.read_bytes()).hexdigest()
            exitfile.write_text(json.dumps({'id':m.PRODUCER_ID,'protocol_sha256':digest,'run_sha256':runsha,
                'process_exited':True,'child_returncode':0,'timeout':False,'automatic_retry':False,'spec_sha256':'launch-sha'}))
            plan={'capture_protocol':{'file':'parent'},'capture_RUN':{'file':'run','sha256':runsha},
                  'capture_EXIT':{'file':'exit'},'capture_assets':assets,'capture_launcher_spec_sha256':'launch-sha'}
            with patch.object(m,'bound',side_effect=lambda base,item:root/item['file']):
                self.assertEqual(m.verify_parent(plan,root)['id'],m.PRODUCER_ID)
                exitfile.write_text(json.dumps({'id':m.PRODUCER_ID,'protocol_sha256':digest,'run_sha256':runsha,
                   'process_exited':True,'child_returncode':1,'timeout':False,'automatic_retry':False}))
                with self.assertRaises(ValueError):m.verify_parent(plan,root)
    def test_same_fd_h_and_gate_hash_then_mask(self):
        import tempfile,json,struct,hashlib
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'packet'
            h=np.array([[0,0x3f80]*32,[0x3f80,0]*32],dtype='<u2').tobytes();gate=bytes(4)
            header={'h':{'dtype':'BF16','shape':[2,64],'data_offsets':[0,len(h)]},
                    'gate':{'dtype':'BF16','shape':[2],'data_offsets':[len(h),len(h)+len(gate)]}}
            body=json.dumps(header).encode();path.write_bytes(struct.pack('<Q',len(body))+body+h+gate)
            case={'file':{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},
                  'gate_shape':[2],'h_raw_sha256':hashlib.sha256(h).hexdigest(),'gate_raw_sha256':hashlib.sha256(gate).hexdigest()}
            charged=[];done=[];result={}
            with patch.object(m,'M',2),patch.object(m,'K',64):
                keep=m.load_keep_from_bound_file(path,root,case,[.1,.2],np,lambda k,n:charged.append((k,n)),lambda k,n:done.append((k,n)),result)
            self.assertEqual(charged,done);self.assertEqual(int(keep[0].sum()),64)
            self.assertTrue(result['input_stat_identity_checked_before_after_each_stage'])
            case['file']['sha256']='0'*64
            with patch.object(m,'M',2),patch.object(m,'K',64),self.assertRaises(ValueError):
                m.load_keep_from_bound_file(path,root,case,[.1,.2],np,lambda *_:None,lambda *_:None,{})
    def test_input_replace_and_inplace_modification_are_rejected(self):
        import tempfile,os
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'operand';path.write_bytes(b'first')
            with path.open('rb') as stream:
                identity=m.file_identity(stream,path);path.write_bytes(b'other longer')
                with self.assertRaisesRegex(ValueError,'replaced or modified'):m.file_identity(stream,path,identity)
            path.write_bytes(b'first')
            with path.open('rb') as stream:
                identity=m.file_identity(stream,path);other=Path(directory)/'new';other.write_bytes(b'first');os.replace(other,path)
                with self.assertRaisesRegex(ValueError,'replaced or modified'):m.file_identity(stream,path,identity)
    def test_invalid_or_wrong_width(self):
        with self.assertRaises(ValueError):m.group_stats(np.zeros((2,63),dtype=bool),None,16,np)
        with self.assertRaises(ValueError):m.bf16_scalar_bits(float('nan'))

if __name__=='__main__':unittest.main()
