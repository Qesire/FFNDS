import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('whole_ffn_bridge_test',ROOT/'tools/e0_s2_whole_ffn_boundary_bridge.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class BridgeTests(unittest.TestCase):
    def test_native_rows_and_exact_payload_work(self):
        self.assertEqual(len(m.ROWS),32);self.assertEqual(len(set(m.ROWS)),32)
        self.assertEqual(list(m.ROWS),[t*29*52+y*52+x for t in m.TAUS for y,x in m.XY])
        self.assertEqual(m.SITES,((0,0),(1,20),(2,20),(3,39)))
        n=m.limits();self.assertEqual(n['video_generations'],4);self.assertEqual(n['native_FFN_calls'],640)
        self.assertEqual(n['native_FFN_linears'],1280);self.assertEqual(n['BF16_factor_GEMM'],640)
        self.assertEqual(n['original_W2_H2D_skips'],324);self.assertEqual(n['factor_CPU_file_loads'],80)
        self.assertEqual(n['raw_postwrite_packets'],16);self.assertEqual(n['operand_row_selections'],96)
        self.assertEqual(n['packet_tensor_D2H_bytes'],16*m.PACKET_BYTES)
        self.assertEqual(n['parent_state_byte_checks'],40);self.assertEqual(n['parent_video_byte_checks'],4)
        for forbidden in ('training_updates','decompositions','official_quality_calls'):self.assertEqual(n[forbidden],0)

    def test_w1_return_selected_before_gelu_and_without_new_mm(self):
        class Ledger:
            def call(self,rid,cost,fn,*a):return fn(*a)
        class Tensor:
            dtype='bf16'
            def __init__(self,shape,label):self.shape=shape;self.label=label
            def stride(self):return (self.shape[1],1)
            def index_select(self,axis,indices):events.append('select_'+self.label);return 'row_'+self.label
        class Device:
            type='cuda';index=0
            def __str__(self):return 'cuda:0'
        class Bias:
            dtype='bf16';shape=(13824,);device=Device()
            def stride(self):return (1,)
        events=[];tap=m.BridgeTap.__new__(m.BridgeTap)
        tap.guard=SimpleNamespace(torch=SimpleNamespace(bfloat16='bf16',is_tensor=lambda v:isinstance(v,Bias)),ledger=Ledger())
        tap.rid='r';tap.indices='indices';tap.block=(0,0)
        tap.pending={'values':{},'source_layout':{},'expert_block':(0,0)}
        w1=SimpleNamespace(bias=Bias());tap.mapping={id(w1):('block',0,'ffn.0')}
        x=Tensor((31668,5120),'x');a=Tensor((31668,13824),'a')
        def actual(*_args):events.append('original_full_M_w1');return a
        with patch.object(m.vm.LightTap,'mm_call',side_effect=actual):
            self.assertIs(tap.mm_call(w1,x,{'bias_stride':[1]},lambda *_:None),a)
        self.assertEqual(events,['select_x','original_full_M_w1','select_a'])
        self.assertEqual(tap.pending['values'],{'V1':'row_x','a':'row_a'})
        self.assertEqual(tap.pending['source_layout']['a']['shape'],[31668,13824])
        self.assertEqual(tap.pending['source_layout']['W1_bias'],{'shape':[13824],'dtype':'bfloat16',
            'device':'cuda:0','stride':[1],'identity':'ACTUAL_FFN0_MM_BIAS'})
        for attr,bad in [('bias',None),('bias',SimpleNamespace()),('bias',Bias())]:
            w1.bias=bad
            if isinstance(bad,Bias):bad.dtype='float32'
            before=list(events)
            with self.assertRaises(ValueError):tap.mm_call(w1,x,{'bias_stride':[1]},lambda *_:None)
            self.assertEqual(events,before)
        w1.bias=Bias()
        with self.assertRaises(ValueError):tap.mm_call(w1,x,{'bias_stride':[2]},lambda *_:None)

    def test_packet_geometry_rejects_tampering_and_identifies_student_Y(self):
        packet={'step':2,'grid':[21,29,52],'population_rows':31668,'tau':list(m.TAUS),
                'xy':[list(v) for v in m.XY],'expert':'low','rows':list(m.ROWS)}
        self.assertTrue(m.packet_geometry_matches(packet))
        for key,bad in [('expert','high'),('tau',[]),('xy',[]),('population_rows',1),('rows',[]),('step',4)]:
            changed=copy.deepcopy(packet);changed[key]=bad
            self.assertFalse(m.packet_geometry_matches(changed),key)
        self.assertEqual(m.y_boundary('dense'),'ORIGINAL_DENSE_W2_OUTPUT')
        self.assertEqual(m.y_boundary('value_all'),'ACTUAL_STUDENT_FACTOR_W2_OUTPUT_NOT_ORIGINAL_DENSE_W2')
        with self.assertRaises(ValueError):m.y_boundary('other')

    def test_same_hardware_control_rejects_changed_states_or_video(self):
        state={'sha256':'a'*64,'bytes':1,'raw_tensor_sha256':'b'*64,'shape':[1],'dtype':'bfloat16'}
        phase={'run_id':'r','states':{str(i):copy.deepcopy(state) for i in range(10)},'video':{'sha256':'c'*64,'bytes':4}}
        ref={'states':copy.deepcopy(phase['states']),'video':copy.deepcopy(phase['video'])}
        class Ledger:
            def __init__(self):self.events=[]
            def call(self,rid,cost,fn):self.events.append(cost);return fn()
        ledger=Ledger();m.verify_neutral(phase,ref,ledger,Path('/tmp'))
        self.assertEqual(len(ledger.events),11)
        bad=copy.deepcopy(ref);bad['states']['7']['raw_tensor_sha256']='d'*64
        with self.assertRaises(ValueError):m.verify_neutral(phase,bad,Ledger(),Path('/tmp'))
        bad=copy.deepcopy(ref);bad['video']['sha256']='e'*64
        with self.assertRaises(ValueError):m.verify_neutral(phase,bad,Ledger(),Path('/tmp'))


    def test_packet_header_binds_dtype_shape_and_contiguous_offsets(self):
        import json,struct,tempfile
        shapes={'a':[2,3],'rows':[2]}; header={'a':{'dtype':'BF16','shape':[2,3],'data_offsets':[0,12]},
                                               'rows':{'dtype':'I64','shape':[2],'data_offsets':[12,28]}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(m,'SHAPES',shapes),patch.object(m,'PACKET_BYTES',28):
            path=Path(tmp)/'p.safetensors'
            def write(h):
                raw=json.dumps(h).encode();path.write_bytes(struct.pack('<Q',len(raw))+raw+b'\0'*28)
            write(header);m.verify_packet_header(path)
            bad=copy.deepcopy(header);bad['a']['dtype']='F32';write(bad)
            with self.assertRaises(ValueError):m.verify_packet_header(path)
            bad=copy.deepcopy(header);bad['rows']['data_offsets']=[11,27];write(bad)
            with self.assertRaises(ValueError):m.verify_packet_header(path)

    def test_failed_operation_retains_predebit(self):
        record={};events=[]
        with patch.object(m.native,'RUN_IDS',('rid',)),patch.object(m.native,'LIMITS',{'probe':1}):
            ledger=m.vm.Ledger(record,events.append)
            def fail():raise RuntimeError('capture failure')
            with self.assertRaises(RuntimeError):ledger.call('rid',{'probe':1},fail)
            self.assertEqual(record['predebited'],{'probe':1});self.assertEqual(record['observed'],{})
            with self.assertRaises(ValueError):ledger.call('rid',{'probe':1},lambda:None)

    def test_scope_authorization_exact(self):
        scope={'id':m.ID,'protocol_sha256':'abc','execution_status':'AUTHORIZED_NOT_STARTED','limits':m.limits(),
               'resources':m.RESOURCES,'run_ids':['a','b','c','d'],'max_gpu_cards':1}
        active={'user_continuation_authorization_20261027':{'new_scope_registration_allowed':True},
                'experiment_authority':{'current_user_authorized_scopes':[scope]}}
        with patch.object(m,'validate'),patch.object(m.native,'read',return_value=active):
            self.assertIs(m.authorize({'run_ids':scope['run_ids']},'abc'),scope)
            with self.assertRaises(ValueError):m.authorize({'run_ids':scope['run_ids']},'different')

    def test_missing_actual_parent_receipts_cannot_make_draft(self):
        with patch.object(m,'parent_proof',side_effect=FileNotFoundError('actual EXIT absent')):
            with self.assertRaises(FileNotFoundError):m.draft(Path('/tmp/hf'))
        with self.assertRaises(ValueError):m.check_parents([],Path('/tmp/hf'))

if __name__=='__main__':unittest.main()
