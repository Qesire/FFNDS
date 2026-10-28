"""Portable controls for native scope boundaries; no model or tensor allocation."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('full_depth_collect', ROOT/'tools/e0_original_bf16_full_depth_collect.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FullDepthControls(unittest.TestCase):
    def test_import_is_stdlib_only_and_native_globals_unchanged(self):
        code = '''import sys
sys.path.insert(0, 'tools')
import e0_original_bf16_videos as n
before=(n.ID,dict(n.LIMITS),list(n.RUN_IDS))
import e0_original_bf16_full_depth_collect
assert before==(n.ID,dict(n.LIMITS),list(n.RUN_IDS))
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
'''
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)

    def test_scope_limits_and_ordinal_rows(self):
        lim=m.limits()
        for key,expected in {'native_full_model_flows':16,'native_FFN_calls':640,'native_FFN_linears':1280,'original_checkpoint_block_buffer_loads':648,'packet_safetensor_writes':640,'video_generations':3,'core_CPU_reports':3,'native_prefix_replays_no_decode':1,'packet_payload_validations':640,'packet_output_byte_reservations':640,'state_output_byte_reservations':40}.items():
            self.assertEqual(lim[key],expected)
        self.assertNotEqual(m.scope_id(0),m.scope_id(1))
        with self.assertRaises(ValueError): m.scope_id(2)
        case={'native_source_ordinal':42,'consumer20_index':12}
        self.assertEqual(m.run_id(case),'raw_full_depth_ord42_seed42_dense')
        rows=m.drawings(case['native_source_ordinal'])
        self.assertEqual(rows,m.drawings(42))
        self.assertNotEqual(rows,m.drawings(case['consumer20_index']))
        self.assertEqual(len(rows),4)
        self.assertTrue(all(len(x)==40 for x in rows.values()))
        self.assertTrue(all(len(v)==len(set(v))==256 and all(0<=r<m.M for r in v) for step in rows.values() for v in step.values()))

    def test_packet_fields_shape_dtype_and_rows(self):
        rows=list(range(m.ROWS))
        meta={k:{'shape':shape[:], 'dtype':'int64' if k=='rows' else 'bfloat16', 'raw_tensor_sha256':'a'*64} for k,shape in m.ROW_SHAPES.items()}
        m.check_packet_metadata(meta,rows)
        broken=copy.deepcopy(meta); del broken['h']
        with self.assertRaises(ValueError): m.check_packet_metadata(broken,rows)
        broken=copy.deepcopy(meta); broken['h']['shape']=[m.ROWS,m.D]
        with self.assertRaises(ValueError): m.check_packet_metadata(broken,rows)
        broken=copy.deepcopy(meta); broken['u']['dtype']='float32'
        with self.assertRaises(ValueError): m.check_packet_metadata(broken,rows)
        broken=copy.deepcopy(meta); broken['Y']['raw_tensor_sha256']='a'*63
        with self.assertRaises(ValueError): m.check_packet_metadata(broken,rows)
        broken=copy.deepcopy(meta); broken['Y']['raw_tensor_sha256']='z'*64
        with self.assertRaises(ValueError): m.check_packet_metadata(broken,rows)
        with self.assertRaises(ValueError): m.check_packet_metadata(meta,rows[:-1]+[0])

    def test_failed_operation_predebit_and_budget(self):
        events=[]; report={}; ledger=m.Ledger(report,['known'],events.append)
        def fail(): raise RuntimeError('synthetic failure')
        with self.assertRaisesRegex(RuntimeError,'synthetic failure'): ledger.call('known',{'packet_safetensor_writes':1},fail)
        self.assertEqual(report['predebited']['packet_safetensor_writes'],1)
        self.assertNotIn('packet_safetensor_writes',report['observed'])
        self.assertEqual(events[0]['kind'],'reserved_before_call')
        with self.assertRaises(ValueError): ledger.call('unknown',{'packet_safetensor_writes':1},lambda:None)
        with self.assertRaises(ValueError): ledger.call('known',{'packet_safetensor_writes':640},lambda:None)

    def test_selection_requires_one_full_native_operand(self):
        class FakeTensor:
            dtype='bf16'
            def __init__(self, shape): self.shape=shape
            def stride(self): return [self.shape[1],1]
            def index_select(self,dim,rows): return ('selected',dim,rows)
        class FakeLedger:
            def call(self,rid,costs,fn,*a): return fn(*a)
        class Obj: pass
        tap=object.__new__(m.CollectorTap); tap.guard=Obj(); tap.guard.torch=Obj(); tap.guard.torch.bfloat16='bf16'; tap.guard.ledger=FakeLedger(); tap.rid='fake'
        tap.pending={'values':{},'source_layout':{},'rows':'native-rows'}
        tap.select('h',FakeTensor([m.M,m.H]))
        self.assertEqual(tap.pending['values']['h'],('selected',0,'native-rows'))
        with self.assertRaises(ValueError): tap.select('h',FakeTensor([m.M,m.H]))
        with self.assertRaises(ValueError): tap.select('V1',FakeTensor([m.ROWS,m.D]))
        tap.pending=None
        with self.assertRaises(ValueError): tap.select('V1',FakeTensor([m.M,m.D]))


if __name__=='__main__': unittest.main()
