"""Software fixtures for geometry, budgets and fail-closed acquisition guards."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('a5_capture',ROOT/'tools/a5_hwg_matched_capture_20261030.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class CaptureFixtures(unittest.TestCase):
    def test_every_tau_global_alignment_and_boundaries(self):
        for ordinal in [0,96,898,979,1040,2**32]:
            rows,starts=m.matched_rows([21,29,52],ordinal,42)
            self.assertEqual(len(rows),1344);self.assertEqual(len(set(rows)),1344)
            self.assertEqual(rows,sorted(rows))
            for tau,start in enumerate(starts):
                self.assertEqual(start%64,0)
                self.assertEqual(start//1508,tau)
                self.assertEqual((start+63)//1508,tau)
            self.assertEqual(m.matched_rows([21,29,52],ordinal,42),(rows,starts))

    def test_geometry_rejects_different_grid(self):
        with self.assertRaises(RuntimeError):m.matched_rows([21,29,51],898,42)

    def test_payload_all_fields_plus_margin(self):
        d=m.payload_budget();per=d['tensor_payload_bytes_by_field']
        self.assertEqual(per['fullM_V1'],51884851200)
        self.assertEqual(per['H'],1344*13824*2*160)
        self.assertEqual(d['total_tensor_payload_bytes'],sum(per.values()))
        self.assertLess(d['file_write_bytes_upper'],m.RES['new_HF_bytes'])
        self.assertEqual(len(per),16)

    def test_writer_budget_and_no_overwrite(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);w=m.Writer(root,root,lambda:None)
            with self.assertRaises(RuntimeError):w.reserve(m.RES['new_HF_bytes']+1)
            w.json(root/'RUN.json',{'state':'RUNNING'})
            w.json(root/'RUN.json',{'state':'COMPLETE'})
            self.assertEqual(json.loads((root/'RUN.json').read_text())['state'],'COMPLETE')
            self.assertEqual(w.written,(root/'RUN.json').stat().st_size+(root/'OPERATIONS.jsonl').stat().st_size)
            w.journal.close()

    def test_hash_identity_fail_closed(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);p=root/'input.json';p.write_text('{}')
            b=m.bind(p,root);self.assertEqual(m.check_binding(root,b),p)
            p.write_text('{ }')
            with self.assertRaises(RuntimeError):m.check_binding(root,b)

    def test_state_name_contract(self):
        self.assertEqual(len(m.STATE_NAMES),10)
        self.assertTrue({'raw_noise_FP32','initial_noise','flow_step3','latent_step3'}<=m.STATE_NAMES)

    def test_verifier_file_raw_sha_dtype_and_corruption(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'tiny.safetensors';t=torch.arange(24).reshape(3,8).to(torch.bfloat16)
            save_file({'V1':t},str(p))
            w=m.Writer(Path(d),Path(d),lambda:None)
            schema={'V1':{'shape':[3,8],'dtype':'bfloat16','raw_tensor_sha256':w.raw_sha(t)}}
            counted=[];observed,header=m.verify_tensor(p,schema,counted.append)
            self.assertEqual(observed,m.sha(p));self.assertEqual(sum(counted),p.stat().st_size)
            self.assertEqual(header['V1']['shape'],[3,8])
            with p.open('r+b')as f:f.seek(-1,2);v=f.read(1);f.seek(-1,2);f.write(bytes([v[0]^1]))
            with self.assertRaisesRegex(RuntimeError,'raw tensor SHA'):m.verify_tensor(p,schema,lambda n:None)
            w.journal.close()

    def test_read_predebit_and_failure_unknown(self):
        b=m.ReadBudget(20)
        b.before(12,'first');b.after(5)
        self.assertEqual(b.summary()['inflight_unknown']['actual_read_bytes'],5)
        with self.assertRaises(RuntimeError):b.before(1,'nested')
        with self.assertRaises(RuntimeError):b.finish()
        b.after(7);b.finish()
        with self.assertRaises(RuntimeError):b.before(9,'over limit')
        self.assertEqual(b.actual,12);self.assertEqual(b.predebited,12)

    def test_partial_tensor_write_is_accounted(self):
        import torch
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);w=m.Writer(root,root,lambda:None);p=root/'broken.safetensors'
            def broken(tensors,path):Path(path).write_bytes(b'partial');raise OSError('fixture failure')
            with patch('safetensors.torch.save_file',broken):
                with self.assertRaises(OSError):w.tensor(p,{'x':torch.ones(2)})
            self.assertEqual(w.files[str(p)]['bytes'],7);self.assertFalse(w.files[str(p)]['complete'])
            self.assertNotIn(str(p),w.holds);self.assertIn('FILE_WRITE_ACCOUNTED',(root/'OPERATIONS.jsonl').read_text());w.journal.close()

    def test_packet_alias_site_and_fixed_schema_rejected(self):
        import copy
        with tempfile.TemporaryDirectory()as d:
            hf=Path(d);source=hf/'results';folder=source/'observed/ffn';folder.mkdir(parents=True)
            paths={k:folder/n for k,n in [('metadata','step1_block00.json'),('payload','step1_block00.safetensors'),('full','fullV1_step1_block00.safetensors')]}
            for p in paths.values():p.write_text('fixture')
            ref=lambda key:{'path':str(paths[key].relative_to(hf))}
            case={'source_id':'fixture','native_source_ordinal':898};p={'case':case,'cache':{'sha256':'c'*64}}
            rows,starts=m.matched_rows([21,29,52],898,42);packet={'step':1,'block':0,'metadata':ref('metadata')}
            meta={'step':1,'block':0,'expert':'high_noise_model','source_id':'fixture','native_source_ordinal':898,'seed':42,'grid':[21,29,52],'numeric_family':'HIF4_FAKE_QDQ_BF16','selected_rows':rows,'tile_starts':starts,'row_count':1344,'tensor_schema':{k:{'shape':s,'dtype':'bfloat16','raw_tensor_sha256':'a'*64}for k,s in m.expected_sample_shapes().items()},'checks':{k:True for k in m.REQUIRED_CHECKS},'quantized_cache':p['cache'],'QDQ_calls':[{'module':f'blocks.0.ffn.{suffix}','input':f'Q{slot}_input','output':f'V{slot}','force_fp32':True,'input_dtype':'torch.bfloat16','output_dtype':'torch.bfloat16','native_group_size':64,'dim':-1}for suffix,slot in [('0','1'),('2','2')]],'effective_W2':{'runtime_shape':[5120,13824],'dtype':'torch.bfloat16','runtime_cache_bytes_match':True,'cache_record_key':['high_noise_model','blocks.0.ffn.2','weight_qdq'],'runtime_operand_sha256':'b'*64},'full_V1':{**ref('full'),'shape':[31668,5120],'dtype':'bfloat16'},'payload':ref('payload'),'gate_native_shape':[1,1,5120]}
            seen={'paths':set(),'inodes':set()};m.validate_packet(meta,packet,p,hf,source,seen)
            with self.assertRaisesRegex(RuntimeError,'duplicate'):m.validate_packet(meta,packet,p,hf,source,seen)
            for mutate in [lambda x:x.update(block=1),lambda x:x['tensor_schema'].pop('H'),lambda x:x['tensor_schema']['H'].update(dtype='float32'),lambda x:x['tensor_schema']['V1'].update(shape=[1343,5120]),lambda x:x['checks'].update(cuda_rng_unchanged=False)]:
                bad=copy.deepcopy(meta);mutate(bad)
                with self.assertRaises(RuntimeError):m.validate_packet(bad,packet,p,hf,source,{'paths':set(),'inodes':set()})


if __name__=='__main__':unittest.main()
