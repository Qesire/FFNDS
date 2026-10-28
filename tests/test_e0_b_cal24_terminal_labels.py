import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import e0_b_cal24_terminal_labels as m
import b_cal24_label_contract as c


def cases():
    return [{'case_id': f'cal24_{i:02d}_step{s}', 'part': i//6, 'source_index': i,
        'source_image_sha256': hashlib.sha256(str(i).encode()).hexdigest(), 'source_id': f'source{i}',
        'source_group': f'group{i}', 'role': 'TRAIN' if i < 16 else 'HELDOUT_EXPOSED',
        'seed': 42+i, 'step': s, 'expert': m.EXPERTS[0 if s < 2 else 1], 'block': 39,
        'rows': list(range(256)), 'sampling_seed': 42+i+s,
        'input_path': f'input{i}.json', 'input_sha256': 'a'*64, 'historical_post_sha256': 'b'*64,
        'operand_path': f'input{i}_step{s}.safetensors', 'operand_sha256': 'c'*64,
        'operand_bytes': 1, 'prompt_sha256': 'd'*64} for i in range(24) for s in range(4)]


def evidence(case):
    e = {k:'a'*64 for k in c.EVIDENCE_SHA_KEYS}
    e['actual_post_sha256'] = case['historical_post_sha256']
    e.update({k:True for k in c.EVIDENCE_BOOL_KEYS})
    e.update(head_weight_stride=[1,5120], head_bias_stride=[1], head_modulation_stride=[10240,5120,1],
             head_shift_stride=[10240,5120,1], head_scale_stride=[10240,5120,1])
    return e


class ContractControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads(); torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.previous_threads)

    def test_import_never_imports_torch(self):
        code = "import sys;sys.path.insert(0,'tools');import e0_b_cal24_terminal_labels;assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_budget_matches96_original_flows_and_global_two_GPU_limit(self):
        self.assertEqual(m.RESOURCES['global_max_simultaneously_assigned_idle_GPUs'], 2)
        self.assertEqual(m.RESOURCES['GPU_per_part'], 1)
        for key, value in c.TOTAL.items(): self.assertEqual(m.PART_LIMITS[key]*4, value)
        self.assertEqual(m.PART_LIMITS['setup_decoded_weight_GPU_transfers'], 800)
        self.assertEqual(m.PART_LIMITS['setup_decoded_weight_H2D_bytes'], 56203673600)
        self.assertEqual(m.PART_LIMITS['CPU_BF16_gate_mul_elements'], 24*256*5120)
        self.assertEqual(m.PART_LIMITS['label_packs'], 24)
        self.assertEqual(m.PART_LIMITS['full_u_files'], 0)
        self.assertEqual(len(m.run_ids()), 96)
        self.assertEqual(set().union(*(set(m.run_ids(p)) for p in range(4))), set(m.run_ids()))

    def test_predebit_is_persisted_before_execution_and_failure_not_relabelled(self):
        record, seen = {}, []
        ledger = m.WorkLedger(record, cases(), 0, lambda: seen.append(copy.deepcopy(record)))
        with self.assertRaises(ValueError): ledger.observe(m.FLOW_COST)
        ledger.reserve(m.FLOW_COST, 'cal24_00_step0')
        self.assertEqual(seen[-1]['predebited']['dense_flows'], 1)
        self.assertEqual(record['observed']['dense_flows'], 0)
        with self.assertRaises(ValueError): ledger.reserve(m.FLOW_COST, 'cal24_06_step0')
        with self.assertRaises(ValueError): ledger.finish()

    def test_exact_two_ledgers_finish_only_after_all_sources_and_cases(self):
        record = {}; ledger = m.WorkLedger(record, cases(), 0)
        ledger.reserve(m.SETUP_COST, 'setup'); ledger.observe(m.SETUP_COST)
        for i in range(6):
            ledger.reserve(m.SOURCE_COST, f'source{i:02d}'); ledger.observe(m.SOURCE_COST)
            ledger.native.mark_source(i)
            for case in [x for x in cases() if x['source_index'] == i]:
                ledger.reserve(m.FLOW_COST, case['case_id']); ledger.observe(m.FLOW_COST)
                ledger.native.mark_case(case)
        self.assertEqual(ledger.finish()['case_count'], 24)
        self.assertEqual(record['observed'], m.PART_LIMITS)
        with self.assertRaises(ValueError): ledger.reserve(m.FLOW_COST, 'cal24_00_step0')

    def test_label_metadata_has_exact_identity_evidence_and_original_rows(self):
        case = cases()[0]
        value = m.label_metadata(case, evidence(case))
        self.assertEqual(set(value), set(c.META_KEYS)|set(c.EVIDENCE_SHA_KEYS)|set(c.EVIDENCE_BOOL_KEYS)|set(c.EVIDENCE_STRIDE_KEYS)|{'rows_sha256'})
        self.assertEqual(value['rows_sha256'], c.rows_sha256(case['rows']))
        self.assertEqual(value['original_v1_byte_equal'], 'true')
        self.assertEqual(value['head_weight_stride'], '[1,5120]')

    def test_real_CPU_pack_roundtrip_header_and_row_identity(self):
        import torch
        from safetensors.torch import load_file
        case = cases()[0]
        pack = {key:torch.zeros(shape, dtype=torch.int64 if dt == 'I64' else torch.bfloat16)
                for key, (dt, shape) in c.PACK_SCHEMA.items()}
        pack['rows'] = torch.tensor(case['rows'], dtype=torch.int64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'labels.safetensors'
            result = m.write_label(path, case, pack, evidence(case))
            self.assertEqual(result['sha256'], m.sha256(path))
            actual = load_file(str(path))
            self.assertEqual(set(actual), set(c.PACK_SCHEMA))
            self.assertTrue(torch.equal(actual['rows'], pack['rows']))
            self.assertTrue(result['all_serialized_tensor_bytes_match_memory'])
            self.assertTrue(result['saved_rows_payload_matches_case'])
            with self.assertRaises(ValueError): m.write_label(path, case, pack, evidence(case))
            pack['rows'][0] = 1
            with self.assertRaisesRegex(ValueError, 'row coordinates'):
                m.write_label(Path(tmp)/'wrong.safetensors', case, pack, evidence(case))

    def test_payload_only_corruption_passes_old_header_and_fails_saved_byte_gate(self):
        import torch
        case=cases()[0]
        pack={key:torch.zeros(shape,dtype=torch.int64 if dt=='I64' else torch.bfloat16)
              for key,(dt,shape) in c.PACK_SCHEMA.items()}
        pack['rows']=torch.tensor(case['rows'],dtype=torch.int64)
        for key in ('rows','r_rows'):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'labels.safetensors'
                result=m.write_label(path,case,pack,evidence(case))
                header,length=m.safetensors_header(path)
                with path.open('r+b') as stream:
                    stream.seek(8+length+header[key]['data_offsets'][0])
                    stream.write(struct.pack('<q',999) if key=='rows' else b'\x01\x00')
                c.validate_label_header(path,case)  # unchanged header/metadata alone cannot detect this.
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'payload differs'):
                    m.validate_saved_payload(path,case,{k:v['raw_tensor_sha256'] for k,v in result['tensors'].items()})

    def test_external_cpu_receipt_gate_requires_real_exit_source_worker_and_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); worker='f'*64; digest='e'*64
            plan={'repo_sources':[{'path':'source.py','bytes':1,'sha256':'a'*64}],
                  'expected_case_identity_sha256':'d'*64,'private_worker_config_sha256':[worker]}
            good={'id':m.ID,'part':0,'protocol_sha256':digest,'status':'PASS_CAL24_LABEL_CPU_ENTRY_NO_TORCH_NO_GPU',
                  'source_bindings_sha256':m.state_sha(plan['repo_sources']),'source_bindings_verified':1,
                  'expected_case_identity_sha256':plan['expected_case_identity_sha256'],'source_indices':list(range(6)),
                  'original_operand_files_full_SHA_verified':24,'Torch_or_CUDA_imported':False,
                  'all_CUDA_devices_hidden':True,'native_model_calls':0,'GPU_calls':0,'worker_config_sha256':worker}
            def bind(path):return {'path':path.name,'bytes':path.stat().st_size,'sha256':m.sha256(path)}
            def active(report,exit_change=None):
                rp=root/'cpu.json';rp.write_text(json.dumps(report));rb=bind(rp)
                exited={'id':m.ID,'part':0,'protocol_sha256':digest,'launcher_returncode':0,'process_exited':True,'RUN_sha256':rb['sha256']}
                if exit_change:exited.update(exit_change)
                ep=root/'exit.json';ep.write_text(json.dumps(exited))
                return {'experiment_authority':{'current_user_authorized_scopes':[{'id':m.ID,'CPU_preflights':[{'part':0,'binding':rb,'launch_exit':bind(ep)}]}]}}
            m.validate_cpu_preflight(plan,digest,active(good),0,root,worker)
            for change in ({'part':1},{'Torch_or_CUDA_imported':True},{'all_CUDA_devices_hidden':False},
                           {'worker_config_sha256':'0'*64},{'source_bindings_sha256':'0'*64},
                           {'protocol_sha256':'0'*64},{'native_model_calls':1}):
                wrong={**good,**change}
                with self.subTest(change=change),self.assertRaises(ValueError):
                    m.validate_cpu_preflight(plan,digest,active(wrong),0,root,worker)
            for change in ({'launcher_returncode':1},{'process_exited':False},{'RUN_sha256':'0'*64}):
                with self.subTest(exit_change=change),self.assertRaises(ValueError):
                    m.validate_cpu_preflight(plan,digest,active(good,change),0,root,worker)
            duplicate=active(good)
            entry=duplicate['experiment_authority']['current_user_authorized_scopes'][0]['CPU_preflights'][0]
            duplicate['experiment_authority']['current_user_authorized_scopes'][0]['CPU_preflights'].append(entry)
            with self.assertRaises(ValueError):m.validate_cpu_preflight(plan,digest,duplicate,0,root,worker)

    def test_unaccepted_operand_or_post_metadata_cannot_pass_header_gate(self):
        import torch
        case = cases()[0]
        pack = {key:torch.zeros(shape, dtype=torch.int64 if dt == 'I64' else torch.bfloat16)
                for key, (dt, shape) in c.PACK_SCHEMA.items()}
        pack['rows'] = torch.tensor(case['rows'], dtype=torch.int64)
        for field, value in [('original_y_byte_equal', False), ('actual_post_sha256', 'e'*64)]:
            with tempfile.TemporaryDirectory() as tmp:
                e = evidence(case); e[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    m.write_label(Path(tmp)/'unaccepted.safetensors', case, pack, e)


class SegmentControls(unittest.TestCase):
    def test_continuous_original_four_steps_no_import_no_decode(self):
        import torch
        from b_final_block_capture import raw_sha
        selected = copy.deepcopy(cases()[:4])
        class Wan22StepDistillScheduler:
            keep_latents_dtype_in_scheduler = False
            sigmas = torch.tensor([1., .9375, .8333333134651184, .625])
            def __init__(self): self.latents = torch.tensor([1., 2.], dtype=torch.float32)
            def _uses_conditioned_latent_prefix(self): return False
            def step_pre(self, step_index):
                self.step_index = step_index
                self.latents = self.latents.bfloat16()
            def step_post(self): self.latents += 1
        scheduler = Wan22StepDistillScheduler()
        counters = dict.fromkeys(('linear_forward_calls','activation_qdq_calls','weight_transfers','cache_misses'),0)
        class Model:
            cur_model_index = -1
            def infer(self, inputs):
                self.cur_model_index = 0 if scheduler.step_index < 2 else 1
                scheduler.noise_pred = torch.zeros_like(scheduler.latents).float()
                counters['linear_forward_calls'] += 400; counters['activation_qdq_calls'] += 400
        runner = SimpleNamespace(scheduler=scheduler, model=Model(), inputs={'condition': torch.ones(1)})
        began, finished = [], []
        for step, case in enumerate(selected): case['historical_post_sha256'] = raw_sha(torch.tensor([2.+step,3.+step]))
        class Tap:
            runtime = SimpleNamespace(counters=counters)
            def begin(self, case): self.case = case; began.append(case['step'])
            def finish(self, actual):
                if actual != self.case['historical_post_sha256']: raise ValueError('mismatch')
                finished.append(self.case['step']); return {'case':self.case}
        registered = cases(); registered[:4] = selected
        record = {}; ledger = m.WorkLedger(record, registered, 0)
        ledger.reserve(m.SOURCE_COST, 'source00')
        output = {}
        segment = m.ContinuousSourceSegment(runner, Tap(), selected, dict(m.SOURCE_COST), ledger,
                    lambda *a:None, output, lambda:None, lambda:None)
        with mock.patch.object(m,'SHAPE',(2,)), mock.patch.object(torch.cuda,'synchronize'):
            result = segment.run_segment()
        self.assertEqual(began, [0,1,2,3]); self.assertEqual(finished, began)
        self.assertEqual(result.tolist(), [5.,6.])
        self.assertEqual(len(output['cases']), 4)
        self.assertFalse(hasattr(runner,'inputs'))
        self.assertEqual(record['observed']['dense_flows'], 4)
        self.assertFalse(torch.cuda.is_initialized())

    def test_source_specific_seed_and_prepare_call_identity(self):
        import torch
        scheduler = SimpleNamespace(generator=None, prepare=lambda:None)
        def noise(seed, latent_shape, dtype=torch.float32): scheduler.latents=torch.zeros(latent_shape,dtype=dtype)
        scheduler.prepare_latents = noise
        calls = []
        runner = SimpleNamespace(scheduler=scheduler, run_input_encoder=lambda:calls.append('input'),
            run_text_encoder=lambda:calls.append('text'), run_vae_encoder=lambda:calls.append('vae'),
            run_image_encoder=lambda:None, run_vae_decoder=lambda:None)
        actual = dict.fromkeys(m.SOURCE_COST, 0)
        with ExitStack() as stack:
            m.source_preparation(stack,runner,{'seed':49},actual)
            runner.run_input_encoder(); runner.run_text_encoder(); runner.run_vae_encoder(); scheduler.prepare()
            with self.assertRaises(ValueError): scheduler.prepare_latents(42,m.SHAPE)
            scheduler.prepare_latents(49,m.SHAPE)
            with self.assertRaises(ValueError): scheduler.prepare_latents(49,m.SHAPE)
            with self.assertRaisesRegex(RuntimeError,'Decode'): runner.run_vae_decoder()
        self.assertEqual(actual,m.SOURCE_COST)
        self.assertEqual(calls,['input','text','vae'])
        self.assertIs(scheduler.prepare_latents,noise)


class TapControls(unittest.TestCase):
    def fixture(self, stack, wrong_operand=False):
        import torch
        tiny = {'M':8,'D':8,'ROWS':4,'OUT':2,'GRID':(1,2,4)}
        for key,value in tiny.items(): stack.enter_context(mock.patch.object(m,key,value))
        for key,value in {'M':8,'D':8,'ROWS':4}.items(): stack.enter_context(mock.patch.object(c,key,value))
        stack.enter_context(mock.patch.object(torch.cuda,'get_rng_state',side_effect=lambda:torch.zeros(4,dtype=torch.uint8)))
        adapter = SimpleNamespace(native_hif4_activation_qdq=lambda value,force_fp32=True:value)
        runtime = SimpleNamespace(counters=dict.fromkeys(('linear_forward_calls','activation_qdq_calls','weight_transfers','cache_misses'),0))
        def apply(mm,value,**kw):
            result=adapter.native_hif4_activation_qdq(value,force_fp32=True)
            runtime.counters['activation_qdq_calls']+=1; runtime.counters['linear_forward_calls']+=1
            return torch.zeros_like(result) if kw['module_name'].endswith('.ffn.2') else result
        runtime.apply=apply
        class Norm:
            eps=1e-6
            def _get_actual_weight(self):return None
            def _get_actual_bias(self):return None
            def apply(self,value):return value
        class Head:
            has_lora_branch=False
            def __init__(self,value):
                self.weight=torch.full((2,8),value,dtype=torch.bfloat16).t()
                self.bias=torch.zeros(2,dtype=torch.bfloat16)
            def _get_actual_weight(self):return self.weight
            def _get_actual_bias(self):return self.bias
            def apply(self,value):return torch.zeros(8,2,dtype=torch.bfloat16)
        class Infer:
            block_idx=0
            def __init__(self,index):self.index=index
            def infer_ffn(self,phase,x,attn,shift,scale,gate=None):
                expert=m.EXPERTS[self.index]
                value=runtime.apply(None,x,expert=expert,step_index=current['step'],block_index=self.block_idx,module_name=f'blocks.{self.block_idx}.ffn.0')
                return runtime.apply(None,value,expert=expert,step_index=current['step'],block_index=self.block_idx,module_name=f'blocks.{self.block_idx}.ffn.2')
            def post_process(self,x,y,gate,pre_out=None):
                x.add_(y*gate.squeeze());return x
            def infer_block(self,block,x,pre_out):
                for suffix in m.LINEAR_SUFFIXES[:8]:
                    runtime.apply(None,x,expert=m.EXPERTS[self.index],step_index=current['step'],block_index=self.block_idx,module_name=f'blocks.{self.block_idx}.{suffix}')
                gate=torch.ones(1,1,8,dtype=torch.bfloat16)
                y=self.infer_ffn(None,x,None,None,None,gate)
                return self.post_process(x,y,gate,pre_out)
            def infer_non_blocks(self,weights,x,embed):
                e=(weights.head_modulation.tensor+embed.unsqueeze(1)).chunk(2,dim=1)
                value=weights.norm.apply(x)
                value=value*(1+e[1])+e[0]
                return weights.head.apply(value.reshape(8,8))
        models=[];heads=[];current={'step':0}
        for index in range(2):
            head=Head(index+1)
            weights=SimpleNamespace(norm=Norm(),head=head,head_modulation=SimpleNamespace(tensor=torch.full((1,2,8),index,dtype=torch.bfloat16)))
            pre_weight=SimpleNamespace(**{key:SimpleNamespace(apply=lambda value:value) for key in
                ('time_embedding_0','time_embedding_2','time_projection_1','text_embedding_0','text_embedding_2','patch_embedding')})
            def pre(weights,inputs):
                for value in vars(weights).values():value.apply(inputs)
                return SimpleNamespace(x=torch.ones(8,8,dtype=torch.bfloat16),embed=torch.zeros(1,8,dtype=torch.bfloat16),grid_sizes=SimpleNamespace(tuple=(1,2,4)))
            models.append(SimpleNamespace(transformer_infer=Infer(index),transformer_weights=weights,
                pre_weight=pre_weight,pre_infer=SimpleNamespace(infer=pre),post_infer=SimpleNamespace(infer=lambda value,pre:[value])))
            heads.append({'head_weight_runtime_KN':head.weight.contiguous(),'head_bias':head.bias,'head_modulation':weights.head_modulation.tensor})
        def load(case):
            return {'V1':torch.ones(4,8,dtype=torch.bfloat16),'Y':torch.full((4,8),int(wrong_operand),dtype=torch.bfloat16),
                    'gate':torch.ones(1,8,dtype=torch.bfloat16)}
        saved=[]
        def save(case,pack,e):
            saved.append((copy.deepcopy(case),{k:v.clone() for k,v in pack.items()},copy.deepcopy(e)))
            return {'cpu_software_fixture_only':True}
        tap=m.Cal24LabelTap(runtime,adapter,models,heads,load,save)
        return torch,tap,models,current,saved

    def invoke(self,tap,model,case):
        tap.begin(case)
        pre=model.pre_infer.infer(model.pre_weight,None)
        value=pre.x
        block=SimpleNamespace(compute_phases=[SimpleNamespace(),SimpleNamespace(),SimpleNamespace()])
        for b in range(40):
            model.transformer_infer.block_idx=b
            value=model.transformer_infer.infer_block(block,value,pre)
        head=model.transformer_infer.infer_non_blocks(model.transformer_weights,value,pre.embed)
        model.post_infer.infer(head,pre)
        return tap.finish(case['historical_post_sha256'])

    def test_four_steps_use_correct_expert_and_write_only_selected_seven_tensors(self):
        with ExitStack() as stack:
            torch,tap,models,current,saved=self.fixture(stack)
            original_apply=tap.runtime.apply
            with tap:
                for step in range(4):
                    current['step']=step;case=copy.deepcopy(cases()[step]);case['rows']=[0,2,5,7]
                    result=self.invoke(tap,models[0 if step<2 else 1],case)
                    self.assertEqual(result['native_boundary_counts']['linear'],400)
                    self.assertTrue(result['native_noop']['CPU_BF16_gate_mul_then_add_byte_equal'])
            self.assertIs(tap.runtime.apply,original_apply)
            self.assertEqual(len(saved),4)
            for case,pack,e in saved:
                self.assertEqual(set(pack),set(c.PACK_SCHEMA))
                self.assertEqual(pack['r_rows'].shape,(4,8));self.assertEqual(pack['u_rows'].shape,(4,8))
                self.assertEqual(pack['head_rows'].shape,(4,2));self.assertEqual(pack['rows'].tolist(),[0,2,5,7])
                self.assertEqual(e['head_weight_stride'],[1,8])
            self.assertFalse(torch.cuda.is_initialized())

    def test_wrong_original_Y_stops_without_accepting_any_label(self):
        with ExitStack() as stack:
            _,tap,models,current,saved=self.fixture(stack,wrong_operand=True)
            case=copy.deepcopy(cases()[0]);case['rows']=[0,2,5,7]
            with tap, self.assertRaisesRegex(ValueError,'operand bytes differ: Y'):
                self.invoke(tap,models[0],case)
            self.assertEqual(saved,[])

    def test_repeated_skipped_linear_or_QDQ_slot_is_rejected_before_extra_native_call(self):
        for wrong_suffix in ('self_attn.q','self_attn.v','ffn.0'):
            with ExitStack() as stack:
                torch,tap,models,current,saved=self.fixture(stack)
                case=copy.deepcopy(cases()[0]);case['rows']=[0,2,5,7]
                with tap:
                    tap.begin(case)
                    value=torch.ones(8,8,dtype=torch.bfloat16)
                    tap.runtime.apply(None,value,expert=m.EXPERTS[0],step_index=0,block_index=0,module_name='blocks.0.self_attn.q')
                    with self.subTest(suffix=wrong_suffix),self.assertRaisesRegex(ValueError,'linear module order'):
                        tap.runtime.apply(None,value,expert=m.EXPERTS[0],step_index=0,block_index=0,module_name='blocks.0.'+wrong_suffix)
                    self.assertEqual(tap.runtime.counters['linear_forward_calls'],1)

    def test_corrupted_event_permutation_with_same_counts_fails_final_exact_graph_gate(self):
        with ExitStack() as stack:
            _,tap,models,current,saved=self.fixture(stack)
            case=copy.deepcopy(cases()[0]);case['rows']=[0,2,5,7]
            original_finish=tap.finish
            def altered(actual):
                # Preserve400 linear and QDQ counts and10/block, but repeat q instead of k.
                for key in ('linear','qdq'):tap.events[key][1]=tap.events[key][0]
                return original_finish(actual)
            tap.finish=altered
            with tap,self.assertRaisesRegex(ValueError,'native graph incomplete'):
                self.invoke(tap,models[0],case)
            self.assertEqual(saved,[])

    def test_duplicate_QDQ_slot_and_reordered_pre_MM_stop_before_extra_original_call(self):
        with ExitStack() as stack:
            torch,tap,models,current,saved=self.fixture(stack)
            case=copy.deepcopy(cases()[0]);case['rows']=[0,2,5,7]
            qdq_calls=[]
            def original(value,force_fp32=True):qdq_calls.append(1);return value
            tap.adapter.native_hif4_activation_qdq=original
            with tap:
                tap.begin(case)
                value=torch.ones(8,8,dtype=torch.bfloat16)
                tap.slot=(0,0,'blocks.0.self_attn.q')
                tap.adapter.native_hif4_activation_qdq(value)
                with self.assertRaisesRegex(ValueError,'QDQ slot order'):
                    tap.adapter.native_hif4_activation_qdq(value)
                self.assertEqual(len(qdq_calls),1)
                tap.slot=None
                models[0].pre_weight.time_embedding_0.apply(value)
                with self.assertRaisesRegex(ValueError,'pre-infer operation order'):
                    models[0].pre_weight.time_embedding_0.apply(value)
                self.assertEqual(len(tap.events['pre_mm']),1)


if __name__=='__main__':unittest.main()
