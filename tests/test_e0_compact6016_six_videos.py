import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import subprocess
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('six', ROOT / 'tools/e0_compact6016_six_videos.py')
six = importlib.util.module_from_spec(spec)
spec.loader.exec_module(six)


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT / 'config/e0_compact6016_six_videos_20261028_draft.json').read_text())

    def test_contract_and_wrong_bank(self):
        self.assertTrue(six.validate_plan(self.plan))
        bad = copy.deepcopy(self.plan)
        bad['bank_schema'] = 'ffnds.compact6016.deployment_bank.v1'
        with self.assertRaises(ValueError): six.validate_plan(bad)

    def test_wrong_source_and_budget(self):
        for key in ('source', 'budget'):
            bad = copy.deepcopy(self.plan)
            if key == 'source': bad['cases'][0]['prompt'] += '!'
            else: bad['limits']['native_full_model_flows'] = 25
            with self.assertRaises(ValueError): six.validate_plan(bad)

    def test_accounting_is_predebited_and_bounded(self):
        events = []
        ledger = six.Ledger({'flow': 1}, lambda r: events.append(dict(r['predebited'])))
        with self.assertRaises(ValueError): ledger.observe({'flow': 1})
        ledger.reserve(six.RUN_IDS[0], {'flow': 1})
        self.assertEqual(events[0], {'flow': 1})
        with self.assertRaises(ValueError): ledger.reserve(six.RUN_IDS[0], {'flow': 1})
        ledger.observe({'flow': 1})
        with self.assertRaises(ValueError): ledger.observe({'flow': 1})
        with self.assertRaises(ValueError): ledger.reserve('unregistered', {'flow': 0})

    def test_binding_sha_and_escape(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a'; p.write_bytes(b'actual')
            binding = {'path': 'a', 'bytes': 6, 'sha256': six.digest(p)}
            self.assertEqual(six.bind_file(d, binding), p)
            p.write_bytes(b'alter!')
            with self.assertRaises(ValueError): six.bind_file(d, binding)
            binding['path'] = '../'
            with self.assertRaises(ValueError): six.bind_file(d, binding)

    def test_missing_complete_bank_is_rejected_before_model_import(self):
        with self.assertRaisesRegex(ValueError, 'NOT_BOUND'):
            six.validate_plan(self.plan, frozen=True)
        with self.assertRaisesRegex(ValueError, 'NOT_BOUND'):
            six.bind_file(ROOT, None)

    def test_import_is_stdlib_and_does_not_import_torch(self):
        code = "import sys;sys.path.insert(0,'tools');import e0_compact6016_six_videos;assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_independent_cpu_exit_and_exact_source_binding_are_required(self):
        import hashlib
        protocol_sha = 'a' * 64
        report = {'id': six.ID, 'status': 'PASS_CPU_BYTES_NATIVE_VIDEO_NOT_EXECUTED',
                  'protocol_sha256': protocol_sha, 'bank_manifest_sha256': six.BANK_SHA,
                  'source_bindings_sha256': hashlib.sha256(json.dumps(self.plan['repo_sources'], sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                  'mixed_bank_cells': 160, 'actual_folded_export_payload_verifications': [{}] * 160,
                  'NumPy_or_Torch_imported': False, 'all_CUDA_devices_hidden': True,
                  'bank_loaded_for_native': False, 'model_calls': 0, 'GPU_calls': 0}
        receipt = {'id': six.ID, 'protocol_sha256': protocol_sha,
                   'launcher_returncode': 0, 'process_exited': True}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def scope_for(payload, exited):
                run = root / 'CPU.json'; run.write_text(json.dumps(payload))
                exit_path = root / 'EXIT.json'
                exit_path.write_text(json.dumps(dict(exited, RUN_sha256=six.digest(run))))
                binding = lambda p: {'path': p.name, 'bytes': p.stat().st_size, 'sha256': six.digest(p)}
                return {'cpu_preflight': {'binding': binding(run), 'launch_exit': binding(exit_path)}}
            scope = scope_for(report, receipt)
            self.assertEqual(six.validate_prior_cpu(scope, self.plan, protocol_sha, {'hf': root}), scope['cpu_preflight'])
            for key, value in (('source_bindings_sha256', 'b' * 64), ('protocol_sha256', 'b' * 64),
                               ('NumPy_or_Torch_imported', True), ('all_CUDA_devices_hidden', False),
                               ('bank_loaded_for_native', True)):
                bad = dict(report, **{key: value})
                with self.assertRaisesRegex(ValueError, 'Prior independent CPU'):
                    six.validate_prior_cpu(scope_for(bad, receipt), self.plan, protocol_sha, {'hf': root})
            for key, value in (('launcher_returncode', 1), ('process_exited', False)):
                with self.assertRaisesRegex(ValueError, 'Prior independent CPU'):
                    six.validate_prior_cpu(scope_for(report, dict(receipt, **{key: value})), self.plan, protocol_sha, {'hf': root})
            with self.assertRaisesRegex(ValueError, 'NOT_BOUND'):
                six.validate_prior_cpu({}, self.plan, protocol_sha, {'hf': root})

    def test_six_actual_jobs_share_noise_only_within_source(self):
        cohort, checked = six.registered_cohort(self.plan, ROOT)
        self.assertFalse(checked['errors'])
        self.assertEqual(checked['summary']['independent_image_clusters'], 2)
        self.assertEqual([j['run_id'] for j in cohort['samples']], list(six.RUN_IDS))
        for case in self.plan['cases']:
            jobs = [j for j in cohort['samples'] if j['source_id'] == case['source_id']]
            self.assertEqual(len({j['initial_noise_id'] for j in jobs}), 1)
            self.assertEqual(len({j['asset_id'] for j in jobs}), 3)

    def test_six_job_views_are_accepted_by_existing_registered_generation_collector(self):
        from ffnds_stage_registered_i2v import collect
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cohort, _ = six.registered_cohort(self.plan, root)
            for index, job in enumerate(cohort['samples']):
                source = next(s for s in cohort['sources'] if s['source_id'] == job['source_id'])
                self.assertTrue(source['image_name'].endswith('.jpg'))
                folder = root / job['run_id']; folder.mkdir()
                video = folder / 'output.mp4'; video.write_bytes(b'synthetic collector placeholder')
                noise = folder / 'initial_noise.safetensors'; noise.write_bytes(b'same actual synthetic initial')
                record = {'schema': 'ffnds.registered.native.video.v1', 'status': 'COMPLETE_NATIVE_GENERATION_ONLY',
                          'job': job, 'source': source, 'native_ffn_sites': 160, 'scheduler_generator_reset': True,
                          'video_path': str(video), 'actual_initial_noise_file': str(noise), 'decoded_frames': 81,
                          'decoded_size': [832, 464], 'native_ffn_counters': {'linear_forward_calls': 1600}}
                (folder / 'RUN.json').write_text(json.dumps(record))
                (root / f'job{index:06d}').symlink_to(job['run_id'], target_is_directory=True)
            checked = collect(cohort, root)
            self.assertEqual(checked['phase'], 'COMPLETE_REGISTERED_GENERATION_VERIFIED')
            self.assertEqual(len(checked['samples']), 6)
            self.assertTrue(all(s['status'] == 'COMPLETED' for s in checked['samples']))

    def test_failed_actual_call_retains_debit_and_journal_before_invocation(self):
        events = []
        ledger = six.Ledger({'flow': 1}, journal=lambda e: events.append(dict(e)))
        def failed():
            self.assertEqual(events[0]['kind'], 'reserve_before_call')
            self.assertEqual(ledger.record['predebited'], {'flow': 1})
            raise RuntimeError('actual call failed')
        with self.assertRaisesRegex(RuntimeError, 'actual call failed'):
            ledger.call(six.RUN_IDS[0], {'flow': 1}, failed)
        self.assertEqual(ledger.record['observed'], {})
        self.assertFalse(ledger.complete())
        self.assertEqual(len(events), 1)

    def test_neutral_comparison_requires_initial_all_post_and_video(self):
        case = self.plan['cases'][0]
        old = case['historical_dense']
        phase = {'initial_noise': {'raw_tensor_sha256': old['initial_raw_BF16_sha256']},
                 'trajectory': [{'raw_tensor_sha256': x} for x in old['post_raw_FP32_sha256']],
                 'video': {'sha256': old['video']['sha256']}}
        self.assertTrue(six.compare_historical(phase, case)['passed'])
        for item in ('initial', 'post', 'video', 'missing'):
            bad = copy.deepcopy(phase)
            if item == 'initial': bad['initial_noise']['raw_tensor_sha256'] = 'wrong'
            elif item == 'post': bad['trajectory'][2]['raw_tensor_sha256'] = 'wrong'
            elif item == 'video': bad['video']['sha256'] = 'wrong'
            else: bad['trajectory'].pop()
            with self.subTest(item=item): self.assertFalse(six.compare_historical(bad, case)['passed'])

    def test_noise_comparison_requires_separate_real_files_and_equal_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = root / 'dense', root / 'candidate'
            left.write_bytes(b'real bits'); right.write_bytes(b'real bits')
            def item(p):
                return {'path': p.name, 'sha256': six.digest(p), 'bytes': p.stat().st_size,
                        'raw_tensor_sha256': 'same', 'shape': [2], 'dtype': 'bfloat16'}
            self.assertTrue(six.verify_noise_pair(item(right), item(left), root))
            with self.assertRaises(ValueError): six.verify_noise_pair(item(left), item(left), root)
            right.write_bytes(b'new bits!')
            with self.assertRaises(ValueError): six.verify_noise_pair(item(right), item(left), root)

    def test_active_scope_must_match_exact_protocol_budget_and_run(self):
        plan = copy.deepcopy(self.plan)
        plan.update(assets_finalized=True, bank_binding_status='BOUND_COMPLETE_MIXED160')
        authority = {'id': six.AUTHORIZATION, 'status': 'ACTIVE_CURRENT_USER_DIRECTED_SESSION',
                     'new_scope_registration_allowed': True, 'max_simultaneously_assigned_idle_GPUs': 99}
        scope = {'id': six.ID, 'protocol_sha256': 'a'*64, 'allowed_run_ids': list(six.RUN_IDS),
                 'execution_status': 'AUTHORIZED_NOT_STARTED', 'limits': six.LIMITS,
                 'max_simultaneously_assigned_idle_GPUs': 1, 'session_max_simultaneously_assigned_idle_GPUs': 2}
        active = {'user_continuation_authorization_20261027': authority,
                  'current_session_resource_constraint_20261028': {'max_simultaneously_assigned_idle_GPUs': 2},
                  'resource_scheduling_policy_20261028': {'mode': 'DYNAMIC_MAX_ACTUALLY_CALLABLE', 'global_gpu_card_cap': None},
                  'experiment_authority': {'current_user_authorized_scopes': [scope]}}
        six.authorize(plan, 'a'*64, active, six.RUN_IDS[0])
        for change in ({'protocol_sha256': 'b'*64}, {'execution_status': 'FAILED'}, {'allowed_run_ids': []}):
            altered = copy.deepcopy(active)
            altered['experiment_authority']['current_user_authorized_scopes'][0].update(change)
            with self.assertRaises(ValueError): six.authorize(plan, 'a'*64, altered)
        with self.assertRaises(ValueError): six.authorize(plan, 'a'*64, active, 'unregistered')

    def test_native_coverage_rejects_missing_site_and_corrupt_elapsed(self):
        timings = [{'step': s, 'block': b, 'elapsed_ms': 1.} for s in range(4) for b in range(40)]
        counts = {'linear_forward_calls': 1600, 'activation_qdq_calls': 1600, 'weight_transfers': 0, 'cache_misses': 0}
        six.validate_native_counts(counts, timings)
        with self.assertRaises(ValueError): six.validate_native_counts(counts, timings[:-1])
        timings[0]['elapsed_ms'] = float('nan')
        with self.assertRaises(ValueError): six.validate_native_counts(counts, timings)

    def test_disjoint_memory_envelope_keeps_original_buffers_and_no_dense_bank_overlap(self):
        envelope = six.memory_envelope(self.plan['physical_cache'], self.plan['resource_limits'])
        self.assertEqual(envelope['maximum_disjoint_resident_bytes'], 56203673600)
        self.assertEqual(envelope['candidate_resident_bytes'], 43416391680)
        self.assertFalse(envelope['dense_plus_candidate_simultaneously'])
        self.assertEqual(envelope['candidate_original_decoded_FFN_matrices'], 0)
        bad = copy.deepcopy(self.plan['physical_cache'])
        bad['candidate_total_resident_tensor_bytes_per_init'] += bad['dense_decoded_matrix_bytes']
        with self.assertRaisesRegex(ValueError, 'sum differs'):
            six.memory_envelope(bad, self.plan['resource_limits'])

    def test_no_teacher_observer_or_wrong_runtime_can_activate(self):
        runtime = object()
        adapter = SimpleNamespace(_ACTIVE_OBSERVER=None, _ACTIVE_RUNTIME=runtime)
        six.require_hook0(adapter, runtime)
        adapter._ACTIVE_OBSERVER = object()
        with self.assertRaisesRegex(RuntimeError, 'zero teacher'):
            six.require_hook0(adapter, runtime)
        adapter._ACTIVE_OBSERVER = None
        adapter._ACTIVE_RUNTIME = object()
        with self.assertRaises(RuntimeError): six.require_hook0(adapter, runtime)


class PayloadByteControls(unittest.TestCase):
    def fixture(self, root):
        import torch
        from safetensors.torch import save_file
        from ub1_consumer_runtime import tensor_sha
        pack = {'W1': torch.zeros(4, 4, dtype=torch.bfloat16), 'W2': torch.zeros(4, 4, dtype=torch.bfloat16),
            'b1': torch.zeros(4, dtype=torch.bfloat16), 'b2': torch.zeros(4, dtype=torch.bfloat16),
            'smooth1': torch.ones(4), 'smooth2': torch.ones(4), 'groups': torch.tensor([0, 2]),
            'channels': torch.tensor([0, 1, 4, 5]), 'training_lora_A1': torch.zeros(2, 4),
            'training_lora_B1': torch.zeros(4, 2), 'training_lora_A2': torch.zeros(2, 4), 'training_lora_B2': torch.zeros(4, 2)}
        path = root / 'folded.safetensors'
        def save():
            save_file(pack, str(path))
            return {'run_id': 'tiny_export', 'groups': [0, 2], 'export': {'bytes': path.stat().st_size, 'sha256': six.digest(path),
                'tensors': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': tensor_sha(v)} for k,v in pack.items()}}}
        return pack, path, save

    def verify(self, path, cell):
        return six.verify_folded_payload(path, cell, dim=4, hidden=8, width=4, group=2, rank=2)

    def test_full_file_and_all_tensor_payloads_are_real_bytes_not_metadata_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack, path, save = self.fixture(Path(tmp))
            cell = save()
            result = self.verify(path, cell)
            self.assertEqual(result['tensor_payloads_verified'], 12)
            self.assertFalse(result['native_loaded'])
            data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'payload SHA'):
                self.verify(path, cell)

    def test_rebound_wrong_group_payload_and_same_element_wrong_shape_are_rejected(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            pack, path, save = self.fixture(Path(tmp))
            pack['groups'] = torch.tensor([1, 2])
            cell = save()  # All fresh hashes agree, but source support differs.
            with self.assertRaisesRegex(ValueError, 'coordinates'):
                self.verify(path, cell)
            pack['groups'] = torch.tensor([0, 2]); pack['W2'] = pack['W2'].reshape(8, 2)
            cell = save()
            with self.assertRaisesRegex(ValueError, 'shape'):
                self.verify(path, cell)


def fake_native():
    """A Python-only call topology; no real model tensor/GEMM or GPU."""
    def identity(*args, **kwargs): return args[0] if args else None
    adapter = SimpleNamespace(native_hif4_activation_qdq=lambda x, **kw: x)
    scheduler = SimpleNamespace(step_index=0, generator=None, prepare=identity, prepare_latents=identity)
    def step_pre(step_index): scheduler.step_index = step_index
    scheduler.step_pre = step_pre
    names = lambda b: [f'blocks.{b}.{name}' for name in six.LINEAR_SUFFIXES]
    records = {e: {name: {} for b in range(40) for name in names(b)}
               for e in ('high_noise_model', 'low_noise_model')}
    base = SimpleNamespace(records=records, counters=dict.fromkeys(
        ('linear_forward_calls', 'activation_qdq_calls', 'weight_transfers', 'cache_misses'), 0))
    def apply(mm, x, **kw):
        x = adapter.native_hif4_activation_qdq(x, force_fp32=True)
        base.counters['activation_qdq_calls'] += 1
        base.counters['linear_forward_calls'] += 1
        return x
    base.apply = apply
    models = []
    for expert in range(2):
        infer = SimpleNamespace(block_idx=0, offload_manager=SimpleNamespace(offload_granularity='block',
            lazy_load=False, need_init_first_buffer=True, cuda_buffers=[SimpleNamespace(load_state_dict=identity) for _ in range(2)]))
        def ffn(x, _infer=infer, _expert=expert):
            e = 'high_noise_model' if _expert == 0 else 'low_noise_model'
            for slot in (0, 2):
                x = base.apply(None, x, expert=e, step_index=scheduler.step_index, block_index=_infer.block_idx,
                               module_name=f'blocks.{_infer.block_idx}.ffn.{slot}')
            return x
        infer.infer_ffn = ffn
        def block(x, _infer=infer, _expert=expert):
            e = 'high_noise_model' if _expert == 0 else 'low_noise_model'
            for i in range(8):
                x = base.apply(None, x, expert=e, step_index=scheduler.step_index, block_index=_infer.block_idx,
                               module_name=f'blocks.{_infer.block_idx}.{six.LINEAR_SUFFIXES[i]}')
            return _infer.infer_ffn(x)
        infer.infer_block = block
        pre_weight = SimpleNamespace(**{name: SimpleNamespace(apply=identity) for name in
            ('time_embedding_0', 'time_embedding_2', 'time_projection_1', 'text_embedding_0', 'text_embedding_2', 'patch_embedding')})
        models.append(SimpleNamespace(transformer_infer=infer, transformer_weights=SimpleNamespace(
            norm=SimpleNamespace(apply=identity), head=SimpleNamespace(apply=identity)),
            pre_infer=SimpleNamespace(infer=identity), post_infer=SimpleNamespace(infer=identity), pre_weight=pre_weight))
    model_wrapper = SimpleNamespace(model=models)
    def flow(x):
        model = models[0 if scheduler.step_index < 2 else 1]
        model.pre_infer.infer(x)
        for operation in vars(model.pre_weight).values(): operation.apply(x)
        manager = model.transformer_infer.offload_manager
        if manager.need_init_first_buffer:
            manager.cuda_buffers[0].load_state_dict({}, 0)
            manager.need_init_first_buffer = False
        for block in range(40):
            model.transformer_infer.block_idx = block
            manager.cuda_buffers[1].load_state_dict({}, (block + 1) % 40)
            x = model.transformer_infer.infer_block(x)
        x = model.transformer_weights.norm.apply(x)
        x = model.transformer_weights.head.apply(x)
        return model.post_infer.infer(x)
    model_wrapper.infer = flow
    runner = SimpleNamespace(model=model_wrapper, scheduler=scheduler, run_input_encoder=identity,
        run_text_encoder=identity, run_vae_encoder=identity, run_vae_decoder=identity, run_image_encoder=identity)
    return SimpleNamespace(runner=runner), adapter, base


class NativePathControls(unittest.TestCase):
    def test_swapped_modules_and_duplicate_QDQ_fail_before_unregistered_work(self):
        pipe, adapter, base = fake_native()
        ledger = six.Ledger(six.LIMITS)
        with six.NativeCallTap(pipe, adapter, base, base, ledger, six.RUN_IDS[0], 'dense'):
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                base.apply(None, object(), expert='high_noise_model', step_index=0, block_index=0, module_name='blocks.0.self_attn.k')
        pipe, adapter, base = fake_native()
        def double(mm, x, **kw):
            x = adapter.native_hif4_activation_qdq(x, force_fp32=True)
            return adapter.native_hif4_activation_qdq(x, force_fp32=True)
        base.apply = double
        with six.NativeCallTap(pipe, adapter, base, base, six.Ledger(six.LIMITS), six.RUN_IDS[0], 'dense'):
            with self.assertRaisesRegex(ValueError, 'Duplicate activation'):
                base.apply(None, object(), expert='high_noise_model', step_index=0, block_index=0, module_name='blocks.0.self_attn.q')

    def test_target_None_stops_before_adapter_can_fall_back_to_original_teacher(self):
        pipe, adapter, base = fake_native()
        base.apply = lambda *args, **kwargs: None
        ledger = six.Ledger(six.LIMITS)
        with six.NativeCallTap(pipe, adapter, base, base, ledger, six.RUN_IDS[0], 'dense'):
            with self.assertRaisesRegex(RuntimeError, 'fallback'):
                base.apply(None, object(), expert='high_noise_model', step_index=0, block_index=0, module_name='blocks.0.self_attn.q')
        self.assertEqual(ledger.record['predebited']['native_non_FFN_linears'], 1)
        self.assertEqual(ledger.record['observed'].get('native_non_FFN_linears', 0), 0)
    def test_actual_wrappers_cover_four_steps_and_restore_every_method(self):
        pipe, adapter, base = fake_native()
        runner, scheduler = pipe.runner, pipe.runner.scheduler
        before = (base.apply, runner.model.infer, scheduler.step_pre, adapter.native_hif4_activation_qdq)
        ledger = six.Ledger(six.LIMITS)
        pre_steps = []
        with six.NativeCallTap(pipe, adapter, base, base, ledger, six.RUN_IDS[0], 'dense',
                               after_step_pre=lambda: pre_steps.append(scheduler.step_index)) as tap:
            runner.run_input_encoder(); runner.run_text_encoder(); runner.run_vae_encoder()
            scheduler.prepare(); scheduler.prepare_latents(42, six.VIDEO['latent_shape'], dtype='torch.float32')
            for step in range(4):
                scheduler.step_pre(step_index=step)
                runner.model.infer(object())
            runner.run_vae_decoder()
            tap.verify_complete()
            self.assertEqual(tap.report()['native_linear_dispatches'], 1600)
            self.assertEqual(ledger.record['observed']['native_FFN_linears'], 320)
            self.assertEqual(ledger.record['observed']['native_FFN_activation_QDQ'], 320)
            self.assertEqual(ledger.record['observed']['native_non_FFN_linears'], 1280)
            self.assertEqual(ledger.record['observed']['native_non_FFN_activation_QDQ'], 1280)
            with self.assertRaisesRegex(RuntimeError, 'CLIP'): runner.run_image_encoder()
        self.assertEqual(pre_steps, [0, 1, 2, 3])
        self.assertEqual(before, (base.apply, runner.model.infer, scheduler.step_pre, adapter.native_hif4_activation_qdq))

    def test_duplicate_native_linear_and_wrong_expert_are_rejected_before_call(self):
        pipe, adapter, base = fake_native()
        ledger = six.Ledger(six.LIMITS)
        with six.NativeCallTap(pipe, adapter, base, base, ledger, six.RUN_IDS[0], 'dense'):
            for name in six.LINEAR_SUFFIXES[:8]:
                base.apply(None, object(), expert='high_noise_model', step_index=0, block_index=0, module_name='blocks.0.' + name)
            kw = dict(expert='high_noise_model', step_index=0, block_index=0, module_name='blocks.0.ffn.0')
            value = object()
            base.apply(None, value, **kw)
            with self.assertRaisesRegex(ValueError, 'duplicate'): base.apply(None, value, **kw)
            kw.update(expert='low_noise_model', module_name='blocks.0.ffn.2')
            with self.assertRaisesRegex(ValueError, 'Unknown'): base.apply(None, value, **kw)
        self.assertEqual(ledger.record['predebited']['native_FFN_linears'], 1)

    def test_partial_wrapper_install_restores_sources_on_unsupported_offload(self):
        pipe, adapter, base = fake_native()
        old = (base.apply, adapter.native_hif4_activation_qdq, pipe.runner.run_input_encoder)
        pipe.runner.model.model[1].transformer_infer.offload_manager.lazy_load = True
        with self.assertRaisesRegex(ValueError, 'non-lazy'):
            with six.NativeCallTap(pipe, adapter, base, base, six.Ledger(six.LIMITS), six.RUN_IDS[0], 'dense'): pass
        self.assertEqual(old, (base.apply, adapter.native_hif4_activation_qdq, pipe.runner.run_input_encoder))

    def test_original_checkpoint_buffer_load_is_called_with_same_objects_and_counts_FFN_bytes(self):
        import torch
        pipe, adapter, base = fake_native()
        buffer = pipe.runner.model.model[0].transformer_infer.offload_manager.cuda_buffers[0]
        calls = []
        marker = object()
        def original(destination, block_index, adapter_block_index=None):
            calls.append((destination, block_index, adapter_block_index))
            return marker
        buffer.load_state_dict = original
        source = {'blocks.0.ffn.0.weight': torch.zeros(2, 3, dtype=torch.bfloat16),
                  'blocks.0.ffn.0.bias': torch.zeros(2, dtype=torch.bfloat16),
                  'blocks.0.attn.weight': torch.zeros(1, dtype=torch.bfloat16)}
        with six.NativeCallTap(pipe, adapter, base, base, six.Ledger(six.LIMITS), six.RUN_IDS[0], 'dense') as tap:
            self.assertIs(buffer.load_state_dict(source, 0), marker)
            self.assertIs(calls[0][0], source)
            observed = tap.report()['checkpoint_block_buffer_loads'][0]
            self.assertEqual(observed['source_tensor_bytes'], 18)
            self.assertEqual(observed['source_FFN_tensor_bytes'], 16)
            self.assertEqual(observed['status'], 'ORIGINAL_BLOCK_BUFFER_LOAD_RETURNED')
        self.assertIs(buffer.load_state_dict, original)

    def test_candidate_own_quantizer_is_intercepted_and_original_quantizer_restored(self):
        pipe, adapter, base = fake_native()
        candidate = SimpleNamespace(quantizer=adapter.native_hif4_activation_qdq)
        def apply(mm, value, **kw):
            return candidate.quantizer(value, force_fp32=True)
        candidate.apply = apply
        original = candidate.quantizer
        ledger = six.Ledger(six.LIMITS)
        with six.NativeCallTap(pipe, adapter, base, candidate, ledger, six.RUN_IDS[2], 'energy94'):
            value = object()
            for name in six.LINEAR_SUFFIXES[:8]:
                candidate.apply(None, value, expert='high_noise_model', step_index=0, block_index=0, module_name='blocks.0.' + name)
            self.assertIs(candidate.apply(None, value, expert='high_noise_model', step_index=0,
                block_index=0, module_name='blocks.0.ffn.0'), value)
            self.assertEqual(ledger.record['observed']['native_narrow6016_FFN_linears'], 1)
            self.assertEqual(ledger.record['observed']['native_FFN_activation_QDQ'], 1)
            self.assertEqual(ledger.record['observed'].get('native_original_width_FFN_linears', 0), 0)
        self.assertIs(candidate.quantizer, original)

    def phase_fixture(self, root):
        import torch
        from ub1_consumer_runtime import tensor_sha
        pipe, adapter, base = fake_native()
        scheduler = pipe.runner.scheduler
        scheduler.latents = torch.tensor([1., 2.], dtype=torch.bfloat16)
        scheduler.denoising_step_list = [1000, 750, 500, 250]
        scheduler.sigmas = [1., .9375, .8333333134651184, .625]
        def prepare_latents(seed, latent_shape, dtype=torch.float32):
            scheduler.generator = torch.Generator(device='cpu').manual_seed(seed)
            scheduler.latents = torch.randn(*latent_shape, dtype=dtype, generator=scheduler.generator)
        scheduler.prepare_latents = prepare_latents
        def step_pre(step_index):
            scheduler.step_index = step_index
            scheduler.latents = scheduler.latents.bfloat16()
        scheduler.step_pre = step_pre
        def post(): scheduler.latents += 1
        scheduler.step_post = post
        case = copy.deepcopy(json.loads((ROOT / 'config/e0_compact6016_six_videos_20261028_draft.json').read_text())['cases'][0])
        case['image']['path'] = 'input.jpg'
        (root / 'input.jpg').write_bytes(b'synthetic input not decoded')
        old = case['historical_dense']
        expected = torch.randn(2, dtype=torch.float32, generator=torch.Generator().manual_seed(42)).bfloat16()
        old['initial_raw_BF16_sha256'] = tensor_sha(expected)
        old['post_raw_FP32_sha256'] = []
        for step in range(4):
            expected.add_(1)
            old['post_raw_FP32_sha256'].append(tensor_sha(expected.float()))
        old['video']['sha256'] = __import__('hashlib').sha256(b'synthetic mp4 placeholder').hexdigest()
        def generate(**kwargs):
            runner = pipe.runner
            runner.run_input_encoder(); runner.run_text_encoder(); runner.run_vae_encoder()
            scheduler.prepare()
            scheduler.prepare_latents(kwargs['seed'], [2], dtype=torch.float32)
            for step in range(4):
                scheduler.step_pre(step_index=step)
                runner.model.infer(scheduler.latents)
                scheduler.step_post()
            runner.run_vae_decoder()
            Path(kwargs['save_result_path']).write_bytes(b'synthetic mp4 placeholder')
        pipe.generate = generate
        timer = SimpleNamespace(events=[], report=lambda: [{'step': s, 'block': b, 'elapsed_ms': .1}
                                                           for s in range(4) for b in range(40)])
        return torch, pipe, adapter, base, case, timer

    def phase_patches(self, stack, torch):
        from run_ub1_consumer_videos import inspect_video
        for name in ('reset_peak_memory_stats', 'synchronize'):
            stack.enter_context(mock.patch.object(torch.cuda, name, return_value=None))
        for name in ('max_memory_allocated', 'max_memory_reserved'):
            stack.enter_context(mock.patch.object(torch.cuda, name, return_value=0))
        video = dict(six.VIDEO, latent_shape=[2])
        stack.enter_context(mock.patch.object(six, 'VIDEO', video))
        def inspect(path, contract):
            return {'path': str(path), 'sha256': six.digest(path), 'frames': 81, 'fps': 16., 'size': [832, 464], 'bytes': path.stat().st_size}
        stack.enter_context(mock.patch('run_ub1_consumer_videos.inspect_video', side_effect=inspect))

    def test_real_phase_plumbing_saves_exact_initial_four_posts_and_restores_scheduler(self):
        import torch
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            torch, pipe, adapter, base, case, timer = self.phase_fixture(root)
            self.phase_patches(stack, torch)
            original_post = pipe.runner.scheduler.step_post
            out = root / 'output'; out.mkdir()
            ledger = six.Ledger(six.LIMITS)
            phase = six.generate_phase(pipe, adapter, base, base, 'dense', case, out, root, ledger, timer,
                                       lambda *a: None, lambda *a: None, lambda: None)
            self.assertEqual(phase['completed_native_flow_steps'], 4)
            self.assertTrue(phase['historical_comparison']['passed'])
            initial = load_file(str(root / phase['initial_noise']['path']))['latents']
            self.assertEqual(initial.dtype, torch.bfloat16)
            from ub1_consumer_runtime import tensor_sha
            self.assertEqual(tensor_sha(initial), case['historical_dense']['initial_raw_BF16_sha256'])
            for step, record in enumerate(phase['trajectory']):
                post = load_file(str(root / record['path']))['latents']
                self.assertEqual(post.dtype, torch.float32)
                self.assertEqual(tensor_sha(post), case['historical_dense']['post_raw_FP32_sha256'][step])
                self.assertEqual(record['native_dtype_before_FP32_storage'], 'torch.bfloat16')
            self.assertIs(pipe.runner.scheduler.step_post, original_post)
            self.assertEqual(ledger.record['observed']['native_full_model_flows'], 4)
            self.assertFalse(torch.cuda.is_initialized())
            raw = load_file(str(root / phase['raw_FP32_noise']['path']))['latents']
            self.assertEqual(raw.dtype, torch.float32)
            self.assertEqual(ledger.record['observed']['actual_raw_FP32_noise_files'], 1)
            self.assertEqual(ledger.record['observed']['matched_offload_buffer_resets'], 2)
            self.assertEqual(ledger.record['observed']['original_checkpoint_block_buffer_loads'], 162)

    def test_wrong_initial_stops_before_first_native_flow_and_keeps_failed_debits(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            torch, pipe, adapter, base, case, timer = self.phase_fixture(root)
            self.phase_patches(stack, torch)
            case['historical_dense']['initial_raw_BF16_sha256'] = 'wrong'
            out = root / 'output'; out.mkdir()
            ledger = six.Ledger(six.LIMITS)
            with self.assertRaisesRegex(ValueError, 'before first native flow'):
                six.generate_phase(pipe, adapter, base, base, 'dense', case, out, root, ledger, timer,
                                   lambda *a: None, lambda *a: None, lambda: None)
            self.assertEqual(ledger.record['predebited']['video_generations'], 1)
            self.assertEqual(ledger.record['observed'].get('video_generations', 0), 0)
            self.assertEqual(ledger.record['predebited'].get('native_full_model_flows', 0), 0)
            self.assertEqual(base.counters['linear_forward_calls'], 0)
            record = json.loads((out / six.RUN_IDS[0] / 'PHASE.json').read_text())
            self.assertEqual(record['status'], 'FAILED_NATIVE_PHASE_PRESERVED_NO_RETRY')


if __name__ == '__main__': unittest.main()
