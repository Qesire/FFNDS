"""Pure CPU software controls; no tensor, VAE, CUDA or real video execution."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import run_e0_saved_flow_decode as driver


class DecodeFixture:
    def __init__(self, base):
        self.out = base / 'render'
        self.out.mkdir()
        self.plan = {'id': 'SYNTHETIC_SOFTWARE_ONLY', 'cudnn_allow_tf32': True,
                     'driver_sources_sha256': {}, 'native_source_sha256': {}}
        for group, names in [('driver_sources_sha256', {'tools/fixture.py'}),
                             ('native_source_sha256', driver.MANDATORY_NATIVE_SOURCES)]:
            for name in names:
                path = self.out / 'EXECUTED_SOURCES' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic source ' + name)
                self.plan[group][name] = driver.sha256(path)
        self.prepared = []
        for index, seed in [(0, 42), (12, 54)]:
            tensor = base / f'fixture{index}.safetensors'
            tensor.write_bytes(b'SOFTWARE_FIXTURE_NOT_A_TENSOR' + bytes([index]))
            reference = base / f'reference{index}.mp4'
            reference.write_bytes(b'SOFTWARE_FIXTURE_NOT_A_VIDEO' + bytes([index]))
            receipt = base / f'reference{index}.json'
            receipt.write_text('{}')
            identity = {'source_id': f'image{index}', 'seed': seed, 'expected_frames': 81,
                'input_prompt_digest': f'prompt{index}', 'model_digest': f'dense_model{index}',
                'protocol_sha256': 'parent_protocol', 'run_sha256': driver.sha256(receipt)}
            self.prepared.append({'pair': {'source_index': index, 'source_id': f'image{index}', 'seed': seed},
                'tensor': str(tensor), 'tensor_sha256': driver.sha256(tensor),
                'flow_RUN_sha256': f'flow{index}', 'flow_sidecar_sha256': f'sidecar{index}',
                'dense_native_identity': f'native_dense{index}',
                'latent_native_identities': {'z_dense_after': f'native_dense{index}',
                                            'z_dense_at_student_after': f'native_hybrid{index}'},
                'assets': {'checkpoint_sha256': 'checkpoint', 'cache_manifest_sha256': 'cache'},
                'bank_RUN_sha256': 'bank',
                'reference': {'video': str(reference), 'video_sha256': driver.sha256(reference),
                    'receipt': str(receipt), 'receipt_sha256': driver.sha256(receipt), 'identity': identity}})
        binding = driver.decoder_binding(self.plan)
        self.run = {'id': self.plan['id'], 'protocol_sha256': 'protocol',
            'render_attempts_started': 0, 'VAE_decodes_started': 0, 'VAE_decodes_completed': 0,
            'video_encodes_started': 0, 'video_encodes_completed': 0,
            'denoiser_NFE': 0, 'FFN_calls': 0, 'videos': [],
            'decoder_binding': binding, 'decoder_digest': driver.object_sha(binding),
            'decoder_configuration': {**driver.DECODE_CONFIG, 'vae_path': 'SHA_BOUND_Wan2.1_VAE.pth'},
            'cost_scope': 'VAE_ONLY_RENDER_NOT_FULL_GENERATION', 'setup_seconds': 1.,
            'peak_gpu_memory_mib': 0.}
        self.receipt = self.out / 'DECODE_RUN.json'

    def renderer(self, source, key, output, mark):
        mark('decode_started')
        mark('decode_completed', decode_seconds=.01,
             input_latent_native_identity=source['latent_native_identities'][key])
        mark('encoding_started')
        output.write_bytes(Path(source['reference']['video']).read_bytes() if key == 'z_dense_after'
                           else b'SYNTHETIC_HYBRID_MEDIA')
        mark('encoding_completed', encode_seconds=.02)
        return {'decoded_shape': [81, 464, 832, 3]}

    def render(self, renderer=None):
        driver.render_inventory(self.out, self.run, self.prepared, renderer or self.renderer,
                                lambda: None, lambda: None)

    def freeze(self):
        self.run['actual_media_inventory'] = [{'path': str(p.relative_to(self.out)),
            'sha256': driver.sha256(p), 'bytes': p.stat().st_size} for p in sorted(self.out.rglob('*.mp4'))]
        driver.write_json(self.receipt, self.run)

    def evaluate(self, failure=None):
        calls = []
        def evaluate(video, reference, metadata):
            calls.append((Path(video).name, copy.deepcopy(metadata)))
            if failure == Path(video).name:
                raise ValueError('injected media decoding failure')
            # Exercise the real evaluator's required metadata access for orphans.
            assert metadata['identity']['expected_frames'] == 81
            return {'identity': metadata['identity'], 'reference_identity': metadata.get('reference_identity'),
                    'runtime': metadata['cost'], 'quality_accepted': False}
        metric = types.SimpleNamespace(evaluate=evaluate, validate=lambda report: [])
        gate = types.SimpleNamespace(inspect=lambda bundle: [])
        with patch.dict(sys.modules, {'ffnds_video_metrics': metric, 'ffnds_video_metric_gate': gate}):
            errors = driver.evaluate_inventory(self.out, self.receipt)
        return errors, calls

    def inspect(self):
        return driver.inspect_inventory(self.out, self.receipt, self.plan, self.prepared, 'protocol')


class SavedDecodeTests(unittest.TestCase):
    def test_module_import_is_torch_free(self):
        code = 'import sys; sys.path.insert(0,sys.argv[1]); import run_e0_saved_flow_decode; assert "torch" not in sys.modules'
        result = subprocess.run([sys.executable, '-B', '-c', code, str(TOOLS)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_raw_key_identity_includes_only_named_bf16_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'software_header.safetensors'
            length = 2 * 16 * 21 * 58 * 104
            header = {'z': {'dtype': 'BF16', 'shape': driver.SHAPE, 'data_offsets': [7, 7 + length]}}
            raw = json.dumps(header).encode()
            with path.open('wb') as stream:
                stream.write(struct.pack('<Q', len(raw)) + raw + b'prefix!')
                stream.write(b'\0' * length)
                stream.write(b'not part of named tensor')
            native_header = {'dtype': 'torch.bfloat16', 'shape': driver.SHAPE,
                             'storage': 'contiguous_little_endian'}
            expected = hashlib.sha256(json.dumps(native_header, sort_keys=True, separators=(',', ':')).encode()
                                      + b'\0' + b'\0' * length).hexdigest()
            self.assertEqual(driver.latent_header(path, 'z')['native_identity'], expected)
            with path.open('r+b') as stream:
                stream.truncate(8 + len(raw) + 7 + length - 1)
            with self.assertRaises(ValueError):
                driver.latent_header(path, 'z')

    def test_all_four_complete_before_frozen_receipt_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = DecodeFixture(Path(temp))
            fixture.render()
            fixture.freeze()
            original = driver.sha256(fixture.receipt)
            errors, calls = fixture.evaluate()
            self.assertEqual(errors, [])
            self.assertEqual(len(calls), 4)
            self.assertEqual(fixture.inspect(), [])
            self.assertEqual(original, driver.sha256(fixture.receipt))
            keys = []
            for entry in fixture.run['videos']:
                report = json.loads((fixture.out / (Path(entry['path']).stem + '_metrics.json')).read_text())
                self.assertEqual(report['identity']['run_sha256'], original)
                self.assertEqual(report['shared_setup_cost_deduplication_key'], original)
                keys.append(report['cost_deduplication_key'])
            self.assertEqual(len(set(keys)), 4)

    def test_consistently_forged_receipt_and_metric_identity_are_rejected(self):
        # Recompute metric metadata from the forged receipt, so mere internal
        # consistency cannot catch these independent provenance failures.
        for field in ['model', 'role', 'decoder', 'reference', 'hybrid_input', 'tensor']:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                fixture = DecodeFixture(Path(temp))
                fixture.render()
                entry = fixture.run['videos'][1]
                if field == 'model':
                    entry['model_digest'] = 'forged_same_in_both'
                elif field == 'role':
                    entry.update(role='dense_control', tensor_key='z_dense_after')
                elif field == 'decoder':
                    fixture.run['decoder_binding']['configuration'] = {'z_dim': 99}
                    fixture.run['decoder_digest'] = driver.object_sha(fixture.run['decoder_binding'])
                    for item in fixture.run['videos']:
                        item['decoder_digest'] = fixture.run['decoder_digest']
                elif field == 'reference':
                    entry['reference'] = copy.deepcopy(entry['reference'])
                    entry['reference']['identity']['model_digest'] = 'wrong_dense_reference'
                elif field == 'hybrid_input':
                    entry['input_latent_native_identity'] = 'wrong_named_key_payload'
                else:
                    entry['tensor_key'] = 'z_dense_after'
                fixture.freeze()
                fixture.evaluate()
                self.assertTrue(fixture.inspect())

    def test_encoding_failure_preserves_decode_cost_and_partial_media(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = DecodeFixture(Path(temp))
            def fail(source, key, output, mark):
                mark('decode_started')
                mark('decode_completed', decode_seconds=.25,
                     input_latent_native_identity=source['latent_native_identities'][key])
                mark('encoding_started')
                output.write_bytes(b'PARTIAL_MP4_SOFTWARE_FIXTURE')
                raise RuntimeError('encoder failed after actual decode')
            with self.assertRaises(RuntimeError):
                fixture.render(fail)
            self.assertEqual(fixture.run['VAE_decodes_started'], 1)
            self.assertEqual(fixture.run['VAE_decodes_completed'], 1)
            self.assertEqual(fixture.run['video_encodes_started'], 1)
            self.assertEqual(fixture.run['video_encodes_completed'], 0)
            entry = fixture.run['videos'][0]
            self.assertEqual(entry['decode_seconds'], .25)
            self.assertGreaterEqual(entry['render_wall_seconds'], 0)
            self.assertEqual(entry['sha256'], driver.sha256(fixture.out / entry['path']))
            fixture.freeze()
            _, calls = fixture.evaluate()
            self.assertEqual(len(calls), 1)
            self.assertTrue(fixture.inspect())

    def test_wrong_neutral_mp4_stops_scope_with_preserved_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = DecodeFixture(Path(temp))
            def wrong(source, key, output, mark):
                details = fixture.renderer(source, key, output, mark)
                output.write_bytes(b'WRONG_NEUTRAL')
                return details
            with self.assertRaisesRegex(RuntimeError, 'Neutral dense'):
                fixture.render(wrong)
            self.assertEqual(fixture.run['render_attempts_started'], 1)
            self.assertEqual(fixture.run['videos'][0]['status'], 'FAILED_DENSE_DECODE_NEUTRALITY')
            self.assertTrue((fixture.out / fixture.run['videos'][0]['path']).is_file())

    def test_metric_failure_does_not_skip_other_actual_clips(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = DecodeFixture(Path(temp))
            fixture.render()
            fixture.freeze()
            errors, calls = fixture.evaluate(failure='source00_dense_control.mp4')
            self.assertEqual(len(calls), 4)
            self.assertTrue(errors)
            self.assertEqual(len(list(fixture.out.glob('*_metrics.json'))), 4)

    def test_orphan_gets_intrinsic_metrics_with_explicit_unknown_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = DecodeFixture(Path(temp))
            fixture.render()
            (fixture.out / 'orphan.mp4').write_bytes(b'UNREGISTERED_SOFTWARE_MEDIA')
            fixture.freeze()
            errors, calls = fixture.evaluate()
            self.assertTrue(errors)
            self.assertEqual(len(calls), 5)
            orphan = json.loads((fixture.out / 'orphan_orphan_forensic.json').read_text())
            self.assertIn('metrics_with_unverified_identity', orphan)
            identity = orphan['metrics_with_unverified_identity']['identity']
            self.assertEqual(identity, {'expected_frames': 81})
            self.assertTrue(fixture.inspect())

    def test_failed_attempt_still_consumes_one_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            study = Path(temp)
            driver.reserve_attempt(study, {'id': 'software'}, 'sha')
            summary = study / 'RUN.json'
            summary.write_text('{"status":"FAILED"}')
            driver.close_attempt(study, 'FAILED', summary)
            with self.assertRaisesRegex(RuntimeError, 'already consumed'):
                driver.reserve_attempt(study, {'id': 'software'}, 'sha')

    def test_artifact_paths_stay_inside_bound_root(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                driver.within(Path(temp), '../escape.mp4')


if __name__ == '__main__':
    unittest.main()
