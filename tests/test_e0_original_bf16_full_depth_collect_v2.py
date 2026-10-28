"""Small, portable storage/worker boundary controls: no numerical imports."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('full_depth_collect_v2_tested', ROOT/'tools/e0_original_bf16_full_depth_collect_v2.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class V2Controls(unittest.TestCase):
    def test_import_and_frozen_numeric_functions_unchanged(self):
        code = '''import sys
sys.path.insert(0, 'tools')
import e0_original_bf16_videos as n
import e0_original_bf16_full_depth_collect as v1
before=(n.ID, dict(n.LIMITS), list(n.RUN_IDS), n.write, n.binding, v1.native, v1.scope_id(0), v1.reserve_output_bytes)
import e0_original_bf16_full_depth_collect_v2 as v2
assert before==(n.ID, dict(n.LIMITS), list(n.RUN_IDS), n.write, n.binding, v1.native, v1.scope_id(0), v1.reserve_output_bytes)
assert v2.legacy.CollectorTap.mm_call.__code__.co_code==v1.CollectorTap.mm_call.__code__.co_code
assert v2.legacy.native_phase.__code__.co_code==v1.native_phase.__code__.co_code
assert v2.limits()==v1.limits() and v2.RESOURCES==v1.RESOURCES
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
'''
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)
        self.assertTrue(m.scope_id(0).endswith('_PART0_V2'))
        self.assertNotEqual(m.output_relative(0), m.output_relative(1))
        with self.assertRaises(ValueError):
            m.scope_id(2)

    def test_exact_utf8_atomic_replace_and_temp_peak(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); counter = m.WrittenBytes(out, 1 << 20)
            first, second = {'中文': '变量'}, {'short': 1}
            counter.write_json(out/'RUN.json', first)
            size1 = len((json.dumps(first, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode())
            self.assertEqual(counter.used, size1)
            counter.write_json(out/'RUN.json', second)
            size2 = (out/'RUN.json').stat().st_size
            self.assertEqual(counter.used, size2)
            self.assertEqual(counter.peak_accounted_with_holds, size1+size2)
            self.assertEqual(counter.binding(out/'RUN.json', out), m.native.binding(out/'RUN.json', out))
            self.assertFalse((out/'RUN.json.tmp').exists())
            counter.limit = counter.used
            before = (out/'RUN.json').read_bytes()
            with self.assertRaises(ValueError):
                counter.write_json(out/'RUN.json', {'too large': 'x'*100})
            self.assertEqual((out/'RUN.json').read_bytes(), before)

    def test_snapshot_actual_bytes_hash_failure_stays_accounted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); out = root/'out'; out.mkdir(); source = root/'source'; source.write_bytes(b'actual bytes')
            counter = m.WrittenBytes(out, 1000)
            item = m.native.binding(source, root); counter.hold('frozen_metadata', 2*item['bytes'])
            counter.copy_bound(source, out/'copy', item)
            self.assertEqual(counter.used, item['bytes'])
            wrong = dict(item, sha256='0'*64)
            with self.assertRaisesRegex(ValueError, 'Snapshot source changed'):
                counter.copy_bound(source, out/'bad-copy', wrong)
            self.assertEqual(counter.used, 2*item['bytes'])
            self.assertEqual(counter.holds['frozen_metadata'], 0)
            self.assertEqual(counter.files[out/'bad-copy']['sha256'], item['sha256'])

    def test_journal_stream_and_all_hot_operations_without_tree_walk(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); counter = m.WrittenBytes(out, 2 << 20)
            with mock.patch.object(Path, 'rglob', side_effect=AssertionError('tree walk forbidden')), \
                 mock.patch.object(Path, 'glob', side_effect=AssertionError('tree walk forbidden')), \
                 mock.patch.object(Path, 'iterdir', side_effect=AssertionError('tree walk forbidden')):
                journal = counter.open_journal(out/'journal.jsonl')
                journal.write('{"type":"预扣"}\n'); journal.write('{"type":"实际返回"}\n'); journal.close()
                counter.write_json(out/'RUN.json', {'status': 'tiny'})
                counter.reserve_payload(17)
                (out/'tensor.safetensors').write_bytes(b'x'*17)
                counter.observe_external(out/'tensor.safetensors')
                counter.check()
            self.assertEqual(counter.used, sum(p.stat().st_size for p in out.iterdir()))
            self.assertEqual(counter.files[out/'journal.jsonl']['sha256'], hashlib.sha256((out/'journal.jsonl').read_bytes()).hexdigest())
            tree = ast.parse((ROOT/'tools/e0_original_bf16_full_depth_collect_v2.py').read_text())
            self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in ('rglob', 'glob', 'walk', 'iterdir') for n in ast.walk(tree)))

    def test_monitor_is_read_once_per_payload_before_write_reservation(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = []
            def check():
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError('foreign GPU already sampled')
            out = Path(folder); counter = m.WrittenBytes(out, 3 << 20, check)
            counter.reserve_payload(8); (out/'a').write_bytes(b'12345678'); counter.observe_external(out/'a')
            with self.assertRaisesRegex(RuntimeError, 'foreign GPU'):
                counter.reserve_payload(8)
            self.assertEqual(len(calls), 2)
            self.assertNotIn('safetensors', counter.holds)
            self.assertEqual(counter.used, 8)

    def test_original_serializers_once_identity_and_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); counter = m.WrittenBytes(out, 4 << 20)
            fake_parent, fake_tensor = types.ModuleType('safetensors'), types.ModuleType('safetensors.torch')
            fake_video = types.ModuleType('run_ub1_consumer_videos'); calls = []; value = object()
            def save(tensors, filename):
                self.assertIs(tensors, value); calls.append('save'); Path(filename).write_bytes(b'one original write')
                return 'same return'
            def inspect(path, contract):
                self.assertIs(contract, value); calls.append('inspect')
                return {'path': str(path), 'bytes': Path(path).stat().st_size, 'sha256': m.native.sha(path)}
            fake_tensor.save_file = save; fake_parent.torch = fake_tensor; fake_video.inspect_video = inspect
            before = m.native.write, m.native.binding, m.legacy.reserve_output_bytes
            modules = {'safetensors': fake_parent, 'safetensors.torch': fake_tensor, 'run_ub1_consumer_videos': fake_video}
            with mock.patch.dict(sys.modules, modules):
                with m.counted_io(counter):
                    counter.reserve_payload(32)
                    self.assertEqual(fake_tensor.save_file(value, out/'tensor'), 'same return')
                    video = out/'output.mp4'; counter.hold(str(video), 128); video.write_bytes(b'video bytes')
                    result = fake_video.inspect_video(video, value)
                    self.assertEqual(counter.binding(video, out)['sha256'], result['sha256'])
                    m.native.write(out/'RUN.json', {'actual': True})
                self.assertIs(fake_tensor.save_file, save); self.assertIs(fake_video.inspect_video, inspect)
            self.assertEqual(before, (m.native.write, m.native.binding, m.legacy.reserve_output_bytes))
            self.assertEqual(calls, ['save', 'inspect'])

    def test_partial_serializer_failure_retains_actual_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); counter = m.WrittenBytes(out, 4 << 20)
            fake_parent, fake_tensor = types.ModuleType('safetensors'), types.ModuleType('safetensors.torch')
            fake_video = types.ModuleType('run_ub1_consumer_videos')
            def fail(_tensors, filename):
                Path(filename).write_bytes(b'partial'); raise OSError('original writer failed')
            fake_tensor.save_file = fail; fake_parent.torch = fake_tensor; fake_video.inspect_video = lambda *_: None
            with mock.patch.dict(sys.modules, {'safetensors': fake_parent, 'safetensors.torch': fake_tensor, 'run_ub1_consumer_videos': fake_video}):
                with m.counted_io(counter):
                    counter.reserve_payload(16)
                    with self.assertRaisesRegex(OSError, 'original writer failed'):
                        fake_tensor.save_file({}, out/'partial')
            self.assertEqual(counter.used, 7)
            self.assertFalse(counter.holds)
            self.assertEqual(counter.files[out/'partial']['sha256'], hashlib.sha256(b'partial').hexdigest())

    def test_external_overflow_keeps_actual_failed_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); counter = m.WrittenBytes(out, 1000)
            counter.hold('small_video', 2); (out/'video').write_bytes(b'actual')
            with self.assertRaisesRegex(ValueError, 'actual bytes retained'):
                counter.observe_external(out/'video', hold='small_video')
            self.assertEqual(counter.used, 6)
            self.assertIn(out/'video', counter.files)

    def test_inherited_per_file_bound_in_real_subprocess(self):
        code = '''import os, pathlib, resource, subprocess, sys, tempfile
sys.path.insert(0, 'tools')
import e0_original_bf16_full_depth_collect_v2 as m
m.REPAIR=dict(m.REPAIR, maximum_single_video_file_bytes=4096)
before=resource.getrlimit(resource.RLIMIT_FSIZE)
with tempfile.TemporaryDirectory() as td:
    p=pathlib.Path(td)/'partial'
    with m.single_file_bound():
        child="import pathlib,sys; f=pathlib.Path(sys.argv[1]).open('wb',buffering=0); f.write(b'x'*8192); f.write(b'x')"
        result=subprocess.run([sys.executable,'-c',child,str(p)],capture_output=True)
        assert result.returncode!=0
        assert p.stat().st_size==4096
        assert resource.getrlimit(resource.RLIMIT_FSIZE)[0]==4096
        c=m.WrittenBytes(pathlib.Path(td),16384); c.hold(str(p),4096); c.observe_external(p,hold=str(p))
        c.write_json(pathlib.Path(td)/'RUN.json',{'status':'FAILED_PARTIAL_PRESERVED'})
        assert c.used==4096+(pathlib.Path(td)/'RUN.json').stat().st_size
assert resource.getrlimit(resource.RLIMIT_FSIZE)==before
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
'''
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)

    def test_current_worker_proof_stat_and_actual_exit_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); models = []
            for kind in ('high', 'low', 'T5', 'VAE'):
                path = root/kind; path.write_bytes(kind.encode())
                item = dict(m.native.binding(path, root), path=str(path), kind=kind)
                if kind in ('high', 'low'): item['header_sha256'] = kind*16 if kind == 'high' else 'a'*64
                models.append(item)
            config_sha = 'a'*64
            protocol = {'status': 'FROZEN_REGISTERED', 'private_worker_config_sha256': config_sha, 'models': models}
            m.native.write(root/'protocol.json', protocol); protocol_binding = m.native.binding(root/'protocol.json', root)
            total = sum(i['bytes'] for i in models)
            run = {'id': m.mirror.ID, 'protocol_sha256': protocol_binding['sha256'],
                'status': 'PASS_FRESH_WORKER_RAW_SHA_STAT_HEADER_NO_MODEL_NO_NUMPY_NO_TORCH',
                'parent_protocol_sha256': m.legacy.PARENT_SHA, 'models_sha256': m.native.canonical(models),
                'private_worker_config_sha256': config_sha, 'predebited_full_payload_bytes': total,
                'actual_full_payload_read_bytes': total, 'all_CUDA_devices_hidden': True,
                'NumPy_or_Torch_imported': False, 'model_calls': 0, 'GPU_calls': 0, 'automatic_retry': False, 'wall_seconds': 1,
                'model_payloads': [dict(i, actual_read_bytes=i['bytes'], predebited_bytes=i['bytes'], status='PASS_FRESH_FULL_SHA', stat=m.native.stat_record(i['path'])) for i in models],
                'native_headers': {i['kind']: {'header_sha256': i['header_sha256']} for i in models if i['kind'] in ('high', 'low')}}
            m.native.write(root/'RUN.json', run); run_binding = m.native.binding(root/'RUN.json', root)
            exit_record = {'id': m.mirror.ID, 'protocol_sha256': protocol_binding['sha256'], 'run_sha256': run_binding['sha256'],
                'process_exited': True, 'child_returncode': 0, 'timeout': False, 'automatic_retry': False}
            m.native.write(root/'EXIT.json', exit_record)
            proof = {'protocol': protocol_binding, 'RUN': run_binding, 'launch_exit': m.native.binding(root/'EXIT.json', root)}
            plan = {'execution_raw_master_CPU_proof': proof, 'execution_private_worker_config_sha256': config_sha, 'models': models}
            with mock.patch.object(m, 'ROOT', root), mock.patch.object(m.mirror, 'validate'), \
                 mock.patch.object(m.mirror, 'TOTAL', total), mock.patch.object(m, 'execution_worker_sha', return_value=config_sha):
                self.assertEqual(m.require_execution_master(plan, root), proof)
                bad_exit = dict(exit_record, child_returncode=1)
                m.native.write(root/'BAD_EXIT.json', bad_exit)
                changed = copy.deepcopy(plan)
                changed['execution_raw_master_CPU_proof']['launch_exit'] = m.native.binding(root/'BAD_EXIT.json', root)
                with self.assertRaisesRegex(ValueError, 'Actual independent exit0'):
                    m.require_execution_master(changed, root)
                Path(models[0]['path']).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'changed after fresh'):
                    m.require_execution_master(plan, root)


if __name__ == '__main__':
    unittest.main()
