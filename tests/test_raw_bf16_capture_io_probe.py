"""Small synthetic serializer and failure controls; no real probe or model."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('io_probe',ROOT/'tools/raw_bf16_capture_io_probe.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class TinyControls(unittest.TestCase):
    def test_import_without_numerics(self):
        subprocess.run([sys.executable,'-c',"import sys;sys.path.insert(0,'tools');import raw_bf16_capture_io_probe;assert not {'torch','numpy'}&set(sys.modules)"],cwd=ROOT,check=True)

    def test_tiny_actual_rust_serializer_and_logs(self):
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'hf';local=Path(root)/'local';out.mkdir();local.mkdir()
            with mock.patch.object(m,'SHAPES',{'x':('BF16',[3,4]),'rows':('I64',[3])}):
                result=m.run_probe({'safetensors_version':'0.8.0'},out,local,'f'*64)
            self.assertEqual(result['status'],'COMPLETE_SYNTHETIC_IO_ONLY')
            self.assertEqual(result['files']['direct'],result['files']['copied'])
            self.assertEqual(result['files']['buffered_log'],result['files']['unbuffered_log'])
            self.assertEqual((out/'buffered.log').read_bytes(),bytes(12800))

    def test_failure_record_predebit(self):
        import safetensors
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'hf';local=Path(root)/'local';out.mkdir();local.mkdir()
            with mock.patch.object(m,'SHAPES',{'x':('BF16',[1])}),mock.patch.object(safetensors,'serialize_file',side_effect=OSError('synthetic failure')):
                with self.assertRaises(OSError):m.run_probe({'safetensors_version':'0.8.0'},out,local,'f'*64)
            saved=json.loads((out/'RUN.json').read_text())
            self.assertEqual(saved['status'],'FAILED_SYNTHETIC_IO_PRESERVED_NO_RETRY')
            self.assertEqual(saved['operations']['direct_serialization'],'PREDEBITED')
            self.assertFalse(saved['automatic_retry'])


if __name__=='__main__':unittest.main()
