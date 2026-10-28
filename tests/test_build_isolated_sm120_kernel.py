import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
from tools.build_isolated_sm120_kernel import commands,run_process

class CommandsTest(unittest.TestCase):
    def test_isolated_compile_commands(self):
        c={'cmake':'cmake','ninja':'/env/bin/ninja','gxx':'/usr/bin/g++','nvcc':'/cuda/bin/nvcc','cuda_root':'/cuda','python':'/env/bin/python','torch_cmake':'/env/torch/share/cmake','cutlass':'/cutlass'}
        a=commands(c,Path('/tmp/src'),Path('/tmp/build'),Path('/tmp/overlay'))
        self.assertIn('-DCMAKE_MAKE_PROGRAM=/env/bin/ninja',a[0]);self.assertIn('-DCMAKE_CXX_COMPILER=/usr/bin/g++',a[0]);self.assertIn('-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++',a[0]);self.assertIn('-DCMAKE_CUDA_ARCHITECTURES=OFF',a[0]);self.assertIn('-DSKBUILD_SABI_VERSION=3.11',a[0]);self.assertIn('-DSKBUILD_SABI_COMPONENT=Development.SABIModule',a[0]);self.assertEqual(a[1][-2:],['--parallel','2']);self.assertIn('-DCMAKE_INSTALL_PREFIX=/tmp/overlay',a[0])

    def test_completed_failure_keeps_exit_and_log(self):
        class Done:
            pid=123
            returncode=7
            def poll(self):return 7
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); record={}
            with patch('tools.build_isolated_sm120_kernel.subprocess.Popen',return_value=Done()), patch('tools.build_isolated_sm120_kernel.os.killpg') as kill:
                with self.assertRaisesRegex(ValueError,'build command failed'):
                    run_process(['bad'],{},root/'log',999999999,root,record)
                kill.assert_not_called()
            self.assertEqual(record['returncode'],7);self.assertEqual(record['log']['bytes'],0)

if __name__=='__main__':unittest.main()
