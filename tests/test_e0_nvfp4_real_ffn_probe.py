"""Small stdlib controls; no real checkpoint, model or GPU."""
import hashlib
import io
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import e0_nvfp4_real_ffn_probe as driver
import nvfp4_real_ffn_assets as assets


class ProbeInfrastructure(unittest.TestCase):
    def test_predebit_multiunit_and_failure(self):
        report={};events=[];ledger=driver.Ledger(report,events.append)
        def fail():
            ledger.units('dynamic_NVFP4_quant',2,lambda: 1)
            ledger.invoke('packed_NVFP4_GEMM',lambda: (_ for _ in ()).throw(RuntimeError('failed')))
        with self.assertRaises(RuntimeError): ledger.work({'dynamic_NVFP4_quant':2,'packed_NVFP4_GEMM':1},fail)
        self.assertEqual(report['predebited'],{'dynamic_NVFP4_quant':2,'packed_NVFP4_GEMM':1})
        self.assertEqual(report['invoked'],{'dynamic_NVFP4_quant':2,'packed_NVFP4_GEMM':1})
        self.assertEqual(report['returned'],{'dynamic_NVFP4_quant':2})
        self.assertEqual(events[0]['kind'],'reserved_before_call')
        with self.assertRaises(ValueError): ledger.invoke('dynamic_NVFP4_quant',lambda:None)

    def test_schedule_counts_independent(self):
        # Four expert/M cells; six timed modes, 2 warmups and 5 timed calls.
        calls=4*(2+7*(1+1+2)); gemm=4*(2+7*(1+1+2))
        dense=2*(2+1)+4*7*2; gelu=4+2+4*7*2
        self.assertEqual((calls,gemm,dense,gelu,4*6*5),(120,120,62,62,120))
        for key,n in [('dynamic_NVFP4_quant',calls),('packed_NVFP4_GEMM',gemm),('decoded_BF16_GEMM',dense),('tanh_GELU',gelu),('CUDA_timed_samples',120)]:
            self.assertEqual(driver.LIMITS[key],n)

    def test_overlay_regular_hash_links_duplicates_and_paths(self):
        names=['lightx2v_kernel/'+x for x in assets.NAMES]
        payload={name:(b'\x7fELFsample' if name.endswith('.so') else name.encode()) for name in names}
        expected={n:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for n,v in payload.items()}
        def archive(members):
            buf=io.BytesIO()
            with tarfile.open(fileobj=buf,mode='w') as tar:
                for name,kind,data in members:
                    info=tarfile.TarInfo(name)
                    if kind=='link': info.type=tarfile.SYMTYPE;info.linkname='elsewhere';tar.addfile(info)
                    else: info.size=len(data);tar.addfile(info,io.BytesIO(data))
            return buf.getvalue()
        normal=[(n,'file',payload[n]) for n in names]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.assertEqual(len(assets.extract_package(archive(normal),expected,root/'ok')),6)
            bads=[normal[:-1]+[normal[0]],normal[:-1]+[('../escape','file',payload[names[-1]])],
                  normal[:-1]+[(names[-1],'link',b'')],normal[:-1]+[(names[-1],'file',b'\x7fELFwrong!')]]
            for i,bad in enumerate(bads):
                with self.assertRaises(ValueError): assets.extract_package(archive(bad),expected,root/str(i))

    def test_exit_zero_requires_integer_and_stat_tracks_identity(self):
        run={'sha256':'run'};good={'id':'id','protocol_sha256':'p','run_sha256':'run','process_exited':True,'child_returncode':0,'timeout':False,'automatic_retry':False}
        assets.exited(good,run,'p','id')
        with self.assertRaises(ValueError): assets.exited(dict(good,child_returncode=False),run,'p','id')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'file';p.write_bytes(b'a');before=assets.stat_record(p);p.unlink();p.write_bytes(b'b')
            self.assertNotEqual(assets.stat_record(p),before)

    def test_stdlib_import_no_numeric_modules(self):
        code="import sys;sys.path.insert(0,'tools');import e0_nvfp4_real_ffn_probe;assert not {'torch','numpy'} & set(sys.modules)"
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


if __name__=='__main__': unittest.main()
