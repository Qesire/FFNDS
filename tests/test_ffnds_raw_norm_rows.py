import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
import numpy as np

spec = importlib.util.spec_from_file_location('raw_norm_rows', Path(__file__).resolve().parents[1] / 'tools/ffnds_raw_norm_rows.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RawNormTests(unittest.TestCase):
    def test_bf16_decode_signed(self):
        raw = np.array([0x3f80, 0xc000, 0x0000], dtype='<u2').tobytes()
        np.testing.assert_array_equal(m.decode_bf16(raw, (3,), np), [1., -2., 0.])

    def test_centered_variance_and_actual_remainder(self):
        d = {'u': np.array([[1., -1., 3., -3.]]), 'Y': np.array([[1., 2., 3., 4.]]), 'gate': np.ones(4), 'rows': np.array([0]), 'V1': np.ones((1,4)), 'r': np.ones((1,4)), 'head_scale': np.ones(4), 'head_shift': np.zeros(4)}
        s = {k:v.copy() for k,v in d.items()}
        s['u'] = 2*d['u']; s['Y'] = d['Y'] + 2*d['u']
        tau, o = m.pair(d,s,np)
        self.assertAlmostEqual(tau[0], np.sqrt(5+1e-6)/np.sqrt(20+1e-6))
        self.assertAlmostEqual(o['signed_cross']['mean'], 10.)
        self.assertAlmostEqual(o['delta_centered_square']['mean'], 5.)
        self.assertAlmostEqual(o['actual_write_remainder_sse'], 20.)
        self.assertAlmostEqual(o['write_closure_residual'], 0.)
        s['r'][0,0] = 0
        with self.assertRaises(ValueError): m.pair(d,s,np)

    def test_fold_gain_and_baseline(self):
        train=np.array([2., 8.]); k=np.exp(np.mean(np.log(train)))
        self.assertAlmostEqual(k,4.)
        result=m.evaluate(np.array([4.,4.]),k,np)
        self.assertEqual(result['relative_ratio_mse'],0.)
        self.assertEqual(result['identity_ratio_mse'],9.)

    def test_failure_preserves_predebit(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            ledger=m.Ledger(out, {'limits': {'capture_loads': 1}}, 'synthetic')
            with self.assertRaisesRegex(ValueError, 'synthetic error'):
                with ledger.work('capture_loads'):
                    raise ValueError('synthetic error')
            ledger.fail(ValueError('synthetic error'), 0.1)
            saved=json.loads((out/'RUN.json').read_text())
            self.assertEqual(saved['status'],'FAILED_PRESERVED_NO_RETRY')
            self.assertEqual(saved['error'], 'synthetic error')
            self.assertEqual(saved['predebited']['capture_loads'],1)
            self.assertEqual(saved['observed']['capture_loads'],0)



if __name__ == '__main__': unittest.main()
