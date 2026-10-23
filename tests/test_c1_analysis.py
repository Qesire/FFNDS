"""Known-answer synthetic fixtures for the saved C1 decision analysis."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'NumPy required for CPU analysis fixtures')
class C1Analysis(unittest.TestCase):
    def fixture(self, root):
        from c1_contract import SOURCES, BLOCKS, atomic, sha
        rows = []
        for source in SOURCES:
            for step in range(4):
                for block in BLOCKS:
                    p = root / f'{source}_{step}_{block}.npz'
                    np.savez_compressed(p, h256=np.zeros((336, 256), dtype=np.uint16),
                        pre_features=np.ones((336, 6)), norm_rstd=np.ones(336),
                        nextnorm_UB1=np.full(21, 100.), nextnorm_swap_A=np.full(21, 70.),
                        nextnorm_swap_B=np.full(21, 130.))
                    rows.append({'source': source, 'step': step, 'block': block,
                                 'payload_HF_relative': p.name, 'payload_sha256': sha(p)})
        atomic(root / 'RUN.json', {'status': 'COMPLETE_NATIVE_C1_FIXED_SCOPE', 'finite_scopes': rows})

    def test_known_fixed_winner_and_no_signal_gain(self):
        from analyze_c1_native import analyze
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root)
            result = analyze(root)
            self.assertEqual(result['results']['UB1']['total_nextnorm_SSE'], 100 * 21 * 72)
            for name, r in result['results'].items():
                if name == 'UB1':
                    continue
                self.assertEqual(r['total_nextnorm_SSE'], 70 * 21 * 72)
                self.assertEqual(r['whole_site_action_counts']['swap_A'], 72)
                self.assertEqual(r['total_regret_to_same_action_oracle'], 0)
            self.assertFalse(result['automatic_followup'])
            self.assertFalse(result['official_video_quality_or_Pareto_or_native_policy_speed_result'])

    def test_changed_payload_and_incomplete_run_rejected(self):
        from analyze_c1_native import analyze
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root)
            run = json.loads((root / 'RUN.json').read_text())
            first = root / run['finite_scopes'][0]['payload_HF_relative']
            first.write_bytes(first.read_bytes() + b'fixture_changed')
            with self.assertRaisesRegex(RuntimeError, 'payload changed'):
                analyze(root)
            run['finite_scopes'].pop()
            (root / 'RUN.json').write_text(json.dumps(run))
            with self.assertRaisesRegex(RuntimeError, 'complete C1 native data'):
                analyze(root)


if __name__ == '__main__':
    unittest.main()
