"""A shared filesystem race must launch only one model child."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from claim_c2_reserved_attempt import claim_once


class SingleClaim(unittest.TestCase):
    def test_two_processes_one_permanent_winner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for slot in ('executor', 'original'):
                (root / ('runs/c2_reserved_wait_' + slot)).mkdir(parents=True)
            code = ('import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);'
                    'from claim_c2_reserved_attempt import claim_once;'
                    'print(int(claim_once(Path(sys.argv[2]),"runs/c2_reserved_wait_"+sys.argv[3],sys.argv[3],"frozen")))')
            def run(slot):
                return subprocess.check_output([sys.executable, '-c', code, str(ROOT / 'tools'), str(root), slot], text=True).strip()
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                results = list(pool.map(run, ('executor', 'original')))
            self.assertEqual(sorted(results), ['0', '1'])
            claim = json.loads((root / 'runs/C2N1_SINGLE_MODEL_CLAIM/CLAIM.json').read_text())
            winner = claim['worker_slot']
            self.assertEqual((root / 'runs/c2_reserved_execution').resolve(), root / ('runs/c2_reserved_wait_' + winner))
            self.assertFalse(claim_once(root, 'runs/c2_reserved_wait_' + winner, winner, 'frozen'))
            self.assertEqual(claim['new_model_calls_at_claim'], 0)

    def test_unexpected_folder_rejected_without_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(RuntimeError):
                claim_once(root, 'runs/another_trial', 'original', 'frozen')
            self.assertFalse((root / 'runs/C2N1_SINGLE_MODEL_CLAIM').exists())


if __name__ == '__main__':
    unittest.main()
