from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ar2_checkpoint import CheckpointStore


class CheckpointIntegrity(unittest.TestCase):
    def test_gain_metadata_only_checkpoint_preserves_na(self):
        with tempfile.TemporaryDirectory() as temp:
            store = CheckpointStore(temp, {"protocol": "synthetic_test"})
            metadata = {"alpha": None, "status": "NA_DENOM_FLOOR", "donor_seed": 20261002}
            store.save("gains", "R", {}, metadata)
            loaded = CheckpointStore(temp, {"protocol": "synthetic_test"}).load("gains", "R")
            self.assertEqual(loaded, {"arrays": {}, "metadata": metadata})

    def test_resume_preserves_finished_parts_but_ignores_interrupted_temp_file(self):
        with tempfile.TemporaryDirectory() as temp:
            identity = {"protocol": "synthetic_test", "rows": 512, "source_sha256": "abc"}
            store = CheckpointStore(temp, identity)
            store.save("scores", "task0", {"score": np.arange(6).reshape(2, 3)}, {"rows": 2})
            parent = Path(temp) / "projections"
            parent.mkdir()
            (parent / "partial.npz.tmp").write_bytes(b"interrupted write")
            resumed = CheckpointStore(temp, identity)
            result = resumed.load("scores", "task0")
            np.testing.assert_array_equal(result["arrays"]["score"], np.arange(6).reshape(2, 3))
            self.assertEqual(result["metadata"], {"rows": 2})
            self.assertIsNone(resumed.load("projections", "partial"))

    def test_changed_protocol_or_assets_cannot_reuse_old_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            CheckpointStore(temp, {"asset_sha256": "old", "group_size": 64})
            for identity in [{"asset_sha256": "new", "group_size": 64},
                             {"asset_sha256": "old", "group_size": 32}]:
                with self.assertRaises(ValueError):
                    CheckpointStore(temp, identity)

    def test_corrupted_payload_is_rejected_instead_of_marked_resumed(self):
        with tempfile.TemporaryDirectory() as temp:
            store = CheckpointStore(temp, {"protocol": "synthetic_test"})
            store.save("scores", "task0", {"score": np.ones((3, 4))}, {})
            (Path(temp) / "scores/task0.npz").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                store.load("scores", "task0")


if __name__ == "__main__":
    unittest.main()
