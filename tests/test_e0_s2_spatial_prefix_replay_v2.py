import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from tools import e0_s2_spatial_prefix_replay as original
from tools import e0_s2_spatial_prefix_replay_v2 as m


class LaunchRepairControls(unittest.TestCase):
    def test_missing_cpu_parent_is_created_before_original_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            hf = Path(directory); out = hf/'new'/'worker'/'CPU'/'RUN.json'
            args = SimpleNamespace(hf_root=hf, out=out)
            m.prepare_parent(args)
            m.native.write(out, {'status': 'SOFTWARE_ONLY'})
            self.assertEqual(m.native.read(out), {'status': 'SOFTWARE_ONLY'})
            with self.assertRaises(ValueError): m.prepare_parent(args)
            with self.assertRaises(ValueError): m.prepare_parent(SimpleNamespace(hf_root=hf/'other', out=hf/'outside'))

    def test_identity_is_versioned_and_native_algorithm_is_unchanged(self):
        self.assertNotEqual(m.core.BASE, original.BASE)
        self.assertEqual(m.core.DESIGN, original.DESIGN)
        self.assertEqual(m.core.RESOURCES, original.RESOURCES)
        self.assertEqual(m.core.limits(), original.limits())
        self.assertIs(m.core.geometry, original.geometry)
        for i in (0, 1): self.assertNotEqual(m.output(i), original.output(i))


if __name__ == '__main__': unittest.main()
