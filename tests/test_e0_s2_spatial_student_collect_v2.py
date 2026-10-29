import unittest
from tools import e0_s2_spatial_student_collect as original
from tools import e0_s2_spatial_student_collect_v2 as m


class PortableCaptureControls(unittest.TestCase):
    def test_executor_namespace_changes_leave_original_scope_and_numeric_budget(self):
        self.assertNotEqual(m.core.BASE_ID, original.BASE_ID)
        for part in (0, 1):
            self.assertNotEqual(m.output_relative(part), original.output_relative(part))
            self.assertEqual(m.core.limits(part), original.limits(part))
        self.assertEqual(m.core.DESIGN, original.DESIGN)
        self.assertEqual(m.core.RESOURCES, original.RESOURCES)
        self.assertIs(m.core.io.mirror.validate, m.proof.validate)
        self.assertEqual(m.core.io.mirror.ID, m.proof.ID)
        self.assertIsNot(m.core.io, original.io)
        self.assertNotEqual(original.io.mirror.ID, m.proof.ID)


if __name__ == '__main__': unittest.main()
