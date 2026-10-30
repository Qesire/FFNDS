import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from a5_hwg_resource_readahead_20261030 import advise_file


class ResourceAdviceTests(unittest.TestCase):
    def exercise(self, decisions):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'asset'
            p.write_bytes(b'01234567890')
            calls, charged, returned = [], [], []
            choices = iter(decisions)
            with patch('os.posix_fadvise', side_effect=lambda fd, off, length, kind: calls.append((off, length))), \
                 patch('os.read', side_effect=AssertionError('must not return payload values')):
                state = advise_file(p, 11, 4, lambda: next(choices),
                                    lambda: charged.append(1), returned.append)
            return state, calls, len(charged), sum(returned)

    def test_parent_exit_midfile_stops_before_next_hint(self):
        self.assertEqual(self.exercise(['CONTINUE', 'PARENT_CLOSED']),
                         ('PARENT_CLOSED', [(0, 4)], 1, 4))

    def test_memory_drop_midfile_pauses_before_next_hint(self):
        self.assertEqual(self.exercise(['CONTINUE', 'LOW_MEMORY']),
                         ('LOW_MEMORY', [(0, 4)], 1, 4))

    def test_exact_finite_ranges_without_payload_read(self):
        self.assertEqual(self.exercise(['CONTINUE'] * 3),
                         ('COMPLETE', [(0, 4), (4, 4), (8, 3)], 3, 11))

    def test_opened_inode_mismatch_rejected_before_first_hint(self):
        with tempfile.TemporaryDirectory() as directory:
            p, other = Path(directory) / 'asset', Path(directory) / 'other'
            p.write_bytes(b'01234567890')
            other.write_bytes(b'abcdefghijk')
            real_open = os.open
            with patch('os.open', side_effect=lambda path, flags: real_open(other, flags)), \
                 patch('os.posix_fadvise') as hint:
                with self.assertRaisesRegex(RuntimeError, 'opened identity differs'):
                    advise_file(p, 11, 4, lambda: 'CONTINUE', lambda: None, lambda n: None)
                hint.assert_not_called()


if __name__ == '__main__': unittest.main()
