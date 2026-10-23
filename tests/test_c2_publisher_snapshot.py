"""Publication must defer when prepared or unrelated decisions have changed."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from publish_c2_finished_attempt import DYNAMIC, ROOT, replace_c2_section, sha, verify_concurrent


class C2PublisherSnapshot(unittest.TestCase):
    def snapshot(self):
        return {'expected_head': 'synthetic_expected_head', 'static_sha256': {
            'tools/norm_scaled_row_dispatch.py': sha(ROOT / 'tools/norm_scaled_row_dispatch.py')},
            'dynamic': {p: (ROOT / p).read_text() for p in DYNAMIC}}

    def verify(self, snapshot):
        git = lambda *args: 'main' if args[0] == 'branch' else 'synthetic_expected_head'
        with patch('publish_c2_finished_attempt.git', side_effect=git), \
             patch('publish_c2_finished_attempt.subprocess.run', return_value=SimpleNamespace(returncode=0)):
            verify_concurrent(snapshot)

    def test_unchanged_snapshot_and_native_closure_fields_are_supported(self):
        self.verify(self.snapshot())

    def test_unrelated_closed_method_edit_defers_publication(self):
        snapshot = self.snapshot(); key = 'research/CURRENT_DECISION.json'
        d = json.loads(snapshot['dynamic'][key]); method = next(iter(d['closed_method_statuses']))
        d['closed_method_statuses'][method] = 'synthetic_concurrent_edit'
        snapshot['dynamic'][key] = json.dumps(d)
        with self.assertRaisesRegex(RuntimeError, 'concurrent decision edit'):
            self.verify(snapshot)

    def test_concurrent_documentation_and_static_edits_defer(self):
        snapshot = self.snapshot(); snapshot['dynamic']['README.md'] += '\nSynthetic differing snapshot\n'
        with self.assertRaisesRegex(RuntimeError, 'documentation change'): self.verify(snapshot)
        snapshot = self.snapshot(); snapshot['static_sha256']['tools/norm_scaled_row_dispatch.py'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'prepared file changed'): self.verify(snapshot)

    def test_result_render_preserves_new_research_and_historical_sections(self):
        for start, end in [('# C2N0 ', '# C1N0 native diagnostic completed'),
                           ('## C2N0：', '## C1N0：')]:
            with self.subTest(start=start):
                prefix = '# Independent PRD successor\n\nKeep residual-value science.\n\n'
                history = end + '\n\nKeep completed C1 evidence.\n'
                old = prefix + start + 'pending\n\nOld C2 status.\n\n' + history
                result = replace_c2_section(old, start, end, start + 'finished\n\nActual C2 receipt.')
                self.assertTrue(result.startswith(prefix))
                self.assertTrue(result.endswith(history))
                self.assertEqual(result.count(start), 1)
                self.assertNotIn('Old C2 status.', result)

    def test_result_render_refuses_changed_document_boundaries(self):
        with self.assertRaisesRegex(RuntimeError, 'boundaries moved'):
            replace_c2_section('# Different research layout\n', '# C2N0 ', '# C1N0 ', 'new')
        with self.assertRaisesRegex(RuntimeError, 'order changed'):
            replace_c2_section('# C1N0 earlier\n# C2N0 later\n', '# C2N0 ', '# C1N0 ', 'new')


if __name__ == '__main__': unittest.main()
