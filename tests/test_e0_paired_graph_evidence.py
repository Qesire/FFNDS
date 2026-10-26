"""Static evidence-state regression: observed traces and open causes differ."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from check_autoresearcher_binding import paired_graph_evidence_is_consistent


class PairedGraphEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.active = json.loads((ROOT / 'research/ACTIVE_RESEARCH.json').read_text())
        self.graph = json.loads((ROOT / 'research/agent_protocol/GRAPH_AGENT_BINDINGS.json').read_text())
        self.result = json.loads((ROOT / 'research/20261026/E0_PAIRED_FULL_DEPTH_RESULTS.json').read_text())

    def test_completed_two_source_native_record_with_open_cause_is_valid(self):
        self.assertTrue(paired_graph_evidence_is_consistent(self.active, self.graph, self.result))

    def test_missing_source_changed_coverage_or_quality_promotion_is_rejected(self):
        variants = []
        value = deepcopy(self.result); value['runs'].pop(); variants.append(value)
        value = deepcopy(self.result); value['actual_budget']['paired_sites'] = 319; variants.append(value)
        value = deepcopy(self.result); value['quality_accepted'] = True; variants.append(value)
        value = deepcopy(self.result); value['protocol_sha256'] = 'wrong'; variants.append(value)
        for value in variants:
            self.assertFalse(paired_graph_evidence_is_consistent(self.active, self.graph, value))

    def test_observation_does_not_close_causality(self):
        for node in self.graph['nodes']:
            if node['id'] == 'P24':
                node['causal_status'] = 'PROVEN'
        self.assertFalse(paired_graph_evidence_is_consistent(self.active, self.graph, self.result))


if __name__ == '__main__':
    unittest.main()
