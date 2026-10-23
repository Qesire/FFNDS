"""Minimal structural and algebraic PRD regressions (illustrative, not model results)."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_representation import RepresentationLedgerError, validate_ledger


class LedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = json.loads(
            (ROOT / "examples" / "representation_ledger.json").read_text(encoding="utf-8")
        )

    def test_valid_illustration(self):
        validate_ledger(self.example)

    def test_duplicate_ids_rejected(self):
        data = copy.deepcopy(self.example)
        data["representations"][1]["id"] = data["representations"][0]["id"]
        with self.assertRaises(RepresentationLedgerError):
            validate_ledger(data)

    def test_missing_connection_rejected(self):
        data = copy.deepcopy(self.example)
        data["connections"][0]["to"] = "unknown"
        with self.assertRaises(RepresentationLedgerError):
            validate_ledger(data)

    def test_design_extension_not_exact(self):
        data = copy.deepcopy(self.example)
        data["connections"][1]["relation"] = "EXACT_EQUIVALENCE"
        with self.assertRaises(RepresentationLedgerError):
            validate_ledger(data)

    def test_false_observation_rejected(self):
        data = copy.deepcopy(self.example)
        data["candidates"][0]["status"] = "OBSERVED"
        with self.assertRaises(RepresentationLedgerError):
            validate_ledger(data)


class AttentionMathTests(unittest.TestCase):
    def test_softmax_dynamic_mlp_rewrite(self):
        # x:[2], X:[3,2], Wq/Wk:[2,2], Wv:[2,2], Wo:[2,2]
        x = [0.3, -0.7]
        X = [[0.1, 0.2], [1.0, -1.0], x]
        Wq = [[0.5, 0.1], [0.2, 0.9]]
        Wk = [[0.7, 0.3], [-0.4, 0.2]]
        Wv = [[1.2, 0.4], [-0.2, 0.6]]
        Wo = [[0.6, 0.1], [0.3, 1.1]]

        def row_mat(row, mat):
            return [sum(a * mat[i][j] for i, a in enumerate(row)) for j in range(len(mat[0]))]

        def dot(a, b):
            return sum(u * v for u, v in zip(a, b))

        def weights(scores):
            es = [math.exp(s - max(scores)) for s in scores]
            return [v / sum(es) for v in es]

        q = row_mat(x, Wq)
        keys = [row_mat(z, Wk) for z in X]
        vals = [row_mat(z, Wv) for z in X]
        p = weights([dot(q, k) / math.sqrt(2.0) for k in keys])
        direct = row_mat([sum(p[i] * vals[i][j] for i in range(len(X))) for j in range(2)], Wo)

        # columns of A = Wq k_i / sqrt(dk); rows of B = (X Wv Wo)_i.
        # Equivalent and clearer: each first-layer score is x(Wq k_i) / sqrt(dk).
        scores = [
            dot(x, [sum(Wq[a][b] * key[b] for b in range(2)) for a in range(2)]) / math.sqrt(2.0)
            for key in keys
        ]
        pp = weights(scores)
        B = [row_mat(v, Wo) for v in vals]
        rewrite = [sum(pp[i] * B[i][j] for i in range(len(X))) for j in range(2)]
        for a, b in zip(direct, rewrite):
            self.assertAlmostEqual(a, b, places=12)

    def test_normalizer_is_query_dependent(self):
        K = [[1.0, 0.0], [0.0, 1.0]]
        def z(q):
            return sum(math.exp(sum(a * b for a, b in zip(q, k))) for k in K)
        self.assertNotAlmostEqual(z([1.0, 0.0]), z([1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
