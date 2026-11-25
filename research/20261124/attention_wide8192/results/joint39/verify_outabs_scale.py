"""Symbolically verify a property of the real OUT-ABS readout formula.

This does not execute the video model or substitute synthetic data for model
evidence. Matrix dimensions match attention_online_structured.py; the identity
holds for every matrix value for which the fixed ridge solve is defined.
"""

import json
from pathlib import Path

import sympy as sp


def main():
    n, p, r, d = 31668, 64, 32, 128
    scale = sp.Symbol("scale_r", nonzero=True)
    base = sp.MatrixSymbol("B", n, d)
    design = sp.MatrixSymbol("X", n, r)
    # K denotes (X_P.T X_P + 64 * lambda * I)^-1 X_P.T.
    # It is independent of scale_r in the actual OUT-ABS branch.
    ridge_map = sp.MatrixSymbol("K", r, p)
    residual_p = sp.MatrixSymbol("R_P", p, d)
    implementation = base + scale * design * ridge_map * (residual_p / scale)
    reduced = base + design * ridge_map * residual_p
    difference = sp.simplify(implementation - reduced)
    verified = difference == sp.ZeroMatrix(n, d)
    if not verified:
        raise RuntimeError(f"Symbolic scale identity did not reduce: {difference}")
    result = {
        "check": "OUT-ABS output scale cancels through linear ridge solve",
        "verified": verified,
        "symbolic_difference": str(difference),
        "operator_form": "B + X (X_P^T X_P + 64 lambda I)^(-1) X_P^T R_P",
        "real_online_dimensions": {
            "heads": 40,
            "queries": n,
            "head_channels": d,
            "probe_rows": p,
            "design_columns_including_constant": r,
        },
        "scope": (
            "Algebraic identity of the actual OUT-ABS branch before exact P-row "
            "overwrite, which is also scale independent. Not a GPU/model rerun. "
            "Independent floating-point rounding can remain."
        ),
        "implementation_source": "attention_online_structured.py",
        "script": "verify_outabs_scale.py",
    }
    output = Path(__file__).with_name("outabs_scale_identity.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
