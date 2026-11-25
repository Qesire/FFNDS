"""Symbolic finite recovery and cache-error identities at actual attention shapes."""
from pathlib import Path
import json
import sympy as s

N, D, Pn = 31668, 128, 64
A = s.MatrixSymbol('A_current', N, D)
G = s.MatrixSymbol('G_current', N, D)
Ea = s.MatrixSymbol('E_anchor', N, D)
eps = s.MatrixSymbol('cache_error', N, D)
P = s.MatrixSymbol('P_row_selection', Pn, N)
K = s.MatrixSymbol('K_current_readout', N, Pn)
I = s.Identity(N)
H = I - P.T * P
delta = A - G - Ea
base = G + Ea + eps
raw = base + K * P * (A - base)
final = H * raw + P.T * P * A
expected = H * (K * P - I) * (delta - eps)
identity = s.simplify(s.expand(final - A - expected)) == s.ZeroMatrix(N, D)

base0 = G + Ea
raw0 = base0 + K * P * (A - base0)
final0 = H * raw0 + P.T * P * A
cache_identity = s.simplify(s.expand(final-final0-H*(I-K*P)*eps)) == s.ZeroMatrix(N, D)
assert identity and cache_identity

result = {
    'evidence_commit': '873e1a879b91d544fdae11894d68bbc6acc854d1',
    'scope': 'Exact symbolic matrix identities; no model activations, inference, speed or quality measurement',
    'shape': {'queries': N, 'head_channels': D, 'probes': Pn},
    'assumptions': [
        'P selects 64 distinct query rows, so P.T P is the probe mask',
        'K is fixed for the current inputs/features and does not depend on additive cache error',
        'Teacher is the full current response on P',
        'Form labels before overwriting predictions at P',
        'Identities use exact real arithmetic; quantization of features and operations needs its own terms'
    ],
    'post_probe_overwrite_finite_innovation_error_identity': identity,
    'post_probe_overwrite_additive_cache_error_identity': cache_identity,
    'full_block_channel_mean_quotient': {
        'channels': 5120, 'null_directions_per_position': 1,
        'fraction_of_channels': 1/5120,
        'condition': 'All subsequent residual branches depend on their input only through channel LayerNorm, time-only gates, and a final channel LayerNorm head; exact real arithmetic'
    }
}
path = Path(__file__).with_name('recovery_operator_verified.json')
path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
