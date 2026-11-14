"""V0 PANEL512 producer: uniform random-order 512 THW token rows, one per source x seed.

Contract (config/g20_vbench_restart_cohort_v1_20261113.json, sampling; route
G20_VBENCH_RESTART_V4_20261113.md section 3):
  One frozen uniformly sampled random-order 512-row sequence per source x
  generation seed, shared across all 4 steps / all 40 layers / all compared arms.
  Prefixes (256, 512) are taken in DRAW order before any sorting; 256 is the
  draw-order prefix of the 512 draw, NEVER an independent choice() call. A sorted
  read must be mappable back to draw order, so the sort permutation AND its
  inverse are preserved (preserve_sort_inverse).

  NO tubelet and NO full-M hidden are required by default
  (extra_tubelets_or_full_m_hidden_required_by_default = false). This producer
  emits ONLY the uniform 512 panel; the legacy 336-row tubelet of panel_v3 and
  the E1 latent_time_stratified_v1 frame-stratified 2048 mask are DIFFERENT
  estimands and are not reproduced here.

This module generates coordinates only. It holds no model state, performs no
response-dependent selection, and does not import torch. The draw key contains
no step/block, so the panel is identical at every capture site of a trajectory;
consumers must verify token_id_sha256_le_int64 across all 160 sites (4 steps x
40 layers). The VERSION namespace string separates this stream from panel_v3
(uniform_random_order_v1) and from the E1 stratified mask.

Source identifier convention (documented for the reviewer): the V0 runner keys
panels by record_id (e.g. 'vbi_1005'), the stable cohort primary key, rather than
image_name, to avoid filename whitespace/case/Unicode ambiguity in the JSON hash
key. record_id is 1:1 with the selected image under the verified cluster
crosswalk. make_panel itself accepts any nonempty string source.
"""
import hashlib
import json

import numpy as np

VERSION = 'G20_VBENCH_RESTART_V4_20261113_ROWS_V1'
PANEL_COUNT = 512
PREFIX_SIZES = (256, 512)
# Stream tag inside the key. The VERSION namespace already separates this panel
# from every legacy stream; the tag is kept identical to the proven panel_v3
# uniform stream tag so the derivation is recognisable. There is exactly one
# stream (no tubelet), so no second tag exists.
DRAW_TAG = 'uniform_random_order'


def _rng(parts):
    key = json.dumps(parts, ensure_ascii=False, separators=(',', ':')).encode('utf8')
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:16], 'little'))


def _sha_i64(ids):
    return hashlib.sha256(np.ascontiguousarray(ids, dtype='<i8').tobytes()).hexdigest()


def make_panel(source, seed, grid, count=PANEL_COUNT):
    """Return (draw_order_ids, info).

    draw_order_ids: int64 little-endian ('<i8'), length count, UNIFORM random
    order (simple random sampling without replacement over t*h*w), NOT sorted.
    info carries the geometry receipt, prefix SHAs, the full-panel SHA, the
    sorted-panel SHA, and the preserved sort/inverse permutations.
    """
    if not isinstance(source, str) or not source:
        raise ValueError('source must be a nonempty string')
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('seed must be an int')
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError('count must be an int')
    t, h, w = (int(v) for v in grid)
    n = t * h * w
    if min(t, h, w) <= 0 or not 0 < count <= n:
        raise ValueError('invalid geometry/count')
    # Only the full 512 draw exists; smaller panels are draw-order prefixes of
    # it, never independent choice() calls (which would not be nested).
    if count != PANEL_COUNT:
        raise ValueError('count must be the frozen 512 panel; use prefix() for smaller sizes')

    ids = _rng([VERSION, source, seed, DRAW_TAG]).choice(n, count, replace=False).astype('<i8')

    # preserve_sort_inverse: a consumer that reads the panel in sorted token order
    # can recover the original draw order via the inverse permutation, and vice
    # versa. sort_perm maps draw->sorted; inverse_perm maps sorted->draw.
    sort_perm = np.argsort(ids, kind='stable').astype('<i8')
    sorted_ids = ids[sort_perm]
    inverse_perm = np.argsort(sort_perm, kind='stable').astype('<i8')
    # Round-trip identities (asserted here so a broken permutation can never ship):
    if not np.array_equal(sorted_ids[inverse_perm], ids):
        raise RuntimeError('inverse permutation round-trip failed')
    if not np.array_equal(ids[sort_perm], sorted_ids):
        raise RuntimeError('sort permutation round-trip failed')

    pi = count / n
    info = {
        'version': VERSION,
        'source': source,
        'seed': seed,
        'grid': [t, h, w],
        'population_rows': n,
        'count': count,
        'order': 'uniform_random_draw_order_unsorted',
        'prefix_sizes_in_draw_order': list(PREFIX_SIZES),
        'prefix_token_id_sha256_le_int64': {str(k): _sha_i64(ids[:k]) for k in PREFIX_SIZES},
        'token_id_sha256_le_int64': _sha_i64(ids),
        'sorted_token_id_sha256_le_int64': _sha_i64(sorted_ids),
        'inclusion_probability_marginal': pi,
        'inverse_inclusion_probability_marginal': 1.0 / pi,
        'tubelet': None,
        'full_m_hidden': None,
        'preserve_sort_inverse': {
            'sort_permutation': sort_perm.tolist(),
            'inverse_permutation': inverse_perm.tolist(),
            'sorted_token_id_sha256_le_int64': _sha_i64(sorted_ids),
            'round_trip_rule': 'draw_order_ids = sorted_ids[inverse_permutation]; sorted_ids = draw_order_ids[sort_permutation]',
            'dtype': '<i8',
        },
        'cross_site_rule': 'identical at all 4 steps x 40 layers x arms within source x seed; key has no step/block',
        'sampling_rng': 'numpy.default_rng PCG64; sha256(json([VERSION,source,seed,tag]))[:16] little-endian; numpy version must be frozen by the consumer',
        'draw_key_parts': [VERSION, source, seed, DRAW_TAG],
    }
    return ids, info


def prefix(ids, k):
    """Draw-order prefix; never sort before truncating."""
    if not isinstance(k, int) or isinstance(k, bool) or not 0 < k <= len(ids):
        raise ValueError('invalid prefix size')
    return ids[:k]


def sorted_read(ids):
    """Return (sorted_ids, sort_permutation, inverse_permutation) for a consumer
    that reads rows in ascending token order but must recover draw order."""
    sort_perm = np.argsort(ids, kind='stable').astype('<i8')
    sorted_ids = ids[sort_perm]
    inverse_perm = np.argsort(sort_perm, kind='stable').astype('<i8')
    return sorted_ids, sort_perm, inverse_perm


def reference_vectors():
    """Frozen reference identities emitted for downstream producer gates."""
    out = {}
    for source, seed in (('v0_panel512_reference', 42), ('v0_panel512_reference', 43)):
        ids, info = make_panel(source, seed, (21, 29, 52))
        out[source + '__seed' + str(seed)] = {
            'token_id_sha256_le_int64': info['token_id_sha256_le_int64'],
            'sorted_token_id_sha256_le_int64': info['sorted_token_id_sha256_le_int64'],
            'prefix_256_sha256_le_int64': info['prefix_token_id_sha256_le_int64']['256'],
            'first_8_draw_order': ids[:8].tolist(),
            'inverse_permutation_first_8': info['preserve_sort_inverse']['inverse_permutation'][:8],
        }
    return out