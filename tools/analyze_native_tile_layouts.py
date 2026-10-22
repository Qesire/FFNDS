"""Exact row-weighted budget and copy-layout arithmetic, no model libraries."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_lengths(indices):
    lengths = []
    for i in indices:
        if not lengths or i != previous + 1:
            lengths.append(1)
        else:
            lengths[-1] += 1
        previous = i
    return lengths


def analyze():
    frames, height, width, d = 21, 29, 52, 5120
    m = frames * height * width
    masks = {
        'left_right_spatial_halves': [i for i in range(m) if i % width < 26],
        'per_frame_row_major_halves': [i for i in range(m) if i % (height * width) < height * width // 2],
        'contiguous_sequence_halves': list(range(m // 2))}
    result = []
    for name, selected in masks.items():
        other = sorted(set(range(m)) - set(selected))
        if len(selected) != m // 2 or len(other) != m // 2:
            raise RuntimeError('Spatial pair must have identical real row counts')
        lengths = [run_lengths(v) for v in (selected, other)]
        result.append({'layout': name, 'rows': [len(selected), len(other)],
            'contiguous_segments': [len(v) for v in lengths],
            'segment_row_counts': [sorted(set(v)) for v in lengths],
            'pair_row_weighted_groups': float(Fraction(108 * len(selected) + 216 * len(other), m)),
            'naive_input_gather_and_output_scatter_extra_traffic_bytes': 4 * m * d * 2 if name != 'contiguous_sequence_halves' else 0,
            'traffic_scope': 'Copy-based layout with one complete BF16 input gather and output scatter, each read+write. Contiguous views can avoid these copies with an appropriate output API. This is not a lower bound on fused/strided kernels.'})
    frame_splits = []
    for wide in (10, 11):
        avg = Fraction(wide * 216 + (frames - wide) * 108, frames)
        global_density = Fraction(1, 2) + (avg - 162) / (216 * 160)
        frame_splits.append({'wide_frames': wide, 'narrow_frames': frames - wide,
            'widths': [216, 108], 'weighted_groups_fraction': str(avg),
            'global_main_MAC_fraction': float(global_density), 'within_50pct': global_density <= Fraction(1, 2)})
    return {'id': 'R12_NATIVE_TILE_LAYOUT_AND_ROW_BUDGET', 'status': 'EXACT_INTEGER_LAYOUT_AND_BUDGET_ONLY',
        'grid': [frames, height, width], 'full_native_rows': m, 'layouts': result,
        'whole_frame_10_vs11_counterexample': frame_splits,
        'arithmetic_only_different_width_example': {'wide_frames': 14, 'wide_groups': 216,
            'narrow_frames': 7, 'narrow_groups': 54, 'weighted_groups': 162,
            'status': 'Different action, not registered or authorized by D0'},
        'new_model_generations': 0, 'native_tile_identity_or_quality_or_speed_proven': False,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == '__main__':
    output = ROOT / 'research/20261022/R12_native_tile_layouts.json'
    if output.exists():
        raise RuntimeError('Preserve prior layout analysis')
    result = analyze()
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
