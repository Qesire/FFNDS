"""Prepare true four-token coordinate groups, without collecting any operands.

This is a data-layout draft for a distinct future operator check. It is not a
model protocol, registration, execution command or approval to collect data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def stripe_rows(stripe, grid=(21, 29, 52)):
    frames, height, width = grid
    if frames <= 0 or height <= 0 or width <= 0 or width % 4:
        raise ValueError('Complete horizontal4 layout requires a divisible width')
    per_row = width // 4
    if not 0 <= stripe < frames * height * per_row:
        raise ValueError('Stripe outside native grid')
    frame, rest = divmod(stripe, height * per_row)
    y, x_group = divmod(rest, per_row)
    first = (frame * height + y) * width + 4 * x_group
    return list(range(first, first + 4))


def select_stripes(index, count=64, grid=(21, 29, 52)):
    frames, height, width = grid
    stripe_rows(0, grid)  # Reject incompatible native boundaries.
    selected = sorted(random.Random(20261022 + index).sample(range(frames * height * (width // 4)), count))
    return selected, sorted(row for stripe in selected for row in stripe_rows(stripe, grid))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve the first coordinate draft')
    cases = []
    for index in (0, 1, 10, 11):
        stripes, rows = select_stripes(index)
        cases.append({'index': index, 'original_R9_fit_source': bool(index % 2),
                      'planned_stripe_ids': stripes, 'planned_native_row_ids': rows,
                      'observed_new_operands': 0})
    result = {'id': 'R13_COMPLETE_HORIZONTAL4_OPERAND_LAYOUT_DRAFT',
        'status': 'COORDINATE_PLAN_ONLY_NOT_COLLECTED_NOT_REGISTERED_NOT_ELIGIBLE',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'position_proposed': [2, 20], 'native_grid': [21, 29, 52],
        'sampling': '64 stripes uniformly without replacement from7917; random.Random(20261022+original_index), label-blind coordinates,4 consecutive x rows per stripe.',
        'planned_rows_per_case': 256,
        'cohort': 'Four already exposed OP2 development sources, two in original R9 odd fit and two nonfit even; not fresh quality evaluation.',
        'cases': cases,
        'required_future_payload': ['V1_post_Q1_BF16', 'original_full_TeacherY', 'frozen_UB1_StudentY', 'current_gate', 'original_row_ids_and_grid'],
        'required_future_identity_controls': 'Original source/model/input/quantizer SHA, fullM/stride, original full TeacherY and UB1 reproduction, exact affine asset bindings, no-op native control. These have not been run for this coordinate plan.',
        'new_model_generations': 0, 'new_operand_payloads': 0,
        'native_registration': None, 'execution_authority': None,
        'scope': 'This resolves coordinate specification only. It grants no model execution, native check, threshold, predictor fitting or rollout. A future registered study must choose its complete controls, measurements, compute/storage caps and stopping rule before collection.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Prepared4 label-blind coordinate cases;64 complete horizontal stripes each;0 operands collected.')


if __name__ == '__main__':
    main()
