"""Separate fixed-mask transfer from conditional room in local value records.

The development-label common mask is an unexecuted additive reference. It is
not the registered training control, a new native action or a usable policy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def analyze_reference(result):
    cases = [case for case in result['cases'] if case['index'] in (0, 10)]
    if len(cases) != 2 or {case['index'] for case in cases} != {0, 10}:
        raise ValueError('Exactly the original two exposed development sources required')
    benefit = [math.fsum(case['one_point_SSE'][g] - case['two_point_SSE'][g] for case in cases)
               for g in range(64)]
    common = sorted(sorted(range(64), key=lambda g: (-benefit[g], g))[:32])
    reference = math.fsum(case['two_point_SSE'][g] if g in common else case['one_point_SSE'][g]
                          for case in cases for g in range(64))
    totals = result['analysis']['evaluation_totals']
    return {'id': 'R14N0_POSTHOC_COMMON_MASK_REFERENCE',
            'evidence_level': 'UNEXECUTED_SHARED_MASK_REFERENCE_FROM_LOCAL_NATIVE_VALUES',
            'development_indices': [0, 10], 'reference_two_point_stripes': common,
            'reference_common_SSE': reference,
            'registered_training_fixed_SSE': totals['fixed_SSE'], 'executed_native_oracle_SSE': totals['oracle_SSE'],
            'fixed_transfer_part_of_gap': totals['fixed_SSE'] - reference,
            'conditional_part_under_additive_reference': reference - totals['oracle_SSE'],
            'oracle_gain_over_development_label_common_reference': 1 - totals['oracle_SSE'] / reference,
            'new_native_FFN_calls': 0, 'new_model_generations': 0,
            'registered_training_mask_changed': False,
            'limits': 'Uses development labels after the frozen experiment. This additional common mask was NOT physically run. The gap decomposition assumes additive per-stripe values with fixed native arithmetic; registered final3masks had composition controls but this extra mask did not. Not a full-scene/global/video bound, independent validation or deployable control.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve the first post-hoc reference')
    result = analyze_reference(json.loads(args.input.read_text()))
    result['native_result_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
