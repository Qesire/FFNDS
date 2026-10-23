"""Portable check of the explicitly synthetic CPU normalization receipt."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root / 'research/20261023/C0_norm_statistics_CPU_readiness.json').read_text())
    assert receipt['status'] == 'PASS_36_SYNTHETIC_CPU_BITWISE_CASES_AND_SCALE_COLLISION'
    assert receipt['case_count'] == len(receipt['cases']) == 36
    for case in receipt['cases']:
        assert case['output_bitwise_equal'] and case['input_unchanged']
        assert case['dimension'] == 5120
        assert case['statistics_shape'] == [case['rows'], 1]
    assert receipt['scale_collision'] == {'output_BF16_bitwise_identical': True,
                                         'inverse_std_scale1': 1.0, 'inverse_std_scale4': 0.25}
    for name, expected in receipt['source_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    assert receipt['device'] == 'cpu' and receipt['CUDA_initialized'] is False
    assert receipt['new_model_calls'] == receipt['new_FFN_calls'] == receipt['new_training_updates'] == 0
    assert receipt['actual_GPU_or_whole_model_bitwise_result'] is None
    assert receipt['actual_GPU_latency_result'] is None
    print('PASS:36 synthetic CPU bit-controls; scale statistic available; no native GPU/model claim')


if __name__ == '__main__':
    main()
