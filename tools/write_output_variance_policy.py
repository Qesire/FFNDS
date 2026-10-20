"""Freeze all160 native output-affine coefficients around an unchanged RR1 base."""
import argparse
import json
from pathlib import Path

from ar2_checkpoint import atomic_json,digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-policy',type=Path,required=True)
    ap.add_argument('--fit-report',type=Path,required=True)
    ap.add_argument('--kernel-probe',type=Path,required=True)
    ap.add_argument('--identity-control',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    fit=json.loads(args.fit_report.read_text());probe=json.loads(args.kernel_probe.read_text())
    kernel=Path(__file__).with_name('output_variance_kernel.py')
    if (fit['status']!='COMPLETED_VM1_FULL160_MOMENT_FIT' or len(fit['positions'])!=160
            or probe['status']!='PASS_VM1_CUDA_AFFINE_BITWISE_AND_IDENTITY'
            or probe['fit_fingerprint']!=fit['fingerprint'] or probe['kernel_source_sha256']!=digest(kernel)):
        raise RuntimeError('Need a complete moment fit and matching native operator proof')
    policy=json.loads(args.base_policy.read_text())
    if policy['kind']!='CALIBRATION_STATIC_PACKED_REFERENCE' or 'residual_fit' not in policy:
        raise RuntimeError('Expected the frozen native RR1 base')
    policy.update(kind='OUTPUT_VARIANCE_CALIBRATED_REFERENCE',
        experiment='VM1_OUTPUT_AFFINE_ON_ORIGINAL12_RR1_20261021',
        base_policy={'path':str(args.base_policy.resolve()),'sha256':digest(args.base_policy)},
        base_policy_sha256=digest(args.base_policy),
        moment_fit={'path':str(args.fit_report.resolve()),'sha256':digest(args.fit_report),'fingerprint':fit['fingerprint']},
        kernel_probe={'path':str(args.kernel_probe.resolve()),'sha256':digest(args.kernel_probe)},
        kernel_source_sha256=digest(kernel),
        runtime_source_sha256=digest(Path(__file__).with_name('output_variance_runtime.py')),
        identity_control=args.identity_control,
        quality_or_speed_acceptance=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists() and json.loads(args.output.read_text())!=policy:
        raise RuntimeError('Refusing to overwrite a different frozen affine policy')
    atomic_json(args.output,policy);print(json.dumps({'path':str(args.output.resolve()),'sha256':digest(args.output)}))


if __name__=='__main__':main()
