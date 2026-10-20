"""Bind a completed all-position residual fit to its frozen packed base policy."""
import argparse
import json
from pathlib import Path

from ar2_checkpoint import atomic_json, digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-policy',type=Path,required=True)
    ap.add_argument('--residual-fit',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--experiment',default='RR1_CALIBRATION12_CSTATIC_WITH_LOW_RANK_RESIDUAL_20261021')
    args=ap.parse_args()
    policy=json.loads(args.base_policy.read_text())
    report=json.loads(args.residual_fit.read_text())
    if (policy['kind']!='CALIBRATION_STATIC_PACKED_REFERENCE' or policy['support']!='C'
            or policy['mode']!='centered_gain_bias' or 'residual_fit' in policy):
        raise RuntimeError('Requires unmodified fixed C-centered base')
    if report['status']!='COMPLETED_RR1_FULL160_CALIBRATION_FIT' or len(report['positions'])!=160:
        raise RuntimeError('Requires all160 residual fits')
    policy.update(experiment=args.experiment,
        base_policy_sha256=digest(args.base_policy),
        residual_fit={'path':str(args.residual_fit.resolve()),'sha256':digest(args.residual_fit),
            'fingerprint':report['fingerprint']},
        auxiliary_compute={'rank':report['selected']['rank'],'two_additional_BF16_linears_per_FFN':True,
            'MACs_over_original_dense_FFN':report['selected']['rank']/13824,
            'included_in_speed_acceptance':False})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists() and json.loads(args.output.read_text())!=policy:
        raise RuntimeError('Refusing to overwrite another learned policy')
    atomic_json(args.output,policy)
    print(json.dumps({'path':str(args.output.resolve()),'sha256':digest(args.output)}))


if __name__=='__main__':
    main()
