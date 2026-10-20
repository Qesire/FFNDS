"""Write an immutable deployment pointer to HF-cached calibration parameters."""
import argparse
import json
from pathlib import Path

from ar2_checkpoint import atomic_json, digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fits',type=Path,nargs=2,required=True)
    ap.add_argument('--support',choices=['C','R'],default='C')
    ap.add_argument('--mode',choices=['unit','uncentered_gain','unit_mean','centered_gain_bias'],default='centered_gain_bias')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    fits={}
    cache_path=ROOT/'.runtime/hif4_rebuilt_cache.json'
    for path in args.fits:
        report=json.loads(path.read_text())
        if report['status']!='COMPLETED_CALIBRATION_PARAMETER_FIT' or len(report['positions'])!=80:
            raise RuntimeError('Complete per-expert training fits required')
        fits[report['expert']]={'path':str(path.resolve()),'sha256':digest(path),'fingerprint':report['fingerprint']}
    if set(fits)!={'high_noise_model','low_noise_model'}:
        raise RuntimeError('Need both trained experts')
    policy={'kind':'CALIBRATION_STATIC_PACKED_REFERENCE','experiment':'CALIBRATION12_CSTATIC_CENTERED_GAIN_BIAS_20261021',
            'support':args.support,'mode':args.mode,'fit_reports':fits,
            'cache_manifest':{'path':str(cache_path),'sha256':digest(cache_path)},
            'training_protocol_sha256':digest(ROOT/'config/cstatic_centered_fit.json'),
            'fixed_candidate_mode':args.support=='C' and args.mode=='centered_gain_bias',
            'gate_or_quality_acceptance':False,'weight_preparation':'CPU per-position packing before generation'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists() and json.loads(args.output.read_text())!=policy:
        raise RuntimeError('Refusing to replace another learned policy identity')
    atomic_json(args.output,policy)
    print(json.dumps({'path':str(args.output.resolve()),'sha256':digest(args.output)}))


if __name__=='__main__':
    main()
