"""Read-only build of the G20 R1 assets manifest from capture sidecars.

Scans CAL4 + DEV5 capture sidecars (small JSON), records file_sha256 from each
sidecar and the sidecar's own SHA. The .pt payloads are NOT hashed here; the
guarded job verifies every payload SHA at load time via selector.load_pair.
Run on the cluster; writes assets_g20.json into the stage directory.
"""
import hashlib
import json
import sys
from pathlib import Path

BASE = Path('/public/home/scc/pb23061276/projects/wan-quant/work/g19_native_capture10_20261103')
CAL = ['chip10_02', 'chip10_04', 'chip10_07', 'chip10_09']
DEV = ['chip10_01', 'chip10_03', 'chip10_05', 'chip10_06', 'chip10_08']
CHECKPOINTS = {
    'high': {'path': '/public/home/scc/pb23061276/projects/wan-quant/weights/Wan2.2-I2V-A14B-LightX2V-Distill-4step/high_noise_model/wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors',
             'sha256': '39d3af74e46e59e989db5cc36f401e666f3ae77c7648eb55d3aff9fee47716fa'},
    'low': {'path': '/public/home/scc/pb23061276/projects/wan-quant/weights/Wan2.2-I2V-A14B-LightX2V-Distill-4step/low_noise_model/wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors',
            'sha256': 'a3b8680affe2ab2e2121210a83ffa7d351f77da18191e57bb359d7839beba8a2'},
}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    stage = Path(sys.argv[1])
    captures = []
    missing = []
    for source in CAL + DEV:
        cdir = BASE / source / ('output_' + source) / 'capture'
        for step in range(4):
            for layer in range(40):
                stem = '%s__seed42__s%02d__b%02d' % (source, step, layer)
                pt = cdir / (stem + '.pt')
                sc = cdir / (stem + '.pt.json')
                if not pt.is_file() or not sc.is_file():
                    missing.append(str(pt))
                    continue
                meta = json.loads(sc.read_text())
                if meta['file_sha256'] is None:
                    raise RuntimeError('sidecar missing file_sha256: ' + str(sc))
                captures.append({'path': str(pt), 'sha256': meta['file_sha256'],
                                 'sidecar': str(sc), 'sidecar_sha256': sha(sc),
                                 'metadata': meta})
    if missing:
        raise RuntimeError('missing captures: %d e.g. %s' % (len(missing), missing[:3]))
    expected = 9 * 4 * 40
    if len(captures) != expected:
        raise RuntimeError('capture count %d != %d' % (len(captures), expected))
    assets = {'schema': 'G20_R1_SCREEN_ASSETS_V1', 'cal_sources': CAL, 'dev_sources': DEV,
              'cal_seed': 42, 'checkpoints': CHECKPOINTS, 'captures': captures,
              'expected_geometry': [21, 29, 52], 'flattened_tokens': 31668,
              'meaning': 'CAL4 train + DEV5 development source-out; uniform 2048-row panels; '
                         'native pre-W1 U (z) and pre-gate W2 output; per-channel current gate; '
                         'DEV is exposed development data, NOT blind confirmation.',
              'status': 'READ_ONLY_SIDECAR_SCAN_PAYLOAD_SHA_VERIFIED_AT_LOAD'}
    dest = stage / 'assets_g20.json'
    with dest.open('x', encoding='utf-8') as f:
        json.dump(assets, f, indent=2, allow_nan=False)
    print(json.dumps({'captures': len(captures), 'cal': len(CAL), 'dev': len(DEV),
                      'assets_sha256': sha(dest), 'path': str(dest)}))


if __name__ == '__main__':
    main()