"""Restore the three archived frozen CAL assets for CPU algebra verification."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

EXPECTED = {
    'plan_assets/bases/P_v.npy': 'cfc5a6febf99b37f04ba2989038546bda0d404d9e15db659793bbc88125ca3ff',
    'calibration/self_high_transform.npz': '8d6f7651ce40d2f6f389ba7f981c573a84b2b3d3d9c8aa1fbeae95d9914ef864',
    'calibration/self_low_transform.npz': '04c15a63f6e4ffd34b61714a1be79d2651c195d104e4007aa5c8c22e2dd356e1',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    archive = json.loads(Path(__file__).with_name('CAL_ASSETS.json').read_text())
    assert {entry['path'] for entry in archive['files']} == set(EXPECTED)
    for entry in archive['files']:
        payload = base64.b64decode(entry['base64'], validate=True)
        digest = hashlib.sha256(payload).hexdigest()
        assert digest == entry['sha256'] == EXPECTED[entry['path']]
        path = args.out / entry['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != payload:
            raise FileExistsError(f'Different file already exists: {path}')
        path.write_bytes(payload)
        print(f'{entry["path"]}: {digest}')


if __name__ == '__main__':
    main()
