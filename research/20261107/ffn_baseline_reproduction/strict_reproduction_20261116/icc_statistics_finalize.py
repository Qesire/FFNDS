"""Aggregate committed source-level ICC statistics into four boundary scales.

No model execution. Source identities and their statistics directories must be
explicitly supplied by the reviewed native calibration manifest. Partial source
transactions, nonfinite vectors, mismatched hashes and duplicate sources fail.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import torch

BOUNDARIES = ('u', 'z', 'h', 'y')


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def finalize(source_dirs, *, expert, family, expected_sources, output, oracle_receipt, collector_file,
             expected_statistics=None, layers=40, d=5120, m=13824, tokens=31668):
    if family not in ('CA','CD') or expert not in ('high','low'):
        raise ValueError('invalid family/expert')
    if len(expected_sources) != 11 or len(set(expected_sources)) != 11 or set(source_dirs) != set(expected_sources):
        raise ValueError('requires exactly 11 distinct, explicitly bound sources')
    if expected_statistics is not None and set(expected_statistics) != {
        (source, layer) for source in expected_sources for layer in range(layers)
    }:
        raise ValueError('missing previously verified source/layer hashes')
    oracle = json.loads(Path(oracle_receipt).read_text(encoding='utf8'))
    if oracle.get('status') != 'PASS' or oracle.get('collector_sha256') != sha(collector_file):
        raise ValueError('missing matching successful collector oracle receipt')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    manifest = {'family':family, 'expert':expert, 'source_order':list(expected_sources),
                'tokens':tokens, 'layers':layers, 'ratio_cap':1000, 'factorization_epsilon':1e-6,
                'collector_sha256':sha(collector_file), 'oracle_receipt_sha256':sha(oracle_receipt),
                'source_statistics':[], 'outputs':[]}
    roots, identities = {}, {}
    for source in expected_sources:
        root = Path(source_dirs[source])
        complete = json.loads((root/'COMPLETE.json').read_text(encoding='utf8'))
        for key, value in (('source',source),('expert',expert),('family',family),('tokens',tokens),('layers',layers)):
            if complete[key] != value: raise ValueError(f'source identity mismatch: {source} {key}')
        roots[source] = root
        identities[source] = complete['identity']
    for layer in range(layers):
        totals = {k:torch.zeros(d if k in ('u','y') else m, dtype=torch.float32) for k in BOUNDARIES}
        for source in expected_sources:
            root = roots[source]; file = root/f'stats_{layer:02d}.pt'
            meta = json.loads((root/f'stats_{layer:02d}.json').read_text(encoding='utf8'))
            for key, value in (('source',source),('expert',expert),('family',family),('tokens',tokens),('layer',layer)):
                if meta[key] != value: raise ValueError(f'statistics identity mismatch: {source} {layer} {key}')
            if meta.get('identity') != identities[source]: raise ValueError('transaction identity mismatch')
            content = file.read_bytes()
            actual_sha = hashlib.sha256(content).hexdigest()
            if actual_sha != meta['stats_sha256']: raise ValueError('statistics content hash mismatch')
            if expected_statistics is not None and actual_sha != expected_statistics[source, layer]:
                raise ValueError('statistics changed after transaction verification')
            data = torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)
            if set(data) != set(BOUNDARIES): raise ValueError('four boundaries required')
            for key in BOUNDARIES:
                value = data[key]['normalized']; mag = data[key]['magnitude']
                if value.shape != totals[key].shape or value.dtype != torch.float32 or mag.shape != value.shape or mag.dtype != torch.float32:
                    raise ValueError('invalid channel statistic')
                if not torch.isfinite(value).all().item() or not torch.isfinite(mag).all().item() or (mag < 0).any().item():
                    raise ValueError('nonfinite or negative statistic')
                expected = (mag-mag.min()+1e-6)/(mag.max()-mag.min()+1e-6)
                if not torch.equal(value, expected): raise ValueError('stored min-max differs from magnitude')
                totals[key] += value
            manifest['source_statistics'].append({'source':source,'layer':layer,'sha256':actual_sha})
        scalars = {}
        for key, total in totals.items():
            mean = total/11; root = mean.sqrt(); capped = root.clamp_min(root.max()/1000)
            scalars[key] = {'mean_normalized':mean,'sqrt':root,'capped':capped}
        path = output/f'scales_{layer:02d}.pt'
        with path.open('xb') as f:
            torch.save(scalars, f); f.flush(); os.fsync(f.fileno())
        manifest['outputs'].append({'layer':layer,'file':path.name,'sha256':sha(path)})
    with (output/'MANIFEST.json').open('x',encoding='utf8') as f:
        json.dump(manifest,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    return manifest


