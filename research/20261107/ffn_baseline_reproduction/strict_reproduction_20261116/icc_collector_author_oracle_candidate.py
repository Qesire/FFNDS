"""REVIEW_REQUESTED: actual upstream stat_upd versus chunked collector, CPU only."""
import argparse
import ast
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from icc_four_boundary_collector import FourBoundaryCollector


def main(author_file, output):
    raw = author_file.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    if blob != '09083cec7c13afbf785f41ce38b8d7a647b3fb78': raise RuntimeError('author blob mismatch')
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'stat_upd']
    if len(nodes) != 1: raise RuntimeError('author function absent')
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(author_file), 'exec'), ns)
    torch.manual_seed(20261116)
    max_mag_error = max_norm_error = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for tokens, chunk_rows in ((17, 3), (31668, 256)):
            for family in ('CA', 'CD'):
                for source in range(2):
                    c = FourBoundaryCollector(torch, Path(tmp) / f'{tokens}_{family}{source}', str(source), 'high', family,
                                              tokens=tokens, d=3, m=5, layers=1, chunk_rows=chunk_rows, identity={'fixture': True})
                    magnitude, normalized = fixture(c, ns['stat_upd'], tokens, constant=source == 0)
                    max_mag_error = max(max_mag_error, magnitude)
                    max_norm_error = max(max_norm_error, normalized)
    collector_source = Path(__file__).with_name('icc_four_boundary_collector.py')
    result = {'status': 'PASS', 'collector_sha256': hashlib.sha256(collector_source.read_bytes()).hexdigest(),
              'author_git_blob': blob, 'fixture_token_counts': [17, 31668], 'chunk_rows': [3, 256],
              'max_abs_magnitude': max_mag_error, 'max_abs_normalized': max_norm_error,
              'scope': 'synthetic small channel widths, full production token count checked, no model or video'}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


def fixture(c, author, tokens, *, constant):
    if constant:
        anchor = {k: torch.ones(tokens, 3 if k in ('u','y') else 5, dtype=torch.bfloat16) for k in ('u','z','h','y')}
        current = anchor
    else:
        anchor, current = {}, {}
        for idx, k in enumerate(('u', 'z', 'h', 'y')):
            width = 3 if k in ('u', 'y') else 5
            scale = torch.arange(1, width + 1) * (idx + 1)
            anchor[k] = (torch.randn(tokens, width) * scale).bfloat16()
            current[k] = (anchor[k].float() + torch.randn(tokens, width) * scale * .2).bfloat16()
    family = c.family
    if family == 'CD':
        c.begin_site('anchor', 0)
        for k, v in anchor.items():
            c.observe(k, v)
        c.finish_site()
    c.begin_site('hit' if family == 'CD' else 'CA', 0)
    for k, v in current.items():
        c.observe(k, v)
    c.finish_site()
    c.finish()
    result = torch.load(c.root / 'stats_00.pt', weights_only=True)
    max_mag_error = max_norm_error = 0.0
    for k in anchor:
        layer = SimpleNamespace()
        if family == 'CD':
            author(layer, anchor[k].float().unsqueeze(0), 'cache', 'v')
        author(layer, current[k].float().unsqueeze(0), 'delta_mag' if family == 'CD' else 'mag', 'v')
        expected_mag = ((current[k].float() - anchor[k].float()) if family == 'CD' else current[k].float()).abs().mean(0)
        max_mag_error = max(max_mag_error, (result[k]['magnitude'] - expected_mag).abs().max().item())
        max_norm_error = max(max_norm_error, (result[k]['normalized'] - layer.v_mag).abs().max().item())
        torch.testing.assert_close(result[k]['magnitude'], expected_mag, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(result[k]['normalized'], layer.v_mag, rtol=2e-6, atol=2e-6)
    return max_mag_error, max_norm_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('author_stat_file', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    main(args.author_stat_file, args.output)
