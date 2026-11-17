"""Candidate tiny CPU author oracle. Run only after project pre-execution review.

Pass a checkout of ccccczzy/icc at the pinned commit. Original function bodies
are AST-extracted after verifying their Git blob identities; no copied oracle.
"""
import argparse
import ast
import hashlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from icc_stage0 import Factors, FourState, SourceStatistics

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'reports/20261114/g20_two_track_core/BWS/v1/source'))
from icc_decompose import factorize

BLOBS = {
    'DiT/adapters/sched_sp_step.py': '283c9e0715a3ea096f12c15c93dba1229e6339a8',
    'DiT/models/models_sp_step_decomp.py': 'b3673a09a44bf87a2a77c525841034a9c8cb09ec',
    'DiT/models/models_stat_svd.py': '09083cec7c13afbf785f41ce38b8d7a647b3fb78',
    'DiT/utils/utils_decomp.py': 'd053db79cfffba9b2c054a7e6b195066c77a4f10',
}

def load_tree(root, path):
    raw = (root / path).read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    if blob != BLOBS[path]:
        raise ValueError(f'author file mismatch: {path}: {blob}')
    return ast.parse(raw, filename=str(root / path))

def extract(tree, names, namespace):
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names): raise ValueError('missing author function')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-author>', 'exec'), namespace)

def main(root):
    ns = {'torch': torch, 'F': F}
    sched_tree = load_tree(root, 'DiT/adapters/sched_sp_step.py')
    sched_class = next(n for n in sched_tree.body if isinstance(n, ast.ClassDef) and n.name == 'sched_PERIOD')
    exec(compile(ast.Module(body=[sched_class], type_ignores=[]), '<pinned-scheduler>', 'exec'), ns)
    scheduler = ns['sched_PERIOD'](num_steps=4, period=2)
    scheduler.clr()
    assert [scheduler.step() for _ in range(4)] == [1, 0, 1, 0]
    scheduler.clr()
    assert scheduler.step() == 1
    sp = load_tree(root, 'DiT/models/models_sp_step_decomp.py')
    stat = load_tree(root, 'DiT/models/models_stat_svd.py')
    decomp = load_tree(root, 'DiT/utils/utils_decomp.py')
    extract(sp, ['linear_decomp'], ns)
    original = next(n for n in sp.body if isinstance(n, ast.ClassDef) and n.name == 'SpMlp')
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in ('forward_upd', 'forward_inc')]
    if len(methods) != 2: raise ValueError('missing original SpMlp methods')
    cls = ast.ClassDef(name='Oracle', bases=[ast.Name(id='object', ctx=ast.Load())], keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), '<pinned-SpMlp>', 'exec'), ns)
    extract(stat, ['stat_upd'], ns)
    extract(decomp, ['wsvd', 'wsvd_merge_lowrank', 'decomp_check', 'shift2ratio'], ns)
    torch.manual_seed(20261116)
    # Actual author stat_upd versus source-equal FP32 accumulator.
    for family in ('CA', 'CD'):
        acc, author = SourceStatistics(family), SimpleNamespace()
        for i, values in enumerate(([2., 4., 18.], [3., 8., 35.])):
            x = torch.tensor([[values, [-v for v in values]]])
            anchor = torch.full_like(x, 0.5)
            if family == 'CD': ns['stat_upd'](author, anchor, 'cache', 'v')
            ns['stat_upd'](author, x, 'mag' if family == 'CA' else 'delta_mag', 'v')
            acc.add(str(i), x, anchor if family == 'CD' else None)
        mean, root_scale, clipped = acc.scales(1000)
        author_mean = author.v_mag / author.v_cnt
        author_root = author_mean.sqrt()
        torch.testing.assert_close(mean, author_mean, rtol=0, atol=0)
        torch.testing.assert_close(root_scale, author_root, rtol=0, atol=0)
        torch.testing.assert_close(clipped, ns['shift2ratio'](author_root, 1000), rtol=0, atol=0)
        assert bool((clipped > root_scale).any())
    # Distinct constant CA and zero-delta CD cases compare directly to author.
    for family in ('CA', 'CD'):
        acc, author = SourceStatistics(family), SimpleNamespace()
        x = torch.full((1, 2, 3), 2.0)
        if family == 'CD': ns['stat_upd'](author, x, 'cache', 'v')
        ns['stat_upd'](author, x, 'mag' if family == 'CA' else 'delta_mag', 'v')
        _, normalized = acc.add('constant', x, x if family == 'CD' else None)
        torch.testing.assert_close(normalized, author.v_mag, rtol=0, atol=0)
        assert bool((normalized == 1).all())
    # Rectangular non-diagonal matrices, author SVD with independent scale clones.
    for shape in ((5, 3), (3, 5)):
        w = torch.randn(*shape); ls = torch.linspace(.1, 1, shape[0]); rs = torch.linspace(.2, 1, shape[1])
        for rank in (2, min(shape)):
            a, b, *_ = ns['wsvd_merge_lowrank'](w, rank, ls.clone(), rs.clone())
            # Test serialization preserves action; comparisons are factor action, not SVD signs.
            buf = io.BytesIO(); torch.save((a, b), buf); buf.seek(0)
            aa, bb = torch.load(buf, weights_only=True)
            x = torch.randn(4, shape[1]); torch.testing.assert_close(F.linear(F.linear(x, b), a), F.linear(F.linear(x, bb), aa), rtol=0, atol=0)
            candidate = factorize(w, rank, ls, rs)
            buf = io.BytesIO(); torch.save((candidate.left, candidate.right), buf); buf.seek(0)
            ca, cb = torch.load(buf, weights_only=True)
            torch.testing.assert_close(candidate.apply(x), F.linear(F.linear(x, cb), ca), rtol=0, atol=0)
            torch.testing.assert_close(candidate.apply(x), F.linear(F.linear(x, b), a), rtol=2e-5, atol=2e-5)
            if rank == min(shape): torch.testing.assert_close(a @ b, w, rtol=1e-5, atol=1e-5)
    # Author forward methods, genuine A/H/A/H reset and repeated-hit stress.
    author = ns['Oracle'](); author.fc1 = nn.Linear(3, 5); author.fc2 = nn.Linear(5, 3)
    author.act = nn.GELU(approximate='tanh'); author.drop1 = author.drop2 = author.norm = nn.Identity()
    author.cache = SimpleNamespace(content={})
    author.fc1_wa, author.fc1_wb = torch.randn(5, 2), torch.randn(2, 3)
    author.fc2_wa, author.fc2_wb = torch.randn(3, 2), torch.randn(2, 5)
    port = FourState(Factors(author.fc1_wa, author.fc1_wb), Factors(author.fc2_wa, author.fc2_wb))
    key = ('request', 'H', 19)
    for cycle in range(2):
        x = torch.randn(1, 4, 3); expected = author.forward_upd(x)
        c = author.cache.content; actual = port.anchor(key, c['fc1_x'], c['fc1_y'], c['fc2_x'], c['fc2_y'])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        before = tuple(v.clone() for v in port.cache)
        for got, name in zip(port.cache, ('fc1_x', 'fc1_y', 'fc2_x', 'fc2_y')):
            torch.testing.assert_close(got, c[name], rtol=0, atol=0)
        zero_delta = port.hit(key, x)
        torch.testing.assert_close(zero_delta, author.forward_inc(x), rtol=0, atol=0)
        torch.testing.assert_close(zero_delta, expected, rtol=0, atol=0)
        for dx in (0.13, -0.27):
            v = x + dx
            torch.testing.assert_close(port.hit(key, v), author.forward_inc(v), rtol=0, atol=0)
            for old, now in zip(before, port.cache):
                torch.testing.assert_close(now, old, rtol=0, atol=0)
    port.reset()
    try:
        port.hit(key, x)
    except RuntimeError as error:
        if str(error) != 'missing or mismatched anchor':
            raise
    else:
        raise AssertionError('reset preserved a usable anchor')
    print('PASS: pinned source identities; CA/CD statistics; wsvd action/reload; author scheduler p2 and SpMlp immutable repeated hits')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('author_checkout', type=Path)
    main(parser.parse_args().author_checkout)



