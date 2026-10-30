"""Versioned launch repair for genuine, exactly bound PEP420 packages.

The executed v1 acquisition remains immutable. Its numerical operations,
resource guards and independent verifier are reused without modification.
"""
from __future__ import annotations

import importlib.machinery
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a5_hwg_matched_capture_20261030 as owner

SELF = 'tools/a5_hwg_matched_capture_namespace_v2_20261030.py'
PREFIX_ROOTS = {
    'HiFloat4': '.runtime/code/HiF4_Sp/HiFloat4',
    'hif4_gpu': '.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu',
    'quant_cy': '.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy',
    'hifxg_quant': '.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/cusrc',
    'quant': '.runtime/code/Wan2.2-I2V-A14B-W4A4/quant',
    'lightx2v': '.runtime/code/LightX2V/lightx2v',
    'lightx2v_platform': '.runtime/code/LightX2V/lightx2v_platform',
}


def namespace_bindings(rows):
    """Exact module-name/directory map derived only from frozen source paths."""
    result = {}
    paths = {r['path'] for r in rows}
    for prefix, base in PREFIX_ROOTS.items():
        if prefix == 'hifxg_quant':  # Extension modules are never namespaces.
            continue
        for rel in paths:
            if not rel.startswith(base + '/'):
                continue
            directory = Path(rel).parent
            while str(directory) == base or str(directory).startswith(base + '/'):
                parts = directory.relative_to(base).parts
                if all(p.isidentifier() for p in parts):
                    name = '.'.join((prefix, *parts))
                    if str(directory / '__init__.py') not in paths:
                        result[name] = str(directory)
                if str(directory) == base:
                    break
                directory = directory.parent
    return dict(sorted(result.items()))


def validate_namespace(name, module, expected, rows):
    """Reject merged, renamed, unbound and escaped namespace search paths."""
    need = owner.need
    spec = getattr(module, '__spec__', None)
    need(spec is not None and spec.name == name, 'namespace spec identity: ' + name)
    need(getattr(module, '__file__', None) is None and spec.origin is None,
         'namespace file/origin identity: ' + name)
    need(isinstance(spec.loader, importlib.machinery.NamespaceLoader)
         and getattr(module, '__loader__', None) is spec.loader,
         'namespace loader identity: ' + name)
    locations = list(getattr(module, '__path__', ()))
    spec_locations = list(spec.submodule_search_locations or ())
    need(len(locations) == len(spec_locations) == 1,
         'namespace must have exactly one bound location: ' + name)
    root = owner.ROOT.resolve()
    target = (root / expected).resolve()
    need(target.is_dir() and target.is_relative_to(root), 'namespace root escaped: ' + name)
    need(Path(locations[0]).resolve() == Path(spec_locations[0]).resolve() == target,
         'namespace search locations differ: ' + name)
    need(not (target / '__init__.py').exists(), 'namespace acquired initializer: ' + name)
    descendants = [r for r in rows if r.startswith(expected + '/')]
    need(bool(descendants), 'namespace has no frozen descendant: ' + name)
    return {'kind': 'PEP420_NAMESPACE', 'path': expected,
            'bound_descendant_count': len(descendants)}


def validate_import_origins(protocol):
    rows = {b['path']: b for b in protocol['repo_sources']}
    expected = namespace_bindings(protocol['repo_sources'])
    owner.need(protocol['namespace_bindings'] == expected, 'namespace map differs from frozen sources')
    seen = {}
    for name, module in list(sys.modules.items()):
        if not any(name == p or name.startswith(p + '.') for p in PREFIX_ROOTS):
            continue
        origin = getattr(module, '__file__', None)
        if origin is None:
            owner.need(name in expected, 'unbound native namespace: ' + name)
            seen[name] = validate_namespace(name, module, expected[name], rows)
            continue
        path = Path(origin).resolve()
        owner.need(path.is_relative_to(owner.ROOT.resolve()), 'native import escaped: ' + name)
        rel = str(path.relative_to(owner.ROOT.resolve()))
        owner.need(rel in rows, 'unbound native file: ' + name + ' -> ' + rel)
        owner.check_binding(owner.ROOT, rows[rel])
        seen[name] = rel
    owner.need(any(n.startswith('quant.hif4_backend') for n in seen),
               'actual HiF4 backend was not imported')
    return seen


def main():
    owner.SELF = SELF
    owner.validate_import_origins = validate_import_origins
    owner.main()


if __name__ == '__main__':
    main()
