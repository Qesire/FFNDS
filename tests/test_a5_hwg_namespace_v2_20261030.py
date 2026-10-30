"""Adversarial pure import fixtures; no model, tensor corpus or CUDA calls."""
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('a5_ns_v2', ROOT / 'tools/a5_hwg_matched_capture_namespace_v2_20261030.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class NamespaceFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directory = self.root / 'fixture_namespace'
        self.directory.mkdir()
        (self.directory / 'leaf.py').write_text('fixture = True\n')
        sys.path.insert(0, str(self.root))
        self.module = importlib.import_module('fixture_namespace')
        self.patcher = patch.object(m.owner, 'ROOT', self.root)
        self.patcher.start()
        self.rows = {'fixture_namespace/leaf.py': {}}

    def tearDown(self):
        self.patcher.stop()
        sys.path.remove(str(self.root))
        sys.modules.pop('fixture_namespace', None)
        self.temp.cleanup()

    def validate(self):
        return m.validate_namespace('fixture_namespace', self.module, 'fixture_namespace', self.rows)

    def test_real_pep420_single_exact_frozen_directory(self):
        self.assertEqual(self.validate()['kind'], 'PEP420_NAMESPACE')

    def test_empty_or_additional_search_location(self):
        for paths in [[], [str(self.directory), str(self.root / 'outside')]]:
            self.module.__path__ = paths
            with self.assertRaises(RuntimeError): self.validate()

    def test_wrong_in_tree_directory(self):
        other = self.root / 'wrong'; other.mkdir()
        self.module.__path__ = [str(other)]
        self.module.__spec__.submodule_search_locations = [str(other)]
        with self.assertRaises(RuntimeError): self.validate()

    def test_symlink_escape(self):
        outside = tempfile.TemporaryDirectory()
        try:
            self.directory.rename(self.root / 'old')
            self.directory.symlink_to(Path(outside.name), target_is_directory=True)
            with self.assertRaises(RuntimeError): self.validate()
        finally: outside.cleanup()

    def test_spec_origin_file_loader_and_identity_spoof(self):
        for attribute, value in [('origin', 'namespace'), ('loader', None), ('name', 'wrong')]:
            old = getattr(self.module.__spec__, attribute)
            setattr(self.module.__spec__, attribute, value)
            with self.assertRaises(RuntimeError): self.validate()
            setattr(self.module.__spec__, attribute, old)
        self.module.__file__ = str(self.directory / 'leaf.py')
        with self.assertRaises(RuntimeError): self.validate()

    def test_missing_spec_or_frozen_descendant(self):
        self.rows.clear()
        with self.assertRaises(RuntimeError): self.validate()
        self.module.__spec__ = None
        with self.assertRaises(RuntimeError): self.validate()

    def test_new_initializer_is_not_a_namespace(self):
        (self.directory / '__init__.py').write_text('')
        with self.assertRaises(RuntimeError): self.validate()

    def test_alias_map_uses_frozen_paths_and_omits_extension_namespaces(self):
        rows = [{'path': '.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/QTensor.py'},
                {'path': '.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/__init__.py'}]
        result = m.namespace_bindings(rows)
        self.assertEqual(result['quant_cy.base'], '.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base')
        self.assertEqual(result['HiFloat4.hif4_gpu.quant_cy.base'], result['quant_cy.base'])
        self.assertNotIn('quant_cy', result)
        self.assertNotIn('hifxg_quant', result)


if __name__ == '__main__': unittest.main()
