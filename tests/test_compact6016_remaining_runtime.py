"""Portable tiny CPU fakes; no model, GPU, torch or real exports loaded."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import compact6016_remaining_runtime as runtime
import compact6016_remaining_contract as contract


class Tensor:
    shape = (1,)
    dtype = 'tiny'
    def __init__(self, name):
        self.name = name


def binding(root, name, payload):
    path = root / name
    path.write_bytes(payload)
    return {'path': name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


def fixture(root):
    old = {'path': 'old.json', 'sha256': 'a' * 64, 'bytes': 10}
    cp = binding(root, 'completion.json', json.dumps({'id': runtime.completion.ID, 'original_protocol': old}).encode())
    training = {'path': 'new.json', 'sha256': 'b' * 64, 'bytes': 11}
    pack_binding = binding(root, 'pack.bin', b'complete pack')
    pack_binding['tensors'] = {k: {'shape': [1], 'dtype': 'tiny', 'sha256': k} for k in runtime.DEPLOY_KEYS}
    pack = {k: Tensor(k) for k in runtime.DEPLOY_KEYS}
    shards, records, biases, cells = [], {e: {} for e in runtime.EXPERTS}, {}, []
    for e in runtime.EXPERTS:
        for b in range(40):
            shards.append({'expert': e, 'block': b, 'tensors': {k: {'shape': [1], 'sha256': k} for k in ('W1', 'W2', 'smooth1', 'smooth2')}})
            records[e][f'blocks.{b}.ffn.0'] = {'weight_qdq': Tensor('W1'), 'smooth_scale': Tensor('smooth1')}
            records[e][f'blocks.{b}.ffn.2'] = {'weight_qdq': Tensor('W2'), 'smooth_scale': Tensor('smooth2')}
            biases[e, b] = {'b1': Tensor('b1'), 'b2': Tensor('b2')}
            for init in runtime.INITIALIZATIONS:
                cell_key = (e, b, init)
                if cell_key in contract.OLD_CELLS:
                    producer, proto, rid = runtime.completion.ORIGINAL_ID, old, runtime.completion._original_run_id(e, b, init)
                elif cell_key in contract.PARENT_CELLS:
                    producer, proto, rid = runtime.completion.ID, cp, runtime.completion.function_id(e, b) + '_' + init
                else:
                    producer, proto, rid = runtime.remaining.ID, training, runtime.remaining.function_id(e, b) + '_' + init
                cells.append({'expert': e, 'block': b, 'initialization': init, 'producer_id': producer,
                              'producer_protocol': proto, 'run_id': rid, 'steps': list(runtime.steps_for(e)),
                              'export': copy.deepcopy(pack_binding), 'result': {}, 'support_file': {}})
    view = binding(root, 'view.json', json.dumps({'shards': shards}).encode())
    manifest = {'schema': 'ffnds.compact6016.deployment_bank.mixed_lineage.v1',
                'status': 'METADATA_COMPLETE_TENSOR_LOAD_PENDING', 'producer_id': runtime.remaining.ID,
                'training_protocol': training, 'completion_protocol': cp, 'asset_view': view,
                'parts': [], 'cells': cells}
    return manifest, SimpleNamespace(records=records), biases, pack


class RemainingRuntimeTests(unittest.TestCase):
    def invoke(self, root, manifest, base, biases, pack, *, verified=None, loader=None):
        calls = []
        def load(path):
            calls.append(path)
            return pack
        with patch.object(runtime.remaining, 'assemble_manifest', return_value=manifest if verified is None else verified), \
             patch.object(runtime, 'tensor_sha', side_effect=lambda value: value.name), \
             patch.object(runtime, 'validate_deployment_pack', return_value={'gate': 'checked'}):
            result = runtime.load_bank(manifest, 'energy94', root, base, biases, repo_root=root, loader=loader or load)
        return result, calls

    def test_all_three_producers_load_exact_80_and_reuse_runtime_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, base, biases, pack = fixture(root)
            positions, calls = self.invoke(root, manifest, base, biases, pack)
            self.assertEqual((len(positions), len(calls)), (80, 80))
            self.assertEqual({v['producer_scope'] for v in positions.values()},
                             {runtime.completion.ORIGINAL_ID, runtime.completion.ID, runtime.remaining.ID})
            import compact6016_native_runtime as native
            self.assertIs(runtime.Compact6016Runtime, native.Compact6016Runtime)
            self.assertIs(runtime.Compact6016Residency, native.Compact6016Residency)

    def test_rejects_wrong_scope_prefix_protocol_and_changed_manifest_before_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, base, biases, pack = fixture(root)
            for producer in (runtime.completion.ORIGINAL_ID, runtime.completion.ID, runtime.remaining.ID):
                for change in ('producer', 'prefix', 'protocol'):
                    manifest = copy.deepcopy(original)
                    cell = next(c for c in manifest['cells'] if c['producer_id'] == producer)
                    if change == 'producer': cell['producer_id'] = 'foreign'
                    if change == 'prefix': cell['run_id'] = 'wrong_prefix'
                    if change == 'protocol': cell['producer_protocol'] = 'wrong_protocol'
                    with self.subTest(producer=producer, change=change), self.assertRaises(ValueError):
                        self.invoke(root, manifest, base, biases, pack, loader=lambda _: self.fail('loaded'))
            manifest = copy.deepcopy(original)
            manifest['producer_id'] = runtime.completion.ID
            with self.assertRaises(ValueError):
                self.invoke(root, manifest, base, biases, pack, loader=lambda _: self.fail('loaded'))
            changed = copy.deepcopy(original)
            changed['cells'][0]['run_id'] = 'altered'
            with self.assertRaises(ValueError):
                self.invoke(root, changed, base, biases, pack, verified=original, loader=lambda _: self.fail('loaded'))

    def test_corrupt_pack_and_actual_tensor_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, base, biases, pack = fixture(root)
            (root / 'pack.bin').write_bytes(b'corrupted pack')
            with self.assertRaises(ValueError):
                self.invoke(root, manifest, base, biases, pack, loader=lambda _: self.fail('loaded'))
            (root / 'pack.bin').write_bytes(b'complete pack')
            pack['W1'] = Tensor('changed')
            with self.assertRaises(ValueError):
                self.invoke(root, manifest, base, biases, pack)


if __name__ == '__main__':
    unittest.main()
