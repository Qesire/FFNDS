"""Byte integrity, lineage and accounting controls; all operands synthetic."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from tools import e0_raw_bf16_full21tau_analysis as m


def tiny_packet():
    shapes = {name: ([168] if name == 'rows' else ([2] if name in ('gate', 'shift', 'scale') else [168, 2]))
              for name in m.SHAPES}
    header, body, meta = {}, b'', {}
    for name, shape in shapes.items():
        raw = struct.pack('<168q', *m.core.rows()) if name == 'rows' else np.zeros(shape, dtype='<u2').tobytes()
        header[name] = {'dtype': 'I64' if name == 'rows' else 'BF16', 'shape': shape,
                        'data_offsets': [len(body), len(body)+len(raw)]}
        body += raw
        meta[name] = {'shape': shape, 'dtype': 'int64' if name == 'rows' else 'bfloat16',
                      'raw_tensor_sha256': hashlib.sha256(raw).hexdigest()}
    h = json.dumps(header).encode()
    raw = struct.pack('<Q', len(h))+h+body
    return raw, {'payload': {'path': 'tiny.safetensors', 'bytes': len(raw), 'sha256': m.digest(raw)},
                 'tensor_metadata': meta, 'metadata': {'path': 'tiny.json'}, 'step': 0, 'block': 0}, shapes


class ByteAndLineageControls(unittest.TestCase):
    def test_exit_boolean_zero_live_or_wrong_RUN_never_passes(self):
        proof = {'protocol': {'sha256': 'a'*64}, 'RUN': {'sha256': 'b'*64}}
        actual = {'id': 'test', 'protocol_sha256': 'a'*64, 'run_sha256': 'b'*64, 'process_exited': True,
                  'child_returncode': 0, 'timeout': False, 'automatic_retry': False}
        m.exited(actual, proof, 'test')
        for changes in ({'child_returncode': False}, {'process_exited': False}, {'timeout': True},
                        {'automatic_retry': True}, {'run_sha256': 'c'*64}):
            with self.assertRaises(ValueError):
                m.exited({**actual, **changes}, proof, 'test')

    def test_single_stream_nine_tensors_and_coordinates_reject_corruption(self):
        raw, entry, shapes = tiny_packet()
        value = m.decode_packet(raw, entry, m.core.rows(), np, shapes)
        self.assertEqual(value['rows'], m.core.rows())
        self.assertEqual(value['h'].shape, (168, 2))
        with self.assertRaises(ValueError):
            m.decode_packet(raw[:-1]+b'x', entry, m.core.rows(), np, shapes)
        with self.assertRaises(ValueError):
            m.decode_packet(raw, entry, list(reversed(m.core.rows())), np, shapes)
        bad = copy.deepcopy(entry)
        bad['tensor_metadata']['Y']['raw_tensor_sha256'] = 'f'*64
        with self.assertRaises(ValueError):
            m.decode_packet(raw, bad, m.core.rows(), np, shapes)

    def test_stream_returns_actual_stat_and_refuses_wrong_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'tiny.json').write_bytes(b'{"synthetic":true}')
            item = m.binding(root/'tiny.json', root)
            report = {'actual_packet_bytes_read': 0}
            raw, info = m.stream_payload(root, item, report)
            self.assertEqual(raw, b'{"synthetic":true}')
            self.assertEqual(info['st_size'], item['bytes'])
            self.assertEqual(report['actual_packet_bytes_read'], item['bytes'])
            (root/'tiny.json').write_bytes(b'{"synthetic":fals}')
            with self.assertRaises(ValueError):
                m.stream_payload(root, item, report)

    def test_manifest_uses_same_validated_buffer_no_second_reader(self):
        raw, entry, shapes = tiny_packet()
        m.decode_packet(raw, entry, m.core.rows(), np, shapes)
        case = {'source_id': 'synthetic', 'native_source_ordinal': 9999, 'cohort_index': 0}
        result = m.audited_manifest_entry(raw, case, entry, {'st_size': len(raw)})
        self.assertTrue(result['actual_full_file_SHA_and_nine_tensor_SHA_passed'])
        self.assertEqual(result['actual_safetensors_header']['rows']['shape'], [168])
        self.assertEqual(result['verified_tensor_raw_metadata'], entry['tensor_metadata'])

    def test_metadata_cache_never_accepts_payload_or_conflicting_identity(self):
        seen = []
        data = b'{"test":true}'
        item = {'path': 'proof.json', 'bytes': len(data), 'sha256': m.digest(data)}
        metadata = m.Metadata(Path('/synthetic'), lambda root, value: seen.append(value['path']) or data)
        metadata.json(item)
        metadata.json(item)
        self.assertEqual(seen, ['proof.json'])
        with self.assertRaises(ValueError):
            metadata.raw({**item, 'sha256': 'f'*64})
        with self.assertRaises(ValueError):
            metadata.raw({**item, 'path': 'operand.safetensors'})

    def test_failure_keeps_predebited_call_and_forbids_excess_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            out = m.OutputBudget(Path(directory), 4 << 20)
            report = {}
            meter = m.Meter(report, out, {'toy_call': 1})
            def fail():
                raise RuntimeError('synthetic failure after debit')
            with self.assertRaises(RuntimeError):
                meter.work({'toy_call': 1}, fail)
            self.assertEqual(report['operation_ledger']['predebited'], {'toy_call': 1})
            self.assertEqual(report['operation_ledger']['observed'], {'toy_call': 0})
            with self.assertRaises(ValueError):
                meter.work({'toy_call': 1}, lambda: None)
            out.close()


if __name__ == '__main__':
    unittest.main()
