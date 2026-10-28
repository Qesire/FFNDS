"""Portable byte/accounting controls; no real model operands or CUDA."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('full80_driver', ROOT/'tools/e0_raw_bf16_full80_spectrum.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class DriverControls(unittest.TestCase):
    def test_import_is_stdlib_and_immutable_parent_namespaces(self):
        code = "import sys; sys.path.insert(0,'tools'); import e0_raw_bf16_full80_spectrum as m; assert not {'torch','numpy'} & set(sys.modules); assert m.infrastructure.parent_v3.parent_v2.RESOURCES['wall_seconds']==1200"
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_precall_multicost_failure_and_atomic_budget_rejection(self):
        report, events = {}, []
        meter = m.Meter(report, events.append)
        def fail():
            self.assertEqual(report['predebited']['raw_W2_tensor_reads'], 1)
            self.assertEqual(report['predebited']['raw_tensor_bytes'], 10)
            raise RuntimeError('actual call failed')
        with self.assertRaisesRegex(RuntimeError, 'actual call failed'):
            meter.work({'raw_W2_tensor_reads': 1, 'raw_tensor_bytes': 10}, fail)
        self.assertEqual(report['observed'], {})
        self.assertEqual(events, [{'kind': 'reserved_before_call', 'costs': {'raw_W2_tensor_reads': 1, 'raw_tensor_bytes': 10}}])
        saved = copy.deepcopy(report)
        with self.assertRaises(ValueError):
            meter.work({'raw_W2_tensor_reads': 80, 'raw_tensor_bytes': 1}, lambda: self.fail('must not invoke'))
        self.assertEqual(report, saved)

    def test_exact_registered_work_geometry(self):
        # Full fits:80 shared+160 step; anchorLOO:8*6 shared+8*2*6 step.
        self.assertEqual(m.LIMITS['FP64_eigh'], 80+160+8*6+8*2*6)
        self.assertEqual(m.LIMITS['FP64_eigvalsh_lower_bound'], 8)
        cases = 80*8*2+160*8+8*6*6*2+8*2*6*6
        self.assertEqual(m.LIMITS['BF16_factor_GEMM'], cases*2)
        self.assertEqual(m.LIMITS['source_case_metrics'], cases*3)
        self.assertEqual(m.LIMITS['BF16_postwrite_calls'], cases*3+8*160)
        self.assertEqual(m.LIMITS['raw_tensor_bytes'], 80*(5120*13824+5120)*2)
        self.assertEqual(m.LIMITS['BF16_export_files'], 80+160)
        self.assertEqual(m.LIMITS['factor_export_D2H_bytes'], 8731852800)
        self.assertEqual(m.LIMITS['neutral_operand_payload_bytes'], 10079436800)
        self.assertEqual(m.DESIGN['neutral_order'], 'ALL1280_DENSE_BETA1_BYTE_NEUTRALS_COMPLETE_BEFORE_ANY_FIT')
        self.assertEqual(m.ANCHORS, [(e,b) for e in (0,1) for b in (0,13,26,39)])

    def _packet(self, directory):
        shapes = {'V1': [2,4], 'h': [2,6], 'Y': [2,4], 'r': [2,4], 'u': [2,4],
            'gate': [4], 'shift': [4], 'scale': [4], 'rows': [2]}
        header, metadata, payload, cursor = {}, {}, bytearray(), 0
        for i, (name, shape) in enumerate(shapes.items()):
            n = 1
            for d in shape: n *= d
            body = struct.pack('<'+'q'*n, *range(n)) if name == 'rows' else bytes([i+1, 0])*n
            header[name] = {'shape': shape, 'dtype': 'I64' if name == 'rows' else 'BF16', 'data_offsets': [cursor, cursor+len(body)]}
            metadata[name] = {'shape': shape, 'dtype': 'int64' if name == 'rows' else 'bfloat16', 'raw_tensor_sha256': hashlib.sha256(body).hexdigest()}
            payload.extend(body); cursor += len(body)
        encoded = json.dumps(header).encode(); raw = struct.pack('<Q', len(encoded))+encoded+payload
        path = directory/'packet.safetensors'; path.write_bytes(raw)
        return {'payload': {'path': path.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}, 'tensor_metadata': metadata}, len(encoded)

    def test_partial_neutral_ranges_and_whole_payload_sha_gate(self):
        with tempfile.TemporaryDirectory() as temporary, patch.multiple(m, ROWS=2, D=4, H=6):
            root = Path(temporary); entry, header_bytes = self._packet(root); io = {}
            payloads, shapes = m.packet_payloads(root, entry, {'Y','r','u','gate'}, whole=False, io=io)
            self.assertEqual(set(payloads), {'Y','r','u','gate'})
            self.assertEqual(io['packet_or_neutral_actual_read_bytes'], 8+header_bytes+56)
            self.assertEqual(shapes['Y'], [2,4])
            # An unrequested h corruption is caught on the later mandatory whole-file pass.
            path = root/entry['payload']['path']; raw = bytearray(path.read_bytes()); raw[8+header_bytes+16] ^= 1; path.write_bytes(raw)
            m.packet_payloads(root, entry, {'Y','r','u','gate'}, whole=False)
            with self.assertRaisesRegex(ValueError, 'Actual packet SHA changed'):
                m.packet_payloads(root, entry, {'h','Y','r','u','gate'}, whole=True)
            # Corrupt requested Y fails already in the global neutral pass.
            entry, header_bytes = self._packet(root); raw = bytearray(path.read_bytes()); raw[8+header_bytes+40] ^= 1; path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'Neutral tensor rawSHA changed'):
                m.packet_payloads(root, entry, {'Y','r','u','gate'}, whole=False)

    def test_packet_coordinate_dtype_and_path_rejection(self):
        with tempfile.TemporaryDirectory() as temporary, patch.multiple(m, ROWS=2, D=4, H=6):
            root = Path(temporary); entry, _ = self._packet(root)
            bad = copy.deepcopy(entry); bad['tensor_metadata']['rows']['dtype'] = 'bfloat16'
            with self.assertRaisesRegex(ValueError, 'layout/dtype'):
                m.packet_payloads(root, bad, {'gate'}, whole=False)
            bad = copy.deepcopy(entry); bad['payload']['path'] = '../packet.safetensors'
            with self.assertRaisesRegex(ValueError, 'escaped HF'):
                m.packet_payloads(root, bad, {'gate'}, whole=False)

    def test_source_level_summary_held_train_is_excluded(self):
        keys = ('gated_Y_relative_SSE', 'post_u_relative_SSE', 'post_centered_relative_SSE',
            'std_ratio_mean','std_ratio_std','std_ratio_MSE','normalized_u_relative_SSE',
            'normalized_direct','normalized_radial','normalized_cross_signed')
        def case(source, step, role, value, fold=None):
            return dict(family='shared', fold=fold, role=role, native_source_ordinal=source,
                local_step=step, gain='one', **{key: value for key in keys})
        values = [case(256,t,'development_transfer',1.) for t in (0,1)]+[case(894,t,'development_transfer',9.) for t in (0,1)]
        values += [case(240,t,'held',2.,0) for t in (0,1)]+[case(288,t,'train',999.,0) for t in (0,1)]
        # Include the step family to maintain the registered complete-family geometry.
        step = [dict(c, family='step0') for c in values if c['local_step']==0]+[dict(c, family='step1') for c in values if c['local_step']==1]
        result = m.summarize([{'expert':'high','block':0,'fits':[{'family':'shared','cases':values}, {'family':'step0','cases':step[:len(step)//2]}, {'family':'step1','cases':step[len(step)//2:]}]}])
        groups = {(g['family'],g['role']):g for g in result['groups']}
        self.assertEqual(groups['shared80','development_transfer']['source_equal']['normalized_u_relative_SSE'], 5.)
        held = groups['shared80','anchor_LOO_held']; self.assertEqual(len(held['sources']),1)
        self.assertEqual(held['source_equal']['normalized_u_relative_SSE'],2.)

    def test_actual_audit_exit_source_roles_and_all_conditioning_positions(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(m, 'ROOT', Path(temporary)):
            root = Path(temporary)
            def write(name, value):
                path = root/name; path.write_text(json.dumps(value)); return m.native.binding(path, root)
            cases = [{'native_source_ordinal': ordinal, 'role': 'calibration' if ordinal in m.CAL else 'development',
                'source_state': 'teacher_input', 'image_sha256': str(i)*64,
                'packets': [{'step':s,'block':b} for s in range(4) for b in range(40)]}
                for i, ordinal in enumerate(m.ORDINALS)]
            index = {'schema':'ffnds.raw_bf16.full_depth.operands_index.v1',
                'totals':{'sources':8,'calibration':6,'development':2,'packets':1280,'native_positions':160},
                'cases':cases, 'conditioning_SHA_observations':[{'step':s,'block':b,'tensor':k,
                    'all_eight_same':True,'distinct_raw_sha256_count':1}
                    for s in range(4) for b in range(40) for k in ('gate','scale','shift')]}
            def proof(data, code=0):
                protocol = write('protocol.json', {'id':'AUDIT'})
                manifest = write('manifest.json', data)
                run = write('run.json', {'id':'AUDIT','protocol_sha256':protocol['sha256'],
                    'status':'COMPLETE_ASSET_BYTES_AUDITED','manifest':manifest})
                exited = write('exit.json', {'id':'AUDIT','protocol_sha256':protocol['sha256'],
                    'process_exited':True,'child_returncode':code,'timeout':False,'automatic_retry':False,'run_sha256':run['sha256']})
                return {'operand_audit':{'protocol':protocol,'RUN':run,'launch_exit':exited,'manifest':manifest}}
            self.assertEqual(set(m.load_index(proof(index), root)), set(m.ORDINALS))
            with self.assertRaisesRegex(ValueError, 'actual|exit0'):
                m.load_index(proof(index, code=False), root)
            bad = copy.deepcopy(index); bad['cases'][0]['role'] = 'development'
            with self.assertRaisesRegex(ValueError, 'source-role'):
                m.load_index(proof(bad), root)
            bad = copy.deepcopy(index); bad['conditioning_SHA_observations'][0] = copy.deepcopy(bad['conditioning_SHA_observations'][1])
            with self.assertRaisesRegex(ValueError, 'Source-static'):
                m.load_index(proof(bad), root)


if __name__ == '__main__':
    unittest.main()
