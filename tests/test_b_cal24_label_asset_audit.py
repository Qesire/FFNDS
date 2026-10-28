"""Tiny pure-stdlib controls for four-part asset audit; no real packs."""
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import b_cal24_label_asset_audit as m


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def fixture(root):
    repo, hf = root / 'repo', root / 'hf'
    repo.mkdir(); hf.mkdir()
    names = ['tools/e0_b_cal24_terminal_labels.py', 'tools/b_cal24_label_contract.py'] + [f'files/{i}' for i in range(103)]
    sources = []
    for name in names:
        p = repo / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(name.encode())
        sources.append(m.binding(p, repo))
    limits = {'count': 1}
    plan = {'id': m.producer.ID, 'repo_sources': sources, 'output_HF_relative': 'out',
            'limits_per_part': limits, 'source_exposure': 'exposed', 'source_split': {}, 'head_assets': {}}
    protocol = repo / 'protocol.json'; write(protocol, plan); protocol_sha = m.digest(protocol)
    cases = [{'case_id': f'cal24_{i:02d}_step{s}', 'part': i//6, 'source_index': i,
              'step': s, 'expert': m.producer.EXPERTS[0 if s < 2 else 1]} for i in range(24) for s in range(4)]
    counter = {'linear_forward_calls': 9600, 'activation_qdq_calls': 9600, 'weight_transfers': 800, 'cache_misses': 0}
    for part in range(4):
        folder = hf / 'out' / f'part{part}'; snapshot = folder / 'EXECUTED_SOURCES'
        for entry in sources:
            p = snapshot / entry['path']; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((repo / entry['path']).read_bytes())
        (snapshot / 'protocol.json').write_bytes(protocol.read_bytes())
        heads = []
        for expert in m.producer.EXPERTS:
            p = folder / expert / 'head'; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(expert.encode())
            heads.append({'expert':expert, **m.binding(p, folder), 'tensors':{}})
        source_rows = []
        for i in range(part*6, part*6+6):
            source_rows.append({'source_index':i, 'seed':42+i, 'status':'COMPLETE_FOUR_CONTINUOUS_STEPS_CAL24_BYTES_ALIGNED',
                                'cases':[{'case':c} for c in cases if c['source_index']==i]})
        run = {'id':plan['id'], 'part':part, 'protocol_sha256':protocol_sha,
               'status':'COMPLETE_CAL24_TERMINAL_LABEL_PART_POST_AND_OPERAND_BYTES_MATCH',
               'predebited':limits, 'observed':limits, 'actual_native_counters':counter,
               'actual_native_counters_including_failed_work':counter, 'original_head_assets':heads, 'sources':source_rows}
        write(folder/'RUN.json',run)
        receipt = {'id':plan['id'],'protocol_sha256':protocol_sha,'child_returncode':0,'process_exited':True,
                   'run_sha256':m.digest(folder/'RUN.json'),'timeout':False,'automatic_retry':False,'error_type':None}
        write(folder/'EXIT_RECEIPT.json',receipt)
    return repo, hf, protocol, cases


def run_audit(repo, hf, protocol, cases):
    with patch.object(m.producer,'validate_plan'), patch.object(m.contract,'build_cases',return_value=cases), \
         patch.object(m,'verify_head',side_effect=lambda path, asset, expert, plan: {'head':expert}), \
         patch.object(m,'audit_case',side_effect=lambda row, expected, *args: {'case':expected}):
        return m.audit(protocol,hf,repo_root=repo)


class AuditTests(unittest.TestCase):
    def test_four_complete_parts_and_missing_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo,hf,p,cases=fixture(Path(tmp))
            result=run_audit(repo,hf,p,cases)
            self.assertEqual(result['schema'],m.SCHEMA)
            self.assertEqual(result['totals']['cases'],96)
            (hf/'out/part3/EXIT_RECEIPT.json').unlink()
            with self.assertRaises((ValueError,FileNotFoundError)):run_audit(repo,hf,p,cases)

    def test_wrong_exit_run_hash_snapshot_and_duplicate_case_reject(self):
        for kind in ('exit','run_sha','snapshot','duplicate'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                repo,hf,p,cases=fixture(Path(tmp)); folder=hf/'out/part0'
                if kind in ('exit','run_sha'):
                    path=folder/'EXIT_RECEIPT.json'; r=json.loads(path.read_text())
                    r['child_returncode' if kind=='exit' else 'run_sha256']=1 if kind=='exit' else 'wrong';write(path,r)
                elif kind=='snapshot': (folder/'EXECUTED_SOURCES/files/4').write_bytes(b'bad')
                else:
                    path=folder/'RUN.json';r=json.loads(path.read_text());r['sources'][0]['cases'][1]=copy.deepcopy(r['sources'][0]['cases'][0]);write(path,r)
                    ep=folder/'EXIT_RECEIPT.json';e=json.loads(ep.read_text());e['run_sha256']=m.digest(path);write(ep,e)
                with self.assertRaises(ValueError):run_audit(repo,hf,p,cases)

    def test_head_payload_byte_hash_and_header_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'head';raw=b'\x01\x02\x03\x04\x05\x06';
            header={'head_weight_runtime_KN':{'dtype':'BF16','shape':[1,1],'data_offsets':[0,2]},
                    'head_bias':{'dtype':'BF16','shape':[1],'data_offsets':[2,4]},
                    'head_modulation':{'dtype':'BF16','shape':[1],'data_offsets':[4,6]}}
            h=json.dumps(header).encode();p.write_bytes(struct.pack('<Q',len(h))+h+raw)
            import hashlib
            sha=lambda x:hashlib.sha256(x).hexdigest()
            asset={'tensors':{k:{'shape':v['shape'],'dtype':'torch.bfloat16','raw_tensor_sha256':sha(raw[slice(*v['data_offsets'])])} for k,v in header.items()}}
            plan={'head_assets':{'e':{'weight_raw_sha256':sha(raw[:2]),'bias_raw_sha256':sha(raw[2:4]),'modulation_raw_sha256':sha(raw[4:6])}}}
            with patch.object(m,'HEAD_SHAPES',{'head_weight_runtime_KN':[1,1],'head_bias':[1],'head_modulation':[1]}):
                m.verify_head(p,asset,'e',plan)
                changed = b'\x09\x08' + raw[2:]
                p.write_bytes(struct.pack('<Q',len(h))+h+changed)
                altered = copy.deepcopy(asset)
                altered['tensors']['head_weight_runtime_KN']['raw_tensor_sha256'] = sha(changed[:2])
                with self.assertRaises(ValueError): m.verify_head(p,altered,'e',plan)
                p.write_bytes(struct.pack('<Q',len(h))+h+raw[:-1]+b'\x00')
                with self.assertRaises(ValueError):m.verify_head(p,asset,'e',plan)

    def test_nonsquare_kn_transposes_to_original_nk_bytes(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'head'; weight=b''.join(bytes((n,0)) for n in (1,2,3,4,5,6)); bias=b'\x08\x00'; modulation=b'\x09\x00'
            payload=weight+bias+modulation
            header={'head_weight_runtime_KN':{'dtype':'BF16','shape':[2,3],'data_offsets':[0,12]},
                    'head_bias':{'dtype':'BF16','shape':[1],'data_offsets':[12,14]},
                    'head_modulation':{'dtype':'BF16','shape':[1],'data_offsets':[14,16]}}
            h=json.dumps(header).encode();p.write_bytes(struct.pack('<Q',len(h))+h+payload)
            sha=lambda v:hashlib.sha256(v).hexdigest()
            asset={'tensors':{k:{'shape':v['shape'],'dtype':'torch.bfloat16','raw_tensor_sha256':sha(payload[slice(*v['data_offsets'])])} for k,v in header.items()}}
            original=b''.join(bytes((n,0)) for n in (1,4,2,5,3,6))
            plan={'head_assets':{'e':{'weight_raw_sha256':sha(original),'bias_raw_sha256':sha(bias),'modulation_raw_sha256':sha(modulation)}}}
            with patch.object(m,'HEAD_SHAPES',{'head_weight_runtime_KN':[2,3],'head_bias':[1],'head_modulation':[1]}):
                m.verify_head(p,asset,'e',plan)
                plan['head_assets']['e']['weight_raw_sha256']=sha(weight)
                with self.assertRaises(ValueError):m.verify_head(p,asset,'e',plan)


if __name__ == '__main__': unittest.main()
