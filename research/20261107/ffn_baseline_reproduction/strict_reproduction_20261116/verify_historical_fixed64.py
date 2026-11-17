"""Read-only audit of 9 historical DEV fixed64 input pairs; JSON stdout only."""
import argparse,hashlib,json
from pathlib import Path
NINE=('vbi_1040','vbi_0886','vbi_0947','vbi_1109','vbi_1084','vbi_0950','vbi_0176','vbi_0392','vbi_0056')
SITES=(('high',4),('high',19),('high',34),('low',4),('low',19),('low',34))
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def audit(manifest, expected_sha, bindings, bindings_sha):
    if sha(manifest)!=expected_sha or sha(bindings)!=bindings_sha: raise ValueError('manifest identity mismatch')
    data=json.loads(Path(manifest).read_text(encoding='utf8')); bind=json.loads(Path(bindings).read_text(encoding='utf8'))
    if data['selected_k']!=5120 or data['rows_are_panel_first64'] or len(data['fixed64_token_ids'])!=64:
        raise ValueError('fixed64 algorithm identity mismatch')
    import struct
    frozen_ids=bind['p64']['fixed64_token_ids']
    if data['fixed64_token_ids'] != frozen_ids or hashlib.sha256(struct.pack('<64q',*frozen_ids)).hexdigest() != data['fixed64_token_id_sha256_le_int64']:
        raise ValueError('fixed64 identity differs from frozen B bindings')
    results=[]
    for source in NINE:
        src=data['sources'][source]
        if src['fixed64_token_id_sha256_le_int64'] != data['fixed64_token_id_sha256_le_int64']:
            raise ValueError('source fixed64 identity mismatch '+source)
        if src['panel_token_id_sha256_le_int64']!=bind['panel']['per_source'][source]['token_id_sha256_le_int64']:
            raise ValueError('source panel identity mismatch '+source)
        for expert,layer in SITES:
            rec=src['sites'][f'{expert}{layer:02d}']; same=rec['same_state']
            if rec['chip_cell_unblocked'] is not True or set(same)!={'anchor_panel_u_bit_match','hit_panel_u_bit_match','anchor_gate_bit_match','hit_gate_bit_match'} or not all(same.values()):
                raise ValueError('same-state receipt incomplete')
            for role,step in zip(('anchor','hit'),(0,1) if expert=='high' else (2,3)):
                item=rec[role]
                if item['step']!=step or item['layer']!=layer or item['U_fixed64_shape']!=[64,5120] or item['U_fixed64_dtype']!='bfloat16':
                    raise ValueError('fixed64 endpoint metadata mismatch')
                path=Path(item['path']); exists=path.is_file()
                matches=exists and path.stat().st_size==item['bytes'] and sha(path)==item['file_sha256']
                results.append({'source':source,'expert':expert,'layer':layer,'role':role,'step':step,
                                'path':str(path),'present':exists,'size_and_sha_match':matches})
    return {'status':'PASS' if len(results)==108 and all(x['size_and_sha_match'] for x in results) else 'MISSING_OR_MISMATCH',
            'expected_files':108,'matching_files':sum(x['size_and_sha_match'] for x in results),'files':results}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('manifest_sha256');p.add_argument('bindings');p.add_argument('bindings_sha256');a=p.parse_args()
    result=audit(a.manifest,a.manifest_sha256,a.bindings,a.bindings_sha256)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False));raise SystemExit(0 if result['status']=='PASS' else 1)


