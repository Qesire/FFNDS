"""Validate real prior Dense admission; other methods require another reviewed verifier."""
import hashlib,json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify(record,scope):
    if record['arm']!='Dense':
        raise NotImplementedError('actual non-Dense producer receipt verifier not yet bound')
    contract_path=Path(scope['paired_contract_path'])
    if sha(contract_path)!=scope['paired_contract_sha256']:raise ValueError('paired contract identity mismatch')
    contract=json.loads(contract_path.read_text(encoding='utf8'))
    admission=Path(scope['dense_admission_path']);receipt=Path(scope['dense_admission_receipt_path'])
    digest=sha(admission)
    if digest!=scope['dense_admission_sha256'] or digest!=contract['dev_cohort_admission_sha256'] or sha(receipt)!=scope['dense_admission_receipt_sha256']:
        raise ValueError('Dense admission or audit receipt identity mismatch')
    audit_receipt=json.loads(receipt.read_text(encoding='utf8'))
    if audit_receipt['status']!='READ_ONLY_AUDIT_COMPLETE' or audit_receipt['exit_code']!=0 or audit_receipt['output_sha256']!=digest:
        raise ValueError('Dense audit receipt incomplete')
    manifest=json.loads(admission.read_text(encoding='utf8'))
    rows=[x for x in manifest['slots'] if x['source']==record['source']]
    if len(rows)!=1 or not rows[0]['quality_video_eligible']:
        raise ValueError('Dense source not admitted for quality')
    row=rows[0]; expected={**row['input_identity'],'initial_noise':row['initial_noise'],'scheduler':row['scheduler']}
    if (record['seed']!=row['seed'] or record['video_sha256']!=row['video']['sha256'] or
        Path(record['video_path']).resolve()!=Path(row['video']['path']).resolve() or
        record['image_sha256']!=row['input_identity']['image_sha256'] or
        record['prompt_sha256']!=row['input_identity']['prompt_sha256'] or
        record['source_cluster_id']!=row['input_identity']['source_cluster_id'] or
        record['input_identity']!=expected):
        raise ValueError('Dense record differs from independently checked admission')
    if contract['source_clusters'][record['source']]!=record['source_cluster_id']:
        raise ValueError('source cluster differs from fixed paired contract')
    return {'status':'PASS','proof_sha256':digest,'main_slot_complete':row['main_slot_complete']}
