"""Expose only hashed historical source identities for independent audit."""
import hashlib
import json
from pathlib import Path


def main():
    from freeze_fresh_i2v_inputs import ROOT, collect_exposure
    from rollout_contract import atomic, read, sha
    audit_path = ROOT / 'research/20261023/R17_fresh23_input_audit.json'
    audit = read(audit_path)
    target = ROOT / 'research/20261023/R17_fresh23_exclusion_manifest.json'
    if target.exists():
        raise RuntimeError('Preserve the first historical exclusion audit')
    seen = {k: set() for k in ('names', 'image_SHA', 'prompts', 'groups')}
    for record in audit['exposure_inventory']:
        path = ROOT / record['path']
        if sha(path) != record['sha256']:
            raise RuntimeError('Original source exposure inventory changed')
        collect_exposure(read(path), seen)
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    report = {'id': 'R17_FRESH23_HISTORICAL_EXCLUSION_PROOF',
              'status': 'PASS_EXACT_ORIGINAL_EXPOSURE_INVENTORY_REPLAY_NO_MODELS',
              'source_sha256': sha(__file__), 'input_audit_sha256': sha(audit_path),
              'prior_image_SHA': sorted(seen['image_SHA']),
              'prior_image_name_SHA': sorted(digest(value) for value in seen['names']),
              'prior_prompt_SHA': sorted(digest(value) for value in seen['prompts']),
              'prior_source_group_SHA': sorted(digest(value) for value in seen['groups']),
              'source_count_is_not_a_new_quality_sample': True,
              'new_model_generations': 0, 'new_metric_model_forwards': 0}
    for row in audit['inputs']:
        if (row['input_provenance']['image_sha256'] in seen['image_SHA'] or
                row['image_name'] in seen['names'] or row['prompt'] in seen['prompts'] or
                row['source_group'] in seen['groups']):
            raise RuntimeError('An allegedly fresh input overlaps historical exposure')
    atomic(target, report)
    print(json.dumps({'status': report['status'], 'excluded_image_SHA': len(seen['image_SHA']),
                      'new_independent_image_SHA': len(audit['inputs'])}))


if __name__ == '__main__':
    main()
