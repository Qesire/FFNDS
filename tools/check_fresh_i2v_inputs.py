"""Public stdlib verification of source-disjoint prospective I2V inputs."""
import hashlib
from pathlib import Path

from rollout_contract import ROOT, read, sha


def validate(audit, exclusion):
    errors = []
    rows = audit['inputs']
    if len(rows) != 23 or [r['index'] for r in rows] != list(range(23)):
        return ['Missing exact frozen23source identities']
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    for row in rows:
        if (row['input_provenance']['image_sha256'] in exclusion['prior_image_SHA'] or
                digest(row['image_name']) in exclusion['prior_image_name_SHA'] or
                digest(row['prompt']) in exclusion['prior_prompt_SHA'] or
                digest(row['source_group']) in exclusion['prior_source_group_SHA']):
            errors.append('Fresh source overlaps historical bytes/name/prompt/group')
        if row['seed'] != 42 or not set(row['core_dimensions']).issubset(row['official_dimensions']):
            errors.append('Seed or original per-dimension source assignment changed')
    if len({r['input_provenance']['image_sha256'] for r in rows}) != 23:
        errors.append('Duplicate image bytes cannot be independent sources')
    counts = {d: sum(d in row['core_dimensions'] for row in rows)
              for d in audit['per_core_dimension_source_count']}
    if counts != audit['per_core_dimension_source_count'] or counts != {
            'i2v_subject': 8, 'i2v_background': 8, 'camera_motion': 7,
            'subject_consistency': 8, 'background_consistency': 8, 'motion_smoothness': 8,
            'dynamic_degree': 8, 'aesthetic_quality': 16, 'imaging_quality': 16}:
        errors.append('Official source subsets cannot be replaced by an identical nine-column population')
    cameras = [r for r in rows if r['stratum'] == 'camera_motion']
    if len(cameras) != 7 or len({r['image_name'] for r in cameras}) != 7 or len({r['camera_instruction'] for r in cameras}) != 7:
        errors.append('Camera commands require seven distinct source images and directions')
    if audit['new_model_generations'] or audit['model_or_evaluator_execution'] or exclusion['new_metric_model_forwards']:
        errors.append('Data preparation cannot masquerade as measured video quality')
    return errors


def main():
    base = ROOT / 'research/20261023'
    audit_path = base / 'R17_fresh23_input_audit.json'
    exclusion = read(base / 'R17_fresh23_exclusion_manifest.json')
    errors = validate(read(audit_path), exclusion)
    if exclusion['input_audit_sha256'] != sha(audit_path):
        errors.append('Exclusion proof belongs to a different frozen selection')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Fresh23 image/name/prompt/group disjoint from98priorimageSHAs;officialnine source subsets consistent. No new model result.')


if __name__ == '__main__':
    main()
