"""Build a source-specific VBench index for explicit multi-category confirmation using historical metric dimensions."""
DIMS = ['aesthetic_quality', 'imaging_quality', 'motion_smoothness']
SOURCE_DIMS = ['i2v_subject', 'i2v_background']


def build_full_info(samples, historical):
    if len(samples) != 20 or len({v['source_group'] for v in samples}) != 20:
        raise ValueError('Exactly20 distinct fixed sources required')
    templates = {}
    for dimension in SOURCE_DIMS:
        references = [v for v in historical if dimension in v['dimension']]
        if len(references) != 10 or any(v['dimension'] != references[0]['dimension']
                or v['image_type'] != references[0]['image_type'] for v in references):
            raise ValueError('Historical source-stratum dimension/type rule is inconsistent')
        if not set(DIMS).issubset(references[0]['dimension']):
            raise ValueError('Each historical stratum must cover all three target metrics')
        templates[dimension] = references[0]
    result = []
    for sample in samples:
        dimension = sample['dimension']
        if dimension not in templates:
            raise ValueError('Camera/unknown source dimension is excluded')
        template = templates[dimension]
        kind = sample.get('image_type', template['image_type'])
        if not isinstance(kind,str) or not kind:
            raise ValueError('An actual registered source type is required')
        result.append({'prompt_en': sample['prompt'], 'dimension': list(template['dimension']),
                       'image_type': kind, 'image_name': sample['image_name']})
    if any(sum(v['dimension'] == dim for v in samples) != 10 for dim in SOURCE_DIMS):
        raise ValueError('Exactly subject10/background10 required')
    if len({v['prompt_en'] for v in result}) != 20:
        raise ValueError('Distinct prompt filenames required for all20 sources')
    return result


def validate_video_name_coverage(full_info, video_names):
    names = set(video_names)
    if len(names) != 100 or len(full_info) != 20:
        raise ValueError('Exactly100 filenames for20 independent sources required')
    counts = {}
    for dimension in DIMS:
        expected = [f'{v["prompt_en"]}-{copy}.mp4' for v in full_info
                    if dimension in v['dimension'] for copy in range(5)]
        if len(expected) != 100 or len(set(expected)) != 100 or set(expected) != names:
            raise ValueError(f'Index does not match all100 video names for {dimension}')
        counts[dimension] = len(expected)
    return counts
