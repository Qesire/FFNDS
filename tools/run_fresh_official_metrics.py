"""Exactly45 official metric calls on the frozen23-source generated cohort."""
import argparse
import gc
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from fresh_quality_contract import ARMS, Budget, CORE, METHOD, ROOT, atomic, read, require_registration, sha


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if hasattr(value, 'item'):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError('Nonfinite official evaluator output')
    return value


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    p = require_registration('METRICS')
    # The model/evaluator phase leaf is separately frozen before this entry.
    current = read(ROOT / 'research/CURRENT_DECISION.json')['registered_fresh_quality']
    leaf = current.get('metric_phase')
    if not leaf or sha(ROOT / leaf['protocol']) != leaf['protocol_sha256']:
        raise RuntimeError('No immutable metric-phase asset binding; no torch/evaluator imports performed')
    metric_protocol = read(ROOT / leaf['protocol'])
    assets_path = ROOT / '.runtime/R17_metric_assets.json'
    if sha(assets_path) != metric_protocol['asset_manifest_sha256']:
        raise RuntimeError('Official metric asset manifest differs')
    assets = read(assets_path)
    hf = Path(os.environ['HF_HOME'])
    root = hf / 'derived/ffnds/R17E0_fresh_quality'
    generation_path = root / 'GENERATIONS.json'
    generations = read(generation_path)
    if generations['status'] != 'COMPLETE_145_GENERATIONS_AWAITING_FRESH_OFFICIAL_METRICS':
        raise RuntimeError('Require the complete fixed145generation plan first')
    output = root / 'metrics'
    output.mkdir(exist_ok=False)
    from offline_i2v_metric_loaders import verify_source_view
    verify_source_view(assets, hf)
    source_view = hf / assets['HF_relative_root'] / 'installed_source_view'
    sys.path.insert(0, str(source_view))
    os.environ.update(VBENCH_CACHE_DIR=str(hf / assets['HF_relative_root']), HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    import torch
    from offline_i2v_metric_loaders import offline_loaders, submodules, install_source_supplement
    from assigned_gpu_resources import AssignedGPUResources
    from worker_identity import require_worker
    require_worker()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('One locked GPU required for official quality evaluation')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    supplement_path = ROOT / 'research/20261023/R17_metric_source_supplement.json'
    if sha(supplement_path) != metric_protocol['source_supplement_sha256']:
        raise RuntimeError('Pinned original wheel source supplement changed')
    install_source_supplement(read(supplement_path), hf)
    sources = read(ROOT / '.runtime/inputs/r17_fresh23/canonical.json')
    requests = {}
    for item in generations['completed']:
        if item['job']['phase'] != 'quality':
            continue
        request_path = hf / item['request_HF_relative']
        if sha(request_path) != item['request_sha256']:
            raise RuntimeError('Actual quality-generation record changed')
        request = read(request_path)
        job = request['job']
        if sha(hf / request['video_HF_relative']) != request['video_sha256']:
            raise RuntimeError('Actual quality video changed')
        requests[job['source'], job['arm']] = request
    if len(requests) != 115:
        raise RuntimeError('Require115distinct fresh-source/arm videos; no filename copies as new samples')
    work = output / 'work'
    images = work / 'vbench2_beta_i2v/data/crop/16-9'
    images.mkdir(parents=True)
    for source in sources:
        if sha(source['image_path']) != source['input_provenance']['image_sha256']:
            raise RuntimeError('Official condition image changed')
        (images / source['image_name']).symlink_to(Path(source['image_path']).resolve())
    budget = Budget(root, existing=True)
    report = {'id': METHOD, 'status': 'RUNNING_FIXED45_OFFICIAL_METRIC_CALLS',
              'generation_receipt_sha256': sha(generation_path),
              'metric_protocol_sha256': sha(ROOT / leaf['protocol']), 'completed': [],
              'unique_fresh_sources': 23, 'parameter_selection': False, 'population_quality_or_Pareto_acceptance': False}
    atomic(output / 'RUN.json', report)
    monitor = AssignedGPUResources()
    before_cwd = Path.cwd()
    def check_resources():
        monitor.check()
        used = sum(path.stat().st_size for path in root.rglob('*') if path.is_file() and not path.is_symlink())
        if (used > p['resources']['new_output_bytes_max'] or
                shutil.disk_usage(root).free < p['resources']['free_disk_reserve_bytes'] or
                torch.cuda.max_memory_allocated() > p['resources']['peak_allocated_bytes_max']):
            raise RuntimeError('Official metric output, disk or allocated-GPU cap exceeded')
        return {'HF_experiment_output_bytes': used, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}

    try:
        monitor.start()
        torch.cuda.reset_peak_memory_stats()
        os.chdir(work)
        for dimension in CORE:
            expected_sources = [s for s in sources if dimension in s['core_dimensions']]
            module = importlib.import_module(('vbench2_beta_i2v.' if dimension in ('i2v_subject', 'i2v_background', 'camera_motion') else 'vbench.') + dimension)
            function = getattr(module, f'compute_{dimension}')
            for arm in ARMS:
                check_resources()
                full_info = []
                folder = output / dimension / arm
                folder.mkdir(parents=True)
                for source in expected_sources:
                    request = requests[source['index'], arm]
                    # Camera scoring reads the instruction from the filename.
                    suffix = (' ' + source['camera_instruction']) if dimension == 'camera_motion' else ''
                    target = folder / f'source{source["index"]:02d}{suffix}.mp4'
                    target.symlink_to((hf / request['video_HF_relative']).resolve())
                    full_info.append({'prompt_en': source['prompt'], 'dimension': source['official_dimensions'],
                                      'image_name': source['image_name'], 'image_type': source['image_type'],
                                      'video_list': [str(target)]})
                info = folder / 'FULL_INFO.json'
                atomic(info, full_info)
                name = dimension + '_' + arm
                with budget.call(name, 'metric_evaluator_calls'), offline_loaders(assets, hf, torch.device('cuda:0')) as verify:
                    names = {
                        'i2v_subject': ['DINO_base'], 'subject_consistency': ['DINO_base'],
                        'background_consistency': ['CLIP_B32'], 'camera_motion': ['CoTracker2'],
                        'motion_smoothness': ['AMT_S'], 'dynamic_degree': ['raft_model:models/raft-things.pth'],
                        'aesthetic_quality': ['CLIP_ViT_L14', 'LAION_aesthetic_linear'],
                        'imaging_quality': ['MUSIQ_SPAQ'], 'i2v_background': []}
                    for checkpoint in names[dimension]:
                        verify(checkpoint)
                    began = time.monotonic()
                    with torch.inference_mode():
                        result = json_value(function(str(info), torch.device('cuda:0'), submodules(dimension, assets, hf)))
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - began
                score = result[0]
                if not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise RuntimeError('Invalid official metric aggregate')
                details = result[-1]
                expected_videos = {row['video_list'][0] for row in full_info}
                if (not isinstance(details, list) or len(details) != len(expected_sources) or
                        {row.get('video_path') for row in details} != expected_videos):
                    raise RuntimeError('Official evaluator did not measure every expected unique video')
                record = {'dimension': dimension, 'arm': arm, 'source_indices': [s['index'] for s in expected_sources],
                          'underlying_source_clusters': len(expected_sources), 'actual_unique_generated_videos': len(expected_sources),
                          'official_return': result, 'official_aggregate': score,
                          'full_info_sha256': sha(info), 'elapsed_seconds': elapsed,
                          'model_quality_definition_modified': False, 'model_cache_loader_adapted': True}
                atomic(folder / 'RESULT.json', record)
                report['completed'].append({'dimension': dimension, 'arm': arm,
                                            'result_HF_relative': str((folder / 'RESULT.json').relative_to(hf)),
                                            'sha256': sha(folder / 'RESULT.json'), 'score': score})
                atomic(output / 'RUN.json', report)
                gc.collect()
                torch.cuda.empty_cache()
                report['last_resource_check'] = check_resources()
                atomic(output / 'RUN.json', report)
                print(f'OfficialI2V {len(report["completed"])}/45 {dimension}/{arm}: {score}', flush=True)
        if len(report['completed']) != 45 or budget.state['used']['metric_evaluator_calls'] != 45:
            raise RuntimeError('Incomplete fixed official metric plan')
        report['status'] = 'COMPLETE_FIXED23_SOURCE_NINE_DIMENSION_FIVE_ARM_EVALUATION'
    except BaseException as error:
        report.update(status='FAILED_FIXED_OFFICIAL_SCOPE_DEBIT_PRESERVED', error_type=type(error).__name__)
        raise
    finally:
        os.chdir(before_cwd)
        monitor.close()
        report['resources'] = monitor.report()
        report['budget'] = budget.state
        if monitor.violations:
            report['status'] = 'FAILED_ASSIGNED_GPU_ISOLATION_DEBIT_PRESERVED'
        atomic(output / 'RUN.json', report)
        budget.close()
    if report['status'] != 'COMPLETE_FIXED23_SOURCE_NINE_DIMENSION_FIVE_ARM_EVALUATION':
        raise RuntimeError('Official evaluation did not complete all fixed controls')
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
