"""Bounded registered execution of the upstream seven-source screenshot design."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import s2_consecutive_video_visual_review as visual

ID = 'E0_S2_CONSECUTIVE_REAL_VIDEO_SCREENSHOTS_CPU_20261029'
MANIFEST = 'config/s2_consecutive_video_visual_review_20261029_draft.json'
OUTPUT = 'ffnds-derived/s2_consecutive_visual_review/attempt1/worker/results'
RESOURCES = {'GPU': 0, 'CPU_threads': 2, 'wall_seconds': 300, 'outer_seconds': 360,
             'new_HF_bytes': 512 << 20, 'max_RSS_bytes': 4 << 30, 'automatic_retry': False}
LIMITS = {'unique_videos': 21, 'full_video_decodes': 21, 'decoded_frames': 1701,
          'selected_frame_positions': 378, 'RGB_sheets': 14, 'difference_sheets': 14,
          'model_calls': 0, 'GPU_calls': 0, 'new_video_generations': 0}


def binding(path, root):
    return {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
            'sha256': visual.file_sha256(path)}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def check_generation(source, arm, gen):
    visual.require(all(gen.get(k) == v for k, v in {
        'mode': arm, 'numeric_path': 'BF16_ORIGINAL', 'native_source_ordinal': source['source'],
        'source_id': source['source_id'], 'image_sha256': source['image_sha256'],
        'prompt_sha256': source['prompt_sha256'], 'seed': 42,
        'status': 'PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE',
        'all160_FFN_covered': True, 'completed_native_flow_steps': 4,
        'scheduler_generator_reset': True}.items()), 'Exact actual generation identity differs')


def draft():
    paths = (MANIFEST, 'tools/s2_consecutive_video_visual_review.py',
             'tools/e0_s2_consecutive_visual_cpu.py', 'tests/test_e0_s2_consecutive_visual_cpu.py')
    return {'id': ID, 'status': 'DRAFT', 'resources': RESOURCES, 'limits': LIMITS,
            'output_HF_relative': OUTPUT, 'automatic_retry': False,
            'repo_sources': [binding(ROOT/p, ROOT) for p in paths]}


def run(args):
    plan = json.loads(args.protocol.read_text()); digest = visual.file_sha256(args.protocol)
    active = json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text())
    scopes = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s['id'] == ID]
    visual.require(plan['id'] == ID and plan['status'] == 'FROZEN_REGISTERED'
        and plan['resources'] == RESOURCES and plan['limits'] == LIMITS
        and plan['output_HF_relative'] == OUTPUT and plan['automatic_retry'] is False
        and len(scopes) == 1 and scopes[0]['protocol_sha256'] == digest
        and scopes[0]['execution_status'] in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
        and scopes[0]['resources'] == RESOURCES and scopes[0]['limits'] == LIMITS
        and scopes[0]['max_gpu_cards'] == 0
        and active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and len(os.sched_getaffinity(0)) <= 2,
        'Exact finite hidden-CUDA CPU scope required')
    visual.require(plan['repo_sources'] == draft()['repo_sources'], 'Frozen screenshot code changed')
    manifest = json.loads((ROOT/MANIFEST).read_text())
    for source in manifest['sources']:
        for arm in visual.ARMS:
            b = source['arms'][arm]['generation_receipt']; p = visual.contained(args.hf_root, b['path'])
            visual.require(binding(p, args.hf_root) == b, 'Frozen generation receipt differs')
            check_generation(source, arm, json.loads(p.read_text()))
    out = args.hf_root/OUTPUT
    visual.require(not out.exists(), 'No repeated extraction allowed')
    started = time.monotonic()
    # Largest possible sheet is below 8 MiB; 28 such sheets plus metadata fit the total cap.
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 << 20, 8 << 20))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Screenshot CPU wall exceeded')))
    signal.alarm(RESOURCES['wall_seconds'])
    try:
        visual.run(SimpleNamespace(hf_root=args.hf_root, manifest=ROOT/MANIFEST, out=out, only_source=None))
        report = json.loads((out/'RUN.json').read_text())
        visual.require(report['status'] == 'COMPLETE_REAL_VIDEO_CONTACT_SHEETS_NOT_HUMAN_ACCEPTANCE'
            and len(report['sources']) == 7, 'Incomplete extraction')
        files = [p for p in out.rglob('*') if p.is_file()]
        visual.require(sum(p.stat().st_size for p in files) < RESOURCES['new_HF_bytes']//2
            and len([p for p in files if p.suffix == '.jpg']) == 28, 'Screenshot output budget/coverage differs')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        visual.require(rss <= RESOURCES['max_RSS_bytes'], 'Screenshot RSS exceeded')
        report.update(id=ID, protocol_sha256=digest, actual_limits=LIMITS,
                      max_RSS_bytes=rss, wrapper_wall_seconds=time.monotonic()-started,
                      visual_review='PENDING_SEPARATE_ACTUAL_INSPECTION', automatic_retry=False)
        write(out/'RUN.json', report)
    finally:
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('draft', 'run')); parser.add_argument('--protocol', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    args = parser.parse_args()
    if args.action == 'draft':
        visual.require(args.out is not None and not args.out.exists(), 'Fresh draft required'); write(args.out, draft())
    else: run(args)


if __name__ == '__main__': main()
