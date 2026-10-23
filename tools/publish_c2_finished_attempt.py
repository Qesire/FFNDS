"""Wait for one C2 attempt, prepare CPU media and publish only uncontested work.

Never launches a model, retries a failed attempt, or grants a visual PASS.
If main or prepared files move, preserve results and defer publication.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from c2_contract import METHOD, ROOT, atomic, read, sha
from publish_fresh_quality_batch import workflow
from worker_identity import ssh_arguments

DYNAMIC = ['AGENTS.md', 'README.md', 'research/CURRENT_DECISION.json',
           'research/20261021/method_triage.json', 'research/20261023/HANDOFF_C2N0.md',
           'research/candidate_rollout/registry/C2N0_norm_scaled_energy.json']
BASE = ROOT / 'research/20261023'
FOLDER = ROOT / 'runs/c2_execution'


def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def replace_c2_section(old, start_marker, end_marker, replacement):
    """Preserve independently added research sections around C2."""
    if old.count(start_marker) != 1 or old.count(end_marker) != 1:
        raise RuntimeError('C2 section boundaries moved; preserve research documentation')
    start, end = old.index(start_marker), old.index(end_marker)
    if end <= start:
        raise RuntimeError('C2 section order changed; preserve research documentation')
    return old[:start] + replacement.rstrip() + '\n\n' + old[end:]


def verify_concurrent(snapshot):
    if git('rev-parse', 'HEAD') != snapshot['expected_head'] or git('branch', '--show-current') != 'main':
        raise RuntimeError('Publication deferred: local main moved')
    if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode:
        raise RuntimeError('Publication deferred: another staged change exists')
    for path, expected in snapshot['static_sha256'].items():
        if sha(ROOT / path) != expected: raise RuntimeError('Publication deferred: prepared file changed: ' + path)
    for path in ['AGENTS.md', 'README.md', 'research/20261023/HANDOFF_C2N0.md']:
        if (ROOT / path).read_text() != snapshot['dynamic'][path]:
            raise RuntimeError('Publication deferred: concurrent documentation change')
    before = json.loads(snapshot['dynamic']['research/CURRENT_DECISION.json'])
    now = read(ROOT / 'research/CURRENT_DECISION.json')
    for key in ['execution_state', 'eligible_model_method_ids', 'latest_c2_native_smoke']:
        if key in now: before[key] = copy.deepcopy(now[key])
    for key in ['active', 'result', 'fixed_scope_finished']:
        if key in now['registered_c2_smoke']: before['registered_c2_smoke'][key] = now['registered_c2_smoke'][key]
    if before != now: raise RuntimeError('Publication deferred: concurrent decision edit')
    before = json.loads(snapshot['dynamic']['research/20261021/method_triage.json'])
    now = read(ROOT / 'research/20261021/method_triage.json')
    before['policy']['execution_state'] = now['policy']['execution_state']
    old = next(r for r in before['methods'] if r['id'] == METHOD); current = next(r for r in now['methods'] if r['id'] == METHOD)
    for key in ['status', 'model_execution_eligibility', 'result']:
        if key in current: old[key] = current[key]
    if before != now: raise RuntimeError('Publication deferred: concurrent triage edit')
    path = DYNAMIC[-1]; before = json.loads(snapshot['dynamic'][path]); now = read(ROOT / path)
    for key in ['status', 'execution_stage', 'actual_result']:
        if key in now: before[key] = now[key]
    if before != now: raise RuntimeError('Publication deferred: concurrent candidate edit')


def launch_failure(job):
    """A missing native receipt is unknown budget evidence, not invented zeros."""
    d = read(ROOT / 'research/CURRENT_DECISION.json'); t = read(ROOT / 'research/20261021/method_triage.json')
    r = d['registered_c2_smoke']
    summary = {'id': METHOD, 'status': 'STOPPED_C2_LAUNCH_WITHOUT_NATIVE_RECEIPT_NO_RETRY',
               'actual_budget': None, 'budget_receipt_missing': True, 'completed_videos': None,
               'actual_child_exit_code': job['child_exit_code'], 'quality_or_Pareto_acceptance': False,
               'automatic_followup': False, 'protocol_sha256': r['protocol_sha256']}
    path = BASE / 'C2N0_execution_summary.json'
    if path.exists(): raise RuntimeError('Preserve existing C2 summary')
    atomic(path, summary)
    r.update(active=False, result=str(path.relative_to(ROOT)), fixed_scope_finished=True)
    d['execution_state'] = r['preserved_model_closure']; d['eligible_model_method_ids'] = []; d['latest_c2_native_smoke'] = summary
    t['policy']['execution_state'] = d['execution_state']
    row = next(x for x in t['methods'] if x['id'] == METHOD)
    row.update(status=summary['status'], model_execution_eligibility='NOT_ELIGIBLE_FIXED_SCOPE_FINISHED', result=str(path.relative_to(ROOT)))
    atomic(ROOT / 'research/CURRENT_DECISION.json', d); atomic(ROOT / 'research/20261021/method_triage.json', t)
    path = ROOT / DYNAMIC[-1]; c = read(path); c.update(status='CLOSED', execution_stage=summary['status']); atomic(path, c)


def render_result():
    s = read(BASE / 'C2N0_execution_summary.json'); media_path = BASE / 'C2N0_video_review.json'
    complete = s['status'] == 'COMPLETE_C2_NATIVE_VIDEOS_PENDING_FULL_REVIEW'
    text = ('固定六条原生视频已完成；配对媒体等待完整人工审阅。' if complete else '固定原生尝试已停止，无自动重试。')
    if media_path.exists(): text += '全部6条视频逐帧解码81帧；新旧路由和逐帧RGB保真数值已保存。像素差异不是总体感知质量，单条耗时不是重复测速。'
    else: text += '尚无完整媒体审阅结果。'
    text += '见[实际执行](research/20261023/C2N0_execution_summary.json)和[当前交接](research/20261023/HANDOFF_C2N0.md)。'
    p = ROOT / 'README.md'
    p.write_text(replace_c2_section(p.read_text(), '## C2N0：', '## C1N0：',
                 '## C2N0：固定尝试已结束，按实际媒体继续审阅\n\n' + text))
    p = ROOT / 'AGENTS.md'
    p.write_text(replace_c2_section(p.read_text(), '# C2N0 ', '# C1N0 native diagnostic completed',
                 '# C2N0 fixed attempt finished; no automatic model followup\n\nRead [actual execution](research/20261023/C2N0_execution_summary.json) and [current handoff](research/20261023/HANDOFF_C2N0.md). '
                 'The registered attempt has completed or stopped. Model eligibility is empty; no retry/expansion. '
                 'Media decode and RGB diagnostics never grant human full-duration smoke PASS, official quality or repeated system speed. '
                 'Inspect actual paired videos and preserve all earlier closures.'))
    p = BASE / 'HANDOFF_C2N0.md'; lines = ['\n## 本次固定尝试的实际结束状态', '',
        '`' + s['status'] + '`。实际扣账：`' + json.dumps(s['actual_budget'], sort_keys=True) + '`。', '', text, '',
        '模型固定范围已经闭合，资格为空。' + (
        '完整配对预览和全部帧接触表留在HF缓存；人工完整播放结论仍待确认，没有自动质量通过。' if complete else
        '没有完成完整视频配对，不存在本轮完整视频人工审阅结论；读取实际扣账和失败记录。')]
    if media_path.exists():
        m = read(media_path); lines += ['', '| 方案 | pooled RGB MSE（0–255） |', '|---|---:|']
        for arm, v in m['RGB_fidelity_to_dense'].items(): lines.append(f'| {arm} | {v["pooled_RGB_MSE_0_255"]:.6f} |')
        lines += ['', '[完整逐帧/路由诊断](C2N0_video_review.json)。这是一个来源的轨迹保真，不能换成总体质量百分比。']
    p.write_text(p.read_text() + '\n'.join(lines) + '\n')
    d = read(ROOT / 'research/CURRENT_DECISION.json'); t = read(ROOT / 'research/20261021/method_triage.json')
    d['next_active_research_task'].update(status='PENDING_FULL_DURATION_VIDEO_REVIEW' if complete else 'FIXED_ATTEMPT_STOPPED_NO_AUTOMATIC_RETRY', model_execution_authority='NONE_FIXED_SCOPE_FINISHED')
    if media_path.exists(): d['latest_c2_native_smoke']['review_media'] = str(media_path.relative_to(ROOT))
    t['next_active_research_task'] = d['next_active_research_task']; t['latest_c2_native_smoke'] = d['latest_c2_native_smoke']
    t['policy']['execution_note'] = 'C2 finite attempt finished; read actual summary/media. No automatic retry or quality PASS. Earlier closures preserved.'
    atomic(ROOT / 'research/CURRENT_DECISION.json', d); atomic(ROOT / 'research/20261021/method_triage.json', t)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args(); snapshot = read(args.snapshot); result = {'status': 'WAITING_FOR_SINGLE_NATIVE_ATTEMPT', 'new_model_calls': 0}
    result_path = FOLDER / 'PUBLICATION_STATUS.json'; atomic(result_path, result)
    try:
        while read(FOLDER / 'JOB.json')['status'] == 'RUNNING': time.sleep(10)
        job = read(FOLDER / 'JOB.json'); verify_concurrent(snapshot)
        subprocess.run(['git', '-c', 'http.version=HTTP/1.1', 'fetch', 'https://github.com/Qesire/FFNDS.git', 'main'], cwd=ROOT, check=True)
        if git('rev-parse', 'FETCH_HEAD') != snapshot['expected_head']: raise RuntimeError('Publication deferred: remote main moved')
        # CPU wrapper imports no model/evaluator. Only completed model receipts are consumed.
        script = '''import os,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tools'))
from rollout_contract import read
root=Path(os.environ['HF_HOME'])/'derived/ffnds/C2N0_norm_scaled_smoke'
if (root/'RUN.json').exists():
 run=read(root/'RUN.json')
 if run['status']=='COMPLETE_C2_NATIVE_VIDEOS_PENDING_FULL_REVIEW':
  from review_c2_native_videos import main
  main()
 from export_c2_native_results import main
 main()
'''
        cmd = ['bash', str(ROOT / 'scripts/worker_cpu_exec.sh'), str(ROOT / '.venv/bin/python'), '-c', script]
        with (FOLDER / 'CPU_POSTPROCESS.log').open('x') as log:
            child = subprocess.run([*ssh_arguments(), shlex.join(cmd)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if child.returncode: raise RuntimeError('CPU media/export failed; preserve raw native output, no model retry')
        verify_concurrent(snapshot)
        if not (BASE / 'C2N0_execution_summary.json').exists(): launch_failure(job)
        render_result()
        rendered_sha = {p: sha(ROOT / p) for p in DYNAMIC}
        env = ROOT / 'runs/c1_execution/offline_checks_env/bin'
        os.environ['PATH'] = str(env) + ':' + os.environ['PATH']
        workflow(FOLDER / 'FINAL_OFFLINE_WORKFLOW.log')
        legacy = ROOT / 'research/20261023/metric_guard_after_R15.json'
        old = subprocess.check_output(['git', 'show', 'HEAD:' + str(legacy.relative_to(ROOT))], cwd=ROOT)
        def canonical(v):
            if isinstance(v, dict): return {k: canonical(x) for k, x in v.items()}
            if isinstance(v, list): return sorted((canonical(x) for x in v), key=lambda x: json.dumps(x, sort_keys=True))
            return v
        if canonical(read(legacy)) != canonical(json.loads(old)): raise RuntimeError('Unexpected legacy metric evidence change')
        legacy.write_bytes(old)
        for path, expected in {**snapshot['static_sha256'], **rendered_sha}.items():
            if sha(ROOT / path) != expected: raise RuntimeError('Publication deferred: file moved during validation: ' + path)
        paths = [os.fsdecode(p) for p in subprocess.check_output(['git', 'ls-files', '--modified', '--others', '--exclude-standard', '-z'], cwd=ROOT).split(b'\0') if p]
        allowed = set(snapshot['allowed_paths'])
        for suffix in ['execution_summary.json', 'native_results.json', 'native_results.json.redaction.json', 'budget.json', 'budget.json.redaction.json', 'video_review.json', 'video_review.json.redaction.json', 'completed_resources.json']:
            allowed.add('research/20261023/C2N0_' + suffix)
        if set(paths) - allowed: raise RuntimeError('Publication deferred: unrelated work appeared')
        if git('rev-parse', 'HEAD') != snapshot['expected_head']: raise RuntimeError('Publication deferred: local main moved')
        subprocess.run(['git', '-c', 'http.version=HTTP/1.1', 'fetch', 'https://github.com/Qesire/FFNDS.git', 'main'], cwd=ROOT, check=True)
        if git('rev-parse', 'FETCH_HEAD') != snapshot['expected_head']: raise RuntimeError('Publication deferred: remote main moved')
        if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode: raise RuntimeError('Concurrent staged work appeared')
        subprocess.run(['git', 'add', '--', *paths], cwd=ROOT, check=True)
        staged = {os.fsdecode(p) for p in subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=ROOT).split(b'\0') if p}
        if staged != set(paths): raise RuntimeError('Publication deferred: staged files changed concurrently')
        subprocess.run(['python3', str(ROOT / 'tools/check_public_metadata.py'), '--staged'], cwd=ROOT, check=True)
        subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
        subprocess.run(['git', 'commit', '-m', 'Run authorized norm-scaled routing pair and preserve complete native evidence'], cwd=ROOT, check=True)
        ssh = 'ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o HostKeyAlias=github.com -p 443 -o Hostname=ssh.github.com'
        subprocess.run(['git', '-c', 'core.sshCommand=' + ssh, 'push', 'origin', 'HEAD:refs/heads/main'], cwd=ROOT, check=True)
        finished = read(BASE / 'C2N0_execution_summary.json')
        result.update(status=('PUBLISHED_ONE_C2_RESULT_BATCH_PENDING_HUMAN_REVIEW'
                              if finished['status'] == 'COMPLETE_C2_NATIVE_VIDEOS_PENDING_FULL_REVIEW'
                              else 'PUBLISHED_C2_STOPPED_ATTEMPT_NO_COMPLETE_VIDEO_REVIEW'), commit=git('rev-parse', 'HEAD'))
    except BaseException as error:
        result.update(status='PUBLICATION_DEFERRED_RESULTS_PRESERVED', error_type=type(error).__name__, reason=str(error)); raise
    finally: atomic(result_path, result)


if __name__ == '__main__': main()
