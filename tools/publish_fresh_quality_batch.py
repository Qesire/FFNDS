"""Wait for the scoped batch, export exact numbers and publish one result batch.

No model calls. Stop publication if local/remote history or unrelated work
changed. Use a normal non-forced main push after offline/privacy checks.
"""
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from export_fresh_quality_results import normalize_paths
from archive_public_json import archive, numeric_values
from rollout_contract import ROOT, atomic, read, sha
from worker_identity import ssh_arguments

FOLDER = ROOT / 'runs/r17_execution'
DYNAMIC = ['README.md', 'AGENTS.md', 'research/CURRENT_DECISION.json', 'research/20261021/method_triage.json']


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def verify_concurrent_files(snapshot):
    for name, expected in snapshot['static_sha256'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('Publication deferred: a prepared file changed during execution')
    need = lambda v: None if v else (_ for _ in ()).throw(RuntimeError('Publication deferred: concurrent live-record edit'))
    summary = read(ROOT / 'research/20261023/R17E0_execution_summary.json')
    status, used = summary['status'], summary['actual_debits']
    notice = '## R17E0固定验证批次已结束\n\n状态：`' + status + '`。实际扣账：`' + json.dumps(used, sort_keys=True) + ' `。'
    notice += '见[执行实录](research/20261023/R17E0_execution_summary.json)与[当前交接](research/20261023/HANDOFF_R17E0.md)。'
    notice += '原始结果留在HF缓存，尚待科学审阅和公开数值整理；没有自动宣称质量或Pareto通过。\n\n'
    publication_notice = '完整本轮数值与范围见[新来源官方质量和重复原生速度](docs/83_新来源官方质量与重复原生速度_R17E0_20261023.md)。\n\n'
    expected_readme = notice + snapshot['dynamic']['README.md']
    need((ROOT / 'README.md').read_text() in (expected_readme, publication_notice + expected_readme))
    marker = '# Latest actual full160 dynamic execution — R16V0 completed (2026-10-23 CST)'
    expected_agents = '# R17E0 fixed validation scope finished (2026-10-23 CST)\n\n'
    expected_agents += status + '. Read [execution summary](research/20261023/R17E0_execution_summary.json) '
    expected_agents += 'and [current handoff](research/20261023/HANDOFF_R17E0.md). '
    expected_agents += 'Model eligibility is empty; no retry or expanded model scope. '
    expected_agents += 'Actual quality/timing interpretation and publication require reading the saved evidence.\n\n'
    expected_agents += marker + snapshot['dynamic']['AGENTS.md'].split(marker, 1)[1]
    need((ROOT / 'AGENTS.md').read_text() == expected_agents)
    before = json.loads(snapshot['dynamic']['research/CURRENT_DECISION.json'])
    now = read(ROOT / 'research/CURRENT_DECISION.json')
    for key in ['execution_state', 'eligible_model_method_ids', 'current_handoff', 'latest_fresh_validation']:
        before[key] = copy.deepcopy(now[key])
    for key in ['active', 'phase', 'metric_phase', 'budget_spent', 'phase_completed_or_stopped', 'result']:
        if key in now['registered_fresh_quality']:
            before['registered_fresh_quality'][key] = copy.deepcopy(now['registered_fresh_quality'][key])
    before['next_preparation']['status'] = now['next_preparation']['status']
    need(before == now and not now['eligible_model_method_ids'] and not now['registered_fresh_quality']['active'])
    before = json.loads(snapshot['dynamic']['research/20261021/method_triage.json'])
    now = read(ROOT / 'research/20261021/method_triage.json')
    before['policy']['execution_state'] = now['policy']['execution_state']
    before['policy']['execution_note'] = now['policy']['execution_note']
    before['latest_fresh_validation'] = copy.deepcopy(now['latest_fresh_validation'])
    before['next_preparation']['status'] = now['next_preparation']['status']
    old_row = next(r for r in before['methods'] if r['id'].startswith('R17E0_'))
    new_row = next(r for r in now['methods'] if r['id'].startswith('R17E0_'))
    for key in ['status', 'model_execution_eligibility', 'result']:
        old_row[key] = new_row[key]
    need(before == now)


def render_result(success):
    base = ROOT / 'research/20261023'
    summary = read(base / 'R17E0_execution_summary.json')
    lines = ['# 固定新来源验证与原生完整测速：R17E0', '',
             '本轮冻结算法和所有参数，按独立授权的固定范围执行。实际结果如下；不自动宣称论文或Pareto通过。', '',
             '实际扣账：`' + json.dumps(summary['actual_debits'], ensure_ascii=False, sort_keys=True) + '`。', '']
    if success:
        analysis = read(base / 'R17E0_analysis.json')
        lines += ['23张新图、115条真实质量视频全部完成81帧解码，九个官方维度各按原始子集评分；另有5条真实预热和25条轮换测速。', '',
                  '| 方案 | 完整生成中位秒 | 相对dense完整生成加速 |', '|---|---:|---:|']
        for arm in ['dense_DQ', 'UB1', 'fixed_stride', 'fixed_hash', 'dynamic_residual']:
            value = analysis['timing'][arm]['full_generation_wall_seconds']['median']
            speed = 1 if arm == 'dense_DQ' else analysis['speed_ratios_dense_over_arm'][arm]['full_generation_wall_seconds']
            lines.append(f'| {arm} | {value:.6f} | {speed:.6f}× |')
        lines += ['', '测速是同一旧来源上每臂5次的技术重复，完整FFN和共享调制结果、全部重复保留在[分析](../research/20261023/R17E0_analysis.json)。', '',
                  '| 官方维度 | 来源数 | dense | UB1 | 固定stride | 固定hash | 动态 |', '|---|---:|---:|---:|---:|---:|---:|']
        for dimension, row in analysis['official_nine_quality'].items():
            means = row['equal_source_means_in_official_aggregate_units']
            lines.append('| ' + dimension + ' | ' + str(len(row['source_indices'])) + ' | ' +
                         ' | '.join(f'{means[a]:.8f}' for a in ['dense_DQ', 'UB1', 'fixed_stride', 'fixed_hash', 'dynamic_residual']) + ' |')
        lines += ['', '此表用原官方总分单位的等来源均值；已复算官方聚合和MUSIQ的100倍明细单位差。每源分数、对四个控制分别的配对区间和零分母次数全部保留。小型冻结子集不是完整官方数据集；一致性、动态程度、条件保持应联合解读。区间未作多重比较校正。', '',
                  '动态相对两个固定方案的全部比较见[分析](../research/20261023/R17E0_analysis.json)；[官方原始数值](../research/20261023/R17E0_official_metric_records.json)、[实际生成凭证](../research/20261023/R17E0_generation_receipts.json)、[逐帧解码](../research/20261023/R17E0_video_decode.json)和[资源/路径脱敏凭证](../research/20261023/R17E0_completed_resources.json)可独立复核。', '',
                  '完整原始事件、视频和模型留在HF缓存。尚缺同前提优化实现/公开方法的直接比较，亦不把RGB偏离直接当作感知质量下降。没有按结果调整算法或自动扩展实验。']
    else:
        lines += ['固定尝试在执行或控制失败后停止，已扣预算保留，没有自动重试。这个执行状态不构成总体动态方法不可能或质量崩坏的结论。', '',
                  '实际阶段状态和扣账见[执行实录](../research/20261023/R17E0_execution_summary.json)。原始日志与中间媒体留在HF/忽略的运行目录；旧结果和拒绝范围不变。']
        interruption = base / 'R17E0_interruption_audit.json'
        if interruption.exists():
            audit = read(interruption)
            lines += ['', f'已保存{audit["completed_generations"]}条完整生成：'
                      f'{audit["completed_warmups"]}条预热、{audit["completed_timing_generations"]}条测速、'
                      f'{audit["completed_fresh_quality_videos"]}条新来源视频。'
                      f'官方评分实际调用{audit["official_metric_calls"]}次。'
                      '全部已完成请求和视频的哈希已逐项复核。', '',
                      f'生成进程记录`{audit["original_error_type"]}`；管理进程的退出凭证缺失，'
                      '恢复核查时本地管理进程和远程生成进程均已退出。'
                      '记录与输出管道或SSH会话中断一致，具体进程终止原因尚未确认。'
                      '原始退出码未知，未补造成功或失败退出码。', '',
                      '见[中断核查](../research/20261023/R17E0_interruption_audit.json)与'
                      '[原始部分生成记录的脱敏副本](../research/20261023/R17E0_partial_generations.json)。']
        prefix = base / 'R17E0_system_timing_prefix.json'
        if prefix.exists():
            lines += ['', '完整测速前缀已经完成并保留，见[25次原生测速](../research/20261023/R17E0_system_timing_prefix.json)。'
                      '这是同一个旧来源上每方案5次的技术重复；新来源官方质量尚未完成。', '',
                      '| 方案 | 完整生成中位秒 | 相对dense完整生成加速 |', '|---|---:|---:|']
            timing = read(prefix)
            for arm in ['dense_DQ', 'UB1', 'fixed_stride', 'fixed_hash', 'dynamic_residual']:
                values = timing['per_arm'][arm]
                seconds = values['complete_generation_wall_seconds']['median']
                speed = 1 if arm == 'dense_DQ' else timing['speed_ratios_dense_over_arm'][arm]['complete_generation_wall_seconds']
                lines.append(f'| {arm} | {seconds:.6f} | {speed:.6f}× |')
    path = ROOT / 'docs/83_新来源官方质量与重复原生速度_R17E0_20261023.md'
    if path.exists():
        raise RuntimeError('Preserve an existing final result document')
    path.write_text('\n'.join(lines) + '\n')
    handoff = base / 'HANDOFF_R17E0.md'
    closing = ('Public numeric evidence and [result table](../../docs/83_新来源官方质量与重复原生速度_R17E0_20261023.md) are exported. No automatic model followup; next scientific review must read all dimensions and matched fixed controls.' if success else
               'Partial execution evidence and completed timing are preserved in the [result record](../../docs/83_新来源官方质量与重复原生速度_R17E0_20261023.md). Official quality remains unavailable; no automatic model retry or followup.')
    handoff.write_text(handoff.read_text() + '\n' + closing + '\n')
    readme = ROOT / 'README.md'
    readme.write_text('完整本轮数值与范围见[新来源官方质量和重复原生速度](docs/83_新来源官方质量与重复原生速度_R17E0_20261023.md)。\n\n' + readme.read_text())


def workflow(log):
    commands, block = [], False
    for line in (ROOT / '.github/workflows/research-agent-validation.yml').read_text().splitlines():
        if line.startswith('        run: |'):
            block = True
        elif line.startswith('        run: '):
            commands.append(line.split('run: ', 1)[1]); block = False
        elif block and line.startswith('          '):
            commands.append(line.strip())
        elif line and not line.startswith('          '):
            block = False
    with log.open('x') as stream:
        for command in commands:
            command = re.sub(r'^python(?=\s)', 'python3', command)
            command = command.replace('then python ', 'then python3 ')
            stream.write('\nCOMMAND ' + command + '\n'); stream.flush()
            subprocess.run(command, shell=True, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2'), check=True)


def main():
    lock = (FOLDER / 'PUBLICATION.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = read(FOLDER / 'PUBLICATION_SNAPSHOT.json')
    result = {'status': 'WAITING_FOR_FIXED_SCOPE', 'new_model_calls': 0}
    try:
        while not (FOLDER / 'PIPELINE_EXIT.json').exists():
            time.sleep(10)
        success = read(FOLDER / 'PIPELINE_EXIT.json')['status'] == 'COMPLETE_FIXED145_GENERATIONS_45_METRICS_AND_CPU_ANALYSIS'
        verify_concurrent_files(snapshot)
        if git('rev-parse', 'HEAD') != snapshot['expected_head'] or git('branch', '--show-current') != 'main':
            raise RuntimeError('Publication deferred: local main moved')
        if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode:
            raise RuntimeError('Publication deferred: another staged change exists')
        hf = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
        if success:
            command = ['bash', str(ROOT / 'scripts/worker_cpu_exec.sh'), str(ROOT / '.venv/bin/python'),
                       str(ROOT / 'tools/export_fresh_quality_results.py')]
            with (FOLDER / 'EXPORT.log').open('x') as log:
                subprocess.run([*ssh_arguments(), shlex.join(command)], stdout=log, stderr=subprocess.STDOUT, check=True)
        else:
            original = hf / 'derived/ffnds/R17E0_fresh_quality/BUDGET.json'
            raw = read(original); neutral = normalize_paths(raw, hf)
            if numeric_values(raw) != numeric_values(neutral):
                raise RuntimeError('Failed-budget public copy changed numbers')
            temporary = hf / 'derived/ffnds/R17E0_fresh_quality/PUBLIC_PARTIAL_BUDGET.json'
            atomic(temporary, neutral)
            archive(temporary, ROOT / 'research/20261023/R17E0_partial_budget.json')
        render_result(success)
        workflow(FOLDER / 'FINAL_OFFLINE_WORKFLOW.log')
        # A legacy probe writes the same errors in hash-dependent order.
        generated = ROOT / 'research/20261023/metric_guard_after_R15.json'
        old = subprocess.check_output(['git', 'show', 'HEAD:research/20261023/metric_guard_after_R15.json'], cwd=ROOT)
        def canonical(v):
            if isinstance(v, dict): return {k: canonical(x) for k, x in v.items()}
            if isinstance(v, list): return sorted((canonical(x) for x in v), key=lambda x: json.dumps(x, sort_keys=True))
            return v
        if canonical(read(generated)) == canonical(json.loads(old)):
            generated.write_bytes(old)
        else:
            raise RuntimeError('Unexpected legacy generated-evidence change')
        remote_head = git('ls-remote', 'https://github.com/Qesire/FFNDS.git', 'refs/heads/main').split()[0]
        if remote_head != snapshot['expected_head'] or git('rev-parse', 'HEAD') != snapshot['expected_head']:
            raise RuntimeError('Publication deferred: main moved; preserve and reconcile intervening work')
        verify_concurrent_files(snapshot)
        if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode:
            raise RuntimeError('Publication deferred: another staged edit appeared')
        names = subprocess.check_output(['git', 'ls-files', '--modified', '--others', '--exclude-standard', '-z'], cwd=ROOT).split(b'\0')
        changed = {os.fsdecode(n) for n in names if n}
        new_result_files = {'research/20261023/R17E0_execution_summary.json',
            'research/20261023/R17E0_metric_phase_protocol.json', 'research/20261023/HANDOFF_R17E0.md',
            'research/20261023/R17E0_interruption_audit.json',
            'research/20261023/R17E0_completed_resources.json',
            'docs/83_新来源官方质量与重复原生速度_R17E0_20261023.md'}
        for name in ['analysis', 'generation_receipts', 'official_metric_records', 'budget_debit', 'video_decode', 'partial_budget', 'partial_generations']:
            path = 'research/20261023/R17E0_' + name + '.json'
            new_result_files.update([path, path + '.redaction.json'])
        extra = changed - set(snapshot['allowed_paths']) - new_result_files
        if extra:
            raise RuntimeError('Publication deferred: unrelated work appeared')
        subprocess.run(['git', 'add', '--', *sorted(changed)], cwd=ROOT, check=True)
        staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=ROOT)
        if {os.fsdecode(name) for name in staged.split(b'\0') if name} != changed:
            raise RuntimeError('Publication deferred: staged files changed concurrently')
        subprocess.run(['python3', str(ROOT / 'tools/check_public_metadata.py'), '--staged'], cwd=ROOT, check=True)
        subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
        title = 'Validate frozen dynamic FFN on fresh sources and repeated native system timing' if success else 'Preserve stopped fixed fresh validation and exact model budget'
        subprocess.run(['git', 'commit', '-m', title], cwd=ROOT, check=True)
        result['commit'] = git('rev-parse', 'HEAD')
        ssh = 'ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o HostKeyAlias=github.com -p 443 -o Hostname=ssh.github.com'
        subprocess.run(['git', '-c', 'core.sshCommand=' + ssh, 'push', 'origin', 'HEAD:refs/heads/main'],
                       cwd=ROOT, env=dict(os.environ, GIT_TERMINAL_PROMPT='0'), check=True)
        result['status'] = 'PUBLISHED_ONE_FIXED_RESULT_BATCH_TO_GITHUB_MAIN'
    except BaseException as error:
        result.update(status='PUBLICATION_DEFERRED_ARTIFACTS_PRESERVED', error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        atomic(FOLDER / 'PUBLICATION_EXIT.json', result)
        print(json.dumps(result), flush=True)
        lock.close()


if __name__ == '__main__':
    main()
