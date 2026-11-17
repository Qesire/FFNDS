"""ICC CAL11 main/paired-diagnostic schedule, with an explicit native adapter seam.

This file does not build a model or start a job. A separately reviewed adapter
must bind source bytes, native VAE/T5/I2V, Wan scheduler and collector calls.
No estimates here are evidence of an executed model call.
"""
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

EXPERT_STEPS = {'high': (0, 1), 'low': (2, 3)}

@dataclass(frozen=True)
class Transaction:
    source: str
    family: str
    expert: str
    indices: tuple
    forward_noise_seed: int
    use_empty_text: bool
    branch: str
    main_id: str
    may_reuse_main: bool

    @property
    def identifier(self):
        return f'{self.main_id}:{self.branch}'


def validate_and_plan(manifest):
    records = manifest['records']
    if len(records) != 11 or manifest['calibration_seed'] != 42:
        raise ValueError('requires exactly frozen CAL11 seed42')
    ids = [r['source_cluster_id'] for r in records]
    if len(set(ids)) != 11: raise ValueError('duplicate source cluster')
    by_rank = sorted(ids, key=lambda x: hashlib.sha256(('icc-clean-diag-v1|' + x).encode()).hexdigest())
    cal3 = by_rank[:3]
    if cal3 != manifest['clean_cal3_source_clusters']:
        raise ValueError('frozen CAL3 mismatch')
    jobs = []
    for rec in records:
        source = rec['source_cluster_id']
        if rec['diagnostic_rank_key'] != hashlib.sha256(('icc-clean-diag-v1|' + source).encode()).hexdigest():
            raise ValueError('bad diagnostic key')
        if hashlib.sha256(rec['prompt'].encode('utf8')).hexdigest() != rec['prompt_sha256_utf8']:
            raise ValueError('prompt identity mismatch')
        draws = rec['draws']
        if len(draws) != 4 or {(d['family'], d['expert']) for d in draws} != {(f,e) for f in ('CA','CD') for e in ('high','low')}:
            raise ValueError('need four unique CA/CD x expert draws')
        for d in draws:
            family, expert = d['family'], d['expert']
            indices = tuple(d['step_indices'])
            if (family == 'CA' and (len(indices) != 1 or indices[0] not in EXPERT_STEPS[expert])) or (family == 'CD' and indices != EXPERT_STEPS[expert]):
                raise ValueError('invalid family/expert timesteps')
            if type(d['use_empty_text']) is not bool or type(d['forward_noise_seed']) is not int:
                raise ValueError('invalid frozen RNG metadata')
            mid = f'{source}|{family}|{expert}'
            base = dict(source=source, family=family, expert=expert, indices=indices,
                        forward_noise_seed=d['forward_noise_seed'], use_empty_text=d['use_empty_text'], main_id=mid)
            jobs.append(Transaction(**base, branch='main', may_reuse_main=False))
            if source in cal3:
                # sigma0=1 makes H CD pair and a H CA index0 identical only
                # after checking actual constructed inputs and context identity.
                can_reuse = expert == 'high' and indices[0] == 0
                jobs.append(Transaction(**base, branch='posterior', may_reuse_main=can_reuse))
                jobs.append(Transaction(**base, branch='text_flip', may_reuse_main=False))
    main = sum(len(t.indices) for t in jobs if t.branch == 'main')
    post_max = sum(len(t.indices) for t in jobs if t.branch == 'posterior' and not t.may_reuse_main)
    text_max = sum(len(t.indices) for t in jobs if t.branch == 'text_flip')
    if (main, text_max) != (66, 18) or post_max > 12: raise ValueError('budget mismatch')
    return jobs, {'main_nfe': main, 'posterior_extra_nfe_max': post_max,
                  'text_extra_nfe_max': text_max, 'main_and_diagnostic_nfe_max': main + post_max + text_max,
                  'cal3_source_clusters': cal3}


class NFEJournal:
    """Counts each attempt before model entry; completed calls are separate events."""
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists(): raise FileExistsError(self.path)
        self.count = self.main_count = self.diagnostic_count = 0

    def _event(self, event):
        with self.path.open('a', encoding='utf8') as f:
            f.write(json.dumps(event, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())

    def attempt(self, txn, step):
        if step not in txn.indices: raise ValueError('unregistered timestep')
        if self.count >= 96: raise RuntimeError('NFE ceiling')
        is_main = txn.branch == 'main'
        if (is_main and self.main_count >= 66) or (not is_main and self.diagnostic_count >= 30):
            raise RuntimeError('branch NFE ceiling')
        self.count += 1
        self.main_count += int(is_main); self.diagnostic_count += int(not is_main)
        self._event({'event':'NFE_ATTEMPT','transaction':txn.identifier,'step':step,'total':self.count})

    def completed(self, txn, step, trace):
        self._event({'event':'NFE_COMPLETED','transaction':txn.identifier,'step':step,'trace':trace})


class NativeAdapter:
    """Interface only. All methods must be implemented and reviewed against live Wan.

    encode_source performs actual bound clip decoding, native VAE once for raw mu
    and logvar, two text contexts and original I2V conditioning. prepare_transaction
    reuses the exact stored noise and source condition for each matched draw.
    first_model_call performs a single native forward under FourBoundaryTap; CD
    step_post is the native scheduler call, not a recomputation or new noise.
    Prepared objects must preserve immutable initial-input and context identities
    even after the CD scheduler advances; reuse compares those frozen identities.
    finish_transaction durably commits only complete source/expert/family stats.
    """
    def encode_source(self, record): raise NotImplementedError
    def prepare_transaction(self, encoded, record, txn): raise NotImplementedError
    def same_native_input_and_conditions(self, left, right): raise NotImplementedError
    def begin_transaction(self, prepared, txn): raise NotImplementedError
    def forward(self, prepared, txn, step, phase): raise NotImplementedError
    def native_step_post(self, prepared, txn, first): raise NotImplementedError
    def finish_transaction(self, prepared, txn, responses): raise NotImplementedError
    def compare_diagnostic(self, main, alternate, txn): raise NotImplementedError
    def abort_transaction(self, prepared, txn): raise NotImplementedError


def execute(manifest, adapter, journal, *, authorize, source_ids=None):
    """Execute reviewed plan; abort on any failure; never retries automatically.

    authorize must validate real ACTIVE/review/code/data/resource state and raise
    before any VAE/T5/model work if not eligible. It is not a boolean bypass.
    """
    if type(adapter) is NativeAdapter: raise TypeError('native adapter is not implemented')
    jobs, budget = validate_and_plan(manifest)
    if source_ids is not None:
        known = {job.source for job in jobs}
        if not source_ids or len(set(source_ids)) != len(source_ids) or not set(source_ids) <= known:
            raise ValueError('shard must bind distinct sources within the frozen CAL11')
        jobs = [job for job in jobs if job.source in source_ids]
    budget = {**budget, 'shard_sources': sorted({job.source for job in jobs}),
              'shard_main_nfe': sum(len(job.indices) for job in jobs if job.branch == 'main'),
              'shard_diagnostic_nfe': sum(len(job.indices) for job in jobs if job.branch != 'main' and not job.may_reuse_main)}
    authorize(manifest, jobs, budget)
    records = {r['source_cluster_id']: r for r in manifest['records']}
    encoded_by_source = None
    current_source = None
    main_results, results = {}, {}
    for txn in jobs:
        rec = records[txn.source]
        if txn.source != current_source:
            encoded_by_source = adapter.encode_source(rec)
            current_source = txn.source
            main_results = {}
        prepared = adapter.prepare_transaction(encoded_by_source, rec, txn)
        main = main_results.get(txn.main_id)
        if txn.may_reuse_main:
            if main is None or not adapter.same_native_input_and_conditions(main['prepared'], prepared):
                raise RuntimeError('required sigma=1 posterior reuse identity mismatch')
            results[txn.identifier] = {'status':'REUSED_IDENTICAL_INPUT', 'main':txn.main_id}
            continue
        responses = []
        try:
            adapter.begin_transaction(prepared, txn)
            for i, step in enumerate(txn.indices):
                phase = 'CA' if txn.family == 'CA' else ('anchor' if i == 0 else 'hit')
                journal.attempt(txn, step)
                response, trace = adapter.forward(prepared, txn, step, phase)
                journal.completed(txn, step, trace)
                responses.append(response)
                if txn.family == 'CD' and i == 0:
                    adapter.native_step_post(prepared, txn, response)
            committed = adapter.finish_transaction(prepared, txn, responses)
        except BaseException:
            adapter.abort_transaction(prepared, txn)
            raise
        if txn.branch == 'main':
            main_results[txn.main_id] = {'prepared': prepared, 'responses': responses, 'committed': committed}
            results[txn.identifier] = committed
        else:
            if main is None: raise RuntimeError('diagnostic lacks paired main')
            results[txn.identifier] = adapter.compare_diagnostic(main, {'prepared':prepared, 'responses':responses, 'committed':committed}, txn)
    if journal.main_count != budget['shard_main_nfe'] or journal.diagnostic_count != budget['shard_diagnostic_nfe']:
        raise RuntimeError('observed NFE accounting mismatch')
    return {'status':'EXECUTED_ADAPTER_RESULTS_REQUIRE_VALIDATION', 'budget':budget,
            'attempted_nfe':journal.count, 'main_nfe':journal.main_count,
            'diagnostic_nfe':journal.diagnostic_count, 'transactions':results}


