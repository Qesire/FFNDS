"""Construct and check the explicit repository source closure for ABBA.

Offline source metadata only. No model assets or science tasks are accessed.
Source hashes must be predebited by the actual registered preflight/owner.
"""
import hashlib,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
OWN={
 'tools/sharq_resident_ffn_load_draft.py','tests/test_sharq_resident_ffn_load_draft.py',
 'tools/sharq_resident_ffn_guard_draft.py','tests/test_sharq_resident_ffn_guard_draft.py',
 'tools/sharq_resident_dense32_runtime_draft.py','tests/test_sharq_resident_dense32_runtime_draft.py',
 'tools/sharq_resident_dense32_driver_draft.py','tests/test_sharq_resident_dense32_driver_draft.py',
 'tools/sharq_resident_abba_core_draft.py','tests/test_sharq_resident_abba_core_draft.py',
 'tools/sharq_resident_prefetch_audit_draft.py','tests/test_sharq_resident_prefetch_audit_draft.py',
 'tools/sharq_resident_abba_coverage_draft.py','tests/test_sharq_resident_abba_coverage_draft.py',
 'tools/sharq_resident_artifact_verifier_draft.py','tests/test_sharq_resident_artifact_verifier_draft.py',
 'tools/sharq_resident_same_input_control_draft.py','tests/test_sharq_resident_same_input_control_draft.py',
 'tools/sharq_resident_copy_ledger_draft.py','tests/test_sharq_resident_copy_ledger_draft.py',
 'tools/sharq_resident_cache_checkpoint_draft.py','tests/test_sharq_resident_cache_checkpoint_draft.py',
 'tools/sharq_resident_reference_gate_draft.py','tests/test_sharq_resident_reference_gate_draft.py',
 'tools/sharq_resident_source_plan_draft.py','tests/test_sharq_resident_source_plan_draft.py',
}

def source_paths(parent,extra=()):
 base=[x['path'] for x in parent['repo_sources']]
 if len(base)!=len(set(base)):raise ValueError('duplicate parent source')
 paths=set(base)|OWN|set(extra)
 if any(pathlib.PurePosixPath(p).is_absolute() or '..' in pathlib.PurePosixPath(p).parts for p in paths):raise ValueError('source path escapes root')
 return sorted(paths)

def bindings(paths,root=ROOT):
 root=pathlib.Path(root).resolve();out=[]
 for name in paths:
  p=(root/name).resolve()
  if not p.is_relative_to(root) or not p.is_file():raise ValueError('missing/escaped source')
  h=hashlib.sha256();n=0
  with p.open('rb') as f:
   while True:
    b=f.read(1<<20)
    if not b:break
    h.update(b);n+=len(b)
  if n!=p.stat().st_size:raise ValueError('source changed')
  out.append({'path':name,'bytes':n,'sha256':h.hexdigest()})
 return out
