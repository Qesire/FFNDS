"""Finite, persistent same-file byte gates for one resident ABBA source.

Does not register work or access files at import. ``reserve`` must persist the
full read predebit before callback, including on partial failure. Stable byte
identity refers to files opened for the comparison, not an adversarial writer
able to forge unchanged filesystem metadata.
"""
from pathlib import Path
from sharq_resident_artifact_verifier_draft import byte_equal

STATES=frozenset(('raw_noise_FP32','initial_noise',*(f'flow_step{i}' for i in range(4)),*(f'latent_step{i}' for i in range(4))))
ARMS=('warmup','legacy_A1','resident_B1','resident_B2','legacy_A2')

def need(x,msg):
 if not x:raise ValueError(msg)

class ReferenceGate:
 def __init__(self,hf,refs,reserve,persist,max_compare_bytes=1<<30):
  self.hf=Path(hf).resolve();self.refs=refs;self.reserve=reserve;self.persist=persist;self.maximum=max_compare_bytes;self.predebited=0;self.completed=0;self.inflight=None;self.checks=[]
 def _path(self,b):
  p=(self.hf/b['path']).resolve();need(p.is_relative_to(self.hf) and p.is_file() and p.stat().st_size==b['bytes'],'bound path or size');return p
 def check(self,arm,kind,current):
  need(arm in ARMS and (kind in STATES or kind=='video'),'unknown gate');old=self.refs['bf16' if arm=='warmup' else 'dense32'][kind]
  if current['bytes']!=old['bytes'] or current['sha256']!=old['sha256']:
   self.checks.append({'arm':arm,'kind':kind,'byte_equal':False,'reason':'SHA_OR_SIZE_MISMATCH_BEFORE_BYTE_READ'});self.persist(self.snapshot());raise ValueError('bound SHA or size differs')
  oldpath=self._path(old);newpath=self._path(current);a=oldpath.stat();b=newpath.stat();need(oldpath!=newpath and (a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino),'reference/output alias');n=current['bytes']+old['bytes'];need(self.inflight is None and self.predebited+n<=self.maximum,'comparison budget')
  self.predebited+=n;self.inflight={'arm':arm,'kind':kind,'reserved_bytes':n,'completion':'UNKNOWN'};self.persist(self.snapshot())
  same=self.reserve(n,lambda:byte_equal(oldpath,newpath,old['bytes']));self.completed+=n;self.inflight=None;self.checks.append({'arm':arm,'kind':kind,'bytes_each':old['bytes'],'byte_equal':same});self.persist(self.snapshot());need(same,'reference byte mismatch')
 def finish(self):
  need(len(self.checks)==55 and {(x['arm'],x['kind']) for x in self.checks}=={(a,k) for a in ARMS for k in (*STATES,'video')} and self.predebited==self.completed and self.inflight is None,'reference coverage')
  return self.snapshot()
 def snapshot(self):return {'predebited_bytes':self.predebited,'completed_bytes':self.completed,'inflight':self.inflight,'checks':list(self.checks),'interpretation':'full byte comparisons; interrupted reads remain UNKNOWN'}
