"""Non-executable staging for two-source P64 synchronization experiment.

Parent-closure preparation is metadata-only. Run/verify are explicitly denied
until seven-phase accounting and typed output verification are implemented.
"""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_chipmunk_input_video_owner as parent
import chipmunk_p64_sync_abba_contract as contract

PARENT_PART=(0,3)
def parent_proof(part):
 if type(part) is not int or part not in (0,1):raise ValueError('part differs')
 old=PARENT_PART[part];p=ROOT/f'config/e0_chipmunk_input_video_20261030_part{old}_pending_review_v1.json'
 oldplan=parent.native.read(p);parent.validate(oldplan,True)
 verified=parent.inspect_verification(SimpleNamespace(protocol=p))
 if verified.get('status')!='PASS_ACTUAL_EXIT_CLOSED_CHIPMUNK_INPUT_VIDEO':raise ValueError('actual independent parent EXIT required')
 out=parent.HF/oldplan['output_HF_relative'];r=parent.native.read(out/'RUN.json');e=parent.native.read(out.parent/'EXIT_RECEIPT.json');rb=parent.native.binding(out/'RUN.json',parent.HF)
 parent.vm.exited(e,rb,parent.native.sha(p),oldplan['id'])
 if r.get('job_spec_sha256')!=e.get('spec_sha256') or r.get('source_ordinal')!=contract.SOURCES[part] or r.get('seed')!=42 or r.get('status')!='COMPLETE_CHIPMUNK_INPUT_VIDEO_EXIT_PENDING':raise ValueError('parent run identity differs')
 phases=r['phases']
 if len(phases)!=5 or phases[0]['mode']!='none' or phases[4]['mode']!='high_x_recompute_p64':raise ValueError('parent phase differs')
 return {'parent_protocol':parent.native.binding(p,ROOT),'RUN':rb,'EXIT':parent.native.binding(out.parent/'EXIT_RECEIPT.json',parent.HF),'verification_RUN':verified['verification_RUN'],'verification_exit':verified['verification_exit'],'source_ordinal':r['source_ordinal'],'case':oldplan['case'],'dense':{'states':phases[0]['states'],'video':phases[0]['video']},'X':{'states':phases[4]['states'],'video':phases[4]['video']}}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('inspect-parent','draft','run','verify-closed'));ap.add_argument('--part',type=int,required=True);a=ap.parse_args()
 if a.action!='inspect-parent':raise RuntimeError('NOT_IMPLEMENTED_NOT_AUTHORIZED: owner, ledger and typed verifier incomplete')
 p=parent_proof(a.part);print(json.dumps({'status':'CLOSED_PARENT_METADATA_ONLY','source_ordinal':p['source_ordinal'],'schedule':contract.schedule(a.part),'counts':contract.expected_counts(a.part)}))
if __name__=='__main__':main()
