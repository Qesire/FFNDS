"""Metadata-only draft for two independent paired Wan video scopes.

This command reads repository JSON and source bytes only, never tensors,
models, GPU or videos; it cannot authorize or execute an experiment.
"""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ORDINALS=(1040,979)
CODE=('tools/e0_sharq_wan_all160_runtime.py','tools/e0_sharq_wan_all160_video_driver.py','tests/test_e0_sharq_wan_all160_runtime.py','tests/test_e0_sharq_wan_all160_video_driver.py','tools/e0_sharq_wan_all160_video_plan.py','tests/test_e0_sharq_wan_all160_video_plan.py','tools/e0_sharq_wan_all160_video_evidence.py','tests/test_e0_sharq_wan_all160_video_evidence.py','tools/e0_sharq_wan_ffn_ops.py','tools/e0_sharq_wan_ffn_probe_v2.py','tools/e0_original_bf16_videos.py','tools/e0_raw_bf16_full160_video20_v2.py')
CONFIGS=('config/e0_s2_new12_dense_baseline_part0_20261029_v2.json','config/e0_s2_new12_dense_baseline_part1_20261029_v2.json','config/e0_s2_new12_sm120_fixed_depth_part0_20261029_v1.json')
def binding(path):
 p=ROOT/path;b=p.read_bytes();return {'path':path,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def plan(part):
 if type(part) is not int or part not in (0,1):raise ValueError('two explicit parts')
 p=[json.loads((ROOT/x).read_text()) for x in CONFIGS];cases=[c for x in p[:2] for c in x['cases'] if c['native_source_ordinal']==ORDINALS[part]]
 if len(cases)!=1:raise ValueError('source identity not unique')
 case=cases[0];models=[{k:m[k] for k in ('kind','filename','repo','revision','bytes','sha256','header_sha256')} for m in p[2]['models'] if m['kind'] in ('high','low')]
 if len(models)!=2 or {x['kind'] for x in models}!={'high','low'}:raise ValueError('two original experts required')
 return {'id':f'E0_SHARQ_WAN_ALL160_VIDEO_20261030_PART{part}','status':'DRAFT_ONLY_NOT_AUTHORIZED','part':part,'source_role':'EXPOSED_DEVELOPMENT_SINGLE_SEED',
  'case':{k:case[k] for k in ('source_id','native_source_ordinal','image','prompt_sha256','prompt','prior_history','seed')},'models':models,
  'scope':{'modes':['bf16_neutral_old','bf16_neutral_new_cache','dense32','sharq'],'complete_videos':4,'steps':4,'blocks_per_step':40,'ffn_sites_per_video':160,'linear_calls_per_video':320,'paired_seed':42,'static_shared_W_pack_calls':160,'dense32_gemms':320,'sparse_gemms':320,'dense_residual_fullK_gemms':320,'original_block_prefetch_observations':648,'original_prefetch_skips':0},
  'weights':'same original BF16 master in both FP4 arms; lazy GPU pack first dense32 and reuse for SharQ; no prefetch skip',
  'neutral':'two same-device BF16 videos: ten state identities and video SHA; both FP4 arms pair raw and initial noise',
  'resources':'NOT_BOUND; exact per-worker GPU/CPU/memory/disk/wall and output limits require independent review',
  'authoritative_requirements':['actual source/asset proofs','independent initial review, author response and exact recheck','separate CPU byte preflight actual EXIT0','ACTIVE registration and exact worker/job/source closure','fresh attempt, GPU isolation, output ledger and actual exit/closed verification'],
  'model_run_authorized':False,'repo_sources':[binding(x) for x in CODE+CONFIGS]}
def main():
 a=argparse.ArgumentParser();a.add_argument('--part',type=int,required=True);a.add_argument('--out',type=Path,required=True);x=a.parse_args()
 if x.out.exists():raise ValueError('fresh draft required')
 x.out.parent.mkdir(parents=True,exist_ok=True);x.out.write_text(json.dumps(plan(x.part),indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
