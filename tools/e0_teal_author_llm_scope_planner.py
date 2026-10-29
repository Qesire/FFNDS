"""Static, fail-closed plans for TEAL author entrypoints. Does not run models.

A launcher needs independent scope review, ACTIVE, receipt/limits enforcement and
asset closure; this planner intentionally has no execute mode.
"""
import argparse
import json
from pathlib import Path

REV='dce01c9b08f87459cf36a430d809084718273017'
SIZE=24246638
SHA='06391b656a06fd3fb9d213160ef2398796c3b7f3dc75ef1e3ced30d461517073'
SOURCE=Path('/home/xiang/.cache/huggingface/ffnds-research-sources/p0_teal_full_llm_20261029')
SUPPORT=Path('/home/xiang/.cache/huggingface/ffnds-research-sources/p0_teal_gptfast_support_20261030')

def calibration(model,output):
 if '/proc/' in model or '/dev/fd/' in model: raise ValueError('persistent HF snapshot required')
 return {'state':'REVIEW_REQUESTED_NOT_EXECUTABLE','entrypoint':str(SOURCE/'teal/grab_acts.py'),'argv':['--model_name',model,'--output_path',output], 'dataset':{'repo':'tatsu-lab/alpaca','revision':REV,'bytes':SIZE,'sha256':SHA,'rows':'streaming train skip(0) take(300)'},'expected':{'layers':32,'sorts':128,'histogram_files':64,'input_files':32,'batch':10,'sequence':2048,'h2_mlp_width':14336},'adaptation_required':'inject pinned offline dataset without changing row order; record token IDs and actual chunk count; reject count != 10 before forward','hard_requirements':['closed HF snapshot, never converter temporary FD paths','reviewed environment and FA2 compatibility','bounded single GPU, CPU, wall, file and aggregate disk','durable exit and source/asset/recipe SHA closure','record CPU FP32 copies, GPU sort and CPU hist cost'],'execute':False}

def generation(checkpoint,hist,prompt,arm):
 if not checkpoint.endswith('/model.pth') or 'llama-3-8b' not in str(Path(checkpoint).parent).lower(): raise ValueError('gpt-fast llama-3-8b checkpoint path required')
 if not prompt: raise ValueError('prompt required')
 if arm not in ('dense','uniform50'): raise ValueError('explicit zero neutral and FFN-only need separate intervention')
 if arm=='uniform50' and not hist: raise ValueError('hist path required')
 args=['--checkpoint_path',checkpoint,'--device','cuda','--prompt',prompt,'--num_samples','2','--max_new_tokens','64','--top_k','200','--temperature','0.8','--compile']
 if arm=='uniform50': args+=['--hist_path',hist,'--sparsity','0.5']
 return {'state':'REVIEW_REQUESTED_NOT_EXECUTABLE','entrypoint':str(SOURCE/'gpt-fast/generate.py'),'author_support_path':str(SUPPORT),'argv':args,'author_seed':1234,'prefill':'multi-token unmasked dense','decode':'single-token sparse only in uniform50','author_compile_warmups':1,'measured_samples':2,'execute':False,'hard_requirements':['closed actual model.pth and tokenizer SHA','64 validated histogram files for uniform50','explicit-zero neutral plus teacher forced logits gate before sparsity interpretation','separate FFN-only scope','budget compile/autotune and GPU/setup/end-to-end timing','reviewed launcher with actual ACTIVE and receipt checks']}

def main():
 a=argparse.ArgumentParser();a.add_argument('action',choices=['calibration','generation']);a.add_argument('--model');a.add_argument('--output');a.add_argument('--checkpoint');a.add_argument('--hist');a.add_argument('--prompt');a.add_argument('--arm',choices=['dense','uniform50']);x=a.parse_args()
 if x.action=='calibration':
  if not x.model or not x.output:a.error('model and output required')
  p=calibration(x.model,x.output)
 else:
  if not x.checkpoint or not x.arm:a.error('checkpoint and arm required')
  p=generation(x.checkpoint,x.hist,x.prompt,x.arm)
 print(json.dumps(p,indent=2))
if __name__=='__main__':main()
