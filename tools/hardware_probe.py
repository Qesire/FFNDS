"""Collect target GPU metadata; does not load a model or benchmark kernels."""
import argparse,json,os,platform,subprocess,sys
from pathlib import Path
def run(argv):
 try:
  p=subprocess.run(argv,capture_output=True,text=True,timeout=20)
  return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 except (OSError,subprocess.TimeoutExpired) as e:return {'error':type(e).__name__+': '+str(e)}
def collect(no_torch=False):
 r={'purpose':'target-device metadata only; no model, training or speed measurement','source_profile':'A100-tuned archive','expected_target':'H20, unconfirmed until observed','platform':platform.platform(),'python':sys.version,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'nvidia_smi':run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader']),'nvcc':run(['nvcc','--version'])}
 if no_torch:r['torch']={'status':'SKIPPED_BY_FLAG'};return r
 try:
  import torch
  d={'version':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'compiled_arch_list':torch.cuda.get_arch_list() if torch.cuda.is_available() else [],'devices':[]}
  if d['cuda_available']:
   for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i);cc=torch.cuda.get_device_capability(i)
    d['devices'].append({'index':i,'name':p.name,'compute_capability':list(cc),'total_memory_bytes':p.total_memory,'multiprocessor_count':p.multi_processor_count,'suggested_torch_cuda_arch_list':str(cc[0])+'.'+str(cc[1])})
  r['torch']=d
 except Exception as e:r['torch']={'status':'IMPORT_OR_DEVICE_QUERY_FAILED','error_type':type(e).__name__,'error':str(e)}
 return r
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path);ap.add_argument('--no-torch',action='store_true',help='Inspect only system commands; do not import torch');a=ap.parse_args()
 r=collect(a.no_torch);s=json.dumps(r,ensure_ascii=False,indent=2);print(s)
 if a.output:
  if a.output.exists():ap.error('output already exists; choose a new filename')
  a.output.write_text(s+'\n',encoding='utf8')
 return 0
if __name__=='__main__':raise SystemExit(main())
