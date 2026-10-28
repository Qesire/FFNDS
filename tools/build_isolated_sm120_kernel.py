"""Registered CPU-only isolated LightX2V kernel build; never imports GPU libraries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import resource
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
ID='E0_SM120_LIGHTX2V_KERNEL_BUILD_CPU_20261029'
REV='d55f6beeebb6df501a250dc82827db97660f06e0'
RESOURCES={'GPU':0,'CPU_threads':2,'wall_seconds':1800,'outer_seconds':1860,'local_output_stop_threshold_bytes':6*1024**3,'single_file_RLIMIT_FSIZE_bytes':2*1024**3,'free_disk_min_bytes':20*1024**3,'host_available_min_bytes':32*1024**3,'HF_output_bytes_max':16*1024**2}


def need(v,msg):
    if not v:raise ValueError(msg)


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def check_file(root,b):
    path=(root/b['path']).resolve();need(path.is_relative_to(root.resolve()) and path.stat().st_size==b['bytes'] and sha(path)==b['sha256'],'bound source differs');return path


def commands(cfg,source,build,overlay):
    return [[cfg['cmake'],'-S',str(source),'-B',str(build),'-G','Ninja',f"-DCMAKE_MAKE_PROGRAM={cfg['ninja']}",f"-DCMAKE_CUDA_COMPILER={cfg['nvcc']}",f"-DCMAKE_CXX_COMPILER={cfg['gxx']}",f"-DCMAKE_CUDA_HOST_COMPILER={cfg['gxx']}",f"-DCUDAToolkit_ROOT={cfg['cuda_root']}",'-DCMAKE_CUDA_ARCHITECTURES=OFF',f"-DCUTLASS_PATH={cfg['cutlass']}",f"-DPython_EXECUTABLE={cfg['python']}",f"-DCMAKE_PREFIX_PATH={cfg['torch_cmake']}",'-DSKBUILD_SABI_VERSION=3.11','-DSKBUILD_SABI_COMPONENT=Development.SABIModule','-DENABLE_CCACHE=OFF','-DCMAKE_BUILD_TYPE=Release',f'-DCMAKE_INSTALL_PREFIX={overlay}'],[cfg['cmake'],'--build',str(build),'--parallel','2'],[cfg['cmake'],'--install',str(build)]]


def run_process(cmd,env,log,deadline,local_root,record):
    p=None
    try:
        with log.open('wb') as f:
            p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            while p.poll() is None:
                need(time.monotonic()<deadline,'build wall exceeded')
                used=sum(x.stat().st_size for x in local_root.rglob('*') if x.is_file() and not x.is_symlink())
                record['sampled_local_output_peak_bytes']=max(record.get('sampled_local_output_peak_bytes',0),used)
                need(used<=RESOURCES['local_output_stop_threshold_bytes'],'local output stop threshold')
                need(shutil.disk_usage(local_root).free>=RESOURCES['free_disk_min_bytes'],'local free disk reserve')
                available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
                need(available>=RESOURCES['host_available_min_bytes'],'host memory reserve')
                time.sleep(1)
            record['returncode']=p.returncode
            need(p.returncode==0,'build command failed')
    except BaseException:
        if p is not None and p.poll() is None:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait()
        if p is not None:record['returncode']=p.returncode
        raise
    finally:
        if log.exists():record['log']={'local_relative':str(log.relative_to(local_root)),'bytes':log.stat().st_size,'sha256':sha(log)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--hf-root',type=Path,required=True);ap.add_argument('--local-root',type=Path,required=True);args=ap.parse_args();raw=args.protocol.read_bytes();p=json.loads(raw);digest=hashlib.sha256(raw).hexdigest();need(p['id']==ID and p['status']=='FROZEN_REGISTERED' and p['resources']==RESOURCES and p['cutlass_commit']==REV,'frozen protocol')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))==2,'CPU-only affinity/visibility')
    active=json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text());standing=active['user_continuation_authorization_20261027'];scopes=[x for x in active['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==ID];need(standing.get('status')=='ACTIVE_CURRENT_USER_DIRECTED_SESSION' and standing.get('new_scope_registration_allowed') is True and standing.get('automatic_retry') is False and len(scopes)==1 and scopes[0].get('protocol_sha256')==digest and scopes[0].get('resources')==RESOURCES and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE'),'ACTIVE authorization')
    for b in p['repo_sources']:check_file(ROOT,b)
    for b in p['kernel_sources']:check_file(ROOT,b)
    cfg=p['private_profile'];need(all(cfg.get(k) and cfg[k]!='NOT_BOUND' for k in ('cmake','ninja','nvcc','gxx','cuda_root','python','torch_cmake','cutlass')),'private compiler profile NOT_BOUND')
    cutlass=Path(cfg['cutlass']);need(subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()==REV and not subprocess.check_output(['git','-C',str(cutlass),'status','--porcelain','--untracked-files=no'],text=True).strip(),'CUTLASS revision or dirty')
    for b in p['cutlass_includes']:check_file(cutlass,b)
    for k in ('cmake','ninja','nvcc','gxx','python'):
        tool=Path(cfg[k]);need(sha(tool)==p['tool_sha256'][k],'compiler/tool SHA mismatch')
    need(sha(Path(cfg['torch_cmake'])/'TorchConfig.cmake')==p['tool_sha256']['torch_config'],'TorchConfig SHA mismatch')
    version=subprocess.check_output([cfg['cmake'],'--version'],text=True).splitlines()[0];need(version==p['tool_versions']['cmake'] and tuple(int(x) for x in version.split()[-1].split('.')[:2])>=(3,26),'CMake 3.26+ USE_SABI required')
    need('release 12.8' in subprocess.check_output([cfg['nvcc'],'--version'],text=True) and subprocess.check_output([cfg['python'],'--version'],text=True).strip()==p['tool_versions']['python'],'CUDA/Python version differs')
    local=args.local_root.resolve();hf=args.hf_root.resolve();out=(hf/p['output_HF_relative']).resolve();need(out.is_relative_to(hf) and not out.exists() and not local.exists(),'fresh isolated output required');need(shutil.disk_usage(local.parent).free>=RESOURCES['free_disk_min_bytes'],'local free disk reserve');soft,hard=resource.getrlimit(resource.RLIMIT_FSIZE);need(hard==resource.RLIM_INFINITY or hard>=RESOURCES['single_file_RLIMIT_FSIZE_bytes'],'single file hard limit');resource.setrlimit(resource.RLIMIT_FSIZE,(RESOURCES['single_file_RLIMIT_FSIZE_bytes'],RESOURCES['single_file_RLIMIT_FSIZE_bytes']));local.mkdir(parents=True);out.mkdir(parents=True)
    source=local/'source';build=local/'build';overlay=local/'overlay';source.mkdir();r={'id':ID,'protocol_sha256':digest,'status':'STARTED_NO_RETRY','commands':[],'CUDA_VISIBLE_DEVICES':'','model_calls':0,'GPU_calls':0,'compiler_version':version};started=time.monotonic()
    def save():
        data=(json.dumps(r,indent=2)+'\n').encode();need(len(data)<=RESOURCES['HF_output_bytes_max'],'manifest budget');tmp=out/'RUN.tmp';tmp.write_bytes(data);os.replace(tmp,out/'RUN.json')
    save()
    try:
        for b in p['kernel_sources']:
            original=check_file(ROOT,b);relative=Path(b['path']).relative_to('.runtime/code/LightX2V/lightx2v_kernel');dst=source/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,dst)
        cmake=source/'CMakeLists.txt';text=cmake.read_text();need(text.count('--threads=32')==1 and 'compute_120a,code=sm_120a' in text,'CMake flags differ');cmake.write_text(text.replace('--threads=32','--threads=1'))
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',CMAKE_BUILD_PARALLEL_LEVEL='2',OMP_NUM_THREADS='2',MAX_JOBS='2')
        for i,cmd in enumerate(commands(cfg,source,build,overlay)):
            r['commands'].append({'argv':cmd,'status':'PREDEBITED'});save();
            try:run_process(cmd,env,local/f'command{i}.log',started+1800,local,r['commands'][-1])
            except BaseException:r['commands'][-1]['status']='FAILED';save();raise
            r['commands'][-1]['status']='COMPLETE';save()
        package=overlay/'lightx2v_kernel';package.mkdir(parents=True,exist_ok=True)
        for name in ('__init__.py','gemm.py','kv_cache.py','utils.py','version.py'):shutil.copyfile(source/'python/lightx2v_kernel'/name,package/name)
        artifacts=list(package.glob('common_ops*.so'));need(len(artifacts)==1 and artifacts[0].open('rb').read(4)==bytes((127,69,76,70)),'single ELF missing')
        f=artifacts[0];r.update(status='COMPLETE_BUILD_NO_GPU_EXECUTION',artifact={'local_relative':str(f.relative_to(local)),'bytes':f.stat().st_size,'sha256':sha(f)},wall_seconds=time.monotonic()-started);save()
    except BaseException as e:
        r.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e),wall_seconds=time.monotonic()-started);save();raise

if __name__=='__main__':main()
