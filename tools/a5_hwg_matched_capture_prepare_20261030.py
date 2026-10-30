"""Bind finite acquisition protocols and private jobs; never authorize/start."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import a5_hwg_matched_capture_20261030 as owner


def write_new(path,data,private=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
    if private:path.chmod(0o600)


def source_inventory():
    files={ROOT/v for v in [owner.SELF,'tools/a5_hwg_matched_capture_prepare_20261030.py',
        'tools/bf16_observer.py','tools/hif4_observer.py','tools/startup_gpu_reservation.py',
        'tools/assigned_gpu_resources.py','tools/worker_identity.py','tools/durable_worker_job.py',
        'scripts/worker_teal_assigned_uuid_exec.sh','tests/test_a5_hwg_matched_capture_20261030.py',
        'config/e0_chipmunk_input_video_sources_20261030_v1.json']}
    for root in [ROOT/'.runtime/code/LightX2V/lightx2v',ROOT/'.runtime/code/LightX2V/lightx2v_platform',ROOT/'.runtime/code/HiF4_Sp/HiFloat4',ROOT/'.runtime/code/Wan2.2-I2V-A14B-W4A4/quant']:
        for dp,dirs,names in os.walk(root):
            dirs[:]=[v for v in dirs if v not in ['.git','__pycache__']]
            for name in names:
                if Path(name).suffix in ['.py','.json','.so']:files.add(Path(dp)/name)
    with ThreadPoolExecutor(max_workers=8)as ex:
        return list(ex.map(lambda p:owner.bind(p,ROOT),sorted(files)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker-plan',type=Path,required=True);ap.add_argument('--private-output',type=Path,required=True);a=ap.parse_args()
    plan=owner.read(a.worker_plan);cases=owner.read(ROOT/'config/e0_chipmunk_input_video_sources_20261030_v1.json')['parts']
    owner.need(len(plan)==4 and sorted(v['part']for v in plan)==list(range(4)),'four fixed source parts required')
    sources=source_inventory();print('bound source files',len(sources),flush=True)
    a.private_output.mkdir(parents=True,exist_ok=False);summary=[]
    template=owner.read(ROOT/'config/e0_teal_wan_all160_video_v3_part0_20261029_pending_review_v1.json')
    for worker in sorted(plan,key=lambda x:x['part']):
        part=worker['part'];case=cases[part]['case'];scope=f'E0_A5_HWG_MATCHED_CAPTURE_20261030_PART{part}_V1'
        private_assets=a.private_output/f'assets_part{part}.json'
        assets={k:worker[k]for k in ['hf_root','cache_path','local_package_root','worker_profile']}
        assets['worker_profile_sha256']=owner.sha(Path(worker['worker_profile']))
        write_new(private_assets,assets,True)
        output=f'ffnds-derived/e0_a5_hwg_matched_capture_20261030/part{part}/attempt1/worker/results'
        protocol=ROOT/f'config/e0_a5_hwg_matched_capture_20261030_part{part}_frozen_v1.json'
        p={'id':scope,'status':'FROZEN_INPUTS_PENDING_INDEPENDENT_RECHECK_NOT_AUTHORIZED',
           'author_agent_id':'/root','part':part,'case':case,'seed':42,'cpu_affinity':worker['cpu_affinity'],
           'resources':owner.RES,'verification_resources':owner.VERIFY_RES,'payload_budget':owner.payload_budget(),
           'output_HF_relative':output,'private_assets_sha256':owner.sha(private_assets),
           'private_worker_profile_sha256':assets['worker_profile_sha256'],
           'model_view':owner.bind(ROOT/'.runtime/model_view.json',ROOT),
           'cache':{'bytes':56223524243,'sha256':'bfb400f94ce42760070bab5916721fd942b0864907bb4e4e5207c008dc4c3df3',
                    'numeric_family':'ORIGINAL_HIF4_FAKE_QDQ_BF16_FORCE_FP32_NATIVE_ACTIVATION_GROUP64'},
           'runtime_versions':template['runtime_versions'],'repo_sources':sources,
           'local_transformers':template['local_transformers'],
           'design':{'source_role':'FOUR_HISTORICALLY_EXPOSED_DEVELOPMENT_IMAGES_NO_BLIND_OR_VIDEO_QUALITY_CLAIM',
            'phases':['plain','observed'],'sites_per_phase':160,'steps':[1000,750,500,250],
            'expert_schedule':['high','high','low','low'],'grid':[21,29,52],
            'coordinates':'GLOBAL_ALIGNED_M64_ONE_TILE_EACH_TAU_SAME_ALL160',
            'fullM_saved_boundary':'ACTUAL_POSTQ1_NATIVE_V1_BEFORE_W1',
            'fullM_preQ1':'NOT_CAPTURED_NO_FULL_Q1_RECONSTRUCTION',
            'observed_rows':1344,'native_linear_calls_per_phase':1600,'native_QDQ_calls_per_phase':1600,
            'observer_control_ops_per_site_upper':{'additional_sample_GELU':1,'sample_index_select':24,'R_clone':2,'R_add':3,'Y_times_gate':1,'GPU_equal':6,'CPU_CUDA_rng_state_reads':6,'CPU_finite_tensor_checks':15,'fullV1_D2H':1,'sample_field_D2H':15,'W2_runtime_D2H_and_SHA':1,'W2_expected_CPU_SHA_max':1},
            'observer_ops_evidence':'SRC_BOUND_PLUS_ACTUAL_SAMPLE_GELU_AND_INDEX_COUNTS_NOT_PERFORMANCE',
            'video_geometry_inspection':owner.VIDEO_INSPECTION,
            'opaque_native_loads':{'pipeline_builds':1,'quantized_cache_loads':1,'internal_deserializer_and_loader_file_IO_bytes':'UNKNOWN_SEPARATE_FROM_EXPLICIT_FILE_SHA_VALIDATION_READS'},
            'neutral':'ALL_TEN_STATE_FILES_AND_COMPLETE_VIDEO_BYTE_EQUAL',
            'proxy_or_fitting_or_candidate_or_official_scoring':False,
            'replay_and_E0_E1_E2':'SEPARATELY_REVIEWED_FUTURE_SCOPES',
            'automatic_retry':False}}
        write_new(protocol,p)
        hf=Path(worker['hf_root']);out=hf/output;verify=out.parent.parent/'verification/worker/results'
        cpu=worker['cpu_affinity'];owner.need(len(cpu)==4 and len(set(cpu))==4,'CPU affinity')
        head=['env','FFNDS_WORKER_CONFIG='+worker['worker_profile'],'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','PATH=/usr/bin:/bin']
        common=[str(ROOT/'.venv/bin/python'),'-I','-B',str(ROOT/owner.SELF)]
        for kind in ['run','verify']:
            sid=scope+('_VERIFY'if kind=='verify'else'');jobpath=a.private_output/f'{kind}_part{part}.json'
            tail=[kind,'--protocol',str(protocol),'--job-spec',str(jobpath),'--private-assets',str(private_assets)]
            if kind=='run':
                argv=head+['bash',str(ROOT/'scripts/worker_teal_assigned_uuid_exec.sh'),str(worker['physical_gpu']),worker['GPU_UUID'],'taskset','-c',','.join(map(str,cpu))]+common+tail
                dest=out;limit=owner.RES['outer_seconds']
            else:
                head_verify=['env','FFNDS_WORKER_CONFIG='+worker['worker_profile'],'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PATH=/usr/bin:/bin','CUDA_VISIBLE_DEVICES=']
                argv=head_verify+['taskset','-c',str(cpu[0])]+common+tail;dest=verify;limit=owner.VERIFY_RES['outer_seconds']
            job={'id':sid,'protocol_path':str(protocol),'protocol_sha256':owner.sha(protocol),'cwd':str(ROOT),'argv':argv,'hf_root':str(hf),'output_dir':str(dest.parent),'run_json':str(dest/'RUN.json'),'external_timeout_seconds':limit}
            write_new(jobpath,job,True)
            summary.append({'id':sid,'protocol':owner.bind(protocol,ROOT),'job_path':str(jobpath),'job_spec_sha256':owner.sha(jobpath),'fingerprint':owner.canonical(p),'kind':kind})
    write_new(a.private_output/'BOUND_JOBS.json',summary,True)
    print(json.dumps([{'id':s['id'],'protocol_sha256':s['protocol']['sha256'],'job_spec_sha256':s['job_spec_sha256']}for s in summary],indent=2))


if __name__=='__main__':main()
