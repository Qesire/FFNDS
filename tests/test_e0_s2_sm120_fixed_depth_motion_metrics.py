import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sm120_motion_test',ROOT/'tools/e0_s2_sm120_fixed_depth_motion_metrics.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class MotionTests(unittest.TestCase):
    def test_four_shards_three_images_and_twelve_clips_each(self):
        self.assertEqual(len(set(sum((list(x) for x in m.SHARDS),[]))),12)
        self.assertEqual(len(m.SHARDS),4)
        self.assertEqual(m.LIMITS['decoded_frames'],12*81)
        self.assertEqual(m.LIMITS['RAFT_forwards'],12*40)
        self.assertEqual(m.LIMITS['RAFT_iterations'],12*40*20)
        self.assertEqual(m.LIMITS['video_generations'],0)

    def test_paired_aggregation_and_zero_dense(self):
        plan={'shard':0};rows=[]
        for ordinal in m.SHARDS[0]:
            for index,mode in enumerate(m.MODES):
                scores=[float(index+1)]*40
                row={'ordinal':ordinal,'mode':mode,'scores':scores,'params':{'thres':10.875,'count_num':10},'official_binary':False}
                row['summary']=m.base.validate_raft(row);rows.append(row)
        summary=m.summarize(plan,rows)
        self.assertEqual(summary['independent_image_sources'],3)
        self.assertEqual(summary['source_equal_means'],dict(zip(m.MODES,(1.,2.,3.,4.))))
        self.assertEqual(summary['per_source'][0]['candidate_to_same_SM120_Dense_ratios'][m.MODES[1]]['value'],2.)
        for row in rows:
            if row['ordinal']==m.SHARDS[0][0] and row['mode']=='dense':
                row['scores']=[0.]*40;row['summary']=m.base.validate_raft(row)
        summary=m.summarize(plan,rows)
        self.assertEqual(summary['per_source'][0]['candidate_to_same_SM120_Dense_ratios'][m.MODES[1]]['status'],'ZERO_DENSE_MOTION')
        self.assertIsNone(summary['mean_of_source_ratios'][m.MODES[1]])

    def test_parent_actual_exit_and_generation_contract_with_small_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'repo';hf=Path(tmp)/'hf';root.mkdir();hf.mkdir()
            cases=[{'native_source_ordinal':o,'index':i,'source_id':f'source{o}',
                    'image':{'sha256':f'{i+1:064x}'},'prompt_sha256':f'{i+10:064x}','prompt':f'prompt{i}','seed':42}
                   for i,o in enumerate(m.SHARDS[0])]
            parent={'part':0,'id':'parent0','limits':{'x':1},'cases':cases,'run_ids':['control0','control1']}
            phases=[{'run_id':'control0'},{'run_id':'control1'}]
            for c in cases:
                for mode in m.video_parent.order(c):
                    ordinal=c['native_source_ordinal'];rid=f'{ordinal}_{mode}'
                    folder=hf/rid;folder.mkdir();(folder/'video.mp4').write_bytes(rid.encode())
                    states={}
                    for name in ('raw_noise_FP32','initial_noise','flow_step0','flow_step1','flow_step2','flow_step3','latent_step0','latent_step1','latent_step2','latent_step3'):
                        f=folder/(name+'.bin');f.write_bytes(name.encode())
                        states[name]={**m.base.binding(f,hf),'root':'hf','raw_tensor_sha256':('a'*64),'shape':[1],'dtype':'bfloat16'}
                    clip={'path':str(folder/'video.mp4'),'bytes':(folder/'video.mp4').stat().st_size,
                          'sha256':m.base.sha(folder/'video.mp4'),'frames':81,'size':[832,464],'fps':16}
                    phase={'id':'parent0','protocol_sha256':'PLACEHOLDER','run_id':rid,'mode':mode,
                           'numeric_path':'BF16_ORIGINAL','seed':42,'source_id':c['source_id'],
                           'image_sha256':c['image']['sha256'],'prompt_sha256':c['prompt_sha256'],
                           'native_source_ordinal':ordinal,'all160_FFN_covered':True,'completed_native_flow_steps':4,
                           'scheduler_generator_reset':True,'status':'PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE',
                           'states':states,'video':clip}
                    parent['run_ids'].append(rid);phases.append(phase)
            pp=root/'parent.json';pp.write_text(json.dumps(parent));pb=m.base.binding(pp,root)
            for phase in phases[2:]:
                phase['protocol_sha256']=pb['sha256'];gp=hf/phase['run_id']/'GENERATION.json'
                gp.write_text(json.dumps(phase));phase['generation_receipt']=m.base.binding(gp,hf)
            run={'part':0,'id':'parent0','protocol_sha256':pb['sha256'],'status':'COMPLETE_SM120_NEW12_FIXED_DEPTH_MATCHED_PART',
                 'actual_full160_videos':14,'actual_neutral_videos':2,'actual_new_source_videos':12,'actual_states':140,
                 'actual_flow_steps':56,'actual_FFN_positions':2240,'SM120_runtime_neutral_all10_and_video_passed':True,
                 'GPU_compute_capability':[12,0],'assigned_GPU_resources':{'violations':[]},
                 'operation_ledger':{'forbidden_attempts':[],'predebited':{'x':1},'observed':{'x':1}},'phases':phases}
            rp=hf/'RUN.json';rp.write_text(json.dumps(run));rb=m.base.binding(rp,hf)
            ep=hf/'EXIT.json';ep.write_text(json.dumps({'id':'parent0','protocol_sha256':pb['sha256'],'child_returncode':0,
                'process_exited':True,'timeout':False,'automatic_retry':False,'run_sha256':rb['sha256']}))
            proof={'protocol':pb,'RUN':rb,'exit':m.base.binding(ep,hf)}
            with mock.patch.object(m,'ROOT',root),mock.patch.object(m,'PARENTS','parent.json'),mock.patch.object(m.video_parent,'verify_closed') as verify:
                rows=m.check_parent(proof,hf,0)
                self.assertEqual(len(rows),12);self.assertEqual({x['ordinal'] for x in rows},set(m.SHARDS[0]))
                self.assertEqual(sum(x['mode']=='dense' for x in rows),3);verify.assert_called_once()
                changed=copy.deepcopy(run);changed['SM120_runtime_neutral_all10_and_video_passed']=False
                rp.write_text(json.dumps(changed));bad_run=m.base.binding(rp,hf)
                receipt=json.loads(ep.read_text());receipt['run_sha256']=bad_run['sha256'];ep.write_text(json.dumps(receipt))
                bad={**proof,'RUN':bad_run,'exit':m.base.binding(ep,hf)}
                with self.assertRaises(ValueError):m.check_parent(bad,hf,0)


    def test_result_identity_and_ratios_cannot_be_tampered(self):
        plan={'shard':0,'id':'id0','videos':[],'raft_assets':[
             {'path':'foo'+m.WEIGHT_SUFFIX,'sha256':m.WEIGHT_SHA,'bytes':1},
             {'path':'foo/vbench/third_party/RAFT/core/raft.py','sha256':'a'*64,'bytes':1}]}
        hf=Path('/tmp/synthetic-hf');rows=[]
        for ordinal in m.SHARDS[0]:
            for index,mode in enumerate(m.MODES):
                path=f'vid/{ordinal}_{mode}.mp4';v={'ordinal':ordinal,'mode':mode,'video':{'path':path,'sha256':f'{ordinal+index:064x}'}}
                plan['videos'].append(v);r={'ordinal':ordinal,'mode':mode,'video_sha256':v['video']['sha256'],
                  'video_path':str(hf/path),'status':'COMPLETE','scores':[float(index+1)]*40,
                  'params':{'thres':10.875,'count_num':10},'official_binary':False}
                r['summary']=m.base.validate_raft(r);rows.append(r)
        report={'id':'id0','status':m.STATUS,'RAFT_observations':rows,
                'source_summary':m.summarize(plan,rows),
                'official_dynamic_degree_only':[0.0,[{'video_path':r['video_path'],'video_results':False} for r in rows]],
                'actual_RAFT_checkpoint_loads':[plan['raft_assets'][0]],'actual_RAFT_source':plan['raft_assets'][1],
                'assigned_GPU_resources':{'violations':[],'sampled_isolation_pass':True}}
        for key in ('predebited','invoked','completed'):report[key]=copy.deepcopy(m.LIMITS)
        m.validate_result(plan,report,hf)
        for kind in ('mode','score','aggregate','work'):
            bad=copy.deepcopy(report)
            if kind=='mode':bad['RAFT_observations'][1]['mode']='dense'
            if kind=='score':bad['RAFT_observations'][1]['scores'][0]=100
            if kind=='aggregate':bad['source_summary']['source_equal_means']['dense']=99
            if kind=='work':bad['completed']['RAFT_forwards']=479
            with self.assertRaises(ValueError):m.validate_result(plan,bad,hf)

    def test_execution_monitor_and_worker_are_separate_bindings(self):
        import worker_identity
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);binary=root/'nvidia-smi';binary.write_bytes(b'new driver')
            private=root/'worker.json';private.write_bytes(b'config')
            plan={'execution_monitor_binary':{'path':str(binary),'bytes':binary.stat().st_size,'sha256':m.base.sha(binary)},
                  'execution_private_worker_config_sha256':m.base.sha(private),
                  'environment':{'nvidia_smi':{'sha256':'f'*64}}}
            with (mock.patch.object(m.shutil,'which',return_value=str(binary)), mock.patch.object(worker_identity,'require_worker'),
                  mock.patch.object(worker_identity,'private_config_path',return_value=private)):
                m.verify_execution_monitor(plan);m.verify_execution_worker(plan)
                binary.write_bytes(b'wrong driver')
                with self.assertRaises(ValueError):m.verify_execution_monitor(plan)
                private.write_bytes(b'changed')
                with self.assertRaises(ValueError):m.verify_execution_worker(plan)


    def test_preflight_byte_closure_covers_all_120_states(self):
        def b(path):return {'path':path,'bytes':1,'sha256':'a'*64}
        plan={'repo_sources':[b('code.py')], 'raft_assets':[b('raft.py')],
              'parents':[{'RUN':b('parent-run'),'exit':b('parent-exit')}], 'videos':[]}
        for i in range(12):
            plan['videos'].append({'generation':b(f'g{i}'),'video':b(f'v{i}'),
                'states':{str(j):b(f's{i}_{j}') for j in range(10)}})
        entries=m.all_bindings(plan)
        self.assertEqual(len(entries),148)
        self.assertEqual(len([b for root,b in entries if root=='HF' and b['path'].startswith('s')]),120)
        broken=copy.deepcopy(plan);broken['videos'][1]['states']['0']={**broken['videos'][0]['states']['0'],'sha256':'b'*64}
        with self.assertRaises(ValueError):m.all_bindings(broken)

    def test_decoder_short_video_records_actual_work(self):
        class Device:
            def __init__(self,n):self.n=n
            def read(self):self.n-=1;return self.n>=0,object() if self.n>=0 else None
            def release(self):pass
        class CV:VideoCapture=lambda self,path:Device(3)
        class Module:cv2=CV()
        report={};ledger=m.base.Ledger(report,limits={'decoded_frames':81})
        old=m.install_decoder_counter(Module,ledger)
        try:
            cap=Module.cv2.VideoCapture('x')
            for _ in range(4):cap.read()
            with self.assertRaises(ValueError):cap.release()
            self.assertEqual(report['completed']['decoded_frames'],3)
        finally:Module.cv2.VideoCapture=old

if __name__=='__main__':unittest.main()
