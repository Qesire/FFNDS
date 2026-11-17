"""Admission policy synthetic fixture; no generated research media."""
import json,tempfile,unittest,hashlib,zipfile
from pathlib import Path
from unittest.mock import patch
import dense_slot_admission as mod
class AdmissionTest(unittest.TestCase):
 def test_latent_storage_hash(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'latent.pt';data=bytes(16*21*58*104*2)
   with zipfile.ZipFile(path,'w') as z:z.writestr('archive/data/0',data)
   rec={'shape':[16,21,58,104],'dtype':'torch.bfloat16','sha256':hashlib.sha256(data).hexdigest()}
   self.assertEqual(mod.verify_latent_storage(path,rec),rec['sha256'])
   rec['sha256']='0'*64
   with self.assertRaises(ValueError):mod.verify_latent_storage(path,rec)
 def test_two_complete_nine_video_only_then_missing(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);v2=root/'v2';new=root/'new';(v2/'output'/'bank').mkdir(parents=True);(new/'output').mkdir(parents=True);(new/'receipt').mkdir()
   write=lambda p,x:Path(p).write_text(json.dumps(x)); v2sha='v'*64;nsha='n'*64
   write(v2/'EXIT.json',{'exit_code':0,'scope_sha256':v2sha,'allocation':{'job_id':'1'}});write(v2/'RUN.json',{'scope_sha256':v2sha,'allocation':{'job_id':'1'}});write(v2/'SHELL_EXIT.json',{'shell_exit_code':0,'job_id':'1'})
   write(new/'receipt'/'EXIT.json',{'exit_code':0,'slurm_job_id':'2'});write(new/'receipt'/'RUN.json',{'scope_sha256':nsha,'job_id':'2'})
   sched=[{'timestep':float(i),'sigma':float(i)/4} for i in range(4)];vs={'checkpoints':{'high':{'sha256':'h'},'low':{'sha256':'l'}}};ns={'schedule':sched};conds={s:{'source_cluster_id':s,'image_sha256':'i','prompt_sha256':'p'} for s in mod.DEV};neutral={s:{'initial_noise_equal':True,'decoded_equal':True,'flows_equal':[True]*4,'latents_equal':[True]*4,'time_sigma_equal':[True]*4} for s in mod.NEW};write(new/'output'/'SUMMARY.json',{'status':'COMPLETE','per_source':neutral})
   bank={};noise={};rgb='d'*64
   for source in mod.DEV:
    c=conds[source];base=new/'output' if source in mod.NEW else v2/'output';video=base/(source+('__full_u.mp4' if source in mod.NEW else '__tap.mp4'));video.write_bytes(source.encode());vsha=mod.sha(video)
    if source in mod.NEW:
     latent=base/(source+'.pt');latent.write_bytes(b'latent');steps=[{'step':i,**sched[i]} for i in range(4)];steps[-1]['final_latent_artifact']={'path':latent.name,'sha256':mod.sha(latent)};steps[-1]['latent']={'sha256':'dummy'}
     write(base/(source+'__full_u.json'),{'source':source,'arm':'full_u','initial_noise':{'sha256':'noise'},'image_sha256':'i','prompt_sha256':'p','dispatch':[{'step':i,'expert':0 if i<2 else 1} for i in range(4)],'steps':steps,'video':{'mp4_sha256':vsha,'decoded_rgb24_sha256':rgb}})
    else:
     noise[source]={'sha256':'noise'};bank[source]={**c,'sites':160,'initial_noise_sha256':'noise'};d=base/'bank'/source;d.mkdir()
     for i in range(4):write(d/f'{source}__seed42__s{i:02d}__b04.pt.json',{'source':source,'step':i,'timestep':sched[i]['timestep'],'checkpoint_sha256':'h' if i<2 else 'l'})
     write(base/f'CAPTURE_{source}.json',{'record_id':source,'arm':'tap','initial_noise':noise[source],'expert_dispatch':[{'step':i,'expert':0 if i<2 else 1} for i in range(4)],'latent_digests':[{'step':i} for i in range(4)],'mp4_sha256':vsha,'decoded_rgb24_sha256':rgb})
   write(v2/'output'/'BANK_MANIFEST.json',bank);write(v2/'output'/'INITIAL_NOISE_IDENTITIES.json',noise);write(v2/'output'/'SUMMARY.json',{'status':'V2_CAPTURE_COMPLETE','neutral_summary':'PASS','bank_manifest_sha256':mod.sha(v2/'output'/'BANK_MANIFEST.json')})
   spec={'v2_scope_sha256':v2sha,'stage1_scope_sha256':nsha}
   with patch.object(mod,'verify_common',return_value=(v2,new,vs,ns,conds,Path('fake'),{})), patch.object(mod,'verify_latent_storage',return_value='dummy'):
    out=mod.audit(spec,decode=lambda *_:{'decoded_rgb24_sha256':rgb})
    self.assertEqual((out['complete_main_slots'],out['quality_eligible_videos'],len(out['missing_main_sources'])),(2,11,9))
    (v2/'output'/'vbi_1040__tap.mp4').unlink()
    out=mod.audit(spec,decode=lambda *_:{'decoded_rgb24_sha256':rgb})
    self.assertEqual((out['complete_main_slots'],out['quality_eligible_videos']),(2,10))
if __name__=='__main__':unittest.main()





