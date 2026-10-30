import hashlib,json,struct,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_chipmunk_p64_local_prefix_v2_owner as owner
import e0_chipmunk_p64_local_prefix_v2_math as math
import e0_chipmunk_p64_local_prefix_v2_jobs as jobs

class Tests(unittest.TestCase):
 def test_numeric_group_crosses_real_json_persistence_boundary(self):
  dense=np.ones((2,5120),dtype=np.float32);shadow=dense*2;student=dense*3
  value=math.ratios(math.aggregate(dense,shadow,student,np.ones(5120),np.zeros(2,dtype=int),np.ones(2),1)[0])
  self.assertIs(type(value['identity_within_tolerance']),bool)
  self.assertIs(type(value['gate_identity_within_tolerance']),bool)
  self.assertIs(type(value['identity_tolerance']),float)
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'worker'/'results'/'RUN.json';path.parent.mkdir(parents=True)
   report={'sites':[{'groups':{'selector':[value],'eval':[value],'full_M':[value]}}]}
   owner.atomic(path,report)
   self.assertEqual(json.loads(path.read_text()),report)
   report.update(status='FAILED_NO_RETRY',error='fixture exercises second persistence')
   owner.atomic(path,report)
   self.assertEqual(json.loads(path.read_text())['status'],'FAILED_NO_RETRY')
 def test_signed_cross_and_identity(self):
  dense=np.array([[1,2],[3,4]],dtype=np.float32);shadow=dense+np.array([[2,-1],[1,1]],dtype=np.float32);student=shadow+np.array([[-1,1],[3,-1]],dtype=np.float32)
  groups=np.array([0,0]);weights=np.ones(2);v=math.aggregate(dense,shadow,student,np.ones(2),groups,weights,1)[0];z=math.ratios(v)
  self.assertLess(v['cross2'],0);self.assertAlmostEqual(z['identity_residual'],0);self.assertAlmostEqual(z['gate_identity_residual'],0)
 def test_zero_denominator_not_infinite(self):
  a=np.zeros((1,2),dtype=np.float32);v=math.ratios(math.aggregate(a,a,a,np.ones(2),np.zeros(1,dtype=int),np.ones(1),1)[0]);self.assertIsNone(v['T2_over_dense2'])
 def test_fp64_vector_identity_is_not_bit_exact_at_extreme_exponents(self):
  dense=np.array([[0]],dtype=np.float32);shadow=np.array([[2.0**100]],dtype=np.float32);student=np.array([[1]],dtype=np.float32)
  value=math.ratios(math.aggregate(dense,shadow,student,np.ones(1),np.zeros(1,dtype=int),np.ones(1),1)[0])
  self.assertEqual(value['vector_residual_max_abs'],1.0);self.assertEqual(value['vector_tolerance_excess_max'],0.0)
 def test_targeted_raw_hash_and_header(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.safetensors';raw=b'\0\0'*(2*5120);other=b'uncharged-other'
   header={'irrelevant':{'dtype':'U8','shape':[len(other)],'data_offsets':[0,len(other)]},'current_dense_Y':{'dtype':'BF16','shape':[2,5120],'data_offsets':[len(other),len(other)+len(raw)]}}
   h=json.dumps(header,separators=(',',':')).encode();p.write_bytes(struct.pack('<Q',len(h))+h+other+raw)
   item={'file':{'path':'x.safetensors','bytes':p.stat().st_size},'tensors':{'current_dense_Y':{'dtype':'bfloat16','shape':[2,5120],'data_offsets':[len(other),len(other)+len(raw)],'raw_sha256':hashlib.sha256(raw).hexdigest()}}}
   debits=[]
   def charge(kind,n,fn,detail=None):debits.append((kind,n));return fn()
   data=owner.read_slice(Path(d),item,['current_dense_Y'],2,charge)
   self.assertEqual(data['current_dense_Y'],raw);self.assertEqual(sum(n for k,n in debits if k=='Y_payload'),len(raw));self.assertEqual(sum(n for k,n in debits),8+len(h)+len(raw))
   item['tensors']['current_dense_Y']['raw_sha256']='0'*64
   with self.assertRaises(ValueError):owner.read_slice(Path(d),item,['current_dense_Y'],2,charge)
 def test_missing_parent_fails_closed(self):
  with patch.object(owner,'parent_proof',side_effect=ValueError('not closed')):
   with self.assertRaises(ValueError):owner.draft(0,Path('/no-profile'),[0,1])
 def test_exact_read_budget(self):
  self.assertEqual(owner.DATA_BYTES,4284579840);self.assertLess(owner.DATA_BYTES,owner.RES['read_bytes_max'])
 def test_metadata_bootstrap_charges_repeated_reads_and_restores(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'fixture.json';p.write_text('{"v":1}')
   old=owner.native.read
   with owner.MetadataBudget() as budget:
    owner.native.read(p);owner.native.read(p)
   self.assertEqual(budget.pre,2*p.stat().st_size);self.assertEqual(budget.done,budget.pre);self.assertEqual(budget.calls,2);self.assertIs(owner.native.read,old)
 def test_bootstrap_failure_persists_predebit_and_unknown(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'fixture.json';p.write_text('{"v":1}');receipt=Path(d)/'STARTUP.json'
   def failed_read(path):raise OSError('fixture read failure')
   with patch.object(owner.native,'read',failed_read):
    budget=owner.MetadataBudget(lambda:receipt.write_text(json.dumps({'pre':budget.pre,'done':budget.done,'inflight':budget.inflight})))
    with self.assertRaises(OSError):
     with budget:owner.native.read(p)
   saved=json.loads(receipt.read_text());self.assertEqual(saved['pre'],p.stat().st_size);self.assertEqual(saved['done'],0);self.assertEqual(saved['inflight']['completed'],'UNKNOWN')
 def test_stream_triplet_chunks_hash_and_alias_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   items=[]
   for i in range(3):
    raw=bytes([i,0])*(5*4);header={'y':{'dtype':'BF16','shape':[5,4],'data_offsets':[0,len(raw)]}};h=json.dumps(header,separators=(',',':')).encode();p=Path(d)/f'{i}.safetensors';p.write_bytes(struct.pack('<Q',len(h))+h+raw)
    items.append({'file':{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()},'tensors':{'y':{'dtype':'bfloat16','shape':[5,4],'data_offsets':[0,len(raw)],'raw_sha256':hashlib.sha256(raw).hexdigest()}}})
   chunks=[];deb=[]
   def charge(kind,n,fn,detail=None):deb.append((kind,n));return fn()
   owner.stream_triplet(Path(d),items,['y']*3,5,4,2,charge,lambda lo,hi,values:chunks.append((lo,hi,[len(v) for v in values])))
   self.assertEqual(chunks,[(0,2,[16]*3),(2,4,[16]*3),(4,5,[8]*3)]);self.assertEqual([n for k,n in deb if k=='Y_payload'],[120])
   with self.assertRaises(ValueError):owner.stream_triplet(Path(d),[items[0],items[0],items[2]],['y']*3,5,4,2,charge,lambda *x:None)
   items[2]['file']['sha256']='0'*64
   with self.assertRaises(ValueError):owner.stream_triplet(Path(d),items,['y']*3,5,4,2,charge,lambda *x:None)
   items[2]['file']['sha256']=hashlib.sha256((Path(d)/items[2]['file']['path']).read_bytes()).hexdigest()
   trail=Path(d)/items[2]['file']['path'];trail.write_bytes(trail.read_bytes()+b'!');items[2]['file']['bytes']=trail.stat().st_size;items[2]['file']['sha256']=hashlib.sha256(trail.read_bytes()).hexdigest()
   with self.assertRaises(ValueError):owner.stream_triplet(Path(d),items,['y']*3,5,4,2,charge,lambda *x:None)
   trail.write_bytes(trail.read_bytes()[:-1]);items[2]['file']['bytes']=trail.stat().st_size;items[2]['file']['sha256']=hashlib.sha256(trail.read_bytes()).hexdigest()
   items[1]['tensors']['y']['raw_sha256']='0'*64
   with self.assertRaises(ValueError):owner.stream_triplet(Path(d),items,['y']*3,5,4,2,charge,lambda *x:None)
 def test_parent_verifier_rejects_mixed_run_and_exit(self):
  run={'path':'run','bytes':1,'sha256':'a'};ex={'path':'exit','bytes':1,'sha256':'b'}
  owner.require_verified_parent_bindings({'RUN':run,'exit':ex},run,ex)
  with self.assertRaises(ValueError):owner.require_verified_parent_bindings({'RUN':dict(run,sha256='other'),'exit':ex},run,ex)
  with self.assertRaises(ValueError):owner.require_verified_parent_bindings({'RUN':run,'exit':dict(ex,sha256='other')},run,ex)
 def test_receipt_wait_allows_delayed_creation_and_reads_nothing(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'START_RECEIPT.json';now=[100.0];sleeps=[]
   def sleep(value):
    sleeps.append(value);now[0]+=value
    if len(sleeps)==3:p.write_text('fixture')
   result=owner.wait_for_start_receipt(p,100.0,clock=lambda:now[0],sleep=sleep)
   self.assertEqual(result['checks'],4);self.assertAlmostEqual(result['elapsed_seconds'],0.15)
   self.assertEqual(p.read_text(),'fixture')
 def test_receipt_wait_times_out_and_respects_total_deadline(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'missing';now=[3599.9]
   def sleep(value):now[0]+=value
   with self.assertRaises(owner.ReceiptWaitTimeout) as caught:
    owner.wait_for_start_receipt(p,0,clock=lambda:now[0],sleep=sleep)
   self.assertLessEqual(caught.exception.elapsed,0.1000001);self.assertLessEqual(now[0],3600.0)
 def test_receipt_wait_uses_at_most_thirty_seconds(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'missing';now=[0.0]
   def sleep(value):now[0]+=value
   with self.assertRaises(owner.ReceiptWaitTimeout) as caught:
    owner.wait_for_start_receipt(p,0,clock=lambda:now[0],sleep=sleep)
   self.assertAlmostEqual(caught.exception.elapsed,30.0);self.assertGreaterEqual(caught.exception.checks,599);self.assertLessEqual(caught.exception.checks,601)
 def test_pair_file_identity_rejects_hardlink(self):
  import os
  with tempfile.TemporaryDirectory() as d:
   a=Path(d)/'a';b=Path(d)/'b';a.write_bytes(b'abc');os.link(a,b)
   with self.assertRaises(ValueError):owner.require_distinct_files(Path(d),[{'path':'a','bytes':3},{'path':'b','bytes':3}])
 def test_jobs_cpu_only_and_exact_timeout(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'protocol';p.write_text('{}');profile=Path(d)/'profile';profile.write_text('fixture')
   plan={'id':'fixture','part':0,'cpu_affinity':[1,2],'output_HF_relative':'fixture/output',
         'execution_private_worker_config_sha256':owner.sha(profile)}
   with patch.object(owner,'validate',return_value=True):spec=jobs.job(plan,p,profile,Path(d)/'job')
   self.assertEqual(spec['external_timeout_seconds'],3720);self.assertIn('CUDA_VISIBLE_DEVICES=',spec['argv']);self.assertIn('OMP_NUM_THREADS=2',spec['argv']);self.assertNotIn('GPU',spec['argv'])

if __name__=='__main__':unittest.main()
