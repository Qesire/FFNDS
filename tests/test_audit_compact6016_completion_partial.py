import copy
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import audit_compact6016_completion_partial as a


class AuditTests(unittest.TestCase):
    def test_termination_is_not_exit(self):
        r={'schema':'ffnds.completion.termination_observation.v1','original_protocol_sha256':'x',
           'parent_returncode':'UNRECOVERED','observed_utc':'2026-10-28T00:00:00Z',
           'scope':'ALL_FOUR_PARTS_BOTH_WORKERS','same_name_processes':0,'assigned_gpu_processes':0,'evidence_type':'READ_ONLY_OBSERVATION'}
        a.check_termination(r,'x')
        for k,v in [('parent_returncode',0),('same_name_processes',1),('assigned_gpu_processes',1),('scope','PARTIAL')]:
            with self.subTest(k=k),self.assertRaises(ValueError): a.check_termination({**r,k:v},'x')

    def test_header_contract_and_corrupt_offsets(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test';h={};pos=0
            for name,(dtype,shape) in a.SHAPES.items():
                n=1
                for x in shape:n*=x
                size=n*{'BF16':2,'F32':4,'I64':8}[dtype];h[name]={'dtype':dtype,'shape':shape,'data_offsets':[pos,pos+size]};pos+=size
            def save():
                raw=json.dumps(h).encode()
                with p.open('wb') as f:f.write(struct.pack('<Q',len(raw)));f.write(raw);f.truncate(8+len(raw)+pos)
            save(); self.assertEqual(a.verify_header(p)['tensors'],12)
            h['W1']['data_offsets'][0]+=1;save()
            with self.assertRaises(ValueError):a.verify_header(p)

    def test_parent_rejects_wrong_exit_or_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parts=[];records={}
            for i in range(4):
                directory=root/f'p{i}';directory.mkdir(); part={'part':i}
                functions=[{'expert':e,'block':b} for e,b in a.completion.functions_for(i)[:8]]
                ids=a.completion.run_ids(i)[:[14,14,13,13][i]]
                cells=[{'run_id':ident,'expert':functions[j//2]['expert'],'block':functions[j//2]['block']} for j,ident in enumerate(ids)]
                budget={k:0 for k in a.completion.PART_LIMITS}
                base={'id':a.completion.ID,'part':i,'protocol_sha256':'x'}
                recs={'run':{**base,'status':'FAILED_NO_RETRY','error_type':'BrokenPipeError','outer_process_completion':'REQUIRES_LAUNCHER_PROCESS_EXIT_RECEIPT','functions':functions,'cells':cells,'predebited':budget,'observed':budget},'budget':{**base,'predebited':budget,'observed':budget},'attempt':{**base,'automatic_retry':False}}
                for key,obj in recs.items():
                    p=directory/(key+'.json');p.write_text(json.dumps(obj));part[key]={'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':a.sha(p)}
                parts.append(part);records[i]=recs
            plan={'parts':parts,'original_protocol':{'sha256':'x'}}
            with patch.object(a.completion,'_completed_cell',side_effect=lambda hf,entry,function,**kw: entry):
                self.assertEqual(len(a.parent_records(root,plan)),54)
                (root/'p0'/'LAUNCH_EXIT.json').write_text('{}')
                with self.assertRaises(ValueError):a.parent_records(root,plan)

    def test_draft_not_authorized(self):
        with self.assertRaises(ValueError):a.authorize({'id':a.ID,'assets_finalized':False},'x')


if __name__=='__main__':unittest.main()
