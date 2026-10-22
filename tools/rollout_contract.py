"""No-torch registration and finite debit guard for the paired R15 smoke."""
import contextlib
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
METHOD='R15V0_MEAN_CONTRAST_PAIRED_ROLLOUT'
STATE='R4_CLOSED_PAIRED_FREE_ROLLOUT_SMOKE_REGISTERED'
CAPS={'full_generations':2,'offline_weight_factorizations':2,'training_steps':0}


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def atomic(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def validate_registration(registration,root):
    errors=[]
    try:
        pp=root/registration['protocol'];p=read(pp)
        if (p['id']!=METHOD or p['status']!='REGISTERED_PAIRED_FULL_FREE_ROLLOUT_SMOKE'
                or sha(pp)!=registration['protocol_sha256'] or p['caps']!=CAPS
                or p['positions']!=[[s,20] for s in range(4)] or p['source_index']!=0
                or p['source_family']!='calibration24' or p['rank']!=256
                or p['factorization']!={'algorithm':'torch.svd_lowrank','q':256,'niter':2,'seed':20261023}
                or p['expected_grid']!=[21,29,52] or p['automatic_followup'] is not False):
            errors.append('R15 paired smoke scope/protocol changed')
        auth_path=root/p['authorization']['path'];auth=read(auth_path)
        if (sha(auth_path)!=p['authorization']['sha256'] or auth['status']!='EXPLICIT_EXACT_SCOPE_USER_AUTHORIZATION'
                or auth['method_id']!=METHOD or auth['caps']!=CAPS or not auth['user_reply'].strip()):
            errors.append('R15 paired smoke lacks direct exact-scope user authority')
        if (p['arms']!=['dense_DQ','mean_contrast_candidate'] or p['max_GPUs']!=1
                or p['resources']['new_output_bytes_max']!=1<<28 or p['reset_between_steps'] is not False):
            errors.append('R15 free-video pairing/resource contract changed')
        for name,value in p['source_sha256'].items():
            if sha(root/name)!=value:errors.append('R15 paired source changed: '+name)
        for name,value in p['input_sha256'].items():
            if sha(root/name)!=value:errors.append('R15 paired input/asset metadata changed: '+name)
    except (OSError,ValueError,KeyError,TypeError) as e:
        errors.append('Invalid R15 paired registration: '+type(e).__name__)
    return errors


def require_registration():
    result=subprocess.run([sys.executable,str(ROOT/'tools/check_research_routes.py'),'--method-id',METHOD],
                          capture_output=True,text=True)
    if result.returncode:raise RuntimeError('No authorized paired R15 rollout; torch/model/CUDA not imported')
    registration=read(ROOT/'research/CURRENT_DECISION.json')['registered_rollout']
    errors=validate_registration(registration,ROOT)
    if errors:raise RuntimeError('; '.join(errors))
    return read(ROOT/registration['protocol'])


class Budget:
    def __init__(self,root):
        self.path=Path(root)/'BUDGET.json'
        self.lock=(Path(root)/'BUDGET.lock').open('a')
        fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if self.path.exists():raise RuntimeError('No implicit full-video/factorization retry')
        self.state={'method_id':METHOD,'caps':CAPS,'used':{k:0 for k in CAPS},'debits':[]}
        atomic(self.path,self.state)

    @contextlib.contextmanager
    def call(self,name):
        charges={'factors':{'offline_weight_factorizations':2},
                 'dense_DQ':{'full_generations':1},'mean_contrast_candidate':{'full_generations':1}}
        if name not in charges or any(r['name']==name for r in self.state['debits']):
            raise RuntimeError('Unregistered or already debited paired operation')
        charge=charges[name]
        used={k:self.state['used'][k]+charge.get(k,0) for k in CAPS}
        if any(used[k]>CAPS[k] for k in CAPS):raise RuntimeError('Paired rollout cap exceeded')
        row={'name':name,'charge':charge,'status':'DEBITED_BEFORE_EXECUTION'}
        self.state['used']=used;self.state['debits'].append(row);atomic(self.path,self.state)
        try:yield row
        except BaseException:
            row['status']='FAILED_DEBIT_PRESERVED';atomic(self.path,self.state);raise
        else:row['status']='COMPLETED';atomic(self.path,self.state)

    def close(self):self.lock.close()
