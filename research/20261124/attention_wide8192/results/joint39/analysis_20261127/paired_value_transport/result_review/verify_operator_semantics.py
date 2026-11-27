"""Source identity and NumPy algebra checks; does not execute Torch or Wan."""
from pathlib import Path
import hashlib
import json
import ast
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'experiments'
executed=json.loads((DATA/'EXECUTION.json').read_text())
end=json.loads((DATA/'E2E_EXECUTION.json').read_text())
hashes=[]
for name in ['attention_paired_value_ops.py','attention_paired_value_replay.py','attention_sparsepr_panel_control.py','attention_paired_value_end_to_end.py','attention_joint_action.py']:
    source=(DATA/name).read_bytes()
    ast.parse(source.decode(),filename=name)
    sha=hashlib.sha256(source).hexdigest()
    expected=[x['files'][name] for x in executed['jobs'].values() if name in x.get('files',{})]
    if name in end['school_verification']['inputs']:
        expected.append(end['school_verification']['inputs'][name])
    assert expected and all(x==sha for x in expected)
    hashes.append(dict(name=name,sha256=sha,execution_receipts=len(expected),match=True))

rng=np.random.default_rng(20261127)
p,d,n,j=64,128,512,256
s=rng.normal(size=(n,d)); s[:,0]=2
teacher=rng.normal(size=(p,d))
mean=s[:p].mean(0)
scale=np.sqrt(s[:p].var(0)+1e-6)
x=(s-mean)/scale
r=teacher-s[:p]; intercept=r.mean(0); centered=r-intercept
primal=np.linalg.solve(x[:p].T@x[:p]+np.eye(d),x[:p].T@centered)
dual=x[:p].T@np.linalg.solve(x[:p]@x[:p].T+np.eye(p),centered)
primal_dual=float(np.max(np.abs(x@(primal-dual))))
assert primal_dual<1e-11

u=rng.normal(size=(j,d)); y=rng.normal(size=(j,d)); diagonal=np.diag(np.exp(rng.normal(size=d)))
ridge=.001
delta=np.linalg.solve(u.T@u/j+ridge*np.eye(d),u.T@(y-u@diagonal)/j)
m=diagonal+delta
gradient=u.T@(u@m-y)/j+ridge*(m-diagonal)
normal_equation=float(np.max(np.abs(gradient)))
assert normal_equation<1e-11

four=json.loads((DATA/'FOURARM_204487.json').read_text())
conditions=[v['ridge_system_max_condition'] for r in four['records'].values() for v in r['sites'].values()]
out=dict(scope='Source hash/AST and NumPy FP64 algebra only; no Torch/Wan execution',source_hashes=hashes,
    shapes=dict(probes=p,channels=d,query_rows=n,value_fit_rows=j),
    v2_primal_dual_prediction_max_abs=primal_dual,
    paired_map_mean_objective_gradient_max_abs=normal_equation,
    actual_regularized_gram_max_condition_range=[min(conditions),max(conditions)],
    actual_condition_site_count=len(conditions))
(ROOT/'OPERATOR_SEMANTICS_VERIFIED.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
