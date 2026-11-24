"""Synthetic weights at actual Wan head dimensions; checks exact head algebra."""
import json
from pathlib import Path
import numpy as np

rng=np.random.default_rng(650064)
D,O=5120,64
eps=1e-6
W=rng.normal(size=(O,D))/np.sqrt(D)
scale=1+rng.normal(size=D)*.15
shift=rng.normal(size=D)*.1
bias=rng.normal(size=O)*.1
x=rng.normal(size=D)

def statistics(x):
    u=x-x.mean()
    return W@(scale*u),float(u@u/D)

def direct(x):
    u=x-x.mean()
    return W@(scale*u/np.sqrt(u@u/D+eps)+shift)+bias

q,var=statistics(x)
reconstructed=q/np.sqrt(var+eps)+W@shift+bias

# At fixed time condition, preserving q and variance is sufficient.
# Build a finite perturbation inside the nullspace of the centered head map,
# preserving the centered norm exactly by a rotation.
A=W*scale[None,:]
A=A-A.mean(-1,keepdims=True)
Q,_=np.linalg.qr(A.T,mode='reduced')
u=x-x.mean();row=Q@(Q.T@u);k=u-row
z=rng.normal(size=D);z=z-z.mean();z=z-Q@(Q.T@z)
z=z-k*(k@z)/(k@k);z=z*np.linalg.norm(k)/np.linalg.norm(z)
theta=.8
u_alt=row+np.cos(theta)*k+np.sin(theta)*z
x_alt=u_alt+x.mean()
q_alt,var_alt=statistics(x_alt)
result={
 'status':'SYNTHETIC_HEAD_ALGEBRA_AT_REAL_DIMENSIONS',
 'real_Wan_weights_used':False,
 'hidden_dim':D,'flow_patch_dim':O,'sufficient_statistic_dim':O+1,
 'readout_dimension_ratio':D/O,
 'head_from_65_statistics_max_error':float(np.max(np.abs(direct(x)-reconstructed))),
 'finite_hidden_change_norm':float(np.linalg.norm(x_alt-x)),
 'finite_hidden_relative_change':float(np.linalg.norm(x_alt-x)/np.linalg.norm(x)),
 'preserved_q_max_error':float(np.max(np.abs(q_alt-q))),
 'preserved_variance_absolute_error':abs(var_alt-var),
 'finite_head_change_max_error':float(np.max(np.abs(direct(x_alt)-direct(x)))),
}
assert result['head_from_65_statistics_max_error']<1e-12
assert result['finite_head_change_max_error']<1e-12
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
