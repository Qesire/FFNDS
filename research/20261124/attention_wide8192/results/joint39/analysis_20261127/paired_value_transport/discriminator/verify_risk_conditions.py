"""CPU identities for history transport and downstream improvement.

Constructed arrays only. No Wan data, no SparsePR author execution, no GPU.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(1627)

def softmax(x):
    z = np.exp(x - x.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)

n, nk, d = 13, 19, 6
qa = rng.normal(size=(n, d))
qt = qa + .18 * rng.normal(size=(n, d))
ka = rng.normal(size=(nk, d))
kt = ka + .12 * rng.normal(size=(nk, d))
va = rng.normal(size=(nk, d))
transform = np.eye(d) + .2*rng.normal(size=(d, d))
vt = va @ transform + rng.normal(size=(1, d)) + .03*rng.normal(size=(nk, d))
la, lt = qa @ ka.T / np.sqrt(d), qt @ kt.T / np.sqrt(d)
pa, pt = softmax(la), softmax(lt)
mask = np.zeros((n, nk), dtype=bool)
for i in range(n):
    mask[i, np.argsort(pa[i])[-7:]] = True
sa = softmax(np.where(mask, la, -np.inf)) @ va
st = softmax(np.where(mask, lt, -np.inf)) @ vt
aa, at = pa @ va, pt @ vt
ha, ht = pa*(~mask), pt*(~mask)
ra, rt = ha.sum(axis=-1, keepdims=True), ht.sum(axis=-1, keepdims=True)
mua, mut = va.mean(axis=0, keepdims=True), vt.mean(axis=0, keepdims=True)
da, dt = va.std(axis=0), vt.std(axis=0)
xa, yt = (va-mua)/da, vt-mut
z = (aa-(1-ra)*sa-ra*mua)/da
mat = np.linalg.lstsq(xa[:12], yt[:12], rcond=None)[0]
base_old = (1-ra)*st + ra*mut + z*dt
base_new = (1-ra)*st + ra*mut + z@mat
route_change = (ht-ha) @ yt
mass_change = (ra-rt) * (st-mut)
value_defect = ha @ (yt-xa@mat)
factorization_error = np.max(np.abs((at-base_new)-(route_change+mass_change+value_defect)))
z_identity_error = np.max(np.abs(z-ha@xa))

# A fixed linear probe reader isolates the physical transport change.
probes = np.array([0, 3, 6, 9, 12])
phi = np.concatenate([np.ones((n, 1)), rng.normal(size=(n, 3))], axis=1)
pp = phi[probes]
reader = phi @ np.linalg.solve(pp.T@pp+.4*np.eye(4), pp.T)
def recover(base):
    out = base + reader @ (at[probes]-base[probes])
    out[probes] = at[probes]
    return out
old, new = recover(base_old), recover(base_new)
delta = z @ (mat-np.diag(dt))
delta_closed = delta-reader@delta[probes]
delta_closed[probes] = 0
probe_composition_error = np.max(np.abs((new-old)-delta_closed))

# Exact nonlinear downstream map, not a Jacobian approximation.
w1, w2 = rng.normal(size=(d, 9)), rng.normal(size=(9, d))
def downstream(x):
    u = (x-x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1,keepdims=True)+.5)
    return x + .1*np.tanh(u@w1)@w2
truth_tail, old_tail, new_tail = map(downstream, (at, old, new))
e0 = truth_tail-old_tail
step = new_tail-old_tail
gain_direct = np.square(e0).sum()-np.square(truth_tail-new_tail).sum()
gain_identity = 2*np.sum(e0*step)-np.square(step).sum()

# Leave-one-label-out formulas for a fixed feature/weight/penalty reader.
p, f, dout = 17, 7, 5
x = np.concatenate([np.ones((p,1)), rng.normal(size=(p,f-1))], axis=1)
weights = rng.uniform(.4, 2, size=p)
weights /= weights.mean()
penalty = np.diag([0]+[.8]*(f-1))
response, history = rng.normal(size=(2,p,dout))
normal = x.T@(weights[:,None]*x)+penalty
reader_p = x @ np.linalg.solve(normal, x.T*weights)
leverage = np.diag(reader_p)
e_loo = (response-reader_p@response)/(1-leverage[:,None])
h_loo = (history-reader_p@history)/(1-leverage[:,None])
explicit_e, explicit_h = [], []
for i in range(p):
    keep = np.arange(p)!=i
    xx, ww = x[keep], weights[keep]
    lift = x[i:i+1] @ np.linalg.solve(xx.T@(ww[:,None]*xx)+penalty, xx.T*ww)
    explicit_e.append(response[i]-lift@response[keep])
    explicit_h.append(history[i]-lift@history[keep])
explicit_e = np.concatenate(explicit_e, axis=0)
explicit_h = np.concatenate(explicit_h, axis=0)
eta = np.clip(np.sum(weights[:,None]*e_loo*h_loo)/(np.sum(weights[:,None]*h_loo*h_loo)+1e-12),0,1)

# Exact finite-distribution conditional risk decomposition.
# C identifies a coarse group; H identifies the within-group historical state.
target = np.array([[0.,1.],[2.,-1.],[3.,0.],[1.,2.]])
prob = np.array([.15,.25,.2,.4])
groups = np.array([0,0,1,1])
current_mean = np.empty_like(target)
for g in (0,1):
    idx = groups==g
    current_mean[idx] = np.sum(prob[idx,None]*target[idx],axis=0)/prob[idx].sum()
history_mean = target.copy()
current_risk = np.sum(prob[:,None]*(target-current_mean)**2)
history_risk = np.sum(prob[:,None]*(target-history_mean)**2)
info_gain = np.sum(prob[:,None]*(history_mean-current_mean)**2)

out = {
    "scope":"Synthetic CPU FP64 identities; no real-model effect or win inferred",
    "history_as_omitted_value_statistic_max_abs":float(z_identity_error),
    "attention_error_factorization_max_abs":float(factorization_error),
    "probe_corrected_transport_difference_max_abs":float(probe_composition_error),
    "nonlinear_downstream_gain_identity_max_abs":float(abs(gain_direct-gain_identity)),
    "constructed_downstream_gain":float(gain_direct),
    "loo_response_identity_max_abs":float(np.max(np.abs(e_loo-explicit_e))),
    "loo_history_identity_max_abs":float(np.max(np.abs(h_loo-explicit_h))),
    "constructed_loo_eta":float(eta),
    "conditional_bayes_risk_identity_max_abs":float(abs(current_risk-history_risk-info_gain)),
    "cost_at_Wan_shape":{
        "N":31668,"heads":40,"value_dim":128,"layers_retained":40,
        "sample_rows":256,
        "paired_anchor_value_BF16_GiB":256*128*40*40*2/2**30,
        "rank8_additional_transport_FLOPs_per_layer_hit":4*31668*128*8*40,
        "full128_additional_transport_FLOPs_per_layer_hit":2*31668*128*128*40,
        "scope":"Specified GEMMs/state only, excluding fit/factorization, gathers, casts, traffic, padding and kernel launch. No measured latency."
    },
    "remarks":[
        "LOO keeps feature normalization, weights and penalty fixed while removing one target row; it is not a full refit of the author's normalization or coverage policy.",
        "An eta chosen on LOO predictions minimizes that empirical score; independent source/trajectory evidence is needed for actual advantage.",
        "The exact nonlinear-tail identity holds on a common upstream background; free-running trajectories require separate measurements."
    ]
}
for key,value in out.items():
    if key.endswith("max_abs"):
        assert value < 1e-11,(key,value)
(ROOT/'RISK_CONDITIONS_VERIFIED.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
