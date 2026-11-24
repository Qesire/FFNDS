"""Deterministic mathematical-structure checks, not real Wan weight experiments."""
import json
from pathlib import Path
import numpy as np

rng=np.random.default_rng(7048192)
N,D,H,K,M=9,12,3,5,29
dh=D//H
eps=1e-6

def softmax(x):
    z=x-x.max(-1,keepdims=True)
    p=np.exp(z)
    return p/p.sum(-1,keepdims=True)

def ln(x):
    v=x-x.mean(-1,keepdims=True)
    return v/np.sqrt((v*v).mean(-1,keepdims=True)+eps)

def rms(x):
    return x/np.sqrt((x*x).mean(-1,keepdims=True)+eps)

def gelu(x):
    return .5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+.044715*x**3)))

def ln_secant_action(x,y,v):
    """Exact divided-difference operator applied to v; exact for v=y-x."""
    ux=x-x.mean(-1,keepdims=True); uy=y-y.mean(-1,keepdims=True)
    sx=np.sqrt((ux*ux).mean(-1,keepdims=True)+eps)
    sy=np.sqrt((uy*uy).mean(-1,keepdims=True)+eps)
    mid=(ux+uy)/2; pv=v-v.mean(-1,keepdims=True)
    a=(sx+sy)/(2*sx*sy)
    b=2/(D*(sx+sy)*sx*sy)
    return a*pv-b*mid*(mid*pv).sum(-1,keepdims=True)

class Block:
    def __init__(self):
        self.wq,self.wk,self.wv,self.wo=[rng.normal(size=(D,D))*.19 for _ in range(4)]
        self.cq,self.ck,self.cv,self.co=[rng.normal(size=(D,D))*.19 for _ in range(4)]
        self.w1=rng.normal(size=(D,M))*.21
        self.w2=rng.normal(size=(M,D))*.17
        self.b1=rng.normal(size=M)*.1;self.b2=rng.normal(size=D)*.1
        self.ctx=rng.normal(size=(K,D))
        self.mod=rng.normal(size=(6,D))*.28
        self.cgamma=1+rng.normal(size=D)*.12
        self.cbeta=rng.normal(size=D)*.1
        self.approx_u=rng.normal(size=(D,4))*.2
        self.approx_v=rng.normal(size=(4,D))*.2

    def self_attention(self,x):
        u=ln(x)*(1+self.mod[1])+self.mod[0]
        q=rms(u@self.wq).reshape(N,H,dh).transpose(1,0,2)
        k=rms(u@self.wk).reshape(N,H,dh).transpose(1,0,2)
        v=(u@self.wv).reshape(N,H,dh).transpose(1,0,2)
        p=softmax(q@k.transpose(0,2,1)/np.sqrt(dh))
        return (p@v).transpose(1,0,2).reshape(N,D)@self.wo

    def cross_data(self,x):
        u=ln(x)*self.cgamma+self.cbeta
        q=rms(u@self.cq).reshape(N,H,dh).transpose(1,0,2)
        k=rms(self.ctx@self.ck).reshape(K,H,dh).transpose(1,0,2)
        v=(self.ctx@self.cv).reshape(K,H,dh).transpose(1,0,2)
        s=q@k.transpose(0,2,1)/np.sqrt(dh)
        p=softmax(s)
        return q,k,v,s,p

    def cross(self,x):
        q,k,v,s,p=self.cross_data(x)
        return (p@v).transpose(1,0,2).reshape(N,D)@self.co

    def ffn(self,x):
        u=ln(x)*(1+self.mod[4])+self.mod[3]
        return (gelu(u@self.w1+self.b1)@self.w2+self.b2)*self.mod[5]

    def ffn_hat(self,x):
        return self.ffn(x)+.2*np.sin(x@self.approx_u)@self.approx_v

    def __call__(self,x):
        z=x+self.self_attention(x)*self.mod[2]
        u=z+self.cross(z)
        return u+self.ffn(u)

blocks=[Block() for _ in range(3)]
x=rng.normal(size=(N,D))
head=rng.normal(size=(D,4))*.3
shift=rng.normal(size=(N,1))*3

def suffix(x):
    for b in blocks:x=b(x)
    return ln(x)@head,x

out,y=suffix(x);out_shift,y_shift=suffix(x+shift)
out_scale,_=suffix(1.3*x)

b=blocks[0]
z=x+b.self_attention(x)*b.mod[2]
ea=rng.normal(size=(N,D))*.35
z1=z+ea
u0=z+b.cross(z);u1=z1+b.cross(z1)
b00=u0+b.ffn(u0);b10=u1+b.ffn(u1)
b01=u0+b.ffn_hat(u0);b11=u1+b.ffn_hat(u1)
e_attention=b10-b00;e_ffn=b01-b00
interaction=b11-b10-b01+b00

# Cross-attention exact finite transport using logarithmic-mean probabilities.
q0,k,v,s0,p0=b.cross_data(z)
q1,_,_,s1,p1=b.cross_data(z1)
log0=np.log(p0);log1=np.log(p1)
dlog=log1-log0
ell=np.divide(p1-p0,dlog,out=(p0+p1).copy()/2,where=np.abs(dlog)>1e-12)
eta=ell.sum(-1,keepdims=True);pi=ell/eta
ds=(q1-q0)@k.transpose(0,2,1)/np.sqrt(dh)
ds_centered=ds-(pi*ds).sum(-1,keepdims=True)
delta_cross=((eta*pi*ds_centered)@v).transpose(1,0,2).reshape(N,D)@b.co

# Exact LN + GELU finite secant action for the current FFN.
du=u1-u0
dm=ln_secant_action(u0,u1,du)*(1+b.mod[4])
h0=(ln(u0)*(1+b.mod[4])+b.mod[3])@b.w1+b.b1
h1=(ln(u1)*(1+b.mod[4])+b.mod[3])@b.w1+b.b1
dhid=h1-h0
beta=np.divide(gelu(h1)-gelu(h0),dhid,out=np.zeros_like(dhid),where=dhid!=0)
delta_ffn=((dm@b.w1)*beta)@b.w2*b.mod[5]
transported=ea+delta_cross+delta_ffn

# A fixed FFN output subspace can compensate exactly its reachable projection.
R=rng.normal(size=(D,6))
T=rng.normal(size=(7,D))
e=rng.normal(size=D)
coeff=-np.linalg.pinv(T@R)@(T@e)
res=T@(e+R@coeff)
projection_res=(np.eye(7)-(T@R)@np.linalg.pinv(T@R))@(T@e)

def maxabs(a):return float(np.max(np.abs(a)))
result={
 'status':'SYNTHETIC_ALGEBRA_AND_ARCHITECTURE_STRUCTURE_CHECKS',
 'real_Wan_model_execution':False,
 'shape':{'N':N,'D':D,'heads':H,'text_keys':K,'ffn_hidden':M,'blocks':len(blocks)},
 'LN_secant_max_error':maxabs(ln_secant_action(u0,u1,u1-u0)-(ln(u1)-ln(u0))),
 'cross_finite_transport_max_error':maxabs(delta_cross-(b.cross(z1)-b.cross(z))),
 'FFN_finite_transport_max_error':maxabs(delta_ffn-(b.ffn(u1)-b.ffn(u0))),
 'full_block_attention_transport_max_error':maxabs(transported-(b10-b00)),
 'four_arm_decomposition_max_error':maxabs((b11-b00)-(e_attention+e_ffn+interaction)),
 'interaction_norm':float(np.linalg.norm(interaction)),
 'suffix_translation_equivariance_max_error':maxabs(y_shift-y-shift),
 'terminal_head_translation_invariance_max_error':maxabs(out_shift-out),
 'terminal_head_scaling_difference_norm':float(np.linalg.norm(out_scale-out)),
 'reachable_correction_projection_identity_max_error':maxabs(res-projection_res),
 'reachable_correction_original_task_error_norm':float(np.linalg.norm(T@e)),
 'reachable_correction_residual_task_error_norm':float(np.linalg.norm(res)),
}
assert max(result[k] for k in result if k.endswith('max_error'))<1e-12
assert result['terminal_head_scaling_difference_norm']>1e-3
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
