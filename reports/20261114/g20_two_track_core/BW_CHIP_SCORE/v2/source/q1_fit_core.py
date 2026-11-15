"""Finite-iterate matrix-prior fitting primitives. Inputs are FP32 CUDA tensors."""
import torch

class Problem:
    def __init__(self, w1, w2, a0, sources, ridge=1e-3, shared=None):
        self.w1,self.w2,self.a0,self.sources,self.ridge=w1.float(),w2.float(),a0.float(),sources,ridge
        self.za=self.a0.square().sum()+1e-12
        self.z=sum(((s['y']*s['g']).square().sum()/s['y'].shape[0]) for s in sources)/len(sources)+1e-12
        for s in self.sources:
            if 'h' not in s: s['h']=s['x']@self.w1.T
        if shared is None:
            self.gram=(self.w1@self.w1.T)*(self.w2.T@self.w2)
            self.k0=((self.w1@self.a0)*self.w2.T).sum(1)
        else:
            self.gram,self.k0=shared.gram,shared.k0
    def matrix(self,alpha):
        return (self.w1.T*alpha)@self.w2.T
    def objective(self,a):
        data=sum((((s['x']@a-s['y'])*s['g']).square().sum()/s['x'].shape[0]) for s in self.sources)/len(self.sources)/self.z
        return data+self.ridge*(a-self.a0).square().sum()/self.za
    def normal(self,v,hidden=False):
        out=self.ridge*(self.gram@v)/self.za
        for s in self.sources:
            h=s['h']
            if hidden:
                out=out+(h.square().sum(0)*v)/(len(self.sources)*h.shape[0]*self.hz)
            else:
                out=out+(h*((((h*v)@self.w2.T)*s['g'].square())@self.w2)).sum(0)/(len(self.sources)*h.shape[0]*self.z)
        return out
    def fit(self,initial,maxiter,tol,hidden=False):
        if hidden:
            self.hz=sum((s['dh'].square().sum()/s['dh'].shape[0]) for s in self.sources)/len(self.sources)+1e-12
        rhs=self.ridge*self.k0/self.za
        for s in self.sources:
            h=s['h']
            t=s['dh'] if hidden else (s['y']*s['g'].square())@self.w2
            rhs=rhs+(h*t).sum(0)/(len(self.sources)*h.shape[0]*(self.hz if hidden else self.z))
        diag=self.ridge*self.gram.diagonal()/self.za
        for s in self.sources:
            h=s['h']
            term=h.square() if hidden else h.square()*(s['g'].square()@self.w2.square())
            diag=diag+term.sum(0)/(len(self.sources)*h.shape[0]*(self.hz if hidden else self.z))
        diag=diag.clamp_min(torch.finfo(diag.dtype).tiny)
        a=initial.clone(); r=rhs-self.normal(a,hidden); r0=r.norm().item(); iterations=0
        z=r/diag; p=z.clone(); rz=r@z; history=[{'iteration':0,'true_residual':r0}]
        target=tol*max(r0,1e-30)
        for i in range(maxiter):
            if r.norm().item()<=target:
                r=rhs-self.normal(a,hidden); true=r.norm().item()
                history.append({'iteration':i,'true_residual':true})
                if true<=target: break
                z=r/diag; p=z.clone(); rz=r@z
            ap=self.normal(p,hidden); den=p@ap
            if not torch.isfinite(den) or den<=0: raise RuntimeError('nonpositive CG curvature')
            step=rz/den; a=a+step*p; r=r-step*ap; iterations=i+1
            if iterations%20==0:
                r=rhs-self.normal(a,hidden); history.append({'iteration':iterations,'true_residual':r.norm().item()})
                z=r/diag; p=z.clone(); rz=r@z
            else:
                z=r/diag; new=r@z
                if not torch.isfinite(new): raise RuntimeError('nonfinite CG residual')
                if rz==0: break
                p=z+(new/rz)*p; rz=new
        true=(rhs-self.normal(a,hidden)).norm().item()
        return a,{'iterations':iterations,'initial_residual':r0,'true_normal_residual':true,'true_relative_residual':true/max(r0,1e-30),'true_residual_history':history,'restart_interval':20,'status':'CONVERGED' if true<=target else 'FINITE_ITERATE_NOT_OPTIMUM','certificate':'CERTIFICATE_UNRESOLVED'}
    def gradient(self,a):
        g=2*self.ridge*(a-self.a0)/self.za
        for s in self.sources:
            g=g+2*s['x'].T@((s['x']@a-s['y'])*s['g'].square())/(len(self.sources)*s['x'].shape[0]*self.z)
        return g
    def rank1(self,a,iterations):
        grad=self.gradient(a); generator=torch.Generator(device=grad.device); generator.manual_seed(19); v=torch.randn(grad.shape[1],device=grad.device,dtype=grad.dtype,generator=generator); v=v/v.norm()
        for _ in range(iterations):
            u=grad@v; un=u.norm()
            if un.item()==0: return a.clone(),{'eta':0.,'direction':'POWER_ITERATION','iterations':iterations},torch.zeros_like(a)
            u=u/un; v=grad.T@u; vn=v.norm()
            if vn.item()==0: return a.clone(),{'eta':0.,'direction':'POWER_ITERATION','iterations':iterations},torch.zeros_like(a)
            v=v/vn
        d=u[:,None]*v[None,:]; gd=(grad*d).sum()
        c=self.ridge*d.square().sum()/self.za
        for s in self.sources: c=c+((s['x']@d*s['g']).square().sum()/s['x'].shape[0])/(len(self.sources)*self.z)
        eta=torch.clamp(gd/(2*c),min=0); out=a-eta*d
        before=self.objective(a).item(); after=self.objective(out).item()
        if after>before+1e-6*max(1.,abs(before)): raise RuntimeError('rank1 objective increased')
        return out,{'eta':eta.item(),'curvature':c.item(),'gradient_dot_direction':gd.item(),'before':before,'after':after,'direction':'POWER_ITERATION_APPROXIMATE','iterations':iterations,'gradient_norm':grad.norm().item(),'direction_residual':(grad@v-gd*u).norm().item()},u.double()[:,None]*v.double()[None,:]

